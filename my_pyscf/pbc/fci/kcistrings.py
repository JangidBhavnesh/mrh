#!/bin/bash 

import ctypes
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

from pyscf.fci.cistring import OIndexList, make_strings

from mrh.lib.helper import load_library

libpbckcistring = load_library('libpbc_kcistring')

# Author: Bhavnesh Jangid

# TODO: Add the openMP parallelization to the link index generation in pbc_kcistring.c.
# TODO: Move the below checks to a unit test.

def gen_linkstr_index_k(orb_list, nocc, orb_k, nkpts, strs=None):
    '''
    Generate momentum (k-aware) labelled link index for FCI strings.
    link_index [str, link, 8]
        str: number of strings
        link: (nocc + nocc*nvir)
        For the last entry (8): [cre, des, target_address, parity, K0, k_cre, k_des, dK]
        cre   : created orbital index
        des   : annihilated orbital index
        target_address : address of target string
        parity         : fermionic sign
        K0             : total momentum of starting spin string
        k_cre          : momentum label of created orbital
        k_des          : momentum label of annihilated orbital
        dK             : (k_cre - k_des) mod nkpts

    args:
        orb_list : list or array
            Orbital labels used to generate strings.
        nocc : int
            Number of occupied orbitals in each string.
        orb_k : array_like, shape (norb,)
            orb_k[p] gives the k-point label of orbital p.
        nkpts : int
            Number of k-points.
        strs : array_like, optional
            Precomputed strings. If None, strings are generated from orb_list.
    returns:
        link_index : ndarray, shape (na, nlink, 8), dtype int32
    '''

    if strs is None:
        strs = make_strings(orb_list, nocc)

    if isinstance(strs, OIndexList):
        raise NotImplementedError(
            "OIndexList path is not implemented for gen_linkstr_index_k yet."
        )

    # The C code uses uint64_t strings.
    strs = np.asarray(strs, dtype=np.uint64)
    assert np.all(strs[:-1] < strs[1:])

    norb = len(orb_list)
    nvir = norb - nocc
    na = strs.shape[0]
    nlink = nocc * nvir + nocc

    # orb_k must be length norb and int32-compatible.
    orb_k = np.asarray(orb_k, dtype=np.int32)
    assert orb_k.shape == (norb,)
    assert np.all(orb_k >= 0)
    assert np.all(orb_k < nkpts)

    link_index = np.empty((na, nlink, 8), dtype=np.int32)

    libpbckcistring.FCIlinkstr_index_k(
        link_index.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(norb),
        ctypes.c_int(na),
        ctypes.c_int(nocc),
        strs.ctypes.data_as(ctypes.c_void_p),
        orb_k.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nkpts),
    )

    return link_index

def _count_det_per_k(link_index):
    '''
    Count the number of determinants in each momentum sector K0 using the link index.
    Assumes that link_index is sorted by K0 (which is true for the output of gen_linkstr_index_k).
    '''
    if isinstance(link_index, tuple):
        return tuple(_count_det_per_k(x) for x in link_index)

    assert link_index.ndim == 3
    assert link_index.shape[2] >= 5

    nstr = link_index.shape[0]
    nlink = link_index.shape[1]

    # Zero-link case, e.g. nelec = 0 for one spin sector.
    # There is still one determinant/string: the vacuum string.
    # Its momentum sector is K0 = 0.
    if nlink == 0:
        return {0: int(nstr)}

    K0_values = np.asarray(link_index[:, 0, 4], dtype=np.int32)
    unique_K0, counts = np.unique(K0_values, return_counts=True)

    return {int(kindx): int(ndet_k)
            for kindx, ndet_k in zip(unique_K0, counts)}

def gen_k_sector_linkstr_info(link_indexa, link_indexb, nkpts, kindx):
    '''
    Building the sector-specific link index info for k-FCI.
    args:
    link_indexa, link_indexb : ndarray, shape (na, nlink, 8)
        The k-aware link index for alpha and beta strings.
    nkpts : int
        Number of k-points.
    kindx : int
         Target total momentum sector, interpreted modulo nkpts.
    returns:
        blocks : ndarray, shape (nblocks, 6)
            Each row is: [ka, kb, na, nb, offset, size]
            where:
                ka      alpha-string momentum sector
                kb      beta-string momentum sector
                na      number of alpha strings in sector ka
                nb      number of beta strings in sector kb
                offset  starting offset of this block in flattened fcivec
                size    na * nb
    '''
    assert link_indexa.ndim == link_indexb.ndim == 3
    assert link_indexa.shape[2] == link_indexb.shape[2] == 8

    kindx = int(kindx) % nkpts

    count_a, count_b = _count_det_per_k((link_indexa, link_indexb))

    blocks = []
    offset = 0

    for ka in range(nkpts):
        kb = (kindx - ka) % nkpts

        na = count_a.get(ka, 0)
        nb = count_b.get(kb, 0)

        size = na * nb
        if size == 0:
            continue

        blocks.append([ka, kb, na, nb, offset, size])
        offset += size

    # Aha, this can also happen.
    if len(blocks) == 0:
        return np.zeros((0, 6), dtype=np.int32)
    
    return np.asarray(blocks, dtype=np.int32)

def _build_sector_map_spin(link_index, nkpts):
    '''
    Building sector string lists and global-to-local lookup for one spin sector.
    args:
        link_index : ndarray, shape (nstr, nlink, 8)
            k-aware link index for one spin sector.
        nkpts : int
            Number of k-points / momentum sectors.
    returns:
        str_k : list of ndarrays
            str_k[k] contains global string ids whose parent-string momentum is k.
        str_k2tot : ndarray, shape (nkpts, nstr)
            str_k2tot[k, str_global] gives the local index of str_global 
            inside sector k. It is -1 if str_global is not in sector k.
    '''
    dtype = np.int32
    assert link_index.ndim == 3
    assert link_index.shape[2] == 8

    nstr = link_index.shape[0]
    nlink = link_index.shape[1]
    # Edge case: no excitation links.
    # This happens, for example, for nelec = 0 in one spin sector.
    # There is still one valid string: the vacuum string.
    # Its total momentum is 0.
    if nlink == 0:
        str_k = [np.empty(0, dtype=dtype) for _ in range(nkpts)]
        str_k[0] = np.arange(nstr, dtype=dtype)
        str_k2tot = -np.ones((nkpts, nstr), dtype=dtype)
        str_k2tot[0, :nstr] = np.arange(nstr, dtype=dtype)
        return str_k, str_k2tot
    
    _str_k = np.asarray(link_index[:, 0, 4], dtype=dtype)

    str_k = [np.where(_str_k == k)[0].astype(dtype, copy=False)
             for k in range(nkpts)]

    str_k2tot = -np.ones((nkpts, nstr), dtype=dtype)
    
    for k, ids in enumerate(str_k):
        ids = np.asarray(ids, dtype=dtype)
        str_k2tot[k, ids] = np.arange(ids.size, dtype=dtype)
    
    return str_k, str_k2tot

def gen_k_sector_maps(link_indexa, link_indexb, nkpts):
    '''
    Build alpha/beta sector string lists and global-to-local maps.
    '''
    alpha_by_kindx, alpha_str_k2tot = _build_sector_map_spin(link_indexa, nkpts)
    beta_by_kindx, beta_str_k2tot = _build_sector_map_spin(link_indexb, nkpts)
    return alpha_by_kindx, beta_by_kindx, alpha_str_k2tot, beta_str_k2tot

@dataclass
class KLink:
    cre: int              # global creation orbital
    des: int              # global destruction orbital
    cre_l: int            # local creation orbital within k_cre
    des_l: int            # local destruction orbital within k_des
    str0_global: int      # source string address, global
    str1_global: int      # target string address, global
    str0_local: int       # source string local index inside k0 sector
    str1_local: int       # target string local index inside k1 sector
    sign: int             # +1 or -1
    k0: int               # source string momentum
    k1: int               # target string momentum
    k_cre: int            # creation momentum
    k_des: int            # destruction momentum
    dK: int               # k_cre - k_des mod nkpts


def build_k_links_spin(link_index, norb, nkpts, str_k, str_k2tot):
    '''
    Build the grouped link lists for a single spin sector, along with the local string index maps.
    The grouping is done by the source string momentum k0 and the momentum transfer dK.
    '''
    # Sanity checks
    assert link_index.ndim == 3
    assert link_index.shape[2] == 8

    nstr, nlink, _ = link_index.shape
    norb_per_k = norb // nkpts

    assert norb_per_k * nkpts == norb

    # links grouped only by starting determinant sector and momentum transfer
    by_k_dk = [[list() for _ in range(nkpts)] 
               for _ in range(nkpts)]

    # links grouped by starting determinant sector, local source string, and dK
    by_k_src_dk = [[[list() for _ in range(nkpts)] 
                    for _ in range(len(str_k[k]))] 
                    for k in range(nkpts)]

    # optional lookup by global source string and dK
    # by_global_src_dk = defaultdict(list)

    for str0_global in range(nstr):
        for j in range(nlink):
            row = link_index[str0_global, j]
            cre = int(row[0])
            des = int(row[1])
            str1_global = int(row[2])
            sign = int(row[3])
            k0 = int(row[4]) % nkpts
            k_cre = int(row[5]) % nkpts
            k_des = int(row[6]) % nkpts
            dK = int(row[7]) % nkpts

            # Sanity: excitation q -> p changes string momentum by k_p - k_q
            dK_check = (k_cre - k_des) % nkpts
            assert dK == dK_check, (f"dK mismatch at str0={str0_global}, link={j}: " 
                                    f"dK={dK}, but k_cre-k_des={dK_check}")

            # Sanity: target string momentum k1 should be (k0 + dK) % nkpts
            k1 = (k0 + dK) % nkpts

            cre_l = cre % norb_per_k
            des_l = des % norb_per_k

            str0_local = int(str_k2tot[k0, str0_global])
            str1_local = int(str_k2tot[k1, str1_global])

            assert str0_local >= 0 and str1_local >= 0, "Momentum sector mismatch"

            k1_from_table = int(link_index[str1_global, 0, 4]) % nkpts
            assert k1_from_table == k1, (
                f"Target string sector mismatch: str0={str0_global}, link={j}, "
                f"str1={str1_global}, expected k1={k1}, table has {k1_from_table}"
            )

            # Create the KLink object
            link = KLink(cre=cre, des=des, cre_l=cre_l, des_l=des_l, str0_global=str0_global, 
                         str1_global=str1_global, str0_local=str0_local, str1_local=str1_local, 
                         sign=sign, k0=k0, k1=k1, k_cre=k_cre, k_des=k_des, dK=dK)

            by_k_dk[k0][dK].append(link)
            by_k_src_dk[k0][str0_local][dK].append(link)
            # by_global_src_dk[(str0_global, dK)].append(link)

    links_info = {
        "str_k": str_k,
        "str_k2tot": str_k2tot,
        "by_k_dk": by_k_dk,
        "by_k_src_dk": by_k_src_dk,
        # "by_global_src_dk": by_global_src_dk,
        # "norb_per_k": norb_per_k,
    }

    return links_info

if __name__ == "__main__":
    from pyscf.fci import cistring

    norb = 16
    nalpha, nbeta = (8, 8)
    nkpts = 8
    assert norb % nkpts == 0
    norb_per_k = norb // nkpts
    orb_k = np.arange(norb, dtype=np.int32) // norb_per_k

    link_index_k = gen_linkstr_index_k(range(norb), nalpha, orb_k, nkpts)
    link_index_ref = cistring.gen_linkstr_index(range(norb), nalpha, tril=False)

    # Sanity check on shapes
    assert link_index_k.shape == (link_index_ref.shape[0], link_index_ref.shape[1], 8)

    # Except the momentum labelling, the first 4 columns of link_index_k should match the 
    # standard link_index_ref.
    assert np.array_equal(link_index_k[:, :, :4], link_index_ref)
   

    # Compute the total momentum K0 for each string using the orbital k-point labels, 
    # and compare with the K0 stored in link_index_k.
    strs = cistring.make_strings(range(norb), nalpha)
    K_str = np.array([sum(orb_k[i] for i in range(norb) 
                          if (int(s) >> i) & 1) % nkpts for s in strs], dtype=np.int32)
    K0_stored = link_index_k[:, :, 4]

    assert np.all(K0_stored == K0_stored[:, [0]])
    assert np.array_equal(K0_stored[:, 0], K_str)

    link_indexa = gen_linkstr_index_k(range(norb), nalpha, orb_k, nkpts)
    link_indexb = gen_linkstr_index_k(range(norb), nbeta, orb_k, nkpts)
    blocks = gen_k_sector_linkstr_info(link_indexa, link_indexb, nkpts=nkpts, kindx=0)

    print(blocks)
    # Note blocks are stored as [ka, kb, na, nb, offset, size]
    print("Number of determinants in kindx 0 =", blocks[:, 5].sum())
    print("Ratio of ndet in kindx 0 / total det =", blocks[:, 5].sum() / (link_indexa.shape[0] * link_indexb.shape[0]))