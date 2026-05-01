#!/bin/bash 

import ctypes
import numpy as np
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