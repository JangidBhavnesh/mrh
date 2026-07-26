#!/bin/bash 

import ctypes
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

from pyscf import lib
from pyscf.fci.cistring import OIndexList, make_strings
from pyscf.fci.addons import _unpack_nelec

from mrh.lib.helper import load_library

libpbckcistring = load_library('libpbc_kcistring')
_contract_structure_builder_configured = False

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


def _build_k_links_spin(link_index, norb, nkpts, str_k, str_k2tot):
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


L_CRE_L       = 0
L_DES_L       = 1
L_STR0_LOCAL  = 2
L_STR1_LOCAL  = 3
L_STR0_GLOBAL = 4
L_STR1_GLOBAL = 5
L_SIGN        = 6
L_K0          = 7
L_K1          = 8
L_K_CRE       = 9
L_K_DES       = 10
L_DK          = 11

NLINK_FIELDS = 12

def build_k_links_spin(link_index, norb, nkpts, str_k, str_k2tot):
    '''
    Build the compact link table for a single spin sector, along with the local string index maps.
    The grouping is done by the source string momentum k0 and the momentum transfer dK.
    '''
    # Sanity checks
    assert link_index.ndim == 3
    assert link_index.shape[2] == 8

    nstr, nlink, _ = link_index.shape
    norb_per_k = norb // nkpts

    assert norb_per_k * nkpts == norb

    rows = []

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

            # If the target string has no links, e.g. zero-electron sector,
            # link_index[str1_global, 0, 4] may be invalid. Only check when nlink > 0.
            if nlink > 0:
                k1_from_table = int(link_index[str1_global, 0, 4]) % nkpts
                assert k1_from_table == k1, (
                    f"Target string sector mismatch: str0={str0_global}, link={j}, "
                    f"str1={str1_global}, expected k1={k1}, table has {k1_from_table}"
                )

            rows.append([
                cre_l,
                des_l,
                str0_local,
                str1_local,
                str0_global,
                str1_global,
                sign,
                k0,
                k1,
                k_cre,
                k_des,
                dK,
            ])

    if len(rows) == 0:
        linktab = np.zeros((0, NLINK_FIELDS), dtype=np.int32)
    else:
        linktab = np.asarray(rows, dtype=np.int32)

    # Sort links by source sector k0 and momentum transfer dK.
    if linktab.shape[0] > 0:
        order = np.lexsort((linktab[:, L_DK], linktab[:, L_K0]))
        linktab = np.asarray(linktab[order], dtype=np.int32, order="C")

    # offset_k_dk[k, dK] : start index of links with source sector k and momentum transfer dK.
    # offset_k_dk[k, dK + 1] : end index.
    offset_k_dk = np.zeros((nkpts, nkpts + 1), dtype=np.int32)

    pos = 0
    for k in range(nkpts):
        offset_k_dk[k, 0] = pos

        for dK in range(nkpts):
            while (
                pos < linktab.shape[0]
                and linktab[pos, L_K0] == k
                and linktab[pos, L_DK] == dK
            ):
                pos += 1

            offset_k_dk[k, dK + 1] = pos

    links_info = {
        "str_k": str_k,
        "str_k2tot": str_k2tot,
        "linktab": linktab,
        "offset_k_dk": offset_k_dk,
    }

    return links_info


def _flatten_sector_ids(str_ids_by_k, nkpts):
    ids = []
    offsets = [0]
    for k in range(nkpts):
        tab = np.asarray(str_ids_by_k[k], dtype=np.int32, order="C")
        ids.append(tab)
        offsets.append(offsets[-1] + tab.size)

    if ids:
        ids = np.asarray(np.concatenate(ids), dtype=np.int32, order="C")
    else:
        ids = np.zeros(0, dtype=np.int32)

    return ids, np.asarray(offsets, dtype=np.int32, order="C")


def _unpack_contract_link_index(norb, nelec, link_index, nkpts, spin=None):
    assert norb % nkpts == 0
    if link_index is None:
        neleca, nelecb = _unpack_nelec(nelec, spin)
        norb_k = norb // nkpts
        orb_k = (np.arange(norb, dtype=np.int32) // norb_k).astype(np.int32)
        link_indexa = gen_linkstr_index_k(range(norb), neleca, orb_k, nkpts)
        if spin == 0 and neleca == nelecb:
            link_indexb = link_indexa
        else:
            link_indexb = gen_linkstr_index_k(range(norb), nelecb,
                                              orb_k, nkpts)
        return link_indexa, link_indexb

    assert link_index[0].shape[2] == link_index[1].shape[2] == 8
    return link_index


def get_links_by_k(links, k):
    linktab = links["linktab"]
    offset = links["offset_k_dk"]

    return linktab[offset[k, 0]:offset[k, -1]]


def get_links_by_k_dk(links, k, dK):
    linktab = links["linktab"]
    offset = links["offset_k_dk"]

    return linktab[offset[k, dK]:offset[k, dK + 1]]


def build_links_by_global_source_array(links):
    linktab = links["linktab"]
    nlinks = linktab.shape[0]

    if nlinks == 0:
        links["global_source_order"] = np.zeros(0, dtype=np.int32)
        links["global_source_ids"] = np.zeros(0, dtype=np.int32)
        links["global_source_offsets"] = np.zeros(1, dtype=np.int32)
        return links

    src = linktab[:, L_STR0_GLOBAL]
    order = np.argsort(src, kind="stable").astype(np.int32)
    src_sorted = src[order]
    unique_src, first = np.unique(src_sorted, return_index=True)

    offsets = np.empty(unique_src.size + 1, dtype=np.int32)
    offsets[:-1] = first.astype(np.int32)
    offsets[-1] = nlinks

    links["global_source_order"] = order
    links["global_source_ids"] = unique_src.astype(np.int32)
    links["global_source_offsets"] = offsets

    return links


def get_link_indices_from_global_source(links, src_global):
    ids = links["global_source_ids"]
    offsets = links["global_source_offsets"]
    order = links["global_source_order"]

    pos = np.searchsorted(ids, src_global)
    if pos >= ids.size or ids[pos] != src_global:
        return order[0:0]

    return order[offsets[pos]:offsets[pos + 1]]


AB_A0      = 0
AB_A1      = 1
AB_B0      = 2
AB_B1      = 3
AB_SIGN    = 4
AB_KA1     = 5
AB_KB1     = 6
AB_KPA     = 7
AB_KQA     = 8
AB_KRB     = 9
AB_PA      = 10
AB_QA      = 11
AB_RB      = 12
AB_SB      = 13
AB_KPB     = 14
AB_KQB     = 15
AB_KRA     = 16
AB_PB      = 17
AB_QB      = 18
AB_RA      = 19
AB_SA      = 20
NAB_FIELDS = 21

SS_0      = 0
SS_1      = 1
SS_SIGN   = 2
SS_K1     = 3
SS_KP     = 4
SS_KQ     = 5
SS_KR     = 6
SS_P      = 7
SS_Q      = 8
SS_R      = 9
SS_S      = 10
NSS_FIELDS = 11


def build_ab_pair_tables(links_a, links_b, nkpts):
    ab_pairs = [[None for _ in range(nkpts)] for _ in range(nkpts)]

    for ka in range(nkpts):
        la_tab = get_links_by_k(links_a, ka)

        for kb in range(nkpts):
            rows = []

            for la in la_tab:
                dKa = int(la[L_DK])
                ka1 = int(la[L_K1])
                dKb_needed = (-dKa) % nkpts

                lb_tab = get_links_by_k_dk(links_b, kb, dKb_needed)

                for lb in lb_tab:
                    rows.append([
                        int(la[L_STR0_LOCAL]),
                        int(la[L_STR1_LOCAL]),
                        int(lb[L_STR0_LOCAL]),
                        int(lb[L_STR1_LOCAL]),
                        int(la[L_SIGN]) * int(lb[L_SIGN]),
                        ka1,
                        int(lb[L_K1]),
                        int(la[L_K_CRE]),
                        int(la[L_K_DES]),
                        int(lb[L_K_CRE]),
                        int(la[L_CRE_L]),
                        int(la[L_DES_L]),
                        int(lb[L_CRE_L]),
                        int(lb[L_DES_L]),
                        int(lb[L_K_CRE]),
                        int(lb[L_K_DES]),
                        int(la[L_K_CRE]),
                        int(lb[L_CRE_L]),
                        int(lb[L_DES_L]),
                        int(la[L_CRE_L]),
                        int(la[L_DES_L]),
                    ])

            if len(rows) == 0:
                ab_pairs[ka][kb] = np.zeros((0, NAB_FIELDS), dtype=np.int32)
            else:
                ab_pairs[ka][kb] = np.asarray(rows, dtype=np.int32)

    return ab_pairs


def build_same_spin_pair_tables(links, nkpts):
    linktab = links["linktab"]
    ss_pairs = [None for _ in range(nkpts)]

    for k in range(nkpts):
        rows = []
        l1_tab = get_links_by_k(links, k)

        for l1 in l1_tab:
            dK1 = int(l1[L_DK])
            src_mid = int(l1[L_STR1_GLOBAL])
            l2_indices = get_link_indices_from_global_source(links, src_mid)

            for idx2 in l2_indices:
                l2 = linktab[idx2]
                if int(l2[L_K0]) != int(l1[L_K1]):
                    continue

                dK2 = int(l2[L_DK])
                if (dK1 + dK2) % nkpts != 0:
                    continue

                rows.append([
                    int(l1[L_STR0_LOCAL]),
                    int(l2[L_STR1_LOCAL]),
                    int(l1[L_SIGN]) * int(l2[L_SIGN]),
                    int(l2[L_K1]),
                    int(l2[L_K_CRE]),
                    int(l2[L_K_DES]),
                    int(l1[L_K_CRE]),
                    int(l2[L_CRE_L]),
                    int(l2[L_DES_L]),
                    int(l1[L_CRE_L]),
                    int(l1[L_DES_L]),
                ])

        if len(rows) == 0:
            ss_pairs[k] = np.zeros((0, NSS_FIELDS), dtype=np.int32)
        else:
            ss_pairs[k] = np.asarray(rows, dtype=np.int32)

    return ss_pairs


def flatten_pair_tables(ab_pairs, aa_pairs, bb_pairs, nkpts):
    ab_rows = []
    ab_offsets = [0]
    for ka in range(nkpts):
        for kb in range(nkpts):
            tab = np.asarray(ab_pairs[ka][kb], dtype=np.int32, order="C")
            tab = tab.reshape(-1, NAB_FIELDS)
            if tab.size:
                ab_rows.append(tab)
            ab_offsets.append(ab_offsets[-1] + tab.shape[0])

    aa_rows = []
    aa_offsets = [0]
    for k in range(nkpts):
        tab = np.asarray(aa_pairs[k], dtype=np.int32, order="C")
        tab = tab.reshape(-1, NSS_FIELDS)
        if tab.size:
            aa_rows.append(tab)
        aa_offsets.append(aa_offsets[-1] + tab.shape[0])

    bb_rows = []
    bb_offsets = [0]
    for k in range(nkpts):
        tab = np.asarray(bb_pairs[k], dtype=np.int32, order="C")
        tab = tab.reshape(-1, NSS_FIELDS)
        if tab.size:
            bb_rows.append(tab)
        bb_offsets.append(bb_offsets[-1] + tab.shape[0])

    if ab_rows:
        ab_tab = np.asarray(np.vstack(ab_rows), dtype=np.int32, order="C")
    else:
        ab_tab = np.zeros((0, NAB_FIELDS), dtype=np.int32)

    if aa_rows:
        aa_tab = np.asarray(np.vstack(aa_rows), dtype=np.int32, order="C")
    else:
        aa_tab = np.zeros((0, NSS_FIELDS), dtype=np.int32)

    if bb_rows:
        bb_tab = np.asarray(np.vstack(bb_rows), dtype=np.int32, order="C")
    else:
        bb_tab = np.zeros((0, NSS_FIELDS), dtype=np.int32)

    return (ab_tab, np.asarray(ab_offsets, dtype=np.int32, order="C"),
            aa_tab, np.asarray(aa_offsets, dtype=np.int32, order="C"),
            bb_tab, np.asarray(bb_offsets, dtype=np.int32, order="C"))


def _eri_index(kp, kq, kr, p, q, r, s, nkpts, ncas):
    idx = int(kp)
    idx = idx * nkpts + int(kq)
    idx = idx * nkpts + int(kr)
    idx = idx * ncas + int(p)
    idx = idx * ncas + int(q)
    idx = idx * ncas + int(r)
    idx = idx * ncas + int(s)
    return idx


def build_ab_sparse_structure(ab_tab, ab_offsets, blocks, nkpts, ncas):
    table_size = nkpts * nkpts
    block_offset = -np.ones(table_size, dtype=np.int32)
    block_nb = np.zeros(table_size, dtype=np.int32)

    for blk in np.asarray(blocks, dtype=np.int32).reshape(-1, 6):
        key = int(blk[0]) * nkpts + int(blk[1])
        block_offset[key] = int(blk[4])
        block_nb[key] = int(blk[3])

    groups = []
    group_offsets = [0]
    src_addrs = []
    dst_addrs = []
    signs = []
    eri_idx_ab = []
    eri_idx_ba = []

    for src_key in range(table_size):
        if block_offset[src_key] < 0:
            group_offsets.append(len(groups))
            continue

        src_nb = int(block_nb[src_key])
        by_dst = {}

        for i in range(int(ab_offsets[src_key]), int(ab_offsets[src_key + 1])):
            row = ab_tab[i]
            dst_key = int(row[AB_KA1]) * nkpts + int(row[AB_KB1])
            if block_offset[dst_key] < 0:
                continue
            by_dst.setdefault(dst_key, []).append(row)

        for dst_key in sorted(by_dst):
            entry0 = len(src_addrs)
            dst_nb = int(block_nb[dst_key])

            for row in by_dst[dst_key]:
                src_addrs.append(int(row[AB_A0]) * src_nb + int(row[AB_B0]))
                dst_addrs.append(int(row[AB_A1]) * dst_nb + int(row[AB_B1]))
                signs.append(int(row[AB_SIGN]))
                eri_idx_ab.append(_eri_index(
                    row[AB_KPA], row[AB_KQA], row[AB_KRB],
                    row[AB_PA], row[AB_QA], row[AB_RB], row[AB_SB],
                    nkpts, ncas))
                eri_idx_ba.append(_eri_index(
                    row[AB_KPB], row[AB_KQB], row[AB_KRA],
                    row[AB_PB], row[AB_QB], row[AB_RA], row[AB_SA],
                    nkpts, ncas))

            groups.append([int(block_offset[dst_key]), entry0,
                           len(src_addrs)])

        group_offsets.append(len(groups))

    if groups:
        group_tab = np.asarray(groups, dtype=np.int32, order="C")
    else:
        group_tab = np.zeros((0, 3), dtype=np.int32)

    return {
        "ab_group_tab": group_tab,
        "ab_group_offsets": np.asarray(group_offsets, dtype=np.int32,
                                       order="C"),
        "ab_src_addr": np.asarray(src_addrs, dtype=np.int32, order="C"),
        "ab_dst_addr": np.asarray(dst_addrs, dtype=np.int32, order="C"),
        "ab_sign": np.asarray(signs, dtype=np.int32, order="C"),
        "ab_eri_idx_ab": np.asarray(eri_idx_ab, dtype=np.int64, order="C"),
        "ab_eri_idx_ba": np.asarray(eri_idx_ba, dtype=np.int64, order="C"),
    }


def build_same_spin_dense_structure(ss_tab, ss_offsets, blocks, nkpts, ncas,
                                    spin):
    block_offset = -np.ones(nkpts * nkpts, dtype=np.int32)
    block_na = np.zeros(nkpts * nkpts, dtype=np.int32)
    block_nb = np.zeros(nkpts * nkpts, dtype=np.int32)

    for ka, kb, na, nb, offset, _ in blocks:
        key = int(ka) * nkpts + int(kb)
        block_offset[key] = int(offset)
        block_na[key] = int(na)
        block_nb[key] = int(nb)

    groups = []
    group_offsets = [0]
    src_addrs = []
    dst_addrs = []
    signs = []
    eri_idx = []

    for src_key in range(nkpts * nkpts):
        ka = src_key // nkpts
        kb = src_key % nkpts
        src_offset = int(block_offset[src_key])

        if src_offset < 0:
            group_offsets.append(len(groups))
            continue

        na = int(block_na[src_key])
        nb = int(block_nb[src_key])
        src_k = ka if spin == "a" else kb
        ss0 = int(ss_offsets[src_k])
        ss1 = int(ss_offsets[src_k + 1])

        for dst_k in range(nkpts):
            if spin == "a":
                dst_key = dst_k * nkpts + kb
                dst_dim = int(block_na[dst_key])
                good_dst = (int(block_offset[dst_key]) >= 0 and
                            dst_dim > 0 and
                            int(block_nb[dst_key]) == nb)
            else:
                dst_key = ka * nkpts + dst_k
                dst_dim = int(block_nb[dst_key])
                good_dst = (int(block_offset[dst_key]) >= 0 and
                            dst_dim > 0)

            if not good_dst:
                continue

            entry0 = len(src_addrs)
            for i in range(ss0, ss1):
                row = ss_tab[i]
                if int(row[SS_K1]) != dst_k:
                    continue

                src_addrs.append(int(row[SS_0]))
                dst_addrs.append(int(row[SS_1]))
                signs.append(int(row[SS_SIGN]))
                eri_idx.append(_eri_index(
                    row[SS_KP], row[SS_KQ], row[SS_KR],
                    row[SS_P], row[SS_Q], row[SS_R], row[SS_S],
                    nkpts, ncas))

            if len(src_addrs) > entry0:
                groups.append([int(block_offset[dst_key]), dst_dim, entry0,
                               len(src_addrs)])

        group_offsets.append(len(groups))

    if groups:
        group_tab = np.asarray(groups, dtype=np.int32, order="C")
    else:
        group_tab = np.zeros((0, 4), dtype=np.int32)

    return {
        "group_tab": group_tab,
        "group_offsets": np.asarray(group_offsets, dtype=np.int32,
                                    order="C"),
        "src_addr": np.asarray(src_addrs, dtype=np.int32, order="C"),
        "dst_addr": np.asarray(dst_addrs, dtype=np.int32, order="C"),
        "sign": np.asarray(signs, dtype=np.int32, order="C"),
        "eri_idx": np.asarray(eri_idx, dtype=np.int64, order="C"),
    }


def _configure_contract_structure_builder():
    global _contract_structure_builder_configured
    if _contract_structure_builder_configured:
        return

    libpbckcistring.FCIcount_contract_k_structures.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
    ]
    libpbckcistring.FCIcount_contract_k_structures.restype = ctypes.c_int

    libpbckcistring.FCIfill_contract_k_structures.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    libpbckcistring.FCIfill_contract_k_structures.restype = ctypes.c_int
    _contract_structure_builder_configured = True


def build_contract_structures_c(link_indexa, link_indexb, str2tot_a,
                                str2tot_b, blocks, nkpts, ncas):
    _configure_contract_structure_builder()

    link_indexa = np.asarray(link_indexa, dtype=np.int32, order="C")
    link_indexb = np.asarray(link_indexb, dtype=np.int32, order="C")
    str2tot_a = np.asarray(str2tot_a, dtype=np.int32, order="C")
    str2tot_b = np.asarray(str2tot_b, dtype=np.int32, order="C")
    blocks = np.asarray(blocks, dtype=np.int32, order="C").reshape(-1, 6)

    nstra, nlinka, _ = link_indexa.shape
    nstrb, nlinkb, _ = link_indexb.shape
    dims = np.zeros(6, dtype=np.int64)

    with lib.with_omp_threads(lib.num_threads()):
        status = libpbckcistring.FCIcount_contract_k_structures(
            link_indexa.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nstra),
            ctypes.c_int(nlinka),
            link_indexb.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nstrb),
            ctypes.c_int(nlinkb),
            blocks.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(blocks.shape[0]),
            ctypes.c_int(nkpts),
            dims.ctypes.data_as(ctypes.c_void_p),
        )
    if status != 0:
        raise RuntimeError("FCIcount_contract_k_structures failed")

    nab_groups, nab_entries = int(dims[0]), int(dims[1])
    naa_groups, naa_entries = int(dims[2]), int(dims[3])
    nbb_groups, nbb_entries = int(dims[4]), int(dims[5])
    table_size = nkpts * nkpts
    _raise_if_contract_structure_too_large(
        nab_entries, naa_entries, nbb_entries)

    arrays = {
        "ab_group_tab": np.empty((nab_groups, 3), dtype=np.int32,
                                 order="C"),
        "ab_group_offsets": np.empty(table_size + 1, dtype=np.int32,
                                     order="C"),
        "ab_src_addr": np.empty(nab_entries, dtype=np.int32, order="C"),
        "ab_dst_addr": np.empty(nab_entries, dtype=np.int32, order="C"),
        "ab_sign": np.empty(nab_entries, dtype=np.int32, order="C"),
        "ab_eri_idx_ab": np.empty(nab_entries, dtype=np.int64, order="C"),
        "ab_eri_idx_ba": np.empty(nab_entries, dtype=np.int64, order="C"),
        "aa_group_tab": np.empty((naa_groups, 4), dtype=np.int32,
                                 order="C"),
        "aa_group_offsets": np.empty(table_size + 1, dtype=np.int32,
                                     order="C"),
        "aa_src_addr": np.empty(naa_entries, dtype=np.int32, order="C"),
        "aa_dst_addr": np.empty(naa_entries, dtype=np.int32, order="C"),
        "aa_sign": np.empty(naa_entries, dtype=np.int32, order="C"),
        "aa_eri_idx": np.empty(naa_entries, dtype=np.int64, order="C"),
        "bb_group_tab": np.empty((nbb_groups, 4), dtype=np.int32,
                                 order="C"),
        "bb_group_offsets": np.empty(table_size + 1, dtype=np.int32,
                                     order="C"),
        "bb_src_addr": np.empty(nbb_entries, dtype=np.int32, order="C"),
        "bb_dst_addr": np.empty(nbb_entries, dtype=np.int32, order="C"),
        "bb_sign": np.empty(nbb_entries, dtype=np.int32, order="C"),
        "bb_eri_idx": np.empty(nbb_entries, dtype=np.int64, order="C"),
    }

    with lib.with_omp_threads(lib.num_threads()):
        status = libpbckcistring.FCIfill_contract_k_structures(
            link_indexa.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nstra),
            ctypes.c_int(nlinka),
            link_indexb.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nstrb),
            ctypes.c_int(nlinkb),
            str2tot_a.ctypes.data_as(ctypes.c_void_p),
            str2tot_b.ctypes.data_as(ctypes.c_void_p),
            blocks.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(blocks.shape[0]),
            ctypes.c_int(nkpts),
            ctypes.c_int(ncas),
            arrays["ab_group_tab"].ctypes.data_as(ctypes.c_void_p),
            arrays["ab_group_offsets"].ctypes.data_as(ctypes.c_void_p),
            arrays["ab_src_addr"].ctypes.data_as(ctypes.c_void_p),
            arrays["ab_dst_addr"].ctypes.data_as(ctypes.c_void_p),
            arrays["ab_sign"].ctypes.data_as(ctypes.c_void_p),
            arrays["ab_eri_idx_ab"].ctypes.data_as(ctypes.c_void_p),
            arrays["ab_eri_idx_ba"].ctypes.data_as(ctypes.c_void_p),
            arrays["aa_group_tab"].ctypes.data_as(ctypes.c_void_p),
            arrays["aa_group_offsets"].ctypes.data_as(ctypes.c_void_p),
            arrays["aa_src_addr"].ctypes.data_as(ctypes.c_void_p),
            arrays["aa_dst_addr"].ctypes.data_as(ctypes.c_void_p),
            arrays["aa_sign"].ctypes.data_as(ctypes.c_void_p),
            arrays["aa_eri_idx"].ctypes.data_as(ctypes.c_void_p),
            arrays["bb_group_tab"].ctypes.data_as(ctypes.c_void_p),
            arrays["bb_group_offsets"].ctypes.data_as(ctypes.c_void_p),
            arrays["bb_src_addr"].ctypes.data_as(ctypes.c_void_p),
            arrays["bb_dst_addr"].ctypes.data_as(ctypes.c_void_p),
            arrays["bb_sign"].ctypes.data_as(ctypes.c_void_p),
            arrays["bb_eri_idx"].ctypes.data_as(ctypes.c_void_p),
        )
    if status != 0:
        raise RuntimeError("FCIfill_contract_k_structures failed")

    return arrays


def _raise_if_contract_structure_too_large(nab_entries, naa_entries,
                                           nbb_entries):
    max_int32 = np.iinfo(np.int32).max
    if max(nab_entries, naa_entries, nbb_entries) <= max_int32:
        return

    # The explicit structural map stores entry offsets/cursors as int32.
    # Building it beyond this limit would overflow in C and can also imply
    # hundreds of GB of sparse-map arrays.
    entry_bytes = (
        4 * (nab_entries + naa_entries + nbb_entries) * 3
        + 8 * (2 * nab_entries + naa_entries + nbb_entries)
    )
    raise MemoryError(
        "k-FCI explicit contract map is too large for the current "
        "int32 sparse-entry representation: "
        f"ab_entries={nab_entries}, aa_entries={naa_entries}, "
        f"bb_entries={nbb_entries}, estimated_entry_storage="
        f"{entry_bytes / 1024**3:.2f} GiB"
    )


def build_contract_pair_tables(link_indexa, link_indexb, norb, nkpts):
    straid_k, strbid_k, str2tot_a, str2tot_b = gen_k_sector_maps(
        link_indexa, link_indexb, nkpts)

    links_a = build_k_links_spin(link_indexa, norb, nkpts,
                                 straid_k, str2tot_a)
    links_b = build_k_links_spin(link_indexb, norb, nkpts,
                                 strbid_k, str2tot_b)

    links_a = build_links_by_global_source_array(links_a)
    links_b = build_links_by_global_source_array(links_b)

    ab_pairs = build_ab_pair_tables(links_a, links_b, nkpts)
    aa_pairs = build_same_spin_pair_tables(links_a, nkpts)
    bb_pairs = build_same_spin_pair_tables(links_b, nkpts)

    return flatten_pair_tables(ab_pairs, aa_pairs, bb_pairs, nkpts)


@dataclass
class KFCIContractMap:
    norb: int
    nelec: tuple
    nkpts: int
    target_k: int
    ncas: int
    sector_size: int
    link_index: tuple
    blocks: np.ndarray
    stra_ids: np.ndarray
    stra_offsets: np.ndarray
    strb_ids: np.ndarray
    strb_offsets: np.ndarray
    str2tot_a: np.ndarray
    str2tot_b: np.ndarray
    ab_tab: np.ndarray
    ab_offsets: np.ndarray
    aa_tab: np.ndarray
    aa_offsets: np.ndarray
    bb_tab: np.ndarray
    bb_offsets: np.ndarray
    has_pair_tables: bool
    ab_group_tab: np.ndarray
    ab_group_offsets: np.ndarray
    ab_src_addr: np.ndarray
    ab_dst_addr: np.ndarray
    ab_sign: np.ndarray
    ab_eri_idx_ab: np.ndarray
    ab_eri_idx_ba: np.ndarray
    aa_group_tab: np.ndarray
    aa_group_offsets: np.ndarray
    aa_src_addr: np.ndarray
    aa_dst_addr: np.ndarray
    aa_sign: np.ndarray
    aa_eri_idx: np.ndarray
    bb_group_tab: np.ndarray
    bb_group_offsets: np.ndarray
    bb_src_addr: np.ndarray
    bb_dst_addr: np.ndarray
    bb_sign: np.ndarray
    bb_eri_idx: np.ndarray

    @classmethod
    def build(cls, norb, nelec, nkpts, target_k, link_index=None,
              build_pair_tables=False, use_c_structures=True):
        nkpts = int(nkpts)
        norb = int(norb)
        ncas = norb // nkpts
        assert ncas * nkpts == norb

        nelec = _unpack_nelec(nelec)
        link_indexa, link_indexb = _unpack_contract_link_index(
            norb, nelec, link_index, nkpts)
        link_indexa = np.asarray(link_indexa, dtype=np.int32, order="C")
        link_indexb = np.asarray(link_indexb, dtype=np.int32, order="C")

        blocks = gen_k_sector_linkstr_info(
            link_indexa, link_indexb, nkpts, target_k)
        blocks = np.asarray(blocks, dtype=np.int32, order="C")
        sector_size = int(blocks[:, 5].sum()) if blocks.size else 0

        straid_k, strbid_k, str2tot_a, str2tot_b = gen_k_sector_maps(
            link_indexa, link_indexb, nkpts)
        stra_ids, stra_offsets = _flatten_sector_ids(straid_k, nkpts)
        strb_ids, strb_offsets = _flatten_sector_ids(strbid_k, nkpts)
        structures = None
        if use_c_structures:
            try:
                structures = build_contract_structures_c(
                    link_indexa, link_indexb, str2tot_a, str2tot_b,
                    blocks, nkpts, ncas)
            except AttributeError:
                structures = None

        if build_pair_tables or structures is None:
            ab_tab, ab_offsets, aa_tab, aa_offsets, bb_tab, bb_offsets = (
                build_contract_pair_tables(link_indexa, link_indexb,
                                           norb, nkpts))
            has_pair_tables = True
            if structures is None:
                ab_sparse = build_ab_sparse_structure(
                    ab_tab, ab_offsets, blocks, nkpts, ncas)
                aa_dense = build_same_spin_dense_structure(
                    aa_tab, aa_offsets, blocks, nkpts, ncas, "a")
                bb_dense = build_same_spin_dense_structure(
                    bb_tab, bb_offsets, blocks, nkpts, ncas, "b")
                structures = {
                    "ab_group_tab": ab_sparse["ab_group_tab"],
                    "ab_group_offsets": ab_sparse["ab_group_offsets"],
                    "ab_src_addr": ab_sparse["ab_src_addr"],
                    "ab_dst_addr": ab_sparse["ab_dst_addr"],
                    "ab_sign": ab_sparse["ab_sign"],
                    "ab_eri_idx_ab": ab_sparse["ab_eri_idx_ab"],
                    "ab_eri_idx_ba": ab_sparse["ab_eri_idx_ba"],
                    "aa_group_tab": aa_dense["group_tab"],
                    "aa_group_offsets": aa_dense["group_offsets"],
                    "aa_src_addr": aa_dense["src_addr"],
                    "aa_dst_addr": aa_dense["dst_addr"],
                    "aa_sign": aa_dense["sign"],
                    "aa_eri_idx": aa_dense["eri_idx"],
                    "bb_group_tab": bb_dense["group_tab"],
                    "bb_group_offsets": bb_dense["group_offsets"],
                    "bb_src_addr": bb_dense["src_addr"],
                    "bb_dst_addr": bb_dense["dst_addr"],
                    "bb_sign": bb_dense["sign"],
                    "bb_eri_idx": bb_dense["eri_idx"],
                }
        else:
            table_size = nkpts * nkpts
            ab_tab = np.zeros((0, NAB_FIELDS), dtype=np.int32)
            ab_offsets = np.zeros(table_size + 1, dtype=np.int32)
            aa_tab = np.zeros((0, NSS_FIELDS), dtype=np.int32)
            aa_offsets = np.zeros(nkpts + 1, dtype=np.int32)
            bb_tab = np.zeros((0, NSS_FIELDS), dtype=np.int32)
            bb_offsets = np.zeros(nkpts + 1, dtype=np.int32)
            has_pair_tables = False

        return cls(
            norb=norb,
            nelec=tuple(nelec),
            nkpts=nkpts,
            target_k=int(target_k) % nkpts,
            ncas=ncas,
            sector_size=sector_size,
            link_index=(link_indexa, link_indexb),
            blocks=blocks,
            stra_ids=stra_ids,
            stra_offsets=stra_offsets,
            strb_ids=strb_ids,
            strb_offsets=strb_offsets,
            str2tot_a=np.asarray(str2tot_a, dtype=np.int32, order="C"),
            str2tot_b=np.asarray(str2tot_b, dtype=np.int32, order="C"),
            ab_tab=ab_tab,
            ab_offsets=ab_offsets,
            aa_tab=aa_tab,
            aa_offsets=aa_offsets,
            bb_tab=bb_tab,
            bb_offsets=bb_offsets,
            has_pair_tables=has_pair_tables,
            ab_group_tab=structures["ab_group_tab"],
            ab_group_offsets=structures["ab_group_offsets"],
            ab_src_addr=structures["ab_src_addr"],
            ab_dst_addr=structures["ab_dst_addr"],
            ab_sign=structures["ab_sign"],
            ab_eri_idx_ab=structures["ab_eri_idx_ab"],
            ab_eri_idx_ba=structures["ab_eri_idx_ba"],
            aa_group_tab=structures["aa_group_tab"],
            aa_group_offsets=structures["aa_group_offsets"],
            aa_src_addr=structures["aa_src_addr"],
            aa_dst_addr=structures["aa_dst_addr"],
            aa_sign=structures["aa_sign"],
            aa_eri_idx=structures["aa_eri_idx"],
            bb_group_tab=structures["bb_group_tab"],
            bb_group_offsets=structures["bb_group_offsets"],
            bb_src_addr=structures["bb_src_addr"],
            bb_dst_addr=structures["bb_dst_addr"],
            bb_sign=structures["bb_sign"],
            bb_eri_idx=structures["bb_eri_idx"],
        )


def make_kfci_contract_map(norb, nelec, nkpts, target_k, link_index=None,
                           build_pair_tables=False, use_c_structures=True):
    return KFCIContractMap.build(norb, nelec, nkpts, target_k,
                                 link_index=link_index,
                                 build_pair_tables=build_pair_tables,
                                 use_c_structures=use_c_structures)

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
