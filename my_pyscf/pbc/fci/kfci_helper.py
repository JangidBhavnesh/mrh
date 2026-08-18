#!/bin/bash

import ctypes
import os
import numpy as np
from dataclasses import dataclass

from pyscf import lib
from pyscf.fci.addons import _unpack_nelec

from mrh.lib.helper import load_library
from mrh.my_pyscf.pbc.fci.kcistrings import (
    KPointMomentum,
    _as_kmom,
    _kadd,
    _ksub,
    gen_k_sector_linkstr_info,
    gen_k_sector_maps,
    gen_linkstr_index_k,
)

# Author: Bhavnesh Jangid

'''
Contraction-map helpers for k-FCI and k-FCI RDM operations.
'''

libpbckcistring = load_library('libpbc_kcistring')
_contract_structure_builder_configured = False
_same_spin_structure_builder_configured = False


# Constants for link table fields
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

def build_k_links_spin(link_index, norb, nkpts, str_k, 
                       str_k2tot,
                       kmom=None, kconserv=None):
    '''
    Build the compact link table for a single spin sector, along 
    with the local string index maps.The grouping is done by the
    source string momentum k0 and the momentum transfer dK.
    '''
    # Sanity checks
    kmom = _as_kmom(nkpts, kmom=kmom, kconserv=kconserv)
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
            dK_check = _ksub(kmom, k_cre, k_des)
            assert dK == dK_check, (f"dK mismatch at str0={str0_global}, link={j}: "
                                    f"dK={dK}, but k_cre-k_des={dK_check}")

            # Sanity: target string momentum k1 should be (k0 + dK) % nkpts
            k1 = _kadd(kmom, k0, dK)

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
    dtype = np.int32
    ids = []
    offsets = [0]
    for k in range(nkpts):
        tab = np.asarray(str_ids_by_k[k], dtype=dtype, order="C")
        ids.append(tab)
        offsets.append(offsets[-1] + tab.size)

    if ids:
        ids = np.asarray(
            np.concatenate(ids), dtype=dtype, order="C")
    else:
        ids = np.zeros(0, dtype=dtype)

    return ids, np.asarray(offsets, dtype=dtype, order="C")


def _unpack_contract_link_index(norb, nelec, link_index, nkpts, spin=None,
                                kmom=None, kconserv=None):
    dtype = np.int32
    assert norb % nkpts == 0
    kmom = _as_kmom(nkpts, kmom=kmom, kconserv=kconserv)
    if link_index is None:
        neleca, nelecb = _unpack_nelec(nelec, spin)
        norb_k = norb // nkpts
        orb_k = (np.arange(norb, dtype=dtype) // norb_k).astype(dtype)
        link_indexa = gen_linkstr_index_k(range(norb), neleca, orb_k,
                                          nkpts, kmom=kmom)
        if spin == 0 and neleca == nelecb:
            link_indexb = link_indexa
        else:
            link_indexb = gen_linkstr_index_k(range(norb), nelecb,
                                              orb_k, nkpts, kmom=kmom)
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
    dtype = np.int32

    if nlinks == 0:
        links["global_source_order"] = np.zeros(0, dtype=dtype)
        links["global_source_ids"] = np.zeros(0, dtype=dtype)
        links["global_source_offsets"] = np.zeros(1, dtype=dtype)
        return links

    src = linktab[:, L_STR0_GLOBAL]
    order = np.argsort(src, kind="stable").astype(dtype)
    src_sorted = src[order]
    unique_src, first = np.unique(src_sorted, return_index=True)

    offsets = np.empty(unique_src.size + 1, dtype=dtype)
    offsets[:-1] = first.astype(dtype)
    offsets[-1] = nlinks

    links["global_source_order"] = order
    links["global_source_ids"] = unique_src.astype(dtype)
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


# Constants for pair table fields
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


def build_ab_pair_tables(links_a, links_b, nkpts, kmom=None, kconserv=None):
    kmom = _as_kmom(nkpts, kmom=kmom, kconserv=kconserv)
    ab_pairs = [[None for _ in range(nkpts)] for _ in range(nkpts)]

    for ka in range(nkpts):
        la_tab = get_links_by_k(links_a, ka)

        for kb in range(nkpts):
            rows = []

            for la in la_tab:
                dKa = int(la[L_DK])
                ka1 = int(la[L_K1])
                dKb_needed = int(kmom.kneg[dKa])

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


def build_same_spin_pair_tables(links, nkpts, kmom=None, kconserv=None):
    kmom = _as_kmom(nkpts, kmom=kmom, kconserv=kconserv)
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
                if _kadd(kmom, dK1, dK2) != int(kmom.zero):
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


def build_contract_pair_tables(link_indexa, link_indexb, norb, nkpts,
                               kmom=None, kconserv=None):
    kmom = _as_kmom(nkpts, kmom=kmom, kconserv=kconserv)
    straid_k, strbid_k, str2tot_a, str2tot_b = gen_k_sector_maps(
        link_indexa, link_indexb, nkpts, kmom=kmom)

    links_a = build_k_links_spin(link_indexa, norb, nkpts,
                                 straid_k, str2tot_a, kmom=kmom)
    links_b = build_k_links_spin(link_indexb, norb, nkpts,
                                 strbid_k, str2tot_b, kmom=kmom)

    links_a = build_links_by_global_source_array(links_a)
    links_b = build_links_by_global_source_array(links_b)

    ab_pairs = build_ab_pair_tables(links_a, links_b, nkpts, kmom=kmom)
    aa_pairs = build_same_spin_pair_tables(links_a, nkpts, kmom=kmom)
    bb_pairs = build_same_spin_pair_tables(links_b, nkpts, kmom=kmom)

    return flatten_pair_tables(ab_pairs, aa_pairs, bb_pairs, nkpts)
