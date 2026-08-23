#!/usr/bin/env python
"""Full CI support for spin-free periodic Hamiltonians at fixed momentum.

The determinant basis has fixed numbers of alpha and beta electrons, and
therefore fixed particle number and :math:`M_S`. Determinants are grouped by
the momenta of their alpha and beta strings. Only blocks whose combined
momentum equals ``target_k`` are stored in the packed CI vector.

The symmetry support is:

=========================================  =========
Symmetry                                   Supported
=========================================  =========
Lattice translation / crystal momentum    Yes
Particle number and :math:`M_S`            Yes
Total spin / singlet adaptation            No
Point-group or additional space-group      No
Time-reversal or Kramers                    No
Hermitian Hamiltonian, real or complex     Yes
Alpha/beta orbital degeneracy               Yes
=========================================  =========

``Alpha/beta orbital degeneracy`` means that alpha and beta electrons use the
same spatial orbitals and integrals. Spin-dependent unrestricted Hamiltonians
are not supported. States of different total spin can occur in the same
:math:`M_S` sector; a spin penalty can target a desired spin, but it does not
make the determinant basis spin-adapted. ``orbsym`` and ``wfnsym`` are not
used.

The one-electron integrals must be block diagonal in k-point, and the
two-electron integrals must obey crystal-momentum conservation. No additional
point-group, time-reversal, or Kramers reduction is applied.
"""

import ctypes
import os
from dataclasses import dataclass

import numpy as np

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


libpbckcistring = load_library('libpbc_kcistring')
_contract_map_builder_configured = False
_same_spin_contract_map_builder_configured = False


# Constants for link table fields
L_CRE_L = 0
L_DES_L = 1
L_STR0_LOCAL = 2
L_STR1_LOCAL = 3
L_STR0_GLOBAL = 4
L_STR1_GLOBAL = 5
L_SIGN = 6
L_K0 = 7
L_K1 = 8
L_K_CRE = 9
L_K_DES = 10
L_DK = 11

NLINK_FIELDS = 12


def build_k_links_spin(link_index, norb, nkpts, str_k, str_k2tot,
                       kmom=None, kconserv=None):
    """Build the compact link table for a single spin sector.

    Links are grouped by source-string momentum ``k0`` and momentum
    transfer ``dK``. The returned table also contains local string indices.
    """
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
            assert dK == dK_check, (
                f"dK mismatch at str0={str0_global}, link={j}: "
                f"dK={dK}, but k_cre-k_des={dK_check}"
            )

            # Sanity: target string momentum k1 should be (k0 + dK) % nkpts
            k1 = _kadd(kmom, k0, dK)

            cre_l = cre % norb_per_k
            des_l = des % norb_per_k

            str0_local = int(str_k2tot[k0, str0_global])
            str1_local = int(str_k2tot[k1, str1_global])

            assert str0_local >= 0 and str1_local >= 0, (
                "Momentum sector mismatch"
            )

            # If the target string has no links, e.g. zero-electron sector,
            # link_index[str1_global, 0, 4] may be invalid. Only check when
            # nlink > 0.
            if nlink > 0:
                k1_from_table = int(link_index[str1_global, 0, 4]) % nkpts
                assert k1_from_table == k1, (
                    "Target string sector mismatch: "
                    f"str0={str0_global}, link={j}, str1={str1_global}, "
                    f"expected k1={k1}, table has {k1_from_table}"
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

    # offset_k_dk[k, dK] is the start of links with source sector k and
    # momentum transfer dK; offset_k_dk[k, dK + 1] is the end.
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
AB_A0 = 0
AB_A1 = 1
AB_B0 = 2
AB_B1 = 3
AB_SIGN = 4
AB_KA1 = 5
AB_KB1 = 6
AB_KPA = 7
AB_KQA = 8
AB_KRB = 9
AB_PA = 10
AB_QA = 11
AB_RB = 12
AB_SB = 13
AB_KPB = 14
AB_KQB = 15
AB_KRA = 16
AB_PB = 17
AB_QB = 18
AB_RA = 19
AB_SA = 20
NAB_FIELDS = 21

SS_0 = 0
SS_1 = 1
SS_SIGN = 2
SS_K1 = 3
SS_KP = 4
SS_KQ = 5
SS_KR = 6
SS_P = 7
SS_Q = 8
SS_R = 9
SS_S = 10
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


def _eri_index(kp, kq, kr, p, q, r, s, nkpts, ncas):
    """Return the C-order flat index for a k-point ERI element."""
    return (
        (((((int(kp) * nkpts + int(kq)) * nkpts + int(kr)) * ncas
            + int(p)) * ncas + int(q)) * ncas + int(r)) * ncas
        + int(s)
    )


def build_ab_sparse_contract_map(ab_tab, ab_offsets, blocks, nkpts, ncas):
    dtype = np.int32
    table_size = nkpts * nkpts
    block_offset = -np.ones(table_size, dtype=dtype)
    block_nb = np.zeros(table_size, dtype=dtype)

    for blk in np.asarray(blocks, dtype=dtype).reshape(-1, 6):
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
        group_tab = np.asarray(groups, dtype=dtype, order="C")
    else:
        group_tab = np.zeros((0, 3), dtype=dtype, order="C")

    return {
        "ab_group_tab": group_tab,
        "ab_group_offsets": np.asarray(group_offsets, dtype=dtype,
                                       order="C"),
        "ab_src_addr": np.asarray(src_addrs, dtype=np.int32, order="C"),
        "ab_dst_addr": np.asarray(dst_addrs, dtype=np.int32, order="C"),
        "ab_sign": np.asarray(signs, dtype=np.int32, order="C"),
        "ab_eri_idx_ab": np.asarray(eri_idx_ab, dtype=np.int64, order="C"),
        "ab_eri_idx_ba": np.asarray(eri_idx_ba, dtype=np.int64, order="C"),
    }


def build_same_spin_dense_contract_map(ss_tab, ss_offsets, blocks, nkpts,
                                       ncas, spin):
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


def _configure_contract_map_builder():
    global _contract_map_builder_configured
    if _contract_map_builder_configured:
        return

    libpbckcistring.FCIcount_contract_map_k.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
    ]
    libpbckcistring.FCIcount_contract_map_k.restype = ctypes.c_int

    libpbckcistring.FCIfill_contract_map_k.argtypes = [
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
    libpbckcistring.FCIfill_contract_map_k.restype = ctypes.c_int
    _contract_map_builder_configured = True


def _configure_same_spin_contract_map_builder():
    global _same_spin_contract_map_builder_configured
    if _same_spin_contract_map_builder_configured:
        return

    libpbckcistring.FCIcount_same_spin_contract_map_k.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_void_p,
    ]
    libpbckcistring.FCIcount_same_spin_contract_map_k.restype = (
        ctypes.c_int)

    libpbckcistring.FCIfill_same_spin_contract_map_k.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    libpbckcistring.FCIfill_same_spin_contract_map_k.restype = (
        ctypes.c_int)
    _same_spin_contract_map_builder_configured = True


def _empty_ab_contract_map(nkpts):
    table_size = int(nkpts) * int(nkpts)
    return {
        "ab_group_tab": np.zeros((0, 3), dtype=np.int32),
        "ab_group_offsets": np.zeros(table_size + 1, dtype=np.int32),
        "ab_src_addr": np.zeros(0, dtype=np.int32),
        "ab_dst_addr": np.zeros(0, dtype=np.int32),
        "ab_sign": np.zeros(0, dtype=np.int32),
        "ab_eri_idx_ab": np.zeros(0, dtype=np.int64),
        "ab_eri_idx_ba": np.zeros(0, dtype=np.int64),
    }


def build_same_spin_contract_map_c(link_index, str2tot, blocks,
                                   nkpts, ncas, spin):
    _configure_same_spin_contract_map_builder()

    link_index = np.asarray(link_index, dtype=np.int32, order="C")
    str2tot = np.asarray(str2tot, dtype=np.int32, order="C")
    blocks = np.asarray(blocks, dtype=np.int32, order="C").reshape(-1, 6)
    nstr, nlink, _ = link_index.shape
    spin_id = 0 if spin == "a" else 1
    dims = np.zeros(2, dtype=np.int64)

    with lib.with_omp_threads(lib.num_threads()):
        status = libpbckcistring.FCIcount_same_spin_contract_map_k(
            link_index.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nstr),
            ctypes.c_int(nlink),
            blocks.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(blocks.shape[0]),
            ctypes.c_int(nkpts),
            ctypes.c_int(spin_id),
            dims.ctypes.data_as(ctypes.c_void_p),
        )
    if status != 0:
        raise RuntimeError("FCIcount_same_spin_contract_map_k failed")

    ngroups, nentries = int(dims[0]), int(dims[1])
    if nentries > np.iinfo(np.int32).max:
        raise MemoryError(
            "k-FCI same-spin explicit contract map is too large for "
            f"int32 sparse entries: spin={spin}, entries={nentries}"
        )

    arrays = {
        "group_tab": np.empty((ngroups, 4), dtype=np.int32, order="C"),
        "group_offsets": np.empty(int(nkpts) * int(nkpts) + 1,
                                  dtype=np.int32, order="C"),
        "src_addr": np.empty(nentries, dtype=np.int32, order="C"),
        "dst_addr": np.empty(nentries, dtype=np.int32, order="C"),
        "sign": np.empty(nentries, dtype=np.int32, order="C"),
        "eri_idx": np.empty(nentries, dtype=np.int64, order="C"),
    }

    with lib.with_omp_threads(lib.num_threads()):
        status = libpbckcistring.FCIfill_same_spin_contract_map_k(
            link_index.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nstr),
            ctypes.c_int(nlink),
            str2tot.ctypes.data_as(ctypes.c_void_p),
            blocks.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(blocks.shape[0]),
            ctypes.c_int(nkpts),
            ctypes.c_int(ncas),
            ctypes.c_int(spin_id),
            arrays["group_tab"].ctypes.data_as(ctypes.c_void_p),
            arrays["group_offsets"].ctypes.data_as(ctypes.c_void_p),
            arrays["src_addr"].ctypes.data_as(ctypes.c_void_p),
            arrays["dst_addr"].ctypes.data_as(ctypes.c_void_p),
            arrays["sign"].ctypes.data_as(ctypes.c_void_p),
            arrays["eri_idx"].ctypes.data_as(ctypes.c_void_p),
        )
    if status != 0:
        raise RuntimeError("FCIfill_same_spin_contract_map_k failed")

    return arrays


def build_contract_map_c(link_indexa, link_indexb, str2tot_a,
                         str2tot_b, blocks, nkpts, ncas,
                         explicit_ab=True):
    _configure_contract_map_builder()

    link_indexa = np.asarray(link_indexa, dtype=np.int32, order="C")
    link_indexb = np.asarray(link_indexb, dtype=np.int32, order="C")
    str2tot_a = np.asarray(str2tot_a, dtype=np.int32, order="C")
    str2tot_b = np.asarray(str2tot_b, dtype=np.int32, order="C")
    blocks = np.asarray(blocks, dtype=np.int32, order="C").reshape(-1, 6)

    nstra, nlinka, _ = link_indexa.shape
    nstrb, nlinkb, _ = link_indexb.shape
    dims = np.zeros(6, dtype=np.int64)

    if not explicit_ab:
        aa_dense = build_same_spin_contract_map_c(
            link_indexa, str2tot_a, blocks, nkpts, ncas, "a")
        bb_dense = build_same_spin_contract_map_c(
            link_indexb, str2tot_b, blocks, nkpts, ncas, "b")
        arrays = _empty_ab_contract_map(nkpts)
        arrays.update({
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
        })
        return arrays

    with lib.with_omp_threads(lib.num_threads()):
        status = libpbckcistring.FCIcount_contract_map_k(
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
        raise RuntimeError("FCIcount_contract_map_k failed")

    nab_groups, nab_entries = int(dims[0]), int(dims[1])
    naa_groups, naa_entries = int(dims[2]), int(dims[3])
    nbb_groups, nbb_entries = int(dims[4]), int(dims[5])
    table_size = nkpts * nkpts
    _raise_if_contract_map_too_large(
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
        status = libpbckcistring.FCIfill_contract_map_k(
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
        raise RuntimeError("FCIfill_contract_map_k failed")

    return arrays


def _raise_if_contract_map_too_large(nab_entries, naa_entries,
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


def build_same_spin_pair_contract_maps_py(
        link_indexa, link_indexb, norb, nkpts, blocks, ncas, kmom=None,
        kconserv=None):
    kmom = _as_kmom(nkpts, kmom=kmom, kconserv=kconserv)
    straid_k, strbid_k, str2tot_a, str2tot_b = gen_k_sector_maps(
        link_indexa, link_indexb, nkpts, kmom=kmom)
    links_a = build_k_links_spin(link_indexa, norb, nkpts,
                                 straid_k, str2tot_a, kmom=kmom)
    links_b = build_k_links_spin(link_indexb, norb, nkpts,
                                 strbid_k, str2tot_b, kmom=kmom)
    links_a = build_links_by_global_source_array(links_a)
    links_b = build_links_by_global_source_array(links_b)

    aa_pairs = build_same_spin_pair_tables(links_a, nkpts, kmom=kmom)
    bb_pairs = build_same_spin_pair_tables(links_b, nkpts, kmom=kmom)

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

    aa_tab = (np.asarray(np.vstack(aa_rows), dtype=np.int32, order="C")
              if aa_rows else np.zeros((0, NSS_FIELDS), dtype=np.int32))
    bb_tab = (np.asarray(np.vstack(bb_rows), dtype=np.int32, order="C")
              if bb_rows else np.zeros((0, NSS_FIELDS), dtype=np.int32))

    aa_dense = build_same_spin_dense_contract_map(
        aa_tab, np.asarray(aa_offsets, dtype=np.int32, order="C"),
        blocks, nkpts, ncas, "a")
    bb_dense = build_same_spin_dense_contract_map(
        bb_tab, np.asarray(bb_offsets, dtype=np.int32, order="C"),
        blocks, nkpts, ncas, "b")

    return (aa_tab, np.asarray(aa_offsets, dtype=np.int32, order="C"),
            bb_tab, np.asarray(bb_offsets, dtype=np.int32, order="C"),
            aa_dense, bb_dense)


def _available_memory_bytes():
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)
    except (AttributeError, OSError, ValueError):
        return None


def estimate_ab_entries_upper_bound(link_indexa, link_indexb, blocks, nkpts,
                                    kmom=None, kconserv=None):
    kmom = _as_kmom(nkpts, kmom=kmom, kconserv=kconserv)
    link_indexa = np.asarray(link_indexa, dtype=np.int32, order="C")
    link_indexb = np.asarray(link_indexb, dtype=np.int32, order="C")
    counts_a = np.zeros((nkpts, nkpts), dtype=np.int64)
    counts_b = np.zeros((nkpts, nkpts), dtype=np.int64)

    for link_index, counts in ((link_indexa, counts_a),
                               (link_indexb, counts_b)):
        flat = link_index.reshape(-1, link_index.shape[-1])
        valid = (flat[:, 3] != 0) & (flat[:, 2] >= 0)
        k0 = np.mod(flat[valid, 4], nkpts)
        dk = np.mod(flat[valid, 7], nkpts)
        np.add.at(counts, (k0, dk), 1)

    nentries = 0
    for ka, kb, *_ in np.asarray(blocks, dtype=np.int32).reshape(-1, 6):
        ka = int(ka)
        kb = int(kb)
        for dka in range(nkpts):
            nentries += (int(counts_a[ka, dka]) *
                         int(counts_b[kb, int(kmom.kneg[dka])]))
    return int(nentries)


def estimate_ab_contract_map_bytes(nentries):
    # ab_src_addr, ab_dst_addr, ab_sign are int32; two eri indices are int64.
    return int(nentries) * (3 * 4 + 2 * 8)


def _resolve_explicit_ab(link_indexa, link_indexb, blocks, nkpts,
                         explicit_ab="auto", max_memory=None,
                         memory_fraction=0.5, kmom=None, kconserv=None):
    if explicit_ab is True or explicit_ab is False:
        return bool(explicit_ab)
    if explicit_ab != "auto":
        raise ValueError("explicit_ab must be True, False, or 'auto'")

    nentries = estimate_ab_entries_upper_bound(
        link_indexa, link_indexb, blocks, nkpts, kmom=kmom,
        kconserv=kconserv)
    # The explicit sparse AB map uses int32 entry offsets.  Even when a
    # large-memory node could hold the arrays, the representation cannot
    # address more than int32.max entries, so force the streamed AB kernel.
    if nentries > np.iinfo(np.int32).max:
        return False
    required = estimate_ab_contract_map_bytes(nentries)
    if max_memory is None:
        max_memory = _available_memory_bytes()
    if max_memory is None:
        return True
    return required <= int(max_memory * memory_fraction)


@dataclass
class KFCILayoutMap:
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
    kmom: KPointMomentum

    @classmethod
    def build(cls, norb, nelec, nkpts, target_k, link_index=None,
              kmom=None, kconserv=None, cell=None, kpts=None, kmesh=None):
        nkpts = int(nkpts)
        kmom = _as_kmom(nkpts, kmom=kmom, kconserv=kconserv,
                        cell=cell, kpts=kpts, kmesh=kmesh)
        norb = int(norb)
        ncas = norb // nkpts
        assert ncas * nkpts == norb

        nelec = _unpack_nelec(nelec)
        link_indexa, link_indexb = _unpack_contract_link_index(
            norb, nelec, link_index, nkpts, kmom=kmom)
        link_indexa = np.asarray(link_indexa, dtype=np.int32, order="C")
        link_indexb = np.asarray(link_indexb, dtype=np.int32, order="C")

        blocks = gen_k_sector_linkstr_info(
            link_indexa, link_indexb, nkpts, target_k, kmom=kmom)
        blocks = np.asarray(blocks, dtype=np.int32, order="C")
        sector_size = int(blocks[:, 5].sum()) if blocks.size else 0

        straid_k, strbid_k, str2tot_a, str2tot_b = gen_k_sector_maps(
            link_indexa, link_indexb, nkpts, kmom=kmom)
        stra_ids, stra_offsets = _flatten_sector_ids(straid_k, nkpts)
        strb_ids, strb_offsets = _flatten_sector_ids(strbid_k, nkpts)

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
            kmom=kmom,
        )


@dataclass
class KFCI2EMap:
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
    explicit_ab: bool = True


@dataclass
class KFCIContractMap:
    layout: KFCILayoutMap
    two_e: KFCI2EMap

    def __getattr__(self, name):
        layout = object.__getattribute__(self, "layout")
        two_e = object.__getattribute__(self, "two_e")
        if name in layout.__dataclass_fields__:
            return getattr(layout, name)
        if name in two_e.__dataclass_fields__:
            return getattr(two_e, name)
        raise AttributeError(name)

    @classmethod
    def build(cls, norb, nelec, nkpts, target_k, link_index=None,
              build_pair_tables=False, use_c_contract_map=True,
              explicit_ab="auto", max_memory=None, memory_fraction=0.5,
              kmom=None, kconserv=None, cell=None, kpts=None, kmesh=None):
        layout = KFCILayoutMap.build(
            norb, nelec, nkpts, target_k, link_index=link_index,
            kmom=kmom, kconserv=kconserv, cell=cell, kpts=kpts,
            kmesh=kmesh)
        link_indexa, link_indexb = layout.link_index
        norb = layout.norb
        nkpts = layout.nkpts
        ncas = layout.ncas
        blocks = layout.blocks
        str2tot_a = layout.str2tot_a
        str2tot_b = layout.str2tot_b
        kmom = layout.kmom
        explicit_ab = _resolve_explicit_ab(
            link_indexa, link_indexb, blocks, nkpts,
            explicit_ab=explicit_ab, max_memory=max_memory,
            memory_fraction=memory_fraction, kmom=kmom)

        contract_arrays = None
        if use_c_contract_map and kmom.scalar:
            try:
                contract_arrays = build_contract_map_c(
                    link_indexa, link_indexb, str2tot_a, str2tot_b,
                    blocks, nkpts, ncas, explicit_ab=explicit_ab)
            except AttributeError:
                contract_arrays = None

        if explicit_ab and (build_pair_tables or contract_arrays is None):
            ab_tab, ab_offsets, aa_tab, aa_offsets, bb_tab, bb_offsets = (
                build_contract_pair_tables(link_indexa, link_indexb,
                                           norb, nkpts, kmom=kmom))
            has_pair_tables = True
            if contract_arrays is None:
                ab_sparse = build_ab_sparse_contract_map(
                    ab_tab, ab_offsets, blocks, nkpts, ncas)
                aa_dense = build_same_spin_dense_contract_map(
                    aa_tab, aa_offsets, blocks, nkpts, ncas, "a")
                bb_dense = build_same_spin_dense_contract_map(
                    bb_tab, bb_offsets, blocks, nkpts, ncas, "b")
                contract_arrays = {
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
        elif contract_arrays is None:
            (aa_tab, aa_offsets, bb_tab, bb_offsets, aa_dense, bb_dense) = (
                build_same_spin_pair_contract_maps_py(
                    link_indexa, link_indexb, norb, nkpts, blocks, ncas,
                    kmom=kmom))
            table_size = nkpts * nkpts
            ab_tab = np.zeros((0, NAB_FIELDS), dtype=np.int32)
            ab_offsets = np.zeros(table_size + 1, dtype=np.int32)
            has_pair_tables = False
            contract_arrays = _empty_ab_contract_map(nkpts)
            contract_arrays.update({
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
            })
        else:
            table_size = nkpts * nkpts
            ab_tab = np.zeros((0, NAB_FIELDS), dtype=np.int32)
            ab_offsets = np.zeros(table_size + 1, dtype=np.int32)
            aa_tab = np.zeros((0, NSS_FIELDS), dtype=np.int32)
            aa_offsets = np.zeros(nkpts + 1, dtype=np.int32)
            bb_tab = np.zeros((0, NSS_FIELDS), dtype=np.int32)
            bb_offsets = np.zeros(nkpts + 1, dtype=np.int32)
            has_pair_tables = False

        two_e = KFCI2EMap(
            ab_tab=ab_tab,
            ab_offsets=ab_offsets,
            aa_tab=aa_tab,
            aa_offsets=aa_offsets,
            bb_tab=bb_tab,
            bb_offsets=bb_offsets,
            has_pair_tables=has_pair_tables,
            ab_group_tab=contract_arrays["ab_group_tab"],
            ab_group_offsets=contract_arrays["ab_group_offsets"],
            ab_src_addr=contract_arrays["ab_src_addr"],
            ab_dst_addr=contract_arrays["ab_dst_addr"],
            ab_sign=contract_arrays["ab_sign"],
            ab_eri_idx_ab=contract_arrays["ab_eri_idx_ab"],
            ab_eri_idx_ba=contract_arrays["ab_eri_idx_ba"],
            aa_group_tab=contract_arrays["aa_group_tab"],
            aa_group_offsets=contract_arrays["aa_group_offsets"],
            aa_src_addr=contract_arrays["aa_src_addr"],
            aa_dst_addr=contract_arrays["aa_dst_addr"],
            aa_sign=contract_arrays["aa_sign"],
            aa_eri_idx=contract_arrays["aa_eri_idx"],
            bb_group_tab=contract_arrays["bb_group_tab"],
            bb_group_offsets=contract_arrays["bb_group_offsets"],
            bb_src_addr=contract_arrays["bb_src_addr"],
            bb_dst_addr=contract_arrays["bb_dst_addr"],
            bb_sign=contract_arrays["bb_sign"],
            bb_eri_idx=contract_arrays["bb_eri_idx"],
            explicit_ab=bool(explicit_ab),
        )
        return cls(layout=layout, two_e=two_e)


def make_kfci_contract_map(norb, nelec, nkpts, target_k, link_index=None,
                           build_pair_tables=False, use_c_contract_map=True,
                           explicit_ab="auto", max_memory=None,
                           memory_fraction=0.5, kmom=None, kconserv=None,
                           cell=None, kpts=None, kmesh=None):
    return KFCIContractMap.build(norb, nelec, nkpts, target_k,
                                 link_index=link_index,
                                 build_pair_tables=build_pair_tables,
                                 use_c_contract_map=use_c_contract_map,
                                 explicit_ab=explicit_ab,
                                 max_memory=max_memory,
                                 memory_fraction=memory_fraction,
                                 kmom=kmom, kconserv=kconserv, cell=cell,
                                 kpts=kpts, kmesh=kmesh)


def contract_ab_pairs(eri, ci0_block, ci1_blocks, ab_pairs, ka, kb):
    '''
    Contracting the alpha-beta excitation pairs for a given (ka, kb) block.
    args:
        eri : np.ndarray
            The two-electron integrals in k-space.
        ci0_block : np.ndarray
            The input CI vector block for the (ka, kb) momentum sector.
        ci1_blocks : list of list of np.ndarray
            The output CI vector blocks for all momentum sectors.
        ab_pairs : list of list of np.ndarray
            The alpha-beta excitation-pair tables grouped by (ka, kb).
        ka : int
            The total momentum of the alpha strings in ci0_block.
        kb : int
            The total momentum of the beta strings in ci0_block.
    returns:
        None
            The ci1_blocks arrays are updated in place.
    '''
    pairtab = ab_pairs[ka][kb]

    for row in pairtab:
        a0 = row[AB_A0]
        a1 = row[AB_A1]
        b0 = row[AB_B0]
        b1 = row[AB_B1]
        sign = row[AB_SIGN]
        ka1 = row[AB_KA1]
        kb1 = row[AB_KB1]
        ci1_block = ci1_blocks[ka1][kb1]

        if ci1_block is None:
            continue

        val_ab = eri[
            row[AB_KPA], row[AB_KQA], row[AB_KRB],
            row[AB_PA], row[AB_QA], row[AB_RB], row[AB_SB],
        ]
        val_ba = eri[
            row[AB_KPB], row[AB_KQB], row[AB_KRA],
            row[AB_PB], row[AB_QB], row[AB_RA], row[AB_SA],
        ]

        ci1_block[a1, b1] += (val_ab + val_ba) * sign * ci0_block[a0, b0]


def contract_aa_pairs(eri, ci0_blocks, ci1_blocks, aa_pairs, ka, kb):
    '''
    Contracting the alpha-alpha excitation pairs for a given (ka, kb) block.
    args:
        eri : np.ndarray
            The two-electron integrals in k-space.
        ci0_blocks : list of list of np.ndarray
            The input CI vector blocks for all momentum sectors.
        ci1_blocks : list of list of np.ndarray
            The output CI vector blocks for all momentum sectors.
        aa_pairs : list of np.ndarray
            The alpha-alpha excitation-pair tables grouped by ka.
        ka : int
            The total momentum of the alpha strings in the input block.
        kb : int
            The total momentum of the beta strings in the input block.
    returns:
        None
            The ci1_blocks arrays are updated in place.
    '''
    ci0_block = ci0_blocks[ka][kb]
    if ci0_block is None:
        return

    pairtab = aa_pairs[ka]

    for row in pairtab:
        a0 = row[SS_0]
        a1 = row[SS_1]
        sign = row[SS_SIGN]
        ka1 = row[SS_K1]

        ci1_block = ci1_blocks[ka1][kb]
        if ci1_block is None:
            continue

        val = eri[
            row[SS_KP], row[SS_KQ], row[SS_KR],
            row[SS_P], row[SS_Q], row[SS_R], row[SS_S],
        ]

        ci1_block[a1, :] += val * sign * ci0_block[a0, :]


def contract_bb_pairs(eri, ci0_blocks, ci1_blocks, bb_pairs, ka, kb):
    '''
    Contracting the beta-beta excitation pairs for a given (ka, kb) block.
    args:
        eri : np.ndarray
            The two-electron integrals in k-space.
        ci0_blocks : list of list of np.ndarray
            The input CI vector blocks for all momentum sectors.
        ci1_blocks : list of list of np.ndarray
            The output CI vector blocks for all momentum sectors.
        bb_pairs : list of np.ndarray
            The beta-beta excitation-pair tables grouped by kb.
        ka : int
            The total momentum of the alpha strings in the input block.
        kb : int
            The total momentum of the beta strings in the input block.
    returns:
        None
            The ci1_blocks arrays are updated in place.
    '''
    ci0_block = ci0_blocks[ka][kb]
    if ci0_block is None:
        return

    pairtab = bb_pairs[kb]

    for row in pairtab:
        b0 = row[SS_0]
        b1 = row[SS_1]
        sign = row[SS_SIGN]
        kb1 = row[SS_K1]

        ci1_block = ci1_blocks[ka][kb1]
        if ci1_block is None:
            continue

        val = eri[
            row[SS_KP], row[SS_KQ], row[SS_KR],
            row[SS_P], row[SS_Q], row[SS_R], row[SS_S],
        ]

        ci1_block[:, b1] += val * sign * ci0_block[:, b0]
