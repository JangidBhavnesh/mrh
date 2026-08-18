#!/bin/bash 

import ctypes
import warnings
import numpy as np
from dataclasses import dataclass

from pyscf.pbc.lib import kpts_helper
from pyscf.fci.cistring import OIndexList, make_strings

from mrh.lib.helper import load_library

libpbckcistring = load_library('libpbc_kcistring')

# Author: Bhavnesh Jangid

# TODO: Add the openMP parallelization to the link index generation in pbc_kcistring.c.
# TODO: Move the below checks to a unit test.


@dataclass
class KPointMomentum:
    '''
    Table-driven k-point arithmetic for total-momentum sector labels.
    '''
    nkpts: int
    kconserv: np.ndarray
    kadd: np.ndarray
    ksub: np.ndarray
    kneg: np.ndarray
    zero: int = 0
    scalar: bool = True


def _scalar_kconserv(nkpts):
    kconserv = np.empty((nkpts, nkpts, nkpts), dtype=np.int32)
    for kp in range(nkpts):
        for kq in range(nkpts):
            for kr in range(nkpts):
                kconserv[kp, kq, kr] = (kp - kq + kr) % nkpts
    return kconserv


def _find_gamma_kpt(cell, kpts):
    '''
    Return the index of Gamma in kpts, falling back to 0 with a warning.
    '''
    kpts = np.asarray(kpts)
    if kpts.size == 0:
        return 0
    try:
        scaled = cell.get_scaled_kpts(kpts)
        frac = scaled - np.rint(scaled)
        err = np.linalg.norm(frac, axis=1)
    except Exception:
        err = np.linalg.norm(kpts, axis=1)
    idx = int(np.argmin(err))
    if err[idx] > 1e-8:
        msg = ("Could not identify Gamma in kpts; using kpts[0] as "
               "the neutral momentum label. The minimum distance to "
               "Gamma is {:.2e}.".format(err[idx]))
        warnings.warn(msg, RuntimeWarning)
    return idx


def _safe_getattr(obj, name, default=None):
    '''
    getattr for optional PySCF properties that can raise while inferring data.
    '''
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def resolve_kpts(cell=None, kpts=None, kmesh=None, kmf=None, kmc=None):
    '''
    Read kpts from kmc/kmf or generate them from kmesh with a warning.
    '''
    if kmc is not None:
        if kpts is None:
            kpts = _safe_getattr(kmc, 'kpts', None)
        scf_obj = _safe_getattr(kmc, '_scf', None)
        if kpts is None and scf_obj is not None:
            kpts = _safe_getattr(scf_obj, 'kpts', None)
        if kmesh is None:
            kmesh = _safe_getattr(kmc, 'kmesh', None)
        if cell is None:
            cell = _safe_getattr(kmc, 'cell', None)
    if kmf is not None:
        if kpts is None:
            kpts = _safe_getattr(kmf, 'kpts', None)
        if kmesh is None:
            kmesh = _safe_getattr(kmf, 'kmesh', None)
        if cell is None:
            cell = _safe_getattr(kmf, 'cell', None)

    if kpts is None and kmesh is not None and cell is not None:
        warnings.warn("kpts were not found on kmc/kmf; generating kpts from "
                      "kmesh. Pass kmf.kpts/kmc.kpts to avoid ambiguity.",
                      RuntimeWarning)
        kpts = cell.make_kpts(kmesh, wrap_around=True)
    return cell, kpts, kmesh


def make_kpoint_momentum(nkpts, cell=None, kpts=None, kmesh=None,
                         kconserv=None, kmf=None, kmc=None):
    '''
    Build table-driven k-point arithmetic from kconserv.
    '''
    nkpts = int(nkpts)
    cell, kpts, kmesh = resolve_kpts(cell=cell, kpts=kpts, kmesh=kmesh,
                                     kmf=kmf, kmc=kmc)
    if kconserv is None:
        if kpts is not None and cell is not None:
            kconserv = kpts_helper.get_kconserv(cell, np.asarray(kpts))
        else:
            kconserv = _scalar_kconserv(nkpts)
    kconserv = np.asarray(kconserv, dtype=np.int32, order="C")
    assert kconserv.shape == (nkpts, nkpts, nkpts)

    zero = _find_gamma_kpt(cell, kpts) if kpts is not None and cell is not None else 0
    kadd = np.asarray(kconserv[:, zero, :], dtype=np.int32, order="C")
    ksub = np.asarray(kconserv[:, :, zero], dtype=np.int32, order="C")
    kneg = np.asarray(kconserv[zero, :, zero], dtype=np.int32, order="C")

    scalar = (zero == 0 and
              np.all(kadd == _scalar_kconserv(nkpts)[:, 0, :]) and
              np.all(ksub == _scalar_kconserv(nkpts)[:, :, 0]) and
              np.all(kneg == np.asarray([(-k) % nkpts
                                         for k in range(nkpts)],
                                        dtype=np.int32)))
    return KPointMomentum(nkpts=nkpts, kconserv=kconserv, kadd=kadd,
                          ksub=ksub, kneg=kneg, zero=int(zero),
                          scalar=bool(scalar))


def _as_kmom(nkpts, kmom=None, kconserv=None, **kwargs):
    if isinstance(kmom, KPointMomentum):
        return kmom
    return make_kpoint_momentum(nkpts, kconserv=kconserv, **kwargs)


def _kadd(kmom, a, b):
    return int(kmom.kadd[int(a), int(b)])

def _ksub(kmom, a, b):
    return int(kmom.ksub[int(a), int(b)])

def _string_momentum(strs, orb_k, kmom):
    '''
    Compute total momentum labels for bit strings.
    '''
    strs = np.asarray(strs, dtype=np.uint64)
    orb_k = np.asarray(orb_k, dtype=np.int32)
    out = np.empty(strs.size, dtype=np.int32)
    for i, s in enumerate(strs):
        k = int(kmom.zero)
        x = int(s)
        while x:
            lsb = x & -x
            orb = lsb.bit_length() - 1
            k = _kadd(kmom, k, int(orb_k[orb]))
            x ^= lsb
        out[i] = k
    return out

def _overwrite_link_momentum(link_index, strs, orb_k, kmom):
    '''
    Replace scalar momentum columns by table-driven k-point labels.
    '''
    if link_index.shape[1] == 0:
        return link_index

    str_k = _string_momentum(strs, orb_k, kmom)
    link_index[:, :, 4] = str_k[:, None]
    flat = link_index.reshape(-1, link_index.shape[-1])
    flat[:, 5] = np.asarray(orb_k[flat[:, 0]], dtype=np.int32)
    flat[:, 6] = np.asarray(orb_k[flat[:, 1]], dtype=np.int32)
    flat[:, 7] = kmom.ksub[flat[:, 5], flat[:, 6]]
    target_k = str_k[flat[:, 2]]
    expected = kmom.kadd[flat[:, 4], flat[:, 7]]
    if not np.all(target_k == expected):
        raise RuntimeError("k-point momentum table is inconsistent with "
                           "generated string links")
    return link_index

def gen_linkstr_index_k(orb_list, nocc, orb_k, nkpts, strs=None,
                        kmom=None, kconserv=None):
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
    kmom = _as_kmom(nkpts, kmom=kmom, kconserv=kconserv)
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

    if not kmom.scalar:
        link_index = _overwrite_link_momentum(link_index, strs, orb_k, kmom)

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

def gen_k_sector_linkstr_info(link_indexa, link_indexb, nkpts, kindx,
                              kmom=None, kconserv=None):
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

    kmom = _as_kmom(nkpts, kmom=kmom, kconserv=kconserv)
    kindx = int(kindx) % nkpts

    count_a, count_b = _count_det_per_k((link_indexa, link_indexb))

    blocks = []
    offset = 0

    for ka in range(nkpts):
        kb = _ksub(kmom, kindx, ka)

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

def _build_sector_map_spin(link_index, nkpts, kmom=None, kconserv=None):
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
    kmom = _as_kmom(nkpts, kmom=kmom, kconserv=kconserv)
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
        str_k[int(kmom.zero)] = np.arange(nstr, dtype=dtype)
        str_k2tot = -np.ones((nkpts, nstr), dtype=dtype)
        str_k2tot[int(kmom.zero), :nstr] = np.arange(nstr, dtype=dtype)
        return str_k, str_k2tot
    
    _str_k = np.asarray(link_index[:, 0, 4], dtype=dtype)

    str_k = [np.where(_str_k == k)[0].astype(dtype, copy=False)
             for k in range(nkpts)]

    str_k2tot = -np.ones((nkpts, nstr), dtype=dtype)
    
    for k, ids in enumerate(str_k):
        ids = np.asarray(ids, dtype=dtype)
        str_k2tot[k, ids] = np.arange(ids.size, dtype=dtype)
    
    return str_k, str_k2tot

def gen_k_sector_maps(link_indexa, link_indexb, nkpts, kmom=None,
                      kconserv=None):
    '''
    Build alpha/beta sector string lists and global-to-local maps.
    '''
    kmom = _as_kmom(nkpts, kmom=kmom, kconserv=kconserv)
    alpha_by_kindx, alpha_str_k2tot = _build_sector_map_spin(
        link_indexa, nkpts, kmom=kmom)
    beta_by_kindx, beta_str_k2tot = _build_sector_map_spin(
        link_indexb, nkpts, kmom=kmom)
    return alpha_by_kindx, beta_by_kindx, alpha_str_k2tot, beta_str_k2tot
