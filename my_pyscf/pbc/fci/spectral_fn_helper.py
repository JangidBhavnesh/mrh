#!/bin/bash

import numpy as np
from dataclasses import dataclass

from pyscf.fci import cistring
from pyscf.fci.addons import _unpack_nelec

from mrh.my_pyscf.pbc.fci import kcistrings


# Author: Bhavnesh Jangid

'''
Helper functions for building k-resolved spectral functions from k-FCI
wavefunctions.
'''


@dataclass
class KSectorLayout:
    '''
    Packed determinant layout for one k-FCI total momentum sector.
    '''
    nelec: tuple
    target_k: int
    link_index: tuple
    stra_id: list
    strb_id: list
    str2loc_a: np.ndarray
    str2loc_b: np.ndarray
    blocks: np.ndarray
    block_map: dict
    sector_size: int


@dataclass
class KOpContext:
    '''
    Cached source/target layouts for a single creation or annihilation operator.
    '''
    cre: bool
    spin: int
    k: int
    p: int
    orb: int
    source_nelec: tuple
    target_nelec: tuple
    source_target_k: int
    target_target_k: int
    source: KSectorLayout
    target: KSectorLayout
    op_index: np.ndarray


@dataclass
class KCASCISpectralRoots:
    '''
    Neutral and charged k-CASCI roots needed before transition amplitudes.
    '''
    neutral: object
    hole: object
    particle: object
    roots: list
    nkpts: int
    ncas: int
    ncastot: int
    nelecastot: tuple
    target_k: int
    mo_coeff: object


def _as_spin_id(spin):
    '''
    Convert alpha/beta spin labels to the integer convention used locally.
    '''
    if isinstance(spin, str):
        key = spin.lower()
        if key in ('a', 'alpha', 'up'):
            return 0
        if key in ('b', 'beta', 'down'):
            return 1
    spin = int(spin)
    if spin not in (0, 1):
        raise ValueError("spin must be 0/alpha or 1/beta")
    return spin


def _orbital_id(norb, nkpts, k, p):
    '''
    Convert a k-point and local active-orbital index to a global orbital id.
    '''
    nkpts = int(nkpts)
    ncas = int(norb) // nkpts
    assert ncas * nkpts == int(norb)
    k = int(k) % nkpts
    p = int(p)
    if p < 0 or p >= ncas:
        raise ValueError(f"local orbital p={p} is outside [0, {ncas})")
    return k * ncas + p


def _target_nelec(nelec, spin, cre):
    '''
    Electron tuple after applying one spin-resolved operator.
    '''
    neleca, nelecb = nelec
    target = [int(neleca), int(nelecb)]
    if cre:
        target[spin] += 1
    else:
        target[spin] -= 1
    if target[spin] < 0:
        raise ValueError("annihilation requested from an empty spin sector")
    return tuple(target)


def _make_link_index(norb, nelec, nkpts):
    '''
    Build k-aware link indices.  Only the K0 labels are used here.
    '''
    neleca, nelecb = nelec
    ncas = int(norb) // int(nkpts)
    orb_k = np.arange(int(norb), dtype=np.int32) // ncas
    link_indexa = kcistrings.gen_linkstr_index_k(
        range(int(norb)), neleca, orb_k, int(nkpts))
    if neleca == nelecb:
        link_indexb = link_indexa
    else:
        link_indexb = kcistrings.gen_linkstr_index_k(
            range(int(norb)), nelecb, orb_k, int(nkpts))
    return link_indexa, link_indexb


def make_k_sector_layout(norb, nelec, nkpts, target_k=0,
                         link_index=None, nelec_spin=None):
    '''
    Build the packed determinant layout for one total momentum sector.
    '''
    nelec = _unpack_nelec(nelec, nelec_spin)
    target_k = int(target_k) % int(nkpts)

    if link_index is None:
        link_index = _make_link_index(norb, nelec, nkpts)
    else:
        assert link_index[0].shape[2] == link_index[1].shape[2] == 8

    stra_id, strb_id, str2loc_a, str2loc_b = \
        kcistrings.gen_k_sector_maps(link_index[0], link_index[1], nkpts)
    blocks = kcistrings.gen_k_sector_linkstr_info(
        link_index[0], link_index[1], nkpts, target_k)
    sector_size = int(blocks[:, 5].sum()) if blocks.size else 0

    block_map = {}
    for blk in blocks:
        ka, kb, nstra, nstrb, offset, size = map(int, blk)
        block_map[(ka, kb)] = (offset, nstra, nstrb, size)

    return KSectorLayout(nelec=nelec, target_k=target_k,
                         link_index=link_index, stra_id=stra_id,
                         strb_id=strb_id, str2loc_a=str2loc_a,
                         str2loc_b=str2loc_b, blocks=blocks,
                         block_map=block_map, sector_size=sector_size)


def make_k_op_context(norb, nelec, nkpts, target_k, k, p, spin, cre=False,
                      source_link_index=None, target_link_index=None,
                      nelec_spin=None):
    '''
    Build source and target layouts for a_{kp spin} or a^dagger_{kp spin}.
    '''
    nkpts = int(nkpts)
    spin = _as_spin_id(spin)
    nelec = _unpack_nelec(nelec, nelec_spin)
    orb = _orbital_id(norb, nkpts, k, p)
    k = int(k) % nkpts
    p = int(p)

    target_nelec = _target_nelec(nelec, spin, cre)
    if target_nelec[spin] > int(norb):
        raise ValueError("creation requested into a full spin sector")
    target_k = int(target_k) % nkpts
    if cre:
        charged_target_k = (target_k + k) % nkpts
        op_index = cistring.gen_cre_str_index(range(int(norb)), nelec[spin])
    else:
        charged_target_k = (target_k - k) % nkpts
        op_index = cistring.gen_des_str_index(range(int(norb)), nelec[spin])

    source = make_k_sector_layout(
        norb, nelec, nkpts, target_k=target_k,
        link_index=source_link_index, nelec_spin=nelec_spin)
    target = make_k_sector_layout(
        norb, target_nelec, nkpts, target_k=charged_target_k,
        link_index=target_link_index, nelec_spin=nelec_spin)

    return KOpContext(cre=bool(cre), spin=spin, k=k, p=p, orb=orb,
                      source_nelec=nelec, target_nelec=target_nelec,
                      source_target_k=target_k,
                      target_target_k=charged_target_k,
                      source=source, target=target, op_index=op_index)


def _apply_alpha_op(fcivec, out, ctx):
    '''
    Apply a single alpha creation/destruction operator to a packed k-sector CI.
    '''
    ORB = 0 if ctx.cre else 1
    TARGET = 2
    SIGN = 3

    for blk in ctx.source.blocks:
        ka, kb, nstra, nstrb, offset, size = map(int, blk)
        if ctx.cre:
            ka1 = (ka + ctx.k) % len(ctx.target.stra_id)
        else:
            ka1 = (ka - ctx.k) % len(ctx.target.stra_id)
        target_blk = ctx.target.block_map.get((ka1, kb))
        if target_blk is None:
            continue

        off1, nstra1, nstrb1, size1 = target_blk
        src = fcivec[offset:offset + size].reshape(nstra, nstrb)
        dst = out[off1:off1 + size1].reshape(nstra1, nstrb1)

        for ia0, str0 in enumerate(ctx.source.stra_id[ka]):
            for link in ctx.op_index[int(str0)]:
                if int(link[ORB]) != ctx.orb:
                    continue
                str1 = int(link[TARGET])
                ia1 = int(ctx.target.str2loc_a[ka1, str1])
                if ia1 < 0:
                    continue
                dst[ia1, :] += int(link[SIGN]) * src[ia0, :]


def _apply_beta_op(fcivec, out, ctx):
    '''
    Apply a single beta creation/destruction operator to a packed k-sector CI.
    '''
    ORB = 0 if ctx.cre else 1
    TARGET = 2
    SIGN = 3
    beta_phase = -1 if (ctx.source_nelec[0] % 2) else 1

    for blk in ctx.source.blocks:
        ka, kb, nstra, nstrb, offset, size = map(int, blk)
        if ctx.cre:
            kb1 = (kb + ctx.k) % len(ctx.target.strb_id)
        else:
            kb1 = (kb - ctx.k) % len(ctx.target.strb_id)
        target_blk = ctx.target.block_map.get((ka, kb1))
        if target_blk is None:
            continue

        off1, nstra1, nstrb1, size1 = target_blk
        src = fcivec[offset:offset + size].reshape(nstra, nstrb)
        dst = out[off1:off1 + size1].reshape(nstra1, nstrb1)

        for ib0, str0 in enumerate(ctx.source.strb_id[kb]):
            for link in ctx.op_index[int(str0)]:
                if int(link[ORB]) != ctx.orb:
                    continue
                str1 = int(link[TARGET])
                ib1 = int(ctx.target.str2loc_b[kb1, str1])
                if ib1 < 0:
                    continue
                dst[:, ib1] += beta_phase * int(link[SIGN]) * src[:, ib0]


def apply_k_op(fcivec, norb, nelec, nkpts, target_k, k, p, spin,
               cre=False, context=None, return_info=False,
               source_link_index=None, target_link_index=None,
               nelec_spin=None):
    '''
    Apply a single k-resolved creation or annihilation operator.
    '''
    if context is None:
        context = make_k_op_context(
            norb, nelec, nkpts, target_k, k, p, spin, cre=cre,
            source_link_index=source_link_index,
            target_link_index=target_link_index, nelec_spin=nelec_spin)

    fcivec = np.asarray(fcivec)
    assert fcivec.size == context.source.sector_size, (
        fcivec.size, context.source.sector_size)
    out = np.zeros(context.target.sector_size, dtype=fcivec.dtype)

    if context.spin == 0:
        _apply_alpha_op(fcivec, out, context)
    else:
        _apply_beta_op(fcivec, out, context)

    if return_info:
        info = {
            'nelec': context.target_nelec,
            'target_k': context.target_target_k,
            'orb': context.orb,
            'k': context.k,
            'p': context.p,
            'spin': context.spin,
            'cre': context.cre,
        }
        return out, info
    return out


def _iter_roots(e_tot, ci):
    '''
    Iterate over roots from the scalar/list conventions returned by k-FCI.
    '''
    e = np.asarray(e_tot)
    if e.ndim == 0:
        yield 0, e.item(), ci
        return

    for root, energy in enumerate(e.reshape(-1)):
        if isinstance(ci, np.ndarray):
            ci_root = ci[root]
        else:
            ci_root = ci[root]
        yield root, energy.item(), ci_root


def _collect_neutral_roots(kmc):
    '''
    Collect neutral KCASCI roots into spectral-function records.
    '''
    nelecas = _unpack_nelec(kmc.nelecas, kmc._scf.cell.spin)
    nelecastot = (kmc.nkpts * nelecas[0], kmc.nkpts * nelecas[1])
    ncastot = kmc.nkpts * kmc.ncas
    roots = []

    for root, energy, ci in _iter_roots(kmc.e_tot, kmc.ci):
        roots.append({
            'kind': 'neutral',
            'charge': 0,
            'target_k': int(kmc.target_k) % kmc.nkpts,
            'root': int(root),
            'energy': energy,
            'energy_supercell': energy * kmc.nkpts,
            'ci': ci,
            'nelecastot': nelecastot,
            'ncastot': ncastot,
            'nkpts': int(kmc.nkpts),
            'converged': bool(np.all(getattr(kmc, 'converged', True))),
        })

    return roots


def _collect_charged_roots(kmc, kind):
    '''
    Collect charged KCASCI roots into spectral-function records.
    '''
    roots = []
    for result in kmc.charged_results:
        for root, energy, ci in _iter_roots(result['e_tot'], result['ci']):
            roots.append({
                'kind': kind,
                'charge': int(result['charge']),
                'target_k': int(result['target_k']),
                'root': int(root),
                'energy': energy,
                'energy_supercell': energy * int(result['nkpts']),
                'ci': ci,
                'nelecastot': result['nelecastot'],
                'ncastot': int(result['ncastot']),
                'nkpts': int(result['nkpts']),
                'converged': bool(result.get('converged', True)),
            })
    return roots


def _set_solver_nroots(kmc, nroots):
    '''
    Set nroots when requested and leave user-configured defaults otherwise.
    '''
    if nroots is not None:
        kmc.fcisolver.nroots = int(nroots)


def compute_kcasci_spectral_roots(kmf, ncas, nelecas, ncore=None,
                                  mo_coeff=None, target_k=0,
                                  nroots_neutral=1, nroots_hole=1,
                                  nroots_particle=1, with_hole=True,
                                  with_particle=True,
                                  charged_spin_hole=None,
                                  charged_spin_particle=None,
                                  solver_setup=None, verbose=None):
    '''
    Run neutral, hole, and particle k-CASCI jobs and collect their roots.

    The returned root table is the input needed for spectral-function transition
    amplitudes.  Energies follow KCASCI's per-cell convention, with supercell
    values also stored for pole-energy differences.
    '''
    from mrh.my_pyscf.pbc import mcscf

    if mo_coeff is None:
        mo_coeff = np.asarray(kmf.mo_coeff)

    kmc_neutral = mcscf.KCASCI(kmf, ncas, nelecas, ncore=ncore,
                              target_k=target_k)
    kmc_neutral.canonicalization = False
    _set_solver_nroots(kmc_neutral, nroots_neutral)
    if solver_setup is not None:
        solver_setup(kmc_neutral, 'neutral')
    kmc_neutral.kernel(mo_coeff=mo_coeff, verbose=verbose)

    roots = _collect_neutral_roots(kmc_neutral)
    kmc_hole = None
    kmc_particle = None

    if with_hole:
        kmc_hole = mcscf.KCASCI(kmf, ncas, nelecas, ncore=ncore,
                                charge=1, target_k=None,
                                charged_spin=charged_spin_hole)
        kmc_hole.canonicalization = False
        _set_solver_nroots(kmc_hole, nroots_hole)
        if solver_setup is not None:
            solver_setup(kmc_hole, 'hole')
        kmc_hole.kernel(mo_coeff=mo_coeff, verbose=verbose)
        roots.extend(_collect_charged_roots(kmc_hole, 'hole'))

    if with_particle:
        kmc_particle = mcscf.KCASCI(kmf, ncas, nelecas, ncore=ncore,
                                    charge=-1, target_k=None,
                                    charged_spin=charged_spin_particle)
        kmc_particle.canonicalization = False
        _set_solver_nroots(kmc_particle, nroots_particle)
        if solver_setup is not None:
            solver_setup(kmc_particle, 'particle')
        kmc_particle.kernel(mo_coeff=mo_coeff, verbose=verbose)
        roots.extend(_collect_charged_roots(kmc_particle, 'particle'))

    nelecas = _unpack_nelec(nelecas, kmf.cell.spin)
    nkpts = len(kmf.kpts)
    return KCASCISpectralRoots(neutral=kmc_neutral, hole=kmc_hole,
                               particle=kmc_particle, roots=roots,
                               nkpts=nkpts, ncas=int(ncas),
                               ncastot=nkpts * int(ncas),
                               nelecastot=(nkpts * nelecas[0],
                                           nkpts * nelecas[1]),
                               target_k=int(target_k) % nkpts,
                               mo_coeff=mo_coeff)


def des_k(fcivec, norb, nelec, nkpts, target_k, k, p, spin,
          context=None, return_info=False, source_link_index=None,
          target_link_index=None, nelec_spin=None):
    '''
    Return a_{k p spin} |Psi_N> in the N-1 target momentum sector.
    '''
    return apply_k_op(fcivec, norb, nelec, nkpts, target_k, k, p, spin,
                      cre=False, context=context, return_info=return_info,
                      source_link_index=source_link_index,
                      target_link_index=target_link_index,
                      nelec_spin=nelec_spin)


def cre_k(fcivec, norb, nelec, nkpts, target_k, k, p, spin,
          context=None, return_info=False, source_link_index=None,
          target_link_index=None, nelec_spin=None):
    '''
    Return a^dagger_{k p spin} |Psi_N> in the N+1 target momentum sector.
    '''
    return apply_k_op(fcivec, norb, nelec, nkpts, target_k, k, p, spin,
                      cre=True, context=context, return_info=return_info,
                      source_link_index=source_link_index,
                      target_link_index=target_link_index,
                      nelec_spin=nelec_spin)


__all__ = [
    'KSectorLayout',
    'KOpContext',
    'KCASCISpectralRoots',
    'make_k_sector_layout',
    'make_k_op_context',
    'compute_kcasci_spectral_roots',
    'apply_k_op',
    'des_k',
    'cre_k',
]
