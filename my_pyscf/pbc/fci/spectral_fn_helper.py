'''
Helper functions for building k-resolved spectral functions from k-FCI
wavefunctions.
'''

import ctypes
import numpy as np
from dataclasses import dataclass

from pyscf.fci import cistring
from pyscf.fci.addons import _unpack_nelec

from mrh.lib.helper import load_library
from mrh.my_pyscf.pbc.fci import kcistrings, kfci_helper


# Author: Bhavnesh Jangid

libpbcspectral = None
_spectral_lib_initialized = False


def _load_spectral_lib():
    '''
    Load the C library for spectral-function broadening.
    '''
    global libpbcspectral, _spectral_lib_initialized
    if _spectral_lib_initialized:
        return libpbcspectral

    try:
        libpbcspectral = load_library('libpbc_spectral_fn')
    except OSError:
        libpbcspectral = None
        _spectral_lib_initialized = True
        return None

    libpbcspectral.FCIspectral_broaden.argtypes = [
        ctypes.c_void_p,  # hole
        ctypes.c_void_p,  # particle
        ctypes.c_void_p,  # total
        ctypes.c_void_p,  # kind
        ctypes.c_void_p,  # k_index
        ctypes.c_void_p,  # orbital
        ctypes.c_void_p,  # spin
        ctypes.c_void_p,  # omega0
        ctypes.c_void_p,  # weight
        ctypes.c_void_p,  # omega_grid
        ctypes.c_int,     # npoles
        ctypes.c_int,     # nomega
        ctypes.c_int,     # nkpts
        ctypes.c_int,     # norb_axis
        ctypes.c_int,     # spin_axis
        ctypes.c_int,     # orbital_resolved
        ctypes.c_int,     # spin_resolved
        ctypes.c_double,  # eta
        ctypes.c_int,     # broadening
    ]
    libpbcspectral.FCIspectral_broaden.restype = None
    libpbcspectral.FCIspectral_apply_k_op.argtypes = [
        ctypes.c_void_p,  # out
        ctypes.c_void_p,  # fcivec
        ctypes.c_void_p,  # blocks
        ctypes.c_int,     # nblocks
        ctypes.c_int,     # nkpts
        ctypes.c_void_p,  # stra_ids
        ctypes.c_void_p,  # stra_offsets
        ctypes.c_void_p,  # strb_ids
        ctypes.c_void_p,  # strb_offsets
        ctypes.c_void_p,  # target_str2loc_a
        ctypes.c_int,     # target_nstra
        ctypes.c_void_p,  # target_str2loc_b
        ctypes.c_int,     # target_nstrb
        ctypes.c_void_p,  # target_block_offset
        ctypes.c_void_p,  # target_block_na
        ctypes.c_void_p,  # target_block_nb
        ctypes.c_void_p,  # op_index
        ctypes.c_int,     # nlink
        ctypes.c_int,     # orb
        ctypes.c_int,     # k_op
        ctypes.c_int,     # spin
        ctypes.c_int,     # cre
        ctypes.c_int,     # beta_phase
    ]
    libpbcspectral.FCIspectral_apply_k_op.restype = None
    _spectral_lib_initialized = True
    return libpbcspectral


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
    kmom: object


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
    kmom: object


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
    kpts: object = None
    kmesh: object = None
    kconserv: object = None
    kmom: object = None
    spin_sector_mode: str = 'representative'


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


def make_k_sector_layout(norb, nelec, nkpts, target_k=0,
                         link_index=None, nelec_spin=None, kmom=None,
                         kconserv=None, cell=None, kpts=None, kmesh=None,
                         kmf=None, kmc=None):
    '''
    Build the packed determinant layout for one total momentum sector.
    '''
    nelec = _unpack_nelec(nelec, nelec_spin)
    kmom = kcistrings._as_kmom(
        nkpts, kmom=kmom, kconserv=kconserv, cell=cell, kpts=kpts,
        kmesh=kmesh, kmf=kmf, kmc=kmc)
    target_k = int(target_k) % int(nkpts)

    if link_index is None:
        link_index = kfci_helper._unpack_contract_link_index(
            norb, nelec, None, nkpts, spin=nelec_spin, kmom=kmom)
    else:
        assert link_index[0].shape[2] == link_index[1].shape[2] == 8

    stra_id, strb_id, str2loc_a, str2loc_b = \
        kcistrings.gen_k_sector_maps(link_index[0], link_index[1], nkpts,
                                     kmom=kmom)
    blocks = kcistrings.gen_k_sector_linkstr_info(
        link_index[0], link_index[1], nkpts, target_k, kmom=kmom)
    sector_size = int(blocks[:, 5].sum()) if blocks.size else 0

    block_map = {}
    for blk in blocks:
        ka, kb, nstra, nstrb, offset, size = map(int, blk)
        block_map[(ka, kb)] = (offset, nstra, nstrb, size)

    return KSectorLayout(nelec=nelec, target_k=target_k,
                         link_index=link_index, stra_id=stra_id,
                         strb_id=strb_id, str2loc_a=str2loc_a,
                         str2loc_b=str2loc_b, blocks=blocks,
                         block_map=block_map, sector_size=sector_size,
                         kmom=kmom)


def make_k_op_context(norb, nelec, nkpts, target_k, k, p, spin, cre=False,
                      source_link_index=None, target_link_index=None,
                      nelec_spin=None, kmom=None, kconserv=None, cell=None,
                      kpts=None, kmesh=None, kmf=None, kmc=None):
    '''
    Build source and target layouts for a_{kp spin} or a^dagger_{kp spin}.
    '''
    nkpts = int(nkpts)
    kmom = kcistrings._as_kmom(
        nkpts, kmom=kmom, kconserv=kconserv, cell=cell, kpts=kpts,
        kmesh=kmesh, kmf=kmf, kmc=kmc)
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
        charged_target_k = kcistrings._kadd(kmom, target_k, k)
        op_index = cistring.gen_cre_str_index(range(int(norb)), nelec[spin])
    else:
        charged_target_k = kcistrings._ksub(kmom, target_k, k)
        op_index = cistring.gen_des_str_index(range(int(norb)), nelec[spin])

    source = make_k_sector_layout(
        norb, nelec, nkpts, target_k=target_k,
        link_index=source_link_index, nelec_spin=nelec_spin, kmom=kmom)
    target = make_k_sector_layout(
        norb, target_nelec, nkpts, target_k=charged_target_k,
        link_index=target_link_index, nelec_spin=nelec_spin, kmom=kmom)

    return KOpContext(cre=bool(cre), spin=spin, k=k, p=p, orb=orb,
                      source_nelec=nelec, target_nelec=target_nelec,
                      source_target_k=target_k,
                      target_target_k=charged_target_k,
                      source=source, target=target, op_index=op_index,
                      kmom=kmom)


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
            ka1 = kcistrings._kadd(ctx.kmom, ka, ctx.k)
        else:
            ka1 = kcistrings._ksub(ctx.kmom, ka, ctx.k)
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
            kb1 = kcistrings._kadd(ctx.kmom, kb, ctx.k)
        else:
            kb1 = kcistrings._ksub(ctx.kmom, kb, ctx.k)
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


def apply_k_op_py(fcivec, norb, nelec, nkpts, target_k, k, p, spin,
                  cre=False, context=None, return_info=False,
                  source_link_index=None, target_link_index=None,
                  nelec_spin=None, kmom=None, kconserv=None, cell=None,
                  kpts=None, kmesh=None, kmf=None, kmc=None):
    '''
    Apply a single k-resolved creation or annihilation operator.
    '''
    if context is None:
        context = make_k_op_context(
            norb, nelec, nkpts, target_k, k, p, spin, cre=cre,
            source_link_index=source_link_index,
            target_link_index=target_link_index, nelec_spin=nelec_spin,
            kmom=kmom, kconserv=kconserv, cell=cell, kpts=kpts,
            kmesh=kmesh, kmf=kmf, kmc=kmc)

    fcivec = np.asarray(fcivec)
    assert fcivec.size == context.source.sector_size, (
        fcivec.size, context.source.sector_size)
    out = np.zeros(context.target.sector_size, dtype=fcivec.dtype)

    if context.spin == 0:
        _apply_alpha_op(fcivec, out, context)
    else:
        _apply_beta_op(fcivec, out, context)

    if return_info:
        return out, _operator_info(context)
    return out


def _operator_info(context):
    '''
    Return target-sector metadata for an applied creation or annihilation
    operator.
    '''
    return {
        'nelec': context.target_nelec,
        'target_k': context.target_target_k,
        'orb': context.orb,
        'k': context.k,
        'p': context.p,
        'spin': context.spin,
        'cre': context.cre,
    }


def _target_block_tables(layout, nkpts):
    '''
    Build target block lookup tables for the C operator helper.
    '''
    table_size = int(nkpts) * int(nkpts)
    offset = np.full(table_size, -1, dtype=np.int32)
    na = np.zeros(table_size, dtype=np.int32)
    nb = np.zeros(table_size, dtype=np.int32)
    for blk in layout.blocks:
        ka, kb, nstra, nstrb, off, _ = map(int, blk)
        key = ka * int(nkpts) + kb
        offset[key] = off
        na[key] = nstra
        nb[key] = nstrb
    return offset, na, nb


def _apply_k_op_c(fcivec, context):
    '''
    Apply one k-resolved operator through the native C helper.
    '''
    if not getattr(context.kmom, 'scalar', True):
        return None

    lib = _load_spectral_lib()
    if lib is None:
        return None

    fcivec = np.ascontiguousarray(fcivec, dtype=np.complex128)
    out = np.zeros(context.target.sector_size, dtype=np.complex128)
    blocks = np.ascontiguousarray(context.source.blocks, dtype=np.int32)
    op_index = np.ascontiguousarray(context.op_index, dtype=np.int32)
    stra_ids, stra_offsets = kfci_helper._flatten_sector_ids(
        context.source.stra_id, len(context.source.stra_id))
    strb_ids, strb_offsets = kfci_helper._flatten_sector_ids(
        context.source.strb_id, len(context.source.strb_id))
    str2loc_a = np.ascontiguousarray(context.target.str2loc_a,
                                     dtype=np.int32)
    str2loc_b = np.ascontiguousarray(context.target.str2loc_b,
                                     dtype=np.int32)
    block_offset, block_na, block_nb = _target_block_tables(
        context.target, len(context.target.stra_id))
    beta_phase = -1 if (context.source_nelec[0] % 2) else 1

    lib.FCIspectral_apply_k_op(
        out.ctypes.data_as(ctypes.c_void_p),
        fcivec.ctypes.data_as(ctypes.c_void_p),
        blocks.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(blocks.shape[0]),
        ctypes.c_int(len(context.source.stra_id)),
        stra_ids.ctypes.data_as(ctypes.c_void_p),
        stra_offsets.ctypes.data_as(ctypes.c_void_p),
        strb_ids.ctypes.data_as(ctypes.c_void_p),
        strb_offsets.ctypes.data_as(ctypes.c_void_p),
        str2loc_a.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(str2loc_a.shape[1]),
        str2loc_b.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(str2loc_b.shape[1]),
        block_offset.ctypes.data_as(ctypes.c_void_p),
        block_na.ctypes.data_as(ctypes.c_void_p),
        block_nb.ctypes.data_as(ctypes.c_void_p),
        op_index.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(op_index.shape[1]),
        ctypes.c_int(context.orb),
        ctypes.c_int(context.k),
        ctypes.c_int(context.spin),
        ctypes.c_int(1 if context.cre else 0),
        ctypes.c_int(beta_phase),
    )
    return out


def apply_k_op(fcivec, norb, nelec, nkpts, target_k, k, p, spin,
               cre=False, context=None, return_info=False,
               source_link_index=None, target_link_index=None,
               nelec_spin=None, use_c=True, kmom=None, kconserv=None,
               cell=None, kpts=None, kmesh=None, kmf=None, kmc=None):
    '''
    Apply a single k-resolved creation or annihilation operator.
    '''
    if context is None:
        context = make_k_op_context(
            norb, nelec, nkpts, target_k, k, p, spin, cre=cre,
            source_link_index=source_link_index,
            target_link_index=target_link_index, nelec_spin=nelec_spin,
            kmom=kmom, kconserv=kconserv, cell=cell, kpts=kpts,
            kmesh=kmesh, kmf=kmf, kmc=kmc)

    if use_c:
        out = _apply_k_op_c(fcivec, context)
    else:
        out = None
    if out is None:
        return apply_k_op_py(
            fcivec, norb, nelec, nkpts, target_k, k, p, spin,
            cre=cre, context=context, return_info=return_info,
            source_link_index=source_link_index,
            target_link_index=target_link_index, nelec_spin=nelec_spin,
            kmom=kmom, kconserv=kconserv, cell=cell, kpts=kpts,
            kmesh=kmesh, kmf=kmf, kmc=kmc)

    if return_info:
        return out, _operator_info(context)
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
        yield root, energy.item(), ci[root]


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
                'charged_spin': int(result['nelecastot'][0] -
                                    result['nelecastot'][1]),
                'ncastot': int(result['ncastot']),
                'nkpts': int(result['nkpts']),
                'converged': bool(result.get('converged', True)),
            })
    return roots


def _prepare_kcasci_job(kmc, kmom, nroots, solver_setup, kind):
    '''
    Apply the common momentum and solver settings to one spectral KCASCI job.
    '''
    kmc.kmom = kmom
    kmc.kconserv = kmom.kconserv
    kmc.fcisolver.kmom = kmom
    kmc.fcisolver.kconserv = kmom.kconserv
    kmc.canonicalization = False
    if nroots is not None:
        kmc.fcisolver.nroots = int(nroots)
    if solver_setup is not None:
        solver_setup(kmc, kind)


def _charged_spin_list(nelecastot, norb, kind, charged_spin,
                       spin_sector_mode='representative'):
    '''
    Return charged spin sectors for the charged KCASCI jobs.
    '''
    if isinstance(charged_spin, str) and charged_spin == 'default':
        return [None]
    if charged_spin is not None:
        if isinstance(charged_spin, (list, tuple, np.ndarray)):
            return list(charged_spin)
        return [charged_spin]

    key = spin_sector_mode.lower()
    if key in ('representative', 'default', 'spin_free'):
        return [None]
    if key not in ('spin_resolved', 'separate', 'all'):
        raise ValueError(f"unknown spin_sector_mode {spin_sector_mode}")

    neleca, nelecb = map(int, nelecastot)
    spin0 = neleca - nelecb
    if kind == 'hole':
        spins = []
        if neleca > 0:
            spins.append(spin0 - 1)
        if nelecb > 0:
            spins.append(spin0 + 1)
    else:
        spins = []
        if neleca < norb:
            spins.append(spin0 + 1)
        if nelecb < norb:
            spins.append(spin0 - 1)

    out = []
    for spin in spins:
        if spin not in out:
            out.append(spin)
    return out


def compute_kcasci_spectral_roots(kmf, ncas, nelecas, ncore=None,
                                  mo_coeff=None, target_k=0,
                                  nroots_neutral=1, nroots_hole=1,
                                  nroots_particle=1, with_hole=True,
                                  with_particle=True,
                                  charged_spin_hole=None,
                                  charged_spin_particle=None,
                                  spin_sector_mode='representative',
                                  solver_setup=None, verbose=None):
    '''
    Run neutral, hole, and particle k-CASCI jobs and collect their roots.

    The returned root table is the input needed for spectral-function transition
    amplitudes.  Energies follow KCASCI's per-cell convention, with supercell
    values also stored for pole-energy differences.  The default uses one
    representative charged spin sector, consistent with spin-free MCSCF/CASCI.
    Set spin_sector_mode='spin_resolved' to run the alpha/beta sectors
    separately.
    '''
    from mrh.my_pyscf.pbc import mcscf

    if mo_coeff is None:
        mo_coeff = np.asarray(kmf.mo_coeff)
    nelecas = _unpack_nelec(nelecas, kmf.cell.spin)
    kpts = kcistrings._safe_getattr(kmf, 'kpts', None)
    kmesh = kcistrings._safe_getattr(kmf, 'kmesh', None)
    if kpts is None:
        if kmesh is None:
            raise ValueError("kmf.kpts or kmf.kmesh is required for "
                             "k-CASCI spectral roots")
        nkpts = int(np.prod(kmesh))
    else:
        nkpts = len(kpts)
    ncastot = nkpts * int(ncas)
    nelecastot = (nkpts * nelecas[0], nkpts * nelecas[1])
    kmom = kcistrings.make_kpoint_momentum(
        nkpts, cell=kmf.cell, kpts=kpts, kmesh=kmesh, kmf=kmf)

    kmc_neutral = mcscf.KCASCI(kmf, ncas, nelecas, ncore=ncore,
                              target_k=target_k)
    _prepare_kcasci_job(kmc_neutral, kmom, nroots_neutral, solver_setup,
                        'neutral')
    kmc_neutral.kernel(mo_coeff=mo_coeff, verbose=verbose)

    roots = _collect_neutral_roots(kmc_neutral)
    hole_jobs = []
    particle_jobs = []

    if with_hole:
        for charged_spin in _charged_spin_list(
                nelecastot, ncastot, 'hole', charged_spin_hole,
                spin_sector_mode=spin_sector_mode):
            kmc_hole = mcscf.KCASCI(kmf, ncas, nelecas, ncore=ncore,
                                    charge=1, target_k=None,
                                    charged_spin=charged_spin)
            _prepare_kcasci_job(kmc_hole, kmom, nroots_hole, solver_setup,
                                'hole')
            kmc_hole.kernel(mo_coeff=mo_coeff, verbose=verbose)
            roots.extend(_collect_charged_roots(kmc_hole, 'hole'))
            hole_jobs.append(kmc_hole)

    if with_particle:
        for charged_spin in _charged_spin_list(
                nelecastot, ncastot, 'particle', charged_spin_particle,
                spin_sector_mode=spin_sector_mode):
            kmc_particle = mcscf.KCASCI(kmf, ncas, nelecas, ncore=ncore,
                                        charge=-1, target_k=None,
                                        charged_spin=charged_spin)
            _prepare_kcasci_job(kmc_particle, kmom, nroots_particle,
                                solver_setup, 'particle')
            kmc_particle.kernel(mo_coeff=mo_coeff, verbose=verbose)
            roots.extend(_collect_charged_roots(kmc_particle, 'particle'))
            particle_jobs.append(kmc_particle)

    kmc_hole = hole_jobs[0] if len(hole_jobs) == 1 else hole_jobs
    kmc_particle = (particle_jobs[0] if len(particle_jobs) == 1
                    else particle_jobs)
    return KCASCISpectralRoots(neutral=kmc_neutral, hole=kmc_hole,
                               particle=kmc_particle, roots=roots,
                               nkpts=nkpts, ncas=int(ncas),
                               ncastot=ncastot,
                               nelecastot=nelecastot,
                               target_k=int(target_k) % nkpts,
                               mo_coeff=mo_coeff,
                               kpts=kpts,
                               kmesh=kmesh,
                               kconserv=kmom.kconserv,
                               kmom=kmom,
                               spin_sector_mode=spin_sector_mode)


def _root_table(spectral_roots):
    '''
    Return the plain list of root dictionaries used by this module.
    '''
    if isinstance(spectral_roots, KCASCISpectralRoots):
        return spectral_roots.roots
    return list(spectral_roots)


def _select_neutral_root(rows, neutral_root=0, target_k=None):
    '''
    Find the neutral root that defines the initial state.
    '''
    roots = [row for row in rows
             if row['kind'] == 'neutral' and int(row['root']) == neutral_root]
    if target_k is not None:
        target_k = int(target_k)
        roots = [row for row in roots if int(row['target_k']) == target_k]
    if len(roots) != 1:
        raise ValueError(f"expected one neutral root, found {len(roots)}")
    return roots[0]


def _charged_root_index(rows):
    '''
    Index charged roots by type, target momentum, and electron number.
    '''
    table = {}
    for row in rows:
        if row['kind'] not in ('hole', 'particle'):
            continue
        key = (row['kind'], int(row['target_k']),
               tuple(map(int, row['nelecastot'])))
        table.setdefault(key, []).append(row)
    return table


def _index_list(values, stop, name):
    '''
    Normalize optional k/orbital index filters.
    '''
    if values is None:
        return list(range(int(stop)))
    out = []
    for value in values:
        value = int(value)
        if value < 0 or value >= int(stop):
            raise ValueError(f"{name}={value} is outside [0, {stop})")
        out.append(value)
    return out


def _append_poles(poles, op_vec, op_info, charged_rows, neutral, kind,
                  k, p, spin, min_weight):
    '''
    Project one operated vector onto all matching charged roots.
    '''
    e0 = neutral['energy_supercell']
    for charged in charged_rows:
        amp = np.vdot(np.asarray(charged['ci']), op_vec)
        weight = abs(amp) ** 2
        if weight < min_weight:
            continue
        e1 = charged['energy_supercell']
        omega = e0 - e1 if kind == 'hole' else e1 - e0
        poles.append({
            'kind': kind,
            'k': int(k),
            'target_k': int(op_info['target_k']),
            'root': int(charged['root']),
            'neutral_root': int(neutral['root']),
            'orbital': int(p),
            'spin': int(spin),
            'omega': np.real_if_close(omega),
            'weight': np.real_if_close(weight),
            'amplitude': amp,
            'energy_neutral': e0,
            'energy_charged': e1,
            'nelecastot': tuple(map(int, charged['nelecastot'])),
        })


def make_spectral_poles(spectral_roots, neutral_root=0, neutral_target_k=None,
                        k_indices=None, orbital_indices=None, spins=(0, 1),
                        include_hole=True, include_particle=True,
                        min_weight=0.0, strict=False):
    '''
    Build k-resolved hole/particle pole table from neutral and charged roots.

    Hole poles use <Psi_N-1| a_kps |Psi_N> and
    omega = E_N - E_N-1.  Particle poles use
    <Psi_N+1| a^dagger_kps |Psi_N> and omega = E_N+1 - E_N.
    '''
    rows = _root_table(spectral_roots)
    if isinstance(spectral_roots, KCASCISpectralRoots):
        nkpts = spectral_roots.nkpts
        ncas = spectral_roots.ncas
        kmom = spectral_roots.kmom
        neutral_target_k = (spectral_roots.target_k if neutral_target_k is None
                            else neutral_target_k)
    else:
        neutral0 = _select_neutral_root(rows, neutral_root,
                                       target_k=neutral_target_k)
        nkpts = neutral0['nkpts']
        ncas = neutral0['ncastot'] // neutral0['nkpts']
        kmom = kcistrings.make_kpoint_momentum(nkpts)

    neutral = _select_neutral_root(rows, neutral_root,
                                  target_k=neutral_target_k)
    charged_index = _charged_root_index(rows)
    nkpts = int(nkpts)
    ncas = int(ncas)
    norb = nkpts * ncas
    nelec = tuple(map(int, neutral['nelecastot']))
    target_k = int(neutral['target_k'])

    k_list = _index_list(k_indices, nkpts, 'k')
    p_list = _index_list(orbital_indices, ncas, 'orbital')
    spin_list = [_as_spin_id(spin) for spin in spins]
    poles = []

    for k in k_list:
        for p in p_list:
            for spin in spin_list:
                if include_hole and nelec[spin] > 0:
                    op_vec, info = des_k(
                        neutral['ci'], norb, nelec, nkpts, target_k,
                        k, p, spin, return_info=True, kmom=kmom)
                    key = ('hole', int(info['target_k']),
                           tuple(map(int, info['nelec'])))
                    charged_rows = charged_index.get(key, ())
                    if strict and not charged_rows:
                        raise ValueError(f"missing hole sector {key[1:]}")
                    _append_poles(poles, op_vec, info, charged_rows, neutral,
                                  'hole', k, p, spin, min_weight)

                if include_particle and nelec[spin] < norb:
                    op_vec, info = cre_k(
                        neutral['ci'], norb, nelec, nkpts, target_k,
                        k, p, spin, return_info=True, kmom=kmom)
                    key = ('particle', int(info['target_k']),
                           tuple(map(int, info['nelec'])))
                    charged_rows = charged_index.get(key, ())
                    if strict and not charged_rows:
                        raise ValueError(f"missing particle sector {key[1:]}")
                    _append_poles(poles, op_vec, info, charged_rows, neutral,
                                  'particle', k, p, spin, min_weight)

    return poles


def make_omega_grid(poles, npts=801, eta=0.05, padding=None,
                    omega_min=None, omega_max=None):
    '''
    Build a real-frequency grid around the pole energies.
    '''
    if omega_min is None or omega_max is None:
        if not poles:
            raise ValueError("omega_min/omega_max are required for no poles")
        omegas = np.asarray([np.real_if_close(row['omega']).real
                             for row in poles])
        if omega_min is None:
            omega_min = float(omegas.min())
        if omega_max is None:
            omega_max = float(omegas.max())

    if padding is None:
        padding = 8.0 * float(eta)
    omega_min = float(omega_min) - float(padding)
    omega_max = float(omega_max) + float(padding)
    if omega_max <= omega_min:
        omega_min -= float(eta)
        omega_max += float(eta)
    return np.linspace(omega_min, omega_max, int(npts))


def broaden_delta(omega, omega0, eta=0.05, broadening='lorentzian'):
    '''
    Return a normalized broadened delta function on omega.
    '''
    omega = np.asarray(omega)
    x = omega - float(np.real_if_close(omega0).real)
    eta = float(eta)
    if eta <= 0:
        raise ValueError("eta must be positive")

    key = broadening.lower()
    if key in ('lorentzian', 'lorentz'):
        return eta / np.pi / (x * x + eta * eta)
    if key in ('gaussian', 'gauss'):
        return np.exp(-0.5 * (x / eta) ** 2) / (eta * np.sqrt(2.0 * np.pi))
    raise ValueError(f"unknown broadening {broadening}")


def _infer_pole_axes(poles, nkpts=None, norb=None):
    '''
    Infer compact k/orbital dimensions from a pole table.
    '''
    if nkpts is None:
        nkpts = max((int(row['k']) for row in poles), default=-1) + 1
    if norb is None:
        norb = max((int(row['orbital']) for row in poles), default=-1) + 1
    return int(nkpts), int(norb)


def make_spectral_function_py(poles, omega_grid=None, eta=0.05,
                              broadening='lorentzian', npts=801,
                              padding=None, omega_min=None, omega_max=None,
                              nkpts=None, norb=None, spin_resolved=False,
                              orbital_resolved=False):
    '''
    Broaden pole weights into A(k, omega).

    The returned arrays have shape (nkpts, norb_axis, spin_axis, nomega).
    If orbital_resolved/spin_resolved is False, the corresponding axis has
    length one and contains the summed contribution.
    '''
    poles = list(poles)
    if omega_grid is None:
        omega_grid = make_omega_grid(
            poles, npts=npts, eta=eta, padding=padding,
            omega_min=omega_min, omega_max=omega_max)
    else:
        omega_grid = np.asarray(omega_grid)

    nkpts, norb = _infer_pole_axes(poles, nkpts=nkpts, norb=norb)
    norb_axis = norb if orbital_resolved else 1
    spin_axis = 2 if spin_resolved else 1
    shape = (nkpts, norb_axis, spin_axis, omega_grid.size)
    spectra = {
        'hole': np.zeros(shape),
        'particle': np.zeros(shape),
        'total': np.zeros(shape),
    }

    for row in poles:
        kind = row['kind']
        if kind not in ('hole', 'particle'):
            continue
        k = int(row['k'])
        orb = int(row['orbital']) if orbital_resolved else 0
        spin = int(row['spin']) if spin_resolved else 0
        weight = float(np.real_if_close(row['weight']).real)
        if weight == 0.0:
            continue
        delta = broaden_delta(omega_grid, row['omega'], eta=eta,
                              broadening=broadening)
        spectra[kind][k, orb, spin] += weight * delta
        spectra['total'][k, orb, spin] += weight * delta

    return {
        'omega': omega_grid,
        'eta': float(eta),
        'broadening': broadening,
        'spectra': spectra,
        'k_axis': list(range(nkpts)),
        'orbital_axis': list(range(norb)) if orbital_resolved else ['sum'],
        'spin_axis': [0, 1] if spin_resolved else ['sum'],
        'orbital_resolved': bool(orbital_resolved),
        'spin_resolved': bool(spin_resolved),
    }


def _broadening_code(broadening):
    '''
    Convert the broadening label to the C helper convention.
    '''
    key = broadening.lower()
    if key in ('lorentzian', 'lorentz'):
        return 0
    if key in ('gaussian', 'gauss'):
        return 1
    raise ValueError(f"unknown broadening {broadening}")


def _pole_arrays_for_broadening(poles):
    '''
    Pack the pole table into contiguous arrays for the C broadening helper.
    '''
    kind_map = {'hole': 0, 'particle': 1}
    npoles = len(poles)
    kind = np.empty(npoles, dtype=np.int32)
    k_index = np.empty(npoles, dtype=np.int32)
    orbital = np.empty(npoles, dtype=np.int32)
    spin = np.empty(npoles, dtype=np.int32)
    omega0 = np.empty(npoles, dtype=np.float64)
    weight = np.empty(npoles, dtype=np.float64)

    for ipole, row in enumerate(poles):
        row_kind = row['kind']
        if row_kind not in kind_map:
            raise ValueError(f"unknown pole kind {row_kind}")
        kind[ipole] = kind_map[row_kind]
        k_index[ipole] = int(row['k'])
        orbital[ipole] = int(row['orbital'])
        spin[ipole] = int(row['spin'])
        omega0[ipole] = float(np.real_if_close(row['omega']).real)
        weight[ipole] = float(np.real_if_close(row['weight']).real)

    return kind, k_index, orbital, spin, omega0, weight


def make_spectral_function(poles, omega_grid=None, eta=0.05,
                           broadening='lorentzian', npts=801,
                           padding=None, omega_min=None, omega_max=None,
                           nkpts=None, norb=None, spin_resolved=False,
                           orbital_resolved=False, use_c=True):
    '''
    Broaden pole weights into A(k, omega), using the C helper when available.
    '''
    poles = list(poles)
    if not use_c:
        return make_spectral_function_py(
            poles, omega_grid=omega_grid, eta=eta, broadening=broadening,
            npts=npts, padding=padding, omega_min=omega_min,
            omega_max=omega_max, nkpts=nkpts, norb=norb,
            spin_resolved=spin_resolved, orbital_resolved=orbital_resolved)

    lib = _load_spectral_lib()
    if lib is None:
        return make_spectral_function_py(
            poles, omega_grid=omega_grid, eta=eta, broadening=broadening,
            npts=npts, padding=padding, omega_min=omega_min,
            omega_max=omega_max, nkpts=nkpts, norb=norb,
            spin_resolved=spin_resolved, orbital_resolved=orbital_resolved)

    broadening_id = _broadening_code(broadening)
    if omega_grid is None:
        omega_grid = make_omega_grid(
            poles, npts=npts, eta=eta, padding=padding,
            omega_min=omega_min, omega_max=omega_max)
    else:
        omega_grid = np.asarray(omega_grid)

    nkpts, norb = _infer_pole_axes(poles, nkpts=nkpts, norb=norb)
    norb_axis = norb if orbital_resolved else 1
    spin_axis = 2 if spin_resolved else 1
    shape = (nkpts, norb_axis, spin_axis, omega_grid.size)
    spectra = {
        'hole': np.zeros(shape),
        'particle': np.zeros(shape),
        'total': np.zeros(shape),
    }

    kind, k_index, orbital, spin, omega0, weight = \
        _pole_arrays_for_broadening(poles)
    omega_grid = np.ascontiguousarray(omega_grid, dtype=np.float64)

    lib.FCIspectral_broaden(
        spectra['hole'].ctypes.data_as(ctypes.c_void_p),
        spectra['particle'].ctypes.data_as(ctypes.c_void_p),
        spectra['total'].ctypes.data_as(ctypes.c_void_p),
        kind.ctypes.data_as(ctypes.c_void_p),
        k_index.ctypes.data_as(ctypes.c_void_p),
        orbital.ctypes.data_as(ctypes.c_void_p),
        spin.ctypes.data_as(ctypes.c_void_p),
        omega0.ctypes.data_as(ctypes.c_void_p),
        weight.ctypes.data_as(ctypes.c_void_p),
        omega_grid.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(kind.size),
        ctypes.c_int(omega_grid.size),
        ctypes.c_int(nkpts),
        ctypes.c_int(norb_axis),
        ctypes.c_int(spin_axis),
        ctypes.c_int(1 if orbital_resolved else 0),
        ctypes.c_int(1 if spin_resolved else 0),
        ctypes.c_double(float(eta)),
        ctypes.c_int(broadening_id),
    )

    return {
        'omega': omega_grid,
        'eta': float(eta),
        'broadening': broadening,
        'spectra': spectra,
        'k_axis': list(range(nkpts)),
        'orbital_axis': list(range(norb)) if orbital_resolved else ['sum'],
        'spin_axis': [0, 1] if spin_resolved else ['sum'],
        'orbital_resolved': bool(orbital_resolved),
        'spin_resolved': bool(spin_resolved),
    }


def label_pole_momenta(poles, kpts):
    '''
    Attach operator and charged-state momentum vectors to each pole record.
    '''
    kpts = np.asarray(kpts)
    labelled = []
    for row in poles:
        out = dict(row)
        out['operator_momentum'] = np.asarray(kpts[int(row['k'])]).copy()
        out['charged_momentum'] = np.asarray(kpts[int(row['target_k'])]).copy()
        labelled.append(out)
    return labelled


def _coeff_for_k(coeff, k):
    '''
    Return the active-to-band coefficient matrix for one k-point.
    '''
    if isinstance(coeff, dict):
        return np.asarray(coeff[int(k)])
    coeff = np.asarray(coeff)
    if coeff.ndim == 2:
        return coeff
    if coeff.ndim == 3:
        return coeff[int(k)]
    raise ValueError("coeff must have shape (ncas, nband) or "
                     "(nkpts, ncas, nband)")


def project_poles_to_band_basis(poles, coeff, band_indices=None,
                                min_weight=0.0):
    '''
    Project active-orbital pole amplitudes to a supplied band basis.

    coeff[p, b] is the coefficient of active orbital p in band b.  Use
    unfiltered active-orbital poles when possible, because amplitudes from
    different active orbitals are summed before squaring.
    '''
    groups = {}
    for row in poles:
        key = (row['kind'], int(row['k']), int(row['target_k']),
               int(row['root']), int(row['neutral_root']), int(row['spin']),
               complex(row['omega']))
        groups.setdefault(key, []).append(row)

    projected = []
    for key, rows in groups.items():
        kind, k, target_k, root, neutral_root, spin, omega = key
        c = _coeff_for_k(coeff, k)
        if band_indices is None:
            bands = range(c.shape[1])
        else:
            bands = [int(b) for b in band_indices]

        amp_by_orb = {int(row['orbital']): row['amplitude'] for row in rows}
        for band in bands:
            if band < 0 or band >= c.shape[1]:
                raise ValueError(f"band={band} is outside [0, {c.shape[1]})")
            amp = 0.0j
            for p in range(c.shape[0]):
                if kind == 'hole':
                    amp += np.conj(c[p, band]) * amp_by_orb.get(p, 0.0)
                else:
                    amp += c[p, band] * amp_by_orb.get(p, 0.0)
            weight = abs(amp) ** 2
            if weight < min_weight:
                continue

            out = dict(rows[0])
            out.update({
                'k': k,
                'target_k': target_k,
                'root': root,
                'neutral_root': neutral_root,
                'orbital': int(band),
                'band': int(band),
                'spin': spin,
                'omega': np.real_if_close(omega),
                'weight': np.real_if_close(weight),
                'amplitude': amp,
                'basis': 'band',
            })
            projected.append(out)
    return projected


def spectral_weight_sum_rules(spectral_roots, poles=None, neutral_root=0,
                              neutral_target_k=None, k_indices=None,
                              orbital_indices=None, spins=(0, 1),
                              available_only=False):
    '''
    Compare accumulated pole weights with exact operator norms.

    Missing weight reports the contribution absent from the charged roots that
    were supplied to make_spectral_poles.  If available_only is True, spin
    channels with no supplied poles are not counted as missing weight.
    '''
    rows = _root_table(spectral_roots)
    if isinstance(spectral_roots, KCASCISpectralRoots):
        nkpts = spectral_roots.nkpts
        ncas = spectral_roots.ncas
        kmom = spectral_roots.kmom
        neutral_target_k = (spectral_roots.target_k if neutral_target_k is None
                            else neutral_target_k)
    else:
        neutral0 = _select_neutral_root(rows, neutral_root,
                                       target_k=neutral_target_k)
        nkpts = neutral0['nkpts']
        ncas = neutral0['ncastot'] // neutral0['nkpts']
        kmom = kcistrings.make_kpoint_momentum(nkpts)

    neutral = _select_neutral_root(rows, neutral_root,
                                  target_k=neutral_target_k)
    if poles is None:
        poles = make_spectral_poles(
            spectral_roots, neutral_root=neutral_root,
            neutral_target_k=neutral_target_k, k_indices=k_indices,
            orbital_indices=orbital_indices, spins=spins, min_weight=0.0)

    nkpts = int(nkpts)
    ncas = int(ncas)
    norb = nkpts * ncas
    nelec = tuple(map(int, neutral['nelecastot']))
    target_k = int(neutral['target_k'])
    k_list = _index_list(k_indices, nkpts, 'k')
    p_list = _index_list(orbital_indices, ncas, 'orbital')
    spin_list = [_as_spin_id(spin) for spin in spins]

    pole_weight = {}
    for row in poles:
        key = (row['kind'], int(row['k']), int(row['orbital']),
               int(row['spin']))
        pole_weight[key] = pole_weight.get(key, 0.0) + \
            float(np.real_if_close(row['weight']).real)

    checks = []
    for k in k_list:
        for p in p_list:
            for spin in spin_list:
                h_key = ('hole', k, p, spin)
                p_key = ('particle', k, p, spin)
                do_hole = not available_only or h_key in pole_weight
                do_particle = not available_only or p_key in pole_weight

                hole_norm = None if not do_hole else 0.0
                particle_norm = None if not do_particle else 0.0
                if do_hole and nelec[spin] > 0:
                    vec = des_k(neutral['ci'], norb, nelec, nkpts,
                                target_k, k, p, spin, kmom=kmom)
                    hole_norm = float(np.real_if_close(np.vdot(vec, vec)).real)
                if do_particle and nelec[spin] < norb:
                    vec = cre_k(neutral['ci'], norb, nelec, nkpts,
                                target_k, k, p, spin, kmom=kmom)
                    particle_norm = float(
                        np.real_if_close(np.vdot(vec, vec)).real)

                h_wt = None if not do_hole else pole_weight.get(h_key, 0.0)
                p_wt = (None if not do_particle
                        else pole_weight.get(p_key, 0.0))
                h_miss = None if not do_hole else hole_norm - h_wt
                p_miss = (None if not do_particle
                          else particle_norm - p_wt)
                total_norm = sum(x for x in (hole_norm, particle_norm)
                                 if x is not None)
                total_weight = sum(x for x in (h_wt, p_wt)
                                   if x is not None)
                checks.append({
                    'k': int(k),
                    'orbital': int(p),
                    'spin': int(spin),
                    'hole_norm': hole_norm,
                    'hole_weight': h_wt,
                    'hole_missing': h_miss,
                    'particle_norm': particle_norm,
                    'particle_weight': p_wt,
                    'particle_missing': p_miss,
                    'total_norm': total_norm,
                    'total_weight': total_weight,
                    'total_missing': total_norm - total_weight,
                })
    return checks


def save_spectral_npz(filename, spectrum, poles=None):
    '''
    Save the broadened spectral function and optional pole table to npz.
    '''
    data = {
        'omega': spectrum['omega'],
        'hole': spectrum['spectra']['hole'],
        'particle': spectrum['spectra']['particle'],
        'total': spectrum['spectra']['total'],
        'eta': np.asarray(spectrum['eta']),
        'broadening': np.asarray(spectrum['broadening']),
        'k_axis': np.asarray(spectrum['k_axis']),
        'orbital_axis': np.asarray(spectrum['orbital_axis']),
        'spin_axis': np.asarray(spectrum['spin_axis']),
    }
    if poles is not None:
        data['poles'] = np.asarray(poles, dtype=object)
    np.savez(filename, **data)


def plot_spectral_function(spectrum, kind='total', k=0, orbital=0, spin=0,
                           ax=None):
    '''
    Plot one A(k, omega) trace from make_spectral_function output.
    '''
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    y = spectrum['spectra'][kind][int(k), int(orbital), int(spin)]
    ax.plot(spectrum['omega'], y)
    ax.set_xlabel(r'$\omega$')
    ax.set_ylabel(f'A_{kind}(k={int(k)})')
    return fig, ax


def des_k_py(fcivec, norb, nelec, nkpts, target_k, k, p, spin,
             context=None, return_info=False, source_link_index=None,
             target_link_index=None, nelec_spin=None, kmom=None,
             kconserv=None, cell=None, kpts=None, kmesh=None, kmf=None,
             kmc=None):
    '''
    Return a_{k p spin} |Psi_N> in the N-1 target momentum sector.
    '''
    return apply_k_op_py(fcivec, norb, nelec, nkpts, target_k, k, p, spin,
                         cre=False, context=context, return_info=return_info,
                         source_link_index=source_link_index,
                         target_link_index=target_link_index,
                         nelec_spin=nelec_spin, kmom=kmom,
                         kconserv=kconserv, cell=cell, kpts=kpts,
                         kmesh=kmesh, kmf=kmf, kmc=kmc)


def cre_k_py(fcivec, norb, nelec, nkpts, target_k, k, p, spin,
             context=None, return_info=False, source_link_index=None,
             target_link_index=None, nelec_spin=None, kmom=None,
             kconserv=None, cell=None, kpts=None, kmesh=None, kmf=None,
             kmc=None):
    '''
    Return a^dagger_{k p spin} |Psi_N> in the N+1 target momentum sector.
    '''
    return apply_k_op_py(fcivec, norb, nelec, nkpts, target_k, k, p, spin,
                         cre=True, context=context, return_info=return_info,
                         source_link_index=source_link_index,
                         target_link_index=target_link_index,
                         nelec_spin=nelec_spin, kmom=kmom,
                         kconserv=kconserv, cell=cell, kpts=kpts,
                         kmesh=kmesh, kmf=kmf, kmc=kmc)


def des_k(fcivec, norb, nelec, nkpts, target_k, k, p, spin,
          context=None, return_info=False, source_link_index=None,
          target_link_index=None, nelec_spin=None, use_c=True, kmom=None,
          kconserv=None, cell=None, kpts=None, kmesh=None, kmf=None,
          kmc=None):
    '''
    Return a_{k p spin} |Psi_N> in the N-1 target momentum sector.
    '''
    return apply_k_op(fcivec, norb, nelec, nkpts, target_k, k, p, spin,
                      cre=False, context=context, return_info=return_info,
                      source_link_index=source_link_index,
                      target_link_index=target_link_index,
                      nelec_spin=nelec_spin, use_c=use_c, kmom=kmom,
                      kconserv=kconserv, cell=cell, kpts=kpts,
                      kmesh=kmesh, kmf=kmf, kmc=kmc)


def cre_k(fcivec, norb, nelec, nkpts, target_k, k, p, spin,
          context=None, return_info=False, source_link_index=None,
          target_link_index=None, nelec_spin=None, use_c=True, kmom=None,
          kconserv=None, cell=None, kpts=None, kmesh=None, kmf=None,
          kmc=None):
    '''
    Return a^dagger_{k p spin} |Psi_N> in the N+1 target momentum sector.
    '''
    return apply_k_op(fcivec, norb, nelec, nkpts, target_k, k, p, spin,
                      cre=True, context=context, return_info=return_info,
                      source_link_index=source_link_index,
                      target_link_index=target_link_index,
                      nelec_spin=nelec_spin, use_c=use_c, kmom=kmom,
                      kconserv=kconserv, cell=cell, kpts=kpts,
                      kmesh=kmesh, kmf=kmf, kmc=kmc)


__all__ = [
    'KSectorLayout',
    'KOpContext',
    'KCASCISpectralRoots',
    'make_k_sector_layout',
    'make_k_op_context',
    'compute_kcasci_spectral_roots',
    'make_spectral_poles',
    'make_omega_grid',
    'broaden_delta',
    'make_spectral_function_py',
    'make_spectral_function',
    'label_pole_momenta',
    'project_poles_to_band_basis',
    'spectral_weight_sum_rules',
    'save_spectral_npz',
    'plot_spectral_function',
    'apply_k_op_py',
    'apply_k_op',
    'des_k_py',
    'des_k',
    'cre_k_py',
    'cre_k',
]
