# !/usr/bin/env python

import numpy as np
from functools import reduce

from pyscf import lib, __config__
from pyscf.lib import logger

from mrh.my_pyscf.pbc.fci.addons import _unpack_nelec
from mrh.my_pyscf.pbc import fci as pbc_fci
from mrh.my_pyscf.pbc.fci import kcistrings
from mrh.my_pyscf.pbc.mcscf import casci

# Author: Bhavnesh Jangid

# k-space CASCI driver for the k-FCI solver.

def h1e_for_cas(mc, mo_coeff=None, ncas=None, ncore=None):
    '''
    Compute the k-space one-electron active-space Hamiltonian and core energy.

    Args:
        mc : pbc.mcscf.KCASCI
            The k-CASCI object.
        mo_coeff : np.ndarray [nk, nao, nmo]
            orbitals at each k-point.
            nao and nmo are the number of 
            atomic orbitals and molecular orbitals at each k-point.
        ncas : int
            number of active orbitals at each k-point.
        ncore : int
            number of core orbitals at each k-point.
    Returns:
        h1eff : np.ndarray [nk, ncas, ncas]
            The effective one-electron Hamiltonian in k-space.
        ecore : np.complex128
            The core energy.
    '''

    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ncas is None: ncas = mc.ncas
    if ncore is None: ncore = mc.ncore

    cell = mc.cell
    kmf = mc._scf
    nkpts = mc.nkpts
    nocc = ncore + ncas

    hcore = mc.get_hcore()
    dtype = mo_coeff[0].dtype
    hcore = hcore.astype(dtype)

    mo_core = [mo[:, :ncore] for mo in mo_coeff]
    mo_cas = np.asarray([mo[:, ncore:nocc] 
                         for mo in mo_coeff], dtype=dtype)

    # Remember, the total energy is divided by nkpts later.
    ecore = mc.energy_nuc() * nkpts
    if ncore > 0:
        dm_core = np.asarray([2.0 * mo_core[k] @ mo_core[k].conj().T
                              for k in range(nkpts)], dtype=dtype)
        corevhf = mc.get_veff(cell, dm_core, hermi=1, kpts=kmf.kpts)
        fock_core = hcore + 0.5 * corevhf
        ecore += sum(np.einsum('ij,ji', dm_core[k], fock_core[k])
                     for k in range(nkpts))
        hcore += corevhf

    h1eff = np.asarray([mo_cas[k].conj().T @ hcore[k] @ mo_cas[k]
                        for k in range(nkpts)], dtype=dtype)
    return h1eff, ecore


def get_h2eff(mc, mo_coeff=None):
    '''
    Compute the k-space two-electron active-space Hamiltonian.

    The returned integrals include the supercell normalization and the factor
    of one half expected by the lower-level k-FCI two-electron contraction.
    args:
        mc : pbc.mcscf.KCASCI
            The k-CASCI object.
        mo_coeff : np.ndarray [nk, nao, nmo]
            orbitals at each k-point.
            nao and nmo are the number of 
            atomic orbitals and molecular orbitals at each k-point.
    returns:
        h2eff : np.ndarray [nk, nk, nk, ncas, ncas, ncas, ncas]
            The effective two-electron Hamiltonian in k-space.
    '''
    if mo_coeff is None: mo_coeff = mc.mo_coeff

    kmf = mc._scf
    nkpts = mc.nkpts
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    dtype = mo_coeff[0].dtype
    mo_cas = np.asarray(
        [mo[:, ncore:nocc] for mo in mo_coeff], dtype=dtype)

    h2eff = kmf.with_df.ao2mo_7d(mo_cas, kpts=kmf.kpts)
    h2eff = np.asarray(h2eff, dtype=dtype) / nkpts
    h2eff *= 0.5
    return h2eff


def _adjust_h1eff_for_kfci(h1eff, h2eff):
    '''
    Apply the one-body correction associated with the 
    k-FCI 0.5*h2 convention. In other words, subtract 
    the effective J contribution from the one-body Hamiltonian.
    Remember: I am not doing a single contraction like done
    in the regular casci, I am doing contract_1e and contract_2e separately.
    
    args:
        h1eff : np.ndarray [nk, ncas, ncas]
            The effective one-electron Hamiltonian in k-space.
        h2eff : np.ndarray [nk, nk, nk, ncas, ncas, ncas, ncas]
            The effective two-electron Hamiltonian in k-space.
    returns:
        h1eff : np.ndarray [nk, ncas, ncas]
            The effective one-electron Hamiltonian in k-space with the J
            contribution subtracted.
    '''
    nkpts = h1eff.shape[0]
    j_eff = np.zeros_like(h1eff)
    for kp in range(nkpts):
        for kq in range(nkpts):
            j_eff[kp] += np.einsum('piis->ps', h2eff[kp, kq, kq])

    return h1eff - j_eff


def _get_kmom_for_kcasci(mc):
    '''
    Build k-point arithmetic from the k-CASCI/SCF kpts metadata.
    '''
    kmf = mc._scf
    # _safe_getattr is used to avoid AttributeError due to dynamic
    # attributes in the k-CASCI object.
    kpts = kcistrings._safe_getattr(mc, 'kpts', None)
    if kpts is None:
        kpts = kcistrings._safe_getattr(kmf, 'kpts', None)
    kmesh = kcistrings._safe_getattr(mc, 'kmesh', None)
    if kmesh is None:
        kmesh = kcistrings._safe_getattr(kmf, 'kmesh', None)
    kconserv = getattr(mc, 'kconserv', None)
    kmom = kcistrings.make_kpoint_momentum(mc.nkpts, cell=mc.cell, 
                                           kpts=kpts, kmesh=kmesh,
                                           kconserv=kconserv, kmf=kmf, kmc=mc)
    return kmom


def _set_solver_kpts(mc, kmom=None):
    '''
    Pass k-point metadata from k-CASCI to the k-FCI solver.
    I don't know how to do this in a more elegant way, 
    but the k-FCI solver needs the kconserv and kpts/kmesh
    attributes to be set before the kernel is called.
    '''
    if kmom is None:
        kmom = _get_kmom_for_kcasci(mc)
    mc.kconserv = kmom.kconserv
    mc.fcisolver.kpts = kcistrings._safe_getattr(
        mc, 'kpts', kcistrings._safe_getattr(mc._scf, 'kpts', None))
    mc.fcisolver.kmesh = kcistrings._safe_getattr(
        mc, 'kmesh', kcistrings._safe_getattr(mc._scf, 'kmesh', None))
    mc.fcisolver.kconserv = kmom.kconserv
    mc.fcisolver.kmom = kmom
    return kmom


def kernel(mc, mo_coeff=None, ci0=None, verbose=logger.NOTE, envs=None):
    '''
    # Kernel for k-CASCI.
    # see mrh.my_pyscf.pbc.mcscf.casci.kernel for more details on the 
    # arguments and return values.
    '''
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci0 is None: ci0 = mc.ci

    log = logger.new_logger(mc, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Start k-CASCI')

    nkpts = mc.nkpts
    ncas = mc.ncas
    nelecas = _unpack_nelec(mc.nelecas, mc._scf.cell.spin)

    h1eff, energy_core = mc.get_h1eff(mo_coeff)
    t1 = log.timer('one-electron integral computation for k-CAS', *t0)
    h2eff = mc.get_h2eff(mo_coeff)
    t1 = log.timer('integral transformation to k-CAS space', *t1)
    h1eff = _adjust_h1eff_for_kfci(h1eff, h2eff)
    log.debug('core energy = %.15g', energy_core.real)

    assert h1eff.shape == (nkpts, ncas, ncas)
    assert h2eff.shape == (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)

    kmom = _set_solver_kpts(mc)

    max_memory = max(4000, mc.max_memory - lib.current_memory()[0])
    ncastot = nkpts * ncas
    nelecastot = (nkpts * nelecas[0], nkpts * nelecas[1])
    if mc.target_k is not None:
        target_k = int(mc.target_k)
        assert isinstance(target_k, (int, np.integer)), \
            "target_k must be an integer or None"
        if not 0 <= target_k < nkpts:
            target_k = int(target_k) % nkpts
            log.warn("target_k is out of bounds, using %d instead", target_k)

    # The k-FCI solver needs to know the total number of k-points, the target
    # k-point, and the k-point arithmetic (kconserv) before the kernel is called.
    mc.fcisolver.nkpts = nkpts
    mc.fcisolver.target_k = target_k
    mc.fcisolver.kmom = kmom

    e_tot, fcivec = mc.fcisolver.kernel(h1eff, h2eff, ncastot, nelecastot,
                                        ci0=ci0, nkpts=nkpts, target_k=target_k,
                                        verbose=log, max_memory=max_memory,
                                        ecore=energy_core)
    t1 = log.timer('k-FCI solver', *t1)
    e_cas = e_tot - energy_core

    # The energy is per-unit cell.
    e_cas /= nkpts
    e_tot /= nkpts
    return e_tot, e_cas, fcivec


def _get_nelecas_for_charged_kcasci(ncas, nkpts, nelecas, cell_spin, charge=0,
                           spin=None):
    '''
    Return the total k-FCI active electron tuple for a charged supercell.
    Positive charge removes active electrons and negative charge adds them.
    If spin is omitted, the lowest valid Ms sector is used for the charged
    electron count.
    '''
    # I am keeping this check: I don't know what would it looks like to have 
    # doubly charged k-CASCI.
    assert -1<=charge<=1,\
          "Only single-electron charged k-CASCI is supported" 
    nelecas = _unpack_nelec(nelecas, cell_spin)
    nelec0 = nkpts * (nelecas[0] + nelecas[1])
    nelec = nelec0 - int(charge)
    norb = nkpts * int(ncas)
    if spin is None:
        spin = nelec % 2
    spin = int(spin)
    if nelec < 0 or nelec > 2 * norb:
        raise ValueError(f"charge={charge} gives {nelec} active electrons "
                         f"for {norb} active orbitals")
    if (nelec + spin) % 2:
        raise ValueError(f"active electron count {nelec} and spin {spin} "
                         "have inconsistent parity")
    neleca = (nelec + spin) // 2
    nelecb = nelec - neleca
    if neleca < 0 or nelecb < 0 or neleca > norb or nelecb > norb:
        raise ValueError(f"charge={charge}, spin={spin} gives invalid "
                         f"active electrons ({neleca}, {nelecb})")
    return int(neleca), int(nelecb)


def _select_root_energy(energy, root=0):
    e = np.asarray(energy)
    if e.ndim == 0:
        if root not in (0, None):
            raise IndexError("root index requested for a scalar energy")
        return e.item()
    return e[int(root)].item()


def compute_band_energies(charged_results, reference_energy, charge=None,
                          root=0, kpts=None, nkpts=None, per_cell=False,
                          reference_target_k=None, kmom=None, cell=None,
                          kconserv=None):
    r'''
    Convert charged-sector KCASCI energies into quasiparticle band energies.
        [
        \epsilon_\text{hole}(k)=E_N-E_{N-1}=-\mathrm{IP}
        ]

        [
        \epsilon_\text{particle}(k)=E_{N+1}-E_N=-\mathrm{EA}
        ]

    The returned removal (hole) pole is E(N) - E(N-1) = -IP, while the
    addition (particle) pole is E(N+1) - E(N) = -EA.  The charged KCASCI
    ``target_k`` labels the total momentum of the N+/-1 state, not directly
    the momentum of the removed or added electron.  For a reference-sector
    momentum K0,

        k_hole = K0 - K(N-1)
        k_particle = K(N+1) - K0

    Momentum subtraction uses the table-driven k-point arithmetic so this
    also works for nontrivial k-point orderings and multidimensional meshes.

    args:
        charged_results : list of dict
            The output of kernel_chrkcasci for one or more target_k sectors.
        reference_energy : float or list of float
            The neutral KCASCI energy for the same active space.
        charge: int +1 or -1
            charge of the charged sector.
            +1 means: hole
            -1 means: particle
        root : int
            The root index of the charged sector to use for the band energy.
        reference_target_k : int or None
            Total-momentum sector of the neutral reference.  Defaults to the
            Gamma index from the k-point momentum table.
        kmom : kcistrings.KPointMomentum or None
            Explicit k-point arithmetic.  This is inferred from a charged
            KCASCI object when possible.
    '''
    if hasattr(charged_results, 'charged_results'):

        mc = charged_results
        if charge is None: charge = mc.charge
        if nkpts is None: nkpts = mc.nkpts
        if kpts is None: kpts = getattr(mc._scf, 'kpts', None)
        if cell is None: cell = mc.cell
        if kconserv is None: kconserv = getattr(mc, 'kconserv', None)
        if kmom is None: kmom = _get_kmom_for_kcasci(mc)

        charged_results = mc.charged_results

    if charge is None:
        charged_results = list(charged_results)
        if not charged_results: return [] # Edge case: no charged results
        charge = charged_results[0].get('charge')

    if int(charge) == 0:
        raise ValueError("charged band energies require a nonzero charge")

    kind = 'hole' if int(charge) > 0 else 'particle'
    momentum_key = f'{kind}_momentum'

    bands = []
    for result in charged_results:
        target_k = int(result['target_k'])
        result_nkpts = result.get('nkpts', nkpts)
        if result_nkpts is None:
            raise ValueError("nkpts is required when charged results do not "
                             "store it")
        result_nkpts = int(result_nkpts)
        scale = 1 if per_cell else result_nkpts

        result_kmom = kmom
        if result_kmom is None or result_kmom.nkpts != result_nkpts:
            result_kmom = kcistrings.make_kpoint_momentum(
                result_nkpts, cell=cell, kpts=kpts, kconserv=kconserv)
        if reference_target_k is None:
            reference_k = int(result_kmom.zero)
        else:
            reference_k = int(reference_target_k) % result_nkpts

        if kind == 'hole':
            band_k = int(result_kmom.ksub[reference_k, target_k])
        else:
            band_k = int(result_kmom.ksub[target_k, reference_k])

        momentum = band_k
        if kpts is not None:
            momentum = np.asarray(kpts[band_k]).copy()
        e_charged = _select_root_energy(result['e_tot'], root=root)
        e_ref = _select_root_energy(reference_energy, root=0)
        if kind == 'hole':
            energy = scale * (e_ref - e_charged)
        else:
            energy = scale * (e_charged - e_ref)
        bands.append({'target_k': target_k,
            'momentum_index': band_k,
            momentum_key: momentum,
            'energy': energy,
            'root': int(root),
            'charge': int(charge),
            'kind': kind,
        })
    return bands


def kernel_chrkcasci(mc, mo_coeff=None, ci0=None, verbose=logger.NOTE,
                   target_k=None, charge=None, charged_spin=None, envs=None):
    '''
    Charged k-CASCI kernel.  This keeps the neutral KCASCI kernel unchanged and
    solves one or all charged total-momentum sectors with a shared Hamiltonian.
    '''
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff

    log = logger.new_logger(mc, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Start charged k-CASCI')

    nkpts = mc.nkpts
    ncas = mc.ncas
    charge = mc.charge if charge is None else int(charge)
    charged_spin = mc.charged_spin if charged_spin is None else charged_spin

    h1eff, energy_core = mc.get_h1eff(mo_coeff)
    h2eff = mc.get_h2eff(mo_coeff)
    h1eff = _adjust_h1eff_for_kfci(h1eff, h2eff)
    t1 = log.timer('integral transformation to charged k-CAS space', *t0)
    log.debug('core energy = %.15g', energy_core.real)

    assert h1eff.shape == (nkpts, ncas, ncas)
    assert h2eff.shape == (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)

    kmom = _set_solver_kpts(mc)
 
    max_memory = max(4000, mc.max_memory - lib.current_memory()[0])
    ncastot = nkpts * ncas
    nelecastot = _get_nelecas_for_charged_kcasci(
        ncas, nkpts, mc.nelecas, mc._scf.cell.spin,
        charge=charge, spin=charged_spin)

    target_ks = mc._target_ks(target_k=target_k)
    if len(target_ks) > 1 and ci0 is not None and not isinstance(ci0, dict):
        raise ValueError("ci0 for a charged target_k sweep must be a dict "
                         "keyed by target_k")

    results = []
    e_tot_all = []
    e_cas_all = []
    ci_all = []
    converged = []
    for tk in target_ks:
        tk = int(tk) % nkpts
        ci0_k = ci0.get(tk) if isinstance(ci0, dict) else ci0
        log.info('Solving charged k-CASCI target_k = %d, nelec = %s',
                 tk, nelecastot)

        mc.fcisolver.nkpts = nkpts
        mc.fcisolver.target_k = tk
        mc.fcisolver.kmom = kmom
        e_tot, fcivec = mc.fcisolver.kernel(
            h1eff, h2eff, ncastot, nelecastot, ci0=ci0_k, nkpts=nkpts,
            target_k=tk, verbose=log, max_memory=max_memory,
            ecore=energy_core)
        t1 = log.timer('charged k-FCI solver target_k = %d' % tk, *t1)
        e_cas = e_tot - energy_core
        e_tot /= nkpts
        e_cas /= nkpts

        conv = getattr(mc.fcisolver, 'converged', True)
        conv = bool(np.all(conv))
        converged.append(conv)
        results.append({
            'target_k': int(tk),
            'charge': int(charge),
            'ncas': int(ncas),
            'ncastot': int(ncastot),
            'nelecas': nelecastot,
            'nelecastot': nelecastot,
            'nkpts': int(nkpts),
            'e_tot': e_tot,
            'e_cas': e_cas,
            'e_tot_supercell': e_tot * nkpts,
            'e_cas_supercell': e_cas * nkpts,
            'ci': fcivec,
            'converged': conv,
        })
        e_tot_all.append(e_tot)
        e_cas_all.append(e_cas)
        ci_all.append(fcivec)

    return results, e_tot_all, e_cas_all, ci_all, nelecastot, converged


def _casdm1_for_kcasci(mc, ci, stav_dm1=False):
    '''
    Build the k-basis active-space 1-RDM for KCASCI.
    '''
    from pyscf.mcscf import addons

    nkpts = mc.nkpts
    ncas = mc.ncas
    nelecas = mc.nelecas
    nelecastot = (nkpts * nelecas[0], nkpts * nelecas[1])

    if (isinstance(ci, (list, tuple, casci.RANGE_TYPE)) and
            not isinstance(mc.fcisolver, addons.StateAverageFCISolver)):
        if not stav_dm1:
            return mc.fcisolver.make_rdm1(ci[0], nkpts * ncas, nelecastot)

        casdm1 = mc.fcisolver.make_rdm1(ci[0], nkpts * ncas, nelecastot)
        for root in range(1, len(ci)):
            casdm1 += mc.fcisolver.make_rdm1(
                ci[root], nkpts * ncas, nelecastot)
        return casdm1 / len(ci)

    return mc.fcisolver.make_rdm1(ci, nkpts * ncas, nelecastot)


def _make_rdm1_ao(mc, mo_coeff, ci, ncas, ncore, nelecas, target_k):
    '''
    Transform a k-FCI active-space 1-RDM to the AO basis at each k-point.
    '''
    nkpts = mc.nkpts
    ncastot = nkpts * ncas
    casdm1 = mc.fcisolver.make_rdm1(
        ci, ncastot, nelecas, nkpts=nkpts, target_k=target_k)
    casdm1 = np.asarray(casdm1)
    if casdm1.shape != (ncastot, ncastot):
        raise ValueError(
            f'Expected an active-space 1-RDM with shape '
            f'{(ncastot, ncastot)}, got {casdm1.shape}')

    nao = mo_coeff[0].shape[0]
    dtype = np.result_type(casdm1, *[mo.dtype for mo in mo_coeff])
    dm1 = np.empty((nkpts, nao, nao), dtype=dtype)
    for k in range(nkpts):
        mocore = mo_coeff[k][:, :ncore]
        mocas = mo_coeff[k][:, ncore:ncore + ncas]
        p0 = k * ncas
        p1 = p0 + ncas
        dm1[k] = 2.0 * mocore @ mocore.conj().T
        dm1[k] += mocas @ casdm1[p0:p1, p0:p1] @ mocas.conj().T
    return dm1


def get_fock(mc, mo_coeff=None, ci=None, eris=None, casdm1=None,
             verbose=None):
    '''
    Generalized Fock matrix for KCASCI using the k-basis active-space RDM.
    '''
    raise NotImplementedError('The Fock matrix is not implemented for KCASCI.')

    if ci is None:
        ci = mc.ci
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if casdm1 is None:
        casdm1 = _casdm1_for_kcasci(mc, ci)

    nkpts = mc.nkpts
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    kmf = mc._scf
    dtype = np.result_type(casdm1, *[mo.dtype for mo in mo_coeff])

    mo_core = [mo[:, :ncore] for mo in mo_coeff]
    dm_k = np.asarray([2.0 * mo_core[k] @ mo_core[k].conj().T
                       for k in range(nkpts)], dtype=dtype)

    for k in range(nkpts):
        mocas = mo_coeff[k][:, ncore:nocc]
        p0 = k * ncas
        p1 = p0 + ncas
        dm_k[k] += reduce(np.dot, (mocas, casdm1[p0:p1, p0:p1],
                                   mocas.conj().T))

    hcore = mc.get_hcore()
    veff = mc.get_veff(mc.cell, dm_k, hermi=1, kpts=kmf.kpts)
    return np.asarray([hcore[k] + veff[k] for k in range(nkpts)], dtype=dtype)


@lib.with_doc(casci.canonicalize.__doc__)
def canonicalize(mc, mo_coeff=None, ci=None, eris=None, sort=False,
                 cas_natorb=False, casdm1=None, verbose=logger.NOTE,
                 with_meta_lowdin=casci.WITH_META_LOWDIN, stav_dm1=False):
    raise NotImplementedError('Canonicalization is not implemented for KCASCI.')

    log = logger.new_logger(mc, verbose)
    log.debug('Canonicalizing KCASCI orbitals')

    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci is None:
        ci = mc.ci
    if cas_natorb:
        raise NotImplementedError

    if casdm1 is None:
        casdm1 = _casdm1_for_kcasci(mc, ci, stav_dm1=stav_dm1)

    nkpts = mc.nkpts
    ncas = mc.ncas
    ncore = mc.ncore
    nocc = ncore + ncas
    nmo = mo_coeff[0].shape[1]

    fock_ao = get_fock(mc, mo_coeff=mo_coeff, ci=ci, casdm1=casdm1,
                       verbose=verbose)
    mo_coeff1 = mo_coeff.copy()

    log.info('Density matrix diagonal elements')
    for k in range(nkpts):
        p0 = k * ncas
        p1 = p0 + ncas
        dm_k = casdm1[p0:p1, p0:p1]
        log.info("k-point %d, only real diagonal = %s",
                 k,
                 np.array2string(np.diag(dm_k).real, precision=5,
                                 floatmode='fixed', separator=', '))

    mo_energy = [
        np.einsum('pi,pi->i', mo_coeff1[k].conj(), fock_ao[k] @ mo_coeff1[k])
        for k in range(nkpts)
    ]

    orbsym_extra = np.zeros(nmo, dtype=int)

    def _diag_subfock_(idx):
        if idx.size > 1:
            for k in range(nkpts):
                c = mo_coeff1[k][:, idx]
                fock = reduce(np.dot, (c.conj().T, fock_ao[k], c))
                w, c = mc._eig(fock, None, None, orbsym_extra[idx])

                if sort:
                    sub_order = np.argsort(w.round(9), kind='mergesort')
                    w = w[sub_order]
                    c = c[:, sub_order]

                mo_coeff1[k][:, idx] = mo_coeff1[k][:, idx].dot(c)
                mo_energy[k][idx] = w

    mask = np.ones(nmo, dtype=bool)
    frozen = getattr(mc, 'frozen', None)
    if frozen is not None:
        if isinstance(frozen, (int, np.integer)):
            mask[:frozen] = False
        else:
            mask[frozen] = False

    core_idx = np.where(mask[:ncore])[0]
    vir_idx = np.where(mask[nocc:])[0] + nocc
    _diag_subfock_(core_idx)
    _diag_subfock_(vir_idx)

    if log.verbose >= logger.DEBUG:
        for k in range(nkpts):
            log.debug('k-point %d', k)
            for i in range(nmo):
                log.debug('i = %d  <i|F|i> = %12.8f',
                          i + 1, mo_energy[k][i].real)

    return mo_coeff1, ci, mo_energy


class PBCKCASCI(casci.PBCCASCI):
    '''
    Child class for the PBC k-CASCI.
    This class solves one total momentum sector of the active-space CI problem
    using the k-FCI solver.
    '''

    _keys = casci.PBCCASCI._keys.union({'target_k', 'kpts', 'kmesh',
                                         'kconserv'})

    def __init__(self, kmf, ncas=0, nelecas=0, ncore=None, target_k=0):
        casci.PBCCASCI.__init__(self, kmf, ncas=ncas, nelecas=nelecas,
                                ncore=ncore)
        self.target_k = target_k
        self.kpts = kcistrings._safe_getattr(kmf, 'kpts', None)
        self.kmesh = kcistrings._safe_getattr(kmf, 'kmesh', None)
        self.kconserv = None
        self.fcisolver = pbc_fci.ksolver(self.cell, nkpts=self.nkpts,
                                         target_k=target_k)
        self.fcisolver.kpts = self.kpts
        self.fcisolver.kmesh = self.kmesh
        self.fcisolver.lindep = getattr(__config__,
                                        'mcscf_casci_CASCI_fcisolver_lindep',
                                        1e-12)
        self.fcisolver.max_cycle = getattr(__config__,
                                           'mcscf_casci_CASCI_fcisolver_max_cycle',
                                           200)
        self.fcisolver.conv_tol = getattr(__config__,
                                          'mcscf_casci_CASCI_fcisolver_conv_tol',
                                          1e-8)
        self.canonicalization = False

    def dump_flags(self, verbose=None):
        casci.PBCCASCI.dump_flags(self, verbose)
        log = logger.new_logger(self, verbose)
        log.info('target_k = %d', self.target_k)
        return self

    def get_h1cas(self, mo_coeff=None, ncas=None, ncore=None):
        '''An alias of get_h1eff method.'''
        return self.get_h1eff(mo_coeff, ncas, ncore)

    get_h1eff = h1e_for_cas = h1e_for_cas
    get_h2eff = get_h2eff

    def make_rdm1(self, mo_coeff=None, ci=None, ncas=None, nelecas=None,
                  ncore=None, **kwargs):
        '''
        Spin-summed one-particle density matrix in the AO basis at each
        k-point.

        Unlike PBC CASCI, KCASCI already uses a k-orbital active-space basis,
        so no Wannier MO-phase transformation is applied.
        '''
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if ci is None:
            ci = self.ci
        if ncas is None:
            ncas = self.ncas
        if nelecas is None:
            nelecas = self.nelecas
        if ncore is None:
            ncore = self.ncore

        nelecas = _unpack_nelec(nelecas, self.cell.spin)
        nelecastot = (self.nkpts * nelecas[0],
                      self.nkpts * nelecas[1])
        target_k = int(self.target_k) % self.nkpts
        return _make_rdm1_ao(
            self, mo_coeff, ci, ncas, ncore, nelecastot, target_k)

    get_fock = get_fock
    canonicalize = canonicalize

    @lib.with_doc(canonicalize.__doc__)
    def canonicalize_(self, mo_coeff=None, ci=None, eris=None, sort=False,
                      cas_natorb=False, casdm1=None, verbose=None,
                      with_meta_lowdin=casci.WITH_META_LOWDIN):
        self.mo_coeff, ci, self.mo_energy = \
            canonicalize(self, mo_coeff, ci, eris, sort, cas_natorb,
                         casdm1, verbose, with_meta_lowdin)
        if cas_natorb:
            self.ci = ci
        return self.mo_coeff, ci, self.mo_energy

    def _finalize(self):
        log = logger.Logger(self.stdout, self.verbose)
        nkpts = self.nkpts
        if (log.verbose >= logger.NOTE and
                getattr(self.fcisolver, 'spin_square', None)):
            if isinstance(self.e_cas, (np.complex128, np.float64)):
                try:
                    ss = self.fcisolver.spin_square(
                        self.ci, nkpts * self.ncas,
                        (nkpts * self.nelecas[0],
                         nkpts * self.nelecas[1]))
                    log.note(
                        'KCASCI E (per k-point)= %#.15g  '
                        'E(CI) = %#.15g  S^2 = %.7f',
                        self.e_tot.real, self.e_cas.real, ss[0])
                except NotImplementedError:
                    log.note(
                        'KCASCI E (per k-point) = %#.15g  E(CI) = %#.15g',
                        self.e_tot.real, self.e_cas.real)
            else:
                for i, e in enumerate(self.e_cas):
                    try:
                        nelecastot = (
                            nkpts * self.nelecas[0],
                            nkpts * self.nelecas[1])
                        ss = self.fcisolver.spin_square(
                            self.ci[i], nkpts * self.ncas, nelecastot)
                        log.note(
                            'KCASCI E (per k-point) state %3d  '
                            'E = %#.15g  E(CI) = %#.15g  S^2 = %.7f',
                            i, self.e_tot[i].real, e.real, ss[0])
                    except NotImplementedError:
                        log.note(
                            'KCASCI E (per k-point) state %3d  '
                            'E = %#.15g  E(CI) = %#.15g',
                            i, self.e_tot[i].real, e.real)
        else:
            if isinstance(self.e_cas, (np.complex128, np.float64)):
                log.note(
                    'KCASCI E (per k-point)= %#.15g  E(CI) = %#.15g',
                    self.e_tot.real, self.e_cas.real)
            else:
                for i, e in enumerate(self.e_cas):
                    log.note(
                        'KCASCI E (per k-point) state %3d  '
                        'E = %#.15g  E(CI) = %#.15g',
                        i, self.e_tot[i].real, e.real)
        return self

    def kernel(self, mo_coeff=None, ci0=None, verbose=None):
        '''
        args:
            mo_coeff:
            ci0:
            verbose:
        returns:
            e_tot:
            e_cas:
            ci:
            mo_coeff:
            mo_energy:
        '''
        if mo_coeff is None:
            mo_coeff = self.mo_coeff

        self.mo_coeff = mo_coeff

        if ci0 is None:
            ci0 = self.ci

        log = logger.new_logger(self, verbose)

        self.check_sanity()
        self.dump_flags(log)

        self.e_tot, self.e_cas, self.ci = kernel(self, mo_coeff=mo_coeff,
                                                 ci0=ci0, verbose=verbose)

        if self.canonicalization:
            self.canonicalize_(mo_coeff, self.ci,
                               sort=self.sorting_mo_energy,
                               cas_natorb=self.natorb, verbose=log)

        if self.natorb:
            raise NotImplementedError

        if getattr(self.fcisolver, 'converged', None) is not None:
            self.converged = np.all(self.fcisolver.converged)
            if self.converged: log.info('KCASCI converged')
            else: log.info('KCASCI not converged')
        else:
            self.converged = True

        self._finalize()

        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy


class ChargedPBCKCASCI(PBCKCASCI):
    '''
    k-CASCI driver for N+/-1 active-space sectors over one or all target
    momentum sectors.
    '''

    _keys = PBCKCASCI._keys.union({
        'charge', 'charged_spin', 'charged_nelecas', 'charged_nelecastot',
        'charged_results',
    })

    def __init__(self, kmf, ncas=0, nelecas=0, ncore=None, charge=1,
                 target_k=None, charged_spin=None):
        solver_target_k = 0 if target_k is None else target_k
        PBCKCASCI.__init__(self, kmf, ncas=ncas, nelecas=nelecas,
                           ncore=ncore, target_k=solver_target_k)
        self.target_k = target_k
        self.charge = int(charge)
        self.charged_spin = charged_spin
        self.charged_nelecas = None
        self.charged_nelecastot = None
        self.charged_results = []
        self.canonicalization = False

    def dump_flags(self, verbose=None):
        casci.PBCCASCI.dump_flags(self, verbose)
        log = logger.new_logger(self, verbose)
        if self.target_k is None:
            target_k = 'all'
        else:
            target_k = str(int(self.target_k) % self.nkpts)
        spin = 'default' if self.charged_spin is None else self.charged_spin
        log.info('target_k = %s', target_k)
        log.info('charge = %d', self.charge)
        log.info('charged_spin = %s', spin)
        return self

    def _target_ks(self, target_k=None):
        if target_k is None:
            target_k = self.target_k
        if target_k is None:
            return list(range(self.nkpts))
        return [int(target_k) % self.nkpts]

    def make_rdm1(self, mo_coeff=None, ci=None, ncas=None, nelecas=None,
                  ncore=None, target_k=None, **kwargs):
        '''
        Spin-summed one-particle density matrix in the AO basis for one
        charged total-momentum sector.

        target_k selects a result in charged_results when CI is not supplied.
        nelecas is the total charged active-electron tuple.
        '''
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if ncas is None:
            ncas = self.ncas
        if ncore is None:
            ncore = self.ncore

        if target_k is None:
            if self.target_k is not None:
                target_k = self.target_k
            elif len(self.charged_results) == 1:
                target_k = self.charged_results[0]['target_k']
            else:
                raise ValueError(
                    'target_k is required when multiple charged KCASCI '
                    'sectors are available')
        target_k = int(target_k) % self.nkpts

        result = next(
            (result for result in self.charged_results
             if int(result['target_k']) % self.nkpts == target_k),
            None)
        if ci is None:
            if result is None:
                raise ValueError(
                    f'No charged KCASCI result is available for '
                    f'target_k={target_k}')
            ci = result['ci']
        if nelecas is None:
            if result is not None:
                nelecas = result['nelecastot']
            else:
                nelecas = self.charged_nelecastot
        if nelecas is None:
            raise ValueError('The charged active-electron count is not set')
        nelecas = tuple(nelecas)

        return _make_rdm1_ao(
            self, mo_coeff, ci, ncas, ncore, nelecas, target_k)

    get_fock = get_fock

    def _finalize(self):
        log = logger.Logger(self.stdout, self.verbose)
        spin_square = getattr(self.fcisolver, 'spin_square', None)
        with_spin = log.verbose >= logger.NOTE and spin_square is not None
        ncas = self.nkpts * self.ncas
        nelecas = self.charged_nelecastot

        for result in self.charged_results:
            target_k = result['target_k']
            e_tot = np.asarray(result['e_tot'])
            e_cas = np.asarray(result['e_cas'])
            ci = result['ci']
            scalar_energy = e_cas.ndim == 0
            e_tot = np.atleast_1d(e_tot)
            e_cas = np.atleast_1d(e_cas)

            for root, e in enumerate(e_cas):
                if scalar_energy:
                    msg = ('charged KCASCI E (per k-point) target k %3d'
                           ' = %#.15g  E(CI) = %#.15g')
                    args = (target_k, e_tot[root].real, e.real)
                    ci_root = ci
                else:
                    msg = ('charged KCASCI E (per k-point) target k %3d'
                           ' state %3d  E = %#.15g  E(CI) = %#.15g')
                    args = (target_k, root, e_tot[root].real, e.real)
                    ci_root = ci[root]

                if with_spin:
                    try:
                        ss = spin_square(
                            ci_root, ncas, nelecas, nkpts=self.nkpts,
                            target_k=target_k)
                        log.note(msg + '  S^2 = %.7f', *args, ss[0])
                        continue
                    except NotImplementedError:
                        pass
                log.note(msg, *args)
        return self

    def kernel(self, mo_coeff=None, ci0=None, verbose=None, target_k=None,
               charge=None, charged_spin=None):
        '''
        Run the charged active-space CI.  If target_k is None and self.target_k
        is None, all total momentum sectors are solved successively.
        '''
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        self.mo_coeff = mo_coeff

        if ci0 is None:
            if self.charged_results:
                ci0 = {r['target_k']: r['ci'] for r in self.charged_results}
            else:
                ci0 = self.ci

        log = logger.new_logger(self, verbose)
        self.check_sanity()
        charge = self.charge if charge is None else int(charge)
        charged_spin = self.charged_spin if charged_spin is None else charged_spin
        self.charge = int(charge)
        self.charged_spin = charged_spin
        self.dump_flags(log)

        results, e_tot_all, e_cas_all, ci_all, nelecastot, converged = \
            kernel_chrkcasci(self, mo_coeff=mo_coeff, ci0=ci0,
                           verbose=verbose, target_k=target_k,
                           charge=charge, charged_spin=charged_spin)
        self.charged_nelecas = nelecastot
        self.charged_nelecastot = nelecastot

        self.charged_results = results
        if len(results) == 1:
            self.e_tot = e_tot_all[0]
            self.e_cas = e_cas_all[0]
            self.ci = ci_all[0]
        else:
            self.e_tot = np.asarray(e_tot_all)
            self.e_cas = np.asarray(e_cas_all)
            self.ci = ci_all

        self.converged = bool(np.all(converged))
        if self.converged: log.info('charged KCASCI converged')
        else: log.info('charged KCASCI not converged')

        self._finalize()

        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    def band_energies(self, reference_energy, root=0, kpts=None,
                      per_cell=False, reference_target_k=None):
        return compute_band_energies(self, reference_energy, root=root, 
                                     kpts=kpts, per_cell=per_cell,
                                     reference_target_k=reference_target_k)

    get_band_energy = band_energies
    band_energy = band_energies


KCASCI = PBCKCASCI
ChargedKCASCI = ChargedPBCKCASCI
