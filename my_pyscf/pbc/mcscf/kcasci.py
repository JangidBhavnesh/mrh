# !/usr/bin/env python

import numpy as np

from pyscf import lib, __config__
from pyscf.lib import logger

from mrh.my_pyscf.pbc.fci.addons import _unpack_nelec
from mrh.my_pyscf.pbc import fci as pbc_fci
from mrh.my_pyscf.pbc.mcscf import casci

# 
# k-space CASCI driver for the k-FCI solver.
# Author: Bhavnesh Jangid <jangidbhavnesh@uchicago.edu>
#

def get_h1e_h2e(mc, mo_coeff=None):
    '''
    Compute the k-space active-space effective Hamiltonian for the k-FCI solver.
    Args:
        mc : pbc.mcscf.KCASCI
            The k-CASCI object.
        mo_coeff : np.ndarray [nk, nao, nmo_k]
            orbitals at each k-point.
    Returns:
        h1eff : np.ndarray [nk, ncas, ncas]
            The effective one-electron Hamiltonian in k-space.
        h2eff : np.ndarray [nk, nk, nk, ncas, ncas, ncas, ncas]
            The effective two-electron Hamiltonian in k-space.
        ecore : np.complex128
            The core energy.
    '''

    if mo_coeff is None:
        mo_coeff = mc.mo_coeff

    cell = mc.cell
    kmf = mc._scf
    nkpts = mc.nkpts
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas

    hcore = mc.get_hcore()
    dtype = np.result_type(hcore, *[mo.dtype for mo in mo_coeff])
    hcore = hcore.astype(dtype)

    mo_core = [mo[:, :ncore] for mo in mo_coeff]
    mo_cas = np.asarray([mo[:, ncore:nocc] for mo in mo_coeff], dtype=dtype)

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

    # k-space two-electron integrals with supercell normalization.
    h2eff = kmf.with_df.ao2mo_7d(mo_cas, kpts=kmf.kpts)
    h2eff = np.asarray(h2eff, dtype=dtype) / nkpts

    # k-FCI contract_2e follows the same lower-level convention as
    # direct_spin1_cplx.contract_2e. The Hamiltonian needs h1 - 0.5*J and 0.5*h2.
    j_eff = np.zeros_like(h1eff)
    for kp in range(nkpts):
        for kq in range(nkpts):
            j_eff[kp] += np.einsum('piis->ps', h2eff[kp, kq, kq])

    h1eff -= 0.5 * j_eff
    h2eff *= 0.5

    return h1eff, h2eff, ecore

def kernel(mc, mo_coeff=None, ci0=None, verbose=logger.NOTE, envs=None):
    '''
    # Passing env to be consistent with molecular CASCI, but currently this is
    # not used.
    '''
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci0 is None:
        ci0 = mc.ci

    log = logger.new_logger(mc, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Start k-CASCI')

    nkpts = mc.nkpts
    ncas = mc.ncas
    nelecas = _unpack_nelec(mc.nelecas, mc._scf.cell.spin)

    h1eff, h2eff, energy_core = mc.get_h1e_h2e(mo_coeff)
    t1 = log.timer('integral transformation to k-CAS space', *t0)
    log.debug('core energy = %.15g', energy_core.real)

    assert h1eff.shape == (nkpts, ncas, ncas)
    assert h2eff.shape == (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)

    if log.verbose >= logger.DEBUG1:
        for k in range(nkpts):
            assert np.linalg.norm(h1eff[k] - h1eff[k].conj().T) < 1e-10, \
                "1e Hamiltonian hermiticity error"
        eri_symm_err = 0.0
        for kp, kq, kr in np.ndindex(nkpts, nkpts, nkpts):
            ks = (kp - kq + kr) % nkpts
            err = np.linalg.norm(h2eff[kp, kq, kr] -
                                 h2eff[kr, ks, kp].transpose(2, 3, 0, 1))
            eri_symm_err = max(eri_symm_err, err)
        assert eri_symm_err < 1e-10, "ERI permutation symmetry error"

    max_memory = max(4000, mc.max_memory - lib.current_memory()[0])
    ncastot = nkpts * ncas
    nelecastot = (nkpts * nelecas[0], nkpts * nelecas[1])
    target_k = int(mc.target_k) % nkpts

    mc.fcisolver.nkpts = nkpts
    mc.fcisolver.target_k = target_k

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

class PBCKCASCI(casci.PBCCASCI):
    '''
    Child class for the PBC k-CASCI.
    This class solves one total momentum sector of the active-space CI problem
    using the k-FCI solver.
    '''

    _keys = casci.PBCCASCI._keys.union({'target_k'})

    def __init__(self, kmf, ncas=0, nelecas=0, ncore=None, target_k=0):
        casci.PBCCASCI.__init__(self, kmf, ncas=ncas, nelecas=nelecas,
                                ncore=ncore)
        self.target_k = target_k
        self.fcisolver = pbc_fci.ksolver(self.cell, nkpts=self.nkpts,
                                         target_k=target_k)
        self.fcisolver.lindep = getattr(__config__,
                                        'mcscf_casci_CASCI_fcisolver_lindep',
                                        1e-12)
        self.fcisolver.max_cycle = getattr(__config__,
                                           'mcscf_casci_CASCI_fcisolver_max_cycle',
                                           200)
        self.fcisolver.conv_tol = getattr(__config__,
                                          'mcscf_casci_CASCI_fcisolver_conv_tol',
                                          1e-8)

    def dump_flags(self, verbose=None):
        casci.PBCCASCI.dump_flags(self, verbose)
        log = logger.new_logger(self, verbose)
        log.info('target_k = %d', self.target_k)
        return self

    def get_h1e_h2e(self, mo_coeff=None):
        return get_h1e_h2e(self, mo_coeff=mo_coeff)

    def get_h1eff(self, mo_coeff=None, ncas=None, ncore=None):
        '''
        Return the effective h1e for the k-FCI solver.
        '''
        h1eff, h2eff, ecore = self.get_h1e_h2e(mo_coeff=mo_coeff)
        return h1eff, ecore

    h1e_for_cas = get_h1eff
    get_h1cas = get_h1eff

    def get_h2eff(self, mo_coeff=None):
        '''
        Return the effective h2e for the k-FCI solver.
        '''
        h1eff, h2eff, ecore = self.get_h1e_h2e(mo_coeff=mo_coeff)
        return h2eff

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

KCASCI = PBCKCASCI

