#!/usr/bin/env python

import numpy as np


# Author: Bhavnesh Jangid

"""Momentum-resolved CASCI for periodic systems."""


def h1e_for_cas(mc, mo_coeff=None, ncas=None, ncore=None):
    """Build the k-space one-electron active-space Hamiltonian.

    Returns the active-space Hamiltonian at each k-point and the core energy
    for the corresponding supercell problem.  The KCASCI kernel divides the
    final energy by the number of k-points to recover the energy per cell.

    Args:
        mc: Periodic KCASCI object.
        mo_coeff: Molecular orbitals with shape ``(nkpts, nao, nmo)``.
        ncas: Number of active orbitals at each k-point.
        ncore: Number of core orbitals at each k-point.

    Returns:
        h1eff: Array with shape ``(nkpts, ncas, ncas)``.
        ecore: Core energy for the supercell Hamiltonian.
    """
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ncas is None:
        ncas = mc.ncas
    if ncore is None:
        ncore = mc.ncore

    mo_coeff = np.asarray(mo_coeff)
    dtype = mo_coeff.dtype
    nkpts = mc.nkpts
    nocc = ncore + ncas

    hcore = np.asarray(mc.get_hcore(), dtype=dtype)
    mo_core = mo_coeff[:, :, :ncore]
    mo_cas = mo_coeff[:, :, ncore:nocc]

    ecore = mc.energy_nuc() * nkpts
    if ncore:
        dm_core = np.asarray([
            2.0 * mo_core[k] @ mo_core[k].conj().T
            for k in range(nkpts)
        ], dtype=dtype)
        corevhf = mc.get_veff(
            mc.cell, dm_core, hermi=1, kpts=mc._scf.kpts,
        )
        ecore += np.einsum(
            "kij,kji->", dm_core, hcore + 0.5 * corevhf,
            optimize=True,
        )
        hcore = hcore + corevhf

    h1eff = np.asarray([
        mo_cas[k].conj().T @ hcore[k] @ mo_cas[k]
        for k in range(nkpts)
    ], dtype=dtype)
    return h1eff, ecore


def get_h2eff(mc, mo_coeff=None):
    """Build the k-space two-electron active-space Hamiltonian.

    The integrals include the supercell normalization and the factor of one
    half expected by the separate two-electron contraction in k-FCI.

    Args:
        mc: Periodic KCASCI object.
        mo_coeff: Molecular orbitals with shape ``(nkpts, nao, nmo)``.

    Returns:
        Array with shape
        ``(nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)``.
    """
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff

    mo_coeff = np.asarray(mo_coeff)
    dtype = mo_coeff.dtype
    nkpts = mc.nkpts
    ncore = mc.ncore
    ncas = mc.ncas
    mo_cas = mo_coeff[:, :, ncore:ncore + ncas]

    h2eff = mc._scf.with_df.ao2mo_7d(mo_cas, kpts=mc._scf.kpts)
    h2eff = np.asarray(h2eff, dtype=dtype) / nkpts
    return h2eff * 0.5


def _adjust_h1eff_for_kfci(h1eff, h2eff):
    """Apply the one-body correction for the k-FCI ``0.5 * h2`` convention."""
    nkpts = h1eff.shape[0]
    j_eff = np.zeros_like(h1eff)
    for kp in range(nkpts):
        for kq in range(nkpts):
            j_eff[kp] += np.einsum("piis->ps", h2eff[kp, kq, kq])
    return h1eff - j_eff
