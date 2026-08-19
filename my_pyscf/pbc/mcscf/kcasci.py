#!/usr/bin/env python

import numpy as np

from pyscf import lib
from pyscf.lib import logger

from mrh.my_pyscf.pbc.fci.addons import _unpack_nelec
from mrh.my_pyscf.pbc.fci import kcistrings
from mrh.my_pyscf.pbc.mcscf import casci


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


def _get_kmom_for_kcasci(mc):
    """Build momentum-arithmetic tables from the KCASCI k-point metadata."""
    kmf = mc._scf
    kpts = kcistrings._safe_getattr(mc, "kpts", None)
    if kpts is None:
        kpts = kcistrings._safe_getattr(kmf, "kpts", None)
    kmesh = kcistrings._safe_getattr(mc, "kmesh", None)
    if kmesh is None:
        kmesh = kcistrings._safe_getattr(kmf, "kmesh", None)
    kconserv = kcistrings._safe_getattr(mc, "kconserv", None)
    return kcistrings.make_kpoint_momentum(
        mc.nkpts, cell=mc.cell, kpts=kpts, kmesh=kmesh,
        kconserv=kconserv, kmf=kmf, kmc=mc,
    )


def _set_solver_kpts(mc, kmom=None):
    """Pass KCASCI k-point metadata to its k-FCI solver."""
    if kmom is None:
        kmom = _get_kmom_for_kcasci(mc)

    mc.kconserv = kmom.kconserv
    mc.fcisolver.kpts = kcistrings._safe_getattr(
        mc, "kpts", kcistrings._safe_getattr(mc._scf, "kpts", None),
    )
    mc.fcisolver.kmesh = kcistrings._safe_getattr(
        mc, "kmesh", kcistrings._safe_getattr(mc._scf, "kmesh", None),
    )
    mc.fcisolver.kconserv = kmom.kconserv
    mc.fcisolver.kmom = kmom
    return kmom


def kernel(mc, mo_coeff=None, ci0=None, verbose=logger.NOTE, envs=None):
    """Run neutral KCASCI in one total-momentum sector.

    The k-FCI problem represents the supercell associated with the k-point
    mesh.  Its total and active-space energies are divided by ``nkpts`` before
    they are returned so that KCASCI follows the periodic per-cell energy
    convention.
    """
    del envs
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci0 is None:
        ci0 = mc.ci

    log = logger.new_logger(mc, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug("Start KCASCI")

    nkpts = mc.nkpts
    ncas = mc.ncas
    nelecas = _unpack_nelec(mc.nelecas, mc.cell.spin)

    h1eff, energy_core = mc.get_h1eff(mo_coeff)
    t1 = log.timer("one-electron integral computation for k-CAS", *t0)
    h2eff = mc.get_h2eff(mo_coeff)
    t1 = log.timer("integral transformation to k-CAS space", *t1)
    h1eff = _adjust_h1eff_for_kfci(h1eff, h2eff)
    log.debug("core energy = %.15g", energy_core.real)

    assert h1eff.shape == (nkpts, ncas, ncas)
    assert h2eff.shape == (
        nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas,
    )

    kmom = _set_solver_kpts(mc)
    if not isinstance(mc.target_k, (int, np.integer)):
        raise ValueError("target_k must be an integer")
    target_k = int(mc.target_k)
    if not 0 <= target_k < nkpts:
        target_k %= nkpts
        log.warn("target_k is out of bounds, using %d instead", target_k)

    ncastot = nkpts * ncas
    nelecastot = (nkpts * nelecas[0], nkpts * nelecas[1])
    max_memory = max(4000, mc.max_memory - lib.current_memory()[0])

    mc.fcisolver.nkpts = nkpts
    mc.fcisolver.target_k = target_k
    mc.fcisolver.kmom = kmom
    e_tot, fcivec = mc.fcisolver.kernel(
        h1eff, h2eff, ncastot, nelecastot, ci0=ci0, nkpts=nkpts,
        target_k=target_k, verbose=log, max_memory=max_memory,
        ecore=energy_core,
    )
    log.timer("k-FCI solver", *t1)

    e_cas = e_tot - energy_core
    e_cas /= nkpts
    e_tot /= nkpts
    return e_tot, e_cas, fcivec


def make_casdm1(mc, ci=None, stav_dm1=False, weights=None, target_k=None):
    """Build the k-basis active-space one-particle density matrix.

    For multiple roots, the first-root density is returned by default.  Set
    ``stav_dm1`` or supply state-average weights to combine root densities.
    A PySCF state-average solver continues to use its own weights unless
    explicit weights are supplied here.
    """
    from pyscf.mcscf import addons

    if ci is None:
        ci = mc.ci

    nkpts = mc.nkpts
    ncas = mc.ncas
    nelecas = _unpack_nelec(mc.nelecas, mc.cell.spin)
    nelecastot = (nkpts * nelecas[0], nkpts * nelecas[1])
    if target_k is None:
        target_k = mc.target_k
    if target_k is None:
        raise ValueError("target_k is required to build a KCASCI 1-RDM")
    target_k = int(target_k) % nkpts
    rdm_kwargs = {"nkpts": nkpts, "target_k": target_k}

    is_multiroot = isinstance(ci, (list, tuple, casci.RANGE_TYPE))
    is_state_average = isinstance(mc.fcisolver, addons.StateAverageFCISolver)
    if is_multiroot:
        if weights is None and is_state_average:
            return mc.fcisolver.make_rdm1(
                ci, nkpts * ncas, nelecastot, **rdm_kwargs,
            )
        if weights is None and not stav_dm1:
            return mc.fcisolver.make_rdm1(
                ci[0], nkpts * ncas, nelecastot, **rdm_kwargs,
            )

        if weights is None:
            weights = np.ones(len(ci), dtype=float) / len(ci)
        else:
            weights = np.asarray(weights, dtype=float)
            if weights.ndim != 1 or weights.size != len(ci):
                raise ValueError(
                    "weights must contain one value for each CI root",
                )
            if not np.all(np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError("weights must be finite and nonnegative")
            weight_sum = weights.sum()
            if weight_sum <= 0:
                raise ValueError("at least one state-average weight is needed")
            weights = weights / weight_sum

        if is_state_average:
            dm1_states = mc.fcisolver.states_make_rdm1(
                ci, nkpts * ncas, nelecastot, **rdm_kwargs,
            )
        else:
            dm1_states = [
                mc.fcisolver.make_rdm1(
                    ci_root, nkpts * ncas, nelecastot, **rdm_kwargs,
                )
                for ci_root in ci
            ]
        return sum(
            weight * dm1
            for weight, dm1 in zip(weights, dm1_states)
        )

    if weights is not None:
        raise ValueError("weights require multiple CI roots")
    return mc.fcisolver.make_rdm1(
        ci, nkpts * ncas, nelecastot, **rdm_kwargs,
    )


def make_rdm1(mc, mo_coeff=None, ci=None, ncas=None, nelecas=None,
              ncore=None, target_k=None):
    """Transform a neutral k-FCI 1-RDM to the AO basis at each k-point."""
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci is None:
        ci = mc.ci
    if ncas is None:
        ncas = mc.ncas
    if nelecas is None:
        nelecas = mc.nelecas
    if ncore is None:
        ncore = mc.ncore
    if target_k is None:
        target_k = mc.target_k
    if target_k is None:
        raise ValueError("target_k is required to build a KCASCI 1-RDM")

    mo_coeff = np.asarray(mo_coeff)
    nkpts = mc.nkpts
    ncastot = nkpts * ncas
    nelecas = _unpack_nelec(nelecas, mc.cell.spin)
    nelecastot = (nkpts * nelecas[0], nkpts * nelecas[1])
    casdm1 = mc.fcisolver.make_rdm1(
        ci, ncastot, nelecastot, nkpts=nkpts,
        target_k=int(target_k) % nkpts,
    )
    casdm1 = np.asarray(casdm1)
    if casdm1.shape != (ncastot, ncastot):
        raise ValueError(
            f"Expected an active-space 1-RDM with shape "
            f"{(ncastot, ncastot)}, got {casdm1.shape}",
        )

    nao = mo_coeff.shape[1]
    dtype = np.result_type(mo_coeff.dtype, casdm1.dtype)
    dm1 = np.empty((nkpts, nao, nao), dtype=dtype)
    for k in range(nkpts):
        mocore = mo_coeff[k, :, :ncore]
        mocas = mo_coeff[k, :, ncore:ncore + ncas]
        p0 = k * ncas
        p1 = p0 + ncas
        dm1[k] = 2.0 * mocore @ mocore.conj().T
        dm1[k] += mocas @ casdm1[p0:p1, p0:p1] @ mocas.conj().T
    return dm1
