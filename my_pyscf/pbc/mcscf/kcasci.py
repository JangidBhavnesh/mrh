#!/usr/bin/env python

import numpy as np

from pyscf import __config__, lib
from pyscf.lib import logger

from mrh.my_pyscf.pbc import fci as pbc_fci
from mrh.my_pyscf.pbc.fci.addons import _unpack_nelec
from mrh.my_pyscf.pbc.fci import kcistrings
from mrh.my_pyscf.pbc.mcscf import casci
from mrh.my_pyscf.pbc.mcscf.casci import (
    get_h2eff_kpts,
    h1e_kpts_for_cas as h1e_for_cas,
)

MAX_MEMORY = getattr(__config__, "MAX_MEMORY", 4000)


# Author: Bhavnesh Jangid

"""Momentum-resolved CASCI (k-CASCI) for periodic systems."""


def get_h2eff(mc, mo_coeff=None):
    """Build the normalized k-space ERIs with the k-FCI factor of one half."""
    dtype = np.asarray(mc.mo_coeff if mo_coeff is None else mo_coeff,).dtype
    return np.asarray(get_h2eff_kpts(mc, mo_coeff), dtype=dtype,) * 0.5


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
    if kpts is None: kpts = kcistrings._safe_getattr(kmf, "kpts", None)
    kmesh = kcistrings._safe_getattr(mc, "kmesh", None)
    if kmesh is None: kmesh = kcistrings._safe_getattr(kmf, "kmesh", None)
    kconserv = kcistrings._safe_getattr(mc, "kconserv", None)
    return kcistrings.make_kpoint_momentum(
        mc.nkpts, cell=mc.cell, kpts=kpts, kmesh=kmesh,
        kconserv=kconserv, kmf=kmf, kmc=mc,
    )


def _set_solver_kpts(mc, kmom=None):
    """Pass KCASCI k-point metadata to its k-FCI solver."""
    if kmom is None: kmom = _get_kmom_for_kcasci(mc)

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
    """Run neutral KCASCI in one sector and return energies per cell."""
    del envs
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci0 is None: ci0 = mc.ci

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
    assert h2eff.shape == (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)

    kmom = _set_solver_kpts(mc)
    if not isinstance(mc.target_k, (int, np.integer)):
        raise ValueError("target_k must be an integer")
    target_k = int(mc.target_k)
    if not 0 <= target_k < nkpts:
        target_k %= nkpts
        log.warn("target_k is out of bounds, using %d instead", target_k)

    ncastot = nkpts * ncas
    nelecastot = (nkpts * nelecas[0], nkpts * nelecas[1])
    max_memory = max(MAX_MEMORY, mc.max_memory - lib.current_memory()[0])

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


def _get_nelecas_for_charged_kcasci(ncas, nkpts, nelecas, cell_spin,
                                     charge=0, spin=None):
    """Return the charged total ``(N_alpha, N_beta)`` active-space sector."""
    if not isinstance(charge, (int, np.integer)):
        raise ValueError("charge must be an integer")
    charge = int(charge)
    if charge not in (-1, 0, 1):
        raise ValueError("only neutral and single-electron charges are supported")

    nelecas = _unpack_nelec(nelecas, cell_spin)
    nelec = nkpts * (nelecas[0] + nelecas[1]) - charge
    norb = nkpts * ncas
    if spin is None: spin = nelec % 2
    if not isinstance(spin, (int, np.integer)):
        raise ValueError("spin must be an integer")
    spin = int(spin)

    if nelec < 0 or nelec > 2 * norb:
        raise ValueError(
            f"charge={charge} gives {nelec} active electrons for "
            f"{norb} active orbitals",
        )
    if (nelec + spin) % 2:
        raise ValueError(
            f"active electron count {nelec} and spin {spin} have "
            "inconsistent parity",
        )

    neleca = (nelec + spin) // 2
    nelecb = nelec - neleca
    if not (0 <= neleca <= norb and 0 <= nelecb <= norb):
        raise ValueError(
            f"charge={charge}, spin={spin} gives invalid active electrons "
            f"({neleca}, {nelecb})",
        )
    return int(neleca), int(nelecb)


def _target_ks_for_charged_kcasci(mc, target_k=None):
    """Return the normalized total-momentum sectors for a charged solve."""
    if target_k is None: target_k = mc.target_k
    if target_k is None: return list(range(mc.nkpts))
    if not isinstance(target_k, (int, np.integer)):
        raise ValueError("target_k must be an integer or None")
    return [int(target_k) % mc.nkpts]


def kernel_chrkcasci(mc, mo_coeff=None, ci0=None, verbose=logger.NOTE,
                     target_k=None, charge=None, charged_spin=None,
                     envs=None):
    """Run charged KCASCI in one or all total-momentum sectors."""
    del envs
    if mo_coeff is None: mo_coeff = mc.mo_coeff

    if charge is None: charge = getattr(mc, "charge", None)
    if charge is None:
        raise ValueError("charge is required for charged KCASCI")
    if not isinstance(charge, (int, np.integer)):
        raise ValueError("charge must be an integer")
    charge = int(charge)
    if charge == 0:
        raise ValueError("charged KCASCI requires a nonzero charge")
    if charged_spin is None: charged_spin = getattr(mc, "charged_spin", None)

    log = logger.new_logger(mc, verbose)
    t0 = (logger.process_clock(), logger.perf_counter())
    log.debug("Start charged KCASCI")

    nkpts = mc.nkpts
    ncas = mc.ncas
    h1eff, energy_core = mc.get_h1eff(mo_coeff)
    t1 = log.timer("one-electron integral computation for charged k-CAS", *t0)
    h2eff = mc.get_h2eff(mo_coeff)
    t1 = log.timer("integral transformation to charged k-CAS space", *t1)
    h1eff = _adjust_h1eff_for_kfci(h1eff, h2eff)
    log.debug("core energy = %.15g", energy_core.real)

    assert h1eff.shape == (nkpts, ncas, ncas)
    assert h2eff.shape == (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)

    kmom = _set_solver_kpts(mc)
    ncastot = nkpts * ncas
    nelecastot = _get_nelecas_for_charged_kcasci(
        ncas, nkpts, mc.nelecas, mc.cell.spin,
        charge=charge, spin=charged_spin,
    )
    target_ks = _target_ks_for_charged_kcasci(mc, target_k=target_k)
    if len(target_ks) > 1 and ci0 is not None and not isinstance(ci0, dict):
        raise ValueError(
            "ci0 for a charged target_k sweep must be a dict keyed by "
            "target_k",
        )

    max_memory = max(MAX_MEMORY, mc.max_memory - lib.current_memory()[0])
    results = []
    e_tot_all = []
    e_cas_all = []
    ci_all = []
    converged = []
    for sector in target_ks:
        sector = int(sector) % nkpts
        ci0_sector = ci0.get(sector) if isinstance(ci0, dict) else ci0
        log.info(
            "Solving charged KCASCI target_k = %d, nelec = %s",
            sector, nelecastot,
        )

        mc.fcisolver.nkpts = nkpts
        mc.fcisolver.target_k = sector
        mc.fcisolver.kmom = kmom
        e_tot_supercell, fcivec = mc.fcisolver.kernel(
            h1eff, h2eff, ncastot, nelecastot, ci0=ci0_sector,
            nkpts=nkpts, target_k=sector, verbose=log,
            max_memory=max_memory, ecore=energy_core,
        )
        t1 = log.timer(f"charged k-FCI solver target_k = {sector}", *t1)
        e_cas_supercell = e_tot_supercell - energy_core
        e_tot = e_tot_supercell / nkpts
        e_cas = e_cas_supercell / nkpts
        sector_converged = bool(np.all(getattr(mc.fcisolver, "converged", True)))

        results.append({
            "target_k": sector,
            "charge": charge,
            "ncas": ncas,
            "ncastot": ncastot,
            "nelecas": nelecastot,
            "nelecastot": nelecastot,
            "nkpts": nkpts,
            "e_tot": e_tot,
            "e_cas": e_cas,
            "e_tot_supercell": e_tot_supercell,
            "e_cas_supercell": e_cas_supercell,
            "ci": fcivec,
            "converged": sector_converged,
        })
        e_tot_all.append(e_tot)
        e_cas_all.append(e_cas)
        ci_all.append(fcivec)
        converged.append(sector_converged)

    return results, e_tot_all, e_cas_all, ci_all, nelecastot, converged


def compute_band_energies(charged_results, reference_energy, charge=None,
                          root=None, kpts=None, nkpts=None, per_cell=False,
                          reference_target_k=None, kmom=None, cell=None,
                          kconserv=None):
    """Convert charged total energies to momentum-labeled particle/hole poles.

    With neutral momentum ``K0``, ``k_hole = K0 - K(N-1)`` and
    ``k_particle = K(N+1) - K0``.  Results are supercell energy differences
    unless ``per_cell=True``.
    """
    if hasattr(charged_results, "charged_results"):
        mc = charged_results
        if charge is None: charge = mc.charge
        if nkpts is None: nkpts = mc.nkpts
        if kpts is None: kpts = kcistrings._safe_getattr(mc._scf, "kpts", None)
        if cell is None: cell = mc.cell
        if kconserv is None: kconserv = kcistrings._safe_getattr(mc, "kconserv", None)
        if kmom is None: kmom = _get_kmom_for_kcasci(mc)
        charged_results = mc.charged_results

    charged_results = list(charged_results)
    if not charged_results: return []
    if charge is None: charge = charged_results[0].get("charge")
    if not isinstance(charge, (int, np.integer)):
        raise ValueError("charge must be an integer")
    charge = int(charge)
    if charge not in (-1, 1):
        raise ValueError("charged band energies require charge +1 or -1")

    kind = "hole" if charge > 0 else "particle"
    momentum_key = f"{kind}_momentum"
    bands = []
    for result in charged_results:
        result_charge = result.get("charge", charge)
        if int(result_charge) != charge:
            raise ValueError("charged results contain inconsistent charges")

        target_k = int(result["target_k"])
        result_nkpts = result.get("nkpts", nkpts)
        if result_nkpts is None:
            raise ValueError(
                "nkpts is required when charged results do not store it",
            )
        result_nkpts = int(result_nkpts)
        scale = 1 if per_cell else result_nkpts

        result_kmom = kmom
        if result_kmom is None or result_kmom.nkpts != result_nkpts:
            result_kmom = kcistrings.make_kpoint_momentum(
                result_nkpts, cell=cell, kpts=kpts,
                kconserv=kconserv,
            )
        if reference_target_k is None:
            reference_k = int(result_kmom.zero)
        else:
            if not isinstance(reference_target_k, (int, np.integer)):
                raise ValueError("reference_target_k must be an integer")
            reference_k = int(reference_target_k) % result_nkpts

        if kind == "hole":
            band_k = int(result_kmom.ksub[reference_k, target_k])
        else:
            band_k = int(result_kmom.ksub[target_k, reference_k])

        momentum = band_k
        if kpts is not None: momentum = np.asarray(kpts[band_k]).copy()
        e_charged = result["e_tot"]
        if root is not None:
            e_charged = e_charged[root]
        if kind == "hole":
            energy = scale * (reference_energy - e_charged)
        else:
            energy = scale * (e_charged - reference_energy)

        bands.append({
            "target_k": target_k,
            "momentum_index": band_k,
            momentum_key: momentum,
            "energy": energy,
            "root": root,
            "charge": charge,
            "kind": kind,
        })
    return bands


def make_casdm1(mc, ci=None, stav_dm1=False, weights=None, target_k=None):
    """Build a single-root or state-averaged active-space 1-RDM."""
    from pyscf.mcscf import addons

    if ci is None: ci = mc.ci

    nkpts = mc.nkpts
    ncas = mc.ncas
    nelecas = _unpack_nelec(mc.nelecas, mc.cell.spin)
    nelecastot = (nkpts * nelecas[0], nkpts * nelecas[1])
    if target_k is None: target_k = mc.target_k
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
        return sum(weight * dm1 for weight, dm1 in zip(weights, dm1_states))

    if weights is not None:
        raise ValueError("weights require multiple CI roots")
    return mc.fcisolver.make_rdm1(
        ci, nkpts * ncas, nelecastot, **rdm_kwargs,
    )


def make_rdm1(mc, mo_coeff=None, ci=None, ncas=None, nelecas=None,
              ncore=None, target_k=None, nelecastot=None):
    """Transform a neutral or charged k-FCI 1-RDM to k-point AO blocks."""
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if ncas is None: ncas = mc.ncas
    if nelecas is None and nelecastot is None: nelecas = mc.nelecas
    if ncore is None: ncore = mc.ncore
    if target_k is None: target_k = mc.target_k
    if target_k is None:
        raise ValueError("target_k is required to build a KCASCI 1-RDM")

    mo_coeff = np.asarray(mo_coeff)
    nkpts = mc.nkpts
    ncastot = nkpts * ncas
    if nelecastot is None:
        nelecas = _unpack_nelec(nelecas, mc.cell.spin)
        nelecastot = (nkpts * nelecas[0], nkpts * nelecas[1])
    else:
        if len(nelecastot) != 2:
            raise ValueError("nelecastot must contain alpha and beta counts")
        nelecastot = tuple(int(value) for value in nelecastot)
    casdm1 = mc.fcisolver.make_rdm1(
        ci, ncastot, nelecastot, nkpts=nkpts,
        target_k=int(target_k) % nkpts,
    )
    casdm1 = np.asarray(casdm1)
    if casdm1.shape != (ncastot, ncastot):
        raise ValueError(f"Expected an active-space 1-RDM with shape "
                         f"{(ncastot, ncastot)}, got {casdm1.shape}")

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


def get_fock(mc, mo_coeff=None, ci=None, eris=None, casdm1=None,
             verbose=None, target_k=None, stav_dm1=False, weights=None):
    """Construct the generalized KCASCI Fock matrix in the AO basis."""
    del eris, verbose
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if casdm1 is None:
        casdm1 = make_casdm1(
            mc, ci, stav_dm1=stav_dm1, weights=weights,
            target_k=target_k,
        )

    mo_coeff = np.asarray(mo_coeff)
    nkpts = mc.nkpts
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    ncastot = nkpts * ncas
    casdm1 = np.asarray(casdm1)
    dtype = np.result_type(mo_coeff.dtype, casdm1.dtype)
    casdm1 = np.asarray(casdm1, dtype=dtype)
    if casdm1.shape != (ncastot, ncastot):
        raise ValueError(f"Expected casdm1 shape {(ncastot, ncastot)}, "
                         f"got {casdm1.shape}")

    mo_core = mo_coeff[:, :, :ncore]
    dm_k = np.asarray([
        2.0 * mo_core[k] @ mo_core[k].conj().T
        for k in range(nkpts)
    ], dtype=dtype)
    for k in range(nkpts):
        mocas = mo_coeff[k, :, ncore:nocc]
        p0 = k * ncas
        p1 = p0 + ncas
        dm_k[k] += mocas @ casdm1[p0:p1, p0:p1] @ mocas.conj().T

    hcore = np.asarray(mc.get_hcore(), dtype=dtype)
    veff = np.asarray(
        mc.get_veff(mc.cell, dm_k, hermi=1, kpts=mc._scf.kpts), dtype=dtype,
    )
    return hcore + veff


@lib.with_doc(casci.canonicalize.__doc__)
def canonicalize(mc, mo_coeff=None, ci=None, eris=None, sort=False,
                 cas_natorb=False, casdm1=None, verbose=logger.NOTE,
                 with_meta_lowdin=casci.WITH_META_LOWDIN, stav_dm1=False,
                 weights=None, target_k=None):
    """Canonicalize the unfrozen KCASCI core and virtual orbitals."""
    del eris, with_meta_lowdin
    log = logger.new_logger(mc, verbose)
    log.debug("Canonicalizing KCASCI orbitals")

    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if cas_natorb:
        raise NotImplementedError("KCASCI natural orbitals are not implemented")
    if casdm1 is None:
        casdm1 = make_casdm1(
            mc, ci, stav_dm1=stav_dm1, weights=weights,
            target_k=target_k,
        )

    mo_coeff = np.asarray(mo_coeff)
    casdm1 = np.asarray(casdm1)
    nkpts = mc.nkpts
    ncas = mc.ncas
    ncore = mc.ncore
    nocc = ncore + ncas
    nmo = mo_coeff.shape[2]

    fock_ao = get_fock(
        mc, mo_coeff=mo_coeff, ci=ci, casdm1=casdm1,
        target_k=target_k,
    )
    mo_coeff1 = mo_coeff.copy()

    log.info("Density matrix diagonal elements")
    for k in range(nkpts):
        p0 = k * ncas
        p1 = p0 + ncas
        dm_k = casdm1[p0:p1, p0:p1]
        log.info(
            "k-point %d, only real diagonal = %s", k,
            np.array2string(
                np.diag(dm_k).real, precision=5, floatmode="fixed",
                separator=", ",
            ),
        )

    mo_energy = [
        np.einsum(
            "pi,pi->i", mo_coeff1[k].conj(), fock_ao[k] @ mo_coeff1[k],
        )
        for k in range(nkpts)
    ]
    orbsym_extra = np.zeros(nmo, dtype=int)

    def _diag_subfock_(idx):
        if idx.size > 1:
            for k in range(nkpts):
                coeff = mo_coeff1[k][:, idx]
                fock = coeff.conj().T @ fock_ao[k] @ coeff
                energy, rotation = mc._eig(fock, None, None, orbsym_extra[idx])
                if sort:
                    order = np.argsort(energy.round(9), kind="mergesort")
                    energy = energy[order]
                    rotation = rotation[:, order]
                mo_coeff1[k][:, idx] = coeff @ rotation
                mo_energy[k][idx] = energy

    mask = np.ones(nmo, dtype=bool)
    frozen = getattr(mc, "frozen", None)
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
            log.debug("k-point %d", k)
            for i in range(nmo):
                log.debug(
                    "i = %d  <i|F|i> = %12.8f",
                    i + 1, mo_energy[k][i].real,
                )

    return mo_coeff1, ci, mo_energy


class PBCKCASCI(casci.PBCCASCI):
    """Periodic CASCI driver restricted to one total-momentum sector."""

    _keys = casci.PBCCASCI._keys.union({
        "target_k", "kpts", "kmesh", "kconserv",
    })

    def __init__(self, kmf, ncas=0, nelecas=0, ncore=None, target_k=0):
        super().__init__(kmf, ncas=ncas, nelecas=nelecas, ncore=ncore)
        self.target_k = target_k
        self.kpts = kcistrings._safe_getattr(kmf, "kpts", None)
        self.kmesh = kcistrings._safe_getattr(kmf, "kmesh", None)
        self.kconserv = None
        self.fcisolver = pbc_fci.ksolver(
            self.cell, nkpts=self.nkpts, target_k=target_k,
            kpts=self.kpts, kmesh=self.kmesh,
        )
        self.fcisolver.lindep = getattr(
            __config__, "mcscf_casci_CASCI_fcisolver_lindep", 1e-12,
        )
        self.fcisolver.max_cycle = getattr(
            __config__, "mcscf_casci_CASCI_fcisolver_max_cycle", 200,
        )
        self.fcisolver.conv_tol = getattr(
            __config__, "mcscf_casci_CASCI_fcisolver_conv_tol", 1e-8,
        )
        self.canonicalization = False

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        logger.new_logger(self, verbose).info("target_k = %s", self.target_k)
        return self

    def get_h1cas(self, mo_coeff=None, ncas=None, ncore=None):
        """Alias for :meth:`get_h1eff`."""
        return self.get_h1eff(mo_coeff, ncas, ncore)

    get_h1eff = h1e_for_cas = h1e_for_cas
    get_h2eff = get_h2eff

    def make_rdm1(self, mo_coeff=None, ci=None, ncas=None, nelecas=None,
                  ncore=None, **kwargs):
        """Return the spin-summed AO 1-RDM at each k-point."""
        target_k = kwargs.pop("target_k", self.target_k)
        return make_rdm1(
            self, mo_coeff=mo_coeff, ci=ci, ncas=ncas,
            nelecas=nelecas, ncore=ncore, target_k=target_k,
        )

    get_fock = get_fock
    canonicalize = canonicalize

    @lib.with_doc(canonicalize.__doc__)
    def canonicalize_(self, mo_coeff=None, ci=None, eris=None, sort=False,
                      cas_natorb=False, casdm1=None, verbose=None,
                      with_meta_lowdin=casci.WITH_META_LOWDIN,
                      stav_dm1=False, weights=None, target_k=None):
        self.mo_coeff, ci, self.mo_energy = canonicalize(
            self, mo_coeff=mo_coeff, ci=ci, eris=eris, sort=sort,
            cas_natorb=cas_natorb, casdm1=casdm1, verbose=verbose,
            with_meta_lowdin=with_meta_lowdin, stav_dm1=stav_dm1,
            weights=weights, target_k=target_k,
        )
        return self.mo_coeff, ci, self.mo_energy

    def _finalize(self):
        log = logger.Logger(self.stdout, self.verbose)
        ncastot = self.nkpts * self.ncas
        nelecastot = (self.nkpts * self.nelecas[0],
                      self.nkpts * self.nelecas[1])
        with_spin = (
            log.verbose >= logger.NOTE
            and getattr(self.fcisolver, "spin_square", None) is not None
        )

        scalar_energy = np.ndim(self.e_cas) == 0
        e_tot = np.atleast_1d(self.e_tot)
        e_cas = np.atleast_1d(self.e_cas)
        ci_roots = [self.ci] if scalar_energy else self.ci
        for root, (e_tot_root, e_cas_root, ci_root) in enumerate(
                zip(e_tot, e_cas, ci_roots)):
            if scalar_energy:
                msg = "KCASCI E (per cell) = %#.15g  E(CI) = %#.15g"
                args = (e_tot_root.real, e_cas_root.real)
            else:
                msg = (
                    "KCASCI E (per cell) state %3d  E = %#.15g  "
                    "E(CI) = %#.15g"
                )
                args = (root, e_tot_root.real, e_cas_root.real)

            if with_spin:
                try:
                    ss = self.fcisolver.spin_square(
                        ci_root, ncastot, nelecastot, nkpts=self.nkpts,
                        target_k=int(self.target_k) % self.nkpts,
                    )
                    log.note(msg + "  S^2 = %.7f", *args, ss[0])
                    continue
                except NotImplementedError:
                    pass
            log.note(msg, *args)
        return self

    def kernel(self, mo_coeff=None, ci0=None, verbose=None):
        """Run KCASCI and return energies, CI vectors, and orbitals."""
        if mo_coeff is None: mo_coeff = self.mo_coeff
        self.mo_coeff = mo_coeff
        if ci0 is None: ci0 = self.ci

        log = logger.new_logger(self, verbose)
        self.check_sanity()
        self.dump_flags(log)
        self.e_tot, self.e_cas, self.ci = kernel(
            self, mo_coeff=mo_coeff, ci0=ci0, verbose=verbose,
        )

        if self.canonicalization:
            self.canonicalize_(
                mo_coeff, self.ci, sort=self.sorting_mo_energy,
                cas_natorb=self.natorb, verbose=log,
            )
        if self.natorb:
            raise NotImplementedError("KCASCI natural orbitals are not implemented")

        converged = getattr(self.fcisolver, "converged", None)
        if converged is None:
            self.converged = True
        else:
            self.converged = bool(np.all(converged))
        if self.converged: log.info("KCASCI converged")
        else: log.info("KCASCI not converged")

        self._finalize()
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy


class ChargedPBCKCASCI(PBCKCASCI):
    """KCASCI driver for an N-1 or N+1 k-mesh active-space sector."""

    _keys = PBCKCASCI._keys.union({
        "charge", "charged_spin", "charged_nelecas",
        "charged_nelecastot", "charged_results",
    })

    def __init__(self, kmf, ncas=0, nelecas=0, ncore=None, charge=1,
                 target_k=None, charged_spin=None):
        if not isinstance(charge, (int, np.integer)):
            raise ValueError("charge must be an integer")
        charge = int(charge)
        if charge not in (-1, 1):
            raise ValueError("charged KCASCI requires charge +1 or -1")
        if (charged_spin is not None
                and not isinstance(charged_spin, (int, np.integer))):
            raise ValueError("charged_spin must be an integer or None")

        solver_target_k = 0 if target_k is None else target_k
        super().__init__(kmf, ncas=ncas, nelecas=nelecas, ncore=ncore,
                         target_k=solver_target_k)
        self.target_k = target_k
        self.charge = charge
        self.charged_spin = None if charged_spin is None else int(charged_spin)
        self.charged_nelecas = None
        self.charged_nelecastot = None
        self.charged_results = []
        self.canonicalization = False

    def dump_flags(self, verbose=None):
        casci.PBCCASCI.dump_flags(self, verbose)
        log = logger.new_logger(self, verbose)
        target_k = "all" if self.target_k is None else str(int(self.target_k) % self.nkpts)
        spin = "default" if self.charged_spin is None else str(self.charged_spin)
        log.info("target_k = %s", target_k)
        log.info("charge = %d", self.charge)
        log.info("charged_spin = %s", spin)
        return self

    def _target_ks(self, target_k=None):
        return _target_ks_for_charged_kcasci(self, target_k=target_k)

    def make_rdm1(self, mo_coeff=None, ci=None, ncas=None, nelecas=None,
                  ncore=None, target_k=None, **kwargs):
        """Return the AO 1-RDM for one charged total-momentum sector."""
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ncas is None: ncas = self.ncas
        if ncore is None: ncore = self.ncore

        if target_k is None:
            if self.target_k is not None:
                target_k = self.target_k
            elif len(self.charged_results) == 1:
                target_k = self.charged_results[0]["target_k"]
            else:
                raise ValueError(
                    "target_k is required when multiple charged KCASCI "
                    "sectors are available",
                )
        target_k = int(target_k) % self.nkpts
        result = next((item for item in self.charged_results
                       if int(item["target_k"]) % self.nkpts == target_k), None)

        if ci is None:
            if result is None:
                raise ValueError(f"No charged KCASCI result is available for "
                                 f"target_k={target_k}")
            ci = result["ci"]
        if nelecas is None:
            if result is not None: nelecas = result["nelecastot"]
            else: nelecas = self.charged_nelecastot
        if nelecas is None:
            raise ValueError("the charged active-electron count is not set")

        return make_rdm1(
            self, mo_coeff=mo_coeff, ci=ci, ncas=ncas, ncore=ncore,
            target_k=target_k, nelecastot=tuple(nelecas),
        )

    def get_fock(self, mo_coeff=None, ci=None, eris=None, casdm1=None,
                 verbose=None, target_k=None, stav_dm1=False, weights=None):
        raise NotImplementedError("The Fock matrix is not implemented for charged KCASCI.")

    def canonicalize(self, mo_coeff=None, ci=None, eris=None, sort=False,
                     cas_natorb=False, casdm1=None, verbose=logger.NOTE,
                     with_meta_lowdin=casci.WITH_META_LOWDIN,
                     stav_dm1=False, weights=None, target_k=None):
        raise NotImplementedError("Canonicalization is not implemented for charged KCASCI.")

    canonicalize_ = canonicalize

    def _finalize(self):
        log = logger.Logger(self.stdout, self.verbose)
        ncastot = self.nkpts * self.ncas
        with_spin = (
            log.verbose >= logger.NOTE
            and getattr(self.fcisolver, "spin_square", None) is not None
        )

        for result in self.charged_results:
            target_k = result["target_k"]
            e_tot = np.atleast_1d(result["e_tot"])
            e_cas = np.atleast_1d(result["e_cas"])
            scalar_energy = np.ndim(result["e_cas"]) == 0
            ci_roots = [result["ci"]] if scalar_energy else result["ci"]
            for root, (e_tot_root, e_cas_root, ci_root) in enumerate(
                    zip(e_tot, e_cas, ci_roots)):
                if scalar_energy:
                    msg = (
                        "Charged KCASCI E (per cell) target_k %3d = "
                        "%#.15g  E(CI) = %#.15g"
                    )
                    args = (target_k, e_tot_root.real, e_cas_root.real)
                else:
                    msg = (
                        "Charged KCASCI E (per cell) target_k %3d "
                        "state %3d  E = %#.15g  E(CI) = %#.15g"
                    )
                    args = (target_k, root, e_tot_root.real, e_cas_root.real)

                if with_spin:
                    try:
                        ss = self.fcisolver.spin_square(
                            ci_root, ncastot, self.charged_nelecastot,
                            nkpts=self.nkpts, target_k=target_k,
                        )
                        log.note(msg + "  S^2 = %.7f", *args, ss[0])
                        continue
                    except NotImplementedError:
                        pass
                log.note(msg, *args)
        return self

    def kernel(self, mo_coeff=None, ci0=None, verbose=None, target_k=None,
               charge=None, charged_spin=None):
        """Run charged KCASCI in one or all requested momentum sectors."""
        if mo_coeff is None: mo_coeff = self.mo_coeff
        self.mo_coeff = mo_coeff

        if ci0 is None:
            if self.charged_results:
                ci0 = {
                    result["target_k"]: result["ci"]
                    for result in self.charged_results
                }
            else:
                ci0 = self.ci

        if charge is None: charge = self.charge
        if not isinstance(charge, (int, np.integer)):
            raise ValueError("charge must be an integer")
        charge = int(charge)
        if charge not in (-1, 1):
            raise ValueError("charged KCASCI requires charge +1 or -1")
        if charged_spin is None: charged_spin = self.charged_spin
        if (charged_spin is not None
                and not isinstance(charged_spin, (int, np.integer))):
            raise ValueError("charged_spin must be an integer or None")
        self.charge = charge
        self.charged_spin = None if charged_spin is None else int(charged_spin)

        log = logger.new_logger(self, verbose)
        self.check_sanity()
        self.dump_flags(log)
        output = kernel_chrkcasci(
            self, mo_coeff=mo_coeff, ci0=ci0, verbose=verbose,
            target_k=target_k, charge=self.charge,
            charged_spin=self.charged_spin,
        )
        results, e_tot_all, e_cas_all, ci_all, nelecastot, converged = output

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
        if self.converged: log.info("Charged KCASCI converged")
        else: log.info("Charged KCASCI not converged")
        self._finalize()
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    def band_energies(self, reference_energy, root=None, kpts=None,
                      per_cell=False, reference_target_k=None):
        """Return quasiparticle energies from the stored charged results."""
        return compute_band_energies(
            self, reference_energy, root=root, kpts=kpts,
            per_cell=per_cell,
            reference_target_k=reference_target_k,
        )

    def print_bands(self, reference_energy, root=None, kpts=None,
                    per_cell=False, reference_target_k=None, verbose=None):
        """Print momentum-ordered particle or hole poles."""
        if kpts is None:
            kpts = self.kpts
        bands = self.band_energies(
            reference_energy, root=root, kpts=kpts, per_cell=per_cell,
            reference_target_k=reference_target_k,
        )
        if not bands:
            return bands

        scaled_kpts = self.cell.get_scaled_kpts(kpts)
        bands.sort(key=lambda band: tuple(
            scaled_kpts[band["momentum_index"]],
        ))
        label, pole = {
            "hole": ("Hole (N-1)", "removal"),
            "particle": ("Particle (N+1)", "addition"),
        }[bands[0]["kind"]]
        log = logger.new_logger(self, verbose)
        log.note("")
        log.note(
            "%s: %d active electrons in %d active orbitals", label,
            sum(self.charged_nelecastot), self.nkpts * self.ncas,
        )
        unit = "Eh/cell" if per_cell else "Eh"
        log.note(
            "  k       scaled k-point       target_k   %s pole (%s)",
            pole, unit,
        )
        for band in bands:
            k = band["momentum_index"]
            sk = scaled_kpts[k]
            log.note(
                "%3d  %8.4f %8.4f %8.4f  %9d  %17.8f",
                k, sk[0], sk[1], sk[2], band["target_k"],
                band["energy"].real,
            )
        return bands

    get_band_energy = band_energies
    band_energy = band_energies


KCASCI = PBCKCASCI
ChargedKCASCI = ChargedPBCKCASCI
