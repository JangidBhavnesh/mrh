import numpy as np

from pyscf.mcpdft import _dms

"""Reduced-density-matrix utilities for periodic MC-PDFT.

The kCASCI solvers store active-space RDMs in flattened Bloch-orbital order,
``k * ncas + orbital``.  This module builds those RDMs, extracts their
momentum-conserving blocks, transforms active 1-RDMs to AO densities, and
forms complex-orbital cumulants.  Charged-sector selection belongs to the
k-MC-PDFT driver; charged helpers here operate on an already selected result.
"""

def dm2_cumulant_complex(dm2, dm1s):
    """Build the two-body cumulant for a complex orbital basis."""
    dm1s = np.asarray(dm1s)
    return _dms.dm2_cumulant(dm2, dm1s.swapaxes(-1, -2))


def _get_kcas_rdm_context(mc, ci, state=0):
    """Resolve one neutral kCASCI state and its RDM arguments."""
    nkpts = int(mc.nkpts)
    if nkpts <= 0:
        raise ValueError("nkpts must be positive")
    if not isinstance(mc.target_k, (int, np.integer)):
        raise ValueError("target_k must be an integer")

    target_k = int(mc.target_k) % nkpts
    fcisolver, ci, nelecas = _dms._get_fcisolver(mc, ci, state=state)
    if len(nelecas) != 2:
        raise ValueError("nelecas must contain alpha and beta counts")

    ncastot = nkpts * int(mc.ncas)
    nelecastot = tuple(nkpts * int(value) for value in nelecas)
    kwargs = {"nkpts": nkpts, "target_k": target_k}
    return fcisolver, ci, ncastot, nelecastot, kwargs


def _get_charged_kcas_rdm_context(mc, result, ci=None, state=0):
    """Resolve RDM arguments for an already selected charged kCAS result."""
    nkpts = int(mc.nkpts)
    target_k = int(result["target_k"]) % nkpts
    if ci is None:
        ci = result.get("ci")
    if ci is None:
        raise ValueError(
            f"The charged KCASCI result for target_k={target_k} has no CI "
            "vector",
        )
    fcisolver, ci, _ = _dms._get_fcisolver(mc, ci, state=state)

    nelecastot = result.get("nelecastot", result.get("nelecas"))
    if nelecastot is None:
        nelecastot = getattr(mc, "charged_nelecastot", None)
    try:
        nelecastot = tuple(int(value) for value in nelecastot)
    except (TypeError, ValueError):
        nelecastot = ()
    if len(nelecastot) != 2:
        raise ValueError("charged nelecastot must contain alpha and beta counts")

    ncastot = nkpts * int(mc.ncas)
    if any(value < 0 or value > ncastot for value in nelecastot):
        raise ValueError(
            f"charged nelecastot {nelecastot} is invalid for "
            f"ncastot={ncastot}",
        )
    kwargs = {"nkpts": nkpts, "target_k": target_k}
    return fcisolver, ci, ncastot, nelecastot, kwargs


def _make_casdm1s(context, label):
    fcisolver, ci, ncastot, nelecastot, kwargs = context
    casdm1s = np.asarray(
        fcisolver.make_rdm1s(ci, ncastot, nelecastot, **kwargs),
    )
    expected = (2, ncastot, ncastot)
    if casdm1s.shape != expected:
        raise ValueError(
            f"Expected spin-separated {label} 1-RDM shape {expected}, "
            f"got {casdm1s.shape}",
        )
    return casdm1s


def _make_casdm2(context, label):
    fcisolver, ci, ncastot, nelecastot, kwargs = context
    try:
        _, casdm2 = fcisolver.make_rdm12(
            ci, ncastot, nelecastot, **kwargs,
        )
    except AttributeError:
        casdm2 = fcisolver.make_rdm2(ci, ncastot, nelecastot, **kwargs)
    casdm2 = np.asarray(casdm2)
    expected = (ncastot,) * 4
    if casdm2.shape != expected:
        raise ValueError(
            f"Expected {label} 2-RDM shape {expected}, got {casdm2.shape}",
        )
    return casdm2


def make_one_casdm1s_kcas(mc, ci, state=0):
    """Build one neutral kCASCI state's spin-separated active 1-RDM."""
    return _make_casdm1s(
        _get_kcas_rdm_context(mc, ci, state=state), "kCASCI",
    )


def make_one_casdm2_kcas(mc, ci, state=0):
    """Build one neutral kCASCI state's spin-summed active 2-RDM."""
    return _make_casdm2(
        _get_kcas_rdm_context(mc, ci, state=state), "kCASCI",
    )


def make_one_casdm1s_charged_kcas(mc, result, ci=None, state=0):
    """Build a selected charged kCASCI state's spin-separated 1-RDM."""
    return _make_casdm1s(
        _get_charged_kcas_rdm_context(mc, result, ci=ci, state=state),
        "charged KCASCI",
    )


def make_one_casdm2_charged_kcas(mc, result, ci=None, state=0):
    """Build a selected charged kCASCI state's spin-summed 2-RDM."""
    return _make_casdm2(
        _get_charged_kcas_rdm_context(mc, result, ci=ci, state=state),
        "charged KCASCI",
    )


def _validate_kspace_layout(nkpts, ncas, kconserv=None):
    nkpts, ncas = int(nkpts), int(ncas)
    if nkpts <= 0:
        raise ValueError("nkpts must be positive")
    if ncas <= 0:
        raise ValueError("ncas must be positive")
    if kconserv is not None:
        kconserv = np.asarray(kconserv)
        if kconserv.shape != (nkpts, nkpts, nkpts):
            raise ValueError(
                f"Expected kconserv shape {(nkpts, nkpts, nkpts)}, "
                f"got {kconserv.shape}",
            )
        if not np.issubdtype(kconserv.dtype, np.integer):
            raise ValueError("kconserv must contain integer indices")
        if np.any(kconserv < 0) or np.any(kconserv >= nkpts):
            raise ValueError("kconserv indices must lie in [0, nkpts)")
    return nkpts, ncas, kconserv


def _check_forbidden_norm(total_sq, forbidden_sq, tolerance, label):
    if tolerance is None:
        return
    if tolerance < 0:
        raise ValueError("momentum_tol must be nonnegative or None")
    total = np.sqrt(max(0.0, float(total_sq)))
    forbidden = np.sqrt(max(0.0, float(forbidden_sq)))
    if forbidden > tolerance * max(1.0, total):
        raise ValueError(
            f"{label} contains momentum-forbidden blocks with norm "
            f"{forbidden:.3e}",
        )


def casdm1s_to_kpts(casdm1s, nkpts, ncas, momentum_tol=1e-8):
    """Extract k-diagonal blocks from a flattened Bloch-basis 1-RDM."""
    nkpts, ncas, _ = _validate_kspace_layout(nkpts, ncas)
    ncastot = nkpts * ncas
    casdm1s = np.asarray(casdm1s)
    expected = (2, ncastot, ncastot)
    if casdm1s.shape != expected:
        raise ValueError(
            f"Expected spin-separated kCASCI 1-RDM shape {expected}, "
            f"got {casdm1s.shape}",
        )

    full = casdm1s.reshape(2, nkpts, ncas, nkpts, ncas)
    blocks = np.stack([full[:, k, :, k, :] for k in range(nkpts)], axis=1)
    forbidden_sq = sum(
        np.vdot(full[:, k1, :, k2, :], full[:, k1, :, k2, :]).real
        for k1 in range(nkpts) for k2 in range(nkpts) if k1 != k2
    )
    _check_forbidden_norm(
        np.vdot(casdm1s, casdm1s).real, forbidden_sq,
        momentum_tol, "kCASCI 1-RDM",
    )
    return blocks


def cascm2_to_kpts(cascm2, nkpts, ncas, kconserv, momentum_tol=1e-8):
    """Extract momentum-conserving blocks from a Bloch-basis cumulant."""
    nkpts, ncas, kconserv = _validate_kspace_layout(
        nkpts, ncas, kconserv,
    )
    ncastot = nkpts * ncas
    cascm2 = np.asarray(cascm2)
    expected = (ncastot,) * 4
    if cascm2.shape != expected:
        raise ValueError(
            f"Expected kCASCI cumulant shape {expected}, got {cascm2.shape}",
        )

    full = cascm2.reshape(
        nkpts, ncas, nkpts, ncas, nkpts, ncas, nkpts, ncas,
    )
    blocks = np.empty(
        (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas),
        dtype=cascm2.dtype,
    )
    forbidden_sq = 0.0
    for k1 in range(nkpts):
        for k2 in range(nkpts):
            for k3 in range(nkpts):
                allowed = int(kconserv[k1, k2, k3])
                blocks[k1, k2, k3] = full[
                    k1, :, k2, :, k3, :, allowed, :,
                ]
                for k4 in range(nkpts):
                    if k4 != allowed:
                        block = full[k1, :, k2, :, k3, :, k4, :]
                        forbidden_sq += np.vdot(block, block).real
    _check_forbidden_norm(
        np.vdot(cascm2, cascm2).real, forbidden_sq,
        momentum_tol, "kCASCI cumulant",
    )
    return blocks


def make_kcas_rdms_kpts(casdm1s, casdm2, nkpts, ncas, kconserv,
                         momentum_tol=1e-8):
    """Convert dense kCASCI RDMs into periodic momentum blocks."""
    dm1s = casdm1s_to_kpts(
        casdm1s, nkpts, ncas, momentum_tol=momentum_tol,
    )
    cumulant = dm2_cumulant_complex(casdm2, casdm1s)
    cm2 = cascm2_to_kpts(
        cumulant, nkpts, ncas, kconserv, momentum_tol=momentum_tol,
    )
    return dm1s, cm2


def casdm1s_kpts_to_dm1s(obj, casdm1s_kpts, mo_coeff, ncore):
    """Transform k-resolved active 1-RDMs to spin-separated AO matrices."""
    mo_coeff = np.asarray(mo_coeff)
    casdm1s_kpts = np.asarray(casdm1s_kpts)
    if mo_coeff.ndim != 3:
        raise ValueError("mo_coeff must have shape (nkpts, nao, nmo)")

    nkpts = mo_coeff.shape[0]
    if casdm1s_kpts.ndim != 4 or casdm1s_kpts.shape[:2] != (2, nkpts):
        raise ValueError(
            "casdm1s_kpts must have shape (2, nkpts, ncas, ncas)",
        )
    ncas = casdm1s_kpts.shape[2]
    expected = (2, nkpts, ncas, ncas)
    if casdm1s_kpts.shape != expected:
        raise ValueError(
            f"Expected casdm1s_kpts shape {expected}, "
            f"got {casdm1s_kpts.shape}",
        )
    if ncore < 0 or ncore + ncas > mo_coeff.shape[2]:
        raise ValueError("ncore and ncas are incompatible with mo_coeff")

    dm1s = [
        _dms.casdm1s_to_dm1s(
            obj, casdm1s_kpts[:, k], mo_coeff=mo_coeff[k],
            ncore=ncore, ncas=ncas,
        )
        for k in range(nkpts)
    ]
    return np.stack(dm1s, axis=1)
