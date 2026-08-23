"""Helpers for k-point and momentum-resolved periodic MC-PDFT."""

import numpy as np

from pyscf.mcpdft._dms import _get_fcisolver

from mrh.my_pyscf.pbc.mcpdft._dms import dm2_cumulant_complex


def _select_charged_kcas_result(mc, target_k=None):
    """Select one stored charged KCASCI momentum-sector result."""
    nkpts = int(mc.nkpts)
    if nkpts <= 0:
        raise ValueError("nkpts must be positive")

    results = list(getattr(mc, "charged_results", ()))
    if target_k is None:
        target_k = getattr(mc, "target_k", None)
    if target_k is None:
        if len(results) != 1:
            raise ValueError(
                "target_k is required when multiple charged KCASCI "
                "sectors are available",
            )
        target_k = results[0]["target_k"]
    if not isinstance(target_k, (int, np.integer)):
        raise ValueError("target_k must be an integer")
    target_k = int(target_k) % nkpts

    matches = [
        result for result in results
        if int(result["target_k"]) % nkpts == target_k
    ]
    if not matches:
        raise ValueError(
            f"No charged KCASCI result is available for target_k={target_k}",
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple charged KCASCI results are available for "
            f"target_k={target_k}",
        )
    return matches[0]


def _get_charged_kcas_rdm_context(mc, ci=None, state=0, target_k=None):
    """Resolve the CI and electron sector for one charged KCASCI result."""
    result = _select_charged_kcas_result(mc, target_k=target_k)
    nkpts = int(mc.nkpts)
    target_k = int(result["target_k"]) % nkpts

    if ci is None:
        ci = result.get("ci")
    if ci is None:
        raise ValueError(
            f"The charged KCASCI result for target_k={target_k} has no CI "
            "vector",
        )
    fcisolver, ci, _ = _get_fcisolver(mc, ci, state=state)

    nelecastot = result.get("nelecastot", result.get("nelecas"))
    if nelecastot is None:
        nelecastot = getattr(mc, "charged_nelecastot", None)
    try:
        nelecastot = tuple(int(value) for value in nelecastot)
    except (TypeError, ValueError):
        nelecastot = ()
    if len(nelecastot) != 2:
        raise ValueError(
            "charged nelecastot must contain alpha and beta counts",
        )

    ncastot = nkpts * int(mc.ncas)
    if any(value < 0 or value > ncastot for value in nelecastot):
        raise ValueError(
            f"charged nelecastot {nelecastot} is invalid for "
            f"ncastot={ncastot}",
        )
    rdm_kwargs = {"nkpts": nkpts, "target_k": target_k}
    return fcisolver, ci, ncastot, nelecastot, rdm_kwargs


def _get_kcas_rdm_context(mc, ci, state=0):
    """Resolve one kCASCI state and its momentum-sector RDM arguments."""
    nkpts = int(mc.nkpts)
    if nkpts <= 0:
        raise ValueError("nkpts must be positive")

    target_k = mc.target_k
    if not isinstance(target_k, (int, np.integer)):
        raise ValueError("target_k must be an integer")
    target_k = int(target_k) % nkpts

    fcisolver, ci, nelecas = _get_fcisolver(mc, ci, state=state)
    if len(nelecas) != 2:
        raise ValueError("nelecas must contain alpha and beta counts")

    ncastot = nkpts * int(mc.ncas)
    nelecastot = tuple(nkpts * int(value) for value in nelecas)
    rdm_kwargs = {"nkpts": nkpts, "target_k": target_k}
    return fcisolver, ci, ncastot, nelecastot, rdm_kwargs


def make_one_casdm1s_kcas(mc, ci, state=0):
    """Build one state's spin-separated kCASCI active-space 1-RDM.

    The returned matrices use the flattened Bloch-orbital ordering
    ``(k * ncas + orbital)``.  They must not be transformed from a Wannier
    basis before use by the periodic MC-PDFT machinery.
    """
    fcisolver, ci, ncastot, nelecastot, rdm_kwargs = \
        _get_kcas_rdm_context(mc, ci, state=state)
    casdm1s = np.asarray(fcisolver.make_rdm1s(
        ci, ncastot, nelecastot, **rdm_kwargs,
    ))
    expected_shape = (2, ncastot, ncastot)
    if casdm1s.shape != expected_shape:
        raise ValueError(
            f"Expected spin-separated kCASCI 1-RDM shape {expected_shape}, "
            f"got {casdm1s.shape}",
        )
    return casdm1s


def make_one_casdm1s_charged_kcas(mc, ci=None, state=0, target_k=None):
    """Build a charged KCASCI state's spin-separated active-space 1-RDM."""
    fcisolver, ci, ncastot, nelecastot, rdm_kwargs = \
        _get_charged_kcas_rdm_context(
            mc, ci=ci, state=state, target_k=target_k,
        )
    casdm1s = np.asarray(fcisolver.make_rdm1s(
        ci, ncastot, nelecastot, **rdm_kwargs,
    ))
    expected_shape = (2, ncastot, ncastot)
    if casdm1s.shape != expected_shape:
        raise ValueError(
            f"Expected spin-separated charged KCASCI 1-RDM shape "
            f"{expected_shape}, got {casdm1s.shape}",
        )
    return casdm1s


def make_one_casdm2_kcas(mc, ci, state=0):
    """Build one state's spin-summed kCASCI active-space 2-RDM."""
    fcisolver, ci, ncastot, nelecastot, rdm_kwargs = \
        _get_kcas_rdm_context(mc, ci, state=state)
    try:
        _, casdm2 = fcisolver.make_rdm12(
            ci, ncastot, nelecastot, **rdm_kwargs,
        )
    except AttributeError:
        casdm2 = fcisolver.make_rdm2(
            ci, ncastot, nelecastot, **rdm_kwargs,
        )

    casdm2 = np.asarray(casdm2)
    expected_shape = (ncastot,) * 4
    if casdm2.shape != expected_shape:
        raise ValueError(
            f"Expected kCASCI 2-RDM shape {expected_shape}, "
            f"got {casdm2.shape}",
        )
    return casdm2


def make_one_casdm2_charged_kcas(mc, ci=None, state=0, target_k=None):
    """Build a charged KCASCI state's spin-summed active-space 2-RDM."""
    fcisolver, ci, ncastot, nelecastot, rdm_kwargs = \
        _get_charged_kcas_rdm_context(
            mc, ci=ci, state=state, target_k=target_k,
        )
    try:
        _, casdm2 = fcisolver.make_rdm12(
            ci, ncastot, nelecastot, **rdm_kwargs,
        )
    except AttributeError:
        casdm2 = fcisolver.make_rdm2(
            ci, ncastot, nelecastot, **rdm_kwargs,
        )

    casdm2 = np.asarray(casdm2)
    expected_shape = (ncastot,) * 4
    if casdm2.shape != expected_shape:
        raise ValueError(
            f"Expected charged KCASCI 2-RDM shape {expected_shape}, "
            f"got {casdm2.shape}",
        )
    return casdm2


def _validate_kspace_layout(nkpts, ncas, kconserv=None):
    """Validate dimensions shared by the k-space RDM converters."""
    nkpts = int(nkpts)
    ncas = int(ncas)
    if nkpts <= 0:
        raise ValueError("nkpts must be positive")
    if ncas <= 0:
        raise ValueError("ncas must be positive")

    if kconserv is not None:
        kconserv = np.asarray(kconserv)
        expected_shape = (nkpts, nkpts, nkpts)
        if kconserv.shape != expected_shape:
            raise ValueError(
                f"Expected kconserv shape {expected_shape}, "
                f"got {kconserv.shape}",
            )
        if not np.issubdtype(kconserv.dtype, np.integer):
            raise ValueError("kconserv must contain integer indices")
        if np.any(kconserv < 0) or np.any(kconserv >= nkpts):
            raise ValueError("kconserv indices must lie in [0, nkpts)")
    return nkpts, ncas, kconserv


def _check_forbidden_norm(total_norm_sq, forbidden_norm_sq, momentum_tol,
                          tensor_name):
    """Reject density-matrix weight outside momentum-conserving blocks."""
    if momentum_tol is None:
        return
    if momentum_tol < 0:
        raise ValueError("momentum_tol must be nonnegative or None")

    total_norm = np.sqrt(max(0.0, float(total_norm_sq)))
    forbidden_norm = np.sqrt(max(0.0, float(forbidden_norm_sq)))
    if forbidden_norm > momentum_tol * max(1.0, total_norm):
        raise ValueError(
            f"{tensor_name} contains momentum-forbidden blocks with norm "
            f"{forbidden_norm:.3e}",
        )


def casdm1s_to_kpts(casdm1s, nkpts, ncas, momentum_tol=1e-8):
    """Extract the k-diagonal blocks of a flattened Bloch-basis 1-RDM."""
    nkpts, ncas, _ = _validate_kspace_layout(nkpts, ncas)
    ncastot = nkpts * ncas
    casdm1s = np.asarray(casdm1s)
    expected_shape = (2, ncastot, ncastot)
    if casdm1s.shape != expected_shape:
        raise ValueError(
            f"Expected spin-separated kCASCI 1-RDM shape {expected_shape}, "
            f"got {casdm1s.shape}",
        )

    casdm1s_full = casdm1s.reshape(
        2, nkpts, ncas, nkpts, ncas,
    )
    casdm1s_kpts = np.stack([
        casdm1s_full[:, k, :, k, :] for k in range(nkpts)
    ], axis=1)

    if momentum_tol is not None:
        total_norm_sq = np.vdot(casdm1s, casdm1s).real
        forbidden_norm_sq = 0.0
        for k1 in range(nkpts):
            for k2 in range(nkpts):
                if k1 != k2:
                    block = casdm1s_full[:, k1, :, k2, :]
                    forbidden_norm_sq += np.vdot(block, block).real
        _check_forbidden_norm(
            total_norm_sq, forbidden_norm_sq, momentum_tol,
            "kCASCI 1-RDM",
        )
    return casdm1s_kpts


def cascm2_to_kpts(cascm2, nkpts, ncas, kconserv, momentum_tol=1e-8):
    """Extract momentum-conserving blocks of a Bloch-basis cumulant."""
    nkpts, ncas, kconserv = _validate_kspace_layout(
        nkpts, ncas, kconserv=kconserv,
    )
    ncastot = nkpts * ncas
    cascm2 = np.asarray(cascm2)
    expected_shape = (ncastot,) * 4
    if cascm2.shape != expected_shape:
        raise ValueError(
            f"Expected kCASCI cumulant shape {expected_shape}, "
            f"got {cascm2.shape}",
        )

    cascm2_full = cascm2.reshape(
        nkpts, ncas, nkpts, ncas, nkpts, ncas, nkpts, ncas,
    )
    cascm2_kpts = np.empty(
        (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas),
        dtype=cascm2.dtype,
    )
    for k1 in range(nkpts):
        for k2 in range(nkpts):
            for k3 in range(nkpts):
                k4 = kconserv[k1, k2, k3]
                cascm2_kpts[k1, k2, k3] = cascm2_full[
                    k1, :, k2, :, k3, :, k4, :,
                ]

    if momentum_tol is not None:
        total_norm_sq = np.vdot(cascm2, cascm2).real
        forbidden_norm_sq = 0.0
        for k1 in range(nkpts):
            for k2 in range(nkpts):
                for k3 in range(nkpts):
                    allowed_k4 = kconserv[k1, k2, k3]
                    for k4 in range(nkpts):
                        if k4 != allowed_k4:
                            block = cascm2_full[
                                k1, :, k2, :, k3, :, k4, :,
                            ]
                            forbidden_norm_sq += np.vdot(block, block).real
        _check_forbidden_norm(
            total_norm_sq, forbidden_norm_sq, momentum_tol,
            "kCASCI cumulant",
        )
    return cascm2_kpts


def make_kcas_rdms_kpts(casdm1s, casdm2, nkpts, ncas, kconserv,
                         momentum_tol=1e-8):
    """Convert dense kCASCI RDMs to the blocks used by periodic MC-PDFT."""
    casdm1s_kpts = casdm1s_to_kpts(
        casdm1s, nkpts, ncas, momentum_tol=momentum_tol,
    )
    cascm2 = dm2_cumulant_complex(casdm2, casdm1s)
    cascm2_kpts = cascm2_to_kpts(
        cascm2, nkpts, ncas, kconserv,
        momentum_tol=momentum_tol,
    )
    return casdm1s_kpts, cascm2_kpts


def casdm1s_kpts_to_dm1s(obj, casdm1s_kpts, mo_coeff, ncore):
    """Transform k-resolved active 1-RDMs to spin-separated AO matrices."""
    from pyscf.mcpdft import _dms

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
    expected_shape = (2, nkpts, ncas, ncas)
    if casdm1s_kpts.shape != expected_shape:
        raise ValueError(
            f"Expected casdm1s_kpts shape {expected_shape}, "
            f"got {casdm1s_kpts.shape}",
        )
    if ncore < 0 or ncore + ncas > mo_coeff.shape[2]:
        raise ValueError("ncore and ncas are incompatible with mo_coeff")

    dm1s_kpts = []
    for k in range(nkpts):
        dm1s = _dms.casdm1s_to_dm1s(
            obj, casdm1s_kpts[:, k], mo_coeff=mo_coeff[k],
            ncore=ncore, ncas=ncas,
        )
        dm1s_kpts.append(np.asarray(dm1s))
    return np.stack(dm1s_kpts, axis=1)
