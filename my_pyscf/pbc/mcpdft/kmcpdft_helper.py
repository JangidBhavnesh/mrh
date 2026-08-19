"""Helpers for k-point and momentum-resolved periodic MC-PDFT."""

import numpy as np

from pyscf.mcpdft._dms import _get_fcisolver


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


def make_one_casdm2_kcas(mc, ci, state=0):
    """Build one state's spin-summed kCASCI active-space 2-RDM."""
    fcisolver, ci, ncastot, nelecastot, rdm_kwargs = \
        _get_kcas_rdm_context(mc, ci, state=state)
    try:
        casdm2 = fcisolver.make_rdm2(
            ci, ncastot, nelecastot, **rdm_kwargs,
        )
    except AttributeError:
        _, casdm2 = fcisolver.make_rdm12(
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
