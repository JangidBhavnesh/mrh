#!/usr/bin/env python

import numpy as np
from pyscf.pbc.lib import kpts_helper

from mrh.my_pyscf.mcscf.lasscf_sync_o0 import (
    LASSCF_HessianOperator as molLASSCF_HessianOperator,
)
from mrh.my_pyscf.pbc.fci import cplx_csf_helper
from mrh.my_pyscf.pbc.mcscf.klas_ao2mo import _ERIS
from mrh.my_pyscf.pbc.mcscf.klasci import (
    PBCLASCINoSymm,
    PBCLASCITransSymm,
)
from mrh.my_pyscf.pbc.mcscf.mc1step import _get_casdm2_kpts
from mrh.my_pyscf.pbc.util.wannier import get_wannier_orbs

# Author: Bhavnesh Jangid

def _check_shape(mat, shape, label="array"):
    """Validate the shape of an array-like object.

    Args:
        mat : array-like
            Object whose shape is checked.
        shape : tuple of int
            Required shape.
        label : str, optional
            Name used to identify mat in the error message.

    Raises:
        ValueError
            If the shape of mat differs from shape.
    """
    shape = tuple(shape)
    if np.shape(mat) != shape:
        msg = f"{label} has shape {np.shape(mat)}; expected {shape}"
        raise ValueError(msg)


class ActiveActiveRotationMap:
    """Map inter-fragment Wannier rotations to Bloch-MO rotations.

    The periodic orbital optimizer represents active rotations as independent
    lower-triangular pairs within each k-point block. The LAS fragment
    partition instead identifies nonredundant active-active rotations between
    Wannier fragments. This class constructs the linear map between those two
    representations and compresses its image to an orthonormal basis.

    For a selected Bloch pair (k, a, b) and Wannier pair (p, q),
    pair_map contains

    mo_phase[k, a, p] * mo_phase[k, b, q].conj().

    The singular vectors spanning the image of this map define the independent
    active-active coordinates used by the k-LASSCF unitary-group generator.
    The map is complex-linear; anti-Hermitian completion is applied only by
    :meth:`unpack`, after the lower-pair coordinates have been expanded.

    Args:
        mo_phase : ndarray of shape (nkpts, ncas, ncastot)
            Unitary transformation from the complete Wannier active space to
            the active Bloch MOs at each k-point. ncastot must equal
            nkpts * ncas.
        ncas_sub : array-like of int
            Numbers of active orbitals assigned to the LAS fragments. Their
            sum must equal ncastot; their order defines the Wannier
            fragment partition.
        block_pair_mask : ndarray of bool, optional
            Mask of shape (nkpts, ncas, ncas) selecting the strictly
            lower-triangular Bloch active pairs available to the optimizer.
            By default, every strictly lower-triangular pair is selected.
        svd_tol : float, optional
            Absolute singular-value cutoff used to determine the rank of the
            pair map. By default, a dimension- and precision-scaled cutoff is
            used.

    Attributes:
        pair_map : ndarray
            Complex-linear map from inter-fragment Wannier lower-pair
            amplitudes to selected Bloch lower-pair amplitudes.
        basis : ndarray
            Orthonormal basis for the image of pair_map. Its number of
            columns is :attr:`nvar`.
        singular_values : ndarray
            Singular values of pair_map in descending order.
    """

    def __init__(
            self, mo_phase, ncas_sub, block_pair_mask=None, svd_tol=None):
        mo_phase = np.asarray(mo_phase)
        ncas_sub = np.asarray(ncas_sub, dtype=int).reshape(-1)
        if mo_phase.ndim != 3:
            msg = (
                "mo_phase must have shape (nkpts, ncas, ncastot); "
                f"got {mo_phase.shape}"
            )
            raise ValueError(msg)
        mo_phase = np.asarray(
            mo_phase, dtype=np.result_type(mo_phase.dtype, np.complex128),
        )
        if svd_tol is not None:
            svd_tol = float(svd_tol)

        self.nkpts, self.ncas, self.ncastot = mo_phase.shape
        if self.ncastot != self.nkpts * self.ncas:
            msg = (
                "mo_phase must map a square stacked block-active space; "
                f"got nkpts*ncas={self.nkpts * self.ncas} and "
                f"ncastot={self.ncastot}"
            )
            raise ValueError(msg)
        if int(ncas_sub.sum()) != self.ncastot:
            msg = (
                f"sum(ncas_sub)={int(ncas_sub.sum())}; expected "
                f"ncastot={self.ncastot}"
            )
            raise ValueError(msg)
        self.mo_phase = mo_phase
        self.ncas_sub = ncas_sub

        fragment = np.repeat(np.arange(ncas_sub.size), ncas_sub)
        self.wannier_pair_idx = np.where(
            fragment[:, None] > fragment[None, :]
        )

        if block_pair_mask is None:
            block_pair_mask = np.broadcast_to(
                np.tril(np.ones((self.ncas, self.ncas), dtype=bool), -1),
                (self.nkpts, self.ncas, self.ncas),
            )
        block_pair_mask = np.asarray(block_pair_mask, dtype=bool)
        _check_shape(
            block_pair_mask, (self.nkpts, self.ncas, self.ncas),
            label="block_pair_mask",
        )
        self.block_pair_mask = np.array(block_pair_mask, copy=True)
        self.block_pair_idx = np.where(self.block_pair_mask)

        block_k, block_row, block_col = self.block_pair_idx
        wannier_row, wannier_col = self.wannier_pair_idx
        self.pair_map = (
            self.mo_phase[
                block_k[:, None], block_row[:, None],
                wannier_row[None, :],
            ]
            * self.mo_phase[
                block_k[:, None], block_col[:, None],
                wannier_col[None, :],
            ].conj()
        )

        nblock_pair, nwannier_pair = self.pair_map.shape
        basis_dtype = np.result_type(self.mo_phase.dtype, np.complex128)
        if nblock_pair == 0 or nwannier_pair == 0:
            self.singular_values = np.empty(0, dtype=float)
            self.svd_tol = 0.0 if svd_tol is None else float(svd_tol)
            self.basis = np.empty((nblock_pair, 0), dtype=basis_dtype)
            # Return early if no valid pairs are found
            return

        left, singular_values, _ = np.linalg.svd(
            self.pair_map, full_matrices=False,
        )
        if svd_tol is None:
            real_dtype = np.empty((), dtype=self.pair_map.real.dtype).dtype
            svd_tol = (
                max(self.pair_map.shape) * np.finfo(real_dtype).eps
                * singular_values[0]
            )
        svd_tol = float(svd_tol)
        rank = int(np.count_nonzero(singular_values > svd_tol))
        self.singular_values = singular_values
        self.svd_tol = svd_tol
        self.basis = np.asarray(left[:, :rank], dtype=basis_dtype)

    @property
    def nvar(self):
        """int: Number of independent active-active coordinates."""
        return self.basis.shape[1]

    def block_to_wannier(self, kappa_active):
        """Transform a block-diagonal Bloch matrix to the Wannier basis.

        Args:
            kappa_active : ndarray of shape (nkpts, ncas, ncas)
                Active-space matrix for each k-point.

        Returns:
            ndarray of shape (ncastot, ncastot)
                Matrix in the complete Wannier active space.
        """
        kappa_active = np.asarray(kappa_active)
        _check_shape(
            kappa_active, (self.nkpts, self.ncas, self.ncas),
            label="kappa_active",
        )
        return np.einsum(
            "kap,kab,kbq->pq", self.mo_phase.conj(), kappa_active,
            self.mo_phase, optimize=True,
        )

    def wannier_to_block(self, kappa_wannier):
        """Transform a Wannier matrix to its k-diagonal Bloch blocks.

        Args:
            kappa_wannier : ndarray of shape (ncastot, ncastot)
                Matrix in the complete Wannier active space.

        Returns:
            ndarray of shape (nkpts, ncas, ncas)
                K-diagonal active-space blocks in the Bloch-MO basis.

        Notes:
            Components that couple different k-points are omitted from the
            returned block representation.
        """
        kappa_wannier = np.asarray(kappa_wannier)
        _check_shape(
            kappa_wannier, (self.ncastot, self.ncastot),
            label="kappa_wannier",
        )
        return np.einsum(
            "kap,pq,kbq->kab", self.mo_phase, kappa_wannier,
            self.mo_phase.conj(), optimize=True,
        )

    def pack(self, kappa_active):
        """Project Bloch active rotations onto independent coordinates.

        Args:
            kappa_active : ndarray of shape (nkpts, ncas, ncas)
                Active-space rotation matrix for each k-point. Only entries
                selected by block_pair_mask are read.

        Returns:
            ndarray of shape (nvar,)
                Coordinates in the orthonormal image of pair_map.
        """
        kappa_active = np.asarray(kappa_active)
        _check_shape(
            kappa_active, (self.nkpts, self.ncas, self.ncas),
            label="kappa_active",
        )
        block_pairs = np.asarray(kappa_active[self.block_pair_idx])
        return np.asarray(self.basis.conj().T @ block_pairs).reshape(-1)

    def unpack(self, coordinates):
        """Expand independent coordinates into Bloch rotation matrices.

        Args:
            coordinates : array-like of shape (nvar,)
                Coordinates in the orthonormal image of pair_map.

        Returns:
            ndarray of shape (nkpts, ncas, ncas)
                Anti-Hermitian active-space rotation matrix at each k-point.

        Raises:
            ValueError
                If the number of coordinates differs from :attr:`nvar`.
        """
        coordinates = np.asarray(coordinates).reshape(-1)
        if coordinates.size != self.nvar:
            msg = (
                f"active-active vector has size {coordinates.size}; "
                f"expected {self.nvar}"
            )
            raise ValueError(msg)
        dtype = np.result_type(coordinates.dtype, self.basis.dtype)
        kappa_active = np.zeros(
            (self.nkpts, self.ncas, self.ncas), dtype=dtype,
        )
        kappa_active[self.block_pair_idx] = self.basis @ coordinates
        return kappa_active - kappa_active.conj().transpose(0, 2, 1)


class KLASSCF_UnitaryGroupGenerators:
    """Pack and unpack the k-LASSCF orbital and CI variables.

    Orbital variables are ordered in two sections. The ordinary nonredundant
    core-active, core-virtual, and active-virtual rotations are stored first,
    grouped by k-point. They are followed by the independent active-active
    rotations obtained from :class:`ActiveActiveRotationMap`. CI variables
    come last, ordered by fragment and root and represented in the complex CSF
    basis.
    
    Args:
        klas : object
            Periodic LAS object supplying the orbital-space dimensions,
            fragment solvers, electron counts, and optional frozen variables.
        mo_coeff : ndarray of shape (nkpts, nao, nmo), optional
            Bloch-MO coefficients. Defaults to klas.mo_coeff.
        ci : sequence, optional
            Nested [fragment][root] determinant-basis CI vectors. Defaults
            to klas.ci.
        mo_phase : ndarray of shape (nkpts, ncas, nkpts*ncas), optional
            Bloch-active to Wannier-active transformation. It defaults to
            klas.mo_phase when available and is otherwise constructed from
            the active MOs.

    Attributes:
        uniq_orb_idx : ndarray of bool
            Mask selecting ordinary nonredundant orbital rotations at every
            k-point.
        active_active_map : ActiveActiveRotationMap
            Projection defining the independent inter-fragment active-active
            rotations.
        ci_transformers : list
            CSF transformers corresponding to each fragment and CI root.
    """

    def __init__(self, klas, mo_coeff=None, ci=None, mo_phase=None):
        if mo_coeff is None:
            mo_coeff = klas.mo_coeff
        if ci is None:
            ci = klas.ci
        mo_coeff = np.asarray(mo_coeff)
        self.nkpts = len(klas.kpts)
        self.nmo = mo_coeff.shape[-1]

        if mo_coeff.ndim != 3:
            msg = (
                "mo_coeff must have shape (nkpts, nao, nmo); "
                f"got {mo_coeff.shape}"
            )
            raise ValueError(msg)

        _check_shape(
            mo_coeff,
            (self.nkpts, mo_coeff.shape[1], mo_coeff.shape[2]),
            label="mo_coeff",
        )

        ncore = klas.ncore
        ncas = klas.ncas
        self.ncore = ncore
        nocc = ncore + ncas
        orb_idx = np.zeros((self.nmo, self.nmo), dtype=bool)
        orb_idx[ncore:nocc, :ncore] = True
        orb_idx[nocc:, :nocc] = True
        nonfrozen = np.ones(self.nmo, dtype=bool)

        # Keep the molecular frozen-orbital convention. This path has not yet
        # been exercised by the periodic optimizer.
        frozen = getattr(klas, "frozen", None)
        if frozen is not None:
            if isinstance(frozen, (int, np.integer)):
                orb_idx[:frozen, :] = False
                orb_idx[:, :frozen] = False
                nonfrozen[:frozen] = False
            else:
                frozen = np.asarray(frozen)
                orb_idx[frozen, :] = False
                orb_idx[:, frozen] = False
                nonfrozen[frozen] = False

        self.uniq_orb_idx = np.broadcast_to(
            orb_idx, (self.nkpts, self.nmo, self.nmo),
        ).copy()
        if mo_phase is None:
            mo_phase = getattr(klas, "mo_phase", None)
        if mo_phase is None:
            mo_act_kpts = mo_coeff[:, :, ncore:nocc]
            mo_phase = get_wannier_orbs(
                klas._scf, klas.kmesh, mo_act_kpts,
            )[-1]
        self.mo_phase = np.asarray(mo_phase)
        _check_shape(
            self.mo_phase, (self.nkpts, ncas, self.nkpts * ncas),
            label="mo_phase",
        )

        active_nonfrozen = nonfrozen[ncore:nocc]
        active_pair_mask = np.broadcast_to(
            np.tril(np.ones((ncas, ncas), dtype=bool), -1),
            (self.nkpts, ncas, ncas),
        ).copy()
        active_pair_mask &= active_nonfrozen[None, :, None]
        active_pair_mask &= active_nonfrozen[None, None, :]
        self.active_active_map = ActiveActiveRotationMap(
            self.mo_phase, klas.ncas_sub,
            block_pair_mask=active_pair_mask,
        )
        self.frozen_ci = set(getattr(klas, "frozen_ci", None) or [])
        self.ci = ci
        self.ci_transformers = []
        for ifrag, (fcibox, norb, nelec, ci_r) in enumerate(zip(
                klas.fciboxes, klas.ncas_sub, klas.nelecas_sub, ci)):
            if len(fcibox.fcisolvers) != len(ci_r):
                msg = (
                    f"cell {ifrag} has {len(fcibox.fcisolvers)} solvers for "
                    f"{len(ci_r)} CI roots"
                )
                raise ValueError(msg)
            transformers = []
            for solver in fcibox.fcisolvers:
                solver.norb = norb
                solver.nelec = fcibox._get_nelec(solver, nelec)
                solver.check_transformer_cache()
                transformers.append(solver.transformer)
            self.ci_transformers.append(transformers)

    @property
    def nvar_orb_external(self):
        """int: Number of ordinary block-diagonal orbital variables."""
        return int(np.count_nonzero(self.uniq_orb_idx))

    @property
    def nvar_orb_active_active(self):
        """int: Number of projected active-active orbital variables."""
        return self.active_active_map.nvar

    @property
    def nvar_orb(self):
        """int: Total number of independent orbital variables."""
        return self.nvar_orb_external + self.nvar_orb_active_active

    @property
    def ncsf_sub(self):
        """ndarray: Numbers of CSFs for the nonfrozen fragment roots."""
        return np.asarray([
            [transformer.ncsf for transformer in transformers]
            for ifrag, transformers in enumerate(self.ci_transformers)
            if ifrag not in self.frozen_ci
        ], dtype=int)

    @property
    def nvar_ci(self):
        """int: Total number of nonfrozen complex CI variables."""
        return int(self.ncsf_sub.sum())

    @property
    def nvar_tot(self):
        """int: Total number of orbital and CI variables."""
        return self.nvar_orb + self.nvar_ci

    def get_gx_idx(self):
        """Return the mask for orbital variables excluded from optimization.

        Returns:
            ndarray of bool, shape (nkpts, nmo, nmo)
                An all-false mask because k-LASSCF currently optimizes every
                orbital variable selected by this generator.
        """
        return np.zeros_like(self.uniq_orb_idx)

    def pack_orb(self, kappa):
        """Pack Bloch orbital rotations into independent coordinates.

        Args:
            kappa : ndarray of shape (nkpts, nmo, nmo)
                Orbital-rotation matrices. The selected lower-pair entries are
                read; redundant entries are ignored.

        Returns:
            ndarray of shape (nvar_orb,)
                Ordinary orbital variables followed by projected
                active-active variables.
        """
        kappa = np.asarray(kappa)
        _check_shape(kappa, (self.nkpts, self.nmo, self.nmo), label="kappa")
        x_external = np.asarray(kappa[self.uniq_orb_idx]).reshape(-1)
        active = slice(
            self.ncore, self.ncore + self.active_active_map.ncas,
        )
        x_active_active = self.active_active_map.pack(
            kappa[:, active, active],
        )
        return np.concatenate((x_external, x_active_active))

    def unpack_orb(self, x_orb):
        """Unpack independent coordinates as anti-Hermitian rotations.

        Args:
            x_orb : array-like of shape (nvar_orb,)
                Packed complex orbital coordinates.

        Returns:
            ndarray of shape (nkpts, nmo, nmo)
                Anti-Hermitian orbital-rotation matrix for each k-point.
        """
        x_orb = np.asarray(x_orb).reshape(-1)
        if x_orb.size != self.nvar_orb:
            msg = (
                f"orbital vector has size {x_orb.size}; "
                f"expected {self.nvar_orb}"
            )
            raise ValueError(msg)
        dtype = np.result_type(
            x_orb.dtype, self.active_active_map.basis.dtype,
        )
        kappa = np.zeros(
            (self.nkpts, self.nmo, self.nmo), dtype=dtype,
        )
        nvar_external = self.nvar_orb_external
        kappa[self.uniq_orb_idx] = x_orb[:nvar_external]
        kappa -= kappa.conj().transpose(0, 2, 1)
        active = slice(
            self.ncore, self.ncore + self.active_active_map.ncas,
        )
        kappa[:, active, active] += self.active_active_map.unpack(
            x_orb[nvar_external:],
        )
        return kappa

    def pack_ci(self, ci):
        """Pack determinant-basis CI vectors in the complex CSF basis.

        Args:
            ci : sequence
                Nested [fragment][root] determinant-basis CI vectors.

        Returns:
            ndarray of shape (nvar_ci,)
                Flattened CSF coefficients for all nonfrozen fragments.
        """
        if len(ci) != len(self.ci_transformers):
            msg = "CI input must contain one entry per cell"
            raise ValueError(msg)
        vectors = []
        for ifrag, (transformers, ci_r) in enumerate(zip(
                self.ci_transformers, ci)):
            if len(ci_r) != len(transformers):
                msg = (
                    f"cell {ifrag} has {len(ci_r)} CI vectors; "
                    f"expected {len(transformers)}"
                )
                raise ValueError(msg)
            if ifrag in self.frozen_ci:
                continue
            for transformer, c in zip(transformers, ci_r):
                c_csf = cplx_csf_helper.vec_det2csf_cplx(
                    transformer, c, normalize=False,
                )
                vectors.append(np.asarray(c_csf).reshape(-1))
        if not vectors:
            return np.empty(0, dtype=np.complex128)
        return np.concatenate(vectors)

    def unpack_ci(self, x_ci):
        """Unpack complex CSF coordinates into determinant-basis vectors.

        Frozen fragments are represented by zero response vectors with the
        same shapes as their reference CI vectors.

        Args:
            x_ci : array-like of shape (nvar_ci,)
                Packed CSF coefficients for the nonfrozen fragments.

        Returns:
            list
                Nested [fragment][root] determinant-basis CI responses.
        """
        x_ci = np.asarray(x_ci).reshape(-1)
        if x_ci.size != self.nvar_ci:
            msg = (
                f"CI vector has size {x_ci.size}; expected {self.nvar_ci}"
            )
            raise ValueError(msg)
        ci = []
        offset = 0
        for ifrag, (transformers, ci_ref_r) in enumerate(zip(
                self.ci_transformers, self.ci)):
            ci_r = []
            for transformer, c_ref in zip(transformers, ci_ref_r):
                if ifrag in self.frozen_ci:
                    dtype = np.result_type(c_ref, x_ci.dtype)
                    ci_r.append(np.zeros(np.shape(c_ref), dtype=dtype))
                    continue
                ncsf = transformer.ncsf
                c = cplx_csf_helper.vec_csf2det_cplx(
                    transformer, x_ci[offset:offset + ncsf],
                    normalize=False,
                )
                ci_r.append(np.asarray(c).reshape(np.shape(c_ref)))
                offset += ncsf
            ci.append(ci_r)
        if offset != x_ci.size:
            msg = (
                f"consumed {offset} CI variables from a vector of size "
                f"{x_ci.size}"
            )
            raise ValueError(msg)
        return ci

    def pack(self, kappa, ci):
        """Pack orbital and CI variables into one complex vector."""
        x_orb = self.pack_orb(kappa)
        x_ci = self.pack_ci(ci)
        dtype = np.result_type(x_orb.dtype, x_ci.dtype)
        x = np.empty(self.nvar_tot, dtype=dtype)
        x[:self.nvar_orb] = x_orb
        x[self.nvar_orb:] = x_ci
        return x

    def unpack(self, x):
        """Unpack a combined vector into orbital and CI variables."""
        x = np.asarray(x).reshape(-1)
        if x.size != self.nvar_tot:
            msg = (
                f"combined vector has size {x.size}; expected {self.nvar_tot}"
            )
            raise ValueError(msg)
        return (
            self.unpack_orb(x[:self.nvar_orb]),
            self.unpack_ci(x[self.nvar_orb:]),
        )


def get_ugg(klas, mo_coeff=None, ci=None, mo_phase=None):
    """Construct the unitary-group generator used by k-LASSCF.

    The construction is dispatched through klas._ugg so that periodic
    LAS subclasses can replace the parameterization without overriding this
    convenience method.

    Args:
        klas : object
            Periodic LAS object for which the parameterization is built.
        mo_coeff : ndarray of shape (nkpts, nao, nmo), optional
            Bloch-MO coefficients. Defaults to klas.mo_coeff in the
            generator constructor.
        ci : sequence, optional
            Nested [fragment][root] determinant-basis CI vectors. Defaults
            to klas.ci in the generator constructor.
        mo_phase : ndarray of shape (nkpts, ncas, nkpts*ncas), optional
            Bloch-active to Wannier-active transformation. When omitted, the
            generator obtains or constructs it from klas.

    Returns:
        KLASSCF_UnitaryGroupGenerators
            Orbital/CI parameterization associated with the supplied state.
    """
    return klas._ugg(
        klas, mo_coeff=mo_coeff, ci=ci, mo_phase=mo_phase,
    )


def get_grad_ci(
        klas, mo_coeff=None, ci=None, ugg=None, casdm1frs=None,
        h1eff=None, h2eff=None):
    """Evaluate the k-LASSCF energy gradient with respect to the CI vectors.

    For each fragment and root, this function constructs the local Hamiltonian
    action and removes its component parallel to the reference CI vector. The
    resulting determinant-basis residual is

    2 * (H c - <c|H c> c).

    Constructing the residual directly keeps the gradient layer independent
    of the k-LASSCF Hessian operator.

    Args:
        klas : object
            Periodic LAS object supplying fragment solvers and active-space
            integral builders.
        mo_coeff : ndarray of shape (nkpts, nao, nmo), optional
            Bloch-MO coefficients. Defaults to klas.mo_coeff.
        ci : sequence, optional
            Nested [fragment][root] determinant-basis CI vectors. Defaults
            to klas.ci.
        ugg : KLASSCF_UnitaryGroupGenerators, optional
            Accepted for compatibility with the combined-gradient interface;
            it is not needed to form determinant-basis CI residuals.
        casdm1frs : sequence, optional
            Fragment- and root-resolved active-space one-particle density
            matrices used to build h1eff when it is not supplied.
        h1eff : sequence, optional
            Effective one-electron Hamiltonians for each fragment, with each
            block shaped (nroots, 2, ncas_frag, ncas_frag).
        h2eff : ndarray of shape (ncastot,)*4, optional
            Two-electron integrals in the complete Wannier active space.

    Returns:
        list
            Nested [fragment][root] determinant-basis CI gradients with
            the same individual shapes as ci.

    Raises:
        ValueError
            If the supplied integral arrays are inconsistent with the active
            spaces or fragment count.
    """
    if mo_coeff is None:
        mo_coeff = klas.mo_coeff
    if ci is None:
        ci = klas.ci
    if h2eff is None:
        h2eff = klas.get_h2cas(mo_coeff)
    h2eff = np.asarray(h2eff)

    ncas_sub = np.asarray(klas.ncas_sub, dtype=int)
    ncastot = int(ncas_sub.sum())
    _check_shape(h2eff, (ncastot,) * 4, label="h2eff")

    if h1eff is None:
        if casdm1frs is None:
            casdm1frs = klas.states_make_casdm1s_sub(
                ci=ci, ncas_sub=ncas_sub,
                nelecas_sub=klas.nelecas_sub,
            )
        casdm1s_sub = klas.make_casdm1s_sub(
            ci=ci, casdm1frs=casdm1frs,
        )
        h1eff = klas.h1e_for_las(
            mo_coeff=mo_coeff, ci=ci, ncas_sub=ncas_sub,
            nelecas_sub=klas.nelecas_sub,
            casdm1s_sub=casdm1s_sub, casdm1frs=casdm1frs,
            eri_cas=h2eff,
        )
    if len(h1eff) != len(ncas_sub):
        raise ValueError(
            "h1eff must contain one block for every fragment/cell"
        )

    gradient = []
    offset = 0
    for ifrag, (fcibox, norb, nelec, h1frs, ci_r) in enumerate(zip(
            klas.fciboxes, ncas_sub, klas.nelecas_sub, h1eff, ci)):
        stop = offset + int(norb)
        _check_shape(
            h1frs, (klas.nroots, 2, norb, norb),
            label=f"h1fr_{ifrag}",
        )
        h2frag = h2eff[offset:stop, offset:stop, offset:stop, offset:stop]
        linkstr = fcibox.states_gen_linkstr(norb, nelec, False)
        absorbed = fcibox.states_absorb_h1e(
            h1frs, h2frag, norb, nelec, 0.5,
        )
        hci_r = fcibox.states_contract_2e(
            absorbed, ci_r, norb, nelec, link_index=linkstr,
        )
        gradient.append([
            2.0 * (hc - np.vdot(c, hc) * c)
            for hc, c in zip(hci_r, ci_r)
        ])
        offset = stop
    return gradient


def get_grad_orb(
        klas, mo_coeff_kpts=None, ci=None, h2eff_sub=None,
        veff_kpts=None, dm1s_kpts=None, hermi=-1):
    """Evaluate the k-LASSCF orbital gradient or effective Fock matrix.

    The one-body contribution is formed independently at each k-point. The
    active-space two-body cumulant is transformed from the Wannier basis to
    momentum-conserving Bloch blocks and contracted with the paaa AO2MO
    intermediates.

    Args:
        klas : object
            Periodic LAS object supplying density matrices, integrals, and
            k-point metadata.
        mo_coeff_kpts : ndarray of shape (nkpts, nao, nmo), optional
            Bloch-MO coefficients. Defaults to klas.mo_coeff.
        ci : sequence, optional
            Nested [fragment][root] Wannier-basis CI vectors. Defaults to
            klas.ci.
        h2eff_sub : _ERIS or ndarray, optional
            Object providing paaa(k1, k2, k3) or an explicit array with
            shape (nkpts, nkpts, nkpts, nmo, ncas, ncas, ncas). It is
            constructed with klas._klasscf_eris when omitted.
        veff_kpts : ndarray of shape (2, nkpts, nao, nao), optional
            Spin-resolved state-averaged effective potential in the AO basis.
        dm1s_kpts : ndarray of shape (2, nkpts, nao, nao), optional
            Spin-resolved state-averaged one-particle density matrix in the AO
            basis.
        hermi : {-1, 0, 1}, optional
            Selects the returned part of the effective Fock matrix. -1
            returns F - F†, the anti-Hermitian orbital gradient; 0
            returns F; and 1 returns (F + F†) / 2.

    Returns:
        ndarray of shape (nkpts, nmo, nmo)
            Orbital gradient or requested Hermitian component. The
            anti-Hermitian result is not divided by the number of k-points.

    Raises:
        ValueError
            If an input has an incompatible shape or hermi is not one of
            -1, 0, and 1.
    """
    cell = klas._scf.cell
    kpts = klas.kpts
    nkpts = len(kpts)

    if mo_coeff_kpts is None:
        mo_coeff_kpts = klas.mo_coeff
    mo_coeff_kpts = np.asarray(mo_coeff_kpts)
    if ci is None:
        ci = klas.ci
    if dm1s_kpts is None:
        dm1s_kpts = klas.make_rdm1s(mo_coeff=mo_coeff_kpts, ci=ci)
    if h2eff_sub is None:
        h2eff_sub = klas._klasscf_eris(klas, mo_coeff_kpts)
    if veff_kpts is None:
        veff_kpts = klas.get_veff(cell, dm_kpts=dm1s_kpts)

    _, nmo = mo_coeff_kpts.shape[-2:]
    ncore = klas.ncore
    ncas = klas.ncas
    nocc = ncore + ncas
    ncastot = nkpts * ncas

    get_paaa = getattr(h2eff_sub, "paaa", None)
    if get_paaa is None:
        _check_shape(
            h2eff_sub,
            (nkpts, nkpts, nkpts, nmo, ncas, ncas, ncas),
            label="h2eff_sub",
        )
        get_paaa = lambda k1, k2, k3: h2eff_sub[k1, k2, k3]

    dtype = np.result_type(
        mo_coeff_kpts.dtype, veff_kpts.dtype, dm1s_kpts.dtype,
    )
    ovlp_kpts = klas._scf.get_ovlp(kpts=kpts)
    hcore_kpts = klas.get_hcore(kpts=kpts)
    h1es_kpts = hcore_kpts[None, :, :, :] + veff_kpts

    f1 = np.empty((nkpts, nmo, nmo), dtype=dtype)
    for k in range(nkpts):
        smo_coeff_k = ovlp_kpts[k] @ mo_coeff_kpts[k]
        dm1s_mo = (
            smo_coeff_k.conj().T @ dm1s_kpts[:, k] @ smo_coeff_k
        )
        h1es_mo = (
            mo_coeff_kpts[k].conj().T
            @ h1es_kpts[:, k]
            @ mo_coeff_kpts[k]
        )
        f1[k] = (
            h1es_mo[0] @ dm1s_mo[0]
            + h1es_mo[1] @ dm1s_mo[1]
        )

    # Convert the spin-summed 2-RDM to its cumulant in the Wannier basis.
    casdm2 = klas.make_casdm2(ci=ci)
    _check_shape(casdm2, (ncastot,) * 4, label="casdm2")
    casdm1s = klas.make_casdm1s(ci=ci)
    _check_shape(casdm1s, (2, ncastot, ncastot), label="casdm1s")
    casdm1 = casdm1s.sum(0)
    casdm2 -= np.multiply.outer(casdm1, casdm1)
    casdm2 += np.multiply.outer(
        casdm1s[0], casdm1s[0],
    ).transpose(0, 3, 2, 1)
    casdm2 += np.multiply.outer(
        casdm1s[1], casdm1s[1],
    ).transpose(0, 3, 2, 1)

    mo_act_kpts = mo_coeff_kpts[:, :, ncore:nocc]
    mo_phase = get_wannier_orbs(
        klas._scf, klas.kmesh, mo_act_kpts,
    )[-1]
    kconserv = kpts_helper.get_kconserv(cell, kpts)

    for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
        k4 = kconserv[k1, k2, k3]
        casdm2_kpts = _get_casdm2_kpts(
            casdm2, mo_phase, (k1, k2, k3, k4),
        )
        f1[k1][:, ncore:nocc] += np.tensordot(
            get_paaa(k1, k2, k3), casdm2_kpts,
            axes=((1, 2, 3), (1, 2, 3)),
        )

    f1_h = f1.conj().transpose(0, 2, 1)
    if hermi == -1:
        return f1 - f1_h
    if hermi == 0:
        return f1
    if hermi == 1:
        return 0.5 * (f1 + f1_h)
    raise ValueError("kwarg 'hermi' must be -1, 0, or +1")


def get_grad(
        klas, mo_coeff=None, ci=None, ugg=None, h2eff_sub=None,
        veff_kpts=None, dm1s_kpts=None, casdm1frs=None,
        h1eff=None, h2eff=None):
    """Return the packed k-LASSCF orbital and CI energy gradient.

    The orbital gradient is packed first, followed by the CI gradient, using
    the ordering defined by ugg.

    Args:
        klas : object
            Periodic LAS object providing the gradient methods.
        mo_coeff : ndarray of shape (nkpts, nao, nmo), optional
            Bloch-MO coefficients. Defaults to klas.mo_coeff.
        ci : sequence, optional
            Nested [fragment][root] determinant-basis CI vectors. Defaults
            to klas.ci.
        ugg : KLASSCF_UnitaryGroupGenerators, optional
            Parameterization used to pack the result. It is constructed from
            mo_coeff and ci when omitted.
        h2eff_sub : _ERIS or ndarray, optional
            paaa intermediates forwarded to :func:`get_grad_orb`.
        veff_kpts : ndarray, optional
            Spin-resolved effective potential forwarded to
            :func:`get_grad_orb`.
        dm1s_kpts : ndarray, optional
            Spin-resolved AO density forwarded to :func:`get_grad_orb`.
        casdm1frs : sequence, optional
            Fragment/root density matrices forwarded to :func:`get_grad_ci`.
        h1eff : sequence, optional
            Fragment effective one-electron Hamiltonians forwarded to
            :func:`get_grad_ci`.
        h2eff : ndarray, optional
            Wannier active-space two-electron integrals forwarded to
            :func:`get_grad_ci`.

    Returns:
        ndarray of shape (ugg.nvar_tot,)
            Packed complex gradient with orbital variables before CI
            variables.
    """
    if mo_coeff is None:
        mo_coeff = klas.mo_coeff
    if ci is None:
        ci = klas.ci
    if ugg is None:
        ugg = klas.get_ugg(mo_coeff=mo_coeff, ci=ci)
    gorb = klas.get_grad_orb(
        mo_coeff_kpts=mo_coeff, ci=ci, h2eff_sub=h2eff_sub,
        veff_kpts=veff_kpts, dm1s_kpts=dm1s_kpts,
    )
    gci = klas.get_grad_ci(
        mo_coeff=mo_coeff, ci=ci, ugg=ugg, casdm1frs=casdm1frs,
        h1eff=h1eff, h2eff=h2eff,
    )
    return ugg.pack(gorb, gci)


class KLASSCF_HessianOperator(molLASSCF_HessianOperator):
    """Matrix-free orbital/CI Hessian operator for k-LASSCF.

    The periodic operator retains one determinant-basis CI vector per
    fragment and root internally. Its external vector layout is delegated to
    the k-LASSCF unitary-group generator.

    This initial interface provides the common vector-layout helpers. The
    response blocks and their intermediates are supplied by subsequent
    implementation layers.
    """

    def __init__(
            self, las, ugg, mo_coeff=None, ci=None, casdm1frs=None,
            h1eff=None, h2eff=None, kpts=None, kmesh=None, casdm2fr=None,
            eris=None, veff_kpts=None, dm1s_kpts=None, mo_phase=None):
        """Initialize the periodic Hessian intermediates.

        Args:
            las : object
                Periodic LASCI object defining the reference state.
            ugg : KLASSCF_UnitaryGroupGenerators
                Orbital/CI parameterization for external trial vectors.
            mo_coeff, ci : optional
                Reference orbitals and CI vectors. They default to the
                corresponding attributes of ``las``.
            casdm1frs, casdm2fr : optional
                Precomputed fragment density matrices in the Wannier basis.
            h1eff, h2eff : optional
                Precomputed local one-electron Hamiltonians and full Wannier
                active-space two-electron integrals.
            kpts, kmesh : optional
                BvK k-points and three-dimensional k-point mesh.
            eris : optional
                Lazy block-MO periodic ERI object.
            veff_kpts, dm1s_kpts : optional
                Spin-resolved AO potential and density at every k-point.
            mo_phase : optional
                Wannier-to-Bloch active-space transformation.
        """
        if mo_coeff is None:
            mo_coeff = las.mo_coeff
        if ci is None:
            ci = las.ci
        if kpts is None:
            kpts = las.kpts
        if kmesh is None:
            kmesh = las.kmesh
        kpts = np.asarray(kpts)
        kmesh = tuple(int(n) for n in kmesh)

        if len(kmesh) != 3 or any(n <= 0 for n in kmesh):
            raise ValueError("kmesh must contain three positive integers")
        ncell = int(np.prod(kmesh))
        if len(kpts) != ncell:
            raise ValueError(
                f"kpts and kmesh are inconsistent: {len(kpts)} != {ncell}"
            )

        self.las = las
        self.ugg = ugg
        self.mo_coeff = np.asarray(mo_coeff)
        self.ci = ci
        self.kpts = kpts
        self.kmesh = kmesh
        self.nkpts = len(kpts)
        self.ncell = ncell

        self.level_shift = las.ah_level_shift
        self.ncore = las.ncore
        self.ncas_sub = np.asarray(las.ncas_sub)
        self.nelecas_sub = np.asarray(las.nelecas_sub)
        self.ncas = int(las.ncas)
        self.ncastot = self.ncas * self.nkpts
        self.nao = self.mo_coeff.shape[-2]
        self.nmo = self.mo_coeff.shape[-1]
        self.nocc = self.ncore + self.ncas
        if np.sum(self.ncas_sub) != self.ncastot:
            msg = (
                "Wannier and block-MO active spaces are inconsistent: "
                f"sum(ncas_sub)={np.sum(self.ncas_sub)}, but ncastot="
                f"ncas*nkpts={self.ncastot}"
            )
            raise ValueError(msg)
        if self.nocc > self.nmo:
            msg = (
                "mo_coeff does not contain the full core and active spaces: "
                f"ncore+ncas={self.nocc}, nmo={self.nmo}"
            )
            raise ValueError(msg)
        self.fciboxes = las.fciboxes
        self.nroots = las.nroots
        self.weights = las.weights
        self.ci_transformers = ugg.ci_transformers
        self.frozen_ci = set(getattr(ugg, "frozen_ci", None) or [])
        if len(self.ci_transformers) != len(self.ci):
            raise ValueError(
                "ugg.ci_transformers must contain one entry per CI cell"
            )
        self.nvar_ci = 0
        for ifrag, (transformers, ci0_r) in enumerate(zip(
                self.ci_transformers, self.ci)):
            if len(transformers) != len(ci0_r):
                msg = (
                    f"cell {ifrag} has {len(transformers)} CSF transformers "
                    f"for {len(ci0_r)} CI roots"
                )
                raise ValueError(msg)
            if ifrag not in self.frozen_ci:
                self.nvar_ci += sum(t.ncsf for t in transformers)

        self._init_dms_(casdm1frs, casdm2fr, dm1s_kpts)
        self._init_ham_(h1eff, h2eff, veff_kpts)
        self._init_eri_(eris)
        self._init_orb_(mo_phase)
        self._init_ci_()
        self._Horb_diag_matvec_cache = None
        self._Horb_diag_external_cache = None
        self._Horb_active_active_cache = None
        self._active_wannier_intermediates_cache = None
        self._Horb_external_active_cross_cache = None

    def _init_dms_(self, casdm1frs, casdm2fr=None, dm1s_kpts=None):
        """Initialize reference density matrices in their natural bases.

        ``casdm1s``, ``casdm2``, and ``cascm2`` are retained in the complete
        Wannier active space. ``dm1s_kpts`` is spin resolved in the AO basis,
        while ``dm1s`` is its block-MO representation.
        """
        if casdm1frs is None:
            casdm1frs = self.las.states_make_casdm1s_sub(
                ci=self.ci,
                ncas_sub=self.ncas_sub,
                nelecas_sub=self.nelecas_sub,
            )

        self.casdm1frs = casdm1frs
        self.casdm1fs = self.las.make_casdm1s_sub(
            casdm1frs=casdm1frs,
        )
        self.casdm1rs = self.las.states_make_casdm1s(
            casdm1frs=casdm1frs,
        )
        self.casdm1s = np.einsum(
            "r,rsij->sij", self.weights, self.casdm1rs,
        )

        if casdm2fr is None:
            casdm2fr = self.las.states_make_casdm2_sub(
                ci=self.ci,
                ncas_sub=self.ncas_sub,
                nelecas_sub=self.nelecas_sub,
            )
        self.casdm2fr = casdm2fr
        self.casdm2 = self.las.make_casdm2(
            ci=self.ci,
            ncas_sub=self.ncas_sub,
            nelecas_sub=self.nelecas_sub,
            casdm1frs=self.casdm1frs,
            casdm2fr=self.casdm2fr,
        )
        _check_shape(
            self.casdm1s, (2, self.ncastot, self.ncastot),
            label="casdm1s",
        )
        _check_shape(
            self.casdm2, (self.ncastot,) * 4, label="casdm2",
        )

        casdm1a, casdm1b = self.casdm1s
        casdm1 = casdm1a + casdm1b
        self.cascm2 = self.casdm2 - np.multiply.outer(casdm1, casdm1)
        self.cascm2 += np.multiply.outer(
            casdm1a, casdm1a,
        ).transpose(0, 3, 2, 1)
        self.cascm2 += np.multiply.outer(
            casdm1b, casdm1b,
        ).transpose(0, 3, 2, 1)

        if dm1s_kpts is None:
            dm1s_kpts = self.las.make_rdm1s(
                mo_coeff=self.mo_coeff,
                ci=self.ci,
                casdm1s_sub=self.casdm1fs,
            )
        self.dm1s_kpts = np.asarray(dm1s_kpts)
        _check_shape(
            self.dm1s_kpts, (2, self.nkpts, self.nao, self.nao),
            label="dm1s_kpts",
        )

        ovlp_kpts = np.asarray(self.las._scf.get_ovlp(kpts=self.kpts))
        _check_shape(
            ovlp_kpts, (self.nkpts, self.nao, self.nao),
            label="ovlp_kpts",
        )
        dtype = np.result_type(
            self.mo_coeff.dtype, self.dm1s_kpts.dtype, ovlp_kpts.dtype,
        )
        self.dm1s = np.empty(
            (2, self.nkpts, self.nmo, self.nmo), dtype=dtype,
        )
        for k in range(self.nkpts):
            smo_coeff = ovlp_kpts[k] @ self.mo_coeff[k]
            self.dm1s[:, k] = (
                smo_coeff.conj().T @ self.dm1s_kpts[:, k] @ smo_coeff
            )

    def _init_ham_(self, h1eff, h2eff, veff_kpts=None):
        """Initialize block-MO and Wannier-basis Hamiltonians.

        ``h1frs[f][r]`` is the spin-resolved effective one-electron
        Hamiltonian for fragment ``f`` and root ``r``. ``eri_cas`` contains
        the two-electron integrals over the complete Wannier active space.
        ``hcore`` and ``h1s`` retain a k-point axis and use the block-MO basis.
        """
        if h2eff is None:
            h2eff = self.las.get_h2cas(self.mo_coeff)
        h2eff = np.asarray(h2eff)
        _check_shape(h2eff, (self.ncastot,) * 4, label="h2eff")

        if veff_kpts is None:
            veff_kpts = self.las.get_veff(
                self.las._scf.cell, dm_kpts=self.dm1s_kpts,
            )
        self.veff_kpts = np.asarray(veff_kpts)
        _check_shape(
            self.veff_kpts, (2, self.nkpts, self.nao, self.nao),
            label="veff_kpts",
        )

        hcore_kpts = np.asarray(self.las.get_hcore(kpts=self.kpts))
        _check_shape(
            hcore_kpts, (self.nkpts, self.nao, self.nao),
            label="hcore_kpts",
        )
        dtype = np.result_type(
            self.mo_coeff.dtype, self.veff_kpts.dtype, hcore_kpts.dtype,
        )
        self.hcore = np.empty(
            (self.nkpts, self.nmo, self.nmo), dtype=dtype,
        )
        self.h1s = np.empty(
            (2, self.nkpts, self.nmo, self.nmo), dtype=dtype,
        )
        for k in range(self.nkpts):
            mo_coeff = self.mo_coeff[k]
            mo_coeff_h = mo_coeff.conj().T
            self.hcore[k] = mo_coeff_h @ hcore_kpts[k] @ mo_coeff
            self.h1s[:, k] = (
                mo_coeff_h
                @ (hcore_kpts[k][None] + self.veff_kpts[:, k])
                @ mo_coeff
            )

        if h1eff is None:
            h1eff = self.las.h1e_for_las(
                mo_coeff=self.mo_coeff,
                ci=self.ci,
                ncas_sub=self.ncas_sub,
                nelecas_sub=self.nelecas_sub,
                casdm1s_sub=self.casdm1fs,
                casdm1frs=self.casdm1frs,
                eri_cas=h2eff,
                veff=self.veff_kpts,
            )

        if len(h1eff) != len(self.ncas_sub):
            raise ValueError(
                "h1eff must contain one block for every fragment/cell"
            )
        for ifrag, (h1fr, ncas) in enumerate(zip(h1eff, self.ncas_sub)):
            _check_shape(
                h1fr, (self.nroots, 2, ncas, ncas),
                label=f"h1fr_{ifrag}",
            )

        self.h1frs = h1eff
        self.eri_cas = h2eff

    def _init_eri_(self, eris=None):
        """Attach lazy block-MO ERI accessors for orbital response.

        The default periodic ERI object stores ``ppaa``, ``papa``, and
        ``paap`` blocks on disk. ``eri_paaa`` remains an accessor rather than
        a materialized supercell tensor. Level one also constructs the compact
        core-orbital intermediates used by the analytic Hessian diagonal.
        """
        if eris is None:
            eris = _ERIS(
                self.las, self.mo_coeff, method="disk", level=1,
            )
        for name in ("ppaa", "papa", "paap", "paaa"):
            if not callable(getattr(eris, name, None)):
                raise TypeError(f"eris.{name} must be callable")
        self.cas_type_eris = eris
        self.eris = eris
        self.eri_paaa = eris.paaa

    def _init_orb_(self, mo_phase=None):
        """Build the reference generalized Fock matrix in block-MO form."""
        if mo_phase is None:
            mo_act_kpts = self.mo_coeff[:, :, self.ncore:self.nocc]
            mo_phase = get_wannier_orbs(
                self.las._scf, self.kmesh, mo_act_kpts,
            )[-1]
        self.mo_phase = np.asarray(mo_phase)
        _check_shape(
            self.mo_phase, (self.nkpts, self.ncas, self.ncastot),
            label="mo_phase",
        )

        dtype = np.result_type(
            self.h1s.dtype, self.dm1s.dtype, self.cascm2.dtype,
        )
        self.fock1 = np.empty(
            (self.nkpts, self.nmo, self.nmo), dtype=dtype,
        )
        for k in range(self.nkpts):
            self.fock1[k] = sum(
                self.h1s[s, k] @ self.dm1s[s, k]
                for s in range(2)
            )

        kconserv = kpts_helper.get_kconserv(
            self.las._scf.cell, self.kpts,
        )
        active = slice(self.ncore, self.nocc)
        for k1, k2, k3 in kpts_helper.loop_kkk(self.nkpts):
            k4 = kconserv[k1, k2, k3]
            cascm2_kpts = _get_casdm2_kpts(
                self.cascm2, self.mo_phase, (k1, k2, k3, k4),
            )
            paaa_kpts = self.eri_paaa(k1, k2, k3)
            _check_shape(
                paaa_kpts,
                (self.nmo, self.ncas, self.ncas, self.ncas),
                label="paaa_kpts",
            )
            self.fock1[k1][:, active] += np.tensordot(
                paaa_kpts, cascm2_kpts,
                axes=((1, 2, 3), (1, 2, 3)),
            )

    def _init_ci_(self):
        """Cache local Hamiltonian actions, energies, and CI residuals."""
        self.linkstrl = []
        self.linkstr = []
        for fcibox, norb, nelec in zip(
                self.fciboxes, self.ncas_sub, self.nelecas_sub):
            # Complex periodic contractions use ordinary link tables without
            # molecular lower-triangular index packing.
            linkstr = fcibox.states_gen_linkstr(norb, nelec, False)
            self.linkstrl.append(linkstr)
            self.linkstr.append(linkstr)
        hc0 = self.Hci_all(None, self.h1frs, self.eri_cas, self.ci)
        self.e0 = [
            [np.vdot(c, hc) for c, hc in zip(ci_r, hc_r)]
            for ci_r, hc_r in zip(self.ci, hc0)
        ]
        self.hci0 = [
            [
                hc - energy * c
                for hc, energy, c in zip(hc_r, e_r, ci_r)
            ]
            for hc_r, e_r, ci_r in zip(hc0, self.e0, self.ci)
        ]

    def make_tdm1s_sub(self, ci1):
        """Build the first-order spin 1-RDM generated by a CI step.

        Returns:
            ndarray of shape (nroots, 2, ncastot, ncastot)
                Hermitian root-resolved transition density in the complete
                Wannier active space.
        """
        return self._make_tdm1s2c_sub(ci1, with_cumulant=False)[0]

    def make_tdm1s2c_sub(self, ci1):
        """Build complex CI transition 1-RDMs and the effective cumulant.

        Returns:
            tuple
                Hermitian root-resolved spin 1-RDMs and the state-averaged
                effective transition cumulant in the Wannier active space.
        """
        return self._make_tdm1s2c_sub(ci1, with_cumulant=True)

    def _make_tdm1s2c_sub(self, ci1, with_cumulant):
        """Implement the shared one- and two-body CI density builders."""
        dtype = np.result_type(self.eri_cas.dtype, np.complex128)
        tdm1rs_one_sided = np.zeros(
            (self.nroots, 2, self.ncastot, self.ncastot), dtype=dtype,
        )
        if with_cumulant:
            tdm2_one_sided = np.zeros(
                (self.ncastot,) * 4, dtype=dtype,
            )

        for ifrag, (fcibox, norb, nelec, c1_r, c0_r) in enumerate(zip(
                self.fciboxes, self.ncas_sub, self.nelecas_sub,
                ci1, self.ci)):
            i = int(np.sum(self.ncas_sub[:ifrag]))
            j = i + int(norb)
            linkstr = None if self.linkstr is None else self.linkstr[ifrag]

            state_arg = fcibox._state_args
            solver_arg = fcibox._solver_args
            nelec_by_solver = [
                fcibox._get_nelec(solver, nelec)
                for solver in fcibox.fcisolvers
            ]
            collect_args = (
                state_arg(c1_r),
                state_arg(c0_r),
                norb,
                solver_arg(nelec_by_solver),
            )
            collect_kwargs = {"link_index": solver_arg(linkstr)}
            contraction = "trans_rdm12s" if with_cumulant else "trans_rdm1s"
            try:
                transition_rdm_r = list(fcibox._collect(
                    contraction, *collect_args, **collect_kwargs,
                ))
            except AttributeError as err:
                # Fall back for builds predating compiled complex transition
                # density symbols.
                missing_symbol = str(err)
                if not any(name in missing_symbol for name in (
                        "FCItrans_rdm1", "FCItdm12")):
                    raise
                transition_rdm_r = list(fcibox._collect(
                    contraction + "_py", *collect_args, **collect_kwargs,
                ))

            if len(transition_rdm_r) != self.nroots:
                msg = (
                    f"fragment {ifrag} produced {len(transition_rdm_r)} "
                    f"transition densities for {self.nroots} roots"
                )
                raise ValueError(msg)

            for iroot, (transition_rdm, c1, c0, dm1s_ref) in enumerate(zip(
                    transition_rdm_r, c1_r, c0_r,
                    self.casdm1frs[ifrag])):
                if with_cumulant:
                    dm1s, dm2s = transition_rdm
                else:
                    dm1s = transition_rdm
                overlap = np.vdot(c1, c0)
                tdm1s = np.stack(dm1s, axis=0) - overlap * dm1s_ref
                tdm1rs_one_sided[iroot, :, i:j, i:j] = tdm1s

                if with_cumulant:
                    dm2_ref = np.asarray(self.casdm2fr[ifrag][iroot])
                    _check_shape(
                        dm2_ref, (norb,) * 4,
                        label=f"casdm2fr[{ifrag}][{iroot}]",
                    )
                    tdm2 = sum(np.asarray(block) for block in dm2s)
                    tdm2 = (tdm2 - overlap * dm2_ref) / 2.0
                    tdm2_one_sided[i:j, i:j, i:j, i:j] += (
                        self.weights[iroot] * tdm2
                    )

        tdm1rs = (
            tdm1rs_one_sided
            + tdm1rs_one_sided.swapaxes(-1, -2).conj()
        )
        if not with_cumulant:
            return tdm1rs, None

        tcm2 = self._make_effective_transition_cumulant(
            tdm1rs_one_sided, tdm2_one_sided,
        )
        return tdm1rs, tcm2

    def _make_effective_transition_cumulant(
            self, tdm1rs_one_sided, tdm2_one_sided):
        """Construct the complex state-averaged effective CI cumulant.

        Both inputs are the one-sided ``<c1|...|c0>`` quantities after
        reference-overlap subtraction. ``tdm2_one_sided`` contains the
        explicitly correlated same-fragment transition blocks. The
        inter-fragment product-state Coulomb and same-spin exchange blocks
        are differentiated explicitly before the cumulant decomposition.
        The latter uses the stored state-averaged ``self.casdm1s`` as its
        reference density and complements one JK response in the orbital-CI
        Hessian action.
        """
        tdm1rs_one_sided = np.asarray(tdm1rs_one_sided)
        tdm2_one_sided = np.asarray(tdm2_one_sided)
        _check_shape(
            tdm1rs_one_sided,
            (self.nroots, 2, self.ncastot, self.ncastot),
            label="one_sided_transition_dm1rs",
        )
        _check_shape(
            tdm2_one_sided, (self.ncastot,) * 4,
            label="one_sided_transition_dm2",
        )
        _check_shape(
            self.casdm1s, (2, self.ncastot, self.ncastot),
            label="casdm1s",
        )
        weights = np.asarray(self.weights)
        _check_shape(weights, (self.nroots,), label="state_average_weights")
        tdm1rs = (
            tdm1rs_one_sided
            + tdm1rs_one_sided.conj().transpose(0, 1, 3, 2)
        )

        # Complete the same-fragment transition 2-RDM blocks. Starting from
        # half of <c1|Gamma|c0>, the first operation supplies its Hermitian
        # partner and the second supplies electron-pair exchange for both.
        tdm2 = np.array(tdm2_one_sided, copy=True)
        tdm2 += tdm2.conj().transpose(1, 0, 3, 2)
        tdm2 += tdm2.transpose(2, 3, 0, 1)

        # Differentiate the off-diagonal product-state blocks constructed by
        # LASCI.make_casdm2. These terms vanish from the reference cumulant,
        # but their 2-RDM derivatives cancel the corresponding derivative of
        # the mean-field products below.
        offsets = np.cumsum(np.concatenate(([0], self.ncas_sub))).astype(int)
        for ifrag in range(len(self.ncas_sub)):
            i, j = offsets[ifrag:ifrag + 2]
            tdm1s_i = tdm1rs[:, :, i:j, i:j]
            dm1s_i = np.asarray(self.casdm1frs[ifrag])
            _check_shape(
                dm1s_i,
                (self.nroots, 2, j - i, j - i),
                label=f"casdm1frs[{ifrag}]",
            )
            for jfrag in range(ifrag + 1, len(self.ncas_sub)):
                k, l = offsets[jfrag:jfrag + 2]
                tdm1s_j = tdm1rs[:, :, k:l, k:l]
                dm1s_j = np.asarray(self.casdm1frs[jfrag])
                _check_shape(
                    dm1s_j,
                    (self.nroots, 2, l - k, l - k),
                    label=f"casdm1frs[{jfrag}]",
                )

                coulomb = np.einsum(
                    "r,rij,rkl->ijkl",
                    weights, tdm1s_i.sum(axis=1), dm1s_j.sum(axis=1),
                    optimize=True,
                )
                coulomb += np.einsum(
                    "r,rij,rkl->ijkl",
                    weights, dm1s_i.sum(axis=1), tdm1s_j.sum(axis=1),
                    optimize=True,
                )
                tdm2[i:j, i:j, k:l, k:l] = coulomb
                tdm2[k:l, k:l, i:j, i:j] = coulomb.transpose(2, 3, 0, 1)

                exchange = np.zeros(
                    (j - i, l - k, l - k, j - i),
                    dtype=np.result_type(
                        tdm2.dtype, dm1s_i.dtype, dm1s_j.dtype,
                    ),
                )
                for spin in range(2):
                    exchange += np.einsum(
                        "r,rij,rkl->ilkj",
                        weights, tdm1s_i[:, spin], dm1s_j[:, spin],
                        optimize=True,
                    )
                    exchange += np.einsum(
                        "r,rij,rkl->ilkj",
                        weights, dm1s_i[:, spin], tdm1s_j[:, spin],
                        optimize=True,
                    )
                tdm2[i:j, k:l, k:l, i:j] = -exchange
                tdm2[k:l, i:j, i:j, k:l] = (
                    -exchange.conj().transpose(1, 0, 3, 2)
                )

        tdm1s = np.einsum(
            "r,rspq->spq", weights, tdm1rs, optimize=True,
        )
        casdm1 = self.casdm1s.sum(axis=0)
        tdm1 = tdm1s.sum(axis=0)
        tcm2 = np.array(tdm2, copy=True)
        tcm2 -= np.multiply.outer(tdm1, casdm1)
        tcm2 -= np.multiply.outer(casdm1, tdm1)
        for spin in range(2):
            tcm2 += np.multiply.outer(
                tdm1s[spin], self.casdm1s[spin],
            ).transpose(0, 3, 2, 1)
            tcm2 += np.multiply.outer(
                self.casdm1s[spin], tdm1s[spin],
            ).transpose(0, 3, 2, 1)
        _check_shape(
            tcm2, (self.ncastot,) * 4,
            label="transition_cumulant",
        )
        return tcm2

    def _transition_dm1s_to_block(self, tdm1rs):
        """Transform the state-averaged CI transition 1-RDM to block MOs.

        ``tdm1rs`` is root resolved in the complete Wannier active space.
        The returned density has shape ``(2, nkpts, nmo, nmo)`` and is zero
        outside its active-active blocks. This routine performs only state
        averaging and basis transformation; the factor-of-two convention of
        the orbital-CI Hessian action is applied by its eventual caller.
        """
        tdm1rs = np.asarray(tdm1rs)
        weights = np.asarray(self.weights)
        _check_shape(
            tdm1rs,
            (self.nroots, 2, self.ncastot, self.ncastot),
            label="transition_dm1rs",
        )
        _check_shape(
            weights, (self.nroots,), label="state_average_weights",
        )
        _check_shape(
            self.mo_phase,
            (self.nkpts, self.ncas, self.ncastot),
            label="mo_phase",
        )

        tdm1s_wannier = np.einsum(
            "r,rspq->spq", weights, tdm1rs, optimize=True,
        )
        tdm1s_active_block = np.einsum(
            "kap,spq,kbq->skab",
            self.mo_phase, tdm1s_wannier, self.mo_phase.conj(),
            optimize=True,
        )
        dtype = np.result_type(
            tdm1rs.dtype, self.mo_phase.dtype, weights.dtype,
        )
        tdm1s_block = np.zeros(
            (2, self.nkpts, self.nmo, self.nmo), dtype=dtype,
        )
        active = slice(self.ncore, self.nocc)
        tdm1s_block[:, :, active, active] = tdm1s_active_block
        return tdm1s_block

    def _transition_cumulant_to_block_fock(self, tcm2):
        """Transform and contract a Wannier CI transition cumulant.

        Each transformed block retains bra-ket-bra-ket order and therefore
        uses ``k1 - k2 + k3 - k4 = G``. The returned generalized-Fock
        contribution has shape ``(nkpts, nmo, nmo)`` and is nonzero only in
        its active columns. As with the transition 1-RDM transformation, no
        orbital-Hessian factor of two is applied here.
        """
        tcm2 = np.asarray(tcm2)
        _check_shape(
            tcm2, (self.ncastot,) * 4,
            label="transition_cumulant",
        )
        _check_shape(
            self.mo_phase,
            (self.nkpts, self.ncas, self.ncastot),
            label="mo_phase",
        )
        dtype = np.result_type(
            tcm2.dtype, self.mo_phase.dtype, self.eri_cas.dtype,
        )
        fock_response = np.zeros(
            (self.nkpts, self.nmo, self.nmo), dtype=dtype,
        )
        active = slice(self.ncore, self.nocc)
        kconserv = kpts_helper.get_kconserv(
            self.las._scf.cell, self.kpts,
        )
        for k1, k2, k3 in kpts_helper.loop_kkk(self.nkpts):
            k4 = kconserv[k1, k2, k3]
            tcm2_kpts = _get_casdm2_kpts(
                tcm2, self.mo_phase, (k1, k2, k3, k4),
            )
            _check_shape(
                tcm2_kpts, (self.ncas,) * 4,
                label=f"transition_cumulant_kpts[{k1},{k2},{k3}]",
            )
            paaa = self.eri_paaa(k1, k2, k3)
            _check_shape(
                paaa,
                (self.nmo, self.ncas, self.ncas, self.ncas),
                label=f"paaa[{k1},{k2},{k3}]",
            )
            fock_response[k1][:, active] += np.tensordot(
                paaa, tcm2_kpts,
                axes=((1, 2, 3), (1, 2, 3)),
            )
        return fock_response

    def get_h1eff_response(self, tdm1rs):
        """Build the effective one-electron response from other cells.

        This linearizes the Coulomb/exchange part of the fragment projection.
        For each output cell, its own transition-density contribution is
        subtracted, leaving the different-cell CI response.
        """
        tdm1rs = np.asarray(tdm1rs)
        _check_shape(
            tdm1rs, (self.nroots, 2, self.ncastot, self.ncastot),
            label="tdm1rs",
        )

        eri = self.eri_cas
        v1rs = np.tensordot(
            tdm1rs, eri, axes=((2, 3), (0, 1)),
        )
        v1rs += v1rs[:, ::-1]
        v1rs -= np.tensordot(
            tdm1rs, eri, axes=((2, 3), (2, 1)),
        )

        h1frs = []
        for ifrag, norb in enumerate(self.ncas_sub):
            i = int(np.sum(self.ncas_sub[:ifrag]))
            j = i + int(norb)
            dm1rs_i = tdm1rs[:, :, i:j, i:j]

            v1rs_i = np.tensordot(
                dm1rs_i,
                eri[i:j, i:j, :, :],
                axes=((2, 3), (0, 1)),
            )
            v1rs_i += v1rs_i[:, ::-1]
            v1rs_i -= np.tensordot(
                dm1rs_i,
                eri[:, i:j, i:j, :],
                axes=((2, 3), (2, 1)),
            )

            h1frs.append(
                v1rs[:, :, i:j, i:j] - v1rs_i[:, :, i:j, i:j]
            )

        return h1frs

    def ci_response_diag(self, ci1):
        """Apply the same-cell blocks of the CI Hessian.

        This is the complex generalization of the molecular CI-diagonal
        response. Both the input and output are projected relative to the
        current CI vector using the Hermitian inner product.
        """
        ci2 = self.Hci_all(
            [[-energy for energy in energy_r] for energy_r in self.e0],
            self.h1frs,
            self.eri_cas,
            ci1,
        )

        response = []
        for ci2_r, ci1_r, ci0_r, residual_r in zip(
                ci2, ci1, self.ci, self.hci0):
            response_r = []
            for hc1, c1, c0, residual in zip(
                    ci2_r, ci1_r, ci0_r, residual_r):
                output_overlap = np.vdot(residual, c1)
                input_overlap = np.vdot(c0, c1)
                response_r.append(2.0 * (
                    hc1
                    - output_overlap * c0
                    - input_overlap * residual
                ))
            response.append(response_r)

        return response

    def _get_Hci_diag(self):
        """Build the CI preconditioner diagonal in packed CSF coordinates."""
        hci_diag = []
        for ifrag, (fcibox, norb, nelec, h1rs, transformers) in enumerate(zip(
                self.fciboxes, self.ncas_sub, self.nelecas_sub,
                self.h1frs, self.ci_transformers)):
            if ifrag in self.frozen_ci:
                continue
            i = int(np.sum(self.ncas_sub[:ifrag]))
            j = i + int(norb)
            h2 = self.eri_cas[i:j, i:j, i:j, i:j]
            hdiag_csf_r = fcibox.states_make_hdiag_csf(
                h1rs, h2, norb, nelec,
            )
            if len(hdiag_csf_r) != len(transformers):
                msg = (
                    f"cell {ifrag} produced {len(hdiag_csf_r)} Hamiltonian "
                    f"diagonals for {len(transformers)} CI roots"
                )
                raise ValueError(msg)

            for iroot, (transformer, hdiag_csf) in enumerate(zip(
                    transformers, hdiag_csf_r)):
                hdiag_csf = np.asarray(transformer.pack_csf(hdiag_csf))
                if hdiag_csf.size != transformer.ncsf:
                    msg = (
                        f"cell {ifrag}, root {iroot} packed Hamiltonian "
                        f"diagonal has size {hdiag_csf.size}; expected "
                        f"{transformer.ncsf}"
                    )
                    raise ValueError(msg)

                hci_diag.append(hdiag_csf.reshape(-1))

        return hci_diag

    def _get_Horb_diag_matvec(self):
        """Return the reference orbital diagonal from Hessian matvecs.

        The periodic orbital-orbital response is complex and uses disk-backed
        ERIs. Reconstructing its diagonal from unit orbital directions gives
        an exact reference for the analytic preconditioner, including the
        ``kappa2/2`` packing convention. The result is cached because the
        Hessian intermediates are immutable for the operator's lifetime.
        """
        cached = getattr(self, "_Horb_diag_matvec_cache", None)
        if cached is not None:
            return np.array(cached, copy=True)

        nvar_orb = self.ugg.nvar_orb
        if nvar_orb == 0:
            diagonal = np.empty(0, dtype=np.complex128)
        else:
            diagonal = np.empty(nvar_orb, dtype=np.complex128)
            unit = np.zeros(nvar_orb, dtype=np.complex128)
            for index in range(nvar_orb):
                unit[index] = 1.0
                kappa = self.ugg.unpack_orb(unit)
                response = np.asarray(
                    self._orbital_hessian_response(kappa)
                )
                _check_shape(
                    response, np.shape(kappa),
                    label=f"orbital_hessian_response[{index}]",
                )
                packed_response = np.asarray(
                    self.ugg.pack_orb(response / 2.0)
                ).reshape(-1)
                _check_shape(
                    packed_response, (nvar_orb,),
                    label=f"packed_orbital_hessian_response[{index}]",
                )
                diagonal[index] = packed_response[index]
                unit[index] = 0.0

        self._Horb_diag_matvec_cache = np.array(diagonal, copy=True)
        return diagonal

    def _get_Horb_diag_external(self):
        """Return analytic diagonals for block-MO external rotations.

        This is the momentum-resolved counterpart of the molecular
        core-active, core-virtual, and active-virtual diagonal formulas. It
        follows the complex periodic construction in ``mc1step.gen_g_hop``
        but contracts only the ``(p,u,p,u)`` elements needed by the diagonal,
        rather than materializing its three large ``hdm2`` tensors.

        The returned vector follows the external prefix of the UGG ordering.
        Active-active coordinates are handled separately.
        """
        cached = getattr(self, "_Horb_diag_external_cache", None)
        if cached is not None:
            return np.array(cached, copy=True)

        nkpts = self.nkpts
        nmo = self.nmo
        ncore = self.ncore
        nocc = self.nocc
        ncas = self.ncas
        active = slice(ncore, nocc)
        dtype = np.result_type(
            self.hcore.dtype, self.h1s.dtype, self.dm1s.dtype,
            self.fock1.dtype, self.casdm2.dtype, np.complex128,
        )

        dm1 = np.asarray(self.dm1s.sum(axis=0), dtype=dtype)
        casdm1_kpts = dm1[:, active, active]
        vhf_c = np.asarray(self.eris.vhf_c, dtype=dtype)
        _check_shape(vhf_c, (nkpts, nmo, nmo), label="vhf_c")
        # The spin average is J[D] - K[D]/2, which is the spin-free potential
        # entering the periodic CASSCF diagonal even when the stored LASSCF
        # effective potentials are spin resolved.
        vhf_ca = (self.h1s[0] + self.h1s[1]) / 2.0 - self.hcore

        j_pc = np.asarray(self.eris.j_pc, dtype=dtype)
        k_pc = np.asarray(self.eris.k_pc, dtype=dtype)
        _check_shape(j_pc, (nkpts, nmo, ncore), label="j_pc")
        _check_shape(k_pc, (nkpts, nmo, ncore), label="k_pc")

        kconserv = kpts_helper.get_kconserv(
            self.las._scf.cell, self.kpts,
        )
        casdm2_kpts = {}

        def get_casdm2(k1, k2, k3):
            key = (k1, k2, k3)
            if key not in casdm2_kpts:
                k4 = kconserv[k1, k2, k3]
                casdm2_kpts[key] = _get_casdm2_kpts(
                    self.casdm2, self.mo_phase, (k1, k2, k3, k4),
                )
            return casdm2_kpts[key]

        jkcaa = np.zeros((nkpts, nocc, ncas), dtype=dtype)
        for k in range(nkpts):
            ppaa = self.eris.ppaa(k, k, k)[:nocc, :nocc]
            paap = self.eris.paap(k, k, k)[:nocc, :, :, :nocc]
            papa = self.eris.papa(k, k, k)[:nocc, :, :nocc]
            bra_ket_pair = -2.0 * np.einsum(
                "ppuv,uv->pu", ppaa, casdm1_kpts[k], optimize=True,
            )
            bra_ket_pair += 4.0 * np.einsum(
                "puvp,uv->pu", paap, casdm1_kpts[k], optimize=True,
            )
            # For complex Bloch orbitals, the two members of the conjugate
            # bra/ket pair contribute through their Hermitian average.
            jkcaa[k] += bra_ket_pair.real
            jkcaa[k] += 2.0 * np.einsum(
                "pupv,uv->pv", papa, casdm1_kpts[k], optimize=True,
            )

        hdm2_diag = np.zeros((nkpts, nmo, ncas), dtype=dtype)
        for k in range(nkpts):
            for kw in range(nkpts):
                # papa[p,w,q,x] D2[u,w,v,x] -> (p,u,q,v), with
                # k labels (k,kw,k,kw).
                papa = self.eris.papa(k, kw, k)
                dm2 = get_casdm2(k, kw, k)
                hdm2_diag[k] += np.einsum(
                    "pwpx,uwux->pu", papa, dm2, optimize=True,
                )

                # ppaa[k,k,kw,kw] D2[kw,kw,k,k] -> (p,u,q,v).
                ppaa = self.eris.ppaa(k, k, kw)
                dm2 = get_casdm2(kw, kw, k)
                hdm2_diag[k] += np.einsum(
                    "ppwx,wxuu->pu", ppaa, dm2, optimize=True,
                ).conj()

                # paap[p,w,x,q] D2[u,w,x,v] -> (p,u,q,v), with
                # k labels (k,kw,kw,k).
                paap = self.eris.paap(k, kw, kw)
                dm2 = get_casdm2(k, kw, kw)
                hdm2_diag[k] += np.einsum(
                    "pwxp,uwxu->pu", paap, dm2, optimize=True,
                ).conj()

        hdiag = np.zeros((nkpts, nmo, nmo), dtype=dtype)
        for k in range(nkpts):
            one_body = np.einsum(
                "ii,jj->ij", self.hcore[k], dm1[k], optimize=True,
            )
            one_body -= self.hcore[k] * dm1[k]
            hdiag[k] = one_body + one_body.conj().T

            fock_diag = self.fock1[k].diagonal().real
            hdiag[k] -= fock_diag + fock_diag[:, None]
            diagonal_indices = np.arange(nmo)
            hdiag[k][diagonal_indices, diagonal_indices] += 2.0 * fock_diag

            potential_diag = vhf_ca[k].diagonal().real
            hdiag[k][:, :ncore] += 2.0 * potential_diag[:, None]
            hdiag[k][:ncore] += 2.0 * potential_diag
            core_indices = np.arange(ncore)
            hdiag[k][core_indices, core_indices] -= (
                4.0 * potential_diag[:ncore]
            )

            core_active = np.einsum(
                "ii,jj->ij", vhf_c[k], casdm1_kpts[k],
                optimize=True,
            )
            hdiag[k][:, active] += core_active
            hdiag[k][active, :] += core_active.conj().T
            active_active = -vhf_c[k][active, active] * casdm1_kpts[k]
            hdiag[k][active, active] += (
                active_active + active_active.conj().T
            )

            core_eri = 6.0 * k_pc[k] - 2.0 * j_pc[k]
            hdiag[k][ncore:, :ncore] += core_eri[ncore:]
            hdiag[k][:ncore, ncore:] += core_eri[ncore:].conj().T

            hdiag[k][:nocc, active] -= jkcaa[k]
            hdiag[k][active, :nocc] -= jkcaa[k].conj().T
            hdiag[k][:, active] += hdm2_diag[k]
            hdiag[k][active, :] += hdm2_diag[k].conj().T

        diagonal = np.asarray(
            hdiag[self.ugg.uniq_orb_idx], dtype=dtype,
        ).reshape(-1) / 2.0
        _check_shape(
            diagonal, (self.ugg.nvar_orb_external,),
            label="external_orbital_hessian_diagonal",
        )
        self._Horb_diag_external_cache = np.array(diagonal, copy=True)
        return diagonal

    @property
    def shape(self):
        """tuple: Shape of the combined orbital/CI Hessian operator."""
        return self.ugg.nvar_tot, self.ugg.nvar_tot

    def _unpack_ci_vector(self, x):
        """Transform packed complex CSF coefficients to determinant arrays.

        Frozen fragments receive zero determinant response vectors with the
        shapes of their reference CI vectors.

        Args:
            x : array-like of shape (nvar_ci,)
                Packed CSF response coefficients for nonfrozen fragments.

        Returns:
            list
                Nested ``[fragment][root]`` determinant-basis responses.
        """
        x_flat = np.asarray(x).reshape(-1)
        if x_flat.size != self.nvar_ci:
            raise ValueError(
                f"trial vector has size {x_flat.size}; expected "
                f"{self.nvar_ci}"
            )

        ci1 = []
        offset = 0
        for ifrag, (transformers, ci0_r) in enumerate(zip(
                self.ci_transformers, self.ci)):
            ci1_r = []
            for transformer, c0 in zip(transformers, ci0_r):
                if ifrag in self.frozen_ci:
                    ci1_r.append(np.zeros_like(c0))
                    continue
                ncsf = transformer.ncsf
                c1 = cplx_csf_helper.vec_csf2det_cplx(
                    transformer, x_flat[offset:offset + ncsf],
                    normalize=False,
                )
                ci1_r.append(np.asarray(c1).reshape(np.shape(c0)))
                offset += ncsf
            ci1.append(ci1_r)
        if offset != x_flat.size:
            raise ValueError(
                f"consumed {offset} CSF coefficients from a vector of size "
                f"{x_flat.size}"
            )
        return ci1

    def _flatten_ci_vector(self, ci):
        """Transform determinant-array responses to packed complex CSFs.

        Args:
            ci : sequence
                Nested ``[fragment][root]`` determinant-basis responses.

        Returns:
            ndarray of shape (nvar_ci,)
                Packed CSF coefficients with frozen fragments omitted.
        """
        if len(ci) != len(self.ci_transformers):
            raise ValueError("CI response must contain one entry per cell")
        vectors = []
        for ifrag, (transformers, ci_r) in enumerate(zip(
                self.ci_transformers, ci)):
            if len(transformers) != len(ci_r):
                raise ValueError(
                    f"cell {ifrag} has {len(ci_r)} CI responses for "
                    f"{len(transformers)} roots"
                )
            if ifrag in self.frozen_ci:
                continue
            for transformer, c0 in zip(transformers, ci_r):
                c0_csf = cplx_csf_helper.vec_det2csf_cplx(
                    transformer, c0, normalize=False,
                )
                vectors.append(np.asarray(c0_csf).reshape(-1))
        if not vectors:
            return np.empty(0, dtype=np.complex128)
        return np.concatenate(vectors)

    def _zero_ci_step(self, dtype):
        """Return zero determinant vectors with the reference CI layout."""
        return [
            [
                np.zeros_like(c0, dtype=np.result_type(c0, dtype))
                for c0 in ci0_r
            ]
            for ci0_r in self.ci
        ]

    @staticmethod
    def _ci_step_is_zero(ci1):
        """Return whether every determinant coefficient in ``ci1`` is zero."""
        return not any(np.any(c1) for ci1_r in ci1 for c1 in ci1_r)


# Install only the parameterization interface at this layer. Gradient methods
# are registered alongside their respective function definitions.
for _klass in (PBCLASCINoSymm, PBCLASCITransSymm):
    _klass._klasscf_eris = _ERIS
    _klass._ugg = KLASSCF_UnitaryGroupGenerators
    _klass.get_ugg = get_ugg
    _klass.get_grad_ci = get_grad_ci
    _klass.get_grad_orb = get_grad_orb
    _klass.get_grad = get_grad
