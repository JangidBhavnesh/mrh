#!/usr/bin/env python

import numpy as np

from mrh.my_pyscf.pbc.fci import cplx_csf_helper
from mrh.my_pyscf.pbc.mcscf.klasci import (
    PBCLASCINoSymm,
    PBCLASCITransSymm,
)
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
            Name used to identify ``mat`` in the error message.

    Raises:
        ValueError
            If the shape of ``mat`` differs from ``shape``.
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

    For a selected Bloch pair ``(k, a, b)`` and Wannier pair ``(p, q)``,
    ``pair_map`` contains

    ``mo_phase[k, a, p] * mo_phase[k, b, q].conj()``.

    The singular vectors spanning the image of this map define the independent
    active-active coordinates used by the k-LASSCF unitary-group generator.
    The map is complex-linear; anti-Hermitian completion is applied only by
    :meth:`unpack`, after the lower-pair coordinates have been expanded.

    Args:
        mo_phase : ndarray of shape (nkpts, ncas, ncastot)
            Unitary transformation from the complete Wannier active space to
            the active Bloch MOs at each k-point. ``ncastot`` must equal
            ``nkpts * ncas``.
        ncas_sub : array-like of int
            Numbers of active orbitals assigned to the LAS fragments. Their
            sum must equal ``ncastot``; their order defines the Wannier
            fragment partition.
        block_pair_mask : ndarray of bool, optional
            Mask of shape ``(nkpts, ncas, ncas)`` selecting the strictly
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
            Orthonormal basis for the image of ``pair_map``. Its number of
            columns is :attr:`nvar`.
        singular_values : ndarray
            Singular values of ``pair_map`` in descending order.
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
        if np.any(ncas_sub < 0):
            raise ValueError("ncas_sub entries must be nonnegative")
        if svd_tol is not None:
            svd_tol = float(svd_tol)
            if not np.isfinite(svd_tol) or svd_tol < 0:
                raise ValueError("svd_tol must be finite and nonnegative")

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
        if not np.all(np.isfinite(mo_phase)):
            raise ValueError("mo_phase must contain only finite values")

        phase_matrix = mo_phase.reshape(self.ncastot, self.ncastot)
        identity = np.eye(self.ncastot, dtype=phase_matrix.dtype)
        if not (
                np.allclose(
                    phase_matrix.conj().T @ phase_matrix, identity,
                    atol=1e-8, rtol=1e-8,
                ) and np.allclose(
                    phase_matrix @ phase_matrix.conj().T, identity,
                    atol=1e-8, rtol=1e-8,
                )):
            raise ValueError("stacked mo_phase must be unitary")

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
        lower_triangle = np.broadcast_to(
            np.tril(np.ones((self.ncas, self.ncas), dtype=bool), -1),
            block_pair_mask.shape,
        )
        if np.any(block_pair_mask & ~lower_triangle):
            raise ValueError(
                "block_pair_mask may select only strictly lower-triangular "
                "active pairs"
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
                selected by ``block_pair_mask`` are read.

        Returns:
            ndarray of shape (nvar,)
                Coordinates in the orthonormal image of ``pair_map``.
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
                Coordinates in the orthonormal image of ``pair_map``.

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
            Bloch-MO coefficients. Defaults to ``klas.mo_coeff``.
        ci : sequence, optional
            Nested ``[fragment][root]`` determinant-basis CI vectors. Defaults
            to ``klas.ci``.
        mo_phase : ndarray of shape (nkpts, ncas, nkpts*ncas), optional
            Bloch-active to Wannier-active transformation. It defaults to
            ``klas.mo_phase`` when available and is otherwise constructed from
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
                Nested ``[fragment][root]`` determinant-basis CI vectors.

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
                Nested ``[fragment][root]`` determinant-basis CI responses.
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

    The construction is dispatched through ``klas._ugg`` so that periodic
    LAS subclasses can replace the parameterization without overriding this
    convenience method.

    Args:
        klas : object
            Periodic LAS object for which the parameterization is built.
        mo_coeff : ndarray of shape (nkpts, nao, nmo), optional
            Bloch-MO coefficients. Defaults to ``klas.mo_coeff`` in the
            generator constructor.
        ci : sequence, optional
            Nested ``[fragment][root]`` determinant-basis CI vectors. Defaults
            to ``klas.ci`` in the generator constructor.
        mo_phase : ndarray of shape (nkpts, ncas, nkpts*ncas), optional
            Bloch-active to Wannier-active transformation. When omitted, the
            generator obtains or constructs it from ``klas``.

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

    ``2 * (H c - <c|H c> c)``.

    Constructing the residual directly keeps the gradient layer independent
    of the k-LASSCF Hessian operator.

    Args:
        klas : object
            Periodic LAS object supplying fragment solvers and active-space
            integral builders.
        mo_coeff : ndarray of shape (nkpts, nao, nmo), optional
            Bloch-MO coefficients. Defaults to ``klas.mo_coeff``.
        ci : sequence, optional
            Nested ``[fragment][root]`` determinant-basis CI vectors. Defaults
            to ``klas.ci``.
        ugg : KLASSCF_UnitaryGroupGenerators, optional
            Accepted for compatibility with the combined-gradient interface;
            it is not needed to form determinant-basis CI residuals.
        casdm1frs : sequence, optional
            Fragment- and root-resolved active-space one-particle density
            matrices used to build ``h1eff`` when it is not supplied.
        h1eff : sequence, optional
            Effective one-electron Hamiltonians for each fragment, with each
            block shaped ``(nroots, 2, ncas_frag, ncas_frag)``.
        h2eff : ndarray of shape (ncastot,)*4, optional
            Two-electron integrals in the complete Wannier active space.

    Returns:
        list
            Nested ``[fragment][root]`` determinant-basis CI gradients with
            the same individual shapes as ``ci``.

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


# Install only the parameterization interface at this layer. Gradient methods
# are registered alongside their respective function definitions.
for _klass in (PBCLASCINoSymm, PBCLASCITransSymm):
    _klass._ugg = KLASSCF_UnitaryGroupGenerators
    _klass.get_ugg = get_ugg
    _klass.get_grad_ci = get_grad_ci
