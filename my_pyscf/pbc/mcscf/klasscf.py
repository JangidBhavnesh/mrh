#!/usr/bin/env python

import numpy as np


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


