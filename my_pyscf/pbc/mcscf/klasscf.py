#!/usr/bin/env python

import numpy as np

from pyscf.pbc.lib import kpts_helper

from mrh.my_pyscf.mcscf.lasscf_sync_o0 import (
    LASSCF_HessianOperator as molLASSCF_HessianOperator,
)
from mrh.my_pyscf.pbc.fci import cplx_csf_helper
from mrh.my_pyscf.pbc.mcscf.mc1step import _get_casdm2_kpts
from mrh.my_pyscf.pbc.mcscf.klas_ao2mo import _ERIS
from mrh.my_pyscf.pbc.mcscf.klasci import (
    PBCLASCINoSymm,
    PBCLASCITransSymm,
)
from mrh.my_pyscf.pbc.util.wannier import get_wannier_orbs


def _check_shape(mat, shape, label="array"):
    """
    Raise ``ValueError`` when ``mat`` does not have ``shape``.
    """
    shape = tuple(shape)
    if np.shape(mat) != shape:
        msg = (
            f"{label} has shape {np.shape(mat)}; expected {shape}"
        )
        raise ValueError(msg)


class ActiveActiveRotationMap:
    """Map inter-fragment Wannier pairs into k-point active pairs.

    The orbital optimizer stores active rotations in block-MO form, whereas
    the LAS fragment partition is defined in the complete Wannier active
    space.  ``pair_map`` transforms the independent, directed lower-pair
    coordinates selected by the molecular LASSCF fragment mask into the
    lower-triangular active pairs available at each k-point.  Its column space
    is compressed to an orthonormal basis so that projected directions are
    included once and only once in the periodic UGG vector.

    The pair map is complex linear.  Anti-Hermitian completion is deliberately
    applied only after projected block-pair coordinates are unpacked.
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
        """Number of independent projected active-active coordinates."""
        return self.basis.shape[1]

    def block_to_wannier(self, kappa_active):
        """Transform block-diagonal active matrices to Wannier form."""
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
        """Return the k-diagonal blocks of a Wannier active matrix."""
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
        """Project block active rotations onto independent coordinates."""
        kappa_active = np.asarray(kappa_active)
        _check_shape(
            kappa_active, (self.nkpts, self.ncas, self.ncas),
            label="kappa_active",
        )
        block_pairs = np.asarray(kappa_active[self.block_pair_idx])
        return np.asarray(self.basis.conj().T @ block_pairs).reshape(-1)

    def unpack(self, coordinates):
        """Expand projected coordinates as block anti-Hermitian rotations."""
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
    """Pack k-point orbital rotations and Wannier-basis CI variations.

    External orbital variables are ordered by k-point and then by the ordinary
    nonredundant CASSCF mask.  Independent active-active directions projected
    from the inter-fragment Wannier pair space follow.  CI variables are last,
    in cell/root order, and are transformed between determinant and CSF
    representations.
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
            mo_coeff, (self.nkpts, mo_coeff.shape[1], mo_coeff.shape[2]), label="mo_coeff"
        )

        ncore = klas.ncore
        ncas = klas.ncas
        self.ncore = ncore
        nocc = ncore + ncas
        orb_idx = np.zeros((self.nmo, self.nmo), dtype=bool)
        orb_idx[ncore:nocc, :ncore] = True
        orb_idx[nocc:, :nocc] = True
        nonfrozen = np.ones(self.nmo, dtype=bool)

        # Keeping the frozen as per molecular version, but have not been
        # tested yet.
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
            orb_idx, (self.nkpts, self.nmo, self.nmo),).copy()
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
        """Number of ordinary core/active/virtual block rotations."""
        return int(np.count_nonzero(self.uniq_orb_idx))

    @property
    def nvar_orb_active_active(self):
        """Number of projected inter-fragment active-active rotations."""
        return self.active_active_map.nvar

    @property
    def nvar_orb(self):
        return self.nvar_orb_external + self.nvar_orb_active_active

    @property
    def ncsf_sub(self):
        return np.asarray([
            [transformer.ncsf for transformer in transformers]
            for ifrag, transformers in enumerate(self.ci_transformers)
            if ifrag not in self.frozen_ci
        ], dtype=int)

    @property
    def nvar_ci(self):
        return int(self.ncsf_sub.sum())

    @property
    def nvar_tot(self):
        # the total number of non-redundant orbital and CI variables
        return self.nvar_orb + self.nvar_ci

    def get_gx_idx(self):
        """k-LASSCF currently optimizes every nonredundant orbital variable."""
        return np.zeros_like(self.uniq_orb_idx)

    def pack_orb(self, kappa):
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
        x_orb = self.pack_orb(kappa)
        x_ci = self.pack_ci(ci)
        dtype = np.result_type(x_orb.dtype, x_ci.dtype)
        x = np.empty(self.nvar_tot, dtype=dtype)
        x[:self.nvar_orb] = x_orb
        x[self.nvar_orb:] = x_ci
        return x

    def unpack(self, x):
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
    return klas._ugg(
        klas, mo_coeff=mo_coeff, ci=ci, mo_phase=mo_phase,
    )

def get_grad_ci(klas, mo_coeff=None, ci=None, ugg=None, 
                casdm1frs=None, h1eff=None, h2eff=None):
    """Return the complex determinant-basis CI energy gradient."""
    if mo_coeff is None:
        mo_coeff = klas.mo_coeff
    if ci is None:
        ci = klas.ci
    if ugg is None:
        ugg = klas.get_ugg(mo_coeff=mo_coeff, ci=ci)

    hop = KLASSCF_HessianOperator(
        klas, ugg, mo_coeff=mo_coeff, ci=ci, casdm1frs=casdm1frs,
        h1eff=h1eff, h2eff=h2eff,
    )
    return [[2.0 * residual for residual in residual_r]
            for residual_r in hop.hci0]

def get_grad_orb (klas, mo_coeff_kpts=None, ci=None, h2eff_sub=None, 
                  veff_kpts=None, dm1s_kpts=None, hermi=-1):
    '''Return energy gradient for orbital rotation.
    
    Note: this function expects arrays of different sizes than
    what is expected by the molecular version of the same function.

    Args:
        klas : instance of :class:`KLASSCF`

    Kwargs:
        mo_coeff_kpts : ndarray of shape (nkpts,nao,nmo)
            Contains molecular orbitals
        ci : list (length=nfrags) of list (length=nroots) of ndarray
            Contains CI vectors (# Note the CI vectors are in the Wannier basis)
        h2eff_sub : ndarray or :class:`_ERIS`
            Either k-LASSCF AO2MO intermediates or ``paaa`` integrals with
            shape ``(nkpts,nkpts,nkpts,nmo,ncas,ncas,ncas)``.
        veff_kpts : ndarray of shape (2,nkpts,nao,nao)
            Spin-separated, state-averaged 1-electron mean-field potential in the AO basis
        dm1s_kpts : ndarray of shape (2,nkpts,nao,nao)
            Spin-separated, state-averaged 1-RDM in the AO basis
        hermi : integer
            Control (anti-)symmetrization. 0 means to return the effective Fock matrix,
            F1 = h.D + g.d. -1 means to return the true orbital-rotation gradient, which is skew-
            symmetric: gorb = F1 - F1.conj().T. +1 means to return the symmetrized effective Fock matrix,
            (F1 + F1.conj().T) / 2. The factor of 2 difference between hermi=-1 and the other two options
            is intentional and necessary.

    Returns:
        gorb : ndarray of shape (nkpts,nmo,nmo)
            Orbital rotation gradients as a square antihermitian array
    '''

    cell = klas._scf.cell
    kpts = klas.kpts
    nkpts = len(kpts)

    if mo_coeff_kpts is None:
        mo_coeff_kpts = klas.mo_coeff
    if ci is None: ci = klas.ci
    if dm1s_kpts is None:
        dm1s_kpts = klas.make_rdm1s (mo_coeff=mo_coeff_kpts, ci=ci)
    if h2eff_sub is None:
        h2eff_sub = klas._klasscf_eris(klas, mo_coeff_kpts)
    if veff_kpts is None:
        veff_kpts = klas.get_veff (cell, dm_kpts=dm1s_kpts)

    nao, nmo = mo_coeff_kpts.shape[-2:]
    ncore = klas.ncore
    ncas = klas.ncas
    nocc = klas.ncore + klas.ncas
    ncastot = nkpts * ncas


    get_paaa = getattr(h2eff_sub, 'paaa', None)
    if get_paaa is None:
        _check_shape(
            h2eff_sub,
            (nkpts, nkpts, nkpts, nmo, ncas, ncas, ncas),
            label="h2eff_sub",
        )
        get_paaa = lambda k1, k2, k3: h2eff_sub[k1, k2, k3]

    dtype = np.result_type(mo_coeff_kpts.dtype, veff_kpts.dtype,
                            dm1s_kpts.dtype)

    ovlp_kpts = klas._scf.get_ovlp (kpts=kpts)
    hcore_kpts = klas.get_hcore (kpts=kpts)
    h1es_kpts = hcore_kpts[None,:,:,:] + veff_kpts
    hcore_kpts = veff_kpts = None

    f1 = np.empty((nkpts, nmo, nmo), dtype=dtype)

    for k in range(nkpts):
        smo_coeff_k = ovlp_kpts[k] @ mo_coeff_kpts[k]
        smo_coeff_k_H = smo_coeff_k.conjugate ().T
        mo_coeff_k = mo_coeff_kpts[k]
        mo_coeff_k_H = mo_coeff_k.conjugate ().T
        dm1s_mo = smo_coeff_k_H @ dm1s_kpts[:,k] @ smo_coeff_k
        h1es_mo = mo_coeff_k_H @ h1es_kpts[:,k] @ mo_coeff_k
        f1[k] = h1es_mo[0] @ dm1s_mo[0] + h1es_mo[1] @ dm1s_mo[1] 

    smo_coeff_k = smo_coeff_k_H = mo_coeff_k = mo_coeff_k_H = None

    # It's in Wannier basis:
    casdm2 = klas.make_casdm2 (ci=ci)
    _check_shape(casdm2, (ncastot,) * 4, label="casdm2")

    # Currently, it's formed by transforming the dm1s, but it would be wiser to just reconstruct it.
    casdm1s = klas.make_casdm1s (ci=ci)
    _check_shape(casdm1s, (2, ncastot, ncastot), label="casdm1s")
    casdm1 = casdm1s.sum (0)
    casdm2 -= np.multiply.outer (casdm1, casdm1)
    casdm2 += np.multiply.outer (casdm1s[0], casdm1s[0]).transpose (0,3,2,1)
    casdm2 += np.multiply.outer (casdm1s[1], casdm1s[1]).transpose (0,3,2,1)

    casdm1 = casdm1s = None

    mo_act_kpts = mo_coeff_kpts[:, :, ncore:nocc]
    mo_phase = get_wannier_orbs(klas._scf, klas.kmesh, mo_act_kpts,)[-1]

    kconserv = kpts_helper.get_kconserv(cell, kpts)

    for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
        k4 = kconserv[k1, k2, k3]
        casdm2_kpts = _get_casdm2_kpts(
            casdm2, mo_phase, (k1, k2, k3, k4),
        )
        paaa_kpts = get_paaa(k1, k2, k3)
        f1[k1][:,ncore:nocc] += np.tensordot (paaa_kpts, casdm2_kpts,
                                             axes=((1,2,3),(1,2,3)))

    casdm2 = None

    if hermi == -1: return f1 - f1.conj().transpose(0,2,1)
    elif hermi == 1: return .5*(f1+f1.conj().transpose(0,2,1))
    elif hermi == 0: return f1
    else: raise ValueError ("kwarg 'hermi' must = -1, 0, or +1")


def get_grad(
        klas, mo_coeff=None, ci=None, ugg=None, h2eff_sub=None,
        veff_kpts=None, dm1s_kpts=None, casdm1frs=None,
        h1eff=None, h2eff=None):
    """Return the packed total ``[orbital, CI]`` k-LASSCF gradient."""
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
    """Periodic Hessian operator for k-LASSCF.

    The operator retains one CI vector for every cell and root and returns a
    Hessian action with the same nested layout after flattening.

    Parameters
    ----------
    las
        Periodic LASCI/LASSCF object.
    ugg
        Unitary-group generator object used to pack and unpack trial vectors.
    mo_coeff, ci
        Current orbitals and CI vectors.  They default to the corresponding
        attributes of ``las``.
        mo_coeff : list or np.ndarray of shape (nkpts, nao, nmo) (in bloch representation)
        ci : list or np.ndarray of shape (nroots, ncastot, ncastot) (in wannier basis)
    casdm1frs, casdm2fr
        Optional precomputed fragment RDMs in the Wannier basis.
    h1eff, h2eff
        Optional precomputed local one-electron Hamiltonians and full Wannier
        active-space two-electron integrals.
    eris
        Optional block-MO periodic ERI object.  When omitted, level-2
        ``ppaa``, ``papa``, and ``paap`` intermediates are built on disk.
    veff_kpts, dm1s_kpts
        Optional spin-resolved AO potential and density at each k-point.
    mo_phase
        Optional transformation from the Wannier active basis to each
        k-point active block.
    kpts, kmesh
        BvK k-points and k-point mesh.  They default to ``las.kpts`` and
        ``las.kmesh`` and must describe the same number of cells.

    Notes
    -----
    The molecular parent constructor is intentionally not called because its
    orbital-response intermediates assume a single molecular MO basis.  Here,
    active-space CI tensors are retained in the Wannier basis, while orbital
    tensors and the disk-backed ``ppaa``, ``papa``, and ``paap`` integrals are
    retained in the block-MO k-point basis.  The orbital Hessian response is
    evaluated in that block-MO basis without materializing a dense supercell
    ERI tensor.
    """

    def __init__(
            self, las, ugg, mo_coeff=None, ci=None, casdm1frs=None,
            h1eff=None, h2eff=None, kpts=None, kmesh=None, casdm2fr=None,
            eris=None, veff_kpts=None, dm1s_kpts=None, mo_phase=None):

        if mo_coeff is None: mo_coeff = las.mo_coeff
        if ci is None: ci = las.ci
        if kpts is None: kpts = las.kpts
        if kmesh is None: kmesh = las.kmesh
        kpts = np.asarray(kpts)
        kmesh = tuple(int(n) for n in kmesh)

        if len(kmesh) != 3 or any(n <= 0 for n in kmesh):
            raise ValueError("kmesh must contain three positive integers")
        ncell = int(np.prod(kmesh))
        if len(kpts) != ncell:
            raise ValueError(f"kpts and kmesh are inconsistent: {len(kpts)} != {ncell}")
        
        self.las = las
        self.ugg = ugg
        self.mo_coeff = mo_coeff
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
        self.nao = mo_coeff.shape[-2]
        self.nmo = mo_coeff.shape[-1]
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
            msg = (f"ugg.ci_transformers must contain one entry per CI cell; ")
            raise ValueError(msg)
        self.nvar_ci = 0

        for ifrag, (transformers, ci0_r) in enumerate(
            zip(self.ci_transformers, self.ci)):
            if len(transformers) != len(ci0_r):
                msg = f"cell {ifrag} has {len(transformers)} CSF transformers \
                    for {len(ci0_r)} CI roots"
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

        ``casdm1s``, ``casdm2``, and ``cascm2`` use the complete Wannier
        active space.  ``dm1s_kpts`` is the AO density at each k-point and
        ``dm1s`` is its block-MO representation.
        """
        if casdm1frs is None:
            casdm1frs = self.las.states_make_casdm1s_sub(
                ci=self.ci,
                ncas_sub=self.ncas_sub,
                nelecas_sub=self.nelecas_sub,
            )

        self.casdm1frs = casdm1frs
        self.casdm1fs = self.las.make_casdm1s_sub(casdm1frs=casdm1frs,)
        self.casdm1rs = self.las.states_make_casdm1s(casdm1frs=casdm1frs,)
        self.casdm1s = np.einsum("r,rsij->sij", self.weights, self.casdm1rs,)

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
        _check_shape(self.casdm1s, (2, self.ncastot, self.ncastot), label="casdm1s")
        _check_shape(self.casdm2, (self.ncastot,) * 4, label="casdm2")

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
            label="dm1s_kpts"
        )

        ovlp_kpts = np.asarray(self.las._scf.get_ovlp(kpts=self.kpts))
        _check_shape(ovlp_kpts, (self.nkpts, self.nao, self.nao), label="ovlp_kpts")
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
        """
        Initialize block-MO one-electron and Wannier CI Hamiltonians.

        ``h1frs[f][r]`` is the spin-separated effective one-electron
        Hamiltonian for fragment/cell ``f`` and root ``r``.  ``eri_cas``
        contains the two-electron integrals over the complete Wannier active
        space.  In contrast, ``h1s`` and ``hcore`` have a k-point axis and use
        the block-MO basis.
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
            label="veff_kpts"
        )
        _check_shape(
            self.veff_kpts, (2, self.nkpts, self.nao, self.nao),
            label="veff_kpts"
        )

        hcore_kpts = np.asarray(self.las.get_hcore(kpts=self.kpts))
        _check_shape(hcore_kpts, (self.nkpts, self.nao, self.nao), label="hcore_kpts")
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
            msg = "h1eff must contain one block for every fragment/cell"
            raise ValueError(msg)
        
        for ifrag, (h1fr, ncas) in enumerate(zip(h1eff, self.ncas_sub)):
            _check_shape(h1fr, (self.nroots, 2, ncas, ncas), label=f"h1fr_{ifrag}")

        self.h1frs = h1eff
        self.eri_cas = h2eff

    def _init_eri_(self, eris=None):
        """Attach lazy block-MO ERI accessors used by orbital response.

        The default periodic ERI object stores ``ppaa``, ``papa``, and
        ``paap`` tensors on disk.  ``eri_paaa`` is deliberately an accessor
        into those block-MO tensors, not a materialized Wannier-basis array.
        Level one additionally constructs the compact ``j_pc`` and ``k_pc``
        intermediates required by the analytic core-orbital diagonal.
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
            label="mo_phase"
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
                label='paaa_kpts'
            )
            self.fock1[k1][:, active] += np.tensordot(
                paaa_kpts, cascm2_kpts,
                axes=((1, 2, 3), (1, 2, 3)),
            )

    def _init_ci_(self):
        """
        Cache local Hamiltonian actions, energies, and CI residuals.
        """
        self.linkstrl = []
        self.linkstr = []
        for fcibox, norb, nelec in zip(
                self.fciboxes, self.ncas_sub, self.nelecas_sub):
            # The complex periodic contractions require ordinary link tables,
            # without the molecular lower-triangular index packing.
            linkstr = fcibox.states_gen_linkstr(norb, nelec, False)
            self.linkstrl.append(linkstr)
            self.linkstr.append(linkstr)
        hc0 = self.Hci_all(None, self.h1frs, self.eri_cas, self.ci)
        self.e0 = [[np.vdot(c, hc) for c, hc in zip(ci_r, hc_r)] 
                   for ci_r, hc_r in zip(self.ci, hc0)]
        self.hci0 = [[hc - energy * c for hc, energy, c in zip(hc_r, e_r, ci_r)] 
                     for hc_r, e_r, ci_r in zip(hc0, self.e0, self.ci)]

    def make_tdm1s_sub(self, ci1):
        """
        Build the first-order spin 1-RDM generated by a CI step.

        The returned array has shape ``(nroots, 2, ncastot, ncastot)``. For a
        complex CI step, each one-sided transition density is combined with
        its Hermitian conjugate after removing the component parallel to the
        current CI vector.
        """
        return self._make_tdm1s2c_sub(ci1, with_cumulant=False)[0]

    def make_tdm1s2c_sub(self, ci1):
        """Build complex CI transition 1-RDMs and the effective cumulant.

        The root-resolved spin 1-RDMs and the state-averaged cumulant use the
        complete Wannier active space.  ``self.casdm1frs`` supplies the
        root/fragment reference densities needed for overlap removal, while
        ``self.casdm1s`` is used directly as the stored state-averaged full
        Wannier 1-RDM in the cumulant decomposition.

        Returns
        -------
        tdm1rs : ndarray
            Hermitian transition 1-RDMs with shape
            ``(nroots, 2, ncastot, ncastot)``.
        tcm2 : ndarray
            State-averaged effective transition cumulant with shape
            ``(ncastot, ncastot, ncastot, ncastot)``.
        """
        return self._make_tdm1s2c_sub(ci1, with_cumulant=True)

    def _make_tdm1s2c_sub(self, ci1, with_cumulant):
        """Implementation shared by the one- and two-body CI builders."""
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
            nelec_by_solver = [fcibox._get_nelec(solver, nelec)
                               for solver in fcibox.fcisolvers]

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
                # Some existing builds predate the compiled complex
                # transition-RDM symbols. Use the equivalent complex Python
                # implementation in that case.
                missing_symbol = str(err)
                if not any(name in missing_symbol for name in (
                        "FCItrans_rdm1", "FCItdm12")):
                    raise
                transition_rdm_r = list(fcibox._collect(
                    contraction + "_py", *collect_args, **collect_kwargs,
                ))

            if len(transition_rdm_r) != self.nroots:
                msg = (f"fragment {ifrag} produced {len(transition_rdm_r)} transition "
                       f"densities for {self.nroots} roots")
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
        reference-overlap subtraction.  ``tdm2_one_sided`` contains the
        explicitly correlated same-fragment transition blocks.  The
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

        # Complete the same-fragment transition 2-RDM blocks.  Starting from
        # half of <c1|Gamma|c0>, the first operation supplies its Hermitian
        # partner and the second supplies electron-pair exchange for both.
        tdm2 = np.array(tdm2_one_sided, copy=True)
        tdm2 += tdm2.conj().transpose(1, 0, 3, 2)
        tdm2 += tdm2.transpose(2, 3, 0, 1)

        # Differentiate the off-diagonal product-state blocks constructed by
        # LASCI.make_casdm2.  These terms vanish from the reference cumulant,
        # but their 2-RDM derivatives are required to cancel the corresponding
        # derivative of the mean-field products below.
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
                    dtype=np.result_type(tdm2.dtype, dm1s_i.dtype, dm1s_j.dtype),
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
        outside its active-active blocks.  This routine performs only state
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
        uses ``k1 - k2 + k3 - k4 = G``.  The returned generalized-Fock
        contribution has shape ``(nkpts, nmo, nmo)`` and is nonzero only in
        its active columns.  As with the transition 1-RDM transformation, no
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
            label="tdm1rs"
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
        """
        Build the CI preconditioner diagonal in packed CSF coordinates.
        """
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

        The periodic OO response is complex and uses disk-backed ERIs.
        Reconstructing its diagonal from unit orbital directions provides an
        exact regression reference for the analytic preconditioner, including
        the ``kappa2/2`` packing convention.  The result is cached because all
        Hessian intermediates are immutable for the lifetime of this operator.
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
        core-active, core-virtual, and active-virtual diagonal formulas.  It
        follows the complex periodic construction in ``mc1step.gen_g_hop``
        but contracts only the ``(p,u,p,u)`` elements needed by the diagonal,
        rather than materializing its three large ``hdm2`` tensors.

        The returned vector follows the external prefix of the UGG ordering.
        Active-active coordinates are evaluated separately in the Wannier
        basis by :meth:`_get_Horb_active_active`.
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
            # These two contractions represent both members of a conjugate
            # bra/ket pair in the real-orbital formula.  For complex Bloch
            # orbitals their Hermitian average, rather than either member
            # alone, contributes to a real-coordinate diagonal probe.
            jkcaa[k] += bra_ket_pair.real
            jkcaa[k] += 2.0 * np.einsum(
                "pupv,uv->pv", papa, casdm1_kpts[k], optimize=True,
            )

        hdm2_diag = np.zeros((nkpts, nmo, ncas), dtype=dtype)
        for k in range(nkpts):
            for kw in range(nkpts):
                # papa[p,w,q,x] D2[u,w,v,x] -> (p,u,q,v), with
                # k labels (k,kw,k,kw).  The two conjugations in the
                # mc1step diagonal cancel for this permutation.
                papa = self.eris.papa(k, kw, k)
                dm2 = get_casdm2(k, kw, k)
                hdm2_diag[k] += np.einsum(
                    "pwpx,uwux->pu", papa, dm2, optimize=True,
                )

                # ppaa[k,k,kw,kw] D2[kw,kw,k,k] -> (p,u,q,v).
                # The resulting outer labels obey the regrouped
                # k(p)+k(u)-k(q)-k(v)=G rule.
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

    def _get_Horb_diag(self):
        """Return the orbital diagonal in packed UGG ordering.

        Core-active, core-virtual, and active-virtual rotations use the
        analytic momentum-resolved block-MO diagonal.  The projected
        active-active slice uses the analytic Wannier Hessian transformed
        into UGG coordinates.  :meth:`_get_Horb_diag_matvec` remains available
        as an independent regression reference.
        """
        pieces = [self._get_Horb_diag_external()]
        nvar_active = getattr(self.ugg, "nvar_orb_active_active", 0)
        if nvar_active:
            hessian, hessian_conj = self._get_Horb_active_active()
            pieces.append(np.diag(hessian + hessian_conj))
        diagonal = np.concatenate(pieces)
        _check_shape(
            diagonal, (self.ugg.nvar_orb,),
            label="orbital_hessian_diagonal",
        )
        return diagonal

    def _get_Hdiag(self):
        """Return the full orbital-plus-CI diagonal in packed UGG ordering."""
        horb_diag = self._get_Horb_diag()
        hci_diag = self._get_Hci_diag()
        pieces = [horb_diag]
        pieces.extend(hci_diag)
        if pieces:
            diagonal = np.concatenate(pieces)
        else:
            diagonal = np.empty(0, dtype=np.complex128)
        _check_shape(
            diagonal, (self.ugg.nvar_tot,), label="Hdiag",
        )
        return diagonal

    def get_grad(self):
        """Return the periodic complex gradient in packed UGG ordering."""
        gorb = self.fock1 - self.fock1.conj().transpose(0, 2, 1)
        gci = [
            [2.0 * residual for residual in residual_r]
            for residual_r in self.hci0
        ]
        gradient = np.asarray(self.ugg.pack(gorb, gci)).reshape(-1)
        _check_shape(
            gradient, (self.ugg.nvar_tot,), label="gradient",
        )
        return gradient

    def ci_response_offdiag(self, h1frs_response):
        """Apply the different-cell blocks of the CI Hessian.

        ``h1frs_response`` is the effective one-electron response returned by
        :meth:`get_h1eff_response`. It contains no self-cell contribution.
        """
        response = []
        for ifrag, (fcibox, norb, nelec, h1rs, ci0_r) in enumerate(zip(
                self.fciboxes, self.ncas_sub, self.nelecas_sub,
                h1frs_response, self.ci)):
            h0_r = [0.0] * self.nroots
            zero_h2 = np.zeros((norb,) * 4, dtype=self.eri_cas.dtype)
            linkstrl = (None if self.linkstrl is None 
                        else self.linkstrl[ifrag])
            response.append(self.Hci(
                fcibox, norb, nelec, h0_r, h1rs, zero_h2, ci0_r,
                linkstrl=linkstrl,
            ))
        response = [[ 2.0 * (hc - np.vdot(c0, hc) * c0) 
                     for hc, c0 in zip(response_r, ci0_r)]
                     for response_r, ci0_r in zip(response, self.ci)]
        return response

    @property
    def shape(self):
        """Shape of the combined orbital/CI Hessian operator."""
        return self.ugg.nvar_tot, self.ugg.nvar_tot

    def _unpack_ci_vector(self, x):
        """Transform a packed complex CSF step to determinant arrays."""
        x_flat = np.asarray(x).reshape(-1)
        if x_flat.size != self.nvar_ci:
            raise ValueError(
                f"trial vector has size {x_flat.size}; expected {self.nvar_ci}"
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
        """Transform determinant-array responses to packed complex CSFs."""
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

    def _ci_hessian_response(self, ci1, tdm1rs=None):
        """Apply the implemented CI-CI Hessian block to determinant vectors."""
        if tdm1rs is None:
            tdm1rs = self.make_tdm1s_sub(ci1)
        h1frs_response = self.get_h1eff_response(tdm1rs)
        ci2_diag = self.ci_response_diag(ci1)
        ci2_offdiag = self.ci_response_offdiag(h1frs_response)

        return [
            [
                diag + offdiag
                for diag, offdiag in zip(diag_r, offdiag_r)
            ]
            for diag_r, offdiag_r in zip(ci2_diag, ci2_offdiag)
        ]

    def _orbital_ci_hessian_response(self, tdm1rs, tcm2):
        """Apply the orbital-output/CI-input Hessian block.

        The transition densities supplied here are already Hermitian
        completed.  The three terms are the one-electron response,
        CI-induced JK response acting on the reference density, and the
        transition-cumulant contraction.  The factor of two belongs here:
        :meth:`_matvec` divides the unpacked orbital response by two when it
        packs the combined Hessian vector, matching the molecular LASSCF
        convention.
        """
        tdm1s_block = self._transition_dm1s_to_block(tdm1rs)
        veff_ci = self._get_ci_veff_response(tdm1s_block)
        cumulant_fock = self._transition_cumulant_to_block_fock(tcm2)
        _check_shape(
            cumulant_fock,
            (self.nkpts, self.nmo, self.nmo),
            label="transition_cumulant_fock",
        )

        dtype = np.result_type(
            tdm1s_block.dtype, veff_ci.dtype, cumulant_fock.dtype,
            self.h1s.dtype, self.dm1s.dtype,
        )
        fock_ci = np.array(cumulant_fock, dtype=dtype, copy=True)
        for k in range(self.nkpts):
            for spin in range(2):
                fock_ci[k] += (
                    self.h1s[spin, k] @ tdm1s_block[spin, k]
                )
                fock_ci[k] += (
                    veff_ci[spin, k] @ self.dm1s[spin, k]
                )

        return 2.0 * (
            fock_ci - fock_ci.conj().transpose(0, 2, 1)
        )

    def _orbital_hamiltonian_response(self, kappa):
        """Differentiate the fragment CI Hamiltonians with the orbitals.

        External rotations are evaluated with the momentum-resolved
        block-MO intermediates and then transformed to the complete Wannier
        active space.  Projected active-active rotations are evaluated
        directly in that Wannier space, where the LAS fragment partition is
        defined.  The returned one- and two-electron operators are the full
        Hermitian first derivatives used by the CI gradient response.
        """
        kappa = np.asarray(kappa)
        _check_shape(
            kappa, (self.nkpts, self.nmo, self.nmo), label="kappa",
        )
        active = slice(self.ncore, self.nocc)
        dtype = np.result_type(
            kappa.dtype, self.h1s.dtype, self.eri_cas.dtype,
            self.mo_phase.dtype, np.complex128,
        )
        h1s_prime = np.zeros(
            (2, self.ncastot, self.ncastot), dtype=dtype,
        )
        h2_prime = np.zeros((self.ncastot,) * 4, dtype=dtype)

        kappa_external = np.array(kappa, copy=True)
        kappa_external[:, active, active] = 0.0
        if np.any(kappa_external):
            odm1s = -np.einsum(
                "skpr,krq->skpq",
                self.dm1s, kappa_external, optimize=True,
            )
            veff_prime = self._get_veff_response(odm1s)
            h1s_block_prime = np.empty(
                (2, self.nkpts, self.ncas, self.ncas), dtype=dtype,
            )
            for spin in range(2):
                for k in range(self.nkpts):
                    h1s_block_prime[spin, k] = (
                        kappa_external[k].conj().T @ self.h1s[spin, k]
                        + self.h1s[spin, k] @ kappa_external[k]
                        + veff_prime[spin, k]
                    )[active, active]
            h1s_prime += np.einsum(
                "kaP,skab,kbQ->sPQ",
                self.mo_phase.conj(), h1s_block_prime, self.mo_phase,
                optimize=True,
            )

            kconserv = kpts_helper.get_kconserv(
                self.las._scf.cell, self.kpts,
            )
            for k1, k2, k3 in kpts_helper.loop_kkk(self.nkpts):
                k4 = kconserv[k1, k2, k3]
                ppaa = self.eris.ppaa(k1, k2, k3)
                papa = self.eris.papa(k1, k2, k3)
                paap = self.eris.paap(k1, k2, k3)
                _check_shape(
                    ppaa,
                    (self.nmo, self.nmo, self.ncas, self.ncas),
                    label=f"ppaa[{k1},{k2},{k3}]",
                )
                _check_shape(
                    papa,
                    (self.nmo, self.ncas, self.nmo, self.ncas),
                    label=f"papa[{k1},{k2},{k3}]",
                )
                _check_shape(
                    paap,
                    (self.nmo, self.ncas, self.ncas, self.nmo),
                    label=f"paap[{k1},{k2},{k3}]",
                )

                eri_block_prime = np.einsum(
                    "pa,pbcd->abcd",
                    kappa_external[k1, :, active].conj(),
                    ppaa[:, active], optimize=True,
                )
                eri_block_prime += np.einsum(
                    "pb,apcd->abcd",
                    kappa_external[k2, :, active],
                    ppaa[active], optimize=True,
                )
                eri_block_prime += np.einsum(
                    "pc,abpd->abcd",
                    kappa_external[k3, :, active].conj(),
                    papa[active], optimize=True,
                )
                eri_block_prime += np.einsum(
                    "pd,abcp->abcd",
                    kappa_external[k4, :, active],
                    paap[active], optimize=True,
                )
                h2_prime += np.einsum(
                    "aP,bQ,abcd,cR,dS->PQRS",
                    self.mo_phase[k1].conj(), self.mo_phase[k2],
                    eri_block_prime,
                    self.mo_phase[k3].conj(), self.mo_phase[k4],
                    optimize=True,
                )

        kappa_active = kappa[:, active, active]
        if np.any(kappa_active):
            rotation_map = self.ugg.active_active_map
            kappa_wannier = rotation_map.block_to_wannier(kappa_active)
            h1_wannier = self._active_wannier_intermediates()[0]
            h1_prime = (
                kappa_wannier.conj().T @ h1_wannier
                + h1_wannier @ kappa_wannier
            )
            eri = self.eri_cas
            eri_prime = np.einsum(
                "ap,aqrs->pqrs", kappa_wannier.conj(), eri,
                optimize=True,
            )
            eri_prime += np.einsum(
                "bq,pbrs->pqrs", kappa_wannier, eri,
                optimize=True,
            )
            eri_prime += np.einsum(
                "cr,pqcs->pqrs", kappa_wannier.conj(), eri,
                optimize=True,
            )
            eri_prime += np.einsum(
                "ds,pqrd->pqrs", kappa_wannier, eri,
                optimize=True,
            )
            coulomb_prime = np.tensordot(
                self.casdm1s, eri_prime, axes=((1, 2), (2, 3)),
            )
            exchange_prime = np.tensordot(
                self.casdm1s, eri_prime, axes=((1, 2), (2, 1)),
            )
            h1s_prime += (
                h1_prime[None] + coulomb_prime
                + coulomb_prime[::-1] - exchange_prime
            )
            h2_prime += eri_prime

        # Differentiate the same state- and fragment-resolved mean-field
        # construction used by pbc.klasci.h1e_for_las.
        h1rs_prime = np.empty(
            (self.nroots, 2, self.ncastot, self.ncastot), dtype=dtype,
        )
        for iroot in range(self.nroots):
            dm1s = self.casdm1rs[iroot] - self.casdm1s
            coulomb = np.tensordot(
                dm1s, h2_prime, axes=((1, 2), (2, 3)),
            )
            exchange = np.tensordot(
                dm1s, h2_prime, axes=((1, 2), (2, 1)),
            )
            h1rs_prime[iroot] = (
                h1s_prime + coulomb + coulomb[::-1] - exchange
            )

        h1frs_prime = []
        offsets = np.cumsum(np.concatenate(([0], self.ncas_sub))).astype(int)
        for ifrag in range(len(self.ncas_sub)):
            i, j = offsets[ifrag:ifrag + 2]
            dm1s = np.asarray(self.casdm1frs[ifrag])
            h2_fragment = h2_prime[i:j, i:j, i:j, i:j]
            coulomb = np.tensordot(
                dm1s, h2_fragment, axes=((2, 3), (2, 3)),
            )
            exchange = np.tensordot(
                dm1s, h2_fragment, axes=((2, 3), (2, 1)),
            )
            h1frs_prime.append(
                h1rs_prime[:, :, i:j, i:j]
                - coulomb - coulomb[:, ::-1] + exchange
            )

        return h1frs_prime, h2_prime

    def _ci_orbital_hessian_response(self, kappa):
        """Apply the CI-output/orbital-input Hessian block."""
        h1frs_prime, h2_prime = self._orbital_hamiltonian_response(kappa)
        hc = self.Hci_all(None, h1frs_prime, h2_prime, self.ci)
        return [
            [
                2.0 * (hc0 - np.vdot(c0, hc0) * c0)
                for hc0, c0 in zip(hc_r, ci0_r)
            ]
            for hc_r, ci0_r in zip(hc, self.ci)
        ]

    def _orbital_hessian_response_block(self, kappa1):
        """Apply the block-MO response contractions without AA correction."""
        odm1s, ocm2 = self._make_orbital_response_dm(kappa1)
        veff_prime = self._get_veff_response(odm1s)
        return self.orbital_response(
            kappa1, odm1s, ocm2, veff_prime,
        )

    def _orbital_hessian_response(self, kappa1):
        """Apply the orbital-orbital Hessian block to ``kappa1``.

        The general block-MO contractions are retained for all external
        sectors.  The contribution from a projected active-active input to
        the active-active output is evaluated by the exact complex Wannier
        formula and replaces the real-orbital one-sided completion in that
        block.
        """
        response = self._orbital_hessian_response_block(kappa1)
        active = slice(self.ncore, self.nocc)
        kappa_active = np.asarray(kappa1)[:, active, active]
        if not np.any(kappa_active):
            return response

        rotation_map = self.ugg.active_active_map
        kappa_wannier = rotation_map.block_to_wannier(kappa_active)
        response_wannier = (
            self._orbital_hessian_response_active_active_wannier(
                kappa_wannier,
            )
        )
        response_active = rotation_map.wannier_to_block(response_wannier)

        kappa_external = np.array(kappa1, copy=True)
        kappa_external[:, active, active] = 0.0
        if np.any(kappa_external):
            response_external = self._orbital_hessian_response_block(
                kappa_external,
            )
        else:
            response_external = np.zeros_like(response)
        response_active += response_external[:, active, active]

        active_coordinates = rotation_map.pack(kappa_active)
        response_cross = self._apply_Horb_active_external_cross(
            active_coordinates,
        )
        x_cross = np.zeros(self.ugg.nvar_orb, dtype=response_cross.dtype)
        x_cross[:self.ugg.nvar_orb_external] = response_cross
        response = response_external + 2.0 * self.ugg.unpack_orb(x_cross)
        response[:, active, active] = response_active
        return response

    def _get_Horb_external_active_cross(self):
        """Return the AA-input/external-output real-coordinate Hessian.

        Complex orbital Hessians are real-linear.  The established block-MO
        response is finite-difference verified for external inputs, including
        its projected active output.  This routine builds that reciprocal
        external-to-active block in real coordinates and transposes it to
        obtain the active-to-external block required for complex AA inputs.
        """
        cached = getattr(self, "_Horb_external_active_cross_cache", None)
        if cached is not None:
            return np.array(cached, copy=True)

        nvar_external = self.ugg.nvar_orb_external
        nvar_active = self.ugg.nvar_orb_active_active
        active_start = nvar_external
        active_stop = active_start + nvar_active
        external_to_active = np.empty(
            (2 * nvar_active, 2 * nvar_external), dtype=float,
        )
        unit = np.zeros(self.ugg.nvar_orb, dtype=np.complex128)
        for index in range(nvar_external):
            unit[index] = 1.0
            kappa = self.ugg.unpack_orb(unit)
            response = self._orbital_hessian_response_block(kappa)
            active_response = self.ugg.pack_orb(
                response / 2.0,
            )[active_start:active_stop]
            external_to_active[:nvar_active, index] = active_response.real
            external_to_active[nvar_active:, index] = active_response.imag

            unit[index] = 1.0j
            kappa = self.ugg.unpack_orb(unit)
            response = self._orbital_hessian_response_block(kappa)
            active_response = self.ugg.pack_orb(
                response / 2.0,
            )[active_start:active_stop]
            column = nvar_external + index
            external_to_active[:nvar_active, column] = active_response.real
            external_to_active[nvar_active:, column] = active_response.imag
            unit[index] = 0.0

        active_to_external = external_to_active.T
        self._Horb_external_active_cross_cache = np.array(
            active_to_external, copy=True,
        )
        return active_to_external

    def _apply_Horb_active_external_cross(self, coordinates):
        """Apply the symmetric AA-input/external-output cross block."""
        coordinates = np.asarray(coordinates).reshape(-1)
        nvar_active = self.ugg.nvar_orb_active_active
        if coordinates.size != nvar_active:
            msg = (
                f"active-active vector has size {coordinates.size}; "
                f"expected {nvar_active}"
            )
            raise ValueError(msg)
        real_coordinates = np.concatenate((
            coordinates.real, coordinates.imag,
        ))
        response = self._get_Horb_external_active_cross() @ real_coordinates
        nvar_external = self.ugg.nvar_orb_external
        return response[:nvar_external] + 1.0j * response[nvar_external:]

    def _active_wannier_intermediates(self):
        """Return active-only Hessian intermediates in the Wannier basis."""
        cached = getattr(self, "_active_wannier_intermediates_cache", None)
        if cached is not None:
            return tuple(np.array(item, copy=True) for item in cached)

        rotation_map = self.ugg.active_active_map
        if not np.allclose(
                rotation_map.mo_phase, self.mo_phase,
                atol=1e-10, rtol=1e-10):
            raise ValueError(
                "UGG and Hessian operator use different Wannier/block maps"
            )

        h1_wannier = np.asarray(self.las.h1e_for_cas(
            mo_coeff=self.mo_coeff, ncas=self.ncas, ncore=self.ncore,
        )[0])
        _check_shape(
            h1_wannier, (self.ncastot, self.ncastot),
            label="h1_wannier",
        )
        coulomb = np.tensordot(
            self.casdm1s, self.eri_cas, axes=((1, 2), (2, 3)),
        )
        exchange = np.tensordot(
            self.casdm1s, self.eri_cas, axes=((1, 2), (2, 1)),
        )
        h1s_wannier = h1_wannier[None] + coulomb + coulomb[::-1] - exchange

        active = slice(self.ncore, self.nocc)
        h1s_block_wannier = np.asarray([
            rotation_map.block_to_wannier(self.h1s[spin, :, active, active])
            for spin in range(2)
        ])
        if not np.allclose(
                h1s_wannier, h1s_block_wannier,
                atol=2e-8, rtol=2e-8):
            error = np.max(np.abs(h1s_wannier - h1s_block_wannier))
            raise ValueError(
                "Wannier and block active one-electron intermediates differ; "
                f"maximum error is {error:.3e}"
            )

        fock1_wannier = sum(
            h1s_wannier[spin] @ self.casdm1s[spin]
            for spin in range(2)
        )
        fock1_wannier += np.tensordot(
            self.eri_cas, self.cascm2,
            axes=((1, 2, 3), (1, 2, 3)),
        )
        self._active_wannier_intermediates_cache = (
            np.array(h1_wannier, copy=True),
            np.array(fock1_wannier, copy=True),
        )
        return h1_wannier, fock1_wannier

    def _orbital_hessian_response_active_active_wannier(
            self, kappa_wannier):
        """Apply the analytic active-active Hessian in Wannier form.

        This is the active-only specialization of the molecular LASSCF OO
        response.  All RDMs and two-electron integrals remain in the complete
        Wannier active space.  The response includes the covariant
        half-commutator used by :meth:`_orbital_hessian_response`.
        """
        kappa_wannier = np.asarray(kappa_wannier)
        _check_shape(
            kappa_wannier, (self.ncastot, self.ncastot),
            label="kappa_wannier",
        )
        h1_wannier, fock1_wannier = (
            self._active_wannier_intermediates()
        )

        # Differentiate U^dagger h U and the four orbital coefficients of
        # (pq|rs) directly.  This is valid for a general complex
        # anti-Hermitian kappa and avoids real-orbital transpose shortcuts.
        h1_prime = (
            h1_wannier @ kappa_wannier
            - kappa_wannier @ h1_wannier
        )
        eri = self.eri_cas
        eri_prime = np.einsum(
            "ap,aqrs->pqrs", kappa_wannier.conj(), eri,
            optimize=True,
        )
        eri_prime += np.einsum(
            "bq,pbrs->pqrs", kappa_wannier, eri,
            optimize=True,
        )
        eri_prime += np.einsum(
            "cr,pqcs->pqrs", kappa_wannier.conj(), eri,
            optimize=True,
        )
        eri_prime += np.einsum(
            "ds,pqrd->pqrs", kappa_wannier, eri,
            optimize=True,
        )

        coulomb_prime = np.tensordot(
            self.casdm1s, eri_prime, axes=((1, 2), (2, 3)),
        )
        exchange_prime = np.tensordot(
            self.casdm1s, eri_prime, axes=((1, 2), (2, 1)),
        )
        h1s_prime = (
            h1_prime[None] + coulomb_prime + coulomb_prime[::-1]
            - exchange_prime
        )
        fock1_prime = sum(
            h1s_prime[spin] @ self.casdm1s[spin]
            for spin in range(2)
        )
        fock1_prime += np.tensordot(
            eri_prime, self.cascm2,
            axes=((1, 2, 3), (1, 2, 3)),
        )

        gradient_prime = fock1_prime - fock1_prime.conj().T
        connection = (
            fock1_wannier @ kappa_wannier
            - kappa_wannier @ fock1_wannier
        ) / 2.0
        connection -= connection.conj().T
        return gradient_prime - connection

    def _apply_Horb_active_active(self, coordinates):
        """Apply the analytic Wannier AA Hessian in projected coordinates."""
        rotation_map = self.ugg.active_active_map
        coordinates = np.asarray(coordinates).reshape(-1)
        if coordinates.size != rotation_map.nvar:
            msg = (
                f"active-active vector has size {coordinates.size}; "
                f"expected {rotation_map.nvar}"
            )
            raise ValueError(msg)
        kappa_block = rotation_map.unpack(coordinates)
        kappa_wannier = rotation_map.block_to_wannier(kappa_block)
        response_wannier = (
            self._orbital_hessian_response_active_active_wannier(
                kappa_wannier,
            )
        )
        response_block = rotation_map.wannier_to_block(response_wannier)
        return rotation_map.pack(response_block / 2.0)

    def _get_Horb_active_active(self):
        """Return the projected complex active-active Hessian blocks.

        For complex orbital coordinates the OO response is real-linear rather
        than complex-linear.  The returned pair ``(H, H_conj)`` represents
        ``Hx = H @ x + H_conj @ x.conj()`` exactly.  Both blocks are evaluated
        analytically in the Wannier active space and then projected through
        the UGG block-coordinate basis.
        """
        cached = getattr(self, "_Horb_active_active_cache", None)
        if cached is not None:
            return tuple(np.array(block, copy=True) for block in cached)

        nvar = self.ugg.nvar_orb_active_active
        dtype = np.result_type(self.mo_coeff.dtype, np.complex128)
        response_real = np.empty((nvar, nvar), dtype=dtype)
        response_imag = np.empty((nvar, nvar), dtype=dtype)
        unit = np.zeros(nvar, dtype=dtype)
        for index in range(nvar):
            unit[index] = 1.0
            response_real[:, index] = self._apply_Horb_active_active(unit)
            unit[index] = 1.0j
            response_imag[:, index] = self._apply_Horb_active_active(unit)
            unit[index] = 0.0

        hessian = (response_real - 1.0j * response_imag) / 2.0
        hessian_conj = (response_real + 1.0j * response_imag) / 2.0
        self._Horb_active_active_cache = (
            np.array(hessian, copy=True),
            np.array(hessian_conj, copy=True),
        )
        return hessian, hessian_conj

    def _make_orbital_response_dm(self, kappa):
        """Build one-sided 1-RDM and cumulant responses.

        ``odm1s`` is in the block-MO basis.  ``ocm2[k1,k2,k3]`` has three
        active indices at ``k1``, ``k2``, and ``k3`` and one general orbital
        index at ``k4``.  These are bra-ket-bra-ket tensor indices, so their
        momentum rule is ``k1 - k2 + k3 - k4 = G``.
        """
        _check_shape(
            kappa, (self.nkpts, self.nmo, self.nmo), label="kappa",
        )
        odm1s = -np.einsum(
            "skpr,krq->skpq", self.dm1s, kappa, optimize=True,
        )

        dtype = np.result_type(self.cascm2.dtype, kappa.dtype)
        ocm2 = np.empty(
            (self.nkpts, self.nkpts, self.nkpts)
            + (self.ncas, self.ncas, self.ncas, self.nmo),
            dtype=dtype,
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
            ocm2[k1, k2, k3] = -np.einsum(
                "abcd,dp->abcp",
                cascm2_kpts, kappa[k4, active, :],
                optimize=True,
            )
        return odm1s, ocm2

    def _get_veff_response(self, odm1s):
        """Return spin-resolved JK response in the block-MO basis."""
        _check_shape(
            odm1s,
            (2, self.nkpts, self.nmo, self.nmo),
            label="odm1s",
        )
        # odm1s = -D kappa is the ket-side response.  Because D is Hermitian
        # and kappa is anti-Hermitian, its bra-side partner is
        # (-D kappa)^dagger = kappa D.  Thus this adjoint is the actual
        # first-order Hermitian density, not a real-orbital transpose shortcut.
        dm1s_mo = odm1s + odm1s.conj().transpose(0, 1, 3, 2)
        return self._get_veff_from_block_dm1s(dm1s_mo)

    def _get_ci_veff_response(self, tdm1s_block):
        """Return JK response to a full Hermitian CI transition density.

        ``tdm1s_block`` is already Hermitian and must not be completed a
        second time.  This helper is normalization neutral: it returns the
        response to exactly the density supplied by its caller.
        """
        _check_shape(
            tdm1s_block,
            (2, self.nkpts, self.nmo, self.nmo),
            label="transition_dm1s_block",
        )
        return self._get_veff_from_block_dm1s(tdm1s_block)

    def _get_veff_from_block_dm1s(self, dm1s_mo):
        """Transform a Hermitian block-MO density through the AO JK map."""
        dm1s_mo = np.asarray(dm1s_mo)
        _check_shape(
            dm1s_mo,
            (2, self.nkpts, self.nmo, self.nmo),
            label="dm1s_mo_response",
        )
        dtype = np.result_type(dm1s_mo.dtype, self.mo_coeff.dtype)
        dm1s_ao = np.empty(
            (2, self.nkpts, self.nao, self.nao), dtype=dtype,
        )
        for k in range(self.nkpts):
            mo_coeff = self.mo_coeff[k]
            dm1s_ao[:, k] = (
                mo_coeff @ dm1s_mo[:, k] @ mo_coeff.conj().T
            )

        veff_ao = np.asarray(self.las.get_veff(
            self.las._scf.cell, dm_kpts=dm1s_ao,
            hermi=1, kpts=self.kpts,
        ))
        _check_shape(
            veff_ao,
            (2, self.nkpts, self.nao, self.nao),
            label="veff_prime_ao",
        )
        veff_mo = np.empty(
            (2, self.nkpts, self.nmo, self.nmo),
            dtype=np.result_type(veff_ao.dtype, self.mo_coeff.dtype),
        )
        for k in range(self.nkpts):
            mo_coeff = self.mo_coeff[k]
            veff_mo[:, k] = (
                mo_coeff.conj().T @ veff_ao[:, k] @ mo_coeff
            )
        return veff_mo

    def _symmetrize_active_ocm2(self, ocm2):
        """Complete the active cumulant response using its two symmetries.

        For bra-ket-bra-ket ordering, Hermiticity is
        ``L[a,b,c,d] = L[b,a,d,c].conj()`` and electron-pair exchange is
        ``L[a,b,c,d] = L[c,d,a,b]``.  The corresponding source k-point blocks
        are ``(k2,k1,k4)`` and ``(k3,k4,k1)``; each still obeys the original
        ``+ - + -`` momentum rule.
        """
        _check_shape(
            ocm2,
            (self.nkpts, self.nkpts, self.nkpts,
             self.ncas, self.ncas, self.ncas, self.nmo),
            label="ocm2",
        )
        active = slice(self.ncore, self.nocc)
        one_sided = ocm2[..., active]
        half = np.empty_like(one_sided)
        result = np.empty_like(one_sided)
        kconserv = kpts_helper.get_kconserv(
            self.las._scf.cell, self.kpts,
        )

        for k1, k2, k3 in kpts_helper.loop_kkk(self.nkpts):
            k4 = kconserv[k1, k2, k3]
            half[k1, k2, k3] = (
                one_sided[k1, k2, k3]
                + one_sided[k2, k1, k4].conj().transpose(1, 0, 3, 2)
            )
        for k1, k2, k3 in kpts_helper.loop_kkk(self.nkpts):
            k4 = kconserv[k1, k2, k3]
            result[k1, k2, k3] = (
                half[k1, k2, k3]
                + half[k3, k4, k1].transpose(2, 3, 0, 1)
            )
        return result

    def orbital_response(self, kappa, odm1s, ocm2, veff_prime):
        """Build the complex block-MO orbital Hessian response."""
        _check_shape(
            kappa, (self.nkpts, self.nmo, self.nmo), label="kappa",
        )
        _check_shape(
            odm1s, (2, self.nkpts, self.nmo, self.nmo), label="odm1s",
        )
        _check_shape(
            veff_prime,
            (2, self.nkpts, self.nmo, self.nmo),
            label="veff_prime",
        )
        edm1s = odm1s + odm1s.conj().transpose(0, 1, 3, 2)
        ecm2 = self._symmetrize_active_ocm2(ocm2)
        dtype = np.result_type(
            edm1s.dtype, ecm2.dtype, veff_prime.dtype, self.fock1.dtype,
        )
        f1_prime = np.zeros(
            (self.nkpts, self.nmo, self.nmo), dtype=dtype,
        )
        fock1_one_body = np.zeros_like(f1_prime)
        active = slice(self.ncore, self.nocc)

        for k in range(self.nkpts):
            for spin in range(2):
                fock1_one_body[k] += (
                    self.h1s[spin, k] @ self.dm1s[spin, k]
                )
                f1_prime[k] += self.h1s[spin, k] @ edm1s[spin, k]
                f1_prime[k] += veff_prime[spin, k] @ self.dm1s[spin, k]
            f1_prime[k] += (
                self.fock1[k] @ kappa[k]
                - kappa[k] @ self.fock1[k]
            ) / 2.0

        for k1, k2, k3 in kpts_helper.loop_kkk(self.nkpts):
            paaa = self.eri_paaa(k1, k2, k3)
            f1_prime[k1][:, active] += np.tensordot(
                paaa, ecm2[k1, k2, k3],
                axes=((1, 2, 3), (1, 2, 3)),
            )

        f1_prime += self._orbital_response_external_cumulant(
            kappa, self.fock1 - fock1_one_body,
        )
        return f1_prime - f1_prime.conj().transpose(0, 2, 1)

    def _orbital_response_external_cumulant(
            self, kappa, fock1_cumulant):
        """Differentiate the cumulant Fock term for external rotations.

        The molecular real-orbital implementation reduces all four integral
        derivatives to ``ppaa`` and ``papa`` by permutational symmetry.  For
        complex Bloch orbitals, some of those permutations also conjugate the
        integrals.  Contracting the three differentiated active integral
        indices directly with the disk-backed ``ppaa``, ``papa``, and
        ``paap`` blocks avoids that real-only assumption.

        The returned matrix omits the final skew-Hermitian completion.  Its
        ``-F_cumulant @ kappa`` connection term combines with the half
        commutator already added by :meth:`orbital_response` to give the
        covariant orbital Hessian used by the molecular implementation.

        This is the first-order expansion of ``mc1step.gorb_update``.  The
        stored ERIs retain bra-ket-bra-ket order and therefore always use
        ``k1 - k2 + k3 - k4 = G``.  ``mc1step`` also constructs a regrouped
        ``hdm2_ppaa[p,u,q,v]`` tensor whose labels obey ``k1 + k2 - k3 - k4``;
        that alternate rule does not apply here because the contractions below
        consume ``kappa`` before such a regrouped Hessian tensor is formed.
        """
        _check_shape(
            kappa, (self.nkpts, self.nmo, self.nmo), label="kappa",
        )
        _check_shape(
            fock1_cumulant,
            (self.nkpts, self.nmo, self.nmo),
            label="fock1_cumulant",
        )
        active = slice(self.ncore, self.nocc)
        kappa_external = np.array(kappa, copy=True)
        kappa_external[:, active, active] = 0
        response = -np.einsum(
            "kpr,krq->kpq", fock1_cumulant, kappa_external,
            optimize=True,
        )
        kconserv = kpts_helper.get_kconserv(
            self.las._scf.cell, self.kpts,
        )

        for k1, k2, k3 in kpts_helper.loop_kkk(self.nkpts):
            k4 = kconserv[k1, k2, k3]
            cascm2_kpts = _get_casdm2_kpts(
                self.cascm2, self.mo_phase, (k1, k2, k3, k4),
            )
            ppaa = self.eris.ppaa(k1, k2, k3)
            papa = self.eris.papa(k1, k2, k3)
            paap = self.eris.paap(k1, k2, k3)
            # Ket index 2 of ppaa: dU/dt = kappa.
            response[k1][:, active] += np.einsum(
                "pxst,xr,qrst->pq",
                ppaa, kappa_external[k2, :, active], cascm2_kpts,
                optimize=True,
            )
            # Bra index 3 of papa: d(U*)/dt = kappa*; using
            # kappa*_{x,s} = -kappa_{s,x} gives the explicit minus sign.
            response[k1][:, active] -= np.einsum(
                "sx,prxt,qrst->pq",
                kappa_external[k3, active, :], papa, cascm2_kpts,
                optimize=True,
            )
            # Ket index 4 of paap: dU/dt = kappa.
            response[k1][:, active] += np.einsum(
                "prsx,xt,qrst->pq",
                paap, kappa_external[k4, :, active], cascm2_kpts,
                optimize=True,
            )
        return response

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

    def _matvec(self, x):
        """Dispatch a combined packed orbital/CI Hessian-vector product.

        The UGG owns the external vector layout.  The existing CI-CI action
        is evaluated only for a nonzero CI component.  A nonzero orbital
        component is routed to both the orbital-orbital and CI-output/orbital-
        input responses, after which the reciprocal orbital-output/CI-input
        and CI-CI responses are added when needed.
        """
        kappa1, ci1 = self.ugg.unpack(x)
        dtype = np.result_type(np.asarray(x).dtype, kappa1.dtype)

        if np.any(kappa1):
            kappa2 = np.asarray(self._orbital_hessian_response(kappa1))
            _check_shape(kappa2, np.shape(kappa1), label="kappa2")
            ci2 = self._ci_orbital_hessian_response(kappa1)
        else:
            kappa2 = np.zeros_like(kappa1, dtype=dtype)
            ci2 = self._zero_ci_step(dtype)

        if not self._ci_step_is_zero(ci1):
            tdm1rs, tcm2 = self.make_tdm1s2c_sub(ci1)
            kappa2 = kappa2 + self._orbital_ci_hessian_response(
                tdm1rs, tcm2,
            )
            ci2_ci = self._ci_hessian_response(ci1, tdm1rs=tdm1rs)
            ci2 = [
                [
                    orbital_response + ci_response
                    for orbital_response, ci_response in zip(
                        orbital_response_r, ci_response_r,
                    )
                ]
                for orbital_response_r, ci_response_r in zip(ci2, ci2_ci)
            ]

        kappa2 = kappa2 + self.level_shift * kappa1
        ci2 = [
            [
                response + self.level_shift * trial
                for response, trial in zip(response_r, trial_r)
            ]
            for response_r, trial_r in zip(ci2, ci1)
        ]

        return self.ugg.pack(kappa2 / 2.0, ci2)

    _rmatvec = _matvec


class KLASSCF_TransSymmHessianOperator(KLASSCF_HessianOperator):
    """Translation-adapted CI Hessian operator for k-LASSCF.

    The external CI vector retains the full ``[cell][root]`` layout used by
    :class:`KLASSCF_HessianOperator`.  Before a CI, RDM, or transition-RDM
    contraction, the translated cell vectors are packed into one representative
    vector using ``phase_per_frag``.  The contracted response is subsequently
    expanded back to the full layout with the same phases.

    Parameters
    ----------
    ref_cell : int, optional
        Representative BvK cell.  Defaults to ``las.ref_cell``.
    phase_per_frag : array_like, optional
        Unit-modulus CI translation phase for each cell.  When omitted, the
        phases are obtained from ``las.get_phase_per_frag(mo_coeff)``.
    validate_trans_symmetry : bool, optional
        Validate translated CI vectors and local Hamiltonian blocks.
    trans_sym_tol : float, optional
        Absolute and relative tolerance used by the validation.

    Other parameters are identical to :class:`KLASSCF_HessianOperator`.
    """

    def __init__(
            self, las, ugg, mo_coeff=None, ci=None, casdm1frs=None,
            h1eff=None, h2eff=None, kpts=None, kmesh=None, ref_cell=None,
            phase_per_frag=None, validate_trans_symmetry=True,
            trans_sym_tol=1e-8, casdm2fr=None, eris=None,
            veff_kpts=None, dm1s_kpts=None, mo_phase=None):
        if mo_coeff is None:
            mo_coeff = las.mo_coeff
        if ci is None:
            ci = las.ci
        if kmesh is None:
            kmesh = las.kmesh

        kmesh = tuple(int(n) for n in kmesh)
        ncell = int(np.prod(kmesh))
        if ref_cell is None:
            ref_cell = getattr(las, "ref_cell", 0)
        if not isinstance(ref_cell, (int, np.integer)):
            raise TypeError("ref_cell must be an integer")
        if not 0 <= ref_cell < ncell:
            raise ValueError(
                f"ref_cell must be in [0, {ncell}); got {ref_cell}"
            )

        if phase_per_frag is None:
            get_phases = getattr(las, "get_phase_per_frag", None)
            if get_phases is None:
                phase_per_frag = np.ones(ncell, dtype=np.complex128)
            else:
                phase_per_frag = get_phases(mo_coeff)

        self.ref_cell = int(ref_cell)
        self.ncell = ncell
        self.phase_per_frag = self._normalize_phase_per_frag(
            phase_per_frag, ncell, self.ref_cell,
        )
        if not isinstance(validate_trans_symmetry, (bool, np.bool_)):
            raise TypeError("validate_trans_symmetry must be a boolean")
        self.validate_trans_symmetry = bool(validate_trans_symmetry)
        self.trans_sym_tol = float(trans_sym_tol)
        if not np.isfinite(self.trans_sym_tol) or self.trans_sym_tol <= 0:
            raise ValueError("trans_sym_tol must be finite and positive")

        ci_ref = self._pack_ci(
            ci, validate=self.validate_trans_symmetry,
            tol=self.trans_sym_tol,
        )
        ci = self._unpack_cif(ci_ref)

        super().__init__(
            las, ugg, mo_coeff=mo_coeff, ci=ci, casdm1frs=casdm1frs,
            h1eff=h1eff, h2eff=h2eff, kpts=kpts, kmesh=kmesh,
            casdm2fr=casdm2fr, eris=eris, veff_kpts=veff_kpts,
            dm1s_kpts=dm1s_kpts, mo_phase=mo_phase,
        )

    @staticmethod
    def _normalize_phase_per_frag(phase_per_frag, ncell, ref_cell):
        """Validate cell phases and use the reference cell as phase origin."""
        phase_per_frag = np.asarray(
            phase_per_frag, dtype=np.result_type(phase_per_frag, np.complex128),
        )
        _check_shape(phase_per_frag, (ncell,), label="phase_per_frag")
        magnitudes = np.abs(phase_per_frag)
        if np.any(~np.isfinite(magnitudes)) or np.any(magnitudes == 0):
            raise ValueError(
                "phase_per_frag must contain finite nonzero phases"
            )
        if not np.allclose(magnitudes, 1.0, atol=1e-8, rtol=0.0):
            raise ValueError("phase_per_frag entries must have unit magnitude")

        phases = phase_per_frag / magnitudes
        phases *= phases[ref_cell].conjugate()
        phases[ref_cell] = 1.0
        return phases

    def _pack_ci(self, ci, validate=False, tol=None):
        """Pack full translated CI vectors into one phase-free cell vector.

        The phase-weighted average is the projector onto the translationally
        adapted CI subspace.  It is also insensitive to the selected
        representative cell.
        """
        if ci is None:
            return None
        if len(ci) != self.ncell:
            raise ValueError(
                f"CI list must contain {self.ncell} cells; got {len(ci)}"
            )
        if tol is None:
            tol = getattr(self, "trans_sym_tol", 1e-8)

        nroots = len(ci[self.ref_cell])
        if any(len(ci_r) != nroots for ci_r in ci):
            raise ValueError("translated cells have inconsistent root counts")

        packed = []
        for iroot in range(nroots):
            ref_shape = np.shape(ci[self.ref_cell][iroot])
            translated = []
            for phase, ci_r in zip(self.phase_per_frag, ci):
                _check_shape(ci_r[iroot], ref_shape, label=f"ci_r[{iroot}]")
                translated.append(
                    phase.conjugate() * np.asarray(ci_r[iroot])
                )
            ci_ref = np.mean(np.stack(translated, axis=0), axis=0)
            if validate:
                scale = max(np.linalg.norm(ci_ref), 1.0)
                error = max(
                    np.linalg.norm(ci_cell - ci_ref)
                    for ci_cell in translated
                )
                if error > tol * scale:
                    raise ValueError(
                        "CI vectors do not obey the requested translation "
                        f"phases; maximum error {error:.3e}"
                    )
            packed.append(ci_ref)
        return packed

    def _unpack_cif(self, ci_ref):
        """Expand packed root CI vectors to all cells with their phases."""
        if ci_ref is None:
            return [None for _ in range(self.ncell)]
        return [
            [np.array(phase * c0, copy=True) for c0 in ci_ref]
            for phase in self.phase_per_frag
        ]

    def _init_dms_(self, casdm1frs, casdm2fr=None, dm1s_kpts=None):
        """Construct reference RDMs once and copy phase-invariant blocks."""
        ref = self.ref_cell
        ncas_ref = int(self.ncas_sub[ref])
        nelec_ref = tuple(self.nelecas_sub[ref])
        if any(int(ncas) != ncas_ref for ncas in self.ncas_sub):
            raise ValueError(
                "translation-adapted cells must have identical active spaces"
            )
        if any(tuple(nelec) != nelec_ref for nelec in self.nelecas_sub):
            raise ValueError(
                "translation-adapted cells must have identical electron counts"
            )
        if casdm1frs is None:
            ci_ref = self._pack_ci(self.ci)
            fcibox = self.fciboxes[ref]
            dm1a, dm1b = fcibox.states_make_rdm1s(
                ci_ref, self.ncas_sub[ref], self.nelecas_sub[ref],
            )
            dm1_ref = np.stack([dm1a, dm1b], axis=1)
        else:
            if len(casdm1frs) != self.ncell:
                raise ValueError(
                    "casdm1frs must contain one block for every cell"
                )
            dm1_ref = np.asarray(casdm1frs[ref])
            if self.validate_trans_symmetry:
                for dm1 in casdm1frs:
                    if not np.allclose(
                            dm1, dm1_ref, atol=self.trans_sym_tol,
                            rtol=self.trans_sym_tol):
                        raise ValueError(
                            "casdm1frs is not translation symmetric"
                        )

        casdm1frs = [np.array(dm1_ref, copy=True) for _ in range(self.ncell)]
        KLASSCF_HessianOperator._init_dms_(
            self, casdm1frs, casdm2fr, dm1s_kpts,
        )

    def _validate_local_hamiltonians(self):
        """Check equivalence of local one- and two-electron blocks."""
        if not self.validate_trans_symmetry:
            return
        ref = self.ref_cell
        h1_ref = np.asarray(self.h1frs[ref])
        for h1 in self.h1frs:
            if not np.allclose(
                    h1, h1_ref, atol=self.trans_sym_tol,
                    rtol=self.trans_sym_tol):
                raise ValueError("h1eff local blocks are not translation symmetric")

        iref = int(np.sum(self.ncas_sub[:ref]))
        jref = iref + int(self.ncas_sub[ref])
        h2_ref = self.eri_cas[iref:jref, iref:jref, iref:jref, iref:jref]
        for ifrag, norb in enumerate(self.ncas_sub):
            i = int(np.sum(self.ncas_sub[:ifrag]))
            j = i + int(norb)
            h2 = self.eri_cas[i:j, i:j, i:j, i:j]
            if not np.allclose(
                    h2, h2_ref, atol=self.trans_sym_tol,
                    rtol=self.trans_sym_tol):
                raise ValueError("h2eff local blocks are not translation symmetric")

    def _init_ci_(self):
        """Cache one representative local Hamiltonian action."""
        self._validate_local_hamiltonians()
        ref = self.ref_cell
        fcibox = self.fciboxes[ref]
        norb = self.ncas_sub[ref]
        nelec = self.nelecas_sub[ref]
        linkstrl_ref = fcibox.states_gen_linkstr(norb, nelec, False)
        linkstr_ref = fcibox.states_gen_linkstr(norb, nelec, False)
        self.linkstrl = [linkstrl_ref for _ in range(self.ncell)]
        self.linkstr = [linkstr_ref for _ in range(self.ncell)]

        i = int(np.sum(self.ncas_sub[:ref]))
        j = i + int(norb)
        h2_ref = self.eri_cas[i:j, i:j, i:j, i:j]
        ci_ref = self._pack_ci(self.ci)
        h0_ref = [0.0] * self.nroots
        hc_ref = self.Hci(
            fcibox, norb, nelec, h0_ref, self.h1frs[ref], h2_ref,
            ci_ref, linkstrl=linkstrl_ref,
        )
        e_ref = [np.vdot(c0, hc0) for c0, hc0 in zip(ci_ref, hc_ref)]
        residual_ref = [
            hc0 - energy * c0
            for hc0, energy, c0 in zip(hc_ref, e_ref, ci_ref)
        ]
        self.e0 = [list(e_ref) for _ in range(self.ncell)]
        self.hci0 = self._unpack_cif(residual_ref)

    def make_tdm1s_sub(self, ci1):
        """Build all cell TDM blocks from one packed CI contraction.

        For ``c_S = phase_S c_ref`` and ``x_S = phase_S x_ref``, the bra
        and ket phases cancel in ``x_S^dagger A c_S``.  The reference
        transition density is therefore copied to every translated cell.
        """
        ci1_ref = self._pack_ci(ci1)
        ci0_ref = self._pack_ci(self.ci)
        ref = self.ref_cell
        fcibox = self.fciboxes[ref]
        norb = self.ncas_sub[ref]
        nelec = self.nelecas_sub[ref]
        linkstr = None if self.linkstr is None else self.linkstr[ref]

        state_arg = fcibox._state_args
        solver_arg = fcibox._solver_args
        nelec_by_solver = [
            fcibox._get_nelec(solver, nelec)
            for solver in fcibox.fcisolvers
        ]
        collect_args = (
            state_arg(ci1_ref), state_arg(ci0_ref), norb,
            solver_arg(nelec_by_solver),
        )
        collect_kwargs = {"link_index": solver_arg(linkstr)}
        try:
            dm1_r = list(fcibox._collect(
                "trans_rdm1s", *collect_args, **collect_kwargs,
            ))
        except AttributeError as err:
            if "FCItrans_rdm1" not in str(err):
                raise
            dm1_r = list(fcibox._collect(
                "trans_rdm1s_py", *collect_args, **collect_kwargs,
            ))
        if len(dm1_r) != self.nroots:
            raise ValueError(
                f"reference cell produced {len(dm1_r)} transition "
                f"densities for {self.nroots} roots"
            )

        dtype = np.result_type(self.eri_cas.dtype, np.complex128)
        tdm1_ref = np.zeros(
            (self.nroots, 2, norb, norb), dtype=dtype,
        )
        for iroot, (dm1s, c1, c0, dm1s_ref) in enumerate(zip(
                dm1_r, ci1_ref, ci0_ref, self.casdm1frs[ref])):
            overlap = np.vdot(c1, c0)
            tdm1s = np.stack(dm1s, axis=0) - overlap * dm1s_ref
            tdm1_ref[iroot] = (
                tdm1s + tdm1s.swapaxes(-1, -2).conj()
            )

        tdm1rs = np.zeros(
            (self.nroots, 2, self.ncastot, self.ncastot), dtype=dtype,
        )
        for ifrag, ncas in enumerate(self.ncas_sub):
            i = int(np.sum(self.ncas_sub[:ifrag]))
            j = i + int(ncas)
            tdm1rs[:, :, i:j, i:j] = tdm1_ref
        return tdm1rs

    def get_h1eff_response(self, tdm1rs):
        """Build one translated effective-Hamiltonian response and copy it."""
        tdm1rs = np.asarray(tdm1rs)
        _check_shape(
            tdm1rs, (self.nroots, 2, self.ncastot, self.ncastot),
            label="tdm1rs"
        )

        eri = self.eri_cas
        v1rs = np.tensordot(tdm1rs, eri, axes=((2, 3), (0, 1)))
        v1rs += v1rs[:, ::-1]
        v1rs -= np.tensordot(
            tdm1rs, eri, axes=((2, 3), (2, 1)),
        )

        ref = self.ref_cell
        i = int(np.sum(self.ncas_sub[:ref]))
        j = i + int(self.ncas_sub[ref])
        dm1rs_ref = tdm1rs[:, :, i:j, i:j]
        v1rs_ref = np.tensordot(
            dm1rs_ref, eri[i:j, i:j, :, :],
            axes=((2, 3), (0, 1)),
        )
        v1rs_ref += v1rs_ref[:, ::-1]
        v1rs_ref -= np.tensordot(
            dm1rs_ref, eri[:, i:j, i:j, :],
            axes=((2, 3), (2, 1)),
        )
        h1_ref = v1rs[:, :, i:j, i:j] - v1rs_ref[:, :, i:j, i:j]
        return [np.array(h1_ref, copy=True) for _ in range(self.ncell)]

    def ci_response_diag(self, ci1):
        """Apply one same-cell CI Hessian block and translate the result."""
        ref = self.ref_cell
        ci1_ref = self._pack_ci(ci1)
        ci0_ref = self._pack_ci(self.ci)
        norb = self.ncas_sub[ref]
        nelec = self.nelecas_sub[ref]
        i = int(np.sum(self.ncas_sub[:ref]))
        j = i + int(norb)
        h2_ref = self.eri_cas[i:j, i:j, i:j, i:j]
        h0_ref = [-energy for energy in self.e0[ref]]
        ci2_ref = self.Hci(
            self.fciboxes[ref], norb, nelec, h0_ref,
            self.h1frs[ref], h2_ref, ci1_ref,
            linkstrl=self.linkstrl[ref],
        )
        response_ref = []
        for hc1, c1, c0, residual in zip(
                ci2_ref, ci1_ref, ci0_ref, self.hci0[ref]):
            output_overlap = np.vdot(residual, c1)
            input_overlap = np.vdot(c0, c1)
            response_ref.append(2.0 * (
                hc1 - output_overlap * c0 - input_overlap * residual
            ))
        return self._unpack_cif(response_ref)

    def ci_response_offdiag(self, h1frs_response):
        """Apply one different-cell CI response and translate the result."""
        if len(h1frs_response) != self.ncell:
            raise ValueError(
                "h1frs_response must contain one block for every cell"
            )
        ref = self.ref_cell
        ci0_ref = self._pack_ci(self.ci)
        norb = self.ncas_sub[ref]
        nelec = self.nelecas_sub[ref]
        zero_h2 = np.zeros(
            (norb,) * 4, dtype=self.eri_cas.dtype,
        )
        hc_ref = self.Hci(
            self.fciboxes[ref], norb, nelec, [0.0] * self.nroots,
            h1frs_response[ref], zero_h2, ci0_ref,
            linkstrl=self.linkstrl[ref],
        )
        response_ref = [
            2.0 * (hc - np.vdot(c0, hc) * c0)
            for hc, c0 in zip(hc_ref, ci0_ref)
        ]
        return self._unpack_cif(response_ref)

# Register the complex orbital/CI parameterization and total gradient on both
# periodic LAS variants. Hessian dispatch is currently developed only for the
# general k-LASSCF operator; no translation-specific dispatch is applied.
PBCLASCINoSymm.get_grad_orb = get_grad_orb
PBCLASCINoSymm._klasscf_eris = _ERIS
PBCLASCINoSymm._ugg = KLASSCF_UnitaryGroupGenerators
PBCLASCINoSymm.get_ugg = get_ugg
PBCLASCINoSymm.get_grad_ci = get_grad_ci
PBCLASCINoSymm.get_grad = get_grad
PBCLASCITransSymm.get_grad_orb = get_grad_orb
PBCLASCITransSymm._klasscf_eris = _ERIS
PBCLASCITransSymm._ugg = KLASSCF_UnitaryGroupGenerators
PBCLASCITransSymm.get_ugg = get_ugg
PBCLASCITransSymm.get_grad_ci = get_grad_ci
PBCLASCITransSymm.get_grad = get_grad
