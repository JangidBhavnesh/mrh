#!/usr/bin/env python

import numpy as np

from mrh.my_pyscf.mcscf.lasscf_sync_o0 import (
    LASSCF_HessianOperator as molLASSCF_HessianOperator,
)
from mrh.my_pyscf.pbc.fci import cplx_csf_helper

class KLASSCF_HessianOperator(molLASSCF_HessianOperator):
    """Periodic CI-only Hessian operator for k-LASSCF.

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
    kpts, kmesh
        BvK k-points and k-point mesh.  They default to ``las.kpts`` and
        ``las.kmesh`` and must describe the same number of cells.

    Notes
    -----
    The molecular parent constructor is intentionally not called: it
    immediately builds molecular orbital-response and ERI intermediates that
    are not valid for the periodic CI-only operator.
    """

    def __init__(
            self, las, ugg, mo_coeff=None, ci=None, casdm1frs=None,
            h1eff=None, h2eff=None, kpts=None, kmesh=None):

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
        self.ncas = int(np.sum(self.ncas_sub))
        self.nao = mo_coeff.shape[-2]
        self.nmo = mo_coeff.shape[-1]
        self.nocc = self.ncore + self.ncas
        self.fciboxes = las.fciboxes
        self.nroots = las.nroots
        self.weights = las.weights
        self.ci_transformers = ugg.ci_transformers
        self.frozen_ci = set(getattr(ugg, "frozen_ci", None) or [])
        if len(self.ci_transformers) != len(self.ci):
            msg = (f"ugg.ci_transformers must contain one entry per CI cell; ")
            raise ValueError(msg)
        self.nvar_ci = 0

        for ifrag, (transformers, ci0_r) in enumerate(zip(self.ci_transformers, self.ci)):
            if len(transformers) != len(ci0_r):
                msg = f"cell {ifrag} has {len(transformers)} CSF transformers \
                    for {len(ci0_r)} CI roots"
                raise ValueError(msg)
                
            if ifrag not in self.frozen_ci:
                self.nvar_ci += sum(t.ncsf for t in transformers)

        self._init_dms_(casdm1frs)
        self._init_ham_(h1eff, h2eff)
        self._init_ci_()

    def _init_dms_(self, casdm1frs):
        """
        Initialize the spin-separated active-space 1-RDMs.

        I think the ``casdm1frs[f][r]`` is the 1-RDM of fragment/cell ``f`` and root
        ``r``.  The CI-only hop needs these reference densities when forming
        the first-order effective one-electron Hamiltonian.  No 2-RDM or
        cumulant or full RDM is initialized here.
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

    def _init_ham_(self, h1eff, h2eff):
        """
        Note: the input args has been changed from the molecular version.
        Initialize the active-space Hamiltonian used by the CI hop.

        ``h1frs[f][r]`` is the spin-separated effective one-electron
        Hamiltonian for fragment/cell ``f`` and root ``r``.  ``eri_cas``
        contains the two-electron integrals over the complete Wannier active
        space; its diagonal cell blocks are used for the local CI actions.
        """
        if h2eff is None:
            h2eff = self.las.get_h2cas(self.mo_coeff)
        h2eff = np.asarray(h2eff)
        ncas_tot = int(np.sum(self.ncas_sub))
        expected_h2_shape = (ncas_tot,) * 4
        if h2eff.shape != expected_h2_shape:
            msg = (f"h2eff must have shape {expected_h2_shape}; " f"got {h2eff.shape}")
            raise ValueError(msg)

        if h1eff is None:
            h1eff = self.las.h1e_for_las(
                mo_coeff=self.mo_coeff,
                ci=self.ci,
                ncas_sub=self.ncas_sub,
                nelecas_sub=self.nelecas_sub,
                casdm1s_sub=self.casdm1fs,
                casdm1frs=self.casdm1frs,
                eri_cas=h2eff,
            )

        if len(h1eff) != len(self.ncas_sub):
            msg = "h1eff must contain one block for every fragment/cell"
            raise ValueError(msg)
        
        for ifrag, (h1fr, ncas) in enumerate(zip(h1eff, self.ncas_sub)):
            expected_h1_shape = (self.nroots, 2, ncas, ncas)
            if np.shape(h1fr) != expected_h1_shape:
                msg = (f"h1eff[{ifrag}] must have shape {expected_h1_shape}; "
                       f"got {np.shape(h1fr)}")
                raise ValueError(msg)

        self.h1frs = h1eff
        self.eri_cas = h2eff

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

        The returned array has shape ``(nroots, 2, ncas, ncas)``. For a
        complex CI step, each one-sided transition density is combined with
        its Hermitian conjugate after removing the component parallel to the
        current CI vector.
        """
        dtype = np.result_type(self.eri_cas.dtype, np.complex128)
        tdm1rs = np.zeros((self.nroots, 2, self.ncas, self.ncas), dtype=dtype,)

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
            try:
                dm1_r = list(fcibox._collect(
                    "trans_rdm1s", *collect_args, **collect_kwargs,
                ))
            except AttributeError as err:
                # Some existing builds predate the compiled complex
                # transition-RDM symbols. Use the equivalent complex Python
                # implementation in that case.
                if "FCItrans_rdm1" not in str(err):
                    raise
                dm1_r = list(fcibox._collect(
                    "trans_rdm1s_py", *collect_args, **collect_kwargs,
                ))

            if len(dm1_r) != self.nroots:
                msg = (f"fragment {ifrag} produced {len(dm1_r)} transition "
                       f"densities for {self.nroots} roots")
                raise ValueError(msg)

            for iroot, (dm1s, c1, c0, dm1s_ref) in enumerate(zip(
                    dm1_r, c1_r, c0_r, self.casdm1frs[ifrag])):
                overlap = np.vdot(c1, c0)
                tdm1s = np.stack(dm1s, axis=0) - overlap * dm1s_ref
                tdm1s = tdm1s + tdm1s.swapaxes(-1, -2).conj()
                tdm1rs[iroot, :, i:j, i:j] = tdm1s

        return tdm1rs

    def get_h1eff_response(self, tdm1rs):
        """Build the effective one-electron response from other cells.

        This linearizes the Coulomb/exchange part of the fragment projection.
        For each output cell, its own transition-density contribution is
        subtracted, leaving the different-cell CI response.
        """
        expected_shape = (self.nroots, 2, self.ncas, self.ncas)
        tdm1rs = np.asarray(tdm1rs)
        if tdm1rs.shape != expected_shape:
            msg = (f"tdm1rs must have shape {expected_shape}; "
                   f"got {tdm1rs.shape}")
            raise ValueError(msg)

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

    def _get_Hdiag(self):
        """Return the CI-only diagonal in the operator's CSF ordering."""
        hci_diag = self._get_Hci_diag()
        if not hci_diag:
            return np.empty(0, dtype=np.result_type(self.eri_cas.dtype))
        return np.concatenate(hci_diag)

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
        """Shape of the temporary CI-only Hessian operator."""
        return self.nvar_ci, self.nvar_ci

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

    def _matvec(self, x):
        """Apply the CI-only Hessian to a flat trial CI vector.

        ``x`` contains packed CSF amplitudes for every non-frozen cell and
        root. Hamiltonian and RDM operations are evaluated in the determinant
        representation before the response is transformed back to CSFs.
        """
        ci1 = self._unpack_ci_vector(x)

        tdm1rs = self.make_tdm1s_sub(ci1)
        h1frs_response = self.get_h1eff_response(tdm1rs)
        ci2_diag = self.ci_response_diag(ci1)
        ci2_offdiag = self.ci_response_offdiag(h1frs_response)

        ci2 = []
        for diag_r, offdiag_r, ci1_r in zip(
                ci2_diag, ci2_offdiag, ci1):
            ci2.append([
                diag + offdiag + self.level_shift * trial
                for diag, offdiag, trial in zip(diag_r, offdiag_r, ci1_r)
            ])

        return self._flatten_ci_vector(ci2)


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
            trans_sym_tol=1e-8):
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
        )

    @staticmethod
    def _normalize_phase_per_frag(phase_per_frag, ncell, ref_cell):
        """Validate cell phases and use the reference cell as phase origin."""
        phase_per_frag = np.asarray(
            phase_per_frag, dtype=np.result_type(phase_per_frag, np.complex128),
        )
        if phase_per_frag.shape != (ncell,):
            raise ValueError(
                f"phase_per_frag must have shape ({ncell},); "
                f"got {phase_per_frag.shape}"
            )
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
                if np.shape(ci_r[iroot]) != ref_shape:
                    raise ValueError(
                        "translated CI vectors have inconsistent shapes"
                    )
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

    def _init_dms_(self, casdm1frs):
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
        KLASSCF_HessianOperator._init_dms_(self, casdm1frs)

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
            (self.nroots, 2, self.ncas, self.ncas), dtype=dtype,
        )
        for ifrag, ncas in enumerate(self.ncas_sub):
            i = int(np.sum(self.ncas_sub[:ifrag]))
            j = i + int(ncas)
            tdm1rs[:, :, i:j, i:j] = tdm1_ref
        return tdm1rs

    def get_h1eff_response(self, tdm1rs):
        """Build one translated effective-Hamiltonian response and copy it."""
        expected_shape = (self.nroots, 2, self.ncas, self.ncas)
        tdm1rs = np.asarray(tdm1rs)
        if tdm1rs.shape != expected_shape:
            raise ValueError(
                f"tdm1rs must have shape {expected_shape}; got {tdm1rs.shape}"
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

    def _matvec(self, x):
        """Project a full CI vector onto the translation-adapted subspace."""
        ci1 = self._unpack_ci_vector(x)
        ci1 = self._unpack_cif(self._pack_ci(ci1))
        x_trans = self._flatten_ci_vector(ci1)
        return KLASSCF_HessianOperator._matvec(self, x_trans)
