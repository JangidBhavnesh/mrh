#!/usr/bin/env python

import numpy as np

from mrh.my_pyscf.mcscf.lasscf_sync_o0 import (
    LASSCF_HessianOperator as molLASSCF_HessianOperator,
)

class KLASSCF_HessianOperator(molLASSCF_HessianOperator):
    """Periodic scaffold for the k-LASSCF Hessian operator.

    Only the constructor is implemented for now.  In particular, the
    molecular ``_matvec`` inherited from the parent class must not be used
    until the periodic CI-hop methods have been supplied.

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
        self.nvar_ci = sum(
            np.size(c0) for ci0_r in self.ci for c0 in ci0_r
        )

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
            linkstrl = (
                None if self.linkstrl is None else self.linkstrl[ifrag]
            )
            response.append(self.Hci(
                fcibox, norb, nelec, h0_r, h1rs, zero_h2, ci0_r,
                linkstrl=linkstrl,
            ))

        response = [
            [
                2.0 * (hc - np.vdot(c0, hc) * c0)
                for hc, c0 in zip(response_r, ci0_r)
            ]
            for response_r, ci0_r in zip(response, self.ci)
        ]
        return response

    @property
    def shape(self):
        """Shape of the temporary CI-only Hessian operator."""
        return self.nvar_ci, self.nvar_ci

    def _matvec(self, x):
        """Apply the CI-only Hessian to a flat trial CI vector.

        For now, ``x`` contains determinant-basis CI amplitudes for every
        cell and root in the same order as ``self.ci``. The returned vector
        uses exactly the same ordering and has the same size.
        """
        x = np.asarray(x)
        x_flat = x.reshape(-1)
        if x_flat.size != self.nvar_ci:
            raise ValueError(
                f"trial vector has size {x_flat.size}; expected {self.nvar_ci}"
            )

        ci1 = []
        offset = 0
        for ci0_r in self.ci:
            ci1_r = []
            for c0 in ci0_r:
                size = np.size(c0)
                ci1_r.append(
                    x_flat[offset:offset + size].reshape(np.shape(c0))
                )
                offset += size
            ci1.append(ci1_r)

        tdm1rs = self.make_tdm1s_sub(ci1)
        h1frs_response = self.get_h1eff_response(tdm1rs)
        ci2_diag = self.ci_response_diag(ci1)
        ci2_offdiag = self.ci_response_offdiag(h1frs_response)

        ci2 = []
        for diag_r, offdiag_r, ci1_r in zip(
                ci2_diag, ci2_offdiag, ci1):
            for diag, offdiag, trial in zip(diag_r, offdiag_r, ci1_r):
                ci2.append(
                    np.asarray(diag + offdiag + self.level_shift * trial)
                    .reshape(-1)
                )

        return np.concatenate(ci2)
