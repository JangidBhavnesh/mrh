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
            self.linkstrl.append(
                fcibox.states_gen_linkstr(norb, nelec, True)
            )
            self.linkstr.append(
                fcibox.states_gen_linkstr(norb, nelec, False)
            )
        hc0 = self.Hci_all(None, self.h1frs, self.eri_cas, self.ci)
        self.e0 = [[np.vdot(c, hc) for c, hc in zip(ci_r, hc_r)] 
                   for ci_r, hc_r in zip(self.ci, hc0)]
        self.hci0 = [[hc - energy * c for hc, energy, c in zip(hc_r, e_r, ci_r)] 
                     for hc_r, e_r, ci_r in zip(hc0, self.e0, self.ci)]
