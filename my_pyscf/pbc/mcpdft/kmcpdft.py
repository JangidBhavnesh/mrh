import numpy as np
from functools import reduce

from pyscf.lib import logger
from pyscf.mcpdft.mcpdft import _PDFT
from pyscf.mcpdft import _dms
from pyscf.pbc.dft import gen_grid as pbc_gen_grid
from pyscf.pbc.lib import kpts_helper

from mrh.my_pyscf.pbc.mcpdft.otfnalperiodic import (
    get_pbc_otfnal_kpts,
    otfnalperiodic_kpts,
)
from mrh.my_pyscf.pbc.mcscf.k2R import get_mo_coeff_k2R
from mrh.my_pyscf.pbc.mcpdft._dms import dm2_cumulant_complex
from mrh.my_pyscf.pbc.mcpdft import kmcpdft_helper
'''
Author: Bhavnesh Jangid
k-MC-PDFT for periodic systems at the gamma point or k-points.
'''

_get_fcisolver = _dms._get_fcisolver

# Need to redefine the casdm1s and casdm2 because of shape mismatch.
def make_one_casdm1s (mc, ci, state=0):
    '''
    Spin-separated 1-RDMs.
    Note: the returned RDM1a, and RDM1b are in the shape of (ncas*nkpts, ncas*nkpts)
    and not (ncas, ncas) and in wannier orbital basis. Transform it before using 
    it k-pts machinary.
    '''
    nkpts = mc.nkpts
    ncastot = mc.ncas *  nkpts
    fcisolver, ci, nelecas = _get_fcisolver (mc, ci, state=state)
    nelecastot = (nelecas[0]*nkpts, nelecas[1]*nkpts)
    return fcisolver.make_rdm1s (ci, ncastot, nelecastot)

def make_one_casdm2 (mc, ci, state=0):
    '''
    Spin-summed 2-RDM
    Note: the returned RDM2 is in the shape of (ncas*nkpts, ncas*nkpts, ncas*nkpts, ncas*nkpts)
    and not (ncas, ncas, ncas, ncas) and in wannier orbital basis. Transform it before using 
    it k-pts machinary.
    '''
    ncas = mc.ncas
    fcisolver, ci, nelecas = _get_fcisolver (mc, ci, state=state)
    ncastot = ncas * mc.nkpts
    nelecastot = (nelecas[0]*mc.nkpts, nelecas[1]*mc.nkpts)
    try:
        casdm2 = fcisolver.make_rdm2 (ci, ncastot, nelecastot)
    except AttributeError:
        _, casdm2 = fcisolver.make_rdm12 (ci, ncastot, nelecastot)
    return casdm2

def energy_mcwfn(mc, mo_coeff=None, ci=None, ot=None, state=0, casdm1s=None,
                casdm2=None, verbose=None):
    # See pyscf.mcpdft.mcpdft.energy_mcwfn for details.

    if ot is None: ot = mc.otfnal
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if verbose is None: verbose = mc.verbose
    if casdm1s is None: casdm1s = mc.make_one_casdm1s(ci=ci, state=state)
    if casdm2 is None: casdm2 = mc.make_one_casdm2(ci=ci, state=state)
    
    cell = mc._scf.cell
    nkpts = mc.nkpts
    ncore = mc.ncore
    ncas = mc.ncas
    kmesh = mc.kmesh

    # Get the MO_PHASE:
    mo_phase = get_mo_coeff_k2R(mc._scf, mo_coeff, ncore, ncas, kmesh=kmesh)[-1]
    log = logger.new_logger(mc, verbose=verbose)

    # First, transform the casdm1s to dm1s for each k-point.
    dm1s_kpts = []
    for k in range(nkpts):
        casdm1s_k = [reduce(np.dot, (mo_phase[k], casdm1s_, mo_phase[k].conj().T)) 
                    for casdm1s_ in casdm1s]
        dm1s =_dms.casdm1s_to_dm1s (ot, casdm1s_k, mo_coeff=mo_coeff[k], ncore=ncore, 
                                        ncas=ncas)
        dm1s_kpts.append(dm1s)
    
    # Making sure the tagging the dm1s doesn't create the weird problems
    # for pbc.
    dm1s_kpts = np.stack([np.asarray(dm1s) 
                          for dm1s in dm1s_kpts], axis=1,)
    
    hyb_x, hyb_c = ot._numint.rsh_and_hybrid_coeff(ot.otxc, mc.mol.spin)[2]

    Vnn = mc.energy_nuc()
    h1e_kpts = mc.get_hcore(kpts=mc.kpts)
    
    assert h1e_kpts.ndim == 3 and dm1s_kpts.ndim == 4 and \
        h1e_kpts.shape == dm1s_kpts[0].shape == dm1s_kpts[1].shape, \
            'h1e_kpts and dm1s_kpts must have shape (nkpts,nao,nao)'

    dm1_kpts = np.array([dm1s_kpts[0][i] + dm1s_kpts[1][i] 
                         for i in range(nkpts)])
    
    if log.verbose >= logger.DEBUG or abs(hyb_x) > 1e-10:
        vj_kpts, vk_kpts = mc._scf.get_jk(cell, dm_kpts=dm1s_kpts, kpts=mc.kpts)
        vj_kpts = vj_kpts[0] + vj_kpts[1] # (nkpts, nao, nao)
    else:
        vj_kpts = mc._scf.get_jk(cell, dm_kpts=dm1_kpts, kpts=mc.kpts, 
                                 hermi=1, with_k=False)[0]
        
        
    Te_Vne = 1./nkpts * np.einsum('kij,kji->', h1e_kpts, dm1_kpts)
    E_j = 1./nkpts * np.einsum('kij,kji->',vj_kpts, dm1_kpts) * 0.5

    log.debug('CAS energy decomposition:')
    log.debug('Vnn = %s', Vnn)
    log.debug('Te + Vne = %s', Te_Vne)
    log.debug('E_j = %s', E_j)

    # Keeping this warning as it is.
    if abs(hyb_x - hyb_c) > 1e-10:
        log.warn("exchange and correlation hybridization differ")
        log.warn("may lead to unphysical results, see https://github.com/pyscf/pyscf-forge/issues/128")

    # Note: this is not the true exchange energy, but just the HF-like exchange
    E_x = 0.0
    if log.verbose >= logger.DEBUG or abs(hyb_x) > 1e-10:
        # (vk_a * dm_a) + (vk_b * dm_b)
        E_x = -1/nkpts * (np.einsum('kij,kji->', vk_kpts[0], dm1s_kpts[0]) +
                         np.einsum('kij,kji->', vk_kpts[1], dm1s_kpts[1]))
        E_x /= 2.0
        log.debug("E_x = %s", E_x)
        log.debug("Adding (%s) * E_x = %s", hyb_x, hyb_x * E_x)

    # This is not correlation, but the 2-body cumulant tensored with the eri's:
    # g_pqrs * l_pqrs / 2
    E_c = 0.0
    if log.verbose >= logger.DEBUG or abs(hyb_c) > 1e-10:
        # Now compute the cascm2:
        cascm2 = dm2_cumulant_complex(casdm2, casdm1s)
        aeri = mc.get_h2eff(mo_coeff = mo_coeff)
        ncastot = mc.ncas * mc.nkpts
        assert aeri.ndim == 4 and aeri.shape == (ncastot,)*4
        E_c = np.tensordot(aeri, cascm2, axes=4) / (2 * nkpts)
        log.debug("E_c = %s", E_c)
        log.debug("Adding (%s) * E_c = %s", hyb_c, hyb_c * E_c)

    e_mcwfn = Vnn + Te_Vne + E_j + (hyb_x * E_x) + (hyb_c * E_c)
    
    return e_mcwfn


def contract_kcas_cumulant(h2eff, cascm2_kpts, nkpts):
    """Contract the kCASCI ERIs with the cumulant per primitive cell.

    ``PBCKCASCI.get_h2eff`` includes both the supercell ``1 / nkpts``
    normalization and the factor of one half used by the k-FCI Hamiltonian.
    Dividing the contraction by ``nkpts`` therefore produces the MC-PDFT
    cumulant contribution per primitive cell.
    """
    h2eff = np.asarray(h2eff)
    cascm2_kpts = np.asarray(cascm2_kpts)
    if h2eff.ndim != 7 or h2eff.shape != cascm2_kpts.shape:
        raise ValueError(
            "h2eff and cascm2_kpts must have matching seven-dimensional "
            "k-space shapes",
        )
    if h2eff.shape[:3] != (nkpts, nkpts, nkpts):
        raise ValueError("The ERI k-point dimensions do not match nkpts")
    return np.einsum(
        "abcuvxy,abcuvxy->", h2eff, cascm2_kpts,
        optimize=True,
    ) / nkpts


def energy_mcwfn_kcas(mc, mo_coeff=None, ci=None, ot=None, state=0,
                      casdm1s=None, casdm2=None, verbose=None,
                      momentum_tol=1e-8):
    """Compute the MC wavefunction part from momentum-resolved kCAS RDMs."""
    if ot is None:
        ot = mc.otfnal
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci is None:
        ci = mc.ci
    if verbose is None:
        verbose = mc.verbose
    if casdm1s is None:
        casdm1s = mc.make_one_casdm1s(ci=ci, state=state)
    if casdm2 is None:
        casdm2 = mc.make_one_casdm2(ci=ci, state=state)

    mo_coeff = np.asarray(mo_coeff)
    nkpts = mc.nkpts
    ncas = mc.ncas
    ncore = mc.ncore
    if mo_coeff.ndim != 3 or mo_coeff.shape[0] != nkpts:
        raise ValueError("mo_coeff must have shape (nkpts, nao, nmo)")

    kconserv = getattr(mc, "kconserv", None)
    if kconserv is None:
        kconserv = kpts_helper.get_kconserv(mc.cell, mc.kpts)
    casdm1s_kpts, cascm2_kpts = \
        kmcpdft_helper.make_kcas_rdms_kpts(
            casdm1s, casdm2, nkpts, ncas, kconserv,
            momentum_tol=momentum_tol,
        )
    dm1s_kpts = kmcpdft_helper.casdm1s_kpts_to_dm1s(
        mc, casdm1s_kpts, mo_coeff, ncore,
    )
    dm1_kpts = dm1s_kpts[0] + dm1s_kpts[1]

    log = logger.new_logger(mc, verbose=verbose)
    hyb_x, hyb_c = ot._numint.rsh_and_hybrid_coeff(
        ot.otxc, mc.cell.spin,
    )[2]
    energy_nuc = mc.energy_nuc()
    h1e_kpts = np.asarray(mc.get_hcore(kpts=mc.kpts))
    if h1e_kpts.shape != dm1_kpts.shape:
        raise ValueError(
            "hcore and AO density matrices must have shape "
            "(nkpts, nao, nao)",
        )

    if log.verbose >= logger.DEBUG or abs(hyb_x) > 1e-10:
        vj_spin, vk_kpts = mc._scf.get_jk(
            mc.cell, dm_kpts=dm1s_kpts, kpts=mc.kpts,
        )
        vj_kpts = vj_spin[0] + vj_spin[1]
    else:
        vj_kpts = mc._scf.get_jk(
            mc.cell, dm_kpts=dm1_kpts, kpts=mc.kpts,
            hermi=1, with_k=False,
        )[0]
        vk_kpts = None

    energy_one = np.einsum(
        "kij,kji->", h1e_kpts, dm1_kpts,
        optimize=True,
    ) / nkpts
    energy_j = 0.5 * np.einsum(
        "kij,kji->", vj_kpts, dm1_kpts,
        optimize=True,
    ) / nkpts

    if abs(hyb_x - hyb_c) > 1e-10:
        log.warn("exchange and correlation hybridization differ")
        log.warn(
            "may lead to unphysical results, see "
            "https://github.com/pyscf/pyscf-forge/issues/128",
        )

    energy_x = 0.0
    if log.verbose >= logger.DEBUG or abs(hyb_x) > 1e-10:
        energy_x = -0.5 * (
            np.einsum(
                "kij,kji->", vk_kpts[0], dm1s_kpts[0],
                optimize=True,
            )
            + np.einsum(
                "kij,kji->", vk_kpts[1], dm1s_kpts[1],
                optimize=True,
            )
        ) / nkpts

    energy_c = 0.0
    if log.verbose >= logger.DEBUG or abs(hyb_c) > 1e-10:
        h2eff = mc.get_h2eff(mo_coeff=mo_coeff)
        energy_c = contract_kcas_cumulant(
            h2eff, cascm2_kpts, nkpts,
        )

    log.debug("CAS energy decomposition:")
    log.debug("Vnn = %s", energy_nuc)
    log.debug("Te + Vne = %s", energy_one)
    log.debug("E_j = %s", energy_j)
    log.debug("E_x = %s", energy_x)
    log.debug("E_c = %s", energy_c)
    return (
        energy_nuc + energy_one + energy_j
        + hyb_x * energy_x + hyb_c * energy_c
    )


def energy_dft_kcas(mc, mo_coeff=None, ci=None, ot=None, state=0,
                    casdm1s=None, casdm2=None, max_memory=None, hermi=1,
                    momentum_tol=1e-8):
    """Evaluate the on-top functional directly from momentum kCAS RDMs."""
    if ot is None:
        ot = mc.otfnal
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci is None:
        ci = mc.ci
    if casdm1s is None:
        casdm1s = mc.make_one_casdm1s(ci, state=state)
    if casdm2 is None:
        casdm2 = mc.make_one_casdm2(ci, state=state)
    if max_memory is None:
        max_memory = mc.max_memory
    return ot.energy_ot_kcas(
        casdm1s, casdm2, mo_coeff, mc.ncore,
        max_memory=max_memory, hermi=hermi,
        momentum_tol=momentum_tol,
    )


class _kMCPDFT(_PDFT):
    '''
    k-MC-PDFT for periodic systems at the gamma point or k-points.
    This class is adding or replacing the functionalities which are not 
    compatible with periodic systems are throwing NotImplementedError. 
    '''

    momentum_resolved = False

    def _init_ot_grids(self, my_ot, grids_attr=None):
        '''
        Initialization of on-top functional and grids for periodic systems.
        '''
        if grids_attr is None:
            grids_attr = {}

        old_grids = getattr(self, 'grids', None)

        if isinstance(my_ot, (str, np.bytes_)):
            # Note: I have changed the input arg. for below function.
            self.otfnal = get_pbc_otfnal_kpts(self._scf, my_ot)
        else:
            self.otfnal = my_ot

        pbc_grid_types = (
            pbc_gen_grid.UniformGrids,
            pbc_gen_grid.BeckeGrids,
        )

        if isinstance(old_grids, pbc_grid_types):
            self.otfnal.grids = old_grids
        else:
            self.otfnal.grids = pbc_gen_grid.BeckeGrids(self.cell,)

        self.otfnal.grids.__dict__.update(grids_attr)

        for key, value in grids_attr.items():
            assert getattr(self.otfnal.grids, key, None) == value

        self.otfnal.verbose = self.verbose
        self.otfnal.stdout = self.stdout    
    
    def multi_state(self, method='Lin'):
        raise NotImplementedError(f"StateAverageMix not available for {method}")

    make_one_casdm1s = make_one_casdm1s
    make_one_casdm2 = make_one_casdm2
    energy_mcwfn = energy_mcwfn

    def dump_chk(self, *args, **kwargs):
        logger.warn(self, "dump_chk is not supported for k-MC-PDFT")
        pass

    def nuc_grad_method(self):
        raise NotImplementedError("Nuclear gradients are not implemented for k-MC-PDFT")
    
    def dip_moment(self):
        raise NotImplementedError("Dipole moment is not implemented for k-MC-PDFT")
    
    def get_energy_decomposition(self, *args, **kwargs):
        raise NotImplementedError("Energy decomposition is not implemented for k-MC-PDFT")

    def update_from_chk(self, chkfile=None, **kwargs):
        raise NotImplementedError("update_from_chk is not implemented for k-MC-PDFT")


class _kKCASPDFT(_kMCPDFT):
    """k-MC-PDFT specialization for one total-momentum kCAS sector."""

    momentum_resolved = True

    make_one_casdm1s = kmcpdft_helper.make_one_casdm1s_kcas
    make_one_casdm2 = kmcpdft_helper.make_one_casdm2_kcas
    energy_mcwfn = energy_mcwfn_kcas
    energy_dft = energy_dft_kcas


def _get_mcpdft_child_class(kmc, ot, pdft_base, **kwargs):
    mc_doc = (kmc.__class__.__doc__ or 'No docstring for MC-SCF parent method')

    class PDFT(pdft_base, kmc.__class__):
        __doc__ = mc_doc + '\n\n' + pdft_base.__doc__
        _mc_class = kmc.__class__

        # MC-PDFT object requires mol object in ot.reset functions
        _mc_class.mol = kmc._scf.cell.to_mol()
        
        def compute_pdft_energy_(self, mo_coeff=None, ci=None, ot=None, otxc=None,
                                 grids_level=None, grids_attr=None, dump_chk=False, **kwargs):
            # Some sanity checks:
            if ot is not None:
                assert isinstance(ot, otfnalperiodic_kpts)
                cell_kpts_info = [getattr(ot, 'kmesh', None), 
                                  getattr(ot, 'kpts', None), 
                                  getattr(ot, 'cell', None)]
                assert all(value is not None for value in cell_kpts_info), \
                    "The kmesh and kpts attributes should be set in the otfnal object"
            assert dump_chk is False, "dump_chk is not supported for k-MC-PDFT"
            return _kMCPDFT.compute_pdft_energy_(self, mo_coeff=mo_coeff, ci=ci, ot=ot, otxc=otxc,
                    grids_level=grids_level, grids_attr=grids_attr, dump_chk=False, **kwargs)
     
    pdft = PDFT(kmc._scf, kmc.ncas, kmc.nelecas, my_ot=ot, **kwargs)
    _keys = pdft._keys.copy()
    pdft.__dict__.update(kmc.__dict__)
    pdft._keys = pdft._keys.union(_keys)
    pdft._keys.add("momentum_resolved")
    return pdft


def get_mcpdft_child_class(kmc, ot, **kwargs):
    """Wrap a conventional periodic CAS object with k-MC-PDFT methods."""
    return _get_mcpdft_child_class(kmc, ot, _kMCPDFT, **kwargs)


def get_kcas_mcpdft_child_class(kmc, ot, **kwargs):
    """Wrap a momentum-resolved kCASCI object with k-MC-PDFT methods."""
    pdft = _get_mcpdft_child_class(kmc, ot, _kKCASPDFT, **kwargs)
    if getattr(kmc, "converged", False):
        pdft.e_mcscf = kmc.e_tot
    return pdft
