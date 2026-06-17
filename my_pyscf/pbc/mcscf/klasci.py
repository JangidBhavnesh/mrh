# /usr/bin/env python3

import numpy as np
import scipy.linalg
from functools import reduce

from pyscf import lib
from pyscf.pbc import scf, dft, df
from pyscf.pbc.lib import kpts_helper

from mrh.my_pyscf.mcscf.lasci import LASCINoSymm
from mrh.my_pyscf.pbc.mcscf import casci
from mrh.my_pyscf.pbc.util.transym import TranslationSymm, get_wannier_orbs

# Author: Bhavnesh Jangid

'''
# Let's start implementing the k-LAS algorithm.
# Steps
1. Implement the k-LASCI
2. Implement the k-LASSCF
3. Implement the LASSI algorithm
'''

def kLASCI(kmf, ncas, nelecas, ncore=None, spin_mult=None, kmesh=None, kpts=None):
    '''
    Wrapper function for k-LASCI. 
    args:
        kmf: pbc.scf object
            mean-field object
        ncas: int
            number of active orbitals per unit cell
        nelecas: int/tuple
            number of active electrons per unit cell
        ncore: int, optional
            number of core orbitals per unit cell
        spin_mult: int, optional (2S + 1)
            spin multiplicity of the active space in the unit cell.
            If not provided, it will be automatically determined based 
            on the number of active electrons.
        kmesh: tuple, optional
            k-point mesh for the calculation.
        kpts: array_like, optional
            k-points for the calculation. 
    returns:
        klas: k-LASCI object
    '''
    assert isinstance(kmf, scf.hf.SCF),  \
        "k-LASCI only works with periodic SCF objects"
    
    if kmf.cell.symmetry:
        raise NotImplementedError("k-LASCI with symmetry is not implemented yet.")
    
    # Sanity check to make sure that DFT mean field objects are not passed to k-LASCI
    if isinstance(kmf, dft.krks.KRKS) or isinstance(kmf, dft.kuks.KUKS) \
        or isinstance(kmf, dft.rks.RKS) or isinstance(kmf, dft.uks.UKS):
        raise NotImplementedError("k-LASCI with DFT is not implemented yet.")
    
    # If the mean-field object is KUHF, convert it to RHF before passing to k-LASCI,
    if isinstance(kmf, scf.kuhf.KUHF):
        kmf = scf.addons.convert_to_rhf(kmf)
    
    # Currently, the k-LAS should work with the GDF density fitting object.
    assert isinstance(kmf.with_df, df.df.GDF), \
        "k-LASCI only works with GDF density fitting object"

    klas = PBCLASCINoSymm(kmf, ncas, nelecas, ncore=ncore, spin_mult=spin_mult, kmesh=kmesh, kpts=kpts)

    return klas

@lib.with_doc(casci.h1e_for_cas.__doc__)
def h1e_for_cas(mc, mo_coeff=None, ncas=None, ncore=None):
    # The difference between this function and one defined in pbc.mcscf.casci is that here 
    # we are constructing the h1e in the Wannier basis, which is different from the k-space MO basis 
    # used in standard PBCASCI.

    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ncas is None: ncas = mc.ncas
    if ncore is None: ncore = mc.ncore
    
    cell = mc.cell
    nao = cell.nao_nr()
    kpts = mc._scf.kpts
    kmesh = mc.kmesh
    dtype = mo_coeff[0].dtype
    nkpts = mc.nkpts

    mo_core_kpts = [mo[:, :ncore] for mo in mo_coeff]
    mo_act_kpts = [mo[:, ncore:ncore+ncas] for mo in mo_coeff]

    h1ao_k = mc.get_hcore().astype(dtype)

    # Remember, I am multiplying by nkpts here because total energy would be divided by nkpts later.
    ecore = mc.energy_nuc() * nkpts
    if ncore == 0:
        corevhf_kpts = 0
    else:
        coredm_kpts = np.asarray([2.0 * (mo_core_kpts[k] @ mo_core_kpts[k].conj().T) 
                                  for k in range(nkpts)], dtype=dtype)
        # corevhf_kpts = mc._scf.get_veff(cell, coredm_kpts, hermi=1)
        corevhf_kpts = mc.get_veff(cell, coredm_kpts, hermi=1, kpts=mc._scf.kpts)
        fock = h1ao_k + 0.5 * corevhf_kpts
        ecore += sum(np.einsum('ij,ji', coredm_kpts[k], fock[k]) for k in range(nkpts))
        fock = None  # Free memory

    h1ao_k += corevhf_kpts

    # Fourier transform h1ao_k to real space to get h1ao_R, which is the one we will use 
    # for constructing the effective 1e Hamiltonian in the Wannier basis.
    ts = TranslationSymm(cell, kmesh)
    R_indices = ts.lattice_indices(kmesh)
    R_cart = np.array([ts.lattice_cart(R) for R in R_indices])
    ncell = len(R_indices)

    assert nkpts == ncell
    assert np.prod(kmesh) == nkpts

    h1ao_R = np.zeros((ncell, nao, ncell, nao), dtype=dtype)

    for ik, k in enumerate(kpts):
        hk = h1ao_k[ik]
        for iR, Rv in enumerate(R_cart):
            for iS, Sv in enumerate(R_cart):
                phase = np.exp(1j * np.dot(k, Rv - Sv))
                h1ao_R[iR, :, iS, :] += phase * hk

    h1ao_R /= nkpts
    h1ao_R = h1ao_R.reshape(nkpts*nao, nkpts*nao)

    wannier_orb, R_indices_check = get_wannier_orbs(mc._scf, kmesh, mo_act_kpts)[:2]
    wannier_orb = wannier_orb.reshape(nkpts*nao, nkpts*ncas)

    # Sanity check to make sure that the R indices from TranslationSymm and get_wannier_orbs match.
    assert np.array_equal(R_indices, R_indices_check), "Something is wrong."
    
    # Transform h1ao_R to the Wannier basis.
    h1eff_R = reduce(np.dot, (wannier_orb.conj().T, h1ao_R, wannier_orb))

    return h1eff_R, ecore

def h2e_for_cas(mc, mo_coeff=None):
    '''
    AO2MO Transformation of the 2e integrals in the Wannier basis for k-LASCI.
    The way we do this is by first transforming the 2e integrals from k-space MO basis to 
    real space MO basis, and then transforming it to the Wannier basis using the wannier 
    orbital coefficients.
    args:
        mc: PBCCASCI object
            periodic CASCI object
        mo_coeff: list/np.ndarray (nkpts, nao, nmo)
            MO coefficients for each k-point. If None, it will be taken from the 
            PBCCASCI object.
    returns:
        h2eff_R: ndarray (nkpts*ncas, nkpts*ncas, nkpts*ncas, nkpts*ncas)
            2e integrals in the Wannier basis in real space.
    '''
    kmf = mc._scf
    cell = kmf.cell
    ncore = mc.ncore
    ncas = mc.ncas
    nkpts = mc.nkpts
    kpts = kmf.kpts
    kmesh = mc.kmesh

    kconserv = kpts_helper.get_kconserv(cell, kpts)
    
    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    
    # Do the ao2mo transformation in the k-space MO basis first to get eri_k, 
    # which is in the k-space MO basis.
    mo_act_kpts = np.array([mo_coeff[i][:, ncore:ncore+ncas] for i in range(nkpts)])
    eri_k = kmf.with_df.ao2mo_7d(mo_act_kpts, kpts=kpts)
    
    # Get the mo phase for the active space orbitals
    mo_phase = get_wannier_orbs(kmf, kmesh, mo_act_kpts)[-1]
    mo_ks = mo_phase[kconserv]

    # This einsum looks very scary but it is just the transformation of the eris from 
    # k-space mo to r-space mo.
    eris = np.einsum('auR,bvS,abcuvwt,cwT,abctU->RSTU',
                        mo_phase.conj(), mo_phase, eri_k, mo_phase.conj(), mo_ks, optimize=True)
    eris *= 1.0/nkpts
    
    assert eris.shape == (nkpts*ncas, nkpts*ncas, nkpts*ncas, nkpts*ncas)
    return eris
    
class _PBCCASCIForLAS(casci.PBCCASCI):
    '''
    Child class of PBCCASCI for k-LASCI. Basically the way the h1e and h2e
    integrals are constructed in the k-LASCI is different from the standard PBCASCI.
    '''
    get_h1eff = h1e_for_cas
    get_h2eff = h2e_for_cas

class PBCLASCINoSymm(_PBCCASCIForLAS, LASCINoSymm):
    '''
    Localized active space CI (LASCI) class for periodic systems without 
    point group symmetry.
    args:
        kmf: pbc.scf object
            mean-field object
        ncas: int
            number of active orbitals per unit cell
        nelecas: int/tuple
            number of active electrons per unit cell
        spin_mult: int, optional (2S + 1)
            spin multiplicity of the active space in the unit cell.
            If not provided, it will be automatically determined based 
            on the number of active electrons.
        ncore: int, optional
            number of core orbitals per unit cell
        kmesh: tuple, optional
            k-point mesh for the calculation.
        kpts: array_like, optional
            k-points for the calculation.
    '''
    def __init__(self, kmf, ncas, nelecas, ncore=None, spin_mult=None, 
                 kmesh=None, kpts=None):
        self.spin_mult = spin_mult
        self.kmesh = kmesh
        self.kpts = kpts if kpts is not None else kmf.kpts
        nkpts = len(self.kpts)
        if kmesh is not None:
            assert nkpts == np.prod(kmesh), "kmesh and kpts do not match."
        self.ncas_sub = nkpts * (ncas,)
        if isinstance(nelecas, int):
            self.nelecas_sub = nkpts * (nelecas,)
        elif isinstance(nelecas, tuple) and len(nelecas) == 2:
            self.nelecas_sub = nkpts * (nelecas,)
        self.spin_sub = spin_mult * nkpts if spin_mult is not None else None
        self.nroots = 1

        # Initialize the parent classes.
        _PBCCASCIForLAS.__init__(self, kmf, ncas, nelecas, ncore=ncore)
        LASCINoSymm.__init__(self, kmf, ncas=self.ncas_sub, nelecas=self.nelecas_sub, ncore=ncore, spin_sub=self.spin_sub)
        
        # Making sure this is for an unit cell, not summed over multiple unit cells.
        self.ncas = ncas
        self.nelecas = nelecas

        # the total active space should be stored as ncastot, nelecstot
        self.ncastot = sum(self.ncas_sub)
        if isinstance(self.nelecas_sub[0], int):
            self.nelecstot = sum(self.nelecas_sub)
        else:
            self.nelecstot = tuple(map(sum, zip(*self.nelecas_sub)))
