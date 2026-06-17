# /usr/bin/env python3

import numpy as np
import scipy.linalg
from functools import reduce

from pyscf import lib
from pyscf.fci import cistring
from pyscf.pbc import scf, dft, df
from pyscf.pbc.lib import kpts_helper

from mrh.my_pyscf.mcscf import lasci as mollasci
from mrh.my_pyscf.mcscf.lasci import LASCINoSymm
from mrh.my_pyscf.mcscf.addons import state_average_n_mix, get_h1e_zipped_fcisolver
from mrh.my_pyscf.mcscf.productstate import ImpureProductStateFCISolver

from mrh.my_pyscf.pbc.mcscf import casci
from mrh.my_pyscf.pbc.util.transym import TranslationSymm, get_wannier_orbs
from mrh.my_pyscf.pbc.fci import csf_solver

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

    klas = PBCLASCINoSymm(kmf, ncas, nelecas, ncore=ncore, spin_mult=spin_mult, 
                          kmesh=kmesh, kpts=kpts)

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

@lib.with_doc(casci.PBCCASCI.get_h2eff.__doc__)
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

    assert kmesh is not None

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

def convert_h1e_mo_k_to_wann(kmf, kmesh, h1e_mo_k):
    '''
    Convert h1e from k-space localized/block MO basis to Wannier basis.
    args:
        kmf:
            PBC mean-field object.
        kmesh:
            k-point mesh.
        h1e_mo_k:
            ndarray of shape (nkpts, norb, norb)
    returns:
        h1eff_R:
            ndarray of shape (nkpts*norb, nkpts*norb)
    '''

    cell = kmf.cell
    kpts = kmf.kpts
    norb = h1e_mo_k.shape[1]
    dtype = np.result_type(h1e_mo_k.dtype, np.complex128)

    ts = TranslationSymm(cell, kmesh)
    R_indices = ts.lattice_indices(kmesh)
    R_cart = np.array([ts.lattice_cart(R) for R in R_indices])

    ncell = len(R_indices)
    nkpts = len(kpts)

    assert nkpts == np.prod(kmesh)
    assert ncell == nkpts

    h1e_mo_k = np.asarray(h1e_mo_k)
    assert h1e_mo_k.ndim == 3
    assert h1e_mo_k.shape[0] == nkpts
    assert h1e_mo_k.shape[1] == h1e_mo_k.shape[2]

    h1eff_R = np.zeros((ncell, norb, ncell, norb), dtype=dtype)
    for ik, k in enumerate(kpts):
        hk = h1e_mo_k[ik]
        for iR, Rv in enumerate(R_cart):
            for iS, Sv in enumerate(R_cart):
                phase = np.exp(1j * np.dot(k, Rv - Sv))
                h1eff_R[iR, :, iS, :] += phase * hk
    h1eff_R /= nkpts
    h1eff_R = h1eff_R.reshape(ncell * norb, ncell * norb)
    return h1eff_R

@lib.with_doc(mollasci.h1e_for_las.__doc__)
def h1e_for_las (las, mo_coeff=None, ncas=None, ncore=None, nelecas=None, ci=None, ncas_sub=None,
                 nelecas_sub=None, veff=None, h2eff_sub=None, casdm1s_sub=None, casdm1frs=None,
                 eri_cas=None):
    cell = las._scf.cell
    if mo_coeff is None: mo_coeff = las.mo_coeff
    if ncas is None: ncas = las.ncas
    if ncore is None: ncore = las.ncore
    if ncas_sub is None: ncas_sub = las.ncas_sub
    if nelecas_sub is None: nelecas_sub = las.nelecas_sub
    if ncore is None: ncore = las.ncore
    if ci is None: ci = las.ci
    if eri_cas is None: eri_cas = las.get_h2cas (mo_coeff)
    if casdm1frs is None: casdm1frs = las.states_make_casdm1s_sub (ci=ci)
    if casdm1s_sub is None: casdm1s_sub = [np.einsum ('rsij,r->sij',dm,las.weights)
                                           for dm in casdm1frs]
    if veff is None:
        veff = las.get_veff (cell, dm_kpts = las.make_rdm1s (mo_coeff=mo_coeff, casdm1s_sub=casdm1s_sub))
        # veff is in the k-space

    nkpts, nao, nmo = mo_coeff.shape
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    ncastot = nkpts * ncas

    # I will implement for the one root first, then we will see, how to go 
    # to multiple roots.
    # First pass: split by root
    hcore_k = las.get_hcore()
    h1e_k = np.empty ((2, nkpts, ncas, ncas), dtype=hcore_k.dtype)
    for k in range(nkpts):
        mo_cas = mo_coeff[k][:, ncore:nocc]
        moH_cas = mo_cas.conj ().T
        h1e_k[:, k] = moH_cas @ (hcore_k[k][None,:,:] + veff[:, k, :, :]) @ mo_cas

    h1e_mo_wann = np.array([convert_h1e_mo_k_to_wann(las._scf, las.kmesh, h1e_k[s]) 
                            for s in range(2)])
    h1e_k = hcore_k = None

    h1e_r = np.empty ((las.nroots, 2, ncastot, ncastot), dtype=h1e_mo_wann.dtype)

    assert (eri_cas.shape==(ncastot,ncastot,ncastot,ncastot))

    h2e = eri_cas

    # casdm1 is in wannier mo basis.
    avgdm1s = np.stack ([scipy.linalg.block_diag (*[dm[spin] for dm in casdm1s_sub])
                         for spin in range (2)], axis=0)
    for state in range (las.nroots):
        statedm1s = np.stack ([scipy.linalg.block_diag (*[dm[state][spin] for dm in casdm1frs])
                               for spin in range (2)], axis=0)
        dm1s = statedm1s - avgdm1s 
        j = np.tensordot (dm1s, h2e, axes=((1,2),(2,3)))
        k = np.tensordot (dm1s, h2e, axes=((1,2),(2,1)))
        h1e_r[state] = h1e_mo_wann + j + j[::-1] - k

    # Second pass: split by fragment and subtract double-counting
    h1e_fr = []
    for ix, casdm1s_r in enumerate (casdm1frs):
        p = sum (las.ncas_sub[:ix])
        q = p + las.ncas_sub[ix]
        h1e = h1e_r[:,:,p:q,p:q]
        h2e = eri_cas[p:q,p:q,p:q,p:q]
        j = np.tensordot (casdm1s_r, h2e, axes=((2,3),(2,3)))
        k = np.tensordot (casdm1s_r, h2e, axes=((2,3),(2,1)))
        h1e_fr.append (h1e - j - j[:,::-1] + k)

    return h1e_fr

@lib.with_doc(mollasci.kernel.__doc__)
def kernel (las, mo_coeff=None, ci0=None, lroots=None, lweights=None, verbose=0,
               assert_no_dupes=False, _dry_run=False):
    if assert_no_dupes: mollasci.assert_no_duplicates (las)
    if lroots is not None and lweights is not None:
        raise RuntimeError ("lroots sets lweights: pass either or none but not both")
    elif lweights is None:
        if lroots is None: lroots = np.ones ((las.nfrags, las.nroots), dtype=int)
        lweights = []
        for i in range (las.nfrags):
            lwi = []
            for j in range (las.nroots):
                lwij = np.zeros (lroots[i,j])
                lwij[0] = 1
                lwi.append (lwij)
            lweights.append (lwi)
    
    nkpts, nao, nmo = mo_coeff.shape
    ncore, ncas = las.ncore, las.ncas
    nocc = ncore + ncas
    ncas_sub = las.ncas_sub
    nelecas_sub = las.nelecas_sub
    orbsym = getattr (mo_coeff, 'orbsym', None)
    if orbsym is not None: orbsym=orbsym[ncore:nocc]
    elif isinstance (las, mollasci.LASCISymm):
        mo_coeff = las.label_symmetry_(mo_coeff)
        orbsym = mo_coeff.orbsym[ncore:nocc]
    log = lib.logger.new_logger (las, verbose)

    h1eff, energy_core = las.h1e_for_cas (mo_coeff=mo_coeff,
        ncas=las.ncas, ncore=las.ncore)
    eri_cas = las.get_h2cas (mo_coeff)
    
    if (ci0 is None or any ([c is None for c in ci0]) or
            any ([any ([c2 is None for c2 in c1]) for c1 in ci0])):
        ci0 = las.get_init_guess_ci (mo_coeff, ci0=ci0, eri_cas=eri_cas)

    e_cas = np.empty (las.nroots)
    e_states = np.empty (las.nroots)
    ci1 = [[None for c2 in c1] for c1 in ci0]
    converged = []
    t = (lib.logger.process_clock(), lib.logger.perf_counter())
    e_lexc = [[None for i in range (las.nroots)] for j in range (las.nfrags)]
    for state in range (las.nroots):
        fcisolvers = [b.fcisolvers[state] for b in las.fciboxes]
        ci0_i = [c[state] for c in ci0]
        solver = ImpureProductStateFCISolver (fcisolvers, stdout=las.stdout,
            lweights=[l[state] for l in lweights], verbose=verbose)
        for ix, s in enumerate (solver.fcisolvers):
            # Set the calling las objects local bottom-layer fcisolvers to the
            # locally-state-averaged ones I just made so that I can more easily get
            # locally-state-averaged density matrices after this function exits
            las.fciboxes[ix].fcisolvers[state] = s
            i = sum (ncas_sub[:ix])
            j = i + ncas_sub[ix]
            if orbsym is not None: s.orbsym = orbsym[i:j]
            s.norb = ncas_sub[ix]
            s.nelec = solver._get_nelec (s, nelecas_sub[ix])
            s.check_transformer_cache ()
        if _dry_run: continue
        conv, e_i, ci_i = solver.kernel (h1eff, eri_cas, ncas_sub, nelecas_sub,
            ecore=0, ci0=ci0_i, orbsym=orbsym, conv_tol_grad=las.conv_tol_grad,
            conv_tol_self=las.conv_tol_self, max_cycle_macro=las.max_cycle_macro)
        e_cas[state] = e_i
        e_states[state] = e_i + energy_core
        for frag, s in enumerate (solver.fcisolvers):
            e_loc = np.atleast_1d (getattr (s, 'e_states', e_i))
            e_lexc[frag][state] = e_loc - e_i
        for c1, c2, s, no, ne in zip (ci1, ci_i, solver.fcisolvers, ncas_sub, nelecas_sub):
            ne = solver._get_nelec (s, ne)
            ndeta, ndetb = [cistring.num_strings (no, n) for n in ne]
            shape = [s.nroots, ndeta, ndetb] if s.nroots>1 else [ndeta, ndetb]
            c1[state] = np.asarray (c2).reshape (*shape)
        if not conv: log.warn ('State %d LASCI not converged!', state)
        converged.append (conv)
        t = log.timer ('State {} LASCI'.format (state), *t)

    e_tot = np.dot (las.weights, e_states)
    return converged, e_tot, e_states, e_cas, e_lexc, ci1


def convert_dmao_R_to_dmao_k(kmf, kmesh, dm_R):
    '''
    Convert the density matrix from real space to k-space.
    '''

    cell = kmf.cell
    kpts = kmf.kpts
    dtype = dm_R.dtype
    
    ts = TranslationSymm(cell, kmesh)
    R_indices = ts.lattice_indices(kmesh)
    R_cart = np.array([ts.lattice_cart(R) for R in R_indices])

    nkpts = len(kpts)
    assert nkpts == np.prod(kmesh) == len(R_cart)
    assert dm_R.ndim == 2

    nao = dm_R.shape[0] // nkpts
    assert nao == cell.nao_nr()

    dm_R = dm_R.reshape(nkpts, nao, nkpts, nao)
    dm_k = np.zeros((nkpts, nao, nao), dtype=dtype)

    for ik, k in enumerate(kpts):
        for iR, Rv in enumerate(R_cart):
            phase_R = np.exp(-1j * np.dot(k, Rv))
            for iS, Sv in enumerate(R_cart):
                phase_S = np.exp(+1j * np.dot(k, Sv))
                dm_k[ik] += phase_R * dm_R[iR, :, iS, :] * phase_S
    dm_k /= nkpts # Remember the normalization.
    return dm_k

class PBCLASCINoSymm(casci.PBCCASCI, LASCINoSymm):
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
      
    # def __init__(self, kmf, ncas, nelecas, ncore=None, spin_mult=None, 
    #              kmesh=None, kpts=None):
    #     self.spin_mult = spin_mult
    #     assert kmesh is not None
    #     nkpts = np.prod(kmesh)           
    #     self.ncas_sub = nkpts * (ncas,)
    #     if isinstance(nelecas, int):
    #         self.nelecas_sub = nkpts * (nelecas,)
    #     elif isinstance(nelecas, tuple) and len(nelecas) == 2:
    #         self.nelecas_sub = nkpts * (nelecas,)
    #     self.spin_sub = spin_mult * nkpts if spin_mult is not None else None
    #     self.nroots = 1
    #     # Initialize the parent classes.
    #     _PBCCASCIForLAS.__init__(self, kmf, ncas, nelecas, ncore=ncore)
    #     LASCINoSymm.__init__(self, kmf, ncas=self.ncas_sub, nelecas=self.nelecas_sub, ncore=ncore, spin_sub=self.spin_sub)

    #     self.kmesh = kmesh
    #     self.kpts = kpts if kpts is not None else kmf.kpts
    #     nkpts = len(self.kpts)
    #     assert nkpts == np.prod(kmesh), "kmesh and kpts do not match."

    #     # Making sure this is for an unit cell, not summed over multiple unit cells.
    #     self.ncas = ncas
    #     self.nelecas = nelecas

    #     # the total active space should be stored as ncastot, nelecstot
    #     self.ncastot = sum(self.ncas_sub)
    #     if isinstance(self.nelecas_sub[0], int):
    #         self.nelecstot = sum(self.nelecas_sub)
    #     else:
    #         self.nelecstot = tuple(map(sum, zip(*self.nelecas_sub)))

    # Currently initializing the LASCINosymm is breaking a lot of things in
    # PBCCASCI module. As an example, the fcisolver class is reassigned to direct_spin1 or the
    # real csf_solver. so I decided to adopt the __init__ function of the LASCINoSymm class and 
    # then register the functions that I need from PBCCASCIForLAS to this class. 
    # This is not a clean way to do it but it is the fastest way to get it working. I will refactor this later.
    
    def __init__(self, kmf, ncas, nelecas, ncore=None, spin_mult=None, 
                 kmesh=None, kpts=None, frozen=None, frozen_ci=None, **kwargs):
        self.init_guess_ci = 'aufbau1'
        self.nroots = 1 # This line is required before initializing the parent class, because of self.converged ..
        casci.PBCCASCI.__init__(self, kmf, ncas, nelecas, ncore=ncore)
        self.chkfile = self._scf.chkfile
        if isinstance(nelecas, int):
            na = nelecas // 2
            nb = nelecas - na
            nelecas = (na, nb)
        elif isinstance(nelecas, tuple) and len(nelecas) == 2:
            nelecas = tuple(nelecas)
        
        assert kmesh is not None
        nkpts = np.prod(kmesh)
        if kpts is None:
            kpts = kmf.kpts

        nkpts_check = len(kpts)
        assert nkpts_check == nkpts, "kmesh and kpts do not match."
        self.kmesh = kmesh
        self.kpts = kpts

        if spin_mult is None: spin_mult = 1 + abs(nelecas[0]-nelecas[1])
        spin_sub = [spin_mult,] * nkpts
        self.ncas_sub = np.asanyarray(nkpts * (ncas,))
        self.nelecas_sub = np.asarray([(nelecas),] * nkpts)
        assert (len (self.nelecas_sub) == self.nfrags)
        self.frozen = frozen
        self.frozen_ci = frozen_ci
        self.conv_tol_grad = 1e-4
        self.conv_tol_self = 1e-10
        self.ah_level_shift = 1e-8
        self.max_cycle_macro = 50
        self.max_cycle_micro = 5
        self.min_cycle_macro = 0
        self.trust_radius = np.pi
        keys = set(('e_states', 'fciboxes', 'nroots', 'weights', 'ncas_sub', 'nelecas_sub',
                    'conv_tol_grad', 'conv_tol_self', 'max_cycle_macro', 'max_cycle_micro',
                    'ah_level_shift', 'states_converged', 'chkfile', 'e_lexc', 'trust_radius'))
        self._keys = set(self.__dict__.keys()).union(keys)
        self.fciboxes = []
        if isinstance(spin_sub,int):
            self.fciboxes.append(self._init_fcibox(spin_sub,self.nelecas_sub[0]))
        else:
            assert (len (spin_sub) == self.nfrags)
            for smult, nel in zip (spin_sub, self.nelecas_sub):
                self.fciboxes.append (self._init_fcibox (smult, nel))
        self.weights = [1.0]
        self.e_states = [0.0]
        self.e_lexc = [[np.array ([0]),],]
    
    def _init_fcibox(self, smult, nelec):
        '''
        Initialize a FCI box for a given spin multiplicity and electron count.
        This has to replaced from the standard LASCI implementation.
        '''
        solver = csf_solver(self.cell, smult)
        solver.spin = nelec[0] - nelec[1]
        return get_h1e_zipped_fcisolver(state_average_n_mix(self, [solver], [1.0]).fcisolver)

    # Register these functions.
    get_sym_fr = LASCINoSymm.get_sym_fr
    get_nelec_frs = LASCINoSymm.get_nelec_frs
    get_h1eff = get_h1las = h1e_for_las = h1e_for_las
    get_h2eff = ao2mo = LASCINoSymm.get_h2eff
    get_h2eff_slice = LASCINoSymm.get_h2eff_slice
    get_h1cas = h1e_for_cas = h1e_for_cas 
    get_h2cas = h2e_for_cas = h2e_for_cas 
    
    @lib.with_doc(kernel.__doc__)
    def kernel (self, mo_coeff=None, ci0=None, lroots=None, lweights=None, verbose=None,
               assert_no_dupes=False, _dry_run=False):
        if mo_coeff is None:
            mo_coeff=self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if ci0 is None: ci0 = self.ci
        if verbose is None: verbose = self.verbose
        converged, e_tot, e_states, e_cas, e_lexc, ci = kernel (
            self, mo_coeff=mo_coeff, ci0=ci0, lroots=lroots, lweights=lweights,
            verbose=verbose, assert_no_dupes=assert_no_dupes, _dry_run=_dry_run)
        if _dry_run: return
        self.converged, self.ci = converged, ci
        self.e_tot, self.e_states, self.e_cas, self.e_lexc = e_tot, e_states, e_cas, e_lexc
        if mo_coeff is self.mo_coeff:
            self.dump_chk ()
        elif getattr (self, 'chkfile', None) is not None:
            lib.logger.warn (self, 'orbitals changed; chkfile not dumped!')
        self._finalize (method='LASCI')
        return self.converged, self.e_tot, self.e_states, self.e_cas, e_lexc, self.ci

    def get_mo_slice (self, idx, mo):
        '''
        Get the molecular orbital slice for a given fragment.
        Note, unlinke molecular LASCI, the mo_coeff are only passed in 
        for the active space part only.
        '''
        for offs in self.ncas_sub[:idx]:
            mo = mo[:,offs:]
        mo = mo[:,:self.ncas_sub[idx]]
        return mo
    
    @lib.with_doc(mollasci.LASCINoSymm.make_rdm1s.__doc__)
    def make_casdm1s_sub (self, ci=None, ncas_sub=None, nelecas_sub=None,
            casdm1frs=None, w=None, **kwargs):
        if casdm1frs is None:
            casdm1frs = self.states_make_casdm1s_sub (ci=ci, ncas_sub=ncas_sub, 
                                                      nelecas_sub=nelecas_sub, **kwargs)
        if w is None: w = self.weights
        # (nfrags, 2, ncas, ncas), this would be wannier basis
        return [np.einsum ('rspq,r->spq', dm1, w) for dm1 in casdm1frs]
    
    def make_rdm1s_sub (self, mo_coeff=None, ci=None, ncas_sub=None,
            nelecas_sub=None, include_core=False, casdm1s_sub=None, **kwargs):
        
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        if casdm1s_sub is None: casdm1s_sub = self.make_casdm1s_sub (ci=ci,
            ncas_sub=ncas_sub, nelecas_sub=nelecas_sub, **kwargs)
        
        # We would need to transform the casdm1s to the AO basis, so using wannierization.
        nkpts, nao, nmo = mo_coeff.shape
        ncas = ncas_sub[0]
        ncore = self.ncore
        mo_act_kpts = [mo_coeff[i][:, ncore:ncore+ncas] for i in range(nkpts)]
        wannier_orb = get_wannier_orbs(self._scf, self.kmesh, mo_act_kpts)[0]
        wannier_orb = wannier_orb.reshape(nkpts*nao, nkpts*ncas)
        
        # These should be in wannier basis.
        rdm1s_ao_wann = []
        for idx, casdm1s in enumerate (casdm1s_sub):
            mo = self.get_mo_slice (idx, wannier_orb)
            moH = mo.conjugate ().T
            rdm1s_ao_wann.append(np.array([reduce(np.dot, (mo, casdm1s[s], moH)) 
                                           for s in range(2)]))
            
        rdm1s_ao_wann = np.array(rdm1s_ao_wann).sum (0)
        
        assert rdm1s_ao_wann.shape == (2, nkpts*nao, nkpts*nao), \
            f"Shape mismatch: {rdm1s_ao_wann.shape} != (2, {nkpts*nao}, {nkpts*nao})"
        
        rdm1s_ao_k = np.array([
            convert_dmao_R_to_dmao_k(self._scf, self.kmesh, rdm1s_ao_wann[s]) 
            for s in range(2)])
        
        assert rdm1s_ao_k.shape == (2, nkpts, nao, nao), \
            f"Shape mismatch: {rdm1s_ao_k.shape} != (2, {nkpts}, {nao}, {nao})"

        if include_core and self.ncore:
            for k in range(nkpts):
                mo_core = mo_coeff[k][:,:self.ncore]
                moH_core = mo_core.conjugate ().T
                dm_core = mo_core @ moH_core
                rdm1s_ao_k[:, k, :, :] += np.array([dm_core, dm_core])

        # Sanity Checks: compare the number of electrons from the density matrix.        
        assert rdm1s_ao_k.shape == (2, nkpts, nao, nao), \
            f"Shape mismatch: {rdm1s_ao_k.shape} != (2, {nkpts}, {nao}, {nao})"

        ovlp = self._scf.get_ovlp(kpts=self.kpts)
        rdm1 = rdm1s_ao_k.sum (0)
        nelecref = self._scf.cell.nelectron
        nelecdiff = [np.trace(ovlp[k] @ rdm1[k]).real - nelecref 
                     for k in range(nkpts)]
        if any(abs(diff) > 1e-6 for diff in nelecdiff):
            raise ValueError(f"Number of electrons mismatch for k-points: {nelecdiff}")
        return rdm1s_ao_k
    
    def make_rdm1s (self, mo_coeff=None, ncore=None, **kwargs):
        '''
        Get spin-separated, rootspace- and locally state-averaged 1-RDMs in the AO basis.
        Kwargs:
            mo_coeff: ndarray of shape (nkpts,nao,nmo)
                Contains MO coefficients at each k-point. If None, it will be taken from the PBCCASCI object.
            ncore: integer
                Number of core orbitals
        Returns:
            dm1s: ndarray of shape (2,nkpts,nao,nao)
                Rootspace- and locally state-averaged, spin-separated 1-RDMs in the AO basis for each k-point.
                Note: the casdm1s are constructed in the Wannier basis which are then transformed to block AO basis
                and then transformed to k-space. The core contribution is added in k-space.
        '''
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if ncore is None: ncore = self.ncore
        nkpts = mo_coeff.shape[0]
        dtype = np.result_type(mo_coeff[0][0].dtype, np.complex128)
        dm_core = np.array([mo_coeff[k][:,:ncore] @ mo_coeff[k][:,:ncore].conj().T 
                            for k in range(nkpts)], dtype=dtype)
        dm_cas = self.make_rdm1s_sub (mo_coeff=mo_coeff, **kwargs)
        dm1s = dm_core[None,:,:] + dm_cas
        return dm1s
    
    def states_make_casdm1s_sub (self, ci=None, ncas_sub=None, nelecas_sub=None, **kwargs):
        dtype = np.result_type(np.array(ci)[0].dtype, np.complex128)
        if ci is None: ci = self.ci
        if ncas_sub is None: ncas_sub = self.ncas_sub
        if nelecas_sub is None: nelecas_sub = self.nelecas_sub
        if ci is None:
            return [np.zeros ((self.nroots,2,ncas,ncas), dtype=dtype) for ncas in ncas_sub] 
        casdm1s = []
        for fcibox, ci_i, ncas, nelecas in zip (self.fciboxes, ci, ncas_sub, nelecas_sub):
            if ci_i is None:
                dm1a = dm1b = np.zeros ((ncas, ncas), dtype=dtype)
            else:
                dm1a, dm1b = fcibox.states_make_rdm1s (ci_i, ncas, nelecas)
            casdm1s.append (np.stack ([dm1a, dm1b], axis=1))
        return casdm1s
    