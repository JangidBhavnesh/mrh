import numpy as np
from scipy import linalg
from pyscf.lib import logger
from pyscf import mcpdft, lib
from pyscf.mcpdft import _dms
from mrh.my_pyscf.lassi.lassi import las_symm_tuple, iterate_subspace_blocks
from mrh.my_pyscf.lassi import op_o1
from pyscf.mcpdft import lpdft as lpdft_fns

'''
This file is taken from pyscf-forge and adopted for the LAS wavefunctions.
'''

def weighted_average_densities(mc, ci=None, weights=None):
    """
	Compute the weighted average 1- and 2-electron LAS densities
	in the selected modal space
	"""
    if ci is None:
        ci = mc.ci
    if weights is None:
        weights = [1 / len(mc.statlis), ] * len(mc.statlis)
    casdm1s = [mc.make_one_casdm1s(ci, state=state) 
               for state in mc.statlis]
    casdm2 = [mc.make_one_casdm2(ci, state=state) 
              for state in mc.statlis]
    weights = [1 / len(mc.statlis), ] * len(mc.statlis)
    return (np.tensordot(weights, casdm1s, axes=1)), (np.tensordot(weights, casdm2, axes=1))


# Importing functions from the PySCF-forge
get_lpdft_hconst = lpdft_fns.get_lpdft_hconst
transformed_h1e_for_cas = lpdft_fns.transformed_h1e_for_cas
get_transformed_h2eff_for_cas = lpdft_fns.get_transformed_h2eff_for_cas

def make_lpdft_ham_(mc, mo_coeff=None, ci=None, ot=None):
    '''Compute the L-PDFT Hamiltonian

    Args:
        mo_coeff : ndarray of shape (nao, nmo)
            A full set of molecular orbital coefficients. Taken from self if
            not provided.

        ci : list of ndarrays of length nroots
            CI vectors should be from a converged CASSCF/CASCI calculation

        ot : an instance of on-top functional class - see otfnal.py

    Returns:
        lpdft_ham : ndarray of shape (nroots, nroots) or (nirreps, nroots, nroots)
            Linear approximation to the MC-PDFT energy expressed as a
            hamiltonian in the basis provided by the CI vectors. If
            StateAverageMix, then returns the block diagonal of the lpdft
            hamiltonian for each irrep.
    '''

    if mo_coeff is None: mo_coeff = mc.mo_coeff
    if ci is None: ci = mc.ci
    if ot is None: ot = mc.otfnal

    ot.reset(mol=mc.mol)

    spin = abs(mc.nelecas[0] - mc.nelecas[1])
    omega, _, hyb = ot._numint.rsh_and_hybrid_coeff(ot.otxc, spin=spin)
    if abs(omega) > 1e-11:
        raise NotImplementedError("range-separated on-top functionals")
    if abs(hyb[0] - hyb[1]) > 1e-11:
        raise NotImplementedError(
            "hybrid functionals with different exchange, correlations components")

    cas_hyb = hyb[0]

    casdm1s_0, casdm2_0 = mc.get_casdm12_0()
    mc.veff1, mc.veff2, E_ot = mc.get_pdft_veff(
        mo=mo_coeff,
        ci=ci,
        casdm1s=casdm1s_0,
        casdm2=casdm2_0,
        drop_mcwfn=True,
        incl_energy=True,
        ot=ot
    )
    print(mc.veff1)
    h1, h0 = mc.get_h1lpdft(E_ot, casdm1s_0, casdm2_0, hyb=1.0 - cas_hyb, mo_coeff=mo_coeff)
    print("h0:", h0)
    h2 = mc.get_h2lpdft()

    statesym, _ = las_symm_tuple(mc, verbose=0)

    # Initialize matrices
    ham_blocks = []
    s2_blocks = []
    si_blocks = []

    e_roots = []
    s2_roots = []
    rootsym = []
    si = []
    s2_mat = []
    idx_allprods = []
    
    # Loop over symmetry blocks
    qn_lbls = ['neleca', 'nelecb', 'irrep']
    for it, (las1, sym, indices, indexed) in enumerate(iterate_subspace_blocks(mc, ci, statesym)):
        idx_space, idx_prod = indices
        ci_blk, nelec_blk = indexed[:2]
        idx_allprods.extend(list(np.where(idx_prod)[0]))
        lib.logger.info(mc, 'Build + diag H matrix L-PDFT-LASSI symmetry block %d\n'
                        + '{} = {}\n'.format(qn_lbls, sym)
                        + '(%d rootspaces; %d states)', it,
                        np.count_nonzero(idx_space),
                        np.count_nonzero(idx_prod))

        ham_blk, s2_blk, ovlp_blk = op_o1.ham(mc, h1, h2, ci_blk, nelec_blk)[:3]
        diag_idx = np.diag_indices_from(ham_blk)
        # Add the diagonal elements of the L-PDFT Hamiltonian to the diagonal of the Hamiltonian matrix
        print(ham_blk[diag_idx].shape, h0.shape, ovlp_blk.shape)
        ham_blk[diag_idx] += h0 #* ovlp_blk


        if abs(cas_hyb) > 1e-14:
            # Select the original LASSI eigenstates belonging to this
            # symmetry block.
            mc_rootsym = np.asarray(mc.rootsym)
            sym_array = np.asarray(sym)

            if mc_rootsym.ndim == 1:
                state_mask = mc_rootsym == sym_array
            else:
                state_mask = np.all(mc_rootsym == sym_array, axis=1)

            state_indices = np.where(state_mask)[0]

            if len(state_indices) != len(np.where(idx_prod)[0]):
                raise RuntimeError(
                    "Cannot reconstruct the hybrid LASSI Hamiltonian: "
                    f"block {sym} contains {len(np.where(idx_prod)[0])} product states "
                    f"but {len(state_indices)} LASSI eigenstates"
                )

            # mc.si transforms LAS product states into LASSI eigenstates.
            c_mc = np.asarray(mc.si)[np.ix_(np.where(idx_prod)[0], state_indices)]
            e_mc = np.asarray(mc.e_roots)[state_indices]

            # Generalized eigen value equation:
            mc_ham_blk = (ovlp_blk @ c_mc
                @ np.diag(e_mc)
                @ c_mc.conj().T
                @ ovlp_blk
            )

            ham_blk += cas_hyb * mc_ham_blk

        try:
            e, c = linalg.eigh(ham_blk, b=ovlp_blk)
        except linalg.LinAlgError as err:
            ovlp_eval, ovlp_vec = linalg.eigh(ovlp_blk)
            lc = 'checking if L-PDFT-LASSI basis has lindeps: |ovlp| = {:.6e}'.format(np.linalg.det(ovlp_blk))
            lib.logger.info(las, 'Caught error %s, %s', str(err), lc)
            if np.linalg.det(ovlp_blk) < LINDEP_THRESH:
                x = canonical_orth_(ovlp_blk, thr=LINDEP_THRESH)
                lib.logger.info(las, '%d/%d linearly independent model states',
                                x.shape[1], x.shape[0])
                xhx = x.conj().T @ ham_blk @ x
                e, c = linalg.eigh(xhx)
                c = x @ c
            else:
                raise (err) from None

        ham_blocks.append(ham_blk)
        s2_blocks.append(s2_blk)
        si_blocks.append(c)

        # Transform the S**2 matrix into the adiabatic basis
        s2_adiabatic = c.conj().T @ s2_blk @ c

        s2_mat.append(s2_blk)
        si.append(c)
        s2_blk = c.conj().T @ s2_blk @ c
        
        lib.logger.debug2(mc, '{}'.format(s2_blk))
        e_roots.extend(list(e))
        s2_roots.extend(list(np.diag(s2_blk)))
        rootsym.extend([sym, ] * c.shape[1])

    # Aseemble the full block:
    ham_full = linalg.block_diag(*ham_blocks)
    s2_full = linalg.block_diag(*s2_blocks)
    si_full = linalg.block_diag(*si_blocks)

    # Sort the product states to match the order of the original LASSI eigenstates
    idx_allprods = np.argsort(np.array(idx_allprods))
    ham_full = ham_full[np.ix_(idx_allprods, idx_allprods)]
    s2_full = s2_full[np.ix_(idx_allprods, idx_allprods)]
    si_full = si_full[:, idx_allprods]

    # Sorting the L-PDFT eigenstate acc to energy
    e_roots = np.array(e_roots)
    s2_roots = np.array(s2_roots)
    rootsym = np.asarray(rootsym)
    
    energy_sort_idx = np.argsort(e_roots)
    e_roots = e_roots[energy_sort_idx]
    s2_roots = s2_roots[energy_sort_idx]
    rootsym = rootsym[energy_sort_idx]
    si_full = si_full[:, energy_sort_idx]
    return ham_full, e_roots, si_full, rootsym, s2_roots, s2_full


def kernel(mc, mo_coeff=None, ot=None, **kwargs):
    if ot is None: ot = mc.otfnal
    if mo_coeff is None: mo_coeff = mc.mo_coeff
    mc.optimize_mcscf_(mo_coeff=mo_coeff, **kwargs)
    mc.lpdft_ham, mc.e_states, mc.si_pdft, mc.rootsym, mc.s2_roots, s2_mat = mc.make_lpdft_ham_(ot=ot)
    logger.debug(mc, f"L-PDFT Hamiltonian in LASSI Basis:\n{mc.get_lpdft_ham()}")

    # print(mc.lpdft_ham)

    logger.debug(mc, f"L-PDFT SI:\n{mc.si_pdft}")

    mc.e_tot = mc.e_states[0]
    logger.info(mc, 'LASSI-LPDFT eigenvalues (%d total):', len(mc.e_states))

    fmt_str = '{:2s}   {:>16s}  {:2s}  '
    col_lbls = ['ix', 'Energy', '<S**2>']
    logger.info(mc, fmt_str.format(*col_lbls))
    fmt_str = '{:2d}  {:16.10f}  {:6.3f}  '
    for ix, (er, s2r) in enumerate(zip(mc.e_states, mc.s2_roots)):
        row = [ix, er, s2r]
        logger.info(mc, fmt_str.format(*row))
        if ix >= 99 and mc.verbose < lib.logger.DEBUG:
            lib.logger.info(mc, ('Remaining %d eigenvalues truncated; '
                                 'increase verbosity to print them all'), len(mc.e_states) - 100)
            break

    return mc.e_tot, mc.e_states, mc.si_pdft, mc.s2_roots


class _LPDFT(mcpdft.MultiStateMCPDFTSolver):
    '''Linerized PDFT

    Saved Results

        e_tot : float
            Weighted-average L-PDFT final energy
        e_states : ndarray of shape (nroots)
            L-PDFT final energies of the adiabatic states
        ci : list of length (nroots) of ndarrays
            CI vectors in the optimized adiabatic basis of MC-SCF. Related to
            the L-PDFT adiabat CI vectors by the expansion coefficients
            ``si_pdft''.
        si_pdft : ndarray of shape (nroots, nroots)
            Expansion coefficients of the L-PDFT adiabats in terms of the
            optimized
            MC-SCF adiabats
        e_mcscf : ndarray of shape (nroots)
            Energies of the MC-SCF adiabatic states
        lpdft_ham : ndarray of shape (nroots, nroots)
            L-PDFT Hamiltonian in the MC-SCF adiabatic basis
        veff1 : ndarray of shape (nao, nao)
            1-body effective potential in the AO basis computed using the
            zeroth-order densities.
        veff2 : pyscf.mcscf.mc_ao2mo._ERIS instance
            Relevant 2-body effective potential in the MO basis.
    '''

    def __init__(self, mc):
        self.__dict__.update(mc.__dict__)
        keys = set(('lpdft_ham', 'si_pdft', 'veff1', 'veff2'))
        self.lpdft_ham = None
        self.si_pdft = None
        self.veff1 = None
        self.veff2 = None
        self._e_states = None
        self._keys = set(self.__dict__.keys()).union(keys)

    make_lpdft_ham_ = make_lpdft_ham_
    make_lpdft_ham_.__doc__ = make_lpdft_ham_.__doc__

    get_lpdft_hconst = get_lpdft_hconst
    get_lpdft_hconst.__doc__ = get_lpdft_hconst.__doc__

    get_h1lpdft = transformed_h1e_for_cas
    get_h1lpdft.__doc__ = transformed_h1e_for_cas.__doc__

    get_h2lpdft = get_transformed_h2eff_for_cas
    get_h2lpdft.__doc__ = get_transformed_h2eff_for_cas.__doc__

    get_casdm12_0 = weighted_average_densities
    get_casdm12_0.__doc__ = weighted_average_densities.__doc__

    def get_lpdft_ham(self):
        '''The L-PDFT effective Hamiltonian matrix

            Returns:
                lpdft_ham : ndarray of shape (nroots, nroots)
                    Contains L-PDFT Hamiltonian elements on the off-diagonals
                    and PDFT approx energies on the diagonals
                '''
        return self.lpdft_ham

    def kernel(self, mo_coeff=None, ci0=None, ot=None, verbose=None):
        '''
        Returns:
            6 elements, they are
            total energy,
            the MCSCF energies,
            the active space CI energy,
            the active space FCI wave function coefficients,
            the MCSCF canonical orbital coefficients,
            the MCSCF canonical orbital energies

        They are attributes of the QLPDFT object, which can be accessed by
        .e_tot, .e_mcscf, .e_cas, .ci, .mo_coeff, .mo_energy
        '''

        if ot is None: ot = self.otfnal
        ot.reset(mol=self.mol)  # scanner mode safety

        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff

        log = logger.new_logger(self, verbose)

        kernel(self, mo_coeff, ot=ot, verbose=log)

        return (
            self.e_tot, self.e_mcscf, self.e_cas, self.ci,
            self.mo_coeff, self.mo_energy)

    def get_lpdft_hcore_only(self, casdm1s_0, hyb=1.0, mo_coeff=None, 
                             ncore=None, ncas=None):
        '''
        Returns the lpdft hcore AO integrals weighted by the
        hybridization factor. Excludes the MC-SCF (wfn) component.
        '''

        dm1s = _dms.casdm1s_to_dm1s(self, casdm1s=casdm1s_0, mo_coeff=mo_coeff, 
                                    ncore=ncore, ncas=ncas)
        dm1 = dm1s[0] + dm1s[1]
        v_j = self._scf.get_j(dm=dm1)
        return hyb * self.get_hcore() + self.veff1 + hyb * v_j

    def get_lpdft_hcore(self, casdm1s_0=None):
        '''
        Returns the full lpdft hcore AO integrals. Includes the MC-SCF
        (wfn) component for hybrid functionals.
        '''
        if casdm1s_0 is None:
            casdm1s_0 = self.get_casdm12_0()[0]

        spin = abs(self.nelecas[0] - self.nelecas[1])
        cas_hyb = self.otfnal._numint.rsh_and_hybrid_coeff(self.otfnal.otxc, spin=spin)[2]
        hyb = 1.0 - cas_hyb[0]

        return cas_hyb[0] * self.get_hcore() + self.get_lpdft_hcore_only(casdm1s_0, hyb=hyb)


def linear_multi_state(mc, **kwargs):
    ''' Build linearized multi-state MC-PDFT method object

    Args:
        mc : instance of class _PDFT

    Kwargs:
        weights : sequence of floats

    Returns:
        si : instance of class _LPDFT
    '''

    base_name = mc.__class__.__name__
    mcbase_class = mc.__class__

    class LPDFT(_LPDFT, mcbase_class):
        pass

    LPDFT.__name__ = "LIN" + base_name
    return LPDFT(mc)
