import numpy as np
import scipy

# Author: Bhavnesh Jangid

class TranslationSymm:
    '''
    Couple of helper functions to deal with translation symmetry operations.
    '''
    def __init__(self, cell, kmesh):
        self.cell = cell
        self.kmesh = kmesh
    
    def lattice_indices(self, kmesh):
        '''
        For a given kmesh, return the BvK (Born–von Karman) supercell 
        lattice indices:
        args:
            kmesh: tuple of ints, (n1, n2, n3)
                Basically, the number of k-points along each reciprocal lattice vector.
        returns:
            R_index: np.ndarray of shape (n1*n2*n3, 3)
                Each row is a triplet of integers (i, j, k) corresponding 
                to the lattice indices of the supercell.
                Basically, [[i, j, k] for i in range(n1) 
                                        for j in range(n2) 
                                            for k in range(n3)]
        '''
        assert len(kmesh) == 3, "kmesh should be a tuple of 3 integers."
        R_index = np.array(list(np.ndindex(tuple(kmesh))), dtype=int)
        return R_index

    def lattice_cart(self, R_index):
        '''
        Convert integer lattice index to cartesian translation vector.
        args:
            cell: pyscf.pbc.Cell object
                The unit cell of the system.
            R_index: array-like of shape (3,)
                The integer lattice index (i, j, k).
        returns:
            R_cart: np.ndarray of shape (3,)
                The cartesian translation vector corresponding to the lattice index.
        '''
        a = self.cell.lattice_vectors()
        R_index = np.asarray(R_index, dtype=int)
        return np.dot(R_index, a)

    def mod_index(self, R_index, kmesh):
        '''
        Periodic modulo index for the finite BvK supercell.
        Basically, we want to wrap the lattice indices back into 
        the range defined by kmesh.
        :: Ri = mod(Ri, ni) for i in {1, 2, 3}
        args:
            R_index: array-like of shape (3,)
                The integer lattice index (i, j, k) that may be outside the BvK supercell.
            kmesh: tuple of ints, (n1, n2, n3)
                The size of the BvK supercell in terms of lattice indices.
        returns:
            R_mod: np.ndarray of shape (3,)
                The lattice index wrapped back into the range
        >>> mod_index((-1,0,0), (10,1,1))
        >>> (9, 0, 0)
        '''
        return tuple(np.mod(np.asarray(R_index), np.asarray(kmesh)))

    def index_map(self, R_indices):
        '''
        In this function, we create a mapping from the lattice indices to a flat cell index.
        This is useful for constructing matrices where we need to index 
        cells in a linear fashion.
        args:
            R_indices: np.ndarray of shape (n_cells, 3)
                Array of lattice indices for each cell in the BvK supercell.
        returns:
            R_to_i: dict
                Dictionary mapping each lattice index (as a tuple) to a unique integer index.
        '''
        return {tuple(R): i for i, R in enumerate(R_indices)}

    def build_phase_matrix(self, kpts, R_indices):
        '''
        Build the phase matrix
        F[R, k] = exp(-i k.R) / sqrt(Nk)

        This will help us to map k-space objects to real-cell objects.
            |R> = \sum_k F[R,k] |k>
        args:
            kpts: np.ndarray of shape (Nk, 3)
                Array of k-points in reciprocal space.
            R_indices: np.ndarray of shape (n_cells, 3)
                Array of lattice indices for each cell in the BvK supercell.
        returns:
            phase: np.ndarray of shape (n_cells, Nk)
                The phase matrix that transforms k-space objects to real-space cell objects.
        '''
        nk = len(kpts)
        assert len(R_indices) == nk, "Number of R indices should match number of k-points."
        dtype = np.complex128
        phase = np.array([[np.exp(-1j * np.dot(k, self.lattice_cart(Rind))) / np.sqrt(nk) 
                           for k in kpts] 
                           for Rind in R_indices], dtype=dtype)
        return phase

    def build_translation_in_real_space(self, T_index):
        '''
        The translation in the real-space block basis 
        is a permutation matrix that permutes the cell indices according 
        to the translation vector T_index.
        Note: this function expects this basis ordering:
            \ket{R, p}, where R is cell index and p is AO/orbital index.
        Therefore, the translation will permute the R indices according to:
            T |R, p> = |R + T, p>
        args:
            T_index: array-like of shape (3,)
                The integer lattice index corresponding to the translation vector T.
        returns:
            perm_mat: np.ndarray of shape (N_total, N_total)
                The permutation matrix representing the 
                translation operator in the real-space block basis.
                Here:
                N_total = n_cells * n_orbitals_per_cell
        '''
        kmesh = self.kmesh
        cell = self.cell
        dtype = np.complex128

        norb_per_cell = cell.nao_nr()
        R_indices = self.lattice_indices(kmesh)
        R_to_i = self.index_map(R_indices)
        ncells = len(R_indices)
        
        assert ncells == np.prod(kmesh), \
            "Number of R indices should match the product of kmesh dimensions."
        
        nao_tot = ncells * norb_per_cell

        perm_mat = np.zeros((nao_tot, nao_tot), dtype=dtype)

        for iR, R in enumerate(R_indices):
            RpT = self.mod_index(R + np.asarray(T_index), kmesh)
            iRpT = R_to_i[RpT]
            for aoorb in range(norb_per_cell):
                row = iRpT * norb_per_cell + aoorb
                col = iR * norb_per_cell + aoorb
                perm_mat[row, col] = 1.0

        return perm_mat

    def build_translation_in_reciprocal_space(self, kpts, T_index):
        '''
        Translation operator in k-space block basis. In this basis, 
        the translation operator is diagonal with phase factors.
        For example, if we have a translation T, then in k-space we have:
            T |k, p> = exp(+i k.T) |k, p>

        args:
            kpts: np.ndarray of shape (Nk, 3)
                Array of k-points in reciprocal space.
            T_index: array-like of shape (3,)
                The integer lattice index corresponding to the translation vector T.
        returns:
            D: np.ndarray of shape (N_total, N_total)
                The diagonal matrix representing the translation operator in k-space 
                block basis.
        '''
        kmesh = self.kmesh
        nk = np.prod(kmesh)
        assert len(kpts) == nk, "Number of k-points should match the product of kmesh dimensions."
        cell = self.cell

        norb_per_k = cell.nao_nr()
        trans_cart = self.lattice_cart(T_index)
        perm_mat = [np.exp(1j * np.dot(k, trans_cart)) * np.eye(norb_per_k) 
                            for k in kpts]
        perm_mat = scipy.linalg.block_diag(*perm_mat)
        return perm_mat

    def get_k_to_wannier_transmat(self, kpts):
        '''
        This function constructs the transformation matrix that maps 
        k-space block basis to real-space block basis.
        In other words, it constructs the matrix F such that:
            |R> = \sum_k F[R,k] |k>
        where |R> is the real-space cell basis and |k> is the k-space block basis.
        '''
        kmesh = self.kmesh
        assert len(kpts) == np.prod(kmesh), "Number of k-points should match the product of kmesh dimensions."
        cell = self.cell
        dtype = np.complex128

        norb = cell.nao_nr()
        R_indices = self.lattice_indices(kmesh)
        phase = self.build_phase_matrix(kpts, R_indices)
        ncell, nk = phase.shape
        transmat = np.zeros((ncell * norb, nk * norb), dtype=dtype)

        for iR in range(ncell):
            for ik in range(nk):
                row = slice(iR * norb, (iR + 1) * norb)
                col = slice(ik * norb, (ik + 1) * norb)
                transmat[row, col] = phase[iR, ik] * np.eye(norb)

        return transmat





def orthogonality_check(mo_coeff, ovlp, tol=1e-8):
    '''
    Orthoganlity check for given set of the mo_coeff.
    '''
    assert np.asarray(mo_coeff).shape == np.asarray(ovlp).shape
    if np.asarray(mo_coeff).ndim == 3:
        for k, (mo_k, ovlp_k) in enumerate(zip(mo_coeff, ovlp)):
            s = mo_k.conj().T @ ovlp_k @ mo_k
            assert np.allclose(s, np.eye(s.shape[0]), atol=tol), \
                f'k-point {k}: max|S - I| = {np.max(np.abs(s - np.eye(s.shape[0])))}'
    else:
        s = mo_coeff.conj().T @ ovlp @ mo_coeff
        assert np.allclose(s, np.eye(s.shape[0]), atol=tol), \
            f'max|S - I| = {np.max(np.abs(s - np.eye(s.shape[0])))}'
        
def lowdin_sym(s, tol=1e-15):
    '''
    Hermitian symmetrization:
    '''
    e, v = scipy.linalg.eigh(s)
    idx = e > tol
    return np.dot(v[:,idx]/np.sqrt(e[idx]), v[:,idx].conj().T)

def meta_lowdin_orbitals(ovlp):
    '''
    Get the meta-lowdin orthogonalized orbitals.
    '''
    from pyscf import lo
    lo_coeff = np.array([lo.orth.orth_ao(cell, 'meta-lowdin', s=ovlp[k]) 
                        for k in range(len(kpts))])
    # Check the orthogonality of the lowdin orbitals.
    orthogonality_check(lo_coeff, ovlp)
    return lo_coeff

def localize_kmf_mo_coeff(kmf, mo0):
    '''
    Rotates the kmf.mo_coeff occupied and virtual orbitals
    in local basis.
    args:
        kmf: mean-field
            instance of the pbc.scf
        mo0: np.ndarray or list (nkpts, nao, nmo)
            kmf mo_coeff or the AVAS mo_coeff
    return:
        lo_coeff: localized mo_coeff.
    '''

    ovlp = kmf.get_ovlp(kpts=kpts)
    lo_coeff = meta_lowdin_orbitals(ovlp)

    mo_coeff = np.array([c.copy() for c in mo0])
    mo_coeff_loc = []
    umat = []

    def _project(mo_coeff_k, lo_coeff_k, ovlp_k):
        norb = mo_coeff_k.shape[1]
        assert norb <= lo_coeff_k.shape[1], f"Less AOs then MOs"

        pmat = ((mo_coeff_k.conj().T @ ovlp_k) @ lo_coeff_k)[:norb, :norb]
        pinv = pmat.conj().T @ pmat
        pinv = lowdin_sym(pinv)
        umat_k = pmat @ pinv

        # Localize the block orbitals
        mo_k = mo_coeff_k @ umat_k

        return mo_k, umat_k
        
    for k in range(len(kpts)):
        mo_occ = kmf.mo_occ[k] > 0
        mo_vir = ~mo_occ
        
        # Project the occupied space
        lo_coeff_k = lo_coeff[k]
        ovlp_k = ovlp[k]
        mo_coeff_k = mo_coeff[k][:, mo_occ]
        mo_k_occ, umat_k_occ = _project(mo_coeff_k, lo_coeff_k, ovlp_k)
        
        # project the virtual space
        lo_coeff_k = lo_coeff[k]
        ovlp_k = ovlp[k]
        mo_coeff_k = mo_coeff[k][:, mo_vir]
        mo_k_vir, umat_k_vir = _project(mo_coeff_k, lo_coeff_k, ovlp_k)
        mo_coeff_loc.append(np.hstack((mo_k_occ, mo_k_vir)))
        umat.append(scipy.linalg.block_diag(umat_k_occ, umat_k_vir))

    orthogonality_check(mo_coeff_loc, ovlp)
    print('Orthogonality check passed!')
    return mo_coeff_loc, umat

def get_wannier_orbs(kmf, kmesh, mo_loc_k):
    '''
    Building the Wannier orbitals:
        W[R, mu, S, n]
            = 1/Nk sum_k exp(+i k.(R-S)) [C(k) U(k)]_{mu,n}
            = 1/Nk sum_k exp(+i k.(R-S)) [C_loc(k)]_{mu,n}
    args:
        ts: TranslationSymm object
            Object containing the translation symmetry information of the system.
        kmf: pbc.scf object
            mean-field object containing the mo_coeff and other information.
        kpts: np.ndarray of shape (Nk, 3)
            Array of k-points in reciprocal space.
        mo_loc_k: list of np.ndarray (nkpts, nao, norb)
            Localized mo_coeff for each k-point.
    returns:
        wannier_orb : ndarray W[R, mu, S, n] (ncell, nao, ncell, nwann)
            Wannier orbitals in real space.
        R_indices : ndarray
            BvK cell indices.
    '''
    cell = kmf.cell
    kpts = kmf.kpts
    dtype = mo_loc_k[0].dtype

    ts = TranslationSymm(cell, kmesh)
    nkpts = len(kpts)
    R_indices = ts.lattice_indices(kmesh)
    ncell = len(R_indices)

    assert np.prod(kmesh) == len(kpts), "kmesh and number of kpts in kmf do not match"
    assert ncell == nkpts

    nao = mo_loc_k[0].shape[0]
    nwann = mo_loc_k[0].shape[1]

    wannier_orb = np.zeros((ncell, nao, ncell, nwann), dtype=dtype)

    R_cart = np.array([ts.lattice_cart(R) for R in R_indices])

    for ik, k in enumerate(kpts):
        Ck = mo_loc_k[ik]
        for iR, Rv in enumerate(R_cart):
            for iS, Sv in enumerate(R_cart):
                phase = np.exp(1j * np.dot(k, Rv - Sv))
                wannier_orb[iR, :, iS, :] += phase * Ck

    wannier_orb /= nkpts

    return wannier_orb, R_indices

def check_wannier_translation(ts, W, R_indices, T_index=(1, 0, 0), tol=1e-8):
    """
    Check W[R+T, mu, S+T, n] = W[R, mu, S, n].
    """

    R_to_i = ts.index_map(R_indices)
    T_index = np.asarray(T_index, dtype=int)

    ncell, nao, ncell2, nwann = W.shape
    assert ncell == ncell2

    max_abs = 0.0
    max_ref = 0.0
    worst = None

    for iR, R in enumerate(R_indices):
        RpT = ts.mod_index(R + T_index, kmesh)
        iRpT = R_to_i[RpT]

        for iS, S in enumerate(R_indices):
            SpT = ts.mod_index(S + T_index, kmesh)
            iSpT = R_to_i[SpT]

            ref = W[iR, :, iS, :]
            shifted = W[iRpT, :, iSpT, :]

            diff = shifted - ref

            local_abs = np.max(np.abs(diff))
            local_ref = np.max(np.abs(ref))

            if local_abs > max_abs:
                max_abs = local_abs
                worst = {
                    "R": tuple(R),
                    "S": tuple(S),
                    "R_plus_T": RpT,
                    "S_plus_T": SpT,
                }

            max_ref = max(max_ref, local_ref)

    rel_err = max_abs / max(max_ref, 1e-14)

    print("Wannier translation covariance check")
    print("------------------------------------")
    print(f"T_index     = {tuple(T_index)}")
    print(f"max abs err = {max_abs:.3e}")
    print(f"max rel err = {rel_err:.3e}")
    print(f"worst block = {worst}")

    if rel_err < tol:
        print("Wannier translation covariance OK.")
    else:
        print("Wannier translation covariance FAILED.")

    return max_abs, rel_err, worst

def check_wannier_against_ref_cell(ts, W, R_indices, ref_cell=0, tol=1e-8):
    """
    Check whether every center S is a translated version of ref_cell:

        W[R, mu, S, n] = W[R - (S - S_ref), mu, S_ref, n]
    """

    R_to_i = ts.index_map(R_indices)

    ncell, nao, ncell2, nwann = W.shape
    assert ncell == ncell2

    S_ref = R_indices[ref_cell]

    max_abs = 0.0
    max_ref = 0.0
    worst = None

    for iS, S in enumerate(R_indices):
        T = S - S_ref

        for iR, R in enumerate(R_indices):
            R_ref = ts.mod_index(R - T)
            iR_ref = R_to_i[R_ref]

            ref = W[iR_ref, :, ref_cell, :]
            target = W[iR, :, iS, :]

            diff = target - ref

            local_abs = np.max(np.abs(diff))
            local_ref = np.max(np.abs(ref))

            if local_abs > max_abs:
                max_abs = local_abs
                worst = {
                    "target_R": tuple(R),
                    "target_S": tuple(S),
                    "ref_R": R_ref,
                    "ref_S": tuple(S_ref),
                    "T": tuple(T),
                }

            max_ref = max(max_ref, local_ref)

    rel_err = max_abs / max(max_ref, 1e-14)

    print("Wannier reference-cell translation check")
    print("----------------------------------------")
    print(f"ref_cell    = {ref_cell}, S_ref = {tuple(S_ref)}")
    print(f"max abs err = {max_abs:.3e}")
    print(f"max rel err = {rel_err:.3e}")
    print(f"worst block = {worst}")

    if rel_err < tol:
        print("All Wannier centers are translated copies of the reference center.")
    else:
        print("Reference-cell translation check FAILED.")

    return max_abs, rel_err, worst

def make_wannier_matrix(wannier_orb):
    '''
    Converting W[R, mu, S, n] to matrix form, such that 
    it can be used as the mo_coeff.
    args:
    wannier_orb: ndarray W[R, mu, S, n] (ncell, nao, ncell, nwann)
        Wannier orbitals in real space.
    returns:
        wannier_mat: np.ndarray of shape (ncell * nao, ncell * nwann)
        Wannier orbitals in matrix form.
    '''
    ncell, nao, ncell2, nwann = wannier_orb.shape
    assert ncell == ncell2

    # dtype = wannier_orb.dtype
    # wannier_mat = np.zeros((ncell * nao, ncell * nwann), dtype=dtype)
    # for cell1 in range(ncell):
    #     for cell2 in range(ncell):
    #         row = slice(cell1 * nao, (cell1 + 1) * nao)
    #         col = slice(cell2 * nwann, (cell2 + 1) * nwann)
    #         wannier_mat[row, col] = wannier_orb[cell1, :, cell2, :]

    wannier_mat = wannier_orb.reshape(ncell*nao, ncell * nwann)
    return wannier_mat

def make_ovlp_mat_in_wannier_basis(kmf, kmesh):
    '''
    Construct the overlap matrix in the Wannier basis.
    args:
        kmf: pbc.scf object
        kmesh: tuple of integers
    returns:
        ovlp_bvk: ndarray of shape (ncell*norb, ncell*norb)
            Overlap matrix in the Wannier basis.
    '''
    ts = TranslationSymm(kmf.cell, kmesh)
    kpts = kmf.kpts
    ovlp_k = kmf.get_ovlp(kpts=kpts)
    nkpts = np.prod(ts.kmesh)
    assert len(ovlp_k) == nkpts == len(kpts), "Number of k-points in ovlp_k "
    "should match the product of kmesh dimensions."
    norb_tot = kmf.cell.nao_nr() * nkpts

    ovlp_k = scipy.linalg.block_diag(*ovlp_k)
    phase = ts.get_k_to_wannier_transmat(kpts)
    ovlp_wannier = phase.conj().T @ ovlp_k @ phase
    
    assert ovlp_wannier.shape[0] == ovlp_wannier.shape[1] == norb_tot, \
        "shape mismatch, something went wrong in constructing the Wannier overlap matrix."
    return ovlp_wannier

def pack_wannier_orb(wannier_orb, ref_cell=0):
    '''
    Convert W[R, mu, S, n] to W[R, mu, n], by taking S=ref_cell block.
    This is useful for printing the Wannier orbitals in Molden format.
    '''
    ncell, nao, ncell2, nwann = wannier_orb.shape
    assert ncell == ncell2
    assert 0 <= ref_cell < ncell
    wannier_orb = wannier_orb[:, :, ref_cell, :]
    wannier_orb = wannier_orb.reshape(ncell*nao, nwann)
    return wannier_orb

def unpack_wannier_orb(wannier_orb_packed, cell, kmesh, ref_cell=0, make_wannier_mat=False):
    '''
    Convert W[R, mu, n] to W[R, mu, S, n], by copying the packed block to all S blocks.
    This is the inverse operation of pack_wannier_orb.
    '''
    ts = TranslationSymm(cell, kmesh)
    R_indices = ts.lattice_indices(kmesh)
    R_to_i = ts.index_map(R_indices)
    ncell = len(R_indices)
    dtype = wannier_orb_packed.dtype

    nao = cell.nao_nr()
    assert wannier_orb_packed.ndim == 2
    naoncell, nwann = wannier_orb_packed.shape
    ncell2 = naoncell // nao
    assert ncell == ncell2, "ncell does not match the shape of the packed Wannier orbitals."
    assert 0 <= ref_cell < ncell

    wannier_ref_cell = wannier_orb_packed.reshape(ncell, nao, nwann)
    
    ref_R = R_indices[ref_cell]

    wannier_orb = np.zeros((ncell, nao, ncell, nwann),dtype=dtype)

    for iS, S in enumerate(R_indices):
        # Translation from ref center to target center:
        # T = S - ref_R
        T = S - ref_R
        for iR, R in enumerate(R_indices):
            # W[R, mu, S, n] = W[R - T, mu, ref_cell, n]
            #
            # since T = S - ref_R,
            # R_ref = R - S + ref_R
            R_ref = ts.mod_index(R - T, kmesh)
            iR_ref = R_to_i[R_ref]
            wannier_orb[iR, :, iS, :] = wannier_ref_cell[iR_ref, :, :]
    if make_wannier_mat:
        wannier_orb = wannier_orb.reshape(ncell*nao, ncell*nwann)
    return wannier_orb


if __name__ == "__main__":
    from pyscf.pbc import gto as pgto

    cell = pgto.Cell()
    cell.atom = '''
    H 0.0 0.0 0.0
    H 0.74 0.0 0.0
'''
    cell.a = np.diag([4.0, 4.0, 4.0])
    cell.basis = 'STO-6G'
    cell.unit = 'Angstrom'
    cell.max_memory = 120000
    cell.ke_cutoff = 100
    cell.precision = 1e-10
    cell.verbose = 3
    cell.build()

    print("Checking translation symmetry in k-space and real-space representations...")    
    kmesh = [10, 1, 1]
    T_index = (1, 0, 0)

    kpts = cell.make_kpts(kmesh, wrap_around=True)

    ts = TranslationSymm(cell, kmesh)

    nao = cell.nao_nr()

    umat = ts.get_k_to_wannier_transmat(kpts,)
    trans_k = ts.build_translation_in_reciprocal_space(kpts, T_index)
    trans_r = ts.build_translation_in_real_space(T_index)

    trans_r_from_k = umat @ trans_k @ umat.conj().T

    err = np.max(np.abs(trans_r_from_k - trans_r))
    rel = err / max(np.max(np.abs(trans_r)), 1e-14)
    print(f"max abs err = {err:.3e}")
    print(f"max rel err = {rel:.3e}")
