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
        dtype = np.complex128

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
