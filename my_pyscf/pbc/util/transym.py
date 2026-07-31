import numpy as np
import scipy

# Author: Bhavnesh Jangid

'''
Translation symmetry operations for periodic systems.
Writing a general translation symmetry class that can be used to construct
translation operators in both k-space and real-space representations.
Atleast would be useful for kLAS.
'''

class TranslationSymm:
    '''
    Couple of helper functions to deal with translation symmetry operations.
    '''
    def __init__(self, cell, kmesh, kpts=None):
        self.cell = cell
        self.kmesh = self._sanity_check_kmesh(kmesh)
        if kpts is not None:
            kpts = self._sanity_check_kpts(kpts, self.kmesh)
        self.kpts = kpts
        # Setting R-based quantities for once and use the same ordering everywhere.
        self.R_indices = self.lattice_indices()
        self.R_cart = np.array([self.lattice_cart(R) for R in self.R_indices])
        self.R_to_i = self.index_map()
        self.ncell = len(self.R_indices)

    @staticmethod
    def _sanity_check_kmesh(kmesh):
        kmesh = np.asarray(kmesh)
        assert kmesh.shape == (3,), \
            "kmesh should be a tuple of 3 integers."
        assert np.issubdtype(kmesh.dtype, np.integer), \
            "kmesh should contain only integers."
        assert np.all(kmesh > 0), \
            "All entries of kmesh should be positive."
        return tuple(kmesh.tolist())

    @staticmethod
    def _sanity_check_kpts(kpts, kmesh):
        if kpts is not None:
            kpts = np.asarray(kpts)
            assert kpts.shape == (np.prod(kmesh), 3), \
                "Number of k-points should match the product of kmesh dimensions."
            return kpts
        return kpts
    
    def lattice_indices(self):
        '''
        For the given kmesh, return the BvK (Born–von Karman) supercell
        lattice indices:
        returns:
            R_indices: np.ndarray of shape (ncell, 3)
                Each row is a triplet of integers (i, j, k) corresponding
                to the lattice indices of the supercell.
                Basically, [[i, j, k] for i in range(n1)
                                        for j in range(n2)
                                            for k in range(n3)]
        '''
        R_indices = np.array(list(np.ndindex(self.kmesh)), dtype=int)
        return R_indices

    def lattice_cart(self, R_index):
        '''
        Convert integer lattice index to cartesian translation vector.
        args:
            R_index: array-like of shape (3,)
                The integer lattice index (i, j, k).
        returns:
            R_cart: np.ndarray of shape (3,)
                The cartesian translation vector corresponding to the lattice index.
        '''
        a = self.cell.lattice_vectors()
        R_index = np.asarray(R_index, dtype=int)
        return np.dot(R_index, a)

    def mod_index(self, R_index):
        '''
        Periodic modulo index for the finite BvK supercell.
        Basically, we want to wrap the lattice indices back into
        the range defined by kmesh.
        :: Ri = mod(Ri, ni) for i in {1, 2, 3}
        args:
            R_index: array-like of shape (3,)
                The integer lattice index (i, j, k) that may be outside the BvK supercell.
        returns:
            R_mod: tuple of shape (3,)
                The lattice index wrapped back into the range.
        >>> ts.mod_index((-1,0,0))
        >>> (n1-1, 0, 0)
        '''
        return tuple(np.mod(np.asarray(R_index), np.asarray(self.kmesh)))

    def index_map(self, R_indices=None):
        '''
        In this function, we create a mapping from the lattice indices to a flat cell index.
        This is useful for constructing matrices where we need to index
        cells in a linear fashion.
        args:
            R_indices: np.ndarray of shape (ncell, 3), optional
                Array of lattice indices for each cell in the BvK supercell.
                The stored lattice indices are used by default.
        returns:
            R_to_i: dict
                Dictionary mapping each lattice index (as a tuple) to a unique integer index.
        '''
        if R_indices is None:
            R_indices = self.R_indices
        return {tuple(R): i for i, R in enumerate(R_indices)}

    def get_kpts(self, kpts=None):
        '''
        Get and check the k-points used for the translation operation.
        args:
            kpts: np.ndarray of shape (Nk, 3), optional
                Array of k-points in reciprocal space. The stored k-points
                are used by default.
        returns:
            kpts: np.ndarray of shape (Nk, 3)
        '''
        if kpts is None:
            kpts = self.kpts
        assert kpts is not None, \
            "kpts should be passed to TranslationSymm or to this function."
        kpts = np.asarray(kpts)
        kpts = self._sanity_check_kpts(kpts, self.kmesh)
        assert kpts.shape == (self.ncell, 3), \
            "Number of k-points should match the product of kmesh dimensions."
        return kpts

    def build_phase_matrix(self, kpts=None, R_indices=None):
        '''
        Build the phase matrix
        F[R, k] = exp(-i k.R) / sqrt(Nk)

        This will help us to map k-space objects to real-cell objects.
            |R> = \sum_k F[R,k] |k>
        args:
            kpts: np.ndarray of shape (Nk, 3), optional
                Array of k-points in reciprocal space.
            R_indices: np.ndarray of shape (ncell, 3), optional
                Array of lattice indices for each cell in the BvK supercell.
        returns:
            phase: np.ndarray of shape (ncell, Nk)
                The phase matrix that transforms k-space objects to real-space cell objects.
        '''
        kpts = self.get_kpts(kpts)
        if R_indices is None:
            R_indices = self.R_indices
            R_cart = self.R_cart
        else:
            R_indices = np.asarray(R_indices)
            assert R_indices.shape == (self.ncell, 3), \
                "Number of R indices should match number of k-points."
            R_cart = np.array([self.lattice_cart(R) for R in R_indices])

        nkpts = len(kpts)
        dtype = np.complex128
        phase = np.array([[np.exp(-1j * np.dot(k, R)) / np.sqrt(nkpts)
                           for k in kpts]
                          for R in R_cart], dtype=dtype)
        return phase

    def build_translation_in_real_space(self, T_index, norb=None):
        '''
        The translation in the real-space block basis
        is a permutation matrix that permutes the cell indices according
        to the translation vector T_index.
        Note: this function expects this basis ordering:
            \ket{R, p}, where R is cell index and p is orbital index.
        Therefore, the translation will permute the R indices according to:
            T |R, p> = |R + T, p>
        args:
            T_index: array-like of shape (3,)
                The integer lattice index corresponding to the translation vector T.
            norb: int, optional
                Number of orbitals in each cell. The number of AOs in the
                unit cell is used by default.
        returns:
            perm_mat: np.ndarray of shape (ncell*norb, ncell*norb)
                The permutation matrix representing the
                translation operator in the real-space block basis.
        '''
        if norb is None:
            norb = self.cell.nao_nr()
        assert isinstance(norb, (int, np.integer)) and norb > 0, \
            "norb should be a positive integer."

        dtype = np.complex128
        perm_mat = np.zeros((self.ncell*norb, self.ncell*norb), dtype=dtype)

        for iR, R in enumerate(self.R_indices):
            RpT = self.mod_index(R + np.asarray(T_index))
            iRpT = self.R_to_i[RpT]
            for iorb in range(norb):
                row = iRpT * norb + iorb
                col = iR * norb + iorb
                perm_mat[row, col] = 1.0

        return perm_mat

    def build_translation_in_reciprocal_space(self, T_index, kpts=None, norb=None):
        '''
        Translation operator in k-space block basis. In this basis,
        the translation operator is diagonal with phase factors.
        For example, if we have a translation T, then in k-space we have:
            T |k, p> = exp(+i k.T) |k, p>

        args:
            T_index: array-like of shape (3,)
                The integer lattice index corresponding to the translation vector T.
            kpts: np.ndarray of shape (Nk, 3), optional
                Array of k-points in reciprocal space.
            norb: int, optional
                Number of orbitals at each k-point. The number of AOs in the
                unit cell is used by default.
        returns:
            trans_mat: np.ndarray of shape (Nk*norb, Nk*norb)
                The diagonal matrix representing the translation operator in k-space
                block basis.
        '''
        kpts = self.get_kpts(kpts)
        if norb is None:
            norb = self.cell.nao_nr()

        # Sanity check:
        assert isinstance(norb, (int, np.integer)) and norb > 0, \
            "norb should be a positive integer."

        trans_cart = self.lattice_cart(T_index)
        trans_mat = [np.exp(1j * np.dot(k, trans_cart)) * np.eye(norb)
                     for k in kpts]
        trans_mat = scipy.linalg.block_diag(*trans_mat)
        return trans_mat

    def get_k_to_cell_transmat(self, kpts=None, norb=None):
        '''
        This function constructs the transformation matrix that maps
        k-space block basis to real-space cell block basis.
        In other words, it constructs the matrix F such that:
            |R> = \sum_k F[R,k] |k>
        where |R> is the real-space cell basis and |k> is the k-space block basis.
        args:
            kpts: np.ndarray of shape (Nk, 3), optional
                Array of k-points in reciprocal space.
            norb: int, optional
                Number of orbitals in each k-point/cell block. The number of
                AOs in the unit cell is used by default.
        returns:
            trans_mat: np.ndarray of shape (ncell*norb, Nk*norb)
        '''
        kpts = self.get_kpts(kpts)
        if norb is None:
            norb = self.cell.nao_nr()
        assert isinstance(norb, (int, np.integer)) and norb > 0, \
            "norb should be a positive integer."

        phase = self.build_phase_matrix(kpts)
        trans_mat = np.kron(phase, np.eye(norb))
        return trans_mat


if __name__ == "__main__":
    from pyscf.pbc import gto as pgto, scf

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
    kmesh = [5, 1, 1]
    T_index = (1, 0, 0)

    kpts = cell.make_kpts(kmesh, wrap_around=True)

    kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
    kmf.max_cycle=1000
    kmf.exxdiv = None
    kmf.conv_tol = 1e-10
    kmf.kernel()

    ts = TranslationSymm(cell, kmesh, kpts=kpts)

    nao = cell.nao_nr()

    umat = ts.get_k_to_cell_transmat()
    trans_k = ts.build_translation_in_reciprocal_space(T_index)
    trans_r = ts.build_translation_in_real_space(T_index)

    trans_r_from_k = umat @ trans_k @ umat.conj().T

    err = np.max(np.abs(trans_r_from_k - trans_r))
    rel = err / max(np.max(np.abs(trans_r)), 1e-14)
    print(f"max abs err = {err:.3e}")
    print(f"max rel err = {rel:.3e}")
