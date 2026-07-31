#!/usr/bin/env python

import unittest
import numpy as np

from pyscf.pbc import gto

from mrh.my_pyscf.pbc.util.transym import TranslationSymm
from mrh.my_pyscf.pbc.util import wannier

get_wannier_orbs = wannier.get_wannier_orbs
make_ovlp_mat_in_wannier_basis = wannier.make_ovlp_mat_in_wannier_basis
make_wannier_matrix = wannier.make_wannier_matrix
pack_wannier_orb = wannier.pack_wannier_orb
unpack_wannier_orb = wannier.unpack_wannier_orb

# Test-0: TranslationSymm should give the expected BvK cell indices and the
#         same translation operator in reciprocal and real space.
# Test-1: Bloch-to-Wannier transformation should preserve orthonormality.
# Test-2: Packed Wannier orbitals should reconstruct every translated cell.

cell = kmf = None
kmesh = [3, 1, 1]

class dummyKMF:

    def __init__(self, cell, kpts):
        self.cell = cell
        self.kpts = kpts

    def get_ovlp(self, kpts=None):
        if kpts is None:
            kpts = self.kpts
        return np.array([np.eye(self.cell.nao_nr()) for k in kpts])


def setUpModule():
    global cell, kmf
    cell = gto.Cell()
    cell.a = np.diag([3.0, 7.0, 9.0])
    cell.atom = "He 0 0 0"
    cell.basis = "sto-3g"
    cell.verbose = 0
    cell.build()

    kpts = cell.make_kpts(kmesh, wrap_around=True)
    kmf = dummyKMF(cell, kpts)


class KnownValues(unittest.TestCase):

    def test_translation_symmetry(self):
        ts = TranslationSymm(cell, kmesh, kpts=kmf.kpts)
        np.testing.assert_array_equal(
            ts.R_indices,
            np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
        )
        self.assertEqual(ts.mod_index((-1, 0, 0)), (2, 0, 0))

        norb = 2
        phase = ts.get_k_to_cell_transmat(norb=norb)
        trans_k = ts.build_translation_in_reciprocal_space(
            (1, 0, 0), norb=norb,
        )
        trans_R = ts.build_translation_in_real_space(
            (1, 0, 0), norb=norb,
        )
        np.testing.assert_allclose(
            phase @ trans_k @ phase.conj().T, trans_R, atol=1e-12,
        )

    def test_bloch_to_wannier(self):
        phases = np.exp(1j * np.array([0.2, -0.3, 0.5]))
        mo_coeff = phases[:, None, None]
        wannier_orb, R_indices, mo_phase = get_wannier_orbs(
            kmf, kmesh, mo_coeff,
        )
        wannier_mat = make_wannier_matrix(wannier_orb)
        ovlp = make_ovlp_mat_in_wannier_basis(kmf, kmesh)

        self.assertEqual(wannier_orb.shape, (3, 1, 3, 1))
        self.assertEqual(R_indices.shape, (3, 3))
        self.assertEqual(mo_phase.shape, (3, 1, 3))
        np.testing.assert_allclose(
            wannier_mat.conj().T @ ovlp @ wannier_mat,
            np.eye(3), atol=1e-12,
        )

    def test_pack_unpack(self):
        phases = np.exp(1j * np.array([0.2, -0.3, 0.5]))
        mo_coeff = phases[:, None, None]
        wannier_orb = get_wannier_orbs(kmf, kmesh, mo_coeff)[0]

        packed = pack_wannier_orb(wannier_orb, ref_cell=1)
        unpacked = unpack_wannier_orb(
            packed, cell, kmesh, ref_cell=1,
        )
        np.testing.assert_allclose(unpacked, wannier_orb, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
