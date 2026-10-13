"""Complex charged LAS matrix elements and independent-fragment references."""

import unittest

import numpy as np

from mrh.my_pyscf.lassi.op_o0 import _ci_outer_product
from mrh.my_pyscf.pbc.fci import direct_spin1_cplx
from mrh.my_pyscf.pbc.mcscf.klassi import build_hamiltonian


def explicit_products(ci, norb, nelec):
    vectors = []
    for state in range(nelec.shape[1]):
        counts = nelec[:, state]
        gen = _ci_outer_product([row[state] for row in ci], norb, counts)[0]
        # Convert fragment-major creation order to all-alpha/all-beta order.
        parity = sum(counts[f, 1] * counts[f+1:, 0].sum() for f in range(len(norb)))
        vectors.append((-1) ** parity * next(gen()).copy())
    return vectors


class MatrixTests(unittest.TestCase):
    def test_complex_projection(self):
        rng = np.random.default_rng(51)
        norb = [2, 2]
        h1 = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        h1 = h1 + h1.conj().T
        factors = rng.normal(size=(3, 4, 4)) + 1j * rng.normal(size=(3, 4, 4))
        factors += factors.conj().transpose(0, 2, 1)
        h2 = np.einsum('xpq,xrs->pqrs', factors, factors)
        for charged in ((1, 0), (2, 1)):
            nelec = np.array([[charged, (1, 1)], [(1, 1), charged]])
            ci = []
            for frag in range(2):
                row = []
                for state in range(2):
                    shape = [2 if n == 1 else 1 for n in nelec[frag, state]]
                    c = rng.normal(size=shape) + 1j * rng.normal(size=shape)
                    row.append(c / np.linalg.norm(c))
                ci.append(row)
            ham, overlap = build_hamiltonian(h1, h2, ci, norb, nelec, ecore=1.23)
            vecs = explicit_products(ci, norb, nelec)
            ne = tuple(nelec[:, 0].sum(axis=0))
            eri = direct_spin1_cplx.absorb_h1e(h1, h2, 4, ne, .5)
            hvecs = [direct_spin1_cplx.contract_2e(eri, v, 4, ne) for v in vecs]
            expected_s = np.array([[np.vdot(b, k) for k in vecs] for b in vecs])
            expected_h = np.array([[np.vdot(b, k) for k in hvecs] for b in vecs])
            np.testing.assert_allclose(overlap, expected_s, atol=1e-12)
            np.testing.assert_allclose(ham, expected_h + 1.23 * expected_s, atol=1e-11)
            np.testing.assert_allclose(ham, ham.conj().T, atol=1e-11)
            self.assertGreater(abs(ham[0, 1].imag), 1e-3)

    def test_one_electron_hopping(self):
        h1 = np.array([[1., .3j], [-.3j, 2.]])
        nelec = np.array([[(1, 0), (0, 0)], [(0, 0), (1, 0)]])
        ci = [[np.ones((1, 1)) for _ in range(2)] for _ in range(2)]
        ham, overlap = build_hamiltonian(h1, np.zeros((2,) * 4), ci, [1, 1], nelec)
        np.testing.assert_allclose(ham, h1)
        np.testing.assert_allclose(overlap, np.eye(2))

    def test_nonorthogonal_core_shift(self):
        ci = [[np.array([[1.], [0.]]), np.array([[.6j], [.8]])]]
        nelec = np.array([[(1, 0), (1, 0)]])
        ham, overlap = build_hamiltonian(np.zeros((2, 2)), np.zeros((2,) * 4),
                                        ci, [2], nelec, ecore=2.)
        self.assertAlmostEqual(overlap[0, 1], .6j)
        np.testing.assert_allclose(ham, 2 * overlap)


if __name__ == '__main__':
    unittest.main()
