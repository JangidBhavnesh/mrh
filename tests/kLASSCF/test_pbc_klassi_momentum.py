"""Momentum labels, nonorthogonal metrics, and block spectra for kLASSI."""

import unittest

import numpy as np
from scipy import linalg

from mrh.my_pyscf.pbc.mcscf.klassi_trans import momentum_basis


def lattice_generators(mesh):
    cells = np.array(list(np.ndindex(*mesh)))
    addresses = {tuple(r): i for i, r in enumerate(cells)}
    generators = np.zeros((3, len(cells), len(cells)), dtype=complex)
    for axis in range(3):
        for source, r in enumerate(cells):
            target = addresses[tuple((r + np.eye(3, dtype=int)[axis]) % mesh)]
            generators[axis, target, source] = 1
    return generators, cells / np.asarray(mesh)


class MomentumBasisTests(unittest.TestCase):
    def test_bloch_sign_with_complex_hopping(self):
        generators, scaled = lattice_generators((3, 1, 1))
        # Reorder and wrap the k mesh: output must follow the actual k points.
        scaled = scaled[[2, 0, 1]]
        scaled[0, 0] -= 1
        bases = momentum_basis(generators, np.eye(3), (3, 1, 1), scaled)
        hopping = .3 + .7j
        ham = hopping * generators[0] + hopping.conjugate() * generators[0].conj().T
        for k, basis in zip(scaled, bases):
            self.assertEqual(basis.shape, (3, 1))
            # A one-electron Bloch orbital has coefficients exp(+i*k*R).
            bloch = np.exp(2j * np.pi * k[0] * np.arange(3)) / np.sqrt(3)
            np.testing.assert_allclose(basis @ basis.conj().T,
                                       np.outer(bloch, bloch.conj()), atol=1e-12)
            expected = 2 * (hopping * np.exp(-2j * np.pi * k[0])).real
            self.assertAlmostEqual((basis.conj().T @ ham @ basis)[0, 0].real, expected)

    def test_multiple_channels_nonorthogonal_basis(self):
        generators, scaled = lattice_generators((2, 3, 1))
        generators = np.array([np.kron(t, np.eye(2)) for t in generators])
        rng = np.random.default_rng(401)
        gauge = np.eye(12) + .08 * (rng.normal(size=(12, 12))
                                   + 1j * rng.normal(size=(12, 12)))
        overlap = gauge.conj().T @ gauge
        transformed = [linalg.solve(gauge, t @ gauge) for t in generators]
        bases = momentum_basis(transformed, overlap, (2, 3, 1), scaled)
        combined = np.concatenate(bases, axis=1)
        np.testing.assert_allclose(combined.conj().T @ overlap @ combined,
                                   np.eye(12), atol=1e-12)
        for k, basis in zip(scaled, bases):
            self.assertEqual(basis.shape, (12, 2))
            for a, t in enumerate(generators):
                np.testing.assert_allclose(t @ gauge @ basis,
                                           np.exp(-2j * np.pi * k[a]) * gauge @ basis,
                                           atol=1e-12)

    def test_empty_sectors_and_gamma(self):
        _, scaled = lattice_generators((3, 1, 1))
        generators = np.ones((3, 1, 1))
        bases = momentum_basis(generators, [[2.]], (3, 1, 1), scaled)
        self.assertEqual([b.shape[1] for b in bases], [1, 0, 0])
        np.testing.assert_allclose(abs(bases[0]), [[1 / np.sqrt(2)]])
        bases = momentum_basis(generators, [[1.]], (1, 1, 1), [[0, 0, 0]])
        np.testing.assert_allclose(abs(bases[0]), [[1.]])

    def test_invalid_representations(self):
        generators, scaled = lattice_generators((3, 1, 1))
        with self.assertRaisesRegex(ValueError, 'unshifted'):
            momentum_basis(generators, np.eye(3), (3, 1, 1), scaled + .1)
        with self.assertRaisesRegex(ValueError, 'duplicates'):
            momentum_basis(generators, np.eye(3), (3, 1, 1), np.zeros((3, 3)))
        with self.assertRaisesRegex(ValueError, 'positive definite'):
            momentum_basis(generators, np.zeros((3, 3)), (3, 1, 1), scaled)
        generators[0, 0, 0] = .1
        with self.assertRaisesRegex(ValueError, 'finite-group'):
            momentum_basis(generators, np.eye(3), (3, 1, 1), scaled)


if __name__ == '__main__':
    unittest.main()
