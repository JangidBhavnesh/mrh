"""Check LAS translations, fermion signs, and closure of the model space."""

import unittest

import numpy as np
from pyscf.fci import cistring

from mrh.my_pyscf.lassi.op_o0 import _ci_outer_product
from mrh.my_pyscf.pbc.mcscf.klassi_trans import translate_product, project_translation


def full_product(ci, norb, nelec):
    generator = _ci_outer_product(ci, norb, nelec)[0]
    parity = sum(nelec[f, 1] * nelec[f+1:, 0].sum() for f in range(len(norb)))
    return (-1) ** int(parity) * next(generator()).copy()


def translate_full_ci(ci, norb, nelec, orbital_map, phases):
    """Independent determinant permutation in all-alpha/all-beta ordering."""
    maps = []
    for count in nelec:
        strings = cistring.make_strings(range(norb), count)
        addresses = {int(bits): i for i, bits in enumerate(strings)}
        entries = []
        for bits in strings:
            occupied = [p for p in range(norb) if int(bits) & (1 << p)]
            target = [orbital_map[p] for p in occupied]
            parity = sum(target[i] > target[j]
                         for i in range(len(target)) for j in range(i+1, len(target)))
            phase = (-1) ** parity * np.prod(phases[occupied])
            entries.append((addresses[sum(1 << p for p in target)], phase))
        maps.append(entries)
    result = np.zeros_like(ci, dtype=complex)
    for a, (ta, pa) in enumerate(maps[0]):
        for b, (tb, pb) in enumerate(maps[1]):
            result[ta, tb] = pa * pb * ci[a, b]
    return result


class ProductTranslationTests(unittest.TestCase):
    def test_translation_matches_explicit_determinants(self):
        rng = np.random.default_rng(20)
        norb = [2, 2, 2]
        phases = np.exp(1j * np.array([.2, -.3, .7]))
        # Include several odd fragments to exercise nontrivial permutation signs.
        for nelec in (np.array([(1, 0), (1, 1), (0, 1)]),
                      np.array([(1, 1), (2, 1), (1, 0)])):
            ci = []
            for ne in nelec:
                shape = [cistring.num_strings(2, n) for n in ne]
                c = rng.normal(size=shape) + 1j * rng.normal(size=shape)
                ci.append(c / np.linalg.norm(c))
            full = full_product(ci, norb, nelec)
            for permutation in (np.array([1, 2, 0]), np.array([2, 0, 1]),
                                np.array([2, 1, 0])):
                translated, counts, phase = translate_product(ci, nelec, permutation, phases)
                orbital_map = np.array([2 * permutation[f] + p
                                        for f in range(3) for p in range(2)])
                expected = translate_full_ci(full, 6, nelec.sum(axis=0), orbital_map,
                                             np.repeat(phases, 2))
                np.testing.assert_allclose(phase * full_product(translated, norb, counts),
                                           expected, atol=1e-12)
                self.assertFalse(np.shares_memory(translated[permutation[0]], ci[0]))

    def test_translation_group_cycle_and_validation(self):
        ci = [np.ones((1, 1)) for _ in range(2)]
        nelec = np.array([(1, 0), (0, 1)])
        ci1, ne1, phase1 = translate_product(ci, nelec, [1, 0])
        ci2, ne2, phase2 = translate_product(ci1, ne1, [1, 0])
        self.assertEqual(phase1, -1)
        self.assertEqual(phase1 * phase2, 1)
        np.testing.assert_array_equal(ne2, nelec)
        for bad in ([0, 0], [1, 2], [.0, 1.]):
            with self.assertRaisesRegex(ValueError, 'permutation'):
                translate_product(ci, nelec, bad)
        with self.assertRaisesRegex(ValueError, 'unit-modulus'):
            translate_product(ci, nelec, [1, 0], [1, 2])


def translated_basis():
    permutation = [1, 2, 0]
    counts = np.array([(0, 0), (1, 0), (1, 0)])
    ci = [np.ones((1, 1)), np.array([[1.], [0.]]), np.array([[1.], [0.]])]
    states, sectors = [], []
    for _ in range(3):
        states.append(ci)
        sectors.append(counts)
        ci, counts, phase = translate_product(ci, counts, permutation)
        ci[0] = ci[0] * phase
    return [[state[f] for state in states] for f in range(3)], np.stack(sectors, axis=1)


class TranslationProjectionTests(unittest.TestCase):
    def test_complex_gauge_and_group(self):
        ci, counts = translated_basis()
        phases = np.exp(1j * np.array([.4, -.7, 1.1]))
        ci[0] = [c * phase for c, phase in zip(ci[0], phases)]
        result = project_translation(ci, counts, [1, 2, 0])
        expected = np.zeros((3, 3), dtype=complex)
        for source in range(3):
            target = (source + 1) % 3
            expected[target, source] = phases[source] / phases[target]
        np.testing.assert_allclose(result['matrix'], expected, atol=1e-12)
        np.testing.assert_allclose(np.linalg.matrix_power(result['matrix'], 3), np.eye(3), atol=1e-12)
        self.assertLess(result['closure_errors'].max(), 1e-7)

    def test_spectator_response_must_translate(self):
        ci, counts = translated_basis()
        ci[1][0] = np.array([[0.], [1.]])
        result = project_translation(ci, counts, [1, 2, 0])
        self.assertGreater(result['closure_errors'].max(), .9)

    def test_projection_uses_overlap_metric(self):
        # A complete, nonorthogonal one-fragment basis is closed under identity.
        ci = [[np.array([[1.], [0.]]), np.array([[.6j], [.8]])]]
        counts = np.array([[(1, 0), (1, 0)]])
        result = project_translation(ci, counts, [0])
        np.testing.assert_allclose(result['matrix'], np.eye(2), atol=1e-12)
        self.assertAlmostEqual(result['overlap'][0, 1], .6j)
        with self.assertRaisesRegex(ValueError, 'positive definite'):
            project_translation([[ci[0][0], ci[0][0]]], counts, [0])


if __name__ == '__main__':
    unittest.main()
