"""Check LAS translations, fermion signs, and closure of the model space."""

import unittest

import numpy as np
from pyscf.fci import cistring

from mrh.my_pyscf.lassi.op_o0 import _ci_outer_product
from mrh.my_pyscf.pbc.mcscf.klassi_trans import translate_product


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


if __name__ == '__main__':
    unittest.main()
