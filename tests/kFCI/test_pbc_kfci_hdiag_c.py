#!/usr/bin/env python


import unittest

import numpy as np

from mrh.my_pyscf.pbc.fci.direct_spin1_kfci import (
    _unpack,
    make_hdiag,
    make_hdiag_py,
    make_kfci_contract_map,
)


# Author: Bhavnesh Jangid

"""Tests for the compiled k-FCI Hamiltonian diagonal."""

class KnownValues(unittest.TestCase):

    @staticmethod
    def _random_complex(rng, shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    def test_make_hdiag_matches_python_reference(self):
        """Compare the compiled Hamiltonian diagonal with Python output.

        The comparison covers multiple orbital and electron configurations in
        every available ``target_k`` sector.
        """
        test_cases = [
            (1, 4, (2, 2)),
            (2, 3, (2, 2)),
            (2, 3, (2, 1)),
            (3, 2, (2, 2)),
            (3, 2, (2, 1)),
        ]
        rng = np.random.default_rng(91)

        for nkpts, ncas, nelec in test_cases:
            norb = nkpts * ncas
            link_index = _unpack(norb, nelec, None, nkpts)
            h1e = self._random_complex(rng, (nkpts, ncas, ncas))
            eri = self._random_complex(
                rng, (nkpts,) * 3 + (ncas,) * 4)

            for target_k in range(nkpts):
                with self.subTest(
                        nkpts=nkpts, ncas=ncas, nelec=nelec,
                        target_k=target_k):
                    contract_map = make_kfci_contract_map(
                        norb, nelec, nkpts, target_k,
                        link_index=link_index)
                    hdiag_ref = make_hdiag_py(
                        h1e, eri, norb, nelec, nkpts, target_k,
                        contract_map=contract_map)
                    hdiag_c = make_hdiag(
                        h1e, eri, norb, nelec, nkpts, target_k,
                        contract_map=contract_map)
                    np.testing.assert_allclose(
                        hdiag_c, hdiag_ref, atol=1e-10, rtol=1e-10)

    def test_make_hdiag_streamed_ab_matches_python_reference(self):
        """Check the diagonal with streamed alpha-beta contractions.

        The compiled streamed-map result is compared with the Python diagonal
        constructed from an explicit alpha-beta contract map.
        """
        nkpts, ncas, nelec = 3, 2, (2, 1)
        norb = nkpts * ncas
        rng = np.random.default_rng(117)
        link_index = _unpack(norb, nelec, None, nkpts)
        h1e = self._random_complex(rng, (nkpts, ncas, ncas))
        eri = self._random_complex(rng, (nkpts,) * 3 + (ncas,) * 4)

        for target_k in range(nkpts):
            with self.subTest(target_k=target_k):
                map_stream = make_kfci_contract_map(
                    norb, nelec, nkpts, target_k,
                    link_index=link_index, explicit_ab=False)
                map_explicit = make_kfci_contract_map(
                    norb, nelec, nkpts, target_k,
                    link_index=link_index, explicit_ab=True)
                self.assertFalse(map_stream.explicit_ab)
                self.assertEqual(map_stream.ab_src_addr.size, 0)

                hdiag_ref = make_hdiag_py(
                    h1e, eri, norb, nelec, nkpts, target_k,
                    contract_map=map_explicit)
                hdiag_c = make_hdiag(
                    h1e, eri, norb, nelec, nkpts, target_k,
                    contract_map=map_stream)
                np.testing.assert_allclose(
                    hdiag_c, hdiag_ref, atol=1e-10, rtol=1e-10)

    def test_hdiag_auto_map_skips_large_ab_map(self):
        """Ensure auto mode omits an oversized alpha-beta map.

        The same-spin contraction maps must remain available when the
        explicit alpha-beta address arrays are not built.
        """
        nkpts, ncas, nelec, target_k = 8, 2, (8, 8), 0
        norb = nkpts * ncas
        link_index = _unpack(norb, nelec, None, nkpts)
        contract_map = make_kfci_contract_map(
            norb, nelec, nkpts, target_k, link_index=link_index,
            explicit_ab="auto")

        self.assertFalse(contract_map.explicit_ab)
        self.assertEqual(contract_map.ab_src_addr.size, 0)
        self.assertGreater(contract_map.aa_src_addr.size, 0)
        self.assertGreater(contract_map.bb_src_addr.size, 0)


if __name__ == "__main__":
    unittest.main()
