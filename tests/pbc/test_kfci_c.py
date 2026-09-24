#!/usr/bin/env python

"""Tests for the compiled k-FCI one-electron contraction."""

import unittest

import numpy as np

from mrh.my_pyscf.pbc.fci.direct_spin1_kfci import (
    _unpack,
    contract_1e_k,
    contract_1e_k_py,
    make_kfci_contract_map,
    sector_size,
)


class KnownValues(unittest.TestCase):

    def test_contract_1e_k_matches_python_reference(self):
        """Compare the compiled contraction with the Python reference."""
        test_cases = [
            (1, 4, (2, 2)),
            (2, 3, (2, 2)),
            (2, 3, (2, 1)),
            (3, 2, (2, 2)),
            (3, 2, (2, 1)),
        ]
        rng = np.random.default_rng(23)

        for nkpts, ncas, nelec in test_cases:
            norb = nkpts * ncas
            link_index = _unpack(norb, nelec, None, nkpts)
            h1e = rng.normal(size=(nkpts, ncas, ncas))
            h1e = h1e + 1j * rng.normal(size=h1e.shape)

            for target_k in range(nkpts):
                with self.subTest(
                        nkpts=nkpts, ncas=ncas, nelec=nelec,
                        target_k=target_k):
                    ndet = sector_size(
                        norb, nelec, nkpts, target_k,
                        link_index=link_index)
                    contract_map = make_kfci_contract_map(
                        norb, nelec, nkpts, target_k,
                        link_index=link_index)
                    ci0 = rng.normal(size=ndet)
                    ci0 = ci0 + 1j * rng.normal(size=ndet)
                    ci0 = np.asarray(ci0, dtype=np.complex128, order="C")

                    sigma_ref = contract_1e_k_py(
                        h1e, ci0, norb, nelec, nkpts, target_k,
                        link_index=link_index)
                    sigma_c = contract_1e_k(
                        h1e, ci0, norb, nelec, nkpts, target_k,
                        contract_map=contract_map)
                    np.testing.assert_allclose(
                        sigma_c, sigma_ref, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
