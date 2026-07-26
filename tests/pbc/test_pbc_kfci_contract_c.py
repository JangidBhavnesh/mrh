#!/bin/bash
import unittest
import numpy as np
from pyscf import lib

from mrh.my_pyscf.pbc.fci.direct_spin1_kfci import _unpack
from mrh.my_pyscf.pbc.fci.direct_spin1_kfci import (
    contract_2e_k,
    contract_2e_k_py,
    make_kfci_contract_map,
    sector_size,
)

class KnownValues(unittest.TestCase):

    def test_contract_2e_k_matches_python_reference(self):
        test_cases = [
            (1, 4, (2, 2)),
            (2, 3, (2, 2)),
            (2, 3, (2, 1)),
            (3, 2, (2, 2)),
            (3, 2, (2, 1)),
        ]

        rng = np.random.default_rng(12)

        for nkpts, ncas, nelec in test_cases:
            norb = nkpts * ncas
            link_index = _unpack(norb, nelec, None, nkpts)
            eri = rng.normal(
                size=(nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)
            )
            eri = eri + 1j * rng.normal(size=eri.shape)

            for target_k in range(nkpts):
                with self.subTest(
                    nkpts=nkpts, ncas=ncas, nelec=nelec, target_k=target_k
                ):
                    ndet = sector_size(
                        norb, nelec, nkpts, target_k, link_index=link_index
                    )
                    contract_map = make_kfci_contract_map(
                        norb, nelec, nkpts, target_k, link_index=link_index)
                    ci0 = rng.normal(size=ndet) + 1j * rng.normal(size=ndet)
                    ci0 = np.asarray(ci0, dtype=np.complex128, order="C")

                    sigma_ref = contract_2e_k_py(
                        eri, ci0, norb, nelec, nkpts, target_k,
                        link_index=link_index,
                    )
                    sigma_c = contract_2e_k(
                        eri, ci0, norb, nelec, nkpts, target_k,
                        contract_map=contract_map,
                    )

                    np.testing.assert_allclose(
                        sigma_c, sigma_ref, atol=1e-10, rtol=1e-10
                    )

    def test_contract_2e_k_thread_consistency(self):
        nkpts = 4
        ncas = 2
        nelec = (4, 4)
        norb = nkpts * ncas
        target_k = 0
        rng = np.random.default_rng(18)
        link_index = _unpack(norb, nelec, None, nkpts)
        contract_map = make_kfci_contract_map(
            norb, nelec, nkpts, target_k, link_index=link_index)
        ndet = sector_size(
            norb, nelec, nkpts, target_k, link_index=link_index
        )
        eri = rng.normal(
            size=(nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)
        )
        eri = eri + 1j * rng.normal(size=eri.shape)
        ci0 = rng.normal(size=ndet) + 1j * rng.normal(size=ndet)
        ci0 = np.asarray(ci0, dtype=np.complex128, order="C")

        saved_threads = lib.num_threads()
        try:
            lib.num_threads(1)
            sigma_1 = contract_2e_k(
                eri, ci0, norb, nelec, nkpts, target_k,
                contract_map=contract_map,
            )
            lib.num_threads(4)
            sigma_4 = contract_2e_k(
                eri, ci0, norb, nelec, nkpts, target_k,
                contract_map=contract_map,
            )
        finally:
            lib.num_threads(saved_threads)

        np.testing.assert_allclose(sigma_4, sigma_1, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
