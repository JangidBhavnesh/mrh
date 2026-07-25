#!/bin/bash
import unittest
import numpy as np

from mrh.my_pyscf.pbc.fci.direct_spin1_kfci import _unpack, contract_2e_k
from mrh.my_pyscf.pbc.fci.direct_spin1_kfci import (
    contract_2e_k_c,
    contract_2e_k_zgemm,
    sector_size,
)

class KnownValues(unittest.TestCase):

    def test_contract_2e_k_c_matches_python(self):
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
                    ci0 = rng.normal(size=ndet) + 1j * rng.normal(size=ndet)
                    ci0 = np.asarray(ci0, dtype=np.complex128, order="C")

                    sigma_ref = contract_2e_k(
                        eri, ci0, norb, nelec, nkpts, target_k,
                        link_index=link_index,
                    )
                    sigma_c = contract_2e_k_c(
                        eri, ci0, norb, nelec, nkpts, target_k,
                        link_index=link_index,
                    )
                    sigma_zgemm = contract_2e_k_zgemm(
                        eri, ci0, norb, nelec, nkpts, target_k,
                        link_index=link_index,
                    )

                    np.testing.assert_allclose(
                        sigma_c, sigma_ref, atol=1e-10, rtol=1e-10
                    )
                    np.testing.assert_allclose(
                        sigma_zgemm, sigma_ref, atol=1e-10, rtol=1e-10
                    )


if __name__ == "__main__":
    unittest.main()
