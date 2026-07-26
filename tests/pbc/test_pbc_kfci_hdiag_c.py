#!/usr/bin/env python

import unittest

import numpy as np

from mrh.my_pyscf.pbc.fci.direct_spin1_kfci import (
    _unpack,
    make_hdiag,
    make_hdiag_py,
    make_kfci_contract_map,
)


class KnownValues(unittest.TestCase):

    def test_make_hdiag_matches_python_reference(self):
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
            h1e = rng.normal(size=(nkpts, ncas, ncas))
            h1e = h1e + 1j * rng.normal(size=h1e.shape)
            eri = rng.normal(
                size=(nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)
            )
            eri = eri + 1j * rng.normal(size=eri.shape)

            for target_k in range(nkpts):
                with self.subTest(
                    nkpts=nkpts, ncas=ncas, nelec=nelec, target_k=target_k
                ):
                    contract_map = make_kfci_contract_map(
                        norb, nelec, nkpts, target_k, link_index=link_index)

                    hdiag_ref = make_hdiag_py(
                        h1e, eri, norb, nelec, nkpts, target_k,
                        contract_map=contract_map,
                    )
                    hdiag_c = make_hdiag(
                        h1e, eri, norb, nelec, nkpts, target_k,
                        contract_map=contract_map,
                    )

                    np.testing.assert_allclose(
                        hdiag_c, hdiag_ref, atol=1e-10, rtol=1e-10
                    )


if __name__ == "__main__":
    unittest.main()
