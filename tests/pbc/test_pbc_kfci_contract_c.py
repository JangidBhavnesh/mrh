#!/bin/bash
import unittest
import numpy as np
from pyscf import lib
from pyscf.pbc import gto

from mrh.my_pyscf.pbc.fci.direct_spin1_kfci import _unpack
from mrh.my_pyscf.pbc.fci.direct_spin1_kfci import (
    contract_2e_k,
    contract_2e_k_py,
    make_kfci_contract_map,
    sector_size,
)
from mrh.my_pyscf.pbc.fci import direct_spin1_kfci
from mrh.my_pyscf.pbc.fci.kcistrings import (
    _raise_if_contract_structure_too_large,
)
from mrh.my_pyscf.pbc.fci import kcistrings

class KnownValues(unittest.TestCase):

    @staticmethod
    def _make_2d_kmom():
        cell = gto.Cell()
        cell.a = np.eye(3) * 4.0
        cell.atom = 'He 0 0 0'
        cell.basis = 'sto-3g'
        cell.verbose = 0
        cell.build()
        kpts = cell.make_kpts([2, 2, 1], wrap_around=True)
        return kcistrings.make_kpoint_momentum(len(kpts), cell=cell,
                                               kpts=kpts)

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

    def test_contract_2e_k_matches_python_reference_2d_kmesh(self):
        kmom = self._make_2d_kmom()
        self.assertFalse(kmom.scalar)

        nkpts = kmom.nkpts
        ncas = 2
        nelec = (2, 1)
        norb = nkpts * ncas
        rng = np.random.default_rng(52)
        link_index = _unpack(norb, nelec, None, nkpts, kmom=kmom)
        eri = rng.normal(
            size=(nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)
        )
        eri = eri + 1j * rng.normal(size=eri.shape)

        for target_k in range(nkpts):
            with self.subTest(target_k=target_k):
                contract_map = make_kfci_contract_map(
                    norb, nelec, nkpts, target_k, link_index=link_index,
                    explicit_ab=True, kmom=kmom)
                ci0 = rng.normal(size=contract_map.sector_size)
                ci0 = ci0 + 1j * rng.normal(size=contract_map.sector_size)
                ci0 = np.asarray(ci0, dtype=np.complex128, order="C")

                sigma_ref = contract_2e_k_py(
                    eri, ci0, norb, nelec, nkpts, target_k,
                    link_index=link_index, kmom=kmom)
                sigma_c = contract_2e_k(
                    eri, ci0, norb, nelec, nkpts, target_k,
                    contract_map=contract_map, kmom=kmom)

                np.testing.assert_allclose(
                    sigma_c, sigma_ref, atol=1e-10, rtol=1e-10
                )

    def test_contract_2e_k_streamed_ab_2d_kmesh(self):
        kmom = self._make_2d_kmom()
        nkpts = kmom.nkpts
        ncas = 2
        nelec = (2, 1)
        norb = nkpts * ncas
        target_k = 3
        rng = np.random.default_rng(53)
        link_index = _unpack(norb, nelec, None, nkpts, kmom=kmom)
        contract_map = make_kfci_contract_map(
            norb, nelec, nkpts, target_k, link_index=link_index,
            explicit_ab=False, kmom=kmom)
        self.assertFalse(contract_map.explicit_ab)

        eri = rng.normal(
            size=(nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)
        )
        eri = eri + 1j * rng.normal(size=eri.shape)
        ci0 = rng.normal(size=contract_map.sector_size)
        ci0 = ci0 + 1j * rng.normal(size=contract_map.sector_size)
        ci0 = np.asarray(ci0, dtype=np.complex128, order="C")

        ref_map = make_kfci_contract_map(
            norb, nelec, nkpts, target_k, link_index=link_index,
            explicit_ab=True, kmom=kmom)
        sigma_ref = contract_2e_k(
            eri, ci0, norb, nelec, nkpts, target_k,
            contract_map=ref_map, kmom=kmom)
        sigma_test = contract_2e_k(
            eri, ci0, norb, nelec, nkpts, target_k,
            contract_map=contract_map, kmom=kmom)

        np.testing.assert_allclose(
            sigma_test, sigma_ref, atol=1e-10, rtol=1e-10
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

    def test_contract_2e_k_streamed_ab_matches_python_reference(self):
        nkpts = 3
        ncas = 2
        nelec = (2, 1)
        norb = nkpts * ncas
        rng = np.random.default_rng(31)
        link_index = _unpack(norb, nelec, None, nkpts)
        eri = rng.normal(
            size=(nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)
        )
        eri = eri + 1j * rng.normal(size=eri.shape)

        for target_k in range(nkpts):
            with self.subTest(target_k=target_k):
                contract_map = make_kfci_contract_map(
                    norb, nelec, nkpts, target_k, link_index=link_index,
                    explicit_ab=False)
                self.assertFalse(contract_map.explicit_ab)
                self.assertEqual(contract_map.ab_src_addr.size, 0)

                ci0 = rng.normal(size=contract_map.sector_size)
                ci0 = ci0 + 1j * rng.normal(size=contract_map.sector_size)
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

    def test_contract_2e_k_does_not_rebuild_streamed_map(self):
        nkpts = 3
        ncas = 2
        nelec = (2, 1)
        norb = nkpts * ncas
        target_k = 1
        rng = np.random.default_rng(41)
        link_index = _unpack(norb, nelec, None, nkpts)
        contract_map = make_kfci_contract_map(
            norb, nelec, nkpts, target_k, link_index=link_index,
            explicit_ab=False)
        ci0 = rng.normal(size=contract_map.sector_size)
        ci0 = ci0 + 1j * rng.normal(size=contract_map.sector_size)
        ci0 = np.asarray(ci0, dtype=np.complex128, order="C")
        eri = rng.normal(
            size=(nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)
        )
        eri = eri + 1j * rng.normal(size=eri.shape)

        old_builder = direct_spin1_kfci.make_kfci_contract_map
        try:
            def fail_rebuild(*args, **kwargs):
                raise AssertionError("contract_2e_k rebuilt the map")
            direct_spin1_kfci.make_kfci_contract_map = fail_rebuild
            contract_2e_k(
                eri, ci0, norb, nelec, nkpts, target_k,
                contract_map=contract_map,
            )
        finally:
            direct_spin1_kfci.make_kfci_contract_map = old_builder

    def test_contract_map_auto_skips_large_ab_structure(self):
        nkpts = 8
        ncas = 2
        norb = nkpts * ncas
        nelec = (8, 8)
        target_k = 0
        link_index = _unpack(norb, nelec, None, nkpts)
        contract_map = make_kfci_contract_map(
            norb, nelec, nkpts, target_k, link_index=link_index,
            explicit_ab="auto")

        self.assertFalse(contract_map.explicit_ab)
        self.assertEqual(contract_map.ab_src_addr.size, 0)
        self.assertGreater(contract_map.aa_src_addr.size, 0)
        self.assertGreater(contract_map.bb_src_addr.size, 0)

    def test_contract_structure_size_guard(self):
        max_int32 = np.iinfo(np.int32).max
        with self.assertRaisesRegex(MemoryError, "ab_entries"):
            _raise_if_contract_structure_too_large(max_int32 + 1, 0, 0)

        _raise_if_contract_structure_too_large(max_int32, 0, 0)


if __name__ == "__main__":
    unittest.main()
