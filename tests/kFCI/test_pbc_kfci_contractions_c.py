#!/usr/bin/env python

"""Test compiled kFCI one- and two-electron Hamiltonian contractions."""

import unittest

import numpy as np

from pyscf import lib
from pyscf.pbc import gto

from mrh.my_pyscf.pbc.fci import direct_spin1_kfci, kcistrings
from mrh.my_pyscf.pbc.fci.direct_spin1_kfci import (
    _unpack,
    contract_1e_k,
    contract_1e_k_py,
    contract_2e_k,
    contract_2e_k_py,
    make_kfci_contract_map,
    sector_size,
)
from mrh.my_pyscf.pbc.fci.kfci_contract_map import (
    _raise_if_contract_map_too_large,
)

# Author: Bhavnesh Jangid


class KnownValues(unittest.TestCase):

    @staticmethod
    def _make_2d_kmom():
        cell = gto.Cell()
        cell.a = np.eye(3) * 4.0
        cell.atom = "He 0 0 0"
        cell.basis = "sto-3g"
        cell.verbose = 0
        cell.build()
        kpts = cell.make_kpts([2, 2, 1], wrap_around=True)
        return kcistrings.make_kpoint_momentum(
            len(kpts), cell=cell, kpts=kpts)

    @staticmethod
    def _random_complex(rng, shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    def test_contract_1e_k_matches_python_reference(self):
        """Compare the compiled 1e contraction with its Python reference."""
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
            h1e = self._random_complex(rng, (nkpts, ncas, ncas))

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
                    ci0 = self._random_complex(rng, ndet)
                    ci0 = np.asarray(ci0, dtype=np.complex128, order="C")

                    sigma_ref = contract_1e_k_py(
                        h1e, ci0, norb, nelec, nkpts, target_k,
                        contract_map=contract_map)
                    sigma_c = contract_1e_k(
                        h1e, ci0, norb, nelec, nkpts, target_k,
                        contract_map=contract_map)
                    np.testing.assert_allclose(
                        sigma_c, sigma_ref, atol=1e-10, rtol=1e-10)

    def test_contract_2e_k_matches_python_reference(self):
        """Compare the compiled 2e contraction with its Python reference.

        Scalar k meshes, multiple electron counts, and every ``target_k``
        sector are included.
        """
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
            eri = self._random_complex(
                rng, (nkpts,) * 3 + (ncas,) * 4)

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
                    ci0 = self._random_complex(rng, ndet)
                    ci0 = np.asarray(ci0, dtype=np.complex128, order="C")

                    sigma_ref = contract_2e_k_py(
                        eri, ci0, norb, nelec, nkpts, target_k,
                        contract_map=contract_map)
                    sigma_c = contract_2e_k(
                        eri, ci0, norb, nelec, nkpts, target_k,
                        contract_map=contract_map)
                    np.testing.assert_allclose(
                        sigma_c, sigma_ref, atol=1e-10, rtol=1e-10)

    def test_contract_2e_k_matches_python_reference_2d_kmesh(self):
        """Compare compiled and Python 2e contractions on a 2D k mesh.

        This exercises non-scalar momentum arithmetic with explicit
        alpha-beta contraction maps in every momentum sector.
        """
        kmom = self._make_2d_kmom()
        self.assertFalse(kmom.scalar)

        nkpts, ncas, nelec = kmom.nkpts, 2, (2, 1)
        norb = nkpts * ncas
        rng = np.random.default_rng(52)
        link_index = _unpack(norb, nelec, None, nkpts, kmom=kmom)
        eri = self._random_complex(rng, (nkpts,) * 3 + (ncas,) * 4)

        for target_k in range(nkpts):
            with self.subTest(target_k=target_k):
                contract_map = make_kfci_contract_map(
                    norb, nelec, nkpts, target_k,
                    link_index=link_index, explicit_ab=True, kmom=kmom)
                ci0 = self._random_complex(
                    rng, contract_map.sector_size)
                ci0 = np.asarray(ci0, dtype=np.complex128, order="C")

                sigma_ref = contract_2e_k_py(
                    eri, ci0, norb, nelec, nkpts, target_k,
                    contract_map=contract_map, kmom=kmom)
                sigma_c = contract_2e_k(
                    eri, ci0, norb, nelec, nkpts, target_k,
                    contract_map=contract_map, kmom=kmom)
                np.testing.assert_allclose(
                    sigma_c, sigma_ref, atol=1e-10, rtol=1e-10)

    def test_contract_2e_k_streamed_ab_2d_kmesh(self):
        """Compare streamed and explicit alpha-beta contractions in 2D.

        The same-spin maps remain explicit while the alpha-beta terms
        are generated from link indices during contraction.
        """
        kmom = self._make_2d_kmom()
        nkpts, ncas, nelec, target_k = kmom.nkpts, 2, (2, 1), 3
        norb = nkpts * ncas
        rng = np.random.default_rng(53)
        link_index = _unpack(norb, nelec, None, nkpts, kmom=kmom)
        contract_map = make_kfci_contract_map(
            norb, nelec, nkpts, target_k, link_index=link_index,
            explicit_ab=False, kmom=kmom)
        self.assertFalse(contract_map.explicit_ab)

        eri = self._random_complex(rng, (nkpts,) * 3 + (ncas,) * 4)
        ci0 = self._random_complex(rng, contract_map.sector_size)
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
            sigma_test, sigma_ref, atol=1e-10, rtol=1e-10)

    def test_contract_2e_k_thread_consistency(self):
        """Verify that one and four OpenMP threads produce the same result."""
        nkpts, ncas, nelec, target_k = 4, 2, (4, 4), 0
        norb = nkpts * ncas
        rng = np.random.default_rng(18)
        link_index = _unpack(norb, nelec, None, nkpts)
        contract_map = make_kfci_contract_map(
            norb, nelec, nkpts, target_k, link_index=link_index)
        ndet = sector_size(
            norb, nelec, nkpts, target_k, link_index=link_index)
        eri = self._random_complex(rng, (nkpts,) * 3 + (ncas,) * 4)
        ci0 = self._random_complex(rng, ndet)
        ci0 = np.asarray(ci0, dtype=np.complex128, order="C")

        saved_threads = lib.num_threads()
        try:
            lib.num_threads(1)
            sigma_1 = contract_2e_k(
                eri, ci0, norb, nelec, nkpts, target_k,
                contract_map=contract_map)
            lib.num_threads(4)
            sigma_4 = contract_2e_k(
                eri, ci0, norb, nelec, nkpts, target_k,
                contract_map=contract_map)
        finally:
            lib.num_threads(saved_threads)

        np.testing.assert_allclose(
            sigma_4, sigma_1, atol=1e-10, rtol=1e-10)

    def test_contract_2e_k_streamed_ab_matches_python_reference(self):
        """Compare streamed alpha-beta contraction with Python output.

        The test covers every ``target_k`` sector of a scalar k mesh without
        storing the explicit alpha-beta sparse map.
        """
        nkpts, ncas, nelec = 3, 2, (2, 1)
        norb = nkpts * ncas
        rng = np.random.default_rng(31)
        link_index = _unpack(norb, nelec, None, nkpts)
        eri = self._random_complex(rng, (nkpts,) * 3 + (ncas,) * 4)

        for target_k in range(nkpts):
            with self.subTest(target_k=target_k):
                contract_map = make_kfci_contract_map(
                    norb, nelec, nkpts, target_k,
                    link_index=link_index, explicit_ab=False)
                self.assertFalse(contract_map.explicit_ab)
                self.assertEqual(contract_map.ab_src_addr.size, 0)

                ci0 = self._random_complex(
                    rng, contract_map.sector_size)
                ci0 = np.asarray(ci0, dtype=np.complex128, order="C")
                sigma_ref = contract_2e_k_py(
                    eri, ci0, norb, nelec, nkpts, target_k,
                    contract_map=contract_map)
                sigma_c = contract_2e_k(
                    eri, ci0, norb, nelec, nkpts, target_k,
                    contract_map=contract_map)
                np.testing.assert_allclose(
                    sigma_c, sigma_ref, atol=1e-10, rtol=1e-10)

    def test_contract_2e_k_does_not_rebuild_streamed_map(self):
        """Ensure contraction reuses a supplied streamed contract map."""
        nkpts, ncas, nelec, target_k = 3, 2, (2, 1), 1
        norb = nkpts * ncas
        rng = np.random.default_rng(41)
        link_index = _unpack(norb, nelec, None, nkpts)
        contract_map = make_kfci_contract_map(
            norb, nelec, nkpts, target_k, link_index=link_index,
            explicit_ab=False)
        ci0 = self._random_complex(rng, contract_map.sector_size)
        ci0 = np.asarray(ci0, dtype=np.complex128, order="C")
        eri = self._random_complex(rng, (nkpts,) * 3 + (ncas,) * 4)

        old_builder = direct_spin1_kfci.make_kfci_contract_map
        try:
            def fail_rebuild(*args, **kwargs):
                raise AssertionError("contract_2e_k rebuilt the map")

            direct_spin1_kfci.make_kfci_contract_map = fail_rebuild
            contract_2e_k(
                eri, ci0, norb, nelec, nkpts, target_k,
                contract_map=contract_map)
        finally:
            direct_spin1_kfci.make_kfci_contract_map = old_builder

    def test_contract_map_auto_skips_large_ab_map(self):
        """Ensure auto mode streams an oversized alpha-beta map.

        Same-spin maps must remain available when the explicit
        alpha-beta address arrays are omitted.
        """
        nkpts, ncas, nelec, target_k = 8, 2, (8, 8), 0
        norb = nkpts * ncas
        link_index = _unpack(norb, nelec, None, nkpts)
        contract_map = make_kfci_contract_map(
            norb, nelec, nkpts, target_k, link_index=link_index,
            explicit_ab="auto", max_memory=1 << 50)

        self.assertFalse(contract_map.explicit_ab)
        self.assertEqual(contract_map.ab_src_addr.size, 0)
        self.assertGreater(contract_map.aa_src_addr.size, 0)
        self.assertGreater(contract_map.bb_src_addr.size, 0)

    def test_contract_map_size_guard(self):
        """Check the signed 32-bit contraction-map size boundary."""
        max_int32 = np.iinfo(np.int32).max
        with self.assertRaisesRegex(MemoryError, "ab_entries"):
            _raise_if_contract_map_too_large(max_int32 + 1, 0, 0)
        _raise_if_contract_map_too_large(max_int32, 0, 0)


if __name__ == "__main__":
    unittest.main()
