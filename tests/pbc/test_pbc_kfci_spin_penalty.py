#!/usr/bin/env python

"""Tests for k-FCI spin-penalty edge cases."""

import unittest

import numpy as np

from mrh.my_pyscf.pbc.fci import direct_spin1_kfci


class KnownValues(unittest.TestCase):

    def test_popcount_uint64_python39_compatible(self):
        values = np.asarray(
            [0, 1, 3, 0xffff, 1 << 63, (1 << 64) - 1],
            dtype=np.uint64)
        expected = np.asarray([0, 1, 2, 16, 1, 64], dtype=np.uint16)
        np.testing.assert_array_equal(
            direct_spin1_kfci._popcount_uint64(values), expected)

    def test_spin_square_diag_matches_contract_ss(self):
        nkpts, ncas, nelec = 3, 2, (2, 1)
        norb = nkpts * ncas

        for target_k in range(nkpts):
            contract_map = direct_spin1_kfci.make_kfci_contract_map(
                norb, nelec, nkpts, target_k)
            diag = direct_spin1_kfci._spin_square_diag_k(
                norb, nelec, nkpts, target_k=target_k,
                contract_map=contract_map)

            reference = np.empty(contract_map.sector_size)
            for index in range(contract_map.sector_size):
                ci0 = np.zeros(
                    contract_map.sector_size, dtype=np.complex128)
                ci0[index] = 1.0
                ci1 = direct_spin1_kfci.contract_ss(
                    ci0, norb, nelec, nkpts, target_k=target_k,
                    link_index=contract_map.link_index)
                reference[index] = ci1[index].real
            np.testing.assert_allclose(diag, reference)

    def test_spin_penalty_hdiag_avoids_contract_ss_loop(self):
        nkpts, ncas, nelec, target_k = 5, 2, (5, 4), 0
        norb = nkpts * ncas
        rng = np.random.default_rng(12)
        h1e = np.zeros((nkpts, ncas, ncas), dtype=np.complex128)
        for kpoint in range(nkpts):
            h1e[kpoint] = np.diag(rng.normal(size=ncas))
        eri = np.zeros(
            (nkpts,) * 3 + (ncas,) * 4, dtype=np.complex128)

        base = direct_spin1_kfci.FCISolver(
            nkpts=nkpts, target_k=target_k)
        spin_pen = direct_spin1_kfci.FCISolver(
            nkpts=nkpts, target_k=target_k)
        spin_pen.fix_spin_(shift=0.2, ss=0.75)

        contract_map = direct_spin1_kfci.make_kfci_contract_map(
            norb, nelec, nkpts, target_k)
        hdiag_base = base.make_hdiag(
            h1e, eri, norb, nelec, nkpts=nkpts, target_k=target_k,
            contract_map=contract_map)
        diag_ss = direct_spin1_kfci._spin_square_diag_k(
            norb, nelec, nkpts, target_k=target_k,
            contract_map=contract_map)
        hdiag_test = spin_pen.make_hdiag(
            h1e, eri, norb, nelec, nkpts=nkpts, target_k=target_k,
            contract_map=contract_map)

        self.assertEqual(hdiag_test.size, 10584)
        np.testing.assert_allclose(
            hdiag_test, hdiag_base + 0.2 * (diag_ss - 0.75))


if __name__ == "__main__":
    unittest.main()
