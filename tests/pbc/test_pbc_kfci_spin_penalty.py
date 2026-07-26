#!/bin/bash
import unittest
import numpy as np

from mrh.my_pyscf.pbc.fci import direct_spin1_kfci


# Author: Bhavnesh Jangid

'''
Tests for k-FCI spin-penalty helpers.
'''


class KnownValues(unittest.TestCase):

    def test_spin_square_diag_matches_contract_ss(self):
        nkpts = 3
        ncas = 2
        norb = nkpts * ncas
        nelec = (2, 1)

        for target_k in range(nkpts):
            contract_map = direct_spin1_kfci.make_kfci_contract_map(
                norb, nelec, nkpts, target_k)
            diag = direct_spin1_kfci._spin_square_diag_k(
                norb, nelec, nkpts, target_k=target_k,
                contract_map=contract_map)

            ref = np.empty(contract_map.sector_size)
            for i in range(contract_map.sector_size):
                ci0 = np.zeros(contract_map.sector_size, dtype=np.complex128)
                ci0[i] = 1.0
                ci1 = direct_spin1_kfci.contract_ss(
                    ci0, norb, nelec, nkpts, target_k=target_k,
                    link_index=contract_map.link_index)
                ref[i] = ci1[i].real

            self.assertTrue(np.allclose(diag, ref))

    def test_spin_penalty_hdiag_avoids_contract_ss_loop(self):
        nkpts = 5
        ncas = 2
        norb = nkpts * ncas
        nelec = (5, 4)
        target_k = 0
        rng = np.random.default_rng(12)
        h1e = np.zeros((nkpts, ncas, ncas), dtype=np.complex128)
        for k in range(nkpts):
            h1e[k] = np.diag(rng.normal(size=ncas))
        eri = np.zeros((nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas),
                       dtype=np.complex128)

        base = direct_spin1_kfci.FCISolver(nkpts=nkpts, target_k=target_k)
        spin_pen = direct_spin1_kfci.FCISolver(nkpts=nkpts, target_k=target_k)
        spin_pen.fix_spin_(shift=0.2, ss=0.75)

        contract_map = direct_spin1_kfci.make_kfci_contract_map(
            norb, nelec, nkpts, target_k)
        hdiag_base = base.make_hdiag(h1e, eri, norb, nelec, nkpts=nkpts,
                                     target_k=target_k,
                                     contract_map=contract_map)
        diag_ss = direct_spin1_kfci._spin_square_diag_k(
            norb, nelec, nkpts, target_k=target_k,
            contract_map=contract_map)
        hdiag_test = spin_pen.make_hdiag(h1e, eri, norb, nelec,
                                         nkpts=nkpts, target_k=target_k,
                                         contract_map=contract_map)

        self.assertEqual(hdiag_test.size, 10584)
        self.assertTrue(np.allclose(hdiag_test,
                                    hdiag_base + 0.2 * (diag_ss - 0.75)))


if __name__ == "__main__":
    unittest.main()
