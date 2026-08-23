#!/usr/bin/env python

"""Test kFCI spin-penalty diagonals and compatibility utilities.

The tests validate the uint64 population-count helper, the analytic diagonal
of the spin-squared operator, and its use in the spin-penalized Hamiltonian
diagonal.
"""

import unittest

import numpy as np
from pyscf.fci import addons as pyscf_fci_addons

from mrh.my_pyscf.pbc.fci import direct_spin1_kfci

# Author: Bhavnesh Jangid


class KnownValues(unittest.TestCase):

    def test_spin_penalty_inherits_pyscf_mixin_and_undoes(self):
        """Exercise inherited setup/undo and k-FCI base dispatch."""
        nkpts, ncas, nelec, target_k = 2, 2, (1, 1), 0
        norb = nkpts * ncas
        rng = np.random.default_rng(8)
        eri = np.zeros(
            (nkpts,) * 3 + (ncas,) * 4, dtype=np.complex128)
        contract_map = direct_spin1_kfci.make_kfci_contract_map(
            norb, nelec, nkpts, target_k)
        ci0 = (rng.normal(size=contract_map.sector_size)
               + 1j * rng.normal(size=contract_map.sector_size))

        solver = direct_spin1_kfci.FCISolver(
            nkpts=nkpts, target_k=target_k)
        base_result = solver.contract_2e(
            eri, ci0, norb, nelec, contract_map=contract_map)
        expected_penalty = 0.2 * direct_spin1_kfci.contract_ss(
            ci0, norb, nelec, nkpts, target_k=target_k,
            contract_map=contract_map)

        solver.fix_spin_(shift=0.2, ss=0)
        self.assertIsInstance(
            solver, pyscf_fci_addons.SpinPenaltyFCISolver)
        result = solver.contract_2e(
            eri, ci0, norb, nelec, contract_map=contract_map)
        np.testing.assert_allclose(result, base_result + expected_penalty)

        plain_solver = solver.undo_fix_spin()
        self.assertIsInstance(plain_solver, direct_spin1_kfci.FCISolver)
        self.assertNotIsInstance(
            plain_solver, direct_spin1_kfci.SpinPenaltyFCISolver)

    def test_spin_square_diag_matches_contract_ss(self):
        """Compare the analytic S^2 diagonal with explicit contractions.

        Each diagonal element is reconstructed by applying ``contract_ss``
        to its determinant basis vector in every ``target_k`` sector.
        """
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
        """Check direct assembly of the spin-penalized Hamiltonian diagonal.

        For the lowest-spin linear penalty, the result must equal the base
        diagonal plus ``shift * (diag(S^2) - ss)`` in a large sector.
        """
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
