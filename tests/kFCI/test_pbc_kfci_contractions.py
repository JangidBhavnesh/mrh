#!/usr/bin/env python

import unittest
import numpy as np
import scipy.linalg

from pyscf import fci

from mrh.my_pyscf.pbc.fci import (
    direct_spin1_cplx,
    direct_spin1_cplx_opt,
    direct_spin1_kfci,
)
from mrh.my_pyscf.pbc.fci.direct_spin1_kfci import (
    contract_1e_k,
    contract_2e_k,
)
from mrh.my_pyscf.pbc.fci.addons import KFCIHelperFunctions


# Author: Bhavnesh Jangid

"""
Integration tests for kFCI Hamiltonian contractions.
"""

class ContractionTests(unittest.TestCase):

    def test_contract_1e_k_as_limit_to_nk1(self):
        # In case of nkpts=1, the k-FCI code should reduce to the cplx-FCI.
        helper = KFCIHelperFunctions()

        nelec_cases = [(1, 1), (2, 0), (0, 2), (2, 2), (3, 3), (4, 4)]

        for nelec in nelec_cases:
            with self.subTest(nelec=nelec):
                rng = np.random.default_rng(12)

                nkpts = 1
                ncas = 8
                norb = nkpts * ncas
                target_k = 0

                na = fci.cistring.num_strings(norb, nelec[0])
                nb = fci.cistring.num_strings(norb, nelec[1])

                ci0 = (rng.normal(size=(na,nb)) + 
                       1j * rng.normal( size=( na, nb)))
                ci0 /= np.linalg.norm(ci0)

                fcivec_k = np.asarray(ci0.reshape(-1), order="C")

                h1e_k = (rng.normal(size=(nkpts, ncas, ncas)) +
                         1j * rng.normal(size=(nkpts, ncas, ncas)))

                # Symmetrize the 1e integrals for each k-point
                for k in range(nkpts):
                    h1e_k[k] = 0.5 * (h1e_k[k] + h1e_k[k].conj().T)

                h1e_full = scipy.linalg.block_diag(*h1e_k)
                sigma_k = contract_1e_k(h1e_k, fcivec_k, norb, nelec, nkpts, 
                                        target_k, link_index=None)
                
                sigma_ref = direct_spin1_cplx.contract_1e(h1e_full, ci0, norb, nelec,
                                                          link_index=None).reshape(-1, order="C")

                self.assertEqual(sigma_k.shape, sigma_ref.shape, )
                self.assertTrue(np.allclose( sigma_k, sigma_ref, atol=1e-12, rtol=1e-12, ))

    def test_contract_1e_k(self):
        # Cover several k-point, active-space, and electron-count combinations.
        # The target_k will be varied in the loop (0 to nkpts-1).
        # Basically, the kFCI problem is embedded to full CI problem: and then compared.
        helper = KFCIHelperFunctions()

        # nkpts, ncas, nelec
        test_cases = [
            (2, 3, (1, 1)), (2, 3, (2, 0)), (2, 3, (0, 2)),
            (2, 3, (2, 1)), (2, 3, (2, 2)), (3, 2, (1, 1)),
            (3, 2, (2, 1)),
        ]

        for nkpts, ncas, nelec in test_cases:
            for target_k in range(nkpts):
                with self.subTest( nkpts=nkpts, ncas=ncas, nelec=nelec, 
                                  target_k=target_k):
                    rng = np.random.default_rng(12)
                    norb = nkpts * ncas

                    sector_data = (
                        helper.embed_random_ksector_fcivec_to_full_ci(
                            nkpts, ncas, nelec, target_k=target_k,
                            seed=12))
                    
                    (fcivec_k, ci_full, _, _,
                     straid_k, strbid_k, blocks) = sector_data

                    h1e_k = (rng.normal(size=(nkpts, ncas, ncas))
                             + 1j * rng.normal(size=(nkpts, ncas, ncas)))

                    for k in range(nkpts):
                        h1e_k[k] = 0.5 * (h1e_k[k] + h1e_k[k].conj().T)

                    h1e_full = scipy.linalg.block_diag(*h1e_k)

                    sigma_k = contract_1e_k(
                        h1e_k, fcivec_k, norb, nelec, nkpts, target_k,
                        link_index=None)
                    sigma_full = direct_spin1_cplx.contract_1e(
                        h1e_full, ci_full, norb, nelec, link_index=None, )
                    sigma_ref_k = helper.extract_sector_from_full_ci(
                        sigma_full, blocks, straid_k, strbid_k)

                    self.assertEqual(sigma_k.shape, sigma_ref_k.shape, )
                    self.assertTrue(np.allclose( sigma_k, sigma_ref_k, atol=1e-12, rtol=1e-12))

    def test_contract_2e_k_as_limit_to_nk1(self):
        # In case of nkpts=1, the k-FCI code should reduce to the cplx-FCI.
        nkpts = 1
        ncas = 6
        norb = nkpts * ncas
        target_k = 0

        nelec_cases = [(1, 1), (2, 0), (0, 2), (3, 3), (4, 4),
                       (6, 0), (0, 6), (6, 2), (2, 6)]

        for nelec in nelec_cases:
            with self.subTest(nelec=nelec):
                rng = np.random.default_rng(12)
                na = fci.cistring.num_strings(norb, nelec[0])
                nb = fci.cistring.num_strings(norb, nelec[1])

                # Initial CI vector
                ci0 = ( rng.normal( size=( na, nb)) + 1j * rng.normal( size=( na, nb)))
                ci0 /= np.linalg.norm(ci0)

                # Making it C-contiguous
                fcivec_k = np.asarray(ci0.reshape(-1), order="C")

                # Generate the 2e integrals and symmetrize it.
                eris = (rng.normal(size=(norb, norb, norb, norb))
                        + 1j * rng.normal(size=(norb, norb, norb, norb)))
                eris = 0.5 * (eris + eris.transpose(2, 3, 0, 1))

                # Reshaping the 2e integrals to the k-FCI format
                eri_k = np.zeros(
                    (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas),
                    dtype=eris.dtype)
                eri_k[0, 0, 0] = eris

                # Compute sigma = H * ci0 using both the k-FCI and cplx-FCI
                # code.
                sigma_k = contract_2e_k(
                    eri_k, fcivec_k, norb, nelec, nkpts, target_k)

                sigma_ref = direct_spin1_cplx_opt.contract_2e(
                    eris, ci0, norb, nelec)
                sigma_ref = np.asarray(sigma_ref.ravel(), order="C")

                self.assertEqual(sigma_k.shape, sigma_ref.shape)
                self.assertTrue( np.allclose( sigma_k, sigma_ref, atol=1e-12, rtol=1e-12),
                    msg=("contract_2e_k failed in the limit of nkpts=1 "
                         f"for nelec={nelec}."))

    def test_contract_2e_k(self):
        # Cover several k-point, active-space, and electron-count combinations.
        # The target_k will be varied in the loop (0 to nkpts-1).
        test_cases = [(2, 3, (1, 1)), (2, 3, (2, 0)), (2, 3, (0, 2)),
                      (2, 3, (2, 1)), (2, 3, (2, 2)), (3, 2, (1, 1)),
                      (3, 2, (2, 1)), (3, 4, (2, 2)), (4, 2, (1, 1)),
                      (5, 2, (1, 1))]

        for nkpts, ncas, nelec in test_cases:
            for target_k in range(nkpts):
                with self.subTest(
                        nkpts=nkpts, ncas=ncas, nelec=nelec,
                        target_k=target_k):
                    helper = KFCIHelperFunctions()
                    rng = np.random.default_rng(12)
                    norb = nkpts * ncas

                    sector_data = (
                        helper.embed_random_ksector_fcivec_to_full_ci(
                            nkpts, ncas, nelec, target_k=target_k,
                            seed=12))
                    (fcivec_k, ci_full, link_indexa, link_indexb,
                     straid_k, strbid_k, blocks) = sector_data

                    eri_k = (
                        rng.normal( size=( nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)) + 1j *
                        rng.normal(size=( nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)))

                    eri_full = helper.eri_k_to_full(eri_k)
                    eri_full = 0.5 * \
                        (eri_full + eri_full.transpose(2, 3, 0, 1))
                    eri_k = helper.eri_full_to_k(eri_full, nkpts, ncas)
                    eri_full = helper.eri_k_to_full(eri_k)

                    sigma_k = contract_2e_k(
                        eri_k, fcivec_k, norb, nelec, nkpts, target_k)
                    sigma_full = direct_spin1_cplx_opt.contract_2e(
                        eri_full, ci_full, norb, nelec)
                    sigma_ref_k = helper.extract_sector_from_full_ci(
                        sigma_full, blocks, straid_k, strbid_k, )

                    self.assertEqual(sigma_k.shape, sigma_ref_k.shape)
                    self.assertTrue( np.allclose( sigma_k, sigma_ref_k, atol=1e-12, rtol=1e-12))

    def test_make_hamiltonian_k(self):
        '''
        Test the explicit Hamiltonian against repeated contractions.
        '''
        rng = np.random.default_rng(12)

        nkpts = 2
        ncas = 2
        norb = nkpts * ncas
        nelec = (1, 1)
        target_k = 0

        h1e = (rng.normal(size=(nkpts, ncas, ncas))
               + 1j * rng.normal(size=(nkpts, ncas, ncas)))
        eri = ( rng.normal( size=( nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)) +
            1j * rng.normal( size=( nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)))

        solver = direct_spin1_kfci.FCISolver(nkpts=nkpts, target_k=target_k)
        hmat = solver.make_hamiltonian(h1e, eri, norb, nelec)
        hdiag = solver.make_hdiag(h1e, eri, norb, nelec)
        self.assertTrue( np.allclose( hdiag, np.diag(hmat), atol=1e-12, rtol=1e-12))

        for i in range(hmat.shape[1]):
            ci0 = np.zeros(hmat.shape[0], dtype=hmat.dtype)
            ci0[i] = 1.0
            sigma = solver.contract_ham(h1e, eri, ci0, norb, nelec)
            self.assertTrue(np.allclose(
                hmat[:, i], sigma, atol=1e-12, rtol=1e-12))

    def test_energy_k(self):
        '''
        Test k-FCI energy against explicit Hamiltonian expectation value.
        '''
        rng = np.random.default_rng(12)

        nkpts = 2
        ncas = 2
        norb = nkpts * ncas
        nelec = (1, 1)
        target_k = 0

        helper = KFCIHelperFunctions()
        fcivec_k = helper.random_ksector_fcivec(
            nkpts, ncas, nelec, target_k=target_k, seed=12)[0]

        h1e = (rng.normal(size=(nkpts, ncas, ncas))
               + 1j * rng.normal(size=(nkpts, ncas, ncas)))
        eri = ( rng.normal( size=( nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)) +
            1j * rng.normal( size=( nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)))

        solver = direct_spin1_kfci.FCISolver(nkpts=nkpts, target_k=target_k)

        hmat = solver.make_hamiltonian(h1e, eri, norb, nelec)
        e_ref = np.vdot(fcivec_k, np.dot(hmat, fcivec_k))
        e_test = solver.energy(h1e, eri, fcivec_k, norb, nelec)

        self.assertTrue(np.allclose(e_test, e_ref, atol=1e-12, rtol=1e-12))


if __name__ == "__main__":
    unittest.main()
