#!/usr/bin/env python
import unittest
import numpy as np

from pyscf.pbc import gto as pgto
from pyscf.pbc import scf

from mrh.my_pyscf.pbc import fci as pbc_fci
from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.fci import (
    direct_spin1_cplx,
    direct_spin1_kfci,
    krdm_helper,
)
from mrh.my_pyscf.pbc.fci.addons import KFCIHelperFunctions


# Author: Bhavnesh Jangid

"""
End-to-end tests for the kFCI solver, spin operations, and energies.
"""


class SolverTests(unittest.TestCase):
    def test_single_determinant_kfci_equals_khf_determinant(self):
        '''
        A fully occupied ncas=1 active space has only one determinant across
        the k mesh.  In that limit k-FCI has no variational/off-diagonal CI
        space, so its energy must be the same single-determinant expectation
        value that k-HF would assign to those occupied k orbitals.
        '''
        rng = np.random.default_rng(19)

        for nkpts in (2, 3, 4):
            with self.subTest(nkpts=nkpts):
                ncas = 1
                norb = nkpts * ncas
                nelec = (nkpts, nkpts)
                target_k = 0

                h1e = rng.normal(size=(nkpts, ncas, ncas))
                h1e = np.asarray(h1e, dtype=np.complex128)
                eri = rng.normal(
                    size=(nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)
                )
                eri = np.asarray(eri, dtype=np.complex128)

                self.assertEqual( direct_spin1_kfci.sector_size( norb, nelec, nkpts, target_k ), 1, )

                ci0 = np.ones(1, dtype=np.complex128)
                sigma = direct_spin1_kfci.contract_ham_k(
                    h1e, eri, ci0, norb, nelec, nkpts, target_k
                )
                e_kfci = direct_spin1_kfci.energy(
                    h1e, eri, ci0, norb, nelec, nkpts, target_k
                )

                e_khf_det = 2.0 * np.sum(h1e[:, 0, 0])
                for ki in range(nkpts):
                    for kj in range(nkpts):
                        coul_ij = eri[ki, ki, kj, 0, 0, 0, 0]
                        coul_ji = eri[kj, kj, ki, 0, 0, 0, 0]
                        e_khf_det += 2.0 * coul_ij
                        e_khf_det += coul_ij + coul_ji

                self.assertTrue(
                    np.allclose(sigma[0], e_khf_det, atol=1e-12, rtol=1e-12)
                )
                self.assertTrue(
                    np.allclose(e_kfci, e_khf_det, atol=1e-12, rtol=1e-12)
                )

    # TODO: Will comment out next line on kCASCI PR:
    @unittest.skipUnless(
        hasattr(mcscf, "KCASCI"), "kCASCI interface is not available")
    def test_single_determinant_kcasci_equals_krhf(self):
        '''
        With one occupied active orbital at each k point and two active
        electrons per cell, the k-FCI sector contains a single determinant.
        The kCASCI energy should therefore reduce to the KRHF determinant
        energy computed from the same orbitals.
        '''
        intraH = 0.74
        interH = 1.5
        vacuum = 17.5

        cell = pgto.Cell()
        cell.a = np.diag([intraH + interH, intraH + interH, vacuum])
        cell.atom = [
            ["H", (0.0, 0.0, vacuum / 2.0)],
            ["H", (intraH, 0.0, vacuum / 2.0)],
        ]
        cell.basis = 'STO-6G'
        cell.unit = 'Angstrom'
        cell.ke_cutoff = 100
        cell.precision = 1e-10
        cell.verbose = 0
        cell.build()

        kmesh = [2, 1, 1]
        kpts = cell.make_kpts(kmesh, wrap_around=True)

        kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
        kmf.max_cycle = 1000
        kmf.exxdiv = None
        kmf.conv_tol = 1e-10
        kmf.verbose = 0
        kmf.kernel()
        self.assertTrue(kmf.converged)

        kmc = mcscf.KCASCI(kmf, 1, 2, target_k=0)
        kmc.kmesh = kmesh
        kmc.verbose = 0
        kmc.fcisolver.verbose = 0
        kmc.canonicalization = False

        e_kcasci = kmc.kernel(np.asarray(kmf.mo_coeff))[0]

        self.assertEqual(np.size(kmc.ci), 1)
        self.assertTrue( np.allclose( e_kcasci, kmf.e_tot, atol=1e-10, rtol=1e-10))

    def test_fix_spin_k(self):
        '''
        Test k-FCI fix_spin against the explicit spin-penalty contraction.
        '''
        rng = np.random.default_rng(12)

        nkpts = 2
        ncas = 2
        norb = nkpts * ncas
        nelec = (1, 1)
        target_k = 0
        shift = 0.3
        ss = 0.0

        helper = KFCIHelperFunctions()
        fcivec_k = helper.random_ksector_fcivec(
            nkpts, ncas, nelec, target_k=target_k, seed=12)[0]

        h1e = (rng.normal(size=(nkpts, ncas, ncas))
               + 1j * rng.normal(size=(nkpts, ncas, ncas)))
        eri = (
            rng.normal( size=( nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)) +
            1j * rng.normal( size=( nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)))

        base_solver = direct_spin1_kfci.FCISolver(
            nkpts=nkpts, target_k=target_k)
        base_2e = base_solver.contract_2e(eri, fcivec_k, norb, nelec)
        base_sigma = base_solver.contract_ham(h1e, eri, fcivec_k, norb, nelec)
        penalty = shift * (base_solver.contract_ss(fcivec_k,
                           norb, nelec) - ss * fcivec_k)

        ksolver = direct_spin1_kfci.FCISolver(nkpts=nkpts, target_k=target_k)
        pbc_fci.addons.fix_spin_(ksolver, shift=shift, ss=ss)

        self.assertTrue( isinstance( ksolver, direct_spin1_kfci.SpinPenaltyFCISolver))

        sigma_2e = ksolver.contract_2e(eri, fcivec_k, norb, nelec)
        self.assertTrue( np.allclose( sigma_2e, base_2e + penalty, atol=1e-12, rtol=1e-12))

        sigma = ksolver.contract_ham(h1e, eri, fcivec_k, norb, nelec)
        self.assertTrue( np.allclose( sigma, base_sigma + penalty, atol=1e-12, rtol=1e-12))

        hmat = ksolver.make_hamiltonian(h1e, eri, norb, nelec)
        sigma_ref = np.dot(hmat, fcivec_k)
        self.assertTrue(
            np.allclose(sigma, sigma_ref, atol=1e-12, rtol=1e-12))

    def test_kfci_kernel_direct_and_davidson(self):
        '''
        Test direct and Davidson diagonalization of a Hermitian Hamiltonian.
        '''
        rng = np.random.default_rng(12)

        nkpts = 2
        ncas = 2
        norb = nkpts * ncas
        nelec = (1, 1)
        target_k = 0

        h1e = (rng.normal(size=(nkpts, ncas, ncas))
               + 1j * rng.normal(size=(nkpts, ncas, ncas)))
        for k in range(nkpts):
            h1e[k] = 0.5 * (h1e[k] + h1e[k].conj().T)

        eri = np.zeros((nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas),
                       dtype=np.complex128)

        solver_direct = direct_spin1_kfci.FCISolver(
            nkpts=nkpts, target_k=target_k)
        solver_direct.verbose = 0
        solver_direct.davidson_only = False
        solver_direct.pspace_size = 100
        e_direct, ci_direct = solver_direct.kernel(
            h1e, eri, norb, nelec, nroots=1)

        solver_davidson = direct_spin1_kfci.FCISolver(
            nkpts=nkpts, target_k=target_k)
        solver_davidson.verbose = 0
        solver_davidson.davidson_only = True
        e_davidson, ci_davidson = solver_davidson.kernel(
            h1e, eri, norb, nelec, nroots=1)

        self.assertTrue( np.allclose( e_direct, e_davidson, atol=1e-10, rtol=1e-10))

        sigma = solver_davidson.contract_ham(
            h1e, eri, ci_davidson, norb, nelec)
        self.assertTrue( np.linalg.norm( sigma - e_davidson * ci_davidson) < 1e-9)

    def test_contract_ss_k(self):
        '''
        Compare k-FCI S^2 operations with embedded full complex-FCI.
        '''
        test_cases = [(2, 2, (1, 1)), (2, 2, (2, 0)), (2, 2, (0, 2)),
                      (2, 2, (2, 1)), (3, 2, (1, 1))]

        helper = KFCIHelperFunctions()

        for nkpts, ncas, nelec in test_cases:
            for target_k in range(nkpts):
                with self.subTest(
                        nkpts=nkpts, ncas=ncas, nelec=nelec,
                        target_k=target_k):
                    norb = nkpts * ncas

                    sector_data = (
                        helper.embed_random_ksector_fcivec_to_full_ci(
                            nkpts, ncas, nelec, target_k=target_k,
                            seed=12))
                    (fcivec_k, ci_full, link_indexa, link_indexb,
                     straid_k, strbid_k, blocks) = sector_data

                    kcisolver = direct_spin1_kfci.FCISolver(
                        nkpts=nkpts, target_k=target_k)
                    refsolver = direct_spin1_cplx.FCISolver()

                    ci1_k = kcisolver.contract_ss(fcivec_k.copy(), norb, nelec)
                    ci1_embedded = krdm_helper.contract_ss_embedded(
                        fcivec_k.copy(), norb, nelec, nkpts,
                        target_k=target_k,
                        link_index=(link_indexa, link_indexb))
                    ci1_full = refsolver.contract_ss(
                        ci_full.copy(), norb, nelec)
                    ci1_ref = helper.extract_sector_from_full_ci(
                        ci1_full, blocks, straid_k, strbid_k)

                    self.assertEqual(ci1_k.shape, ci1_ref.shape)
                    self.assertTrue( np.allclose( ci1_k, ci1_ref, atol=1e-12, rtol=1e-12))
                    self.assertTrue(np.allclose(
                        ci1_k, ci1_embedded, atol=1e-12, rtol=1e-12))

                    ss_k, mult_k = kcisolver.spin_square(
                        fcivec_k.copy(), norb, nelec)
                    ss_ref, mult_ref = refsolver.spin_square(
                        ci_full.copy(), norb, nelec)

                    self.assertTrue( np.allclose( ss_k, ss_ref, atol=1e-12, rtol=1e-12))
                    self.assertTrue( np.allclose( mult_k, mult_ref, atol=1e-12, rtol=1e-12))


if __name__ == "__main__":
    unittest.main()
