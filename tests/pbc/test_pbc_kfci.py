#!/bin/bash
import unittest
import numpy as np
import scipy

from pyscf import fci
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc import fci as pbc_fci
from mrh.my_pyscf.pbc.fci import direct_spin1_cplx, direct_spin1_cplx_opt
from mrh.my_pyscf.pbc.fci import direct_spin1_kfci
from mrh.my_pyscf.pbc.fci.direct_spin1_kfci import contract_2e_k, contract_1e_k, _unpack
from mrh.my_pyscf.pbc.fci.kcistrings import gen_k_sector_maps, gen_k_sector_linkstr_info

# Author: Bhavnesh Jangid

'''
Testing k-FCI implementation.
'''

class kFCIHelperFunctions:
    '''
    This class contains helper functions to test the k-FCI implementation. These functions are used to
    arrange the 1e/2e integrals and CI vectors between the k-space format and the full format, and to 
    compare the results from the k-FCI code with the full cplx-FCI code.
    '''
    def __init__(self):
        pass
    
    def h1e_k_to_full(self, h1e_k):
        '''
        Arrange k-space 1e integrals to full 1e integrals. (still in k-space only)
        '''
        h1e_full = scipy.linalg.block_diag(*h1e_k)
        return h1e_full
    
    def eri_k_to_full(self, eri_k):
        '''
        Arrange k-space 2e integrals to full 2e integrals. (still in k-space only)
        '''
        nkpts, ncas = eri_k.shape[0], eri_k.shape[-1]
        norb = nkpts * ncas
        eri_full = np.zeros((norb, norb, norb, norb), dtype=eri_k.dtype)
        for kp, kq, kr in np.ndindex(nkpts, nkpts, nkpts):
            ks = (kp - kq + kr) % nkpts
            P, Q, R, S = kp * ncas, kq * ncas, kr * ncas, ks * ncas
            eri_full[P:P + ncas, Q:Q + ncas, R:R + ncas, S:S + ncas] = eri_k[kp, kq, kr]
        return eri_full

    def eri_full_to_k(self, eri_full, nkpts, ncas):
        '''
        Arrange full 2e integrals to k-space 2e integrals.
        '''
        ef = eri_full.reshape(nkpts, ncas, nkpts, ncas, nkpts, ncas, nkpts, ncas)
        eri_k = np.zeros((nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas), dtype=eri_full.dtype)
        for kp, kq, kr in np.ndindex(nkpts, nkpts, nkpts):
            ks = (kp - kq + kr) % nkpts
            eri_k[kp, kq, kr] = ef[kp, :, kq, :, kr, :, ks, :]
        return eri_k

    def get_ksector_info(self, norb, nelec, nkpts, target_k):
        '''
        Generate the k-sector string maps and block information.
        '''
        link_indexa, link_indexb = _unpack(norb, nelec, None, nkpts)
        straid_k, strbid_k = gen_k_sector_maps(link_indexa, link_indexb, nkpts)[:2]
        blocks = gen_k_sector_linkstr_info(link_indexa, link_indexb, nkpts, target_k)
        return link_indexa, link_indexb, straid_k, strbid_k, blocks

    def embed_sector_fcivec_to_full_ci(self, fcivec_k, blocks, straid_k, strbid_k, 
                                    nstra_total, nstrb_total):
        '''
        In this function, I will create the full CI vector (as zeros) and then only
        fill in the specific block corresponding to each (ka, kb) sector with the sector-wise CI vector.
        '''
        
        ci_full = np.zeros( (nstra_total, nstrb_total), dtype=fcivec_k.dtype)

        for blk in blocks:
            ka, kb, nstra, nstrb, offset, size = map(int, blk)
            block_ka_kb = fcivec_k[offset:offset + size].reshape(nstra, nstrb)

            astrs = straid_k[ka]
            bstrs = strbid_k[kb]

            ci_full[np.ix_(astrs, bstrs)] = block_ka_kb

        return ci_full

    def extract_sector_from_full_ci(self, ci_full, blocks, straid_k, strbid_k):
        '''
        This function extracts the sector-wise CI vector from the full CI vector.
        '''

        sector_size = int(blocks[:, 5].sum())

        fcivec_k = np.zeros(sector_size, dtype=ci_full.dtype)

        for blk in blocks:
            ka, kb, nstra, nstrb, offset, size = map(int, blk)
        
            astrs = straid_k[ka]
            bstrs = strbid_k[kb]

            block = ci_full[np.ix_(astrs, bstrs)]

            fcivec_k[offset:offset + size] = block.reshape(-1)

        return fcivec_k

    def random_ksector_fcivec(self, nkpts, ncas, nelec, target_k=0, seed=12):
        '''
        Generate a random normalized k-sector CI vector and the corresponding maps.
        '''
        rng = np.random.default_rng(seed)
        norb = nkpts * ncas
        link_indexa, link_indexb, straid_k, strbid_k, blocks = \
            self.get_ksector_info(norb, nelec, nkpts, target_k)
        sector_size = int(blocks[:, 5].sum())

        fcivec_k = (rng.normal(size=sector_size) 
                    + 1j * rng.normal(size=sector_size))
        fcivec_k /= np.linalg.norm(fcivec_k)

        return fcivec_k, link_indexa, link_indexb, straid_k, strbid_k, blocks

    def embed_random_ksector_fcivec_to_full_ci(self, nkpts, ncas, nelec, target_k=0, seed=12):
        '''
        Generate a random normalized k-sector CI vector and embed it to the full CI matrix.
        '''
        norb = nkpts * ncas
        fcivec_k, link_indexa, link_indexb, straid_k, strbid_k, blocks = \
            self.random_ksector_fcivec(nkpts, ncas, nelec, target_k=target_k, seed=seed)

        nstra_total = fci.cistring.num_strings(norb, nelec[0])
        nstrb_total = fci.cistring.num_strings(norb, nelec[1])
        ci_full = self.embed_sector_fcivec_to_full_ci(
            fcivec_k, blocks, straid_k, strbid_k, nstra_total, nstrb_total)

        return fcivec_k, ci_full, link_indexa, link_indexb, straid_k, strbid_k, blocks

class KnownValues(unittest.TestCase):

    def test_contract_1e_k_as_limit_to_nk1(self):
        helper = kFCIHelperFunctions()

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

                ci0 = (rng.normal(size=(na, nb)) + 1j * rng.normal(size=(na, nb)))
                ci0 /= np.linalg.norm(ci0)

                fcivec_k = np.asarray(ci0.reshape(-1), order="C")

                h1e_k = ( rng.normal(size=(nkpts, ncas, ncas)) + 
                         1j * rng.normal(size=(nkpts, ncas, ncas)))
                
                for k in range(nkpts):
                    h1e_k[k] = 0.5 * (h1e_k[k] + h1e_k[k].conj().T)

                h1e_full = helper.h1e_k_to_full(h1e_k)

                sigma_k = contract_1e_k(h1e_k, fcivec_k, norb, nelec, nkpts, target_k, link_index=None)
                sigma_ref = direct_spin1_cplx.contract_1e(h1e_full, ci0, norb, nelec, link_index=None).reshape(-1, order="C")

                self.assertEqual( sigma_k.shape, sigma_ref.shape, )
                self.assertTrue( np.allclose( sigma_k, sigma_ref, atol=1e-12, rtol=1e-12, ))

    def test_contract_1e_k(self):
        helper = kFCIHelperFunctions()

        test_cases = [ (2, 3, (1, 1)), (2, 3, (2, 0)), (2, 3, (0, 2)), (2, 3, (2, 1)), 
                      (2, 3, (2, 2)), (3, 2, (1, 1)), (3, 2, (2, 1))]

        for nkpts, ncas, nelec in test_cases:
            for target_k in range(nkpts):
                with self.subTest(nkpts=nkpts, ncas=ncas, nelec=nelec, target_k=target_k):
                    rng = np.random.default_rng(12)
                    norb = nkpts * ncas

                    fcivec_k, ci_full, link_indexa, link_indexb, straid_k, strbid_k, blocks = \
                        helper.embed_random_ksector_fcivec_to_full_ci(nkpts, ncas, nelec, target_k=target_k, seed=12)

                    h1e_k = (rng.normal(size=(nkpts, ncas, ncas)) 
                             + 1j * rng.normal(size=(nkpts, ncas, ncas)))

                    for k in range(nkpts):
                        h1e_k[k] = 0.5 * (h1e_k[k] + h1e_k[k].conj().T)

                    h1e_full = helper.h1e_k_to_full(h1e_k)

                    sigma_k = contract_1e_k( h1e_k, fcivec_k, norb, nelec, nkpts, target_k, link_index=None)
                    sigma_full = direct_spin1_cplx.contract_1e( h1e_full, ci_full, norb, nelec, link_index=None, )
                    sigma_ref_k = helper.extract_sector_from_full_ci( sigma_full, blocks, straid_k, strbid_k)

                    self.assertEqual( sigma_k.shape, sigma_ref_k.shape, )
                    self.assertTrue( np.allclose( sigma_k, sigma_ref_k, atol=1e-12, rtol=1e-12))

    def test_contract_2e_k_as_limit_to_nk1(self):
        # In case of nkpts=1, the k-FCI code should reduce to the cplx-FCI.        
        nkpts = 1
        ncas = 6
        norb = nkpts * ncas
        target_k = 0

        nelec_cases = [(1, 1), (2, 0), (0, 2), (3, 3), (4, 4), (6, 0), (0, 6), (6, 2), (2, 6)]

        for nelec in nelec_cases:
            with self.subTest(nelec=nelec):
                rng = np.random.default_rng(12)
                na = fci.cistring.num_strings(norb, nelec[0])
                nb = fci.cistring.num_strings(norb, nelec[1])

                # Initial CI vector
                ci0 = (rng.normal(size=(na, nb)) + 1j * rng.normal(size=(na, nb)))
                ci0 /= np.linalg.norm(ci0)

                # Making it C-contiguous
                fcivec_k = np.asarray(ci0.reshape(-1), order="C")
                
                # Generate the 2e integrals and symmetrize it.
                eris = (rng.normal(size=(norb, norb, norb, norb)) 
                           + 1j * rng.normal(size=(norb, norb, norb, norb)))
                eris = 0.5 * (eris + eris.transpose(2, 3, 0, 1))

                # Reshaping the 2e integrals to the k-FCI format
                eri_k = np.zeros((nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas), dtype=eris.dtype)
                eri_k[0, 0, 0] = eris

                # Compute sigma = H * ci0 using both the k-FCI and cplx-FCI code.
                sigma_k = contract_2e_k(eri_k, fcivec_k, norb, nelec, nkpts, target_k)

                sigma_ref = direct_spin1_cplx_opt.contract_2e(eris, ci0, norb, nelec)
                sigma_ref = np.asarray(sigma_ref.ravel(), order="C")

                self.assertEqual(sigma_k.shape, sigma_ref.shape)
                self.assertTrue(np.allclose(sigma_k, sigma_ref, atol=1e-12, rtol=1e-12), 
                                msg=(f"contract_2e_k failed in the limit of nkpts=1 for nelec={nelec}."))

    def test_contract_2e_k(self):
        # Defining a variety of test cases with different nkpts, ncas, and nelec. 
        # The target_k will be varied in the loop (0 to nkpts-1).
        test_cases = [(2, 3, (1, 1)), (2, 3, (2, 0)), (2, 3, (0, 2)), 
                      (2, 3, (2, 1)), (2, 3, (2, 2)), (3, 2, (1, 1)), 
                      (3, 2, (2, 1)), (3, 4, (2, 2)), (4, 2, (1, 1)), 
                      (5, 2, (1, 1))]
        
        for nkpts, ncas, nelec in test_cases:
            for target_k in range(nkpts):
                with self.subTest( nkpts=nkpts, ncas=ncas, nelec=nelec, target_k=target_k):
                    helper = kFCIHelperFunctions()
                    rng = np.random.default_rng(12)
                    norb = nkpts * ncas

                    fcivec_k, ci_full, link_indexa, link_indexb, straid_k, strbid_k, blocks = \
                        helper.embed_random_ksector_fcivec_to_full_ci(nkpts, ncas, nelec, target_k=target_k, seed=12)

                    eri_k = ( rng.normal( size=(nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas) ) + 
                            1j * rng.normal( size=(nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas) ) )

                    eri_full = helper.eri_k_to_full(eri_k)
                    eri_full = 0.5 * ( eri_full + eri_full.transpose(2, 3, 0, 1))
                    eri_k = helper.eri_full_to_k( eri_full, nkpts, ncas)
                    eri_full = helper.eri_k_to_full(eri_k)

                    sigma_k = contract_2e_k(eri_k, fcivec_k, norb, nelec, nkpts, target_k)
                    sigma_full = direct_spin1_cplx_opt.contract_2e(eri_full, ci_full, norb, nelec)
                    sigma_ref_k = helper.extract_sector_from_full_ci(sigma_full, blocks, straid_k, strbid_k, )

                    self.assertEqual(sigma_k.shape, sigma_ref_k.shape)
                    self.assertTrue(np.allclose(sigma_k, sigma_ref_k, atol=1e-12, rtol=1e-12))

    def test_make_hamiltonian_k(self):
        '''
        Test explicit Hamiltonian construction against repeated Hamiltonian-vector contractions.
        '''
        rng = np.random.default_rng(12)

        nkpts = 2
        ncas = 2
        norb = nkpts * ncas
        nelec = (1, 1)
        target_k = 0

        h1e = (rng.normal(size=(nkpts, ncas, ncas)) 
               + 1j * rng.normal(size=(nkpts, ncas, ncas)))
        eri = (rng.normal(size=(nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)) 
               + 1j * rng.normal(size=(nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)))

        solver = direct_spin1_kfci.FCISolver(nkpts=nkpts, target_k=target_k)
        hmat = solver.make_hamiltonian(h1e, eri, norb, nelec)
        hdiag = solver.make_hdiag(h1e, eri, norb, nelec)
        self.assertTrue(np.allclose(hdiag, np.diag(hmat), atol=1e-12, rtol=1e-12))

        for i in range(hmat.shape[1]):
            ci0 = np.zeros(hmat.shape[0], dtype=hmat.dtype)
            ci0[i] = 1.0
            sigma = solver.contract_ham(h1e, eri, ci0, norb, nelec)
            self.assertTrue(np.allclose(hmat[:, i], sigma, atol=1e-12, rtol=1e-12))

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

        helper = kFCIHelperFunctions()
        fcivec_k = helper.random_ksector_fcivec(nkpts, ncas, nelec, target_k=target_k, seed=12)[0]

        h1e = (rng.normal(size=(nkpts, ncas, ncas)) 
               + 1j * rng.normal(size=(nkpts, ncas, ncas)))
        eri = (rng.normal(size=(nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)) 
               + 1j * rng.normal(size=(nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)))

        solver = direct_spin1_kfci.FCISolver(nkpts=nkpts, target_k=target_k)

        hmat = solver.make_hamiltonian(h1e, eri, norb, nelec)
        e_ref = np.vdot(fcivec_k, np.dot(hmat, fcivec_k))
        e_test = solver.energy(h1e, eri, fcivec_k, norb, nelec)

        self.assertTrue(np.allclose(e_test, e_ref, atol=1e-12, rtol=1e-12))

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

                self.assertEqual(
                    direct_spin1_kfci.sector_size(
                        norb, nelec, nkpts, target_k
                    ),
                    1,
                )

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
        cell.max_memory = 100000
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
        self.assertTrue(np.allclose(e_kcasci, kmf.e_tot, atol=1e-10, rtol=1e-10))

    def test_fix_spin_k(self):
        '''
        Test k-FCI fix_spin against the explicit spin-penalty Hamiltonian-vector product.
        '''
        rng = np.random.default_rng(12)

        nkpts = 2
        ncas = 2
        norb = nkpts * ncas
        nelec = (1, 1)
        target_k = 0
        shift = 0.3
        ss = 0.0

        helper = kFCIHelperFunctions()
        fcivec_k = helper.random_ksector_fcivec(nkpts, ncas, nelec, target_k=target_k, seed=12)[0]

        h1e = (rng.normal(size=(nkpts, ncas, ncas)) 
               + 1j * rng.normal(size=(nkpts, ncas, ncas)))
        eri = (rng.normal(size=(nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)) 
               + 1j * rng.normal(size=(nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)))

        base_solver = direct_spin1_kfci.FCISolver(nkpts=nkpts, target_k=target_k)
        base_2e = base_solver.contract_2e(eri, fcivec_k, norb, nelec)
        base_sigma = base_solver.contract_ham(h1e, eri, fcivec_k, norb, nelec)
        penalty = shift * (base_solver.contract_ss(fcivec_k, norb, nelec) - ss * fcivec_k)

        ksolver = direct_spin1_kfci.FCISolver(nkpts=nkpts, target_k=target_k)
        pbc_fci.addons.fix_spin_(ksolver, shift=shift, ss=ss)

        self.assertTrue(isinstance(ksolver, direct_spin1_kfci.SpinPenaltyFCISolver))

        sigma_2e = ksolver.contract_2e(eri, fcivec_k, norb, nelec)
        self.assertTrue(np.allclose(sigma_2e, base_2e + penalty, atol=1e-12, rtol=1e-12))

        sigma = ksolver.contract_ham(h1e, eri, fcivec_k, norb, nelec)
        self.assertTrue(np.allclose(sigma, base_sigma + penalty, atol=1e-12, rtol=1e-12))

        hmat = ksolver.make_hamiltonian(h1e, eri, norb, nelec)
        sigma_ref = np.dot(hmat, fcivec_k)
        self.assertTrue(np.allclose(sigma, sigma_ref, atol=1e-12, rtol=1e-12))

    def test_kfci_kernel_direct_and_davidson(self):
        '''
        Test direct and Davidson k-FCI diagonalization for a small Hermitian Hamiltonian.
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

        solver_direct = direct_spin1_kfci.FCISolver(nkpts=nkpts, target_k=target_k)
        solver_direct.verbose = 0
        solver_direct.davidson_only = False
        solver_direct.pspace_size = 100
        e_direct, ci_direct = solver_direct.kernel(h1e, eri, norb, nelec, nroots=1)

        solver_davidson = direct_spin1_kfci.FCISolver(nkpts=nkpts, target_k=target_k)
        solver_davidson.verbose = 0
        solver_davidson.davidson_only = True
        e_davidson, ci_davidson = solver_davidson.kernel(h1e, eri, norb, nelec, nroots=1)

        self.assertTrue(np.allclose(e_direct, e_davidson, atol=1e-10, rtol=1e-10))

        sigma = solver_davidson.contract_ham(h1e, eri, ci_davidson, norb, nelec)
        self.assertTrue(np.linalg.norm(sigma - e_davidson * ci_davidson) < 1e-9)

    def test_contract_ss_k(self):
        '''
        Test S^2 contraction and spin_square for k-FCI against full complex-FCI after embedding.
        '''
        test_cases = [ (2, 2, (1, 1)), (2, 2, (2, 0)), (2, 2, (0, 2)), 
                      (2, 2, (2, 1)), (3, 2, (1, 1))]

        helper = kFCIHelperFunctions()

        for nkpts, ncas, nelec in test_cases:
            for target_k in range(nkpts):
                with self.subTest(nkpts=nkpts, ncas=ncas, nelec=nelec, target_k=target_k):
                    norb = nkpts * ncas

                    fcivec_k, ci_full, link_indexa, link_indexb, straid_k, strbid_k, blocks = \
                        helper.embed_random_ksector_fcivec_to_full_ci(
                            nkpts, ncas, nelec, target_k=target_k, seed=12)

                    kcisolver = direct_spin1_kfci.FCISolver(nkpts=nkpts, target_k=target_k)
                    refsolver = direct_spin1_cplx.FCISolver()

                    ci1_k = kcisolver.contract_ss(fcivec_k.copy(), norb, nelec)
                    ci1_full = refsolver.contract_ss(ci_full.copy(), norb, nelec)
                    ci1_ref = helper.extract_sector_from_full_ci(ci1_full, blocks, straid_k, strbid_k)

                    self.assertEqual(ci1_k.shape, ci1_ref.shape)
                    self.assertTrue(np.allclose(ci1_k, ci1_ref, atol=1e-12, rtol=1e-12))

                    ss_k, mult_k = kcisolver.spin_square(fcivec_k.copy(), norb, nelec)
                    ss_ref, mult_ref = refsolver.spin_square(ci_full.copy(), norb, nelec)

                    self.assertTrue(np.allclose(ss_k, ss_ref, atol=1e-12, rtol=1e-12))
                    self.assertTrue(np.allclose(mult_k, mult_ref, atol=1e-12, rtol=1e-12))

if __name__ == "__main__":
    unittest.main()
