#!/bin/env python

import unittest
import numpy as np

from pyscf import ao2mo, gto, scf
from pyscf.csf_fci import csf_solver
from pyscf.fci import cistring

from mrh.my_pyscf.pbc.fci import cplx_csf_helper, csf_cplx
from mrh.my_pyscf.pbc.fci import csf_solver as csf_solver_cplx

# Let me set up the mol once and use it for all the tests.
mol = mf = h0 = h1 = h2 = h0cplx = h1cplx = h2cplx = None

# Author: Bhavnesh Jangid

# Test-0: CSFSolver vs CSFSolver_cplx for Hermitian Hamiltonian with
#         small imaginary noise. The energies should match within a 
#         tight tolerance.
# Test-1: Compare the low-level complex spin-separated one-electron
#         contraction against an explicit determinant-space reference.
# Test-2: Compare the Davidson vs non-davidson solvers.
# Test-3: CSFSolver with various integeric and non-integeric spin-states
# Test-4: Verify that the complex one-electron contraction rejects
#       packed-triangular link-index tables.
# Test-5: Verify the spin-dependent one-electron contribution through the
#          complex CSF solver's absorb_h1e and contract_2e path.
# Test-6: Compare the batched complex determinant-to-CSF transformation
#         against separate real and imaginary transformations.
# Test-7: Compare the batched complex CSF-to-determinant transformation
#         against separate real and imaginary transformations.
# Test-8: Compare multi-root Davidson against full CSF p-space
#         diagonalization for a genuinely complex Hermitian Hamiltonian.

def gen_hermi_ham(h0, h1, h2):
    np.random.seed(12)
    h0 = h0.astype(np.complex128)
    h1 = np.asarray(h1).astype(np.complex128)
    h2 = np.asarray(h2).astype(np.complex128)

    h1.imag += 1e-4
    h2.imag += 1e-4

    # Restore symmetries
    h1 = 0.5 * (h1 + h1.conj().T)
    h2 = 0.5 * (h2 + h2.conj().transpose(1,0,2,3))
    h2 = 0.5 * (h2 + h2.conj().transpose(0,1,3,2))
    h2 = 0.5 * (h2 + h2.conj().transpose(2,3,0,1))
    return h0, h1, h2

def gen_random_hermi_ham(norb):
    np.random.seed(12)
    h1 = np.random.random((norb,norb))
    h2 = np.random.random((norb,norb,norb,norb))

    h1 = np.asarray(h1).astype(np.complex128)
    h2 = np.asarray(h2).astype(np.complex128)

    h1.imag += 1e-4
    h2.imag += 1e-4

    # Restore symmetries
    h1 = h1 + h1.conj().T
    h2 = h2 + h2.conj().transpose(1,0,2,3)
    h2 = h2 + h2.conj().transpose(0,1,3,2)
    h2 = h2 + h2.conj().transpose(2,3,0,1)
    return h1, h2

def gen_random_complex_hermi_ham(norb):
    rng = np.random.default_rng(34)
    h1_ao = rng.standard_normal((norb, norb))
    h1_ao = h1_ao + h1_ao.T

    chol = rng.standard_normal((2 * norb, norb, norb))
    chol = chol + chol.transpose(0, 2, 1)
    eri_ao = np.einsum('Lpq,Lrs->pqrs', chol, chol)

    mo = (
        rng.standard_normal((norb, norb))
        + 1j * rng.standard_normal((norb, norb))
    )
    mo = np.linalg.qr(mo)[0]

    h1 = np.einsum('pi,pq,qj->ij', mo.conj(), h1_ao, mo)
    eri = np.einsum(
        'pi,qj,rk,sl,pqrs->ijkl',
        mo.conj(), mo, mo.conj(), mo, eri_ao,
        optimize=True,
    )
    return h1, eri

def contract_1e_reference(h1e, fcivec, link_index):
    link_indexa, link_indexb = link_index
    na = link_indexa.shape[0]
    nb = link_indexb.shape[0]
    ci0 = np.asarray(fcivec).reshape(na, nb)
    ci1 = np.zeros_like(ci0, dtype=np.result_type(h1e, fcivec))

    for stra, links in enumerate(link_indexa):
        for a, i, str1, sign in links:
            ci1[str1, :] += sign * h1e[0, a, i] * ci0[stra, :]

    for strb, links in enumerate(link_indexb):
        for a, i, str1, sign in links:
            ci1[:, str1] += sign * h1e[1, a, i] * ci0[:, strb]

    return ci1

def auto_setup():
    mol = gto.Mole(atom='H 0 0 0; F 0 0 1.1', basis='STO-6G',
                verbose=0, output=None)
    mol.build ()
    mf = scf.RHF (mol).run ()
    
    h0 = mf.energy_nuc ()
    h1 = mf.mo_coeff.conj ().T @ mf.get_hcore () @ mf.mo_coeff
    h2 = ao2mo.restore (1, ao2mo.full (mf._eri, mf.mo_coeff), mol.nao_nr ())
    h0cplx, h1cplx, h2cplx = gen_hermi_ham(h0, h1, h2)
    return mol, mf, h0, h1, h2, h0cplx, h1cplx, h2cplx

def setUpModule():
    global mol, mf, h0, h1, h2, h0cplx, h1cplx, h2cplx, norb
    mol, mf, h0, h1, h2, h0cplx, h1cplx, h2cplx = auto_setup()
    norb = mol.nao_nr ()

class KnownValues(unittest.TestCase):
    def get_transformer(self, norb=5, nelec=(3, 2), smult=2):
        solver = csf_cplx.FCISolver(gto.M(verbose=0), smult=smult)
        solver.norb = norb
        solver.nelec = nelec
        solver.check_transformer_cache()
        return solver.transformer

    def get_random_uhf_h1e_data(self):
        norb = 5
        nelec = (3, 2)
        rng = np.random.default_rng(31)

        h1e = rng.standard_normal((2, norb, norb))
        h1e = h1e + 1j * rng.standard_normal(h1e.shape)
        h1e = h1e + h1e.conj().transpose(0, 2, 1)

        na = cistring.num_strings(norb, nelec[0])
        nb = cistring.num_strings(norb, nelec[1])
        fcivec = (
            rng.standard_normal((na, nb))
            + 1j * rng.standard_normal((na, nb))
        )
        link_index = (
            cistring.gen_linkstr_index(range(norb), nelec[0]),
            cistring.gen_linkstr_index(range(norb), nelec[1]),
        )
        return norb, nelec, h1e, fcivec, link_index

    def test_contract_1e_cplx_uhf(self):
        norb, nelec, h1e, fcivec, link_index = (
            self.get_random_uhf_h1e_data()
        )
        reference = contract_1e_reference(h1e, fcivec, link_index)
        result = cplx_csf_helper.contract_1e(
            h1e, fcivec, norb, nelec, link_index
        )
        self.assertLess(np.max(np.abs(result - reference)), 1e-12)

    def test_contract_1e_cplx_uhf_rejects_tril_link_index(self):
        norb, nelec, h1e, fcivec, _ = self.get_random_uhf_h1e_data()
        tril_links = (
            cistring.gen_linkstr_index_trilidx(range(norb), nelec[0]),
            cistring.gen_linkstr_index_trilidx(range(norb), nelec[1]),
        )
        with self.assertRaises(ValueError):
            cplx_csf_helper.contract_1e(
                h1e, fcivec, norb, nelec, tril_links
            )

    def test_csf_spin_dependent_h1e_path(self):
        norb, nelec, h1e, fcivec, link_index = (
            self.get_random_uhf_h1e_data()
        )
        solver = csf_cplx.FCISolver(gto.M(verbose=0), smult=2)
        eri = np.zeros((norb,) * 4, dtype=np.complex128)

        h2eff = solver.absorb_h1e(h1e, eri, norb, nelec, fac=0.5)
        result = solver.contract_2e(
            h2eff, fcivec, norb, nelec, link_index
        )
        reference = contract_1e_reference(h1e, fcivec, link_index)
        self.assertLess(np.max(np.abs(result - reference)), 1e-11)

        hdiag_det = solver.make_hdiag(h1e, eri, norb, nelec)
        self.assertEqual(np.max(np.abs(hdiag_det.imag)), 0)
        hdiag_csf = solver.make_hdiag_csf(
            h1e, eri, norb, nelec, hdiag_det=hdiag_det
        )
        _, h0 = solver.pspace(
            h1e, eri, norb, nelec,
            hdiag_det=hdiag_det, hdiag_csf=hdiag_csf, npsp=3,
        )
        self.assertLess(np.max(np.abs(h0 - h0.conj().T)), 1e-12)
        self.assertEqual(np.max(np.abs(np.diag(h0).imag)), 0)

    def test_vec_det2csf_cplx(self):
        transformer = self.get_transformer()
        rng = np.random.default_rng(32)
        civec = (
            rng.standard_normal((3, transformer.ndet))
            + 1j * rng.standard_normal((3, transformer.ndet))
        )
        reference = (
            transformer.vec_det2csf(civec.real, normalize=False)
            + 1j * transformer.vec_det2csf(civec.imag, normalize=False)
        )
        result = cplx_csf_helper.vec_det2csf_cplx(
            transformer, civec, normalize=False
        )
        self.assertLess(np.max(np.abs(result - reference)), 1e-12)
        result_single = cplx_csf_helper.vec_det2csf_cplx(
            transformer, civec[0], normalize=False
        )
        self.assertLess(np.max(np.abs(result_single - reference[0])), 1e-12)

        result, norms = cplx_csf_helper.vec_det2csf_cplx(
            transformer, civec, normalize=True, return_norm=True
        )
        self.assertLess(np.max(np.abs(norms - np.linalg.norm(reference, axis=1))), 1e-12)
        self.assertLess(np.max(np.abs(np.linalg.norm(result, axis=1) - 1)), 1e-12)

    def test_vec_csf2det_cplx(self):
        transformer = self.get_transformer()
        rng = np.random.default_rng(33)
        civec = (
            rng.standard_normal((3, transformer.ncsf))
            + 1j * rng.standard_normal((3, transformer.ncsf))
        )
        reference = (
            transformer.vec_csf2det(civec.real, normalize=False)
            + 1j * transformer.vec_csf2det(civec.imag, normalize=False)
        )
        result = cplx_csf_helper.vec_csf2det_cplx(
            transformer, civec, normalize=False
        )
        self.assertLess(np.max(np.abs(result - reference)), 1e-12)
        result_single = cplx_csf_helper.vec_csf2det_cplx(
            transformer, civec[0], normalize=False
        )
        self.assertLess(np.max(np.abs(result_single - reference[0])), 1e-12)

        result, norms = cplx_csf_helper.vec_csf2det_cplx(
            transformer, civec, normalize=True, return_norm=True
        )
        self.assertLess(np.max(np.abs(norms - np.linalg.norm(reference, axis=1))), 1e-12)
        self.assertLess(np.max(np.abs(np.linalg.norm(result, axis=1) - 1)), 1e-12)

    def test_multiroot_davidson_cplx(self):
        norb = 4
        nelec = (2, 2)
        h1e, eri = gen_random_complex_hermi_ham(norb)
        self.assertGreater(np.max(np.abs(h1e.imag)), 1e-6)
        self.assertGreater(np.max(np.abs(eri.imag)), 1e-6)

        exact_solver = csf_solver_cplx(mol, smult=1)
        exact_solver.nroots = 2
        exact_solver.davidson_only = False
        exact_solver.pspace_size = 200
        e_exact = exact_solver.kernel(h1e, eri, norb, nelec)[0]

        davidson_solver = csf_solver_cplx(mol, smult=1)
        davidson_solver.nroots = 2
        davidson_solver.davidson_only = True
        davidson_solver.pspace_size = 0
        e_davidson = davidson_solver.kernel(h1e, eri, norb, nelec)[0]

        self.assertLess(np.max(np.abs(e_davidson - e_exact)), 1e-9)

    def test_vanilla_csf_solver_cplx(self):
        nelec = (5, 5)
        real_cisolver = csf_solver (mol, smult=1)
        eci, civec = real_cisolver.kernel (h1, h2, norb, nelec, ecore=h0)
        eci_energyFunc = real_cisolver.energy(h1, h2, civec, norb, nelec) + h0
        norm_ci_dev = 1 - np.linalg.norm(civec)

        cplx_cisolver = csf_solver_cplx(mol, smult=1)
        eci1, civec1 = cplx_cisolver.kernel (h1cplx, h2cplx, norb, nelec, ecore=h0cplx)
        eci1_energyFunc = cplx_cisolver.energy(h1cplx, h2cplx, civec1, norb, nelec) + h0cplx
        norm_ci1_dev = 1 - np.linalg.norm(civec1)

        self.assertAlmostEqual(eci_energyFunc, eci, places=6)
        self.assertAlmostEqual(norm_ci_dev, 1e-7, places=6)
        self.assertAlmostEqual(eci1_energyFunc, eci1, places=6)
        self.assertAlmostEqual(norm_ci1_dev, 1e-7, places=6)
        self.assertAlmostEqual(eci, eci1, places=6)
        self.assertAlmostEqual(eci_energyFunc, eci1_energyFunc, places=6)

    def test_vanilla_csf_solver_cplx_solvers(self):
        nelec = (5, 5)
        real_cisolver = csf_solver (mol, smult=1)
        eci = real_cisolver.kernel (h1, h2, norb, nelec, ecore=h0)[0]
        
        cplx_cisolver = csf_solver_cplx(mol, smult=1)
        cplx_cisolver.davidson_only = False
        eci1, civec1 = cplx_cisolver.kernel (h1cplx, h2cplx, norb, nelec, ecore=h0cplx)
        norm_ci1_dev = 1 - np.linalg.norm(civec1)

        # p-space davidson diagonalization
        cplx_cisolver = csf_solver_cplx(mol, smult=1)
        cplx_cisolver.davidson_only = True
        cplx_cisolver.pspace_size = 10
        eci2, civec2 = cplx_cisolver.kernel (h1cplx, h2cplx, norb, nelec, ecore=h0cplx)
        norm_ci2_dev = 1 - np.linalg.norm(civec2)

        # non-p-space davidson diagonalization
        cplx_cisolver = csf_solver_cplx(mol, smult=1)
        cplx_cisolver.davidson_only = True
        cplx_cisolver.pspace_size = 0
        eci3, civec3 = cplx_cisolver.kernel (h1cplx, h2cplx, norb, nelec, ecore=h0cplx)
        norm_ci3_dev = 1 - np.linalg.norm(civec3)

        self.assertAlmostEqual(eci1, eci, places=6)
        self.assertAlmostEqual(eci2, eci, places=6)
        self.assertAlmostEqual(eci3, eci, places=6)
        self.assertAlmostEqual(norm_ci1_dev, 1e-7, places=6)
        self.assertAlmostEqual(norm_ci2_dev, 1e-7, places=6)
        self.assertAlmostEqual(norm_ci3_dev, 1e-7, places=6)

    def test_vanilla_csf_solver_cplx_spin_states(self):
        norb = 4
        h1cplx, h2cplx = gen_random_hermi_ham(norb)
        h1, h2 = h1cplx.real, h2cplx.real
        h0 = h0cplx.real

        def run_real_csf_solver(nelec, smult):
            real_cisolver = csf_solver (mol, smult=smult)
            eci, civec = real_cisolver.kernel (h1, h2, norb, nelec, ecore=h0)
            from pyscf.fci import spin_square
            s2, smultout = spin_square(civec, norb, nelec)
            assert smultout - smult < 1e-4
            return eci
        
        def run_cplx_csf_solver(nelec, smult):
            cplx_cisolver = csf_solver_cplx(mol, smult=smult)
            eci1, civec1 = cplx_cisolver.kernel (h1cplx, h2cplx, norb, nelec, ecore=h0cplx)
            from mrh.my_pyscf.pbc.fci import spin_op
            s2, smultout = spin_op.spin_square0(civec1, norb, nelec)
            assert smultout - smult < 1e-4
            return eci1
        
        eci_sing = run_real_csf_solver((2, 2), 1)
        eci1_sing = run_cplx_csf_solver((2, 2), 1)
        eci_trip = run_real_csf_solver((3, 1), 3)
        eci1_trip = run_cplx_csf_solver((3, 1), 3)
       
        self.assertAlmostEqual(eci_sing, eci1_sing, places=6)
        self.assertAlmostEqual(eci_trip, eci1_trip, places=6)

        # Also, testing non-integetic spin
        eci_doub = run_real_csf_solver((2, 1), 2)
        eci1_doubt = run_cplx_csf_solver((2, 1), 2)
        eci_quart = run_real_csf_solver((3, 0), 4)
        eci1_quart = run_cplx_csf_solver((3, 0),4)
        
        self.assertAlmostEqual(eci_doub, eci1_doubt, places=6)
        self.assertAlmostEqual(eci_quart, eci1_quart, places=6)

if __name__ == "__main__":
    unittest.main()
