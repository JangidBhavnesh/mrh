   
#!/bin/bash
import unittest
import numpy as np
from ase import Atoms
from functools import reduce

from pyscf import ao2mo, fci, gto, scf
from pyscf.csf_fci import csf_solver

from mrh.my_pyscf.pbc.fci import direct_spin1_cplx
from mrh.my_pyscf.pbc.fci import csf_solver as csf_solver_cplx

# Let me set up the mol once and use it for all the tests.
mol = mf = h0 = h1 = h2 = h0cplx = h1cplx = h2cplx = None

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
    # Test-0: CSFSolver vs CSFSolver_cplx for Hermitian Hamiltonian with
    # small imaginary noise. The energies should match within a tight tolerance.
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

    # Test-1: CSFSolver vs CSFSolver_cplx for Hermitian Hamiltonian with
    # small imaginary noise. The energies should match within a tight tolerance.
    def test_vanilla_csf_solver_cplx_solvers(self):
        nelec = (5, 5)
        real_cisolver = csf_solver (mol, smult=1)
        eci = real_cisolver.kernel (h1, h2, norb, nelec, ecore=h0)[0]
        
        cplx_cisolver = csf_solver_cplx(mol, smult=1)
        cplx_cisolver.davidson_only = False
        eci1, civec1 = cplx_cisolver.kernel (h1cplx, h2cplx, norb, nelec, ecore=h0cplx)
        norm_ci1_dev = 1 - np.linalg.norm(civec1)

        cplx_cisolver = csf_solver_cplx(mol, smult=1)
        cplx_cisolver.davidson_only = True
        cplx_cisolver.pspace_size = 10
        eci2, civec2 = cplx_cisolver.kernel (h1cplx, h2cplx, norb, nelec, ecore=h0cplx)
        norm_ci2_dev = 1 - np.linalg.norm(civec2)

        self.assertAlmostEqual(eci1, eci, places=6)
        self.assertAlmostEqual(eci2, eci, places=6)
        self.assertAlmostEqual(norm_ci1_dev, 1e-7, places=6)
        self.assertAlmostEqual(norm_ci2_dev, 1e-7, places=6)

    # Test-2: CSFSolver vs CSFSolver_cplx for Hermitian Hamiltonian with
    # small imaginary noise. The energies should match within a tight tolerance.
    def test_vanilla_csf_solver_cplx_spin_states(self):
        nelec = (5, 5)
        real_cisolver = csf_solver (mol, smult=1)
        eci = real_cisolver.kernel (h1, h2, norb, nelec, ecore=h0)[0]
        
        cplx_cisolver = csf_solver_cplx(mol, smult=1)
        cplx_cisolver.davidson_only = False
        eci1, civec1 = cplx_cisolver.kernel (h1cplx, h2cplx, norb, nelec, ecore=h0cplx)
        norm_ci1_dev = 1 - np.linalg.norm(civec1)

        cplx_cisolver = csf_solver_cplx(mol, smult=1)
        cplx_cisolver.davidson_only = True
        cplx_cisolver.pspace_size = 10
        eci2, civec2 = cplx_cisolver.kernel (h1cplx, h2cplx, norb, nelec, ecore=h0cplx)
        norm_ci2_dev = 1 - np.linalg.norm(civec2)

        self.assertAlmostEqual(eci1, eci, places=6)
        self.assertAlmostEqual(eci2, eci, places=6)
        self.assertAlmostEqual(norm_ci1_dev, 1e-7, places=6)
        self.assertAlmostEqual(norm_ci2_dev, 1e-7, places=6)


if __name__ == "__main__":
    unittest.main()