import unittest

import numpy as np

from pyscf import lib
from pyscf.pbc import scf
from pyscf.pbc import gto as pgto

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.fci import csf_solver

# Unit test: Orbital gradient for KHF 

class KnownValues(unittest.TestCase):
    
    def test_kmf_kmc_orb_grad(self):
        cell = pgto.Cell()
        cell.a = np.diag([5.0, 10.0, 10.0])
        cell.atom = '''
        Be 0.0 0.0 0.0
        Be 2.0 0.0 0.0
        '''
        cell.basis = '6-31G'
        cell.unit = 'Angstrom'
        cell.max_memory = 100000
        cell.ke_cutoff = 100
        cell.precision = 1e-12
        cell.verbose = lib.logger.INFO
        cell.build()

        kmesh1D = [3, 1, 1]

        kpts = cell.make_kpts(kmesh1D, wrap_around=True)

        kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
        kmf.exxdiv = None
        kmf.conv_tol = 1e-1
        kmf.kernel()

        mo_ref = kmf.mo_coeff.copy()

        kmc = mcscf.CASSCF(kmf, 4, (4,4))
        kmc.max_cycle_macro = 0
        kmc.fcisolver = csf_solver(cell, smult=1)
        kmc.kernel(mo_ref)

        # Now compute the gradients
        kmf_orb_grad = kmf.get_grad(mo_ref, kmf.mo_occ)
        kmc_orb_grad = kmc.get_grad(mo_coeff=mo_ref)

        assert np.allclose(kmf_orb_grad, kmc_orb_grad, atol=1e-7)

if __name__ == "__main__":
    # Orbital gradient test for k-CASSCF.
    unittest.main()