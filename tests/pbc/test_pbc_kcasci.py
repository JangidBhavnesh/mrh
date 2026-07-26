import unittest
import numpy as np

from pyscf.pbc import scf
from pyscf.pbc import gto as pgto

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import kcasci
from mrh.my_pyscf.pbc.fci import direct_spin1_cplx


class KnownValues(unittest.TestCase):

    def test_charged_active_nelecas(self):
        self.assertEqual(
            kcasci.charged_active_nelecas(2, 8, 2, 0, charge=1),
            (8, 7))
        self.assertEqual(
            kcasci.charged_active_nelecas(2, 8, 2, 0, charge=-1),
            (9, 8))
        self.assertEqual(
            kcasci.charged_active_nelecas(2, 8, (1, 1), 0,
                                          charge=1, spin=-1),
            (7, 8))

    def test_charged_band_energies(self):
        results = [
            {'target_k': 0, 'charge': 1, 'nkpts': 2,
             'e_tot': np.asarray([-1.0, -0.8])},
            {'target_k': 1, 'charge': 1, 'nkpts': 2,
             'e_tot': np.asarray([-0.9, -0.7])},
        ]
        kpts = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        bands = kcasci.charged_band_energies(
            results, reference_energy=-1.2, root=1, kpts=kpts)

        self.assertEqual([b['target_k'] for b in bands], [0, 1])
        self.assertIn('hole_momentum', bands[0])
        self.assertTrue(np.allclose(bands[1]['hole_momentum'], kpts[1]))
        self.assertAlmostEqual(bands[0]['energy'], 0.8)
        self.assertAlmostEqual(bands[1]['energy'], 1.0)

    def test_kcasci_target_k0_vs_casci(self):
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

        mo_coeff = np.asarray(kmf.mo_coeff)

        kmc_default = mcscf.KCASCI(kmf, 2, 2)
        self.assertIsInstance(kmc_default, kcasci.PBCKCASCI)
        self.assertNotIsInstance(kmc_default, kcasci.ChargedKCASCI)
        self.assertEqual(kmc_default.target_k, 0)
        kmc_charged = mcscf.KCASCI(kmf, 2, 2, charge=1)
        self.assertIsInstance(kmc_charged, kcasci.ChargedKCASCI)
        self.assertIsNone(kmc_charged.target_k)
        self.assertFalse(hasattr(mcscf, 'ChargedKCASCI'))

        kmc_ref = mcscf.CASCI(kmf, 2, 2)
        kmc_ref.kmesh = kmesh
        kmc_ref.fcisolver = direct_spin1_cplx.FCISolver(cell)
        kmc_ref.fcisolver.verbose = 0
        e_ref = kmc_ref.kernel(mo_coeff)[0]

        kmc = mcscf.KCASCI(kmf, 2, 2, target_k=0)
        kmc.kmesh = kmesh
        kmc.verbose = 0
        kmc.fcisolver.verbose = 0
        e_test = kmc.kernel(mo_coeff)[0]

        self.assertTrue(np.allclose(e_test, e_ref, atol=1e-10, rtol=1e-10))


if __name__ == "__main__":
    unittest.main()
