import unittest

import numpy as np
from pyscf.pbc import dft, gto, scf

from mrh.my_pyscf.pbc import mcpdft
from mrh.my_pyscf.pbc.mcpdft.laspdft import _LASPDFT


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.a = np.eye(3) * 8
        cell.atom = "H 0 0 0; H 1.4 0 0"
        cell.unit = "Bohr"
        cell.basis = "6-31g"
        cell.precision = 1e-8
        cell.verbose = 0
        cell.build()

        mf = scf.RHF(cell).density_fit()
        mf.exxdiv = None
        mf.conv_tol = 1e-10
        mf.kernel()
        cls.mf = mf

    def test_gamma_laspdft(self):
        mc = mcpdft.LASSCF(
            self.mf,
            "tPBE",
            (1, 1),
            (1, 1),
            spin_sub=(2, 2),
            grids_level=1,
        )

        self.assertIsInstance(mc, _LASPDFT)
        self.assertIsInstance(mc.grids, dft.gen_grid.BeckeGrids)

        mo = mc.localize_init_guess(([0], [1]))
        mc.kernel(mo)

        self.assertTrue(mc.converged)
        self.assertAlmostEqual(mc.e_tot, -0.7389379476115105, 7)

    def test_legacy_laspdft_dispatch(self):
        from mrh.my_pyscf import mcpdft as legacy_mcpdft

        mc = legacy_mcpdft.LASSCF(
            self.mf,
            "tPBE",
            (1, 1),
            (1, 1),
            spin_sub=(2, 2),
            grids_level=1,
        )

        self.assertIsInstance(mc, _LASPDFT)

    def test_legacy_mcpdft_dispatch(self):
        from mrh.my_pyscf import mcpdft as legacy_mcpdft
        from mrh.my_pyscf.pbc.mcpdft.mcpdft import _MCPDFT

        mc = legacy_mcpdft.CASCI(self.mf, "tPBE", 1, 2)

        self.assertIsInstance(mc, _MCPDFT)

    def test_legacy_periodic_imports(self):
        from mrh.my_pyscf.mcpdft import otfnalperiodic as legacy
        from mrh.my_pyscf.pbc.mcpdft import otfnalperiodic as periodic

        self.assertIs(legacy.otfnalperiodic, periodic.otfnalperiodic_gamma)
        self.assertIs(legacy._get_transfnal, periodic.get_pbc_otfnal_gamma)
        self.assertIs(legacy.sanity_check_for_df, periodic._get_ks_obj)


if __name__ == "__main__":
    unittest.main()
