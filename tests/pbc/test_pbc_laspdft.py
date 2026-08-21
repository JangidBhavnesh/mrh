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
        cell.basis = "sto-3g"
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
        self.assertAlmostEqual(mc.e_tot, -0.5329013391934931, 7)


if __name__ == "__main__":
    unittest.main()
