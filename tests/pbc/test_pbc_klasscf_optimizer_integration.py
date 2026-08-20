#!/usr/bin/env python

import unittest

import numpy as np
from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.mcscf import klasscf


class KnownValuesKLASSCFOptimizerIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.a = np.diag([4.0, 10.0, 10.0])
        cell.atom = "H 0 0 0; H 1.5 0 0"
        cell.basis = "sto-3g"
        cell.unit = "Angstrom"
        cell.precision = 1e-10
        cell.ke_cutoff = 20
        cell.verbose = lib.logger.QUIET
        cell.build()

        kmesh = (2, 1, 1)
        kpts = cell.make_kpts(kmesh, wrap_around=True)
        kmf = scf.KRHF(cell, kpts=kpts).density_fit()
        kmf.exxdiv = None
        kmf.max_cycle = 0
        kmf.kernel()

        mo_avas = avas.kernel(kmf, ["H 1s"], minao=cell.basis)[2]
        las = mcscf.KLASSCF(
            kmf, 2, (1, 1), kmesh=kmesh, trans_sym=False,
        )
        mo_guess = las.localize_init_guess(
            ["H 1s"], mo_coeff=mo_avas,
        )
        cls.las = las
        cls.mo_guess = mo_guess

    def test_public_optimizer_builds_physical_keyframe(self):
        self.las.max_cycle_macro = 0
        result = self.las.kernel(mo_coeff=self.mo_guess)
        e_tot, e_cas, ci, mo_coeff, mo_energy, h2eff, veff = result

        self.assertIsInstance(self.las, klasscf.PBCLASSCFNoSymm)
        self.assertTrue(np.isfinite(e_tot))
        self.assertTrue(np.all(np.isfinite(e_cas)))
        self.assertEqual(len(ci), self.las.nfrags)
        self.assertEqual(np.shape(mo_coeff), np.shape(self.mo_guess))
        self.assertEqual(np.shape(mo_energy), (
            self.las.nkpts, mo_coeff.shape[-1],
        ))
        ncastot = int(np.sum(self.las.ncas_sub))
        self.assertEqual(np.shape(h2eff), (ncastot,) * 4)
        self.assertEqual(np.shape(veff), (
            2, self.las.nkpts, mo_coeff.shape[-1], mo_coeff.shape[-1],
        ))


if __name__ == "__main__":
    unittest.main()
