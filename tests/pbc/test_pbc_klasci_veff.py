import unittest

import numpy as np

from mrh.my_pyscf.pbc.mcscf.klasci import PBCLASCINoSymm
from mrh.my_pyscf.pbc.mcscf.casci import PBCCASBASE


class _FakeLAS:

    def get_jk(self, cell, dm_kpts, **kwargs):
        size = np.prod(dm_kpts.shape)
        vj = np.arange(size, dtype=float).reshape(dm_kpts.shape)
        vk = vj + 3.0
        if dm_kpts.ndim == 4:
            vj = vj.sum(axis=0)
        return vj, vk


class _FakeSCF:

    def __init__(self, output_shape):
        self.output_shape = output_shape

    def get_jk(self, **kwargs):
        vj = np.zeros(self.output_shape)
        vk = np.zeros(self.output_shape)
        return vj, vk


class _FakeCAS:

    def __init__(self, output_shape):
        self._scf = _FakeSCF(output_shape)


class KnownValues(unittest.TestCase):

    def test_spin_separated_veff(self):
        las = _FakeLAS()
        dm1s = np.zeros((2, 2, 3, 3))
        vj, vk = las.get_jk(None, dm1s)

        veff = PBCLASCINoSymm.get_veff(las, dm_kpts=dm1s)

        self.assertEqual(veff.shape, dm1s.shape)
        np.testing.assert_allclose(veff, vj[None] - vk)

    def test_spin_summed_veff(self):
        las = _FakeLAS()
        dm = np.zeros((2, 3, 3))
        vj, vk = las.get_jk(None, dm)

        veff = PBCLASCINoSymm.get_veff(las, dm_kpts=dm)

        self.assertEqual(veff.shape, dm.shape)
        np.testing.assert_allclose(veff, vj - 0.5 * vk)

    def test_get_jk_sums_only_the_spin_axis_of_j(self):
        shape = (2, 2, 3, 3)
        cas = _FakeCAS(shape)
        dm1s = np.zeros(shape)

        vj, vk = PBCCASBASE.get_jk(cas, None, dm1s)

        self.assertEqual(vj.shape, shape[1:])
        self.assertEqual(vk.shape, shape)

    def test_get_jk_rejects_unexpected_shape(self):
        cas = _FakeCAS((3, 2, 3, 3))
        dm1s = np.zeros((2, 2, 3, 3))

        with self.assertRaisesRegex(RuntimeError, "Unexpected J shape"):
            PBCCASBASE.get_jk(cas, None, dm1s)


if __name__ == "__main__":
    unittest.main()
