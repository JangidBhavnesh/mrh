import unittest

import numpy as np

from mrh.my_pyscf.pbc.mcscf.klas_ao2mo import _ERIS
from mrh.my_pyscf.pbc.mcscf.klasscf import get_grad_orb
from mrh.my_pyscf.pbc.mcscf.klasci import (
    PBCLASCINoSymm,
    PBCLASCITransSymm,
)


class _FakeSCF:
    cell = object()


class _FakeKLAS:
    _scf = _FakeSCF()
    kpts = np.zeros((1, 3))
    mo_coeff = np.ones((1, 1, 1), dtype=np.complex128)
    ci = [[np.ones(1)]]
    ncore = 0
    ncas = 1


class KnownValues(unittest.TestCase):

    def test_is_registered(self):
        for cls in (PBCLASCINoSymm, PBCLASCITransSymm):
            self.assertIs(cls._klasscf_eris, _ERIS)
            self.assertIs(cls.get_grad_orb, get_grad_orb)

    def test_rejects_inconsistent_paaa_shape(self):
        with self.assertRaisesRegex(ValueError, "h2eff_sub has shape"):
            get_grad_orb(
                _FakeKLAS(), h2eff_sub=np.zeros((1,)),
                dm1s_kpts=np.zeros((2, 1, 1, 1)),
                veff_kpts=np.zeros((2, 1, 1, 1)),
            )


if __name__ == "__main__":
    unittest.main()
