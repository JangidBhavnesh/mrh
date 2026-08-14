import unittest
from unittest import mock

import numpy as np

from mrh.my_pyscf.pbc.util import pbcmolden

# Author: Bhavnesh jangid

'''
Test cases for the pbcmolden module.
'''

class FakeCell:
    nelectron = 2
    def nao_nr(self):
        return 3

class FakeKMF:
    def __init__(self):
        self.cell = FakeCell()
        self.kpts = np.zeros((2, 3))
        self.mo_coeff = np.arange(24, dtype=float).reshape(2, 3, 4)
        self.mo_energy = np.arange(8, dtype=float).reshape(2, 4)
        self.mo_occ = np.array([[2.0, 1.0, 0.0, 0.0],
                                [2.0, 1.0, 0.0, 0.0]])

class FakeMC:
    def __init__(self, kmf):
        self._scf = kmf
        self.mo_coeff = kmf.mo_coeff
        self.mo_energy = kmf.mo_energy
        self.mo_occ = kmf.mo_occ
        self.kmesh = (2, 1, 1)
        self.ncore = 1
        self.ncas = 2


class KnownValues(unittest.TestCase):

    def test_from_mo_active_k2r(self):
        '''Test active orbitals in the k-to-R basis.'''
        kmf = FakeKMF()
        coeff_R = np.ones((6, 4), dtype=complex)
        coeff_R += 0.25j

        with mock.patch.object(
                pbcmolden, "get_mo_coeff_k2R",
                return_value=("supercell", None, coeff_R, None)
             ) as get_k2r, \
             mock.patch.object(pbcmolden.molden, "from_mo") as write_molden, \
             mock.patch.object(pbcmolden.lib.logger, "warn") as warn:
            pbcmolden.from_mo(
                kmf, "active.molden", kmesh=(2, 1, 1),
                only_active=True, ncore=1, ncas=2,
                occ=kmf.mo_occ, ene=kmf.mo_energy,
            )

        mo_selected = get_k2r.call_args.args[1]
        np.testing.assert_allclose(mo_selected, kmf.mo_coeff[:, :, 1:3])
        self.assertEqual(get_k2r.call_args.args[2:4], (0, 2))
        warn.assert_called_once()

        args = write_molden.call_args.args
        self.assertEqual(args[:2], ("supercell", "active.molden"))
        np.testing.assert_allclose(args[2], coeff_R.real)
        np.testing.assert_allclose(
            write_molden.call_args.kwargs["occ"],
            kmf.mo_occ[:, 1:3].reshape(-1),
        )
        np.testing.assert_allclose(
            write_molden.call_args.kwargs["ene"],
            kmf.mo_energy[:, 1:3].reshape(-1),
        )

    def test_from_mo_active_wannier(self):
        '''Test active orbitals in the Wannier basis.'''
        kmf = FakeKMF()
        wannier_orb = np.arange(24, dtype=float).reshape(2, 3, 2, 2)

        with mock.patch.object(
                pbcmolden, "get_wannier_orbs",
                return_value=(wannier_orb, None, None)
             ) as get_wannier, \
             mock.patch.object(pbcmolden, "super_cell",
                               return_value="supercell"), \
             mock.patch.object(pbcmolden.molden, "from_mo") as write_molden:
            pbcmolden.from_mo(
                kmf, "wannier.molden", kmesh=(2, 1, 1), wannier=True,
                only_active=True, ncore=1, ncas=2,
                occ=kmf.mo_occ, ene=kmf.mo_energy,
            )

        mo_selected = get_wannier.call_args.args[2]
        np.testing.assert_allclose(mo_selected, kmf.mo_coeff[:, :, 1:3])
        np.testing.assert_allclose(
            write_molden.call_args.args[2], wannier_orb.reshape(6, 4),
        )
        np.testing.assert_allclose(
            write_molden.call_args.kwargs["occ"], [1.0, 0.0, 1.0, 0.0],
        )
        self.assertIsNone(write_molden.call_args.kwargs["ene"])

    def test_object_wrappers(self):
        '''Test wrapper defaults.'''
        kmf = FakeKMF()
        mc = FakeMC(kmf)

        with mock.patch.object(pbcmolden, "from_mo") as from_mo:
            pbcmolden.from_scf(kmf, "scf.molden", kmesh=(2, 1, 1))
            scf_kwargs = from_mo.call_args.kwargs
            self.assertFalse(scf_kwargs["wannier"])
            self.assertFalse(scf_kwargs["only_active"])

            pbcmolden.from_lasscf(mc, "las.molden")
            las_kwargs = from_mo.call_args.kwargs
            self.assertTrue(las_kwargs["wannier"])
            self.assertTrue(las_kwargs["only_active"])
            self.assertEqual((las_kwargs["ncore"], las_kwargs["ncas"]),
                             (1, 2))

            pbcmolden.from_kcasscf(mc, "casscf.molden")
            casscf_kwargs = from_mo.call_args.kwargs
            self.assertFalse(casscf_kwargs["wannier"])
            self.assertFalse(casscf_kwargs["only_active"])


if __name__ == "__main__":
    unittest.main()
