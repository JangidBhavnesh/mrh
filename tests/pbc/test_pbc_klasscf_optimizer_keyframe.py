#!/usr/bin/env python

import sys
import unittest
from types import SimpleNamespace

import numpy as np
from pyscf import lib

from mrh.my_pyscf.pbc.mcscf import klasscf


class FakeFCIBox:
    def __init__(self, energy):
        self.energy = energy
        self.calls = []

    def kernel(self, h1, h2, norb, nelec, **kwargs):
        self.calls.append((h1, h2, norb, nelec, kwargs))
        return self.energy, kwargs["ci0"]


class KnownValuesKLASSCFKeyframe(unittest.TestCase):

    def test_missing_ci_guess_detects_nested_none(self):
        self.assertTrue(klasscf._ci_guess_is_missing(None))
        self.assertTrue(klasscf._ci_guess_is_missing([None]))
        self.assertTrue(klasscf._ci_guess_is_missing([[None]]))
        self.assertFalse(
            klasscf._ci_guess_is_missing([[np.array([1.0])]])
        )

    def test_keyframe_densities_follow_periodic_las_pipeline(self):
        calls = []
        casdm1frs = [np.ones((1, 2, 1, 1))]
        casdm1s_sub = [np.ones((2, 1, 1))]
        dm1s_kpts = np.ones((2, 1, 1, 1))
        veff_kpts = np.full((2, 1, 1, 1), 2.0)

        class FakeLAS:
            _scf = SimpleNamespace(cell=object())

            def states_make_casdm1s_sub(self, ci=None):
                calls.append(("states", ci))
                return casdm1frs

            def make_casdm1s_sub(self, ci=None, casdm1frs=None):
                calls.append(("active", ci, casdm1frs))
                return casdm1s_sub

            def make_rdm1s(self, **kwargs):
                calls.append(("ao", kwargs))
                return dm1s_kpts

            def get_veff(self, cell, dm_kpts=None):
                calls.append(("veff", cell, dm_kpts))
                return veff_kpts

        mo = np.ones((1, 1, 1))
        ci = [[np.array([1.0])]]
        actual = klasscf._make_keyframe_densities(FakeLAS(), mo, ci)

        self.assertIs(actual[0], casdm1frs)
        self.assertIs(actual[1], casdm1s_sub)
        self.assertIs(actual[2], dm1s_kpts)
        self.assertIs(actual[3], veff_kpts)
        self.assertEqual([call[0] for call in calls], [
            "states", "active", "ao", "veff",
        ])

    def test_ci_cycle_solves_each_unfrozen_fragment_once(self):
        boxes = [FakeFCIBox(-0.4), FakeFCIBox(-0.3)]

        class FakeLAS:
            fciboxes = boxes
            ncas_sub = np.array([1, 1])
            nelecas_sub = np.array([(1, 0), (0, 1)])
            frozen_ci = None
            max_memory = 1000

            def h1e_for_las(self, **kwargs):
                return [
                    np.full((1, 2, 1, 1), 0.2),
                    np.full((1, 2, 1, 1), 0.3),
                ]

        ci0 = [[np.array([1.0])], [np.array([1.0])]]
        h2eff = np.arange(16.0).reshape((2, 2, 2, 2))
        energies, ci1 = klasscf.ci_cycle(
            FakeLAS(), np.ones((1, 1, 1)), ci0,
            np.ones((2, 1, 1, 1)), h2eff,
            [np.ones((1, 2, 1, 1))] * 2,
            lib.logger.Logger(sys.stdout, lib.logger.QUIET),
        )

        np.testing.assert_allclose(energies, [-0.4, -0.3])
        self.assertEqual(ci1, ci0)
        self.assertEqual([len(box.calls) for box in boxes], [1, 1])
        np.testing.assert_allclose(
            boxes[0].calls[0][1], h2eff[:1, :1, :1, :1],
        )
        np.testing.assert_allclose(
            boxes[1].calls[0][1], h2eff[1:, 1:, 1:, 1:],
        )

    def test_ci_cycle_preserves_frozen_fragment(self):
        boxes = [FakeFCIBox(-0.4), FakeFCIBox(-0.3)]

        class FakeLAS:
            fciboxes = boxes
            ncas_sub = np.array([1, 1])
            nelecas_sub = np.array([(1, 0), (0, 1)])
            frozen_ci = [1]
            max_memory = 1000

            def h1e_for_las(self, **kwargs):
                return [np.zeros((1, 2, 1, 1))] * 2

        ci0 = [[np.array([1.0])], [np.array([2.0])]]
        energies, ci1 = klasscf.ci_cycle(
            FakeLAS(), np.ones((1, 1, 1)), ci0,
            np.ones((2, 1, 1, 1)), np.zeros((2, 2, 2, 2)),
            [np.ones((1, 2, 1, 1))] * 2,
            lib.logger.Logger(sys.stdout, lib.logger.QUIET),
        )

        np.testing.assert_allclose(energies, [-0.4, 0.0])
        self.assertEqual([len(box.calls) for box in boxes], [1, 0])
        self.assertIs(ci1[1], ci0[1])


if __name__ == "__main__":
    unittest.main()
