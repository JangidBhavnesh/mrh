#!/usr/bin/env python

import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pyscf import lib

from mrh.my_pyscf.pbc.mcscf import klasscf


class FakeUGG:
    nvar_orb = 1
    nvar_tot = 1


class FakeHop:
    def __init__(self, gradient, curvature=2.0):
        self.gradient = np.asarray(gradient, dtype=np.complex128)
        self.curvature = curvature
        self.steps = []
        self.h1s = np.ones((2, 1, 1, 1))
        self.eri_cas = np.full((1, 1, 1, 1), curvature)
        self.veff_kpts = np.full((2, 1, 1, 1), curvature)

    def get_grad(self):
        return self.gradient

    def _matvec(self, vector):
        return self.curvature * np.asarray(vector)

    def update_mo_ci_eri(self, step, h2eff):
        self.steps.append(np.array(step, copy=True))
        return (
            np.array([[[step[0]]]]),
            [[np.array([1.0 + 0.0j])]],
            np.asarray(h2eff) + 1.0,
        )


class FakeKLASSCF:
    def __init__(self, hops, trust_radius=10.0):
        self.mo_coeff = np.zeros((1, 1, 1), dtype=np.complex128)
        self.ci = [[np.array([1.0 + 0.0j])]]
        self.conv_tol_grad = 1e-9
        self.max_cycle_macro = 1
        self.max_cycle_micro = 5
        self.min_cycle_macro = 0
        self.trust_radius = trust_radius
        self.weights = [1.0]
        self.nroots = 1
        self.nfrags = 1
        self.nkpts = 1
        self.ncas = 1
        self.ncore = 0
        self.verbose = lib.logger.QUIET
        self.stdout = sys.stdout
        self._scf = SimpleNamespace(cell=object())
        self.hops = list(hops)
        self.uggs = []
        self.hop_kwargs = []

    def get_h2cas(self, mo_coeff):
        return np.full((1, 1, 1, 1), 3.0)

    def states_make_casdm1s_sub(self, ci=None):
        return [np.ones((1, 2, 1, 1))]

    def make_casdm1s_sub(self, ci=None, casdm1frs=None):
        return [np.ones((2, 1, 1))]

    def make_rdm1s(self, mo_coeff=None, ci=None, casdm1s_sub=None):
        return np.ones((2, 1, 1, 1))

    def get_veff(self, cell, dm_kpts=None):
        return np.ones((2, 1, 1, 1))

    def get_ugg(self, mo_coeff=None, ci=None):
        ugg = FakeUGG()
        self.uggs.append(ugg)
        return ugg

    def get_hop(self, mo_coeff=None, ci=None, ugg=None, **kwargs):
        self.hop_kwargs.append(kwargs)
        return self.hops.pop(0)


def ci_cycle_result():
    """Return one local CI refresh for the fake driver."""
    return [0.0], [[np.array([1.0 + 0.0j])]]


def fixed_energy(energy):
    """Return fixed-CI energy data for one root."""
    return (
        energy, np.array([energy]), np.array([energy - 0.5]),
        [[np.array([0.0])]],
    )


class KnownValuesKLASSCFKernel(unittest.TestCase):

    def test_macro_driver_applies_the_complex_newton_step(self):
        # The first keyframe has H=2 and a complex gradient.  MINRES should
        # solve x=-g/2.  The second keyframe has zero gradient, so the macro
        # driver should report convergence after applying that one step.
        first_hop = FakeHop([0.4 + 0.2j], curvature=2.0)
        final_hop = FakeHop([0.0j], curvature=2.0)
        las = FakeKLASSCF([first_hop, final_hop])
        las.verbose = lib.logger.INFO
        las.stdout = io.StringIO()

        with patch.object(
                klasscf, "ci_cycle",
                side_effect=[ci_cycle_result(), ci_cycle_result()]), \
                patch.object(
                    klasscf, "_fixed_ci_energies",
                    side_effect=[fixed_energy(-1.0), fixed_energy(-1.1)],
                ):
            result = klasscf.kernel(las)

        self.assertTrue(result[0])
        self.assertEqual(len(first_hop.steps), 1)
        np.testing.assert_allclose(
            first_hop.steps[0], np.array([-0.2 - 0.1j]), atol=1e-12,
        )
        np.testing.assert_allclose(result[1], -1.1)
        self.assertEqual(len(las.uggs), 2)
        self.assertIsNot(las.uggs[0], las.uggs[1])
        np.testing.assert_allclose(las.hop_kwargs[0]["h2eff"], 3.0)
        np.testing.assert_allclose(las.hop_kwargs[1]["h2eff"], 4.0)
        self.assertIn("k-LASSCF micro 1", las.stdout.getvalue())

    def test_macro_driver_limits_a_large_step_to_the_trust_radius(self):
        # H=1 and g=10 give the raw Newton step x=-10.  A trust radius of
        # 0.25 must scale the accepted step before the orbital/CI update.
        first_hop = FakeHop([10.0 + 0.0j], curvature=1.0)
        final_hop = FakeHop([1.0 + 0.0j], curvature=1.0)
        las = FakeKLASSCF([first_hop, final_hop], trust_radius=0.25)

        with patch.object(
                klasscf, "ci_cycle",
                side_effect=[ci_cycle_result(), ci_cycle_result()]), \
                patch.object(
                    klasscf, "_fixed_ci_energies",
                    side_effect=[fixed_energy(-1.0), fixed_energy(-1.05)],
                ):
            result = klasscf.kernel(las)

        self.assertFalse(result[0])
        self.assertEqual(len(first_hop.steps), 1)
        np.testing.assert_allclose(first_hop.steps[0], [-0.25 + 0.0j])

    def test_ci_cycle_solves_each_fragment_once(self):
        # The synchronous LASSCF CI refresh diagonalizes each fragment once
        # in the environment built at the beginning of the keyframe.  It must
        # not invoke the outer product-state fixed-point solver.
        class FakeFCIBox:
            def __init__(self, energy):
                self.energy = energy
                self.calls = []

            def kernel(self, h1, h2, norb, nelec, **kwargs):
                self.calls.append((h1, h2, norb, nelec, kwargs))
                return self.energy, kwargs["ci0"]

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
        np.testing.assert_allclose(boxes[0].calls[0][1], h2eff[:1, :1, :1, :1])
        np.testing.assert_allclose(boxes[1].calls[0][1], h2eff[1:, 1:, 1:, 1:])


if __name__ == "__main__":
    unittest.main()
