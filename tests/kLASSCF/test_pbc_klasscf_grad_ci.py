import unittest

import numpy as np

from mrh.my_pyscf.pbc.mcscf.klasscf import get_grad_ci

# Author: Bhavnesh Jangid

"""Unit tests for the k-LASSCF CI-gradient builder.

Test-0: Evaluate the complex CI residual with explicitly supplied integrals.
Test-1: Build missing effective integrals and density intermediates.
Test-2: Reject an effective two-electron integral with an invalid shape.
"""

class _ResidualFCIBox:
    def __init__(self, test_case):
        self.test_case = test_case

    def states_gen_linkstr(self, norb, nelec, tril):
        self.test_case.assertFalse(tril)
        return "ordinary-links"

    def states_absorb_h1e(self, h1frs, h2, norb, nelec, fac):
        self.test_case.assertEqual(fac, 0.5)
        return [h1frs[0].sum(axis=0)]

    def states_contract_2e(
            self, hamiltonians, ci, norb, nelec, link_index=None):
        self.test_case.assertEqual(link_index, "ordinary-links")
        return [hamiltonians[0] @ ci[0]]


class _ResidualKLAS:
    mo_coeff = None
    nroots = 1
    ncas_sub = np.array([2])
    nelecas_sub = np.array([(1, 0)])

    def __init__(self, test_case):
        self.fciboxes = [_ResidualFCIBox(test_case)]


class GradientCITests(unittest.TestCase):

    def setUp(self):
        self.ci = [[np.array([1.0, 1.0j]) / np.sqrt(2.0)]]
        self.h1eff = np.array([[
            [[1.0, 0.4j], [-0.4j, 2.0]],
            [[0.5, -0.2], [-0.2, -0.3]],
        ]])
        self.h2eff = np.zeros((2, 2, 2, 2))

    def test_complex_ci_residual_does_not_require_hessian(self):
        klas = _ResidualKLAS(self)
        hamiltonian = self.h1eff[0, 0] + self.h1eff[0, 1]
        hc = hamiltonian @ self.ci[0][0]
        expected = 2.0 * (
            hc - np.vdot(self.ci[0][0], hc) * self.ci[0][0]
        )

        actual = get_grad_ci(
            klas, ci=self.ci, h1eff=[self.h1eff], h2eff=self.h2eff,
        )

        np.testing.assert_allclose(actual[0][0], expected)
        np.testing.assert_allclose(
            np.vdot(self.ci[0][0], actual[0][0]), 0.0, atol=1e-14,
        )

    def test_builds_missing_effective_integrals(self):
        klas = _ResidualKLAS(self)
        klas.mo_coeff = object()
        calls = []

        def get_h2cas(mo_coeff):
            calls.append(("get_h2cas", mo_coeff))
            return self.h2eff

        def states_make_casdm1s_sub(**kwargs):
            calls.append(("states_make_casdm1s_sub", kwargs))
            return "root-densities"

        def make_casdm1s_sub(**kwargs):
            calls.append(("make_casdm1s_sub", kwargs))
            return "state-density"

        def h1e_for_las(**kwargs):
            calls.append(("h1e_for_las", kwargs))
            return [self.h1eff]

        klas.get_h2cas = get_h2cas
        klas.states_make_casdm1s_sub = states_make_casdm1s_sub
        klas.make_casdm1s_sub = make_casdm1s_sub
        klas.h1e_for_las = h1e_for_las

        get_grad_ci(klas, ci=self.ci)

        self.assertEqual(
            [name for name, value in calls],
            [
                "get_h2cas",
                "states_make_casdm1s_sub",
                "make_casdm1s_sub",
                "h1e_for_las",
            ],
        )
        self.assertIs(calls[0][1], klas.mo_coeff)
        self.assertEqual(
            calls[-1][1]["casdm1frs"], "root-densities",
        )
        self.assertEqual(
            calls[-1][1]["casdm1s_sub"], "state-density",
        )

    def test_rejects_inconsistent_h2eff_shape(self):
        with self.assertRaisesRegex(ValueError, "h2eff has shape"):
            get_grad_ci(
                _ResidualKLAS(self), ci=self.ci,
                h1eff=[self.h1eff], h2eff=np.zeros((1, 1, 1, 1)),
            )


if __name__ == "__main__":
    unittest.main()
