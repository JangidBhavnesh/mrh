"""Tests for momentum-resolved periodic CASCI-PDFT."""

import unittest
from types import SimpleNamespace

import numpy as np

from mrh.my_pyscf.pbc.mcpdft import kmcpdft_helper


class RecordingSolver:

    def __init__(self, nroots=1):
        self.nroots = nroots
        self.calls = []

    def make_rdm1s(self, ci, norb, nelec, **kwargs):
        self.calls.append(("make_rdm1s", ci, norb, nelec, kwargs))
        return np.stack((np.eye(norb), 2.0 * np.eye(norb)))

    def make_rdm2(self, ci, norb, nelec, **kwargs):
        self.calls.append(("make_rdm2", ci, norb, nelec, kwargs))
        return np.zeros((norb,) * 4)


def make_mc(solver, target_k=0, nkpts=3, ncas=2, nelecas=(1, 1)):
    return SimpleNamespace(
        nkpts=nkpts,
        ncas=ncas,
        nelecas=nelecas,
        target_k=target_k,
        fcisolver=solver,
    )


class KCASPDFTRDMTests(unittest.TestCase):

    def test_make_one_casdm1s_passes_target_sector(self):
        solver = RecordingSolver()
        mc = make_mc(solver, target_k=5)

        casdm1s = kmcpdft_helper.make_one_casdm1s_kcas(mc, "ci")

        self.assertEqual(casdm1s.shape, (2, 6, 6))
        self.assertEqual(
            solver.calls,
            [("make_rdm1s", "ci", 6, (3, 3),
              {"nkpts": 3, "target_k": 2})],
        )

    def test_make_one_casdm1s_selects_requested_root(self):
        solver = RecordingSolver(nroots=2)
        mc = make_mc(solver, target_k=1, nkpts=2, ncas=1)

        kmcpdft_helper.make_one_casdm1s_kcas(
            mc, ["root-0", "root-1"], state=1,
        )

        _, ci, norb, nelec, kwargs = solver.calls[0]
        self.assertEqual(ci, "root-1")
        self.assertEqual(norb, 2)
        self.assertEqual(nelec, (2, 2))
        self.assertEqual(kwargs, {"nkpts": 2, "target_k": 1})

    def test_make_one_casdm2_passes_target_sector(self):
        solver = RecordingSolver()
        mc = make_mc(solver, target_k=-1)

        casdm2 = kmcpdft_helper.make_one_casdm2_kcas(mc, "ci")

        self.assertEqual(casdm2.shape, (6, 6, 6, 6))
        self.assertEqual(
            solver.calls,
            [("make_rdm2", "ci", 6, (3, 3),
              {"nkpts": 3, "target_k": 2})],
        )

    def test_make_one_casdm2_falls_back_to_make_rdm12(self):
        calls = []

        class RDM12Solver:
            nroots = 1

            def make_rdm12(self, ci, norb, nelec, **kwargs):
                calls.append((ci, norb, nelec, kwargs))
                return np.eye(norb), np.zeros((norb,) * 4)

        mc = make_mc(RDM12Solver(), target_k=0, nkpts=2, ncas=1)
        casdm2 = kmcpdft_helper.make_one_casdm2_kcas(mc, "ci")

        self.assertEqual(casdm2.shape, (2, 2, 2, 2))
        self.assertEqual(
            calls,
            [("ci", 2, (2, 2), {"nkpts": 2, "target_k": 0})],
        )

    def test_invalid_target_k_is_rejected(self):
        mc = make_mc(RecordingSolver(), target_k=None)
        with self.assertRaisesRegex(ValueError, "target_k must be an integer"):
            kmcpdft_helper.make_one_casdm1s_kcas(mc, "ci")

    def test_invalid_rdm_shapes_are_rejected(self):
        class BadShapeSolver(RecordingSolver):
            def make_rdm1s(self, ci, norb, nelec, **kwargs):
                return np.zeros((norb, norb))

            def make_rdm2(self, ci, norb, nelec, **kwargs):
                return np.zeros((norb, norb))

        mc = make_mc(BadShapeSolver(), nkpts=2, ncas=1)
        with self.assertRaisesRegex(ValueError, "1-RDM shape"):
            kmcpdft_helper.make_one_casdm1s_kcas(mc, "ci")
        with self.assertRaisesRegex(ValueError, "2-RDM shape"):
            kmcpdft_helper.make_one_casdm2_kcas(mc, "ci")


if __name__ == "__main__":
    unittest.main()
