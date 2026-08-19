"""Tests for momentum-resolved periodic CASCI-PDFT."""

import unittest
from types import SimpleNamespace

import numpy as np

from mrh.my_pyscf.pbc.fci import direct_spin1_kfci
from mrh.my_pyscf.pbc.mcpdft import kmcpdft_helper
from mrh.my_pyscf.pbc.mcpdft._dms import dm2_cumulant_complex


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


def make_kconserv(nkpts):
    kconserv = np.empty((nkpts, nkpts, nkpts), dtype=int)
    for k1 in range(nkpts):
        for k2 in range(nkpts):
            for k3 in range(nkpts):
                kconserv[k1, k2, k3] = (k1 - k2 + k3) % nkpts
    return kconserv


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


class KCASPDFTKBlockTests(unittest.TestCase):

    def test_casdm1s_to_kpts_extracts_diagonal_k_blocks(self):
        nkpts, ncas = 3, 2
        ncastot = nkpts * ncas
        casdm1s = np.zeros((2, ncastot, ncastot), dtype=complex)
        expected = np.empty((2, nkpts, ncas, ncas), dtype=complex)
        for spin in range(2):
            for k in range(nkpts):
                block = np.arange(4).reshape(2, 2)
                block = block + 10 * k + 100 * spin + 1j * (spin + k)
                p0 = k * ncas
                casdm1s[spin, p0:p0 + ncas, p0:p0 + ncas] = block
                expected[spin, k] = block

        result = kmcpdft_helper.casdm1s_to_kpts(
            casdm1s, nkpts, ncas,
        )
        np.testing.assert_allclose(result, expected)

    def test_casdm1s_to_kpts_rejects_forbidden_blocks(self):
        casdm1s = np.zeros((2, 2, 2))
        casdm1s[0, 0, 1] = 1e-3

        with self.assertRaisesRegex(ValueError, "momentum-forbidden"):
            kmcpdft_helper.casdm1s_to_kpts(
                casdm1s, nkpts=2, ncas=1, momentum_tol=1e-8,
            )

        result = kmcpdft_helper.casdm1s_to_kpts(
            casdm1s, nkpts=2, ncas=1, momentum_tol=None,
        )
        np.testing.assert_allclose(result, 0.0)

    def test_cascm2_to_kpts_uses_kconserv(self):
        nkpts, ncas = 3, 1
        kconserv = make_kconserv(nkpts)
        cascm2 = np.zeros((nkpts,) * 4, dtype=complex)
        expected = np.empty((nkpts, nkpts, nkpts, 1, 1, 1, 1),
                            dtype=complex)
        for k1 in range(nkpts):
            for k2 in range(nkpts):
                for k3 in range(nkpts):
                    k4 = kconserv[k1, k2, k3]
                    value = k1 + 10 * k2 + 100 * k3 + 0.5j
                    cascm2[k1, k2, k3, k4] = value
                    expected[k1, k2, k3, 0, 0, 0, 0] = value

        result = kmcpdft_helper.cascm2_to_kpts(
            cascm2, nkpts, ncas, kconserv,
        )
        np.testing.assert_allclose(result, expected)

    def test_cascm2_to_kpts_rejects_forbidden_blocks(self):
        nkpts = 2
        kconserv = make_kconserv(nkpts)
        cascm2 = np.zeros((nkpts,) * 4)
        forbidden_k4 = 1 - kconserv[0, 0, 0]
        cascm2[0, 0, 0, forbidden_k4] = 1e-4

        with self.assertRaisesRegex(ValueError, "momentum-forbidden"):
            kmcpdft_helper.cascm2_to_kpts(
                cascm2, nkpts, 1, kconserv,
            )

    def test_make_kcas_rdms_kpts_builds_complex_cumulant(self):
        nkpts, ncas = 2, 1
        ncastot = nkpts * ncas
        kconserv = make_kconserv(nkpts)
        casdm1s = np.zeros((2, ncastot, ncastot), dtype=complex)
        casdm1s[0] = np.diag([0.8 + 0.0j, 0.2 + 0.0j])
        casdm1s[1] = np.diag([0.3 + 0.0j, 0.7 + 0.0j])
        casdm2 = np.zeros((ncastot,) * 4, dtype=complex)

        dm1s_kpts, cascm2_kpts = kmcpdft_helper.make_kcas_rdms_kpts(
            casdm1s, casdm2, nkpts, ncas, kconserv,
        )

        cumulant = dm2_cumulant_complex(casdm2, casdm1s)
        expected_cumulant = kmcpdft_helper.cascm2_to_kpts(
            cumulant, nkpts, ncas, kconserv,
        )
        np.testing.assert_allclose(
            dm1s_kpts[:, :, 0, 0], [[0.8, 0.2], [0.3, 0.7]],
        )
        np.testing.assert_allclose(cascm2_kpts, expected_cumulant)

    def test_invalid_kconserv_is_rejected(self):
        cascm2 = np.zeros((2,) * 4)
        with self.assertRaisesRegex(ValueError, "kconserv shape"):
            kmcpdft_helper.cascm2_to_kpts(
                cascm2, 2, 1, np.zeros((2, 2), dtype=int),
            )
        bad_kconserv = np.zeros((2, 2, 2), dtype=int)
        bad_kconserv[0, 0, 0] = 2
        with self.assertRaisesRegex(ValueError, "indices"):
            kmcpdft_helper.cascm2_to_kpts(
                cascm2, 2, 1, bad_kconserv,
            )

    def test_kfci_rdms_obey_momentum_block_layout(self):
        nkpts, ncas = 2, 2
        norb = nkpts * ncas
        nelec = (1, 1)
        kconserv = make_kconserv(nkpts)
        rng = np.random.default_rng(19)

        for target_k in range(nkpts):
            with self.subTest(target_k=target_k):
                solver = direct_spin1_kfci.FCISolver(
                    nkpts=nkpts, target_k=target_k,
                )
                sector_size = direct_spin1_kfci.sector_size(
                    norb, nelec, nkpts, target_k,
                )
                ci = (
                    rng.normal(size=sector_size)
                    + 1j * rng.normal(size=sector_size)
                )
                ci /= np.linalg.norm(ci)

                casdm1s = solver.make_rdm1s(ci, norb, nelec)
                _, casdm2 = solver.make_rdm12(ci, norb, nelec)
                casdm1s_kpts, cascm2_kpts = \
                    kmcpdft_helper.make_kcas_rdms_kpts(
                        casdm1s, casdm2, nkpts, ncas, kconserv,
                    )

                self.assertEqual(
                    casdm1s_kpts.shape, (2, nkpts, ncas, ncas),
                )
                self.assertEqual(
                    cascm2_kpts.shape,
                    (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas),
                )
                self.assertAlmostEqual(np.trace(casdm1s[0]).real, 1.0)
                self.assertAlmostEqual(np.trace(casdm1s[1]).real, 1.0)


if __name__ == "__main__":
    unittest.main()
