#!/usr/bin/env python
import io
import unittest
from types import SimpleNamespace

import numpy as np

from pyscf import lib

from mrh.my_pyscf.pbc.mcscf import kcasci

# Author: Bhavnesh Jangid

"""
API tests for momentum-resolved periodic CASCI.
"""


class KCASCIAPITests(unittest.TestCase):
    """Test KCASCI tensor helpers, validation, and reporting APIs."""

    def test_adjust_h1eff_for_kfci(self):
        """Apply the k-FCI one-body correction without mutating its input."""
        nkpts = 2
        h1eff = np.arange(8, dtype=float).reshape(nkpts, 2, 2)
        h2eff = np.zeros((nkpts, nkpts, nkpts, 2, 2, 2, 2))
        h2eff[0, 0, 0, 0, 0, 0, 1] = 0.25
        h2eff[0, 1, 1, 1, 0, 0, 0] = 0.50
        h2eff[1, 0, 0, 0, 1, 1, 0] = 0.75

        expected = h1eff.copy()
        for kp in range(nkpts):
            for kq in range(nkpts):
                expected[kp] -= np.einsum("piis->ps", h2eff[kp, kq, kq])

        result = kcasci._adjust_h1eff_for_kfci(h1eff, h2eff)
        self.assertTrue(np.allclose(result, expected))
        self.assertFalse(np.shares_memory(result, h1eff))

    def test_make_casdm1_weighted_roots_and_target_k(self):
        """Build a weighted multiroot density in the requested momentum sector."""
        calls = []

        class RecordingSolver:
            def make_rdm1(self, ci, norb, nelec, **kwargs):
                calls.append((ci, norb, nelec, kwargs))
                return float(ci) * np.eye(norb)

        mc = SimpleNamespace(
            nkpts=3, ncas=2, nelecas=(1, 1), target_k=1,
            cell=SimpleNamespace(spin=0),
            fcisolver=RecordingSolver(), ci=None,
        )
        casdm1 = kcasci.make_casdm1(
            mc, [1.0, 3.0], stav_dm1=True, weights=[1.0, 3.0],
            target_k=2,
        )

        self.assertTrue(np.allclose(casdm1, 2.5 * np.eye(6)))
        self.assertEqual(len(calls), 2)
        for _, norb, nelec, kwargs in calls:
            self.assertEqual(norb, 6)
            self.assertEqual(nelec, (3, 3))
            self.assertEqual(kwargs, {"nkpts": 3, "target_k": 2})

    def test_make_casdm1_weight_validation(self):
        """Reject malformed state-average weights."""
        solver = SimpleNamespace(
            make_rdm1=lambda ci, norb, nelec, **kwargs: np.eye(norb),
        )
        mc = SimpleNamespace(
            nkpts=2, ncas=1, nelecas=(1, 0), target_k=0,
            cell=SimpleNamespace(spin=1),
            fcisolver=solver, ci=None,
        )

        with self.assertRaisesRegex(ValueError, "one value for each CI root"):
            kcasci.make_casdm1(mc, [1, 2], weights=[1])
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            kcasci.make_casdm1(mc, [1, 2], weights=[1, -1])
        with self.assertRaisesRegex(ValueError, "at least one"):
            kcasci.make_casdm1(mc, [1, 2], weights=[0, 0])
        with self.assertRaisesRegex(ValueError, "multiple CI roots"):
            kcasci.make_casdm1(mc, np.ones(1), weights=[1])

    def test_charged_active_electron_sectors(self):
        """Validate neutral, hole, particle, and explicit-spin electron sectors."""
        get_nelec = kcasci._get_nelecas_for_charged_kcasci
        self.assertEqual(get_nelec(2, 8, 2, 0, charge=0), (8, 8))
        self.assertEqual(get_nelec(2, 8, 2, 0, charge=1), (8, 7))
        self.assertEqual(get_nelec(2, 8, 2, 0, charge=-1), (9, 8))
        self.assertEqual(get_nelec(2, 8, (1, 1), 0, charge=1, spin=-1), (7, 8))

        with self.assertRaisesRegex(ValueError, "single-electron"):
            get_nelec(2, 8, 2, 0, charge=2)
        with self.assertRaisesRegex(ValueError, "charge must be an integer"):
            get_nelec(2, 8, 2, 0, charge=0.5)
        with self.assertRaisesRegex(ValueError, "inconsistent parity"):
            get_nelec(2, 8, 2, 0, charge=1, spin=0)
        with self.assertRaisesRegex(ValueError, "active electrons"):
            get_nelec(1, 1, 2, 0, charge=-1)
        with self.assertRaisesRegex(ValueError, "spin must be an integer"):
            get_nelec(2, 8, 2, 0, charge=1, spin=1.0)

    def test_charged_band_energy_conventions(self):
        """Map charged sectors to particle/hole momenta and energy conventions."""
        kpts = np.asarray([
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [-0.25, 0.0, 0.0],
        ])
        hole_results = [
            {"target_k": 0, "charge": 1, "nkpts": 3,
             "e_tot": np.asarray([-1.0, -0.8])},
            {"target_k": 1, "charge": 1, "nkpts": 3,
             "e_tot": np.asarray([-0.9, -0.7])},
            {"target_k": 2, "charge": 1, "nkpts": 3,
             "e_tot": np.asarray([-0.85, -0.65])},
        ]
        holes = kcasci.compute_band_energies(
            hole_results, reference_energy=-1.2, root=1, kpts=kpts,
        )
        self.assertEqual([band["momentum_index"] for band in holes], [0, 2, 1])
        self.assertTrue(np.allclose(
            [band["energy"] for band in holes], [-1.2, -1.5, -1.65]))
        self.assertTrue(all(band["kind"] == "hole" for band in holes))
        self.assertTrue(np.allclose(holes[1]["hole_momentum"], kpts[2]))
        all_holes = kcasci.compute_band_energies(
            hole_results, reference_energy=-1.2, kpts=kpts,
        )
        self.assertTrue(np.allclose(
            [band["energy"] for band in all_holes],
            [[-0.6, -1.2], [-0.9, -1.5], [-1.05, -1.65]],
        ))
        self.assertTrue(all(band["root"] is None for band in all_holes))

        particle_results = [
            {"target_k": 0, "charge": -1, "nkpts": 3,
             "e_tot": np.asarray([-1.1, -0.9])},
            {"target_k": 1, "charge": -1, "nkpts": 3,
             "e_tot": np.asarray([-1.0, -0.8])},
            {"target_k": 2, "charge": -1, "nkpts": 3,
             "e_tot": np.asarray([-0.95, -0.75])},
        ]
        particles = kcasci.compute_band_energies(
            particle_results, reference_energy=-1.2, root=1, kpts=kpts,
        )
        self.assertEqual([band["momentum_index"] for band in particles], [0, 1, 2])
        self.assertTrue(np.allclose(
            [band["energy"] for band in particles], [0.9, 1.2, 1.35]))
        self.assertTrue(all(band["kind"] == "particle" for band in particles))

        shifted = kcasci.compute_band_energies(
            hole_results, reference_energy=-1.2, root=1, kpts=kpts,
            reference_target_k=1,
        )
        self.assertEqual([band["momentum_index"] for band in shifted], [1, 0, 2])
        per_cell = kcasci.compute_band_energies(
            hole_results, reference_energy=-1.2, root=1, kpts=kpts,
            per_cell=True,
        )
        self.assertTrue(np.allclose(
            [band["energy"] for band in per_cell], [-0.4, -0.5, -0.55]))

    def test_charged_band_energy_validation(self):
        """Reject inconsistent charge metadata."""
        result = [{
            "target_k": 0, "charge": 1, "nkpts": 2,
            "e_tot": np.asarray([-1.0]),
        }]
        self.assertEqual(kcasci.compute_band_energies([], -1.2), [])
        with self.assertRaisesRegex(ValueError, r"charge \+1 or -1"):
            kcasci.compute_band_energies(result, -1.2, charge=0)

        mixed = result + [{
            "target_k": 1, "charge": -1, "nkpts": 2,
            "e_tot": -1.0,
        }]
        with self.assertRaisesRegex(ValueError, "inconsistent charges"):
            kcasci.compute_band_energies(mixed, -1.2, charge=1)

    def test_charged_finalize_forwards_result_target_k(self):
        """Pass each result's momentum sector to spin diagnostics."""
        calls = []

        class RecordingSolver:
            def spin_square(self, ci, norb, nelec, **kwargs):
                calls.append((ci, norb, nelec, kwargs))
                return 0.75, 2.0

        mc = object.__new__(kcasci.ChargedPBCKCASCI)
        mc.stdout = io.StringIO()
        mc.verbose = lib.logger.NOTE
        mc.fcisolver = RecordingSolver()
        mc.nkpts = 3
        mc.ncas = 2
        mc.charged_nelecastot = (3, 2)
        mc.charged_results = [
            {
                "target_k": target_k,
                "e_tot": np.asarray(-1.0),
                "e_cas": np.asarray(-0.5),
                "ci": np.asarray([target_k], dtype=complex),
            }
            for target_k in range(mc.nkpts)
        ]

        mc._finalize()
        self.assertEqual(len(calls), mc.nkpts)
        for target_k, (ci, norb, nelec, kwargs) in enumerate(calls):
            self.assertTrue(np.array_equal(ci, mc.charged_results[target_k]["ci"]))
            self.assertEqual(norb, mc.nkpts * mc.ncas)
            self.assertEqual(nelec, mc.charged_nelecastot)
            self.assertEqual(kwargs["nkpts"], mc.nkpts)
            self.assertEqual(kwargs["target_k"], target_k)


if __name__ == "__main__":
    unittest.main()
