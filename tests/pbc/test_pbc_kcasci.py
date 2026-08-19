#!/usr/bin/env python

"""Tests for neutral and charged momentum-resolved periodic CASCI."""

import io
import sys
import unittest
from types import SimpleNamespace

import numpy as np

from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.fci import direct_spin1_cplx
from mrh.my_pyscf.pbc.mcscf import kcasci


class KCASCIHelperTests(unittest.TestCase):

    def test_adjust_h1eff_for_kfci(self):
        nkpts = 2
        h1eff = np.arange(8, dtype=float).reshape(nkpts, 2, 2)
        h2eff = np.zeros((nkpts, nkpts, nkpts, 2, 2, 2, 2))
        h2eff[0, 0, 0, 0, 0, 0, 1] = 0.25
        h2eff[0, 1, 1, 1, 0, 0, 0] = 0.50
        h2eff[1, 0, 0, 0, 1, 1, 0] = 0.75

        expected = h1eff.copy()
        for kp in range(nkpts):
            for kq in range(nkpts):
                expected[kp] -= np.einsum(
                    "piis->ps", h2eff[kp, kq, kq],
                )

        result = kcasci._adjust_h1eff_for_kfci(h1eff, h2eff)
        self.assertTrue(np.allclose(result, expected))
        self.assertFalse(np.shares_memory(result, h1eff))

    def test_make_casdm1_weighted_roots_and_target_k(self):
        calls = []

        class RecordingSolver:
            def make_rdm1(self, ci, norb, nelec, **kwargs):
                calls.append((ci, norb, nelec, kwargs))
                return float(ci) * np.eye(norb)

        mc = SimpleNamespace(
            nkpts=3,
            ncas=2,
            nelecas=(1, 1),
            target_k=1,
            cell=SimpleNamespace(spin=0),
            fcisolver=RecordingSolver(),
            ci=None,
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
        solver = SimpleNamespace(
            make_rdm1=lambda ci, norb, nelec, **kwargs: np.eye(norb),
        )
        mc = SimpleNamespace(
            nkpts=2,
            ncas=1,
            nelecas=(1, 0),
            target_k=0,
            cell=SimpleNamespace(spin=1),
            fcisolver=solver,
            ci=None,
        )

        with self.assertRaisesRegex(ValueError, "one value for each CI root"):
            kcasci.make_casdm1(mc, [1, 2], weights=[1])
        with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
            kcasci.make_casdm1(mc, [1, 2], weights=[1, -1])
        with self.assertRaisesRegex(ValueError, "at least one"):
            kcasci.make_casdm1(mc, [1, 2], weights=[0, 0])
        with self.assertRaisesRegex(ValueError, "multiple CI roots"):
            kcasci.make_casdm1(mc, np.ones(1), weights=[1])

    def test_fock_and_canonicalize_core_virtual_subspaces(self):
        nkpts = 2
        mo_coeff = np.tile(np.eye(5, dtype=complex), (nkpts, 1, 1))
        fock = np.asarray([
            [1.0, 0.2, 0.0, 0.0, 0.0],
            [0.2, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 0.3],
            [0.0, 0.0, 0.0, 0.3, 5.0],
        ], dtype=complex)
        hcore = np.tile(fock, (nkpts, 1, 1))
        casdm1 = np.diag([0.5, 1.5])
        seen = {}

        def get_veff(cell, dm_k, **kwargs):
            seen["dm_k"] = dm_k.copy()
            return np.zeros_like(dm_k)

        mc = SimpleNamespace(
            nkpts=nkpts,
            ncore=2,
            ncas=1,
            cell=SimpleNamespace(),
            _scf=SimpleNamespace(kpts=np.zeros((nkpts, 3))),
            mo_coeff=mo_coeff,
            ci=np.ones(1),
            frozen=None,
            verbose=0,
            stdout=sys.stdout,
            get_hcore=lambda: hcore,
            get_veff=get_veff,
            _eig=lambda matrix, *args: np.linalg.eigh(matrix),
        )

        fock_ao = kcasci.get_fock(mc, casdm1=casdm1)
        self.assertTrue(np.allclose(fock_ao, hcore))
        self.assertTrue(np.allclose(
            seen["dm_k"][0], np.diag([2.0, 2.0, 0.5, 0.0, 0.0]),
        ))
        self.assertTrue(np.allclose(
            seen["dm_k"][1], np.diag([2.0, 2.0, 1.5, 0.0, 0.0]),
        ))

        mo1, ci1, mo_energy = kcasci.canonicalize(
            mc, casdm1=casdm1,
        )
        self.assertIs(ci1, mc.ci)
        self.assertEqual(np.asarray(mo_energy).shape, (nkpts, 5))
        self.assertTrue(np.allclose(mo1[:, :, 2], mo_coeff[:, :, 2]))
        for k in range(nkpts):
            self.assertTrue(np.allclose(mo1[k].conj().T @ mo1[k], np.eye(5)))
            transformed = mo1[k].conj().T @ fock_ao[k] @ mo1[k]
            self.assertTrue(np.allclose(
                transformed[:2, :2], np.diag(mo_energy[k][:2]),
            ))
            self.assertTrue(np.allclose(
                transformed[3:, 3:], np.diag(mo_energy[k][3:]),
            ))

    def test_charged_active_electron_sectors(self):
        get_nelec = kcasci._get_nelecas_for_charged_kcasci
        self.assertEqual(get_nelec(2, 8, 2, 0, charge=0), (8, 8))
        self.assertEqual(get_nelec(2, 8, 2, 0, charge=1), (8, 7))
        self.assertEqual(get_nelec(2, 8, 2, 0, charge=-1), (9, 8))
        self.assertEqual(
            get_nelec(2, 8, (1, 1), 0, charge=1, spin=-1),
            (7, 8),
        )

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
        self.assertEqual(
            [band["momentum_index"] for band in holes], [0, 2, 1],
        )
        self.assertTrue(np.allclose(
            [band["energy"] for band in holes], [-1.2, -1.5, -1.65],
        ))
        self.assertTrue(all(band["kind"] == "hole" for band in holes))
        self.assertTrue(np.allclose(holes[1]["hole_momentum"], kpts[2]))

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
        self.assertEqual(
            [band["momentum_index"] for band in particles], [0, 1, 2],
        )
        self.assertTrue(np.allclose(
            [band["energy"] for band in particles], [0.9, 1.2, 1.35],
        ))
        self.assertTrue(
            all(band["kind"] == "particle" for band in particles),
        )

        shifted = kcasci.compute_band_energies(
            hole_results, reference_energy=-1.2, root=1, kpts=kpts,
            reference_target_k=1,
        )
        self.assertEqual(
            [band["momentum_index"] for band in shifted], [1, 0, 2],
        )
        per_cell = kcasci.compute_band_energies(
            hole_results, reference_energy=-1.2, root=1, kpts=kpts,
            per_cell=True,
        )
        self.assertTrue(np.allclose(
            [band["energy"] for band in per_cell], [-0.4, -0.5, -0.55],
        ))

    def test_charged_band_energy_validation(self):
        result = [{
            "target_k": 0, "charge": 1, "nkpts": 2,
            "e_tot": np.asarray([-1.0]),
        }]
        self.assertEqual(kcasci.compute_band_energies([], -1.2), [])
        with self.assertRaisesRegex(IndexError, "scalar energy"):
            kcasci._select_root_energy(-1.0, root=1)
        with self.assertRaisesRegex(IndexError, "nonnegative"):
            kcasci._select_root_energy([-1.0], root=-1)
        with self.assertRaisesRegex(ValueError, r"charge \+1 or -1"):
            kcasci.compute_band_energies(result, -1.2, charge=0)

        mixed = result + [{
            "target_k": 1, "charge": -1, "nkpts": 2,
            "e_tot": -1.0,
        }]
        with self.assertRaisesRegex(ValueError, "inconsistent charges"):
            kcasci.compute_band_energies(mixed, -1.2, charge=1)

    def test_charged_finalize_forwards_result_target_k(self):
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
            self.assertTrue(np.array_equal(
                ci, mc.charged_results[target_k]["ci"],
            ))
            self.assertEqual(norb, mc.nkpts * mc.ncas)
            self.assertEqual(nelec, mc.charged_nelecastot)
            self.assertEqual(kwargs["nkpts"], mc.nkpts)
            self.assertEqual(kwargs["target_k"], target_k)


class KCASCIIntegrationTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        intra_h = 0.74
        inter_h = 1.5
        vacuum = 17.5

        cell = pgto.Cell()
        cell.a = np.diag([intra_h + inter_h, intra_h + inter_h, vacuum])
        cell.atom = [
            ["H", (0.0, 0.0, vacuum / 2.0)],
            ["H", (intra_h, 0.0, vacuum / 2.0)],
        ]
        cell.basis = "STO-6G"
        cell.unit = "Angstrom"
        cell.max_memory = 100000
        cell.ke_cutoff = 100
        cell.precision = 1e-10
        cell.verbose = 0
        cell.build()

        cls.kmesh = [2, 1, 1]
        kpts = cell.make_kpts(cls.kmesh, wrap_around=True)
        kmf = scf.KRHF(cell, kpts=kpts).density_fit(
            auxbasis="def2-svp-jkfit",
        )
        kmf.max_cycle = 1000
        kmf.exxdiv = None
        kmf.conv_tol = 1e-10
        kmf.verbose = 0
        kmf.kernel()
        if not kmf.converged:
            raise RuntimeError("KRHF reference did not converge")

        cls.cell = cell
        cls.kmf = kmf
        cls.mo_coeff = np.asarray(kmf.mo_coeff)

    def make_kcasci(self, ncas=2, target_k=0):
        mc = mcscf.KCASCI(
            self.kmf, ncas, 2, ncore=0, target_k=target_k,
        )
        mc.kmesh = self.kmesh
        mc.verbose = 0
        mc.fcisolver.verbose = 0
        mc.canonicalization = False
        return mc

    def make_charged_kcasci(self, charge, target_k=None,
                            charged_spin=None):
        mc = mcscf.KCASCI(
            self.kmf, 2, 2, ncore=0, target_k=target_k,
            charge=charge, charged_spin=charged_spin,
        )
        mc.kmesh = self.kmesh
        mc.verbose = 0
        mc.fcisolver.verbose = 0
        return mc

    def test_single_determinant_kcasci_equals_krhf(self):
        mc = self.make_kcasci(ncas=1)
        self.assertIsInstance(mc, kcasci.PBCKCASCI)
        self.assertEqual(mc.target_k, 0)

        e_kcasci = mc.kernel(self.mo_coeff)[0]
        self.assertEqual(np.size(mc.ci), 1)
        self.assertTrue(np.allclose(
            e_kcasci, self.kmf.e_tot, atol=1e-10, rtol=1e-10,
        ))

        dm1 = mc.make_rdm1()
        dm1_ref = np.asarray(self.kmf.make_rdm1())
        self.assertTrue(np.allclose(dm1, dm1_ref, atol=1e-10, rtol=1e-10))

    def test_target_k0_matches_full_casci(self):
        mc_ref = mcscf.CASCI(self.kmf, 2, 2, ncore=0)
        mc_ref.kmesh = self.kmesh
        mc_ref.verbose = 0
        mc_ref.fcisolver = direct_spin1_cplx.FCISolver(self.cell)
        mc_ref.fcisolver.verbose = 0
        mc_ref.canonicalization = False
        e_ref = mc_ref.kernel(self.mo_coeff)[0]

        mc = self.make_kcasci(target_k=0)
        h1eff, ecore = mc.get_h1eff(self.mo_coeff)
        h1alias, ecore_alias = mc.get_h1cas(self.mo_coeff)
        h2eff = mc.get_h2eff(self.mo_coeff)
        self.assertEqual(h1eff.shape, (2, 2, 2))
        self.assertEqual(h2eff.shape, (2, 2, 2, 2, 2, 2, 2))
        self.assertTrue(np.allclose(h1eff, h1alias))
        self.assertTrue(np.allclose(ecore, ecore_alias))

        e_test = mc.kernel(self.mo_coeff)[0]
        self.assertTrue(np.allclose(
            e_test, e_ref, atol=1e-10, rtol=1e-10,
        ))

        casdm1 = kcasci.make_casdm1(mc)
        self.assertEqual(casdm1.shape, (4, 4))
        self.assertTrue(np.allclose(casdm1, casdm1.conj().T))
        self.assertAlmostEqual(np.trace(casdm1).real, 4.0, places=10)

        dm1 = mc.make_rdm1()
        self.assertEqual(dm1.shape, self.mo_coeff.shape)
        for dm1_k in dm1:
            self.assertTrue(np.allclose(dm1_k, dm1_k.conj().T))
        overlap = np.asarray(self.kmf.get_ovlp())
        nelec = np.einsum("kij,kji->", dm1, overlap).real / mc.nkpts
        self.assertAlmostEqual(nelec, self.cell.nelectron, places=9)

        fock = mc.get_fock(target_k=0)
        self.assertEqual(fock.shape, self.mo_coeff.shape)
        for fock_k in fock:
            self.assertTrue(np.allclose(
                fock_k, fock_k.conj().T, atol=1e-10, rtol=1e-10,
            ))

        mo_canonical, ci, mo_energy = mc.canonicalize_(target_k=0)
        self.assertEqual(mo_canonical.shape, self.mo_coeff.shape)
        self.assertIs(ci, mc.ci)
        self.assertEqual(np.asarray(mo_energy).shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(mo_energy)))

    def test_target_k_wraps_to_equivalent_sector(self):
        mc1 = self.make_kcasci(target_k=1)
        e1 = mc1.kernel(self.mo_coeff)[0]

        mc_wrapped = self.make_kcasci(target_k=3)
        e_wrapped = mc_wrapped.kernel(self.mo_coeff)[0]
        self.assertEqual(mc_wrapped.fcisolver.target_k, 1)
        self.assertTrue(np.allclose(
            e_wrapped, e1, atol=1e-10, rtol=1e-10,
        ))

    def test_charged_hole_sweep_density_and_explicit_sector(self):
        hole = self.make_charged_kcasci(charge=1)
        self.assertIsInstance(hole, kcasci.ChargedPBCKCASCI)
        self.assertIsNone(hole.target_k)
        self.assertFalse(hasattr(mcscf, "ChargedKCASCI"))
        with self.assertRaisesRegex(NotImplementedError, "Fock matrix"):
            hole.get_fock()
        with self.assertRaisesRegex(NotImplementedError, "Canonicalization"):
            hole.canonicalize()

        with self.assertRaisesRegex(ValueError, "dict keyed by target_k"):
            hole.kernel(self.mo_coeff, ci0=np.ones(1))

        e_tot, e_cas, ci, _, _ = hole.kernel(self.mo_coeff)
        self.assertEqual(np.asarray(e_tot).shape, (2,))
        self.assertEqual(np.asarray(e_cas).shape, (2,))
        self.assertEqual(len(ci), 2)
        self.assertEqual(hole.charged_nelecastot, (2, 1))
        self.assertEqual(
            [result["target_k"] for result in hole.charged_results],
            [0, 1],
        )
        self.assertTrue(hole.converged)

        with self.assertRaisesRegex(ValueError, "target_k is required"):
            hole.make_rdm1()
        overlap = np.asarray(self.kmf.get_ovlp())
        for result in hole.charged_results:
            self.assertEqual(result["charge"], 1)
            self.assertEqual(result["nelecastot"], (2, 1))
            self.assertTrue(np.allclose(
                result["e_tot_supercell"],
                hole.nkpts * result["e_tot"],
            ))
            self.assertTrue(np.allclose(
                result["e_cas_supercell"],
                hole.nkpts * result["e_cas"],
            ))
            dm1 = hole.make_rdm1(target_k=result["target_k"])
            electron_count = np.einsum(
                "kij,kji->", dm1, overlap,
            ).real / hole.nkpts
            self.assertAlmostEqual(electron_count, 1.5, places=9)

        explicit = self.make_charged_kcasci(charge=1, target_k=1)
        e_explicit = explicit.kernel(self.mo_coeff)[0]
        self.assertEqual(len(explicit.charged_results), 1)
        self.assertTrue(np.allclose(
            e_explicit, hole.charged_results[1]["e_tot"],
            atol=1e-10, rtol=1e-10,
        ))

    def test_charged_particle_sweep_and_band_energies(self):
        neutral = self.make_kcasci(target_k=0)
        e_neutral = neutral.kernel(self.mo_coeff)[0]

        hole = self.make_charged_kcasci(charge=1)
        hole.kernel(self.mo_coeff)
        particle = self.make_charged_kcasci(charge=-1)
        particle.kernel(self.mo_coeff)
        self.assertEqual(particle.charged_nelecastot, (3, 2))

        hole_bands = hole.band_energies(e_neutral)
        particle_bands = particle.band_energies(e_neutral)
        self.assertEqual(len(hole_bands), hole.nkpts)
        self.assertEqual(len(particle_bands), particle.nkpts)
        self.assertEqual(
            sorted(band["momentum_index"] for band in hole_bands),
            list(range(hole.nkpts)),
        )
        self.assertEqual(
            sorted(band["momentum_index"] for band in particle_bands),
            list(range(particle.nkpts)),
        )

        for band in hole_bands:
            result = hole.charged_results[band["target_k"]]
            expected = hole.nkpts * (e_neutral - result["e_tot"])
            self.assertTrue(np.allclose(band["energy"], expected))
            self.assertEqual(band["kind"], "hole")
        for band in particle_bands:
            result = particle.charged_results[band["target_k"]]
            expected = particle.nkpts * (result["e_tot"] - e_neutral)
            self.assertTrue(np.allclose(band["energy"], expected))
            self.assertEqual(band["kind"], "particle")

        per_cell = particle.band_energies(e_neutral, per_cell=True)
        self.assertTrue(np.allclose(
            [band["energy"] for band in particle_bands],
            particle.nkpts * np.asarray([
                band["energy"] for band in per_cell
            ]),
        ))

        dm1 = particle.make_rdm1(target_k=0)
        overlap = np.asarray(self.kmf.get_ovlp())
        electron_count = np.einsum(
            "kij,kji->", dm1, overlap,
        ).real / particle.nkpts
        self.assertAlmostEqual(electron_count, 2.5, places=9)

    def test_charged_explicit_negative_spin_sector(self):
        hole = self.make_charged_kcasci(
            charge=1, target_k=0, charged_spin=-1,
        )
        hole.kernel(self.mo_coeff)
        self.assertEqual(hole.charged_nelecastot, (1, 2))
        self.assertEqual(hole.charged_results[0]["nelecastot"], (1, 2))


if __name__ == "__main__":
    unittest.main()
