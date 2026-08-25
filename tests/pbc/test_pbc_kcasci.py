#!/usr/bin/env python
import io
import unittest

import numpy as np

from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.fci import direct_spin1_cplx
from mrh.my_pyscf.pbc.mcscf import kcasci


# Author: Bhavnesh Jangid

# The reference energies are the per-cell total energies generated with the 
# commit:69a0340711baea687d0aadd99d0c18a427ab1dd5

THREE_K_NEUTRAL_REFERENCE_ENERGIES = (
    -1.470102921533587,
    -1.262884393685463,
    -1.262884393685464,
)

THREE_K_CHARGED_REFERENCE_ENERGIES = {
    1: (
        -1.147816186679004,
        -1.191635629937168,
        -1.191635629937171,
    ),
    -1: (
        -1.136544208043976,
        -1.308553153122311,
        -1.308553153122315,
    ),
}


def _make_periodic_h2_cell():
    """Build the periodic H2 cell shared by the kCASCI energy tests."""
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
    cell.ke_cutoff = 100
    cell.precision = 1e-10
    cell.verbose = 0
    cell.build()
    return cell

"""
Reference-energy tests for momentum-resolved periodic CASCI.
"""

class KCASCIReferenceEnergyTests(unittest.TestCase):
    """Test KCASCI against reference calculations on a periodic H chain."""

    @classmethod
    def setUpClass(cls):
        """Build the shared two-k-point KRHF reference."""
        cell = _make_periodic_h2_cell()

        cls.kmesh = [2, 1, 1]
        kpts = cell.make_kpts(cls.kmesh, wrap_around=True)
        kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis="def2-svp-jkfit")
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
        """Create a quiet neutral KCASCI solver."""
        mc = mcscf.KCASCI(self.kmf, ncas, 2, ncore=0, target_k=target_k)
        mc.kmesh = self.kmesh
        mc.verbose = 0
        mc.fcisolver.verbose = 0
        mc.canonicalization = False
        return mc

    def make_charged_kcasci(self, charge, target_k=None,
                            charged_spin=None):
        """Create a quiet charged KCASCI solver."""
        mc = mcscf.KCASCI(
            self.kmf, 2, 2, ncore=0, target_k=target_k,
            charge=charge, charged_spin=charged_spin,
        )
        mc.kmesh = self.kmesh
        mc.verbose = 0
        mc.fcisolver.verbose = 0
        return mc

    def test_single_kpoint_charged_energies_match_nelecas(self):
        """Match charged energies to neutral solves with N-1 and N+1 electrons."""
        kmesh = [1, 1, 1]
        kpts = self.cell.make_kpts(kmesh, wrap_around=True)
        kmf = scf.KRHF(self.cell, kpts=kpts).density_fit(
            auxbasis="def2-svp-jkfit",
        )
        kmf.max_cycle = 1000
        kmf.exxdiv = None
        kmf.conv_tol = 1e-10
        kmf.verbose = 0
        kmf.kernel()
        self.assertTrue(kmf.converged)
        mo_coeff = np.asarray(kmf.mo_coeff)

        for charge in (1, -1):
            charged = mcscf.KCASCI(kmf, 2, 2, ncore=0, charge=charge,)
            charged.kmesh = kmesh
            charged.verbose = charged.fcisolver.verbose = 0
            e_charged = charged.kernel(mo_coeff)[0]

            nelecas = 2 - charge
            reference = mcscf.KCASCI(kmf, 2, nelecas, ncore=0, target_k=0,)
            reference.kmesh = kmesh
            reference.verbose = reference.fcisolver.verbose = 0
            e_reference = kcasci.kernel(reference, mo_coeff, verbose=0)[0]

            self.assertEqual(sum(charged.charged_nelecastot), nelecas)
            self.assertTrue(np.allclose(e_charged, e_reference, 
                                        atol=1e-10, rtol=1e-10,))

    def test_single_determinant_kcasci_equals_krhf(self):
        """Recover KRHF from a single-determinant active space."""
        mc = self.make_kcasci(ncas=1)
        self.assertIsInstance(mc, kcasci.PBCKCASCI)
        self.assertEqual(mc.target_k, 0)

        e_kcasci = mc.kernel(self.mo_coeff)[0]
        self.assertEqual(np.size(mc.ci), 1)
        self.assertTrue(np.allclose(
            e_kcasci, self.kmf.e_tot, atol=1e-10, rtol=1e-10))

        dm1 = mc.make_rdm1()
        dm1_ref = np.asarray(self.kmf.make_rdm1())
        self.assertTrue(np.allclose(dm1, dm1_ref, atol=1e-10, rtol=1e-10))

    def test_target_k0_matches_full_casci(self):
        """Match full CASCI and validate neutral KCASCI observables."""
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
        self.assertTrue(np.allclose(e_test, e_ref, atol=1e-10, rtol=1e-10))

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
                fock_k, fock_k.conj().T, atol=1e-10, rtol=1e-10))

        mo_canonical, ci, mo_energy = mc.canonicalize_(target_k=0)
        self.assertEqual(mo_canonical.shape, self.mo_coeff.shape)
        self.assertIs(ci, mc.ci)
        self.assertEqual(np.asarray(mo_energy).shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(mo_energy)))

    def test_target_k_wraps_to_equivalent_sector(self):
        """Wrap out-of-range target momenta to equivalent sectors."""
        mc1 = self.make_kcasci(target_k=1)
        e1 = mc1.kernel(self.mo_coeff)[0]

        mc_wrapped = self.make_kcasci(target_k=3)
        e_wrapped = mc_wrapped.kernel(self.mo_coeff)[0]
        self.assertEqual(mc_wrapped.fcisolver.target_k, 1)
        self.assertTrue(np.allclose(e_wrapped, e1, atol=1e-10, rtol=1e-10))

    def test_charged_hole_sweep_density_and_explicit_sector(self):
        """Compare all-sector and explicit-sector hole calculations."""
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
            [result["target_k"] for result in hole.charged_results], [0, 1])
        self.assertTrue(hole.converged)

        with self.assertRaisesRegex(ValueError, "target_k is required"):
            hole.make_rdm1()
        overlap = np.asarray(self.kmf.get_ovlp())
        for result in hole.charged_results:
            self.assertEqual(result["charge"], 1)
            self.assertEqual(result["nelecastot"], (2, 1))
            self.assertTrue(np.allclose(
                result["e_tot_supercell"], hole.nkpts * result["e_tot"]))
            self.assertTrue(np.allclose(
                result["e_cas_supercell"], hole.nkpts * result["e_cas"]))
            dm1 = hole.make_rdm1(target_k=result["target_k"])
            electron_count = np.einsum("kij,kji->", dm1, overlap).real / hole.nkpts
            self.assertAlmostEqual(electron_count, 1.5, places=9)

        explicit = self.make_charged_kcasci(charge=1, target_k=1)
        e_explicit = explicit.kernel(self.mo_coeff)[0]
        self.assertEqual(len(explicit.charged_results), 1)
        self.assertTrue(np.allclose(
            e_explicit, hole.charged_results[1]["e_tot"],
            atol=1e-10, rtol=1e-10))

    def test_charged_particle_sweep_and_band_energies(self):
        """Validate particle/hole band energies and the particle density."""
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
            list(range(hole.nkpts)))
        self.assertEqual(
            sorted(band["momentum_index"] for band in particle_bands),
            list(range(particle.nkpts)))

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
            particle.nkpts * np.asarray([band["energy"] for band in per_cell])))

        particle.stdout = io.StringIO()
        printed_bands = particle.print_bands(
            e_neutral, verbose=lib.logger.NOTE,
        )
        output = particle.stdout.getvalue()
        self.assertIn("Particle (N+1)", output)
        self.assertIn("addition pole", output)
        scaled_kpts = particle.cell.get_scaled_kpts(particle.kpts)
        scaled_kx = [
            scaled_kpts[band["momentum_index"], 0]
            for band in printed_bands
        ]
        self.assertEqual(scaled_kx, sorted(scaled_kx))

        dm1 = particle.make_rdm1(target_k=0)
        overlap = np.asarray(self.kmf.get_ovlp())
        electron_count = np.einsum("kij,kji->", dm1, overlap).real / particle.nkpts
        self.assertAlmostEqual(electron_count, 2.5, places=9)

    def test_charged_explicit_negative_spin_sector(self):
        """Solve an explicit negative-spin hole sector."""
        hole = self.make_charged_kcasci(
            charge=1, target_k=0, charged_spin=-1,
        )
        hole.kernel(self.mo_coeff)
        self.assertEqual(hole.charged_nelecastot, (1, 2))
        self.assertEqual(hole.charged_results[0]["nelecastot"], (1, 2))


class KCASCIThreeKPointReferenceEnergyTests(unittest.TestCase):
    """Compare three-sector periodic H2 energies with fixed references."""

    @classmethod
    def setUpClass(cls):
        cls.cell = _make_periodic_h2_cell()
        cls.kmesh = [3, 1, 1]
        kpts = cls.cell.make_kpts(cls.kmesh, wrap_around=True)
        cls.kmf = scf.KRHF(cls.cell, kpts=kpts).density_fit(
            auxbasis="def2-svp-jkfit",
        )
        cls.kmf.max_cycle = 1000
        cls.kmf.exxdiv = None
        cls.kmf.conv_tol = 1e-10
        cls.kmf.verbose = 0
        cls.kmf.kernel()
        if not cls.kmf.converged:
            raise RuntimeError("Three-k-point KRHF reference did not converge")
        cls.mo_coeff = np.asarray(cls.kmf.mo_coeff)

    def make_kcasci(self, target_k=None, charge=None):
        mc = mcscf.KCASCI(
            self.kmf, 2, 2, ncore=0, target_k=target_k, charge=charge,
        )
        mc.kmesh = self.kmesh
        mc.verbose = 0
        mc.fcisolver.verbose = 0
        mc.canonicalization = False
        return mc

    def assert_reference_energy(self, actual, reference, *, sector):
        self.assertAlmostEqual(
            float(np.real(actual)), reference, places=7,
            msg=(f"{sector} energy mismatch: actual={actual}, reference={reference}"),
        )

    def test_neutral_energies_for_all_target_k(self):
        """Check the neutral CAS(2,2) energy in all three momentum sectors."""
        for target_k, reference in enumerate(
                THREE_K_NEUTRAL_REFERENCE_ENERGIES):
            with self.subTest(target_k=target_k):
                mc = self.make_kcasci(target_k=target_k)
                energy = mc.kernel(self.mo_coeff)[0]
                self.assert_reference_energy(
                    energy, reference, sector=f"neutral target_k={target_k}",
                )

    def test_charged_energies_for_all_target_k(self):
        """Check N-1 and N+1 energies in all three momentum sectors."""
        for charge, references in THREE_K_CHARGED_REFERENCE_ENERGIES.items():
            mc = self.make_kcasci(charge=charge)
            energies = np.asarray(mc.kernel(self.mo_coeff)[0])
            self.assertEqual(energies.shape, (3,))
            self.assertEqual(
                [result["target_k"] for result in mc.charged_results],
                [0, 1, 2],
            )
            for target_k, (energy, reference) in enumerate(
                    zip(energies, references)):
                with self.subTest(charge=charge, target_k=target_k):
                    self.assert_reference_energy(
                        energy, reference,
                        sector=f"charge={charge:+d} target_k={target_k}",
                    )


if __name__ == "__main__":
    unittest.main()
