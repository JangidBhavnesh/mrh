"""Integration checks for the Python kCASCI spectral workflow."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from pyscf.pbc import scf

from mrh.my_pyscf.pbc.fci import spectral_fn_helper as sfh
from mrh.my_pyscf.pbc.mcscf import kcasci
from mrh.tests.pbc.test_pbc_kcasci import _make_periodic_h2_cell


class SpectralWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cell = _make_periodic_h2_cell()
        cls.kpts = cls.cell.make_kpts([2, 1, 1], wrap_around=True)
        cls.kmf = scf.KRHF(cls.cell, cls.kpts).density_fit(
            auxbasis='def2-svp-jkfit')
        cls.kmf.exxdiv = None
        cls.kmf.conv_tol = 1e-10
        cls.kmf.kernel()
        if not cls.kmf.converged:
            raise RuntimeError('Spectral test KRHF did not converge')

    def test_root_workflow_single_multiple_and_davidson(self):
        orbitals = np.array(self.kmf.mo_coeff, copy=True)
        for davidson in (False, True):
            for nroots in (1, 2):
                with self.subTest(davidson=davidson, nroots=nroots):
                    def setup(mc, kind):
                        mc.kmesh = [2, 1, 1]
                        mc.verbose = mc.fcisolver.verbose = 0
                        mc.fcisolver.davidson_only = davidson
                        mc.fcisolver.conv_tol = 1e-10

                    roots = sfh.compute_kcasci_spectral_roots(
                        self.kmf, 2, 2, ncore=0, mo_coeff=orbitals,
                        target_k=1, nroots_neutral=nroots, nroots_hole=nroots,
                        nroots_particle=nroots, spin_sector_mode='spin_resolved',
                        solver_setup=setup)
                    self.assertEqual(len(roots.roots), 9 * nroots)
                    self.assertTrue(all(r['converged'] for r in roots.roots))
                    self.assertEqual(roots.nelecastot, (2, 2))
                    self.assertEqual(roots.ncastot, 4)
                    np.testing.assert_array_equal(orbitals, self.kmf.mo_coeff)
                    for job in [roots.neutral] + roots.hole + roots.particle:
                        np.testing.assert_array_equal(job.mo_coeff, orbitals)
                    for row in roots.roots:
                        self.assertAlmostEqual(row['energy_supercell'],
                                               2 * row['energy'])
                        self.assertAlmostEqual(np.vdot(row['ci'], row['ci']).real, 1)

                    poles = sfh.make_spectral_poles(
                        roots, neutral_root=nroots - 1, strict=True)
                    self.assertTrue(poles)
                    e0 = np.asarray(roots.neutral.e_tot).reshape(-1)[nroots - 1]
                    for jobs, kind in ((roots.hole, 'hole'),
                                       (roots.particle, 'particle')):
                        for job in jobs:
                            bands = job.band_energies(e0, reference_target_k=1)
                            for band in bands:
                                selected = [p for p in poles if p['kind'] == kind
                                            and p['target_k'] == band['target_k']
                                            and p['nelecastot'] == job.charged_nelecastot]
                                self.assertTrue(selected)
                                for pole in selected:
                                    expected = np.asarray(band['energy']).reshape(-1)[pole['root']]
                                    self.assertAlmostEqual(pole['omega'], expected)
                                    self.assertEqual(pole['k'], band['momentum_index'])
                    spectrum = sfh.make_spectral_function(
                        poles, npts=51, nkpts=2, norb=2,
                        spin_resolved=True, orbital_resolved=True)
                    self.assertEqual(spectrum['spectra']['total'].shape, (2, 2, 2, 51))
                    self.assertTrue(np.all(spectrum['spectra']['total'] >= 0))

    def test_setup_rejects_incompatible_changes_before_kernel(self):
        def change(name, value, on_solver=False):
            def setup(mc, kind):
                setattr(mc.fcisolver if on_solver else mc, name, value)
            return setup

        cases = [
            ('canonicalization', change('canonicalization', True)),
            ('natorb', change('natorb', True)),
            ('ncas', change('ncas', 1)),
            ('ncore', change('ncore', 1)),
            ('nelecas', change('nelecas', (0, 1))),
            ('target_k', change('target_k', 1)),
            ('kpts', change('kpts', self.kpts[::-1].copy())),
            ('kmesh', change('kmesh', [1, 2, 1])),
            ('kconserv', change('kconserv', np.zeros((2, 2, 2), dtype=int))),
            ('fcisolver.kconserv', change('kconserv', np.zeros((2, 2, 2), dtype=int), True)),
            ('mo_coeff', lambda mc, kind: mc.mo_coeff.fill(0)),
            ('kmom.kadd', lambda mc, kind: mc.kmom.kadd.fill(0)),
        ]
        orbitals = np.array(self.kmf.mo_coeff, copy=True)
        with mock.patch.object(kcasci.PBCKCASCI, 'kernel') as kernel:
            for name, setup in cases:
                with self.subTest(setting=name):
                    with self.assertRaisesRegex(ValueError, 'common orbital basis'):
                        sfh.compute_kcasci_spectral_roots(
                            self.kmf, 2, 2, ncore=0, solver_setup=setup)
            kernel.assert_not_called()
        np.testing.assert_array_equal(self.kmf.mo_coeff, orbitals)

    def test_neutral_only_and_representative_jobs(self):
        roots = sfh.compute_kcasci_spectral_roots(
            self.kmf, 2, 2, ncore=0, with_hole=False, with_particle=False)
        self.assertEqual(len(roots.roots), 1)
        self.assertEqual(roots.hole, [])
        self.assertEqual(roots.particle, [])
        self.assertEqual(sfh.make_spectral_poles(roots), [])
        roots = sfh.compute_kcasci_spectral_roots(self.kmf, 2, 2, ncore=0)
        self.assertIsInstance(roots.hole, kcasci.ChargedPBCKCASCI)
        self.assertIsInstance(roots.particle, kcasci.ChargedPBCKCASCI)
        self.assertEqual(len(roots.roots), 5)
        with self.assertRaisesRegex(ValueError, 'missing .* sector'):
            sfh.make_spectral_poles(roots, strict=True)

    def test_charged_collection_prefers_stored_supercell_energy(self):
        for energies, ci in ((2., np.ones(1)),
                             (np.array([2., 3.]), [np.ones(1), np.ones(1)])):
            stored = np.asarray(energies) * 2 + 0.125
            job = SimpleNamespace(charged_results=[dict(
                e_tot=energies, e_tot_supercell=stored, ci=ci, charge=1,
                target_k=0, nelecastot=(1, 0), ncastot=2, nkpts=2,
                converged=True)])
            rows = sfh._collect_charged_roots(job, 'hole')
            np.testing.assert_allclose([r['energy_supercell'] for r in rows],
                                       stored.reshape(-1))


if __name__ == '__main__':
    unittest.main()
