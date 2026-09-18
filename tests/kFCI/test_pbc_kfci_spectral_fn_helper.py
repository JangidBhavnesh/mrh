#!/usr/bin/env python

import os
import tempfile
import unittest
from dataclasses import replace
import numpy as np

from pyscf.fci import cistring
from pyscf.fci.addons import _unpack_nelec
from pyscf.pbc import gto

from mrh.my_pyscf.pbc.fci import kcistrings
from mrh.my_pyscf.pbc.fci import krdm_helper
from mrh.my_pyscf.pbc.fci import spectral_fn_helper as sfh


# Author: Bhavnesh Jangid

'''
Tests for number-changing k-FCI helper functions used in spectral functions.
'''


def _apply_full_op(ci, norb, nelec, orb, spin, cre=False):
    '''
    Reference full spin-string CI implementation of one creation/destruction.
    '''
    neleca, nelecb = _unpack_nelec(nelec)
    spin = 0 if spin in (0, 'a', 'alpha') else 1
    target_nelec = [neleca, nelecb]
    target_nelec[spin] += 1 if cre else -1
    target_nelec = tuple(target_nelec)

    nstra1 = cistring.num_strings(norb, target_nelec[0])
    nstrb1 = cistring.num_strings(norb, target_nelec[1])
    out = np.zeros((nstra1, nstrb1), dtype=ci.dtype)

    ORB = 0 if cre else 1
    TARGET = 2
    SIGN = 3

    if spin == 0:
        op_index = (cistring.gen_cre_str_index(range(norb), neleca)
                    if cre else
                    cistring.gen_des_str_index(range(norb), neleca))
        for ia0 in range(ci.shape[0]):
            for link in op_index[ia0]:
                if int(link[ORB]) != orb:
                    continue
                out[int(link[TARGET]), :] += int(link[SIGN]) * ci[ia0, :]
    else:
        op_index = (cistring.gen_cre_str_index(range(norb), nelecb)
                    if cre else
                    cistring.gen_des_str_index(range(norb), nelecb))
        beta_phase = -1 if (neleca % 2) else 1
        for ib0 in range(ci.shape[1]):
            for link in op_index[ib0]:
                if int(link[ORB]) != orb:
                    continue
                out[:, int(link[TARGET])] += (
                    beta_phase * int(link[SIGN]) * ci[:, ib0])

    return out, target_nelec


def _one_alpha_roots():
    '''
    Build a tiny exact root table with one occupied alpha k-orbital.
    '''
    nkpts = 2
    ncas = 1
    norb = nkpts * ncas
    nelec = (1, 0)
    return sfh.KCASCISpectralRoots(
        neutral=None, hole=None, particle=None,
        roots=[
            {
                'kind': 'neutral',
                'charge': 0,
                'target_k': 0,
                'root': 0,
                'energy': 1.0,
                'energy_supercell': 1.0,
                'ci': np.ones(1),
                'nelecastot': nelec,
                'ncastot': norb,
                'nkpts': nkpts,
                'converged': True,
            },
            {
                'kind': 'hole',
                'charge': 1,
                'target_k': 0,
                'root': 0,
                'energy': 0.25,
                'energy_supercell': 0.25,
                'ci': np.ones(1),
                'nelecastot': (0, 0),
                'ncastot': norb,
                'nkpts': nkpts,
                'converged': True,
            },
            {
                'kind': 'particle',
                'charge': -1,
                'target_k': 1,
                'root': 0,
                'energy': 1.75,
                'energy_supercell': 1.75,
                'ci': np.ones(1),
                'nelecastot': (2, 0),
                'ncastot': norb,
                'nkpts': nkpts,
                'converged': True,
            },
        ],
        nkpts=nkpts, ncas=ncas, ncastot=norb, nelecastot=nelec,
        target_k=0, mo_coeff=None)


def _complete_2d_roots(kmom, target_k=2):
    '''A complex neutral state and complete charged determinant bases.'''
    nkpts = kmom.nkpts
    norb = nkpts
    nelec = (1, 1)
    layout = sfh.make_k_sector_layout(norb, nelec, nkpts, target_k, kmom=kmom)
    rng = np.random.default_rng(41)
    ci = rng.normal(size=layout.sector_size) + 1j * rng.normal(size=layout.sector_size)
    ci /= np.linalg.norm(ci)
    common = dict(ncastot=norb, nkpts=nkpts, converged=True)
    rows = [dict(common, kind='neutral', charge=0, target_k=target_k, root=0,
                 energy=-2., energy_supercell=-8., ci=ci, nelecastot=nelec)]
    for kind, charge, sectors in (('hole', 1, ((0, 1), (1, 0))),
                                  ('particle', -1, ((2, 1), (1, 2)))):
        for electrons in sectors:
            for momentum in range(nkpts):
                layout = sfh.make_k_sector_layout(
                    norb, electrons, nkpts, momentum, kmom=kmom)
                for root, vec in enumerate(np.eye(layout.sector_size)):
                    rows.append(dict(common, kind=kind, charge=charge,
                                     target_k=momentum, root=root, ci=vec,
                                     energy=-1., energy_supercell=-4.,
                                     nelecastot=electrons))
    return sfh.KCASCISpectralRoots(
        neutral=None, hole=None, particle=None, roots=rows, nkpts=nkpts,
        ncas=1, ncastot=norb, nelecastot=nelec, target_k=target_k,
        mo_coeff=None, kmom=kmom, spin_sector_mode='spin_resolved')


class KnownValues(unittest.TestCase):

    @staticmethod
    def _make_2d_kmom():
        '''
        Build a small non-scalar 2D k-point table.
        '''
        cell = gto.Cell()
        cell.a = np.eye(3) * 4.0
        cell.atom = 'He 0 0 0'
        cell.basis = 'sto-3g'
        cell.verbose = 0
        cell.build()
        kpts = cell.make_kpts([2, 2, 1], wrap_around=True)
        return kcistrings.make_kpoint_momentum(len(kpts), cell=cell,
                                               kpts=kpts)

    def test_des_k_matches_full_ci_operator(self):
        self._check_k_operator(cre=False)

    def test_cre_k_matches_full_ci_operator(self):
        self._check_k_operator(cre=True)

    def test_odd_alpha_count_matches_full_ci_operator(self):
        for cre in (False, True):
            self._check_k_operator(cre=cre, nelec=(1, 1))

    def test_creation_on_vacuum_and_destruction_on_full_spin(self):
        for spin in (0, 1):
            vacuum = np.ones(1, dtype=complex)
            created, info = sfh.cre_k(
                vacuum, 2, (0, 0), 2, 0, 1, 0, spin, return_info=True)
            self.assertEqual(info['target_k'], 1)
            restored = sfh.des_k(created, 2, info['nelec'], 2, 1, 1, 0, spin)
            np.testing.assert_allclose(restored, vacuum)
            full_nelec = (2, 0) if spin == 0 else (0, 2)
            destroyed, info = sfh.des_k(
                vacuum, 2, full_nelec, 2, 1, 1, 0, spin, return_info=True)
            restored = sfh.cre_k(destroyed, 2, info['nelec'], 2, 0, 1, 0, spin)
            np.testing.assert_allclose(restored, vacuum)
            with self.assertRaisesRegex(ValueError, 'empty spin sector'):
                sfh.des_k(vacuum, 2, (0, 0), 2, 0, 0, 0, spin)
            with self.assertRaisesRegex(ValueError, 'full spin sector'):
                sfh.cre_k(vacuum, 2, full_nelec, 2, 1, 0, 0, spin)

    def test_opposite_spin_operators_anticommute(self):
        def annihilate(ci, nelec, momentum, spin):
            return sfh.des_k(ci, 2, nelec, 2, momentum, 0, 0, spin,
                             return_info=True)
        layout = sfh.make_k_sector_layout(2, (1, 1), 2)
        ci = np.arange(1, layout.sector_size + 1, dtype=complex)
        a, ai = annihilate(ci, (1, 1), 0, 0)
        ab, _ = annihilate(a, ai['nelec'], ai['target_k'], 1)
        b, bi = annihilate(ci, (1, 1), 0, 1)
        ba, _ = annihilate(b, bi['nelec'], bi['target_k'], 0)
        self.assertGreater(np.linalg.norm(ab), 0)
        np.testing.assert_allclose(ab, -ba)

    def test_k_operators_match_full_ci_on_2d_kmesh(self):
        kmom = self._make_2d_kmom()
        self.assertFalse(kmom.scalar)

        nkpts = kmom.nkpts
        ncas = 2
        norb = nkpts * ncas
        nelec = (2, 1)
        rng = np.random.default_rng(31)

        for cre in (False, True):
            for target_k in range(nkpts):
                src_layout = sfh.make_k_sector_layout(
                    norb, nelec, nkpts, target_k=target_k, kmom=kmom)
                fcivec = (rng.normal(size=src_layout.sector_size)
                          + 1j * rng.normal(size=src_layout.sector_size))
                ci_full = krdm_helper.embed_ksector_ci_to_full(
                    fcivec, norb, nelec, nkpts, target_k=target_k,
                    link_index=src_layout.link_index, kmom=kmom)

                for k in range(nkpts):
                    for spin in (0, 1):
                        with self.subTest(cre=cre, target_k=target_k,
                                          k=k, spin=spin):
                            if cre:
                                test, info = sfh.cre_k(
                                    fcivec, norb, nelec, nkpts, target_k,
                                    k, 0, spin, return_info=True,
                                    source_link_index=src_layout.link_index,
                                    kmom=kmom)
                            else:
                                test, info = sfh.des_k(
                                    fcivec, norb, nelec, nkpts, target_k,
                                    k, 0, spin, return_info=True,
                                    source_link_index=src_layout.link_index,
                                    kmom=kmom)

                            orb = k * ncas
                            ref_full, target_nelec = _apply_full_op(
                                ci_full, norb, nelec, orb, spin, cre=cre)
                            target_layout = sfh.make_k_sector_layout(
                                norb, target_nelec, nkpts,
                                target_k=info['target_k'], kmom=kmom)
                            ref = krdm_helper.extract_ksector_ci_from_full(
                                ref_full, norb, target_nelec, nkpts,
                                target_k=info['target_k'],
                                link_index=target_layout.link_index,
                                kmom=kmom)

                            self.assertEqual(info['nelec'], target_nelec)
                            self.assertEqual(info['target_k'],
                                             (kcistrings._kadd(kmom, target_k, k)
                                              if cre else
                                              kcistrings._ksub(kmom, target_k, k)))
                            self.assertEqual(test.shape, ref.shape)
                            self.assertTrue(np.allclose(test, ref))

    def test_2d_poles_and_sums_accept_explicit_momentum_for_plain_roots(self):
        kmom = self._make_2d_kmom()
        roots = _complete_2d_roots(kmom)
        poles = sfh.make_spectral_poles(roots, strict=True)
        plain = sfh.make_spectral_poles(iter(roots.roots), kmom=kmom, strict=True)
        self.assertEqual(len(poles), len(plain))
        for ref, actual in zip(poles, plain):
            self.assertEqual(ref, actual)
            # Addition/subtraction are componentwise modulo two, not modulo four.
            k = ref['k']
            expected = 2 * ((roots.target_k // 2 + k // 2) % 2)
            expected += (roots.target_k % 2 + k % 2) % 2
            self.assertEqual(ref['target_k'], expected)
        dm1s = krdm_helper.make_rdm1s_ref(
            roots.roots[0]['ci'], 4, (1, 1), 4, target_k=2, kmom=kmom)
        reference = sfh.spectral_weight_sum_rules(roots, poles)
        # Omitted poles forces the internal pole builder to receive kmom too.
        plain_checks = sfh.spectral_weight_sum_rules(iter(roots.roots), kmom=kmom)
        for ref, actual in zip(reference, plain_checks):
            self.assertEqual(ref, actual)
            occupation = dm1s[ref['spin']][ref['k'], ref['k']].real
            self.assertAlmostEqual(ref['hole_weight'], occupation, places=12)
            self.assertAlmostEqual(ref['particle_weight'], 1 - occupation, places=12)
            self.assertAlmostEqual(ref['full_total_norm'], 1, places=12)
            self.assertAlmostEqual(ref['full_total_missing'], 0, places=12)

    def test_momentum_metadata_validation_and_stored_table_fallback(self):
        kmom = self._make_2d_kmom()
        roots = _complete_2d_roots(kmom)
        table_only = replace(roots, kmom=None, kconserv=kmom.kconserv)
        self.assertEqual(sfh.make_spectral_poles(roots),
                         sfh.make_spectral_poles(table_only))
        for fn in (sfh.make_spectral_poles, sfh.spectral_weight_sum_rules):
            with self.assertRaisesRegex(ValueError, 'matching nkpts'):
                fn(roots.roots, kmom=kcistrings.make_kpoint_momentum(3))
            with self.assertRaisesRegex(ValueError, 'conflicts'):
                fn(roots, kmom=kcistrings.make_kpoint_momentum(4))

    def test_plain_roots_preserve_nonzero_gamma_index(self):
        cell = gto.Cell()
        cell.a = np.eye(3) * 4
        cell.atom = 'He 0 0 0'
        cell.basis = 'sto-3g'
        cell.verbose = 0
        cell.build()
        kpts = cell.make_kpts([2, 2, 1], wrap_around=True)[[3, 0, 1, 2]]
        kmom = kcistrings.make_kpoint_momentum(4, cell=cell, kpts=kpts)
        self.assertEqual(kmom.zero, 1)
        roots = _complete_2d_roots(kmom, target_k=2)
        poles = sfh.make_spectral_poles(roots.roots, kmom=kmom, strict=True)
        for pole in sfh.label_pole_momenta(poles, kpts):
            sign = -1 if pole['kind'] == 'hole' else 1
            delta = cell.get_scaled_kpts(
                kpts[roots.target_k] + sign * pole['operator_momentum']
                - pole['charged_momentum'])
            np.testing.assert_allclose(delta, np.rint(delta), atol=1e-12)
        checks = sfh.spectral_weight_sum_rules(roots.roots, kmom=kmom)
        for check in checks:
            self.assertAlmostEqual(check['full_total_norm'], 1, places=12)
            self.assertAlmostEqual(check['full_total_missing'], 0, places=12)

    def test_strict_and_full_totals_allow_zero_norm_missing_sectors(self):
        roots = _one_alpha_roots()
        poles = sfh.make_spectral_poles(roots, spins=(0,), strict=True)
        checks = sfh.spectral_weight_sum_rules(roots, poles, spins=(0,))
        for row in checks:
            self.assertAlmostEqual(row['full_total_norm'], 1)
            self.assertAlmostEqual(row['full_total_missing'], 0)
        self.assertFalse(checks[1]['hole_sector_available'])
        self.assertEqual(checks[1]['hole_norm'], 0)

    def test_spin_sector_choices_respect_empty_and_full_spin(self):
        # (N_alpha, N_beta)=(2,0) in two orbitals: only alpha removal and beta addition.
        self.assertEqual(sfh._charged_spin_list(
            (2, 0), 2, 'hole', None, 'spin_resolved'), [1])
        self.assertEqual(sfh._charged_spin_list(
            (2, 0), 2, 'particle', None, 'spin_resolved'), [1])
        self.assertEqual(sfh._charged_spin_list(
            (0, 0), 2, 'hole', None, 'spin_resolved'), [])
        self.assertEqual(sfh._charged_spin_list(
            (2, 2), 2, 'particle', None, 'spin_resolved'), [])
        self.assertEqual(sfh._charged_spin_list(
            (1, 1), 2, 'hole', -1, 'representative'), [-1])
        self.assertEqual(sfh._charged_spin_list(
            (1, 1), 2, 'hole', [1], 'spin_resolved'), [1])

    def test_make_spectral_poles_from_charged_roots(self):
        roots = _one_alpha_roots()

        poles = sfh.make_spectral_poles(
            roots, k_indices=(0, 1), orbital_indices=(0,),
            spins=(0,), min_weight=1e-12)
        labels = {(row['kind'], row['k']): row for row in poles}

        self.assertEqual(set(labels), {('hole', 0), ('particle', 1)})
        self.assertAlmostEqual(labels[('hole', 0)]['omega'], 0.75)
        self.assertAlmostEqual(labels[('hole', 0)]['weight'], 1.0)
        self.assertAlmostEqual(labels[('particle', 1)]['omega'], 0.75)
        self.assertAlmostEqual(labels[('particle', 1)]['weight'], 1.0)

    def test_make_spectral_function_broadens_poles(self):
        roots = _one_alpha_roots()
        poles = sfh.make_spectral_poles(
            roots, k_indices=(0, 1), orbital_indices=(0,),
            spins=(0,), min_weight=1e-12)

        spectrum = sfh.make_spectral_function(
            poles, eta=0.05, npts=101, nkpts=2, norb=1,
            spin_resolved=True, orbital_resolved=True)

        self.assertEqual(spectrum['spectra']['total'].shape, (2, 1, 2, 101))
        self.assertTrue(np.all(spectrum['spectra']['total'] >= 0.0))
        self.assertGreater(spectrum['spectra']['hole'][0, 0, 0].max(), 0.0)
        self.assertGreater(spectrum['spectra']['particle'][1, 0, 0].max(),
                           0.0)

    def test_broadening_normalization_and_empty_poles(self):
        poles = sfh.make_spectral_poles(_one_alpha_roots(), spins=(0,))
        omega = np.linspace(-50, 50, 20001)
        for broadening in ('lorentzian', 'gaussian'):
            spectrum = sfh.make_spectral_function(
                poles, omega_grid=omega, eta=0.05, broadening=broadening,
                nkpts=2, norb=1, use_c=False)
            total = spectrum['spectra']['total']
            np.testing.assert_allclose(
                np.trapz(total, omega, axis=-1), 1, atol=7e-4, rtol=0)
            np.testing.assert_allclose(total, spectrum['spectra']['hole']
                                       + spectrum['spectra']['particle'])
        empty = sfh.make_spectral_function([], omega_grid=omega,
                                           nkpts=2, norb=1)
        self.assertFalse(empty['spectra']['total'].any())

    def test_complex_amplitudes_and_coherent_band_projection(self):
        roots = _one_alpha_roots()
        roots.nkpts = 1
        roots.ncas = 2
        roots.roots = roots.roots[:2]
        for row in roots.roots:
            row['nkpts'] = 1
        roots.roots[0]['ci'] = np.array([1, 1j]) / np.sqrt(2)
        poles = sfh.make_spectral_poles(roots, spins=(0,), include_particle=False)
        np.testing.assert_allclose([p['amplitude'] for p in poles],
                                   np.array([1, 1j]) / np.sqrt(2))
        coeff = np.array([[1, 1], [1j, -1j]]) / np.sqrt(2)
        projected = sfh.project_poles_to_band_basis(poles, coeff)
        np.testing.assert_allclose([p['weight'] for p in projected], [1, 0],
                                   atol=1e-14)
        # Changing the charged state's phase must preserve all intensities.
        roots.roots[1]['ci'] = np.array([1j])
        phased = sfh.make_spectral_poles(roots, spins=(0,), include_particle=False)
        np.testing.assert_allclose([p['weight'] for p in poles],
                                   [p['weight'] for p in phased])

    def test_spectral_weight_sum_rules(self):
        roots = _one_alpha_roots()
        poles = sfh.make_spectral_poles(
            roots, k_indices=(0, 1), orbital_indices=(0,),
            spins=(0,), min_weight=1e-12)
        checks = sfh.spectral_weight_sum_rules(
            roots, poles=poles, k_indices=(0, 1), orbital_indices=(0,),
            spins=(0,))
        by_k = {row['k']: row for row in checks}

        self.assertAlmostEqual(by_k[0]['hole_norm'], 1.0)
        self.assertAlmostEqual(by_k[0]['hole_missing'], 0.0)
        self.assertAlmostEqual(by_k[1]['particle_norm'], 1.0)
        self.assertAlmostEqual(by_k[1]['particle_missing'], 0.0)

    def test_labels_projection_and_npz_output(self):
        roots = _one_alpha_roots()
        poles = sfh.make_spectral_poles(
            roots, k_indices=(0, 1), orbital_indices=(0,),
            spins=(0,), min_weight=1e-12)

        kpts = np.asarray([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        labelled = sfh.label_pole_momenta(poles, kpts)
        self.assertTrue(np.allclose(labelled[0]['operator_momentum'],
                                    kpts[labelled[0]['k']]))

        coeff = np.ones((2, 1, 1))
        projected = sfh.project_poles_to_band_basis(labelled, coeff)
        self.assertEqual(len(projected), len(labelled))
        self.assertEqual(projected[0]['basis'], 'band')
        self.assertAlmostEqual(projected[0]['weight'], labelled[0]['weight'])

        spectrum = sfh.make_spectral_function(
            projected, eta=0.05, npts=31, nkpts=2, norb=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'spectral.npz')
            sfh.save_spectral_npz(filename, spectrum, poles=projected)
            data = np.load(filename, allow_pickle=True)
            self.assertIn('omega', data.files)
            self.assertIn('total', data.files)
            self.assertIn('poles', data.files)

    def _check_k_operator(self, cre=False, nelec=(2, 1)):
        nkpts = 3
        ncas = 2
        norb = nkpts * ncas
        rng = np.random.default_rng(12 if cre else 11)

        for target_k in range(nkpts):
            src_layout = sfh.make_k_sector_layout(
                norb, nelec, nkpts, target_k=target_k)
            fcivec = (rng.normal(size=src_layout.sector_size)
                      + 1j * rng.normal(size=src_layout.sector_size))

            ci_full = krdm_helper.embed_ksector_ci_to_full(
                fcivec, norb, nelec, nkpts, target_k=target_k,
                link_index=src_layout.link_index)

            for k in range(nkpts):
                for p in range(ncas):
                    orb = k * ncas + p
                    for spin in (0, 1):
                        with self.subTest(cre=cre, target_k=target_k,
                                          k=k, p=p, spin=spin):
                            if cre:
                                test, info = sfh.cre_k(
                                    fcivec, norb, nelec, nkpts, target_k,
                                    k, p, spin, return_info=True,
                                    source_link_index=src_layout.link_index)
                            else:
                                test, info = sfh.des_k(
                                    fcivec, norb, nelec, nkpts, target_k,
                                    k, p, spin, return_info=True,
                                    source_link_index=src_layout.link_index)

                            ref_full, target_nelec = _apply_full_op(
                                ci_full, norb, nelec, orb, spin, cre=cre)
                            target_layout = sfh.make_k_sector_layout(
                                norb, target_nelec, nkpts,
                                target_k=info['target_k'])
                            ref = krdm_helper.extract_ksector_ci_from_full(
                                ref_full, norb, target_nelec, nkpts,
                                target_k=info['target_k'],
                                link_index=target_layout.link_index)

                            self.assertEqual(info['nelec'], target_nelec)
                            self.assertEqual(test.shape, ref.shape)
                            self.assertTrue(np.allclose(test, ref))


if __name__ == "__main__":
    unittest.main()
