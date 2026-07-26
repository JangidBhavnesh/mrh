#!/bin/bash

import os
import tempfile
import unittest
import numpy as np

from pyscf.fci import cistring
from pyscf.fci.addons import _unpack_nelec

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


class KnownValues(unittest.TestCase):

    def test_des_k_matches_full_ci_operator(self):
        self._check_k_operator(cre=False)

    def test_cre_k_matches_full_ci_operator(self):
        self._check_k_operator(cre=True)

    def test_c_k_operators_match_python(self):
        if sfh._load_spectral_lib() is None:
            self.skipTest("libpbc_spectral_fn is not built")

        nkpts = 3
        ncas = 2
        norb = nkpts * ncas
        nelec = (2, 1)
        rng = np.random.default_rng(21)

        for cre in (False, True):
            for target_k in range(nkpts):
                layout = sfh.make_k_sector_layout(
                    norb, nelec, nkpts, target_k=target_k)
                fcivec = (rng.normal(size=layout.sector_size)
                          + 1j * rng.normal(size=layout.sector_size))

                for k in range(nkpts):
                    for p in range(ncas):
                        for spin in (0, 1):
                            with self.subTest(cre=cre, target_k=target_k,
                                              k=k, p=p, spin=spin):
                                if cre:
                                    ref, info_ref = sfh.cre_k_py(
                                        fcivec, norb, nelec, nkpts, target_k,
                                        k, p, spin, return_info=True,
                                        source_link_index=layout.link_index)
                                    test, info = sfh.cre_k(
                                        fcivec, norb, nelec, nkpts, target_k,
                                        k, p, spin, return_info=True,
                                        source_link_index=layout.link_index)
                                else:
                                    ref, info_ref = sfh.des_k_py(
                                        fcivec, norb, nelec, nkpts, target_k,
                                        k, p, spin, return_info=True,
                                        source_link_index=layout.link_index)
                                    test, info = sfh.des_k(
                                        fcivec, norb, nelec, nkpts, target_k,
                                        k, p, spin, return_info=True,
                                        source_link_index=layout.link_index)

                                self.assertEqual(info, info_ref)
                                self.assertTrue(np.allclose(test, ref,
                                                            rtol=1e-13,
                                                            atol=1e-13))

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

    def test_c_spectral_function_matches_python(self):
        if sfh._load_spectral_lib() is None:
            self.skipTest("libpbc_spectral_fn is not built")

        roots = _one_alpha_roots()
        poles = sfh.make_spectral_poles(
            roots, k_indices=(0, 1), orbital_indices=(0,),
            spins=(0,), min_weight=1e-12)
        omega = np.linspace(-1.0, 2.0, 173)

        for broadening in ('lorentzian', 'gaussian'):
            with self.subTest(broadening=broadening):
                ref = sfh.make_spectral_function_py(
                    poles, omega_grid=omega, eta=0.07,
                    broadening=broadening, nkpts=2, norb=1,
                    spin_resolved=True, orbital_resolved=True)
                test = sfh.make_spectral_function(
                    poles, omega_grid=omega, eta=0.07,
                    broadening=broadening, nkpts=2, norb=1,
                    spin_resolved=True, orbital_resolved=True)

                for key in ('hole', 'particle', 'total'):
                    self.assertTrue(np.allclose(test['spectra'][key],
                                                ref['spectra'][key],
                                                rtol=1e-13, atol=1e-13))

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

    def _check_k_operator(self, cre=False):
        nkpts = 3
        ncas = 2
        norb = nkpts * ncas
        nelec = (2, 1)
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
