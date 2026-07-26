#!/bin/bash

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


class KnownValues(unittest.TestCase):

    def test_des_k_matches_full_ci_operator(self):
        self._check_k_operator(cre=False)

    def test_cre_k_matches_full_ci_operator(self):
        self._check_k_operator(cre=True)

    def test_make_spectral_poles_from_charged_roots(self):
        nkpts = 2
        ncas = 1
        norb = nkpts * ncas
        nelec = (1, 0)

        roots = sfh.KCASCISpectralRoots(
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

        poles = sfh.make_spectral_poles(
            roots, k_indices=(0, 1), orbital_indices=(0,),
            spins=(0,), min_weight=1e-12)
        labels = {(row['kind'], row['k']): row for row in poles}

        self.assertEqual(set(labels), {('hole', 0), ('particle', 1)})
        self.assertAlmostEqual(labels[('hole', 0)]['omega'], 0.75)
        self.assertAlmostEqual(labels[('hole', 0)]['weight'], 1.0)
        self.assertAlmostEqual(labels[('particle', 1)]['omega'], 0.75)
        self.assertAlmostEqual(labels[('particle', 1)]['weight'], 1.0)

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
