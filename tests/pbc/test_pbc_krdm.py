#!/bin/bash
import unittest
import numpy as np

from mrh.my_pyscf.pbc.fci import direct_spin1_cplx
from mrh.my_pyscf.pbc.fci import direct_spin1_kfci

try:
    from mrh.tests.pbc.kfci_test_helper import kFCIHelperFunctions
except ImportError:
    from kfci_test_helper import kFCIHelperFunctions

# Author: Bhavnesh Jangid

'''
Testing k-FCI RDM construction.

There are four make_rdm functions
1. make_rdm1s (spin-separated 1-RDM)
2. make_rdm12s (spin-separated 1-RDM and 2-RDM)
3. make_rdm1 (spin-summed 1-RDM)
4. make_rdm12 (spin-summed 2-RDM)
'''

def _check_kRDM_and_kRDMs_thoroughly(rdm1, rdm2, rdm1a, rdm1b, rdm2aa, rdm2ab, rdm2bb,
                                     rdm1_from_make_rdm1, rdm1a_from_make_rdm1s, 
                                     rdm1b_from_make_rdm1s, nelecas):
    dm1 = rdm1a + rdm1b
    dm2 = rdm2aa + rdm2bb + rdm2ab + rdm2ab.transpose(2,3,0,1)

    # Check the consistency of make_rdm1s_py and make_rdm12s_py
    np.testing.assert_allclose(rdm1a, rdm1a_from_make_rdm1s, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(rdm1b, rdm1b_from_make_rdm1s, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(dm1.conj().T, rdm1_from_make_rdm1, atol=1e-10, rtol=1e-10)
    
    # Compare the spin-summed RDMs constructed from spin-separated RDMs
    np.testing.assert_allclose(dm1.conj().T, rdm1, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(dm2, rdm2, atol=1e-10, rtol=1e-10)

    # Compare the trace of 1-RDM with the number of electrons
    np.testing.assert_allclose(np.trace(rdm1a), nelecas[0], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.trace(rdm1b), nelecas[1], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(np.trace(rdm1), sum(nelecas), atol=1e-10, rtol=1e-10)

    # Check the hermiticity of RDMs
    np.testing.assert_allclose(rdm1a, rdm1a.conj().T, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(rdm1b, rdm1b.conj().T, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(rdm1, rdm1.conj().T, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(rdm2aa, rdm2aa.transpose(1,0,3,2).conj(), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(rdm2bb, rdm2bb.transpose(1,0,3,2).conj(), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(rdm2ab, rdm2ab.transpose(1,0,3,2).conj(), atol=1e-10, rtol=1e-10)

    np.testing.assert_allclose(rdm2aa - rdm2aa.transpose(2,3,0,1), 0, atol=1e-8)
    np.testing.assert_allclose(rdm2bb - rdm2bb.transpose(2,3,0,1), 0, atol=1e-8)
    
    # Compare the trace of 2-RDM with the number of electron pairs
    np.testing.assert_allclose(np.einsum("ppqq->", rdm2aa, optimize=True), nelecas[0]*(nelecas[0]-1), atol=1e-6)
    np.testing.assert_allclose(np.einsum("ppqq->", rdm2bb, optimize=True), nelecas[1]*(nelecas[1]-1), atol=1e-6)
    np.testing.assert_allclose(np.einsum("ppqq->", rdm2ab, optimize=True), nelecas[0]*nelecas[1], atol=1e-6)

def _compare_two_RDM(dmA, dmB):
    assert dmA.shape == dmB.shape
    np.testing.assert_allclose(dmA.real, dmB.real, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(dmA.imag, dmB.imag, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(dmA, dmB, atol=1e-10, rtol=1e-10)

class KnownValues(unittest.TestCase):

    def test_krdm12_and_krdm12s(self):
        '''
        Checking few properties of k-RDMs and consistency of make_rdm* functions.
        '''
        test_cases = [ (2, 2, (1, 1)), (2, 2, (2, 0)), (2, 2, (0, 2)), 
                      (2, 2, (2, 1)), (3, 2, (1, 1))]

        helper = kFCIHelperFunctions()

        for nkpts, ncas, nelecas in test_cases:
            for target_k in range(nkpts):
                with self.subTest(nkpts=nkpts, ncas=ncas, nelecas=nelecas, target_k=target_k):
                    norb = nkpts * ncas
                    fcivec_k = helper.random_ksector_fcivec(
                        nkpts, ncas, nelecas, target_k=target_k, seed=12)[0]

                    cisolver = direct_spin1_kfci.FCISolver(nkpts=nkpts, target_k=target_k)

                    rdm1, rdm2 = cisolver.make_rdm12_py(fcivec_k.copy(), norb, nelecas, reorder=True)
                    (rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2bb) = \
                        cisolver.make_rdm12s_py(fcivec_k.copy(), norb, nelecas)
                    rdm1a_from_make_rdm1s, rdm1b_from_make_rdm1s = \
                        cisolver.make_rdm1s_py(fcivec_k.copy(), norb, nelecas, link_index=None)
                    rdm1_from_make_rdm1 = \
                        cisolver.make_rdm1_py(fcivec_k.copy(), norb, nelecas, link_index=None)
                    
                    _check_kRDM_and_kRDMs_thoroughly(
                        rdm1, rdm2, rdm1a, rdm1b, rdm2aa, rdm2ab, rdm2bb,
                        rdm1_from_make_rdm1, rdm1a_from_make_rdm1s, rdm1b_from_make_rdm1s,
                        nelecas)

    def test_krdm_vs_full_cplx_fci_rdm(self):
        '''
        Test k-FCI RDMs against full complex-FCI RDMs after embedding the sector CI vector.
        '''
        test_cases = [ (2, 2, (1, 1)), (2, 2, (2, 0)), (2, 2, (0, 2)), 
                      (2, 2, (2, 1)), (3, 2, (1, 1))]

        helper = kFCIHelperFunctions()

        for nkpts, ncas, nelecas in test_cases:
            for target_k in range(nkpts):
                with self.subTest(nkpts=nkpts, ncas=ncas, nelecas=nelecas, target_k=target_k):
                    norb = nkpts * ncas

                    fcivec_k, ci_full, link_indexa, link_indexb, straid_k, strbid_k, blocks = \
                        helper.embed_random_ksector_fcivec_to_full_ci(
                            nkpts, ncas, nelecas, target_k=target_k, seed=12)

                    cisolver_k = direct_spin1_kfci.FCISolver(nkpts=nkpts, target_k=target_k)

                    rdm1s_k = cisolver_k.make_rdm1s_py(fcivec_k.copy(), norb, nelecas)
                    rdm1s_ref = direct_spin1_cplx.make_rdm1s_py(ci_full.copy(), norb, nelecas)

                    _compare_two_RDM(rdm1s_k[0], rdm1s_ref[0])
                    _compare_two_RDM(rdm1s_k[1], rdm1s_ref[1])

                    rdm1_k = cisolver_k.make_rdm1_py(fcivec_k.copy(), norb, nelecas)
                    rdm1_ref = direct_spin1_cplx.make_rdm1_py(ci_full.copy(), norb, nelecas)

                    _compare_two_RDM(rdm1_k, rdm1_ref)

                    rdm12s_k = cisolver_k.make_rdm12s_py(fcivec_k.copy(), norb, nelecas)
                    rdm12s_ref = direct_spin1_cplx.make_rdm12s_py(ci_full.copy(), norb, nelecas)

                    for dmA, dmB in zip(rdm12s_k[0], rdm12s_ref[0]):
                        _compare_two_RDM(dmA, dmB)

                    for dmA, dmB in zip(rdm12s_k[1], rdm12s_ref[1]):
                        _compare_two_RDM(dmA, dmB)

                    rdm1_k, rdm2_k = cisolver_k.make_rdm12_py(fcivec_k.copy(), norb, nelecas)
                    rdm1_ref, rdm2_ref = direct_spin1_cplx.make_rdm12_py(ci_full.copy(), norb, nelecas)

                    _compare_two_RDM(rdm1_k, rdm1_ref)
                    _compare_two_RDM(rdm2_k, rdm2_ref)

if __name__ == "__main__":
    unittest.main()

