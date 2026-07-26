#!/bin/bash
import unittest
import numpy as np
import scipy

from pyscf import fci

from mrh.my_pyscf.pbc.fci import direct_spin1_cplx
from mrh.my_pyscf.pbc.fci import direct_spin1_kfci
from mrh.my_pyscf.pbc.fci import krdm_helper
from mrh.my_pyscf.pbc.fci import kcistrings
from mrh.my_pyscf.pbc.fci.direct_spin1_kfci import _unpack
from mrh.my_pyscf.pbc.fci.kcistrings import gen_k_sector_maps, gen_k_sector_linkstr_info

# Author: Bhavnesh Jangid

'''
Testing k-FCI RDM construction.

There are four make_rdm functions
1. make_rdm1s (spin-separated 1-RDM)
2. make_rdm12s (spin-separated 1-RDM and 2-RDM)
3. make_rdm1 (spin-summed 1-RDM)
4. make_rdm12 (spin-summed 2-RDM)
'''

class kFCIHelperFunctions:
    '''
    This class contains helper functions to test the k-FCI implementation. These functions are used to
    arrange the 1e/2e integrals and CI vectors between the k-space format and the full format, and to 
    compare the results from the k-FCI code with the full cplx-FCI code.
    '''
    def __init__(self):
        pass
    
    def h1e_k_to_full(self, h1e_k):
        '''
        Arrange k-space 1e integrals to full 1e integrals. (still in k-space only)
        '''
        h1e_full = scipy.linalg.block_diag(*h1e_k)
        return h1e_full
    
    def eri_k_to_full(self, eri_k):
        '''
        Arrange k-space 2e integrals to full 2e integrals. (still in k-space only)
        '''
        nkpts, ncas = eri_k.shape[0], eri_k.shape[-1]
        norb = nkpts * ncas
        kmom = kcistrings.make_kpoint_momentum(nkpts)
        eri_full = np.zeros((norb, norb, norb, norb), dtype=eri_k.dtype)
        for kp, kq, kr in np.ndindex(nkpts, nkpts, nkpts):
            ks = int(kmom.kconserv[kp, kq, kr])
            P, Q, R, S = kp * ncas, kq * ncas, kr * ncas, ks * ncas
            eri_full[P:P + ncas, Q:Q + ncas, R:R + ncas, S:S + ncas] = eri_k[kp, kq, kr]
        return eri_full

    def eri_full_to_k(self, eri_full, nkpts, ncas):
        '''
        Arrange full 2e integrals to k-space 2e integrals.
        '''
        ef = eri_full.reshape(nkpts, ncas, nkpts, ncas, nkpts, ncas, nkpts, ncas)
        kmom = kcistrings.make_kpoint_momentum(nkpts)
        eri_k = np.zeros((nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas), dtype=eri_full.dtype)
        for kp, kq, kr in np.ndindex(nkpts, nkpts, nkpts):
            ks = int(kmom.kconserv[kp, kq, kr])
            eri_k[kp, kq, kr] = ef[kp, :, kq, :, kr, :, ks, :]
        return eri_k

    def get_ksector_info(self, norb, nelec, nkpts, target_k):
        '''
        Generate the k-sector string maps and block information.
        '''
        link_indexa, link_indexb = _unpack(norb, nelec, None, nkpts)
        straid_k, strbid_k = gen_k_sector_maps(link_indexa, link_indexb, nkpts)[:2]
        blocks = gen_k_sector_linkstr_info(link_indexa, link_indexb, nkpts, target_k)
        return link_indexa, link_indexb, straid_k, strbid_k, blocks

    def embed_sector_fcivec_to_full_ci(self, fcivec_k, blocks, straid_k, strbid_k, 
                                    nstra_total, nstrb_total):
        '''
        In this function, I will create the full CI vector (as zeros) and then only
        fill in the specific block corresponding to each (ka, kb) sector with the sector-wise CI vector.
        '''
        
        ci_full = np.zeros( (nstra_total, nstrb_total), dtype=fcivec_k.dtype)

        for blk in blocks:
            ka, kb, nstra, nstrb, offset, size = map(int, blk)
            block_ka_kb = fcivec_k[offset:offset + size].reshape(nstra, nstrb)

            astrs = straid_k[ka]
            bstrs = strbid_k[kb]

            ci_full[np.ix_(astrs, bstrs)] = block_ka_kb

        return ci_full

    def extract_sector_from_full_ci(self, ci_full, blocks, straid_k, strbid_k):
        '''
        This function extracts the sector-wise CI vector from the full CI vector.
        '''

        sector_size = int(blocks[:, 5].sum())

        fcivec_k = np.zeros(sector_size, dtype=ci_full.dtype)

        for blk in blocks:
            ka, kb, nstra, nstrb, offset, size = map(int, blk)
        
            astrs = straid_k[ka]
            bstrs = strbid_k[kb]

            block = ci_full[np.ix_(astrs, bstrs)]

            fcivec_k[offset:offset + size] = block.reshape(-1)

        return fcivec_k

    def random_ksector_fcivec(self, nkpts, ncas, nelec, target_k=0, seed=12):
        '''
        Generate a random normalized k-sector CI vector and the corresponding maps.
        '''
        rng = np.random.default_rng(seed)
        norb = nkpts * ncas
        link_indexa, link_indexb, straid_k, strbid_k, blocks = \
            self.get_ksector_info(norb, nelec, nkpts, target_k)
        sector_size = int(blocks[:, 5].sum())

        fcivec_k = (rng.normal(size=sector_size) 
                    + 1j * rng.normal(size=sector_size))
        fcivec_k /= np.linalg.norm(fcivec_k)

        return fcivec_k, link_indexa, link_indexb, straid_k, strbid_k, blocks

    def embed_random_ksector_fcivec_to_full_ci(self, nkpts, ncas, nelec, target_k=0, seed=12):
        '''
        Generate a random normalized k-sector CI vector and embed it to the full CI matrix.
        '''
        norb = nkpts * ncas
        fcivec_k, link_indexa, link_indexb, straid_k, strbid_k, blocks = \
            self.random_ksector_fcivec(nkpts, ncas, nelec, target_k=target_k, seed=seed)

        nstra_total = fci.cistring.num_strings(norb, nelec[0])
        nstrb_total = fci.cistring.num_strings(norb, nelec[1])
        ci_full = self.embed_sector_fcivec_to_full_ci(
            fcivec_k, blocks, straid_k, strbid_k, nstra_total, nstrb_total)

        return fcivec_k, ci_full, link_indexa, link_indexb, straid_k, strbid_k, blocks

def _check_kRDM_and_kRDMs_thoroughly(rdm1, rdm2, rdm1a, rdm1b, rdm2aa, rdm2ab, rdm2bb,
                                     rdm1_from_make_rdm1, rdm1a_from_make_rdm1s, 
                                     rdm1b_from_make_rdm1s, nelecas):
    dm1 = rdm1a + rdm1b
    dm2 = rdm2aa + rdm2bb + rdm2ab + rdm2ab.transpose(2,3,0,1)

    # Check the consistency of make_rdm1s and make_rdm12s.
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

                    rdm1, rdm2 = cisolver.make_rdm12(fcivec_k.copy(), norb, nelecas, reorder=True)
                    (rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2bb) = \
                        cisolver.make_rdm12s(fcivec_k.copy(), norb, nelecas)
                    rdm1a_from_make_rdm1s, rdm1b_from_make_rdm1s = \
                        cisolver.make_rdm1s(fcivec_k.copy(), norb, nelecas, link_index=None)
                    rdm1_from_make_rdm1 = \
                        cisolver.make_rdm1(fcivec_k.copy(), norb, nelecas, link_index=None)
                    
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

                    rdm1s_k = cisolver_k.make_rdm1s(fcivec_k.copy(), norb, nelecas)
                    rdm1s_ref = direct_spin1_cplx.make_rdm1s(ci_full.copy(), norb, nelecas)

                    _compare_two_RDM(rdm1s_k[0], rdm1s_ref[0])
                    _compare_two_RDM(rdm1s_k[1], rdm1s_ref[1])

                    rdm1_k = cisolver_k.make_rdm1(fcivec_k.copy(), norb, nelecas)
                    rdm1_ref = direct_spin1_cplx.make_rdm1(ci_full.copy(), norb, nelecas)

                    _compare_two_RDM(rdm1_k, rdm1_ref)

                    rdm12s_k = cisolver_k.make_rdm12s(fcivec_k.copy(), norb, nelecas)
                    rdm12s_ref = direct_spin1_cplx.make_rdm12s(ci_full.copy(), norb, nelecas)

                    for dmA, dmB in zip(rdm12s_k[0], rdm12s_ref[0]):
                        _compare_two_RDM(dmA, dmB)

                    for dmA, dmB in zip(rdm12s_k[1], rdm12s_ref[1]):
                        _compare_two_RDM(dmA, dmB)

                    rdm1_k, rdm2_k = cisolver_k.make_rdm12(fcivec_k.copy(), norb, nelecas)
                    rdm1_ref, rdm2_ref = direct_spin1_cplx.make_rdm12(ci_full.copy(), norb, nelecas)

                    _compare_two_RDM(rdm1_k, rdm1_ref)
                    _compare_two_RDM(rdm2_k, rdm2_ref)

    def test_krdm_c_backend_and_ham_energy(self):
        '''
        Test the C k-sector CI embedding used by k-RDMs and verify that the
        unreordered RDMs reproduce the current raw k-FCI Hamiltonian energy.
        '''
        nkpts = 2
        ncas = 2
        nelec = (1, 1)
        target_k = 0
        norb = nkpts * ncas

        rng = np.random.default_rng(91)
        helper = kFCIHelperFunctions()
        fcivec_k, ci_full_ref, link_indexa, link_indexb, straid_k, strbid_k, blocks = \
            helper.embed_random_ksector_fcivec_to_full_ci(
                nkpts, ncas, nelec, target_k=target_k, seed=12)

        ci_full_c = krdm_helper.embed_ksector_ci_to_full(
            fcivec_k.copy(), norb, nelec, nkpts, target_k=target_k,
            link_index=(link_indexa, link_indexb))
        np.testing.assert_allclose(ci_full_c, ci_full_ref,
                                   atol=1e-12, rtol=1e-12)

        cisolver_k = direct_spin1_kfci.FCISolver(nkpts=nkpts, target_k=target_k)
        rdm1_k, rdm2_k = cisolver_k.make_rdm12(
            fcivec_k.copy(), norb, nelec, reorder=True)
        rdm1_ref, rdm2_ref = direct_spin1_cplx.make_rdm12(
            ci_full_ref.copy(), norb, nelec, reorder=True)
        _compare_two_RDM(rdm1_k, rdm1_ref)
        _compare_two_RDM(rdm2_k, rdm2_ref)

        h1e_k = (rng.normal(size=(nkpts, ncas, ncas))
                 + 1j * rng.normal(size=(nkpts, ncas, ncas)))
        for k in range(nkpts):
            h1e_k[k] = 0.5 * (h1e_k[k] + h1e_k[k].conj().T)

        eri_k = (rng.normal(size=(nkpts, nkpts, nkpts,
                                  ncas, ncas, ncas, ncas))
                 + 1j * rng.normal(size=(nkpts, nkpts, nkpts,
                                         ncas, ncas, ncas, ncas)))
        eri_full = helper.eri_k_to_full(eri_k)
        eri_full = 0.5 * (eri_full + eri_full.transpose(2, 3, 0, 1).conj())
        eri_full = 0.5 * (eri_full + eri_full.transpose(1, 0, 3, 2).conj())
        eri_full = 0.5 * (eri_full + eri_full.transpose(3, 2, 1, 0).conj())
        eri_k = helper.eri_full_to_k(eri_full, nkpts, ncas)
        eri_full = helper.eri_k_to_full(eri_k)

        h1e_full = helper.h1e_k_to_full(h1e_k)
        rdm1_raw, rdm2_raw = cisolver_k.make_rdm12(
            fcivec_k.copy(), norb, nelec, reorder=False)
        e_rdm = (np.einsum("ij,ji", h1e_full, rdm1_raw, optimize=True)
                 + np.einsum("ijkl,ijkl", eri_full, rdm2_raw, optimize=True))
        e_direct = cisolver_k.energy(h1e_k, eri_k, fcivec_k.copy(), norb, nelec)

        np.testing.assert_allclose(e_rdm, e_direct, atol=1e-10, rtol=1e-10)

if __name__ == "__main__":
    unittest.main()
