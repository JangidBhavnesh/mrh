import unittest
import numpy as np
import scipy


from pyscf import fci

from mrh.my_pyscf.pbc.fci import direct_spin1_cplx, direct_spin1_cplx_opt
from mrh.my_pyscf.pbc.fci.direct_spin1_kfci import contract_2e_k, _unpack, contract_1e_k
from mrh.my_pyscf.pbc.fci.kcistrings import gen_k_sector_maps, gen_k_sector_linkstr_info

# Author: Bhavnesh Jangid

'''
Testing k-FCI implementation.
'''


class kFCIHelperFunctions:
    '''
    This class contains helper functions to test the k-FCI contract_2e_k implementation. These functions are used to
    arrange the 2e integrals and CI vectors between the k-space format and the full format, and to 
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
        eri_full = np.zeros((norb, norb, norb, norb), dtype=eri_k.dtype)
        for kp, kq, kr in np.ndindex(nkpts, nkpts, nkpts):
            ks = (kp - kq + kr) % nkpts
            P, Q, R, S = kp * ncas, kq * ncas, kr * ncas, ks * ncas
            eri_full[P:P + ncas, Q:Q + ncas, R:R + ncas, S:S + ncas] = eri_k[kp, kq, kr]
        return eri_full

    def eri_full_to_k(self, eri_full, nkpts, ncas):
        '''
        Arrange full 2e integrals to k-space 2e integrals.
        '''
        ef = eri_full.reshape(nkpts, ncas, nkpts, ncas, nkpts, ncas, nkpts, ncas)
        eri_k = np.zeros((nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas), dtype=eri_full.dtype)
        for kp, kq, kr in np.ndindex(nkpts, nkpts, nkpts):
            ks = (kp - kq + kr) % nkpts
            eri_k[kp, kq, kr] = ef[kp, :, kq, :, kr, :, ks, :]
        return eri_k

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
        
    def compare_contract_2e_k(self, nkpts=3, ncas=3, nelec=(2, 1), target_k=0, seed=12):
        '''
        This function compares the sigma vector obtained from the k-FCI code with the sigma vector obtained
        from the full cplx-FCI code, for a randomly generated CI vector and 2e integrals. The comparison is done for nkpts > 1.
        '''
        rng = np.random.default_rng(seed)

        norb = nkpts * ncas

        link_index = None

        link_indexa, link_indexb = _unpack( norb, nelec, link_index, nkpts, )
        
        straid_k, strbid_k = gen_k_sector_maps(link_indexa, link_indexb, nkpts, )[:2]

        blocks = gen_k_sector_linkstr_info( link_indexa, link_indexb, nkpts, target_k, )

        sector_size = int(blocks[:, 5].sum())

        fcivec_k = ( rng.normal(size=sector_size) + 1j * rng.normal(size=sector_size))
        fcivec_k /= np.linalg.norm(fcivec_k)

        eri_k = ( rng.normal( size=(nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas) ) + 
                1j * rng.normal( size=(nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas) ) )

        eri_full = self.eri_k_to_full(eri_k)

        eri_full = 0.5 * ( eri_full + eri_full.transpose(2, 3, 0, 1))

        eri_k = self.eri_full_to_k( eri_full, nkpts, ncas)

        eri_full = self.eri_k_to_full(eri_k)

        nstra_total = fci.cistring.num_strings(norb, nelec[0])
        nstrb_total = fci.cistring.num_strings(norb, nelec[1])

        ci_full = self.embed_sector_fcivec_to_full_ci( fcivec_k, blocks, straid_k, strbid_k, 
                                                nstra_total, nstrb_total)

        sigma_k = contract_2e_k(eri_k, fcivec_k, norb, nelec, nkpts, target_k)
        sigma_full = direct_spin1_cplx_opt.contract_2e(eri_full, ci_full, norb, nelec)

        sigma_ref_k = self.extract_sector_from_full_ci(sigma_full, blocks, straid_k, strbid_k, )

        diff = sigma_k - sigma_ref_k

        return sigma_k, sigma_ref_k, diff
        
    def compare_contract_1e_k_nkpts1_vs_mol(self, ncas=8, nelec=(2, 2), seed=12):
        '''
        Compare contract_1e_k against molecular contract_1e in the nkpts = 1 limit.
        '''
        rng = np.random.default_rng(seed)

        nkpts = 1
        norb = nkpts * ncas
        target_k = 0

        na = fci.cistring.num_strings(norb, nelec[0])
        nb = fci.cistring.num_strings(norb, nelec[1])

        ci0 = (rng.normal(size=(na, nb)) + 1j * rng.normal(size=(na, nb)))
        ci0 /= np.linalg.norm(ci0)

        fcivec_k = np.asarray(ci0.reshape(-1), order="C")

        h1e_k = ( rng.normal(size=(nkpts, ncas, ncas)) + 
                 1j * rng.normal(size=(nkpts, ncas, ncas)))
        
        for k in range(nkpts):
            h1e_k[k] = 0.5 * (h1e_k[k] + h1e_k[k].conj().T)

        h1e_full = self.h1e_k_to_full(h1e_k)

        sigma_k = contract_1e_k(h1e_k, fcivec_k, norb, nelec, nkpts, target_k, link_index=None)

        sigma_ref = direct_spin1_cplx.contract_1e(h1e_full, ci0, norb, nelec, link_index=None).reshape(-1, order="C")

        diff = sigma_k - sigma_ref

        return sigma_k, sigma_ref, diff

    def compare_contract_1e_k_nkpts_gt1_vs_full_mol(self, nkpts=3, ncas=3, nelec=(2, 1), target_k=0, seed=12):
        '''
        Compare contract_1e_k against full molecular contract_1e by embedding the k-sector CI vector
        into the full determinant space and extracting the target sector after contraction.
        '''
        rng = np.random.default_rng(seed)

        norb = nkpts * ncas

        link_index = None
        link_indexa, link_indexb = _unpack(norb, nelec, link_index, nkpts)

        straid_k, strbid_k, str2tot_a, str2tot_b = gen_k_sector_maps(
            link_indexa,
            link_indexb,
            nkpts,
        )

        blocks = gen_k_sector_linkstr_info(link_indexa, link_indexb, nkpts, target_k)

        sector_size = int(blocks[:, 5].sum())

        fcivec_k = ( rng.normal(size=sector_size) 
                    + 1j * rng.normal(size=sector_size))

        fcivec_k /= np.linalg.norm(fcivec_k)

        h1e_k = (rng.normal(size=(nkpts, ncas, ncas)) 
                 + 1j * rng.normal(size=(nkpts, ncas, ncas)))

        for k in range(nkpts):
            h1e_k[k] = 0.5 * (h1e_k[k] + h1e_k[k].conj().T)

        h1e_full = self.h1e_k_to_full(h1e_k)

        nstra_total = fci.cistring.num_strings(norb, nelec[0])
        nstrb_total = fci.cistring.num_strings(norb, nelec[1])

        ci_full = self.embed_sector_fcivec_to_full_ci( fcivec_k, blocks, straid_k, strbid_k, 
                                                      nstra_total, nstrb_total)

        sigma_k = contract_1e_k( h1e_k, fcivec_k, norb, nelec, nkpts, target_k, link_index=None)

        sigma_full = direct_spin1_cplx.contract_1e( h1e_full, ci_full, norb, nelec, link_index=None, )

        sigma_ref_k = self.extract_sector_from_full_ci( sigma_full, blocks, straid_k, strbid_k)

        diff = sigma_k - sigma_ref_k

        return sigma_k, sigma_ref_k, diff


class KnownValues(unittest.TestCase):

    def test_contract_1e_k_as_limit_to_nk1(self):
        helper = kFCIHelperFunctions()

        nelec_cases = [(1, 1), (2, 0), (0, 2), (2, 2), (3, 3), (4, 4)]

        for nelec in nelec_cases:
            with self.subTest(nelec=nelec):
                sigma_k, sigma_mol_flat, diff = (
                    helper.compare_contract_1e_k_nkpts1_vs_mol( ncas=8, nelec=nelec, seed=12, ) )
                self.assertEqual( sigma_k.shape, sigma_mol_flat.shape, )

                self.assertTrue( np.allclose( sigma_k, sigma_mol_flat, atol=1e-12, rtol=1e-12, ))

    def test_contract_1e_k(self):
        helper = kFCIHelperFunctions()

        test_cases = [ (2, 3, (1, 1)), (2, 3, (2, 0)), (2, 3, (0, 2)), (2, 3, (2, 1)), 
                      (2, 3, (2, 2)), (3, 2, (1, 1)), (3, 2, (2, 1))]

        for nkpts, ncas, nelec in test_cases:
            for target_k in range(nkpts):
                with self.subTest(nkpts=nkpts, ncas=ncas, nelec=nelec, target_k=target_k):
                    sigma_k, sigma_ref_k, diff = ( 
                        helper.compare_contract_1e_k_nkpts_gt1_vs_full_mol( nkpts=nkpts, ncas=ncas, 
                                                                           nelec=nelec, target_k=target_k, seed=12, ) )
                    self.assertEqual( sigma_k.shape, sigma_ref_k.shape, )
                    self.assertTrue( np.allclose( sigma_k, sigma_ref_k, atol=1e-12, rtol=1e-12))

    def test_contract_2e_k_as_limit_to_nk1(self):
        # In case of nkpts=1, the k-FCI code should reduce to the cplx-FCI.        
        nkpts = 1
        ncas = 6
        norb = nkpts * ncas
        target_k = 0

        nelec_cases = [(1, 1), (2, 0), (0, 2), (3, 3), (4, 4), (6, 0), (0, 6), (6, 2), (2, 6)]

        for nelec in nelec_cases:
            with self.subTest(nelec=nelec):
                rng = np.random.default_rng(12)
                na = fci.cistring.num_strings(norb, nelec[0])
                nb = fci.cistring.num_strings(norb, nelec[1])

                # Initial CI vector
                ci0 = (rng.normal(size=(na, nb)) + 1j * rng.normal(size=(na, nb)))
                ci0 /= np.linalg.norm(ci0)

                # Making it C-contiguous
                fcivec_k = np.asarray(ci0.reshape(-1), order="C")
                
                # Generate the 2e integrals and symmetrize it.
                eris = (rng.normal(size=(norb, norb, norb, norb)) 
                           + 1j * rng.normal(size=(norb, norb, norb, norb)))
                eris = 0.5 * (eris + eris.transpose(2, 3, 0, 1))

                # Reshaping the 2e integrals to the k-FCI format
                eri_k = np.zeros((nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas), dtype=eris.dtype)
                eri_k[0, 0, 0] = eris

                # Compute sigma = H * ci0 using both the k-FCI and cplx-FCI code.
                sigma_k = contract_2e_k(eri_k, fcivec_k, norb, nelec, nkpts, target_k)

                sigma_ref = direct_spin1_cplx_opt.contract_2e(eris, ci0, norb, nelec)
                sigma_ref = np.asarray(sigma_ref.ravel(), order="C")

                self.assertEqual(sigma_k.shape, sigma_ref.shape)
                self.assertTrue(np.allclose(sigma_k, sigma_ref, atol=1e-12, rtol=1e-12), 
                                msg=(f"contract_2e_k failed in the limit of nkpts=1 for nelec={nelec}."))

    def test_contract_2e_k(self):
        # Defining a variety of test cases with different nkpts, ncas, and nelec. 
        # The target_k will be varied in the loop (0 to nkpts-1).
        test_cases = [(2, 3, (1, 1)), (2, 3, (2, 0)), (2, 3, (0, 2)), 
                      (2, 3, (2, 1)), (2, 3, (2, 2)), (3, 2, (1, 1)), 
                      (3, 2, (2, 1)), (3, 4, (2, 2)), (4, 2, (1, 1)), 
                      (5, 2, (1, 1))]
        
        for nkpts, ncas, nelec in test_cases:
            for target_k in range(nkpts):
                with self.subTest( nkpts=nkpts, ncas=ncas, nelec=nelec, target_k=target_k):
                    helper = kFCIHelperFunctions()
                    sigma_k, sigma_ref_k, diff = (
                        helper.compare_contract_2e_k(nkpts=nkpts, ncas=ncas, nelec=nelec, 
                                                           target_k=target_k, seed=12))
                    
                    # Compare the shape
                    self.assertEqual(sigma_k.shape, sigma_ref_k.shape)

                    # Now compare the absolute values
                    self.assertTrue(np.allclose(sigma_k, sigma_ref_k, atol=1e-12, rtol=1e-12))

if __name__ == "__main__":
    unittest.main()
