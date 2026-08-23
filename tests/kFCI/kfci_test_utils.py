
import numpy as np
import scipy

from pyscf import fci

from mrh.my_pyscf.pbc.fci import kcistrings
from mrh.my_pyscf.pbc.fci.direct_spin1_kfci import _unpack
from mrh.my_pyscf.pbc.fci.kcistrings import (
    gen_k_sector_linkstr_info,
    gen_k_sector_maps,
)

# Author: Bhavnesh Jangid

"""
Shared layout helpers for momentum-sector kFCI tests.
Instead of duplicating the code, I am using this file to 
provide the same helper fns across the folder.
"""


class KFCIHelperFunctions:
    '''
    Arrange integrals and CI vectors in momentum-sector and full layouts for
    comparisons with the complex-FCI implementation.
    '''

    def __init__(self):
        pass

    def h1e_k_to_full(self, h1e_k):
        '''
        Arrange k-space one-electron integrals as a full matrix.
        '''
        h1e_full = scipy.linalg.block_diag(*h1e_k)
        return h1e_full

    def eri_k_to_full(self, eri_k):
        '''
        Arrange k-space two-electron integrals as a full tensor.
        '''
        nkpts, ncas = eri_k.shape[0], eri_k.shape[-1]
        norb = nkpts * ncas
        kmom = kcistrings.make_kpoint_momentum(nkpts)
        eri_full = np.zeros((norb, norb, norb, norb), dtype=eri_k.dtype)
        for kp, kq, kr in np.ndindex(nkpts, nkpts, nkpts):
            ks = int(kmom.kconserv[kp, kq, kr])
            P, Q, R, S = kp * ncas, kq * ncas, kr * ncas, ks * ncas
            eri_full[P:P + ncas, Q:Q + ncas, R:R +
                     ncas, S:S + ncas] = eri_k[kp, kq, kr]
        return eri_full

    def eri_full_to_k(self, eri_full, nkpts, ncas):
        '''
        Arrange full 2e integrals to k-space 2e integrals.
        '''
        ef = eri_full.reshape( nkpts, ncas, nkpts, ncas, nkpts, ncas, nkpts, ncas)
        kmom = kcistrings.make_kpoint_momentum(nkpts)
        eri_k = np.zeros( (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas), 
                         dtype=eri_full.dtype)
        for kp, kq, kr in np.ndindex(nkpts, nkpts, nkpts):
            ks = int(kmom.kconserv[kp, kq, kr])
            eri_k[kp, kq, kr] = ef[kp, :, kq, :, kr, :, ks, :]
        return eri_k

    def get_ksector_info(self, norb, nelec, nkpts, target_k):
        '''
        Generate the k-sector string maps and block information.
        '''
        link_indexa, link_indexb = _unpack(norb, nelec, None, nkpts)
        straid_k, strbid_k = gen_k_sector_maps(
            link_indexa, link_indexb, nkpts)[:2]
        blocks = gen_k_sector_linkstr_info(
            link_indexa, link_indexb, nkpts, target_k)
        return link_indexa, link_indexb, straid_k, strbid_k, blocks

    def embed_sector_fcivec_to_full_ci(
            self,
            fcivec_k,
            blocks,
            straid_k,
            strbid_k,
            nstra_total,
            nstrb_total):
        '''
        Embed a momentum-sector vector in the full CI table.
        '''

        ci_full = np.zeros((nstra_total, nstrb_total), dtype=fcivec_k.dtype)

        for blk in blocks:
            ka, kb, nstra, nstrb, offset, size = map(int, blk)
            block_ka_kb = fcivec_k[offset:offset + size].reshape(nstra, nstrb)

            astrs = straid_k[ka]
            bstrs = strbid_k[kb]

            ci_full[np.ix_(astrs, bstrs)] = block_ka_kb

        return ci_full

    def extract_sector_from_full_ci(
            self, ci_full, blocks, straid_k, strbid_k):
        '''
        Extract the sector CI vector from a full CI table.
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
        Generate a normalized random sector CI vector and its maps.
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

    def embed_random_ksector_fcivec_to_full_ci(
            self, nkpts, ncas, nelec, target_k=0, seed=12):
        '''
        Generate a normalized random sector vector and embed it in full CI.
        '''
        norb = nkpts * ncas
        fcivec_k, link_indexa, link_indexb, straid_k, strbid_k, blocks = \
            self.random_ksector_fcivec(
                nkpts, ncas, nelec, target_k=target_k, seed=seed)

        nstra_total = fci.cistring.num_strings(norb, nelec[0])
        nstrb_total = fci.cistring.num_strings(norb, nelec[1])
        ci_full = self.embed_sector_fcivec_to_full_ci(
            fcivec_k, blocks, straid_k, strbid_k, nstra_total, nstrb_total)

        return (
            fcivec_k, ci_full, link_indexa, link_indexb, straid_k,
            strbid_k, blocks)
