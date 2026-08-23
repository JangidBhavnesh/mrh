#!/usr/bin/env python

"""Tests for momentum-sector FCI reduced density matrices."""

import unittest

import numpy as np
import scipy.linalg

from pyscf import fci

from mrh.my_pyscf.pbc.fci import direct_spin1_cplx
from mrh.my_pyscf.pbc.fci import direct_spin1_kfci
from mrh.my_pyscf.pbc.fci import kcistrings
from mrh.my_pyscf.pbc.fci import krdm_helper
from mrh.my_pyscf.pbc.fci.direct_spin1_kfci import _unpack
from mrh.my_pyscf.pbc.fci.kcistrings import (
    gen_k_sector_linkstr_info,
    gen_k_sector_maps,
)


class KFCIHelperFunctions:
    """Convert integrals and CI vectors between sector and full layouts."""

    @staticmethod
    def h1e_k_to_full(h1e_k):
        """Arrange k-space one-electron integrals as a full matrix."""
        return scipy.linalg.block_diag(*h1e_k)

    @staticmethod
    def eri_k_to_full(eri_k):
        """Arrange momentum-conserving k-space integrals as a full tensor."""
        nkpts, ncas = eri_k.shape[0], eri_k.shape[-1]
        norb = nkpts * ncas
        kmom = kcistrings.make_kpoint_momentum(nkpts)
        eri_full = np.zeros((norb,) * 4, dtype=eri_k.dtype)
        for kp, kq, kr in np.ndindex(nkpts, nkpts, nkpts):
            ks = int(kmom.kconserv[kp, kq, kr])
            p0, q0, r0, s0 = (
                kp * ncas, kq * ncas, kr * ncas, ks * ncas)
            eri_full[
                p0:p0 + ncas, q0:q0 + ncas,
                r0:r0 + ncas, s0:s0 + ncas] = eri_k[kp, kq, kr]
        return eri_full

    @staticmethod
    def eri_full_to_k(eri_full, nkpts, ncas):
        """Extract momentum-conserving blocks from a full ERI tensor."""
        eri_view = eri_full.reshape((nkpts, ncas) * 4)
        kmom = kcistrings.make_kpoint_momentum(nkpts)
        eri_k = np.zeros(
            (nkpts,) * 3 + (ncas,) * 4, dtype=eri_full.dtype)
        for kp, kq, kr in np.ndindex(nkpts, nkpts, nkpts):
            ks = int(kmom.kconserv[kp, kq, kr])
            eri_k[kp, kq, kr] = eri_view[
                kp, :, kq, :, kr, :, ks, :]
        return eri_k

    @staticmethod
    def get_ksector_info(norb, nelec, nkpts, target_k):
        """Generate momentum-sector string maps and block information."""
        link_indexa, link_indexb = _unpack(
            norb, nelec, None, nkpts)
        straid_k, strbid_k = gen_k_sector_maps(
            link_indexa, link_indexb, nkpts)[:2]
        blocks = gen_k_sector_linkstr_info(
            link_indexa, link_indexb, nkpts, target_k)
        return link_indexa, link_indexb, straid_k, strbid_k, blocks

    @staticmethod
    def embed_sector_fcivec_to_full_ci(
            fcivec_k, blocks, straid_k, strbid_k, nstra, nstrb):
        """Embed a packed momentum-sector vector in the full CI table."""
        ci_full = np.zeros((nstra, nstrb), dtype=fcivec_k.dtype)
        for block in blocks:
            ka, kb, na, nb, offset, size = map(int, block)
            ci_block = fcivec_k[offset:offset + size].reshape(na, nb)
            ci_full[np.ix_(straid_k[ka], strbid_k[kb])] = ci_block
        return ci_full

    def random_ksector_fcivec(self, nkpts, ncas, nelec, target_k=0,
                              seed=12):
        """Generate a normalized random sector vector and its maps."""
        rng = np.random.default_rng(seed)
        norb = nkpts * ncas
        sector_info = self.get_ksector_info(
            norb, nelec, nkpts, target_k)
        link_indexa, link_indexb, straid_k, strbid_k, blocks = sector_info
        sector_size = int(blocks[:, 5].sum())
        fcivec_k = (rng.normal(size=sector_size)
                    + 1j * rng.normal(size=sector_size))
        fcivec_k /= np.linalg.norm(fcivec_k)
        return (
            fcivec_k, link_indexa, link_indexb, straid_k, strbid_k,
            blocks)

    def embed_random_ksector_fcivec_to_full_ci(
            self, nkpts, ncas, nelec, target_k=0, seed=12):
        """Generate a random sector vector and embed it in full CI space."""
        norb = nkpts * ncas
        sector_data = self.random_ksector_fcivec(
            nkpts, ncas, nelec, target_k=target_k, seed=seed)
        fcivec_k, linka, linkb, straid_k, strbid_k, blocks = sector_data
        nstra = fci.cistring.num_strings(norb, nelec[0])
        nstrb = fci.cistring.num_strings(norb, nelec[1])
        ci_full = self.embed_sector_fcivec_to_full_ci(
            fcivec_k, blocks, straid_k, strbid_k, nstra, nstrb)
        return fcivec_k, ci_full, linka, linkb, straid_k, strbid_k, blocks


def assert_rdm_consistency(rdm1, rdm2, rdm1s, rdm2s, rdm1_direct,
                           rdm1s_direct, nelec):
    """Check internal RDM consistency, traces, and Hermiticity."""
    rdm1a, rdm1b = rdm1s
    rdm2aa, rdm2ab, rdm2bb = rdm2s
    dm1 = rdm1a + rdm1b
    dm2 = rdm2aa + rdm2bb + rdm2ab + rdm2ab.transpose(2, 3, 0, 1)

    np.testing.assert_allclose(rdm1a, rdm1s_direct[0], atol=1e-10)
    np.testing.assert_allclose(rdm1b, rdm1s_direct[1], atol=1e-10)
    np.testing.assert_allclose(dm1.conj().T, rdm1_direct, atol=1e-10)
    np.testing.assert_allclose(dm1.conj().T, rdm1, atol=1e-10)
    np.testing.assert_allclose(dm2, rdm2, atol=1e-10)

    np.testing.assert_allclose(np.trace(rdm1a), nelec[0], atol=1e-10)
    np.testing.assert_allclose(np.trace(rdm1b), nelec[1], atol=1e-10)
    np.testing.assert_allclose(np.trace(rdm1), sum(nelec), atol=1e-10)
    np.testing.assert_allclose(rdm1a, rdm1a.conj().T, atol=1e-10)
    np.testing.assert_allclose(rdm1b, rdm1b.conj().T, atol=1e-10)
    np.testing.assert_allclose(rdm1, rdm1.conj().T, atol=1e-10)

    for same_spin_rdm in (rdm2aa, rdm2bb):
        np.testing.assert_allclose(
            same_spin_rdm, same_spin_rdm.transpose(1, 0, 3, 2).conj(),
            atol=1e-10)
        np.testing.assert_allclose(
            same_spin_rdm, same_spin_rdm.transpose(2, 3, 0, 1),
            atol=1e-8)
    np.testing.assert_allclose(
        rdm2ab, rdm2ab.transpose(1, 0, 3, 2).conj(), atol=1e-10)

    np.testing.assert_allclose(
        np.einsum("ppqq->", rdm2aa), nelec[0] * (nelec[0] - 1),
        atol=1e-6)
    np.testing.assert_allclose(
        np.einsum("ppqq->", rdm2bb), nelec[1] * (nelec[1] - 1),
        atol=1e-6)
    np.testing.assert_allclose(
        np.einsum("ppqq->", rdm2ab), nelec[0] * nelec[1], atol=1e-6)


def assert_same_rdm(actual, reference):
    """Compare two complex RDMs."""
    np.testing.assert_allclose(actual, reference, atol=1e-10, rtol=1e-10)


class KnownValues(unittest.TestCase):

    CASES = [
        (2, 2, (1, 1)),
        (2, 2, (2, 0)),
        (2, 2, (0, 2)),
        (2, 2, (2, 1)),
        (3, 2, (1, 1)),
    ]

    # An even electron count permits odd spin multiplicities, while an odd
    # electron count permits even spin multiplicities.  The random vectors do
    # not need to be spin eigenfunctions for this parity distinction.
    MULTIPLICITY_PARITY_CASES = [
        ("odd", 2, 2, (1, 1)),
        ("even", 2, 2, (2, 1)),
    ]

    def test_krdm12_and_krdm12s(self):
        """Check RDM properties and consistency among public functions."""
        helper = KFCIHelperFunctions()
        for nkpts, ncas, nelec in self.CASES:
            for target_k in range(nkpts):
                with self.subTest(
                        nkpts=nkpts, ncas=ncas, nelec=nelec,
                        target_k=target_k):
                    norb = nkpts * ncas
                    fcivec = helper.random_ksector_fcivec(
                        nkpts, ncas, nelec, target_k=target_k)[0]
                    solver = direct_spin1_kfci.FCISolver(
                        nkpts=nkpts, target_k=target_k)
                    rdm1, rdm2 = solver.make_rdm12(
                        fcivec.copy(), norb, nelec, reorder=True)
                    rdm1s, rdm2s = solver.make_rdm12s(
                        fcivec.copy(), norb, nelec)
                    rdm1s_direct = solver.make_rdm1s(
                        fcivec.copy(), norb, nelec)
                    rdm1_direct = solver.make_rdm1(
                        fcivec.copy(), norb, nelec)
                    assert_rdm_consistency(
                        rdm1, rdm2, rdm1s, rdm2s, rdm1_direct,
                        rdm1s_direct, nelec)

    def test_direct_vs_embedded_ref_for_multiplicity_parities(self):
        """Compare direct and reference RDMs for odd/even multiplicities."""
        helper = KFCIHelperFunctions()
        for parity, nkpts, ncas, nelec in self.MULTIPLICITY_PARITY_CASES:
            for target_k in range(nkpts):
                for reorder in (False, True):
                    with self.subTest(
                            multiplicity_parity=parity, nkpts=nkpts,
                            ncas=ncas, nelec=nelec, target_k=target_k,
                            reorder=reorder):
                        norb = nkpts * ncas
                        fcivec = helper.random_ksector_fcivec(
                            nkpts, ncas, nelec, target_k=target_k)[0]

                        direct1s = krdm_helper.make_rdm1s(
                            fcivec, norb, nelec, nkpts,
                            target_k=target_k)
                        reference1s = krdm_helper.make_rdm1s_ref(
                            fcivec, norb, nelec, nkpts,
                            target_k=target_k)
                        for actual, reference in zip(
                                direct1s, reference1s):
                            assert_same_rdm(actual, reference)

                        direct12s = krdm_helper.make_rdm12s(
                            fcivec, norb, nelec, nkpts,
                            target_k=target_k, reorder=reorder)
                        reference12s = krdm_helper.make_rdm12s_ref(
                            fcivec, norb, nelec, nkpts,
                            target_k=target_k, reorder=reorder)
                        for actual_group, reference_group in zip(
                                direct12s, reference12s):
                            for actual, reference in zip(
                                    actual_group, reference_group):
                                assert_same_rdm(actual, reference)

                        direct12 = krdm_helper.make_rdm12(
                            fcivec, norb, nelec, nkpts,
                            target_k=target_k, reorder=reorder)
                        reference12 = krdm_helper.make_rdm12_ref(
                            fcivec, norb, nelec, nkpts,
                            target_k=target_k, reorder=reorder)
                        for actual, reference in zip(
                                direct12, reference12):
                            assert_same_rdm(actual, reference)

    def test_direct_rdm_energy_matches_kcasci_active_space_energy(self):
        """Match the direct-RDM energy to the kCASCI FCI energy path."""
        nkpts, ncas, nelec, target_k = 2, 2, (1, 1), 0
        norb = nkpts * ncas
        rng = np.random.default_rng(91)
        helper = KFCIHelperFunctions()
        data = helper.embed_random_ksector_fcivec_to_full_ci(
            nkpts, ncas, nelec, target_k=target_k)
        fcivec, ci_full_ref, linka, linkb, *_ = data

        ci_full_c = krdm_helper.embed_ksector_ci_to_full(
            fcivec.copy(), norb, nelec, nkpts, target_k=target_k,
            link_index=(linka, linkb))
        np.testing.assert_allclose(ci_full_c, ci_full_ref, atol=1e-12)

        solver = direct_spin1_kfci.FCISolver(
            nkpts=nkpts, target_k=target_k)
        rdm12_k = solver.make_rdm12(
            fcivec.copy(), norb, nelec, reorder=True)
        rdm12_ref = direct_spin1_cplx.make_rdm12(
            ci_full_ref.copy(), norb, nelec, reorder=True)
        for actual, reference in zip(rdm12_k, rdm12_ref):
            assert_same_rdm(actual, reference)

        h1e_k = (rng.normal(size=(nkpts, ncas, ncas))
                 + 1j * rng.normal(size=(nkpts, ncas, ncas)))
        h1e_k = 0.5 * (h1e_k + h1e_k.transpose(0, 2, 1).conj())

        eri_k = (rng.normal(size=(nkpts,) * 3 + (ncas,) * 4)
                 + 1j * rng.normal(size=(nkpts,) * 3 + (ncas,) * 4))
        eri_full = helper.eri_k_to_full(eri_k)
        eri_full = 0.5 * (
            eri_full + eri_full.transpose(2, 3, 0, 1).conj())
        eri_full = 0.5 * (
            eri_full + eri_full.transpose(1, 0, 3, 2).conj())
        eri_full = 0.5 * (
            eri_full + eri_full.transpose(3, 2, 1, 0).conj())
        eri_k = helper.eri_full_to_k(eri_full, nkpts, ncas)
        eri_full = helper.eri_k_to_full(eri_k)

        rdm1_raw, rdm2_raw = solver.make_rdm12(
            fcivec.copy(), norb, nelec, reorder=False)
        e_rdm = (
            np.einsum(
                "ij,ji", helper.h1e_k_to_full(h1e_k), rdm1_raw,
                optimize=True)
            + np.einsum(
                "ijkl,ijkl", eri_full, rdm2_raw, optimize=True))
        e_direct = solver.energy(h1e_k, eri_k, fcivec, norb, nelec)
        np.testing.assert_allclose(e_rdm, e_direct, atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()
