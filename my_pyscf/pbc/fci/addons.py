# !/usr/bin/env python
import numpy as np

from pyscf.fci import addons, cistring

from mrh.my_pyscf.pbc.fci import kcistrings

# Author: Bhavnesh Jangid

"""
Periodic FCI add-ons and compatibility dispatch.
"""

SpinPenaltyFCISolver = addons.SpinPenaltyFCISolver


def get_kfci_integrals(kmc, mo_coeff):
    """Build the active-space effective Hamiltonian in k-space."""
    cell = kmc.cell
    kmf = kmc._scf
    nkpts = kmc.nkpts
    ncore = kmc.ncore
    ncas = kmc.ncas
    nocc = ncore + ncas

    hcore = kmc.get_hcore()
    dtype = np.result_type(hcore, *[mo.dtype for mo in mo_coeff])
    hcore = hcore.astype(dtype)

    mo_core = [mo[:, :ncore] for mo in mo_coeff]
    mo_cas = np.asarray(
        [mo[:, ncore:nocc] for mo in mo_coeff], dtype=dtype)

    # The final total energy is divided by nkpts, so accumulate the nuclear
    # and core contributions with the corresponding cell normalization.
    ecore = kmc.energy_nuc() * nkpts
    if ncore > 0:
        dm_core = np.asarray([
            2.0 * mo_core[kpoint] @ mo_core[kpoint].conj().T
            for kpoint in range(nkpts)
        ], dtype=dtype)
        core_vhf = kmc.get_veff(cell, dm_core, hermi=1, kpts=kmf.kpts)
        fock_core = hcore + 0.5 * core_vhf
        ecore += sum(
            np.einsum("ij,ji", dm_core[kpoint], fock_core[kpoint])
            for kpoint in range(nkpts))
        hcore += core_vhf

    h1e = np.asarray([
        mo_cas[kpoint].conj().T @ hcore[kpoint] @ mo_cas[kpoint]
        for kpoint in range(nkpts)
    ], dtype=dtype)

    # The 1/nkpts factor gives the supercell normalization.
    h2e = kmf.with_df.ao2mo_7d(mo_cas, kpts=kmf.kpts)
    h2e = np.asarray(h2e, dtype=dtype) / nkpts

    # contract_2e follows PySCF direct_spin1.contract_2e conventions, so use
    # the effective one-electron Hamiltonian h1 - J/2 and two-electron tensor
    # h2/2.
    j_eff = np.zeros_like(h1e)
    for kp in range(nkpts):
        for kq in range(nkpts):
            j_eff[kp] += np.einsum("piis->ps", h2e[kp, kq, kq])
    h1e -= 0.5 * j_eff
    h2e *= 0.5
    return h1e, h2e, ecore

def fix_spin(fciobj, shift=0.1, ss=None, **kwargs):
    """Dispatch spin-penalty construction to the matching solver family."""
    from mrh.my_pyscf.pbc.fci import direct_spin1_kfci

    if isinstance(fciobj, direct_spin1_kfci.FCISolver):
        return direct_spin1_kfci.fix_spin(fciobj, shift=shift,
                                          ss=ss, **kwargs)
    return addons.fix_spin(fciobj, shift=shift, ss=ss, **kwargs)

def fix_spin_(fciobj, shift=0.1, ss=None, **kwargs):
    """Apply the appropriate spin-penalty mixin in place."""
    from mrh.my_pyscf.pbc.fci import direct_spin1_kfci

    if isinstance(fciobj, direct_spin1_kfci.FCISolver):
        return direct_spin1_kfci.fix_spin_(fciobj, shift=shift,
                                           ss=ss, **kwargs)
    return addons.fix_spin_(fciobj, shift=shift, ss=ss, **kwargs)

def _unpack_nelec(nelec, spin=None):
    """Normalize electron counts to an alpha/beta tuple."""
    if isinstance(nelec, tuple):
        return nelec[0], nelec[1]
    return addons._unpack_nelec(nelec, spin)


def _unpack_k(norb, nelec, nkpts, link_index=None, spin=None, kmom=None,
              kconserv=None):
    """Generate or validate momentum-aware alpha/beta link indices."""
    assert norb % nkpts == 0
    if link_index is not None:
        assert link_index[0].shape[2] == link_index[1].shape[2] == 8
        return link_index

    neleca, nelecb = _unpack_nelec(nelec, spin)
    norb_k = norb // nkpts
    orb_k = (np.arange(norb, dtype=np.int32) // norb_k).astype(np.int32)
    if spin == 0 and neleca == nelecb:
        link_indexa = link_indexb = kcistrings.gen_linkstr_index_k(
            range(norb), neleca, orb_k, nkpts, kmom=kmom,
            kconserv=kconserv)
    else:
        link_indexa = kcistrings.gen_linkstr_index_k(
            range(norb), neleca, orb_k, nkpts, kmom=kmom,
            kconserv=kconserv)
        link_indexb = kcistrings.gen_linkstr_index_k(
            range(norb), nelecb, orb_k, nkpts, kmom=kmom,
            kconserv=kconserv)
    return link_indexa, link_indexb


def _unpack(norb, nelec, link_index, spin=None):
    """
    Generate molecular link indices when none are supplied.
    Note: this is not exactly the same as the _unpack() in pyscf.fci.addons, because it does not
    store the lower-triangular link indices.
    """
    if link_index is None:
        neleca, nelecb = _unpack_nelec(nelec, spin)
        link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
        return link_indexa, link_indexb
    return link_index


class KFCIHelperFunctions:
    """Convert k-FCI integrals and CI vectors between sector layouts."""

    def eri_k_to_full(self, eri_k):
        """Arrange k-space two-electron integrals as a full tensor."""
        nkpts, ncas = eri_k.shape[0], eri_k.shape[-1]
        norb = nkpts * ncas
        kmom = kcistrings.make_kpoint_momentum(nkpts)
        eri_full = np.zeros((norb, norb, norb, norb), dtype=eri_k.dtype)
        for kp, kq, kr in np.ndindex(nkpts, nkpts, nkpts):
            ks = int(kmom.kconserv[kp, kq, kr])
            p, q, r, s = (kp * ncas, kq * ncas,
                          kr * ncas, ks * ncas)
            eri_full[p:p + ncas, q:q + ncas,
                     r:r + ncas, s:s + ncas] = eri_k[kp, kq, kr]
        return eri_full

    def eri_full_to_k(self, eri_full, nkpts, ncas):
        """Arrange full two-electron integrals as k-space integrals."""
        eri_full_blocks = eri_full.reshape(
            nkpts, ncas, nkpts, ncas, nkpts, ncas, nkpts, ncas)
        kmom = kcistrings.make_kpoint_momentum(nkpts)
        eri_k = np.zeros(
            (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas),
            dtype=eri_full.dtype)
        for kp, kq, kr in np.ndindex(nkpts, nkpts, nkpts):
            ks = int(kmom.kconserv[kp, kq, kr])
            eri_k[kp, kq, kr] = eri_full_blocks[
                kp, :, kq, :, kr, :, ks, :]
        return eri_k

    def get_ksector_info(self, norb, nelec, nkpts, target_k):
        """Generate k-sector string maps and block information."""
        link_indexa, link_indexb = _unpack_k(norb, nelec, nkpts)
        straid_k, strbid_k = kcistrings.gen_k_sector_maps(
            link_indexa, link_indexb, nkpts)[:2]
        blocks = kcistrings.gen_k_sector_linkstr_info(
            link_indexa, link_indexb, nkpts, target_k)
        return link_indexa, link_indexb, straid_k, strbid_k, blocks

    def embed_sector_fcivec_to_full_ci(
            self, fcivec_k, blocks, straid_k, strbid_k,
            nstra_total, nstrb_total):
        """Embed a momentum-sector vector in the full CI table."""
        ci_full = np.zeros((nstra_total, nstrb_total), dtype=fcivec_k.dtype)

        for block in blocks:
            ka, kb, nstra, nstrb, offset, size = map(int, block)
            block_ka_kb = fcivec_k[offset:offset + size].reshape(
                nstra, nstrb)
            astrs = straid_k[ka]
            bstrs = strbid_k[kb]
            ci_full[np.ix_(astrs, bstrs)] = block_ka_kb

        return ci_full

    def extract_sector_from_full_ci(
            self, ci_full, blocks, straid_k, strbid_k):
        """Extract a momentum-sector vector from a full CI table."""
        sector_size = int(blocks[:, 5].sum())
        fcivec_k = np.zeros(sector_size, dtype=ci_full.dtype)

        for block in blocks:
            ka, kb, nstra, nstrb, offset, size = map(int, block)
            astrs = straid_k[ka]
            bstrs = strbid_k[kb]
            block_ka_kb = ci_full[np.ix_(astrs, bstrs)]
            fcivec_k[offset:offset + size] = block_ka_kb.reshape(-1)

        return fcivec_k

    def random_ksector_fcivec(
            self, nkpts, ncas, nelec, target_k=0, seed=12):
        """Generate a normalized random sector CI vector and its maps."""
        rng = np.random.default_rng(seed)
        norb = nkpts * ncas
        (link_indexa, link_indexb, straid_k, strbid_k,
         blocks) = self.get_ksector_info(norb, nelec, nkpts, target_k)
        sector_size = int(blocks[:, 5].sum())

        fcivec_k = (rng.normal(size=sector_size)
                    + 1j * rng.normal(size=sector_size))
        fcivec_k /= np.linalg.norm(fcivec_k)

        return (fcivec_k, link_indexa, link_indexb, straid_k, strbid_k,
                blocks)

    def embed_random_ksector_fcivec_to_full_ci(
            self, nkpts, ncas, nelec, target_k=0, seed=12):
        """Generate a random sector vector and embed it in the full CI table."""
        norb = nkpts * ncas
        (fcivec_k, link_indexa, link_indexb, straid_k, strbid_k,
         blocks) = self.random_ksector_fcivec(
             nkpts, ncas, nelec, target_k=target_k, seed=seed)

        nstra_total = cistring.num_strings(norb, nelec[0])
        nstrb_total = cistring.num_strings(norb, nelec[1])
        ci_full = self.embed_sector_fcivec_to_full_ci(
            fcivec_k, blocks, straid_k, strbid_k,
            nstra_total, nstrb_total)

        return (fcivec_k, ci_full, link_indexa, link_indexb, straid_k,
                strbid_k, blocks)
