import numpy as np
import scipy.linalg
import sys
import types
import warnings
import ctypes

from pyscf import lib, __config__
from pyscf.fci import direct_spin1
from pyscf.fci.addons import _unpack_nelec
from mrh.my_pyscf.pbc.fci import rdm_helper, kcistrings, krdm_helper
from mrh.lib.helper import load_library


from mrh.my_pyscf.pbc.fci.kcistrings import gen_k_sector_linkstr_info, gen_k_sector_maps, build_k_links_spin


# Author: Bhavnesh Jangid

'''
Implementation of k-FCI.
'''

logger = lib.logger

HDIAG_IMAG_TOL = 1e-3
HERMI_THRESH = 1e-8
libpbcfci_k = None
contract_2e_threads = getattr(
    __config__, "pbc_k_contract_2e_threads",
    getattr(__config__, "pbc_contract_2e_threads", None),
)

def _load_k_contract_lib():
    global libpbcfci_k
    if libpbcfci_k is None:
        libpbcfci_k = load_library("libpbc_fci_contract_k")
        libpbcfci_k.FCIcontract_1e_k.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        libpbcfci_k.FCIcontract_1e_k.restype = None
        contract_2e_k_args = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        libpbcfci_k.FCIcontract_2e_k.argtypes = contract_2e_k_args
        libpbcfci_k.FCIcontract_2e_k.restype = None
        libpbcfci_k.FCIcontract_2e_k_zgemm.argtypes = contract_2e_k_args
        libpbcfci_k.FCIcontract_2e_k_zgemm.restype = None
    return libpbcfci_k

def _flatten_sector_ids(str_ids_by_k, nkpts):
    ids = []
    offsets = [0]
    for k in range(nkpts):
        tab = np.asarray(str_ids_by_k[k], dtype=np.int32, order="C")
        ids.append(tab)
        offsets.append(offsets[-1] + tab.size)

    if ids:
        ids = np.asarray(np.concatenate(ids), dtype=np.int32, order="C")
    else:
        ids = np.zeros(0, dtype=np.int32)

    return ids, np.asarray(offsets, dtype=np.int32, order="C")

def _unpack(norb, nelec, link_index, nkpts, spin=None):
    assert norb % nkpts == 0
    if link_index is None:
        neleca, nelecb = _unpack_nelec(nelec, spin)
        norb_k = norb // nkpts
        orb_k = (np.arange(norb, dtype=np.int32) // norb_k).astype(np.int32)
        if spin == 0 and neleca == nelecb:
            link_indexa = link_indexb = kcistrings.gen_linkstr_index_k(range(norb), neleca, orb_k, nkpts)
        else:
            link_indexa = kcistrings.gen_linkstr_index_k(range(norb), neleca, orb_k, nkpts)
            link_indexb = kcistrings.gen_linkstr_index_k(range(norb), nelecb, orb_k, nkpts)
        return link_indexa, link_indexb
    else:
        assert link_index[0].shape[2] == link_index[1].shape[2] == 8
        return link_index

def contract_1e_k(h1e, fcivec, norb, nelec, nkpts, kindx, link_index=None):
    '''
    Contract one-electron Hamiltonian with a k-FCI vector in a fixed 
    total momentum sector.
    args:
        h1e : ndarray, shape (nkpts, norb_k, norb_k)
            One-electron integrals in k-space, where norb_k = norb // nkpts.
        fcivec : ndarray, shape (sector_size,)
            k-FCI vector in the target total momentum sector.
        norb : int
            Total number of orbitals.
        nelec : tuple of 2 ints
            Number of alpha and beta electrons.
        nkpts : int
            Number of k-points / momentum sectors.
        kindx : int
            Target total momentum sector. (0<=kindx < nkpts)
        link_index : tuple of 2 ndarrays or None
            Look up tables/link index for alpha and beta strings. 
            If None, it will be generated on the fly.
        Note: these are k-aware link indices, and the link columns are:
            [cre, des, target_address, parity, k0, k_cre, k_des, dK].
            and overall shape is (nstr, nlink, 8) for each spin sector.
    returns:
        sigma_ci : ndarray, shape (sector_size,)
            Result of the Hamiltonian-vector product in the target momentum sector.
    '''

    link_indexa, link_indexb = _unpack(norb, nelec, link_index, nkpts)
    dtype = np.result_type(h1e, fcivec)

    # Sanity checks
    assert link_indexa.ndim == link_indexb.ndim == 3
    assert link_indexa.shape[2] == link_indexb.shape[2] == 8
    assert h1e.ndim == 3
    ncas = norb // nkpts
    assert h1e.shape == (nkpts, ncas, ncas)

    kindx = int(kindx) % nkpts

    # Generate the k-sector blocks and the corresponding alpha/beta string
    # lists and global-to-local (specific k-sector) maps.
    # rows are [ka, kb, na, nb, offset, size]
    blocks = gen_k_sector_linkstr_info(link_indexa, link_indexb, nkpts, kindx)
    sector_size = int(blocks[:, 5].sum())

    assert fcivec.size == sector_size

    straid_k, strbid_k, tota_2k, totb_2k = gen_k_sector_maps(link_indexa, link_indexb, nkpts)

    # Making sure fcivec is in the right dtype and C-contiguous.
    h1e = np.asarray(h1e, dtype=dtype, order="C")
    fcivec = np.asarray(fcivec, dtype=dtype, order="C")
    sigma_ci = np.zeros(fcivec.shape, dtype=dtype, order="C")

    # link columns: [cre, des, target_address, parity, k0, k_cre, k_des, dK]
    CRE = 0
    DES = 1
    TARGET = 2
    SIGN = 3
    K_CRE = 5
    K_DES = 6
    DK = 7

    for ka, kb, na, nb, offset, size in blocks:
        Cblk = fcivec[offset:offset + size].reshape(na, nb)
        Sblk = sigma_ci[offset:offset + size].reshape(na, nb)

        alpha_ids = straid_k[ka]
        beta_ids = strbid_k[kb]

        # h1e contraction for the alpha strings.
        for ia0_local, astr0 in enumerate(alpha_ids):
            astr0 = int(astr0)
            for link in link_indexa[astr0]:
                p = int(link[CRE])
                q = int(link[DES])
                astr1 = int(link[TARGET])
                sign = link[SIGN]

                k_cre = int(link[K_CRE]) % nkpts
                k_des = int(link[K_DES]) % nkpts
                dK = int(link[DK]) % nkpts

                # h1e[k, p, q] is k-diagonal, so only k_cre == k_des contributes.
                # which means only dk=0 contributes.
                if (k_cre != k_des) or (dK != 0): continue

                # Note that p and q are in the global orbital indexing, 
                # but h1e is in the k-space orbital indexing, so we need 
                # to mod by ncas to get the correct orbital indices for h1e.
                hpq = h1e[k_cre, p % ncas, q % ncas]

                # Check if the excitation is out of the momentum sector then skip this.
                ia1_local = tota_2k[ka, astr1]
                if ia1_local < 0: continue
                Sblk[ia1_local, :] += sign * hpq * Cblk[ia0_local, :]

        # h1e contraction for the beta strings.
        for ib0_local, bstr0 in enumerate(beta_ids):
            bstr0 = int(bstr0)
            for link in link_indexb[bstr0]:
                p = int(link[CRE])
                q = int(link[DES])
                bstr1 = int(link[TARGET])
                sign = link[SIGN]
                k_cre = int(link[K_CRE]) % nkpts
                k_des = int(link[K_DES]) % nkpts
                dK = int(link[DK]) % nkpts
                # h1e[k, p, q] is k-diagonal, so only k_cre == k_des contributes.
                # which means only dk=0 contributes.
                if (k_cre != k_des) or (dK != 0): continue

                hpq = h1e[k_cre, p % ncas, q % ncas]
                
                # Check if the excitation is out of the momentum sector then skip this.
                ib1_local = totb_2k[kb, bstr1]
                if ib1_local < 0:
                    continue

                Sblk[:, ib1_local] += sign * hpq * Cblk[:, ib0_local]

    return sigma_ci

def contract_1e_k_c(h1e, fcivec, norb, nelec, nkpts, kindx,
                    link_index=None):
    '''
    C implementation of contract_1e_k using k-sector link maps generated in
    Python.  The result is returned as complex128 to match the C kernel.
    '''
    nkpts = int(nkpts)
    ncas = int(norb) // nkpts
    assert ncas * nkpts == int(norb)

    link_indexa, link_indexb = _unpack(norb, nelec, link_index, nkpts)
    blocks = gen_k_sector_linkstr_info(link_indexa, link_indexb, nkpts,
                                       kindx)
    sector_size = int(blocks[:, 5].sum()) if blocks.size else 0
    assert fcivec.size == sector_size

    straid_k, strbid_k, str2tot_a, str2tot_b = gen_k_sector_maps(
        link_indexa, link_indexb, nkpts)
    stra_ids, stra_offsets = _flatten_sector_ids(straid_k, nkpts)
    strb_ids, strb_offsets = _flatten_sector_ids(strbid_k, nkpts)

    h1e = np.asarray(h1e, dtype=np.complex128, order="C")
    fcivec = np.asarray(fcivec, dtype=np.complex128, order="C")
    blocks = np.asarray(blocks, dtype=np.int32, order="C")
    link_indexa = np.asarray(link_indexa, dtype=np.int32, order="C")
    link_indexb = np.asarray(link_indexb, dtype=np.int32, order="C")
    str2tot_a = np.asarray(str2tot_a, dtype=np.int32, order="C")
    str2tot_b = np.asarray(str2tot_b, dtype=np.int32, order="C")
    sigma_ci = np.zeros(fcivec.shape, dtype=np.complex128, order="C")

    assert h1e.shape == (nkpts, ncas, ncas)

    libpbcfci = _load_k_contract_lib()
    with lib.with_omp_threads(contract_2e_threads):
        libpbcfci.FCIcontract_1e_k(
            h1e.ctypes.data_as(ctypes.c_void_p),
            fcivec.ctypes.data_as(ctypes.c_void_p),
            sigma_ci.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nkpts),
            ctypes.c_int(ncas),
            ctypes.c_int(blocks.shape[0]),
            blocks.ctypes.data_as(ctypes.c_void_p),
            link_indexa.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(link_indexa.shape[0]),
            ctypes.c_int(link_indexa.shape[1]),
            link_indexb.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(link_indexb.shape[0]),
            ctypes.c_int(link_indexb.shape[1]),
            stra_ids.ctypes.data_as(ctypes.c_void_p),
            stra_offsets.ctypes.data_as(ctypes.c_void_p),
            strb_ids.ctypes.data_as(ctypes.c_void_p),
            strb_offsets.ctypes.data_as(ctypes.c_void_p),
            str2tot_a.ctypes.data_as(ctypes.c_void_p),
            str2tot_b.ctypes.data_as(ctypes.c_void_p),
        )
    return sigma_ci


def _get_ci_sectors(fcivec, blocks, nkpts):
    '''
    Extract blocked CI vectors from a full CI vector based on k-sector information.
    '''
    ci_blocks = [[None for _ in range(nkpts)] for _ in range(nkpts)]
    for blk in blocks:
        ka, kb, nstra, nstrb, offset, size = map(int, blk)
        ci_blocks[ka][kb] = fcivec[offset:offset + size].reshape(nstra, nstrb)
    return ci_blocks


L_CRE_L       = 0
L_DES_L       = 1
L_STR0_LOCAL  = 2
L_STR1_LOCAL  = 3
L_STR0_GLOBAL = 4
L_STR1_GLOBAL = 5
L_SIGN        = 6
L_K0          = 7
L_K1          = 8
L_K_CRE       = 9
L_K_DES       = 10
L_DK          = 11

NLINK_FIELDS = 12


def compact_links_from_object_links(links, nkpts):
    rows = []

    for k in range(nkpts):
        for dK in range(nkpts):
            for link in links["by_k_dk"][k][dK]:
                rows.append([
                    link.cre_l,
                    link.des_l,
                    link.str0_local,
                    link.str1_local,
                    link.str0_global,
                    link.str1_global,
                    link.sign,
                    link.k0,
                    link.k1,
                    link.k_cre,
                    link.k_des,
                    link.dK,
                ])

    if len(rows) == 0:
        linktab = np.zeros((0, NLINK_FIELDS), dtype=np.int32)
    else:
        linktab = np.asarray(rows, dtype=np.int32)

    if linktab.shape[0] > 0:
        order = np.lexsort((linktab[:, L_DK], linktab[:, L_K0]))
        linktab = np.asarray(linktab[order], dtype=np.int32, order="C")

    offset_k_dk = np.zeros((nkpts, nkpts + 1), dtype=np.int32)

    pos = 0
    for k in range(nkpts):
        offset_k_dk[k, 0] = pos

        for dK in range(nkpts):
            while (
                pos < linktab.shape[0]
                and linktab[pos, L_K0] == k
                and linktab[pos, L_DK] == dK
            ):
                pos += 1

            offset_k_dk[k, dK + 1] = pos

    compact = {
        "str_k": links["str_k"],
        "str_k2tot": links["str_k2tot"],
        "linktab": linktab,
        "offset_k_dk": offset_k_dk,
    }

    return compact


def get_links_by_k(links, k):
    linktab = links["linktab"]
    offset = links["offset_k_dk"]

    start = offset[k, 0]
    end = offset[k, -1]

    return linktab[start:end]


def get_links_by_k_dk(links, k, dK):
    linktab = links["linktab"]
    offset = links["offset_k_dk"]

    start = offset[k, dK]
    end = offset[k, dK + 1]

    return linktab[start:end]


def build_links_by_global_source_array(links):
    linktab = links["linktab"]
    nlinks = linktab.shape[0]

    if nlinks == 0:
        links["global_source_order"] = np.zeros(0, dtype=np.int32)
        links["global_source_ids"] = np.zeros(0, dtype=np.int32)
        links["global_source_offsets"] = np.zeros(1, dtype=np.int32)
        return links

    src = linktab[:, L_STR0_GLOBAL]

    order = np.argsort(src, kind="stable").astype(np.int32)
    src_sorted = src[order]

    unique_src, first = np.unique(src_sorted, return_index=True)

    offsets = np.empty(unique_src.size + 1, dtype=np.int32)
    offsets[:-1] = first.astype(np.int32)
    offsets[-1] = nlinks

    links["global_source_order"] = order
    links["global_source_ids"] = unique_src.astype(np.int32)
    links["global_source_offsets"] = offsets

    return links


def get_link_indices_from_global_source(links, src_global):
    ids = links["global_source_ids"]
    offsets = links["global_source_offsets"]
    order = links["global_source_order"]

    pos = np.searchsorted(ids, src_global)

    if pos >= ids.size or ids[pos] != src_global:
        return order[0:0]

    start = offsets[pos]
    end = offsets[pos + 1]

    return order[start:end]

AB_A0      = 0
AB_A1      = 1
AB_B0      = 2
AB_B1      = 3
AB_SIGN    = 4
AB_KA1     = 5
AB_KB1     = 6

AB_KPA     = 7
AB_KQA     = 8
AB_KRB     = 9
AB_PA      = 10
AB_QA      = 11
AB_RB      = 12
AB_SB      = 13

AB_KPB     = 14
AB_KQB     = 15
AB_KRA     = 16
AB_PB      = 17
AB_QB      = 18
AB_RA      = 19
AB_SA      = 20

NAB_FIELDS = 21

def build_ab_pair_tables(links_a, links_b, nkpts):
    ab_pairs = [[None for _ in range(nkpts)] for _ in range(nkpts)]

    for ka in range(nkpts):
        la_tab = get_links_by_k(links_a, ka)

        for kb in range(nkpts):
            rows = []

            for la in la_tab:
                dKa = int(la[L_DK])
                ka1 = int(la[L_K1])
                dKb_needed = (-dKa) % nkpts

                lb_tab = get_links_by_k_dk(links_b, kb, dKb_needed)

                for lb in lb_tab:
                    kb1 = int(lb[L_K1])

                    rows.append([
                        int(la[L_STR0_LOCAL]),
                        int(la[L_STR1_LOCAL]),
                        int(lb[L_STR0_LOCAL]),
                        int(lb[L_STR1_LOCAL]),
                        int(la[L_SIGN]) * int(lb[L_SIGN]),
                        ka1,
                        kb1,

                        int(la[L_K_CRE]),
                        int(la[L_K_DES]),
                        int(lb[L_K_CRE]),
                        int(la[L_CRE_L]),
                        int(la[L_DES_L]),
                        int(lb[L_CRE_L]),
                        int(lb[L_DES_L]),

                        int(lb[L_K_CRE]),
                        int(lb[L_K_DES]),
                        int(la[L_K_CRE]),
                        int(lb[L_CRE_L]),
                        int(lb[L_DES_L]),
                        int(la[L_CRE_L]),
                        int(la[L_DES_L]),
                    ])

            if len(rows) == 0:
                ab_pairs[ka][kb] = np.zeros((0, NAB_FIELDS), dtype=np.int32)
            else:
                ab_pairs[ka][kb] = np.asarray(rows, dtype=np.int32)

    return ab_pairs

SS_0      = 0
SS_1      = 1
SS_SIGN   = 2
SS_K1     = 3
SS_KP     = 4
SS_KQ     = 5
SS_KR     = 6
SS_P      = 7
SS_Q      = 8
SS_R      = 9
SS_S      = 10

NSS_FIELDS = 11

def build_same_spin_pair_tables(links, nkpts):
    linktab = links["linktab"]
    ss_pairs = [None for _ in range(nkpts)]

    for k in range(nkpts):
        rows = []
        l1_tab = get_links_by_k(links, k)

        for l1 in l1_tab:
            dK1 = int(l1[L_DK])
            src_mid = int(l1[L_STR1_GLOBAL])

            l2_indices = get_link_indices_from_global_source(links, src_mid)

            for idx2 in l2_indices:
                l2 = linktab[idx2]

                if int(l2[L_K0]) != int(l1[L_K1]):
                    continue

                dK2 = int(l2[L_DK])

                if (dK1 + dK2) % nkpts != 0:
                    continue

                # l1 acts first: E_rs
                r = int(l1[L_CRE_L])
                s = int(l1[L_DES_L])
                kr = int(l1[L_K_CRE])

                # l2 acts second: E_pq
                p = int(l2[L_CRE_L])
                q = int(l2[L_DES_L])
                kp = int(l2[L_K_CRE])
                kq = int(l2[L_K_DES])

                rows.append([ int(l1[L_STR0_LOCAL]), int(l2[L_STR1_LOCAL]), int(l1[L_SIGN]) * int(l2[L_SIGN]), 
                             int(l2[L_K1]), kp, kq, kr, p, q, r, s, ])

        if len(rows) == 0:
            ss_pairs[k] = np.zeros((0, NSS_FIELDS), dtype=np.int32)
        else:
            ss_pairs[k] = np.asarray(rows, dtype=np.int32)

    return ss_pairs

def contract_ab_pairs(eri, ci0_block, ci1_blocks, ab_pairs, ka, kb):
    pairtab = ab_pairs[ka][kb]

    for row in pairtab:
        a0 = row[AB_A0]
        a1 = row[AB_A1]
        b0 = row[AB_B0]
        b1 = row[AB_B1]
        sign = row[AB_SIGN]
        ka1 = row[AB_KA1]
        kb1 = row[AB_KB1]
        ci1_block = ci1_blocks[ka1][kb1]

        if ci1_block is None:
            continue

        val_ab = eri[row[AB_KPA], row[AB_KQA], row[AB_KRB], row[AB_PA], row[AB_QA], row[AB_RB], row[AB_SB]]
        
        val_ba = eri[row[AB_KPB], row[AB_KQB], row[AB_KRA], row[AB_PB], row[AB_QB], row[AB_RA], row[AB_SA]]
        
        ci1_block[a1, b1] += ((val_ab + val_ba) * sign * ci0_block[a0, b0])

def contract_aa_pairs(eri, ci0_blocks, ci1_blocks, aa_pairs, ka, kb):
    ci0_block = ci0_blocks[ka][kb]
    if ci0_block is None:
        return

    pairtab = aa_pairs[ka]

    for row in pairtab:
        a0 = row[SS_0]
        a1 = row[SS_1]
        sign = row[SS_SIGN]
        ka1 = row[SS_K1]

        ci1_block = ci1_blocks[ka1][kb]
        if ci1_block is None:
            continue

        val = eri[row[SS_KP], row[SS_KQ], row[SS_KR], row[SS_P], row[SS_Q], row[SS_R], row[SS_S]]

        ci1_block[a1, :] += val * sign * ci0_block[a0, :]

def contract_bb_pairs(eri, ci0_blocks, ci1_blocks, bb_pairs, ka, kb):
    ci0_block = ci0_blocks[ka][kb]
    if ci0_block is None:
        return

    pairtab = bb_pairs[kb]

    for row in pairtab:
        b0 = row[SS_0]
        b1 = row[SS_1]
        sign = row[SS_SIGN]
        kb1 = row[SS_K1]

        ci1_block = ci1_blocks[ka][kb1]
        if ci1_block is None:
            continue

        val = eri[row[SS_KP], row[SS_KQ], row[SS_KR], row[SS_P], row[SS_Q], row[SS_R], row[SS_S]]

        ci1_block[:, b1] += val * sign * ci0_block[:, b0]

def _flatten_pair_tables(ab_pairs, aa_pairs, bb_pairs, nkpts):
    ab_rows = []
    ab_offsets = [0]
    for ka in range(nkpts):
        for kb in range(nkpts):
            tab = np.asarray(ab_pairs[ka][kb], dtype=np.int32, order="C")
            if tab.size:
                ab_rows.append(tab.reshape(-1, NAB_FIELDS))
            ab_offsets.append(ab_offsets[-1] + tab.reshape(-1, NAB_FIELDS).shape[0])

    aa_rows = []
    aa_offsets = [0]
    for k in range(nkpts):
        tab = np.asarray(aa_pairs[k], dtype=np.int32, order="C")
        if tab.size:
            aa_rows.append(tab.reshape(-1, NSS_FIELDS))
        aa_offsets.append(aa_offsets[-1] + tab.reshape(-1, NSS_FIELDS).shape[0])

    bb_rows = []
    bb_offsets = [0]
    for k in range(nkpts):
        tab = np.asarray(bb_pairs[k], dtype=np.int32, order="C")
        if tab.size:
            bb_rows.append(tab.reshape(-1, NSS_FIELDS))
        bb_offsets.append(bb_offsets[-1] + tab.reshape(-1, NSS_FIELDS).shape[0])

    if ab_rows:
        ab_tab = np.asarray(np.vstack(ab_rows), dtype=np.int32, order="C")
    else:
        ab_tab = np.zeros((0, NAB_FIELDS), dtype=np.int32)

    if aa_rows:
        aa_tab = np.asarray(np.vstack(aa_rows), dtype=np.int32, order="C")
    else:
        aa_tab = np.zeros((0, NSS_FIELDS), dtype=np.int32)

    if bb_rows:
        bb_tab = np.asarray(np.vstack(bb_rows), dtype=np.int32, order="C")
    else:
        bb_tab = np.zeros((0, NSS_FIELDS), dtype=np.int32)

    return (ab_tab, np.asarray(ab_offsets, dtype=np.int32, order="C"),
            aa_tab, np.asarray(aa_offsets, dtype=np.int32, order="C"),
            bb_tab, np.asarray(bb_offsets, dtype=np.int32, order="C"))

def _build_contract_pair_tables(link_indexa, link_indexb, norb, nkpts):
    straid_k, strbid_k, str2tot_a, str2tot_b = gen_k_sector_maps(
        link_indexa, link_indexb, nkpts)

    links_a = build_k_links_spin(link_indexa, norb, nkpts,
                                 straid_k, str2tot_a)
    links_b = build_k_links_spin(link_indexb, norb, nkpts,
                                 strbid_k, str2tot_b)

    links_a = build_links_by_global_source_array(links_a)
    links_b = build_links_by_global_source_array(links_b)

    ab_pairs = build_ab_pair_tables(links_a, links_b, nkpts)
    aa_pairs = build_same_spin_pair_tables(links_a, nkpts)
    bb_pairs = build_same_spin_pair_tables(links_b, nkpts)

    return _flatten_pair_tables(ab_pairs, aa_pairs, bb_pairs, nkpts)

def _contract_2e_k_c_kernel(kernel_name, eri, fcivec, norb, nelec, nkpts,
                            target_k, link_index=None):
    nkpts = int(nkpts)
    ncas = int(norb) // nkpts
    assert ncas * nkpts == int(norb)

    link_indexa, link_indexb = _unpack(norb, nelec, link_index, nkpts)
    blocks = gen_k_sector_linkstr_info(link_indexa, link_indexb, nkpts,
                                       target_k)
    sector_size = int(blocks[:, 5].sum()) if blocks.size else 0
    assert fcivec.size == sector_size

    ab_tab, ab_offsets, aa_tab, aa_offsets, bb_tab, bb_offsets = (
        _build_contract_pair_tables(link_indexa, link_indexb, norb, nkpts))

    eri = np.asarray(eri, dtype=np.complex128, order="C")
    fcivec = np.asarray(fcivec, dtype=np.complex128, order="C")
    blocks = np.asarray(blocks, dtype=np.int32, order="C")
    sigma_ci = np.zeros(fcivec.shape, dtype=np.complex128, order="C")

    assert eri.shape == (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)

    libpbcfci = _load_k_contract_lib()
    kernel = getattr(libpbcfci, kernel_name)
    with lib.with_omp_threads(contract_2e_threads):
        kernel(
            eri.ctypes.data_as(ctypes.c_void_p),
            fcivec.ctypes.data_as(ctypes.c_void_p),
            sigma_ci.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nkpts),
            ctypes.c_int(ncas),
            ctypes.c_int(blocks.shape[0]),
            blocks.ctypes.data_as(ctypes.c_void_p),
            ab_tab.ctypes.data_as(ctypes.c_void_p),
            ab_offsets.ctypes.data_as(ctypes.c_void_p),
            aa_tab.ctypes.data_as(ctypes.c_void_p),
            aa_offsets.ctypes.data_as(ctypes.c_void_p),
            bb_tab.ctypes.data_as(ctypes.c_void_p),
            bb_offsets.ctypes.data_as(ctypes.c_void_p),
        )
    return sigma_ci

def contract_2e_k_c(eri, fcivec, norb, nelec, nkpts, target_k,
                    link_index=None):
    '''
    C implementation of contract_2e_k using Python-built k pair tables.
    This wrapper keeps the current Python implementation available as the
    reference path while the lower-level kernel is validated.
    '''
    return _contract_2e_k_c_kernel(
        "FCIcontract_2e_k", eri, fcivec, norb, nelec, nkpts, target_k,
        link_index=link_index,
    )

def contract_2e_k_zgemm(eri, fcivec, norb, nelec, nkpts, target_k,
                        link_index=None):
    '''
    BLAS-backed C implementation of contract_2e_k using Python-built k pair
    tables.  The alpha-alpha and beta-beta same-spin contractions are applied
    with zgemm; the alpha-beta terms are packed into sparse source/destination
    block groups.  OpenMP threads follow
    pbc_k_contract_2e_threads/pbc_contract_2e_threads.
    '''
    return _contract_2e_k_c_kernel(
        "FCIcontract_2e_k_zgemm", eri, fcivec, norb, nelec, nkpts, target_k,
        link_index=link_index,
    )

def contract_2e_k(eri, fcivec, norb, nelec, nkpts, target_k, link_index=None):
    '''
    Contract two-electron Hamiltonian with a k-FCI vector in a fixed total momentum sector.
    args:
        eri : ndarray, shape (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)
            Two-electron integrals in k-space, in chemist notation.
        fcivec : ndarray, shape (sector_size,)
            k-FCI vector in the target total momentum sector.
        norb : int
            Total number of orbitals.
        nelec : tuple of 2 ints
            Number of alpha and beta electrons.
        nkpts : int
            Number of k-points / momentum sectors.
        target_k : int
            Target total momentum sector for the output sigma vector.
        link_indexa, link_indexb : tuple of 2 ndarrays
            Look up tables/link index for alpha and beta strings. 
            These should be k-aware link indices, and the link columns are:
            [cre, des, target_address, parity, k0, k_cre, k_des, dK].
    returns:
        sigma_ci : ndarray, shape (sector_size,)
            Result of the Hamiltonian-vector product in the target momentum sector.    
    '''
    nkpts = eri.shape[0]
    dtype = np.result_type(eri.dtype, fcivec.dtype)

    link_indexa, link_indexb = _unpack(norb, nelec, link_index, nkpts)
    straid_k, strbid_k, str2tot_a, str2tot_b = gen_k_sector_maps(link_indexa, link_indexb, nkpts)
    
    
    links_a = build_k_links_spin(link_indexa, norb, nkpts, straid_k, str2tot_a)
    links_b = build_k_links_spin(link_indexb, norb, nkpts, strbid_k, str2tot_b)
    
    # links_a = compact_links_from_object_links(links_a, nkpts)
    # links_b = compact_links_from_object_links(links_b, nkpts)

    links_a = build_links_by_global_source_array(links_a)
    links_b = build_links_by_global_source_array(links_b)

    ab_pairs = build_ab_pair_tables(links_a, links_b, nkpts)
    aa_pairs = build_same_spin_pair_tables(links_a, nkpts)
    bb_pairs = build_same_spin_pair_tables(links_b, nkpts)

    straid_k = strbid_k = str2tot_a = str2tot_b = None # free up some memory

    # straid_k = strbid_k = str2tot_a = str2tot_b = None # free up some memory

    sigma_ci = np.zeros(fcivec.shape, dtype=dtype, order="C")
    
    blocks = gen_k_sector_linkstr_info(link_indexa, link_indexb, nkpts, target_k)

    # sanity check: the total size of the k-sector blocks should match the size of the fcivec.
    sector_size = int(blocks[:, 5].sum())
    assert fcivec.size == sector_size
    
    # Now rearrange the fcivec into a blocked structure based on the k-sectors of the 
    # alpha and beta strings.
    ci0_blocks = _get_ci_sectors(fcivec, blocks, nkpts)
    ci1_blocks = _get_ci_sectors(sigma_ci, blocks, nkpts)

    # Free up some memory.
    blocks = None

    for ka in range(nkpts):
        kb = (target_k - ka) % nkpts

        if ci0_blocks[ka][kb] is None:
            continue

        # contract2e for alpha beta blocks
        # contract_ab(eri, ci0_blocks[ka][kb], ci1_blocks, links_a, links_b, ka, kb)

        # # same-spin alpha-alpha part
        # contract_aa(eri, ci0_blocks, ci1_blocks, links_a, ka, kb)
        
        # # same-spin beta-beta part
        # contract_bb(eri, ci0_blocks, ci1_blocks, links_b, ka, kb)
        contract_ab_pairs(eri, ci0_blocks[ka][kb], ci1_blocks, ab_pairs, ka, kb)

        contract_aa_pairs(eri, ci0_blocks, ci1_blocks, aa_pairs, ka, kb)

        contract_bb_pairs(eri, ci0_blocks, ci1_blocks, bb_pairs, ka, kb)

    return sigma_ci

def sector_size(norb, nelec, nkpts, target_k=0, link_index=None):
    '''
    Number of determinants in a fixed total momentum sector.
    args:
        norb : int
            Total number of active orbitals across all k-points.
        nelec : tuple of 2 ints
            Number of alpha and beta electrons.
        nkpts : int
            Number of k-points.
        target_k : int, optional
            Total momentum sector.
        link_index : tuple of 2 ndarrays or None
            k-aware link indices. If None, they are generated on the fly.
    returns:
        ndet_k : int
            Number of determinants in the target momentum sector.
    '''
    link_indexa, link_indexb = _unpack(norb, nelec, link_index, nkpts)
    blocks = gen_k_sector_linkstr_info(link_indexa, link_indexb, nkpts, target_k)
    if blocks.size == 0:
        return 0
    return int(blocks[:, 5].sum())

def contract_ham_k(h1e, eri, fcivec, norb, nelec, nkpts, target_k=0,
                   link_index=None):
    '''
    Contract the k-FCI Hamiltonian with a CI vector.
    Currently, I am keeping the one-electron and two-electron
    Hamiltonian contractions separate. I am not absorbing h1e into the
    two-electron Hamiltonian here.

    args:
        h1e : ndarray, shape (nkpts, norb_k, norb_k)
            One-electron Hamiltonian in k-space.
        eri : ndarray, shape (nkpts, nkpts, nkpts, norb_k, norb_k, norb_k, norb_k)
            Two-electron Hamiltonian in k-space and in the same convention as
            contract_2e_k.
        fcivec : ndarray, shape (sector_size,)
            k-FCI vector in the target momentum sector.
        norb : int
            Total number of active orbitals across all k-points.
        nelec : tuple of 2 ints
            Number of alpha and beta electrons.
        nkpts : int
            Number of k-points.
        target_k : int, optional
            Total momentum sector.
        link_index : tuple of 2 ndarrays or None
            k-aware link indices. If None, they are generated on the fly.
    returns:
        sigma_ci : ndarray, shape (sector_size,)
            Result of applying H to fcivec.
    '''
    dtype = np.result_type(h1e, eri, fcivec)
    fcivec = np.asarray(fcivec, dtype=dtype, order="C")
    link_index = _unpack(norb, nelec, link_index, nkpts)

    sigma_ci = contract_1e_k(h1e, fcivec, norb, nelec, nkpts, target_k,
                             link_index=link_index)
    sigma_ci += contract_2e_k(eri, fcivec, norb, nelec, nkpts, target_k,
                              link_index=link_index)
    return sigma_ci

def make_hdiag(h1e, eri, norb, nelec, nkpts, target_k=0, link_index=None):
    '''
    Diagonal of the k-FCI Hamiltonian in a fixed total momentum sector.
    This is a pure Python implementation for now. The diagonal is computed by
    applying the separated Hamiltonian contractions to each determinant basis
    vector and reading back the matching diagonal element.
    '''
    link_index = _unpack(norb, nelec, link_index, nkpts)
    ndet = sector_size(norb, nelec, nkpts, target_k, link_index=link_index)
    dtype = np.result_type(h1e, eri)
    hdiag = np.empty(ndet, dtype=dtype)

    for i in range(ndet):
        ci0 = np.zeros(ndet, dtype=dtype)
        ci0[i] = 1.0
        sigma = contract_ham_k(h1e, eri, ci0, norb, nelec, nkpts, target_k,
                               link_index=link_index)
        hdiag[i] = sigma[i]

    return hdiag

def get_init_guess_k(norb, nelec, nkpts, target_k, nroots, hdiag):
    '''
    Get initial guess vectors for k-FCI in a fixed total momentum sector.
    The guesses are determinant basis vectors corresponding to the lowest
    diagonal Hamiltonian elements.
    '''
    hdiag = np.asarray(hdiag)
    ndet = hdiag.size
    nroots = min(int(nroots), ndet)
    dtype = hdiag.dtype

    if nroots == 0:
        return []

    try:
        addr = np.argpartition(hdiag.real, nroots - 1)[:nroots]
        addr = addr[np.argsort(hdiag.real[addr], kind="stable")]
    except AttributeError:
        addr = np.argsort(hdiag.real, kind="stable")[:nroots]

    ci0 = []
    for i in range(nroots):
        x = np.zeros(ndet, dtype=dtype)
        x[int(addr[i])] = 1.0
        ci0.append(x)
    return ci0

def make_hamiltonian_k(h1e, eri, norb, nelec, nkpts, target_k=0,
                       link_index=None):
    '''
    Construct the explicit k-FCI Hamiltonian in a fixed total momentum sector.
    This routine is intended for small determinant spaces and for debugging.
    For large spaces, kernel_ms1 uses Davidson with contract_ham_k instead.
    '''
    link_index = _unpack(norb, nelec, link_index, nkpts)
    ndet = sector_size(norb, nelec, nkpts, target_k, link_index=link_index)
    dtype = np.result_type(h1e, eri)
    hmat = np.empty((ndet, ndet), dtype=dtype, order="F")

    for i in range(ndet):
        ci0 = np.zeros(ndet, dtype=dtype)
        ci0[i] = 1.0
        hmat[:, i] = contract_ham_k(h1e, eri, ci0, norb, nelec, nkpts,
                                    target_k, link_index=link_index)

    return hmat

def energy(h1e, eri, fcivec, norb, nelec, nkpts, target_k=0, link_index=None):
    '''
    Compute the k-FCI electronic energy for a CI vector.
    The one-electron and two-electron Hamiltonian contractions are evaluated
    separately; h1e is not absorbed into eri.
    '''
    ci0 = np.asarray(fcivec)
    sigma = contract_ham_k(h1e, eri, ci0, norb, nelec, nkpts, target_k,
                           link_index=link_index)
    return np.vdot(ci0, sigma)

def make_rdm1s_py(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
                  spin=None):
    '''
    Python implementation of spin-separated 1-RDMs for a k-FCI vector.
    '''
    return krdm_helper.make_rdm1s_py(fcivec, norb, nelec, nkpts,
                                     target_k=target_k,
                                     link_index=link_index, spin=spin)

def make_rdm1_py(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
                 spin=None):
    '''
    Python implementation of spin-summed 1-RDM for a k-FCI vector.
    '''
    return krdm_helper.make_rdm1_py(fcivec, norb, nelec, nkpts,
                                    target_k=target_k,
                                    link_index=link_index, spin=spin)

def make_rdm12s_py(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
                   reorder=True, spin=None):
    '''
    Python implementation of spin-separated 1-RDMs and 2-RDMs for a k-FCI vector.
    '''
    return krdm_helper.make_rdm12s_py(fcivec, norb, nelec, nkpts,
                                      target_k=target_k,
                                      link_index=link_index,
                                      reorder=reorder, spin=spin)

def make_rdm12_py(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
                  reorder=True, spin=None):
    '''
    Python implementation of spin-summed 1-RDM and 2-RDM for a k-FCI vector.
    '''
    return krdm_helper.make_rdm12_py(fcivec, norb, nelec, nkpts,
                                     target_k=target_k,
                                     link_index=link_index,
                                     reorder=reorder, spin=spin)

def contract_ss(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
                spin=None):
    '''
    Apply S^2 to a k-FCI vector in a fixed total momentum sector.
    '''
    return krdm_helper.contract_ss(fcivec, norb, nelec, nkpts,
                                   target_k=target_k,
                                   link_index=link_index, spin=spin)

def spin_square(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
                spin=None, **kwargs):
    '''
    Spin square for a k-FCI vector in a fixed total momentum sector.
    '''
    return krdm_helper.spin_square(fcivec, norb, nelec, nkpts,
                                   target_k=target_k,
                                   link_index=link_index, spin=spin,
                                   **kwargs)

def _get_spin_penalty(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
                      spin=None, ss_value=None, ss_penalty=0.1):
    '''
    Apply the spin penalty operator to a k-FCI vector in one momentum sector.
    This follows the same logic as pyscf.fci.addons.SpinPenaltyFCISolver.
    '''
    nelec = _unpack_nelec(nelec, spin)
    sz = abs(nelec[0] - nelec[1]) * 0.5
    if ss_value is None:
        ss = sz * (sz + 1)
    else:
        ss = ss_value

    fcivec = np.asarray(fcivec)
    if ss < sz * (sz + 1) + 0.1:
        # (S^2-ss)|Psi> to shift state other than the lowest state.
        ci1 = contract_ss(fcivec, norb, nelec, nkpts,
                          target_k=target_k,
                          link_index=link_index, spin=spin).reshape(fcivec.shape)
        ci1 -= ss * fcivec
    else:
        # (S^2-ss)^2|Psi> to shift states except the given spin.
        tmp = contract_ss(fcivec, norb, nelec, nkpts,
                          target_k=target_k,
                          link_index=link_index, spin=spin).reshape(fcivec.shape)
        tmp -= ss * fcivec
        ci1 = -ss * tmp
        ci1 += contract_ss(tmp, norb, nelec, nkpts,
                           target_k=target_k,
                           link_index=link_index, spin=spin).reshape(fcivec.shape)
        tmp = None
    ci1 *= ss_penalty
    return ci1

def _make_diag_precond(hdiag, level_shift=1e-3):
    '''
    Diagonal preconditioner for the Davidson solver.
    '''
    hdiag = np.asarray(hdiag)
    if np.iscomplexobj(hdiag) and np.max(np.abs(hdiag.imag)) > HDIAG_IMAG_TOL:
        warnings.warn("The k-FCI Hamiltonian diagonal has non-negligible "
                      "imaginary parts: max |Im(hdiag)| = "
                      f"{np.max(np.abs(hdiag.imag))}.")

    def precond(dx, e, *args):
        diagd = hdiag - (np.real(e) - level_shift)
        diagd = diagd.astype(hdiag.dtype, copy=True)
        diagd[np.abs(diagd) < 1e-8] = 1e-8
        return dx / diagd

    return precond

def make_diag_precond(hdiag, pspaceig=None, pspaceci=None, addr=None,
                      level_shift=0):
    '''
    Wrapper to match the PySCF direct_spin1 preconditioner interface.
    '''
    return _make_diag_precond(hdiag, level_shift)

def kernel_ms1(fci, h1e, eri, norb, nelec, nkpts, target_k=0, ci0=None,
               link_index=None, tol=None, lindep=None, max_cycle=None,
               max_space=None, nroots=None, davidson_only=None,
               pspace_size=None, max_memory=None, verbose=None, ecore=0,
               **kwargs):
    '''
    k-FCI kernel in a fixed total momentum sector.
    This follows the direct_spin1 control flow: construct the explicit
    Hamiltonian for small spaces when memory allows, otherwise use Davidson.
    The Hamiltonian-vector product is contract_1e_k + contract_2e_k; no
    absorb_h1e step is used.
    '''
    if nroots is None: nroots = fci.nroots
    if davidson_only is None: davidson_only = fci.davidson_only
    if pspace_size is None: pspace_size = fci.pspace_size
    if max_memory is None: max_memory = fci.max_memory - lib.current_memory()[0]

    log = logger.new_logger(fci, verbose)
    nelec = _unpack_nelec(nelec, fci.spin)
    target_k = int(target_k) % nkpts
    link_index = _unpack(norb, nelec, link_index, nkpts, spin=fci.spin)

    assert norb % nkpts == 0
    ncas = norb // nkpts
    assert h1e.shape == (nkpts, ncas, ncas)
    assert eri.shape == (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)

    hdiag = fci.make_hdiag(h1e, eri, norb, nelec, nkpts, target_k,
                           link_index=link_index).ravel()
    civec_size = hdiag.size

    if civec_size == 0:
        raise RuntimeError(f"No determinants in k-FCI sector target_k={target_k}.")

    nroots = min(int(nroots), civec_size)
    hmat_mem = civec_size * civec_size * np.dtype(hdiag.dtype).itemsize * 1e-6
    min_davidson_mem = civec_size * 6 * np.dtype(hdiag.dtype).itemsize * 1e-6

    if max_memory < min_davidson_mem:
        log.warn("Not enough memory for k-FCI solver. "
                 "The minimal Davidson requirement is %.0f MB",
                 min_davidson_mem)

    do_direct = ((not davidson_only)
                 and civec_size <= pspace_size
                 and hmat_mem < max_memory)

    if do_direct:
        hmat = fci.make_hamiltonian(h1e, eri, norb, nelec, nkpts, target_k,
                                    link_index=link_index)
        e, c = fci.eig(hmat)
        e = e[:nroots]
        if nroots == 1:
            c = c[:, 0]
            e = e[0]
        else:
            c = c[:, :nroots].T
        return e + ecore, c

    precond = fci.make_precond(hdiag)

    cpu0 = [logger.process_clock(), logger.perf_counter()]
    def hop(c):
        hc = fci.contract_ham(h1e, eri, c, norb, nelec, nkpts, target_k,
                              link_index=link_index)
        cpu0[:] = log.timer_debug1("contract_ham_k", *cpu0)
        return hc.ravel()

    def init_guess():
        return fci.get_init_guess(norb, nelec, nkpts, target_k, nroots, hdiag)

    if ci0 is None:
        ci0 = init_guess
    elif not callable(ci0):
        if isinstance(ci0, np.ndarray):
            ci0 = [ci0.ravel()]
        else:
            ci0 = [x.ravel() for x in ci0]
        if len(ci0) < nroots:
            ci0.extend(init_guess()[len(ci0):])

    if tol is None: tol = fci.conv_tol
    if lindep is None: lindep = fci.lindep
    if max_cycle is None: max_cycle = fci.max_cycle
    if max_space is None: max_space = fci.max_space
    tol_residual = getattr(fci, "conv_tol_residual", None)

    with lib.with_omp_threads(fci.threads):
        e, c = fci.eig(hop, ci0, precond, tol=tol, lindep=lindep,
                       max_cycle=max_cycle, max_space=max_space,
                       nroots=nroots, max_memory=max_memory, verbose=log,
                       follow_state=True, tol_residual=tol_residual,
                       **kwargs)
    return e + ecore, c

class SpinPenaltyFCISolver:
    __name_mixin__ = 'SpinPenalty'
    _keys = {'ss_value', 'ss_penalty', 'base'}

    def __init__(self, fcibase, shift, ss_value):
        self.base = fcibase.copy()
        self.__dict__.update(fcibase.__dict__)
        self.ss_value = ss_value
        self.ss_penalty = shift
        self.davidson_only = self.base.davidson_only = True

    def undo_fix_spin(self):
        obj = lib.view(self, lib.drop_class(self.__class__, SpinPenaltyFCISolver))
        del obj.base
        del obj.ss_value
        del obj.ss_penalty
        return obj

    def base_contract_ham(self, *args, **kwargs):
        return super().contract_ham(*args, **kwargs)

    def contract_spin_penalty(self, fcivec, norb, nelec, nkpts=None,
                              target_k=None, link_index=None):
        if nkpts is None: nkpts = self.nkpts
        if target_k is None: target_k = self.target_k
        nelec = _unpack_nelec(nelec, self.spin)
        return _get_spin_penalty(fcivec, norb, nelec, nkpts,
                                 target_k=target_k,
                                 link_index=link_index, spin=self.spin,
                                 ss_value=self.ss_value,
                                 ss_penalty=self.ss_penalty)

    def contract_ham(self, h1e, eri, fcivec, norb, nelec, nkpts=None,
                     target_k=None, link_index=None):
        if nkpts is None: nkpts = self.nkpts
        if target_k is None: target_k = self.target_k
        ci0 = super().contract_ham(h1e, eri, fcivec, norb, nelec,
                                   nkpts=nkpts, target_k=target_k,
                                   link_index=link_index)
        ci1 = self.contract_spin_penalty(fcivec, norb, nelec,
                                         nkpts=nkpts, target_k=target_k,
                                         link_index=link_index)
        ci1 += ci0.reshape(fcivec.shape)
        return ci1

    def make_hdiag(self, h1e, eri, norb, nelec, nkpts=None, target_k=None,
                   link_index=None, compress=False):
        if nkpts is None: nkpts = self.nkpts
        if target_k is None: target_k = self.target_k
        nelec = _unpack_nelec(nelec, self.spin)
        link_index = _unpack(norb, nelec, link_index, nkpts, spin=self.spin)
        ndet = sector_size(norb, nelec, nkpts, target_k, link_index=link_index)
        dtype = np.result_type(h1e, eri, np.complex128)
        hdiag = np.empty(ndet, dtype=dtype)

        for i in range(ndet):
            ci0 = np.zeros(ndet, dtype=dtype)
            ci0[i] = 1.0
            sigma = self.contract_ham(h1e, eri, ci0, norb, nelec,
                                      nkpts=nkpts, target_k=target_k,
                                      link_index=link_index)
            hdiag[i] = sigma[i]

        return hdiag

    def make_hamiltonian(self, h1e, eri, norb, nelec, nkpts=None,
                         target_k=None, link_index=None):
        if nkpts is None: nkpts = self.nkpts
        if target_k is None: target_k = self.target_k
        nelec = _unpack_nelec(nelec, self.spin)
        link_index = _unpack(norb, nelec, link_index, nkpts, spin=self.spin)
        ndet = sector_size(norb, nelec, nkpts, target_k, link_index=link_index)
        dtype = np.result_type(h1e, eri, np.complex128)
        hmat = np.empty((ndet, ndet), dtype=dtype, order="F")

        for i in range(ndet):
            ci0 = np.zeros(ndet, dtype=dtype)
            ci0[i] = 1.0
            hmat[:, i] = self.contract_ham(h1e, eri, ci0, norb, nelec,
                                           nkpts=nkpts, target_k=target_k,
                                           link_index=link_index)

        return hmat

    def energy(self, h1e, eri, fcivec, norb, nelec, nkpts=None,
               target_k=None, link_index=None):
        if nkpts is None: nkpts = self.nkpts
        if target_k is None: target_k = self.target_k
        nelec = _unpack_nelec(nelec, self.spin)
        sigma = self.contract_ham(h1e, eri, fcivec, norb, nelec,
                                  nkpts=nkpts, target_k=target_k,
                                  link_index=link_index)
        return np.vdot(fcivec, sigma)

def fix_spin(fciobj, shift=0.1, ss=None, **kwargs):
    '''
    Add a spin penalty to the k-FCI solver.
    '''
    if isinstance(fciobj, types.ModuleType):
        raise DeprecationWarning('fix_spin should be applied on FCI object only')

    if 'ss_value' in kwargs:
        sys.stderr.write('fix_spin_: kwarg "ss_value" will be removed in future release. '
                         'It was replaced by "ss"\n')
        ss_value = kwargs['ss_value']
    else:
        ss_value = ss

    if isinstance(fciobj, SpinPenaltyFCISolver):
        fciobj.ss_penalty = shift
        fciobj.ss_value = ss_value
        return fciobj

    return lib.set_class(SpinPenaltyFCISolver(fciobj, shift, ss_value),
                         (SpinPenaltyFCISolver, fciobj.__class__))

def fix_spin_(fciobj, shift=0.1, ss=None, **kwargs):
    sp_fci = fix_spin(fciobj, shift=shift, ss=ss, **kwargs)
    fciobj.__class__ = sp_fci.__class__
    fciobj.__dict__ = sp_fci.__dict__
    return fciobj

class FCISolver(direct_spin1.FCISolver):
    '''
    k-FCI solver for periodic active spaces.
    This solver works in one total momentum sector at a time. The CI vector is
    stored only for that sector, using the k-aware link tables generated by
    kcistrings.py.
    '''
    def __init__(self, *args, **kwargs):
        nkpts = kwargs.pop("nkpts", None)
        target_k = kwargs.pop("target_k", 0)
        direct_spin1.FCISolver.__init__(self, *args, **kwargs)
        self.nkpts = nkpts
        self.target_k = target_k
        self.davidson_only = False

    def contract_1e(self, h1e, fcivec, norb, nelec, nkpts=None,
                    target_k=None, link_index=None):
        if nkpts is None: nkpts = self.nkpts
        if target_k is None: target_k = self.target_k
        return contract_1e_k(h1e, fcivec, norb, nelec, nkpts, target_k,
                             link_index=link_index)

    def contract_2e(self, eri, fcivec, norb, nelec, nkpts=None,
                    target_k=None, link_index=None):
        if nkpts is None: nkpts = self.nkpts
        if target_k is None: target_k = self.target_k
        return contract_2e_k(eri, fcivec, norb, nelec, nkpts, target_k,
                             link_index=link_index)

    def contract_ham(self, h1e, eri, fcivec, norb, nelec, nkpts=None,
                     target_k=None, link_index=None):
        if nkpts is None: nkpts = self.nkpts
        if target_k is None: target_k = self.target_k
        return contract_ham_k(h1e, eri, fcivec, norb, nelec, nkpts, target_k,
                              link_index=link_index)

    def make_hdiag(self, h1e, eri, norb, nelec, nkpts=None, target_k=None,
                   link_index=None, compress=False):
        if nkpts is None: nkpts = self.nkpts
        if target_k is None: target_k = self.target_k
        nelec = _unpack_nelec(nelec, self.spin)
        return make_hdiag(h1e, eri, norb, nelec, nkpts, target_k,
                          link_index=link_index)

    def make_hamiltonian(self, h1e, eri, norb, nelec, nkpts=None,
                         target_k=None, link_index=None):
        if nkpts is None: nkpts = self.nkpts
        if target_k is None: target_k = self.target_k
        nelec = _unpack_nelec(nelec, self.spin)
        return make_hamiltonian_k(h1e, eri, norb, nelec, nkpts, target_k,
                                  link_index=link_index)

    def energy(self, h1e, eri, fcivec, norb, nelec, nkpts=None,
               target_k=None, link_index=None):
        if nkpts is None: nkpts = self.nkpts
        if target_k is None: target_k = self.target_k
        nelec = _unpack_nelec(nelec, self.spin)
        return energy(h1e, eri, fcivec, norb, nelec, nkpts, target_k,
                      link_index=link_index)

    def make_rdm1s_py(self, fcivec, norb, nelec, nkpts=None, target_k=None,
                      link_index=None):
        if nkpts is None: nkpts = self.nkpts
        if target_k is None: target_k = self.target_k
        nelec = _unpack_nelec(nelec, self.spin)
        return make_rdm1s_py(fcivec, norb, nelec, nkpts, target_k,
                             link_index=link_index, spin=self.spin)

    def make_rdm1_py(self, fcivec, norb, nelec, nkpts=None, target_k=None,
                     link_index=None):
        if nkpts is None: nkpts = self.nkpts
        if target_k is None: target_k = self.target_k
        nelec = _unpack_nelec(nelec, self.spin)
        return make_rdm1_py(fcivec, norb, nelec, nkpts, target_k,
                            link_index=link_index, spin=self.spin)

    def make_rdm12s_py(self, fcivec, norb, nelec, nkpts=None, target_k=None,
                       link_index=None, reorder=True):
        if nkpts is None: nkpts = self.nkpts
        if target_k is None: target_k = self.target_k
        nelec = _unpack_nelec(nelec, self.spin)
        return make_rdm12s_py(fcivec, norb, nelec, nkpts, target_k,
                              link_index=link_index, reorder=reorder,
                              spin=self.spin)

    def make_rdm12_py(self, fcivec, norb, nelec, nkpts=None, target_k=None,
                      link_index=None, reorder=True):
        if nkpts is None: nkpts = self.nkpts
        if target_k is None: target_k = self.target_k
        nelec = _unpack_nelec(nelec, self.spin)
        return make_rdm12_py(fcivec, norb, nelec, nkpts, target_k,
                             link_index=link_index, reorder=reorder,
                             spin=self.spin)

    def make_rdm1s(self, fcivec, norb, nelec, nkpts=None, target_k=None,
                   link_index=None):
        return self.make_rdm1s_py(fcivec, norb, nelec, nkpts=nkpts,
                                  target_k=target_k, link_index=link_index)

    def make_rdm1(self, fcivec, norb, nelec, nkpts=None, target_k=None,
                  link_index=None):
        return self.make_rdm1_py(fcivec, norb, nelec, nkpts=nkpts,
                                 target_k=target_k, link_index=link_index)

    def make_rdm12s(self, fcivec, norb, nelec, nkpts=None, target_k=None,
                    link_index=None, reorder=True):
        return self.make_rdm12s_py(fcivec, norb, nelec, nkpts=nkpts,
                                   target_k=target_k, link_index=link_index,
                                   reorder=reorder)

    def make_rdm12(self, fcivec, norb, nelec, nkpts=None, target_k=None,
                   link_index=None, reorder=True):
        return self.make_rdm12_py(fcivec, norb, nelec, nkpts=nkpts,
                                  target_k=target_k, link_index=link_index,
                                  reorder=reorder)

    def contract_ss(self, fcivec, norb, nelec, nkpts=None, target_k=None,
                    link_index=None):
        if nkpts is None: nkpts = self.nkpts
        if target_k is None: target_k = self.target_k
        nelec = _unpack_nelec(nelec, self.spin)
        return contract_ss(fcivec, norb, nelec, nkpts, target_k,
                           link_index=link_index, spin=self.spin)

    def spin_square(self, fcivec, norb, nelec, nkpts=None, target_k=None,
                    link_index=None, **kwargs):
        if nkpts is None: nkpts = self.nkpts
        if target_k is None: target_k = self.target_k
        nelec = _unpack_nelec(nelec, self.spin)
        return spin_square(fcivec, norb, nelec, nkpts, target_k,
                           link_index=link_index, spin=self.spin, **kwargs)

    def make_precond(self, hdiag, pspaceig=None, pspaceci=None, addr=None):
        return make_diag_precond(hdiag, pspaceig, pspaceci, addr,
                                 self.level_shift)

    def get_init_guess(self, norb, nelec, nkpts, target_k, nroots, hdiag):
        return get_init_guess_k(norb, nelec, nkpts, target_k, nroots, hdiag)

    def kernel(self, h1e, eri, norb, nelec, ci0=None, nkpts=None,
               target_k=None, tol=None, lindep=None, max_cycle=None,
               max_space=None, nroots=None, davidson_only=None,
               pspace_size=None, orbsym=None, wfnsym=None, ecore=0,
               **kwargs):
        if nkpts is None:
            nkpts = self.nkpts
        if nkpts is None:
            nkpts = h1e.shape[0]
        if target_k is None:
            target_k = self.target_k

        link_index = _unpack(norb, nelec, None, nkpts, spin=self.spin)
        e, c = kernel_ms1(self, h1e, eri, norb, nelec, nkpts, target_k,
                          ci0=ci0, link_index=link_index, tol=tol,
                          lindep=lindep, max_cycle=max_cycle,
                          max_space=max_space, nroots=nroots,
                          davidson_only=davidson_only,
                          pspace_size=pspace_size, ecore=ecore, **kwargs)
        self.eci, self.ci = e, c
        return e, c

    def fix_spin_(self, shift=0.1, ss=None, **kwargs):
        return fix_spin_(self, shift=shift, ss=ss, **kwargs)

    fix_spin = fix_spin_

    def eig(self, op, x0=None, precond=None, **kwargs):
        if isinstance(op, np.ndarray):
            hermi_err = np.linalg.norm(op - op.conj().T)
            if hermi_err < HERMI_THRESH:
                self.converged = True
                return scipy.linalg.eigh(op)
            self.converged = True
            return scipy.linalg.eig(op)

        self.converged, e, ci = \
                lib.davidson1(lambda xs: [op(x) for x in xs],
                              x0, precond, lessio=self.lessio, **kwargs)

        if kwargs.get("nroots", 1) == 1:
            self.converged = self.converged[0]
            e = e[0]
            ci = ci[0]
        return e, ci

FCI = FCISolver

if __name__ == '__main__':
    
    TEST1 = False
    TEST2 = False
    TEST3 = False
    TEST4 = True
    if TEST1:
        ncastot = 8
        nelectot = (4, 4)
        nkpts = 2 #(3, 3, 1)
        link_indexa, link_indexb = _unpack(ncastot, nelectot, None, nkpts, spin=None)
        print("link_indexa shape:", link_indexa.shape)
        print("link_indexb shape:", link_indexb.shape)
        print("----")

        # Possible k0 sectors:
        print("Possible k0 sectors: alpha str", np.unique(link_indexa[:, :, 4]))
        print("Possible k0 sectors: beta str", np.unique(link_indexb[:, :, 4])) 
        print("----")
        # Compare the alpha and beta string counts per k0 sector. 
        # They should be the same for spin-0 cases.
        det_count = kcistrings._count_det_per_k((link_indexa, link_indexb))
        print("Determinant count alpha:", det_count[0])
        print("Determinant count  beta:", det_count[1])
        print("----")
        # Now for det_alpha * det_beta, the total number of determinants 
        # would be k0 = (Ka + Kb) % nkpts. 
        # Let's count how many determinants we have in each k0 sector.
        counts_det = {k: 0 for k in range(nkpts)}
        for Ka, Na in det_count[0].items():
            for Kb, Nb in det_count[1].items():
                Kdet = (Ka + Kb) % nkpts
                if Kdet == 0:
                    print(f"Ka={Ka}, Kb={Kb}, Na={Na}, Nb={Nb}, Kdet={Kdet}")
                counts_det[Kdet] += Na * Nb
        
        print(counts_det)
        print("----")
        # When I will solve the kFCI problem, I will be solving it for the one of
        # the k0 sectors. Overall, the total number of determinants would be appro.
        # ntot_det / nkpts. Which is reduction in total number of determinants, but 
        # this will be huge headache to workout the proper vectorization.
        
        # Anyways:
        # Now compare with the total number of determinants from cistring.
        from pyscf.fci import cistring
        na, nb = _unpack_nelec(nelectot, spin=None)
        strsa = cistring.gen_linkstr_index(range(ncastot), na)
        strsb = cistring.gen_linkstr_index(range(ncastot), nb)
        tot_det = len(strsa) * len(strsb)

        print("Total determinants from cistring:", tot_det)
        print("Total determinants from link_index:", sum(counts_det.values()))
        print("Comparison at one of k-pts:", tot_det/list(counts_det.values())[0]) # Almost an order of magnitude.
        print("----")
    

    if TEST2:
        rng = np.random.default_rng(12)

        nkpts = 4
        ncas = 4
        norb = nkpts * ncas

        nelec = (2, 2)
        kindx = 0

        h1e = (
            rng.normal(size=(nkpts, ncas, ncas))
            + 1j * rng.normal(size=(nkpts, ncas, ncas))
        )

        # Optional: make each h1e[k] Hermitian
        for k in range(nkpts):
            h1e[k] = 0.5 * (h1e[k] + h1e[k].conj().T)

        link_index = None
        link_indexa, link_indexb = _unpack(norb, nelec, link_index, nkpts)

        for kindx in range(nkpts):
            blocks = gen_k_sector_linkstr_info(link_indexa, link_indexb, nkpts, kindx)
            sector_size = int(blocks[:, 5].sum())

            fcivec = (
                rng.normal(size=sector_size)
                + 1j * rng.normal(size=sector_size)
            )
            fcivec /= np.linalg.norm(fcivec)

            sigma_ci = contract_1e_k(
                h1e,
                fcivec,
                norb,
                nelec,
                nkpts,
                kindx,
                link_index=None
            )

            print("blocks:")
            print(blocks)
            print("sector_size =", sector_size)
            print("fcivec shape   =", fcivec.shape)
            print("sigma_ci shape =", sigma_ci.shape)
            print("||fcivec||     =", np.linalg.norm(fcivec))
            print("||sigma_ci||   =", np.linalg.norm(sigma_ci))
            # print("sigma_ci[:10]  =", sigma_ci[:10])


    if TEST3:
        rng = np.random.default_rng(12)

        nkpts = 3
        ncas = 2
        norb = nkpts * ncas

        nelec = (1*nkpts, 1*nkpts)

        link_index = None
        link_indexa, link_indexb = _unpack(norb, nelec, link_index, nkpts)

        eri = (
            rng.normal(size=(nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas))
            + 1j * rng.normal(size=(nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas))
        )

        for kindx in range(nkpts):
            blocks = gen_k_sector_linkstr_info(
                link_indexa,
                link_indexb,
                nkpts,
                kindx,
            )

            sector_size = int(blocks[:, 5].sum())

            fcivec = (
                rng.normal(size=sector_size)
                + 1j * rng.normal(size=sector_size)
            )

            fcivec /= np.linalg.norm(fcivec)

            sigma_ci = contract_2e_k(
                eri,
                fcivec,
                norb,
                nelec,
                nkpts,
                kindx,
                link_index=None,
            )

            print("=" * 80)
            print("target momentum sector kindx =", kindx)
            print("blocks:")
            print(blocks)
            print("sector_size      =", sector_size)
            print("fcivec shape     =", fcivec.shape)
            print("sigma_ci shape   =", sigma_ci.shape)
            print("||fcivec||       =", np.linalg.norm(fcivec))
            print("||sigma_ci||     =", np.linalg.norm(sigma_ci))
            print("finite sigma?    =", np.all(np.isfinite(sigma_ci)))
    

        from pyscf import fci

        from mrh.my_pyscf.pbc.fci import direct_spin1_cplx_opt
        # from mrh.my_pyscf.pbc.fci.direct_spin1_kfci import contract_2e_k
        # from mrh.my_pyscf.pbc.fci.direct_spin1_kfci import _unpack
        # from mrh.my_pyscf.pbc.fci.kcistrings import gen_k_sector_maps
        # from mrh.my_pyscf.pbc.fci.kcistrings import gen_k_sector_linkstr_info

        def eri_k_to_full(eri_k):
            nkpts, ncas = eri_k.shape[0], eri_k.shape[-1]
            norb = nkpts * ncas
            eri_full = np.zeros((norb, norb, norb, norb), dtype=eri_k.dtype)
            for kp, kq, kr in np.ndindex(nkpts, nkpts, nkpts):
                ks = (kp - kq + kr) % nkpts
                P, Q, R, S = kp * ncas, kq * ncas, kr * ncas, ks * ncas
                eri_full[P:P + ncas, Q:Q + ncas, R:R + ncas, S:S + ncas] = eri_k[kp, kq, kr]
            return eri_full
        
        def eri_full_to_k(eri_full, nkpts, ncas):
            ef = eri_full.reshape(nkpts, ncas, nkpts, ncas, nkpts, ncas, nkpts, ncas)
            eri_k = np.zeros((nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas), dtype=eri_full.dtype)
            for kp, kq, kr in np.ndindex(nkpts, nkpts, nkpts):
                ks = (kp - kq + kr) % nkpts
                eri_k[kp, kq, kr] = ef[kp, :, kq, :, kr, :, ks, :]
            return eri_k
      
        def embed_sector_fcivec_to_full_ci(fcivec_k, blocks, straid_k, strbid_k, 
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

        
        def extract_sector_from_full_ci(ci_full, blocks, straid_k, strbid_k):
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
    

        def compare_kfci_nkpts_gt1_vs_full_mol(
            nkpts=3,
            ncas=3,
            nelec=(2, 1),
            target_k=0,
            seed=12,
        ):
            rng = np.random.default_rng(seed)

            norb = nkpts * ncas

            link_index = None

            link_indexa, link_indexb = _unpack(
                norb,
                nelec,
                link_index,
                nkpts,
            )

            straid_k, strbid_k, str2tot_a, str2tot_b = gen_k_sector_maps(
                link_indexa,
                link_indexb,
                nkpts,
            )

            blocks = gen_k_sector_linkstr_info(
                link_indexa,
                link_indexb,
                nkpts,
                target_k,
            )

            sector_size = int(blocks[:, 5].sum())

            fcivec_k = (
                rng.normal(size=sector_size)
                + 1j * rng.normal(size=sector_size)
            )
            fcivec_k /= np.linalg.norm(fcivec_k)

            eri_k = (
                rng.normal(
                    size=(nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)
                )
                + 1j * rng.normal(
                    size=(nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)
                )
            )

            eri_full = eri_k_to_full(eri_k)

            eri_full = 0.5 * (
                eri_full
                + eri_full.transpose(2, 3, 0, 1)
            )

            eri_k = eri_full_to_k(
                eri_full,
                nkpts,
                ncas,
            )

            eri_full = eri_k_to_full(eri_k)

            nstra_total = fci.cistring.num_strings(norb, nelec[0])
            nstrb_total = fci.cistring.num_strings(norb, nelec[1])

            ci_full = embed_sector_fcivec_to_full_ci(
                fcivec_k,
                blocks,
                straid_k,
                strbid_k,
                nstra_total,
                nstrb_total,
            )

            sigma_k = contract_2e_k(
                eri_k,
                fcivec_k,
                norb,
                nelec,
                nkpts,
                target_k,
                link_index=None,
            )

            sigma_full = direct_spin1_cplx_opt.contract_2e(
                eri_full,
                ci_full,
                norb,
                nelec,
                link_index=None,
            )

            sigma_ref_k = extract_sector_from_full_ci(
                sigma_full,
                blocks,
                straid_k,
                strbid_k,
            )

            diff = sigma_k - sigma_ref_k

            return sigma_k, sigma_ref_k, diff
    

        import unittest

        class KnownValues(unittest.TestCase):

            def test_contract_2e_k_nkpts_gt1_vs_full_mol(self):
                test_cases = [
                    (2, 3, (1, 1)),
                    (2, 3, (2, 0)),
                    (2, 3, (0, 2)),
                    (2, 3, (2, 1)),
                    (2, 3, (2, 2)),
                    (3, 2, (1, 1)),
                    (3, 2, (2, 1)),
                    (3, 4, (2, 2)),
                    (4, 2, (1, 1)),
                    (5, 2, (1, 1)),
                ]

                for nkpts, ncas, nelec in test_cases:
                    for target_k in range(nkpts):
                        with self.subTest(
                            nkpts=nkpts,
                            ncas=ncas,
                            nelec=nelec,
                            target_k=target_k,
                        ):
                            sigma_k, sigma_ref_k, diff = (
                                compare_kfci_nkpts_gt1_vs_full_mol(
                                    nkpts=nkpts,
                                    ncas=ncas,
                                    nelec=nelec,
                                    target_k=target_k,
                                    seed=12,
                                )
                            )

                            self.assertEqual(
                                sigma_k.shape,
                                sigma_ref_k.shape,
                            )

                            print("=" * 80)
                            print(
                                    f"contract_2e_k failed nkpts>1 full-space "
                                    f"reference test for "
                                    f"nkpts={nkpts}, ncas={ncas}, "
                                    f"nelec={nelec}, target_k={target_k}. "
                                    f"||diff||={np.linalg.norm(diff)}, "
                                    f"||ref||={np.linalg.norm(sigma_ref_k)}, "
                                    f"rel={np.linalg.norm(diff) / max(np.linalg.norm(sigma_ref_k), 1e-14)}, "
                                    f"max_abs={np.max(np.abs(diff))}"
                                )
                            self.assertTrue(
                                np.allclose(
                                    sigma_k,
                                    sigma_ref_k,
                                    atol=1e-12,
                                    rtol=1e-12,
                                ),
                                msg=(
                                    f"contract_2e_k failed nkpts>1 full-space "
                                    f"reference test for "
                                    f"nkpts={nkpts}, ncas={ncas}, "
                                    f"nelec={nelec}, target_k={target_k}. "
                                    f"||diff||={np.linalg.norm(diff)}, "
                                    f"||ref||={np.linalg.norm(sigma_ref_k)}, "
                                    f"rel={np.linalg.norm(diff) / max(np.linalg.norm(sigma_ref_k), 1e-14)}, "
                                    f"max_abs={np.max(np.abs(diff))}"
                                ),
                            )


        # if __name__ == "__main__":
        unittest.main()
