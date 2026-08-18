#!/usr/bin/env python

"""Momentum-sector FCI contractions for periodic systems."""

import ctypes

import numpy as np

from pyscf import lib
from pyscf.fci.addons import _unpack_nelec

from mrh.lib.helper import load_library
from mrh.my_pyscf.pbc.fci import kfci_helper, kcistrings
from mrh.my_pyscf.pbc.fci.kcistrings import (
    gen_k_sector_linkstr_info,
    gen_k_sector_maps,
)
from mrh.my_pyscf.pbc.fci.kfci_helper import (
    KFCIContractMap,
    _unpack_contract_link_index as _unpack,
    build_ab_pair_tables,
    build_k_links_spin,
    build_links_by_global_source_array,
    build_same_spin_pair_tables,
    make_kfci_contract_map,
)


# Author: Bhavnesh Jangid

logger = lib.logger
libpbcfci_k = None


def _timer_start():
    """Start a PySCF-style CPU/wall timer pair."""
    return logger.process_clock(), logger.perf_counter()


def _timer_debug1(obj, msg, t0):
    """Emit a DEBUG1 timer message using the PySCF logger API."""
    if obj is None:
        return logger.process_clock(), logger.perf_counter()
    log = logger.new_logger(obj)
    return log.timer_debug1(msg, *t0)


def _load_k_contract_lib():
    """Load and configure the C library for k-FCI contraction."""
    global libpbcfci_k
    if libpbcfci_k is None:
        t0 = _timer_start()
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
            ctypes.c_int,
        ]
        libpbcfci_k.FCIcontract_1e_k.restype = None
        libpbcfci_k.FCIcontract_2e_k.argtypes = [
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
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        libpbcfci_k.FCIcontract_2e_k.restype = None
        libpbcfci_k.FCIcontract_2e_k_stream_ab.argtypes = [
            ctypes.c_void_p,  # eri
            ctypes.c_void_p,  # ci0
            ctypes.c_void_p,  # ci1
            ctypes.c_int,     # nkpts
            ctypes.c_int,     # ncas
            ctypes.c_int,     # nblocks
            ctypes.c_void_p,  # blocks
            ctypes.c_void_p,  # linka
            ctypes.c_int,     # nstra
            ctypes.c_int,     # nlinka
            ctypes.c_void_p,  # linkb
            ctypes.c_int,     # nstrb
            ctypes.c_int,     # nlinkb
            ctypes.c_void_p,  # stra_ids
            ctypes.c_void_p,  # stra_offsets
            ctypes.c_void_p,  # strb_ids
            ctypes.c_void_p,  # strb_offsets
            ctypes.c_void_p,  # str2tot_a
            ctypes.c_void_p,  # str2tot_b
            ctypes.c_void_p,  # kneg
        ]
        libpbcfci_k.FCIcontract_2e_k_stream_ab.restype = None
        _timer_debug1(None, "k-FCI load C contract library", t0)
    return libpbcfci_k


def _as_contract_map(norb, nelec, nkpts, target_k, link_index=None,
                     contract_map=None, need_pair_tables=False,
                     explicit_ab="auto", log_obj=None, kmom=None):
    '''
    Helper function to ensure that a KFCIContractMap is available for the given
    k-FCI contraction. If a contract_map is provided, it checks for consistency
    with the provided parameters. If not, it creates a new KFCIContractMap.
    args:
        norb : int
            Total number of orbitals across all k-points.
        nelec : tuple of 2 ints
            Number of alpha and beta electrons.
        nkpts : int
            Number of k-points / momentum sectors.
        target_k : int
            Total momentum sector for the k-FCI contraction.
        link_index : tuple of 2 ndarrays or None
            Look up tables/link index for alpha and beta strings.
            If None, it will be generated on the fly.
        contract_map : KFCIContractMap or None
            Precomputed contraction map. If None, a new one will be created.
        need_pair_tables : bool
            If True, ensures that pair tables are built in the contract_map.
    '''
    t0 = _timer_start()
    if contract_map is None and isinstance(link_index, KFCIContractMap):
        contract_map = link_index
        link_index = None

    if contract_map is None:
        contract_map = make_kfci_contract_map(
            norb, nelec, nkpts, target_k,
            link_index=link_index,
            build_pair_tables=need_pair_tables,
            explicit_ab=explicit_ab,
            kmom=kmom,
        )
        _timer_debug1(log_obj, "k-FCI build contract map", t0)
        return contract_map

    assert contract_map.norb == int(norb)
    assert contract_map.nkpts == int(nkpts)
    assert contract_map.ncas * contract_map.nkpts == contract_map.norb
    assert contract_map.target_k == int(target_k) % int(nkpts)
    assert tuple(contract_map.nelec) == tuple(_unpack_nelec(nelec))

    if need_pair_tables and not getattr(contract_map, "has_pair_tables", True):
        contract_map = make_kfci_contract_map(
            norb, nelec, nkpts, target_k,
            link_index=contract_map.link_index,
            build_pair_tables=True,
            explicit_ab=explicit_ab,
            kmom=kmom,
        )
        _timer_debug1(
            log_obj, "k-FCI rebuild contract map with pair tables", t0)
        return contract_map
    if explicit_ab is True and not getattr(contract_map, "explicit_ab", True):
        contract_map = make_kfci_contract_map(
            norb, nelec, nkpts, target_k,
            link_index=contract_map.link_index,
            build_pair_tables=need_pair_tables,
            explicit_ab=True,
            kmom=kmom,
        )
        _timer_debug1(
            log_obj, "k-FCI rebuild contract map with explicit AB", t0)
        return contract_map
    _timer_debug1(log_obj, "k-FCI validate contract map", t0)
    return contract_map


def contract_1e_k_py(h1e, fcivec, norb, nelec, nkpts, kindx,
                     link_index=None, kmom=None):
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
            Result of the Hamiltonian-vector product in the target momentum
            sector.
    '''

    t0 = _timer_start()
    link_indexa, link_indexb = _unpack(norb, nelec, link_index, nkpts,
                                       kmom=kmom)
    t0 = _timer_debug1(None, "k-FCI contract_1e_py link_index", t0)
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
    blocks = gen_k_sector_linkstr_info(link_indexa, link_indexb, nkpts,
                                       kindx, kmom=kmom)
    sector_size = int(blocks[:, 5].sum())
    t0 = _timer_debug1(None, "k-FCI contract_1e_py sector blocks", t0)

    assert fcivec.size == sector_size

    straid_k, strbid_k, tota_2k, totb_2k = gen_k_sector_maps(
        link_indexa, link_indexb, nkpts, kmom=kmom)
    t0 = _timer_debug1(None, "k-FCI contract_1e_py sector maps", t0)

    # Making sure fcivec is in the right dtype and C-contiguous.
    h1e = np.asarray(h1e, dtype=dtype, order="C")
    fcivec = np.asarray(fcivec, dtype=dtype, order="C")
    sigma_ci = np.zeros(fcivec.shape, dtype=dtype, order="C")
    t0 = _timer_debug1(None, "k-FCI contract_1e_py array setup", t0)

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

                # h1e[k, p, q] is k-diagonal, so only k_cre == k_des
                # contributes.
                # which means only dk=0 contributes.
                if (k_cre != k_des) or (dK != 0):
                    continue

                # Note that p and q are in the global orbital indexing,
                # but h1e is in the k-space orbital indexing, so we need
                # to mod by ncas to get the correct orbital indices for h1e.
                hpq = h1e[k_cre, p % ncas, q % ncas]

                # Skip excitations that leave this momentum sector.
                ia1_local = tota_2k[ka, astr1]
                if ia1_local < 0:
                    continue
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
                # h1e[k, p, q] is k-diagonal, so only k_cre == k_des
                # contributes.
                # which means only dk=0 contributes.
                if (k_cre != k_des) or (dK != 0):
                    continue

                hpq = h1e[k_cre, p % ncas, q % ncas]

                # Skip excitations that leave this momentum sector.
                ib1_local = totb_2k[kb, bstr1]
                if ib1_local < 0:
                    continue

                Sblk[:, ib1_local] += sign * hpq * Cblk[:, ib0_local]

    _timer_debug1(None, "k-FCI contract_1e_py contraction loops", t0)
    return sigma_ci


def contract_1e_k(h1e, fcivec, norb, nelec, nkpts, kindx,
                  link_index=None, contract_map=None, log_obj=None,
                  kmom=None):
    '''
    C implementation of contract_1e_k using structural k-sector contraction
    maps.  The result is returned as complex128 to match the C kernel.
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
        contract_map : KFCIContractMap or None
            Precomputed contraction map. If None, a new one will be created.
    returns:
        sigma_ci : ndarray, shape (sector_size,)
            Result of the Hamiltonian-vector product in the target momentum
            sector.
    '''
    nkpts = int(nkpts)
    ncas = int(norb) // nkpts
    assert ncas * nkpts == int(norb)

    t0 = _timer_start()
    contract_map = _as_contract_map(
        norb, nelec, nkpts, kindx, link_index=link_index,
        contract_map=contract_map, log_obj=log_obj, kmom=kmom)
    assert fcivec.size == contract_map.sector_size
    t0 = _timer_debug1(log_obj, "k-FCI contract_1e map setup", t0)

    h1e = np.asarray(h1e, dtype=np.complex128, order="C")
    fcivec = np.asarray(fcivec, dtype=np.complex128, order="C")
    sigma_ci = np.zeros(fcivec.shape, dtype=np.complex128, order="C")
    t0 = _timer_debug1(log_obj, "k-FCI contract_1e array setup", t0)

    assert h1e.shape == (nkpts, ncas, ncas)
    link_indexa, link_indexb = contract_map.link_index

    libpbcfci = _load_k_contract_lib()
    with lib.with_omp_threads(lib.num_threads()):
        libpbcfci.FCIcontract_1e_k(
            h1e.ctypes.data_as(ctypes.c_void_p),
            fcivec.ctypes.data_as(ctypes.c_void_p),
            sigma_ci.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nkpts),
            ctypes.c_int(ncas),
            ctypes.c_int(contract_map.blocks.shape[0]),
            contract_map.blocks.ctypes.data_as(ctypes.c_void_p),
            link_indexa.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(link_indexa.shape[0]),
            ctypes.c_int(link_indexa.shape[1]),
            link_indexb.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(link_indexb.shape[0]),
            ctypes.c_int(link_indexb.shape[1]),
            contract_map.stra_ids.ctypes.data_as(ctypes.c_void_p),
            contract_map.stra_offsets.ctypes.data_as(ctypes.c_void_p),
            contract_map.strb_ids.ctypes.data_as(ctypes.c_void_p),
            contract_map.strb_offsets.ctypes.data_as(ctypes.c_void_p),
            contract_map.str2tot_a.ctypes.data_as(ctypes.c_void_p),
            contract_map.str2tot_b.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(contract_map.kmom.zero),
        )
    _timer_debug1(log_obj, "k-FCI contract_1e C kernel", t0)
    return sigma_ci


def sector_size(norb, nelec, nkpts, target_k=0, link_index=None,
                kmom=None):
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
    t0 = _timer_start()
    link_indexa, link_indexb = _unpack(norb, nelec, link_index, nkpts,
                                       kmom=kmom)
    blocks = gen_k_sector_linkstr_info(link_indexa, link_indexb, nkpts,
                                       target_k, kmom=kmom)
    t0 = _timer_debug1(None, "k-FCI sector_size setup", t0)
    if blocks.size == 0:
        return 0
    size = int(blocks[:, 5].sum())
    _timer_debug1(None, "k-FCI sector_size sum", t0)
    return size


def _get_ci_sectors(fcivec, blocks, nkpts):
    '''
    Extract blocked CI vectors using k-sector information.
    '''
    ci_blocks = [[None for _ in range(nkpts)]
                 for _ in range(nkpts)]
    for blk in blocks:
        ka, kb, nstra, nstrb, offset, size = map(int, blk)
        ci_blocks[ka][kb] = fcivec[offset:offset + size].reshape(nstra, nstrb)
    return ci_blocks


def contract_2e_k_py(eri, fcivec, norb, nelec, nkpts, target_k,
                     link_index=None, kmom=None):
    '''
    Contract the two-electron Hamiltonian with a fixed-sector k-FCI vector.
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
            Result of the Hamiltonian-vector product in the target momentum
            sector.
    '''
    t0 = _timer_start()
    nkpts = eri.shape[0]
    dtype = np.result_type(eri.dtype, fcivec.dtype)

    link_indexa, link_indexb = _unpack(norb, nelec, link_index, nkpts,
                                       kmom=kmom)
    straid_k, strbid_k, str2tot_a, str2tot_b = gen_k_sector_maps(
        link_indexa, link_indexb, nkpts, kmom=kmom)
    t0 = _timer_debug1(None, "k-FCI contract_2e_py link and sector maps", t0)

    links_a = build_k_links_spin(link_indexa, norb, nkpts, straid_k,
                                 str2tot_a, kmom=kmom)
    links_b = build_k_links_spin(link_indexb, norb, nkpts, strbid_k,
                                 str2tot_b, kmom=kmom)
    t0 = _timer_debug1(None, "k-FCI contract_2e_py spin link tables", t0)

    links_a = build_links_by_global_source_array(links_a)
    links_b = build_links_by_global_source_array(links_b)
    t0 = _timer_debug1(None, "k-FCI contract_2e_py source link indices", t0)

    ab_pairs = build_ab_pair_tables(links_a, links_b, nkpts, kmom=kmom)
    aa_pairs = build_same_spin_pair_tables(links_a, nkpts, kmom=kmom)
    bb_pairs = build_same_spin_pair_tables(links_b, nkpts, kmom=kmom)
    t0 = _timer_debug1(None, "k-FCI contract_2e_py pair tables", t0)

    sigma_ci = np.zeros(fcivec.shape, dtype=dtype, order="C")

    blocks = gen_k_sector_linkstr_info(link_indexa, link_indexb, nkpts,
                                       target_k, kmom=kmom)

    # The k-sector blocks must span the full input vector.
    sector_size = int(blocks[:, 5].sum())
    assert fcivec.size == sector_size

    # Rearrange the CI vectors into alpha/beta momentum blocks.
    ci0_blocks = _get_ci_sectors(fcivec, blocks, nkpts)
    ci1_blocks = _get_ci_sectors(sigma_ci, blocks, nkpts)
    t0 = _timer_debug1(None, "k-FCI contract_2e_py CI block setup", t0)

    # Free up some memory.
    blocks = None

    kmom = kcistrings._as_kmom(nkpts, kmom=kmom)
    for ka in range(nkpts):
        kb = kcistrings._ksub(kmom, target_k, ka)

        if ci0_blocks[ka][kb] is None:
            continue

        kfci_helper.contract_ab_pairs(
            eri, ci0_blocks[ka][kb], ci1_blocks, ab_pairs, ka, kb)

        kfci_helper.contract_aa_pairs(eri, ci0_blocks, ci1_blocks,
                                      aa_pairs, ka, kb)

        kfci_helper.contract_bb_pairs(eri, ci0_blocks, ci1_blocks,
                                      bb_pairs, ka, kb)

    _timer_debug1(None, "k-FCI contract_2e_py contraction loops", t0)
    return sigma_ci


def contract_2e_k(eri, fcivec, norb, nelec, nkpts, target_k,
                  link_index=None, contract_map=None, log_obj=None,
                  kmom=None):
    '''
    C implementation using structural k-sector contraction maps.
    The same-spin contractions are applied with zgemm and the alpha-beta terms
    are packed into sparse source/destination block groups.
    OpenMP threads follow lib.num_threads().
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
        link_index : tuple of 2 ndarrays or None
            Look up tables/link index for alpha and beta strings.
            If None, it will be generated on the fly.
        contract_map : KFCIContractMap or None
            Precomputed contraction map. If None, a new one will be created.
    returns:
        sigma_ci : ndarray, shape (sector_size,)
            Result of the Hamiltonian-vector product in the target momentum
            sector.
    '''
    nkpts = int(nkpts)
    ncas = int(norb) // nkpts
    assert ncas * nkpts == int(norb)

    t0 = _timer_start()
    contract_map = _as_contract_map(
        norb, nelec, nkpts, target_k, link_index=link_index,
        contract_map=contract_map, log_obj=log_obj, kmom=kmom)
    assert fcivec.size == contract_map.sector_size
    t0 = _timer_debug1(log_obj, "k-FCI contract_2e map setup", t0)

    eri = np.asarray(eri, dtype=np.complex128, order="C")
    fcivec = np.asarray(fcivec, dtype=np.complex128, order="C")
    sigma_ci = np.zeros(fcivec.shape, dtype=np.complex128, order="C")
    t0 = _timer_debug1(log_obj, "k-FCI contract_2e array setup", t0)

    assert eri.shape == (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)

    libpbcfci = _load_k_contract_lib()
    kernel = libpbcfci.FCIcontract_2e_k
    with lib.with_omp_threads(lib.num_threads()):
        kernel(
            eri.ctypes.data_as(ctypes.c_void_p),
            fcivec.ctypes.data_as(ctypes.c_void_p),
            sigma_ci.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nkpts),
            ctypes.c_int(ncas),
            ctypes.c_int(contract_map.blocks.shape[0]),
            contract_map.blocks.ctypes.data_as(ctypes.c_void_p),
            contract_map.ab_group_tab.ctypes.data_as(ctypes.c_void_p),
            contract_map.ab_group_offsets.ctypes.data_as(ctypes.c_void_p),
            contract_map.ab_src_addr.ctypes.data_as(ctypes.c_void_p),
            contract_map.ab_dst_addr.ctypes.data_as(ctypes.c_void_p),
            contract_map.ab_sign.ctypes.data_as(ctypes.c_void_p),
            contract_map.ab_eri_idx_ab.ctypes.data_as(ctypes.c_void_p),
            contract_map.ab_eri_idx_ba.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(contract_map.ab_src_addr.size),
            contract_map.aa_group_tab.ctypes.data_as(ctypes.c_void_p),
            contract_map.aa_group_offsets.ctypes.data_as(ctypes.c_void_p),
            contract_map.aa_src_addr.ctypes.data_as(ctypes.c_void_p),
            contract_map.aa_dst_addr.ctypes.data_as(ctypes.c_void_p),
            contract_map.aa_sign.ctypes.data_as(ctypes.c_void_p),
            contract_map.aa_eri_idx.ctypes.data_as(ctypes.c_void_p),
            contract_map.bb_group_tab.ctypes.data_as(ctypes.c_void_p),
            contract_map.bb_group_offsets.ctypes.data_as(ctypes.c_void_p),
            contract_map.bb_src_addr.ctypes.data_as(ctypes.c_void_p),
            contract_map.bb_dst_addr.ctypes.data_as(ctypes.c_void_p),
            contract_map.bb_sign.ctypes.data_as(ctypes.c_void_p),
            contract_map.bb_eri_idx.ctypes.data_as(ctypes.c_void_p),
        )
        if not getattr(contract_map, "explicit_ab", True):
            link_indexa, link_indexb = contract_map.link_index
            libpbcfci.FCIcontract_2e_k_stream_ab(
                eri.ctypes.data_as(ctypes.c_void_p),
                fcivec.ctypes.data_as(ctypes.c_void_p),
                sigma_ci.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nkpts),
                ctypes.c_int(ncas),
                ctypes.c_int(contract_map.blocks.shape[0]),
                contract_map.blocks.ctypes.data_as(ctypes.c_void_p),
                link_indexa.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(link_indexa.shape[0]),
                ctypes.c_int(link_indexa.shape[1]),
                link_indexb.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(link_indexb.shape[0]),
                ctypes.c_int(link_indexb.shape[1]),
                contract_map.stra_ids.ctypes.data_as(ctypes.c_void_p),
                contract_map.stra_offsets.ctypes.data_as(ctypes.c_void_p),
                contract_map.strb_ids.ctypes.data_as(ctypes.c_void_p),
                contract_map.strb_offsets.ctypes.data_as(ctypes.c_void_p),
                contract_map.str2tot_a.ctypes.data_as(ctypes.c_void_p),
                contract_map.str2tot_b.ctypes.data_as(ctypes.c_void_p),
                contract_map.kmom.kneg.ctypes.data_as(ctypes.c_void_p),
            )
    _timer_debug1(log_obj, "k-FCI contract_2e C kernel", t0)
    return sigma_ci
