#!/usr/bin/env python
"""Full CI support for spin-free periodic Hamiltonians at fixed momentum.

The determinant basis has fixed numbers of alpha and beta electrons, and
therefore fixed particle number and :math:`M_S`. Determinants are grouped by
the momenta of their alpha and beta strings. Only blocks whose combined
momentum equals ``target_k`` are stored in the packed CI vector.

The symmetry support is:

=========================================  =========
Symmetry                                   Supported
=========================================  =========
Lattice translation / crystal momentum    Yes
Particle number and :math:`M_S`            Yes
Total spin / singlet adaptation            No
Point-group or additional space-group      No
Time-reversal or Kramers                    No
Hermitian Hamiltonian, real or complex     Yes
Alpha/beta orbital degeneracy               Yes
=========================================  =========

``Alpha/beta orbital degeneracy`` means that alpha and beta electrons use the
same spatial orbitals and integrals. Spin-dependent unrestricted Hamiltonians
are not supported. States of different total spin can occur in the same
:math:`M_S` sector; a spin penalty can target a desired spin, but it does not
make the determinant basis spin-adapted. ``orbsym`` and ``wfnsym`` are not
used.

The one-electron integrals must be block diagonal in k-point, and the
two-electron integrals must obey crystal-momentum conservation. No additional
point-group, time-reversal, or Kramers reduction is applied.
"""

import ctypes
import types
import warnings

import numpy as np
import scipy.linalg

from pyscf import lib
from pyscf.fci import cistring, direct_spin1
from pyscf.fci.addons import (
    SpinPenaltyFCISolver as _PySCFSpinPenaltyFCISolver,
    _unpack_nelec,
)

from mrh.lib.helper import load_library
from mrh.my_pyscf.pbc.fci import kfci_contract_map, kcistrings, krdm_helper
from mrh.my_pyscf.pbc.fci.kcistrings import (
    gen_k_sector_linkstr_info,
)
from mrh.my_pyscf.pbc.fci.kfci_contract_map import (
    KFCIContractMap,
    _unpack_contract_link_index as _unpack,
    make_kfci_contract_map,
)

# Author: Bhavnesh Jangid

logger = lib.logger
HDIAG_IMAG_TOL = 1e-3
HERMI_THRESH = 1e-8
libpbcfci_k = None
libpbckfci_hdiag = None

# Lookup table for the population count (number of set bits) of every
# possible byte.  FCI occupation strings use one bit per spatial orbital, so
# a population count is also a count of occupied orbitals.  Looking up all
# bytes with NumPy avoids a Python loop over determinant strings.
_UINT8_POPCOUNT = np.asarray(
    [bin(value).count("1") for value in range(256)], dtype=np.uint8)

def _popcount_uint64(values):
    """Return the number of set bits in each ``uint64`` occupation string.

    Each 64-bit value is viewed as eight bytes.  Those bytes index
    ``_UINT8_POPCOUNT``, and summing the eight lookup results gives the
    population count of the original value.  The byte order does not matter
    because only the sum is retained.  The result has the same shape as
    ``values`` and uses ``uint16`` as a safe accumulator type.

    This vectorized lookup avoids calling ``int.bit_count`` separately for
    every element and also supports Python versions on which that method is
    unavailable.
    """
    # A contiguous uint64 array can be reinterpreted reliably as eight bytes
    # per element without copying during ``view``.
    values = np.ascontiguousarray(values, dtype=np.uint64)
    byte_values = values.view(np.uint8).reshape(values.shape + (8,))
    return _UINT8_POPCOUNT[byte_values].sum(axis=-1, dtype=np.uint16)


def _load_k_contract_lib():
    """Load and configure the C entry points for k-FCI contractions.

    The library exports three kernels:

    ``FCIcontract_1e_k``
        Applies the one-electron Hamiltonian.  It receives the one-electron
        integrals, packed input/output CI vectors, the six-column momentum
        block table, alpha/beta link tables, string IDs and their momentum
        offsets, global-to-block-local string maps, and the momentum label
        representing zero transfer.  The kernel initializes the output.

    ``FCIcontract_2e_k``
        Applies the two-electron Hamiltonian from explicit structural maps.
        The AB, AA, and BB arrays give source/destination addresses, fermionic
        signs, and flattened ERI indices, grouped by source and destination
        momentum blocks.  The kernel initializes the output and evaluates all
        terms represented by those maps.  When ``explicit_ab`` is false, the
        AB arrays are empty, so this call evaluates only the mapped AA and BB
        terms before the streamed AB call below.

    ``FCIcontract_2e_k_stream_ab``
        Generates opposite-spin (alpha-beta) link pairs while contracting,
        instead of storing the potentially very large explicit AB map.  Here
        "stream" means on-the-fly traversal, not file or network I/O.  It is
        also not restricted to contractions outside a momentum block: it
        includes both within-block and between-block AB contributions.  An
        alpha transfer ``dK`` is paired with the beta transfer ``-dK``, so a
        source and destination may have different individual-spin momentum
        blocks while remaining in the same total-momentum sector.  The kernel
        accumulates into the output already containing the mapped AA/BB terms.

    Every pointer below refers to a caller-owned, C-contiguous NumPy array.
    ``ctypes`` records only the pointer/integer ABI; the Python call sites are
    responsible for supplying the documented dtype, shape, and lifetime.

    Returns
    -------
    ctypes.CDLL
        Cached handle to ``libpbc_fci_contract_k``.
    """
    global libpbcfci_k
    if libpbcfci_k is None:
        libpbcfci_k = load_library("libpbc_fci_contract_k")
        # FCIcontract_1e_k arguments, in C-signature order.
        libpbcfci_k.FCIcontract_1e_k.argtypes = [
            ctypes.c_void_p,  # h1e: complex128[nkpts, ncas, ncas]
            ctypes.c_void_p,  # ci0: input complex128[sector_size]
            ctypes.c_void_p,  # ci1: output complex128[sector_size]
            ctypes.c_int,     # nkpts: number of momentum points
            ctypes.c_int,     # ncas: active orbitals per k-point
            ctypes.c_int,     # nblocks: number of packed CI blocks
            ctypes.c_void_p,  # blocks: int32[nblocks, 6]
            ctypes.c_void_p,  # linka: int32[nstra, nlinka, 8]
            ctypes.c_int,     # nstra: total number of alpha strings
            ctypes.c_int,     # nlinka: alpha links per string
            ctypes.c_void_p,  # linkb: int32[nstrb, nlinkb, 8]
            ctypes.c_int,     # nstrb: total number of beta strings
            ctypes.c_int,     # nlinkb: beta links per string
            ctypes.c_void_p,  # stra_ids: alpha string IDs grouped by k
            ctypes.c_void_p,  # stra_offsets: offsets into stra_ids
            ctypes.c_void_p,  # strb_ids: beta string IDs grouped by k
            ctypes.c_void_p,  # strb_offsets: offsets into strb_ids
            ctypes.c_void_p,  # str2tot_a: alpha global-to-local map
            ctypes.c_void_p,  # str2tot_b: beta global-to-local map
            ctypes.c_int,     # dk_zero: label of zero momentum transfer
        ]
        libpbcfci_k.FCIcontract_1e_k.restype = None

        # FCIcontract_2e_k arguments, in C-signature order.  Each ``*_addr``
        # is local to its source/destination packed block; each ``*_eri_idx``
        # indexes the flattened complex ERI array.
        libpbcfci_k.FCIcontract_2e_k.argtypes = [
            ctypes.c_void_p,  # eri: flattened complex128 k-point ERIs
            ctypes.c_void_p,  # ci0: input complex128[sector_size]
            ctypes.c_void_p,  # ci1: output complex128[sector_size]
            ctypes.c_int,     # nkpts: number of momentum points
            ctypes.c_int,     # ncas: active orbitals per k-point
            ctypes.c_int,     # nblocks: number of packed CI blocks
            ctypes.c_void_p,  # blocks: int32[nblocks, 6]
            ctypes.c_void_p,  # ab_group_tab: [dst_offset, begin, end]
            ctypes.c_void_p,  # ab_group_offsets: groups by source block
            ctypes.c_void_p,  # ab_src_addr: AB source determinant addresses
            ctypes.c_void_p,  # ab_dst_addr: AB target determinant addresses
            ctypes.c_void_p,  # ab_sign: AB products of fermionic signs
            ctypes.c_void_p,  # ab_eri_idx_ab: ERI indices in AB ordering
            ctypes.c_void_p,  # ab_eri_idx_ba: ERI indices in BA ordering
            ctypes.c_int,     # nab_entries: number of explicit AB entries
            ctypes.c_void_p,  # aa_group_tab: [dst, dst_na, begin, end]
            ctypes.c_void_p,  # aa_group_offsets: groups by source block
            ctypes.c_void_p,  # aa_src_addr: alpha source-row addresses
            ctypes.c_void_p,  # aa_dst_addr: alpha target-row addresses
            ctypes.c_void_p,  # aa_sign: alpha-alpha fermionic signs
            ctypes.c_void_p,  # aa_eri_idx: alpha-alpha ERI indices
            ctypes.c_void_p,  # bb_group_tab: [dst, dst_nb, begin, end]
            ctypes.c_void_p,  # bb_group_offsets: groups by source block
            ctypes.c_void_p,  # bb_src_addr: beta source-column addresses
            ctypes.c_void_p,  # bb_dst_addr: beta target-column addresses
            ctypes.c_void_p,  # bb_sign: beta-beta fermionic signs
            ctypes.c_void_p,  # bb_eri_idx: beta-beta ERI indices
        ]
        libpbcfci_k.FCIcontract_2e_k.restype = None

        # FCIcontract_2e_k_stream_ab arguments.  Unlike the mapped kernel, it
        # needs raw link and string-layout tables to construct AB pairs on the
        # fly.  It adds to ci1 rather than clearing it.
        libpbcfci_k.FCIcontract_2e_k_stream_ab.argtypes = [
            ctypes.c_void_p,  # eri: flattened complex128 k-point ERIs
            ctypes.c_void_p,  # ci0: input complex128[sector_size]
            ctypes.c_void_p,  # ci1: accumulated complex128 output
            ctypes.c_int,     # nkpts: number of momentum points
            ctypes.c_int,     # ncas: active orbitals per k-point
            ctypes.c_int,     # nblocks: number of packed CI blocks
            ctypes.c_void_p,  # blocks: int32[nblocks, 6]
            ctypes.c_void_p,  # linka: int32[nstra, nlinka, 8]
            ctypes.c_int,     # nstra: total number of alpha strings
            ctypes.c_int,     # nlinka: alpha links per string
            ctypes.c_void_p,  # linkb: int32[nstrb, nlinkb, 8]
            ctypes.c_int,     # nstrb: total number of beta strings
            ctypes.c_int,     # nlinkb: beta links per string
            ctypes.c_void_p,  # stra_ids: alpha string IDs grouped by k
            ctypes.c_void_p,  # stra_offsets: offsets into stra_ids
            ctypes.c_void_p,  # strb_ids: beta string IDs grouped by k
            ctypes.c_void_p,  # strb_offsets: offsets into strb_ids
            ctypes.c_void_p,  # str2tot_a: alpha global-to-local map
            ctypes.c_void_p,  # str2tot_b: beta global-to-local map
            ctypes.c_void_p,  # kneg[dK]: additive inverse of dK
        ]
        libpbcfci_k.FCIcontract_2e_k_stream_ab.restype = None
    return libpbcfci_k


def _load_k_hdiag_lib():
    """Load and configure the C entry points for the k-FCI diagonal.

    ``FCIhdiag_k`` initializes ``hdiag`` and adds its one-electron, mapped AB,
    mapped AA, and mapped BB contributions.  In addition to the integral,
    block, link, and string-layout arrays, it receives the same explicit
    two-electron structural maps used by ``FCIcontract_2e_k``.  Only map
    entries whose source and destination are the same determinant contribute
    to the diagonal.

    ``FCIhdiag_k_stream_ab`` is called afterward when the contract map omits
    explicit AB entries.  It reconstructs and accumulates every diagonal AB
    contribution directly from the alpha/beta links.  Unlike the full
    streamed contraction, this routine is not an out-of-block contraction:
    a Hamiltonian diagonal element has identical source and destination
    determinants, so it never couples distinct determinants or blocks.  Its
    "streaming" label only means that no explicit AB structural map is stored.

    All pointer arguments are caller-owned, C-contiguous NumPy arrays.  The
    comments beside the ``argtypes`` entries document their required order and
    meaning; ``ctypes`` itself cannot validate the pointed-to dtype or shape.

    Returns
    -------
    ctypes.CDLL
        Cached handle to ``libpbc_kfci_hdiag``.
    """
    global libpbckfci_hdiag
    if libpbckfci_hdiag is None:
        libpbckfci_hdiag = load_library("libpbc_kfci_hdiag")
        # FCIhdiag_k arguments, in C-signature order.
        libpbckfci_hdiag.FCIhdiag_k.argtypes = [
            ctypes.c_void_p,  # hdiag: output complex128[sector_size]
            ctypes.c_void_p,  # h1e: complex128[nkpts, ncas, ncas]
            ctypes.c_void_p,  # eri: flattened complex128 k-point ERIs
            ctypes.c_int,     # nkpts: number of momentum points
            ctypes.c_int,     # ncas: active orbitals per k-point
            ctypes.c_int,     # nblocks: number of packed CI blocks
            ctypes.c_void_p,  # blocks: int32[nblocks, 6]
            ctypes.c_void_p,  # linka: int32 alpha link table
            ctypes.c_int,     # nlinka: alpha links per string
            ctypes.c_void_p,  # linkb: int32 beta link table
            ctypes.c_int,     # nlinkb: beta links per string
            ctypes.c_void_p,  # stra_ids: alpha string IDs grouped by k
            ctypes.c_void_p,  # stra_offsets: offsets into stra_ids
            ctypes.c_void_p,  # strb_ids: beta string IDs grouped by k
            ctypes.c_void_p,  # strb_offsets: offsets into strb_ids
            ctypes.c_int,     # dk_zero: label of zero momentum transfer
            ctypes.c_void_p,  # ab_group_tab: [dst_offset, begin, end]
            ctypes.c_void_p,  # ab_group_offsets: groups by source block
            ctypes.c_void_p,  # ab_src_addr: AB source determinant addresses
            ctypes.c_void_p,  # ab_dst_addr: AB target determinant addresses
            ctypes.c_void_p,  # ab_sign: AB products of fermionic signs
            ctypes.c_void_p,  # ab_eri_idx_ab: ERI indices in AB ordering
            ctypes.c_void_p,  # ab_eri_idx_ba: ERI indices in BA ordering
            ctypes.c_void_p,  # aa_group_tab: [dst, dst_na, begin, end]
            ctypes.c_void_p,  # aa_group_offsets: groups by source block
            ctypes.c_void_p,  # aa_src_addr: alpha source-row addresses
            ctypes.c_void_p,  # aa_dst_addr: alpha target-row addresses
            ctypes.c_void_p,  # aa_sign: alpha-alpha fermionic signs
            ctypes.c_void_p,  # aa_eri_idx: alpha-alpha ERI indices
            ctypes.c_void_p,  # bb_group_tab: [dst, dst_nb, begin, end]
            ctypes.c_void_p,  # bb_group_offsets: groups by source block
            ctypes.c_void_p,  # bb_src_addr: beta source-column addresses
            ctypes.c_void_p,  # bb_dst_addr: beta target-column addresses
            ctypes.c_void_p,  # bb_sign: beta-beta fermionic signs
            ctypes.c_void_p,  # bb_eri_idx: beta-beta ERI indices
        ]
        libpbckfci_hdiag.FCIhdiag_k.restype = None

        # FCIhdiag_k_stream_ab arguments.  It needs only raw link/layout data
        # because it constructs the diagonal AB contribution on the fly and
        # accumulates it into the hdiag initialized by FCIhdiag_k.
        libpbckfci_hdiag.FCIhdiag_k_stream_ab.argtypes = [
            ctypes.c_void_p,  # hdiag: accumulated complex128 diagonal
            ctypes.c_void_p,  # eri: flattened complex128 k-point ERIs
            ctypes.c_int,     # nkpts: number of momentum points
            ctypes.c_int,     # ncas: active orbitals per k-point
            ctypes.c_int,     # nblocks: number of packed CI blocks
            ctypes.c_void_p,  # blocks: int32[nblocks, 6]
            ctypes.c_void_p,  # linka: int32 alpha link table
            ctypes.c_int,     # nlinka: alpha links per string
            ctypes.c_void_p,  # linkb: int32 beta link table
            ctypes.c_int,     # nlinkb: beta links per string
            ctypes.c_void_p,  # stra_ids: alpha string IDs grouped by k
            ctypes.c_void_p,  # stra_offsets: offsets into stra_ids
            ctypes.c_void_p,  # strb_ids: beta string IDs grouped by k
            ctypes.c_void_p,  # strb_offsets: offsets into strb_ids
            ctypes.c_int,     # dk_zero: label of zero momentum transfer
        ]
        libpbckfci_hdiag.FCIhdiag_k_stream_ab.restype = None
    return libpbckfci_hdiag


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
        explicit_ab : bool or 'auto'
            If True, ensures that the contract_map is built with explicit
            alpha-beta pair tables. If 'auto', it will be determined based on
            the availability of memory.
        log_obj : object or None
            Logger object for tracking the progress of the function.
        kmom : KPointMomentum or None
            Precomputed momentum-arithmetic tables. If None, they are built
            from ``nkpts``.

    '''
    log = logger.new_logger(
        log_obj, getattr(log_obj, "verbose", logger.QUIET))
    t0 = (logger.process_clock(), logger.perf_counter())
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
        log.timer_debug1("k-FCI build contract map", *t0)
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
        log.timer_debug1(
            "k-FCI rebuild contract map with pair tables", *t0)
        return contract_map

    if explicit_ab is True and not getattr(contract_map, "explicit_ab", True):
        contract_map = make_kfci_contract_map(
            norb, nelec, nkpts, target_k,
            link_index=contract_map.link_index,
            build_pair_tables=need_pair_tables,
            explicit_ab=True,
            kmom=kmom,
        )
        log.timer_debug1(
            "k-FCI rebuild contract map with explicit AB", *t0)
        return contract_map
    log.timer_debug1("k-FCI validate contract map", *t0)
    return contract_map


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
        kmom : KPointMomentum or None
            Precomputed momentum-arithmetic tables. If None, they are built
            from ``nkpts``.

    returns:
        ndet_k : int
            Number of determinants in the target momentum sector.
    '''
    link_indexa, link_indexb = _unpack(norb, nelec, link_index, nkpts,
                                       kmom=kmom)
    blocks = gen_k_sector_linkstr_info(link_indexa, link_indexb, nkpts,
                                       target_k, kmom=kmom)
    if blocks.size == 0:
        return 0
    return int(blocks[:, 5].sum())

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


def contract_1e_k_py(h1e, fcivec, norb, nelec, nkpts, kindx,
                     link_index=None, contract_map=None, log_obj=None,
                     kmom=None):
    """Python reference implementation of :func:`contract_1e_k`.

    This routine has the same call signature and output convention as
    ``contract_1e_k``.  See that function for the argument and return-value
    documentation.
    """
    nkpts = int(nkpts)
    ncas = int(norb) // nkpts
    assert ncas * nkpts == int(norb)

    log = logger.new_logger(
        log_obj, getattr(log_obj, "verbose", logger.QUIET))
    t0 = (logger.process_clock(), logger.perf_counter())
    contract_map = _as_contract_map(
        norb, nelec, nkpts, kindx, link_index=link_index,
        contract_map=contract_map, log_obj=log_obj, kmom=kmom)
    assert fcivec.size == contract_map.sector_size
    t0 = log.timer_debug1("k-FCI contract_1e_py map setup", *t0)

    link_indexa, link_indexb = contract_map.link_index
    dtype = np.complex128

    # Sanity checks
    assert link_indexa.ndim == link_indexb.ndim == 3
    assert link_indexa.shape[2] == link_indexb.shape[2] == 8
    assert h1e.ndim == 3
    assert h1e.shape == (nkpts, ncas, ncas)

    blocks = contract_map.blocks
    stra_ids = contract_map.stra_ids
    stra_offsets = contract_map.stra_offsets
    strb_ids = contract_map.strb_ids
    strb_offsets = contract_map.strb_offsets
    tota_2k = contract_map.str2tot_a
    totb_2k = contract_map.str2tot_b
    dk_zero = int(contract_map.kmom.zero)

    # Making sure fcivec is in the right dtype and C-contiguous.
    h1e = np.asarray(h1e, dtype=dtype, order="C")
    fcivec = np.asarray(fcivec, dtype=dtype, order="C")
    sigma_ci = np.zeros(fcivec.shape, dtype=dtype, order="C")
    t0 = log.timer_debug1("k-FCI contract_1e_py array setup", *t0)

    # link columns: [cre, des, target_address, parity, k0, k_cre, k_des, dK]
    CRE = 0
    DES = 1
    TARGET = 2
    SIGN = 3
    K_CRE = 5
    K_DES = 6
    DK = 7

    for ka, kb, na, nb, offset, size in blocks:
        ka, kb, na, nb, offset, size = map(
            int, (ka, kb, na, nb, offset, size))
        Cblk = fcivec[offset:offset + size].reshape(na, nb)
        Sblk = sigma_ci[offset:offset + size].reshape(na, nb)

        alpha_ids = stra_ids[stra_offsets[ka]:stra_offsets[ka + 1]]
        beta_ids = strb_ids[strb_offsets[kb]:strb_offsets[kb + 1]]

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
                # which means only the zero-transfer dK label contributes.
                if (k_cre != k_des) or (dK != dk_zero):
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
                # which means only the zero-transfer dK label contributes.
                if (k_cre != k_des) or (dK != dk_zero):
                    continue

                hpq = h1e[k_cre, p % ncas, q % ncas]

                # Skip excitations that leave this momentum sector.
                ib1_local = totb_2k[kb, bstr1]
                if ib1_local < 0:
                    continue

                Sblk[:, ib1_local] += sign * hpq * Cblk[:, ib0_local]

    log.timer_debug1("k-FCI contract_1e_py Python kernel", *t0)
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
        log_obj : object or None
            Logger object for tracking the progress of the function.
        kmom : KPointMomentum or None
            Precomputed momentum-arithmetic tables. If None, they are built
            from ``nkpts``.

    returns:
        sigma_ci : ndarray, shape (sector_size,)
            Result of the Hamiltonian-vector product in the target momentum
            sector.
    '''
    nkpts = int(nkpts)
    ncas = int(norb) // nkpts
    assert ncas * nkpts == int(norb)

    log = logger.new_logger(
        log_obj, getattr(log_obj, "verbose", logger.QUIET))
    t0 = (logger.process_clock(), logger.perf_counter())
    contract_map = _as_contract_map(
        norb, nelec, nkpts, kindx, link_index=link_index,
        contract_map=contract_map, log_obj=log_obj, kmom=kmom)
    assert fcivec.size == contract_map.sector_size
    t0 = log.timer_debug1("k-FCI contract_1e map setup", *t0)

    h1e = np.asarray(h1e, dtype=np.complex128, order="C")
    fcivec = np.asarray(fcivec, dtype=np.complex128, order="C")
    sigma_ci = np.zeros(fcivec.shape, dtype=np.complex128, order="C")
    t0 = log.timer_debug1("k-FCI contract_1e array setup", *t0)

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
    log.timer_debug1("k-FCI contract_1e C kernel", *t0)
    return sigma_ci


def contract_2e_k_py(eri, fcivec, norb, nelec, nkpts, target_k,
                     link_index=None, contract_map=None, log_obj=None,
                     kmom=None):
    """Python reference implementation of :func:`contract_2e_k`.

    This routine has the same call signature and output convention as
    ``contract_2e_k``.  See that function for the argument and return-value
    documentation.
    """
    nkpts = int(nkpts)
    ncas = int(norb) // nkpts
    assert ncas * nkpts == int(norb)

    log = logger.new_logger(
        log_obj, getattr(log_obj, "verbose", logger.QUIET))
    t0 = (logger.process_clock(), logger.perf_counter())
    # The readable Python loops consume the pair-table representation.  Ask
    # _as_contract_map to build it only when the supplied map does not already
    # contain it.
    contract_map = _as_contract_map(
        norb, nelec, nkpts, target_k, link_index=link_index,
        contract_map=contract_map, need_pair_tables=True, explicit_ab=True,
        log_obj=log_obj, kmom=kmom)
    assert fcivec.size == contract_map.sector_size
    t0 = log.timer_debug1("k-FCI contract_2e_py map setup", *t0)

    eri = np.asarray(eri, dtype=np.complex128, order="C")
    fcivec = np.asarray(fcivec, dtype=np.complex128, order="C")
    sigma_ci = np.zeros(fcivec.shape, dtype=np.complex128, order="C")
    assert eri.shape == (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)
    t0 = log.timer_debug1("k-FCI contract_2e_py array setup", *t0)

    # Re-expose the flattened pair tables as the nested views expected by the
    # reference contraction helpers.  No pair entries are copied.
    ab_pairs = [[None for _ in range(nkpts)] for _ in range(nkpts)]
    for ka in range(nkpts):
        for kb in range(nkpts):
            key = ka * nkpts + kb
            i0 = int(contract_map.ab_offsets[key])
            i1 = int(contract_map.ab_offsets[key + 1])
            ab_pairs[ka][kb] = contract_map.ab_tab[i0:i1]
    aa_pairs = [
        contract_map.aa_tab[
            int(contract_map.aa_offsets[k]):
            int(contract_map.aa_offsets[k + 1])]
        for k in range(nkpts)
    ]
    bb_pairs = [
        contract_map.bb_tab[
            int(contract_map.bb_offsets[k]):
            int(contract_map.bb_offsets[k + 1])]
        for k in range(nkpts)
    ]

    blocks = contract_map.blocks

    # Rearrange the CI vectors into alpha/beta momentum blocks.
    ci0_blocks = _get_ci_sectors(fcivec, blocks, nkpts)
    ci1_blocks = _get_ci_sectors(sigma_ci, blocks, nkpts)

    kmom = contract_map.kmom
    for ka in range(nkpts):
        kb = kcistrings._ksub(kmom, contract_map.target_k, ka)

        if ci0_blocks[ka][kb] is None:
            continue

        kfci_contract_map.contract_ab_pairs(
            eri, ci0_blocks[ka][kb], ci1_blocks, ab_pairs, ka, kb)

        kfci_contract_map.contract_aa_pairs(
            eri, ci0_blocks, ci1_blocks, aa_pairs, ka, kb)

        kfci_contract_map.contract_bb_pairs(
            eri, ci0_blocks, ci1_blocks, bb_pairs, ka, kb)

    log.timer_debug1("k-FCI contract_2e_py Python kernel", *t0)
    return sigma_ci


def contract_2e_k(eri, fcivec, norb, nelec, nkpts, target_k,
                  link_index=None, contract_map=None, log_obj=None,
                  kmom=None):
    '''
    C implementation using structural k-sector contraction maps.
    The same-spin contractions are applied with zgemm and the alpha-beta terms
    are packed into sparse source/destination block groups.
    For now I am using OpenMP threads follow lib.num_threads(), however in future
    need to benchmark and come up with a better threading strategy.

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
        log_obj : object or None
            Logger object for tracking the progress of the function.
        kmom : KPointMomentum or None
            Precomputed momentum-arithmetic tables. If None, they are built
            from ``nkpts``.

    returns:
        sigma_ci : ndarray, shape (sector_size,)
            Result of the Hamiltonian-vector product in the target momentum
            sector.
    '''
    nkpts = int(nkpts)
    ncas = int(norb) // nkpts
    assert ncas * nkpts == int(norb)

    log = logger.new_logger(
        log_obj, getattr(log_obj, "verbose", logger.QUIET))
    t0 = (logger.process_clock(), logger.perf_counter())
    contract_map = _as_contract_map(
        norb, nelec, nkpts, target_k, link_index=link_index,
        contract_map=contract_map, log_obj=log_obj, kmom=kmom)
    assert fcivec.size == contract_map.sector_size
    t0 = log.timer_debug1("k-FCI contract_2e map setup", *t0)

    eri = np.asarray(eri, dtype=np.complex128, order="C")
    fcivec = np.asarray(fcivec, dtype=np.complex128, order="C")
    sigma_ci = np.zeros(fcivec.shape, dtype=np.complex128, order="C")
    t0 = log.timer_debug1("k-FCI contract_2e array setup", *t0)

    assert eri.shape == (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)

    libpbcfci = _load_k_contract_lib()
    with lib.with_omp_threads(lib.num_threads()):
        # This mapped C call first clears sigma_ci, then contracts every spin
        # channel represented by the supplied structural maps:
        #
        # AB: one alpha and one beta excitation.  Each entry combines the AB
        #     and BA integral orderings and updates individual determinant
        #     addresses in the destination block.
        # AA: two alpha excitations.  The beta string is a spectator, so the
        #     mapped alpha-row transformation is applied with zgemm.
        # BB: two beta excitations.  The alpha string is a spectator, so the
        #     mapped beta-column transformation is applied with zgemm.
        libpbcfci.FCIcontract_2e_k(
            eri.ctypes.data_as(ctypes.c_void_p),
            fcivec.ctypes.data_as(ctypes.c_void_p),
            sigma_ci.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nkpts),
            ctypes.c_int(ncas),
            ctypes.c_int(contract_map.blocks.shape[0]),
            contract_map.blocks.ctypes.data_as(ctypes.c_void_p),

            # Opposite-spin alpha-beta contraction map.
            contract_map.ab_group_tab.ctypes.data_as(ctypes.c_void_p),
            contract_map.ab_group_offsets.ctypes.data_as(ctypes.c_void_p),
            contract_map.ab_src_addr.ctypes.data_as(ctypes.c_void_p),
            contract_map.ab_dst_addr.ctypes.data_as(ctypes.c_void_p),
            contract_map.ab_sign.ctypes.data_as(ctypes.c_void_p),
            contract_map.ab_eri_idx_ab.ctypes.data_as(ctypes.c_void_p),
            contract_map.ab_eri_idx_ba.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(contract_map.ab_src_addr.size),

            # Same-spin alpha-alpha contraction map.
            contract_map.aa_group_tab.ctypes.data_as(ctypes.c_void_p),
            contract_map.aa_group_offsets.ctypes.data_as(ctypes.c_void_p),
            contract_map.aa_src_addr.ctypes.data_as(ctypes.c_void_p),
            contract_map.aa_dst_addr.ctypes.data_as(ctypes.c_void_p),
            contract_map.aa_sign.ctypes.data_as(ctypes.c_void_p),
            contract_map.aa_eri_idx.ctypes.data_as(ctypes.c_void_p),

            # Same-spin beta-beta contraction map.
            contract_map.bb_group_tab.ctypes.data_as(ctypes.c_void_p),
            contract_map.bb_group_offsets.ctypes.data_as(ctypes.c_void_p),
            contract_map.bb_src_addr.ctypes.data_as(ctypes.c_void_p),
            contract_map.bb_dst_addr.ctypes.data_as(ctypes.c_void_p),
            contract_map.bb_sign.ctypes.data_as(ctypes.c_void_p),
            contract_map.bb_eri_idx.ctypes.data_as(ctypes.c_void_p),
        )
        if not getattr(contract_map, "explicit_ab", True):
            # This second C call contracts only the AB channel.  The mapped
            # call above has already added AA and BB; because its AB arrays
            # were empty, the streamed kernel now generates all AB pairs from
            # the raw links and accumulates them into sigma_ci.  These terms
            # may stay in one (ka, kb) block or connect two such blocks;
            # opposite alpha/beta transfers preserve total momentum.
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
    log.timer_debug1("k-FCI contract_2e C kernel", *t0)
    return sigma_ci


def contract_ham_k(h1e, eri, fcivec, norb, nelec, nkpts, target_k=0,
                   link_index=None, contract_map=None, log_obj=None,
                   kmom=None):
    '''
    Contract the k-FCI Hamiltonian with a CI vector.
    Currently, I am keeping the one-electron and two-electron
    Hamiltonian contractions separate. I am not absorbing h1e into the
    two-electron Hamiltonian here.

    args:
        h1e : ndarray, shape (nkpts, norb_k, norb_k)
            One-electron Hamiltonian in k-space.
        eri : ndarray
            Shape ``(nkpts, nkpts, nkpts, norb_k, norb_k, norb_k,
            norb_k)``.
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
        contract_map : KFCIContractMap or None
            Precomputed contraction map. If None, a new one will be created.
        log_obj : object or None
            Logger object for tracking the progress of the function.
        kmom : KPointMomentum or None
            Precomputed momentum-arithmetic tables. If None, they are built
            from ``nkpts``.

    returns:
        sigma_ci : ndarray, shape (sector_size,)
            Result of applying H to fcivec.
    '''
    log = logger.new_logger(
        log_obj, getattr(log_obj, "verbose", logger.QUIET))
    t0 = (logger.process_clock(), logger.perf_counter())
    dtype = np.result_type(h1e, eri, fcivec)
    fcivec = np.asarray(fcivec, dtype=dtype, order="C")
    contract_map = _as_contract_map(
        norb, nelec, nkpts, target_k, link_index=link_index,
        contract_map=contract_map, log_obj=log_obj, kmom=kmom)
    link_index = contract_map.link_index
    t0 = log.timer_debug1("k-FCI contract_ham setup", *t0)

    sigma_ci = contract_1e_k(h1e, fcivec, norb, nelec, nkpts, target_k,
                             link_index=link_index,
                             contract_map=contract_map, log_obj=log_obj,
                             kmom=kmom)
    t0 = log.timer_debug1("k-FCI contract_ham 1e", *t0)
    sigma_ci += contract_2e_k(eri, fcivec, norb, nelec, nkpts, target_k,
                              link_index=link_index,
                              contract_map=contract_map, log_obj=log_obj,
                              kmom=kmom)
    log.timer_debug1("k-FCI contract_ham 2e", *t0)
    return sigma_ci


def _add_same_spin_hdiag_py(hdiag, eri_flat, blocks, group_tab, group_offsets,
                            src_addr, dst_addr, sign, eri_idx, nkpts, spin):
    """Accumulate diagonal AA or BB terms from a structural contraction map.

    A same-spin structural entry describes a transformation of either an
    alpha-string row (``spin == "a"``) or a beta-string column
    (``spin == "b"``).  A Hamiltonian-diagonal contribution must satisfy both
    of the following conditions:

    * its source and destination momentum blocks have the same packed offset;
    * its source and destination same-spin addresses are equal.

    Entries that do not meet both conditions are off-diagonal and are skipped.
    For an AA entry, the beta string is unchanged, so the selected alpha-row
    value is added across every beta string in the block.  For a BB entry, the
    alpha string is unchanged and the beta-column value is added across every
    alpha string.

    Parameters
    ----------
    hdiag : ndarray, shape (sector_size,)
        Packed Hamiltonian diagonal.  It is updated in place.
    eri_flat : ndarray
        C-order flattened two-electron integral array.  ``eri_idx`` contains
        indices into this array.
    blocks : ndarray, shape (nblocks, 6)
        Packed CI block records ``[ka, kb, na, nb, offset, size]``.
    group_tab : ndarray, shape (ngroups, 4)
        Same-spin group records
        ``[destination_offset, destination_spin_size, entry0, entry1]``.
    group_offsets : ndarray, shape (nkpts * nkpts + 1,)
        Ranges of ``group_tab`` belonging to each flattened source block key
        ``ka * nkpts + kb``.
    src_addr, dst_addr : ndarray, shape (nentries,)
        Source and destination row indices for AA, or column indices for BB,
        local to their packed momentum blocks.
    sign : ndarray, shape (nentries,)
        Product of the two fermionic excitation signs for each entry.
    eri_idx : ndarray, shape (nentries,)
        Flattened ERI index associated with each structural entry.
    nkpts : int
        Number of momentum points; used to flatten ``(ka, kb)`` block keys.
    spin : {"a", "b"}
        Selects the alpha-alpha or beta-beta contraction map.

    Returns
    -------
    None
        Contributions are accumulated directly into ``hdiag``.
    """
    block_offsets = -np.ones(nkpts * nkpts, dtype=np.int64)
    block_na = np.zeros(nkpts * nkpts, dtype=np.int64)
    block_nb = np.zeros(nkpts * nkpts, dtype=np.int64)

    for ka, kb, na, nb, offset, _ in blocks:
        key = int(ka) * nkpts + int(kb)
        block_offsets[key] = int(offset)
        block_na[key] = int(na)
        block_nb[key] = int(nb)

    for src_key in range(nkpts * nkpts):
        src_offset = int(block_offsets[src_key])
        if src_offset < 0:
            continue

        nb = int(block_nb[src_key])
        na = int(block_na[src_key])
        g0 = int(group_offsets[src_key])
        g1 = int(group_offsets[src_key + 1])

        for ig in range(g0, g1):
            dst_offset, _, entry0, entry1 = map(int, group_tab[ig])
            if dst_offset != src_offset:
                continue

            for itab in range(entry0, entry1):
                if int(dst_addr[itab]) != int(src_addr[itab]):
                    continue

                val = sign[itab] * eri_flat[int(eri_idx[itab])]
                if spin == "a":
                    ia = int(src_addr[itab])
                    hdiag[src_offset + ia *
                          nb:src_offset + (ia + 1) * nb] += val
                else:
                    ib = int(src_addr[itab])
                    hdiag[src_offset + ib:src_offset + na * nb:nb] += val


def make_hdiag_py(h1e, eri, norb, nelec, nkpts, target_k=0, link_index=None,
                  contract_map=None, log_obj=None, kmom=None):
    """Python reference implementation of :func:`make_hdiag`.

    See ``make_hdiag`` for the argument and return-value documentation.  This
    reference path requests an explicit AB structural map, then retains only
    entries with identical source and destination determinants.  It evaluates
    the one-electron, AB, AA, and BB diagonal contributions separately for
    readability and testing.  Unlike the compiled path, its result dtype is
    ``numpy.result_type(h1e, eri)`` rather than unconditionally ``complex128``.
    """
    log = logger.new_logger(
        log_obj, getattr(log_obj, "verbose", logger.QUIET))
    t0 = (logger.process_clock(), logger.perf_counter())
    contract_map = _as_contract_map(
        norb, nelec, nkpts, target_k, link_index=link_index,
        contract_map=contract_map, explicit_ab=True, log_obj=log_obj,
        kmom=kmom)
    link_index = contract_map.link_index
    ndet = contract_map.sector_size
    dtype = np.result_type(h1e, eri)
    hdiag = np.zeros(ndet, dtype=dtype)
    t0 = log.timer_debug1("k-FCI make_hdiag_py map setup", *t0)

    h1e = np.asarray(h1e, dtype=dtype, order="C")
    eri = np.asarray(eri, dtype=dtype, order="C")
    eri_flat = eri.reshape(-1)
    ncas = int(norb) // int(nkpts)
    dk_zero = int(contract_map.kmom.zero)

    link_indexa, link_indexb = link_index
    blocks = np.asarray(contract_map.blocks, dtype=np.int32, order="C")
    t0 = log.timer_debug1("k-FCI make_hdiag_py array setup", *t0)

    CRE = 0
    DES = 1
    TARGET = 2
    SIGN = 3
    K_CRE = 5
    K_DES = 6
    DK = 7

    for ka, kb, na, nb, offset, _ in blocks:
        ka = int(ka)
        kb = int(kb)
        na = int(na)
        nb = int(nb)
        offset = int(offset)

        hblk = hdiag[offset:offset + na * nb].reshape(na, nb)
        alpha_ids = contract_map.stra_ids[
            contract_map.stra_offsets[ka]:contract_map.stra_offsets[ka + 1]]
        beta_ids = contract_map.strb_ids[
            contract_map.strb_offsets[kb]:contract_map.strb_offsets[kb + 1]]

        for ia, astr0 in enumerate(alpha_ids):
            val = 0
            for link in link_indexa[int(astr0)]:
                if int(link[SIGN]) == 0:
                    break
                if int(link[TARGET]) != int(astr0):
                    continue
                k_cre = int(link[K_CRE]) % nkpts
                if (k_cre != int(link[K_DES]) % nkpts or
                        int(link[DK]) % nkpts != dk_zero):
                    continue
                val += link[SIGN] * h1e[k_cre,
                                        int(link[CRE]) % ncas,
                                        int(link[DES]) % ncas]
            hblk[ia, :] += val

        for ib, bstr0 in enumerate(beta_ids):
            val = 0
            for link in link_indexb[int(bstr0)]:
                if int(link[SIGN]) == 0:
                    break
                if int(link[TARGET]) != int(bstr0):
                    continue
                k_cre = int(link[K_CRE]) % nkpts
                if (k_cre != int(link[K_DES]) % nkpts or
                        int(link[DK]) % nkpts != dk_zero):
                    continue
                val += link[SIGN] * h1e[k_cre,
                                        int(link[CRE]) % ncas,
                                        int(link[DES]) % ncas]
            hblk[:, ib] += val
    t0 = log.timer_debug1("k-FCI make_hdiag_py one-electron diag", *t0)

    block_offsets = {
        int(ka) * nkpts + int(kb): int(offset)
        for ka, kb, _, _, offset, _ in blocks
    }
    for src_key in range(nkpts * nkpts):
        g0 = int(contract_map.ab_group_offsets[src_key])
        g1 = int(contract_map.ab_group_offsets[src_key + 1])
        for ig in range(g0, g1):
            dst_offset, entry0, entry1 = map(
                int, contract_map.ab_group_tab[ig])
            src_offset = block_offsets.get(src_key, -1)
            if src_offset < 0 or dst_offset != src_offset:
                continue

            for itab in range(entry0, entry1):
                if (int(contract_map.ab_dst_addr[itab]) !=
                        int(contract_map.ab_src_addr[itab])):
                    continue
                addr = src_offset + int(contract_map.ab_src_addr[itab])
                hdiag[addr] += contract_map.ab_sign[itab] * (
                    eri_flat[int(contract_map.ab_eri_idx_ab[itab])] +
                    eri_flat[int(contract_map.ab_eri_idx_ba[itab])])
    t0 = log.timer_debug1("k-FCI make_hdiag_py alpha-beta diag", *t0)

    _add_same_spin_hdiag_py(
        hdiag, eri_flat, blocks, contract_map.aa_group_tab,
        contract_map.aa_group_offsets, contract_map.aa_src_addr,
        contract_map.aa_dst_addr, contract_map.aa_sign,
        contract_map.aa_eri_idx, nkpts, "a")
    t0 = log.timer_debug1("k-FCI make_hdiag_py alpha-alpha diag", *t0)
    _add_same_spin_hdiag_py(
        hdiag, eri_flat, blocks, contract_map.bb_group_tab,
        contract_map.bb_group_offsets, contract_map.bb_src_addr,
        contract_map.bb_dst_addr, contract_map.bb_sign,
        contract_map.bb_eri_idx, nkpts, "b")
    log.timer_debug1("k-FCI make_hdiag_py beta-beta diag", *t0)

    return hdiag


def make_hdiag(h1e, eri, norb, nelec, nkpts, target_k=0, link_index=None,
               contract_map=None, log_obj=None, kmom=None):
    """Build the k-FCI Hamiltonian diagonal in one total-momentum sector.

    The returned element at packed address ``I`` is ``<I|H|I>``.  Determinants
    are ordered by the six-column ``contract_map.blocks`` table, with each
    block holding alpha strings of momentum ``ka`` and beta strings of momentum
    ``kb``.  The compiled kernel initializes the output and adds four kinds of
    diagonal contribution:

    * one-electron self-links for the alpha and beta strings;
    * opposite-spin alpha-beta (AB) two-electron entries;
    * same-spin alpha-alpha (AA) entries, broadcast over beta spectators;
    * same-spin beta-beta (BB) entries, broadcast over alpha spectators.

    When ``contract_map.explicit_ab`` is false, the first C kernel adds the
    one-electron and mapped AA/BB terms.  A second kernel then generates the AB
    diagonal directly from link tables and accumulates it into the result.
    Streaming avoids storing the large explicit AB map; because only diagonal
    elements are requested, it never transfers amplitude between determinants
    or packed momentum blocks.

    Parameters
    ----------
    h1e : ndarray, shape (nkpts, ncas, ncas)
        One-electron integrals in k-space, where
        ``ncas = norb // nkpts``.  The integral is diagonal in k-point.
    eri : ndarray, shape (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)
        Two-electron integrals in the k-point storage convention used by
        :func:`contract_2e_k`.
    norb : int
        Total number of active orbitals across all k-points.  It must be
        divisible by ``nkpts``.
    nelec : int or tuple of two ints
        Total electron count or ``(N_alpha, N_beta)``.
    nkpts : int
        Number of k-points or momentum labels.
    target_k : int, optional
        Total-momentum sector whose packed determinant diagonal is built.
    link_index : tuple of two ndarrays or KFCIContractMap or None, optional
        Alpha and beta k-aware link tables.  A ``KFCIContractMap`` may also be
        supplied here for compatibility; otherwise missing tables are built.
    contract_map : KFCIContractMap or None, optional
        Precomputed packed layout and two-electron structural maps.  Supplying
        it avoids rebuilding those arrays.
    log_obj : object or None, optional
        Object used to configure PySCF timing and verbosity messages.
    kmom : KPointMomentum or None, optional
        Precomputed momentum arithmetic.  It is built from ``nkpts`` when no
        contract map or explicit momentum object is supplied.

    Returns
    -------
    hdiag : ndarray, shape (sector_size,), dtype complex128
        Hamiltonian diagonal in the same packed block order as the sector CI
        vectors consumed by :func:`contract_1e_k` and :func:`contract_2e_k`.
    """
    log = logger.new_logger(
        log_obj, getattr(log_obj, "verbose", logger.QUIET))
    t0 = (logger.process_clock(), logger.perf_counter())
    nkpts = int(nkpts)
    ncas = int(norb) // nkpts
    assert ncas * nkpts == int(norb)

    contract_map = _as_contract_map(
        norb, nelec, nkpts, target_k, link_index=link_index,
        contract_map=contract_map, log_obj=log_obj, kmom=kmom)
    ndet = contract_map.sector_size
    t0 = log.timer_debug1("k-FCI make_hdiag map setup", *t0)

    h1e = np.asarray(h1e, dtype=np.complex128, order="C")
    eri = np.asarray(eri, dtype=np.complex128, order="C")
    hdiag = np.empty(ndet, dtype=np.complex128, order="C")
    assert h1e.shape == (nkpts, ncas, ncas)
    assert eri.shape == (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)
    t0 = log.timer_debug1("k-FCI make_hdiag array setup", *t0)

    link_indexa, link_indexb = contract_map.link_index
    libpbcfci = _load_k_hdiag_lib()
    t0 = log.timer_debug1("k-FCI make_hdiag library setup", *t0)
    with lib.with_omp_threads(lib.num_threads()):
        libpbcfci.FCIhdiag_k(
            hdiag.ctypes.data_as(ctypes.c_void_p),
            h1e.ctypes.data_as(ctypes.c_void_p),
            eri.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nkpts),
            ctypes.c_int(ncas),
            ctypes.c_int(contract_map.blocks.shape[0]),
            contract_map.blocks.ctypes.data_as(ctypes.c_void_p),
            link_indexa.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(link_indexa.shape[1]),
            link_indexb.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(link_indexb.shape[1]),
            contract_map.stra_ids.ctypes.data_as(ctypes.c_void_p),
            contract_map.stra_offsets.ctypes.data_as(ctypes.c_void_p),
            contract_map.strb_ids.ctypes.data_as(ctypes.c_void_p),
            contract_map.strb_offsets.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(contract_map.kmom.zero),
            contract_map.ab_group_tab.ctypes.data_as(ctypes.c_void_p),
            contract_map.ab_group_offsets.ctypes.data_as(ctypes.c_void_p),
            contract_map.ab_src_addr.ctypes.data_as(ctypes.c_void_p),
            contract_map.ab_dst_addr.ctypes.data_as(ctypes.c_void_p),
            contract_map.ab_sign.ctypes.data_as(ctypes.c_void_p),
            contract_map.ab_eri_idx_ab.ctypes.data_as(ctypes.c_void_p),
            contract_map.ab_eri_idx_ba.ctypes.data_as(ctypes.c_void_p),
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
            # FCIhdiag_k initialized hdiag and added 1e plus mapped AA/BB
            # terms.  The streamed kernel adds the omitted AB diagonal terms
            # without constructing an explicit AB map.  A diagonal operation
            # never transfers amplitude between packed momentum blocks.
            libpbcfci.FCIhdiag_k_stream_ab(
                hdiag.ctypes.data_as(ctypes.c_void_p),
                eri.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nkpts),
                ctypes.c_int(ncas),
                ctypes.c_int(contract_map.blocks.shape[0]),
                contract_map.blocks.ctypes.data_as(ctypes.c_void_p),
                link_indexa.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(link_indexa.shape[1]),
                link_indexb.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(link_indexb.shape[1]),
                contract_map.stra_ids.ctypes.data_as(ctypes.c_void_p),
                contract_map.stra_offsets.ctypes.data_as(ctypes.c_void_p),
                contract_map.strb_ids.ctypes.data_as(ctypes.c_void_p),
                contract_map.strb_offsets.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(contract_map.kmom.zero),
            )
    log.timer_debug1("k-FCI make_hdiag C kernel", *t0)
    return hdiag


def get_init_guess_k(norb, nelec, nkpts, target_k, nroots, hdiag,
                     log_obj=None):
    '''
    Get initial guess vectors for k-FCI in a fixed total momentum sector.
    The guesses are determinant basis vectors corresponding to the lowest
    diagonal Hamiltonian elements.
    '''
    log = logger.new_logger(
        log_obj, getattr(log_obj, "verbose", logger.QUIET))
    t0 = (logger.process_clock(), logger.perf_counter())
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
    t0 = log.timer_debug1("k-FCI get_init_guess sort hdiag", *t0)

    ci0 = []
    for i in range(nroots):
        x = np.zeros(ndet, dtype=dtype)
        x[int(addr[i])] = 1.0
        ci0.append(x)
    log.timer_debug1("k-FCI get_init_guess vectors", *t0)
    return ci0


def make_hamiltonian_k(h1e, eri, norb, nelec, nkpts, target_k=0,
                       link_index=None, contract_map=None, log_obj=None,
                       kmom=None, contract_fn=None):
    '''
    Construct the explicit k-FCI Hamiltonian in a fixed total momentum sector.
    This routine is intended for small determinant spaces and for debugging.
    For large spaces, kernel_ms1 uses Davidson with contract_ham_k instead.

    ``kmom`` may provide precomputed :class:`KPointMomentum` arithmetic
    tables; they are constructed from ``nkpts`` when omitted.
    '''
    log = logger.new_logger(
        log_obj, getattr(log_obj, "verbose", logger.QUIET))
    t0 = (logger.process_clock(), logger.perf_counter())
    contract_map = _as_contract_map(
        norb, nelec, nkpts, target_k, link_index=link_index,
        contract_map=contract_map, log_obj=log_obj, kmom=kmom)
    link_index = contract_map.link_index
    ndet = contract_map.sector_size
    dtype = np.result_type(h1e, eri, np.complex128)
    hmat = np.empty((ndet, ndet), dtype=dtype, order="F")
    t0 = log.timer_debug1("k-FCI make_hamiltonian setup", *t0)

    if contract_fn is None:
        def apply_hamiltonian(ci):
            return contract_ham_k(
                h1e, eri, ci, norb, nelec, nkpts, target_k,
                link_index=link_index, contract_map=contract_map,
                log_obj=log_obj, kmom=kmom)
    else:
        def apply_hamiltonian(ci):
            return contract_fn(
                h1e, eri, ci, norb, nelec, nkpts=nkpts,
                target_k=target_k, link_index=link_index,
                contract_map=contract_map)

    for i in range(ndet):
        ci0 = np.zeros(ndet, dtype=dtype)
        ci0[i] = 1.0
        hmat[:, i] = apply_hamiltonian(ci0)

    log.timer_debug1("k-FCI make_hamiltonian columns", *t0)
    return hmat


def energy(h1e, eri, fcivec, norb, nelec, nkpts, target_k=0,
           link_index=None, contract_map=None, log_obj=None,
           kmom=None, contract_fn=None):
    '''
    Compute the k-FCI electronic energy for a CI vector.
    The one-electron and two-electron Hamiltonian contractions are evaluated
    separately; h1e is not absorbed into eri.

    ``kmom`` may provide precomputed :class:`KPointMomentum` arithmetic
    tables; they are constructed from ``nkpts`` when omitted.
    '''
    log = logger.new_logger(
        log_obj, getattr(log_obj, "verbose", logger.QUIET))
    t0 = (logger.process_clock(), logger.perf_counter())
    ci0 = np.asarray(fcivec)
    if contract_fn is None:
        sigma = contract_ham_k(
            h1e, eri, ci0, norb, nelec, nkpts, target_k,
            link_index=link_index, contract_map=contract_map,
            log_obj=log_obj, kmom=kmom)
    else:
        sigma = contract_fn(
            h1e, eri, ci0, norb, nelec, nkpts=nkpts,
            target_k=target_k, link_index=link_index,
            contract_map=contract_map)
    t0 = log.timer_debug1("k-FCI energy contract_ham", *t0)
    e = np.vdot(ci0, sigma)
    log.timer_debug1("k-FCI energy dot", *t0)
    return e


def make_rdm1s(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
               spin=None, kmom=None, kconserv=None):
    """Build spin-separated one-particle RDMs for a k-FCI vector.

    ``kmom`` optionally supplies precomputed momentum-arithmetic tables.

    Returns
    -------
    rdm1s : tuple of two ndarrays
        ``(dm1a, dm1b)``.  Each spin-resolved one-particle RDM has shape
        ``(norb, norb)`` and dtype ``complex128``.
    """
    return krdm_helper.make_rdm1s(
        fcivec, norb, nelec, nkpts, target_k=target_k,
        link_index=link_index, spin=spin, kmom=kmom, kconserv=kconserv)


def make_rdm1(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
              spin=None, kmom=None, kconserv=None):
    """Build the spin-summed one-particle RDM for a k-FCI vector.

    ``kmom`` optionally supplies precomputed momentum-arithmetic tables.

    Returns
    -------
    dm1 : ndarray, shape (norb, norb), dtype complex128
        Spin-summed one-particle RDM.
    """
    return krdm_helper.make_rdm1(
        fcivec, norb, nelec, nkpts, target_k=target_k,
        link_index=link_index, spin=spin, kmom=kmom, kconserv=kconserv)


def make_rdm12s(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
                reorder=True, spin=None, kmom=None, kconserv=None):
    """Build spin-separated one- and two-particle RDMs.

    ``kmom`` optionally supplies precomputed momentum-arithmetic tables.

    Returns
    -------
    rdm1s : tuple of two ndarrays
        ``(dm1a, dm1b)``.  Each array has shape ``(norb, norb)`` and dtype
        ``complex128``.
    rdm2s : tuple of three ndarrays
        ``(dm2aa, dm2ab, dm2bb)``.  Each spin-resolved two-particle RDM has
        shape ``(norb, norb, norb, norb)`` and dtype ``complex128``.
    """
    return krdm_helper.make_rdm12s(
        fcivec, norb, nelec, nkpts, target_k=target_k,
        link_index=link_index, reorder=reorder, spin=spin, kmom=kmom,
        kconserv=kconserv)

def make_rdm12(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
               reorder=True, spin=None, kmom=None, kconserv=None):
    """Build spin-summed one- and two-particle RDMs.

    ``kmom`` optionally supplies precomputed momentum-arithmetic tables.

    Returns
    -------
    dm1 : ndarray, shape (norb, norb), dtype complex128
        Spin-summed one-particle RDM.
    dm2 : ndarray, shape (norb, norb, norb, norb), dtype complex128
        Spin-summed two-particle RDM.
    """
    return krdm_helper.make_rdm12(
        fcivec, norb, nelec, nkpts, target_k=target_k,
        link_index=link_index, reorder=reorder, spin=spin, kmom=kmom,
        kconserv=kconserv)

def contract_ss(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
                spin=None, contract_map=None, kmom=None, kconserv=None):
    """Apply the spin-squared operator to a k-FCI vector.
    ``kmom`` optionally supplies precomputed momentum-arithmetic tables.
    """
    return krdm_helper.contract_ss(
        fcivec, norb, nelec, nkpts, target_k=target_k,
        link_index=link_index, spin=spin, contract_map=contract_map,
        kmom=kmom, kconserv=kconserv)

def spin_square(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
                spin=None, kmom=None, kconserv=None, **kwargs):
    """Evaluate spin-squared for a k-FCI vector.
    ``kmom`` optionally supplies precomputed momentum-arithmetic tables.
    """
    return krdm_helper.spin_square(
        fcivec, norb, nelec, nkpts, target_k=target_k,
        link_index=link_index, spin=spin, kmom=kmom, kconserv=kconserv,
        **kwargs)

def _get_spin_penalty(fcivec, norb, nelec, nkpts, target_k=0,
                      link_index=None, spin=None, ss_value=None,
                      ss_penalty=0.1, log_obj=None, contract_map=None,
                      kmom=None, kconserv=None):
    """Apply the configured spin-penalty operator in one momentum sector.
    ``kmom`` optionally supplies precomputed momentum-arithmetic tables.
    """
    log = logger.new_logger(
        log_obj, getattr(log_obj, "verbose", logger.QUIET))
    t0 = (logger.process_clock(), logger.perf_counter())
    nelec = _unpack_nelec(nelec, spin)
    sz = abs(nelec[0] - nelec[1]) * 0.5
    ss = sz * (sz + 1) if ss_value is None else ss_value
    fcivec = np.asarray(fcivec)

    if ss < sz * (sz + 1) + 0.1:
        # Shift all states except the lowest-spin state with (S^2 - ss).
        ci1 = contract_ss(
            fcivec, norb, nelec, nkpts, target_k=target_k,
            link_index=link_index, spin=spin, contract_map=contract_map,
            kmom=kmom, kconserv=kconserv).reshape(fcivec.shape)
        ci1 -= ss * fcivec
        t0 = log.timer_debug1("k-FCI spin penalty contract_ss", *t0)
    else:
        # Select a specified spin with (S^2 - ss)^2.
        residual = contract_ss(
            fcivec, norb, nelec, nkpts, target_k=target_k,
            link_index=link_index, spin=spin, contract_map=contract_map,
            kmom=kmom, kconserv=kconserv).reshape(fcivec.shape)
        residual -= ss * fcivec
        t0 = log.timer_debug1(
            "k-FCI spin penalty first contract_ss", *t0)
        ci1 = contract_ss(
            residual, norb, nelec, nkpts, target_k=target_k,
            link_index=link_index, spin=spin, contract_map=contract_map,
            kmom=kmom, kconserv=kconserv).reshape(fcivec.shape)
        ci1 -= ss * residual
        t0 = log.timer_debug1(
            "k-FCI spin penalty second contract_ss", *t0)

    ci1 *= ss_penalty
    log.timer_debug1("k-FCI spin penalty scale", *t0)
    return ci1


def _spin_square_diag_k(norb, nelec, nkpts, target_k=0, link_index=None,
                        contract_map=None, kmom=None):
    """Build the diagonal of spin-squared in a packed momentum sector.
    ``kmom`` optionally supplies precomputed momentum-arithmetic tables.
    """
    nelec = _unpack_nelec(nelec)
    contract_map = _as_contract_map(
        norb, nelec, nkpts, target_k, link_index=link_index,
        contract_map=contract_map, kmom=kmom)
    strs_a = np.asarray(
        cistring.make_strings(range(norb), nelec[0]), dtype=np.uint64)
    strs_b = np.asarray(
        cistring.make_strings(range(norb), nelec[1]), dtype=np.uint64)

    sz = 0.5 * (nelec[0] - nelec[1])
    diag0 = sz * sz + 0.5 * sum(nelec)
    hdiag = np.empty(contract_map.sector_size, dtype=np.float64)

    for block in contract_map.blocks:
        ka, kb, nstra, nstrb, offset, size = map(int, block)
        a0 = int(contract_map.stra_offsets[ka])
        a1 = int(contract_map.stra_offsets[ka + 1])
        b0 = int(contract_map.strb_offsets[kb])
        b1 = int(contract_map.strb_offsets[kb + 1])
        astrs = strs_a[contract_map.stra_ids[a0:a1]]
        bstrs = strs_b[contract_map.strb_ids[b0:b1]]
        assert astrs.size == nstra
        assert bstrs.size == nstrb

        hblk = hdiag[offset:offset + size].reshape(nstra, nstrb)
        for ia, astr in enumerate(astrs):
            # An alpha and beta determinant are uint64 bit strings with one
            # bit per spatial orbital.  Their bitwise intersection therefore
            # has one set bit for every doubly occupied orbital.  For a Slater
            # determinant, the diagonal spin-squared matrix element is
            #
            #   <D|S^2|D> = Sz^2 + (N_alpha + N_beta)/2 - N_common,
            #
            # where N_common is this intersection's population count.
            # ``bstrs`` is an array, so the lookup computes all beta-string
            # counts paired with the current alpha string at once.
            common = np.bitwise_and(astr, bstrs)
            hblk[ia] = diag0 - _popcount_uint64(common)

    return hdiag


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
    t0 = (logger.process_clock(), logger.perf_counter())
    if nroots is None:
        nroots = fci.nroots
    if davidson_only is None:
        davidson_only = fci.davidson_only
    if pspace_size is None:
        pspace_size = fci.pspace_size
    if max_memory is None:
        max_memory = fci.max_memory - lib.current_memory()[0]

    log = logger.new_logger(fci, verbose)
    nelec = _unpack_nelec(nelec, fci.spin)
    target_k = int(target_k) % nkpts
    kmom = fci.get_kmom(nkpts) if hasattr(fci, "get_kmom") else None
    link_index = _unpack(norb, nelec, link_index, nkpts, spin=fci.spin,
                         kmom=kmom)
    t0 = log.timer_debug1("k-FCI kernel setup options/link_index", *t0)
    contract_map = _as_contract_map(
        norb, nelec, nkpts, target_k, link_index=link_index,
        explicit_ab=False, log_obj=fci, kmom=kmom)
    link_index = contract_map.link_index
    kmom = contract_map.kmom
    t0 = log.timer_debug1("k-FCI kernel contract map", *t0)

    assert norb % nkpts == 0
    ncas = norb // nkpts
    assert h1e.shape == (nkpts, ncas, ncas)
    assert eri.shape == (nkpts, nkpts, nkpts, ncas, ncas, ncas, ncas)
    t0 = log.timer_debug1("k-FCI kernel shape checks", *t0)

    hdiag = fci.make_hdiag(h1e, eri, norb, nelec, nkpts, target_k,
                           link_index=link_index,
                           contract_map=contract_map).ravel()
    civec_size = hdiag.size
    t0 = log.timer_debug1("k-FCI kernel hdiag", *t0)

    if civec_size == 0:
        raise RuntimeError(
            f"No determinants in k-FCI sector target_k={target_k}.")

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
    t0 = log.timer_debug1("k-FCI kernel memory/direct branch decision", *t0)

    if do_direct:
        hmat = fci.make_hamiltonian(h1e, eri, norb, nelec, nkpts, target_k,
                                    link_index=link_index,
                                    contract_map=contract_map)
        t0 = log.timer_debug1("k-FCI kernel direct Hamiltonian", *t0)
        e, c = fci.eig(hmat)
        t0 = log.timer_debug1("k-FCI kernel direct eig", *t0)
        e = e[:nroots]
        if nroots == 1:
            c = c[:, 0]
            e = e[0]
        else:
            c = c[:, :nroots].T
        log.timer_debug1("k-FCI kernel direct postprocess", *t0)
        return e + ecore, c

    precond = fci.make_precond(hdiag)
    t0 = log.timer_debug1("k-FCI kernel preconditioner", *t0)

    cpu0 = [logger.process_clock(), logger.perf_counter()]

    def hop(c):
        hc = fci.contract_ham(h1e, eri, c, norb, nelec, nkpts, target_k,
                              link_index=link_index,
                              contract_map=contract_map)
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
    t0 = log.timer_debug1("k-FCI kernel initial guess setup", *t0)

    if tol is None:
        tol = fci.conv_tol
    if lindep is None:
        lindep = fci.lindep
    if max_cycle is None:
        max_cycle = fci.max_cycle
    if max_space is None:
        max_space = fci.max_space
    tol_residual = getattr(fci, "conv_tol_residual", None)
    t0 = log.timer_debug1("k-FCI kernel Davidson parameters", *t0)

    with lib.with_omp_threads(lib.num_threads()):
        e, c = fci.eig(hop, ci0, precond, tol=tol, lindep=lindep,
                       max_cycle=max_cycle, max_space=max_space,
                       nroots=nroots, max_memory=max_memory, verbose=log,
                       follow_state=True, tol_residual=tol_residual,
                       **kwargs)
    log.timer_debug1("k-FCI kernel Davidson eig", *t0)
    return e + ecore, c


class SpinPenaltyFCISolver(_PySCFSpinPenaltyFCISolver):
    """PySCF spin-penalty mixin specialized for a k-FCI solver.

    Initialization, bookkeeping, and ``undo_fix_spin`` are inherited from
    PySCF.  Only operations whose signatures or data layout differ for a
    packed momentum sector are overridden here.
    """

    def contract_2e(self, eri, fcivec, norb, nelec, nkpts=None,
                    target_k=None, link_index=None, contract_map=None):
        """Add the spin penalty to the base two-electron contraction."""
        if nkpts is None:
            nkpts = self.nkpts
        if target_k is None:
            target_k = self.target_k
        kmom = self.get_kmom(nkpts) if hasattr(self, "get_kmom") else None
        nelec = _unpack_nelec(nelec, self.spin)
        # PySCF's base_contract_2e deliberately skips its molecular
        # SpinPenaltyFCISolver.contract_2e in the dynamic mixin MRO and calls
        # the underlying k-FCI solver contraction.
        ci0 = self.base_contract_2e(
            eri, fcivec, norb, nelec, nkpts=nkpts, target_k=target_k,
            link_index=link_index, contract_map=contract_map)
        if contract_map is not None:
            link_index = contract_map.link_index
            kmom = contract_map.kmom
        ci1 = _get_spin_penalty(
            fcivec, norb, nelec, nkpts, target_k=target_k,
            link_index=link_index, spin=self.spin, ss_value=self.ss_value,
            ss_penalty=self.ss_penalty, log_obj=self,
            contract_map=contract_map, kmom=kmom)
        ci1 += ci0.reshape(fcivec.shape)
        return ci1

    def make_hdiag(self, h1e, eri, norb, nelec, nkpts=None, target_k=None,
                   link_index=None, compress=False, contract_map=None):
        """Add the spin-penalty diagonal to the Hamiltonian diagonal."""
        if nkpts is None:
            nkpts = self.nkpts
        if target_k is None:
            target_k = self.target_k
        kmom = self.get_kmom(nkpts) if hasattr(self, "get_kmom") else None
        nelec = _unpack_nelec(nelec, self.spin)
        link_index = _unpack(
            norb, nelec, link_index, nkpts, spin=self.spin, kmom=kmom)
        contract_map = _as_contract_map(
            norb, nelec, nkpts, target_k, link_index=link_index,
            contract_map=contract_map, log_obj=self, kmom=kmom)
        link_index = contract_map.link_index
        kmom = contract_map.kmom

        hdiag = super().make_hdiag(
            h1e, eri, norb, nelec, nkpts=nkpts, target_k=target_k,
            link_index=link_index, compress=compress,
            contract_map=contract_map)

        sz = abs(nelec[0] - nelec[1]) * 0.5
        ss = sz * (sz + 1) if self.ss_value is None else self.ss_value
        diag_ss = _spin_square_diag_k(
            norb, nelec, nkpts, target_k=target_k,
            link_index=link_index, contract_map=contract_map, kmom=kmom)
        if ss < sz * (sz + 1) + 0.1:
            hdiag = hdiag + self.ss_penalty * (diag_ss - ss)
        else:
            # This is a preconditioner diagonal; the contraction still uses
            # the complete squared spin-penalty operator.
            hdiag = hdiag + self.ss_penalty * (diag_ss - ss) ** 2
        return hdiag


def fix_spin(fciobj, shift=0.1, ss=None, **kwargs):
    """Return a k-FCI solver with a spin-penalty mixin."""
    if isinstance(fciobj, types.ModuleType):
        raise DeprecationWarning("fix_spin should be applied to an FCI object")

    ss_value = kwargs.get("ss_value", ss)
    if isinstance(fciobj, SpinPenaltyFCISolver):
        fciobj.ss_penalty = shift
        fciobj.ss_value = ss_value
        return fciobj

    spin_solver = SpinPenaltyFCISolver(fciobj, shift, ss_value)
    return lib.set_class(
        spin_solver, (SpinPenaltyFCISolver, fciobj.__class__))


def fix_spin_(fciobj, shift=0.1, ss=None, **kwargs):
    """Apply the k-FCI spin-penalty mixin in place."""
    spin_solver = fix_spin(fciobj, shift=shift, ss=ss, **kwargs)
    fciobj.__class__ = spin_solver.__class__
    fciobj.__dict__ = spin_solver.__dict__
    return fciobj


class FCISolver(direct_spin1.FCISolver):
    """FCI solver restricted to one total-momentum sector."""

    _keys = direct_spin1.FCISolver._keys | {
        "nkpts", "target_k", "kpts", "kmesh", "kconserv", "kmom"
    }

    def __init__(self, *args, **kwargs):
        nkpts = kwargs.pop("nkpts", None)
        target_k = kwargs.pop("target_k", 0)
        kpts = kwargs.pop("kpts", None)
        kmesh = kwargs.pop("kmesh", None)
        kconserv = kwargs.pop("kconserv", None)
        super().__init__(*args, **kwargs)
        self.nkpts = nkpts
        self.target_k = target_k
        self.kpts = kpts
        self.kmesh = kmesh
        self.kconserv = kconserv
        self.kmom = None
        self.davidson_only = False

    def _resolve_sector(self, nkpts=None, target_k=None):
        """Resolve optional sector arguments against the solver defaults."""
        if nkpts is None:
            nkpts = self.nkpts
        if nkpts is None:
            raise ValueError("nkpts must be supplied or set on the solver")
        if target_k is None:
            target_k = self.target_k
        return int(nkpts), int(target_k) % int(nkpts)

    def get_kmom(self, nkpts=None):
        """Return cached table-driven k-point arithmetic."""
        nkpts, _ = self._resolve_sector(nkpts)
        if self.kmom is not None and self.kmom.nkpts == nkpts:
            return self.kmom
        cell = getattr(self, "cell", None)
        if cell is None:
            cell = getattr(self, "mol", None)
        self.kmom = kcistrings.make_kpoint_momentum(
            nkpts, cell=cell, kpts=self.kpts, kmesh=self.kmesh,
            kconserv=self.kconserv)
        self.kconserv = self.kmom.kconserv
        return self.kmom

    def contract_1e(self, h1e, fcivec, norb, nelec, nkpts=None,
                    target_k=None, link_index=None, contract_map=None):
        """Contract the one-electron Hamiltonian with a sector CI vector."""
        nkpts, target_k = self._resolve_sector(nkpts, target_k)
        t0 = (logger.process_clock(), logger.perf_counter())
        ci1 = contract_1e_k(
            h1e, fcivec, norb, nelec, nkpts, target_k,
            link_index=link_index, contract_map=contract_map, log_obj=self,
            kmom=self.get_kmom(nkpts))
        logger.new_logger(self).timer_debug1("k-FCI contract_1e", *t0)
        return ci1

    def contract_2e(self, eri, fcivec, norb, nelec, nkpts=None,
                    target_k=None, link_index=None, contract_map=None):
        """Contract the two-electron Hamiltonian with a sector CI vector."""
        nkpts, target_k = self._resolve_sector(nkpts, target_k)
        t0 = (logger.process_clock(), logger.perf_counter())
        ci1 = contract_2e_k(
            eri, fcivec, norb, nelec, nkpts, target_k,
            link_index=link_index, contract_map=contract_map, log_obj=self,
            kmom=self.get_kmom(nkpts))
        logger.new_logger(self).timer_debug1("k-FCI contract_2e", *t0)
        return ci1

    def contract_ham(self, h1e, eri, fcivec, norb, nelec, nkpts=None,
                     target_k=None, link_index=None, contract_map=None):
        """Contract the complete electronic Hamiltonian."""
        nkpts, target_k = self._resolve_sector(nkpts, target_k)
        t0 = (logger.process_clock(), logger.perf_counter())
        contract_map = _as_contract_map(
            norb, nelec, nkpts, target_k, link_index=link_index,
            contract_map=contract_map, log_obj=self,
            kmom=self.get_kmom(nkpts))
        link_index = contract_map.link_index
        ci1 = self.contract_1e(
            h1e, fcivec, norb, nelec, nkpts=nkpts, target_k=target_k,
            link_index=link_index, contract_map=contract_map)
        ci1 += self.contract_2e(
            eri, fcivec, norb, nelec, nkpts=nkpts, target_k=target_k,
            link_index=link_index, contract_map=contract_map)
        logger.new_logger(self).timer_debug1("k-FCI contract_ham", *t0)
        return ci1

    def make_hdiag(self, h1e, eri, norb, nelec, nkpts=None, target_k=None,
                   link_index=None, compress=False, contract_map=None):
        """Build the Hamiltonian diagonal for the selected sector."""
        del compress
        nkpts, target_k = self._resolve_sector(nkpts, target_k)
        t0 = (logger.process_clock(), logger.perf_counter())
        nelec = _unpack_nelec(nelec, self.spin)
        hdiag = make_hdiag(
            h1e, eri, norb, nelec, nkpts, target_k,
            link_index=link_index, contract_map=contract_map, log_obj=self,
            kmom=self.get_kmom(nkpts))
        logger.new_logger(self).timer_debug1("k-FCI make_hdiag", *t0)
        return hdiag

    def make_hamiltonian(self, h1e, eri, norb, nelec, nkpts=None,
                         target_k=None, link_index=None, contract_map=None):
        """Construct the explicit Hamiltonian for the selected sector."""
        nkpts, target_k = self._resolve_sector(nkpts, target_k)
        t0 = (logger.process_clock(), logger.perf_counter())
        nelec = _unpack_nelec(nelec, self.spin)
        hmat = make_hamiltonian_k(
            h1e, eri, norb, nelec, nkpts, target_k,
            link_index=link_index, contract_map=contract_map, log_obj=self,
            kmom=self.get_kmom(nkpts), contract_fn=self.contract_ham)
        logger.new_logger(self).timer_debug1(
            "k-FCI make_hamiltonian", *t0)
        return hmat

    def energy(self, h1e, eri, fcivec, norb, nelec, nkpts=None,
               target_k=None, link_index=None, contract_map=None):
        """Evaluate the electronic energy of a sector CI vector."""
        nkpts, target_k = self._resolve_sector(nkpts, target_k)
        t0 = (logger.process_clock(), logger.perf_counter())
        nelec = _unpack_nelec(nelec, self.spin)
        value = energy(
            h1e, eri, fcivec, norb, nelec, nkpts, target_k,
            link_index=link_index, contract_map=contract_map, log_obj=self,
            kmom=self.get_kmom(nkpts), contract_fn=self.contract_ham)
        logger.new_logger(self).timer_debug1("k-FCI energy", *t0)
        return value

    def make_rdm1s(self, fcivec, norb, nelec, nkpts=None, target_k=None,
                   link_index=None):
        """Build spin-separated one-particle RDMs.

        Returns
        -------
        rdm1s : tuple of two ndarrays
            ``(dm1a, dm1b)``, each with shape ``(norb, norb)`` and dtype
            ``complex128``.
        """
        nkpts, target_k = self._resolve_sector(nkpts, target_k)
        t0 = (logger.process_clock(), logger.perf_counter())
        result = make_rdm1s(
            fcivec, norb, _unpack_nelec(nelec, self.spin), nkpts,
            target_k=target_k, link_index=link_index,
            kmom=self.get_kmom(nkpts))
        logger.new_logger(self).timer_debug1("k-FCI make_rdm1s", *t0)
        return result

    def make_rdm1(self, fcivec, norb, nelec, nkpts=None, target_k=None,
                  link_index=None):
        """Build the spin-summed one-particle RDM.

        Returns
        -------
        dm1 : ndarray, shape (norb, norb), dtype complex128
            Spin-summed one-particle RDM.
        """
        nkpts, target_k = self._resolve_sector(nkpts, target_k)
        t0 = (logger.process_clock(), logger.perf_counter())
        result = make_rdm1(
            fcivec, norb, _unpack_nelec(nelec, self.spin), nkpts,
            target_k=target_k, link_index=link_index,
            kmom=self.get_kmom(nkpts))
        logger.new_logger(self).timer_debug1("k-FCI make_rdm1", *t0)
        return result

    def make_rdm12s(self, fcivec, norb, nelec, nkpts=None, target_k=None,
                    link_index=None, reorder=True):
        """Build spin-separated one- and two-particle RDMs.

        Returns
        -------
        rdm1s : tuple of two ndarrays
            ``(dm1a, dm1b)``, each with shape ``(norb, norb)`` and dtype
            ``complex128``.
        rdm2s : tuple of three ndarrays
            ``(dm2aa, dm2ab, dm2bb)``, each with shape
            ``(norb, norb, norb, norb)`` and dtype ``complex128``.
        """
        nkpts, target_k = self._resolve_sector(nkpts, target_k)
        t0 = (logger.process_clock(), logger.perf_counter())
        result = make_rdm12s(
            fcivec, norb, _unpack_nelec(nelec, self.spin), nkpts,
            target_k=target_k, link_index=link_index, reorder=reorder,
            kmom=self.get_kmom(nkpts))
        logger.new_logger(self).timer_debug1("k-FCI make_rdm12s", *t0)
        return result

    def make_rdm12(self, fcivec, norb, nelec, nkpts=None, target_k=None,
                   link_index=None, reorder=True):
        """Build spin-summed one- and two-particle RDMs.

        Returns
        -------
        dm1 : ndarray, shape (norb, norb), dtype complex128
            Spin-summed one-particle RDM.
        dm2 : ndarray, shape (norb, norb, norb, norb), dtype complex128
            Spin-summed two-particle RDM.
        """
        nkpts, target_k = self._resolve_sector(nkpts, target_k)
        t0 = (logger.process_clock(), logger.perf_counter())
        result = make_rdm12(
            fcivec, norb, _unpack_nelec(nelec, self.spin), nkpts,
            target_k=target_k, link_index=link_index, reorder=reorder,
            kmom=self.get_kmom(nkpts))
        logger.new_logger(self).timer_debug1("k-FCI make_rdm12", *t0)
        return result

    def contract_ss(self, fcivec, norb, nelec, nkpts=None, target_k=None,
                    link_index=None, contract_map=None):
        """Apply spin-squared to a sector CI vector."""
        nkpts, target_k = self._resolve_sector(nkpts, target_k)
        t0 = (logger.process_clock(), logger.perf_counter())
        ci1 = contract_ss(
            fcivec, norb, _unpack_nelec(nelec, self.spin), nkpts, target_k,
            link_index=link_index, spin=self.spin,
            contract_map=contract_map, kmom=self.get_kmom(nkpts))
        logger.new_logger(self).timer_debug1("k-FCI contract_ss", *t0)
        return ci1

    def spin_square(self, fcivec, norb, nelec, nkpts=None, target_k=None,
                    link_index=None, **kwargs):
        """Evaluate spin-squared and the spin multiplicity."""
        nkpts, target_k = self._resolve_sector(nkpts, target_k)
        t0 = (logger.process_clock(), logger.perf_counter())
        result = spin_square(
            fcivec, norb, _unpack_nelec(nelec, self.spin), nkpts, target_k,
            link_index=link_index, spin=self.spin,
            kmom=self.get_kmom(nkpts), **kwargs)
        logger.new_logger(self).timer_debug1("k-FCI spin_square", *t0)
        return result

    def make_precond(self, hdiag, pspaceig=None, pspaceci=None, addr=None):
        """Construct the diagonal Davidson preconditioner."""
        return make_diag_precond(
            hdiag, pspaceig, pspaceci, addr, self.level_shift)

    def get_init_guess(self, norb, nelec, nkpts, target_k, nroots, hdiag):
        """Construct determinant-basis Davidson initial guesses."""
        return get_init_guess_k(
            norb, nelec, nkpts, target_k, nroots, hdiag, log_obj=self)

    def kernel(self, h1e, eri, norb, nelec, ci0=None, nkpts=None,
               target_k=None, tol=None, lindep=None, max_cycle=None,
               max_space=None, nroots=None, davidson_only=None,
               pspace_size=None, orbsym=None, wfnsym=None, ecore=0,
               **kwargs):
        """Solve the selected total-momentum sector."""
        del orbsym, wfnsym
        if nkpts is None:
            nkpts = self.nkpts
        if nkpts is None:
            nkpts = h1e.shape[0]
        if target_k is None:
            target_k = self.target_k
        self.nkpts = int(nkpts)
        self.target_k = int(target_k) % self.nkpts

        kmom = self.get_kmom(self.nkpts)
        link_index = _unpack(
            norb, nelec, None, self.nkpts, spin=self.spin, kmom=kmom)
        e, c = kernel_ms1(
            self, h1e, eri, norb, nelec, self.nkpts, self.target_k,
            ci0=ci0, link_index=link_index, tol=tol, lindep=lindep,
            max_cycle=max_cycle, max_space=max_space, nroots=nroots,
            davidson_only=davidson_only, pspace_size=pspace_size,
            ecore=ecore, **kwargs)
        self.eci, self.ci = e, c
        return e, c

    def fix_spin_(self, shift=0.1, ss=None, **kwargs):
        """Apply a spin penalty to this solver in place."""
        return fix_spin_(self, shift=shift, ss=ss, **kwargs)

    fix_spin = fix_spin_

    def eig(self, op, x0=None, precond=None, **kwargs):
        """Diagonalize an explicit matrix or iterative linear operator."""
        if isinstance(op, np.ndarray):
            hermi_err = np.linalg.norm(op - op.conj().T)
            self.converged = True
            if hermi_err < HERMI_THRESH:
                return scipy.linalg.eigh(op)
            return scipy.linalg.eig(op)

        self.converged, e, ci = lib.davidson1(
            lambda xs: [op(x) for x in xs], x0, precond,
            lessio=self.lessio, **kwargs)
        if kwargs.get("nroots", 1) == 1:
            self.converged = self.converged[0]
            e = e[0]
            ci = ci[0]
        return e, ci


FCI = FCISolver
