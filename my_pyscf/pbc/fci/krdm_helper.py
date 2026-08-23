
import ctypes
import sys

import numpy as np

from pyscf import lib
from pyscf.fci import cistring
from pyscf.fci.addons import _unpack_nelec

from mrh.lib.helper import load_library
from mrh.my_pyscf.pbc.fci import (
    direct_spin1_cplx,
    kcistrings,
    kfci_contract_map,
    rdm_helper,
    spin_op,
)

# Author: Bhavnesh Jangid

"""
Reduced-density-matrix and spin helpers for momentum-sector FCI.
"""

_kci_lib_initialized = False
_direct_rdm_lib_initialized = False
_contract_ss_lib_initialized = False
libpbckrdm = None
libpbcfci_k = None


def _init_kci_lib():
    """Configure C entry points for packed/full CI-vector conversion.

    The two functions are provided by ``rdm_helper.libpbcrdm``.  Their C
    arguments, in call order, are as follows.

    ``FCIkci_sector_to_full_cplx(ci_full, ci_sector, nblocks, blocks,``
    ``stra_ids, stra_offsets, strb_ids, strb_offsets, na_full, nb_full)``

    ``ci_full`` : ``double complex *``
        Output table of shape ``(na_full, nb_full)``.  The kernel zeroes the
        table and scatters the selected sector into it.
    ``ci_sector`` : ``double complex *``
        Input packed CI vector of length ``sector_size``.
    ``nblocks`` : ``int``
        Number of occupied alpha/beta momentum-block pairs.
    ``blocks`` : ``int *``
        Contiguous ``int32`` array of shape ``(nblocks, 6)``.  Each row is
        ``[ka, kb, na, nb, offset, size]``: alpha and beta momenta, their
        string counts, the block's starting packed-vector address, and
        ``size = na * nb``.
    ``stra_ids`` / ``strb_ids`` : ``int *``
        Global alpha/beta string addresses, flattened after grouping the
        strings by momentum.  Their lengths are ``na_full`` and ``nb_full``.
    ``stra_offsets`` / ``strb_offsets`` : ``int *``
        Arrays of length ``nkpts + 1``; entries ``[k:k+2]`` delimit the IDs
        belonging to momentum ``k`` in the corresponding ID array.
    ``na_full`` / ``nb_full`` : ``int``
        Total numbers of alpha and beta strings in the full determinant
        space.  ``nb_full`` is also the row-major stride of ``ci_full``.

    ``FCIkci_full_to_sector_cplx(ci_sector, ci_full, nblocks, blocks,``
    ``stra_ids, stra_offsets, strb_ids, strb_offsets, nb_full)``

    ``ci_sector`` : ``double complex *``
        Output packed CI vector of length ``sector_size``.
    ``ci_full`` : ``double complex *``
        Input full CI table with shape ``(na_full, nb_full)``.
    ``nblocks``, ``blocks``, ``stra_ids``, ``stra_offsets``, ``strb_ids``,
    ``strb_offsets``
        The same layout arrays described for the scatter function above.
    ``nb_full`` : ``int``
        Number of beta strings and therefore the row-major stride of
        ``ci_full``.  ``na_full`` is not needed when gathering only listed
        block entries.

    The configuration is idempotent: subsequent calls return immediately.

    Returns
    -------
    None
        The function mutates the ``ctypes`` signatures and initialization
        flag; it does not execute either C kernel.
    """
    global _kci_lib_initialized
    if _kci_lib_initialized:
        return

    rdm_helper.libpbcrdm.FCIkci_sector_to_full_cplx.argtypes = [
        ctypes.c_void_p,  # ci_full
        ctypes.c_void_p,  # ci_sector
        ctypes.c_int,     # nblocks
        ctypes.c_void_p,  # blocks
        ctypes.c_void_p,  # stra_ids
        ctypes.c_void_p,  # stra_offsets
        ctypes.c_void_p,  # strb_ids
        ctypes.c_void_p,  # strb_offsets
        ctypes.c_int,     # na_full
        ctypes.c_int,     # nb_full
    ]
    rdm_helper.libpbcrdm.FCIkci_sector_to_full_cplx.restype = None
    rdm_helper.libpbcrdm.FCIkci_full_to_sector_cplx.argtypes = [
        ctypes.c_void_p,  # ci_sector
        ctypes.c_void_p,  # ci_full
        ctypes.c_int,     # nblocks
        ctypes.c_void_p,  # blocks
        ctypes.c_void_p,  # stra_ids
        ctypes.c_void_p,  # stra_offsets
        ctypes.c_void_p,  # strb_ids
        ctypes.c_void_p,  # strb_offsets
        ctypes.c_int,     # nb_full
    ]
    rdm_helper.libpbcrdm.FCIkci_full_to_sector_cplx.restype = None
    _kci_lib_initialized = True


def _init_contract_ss_lib():
    """Load and configure the direct packed-sector spin-squared kernel.

    The C signature is ``FCIcontract_ss_k(ci0, ci1, norb, neleca, nelecb,``
    ``nkpts, nblocks, blocks, linka, nstra, nlinka, linkb, nstrb, nlinkb,``
    ``stra_ids, stra_offsets, strb_ids, strb_offsets, str2tot_a,``
    ``str2tot_b)``.  Its arguments are:

    ``ci0`` : ``double complex *``
        Input packed CI vector of length ``sector_size``.
    ``ci1`` : ``double complex *``
        Output packed vector of the same length, containing ``S**2 |ci0>``.
    ``norb`` : ``int``
        Total number of active spin-independent orbitals across all k-points.
    ``neleca`` / ``nelecb`` : ``int``
        Numbers of alpha and beta electrons.
    ``nkpts`` : ``int``
        Number of momentum points.
    ``nblocks`` : ``int``
        Number of alpha/beta momentum-block pairs in the packed sector.
    ``blocks`` : ``int *``
        ``int32[nblocks, 6]`` array whose rows are
        ``[ka, kb, na, nb, offset, size]``.  The fields give the alpha and
        beta momenta, string counts, packed-vector offset, and
        ``size = na * nb``.
    ``linka`` / ``linkb`` : ``int *``
        Alpha/beta link tables with shapes ``(nstra, nlinka, 8)`` and
        ``(nstrb, nlinkb, 8)``.  Each record is ``[p, q, target, sign, k0,``
        ``k_p, k_q, dK]``: creation and annihilation orbital addresses,
        target global string address, fermionic sign, source-string momentum,
        creation/annihilation momenta, and momentum transfer.
    ``nstra`` / ``nstrb`` : ``int``
        Total numbers of alpha/beta strings in the full string spaces.
    ``nlinka`` / ``nlinkb`` : ``int``
        Numbers of link records stored per alpha/beta source string.
    ``stra_ids`` / ``strb_ids`` : ``int *``
        Global alpha/beta string addresses grouped contiguously by momentum;
        their lengths are ``nstra`` and ``nstrb``.
    ``stra_offsets`` / ``strb_offsets`` : ``int *``
        ``int32[nkpts + 1]`` boundaries for the momentum groups in
        ``stra_ids`` and ``strb_ids``.
    ``str2tot_a`` / ``str2tot_b`` : ``int *``
        Global-to-local lookup tables with shapes ``(nkpts, nstra)`` and
        ``(nkpts, nstrb)``.  Entry ``[k, global_id]`` is the string's local
        address in momentum group ``k``, or ``-1`` if it is not in that group.

    The kernel applies ``S**2`` without embedding the vector in the full CI
    space and has a ``void`` return type.

    The shared library and its ``ctypes`` signature are initialized only once.

    Returns
    -------
    None
        Updates ``libpbcfci_k`` and the module initialization flag.
    """
    global _contract_ss_lib_initialized, libpbcfci_k
    if _contract_ss_lib_initialized:
        return

    libpbcfci_k = load_library("libpbc_fci_contract_k")
    libpbcfci_k.FCIcontract_ss_k.argtypes = [
        ctypes.c_void_p,  # ci0
        ctypes.c_void_p,  # ci1
        ctypes.c_int,     # norb
        ctypes.c_int,     # neleca
        ctypes.c_int,     # nelecb
        ctypes.c_int,     # nkpts
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
    ]
    libpbcfci_k.FCIcontract_ss_k.restype = None
    _contract_ss_lib_initialized = True


def _init_direct_rdm_lib():
    """Load and configure direct packed-sector RDM kernels.

    ``FCIkci_make_rdm1s_direct(dm1a, dm1b, ci, norb, nkpts, nblocks,``
    ``blocks, linka, nstra, nlinka, linkb, nstrb, nlinkb, str2tot_a,``
    ``str2tot_b)`` builds the spin-resolved 1-RDMs.  Its arguments are:

    ``dm1a`` / ``dm1b`` : ``double complex *``
        Output alpha/beta 1-RDM buffers, each with shape ``(norb, norb)``.
    ``ci`` : ``double complex *``
        Input packed CI vector of length ``sector_size``.
    ``norb`` : ``int``
        Total number of active spin-independent orbitals across all k-points.
    ``nkpts`` : ``int``
        Number of momentum points.
    ``nblocks`` : ``int``
        Number of alpha/beta momentum-block pairs in the packed sector.
    ``blocks`` : ``int *``
        ``int32[nblocks, 6]`` array.  Each row is
        ``[ka, kb, na, nb, offset, size]``: alpha/beta momentum, string
        counts, packed-vector offset, and ``size = na * nb``.
    ``linka`` / ``linkb`` : ``int *``
        Alpha/beta excitation tables with shapes ``(nstra, nlinka, 8)`` and
        ``(nstrb, nlinkb, 8)``.  Each record is ``[p, q, target, sign, k0,``
        ``k_p, k_q, dK]``; see :func:`_init_contract_ss_lib` for the meaning
        of those fields.
    ``nstra`` / ``nstrb`` : ``int``
        Total numbers of alpha/beta strings in the full string spaces.
    ``nlinka`` / ``nlinkb`` : ``int``
        Numbers of excitation records per alpha/beta source string.
    ``str2tot_a`` / ``str2tot_b`` : ``int *``
        ``int32`` global-to-local string maps with shapes
        ``(nkpts, nstra)`` and ``(nkpts, nstrb)``.  Entry ``[k, global_id]``
        is the local address in momentum group ``k``, or ``-1`` when absent.

    ``FCIkci_make_rdm12s_direct(dm1a, dm1b, dm2aa, dm2ab, dm2bb, ci,``
    ``norb, nkpts, nblocks, blocks, linka, nstra, nlinka, linkb, nstrb,``
    ``nlinkb, stra_ids, stra_offsets, strb_ids, strb_offsets, str2tot_a,``
    ``str2tot_b, kneg)`` builds both RDM orders.  Its arguments are:

    ``dm1a`` / ``dm1b`` : ``double complex *``
        Output alpha/beta 1-RDM buffers of shape ``(norb, norb)``.
    ``dm2aa`` / ``dm2ab`` / ``dm2bb`` : ``double complex *``
        Output alpha-alpha, alpha-beta, and beta-beta 2-RDM buffers, each of
        shape ``(norb, norb, norb, norb)``.
    ``ci`` : ``double complex *``
        Input packed CI vector of length ``sector_size``.
    ``norb``, ``nkpts``, ``nblocks``, ``blocks``, ``linka``, ``nstra``,
    ``nlinka``, ``linkb``, ``nstrb``, ``nlinkb``
        The same orbital, layout, and link-table arguments defined for
        ``FCIkci_make_rdm1s_direct`` above.
    ``stra_ids`` / ``strb_ids`` : ``int *``
        Global alpha/beta string addresses grouped contiguously by momentum;
        their lengths are ``nstra`` and ``nstrb``.
    ``stra_offsets`` / ``strb_offsets`` : ``int *``
        ``int32[nkpts + 1]`` boundaries for the momentum groups in the
        corresponding string-ID arrays.
    ``str2tot_a`` / ``str2tot_b`` : ``int *``
        The same global-to-local string maps defined for the 1-RDM kernel.
    ``kneg`` : ``int *``
        ``int32[nkpts]`` additive-inverse lookup: ``kneg[k]`` is the momentum
        that sums with ``k`` to the zero-momentum element under the active
        momentum algebra.

    Both C functions return a nonzero failure status for temporary allocation
    or layout-construction errors, represented as ``ctypes.c_int`` (zero
    means success).  The Python callers translate any nonzero value to
    :class:`MemoryError`.  Configuration is idempotent.

    Returns
    -------
    None
        Updates ``libpbckrdm`` and the module initialization flag.
    """
    global _direct_rdm_lib_initialized, libpbckrdm
    if _direct_rdm_lib_initialized:
        return

    libpbckrdm = load_library("libpbc_kfci_rdm")
    void_p = ctypes.c_void_p
    int_t = ctypes.c_int
    libpbckrdm.FCIkci_make_rdm1s_direct.argtypes = [
        void_p,  # dm1a
        void_p,  # dm1b
        void_p,  # ci
        int_t,   # norb
        int_t,   # nkpts
        int_t,   # nblocks
        void_p,  # blocks
        void_p,  # linka
        int_t,   # nstra
        int_t,   # nlinka
        void_p,  # linkb
        int_t,   # nstrb
        int_t,   # nlinkb
        void_p,  # str2tot_a
        void_p,  # str2tot_b
    ]
    libpbckrdm.FCIkci_make_rdm1s_direct.restype = int_t
    libpbckrdm.FCIkci_make_rdm12s_direct.argtypes = [
        void_p,  # dm1a
        void_p,  # dm1b
        void_p,  # dm2aa
        void_p,  # dm2ab
        void_p,  # dm2bb
        void_p,  # ci
        int_t,   # norb
        int_t,   # nkpts
        int_t,   # nblocks
        void_p,  # blocks
        void_p,  # linka
        int_t,   # nstra
        int_t,   # nlinka
        void_p,  # linkb
        int_t,   # nstrb
        int_t,   # nlinkb
        void_p,  # stra_ids
        void_p,  # stra_offsets
        void_p,  # strb_ids
        void_p,  # strb_offsets
        void_p,  # str2tot_a
        void_p,  # str2tot_b
        void_p,  # kneg
    ]
    libpbckrdm.FCIkci_make_rdm12s_direct.restype = int_t
    _direct_rdm_lib_initialized = True


def _unpack_k(norb, nelec, nkpts, link_index=None, spin=None, kmom=None,
              kconserv=None):
    """Return momentum-aware alpha and beta link-index tables.

    Parameters
    ----------
    norb : int
        Total active-orbital count; it must be divisible by ``nkpts``.
    nelec : int or tuple of two ints
        Total electron count or ``(N_alpha, N_beta)``.
    nkpts : int
        Number of momentum points.
    link_index : tuple of two ndarrays or None, optional
        Existing alpha/beta link tables.  Each table must have eight fields in
        its final dimension.  Valid tables are returned unchanged.
    spin : int or None, optional
        ``N_alpha - N_beta`` used when an integer ``nelec`` is unpacked.  A
        value of zero also permits alpha and beta tables to share storage.
    kmom : KPointMomentum or None, optional
        Precomputed momentum arithmetic used to generate missing tables.
    kconserv : ndarray or None, optional
        Momentum-conservation table used when ``kmom`` is not supplied.

    Returns
    -------
    link_indexa, link_indexb : tuple of ndarrays
        Momentum-aware link tables with shapes ``(nstra, nlinka, 8)`` and
        ``(nstrb, nlinkb, 8)`` and dtype ``int32`` when generated here.

    Notes
    -----
    This function duplicates ``pbc.fci.addons._unpack_k``.  Maintaining both
    copies risks divergent validation or generation behavior; ``krdm_helper``
    should import the shared add-on helper instead of defining this copy.
    """
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


def _as_contract_map(norb, nelec, nkpts, target_k=0, link_index=None,
                     spin=None, contract_map=None, kmom=None,
                     kconserv=None):
    """Return a determinant-layout map for RDM and spin operations.

    An existing ``contract_map`` is returned unchanged.  Otherwise the
    function obtains momentum-aware link tables with :func:`_unpack_k` and
    constructs a :class:`KFCILayoutMap` for ``target_k``.

    This helper differs intentionally from
    ``direct_spin1_kfci._as_contract_map``: the latter validates or builds a
    full :class:`KFCIContractMap`, including two-electron contraction
    structures.  RDM and spin kernels require only the determinant layout.

    Parameters
    ----------
    norb, nelec, nkpts, target_k, link_index, spin, kmom, kconserv
        See :func:`_unpack_k` for the link-generation arguments;
        ``target_k`` selects the packed total-momentum sector.
    contract_map : KFCILayoutMap or KFCIContractMap or None, optional
        Existing compatible layout or full contraction map.

    Returns
    -------
    layout : KFCILayoutMap or KFCIContractMap
        Object exposing packed blocks, link tables, string maps, sector size,
        and momentum arithmetic.
    """
    if contract_map is not None:
        return contract_map
    link_index = _unpack_k(
        norb, nelec, nkpts, link_index=link_index, spin=spin, kmom=kmom,
        kconserv=kconserv)
    return kfci_contract_map.KFCILayoutMap.build(
        norb, nelec, nkpts, target_k, link_index=link_index, kmom=kmom,
        kconserv=kconserv)


def embed_ksector_ci_to_full(fcivec, norb, nelec, nkpts, target_k=0,
                             link_index=None, spin=None, contract_map=None,
                             kmom=None, kconserv=None):
    """Scatter a packed momentum-sector CI vector into the full CI table.

    Determinants belonging to ``target_k`` are copied to their global alpha
    and beta string addresses.  Entries belonging to every other total-
    momentum sector are set to zero by the C kernel.

    Parameters
    ----------
    fcivec : array_like, shape (sector_size,)
        CI coefficients in packed momentum-block order.
    norb : int
        Total number of active orbitals.
    nelec : int or tuple of two ints
        Total electron count or ``(N_alpha, N_beta)``.
    nkpts : int
        Number of momentum points.
    target_k : int, optional
        Total-momentum sector represented by ``fcivec``.
    link_index : tuple of two ndarrays or None, optional
        Momentum-aware alpha/beta link tables used to build the layout.
    spin : int or None, optional
        Spin projection used to unpack an integer electron count.
    contract_map : KFCILayoutMap or KFCIContractMap or None, optional
        Precomputed compatible layout.
    kmom : KPointMomentum or None, optional
        Precomputed momentum arithmetic.
    kconserv : ndarray or None, optional
        Momentum-conservation table used when constructing ``kmom``.

    Returns
    -------
    ci_full : ndarray, shape (nstra, nstrb), dtype complex128
        Full alpha-by-beta spin-string coefficient table, where
        ``nstra = C(norb, N_alpha)`` and ``nstrb = C(norb, N_beta)``.
    """
    neleca, nelecb = _unpack_nelec(nelec, spin)
    contract_map = _as_contract_map(
        norb, (neleca, nelecb), nkpts, target_k=target_k,
        link_index=link_index, spin=spin, contract_map=contract_map,
        kmom=kmom, kconserv=kconserv)
    fcivec = np.asarray(fcivec, dtype=np.complex128, order="C")
    assert fcivec.size == contract_map.sector_size, (
        fcivec.size, contract_map.sector_size)

    nstra = cistring.num_strings(norb, neleca)
    nstrb = cistring.num_strings(norb, nelecb)
    ci_full = np.empty((nstra, nstrb), dtype=np.complex128, order="C")

    _init_kci_lib()
    rdm_helper.libpbcrdm.FCIkci_sector_to_full_cplx(
        ci_full.ctypes.data_as(ctypes.c_void_p),
        fcivec.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(contract_map.blocks.shape[0]),
        contract_map.blocks.ctypes.data_as(ctypes.c_void_p),
        contract_map.stra_ids.ctypes.data_as(ctypes.c_void_p),
        contract_map.stra_offsets.ctypes.data_as(ctypes.c_void_p),
        contract_map.strb_ids.ctypes.data_as(ctypes.c_void_p),
        contract_map.strb_offsets.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nstra),
        ctypes.c_int(nstrb),
    )
    return ci_full


def extract_ksector_ci_from_full(ci_full, norb, nelec, nkpts, target_k=0,
                                 link_index=None, spin=None,
                                 contract_map=None, kmom=None,
                                 kconserv=None):
    """Gather one packed total-momentum sector from a full CI table.

    Parameters are the same as :func:`embed_ksector_ci_to_full`, except
    ``ci_full`` has shape ``(nstra, nstrb)`` and contains coefficients at
    global spin-string addresses.

    Returns
    -------
    fcivec : ndarray, shape (sector_size,), dtype complex128
        Coefficients in the packed block order described by the layout map.

    Notes
    -----
    Layout resolution and the C pointer list intentionally mirror
    :func:`embed_ksector_ci_to_full`; only the data-flow direction and the
    final full-table dimension passed to C differ.
    """
    neleca, nelecb = _unpack_nelec(nelec, spin)
    contract_map = _as_contract_map(
        norb, (neleca, nelecb), nkpts, target_k=target_k,
        link_index=link_index, spin=spin, contract_map=contract_map,
        kmom=kmom, kconserv=kconserv)

    nstrb = cistring.num_strings(norb, nelecb)
    ci_full = np.asarray(ci_full, dtype=np.complex128, order="C")
    fcivec = np.empty(
        contract_map.sector_size, dtype=np.complex128, order="C")

    _init_kci_lib()
    rdm_helper.libpbcrdm.FCIkci_full_to_sector_cplx(
        fcivec.ctypes.data_as(ctypes.c_void_p),
        ci_full.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(contract_map.blocks.shape[0]),
        contract_map.blocks.ctypes.data_as(ctypes.c_void_p),
        contract_map.stra_ids.ctypes.data_as(ctypes.c_void_p),
        contract_map.stra_offsets.ctypes.data_as(ctypes.c_void_p),
        contract_map.strb_ids.ctypes.data_as(ctypes.c_void_p),
        contract_map.strb_offsets.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(nstrb),
    )
    return fcivec


def make_rdm1s_ref(fcivec, norb, nelec, nkpts, target_k=0,
                   link_index=None, spin=None, kmom=None, kconserv=None):
    """Build reference spin-resolved 1-RDMs by full-CI embedding.

    The packed vector is first scattered into the full alpha-by-beta string
    table and then passed to :func:`direct_spin1_cplx.make_rdm1s`.  This path
    is primarily an independent reference for :func:`make_rdm1s`.

    Parameters
    ----------
    fcivec, norb, nelec, nkpts, target_k, link_index, spin, kmom, kconserv
        See :func:`embed_ksector_ci_to_full`.

    Returns
    -------
    rdm1a, rdm1b : tuple of ndarrays
        Alpha and beta 1-RDMs, each with shape ``(norb, norb)`` and dtype
        ``complex128``.

    Notes
    -----
    The embedding setup is intentionally repeated in :func:`make_rdm12s_ref`
    so that each reference routine can call the corresponding full-CI RDM
    implementation directly.
    """
    ci_full = embed_ksector_ci_to_full(
        fcivec, norb, nelec, nkpts, target_k=target_k,
        link_index=link_index, spin=spin, kmom=kmom, kconserv=kconserv)
    return direct_spin1_cplx.make_rdm1s(
        ci_full, norb, _unpack_nelec(nelec, spin), link_index=None)


def make_rdm1_ref(fcivec, norb, nelec, nkpts, target_k=0,
                  link_index=None, spin=None, kmom=None, kconserv=None):
    """Build a reference spin-summed 1-RDM by full-CI embedding.

    Parameters
    ----------
    fcivec, norb, nelec, nkpts, target_k, link_index, spin, kmom, kconserv
        See :func:`make_rdm1s_ref`.

    Returns
    -------
    rdm1 : ndarray, shape (norb, norb), dtype complex128
        Spin-summed 1-RDM in the public ``(p, q)`` convention.

    Notes
    -----
    The spin sum and conjugate transpose below are duplicated in
    :func:`make_rdm1`; the reference and direct paths must retain the same
    output convention.
    """
    rdm1a, rdm1b = make_rdm1s_ref(
        fcivec, norb, nelec, nkpts, target_k=target_k,
        link_index=link_index, spin=spin, kmom=kmom, kconserv=kconserv)
    return (rdm1a + rdm1b).conj().T


def make_rdm12s_ref(fcivec, norb, nelec, nkpts, target_k=0,
                    link_index=None, reorder=True, spin=None, kmom=None,
                    kconserv=None):
    """Build reference spin-resolved 1-/2-RDMs by full-CI embedding.

    Parameters
    ----------
    fcivec, norb, nelec, nkpts, target_k, link_index, spin, kmom, kconserv
        See :func:`embed_ksector_ci_to_full`.
    reorder : bool, optional
        If true, convert the same-spin 2-RDMs from the raw excitation-
        operator ordering to PySCF's reordered RDM convention.

    Returns
    -------
    (rdm1a, rdm1b) : tuple of ndarrays
        Alpha and beta 1-RDMs, each of shape ``(norb, norb)``.
    (rdm2aa, rdm2ab, rdm2bb) : tuple of ndarrays
        Alpha-alpha, alpha-beta, and beta-beta 2-RDMs, each of shape
        ``(norb, norb, norb, norb)``.  All returned arrays have dtype
        ``complex128``.

    Notes
    -----
    This repeats the embedding step in :func:`make_rdm1s_ref` but calls the
    full-CI routine that produces both RDM orders in one pass.
    """
    ci_full = embed_ksector_ci_to_full(
        fcivec, norb, nelec, nkpts, target_k=target_k,
        link_index=link_index, spin=spin, kmom=kmom, kconserv=kconserv)
    return direct_spin1_cplx.make_rdm12s(
        ci_full, norb, _unpack_nelec(nelec, spin), link_index=None,
        reorder=reorder)


def make_rdm12_ref(fcivec, norb, nelec, nkpts, target_k=0,
                   link_index=None, reorder=True, spin=None, kmom=None,
                   kconserv=None):
    """Build reference spin-summed 1-/2-RDMs by full-CI embedding.

    Parameters
    ----------
    fcivec, norb, nelec, nkpts, target_k, link_index, reorder, spin, kmom,
    kconserv
        See :func:`make_rdm12s_ref`.

    Returns
    -------
    rdm1 : ndarray, shape (norb, norb), dtype complex128
        Spin-summed 1-RDM.
    rdm2 : ndarray, shape (norb, norb, norb, norb), dtype complex128
        Spin-summed 2-RDM.  The beta-alpha contribution is recovered from
        ``rdm2ab`` by exchanging its particle-index pairs.

    Notes
    -----
    The spin-combination formulas below are duplicated in
    :func:`make_rdm12`; keeping them identical makes the embedded reference
    and direct packed-sector results directly comparable.
    """
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = make_rdm12s_ref(
        fcivec, norb, nelec, nkpts, target_k=target_k,
        link_index=link_index, reorder=reorder, spin=spin, kmom=kmom,
        kconserv=kconserv)
    dm1 = dm1a + dm1b
    dm2 = dm2aa + dm2bb + dm2ab + dm2ab.transpose(2, 3, 0, 1)
    return dm1.conj().T, dm2


def _direct_rdm_inputs(fcivec, norb, nelec, nkpts, target_k, link_index,
                       spin, kmom, kconserv):
    """Validate and prepare common inputs for direct packed-sector RDM calls.

    Parameters
    ----------
    fcivec : array_like, shape (sector_size,)
        Packed momentum-sector CI coefficients.
    norb, nelec, nkpts, target_k, link_index, spin, kmom, kconserv
        See :func:`embed_ksector_ci_to_full`.

    Returns
    -------
    fcivec : ndarray, shape (sector_size,), dtype complex128
        C-contiguous CI coefficients.  The input's dimensional shape is
        retained, although direct kernels interpret its storage as flat.
    layout : KFCILayoutMap or KFCIContractMap
        Validated determinant layout used by the C kernel.

    Raises
    ------
    ValueError
        If the number of CI coefficients differs from the selected sector's
        determinant count.

    Notes
    -----
    This helper factors the layout construction and size check shared by
    :func:`make_rdm1s` and :func:`make_rdm12s`; those functions still repeat
    their output allocation and C-pointer setup because their ABIs differ.
    """
    neleca, nelecb = _unpack_nelec(nelec, spin)
    layout = _as_contract_map(
        norb, (neleca, nelecb), nkpts, target_k=target_k,
        link_index=link_index, spin=spin, kmom=kmom, kconserv=kconserv)
    fcivec = np.asarray(fcivec, dtype=np.complex128, order="C")
    if fcivec.size != layout.sector_size:
        raise ValueError(
            "CI vector has size {}, expected {} for momentum sector {}"
            .format(fcivec.size, layout.sector_size, layout.target_k))
    return fcivec, layout


def make_rdm1s(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
               spin=None, kmom=None, kconserv=None):
    """Build spin-resolved 1-RDMs directly in the packed momentum sector.

    Parameters
    ----------
    fcivec, norb, nelec, nkpts, target_k, link_index, spin, kmom, kconserv
        See :func:`_direct_rdm_inputs`.

    Returns
    -------
    rdm1a, rdm1b : tuple of ndarrays
        Alpha and beta 1-RDMs, each with shape ``(norb, norb)`` and dtype
        ``complex128``.

    Raises
    ------
    ValueError
        If ``fcivec`` does not have the selected sector size.
    MemoryError
        If the C kernel cannot allocate its temporary storage.

    Notes
    -----
    Unlike :func:`make_rdm1s_ref`, this routine never constructs the full
    ``nstra * nstrb`` CI table.
    """
    fcivec, layout = _direct_rdm_inputs(
        fcivec, norb, nelec, nkpts, target_k, link_index, spin, kmom,
        kconserv)
    linka, linkb = layout.link_index
    dm1a = np.empty((norb, norb), dtype=np.complex128, order="C")
    dm1b = np.empty_like(dm1a)
    _init_direct_rdm_lib()
    with lib.with_omp_threads(lib.num_threads()):
        err = libpbckrdm.FCIkci_make_rdm1s_direct(
            dm1a.ctypes.data_as(ctypes.c_void_p),
            dm1b.ctypes.data_as(ctypes.c_void_p),
            fcivec.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(norb), ctypes.c_int(nkpts),
            ctypes.c_int(layout.blocks.shape[0]),
            layout.blocks.ctypes.data_as(ctypes.c_void_p),
            linka.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(linka.shape[0]), ctypes.c_int(linka.shape[1]),
            linkb.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(linkb.shape[0]), ctypes.c_int(linkb.shape[1]),
            layout.str2tot_a.ctypes.data_as(ctypes.c_void_p),
            layout.str2tot_b.ctypes.data_as(ctypes.c_void_p),
        )
    if err:
        raise MemoryError("direct momentum-sector 1-RDM allocation failed")
    return dm1a, dm1b


def make_rdm1(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
              spin=None, kmom=None, kconserv=None):
    """Build a spin-summed 1-RDM directly in the momentum sector.

    Parameters
    ----------
    fcivec, norb, nelec, nkpts, target_k, link_index, spin, kmom, kconserv
        See :func:`make_rdm1s`.

    Returns
    -------
    rdm1 : ndarray, shape (norb, norb), dtype complex128
        Spin-summed 1-RDM in the public ``(p, q)`` convention.

    Notes
    -----
    The final spin sum and conjugate transpose duplicate
    :func:`make_rdm1_ref` so both implementations expose the same convention.
    """
    dm1a, dm1b = make_rdm1s(
        fcivec, norb, nelec, nkpts, target_k=target_k,
        link_index=link_index, spin=spin, kmom=kmom, kconserv=kconserv)
    return (dm1a + dm1b).conj().T


def make_rdm12s(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
                reorder=True, spin=None, kmom=None, kconserv=None):
    """Build spin-resolved 1-/2-RDMs directly in the momentum sector.

    Parameters
    ----------
    fcivec, norb, nelec, nkpts, target_k, link_index, spin, kmom, kconserv
        See :func:`_direct_rdm_inputs`.
    reorder : bool, optional
        If true, convert the same-spin 2-RDMs from the raw excitation-
        operator ordering to PySCF's reordered RDM convention.  The mixed-
        spin block requires no such correction.

    Returns
    -------
    (rdm1a, rdm1b) : tuple of ndarrays
        Alpha and beta 1-RDMs, each of shape ``(norb, norb)`` and dtype
        ``complex128``.
    (rdm2aa, rdm2ab, rdm2bb) : tuple of ndarrays
        Alpha-alpha, alpha-beta, and beta-beta 2-RDMs, each of shape
        ``(norb, norb, norb, norb)`` and dtype ``complex128``.

    Raises
    ------
    ValueError
        If ``fcivec`` does not have the selected sector size.
    MemoryError
        If the C kernel cannot allocate its temporary storage.

    Notes
    -----
    This shares input preparation with :func:`make_rdm1s`, but necessarily
    repeats output allocation and C-pointer plumbing for the larger C ABI.
    """
    fcivec, layout = _direct_rdm_inputs(
        fcivec, norb, nelec, nkpts, target_k, link_index, spin, kmom,
        kconserv)
    linka, linkb = layout.link_index
    dm1a = np.empty((norb, norb), dtype=np.complex128, order="C")
    dm1b = np.empty_like(dm1a)
    dm2aa = np.empty((norb,) * 4, dtype=np.complex128, order="C")
    dm2ab = np.empty_like(dm2aa)
    dm2bb = np.empty_like(dm2aa)
    _init_direct_rdm_lib()
    with lib.with_omp_threads(lib.num_threads()):
        err = libpbckrdm.FCIkci_make_rdm12s_direct(
            dm1a.ctypes.data_as(ctypes.c_void_p),
            dm1b.ctypes.data_as(ctypes.c_void_p),
            dm2aa.ctypes.data_as(ctypes.c_void_p),
            dm2ab.ctypes.data_as(ctypes.c_void_p),
            dm2bb.ctypes.data_as(ctypes.c_void_p),
            fcivec.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(norb), ctypes.c_int(nkpts),
            ctypes.c_int(layout.blocks.shape[0]),
            layout.blocks.ctypes.data_as(ctypes.c_void_p),
            linka.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(linka.shape[0]), ctypes.c_int(linka.shape[1]),
            linkb.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(linkb.shape[0]), ctypes.c_int(linkb.shape[1]),
            layout.stra_ids.ctypes.data_as(ctypes.c_void_p),
            layout.stra_offsets.ctypes.data_as(ctypes.c_void_p),
            layout.strb_ids.ctypes.data_as(ctypes.c_void_p),
            layout.strb_offsets.ctypes.data_as(ctypes.c_void_p),
            layout.str2tot_a.ctypes.data_as(ctypes.c_void_p),
            layout.str2tot_b.ctypes.data_as(ctypes.c_void_p),
            layout.kmom.kneg.ctypes.data_as(ctypes.c_void_p),
        )
    if err:
        raise MemoryError("direct momentum-sector 1-/2-RDM allocation failed")
    if reorder:
        # The full-CI driver reorders with its internal transposed 1-RDM and
        # transposes that matrix only when returning it to Python.
        rdm_helper.reorder_rdm(dm1a.T, dm2aa, inplace=True)
        rdm_helper.reorder_rdm(dm1b.T, dm2bb, inplace=True)
    return (dm1a, dm1b), (dm2aa, dm2ab, dm2bb)


def make_rdm12(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
               reorder=True, spin=None, kmom=None, kconserv=None):
    """Build spin-summed 1-/2-RDMs directly in the momentum sector.

    Parameters
    ----------
    fcivec, norb, nelec, nkpts, target_k, link_index, reorder, spin, kmom,
    kconserv
        See :func:`make_rdm12s`.

    Returns
    -------
    rdm1 : ndarray, shape (norb, norb), dtype complex128
        Spin-summed 1-RDM.
    rdm2 : ndarray, shape (norb, norb, norb, norb), dtype complex128
        Spin-summed 2-RDM, including the beta-alpha block reconstructed by
        exchanging the two index pairs of ``rdm2ab``.

    Notes
    -----
    The composition formulas duplicate :func:`make_rdm12_ref` deliberately,
    providing a direct convention check between the C and reference paths.
    """
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = make_rdm12s(
        fcivec, norb, nelec, nkpts, target_k=target_k,
        link_index=link_index, reorder=reorder, spin=spin, kmom=kmom,
        kconserv=kconserv)
    dm1 = dm1a + dm1b
    dm2 = dm2aa + dm2bb + dm2ab + dm2ab.transpose(2, 3, 0, 1)
    return dm1.conj().T, dm2


def contract_ss_embedded(fcivec, norb, nelec, nkpts, target_k=0,
                         link_index=None, spin=None, contract_map=None,
                         kmom=None, kconserv=None):
    """Apply ``S**2`` through a full-space embedded CI vector.

    This reference implementation scatters the packed vector into the full
    determinant table, calls :func:`spin_op.contract_ss0`, and gathers the
    selected momentum sector back into packed order.

    Parameters
    ----------
    fcivec, norb, nelec, nkpts, target_k, link_index, spin, contract_map,
    kmom, kconserv
        See :func:`embed_ksector_ci_to_full`.

    Returns
    -------
    ci1 : ndarray, shape (sector_size,), dtype complex128
        Packed coefficients of ``S**2 |fcivec>``.

    Notes
    -----
    The scatter/operate/gather sequence intentionally duplicates the data
    mapping exercised independently by :func:`embed_ksector_ci_to_full` and
    :func:`extract_ksector_ci_from_full`.  It provides a reference for the
    direct packed-sector kernel in :func:`contract_ss`.
    """
    ci_full = embed_ksector_ci_to_full(
        fcivec, norb, nelec, nkpts, target_k=target_k,
        link_index=link_index, spin=spin, contract_map=contract_map,
        kmom=kmom, kconserv=kconserv)
    ci_full = np.asarray(ci_full, dtype=np.complex128, order="C")
    ci1_full = spin_op.contract_ss0(
        ci_full, norb, _unpack_nelec(nelec, spin))
    return extract_ksector_ci_from_full(
        ci1_full, norb, nelec, nkpts, target_k=target_k,
        link_index=link_index, spin=spin, contract_map=contract_map,
        kmom=kmom, kconserv=kconserv)


def contract_ss(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
                spin=None, contract_map=None, kmom=None, kconserv=None):
    """Apply ``S**2`` directly within a fixed packed momentum sector.

    Parameters
    ----------
    fcivec : array_like, shape (sector_size,)
        Packed CI coefficients.  They are converted to C-contiguous
        ``complex128`` storage before entering the C kernel.
    norb, nelec, nkpts, target_k, link_index, spin, contract_map, kmom,
    kconserv
        See :func:`embed_ksector_ci_to_full`.

    Returns
    -------
    ci1 : ndarray, dtype complex128
        Coefficients of ``S**2 |fcivec>`` with the same shape as the converted
        input array.

    Raises
    ------
    AssertionError
        If the number of input coefficients differs from the layout's sector
        size.

    Notes
    -----
    This computes the same operation as :func:`contract_ss_embedded` while
    avoiding the full alpha-by-beta determinant table.
    """
    neleca, nelecb = _unpack_nelec(nelec, spin)
    contract_map = _as_contract_map(
        norb, (neleca, nelecb), nkpts, target_k=target_k,
        link_index=link_index, spin=spin, contract_map=contract_map,
        kmom=kmom, kconserv=kconserv)
    fcivec = np.asarray(fcivec, dtype=np.complex128, order="C")
    assert fcivec.size == contract_map.sector_size

    ci1 = np.empty(fcivec.shape, dtype=np.complex128, order="C")
    link_indexa, link_indexb = contract_map.link_index
    _init_contract_ss_lib()
    with lib.with_omp_threads(lib.num_threads()):
        libpbcfci_k.FCIcontract_ss_k(
            fcivec.ctypes.data_as(ctypes.c_void_p),
            ci1.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(norb),
            ctypes.c_int(neleca),
            ctypes.c_int(nelecb),
            ctypes.c_int(nkpts),
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
        )
    return ci1


def spin_square(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
                spin=None, contract_map=None, kmom=None, kconserv=None,
                **kwargs):
    """Return the ``S**2`` matrix element and corresponding multiplicity.

    Parameters
    ----------
    fcivec : array_like, shape (sector_size,)
        Packed CI coefficients.  The vector is normally expected to be
        normalized; this function does not divide by its norm.
    norb, nelec, nkpts, target_k, link_index, spin, contract_map, kmom,
    kconserv
        See :func:`contract_ss`.
    **kwargs
        Optional ``verbose`` level used when reporting a non-negligible
        imaginary component of ``<fcivec|S**2|fcivec>``.

    Returns
    -------
    ss : numpy.float64
        Real part of ``<fcivec|S**2|fcivec>``.
    multiplicity : numpy.float64
        ``2*S + 1``, where ``S = sqrt(ss + 1/4) - 1/2``.

    Notes
    -----
    If the spin-squared matrix element has an imaginary component larger than
    ``1e-3``, a warning is emitted through the PySCF logger.  The real part is
    still used to compute both return values.
    """
    fcivec = np.asarray(fcivec, dtype=np.complex128, order="C")
    ci1 = contract_ss(
        fcivec, norb, nelec, nkpts, target_k=target_k,
        link_index=link_index, spin=spin, contract_map=contract_map,
        kmom=kmom, kconserv=kconserv)
    ss_complex = np.vdot(fcivec.ravel(), ci1.ravel())

    if abs(ss_complex.imag) > 1e-3:
        log = lib.logger.Logger(sys.stdout, kwargs.get("verbose", 0))
        log.warn("Spin square is not real. Imaginary part = %s",
                 ss_complex.imag)

    ss = ss_complex.real
    spin = np.sqrt(ss + 0.25) - 0.5
    return ss, 2 * spin + 1
