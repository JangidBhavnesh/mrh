import numpy as np
import ctypes

from pyscf.fci import cistring
from pyscf.fci.addons import _unpack_nelec

from mrh.my_pyscf.pbc.fci import direct_spin1_cplx, kcistrings, rdm_helper, spin_op

# Author: Bhavnesh Jangid

'''
Python RDM helpers for k-FCI vectors.

The k-FCI CI vector is stored only in one total momentum sector. For now, I am
embedding that sector vector into the full determinant table with a C helper
and then using the existing complex-FCI C RDM code. This keeps the tensor
convention identical to direct_spin1_cplx.
'''

_kci_lib_initialized = False


def _init_kci_lib():
    global _kci_lib_initialized
    if _kci_lib_initialized:
        return

    lib = rdm_helper.libpbcrdm
    lib.FCIkci_sector_to_full_cplx.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.FCIkci_sector_to_full_cplx.restype = None
    lib.FCIkci_full_to_sector_cplx.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    lib.FCIkci_full_to_sector_cplx.restype = None
    _kci_lib_initialized = True


def _as_contract_map(norb, nelec, nkpts, target_k=0, link_index=None,
                     spin=None, contract_map=None):
    if contract_map is not None:
        return contract_map

    link_index = _unpack_k(norb, nelec, nkpts, link_index=link_index,
                           spin=spin)
    return kcistrings.make_kfci_contract_map(
        norb, nelec, nkpts, target_k, link_index=link_index,
        build_pair_tables=False)

def _unpack_k(norb, nelec, nkpts, link_index=None, spin=None):
    '''
    Generate or unpack the k-aware link indices.
    '''
    assert norb % nkpts == 0
    if link_index is None:
        neleca, nelecb = _unpack_nelec(nelec, spin)
        norb_k = norb // nkpts
        orb_k = (np.arange(norb, dtype=np.int32) // norb_k).astype(np.int32)

        if spin == 0 and neleca == nelecb:
            link_indexa = link_indexb = kcistrings.gen_linkstr_index_k(
                range(norb), neleca, orb_k, nkpts)
        else:
            link_indexa = kcistrings.gen_linkstr_index_k(
                range(norb), neleca, orb_k, nkpts)
            link_indexb = kcistrings.gen_linkstr_index_k(
                range(norb), nelecb, orb_k, nkpts)
        return link_indexa, link_indexb

    assert link_index[0].shape[2] == link_index[1].shape[2] == 8
    return link_index

def _embed_ksector_ci_to_full_python(fcivec, norb, nelec, nkpts, target_k=0,
                                     link_index=None, spin=None):
    '''
    Embed a k-FCI sector vector into the full spin-string CI matrix.
    args:
        fcivec : ndarray, shape (sector_size,)
            CI vector stored in one total momentum sector.
        norb : int
            Total number of orbitals across all k-points.
        nelec : tuple of 2 ints
            Number of alpha and beta electrons.
        nkpts : int
            Number of k-points.
        target_k : int, optional
            Total momentum sector.
        link_index : tuple of 2 ndarrays or None
            k-aware link tables.
        spin : int or None
            Spin value passed to _unpack_nelec.
    returns:
        ci_full : ndarray, shape (nstra, nstrb)
            Full CI matrix with zeros outside the target momentum sector.
    '''
    neleca, nelecb = _unpack_nelec(nelec, spin)
    target_k = int(target_k) % nkpts

    link_indexa, link_indexb = _unpack_k(norb, (neleca, nelecb), nkpts,
                                         link_index=link_index, spin=spin)
    straid_k, strbid_k = kcistrings.gen_k_sector_maps(
        link_indexa, link_indexb, nkpts)[:2]
    blocks = kcistrings.gen_k_sector_linkstr_info(
        link_indexa, link_indexb, nkpts, target_k)

    sector_size = int(blocks[:, 5].sum()) if blocks.size else 0
    assert fcivec.size == sector_size, (fcivec.size, sector_size)

    nstra = cistring.num_strings(norb, neleca)
    nstrb = cistring.num_strings(norb, nelecb)
    ci_full = np.zeros((nstra, nstrb), dtype=np.asarray(fcivec).dtype)

    for blk in blocks:
        ka, kb, nstra_k, nstrb_k, offset, size = map(int, blk)
        ci_blk = fcivec[offset:offset + size].reshape(nstra_k, nstrb_k)

        astrs = straid_k[ka]
        bstrs = strbid_k[kb]

        ci_full[np.ix_(astrs, bstrs)] = ci_blk

    return ci_full


def embed_ksector_ci_to_full(fcivec, norb, nelec, nkpts, target_k=0,
                             link_index=None, spin=None, contract_map=None):
    '''
    Embed a k-FCI sector vector into the full spin-string CI matrix using the
    backend C mapping helper.
    '''
    neleca, nelecb = _unpack_nelec(nelec, spin)
    contract_map = _as_contract_map(
        norb, (neleca, nelecb), nkpts, target_k=target_k,
        link_index=link_index, spin=spin, contract_map=contract_map)
    assert fcivec.size == contract_map.sector_size, (
        fcivec.size, contract_map.sector_size)

    nstra = cistring.num_strings(norb, neleca)
    nstrb = cistring.num_strings(norb, nelecb)
    fcivec = np.asarray(fcivec, dtype=np.complex128, order="C")
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

def _extract_ksector_ci_from_full_python(ci_full, norb, nelec, nkpts,
                                         target_k=0, link_index=None,
                                         spin=None):
    '''
    Extract a k-FCI sector vector from the full spin-string CI matrix.
    args:
        ci_full : ndarray, shape (nstra, nstrb)
            Full CI matrix.
        norb : int
            Total number of orbitals across all k-points.
        nelec : tuple of 2 ints
            Number of alpha and beta electrons.
        nkpts : int
            Number of k-points.
        target_k : int, optional
            Total momentum sector.
        link_index : tuple of 2 ndarrays or None
            k-aware link tables.
        spin : int or None
            Spin value passed to _unpack_nelec.
    returns:
        fcivec : ndarray, shape (sector_size,)
            CI vector in the target momentum sector.
    '''
    neleca, nelecb = _unpack_nelec(nelec, spin)
    target_k = int(target_k) % nkpts

    link_indexa, link_indexb = _unpack_k(norb, (neleca, nelecb), nkpts,
                                         link_index=link_index, spin=spin)
    straid_k, strbid_k = kcistrings.gen_k_sector_maps(
        link_indexa, link_indexb, nkpts)[:2]
    blocks = kcistrings.gen_k_sector_linkstr_info(
        link_indexa, link_indexb, nkpts, target_k)

    ci_full = np.asarray(ci_full)
    sector_size = int(blocks[:, 5].sum()) if blocks.size else 0
    fcivec = np.zeros(sector_size, dtype=ci_full.dtype)

    for blk in blocks:
        ka, kb, nstra_k, nstrb_k, offset, size = map(int, blk)

        astrs = straid_k[ka]
        bstrs = strbid_k[kb]

        ci_blk = ci_full[np.ix_(astrs, bstrs)]
        fcivec[offset:offset + size] = ci_blk.reshape(-1)

    return fcivec


def extract_ksector_ci_from_full(ci_full, norb, nelec, nkpts, target_k=0,
                                 link_index=None, spin=None,
                                 contract_map=None):
    '''
    Extract a k-FCI sector vector from the full spin-string CI matrix using the
    backend C mapping helper.
    '''
    neleca, nelecb = _unpack_nelec(nelec, spin)
    contract_map = _as_contract_map(
        norb, (neleca, nelecb), nkpts, target_k=target_k,
        link_index=link_index, spin=spin, contract_map=contract_map)

    nstrb = cistring.num_strings(norb, nelecb)
    ci_full = np.asarray(ci_full, dtype=np.complex128, order="C")
    fcivec = np.empty(contract_map.sector_size, dtype=np.complex128, order="C")

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

def make_rdm1s(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
               spin=None):
    '''
    Spin-separated 1-RDMs for a k-FCI vector.
    '''
    ci_full = embed_ksector_ci_to_full(fcivec, norb, nelec, nkpts,
                                       target_k=target_k,
                                       link_index=link_index, spin=spin)
    return direct_spin1_cplx.make_rdm1s(ci_full, norb,
                                        _unpack_nelec(nelec, spin),
                                        link_index=None)

def make_rdm1(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
              spin=None):
    '''
    Spin-summed 1-RDM for a k-FCI vector.
    '''
    rdm1a, rdm1b = make_rdm1s(fcivec, norb, nelec, nkpts,
                              target_k=target_k, link_index=link_index,
                              spin=spin)
    rdm1 = rdm1a + rdm1b
    return rdm1.conj().T

def make_rdm12s(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
                reorder=True, spin=None):
    '''
    Spin-separated 1-RDMs and 2-RDMs for a k-FCI vector.
    returns:
        (dm1a, dm1b), (dm2aa, dm2ab, dm2bb)
    '''
    ci_full = embed_ksector_ci_to_full(fcivec, norb, nelec, nkpts,
                                       target_k=target_k,
                                       link_index=link_index, spin=spin)
    return direct_spin1_cplx.make_rdm12s(ci_full, norb,
                                         _unpack_nelec(nelec, spin),
                                         link_index=None, reorder=reorder)

def make_rdm12(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
               reorder=True, spin=None):
    '''
    Spin-summed 1-RDM and 2-RDM for a k-FCI vector.
    '''
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = make_rdm12s(
        fcivec, norb, nelec, nkpts, target_k=target_k,
        link_index=link_index, reorder=reorder, spin=spin)
    rdm1 = dm1a + dm1b
    rdm2 = dm2aa + dm2bb + dm2ab + dm2ab.transpose(2, 3, 0, 1)
    return rdm1.conj().T, rdm2

def contract_ss(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
                spin=None):
    '''
    Apply S^2 to a k-FCI vector in a fixed total momentum sector.
    The S^2 operator does not change the spatial total momentum sector, so the
    full-space result is extracted back to the same k sector.
    '''
    ci_full = embed_ksector_ci_to_full(fcivec, norb, nelec, nkpts,
                                       target_k=target_k,
                                       link_index=link_index, spin=spin)
    ci_full = np.asarray(ci_full, dtype=np.complex128, order="C")
    ci1_full = spin_op.contract_ss0(ci_full, norb, _unpack_nelec(nelec, spin))
    return extract_ksector_ci_from_full(ci1_full, norb, nelec, nkpts,
                                        target_k=target_k,
                                        link_index=link_index, spin=spin)

def spin_square(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
                spin=None, **kwargs):
    '''
    Spin square for a k-FCI vector in a fixed total momentum sector.
    '''
    ci_full = embed_ksector_ci_to_full(fcivec, norb, nelec, nkpts,
                                       target_k=target_k,
                                       link_index=link_index, spin=spin)
    ci_full = np.asarray(ci_full, dtype=np.complex128, order="C")
    return spin_op.spin_square0(ci_full, norb, _unpack_nelec(nelec, spin),
                                **kwargs)
