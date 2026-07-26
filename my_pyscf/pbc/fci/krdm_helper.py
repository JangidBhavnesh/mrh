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
                     spin=None, contract_map=None, kmom=None,
                     kconserv=None):
    if contract_map is not None:
        return contract_map

    link_index = _unpack_k(norb, nelec, nkpts, link_index=link_index,
                           spin=spin, kmom=kmom, kconserv=kconserv)
    return kcistrings.make_kfci_contract_map(
        norb, nelec, nkpts, target_k, link_index=link_index,
        build_pair_tables=False, kmom=kmom, kconserv=kconserv)

def _unpack_k(norb, nelec, nkpts, link_index=None, spin=None, kmom=None,
              kconserv=None):
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

    assert link_index[0].shape[2] == link_index[1].shape[2] == 8
    return link_index

def embed_ksector_ci_to_full(fcivec, norb, nelec, nkpts, target_k=0,
                             link_index=None, spin=None, contract_map=None,
                             kmom=None, kconserv=None):
    '''
    Embed a k-FCI sector vector into the full spin-string CI matrix using the
    backend C mapping helper.
    '''
    neleca, nelecb = _unpack_nelec(nelec, spin)
    contract_map = _as_contract_map(
        norb, (neleca, nelecb), nkpts, target_k=target_k,
        link_index=link_index, spin=spin, contract_map=contract_map,
        kmom=kmom, kconserv=kconserv)
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

def extract_ksector_ci_from_full(ci_full, norb, nelec, nkpts, target_k=0,
                                 link_index=None, spin=None,
                                 contract_map=None, kmom=None,
                                 kconserv=None):
    '''
    Extract a k-FCI sector vector from the full spin-string CI matrix using the
    backend C mapping helper.
    '''
    neleca, nelecb = _unpack_nelec(nelec, spin)
    contract_map = _as_contract_map(
        norb, (neleca, nelecb), nkpts, target_k=target_k,
        link_index=link_index, spin=spin, contract_map=contract_map,
        kmom=kmom, kconserv=kconserv)

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
               spin=None, kmom=None, kconserv=None):
    '''
    Spin-separated 1-RDMs for a k-FCI vector.
    '''
    ci_full = embed_ksector_ci_to_full(fcivec, norb, nelec, nkpts,
                                       target_k=target_k,
                                       link_index=link_index, spin=spin,
                                       kmom=kmom, kconserv=kconserv)
    return direct_spin1_cplx.make_rdm1s(ci_full, norb,
                                        _unpack_nelec(nelec, spin),
                                        link_index=None)

def make_rdm1(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
              spin=None, kmom=None, kconserv=None):
    '''
    Spin-summed 1-RDM for a k-FCI vector.
    '''
    rdm1a, rdm1b = make_rdm1s(fcivec, norb, nelec, nkpts,
                              target_k=target_k, link_index=link_index,
                              spin=spin, kmom=kmom, kconserv=kconserv)
    rdm1 = rdm1a + rdm1b
    return rdm1.conj().T

def make_rdm12s(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
                reorder=True, spin=None, kmom=None, kconserv=None):
    '''
    Spin-separated 1-RDMs and 2-RDMs for a k-FCI vector.
    returns:
        (dm1a, dm1b), (dm2aa, dm2ab, dm2bb)
    '''
    ci_full = embed_ksector_ci_to_full(fcivec, norb, nelec, nkpts,
                                       target_k=target_k,
                                       link_index=link_index, spin=spin,
                                       kmom=kmom, kconserv=kconserv)
    return direct_spin1_cplx.make_rdm12s(ci_full, norb,
                                         _unpack_nelec(nelec, spin),
                                         link_index=None, reorder=reorder)

def make_rdm12(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
               reorder=True, spin=None, kmom=None, kconserv=None):
    '''
    Spin-summed 1-RDM and 2-RDM for a k-FCI vector.
    '''
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = make_rdm12s(
        fcivec, norb, nelec, nkpts, target_k=target_k,
        link_index=link_index, reorder=reorder, spin=spin, kmom=kmom,
        kconserv=kconserv)
    rdm1 = dm1a + dm1b
    rdm2 = dm2aa + dm2bb + dm2ab + dm2ab.transpose(2, 3, 0, 1)
    return rdm1.conj().T, rdm2

def contract_ss(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
                spin=None, kmom=None, kconserv=None):
    '''
    Apply S^2 to a k-FCI vector in a fixed total momentum sector.
    The S^2 operator does not change the spatial total momentum sector, so the
    full-space result is extracted back to the same k sector.
    '''
    ci_full = embed_ksector_ci_to_full(fcivec, norb, nelec, nkpts,
                                       target_k=target_k,
                                       link_index=link_index, spin=spin,
                                       kmom=kmom, kconserv=kconserv)
    ci_full = np.asarray(ci_full, dtype=np.complex128, order="C")
    ci1_full = spin_op.contract_ss0(ci_full, norb, _unpack_nelec(nelec, spin))
    return extract_ksector_ci_from_full(ci1_full, norb, nelec, nkpts,
                                        target_k=target_k,
                                        link_index=link_index, spin=spin,
                                        kmom=kmom, kconserv=kconserv)

def spin_square(fcivec, norb, nelec, nkpts, target_k=0, link_index=None,
                spin=None, kmom=None, kconserv=None, **kwargs):
    '''
    Spin square for a k-FCI vector in a fixed total momentum sector.
    '''
    ci_full = embed_ksector_ci_to_full(fcivec, norb, nelec, nkpts,
                                       target_k=target_k,
                                       link_index=link_index, spin=spin,
                                       kmom=kmom, kconserv=kconserv)
    ci_full = np.asarray(ci_full, dtype=np.complex128, order="C")
    return spin_op.spin_square0(ci_full, norb, _unpack_nelec(nelec, spin),
                                **kwargs)
