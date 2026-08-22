"""Reduced-density-matrix and spin helpers for momentum-sector FCI."""

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
    kfci_helper,
    rdm_helper,
    spin_op,
)


_kci_lib_initialized = False
_contract_ss_lib_initialized = False
libpbcfci_k = None


def _init_kci_lib():
    """Configure the C helpers that map between sector and full CI arrays."""
    global _kci_lib_initialized
    if _kci_lib_initialized:
        return

    rdm_helper.libpbcrdm.FCIkci_sector_to_full_cplx.argtypes = [
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
    rdm_helper.libpbcrdm.FCIkci_sector_to_full_cplx.restype = None
    rdm_helper.libpbcrdm.FCIkci_full_to_sector_cplx.argtypes = [
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
    rdm_helper.libpbcrdm.FCIkci_full_to_sector_cplx.restype = None
    _kci_lib_initialized = True


def _init_contract_ss_lib():
    """Load and configure the C spin-squared contraction."""
    global _contract_ss_lib_initialized, libpbcfci_k
    if _contract_ss_lib_initialized:
        return

    libpbcfci_k = load_library("libpbc_fci_contract_k")
    libpbcfci_k.FCIcontract_ss_k.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
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
    libpbcfci_k.FCIcontract_ss_k.restype = None
    _contract_ss_lib_initialized = True


def _unpack_k(norb, nelec, nkpts, link_index=None, spin=None, kmom=None,
              kconserv=None):
    """Generate or validate momentum-aware link indices."""
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
    """Return a compatible layout map, constructing it when necessary."""
    if contract_map is not None:
        return contract_map
    link_index = _unpack_k(
        norb, nelec, nkpts, link_index=link_index, spin=spin, kmom=kmom,
        kconserv=kconserv)
    return kfci_helper.KFCILayoutMap.build(
        norb, nelec, nkpts, target_k, link_index=link_index, kmom=kmom,
        kconserv=kconserv)


def embed_ksector_ci_to_full(fcivec, norb, nelec, nkpts, target_k=0,
                             link_index=None, spin=None, contract_map=None,
                             kmom=None, kconserv=None):
    """Embed a momentum-sector vector in the full spin-string CI table."""
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
    """Extract one momentum sector from a full spin-string CI table."""
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
    """Build reference 1-RDMs through an embedded full-CI vector."""
    ci_full = embed_ksector_ci_to_full(
        fcivec, norb, nelec, nkpts, target_k=target_k,
        link_index=link_index, spin=spin, kmom=kmom, kconserv=kconserv)
    return direct_spin1_cplx.make_rdm1s(
        ci_full, norb, _unpack_nelec(nelec, spin), link_index=None)


def make_rdm1_ref(fcivec, norb, nelec, nkpts, target_k=0,
                  link_index=None, spin=None, kmom=None, kconserv=None):
    """Build a reference spin-summed 1-RDM through full-CI embedding."""
    rdm1a, rdm1b = make_rdm1s_ref(
        fcivec, norb, nelec, nkpts, target_k=target_k,
        link_index=link_index, spin=spin, kmom=kmom, kconserv=kconserv)
    return (rdm1a + rdm1b).conj().T


def make_rdm12s_ref(fcivec, norb, nelec, nkpts, target_k=0,
                    link_index=None, reorder=True, spin=None, kmom=None,
                    kconserv=None):
    """Build reference 1-/2-RDMs through an embedded full-CI vector."""
    ci_full = embed_ksector_ci_to_full(
        fcivec, norb, nelec, nkpts, target_k=target_k,
        link_index=link_index, spin=spin, kmom=kmom, kconserv=kconserv)
    return direct_spin1_cplx.make_rdm12s(
        ci_full, norb, _unpack_nelec(nelec, spin), link_index=None,
        reorder=reorder)


def make_rdm12_ref(fcivec, norb, nelec, nkpts, target_k=0,
                   link_index=None, reorder=True, spin=None, kmom=None,
                   kconserv=None):
    """Build reference spin-summed RDMs through full-CI embedding."""
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = make_rdm12s_ref(
        fcivec, norb, nelec, nkpts, target_k=target_k,
        link_index=link_index, reorder=reorder, spin=spin, kmom=kmom,
        kconserv=kconserv)
    dm1 = dm1a + dm1b
    dm2 = dm2aa + dm2bb + dm2ab + dm2ab.transpose(2, 3, 0, 1)
    return dm1.conj().T, dm2


# Keep the public API working until the direct momentum-sector kernels are
# registered below.  These aliases are replaced by direct implementations in
# the next implementation step; the ``*_ref`` names remain for benchmarks.
make_rdm1s = make_rdm1s_ref
make_rdm1 = make_rdm1_ref
make_rdm12s = make_rdm12s_ref
make_rdm12 = make_rdm12_ref


def contract_ss_embedded(fcivec, norb, nelec, nkpts, target_k=0,
                         link_index=None, spin=None, contract_map=None,
                         kmom=None, kconserv=None):
    """Apply spin-squared through a full-space embedded CI vector."""
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
    """Apply spin-squared directly within a fixed momentum sector."""
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
    """Return the spin-squared expectation and spin multiplicity."""
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
