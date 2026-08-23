import ctypes
import unittest

import numpy as np
from numpy.testing import assert_array_equal
from pyscf.fci import cistring

from mrh.my_pyscf.pbc.fci import kcistrings


# Author: Bhavnesh Jangid

"""
Test momentum-aware CI link indices and determinant-sector mappings.

The k-point link index extends PySCF's single-excitation table with momentum
labels and transfers. Sector maps then group alpha and beta strings whose
combined momentum matches ``target_k``.
"""


def _string_momenta(strings, orb_k, nkpts):
    """
    Return determinant momenta from their occupied orbitals.
    Basically: sum(orb_k[occupied]) % nkpts for each determinant.
    """
    momenta = []
    for string in strings:
        occupied = [orb for orb in range(len(orb_k))
                    if int(string) & (1 << orb)]
        momenta.append(sum(orb_k[occupied]) % nkpts)
    return np.asarray(momenta, dtype=np.int32)

class KnownValues(unittest.TestCase):

    def test_c_link_index_matches_python_reference(self):
        """Check the compiled momentum-aware link-index builder.

        Its excitation addresses and signs are compared with PySCF, while its
        determinant, creation, destruction, and transfer momenta are built
        independently from the orbital momentum labels.
        """
        nkpts = 3
        norb = 6
        nocc = 2
        orb_k = np.repeat(np.arange(nkpts, dtype=np.int32), 2)
        strings = np.asarray(
            cistring.make_strings(range(norb), nocc), dtype=np.uint64)
        nlink = nocc * (norb - nocc) + nocc # number of single excitations per determinant

        link_c = np.empty((strings.size, nlink, 8), dtype=np.int32)
        kcistrings.libpbckcistring.FCIlinkstr_index_k(
            link_c.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(norb),
            ctypes.c_int(strings.size),
            ctypes.c_int(nocc),
            strings.ctypes.data_as(ctypes.c_void_p),
            orb_k.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nkpts),
        )

        link_ref = np.empty_like(link_c)
        link_ref[:, :, :4] = cistring.gen_linkstr_index(
            range(norb), nocc, strs=strings, tril=False)
        string_k = _string_momenta(strings, orb_k, nkpts)
        link_ref[:, :, 4] = string_k[:, None]
        link_ref[:, :, 5] = orb_k[link_ref[:, :, 0]]
        link_ref[:, :, 6] = orb_k[link_ref[:, :, 1]]
        link_ref[:, :, 7] = (
            link_ref[:, :, 5] - link_ref[:, :, 6]) % nkpts

        assert_array_equal(link_c, link_ref)
        assert_array_equal(
            kcistrings.gen_linkstr_index_k(
                range(norb), nocc, orb_k, nkpts, strs=strings),
            link_ref,
        )

    def test_python_sector_maps_match_brute_force(self):
        """Check momentum-sector maps and packed ``target_k`` blocks.

        Determinants are grouped by brute-force occupied-orbital momenta, and
        the resulting string IDs, inverse maps, block sizes, and offsets are
        compared with the Python kFCI sector-map builders.
        """
        nkpts = 3
        norb = 6
        target_k = 2
        
        orb_k = np.repeat(np.arange(nkpts, dtype=np.int32), 2)
        strings_a = np.asarray(
            cistring.make_strings(range(norb), 2), dtype=np.uint64)
        strings_b = np.asarray(
            cistring.make_strings(range(norb), 1), dtype=np.uint64)
        momentum_a = _string_momenta(strings_a, orb_k, nkpts)
        momentum_b = _string_momenta(strings_b, orb_k, nkpts)

        kmom = kcistrings.make_kpoint_momentum(nkpts)
        expected_add = np.fromfunction(
            lambda ka, kb: (ka + kb) % nkpts,
            (nkpts, nkpts), dtype=int).astype(np.int32)
        expected_sub = np.fromfunction(
            lambda ka, kb: (ka - kb) % nkpts,
            (nkpts, nkpts), dtype=int).astype(np.int32)
        assert_array_equal(kmom.kadd, expected_add)
        assert_array_equal(kmom.ksub, expected_sub)

        link_a = kcistrings.gen_linkstr_index_k(
            range(norb), 2, orb_k, nkpts, strs=strings_a)
        link_b = kcistrings.gen_linkstr_index_k(
            range(norb), 1, orb_k, nkpts, strs=strings_b)
        by_k_a, by_k_b, inverse_a, inverse_b = (
            kcistrings.gen_k_sector_maps(link_a, link_b, nkpts))

        for momentum, by_k, inverse in (
                (momentum_a, by_k_a, inverse_a),
                (momentum_b, by_k_b, inverse_b)):
            expected_inverse = -np.ones_like(inverse)
            for k in range(nkpts):
                expected_ids = np.flatnonzero(momentum == k).astype(np.int32)
                assert_array_equal(by_k[k], expected_ids)
                expected_inverse[k, expected_ids] = np.arange(
                    expected_ids.size, dtype=np.int32)
            assert_array_equal(inverse, expected_inverse)

        expected_blocks = []
        offset = 0
        for ka in range(nkpts):
            kb = (target_k - ka) % nkpts
            na = np.count_nonzero(momentum_a == ka)
            nb = np.count_nonzero(momentum_b == kb)
            size = na * nb
            if size:
                expected_blocks.append([ka, kb, na, nb, offset, size])
                offset += size

        blocks = kcistrings.gen_k_sector_linkstr_info(
            link_a, link_b, nkpts, target_k)
        assert_array_equal(blocks, np.asarray(expected_blocks, dtype=np.int32))


if __name__ == '__main__':
    print("Running unit tests for kcistring utilities...")
    unittest.main()
