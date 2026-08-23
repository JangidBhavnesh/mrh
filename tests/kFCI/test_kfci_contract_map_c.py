#!/usr/bin/env python

"""Verify the low-level C contract-map builder against the Python reference.

A contract map organizes momentum-sector CI addresses, fermionic signs, and
ERI indices used by the two-electron contraction and Hamiltonian-diagonal
kernels. It combines and reorganizes link indices into alpha-beta,
alpha-alpha, and beta-beta excitation pairs restricted to ``target_k``.
"""

import unittest
import numpy as np

from mrh.my_pyscf.pbc.fci.kfci_contract_map import (
    _unpack_contract_link_index,
    make_kfci_contract_map,
)

# Author: Bhavnesh Jangid


class KnownValues(unittest.TestCase):

    def test_c_contract_map_matches_python_builder(self):
        """Compare every compiled contract-map field with Python output.

        The comparison covers several orbital and electron configurations in
        every available total-momentum sector.
        """
        # nkpts, ncas, nelec
        test_cases = [
            (1, 4, (2, 2)),
            (2, 3, (2, 2)),
            (2, 3, (2, 1)),
            (3, 2, (2, 2)),
        ]
        fields = [
            "ab_group_tab",
            "ab_group_offsets",
            "ab_src_addr",
            "ab_dst_addr",
            "ab_sign",
            "ab_eri_idx_ab",
            "ab_eri_idx_ba",
            "aa_group_tab",
            "aa_group_offsets",
            "aa_src_addr",
            "aa_dst_addr",
            "aa_sign",
            "aa_eri_idx",
            "bb_group_tab",
            "bb_group_offsets",
            "bb_src_addr",
            "bb_dst_addr",
            "bb_sign",
            "bb_eri_idx",
        ]

        for nkpts, ncas, nelec in test_cases:
            norb = nkpts * ncas
            link_index = _unpack_contract_link_index(
                norb, nelec, None, nkpts
            )

            for target_k in range(nkpts):
                with self.subTest(
                    nkpts=nkpts, ncas=ncas, nelec=nelec,
                    target_k=target_k
                ):
                    map_c = make_kfci_contract_map(
                        norb, nelec, nkpts, target_k,
                        link_index=link_index,
                        use_c_contract_map=True,
                    )
                    map_py = make_kfci_contract_map(
                        norb, nelec, nkpts, target_k,
                        link_index=link_index,
                        build_pair_tables=True,
                        use_c_contract_map=False,
                    )

                    self.assertFalse(map_c.has_pair_tables)
                    self.assertTrue(map_py.has_pair_tables)
                    np.testing.assert_array_equal(map_c.blocks, map_py.blocks)

                    for field in fields:
                        np.testing.assert_array_equal(
                            getattr(map_c, field), getattr(map_py, field),
                            err_msg=field,
                        )

if __name__ == "__main__":
    print("Running unit tests for contract-map builder...")
    unittest.main()
