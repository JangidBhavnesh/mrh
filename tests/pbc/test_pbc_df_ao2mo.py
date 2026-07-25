#!/bin/bash
import unittest
import numpy as np

from pyscf.pbc import gto as pgto, scf

from mrh.my_pyscf.pbc.df.df_ao2mo import ao2mo_7d

def get_cell():
    cell = pgto.Cell()
    cell.atom = """
    He  0.000000  0.000000  0.000000
    He  0.891700  0.891700  0.891700"""
    cell.a = """
    0.000000  1.783400  1.783400
    1.783400  0.000000  1.783400
    1.783400  1.783400  0.000000
    """
    cell.basis = "631G"
    cell.unit = "B"
    cell.verbose = 0
    cell.build()
    return cell

class KnownValues(unittest.TestCase):

    def test_ao2mo_1D(self):
        cell = get_cell()
        kmesh = [2, 1, 1]
        kpts = cell.make_kpts(kmesh)
        nkpts = len(kpts)

        # Initializing the mean-field object for some initial orbitals
        kmf = scf.KRHF(cell, kpts, exxdiv=None).density_fit(auxbasis="weigend")
        kmf.max_cycle = 0 
        kmf.kernel()
        
        mo_coeff_kpts = np.asarray([kmf.mo_coeff[k][:, :2] for k in range(nkpts)])

        eri_ref = kmf.with_df.ao2mo_7d(mo_coeff_kpts, kpts=kpts)
        eri_new = ao2mo_7d(kmf.with_df, mo_coeff_kpts, kpts=kpts)
        
        np.testing.assert_allclose(eri_ref, eri_new, atol=1e-9, rtol=1e-9)

        del kmf, mo_coeff_kpts, eri_ref, eri_new

if __name__ == "__main__":
    # print("Full Tests for Optimized version of ao2mo_7d")
    unittest.main()