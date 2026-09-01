#!/usr/bin/env python
import numpy as np

from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcpdft

# Author: Bhavnesh Jangid

"""
Example to run the  k-MC-PDFT in one total-momentum sector.
"""

cell = gto.Cell()
cell.a = np.diag([2.24, 2.24, 12.0])
cell.atom = [
    ["H", (0.0, 0.0, 6.0)],
    ["H", (1.1, 0.0, 6.0)],
]
cell.basis = "CC-PVDZ"
cell.unit = "Angstrom"
cell.precision = 1e-9
cell.verbose = lib.logger.INFO
cell.build()

kmesh = [3, 1, 1]
kpts = cell.make_kpts(kmesh, wrap_around=True)
kmf = scf.KRHF(cell, kpts=kpts).density_fit()
kmf.exxdiv = None
kmf.kernel()

target_k = 1
kmc = mcpdft.KCASCI(kmf, "tPBE", 2, 2, target_k=target_k)
kmc.kmesh = kmesh
kmc.kernel(np.asarray(kmf.mo_coeff))

print(f"For the target_k={target_k}")
print(f"kCASCI energy    : {kmc.e_mcscf.real:16.12f}")
print(f"k-MC-PDFT energy : {kmc.e_tot.real:16.12f}")
