#!/usr/bin/env python

"""Run momentum-resolved k-MC-PDFT in selected total-momentum sectors.

The established ``mcpdft.KCASCI`` API continues to use the conventional
Wannier-basis periodic CASCI implementation by default.  Set
``momentum_resolved=True`` to use ``PBCKCASCI`` and choose a total-momentum
sector with ``target_k``.
"""

import numpy as np

from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf, mcpdft


cell = gto.Cell()
cell.a = np.diag([2.24, 2.24, 12.0])
cell.atom = [
    ["H", (0.0, 0.0, 6.0)],
    ["H", (0.74, 0.0, 6.0)],
]
cell.basis = "STO-6G"
cell.unit = "Angstrom"
cell.precision = 1e-9
cell.verbose = lib.logger.INFO
cell.build()

kmesh = [2, 1, 1]
kpts = cell.make_kpts(kmesh, wrap_around=True)
kmf = scf.KRHF(cell, kpts=kpts).density_fit()
kmf.exxdiv = None
kmf.conv_tol = 1e-9
kmf.kernel()

mo_coeff = np.asarray(kmf.mo_coeff)
ncas = 2
nelecas = 2
grids_attr = {"level": 2}

# The default API remains the conventional periodic CASCI-PDFT route.
reference = mcpdft.KCASCI(
    kmf, "tPBE", ncas, nelecas, ncore=0,
    grids_attr=grids_attr,
)
reference.kpts = kpts
reference.kmesh = kmesh
reference.canonicalization = False
reference.kernel(mo_coeff)

print()
print(f"k-RHF energy                 : {kmf.e_tot.real:16.12f}")
print(f"conventional k-MC-PDFT       : {reference.e_tot.real:16.12f}")
print()

# Each calculation below diagonalizes the active-space Hamiltonian only in
# the requested total-momentum sector.  target_k=0 reproduces the conventional
# ground-state result for this example, while target_k=1 accesses the other
# sector directly.
for target_k in range(len(kpts)):
    kmc = mcpdft.KCASCI(
        kmf, "tPBE", ncas, nelecas, ncore=0,
        momentum_resolved=True,
        target_k=target_k,
        grids_attr=grids_attr,
    )
    kmc.kmesh = kmesh
    kmc.canonicalization = False
    kmc.kernel(mo_coeff)

    print(f"target_k={target_k}")
    print(f"  KCASCI energy              : {kmc.e_mcscf.real:16.12f}")
    print(f"  on-top energy              : {kmc.e_ot.real:16.12f}")
    print(f"  k-MC-PDFT energy           : {kmc.e_tot.real:16.12f}")
    if target_k == 0:
        print(
            "  difference from reference : "
            f"{(kmc.e_tot - reference.e_tot).real:16.12e}"
        )
    print()

# An already converged PBCKCASCI object can also be promoted to MC-PDFT.  Its
# target_k and CI vector are retained; a conflicting target_k is rejected.
kcas = mcscf.KCASCI(
    kmf, ncas, nelecas, ncore=0, target_k=1,
)
kcas.kmesh = kmesh
kcas.canonicalization = False
kcas.kernel(mo_coeff)

pdft_from_kcas = mcpdft.KCASCI(
    kcas, "tPBE", ncas, nelecas, ncore=0,
    momentum_resolved=True,
    grids_attr=grids_attr,
)
pdft_from_kcas.compute_pdft_energy_()
print("Existing PBCKCASCI route")
print(f"  retained target_k          : {pdft_from_kcas.target_k}")
print(f"  k-MC-PDFT energy           : {pdft_from_kcas.e_tot.real:16.12f}")

# With one active orbital and two active electrons per cell, the complete
# k-mesh active space contains only one determinant.  Its target_k=0 KCASCI
# wavefunction energy reduces to the k-RHF determinant energy.
single = mcpdft.KCASCI(
    kmf, "tPBE", 1, 2, ncore=0,
    momentum_resolved=True,
    target_k=0,
    grids_attr=grids_attr,
)
single.kmesh = kmesh
single.canonicalization = False
single.kernel(mo_coeff)
print()
print("Single-determinant limit")
print(f"  CI dimension               : {np.size(single.ci)}")
print(f"  KCASCI - k-RHF             : "
      f"{(single.e_mcscf - kmf.e_tot).real:16.12e}")
