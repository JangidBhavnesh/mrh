#!/usr/bin/env python
import numpy as np

from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc import scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.fci import direct_spin1_cplx

"""
Example for the kCASCI calculations in fixed total-momentum sectors.

In this example we use the periodic H2 chain on a given k mesh and solves its
neutral active-space problem separately for every total crystal momentum.
kCASCI keeps the active orbitals in the bloch MO basis and uses the
momentum-resolved kFCI solver.  Note: the ``ncas`` and ``nelecas`` are specified per
primitive cell, while the CI problem spans the complete k-point mesh; kCASCI
energies are reported per primitive cell.

kCASCI provides the spin operators, and RDM computation functionalities.
"""

intra_h = 0.74
inter_h = 1.5
vacuum = 17.5

cell = gto.Cell()
cell.a = np.diag([intra_h + inter_h, intra_h + inter_h, vacuum])
cell.atom = [
    ["H", (0.0, 0.0, vacuum / 2.0)],
    ["H", (intra_h, 0.0, vacuum / 2.0)],
]
cell.basis = "STO-6G"
cell.unit = "Angstrom"
cell.ke_cutoff = 100
cell.precision = 1e-10
cell.verbose = lib.logger.INFO
cell.build()

kmesh = [3, 1, 1]
kpts = cell.make_kpts(kmesh, wrap_around=True)
nkpts = len(kpts)

kmf = scf.KRHF(cell, kpts=kpts).density_fit(
    auxbasis="def2-svp-jkfit",
)
kmf.max_cycle = 1000
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

mo_coeff = np.asarray(kmf.mo_coeff)
ncas = 2
nelecas = 2


# Reference complex CASCI calculation without resolving total momentum.
# 
mc_ref = mcscf.CASCI(kmf, ncas, nelecas, ncore=0)
mc_ref.kmesh = kmesh
mc_ref.fcisolver = direct_spin1_cplx.FCISolver(cell)
mc_ref.fcisolver.verbose = 0
mc_ref.canonicalization = False
e_ref = mc_ref.kernel(mo_coeff)[0]

print()
print(f"k-RHF energy              : {kmf.e_tot.real:16.12f}")
print(f"full CASCI energy         : {e_ref.real:16.12f}")
print(f"active-space dimensions   : {nkpts * ncas} orbitals, "
      f"{nkpts * nelecas} electrons")
print()

e_kCASCI = []
for target_k in range(nkpts):
    kmc = mcscf.KCASCI(
        kmf, ncas, nelecas, ncore=0, target_k=target_k,
    )
    kmc.kmesh = kmesh
    kmc.fcisolver.conv_tol = 1e-10
    kmc.canonicalization = False
    e_tot = kmc.kernel(mo_coeff)[0]
    e_kCASCI.append(e_tot.real)

    # Demonstration of computing the spin-square and RDM comp.
    ncastot = nkpts * ncas
    nelecastot = (
        nkpts * kmc.nelecas[0],
        nkpts * kmc.nelecas[1],
    )
    spin_square, multiplicity = kmc.fcisolver.spin_square(
        kmc.ci, ncastot, nelecastot,
    )
    dm1 = kmc.make_rdm1()
   
print()
for k in range(nkpts):
    print(f"kpts {k}: kCASCI energy = {e_kCASCI[k]:16.12f}")

# Also note the difference between the momentum-resolved kCASCI energies and the cplx CASCI energy.
print(f"kCASCI - cplxCASCI   =  {(e_kCASCI[0] - e_ref).real:16.12e}")
print()
