#!/usr/bin/env python

"""Run neutral KCASCI in each total-momentum sector.

KCASCI keeps the active orbitals in the k-point basis and uses the
momentum-resolved kFCI solver.  ``ncas`` and ``nelecas`` are specified per
primitive cell; the CI problem spans the complete k-point mesh.  Energies
reported by the driver are normalized per primitive cell.
"""

import numpy as np

from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc import scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.fci import direct_spin1_cplx


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
cell.max_memory = 100000
cell.ke_cutoff = 100
cell.precision = 1e-10
cell.verbose = lib.logger.INFO
cell.build()

kmesh = [2, 1, 1]
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

# A conventional periodic CASCI calculation solves the complete active-space
# problem without resolving its total momentum.  It provides a useful check:
# for this system, its ground state belongs to target_k=0.
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

for target_k in range(nkpts):
    kmc = mcscf.KCASCI(
        kmf, ncas, nelecas, ncore=0, target_k=target_k,
    )
    kmc.kmesh = kmesh
    kmc.fcisolver.conv_tol = 1e-10
    kmc.canonicalization = False
    e_tot = kmc.kernel(mo_coeff)[0]

    ncastot = nkpts * ncas
    nelecastot = (
        nkpts * kmc.nelecas[0],
        nkpts * kmc.nelecas[1],
    )
    spin_square, multiplicity = kmc.fcisolver.spin_square(
        kmc.ci, ncastot, nelecastot,
    )
    dm1 = kmc.make_rdm1()
    overlap = np.asarray(kmf.get_ovlp())
    electron_count = np.einsum(
        "kij,kji->", dm1, overlap,
    ).real / nkpts

    print(f"target_k = {target_k}")
    print(f"  KCASCI energy           : {e_tot.real:16.12f}")
    print(f"  <S^2>                   : {spin_square.real:16.12f}")
    print(f"  2S+1                    : {multiplicity.real:16.12f}")
    print(f"  AO 1-RDM electron count : {electron_count:16.12f}")
    if target_k == 0:
        print(f"  KCASCI - full CASCI     : {(e_tot - e_ref).real:16.12e}")
    print()

# Fock construction and canonicalization use the density of the most recent
# KCASCI solution.  The canonicalizer preserves the active orbitals and
# diagonalizes only unfrozen core and virtual subspaces.
fock = kmc.get_fock()
mo_canonical, _, mo_energy = kmc.canonicalize_()
print(f"Fock matrix shape         : {fock.shape}")
print(f"Canonical orbitals shape  : {mo_canonical.shape}")
print(f"Orbital energies shape    : {np.asarray(mo_energy).shape}")
