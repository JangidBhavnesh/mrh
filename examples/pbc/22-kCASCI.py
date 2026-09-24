#!/usr/bin/env python

"""Run neutral and charged KCASCI in total-momentum sectors.

KCASCI keeps the active orbitals in the k-point basis and uses the
momentum-resolved kFCI solver.  ``ncas`` and ``nelecas`` are specified per
primitive cell; the CI problem spans the complete k-point mesh.  Energies
reported by the driver are normalized per primitive cell.

For charged calculations, omitting ``target_k`` sweeps every N-1 or N+1
total-momentum sector.  The band-energy helper converts those total momenta
to the physical momentum of the removed or added electron.
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

e_neutral = None
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
        e_neutral = e_tot
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

# A positive charge removes one electron from the complete k-mesh active
# space; a negative charge adds one.  With target_k omitted, the driver solves
# every charged total-momentum sector using the same transformed integrals.
hole = mcscf.KCASCI(
    kmf, ncas, nelecas, ncore=0, charge=1,
)
particle = mcscf.KCASCI(
    kmf, ncas, nelecas, ncore=0, charge=-1,
)
for charged_mc in (hole, particle):
    charged_mc.kmesh = kmesh
    charged_mc.fcisolver.conv_tol = 1e-10
    charged_mc.kernel(mo_coeff)

hole_bands = hole.band_energies(e_neutral)
particle_bands = particle.band_energies(e_neutral)
hole_by_k = {
    band["momentum_index"]: band for band in hole_bands
}
particle_by_k = {
    band["momentum_index"]: band for band in particle_bands
}

print()
print(f"N-1 active-space sector   : {sum(hole.charged_nelecastot)} "
      f"electrons in {hole.nkpts * hole.ncas} orbitals")
print(f"N+1 active-space sector   : {sum(particle.charged_nelecastot)} "
      f"electrons in {particle.nkpts * particle.ncas} orbitals")
print()
print("Quasiparticle energies (Eh)")
print("  k   scaled kx   N-1 target   removal pole   "
      "N+1 target   addition pole")
scaled_kpts = cell.get_scaled_kpts(kpts)
for k in np.argsort(scaled_kpts[:, 0]):
    hole_band = hole_by_k[k]
    particle_band = particle_by_k[k]
    print(
        f"{k:3d}  {scaled_kpts[k, 0]:10.6f}  "
        f"{hole_band['target_k']:10d}  {hole_band['energy'].real:13.8f}  "
        f"{particle_band['target_k']:10d}  "
        f"{particle_band['energy'].real:13.8f}"
    )
