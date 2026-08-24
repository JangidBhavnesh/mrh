#!/usr/bin/env python
import numpy as np

from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc import scf

from mrh.my_pyscf.pbc import mcscf

"""
Example of charged kCASCI calculations and extract quasiparticle energies.

This example demonstrate the charged kCASCI to compute the N-1 and N+1 
active-space problems.  Positive ``charge`` removes one electron from 
the full k-mesh active space, and a negative ``charge`` adds
one.  Omitting ``target_k`` makes the charged kCASCI driver reuse the
transformed integrals while sweeping all allowed total momenta (0 to nkpts-1)
 
The ``band_energies`` helper combines the charged energies with the neutral
reference and maps each many-electron total momentum to the physical momentum
of the removed or added electron.
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
cell.max_memory = 100000
cell.ke_cutoff = 100
cell.precision = 1e-10
cell.verbose = lib.logger.INFO
cell.build()

kmesh = [3, 1, 1]
kpts = cell.make_kpts(kmesh, wrap_around=True)

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

# The charged poles are energy differences relative to the neutral ground
# state, which belongs to target_k=0 for this system.
neutral = mcscf.KCASCI(
    kmf, ncas, nelecas, ncore=0, target_k=0,
)
neutral.kmesh = kmesh
neutral.fcisolver.conv_tol = 1e-10
neutral.canonicalization = False
e_neutral = neutral.kernel(mo_coeff)[0]

# Omitting target_k solves every charged total-momentum sector using the same
# transformed integrals.
hole = mcscf.KCASCI(kmf, ncas, nelecas, ncore=0, charge=1,)
particle = mcscf.KCASCI(kmf, ncas, nelecas, ncore=0, charge=-1,)
for charged_mc in (hole, particle):
    charged_mc.kmesh = kmesh
    charged_mc.fcisolver.conv_tol = 1e-10
    charged_mc.kernel(mo_coeff)

hole_bands = hole.band_energies(e_neutral)
particle_bands = particle.band_energies(e_neutral)
hole_by_k = {band["momentum_index"]: band for band in hole_bands}
particle_by_k = {band["momentum_index"]: band for band in particle_bands}

print()
print(f"neutral kCASCI energy     : {e_neutral.real:16.12f}")
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
