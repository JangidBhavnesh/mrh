#!/usr/bin/env python

"""Compute hole and particle energies with charged kCASCI.

Positive ``charge`` removes an electron from the full k-mesh active space;
negative ``charge`` adds one.  When ``target_k`` is omitted, charged kCASCI
solves every many-electron momentum sector.  ``band_energies`` then combines
those results with the neutral reference and reports each pole at the physical
momentum of the removed or added electron.
"""

import numpy as np

from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc import scf

from mrh.my_pyscf.pbc import mcscf

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
scaled_kpts = cell.get_scaled_kpts(kpts)


def print_bands(label, mc, bands, pole):
    """Print one charged sector ordered by physical crystal momentum."""
    bands = sorted(
        bands, key=lambda band: scaled_kpts[band["momentum_index"], 0],
    )
    print(f"\n{label}: {sum(mc.charged_nelecastot)} electrons in "
          f"{mc.nkpts * mc.ncas} orbitals")
    print(f"  k   scaled kx   target_k   {pole} pole (Eh)")
    for band in bands:
        k = band["momentum_index"]
        print(f"{k:3d}  {scaled_kpts[k, 0]:10.6f}  "
              f"{band['target_k']:9d}  {band['energy'].real:17.8f}")


# Hole calculation: charge=+1 solves the N-1 sectors.  The helper changes
# their many-electron total momenta into removed-electron momenta.
hole = mcscf.KCASCI(kmf, ncas, nelecas, ncore=0, charge=1)
hole.kmesh = kmesh
hole.fcisolver.conv_tol = 1e-10
hole.kernel(mo_coeff)

print(f"\nNeutral kCASCI energy: {e_neutral.real:16.12f}")
print_bands("Hole (N-1)", hole, hole.band_energies(e_neutral), "removal")

# Particle calculation: charge=-1 independently solves the N+1 sectors and
# maps them to the momentum of the added electron.
particle = mcscf.KCASCI(kmf, ncas, nelecas, ncore=0, charge=-1)
particle.kmesh = kmesh
particle.fcisolver.conv_tol = 1e-10
particle.kernel(mo_coeff)
print_bands(
    "Particle (N+1)", particle, particle.band_energies(e_neutral), "addition",
)
