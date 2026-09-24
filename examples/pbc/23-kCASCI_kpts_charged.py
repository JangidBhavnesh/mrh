#!/usr/bin/env python

"""Compute hole and particle energies with charged kCASCI.

Positive ``charge`` removes an electron from the full k-mesh active space;
negative ``charge`` adds one.  When ``target_k`` is omitted, charged kCASCI
solves every many-electron momentum sector.  ``band_energies`` then combines
those results with the neutral reference and reports each pole at the physical
momentum of the removed or added electron.  The signed poles are
``E(N) - E(N-1)`` for removal and ``E(N+1) - E(N)`` for addition; the former
is the negative of the conventional positive ionization potential.
"""

# Author: Bhavnesh Jangid

import numpy as np

from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc import scf

from mrh.my_pyscf.pbc import mcscf

intra_h = 0.74
inter_h = 1.5
vacuum = 17.5

cell = gto.Cell()
cell.a = np.diag([intra_h + inter_h, vacuum, vacuum])
cell.atom = [
    ["H", (0.0, vacuum / 2.0, vacuum / 2.0)],
    ["H", (intra_h, vacuum / 2.0, vacuum / 2.0)],
]
cell.basis = "STO-6G"
cell.unit = "Angstrom"
cell.ke_cutoff = 100
cell.precision = 1e-10
cell.verbose = lib.logger.INFO
cell.build()

kmesh = [3, 1, 1]
kpts = cell.make_kpts(kmesh, wrap_around=True)

kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis="def2-svp-jkfit",)
kmf.max_cycle = 100
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

mo_coeff = np.asarray(kmf.mo_coeff)
ncas = 2
nelecas = 2

# Neutral N-electron reference in the zero-total-momentum sector.
neutral = mcscf.KCASCI(kmf, ncas, nelecas, ncore=0, target_k=0)
neutral.kmesh = kmesh
neutral.fcisolver.conv_tol = 1e-10
neutral.canonicalization = False
e_neutral = neutral.kernel(mo_coeff)[0]

# charge=+1 removes one active electron.  Omitting target_k solves every
# N-1 total-momentum sector.
hole = mcscf.KCASCI(kmf, ncas, nelecas, ncore=0, charge=1)
hole.kmesh = kmesh
hole.fcisolver.conv_tol = 1e-10
hole.kernel(mo_coeff)

# charge=-1 adds one active electron and solves every N+1 sector.
particle = mcscf.KCASCI(kmf, ncas, nelecas, ncore=0, charge=-1)
particle.kmesh = kmesh
particle.fcisolver.conv_tol = 1e-10
particle.kernel(mo_coeff)

# Results:
print(f"\nNeutral kCASCI energy/cell: {e_neutral.real:16.12f}")
hole.print_bands(e_neutral)
particle.print_bands(e_neutral)
