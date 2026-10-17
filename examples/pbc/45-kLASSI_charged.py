#!/usr/bin/env python
"""Resolve charged kLASSI states into total-momentum blocks in kLASSCF orbitals.

Each reference cell gets a separate charged LASCI calculation. The resulting
states are coupled in the Wannier product basis, then adapted to translations.
The momentum blocks must reproduce the full state-interaction spectrum.
"""

import numpy as np
from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas


# Build a periodic H2 chain and a neutral mean-field reference.
cell = gto.Cell()
cell.a = np.diag([3.0, 10.0, 10.0])
cell.atom = "H 0 0 0; H 0.74 0 0"
cell.basis = "6-31g"
cell.unit = "Angstrom"
cell.precision = 1e-8
cell.ke_cutoff = 30
cell.verbose = lib.logger.INFO
cell.build()

kmesh = (3, 1, 1)
kpts = cell.make_kpts(kmesh, wrap_around=True)
kmf = scf.KRHF(cell, kpts=kpts).density_fit()
kmf.exxdiv = None
kmf.kernel()
if not kmf.converged:
    raise RuntimeError("Neutral KRHF did not converge")

mo_avas = np.asarray(avas.kernel(kmf, ["H 1s"], minao=cell.basis)[2])

# Optimize the neutral orbitals starting from localized AVAS orbitals.
klasscf = mcscf.KLASSCF(kmf, 2, (1, 1), kmesh=kmesh)
mo_guess = klasscf.localize_init_guess(["H 1s"], mo_coeff=mo_avas)
klasscf.kernel(mo_guess)
if not klasscf.converged:
    raise RuntimeError("Neutral kLASSCF did not converge")

mo = klasscf.mo_coeff

# Recompute the neutral CI energy in precisely the same fixed orbitals.
neutral = mcscf.KLASCI(kmf, 2, (1, 1), kmesh=kmesh)
neutral.kernel(mo.copy())
if not neutral.converged:
    raise RuntimeError("Neutral kLASCI did not converge")

# Positive charge removes an electron; negative charge adds one.
hole = mcscf.KLASSI(neutral, charge=+1)
particle = mcscf.KLASSI(neutral, charge=-1)

# Each kernel solves all three charge locations independently before coupling.
states = {+1: hole, -1: particle}
for charge, lsi in states.items():
    lsi.kernel()
    print(f"Charge {charge:+d}: independent LAS energies/cell (Ha)")

    # Check translations, form U_K, and solve H_K C_K = S_K C_K E_K.
    # K labels total many-electron momentum, with T_a = exp(-i K.a).
    results = lsi.kernel_kpts()
    block_energies = np.sort(np.concatenate([r["e_tot"] for r in results]))
    np.testing.assert_allclose(block_energies, lsi.e_tot, atol=1e-8, rtol=0)
    print("Maximum block/full energy difference (Ha/cell):",
          np.max(abs(block_energies - lsi.e_tot)))

# Convert per-cell energies to full-mesh removal/addition energy differences.
nkpts = len(kpts)
ip = nkpts * (hole.e_tot[0] - neutral.e_tot.real)
ea = nkpts * (neutral.e_tot.real - particle.e_tot[0])
print(f"Lowest-root IP: {ip:.12f} Ha")
print(f"Lowest-root EA: {ea:.12f} Ha")
