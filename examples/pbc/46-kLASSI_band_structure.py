#!/usr/bin/env python
"""Compute IP, EA, and the finite-mesh charge gap in neutral kLASSCF orbitals.

The neutral energy is from LASCI; N-1 and N+1 energies are the lowest LASSI
roots. All calculations use the same geometry, orbitals, and k-mesh.
The gap is IP - EA, not an optical excitation energy.
"""

import numpy as np
from pyscf import lib
from pyscf.data import nist
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas

# Build the neutral periodic H2 reference.
cell = gto.Cell()
cell.a = np.diag([3., 10., 10.])
cell.atom = 'H 0 0 0; H .74 0 0'
cell.basis = '6-31g'
cell.unit = 'Angstrom'
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
    raise RuntimeError('Neutral KRHF did not converge')

# Optimize the neutral orbitals, then keep them fixed in every charge sector.
neutral = mcscf.KLASSCF(kmf, 2, (1, 1), kmesh=kmesh)
mo_avas = np.asarray(avas.kernel(kmf, ['H 1s'], minao=cell.basis)[2])
mo_guess = neutral.localize_init_guess(['H 1s'], mo_coeff=mo_avas)
neutral.kernel(mo_guess)
if not neutral.converged:
    raise RuntimeError('Neutral kLASSCF did not converge')

neutral_ci = mcscf.KLASCI(kmf, 2, (1, 1), kmesh=kmesh)
neutral_ci.kernel(neutral.mo_coeff.copy())
if not neutral_ci.converged:
    raise RuntimeError('Neutral kLASCI did not converge')

# Solve each charged cell independently and couple its LAS state with the others.
hole = mcscf.KLASSI(neutral_ci, charge=+1)
hole.kernel()
particle = mcscf.KLASSI(neutral_ci, charge=-1)
particle.kernel()

# Convert per-cell energies to full-mesh energies before taking differences.
nkpts = len(kpts)
e_n = nkpts * neutral_ci.e_tot.real
e_nm1 = nkpts * hole.e_tot[0]
e_np1 = nkpts * particle.e_tot[0]
ip = e_nm1 - e_n
ea = e_n - e_np1
gap = ip - ea  # E(N-1) + E(N+1) - 2 E(N).

print('\nFull-mesh energies (Ha):')
print(f'E(N)   = {e_n:.12f}')
print(f'E(N-1) = {e_nm1:.12f}')
print(f'E(N+1) = {e_np1:.12f}')
print('\nEnergy differences against the neutral LAS reference:')
for label, energy in (('IP', ip), ('EA', ea), ('Gap', gap)):
    print(f'{label:3s} = {energy:16.12f} Ha = {energy * nist.HARTREE2EV:12.6f} eV')
