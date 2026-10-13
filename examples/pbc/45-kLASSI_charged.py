#!/usr/bin/env python
"""Interact independently optimized charged LAS states in kLASSCF orbitals.

Each reference cell gets a separate charged LASCI calculation. The resulting
states are coupled directly; no translation adaptation or momentum assignment
is performed in this example.
"""

import numpy as np
from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas


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

# Optimize one common neutral orbital set.
neutral = mcscf.KLASSCF(kmf, 2, (1, 1), kmesh=kmesh)
mo_avas = np.asarray(avas.kernel(kmf, ['H 1s'], minao=cell.basis)[2])
mo_guess = neutral.localize_init_guess(['H 1s'], mo_coeff=mo_avas)
neutral.kernel(mo_guess)
if not neutral.converged:
    raise RuntimeError('Neutral kLASSCF did not converge')

# Recompute the neutral CI energy in precisely the same fixed orbitals.
neutral_ci = mcscf.KLASCI(kmf, 2, (1, 1), kmesh=kmesh)
neutral_ci.kernel(neutral.mo_coeff.copy())
if not neutral_ci.converged:
    raise RuntimeError('Neutral kLASCI did not converge')

# Each kernel solves all three charge locations independently before coupling.
hole = mcscf.KLASSI(neutral_ci, charge=+1)
particle = mcscf.KLASSI(neutral_ci, charge=-1)
for lsi in (hole, particle):
    lsi.kernel()
    print(f'Charge {lsi.charge:+d}: independent LAS energies/cell (Ha)')
    print([state.e_tot.real for state in lsi.charged_states])
    print('LASSI energies/cell (Ha):', lsi.e_tot)
    print('SI coefficients (rows = charged cells, columns = roots):')
    print(lsi.si)

# Full-mesh energy differences against the neutral LAS reference.
ip = len(kpts) * (hole.e_tot[0] - neutral_ci.e_tot.real)
ea = len(kpts) * (neutral_ci.e_tot.real - particle.e_tot[0])
print(f'Lowest-root IP: {ip:.12f} Ha')
print(f'Lowest-root EA: {ea:.12f} Ha')
