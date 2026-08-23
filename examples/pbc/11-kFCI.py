#!/usr/bin/env python
"""Run k-FCI independently in each total crystal-momentum sector."""

import numpy as np

from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc import scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.fci import addons, ksolver

# Author: Bhavnesh Jangid

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
    auxbasis="def2-svp-jkfit")
kmf.max_cycle = 1000
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

# Two active orbitals and electrons per primitive cell correspond to a
# six-orbital, six-electron active space across this three-point mesh.
kmc = mcscf.CASCI(kmf, 2, 2)
kmc.kpts = kpts
kmc.kmesh = kmesh

mo_coeff = np.asarray(kmf.mo_coeff)
h1e, h2e, ecore = addons.get_kfci_integrals(kmc, mo_coeff)
norb = nkpts * kmc.ncas
nelecas = (nkpts * kmc.nelecas[0], nkpts * kmc.nelecas[1])

print(f"k-RHF energy: {kmf.e_tot.real:12.8f}")

# The k-FCI solver provides spin penalization but not a CSF solver.
# Note: k-FCI does support the computation of S2 and RDMs.
print("k-FCI results:")
print(f"{'k-point':<36} {'k-FCI energy':>16}")
for target_k in range(nkpts):
    kmc.fcisolver = ksolver(cell, nkpts=nkpts, target_k=target_k)
    kmc.fcisolver.conv_tol = 1e-10
    kmc.fcisolver.fix_spin_(shift=0.2, ss=0.0)
    e_tot, ci = kmc.fcisolver.kernel(
        h1e, h2e, norb, nelecas, ecore=ecore)
    kpoint = np.array2string(
        kpts[target_k], precision=1, suppress_small=True)
    print(f"{kpoint:<36} {e_tot.real / nkpts:16.8f}")
