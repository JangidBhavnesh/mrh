# !/usr/bin/env python
import numpy as np

from pyscf import gto, fci, scf, lib, ao2mo
from pyscf.csf_fci import csf_solver as real_csf_solver

from mrh.my_pyscf.pbc.fci import csf_solver as cplx_CSFSolver

# An example to run the complex FCI solver in CSFSolver basis.

mol = gto.Mole(atom='H 0 0 0; F 0 0 1.1', 
               basis='STO-6G',
               spin = 2,
               charge = 0,
               verbose=lib.logger.DEBUG)
mol.build ()

mf = scf.RHF (mol)
mf.kernel()

h0 = mf.energy_nuc ()
h1 = mf.mo_coeff.conj ().T @ mf.get_hcore () @ mf.mo_coeff
h2 = ao2mo.restore (1, ao2mo.full (mf._eri, mf.mo_coeff), mol.nao_nr ())

# Defining the size of FCI problem
norb = mol.nao_nr ()
neleca = mol.nelectron // 2
nelecb = mol.nelectron - neleca
nelec = (neleca, nelecb)

# Real integrals based FCI solver
cisolver = real_csf_solver(mol, smult=3)
e_real, ci_real = cisolver.kernel (h1, h2, norb, nelec, ecore=h0)
e_real_check = cisolver.energy (h1, h2, ci_real, norb, nelec) + h0

assert abs(e_real - e_real_check) < 1e-10, \
    "There is some problem, these two energies should be the same!"

# Complex integrals based FCI solver
noise = 1e-3
h0cplx = h0 + noise * 1j   

h1cplx = h1 + noise * 1j
h1cplx = 0.5*(h1cplx + h1cplx.T.conj())

h2cplx = h2 + noise * 1j
h2cplx = 0.5*(h2cplx + h2cplx.transpose(2, 3, 0, 1))
h2cplx = 0.5*(h2cplx + h2cplx.transpose(1, 0, 3, 2).conj())
h2cplx = 0.5*(h2cplx + h2cplx.transpose(3, 2, 1, 0).conj())

cisolver_cplx = cplx_CSFSolver(mol, smult=3)
e_cplx, ci_cplx = cisolver_cplx.kernel (h1cplx, h2cplx, norb, nelec, ecore=h0cplx)
e_cplx_check = cisolver_cplx.energy (h1cplx, h2cplx, ci_cplx, norb, nelec) + h0cplx
assert abs(e_cplx.real - e_cplx_check.real) < 1e-10, \
    "There is some problem, these two energies should be the same!"


# Comparison time:
print(f"Real FCI Energy     : {e_real:.12f}")
print(f"Complex FCI Energy  : {e_cplx.real:.12f}")

# Also compare the spin-square operator expectation value:
from pyscf.fci import spin_op
from mrh.my_pyscf.pbc.fci import spin_op as spin_op_cplx

ss_real = spin_op.spin_square (ci_real, norb, nelec)[0]
ss_cplx = spin_op_cplx.spin_square0 (ci_cplx, norb, nelec)[0]

print(f"Real FCI <S^2>     : {ss_real:.6f}")
print(f"Complex FCI <S^2>  : {ss_cplx.real:.6f}")


import os
import numpy as np
from ase import Atoms
from pyscf import lib
from pyscf.pbc import scf, df
from pyscf.pbc import gto as pgto
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc import mcscf

def make_h2_1D(intraH=1.0, interH=1.5, nx=1, vacuum=17.5):
    Lx = nx * (intraH + interH)
    cell = np.diag([Lx, vacuum, vacuum])
    y0 = vacuum / 2.0
    z0 = vacuum / 2.0
    positions = []
    symbols = []
    for i in range(nx):
        x0 = i * (intraH + interH)
        positions.append([x0,        y0, z0])   # H1
        positions.append([x0+intraH, y0, z0])   # H2
        symbols += ['H', 'H']
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, )
    atoms.center()
    return atoms


# H2 Molecule in 1D
atoms1D = make_h2_1D(intraH=0.74, interH=1.5, nx = 1, vacuum=17.5)
cell = pgto.Cell()
cell.a = atoms1D.cell.array
pos = atoms1D.get_positions()
sym = atoms1D.get_chemical_symbols()
cell.atom = [(sym[i], tuple(pos[i])) for i in range(len(sym))]
cell.basis = 'STO-6G'
cell.unit = 'Angstrom'
cell.max_memory = 100000
cell.ke_cutoff = 100
cell.precision = 1e-10
cell.verbose = lib.logger.INFO
cell.build()

kmesh1D = [7, 1, 1]

kpts = cell.make_kpts(kmesh1D, wrap_around=True)

kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
kmf.max_cycle=1000
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

# This is equivalent to (14e, 14o) in the supercell
kmc = mcscf.CASCI(kmf, 2, 2)
kmc.fcisolver = cplx_CSFSolver(cell, smult=1)
kmc.kernel(kmf.mo_coeff)