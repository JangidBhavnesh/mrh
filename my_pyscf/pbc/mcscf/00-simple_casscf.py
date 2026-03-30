
import os
import numpy as np
from ase import Atoms
from pyscf import lib, mcscf
from pyscf.pbc import scf, df
from pyscf.pbc import gto as pgto
# from mrh.my_pyscf.pbc.mcscf import avas
# from mrh.my_pyscf.pbc import mcscf

# Example to run the CASCI for kmf object (1D, 2D or 3D)

# TODO: Write one general function to make the H2 chain in 1D, 2D and 3D. 
# The current code is a bit redundant.
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
cell.basis = '6-31G'
cell.unit = 'Angstrom'
cell.max_memory = 100000
cell.ke_cutoff = 100
cell.precision = 1e-10
cell.verbose = lib.logger.INFO
# cell.output = 'h2_1D_CASCI.log'
cell.build()

kmesh1D = [3, 1, 1]

kpts = cell.make_kpts(kmesh1D, wrap_around=True)

kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
kmf.max_cycle=1000
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

from pyscf.pbc.tools import k2gamma

mf = k2gamma.k2gamma(kmf)

print(" ")
kmc2 = mcscf.CASSCF(mf, 6, (3,3))
kmc2.max_cycle_macro = 0
kmc2.fix_spin_(ss=0)
kmc2.kernel()

np.set_printoptions(precision=3, suppress=True, linewidth=120)
grad = kmc2.get_grad()
print(kmc2.unpack_uniq_var(grad))
# print(grad.shape)
# ecasscf = kmc2.e_tot
