import numpy as np

from pyscf import lib
from pyscf.pbc import scf
from pyscf.pbc import gto as pgto

from mrh.my_pyscf.pbc.fci import csf_solver
from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.mcscf import KLASCI
from mrh.my_pyscf.pbc.util.pbcmolden import print_molden_only_as, print_molden_natorbs

# Example file to use the kLASCI.
# Author: Bhavnesh Jangid

cell = pgto.Cell()
cell.a = np.diag([3.0, 17.5, 17.5])
cell.atom ='''
H 0.0 0.0 0.0
H 0.74 0.0 0.0
'''
cell.basis = '6-31G'
cell.unit = 'Angstrom'
cell.max_memory = 120000
cell.ke_cutoff = 100
cell.precision = 1e-10
cell.verbose = lib.logger.INFO
cell.build()

kmesh = [3, 1, 1]
kpts = cell.make_kpts(kmesh, wrap_around=True)

kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
kmf.max_cycle=1000
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

mo_coeff = avas.kernel(kmf, ['H 1s'], minao=cell.basis)[2]
# Print the molden file for the active space orbitals.
print_molden_only_as(kmf, mo_coeff, kmesh, 'mo_coeff.molden', 2, 0)

kmc = mcscf.CASSCF(kmf, 2, 2)
kmc.kpts = kpts
kmc.kmesh = kmesh
kmc.fcisolver = csf_solver(cell, smult=1)
kmc.max_cycle_macro = 50
kmc.kernel(mo_coeff)

print_molden_natorbs(kmc, kmesh, 'natorbs.molden')