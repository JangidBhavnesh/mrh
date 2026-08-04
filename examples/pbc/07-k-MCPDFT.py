import numpy as np
from pyscf import lib

from pyscf.pbc import scf
from pyscf.pbc import gto as pgto

from mrh.my_pyscf.pbc.fci import csf_solver
from mrh.my_pyscf.pbc.mcscf import avas

# Example usage of k-MC-PDFT for periodic systems at k-points.

cell = pgto.Cell()
cell.a = [[4.175, 0.0, 0.0],
          [0.0, 20.0, 0.0],
          [0.0, 0.0, 20.0]]
cell.atom = '''
Ni 2.73619  10.00000  10.00000
O 0.64880  10.00000  10.00000
O 2.73619  12.08739  10.00000
O 2.73619   7.91261  10.00000
O 2.73619  10.00000   7.91261
O 2.73619  10.00000  12.08739
H 2.73619  12.65763   9.21000
H 2.73619  12.65763  10.78999
H 1.94620  10.00000  12.65763
H 3.52619  10.00000  12.65763
H 2.73619   7.34236  10.78999
H 2.73619   7.34236   9.21000
H 1.94620   9.99999   7.34236
H 3.52619  10.00000   7.34236
'''
cell.basis = {'Ni': 'gth-szv-molopt-sr', 
              'default': 'gth-szv'}
cell.pseudo = 'gth-pade'
cell.max_memory = 100000
cell.ke_cutoff = 100
cell.precision = 1e-10
cell.verbose = lib.logger.INFO
cell.output = 'niaqua.log'
cell.max_memory = 100000
cell.build()

# Choose the k-mesh
kmesh = [3, 1, 1]

kpts = cell.make_kpts(kmesh, wrap_around=True)

kmf = scf.KUHF(cell, kpts=kpts).density_fit()
kmf.max_cycle=1000
kmf.exxdiv = None
kmf.conv_tol = 1e-7
kmf.kernel()

mo_coeff = avas.kernel(kmf, ['Ni 3dz^2', 'Ni 3dx2-y2'], minao=cell.basis)[2]
mo_coeff = np.array(mo_coeff)

from mrh.my_pyscf.pbc.fci import csf_solver
from mrh.my_pyscf.pbc import mcpdft

kmc = mcpdft.KCASCI(kmf, 'tPBE', 2, 2)
kmc.kpts = kpts
kmc.kmesh = kmesh
kmc.fcisolver = csf_solver(cell, smult=1)
kmc.max_cycle_macro = 50
kmc.kernel(mo_coeff)

print(f"k-UHF energy: {kmf.e_tot.real:12.8f}")
print(f"k-MC-PDFT energy: {kmc.e_tot.real:12.8f}")
