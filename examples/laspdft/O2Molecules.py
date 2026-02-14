import sys
from pyscf import gto, scf, tools, dft, lib
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_dmet import localintegrals, dmet, fragments
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf import lassi

# Initialization
mol = gto.M()
mol.atom='''O -5.10574 2.01997 0.00000;  O -4.10369 2.08633 0.00000; 
O -3.10185 3.22603 0.00000; O -2.10672 3.35095 0.00000''' 
mol.basis='6-31g'
mol.verbose=4
mol.spin = 4
mol.build()

# Mean field calculation
mf = scf.ROHF(mol).newton().run()

# LASSCF Calculations
las = LASSCF(mf,(8, 8),(12, 12),spin_sub=(3, 3))
frag_atom_list = ([0, 1], [2, 3])
mo0 = las.localize_init_guess(frag_atom_list, freeze_cas_spaces=True)

from pyscf.tools import molden
molden.from_mo(mol, 'N2.molden', mo0)

mo1 = las.localize_init_guess(frag_atom_list, freeze_cas_spaces=True,maintains_det=True, spin_sub=(3, 3))
molden.from_mo(mol, 'N2.C.molden', mo1)

las.lasci(mo0)
print("DOne")
las.lasci(mo1)


