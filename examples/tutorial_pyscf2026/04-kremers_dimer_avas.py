from pyscf import gto, scf, mcscf
from pyscf.tools import molden
from pyscf.csf_fci import csf_solver

xyz='''Cr    -1.320780000000   0.000050000000  -0.000070000000
Cr     1.320770000000   0.000050000000  -0.000070000000
O      0.000000000000  -0.165830000000   1.454680000000
O      0.000000000000   1.342770000000  -0.583720000000
O      0.000000000000  -1.176830000000  -0.871010000000
H      0.000020000000   0.501280000000   2.159930000000
H      0.000560000000   1.618690000000  -1.514480000000
H     -0.000440000000  -2.120790000000  -0.644130000000
N     -2.649800000000  -1.445690000000   0.711420000000
H     -2.186960000000  -2.181980000000   1.244400000000
H     -3.053960000000  -1.844200000000  -0.136070000000
H     -3.367270000000  -1.005120000000   1.287210000000
N     -2.649800000000   1.339020000000   0.896300000000
N     -2.649800000000   0.106770000000  -1.607770000000
H     -3.367270000000  -0.612160000000  -1.514110000000
H     -3.053960000000   0.804320000000   1.665160000000
N      2.649800000000  -1.445680000000   0.711420000000
N      2.649790000000   1.339030000000   0.896300000000
N      2.649800000000   0.106780000000  -1.607770000000
H     -2.186970000000   2.168730000000   1.267450000000
H     -3.367270000000   1.617370000000   0.226860000000
H     -2.186960000000   0.013340000000  -2.511900000000
H     -3.053970000000   1.039980000000  -1.529140000000
H      2.186960000000  -2.181970000000   1.244400000000
H      3.053960000000  -1.844190000000  -0.136080000000
H      3.367270000000  -1.005100000000   1.287200000000
H      2.186950000000   2.168740000000   1.267450000000
H      3.053960000000   0.804330000000   1.665160000000
H      3.367260000000   1.617380000000   0.226850000000
H      2.186960000000   0.013350000000  -2.511900000000
H      3.053960000000   1.039990000000  -1.529140000000
H      3.367270000000  -0.612150000000  -1.514110000000'''
basis = {'C': 'sto-3g','H': 'sto-3g','O': 'sto-3g','N': 'sto-3g','Cr': 'cc-pvdz'}
mol = gto.M (atom=xyz, spin=6, charge=3, basis=basis,
             verbose=4, output='04-kremers_dimer_avas.log') 
mol.max_memory=8000
mf = scf.ROHF(mol)
mf.chkfile = '04-kremers_dimer_avas.chk'
mf.kernel () 

# Using "AVAS" to try to automatically find the 3d orbitals
from pyscf.mcscf import avas
ncas, nelecas, mo_coeff = avas.kernel (mf, ['Cr 3d'], openshell_option=3)
molden.from_mo (mol, '04-kremers_dimer_avas_guess.molden', mo_coeff, occ=mf.mo_occ)

# Optimizing orbitals for the open-shell singlet
mc = mcscf.CASSCF (mf, ncas, (nelecas//2,nelecas//2))

# Guarantee spin singlet (instead of, i.e., ms=0 triplet)
mc.fcisolver = csf_solver (mol, smult=1)
mc.kernel (mo_coeff)

# QOL over PySCF's molden tools: sets E of active orbitals to 0 so jmol doesn't scramble them as badly
from mrh.my_pyscf.tools import molden as my_molden
my_molden.from_mcscf (mc, '04-kremers_dimer_avas.molden', cas_natorb=True)





