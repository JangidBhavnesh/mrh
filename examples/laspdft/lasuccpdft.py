import numpy as np
from scipy import linalg
from pyscf import gto, scf, lib, mcscf, mcpdft
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.exploratory.citools import fockspace
from mrh.exploratory.unitary_cc import lasuccsd 

xyz = '''H 0.0 0.0 0.0
         H 1.0 0.0 0.0
         H 0.2 3.9 0.1
         H 1.159166 4.1 -0.1'''
mol = gto.M (atom = xyz, basis = '6-31g', output='h4_sto3g.log',
    verbose=lib.logger.DEBUG)
mf = scf.RHF (mol).run ()

las = LASSCF (mf, (2,2), (2,2), spin_sub=(1,1))
frag_atom_list = ((0,1),(2,3))
mo_loc = las.localize_init_guess (frag_atom_list, mf.mo_coeff)
las.kernel (mo_loc)

# LASUCC is implemented as a FCI solver for MC-SCF
# It's compatible with CASSCF as well as CASCI, but it's really slow
mc = mcpdft.CASCI (mf, 'tPBE', 4, 4)
mc.mo_coeff = las.mo_coeff
mc.fcisolver = lasuccsd.FCISolver (mol)
mc.fcisolver.norb_f = [2,2] # Number of orbitals per fragment
mc.kernel ()

ref = mcpdft.CASCI(mf, 'tPBE', 4, 4)
ref.mo_coeff = las.mo_coeff
ref.kernel() 


print ("CASCIPDFT  energy:      {:.9f}".format (ref.e_tot))
print ("LASUCCSD-PDFT energy: {:.9f}\n".format (mc.e_tot))

