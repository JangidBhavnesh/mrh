import numpy as np
from pyscf import gto, scf, cc
from mrh.my_pyscf.dmet import runDMET

'''
Author: Bhavnesh Jangid

Example of using DMET to run the unrestricted coupled cluster calculations on a molecule.
Basically, we need to start with the ROHF/RHF type HF but later on if we want to run
the UCC type wave functions, we need (or atleast it's better to use) UHF reference 
wave function. In this example, we will show how to get the UHF wave function from the ROHF wave function.
'''

np.set_printoptions(precision=4)

mol = gto.Mole(basis='6-31G', spin=1, charge=0, verbose=4, max_memory=10000)
mol.atom='''
P  -5.64983   3.02383   0.00000
H  -4.46871   3.02383   0.00000
H  -6.24038   2.19489   0.59928
Ne 0 0 10
'''
mol.build()

mf = scf.ROHF(mol).density_fit()
mf.kernel()

# To get back the UHF wave function, turn on the uhf=True flag.
dmet_mf, mydmet = runDMET(mf, lo_method='lowdin', 
                          bath_tol=1e-6, atmlst=[0, ], density_fit=True)

# Convert the DMET mean-field object to UHF mean-field object and run the UHF calculation.
dmet_mf = dmet_mf.to_uhf()
dmet_mf.kernel()

# Using the dmet_mf to run the UCCSD or type of calculations.
# Feel free to set the parameters of the UCCSD object as per your requirement.
mycc = cc.UCCSD(dmet_mf)
mycc.kernel()

# Solve the lambda equations to get the spin square value.
s2 = mycc.spin_square()[0]

print("---"*30)
print("DMET RHF Energy:", dmet_mf.e_tot, " and DMET RHF Spin Square: ", dmet_mf.spin_square()[0])
print("UCCSD Energy: ", mycc.e_tot, " and UCCSD Spin Square: ", s2)