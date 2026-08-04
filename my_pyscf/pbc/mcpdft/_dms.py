import numpy as np
from pyscf.mcpdft import _dms

# Author: Bhavnesh Jangid

# Proud of me to find out this problem.

def dm2_cumulant_complex(dm2, dm1s):
    r'''
    Build a two-body cumulant from PySCF RDMs in a complex orbital basis.
    This wrapper function is necessary because the standard PySCF implementation
    assumes real orbitals.

    Note: PySCF stores 
        - 1-RDM as``dm1[p,q] = <q^\dagger p>`` 
        - 2-RDM as``dm2[p,q,r,s] = <p^\dagger r^\dagger s q>``.
    
    Directly using mcpdft._dms.dm2_cumulant, which is designed for real orbitals, 
    will not work correctly in a complex basis. The molecular MC-PDFT
    cumulant expression assumes real orbitals, where transposing the 1-RDM is
    not that important.  In a complex basis, the transposed 1-RDM is required for the
    disconnected terms to transform covariantly with the 2-RDM.
    '''
    dm1s = np.asarray(dm1s)
    return _dms.dm2_cumulant(dm2, dm1s.swapaxes(-1, -2))
