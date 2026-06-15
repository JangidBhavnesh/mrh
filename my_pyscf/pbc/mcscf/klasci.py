# /usr/bin/env python3

import numpy as np
import scipy.linalg

from pyscf import lib
from pyscf.pbc import scf, dft, df


from mrh.my_pyscf.mcscf.lasci import LASCINoSymm
from mrh.my_pyscf.pbc.mcscf import casci

# Author: Bhavnesh Jangid

'''
# Let's start implementing the k-LAS algorithm.
# Steps
1. Implement the k-LASCI
2. Implement the k-LASSCF
3. Implement the LASSI algorithm
'''


def kLASCI(kmf, ncas, nelecas, ncore=None, kmesh=None, kpts=None):
    '''
    Wrapper function for k-LASCI. 
    args:
        kmf: pbc.scf object
            mean-field object
        ncas: int
            number of active orbitals per unit cell
        nelecas: int/tuple
            number of active electrons per unit cell
        ncore: int, optional
            number of core orbitals per unit cell
        kmesh: tuple, optional
            k-point mesh for the calculation.
        kpts: array_like, optional
            k-points for the calculation. 
    returns:
        klas: k-LASCI object
    '''
    assert isinstance(kmf, scf.hf.SCF),  \
        "k-LASCI only works with periodic SCF objects"
    
    if kmf.cell.symmetry:
        raise NotImplementedError("k-LASCI with symmetry is not implemented yet.")
    
    # Sanity check to make sure that DFT mean field objects are not passed to k-LASCI
    if isinstance(kmf, dft.krks.KRKS) or isinstance(kmf, dft.kuks.KUKS) \
        or isinstance(kmf, dft.rks.RKS) or isinstance(kmf, dft.uks.UKS):
        raise NotImplementedError("k-LASCI with DFT is not implemented yet.")
    
    # If the mean-field object is KUHF, convert it to RHF before passing to k-LASCI,
    if isinstance(kmf, scf.kuhf.KUHF):
        kmf = scf.addons.convert_to_rhf(kmf)
    
    # Currently, the k-LAS should work with the GDF density fitting object.
    assert isinstance(kmf.with_df, df.df.GDF), \
        "k-LASCI only works with GDF density fitting object"

    klas = LASCINoSymm(kmf, ncas, nelecas, ncore, kmesh, kpts)

    return klas

class LASCINoSymm(casci.CASCI, LASCINoSymm):
    '''
    Localized active space CI (LASCI) class for periodic systems without 
    point group symmetry.
    args:
        kmf: pbc.scf object
            mean-field object
        ncas: int
            number of active orbitals per unit cell
        nelecas: int/tuple
            number of active electrons per unit cell
        spin_sub: int, optional (2S + 1)
            spin multiplicity of the active space in the unit cell.
            If not provided, it will be automatically determined based 
            on the number of active electrons.
        ncore: int, optional
            number of core orbitals per unit cell
        kmesh: tuple, optional
            k-point mesh for the calculation.
        kpts: array_like, optional
            k-points for the calculation.
    '''
    def __init__(self, kmf, ncas, nelecas, ncore=None, spin_sub=None, 
                 kmesh=None, kpts=None):
        super().__init__()
        self.kmf = kmf
        self.ncas = ncas
        self.nelecas = nelecas
        self.ncore = ncore
        self.spin_sub = spin_sub
        self.kmesh = kmesh
        self.kpts = kpts
