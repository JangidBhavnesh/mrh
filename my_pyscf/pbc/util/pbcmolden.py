import numpy as np

from pyscf.tools import molden
from mrh.my_pyscf.pbc.mcscf.k2R import  get_mo_coeff_k2R

# Author: Bhavnesh Jangid
# Some wrapper function to print the orbitals from the kRHF or the k-CASSCF 
# on the molden file.

def print_molden(kmf, mo_coeff, kmesh, filename, only_occ=False, only_virt=False):
    '''
    Print molecular orbitals in Molden format.
    Basically a wrapper around pyscf.tools.molden.from_mo to convert
    k-point MO coefficients to real space MO coefficients.
    args:
        kmf: pbc.scf object
            mean-field object
        mo_coeff: list or np.ndarray (nkpts, nao, nmo)
            k-point MO coefficients
        kmesh: tuple
            k-point mesh for the calculation.
        filename: str
            name of the output molden file.
    '''
    nelec = kmf.cell.nelectron
    nocc = nelec // 2 + nelec % 2
    ncore = 0
    assert np.prod(kmesh) == len(kmf.kpts), "kmesh and number of kpts in kmf do not match"

    scell, _, mo_coeff_R =  get_mo_coeff_k2R(kmf, mo_coeff, ncore, nocc, kmesh=kmesh)[:2]

    if only_occ:
        mo_coeff_R = mo_coeff_R[:,:nocc]
    elif only_virt:
        mo_coeff_R = mo_coeff_R[:,nocc:]

    molden.from_mo(scell, filename, mo_coeff_R.real)

    return None

def print_molden_only_as(kmf, mo_coeff, kmesh, filename, ncas, ncore):
    '''
    Print only the active space orbitals in Molden format.
    This is useful for visualizing the active space orbitals.
    args:
        kmf: pbc.scf object
            mean-field object
        mo_coeff: list or np.ndarray (nkpts, nao, nmo)
            k-point MO coefficients
        kmesh: tuple
            k-point mesh for the calculation.
        filename: str
            name of the output molden file.
        ncas: int
            number of active orbitals per unit cell
        ncore: int
            number of core orbitals per unit cell
    '''

    from pyscf.tools import molden
    scell, _, mo_coeff_R =  get_mo_coeff_k2R(kmf, mo_coeff, ncore, ncas, kmesh=kmesh)[:2]
    molden.from_mo(scell, filename, mo_coeff_R.real)

    return None
