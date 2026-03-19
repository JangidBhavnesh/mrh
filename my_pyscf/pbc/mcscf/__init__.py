
from pyscf.pbc import scf
from mrh.my_pyscf.pbc.mcscf import casci
from mrh.my_pyscf.pbc.mcscf import mc1step

def CASCI(kmf, ncas, nelecas, ncore=None):
    assert isinstance(kmf, scf.hf.SCF),  "CASCI only works with periodic SCF objects"
    kmc = casci.CASCI(kmf, ncas, nelecas, ncore)
    return kmc

def CASSCF(kmf, ncas, nelecas, ncore=None):
    assert isinstance(kmf, scf.hf.SCF),  "CASSCF only works with periodic SCF objects"
    kmc = mc1step.CASSCF(kmf, ncas, nelecas, ncore)
    return kmc