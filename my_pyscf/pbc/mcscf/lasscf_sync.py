
from pyscf.pbc import scf
from mrh.my_pyscf.mcscf import lasci

# Implementation of LASSCF with periodic boundary condition

class PBCLASSCFNoSym(lasci.LASCINoSymm):
    pass

def LASSCF(kmf, ncas, nelecas, **kwargs):
    assert isinstance(kmf, scf.hf.SCF),  "This LAS only works with periodic SCF objects"
    
    with_df = kmf.with_df
    
    if with_df is not isinstance(with_df, scf.ao2mo.AO2MO):
        raise NotImplementedError("LAS only works with SCF objects that have an AO2MO object for density fitting")
    
    klas = PBCLASSCFNoSym(kmf, ncas, nelecas, **kwargs) 
    klas = lasci.density_fit(klas, with_df)
    return klas
