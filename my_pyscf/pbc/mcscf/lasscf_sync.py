
from pyscf.pbc import scf, df
from mrh.my_pyscf.mcscf import lasci

# Implementation of LASSCF with periodic boundary condition

class PBCLASSCFNoSym(lasci.LASCINoSymm):
    pass

def LASSCF(kmf, ncas, nelecas, **kwargs):
    assert isinstance(kmf, scf.hf.SCF),  \
        "This LAS only works with periodic SCF objects"
    
    with_df = kmf.with_df
    
    if not isinstance(with_df, df.GDF):
        wrn_msg = "Currently, LAS only works with GDF. " 
        raise NotImplementedError(wrn_msg)
    
    klas = PBCLASSCFNoSym(kmf, ncas, nelecas, **kwargs) 
    klas = lasci.density_fit(klas, with_df)
    return klas
