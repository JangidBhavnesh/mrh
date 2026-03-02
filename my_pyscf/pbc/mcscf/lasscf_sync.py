
import numpy as np
from pyscf.pbc import scf, df
from mrh.my_pyscf.mcscf import lasci

# Implementation of LASSCF with periodic boundary condition

class PBCLASSCFNoSym(lasci.LASCINoSymm):
    pass


class _DFLASCI:
    pass



def LASSCF(kmf, ncas_sub, nelecas_sub, **kwargs):
    '''
    Wrapper function for calling LASSCF.
    args:
        kmf: instance of pyscf.pbc.scf
            periodic SCF object.
            TODO: Restructure this if the incoming kmf is UHF.
        ncas_sub: tuple
            number of active orbitals in each subspace.
            Note: The total number of active orbitals will be same in each fragment, that is
            at each k-point. So, i guess it can be replaced by single integer as we do for the
            PBCCASCI. But, for now I am keeping it as tuple to be consistent with the molecular case.
        nelecas_sub: tuple
            number of active electrons in each subspace. Similar comment as above for ncas_sub.
    kwargs:
        spin_sub: tuple
            spin multiplicity of each subspace.
    returns:
        klas: instance of PBCLASSCFNoSym
            LAS class for periodic systems without point group symmetry.
    '''
    assert isinstance(kmf, scf.hf.SCF),  \
        "This LAS only works with periodic SCF objects"
    
    # Sanity Checks:
    if len(ncas_sub) > 1:
        ncas = ncas_sub[0]
        wrn_msg = "Currently, only same number of active orbitals in each subspace is supported. "
        for n in ncas_sub: assert n == ncas, wrn_msg

    if len(nelecas_sub) > 1:
        nelecas_sub_temp = []
        for nelecas in nelecas_sub:
            if isinstance(nelecas, tuple):
                nelecas_sub_temp.append(sum(nelecas))
            else:
                nelecas_sub_temp.append(nelecas)
        wrn_msg = "Currently, only same number of active electrons in each subspace is supported. "
        for n in nelecas_sub_temp: assert n == nelecas_sub_temp[0], wrn_msg
    
    with_df = kmf.with_df
    
    if not isinstance(with_df, df.GDF):
        wrn_msg = "Currently, LAS only works with GDF. " 
        raise NotImplementedError(wrn_msg)
    
    klas = PBCLASSCFNoSym(kmf, ncas_sub, nelecas_sub, **kwargs)

    # Similar to molecular LAS code, I am tagging the with_df object to the LAS class.
    # But, in case of the periodic case the with_df will always be true.
    # TODO: Restructure this in future.
    class DFLASCI (klas.__class__, _DFLASCI):
        def __init__(self, scf, ncas_sub, nelecas_sub):
            self.with_df = with_df
            self._keys = self._keys.union(['with_df'])
            klas.__class__.__init__(self, scf, ncas_sub, nelecas_sub)
    
    new_las = DFLASCI (klas._scf, klas.ncas_sub, klas.nelecas_sub)
    new_las.__dict__.update (klas.__dict__)
    
    return new_las


if __name__ == "__main__":
    from pyscf import lib
    from pyscf.pbc import gto, scf
    # Timer level prints:
    lib.logger.TIMER_LEVEL = lib.logger.INFO

    cell = gto.Cell()
    cell.a = np.eye(3)*3.5668
    cell.atom = 'C 0 0 0'
    cell.basis = 'CC-PVDZ'
    cell.verbose = lib.logger.TIMER_LEVEL
    cell.build()

    kmesh = [1, 1, 2]
    kpts = cell.make_kpts(kmesh)
    
    kmf = scf.KRHF(cell, kpts).density_fit()
    kmf.exxdiv = None
    kmf.max_cycle = 100
    kmf.conv_tol = 1e-12
    kmf.kernel()

    las = LASSCF(kmf, (2,), ((1, 1),))
    print(las.__class__)