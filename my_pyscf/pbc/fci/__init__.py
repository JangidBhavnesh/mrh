from mrh.my_pyscf.pbc.fci import direct_spin1_cplx

try:
    from pyscf import dmrgscf
    from mrh.my_pyscf.pbc.fci import dmrgci_cplx_helper
    DMRGCIComplex = dmrgci_cplx_helper.DMRGCIComplex
except ImportError:
    pass

def solver(cell, singlet, symm=None):
    # Will add the singlet and symm options later.
    return direct_spin1_cplx.FCISolver(cell)