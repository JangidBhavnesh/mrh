from pyscf.fci import addons, cistring

# Calling the spin-penalty function in addons file to keep it consistent with PySCF.
# I have added the contract_ss, so I hope I don't need to write the code for these
# function, rather this should work.

SpinPenaltyFCISolver = addons.SpinPenaltyFCISolver

def _is_kfci_solver(fciobj):
    from mrh.my_pyscf.pbc.fci import direct_spin1_kfci
    return isinstance(fciobj, direct_spin1_kfci.FCISolver)

def fix_spin(fciobj, shift=.1, ss=None, **kwargs):
    if _is_kfci_solver(fciobj):
        from mrh.my_pyscf.pbc.fci import direct_spin1_kfci
        return direct_spin1_kfci.fix_spin(fciobj, shift=shift, ss=ss, **kwargs)
    return addons.fix_spin(fciobj, shift=shift, ss=ss, **kwargs)

def fix_spin_(fciobj, shift=.1, ss=None, **kwargs):
    if _is_kfci_solver(fciobj):
        from mrh.my_pyscf.pbc.fci import direct_spin1_kfci
        return direct_spin1_kfci.fix_spin_(fciobj, shift=shift, ss=ss, **kwargs)
    return addons.fix_spin_(fciobj, shift=shift, ss=ss, **kwargs)

# Helper function to unpack electrons and active space.
def _unpack_nelec(nelec, spin=None):
    if isinstance(nelec, tuple):
        return nelec[0], nelec[1]
    return addons._unpack_nelec(nelec, spin)

def _unpack(norb, nelec, link_index, spin=None):
    if link_index is None:
        neleca, nelecb = _unpack_nelec(nelec, spin)
        link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
        return link_indexa, link_indexb
    else:
        return link_index
