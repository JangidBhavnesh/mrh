"""Periodic FCI add-ons and compatibility dispatch."""

from pyscf.fci import addons, cistring


SpinPenaltyFCISolver = addons.SpinPenaltyFCISolver


def _is_kfci_solver(fciobj):
    """Return whether an object uses the momentum-sector FCI solver."""
    from mrh.my_pyscf.pbc.fci import direct_spin1_kfci

    return isinstance(fciobj, direct_spin1_kfci.FCISolver)


def fix_spin(fciobj, shift=0.1, ss=None, **kwargs):
    """Dispatch spin-penalty construction to the matching solver family."""
    if _is_kfci_solver(fciobj):
        from mrh.my_pyscf.pbc.fci import direct_spin1_kfci

        return direct_spin1_kfci.fix_spin(
            fciobj, shift=shift, ss=ss, **kwargs)
    return addons.fix_spin(fciobj, shift=shift, ss=ss, **kwargs)


def fix_spin_(fciobj, shift=0.1, ss=None, **kwargs):
    """Apply the appropriate spin-penalty mixin in place."""
    if _is_kfci_solver(fciobj):
        from mrh.my_pyscf.pbc.fci import direct_spin1_kfci

        return direct_spin1_kfci.fix_spin_(
            fciobj, shift=shift, ss=ss, **kwargs)
    return addons.fix_spin_(fciobj, shift=shift, ss=ss, **kwargs)


def _unpack_nelec(nelec, spin=None):
    """Normalize electron counts to an alpha/beta tuple."""
    if isinstance(nelec, tuple):
        return nelec[0], nelec[1]
    return addons._unpack_nelec(nelec, spin)


def _unpack(norb, nelec, link_index, spin=None):
    """Generate molecular link indices when none are supplied."""
    if link_index is None:
        neleca, nelecb = _unpack_nelec(nelec, spin)
        link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
        return link_indexa, link_indexb
    return link_index
