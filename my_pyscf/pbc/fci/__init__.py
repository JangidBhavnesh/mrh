"""Periodic full-CI solver factories."""

from mrh.my_pyscf.pbc.fci import direct_spin1_cplx
from mrh.my_pyscf.pbc.fci import direct_spin1_kfci
from mrh.my_pyscf.pbc.fci import csf_cplx
from mrh.my_pyscf.pbc.fci import spin_op as spin_op
from mrh.my_pyscf.pbc.fci import addons as addons


__all__ = [
    "DMRGCICPLX",
    "addons",
    "csf_solver",
    "ksolver",
    "solver",
    "spin_op",
]


# Author: Bhavnesh Jangid

try:
    from .dmrg_cplx_helper import DMRGCICPLX
except ImportError:
    class DMRGCICPLX:
        def __init__(self, cell, **kwargs):
            raise ImportError(
                "DMRGCI with complex integrals is not available. Please "
                "install the block2 module. See: https://block2.readthedocs."
                "io/en/latest/user/installation.html")


def solver(cell, singlet, symm=None):
    """Construct the default periodic complex FCI solver."""
    del singlet
    if symm is not None and symm is not False:
        raise NotImplementedError(
            "Symmetry is not implemented for FCI in PBC yet.")
    return direct_spin1_cplx.FCISolver(cell)


def ksolver(cell=None, nkpts=None, target_k=0, symm=None, kpts=None,
            kmesh=None, kconserv=None):
    """Construct an FCI solver for one total-momentum sector."""
    if symm is not None and symm is not False:
        raise NotImplementedError(
            "Symmetry is not implemented for k-FCI in PBC yet.")
    return direct_spin1_kfci.FCISolver(
        cell, nkpts=nkpts, target_k=target_k, kpts=kpts, kmesh=kmesh,
        kconserv=kconserv)


def csf_solver(cell, smult, symm=None):
    """Construct a periodic complex CSF solver."""
    if symm is not None and symm is not False:
        raise NotImplementedError(
            "Symmetry is not implemented for CSF-FCI in PBC yet.")
    if smult == 1:
        return csf_cplx.FCISolverSpin0(cell, smult)
    return csf_cplx.FCISolver(cell, smult)
