# !/usr/bin/env python
from mrh.my_pyscf.pbc.fci import direct_spin1_cplx
from mrh.my_pyscf.pbc.fci import direct_spin1_kfci
from mrh.my_pyscf.pbc.fci import csf_cplx
from mrh.my_pyscf.pbc.fci import spin_op as spin_op
from mrh.my_pyscf.pbc.fci import addons as addons

# Author: Bhavnesh Jangid

"""
Periodic full-CI solvers
"""

# TODO: For the kFCI the CSFSolver is not implemented yet.

# Register all the modules for cleaner API.
__all__ = [
    "DMRGCICPLX",
    "addons",
    "csf_solver",
    "ksolver",
    "solver",
    "spin_op",
]

try:
    from .dmrg_cplx_helper import DMRGCICPLX
except ImportError:
    class DMRGCICPLX:
        def __init__(self, cell, **kwargs):
            raise ImportError(
                "DMRGCI with complex integrals is not available. Please "
                "install the block2 module. See: https://block2.readthedocs."
                "io/en/latest/user/installation.html" \
                "Make sure to turn on Complex build option while " \
                "building the block2 module.")

def solver(cell, singlet, symm=None):
    """Construct the default complex FCI solver."""
    if symm is not None and symm is not False:
        msg = "Point Group Symmetry is not implemented for FCI in PBC yet."
        raise NotImplementedError(msg)
    return direct_spin1_cplx.FCISolver(cell)

def ksolver(cell=None, nkpts=None, target_k=0, symm=None, kpts=None,
            kmesh=None, kconserv=None):
    """Construct an FCI solver for one total-momentum sector."""
    if symm is not None and symm is not False:
        msg = "Point Group Symmetry is not implemented for k-FCI in PBC yet."
        raise NotImplementedError(msg)
    return direct_spin1_kfci.FCISolver(
        cell, nkpts=nkpts, target_k=target_k, kpts=kpts, kmesh=kmesh,
        kconserv=kconserv)

def csf_solver(cell, smult, symm=None):
    """Construct a complex CSF solver."""
    if symm is not None and symm is not False:
        msg = "Point Group Symmetry is not implemented for CSF-FCI in PBC yet."
        raise NotImplementedError(msg)
    if smult == 1:
        return csf_cplx.FCISolverSpin0(cell, smult)
    return csf_cplx.FCISolver(cell, smult)
