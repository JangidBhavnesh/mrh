# !/usr/bin/env python

from mrh.my_pyscf.pbc.mcpdft.otfnalperiodic import (
    _get_ks_obj,
    _get_mol_or_cell,
    get_pbc_otfnal_gamma,
    otfnalperiodic_gamma,
    periodicpdft,
    redefine_fnal,
    redefine_ftransfnal,
    redefine_transfnal,
    sanity_check_for_kpts,
)

# Author: Bhavnesh Jangid

"""
Backward-compatible imports for periodic on-top functionals.

Periodic MC-PDFT is implemented in :mod:`mrh.my_pyscf.pbc.mcpdft`.  This
module retains the historical import paths used by downstream callers.
"""

otfnalperiodic = otfnalperiodic_gamma
_get_transfnal = get_pbc_otfnal_gamma
sanity_check_for_df = _get_ks_obj
