# !/usr/bin/env python

import numpy as np

from mrh.my_pyscf.pbc.util.orth import meta_lowdin_orbitals
from mrh.my_pyscf.pbc.util.orth import orthogonality_check


# Author: Bhavnesh Jangid

'''
In this file: I have added localize_init_guess function which is used to localize the 
active space orbitals in periodic systems unit cell by unit-cell. This fn assumes that
the active space is defined for the unit-cell and the same active space is repeated in
all unit-cells.

Tried my best to keep the code similar to molecular fns.
'''


def _interpret_fragment_orbitals(cell, frag_atoms, frags_by_AOs=False):
    '''
    Convert one unit-cell fragment specification to AO indices.
    The input frag_atoms can be specified in two ways:
        1. list of atom indices (integers) or AO indices (if frags_by_AOs=True)
           This code will convert the atom indices to AO indices using the cell's 
           offset_ao_by_atom() method.
        2. list of AO-label strings (e.g., ['C 2s', 'H 1s']).
    
    args:
        cell: 
            The unit-cell object.
        frag_atoms: list of atom indices (integers) or 
                    AO-label strings (e.g., ['C 2s', 'H 1s']).
                    The fragment specification.
        frags_by_AOs: boolean, optional, (default: True)
            If True, frag_atoms is interpreted as AO indices.
    returns:
        ao_idx: np.array 
            AO indices corresponding to the specified fragment
    '''

    # Sanity check for frag_atoms
    if frag_atoms is None: return None
    if isinstance(frag_atoms, (str, np.character, int, np.integer)):
        frag_atoms = [frag_atoms]
    elif len(frag_atoms) == 1 and isinstance(
            frag_atoms[0], (list, tuple, np.ndarray)):
        frag_atoms = frag_atoms[0]

    # Check if frag_atoms is a list of integers (atom indices or AO indices)
    is_int = all(isinstance(i, (int, np.integer)) for i in frag_atoms)
    is_str = all(isinstance(i, (str, np.character)) for i in frag_atoms)

    if is_int:
        if frags_by_AOs: 
            ao_idx = np.asarray(frag_atoms, dtype=int)
        else:
            ao_offset = cell.offset_ao_by_atom()
            ao_idx = np.asarray([ao 
                                 for atom in frag_atoms 
                                 for ao in range(ao_offset[atom, 2], 
                                                 ao_offset[atom, 3])], dtype=int)
    elif is_str:
        ao_idx = np.asarray(sorted(set(ao 
                                       for label in frag_atoms 
                                       for ao in cell.search_ao_label(label))), dtype=int)
    else:
        msg = ("Fragment must be specified using only atom/AO indices or only "
               "AO-label strings")
        raise TypeError(msg)

    # Final sanity checks for ao_idx
    if ao_idx.size == 0:
        msg = "The fragment specification does not select any AOs"
        raise ValueError(msg)
    if np.unique(ao_idx).size != ao_idx.size:
        msg = "The fragment specification contains duplicate AOs"
        raise ValueError(msg)
    if np.any(ao_idx < 0) or np.any(ao_idx >= cell.nao_nr()):
        msg = "Fragment AO index is outside the unit-cell AO space"
        raise IndexError(msg)
    
    return ao_idx

_interpret_unit_cell_orbitals = _interpret_fragment_orbitals