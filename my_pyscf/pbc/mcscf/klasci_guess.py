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


def _active_occupations(klas, mo_occ, nkpts, nmo, ncore, ncas):
    kmf = klas._scf
    if mo_occ is None: return [None] * nkpts
    mo_occ = np.asarray(mo_occ)
    if mo_occ.shape == (nmo,):
        mo_occ = np.broadcast_to(mo_occ, (nkpts, nmo))
    if mo_occ.shape != (nkpts, nmo):
        msg = (f"mo_occ must have shape ({nmo},) or ({nkpts}, {nmo}); "
               f"got {mo_occ.shape}")
        raise ValueError(msg)
    return list(mo_occ[:, ncore:ncore+ncas])


def localize_init_guess(klas, frag_atoms=None, mo_coeff=None, spin=None, 
                        lo_coeff=None,fock=None, mo_occ=None, freeze_cas_spaces=True,
                        frags_by_AOs=False, smults_f=None, nelec_f=None, 
                        return_umat=False, return_svals=False, sval_thresh=1e-8):
    '''
    Localize one active space per unit cell.Some args are not used in this function
    but are kept for API compatibility with molecular LAS localization. Those variables
    are ``spin``, ``fock``, ``smults_f`` and ``nelec_f``.  They are not required
    when there is one translationally repeated active space per unit cell.

    Now coming back to this function, it is the periodic, active-space-preserving 
    analogue of molecular ``localize_init_guess``.  At every k-point, an overlap
    SVD selects the combinations of the complete active-band manifold with the largest
    projection onto the same unit-cell orbital space.
    
    Note: The core and virtual orbitals are not changed at all. Only the active orbitals 
    are localized.
    
    args:
        klas: instance of mrh.my_pyscf.pbc.mcscf.klasci 
              periodic LASCI object
        frag_atoms: one unit-cell fragment, specified by atom indices, 
                    AO-label strings, or AO indices when ``frags_by_AOs=True``

    kwargs:
        mo_coeff: np.ndarray or the list of np.arrays, Shape: (nkpts, nao, nmo)
            molecular orbitals for each k-point. If not provided, the orbitals 
            from klas._scf.mo_coeff are used.
        lo_coeff: np.ndarray or the list of np.arrays, Shape: (nkpts, nao, nmo)
                  orthonormal trial orbitals.  meta-Lowdin AOs are used by default.
        mo_occ: np.array or the list of np.arrays, Shape: (nkpts, nmo)
            optional occupation labels.  When supplied, only active
            orbitals with the same occupation are mixed, as in the molecular
            implementation. Basically trying to preserve the reference HF Det.
        freeze_cas_spaces: bool, optional, (default: True)
            If True, the active space is preserved and only the gauge of the
            active orbitals is changed.  If False, the active space is allowed to
            change, as in the molecular implementation.  But currently, this is not
            implemented for periodic systems and with the LAS framework.
        frags_by_AOs: see above.

    returns:
        return_umat: bool, optional, (default: False)
            If True, also return the full k-point MO rotation matrices.
            Umat[k] is the unitary matrix that transforms the input orbitals at k-point k
            to the localized orbitals at k-point k.
        return_svals: bool, optional, (default: False)
            If True, also return the fragment-overlap singular values.
        sval_thresh: float, optional, (default: 1e-8)
            Minimum accepted singular value.  If any singular value is below this
            threshold, an error is raised. 
    '''
    # making sure that the unused args are not used in this function
    del spin, fock, smults_f, nelec_f

    if not freeze_cas_spaces:
        msg = ("Periodic active-band localization always preserves the"
               "active space; freeze_cas_spaces must be True")
        raise NotImplementedError(msg)
    
    if mo_coeff is None: mo_coeff = klas.mo_coeff
    mo_coeff = np.asarray(mo_coeff)

    kmf = klas._scf
    cell = kmf.cell
    kpts = kmf.kpts
    nkpts = len(kpts)
    ncore = klas.ncore
    ncas = klas.ncas
    nocc = ncore + ncas

    # Some sanity checks for mo_coeff
    if mo_coeff.ndim != 3:
        msg = f"mo_coeff must have shape (nkpts, nao, nmo); got {mo_coeff.shape}"
        raise ValueError(msg)
    
    if mo_coeff.shape[0] != nkpts:
        msg = (f"mo_coeff contains {mo_coeff.shape[0]} k-points; "
               f"expected {nkpts}")
        raise ValueError(msg)
    
    nao, nmo = mo_coeff.shape[1:]
    if nao != cell.nao_nr():
        msg = (f"mo_coeff AO dimension is {nao}; expected {cell.nao_nr()}")
        raise ValueError(msg)

    # Get the ovlp from the kmf object only, in case of the pseudo-potential
    # directly using the cell.pbc_intro might be dangerous.
    ovlp = np.asarray(kmf.get_ovlp(kpts=kpts))

    # Localize the orbitals
    if lo_coeff is None: lo_coeff = meta_lowdin_orbitals(cell, ovlp)
    lo_coeff = np.asarray(lo_coeff)

    if lo_coeff.ndim != 3 or lo_coeff.shape[:2] != (nkpts, nao):
        msg = (f"lo_coeff must have shape (nkpts, nao, nlo); got {lo_coeff.shape}")
        raise ValueError(msg)

    frag_orbs = _interpret_unit_cell_orbitals(cell, frag_atoms, 
                                              frags_by_AOs=frags_by_AOs
    )
    if frag_orbs is None:
        frag_orbs = np.arange(lo_coeff.shape[2])
    if np.any(frag_orbs >= lo_coeff.shape[2]):
        msg = ("Fragment AO index is outside the localized trial-orbital space")
        raise IndexError(msg)
    
    if frag_orbs.size < ncas:
        msg = (f"Cannot localize {ncas} active bands using only "
               f"{frag_orbs.size} trial orbitals")
        raise ValueError(msg)

    # If we want to preserve the active space determinant.
    # Collect the active-band occupations for each k-point.  If not provided,
    # the default is to allow all active bands to mix.
    active_occ = _active_occupations(klas, mo_occ, nkpts, nmo, ncore, ncas)

    result_dtype = np.result_type(mo_coeff.dtype, lo_coeff.dtype)
    mo_out = np.array(mo_coeff, dtype=result_dtype, copy=True)
    umat = np.zeros((nkpts, nmo, nmo), dtype=mo_out.dtype)
    svals_out = []

    for k in range(nkpts):
        c_act = mo_coeff[k, :, ncore:nocc]
        trial = lo_coeff[k][:, frag_orbs]
        if active_occ[k] is None:
            # The right singular vectors alone choose principal directions in
            # the active space, but rotate the trial orbitals at the same
            # time.  For one complete active space per unit cell, use the
            # polar factor instead so each active band is aligned with the
            # requested trial orbital and obtains a reproducible k-point
            # gauge.  This is also invariant to the input active-band gauge.
            overlap = c_act.conj().T @ ovlp[k] @ trial
            left, svals, right_h = np.linalg.svd(
                overlap, full_matrices=False
            )
            if trial.shape[1] == ncas:
                c_local = c_act @ left @ right_h
            else:
                # With an overcomplete trial space there is no one-to-one
                # orbital labelling.  Retain its best ncas principal
                # directions, consistent with the molecular SVD framework.
                c_local = c_act @ left
        else:
            _, svals, c_local, _ = klas._svd(
                trial, c_act, s=ovlp[k], mo_occ=active_occ[k]
            )

        if len(svals) < ncas:
            msg = "Note sufficient AOs were selected to localize the active space."
            raise ValueError(msg)

        # Sanity check, if the active space have very minimal overlap with the target
        # fragment orbitals.
        svals = np.asarray(svals[:ncas])
        if np.min(svals) < sval_thresh:
            msg = (f"k-point {k}: fragment orbitals localization is poorer, "
                   f"singular values = {svals}")
            raise ValueError(msg)
        
        c_local = np.asarray(c_local[:, :ncas])
        mo_out[k, :, ncore:nocc] = c_local

        umat[k] = np.eye(nmo, dtype=mo_out.dtype)
        umat[k, ncore:nocc, ncore:nocc] = (c_act.conj().T @ ovlp[k] @ c_local)
        svals_out.append(svals)

    # Check orthogonality of the output orbitals
    orthogonality_check(mo_out, ovlp)
    svals_out = np.asarray(svals_out)

    result = [mo_out]
    if return_umat:
        result.append(umat)
    if return_svals:
        result.append(svals_out)

    if len(result) == 1:
        return result[0]
    else:
        return tuple(result)