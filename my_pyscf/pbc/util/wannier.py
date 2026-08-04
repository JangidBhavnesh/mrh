# !/usr/bin/env python

import numpy as np
import scipy

from pyscf import lib
from mrh.my_pyscf.pbc.util.transym import TranslationSymm
from mrh.my_pyscf.pbc.util import orth 

lowdin_sym = orth.lowdin_sym
meta_lowdin_orbitals = orth.meta_lowdin_orbitals
orthogonality_check = orth.orthogonality_check

# Author: Bhavnesh Jangid

'''
Bloch-to-Wannier transformations and Wannier gauges related functions
for k-LAS
'''

def localize_kmf_mo_coeff(kmf, mo0):
    '''
    Rotates the kmf.mo_coeff occupied and virtual orbitals
    in local orthonormal meta-lowdin basis.
    args:
        kmf: mean-field
            instance of the pbc.scf
        mo0: np.ndarray or list (nkpts, nao, nmo)
            kmf mo_coeff or the AVAS mo_coeff
    return:
        lo_coeff: localized mo_coeff.
    '''
    cell = kmf.cell
    kpts = kmf.kpts
    ovlp = kmf.get_ovlp(kpts=kpts)
    lo_coeff = meta_lowdin_orbitals(cell, ovlp)

    log = lib.logger.Logger(kmf.stdout, kmf.verbose)

    mo_coeff = np.array([c.copy() for c in mo0])
    mo_coeff_loc = []
    umat = []

    def _project(mo_coeff_k, lo_coeff_k, ovlp_k):
        norb = mo_coeff_k.shape[1]
        assert norb <= lo_coeff_k.shape[1], f"Less AOs then MOs"

        pmat = ((mo_coeff_k.conj().T @ ovlp_k) @ lo_coeff_k)[:norb, :norb]
        pinv = pmat.conj().T @ pmat
        pinv = lowdin_sym(pinv)
        umat_k = pmat @ pinv

        # Localize the block orbitals
        mo_k = mo_coeff_k @ umat_k

        return mo_k, umat_k
        
    for k in range(len(kpts)):
        mo_occ = kmf.mo_occ[k] > 0
        mo_vir = ~mo_occ
        
        # Project the occupied space
        lo_coeff_k = lo_coeff[k]
        ovlp_k = ovlp[k]
        mo_coeff_k = mo_coeff[k][:, mo_occ]
        mo_k_occ, umat_k_occ = _project(mo_coeff_k, lo_coeff_k, ovlp_k)
        
        # project the virtual space
        lo_coeff_k = lo_coeff[k]
        ovlp_k = ovlp[k]
        mo_coeff_k = mo_coeff[k][:, mo_vir]
        mo_k_vir, umat_k_vir = _project(mo_coeff_k, lo_coeff_k, ovlp_k)
        mo_coeff_loc.append(np.hstack((mo_k_occ, mo_k_vir)))
        umat.append(scipy.linalg.block_diag(umat_k_occ, umat_k_vir))

    # Check the orthogonality of the localized orbitals.
    orthogonality_check(mo_coeff_loc, ovlp)
    log.info('Orthogonality check passed!')
    return np.array(mo_coeff_loc), np.array(umat)

def get_wannier_orbs(kmf, kmesh, mo_loc_k):
    '''
    Building the Wannier orbitals:
        W[R, mu, S, n]
            = 1/Nk sum_k exp(+i k.(R-S)) [C(k) U(k)]_{mu,n}
            = 1/Nk sum_k exp(+i k.(R-S)) [C_loc(k)]_{mu,n}
    args:
        ts: TranslationSymm object
            Object containing the translation symmetry information of the system.
        kmf: pbc.scf object
            mean-field object containing the mo_coeff and other information.
        kpts: np.ndarray of shape (Nk, 3)
            Array of k-points in reciprocal space.
        mo_loc_k: list of np.ndarray (nkpts, nao, norb)
            Localized mo_coeff for each k-point.
            norb can be nmo or just the ncas.
    returns:
        wannier_orb : ndarray W[R, mu, S, n] (ncell, nao, ncell, nwann)
            Wannier orbitals in real space.
        R_indices : ndarray
            BvK cell indices.
        mo_phase : ndarray (nkpts, nwann, ncell*nwann)
            Projection of the Wannier orbitals onto the k-space localized orbitals:
                mo_phase[k, m, S*nwann+n] = < C_loc(k,m) | W[S,n] >
    '''
    cell = kmf.cell
    kpts = kmf.kpts

    # Use complex dtype because the Fourier phase is complex.
    dtype = np.result_type(mo_loc_k[0].dtype, np.complex128)

    ts = TranslationSymm(cell, kmesh, kpts=kpts)
    nkpts = len(kpts)
    R_indices = ts.R_indices
    ncell = len(R_indices)

    assert np.prod(kmesh) == len(kpts), "kmesh and number of kpts in kmf do not match"
    assert ncell == nkpts

    nao = mo_loc_k[0].shape[0]
    nwann = mo_loc_k[0].shape[1]

    mo_loc_k = np.asarray(mo_loc_k, dtype=dtype)

    wannier_orb = np.zeros((ncell, nao, ncell, nwann), dtype=dtype)

    R_cart = np.array([ts.lattice_cart(R) for R in R_indices])

    for ik, k in enumerate(kpts):
        Ck = mo_loc_k[ik]
        for iR, Rv in enumerate(R_cart):
            for iS, Sv in enumerate(R_cart):
                phase_RS = np.exp(1j * np.dot(k, Rv - Sv))
                wannier_orb[iR, :, iS, :] += phase_RS * Ck

    wannier_orb /= nkpts

    mo_coeff_R = wannier_orb.reshape(ncell*nao, ncell*nwann)

    phase_Rk = np.zeros((ncell, nkpts), dtype=dtype)
    for ik, k in enumerate(kpts):
        for iR, Rv in enumerate(R_cart):
            phase_Rk[iR, ik] = np.exp(1j * np.dot(k, Rv)) / np.sqrt(nkpts)

    s_k = np.asarray(kmf.get_ovlp(kpts=kpts), dtype=dtype)
    s_k_g = np.einsum('kuv,Rk->kuRv', s_k, phase_Rk.conj())
    s_k_g = s_k_g.reshape(nkpts, nao, ncell*nao)

    mo_phase = np.einsum('kum,kui->kmi',mo_loc_k.conj(),
                         np.dot(s_k_g, mo_coeff_R),optimize=True)

    return wannier_orb, R_indices, mo_phase

def check_wannier_orbital_translation(ts, W, R_indices=None,
                                      T_index=(1, 0, 0), tol=1e-8):
    '''
    Check W[R+T, mu, S+T, n] = W[R, mu, S, n].
    '''
    if R_indices is None:
        R_indices = ts.R_indices
    R_indices = np.asarray(R_indices)
    R_to_i = ts.index_map(R_indices)
    T_index = np.asarray(T_index, dtype=int)

    log = lib.logger.Logger(ts.cell.stdout, ts.cell.verbose)
    ncell, nao, ncell2, nwann = W.shape
    assert ncell == ncell2

    max_abs = 0.0
    max_ref = 0.0
    worst = None

    for iR, R in enumerate(R_indices):
        RpT = ts.mod_index(R + T_index)
        iRpT = R_to_i[RpT]

        for iS, S in enumerate(R_indices):
            SpT = ts.mod_index(S + T_index)
            iSpT = R_to_i[SpT]

            ref = W[iR, :, iS, :]
            shifted = W[iRpT, :, iSpT, :]

            diff = shifted - ref

            local_abs = np.max(np.abs(diff))
            local_ref = np.max(np.abs(ref))

            if local_abs > max_abs:
                max_abs = local_abs
                worst = {
                    "R": tuple(R),
                    "S": tuple(S),
                    "R_plus_T": RpT,
                    "S_plus_T": SpT,
                }

            max_ref = max(max_ref, local_ref)

    rel_err = max_abs / max(max_ref, 1e-14)
    log.debug("Wannier translation covariance check")
    log.debug("------------------------------------")
    log.debug(f"T_index     = {tuple(T_index)}")
    log.debug(f"max abs err = {max_abs:.3e}")
    log.debug(f"max rel err = {rel_err:.3e}")
    log.debug(f"worst block = {worst}")

    if rel_err < tol:
        log.debug("Wannier translation covariance OK.")
    else:
        log.debug("Wannier translation covariance FAILED.")

    return max_abs, rel_err, worst


def check_wannier_translation(ts, W, R_indices=None,
                              T_index=(1, 0, 0), tol=1e-8):
    '''Backward-compatible alias for check_wannier_orbital_translation.'''
    return check_wannier_orbital_translation(
        ts, W, R_indices=R_indices, T_index=T_index, tol=tol,
    )


def _translation_cell_indices(ts, T_index):
    '''Return the flat-cell permutation generated by a lattice translation.'''
    T_index = np.asarray(T_index, dtype=int)
    return np.asarray([
        ts.R_to_i[ts.mod_index(R + T_index)] for R in ts.R_indices
    ], dtype=int)


def _shift_cell_axes(tensor, cell_axes, translated_cells):
    '''Translate every cell axis of a block tensor by the same vector.'''
    shifted = tensor
    for axis in cell_axes:
        shifted = np.take(shifted, translated_cells, axis=axis)
    return shifted


def _translation_errors(reference, shifted):
    '''Return the maximum absolute and relative translation errors.'''
    max_abs = float(np.max(np.abs(shifted - reference)))
    max_ref = float(np.max(np.abs(reference)))
    rel_err = max_abs / max(max_ref, 1e-14)
    return max_abs, rel_err


def check_h1e_translation(ts, h1e, T_index=(1, 0, 0), tol=1e-8):
    '''
    Check h1[R+T,p,S+T,q] = h1[R,p,S,q].

    ``h1e`` may have flattened shape ``(ncell*norb, ncell*norb)`` or
    cell-block shape ``(ncell, norb, ncell, norb)``.
    '''
    h1e = np.asarray(h1e)
    ncell = ts.ncell
    if h1e.ndim == 2:
        if h1e.shape[0] != h1e.shape[1] or h1e.shape[0] % ncell:
            msg = "h1e must be square with a dimension divisible by ncell"
            raise ValueError(msg)
        
        norb = h1e.shape[0] // ncell
        h1e_blocks = h1e.reshape(ncell, norb, ncell, norb)

    elif h1e.ndim == 4:
        if (h1e.shape[0] != ncell or h1e.shape[2] != ncell
                or h1e.shape[1] != h1e.shape[3]):
            msg = "block h1e must have shape (ncell, norb, ncell, norb)"
            raise ValueError(msg)
        h1e_blocks = h1e
    else:
        raise ValueError("h1e must be a two- or four-dimensional array")

    translated_cells = _translation_cell_indices(ts, T_index)
    shifted = _shift_cell_axes(h1e_blocks, (0, 2), translated_cells)
    max_abs, rel_err = _translation_errors(h1e_blocks, shifted)

    worst = None
    if max_abs > 0:
        block_error = np.max(np.abs(shifted - h1e_blocks), axis=(1, 3))
        iR, iS = np.unravel_index(np.argmax(block_error), block_error.shape)
        worst = {
            "R": tuple(ts.R_indices[iR]),
            "S": tuple(ts.R_indices[iS]),
            "R_plus_T": tuple(ts.R_indices[translated_cells[iR]]),
            "S_plus_T": tuple(ts.R_indices[translated_cells[iS]]),
        }

    log = lib.logger.Logger(ts.cell.stdout, ts.cell.verbose)
    log.debug("One-electron Hamiltonian translation check")
    log.debug(f"T_index     = {tuple(T_index)}")
    log.debug(f"max abs err = {max_abs:.3e}")
    log.debug(f"max rel err = {rel_err:.3e}")
    log.debug(f"worst block = {worst}")
    log.debug("One-electron translation %s.",
             "OK" if rel_err < tol else "FAILED")
    return max_abs, rel_err, worst


def check_h2e_translation(ts, h2e, T_index=(1, 0, 0), tol=1e-8):
    '''Check simultaneous translation of all four h2 cell indices.

    ``h2e`` may have flattened shape ``(ncell*norb,) * 4`` or cell-block
    shape ``(ncell, norb, ncell, norb, ncell, norb, ncell, norb)``.
    '''
    h2e = np.asarray(h2e)
    ncell = ts.ncell
    if h2e.ndim == 4:
        if len(set(h2e.shape)) != 1 or h2e.shape[0] % ncell:
            msg = "flattened h2e must have four equal dimensions divisible by ncell"
            raise ValueError(msg)
        
        norb = h2e.shape[0] // ncell
        block_shape = (ncell, norb) * 4
        h2e_blocks = h2e.reshape(block_shape)
    elif h2e.ndim == 8:
        cell_shape = tuple(h2e.shape[axis] for axis in (0, 2, 4, 6))
        orbital_shape = tuple(h2e.shape[axis] for axis in (1, 3, 5, 7))
        if cell_shape != (ncell,) * 4 or len(set(orbital_shape)) != 1:
            msg = "block h2e must have alternating (ncell, norb) dimensions"
            raise ValueError(msg)
        h2e_blocks = h2e
    else:
        raise ValueError("h2e must be a four- or eight-dimensional array")

    translated_cells = _translation_cell_indices(ts, T_index)
    cell_axes = (0, 2, 4, 6)
    shifted = _shift_cell_axes(h2e_blocks, cell_axes, translated_cells)
    max_abs, rel_err = _translation_errors(h2e_blocks, shifted)

    worst = None
    if max_abs > 0:
        orbital_axes = (1, 3, 5, 7)
        block_error = np.max(np.abs(shifted - h2e_blocks), axis=orbital_axes)
        cells = np.unravel_index(np.argmax(block_error), block_error.shape)
        labels = ("R", "S", "U", "V")
        worst = {}
        for label, cell_index in zip(labels, cells):
            worst[label] = tuple(ts.R_indices[cell_index])
            worst[f"{label}_plus_T"] = tuple(
                ts.R_indices[translated_cells[cell_index]]
            )

    log = lib.logger.Logger(ts.cell.stdout, ts.cell.verbose)
    log.debug("Two-electron Hamiltonian translation check")
    log.debug(f"T_index     = {tuple(T_index)}")
    log.debug(f"max abs err = {max_abs:.3e}")
    log.debug(f"max rel err = {rel_err:.3e}")
    log.debug(f"worst block = {worst}")
    log.debug("Two-electron translation %s.",
             "OK" if rel_err < tol else "FAILED")
    return max_abs, rel_err, worst

def check_wannier_against_ref_cell(ts, W, R_indices, ref_cell=0, tol=1e-8):
    '''
    Check whether every center S is a translated version of ref_cell:

        W[R, mu, S, n] = W[R - (S - S_ref), mu, S_ref, n]
    '''

    R_to_i = ts.index_map(R_indices)

    ncell, nao, ncell2, nwann = W.shape
    assert ncell == ncell2

    S_ref = R_indices[ref_cell]

    max_abs = 0.0
    max_ref = 0.0
    worst = None

    log = lib.logger.Logger(ts.cell.stdout, ts.cell.verbose)
    for iS, S in enumerate(R_indices):
        T = S - S_ref

        for iR, R in enumerate(R_indices):
            R_ref = ts.mod_index(R - T)
            iR_ref = R_to_i[R_ref]

            ref = W[iR_ref, :, ref_cell, :]
            target = W[iR, :, iS, :]

            diff = target - ref

            local_abs = np.max(np.abs(diff))
            local_ref = np.max(np.abs(ref))

            if local_abs > max_abs:
                max_abs = local_abs
                worst = {
                    "target_R": tuple(R),
                    "target_S": tuple(S),
                    "ref_R": R_ref,
                    "ref_S": tuple(S_ref),
                    "T": tuple(T),
                }

            max_ref = max(max_ref, local_ref)

    rel_err = max_abs / max(max_ref, 1e-14)
    log.debug("Wannier reference-cell translation check")
    log.debug("----------------------------------------")
    log.debug(f"ref_cell    = {ref_cell}, S_ref = {tuple(S_ref)}")
    log.debug(f"max abs err = {max_abs:.3e}")
    log.debug(f"max rel err = {rel_err:.3e}")
    log.debug(f"worst block = {worst}")

    if rel_err < tol:
        log.info("All Wannier centers are translated copies of " \
        "the reference center.")
    else:
        log.info("Reference-cell translation check FAILED.")

    return max_abs, rel_err, worst

def make_wannier_matrix(wannier_orb):
    '''
    Converting W[R, mu, S, n] to matrix form, such that 
    it can be used as the mo_coeff.
    args:
    wannier_orb: ndarray W[R, mu, S, n] (ncell, nao, ncell, nwann)
        Wannier orbitals in real space.
    returns:
        wannier_mat: np.ndarray of shape (ncell * nao, ncell * nwann)
        Wannier orbitals in matrix form.
    '''
    ncell, nao, ncell2, nwann = wannier_orb.shape
    assert ncell == ncell2

    # dtype = wannier_orb.dtype
    # wannier_mat = np.zeros((ncell * nao, ncell * nwann), dtype=dtype)
    # for cell1 in range(ncell):
    #     for cell2 in range(ncell):
    #         row = slice(cell1 * nao, (cell1 + 1) * nao)
    #         col = slice(cell2 * nwann, (cell2 + 1) * nwann)
    #         wannier_mat[row, col] = wannier_orb[cell1, :, cell2, :]

    wannier_mat = wannier_orb.reshape(ncell*nao, ncell * nwann)
    return wannier_mat

def make_ovlp_mat_in_wannier_basis(kmf, kmesh):
    '''
    Construct the overlap matrix in the Wannier basis.
    args:
        kmf: pbc.scf object
        kmesh: tuple of integers
    returns:
        ovlp_bvk: ndarray of shape (ncell*norb, ncell*norb)
            Overlap matrix in the Wannier basis.
    '''
    kpts = kmf.kpts
    ts = TranslationSymm(kmf.cell, kmesh, kpts=kpts)
    ovlp_k = kmf.get_ovlp(kpts=kpts)
    nkpts = np.prod(ts.kmesh)
    msg = "Number of k-points in ovlp_k should match the product of kmesh dimensions."
    assert len(ovlp_k) == nkpts == len(kpts), msg
    norb_tot = kmf.cell.nao_nr() * nkpts

    ovlp_k = scipy.linalg.block_diag(*ovlp_k)
    phase = ts.get_k_to_cell_transmat()
    ovlp_wannier = phase.conj().T @ ovlp_k @ phase
    
    assert ovlp_wannier.shape[0] == ovlp_wannier.shape[1] == norb_tot, \
        "shape mismatch, something went wrong in constructing the Wannier overlap matrix."
    return ovlp_wannier

def pack_wannier_orb(wannier_orb, ref_cell=0):
    '''
    Convert W[R, mu, S, n] to W[R, mu, n], by taking S=ref_cell block.
    This is useful for printing the Wannier orbitals in Molden format.
    '''
    ncell, nao, ncell2, nwann = wannier_orb.shape
    assert ncell == ncell2
    assert 0 <= ref_cell < ncell
    wannier_orb = wannier_orb[:, :, ref_cell, :]
    wannier_orb = wannier_orb.reshape(ncell*nao, nwann)
    return wannier_orb

def unpack_wannier_orb(wannier_orb_packed, cell, kmesh, ref_cell=0, 
                       make_wannier_mat=False):
    '''
    Convert W[R, mu, n] to W[R, mu, S, n], by copying the packed block to all S blocks.
    This is the inverse operation of pack_wannier_orb.
    '''
    ts = TranslationSymm(cell, kmesh)
    R_indices = ts.R_indices
    R_to_i = ts.R_to_i
    ncell = len(R_indices)
    dtype = wannier_orb_packed.dtype

    nao = cell.nao_nr()
    assert wannier_orb_packed.ndim == 2
    naoncell, nwann = wannier_orb_packed.shape
    ncell2 = naoncell // nao
    assert ncell == ncell2, "ncell does not match the shape of the packed Wannier orbitals."
    assert 0 <= ref_cell < ncell

    wannier_ref_cell = wannier_orb_packed.reshape(ncell, nao, nwann)
    
    ref_R = R_indices[ref_cell]

    wannier_orb = np.zeros((ncell, nao, ncell, nwann),dtype=dtype)

    for iS, S in enumerate(R_indices):
        # Translation from ref center to target center:
        # T = S - ref_R
        T = S - ref_R
        for iR, R in enumerate(R_indices):
            # W[R, mu, S, n] = W[R - T, mu, ref_cell, n]
            # R_ref = R - S + ref_R
            R_ref = ts.mod_index(R - T)
            iR_ref = R_to_i[R_ref]
            wannier_orb[iR, :, iS, :] = wannier_ref_cell[iR_ref, :, :]
    if make_wannier_mat:
        wannier_orb = wannier_orb.reshape(ncell*nao, ncell*nwann)
    return wannier_orb
