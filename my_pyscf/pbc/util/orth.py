# !/usr/bin/env python
import numpy as np
import scipy

# Author: Bhavnesh Jangid

'''
Orbital orthogonalization helpers for periodic systems.
'''

def orthogonality_check(mo_coeff, ovlp, tol=1e-8):
    '''
    Orthoganlity check for given set of the mo_coeff.
    '''
    # assert np.asarray(mo_coeff).shape == np.asarray(ovlp).shape
    if np.asarray(mo_coeff).ndim == 3:
        for k, (mo_k, ovlp_k) in enumerate(zip(mo_coeff, ovlp)):
            s = mo_k.conj().T @ ovlp_k @ mo_k
            assert np.allclose(s, np.eye(s.shape[0]), atol=tol), \
                f'k-point {k}: max|S - I| = {np.max(np.abs(s - np.eye(s.shape[0])))}'
    else:
        s = mo_coeff.conj().T @ ovlp @ mo_coeff
        assert np.allclose(s, np.eye(s.shape[0]), atol=tol), \
            f'max|S - I| = {np.max(np.abs(s - np.eye(s.shape[0])))}'


def lowdin_sym(s, tol=1e-15):
    '''
    Hermitian symmetrization:
    '''
    e, v = scipy.linalg.eigh(s)
    idx = e > tol
    return np.dot(v[:,idx]/np.sqrt(e[idx]), v[:,idx].conj().T)


def meta_lowdin_orbitals(cell, ovlp):
    '''
    Get the meta-lowdin orthogonalized orbitals.
    '''
    from pyscf import lo
    nkpts = len(ovlp)
    lo_coeff = np.array([lo.orth.orth_ao(cell, 'meta-lowdin', s=ovlp[k])
                        for k in range(nkpts)])
    # Check the orthogonality of the lowdin orbitals.
    orthogonality_check(lo_coeff, ovlp)
    return lo_coeff
