import inspect

import numpy as np
import ctypes
import scipy
import warnings

from pyscf import lib
from pyscf import __config__
from pyscf.fci import direct_uhf, direct_spin1, cistring
from pyscf.csf_fci.csf import CSFFCISolver as realCSFFCISolver, FCISolver as realFCISolver
from pyscf.csf_fci.csf import unpack_h1e_cs, unpack_h1e_ab, get_init_guess, make_hdiag_csf as make_hdiag_csf_real
from pyscf.csf_fci.csf import _debug_g2e as _debug_g2e_real, pspace as pspace_real, get_init_guess as get_init_guess_real
from pyscf.lib.numpy_helper import tag_array
from pyscf.csf_fci.csfstring import count_all_csfs
from pyscf.csf_fci.csfstring import CSFTransformer

from mrh.my_pyscf.pbc.fci import direct_spin1_cplx

libfci = lib.load_library('libfci')

# Note, that the _unpack function should be called from the direct_spin1_cplx, as the 
# _unpack from direct_spin1 will store the link_idx in the tril format.
_unpack = direct_spin1_cplx._unpack
_unpack_nelec = direct_spin1_cplx._unpack_nelec
_get_init_guess = direct_spin1_cplx._get_init_guess_cplx

# Global variables:
HDIAG_IMAG_TOL = 1e-3
IMAG_NOISE = 1e-12

'''
# Okay Great. Let me implement the CSFsolver with complex Hamiltonian.
# Logic of CSFSolvers in PySCF:
#   1. Init guess in det basis, then transform to CSF basis.
#   2. Contruct the Hamiltonian in det basis (using PySCF infrastructure), then transform to CSF basis.
#   3. Solve the eigenvalue problem (exact or Davidson) in CSF basis.
#   4. Transform the eigenvectors back to det basis.
#   5. For the Davidson solver, the matrix-vector product are computed in the det basis and then transformed 
#      to CSF basis for the Davidson solver.

# Now for the complex CI vec of type (a+ib), to use the CSFSolver, the transformation (det to CSF)
# will be a_csf + i*b_csf = U * (a_det + i*b_det), where U is the det to CSF transformation matrix (Note: this
# matrix is not constructed and stored in the memory). The U matrix is real only.
'''

def get_init_guess(norb, nelec, nroots, hdiag_csf, transformer):
    '''
    Also check the doc: csf_fci.csf.get_init_guess
    Get the initial guess for the FCI calculation in the CSF basis.
    Note: for initial guess, I am using the real part of CI vectors only.
    The imaginary part would be small, since the hdiag_csf should be real-dominated.
    args:
        norb: int
            number of active space orbitals
        nelec: tuple (neleca, nelecb) or int
            number of electrons
        nroots: int
            number of roots
        hdiag_csf: np.ndarray of size (1, ncsf)
            diagonal of the Hamiltonian in the CSF basis
        transformer: CSFTransformer object
            the transformer object that can transform between det and CSF basis.
    returns:
        ciout: list of np.ndarray
            list of initial guess CI vectors in the determinant basis.
    '''
    assert np.iscomplexobj(hdiag_csf), "You are using wrong function for real Hamiltonian"
    cireal = get_init_guess_real(norb, nelec, nroots, hdiag_csf.real, transformer)
    ciout = []
    for c in cireal:
        cout = c.astype(hdiag_csf.dtype)
        cout.real = c
        cout.imag = IMAG_NOISE # Only adding some noise.
        cout /= np.linalg.norm(cout) # Normalizing it.
        ciout.append(cout)
    cireal = None
    return ciout

def make_hdiag_det (fci, h1e, eri, norb, nelec):
    '''
    hdiag = <\psi_I|H_real + i*H_imag|\psi_I> = <\psi_I|H_real|\psi_I> + i*<\psi_I|H_imag|\psi_I>.
    For the Hermitian Hamiltonian, the diagoan elements are real. Still the output array would be
    complex to avoid any datatype bug in any other part of the code.
    args:
        fci: FCISolver object?
            I don't know why it's needed but keeping it consistent with actual csfsolver function.
        h1e: np.ndarray of shape (norb, norb)
            one-electron integrals
        eri: np.ndarray of shape (norb, norb, norb, norb)
            two-electron integrals in chemist's notation
        norb: int
            number of active space orbitals
        nelec: tuple (neleca, nelecb) or int
            number of active space electrons
    returns:
        hdiag: np.ndarray of shape (ndet,)
            diagonal of the Hamiltonian in the determinant basis.
    '''
    assert np.iscomplexobj(h1e) and np.iscomplexobj(eri), \
        "You are using wrong function for real Hamiltonian"
    
    dtype = h1e.dtype
    
    h1ea, h1eb = unpack_h1e_ab(h1e)
    hdiag_real = direct_uhf.make_hdiag(
        [h1ea.real, h1eb.real], 
        [eri.real, eri.real, eri.real], 
        norb, nelec)
    hdiag = hdiag_real.astype(dtype)
    hdiag.real = hdiag_real

    if fci is not None and fci.verbose > lib.logger.DEBUG:
        hdiag_imag  = direct_uhf.make_hdiag(
            [h1ea.imag, h1eb.imag], 
            [eri.imag, eri.imag, eri.imag], 
            norb, nelec)
        if np.abs(hdiag_imag).max() > HDIAG_IMAG_TOL:
            lib.logger.warning("The imaginary part of the Hamiltonian diagonal in the determinant basis "
                               "is not negligible: max imaginary part = %s", np.max(np.abs(hdiag_imag)))
            
    hdiag.imag = IMAG_NOISE

    hdiag_real = hdiag_imag = h1ea = h1eb =None
    
    return hdiag

def make_hdiag_csf (h1e, eri, norb, nelec, transformer, hdiag_det=None, max_memory=None, verbose=lib.logger.INFO):
    '''
    Make the diagonal of the Hamiltonian in the CSF basis. Basically, we have the Hamiltonian
    diagonal in the determinant basis (hdiag_det). We will transform it to the CSF basis.
    args:
        h1e: np.ndarray of shape (norb, norb)
            one-electron integrals
        eri: np.ndarray of shape (norb, norb, norb, norb)
            two-electron integrals in chemist's notation
        norb: int
            number of active space orbitals
        nelec: tuple (neleca, nelecb) or int
            number of active space electrons
        transformer: CSFTransformer object
            the transformer object that can transform between det and CSF basis.
        hdiag_det: np.ndarray of shape (ndet,)
            diagonal of the Hamiltonian in the determinant basis.
        max_memory: float
            maximum memory usage.
        verbose: int
            verbosity level.
    returns:
        hdiag_csf: np.ndarray of shape (ncsf,)
            diagonal of the Hamiltonian in the CSF basis.
    '''
    # Constructs the hdiag_det if it is not provided.
    if hdiag_det is None:
        hdiag_det = make_hdiag_det (None, h1e, eri, norb, nelec)

    assert np.iscomplexobj(h1e) and np.iscomplexobj(hdiag_det), \
        "You are using wrong function for real Hamiltonian"
    dtype = h1e.dtype
    hdiag_csf_real = make_hdiag_csf_real(
        h1e.real, eri.real, norb, nelec, transformer, 
        hdiag_det=hdiag_det.real, max_memory=max_memory)
    
    hdiag_csf = hdiag_csf_real.astype(dtype)
    hdiag_csf.real = hdiag_csf_real

    if verbose > lib.logger.DEBUG:
        hdiag_csf_imag = make_hdiag_csf_real(
            h1e.imag, eri.imag, norb, nelec, transformer, 
            hdiag_det=hdiag_det.imag, max_memory=max_memory)
        if np.abs(hdiag_csf_imag).max() > HDIAG_IMAG_TOL:
            lib.logger.warning("The imaginary part of the Hamiltonian diagonal in the CSF basis "
            "is not negligible: max imaginary part = %s", np.amax(np.abs(hdiag_csf_imag)))

    hdiag_csf.imag = 0
    hdiag_csf_real = hdiag_csf_imag = None
    return hdiag_csf

def _debug_g2e (fci, g2e, eri, norb):
    '''
    Debugging 2e part, I am blindly copying the code from csf/csf.py and applying it for the real
    and imag part. Can it be done more efficiently? Probably yes. 
    '''
    _debug_g2e_real (fci, g2e.real, eri.real, norb)
    _debug_g2e_real (fci, g2e.imag, eri.imag, norb)
    return None

def pspace(fci, h1e, eri, norb, nelec, transformer,
           hdiag_det=None, hdiag_csf=None, npsp=200, max_memory=None):
    '''
    In the pspace Davidson solver, we construct a Hamiltonian in smaller subspace 
    and then those eigenvectos are projected to entire subspace and been used for 
    the Davidson solver. 
    # Total Hamiltonian:
        H = H_real + 1j * H_imag
          = <I|H_real|J> + 1j * <I|H_imag|J>
    args:
        fci: FCISolver object
            the FCI solver object, needed for logging and some other utilities.
        h1e: np.ndarray of shape (norb, norb)
            one-electron integrals
        eri: np.ndarray of shape (norb, norb, norb, norb)
            two-electron integrals in chemist's notation
        norb: int
            number of active space orbitals
        nelec: tuple (neleca, nelecb) or int
            number of active space electrons
        transformer: CSFTransformer object
            the transformer object that can transform between det and CSF basis.
        hdiag_det: np.ndarray of shape (ndet,)
            diagonal of the Hamiltonian in the determinant basis. If not provided, it will be constructed.
        hdiag_csf: np.ndarray of shape (ncsf,)
            diagonal of the Hamiltonian in the CSF basis. If not provided, it will be constructed.
        npsp: int
            number of CSFs in the pspace.
        max_memory: float
            maximum memory usage in MB. If not provided, it will be taken from fci.max_memory.
    returns:
        csf_addr: np.ndarray of shape (npsp,)
            the addresses of the CSFs in the pspace.
        h0: np.ndarray of shape (npsp, npsp)
            the Hamiltonian in the pspace.
    '''
    m0 = lib.current_memory ()[0]
    
    if norb > 63: raise NotImplementedError('norb > 63')
    if max_memory is None: max_memory=fci.max_memory

    # In complex Hamiltonian, I don't think ao2mo.restore works,
    # so I always keep the eri in the 4D format. 
    assert (h1e.dtype == eri.dtype), \
        "h1e and eri must have the same dtype"
    assert h1e.shape == (norb, norb), \
        "h1e must be a square matrix of shape (norb, norb)"
    assert eri.shape == (norb, norb, norb, norb), \
        "eri must be a 4D array of shape (norb, norb, norb, norb)"

    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    neleca, nelecb = _unpack_nelec(nelec)
    h1e = np.ascontiguousarray(h1e)
    nb = cistring.num_strings(norb, nelecb)

    # Compute the diagonal elements in both the determinant and CSF basis.
    if hdiag_det is None:
        hdiag_det = fci.make_hdiag(h1e, eri, norb, nelec)
    if hdiag_csf is None:
        hdiag_csf = fci.make_hdiag_csf(h1e, eri, norb, nelec, 
                                       hdiag_det=hdiag_det, max_memory=max_memory, 
                                       verbose=fci.verbose)
    
    assert hdiag_csf.dtype == hdiag_det.dtype == h1e.dtype

    # For Hermitian Hamiltonians, diagonals elements are dominated by the 
    # real-part, that's why I am using only real-part for ranking/selection.
    
    hdiag_csf_real = hdiag_csf.real
    csf_addr = np.arange(hdiag_csf_real.size, dtype=np.int32)
    
    if transformer.wfnsym is None:
        ncsf_sym = hdiag_csf_real.size
    else:
        idx_sym = transformer.confsym[transformer.econf_csf_mask] == transformer.wfnsym
        ncsf_sym = np.count_nonzero(idx_sym)
        csf_addr = csf_addr[idx_sym]

    if ncsf_sym > npsp:
        try:
            csf_addr = csf_addr[np.argpartition(hdiag_csf_real[csf_addr], npsp - 1)[:npsp]]
        except AttributeError:
            csf_addr = csf_addr[np.argsort(hdiag_csf_real[csf_addr])[:npsp]]

    econf_addr = np.unique(transformer.econf_csf_mask[csf_addr])
    det_addr = np.concatenate([np.nonzero(transformer.econf_det_mask == conf)[0] 
                               for conf in econf_addr])
    
    npsp_det = len(det_addr)
    npsp_csf = len(csf_addr)

    lib.logger.debug(fci, ("csf.pspace: Lowest-energy %s CSFs correspond to %s configurations "
                           "which are spanned by %s determinants"), npsp_csf, econf_addr.size, npsp_det)

    addra, addrb = divmod(det_addr, nb)
    stra = cistring.addrs2str(norb, neleca, addra)
    strb = cistring.addrs2str(norb, nelecb, addrb)

    safety_factor = 1.2
    nfloats_h0 = (npsp_det + npsp_csf) ** 2.0
    mem_h0 = safety_factor * nfloats_h0 * h1e.dtype.itemsize / 1e6
    deltam = lib.current_memory()[0] - m0
    mem_remaining = max_memory - deltam
    memstr = ("pspace_size of {} CSFs -> {} determinants requires {} MB, cf {} MB "
              "remaining memory").format(npsp_csf, npsp_det, mem_h0, mem_remaining)
    
    if mem_h0 > mem_remaining:
        raise MemoryError(memstr)
    
    lib.logger.debug(fci, memstr)

    h1e_ab = unpack_h1e_ab(h1e)
    h1e_a = np.asarray(h1e_ab[0], order='C')
    h1e_b = np.asarray(h1e_ab[1], order='C')
    g2e = np.asarray(eri, order='C')
    g2e_aa = g2e_ab = g2e_bb = g2e

    # Check the g2e for nans/infs.
    _debug_g2e(fci, g2e, eri, norb)
    
    t0 = lib.logger.timer_debug1(fci, "csf.pspace: index manipulation", *t0)

    # Compute the subspace Hamiltonian in the det basis.
    def _build_h0tril(h1e_a, h1e_b, g2e_aa, g2e_ab, g2e_bb):
        h0tril = np.ascontiguousarray(np.zeros((npsp_det, npsp_det), dtype=np.float64, order='C'))
        h1e_a = np.ascontiguousarray(h1e_a)
        h1e_b = np.ascontiguousarray(h1e_b)
        g2e_aa = np.ascontiguousarray(g2e_aa)
        g2e_ab = np.ascontiguousarray(g2e_ab)
        g2e_bb = np.ascontiguousarray(g2e_bb)

        libfci.FCIpspace_h0tril_uhf(
            h0tril.ctypes.data_as(ctypes.c_void_p),
            np.ascontiguousarray(h1e_a,  dtype=np.float64).ctypes.data_as(ctypes.c_void_p),
            np.ascontiguousarray(h1e_b,  dtype=np.float64).ctypes.data_as(ctypes.c_void_p),
            np.ascontiguousarray(g2e_aa, dtype=np.float64).ctypes.data_as(ctypes.c_void_p),
            np.ascontiguousarray(g2e_ab, dtype=np.float64).ctypes.data_as(ctypes.c_void_p),
            np.ascontiguousarray(g2e_bb, dtype=np.float64).ctypes.data_as(ctypes.c_void_p),
            stra.ctypes.data_as(ctypes.c_void_p),
            strb.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(norb), ctypes.c_int(npsp_det))
        return h0tril

    h0real = _build_h0tril(h1e_a.real, h1e_b.real, g2e_aa.real, g2e_ab.real, g2e_bb.real)
    h0 = h0real.astype(h1e.dtype)
    h0.real = h0real
    h0.imag = _build_h0tril(h1e_a.imag, h1e_b.imag, g2e_aa.imag, g2e_ab.imag, g2e_bb.imag)
    # Note, imaginary part of the Hamiltonian will be anti-hermitian.
    # print("Is it antihermitian:", np.allclose(h0.imag, -h0.imag.conj().T)) # True
    h0real = None

    # Fill the diagonal elements.
    # for i in range(npsp_det):
    #     h0.real[i, i] = hdiag_det.real[det_addr[i]]
    #     h0.imag[i, i] = hdiag_det.imag[det_addr[i]]
    h0[np.arange(npsp_det), np.arange(npsp_det)] = hdiag_det[det_addr]

    t0 = lib.logger.timer_debug1(fci, "csf.pspace: pspace Hamiltonian in determinant basis", *t0)

    # # Now fill the upper triangular part.
    # h0.real = lib.hermi_triu(h0.real)
    # h0.imag = lib.hermi_triu(h0.imag, hermi=2)
    h0real = np.ascontiguousarray(h0.real)
    h0imag = np.ascontiguousarray(h0.imag)
    h0 = h0.astype(h1e.dtype)
    h0.real = lib.hermi_triu(h0real)
    h0.imag = lib.hermi_triu(h0imag, hermi=2) # 2 For Anti-Hermitian part.
    h0real = h0imag = None
    
    # Sanity Checks CSF transformations
    try:
        if fci.verbose > lib.logger.DEBUG1:
            evals_before = scipy.linalg.eigh(h0)[0]
    except ValueError as e:
        lib.logger.debug1(fci,("ERROR: h0 has {} infs, {} nans; h1e_a has {} infs, {} nans; "
            "h1e_b has {} infs, {} nans; g2e has {} infs, {} nans, norb = {}, npsp_det = {}").format(
            np.count_nonzero(np.isinf(h0)), np.count_nonzero(np.isnan(h0)),
            np.count_nonzero(np.isinf(h1e_a)), np.count_nonzero(np.isnan(h1e_a)),
            np.count_nonzero(np.isinf(h1e_b)), np.count_nonzero(np.isnan(h1e_b)),
            np.count_nonzero(np.isinf(g2e)), np.count_nonzero(np.isnan(g2e)),
            norb, npsp_det))
        evals_before = np.zeros(npsp_det)
        raise (e) from None

    # It's time to transform determinant basis h0 to CSF basis
    h0csf_real, csf_addr = transformer.mat_det2csf_confspace(h0.real, econf_addr)
    h0csf_imag, csf_addr_temp = transformer.mat_det2csf_confspace(h0.imag, econf_addr)
    h0 = None
    h0 = h0csf_real.astype(h1e.dtype)
    h0.real = h0csf_real
    h0.imag = h0csf_imag

    # Sanity Check
    assert np.array_equal(csf_addr, csf_addr_temp), \
        "Real and Imaginary part transformation resulted in different CSF "\
        "addresses; There might be some problem"
    csf_addr_temp = h0csf_real = h0csf_imag = None

    t0 = lib.logger.timer_debug1(fci, "csf.pspace: transform pspace Hamiltonian into CSF basis", *t0)

    # Sanity Checks after the transformation
    if fci.verbose > lib.logger.DEBUG1:
        lib.logger.debug1(fci, "csf.pspace: eigenvalues of h0 before transformation %s", evals_before)
        evals_after = scipy.linalg.eigh(h0)[0]
        lib.logger.debug1(fci, "csf.pspace: eigenvalues of h0 after transformation %s", evals_after)
        idx = [np.argmin(np.abs(evals_before - ev)) for ev in evals_after]
        resid = evals_after - evals_before[idx]
        lib.logger.debug1(fci, "csf.pspace: best h0 eigenvalue matching differences after transformation: %s", resid)
        lib.logger.debug1(fci, "csf.pspace: if the transformation of h0 worked the following number " \
        "will be zero: %s", np.max(np.abs(resid)))

    lib.logger.debug1(fci, "csf_solver.pspace: asked for %s-CSF pspace; found %s CSFs", 
                      csf_addr.size, npsp_csf)
    
    if csf_addr.size > npsp_csf:
        h0diag_real = np.diag(h0).real
        try:
            csf_addr_2 = np.argpartition(h0diag_real, npsp_csf - 1)[:npsp_csf]
        except AttributeError:
            csf_addr_2 = np.argsort(h0diag_real)[:npsp_csf]
        csf_addr = csf_addr[csf_addr_2]
        h0 = h0[np.ix_(csf_addr_2, csf_addr_2)]

    t0 = lib.logger.timer_debug1(fci, "csf.pspace wrapup", *t0)
    return csf_addr, h0

# The Davidson solver is used for iterative solving the eigen vectos for the a large sparse matrix.
# Initially, a few guess vectors are provided, then in each iteration, the matrix-vector product is computed and then
# the subspace Hamiltonian is constructed and diagonalized to get the Ritz values and Ritz vectors, then the residual is 
# computed and preconditioned to get the new guess vector for the next iteration. Till the convergence is reached.
# Alternatively, in QC, there is pspace Davidson solver, where the Hamiltonian is constructed in a smaller subspace exactly
# and then those eigenvectors are projected to entire subspace and been used for the Davidson solver.
# Below I am defining the preconditioners for the Davidson solvers.

def make_diag_precond(hdiag, level_shift=1e-3):
    '''
    Diagonal preconditioner for the Davidson solver.
    '''
    hdiag = np.asarray(hdiag)
    if np.max(np.abs(hdiag.imag)) > HDIAG_IMAG_TOL:
        msg = ("The diagonal elements of the Hamiltonian have non-negligible "
            f"imaginary parts: max |Im(hdiag)| = {np.max(np.abs(hdiag.imag))}.")
        warnings.warn(msg)
    def precond(dx, e, *args):
        diagd = hdiag - (np.real(e) - level_shift)
        diagd = diagd.copy()
        diagd[np.abs(diagd) < 1e-8] = 1e-8
        return dx / diagd
    return precond

def make_pspace_precond(hdiag, pspaceig, pspaceci, addr, level_shift=0):
    '''
    Code adapated from the pyscf/fci/direct_spin1.py, but with some modification 
    to handle the complex Hamiltonian.
    args:
        hdiag: np.ndarray of shape (ndet,)
            diagonal of the Hamiltonian in the determinant basis.
        pspaceig: np.ndarray of shape (npsp,)
            eigenvalues of the Hamiltonian in the pspace.
        pspaceci: np.ndarray of shape (npsp, npsp)
            eigenvectors of the Hamiltonian in the pspace.
        addr: np.ndarray of shape (npsp,)
            the addresses of the determinants in the pspace.
        level_shift: float
            level shift for the preconditioner, default is 0.
    returns:
        precond: function
            the preconditioner function that can be used in the Davidson solver.
    '''
    hdiag = np.asarray(hdiag)
    pspaceig = np.asarray(pspaceig)
    pspaceci = np.asarray(pspaceci)
    addr = np.asarray(addr)

    def precond(r, e0, x0, *args):
        r = np.asarray(r)
        x0 = np.asarray(x0)
        e0r = np.real(e0)

        # Full-space diagonal inverse
        denom = hdiag - (e0r - level_shift)
        denom = denom.astype(hdiag.dtype, copy=True)
        
        # For very small denominators, lets set them to a small number to 
        # avoid numerical instability. 
        denom[np.abs(denom) < 1e-8] = 1e-8 + 0j

        hdiaginv = 1.0 / denom
        hdiaginv[np.abs(hdiaginv) > 1e8] = 1e8

        # P-space inverse in eigenbasis:
        pdenom = pspaceig - (e0r - level_shift)
        pdenom = np.asarray(pdenom, dtype=pspaceig.dtype).copy()
        pdenom[np.abs(pdenom) < 1e-8] = 1e-8
        h0e0inv = np.dot(pspaceci / pdenom, pspaceci.conj().T)

        # Apply preconditioner to x0 and residual
        h0x0 = x0 * hdiaginv
        h0x0[addr] = np.dot(h0e0inv, x0[addr])

        h0r = r * hdiaginv
        h0r[addr] = np.dot(h0e0inv, r[addr])

        # For complex number innder product, using vdot instead of dot.
        denom_e1 = np.vdot(x0, h0x0)
        if abs(denom_e1) < 1e-14: e1 = 0.0
        else: e1 = np.vdot(x0, h0r) / denom_e1
        x1 = r - e1 * x0
        x1 *= hdiaginv
        return x1
    return precond

def kernel(fci, h1e, eri, norb, nelec, smult=None, idx_sym=None, ci0=None,
           tol=None, lindep=None, max_cycle=None, max_space=None,
           nroots=None, davidson_only=None, pspace_size=None, max_memory=None,
           orbsym=None, wfnsym=None, ecore=0, transformer=None, **kwargs):
    '''
    kernel function.
    '''
    t0 = (lib.logger.process_clock (), lib.logger.perf_counter ())
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
        kwargs.pop ('verbose')
    else:
        verbose = lib.logger.Logger (stdout=fci.stdout, verbose=fci.verbose)
    
    # I think we should do the sanity check always:
    fci.check_sanity()

    if nroots is None: nroots = fci.nroots
    if pspace_size is None: pspace_size = fci.pspace_size
    if davidson_only is None: davidson_only = fci.davidson_only
    if transformer is None: transformer = fci.transformer
    if max_memory is None: max_memory = fci.max_memory

    nelec = neleca, nelecb = _unpack_nelec(nelec, fci.spin)
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: throat-clearing", *t0)
    hdiag_det = fci.make_hdiag (h1e, eri, norb, nelec)
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: hdiag_det", *t0)
    hdiag_csf = fci.make_hdiag_csf (h1e, eri, norb, nelec, hdiag_det=hdiag_det, 
                                    max_memory=max_memory)
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: hdiag_csf", *t0)
    
    ncsf_all = count_all_csfs (norb, neleca, nelecb, smult)
    if idx_sym is None: ncsf_sym = ncsf_all
    else: ncsf_sym = np.count_nonzero (idx_sym)
    nroots = min(ncsf_sym, nroots)
    if nroots is not None:
        assert (ncsf_sym >= nroots), "Can't find {} roots among only {} CSFs".format (nroots, ncsf_sym)
    
    link_indexa, link_indexb = _unpack(norb, nelec, None, spin=0)

    na = link_indexa.shape[0]
    nb = link_indexb.shape[0]
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: throat-clearing", *t0)
    addr, h0 = fci.pspace(h1e, eri, norb, nelec, idx_sym=idx_sym, hdiag_det=hdiag_det, hdiag_csf=hdiag_csf,
                        npsp=max(pspace_size,nroots))
    lib.logger.debug1 (fci, 'csf.kernel: error of hdiag_csf: %s', np.amax (np.abs (hdiag_csf[addr]-np.diag (h0))))
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: make pspace", *t0)

    # Making sure h0 is Hermitian.
    h0 = 0.5 * (h0 + h0.conj().T)
    if pspace_size > 0:
        pw, pv = fci.eig(h0)
    else:
        pw = pv = None

    dtype = h1e.dtype

    if pspace_size >= ncsf_sym and not davidson_only:
        if ncsf_sym == 1:
            civecreal = transformer.vec_csf2det (pv[:,0].real.reshape (1,1))
            civec = civecreal.astype(dtype)
            civec.real = civecreal
            civec.imag = transformer.vec_csf2det (pv[:,0].imag.reshape (1,1))
            civecreal = None
            return pw[0]+ecore, civec
        elif nroots > 1:
            civeccsf = np.empty((nroots,ncsf_sym), dtype=dtype)
            civeccsf[:,:] = pv[:,:nroots].T # Should I take the conj here?: I think no.
            civecreal = transformer.vec_csf2det (civeccsf.real)
            civec = civecreal.astype(dtype)
            civec.real = civecreal
            civec.imag = transformer.vec_csf2det (civeccsf.imag)
            civecreal = None
            return pw[:nroots]+ecore, [c.reshape(na,nb) for c in civec]
        
        elif abs(pw[0]-pw[1]) > 1e-12:
            civeccsf = np.empty((ncsf_sym), dtype=dtype)
            civeccsf[:] = pv[:,0]
            civecreal = transformer.vec_csf2det (civeccsf.real, normalize=False)
            civec = civecreal.astype(dtype)
            civec.real = civecreal
            civec.imag = transformer.vec_csf2det (civeccsf.imag, normalize=False)
            civecreal = None
            civec /= np.linalg.norm(civec)
            return pw[0]+ecore, civec.reshape(na,nb)
        return None

    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: throat-clearing", *t0)
    if idx_sym is None:
        precond = fci.make_precond(hdiag_csf, pw, pv, addr)
    else:
        addr_bool = np.zeros (ncsf_all, dtype=np.bool_)
        addr_bool[addr] = True
        precond = fci.make_precond(hdiag_csf[idx_sym], pw, pv, addr_bool[idx_sym])

    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: make preconditioner", *t0)
    h2e = fci.absorb_h1e(h1e, eri, norb, nelec, 0.5)
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: h2e", *t0)
    
    def hop(x):
        x_det = (transformer.vec_csf2det(x.real, normalize=False)
                + 1j * transformer.vec_csf2det(x.imag, normalize=False))
        x_det /= np.linalg.norm(x_det)
        hx = fci.contract_2e(h2e, x_det, norb, nelec, (link_indexa, link_indexb))
        hx_out = (transformer.vec_det2csf(hx.real, normalize=False).ravel()
                + 1j * transformer.vec_det2csf(hx.imag, normalize=False).ravel())
        return hx_out.ravel()

    # def hop(x):
    #     x_detreal = transformer.vec_csf2det (x.real)
    #     x_det = x_detreal.astype(x.dtype)
    #     x_det.real = x_detreal
    #     x_det.imag = transformer.vec_csf2det (x.imag)
    #     hx = fci.contract_2e(h2e, x_det, norb, nelec, (link_indexa,link_indexb))
    #     hx_real = transformer.vec_det2csf (hx.real, normalize=False).ravel ()
    #     hx_imag = transformer.vec_det2csf (hx.imag, normalize=False).ravel ()
    #     hx_out = hx_real.astype(x.dtype)
    #     hx_out.real = hx_real
    #     hx_out.imag = hx_imag
    #     hx_real = hx_imag = x_detreal = x_det = None
    #     return hx_out.ravel()
    
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: make hop", *t0)

    if ci0 is None:
        if hasattr(fci, 'get_init_guess'):
            def ci0 ():
                ci0_det = fci.get_init_guess(norb, nelec, nroots, hdiag_csf)
                dtype = ci0_det[0].dtype
                ci0_csfout = []
                for c in ci0_det:
                    ci0_csfreal = transformer.vec_det2csf (c.real)
                    ci0_csf = ci0_csfreal.astype(dtype)
                    ci0_csf.real = ci0_csfreal
                    ci0_csf.imag = 1e-8 #transformer.vec_det2csf (ci0_det.imag)
                    ci0_csf /= np.linalg.norm(ci0_csf)
                    ci0_csfout.append(ci0_csf)
                return ci0_csfout
        else:
            def ci0():
                x0 = []
                for i in range(nroots):
                    x = np.zeros(ncsf_sym, dtype=h1e.dtype)
                    x[addr[i]] = 1.0 + 1e-10j
                    x0.append(x)
                return x0
    else:
        if isinstance(ci0, np.ndarray) and ci0.size == na*nb:
            ci0real = transformer.vec_det2csf (ci0.real.ravel ())
            ci0imag = transformer.vec_det2csf (ci0.imag.ravel ())
            ci0_out = np.asarray(ci0real, dtype=ci0.dtype)
            ci0_out.real = ci0real
            ci0_out.imag = ci0imag
            ci0real = ci0imag = None
            ci0 = [ci0_out]
        else:
            nrow = len (ci0)
            def to_csf_vec (ci0):
                ci0 = np.asarray (ci0).reshape (nrow, -1, order='C')
                ci0 = np.ascontiguousarray (ci0)
                if nrow==1: ci0 = ci0[0]
                ci0 = transformer.vec_det2csf (ci0)
                ci0 = np.asarray(ci0).reshape(nrow, -1)
                return [c for c in ci0]
            
            ci0real = to_csf_vec (ci0.real)
            ci0imag = to_csf_vec (ci0.imag)
            ci0_out = []
            for r, im in zip(ci0real, ci0imag):
                c = np.asarray(r, dtype=np.complex128)
                c.real = r
                c.imag = im
                ci0_out.append(c)

            ci0 = ci0_out
    
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: ci0 handling", *t0)

    if tol is None: tol = fci.conv_tol
    if lindep is None: lindep = fci.lindep
    if max_cycle is None: max_cycle = fci.max_cycle
    if max_space is None: max_space = fci.max_space
    tol_residual = getattr(fci, 'conv_tol_residual', None)

    e, c = fci.eig(hop, ci0, precond, tol=tol, lindep=lindep,
                    max_cycle=max_cycle, max_space=max_space, nroots=nroots,
                    max_memory=max_memory, verbose=verbose, follow_state=True,
                    tol_residual=tol_residual, **kwargs)
    
    dtype = c.dtype

    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: running fci.eig", *t0)
    if nroots > 1:
        cout = []
        for ciroot in c:
            creal = transformer.vec_csf2det (ciroot.real, order='C', normalize=False)
            cimag = transformer.vec_csf2det (ciroot.imag, order='C', normalize=False)
            croot = creal.astype(dtype)
            croot.real = creal
            croot.imag = cimag
            croot /= np.linalg.norm(croot)
            cout.append(croot)
        creal = cimag = None
    else:
        creal = transformer.vec_csf2det (c.real, order='C', normalize=False)
        cimag = transformer.vec_csf2det (c.imag, order='C', normalize=False)
        cout = creal.astype(dtype)
        cout.real = creal
        cout.imag = cimag
        cout /= np.linalg.norm(cout)
        creal = cimag = None
        
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: transforming final ci vector", *t0)
    if nroots > 1:
        return e+ecore, [ci.reshape(na,nb) for ci in cout]
    else:
        return e+ecore, cout.reshape(na,nb)
    
class cplxCSFFCISolver:
    '''
    Parent class for the complex FCI solver in CSF basis. This class will implement the 
    necessary functions. 
    # Borrowing functions from the real CSF solver. Only modifying the functions
    # which are directly needed.
    # This class won't be of any use for standalone.
    '''
    _keys = {'smult', 'transformer'}
    pspace_size = getattr(__config__, 'fci_csf_FCI_pspace_size', 200)
    make_hdiag = make_hdiag_det

    def __init__ (self, mol, smult, **args):
        self.mol = mol
        self.smult = smult
        self.transformer = None
        super().__init__ (**args)

    def make_hdiag_csf(self, h1e, eri, norb, nelec, hdiag_det=None, smult=None, max_memory=None):
        self.norb = norb
        self.nelec = nelec
        if smult is not None:
            self.smult = smult
        self.check_transformer_cache ()
        max_memory = max_memory if max_memory is not None else self.max_memory
        return make_hdiag_csf(h1e, eri, norb, nelec, self.transformer, hdiag_det=hdiag_det, 
                                   max_memory=max_memory)

    def absorb_h1e(self, h1e, eri, norb, nelec, fac=1):
        h1e_c, h1e_s = unpack_h1e_cs(h1e)
        h2eff = super().absorb_h1e(h1e_c, eri, norb, nelec, fac=fac)
        if h1e_s is not None:
            h2eff = tag_array(h2eff, h1e_s=h1e_s)
        return h2eff
    
    log_transformer_cache = realCSFFCISolver.log_transformer_cache
    print_transformer_cache = realCSFFCISolver.print_transformer_cache

    def contract_2e(self, eris, fcivec, norb, nelec, link_index=None, **kwargs):
        hc = super().contract_2e(eris, fcivec, norb, nelec, link_index=None, **kwargs)
        if hasattr(eris, 'h1e_s'):
            hc_real = direct_uhf.contract_1e ([eris.h1e_s.real, -eris.h1e_s.real], fcivec.real, norb, nelec, link_index)
            hc_real -= direct_uhf.contract_1e ([eris.h1e_s.imag, -eris.h1e_s.imag], fcivec.imag, norb, nelec, link_index)
            hc.real += hc_real
            hc_real = None
            hc_imag = direct_uhf.contract_1e ([eris.h1e_s.real, -eris.h1e_s.real], fcivec.imag, norb, nelec, link_index)
            hc_imag += direct_uhf.contract_1e ([eris.h1e_s.imag, -eris.h1e_s.imag], fcivec.real, norb, nelec, link_index)
            hc.imag += hc_imag
            hc_imag = None
        return hc
    
    def pspace (self, h1e, eri, norb, nelec, hdiag_det=None, hdiag_csf=None, npsp=200, **kwargs):
        self.norb = norb
        self.nelec = nelec
        self.smult = kwargs.pop('smult', self.smult)
        self.check_transformer_cache ()
        max_memory = kwargs.get ('max_memory', self.max_memory)
        return pspace (self, h1e, eri, norb, nelec, self.transformer, hdiag_det=hdiag_det,
                       hdiag_csf=hdiag_csf, npsp=npsp, max_memory=max_memory)

# Good chance to learn Inheritance, and MRO Method:
class FCISolver(cplxCSFFCISolver, direct_spin1_cplx.FCISolver):
    '''
    Complex FCI in CSFSolver. 
    '''

    def get_init_guess(self, norb, nelec, nroots, hdiag_csf, **kwargs):
        '''
        Get the initial guess for the FCI calculation in the CSF basis.
        '''
        self.norb = norb
        self.nelec = nelec
        self.check_transformer_cache ()
        return get_init_guess (norb, nelec, nroots, hdiag_csf, self.transformer)

    # Needed to replace the pspace preconditioner with the one suitable for complex Hamiltonian.
    def make_precond(self, hdiag, pspaceig=None, pspaceci=None, addr=None):
        if pspaceig is None:
            return make_diag_precond(hdiag, pspaceig, pspaceci, addr,
                                     self.level_shift)
        else:
            # Note: H0 in pspace may break symmetry.
            return make_pspace_precond(hdiag, pspaceig, pspaceci, addr,
                                       self.level_shift)
        
    def kernel(self, h1e, eri, norb, nelec, ci0=None, **kwargs):
        self.norb = norb
        self.nelec = nelec
        self.smult = kwargs.pop('smult', self.smult)
        self.check_transformer_cache ()
        self.log_transformer_cache (lib.logger.DEBUG)

        e, c = kernel (self, h1e, eri, norb, nelec, smult=self.smult,
                       idx_sym=None, ci0=ci0, transformer=self.transformer,
                       verbose=lib.logger.DEBUG,
                       **kwargs)

        self.eci, self.ci = e, c
        return e, c

    check_transformer_cache = realFCISolver.check_transformer_cache

    
if __name__ == "__main__":
    from pyscf import gto, fci, scf, lib, ao2mo
    from pyscf.csf_fci import csf_solver
    from mrh.my_pyscf.pbc.fci import direct_spin1_cplx
    mol = gto.Mole(atom='H 0 0 0; H 0 0 1.1', basis='STO-6G',
                verbose=0) #lib.logger.DEBUG)
    mol.build ()
    mf = scf.RHF (mol).run ()
    h0 = mf.energy_nuc ()
    h1 = mf.mo_coeff.conj ().T @ mf.get_hcore () @ mf.mo_coeff
    h2 = ao2mo.restore (1, ao2mo.full (mf._eri, mf.mo_coeff), mol.nao_nr ())
    cisolver = csf_solver (mol, smult=1)

    norb = mol.nao_nr ()
    nelec = (1, 1)
    
    def real_CSFSolver(h0, h1, h2, norb, nelec, davidson_only=False, pspace_size=400):
        real_cisolver = csf_solver (mol, smult=1)
        real_cisolver.pspace_size = pspace_size
        real_cisolver.davidson_only = davidson_only
        real_cisolver.max_cycle = 50
        e, c = real_cisolver.kernel (h1, h2, norb, nelec, ecore=h0)
        e_tot = real_cisolver.energy(h1, h2, c, norb, nelec)
        assert abs(e_tot+h0-e) < 1e-6, "Energy doesn't match the expected value; "
        print("CSF FCI Converged?", real_cisolver.converged, 
              "Norm of wave function: ", np.linalg.norm(c), 
              "Total energy: ", e)
        return e
    
    # Generate complex Hermitian Hamiltonian
    noise = 1e-4
    h0cplx = h0 + noise * 1j   
    h1cplx = h1 + noise * 1j
    h1cplx = 0.5*(h1cplx + h1cplx.T.conj())

    h2cplx = h2 + noise * 1j
    h2cplx = 0.5*(h2cplx + h2cplx.transpose(2, 3, 0, 1))
    h2cplx = 0.5*(h2cplx + h2cplx.transpose(1, 0, 3, 2).conj())
    h2cplx = 0.5*(h2cplx + h2cplx.transpose(3, 2, 1, 0).conj())

    def cplx_CSFSolver(h0, h1cplx, h2cplx, norb, nelec, davidson_only=False, pspace_size=400):
        cplx_cisolver = FCISolver (mol, smult=1)
        cplx_cisolver.davidson_only = davidson_only
        cplx_cisolver.pspace_size = pspace_size
        cplx_cisolver.max_cycle = 50
        e_cplx, c_cplx = cplx_cisolver.kernel (h1cplx, h2cplx, norb, nelec, ecore=h0)
        print("Done with FCI")
        e_tot = cplx_cisolver.energy(h1cplx, h2cplx, c_cplx, norb, nelec)
        assert abs(e_tot+h0-e_cplx) < 1e-6, "Energy doesn't match the expected value; "
        print("Complex CSF FCI Converged?", cplx_cisolver.converged, 
            "Norm of wave function: ", np.linalg.norm(c_cplx), 
            "Total energy: ", e_cplx)
        return e_cplx
    
    def cplx_detSolver(h0, h1cplx, h2cplx, norb, nelec, davidson_only=False, pspace_size=400):
        cplx_cisolver_det = direct_spin1_cplx.FCISolver ()
        cplx_cisolver_det.davidson_only = davidson_only
        cplx_cisolver_det.pspace_size = pspace_size

        e_cplx_det, c_cplx_det = cplx_cisolver_det.kernel (h1cplx, h2cplx, norb, nelec, ecore=h0)
        e_tot = cplx_cisolver_det.energy(h1cplx, h2cplx, c_cplx_det, norb, nelec)
        assert abs(e_tot+h0-e_cplx_det) < 1e-6, "Energy doesn't match the expected value; "
        print("Complex Determinant FCI Converged?", cplx_cisolver_det.converged, 
              "Norm of wave function: ", np.linalg.norm(c_cplx_det), 
              "Total energy: ", e_cplx_det)
        return e_cplx_det
    
    print("Running real CSF FCI solver...")
    e_real_csf = real_CSFSolver(h0, h1, h2, norb, nelec, davidson_only=True, pspace_size=400)
    print("\nRunning complex CSF FCI solver...")
    e_cplx_csf = cplx_CSFSolver(h0cplx, h1cplx, h2cplx, norb, nelec, davidson_only=True, pspace_size=400)
    print("\nRunning complex Determinant FCI solver...")
    e_cplx_det = cplx_detSolver(h0cplx, h1cplx, h2cplx, norb, nelec, davidson_only=True, pspace_size=400)
    print("\nSummary of results:")
    print(f"Real CSF FCI Energy   : {e_real_csf:.12f}")
    print(f"Complex CSF FCI Energy: {e_cplx_csf.real:.12f}")
    print(f"Complex Det FCI Energy: {e_cplx_det.real:.12f}")
    