import numpy as np
import scipy
import ctypes
import warnings
from pyscf.lib import logger
from pyscf import lib, ao2mo
from pyscf.fci.addons import _unpack_nelec
from pyscf.fci import direct_nosym, direct_spin1, cistring

libfci = direct_spin1.libfci
_unpack = direct_nosym._unpack
FCIvector = direct_spin1.FCIvector

'''
FCI for the complex Hamiltonian and complex CI vecs
'''

# Author: Bhavnesh Jangid

def contract_1e(h1e, fcivec, norb, nelec, link_index=None):
    '''
    Contract the 1-electron integrals with the CI vector.
    args:
        h1e: np.array (norb, norb), dtype='complex128'
            one-electron integral matrix
        fcivec: np.array (na, nb), dtype='complex128'
            CI vector
        norb: int
            number of active space orbitals
        nelec: tuple (#alpha, #beta)
            number of active electrons
        link_index: tuple of (link_indexa, link_indexb)
            Lookup tables for det interactions
    return
        hc = h*c : np.array (norb, norb), dtype='complex128'
            contracted 1-electron integral matrix
    '''
    link_indexa, link_indexb = _unpack(norb, nelec, link_index)
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]
    assert fcivec.size == na*nb
    if fcivec.dtype == h1e.dtype == np.float64:
        h1e = np.asarray(h1e, order='C')
        fcivec = np.asarray(fcivec, order='C')
        ci1 = np.zeros_like(fcivec)

        libfci.FCIcontract_a_1e_nosym(h1e.ctypes.data_as(ctypes.c_void_p),
                                    fcivec.ctypes.data_as(ctypes.c_void_p),
                                    ci1.ctypes.data_as(ctypes.c_void_p),
                                    ctypes.c_int(norb),
                                    ctypes.c_int(na), ctypes.c_int(nb),
                                    ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                                    link_indexa.ctypes.data_as(ctypes.c_void_p),
                                    link_indexb.ctypes.data_as(ctypes.c_void_p))
        libfci.FCIcontract_b_1e_nosym(h1e.ctypes.data_as(ctypes.c_void_p),
                                    fcivec.ctypes.data_as(ctypes.c_void_p),
                                    ci1.ctypes.data_as(ctypes.c_void_p),
                                    ctypes.c_int(norb),
                                    ctypes.c_int(na), ctypes.c_int(nb),
                                    ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                                    link_indexa.ctypes.data_as(ctypes.c_void_p),
                                    link_indexb.ctypes.data_as(ctypes.c_void_p))
        return ci1.view(FCIvector)
    
    ciR = np.asarray(fcivec.real, order='C')
    ciI = np.asarray(fcivec.imag, order='C')
    h1eR = np.asarray(h1e.real, order='C')
    h1eI = np.asarray(h1e.imag, order='C')
    link_index = (link_indexa, link_indexb)

    outR = contract_1e(h1eR, ciR, norb, nelec, link_index=link_index)
    outI = contract_1e(h1eI, ciI, norb, nelec, link_index=link_index)
    out = outR.astype(np.complex128)
    out.imag = outI
    return out

def contract_2e(eri, fcivec, norb, nelec, link_index=None):
    '''
    Contract the 2-electron Hamiltonian with a FCI vector to get a new FCIvector.
        args:
            eri: np.array (norb, norb, norb, norb), dtype='complex128'
                two electron Hamiltonian (not just ERI)
            fcivec: np.array (na, nb), dtype='complex128'
                CI vector
            norb: int
                number of active space orbitals
            nelec: tuple (#alpha, #beta)
                number of active electrons
            link_index: tuple of (link_indexa, link_indexb)
                Lookup tables for det interactions
        return
            Hc: np.array (na, nb), dtype='complex128'

    See also :func:`direct_nosym.contract_2e`
    '''
    link_indexa, link_indexb = _unpack(norb, nelec, link_index)
    na, nlinka = link_indexa.shape[:2]
    nb, nlinkb = link_indexb.shape[:2]

    assert fcivec.size == na*nb

    if fcivec.dtype == eri.dtype == np.float64:
        eri = np.asarray(eri, order='C')
        fcivec = np.asarray(fcivec, order='C')
        ci1 = np.empty_like(fcivec)
        libfci.FCIcontract_2es1(eri.ctypes.data_as(ctypes.c_void_p),
                                fcivec.ctypes.data_as(ctypes.c_void_p),
                                ci1.ctypes.data_as(ctypes.c_void_p),
                                ctypes.c_int(norb),
                                ctypes.c_int(na), ctypes.c_int(nb),
                                ctypes.c_int(nlinka), ctypes.c_int(nlinkb),
                                link_indexa.ctypes.data_as(ctypes.c_void_p),
                                link_indexb.ctypes.data_as(ctypes.c_void_p))
        return ci1.view(direct_spin1.FCIvector)

    ciR = np.asarray(fcivec.real, order='C')
    ciI = np.asarray(fcivec.imag, order='C')
    eriR = np.asarray(eri.real, order='C')
    eriI = np.asarray(eri.imag, order='C')
    link_index = (link_indexa, link_indexb)
    outR  = contract_2e(eriR, ciR, norb, nelec, link_index=link_index)
    outR -= contract_2e(eriI, ciI, norb, nelec, link_index=link_index)
    outI  = contract_2e(eriR, ciI, norb, nelec, link_index=link_index)
    outI += contract_2e(eriI, ciR, norb, nelec, link_index=link_index)
    out = outR.astype(np.complex128)
    out.imag = outI
    return out

def absorb_h1e(h1e, eri, norb, nelec, fac=1):
    __doc__ = direct_nosym.absorb_h1e.__doc__
    
    assert eri.ndim == 4
    if not isinstance(nelec, (int, np.number)): nelec = sum(nelec)

    h2e = eri.astype(dtype=np.result_type(h1e, eri), copy=True)
    f1e = h1e - np.einsum('jiik->jk', h2e) * .5
    f1e = f1e * (1./(nelec+1e-100))
    
    for k in range(norb):
        h2e[k,k,:,:] += f1e
        h2e[:,:,k,k] += f1e
    
    return h2e * fac

def energy(h1e, eri, fcivec, norb, nelec, link_index=None):
    '''
    Compute the FCI electronic energy for given Hamiltonian and FCI vector.
    '''
    h2e = absorb_h1e(h1e, eri, norb, nelec, .5)
    ci1 = contract_2e(h2e, fcivec, norb, nelec, link_index)
    return np.dot(fcivec.reshape(-1), ci1.reshape(-1))

def make_hdiag(h1e, eri, norb, nelec, compress=False):
    if h1e.dtype == np.complex128: h1e = h1e.real.copy()
    if eri.dtype == np.complex128: eri = eri.real.copy()
    return direct_spin1.make_hdiag(h1e, eri, norb, nelec, compress)

class FCISolver(direct_nosym.FCISolver):
    def __init__(self, *args, **kwargs):
        direct_spin1.FCISolver.__init__(self, *args, **kwargs)
        self.davidson_only = True

    def contract_1e(self, h1e, fcivec, norb, nelec, link_index=None):
        return contract_1e(h1e, fcivec, norb, nelec, link_index)

    def contract_2e(self, eri, fcivec, norb, nelec, link_index=None):
        return contract_2e(eri, fcivec, norb, nelec, link_index)

    def absorb_h1e(self, h1e, eri, norb, nelec, fac=1):
        return absorb_h1e(h1e, eri, norb, nelec, fac)

    def make_hdiag(self, h1e, eri, norb, nelec, compress=False):
        nelec = direct_spin1._unpack_nelec(nelec, self.spin)
        return make_hdiag(h1e, eri, norb, nelec, compress)
    
    def kernel(self, h1e, eri, norb, nelec, ci0=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None,
               orbsym=None, wfnsym=None, ecore=0, **kwargs):
        
        if isinstance(nelec, (int, np.number)):
            nelecb = nelec//2
            neleca = nelec - nelecb
        else:
            neleca, nelecb = nelec

        davidson_only = True
        link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)

        e, c = kernel_ms1(self, h1e, eri, norb, nelec, ci0,
                                       (link_indexa,link_indexb),
                                       tol, lindep, max_cycle, max_space, nroots,
                                       davidson_only, pspace_size, ecore=ecore,
                                       **kwargs)
        self.eci, self.ci = e, c
        return e, c
    
    def eig(self, op, x0=None, precond=None, **kwargs):
        if isinstance(op, np.ndarray):
            self.converged = True
            return scipy.linalg.eigh(op)
        else:
            raise NotImplementedError("Haven't figured out davidson part yet.")



def kernel_ms1(fci, h1e, eri, norb, nelec, ci0=None, link_index=None,
               tol=None, lindep=None, max_cycle=None, max_space=None,
               nroots=None, davidson_only=None, pspace_size=None, hop=None,
               max_memory=None, verbose=None, ecore=0, **kwargs):
 
    if nroots is None: nroots = fci.nroots
    if davidson_only is None: davidson_only = fci.davidson_only
    if pspace_size is None: pspace_size = fci.pspace_size
    if max_memory is None:
        max_memory = fci.max_memory - lib.current_memory()[0]
    log = logger.new_logger(fci, verbose)

    nelec = _unpack_nelec(nelec, fci.spin)
    assert (0 <= nelec[0] <= norb and 0 <= nelec[1] <= norb)

    hdiag = fci.make_hdiag(h1e, eri, norb, nelec, compress=False).ravel()
    num_dets = hdiag.size
    pspace_size = min(num_dets, pspace_size)
    addr = [0]
    pw = pv = None

    # if pspace_size > 0 and norb < 64:
    #     addr, h0 = fci.pspace(h1e, eri, norb, nelec, hdiag, pspace_size)
    #     pw, pv = fci.eig(h0)
    #     pspace_size = len(addr)

    if getattr(fci, 'sym_allowed_idx', None):
        raise NotImplementedError("Symmetry allowed index not implemented")
    else:
        sym_idx = None
        civec_size = num_dets

    if max_memory < civec_size*6*8e-6:
        log.warn('Not enough memory for FCI solver. '
                 'The minimal requirement is %.0f MB', civec_size*60e-6)

    if pspace_size >= civec_size and ci0 is None and not davidson_only:
        if nroots > 1:
            nroots = min(civec_size, nroots)
            civec = np.empty((nroots,civec_size))
            civec[:,addr] = pv[:,:nroots].T
            return pw[:nroots]+ecore, civec
        elif pspace_size == 1 or abs(pw[0]-pw[1]) > 1e-12:
            # Check degeneracy. Degenerated wfn may break point group symmetry.
            # Davidson iteration with a proper initial guess can avoid this problem.
            civec = np.empty(civec_size)
            civec[addr] = pv[:,0]
            return pw[0]+ecore, civec
    pw = pv = h0 = None

    if sym_idx is None:
        precond = fci.make_precond(hdiag)
    else:
        precond = fci.make_precond(hdiag[sym_idx])

    h2e = fci.absorb_h1e(h1e, eri, norb, nelec, .5)
    if hop is None:
        cpu0 = [logger.process_clock(), logger.perf_counter()]
        def hop(c):
            hc = fci.contract_2e(h2e, c, norb, nelec, link_index)
            cpu0[:] = log.timer_debug1('contract_2e', *cpu0)
            return hc.ravel()

    def init_guess():
        if callable(getattr(fci, 'get_init_guess', None)):
            return fci.get_init_guess(norb, nelec, nroots, hdiag)
        else:
            x0 = []
            for i in range(min(len(addr), nroots)):
                x = np.zeros(civec_size)
                x[addr[i]] = 1
                x0.append(x)
            return x0

    if ci0 is None:
        ci0 = init_guess  # lazy initialization to reduce memory footprint
    elif not callable(ci0):
        if isinstance(ci0, np.ndarray):
            ci0 = [ci0.ravel()]
        else:
            ci0 = [x.ravel() for x in ci0]
        if sym_idx is not None and ci0[0].size != civec_size:
            ci0 = [x[sym_idx] for x in ci0]
        # If provided initial guess ci0 are accidentally the eigenvectors of the
        # system, Davidson solver may be failed to find enough roots as it is
        # unable to generate more subspace basis from ci0. Adding vectors so
        # initial guess to help Davidson solver generate enough basis.
        if len(ci0) < nroots:
            ci0.extend(init_guess()[len(ci0):])

    if tol is None: tol = fci.conv_tol
    if lindep is None: lindep = fci.lindep
    if max_cycle is None: max_cycle = fci.max_cycle
    if max_space is None: max_space = fci.max_space
    tol_residual = getattr(fci, 'conv_tol_residual', None)

    ci0 = init_guess() if callable(ci0) else ci0
    ci0 = np.asarray(ci0).reshape(2,2).astype(np.complex128)
    hop = hop(ci0).reshape(2,2)
    with lib.with_omp_threads(fci.threads):
        e, c = fci.eig(hop, ci0, precond, tol=tol, lindep=lindep,
                       max_cycle=max_cycle, max_space=max_space, nroots=nroots,
                       max_memory=max_memory, verbose=log, follow_state=True,
                       tol_residual=tol_residual, **kwargs)
    return e+ecore, c



# To be consistent with PySCF
import sys
def kernel(h1e, eri, norb, nelec, ci0=None, level_shift=1e-3, tol=1e-10,
           lindep=1e-14, max_cycle=50, max_space=12, nroots=1,
           davidson_only=False, pspace_size=400, orbsym=None, wfnsym=None,
           ecore=0, **kwargs):
    return _kfactory(FCISolver, h1e, eri, norb, nelec, ci0, level_shift,
                     tol, lindep, max_cycle, max_space, nroots,
                     davidson_only, pspace_size, ecore=ecore, **kwargs)

def _kfactory(Solver, h1e, eri, norb, nelec, ci0=None, level_shift=1e-3,
              tol=1e-10, lindep=1e-14, max_cycle=50, max_space=12, nroots=1,
              davidson_only=False, pspace_size=400, ecore=0, **kwargs):
    cis = Solver(None)
    cis.level_shift = level_shift
    cis.conv_tol = tol
    cis.lindep = lindep
    cis.max_cycle = max_cycle
    cis.max_space = max_space
    cis.nroots = nroots
    cis.davidson_only = davidson_only
    cis.pspace_size = pspace_size

    unknown = {}
    for k in kwargs:
        if not hasattr(cis, k):
            unknown[k] = kwargs[k]
        setattr(cis, k, kwargs[k])
    if unknown:
        sys.stderr.write('Unknown keys %s for FCI kernel %s\n' %
                         (str(unknown.keys()), __name__))
    e, c = cis.kernel(h1e, eri, norb, nelec, ci0, ecore=ecore, **unknown)
    return e, c

