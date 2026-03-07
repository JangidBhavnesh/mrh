import numpy as np

from pyscf import lib
from pyscf import __config__
from pyscf.fci import direct_uhf, direct_spin1
from pyscf.csf_fci.csf import CSFFCISolver as realCSFFCISolver
from pyscf.csf_fci.csf import unpack_h1e_cs, get_init_guess, make_hdiag_csf as make_hdiag_csf_real
from pyscf.csf_fci.csf import _debug_g2e as _debug_g2e_real
from pyscf.lib.numpy_helper import tag_array
from pyscf.csf_fci.csfstring import count_all_csfs, get_spin_evecs

from mrh.my_pyscf.pbc.fci import direct_spin1_cplx

_unpack_nelec = direct_spin1_cplx._unpack_nelec
unpack_h1e_ab = direct_spin1_cplx.unpack_h1e_ab

_get_init_guess = direct_spin1_cplx.get_init_guess
'''
# Okay Great. Let me implement the CSFsolver with complex Hamiltonian.
# Logic of CSFSolvers in PySCF:
#   1. Init guess in det basis, then transform to CSF basis.
#   2. Contruct the Hamiltonian in det basis (using PySCF infrastructure), then transform to CSF basis.
#   3. Solve the eigenvalue problem (exact or Davidson) in CSF basis.
#   4. Transform the eigenvectors back to det basis.

# Now for the complex CI vec of type (a+ib), to use the CSFSolver, the transformation
# will be a_csf + i*b_csf = U * (a_det + i*b_det), where U is the det to CSF transformation matrix (not this is
# matrix is not constructed and stored in the memory). The U matrix is real only.
'''

@lib.with_doc(get_init_guess.__doc__)
def get_init_guess(norb, nelec, nroots, hdiag_csf, transformer):
    '''
    Get the initial guess for the FCI calculation in the CSF basis
    '''
    ncsf_sym = transformer.ncsf
    assert (ncsf_sym >= nroots), "Can't find {} roots among only {} CSFs of symmetry {}".format (
        nroots, ncsf_sym, transformer.wfnsym)
    hdiag_csf_real = transformer.pack_csf (hdiag_csf.real)
    hdiag_csf_imag = transformer.pack_csf (hdiag_csf.imag)
    hdiag_csf = hdiag_csf_real + 1j * hdiag_csf_imag
    hdiag_csf_real = hdiag_csf_imag = None
    ci0 = _get_init_guess(ncsf_sym, 1, nroots, hdiag_csf, nelec)
    assert ci0[0].dtype == hdiag_csf[0].dtype == np.complex128
    ci0out = ci0.astype(np.complex128)
    ci0out.real = transformer.vec_csf2det (ci0.real)
    ci0out.imag = transformer.vec_csf2det (ci0.imag)
    ci0 = None
    return ci0out

def make_hdiag_det (fci, h1e, eri, nrob, nelec):
    '''
    hdiag = <\psi_I|H_real + i*H_imag|\psi_I> = <\psi_I|H_real|\psi_I> + i*<\psi_I|H_imag|\psi_I> 
    '''
    h1ea, h1eb = unpack_h1e_ab(h1e)
    hdiag = direct_uhf.make_hdiag([h1ea.real, h1eb.real], [eri.real, eri.real, eri.real], nrob, nelec)
    hdiag_out = hdiag.astype(np.complex128)
    hdiag_out.real = hdiag
    hdiag_out.imag  = direct_uhf.make_hdiag([h1ea.imag, h1eb.imag], [eri.imag, eri.imag, eri.imag], nrob, nelec)
    hdiag = None
    return hdiag_out

def make_hdiag_csf (h1e, eri, norb, nelec, transformer, hdiag_det=None, max_memory=None):
    '''
    Make the diagonal of the Hamiltonian in the CSF basis. Basically, we have the Hamiltonian
    diagonal in the determinant basis (hdiag_det). We will transform the real and imaginary parts of the Hamiltonian
    separately to get the Hamiltonian diagonal in the CSF basis.
    '''
    if hdiag_det is None:
        hdiag_det = make_hdiag_det (None, h1e, eri, norb, nelec)

    hdiag_csf = make_hdiag_csf_real(
        h1e.real, eri.real, norb, nelec, transformer, hdiag_det=hdiag_det.real, max_memory=max_memory)
    hdiag_csf_out = hdiag_csf.astype(np.complex128)
    hdiag_csf_out.real = hdiag_csf
    hdiag_csf_out.imag = make_hdiag_csf_real(
        h1e.imag, eri.imag, norb, nelec, transformer, hdiag_det=hdiag_det.imag, max_memory=max_memory)
    hdiag_csf = None
    return hdiag_csf_out

def _debug_g2e (fci, g2e, eri, norb):
    _debug_g2e_real (fci, g2e.real, eri.real, norb)
    _debug_g2e_real (fci, g2e.imag, eri.imag, norb)
    return

def pspace(**args):
    pass


def kernel(fci, h1e, eri, norb, nelec, smult=None, idx_sym=None, ci0=None,
           tol=None, lindep=None, max_cycle=None, max_space=None,
           nroots=None, davidson_only=None, pspace_size=None, max_memory=None,
           orbsym=None, wfnsym=None, ecore=0, transformer=None, **kwargs):
    '''
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
    hdiag_csf = fci.make_hdiag_csf (h1e, eri, norb, nelec, hdiag_det=hdiag_det, max_memory=max_memory)
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: hdiag_csf", *t0)
    ncsf_all = count_all_csfs (norb, neleca, nelecb, smult)
    if idx_sym is None: ncsf_sym = ncsf_all
    else: ncsf_sym = np.count_nonzero (idx_sym)
    nroots = min(ncsf_sym, nroots)
    if nroots is not None:
        assert (ncsf_sym >= nroots), "Can't find {} roots among only {} CSFs".format (nroots, ncsf_sym)
    link_indexa, link_indexb = direct_spin1._unpack(norb, nelec, None)
    na = link_indexa.shape[0]
    nb = link_indexb.shape[0]
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: throat-clearing", *t0)
    addr, h0 = fci.pspace(h1e, eri, norb, nelec, idx_sym=idx_sym, hdiag_det=hdiag_det, hdiag_csf=hdiag_csf,
                        npsp=max(pspace_size,nroots))
    lib.logger.debug1 (fci, 'csf.kernel: error of hdiag_csf: %s', np.amax (np.abs (hdiag_csf[addr]-np.diag (h0))))
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: make pspace", *t0)
    if pspace_size > 0:
        pw, pv = fci.eig (h0)
    else:
        pw = pv = None

    if pspace_size >= ncsf_sym and not davidson_only:
        if ncsf_sym == 1:
            civecreal = transformer.vec_csf2det (pv[:,0].real.reshape (1,1))
            civec = civecreal.astype(np.complex128)
            civec.real = civecreal
            civec.imag = transformer.vec_csf2det (pv[:,0].imag.reshape (1,1))
            civecreal = None
            return pw[0]+ecore, civec
        elif nroots > 1:
            civeccsf = np.empty((nroots,ncsf_sym), dtype=pw.dtype)
            civeccsf[:,:] = pv[:,:nroots].T # Should I take the conj here?
            civecreal = transformer.vec_csf2det (civeccsf)
            civec = civecreal.astype(pw.dtype)
            civec.real = civecreal
            civec.imag = transformer.vec_csf2det (civeccsf.imag)
            civecreal = None
            return pw[:nroots]+ecore, [c.reshape(na,nb) for c in civec]
        
        elif abs(pw[0]-pw[1]) > 1e-12:
            civeccsf = np.empty((ncsf_sym), dtype=pw.dtype)
            civeccsf[:] = pv[:,0]
            civecreal = transformer.vec_csf2det (civeccsf)
            civec = civecreal.astype(pw.dtype)
            civec.real = civecreal
            civec.imag = transformer.vec_csf2det (civeccsf.imag)
            civecreal = None
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
        x_det = transformer.vec_csf2det (x.real)
        x_det += 1j * transformer.vec_csf2det (x.imag)
        hx = fci.contract_2e(h2e, x_det, norb, nelec, (link_indexa,link_indexb))
        hx_real = transformer.vec_det2csf (hx.real, normalize=False).ravel ()
        hx_imag = transformer.vec_det2csf (hx.imag, normalize=False).ravel ()
        hx_out = hx_real.astype(hx.dtype)
        hx_out.real = hx_real
        hx_out.imag = hx_imag
        hx_real = hx_imag = None
        return hx_out
    t0 = lib.logger.timer_debug1 (fci, "csf.kernel: make hop", *t0)

    if ci0 is None:
        if hasattr(fci, 'get_init_guess'):
            def ci0 ():
                ci0_det = fci.get_init_guess(norb, nelec, nroots, hdiag_csf)
                ci0_csfreal = transformer.vec_det2csf (ci0_det.real)
                ci0_csf = ci0_csfreal.astype(ci0_det.dtype)
                ci0_csf.real = ci0_csfreal
                ci0_csf.imag = transformer.vec_det2csf (ci0_det.imag)
                ci0_csfreal = ci0_det = None
                return ci0_csf
        else:
            def ci0():
                x0 = []
                for i in range(nroots):
                    x = np.zeros(ncsf_sym, dtype=h1e.dtype)
                    x[addr[i]] = 1.0 + 1e-10j
                    x0.append(x)
                return x0
    # Done uptill here.
    else:
        if isinstance(ci0, np.ndarray) and ci0.size == na*nb:
            ci0 = [transformer.vec_det2csf (ci0.ravel ())]
        else:
            nrow = len (ci0)
            ci0 = np.asarray (ci0).reshape (nrow, -1, order='C')
            ci0 = np.ascontiguousarray (ci0)
            if nrow==1: ci0 = ci0[0]
            ci0 = transformer.vec_det2csf (ci0)
            ci0 = [c for c in ci0.reshape (nrow, -1)]

class cplxCSFFCISolver:
    '''
    Parent class for the complex FCI solver in CSF basis. This class will implement the 
    necessary functions. 
    # Borrowing functions from the real CSF solver. Only modifying the functions
    # which are directly needed.
    # This class won't be of any use for standalone.
    '''
    _keys = {'smult', 'trasnsformers'}
    pspace_size = getattr(__config__, 'fci_csf_FCI_pspace_size', 200)
    make_hdiag = make_hdiag_det

    def __init__ (self, smult, **args):
        self.smult = smult
        self.transformers = None
        super().__init__ (**args)

    def make_hdiag_csf(self, h1e, eri, norb, nelec, hdiag_det=None, smult=None, max_memory=None):
        self.norb = norb
        self.nelec = nelec
        if smult is not None:
            self.smult = smult
        self.check_transformer_cache ()
        max_memory = max_memory if max_memory is not None else self.max_memory

        return self.make_hdiag_csf(h1e, eri, norb, nelec, self.transformer, hdiag_det=hdiag_det, 
                                   max_memory=max_memory)

    def absorb_h1e(self, h1e, eri, norb, nelec, fac=1):
        h1e_c, h1e_s = unpack_h1e_cs(h1e)
        h2eff = super().absorb_h1e(h1e_c, eri, norb, nelec, fac=fac)
        if h1e_s is not None:
            h2eff = tag_array(h2eff, h1e_s=h1e_s)
        return h2eff
    
    log_transformer_cache = lib.module_method(realCSFFCISolver.log_transformer_cache)
    print_transformer_cache = lib.module_method(realCSFFCISolver.print_transformer_cache)

    def contract_2e(self, eris, fcivec, norb, nelec, link_index=None, **kwargs):
        hc = super().contract_2e(eris, fcivec, norb, nelec, link_index=link_index, **kwargs)
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

    def kernel(self, h1e, eri, norb, nelec, ci0=None, **kwargs):
        self.norb = norb
        self.nelec = nelec
        self.smult = kwargs.pop('smult', self.smult)
        self.check_transformer_cache ()
        self.log_transformer_cache (lib.logger.DEBUG)

        e, c = kernel (self, h1e, eri, norb, nelec, smult=self.smult,
                       idx_sym=None, ci0=ci0, transformer=self.transformer,
                       **kwargs)

        self.eci, self.ci = e, c
        return e, c

    check_transformer_cache = lib.module_method(realCSFFCISolver.check_transformer_cache)
    