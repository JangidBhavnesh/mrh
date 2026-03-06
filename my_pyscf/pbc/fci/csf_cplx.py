import numpy as np

from pyscf import lib
from pyscf import __config__
from pyscf.fci import direct_uhf, direct_spin1
from pyscf.csf_fci.csf import CSFFCISolver as realCSFFCISolver
from pyscf.csf_fci.csf import unpack_h1e_cs, get_init_guess, make_hdiag_csf as make_hdiag_csf_real
from pyscf.lib.numpy_helper import tag_array

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

def pspace(**args):
    pass



def kernel(**args):
    pass


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
    