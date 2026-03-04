import numpy as np

from pyscf import lib
from pyscf import __config__
from pyscf.csf_fci.csf import CSFFCISolver as realCSFFCISolver

from mrh.my_pyscf.pbc.fci import direct_spin1_cplx

# Okay Great. Let me implement the CSFsolver with complex Hamiltonian.

def pspace(**args):
    pass

def get_init_guess(**args):
    pass

def kernel(**args):
    pass

def make_hdiag_det (h1e, eri, nrob, nelec):
    # Implement the make_hdiag the real csfsolver calls the direct_uhf.make_hdiag.
    # However that won't be directly applicable as the h1e and eri are complex.
    pass

class CSFTransformer:
    pass

class cplxCSFFCISolver:
    # This class won't be of any use for standalone.
    _keys = {'smult', 'trasnsformers'}
    pspace_size = getattr(__config__, 'fci_csf_FCI_pspace_size', 200)
    make_hdiag = make_hdiag_det

    def __init__ (self, smult, **args):
        self.smult = smult
        self.transformers = None
        super().__init__ (**args)

    # Borrowing functions from the real CSF solver. Only modifying the functions
    # which are directly needed.
    make_hdiag_csf = lib.module_method(realCSFFCISolver.make_hdiag_csf)
    absorb_h1e = lib.module_method(realCSFFCISolver.absorb_h1e)
    log_transformer_cache = lib.module_method(realCSFFCISolver.log_transformer_cache)
    print_transformer_cache = lib.module_method(realCSFFCISolver.print_transformer_cache)

    def contract_2e(self, eris, fcivec, norb, nelec, link_index=None, **kwargs):
        hc = super().contract_2e(eris, fcivec, norb, nelec, link_index=link_index, **kwargs)
        if hasattr(eris, 'h1e_s'):
            pass
            # Need to implement this function.
            #hc += direct_uhf.contract_1e ([eri.h1e_s, -eri.h1e_s], fcivec, norb, nelec, link_index)
        return hc
    
    def pspace (self, h1e, eri, norb, nelec, hdiag_det=None, hdiag_csf=None, npsp=200, **kwargs):
        self.norb = norb
        self.nelec = nelec
        self.smult = kwargs.pop('smult', self.smult)
        self.check_transformer_cache ()
        max_memory = kwargs.get ('max_memory', self.max_memory)
        return pspace (self, h1e, eri, norb, nelec, self.transformer, hdiag_det=hdiag_det,
                       hdiag_csf=hdiag_csf, npsp=npsp, max_memory=max_memory)


# Good chance to learn Inheritance, and MRI Method:
class FCISolver(cplxCSFFCISolver, direct_spin1_cplx.FCISolver):
    def get_init_guess(self, norb, nelec, nroots, hdiag_csf, **kwargs):
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
            idx_sym=None, ci0=ci0, transformer=self.transformer, **kwargs)
        self.eci, self.ci = e, c
        return e, c

    def check_transformer_cache (self):
        assert (isinstance (self.smult, (int, np.number)))
        neleca, nelecb = direct_spin1_cplx._unpack_nelec (self.nelec)
        if self.transformer is None:
            self.transformer = CSFTransformer (self.norb, neleca, nelecb, self.smult,
                                               max_memory=self.max_memory)
        else:
            self.transformer._update_spin_cache (self.norb, neleca, nelecb, self.smult)