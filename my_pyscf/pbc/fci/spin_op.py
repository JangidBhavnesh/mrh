
import numpy as np
from pyscf.fci.spin_op import contract_ss

def spin_square0(fcivec, norb, nelec, **kwargs):
    '''
    Spin square for complex RHF-FCI CI wfn only.
    (a-ib)*S^2*(a+ib) = a*S^2*a + b*S^2*b + i(a*S^2*b - b*S^2*a)
    '''
    assert fcivec.dtype == np.complex128
    verbose = kwargs.get('verbose', 0)

    def s2(ci1, ci2):
        ci1ssket = contract_ss(ci1, norb, nelec)
        return np.vdot(ci2, ci1ssket)
    
    ssreal = s2(fcivec.real, fcivec.real)
    ssreal += s2(fcivec.imag, fcivec.imag)
    
    ssimag = (s2(fcivec.real, fcivec.imag) 
              - s2(fcivec.imag, fcivec.real))
    
    if abs(ssimag) > 1e-3:
        print ("Warning: Spin square is not real. Imaginary part =", ssimag)
    
    ss = ssreal
    s = np.sqrt(ss + 0.25) - 0.5
    multip = 2*s + 1

    # Although, I can take the sqrt for the complex numbers as well.
    if verbose > 5:
        sstot = ssreal + 1j*ssimag
        stot = np.sqrt(sstot + 0.25) - 0.5
        multip_tot = 2*stot + 1
        print("Spin expectation value including the complex part")
        print("Spin square =", sstot, "Spin =", stot, "Multiplicity =", multip_tot)

    return ss, multip