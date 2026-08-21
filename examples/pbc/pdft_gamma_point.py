
'''
Here, I am doing the gamma-point CASSCF followed by PDFT.

Note:
    1. mf.exxdiv=None should be used. Post-SCF methods require this.
    2. This example uses GDF. The default FFTDF path has not been tested for
       periodic MC-PDFT.
    3. Periodic MC-PDFT must be initialized from a periodic HF object. Do not
       pass an RKS/UKS object or convert one to RHF: the required density-fitting
       ERIs may not be available or consistent. Build and run the HF/GDF object
       directly so that its integrals are generated correctly.
    4. For a single-determinant active space and fixed HF orbitals, tPBE equals
       PBE evaluated non-self-consistently on the same HF density.
'''


POSCAR='''
Polyacetylene Unit Cell
1.0
2.4700000000  0.0000000000  0.0000000000
0.0000000000  17.5000000000  0.0000000000
0.0000000000  0.0000000000  17.5000000000
C    H
2   2
Cartesian
-0.5892731038  0.3262391909  0.0000000000
0.5916281105  -0.3261693897  0.0000000000
-0.5866101958  1.4126530287  0.0000000000
0.5889652025  -1.4125832275  0.0000000000
'''

import numpy as np
from pyscf.pbc import gto, scf, dft
from pyscf import mcscf
from mrh.my_pyscf.pbc import mcpdft

# Periodic Calculation for CH=CH uni, using CASCI vs RHF
def getcell():
    cell = gto.Cell()
    cell.a='''
    2.4700000000  0.0000000000  0.0000000000
    0.0000000000  17.5000000000  0.0000000000
    0.0000000000  0.0000000000  17.5000000000
    '''
    cell.atom='''
    C -0.5892731038  0.3262391909  0.0000000000
    C 0.5916281105  -0.3261693897  0.0000000000
    H -0.5866101958  1.4126530287  0.0000000000
    H 0.5889652025  -1.4125832275  0.0000000000
    '''
    cell.basis = '321g'
    cell.precision=1e-12
    cell.verbose = 4
    cell.build()
    return cell

def periodicPDFT():
    cell = getcell()
    mf = scf.RHF(cell).density_fit()
    mf.exxdiv = None
    mf.kernel()

    # Evaluate PBE on the HF density without optimizing DFT orbitals.
    ks = dft.RKS(cell).density_fit()
    ks.verbose = 4
    ks.exxdiv = None
    ks.xc = 'pbe'
    ks.max_cycle = 0
    eperpbe = ks.kernel(mf.make_rdm1())

    mc = mcpdft.CASCI(mf, 'tPBE', 1,2)
    epdftper = mc.kernel(mf.mo_coeff)[0]
    print("Periodic Cal (PBE@HF vs tPBE): ", np.allclose(eperpbe,epdftper, 1e-7))

def periodicHF():
    cell = getcell()
    mf = scf.RHF(cell).density_fit()
    mf.exxdiv=None
    eper = mf.kernel()
    mc = mcscf.CASCI(mf, 1,2)
    ecasper = mc.kernel(mf.mo_coeff)[0]
    print("Periodic Cal (CASCI vs RHF): ", np.allclose(eper,ecasper, 1e-7))

if __name__ == "__main__":
    periodicHF()
    periodicPDFT()
