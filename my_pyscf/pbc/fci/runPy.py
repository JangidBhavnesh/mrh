#!/usr/bin/env python

'''
CCSD with k-point sampling or at an individual k-point
'''

from functools import reduce
import numpy
from pyscf.pbc import gto, scf, cc

cell = gto.Cell()
cell.atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = 'gth-szv'
cell.pseudo = 'gth-pade'
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
cell.verbose = 3
cell.build()

#
# The PBC module provides an separated implementation specified for the single
# k-point calculations.  They are more efficient than the general implementation
# with k-point sampling.  For gamma point, integrals and orbitals are all real
# in this implementation.  They can be mixed with other post-HF methods that
# were provided in the molecular program.
#
kpt = cell.get_abs_kpts([0.25, 0.25, 0.25])
mf = scf.RHF(cell, kpt=kpt)
ehf = mf.kernel()

mycc = cc.RCCSD(mf).run()
print("RCCSD energy (per unit cell) at k-point =", mycc.e_tot)
dm1 = mycc.make_rdm1()
dm2 = mycc.make_rdm2()
nmo = mf.mo_coeff.shape[1]
eri_mo = mf.with_df.ao2mo(mf.mo_coeff, kpts=kpt).reshape([nmo]*4)
h1 = reduce(numpy.dot, (mf.mo_coeff.conj().T, mf.get_hcore(), mf.mo_coeff))
e_tot = numpy.einsum('ij,ji', h1, dm1) + numpy.einsum('ijkl,jilk', eri_mo, dm2)*.5 + mf.energy_nuc()
print("RCCSD energy based on CCSD density matrices =", e_tot.real)

