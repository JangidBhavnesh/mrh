import numpy as np

from pyscf import lib
from pyscf.pbc import scf
from pyscf.pbc import gto as pgto

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas

np.set_printoptions(precision=4, suppress=True)

# Author: Bhavnesh Jangid
# Example file to use the kLASCI.

cell = pgto.Cell()
cell.a = np.diag([3.0, 17.5, 17.5])
cell.atom ='''
H 0.0 0.0 0.0
H 0.74 0.0 0.0
'''
cell.basis = 'CC-PVDZ'
cell.unit = 'Angstrom'
cell.max_memory = 120000
cell.ke_cutoff = 100
cell.precision = 1e-10
cell.verbose = lib.logger.INFO
cell.build()

# Define the k-point mesh and the k-points for the calculation.
kmesh = [10, 1, 1]
kpts = cell.make_kpts(kmesh, wrap_around=True)

# Mean-field calculation.
kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
kmf.max_cycle= 200
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

# Active space selection.
mo_coeff = avas.kernel(kmf, ['H 1s'], minao=cell.basis)[2]

# Now we can run the k-LASCI calculations with and without translation
# symmetry.Currently, we are using nkpts=10, so in total
# the active space of (20e, 20o)
klas_no_trans = mcscf.KLASCI(kmf, 2, (1, 1), kmesh=kmesh, trans_sym=False,)
lo_coeff = klas_no_trans.localize_init_guess(['H 1s'], mo_coeff=mo_coeff,)
e_klas_no_trans = klas_no_trans.kernel(np.array(lo_coeff, copy=True),)[1]

# With translation symmetry,
klas_trans = mcscf.KLASCI(kmf, 2, (1, 1), kmesh=kmesh, trans_sym=True,)
e_klas_trans = klas_trans.kernel(np.array(lo_coeff, copy=True),)[1]

print('k-LASCI energy without translation symmetry:', e_klas_no_trans.real)
print('k-LASCI energy with translation symmetry:   ', e_klas_trans.real)
print('Absolute energy difference:                 ', abs(e_klas_trans - e_klas_no_trans))
