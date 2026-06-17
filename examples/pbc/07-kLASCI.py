import numpy as np
import scipy

from pyscf import lib
from pyscf.pbc import scf
from pyscf.pbc import gto as pgto

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.util.transym import localize_mo_coeff

np.set_printoptions(precision=4, suppress=True)

# Example file to use the kLASCI.
# Author: Bhavnesh Jangid

cell = pgto.Cell()
cell.a = np.diag([3.0, 17.5, 17.5])
cell.atom ='''
H 0.0 0.0 0.0
H 0.74 0.0 0.0
'''
cell.basis = '631G'
cell.unit = 'Angstrom'
cell.max_memory = 120000
cell.ke_cutoff = 100
cell.precision = 1e-10
cell.verbose = lib.logger.INFO
cell.build()

# Define the k-point mesh and the k-points for the calculation.
kmesh = [3, 1, 1]
kpts = cell.make_kpts(kmesh, wrap_around=True)

# Mean-field calculation.
kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
kmf.max_cycle=1000
# kmf.with_df._cderi_to_save = "cderi.h5"
kmf.with_df._cderi = 'cderi.h5'
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

# Active space selection.
mo_coeff = avas.kernel(kmf, ['H 1s'], minao=cell.basis)[2]

# Now we can run the k-LASCI calculation.
klas = mcscf.KLASCI(kmf, 2, (1, 1), kmesh=kmesh)
# Localization of the active space orbitals, this is important.
lo_coeff = localize_mo_coeff(klas, mo_coeff, ['H 1s'])[0]
klas.kernel(lo_coeff)
