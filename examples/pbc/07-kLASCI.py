import numpy as np
import scipy

from pyscf import lib
from pyscf.pbc import scf
from pyscf.pbc import gto as pgto

from mrh.my_pyscf.pbc.fci import csf_solver
from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.mcscf import KLASCI
from mrh.my_pyscf.pbc.util.pbcmolden import print_molden_only_as, print_molden_wannier, print_molden_natorbs
from mrh.my_pyscf.pbc.mcscf.k2R import  get_mo_coeff_k2R

np.set_printoptions(precision=4, suppress=True)

# Example file to use the kLASCI.
# Author: Bhavnesh Jangid

cell = pgto.Cell()
cell.a = np.diag([3.0, 17.5, 17.5])
cell.atom ='''
H 0.0 0.0 0.0
H 0.74 0.0 0.0
'''
cell.basis = '6-31G'
cell.unit = 'Angstrom'
cell.max_memory = 120000
cell.ke_cutoff = 100
cell.precision = 1e-10
cell.verbose = lib.logger.INFO
cell.build()

kmesh = [6, 6, 6]
kpts = cell.make_kpts(kmesh, wrap_around=True)

kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
kmf.max_cycle=1000
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

mo_coeff = avas.kernel(kmf, ['H 1s'], minao=cell.basis)[2]

from mrh.my_pyscf.pbc.util.transym import TranslationSymm, localize_kmf_mo_coeff, check_wannier_translation, get_wannier_orbs,pack_wannier_orb, unpack_wannier_orb, make_wannier_matrix, make_ovlp_mat_in_wannier_basis, orthogonality_check
ts = TranslationSymm(cell, kmesh)

lo_coeff = localize_kmf_mo_coeff(kmf, mo_coeff)[0]

wannier_orb, R_indices = get_wannier_orbs(kmf, kmesh, lo_coeff)
check_wannier_translation(ts, wannier_orb, R_indices)

# # Orthogonality check for the Wannier orbitals in packed form.
# wannier_orb = make_wannier_matrix(wannier_orb)
# ovlp_k = kmf.get_ovlp(kpts=kpts)
# ovlp_bvk = make_ovlp_mat_in_wannier_basis(kmf, kmesh)
# orthogonality_check(wannier_orb, ovlp_bvk)
# print('Orthogonality check passed! for wannier orbitals')

wannier_packed = pack_wannier_orb(wannier_orb, ref_cell=0)
wannier_unpacked = unpack_wannier_orb(wannier_packed, cell, kmesh, ref_cell=0)
assert np.allclose(wannier_orb, wannier_unpacked), "Wannier unpacking failed, something went wrong in the packing/unpacking functions."
print("Wannier packing/unpacking check passed!")

print_molden_wannier(wannier_packed, cell, kmesh, 'wannier_orb.molden', occ=None)
exit()

# Print the molden file for the active space orbitals.
print_molden_only_as(kmf, mo_coeff, kmesh, 'mo_coeff.molden', 2, 0)


# kmc = mcscf.CASSCF(kmf, 2, 2)
# kmc.kpts = kpts
# kmc.kmesh = kmesh
# kmc.fcisolver = csf_solver(cell, smult=1)
# kmc.max_cycle_macro = 50
# kmc.kernel(mo_coeff)

# print_molden_natorbs(kmc, kmesh, 'natorbs.molden')
