"""LSI-LUSCC for stretched C2H4N4.

This example retains the four leading LASSIS product states and the four
interfragment excitations with the largest energy gradients (m=4, n=4).
"""

import numpy as np
from pyscf import scf

from mrh.exploratory.luscc import LSI_LUSCC, get_grad_exact_lassi
from mrh.my_pyscf import lassi
from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
from mrh.tests.lasscf.c2h4n4_struct import structure


# A high-spin SCF reference supplies a clean set of singly occupied pi
# orbitals; LASSCF below couples the three active fragments as singlets.
mol = structure(1.0, 1.0, basis="6-31g", spin=8, verbose=4)
mf = scf.RHF(mol).run()

# The active space describes the two terminal N=N bonds and central C=C bond.
ncas_sub = (4, 2, 4)
nelecas_sub = ((2, 2), (1, 1), (2, 2))
frag_atoms = ((0, 1, 2), (3, 4, 5, 6), (7, 8, 9))
las = LASSCF(mf, ncas_sub, nelecas_sub, spin_sub=(1, 1, 1))
mo_guess = las.localize_init_guess(frag_atoms, mf.mo_coeff)
las.kernel(mo_guess)

# First form the LASSIS reference, then retain its four leading product states.
lsi = lassi.LASSIS(las)
e_lassis, _ = lsi.kernel()
lsi_prime = LSI_LUSCC(lsi, [], [], top_m=4)
e_lsi_prime, _ = lsi_prime.kernel()

# Rank all interfragment excitations by the LSI-reference energy gradient.
gradients, _, a_idxs, i_idxs = get_grad_exact_lassi(lsi_prime)
order = np.argsort(-np.abs(gradients))[:4]
a_selected = [a_idxs[i] for i in order]
i_selected = [i_idxs[i] for i in order]

luscc = LSI_LUSCC(lsi_prime, a_selected, i_selected, top_m=4)
e_luscc, _ = luscc.kernel()

print(f"LASSCF energy:       {las.e_tot:.12f}")
print(f"LASSIS energy:       {e_lassis[0]:.12f}")
print(f"4-state LSI energy:  {e_lsi_prime[0]:.12f}")
print(f"LSI-LUSCC(4,4):      {e_luscc[0]:.12f}")
