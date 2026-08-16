"""Small k-LASSCF example for a periodic H2 chain.

The active space contains both H 1s orbitals in each cell.  The calculation
uses two k-points, so the LAS wave function contains one local CI problem for
each of the two cells in the Born-von Karman supercell.

State-averaged and translation-packed k-LASSCF optimization are not enabled
yet.  This example therefore runs one state with ``trans_sym=False``.
"""

import numpy as np

from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas


# Build a small one-dimensional H2 chain with vacuum in y and z.
cell = gto.Cell()
cell.a = np.diag([3.0, 8.0, 8.0])
cell.atom = "H 0.0 0.0 0.0; H 0.74 0.0 0.0"
cell.basis = "631G"
cell.unit = "Angstrom"
cell.precision = 1e-9
cell.ke_cutoff = 20
cell.verbose = lib.logger.INFO
cell.build()

kmesh = (10, 1, 1)
kpts = cell.make_kpts(kmesh, wrap_around=True)

# k-LAS currently requires Gaussian density fitting.
kmf = scf.KRHF(cell, kpts=kpts).density_fit()
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

# Select and localize the two H 1s active orbitals per unit cell.
active_labels = ["H 1s"]
mo_avas = avas.kernel(kmf, active_labels, minao=cell.basis)[2]

las = mcscf.KLASSCF(
    kmf,
    ncas=2,
    nelecas=(1, 1),
    kmesh=kmesh,
    trans_sym=False,
)
mo_guess = las.localize_init_guess(active_labels, mo_coeff=mo_avas)

# Move slightly away from the stationary orbitals so that this small example
# actually demonstrates an orbital Newton step.  Opposite rotations at the
# two k-points change the Wannier orbitals in different cells.
# for k, angle in enumerate((0.08, -0.08)):
#     rotation = np.array([
#         [np.cos(angle), -np.sin(angle)],
#         [np.sin(angle), np.cos(angle)],
#     ])
#     mo_guess[k] = mo_guess[k] @ rotation

# Each macroiteration builds a new orbital/CI Hessian.  MINRES performs the
# real-coordinate Newton microiterations inside that fixed Hessian keyframe.
las.conv_tol_grad = 1e-5
las.max_cycle_macro = 50
las.max_cycle_micro = 5
las.trust_radius = np.pi

e_lasscf, e_cas, ci, mo_coeff, mo_energy, h2eff, veff = las.kernel(
    mo_coeff=mo_guess,
)

print(f"k-RHF energy    : {kmf.e_tot.real: .12f}")
print(f"k-LASSCF energy : {e_lasscf.real: .12f}")
print(f"k-LASSCF converged: {las.converged}")
