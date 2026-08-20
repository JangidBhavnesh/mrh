"""Run a small non-translation-adapted k-LASSCF calculation.

The active space contains both H 1s orbitals in each Born-von Karman cell.
State-averaged and translation-adapted optimization are not enabled yet, so
this example uses one state and ``trans_sym=False``.
"""

import numpy as np

from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas


cell = gto.Cell()
cell.a = np.diag([4.0, 10.0, 10.0])
cell.atom = "H 0.0 0.0 0.0; H 1.5 0.0 0.0"
cell.basis = "sto-3g"
cell.unit = "Angstrom"
cell.precision = 1e-9
cell.ke_cutoff = 20
cell.verbose = lib.logger.INFO
cell.build()

kmesh = (2, 1, 1)
kpts = cell.make_kpts(kmesh, wrap_around=True)

# k-LAS currently requires Gaussian density fitting.
kmf = scf.KRHF(cell, kpts=kpts).density_fit()
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

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

las.conv_tol_grad = 1e-5
las.max_cycle_macro = 20
las.max_cycle_micro = 5
las.trust_radius = np.pi

e_lasscf, e_cas, ci, mo_coeff, mo_energy, h2eff, veff = las.kernel(
    mo_coeff=mo_guess,
)

print(f"k-RHF energy       : {kmf.e_tot.real: .12f}")
print(f"k-LASSCF energy    : {e_lasscf.real: .12f}")
print(f"k-LASSCF converged : {las.converged}")
