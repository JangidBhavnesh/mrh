import os
import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["XDG_CACHE_HOME"] = "/tmp"

import matplotlib

from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf

from mrh.my_pyscf.pbc.fci import spectral_fn_helper as sfh


matplotlib.use("Agg")

'''
Example setup for a k-CASCI spectral function.

This example computes neutral/charged k-CASCI roots and projects the
k-resolved creation/destruction operators onto those roots to form poles.
'''


intraH = 0.74
interH = 1.5
vac = 17.5

cell = pgto.Cell()
cell.a = np.diag([intraH + interH, intraH + interH, vac])
cell.atom = [
    ["H", (0.0, 0.0, vac / 2.0)],
    ["H", (intraH, 0.0, vac / 2.0)],
]
cell.basis = "STO-6G"
cell.unit = "Angstrom"
cell.max_memory = 100000
cell.ke_cutoff = 100
cell.precision = 1e-10
cell.verbose = lib.logger.WARN
cell.build()

kmesh = [2, 1, 1]
kpts = cell.make_kpts(kmesh, wrap_around=True)

kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis="def2-svp-jkfit")
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.verbose = 0
kmf.kernel()

mo_coeff = np.asarray(kmf.mo_coeff)


def setup_solver(kmc, kind):
    '''
    Keep the same solver settings for the three sectors in this example.
    '''
    kmc.kmesh = kmesh
    kmc.verbose = 0
    kmc.fcisolver.verbose = 0
    kmc.fcisolver.conv_tol = 1e-10


roots = sfh.compute_kcasci_spectral_roots(
    kmf, 2, 2, mo_coeff=mo_coeff, target_k=0,
    nroots_neutral=1, nroots_hole=2, nroots_particle=2,
    solver_setup=setup_solver)

print(f"k-RHF energy: {kmf.e_tot.real:12.8f}")
print(f"neutral active space: {sum(roots.nelecastot)}e, {roots.ncastot}o")
print("")
print("kind       target_k  root       energy       supercell energy  nelec")

for row in roots.roots:
    print(f"{row['kind']:10s}"
          f"{row['target_k']:8d}"
          f"{row['root']:6d}"
          f"{row['energy'].real:15.8f}"
          f"{row['energy_supercell'].real:21.8f}"
          f"  {row['nelecastot']}")

poles = sfh.make_spectral_poles(roots, min_weight=1e-8)
poles = sfh.label_pole_momenta(poles, kpts)
spectrum = sfh.make_spectral_function(
    poles, eta=0.05, npts=401, spin_resolved=False,
    orbital_resolved=False)
checks = sfh.spectral_weight_sum_rules(roots, poles=poles,
                                       available_only=True)

print("")
print("kind          k  target_k  root  orb  spin       omega        weight")
for row in poles:
    spin = 'a' if row['spin'] == 0 else 'b'
    print(f"{row['kind']:10s}"
          f"{row['k']:4d}"
          f"{row['target_k']:8d}"
          f"{row['root']:6d}"
          f"{row['orbital']:5d}"
          f"{spin:>6s}"
          f"{row['omega'].real:14.8f}"
          f"{row['weight'].real:14.8f}")

print("")
print("k       peak omega        peak A(k,w)")
for k in range(roots.nkpts):
    ak = spectrum['spectra']['total'][k, 0, 0]
    imax = np.argmax(ak)
    print(f"{k:1d}"
          f"{spectrum['omega'][imax]:18.8f}"
          f"{ak[imax]:18.8f}")

max_missing = max(abs(row['total_missing']) for row in checks)
print("")
print(f"max supplied-sector spectral weight missing: {max_missing:.3e}")

fig = None
ax = None
for k in range(roots.nkpts):
    fig, ax = sfh.plot_spectral_function(spectrum, kind='total', k=k, ax=ax)
    ax.lines[-1].set_label(f"k = {k}")
ax.legend()
fig.tight_layout()
fig.savefig("kcasci_spectral_function.png", dpi=200)
print("saved plot: kcasci_spectral_function.png")
