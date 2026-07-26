import os

import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["XDG_CACHE_HOME"] = "/tmp"

import matplotlib

from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc.fci import spectral_fn_helper as sfh


matplotlib.use("Agg")

'''
Two-dimensional k-CASCI spectral function for a square hydrogen lattice.

The conventional cell contains two hydrogen atoms in the xy plane. A
2 x 2 x 1 k-point mesh therefore gives an eight-orbital, eight-electron
neutral active space, which is small enough for this example.
'''


lattice = 2.0
vacuum = 15.0
kmesh = [2, 2, 1]

cell = gto.Cell()
cell.a = np.diag([lattice, lattice, vacuum])
cell.atom = [
    ["H", (0.0, 0.0, vacuum / 2.0)],
    ["H", (lattice / 2.0, lattice / 2.0, vacuum / 2.0)],
]
cell.basis = "STO-3G"
cell.unit = "Angstrom"
cell.precision = 1e-9
cell.verbose = lib.logger.WARN
cell.build()

kpts = cell.make_kpts(kmesh, wrap_around=True)
kmf = scf.KRHF(cell, kpts=kpts).density_fit()
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.verbose = lib.logger.WARN
kmf.kernel()


def setup_solver(kmc, kind):
    '''
    Apply the same convergence settings to every neutral and charged job.
    '''
    kmc.kmesh = kmesh
    kmc.verbose = lib.logger.WARN
    kmc.fcisolver.verbose = lib.logger.WARN
    kmc.fcisolver.conv_tol = 1e-10


roots = sfh.compute_kcasci_spectral_roots(
    kmf,
    ncas=2,
    nelecas=2,
    mo_coeff=np.asarray(kmf.mo_coeff),
    target_k=0,
    nroots_neutral=1,
    nroots_hole=1,
    nroots_particle=1,
    solver_setup=setup_solver,
)

poles = sfh.make_spectral_poles(roots, min_weight=1e-9)
poles = sfh.label_pole_momenta(poles, kpts)
spectrum = sfh.make_spectral_function(
    poles,
    eta=0.08,
    npts=501,
    spin_resolved=False,
    orbital_resolved=False,
)

scaled_kpts = cell.get_scaled_kpts(kpts)
print(f"k-RHF energy per cell: {kmf.e_tot.real:16.10f}")
print(f"active space: {sum(roots.nelecastot)}e, {roots.ncastot}o")
print(f"number of spectral poles: {len(poles)}")
print("")
print(" k       scaled kx   scaled ky       peak omega       peak A(k,w)")

fig = None
ax = None
for k, scaled_kpt in enumerate(scaled_kpts):
    spectral_trace = spectrum["spectra"]["total"][k, 0, 0]
    peak = int(np.argmax(spectral_trace))
    print(f"{k:2d}"
          f"{scaled_kpt[0]:15.8f}"
          f"{scaled_kpt[1]:12.8f}"
          f"{spectrum['omega'][peak]:17.8f}"
          f"{spectral_trace[peak]:18.8f}")

    fig, ax = sfh.plot_spectral_function(
        spectrum, kind="total", k=k, ax=ax)
    ax.lines[-1].set_label(
        rf"$k=({scaled_kpt[0]:.1f},{scaled_kpt[1]:.1f})$")

ax.legend()
ax.set_title(r"Square hydrogen lattice: $A(k,\omega)$")
fig.tight_layout()
fig.savefig("kcasci_spectral_function_2d.png", dpi=200)
sfh.save_spectral_npz("kcasci_spectral_function_2d.npz", spectrum, poles)

print("")
print("saved plot: kcasci_spectral_function_2d.png")
print("saved data: kcasci_spectral_function_2d.npz")
