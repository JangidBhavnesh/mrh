#!/usr/bin/env python

"""Compute the kCASCI spectral function for a 1D H chain on a 5x1x1 mesh.

Two H atoms form a dimerized primitive cell along x. As in the charged-kCASCI
example, transverse vacuum in a 3D supercell models the isolated chain; only
the x direction is sampled. Vacuum-size convergence is not established here.
The active space spans 10 electrons in 10 orbitals across the five k-points.

Both spin channels are included. Increase nroots to converge the charged-state
expansion; the sum rules report the weight omitted by the finite root count.
Frequencies are E(N)-E(N-1) for removal and E(N+1)-E(N) for addition, in Hartree
without a chemical-potential shift. The helper uses supercell energies for
these differences, while kCASCI reports energies per primitive cell.

The example writes separate-momentum and combined PNG plots, an NPZ of spectra
and poles, and a CSV of sum rules.
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc import scf

from mrh.my_pyscf.pbc.fci import spectral_fn_helper as sfh

lib.num_threads(1)

intra_h = 0.74
inter_h = 1.5
vacuum = 17.5

cell = gto.Cell()
cell.a = np.diag([intra_h + inter_h, vacuum, vacuum])
cell.atom = [
    ["H", (0.0, vacuum / 2.0, vacuum / 2.0)],
    ["H", (intra_h, vacuum / 2.0, vacuum / 2.0)],
]
cell.basis = "STO-6G"
cell.unit = "Angstrom"
cell.ke_cutoff = 100
cell.precision = 1e-10
cell.max_memory = 4000
cell.verbose = lib.logger.INFO
cell.build()

kmesh = [5, 1, 1]
kpts = cell.make_kpts(kmesh, wrap_around=True)

kmf = scf.KRHF(cell, kpts=kpts).density_fit(
    auxbasis="def2-svp-jkfit",
)
kmf.max_cycle = 200
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()
if not kmf.converged:
    raise RuntimeError("KRHF did not converge")

mo_coeff = np.asarray(kmf.mo_coeff)
ncas = 2
nelecas = 2
nroots = 2
eta = 0.03  # Lorentzian half-width in Hartree.
output_dir = Path(".")


def setup_solver(kmc, kind):
    kmc.kmesh = kmesh
    kmc.verbose = kmc.fcisolver.verbose = 0
    kmc.fcisolver.conv_tol = 1e-9
    kmc.fcisolver.max_cycle = 200
    kmc.fcisolver.davidson_only = True
    spin = getattr(kmc, "charged_spin", 0)
    print(f"Solving {kind} roots: N_alpha - N_beta = {spin}", flush=True)


# Neutral reference and all N-1/N+1 momentum sectors, in a common orbital basis.
roots = sfh.compute_kcasci_spectral_roots(
    kmf, ncas, nelecas, ncore=0, target_k=0, mo_coeff=mo_coeff,
    nroots_neutral=1, nroots_hole=nroots, nroots_particle=nroots,
    spin_sector_mode="spin_resolved", solver_setup=setup_solver,
)
if not all(row["converged"] for row in roots.roots):
    raise RuntimeError("Some kCASCI roots did not converge")

# Keep all pole weights for diagnostics, including those too small to plot.
poles = sfh.make_spectral_poles(roots, strict=True, min_weight=0)
poles = sfh.label_pole_momenta(poles, kpts)
checks = sfh.spectral_weight_sum_rules(roots, poles=poles)
norm_error = max(abs(row["full_total_norm"] - 1) for row in checks)
if norm_error > 1e-8:
    raise RuntimeError(f"Operator normalization error: {norm_error:.3e}")

# Sum both spins and active orbitals. Native kernels are used when available.
spectrum = sfh.make_spectral_function(
    poles, eta=eta, npts=1201, nkpts=roots.nkpts, norb=ncas,
    broadening="lorentzian", spin_resolved=False, orbital_resolved=False,
)

# Results:
print(f"\nKRHF energy/cell:         {kmf.e_tot.real:16.12f}")
print(f"Neutral kCASCI/cell:      {roots.neutral.e_tot.real:16.12f}")
print(f"Charged roots/sector:     {nroots}")
print(f"Pole records:            {len(poles)}")
print(f"Maximum norm error:      {norm_error:.3e}")
print("Weights sum both spins and active orbitals; complete weight/k = 4.")
scaled_kx = cell.get_scaled_kpts(kpts)[:, 0]
order = np.argsort(scaled_kx)
print(" k    k_x/(2pi/a)     captured weight     missing weight")
for k in order:
    rows = [row for row in checks if row["k"] == k]
    captured = sum(row["full_total_weight"] for row in rows)
    missing = sum(row["full_total_missing"] for row in rows)
    print(f"{k:2d}    {scaled_kx[k]:9.4f}      {captured:14.8f}     {missing:14.8f}")

# Plot each momentum and export the spectrum, poles, and sum rules.
fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
for ax, k in zip(axes, order):
    for kind, color in (("hole", "tab:blue"), ("particle", "tab:red"),
                        ("total", "black")):
        ax.plot(spectrum["omega"], spectrum["spectra"][kind][k, 0, 0],
                color=color, label=kind, linewidth=1)
    ax.set_ylabel(r"$A(k,\omega)$")
    ax.set_title(f"k_x = {scaled_kx[k]:.1f} (2pi/a)", fontsize=10)
axes[0].legend(loc="upper right")
axes[-1].set_xlabel(r"$\omega$ (Hartree; no chemical-potential shift)")
fig.suptitle(f"1D H chain: kCASCI, 5x1x1 mesh, {nroots} charged roots/sector")
fig.tight_layout(rect=(0, 0, 1, 0.97))

output_dir.mkdir(parents=True, exist_ok=True)
prefix = output_dir / "kcasci_spectral_fn_1d_5x1x1"
fig.savefig(prefix.with_suffix(".png"), dpi=180)
plt.close(fig)

# Overlay the total spectral function at all five momenta on one graph.
# Dashed positive-k curves help distinguish overlapping +k and -k spectra.
fig, ax = plt.subplots(figsize=(8, 5))
for k in order:
    ax.plot(spectrum["omega"], spectrum["spectra"]["total"][k, 0, 0],
            label=f"k_x = {scaled_kx[k]:.1f} (2pi/a)",
            linestyle="--" if scaled_kx[k] > 0 else "-", linewidth=1.5)
ax.set_xlabel(r"$\omega$ (Hartree; no chemical-potential shift)")
ax.set_ylabel(r"$A(k,\omega)$")
ax.set_title("1D H chain: total spectral function at all k-points")
ax.legend()
fig.tight_layout()
combined_path = prefix.with_name(prefix.name + "_combined").with_suffix(".png")
fig.savefig(combined_path, dpi=180)
plt.close(fig)

sfh.save_spectral_npz(prefix.with_suffix(".npz"), spectrum, poles=poles)
with prefix.with_suffix(".csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(checks[0]))
    writer.writeheader()
    writer.writerows(checks)
for suffix in (".png", ".npz", ".csv"):
    print(f"Saved {prefix.with_suffix(suffix)}")
print(f"Saved {combined_path}")
