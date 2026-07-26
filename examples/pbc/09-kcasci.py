import numpy as np
import matplotlib.pyplot as plt

from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.fci import csf_solver

'''
Basic k-CASCI example.
The k-CASCI solver works in one total momentum sector at a time.  This example
runs the same active space for each target_k sector in a small periodic H2
system.
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
cell.basis = "CC-PVDZ"
cell.unit = "Angstrom"
cell.max_memory = 100000
cell.ke_cutoff = 100
cell.precision = 1e-10
cell.verbose = lib.logger.INFO
cell.build()

kmesh = [5, 1, 1]
kpts = cell.make_kpts(kmesh, wrap_around=True)
nkpts = len(kpts)

kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis="def2-svp-jkfit")
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

print(f"k-RHF energy: {kmf.e_tot.real:12.8f}")

mo_coeff = avas.kernel(kmf, ['H 1s'], minao=cell.basis)[2]

kmc = mcscf.CASSCF(kmf, 2, 2)
kmc.kpts = kpts
kmc.kmesh = kmesh
kmc.fcisolver = csf_solver(cell, smult=1)
kmc.kernel(mo_coeff)

mo_coeff = np.array(kmc.mo_coeff).copy()

# Ground-State:
# Neutral reference energy in the KCASCI per-cell convention.
kmc_neutral = mcscf.KCASCI(kmf, 2, 2, target_k=0)
kmc_neutral.kmesh = kmesh
kmc_neutral.fcisolver.fix_spin_(shift=0.2, ss=0.0)
e_neutral = kmc_neutral.kernel(mo_coeff)[0]
    
# N-1 charged state:
# charge=1 means one electron is removed from the complete k-mesh active
# space.  target_k=None sweeps all charged momentum sectors.
kmc_hole = mcscf.KCASCI(kmf, 2, 2, charge=1)
kmc_hole.kmesh = kmesh
kmc_hole.fcisolver.nroots = 1
kmc_hole.fcisolver.fix_spin_(shift=0.2, ss=0.75)
kmc_hole.kernel(mo_coeff)

# N+1 charged state:
# charge=-1 means one electron is added to the complete k-mesh active space.
kmc_particle = mcscf.KCASCI(kmf, 2, 2, charge=-1)
kmc_particle.kmesh = kmesh
kmc_particle.fcisolver.nroots = 1
kmc_particle.fcisolver.fix_spin_(shift=0.2, ss=0.75)
kmc_particle.kernel(mo_coeff)

hole_bands = kmc_hole.band_energies(e_neutral, kpts=kpts)
particle_bands = kmc_particle.band_energies(e_neutral, kpts=kpts)
hole_by_k = {band["momentum_index"]: band for band in hole_bands}
particle_by_k = {band["momentum_index"]: band for band in particle_bands}

print(f"N-1 active space: {sum(kmc_hole.charged_nelecastot)}e, "
      f"{kmc_hole.nkpts * kmc_hole.ncas}o")
print(f"N+1 active space: {sum(kmc_particle.charged_nelecastot)}e, "
      f"{kmc_particle.nkpts * kmc_particle.ncas}o")
print()
print("Quasiparticle energies (Eh)")
print("  k     scaled kx    target(N-1)  kHF EA      KCASCI EA   "
      "target(N+1)  kHF IP      KCASCI IP")

# This active space has one occupied and one virtual kHF orbital per k-point.
scaled_kpts = cell.get_scaled_kpts(kpts)
k_order = np.argsort(scaled_kpts[:, 0])
x = scaled_kpts[k_order, 0]
e_minus_khf_band = np.asarray(
    [kmf.mo_energy[k][0].real for k in k_order])
e_plus_khf_band = np.asarray(
    [kmf.mo_energy[k][1].real for k in k_order])
e_minus_kcasci = np.asarray(
    [hole_by_k[k]["energy"].real for k in k_order])
e_plus_kcasci = np.asarray(
    [particle_by_k[k]["energy"].real for k in k_order])
au_to_ev = 27.211386245988
for k in k_order:
    hole = hole_by_k[k]
    particle = particle_by_k[k]
    e_minus_khf = np.asarray(kmf.mo_energy[k])[0].real
    e_plus_khf = np.asarray(kmf.mo_energy[k])[1].real
    print(f"{k:3d}  {scaled_kpts[k, 0]:11.6f}"
          f"  {hole['target_k']:11.6f}  {au_to_ev * e_minus_khf:11.8f}"
          f"  {au_to_ev * hole['energy'].real:11.8f}"
          f"  {particle['target_k']:11.6f}  {au_to_ev * e_plus_khf:11.8f}"
          f"  {au_to_ev * particle['energy'].real:11.8f}")

# Plot quasiparticle band energies.
fig, ax = plt.subplots(figsize=(6.0, 4.2))
ax.plot(x, au_to_ev * e_minus_khf_band, "o-", label="kHF EA")
ax.plot(x, au_to_ev * e_minus_kcasci, "s--", label="KCASCI EA")
ax.plot(x, au_to_ev * e_plus_khf_band, "o-", label="kHF IP")
ax.plot(x, au_to_ev * e_plus_kcasci, "s--", label="KCASCI IP")
ax.axhline(0.0, color="0.7", linewidth=0.8)
ax.set_xlabel(r"scaled $k_x$")
ax.set_ylabel("quasiparticle energy (Eh)")
ax.set_xticks(x)
ax.legend()
fig.tight_layout()
plt.savefig("kcasci_bands.png", dpi=300)
