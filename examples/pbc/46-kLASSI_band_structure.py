#!/usr/bin/env python
"""Plot charged kLASSI bands and compute the finite-mesh IP, EA, and gap.

The neutral energy is from LASCI; N-1 and N+1 energies are the lowest LASSI
roots. All calculations use the same geometry, orbitals, and k-mesh.
The bands use electron-removal (-IP) and electron-addition (-EA) energies.
The gap is IP - EA, not an optical excitation energy. Increase kmesh[0] for
denser sampling along this one-dimensional chain; full LAS matrices are built.
PNG and PDF plots are saved in the working directory, with the valence maximum
at zero. Markers are computed mesh points; connecting lines only guide the eye.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyscf import lib
from pyscf.data import nist
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.mcscf.kcasci import compute_band_energies
from mrh.my_pyscf.pbc.mcscf.klassi_trans import (
    orbital_translation, project_translation, momentum_basis,
)

# Build the neutral periodic H2 reference.
cell = gto.Cell()
cell.a = np.diag([3., 10., 10.])
cell.atom = 'H 0 0 0; H .74 0 0'
cell.basis = '6-31g'
cell.unit = 'Angstrom'
cell.precision = 1e-8
cell.ke_cutoff = 30
cell.verbose = lib.logger.INFO
cell.build()

kmesh = (3, 1, 1)
kpts = cell.make_kpts(kmesh, wrap_around=True)
kmf = scf.KRHF(cell, kpts=kpts).density_fit()
kmf.exxdiv = None
kmf.kernel()
if not kmf.converged:
    raise RuntimeError('Neutral KRHF did not converge')

# Optimize the neutral orbitals, then keep them fixed in every charge sector.
neutral = mcscf.KLASSCF(kmf, 2, (1, 1), kmesh=kmesh)
mo_avas = np.asarray(avas.kernel(kmf, ['H 1s'], minao=cell.basis)[2])
mo_guess = neutral.localize_init_guess(['H 1s'], mo_coeff=mo_avas)
neutral.kernel(mo_guess)
if not neutral.converged:
    raise RuntimeError('Neutral kLASSCF did not converge')

neutral_ci = mcscf.KLASCI(kmf, 2, (1, 1), kmesh=kmesh)
neutral_ci.kernel(neutral.mo_coeff.copy())
if not neutral_ci.converged:
    raise RuntimeError('Neutral kLASCI did not converge')

# Solve each charged cell independently and couple its LAS state with the others.
hole = mcscf.KLASSI(neutral_ci, charge=+1)
hole.kernel()
particle = mcscf.KLASSI(neutral_ci, charge=-1)
particle.kernel()
hole_results = hole.kernel_kpts()
particle_results = particle.kernel_kpts()

# Determine K0 from the neutral product, including fermion and orbital phases.
neutral_translations = []
for direction in np.eye(3, dtype=int):
    permutation, phases = orbital_translation(hole, direction)
    translated = project_translation(neutral_ci.ci, neutral_ci.get_nelec_frs(),
                                     permutation, phases)
    if np.max(translated['closure_errors']) > 1e-5:
        raise RuntimeError('Neutral LAS reference is not a translation eigenstate')
    neutral_translations.append(translated['matrix'])
scaled_kpts = cell.get_scaled_kpts(kpts)
neutral_bases = momentum_basis(neutral_translations, translated['overlap'],
                               kmesh, scaled_kpts)
reference_k = next(i for i, basis in enumerate(neutral_bases) if basis.shape[1])
print('Neutral total momentum K0 (scaled):', scaled_kpts[reference_k])

# k_removal = K0 - K(N-1); k_addition = K(N+1) - K0.
hole_bands = compute_band_energies(hole_results, neutral_ci.e_tot.real, root=0,
                                  kpts=kpts, cell=cell, reference_target_k=reference_k)
particle_bands = compute_band_energies(particle_results, neutral_ci.e_tot.real, root=0,
                                      kpts=kpts, cell=cell, reference_target_k=reference_k)
valence = np.empty(len(kpts))
conduction = np.empty(len(kpts))
for bands, energies in ((hole_bands, valence), (particle_bands, conduction)):
    for band in bands:
        energies[band['momentum_index']] = band['energy'].real

# Convert per-cell energies to full-mesh energies before taking differences.
nkpts = len(kpts)
e_n = nkpts * neutral_ci.e_tot.real
e_nm1 = nkpts * hole.e_tot[0]
e_np1 = nkpts * particle.e_tot[0]
ip = e_nm1 - e_n
ea = e_n - e_np1
gap = ip - ea  # E(N-1) + E(N+1) - 2 E(N).
np.testing.assert_allclose(conduction.min() - valence.max(), gap, atol=1e-8, rtol=0)

print('\nFull-mesh energies (Ha):')
print(f'E(N)   = {e_n:.12f}')
print(f'E(N-1) = {e_nm1:.12f}')
print(f'E(N+1) = {e_np1:.12f}')
print('\nEnergy differences against the neutral LAS reference:')
for label, energy in (('IP', ip), ('EA', ea), ('Gap', gap)):
    print(f'{label:3s} = {energy:16.12f} Ha = {energy * nist.HARTREE2EV:12.6f} eV')

# Sort physical band momenta along the chain, wrapping into [-1/2, 1/2).
kx = (scaled_kpts[:, 0] + .5) % 1 - .5
order = np.argsort(kx)
vbm = valence.max()
valence_ev = (valence - vbm) * nist.HARTREE2EV
conduction_ev = (conduction - vbm) * nist.HARTREE2EV
gap_ev = gap * nist.HARTREE2EV

fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
ax.plot(kx[order], valence_ev[order], 'o-', color='tab:blue',
        label='Electron removal (valence)')
ax.plot(kx[order], conduction_ev[order], 's-', color='tab:orange',
        label='Electron addition (conduction)')
ax.axhline(0, color='0.6', linewidth=.8, linestyle=':')
ax.axhline(gap_ev, color='0.6', linewidth=.8, linestyle=':')
ax.text(.5, gap_ev / 2, f'Finite-mesh gap: {gap_ev:.4f} eV',
        transform=ax.get_yaxis_transform(), ha='center', va='center')
ax.set_xticks(kx[order], [r'$\Gamma$' if abs(k) < 1e-10 else f'{k:.3f}'
                          for k in kx[order]])
ax.set_xlabel(r'Crystal momentum $k_x a / (2\pi)$')
ax.set_ylabel('Energy relative to valence maximum (eV)')
ax.set_title(r'Charged kLASSI bands: periodic H$_2$')
ax.legend(loc='upper center')
ax.margins(x=.08, y=.2)
ax.grid(axis='x', alpha=.2)

# Save both a preview and a vector figure without opening a GUI window.
for suffix in ('.png', '.pdf'):
    output = Path(Path(__file__).stem + suffix)
    fig.savefig(output, dpi=180)
    print('Saved band structure:', output.resolve())
plt.close(fig)
