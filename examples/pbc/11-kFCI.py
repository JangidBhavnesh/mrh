#!/usr/bin/env python
import numpy as np

from mrh.my_pyscf.pbc.fci import ksolver

# Author: Bhavnesh Jangid

"""
kFCI performs full configuration interaction while conserving total crystal
momentum. ``target_k`` selects the momentum sector solved independently.
"""

nkpts = 3
ncas = 2
norb = nkpts * ncas
nelecas = (3, 3)
ecore = 0.0
rng = np.random.default_rng(12)

# Generating h1e and h2e
h1e = (rng.standard_normal((nkpts, ncas, ncas))
       + 1j * rng.standard_normal((nkpts, ncas, ncas)))
h1e = 0.5 * (h1e + h1e.conj().transpose(0, 2, 1))

# Generate the ERIs in the full orbital labeling so that both complex
# two-electron symmetries can be imposed with simple transposes.  The mask
# retains only kp - kq + kr - ks = 0 matrix elements.
eri = (rng.standard_normal((norb,) * 4)
       + 1j * rng.standard_normal((norb,) * 4))
orb_k = np.arange(norb) // ncas
eri *= (
    (orb_k[:, None, None, None] - orb_k[None, :, None, None]
     + orb_k[None, None, :, None] - orb_k[None, None, None, :])
    % nkpts == 0
)
eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
eri = 0.5 * (eri + eri.conj().transpose(1, 0, 3, 2))

# Store only kp, kq, and kr; momentum conservation determines ks.
kp, kq, kr = np.indices((nkpts,) * 3)
ks = (kp - kq + kr) % nkpts
eri = eri.reshape((nkpts, ncas) * 4)
h2e = eri[kp, :, kq, :, kr, :, ks, :]

# Solving the k-FCI problem for each momentum sector independently.  The k-FCI solver
# currently provides spin penalization but not a CSF solver.
for target_k in range(nkpts):
    fcisolver = ksolver(nkpts=nkpts, target_k=target_k)
    fcisolver.conv_tol = 1e-10
    fcisolver.fix_spin_(shift=0.2, ss=0.0)
    e_tot, ci = fcisolver.kernel(
        h1e, h2e, norb, nelecas, ecore=ecore)
    ss, multiplicity = fcisolver.spin_square(ci, norb, nelecas)
    rdm1, rdm2 = fcisolver.make_rdm12(
        ci, norb, nelecas, nkpts, target_k=target_k)

    print(f"target_k     : {target_k}")
    print(f"k-FCI energy : {e_tot.real:12.8f}")
    print(f"<S^2>        : {ss.real:12.8f}")
    print(f"2S+1         : {multiplicity.real:12.8f}")
    print(f"1-RDM shape  : {rdm1.shape}")
    print(f"2-RDM shape  : {rdm2.shape}")
    print(f"1-RDM trace  : {np.trace(rdm1).real:12.8f}")
