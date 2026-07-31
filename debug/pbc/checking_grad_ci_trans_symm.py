#!/usr/bin/env python

'''Check translation symmetry of the fragment CI gradients in PBC-LASCI.

For identical translated CI vectors, every fragment CI-gradient block should
be a copy of the reference-cell block.  When the CI vectors use different
global phases, their gradients should transform by the same phases.

This example uses one root per unit cell.
'''

import numpy as np

from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.mcscf.productstate import (
    PBCTransSymmImpureProductStateFCISolver,
)

# Author: Bhavnesh Jangid


def split_ci_gradients(grad, fcisolvers):
    '''Split the packed product-state CI gradient into fragment blocks.'''
    grad_fragments = []
    offset = 0
    for solver in fcisolvers:
        nroots = solver.nroots
        size = nroots * solver.transformer.ncsf
        if nroots > 1 and getattr(solver, 'weights', None) is not None:
            size += nroots * (nroots - 1) // 2
        grad_fragments.append(grad[offset:offset + size])
        offset += size
    assert offset == grad.size
    return grad_fragments


cell = gto.Cell()
cell.a = np.diag([3.0, 10.0, 10.0])
cell.atom = 'H 0 0 0; H 0.74 0 0'
cell.basis = '6-31g'
cell.unit = 'Angstrom'
cell.precision = 1e-8
cell.ke_cutoff = 20
cell.verbose = lib.logger.WARN
cell.build()

kmesh = [2, 1, 1]
kpts = cell.make_kpts(kmesh, wrap_around=True)
kmf = scf.KRHF(cell, kpts=kpts).density_fit()
kmf.exxdiv = None
kmf.max_cycle = 0
kmf.kernel()

mo_coeff = avas.kernel(kmf, ['H 1s'], minao=cell.basis)[2]
pbclasci = mcscf.KLASCI(
    kmf, 2, (1, 1), kmesh=kmesh, trans_sym=True, ref_cell=0,
)
lo_coeff = pbclasci.localize_init_guess(['H 1s'], mo_coeff=mo_coeff)
pbclasci.kernel(lo_coeff)

h1e = pbclasci.h1e_for_cas(
    mo_coeff=lo_coeff, ncas=pbclasci.ncas, ncore=pbclasci.ncore,
)[0]
h2e = pbclasci.get_h2cas(lo_coeff)
fcisolvers = [box.fcisolvers[0] for box in pbclasci.fciboxes]
product_solver = PBCTransSymmImpureProductStateFCISolver(
    fcisolvers, lweights=[[1.0] for _ in fcisolvers],
    ref_cell=pbclasci.ref_cell,
)

# Use the PBCLASCI reference CI, with a small common perturbation so that the
# gradients are nonzero and their translation relation is tested explicitly.
ci_pbclasci = [np.asarray(ci_frag)[0] for ci_frag in pbclasci.ci]
ci_ref_pbclasci = ci_pbclasci[pbclasci.ref_cell]
ci_ref_trial = np.array(ci_ref_pbclasci, copy=True)
ci_ref_trial[0, 0] += 0.05
ci_ref_trial /= np.linalg.norm(ci_ref_trial)


def get_ci_gradients(ci):
    h1eff, _, _ = product_solver.project_hfrag(
        h1e, h2e, ci, pbclasci.ncas_sub, pbclasci.nelecas_sub,
    )
    grad = product_solver._get_grad(
        h1eff, h2e, ci, pbclasci.ncas_sub, pbclasci.nelecas_sub,
    )
    return split_ci_gradients(grad, product_solver.fcisolvers)


# Same CI gauge in every cell: the gradient blocks are literal copies.
ci_copies = product_solver._unpack_cif(ci_ref_trial)
grad_copies = get_ci_gradients(ci_copies)
grad_ref = grad_copies[pbclasci.ref_cell]
copy_error = max(
    np.max(np.abs(grad_frag - grad_ref)) for grad_frag in grad_copies
)

# Derive the fragment CI phases from translation of the localized active
# orbitals.  No CI-vector overlaps enter this construction.
phases = pbclasci.get_phase_per_frag(lo_coeff)

ci_phased = product_solver._unpack_cif(ci_ref_trial, phases=phases)
grad_phased = get_ci_gradients(ci_phased)
phase_ref = phases[pbclasci.ref_cell]
phase_error = max(
    np.max(np.abs(
        grad_frag - phase / phase_ref * grad_phased[pbclasci.ref_cell]
    ))
    for grad_frag, phase in zip(grad_phased, phases)
)

print('Reference CI-gradient norm:', np.linalg.norm(grad_ref))
print('Orbital-derived fragment CI phases:', phases)
print('Maximum literal-copy error:', copy_error)
print('Maximum phase-transformed error:', phase_error)

assert np.linalg.norm(grad_ref) > 1e-8
assert copy_error < 1e-10
assert phase_error < 1e-10
