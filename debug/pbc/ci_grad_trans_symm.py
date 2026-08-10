#!/usr/bin/env python

import numpy as np

from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.mcscf.productstate import (
    PBCProductStateFCISolver,
    PBCTransSymmImpureProductStateFCISolver,
)

# Author: Bhavnesh Jangid


'''
Checking the translation symmetry of the k-LASCI CI gradient.
In this script, we evaluate the symmetry of the full CI-gradient vector in two ways:

    1. ``PBCProductStateFCISolver`` computes every cell block independently.
    2. ``PBCTransSymmImpureProductStateFCISolver`` computes the reference-cell
        block and reconstructs the remaining blocks by translation.

For the scalar CI-gauge transformation used here, basically the orbitals of the 
other cells are related via the some phase factor to the reference cell's orbitals.  

The CI vectors are related by the same phase factor:
    c_R = (phi_R / phi_0) c_0

Similarly, the CI gradients are related by the same phase factor:
    g_R = (phi_R / phi_0) g_0

Thus the directly computed full gradient and the gradient assembled from
individually translated reference blocks should agree.
The script reports the difference for every cell, the difference between the
two full vectors, and the difference from the translation-symmetric solver's
assembled gradient.  A small perturbation is added to the converged reference
CI vector so that these checks use a nonzero gradient.  

Note: We are checking one root per unit cell.
'''



def split_ci_gradients(grad, fcisolvers):
    '''
    Split the packed product-state CI gradient into fragment (unit cell) blocks.
    '''
    grad_fragments = []
    offset = 0
    for solver in fcisolvers:
        nroots = solver.nroots
        size = nroots * solver.transformer.ncsf
        if nroots > 1 and getattr(solver, 'weights', None) is not None:
            raise NotImplementedError('Weighted multi-root gradients not implemented.')
        grad_fragments.append(grad[offset:offset + size])
        offset += size
    assert offset == grad.size
    return grad_fragments


# Step 1: Build a simple 1D H2 chain with a small basis and a single k-point in the y and z directions.
cell = gto.Cell()
cell.a = np.diag([3.0, 10.0, 10.0])
cell.atom = 'H 0 0 0; H 0.74 0 0'
cell.basis = '6-31g'
cell.unit = 'Angstrom'
cell.precision = 1e-8
cell.ke_cutoff = 20
cell.verbose = lib.logger.WARN
cell.build()

# Step-2: Choose a k-point mesh and run a KRHF calculation to get the initial orbitals.
kmesh = [5, 1, 1]
kpts = cell.make_kpts(kmesh, wrap_around=True)

kmf = scf.KRHF(cell, kpts=kpts).density_fit()
kmf.exxdiv = None
kmf.max_cycle = 0 # Note, we are not converging the SCF just using the initial orbitals to make sure gradients are non-zero.
kmf.kernel()

# Step-3: Active space selection.
mo_coeff = avas.kernel(kmf, ['H 1s'], minao=cell.basis)[2]

# Step-4: Run the k-LASCI calculation with translation symmetry enabled.
klasci = mcscf.KLASCI(kmf, 2, (1, 1), kmesh=kmesh, trans_sym=True,)
lo_coeff = klasci.localize_init_guess(['H 1s'], mo_coeff=mo_coeff)
klasci.kernel(lo_coeff)

# Step-5: Compute the CI gradients using both the direct and translation-symmetric solvers.
h1e = klasci.h1e_for_cas(mo_coeff=lo_coeff, ncas=klasci.ncas, ncore=klasci.ncore,)[0]
h2e = klasci.get_h2cas(lo_coeff)
fcisolvers = [box.fcisolvers[0] 
              for box in klasci.fciboxes]

# Step-6: Create the solver and compute the gradients:
product_solver = PBCTransSymmImpureProductStateFCISolver(
    fcisolvers, lweights=[[1.0] for _ in fcisolvers],
    ref_cell=klasci.ref_cell,)

plain_solver = PBCProductStateFCISolver(fcisolvers)

# Step-7: Initial guess
# Use the klasci reference CI, with a small common perturbation so that the
# gradients are nonzero and their translation relation is tested explicitly.
ci_klasci = [np.asarray(ci_frag)[0] for ci_frag in klasci.ci]
ci_ref_klasci = ci_klasci[klasci.ref_cell]
ci_ref_trial = np.array(ci_ref_klasci, copy=True)
ci_ref_trial[0, 0] += 0.05
ci_ref_trial /= np.linalg.norm(ci_ref_trial)


def _get_ci_gradients(solver, ci):
    '''
    Return the packed full gradient and its individual cell blocks.
    '''
    h1eff = solver.project_hfrag(
        h1e, h2e, ci, klasci.ncas_sub, klasci.nelecas_sub,)[0]
    grad = solver._get_grad(
        h1eff, h2e, ci, klasci.ncas_sub, klasci.nelecas_sub,)
    grad_cells = split_ci_gradients(grad, solver.fcisolvers)
    return grad, grad_cells

# Step-8: Compute the gradients for both solvers.
# Same CI gauge in every cell: the gradient blocks are literal copies.
ci_copies = product_solver._unpack_cif(ci_ref_trial)
grad_overall_copies, grad_copies = _get_ci_gradients(plain_solver, ci_copies)
grad_assembled_copies, _ = _get_ci_gradients(product_solver, ci_copies)
grad_ref = grad_copies[klasci.ref_cell]

copy_error = max(np.max(np.abs(grad_frag - grad_ref)) 
                 for grad_frag in grad_copies)
copy_full_error = np.max(
    np.abs(grad_overall_copies - np.concatenate([grad_ref] * len(grad_copies))))
copy_assembly_error = np.max(np.abs(
    grad_overall_copies - grad_assembled_copies))

# Step-8: Derive the fragment CI phases from translation of the localized active
# orbitals.  
phases = klasci.get_phase_per_frag(lo_coeff)

product_solver.phase_per_frag = product_solver._normalize_phase_per_frag(phases)
ci_phased = product_solver._unpack_cif(ci_ref_trial)
grad_overall, grad_cells = _get_ci_gradients(plain_solver, ci_phased)
grad_assembled, _ = _get_ci_gradients(product_solver, ci_phased)

normalized_phases = product_solver.phase_per_frag
grad_ref_phased = grad_cells[klasci.ref_cell]
grad_cells_translated = [
    phase * grad_ref_phased for phase in normalized_phases
]
grad_individual_translated = np.concatenate(grad_cells_translated)

cell_errors = np.asarray([np.max(np.abs(grad_direct - grad_translated))
    for grad_direct, grad_translated
    in zip(grad_cells, grad_cells_translated)])
overall_difference = grad_overall - grad_individual_translated
assembly_difference = grad_overall - grad_assembled


# Printing the errors and differences for analysis.
print('Reference CI-gradient norm:', np.linalg.norm(grad_ref))
print('Orbital-derived fragment CI phases:', phases)
print('Maximum literal-copy error:', copy_error)
print('Full copy-gradient error:', copy_full_error)
print('Copy-gradient assembly error:', copy_assembly_error)
print('Per-cell direct-minus-translated maximum errors:')
for cell_index, error in enumerate(cell_errors):
    print(f'  cell {cell_index}: {error:.3e}')
print('Overall direct-minus-translated difference norm:',
      np.linalg.norm(overall_difference))
print('Overall direct-minus-translated maximum difference:',
      np.max(np.abs(overall_difference)))
print('Direct-overall minus trans-symmetric assembly maximum difference:',
      np.max(np.abs(assembly_difference)))

assert np.linalg.norm(grad_ref) > 1e-8
assert copy_error < 1e-10
assert copy_full_error < 1e-10
assert copy_assembly_error < 1e-10
assert np.max(cell_errors) < 1e-10
assert np.max(np.abs(overall_difference)) < 1e-10
assert np.max(np.abs(assembly_difference)) < 1e-10
