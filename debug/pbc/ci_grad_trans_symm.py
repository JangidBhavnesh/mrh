#!/usr/bin/env python

'''Check translation symmetry of the k-LASCI CI gradient.

README
======
The script evaluates the full CI-gradient vector in two ways:

1. ``PBCProductStateFCISolver`` computes every unit-cell block independently.
2. ``PBCTransSymmImpureProductStateFCISolver`` computes the reference-cell
   block and reconstructs the remaining blocks by translation.

For the scalar CI-gauge transformation used here, the translated CI vectors
and gradients obey

    c_R = (phi_R / phi_0) c_0,
    g_R = (phi_R / phi_0) g_0.

Consequently, the directly computed full gradient must agree with the column
of individually translated reference-cell gradients.  The script reports the
difference for every cell, the full-vector difference, and the difference from
the translation-symmetric solver's assembled gradient.  A small perturbation
is added to the converged reference CI vector so that the check uses a nonzero
gradient.  One root per unit cell is tested.

The command-line inputs ``1D``, ``2D``, and ``3D`` select these presets:

    1D: kmesh = [5, 1, 1], lattice = [4, 10, 10] Angstrom
    2D: kmesh = [2, 2, 1], lattice = [4, 4, 10] Angstrom
    3D: kmesh = [2, 2, 2], lattice = [4, 4, 4] Angstrom

Examples
--------
Run one case:

    python debug/pbc/ci_grad_trans_symm.py 2D

Run selected cases:

    python debug/pbc/ci_grad_trans_symm.py 1D 3D

With no arguments, all three cases are run.
'''

import sys

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


CASE_CONFIGS = {
    '1D': {
        'lattice': (4.0, 10.0, 10.0),
        'kmesh': (5, 1, 1),
    },
    '2D': {
        'lattice': (4.0, 4.0, 10.0),
        'kmesh': (2, 2, 1),
    },
    '3D': {
        'lattice': (4.0, 4.0, 4.0),
        'kmesh': (2, 2, 2),
    },
}

HH_DISTANCE = 1.5
GRADIENT_TOL = 1e-10


def split_ci_gradients(grad, fcisolvers):
    '''
    Split the packed product-state CI gradient into unit-cell blocks.
    '''
    grad_fragments = []
    offset = 0
    for solver in fcisolvers:
        nroots = solver.nroots
        size = nroots * solver.transformer.ncsf
        if nroots > 1 and getattr(solver, 'weights', None) is not None:
            raise NotImplementedError(
                'Weighted multi-root gradients are not implemented.'
            )
        grad_fragments.append(grad[offset:offset + size])
        offset += size
    assert offset == grad.size
    return grad_fragments


def build_cell(lattice):
    '''Build the H2 periodic cell for a selected dimensional preset.'''
    cell = gto.Cell()
    cell.a = np.diag(lattice)
    cell.atom = f'H 0 0 0; H {HH_DISTANCE} 0 0'
    cell.basis = '6-31g'
    cell.unit = 'Angstrom'
    cell.precision = 1e-8
    cell.ke_cutoff = 20
    cell.verbose = lib.logger.WARN
    cell.build()
    return cell


def build_klasci(case_name):
    '''Build and run the translation-symmetric k-LASCI reference.'''
    config = CASE_CONFIGS[case_name]
    kmesh = list(config['kmesh'])
    cell = build_cell(config['lattice'])
    kpts = cell.make_kpts(kmesh, wrap_around=True)

    # The SCF is intentionally not converged; it only supplies initial
    # orbitals, and a nonzero CI gradient is useful for this diagnostic.
    kmf = scf.KRHF(cell, kpts=kpts).density_fit()
    kmf.exxdiv = None
    kmf.max_cycle = 0
    kmf.kernel()

    mo_coeff = avas.kernel(kmf, ['H 1s'], minao=cell.basis)[2]
    klasci = mcscf.KLASCI(
        kmf, 2, (1, 1), kmesh=kmesh, trans_sym=True,
    )
    lo_coeff = klasci.localize_init_guess(['H 1s'], mo_coeff=mo_coeff)
    klasci.kernel(lo_coeff)
    return klasci, lo_coeff


def build_gradient_context(klasci, lo_coeff):
    '''Build Hamiltonians, solvers, and a perturbed reference CI vector.'''
    h1e = klasci.h1e_for_cas(
        mo_coeff=lo_coeff, ncas=klasci.ncas, ncore=klasci.ncore,
    )[0]
    h2e = klasci.get_h2cas(lo_coeff)
    fcisolvers = [box.fcisolvers[0] for box in klasci.fciboxes]

    trans_solver = PBCTransSymmImpureProductStateFCISolver(
        fcisolvers,
        lweights=[[1.0] for _ in fcisolvers],
        ref_cell=klasci.ref_cell,
    )
    plain_solver = PBCProductStateFCISolver(fcisolvers)

    ci_klasci = [np.asarray(ci_frag)[0] for ci_frag in klasci.ci]
    ci_ref_trial = np.array(ci_klasci[klasci.ref_cell], copy=True)
    ci_ref_trial[0, 0] += 0.05
    ci_ref_trial /= np.linalg.norm(ci_ref_trial)

    return h1e, h2e, plain_solver, trans_solver, ci_ref_trial


def get_ci_gradients(solver, ci, h1e, h2e, klasci):
    '''Return the packed full gradient and its individual unit-cell blocks.'''
    h1eff = solver.project_hfrag(
        h1e, h2e, ci, klasci.ncas_sub, klasci.nelecas_sub,
    )[0]
    grad = solver._get_grad(
        h1eff, h2e, ci, klasci.ncas_sub, klasci.nelecas_sub,
    )
    grad_cells = split_ci_gradients(grad, solver.fcisolvers)
    return grad, grad_cells


def compute_gradient_errors(klasci, lo_coeff):
    '''Compute direct, translated, and assembled CI-gradient differences.'''
    (h1e, h2e, plain_solver, trans_solver,
     ci_ref_trial) = build_gradient_context(klasci, lo_coeff)

    # First use the same CI gauge in every cell.  The independently computed
    # gradient blocks should then be literal copies of the reference block.
    ci_copies = trans_solver._unpack_cif(ci_ref_trial)
    grad_overall_copies, grad_copies = get_ci_gradients(
        plain_solver, ci_copies, h1e, h2e, klasci,
    )
    grad_assembled_copies, _ = get_ci_gradients(
        trans_solver, ci_copies, h1e, h2e, klasci,
    )
    grad_ref = grad_copies[klasci.ref_cell]

    copy_error = max(
        np.max(np.abs(grad_frag - grad_ref))
        for grad_frag in grad_copies
    )
    grad_copies_translated = np.concatenate([grad_ref] * len(grad_copies))
    copy_full_error = np.max(np.abs(
        grad_overall_copies - grad_copies_translated
    ))
    copy_assembly_error = np.max(np.abs(
        grad_overall_copies - grad_assembled_copies
    ))

    # Next derive each scalar CI phase from the translated localized active
    # orbitals and compare direct cell gradients with translated reference
    # gradients in that phase convention.
    phases = klasci.get_phase_per_frag(lo_coeff)
    trans_solver.phase_per_frag = trans_solver._normalize_phase_per_frag(phases)
    ci_phased = trans_solver._unpack_cif(ci_ref_trial)

    grad_overall, grad_cells = get_ci_gradients(
        plain_solver, ci_phased, h1e, h2e, klasci,
    )
    grad_assembled, _ = get_ci_gradients(
        trans_solver, ci_phased, h1e, h2e, klasci,
    )

    grad_ref_phased = grad_cells[klasci.ref_cell]
    grad_cells_translated = [
        phase * grad_ref_phased for phase in trans_solver.phase_per_frag
    ]
    grad_individual_translated = np.concatenate(grad_cells_translated)

    cell_errors = np.asarray([
        np.max(np.abs(grad_direct - grad_translated))
        for grad_direct, grad_translated
        in zip(grad_cells, grad_cells_translated)
    ])
    overall_difference = grad_overall - grad_individual_translated
    assembly_difference = grad_overall - grad_assembled

    return {
        'reference_norm': np.linalg.norm(grad_ref),
        'phases': phases,
        'copy_error': copy_error,
        'copy_full_error': copy_full_error,
        'copy_assembly_error': copy_assembly_error,
        'cell_errors': cell_errors,
        'overall_difference_norm': np.linalg.norm(overall_difference),
        'overall_difference_max': np.max(np.abs(overall_difference)),
        'assembly_difference_max': np.max(np.abs(assembly_difference)),
    }


def print_results(case_name, results):
    '''Print the diagnostic results for one dimensional preset.'''
    print(f'\n=== {case_name} CI-gradient translation check ===')
    print('k-point mesh:', CASE_CONFIGS[case_name]['kmesh'])
    print('Reference CI-gradient norm:', results['reference_norm'])
    print('Orbital-derived fragment CI phases:', results['phases'])
    print('Maximum literal-copy error:', results['copy_error'])
    print('Full copy-gradient error:', results['copy_full_error'])
    print('Copy-gradient assembly error:', results['copy_assembly_error'])
    print('Per-cell direct-minus-translated maximum errors:')
    for cell_index, error in enumerate(results['cell_errors']):
        print(f'  cell {cell_index}: {error:.3e}')
    print('Overall direct-minus-translated difference norm:',
          results['overall_difference_norm'])
    print('Overall direct-minus-translated maximum difference:',
          results['overall_difference_max'])
    print('Direct-overall minus trans-symmetric assembly maximum difference:',
          results['assembly_difference_max'])


def validate_results(results):
    '''Assert that a case has a nonzero gradient and obeys translation.'''
    assert results['reference_norm'] > 1e-8
    assert results['copy_error'] < GRADIENT_TOL
    assert results['copy_full_error'] < GRADIENT_TOL
    assert results['copy_assembly_error'] < GRADIENT_TOL
    assert np.max(results['cell_errors']) < GRADIENT_TOL
    assert results['overall_difference_max'] < GRADIENT_TOL
    assert results['assembly_difference_max'] < GRADIENT_TOL


def run_case(case_name):
    '''Run and validate one of the 1D, 2D, or 3D presets.'''
    print(f'\nBuilding {case_name} case...')
    klasci, lo_coeff = build_klasci(case_name)
    results = compute_gradient_errors(klasci, lo_coeff)
    print_results(case_name, results)
    validate_results(results)
    return results


def parse_case_name(value):
    '''Normalize and validate a dimensional case supplied on the CLI.'''
    case_name = value.upper()
    if case_name not in CASE_CONFIGS:
        choices = ', '.join(CASE_CONFIGS)
        raise ValueError(f'unknown case {value!r}; choose from {choices}')
    return case_name


def main(argv=None):
    '''Run the dimensional cases requested on the command line.'''
    argv = sys.argv[1:] if argv is None else argv
    if any(value in ('-h', '--help') for value in argv):
        print('Usage: ci_grad_trans_symm.py [1D] [2D] [3D]')
        print('With no case arguments, all three cases are run.')
        return
    try:
        cases = [parse_case_name(value) for value in argv]
    except ValueError as error:
        raise SystemExit(str(error)) from error
    cases = cases or list(CASE_CONFIGS)
    for case_name in cases:
        run_case(case_name)


if __name__ == '__main__':
    main()
