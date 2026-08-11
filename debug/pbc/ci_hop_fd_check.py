#!/usr/bin/env python

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.mcscf.klasscf import KLASSCF_HessianOperator
from mrh.my_pyscf.pbc.mcscf.productstate import PBCProductStateFCISolver


# Author: Bhavnesh Jangid

'''
CI-hop finite-difference check for k-LASSCF.
'''

# Different dimensional cases:
CASE_CONFIGS = {
    '1D': {
        'lattice': (4.0, 10.0, 10.0),
        'kmesh': (2, 1, 1),
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

# Some global variable:
HH_DISTANCE = 1.5
TRIAL_NORMS = np.asarray([
    1e-1, 5e-2, 1e-2, 5e-3, 1e-3,
    5e-4, 1e-4, 5e-5, 1e-5, 5e-6,
])

def build_cell(lattice):
    '''
    Build the periodic H2 cell used by a dimensional preset.
    '''
    cell = gto.Cell()
    cell.a = np.diag(lattice)
    cell.atom = f'H 0 0 0; H {HH_DISTANCE} 0 0'
    cell.basis = '6-31g'
    cell.unit = 'Angstrom'
    cell.precision = 1e-12
    cell.ke_cutoff = 20
    cell.verbose = lib.logger.WARN
    cell.build()
    return cell


def run_klasci(case_name):
    '''
    Build and converge the k-LASCI reference for given dimensional case.
    '''
    config = CASE_CONFIGS[case_name]
    kmesh = list(config['kmesh'])
    cell = build_cell(config['lattice'])
    kpts = cell.make_kpts(kmesh, wrap_around=True)

    kmf = scf.KRHF(cell, kpts=kpts).density_fit()
    kmf.exxdiv = None
    kmf.max_cycle = 0
    kmf.kernel()

    mo_coeff = avas.kernel(kmf, ['H 1s'], minao=cell.basis)[2]

    klasci = mcscf.KLASCI(kmf, 2, (1, 1), kmesh=kmesh)
    klasci.conv_tol_grad = 1e-8
    klasci.conv_tol_self = 1e-10
    lo_coeff = klasci.localize_init_guess(['H 1s'], mo_coeff=mo_coeff)
    klasci.kernel(lo_coeff)

    return klasci, lo_coeff

def copy_ci(ci):
    return [[np.array(c0, copy=True) 
             for c0 in ci0_r] 
             for ci0_r in ci]

def flatten_ci(ci):
    '''
    Flatten a cell/root nested CI list in Hessian-operator order.
    '''
    return np.concatenate([
        np.asarray(c0).reshape(-1) 
        for ci0_r in ci 
        for c0 in ci0_r])


def build_product_solver(klasci):
    '''
    Build the full-cell product solver for the single global root.
    '''
    fcisolvers = [box.fcisolvers[0] 
                  for box in klasci.fciboxes]
    return PBCProductStateFCISolver(fcisolvers)


def get_h1frs(product_solver, h1e, h2e, ci, klasci):
    '''
    Build root-indexed effective one-electron Hamiltonian blocks.
    '''
    ci_cells = [ci0_r[0] for ci0_r in ci]
    h1eff = product_solver.project_hfrag(
        h1e, h2e, ci_cells, klasci.ncas_sub, klasci.nelecas_sub,
    )[0]
    return [np.asarray(h1eff_f)[None, ...] 
            for h1eff_f in h1eff]

def get_ci_gradient(hop, product_solver, h1e, h2e, ci, klasci):
    '''
    Evaluate the determinant-basis CI gradient 
    Reminder: ``g_c = 2 Q H_eff c``.
    '''
    h1frs = get_h1frs(product_solver, h1e, h2e, ci, klasci)
    hc = hop.Hci_all(None, h1frs, h2e, ci)
    gradient = []
    for hc_r, ci0_r in zip(hc, ci):
        for hc0, c0 in zip(hc_r, ci0_r):
            residual = hc0 - np.vdot(c0, hc0) * c0
            gradient.append((2.0 * residual).reshape(-1))
    return np.concatenate(gradient)


def make_trial_direction(klasci, ci, seed=17):
    '''
    Build a normalized complex, spin-adapted tangent CI direction.
    '''
    dtype = ci[0][0].dtype
    rng = np.random.default_rng(seed)
    direction = []
    for fcibox, ci0_r in zip(klasci.fciboxes, ci):
        direction_r = []
        for solver, c0 in zip(fcibox.fcisolvers, ci0_r):
            transformer = solver.transformer
            d_csf = (
                rng.standard_normal(transformer.ncsf)
                + 1j * rng.standard_normal(transformer.ncsf)
            )
            d_csf = d_csf.astype(dtype)
            d_real = transformer.vec_csf2det(
                d_csf.real, order='C', normalize=False,
            )
            d_imag = transformer.vec_csf2det(
                d_csf.imag, order='C', normalize=False,
            )
            d = d_real.astype(np.complex128)
            d.real = d_real
            d.imag = d_imag
            d = d.reshape(np.shape(c0))
            d -= np.vdot(c0, d) * c0
            direction_r.append(d)
        direction.append(direction_r)

    norm = np.linalg.norm(flatten_ci(direction))
    if norm == 0:
        raise RuntimeError('the projected trial CI direction is zero')
    out_x = [[d / norm for d in direction_r] 
             for direction_r in direction]
    return out_x

def displace_ci(ci, direction, step):
    '''
    Apply and normalize a CI displacement independently in every cell.
    '''
    displaced = []
    for ci0_r, direction_r in zip(ci, direction):
        displaced_r = []
        for c0, d in zip(ci0_r, direction_r):
            c1 = c0 + step * d
            c1 /= np.linalg.norm(c1)
            displaced_r.append(c1)
        displaced.append(displaced_r)
    return displaced

def extrapolate_zero_step(trial_norms, finite_difference_hops):
    '''
    Extrapolate centered finite differences linearly in the squared step.
    '''
    fit_slice = slice(2, 7)
    x = trial_norms[fit_slice] ** 2
    y = np.asarray(finite_difference_hops)[fit_slice]
    coefficients = np.polynomial.polynomial.polyfit(x, y, 1)
    return coefficients[0]



def compute_hop_errors(case_name):
    '''
    Compute analytic and finite-difference CI hops for one case.
    '''
    klasci, lo_coeff = run_klasci(case_name)
    h1e = klasci.h1e_for_cas(mo_coeff=lo_coeff, 
                             ncas=klasci.ncas, 
                             ncore=klasci.ncore,)[0]
    h2e = klasci.get_h2cas(lo_coeff)
    product_solver = build_product_solver(klasci)
    ci0 = copy_ci(klasci.ci)
    h1frs = get_h1frs(product_solver, h1e, h2e, ci0, klasci)

    hop = KLASSCF_HessianOperator(
        klasci, None, mo_coeff=lo_coeff, ci=ci0,
        h1eff=h1frs, h2eff=h2e,
    )

    # The level shift is an iterative-solver aid, not part of the derivative
    # of the physical CI gradient used in this finite-difference check.
    hop.level_shift = 0.0

    direction = make_trial_direction(klasci, ci0)
    direction_flat = flatten_ci(direction)
    analytic_hop = hop.matvec(direction_flat)
    gradient_ref = get_ci_gradient(
        hop, product_solver, h1e, h2e, ci0, klasci,)

    finite_difference_hops = []
    absolute_errors = []
    relative_errors = []
    for trial_norm in TRIAL_NORMS:
        ci_plus = displace_ci(ci0, direction, trial_norm)
        ci_minus = displace_ci(ci0, direction, -trial_norm)
        gradient_plus = get_ci_gradient(
            hop, product_solver, h1e, h2e, ci_plus, klasci,)
        gradient_minus = get_ci_gradient(
            hop, product_solver, h1e, h2e, ci_minus, klasci,)

        finite_difference_hop = (gradient_plus - gradient_minus) / (2.0 * trial_norm)
        error = np.linalg.norm(analytic_hop - finite_difference_hop)
        finite_difference_hops.append(finite_difference_hop)
        absolute_errors.append(error)
        relative_errors.append(error / np.linalg.norm(analytic_hop))

    zero_step_hop = extrapolate_zero_step(TRIAL_NORMS, finite_difference_hops,)
    zero_step_error = np.linalg.norm(analytic_hop - zero_step_hop)

    results = {
        'ci_size': direction_flat.size,
        'direction_norm': np.linalg.norm(direction_flat),
        'reference_gradient_norm': np.linalg.norm(gradient_ref),
        'analytic_hop_norm': np.linalg.norm(analytic_hop),
        'absolute_errors': np.asarray(absolute_errors),
        'relative_errors': np.asarray(relative_errors),
        'zero_step_error': zero_step_error,
        'zero_step_relative_error': (
            zero_step_error / np.linalg.norm(analytic_hop)
        ),
    }
    return results


def print_results(case_name, results):
    '''Print the step-size convergence table.'''
    print(f'\n=== {case_name} k-LASSCF CI-hop finite-difference check ===')
    print('k-point mesh:', CASE_CONFIGS[case_name]['kmesh'])
    print('CI vector size:', results['ci_size'])
    print('Trial direction norm:', results['direction_norm'])
    print('Reference CI-gradient norm:', results['reference_gradient_norm'])
    print('Analytic CI-hop norm:', results['analytic_hop_norm'])
    print('\n trial norm       absolute error       relative error')
    for norm, absolute, relative in zip(
            TRIAL_NORMS,
            results['absolute_errors'],
            results['relative_errors']):
        print(f' {norm:10.3e}    {absolute:16.8e}    {relative:16.8e}')
    print('\nMinimum relative error:', np.min(results['relative_errors']))
    print('Zero-step extrapolated absolute error:',
          results['zero_step_error'])
    print('Zero-step extrapolated relative error:',
          results['zero_step_relative_error'])

def plot_errors(case_name, results, output_path=None):
    if output_path is None:
        output_path = f'ci_hop_error_{case_name.lower()}.png'
   
    errors = results['relative_errors']
    minimum_index = int(np.argmin(errors))

    figure, axis = plt.subplots(figsize=(6.0, 4.2))
    axis.loglog(TRIAL_NORMS, errors, 'o-', linewidth=1.5,
        label='Centered finite difference',
    )
    quadratic_reference = errors[0] * (
        TRIAL_NORMS / TRIAL_NORMS[0]
    ) ** 2
    axis.loglog(
        TRIAL_NORMS, quadratic_reference, ':', color='0.4',
        linewidth=1.2, label=r'$O(\epsilon^2)$ reference',
    )
    axis.scatter(
        TRIAL_NORMS[minimum_index], errors[minimum_index],
        color='tab:red', zorder=3, label='Minimum error',
    )
    axis.axhline(
        results['zero_step_relative_error'], color='tab:green',
        linestyle='--', linewidth=1.2, label='Zero-step extrapolation',
    )
    axis.set_xlabel('Trial-vector norm')
    axis.set_ylabel('Relative CI-hop error')
    axis.set_title(f'{case_name} k-LASSCF CI-hop accuracy')
    axis.grid(True, which='both', alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path

def validate_results(results):
    '''
    Check that reducing the trial norm improves the hop comparison.
    '''
    if results['reference_gradient_norm'] >= 1e-7:
        raise AssertionError('the reference CI gradient is not converged')
    if results['analytic_hop_norm'] == 0:
        raise AssertionError('the analytic CI hop is zero')
    if np.min(results['relative_errors']) >= results['relative_errors'][0]:
        raise AssertionError(
            'the finite-difference hop did not improve as the trial norm fell'
        )
    if results['zero_step_relative_error'] >= results['relative_errors'][0]:
        raise AssertionError(
            'the zero-step extrapolation did not improve the comparison'
        )

if __name__ == '__main__':
    for case in CASE_CONFIGS.keys():
        results = compute_hop_errors(case)
        print_results(case, results)
        plot_path = plot_errors(case, results)
        print(f'Plot saved to: {plot_path}')
        validate_results(results)