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
