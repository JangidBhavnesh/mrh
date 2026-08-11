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

