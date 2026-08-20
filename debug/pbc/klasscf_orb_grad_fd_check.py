#!/usr/bin/env python

"""Finite-difference checks for the complex k-LASSCF orbital gradient.

The CI vectors are held fixed so the numerical derivative probes only the
orbital gradient. Centered differences are extrapolated linearly in ``h**2``
to remove their leading truncation error.

Run all cases from the repository's parent directory with

``python mrh/debug/pbc/klasscf_orb_grad_fd_check.py``.
"""

import numpy as np
from scipy import linalg

from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.mcscf.klasscf import get_grad_orb
from mrh.my_pyscf.pbc.mcscf.productstate import (
    ImpureProductStateFCISolver,
)


FD_STEPS = np.asarray([1e-2, 5e-3, 2.5e-3, 1.25e-3])
ZERO_STEP_RTOL = 1e-5
CASES = {
    "1D core-active": {
        "lattice": (4.0, 10.0, 10.0),
        "kmesh": (2, 1, 1),
        "rotation_blocks": ("core-active",),
        "seed": 17,
    },
    "2D active-virtual": {
        "lattice": (4.0, 4.0, 10.0),
        "kmesh": (2, 2, 1),
        "rotation_blocks": ("active-virtual",),
        "seed": 23,
    },
    "3D all nonredundant blocks": {
        "lattice": (4.0, 4.0, 4.0),
        "kmesh": (2, 2, 2),
        "rotation_blocks": (
            "core-active", "core-virtual", "active-virtual",
        ),
        "seed": 31,
    },
}


def build_reference(lattice, kmesh):
    """Build the fixed-CI periodic LiH reference."""
    cell = gto.Cell()
    cell.a = np.diag(lattice)
    cell.atom = "Li 0 0 0; H 1.6 0 0"
    cell.basis = "sto-3g"
    cell.unit = "Angstrom"
    cell.precision = 1e-12
    cell.ke_cutoff = 20
    cell.verbose = lib.logger.QUIET
    cell.build()

    kpts = cell.make_kpts(kmesh, wrap_around=True)
    kmf = scf.KRHF(cell, kpts=kpts).density_fit()
    kmf.exxdiv = None
    kmf.max_cycle = 0
    kmf.kernel()

    active_labels = ["Li 2s", "H 1s"]
    mo_coeff = avas.kernel(kmf, active_labels, minao=cell.basis)[2]
    klas = mcscf.KLASCI(kmf, 2, (1, 1), kmesh=kmesh)
    klas.conv_tol_grad = 1e-8
    klas.conv_tol_self = 1e-10
    mo_ref = klas.localize_init_guess(active_labels, mo_coeff=mo_coeff)
    klas.kernel(mo_ref)
    return klas, np.asarray(mo_ref)


def copy_ci(ci):
    return [[np.array(c, copy=True) for c in roots] for roots in ci]


def make_direction(klas, rotation_blocks, seed):
    """Build a normalized complex anti-Hermitian orbital direction."""
    nkpts, nmo = klas.nkpts, klas.mo_coeff.shape[-1]
    ncore = klas.ncore
    nocc = ncore + klas.ncas
    spaces = {
        "core": slice(0, ncore),
        "active": slice(ncore, nocc),
        "virtual": slice(nocc, nmo),
    }
    dimensions = {
        "core": ncore,
        "active": klas.ncas,
        "virtual": nmo - nocc,
    }

    rng = np.random.default_rng(seed)
    kappa = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)
    for k in range(nkpts):
        for block_name in rotation_blocks:
            left_name, right_name = block_name.split("-")
            nleft = dimensions[left_name]
            nright = dimensions[right_name]
            if nleft == 0 or nright == 0:
                raise RuntimeError(f"empty orbital space in {block_name}")
            block = (
                rng.standard_normal((nright, nleft))
                + 1j * rng.standard_normal((nright, nleft))
            )
            left = spaces[left_name]
            right = spaces[right_name]
            kappa[k, right, left] = block
            kappa[k, left, right] = -block.conj().T

    kappa /= np.linalg.norm(kappa)
    return kappa


def rotate_mos(mo_ref, kappa, step):
    """Apply ``exp(step * kappa)`` to the MOs at every k-point."""
    return np.asarray([
        mo_k @ linalg.expm(step * kappa_k)
        for mo_k, kappa_k in zip(mo_ref, kappa)
    ])


def fixed_ci_energy(klas, mo_coeff, ci):
    """Evaluate the per-cell k-LAS energy without relaxing CI vectors."""
    h1eff, ecore = klas.h1e_for_cas(
        mo_coeff=mo_coeff, ncas=klas.ncas, ncore=klas.ncore,
    )
    h2eff = klas.get_h2cas(mo_coeff)
    fcisolvers = [box.fcisolvers[0] for box in klas.fciboxes]
    solver = ImpureProductStateFCISolver(
        fcisolvers,
        lweights=[[1.0] for _ in fcisolvers],
        stdout=klas.stdout,
        verbose=lib.logger.QUIET,
    )
    energy = solver.energy_elec(
        h1eff, h2eff, [roots[0] for roots in ci],
        klas.ncas_sub, klas.nelecas_sub, ecore=ecore,
    ) / klas.nkpts
    if abs(np.imag(energy)) > 1e-9:
        raise AssertionError(f"energy has imaginary part {np.imag(energy)}")
    return float(np.real(energy))


def check_case(case_name, config):
    """Run and validate one directional finite-difference check."""
    klas, mo_ref = build_reference(config["lattice"], config["kmesh"])
    ci_ref = copy_ci(klas.ci)
    kappa = make_direction(
        klas, config["rotation_blocks"], config["seed"],
    )
    gorb = get_grad_orb(klas, mo_coeff_kpts=mo_ref, ci=ci_ref)

    np.testing.assert_allclose(
        kappa + kappa.conj().transpose(0, 2, 1), 0.0, atol=1e-13,
    )
    np.testing.assert_allclose(
        gorb + gorb.conj().transpose(0, 2, 1), 0.0, atol=1e-9,
    )
    if np.linalg.norm(kappa.imag) <= 1e-8:
        raise AssertionError("orbital direction is not genuinely complex")

    analytic = np.real(np.vdot(gorb, kappa)) / klas.nkpts
    finite_differences = []
    for step in FD_STEPS:
        energy_plus = fixed_ci_energy(
            klas, rotate_mos(mo_ref, kappa, step), ci_ref,
        )
        energy_minus = fixed_ci_energy(
            klas, rotate_mos(mo_ref, kappa, -step), ci_ref,
        )
        finite_differences.append(
            (energy_plus - energy_minus) / (2.0 * step)
        )

    zero_step = np.polynomial.polynomial.polyfit(
        FD_STEPS ** 2, finite_differences, 1,
    )[0]
    relative_error = abs(zero_step - analytic) / max(abs(analytic), 1e-14)
    print(f"{case_name}: analytic = {analytic:.12e}")
    print(f"{case_name}: extrapolated FD = {zero_step:.12e}")
    print(f"{case_name}: relative error = {relative_error:.3e}")
    if relative_error >= ZERO_STEP_RTOL:
        raise AssertionError(
            f"relative error {relative_error:.3e} exceeds "
            f"{ZERO_STEP_RTOL:.3e}"
        )


if __name__ == "__main__":
    for name, case in CASES.items():
        check_case(name, case)
