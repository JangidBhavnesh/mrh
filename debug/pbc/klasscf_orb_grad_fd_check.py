#!/usr/bin/env python

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from scipy import linalg

from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.mcscf.productstate import (
    ImpureProductStateFCISolver,
)


# Author: Bhavnesh Jangid

"""
Finite-difference check of the k-LASSCF orbital gradient.

The CI vectors are held fixed so that the numerical derivative tests only the
orbital part of the gradient.  Centered finite differences have an O(h**2)
error, so their slopes are linearly extrapolated against h**2 to h = 0.
"""


CASE_CONFIGS = {
    "1D core-active": {
        "lattice": (4.0, 10.0, 10.0),
        "kmesh": (2, 1, 1),
        "rotation_blocks": ("core-active",),
    },
    "2D active-virtual": {
        "lattice": (4.0, 4.0, 10.0),
        "kmesh": (2, 2, 1),
        "rotation_blocks": ("active-virtual",),
    },
    "3D all blocks": {
        "lattice": (4.0, 4.0, 4.0),
        "kmesh": (2, 2, 2),
        "rotation_blocks": (
            "core-active", "core-virtual", "active-virtual",
        ),
    },
}

LIH_DISTANCE = 1.6
TRIAL_STEPS = np.asarray([
    8e-2, 4e-2, 2e-2, 1e-2, 5e-3, 2.5e-3, 1.25e-3,
    1e-4, 1e-5, 1e-6, 1e-7,
])
PLOT_FILE = "klasscf_orb_grad_taylor_error.png"
ZERO_STEP_RTOL = 1e-4


def build_reference(case_name):
    """Build a LiH reference with core, active, and virtual orbitals."""
    config = CASE_CONFIGS[case_name]
    kmesh = config["kmesh"]

    cell = gto.Cell()
    cell.a = np.diag(config["lattice"])
    cell.atom = f"Li 0 0 0; H {LIH_DISTANCE} 0 0"
    cell.basis = "sto-3g"
    cell.unit = "Angstrom"
    cell.precision = 1e-12
    cell.ke_cutoff = 20
    cell.verbose = lib.logger.WARN
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

    # CI stationarity is not required: the same returned CI vectors are held
    # fixed in the analytic and finite-difference orbital derivatives.
    return klas, np.asarray(mo_ref)


def copy_ci(ci):
    return [[np.array(c, copy=True) for c in roots] for roots in ci]


def make_complex_orbital_direction(klas, rotation_blocks, seed):
    """Return a normalized complex anti-Hermitian orbital rotation."""
    nkpts, nmo = klas.nkpts, klas.mo_coeff.shape[-1]
    ncore = klas.ncore
    nocc = ncore + klas.ncas
    nvir = nmo - nocc
    spaces = {
        "core": slice(0, ncore),
        "active": slice(ncore, nocc),
        "virtual": slice(nocc, nmo),
    }
    dimensions = {
        "core": ncore,
        "active": klas.ncas,
        "virtual": nvir,
    }

    rng = np.random.default_rng(seed)
    kappa = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)
    for k in range(nkpts):
        for block_name in rotation_blocks:
            left_name, right_name = block_name.split("-")
            nleft = dimensions[left_name]
            nright = dimensions[right_name]
            if nleft == 0 or nright == 0:
                raise RuntimeError(
                    f"the {block_name} test needs nonempty {left_name} and "
                    f"{right_name} spaces"
                )
            block = (
                rng.standard_normal((nright, nleft))
                + 1j * rng.standard_normal((nright, nleft))
            )
            left = spaces[left_name]
            right = spaces[right_name]
            kappa[k, right, left] = block
            kappa[k, left, right] = -block.conj().T

    antihermiticity_error = np.max(
        np.abs(kappa + kappa.conj().transpose(0, 2, 1))
    )
    if antihermiticity_error > 1e-13:
        raise AssertionError("the orbital direction is not anti-Hermitian")

    norm = np.linalg.norm(kappa)
    if norm == 0:
        raise RuntimeError("the orbital direction has zero norm")
    return kappa / norm


def rotate_mos(mo_ref, kappa, step):
    """Apply the complex unitary exp(step * kappa) at every k-point."""
    return np.asarray([
        mo_k @ linalg.expm(step * kappa_k)
        for mo_k, kappa_k in zip(mo_ref, kappa)
    ])


def fixed_ci_energy(klas, mo_coeff, ci):
    """Evaluate the per-cell k-LAS energy without relaxing the CI vectors."""
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
    ci_state = [roots[0] for roots in ci]
    energy = solver.energy_elec(
        h1eff, h2eff, ci_state, klas.ncas_sub, klas.nelecas_sub,
        ecore=ecore,
    ) / klas.nkpts
    if abs(np.imag(energy)) > 1e-9:
        raise RuntimeError(f"fixed-CI energy has imaginary part {np.imag(energy)}")
    return float(np.real(energy))


def extrapolate_zero_step(steps, derivatives):
    """Fit D(h) = D(0) + a h**2 and return D(0)."""
    intercept, slope = np.polynomial.polynomial.polyfit(
        np.asarray(steps) ** 2, np.asarray(derivatives), 1,
    )
    return intercept, slope


def compute_gradient_error(case_name, seed=17):
    klas, mo_ref = build_reference(case_name)
    ci_ref = copy_ci(klas.ci)
    rotation_blocks = CASE_CONFIGS[case_name]["rotation_blocks"]
    kappa = make_complex_orbital_direction(klas, rotation_blocks, seed)

    gorb = klas.get_grad_orb(mo_coeff_kpts=mo_ref, ci=ci_ref)
    antihermiticity_error = np.max(
        np.abs(gorb + gorb.conj().transpose(0, 2, 1))
    )
    if antihermiticity_error > 1e-9:
        raise AssertionError("the analytic orbital gradient is not anti-Hermitian")

    # Both matrices contain both halves of each independent complex rotation.
    # The full-matrix contraction therefore uses 1/nkpts, equivalent to the
    # usual 2/nkpts contraction of packed independent orbital variables.
    analytic = np.real(np.vdot(gorb, kappa)) / klas.nkpts

    energy_ref = fixed_ci_energy(klas, mo_ref, ci_ref)
    derivatives = []
    taylor_remainders = []
    for step in TRIAL_STEPS:
        energy_plus = fixed_ci_energy(
            klas, rotate_mos(mo_ref, kappa, step), ci_ref,
        )
        energy_minus = fixed_ci_energy(
            klas, rotate_mos(mo_ref, kappa, -step), ci_ref,
        )
        derivatives.append((energy_plus - energy_minus) / (2.0 * step))
        # This is |E(x) - E(0) - g.x| / |x|.  Since ||kappa|| = 1,
        # the displacement norm is |step|.
        remainder = abs(energy_plus - energy_ref - step * analytic) / abs(step)
        taylor_remainders.append(remainder)

    derivatives = np.asarray(derivatives)
    taylor_remainders = np.asarray(taylor_remainders)
    extrapolated, fit_slope = extrapolate_zero_step(TRIAL_STEPS, derivatives)
    scale = max(abs(analytic), 1e-14)
    return {
        "nkpts": klas.nkpts,
        "rotation_blocks": rotation_blocks,
        "direction_norm": np.linalg.norm(kappa),
        "direction_imag_norm": np.linalg.norm(kappa.imag),
        "gradient_antihermiticity_error": antihermiticity_error,
        "analytic": analytic,
        "derivatives": derivatives,
        "absolute_errors": np.abs(derivatives - analytic),
        "relative_errors": np.abs(derivatives - analytic) / scale,
        "extrapolated": extrapolated,
        "extrapolated_relative_error": abs(extrapolated - analytic) / scale,
        "fit_slope": fit_slope,
        "taylor_remainders": taylor_remainders,
    }


def print_results(case_name, results):
    print(f"\n=== {case_name} k-LASSCF orbital-gradient check ===")
    print("k-point mesh:", CASE_CONFIGS[case_name]["kmesh"])
    print("rotation blocks:", ", ".join(results["rotation_blocks"]))
    print("number of k-points:", results["nkpts"])
    print("orbital-direction norm:", results["direction_norm"])
    print("imaginary-component norm:", results["direction_imag_norm"])
    print("gradient anti-Hermiticity error:",
          results["gradient_antihermiticity_error"])
    print("analytic directional derivative:", results["analytic"])
    print("\n step             centered FD          absolute error       relative error")
    for step, derivative, absolute, relative in zip(
            TRIAL_STEPS, results["derivatives"],
            results["absolute_errors"], results["relative_errors"]):
        print(
            f" {step:10.3e}    {derivative:16.9e}    "
            f"{absolute:16.8e}    {relative:16.8e}"
        )
    print("\nzero-step extrapolated derivative:", results["extrapolated"])
    print("zero-step extrapolated relative error:",
          results["extrapolated_relative_error"])


def validate_results(results):
    if results["direction_imag_norm"] < 1e-8:
        raise AssertionError("the test direction is not genuinely complex")
    if results["extrapolated_relative_error"] > ZERO_STEP_RTOL:
        raise AssertionError(
            "zero-step extrapolation disagrees with the analytic gradient: "
            f"{results['extrapolated_relative_error']:.3e} > "
            f"{ZERO_STEP_RTOL:.3e}"
        )


def test_1d_core_active():
    return compute_gradient_error("1D core-active", seed=17)


def test_2d_active_virtual():
    return compute_gradient_error("2D active-virtual", seed=23)


def test_3d_all_blocks():
    return compute_gradient_error("3D all blocks", seed=31)


def plot_taylor_remainders(all_results, output_path=PLOT_FILE):
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    for case_name, results in all_results.items():
        axis.loglog(
            TRIAL_STEPS, results["taylor_remainders"], "o-",
            linewidth=1.4, label=case_name,
        )
    axis.set_xlabel(r"$|x|$")
    axis.set_ylabel(r"$|E(x)-E(0)-g\cdot x|/|x|$")
    axis.set_title("k-LASSCF complex orbital-gradient Taylor error")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


if __name__ == "__main__":
    tests = (
        ("1D core-active", test_1d_core_active),
        # ("2D active-virtual", test_2d_active_virtual),
        # ("3D all blocks", test_3d_all_blocks),
    )
    all_results = {}
    for case_name, test in tests:
        results = test()
        print_results(case_name, results)
        validate_results(results)
        all_results[case_name] = results

    plot_path = plot_taylor_remainders(all_results)
    print(f"\nTaylor-error plot saved to: {plot_path}")
