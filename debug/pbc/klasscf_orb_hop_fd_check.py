#!/usr/bin/env python

"""Complex-orbital finite-difference check of the k-LASSCF orbital hop.

For each dimensional preset, the script compares the analytic orbital-only
Hessian action with centered finite differences of the fixed-CI orbital
gradient. Centered differences have an O(h**2) error, so the vector-valued
derivatives are fitted linearly against h**2 and extrapolated to h = 0.
"""

import argparse

import matplotlib
import numpy as np
from scipy import linalg

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.mcscf.klasscf import KLASSCF_HessianOperator


# Author: Bhavnesh Jangid


CASE_CONFIGS = {
    "1D": {
        "label": "1D core-active",
        "lattice": (4.0, 10.0, 10.0),
        "kmesh": (3, 1, 1),
        "rotation_blocks": ("core-active",),
        "seed": 41,
    },
    "2D": {
        "label": "2D active-virtual",
        "lattice": (4.0, 4.0, 10.0),
        "kmesh": (2, 2, 1),
        "rotation_blocks": ("active-virtual",),
        "seed": 43,
    },
    "3D": {
        "label": "3D all nonredundant blocks",
        "lattice": (4.0, 4.0, 4.0),
        "kmesh": (2, 2, 2),
        "rotation_blocks": (
            "core-active", "core-virtual", "active-virtual",
        ),
        "seed": 47,
    },
}

TRIAL_STEPS = np.asarray([2e-2, 1e-2, 5e-3, 2.5e-3, 1.25e-3])
ZERO_STEP_RTOL = 1e-7
PLOT_FILE = "klasscf_orb_hop_extrapolation.png"
LIH_DISTANCE = 1.6


def build_reference(config):
    """Build the LiH k-LASCI reference for one dimensional preset."""
    cell = gto.Cell()
    cell.a = np.diag(config["lattice"])
    cell.atom = f"Li 0 0 0; H {LIH_DISTANCE} 0 0"
    cell.basis = "sto-3g"
    cell.unit = "Angstrom"
    cell.precision = 1e-12
    cell.ke_cutoff = 20
    cell.verbose = lib.logger.WARN
    cell.build()

    kpts = cell.make_kpts(config["kmesh"], wrap_around=True)
    kmf = scf.KRHF(cell, kpts=kpts).density_fit()
    kmf.exxdiv = None
    kmf.max_cycle = 0
    kmf.kernel()

    active_labels = ["Li 2s", "H 1s"]
    mo_coeff = avas.kernel(kmf, active_labels, minao=cell.basis)[2]
    klas = mcscf.KLASCI(kmf, 2, (1, 1), kmesh=config["kmesh"])
    klas.conv_tol_grad = 1e-8
    klas.conv_tol_self = 1e-10
    mo_ref = klas.localize_init_guess(active_labels, mo_coeff=mo_coeff)
    klas.kernel(mo_ref)
    return klas, np.asarray(mo_ref)


def copy_ci(ci):
    """Copy the cell/root CI layout without changing its array shapes."""
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

    norm = np.linalg.norm(kappa)
    if norm == 0:
        raise RuntimeError("the orbital direction is zero")
    return kappa / norm


def rotate_mos(mo_ref, kappa, step):
    """Apply exp(step * kappa) independently at every k-point."""
    return np.asarray([
        mo_k @ linalg.expm(step * kappa_k)
        for mo_k, kappa_k in zip(mo_ref, kappa)
    ])


def compute_hop_errors(case_name):
    """Compute finite-step and zero-step orbital-hop errors."""
    config = CASE_CONFIGS[case_name]
    klas, mo_ref = build_reference(config)
    ci_ref = copy_ci(klas.ci)
    kappa = make_direction(
        klas, config["rotation_blocks"], config["seed"],
    )
    ugg = klas.get_ugg(mo_coeff=mo_ref, ci=ci_ref)
    hop = KLASSCF_HessianOperator(
        klas, ugg, mo_coeff=mo_ref, ci=ci_ref,
    )
    hop.level_shift = 0.0

    trial = np.zeros(ugg.nvar_tot, dtype=np.complex128)
    trial[:ugg.nvar_orb] = ugg.pack_orb(kappa)
    combined_action = hop._matvec(trial)
    ci_action_norm = np.linalg.norm(combined_action[ugg.nvar_orb:])

    # The molecular Hessian convention packs kappa2/2.
    analytic = 2.0 * combined_action[:ugg.nvar_orb]

    connection = np.asarray([
        (fock @ kappa_k - kappa_k @ fock) / 2.0
        for fock, kappa_k in zip(hop.fock1, kappa)
    ])
    connection -= connection.conj().transpose(0, 2, 1)
    packed_connection = ugg.pack_orb(connection)

    finite_difference_hops = []
    for step in TRIAL_STEPS:
        gradient_plus = ugg.pack_orb(klas.get_grad_orb(
            mo_coeff_kpts=rotate_mos(mo_ref, kappa, step),
            ci=ci_ref,
        ))
        gradient_minus = ugg.pack_orb(klas.get_grad_orb(
            mo_coeff_kpts=rotate_mos(mo_ref, kappa, -step),
            ci=ci_ref,
        ))
        derivative = (gradient_plus - gradient_minus) / (2.0 * step)
        finite_difference_hops.append(derivative - packed_connection)

    finite_difference_hops = np.asarray(finite_difference_hops)
    extrapolated, fit_slope = np.polynomial.polynomial.polyfit(
        TRIAL_STEPS ** 2, finite_difference_hops, 1,
    )
    scale = max(np.linalg.norm(analytic), 1e-14)
    absolute_errors = np.linalg.norm(
        finite_difference_hops - analytic[None, :], axis=1,
    )
    zero_step_error = np.linalg.norm(extrapolated - analytic)

    return {
        "label": config["label"],
        "kmesh": config["kmesh"],
        "rotation_blocks": config["rotation_blocks"],
        "direction_norm": np.linalg.norm(kappa),
        "direction_imag_norm": np.linalg.norm(kappa.imag),
        "analytic_norm": np.linalg.norm(analytic),
        "ci_action_norm": ci_action_norm,
        "absolute_errors": absolute_errors,
        "relative_errors": absolute_errors / scale,
        "extrapolated": extrapolated,
        "fit_slope": fit_slope,
        "zero_step_error": zero_step_error,
        "zero_step_relative_error": zero_step_error / scale,
    }


def print_results(results):
    """Print the centered-difference convergence table."""
    print(f"\n=== {results['label']} orbital-hop extrapolation ===")
    print("k-point mesh:", results["kmesh"])
    print("rotation blocks:", ", ".join(results["rotation_blocks"]))
    print("direction norm:", results["direction_norm"])
    print("imaginary-component norm:", results["direction_imag_norm"])
    print("analytic hop norm:", results["analytic_norm"])
    print("CI output norm:", results["ci_action_norm"])
    print("\n step             absolute error       relative error")
    for step, absolute, relative in zip(
            TRIAL_STEPS, results["absolute_errors"],
            results["relative_errors"]):
        print(
            f" {step:10.3e}    {absolute:16.8e}    {relative:16.8e}"
        )
    print("\nzero-step extrapolated absolute error:",
          results["zero_step_error"])
    print("zero-step extrapolated relative error:",
          results["zero_step_relative_error"])


def validate_results(results):
    """Validate the complex orbital direction and OO extrapolation."""
    if results["direction_imag_norm"] < 1e-8:
        raise AssertionError("the orbital direction is not genuinely complex")
    if not np.isfinite(results["ci_action_norm"]):
        raise AssertionError("the orbital-input CI response is not finite")
    if results["zero_step_relative_error"] > ZERO_STEP_RTOL:
        raise AssertionError(
            "zero-step extrapolation disagrees with the analytic hop: "
            f"{results['zero_step_relative_error']:.3e} > "
            f"{ZERO_STEP_RTOL:.3e}"
        )


def plot_errors(all_results, output_path=PLOT_FILE):
    """Plot finite-step relative errors for every selected case."""
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    for results in all_results.values():
        axis.loglog(
            TRIAL_STEPS, results["relative_errors"], "o-",
            linewidth=1.4, label=results["label"],
        )
    reference = next(iter(all_results.values()))["relative_errors"][0]
    axis.loglog(
        TRIAL_STEPS,
        reference * (TRIAL_STEPS / TRIAL_STEPS[0]) ** 2,
        ":", color="0.35", linewidth=1.2, label=r"$O(h^2)$ reference",
    )
    axis.set_xlabel("Orbital-rotation step h")
    axis.set_ylabel("Relative orbital-hop error")
    axis.set_title("k-LASSCF orbital-hop linear extrapolation")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", action="append", choices=tuple(CASE_CONFIGS),
        help="Run only this dimensional preset; repeat to select several.",
    )
    parser.add_argument(
        "--plot", default=PLOT_FILE,
        help="Output path for the relative-error plot.",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Do not write a plot.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_cases = args.case or tuple(CASE_CONFIGS)
    all_results = {}
    for selected_case in selected_cases:
        case_results = compute_hop_errors(selected_case)
        print_results(case_results)
        validate_results(case_results)
        all_results[selected_case] = case_results

    if not args.no_plot:
        plot_path = plot_errors(all_results, output_path=args.plot)
        print(f"\nOrbital-hop error plot saved to: {plot_path}")
