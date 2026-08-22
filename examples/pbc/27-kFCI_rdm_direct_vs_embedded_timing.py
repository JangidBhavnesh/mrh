#!/usr/bin/env python

"""Benchmark direct and embedded-reference k-FCI RDM construction.

No SCF, CASCI, or MC-PDFT calculation is performed.  For each requested
total active-orbital count, this script constructs one normalized dense dummy
CI vector in a fixed momentum sector and times ``make_rdm12s`` using:

1. the direct packed momentum-sector implementation; and
2. the retained full-CI embedding implementation (``make_rdm12s_ref``).

The default scan is 2, 4, ..., 16 total orbitals.  It uses two active
orbitals and two neutral active electrons per k-point, then removes one beta
electron (``charge=+1``) to reproduce the charged graphene sector:

    total orbitals = 2 * nkpts
    total electrons = (nkpts alpha, nkpts - 1 beta)

Run from the repository root with::

    PYTHONPATH=.. python \
        examples/pbc/27-kFCI_rdm_direct_vs_embedded_timing.py

Use ``--charge 0`` for the neutral sectors.  Each completed size is flushed
immediately to CSV so earlier timings survive if a large embedded-reference
calculation is interrupted.
"""

import argparse
import csv
import gc
import math
import time
from pathlib import Path

import numpy as np

from pyscf import lib

from mrh.my_pyscf.pbc.fci import direct_spin1_kfci, krdm_helper


DEFAULT_ACTIVE_ORBITALS = tuple(range(2, 17, 2))
ACTIVE_ORBITALS_PER_KPOINT = 2
ACTIVE_ELECTRONS_PER_KPOINT = (1, 1)

RESULT_FIELDS = (
    "active_orbitals_total",
    "nkpts",
    "neleca",
    "nelecb",
    "target_k",
    "sector_ci_elements",
    "sector_ci_gib",
    "embedded_ci_elements",
    "embedded_ci_gib",
    "direct_seconds",
    "embedded_seconds",
    "speedup",
    "maximum_rdm_error",
    "status",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--active-orbitals",
        type=int,
        nargs="+",
        default=DEFAULT_ACTIVE_ORBITALS,
        help="total active-orbital sizes (default: 2 4 ... 16)",
    )
    parser.add_argument(
        "--charge",
        type=int,
        choices=(0, 1),
        default=1,
        help="remove one beta electron for +1, or remain neutral for 0",
    )
    parser.add_argument(
        "--target-k",
        type=int,
        default=0,
        help="total momentum-sector index (default: 0)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=32,
        help="OpenMP/BLAS threads for the direct implementation (default: 32)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12,
        help="base random-number seed (default: 12)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("krdm_direct_vs_embedded_timings.csv"),
        help="checkpoint CSV path",
    )
    return parser.parse_args()


def validate_args(args):
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    for norb in args.active_orbitals:
        if norb < ACTIVE_ORBITALS_PER_KPOINT:
            raise ValueError("active-orbital sizes must be at least 2")
        if norb % ACTIVE_ORBITALS_PER_KPOINT:
            raise ValueError(
                "each active-orbital size must be divisible by 2")


def electron_counts(nkpts, charge):
    neleca = nkpts * ACTIVE_ELECTRONS_PER_KPOINT[0]
    nelecb = nkpts * ACTIVE_ELECTRONS_PER_KPOINT[1] - charge
    if nelecb < 0:
        raise ValueError(
            f"charge={charge} is incompatible with nkpts={nkpts}")
    return neleca, nelecb


def make_dummy_ci(size, seed):
    """Allocate a dense complex CI vector without large RNG temporaries."""
    rng = np.random.default_rng(seed)
    ci = np.empty(size, dtype=np.complex128)
    rng.standard_normal(size=2 * size, out=ci.view(np.float64))
    ci /= np.linalg.norm(ci)
    return ci


def maximum_rdm_error(direct, reference):
    maximum = 0.0
    for direct_group, reference_group in zip(direct, reference):
        for direct_dm, reference_dm in zip(direct_group, reference_group):
            maximum = max(
                maximum,
                float(np.max(np.abs(direct_dm - reference_dm))),
            )
    return maximum


def time_rdm(function, ci, norb, nelec, nkpts, target_k):
    gc.collect()
    start = time.perf_counter()
    result = function(
        ci,
        norb,
        nelec,
        nkpts,
        target_k=target_k,
        reorder=True,
    )
    return time.perf_counter() - start, result


def warm_up():
    """Exclude shared-library loading from the reported size-two timing."""
    norb = 2
    nkpts = 1
    nelec = (1, 0)
    ci = np.asarray([1.0, 1.0j], dtype=np.complex128)
    ci /= np.linalg.norm(ci)
    for function in (
            krdm_helper.make_rdm12s,
            krdm_helper.make_rdm12s_ref):
        function(ci, norb, nelec, nkpts, target_k=0, reorder=True)


def benchmark_size(norb, charge, target_k, threads, seed):
    nkpts = norb // ACTIVE_ORBITALS_PER_KPOINT
    nelec = electron_counts(nkpts, charge)
    resolved_target_k = target_k % nkpts
    sector_size = direct_spin1_kfci.sector_size(
        norb, nelec, nkpts, resolved_target_k)
    embedded_size = (
        math.comb(norb, nelec[0]) * math.comb(norb, nelec[1]))
    complex_bytes = np.dtype(np.complex128).itemsize

    ci = make_dummy_ci(sector_size, seed)

    lib.num_threads(threads)
    direct_seconds, direct = time_rdm(
        krdm_helper.make_rdm12s,
        ci,
        norb,
        nelec,
        nkpts,
        resolved_target_k,
    )

    # The reference path intentionally retains its current threading behavior.
    embedded_seconds, reference = time_rdm(
        krdm_helper.make_rdm12s_ref,
        ci,
        norb,
        nelec,
        nkpts,
        resolved_target_k,
    )
    error = maximum_rdm_error(direct, reference)
    if error > 1e-10:
        raise AssertionError(
            f"direct/reference RDM mismatch for {norb} orbitals: {error}")

    row = {
        "active_orbitals_total": norb,
        "nkpts": nkpts,
        "neleca": nelec[0],
        "nelecb": nelec[1],
        "target_k": resolved_target_k,
        "sector_ci_elements": sector_size,
        "sector_ci_gib": sector_size * complex_bytes / 1024**3,
        "embedded_ci_elements": embedded_size,
        "embedded_ci_gib": embedded_size * complex_bytes / 1024**3,
        "direct_seconds": direct_seconds,
        "embedded_seconds": embedded_seconds,
        "speedup": embedded_seconds / direct_seconds,
        "maximum_rdm_error": error,
        "status": "passed",
    }

    del direct, reference, ci
    gc.collect()
    return row


def print_result(row):
    print(
        f"{row['active_orbitals_total']:2d} orbitals  "
        f"nk={row['nkpts']:2d}  "
        f"CI={row['sector_ci_elements']:12d}  "
        f"direct={row['direct_seconds']:12.6f} s  "
        f"embedded={row['embedded_seconds']:12.6f} s  "
        f"speedup={row['speedup']:10.2f}x  "
        f"error={row['maximum_rdm_error']:.3e}",
        flush=True,
    )


def main():
    args = parse_args()
    validate_args(args)
    lib.num_threads(args.threads)
    warm_up()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        output_file.flush()

        for index, norb in enumerate(args.active_orbitals):
            row = benchmark_size(
                norb=norb,
                charge=args.charge,
                target_k=args.target_k,
                threads=args.threads,
                seed=args.seed + index,
            )
            writer.writerow(row)
            output_file.flush()
            print_result(row)


if __name__ == "__main__":
    main()
