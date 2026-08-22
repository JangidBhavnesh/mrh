#!/usr/bin/env python

"""Benchmark direct momentum-sector k-FCI RDM construction.

No SCF, CASCI, or MC-PDFT calculation is performed.  For each requested
total active-orbital count, this script constructs one normalized dense dummy
CI vector in a fixed momentum sector and times the direct packed
``make_rdm12s`` implementation.

The scan is 2, 4, ..., 16 total orbitals.  It uses two active
orbitals and two neutral active electrons per k-point, then removes one beta
electron (``charge=+1``) to reproduce the charged graphene sector:

    total orbitals = 2 * nkpts
    total electrons = (nkpts alpha, nkpts - 1 beta)

Run from the repository root with::

    PYTHONPATH=.. python \
        examples/pbc/27-kFCI_rdm_direct_vs_embedded_timing.py

Edit the calculation settings below to change the charge, target momentum,
thread count, random seed, or output file.  Each completed size is flushed to
CSV immediately.
"""

import csv
import gc
import time
from pathlib import Path

import numpy as np

from pyscf import lib

from mrh.my_pyscf.pbc.fci import direct_spin1_kfci, krdm_helper


# ------------------------------------------------------------
# Calculation settings
# ------------------------------------------------------------

active_orbitals = range(2, 17, 2)
active_orbitals_per_kpoint = 2
active_electrons_per_kpoint = (1, 1)
charge = +1
target_k = 0
num_threads = 32
seed = 12
output = Path("krdm_direct_timings.csv")

RESULT_FIELDS = (
    "active_orbitals_total",
    "nkpts",
    "neleca",
    "nelecb",
    "target_k",
    "sector_ci_elements",
    "sector_ci_gib",
    "direct_seconds",
)


def electron_counts(nkpts, charge):
    neleca = nkpts * active_electrons_per_kpoint[0]
    nelecb = nkpts * active_electrons_per_kpoint[1] - charge
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
    krdm_helper.make_rdm12s(
        ci, norb, nelec, nkpts, target_k=0, reorder=True)


def benchmark_size(norb, charge, target_k, threads, seed):
    nkpts = norb // active_orbitals_per_kpoint
    nelec = electron_counts(nkpts, charge)
    resolved_target_k = target_k % nkpts
    sector_size = direct_spin1_kfci.sector_size(
        norb, nelec, nkpts, resolved_target_k)
    complex_bytes = np.dtype(np.complex128).itemsize

    ci = make_dummy_ci(sector_size, seed)

    lib.num_threads(threads)
    direct_seconds, direct_rdm = time_rdm(
        krdm_helper.make_rdm12s,
        ci,
        norb,
        nelec,
        nkpts,
        resolved_target_k,
    )

    row = {
        "active_orbitals_total": norb,
        "nkpts": nkpts,
        "neleca": nelec[0],
        "nelecb": nelec[1],
        "target_k": resolved_target_k,
        "sector_ci_elements": sector_size,
        "sector_ci_gib": sector_size * complex_bytes / 1024**3,
        "direct_seconds": direct_seconds,
    }

    del direct_rdm, ci
    gc.collect()
    return row


def print_result(row):
    print(
        f"active_space={row['active_orbitals_total']:2d}  "
        f"direct_rdm_time={row['direct_seconds']:.6f} s",
        flush=True,
    )


# ------------------------------------------------------------
# Direct RDM timing scan
# ------------------------------------------------------------

if num_threads < 1:
    raise ValueError("num_threads must be positive")

for norb in active_orbitals:
    if norb < active_orbitals_per_kpoint:
        raise ValueError("active-orbital sizes must be at least 2")
    if norb % active_orbitals_per_kpoint:
        raise ValueError("each active-orbital size must be divisible by 2")

lib.num_threads(num_threads)
warm_up()

output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", newline="") as output_file:
    writer = csv.DictWriter(output_file, fieldnames=RESULT_FIELDS)
    writer.writeheader()
    output_file.flush()

    for index, norb in enumerate(active_orbitals):
        row = benchmark_size(
            norb=norb,
            charge=charge,
            target_k=target_k,
            threads=num_threads,
            seed=seed + index,
        )
        writer.writerow(row)
        output_file.flush()
        print_result(row)
