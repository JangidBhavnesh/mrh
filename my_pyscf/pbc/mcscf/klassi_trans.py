"""Many-electron translation checks for independently optimized LAS states."""

import numpy as np
from scipy import linalg


def translate_product(ci, nelec, permutation, orbital_phases=None):
    """Translate a fragment-major product, returning (CI, counts, phase).

    permutation[f] is the destination of fragment f. orbital_phases[f]
    specifies T|f,p> = phase[f]|permutation[f],p> for every local orbital p.
    The returned scalar includes orbital phases and the fermionic sign needed
    to restore fragment-major order. Input vectors are neither changed nor
    reused as writable output arrays.
    """
    permutation = np.asarray(permutation)
    nfrags = len(ci)
    if (permutation.shape != (nfrags,)
            or not np.issubdtype(permutation.dtype, np.integer)
            or not np.array_equal(np.sort(permutation), np.arange(nfrags))):
        raise ValueError("permutation must contain each fragment index exactly once")
    nelec = np.asarray(nelec)
    if (nelec.shape != (nfrags, 2)
            or not np.issubdtype(nelec.dtype, np.integer) or np.any(nelec < 0)):
        raise ValueError("nelec must contain nonnegative integer alpha/beta counts")
    if orbital_phases is None:
        orbital_phases = np.ones(nfrags, dtype=complex)
    orbital_phases = np.asarray(orbital_phases)
    if (orbital_phases.shape != (nfrags,)
            or not np.all(np.isfinite(orbital_phases))
            or not np.allclose(abs(orbital_phases), 1., atol=1e-10, rtol=0)):
        raise ValueError("orbital_phases must contain one unit-modulus phase per fragment")
    counts = nelec.sum(axis=1)
    # Swapping two occupied fragment blocks costs (-1) ** (N_f * N_g).
    parity = sum(counts[f] * counts[g]
                 for f in range(nfrags) for g in range(f + 1, nfrags)
                 if permutation[f] > permutation[g])
    phase = (-1) ** int(parity) * np.prod(orbital_phases ** counts)
    translated_ci = [None] * nfrags
    translated_nelec = np.empty_like(nelec)
    for source, target in enumerate(permutation):
        translated_ci[target] = np.array(ci[source], copy=True)
        translated_nelec[target] = nelec[source]
    return translated_ci, translated_nelec, complex(phase)


def _product_overlap(bra, ket, nelec_bra, nelec_ket):
    if not np.array_equal(nelec_bra, nelec_ket):
        return 0j
    return np.prod([np.vdot(b, k) for b, k in zip(bra, ket)])


def project_translation(ci, nelec_frs, permutation, orbital_phases=None):
    """Project physical translation into a possibly nonorthogonal LAS basis.

    Returns a dict containing matrix T = S^{-1} B, overlap S, and the relative
    norm of each translated ket outside the model space (closure_errors).
    B_ij = <Phi_i|translation|Phi_j>. No phases are fitted to the Hamiltonian,
    and references that are not equivalent under translation are not replaced.
    """
    nelec_frs = np.asarray(nelec_frs)
    if (nelec_frs.ndim != 3 or nelec_frs.shape[0] != len(ci)
            or nelec_frs.shape[2] != 2 or not nelec_frs.shape[1]):
        raise ValueError("nelec_frs must have shape (nfrags, nstates, 2)")
    nstates = nelec_frs.shape[1]
    if any(len(row) != nstates for row in ci):
        raise ValueError("ci must contain one vector per fragment and state")
    products = [[row[state] for row in ci] for state in range(nstates)]
    translated = [translate_product(p, nelec_frs[:, state], permutation, orbital_phases)
                  for state, p in enumerate(products)]
    overlap = np.empty((nstates, nstates), dtype=complex)
    cross = np.empty_like(overlap)
    for bra in range(nstates):
        for ket in range(nstates):
            overlap[bra, ket] = _product_overlap(
                products[bra], products[ket], nelec_frs[:, bra], nelec_frs[:, ket])
            ci_t, nelec_t, phase = translated[ket]
            cross[bra, ket] = phase * _product_overlap(
                products[bra], ci_t, nelec_frs[:, bra], nelec_t)
    if not np.all(np.isfinite(overlap)) or not np.all(np.isfinite(cross)):
        raise ValueError("LAS translation overlaps must be finite")
    if np.min(linalg.eigvalsh(overlap)) <= 1e-10:
        raise ValueError("LAS overlap must be positive definite for translation projection")
    matrix = linalg.solve(overlap, cross, assume_a='pos')
    projected_norm = np.einsum('ij,ij->j', cross.conj(), matrix).real
    residuals = np.sqrt(np.maximum(0., 1. - projected_norm / overlap.diagonal().real))
    return {'matrix': matrix, 'overlap': overlap, 'closure_errors': residuals}
