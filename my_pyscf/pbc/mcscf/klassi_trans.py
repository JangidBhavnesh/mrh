"""Many-electron translation checks for independently optimized LAS states."""

import numpy as np


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
