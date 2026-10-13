"""Many-electron translations and momentum sectors of independent LAS states."""

import numpy as np
from scipy import linalg

from mrh.my_pyscf.pbc.util.transym import TranslationSymm
from mrh.my_pyscf.pbc.util.wannier import get_wannier_orbs


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


def orbital_translation(lsi, translation, tol=1e-8):
    """Return the cell permutation and T|f,p> orbital phases on a Gamma mesh.

    General local orbital mixing and shifted/twisted meshes are not supported
    in this first implementation. Doubly occupied frozen-core bands contribute
    unit total translation phase on the supported unshifted complete meshes.
    """
    translation = np.asarray(translation)
    if translation.shape != (3,) or not np.issubdtype(translation.dtype, np.integer):
        raise ValueError("translation must contain three integer lattice indices")
    las = lsi._las
    mesh = np.asarray(las.kmesh)
    ts = TranslationSymm(las.cell, mesh, kpts=las.kpts)
    grid = las.cell.get_scaled_kpts(las.kpts) * mesh
    if (not np.allclose(grid, np.rint(grid), atol=tol, rtol=0)
            or len(set(map(tuple, np.mod(np.rint(grid).astype(int), mesh)))) != ts.ncell):
        raise NotImplementedError("LAS translations require a complete unshifted Gamma-centered mesh")
    permutation = np.array([ts.R_to_i[ts.mod_index(r + translation)] for r in ts.R_indices])
    source_rows = np.argsort(permutation)
    active = lsi.mo_coeff[:, :, las.ncore:las.ncore + las.ncas]
    orbitals = get_wannier_orbs(las._scf, las.kmesh, active)[0]
    phases = np.empty(ts.ncell, dtype=complex)
    for source, target in enumerate(permutation):
        translated = orbitals[source_rows, :, source, :]
        expected = orbitals[:, :, target, :]
        overlap = np.vdot(expected, translated)
        if abs(overlap) < 1e-14:
            raise ValueError(f"zero translated-orbital overlap for cell {source}")
        phases[source] = overlap / abs(overlap)
        error = np.linalg.norm(translated - phases[source] * expected) / np.linalg.norm(expected)
        if error > tol:
            raise ValueError(f"translated orbitals on cell {source} require a nonscalar "
                             f"transformation (relative error {error:.3e})")
    return permutation, phases


def check_translation(lsi, translation, tol=1e-5, energy_tol=1e-7, orbital_tol=1e-8):
    """Validate one translation and return its matrix and residual diagnostics.

    tol bounds relative state leakage and dimensionless matrix errors;
    energy_tol bounds max|T^dagger H T - H| in Hartree per supercell.
    Failures do not alter the independently optimized LAS states or energies.
    """
    if not lsi.converged or lsi.ci is None:
        raise ValueError("run kLASSI before checking translation symmetry")
    if any(not np.isfinite(t) or t <= 0 for t in (tol, energy_tol, orbital_tol)):
        raise ValueError("translation tolerances must be finite and positive")
    permutation, phases = orbital_translation(lsi, translation, tol=orbital_tol)
    result = project_translation(lsi.ci, lsi.nelec_frs, permutation, phases)
    if not np.allclose(result['overlap'], lsi.ovlp, atol=1e-10, rtol=1e-10):
        raise ValueError("stored overlap does not match the current LAS basis")
    leakage = result['closure_errors'].max()
    if leakage > tol:
        ket = int(np.argmax(result['closure_errors']))
        raise ValueError(f"LAS basis is not closed under translation {tuple(translation)}: "
                         f"state {ket} has relative leakage {leakage:.3e} > {tol:.3e}")
    matrix = result['matrix']
    result['overlap_error'] = float(np.max(abs(matrix.conj().T @ lsi.ovlp @ matrix - lsi.ovlp)))
    result['hamiltonian_error'] = float(np.max(abs(matrix.conj().T @ lsi.ham @ matrix - lsi.ham)))
    mesh = np.asarray(lsi._las.kmesh)
    order = int(np.lcm.reduce(mesh // np.gcd(mesh, np.asarray(translation))))
    result['group_error'] = float(np.max(abs(np.linalg.matrix_power(matrix, order)
                                             - np.eye(len(matrix)))))
    for key, threshold in (('overlap_error', tol), ('group_error', tol),
                           ('hamiltonian_error', energy_tol)):
        if result[key] > threshold:
            raise ValueError(f"translation {tuple(translation)} fails {key}: "
                             f"{result[key]:.3e} > {threshold:.3e}")
    return result


def check_translation_symmetry(lsi, tol=1e-5, energy_tol=1e-7, orbital_tol=1e-8):
    """Check all three primitive translations, including their commutation."""
    generators = [tuple(row) for row in np.eye(3, dtype=int)]
    results = {t: check_translation(lsi, t, tol, energy_tol, orbital_tol) for t in generators}
    for i, left in enumerate(generators):
        for right in generators[i+1:]:
            a, b = results[left]['matrix'], results[right]['matrix']
            error = float(np.max(abs(a @ b - b @ a)))
            if error > tol:
                raise ValueError(f"translation generators {left} and {right} do not commute: {error:.3e}")
    return results


def momentum_basis(translations, ovlp, kmesh, scaled_kpts, tol=1e-5):
    """Return S-orthonormal bases for total many-electron momentum sectors.

    translations[a] acts on product-basis coefficients for translation by
    lattice vector a. The convention is T_a|K> = exp(-2*pi*i*K[a])|K>,
    where scaled_kpts are reciprocal-lattice coordinates. Results follow the
    supplied k-point order, including (nstates, 0) arrays for empty sectors.
    Arbitrary product-state phases and positive-definite overlaps are allowed.
    """
    mesh = np.asarray(kmesh)
    if (mesh.shape != (3,) or not np.issubdtype(mesh.dtype, np.integer)
            or np.any(mesh < 1)):
        raise ValueError("kmesh must contain three positive integers")
    if not np.isfinite(tol) or tol <= 0 or tol >= .5:
        raise ValueError("momentum tolerance must be finite and between zero and 0.5")
    scaled = np.asarray(scaled_kpts)
    if scaled.shape != (int(np.prod(mesh)), 3) or not np.all(np.isfinite(scaled)):
        raise ValueError("scaled_kpts must contain one finite momentum per mesh cell")
    grid = scaled * mesh
    labels = np.mod(np.rint(grid).astype(int), mesh)
    if (not np.allclose(grid, np.rint(grid), atol=tol, rtol=0)
            or len(set(map(tuple, labels))) != len(scaled)):
        raise ValueError("momenta must cover a complete unshifted mesh without duplicates")
    ovlp = np.asarray(ovlp)
    if (ovlp.ndim != 2 or ovlp.shape[0] != ovlp.shape[1] or not len(ovlp)
            or not np.all(np.isfinite(ovlp))
            or not np.allclose(ovlp, ovlp.conj().T, atol=1e-10, rtol=0)):
        raise ValueError("overlap must be finite, square, and Hermitian")
    values, vectors = linalg.eigh(ovlp)
    if values[0] <= 1e-10:
        raise ValueError("overlap must be positive definite for momentum projection")
    x = (vectors / np.sqrt(values)) @ vectors.conj().T
    xinv = (vectors * np.sqrt(values)) @ vectors.conj().T
    translations = np.asarray(translations)
    if translations.shape != (3, len(ovlp), len(ovlp)) or not np.all(np.isfinite(translations)):
        raise ValueError("translations must contain three finite square generators")
    unit = np.eye(len(ovlp))
    generators = [xinv @ t @ x for t in translations]
    for a, t in enumerate(generators):
        if (np.max(abs(t.conj().T @ t - unit)) > tol
                or np.max(abs(np.linalg.matrix_power(t, int(mesh[a])) - unit)) > tol):
            raise ValueError("translations must be S-unitary finite-group generators")
        for other in generators[:a]:
            if np.max(abs(t @ other - other @ t)) > tol:
                raise ValueError("translation generators must commute")

    # Character projectors separate K even when energies are degenerate.
    axis_projectors = []
    for t, order in zip(generators, mesh):
        powers = [unit]
        for _ in range(1, order):
            powers.append(powers[-1] @ t)
        axis_projectors.append([
            sum(np.exp(2j * np.pi * m * r / order) * p
                for r, p in enumerate(powers)) / order for m in range(order)])
    bases = []
    for label, k in zip(labels, scaled):
        projector = axis_projectors[0][label[0]] @ axis_projectors[1][label[1]]
        projector = projector @ axis_projectors[2][label[2]]
        if np.max(abs(projector - projector.conj().T)) > tol:
            raise ValueError("momentum projector is not Hermitian")
        weights, vecs = linalg.eigh((projector + projector.conj().T) / 2)
        if np.max(np.minimum(abs(weights), abs(weights - 1))) > tol:
            raise ValueError("momentum projector is not idempotent")
        basis = x @ vecs[:, weights > .5]
        for a, t in enumerate(generators):
            white_basis = xinv @ basis
            residual = t @ white_basis - np.exp(-2j * np.pi * k[a]) * white_basis
            if residual.size and np.max(abs(residual)) > tol:
                raise ValueError("momentum basis fails its translation eigenvalue check")
        bases.append(basis)
    combined = np.concatenate(bases, axis=1)
    if (combined.shape != ovlp.shape
            or np.max(abs(combined.conj().T @ ovlp @ combined - unit)) > tol):
        raise ValueError("momentum sectors do not form a complete S-orthonormal basis")
    return bases
