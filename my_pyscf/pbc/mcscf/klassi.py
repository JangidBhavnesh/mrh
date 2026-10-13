"""Fixed-orbital periodic LAS state interaction without translation adaptation.

Product states use fragment-major fermion ordering: alpha then beta orbitals
within each fragment. Integrals use the spatial-orbital chemists' convention.
Matrix elements are contracted from local transition tensors, without expanding
the product states into the full-supercell determinant space.
"""

from itertools import product

import numpy as np
from pyscf.fci import addons, cistring


def _apply_local(ci, norb, nelec, spin, create, orbital):
    """Apply a local fermion operator, retaining complex CI coefficients."""
    operation = ((addons.des_a, addons.des_b),
                 (addons.cre_a, addons.cre_b))[create][spin]
    return (operation(ci.real, norb, nelec, orbital)
            + 1j * operation(ci.imag, norb, nelec, orbital))


class _ProductTransition:
    """Cache fragment transition tensors for one bra/ket product pair."""

    def __init__(self, bra, ket, norb, nelec_bra, nelec_ket):
        self.bra, self.ket = bra, ket
        self.norb = norb
        self.nelec_bra = np.asarray(nelec_bra)
        self.nelec_ket = np.asarray(nelec_ket)
        self.cache = {}

    def local_tensor(self, frag, operators):
        key = (frag, tuple(operators))
        if key in self.cache:
            return self.cache[key]
        norb = self.norb[frag]
        tensor = np.zeros((norb,) * len(operators), dtype=complex)
        for orbitals in np.ndindex(tensor.shape):
            ci = self.ket[frag]
            nelec = self.nelec_ket[frag].copy()
            for (spin, create), orbital in reversed(list(zip(operators, orbitals))):
                if (create and nelec[spin] == norb) or (not create and nelec[spin] == 0):
                    ci = None
                    break
                ci = _apply_local(ci, norb, tuple(nelec), spin, create, orbital)
                nelec[spin] += 1 if create else -1
            if ci is not None:
                tensor[orbitals] = np.vdot(self.bra[frag], ci)
        self.cache[key] = tensor
        return tensor

    def contract(self, integrals, operators):
        """Contract operators (fragment, spin, creation, integral-axis label)."""
        nelec = self.nelec_ket.copy()
        sign = 1
        # Jordan-Wigner parity follows the occupations at each operator action.
        for frag, spin, create, _ in reversed(operators):
            if (create and nelec[frag, spin] == self.norb[frag]
                    or not create and nelec[frag, spin] == 0):
                return 0j
            if nelec[:frag].sum() % 2:
                sign = -sign
            nelec[frag, spin] += 1 if create else -1
        if not np.array_equal(nelec, self.nelec_bra):
            return 0j
        args = [integrals, list(range(integrals.ndim))]
        for frag in range(len(self.norb)):
            local = [(spin, create) for f, spin, create, _ in operators if f == frag]
            labels = [label for f, _, _, label in operators if f == frag]
            args.extend((self.local_tensor(frag, local), labels))
        return sign * np.einsum(*args, [], optimize=True)


def ham_2q(las, mo_coeff=None):
    """Return (ecore, h1, h2) in the full-supercell Wannier active space."""
    if mo_coeff is None:
        mo_coeff = las.mo_coeff
    h1, ecore = las.h1e_for_cas(mo_coeff=mo_coeff)
    return ecore, h1, las.get_h2cas(mo_coeff)


def build_hamiltonian(h1, h2, ci, ncas_sub, nelec_frs, ecore=0):
    """Build complex H and S from single-root LAS product states.

    ci[fragment][state] contains a determinant-basis fragment CI matrix;
    nelec_frs has shape (nfrags, nstates, 2). All states must have the same
    total alpha and beta counts and use the same orthonormal orbital set.
    H includes ecore*S and is in full-supercell energy units.

    This general fragment-contraction implementation prioritizes correctness;
    it visits all one- and two-electron fragment blocks and screens forbidden
    electron transfers before constructing local transition tensors.
    """
    ncas_sub = np.asarray(ncas_sub, dtype=int)
    nelec_frs = np.asarray(nelec_frs)
    if (nelec_frs.ndim != 3 or nelec_frs.shape[0] != len(ncas_sub)
            or nelec_frs.shape[2] != 2 or nelec_frs.shape[1] == 0
            or not np.issubdtype(nelec_frs.dtype, np.integer)):
        raise ValueError("nelec_frs must have shape (nfrags, nstates, 2) with integer counts")
    nfrags, nstates = nelec_frs.shape[:2]
    if (np.any(ncas_sub <= 0) or np.any(nelec_frs < 0)
            or np.any(nelec_frs > ncas_sub[:, None, None])):
        raise ValueError("invalid fragment orbital or electron counts")
    totals = nelec_frs.sum(axis=0)
    if np.any(totals != totals[0]):
        raise ValueError("all LAS states must have the same total alpha and beta counts")
    norb = int(ncas_sub.sum())
    h1, h2 = np.asarray(h1), np.asarray(h2)
    if h1.shape != (norb, norb) or h2.shape != (norb,) * 4:
        raise ValueError("integral dimensions do not match the full active space")
    if len(ci) != nfrags or any(len(row) != nstates for row in ci):
        raise ValueError("ci must contain one vector per fragment and state")
    ci = [[np.asarray(c, dtype=complex) for c in row] for row in ci]
    for frag, row in enumerate(ci):
        for state, c in enumerate(row):
            shape = tuple(cistring.num_strings(ncas_sub[frag], n)
                          for n in nelec_frs[frag, state])
            if c.shape != shape or not np.all(np.isfinite(c)):
                raise ValueError("fragment CI shape or values are invalid")
    offsets = np.concatenate(([0], np.cumsum(ncas_sub)))
    slices = [slice(i, j) for i, j in zip(offsets[:-1], offsets[1:])]
    ham = np.zeros((nstates, nstates), dtype=complex)
    ovlp = np.zeros_like(ham)
    for bra, ket in product(range(nstates), repeat=2):
        transition = _ProductTransition(
            [row[bra] for row in ci], [row[ket] for row in ci], ncas_sub,
            nelec_frs[:, bra], nelec_frs[:, ket])
        ovlp[bra, ket] = transition.contract(np.asarray(1.), [])
        for p, q in product(range(nfrags), repeat=2):
            block = h1[slices[p], slices[q]]
            for spin in range(2):
                ham[bra, ket] += transition.contract(
                    block, [(p, spin, True, 0), (q, spin, False, 1)])
        for p, q, r, s in product(range(nfrags), repeat=4):
            block = h2[slices[p], slices[q], slices[r], slices[s]]
            for sigma, tau in product(range(2), repeat=2):
                ham[bra, ket] += .5 * transition.contract(block, [
                    (p, sigma, True, 0), (r, tau, True, 2),
                    (s, tau, False, 3), (q, sigma, False, 1)])
    return ham + ecore * ovlp, ovlp
