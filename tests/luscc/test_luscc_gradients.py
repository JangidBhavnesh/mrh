import numpy as np
from pyscf import fci

from mrh.exploratory.luscc import get_grad_exact_lassi
from mrh.exploratory.luscc.excitations import apply_operator_string_fci
from mrh.exploratory.unitary_cc import lasuccsd
from mrh.my_pyscf.lassi import op_o0


def _excitation_ops(creation, annihilation, norb):
    """Translate spin-orbital indices into their action order on a CI ket."""
    annihilators = [
        ("ann", int(index % norb), int(index // norb))
        for index in annihilation
    ]
    creators = [
        ("cre", int(index % norb), int(index // norb))
        for index in creation[::-1]
    ]
    return tuple(annihilators + creators)


def test_lassis_gradients_equal_full_ci_commutators(h4_lassis):
    """Check the RDM gradient against an explicit full-CAS commutator."""
    gradients, _, _, _ = get_grad_exact_lassi(h4_lassis)
    uop = lasuccsd.gen_uccsd_op(
        h4_lassis.ncas, h4_lassis.ncas_sub)
    a_idxs, i_idxs = uop.a_idxs, uop.i_idxs

    # Reconstruct the mLAS eigenvector in the full-CAS determinant basis. This
    # deliberately bypasses the spin-resolved 1-, 2-, and 3-RDM contractions
    # used by get_grad_exact_lassi.
    ci_products, electron_numbers = op_o0.ci_outer_product(
        h4_lassis.ci,
        h4_lassis.ncas_sub,
        h4_lassis.get_nelec_frs(),
    )
    assert len({tuple(nelec) for nelec in electron_numbers}) == 1
    assert np.count_nonzero(np.abs(h4_lassis.si[:, 0]) > 1e-10) > 1
    psi = sum(coefficient * ci for coefficient, ci in
              zip(h4_lassis.si[:, 0], ci_products))
    nelec = tuple(electron_numbers[0])

    _, h1, h2 = h4_lassis.ham_2q()
    fci_solver = fci.solver(h4_lassis.mol)
    h2eff = fci_solver.absorb_h1e(
        h1, h2, h4_lassis.ncas, nelec, fac=0.5)
    h_psi = fci_solver.contract_2e(
        h2eff, psi, h4_lassis.ncas, nelec)

    commutator_gradients = []
    for creation, annihilation in zip(a_idxs, i_idxs):
        t_psi, _ = apply_operator_string_fci(
            psi, h4_lassis.ncas, nelec,
            _excitation_ops(creation, annihilation, h4_lassis.ncas),
        )
        t_dagger_psi, _ = apply_operator_string_fci(
            psi, h4_lassis.ncas, nelec,
            _excitation_ops(annihilation, creation, h4_lassis.ncas),
        )
        if t_psi is None:
            t_psi = np.zeros_like(psi)
        if t_dagger_psi is None:
            t_dagger_psi = np.zeros_like(psi)
        generator_psi = t_psi - t_dagger_psi
        commutator_gradients.append(
            2 * np.vdot(generator_psi, h_psi).real)

    np.testing.assert_allclose(
        gradients, commutator_gradients, atol=1e-10, rtol=1e-9)
