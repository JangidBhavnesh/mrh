import numpy as np
from pyscf import fci
from scipy import linalg

from mrh.exploratory.luscc import LUSCC
from mrh.exploratory.luscc.excitations import apply_operator_string_fci
from mrh.my_pyscf.lassi import op_o0
from mrh.my_pyscf.lassi.lassi import ham_2q


def _apply_alpha_excitation(ci, norb, nelec, creation, annihilation):
    result, result_nelec = apply_operator_string_fci(
        ci,
        norb,
        nelec,
        ops=(("ann", annihilation, "alpha"),
             ("cre", creation, "alpha")),
    )
    assert result_nelec == nelec
    assert result is not None
    return result / np.linalg.norm(result)


def _full_ci_model_space_energy(las, creation, annihilation):
    """Diagonalize LAS, T|LAS>, and T-dagger|LAS> in the full-CAS basis."""
    ci_products, electron_numbers = op_o0.ci_outer_product(
        las.ci, las.ncas_sub, las.get_nelec_frs())
    assert len(ci_products) == 1
    reference = ci_products[0] / np.linalg.norm(ci_products[0])
    nelec = tuple(electron_numbers[0])
    states = [
        reference,
        _apply_alpha_excitation(
            reference, las.ncas, nelec, creation, annihilation),
        _apply_alpha_excitation(
            reference, las.ncas, nelec, annihilation, creation),
    ]

    h0, h1, h2 = ham_2q(las, las.mo_coeff)
    fci_solver = fci.solver(las.mol)
    h2eff = fci_solver.absorb_h1e(
        h1, h2, las.ncas, nelec, fac=0.5)
    h_states = [
        fci_solver.contract_2e(h2eff, state, las.ncas, nelec)
        for state in states
    ]
    overlap = np.asarray([
        [np.vdot(bra, ket) for ket in states]
        for bra in states
    ])
    hamiltonian = np.asarray([
        [np.vdot(bra, h_ket) + h0 * overlap[i, j]
         for j, h_ket in enumerate(h_states)]
        for i, bra in enumerate(states)
    ])
    return linalg.eigh(hamiltonian, overlap, eigvals_only=True)[0]


def test_h4_luscc_energy(h4_las):
    # One fixed alpha charge-transfer generator and its de-excitation produce
    # a small, deterministic end-to-end LUSCC model space.
    energy, _ = LUSCC(
        h4_las,
        a_idxs=[np.array([2])],
        i_idxs=[np.array([0])],
    ).kernel()
    reference_energy = _full_ci_model_space_energy(
        h4_las, creation=2, annihilation=0)
    np.testing.assert_allclose(energy[0], reference_energy, atol=1e-10)
    assert energy[0] <= h4_las.e_tot + 1e-10
