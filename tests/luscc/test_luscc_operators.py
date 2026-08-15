import numpy as np
import pytest

from mrh.exploratory.luscc.excitations import apply_operator_string_fci


def test_operator_application_updates_particle_number():
    ci = np.array([[1.0], [0.0]])
    result, nelec = apply_operator_string_fci(
        ci, norb=2, neleca_nelecb=(1, 0),
        ops=(("ann", 0, "alpha"), ("cre", 1, "alpha")),
    )
    assert nelec == (1, 0)
    assert np.isclose(np.linalg.norm(result), 1.0)


def test_operator_application_has_correct_determinant_and_phase():
    # Address zero is |110>; a^dagger_2 a_0 |110> = -|011>.
    ci = np.array([[1.0], [0.0], [0.0]])
    result, nelec = apply_operator_string_fci(
        ci, norb=3, neleca_nelecb=(2, 0),
        ops=(("ann", 0, "alpha"), ("cre", 2, "alpha")),
    )
    expected = np.array([[0.0], [0.0], [-1.0]])
    assert nelec == (2, 0)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "ci, nelec, op, expected_nelec",
    [
        # Pauli-forbidden operations on a partly occupied spin sector.
        (np.array([[1.0], [0.0]]), (1, 0), ("ann", 1, "alpha"), (0, 0)),
        (np.array([[1.0], [0.0]]), (1, 0), ("cre", 0, "alpha"), (2, 0)),
        # Fast paths for empty and full spin sectors.
        (np.array([[1.0]]), (0, 0), ("ann", 0, "alpha"), (0, 0)),
        (np.array([[1.0]]), (2, 0), ("cre", 0, "alpha"), (2, 0)),
    ],
)
def test_forbidden_operator_application_returns_zero(
        ci, nelec, op, expected_nelec):
    result, result_nelec = apply_operator_string_fci(
        ci, norb=2, neleca_nelecb=nelec, ops=(op,),
    )
    assert result is None
    assert result_nelec == expected_nelec
