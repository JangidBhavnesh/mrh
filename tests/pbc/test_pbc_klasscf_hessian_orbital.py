import unittest

import numpy as np

from mrh.my_pyscf.pbc.mcscf.klasscf import KLASSCF_HessianOperator


class _OrbitalUGG:

    pairs = ((1, 0), (2, 0), (2, 1))
    nvar_orb = len(pairs)

    def unpack_orb(self, vector):
        vector = np.asarray(vector)
        kappa = np.zeros((1, 3, 3), dtype=np.result_type(vector, complex))
        for value, (p, q) in zip(vector, self.pairs):
            kappa[0, p, q] = value
            kappa[0, q, p] = -value.conjugate()
        return kappa

    def pack_orb(self, kappa):
        return np.asarray([kappa[0, p, q] for p, q in self.pairs])


class KnownValues(unittest.TestCase):

    def test_orbital_hdiag_matches_unit_matvecs_and_is_cached(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.ugg = _OrbitalUGG()
        operator._Horb_diag_matvec_cache = None
        hessian = np.array([
            [3.0, 0.2 - 0.1j, -0.4j],
            [0.7 + 0.3j, -1.5, 0.6],
            [0.1, -0.2j, 2.25],
        ], dtype=np.complex128)
        calls = []

        def orbital_response(kappa):
            direction = operator.ugg.pack_orb(kappa)
            calls.append(np.array(direction, copy=True))
            return 2.0 * operator.ugg.unpack_orb(hessian @ direction)

        operator._orbital_hessian_response = orbital_response

        diagonal = operator._get_Horb_diag_matvec()

        np.testing.assert_allclose(diagonal, np.diag(hessian))
        np.testing.assert_allclose(calls, np.eye(3))
        self.assertEqual(len(calls), operator.ugg.nvar_orb)

        diagonal[0] = 999.0
        cached = operator._get_Horb_diag_matvec()
        np.testing.assert_allclose(cached, np.diag(hessian))
        self.assertEqual(len(calls), operator.ugg.nvar_orb)

    def test_orbital_hdiag_handles_an_empty_orbital_space(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.ugg = type("EmptyUGG", (), {"nvar_orb": 0})()
        operator._Horb_diag_matvec_cache = None
        operator._orbital_hessian_response = lambda kappa: self.fail(
            "orbital response should not be evaluated"
        )

        diagonal = operator._get_Horb_diag_matvec()

        self.assertEqual(diagonal.shape, (0,))
        self.assertEqual(diagonal.dtype, np.dtype(np.complex128))
        np.testing.assert_array_equal(
            operator._Horb_diag_matvec_cache, diagonal,
        )

    def test_orbital_hdiag_rejects_response_shape_mismatch(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.ugg = _OrbitalUGG()
        operator._Horb_diag_matvec_cache = None
        operator._orbital_hessian_response = lambda kappa: np.zeros((2, 2))

        with self.assertRaisesRegex(
                ValueError, r"orbital_hessian_response\[0\] has shape"):
            operator._get_Horb_diag_matvec()


if __name__ == "__main__":
    unittest.main()
