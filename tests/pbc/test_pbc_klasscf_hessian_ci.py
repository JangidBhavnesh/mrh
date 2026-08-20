import unittest

import numpy as np

from mrh.my_pyscf.pbc.mcscf.klasscf import KLASSCF_HessianOperator


class KnownValues(unittest.TestCase):

    def test_ci_response_diag_uses_hermitian_projection(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        c0 = np.array([1.0, 1.0j], dtype=np.complex128) / np.sqrt(2)
        trial = np.array([0.3 + 0.2j, -0.4j], dtype=np.complex128)
        hamiltonian = np.array(
            [[0.7, 0.2 - 0.5j], [0.2 + 0.5j, -0.1]],
            dtype=np.complex128,
        )
        energy = np.vdot(c0, hamiltonian @ c0)
        shifted_hamiltonian = hamiltonian - energy * np.eye(2)
        residual = shifted_hamiltonian @ c0
        operator.ci = [[c0]]
        operator.e0 = [[energy]]
        operator.hci0 = [[residual]]
        operator.h1frs = object()
        operator.eri_cas = object()
        calls = []

        def apply_hamiltonian(h0, h1, h2, ci):
            calls.append((h0, h1, h2, ci))
            return [[shifted_hamiltonian @ ci[0][0]]]

        operator.Hci_all = apply_hamiltonian

        actual = operator.ci_response_diag([[trial]])[0][0]

        projector = np.eye(2) - np.outer(c0, c0.conj())
        expected = (
            2.0 * projector @ shifted_hamiltonian @ projector @ trial
        )
        np.testing.assert_allclose(actual, expected)
        self.assertEqual(len(calls), 1)
        h0, h1, h2, ci = calls[0]
        np.testing.assert_allclose(h0, [[-energy]])
        self.assertIs(h1, operator.h1frs)
        self.assertIs(h2, operator.eri_cas)
        self.assertIs(ci[0][0], trial)


if __name__ == "__main__":
    unittest.main()
