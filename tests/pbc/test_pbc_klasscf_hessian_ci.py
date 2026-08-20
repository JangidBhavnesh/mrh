import unittest

import numpy as np

from mrh.my_pyscf.pbc.mcscf.klasscf import KLASSCF_HessianOperator


class _RecordingFCIBox:

    def __init__(self, diagonals):
        self.diagonals = diagonals
        self.calls = []

    def states_make_hdiag_csf(self, h1, h2, norb, nelec):
        self.calls.append((h1, h2, norb, nelec))
        return [np.array(diagonal, copy=True) for diagonal in self.diagonals]


class _SelectingTransformer:

    def __init__(self, indices, ncsf=None):
        self.indices = np.asarray(indices)
        self.ncsf = len(self.indices) if ncsf is None else ncsf

    def pack_csf(self, diagonal):
        return np.asarray(diagonal)[self.indices]


def make_diagonal_operator():
    operator = KLASSCF_HessianOperator.__new__(KLASSCF_HessianOperator)
    boxes = [
        _RecordingFCIBox([
            np.array([0.2, 0.4, 0.6]),
            np.array([-0.1, 0.3]),
        ]),
        _RecordingFCIBox([np.array([9.0])]),
        _RecordingFCIBox([
            np.array([1.0 + 1.0j, 2.0 - 0.5j]),
        ]),
    ]
    operator.fciboxes = boxes
    operator.ncas_sub = np.array([1, 2, 1])
    operator.nelecas_sub = np.array([(1, 0), (1, 1), (0, 1)])
    operator.h1frs = [object(), object(), object()]
    operator.ci_transformers = [
        [_SelectingTransformer([2, 0]), _SelectingTransformer([1])],
        [_SelectingTransformer([0])],
        [_SelectingTransformer([1, 0])],
    ]
    operator.frozen_ci = {1}
    operator.eri_cas = (
        np.arange(4 ** 4, dtype=float).reshape((4,) * 4)
        + 0.1j
    )
    return operator, boxes


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

    def test_hci_diag_packs_nonfrozen_fragments_in_layout_order(self):
        operator, boxes = make_diagonal_operator()

        actual = operator._get_Hci_diag()

        self.assertEqual(len(actual), 3)
        np.testing.assert_allclose(actual[0], [0.6, 0.2])
        np.testing.assert_allclose(actual[1], [0.3])
        np.testing.assert_allclose(actual[2], [2.0 - 0.5j, 1.0 + 1.0j])
        self.assertEqual(len(boxes[0].calls), 1)
        self.assertEqual(len(boxes[1].calls), 0)
        self.assertEqual(len(boxes[2].calls), 1)

        h1_0, h2_0, norb_0, nelec_0 = boxes[0].calls[0]
        self.assertIs(h1_0, operator.h1frs[0])
        np.testing.assert_array_equal(
            h2_0, operator.eri_cas[0:1, 0:1, 0:1, 0:1],
        )
        self.assertEqual(norb_0, 1)
        np.testing.assert_array_equal(nelec_0, [1, 0])

        h1_2, h2_2, norb_2, nelec_2 = boxes[2].calls[0]
        self.assertIs(h1_2, operator.h1frs[2])
        np.testing.assert_array_equal(
            h2_2, operator.eri_cas[3:4, 3:4, 3:4, 3:4],
        )
        self.assertEqual(norb_2, 1)
        np.testing.assert_array_equal(nelec_2, [0, 1])

    def test_hci_diag_rejects_inconsistent_root_count(self):
        operator, boxes = make_diagonal_operator()
        boxes[0].diagonals = [np.array([0.2, 0.4, 0.6])]

        with self.assertRaisesRegex(
                ValueError, "cell 0 produced 1 Hamiltonian diagonals for 2"):
            operator._get_Hci_diag()

    def test_hci_diag_rejects_inconsistent_packed_size(self):
        operator, _ = make_diagonal_operator()
        operator.ci_transformers[0][0] = _SelectingTransformer(
            [0], ncsf=2,
        )

        with self.assertRaisesRegex(
                ValueError, "cell 0, root 0 packed Hamiltonian diagonal"):
            operator._get_Hci_diag()


if __name__ == "__main__":
    unittest.main()
