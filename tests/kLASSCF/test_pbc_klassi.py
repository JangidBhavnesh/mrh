"""Complex charged LAS matrix elements and independent-fragment references."""

import unittest
import io
import copy
from unittest import mock

import numpy as np
from pyscf.pbc import gto, scf

from mrh.my_pyscf.lassi.op_o0 import _ci_outer_product
from mrh.my_pyscf.pbc.fci import direct_spin1_cplx, spin_op
from mrh.my_pyscf.pbc.mcscf.klassi import build_hamiltonian
from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import klassi


def explicit_products(ci, norb, nelec):
    vectors = []
    for state in range(nelec.shape[1]):
        counts = nelec[:, state]
        gen = _ci_outer_product([row[state] for row in ci], norb, counts)[0]
        # Convert fragment-major creation order to all-alpha/all-beta order.
        parity = sum(counts[f, 1] * counts[f+1:, 0].sum() for f in range(len(norb)))
        vectors.append((-1) ** parity * next(gen()).copy())
    return vectors


class MatrixTests(unittest.TestCase):
    def test_complex_projection(self):
        rng = np.random.default_rng(51)
        norb = [2, 2]
        h1 = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        h1 = h1 + h1.conj().T
        factors = rng.normal(size=(3, 4, 4)) + 1j * rng.normal(size=(3, 4, 4))
        factors += factors.conj().transpose(0, 2, 1)
        h2 = np.einsum('xpq,xrs->pqrs', factors, factors)
        for charged in ((1, 0), (2, 1)):
            nelec = np.array([[charged, (1, 1)], [(1, 1), charged]])
            ci = []
            for frag in range(2):
                row = []
                for state in range(2):
                    shape = [2 if n == 1 else 1 for n in nelec[frag, state]]
                    c = rng.normal(size=shape) + 1j * rng.normal(size=shape)
                    row.append(c / np.linalg.norm(c))
                ci.append(row)
            ham, overlap = build_hamiltonian(h1, h2, ci, norb, nelec, ecore=1.23)
            vecs = explicit_products(ci, norb, nelec)
            ne = tuple(nelec[:, 0].sum(axis=0))
            eri = direct_spin1_cplx.absorb_h1e(h1, h2, 4, ne, .5)
            hvecs = [direct_spin1_cplx.contract_2e(eri, v, 4, ne) for v in vecs]
            expected_s = np.array([[np.vdot(b, k) for k in vecs] for b in vecs])
            expected_h = np.array([[np.vdot(b, k) for k in hvecs] for b in vecs])
            np.testing.assert_allclose(overlap, expected_s, atol=1e-12)
            np.testing.assert_allclose(ham, expected_h + 1.23 * expected_s, atol=1e-11)
            np.testing.assert_allclose(ham, ham.conj().T, atol=1e-11)
            self.assertGreater(abs(ham[0, 1].imag), 1e-3)

    def test_one_electron_hopping(self):
        h1 = np.array([[1., .3j], [-.3j, 2.]])
        nelec = np.array([[(1, 0), (0, 0)], [(0, 0), (1, 0)]])
        ci = [[np.ones((1, 1)) for _ in range(2)] for _ in range(2)]
        ham, overlap = build_hamiltonian(h1, np.zeros((2,) * 4), ci, [1, 1], nelec)
        np.testing.assert_allclose(ham, h1)
        np.testing.assert_allclose(overlap, np.eye(2))

    def test_nonorthogonal_core_shift(self):
        ci = [[np.array([[1.], [0.]]), np.array([[.6j], [.8]])]]
        nelec = np.array([[(1, 0), (1, 0)]])
        ham, overlap = build_hamiltonian(np.zeros((2, 2)), np.zeros((2,) * 4),
                                        ci, [2], nelec, ecore=2.)
        self.assertAlmostEqual(overlap[0, 1], .6j)
        np.testing.assert_allclose(ham, 2 * overlap)

    def test_generalized_diagonalization(self):
        overlap = np.array([[1., .2j], [-.2j, 1.]])
        ham = np.array([[1., .3j], [-.3j, 2.]])
        energies, si = klassi.diagonalize(ham, overlap)
        np.testing.assert_allclose(ham @ si, (overlap @ si) * energies, atol=1e-12)
        np.testing.assert_allclose(si.conj().T @ overlap @ si, np.eye(2), atol=1e-12)
        with self.assertRaisesRegex(ValueError, 'positive definite'):
            klassi.diagonalize(ham, np.ones((2, 2)))
        with self.assertRaisesRegex(ValueError, 'Hermitian'):
            klassi.diagonalize(np.array([[1., 1.], [0., 2.]]), np.eye(2))


class ChargedReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cell = gto.Cell()
        cell.a = np.diag([3., 10., 10.])
        cell.atom = 'H 0 0 0; H .74 0 0'
        cell.basis = 'sto-3g'
        cell.precision = 1e-8
        cell.ke_cutoff = 20
        cell.verbose = 0
        cell.build()
        kmf = scf.KRHF(cell, cell.make_kpts((3, 1, 1), wrap_around=True)).density_fit()
        kmf.exxdiv = None
        kmf.max_cycle = 0
        kmf.kernel()
        cls.las = mcscf.KLASCI(kmf, 2, (1, 1), kmesh=(3, 1, 1))
        cls.las.mo_coeff = cls.las.localize_init_guess(
            ['H 1s'], mo_coeff=np.asarray(kmf.mo_coeff))

    def test_independent_states_and_diagonal_energies(self):
        before = self.las.mo_coeff.copy()
        h0, h1, h2 = klassi.ham_2q(self.las)
        for charge in (1, -1):
            with mock.patch.object(klassi, 'kLASCI', wraps=mcscf.KLASCI) as factory:
                states = klassi.make_charged_states(self.las, charge)
            self.assertEqual([c.kwargs['ref_cell'] for c in factory.call_args_list], [0, 1, 2])
            nelec = np.stack([s.get_nelec_frs()[:, 0] for s in states], axis=1)
            ci = [[s.ci[f][0] for s in states] for f in range(3)]
            ham, overlap = build_hamiltonian(h1, h2, ci, self.las.ncas_sub, nelec, h0)
            np.testing.assert_allclose(np.diag(ham), [3 * s.e_tot for s in states], atol=1e-9)
            np.testing.assert_allclose(overlap, np.eye(3), atol=1e-10)
            np.testing.assert_allclose(ham, ham.conj().T, atol=1e-10)
            self.assertGreater(np.max(np.abs(ham - np.diag(np.diag(ham)))), 1e-4)
            for ref, state in enumerate(states):
                self.assertEqual(nelec[:, ref].sum(), 6 - charge)
                self.assertEqual(nelec[ref, ref].sum(), 2 - charge)
                np.testing.assert_array_equal(state.mo_coeff, before)
                self.assertIsNot(state.fciboxes[0], self.las.fciboxes[0])
            self.assertFalse(np.shares_memory(states[0].ci[0][0], states[1].ci[0][0]))
            # A three-fragment explicit projection checks spectator parity signs.
            vecs = explicit_products(ci, self.las.ncas_sub, nelec)
            ne = tuple(nelec[:, 0].sum(axis=0))
            eri = direct_spin1_cplx.absorb_h1e(h1, h2, 6, ne, .5)
            hvecs = [direct_spin1_cplx.contract_2e(eri, v, 6, ne) for v in vecs]
            expected = np.array([[np.vdot(b, k) for k in hvecs] for b in vecs])
            np.testing.assert_allclose(ham, expected + h0 * overlap, atol=1e-9)
        np.testing.assert_array_equal(self.las.mo_coeff, before)
        self.assertIsNone(self.las.ci)

    def test_reference_validation_and_nonconvergence(self):
        for charge in (0, 2, .5):
            with self.assertRaisesRegex(ValueError, 'charge'):
                klassi.make_charged_states(self.las, charge)
        with self.assertRaisesRegex(ValueError, 'mo_coeff'):
            klassi.make_charged_states(self.las, 1, mo_coeff=self.las.mo_coeff[0])
        with mock.patch.object(self.las, 'max_cycle_macro', 1):
            with self.assertRaisesRegex(RuntimeError, 'did not converge'):
                klassi.make_charged_states(self.las, 1)

    def test_public_driver(self):
        for charge in (1, -1):
            lsi = mcscf.KLASSI(self.las, charge=charge)
            with self.assertRaisesRegex(ValueError, 'run kLASSI'):
                lsi.kernel_kpts()
            energies, si = lsi.kernel()
            self.assertTrue(lsi.converged)
            self.assertEqual(len(lsi.charged_states), 3)
            np.testing.assert_array_equal(lsi.ref_cells, [0, 1, 2])
            np.testing.assert_allclose(lsi.ham @ si,
                                       (lsi.ovlp @ si) * (energies * 3), atol=1e-10)
            np.testing.assert_allclose(si.conj().T @ lsi.ovlp @ si, np.eye(3), atol=1e-10)
            self.assertLessEqual(energies[0], min(s.e_tot.real for s in lsi.charged_states))
            self.assertIsNone(self.las.ci)
            np.testing.assert_allclose(lsi.s2, .75, atol=1e-10)
            np.testing.assert_allclose(lsi.spin_multiplicity, 2., atol=1e-10)
            with mock.patch.object(lsi, 'stdout', io.StringIO()) as output, \
                    mock.patch.object(lsi, 'verbose', 4):
                lsi._finalize()
                table = output.getvalue()
            self.assertIn('Energies (Ha/cell)', table)
            self.assertIn('Total Active Space (e,o)', table)
            self.assertIn('Spin (2S+1)', table)
            self.assertIn(f'({6 - charge}, 6)', table)
            self.assertIn(f'{energies[0]:.12f}', table)
            before = lsi.si.copy()
            checks = lsi.check_translation_symmetry()
            translation = lsi.get_translation_matrix((1, 0, 0))
            np.testing.assert_allclose(translation, checks[(1, 0, 0)]['matrix'], atol=1e-12)
            np.testing.assert_allclose(np.linalg.matrix_power(translation, 3), np.eye(3), atol=1e-8)
            np.testing.assert_allclose(translation.conj().T @ lsi.ham @ translation,
                                       lsi.ham, atol=1e-8)
            np.testing.assert_array_equal(lsi.si, before)
            with mock.patch.object(lsi, 'stdout', io.StringIO()) as output, \
                    mock.patch.object(lsi, 'verbose', 4):
                results = lsi.kernel_kpts()
                table = output.getvalue()
            self.assertIs(results, lsi.k_results)
            self.assertIn('total many-electron momentum', table)
            for heading in ('K (scaled)', 'Total Active Space (e,o)',
                            'Energies (Ha/cell)', 'Spin (2S+1)'):
                self.assertIn(heading, table)
            np.testing.assert_allclose(np.sort(np.concatenate([r['e_tot'] for r in results])),
                                       lsi.e_tot, atol=1e-10)
            for index, result in enumerate(results):
                self.assertEqual(result['target_k'], index)
                self.assertEqual(result['charge'], charge)
                self.assertEqual(result['nkpts'], 3)
                self.assertEqual(len(result['e_tot']), 1)
                np.testing.assert_allclose(result['kpt'], self.las.kpts[index])
                np.testing.assert_allclose(result['spin_multiplicity'], 2., atol=1e-10)
                np.testing.assert_allclose(lsi.ham @ result['si'],
                                           (lsi.ovlp @ result['si']) * result['e_tot_supercell'],
                                           atol=1e-8)
                phase = np.exp(-2j * np.pi * result['scaled_k'][0])
                np.testing.assert_allclose(translation @ result['si'],
                                           phase * result['si'], atol=1e-8)
            np.testing.assert_array_equal(lsi.si, before)
            np.testing.assert_array_equal(lsi.e_tot, energies)
            # Independently chosen phases must produce the covariant T matrix.
            gauged = copy.copy(lsi)
            phases = np.exp(1j * np.array([.4, -.2, .8]))
            gauged.ci = copy.deepcopy(lsi.ci)
            gauged.ci[0] = [c * phase for c, phase in zip(gauged.ci[0], phases)]
            gauged.ham = phases.conj()[:, None] * lsi.ham * phases[None, :]
            np.testing.assert_allclose(gauged.get_translation_matrix((1, 0, 0)),
                                       phases.conj()[:, None] * translation * phases[None, :],
                                       atol=1e-8)
            for original, changed in zip(results, gauged.kernel_kpts()):
                np.testing.assert_allclose(original['e_tot'], changed['e_tot'], atol=1e-10)
            # A changed spectator is not repaired or silently symmetrized.
            broken = copy.copy(lsi)
            broken.ci = copy.deepcopy(lsi.ci)
            c = broken.ci[1][0].ravel()
            other = np.zeros_like(c)
            other[np.argmin(abs(c))] = 1
            other -= c * np.vdot(c, other)
            broken.ci[1][0] = (other / np.linalg.norm(other)).reshape(lsi.ci[1][0].shape)
            with self.assertRaisesRegex(ValueError, 'not closed under translation'):
                broken.kernel_kpts()
            self.assertIsNone(broken.k_results)
            broken = copy.copy(lsi)
            broken.ham = lsi.ham.copy()
            broken.ham[0, 0] += .1
            with self.assertRaisesRegex(ValueError, 'hamiltonian_error'):
                broken.kernel_kpts()
            with self.assertRaisesRegex(ValueError, 'integer lattice indices'):
                lsi.get_translation_matrix((.5, 0, 0))
            scaled = self.las.cell.get_scaled_kpts(self.las.kpts) + .1
            with mock.patch.object(self.las.cell, 'get_scaled_kpts', return_value=scaled):
                with self.assertRaisesRegex(NotImplementedError, 'Gamma-centered'):
                    lsi.get_translation_matrix((1, 0, 0))
        with self.assertRaisesRegex(ValueError, 'charge'):
            mcscf.KLASSI(self.las, charge=0)
        with self.assertRaisesRegex(TypeError, 'kLASCI or kLASSCF'):
            mcscf.KLASSI(self.las._scf)

    def test_one_cell_with_core(self):
        kmf = scf.KRHF(self.las.cell, self.las.cell.make_kpts((1, 1, 1))).density_fit()
        kmf.exxdiv = None
        kmf.max_cycle = 0
        kmf.kernel()
        source = mcscf.KLASCI(kmf, 1, (0, 0), kmesh=(1, 1, 1))
        source.mo_coeff = np.asarray(kmf.mo_coeff)
        lsi = mcscf.KLASSI(source, charge=-1)
        energies, si = lsi.kernel()
        self.assertEqual(lsi.charged_states[0].ncore, 1)
        np.testing.assert_allclose(energies, [lsi.charged_states[0].e_tot], atol=1e-10)
        np.testing.assert_allclose(abs(si), [[1.]])
        for result in lsi.check_translation_symmetry().values():
            np.testing.assert_allclose(result['matrix'], [[1.]], atol=1e-10)
        results = lsi.kernel_kpts()
        np.testing.assert_allclose(results[0]['e_tot'], energies, atol=1e-10)
        # A fresh kernel must discard results made in the previous LAS basis.
        lsi.kernel()
        self.assertIsNone(lsi.k_results)

    def test_total_spin_with_magnetic_fragments(self):
        source = mcscf.KLASCI(self.las._scf, 2, (1, 1), spin_mult=3, kmesh=(3, 1, 1))
        source.mo_coeff = self.las.mo_coeff.copy()
        lsi = mcscf.KLASSI(source, charge=1)
        lsi.kernel()
        vectors = explicit_products(lsi.ci, lsi.ncas_sub, lsi.nelec_frs)
        nelec = tuple(lsi.nelec_frs[:, 0].sum(axis=0))
        for root in range(len(lsi.e_tot)):
            full_ci = sum(c * v for c, v in zip(lsi.si[:, root], vectors))
            s2, multiplicity = spin_op.spin_square0(full_ci, 6, nelec)
            self.assertAlmostEqual(lsi.s2[root], s2, places=10)
            self.assertAlmostEqual(lsi.spin_multiplicity[root], multiplicity, places=10)
        # The full product's spin is not just the charged fragment's doublet.
        self.assertTrue(np.all(lsi.spin_multiplicity > 2))


if __name__ == '__main__':
    unittest.main()
