#!/usr/bin/env python

import unittest
from unittest.mock import patch

import numpy as np
from scipy import linalg

from pyscf import lib

from mrh.my_pyscf.pbc.fci import direct_spin1_cplx
from mrh.my_pyscf.pbc.mcscf import klasscf
from mrh.my_pyscf.pbc.mcscf.klasscf import (
    KLASSCF_HessianOperator,
    KLASSCF_TransSymmHessianOperator,
)


class _FakeFCIBox:
    """Small transition-RDM backend for CI-Hessian unit tests."""

    _state_args = staticmethod(lambda value: value)
    _solver_args = staticmethod(lambda value: value)

    def __init__(self):
        self.fcisolvers = [object()]
        self.collect_calls = 0
        self.hdiag_calls = 0
        self.transition_operator = np.array(
            [[0.7, 0.2 - 0.3j], [-0.1 + 0.4j, -0.2]],
            dtype=np.complex128,
        )
        self.dm1a_operator = np.array(
            [[1.0, 0.2j], [-0.3j, 0.4]], dtype=np.complex128,
        )
        self.dm1b_operator = np.array(
            [[-0.2, 0.1 + 0.2j], [0.3j, 0.8]], dtype=np.complex128,
        )
        self.dm2_operator = (
            np.arange(16, dtype=float).reshape((2,) * 4)
            + 0.1j * np.arange(16, 0, -1).reshape((2,) * 4)
        ) / 17.0
        self.dm2_scales = np.array([0.5, -0.2j, 0.3j, 0.7])

    def _get_nelec(self, solver, nelec):
        return tuple(nelec)

    def states_make_hdiag_csf(self, h1, h2, norb, nelec):
        self.hdiag_calls += 1
        return [np.array([1.25, 2.5], dtype=np.complex128)]

    def _collect(
            self, name, ci1, ci0, norb, nelec, link_index=None, **kwargs):
        if name not in (
                "trans_rdm1s", "trans_rdm1s_py",
                "trans_rdm12s", "trans_rdm12s_py"):
            raise AssertionError(f"unexpected contraction {name}")
        self.collect_calls += 1
        bra = np.asarray(ci1[0]).reshape(-1)
        ket = np.asarray(ci0[0]).reshape(-1)
        amplitude = np.vdot(bra, self.transition_operator @ ket)
        dm1s = (
            amplitude * self.dm1a_operator,
            amplitude * self.dm1b_operator,
        )
        if name.startswith("trans_rdm12s"):
            dm2s = tuple(
                amplitude * scale * self.dm2_operator
                for scale in self.dm2_scales
            )
            return [(dm1s, dm2s)]
        return [dm1s]


class _IdentityCSFTransformer:
    """Two-determinant/two-CSF transform used to test complex packing."""

    ndet = 2
    ncsf = 2

    @staticmethod
    def vec_det2csf(civec, order="C", normalize=False):
        return np.array(civec, copy=True)

    @staticmethod
    def vec_csf2det(civec, order="C", normalize=False):
        return np.array(civec, copy=True)

    @staticmethod
    def pack_csf(civec):
        return np.array(civec, copy=True)


class _SingleSolverFCIBox:
    """Minimal state-average wrapper around one complex FCI solver."""

    _state_args = staticmethod(lambda value: value)
    _solver_args = staticmethod(lambda value: value)

    def __init__(self):
        self.solver = direct_spin1_cplx.FCISolver()
        self.fcisolvers = [self.solver]

    @staticmethod
    def _get_nelec(solver, nelec):
        return tuple(nelec)

    def _collect(
            self, name, ci1, ci0, norb, nelec, link_index=None, **kwargs):
        method = getattr(self.solver, name)
        link = None if link_index is None else link_index[0]
        return [method(
            ci1[0], ci0[0], norb, nelec[0], link_index=link,
        )]


class _ConstructorSCF:
    """Identity-overlap SCF container for constructor tests."""

    cell = object()

    @staticmethod
    def get_ovlp(kpts=None):
        return np.broadcast_to(np.eye(3), (len(kpts), 3, 3)).copy()


class _ConstructorLAS:
    """Minimal LAS API needed to build all Hessian intermediates."""

    def __init__(self, casdm2, hcore):
        self._scf = _ConstructorSCF()
        self.kpts = np.zeros((2, 3))
        self.kmesh = (2, 1, 1)
        self.mo_coeff = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
        self.ci = [
            [np.array([1.0, 0.0], dtype=np.complex128)],
            [np.array([1.0, 0.0], dtype=np.complex128)],
        ]
        self.ah_level_shift = 1e-8
        self.ncore = 1
        self.ncas = 1
        self.ncas_sub = np.array([1, 1])
        self.nelecas_sub = np.array([(1, 0), (1, 0)])
        self.fciboxes = [object(), object()]
        self.nroots = 1
        self.weights = np.array([1.0])
        self._casdm2 = np.asarray(casdm2)
        self._hcore = np.asarray(hcore)

    @staticmethod
    def make_casdm1s_sub(casdm1frs=None):
        return [np.asarray(dm[0]) for dm in casdm1frs]

    @staticmethod
    def states_make_casdm1s(casdm1frs=None):
        result = np.zeros((1, 2, 2, 2), dtype=np.complex128)
        for cell, dm1 in enumerate(casdm1frs):
            result[:, :, cell, cell] = dm1[:, :, 0, 0]
        return result

    def make_casdm2(self, **kwargs):
        return np.array(self._casdm2, copy=True)

    def get_hcore(self, kpts=None):
        return np.array(self._hcore, copy=True)


class _ConstructorUGG:
    def __init__(self):
        self.ci_transformers = [
            [_IdentityCSFTransformer()],
            [_IdentityCSFTransformer()],
        ]
        self.frozen_ci = set()


class _DiskERIs:
    """Lazy-ERI stand-in that records every requested k-point block."""

    def __init__(self):
        self.paaa_calls = []

    @staticmethod
    def ppaa(k1, k2, k3):
        return None

    @staticmethod
    def papa(k1, k2, k3):
        return None

    @staticmethod
    def paap(k1, k2, k3):
        return None

    def paaa(self, k1, k2, k3):
        self.paaa_calls.append((k1, k2, k3))
        value = 1.0 + k1 + 2.0 * k2 + 3.0 * k3
        return value * np.arange(1, 4, dtype=float).reshape(3, 1, 1, 1)


def _set_csf_layout(operator):
    operator.ci_transformers = [
        [_IdentityCSFTransformer()] for _ in operator.ci
    ]
    operator.frozen_ci = set()
    operator.nvar_ci = sum(
        transformer.ncsf
        for transformers in operator.ci_transformers
        for transformer in transformers
    )


class _DispatchUGG:
    """One-orbital-variable UGG for combined Hessian dispatch tests."""

    def __init__(self, operator):
        self.operator = operator
        self.nvar_orb = 1
        self.nvar_ci = operator.nvar_ci
        self.nvar_tot = self.nvar_orb + self.nvar_ci

    def unpack(self, x):
        x = np.asarray(x).reshape(-1)
        if x.size != self.nvar_tot:
            raise ValueError(
                f"combined vector has size {x.size}; expected {self.nvar_tot}"
            )
        kappa = self.unpack_orb(x[:self.nvar_orb])
        ci = self.operator._unpack_ci_vector(x[self.nvar_orb:])
        return kappa, ci

    def pack(self, kappa, ci):
        x_orb = self.pack_orb(kappa)
        x_ci = self.operator._flatten_ci_vector(ci)
        return np.concatenate((x_orb, x_ci))

    def unpack_orb(self, x_orb):
        x_orb = np.asarray(x_orb).reshape(-1)
        if x_orb.size != self.nvar_orb:
            raise ValueError(
                f"orbital vector has size {x_orb.size}; "
                f"expected {self.nvar_orb}"
            )
        kappa = np.zeros(
            (1, 2, 2), dtype=np.result_type(x_orb, complex),
        )
        kappa[0, 1, 0] = x_orb[0]
        kappa[0, 0, 1] = -x_orb[0].conjugate()
        return kappa

    @staticmethod
    def pack_orb(kappa):
        return np.asarray([kappa[0, 1, 0]])


def _new_tdm_operator(cls, phases=None):
    operator = cls.__new__(cls)
    operator.nroots = 1
    operator.ncas_sub = np.array([2, 2])
    operator.nelecas_sub = np.array([(1, 0), (1, 0)])
    operator.ncas = 2
    operator.ncastot = 4
    operator.ncell = 2
    operator.weights = np.array([1.0])
    operator.fciboxes = [_FakeFCIBox(), _FakeFCIBox()]
    operator.linkstr = [None, None]
    operator.eri_cas = np.zeros((4, 4, 4, 4), dtype=np.complex128)
    operator.casdm1frs = [
        np.zeros((1, 2, 2, 2), dtype=np.complex128)
        for _ in range(2)
    ]
    operator.casdm1s = np.zeros((2, 4, 4), dtype=np.complex128)
    operator.casdm2fr = [
        np.zeros((1, 2, 2, 2, 2), dtype=np.complex128)
        for _ in range(2)
    ]

    ci_ref = np.array([[0.8], [0.6j]], dtype=np.complex128)
    if phases is None:
        phases = np.ones(2, dtype=np.complex128)
    phases = np.asarray(phases)
    operator.ref_cell = 0
    operator.phase_per_frag = phases
    operator.ci = [[phase * ci_ref] for phase in phases]
    _set_csf_layout(operator)
    return operator, ci_ref


def _set_toy_matvec_pipeline(operator):
    operator.make_tdm1s_sub = lambda ci1: "tdm"

    def make_tdm1s2c_sub(ci1):
        operator._last_ci1 = ci1
        return "tdm", "tcm2"

    operator.make_tdm1s2c_sub = make_tdm1s2c_sub

    def orbital_ci_response(tdm1rs, tcm2):
        assert tdm1rs == "tdm"
        assert tcm2 == "tcm2"
        value = operator._last_ci1[0][0][0, 0]
        response = np.zeros((1, 2, 2), dtype=np.complex128)
        response[0, 1, 0] = 2.0 * value
        response[0, 0, 1] = -2.0 * value.conjugate()
        return response

    operator._orbital_ci_hessian_response = orbital_ci_response
    operator.get_h1eff_response = lambda tdm: "h1-response"
    operator.ci_response_diag = lambda ci1: [
        [2.0 * trial for trial in trial_r] for trial_r in ci1
    ]
    operator.ci_response_offdiag = lambda h1: [
        [3.0 * trial for trial in trial_r] for trial_r in operator._last_ci1
    ]

    original_diag = operator.ci_response_diag

    def cache_and_apply(ci1):
        operator._last_ci1 = ci1
        return original_diag(ci1)

    operator.ci_response_diag = cache_and_apply


class KnownValuesKLASSCFHessianOperator(unittest.TestCase):

    def test_check_shape_raises_for_mismatched_array(self):
        klasscf._check_shape(np.zeros((2, 3)), (2, 3))
        with self.assertRaisesRegex(
                ValueError,
                r"array has shape \(2, 3\); expected \(3, 2\)"):
            klasscf._check_shape(np.zeros((2, 3)), (3, 2))

    def test_periodic_orbital_update_uses_half_generator_per_kpoint(self):
        rng = np.random.default_rng(311)
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.nkpts = 3
        operator.nao = 4
        operator.nmo = 3
        operator.mo_coeff = np.asarray([
            np.linalg.qr(
                rng.standard_normal((4, 3))
                + 1j * rng.standard_normal((4, 3))
            )[0]
            for _ in range(operator.nkpts)
        ])
        mo0 = np.array(operator.mo_coeff, copy=True)
        kappa = (
            rng.standard_normal((3, 3, 3))
            + 1j * rng.standard_normal((3, 3, 3))
        )
        kappa = kappa - kappa.conj().transpose(0, 2, 1)
        kappa *= 0.2 / np.linalg.norm(kappa)

        mo1 = operator._update_mo(kappa)

        expected = np.asarray([
            mo_k @ linalg.expm(kappa_k / 2.0)
            for mo_k, kappa_k in zip(mo0, kappa)
        ])
        np.testing.assert_allclose(mo1, expected, atol=2e-14, rtol=2e-14)
        np.testing.assert_array_equal(operator.mo_coeff, mo0)
        for k in range(operator.nkpts):
            np.testing.assert_allclose(
                mo1[k].conj().T @ mo1[k],
                mo0[k].conj().T @ mo0[k],
                atol=2e-13,
                rtol=2e-13,
            )

        bad_kappa = np.array(kappa, copy=True)
        bad_kappa[0, 0, 0] = 0.1
        with self.assertRaisesRegex(ValueError, "must be anti-Hermitian"):
            operator._update_mo(bad_kappa)

    def test_periodic_ci_update_retracts_complex_projected_tangents(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        ci00 = np.array([[1.0], [1.0j]]) / np.sqrt(2.0)
        ci10 = np.array([[0.5, -0.5j, np.sqrt(0.5)]])
        operator.ci = [[ci00], [ci10]]
        dci = [[np.array([[0.3 + 0.1j], [-0.2 + 0.4j]])], [
            np.array([[0.1j, 0.25, -0.35j]])
        ]]
        dci0 = [[np.array(dc, copy=True) for dc in dc_r] for dc_r in dci]

        ci1 = operator._update_ci(dci)

        for ifrag, (ci0_r, dc_r, ci1_r) in enumerate(zip(
                operator.ci, dci, ci1)):
            for iroot, (ci0, dc, c1) in enumerate(zip(
                    ci0_r, dc_r, ci1_r)):
                reference = ci0.reshape(-1)
                tangent = dc.reshape(-1)
                tangent = tangent - reference * np.vdot(
                    reference, tangent,
                )
                tangent_norm = np.linalg.norm(tangent)
                expected = (
                    np.cos(tangent_norm) * reference
                    + np.sinc(tangent_norm / np.pi) * tangent
                ).reshape(ci0.shape)
                with self.subTest(cell=ifrag, root=iroot):
                    np.testing.assert_allclose(c1, expected, atol=2e-14)
                    np.testing.assert_allclose(np.linalg.norm(c1), 1.0)
                    np.testing.assert_allclose(
                        np.vdot(reference, tangent), 0.0, atol=2e-16,
                    )
                    np.testing.assert_array_equal(dc, dci0[ifrag][iroot])

        parallel_step = [
            [(0.2 - 0.3j) * ci00],
            [np.zeros_like(ci10)],
        ]
        parallel_result = operator._update_ci(parallel_step)
        np.testing.assert_allclose(parallel_result[0][0], ci00)
        np.testing.assert_allclose(parallel_result[1][0], ci10)

    def test_combined_update_rebuilds_wannier_active_integrals(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.nkpts = 1
        operator.nao = 2
        operator.nmo = 2
        operator.ncastot = 2
        operator.mo_coeff = np.eye(2, dtype=np.complex128)[None]
        ci0 = np.array([[np.sqrt(0.7)], [1j * np.sqrt(0.3)]])
        operator.ci = [[ci0]]
        kappa = np.array([[
            [0.0, -0.2 - 0.1j],
            [0.2 - 0.1j, 0.0],
        ]])
        dci = [[np.array([[0.15j], [0.2]])]]

        class StepUGG:
            nvar_tot = 2

            def __init__(self, orbital_step, ci_step):
                self.orbital_step = orbital_step
                self.ci_step = ci_step

            def unpack(self, x):
                return self.orbital_step, self.ci_step

        class StepLAS:
            updated_mo = None

            def get_h2cas(self, mo_coeff):
                self.updated_mo = np.array(mo_coeff, copy=True)
                return np.full((2,) * 4, 0.4 - 0.2j)

        operator.ugg = StepUGG(kappa, dci)
        operator.las = StepLAS()

        mo1, ci1, h2eff1 = operator.update_mo_ci_eri(
            np.array([0.1, -0.2j]), h2eff_sub=np.zeros((2,) * 4),
        )

        expected_mo = np.asarray([
            operator.mo_coeff[0] @ linalg.expm(kappa[0] / 2.0)
        ])
        np.testing.assert_allclose(mo1, expected_mo)
        np.testing.assert_allclose(operator.las.updated_mo, expected_mo)
        np.testing.assert_allclose(h2eff1, 0.4 - 0.2j)
        np.testing.assert_allclose(np.linalg.norm(ci1[0][0]), 1.0)
        self.assertEqual(ci1[0][0].shape, ci0.shape)

    def test_constructor_builds_orbital_and_ci_intermediates(self):
        casdm1frs = [
            np.array([[[[0.8]], [[0.2]]]], dtype=np.complex128),
            np.array([[[[0.3]], [[0.7]]]], dtype=np.complex128),
        ]
        casdm2fr = [np.zeros((1, 1, 1, 1, 1)) for _ in range(2)]
        casdm2 = np.arange(16, dtype=float).reshape((2,) * 4) / 13.0
        h2eff = np.arange(16, dtype=float).reshape((2,) * 4) / 17.0
        hcore = np.array([
            np.diag([1.0, 2.0, 3.0]),
            np.diag([1.5, 2.5, 3.5]),
        ])
        veff_kpts = np.zeros((2, 2, 3, 3), dtype=np.complex128)
        veff_kpts[0, 0] = np.diag([0.1, 0.2, 0.3])
        veff_kpts[1, 0] = np.diag([0.4, 0.5, 0.6])
        veff_kpts[0, 1] = np.diag([0.7, 0.8, 0.9])
        veff_kpts[1, 1] = np.diag([1.0, 1.1, 1.2])
        dm1s_kpts = np.zeros((2, 2, 3, 3), dtype=np.complex128)
        dm1s_kpts[0, 0] = np.diag([1.0, 0.8, 0.0])
        dm1s_kpts[1, 0] = np.diag([1.0, 0.2, 0.0])
        dm1s_kpts[0, 1] = np.diag([1.0, 0.3, 0.0])
        dm1s_kpts[1, 1] = np.diag([1.0, 0.7, 0.0])
        h1eff = [
            np.zeros((1, 2, 1, 1), dtype=np.complex128)
            for _ in range(2)
        ]
        mo_phase = np.array([
            [[1.0, 1.0]],
            [[1.0, -1.0]],
        ], dtype=np.complex128) / np.sqrt(2.0)
        transformed_cumulants = {
            (k1, k2, k3): np.full(
                (1, 1, 1, 1), 0.25 + k1 + 2 * k2 + 4 * k3,
                dtype=np.complex128,
            )
            for k1 in range(2)
            for k2 in range(2)
            for k3 in range(2)
        }

        las = _ConstructorLAS(casdm2, hcore)
        eris = _DiskERIs()
        transformed_inputs = []

        def transform(cumulant, phase, klabel):
            transformed_inputs.append((cumulant, phase, tuple(klabel)))
            return transformed_cumulants[tuple(klabel[:3])]

        with patch.object(
                KLASSCF_HessianOperator, "_init_ci_"), patch.object(
                klasscf.kpts_helper, "get_kconserv",
                return_value=np.zeros((2, 2, 2), dtype=int)), patch.object(
                klasscf, "_get_casdm2_kpts", side_effect=transform):
            operator = KLASSCF_HessianOperator(
                las, _ConstructorUGG(), casdm1frs=casdm1frs,
                casdm2fr=casdm2fr, h1eff=h1eff, h2eff=h2eff,
                eris=eris, veff_kpts=veff_kpts,
                dm1s_kpts=dm1s_kpts, mo_phase=mo_phase,
            )

        expected_casdm1s = np.zeros((2, 2, 2), dtype=np.complex128)
        expected_casdm1s[:, 0, 0] = [0.8, 0.2]
        expected_casdm1s[:, 1, 1] = [0.3, 0.7]
        casdm1 = expected_casdm1s.sum(axis=0)
        expected_cascm2 = casdm2 - np.multiply.outer(casdm1, casdm1)
        for spin_dm in expected_casdm1s:
            expected_cascm2 += np.multiply.outer(
                spin_dm, spin_dm,
            ).transpose(0, 3, 2, 1)

        np.testing.assert_allclose(operator.casdm1s, expected_casdm1s)
        np.testing.assert_allclose(operator.casdm2, casdm2)
        np.testing.assert_allclose(operator.cascm2, expected_cascm2)
        self.assertEqual(operator.ncas, 1)
        self.assertEqual(operator.ncastot, 2)
        self.assertFalse(hasattr(operator, "ncas_kpts"))
        for cumulant, phase, _ in transformed_inputs:
            self.assertIs(cumulant, operator.cascm2)
            np.testing.assert_array_equal(phase, mo_phase)
        np.testing.assert_allclose(operator.dm1s_kpts, dm1s_kpts)
        np.testing.assert_allclose(operator.dm1s, dm1s_kpts)
        np.testing.assert_allclose(operator.h1s, hcore[None] + veff_kpts)
        np.testing.assert_allclose(operator.eri_cas, h2eff)
        self.assertIs(operator.cas_type_eris, eris)
        self.assertIs(operator.eris, eris)

        expected_fock1 = np.empty((2, 3, 3), dtype=np.complex128)
        for k in range(2):
            expected_fock1[k] = sum(
                operator.h1s[s, k] @ dm1s_kpts[s, k]
                for s in range(2)
            )
        for k1, k2, k3 in klasscf.kpts_helper.loop_kkk(2):
            paaa = 1.0 + k1 + 2.0 * k2 + 3.0 * k3
            paaa *= np.arange(1, 4, dtype=float).reshape(3, 1)
            expected_fock1[k1, :, 1:2] += (
                paaa * transformed_cumulants[(k1, k2, k3)].item()
            )
        np.testing.assert_allclose(operator.fock1, expected_fock1)
        self.assertCountEqual(
            eris.paaa_calls,
            list(klasscf.kpts_helper.loop_kkk(2)),
        )

    def test_default_eris_is_disk_backed_with_diagonal_intermediates(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.las = object()
        operator.mo_coeff = np.zeros((1, 2, 2))
        fake_eris = _DiskERIs()

        with patch.object(klasscf, "_ERIS", return_value=fake_eris) as eris_cls:
            operator._init_eri_()

        eris_cls.assert_called_once_with(
            operator.las, operator.mo_coeff, method="disk", level=1,
        )
        self.assertIs(operator.cas_type_eris, fake_eris)
        self.assertIs(operator.eri_paaa.__self__, fake_eris)

    def test_complex_orbital_step_builds_one_sided_density_responses(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.nkpts = 2
        operator.nmo = 3
        operator.ncas = 1
        operator.ncore = 1
        operator.nocc = 2
        operator.kpts = np.zeros((2, 3))
        operator.las = type("LAS", (), {
            "_scf": type("SCF", (), {"cell": object()})(),
        })()
        operator.mo_phase = np.ones((2, 1, 2), dtype=np.complex128)
        operator.cascm2 = np.ones((2,) * 4, dtype=np.complex128)
        operator.dm1s = np.arange(36, dtype=float).reshape(2, 2, 3, 3)
        operator.dm1s = operator.dm1s.astype(np.complex128)
        operator.dm1s += 0.2j * operator.dm1s.transpose(0, 1, 3, 2)

        kappa = np.zeros((2, 3, 3), dtype=np.complex128)
        kappa[:, 1, 0] = [0.3 + 0.2j, -0.1 + 0.4j]
        kappa[:, 0, 1] = -kappa[:, 1, 0].conj()
        kappa[:, 2, 1] = [-0.2j, 0.25 - 0.15j]
        kappa[:, 1, 2] = -kappa[:, 2, 1].conj()
        kconserv = np.fromfunction(
            lambda k1, k2, k3: (k1 - k2 + k3) % 2,
            (2, 2, 2), dtype=int,
        ).astype(int)

        def transform(cumulant, phase, klabel):
            value = 1.0 + klabel[0] + 2 * klabel[1] + 4 * klabel[2]
            return np.full((1,) * 4, value + 0.5j)

        with patch.object(
                klasscf.kpts_helper, "get_kconserv",
                return_value=kconserv), patch.object(
                klasscf, "_get_casdm2_kpts", side_effect=transform):
            odm1s, ocm2 = operator._make_orbital_response_dm(kappa)

        np.testing.assert_allclose(
            odm1s,
            -np.einsum("skpr,krq->skpq", operator.dm1s, kappa),
        )
        for k1, k2, k3 in klasscf.kpts_helper.loop_kkk(2):
            k4 = kconserv[k1, k2, k3]
            value = 1.0 + k1 + 2 * k2 + 4 * k3 + 0.5j
            np.testing.assert_allclose(
                ocm2[k1, k2, k3, 0, 0, 0],
                -value * kappa[k4, 1],
            )

    def test_jk_response_uses_hermitian_complex_ao_density(self):
        captured = {}

        class FakeLAS:
            _scf = type("SCF", (), {"cell": object()})()

            @staticmethod
            def get_veff(cell, dm_kpts=None, hermi=None, kpts=None):
                captured["hermi"] = hermi
                captured["kpts"] = np.array(kpts, copy=True)
                captured["dm_kpts"] = np.array(dm_kpts, copy=True)
                return (1.7 - 0.2j) * dm_kpts

        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.las = FakeLAS()
        operator.nkpts = 1
        operator.nao = 2
        operator.nmo = 2
        operator.kpts = np.zeros((1, 3))
        operator.mo_coeff = np.array([[
            [1.0, 0.2j],
            [0.1 - 0.3j, 0.9],
        ]], dtype=np.complex128)
        odm1s = np.array([[[
            [0.2 + 0.1j, -0.3j],
            [0.4 + 0.2j, -0.1],
        ]], [[
            [-0.2j, 0.15 + 0.05j],
            [-0.25j, 0.3 - 0.1j],
        ]]], dtype=np.complex128)

        actual = operator._get_veff_response(odm1s)

        edm1s = odm1s + odm1s.conj().transpose(0, 1, 3, 2)
        mo = operator.mo_coeff[0]
        expected_dm_ao = mo @ edm1s[:, 0] @ mo.conj().T
        expected_veff = (
            mo.conj().T @ ((1.7 - 0.2j) * expected_dm_ao) @ mo
        )
        np.testing.assert_allclose(captured["dm_kpts"][:, 0], expected_dm_ao)
        self.assertEqual(captured["hermi"], 1)
        np.testing.assert_array_equal(captured["kpts"], operator.kpts)
        np.testing.assert_allclose(
            captured["dm_kpts"],
            captured["dm_kpts"].conj().transpose(0, 1, 3, 2),
        )
        np.testing.assert_allclose(actual[:, 0], expected_veff)

        captured.clear()
        actual_ci = operator._get_ci_veff_response(edm1s)
        np.testing.assert_allclose(captured["dm_kpts"][:, 0], expected_dm_ao)
        self.assertEqual(captured["hermi"], 1)
        np.testing.assert_array_equal(captured["kpts"], operator.kpts)
        np.testing.assert_allclose(actual_ci[:, 0], expected_veff)

    def test_orbital_ci_response_assembles_all_terms_and_normalization(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.nkpts = 2
        operator.nmo = 3
        rng = np.random.default_rng(101)
        operator.h1s = (
            rng.standard_normal((2, 2, 3, 3))
            + 1j * rng.standard_normal((2, 2, 3, 3))
        )
        operator.dm1s = (
            rng.standard_normal((2, 2, 3, 3))
            + 1j * rng.standard_normal((2, 2, 3, 3))
        )
        tdm1s_block = (
            rng.standard_normal((2, 2, 3, 3))
            + 1j * rng.standard_normal((2, 2, 3, 3))
        )
        veff_ci = (
            rng.standard_normal((2, 2, 3, 3))
            + 1j * rng.standard_normal((2, 2, 3, 3))
        )
        cumulant_fock = (
            rng.standard_normal((2, 3, 3))
            + 1j * rng.standard_normal((2, 3, 3))
        )
        tdm1rs = np.array([17.0])
        tcm2 = np.array([23.0])
        calls = []

        def transform_dm1(actual):
            np.testing.assert_array_equal(actual, tdm1rs)
            calls.append("dm1")
            return tdm1s_block

        def respond_jk(actual):
            np.testing.assert_array_equal(actual, tdm1s_block)
            calls.append("jk")
            return veff_ci

        def contract_cumulant(actual):
            np.testing.assert_array_equal(actual, tcm2)
            calls.append("cumulant")
            return cumulant_fock

        operator._transition_dm1s_to_block = transform_dm1
        operator._get_ci_veff_response = respond_jk
        operator._transition_cumulant_to_block_fock = contract_cumulant

        fock_expected = np.array(cumulant_fock, copy=True)
        for k in range(operator.nkpts):
            for spin in range(2):
                fock_expected[k] += (
                    operator.h1s[spin, k] @ tdm1s_block[spin, k]
                )
                fock_expected[k] += (
                    veff_ci[spin, k] @ operator.dm1s[spin, k]
                )
        expected = 2.0 * (
            fock_expected - fock_expected.conj().transpose(0, 2, 1)
        )

        actual = operator._orbital_ci_hessian_response(tdm1rs, tcm2)

        self.assertEqual(calls, ["dm1", "jk", "cumulant"])
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(
            actual + actual.conj().transpose(0, 2, 1), 0.0,
            atol=1e-13,
        )

    def test_complex_external_cumulant_response_matches_integral_derivative(self):
        rng = np.random.default_rng(19)
        nkpts = 3
        nmo = 3
        active = slice(1, 2)
        kconserv = np.fromfunction(
            lambda k1, k2, k3: (k1 - k2 + k3) % nkpts,
            (nkpts,) * 3, dtype=int,
        ).astype(int)
        # Unlike a two-point mesh, this mesh distinguishes the ERI rule
        # k1-k2+k3-k4 from the regrouped Hessian rule k1+k2-k3-k4.
        self.assertNotEqual(kconserv[0, 1, 0], (0 + 1 - 0) % nkpts)
        eri = {
            klabel: (
                rng.standard_normal((nmo,) * 4)
                + 1j * rng.standard_normal((nmo,) * 4)
            )
            for klabel in np.ndindex((nkpts,) * 3)
        }
        cumulant = {
            klabel: np.full(
                (1,) * 4,
                rng.standard_normal() + 1j * rng.standard_normal(),
            )
            for klabel in np.ndindex((nkpts,) * 3)
        }

        class FakeERIs:
            @staticmethod
            def ppaa(k1, k2, k3):
                return eri[k1, k2, k3][:, :, active, active]

            @staticmethod
            def papa(k1, k2, k3):
                return eri[k1, k2, k3][:, active, :, active]

            @staticmethod
            def paap(k1, k2, k3):
                return eri[k1, k2, k3][:, active, active, :]

        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.nkpts = nkpts
        operator.nmo = nmo
        operator.ncas = 1
        operator.ncore = 1
        operator.nocc = 2
        operator.kpts = np.zeros((nkpts, 3))
        operator.las = type("LAS", (), {
            "_scf": type("SCF", (), {"cell": object()})(),
        })()
        operator.eris = FakeERIs()
        operator.cascm2 = np.zeros((nkpts,) * 4, dtype=np.complex128)
        operator.mo_phase = np.ones(
            (nkpts, 1, nkpts), dtype=np.complex128,
        )

        kappa = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)
        kappa[:, 1, 0] = [0.3 + 0.2j, -0.2 + 0.1j, 0.15 - 0.25j]
        kappa[:, 2, 0] = [-0.1 + 0.15j, 0.25j, -0.3 - 0.1j]
        kappa[:, 2, 1] = [0.25 - 0.35j, -0.2j, 0.1 + 0.3j]
        kappa -= kappa.conj().transpose(0, 2, 1)
        fock1_cumulant = np.zeros(
            (nkpts, nmo, nmo), dtype=np.complex128,
        )
        for k1, k2, k3 in np.ndindex((nkpts,) * 3):
            fock1_cumulant[k1, :, 1] += (
                eri[k1, k2, k3][:, 1, 1, 1]
                * cumulant[k1, k2, k3].item()
            )

        def transform(cascm2, mo_phase, klabel):
            return cumulant[tuple(klabel[:3])]

        with patch.object(
                klasscf.kpts_helper, "get_kconserv",
                return_value=kconserv), patch.object(
                klasscf, "_get_casdm2_kpts", side_effect=transform):
            response = operator._orbital_response_external_cumulant(
                kappa, fock1_cumulant,
            )

        commutator = (
            fock1_cumulant @ kappa - kappa @ fock1_cumulant
        ) / 2.0
        actual = response + commutator
        actual = actual - actual.conj().transpose(0, 2, 1)

        def cumulant_gradient(step):
            unitary = np.asarray([
                linalg.expm(step * kappa_k) for kappa_k in kappa
            ])
            fock = np.zeros(
                (nkpts, nmo, nmo), dtype=np.complex128,
            )
            for k1, k2, k3 in np.ndindex((nkpts,) * 3):
                k4 = kconserv[k1, k2, k3]
                eri_rotated = np.einsum(
                    "xp,yr,zs,wt,xyzw->prst",
                    unitary[k1].conj(), unitary[k2],
                    unitary[k3].conj(), unitary[k4],
                    eri[k1, k2, k3], optimize=True,
                )
                fock[k1, :, 1] += (
                    eri_rotated[:, 1, 1, 1]
                    * cumulant[k1, k2, k3].item()
                )
            return fock - fock.conj().transpose(0, 2, 1)

        step = 1e-5
        gradient_derivative = (
            cumulant_gradient(step) - cumulant_gradient(-step)
        ) / (2.0 * step)
        connection = commutator - commutator.conj().transpose(0, 2, 1)
        expected = gradient_derivative - connection
        np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-9)
        np.testing.assert_allclose(actual + actual.conj().transpose(0, 2, 1), 0)

    def test_orbital_hessian_response_is_skew_hermitian_for_complex_step(self):
        rng = np.random.default_rng(29)
        nmo = 3
        active = slice(1, 2)
        eri = (
            rng.standard_normal((nmo,) * 4)
            + 1j * rng.standard_normal((nmo,) * 4)
        )
        cumulant = np.full((1,) * 4, 0.4, dtype=np.complex128)

        class FakeLAS:
            _scf = type("SCF", (), {"cell": object()})()

            @staticmethod
            def get_veff(cell, dm_kpts=None, hermi=None, kpts=None):
                return np.zeros_like(dm_kpts)

        class FakeERIs:
            @staticmethod
            def ppaa(k1, k2, k3):
                return eri[:, :, active, active]

            @staticmethod
            def papa(k1, k2, k3):
                return eri[:, active, :, active]

            @staticmethod
            def paap(k1, k2, k3):
                return eri[:, active, active, :]

            @staticmethod
            def paaa(k1, k2, k3):
                return eri[:, active, active, active]

        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.las = FakeLAS()
        operator.eris = FakeERIs()
        operator.eri_paaa = operator.eris.paaa
        operator.nkpts = 1
        operator.nao = nmo
        operator.nmo = nmo
        operator.ncas = 1
        operator.ncore = 1
        operator.nocc = 2
        operator.kpts = np.zeros((1, 3))
        operator.mo_coeff = np.eye(nmo, dtype=np.complex128)[None]
        operator.mo_phase = np.ones((1, 1, 1), dtype=np.complex128)
        operator.cascm2 = cumulant
        operator.dm1s = np.zeros((2, 1, nmo, nmo), dtype=np.complex128)
        operator.dm1s[:, 0] = np.asarray([
            np.diag([1.0, 0.7, 0.0]),
            np.diag([1.0, 0.3, 0.0]),
        ])
        operator.h1s = (
            rng.standard_normal((2, 1, nmo, nmo))
            + 1j * rng.standard_normal((2, 1, nmo, nmo))
        )
        operator.fock1 = np.zeros((1, nmo, nmo), dtype=np.complex128)
        for spin in range(2):
            operator.fock1[0] += (
                operator.h1s[spin, 0] @ operator.dm1s[spin, 0]
            )
        operator.fock1[0, :, 1] += eri[:, 1, 1, 1] * cumulant.item()

        kappa = np.zeros((1, nmo, nmo), dtype=np.complex128)
        kappa[0, 1, 0] = 0.2 + 0.3j
        kappa[0, 2, 1] = -0.15 + 0.25j
        kappa -= kappa.conj().transpose(0, 2, 1)

        with patch.object(
                klasscf.kpts_helper, "get_kconserv",
                return_value=np.zeros((1, 1, 1), dtype=int)), patch.object(
                klasscf, "_get_casdm2_kpts", return_value=cumulant):
            response = operator._orbital_hessian_response(kappa)

        self.assertGreater(np.linalg.norm(response), 1e-10)
        np.testing.assert_allclose(
            response + response.conj().transpose(0, 2, 1), 0,
            atol=1e-13,
        )

    def test_matvec_dispatches_combined_vector_to_ci_block(self):
        operator = KLASSCF_HessianOperator.__new__(KLASSCF_HessianOperator)
        operator.ci = [
            [np.zeros((2, 1), dtype=np.complex128)],
            [np.zeros((2, 1), dtype=np.complex128)],
        ]
        _set_csf_layout(operator)
        operator.ugg = _DispatchUGG(operator)
        operator.level_shift = 0.25
        _set_toy_matvec_pipeline(operator)

        ci_trial = np.array(
            [0.2 + 0.1j, -0.3j, 0.4 - 0.2j, -0.1],
            dtype=np.complex128,
        )
        trial = np.concatenate(([0.0j], ci_trial))
        result = operator._matvec(trial)

        self.assertEqual(result.shape, trial.shape)
        self.assertEqual(operator.shape, (trial.size, trial.size))
        self.assertTrue(np.issubdtype(result.dtype, np.complexfloating))
        np.testing.assert_allclose(result[0], ci_trial[0])
        np.testing.assert_allclose(result[1:], 5.25 * ci_trial)

    def test_matvec_dispatches_orbital_only_step(self):
        operator = KLASSCF_HessianOperator.__new__(KLASSCF_HessianOperator)
        operator.ci = [
            [np.zeros((2, 1), dtype=np.complex128)],
            [np.zeros((2, 1), dtype=np.complex128)],
        ]
        _set_csf_layout(operator)
        operator.ugg = _DispatchUGG(operator)
        operator.level_shift = 0.0
        calls = []

        def orbital_response(kappa1):
            calls.append(np.array(kappa1, copy=True))
            return 2.0 * kappa1

        operator._orbital_hessian_response = orbital_response

        def ci_orbital_response(kappa1):
            value = kappa1[0, 1, 0]
            return [
                [value * np.ones_like(c0) for c0 in ci0_r]
                for ci0_r in operator.ci
            ]

        operator._ci_orbital_hessian_response = ci_orbital_response
        trial = np.zeros(operator.ugg.nvar_tot, dtype=np.complex128)
        trial[0] = 0.3 - 0.2j

        result = operator._matvec(trial)

        self.assertEqual(len(calls), 1)
        np.testing.assert_allclose(calls[0][0, 1, 0], trial[0])
        np.testing.assert_allclose(result[0], trial[0])
        np.testing.assert_allclose(result[1:], trial[0])

    def test_orbital_hdiag_matches_matvec_diagonal_and_is_cached(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.ci = [[np.zeros((2, 1), dtype=np.complex128)]]
        _set_csf_layout(operator)
        operator.ugg = _DispatchUGG(operator)
        operator.level_shift = 0.0
        operator._Horb_diag_matvec_cache = None
        calls = []

        def orbital_response(kappa):
            calls.append(np.array(kappa, copy=True))
            return 6.0 * kappa

        operator._orbital_hessian_response = orbital_response
        operator._ci_orbital_hessian_response = (
            lambda kappa: operator._zero_ci_step(kappa.dtype)
        )
        diagonal = operator._get_Horb_diag_matvec()

        trial = np.zeros(operator.ugg.nvar_tot, dtype=np.complex128)
        trial[0] = 1.0
        result = operator._matvec(trial)
        np.testing.assert_allclose(diagonal, result[:1])
        np.testing.assert_allclose(diagonal, [3.0])
        self.assertEqual(len(calls), 2)

        np.testing.assert_allclose(
            operator._get_Horb_diag_matvec(), diagonal,
        )
        self.assertEqual(len(calls), 2)

    def test_active_active_hessian_keeps_direct_and_conjugate_blocks(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.ugg = type("UGG", (), {
            "nvar_orb_active_active": 2,
        })()
        operator.mo_coeff = np.zeros((1, 1, 1), dtype=np.complex128)
        operator._Horb_active_active_cache = None
        direct = np.array([
            [1.2 + 0.1j, -0.3j],
            [0.4 - 0.2j, 0.7],
        ])
        conjugate = np.array([
            [0.25, 0.1 + 0.15j],
            [-0.2j, -0.35 + 0.05j],
        ])
        calls = []

        def apply(coordinates):
            calls.append(np.array(coordinates, copy=True))
            return (
                direct @ coordinates
                + conjugate @ coordinates.conj()
            )

        operator._apply_Horb_active_active = apply
        actual_direct, actual_conjugate = (
            operator._get_Horb_active_active()
        )

        np.testing.assert_allclose(actual_direct, direct)
        np.testing.assert_allclose(actual_conjugate, conjugate)
        self.assertEqual(len(calls), 4)

        probe = np.array([0.2 - 0.4j, -0.1 + 0.3j])
        np.testing.assert_allclose(
            actual_direct @ probe + actual_conjugate @ probe.conj(),
            apply(probe),
        )
        self.assertEqual(len(calls), 5)

        cached_direct, cached_conjugate = (
            operator._get_Horb_active_active()
        )
        np.testing.assert_allclose(cached_direct, direct)
        np.testing.assert_allclose(cached_conjugate, conjugate)
        self.assertEqual(len(calls), 5)

    def test_hdiag_combines_external_and_active_analytic_slices(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.ugg = type("UGG", (), {
            "nvar_orb": 4,
            "nvar_orb_external": 2,
            "nvar_orb_active_active": 2,
        })()
        operator._get_Horb_diag_external = lambda: np.array(
            [1.0, 2.0], dtype=np.complex128,
        )
        direct = np.array([
            [3.0, 0.2j],
            [-0.1j, 4.0],
        ])
        conjugate = np.array([
            [0.5, 0.1],
            [0.2, -0.25],
        ])
        operator._get_Horb_active_active = lambda: (direct, conjugate)

        diagonal = operator._get_Horb_diag()

        np.testing.assert_allclose(diagonal, [1.0, 2.0, 3.5, 3.75])

    def test_hdiag_combines_orbital_and_csf_operator_order(self):
        operator, _ = _new_tdm_operator(KLASSCF_HessianOperator)
        operator.h1frs = [
            np.zeros((1, 2, 2, 2), dtype=np.complex128)
            for _ in range(2)
        ]
        operator.ugg = _DispatchUGG(operator)
        operator._get_Horb_diag_external = lambda: np.array([3.0])

        hdiag = operator._get_Hdiag()

        np.testing.assert_allclose(
            hdiag, [3.0, 1.25, 2.5, 1.25, 2.5],
        )
        self.assertEqual(
            [box.hdiag_calls for box in operator.fciboxes], [1, 1]
        )

    def test_preconditioner_uses_complete_shifted_hdiag(self):
        operator, _ = _new_tdm_operator(KLASSCF_HessianOperator)
        operator.h1frs = [
            np.zeros((1, 2, 2, 2), dtype=np.complex128)
            for _ in range(2)
        ]
        operator.ugg = _DispatchUGG(operator)
        operator.level_shift = 0.25
        operator._get_Horb_diag_external = lambda: np.array([3.0])
        operator.get_grad = lambda: np.full(
            operator.ugg.nvar_tot, 0.1, dtype=np.complex128,
        )
        operator.las = lib.StreamObject()
        operator.las.verbose = lib.logger.QUIET

        preconditioner = operator.get_prec()
        expected_hdiag = np.asarray(
            [3.0, 1.25, 2.5, 1.25, 2.5], dtype=np.complex128,
        ) + operator.level_shift
        np.testing.assert_allclose(preconditioner.Hdiag, expected_hdiag)
        np.testing.assert_allclose(
            preconditioner @ np.ones(operator.ugg.nvar_tot),
            1.0 / expected_hdiag,
        )

    def test_operator_gradient_uses_complex_adjoint_and_combined_order(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.ci = [[np.zeros((2, 1), dtype=np.complex128)]]
        _set_csf_layout(operator)
        operator.ugg = _DispatchUGG(operator)
        operator.fock1 = np.array([[
            [0.7, 0.2 + 0.4j],
            [-0.3 + 0.1j, -0.2],
        ]], dtype=np.complex128)
        operator.hci0 = [[np.array(
            [[0.15 + 0.2j], [-0.25j]], dtype=np.complex128,
        )]]

        gradient = operator.get_grad()

        gorb = operator.fock1 - operator.fock1.conj().transpose(0, 2, 1)
        expected = np.concatenate((
            [gorb[0, 1, 0]],
            2.0 * operator.hci0[0][0].reshape(-1),
        ))
        np.testing.assert_allclose(gradient, expected)

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
        operator.h1frs = None
        operator.eri_cas = None
        operator.Hci_all = lambda h0, h1, h2, ci: [[
            shifted_hamiltonian @ ci[0][0]
        ]]

        actual = operator.ci_response_diag([[trial]])[0][0]

        projector = np.eye(2) - np.outer(c0, c0.conj())
        expected = (
            2.0 * projector @ shifted_hamiltonian @ projector @ trial
        )
        np.testing.assert_allclose(actual, expected)

    def test_transition_rdm_contracts_every_ci_cell(self):
        operator, ci_ref = _new_tdm_operator(KLASSCF_HessianOperator)
        trial_ref = np.array([[0.3j], [0.4]], dtype=np.complex128)
        ci1 = [[trial_ref], [trial_ref.copy()]]

        tdm1rs = operator.make_tdm1s_sub(ci1)

        self.assertEqual(tdm1rs.shape, (1, 2, 4, 4))
        self.assertEqual(
            [box.collect_calls for box in operator.fciboxes], [1, 1]
        )
        np.testing.assert_allclose(tdm1rs[:, :, :2, :2],
                                   tdm1rs[:, :, 2:, 2:])
        np.testing.assert_allclose(
            tdm1rs, tdm1rs.swapaxes(-1, -2).conj()
        )

    def test_transition_cumulant_uses_stored_state_average_casdm1s(self):
        operator, _ = _new_tdm_operator(KLASSCF_HessianOperator)
        trial = np.array([[0.3j], [0.4]], dtype=np.complex128)
        ci1 = [[trial], [trial.copy()]]
        operator.casdm1s = np.zeros((2, 4, 4), dtype=np.complex128)
        operator.casdm1s[0] = np.diag([0.7, 0.2, 0.4, 0.1])
        operator.casdm1s[1] = np.diag([0.1, 0.5, 0.2, 0.6])
        operator.casdm1frs = [
            np.array([[np.diag([0.6, 0.1]), np.diag([0.2, 0.4])]])
            for _ in range(2)
        ]
        operator.casdm2fr = [
            np.full(
                (1, 2, 2, 2, 2), 0.03 * (cell + 1),
                dtype=np.complex128,
            )
            for cell in range(2)
        ]

        tdm1rs, tcm2 = operator.make_tdm1s2c_sub(ci1)

        tdm1rs_one_sided = np.zeros_like(tdm1rs)
        tcm2_one_sided = np.zeros_like(tcm2)
        for cell, box in enumerate(operator.fciboxes):
            i, j = 2 * cell, 2 * (cell + 1)
            c0 = operator.ci[cell][0]
            overlap = np.vdot(trial, c0)
            amplitude = np.vdot(
                trial.reshape(-1),
                box.transition_operator @ c0.reshape(-1),
            )
            tdm1rs_one_sided[0, :, i:j, i:j] = (
                amplitude * np.stack(
                    (box.dm1a_operator, box.dm1b_operator), axis=0,
                )
                - overlap * operator.casdm1frs[cell][0]
            )
            transition_dm2 = (
                amplitude * box.dm2_scales.sum() * box.dm2_operator
            )
            tcm2_one_sided[i:j, i:j, i:j, i:j] = (
                transition_dm2
                - overlap * operator.casdm2fr[cell][0]
            ) / 2.0

        expected_tdm1rs = (
            tdm1rs_one_sided
            + tdm1rs_one_sided.swapaxes(-1, -2).conj()
        )
        expected_tdm2 = np.array(tcm2_one_sided, copy=True)
        expected_tdm2 += expected_tdm2.conj().transpose(1, 0, 3, 2)
        expected_tdm2 += expected_tdm2.transpose(2, 3, 0, 1)

        tdm1s_0 = expected_tdm1rs[0, :, :2, :2]
        tdm1s_1 = expected_tdm1rs[0, :, 2:, 2:]
        dm1s_0 = operator.casdm1frs[0][0]
        dm1s_1 = operator.casdm1frs[1][0]
        coulomb = np.einsum(
            "ij,kl->ijkl", tdm1s_0.sum(axis=0), dm1s_1.sum(axis=0),
        )
        coulomb += np.einsum(
            "ij,kl->ijkl", dm1s_0.sum(axis=0), tdm1s_1.sum(axis=0),
        )
        expected_tdm2[:2, :2, 2:, 2:] = coulomb
        expected_tdm2[2:, 2:, :2, :2] = coulomb.transpose(2, 3, 0, 1)
        exchange = sum(
            np.einsum("ij,kl->ilkj", tdm1s_0[spin], dm1s_1[spin])
            + np.einsum("ij,kl->ilkj", dm1s_0[spin], tdm1s_1[spin])
            for spin in range(2)
        )
        expected_tdm2[:2, 2:, 2:, :2] = -exchange
        expected_tdm2[2:, :2, :2, 2:] = (
            -exchange.conj().transpose(1, 0, 3, 2)
        )

        tdm1s = expected_tdm1rs[0]
        expected_tcm2 = np.array(expected_tdm2, copy=True)
        expected_tcm2 -= np.multiply.outer(
            tdm1s.sum(axis=0),
            operator.casdm1s.sum(axis=0),
        )
        expected_tcm2 -= np.multiply.outer(
            operator.casdm1s.sum(axis=0),
            tdm1s.sum(axis=0),
        )
        for spin in range(2):
            expected_tcm2 += np.multiply.outer(
                tdm1s[spin], operator.casdm1s[spin],
            ).transpose(0, 3, 2, 1)
            expected_tcm2 += np.multiply.outer(
                operator.casdm1s[spin], tdm1s[spin],
            ).transpose(0, 3, 2, 1)

        np.testing.assert_allclose(tdm1rs, expected_tdm1rs)
        np.testing.assert_allclose(tcm2, expected_tcm2)
        np.testing.assert_allclose(
            tdm1rs, tdm1rs.swapaxes(-1, -2).conj(),
        )
        np.testing.assert_allclose(
            tcm2, tcm2.conj().transpose(1, 0, 3, 2),
        )
        np.testing.assert_allclose(
            tcm2, tcm2.transpose(2, 3, 0, 1),
        )

    def test_complex_transition_cumulant_matches_finite_difference(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        box = _SingleSolverFCIBox()
        norb = 2
        nelec = (1, 1)
        rng = np.random.default_rng(91)
        c0 = (
            rng.standard_normal((2, 2))
            + 1j * rng.standard_normal((2, 2))
        )
        c0 /= np.linalg.norm(c0)
        c1 = (
            rng.standard_normal((2, 2))
            + 1j * rng.standard_normal((2, 2))
        )
        c1 -= np.vdot(c0, c1) * c0
        c1 /= np.linalg.norm(c1)

        dm1s_ref = np.asarray(box.solver.make_rdm1s(c0, norb, nelec))
        dm2_ref = np.asarray(box.solver.make_rdm12(c0, norb, nelec)[1])
        operator.nroots = 1
        operator.ncastot = norb
        operator.ncas_sub = np.array([norb])
        operator.nelecas_sub = np.array([nelec])
        operator.weights = np.array([1.0])
        operator.fciboxes = [box]
        operator.linkstr = [None]
        operator.ci = [[c0]]
        operator.eri_cas = np.zeros((norb,) * 4, dtype=np.complex128)
        operator.casdm1frs = [dm1s_ref[None]]
        operator.casdm1s = dm1s_ref
        operator.casdm2fr = [dm2_ref[None]]

        tdm1rs, tcm2 = operator.make_tdm1s2c_sub([[c1]])

        def make_cumulant(c):
            dm1s = np.asarray(box.solver.make_rdm1s(c, norb, nelec))
            dm2 = np.asarray(box.solver.make_rdm12(c, norb, nelec)[1])
            dm1 = dm1s.sum(axis=0)
            cumulant = dm2 - np.multiply.outer(dm1, dm1)
            for spin in range(2):
                cumulant += np.multiply.outer(
                    dm1s[spin], dm1s[spin],
                ).transpose(0, 3, 2, 1)
            return dm1s, cumulant

        step = 1e-5
        c_plus = c0 + step * c1
        c_plus /= np.linalg.norm(c_plus)
        c_minus = c0 - step * c1
        c_minus /= np.linalg.norm(c_minus)
        dm1s_plus, cumulant_plus = make_cumulant(c_plus)
        dm1s_minus, cumulant_minus = make_cumulant(c_minus)
        dm1s_derivative = (dm1s_plus - dm1s_minus) / (2.0 * step)
        cumulant_derivative = (
            cumulant_plus - cumulant_minus
        ) / (2.0 * step)

        np.testing.assert_allclose(
            tdm1rs[0], dm1s_derivative, atol=2e-9, rtol=2e-9,
        )
        np.testing.assert_allclose(
            tcm2, cumulant_derivative, atol=2e-9, rtol=2e-9,
        )

    def test_transition_dm1s_transforms_to_active_block_mos(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        operator.nroots = 2
        operator.weights = np.array([0.3, 0.7])
        operator.nkpts = 2
        operator.ncas = 2
        operator.ncastot = 4
        operator.ncore = 1
        operator.nocc = 3
        operator.nmo = 5
        operator.mo_phase = np.zeros((2, 2, 4), dtype=np.complex128)
        fourier = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
        for k in range(2):
            for cell in range(2):
                for band in range(2):
                    operator.mo_phase[k, band, 2 * cell + band] = (
                        fourier[k, cell]
                    )

        rng = np.random.default_rng(113)
        tdm1rs = (
            rng.standard_normal((2, 2, 4, 4))
            + 1j * rng.standard_normal((2, 2, 4, 4))
        )
        tdm1rs += tdm1rs.swapaxes(-1, -2).conj()

        actual = operator._transition_dm1s_to_block(tdm1rs)

        averaged = np.einsum(
            "r,rspq->spq", operator.weights, tdm1rs, optimize=True,
        )
        expected_active = np.asarray([
            [
                operator.mo_phase[k] @ averaged[spin]
                @ operator.mo_phase[k].conj().T
                for k in range(operator.nkpts)
            ]
            for spin in range(2)
        ])
        expected = np.zeros((2, 2, 5, 5), dtype=np.complex128)
        expected[:, :, 1:3, 1:3] = expected_active

        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(
            actual, actual.conj().transpose(0, 1, 3, 2),
        )
        np.testing.assert_allclose(actual[:, :, :1], 0.0)
        np.testing.assert_allclose(actual[:, :, 3:], 0.0)
        np.testing.assert_allclose(actual[:, :, :, :1], 0.0)
        np.testing.assert_allclose(actual[:, :, :, 3:], 0.0)

    def test_transition_cumulant_uses_bra_ket_momentum_blocks(self):
        operator = KLASSCF_HessianOperator.__new__(
            KLASSCF_HessianOperator
        )
        nkpts = 3
        operator.nkpts = nkpts
        operator.ncas = 1
        operator.ncastot = nkpts
        operator.ncore = 1
        operator.nocc = 2
        operator.nmo = 3
        operator.kpts = np.zeros((nkpts, 3))
        operator.las = type("LAS", (), {
            "_scf": type("SCF", (), {"cell": object()})(),
        })()
        phase = np.exp(
            2j * np.pi * np.arange(nkpts)[:, None]
            * np.arange(nkpts)[None, :] / nkpts
        ) / np.sqrt(nkpts)
        operator.mo_phase = phase[:, None, :]
        operator.eri_cas = np.zeros((nkpts,) * 4, dtype=np.complex128)
        rng = np.random.default_rng(127)
        tcm2 = (
            rng.standard_normal((nkpts,) * 4)
            + 1j * rng.standard_normal((nkpts,) * 4)
        )
        paaa = {
            key: (
                rng.standard_normal((operator.nmo, 1, 1, 1))
                + 1j * rng.standard_normal((operator.nmo, 1, 1, 1))
            )
            for key in np.ndindex((nkpts,) * 3)
        }
        calls = []

        def get_paaa(k1, k2, k3):
            calls.append((k1, k2, k3))
            return paaa[k1, k2, k3]

        operator.eri_paaa = get_paaa
        kconserv = np.fromfunction(
            lambda k1, k2, k3: (k1 - k2 + k3) % nkpts,
            (nkpts,) * 3, dtype=int,
        ).astype(int)

        expected = np.zeros(
            (nkpts, operator.nmo, operator.nmo), dtype=np.complex128,
        )
        for k1, k2, k3 in np.ndindex((nkpts,) * 3):
            k4 = kconserv[k1, k2, k3]
            transformed = np.einsum(
                "iP,jQ,PQRS,kR,lS->ijkl",
                operator.mo_phase[k1].conj(),
                operator.mo_phase[k2],
                tcm2,
                operator.mo_phase[k3].conj(),
                operator.mo_phase[k4],
                optimize=True,
            )
            expected[k1, :, 1:2] += np.tensordot(
                paaa[k1, k2, k3], transformed,
                axes=((1, 2, 3), (1, 2, 3)),
            )

        with patch.object(
                klasscf.kpts_helper, "get_kconserv",
                return_value=kconserv):
            actual = operator._transition_cumulant_to_block_fock(tcm2)

        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(actual[:, :, :1], 0.0)
        np.testing.assert_allclose(actual[:, :, 2:], 0.0)
        self.assertCountEqual(calls, list(np.ndindex((nkpts,) * 3)))


class KnownValuesKLASSCFTransSymmHessianOperator(unittest.TestCase):

    def test_is_child_of_general_k_lasscf_hessian(self):
        self.assertTrue(issubclass(
            KLASSCF_TransSymmHessianOperator,
            KLASSCF_HessianOperator,
        ))

    def test_ci_pack_and_unpack_apply_translation_phases(self):
        phases = np.exp(1j * np.array([0.0, 0.43]))
        operator, ci_ref = _new_tdm_operator(
            KLASSCF_TransSymmHessianOperator, phases=phases,
        )

        packed = operator._pack_ci(operator.ci)
        unpacked = operator._unpack_cif(packed)

        np.testing.assert_allclose(packed[0], ci_ref)
        for phase, ci_cell in zip(phases, unpacked):
            np.testing.assert_allclose(ci_cell[0], phase * ci_ref)

    def test_transition_rdm_uses_only_packed_reference_ci(self):
        phases = np.exp(1j * np.array([0.0, -0.37]))
        plain, _ = _new_tdm_operator(
            KLASSCF_HessianOperator, phases=phases,
        )
        adapted, _ = _new_tdm_operator(
            KLASSCF_TransSymmHessianOperator, phases=phases,
        )
        trial_ref = np.array([[0.3j], [0.4]], dtype=np.complex128)
        ci1 = [[phase * trial_ref] for phase in phases]

        expected = plain.make_tdm1s_sub(ci1)
        actual = adapted.make_tdm1s_sub(ci1)

        np.testing.assert_allclose(actual, expected)
        self.assertEqual(
            [box.collect_calls for box in adapted.fciboxes], [1, 0]
        )

    @unittest.skip("translation-symmetric Hessian dispatch is out of scope")
    def test_matvec_keeps_full_phase_adapted_ci_layout(self):
        phases = np.exp(1j * np.array([0.0, 0.61]))
        operator, _ = _new_tdm_operator(
            KLASSCF_TransSymmHessianOperator, phases=phases,
        )
        operator.level_shift = 0.25
        _set_toy_matvec_pipeline(operator)
        trial_ref = np.array([[0.2 + 0.1j], [-0.3j]])
        trial = np.concatenate([
            (phase * trial_ref).reshape(-1) for phase in phases
        ])

        result = operator._matvec(trial)

        self.assertEqual(result.shape, trial.shape)
        np.testing.assert_allclose(result, 5.25 * trial)
        result_cells = result.reshape(2, -1)
        np.testing.assert_allclose(
            result_cells[1], phases[1] * result_cells[0]
        )


if __name__ == "__main__":
    unittest.main()
