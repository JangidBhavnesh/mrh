import unittest

import numpy as np
from pyscf import lib

from mrh.my_pyscf.pbc.mcscf import mc_ao2mo
from mrh.my_pyscf.pbc.mcscf.klasscf import (
    ActiveActiveRotationMap,
    KLASSCF_UnitaryGroupGenerators,
    get_grad,
    get_grad_ci,
    get_grad_orb,
    get_ugg,
)
from mrh.my_pyscf.pbc.mcscf.klas_ao2mo import _ERIS
from mrh.my_pyscf.pbc.mcscf.klasci import (
    PBCLASCINoSymm,
    PBCLASCITransSymm,
)


def _fourier_mo_phase(nkpts, ncas):
    """Return a band-preserving complex k-to-Wannier unitary."""
    phase = np.zeros(
        (nkpts, ncas, nkpts * ncas), dtype=np.complex128,
    )
    fourier = np.exp(
        2j * np.pi * np.arange(nkpts)[:, None]
        * np.arange(nkpts)[None, :] / nkpts
    ) / np.sqrt(nkpts)
    for k in range(nkpts):
        for cell in range(nkpts):
            i = cell * ncas
            phase[k, :, i:i + ncas] = fourier[k, cell] * np.eye(ncas)
    return phase


class KnownValues(unittest.TestCase):

    def test_inherits_kcasscf_eris(self):
        self.assertTrue(issubclass(_ERIS, mc_ao2mo._ERIS))

    def test_gradient_and_eris_are_registered(self):
        for cls in (PBCLASCINoSymm, PBCLASCITransSymm):
            self.assertIs(cls.get_grad_orb, get_grad_orb)
            self.assertIs(cls._klasscf_eris, _ERIS)
            self.assertIs(cls._ugg, KLASSCF_UnitaryGroupGenerators)
            self.assertIs(cls.get_ugg, get_ugg)
            self.assertIs(cls.get_grad_ci, get_grad_ci)
            self.assertIs(cls.get_grad, get_grad)

    def test_paaa_is_active_slice_of_ppaa(self):
        eris = _ERIS.__new__(_ERIS)
        eris.ncore = 1
        eris.nocc = 3
        ppaa = np.arange(5 * 5 * 2 * 2).reshape(5, 5, 2, 2)
        eris.get_ppaa = lambda k1, k2, k3: ppaa

        paaa = eris.get_paaa(0, 1, 2)

        self.assertEqual(paaa.shape, (5, 2, 2, 2))
        np.testing.assert_array_equal(paaa, ppaa[:, 1:3])
        self.assertTrue(np.shares_memory(paaa, ppaa))

    def test_ppaa_papa_paap_and_paaa_are_read_from_disk(self):
        eris = _ERIS.__new__(_ERIS)
        eris.ncore = 1
        eris.nocc = 3
        eris.ppaa_kpts = None
        eris.papa_kpts = None
        eris.paap_kpts = None
        eris.erifile = lib.H5TmpFile()
        eris.erifile.require_group("ppaa")
        eris.erifile.require_group("papa")
        eris.erifile.require_group("paap")
        ppaa = np.arange(5 * 5 * 2 * 2).reshape(5, 5, 2, 2)
        papa = 100 + np.arange(5 * 2 * 5 * 2).reshape(5, 2, 5, 2)
        paap = 200 + np.arange(5 * 2 * 2 * 5).reshape(5, 2, 2, 5)
        eris.erifile["ppaa/0_1_0"] = ppaa
        eris.erifile["papa/0_1_0"] = papa
        eris.erifile["paap/0_1_0"] = paap

        np.testing.assert_array_equal(eris.get_ppaa(0, 1, 0), ppaa)
        np.testing.assert_array_equal(eris.get_papa(0, 1, 0), papa)
        np.testing.assert_array_equal(eris.get_paap(0, 1, 0), paap)
        np.testing.assert_array_equal(eris.get_paaa(0, 1, 0), ppaa[:, 1:3])

    def test_complex_orbital_and_ci_pack_unpack(self):
        klas = _FakeKLAS()
        ugg = KLASSCF_UnitaryGroupGenerators(klas)

        x_orb = (
            np.arange(ugg.nvar_orb, dtype=float)
            + 1j * np.arange(ugg.nvar_orb, dtype=float)[::-1]
        ) / 7.0
        kappa = ugg.unpack_orb(x_orb)
        np.testing.assert_allclose(ugg.pack_orb(kappa), x_orb)
        np.testing.assert_allclose(
            kappa + kappa.conj().transpose(0, 2, 1), 0.0,
        )

        ci = [
            [np.array([0.2 + 0.3j, -0.4j])],
            [np.array([-0.1 + 0.5j, 0.7])],
        ]
        packed = ugg.pack(kappa, ci)
        kappa_out, ci_out = ugg.unpack(packed)

        self.assertEqual(packed.shape, (ugg.nvar_tot,))
        self.assertTrue(np.issubdtype(packed.dtype, np.complexfloating))
        np.testing.assert_allclose(kappa_out, kappa)
        for actual_r, expected_r in zip(ci_out, ci):
            np.testing.assert_allclose(actual_r[0], expected_r[0])

    def test_active_active_wannier_block_matrix_map(self):
        nkpts = 3
        ncas = 2
        rotation_map = ActiveActiveRotationMap(
            _fourier_mo_phase(nkpts, ncas),
            np.full(nkpts, ncas),
        )
        rng = np.random.default_rng(12)
        lower = (
            rng.standard_normal((nkpts, ncas, ncas))
            + 1j * rng.standard_normal((nkpts, ncas, ncas))
        )
        lower = np.tril(lower, -1)
        diagonal = 1j * rng.standard_normal((nkpts, ncas))
        block_rotation = lower - lower.conj().transpose(0, 2, 1)
        block_rotation += np.asarray([
            np.diag(values) for values in diagonal
        ])

        wannier_rotation = rotation_map.block_to_wannier(block_rotation)
        block_round_trip = rotation_map.wannier_to_block(wannier_rotation)

        np.testing.assert_allclose(block_round_trip, block_rotation)
        np.testing.assert_allclose(
            wannier_rotation + wannier_rotation.conj().T, 0.0,
            atol=1e-13,
        )

        arbitrary_wannier = (
            rng.standard_normal((nkpts * ncas,) * 2)
            + 1j * rng.standard_normal((nkpts * ncas,) * 2)
        )
        projected = rotation_map.block_to_wannier(
            rotation_map.wannier_to_block(arbitrary_wannier)
        )
        projected_twice = rotation_map.block_to_wannier(
            rotation_map.wannier_to_block(projected)
        )
        np.testing.assert_allclose(projected_twice, projected)

    def test_active_active_pair_projection_removes_redundancy(self):
        rotation_map = ActiveActiveRotationMap(
            _fourier_mo_phase(2, 2), np.array([2, 2]),
        )

        self.assertEqual(rotation_map.pair_map.shape, (2, 4))
        self.assertEqual(rotation_map.nvar, 1)
        expected_projector = np.array([
            [0.5, -0.5],
            [-0.5, 0.5],
        ])
        np.testing.assert_allclose(
            rotation_map.basis @ rotation_map.basis.conj().T,
            expected_projector,
            atol=1e-13,
        )

        coordinates = np.array([0.4 - 0.3j])
        block_rotation = rotation_map.unpack(coordinates)
        np.testing.assert_allclose(
            rotation_map.pack(block_rotation), coordinates,
        )
        np.testing.assert_allclose(
            block_rotation[0, 1, 0] + block_rotation[1, 1, 0], 0.0,
            atol=1e-13,
        )

        arbitrary = np.zeros((2, 2, 2), dtype=np.complex128)
        arbitrary[:, 1, 0] = [0.2 + 0.1j, -0.4 + 0.3j]
        arbitrary -= arbitrary.conj().transpose(0, 2, 1)
        projected = rotation_map.unpack(rotation_map.pack(arbitrary))
        lower_pairs = arbitrary[rotation_map.block_pair_idx]
        expected_pairs = (
            rotation_map.basis @ rotation_map.basis.conj().T @ lower_pairs
        )
        np.testing.assert_allclose(
            projected[rotation_map.block_pair_idx], expected_pairs,
        )

    def test_ugg_includes_projected_active_active_coordinates(self):
        klas = _FakeKLASActive()
        ugg = KLASSCF_UnitaryGroupGenerators(klas)

        self.assertEqual(ugg.nvar_orb_external, 10)
        self.assertEqual(ugg.nvar_orb_active_active, 1)
        self.assertEqual(ugg.nvar_orb, 11)

        x_orb = np.linspace(0.1, 1.1, ugg.nvar_orb).astype(complex)
        x_orb += 1j * np.linspace(-0.7, 0.4, ugg.nvar_orb)
        kappa = ugg.unpack_orb(x_orb)

        np.testing.assert_allclose(ugg.pack_orb(kappa), x_orb)
        np.testing.assert_allclose(
            kappa + kappa.conj().transpose(0, 2, 1), 0.0,
            atol=1e-13,
        )
        active = slice(klas.ncore, klas.ncore + klas.ncas)
        active_rotation = kappa[:, active, active]
        np.testing.assert_allclose(
            active_rotation[0, 1, 0] + active_rotation[1, 1, 0],
            0.0,
            atol=1e-13,
        )
        wannier_rotation = ugg.active_active_map.block_to_wannier(
            active_rotation,
        )
        np.testing.assert_allclose(
            wannier_rotation[:2, :2], 0.0, atol=1e-13,
        )
        np.testing.assert_allclose(
            wannier_rotation[2:, 2:], 0.0, atol=1e-13,
        )

    def test_total_gradient_places_orbitals_before_ci(self):
        klas = _FakeKLAS()
        ugg = KLASSCF_UnitaryGroupGenerators(klas)
        x_orb = np.linspace(0.1, 0.5, ugg.nvar_orb).astype(complex)
        x_orb += 1j * np.linspace(-0.2, 0.3, ugg.nvar_orb)
        gorb = ugg.unpack_orb(x_orb)
        gci = [
            [np.array([0.2 + 0.1j, -0.3j])],
            [np.array([0.4j, -0.5 + 0.2j])],
        ]
        klas.get_grad_orb = lambda **kwargs: gorb
        klas.get_grad_ci = lambda **kwargs: gci

        gradient = get_grad(klas, ugg=ugg)

        np.testing.assert_allclose(gradient[:ugg.nvar_orb], x_orb)
        np.testing.assert_allclose(gradient[ugg.nvar_orb:], ugg.pack_ci(gci))
        self.assertTrue(np.issubdtype(gradient.dtype, np.complexfloating))


class _IdentityTransformer:
    ncsf = 2
    ndet = 2

    @staticmethod
    def vec_det2csf(vector, order="C", normalize=False):
        return np.array(vector, copy=True)

    @staticmethod
    def vec_csf2det(vector, order="C", normalize=False):
        return np.array(vector, copy=True)


class _FakeSolver:
    def __init__(self):
        self.transformer = _IdentityTransformer()

    def check_transformer_cache(self):
        pass


class _FakeFCIBox:
    def __init__(self):
        self.fcisolvers = [_FakeSolver()]

    @staticmethod
    def _get_nelec(solver, nelec):
        return tuple(nelec)


class _FakeKLAS:
    def __init__(self):
        self.kpts = np.zeros((2, 3))
        self.mo_coeff = np.zeros((2, 3, 4), dtype=np.complex128)
        self.ncore = 1
        self.ncas = 1
        self.frozen = None
        self.frozen_ci = None
        self.ncas_sub = np.array([1, 1])
        self.nelecas_sub = np.array([(1, 0), (1, 0)])
        self.fciboxes = [_FakeFCIBox(), _FakeFCIBox()]
        self.ci = [
            [np.array([1.0, 0.0], dtype=np.complex128)],
            [np.array([1.0, 0.0], dtype=np.complex128)],
        ]
        self.mo_phase = _fourier_mo_phase(2, 1)

    @staticmethod
    def uniq_var_indices(nmo, ncore, ncas, frozen):
        nocc = ncore + ncas
        idx = np.zeros((nmo, nmo), dtype=bool)
        idx[ncore:, :ncore] = True
        idx[nocc:, ncore:nocc] = True
        return idx


class _FakeKLASActive(_FakeKLAS):
    def __init__(self):
        super().__init__()
        self.ncas = 2
        self.ncas_sub = np.array([2, 2])
        self.mo_phase = _fourier_mo_phase(2, 2)


if __name__ == "__main__":
    unittest.main()
