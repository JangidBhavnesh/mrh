import unittest

import numpy as np

from mrh.my_pyscf.pbc.mcscf import mc_ao2mo
from mrh.my_pyscf.pbc.mcscf.klasscf import (
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

    @staticmethod
    def uniq_var_indices(nmo, ncore, ncas, frozen):
        nocc = ncore + ncas
        idx = np.zeros((nmo, nmo), dtype=bool)
        idx[ncore:, :ncore] = True
        idx[nocc:, ncore:nocc] = True
        return idx


if __name__ == "__main__":
    unittest.main()
