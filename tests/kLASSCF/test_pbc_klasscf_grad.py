import unittest

import numpy as np

from mrh.my_pyscf.pbc.mcscf.klasscf import get_grad
from mrh.my_pyscf.pbc.mcscf.klasci import (
    PBCLASCINoSymm,
    PBCLASCITransSymm,
)


class _PackingUGG:
    def __init__(self):
        self.inputs = None

    def pack(self, gorb, gci):
        self.inputs = (gorb, gci)
        return np.concatenate((gorb, gci)).astype(np.complex128)


class _FakeKLAS:
    def __init__(self):
        self.mo_coeff = object()
        self.ci = object()
        self.ugg = _PackingUGG()
        self.calls = []
        self.gorb = np.array([0.2 + 0.1j, -0.3j])
        self.gci = np.array([-0.4 + 0.2j, 0.5j, 0.6])

    def get_ugg(self, **kwargs):
        self.calls.append(("get_ugg", kwargs))
        return self.ugg

    def get_grad_orb(self, **kwargs):
        self.calls.append(("get_grad_orb", kwargs))
        return self.gorb

    def get_grad_ci(self, **kwargs):
        self.calls.append(("get_grad_ci", kwargs))
        return self.gci


class KnownValues(unittest.TestCase):

    def test_is_registered(self):
        for cls in (PBCLASCINoSymm, PBCLASCITransSymm):
            self.assertIs(cls.get_grad, get_grad)

    def test_orbitals_precede_ci_and_defaults_are_forwarded(self):
        klas = _FakeKLAS()
        h2eff_sub = object()
        veff_kpts = object()
        dm1s_kpts = object()
        casdm1frs = object()
        h1eff = object()
        h2eff = object()

        gradient = get_grad(
            klas, h2eff_sub=h2eff_sub, veff_kpts=veff_kpts,
            dm1s_kpts=dm1s_kpts, casdm1frs=casdm1frs,
            h1eff=h1eff, h2eff=h2eff,
        )

        expected_orb = np.array([0.2 + 0.1j, -0.3j])
        expected_ci = np.array([-0.4 + 0.2j, 0.5j, 0.6])
        np.testing.assert_allclose(gradient[:2], expected_orb)
        np.testing.assert_allclose(gradient[2:], expected_ci)
        self.assertTrue(np.issubdtype(gradient.dtype, np.complexfloating))
        self.assertIs(klas.ugg.inputs[0], klas.gorb)
        self.assertIs(klas.ugg.inputs[1], klas.gci)

        self.assertEqual(
            [name for name, kwargs in klas.calls],
            ["get_ugg", "get_grad_orb", "get_grad_ci"],
        )
        ugg_kwargs = klas.calls[0][1]
        self.assertIs(ugg_kwargs["mo_coeff"], klas.mo_coeff)
        self.assertIs(ugg_kwargs["ci"], klas.ci)

        orb_kwargs = klas.calls[1][1]
        self.assertIs(orb_kwargs["mo_coeff_kpts"], klas.mo_coeff)
        self.assertIs(orb_kwargs["ci"], klas.ci)
        self.assertIs(orb_kwargs["h2eff_sub"], h2eff_sub)
        self.assertIs(orb_kwargs["veff_kpts"], veff_kpts)
        self.assertIs(orb_kwargs["dm1s_kpts"], dm1s_kpts)

        ci_kwargs = klas.calls[2][1]
        self.assertIs(ci_kwargs["mo_coeff"], klas.mo_coeff)
        self.assertIs(ci_kwargs["ci"], klas.ci)
        self.assertIs(ci_kwargs["ugg"], klas.ugg)
        self.assertIs(ci_kwargs["casdm1frs"], casdm1frs)
        self.assertIs(ci_kwargs["h1eff"], h1eff)
        self.assertIs(ci_kwargs["h2eff"], h2eff)

    def test_supplied_ugg_skips_factory(self):
        klas = _FakeKLAS()
        supplied_ugg = _PackingUGG()

        get_grad(
            klas, mo_coeff=klas.mo_coeff, ci=klas.ci,
            ugg=supplied_ugg,
        )

        self.assertNotIn("get_ugg", [name for name, kwargs in klas.calls])
        self.assertIs(supplied_ugg.inputs[0], klas.gorb)
        self.assertIs(supplied_ugg.inputs[1], klas.gci)


if __name__ == "__main__":
    unittest.main()
