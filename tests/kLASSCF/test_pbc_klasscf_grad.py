
import unittest

import numpy as np
from scipy import linalg

from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.mcscf.klasscf import get_grad, get_grad_orb
from mrh.my_pyscf.pbc.mcscf.productstate import ImpureProductStateFCISolver

"""Tests for k-LASSCF gradients.

Test-0: Pack orbital and CI gradients in order and forward intermediates.
Test-1: Extrapolate a LiH centered orbital finite difference to validate g_orb.
Test-2: Check that the LiH forward orbital Taylor residual has no linear term.
Test-3: Use a supplied unitary-group generator without invoking its factory.
"""

# Author: Bhavnesh Jangid


STEPS = np.asarray([1e-2, 5e-3, 1e-3, 5e-4])
RTOL = 1e-5


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


def _copy_ci(ci):
    return [[np.array(root, copy=True) for root in fragment] for fragment in ci]


def _energy(klas, mo_coeff, ci):
    """Return the fixed-CI k-LAS energy per unit cell."""
    h1eff, ecore = klas.h1e_for_cas(
        mo_coeff=mo_coeff, ncas=klas.ncas, ncore=klas.ncore,
    )
    solver = ImpureProductStateFCISolver(
        [box.fcisolvers[0] for box in klas.fciboxes],
        lweights=[[1.0] for _ in klas.fciboxes], stdout=klas.stdout,
        verbose=lib.logger.QUIET,
    )
    energy = solver.energy_elec(
        h1eff, klas.get_h2cas(mo_coeff), [roots[0] for roots in ci],
        klas.ncas_sub, klas.nelecas_sub, ecore=ecore,
    ) / klas.nkpts
    np.testing.assert_allclose(energy.imag, 0.0, atol=1e-9)
    return energy.real


def _rotate_mos(mo_coeff, kappa, step):
    return np.asarray([
        coeff @ linalg.expm(step * rotation)
        for coeff, rotation in zip(mo_coeff, kappa)
    ])


def _orbital_direction(klas, seed):
    """Return a complex unit vector spanning all nonredundant rotations."""
    ncore, ncas = klas.ncore, klas.ncas
    nmo = klas.mo_coeff.shape[-1]
    spaces = (
        (slice(0, ncore), slice(ncore, ncore + ncas)),
        (slice(0, ncore), slice(ncore + ncas, nmo)),
        (slice(ncore, ncore + ncas), slice(ncore + ncas, nmo)),
    )
    rng = np.random.default_rng(seed)
    kappa = np.zeros((klas.nkpts, nmo, nmo), dtype=np.complex128)
    for kappa_k in kappa:
        for left, right in spaces:
            block = rng.standard_normal((right.stop - right.start,
                                         left.stop - left.start))
            block = block + 1j * rng.standard_normal(block.shape)
            kappa_k[right, left] = block
            kappa_k[left, right] = -block.conj().T
    return kappa / np.linalg.norm(kappa)


def _build_lih():
    """Build the smallest complex-k LiH k-LASCI reference."""
    cell = gto.Cell()
    cell.a = np.diag((4.0, 10.0, 10.0))
    cell.atom = "Li 0 0 0; H 1.6 0 0"
    cell.basis = "sto-3g"
    cell.unit = "Angstrom"
    cell.precision, cell.ke_cutoff = 1e-12, 20
    cell.verbose = lib.logger.QUIET
    cell.build()
    kmesh = (2, 1, 1)
    kmf = scf.KRHF(cell, kpts=cell.make_kpts(kmesh, wrap_around=True)).density_fit()
    kmf.exxdiv, kmf.max_cycle = None, 0
    kmf.kernel()
    mo_coeff = avas.kernel(kmf, ["Li 2s", "H 1s"], minao=cell.basis)[2]
    klas = mcscf.KLASCI(kmf, 2, (1, 1), kmesh=kmesh)
    mo_coeff = np.asarray(
        klas.localize_init_guess(["Li 2s", "H 1s"], mo_coeff=mo_coeff),
    )
    klas.kernel(mo_coeff)
    return klas, mo_coeff, _copy_ci(klas.ci)


class GradientTests(unittest.TestCase):

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


class LiHOrbitalGradientTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.klas, cls.mo_coeff, cls.ci = _build_lih()

    def test_orbital_gradient_centered_difference(self):
        # Main step: centered differences cancel even Taylor terms, so the
        # h**2 -> 0 intercept must equal the analytic orbital derivative.
        kappa = _orbital_direction(self.klas, seed=17)
        gorb = get_grad_orb(self.klas, mo_coeff_kpts=self.mo_coeff, ci=self.ci)
        analytic = np.real(np.vdot(gorb, kappa)) / self.klas.nkpts
        differences = [
            (_energy(self.klas, _rotate_mos(self.mo_coeff, kappa, step), self.ci)
             - _energy(self.klas, _rotate_mos(self.mo_coeff, kappa, -step), self.ci))
            / (2.0 * step)
            for step in STEPS
        ]
        extrapolated = np.polynomial.polynomial.polyfit(STEPS ** 2, differences, 1)[0]
        self.assertLess(
            abs(extrapolated - analytic) / max(abs(analytic), 1e-10), RTOL,
            msg=f"analytic={analytic:.12e}, extrapolated={extrapolated:.12e}",
        )

    def test_orbital_gradient_taylor_residual(self):
        # Main step: after subtracting h*g.kappa, R(h) / h must extrapolate
        # to zero because the remaining energy error is quadratic in h.
        kappa = _orbital_direction(self.klas, seed=23)
        gorb = get_grad_orb(self.klas, mo_coeff_kpts=self.mo_coeff, ci=self.ci)
        analytic = np.real(np.vdot(gorb, kappa)) / self.klas.nkpts
        energy_zero = _energy(self.klas, self.mo_coeff, self.ci)
        residual_per_step = np.asarray([
            (_energy(self.klas, _rotate_mos(self.mo_coeff, kappa, step), self.ci)
             - energy_zero - step * analytic) / step
            for step in STEPS
        ])
        coefficients = np.polynomial.polynomial.polyfit(STEPS, residual_per_step, 2)
        scale = max(abs(analytic), abs(coefficients[1]) * max(STEPS), 1e-10)
        self.assertLess(abs(coefficients[0]) / scale, RTOL)


if __name__ == "__main__":
    unittest.main()
