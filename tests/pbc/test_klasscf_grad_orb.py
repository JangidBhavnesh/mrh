import unittest
from unittest.mock import patch

import numpy as np
from scipy import linalg

from pyscf import lib
from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas, klasscf
from mrh.my_pyscf.pbc.mcscf.klasscf import (
    KLASSCF_HessianOperator,
    get_grad_orb,  # noqa: F401
)
from mrh.my_pyscf.pbc.mcscf.productstate import (
    ImpureProductStateFCISolver,
)


FD_STEPS = np.asarray([1e-2, 5e-3, 2.5e-3, 1.25e-3])
ZERO_STEP_RTOL = 1e-5
HOP_ZERO_STEP_RTOL = 1e-8


def build_reference(lattice, kmesh):
    cell = gto.Cell()
    cell.a = np.diag(lattice)
    cell.atom = "Li 0 0 0; H 1.6 0 0"
    cell.basis = "sto-3g"
    cell.unit = "Angstrom"
    cell.precision = 1e-12
    cell.ke_cutoff = 20
    cell.verbose = lib.logger.QUIET
    cell.build()

    kpts = cell.make_kpts(kmesh, wrap_around=True)
    kmf = scf.KRHF(cell, kpts=kpts).density_fit()
    kmf.exxdiv = None
    kmf.max_cycle = 0
    kmf.kernel()

    active_labels = ["Li 2s", "H 1s"]
    mo_coeff = avas.kernel(kmf, active_labels, minao=cell.basis)[2]
    klas = mcscf.KLASCI(kmf, 2, (1, 1), kmesh=kmesh)
    klas.conv_tol_grad = 1e-8
    klas.conv_tol_self = 1e-10
    mo_ref = klas.localize_init_guess(active_labels, mo_coeff=mo_coeff)
    klas.kernel(mo_ref)
    return klas, np.asarray(mo_ref)


def copy_ci(ci):
    return [[np.array(c, copy=True) for c in roots] for roots in ci]


def make_direction(klas, rotation_blocks, seed):
    nkpts, nmo = klas.nkpts, klas.mo_coeff.shape[-1]
    ncore = klas.ncore
    nocc = ncore + klas.ncas
    spaces = {
        "core": slice(0, ncore),
        "active": slice(ncore, nocc),
        "virtual": slice(nocc, nmo),
    }
    dimensions = {
        "core": ncore,
        "active": klas.ncas,
        "virtual": nmo - nocc,
    }

    rng = np.random.default_rng(seed)
    kappa = np.zeros((nkpts, nmo, nmo), dtype=np.complex128)
    for k in range(nkpts):
        for block_name in rotation_blocks:
            left_name, right_name = block_name.split("-")
            nleft = dimensions[left_name]
            nright = dimensions[right_name]
            if nleft == 0 or nright == 0:
                raise RuntimeError(f"empty orbital space in {block_name}")
            block = (
                rng.standard_normal((nright, nleft))
                + 1j * rng.standard_normal((nright, nleft))
            )
            left = spaces[left_name]
            right = spaces[right_name]
            kappa[k, right, left] = block
            kappa[k, left, right] = -block.conj().T

    kappa /= np.linalg.norm(kappa)
    return kappa


def rotate_mos(mo_ref, kappa, step):
    return np.asarray([
        mo_k @ linalg.expm(step * kappa_k)
        for mo_k, kappa_k in zip(mo_ref, kappa)
    ])


def fixed_ci_energy(klas, mo_coeff, ci):
    h1eff, ecore = klas.h1e_for_cas(
        mo_coeff=mo_coeff, ncas=klas.ncas, ncore=klas.ncore,
    )
    h2eff = klas.get_h2cas(mo_coeff)
    fcisolvers = [box.fcisolvers[0] for box in klas.fciboxes]
    solver = ImpureProductStateFCISolver(
        fcisolvers,
        lweights=[[1.0] for _ in fcisolvers],
        stdout=klas.stdout,
        verbose=lib.logger.QUIET,
    )
    energy = solver.energy_elec(
        h1eff, h2eff, [roots[0] for roots in ci],
        klas.ncas_sub, klas.nelecas_sub, ecore=ecore,
    ) / klas.nkpts
    np.testing.assert_allclose(np.imag(energy), 0.0, atol=1e-9)
    return float(np.real(energy))


class KnownValues(unittest.TestCase):

    def assert_gradient_matches_finite_difference(
            self, lattice, kmesh, rotation_blocks, seed):
        klas, mo_ref = build_reference(lattice, kmesh)
        ci_ref = copy_ci(klas.ci)
        kappa = make_direction(klas, rotation_blocks, seed)
        gorb = klas.get_grad_orb(mo_coeff_kpts=mo_ref, ci=ci_ref)

        np.testing.assert_allclose(
            kappa + kappa.conj().transpose(0, 2, 1), 0.0, atol=1e-13,
        )
        np.testing.assert_allclose(
            gorb + gorb.conj().transpose(0, 2, 1), 0.0, atol=1e-9,
        )
        self.assertGreater(np.linalg.norm(kappa.imag), 1e-8)

        # Full anti-Hermitian matrices contain both halves of every complex
        # rotation, so this equals the usual 2/nkpts packed contraction.
        analytic = np.real(np.vdot(gorb, kappa)) / klas.nkpts
        finite_differences = []
        for step in FD_STEPS:
            energy_plus = fixed_ci_energy(
                klas, rotate_mos(mo_ref, kappa, step), ci_ref,
            )
            energy_minus = fixed_ci_energy(
                klas, rotate_mos(mo_ref, kappa, -step), ci_ref,
            )
            finite_differences.append(
                (energy_plus - energy_minus) / (2.0 * step)
            )

        zero_step = np.polynomial.polynomial.polyfit(
            FD_STEPS ** 2, finite_differences, 1,
        )[0]
        relative_error = abs(zero_step - analytic) / max(abs(analytic), 1e-14)
        self.assertLess(relative_error, ZERO_STEP_RTOL)

    def test_1d_core_active(self):
        self.assert_gradient_matches_finite_difference(
            (4.0, 10.0, 10.0), (2, 1, 1), ("core-active",), 17,
        )

    def test_2d_active_virtual(self):
        self.assert_gradient_matches_finite_difference(
            (4.0, 4.0, 10.0), (2, 2, 1), ("active-virtual",), 23,
        )

    def test_3d_all_nonredundant_blocks(self):
        self.assert_gradient_matches_finite_difference(
            (4.0, 4.0, 4.0), (2, 2, 2),
            ("core-active", "core-virtual", "active-virtual"), 31,
        )

    def assert_orbital_hop_matches_linear_extrapolation(
            self, lattice, kmesh, rotation_blocks, seed):
        klas, mo_ref = build_reference(lattice, kmesh)
        ci_ref = copy_ci(klas.ci)
        kappa = make_direction(
            klas, rotation_blocks, seed,
        )
        ugg = klas.get_ugg(mo_coeff=mo_ref, ci=ci_ref)
        hop = KLASSCF_HessianOperator(
            klas, ugg, mo_coeff=mo_ref, ci=ci_ref,
        )
        hop.level_shift = 0.0

        trial = np.zeros(ugg.nvar_tot, dtype=np.complex128)
        trial[:ugg.nvar_orb] = ugg.pack_orb(kappa)
        hop_action = hop._matvec(trial)
        self.assertTrue(np.all(np.isfinite(hop_action[ugg.nvar_orb:])))

        # _matvec follows the molecular operator and packs kappa2/2.  Undo
        # that packing factor before comparing with the derivative of the
        # packed anti-Hermitian orbital gradient.
        analytic = 2.0 * hop_action[:ugg.nvar_orb]
        gradient_derivatives = []
        for step in FD_STEPS:
            gradient_plus = ugg.pack_orb(klas.get_grad_orb(
                mo_coeff_kpts=rotate_mos(mo_ref, kappa, step),
                ci=ci_ref,
            ))
            gradient_minus = ugg.pack_orb(klas.get_grad_orb(
                mo_coeff_kpts=rotate_mos(mo_ref, kappa, -step),
                ci=ci_ref,
            ))
            gradient_derivatives.append(
                (gradient_plus - gradient_minus) / (2.0 * step)
            )

        zero_step_derivative = np.polynomial.polynomial.polyfit(
            FD_STEPS ** 2, np.asarray(gradient_derivatives), 1,
        )[0]

        # The molecular orbital Hessian is covariant.  At a nonstationary
        # reference it differs from the direct gradient derivative by this
        # half-commutator connection.
        connection = np.asarray([
            (fock @ kappa_k - kappa_k @ fock) / 2.0
            for fock, kappa_k in zip(hop.fock1, kappa)
        ])
        connection -= connection.conj().transpose(0, 2, 1)
        extrapolated = zero_step_derivative - ugg.pack_orb(connection)

        relative_error = (
            np.linalg.norm(analytic - extrapolated)
            / max(np.linalg.norm(extrapolated), 1e-14)
        )
        self.assertLess(relative_error, HOP_ZERO_STEP_RTOL)

    def test_1d_core_active_orbital_hop(self):
        self.assert_orbital_hop_matches_linear_extrapolation(
            (4.0, 10.0, 10.0), (3, 1, 1), ("core-active",), 41,
        )

    def test_2d_active_virtual_orbital_hop(self):
        self.assert_orbital_hop_matches_linear_extrapolation(
            (4.0, 4.0, 10.0), (2, 2, 1), ("active-virtual",), 43,
        )

    def test_3d_all_nonredundant_blocks_orbital_hop(self):
        self.assert_orbital_hop_matches_linear_extrapolation(
            (4.0, 4.0, 4.0), (2, 2, 2),
            ("core-active", "core-virtual", "active-virtual"), 47,
        )

    def test_1d_projected_active_active_orbital_hop(self):
        klas, mo_ref = build_reference(
            (4.0, 10.0, 10.0), (2, 1, 1),
        )
        ci_ref = copy_ci(klas.ci)
        ugg = klas.get_ugg(mo_coeff=mo_ref, ci=ci_ref)
        self.assertGreater(ugg.nvar_orb_active_active, 0)

        x_orb = np.zeros(ugg.nvar_orb, dtype=np.complex128)
        active_start = ugg.nvar_orb_external
        active_stop = active_start + ugg.nvar_orb_active_active
        rng = np.random.default_rng(53)
        x_orb[active_start:active_stop] = (
            rng.standard_normal(ugg.nvar_orb_active_active)
            + 1j * rng.standard_normal(ugg.nvar_orb_active_active)
        )
        kappa = ugg.unpack_orb(x_orb)
        kappa /= np.linalg.norm(kappa)
        x_orb = ugg.pack_orb(kappa)

        hop = KLASSCF_HessianOperator(
            klas, ugg, mo_coeff=mo_ref, ci=ci_ref,
        )
        hop.level_shift = 0.0
        trial = np.zeros(ugg.nvar_tot, dtype=np.complex128)
        trial[:ugg.nvar_orb] = x_orb
        hop_action = hop._matvec(trial)[:ugg.nvar_orb]
        analytic = 2.0 * hop_action
        hessian, hessian_conj = hop._get_Horb_active_active()
        active_coordinates = x_orb[active_start:active_stop]
        analytic_active = (
            hessian @ active_coordinates
            + hessian_conj @ active_coordinates.conj()
        )

        gradient_derivatives = []
        for step in FD_STEPS:
            gradient_plus = ugg.pack_orb(klas.get_grad_orb(
                mo_coeff_kpts=rotate_mos(mo_ref, kappa, step),
                ci=ci_ref,
            ))
            gradient_minus = ugg.pack_orb(klas.get_grad_orb(
                mo_coeff_kpts=rotate_mos(mo_ref, kappa, -step),
                ci=ci_ref,
            ))
            gradient_derivatives.append(
                (gradient_plus - gradient_minus) / (2.0 * step)
            )

        zero_step_derivative = np.polynomial.polynomial.polyfit(
            FD_STEPS ** 2, np.asarray(gradient_derivatives), 1,
        )[0]
        connection = np.asarray([
            (fock @ kappa_k - kappa_k @ fock) / 2.0
            for fock, kappa_k in zip(hop.fock1, kappa)
        ])
        connection -= connection.conj().transpose(0, 2, 1)
        extrapolated = zero_step_derivative - ugg.pack_orb(connection)
        relative_error = (
            np.linalg.norm(analytic - extrapolated)
            / max(np.linalg.norm(extrapolated), 1e-14)
        )
        self.assertLess(relative_error, HOP_ZERO_STEP_RTOL)
        np.testing.assert_allclose(
            analytic_active,
            hop_action[active_start:active_stop],
            atol=1e-11,
            rtol=1e-11,
        )

    def test_direct_active_external_cross_matches_real_adjoint(self):
        """Check the direct cross action and every two-electron k label."""
        klas, mo_ref = build_reference(
            (4.0, 10.0, 10.0), (3, 1, 1),
        )
        ci_ref = copy_ci(klas.ci)
        ugg = klas.get_ugg(mo_coeff=mo_ref, ci=ci_ref)
        hop = KLASSCF_HessianOperator(
            klas, ugg, mo_coeff=mo_ref, ci=ci_ref,
        )
        hop.level_shift = 0.0
        self.assertEqual(klas.nkpts, 3)

        nvar_external = ugg.nvar_orb_external
        nvar_active = ugg.nvar_orb_active_active
        self.assertGreater(nvar_external, 0)
        self.assertGreater(nvar_active, 0)

        rng = np.random.default_rng(59)
        active_directions = (
            rng.standard_normal(nvar_active),
            1.0j * rng.standard_normal(nvar_active),
            rng.standard_normal(nvar_active)
            + 1.0j * rng.standard_normal(nvar_active),
        )

        # The old implementation explicitly forms both real quadratures of
        # every external column.  It is slow, but is an independent oracle for
        # the new analytic real-adjoint action.
        for coordinates in active_directions:
            direct = hop._apply_Horb_active_external_cross(coordinates)
            reference = (
                hop._apply_Horb_active_external_cross_reference(coordinates)
            )
            np.testing.assert_allclose(
                direct, reference, atol=2e-11, rtol=2e-11,
            )

        # Every direct two-electron contraction must request the original
        # bra-ket-bra-ket block (k1,k2,k3,k4), where k4 is fixed by
        # k1-k2+k3-k4=G.  No regrouped Hessian momentum convention belongs in
        # this action.
        original_transform = klasscf._get_casdm2_kpts
        observed_momenta = []

        def record_transform(casdm2, mo_phase, klabel):
            observed_momenta.append(tuple(int(k) for k in klabel))
            return original_transform(casdm2, mo_phase, klabel)

        kconserv = klasscf.kpts_helper.get_kconserv(
            klas._scf.cell, klas.kpts,
        )
        expected_momenta = [
            (k1, k2, k3, int(kconserv[k1, k2, k3]))
            for k1, k2, k3 in klasscf.kpts_helper.loop_kkk(klas.nkpts)
        ]
        with patch.object(
                klasscf, "_get_casdm2_kpts",
                side_effect=record_transform):
            hop._apply_Horb_active_external_cross(active_directions[-1])
        self.assertEqual(observed_momenta, expected_momenta)

        # Check symmetry directly in the independent real optimizer metric.
        external_coordinates = (
            rng.standard_normal(nvar_external)
            + 1.0j * rng.standard_normal(nvar_external)
        )
        orbital_coordinates = np.zeros(ugg.nvar_orb, dtype=np.complex128)
        orbital_coordinates[:nvar_external] = external_coordinates
        external_kappa = ugg.unpack_orb(orbital_coordinates)
        external_response = hop._orbital_hessian_response_block(
            external_kappa,
        )
        active_response = ugg.pack_orb(
            external_response / 2.0,
        )[nvar_external:nvar_external + nvar_active]
        active_coordinates = active_directions[-1]
        direct_response = hop._apply_Horb_active_external_cross(
            active_coordinates,
        )
        np.testing.assert_allclose(
            np.real(np.vdot(external_coordinates, direct_response)),
            np.real(np.vdot(active_response, active_coordinates)),
            atol=2e-11,
            rtol=2e-11,
        )

    def test_three_kpoint_orbital_ci_hop_matches_gradient_extrapolation(self):
        """Validate Hoc for both quadratures of a complex CI tangent."""
        klas, mo_ref = build_reference(
            (4.0, 10.0, 10.0), (3, 1, 1),
        )
        ci_ref = copy_ci(klas.ci)
        ugg = klas.get_ugg(mo_coeff=mo_ref, ci=ci_ref)
        self.assertEqual(klas.nkpts, 3)
        self.assertGreater(np.linalg.norm(mo_ref.imag), 1e-8)
        self.assertGreater(ugg.nvar_orb_external, 0)
        self.assertGreater(ugg.nvar_orb_active_active, 0)

        # Begin with a real packed direction, transform it to determinant
        # space, and project out each reference CI vector.  The two tested
        # quadratures below are c1 and i*c1.  Complex parallel components are
        # then deliberately restored so that the transition builders and the
        # normalized finite-difference path both exercise overlap removal.
        rng = np.random.default_rng(151)
        packed_seed = rng.standard_normal(ugg.nvar_ci)
        ci_raw = ugg.unpack_ci(packed_seed)
        ci_perp = []
        for raw_r, ref_r in zip(ci_raw, ci_ref):
            perp_r = []
            for raw, c0 in zip(raw_r, ref_r):
                overlap = np.vdot(c0, raw) / np.vdot(c0, c0)
                perpendicular = raw - overlap * c0
                np.testing.assert_allclose(
                    np.vdot(c0, perpendicular), 0.0, atol=1e-13,
                )
                perp_r.append(perpendicular)
            ci_perp.append(perp_r)
        tangent_norm = np.sqrt(sum(
            np.vdot(c1, c1).real
            for c1_r in ci_perp for c1 in c1_r
        ))
        self.assertGreater(tangent_norm, 1e-10)
        ci_perp = [
            [c1 / tangent_norm for c1 in c1_r]
            for c1_r in ci_perp
        ]
        ci_tangent = []
        for ifrag, (perp_r, ref_r) in enumerate(zip(ci_perp, ci_ref)):
            tangent_r = []
            for iroot, (perpendicular, c0) in enumerate(zip(perp_r, ref_r)):
                parallel = (
                    0.17 * (ifrag + 1)
                    + 0.11j * (iroot + 1)
                )
                tangent_r.append(perpendicular + parallel * c0)
            ci_tangent.append(tangent_r)

        hop = KLASSCF_HessianOperator(
            klas, ugg, mo_coeff=mo_ref, ci=ci_ref,
        )
        hop.level_shift = 0.0
        active_start = ugg.nvar_orb_external
        external_pairs = np.argwhere(ugg.uniq_orb_idx)
        self.assertEqual(len(external_pairs), ugg.nvar_orb_external)

        original_transform = klasscf._get_casdm2_kpts
        kconserv = klasscf.kpts_helper.get_kconserv(
            klas._scf.cell, klas.kpts,
        )
        expected_momenta = [
            (k1, k2, k3, int(kconserv[k1, k2, k3]))
            for k1, k2, k3 in klasscf.kpts_helper.loop_kkk(klas.nkpts)
        ]

        def normalized_ci(step, direction):
            result = []
            for ref_r, direction_r in zip(ci_ref, direction):
                result_r = []
                for c0, c1 in zip(ref_r, direction_r):
                    shifted = c0 + step * c1
                    shifted /= np.sqrt(np.vdot(shifted, shifted).real)
                    result_r.append(shifted)
                result.append(result_r)
            return result

        for label, phase in (("real", 1.0), ("imaginary", 1.0j)):
            with self.subTest(ci_direction=label):
                direction = [
                    [phase * c1 for c1 in c1_r]
                    for c1_r in ci_tangent
                ]
                perpendicular_direction = [
                    [phase * c1 for c1 in c1_r]
                    for c1_r in ci_perp
                ]

                # A nonzero reference-parallel component must make no
                # contribution after the overlap subtraction.
                self.assertGreater(max(
                    abs(np.vdot(c1, c0))
                    for c1_r, c0_r in zip(direction, ci_ref)
                    for c1, c0 in zip(c1_r, c0_r)
                ), 1e-3)
                tdm1rs, tcm2 = hop.make_tdm1s2c_sub(direction)
                tdm1rs_perp, tcm2_perp = hop.make_tdm1s2c_sub(
                    perpendicular_direction,
                )
                np.testing.assert_allclose(
                    tdm1rs, tdm1rs_perp, atol=2e-12, rtol=2e-12,
                )
                np.testing.assert_allclose(
                    tcm2, tcm2_perp, atol=2e-12, rtol=2e-12,
                )
                np.testing.assert_allclose(
                    tdm1rs,
                    tdm1rs.conj().transpose(0, 1, 3, 2),
                    atol=2e-12,
                    rtol=2e-12,
                )
                np.testing.assert_allclose(
                    tcm2,
                    tcm2.conj().transpose(1, 0, 3, 2),
                    atol=2e-12,
                    rtol=2e-12,
                )
                np.testing.assert_allclose(
                    tcm2,
                    tcm2.transpose(2, 3, 0, 1),
                    atol=2e-12,
                    rtol=2e-12,
                )

                momentum_calls = []

                def track_transform(cumulant, mo_phase, momenta):
                    momentum_calls.append(tuple(int(k) for k in momenta))
                    return original_transform(cumulant, mo_phase, momenta)

                trial = np.zeros(ugg.nvar_tot, dtype=np.complex128)
                trial[ugg.nvar_orb:] = ugg.pack_ci(direction)
                with patch.object(
                        klasscf, "_get_casdm2_kpts",
                        side_effect=track_transform):
                    analytic = hop._matvec(trial)[:ugg.nvar_orb]

                self.assertCountEqual(momentum_calls, expected_momenta)
                for k1, k2, k3, k4 in momentum_calls:
                    self.assertEqual(k4, kconserv[k1, k2, k3])
                    self.assertEqual((k1 - k2 + k3 - k4) % klas.nkpts, 0)

                gradient_derivatives = []
                for step in FD_STEPS:
                    gradient_plus = ugg.pack_orb(klas.get_grad_orb(
                        mo_coeff_kpts=mo_ref,
                        ci=normalized_ci(step, direction),
                        h2eff_sub=hop.eris,
                    ))
                    gradient_minus = ugg.pack_orb(klas.get_grad_orb(
                        mo_coeff_kpts=mo_ref,
                        ci=normalized_ci(-step, direction),
                        h2eff_sub=hop.eris,
                    ))
                    gradient_derivatives.append(
                        (gradient_plus - gradient_minus) / (2.0 * step)
                    )
                extrapolated = np.polynomial.polynomial.polyfit(
                    FD_STEPS ** 2,
                    np.asarray(gradient_derivatives),
                    1,
                )[0]

                self.assertGreater(
                    np.linalg.norm(analytic[:active_start]), 1e-9,
                )
                self.assertGreater(
                    np.linalg.norm(analytic[active_start:]), 1e-9,
                )
                for index, (kpoint, row, column) in enumerate(external_pairs):
                    with self.subTest(
                            ci_direction=label, orbital="external",
                            kpoint=kpoint, row=row, column=column):
                        np.testing.assert_allclose(
                            analytic[index], extrapolated[index],
                            atol=2e-7, rtol=2e-7,
                        )
                for coordinate in range(ugg.nvar_orb_active_active):
                    index = active_start + coordinate
                    with self.subTest(
                            ci_direction=label,
                            orbital="projected-active-active",
                            coordinate=coordinate):
                        np.testing.assert_allclose(
                            analytic[index], extrapolated[index],
                            atol=2e-7, rtol=2e-7,
                        )

    def test_three_kpoint_ci_orbital_hop_matches_gradient_extrapolation(self):
        """Validate Hco for both quadratures of a complex orbital tangent."""
        klas, mo_ref = build_reference(
            (4.0, 10.0, 10.0), (3, 1, 1),
        )
        ci_ref = copy_ci(klas.ci)
        ugg = klas.get_ugg(mo_coeff=mo_ref, ci=ci_ref)
        hop = KLASSCF_HessianOperator(
            klas, ugg, mo_coeff=mo_ref, ci=ci_ref,
        )
        hop.level_shift = 0.0
        self.assertEqual(klas.nkpts, 3)
        self.assertGreater(np.linalg.norm(mo_ref.imag), 1e-8)
        self.assertGreater(ugg.nvar_orb_external, 0)
        self.assertGreater(ugg.nvar_orb_active_active, 0)
        self.assertGreater(ugg.nvar_ci, 0)

        rng = np.random.default_rng(181)
        packed_seed = rng.standard_normal(ugg.nvar_orb)
        self.assertGreater(
            np.linalg.norm(packed_seed[:ugg.nvar_orb_external]), 1e-8,
        )
        self.assertGreater(
            np.linalg.norm(packed_seed[ugg.nvar_orb_external:]), 1e-8,
        )

        def packed_ci_gradient(mo_coeff):
            h2eff = klas.get_h2cas(mo_coeff)
            dm1s = klas.make_rdm1s(
                mo_coeff=mo_coeff,
                ci=ci_ref,
                casdm1s_sub=hop.casdm1fs,
            )
            veff = klas.get_veff(
                klas._scf.cell, dm_kpts=dm1s,
            )
            h1eff = klas.h1e_for_las(
                mo_coeff=mo_coeff,
                ci=ci_ref,
                ncas_sub=klas.ncas_sub,
                nelecas_sub=klas.nelecas_sub,
                casdm1s_sub=hop.casdm1fs,
                casdm1frs=hop.casdm1frs,
                eri_cas=h2eff,
                veff=veff,
            )
            hc = hop.Hci_all(None, h1eff, h2eff, ci_ref)
            gradient = [
                [
                    2.0 * (hc0 - np.vdot(c0, hc0) * c0)
                    for hc0, c0 in zip(hc_r, ci0_r)
                ]
                for hc_r, ci0_r in zip(hc, ci_ref)
            ]
            return ugg.pack_ci(gradient)

        for label, phase in (("real", 1.0), ("imaginary", 1.0j)):
            with self.subTest(orbital_direction=label):
                packed_orbital = phase * packed_seed
                kappa = ugg.unpack_orb(packed_orbital)
                kappa /= np.linalg.norm(kappa)
                packed_orbital = ugg.pack_orb(kappa)
                self.assertGreater(
                    np.linalg.norm(
                        packed_orbital[:ugg.nvar_orb_external]
                    ), 1e-8,
                )
                self.assertGreater(
                    np.linalg.norm(
                        packed_orbital[ugg.nvar_orb_external:]
                    ), 1e-8,
                )

                h1frs_prime, h2_prime = (
                    hop._orbital_hamiltonian_response(kappa)
                )
                np.testing.assert_allclose(
                    h2_prime,
                    h2_prime.conj().transpose(1, 0, 3, 2),
                    atol=2e-10,
                    rtol=2e-10,
                )
                np.testing.assert_allclose(
                    h2_prime,
                    h2_prime.transpose(2, 3, 0, 1),
                    atol=2e-10,
                    rtol=2e-10,
                )
                for ifrag, h1rs_prime in enumerate(h1frs_prime):
                    with self.subTest(
                            orbital_direction=label, fragment=ifrag):
                        np.testing.assert_allclose(
                            h1rs_prime,
                            h1rs_prime.conj().transpose(0, 1, 3, 2),
                            atol=2e-10,
                            rtol=2e-10,
                        )

                trial = np.zeros(ugg.nvar_tot, dtype=np.complex128)
                trial[:ugg.nvar_orb] = packed_orbital
                analytic = hop._matvec(trial)[ugg.nvar_orb:]

                gradient_derivatives = []
                for step in FD_STEPS:
                    gradient_plus = packed_ci_gradient(
                        rotate_mos(mo_ref, kappa, step),
                    )
                    gradient_minus = packed_ci_gradient(
                        rotate_mos(mo_ref, kappa, -step),
                    )
                    gradient_derivatives.append(
                        (gradient_plus - gradient_minus) / (2.0 * step)
                    )
                extrapolated = np.polynomial.polynomial.polyfit(
                    FD_STEPS ** 2,
                    np.asarray(gradient_derivatives),
                    1,
                )[0]

                self.assertGreater(np.linalg.norm(analytic), 1e-9)
                self.assertEqual(analytic.size, ugg.nvar_ci)
                for coordinate in range(ugg.nvar_ci):
                    with self.subTest(
                            orbital_direction=label,
                            ci_coordinate=coordinate):
                        np.testing.assert_allclose(
                            analytic[coordinate], extrapolated[coordinate],
                            atol=2e-7, rtol=2e-7,
                        )

    def test_1d_complete_hdiag_builds_preconditioner(self):
        klas, mo_ref = build_reference(
            (4.0, 10.0, 10.0), (2, 1, 1),
        )
        ci_ref = copy_ci(klas.ci)
        ugg = klas.get_ugg(mo_coeff=mo_ref, ci=ci_ref)
        hop = KLASSCF_HessianOperator(
            klas, ugg, mo_coeff=mo_ref, ci=ci_ref,
        )

        hdiag = hop._get_Hdiag()

        self.assertEqual(hdiag.shape, (ugg.nvar_tot,))
        self.assertTrue(np.all(np.isfinite(hdiag)))
        self.assertGreater(np.linalg.norm(hdiag[:ugg.nvar_orb]), 1e-10)
        self.assertGreater(np.linalg.norm(hdiag[ugg.nvar_orb:]), 1e-10)

        preconditioner = hop.get_prec()
        self.assertEqual(preconditioner.shape, (ugg.nvar_tot,) * 2)
        probe = np.ones(ugg.nvar_tot, dtype=np.complex128)
        self.assertTrue(np.all(np.isfinite(preconditioner @ probe)))

    def test_three_kpoint_analytic_hdiag_matches_matvec_reference(self):
        klas, mo_ref = build_reference(
            (4.0, 10.0, 10.0), (3, 1, 1),
        )
        ci_ref = copy_ci(klas.ci)
        ugg = klas.get_ugg(mo_coeff=mo_ref, ci=ci_ref)
        self.assertGreater(np.linalg.norm(mo_ref.imag), 1e-8)
        self.assertGreater(ugg.nvar_orb_active_active, 0)

        hop = KLASSCF_HessianOperator(
            klas, ugg, mo_coeff=mo_ref, ci=ci_ref,
        )
        reference = hop._get_Horb_diag_matvec()
        analytic_external = hop._get_Horb_diag_external()
        hessian, hessian_conj = hop._get_Horb_active_active()
        analytic = np.diag(hessian + hessian_conj)
        active_start = ugg.nvar_orb_external

        external_pairs = np.argwhere(ugg.uniq_orb_idx)
        sectors = set()
        self.assertEqual(analytic_external.size, len(external_pairs))
        for index, ((k, row, column), analytic_element) in enumerate(zip(
                external_pairs, analytic_external)):
            if row < klas.ncore + klas.ncas:
                sector = "core-active"
            elif column < klas.ncore:
                sector = "core-virtual"
            else:
                sector = "active-virtual"
            sectors.add(sector)
            with self.subTest(
                    sector=sector, kpoint=k, row=row, column=column):
                np.testing.assert_allclose(
                    analytic_element,
                    reference[index],
                    atol=2e-9,
                    rtol=2e-9,
                )
        self.assertEqual(sectors, {
            "core-active", "core-virtual", "active-virtual",
        })

        self.assertEqual(analytic.size, ugg.nvar_orb_active_active)
        for index, analytic_element in enumerate(analytic):
            with self.subTest(active_active_coordinate=index):
                np.testing.assert_allclose(
                    analytic_element,
                    reference[active_start + index],
                    atol=2e-9,
                    rtol=2e-9,
                )
        np.testing.assert_allclose(
            hop._get_Horb_diag(), reference,
            atol=2e-9,
            rtol=2e-9,
        )


if __name__ == "__main__":
    unittest.main()
