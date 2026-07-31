#!/usr/bin/env python

import unittest
from unittest import mock

import numpy as np

from pyscf.pbc import gto, scf

from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.mcscf import klasci as klasci_module
from mrh.my_pyscf.pbc.mcscf.klasci import (
    PBCLASCINoSymm,
    PBCLASCITransSymm,
    kLASCI,
)
from mrh.my_pyscf.pbc.mcscf.productstate import (
    PBCTransSymmImpureProductStateFCISolver,
)

# Author: Bhavnesh Jangid

# Test-0: trans_sym=True should select PBCLASCITransSymm, validate the
#          Wannier orbitals and Hamiltonians, and use the translation-symmetric
#         product-state solver with the requested reference cell.
# Test-1: PBCLASCITransSymm and PBCLASCINoSymm should produce the same energy
#         when they use the same localized active orbitals.
# Test-2: The product-state CI helpers should select the reference-cell CI
#         vector and reconstruct independent fragment vectors with optional
#         translation phases.
# Test-3: The reference-cell initial guess should be generated from the real
#         H2 active-space Hamiltonian and preserve a supplied CI vector.



cell = kmf = mo_coeff = None
kmesh = [2, 1, 1]


def setUpModule():
    global cell, kmf, mo_coeff
    cell = gto.Cell()
    cell.a = np.diag([3.0, 10.0, 10.0])
    cell.atom = "H 0 0 0; H 0.74 0 0"
    cell.basis = "6-31g"
    cell.unit = "Angstrom"
    cell.precision = 1e-8
    cell.ke_cutoff = 20
    cell.verbose = 0
    cell.build()

    kpts = cell.make_kpts(kmesh, wrap_around=True)
    kmf = scf.KRHF(cell, kpts=kpts).density_fit()
    kmf.exxdiv = None
    kmf.max_cycle = 0
    kmf.kernel()
    mo_coeff = avas.kernel(kmf, ["H 1s"], minao=cell.basis)[2]


class KnownValues(unittest.TestCase):

    def test_trans_sym_checks_wannier_hamiltonians(self):
        trans_klas = kLASCI(
            kmf, 2, (1, 1), kmesh=kmesh, trans_sym=True, ref_cell=1,
        )
        mo_loc = trans_klas.localize_init_guess(
            ["H 1s"], mo_coeff=mo_coeff,
        )

        with mock.patch.object(
                klasci_module, "check_wannier_orbital_translation",
                wraps=klasci_module.check_wannier_orbital_translation
             ) as check_orbitals, \
             mock.patch.object(
                klasci_module, "check_h1e_translation",
                wraps=klasci_module.check_h1e_translation) as check_h1e, \
             mock.patch.object(
                klasci_module, "check_h2e_translation",
                wraps=klasci_module.check_h2e_translation) as check_h2e, \
             mock.patch.object(
                klasci_module, "PBCTransSymmImpureProductStateFCISolver",
                wraps=PBCTransSymmImpureProductStateFCISolver
             ) as trans_solver:
            trans_klas.kernel(mo_loc)

        self.assertIs(type(trans_klas), PBCLASCITransSymm)
        self.assertTrue(trans_klas.trans_sym)
        self.assertEqual(trans_klas.ref_cell, 1)
        self.assertEqual(len(trans_klas.ci), np.prod(kmesh))
        check_orbitals.assert_called_once()
        check_h1e.assert_called_once()
        check_h2e.assert_called_once()
        trans_solver.assert_called_once()
        self.assertEqual(trans_solver.call_args.kwargs["ref_cell"], 1)

        with self.assertRaises(NotImplementedError):
            trans_klas.pack_h1(np.empty((0, 0)))
        with self.assertRaises(NotImplementedError):
            trans_klas.pack_h2(np.empty((0, 0, 0, 0)))

        ncas_sub = trans_klas.ncas_sub.copy()
        try:
            trans_klas.ncas_sub[1] += 1
            with self.assertRaisesRegex(
                    ValueError, "active-space consistency check failed"):
                trans_klas._sanity_check_active_space_consistency(mo_loc)
        finally:
            trans_klas.ncas_sub = ncas_sub

        with self.assertRaisesRegex(TypeError, "trans_sym must be a boolean"):
            kLASCI(kmf, 2, (1, 1), kmesh=kmesh, trans_sym="yes")
        with self.assertRaisesRegex(ValueError, "ref_cell must be in"):
            kLASCI(
                kmf, 2, (1, 1), kmesh=kmesh,
                trans_sym=True, ref_cell=np.prod(kmesh),
            )

    def test_trans_sym_class_api_and_energy(self):
        plain_klas = kLASCI(kmf, 2, (1, 1), kmesh=kmesh)
        trans_klas = kLASCI(
            kmf, 2, (1, 1), kmesh=kmesh,
            trans_sym=True, ref_cell=1,
        )

        self.assertIs(type(plain_klas), PBCLASCINoSymm)
        self.assertIs(type(trans_klas), PBCLASCITransSymm)
        self.assertTrue(trans_klas.trans_sym)
        self.assertEqual(trans_klas.ref_cell, 1)
        self.assertTrue(callable(trans_klas.pack_h1))
        self.assertTrue(callable(trans_klas.pack_h2))

        mo_loc = plain_klas.localize_init_guess(
            ["H 1s"], mo_coeff=mo_coeff,
        )
        energy_plain = plain_klas.kernel(np.array(mo_loc, copy=True))[1]
        energy_trans = trans_klas.kernel(np.array(mo_loc, copy=True))[1]

        self.assertAlmostEqual(energy_plain.real, energy_trans.real, places=10)
        self.assertAlmostEqual(energy_plain.imag, energy_trans.imag, places=10)

    def test_productstate_pack_and_unpack_ci(self):

        trans_klas = kLASCI(
            kmf, 2, (1, 1), kmesh=kmesh,
            trans_sym=True, ref_cell=1,)
        
        mo_loc = trans_klas.localize_init_guess(
            ["H 1s"], mo_coeff=mo_coeff,
        )
        trans_klas.kernel(mo_loc)

        fcisolvers = [box.fcisolvers[0] for box in trans_klas.fciboxes]
        solver = PBCTransSymmImpureProductStateFCISolver(
            fcisolvers,
            lweights=[[1.0], [1.0]],
            ref_cell=1,
        )
        ci = [np.asarray(ci_frag) 
              for ci_frag in trans_klas.ci]

        ci_ref = solver._pack_ci(ci)
        phases = []
        for ci_frag in ci:
            overlap = np.vdot(ci_ref, ci_frag)
            phases.append(overlap / abs(overlap))

        ci_fragments = solver._unpack_cif(ci_ref, phases=phases)

        self.assertEqual(ci_ref.shape, (1, 2, 2))
        self.assertAlmostEqual(np.linalg.norm(ci_ref), 1.0, places=10)
        self.assertLess(np.max(np.abs(ci_ref - ci[1])), 1e-12)
        self.assertIsNot(ci_ref, ci[1])

        for ci_unpacked, ci_actual in zip(ci_fragments, ci):
            self.assertLess(
                np.max(np.abs(ci_unpacked - ci_actual)), 1e-10,
            )
            self.assertFalse(np.shares_memory(ci_unpacked, ci_ref))
            
        self.assertIsNone(solver._pack_ci(None))
        self.assertEqual(solver._unpack_cif(None), [None, None])

    def test_productstate_ref_init_guess(self):
        trans_klas = kLASCI(
            kmf, 2, (1, 1), kmesh=kmesh,
            trans_sym=True, ref_cell=1,
        )
        mo_loc = trans_klas.localize_init_guess(
            ["H 1s"], mo_coeff=mo_coeff,
        )
        h1e = trans_klas.h1e_for_cas(
            mo_coeff=mo_loc, ncas=trans_klas.ncas,
            ncore=trans_klas.ncore,
        )[0]
        h2e = trans_klas.get_h2cas(mo_loc)

        fcisolvers = [box.fcisolvers[0] for box in trans_klas.fciboxes]
        solver = PBCTransSymmImpureProductStateFCISolver(
            fcisolvers,
            lweights=[[1.0], [1.0]],
            ref_cell=1,
        )
        ci_ref = solver._get_ref_init_guess(
            None, trans_klas.ncas_sub, trans_klas.nelecas_sub, h1e, h2e,
        )

        self.assertEqual(ci_ref.shape, (1, 2, 2))
        self.assertAlmostEqual(np.linalg.norm(ci_ref), 1.0, places=10)

        ci_supplied = np.exp(0.3j) * ci_ref
        ci_preserved = solver._get_ref_init_guess(
            ci_supplied, trans_klas.ncas_sub, trans_klas.nelecas_sub,
            h1e, h2e,
        )
        self.assertLess(np.max(np.abs(ci_preserved - ci_supplied)), 1e-12)
        self.assertIsNot(ci_preserved, ci_supplied)


if __name__ == "__main__":
    unittest.main()
