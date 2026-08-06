#!/bin/bash
import unittest
import numpy as np

from pyscf.pbc import gto as pgto
from pyscf.fci import direct_spin1
from pyscf.fci import cistring

from mrh.my_pyscf.pbc.fci import direct_spin1_cplx

# Author: Bhavnesh Jangid

'''
There are four transition-density functions
    1. trans_rdm1s (spin-separated TDM1)
    2. trans_rdm1 (spin-summed TDM1)
    3. trans_rdm12s (spin-separated TDM1 and TDM2)
    4. trans_rdm12 (spin-summed TDM1 and TDM2)

The complex C implementation is compared against the Python
implementation and a complex reference constructed from real PySCF calls.
'''

def dummy_cell():
    cell = pgto.Cell()
    cell.atom = '''
    Ne 0.000000000000 0.000000000000 0.000000000000
    Ne 0.000000000000 0.000000000000 3.500000000000
    '''
    cell.a = np.diag([10.0, 10.0, 10.0])
    cell.basis = '6-31G'
    cell.unit = 'Angstrom'
    cell.verbose = 0
    cell.build()
    return cell


def _compare_two_TDM(dmA, dmB):
    assert dmA.shape == dmB.shape
    np.testing.assert_allclose(dmA.real, dmB.real,
                               atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(dmA.imag, dmB.imag,
                               atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(dmA, dmB, atol=1e-10, rtol=1e-10)


def _complex_tdm_reference(real_tdm_fn, cibra, ciket, *args, **kwargs):
    r'''Build a complex TDM from four real PySCF calls.'''
    tdm_rr = real_tdm_fn(cibra.real, ciket.real, *args, **kwargs)
    tdm_ii = real_tdm_fn(cibra.imag, ciket.imag, *args, **kwargs)
    tdm_ri = real_tdm_fn(cibra.real, ciket.imag, *args, **kwargs)
    tdm_ir = real_tdm_fn(cibra.imag, ciket.real, *args, **kwargs)
    def combine(tdm_rr, tdm_ii, tdm_ri, tdm_ir):
        if isinstance(tdm_rr, (tuple, list)):
            return tuple(combine(*blocks) 
                         for blocks in zip(tdm_rr, tdm_ii, tdm_ri, tdm_ir))
        return tdm_rr + tdm_ii + 1j * (tdm_ri - tdm_ir)
    return combine(tdm_rr, tdm_ii, tdm_ri, tdm_ir)


class KnownValues(unittest.TestCase):

    def test_complex_tdm12_and_tdm12s(self):
        ncas = 5
        nelecas = (2, 2)
        na = cistring.num_strings(ncas, nelecas[0])
        nb = cistring.num_strings(ncas, nelecas[1])
        rng = np.random.default_rng(12)
        cibra = rng.random((na, nb)) + 1j * rng.random((na, nb))
        ciket = rng.random((na, nb)) + 1j * rng.random((na, nb))
        cibra /= np.linalg.norm(cibra)
        ciket /= np.linalg.norm(ciket)

        cisolver = direct_spin1_cplx.FCISolver(dummy_cell())
        for reorder in (False, True):
            tdm12s_ref = _complex_tdm_reference(direct_spin1.trans_rdm12s, 
                                                cibra, ciket, ncas, nelecas, 
                                                reorder=reorder)
            (tdm1a_ref, tdm1b_ref), (
                tdm2aa_ref, tdm2ab_ref, tdm2ba_ref, tdm2bb_ref) = tdm12s_ref

            (tdm1a, tdm1b), (
                tdm2aa, tdm2ab, tdm2ba, tdm2bb) = cisolver.trans_rdm12s(
                    cibra, ciket, ncas, nelecas, reorder=reorder)
            
            (tdm1a_py, tdm1b_py), (
                tdm2aa_py, tdm2ab_py, tdm2ba_py,
                tdm2bb_py) = cisolver.trans_rdm12s_py(
                    cibra, ciket, ncas, nelecas, reorder=reorder)

            # Compare the complex C implementation, the Python implementation, 
            # and the reference from real PySCF calls.
            for dmA, dmB in [
                    (tdm1a, tdm1a_ref), (tdm1b, tdm1b_ref),
                    (tdm2aa, tdm2aa_ref), (tdm2ab, tdm2ab_ref),
                    (tdm2ba, tdm2ba_ref), (tdm2bb, tdm2bb_ref),
                    (tdm1a_py, tdm1a), (tdm1b_py, tdm1b),
                    (tdm2aa_py, tdm2aa), (tdm2ab_py, tdm2ab),
                    (tdm2ba_py, tdm2ba), (tdm2bb_py, tdm2bb)]:
                _compare_two_TDM(dmA, dmB)

            tdm1, tdm2 = cisolver.trans_rdm12(
                cibra, ciket, ncas, nelecas, reorder=reorder)
            _compare_two_TDM(tdm1, tdm1a_ref + tdm1b_ref)
            _compare_two_TDM(
                tdm2, tdm2aa_ref + tdm2ab_ref + tdm2ba_ref + tdm2bb_ref)

        tdm1a_from_trans_rdm1s, tdm1b_from_trans_rdm1s = \
            cisolver.trans_rdm1s(cibra, ciket, ncas, nelecas)
        tdm1a_from_trans_rdm1s_py, tdm1b_from_trans_rdm1s_py = \
            cisolver.trans_rdm1s_py(cibra, ciket, ncas, nelecas)
        tdm1_from_trans_rdm1 = cisolver.trans_rdm1(
            cibra, ciket, ncas, nelecas)
        _compare_two_TDM(tdm1a_from_trans_rdm1s, tdm1a)
        _compare_two_TDM(tdm1b_from_trans_rdm1s, tdm1b)
        _compare_two_TDM(tdm1a_from_trans_rdm1s_py, tdm1a)
        _compare_two_TDM(tdm1b_from_trans_rdm1s_py, tdm1b)
        _compare_two_TDM(tdm1_from_trans_rdm1, tdm1a + tdm1b)

        link_index = (
            cistring.gen_linkstr_index(range(ncas), nelecas[0]),
            cistring.gen_linkstr_index(range(ncas), nelecas[1]))
        (tdm1a_with_links, tdm1b_with_links), (
            tdm2aa_with_links, tdm2ab_with_links,
            tdm2ba_with_links, tdm2bb_with_links) = cisolver.trans_rdm12s(
                cibra, ciket, ncas, nelecas, link_index=link_index)
        for dmA, dmB in [
                (tdm1a_with_links, tdm1a),
                (tdm1b_with_links, tdm1b),
                (tdm2aa_with_links, tdm2aa),
                (tdm2ab_with_links, tdm2ab),
                (tdm2ba_with_links, tdm2ba),
                (tdm2bb_with_links, tdm2bb)]:
            _compare_two_TDM(dmA, dmB)

        overlap = np.vdot(cibra, ciket)
        np.testing.assert_allclose(
            np.trace(tdm1a), nelecas[0] * overlap, atol=1e-10)
        np.testing.assert_allclose(
            np.trace(tdm1b), nelecas[1] * overlap, atol=1e-10)
        pair_counts = (nelecas[0] * (nelecas[0] - 1),
                       nelecas[0] * nelecas[1],
                       nelecas[1] * nelecas[0],
                       nelecas[1] * (nelecas[1] - 1))
        for dm2, count in zip(
                (tdm2aa, tdm2ab, tdm2ba, tdm2bb), pair_counts):
            np.testing.assert_allclose(
                np.einsum('ppqq->', dm2), count * overlap, atol=1e-10)

    def test_tdm_in_limit_to_RDM(self):
        ncas = 5
        nelecas = (2, 2)
        na = cistring.num_strings(ncas, nelecas[0])
        nb = cistring.num_strings(ncas, nelecas[1])
        rng = np.random.default_rng(21)
        fcivec = rng.random((na, nb)) + 1j * rng.random((na, nb))
        fcivec /= np.linalg.norm(fcivec)

        cisolver = direct_spin1_cplx.FCISolver(dummy_cell())

        rdm1a_from_make_rdm1s, rdm1b_from_make_rdm1s = \
            cisolver.make_rdm1s(fcivec, ncas, nelecas)
        rdm1_from_make_rdm1 = cisolver.make_rdm1(fcivec, ncas, nelecas)
        (rdm1a, rdm1b), (rdm2aa, rdm2ab, rdm2bb) = \
            cisolver.make_rdm12s(fcivec, ncas, nelecas)
        rdm1, rdm2 = cisolver.make_rdm12(fcivec, ncas, nelecas)

        tdm1a_from_trans_rdm1s, tdm1b_from_trans_rdm1s = \
            cisolver.trans_rdm1s(fcivec, fcivec, ncas, nelecas)
        tdm1_from_trans_rdm1 = cisolver.trans_rdm1(
            fcivec, fcivec, ncas, nelecas)
        (tdm1a, tdm1b), (tdm2aa, tdm2ab, tdm2ba, tdm2bb) = \
            cisolver.trans_rdm12s(fcivec, fcivec, ncas, nelecas)
        tdm1, tdm2 = cisolver.trans_rdm12(
            fcivec, fcivec, ncas, nelecas)

        for dmA, dmB in [
                (tdm1a_from_trans_rdm1s, rdm1a_from_make_rdm1s),
                (tdm1b_from_trans_rdm1s, rdm1b_from_make_rdm1s),
                (tdm1_from_trans_rdm1, rdm1_from_make_rdm1),
                (tdm1a, rdm1a), (tdm1b, rdm1b),
                (tdm2aa, rdm2aa), (tdm2ab, rdm2ab),
                (tdm2ba, rdm2ab.transpose(2, 3, 0, 1)),
                (tdm2bb, rdm2bb), (tdm1, rdm1), (tdm2, rdm2)]:
            _compare_two_TDM(dmA, dmB)

        (rdm1a_py, rdm1b_py), (
            rdm2aa_py, rdm2ab_py, rdm2bb_py) = cisolver.make_rdm12s_py(
                fcivec.copy(), ncas, nelecas)
        (tdm1a_py, tdm1b_py), (
            tdm2aa_py, tdm2ab_py, tdm2ba_py,
            tdm2bb_py) = cisolver.trans_rdm12s_py(
                fcivec, fcivec, ncas, nelecas)
        rdm1a_from_make_rdm1s_py, rdm1b_from_make_rdm1s_py = \
            cisolver.make_rdm1s_py(fcivec, ncas, nelecas)
        tdm1a_from_trans_rdm1s_py, tdm1b_from_trans_rdm1s_py = \
            cisolver.trans_rdm1s_py(fcivec, fcivec, ncas, nelecas)

        for dmA, dmB in [
                (tdm1a_from_trans_rdm1s_py, rdm1a_from_make_rdm1s_py),
                (tdm1b_from_trans_rdm1s_py, rdm1b_from_make_rdm1s_py),
                (tdm1a_py, rdm1a_py), (tdm1b_py, rdm1b_py),
                (tdm2aa_py, rdm2aa_py), (tdm2ab_py, rdm2ab_py),
                (tdm2ba_py, rdm2ab_py.transpose(2, 3, 0, 1)),
                (tdm2bb_py, rdm2bb_py)]:
            _compare_two_TDM(dmA, dmB)

    def test_tdm_bra_ket_adjoint(self):
        ncas = 4
        nelecas = (2, 1)
        na = cistring.num_strings(ncas, nelecas[0])
        nb = cistring.num_strings(ncas, nelecas[1])
        rng = np.random.default_rng(7)
        cibra = rng.random((na, nb)) + 1j * rng.random((na, nb))
        ciket = rng.random((na, nb)) + 1j * rng.random((na, nb))

        cisolver = direct_spin1_cplx.FCISolver(dummy_cell())
        (tdm1a_bk, tdm1b_bk), (
            tdm2aa_bk, tdm2ab_bk, tdm2ba_bk,
            tdm2bb_bk) = cisolver.trans_rdm12s(
                cibra, ciket, ncas, nelecas)
        (tdm1a_kb, tdm1b_kb), (
            tdm2aa_kb, tdm2ab_kb, tdm2ba_kb,
            tdm2bb_kb) = cisolver.trans_rdm12s(
                ciket, cibra, ncas, nelecas)

        for dmA, dmB in [
                (tdm1a_bk, tdm1a_kb.conj().T),
                (tdm1b_bk, tdm1b_kb.conj().T),
                (tdm2aa_bk, tdm2aa_kb.transpose(1, 0, 3, 2).conj()),
                (tdm2ab_bk, tdm2ab_kb.transpose(1, 0, 3, 2).conj()),
                (tdm2ba_bk, tdm2ba_kb.transpose(1, 0, 3, 2).conj()),
                (tdm2bb_bk, tdm2bb_kb.transpose(1, 0, 3, 2).conj())]:
            _compare_two_TDM(dmA, dmB)

        tdm1_bk, tdm2_bk = cisolver.trans_rdm12(
            cibra, ciket, ncas, nelecas)
        tdm1_kb, tdm2_kb = cisolver.trans_rdm12(
            ciket, cibra, ncas, nelecas)
        _compare_two_TDM(tdm1_bk, tdm1_kb.conj().T)
        _compare_two_TDM(
            tdm2_bk, tdm2_kb.transpose(1, 0, 3, 2).conj())

    def test_tdm_electron_sectors(self):
        ncas = 4
        nelecaslst = [(0, 0), (4, 0), (0, 4), (4, 4), (2, 1), (1, 3)]
        rng = np.random.default_rng(91)
        cisolver = direct_spin1_cplx.FCISolver(dummy_cell())

        for nelecas in nelecaslst:
            with self.subTest(nelecas=nelecas):
                na = cistring.num_strings(ncas, nelecas[0])
                nb = cistring.num_strings(ncas, nelecas[1])
                cibra = rng.random((na, nb)) + 1j * rng.random((na, nb))
                ciket = rng.random((na, nb)) + 1j * rng.random((na, nb))

                (tdm1a, tdm1b), (
                    tdm2aa, tdm2ab, tdm2ba,
                    tdm2bb) = cisolver.trans_rdm12s(
                        cibra, ciket, ncas, nelecas)
                
                (tdm1a_py, tdm1b_py), (
                    tdm2aa_py, tdm2ab_py, tdm2ba_py,
                    tdm2bb_py) = cisolver.trans_rdm12s_py(
                        cibra, ciket, ncas, nelecas)

                for dmA, dmB in [
                        (tdm1a, tdm1a_py), (tdm1b, tdm1b_py),
                        (tdm2aa, tdm2aa_py), (tdm2ab, tdm2ab_py),
                        (tdm2ba, tdm2ba_py), (tdm2bb, tdm2bb_py)]:
                    _compare_two_TDM(dmA, dmB)

                overlap = np.vdot(cibra, ciket)
                np.testing.assert_allclose(
                    np.trace(tdm1a), nelecas[0] * overlap, atol=1e-10)
                np.testing.assert_allclose(
                    np.trace(tdm1b), nelecas[1] * overlap, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
    
