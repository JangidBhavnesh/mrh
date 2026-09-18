"""Native spectral kernels, Python parity, and explicit fallback coverage."""

import unittest
from unittest import mock

import numpy as np
from pyscf import lib
from pyscf.pbc import gto

from mrh.my_pyscf.pbc.fci import kcistrings
from mrh.my_pyscf.pbc.fci import spectral_fn_helper as sfh
from mrh.tests.kFCI.test_pbc_kfci_spectral_fn_helper import _complete_2d_roots


def _kmom_2d():
    cell = gto.Cell()
    cell.a = np.eye(3) * 4
    cell.atom = 'He 0 0 0'
    cell.basis = 'sto-3g'
    cell.verbose = 0
    cell.build()
    kpts = cell.make_kpts([2, 2, 1], wrap_around=True)
    return kcistrings.make_kpoint_momentum(4, cell=cell, kpts=kpts)


def _poles():
    rng = np.random.default_rng(67)
    return [dict(kind='hole' if i % 2 else 'particle', k=int(rng.integers(4)),
                 orbital=int(rng.integers(3)), spin=int(rng.integers(2)),
                 omega=rng.normal(), weight=rng.random()) for i in range(240)]


class NativeSpectralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.native = sfh._load_spectral_lib()
        if cls.native is None:
            raise unittest.SkipTest('libpbc_spectral_fn is not built')

    def test_creation_and_annihilation_match_python(self):
        rng = np.random.default_rng(59)
        cases = [(1, 2, (1, 1)), (3, 2, (2, 1)), (2, 1, (1, 1)),
                 (2, 1, (0, 0)), (2, 1, (2, 2)), (2, 1, (0, 1))]
        for nkpts, ncas, nelec in cases:
            norb = nkpts * ncas
            for target_k in range(nkpts):
                layout = sfh.make_k_sector_layout(norb, nelec, nkpts, target_k)
                for dtype in (np.float64, np.complex128):
                    data = rng.normal(size=2 * layout.sector_size).astype(dtype)
                    if dtype == np.complex128:
                        data += 1j * rng.normal(size=data.size)
                    ci = data[::2]  # Exercise noncontiguous, read-only inputs.
                    ci.flags.writeable = False
                    for cre in (False, True):
                        for spin in (0, 1):
                            if (cre and nelec[spin] == norb) or (not cre and nelec[spin] == 0):
                                continue
                            for k in range(nkpts):
                                for p in range(ncas):
                                    with self.subTest(nkpts=nkpts, nelec=nelec,
                                                      cre=cre, spin=spin, k=k,
                                                      p=p, target_k=target_k, dtype=dtype):
                                        args = (ci, norb, nelec, nkpts, target_k, k, p, spin)
                                        ref, ref_info = sfh.apply_k_op_py(
                                            *args, cre=cre, return_info=True)
                                        with mock.patch.object(sfh, 'apply_k_op_py',
                                                side_effect=AssertionError('unexpected fallback')):
                                            actual, info = sfh.apply_k_op(
                                                *args, cre=cre, return_info=True)
                                        self.assertEqual(info, ref_info)
                                        self.assertEqual(actual.dtype, ref.dtype)
                                        np.testing.assert_allclose(actual, ref, atol=1e-13, rtol=1e-13)

    def test_operator_parallel_block_loop(self):
        nkpts, norb, nelec, target_k = 9, 9, (2, 1), 4
        rng = np.random.default_rng(71)
        layout = sfh.make_k_sector_layout(norb, nelec, nkpts, target_k)
        self.assertGreater(len(layout.blocks), 8)  # Native OpenMP threshold.
        ci = rng.normal(size=layout.sector_size) + 1j * rng.normal(size=layout.sector_size)
        for cre in (False, True):
            for spin in (0, 1):
                args = (ci, norb, nelec, nkpts, target_k, 8, 0, spin)
                ref = sfh.apply_k_op_py(*args, cre=cre)
                for nthreads in (1, 4):
                    with lib.with_omp_threads(nthreads), mock.patch.object(
                            sfh, 'apply_k_op_py', side_effect=AssertionError('unexpected fallback')):
                        actual = sfh.apply_k_op(*args, cre=cre)
                    np.testing.assert_allclose(actual, ref, atol=1e-13, rtol=1e-13)

    def test_broadening_matches_python_serial_and_parallel(self):
        poles = _poles()
        omega = np.linspace(-4, 4, 514)[::2]
        self.assertGreater(len(poles) * omega.size, 10000)
        for broadening in ('lorentzian', 'gaussian'):
            for orbital_resolved in (False, True):
                for spin_resolved in (False, True):
                    kwargs = dict(omega_grid=omega, eta=0.07, broadening=broadening,
                                  nkpts=4, norb=3, orbital_resolved=orbital_resolved,
                                  spin_resolved=spin_resolved)
                    ref = sfh.make_spectral_function_py(poles, **kwargs)
                    for nthreads in (1, 4):
                        with lib.with_omp_threads(nthreads), mock.patch.object(
                                sfh, 'make_spectral_function_py',
                                side_effect=AssertionError('unexpected fallback')):
                            actual = sfh.make_spectral_function(poles, **kwargs)
                        for kind in ('hole', 'particle', 'total'):
                            np.testing.assert_allclose(actual['spectra'][kind],
                                                       ref['spectra'][kind], rtol=1e-12, atol=1e-12)
                        np.testing.assert_allclose(actual['spectra']['total'],
                                                   actual['spectra']['hole'] + actual['spectra']['particle'])

    def test_multidimensional_poles_and_empty_poles_use_native_broadening(self):
        roots = _complete_2d_roots(_kmom_2d())
        for poles in (sfh.make_spectral_poles(roots), []):
            kwargs = dict(omega_min=-5, omega_max=5, npts=101, nkpts=4, norb=1)
            ref = sfh.make_spectral_function_py(poles, **kwargs)
            with mock.patch.object(sfh, 'make_spectral_function_py',
                                   side_effect=AssertionError('unexpected fallback')):
                actual = sfh.make_spectral_function(poles, **kwargs)
            np.testing.assert_allclose(actual['spectra']['total'], ref['spectra']['total'])

    def test_library_loading_is_lazy_and_cached(self):
        with mock.patch.multiple(sfh, libpbcspectral=None, _spectral_lib_initialized=False), \
                mock.patch.object(sfh, 'load_library', return_value=self.native) as loader:
            self.assertIs(sfh._load_spectral_lib(), self.native)
            self.assertIs(sfh._load_spectral_lib(), self.native)
            loader.assert_called_once_with('libpbc_spectral_fn')
        self.assertIsNone(self.native.FCIspectral_apply_k_op.restype)
        self.assertIsNone(self.native.FCIspectral_broaden.restype)


class SpectralFallbackAndValidationTests(unittest.TestCase):
    def test_missing_library_falls_back_for_both_kernels(self):
        args = (np.array([1., 2.]), 2, (1, 1), 2, 0, 1, 0, 1)
        ref_op = sfh.apply_k_op_py(*args)
        poles = _poles()
        ref_spectrum = sfh.make_spectral_function_py(poles, npts=31)
        with mock.patch.multiple(sfh, libpbcspectral=None, _spectral_lib_initialized=False), \
                mock.patch.object(sfh, 'load_library', side_effect=OSError('missing')) as loader:
            np.testing.assert_allclose(sfh.apply_k_op(*args), ref_op)
            actual = sfh.make_spectral_function(poles, npts=31)
            np.testing.assert_allclose(actual['spectra']['total'], ref_spectrum['spectra']['total'])
            loader.assert_called_once()

    def test_explicit_python_does_not_attempt_library_loading(self):
        with mock.patch.object(sfh, '_load_spectral_lib', side_effect=AssertionError('native load')):
            sfh.des_k(np.array([1., 2.]), 2, (1, 1), 2, 0, 1, 0, 1, use_c=False)
            sfh.make_spectral_function(_poles(), npts=31, use_c=False)

    def test_multidimensional_operator_and_single_precision_fallback(self):
        kmom = _kmom_2d()
        roots = _complete_2d_roots(kmom)
        cases = [((roots.roots[0]['ci'], 4, (1, 1), 4, 2, 1, 0, 1), dict(kmom=kmom)),
                 ((np.array([1., 2.], dtype=np.float32), 2, (1, 1), 2, 0, 1, 0, 1), {})]
        for args, kwargs in cases:
            ref = sfh.apply_k_op_py(*args, **kwargs)
            with mock.patch.object(sfh, '_load_spectral_lib', side_effect=AssertionError('native load')):
                actual = sfh.apply_k_op(*args, **kwargs)
            self.assertEqual(actual.dtype, ref.dtype)
            np.testing.assert_allclose(actual, ref)

    def test_bad_ci_size_is_rejected_before_native_access(self):
        for use_c in (False, True):
            with mock.patch.object(sfh, '_load_spectral_lib', side_effect=AssertionError('native load')):
                with self.assertRaisesRegex(ValueError, 'CI vector size'):
                    sfh.des_k(np.zeros(1), 2, (1, 1), 2, 0, 1, 0, 1, use_c=use_c)

    def test_broadening_input_errors_agree_across_backends(self):
        pole = dict(kind='hole', k=0, orbital=0, spin=0, omega=0., weight=1.)
        kwargs = dict(omega_grid=np.linspace(-1, 1, 5), nkpts=1, norb=1)
        bad_poles = [dict(pole, **update) for update in (
            {'kind': 'other'}, {'k': -1}, {'k': 1}, {'orbital': 1},
            {'spin': 2}, {'spin': 0.5}, {'omega': np.nan}, {'weight': np.inf})]
        bad_kwargs = [dict(kwargs, **update) for update in (
            {'eta': 0}, {'eta': -1}, {'eta': np.nan}, {'broadening': 'other'},
            {'omega_grid': np.zeros((2, 2))}, {'omega_grid': 0.},
            {'omega_grid': [0, np.inf]})]
        for use_c in (False, True):
            for row in bad_poles:
                with self.subTest(use_c=use_c, row=row), self.assertRaises(ValueError):
                    sfh.make_spectral_function([row], use_c=use_c, **kwargs)
            for invalid in bad_kwargs:
                with self.subTest(use_c=use_c, kwargs=invalid), self.assertRaises(ValueError):
                    sfh.make_spectral_function([pole], use_c=use_c, **invalid)


if __name__ == '__main__':
    unittest.main()
