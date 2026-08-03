import numpy as np
from scipy import linalg
from itertools import combinations

from pyscf import lib
from pyscf.scf.addons import canonical_orth_
from pyscf.fci import cistring
from pyscf.mcscf.addons import state_average as state_average_mcscf

from mrh.my_pyscf.pbc.fci.csf_cplx import cplxCSFFCISolver as CSFFCISolver
from mrh.my_pyscf.mcscf.productstate import ProductStateFCISolver as molProductStateFCISolver, state_average_fcisolver


# Author: Bhavnesh Jangid


'''
# TODO-1: add multiple root testing for the PBCTransSymmImpureProductStateFCISolver class.
# The current implementation does support multiple roots, but it has not been tested with
# more than one root per fragment.


'''

class PBCProductStateFCISolver (molProductStateFCISolver):

    def kernel (self, h1, h2, norb_f, nelec_f, ecore=0, ci0=None, orbsym=None,
            conv_tol_grad=1e-4, conv_tol_self=1e-10, max_cycle_macro=50,
            serialfrag=False, **kwargs):
        log = self.log
        converged = False
        e_sigma = 0.0
        e = [0 for n in norb_f]
        ci1 = ci0
        log.info ('Entering product-state fixed-point CI iteration')
        for it in range (max_cycle_macro):
            ci0 = self.get_init_guess (ci1, norb_f, nelec_f, h1, h2)
            # Issue #86: put get_init_guess INSIDE the iteration in case _1shot below encounters
            # linear dependencies and can't populate all CI vectors for all fragments.
            h1eff, h0eff, ci0 = self.project_hfrag (h1, h2, ci0, norb_f, nelec_f,
                ecore=ecore, **kwargs)

            grad = self._get_grad (h1eff, h2, ci0, norb_f, nelec_f, **kwargs)
            grad_max = np.amax (np.abs (grad))
            solvers_converged = [np.all (np.asarray (s.converged)) for s in self.fcisolvers]
            nconv = sum ([int (c) for c in solvers_converged])
            log.info ('Cycle %d: max grad = %e ; sigma = %e ; %d/%d fragment CI solvers converged',
                      it, grad_max, e_sigma.real, nconv, len (self.fcisolvers))
            log.debug ('e vector = {}'.format (e))
            if nconv<len(self.fcisolvers): log.debug ('unconverged fragment CI solvers: {}'.format (
                list(np.where (np.logical_not (solvers_converged))[0])))
            if ((grad_max < conv_tol_grad) and (e_sigma < conv_tol_self)
                and all ([solvers_converged]) and it>0):
                converged = True
                break
            e, ci1 = self._1shot (it, h0eff, h1eff, h2, e, ci0, norb_f, nelec_f,
                orbsym=orbsym, serialfrag=serialfrag, **kwargs)
            e_sigma = np.amax (e) - np.amin (e)
        conv_str = ['NOT converged','converged'][int (converged)]
        log.info (('Product_state fixed-point CI iteration {} after {} '
                   'cycles').format (conv_str, it))
        if not converged:
            ci1 = self.get_init_guess (ci1, norb_f, nelec_f, h1, h2)
            # Issue #86: see above, same problem
            self._debug_csfs (log, ci0, ci1, norb_f, nelec_f, grad)
        energy_elec = self.energy_elec (h1, h2, ci1, norb_f, nelec_f,
            ecore=ecore, efinal=e, **kwargs)
        return converged, energy_elec, ci1

    def get_init_guess (self, ci0, norb_f, nelec_f, h1, h2, nroots=None):
        '''Generate CI guess vectors for all fragments.

        Args:
            ci0: list of length nfrag or None
                Contains either None or ndarrays of guess CI vectors. Any new guess CI vectors
                constructed by this function are constrained to be orthogonal to those already
                provided here, if any.
            norb_f: list of length nfrag of integers
                Number of orbitals in each fragment
            nelec_f: list of length nfrag of integers
                Number of electrons (in reference state) in each fragment
            h1: ndarray of shape (ncas,ncas) or (2,ncas,ncas)
                One-electron Hamiltonian amplitudes
            h2: ndarray of shape (ncas,ncas,ncas,ncas)
                Two-electron Hamiltonian amplitudes

        Returns:
            ci1: list of length nfrag of ndarrays
                Orthonormal guess CI vectors. Any vectors present in ci0 are preserved unaltered.
        '''
        if ci0 is None: ci0 = [None for i in range (len (norb_f))]
        ci1 = [c for c in ci0] # reference safety
        if h1.ndim < 3: h1 = np.stack ([h1, h1], axis=0)
        for ix, (no, ne, solver) in enumerate (zip (norb_f, nelec_f, self.fcisolvers)):
            solver.check_transformer_cache ()
            snroots = solver.nroots if nroots is None else min (nroots, solver.transformer.ncsf)
            nelec = self._get_nelec (solver, ne)
            i = sum (norb_f[:ix])
            j = i + norb_f[ix]
            hdiag_csf = solver.make_hdiag_csf (h1[:,i:j,i:j], h2[i:j,i:j,i:j,i:j],
                                               no, nelec)
            ci1_guess = solver.get_init_guess (no, nelec, snroots, hdiag_csf)
            na = cistring.num_strings (no, nelec[0])
            nb = cistring.num_strings (no, nelec[1])
            if ci1[ix] is None:
                ci1[ix] = ci1_guess
            elif np.asarray (ci1[ix]).reshape (-1,na*nb).shape[0] < snroots:
                ci1_inp = np.asarray (ci1[ix]).reshape (-1,na*nb)
                ci1_guess = np.asarray (ci1_guess).reshape (-1,na*nb)
                x = np.append (ci1_inp, ci1_guess, axis=0)
                x = canonical_orth_(x.conj () @ x.T).T @ x
                # ^ an orthonormal basis
                assert (x.shape[0] >= snroots)
                x2inp = x.conj () @ ci1_inp.T
                ninp = ci1_inp.shape[0]
                u, svals, vh = linalg.svd (x2inp, full_matrices=True)
                u[:,:ninp] = u[:,:ninp] @ vh
                ci1_new = u.T @ x
                nnew = ci1_new.shape[0]
                ovlp = ci1_new.conj () @ ci1_inp.T
                assert (np.all (np.abs (ovlp[:ninp,:ninp] - np.eye (ninp)) < 1e-3)), '{}'.format (ovlp)
                ovlp = (ci1_new.conj () @ ci1_new.T)
                assert (np.all (np.abs (ovlp - np.eye (nnew)) < 1e-3)), '{}'.format (ovlp)
                ci1[ix] = ci1_new[:snroots].reshape (snroots, na, nb)
        return self._check_init_guess (ci1, norb_f, nelec_f, nroots=nroots)

    def _check_init_guess (self, ci0, norb_f, nelec_f, nroots=None):
        ci1 = []
        if ci0 is None: ci0 = [None for i in range (len (norb_f))]
        for ix, (no, ne, solver) in enumerate (zip (norb_f, nelec_f, self.fcisolvers)):
            solver.check_transformer_cache ()
            snroots = solver.nroots if nroots is None else min (nroots, solver.transformer.ncsf)
            nelec = self._get_nelec (solver, ne)
            neleca, nelecb = nelec
            na = cistring.num_strings (no, neleca)
            nb = cistring.num_strings (no, nelecb)
            zguess = np.zeros ((snroots,na,nb), dtype=np.complex128)
            cguess = np.asarray (ci0[ix]).reshape (-1,na,nb)
            ngroots = min (zguess.shape[0], cguess.shape[0])
            zguess[:ngroots,:,:] = cguess[:ngroots,:,:]
            ci1.append (zguess)
            if snroots>na*nb:
                raise RuntimeError ("{} roots > {} determinants in fragment {}".format (
                    snroots, na*nb, ix))
            if isinstance (solver, CSFFCISolver):
                solver.check_transformer_cache ()
                if snroots>solver.transformer.ncsf:
                    raise RuntimeError ("{} roots > {} CSFs in fragment {} (nelec={}, smult={})".format (
                        snroots, solver.transformer.ncsf, ix, solver.nelec, solver.smult))
        return ci1
                
    def _debug_csfs (self, log, ci0, ci1, norb_f, nelec_f, grad, nroots=None):
        if not all ([isinstance (s, CSFFCISolver) for s in self.fcisolvers]):
            return
        if log.verbose < lib.logger.INFO: return
        transformers = [s.transformer for s in self.fcisolvers]
        grad_f = []
        for s,t in zip (self.fcisolvers, transformers):
            snroots = nroots if nroots is not None else s.nroots
            grad_f.append (grad[:t.ncsf*snroots].reshape (snroots, t.ncsf))
            offs = (t.ncsf*snroots) + (snroots*(snroots-1)//2)
            grad = grad[offs:]
        assert (len (grad) == 0)
        log.info ('Debugging CI and gradient vectors...')
        for ix, (grad, c0, c1, s, t) in enumerate (zip (grad_f, ci0, ci1, self.fcisolvers, transformers)):
            log.info ('Fragment %d', ix)
            c0_csf, c0_norm = t.vec_det2csf (c0, normalize=True, return_norm=True)
            c1_csf, c1_norm = t.vec_det2csf (c1, normalize=True, return_norm=True)
            log.info ('CI vector norm = %s', str(c1_norm))
            grad_norm = linalg.norm (grad)
            log.info ('Gradient norm = %e', grad_norm)
            c0_lbls, c0_coeffs = t.printable_largest_csf (c0_csf, 10)
            c1_lbls, c1_coeffs = t.printable_largest_csf (c1_csf, 10)
            g_lbls, g_coeffs = t.printable_largest_csf (grad, 10, normalize=False)
            nroots = len (c0_lbls)
            for i in range (nroots):
                log.info ('Previous CI vector leading components (%d/%d):', i, nroots)
                for l, c in zip (c0_lbls[i], c0_coeffs[i]):
                    log.info ('%s : %e', l, c)
                log.info ('Current CI vector leading components (%d/%d):', i, nroots)
                for l, c in zip (c1_lbls[i], c1_coeffs[i]):
                    log.info ('%s : %e', l, c)
                log.info ('Grad vector leading components (%d/%d):', i, nroots)
                for l, c in zip (g_lbls[i], g_coeffs[i]):
                    log.info ('%s : %e', l, c)

    def _1shot (self, it, h0eff, h1eff, h2, e0, ci0, norb_f, nelec_f, orbsym=None,
                serialfrag=False, **kwargs):
        nfrag = len (norb_f)
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        zipper = [h0eff, h1eff, ci0, norb_f, nelec_f, self.fcisolvers, ni, nj]
        e1 = [e for e in e0]
        ci1 = [c for c in ci0]

        for ifrag, (h0e, h1e, c, no, ne, solver, i, j) in enumerate (zip (*zipper)):
            if serialfrag and it % nfrag != ifrag: continue
            h2e = h2[i:j,i:j,i:j,i:j]
            osym = getattr (solver, 'orbsym', None)
            if orbsym is not None: osym=orbsym[i:j]
            nelec = self._get_nelec (solver, ne)
            e, c1 = solver.kernel (h1e, h2e, no, nelec, ci0=c, ecore=h0e,
                orbsym=osym, **kwargs)
            e1[ifrag] = e
            ci1[ifrag] = c1
        return e1, ci1

    def _get_grad (self, h1eff, h2, ci, norb_f, nelec_f, orbsym=None,
            **kwargs):
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        zipper = [h1eff, ci, norb_f, nelec_f, self.fcisolvers, ni, nj]
        grad = []
        for h1e, c, no, ne, solver, i, j in zip (*zipper):
            nelec = self._get_nelec (solver, ne)
            nroots = solver.nroots
            h2e = h2[i:j,i:j,i:j,i:j]
            h2e = solver.absorb_h1e (h1e, h2e, no, nelec, 0.5)
            if nroots==1: c=c[None,:] # nroots, na, nb
            hc = [solver.contract_2e (h2e, col, no, nelec) for col in c]
            c, hc = np.asarray (c), np.asarray (hc)
            chc = np.dot (np.asarray (c).reshape (nroots,-1).conj (),
                          np.asarray (hc).reshape (nroots,-1).T)
            hc = hc - np.tensordot (chc, c, axes=1)
            if isinstance (solver, CSFFCISolver):
                #hc = solver.transformer.vec_det2csf (hc, normalize=False)
                creal = solver.transformer.vec_det2csf (hc.real, order='C', normalize=False)
                cimag = solver.transformer.vec_det2csf (hc.imag, order='C', normalize=False)
                cout = creal.astype(h1e.dtype)
                cout.real = creal
                cout.imag = cimag
                hc = cout
            # External degrees of freedom: not weighted, because I want
            # to converge all of the roots even if they don't contribute
            # to the mean field
            assert (hc.size == nroots*solver.transformer.ncsf)
            grad.append (hc.ravel ())
            # Internal degrees of freedom: weighted and lower-triangular
            # TODO: confirm the sign choice below before using this gradient
            # for something more advanced than convergence checking
            if nroots>1 and getattr (solver, 'weights', None) is not None:
                chc *= np.asarray (solver.weights)[:,None]
                chc -= chc.T
                grad.append (chc[np.tril_indices (nroots,k=-1)])
        return np.concatenate (grad)

    def energy_elec (self, h1, h2, ci, norb_f, nelec_f, ecore=0, **kwargs):
        dm1s = np.stack (self.make_rdm1s (ci, norb_f, nelec_f), axis=0)
        if h1.ndim < 3: h1 = np.stack ([h1, h1], axis=0)
        dm2 = self.make_rdm2 (ci, norb_f, nelec_f)
        energy_tot = (ecore + np.tensordot (h1, dm1s, axes=3)
                        + 0.5*np.tensordot (h2, dm2, axes=4))
        return energy_tot

    def project_hfrag (self, h1, h2, ci, norb_f, nelec_f, 
                       ecore=0, dm1s=None, dm2=None, **kwargs):
        '''
        Project the h1e and h2e on the fragment space.
        '''
        if dm1s is None:
            dm1s = np.stack (self.make_rdm1s (ci, norb_f, nelec_f), axis=0)
        if h1.ndim < 3:
            h1 = np.stack ([h1,h1], axis=0)
        if dm2 is None:
            dm2 = self.make_rdm2 (ci, norb_f, nelec_f)
        energy_tot = (ecore + np.tensordot (h1, dm1s, axes=3)
                        + 0.5*np.tensordot (h2, dm2, axes=4))
        v1  = np.tensordot (dm1s, h2, axes=2)
        v1 += v1[::-1] # ja + jb
        v1 -= np.tensordot (dm1s, h2, axes=((1,2),(2,1)))
        f1 = h1 + v1
        h1eff = []
        h0eff = []
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        for i, j in zip (ni, nj):
            dm1s_i = dm1s[:,i:j,i:j]
            dm2_i = dm2[i:j,i:j,i:j,i:j]
            # v1 self-interaction
            h2_i = h2[i:j,i:j,:,:]
            v1_i = np.tensordot (dm1s_i, h2_i, axes=2)
            v1_i += v1_i[::-1] # ja + jb
            h2_i = h2[:,i:j,i:j,:]
            v1_i -= np.tensordot (dm1s_i, h2_i, axes=((1,2),(2,1)))
            # cancel off-diagonal energy double-counting
            e_i = energy_tot - np.tensordot (dm1s, v1_i, axes=3) # overcorrects
            # cancel h1eff double-counting
            v1_i = v1_i[:,i:j,i:j] 
            h1eff.append (f1[:,i:j,i:j]-v1_i)
            # cancel diagonal energy double-counting
            h1_i = h1[:,i:j,i:j] - v1_i # v1_i fixes overcorrect
            h2_i = h2[i:j,i:j,i:j,i:j]
            e_i -= (np.tensordot (h1_i, dm1s_i, axes=3)
              + 0.5*np.tensordot (h2_i, dm2_i, axes=4))
            h0eff.append (e_i)
        return h1eff, h0eff, ci

    def make_rdm1s (self, ci, norb_f, nelec_f, **kwargs):
        dtype = np.array(ci).dtype
        norb = sum (norb_f)
        dm1a = np.zeros ((norb, norb), dtype=dtype)
        dm1b = np.zeros ((norb, norb), dtype=dtype)
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        for ix, (i, j, c, no, ne, s) in enumerate (zip (ni, nj, ci, norb_f, nelec_f, self.fcisolvers)):
            nelec = self._get_nelec (s, ne)
            if getattr (c, 'ndim', 3) == 3: c = list (c)
            try:
                a, b = s.make_rdm1s (c, no, nelec)
            except AssertionError as e:
                print (type (c), np.asarray (c).shape, no, nelec, ix, type (s), getattr (s, 'weights', None))
                raise (e)
            except ValueError as e:
                print ("frag=",ix,"nroots=",s.nroots,"no=",no,"ne=",nelec,'c.shape=',np.asarray(c).shape)
                if isinstance (s, CSFFCISolver):
                    print ("smult=",s.smult,"ncsf=",s.transformer.ncsf)
                raise (e)
            dm1a[i:j,i:j] = a[:,:]
            dm1b[i:j,i:j] = b[:,:]
        return dm1a, dm1b

    def make_rdm1 (self, ci, norb_f, nelec_f, **kwargs):
        dm1a, dm1b = self.make_rdm1s (ci, norb_f, nelec_f, **kwargs)
        return dm1a + dm1b

    def make_rdm2 (self, ci, norb_f, nelec_f, **kwargs):
        dtype = np.array(ci).dtype
        norb = sum (norb_f)
        dm2 = np.zeros ([norb,]*4, dtype=dtype)
        nj = np.cumsum (norb_f)
        ni = nj - norb_f
        dm1a, dm1b = self.make_rdm1s (ci, norb_f, nelec_f, **kwargs)
        for i, j, c, no, ne, s in zip (ni, nj, ci, norb_f, nelec_f, self.fcisolvers):
            nelec = self._get_nelec (s, ne)
            dm2[i:j,i:j,i:j,i:j] = s.make_rdm2 (c, no, nelec)
        dm1 = dm1a + dm1b
        for (i,j), (k,l) in combinations (zip (ni, nj), 2):
            d1_ij, d1a_ij, d1b_ij = dm1[i:j,i:j], dm1a[i:j,i:j], dm1b[i:j,i:j]
            d1_kl, d1a_kl, d1b_kl = dm1[k:l,k:l], dm1a[k:l,k:l], dm1b[k:l,k:l]
            d2 = np.multiply.outer (d1_ij, d1_kl)
            dm2[i:j,i:j,k:l,k:l] = d2
            dm2[k:l,k:l,i:j,i:j] = d2.transpose (2,3,0,1)
            d2  = np.multiply.outer (d1a_ij, d1a_kl)
            d2 += np.multiply.outer (d1b_ij, d1b_kl)
            dm2[i:j,k:l,k:l,i:j] = -d2.transpose (0,2,3,1)
            dm2[k:l,i:j,i:j,k:l] = -d2.transpose (2,0,1,3)
        return dm2


class ImpureProductStateFCISolver (PBCProductStateFCISolver):
    r'''Minimize the energy of an impure state:

    E = \sum_n1 w_n1 \sum_n2 w_n2 \sum_n3 w_n3 ... <n1n2n3...|H|n1n2n3...>

    over orthonormal sets of CI vectors {nK} for fragment K.'''

    def __init__(self, fcisolvers, stdout=None, verbose=0, lroots=None, lweights=None, **kwargs):
        PBCProductStateFCISolver.__init__(self, fcisolvers, stdout=stdout, verbose=verbose, **kwargs)
        if lweights is None:
            if lroots is not None:
                lweights = []
                for lroot in lroots:
                    l = np.zeros (lroot)
                    l[0] = 1
                    lweights.append (l)
            else:
                lweights = [[.5,.5],]*len(fcisolvers)
        for ix, (fcisolver, weights) in enumerate (zip (self.fcisolvers, lweights)):
            if len (weights) > 1:
                self.fcisolvers[ix] = state_average_fcisolver (fcisolver, weights=weights)


class PBCTransSymmImpureProductStateFCISolver(ImpureProductStateFCISolver):
    '''
    Translation-symmetry adapted product-state solver.

    This class has currently been tested with one root per unit cell.  The
    implementation retains the generalized multi-root and state-averaged
    code paths.

    The full-fragment interface is assembled from a reference-cell value and
    its translation phases.  Only the reference fragment is optimized.

    '''

    trans_sym = True

    def __init__(self, fcisolvers, stdout=None, verbose=0, lroots=None,
                 lweights=None, ref_cell=0, phase_per_frag=None,
                 pack_h1=None, pack_h2=None, **kwargs):
        '''
        For documentation see above, the new args are:
        ref_cell: int
            Index of the reference fragment.  The reference fragment is the only
            fragment that is optimized, and the other fragments are generated by
            translation of the reference fragment.
        phase_per_frag: np.array or list of np.array or None
            Optional translation phases for each fragment. Generated from the overlap of the 
            orbitals of the reference fragment with the orbitals of each other fragment.
            If None, all phases are set to one
        pack_h1: callable or None
            This function will transform the one-electron integrals into a packed form.
        pack_h2: callable or None
            This function will transform the two-electron integrals into a packed form.
        '''
        super().__init__(fcisolvers, stdout=stdout, verbose=verbose, lroots=lroots,
                         lweights=lweights, **kwargs)
        # Checks:
        if not isinstance(ref_cell, (int, np.integer)):
            msg = f"ref_cell must be an integer, got {type(ref_cell)}"
            raise TypeError(msg)
        if not 0 <= ref_cell < len(self.fcisolvers):
            msg = f"ref_cell must be in [0, {len(self.fcisolvers)}); got {ref_cell}"
            raise ValueError(msg)
        self.ref_cell = int(ref_cell)
        self.phase_per_frag = self._normalize_phase_per_frag(phase_per_frag)
        if (pack_h1 is None) != (pack_h2 is None):
            raise ValueError("pack_h1 and pack_h2 must be provided together")
        if pack_h1 is not None and not callable(pack_h1):
            raise TypeError("pack_h1 must be callable")
        if pack_h2 is not None and not callable(pack_h2):
            raise TypeError("pack_h2 must be callable")
        self.pack_h1 = pack_h1
        self.pack_h2 = pack_h2

    def _normalize_phase_per_frag(self, phase_per_frag):
        '''
        Validate fragment phases and fix the reference-cell phase to one.
        '''
        nfrag = len(self.fcisolvers)
        if phase_per_frag is None:
            return np.ones(nfrag, dtype=np.complex128)
        dtype = np.result_type(phase_per_frag)
        phase_per_frag = np.asarray(phase_per_frag, dtype=dtype)

        if phase_per_frag.shape != (nfrag,):
            msg = f"phase_per_frag must have shape ({nfrag},), got {phase_per_frag.shape}"
            raise ValueError(msg)
        
        magnitudes = np.abs(phase_per_frag)

        # Extreme sanity checks to avoid division by zero or NaN propagation
        if np.any(~np.isfinite(magnitudes)) or np.any(magnitudes == 0):
            raise ValueError("phase_per_frag must contain finite nonzero phases")
        if np.max(np.abs(magnitudes - 1.0)) >= 1e-8:
            raise ValueError("phase_per_frag entries must have unit magnitude")

        phase_per_frag = phase_per_frag / magnitudes
        phase_per_frag *= phase_per_frag[self.ref_cell].conjugate()
        phase_per_frag[self.ref_cell] = 1.0

        return phase_per_frag

    def _pack_ci(self, ci):
        '''
        In short: CI_tot -> CI_ref
        Select the reference CI and remove its stored translation phase.
        By reference CI, I mean the CI vector corresponding to the reference
        fragment, which is the only one that is optimized.  The other fragments
        are generated by translation of the reference fragment.
        '''
        # Just for safety:
        if ci is None or ci[self.ref_cell] is None:
            return None
        ci_ref = np.asarray(ci[self.ref_cell])
        ci_ref /= self.phase_per_frag[self.ref_cell]
        return np.array(ci_ref, copy=True)

    def _unpack_cif(self, ci_ref, phases=None):
        '''
        In short: CI_ref -> CI_tot

        Expand a reference cell CI vector into one independent vector per fragment.
        Optional phase factors allow translated fragment CI vectors to use
        different global phases.
        args:
            ci_ref: np.ndarray
                Reference cell CI vector to be expanded.
            phases: array_like of shape (nfrag,) or None
                Optional phase factors for each fragment. If None, the stored
                ``phase_per_frag`` values are used.
        returns:
            ci_tot: list of np.ndarray
                List of CI vectors for each fragment, with the reference cell
                vector copied and optionally phase-shifted.
        '''

        # TODO: use the phase_per_frag attribute not phases.
        # TODO: add a check if the ci_ref is more than one root.

        nfrag = len(self.fcisolvers)

        if ci_ref is None:
            return [None for _ in range(nfrag)]

        if phases is None:
            phases = self.phase_per_frag
        else:
            phases = self._normalize_phase_per_frag(phases)

        ci_tot = [np.array(phases[ifrag] * ci_ref, copy=True) 
                  for ifrag in range(nfrag)]
        return ci_tot

    def _unpack_hfrag(self, h1eff_ref, h0eff_ref):
        '''
        In short: h1eff_ref, h0eff_ref -> h1eff, h0eff
        Assemble full-fragment effective Hamiltonians from the reference
        cell effective Hamiltonians
        '''
        # TODO: I forgot to add the phases here.
        
        nfrag = len(self.fcisolvers)
        h1eff = [np.array(h1eff_ref, copy=True) 
                 for _ in range(nfrag)]
        h0eff = [np.array(h0eff_ref, copy=True) 
                 for _ in range(nfrag)]
        return h1eff, h0eff

    def _make_ref_rdm1s(self, ci_ref, norb_f, nelec_f):
        '''
        Calculate spin-separated one-body RDMs for the reference cell.
        '''
        # TODO: add the phases here.
        dtype = np.result_type(ci_ref)
        ref = self.ref_cell
        norb_ref = norb_f[ref]
        solver_ref = self.fcisolvers[ref]
        nelec_ref = self._get_nelec(solver_ref, nelec_f[ref])
        ci_solver = ci_ref
        if getattr(ci_solver, 'ndim', 3) == 3:
            ci_solver = list(ci_solver)

        dm1a_ref, dm1b_ref = solver_ref.make_rdm1s(ci_solver, norb_ref, nelec_ref,)

        dm1a_ref = np.asarray(dm1a_ref, dtype=dtype)
        dm1b_ref = np.asarray(dm1b_ref, dtype=dtype)

        # Sanity check:
        assert dm1a_ref == dm1a_ref.conj().T, "dm1a_ref is not Hermitian"
        assert dm1b_ref == dm1b_ref.conj().T, "dm1b_ref is not Hermitian"
        nelec_ref = sum(nelec_ref)
        nelec_check = np.trace(dm1a_ref) + np.trace(dm1b_ref)
        assert nelec_ref - nelec_check < 1e-8, \
            f"nelec_ref ({nelec_ref}) does not match trace(dm1a_ref + dm1b_ref) ({nelec_check})"

        return dm1a_ref, dm1b_ref

    def _make_ref_rdm2(self, ci_ref, norb_f, nelec_f):
        '''
        Calculate the two-body RDM for the reference cell.
        '''
        ref = self.ref_cell
        norb_ref = norb_f[ref]
        dtype = np.result_type(ci_ref)
        solver_ref = self.fcisolvers[ref]
        nelec_ref = self._get_nelec(solver_ref, nelec_f[ref])
        ci_solver = ci_ref
        if getattr(ci_solver, 'ndim', 3) == 3:
            ci_solver = list(ci_solver)

        rdm2 = np.asarray(solver_ref.make_rdm2(ci_solver, norb_ref, nelec_ref), 
                          dtype=dtype)
        return rdm2

    def _unpack_rdm1s(self, dm1a_ref, dm1b_ref, norb_f, nelec_f):
        '''
        Assemble full one-body RDMs from the reference-cell blocks.
        '''
        # TODO: add the phases here.

        ref = self.ref_cell
        norb_ref = norb_f[ref]
        nelec_ref = self._get_nelec(self.fcisolvers[ref], nelec_f[ref],)
        norb = sum(norb_f)
        dtype = np.result_type(dm1a_ref.dtype, dm1b_ref.dtype)
        dm1a = np.zeros((norb, norb), dtype=dtype)
        dm1b = np.zeros((norb, norb), dtype=dtype)

        nj = np.cumsum(norb_f)
        ni = nj - norb_f
        for ifrag, (i, j, solver) in enumerate(zip(ni, nj, self.fcisolvers)):
            nelec = self._get_nelec(solver, nelec_f[ifrag])

            if norb_f[ifrag] != norb_ref \
                or tuple(nelec) != tuple(nelec_ref):
                msg = "translated fragments have inconsistent active spaces"
                raise ValueError(msg)

            dm1a[i:j, i:j] = dm1a_ref
            dm1b[i:j, i:j] = dm1b_ref
        return dm1a, dm1b

    def make_rdm1s(self, ci, norb_f, nelec_f, **kwargs):
        '''
        Assemble full spin-separated one-body RDMs from the reference.
        '''

        ci_ref = self._pack_ci(ci)
        dm1a_ref, dm1b_ref = self._make_ref_rdm1s(ci_ref, norb_f, nelec_f)
        rdm1a, rdm1b = self._unpack_rdm1s(dm1a_ref, dm1b_ref, norb_f, nelec_f)
        return rdm1a, rdm1b

    def make_rdm1(self, ci, norb_f, nelec_f, **kwargs):
        '''
        Assemble the full spin-summed one-body RDM from the reference.
        '''
        dm1a, dm1b = self.make_rdm1s(ci, norb_f, nelec_f, **kwargs)
        dm1 = dm1a + dm1b
        dm1 = 0.5 * (dm1 + dm1.conj().T)  # Ensure Hermiticity
        return dm1

    def make_rdm2(self, ci, norb_f, nelec_f, **kwargs):
        '''
        Assemble the full product-state two-body RDM from the reference.
        #TODO: add the doc_str
        '''

        ci_ref = self._pack_ci(ci)
        dm2_ref = self._make_ref_rdm2(ci_ref, norb_f, nelec_f,)
        dm1a_ref, dm1b_ref = self._make_ref_rdm1s(ci_ref, norb_f, nelec_f,)
        dm1a, dm1b = self._unpack_rdm1s(dm1a_ref, dm1b_ref, norb_f, nelec_f,)

        dm1 = dm1a + dm1b
        dm1 = 0.5 * (dm1 + dm1.conj().T)  # Ensure Hermiticity

        norb = sum(norb_f)
        dtype = np.result_type(dm2_ref.dtype, dm1.dtype)
        dm2 = np.zeros((norb,) * 4, dtype=dtype)
        nj = np.cumsum(norb_f)
        ni = nj - norb_f

        for i, j in zip(ni, nj):
            dm2[i:j, i:j, i:j, i:j] = dm2_ref

        for (i, j), (k, l) in combinations(zip(ni, nj), 2):
            d1_ij = dm1[i:j, i:j]
            d1a_ij = dm1a[i:j, i:j]
            d1b_ij = dm1b[i:j, i:j]
            d1_kl = dm1[k:l, k:l]
            d1a_kl = dm1a[k:l, k:l]
            d1b_kl = dm1b[k:l, k:l]

            d2 = np.multiply.outer(d1_ij, d1_kl)
            dm2[i:j, i:j, k:l, k:l] = d2
            dm2[k:l, k:l, i:j, i:j] = d2.transpose(2, 3, 0, 1)

            d2 = np.multiply.outer(d1a_ij, d1a_kl)
            d2 += np.multiply.outer(d1b_ij, d1b_kl)
            dm2[i:j, k:l, k:l, i:j] = -d2.transpose(0, 2, 3, 1)
            dm2[k:l, i:j, i:j, k:l] = -d2.transpose(2, 0, 1, 3)
        return dm2

    def energy_ref(self, h1_packed, h2_packed, ci_ref,
                   norb_f, nelec_f, ecore=0, **kwargs):
        '''
        Calculate the energy contribution associated with one cell.

        ``ecore`` is the total system core energy.  An equal ``1/ncell``
        share is included in the returned reference-cell energy.
        '''

        ncell = len(self.fcisolvers)
        norb_ref = norb_f[self.ref_cell]
        h1_packed = np.asarray(h1_packed)
        h2_packed = np.asarray(h2_packed)

        if h1_packed.shape == (ncell, norb_ref, norb_ref):
            h1_packed = np.stack([h1_packed, h1_packed], axis=0)

        expected_h1 = (2, ncell, norb_ref, norb_ref)
        expected_h2 = (ncell, ncell, ncell, norb_ref, norb_ref, norb_ref, norb_ref,)

        if h1_packed.shape != expected_h1:
            msg = (f"packed h1 must have shape {expected_h1}; "
                   f"got {h1_packed.shape}")
            raise ValueError(msg)
        
        if h2_packed.shape != expected_h2:
            msg = (f"packed h2 must have shape {expected_h2}; "
                   f"got {h2_packed.shape}")
            raise ValueError(msg)

        dm1a_ref, dm1b_ref = self._make_ref_rdm1s(ci_ref, norb_f, nelec_f,)
        dm1s_ref = np.stack([dm1a_ref, dm1b_ref], axis=0)
        dm1_ref = dm1a_ref + dm1b_ref
        dm2_ref = self._make_ref_rdm2(ci_ref, norb_f, nelec_f)

        e1 = np.einsum('spq,spq->', h1_packed[:, 0], dm1s_ref,)
        e2 = np.einsum('pqrs,pqrs->', h2_packed[0, 0, 0], dm2_ref,)

        for delta in range(1, ncell):
            e2 += np.einsum('pqrs,pq,rs->',
                            h2_packed[0, delta, delta], dm1_ref, dm1_ref,)
            for spin in range(2):
                e2 -= np.einsum('pqrs,ps,qr->',h2_packed[delta, delta, 0],
                                dm1s_ref[spin], dm1s_ref[spin],)

        return ecore / ncell + e1 + 0.5 * e2

    def energy_elec(self, h1, h2, ci, norb_f, nelec_f,
                    ecore=0, **kwargs):
        '''
        Calculate the total energy from the packed reference-cell energy.
        '''
        if self.pack_h1 is None:
            return super().energy_elec(h1, h2, ci, norb_f, nelec_f,
                                       ecore=ecore, **kwargs,)

        h1_packed = self.pack_h1(h1)
        h2_packed = self.pack_h2(h2)
        ci_ref = self._pack_ci(ci)
        energy_ref = self.energy_ref(h1_packed, h2_packed, ci_ref, norb_f, nelec_f,
                                     ecore=ecore, **kwargs,)
        ncells = len(self.fcisolvers)
        return ncells * energy_ref

    def _unpack_grad(self, grad_ref):
        '''
        Assemble the packed full-fragment gradient from the reference.
        '''
        ref_solver = self.fcisolvers[self.ref_cell]
        nroots = ref_solver.nroots
        external_size = nroots * ref_solver.transformer.ncsf
        internal_size = 0
        if nroots > 1 and getattr(ref_solver, 'weights', None) is not None:
            internal_size = nroots * (nroots - 1) // 2

        grad_ref = np.asarray(grad_ref).reshape(-1)
        if grad_ref.size != external_size + internal_size:
            raise ValueError("reference gradient has an inconsistent size")

        grad = []
        for phase, solver in zip(self.phase_per_frag, self.fcisolvers):
            solver_external_size = solver.nroots * solver.transformer.ncsf
            solver_internal_size = 0
            if (solver.nroots > 1
                    and getattr(solver, 'weights', None) is not None):
                solver_internal_size = solver.nroots * (solver.nroots - 1) // 2
            if (solver_external_size != external_size
                    or solver_internal_size != internal_size):
                msg = ("translated fragment gradients have inconsistent sizes: "
                       f"expected ({external_size}, {internal_size}), "
                       f"got ({solver_external_size}, {solver_internal_size})")
                raise ValueError(msg)

            grad_external = phase * grad_ref[:external_size]
            grad.append(grad_external)
            if internal_size:
                grad.append(np.array(grad_ref[external_size:], copy=True))
        return np.concatenate(grad)

    def get_init_guess(self, ci0, norb_f, nelec_f, h1, h2, nroots=None):
        '''
        Assemble the full initial guess from the reference-cell guess.
        '''
        ci_ref = self._pack_ci(ci0)
        ci_ref = self._get_ref_init_guess(ci_ref, norb_f, nelec_f, 
                                          h1, h2, nroots=nroots,)
        return self._unpack_cif(ci_ref)

    def project_hfrag(self, h1, h2, ci, norb_f, nelec_f,
                      ecore=0, dm1s=None, dm2=None, **kwargs):
        '''
        Assemble all effective fragment Hamiltonians from the reference
        cell.
        '''
        ci_ref = self._pack_ci(ci)
        h1eff_ref, h0eff_ref = self._project_ref_hfrag(
            h1, h2, ci_ref, norb_f, nelec_f, ecore=ecore,
            dm1s=dm1s, dm2=dm2, **kwargs,)
        h1eff, h0eff = self._unpack_hfrag(h1eff_ref, h0eff_ref)
        return h1eff, h0eff, self._unpack_cif(ci_ref)

    def _get_grad(self, h1eff, h2, ci, norb_f, nelec_f, orbsym=None,
                  **kwargs):
        '''
        Assemble the full CI gradient from the reference-cell gradient.
        '''
        h1eff_ref = h1eff[self.ref_cell]
        ci_ref = self._pack_ci(ci)
        grad_ref = self._get_ref_grad(
            h1eff_ref, h2, ci_ref, norb_f, nelec_f,
            orbsym=orbsym, **kwargs,)
        return self._unpack_grad(grad_ref)

    def _1shot_ref(self, h0eff_ref, h1eff_ref, h2, ci_ref,
                   norb_f, nelec_f, orbsym=None, **kwargs):
        '''
        Optimize only the reference-fragment CI vector once.
        '''
        ref = self.ref_cell
        i = sum(norb_f[:ref])
        j = i + norb_f[ref]
        norb = norb_f[ref]
        solver = self.fcisolvers[ref]
        nelec = self._get_nelec(solver, nelec_f[ref])
        h2_ref = h2[i:j, i:j, i:j, i:j]

        orbsym_ref = getattr(solver, 'orbsym', None)
        if orbsym is not None:
            orbsym_ref = orbsym[i:j]

        energy_ref, ci_ref = solver.kernel(h1eff_ref, h2_ref, norb, nelec, 
                                           ci0=ci_ref, ecore=h0eff_ref, 
                                           orbsym=orbsym_ref, **kwargs,)
        return energy_ref, np.array(ci_ref, copy=True)

    def kernel(self, h1, h2, norb_f, nelec_f, ecore=0, ci0=None,
               orbsym=None, conv_tol_grad=1e-4, conv_tol_self=1e-10,
               max_cycle_macro=50, serialfrag=False, **kwargs):
        '''
        Optimize the packed reference CI and expand it on return.
        '''
        log = self.log
        converged = False
        energy_ref = 0.0
        energy_sigma = 0.0
        ci_ref = self._pack_ci(ci0)
        solver_ref = self.fcisolvers[self.ref_cell]
        if max_cycle_macro < 1:
            raise ValueError("max_cycle_macro must be positive")

        log.info('Entering translation-symmetric reference-cell CI iteration')
        
        for it in range(max_cycle_macro):

            ci_ref = self._get_ref_init_guess(
                ci_ref, norb_f, nelec_f, h1, h2, **kwargs)

            h1eff_ref, h0eff_ref = self._project_ref_hfrag(
                h1, h2, ci_ref, norb_f, nelec_f,
                ecore=ecore, **kwargs,)

            grad_ref = self._get_ref_grad(h1eff_ref, h2, ci_ref, norb_f, nelec_f,
                orbsym=orbsym, **kwargs,)

            grad_max = np.amax(np.abs(grad_ref))

            solver_converged = np.all(
                np.asarray(getattr(solver_ref, 'converged', False)))

            log.info('Cycle %d: max ref grad = %e ; sigma = %e ; '
                'reference solver converged = %s',
                it, grad_max, energy_sigma, solver_converged,)
            if (grad_max < conv_tol_grad
                    and energy_sigma < conv_tol_self
                    and solver_converged and it > 0):
                converged = True
                break

            energy_ref, ci_ref = self._1shot_ref(
                h0eff_ref, h1eff_ref, h2, ci_ref,
                norb_f, nelec_f, orbsym=orbsym, **kwargs,
            )
            # All translated fragment energies are identical, so their spread
            # is zero by construction.
            energy_sigma = 0.0

        conv_str = ['NOT converged', 'converged'][int(converged)]
        log.info(
            'Translation-symmetric reference-cell CI iteration %s after %d '
            'cycles', conv_str, it + 1,
        )

        ci = self._unpack_cif(ci_ref)
        energy_elec = self.energy_elec(
            h1, h2, ci, norb_f, nelec_f,
            ecore=ecore, efinal=energy_ref, **kwargs,
        )
        return converged, energy_elec, ci

    def _project_ref_hfrag(self, h1, h2, ci_ref, norb_f, nelec_f,
                           ecore=0, phases=None, dm1s=None, dm2=None,
                           **kwargs):
        '''
        Project the Hamiltonian for the reference fragment.

        This initial implementation expands ``ci_ref`` to all translated
        fragments and reuses the parent dense projection.  A later packed
        implementation can replace this step without changing the reference-
        fragment interface.
        '''
        ci = self._unpack_cif(ci_ref, phases=phases)
        h1eff, h0eff, _ = super().project_hfrag(
            h1, h2, ci, norb_f, nelec_f, ecore=ecore,
            dm1s=dm1s, dm2=dm2, **kwargs,
        )
        return h1eff[self.ref_cell], h0eff[self.ref_cell]

    def _get_ref_grad(self, h1eff_ref, h2, ci_ref, norb_f, nelec_f,
                      orbsym=None, **kwargs):
        r'''
        Calculate the CI gradient for only the reference fragment.
        TODO: Add the equation to make it more clear.
        '''
        ref = self.ref_cell
        i = sum(norb_f[:ref])
        j = i + norb_f[ref]
        norb = norb_f[ref]
        solver = self.fcisolvers[ref]
        nelec = self._get_nelec(solver, nelec_f[ref])
        nroots = solver.nroots

        ndeta = cistring.num_strings(norb, nelec[0])
        ndetb = cistring.num_strings(norb, nelec[1])
        ci_ref = np.asarray(ci_ref).reshape(nroots, ndeta, ndetb)
        h2_ref = h2[i:j, i:j, i:j, i:j]
        h2eff_ref = solver.absorb_h1e(
            h1eff_ref, h2_ref, norb, nelec, 0.5,)
        hc = np.asarray([solver.contract_2e(h2eff_ref, root, norb, nelec) 
                         for root in ci_ref])
        chc = np.dot(ci_ref.reshape(nroots, -1).conj(), 
                     hc.reshape(nroots, -1).T)
        hc = hc - np.tensordot(chc, ci_ref, axes=1)

        if isinstance(solver, CSFFCISolver):
            hc_real = solver.transformer.vec_det2csf(
                hc.real, order='C', normalize=False,
            )
            hc_imag = solver.transformer.vec_det2csf(
                hc.imag, order='C', normalize=False,
            )
            hc_csf = hc_real.astype(h1eff_ref.dtype)
            hc_csf.real = hc_real
            hc_csf.imag = hc_imag
            hc = hc_csf

        assert hc.size == nroots * solver.transformer.ncsf
        grad = [hc.ravel()]
        if nroots > 1 and getattr(solver, 'weights', None) is not None:
            chc *= np.asarray(solver.weights)[:, None]
            chc -= chc.T
            grad.append(chc[np.tril_indices(nroots, k=-1)])
        return np.concatenate(grad)

    def _get_ref_init_guess(self, ci_ref, norb_f, nelec_f, h1, h2,
                            nroots=None):
        '''
        Get an initial CI vector for the reference fragment.

        A supplied reference CI vector is preserved. Otherwise, the initial
        guess is generated from the diagonal Hamiltonian of ``ref_cell``.
        Note: the ci_ref is just for the given reference cell, not for all the
        fragments.
        '''
        if ci_ref is not None:
            return np.array(ci_ref, copy=True)

        ref = self.ref_cell
        i = sum(norb_f[:ref])
        j = i + norb_f[ref]
        norb = norb_f[ref]
        solver = self.fcisolvers[ref]
        nelec = self._get_nelec(solver, nelec_f[ref])
        solver.norb = norb
        solver.nelec = nelec

        if h1.ndim < 3:
            h1 = np.stack([h1, h1], axis=0)
        h1_ref = h1[:, i:j, i:j]
        h2_ref = h2[i:j, i:j, i:j, i:j]

        solver.check_transformer_cache()
        if nroots is None:
            nroots = solver.nroots
        nroots = min(nroots, solver.transformer.ncsf)
        hdiag = solver.make_hdiag_csf(h1_ref, h2_ref, norb, nelec)
        ci_ref = solver.get_init_guess(norb, nelec, nroots, hdiag)
        ndeta = cistring.num_strings(norb, nelec[0])
        ndetb = cistring.num_strings(norb, nelec[1])
        return np.array(ci_ref, copy=True).reshape(nroots, ndeta, ndetb)
