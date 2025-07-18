import sys
import numpy as np
import functools
from scipy import linalg
from pyscf.fci import cistring
from pyscf import fci, lib
from pyscf.fci.direct_nosym import contract_1e as contract_1e_nosym
from pyscf.fci.direct_spin1 import _unpack_nelec
from pyscf.fci.spin_op import contract_ss, spin_square
from pyscf.data import nist
from itertools import combinations
from mrh.my_pyscf.mcscf import soc_int as soc_int
from mrh.my_pyscf.lassi import citools
from mrh.my_pyscf.lassi import dms as lassi_dms
from mrh.my_pyscf.fci.csf import unpack_h1e_cs

def memcheck (las, ci, soc=None):
    '''Check if the system has enough memory to run these functions! ONLY checks
    if the CI vectors can be stored in memory!!!'''
    nfrags = len (ci)
    nroots = len (ci[0])
    assert (all ([len (c) == nroots for c in ci]))
    lroots_fr = np.array ([[1 if c.ndim<3 else c.shape[0]
                            for c in ci_r]
                           for ci_r in ci])
    lroots_r = np.prod (lroots_fr, axis=0)
    nelec_frs = np.array ([[list (_unpack_nelec (fcibox._get_nelec (solver, nelecas)))
                            for solver in fcibox.fcisolvers]
                           for fcibox, nelecas in zip (las.fciboxes, las.nelecas_sub)])
    nelec_rs = np.unique (nelec_frs.sum (0), axis=0)
    ndet_spinfree = max ([cistring.num_strings (las.ncas, na) 
                          *cistring.num_strings (las.ncas, nb)
                          for na, nb in nelec_rs])
    ndet_soc = max ([cistring.num_strings (2*las.ncas, nelec) for nelec in nelec_rs.sum (1)])
    nbytes_per_sfvec = ndet_spinfree * np.dtype (float).itemsize 
    nbytes_per_sovec = ndet_soc * np.dtype (complex).itemsize
    # 2 vectors: the ket (from a generator), and op|ket>
    # for SOC, the generator also briefly stores the spinfree version but this
    # should always be smaller
    if soc:
        nbytes = 2*nbytes_per_sfvec
    else:
        nbytes = 2*nbytes_per_sfvec
    # memory load of ci_dp vectors
    nbytes += sum ([np.prod ([float (c[iroot].size) for c in ci])
                    * np.amax ([c[iroot].dtype.itemsize for c in ci])
                    for iroot in range (nroots)])
    safety_factor = 1.2
    mem = nbytes * safety_factor / 1e6
    max_memory = las.max_memory - lib.current_memory ()[0]
    lib.logger.info (las,
        "LASSI op_o0 memory check: {} MB needed of {} MB available ({} MB max)".format (mem,\
        max_memory, las.max_memory))
    return mem < max_memory

def civec_spinless_repr_generator (ci0_r, norb, nelec_r):
    '''Put CI vectors in the spinless representation; i.e., map
        norb -> 2 * norb
        (neleca, nelecb) -> (neleca+nelecb, 0)
    This permits linear combinations of CI vectors with different
    M == neleca-nelecb at the price of higher memory cost. This function
    does NOT change the datatype.

    Args:
        ci0_r: sequence or generator of ndarray of length nprods
            CAS-CI vectors in the spin-pure representation
        norb: integer
            Number of orbitals
        nelec_r: sequence of tuple of length (2)
            (neleca, nelecb) for each element of ci0_r

    Returns:
        ci1_r_gen: callable that returns a generator of length nprods
            generates spinless CAS-CI vectors
        ss2spinless: callable
            Put a CAS-CI vector in the spinless representation
            Args:
                ci0: ndarray
                    CAS-CI vector
                ne: tuple of length 2
                    neleca, nelecb of target Hilbert space
            Returns:
                ci1: ndarray
                    spinless CAS-CI vector
        spinless2ss: callable
            Perform the reverse operation on a spinless CAS-CI vector
            Args:
                ci2: ndarray
                    spinless CAS-CI vector
                ne: tuple of length 2
                    neleca, nelecb target Hilbert space
            Returns:
                ci3: ndarray
                    CAS-CI vector of ci2 in the (neleca, nelecb) Hilbert space
    '''
    nelec_r_tot = [sum (n) for n in nelec_r]
    if len (set (nelec_r_tot)) > 1:
        raise NotImplementedError ("Different particle-number subspaces")
    nelec = nelec_r_tot[0]
    addrs = {}
    ndet_sp = {}
    for ne in set (nelec_r):
        neleca, nelecb = _unpack_nelec (ne)
        ndeta = cistring.num_strings (norb, neleca)
        ndetb = cistring.num_strings (norb, nelecb)
        strsa = cistring.addrs2str (norb, neleca, list(range(ndeta)))
        strsb = cistring.addrs2str (norb, nelecb, list(range(ndetb)))
        strs = np.add.outer (strsa, np.left_shift (strsb, norb)).ravel ()
        addrs[ne] = cistring.strs2addr (2*norb, nelec, strs)
        ndet_sp[ne] = tuple((ndeta,ndetb))
    strs = strsa = strsb = None
    ndet = cistring.num_strings (2*norb, nelec)
    nstates = len (nelec_r)
    def ss2spinless (ci0, ne, buf=None):
        if buf is None:
            ci1 = np.empty (ndet, dtype=ci0.dtype)
        else:
            ci1 = np.asarray (buf).flat[:ndet]
        ci1[:] = 0.0
        ci1[addrs[ne]] = ci0[:,:].ravel ()
        neleca, nelecb = _unpack_nelec (ne)
        if abs(neleca*nelecb)%2: ci1[:] *= -1
        # Sign comes from changing representation:
        # ... a2' a1' a0' ... b2' b1' b0' |vac>
        # ->
        # ... b2' b1' b0' .. a2' a1' a0' |vac>
        # i.e., strictly decreasing from left to right
        # (the ordinality of spin-down is conventionally greater than spin-up)
        return ci1[:,None]
    def spinless2ss (ci2, ne):
        ''' Generate the spin-separated CI vector in a particular M
        Hilbert space from a spinless CI vector '''
        ci3 = ci2[addrs[ne]].reshape (ndet_sp[ne])
        neleca, nelecb = _unpack_nelec (ne)
        if abs(neleca*nelecb)%2: ci3[:] *= -1
        return ci3
    def ci1_r_gen (buf=None):
        if callable (ci0_r):
            ci0_r_gen = ci0_r ()
        else:
            ci0_r_gen = (c for c in ci0_r)
        for ci0, ne in zip (ci0_r_gen, nelec_r):
            # Doing this in two lines saves memory: ci0 is overwritten
            ci0 = ss2spinless (ci0, ne)
            yield ci0
    return ci1_r_gen, ss2spinless, spinless2ss

def civec_spinless_repr (ci0_r, norb, nelec_r):
    '''Put CI vectors in the spinless representation; i.e., map
        norb -> 2 * norb
        (neleca, nelecb) -> (neleca+nelecb, 0)
    This permits linear combinations of CI vectors with different
    M == neleca-nelecb at the price of higher memory cost. This function
    does NOT change the datatype.

    Args:
        ci0_r: sequence or generator of ndarray of length nprods
            CAS-CI vectors in the spin-pure representation
        norb: integer
            Number of orbitals
        nelec_r: sequence of tuple of length (2)
            (neleca, nelecb) for each element of ci0_r

    Returns:
        ci1_r: ndarray of shape (nprods, ndet_spinless)
            spinless CAS-CI vectors
    '''
    ci1_r_gen, _, _ = civec_spinless_repr_generator (ci0_r, norb, nelec_r)
    ci1_r = np.stack ([x.copy () for x in ci1_r_gen ()], axis=0)
    return ci1_r

def addr_outer_product (norb_f, nelec_f):
    '''Build index arrays for reshaping a direct product of LAS CI
    vectors into the appropriate orbital ordering for a CAS CI vector'''
    norb = sum (norb_f)
    nelec = sum (nelec_f)
    # Must skip over cases where there are no electrons of a specific spin in a particular subspace
    norbrange = np.cumsum (norb_f)
    addrs = []
    for i in range (0, len (norbrange)):
        irange = range (norbrange[i]-norb_f[i], norbrange[i])
        new_addrs = cistring.sub_addrs (norb, nelec, irange, nelec_f[i]) if nelec_f[i] else []
        if len (addrs) == 0:
            addrs = new_addrs
        elif len (new_addrs) > 0:
            addrs = np.intersect1d (addrs, new_addrs)
    if not len (addrs): addrs=[0] # No beta electrons edge case
    return addrs

def _ci_outer_product (ci_f, norb_f, nelec_f):
    '''Compute outer-product CI vector for one space table from fragment LAS CI vectors.
    See "ci_outer_product"'''
    nfrags = len (norb_f)
    neleca_f = [ne[0] for ne in nelec_f]
    nelecb_f = [ne[1] for ne in nelec_f]
    lroots_f = [1 if ci.ndim<3 else ci.shape[0] for ci in ci_f]
    nprods = np.prod (lroots_f)
    shape_f = [(lroots, cistring.num_strings (norb, neleca), cistring.num_strings (norb, nelecb))
              for lroots, norb, neleca, nelecb in zip (lroots_f, norb_f, neleca_f, nelecb_f)]
    addrs_a = addr_outer_product (norb_f, neleca_f)
    addrs_b = addr_outer_product (norb_f, nelecb_f)
    idx = np.ix_(addrs_a,addrs_b)
    addrs_a = addrs_b = None
    neleca = sum (neleca_f)
    nelecb = sum (nelecb_f)
    nelec = tuple ((neleca, nelecb))
    ndet_a = cistring.num_strings (sum (norb_f), neleca)
    ndet_b = cistring.num_strings (sum (norb_f), nelecb)
    ci_dp = ci_f[-1].reshape (shape_f[-1])
    for ci_r, shape in zip (ci_f[-2::-1], shape_f[-2::-1]):
        lroots, ndeta, ndetb = ci_dp.shape
        ci_dp = np.multiply.outer (ci_dp, ci_r.reshape (shape))
        ci_dp = ci_dp.transpose (0,3,1,4,2,5).reshape (
            lroots*shape[0], ndeta*shape[1], ndetb*shape[2]
        )
    #norm_dp = linalg.norm (ci_dp.reshape (ci_dp.shape[0],-1), axis=1)
    #ci_dp /= norm_dp[:,None,None]
    def gen_ci_dp (buf=None):
        if buf is None:
            ci = np.empty ((ndet_a,ndet_b), dtype=ci_f[-1].dtype)
        else:
            ci = np.asarray (buf.flat[:ndet_a*ndet_b]).reshape (ndet_a, ndet_b)
        #ci_dp = ci_f[-1].reshape (shape_f[-1])
        #for ci_r, shape in zip (ci_f[-2::-1], shape_f[-2::-1]):
        #    lroots, ndeta, ndetb = ci_dp.shape
        #    ci_dp = np.multiply.outer (ci_dp, ci_r.reshape (shape))
        #    ci_dp = ci_dp.transpose (0,3,1,4,2,5).reshape (
        #        lroots*shape[0], ndeta*shape[1], ndetb*shape[2]
        #    )
        #norm_dp = linalg.norm (ci_dp.reshape (ci_dp.shape[0],-1), axis=1)
        #ci_dp /= norm_dp[:,None,None]
        for vec in ci_dp:
            ci[:,:] = 0.0
            ci[idx] = vec[:,:]
            yield ci
    def dotter (c1, nelec1, skip=None):
        if nelec1 != nelec: return np.zeros (nprods)
        if skip is None: return np.dot (ci_dp.reshape (nprods, -1),
                                        c1[idx].ravel ())
        try:
            skip = set (skip)
        except TypeError as e:
            skip = set ([skip])
        c1_dp = c1[idx][None,:]
        skipdims = 1
        for ifrag, (ci_r, shape) in enumerate (zip (ci_f, shape_f)):
            new_shape = [x for s,t in zip (c1_dp.shape[skipdims:],shape[1:]) for x in (s//t, t)]
            new_shape = list(c1_dp.shape[0:skipdims]) + new_shape
            c1_dp = c1_dp.reshape (*new_shape)
            if ifrag in skip:
                dimorder = [0,skipdims+1,skipdims+3,] + list (range(1, skipdims+1)) + [skipdims+2,]
                c1_dp = c1_dp.transpose (*dimorder)
                skipdims += 2
            else:
                c1_dp = np.tensordot (ci_r.reshape (shape), c1_dp,
                                      axes=((1,2),(skipdims+1,skipdims+3)))
                new_shape = [c1_dp.shape[0]*c1_dp.shape[1],] + list (c1_dp.shape[2:])
                c1_dp = c1_dp.reshape (*new_shape)
        c1_dp = c1_dp.reshape (c1_dp.shape[0:skipdims])
        return c1_dp
    return gen_ci_dp, nprods, dotter

def ci_outer_product_generator (ci_fr, norb_f, nelec_fr):
    '''Compute outer-product CI vectors from fragment LAS CI vectors and return
    result as a generator

    Args:
        ci_fr : nested list of shape (nfrags, nroots)
            Contains CI vectors; element [i,j] is ndarray of shape
            (lroots[i,j],ndeta[i,j],ndetb[i,j])
        norb_f : list of length (nfrags)
            Number of orbitals in each fragment
        nelec_fr : ndarray-like of shape (nfrags, nroots, 2)
            Number of spin-up and spin-down electrons in each fragment
            and root

    Returns:
        ci_r_gen : callable that returns a generator of length (nprods)
            Generates all direct-product CAS CI vectors
        nelec_p : list of length (nprods) of tuple of length 2
            (neleca, nelecb) for each product state
        dotter : callable
            Performs the dot product in the outer product basis
            on a CAS CI vector, without explicitly constructing
            any direct-product CAS CI vectors (again).
            Args:
                ket : ndarray
                    contains CAS-CI vector
                nelec_ket : tuple of length 2
                    neleca, nelecb
            Kwargs:
                spinless2ss : callable
                    Converts ci back from the spinless representation
                    into the (neleca,nelecb) representation. Takes c1
                    and nelec1 and returns a new ci vector
                iket : integer
                    Used to filter zero sectors when passed with oporder
                oporder: integer
                    Used to filter zero sectors when passed with oporder
                    ket is identified as product number iket, acted on by
                    some operator of up to order oporder. Any rootspace
                    which differs by more than oporder electron hops
                    therefore has zero projection onto ket.
            Returns:
                ndarray of length (nprods)
                    Expansion coefficients for c1 in terms of direct-
                    product states of ci_fr
    '''

    norb = sum (norb_f)
    ndet = max ([cistring.num_strings (norb, ne[0]) * cistring.num_strings (norb, ne[1])
                for ne in np.sum (nelec_fr, axis=0)])
    gen_ci_r = []
    nelec_p = []
    dotter_r = []
    space_p = []
    for space in range (len (ci_fr[0])):
        ci_f = [ci[space] for ci in ci_fr]
        nelec_f = [nelec[space] for nelec in nelec_fr]
        nelec = (sum ([ne[0] for ne in nelec_f]), sum ([ne[1] for ne in nelec_f]))
        gen_ci, nprods, dotter = _ci_outer_product (ci_f, norb_f, nelec_f)
        gen_ci_r.append (gen_ci)
        nelec_p.extend ([nelec,]*nprods)
        space_p.extend ([space,]*nprods)
        dotter_r.append ([dotter, nelec, nprods])
    def ci_r_gen (buf=None):
        if buf is None:
            buf1 = np.empty (ndet, dtype=ci_fr[-1][0].dtype)
        else:
            buf1 = np.asarray (buf.flat[:ndet])
        for gen_ci in gen_ci_r:
            for x in gen_ci (buf=buf1):
                yield x
    def dotter (ket, nelec_ket, spinless2ss=None, iket=None, oporder=None, skip=None):
        vec = []
        if callable (spinless2ss):
            parse_nelec = lambda nelec: sum(nelec)
        else:
            parse_nelec = lambda nelec: nelec
            spinless2ss = lambda vec, ne: vec
        filtbra = lambda *args: False
        if iket is not None and oporder is not None:
            iket = space_p[iket]
            nelec_f_ket = nelec_fr[:,iket]
            def filtbra (ibra):
                nelec_f_bra = nelec_fr[:,ibra]
                nhop_f = nelec_f_bra-nelec_f_ket
                return np.sum (np.abs(nhop_f))>2*oporder

        for ibra, (dot, nelec_bra, nprods) in enumerate (dotter_r):
            if parse_nelec (nelec_bra) != parse_nelec (nelec_ket) or filtbra (ibra):
                vec.append (np.zeros (nprods))
                continue
            vec.append (dot (spinless2ss (ket, nelec_bra), nelec_bra, skip=skip))
        if skip is None: return np.concatenate (vec)
        else: return vec
    return ci_r_gen, nelec_p, dotter

def ci_outer_product (ci_fr, norb_f, nelec_fr):
    '''Compute outer-product CI vectors from fragment LAS CI vectors.

    Args:
        ci_fr : nested list of shape (nfrags, nroots)
            Contains CI vectors; element [i,j] is ndarray of shape
            (lroots[i,j],ndeta[i,j],ndetb[i,j])
        norb_f : list of length (nfrags)
            Number of orbitals in each fragment
        nelec_fr : ndarray-like of shape (nfrags, nroots, 2)
            Number of spin-up and spin-down electrons in each fragment
            and root

    Returns:
        ci_r : list of length (nroots)
            Contains full CAS CI vector
        nelec_r : list of length (nroots) of tuple of length 2
            (neleca, nelecb) for each product state
    '''
    ci_r_gen, nelec_r, _ = ci_outer_product_generator (ci_fr, norb_f, nelec_fr)
    ci_r = [x.copy () for x in ci_r_gen ()]
    return ci_r, nelec_r

#def si_soc (las, h1, ci, nelec, norb):
#
#### function adapted from github.com/hczhai/fci-siso/blob/master/fcisiso.py ###
#
##    au2cm = nist.HARTREE2J / nist.PLANCK / nist.LIGHT_SPEED_SI * 1e-2
#    nroots = len(ci)
#    hsiso = np.zeros((nroots, nroots), dtype=complex)
#    ncas = las.ncas
#    hso_m1 = h1[ncas:2*ncas,0:ncas]
#    hso_p1 = h1[0:ncas,ncas:2*ncas]
#    hso_ze = (h1[0:ncas,0:ncas] - h1[ncas:2*ncas,ncas:2*ncas])/2 
#
#    for istate, (ici, inelec) in enumerate(zip(ci, nelec)):
#        for jstate, (jci, jnelec) in enumerate(zip(ci, nelec)):
#            if jstate > istate:
#                continue
#
#            tp1 = lassi_dms.make_trans(1, ici, jci, norb, inelec, jnelec)
#            tze = lassi_dms.make_trans(0, ici, jci, norb, inelec, jnelec)
#            tm1 = lassi_dms.make_trans(-1, ici, jci, norb, inelec, jnelec)
#
#            if tp1.shape == ():
#                tp1 = np.zeros((ncas,ncas))
#            if tze.shape == ():
#                tze = np.zeros((ncas,ncas))
#            if tm1.shape == ():
#                tm1 = np.zeros((ncas,ncas))
#
#            somat = np.einsum('ri, ir ->', tm1, hso_m1)
#            somat += np.einsum('ri, ir ->', tp1, hso_p1)
#            #somat = somat/2
#            somat += np.einsum('ri, ir ->', tze, hso_ze)
#
#            hsiso[jstate, istate] = somat
#            if istate!= jstate:
#                hsiso[istate, jstate] = somat.conj()
##            somat *= au2cm
#
#    #heigso, hvecso = np.linalg.eigh(hsiso)
#
#    return hsiso

def get_ovlp (ci_fr, norb_f, nelec_frs, rootidx=None):
    if rootidx is not None:
        ci_fr = [[c[iroot] for iroot in rootidx] for c in ci_fr]
        nelec_frs = nelec_frs[:,rootidx,:]
    ci_r_generator, nelec_r, dotter = ci_outer_product_generator (ci_fr, norb_f, nelec_frs)
    ndim = len (nelec_r)
    ovlp = np.zeros ((ndim, ndim))
    for i, ket in zip(range(ndim), ci_r_generator ()):
        nelec_ket = nelec_r[i]
        ovlp[i,:] = dotter (ket, nelec_ket, iket=i, oporder=0)
    return ovlp

def get_orth_basis (ci_fr, norb_f, nelec_frs, _get_ovlp=get_ovlp):
    nfrags, nroots = nelec_frs.shape[:2]
    unique, uniq_idx, inverse, cnts = np.unique (nelec_frs, axis=1, return_index=True,
                                                 return_inverse=True, return_counts=True)
    if not np.count_nonzero (cnts>1):
        def raw2orth (rawarr):
            return rawarr
        def orth2raw (ortharr):
            return ortharr
        return raw2orth, orth2raw
    lroots_fr = np.array ([[1 if c.ndim<3 else c.shape[0]
                            for c in ci_r]
                           for ci_r in ci_fr])
    nprods_r = np.prod (lroots_fr, axis=0)
    offs1 = np.cumsum (nprods_r)
    offs0 = offs1 - nprods_r
    uniq_prod_idx = []
    for i in uniq_idx[cnts==1]: uniq_prod_idx.extend (list(range(offs0[i],offs1[i])))
    manifolds_prod_idx = []
    manifolds_xmat = []
    nuniq_prod = north = len (uniq_prod_idx)
    for manifold_idx in np.where (cnts>1)[0]:
        manifold = np.where (inverse==manifold_idx)[0]
        manifold_prod_idx = []
        for i in manifold: manifold_prod_idx.extend (list(range(offs0[i],offs1[i])))
        manifolds_prod_idx.append (manifold_prod_idx)
        ovlp = _get_ovlp (ci_fr, norb_f, nelec_frs, rootidx=manifold)
        xmat = canonical_orth_(ovlp, thr=LINDEP_THRESH)
        north += xmat.shape[1]
        manifolds_xmat.append (xmat)

    nraw = offs1[-1]
    def raw2orth (rawarr):
        col_shape = rawarr.shape[1:]
        orth_shape = [north,] + list (col_shape)
        ortharr = np.zeros (orth_shape, dtype=rawarr.dtype)
        ortharr[:nuniq_prod] = rawarr[uniq_prod_idx]
        i = nuniq_prod
        for prod_idx, xmat in zip (manifolds_prod_idx, manifolds_xmat):
            j = i + xmat.shape[1]
            ortharr[i:j] = np.tensordot (xmat.T, rawarr[prod_idx], axes=1)
            i = j
        return ortharr

    def orth2raw (ortharr):
        col_shape = ortharr.shape[1:]
        raw_shape = [nraw,] + list (col_shape)
        rawarr = np.zeros (raw_shape, dtype=ortharr.dtype)
        rawarr[uniq_prod_idx] = ortharr[:nuniq_prod]
        i = nuniq_prod
        for prod_idx, xmat in zip (manifolds_prod_idx, manifolds_xmat):
            j = i + xmat.shape[1]
            rawarr[prod_idx] = np.tensordot (xmat.conj (), ortharr[i:j], axes=1)
            i = j
        return rawarr

    return raw2orth, orth2raw

def ham (las, h1, h2, ci_fr, nelec_frs, soc=0, orbsym=None, wfnsym=None, **kwargs):
    '''Build LAS state interaction Hamiltonian, S2, and ovlp matrices

    Args:
        las : instance of class LASSCF
        h1 : ndarray of shape (ncas, ncas)
            Spin-orbit-free one-body CAS Hamiltonian
        h2 : ndarray of shape (ncas, ncas, ncas, ncas)
            Spin-orbit-free two-body CAS Hamiltonian
        ci_fr : nested list of shape (nfrags, nroots)
            Contains CI vectors; element [i,j] is ndarray of shape
            (ndeta[i,j],ndetb[i,j])
        nelec_frs : ndarray of shape (nfrags,nroots,2)
            Number of electrons of each spin in each rootspace in each
            fragment

    Kwargs:
        soc : integer
            Order of spin-orbit coupling included in the Hamiltonian
        orbsym : list of int of length (ncas)
            Irrep ID for each orbital
        wfnsym : int
            Irrep ID for target matrix block

    Returns:
        ham_eff : square ndarray of length (ndim)
            Spin-orbit-free Hamiltonian in state-interaction basis
        s2_eff : square ndarray of length (ndim)
            S2 operator matrix in state-interaction basis
        ovlp_eff : square ndarray of length (ndim)
            Overlap matrix in state-interaction basis
        _get_ovlp : callable with kwarg rootidx
            Produce the overlap matrix between model states in a set of rootspaces,
            identified by ndarray or list "rootidx"
    '''
    if soc>1:
        raise NotImplementedError ("Two-electron spin-orbit coupling")
    mol = las.mol
    norb_f = las.ncas_sub
    norb = norb_ss = sum (norb_f)
    nroots = len (ci_fr[0])

    # The function below is the main workhorse of this whole implementation
    ci_r_generator, nelec_r, dotter = ci_outer_product_generator (ci_fr, norb_f, nelec_frs)
    ndim = len(nelec_r)
    nelec_r_spinless = [tuple((n[0] + n[1], 0)) for n in nelec_r]
    nelec_r_ss = nelec_r
    if not len (set (nelec_r_spinless)) == 1:
        raise NotImplementedError ("States with different numbers of electrons")
    # Hamiltonian may be complex
    h1_re = h1.real
    h2_re = h2.real
    h1_im = None
    if soc:
        h1_im = h1.imag
        h2_re = np.zeros ([2,norb,]*4, dtype=h1_re.dtype)
        h2_re[0,:,0,:,0,:,0,:] = h2[:]
        h2_re[1,:,1,:,0,:,0,:] = h2[:]
        h2_re[0,:,0,:,1,:,1,:] = h2[:]
        h2_re[1,:,1,:,1,:,1,:] = h2[:]
        h2_re = h2_re.reshape ([2*norb,]*4)
        ss2spinless, spinless2ss = civec_spinless_repr_generator (
            ci_r_generator, norb, nelec_r)[1:3]
        nelec_r = nelec_r_spinless
        norb = 2 * norb
        if orbsym is not None: orbsym *= 2
    else:
        spinless2ss = None
        ss2spinless = lambda *args: args[0]

    solver = fci.solver (mol, symm=(wfnsym is not None)).set (orbsym=orbsym, wfnsym=wfnsym)
    h1_re_c, h1_re_s = h1_re, 0
    if h1_re.ndim > 2:
        h1_re_c, h1_re_s = unpack_h1e_cs (h1_re)
    def contract_h_re (c, nel):
        h2eff = solver.absorb_h1e (h1_re_c, h2_re, norb, nel, 0.5)
        return solver.contract_2e (h2eff, c, norb, nel)
    if h1_im is not None:
        def contract_h (c, nel):
            hc = contract_h_re (c, nel)
            hc = hc + 1j*contract_1e_nosym (h1_im, c, norb, nel)
            return hc
    else:
        contract_h = contract_h_re

    ham_eff = np.zeros ((ndim, ndim), dtype=h1.dtype)
    ovlp_eff = np.zeros ((ndim, ndim))
    s2_eff = np.zeros ((ndim,ndim))
    for i, ket in zip(range(ndim), ci_r_generator ()):
        nelec_ket = nelec_r_ss[i]
        ovlp_eff[i,:] = dotter (ket, nelec_ket, iket=i, oporder=0)
        s2ket = contract_ss (ket, norb_ss, nelec_ket)
        s2_eff[i,:] = dotter (s2ket, nelec_ket, iket=i, oporder=2)
        s2ket, ket = None, ss2spinless (ket, nelec_ket)
        nelec_ket = nelec_r[i]
        hket = contract_h (ket, nelec_ket)
        ket = None
        ham_eff[i,:] = dotter (hket, nelec_ket, spinless2ss=spinless2ss, iket=i, oporder=2)
    
    _get_ovlp = functools.partial (get_ovlp, ci_fr, norb_f, nelec_frs)
    #raw2orth = citools.get_orth_basis (ci_fr, norb_f, nelec_frs, _get_ovlp=_get_ovlp)
    return ham_eff, s2_eff, ovlp_eff, _get_ovlp #raw2orth

def contract_ham_ci (las, h1, h2, ci_fr, nelec_frs, si_bra=None, si_ket=None, ci_fr_bra=None,
                     nelec_frs_bra=None, h0=0, soc=0, sum_bra=False, orbsym=None, wfnsym=None,
                     add_transpose=False, accum=None, **kwargs):
    '''Evaluate the action of the state interaction Hamiltonian on a set of ket CI vectors,
    projected onto a basis of bra CI vectors, leaving one fragment of the bra uncontracted.

    Args:
        las : instance of class LASSCF
        h1 : ndarray of shape (ncas, ncas)
            Spin-orbit-free one-body CAS Hamiltonian
        h2 : ndarray of shape (ncas, ncas, ncas, ncas)
            Spin-orbit-free two-body CAS Hamiltonian
        ci_fr : nested list of shape (nfrags, nroots)
            Contains CI vectors for the ket; element [i,j] is ndarray of shape
            (ndeta_ket[i,j],ndetb_ket[i,j])
        nelec_frs : ndarray of shape (nfrags, nroots, 2)
            Number of electrons of each spin in each rootspace in each
            fragment

    Kwargs:
        si_bra : ndarray of shape (ndim_bra, *)
            SI vectors for the bra. If provided, the p dimension on the return object is contracted
        si_ket : ndarray of shape (ndim_ket, *)
            SI vectors for the bra. If provided, the q dimension on the return object is contracted
        ci_fr_bra : nested list of shape (nfrags, nroots_bra)
            Contains CI vectors for the bra; element [i,j] is ndarray of shape
            (ndeta_bra[i,j],ndetb_bra[i,j]). Defaults to ci_fr.
        nelec_frs_bra : ndarray of shape (nfrags, nroots_bra, 2)
            Number of electrons of each spin in each fragment for the bra vectors.
            Defaults to nelec_frs.
        soc : integer
            Order of spin-orbit coupling included in the Hamiltonian
        h0 : float
            Constant term in the Hamiltonian
        sum_bra : logical
            Currently does nothing whatsoever (TODO)
        orbsym : list of int of length (ncas)
            Irrep ID for each orbital
        wfnsym : int
            Irrep ID for target matrix block

    Returns:
        hket_fr_pabq : nested list of shape (nfrags, nroots_bra)
            Element i,j is an ndarray of shape (ndim_bra//ci_fr_bra[i][j].shape[0],
            ndeta_bra[i,j],ndetb_bra[i,j],ndim_ket). 
    '''
    if add_transpose:
        assert (ci_fr_bra is None)
        assert (nelec_frs_bra is None)
        hket_fr = contract_ham_ci (las, h1, h2, ci_fr, nelec_frs, si_bra=si_bra, si_ket=si_ket,
                                   h0=h0, soc=soc, sum_bra=sum_bra, orbsym=orbsym, wfnsym=wfnsym,
                                   add_transpose=False, **kwargs)
        hketT_fr = contract_ham_ci (las, h1, h2, ci_fr, nelec_frs, si_bra=si_ket, si_ket=si_bra,
                                    h0=h0, soc=soc, sum_bra=sum_bra, orbsym=orbsym, wfnsym=wfnsym,
                                    add_transpose=False, **kwargs)
        for f, hketT_r in enumerate (hketT_fr):
            for r, hketT in enumerate (hketT_r):
                hket_fr[f][r] += hketT
        return hket_fr
    ci_fr_ket = ci_fr
    nelec_frs_ket = nelec_frs
    if ci_fr_bra is None: ci_fr_bra = ci_fr_ket
    if nelec_frs_bra is None: nelec_frs_bra = nelec_frs_ket
    if soc>1:
        raise NotImplementedError ("Two-electron spin-orbit coupling")
    mol = las.mol
    norb_f = las.ncas_sub
    norb = norb_ss = sum (norb_f)
    nfrags = len (norb_f)
    nroots_ket = len (ci_fr_ket[0])
    nroots_bra = len (ci_fr_bra[0])

    # The function below is the main workhorse of this whole implementation
    ci_r_ket_gen, nelec_r_ket, dotter_ket = ci_outer_product_generator (ci_fr_ket, norb_f, nelec_frs_ket)
    ci_r_bra_gen, nelec_r_bra, dotter_bra = ci_outer_product_generator (ci_fr_bra, norb_f, nelec_frs_bra)
    ndim_ket = len(nelec_r_ket)
    ndim_bra = len(nelec_r_bra)
    nelec_r_ket_ss = nelec_r_ket
    nelec_r_spinless = [tuple((n[0] + n[1], 0)) for n in nelec_r_ket]
    if not len (set (nelec_r_spinless)) == 1:
        raise NotImplementedError ("States with different numbers of electrons")
    # Hamiltonian may be complex
    h1_re = h1.real
    h2_re = h2.real
    h1_im = None
    if soc:
        h1_im = h1.imag
        h2_re = np.zeros ([2,norb,]*4, dtype=h1_re.dtype)
        h2_re[0,:,0,:,0,:,0,:] = h2[:]
        h2_re[1,:,1,:,0,:,0,:] = h2[:]
        h2_re[0,:,0,:,1,:,1,:] = h2[:]
        h2_re[1,:,1,:,1,:,1,:] = h2[:]
        h2_re = h2_re.reshape ([2*norb,]*4)
        ss2spinless_ket, spinless2ss_ket = civec_spinless_repr_generator (
            ci_r_ket_gen, norb, nelec_r_ket)[1:3]
        ss2spinless_bra, spinless2ss_bra = civec_spinless_repr_generator (
            ci_r_bra_gen, norb, nelec_r_ket)[1:3]
        nelec_r_ket = nelec_r_spinless
        nelec_r_bra = [tuple ((n[0] + n[1], 0)) for n in nelec_r_bra]
        norb = 2 * norb
        if orbsym is not None: orbsym *= 2
    else:
        spinless2ss_ket = spinless2ss_bra = None
        ss2spinless_ket = ss2spinless_bra = lambda *args: args[0]

    solver = fci.solver (mol, symm=(wfnsym is not None)).set (orbsym=orbsym, wfnsym=wfnsym)
    h1_re_c, h1_re_s = h1_re, 0
    if h1_re.ndim > 2:
        h1_re_c, h1_re_s = unpack_h1e_cs (h1_re)
    def contract_h_re (c, nel):
        h2eff = solver.absorb_h1e (h1_re_c, h2_re, norb, nel, 0.5)
        hc = solver.contract_2e (h2eff, c, norb, nel)
        return hc + h0*c
    if h1_im is not None:
        def contract_h (c, nel):
            hc = contract_h_re (c, nel)
            hc = hc + 1j*contract_1e_nosym (h1_im, c, norb, nel)
            return hc + h0*c
    else:
        contract_h = contract_h_re

    # <p|H|q>, but the space of p is partially uncontracted
    hket_fr_pabq = [[[] for i in range (nroots_bra)] for j in range (nfrags)]
    for i, ket in zip(range(ndim_ket), ci_r_ket_gen ()):
        nelec_ket = nelec_r_ket_ss[i]
        ket = ss2spinless_ket (ket, nelec_ket)
        nelec_ket = nelec_r_ket[i]
        hket = contract_h (ket, nelec_ket)
        for ifrag in range (nfrags):
            hket_r = dotter_bra (hket, nelec_ket, spinless2ss=spinless2ss_bra, skip=ifrag)
            for ibra in range (nroots_bra):
                hket_fr_pabq[ifrag][ibra].append (hket_r[ibra])
    for ifrag in range (nfrags):
        for ibra in range (nroots_bra):
            hket_fr_pabq[ifrag][ibra] = np.stack (hket_fr_pabq[ifrag][ibra], axis=-1)
    return citools.hci_dot_sivecs (hket_fr_pabq, si_bra, si_ket, citools.get_lroots (ci_fr_bra))

def make_stdm12s (las, ci_fr, nelec_frs, orbsym=None, wfnsym=None):
    '''Build LAS state interaction transition density matrices

    Args:
        las : instance of class LASSCF
        ci_fr : nested list of shape (nfrags, nroots)
            Contains CI vectors; element [i,j] is ndarray of shape
            (ndeta[i,j],ndetb[i,j])
        nelec_frs : ndarray of shape (nfrags,nroots,2)
            Number of electrons of each spin in each rootspace in each
            fragment

    Kwargs:
        orbsym : list of int of length (ncas)
            Irrep ID for each orbital
        wfnsym : int
            Irrep ID for target matrix block

    Returns:
        stdm1s : ndarray of shape (nroots,2,ncas,ncas,nroots) OR (nroots,2*ncas,2*ncas,nroots)
            One-body transition density matrices between LAS states.
            If states with different spin projections (i.e., neleca-nelecb) are present, the 4d
            spinorbital array is returned. Otherwise, the 5d spatial-orbital array is returned.
        stdm2s : ndarray of shape [nroots,]+ [2,ncas,ncas,]*2 + [nroots,]
            Two-body transition density matrices between LAS states
    '''
    mol = las.mol
    norb_f = las.ncas_sub
    norb = sum (norb_f) 
    ci_r_generator, nelec_r, dotter = ci_outer_product_generator (ci_fr, norb_f, nelec_frs)
    nelec_r_spinless = [tuple((n[0] + n[1], 0)) for n in nelec_r]
    nroots = len (nelec_r)
    if not len (set (nelec_r_spinless)) == 1:
        raise NotImplementedError ("States with different numbers of electrons")
    spin_pure = len (set (nelec_r)) == 1

    dtype = ci_fr[-1][0].dtype
    if not spin_pure:
        # Map to "spinless electrons": 
        ci_r_generator = civec_spinless_repr_generator (ci_r_generator, norb, nelec_r)[0]
        nelec_r = nelec_r_spinless
        norb = 2 * norb
        if orbsym is not None: orbsym *= 2

    solver = fci.solver (mol).set (orbsym=orbsym, wfnsym=wfnsym)
    stdm1s = np.zeros ((nroots, nroots, 2, norb, norb),
        dtype=dtype).transpose (0,2,3,4,1)
    stdm2s = np.zeros ((nroots, nroots, 2, norb, norb, 2, norb, norb),
        dtype=dtype).transpose (0,2,3,4,5,6,7,1)
    for i, (ci, ne) in enumerate (zip (ci_r_generator (), nelec_r)):
        rdm1s, rdm2s = solver.make_rdm12s (ci, norb, ne)
        stdm1s[i,0,:,:,i] = rdm1s[0]
        stdm1s[i,1,:,:,i] = rdm1s[1]
        stdm2s[i,0,:,:,0,:,:,i] = rdm2s[0]
        stdm2s[i,0,:,:,1,:,:,i] = rdm2s[1]
        stdm2s[i,1,:,:,0,:,:,i] = rdm2s[1].transpose (2,3,0,1)
        stdm2s[i,1,:,:,1,:,:,i] = rdm2s[2]

    spin_sector_offset = np.zeros ((nroots,nroots))
    for i, (ci_bra, ne_bra) in enumerate (zip (ci_r_generator (), nelec_r)):
        for j, (ci_ket, ne_ket) in enumerate (zip (ci_r_generator (), nelec_r)):
            if j==i: break
            M_bra = ne_bra[1] - ne_bra[0]
            M_ket = ne_ket[0] - ne_ket[1]
            N_bra = sum (ne_bra)
            N_ket = sum (ne_ket)
            if ne_bra == ne_ket:
                tdm1s, tdm2s = solver.trans_rdm12s (ci_bra, ci_ket, norb, ne_bra)
                stdm1s[i,0,:,:,j] = tdm1s[0]
                stdm1s[i,1,:,:,j] = tdm1s[1]
                stdm1s[j,0,:,:,i] = tdm1s[0].T
                stdm1s[j,1,:,:,i] = tdm1s[1].T
                for spin, tdm2 in enumerate (tdm2s):
                    p = spin // 2
                    q = spin % 2
                    stdm2s[i,p,:,:,q,:,:,j] = tdm2
                    stdm2s[j,p,:,:,q,:,:,i] = tdm2.transpose (1,0,3,2)

    if not spin_pure: # cleanup the "spinless mapping"
        stdm1s = stdm1s[:,0,:,:,:]
        # TODO: 2e- spin-orbit coupling support in caller
        n = norb // 2
        stdm2s_ = np.zeros ((nroots, nroots, 2, n, n, 2, n, n),
            dtype=dtype).transpose (0,2,3,4,5,6,7,1)
        stdm2s_[:,0,:,:,0,:,:,:] = stdm2s[:,0,:n,:n,0,:n,:n,:]
        stdm2s_[:,0,:,:,1,:,:,:] = stdm2s[:,0,:n,:n,0,n:,n:,:]
        stdm2s_[:,1,:,:,0,:,:,:] = stdm2s[:,0,n:,n:,0,:n,:n,:]
        stdm2s_[:,1,:,:,1,:,:,:] = stdm2s[:,0,n:,n:,0,n:,n:,:]
        stdm2s = stdm2s_

    return stdm1s, stdm2s 

def root_trans_rdm12s (las, ci_fr, nelec_frs, si_bra, si_ket, ix, orbsym=None, wfnsym=None,
                       **kwargs):
    '''Build LAS state interaction reduced transition density matrices for 1 final pair of
    LASSI eigenstates.

    Args:
        las : instance of class LASSCF
        ci_fr : nested list of shape (nfrags, nroots)
            Contains CI vectors; element [i,j] is ndarray of shape
            (ndeta[i,j],ndetb[i,j])
        nelec_frs : ndarray of shape (nfrags,nroots,2)
            Number of electrons of each spin in each rootspace in each
            fragment
        si_bra : ndarray of shape (nroots, nroots)
            Unitary matrix defining final LASSI states in terms of
            non-interacting LAS states for the bra
        si_ket : ndarray of shape (nroots, nroots)
            Unitary matrix defining final LASSI states in terms of
            non-interacting LAS states for the ket
        ix : integer
            Index of columns of si_bra and si_ket to use

    Kwargs:
        orbsym : list of int of length (ncas)
            Irrep ID for each orbital
        wfnsym : int
            Irrep ID for target matrix block

    Returns:
        rdm1s : ndarray of shape (2, ncas, ncas) OR (2*ncas, 2*ncas)
            One-body transition density matrices between LAS states
            If states with different spin projections (i.e., neleca-nelecb) are present, the 2d
            spinorbital array is returned. Otherwise, the 3d spatial-orbital array is returned.
        rdm2s : ndarray of length (2, ncas, ncas, 2, ncas, ncas)
            Two-body transition density matrices between LAS states
    '''
    mol = las.mol
    norb_f = las.ncas_sub
    ci_r_gen, nelec_r, dotter = ci_outer_product_generator (ci_fr, norb_f, nelec_frs)
    nelec_r_spinless = [tuple((n[0] + n[1], 0)) for n in nelec_r]
    nroots = len (nelec_r)
    norb = sum (norb_f)
    if not len (set (nelec_r_spinless)) == 1:
        raise NotImplementedError ("States with different numbers of electrons")
    spin_pure = len (set (nelec_r)) == 1
    if not spin_pure:
        # Map to "spinless electrons": 
        ci_r_gen = civec_spinless_repr_generator (ci_r_gen, norb, nelec_r)[0]
        nelec_r = nelec_r_spinless
        norb = 2 * norb
        if orbsym is not None: orbsym *= 2
    nelec_r = nelec_r[0]

    ndeta = cistring.num_strings (norb, nelec_r[0])
    ndetb = cistring.num_strings (norb, nelec_r[1])
    cib_r = np.zeros ((ndeta,ndetb), dtype=si_bra.dtype)
    cik_r = np.zeros ((ndeta,ndetb), dtype=si_ket.dtype)
    for coeffb, coeffk, c in zip (si_bra[:,ix], si_ket[:,ix], ci_r_gen ()):
        try:
            cib_r[:,:] += coeffb * c
            cik_r[:,:] += coeffk * c
        except ValueError as err:
            print (cib_r.shape, cik_r.shape, ndeta, ndetb, c.shape)
            raise (err)
    cib_r_real = np.ascontiguousarray (cib_r.real)
    cik_r_real = np.ascontiguousarray (cik_r.real)
    is_bra_complex = np.iscomplexobj (cib_r)
    is_ket_complex = np.iscomplexobj (cik_r)
    is_complex = is_bra_complex or is_ket_complex
    dtype = cik_r.dtype
    if is_complex and is_bra_complex: dtype = cib_r.dtype
    rdm1s = np.zeros ((2, norb, norb), dtype=dtype)
    rdm2s = np.zeros ((2, norb, norb, 2, norb, norb), dtype=dtype)
    if is_bra_complex:
        cib_r_imag = np.ascontiguousarray (cib_r.imag)
    else:
        cib_r_imag = [0,]*nroots
    if is_ket_complex:
        cik_r_imag = np.ascontiguousarray (cik_r.imag)
    else:
        cik_r_imag = [0,]*nroots
        #solver = fci.solver (mol).set (orbsym=orbsym, wfnsym=wfnsym)

    solver = fci.solver (mol).set (orbsym=orbsym, wfnsym=wfnsym)
    d1s, d2s = solver.trans_rdm12s (cib_r_real, cik_r_real, norb, nelec_r)
    if is_complex:
        d1s = np.asarray (d1s, dtype=complex)
        d2s = np.asarray (d2s, dtype=complex)
        d1s2, d2s2 = solver.trans_rdm12s (cib_r_imag, cik_r_imag, norb, nelec_r)
        d1s += np.asarray (d1s2)
        d2s += np.asarray (d2s2)
        d1s2, d2s2 = solver.trans_rdm12s (cib_r_real, cik_r_imag, norb, nelec_r)
        d1s += 1j * np.asarray (d1s2)
        d2s -= 1j * np.asarray (d2s2)
        d1s2, d2s2 = solver.trans_rdm12s (cib_r_imag, cik_r_real, norb, nelec_r)
        d1s -= 1j * np.asarray (d1s2)
        d2s += 1j * np.asarray (d2s2)
    rdm1s[0,:,:] = d1s[0]
    rdm1s[1,:,:] = d1s[1]
    rdm2s[0,:,:,0,:,:] = d2s[0]
    rdm2s[0,:,:,1,:,:] = d2s[1]
    rdm2s[1,:,:,0,:,:] = d2s[2]
    rdm2s[1,:,:,1,:,:] = d2s[3]

    if not spin_pure: # cleanup the "spinless mapping"
        rdm1s = rdm1s[0,:,:]
        # TODO: 2e- SOC
        n = norb // 2
        rdm2s_ = np.zeros ((2, n, n, 2, n, n), dtype=dtype)
        rdm2s_[0,:,:,0,:,:] = rdm2s[0,:n,:n,0,:n,:n]
        rdm2s_[0,:,:,1,:,:] = rdm2s[0,:n,:n,0,n:,n:]
        rdm2s_[1,:,:,0,:,:] = rdm2s[0,n:,n:,0,:n,:n]
        rdm2s_[1,:,:,1,:,:] = rdm2s[0,n:,n:,0,n:,n:]
        rdm2s = rdm2s_

    return rdm1s, rdm2s

def root_make_rdm12s (las, ci_fr, nelec_frs, si, ix, orbsym=None, wfnsym=None, **kwargs):
    '''Build LAS state interaction reduced density matrices for 1 final
    LASSI eigenstate.

    Args:
        las : instance of class LASSCF
        ci_fr : nested list of shape (nfrags, nroots)
            Contains CI vectors; element [i,j] is ndarray of shape
            (ndeta[i,j],ndetb[i,j])
        nelec_frs : ndarray of shape (nfrags,nroots,2)
            Number of electrons of each spin in each rootspace in each
            fragment
        si : ndarray of shape (nroots, nroots)
            Unitary matrix defining final LASSI states in terms of
            non-interacting LAS states
        ix : integer
            Index of column of si to use

    Kwargs:
        orbsym : list of int of length (ncas)
            Irrep ID for each orbital
        wfnsym : int
            Irrep ID for target matrix block

    Returns:
        rdm1s : ndarray of shape (2, ncas, ncas) OR (2*ncas, 2*ncas)
            One-body transition density matrices between LAS states
            If states with different spin projections (i.e., neleca-nelecb) are present, the 2d
            spinorbital array is returned. Otherwise, the 3d spatial-orbital array is returned.
        rdm2s : ndarray of length (2, ncas, ncas, 2, ncas, ncas)
            Two-body transition density matrices between LAS states
    '''
    return root_trans_rdm12s (las, ci_fr, nelec_frs, si, si, ix, orbsym=None, wfnsym=None,
                              **kwargs)

def roots_make_rdm12s (las, ci_fr, nelec_frs, si, orbsym=None, wfnsym=None, **kwargs):
    '''Build LAS state interaction reduced density matrices for final
    LASSI eigenstates.

    Args:
        las : instance of class LASSCF
        ci_fr : nested list of shape (nfrags, nroots)
            Contains CI vectors; element [i,j] is ndarray of shape
            (ndeta[i,j],ndetb[i,j])
        nelec_frs : ndarray of shape (nfrags,nroots,2)
            Number of electrons of each spin in each rootspace in each
            fragment
        si : ndarray of shape (nroots, nroots)
            Unitary matrix defining final LASSI states in terms of
            non-interacting LAS states

    Kwargs:
        orbsym : list of int of length (ncas)
            Irrep ID for each orbital
        wfnsym : int
            Irrep ID for target matrix block

    Returns:
        rdm1s : ndarray of shape (nroots, 2, ncas, ncas) OR (nroots, 2*ncas, 2*ncas)
            One-body transition density matrices between LAS states
            If states with different spin projections (i.e., neleca-nelecb) are present, the 3d
            spinorbital array is returned. Otherwise, the 4d spatial-orbital array is returned.
        rdm2s : ndarray of length (nroots, 2, ncas, ncas, 2, ncas, ncas)
            Two-body transition density matrices between LAS states
    '''
    rdm1s = []
    rdm2s = []
    for ix in range (si.shape[1]):
        d1, d2 = root_make_rdm12s (las, ci_fr, nelec_frs, si, ix, orbsym=orbsym, wfnsym=wfnsym,
                                   **kwargs)
        rdm1s.append (d1)
        rdm2s.append (d2)
    return np.stack (rdm1s, axis=0), np.stack (rdm2s, axis=0)

def roots_trans_rdm12s (las, ci_fr, nelec_frs, si_bra, si_ket, orbsym=None, wfnsym=None, **kwargs):
    '''Build LAS state interaction reduced transition density matrices for final
    LASSI eigenstates.

    Args:
        las : instance of class LASSCF
        ci_fr : nested list of shape (nfrags, nroots)
            Contains CI vectors; element [i,j] is ndarray of shape
            (ndeta[i,j],ndetb[i,j])
        nelec_frs : ndarray of shape (nfrags,nroots,2)
            Number of electrons of each spin in each rootspace in each
            fragment
        si_bra : ndarray of shape (nroots, nroots)
            Unitary matrix defining final LASSI states in terms of
            non-interacting LAS states for the bra
        si_ket : ndarray of shape (nroots, nroots)
            Unitary matrix defining final LASSI states in terms of
            non-interacting LAS states for the ket

    Kwargs:
        orbsym : list of int of length (ncas)
            Irrep ID for each orbital
        wfnsym : int
            Irrep ID for target matrix block

    Returns:
        rdm1s : ndarray of shape (nroots, 2, ncas, ncas) OR (nroots, 2*ncas, 2*ncas)
            One-body transition density matrices between LAS states
            If states with different spin projections (i.e., neleca-nelecb) are present, the 3d
            spinorbital array is returned. Otherwise, the 4d spatial-orbital array is returned.
        rdm2s : ndarray of length (nroots, 2, ncas, ncas, 2, ncas, ncas)
            Two-body transition density matrices between LAS states
    '''
    rdm1s = []
    rdm2s = []
    assert (si_bra.shape == si_ket.shape)
    for ix in range (si_bra.shape[1]):
        d1, d2 = root_trans_rdm12s (las, ci_fr, nelec_frs, si_bra, si_ket, ix, orbsym=orbsym,
                                    wfnsym=wfnsym, **kwargs)
        rdm1s.append (d1)
        rdm2s.append (d2)
    return np.stack (rdm1s, axis=0), np.stack (rdm2s, axis=0)

if __name__ == '__main__':
    from pyscf import scf, lib
    from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
    import os
    class cd:
        """Context manager for changing the current working directory"""
        def __init__(self, newPath):
            self.newPath = os.path.expanduser(newPath)

        def __enter__(self):
            self.savedPath = os.getcwd()
            os.chdir(self.newPath)

        def __exit__(self, etype, value, traceback):
            os.chdir(self.savedPath)
    from mrh.examples.lasscf.c2h6n4.c2h6n4_struct import structure as struct
    with cd ("/home/herme068/gits/mrh/examples/lasscf/c2h6n4"):
        mol = struct (2.0, 2.0, '6-31g', symmetry=False)
    mol.verbose = lib.logger.DEBUG
    mol.output = 'sa_lasscf_slow_ham.log'
    mol.build ()
    mf = scf.RHF (mol).run ()
    tol = 1e-6 if len (sys.argv) < 2 else float (sys.argv[1])
    las = LASSCF (mf, (4,4), (4,4)).set (conv_tol_grad = tol)
    mo = las.localize_init_guess ((list(range(3)),list(range(9,12))), mo_coeff=mf.mo_coeff)
    las.state_average_(weights = [0.5, 0.5], spins=[[0,0],[2,-2]])
    h2eff_sub, veff = las.kernel (mo)[-2:]
    e_states = las.e_states

    ncore, ncas, nocc = las.ncore, las.ncas, las.ncore + las.ncas
    mo_coeff = las.mo_coeff
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]
    e0 = las._scf.energy_nuc () + 2 * (((las._scf.get_hcore () + veff.c/2) @ mo_core) * mo_core).sum () 
    h1 = mo_cas.conj ().T @ (las._scf.get_hcore () + veff.c) @ mo_cas
    h2 = h2eff_sub[ncore:nocc].reshape (ncas*ncas, ncas * (ncas+1) // 2)
    h2 = lib.numpy_helper.unpack_tril (h2).reshape (ncas, ncas, ncas, ncas)
    nelec_fr = []
    for fcibox, nelec in zip (las.fciboxes, las.nelecas_sub):
        ne = sum (nelec)
        nelec_fr.append ([_unpack_nelec (fcibox._get_nelec (solver, ne)) for solver in fcibox.fcisolvers])
    ham_eff = slow_ham (las.mol, h1, h2, las.ci, las.ncas_sub, nelec_fr)[0]
    print (las.converged, e_states - (e0 + np.diag (ham_eff)))

gen_contract_op_si_hdiag = functools.partial (citools._fake_gen_contract_op_si_hdiag, ham)


