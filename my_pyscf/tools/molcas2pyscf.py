import h5py
import numpy as np
from pyscf.symm.param import IRREP_ID_MOLPRO as IRREP_ID_MOLCAS
from pyscf.symm.geom import detect_symm, get_subgroup
from pyscf.symm import label_orb_symm
from pyscf.lib import logger
from pyscf import gto
from pyscf.tools import molden

def get_mo_from_h5 (mol, h5fname, symmetry=None):
    ''' Get MO vectors for a pyscf molecule from an h5 file written by OpenMolcas

        Args:
            mol : instance gto.mole
                Must be in the same point group as the OpenMolcas calculation,
                or set the symmetry argument
            h5fname : str
                Path to an h5 file generated by OpenMolcas containing (at least) groups
                'BASIS_FUNCTION_IDS', 'DESYM_MATRIX', 'MO_VECTORS', and 'MO_ENERGIES'
            symmetry : str
                Point group of the calculation in OpenMolcas. 
                If None, use mol.groupname instead

        Returns:
            mo_coeff : ndarray of shape (nao_nr, nao_nr)
    '''

    if symmetry is not None:
        mol = mol.copy()
        mol.build(symmetry=symmetry)
    idx_irrep = np.argsort(IRREP_ID_MOLCAS[mol.groupname])
    try:
        nmo_irrep = [mol.symm_orb[ir].shape[1] for ir in idx_irrep]
    except:
        assert (not mol.symmetry), mol.symmetry

    with h5py.File (h5fname, 'r') as f:
        try:
            molcas_basids = f['DESYM_BASIS_FUNCTION_IDS'][()]
            molcas_usymm = f['DESYM_MATRIX'][()]
        except KeyError:
            assert (not mol.symmetry), "Can't find desym_ data; mol.symmetry = {}".format (mol.symmetry)
            molcas_basids = f['BASIS_FUNCTION_IDS'][()]
        molcas_coeff = f['MO_VECTORS'][()]
        mo_energy = f['MO_ENERGIES'][()]
        mo_occ = f['MO_OCCUPATIONS'][()]

    usymm_list = []
    if mol.symmetry:
        uuu = [m_ir * mol.nao_nr () for m_ir in nmo_irrep]
        usymm_irrep_offset = [0] + [sum (uuu[:i+1]) for i in range (len (uuu)-1)]
        uuu = [m_ir * m_ir for m_ir in nmo_irrep]
        coeff_irrep_offset = [0] + [sum (uuu[:i+1]) for i in range (len (uuu)-1)]
        orb_irrep_offset = [0] + [sum (nmo_irrep[:i+1]) for i in range (len (nmo_irrep)-1)]
        mo_coeff = np.zeros ((mol.nao_nr (), mol.nao_nr ()), dtype=np.float_)
        for m_ir, orb_off, usymm_off, coeff_off in zip (nmo_irrep, orb_irrep_offset, usymm_irrep_offset, coeff_irrep_offset):
            usymm = molcas_usymm[usymm_off:usymm_off+(m_ir*mol.nao_nr ())].reshape (m_ir, mol.nao_nr ()).T
            usymm_list.append (usymm)
            coeff = molcas_coeff[coeff_off:coeff_off+(m_ir*m_ir)].reshape (m_ir, m_ir).T
            mo_coeff[:,orb_off:orb_off+m_ir] = np.dot (usymm, coeff)
    else:
        assert (molcas_coeff.shape == (mol.nao_nr ()**2,)), 'mo_vectors.shape = {} but {} AOs'.format (
            molcas_coeff.shape, mol.nao_nr ())
        mo_coeff = molcas_coeff.reshape (mol.nao_nr (), mol.nao_nr ()).T
        
    idx_ao = []
    for (c, n, l, m) in molcas_basids:
        # 0-index atom list in PySCF, 1-index atom list in Molcas
        c -= 1
        # Actual principal quantum number in PySCF, 1-indexed list in Molcas
        n += l
        # l=1, ml=(-1,0,1) is (x,y,z) in PySCF, (y,z,x) in Molcas
        if l == 1:
            m = m - 2 if m > 0 else m + 1
        idx_ao.append (mol.search_ao_nr (c, l, m, n))
    idx_ao = np.argsort (np.asarray (idx_ao))
    mo_coeff = mo_coeff[idx_ao,:]
    if mol.symmetry: usymm_list = [u[idx_ao,:] for u in usymm_list]

    # 'mergesort' keeps degenerate or active-space orbitals in the provided order!
    #  idx_ene = np.argsort (mo_energy, kind='mergesort')

    # modified by Dayou: sort by mo_occ first, then mo_energy
    sort_key = np.min(mo_energy) * 100000 * mo_occ + mo_energy
    idx_ene = np.argsort(sort_key, kind='mergesort')
    mo_coeff = mo_coeff[:,idx_ene]
    mo_occ = mo_occ[idx_ene]
    mo_energy = mo_energy[idx_ene]

    nao, nmo = mo_coeff.shape
    s0 = mol.intor_symmetric ('int1e_ovlp')
    err = np.amax (np.abs ((mo_coeff.conj ().T @ s0 @ mo_coeff) - np.eye (nmo)))
    print ("Are {} h5 MOs orthonormal in AO basis? {}".format (mol.output, err)) 
    err = np.amax (np.abs ((mo_coeff.conj ().T @ mo_coeff) - np.eye (nmo)))
    print ("Are {} h5 MOs in an orthonormal basis? {}".format (mol.output, err)) 
    if len (usymm_list):
        usymm = np.concatenate (usymm_list, axis=-1)
        err = np.amax (np.abs ((usymm.conj ().T @ usymm) - np.eye (nmo)))
        print ("Are {} h5 symm_orb in an orthonormal basis? {}".format (
            mol.output, err))
        usymm = np.concatenate (mol.symm_orb, axis=-1)
        err = np.amax (np.abs ((usymm.conj ().T @ usymm) - np.eye (nmo)))
        print ("Are {} mol symm_orb in an orthonormal basis? {}".format (
            mol.output, err))
    if mol.verbose > logger.INFO:
        fname = str (mol.output)[:-4] + "_h5debug.molden"
        molden.from_mo (mol, fname, mo_coeff, occ=mo_occ, ene=mo_energy)
    if mol.symmetry:
        try:
            orbsym = label_orb_symm (mol, mol.irrep_name, usymm_list, mo_coeff)
        except ValueError as e:
            print (e)
            with h5py.File (h5fname, 'r') as f:
                

    return mo_coeff

def get_mol_from_h5 (h5fname, **kwargs):
    ''' Build a gto.mole object from an h5 file written by OpenMolcas 

        Args:
            h5fname : str
                Path to an h5 file generated by OpenMolcas containing (at least) groups
                'BASIS_FUNCTION_IDS', 'DESYM_MATRIX', 'MO_VECTORS', and 'MO_ENERGIES'
        Kwargs:
            any of the kwargs of gto.M other than atom, basis, symmetry, and unit

        Returns:
            mol: gto.mole object
    '''

    my_symmetry = True
    with h5py.File (h5fname, 'r') as f:
        try:
            symbs = np.asarray (f['DESYM_CENTER_LABELS']).astype ('|U10')
            carts = np.asarray (f['DESYM_CENTER_COORDINATES'])
        except KeyError:
            my_symmetry = False
            symbs = np.asarray (f['CENTER_LABELS']).astype ('|U10')
            carts = np.asarray (f['CENTER_COORDINATES'])
        primids = np.asarray (f['PRIMITIVE_IDS'])
        prims = np.asarray (f['PRIMITIVES'])

    symbs = [s.split (' ')[0] for s in symbs]
    my_atom = [[s, xyz.tolist ()] for s, xyz in zip (symbs, carts)]
    if my_symmetry:
        symm, charge_center, axes = detect_symm (my_atom)
        my_symmetry, axes = get_subgroup (symm, axes)
    
    my_basis = {}
    for idx, symb in enumerate (symbs):
        if symb in my_basis:
            continue
        thisatom = (primids[:,0] == idx+1)
        atomid = primids[thisatom,1:]
        atomprim = prims[thisatom,:]
        my_basis[symb] = []
        for l in range (np.max (atomid[:,0]+1)):
            thisl = (atomid[:,0] == l)
            lid = atomid[thisl,1:]
            lprim = atomprim[thisl,:]
            thisshl = lid[:,0]==1
            a = lprim[thisshl,0]
            c = lprim[thisshl,1][:,np.newaxis]
            idx_shl = 0
            for n in range (2,np.max (lid)+1):
                thisshl = lid[:,0]==n
                if np.count_nonzero (thisshl) == len (a) and np.all (a == lprim[thisshl,0]):
                    c = np.append (c, lprim[thisshl,1][:,np.newaxis], axis=1)
                else:
                    my_basis[symb].append ([l, np.insert (c, 0, a, axis=1).tolist ()])
                    a = lprim[thisshl,0]
                    c = lprim[thisshl,1][:,np.newaxis]
            my_basis[symb].append ([l, *np.insert (c, 0, a, axis=1).tolist ()])

    for check in ('atom', 'basis', 'symmetry', 'unit'):
        if check in kwargs:
            kwargs.pop (check)
    return gto.M (atom = my_atom, basis = my_basis, symmetry = my_symmetry, unit = 'Bohr', **kwargs)
    #return my_atom, my_basis, my_symmetry


