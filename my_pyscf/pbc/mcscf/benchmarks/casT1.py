import numpy as np
import os
from functools import reduce
from pyscf import ao2mo, fci
from pyscf import lib
from pyscf.pbc import scf, df
from pyscf.pbc import gto as pgto

'''
Aim: Generate the electronic Hamiltonian, and use that to solve
the FCI problem. For the Gamma point you can directly compare it with the mcscf.CASCI, if both the
energies are match that means I somewhat understand how to construct the CAS Hamiltonian.
'''

def get_xyz(nU=1, d= 2.47):
    '''
    Generate atomic coordinates for a system with nU unit cells.
    args:
        nU: Number of unit cells
        d:  lattice vector of UC
    '''
    coords = [
    ("C", -0.5892731038,  0.3262391909,  0.0000000000),
    ("H", -0.5866101958,  1.4126530287,  0.0000000000),
    ("C",  0.5916281105, -0.3261693897,  0.0000000000),
    ("H",  0.5889652025, -1.4125832275,  0.0000000000)]

    translated_coords = []
    for t in range(nU):
        shift = t * d
        translated_coords.extend([(elem, x + shift, y, z)
            for elem, x, y, z in coords])
    return translated_coords

def get_gdf(filename, kpts, restart=False):
    """
    Calculate the 2e Integrals
    Using the Gaussian Density Fitting.
    """
    if not os.path.exists(filename) or restart:
        gdf = df.GDF(cell, kpts=kpts)
        gdf._cderi_to_save = filename
        gdf.build()
    return filename

# Build the Cell object
nU = 1
d = 2.47
basis = '6-31G'
pseudo = None
maxMem = 50000
nC = nU * 2

cell = pgto.Cell(atom = get_xyz(nU, d),
                 a = np.diag([d*nU, 17.5, 17.5]),
                 basis = basis,
                 pseudo = pseudo,
                 precision = 1e-12,
                 verbose = lib.logger.INFO,
                 max_memory = maxMem,
                 ke_cutoff = 40,
                 output = f"PAChain.{2}.{nC}.log"
)
cell.build()

kmf = scf.RHF(cell).density_fit()
kmf.max_cycle=1000
kmf.chkfile = f'PAchain.{nC}.chk'
kmf.with_df._cderi = get_gdf(kmf.chkfile.rstrip('.chk')+'.h5', kpts=[0,0,0]) # Only Gamma
kmf.exxdiv = None
kmf.conv_tol = 1e-12
meanfieldenergy = kmf.kernel()

# Use the right AVAS, for the active space selection.
from mrh.my_pyscf.pbc.mcscf import avas
ncas, nelecas, mo_coeff = avas.kernel(kmf, ['C 2pz'], minao=cell.basis, threshold=0.3)

# Now run CASCI and generate the reference energy value.
from pyscf import mcscf
mc = mcscf.CASCI(kmf, 2, 2)
mc.natorb=True
ref_energy = mc.kernel(mo_coeff=mo_coeff)[0]

assert ref_energy < meanfieldenergy, "Reference energy should be lower than mean field energy."

# Now my turn to solve the CAS problem
ncas = mc.ncas
ncore = mc.ncore
eri = kmf._eri # This might not be in this form, depends on the basis function size.

# Generate the CAS Hamiltonian
def _basis_transformation(mat, mo_coeff):
    return reduce(np.dot, (mo_coeff.conj().T, mat, mo_coeff))

def _ao2mo(eri, mo_cas):
    '''
    Transform the eri into mo basis
    '''
    return ao2mo.full(eri, mo_cas, compact=False)

def get_coredm(mo_c):
    '''
    Basically the cdm is 2* (C @ C.T)
    '''
    return 2 * (mo_c @ mo_c.conj().T)

def get_fci_ham(kmf, ncore, ncas, mo_coeff, eri):
    '''
    Get the FCI Hamiltonian in the CAS space.
    First compute the one-e Ham and 2e Ham in AO basis, then transform that to MO basis.
    At the same time, you can compute the core energy.
    '''
    cell = kmf.cell
    energy_nuc = cell.energy_nuc()
    mo_cas = mo_coeff[:, ncore:ncore+ncas]
    mo_core = mo_coeff[:, :ncore]
    # Get the one-electron integrals
    hcore = kmf.get_hcore()
    coredm = get_coredm(mo_core)
    veff = kmf.get_veff(cell, coredm)
    # Core energy
    Fpq = hcore + 0.5 * veff
    ecore = np.einsum('ij, ji', Fpq, coredm)
    ecore += energy_nuc
    # Transform the integrals to MO basis
    h1ao = hcore + veff
    h1 = _basis_transformation(h1ao, mo_cas)
    h2 = _ao2mo(eri, mo_cas)
    coredm, h1ao, veff, hcore = None, None, None, None  # Free memory
    return ecore, h1, h2

h0, h1, h2 = get_fci_ham(kmf, ncore, ncas, mo_coeff, eri)

# FCI kernel
efci, fcivec = fci.direct_spin1.kernel(h1, h2, ncas, nelecas, ecore=h0)

# Check the difference between my and Pyscf energy
print("Energy diff:", efci - ref_energy)


#TODO: Check the same with GTH-PP or some ECP