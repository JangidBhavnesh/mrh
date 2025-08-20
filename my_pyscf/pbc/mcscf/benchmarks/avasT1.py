import numpy as np
import os
from functools import reduce
from pyscf import lib
from pyscf.pbc import scf, df
from pyscf.pbc import gto as pgto


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

def get_gdf(filename, kpts=[0,0,0], restart=False):
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

cell = pgto.Cell()
cell.atom = get_xyz(nU, d)
cell.a = np.diag([2.47*nU, 17.5, 17.5])
cell.basis = basis
cell.pseudo = pseudo
cell.precision=1e-12
cell.verbose = lib.logger.INFO
cell.max_memory = maxMem
cell.ke_cutoff = 40
cell.output = f"PAChain.{2}.{nC}.log"
cell.build()

# SCF
kmf = scf.RHF(cell, ).density_fit()
kmf.max_cycle=1000
kmf.chkfile = f'PAchain.{nC}.chk'
kmf.with_df._cderi = get_gdf(kmf.chkfile.rstrip('.chk')+'.h5')
kmf.exxdiv = None
kmf.conv_tol = 1e-12
kmf.kernel()

ref_energy = kmf.energy_elec()[0]

# Now let's select the AVAS initial guess.
# Test-1
from pyscf.mcscf import avas
mo_coeff = avas.kernel(kmf, ['C 2pz'], minao=cell.basis)[2]
dm = kmf.make_rdm1(mo_coeff, kmf.mo_occ)
avas_energy_current = kmf.energy_elec(dm)[0]

from mrh.my_pyscf.pbc.mcscf import avas
mo_coeff = avas.kernel(kmf, ['C 2pz'], minao=cell.basis)[2]
dm = kmf.make_rdm1(mo_coeff, kmf.mo_occ)
avas_energy = kmf.energy_elec(dm)[0]

print('The energy change following AVAS:', ref_energy - avas_energy_current)
print('The energy change following periodi AVAS:', ref_energy - avas_energy)

#The energy change following AVAS: -0.00025995409606593967
#The energy change following periodic AVAS: 2.842170943040401e-14
