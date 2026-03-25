from pyscf.pbc import gto as pgto, scf
from pyscf import lib
from pyscf.lo import orth
import numpy as np
from ase import Atoms

def make_h2_3D(intraH=1.0, interH=1.5, nx=1, ny=1, nz=1, vacuum=17.5):
    pitch = intraH + interH
    ax = nx * pitch
    by = ny * pitch
    cz = nz * pitch
    cell = np.diag([ax, by, cz])

    positions = []
    symbols = []

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                x0 = ix * pitch
                y0 = iy * pitch
                z0 = iz * pitch
                positions.append([x0,          y0, z0])
                positions.append([x0 + intraH, y0, z0])
                symbols += ['H', 'H']

    atoms = Atoms(symbols, positions=positions, cell=cell, pbc=True)
    atoms.center()
    return atoms

atoms3D = make_h2_3D(intraH=0.74, interH=1.5, nx = 1, ny = 1, nz=1, vacuum=17.5)
cell = pgto.Cell()
cell.a = atoms3D.cell.array
pos = atoms3D.get_positions()
sym = atoms3D.get_chemical_symbols()
cell.atom = [(sym[i], tuple(pos[i])) for i in range(len(sym))]
cell.basis = '6-31G'
cell.unit = 'Angstrom'
cell.max_memory = 100000
cell.ke_cutoff = 100
cell.precision = 1e-10
cell.verbose = lib.logger.INFO
# cell.output = 'h2_3D_CASCI.log'
cell.build()

natom = cell.natm

kmesh = [2, 1, 1]
kpts = cell.make_kpts(kmesh)
kmf = scf.KRHF(cell, kpts=kpts).density_fit('def2-SVP-JKFIT')
kmf.exxdiv = None
kmf.kernel()


from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCF
from pyscf.pbc.tools import k2gamma

def get_frag_atms_kpts(cell, nkpts, frag_atms):
    '''
    Get the fragment atoms in each k-point.
    '''
    natm = cell.natm
    atms_kpts = [[atmindx + i*natm 
                  for atmindx in frag_atms] 
                  for i in range(nkpts)]
    return tuple(atms_kpts)

mf = k2gamma.k2gamma(kmf, kmesh)
scell = mf.cell

nkpts = np.prod(kmesh)
natom = cell.natm

frag_atom = get_frag_atms_kpts(cell, nkpts, [i for i in range(natom)])
las = LASSCF(mf, (natom,)*nkpts, (natom,)*nkpts)
mo = las.localize_init_guess(frag_atom)


from pyscf.tools import molden
molden.from_mo(scell, 'h2_3D_CASCI.las.molden', mo)

print('mo.shape = ', mo.shape)
