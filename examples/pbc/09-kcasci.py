import numpy as np

from pyscf import lib
from pyscf.pbc import gto as pgto
from pyscf.pbc import scf

from mrh.my_pyscf.pbc import mcscf


'''
Basic k-CASCI example.

The k-CASCI solver works in one total momentum sector at a time.  This example
runs the same active space for each target_k sector in a small periodic H2
system.
'''

intraH = 0.74
interH = 1.5
vac = 17.5

cell = pgto.Cell()
cell.a = np.diag([intraH + interH, intraH + interH, vac])
cell.atom = [
    ["H", (0.0, 0.0, vac / 2.0)],
    ["H", (intraH, 0.0, vac / 2.0)],
]
cell.basis = "STO-6G"
cell.unit = "Angstrom"
cell.max_memory = 100000
cell.ke_cutoff = 100
cell.precision = 1e-10
cell.verbose = lib.logger.INFO
cell.build()

kmesh = [3, 1, 1]
kpts = cell.make_kpts(kmesh, wrap_around=True)
nkpts = len(kpts)

kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis="def2-svp-jkfit")
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

mo_coeff = np.asarray(kmf.mo_coeff)

print(f"k-RHF energy: {kmf.e_tot.real:12.8f}")

for target_k in range(nkpts):
    kmc = mcscf.KCASCI(kmf, 2, 2, target_k=target_k)
    kmc.kmesh = kmesh
    kmc.canonicalization = False

    e_tot, e_cas, ci, mo_coeff, mo_energy = kmc.kernel(mo_coeff)

    print(f"target_k = {target_k}")
    print(f"k-CASCI energy: {e_tot.real:12.8f}")
