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

# Ground-State:
# Neutral reference energy in the KCASCI per-cell convention.
kmc_neutral = mcscf.KCASCI(kmf, 2, 2, target_k=0)
kmc_neutral.kmesh = kmesh
kmc_neutral.fcisolver.fix_spin_(shift=0.2, ss=0.0)
e_neutral = kmc_neutral.kernel(mo_coeff)[0]
    
# Charged state:
# charge=1 means one electron is removed from the complete k-mesh active
# space.  target_k=None sweeps all charged momentum sectors.
kmc_hole = mcscf.KCASCI(kmf, 2, 2, charge=1)
kmc_hole.kmesh = kmesh
kmc_hole.fcisolver.nroots = 1
kmc_hole.fcisolver.fix_spin_(shift=0.2, ss=0.75)
kmc_hole.kernel(mo_coeff)

print(f"charged active space: {sum(kmc_hole.charged_nelecastot)}e, "
      f"{kmc_hole.nkpts * kmc_hole.ncas}o")
for band in kmc_hole.band_energies(e_neutral, kpts=kpts):
    kvec = np.asarray(band["hole_momentum"]).real
    print("hole momentum: "
          f"[{kvec[0]:9.6f}, {kvec[1]:9.6f}, {kvec[2]:9.6f}]  "
          f"target_k = {band['target_k']}  "
          f"band energy = {band['energy'].real:12.8f}")

