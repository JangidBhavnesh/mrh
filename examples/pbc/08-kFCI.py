import numpy as np

from pyscf import lib
from pyscf.pbc import scf
from pyscf.pbc import gto as pgto

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.fci import ksolver

# Author: Bhavnesh Jangid

'''
In this file, I am showing a basic k-FCI calculation.
The k-FCI solver works in one total momentum sector at a time.
'''


def get_kfci_integrals(kmc, mo_coeff):
    '''
    Build the active-space Hamiltonian in k-space for the k-FCI solver.
    These are in block MO basis, not the wannier MO basis.
    '''
    cell = kmc.cell
    kmf = kmc._scf
    nkpts = kmc.nkpts
    ncore = kmc.ncore
    ncas = kmc.ncas
    nocc = ncore + ncas

    hcore = kmc.get_hcore()
    dtype = mo_coeff[0].dtype
    hcore = hcore.astype(dtype)

    mo_core = [mo[:, :ncore] for mo in mo_coeff]
    mo_cas = np.asarray([mo[:, ncore:nocc] 
                         for mo in mo_coeff], dtype=dtype)

    ecore = kmc.energy_nuc() #* nkpts
    if ncore > 0:
        dm_core = np.asarray([2.0 * mo_core[k] @ mo_core[k].conj().T
                              for k in range(nkpts)], dtype=dtype)
        corevhf = kmc.get_veff(cell, dm_core, hermi=1, kpts=kmf.kpts)
        fock_core = hcore + 0.5 * corevhf
        ecore += sum(np.einsum('ij,ji', dm_core[k], fock_core[k])
                     for k in range(nkpts))
        hcore += corevhf

    h1e = np.asarray([mo_cas[k].conj().T @ hcore[k] @ mo_cas[k]
                      for k in range(nkpts)], dtype=dtype)

    # Two-electron integrals in k-space. The 1/nkpts factor gives the supercell normalization.
    h2e = kmf.with_df.ao2mo_7d(mo_cas, kpts=kmf.kpts)
    h2e = np.asarray(h2e, dtype=dtype) / nkpts

    return h1e, h2e, ecore


intraH = 0.74
interH = 1.5
nx = 1
ny = 1
vac = 17.5

ax = nx * (intraH + interH)
by = ny * (intraH + interH)
cz = vac

cell = pgto.Cell()
cell.a = np.diag([ax, by, cz])
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

kmesh1D = [3, 1, 1] 
kpts = cell.make_kpts(kmesh1D, wrap_around=True)
nkpts = len(kpts)

kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
kmf.max_cycle=1000
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

# This is equivalent to (6e, 6o) in the supercell.
kmc = mcscf.CASCI(kmf, 2, 2)
kmc.kpts = kpts
kmc.kmesh = kmesh1D

mo_coeff = np.asarray(kmf.mo_coeff)
h1e, h2e, ecore = get_kfci_integrals(kmc, mo_coeff)

norb = nkpts * kmc.ncas
nelecas = (nkpts * kmc.nelecas[0], nkpts * kmc.nelecas[1])

for target_k in range(nkpts):
    kmc.fcisolver = ksolver(cell, nkpts=nkpts, target_k=target_k)
    kmc.fcisolver.conv_tol = 1e-10
    e_tot, ci = kmc.fcisolver.kernel(h1e, h2e, norb, nelecas, ecore=ecore)
    ss, smult = kmc.fcisolver.spin_square(ci, norb, nelecas)

    print(f"target_k = {target_k}")
    print(f"k-FCI energy: {e_tot.real / nkpts:12.8f}")
    print(f"<S^2>       : {ss.real:12.8f}")
    print(f"2S+1        : {smult.real:12.8f}")
