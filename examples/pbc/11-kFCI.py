#!/usr/bin/env python
import numpy as np

from pyscf import lib
from pyscf.pbc import gto
from pyscf.pbc import scf

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.fci import ksolver

# Author: Bhavnesh Jangid

"""
Run k-FCI independently in each total-momentum sector.
"""

def get_kfci_integrals(kmc, mo_coeff):
    """Build the active-space effective Hamiltonian in k-space."""
    cell = kmc.cell
    kmf = kmc._scf
    nkpts = kmc.nkpts
    ncore = kmc.ncore
    ncas = kmc.ncas
    nocc = ncore + ncas

    hcore = kmc.get_hcore()
    dtype = np.result_type(hcore, *[mo.dtype for mo in mo_coeff])
    hcore = hcore.astype(dtype)

    mo_core = [mo[:, :ncore] for mo in mo_coeff]
    mo_cas = np.asarray(
        [mo[:, ncore:nocc] for mo in mo_coeff], dtype=dtype)

    # The final total energy is divided by nkpts, so accumulate the nuclear
    # and core contributions with the corresponding cell normalization.
    ecore = kmc.energy_nuc() * nkpts
    if ncore > 0:
        dm_core = np.asarray([
            2.0 * mo_core[kpoint] @ mo_core[kpoint].conj().T
            for kpoint in range(nkpts)
        ], dtype=dtype)
        core_vhf = kmc.get_veff(cell, dm_core, hermi=1, kpts=kmf.kpts)
        fock_core = hcore + 0.5 * core_vhf
        ecore += sum(
            np.einsum("ij,ji", dm_core[kpoint], fock_core[kpoint])
            for kpoint in range(nkpts))
        hcore += core_vhf

    h1e = np.asarray([
        mo_cas[kpoint].conj().T @ hcore[kpoint] @ mo_cas[kpoint]
        for kpoint in range(nkpts)
    ], dtype=dtype)

    # The 1/nkpts factor gives the supercell normalization.
    h2e = kmf.with_df.ao2mo_7d(mo_cas, kpts=kmf.kpts)
    h2e = np.asarray(h2e, dtype=dtype) / nkpts

    # contract_2e follows PySCF direct_spin1.contract_2e conventions, so use
    # the effective one-electron Hamiltonian h1 - J/2 and two-electron tensor
    # h2/2.
    j_eff = np.zeros_like(h1e)
    for kp in range(nkpts):
        for kq in range(nkpts):
            j_eff[kp] += np.einsum("piis->ps", h2e[kp, kq, kq])
    h1e -= 0.5 * j_eff
    h2e *= 0.5
    return h1e, h2e, ecore


intra_h = 0.74
inter_h = 1.5
vacuum = 17.5

cell = gto.Cell()
cell.a = np.diag([intra_h + inter_h, intra_h + inter_h, vacuum])
cell.atom = [
    ["H", (0.0, 0.0, vacuum / 2.0)],
    ["H", (intra_h, 0.0, vacuum / 2.0)],
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

kmf = scf.KRHF(cell, kpts=kpts).density_fit(
    auxbasis="def2-svp-jkfit")
kmf.max_cycle = 1000
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

# Two active orbitals and electrons per primitive cell correspond to a
# six-orbital, six-electron active space across this three-point mesh.
kmc = mcscf.CASCI(kmf, 2, 2)
kmc.kpts = kpts
kmc.kmesh = kmesh

mo_coeff = np.asarray(kmf.mo_coeff)
h1e, h2e, ecore = get_kfci_integrals(kmc, mo_coeff)
norb = nkpts * kmc.ncas
nelecas = (nkpts * kmc.nelecas[0], nkpts * kmc.nelecas[1])

print(f"k-RHF energy: {kmf.e_tot.real:12.8f}")

# The k-FCI solver currently provides spin penalization but not a CSF solver.
for target_k in range(nkpts):
    kmc.fcisolver = ksolver(cell, nkpts=nkpts, target_k=target_k)
    kmc.fcisolver.conv_tol = 1e-10
    kmc.fcisolver.fix_spin_(shift=0.2, ss=0.0)
    e_tot, ci = kmc.fcisolver.kernel(
        h1e, h2e, norb, nelecas, ecore=ecore)
    ss, multiplicity = kmc.fcisolver.spin_square(ci, norb, nelecas)
    rdm1, rdm2 = kmc.fcisolver.make_rdm12(
        ci, norb, nelecas, nkpts, target_k=target_k)

    print(f"target_k     : {target_k}")
    print(f"k-FCI energy : {e_tot.real / nkpts:12.8f}")
    print(f"<S^2>        : {ss.real:12.8f}")
    print(f"2S+1         : {multiplicity.real:12.8f}")
    print(f"1-RDM shape  : {rdm1.shape}")
    print(f"2-RDM shape  : {rdm2.shape}")
    print(f"1-RDM trace  : {np.trace(rdm1).real:12.8f}")
