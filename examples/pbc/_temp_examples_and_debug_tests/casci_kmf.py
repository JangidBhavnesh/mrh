
import os
import numpy as np
from functools import reduce

from pyscf import lib
from pyscf.pbc import scf, df
from pyscf.pbc import gto as pgto
from pyscf.pbc.lib import kpts_helper

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.mcscf import avas
from mrh.my_pyscf.pbc.mcscf.k2R import  get_mo_coeff_k2R

'''
Run the CASCI calculation for the PA chain system.
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

def get_gdf(filename, kpts=None, restart=True):
    """
    Calculate the 2e Integrals using the Gaussian Density Fitting.
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
maxMem = 100000

cell = pgto.Cell(atom = get_xyz(nU, d),
                 a = np.diag([d*nU, 17.5, 17.5]),
    basis = basis,
    pseudo = pseudo,
    precision = 1e-10,
    verbose = 3, #lib.logger.INFO,
    max_memory = maxMem,
    ke_cutoff = 40,
)
cell.build()

nk = [3, 1, 1]
kpts = cell.make_kpts(nk, wrap_around=True)
nC = nU * nk[0]

kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
kmf.max_cycle=1000
kmf.chkfile = f'PAchain.{nC}.chk'
kmf.with_df._cderi = get_gdf(f'PAchain.{nC}.gdf', kpts, restart=False)
kmf.exxdiv = None
kmf.init_guess = 'chk'
kmf.conv_tol = 1e-10
meanfieldenergy = kmf.kernel()

# Use the right AVAS, for the active space selection.
ncas, nelecas, mo_coeff = avas.kernel(kmf, ['C 2pz'], minao=cell.basis, 
                                      threshold=0.01, canonicalize=True)[:3]

# CASCI calculation.
kmc = mcscf.CASCI(kmf, 2, (1,1))
kmc.kernel(mo_coeff)

# Print the molden file for the active space orbitals.
from pyscf.tools import molden
molden.from_mo(kmf.cell, f'PAchain.{nC}.molden', 
               np.hstack([mo_coeff[k][:, kmc.ncore:kmc.ncore+kmc.ncas] for k in range(len(kpts))]).real ) 

# Spin-summed and spin-separated 1-RDMs in the AO basis.
# dm1 = kmc.make_rdm1()
# print("dm1 shape", dm1.shape)

# dm1s = kmc.make_rdm1s()
# print("dm1s shape", dm1s[0].shape)

# Example to use the DMRGCIComplex solver.
# This is running but
# 1. This is very slow.
# 2. I need to test the total energy and the 1-RDMs to make sure they are correct.

# from mrh.my_pyscf.pbc.fci import DMRGCIComplex
# kmc_dmrg = mcscf.CASCI(kmf, 2, 2)
# kmc_dmrg.fcisolver = DMRGCIComplex(kmf.cell)
# kmc_dmrg.fcisolver.memory = int(maxMem/1000)
# kmc_dmrg.fcisolver.threads = 36
# kmc_dmrg.fcisolver.maxM = 100
# kmc_dmrg.kernel(kmf.mo_coeff)


# Compute the CASCI energy in k-space by transforming the casdm1 and casdm2 from 
# the r-space to k-space.

# nkpts = np.prod(nk)
# ncore = kmc.ncore
# ncas = kmc.ncas
# nelecas = kmc.nelecas
# kconserv = kpts_helper.get_kconserv(kmf.cell, kmf.kpts)
# casdm1 = kmc.fcisolver.make_rdm1(kmc.ci, nkpts*ncas, (nkpts*nelecas[0], nkpts*nelecas[1]))
# (dm2aa, dm2ab, dm2bb) = \
#     kmc.fcisolver.make_rdm12s(kmc.ci, nkpts*ncas, (nkpts*nelecas[0], nkpts*nelecas[1]))[1]
# dm2ba = dm2ab.conj().transpose(1,0,3,2)
# casdm2 = dm2aa + dm2bb + dm2ab + dm2ba
# casdm2 = 0.5*(casdm2 + casdm2.conj().transpose(1,0,3,2))

# mo_core_kpts = [mo_coeff[k][:, :ncore] for k in range(nkpts)]
# h1ao_kpts = kmf.get_hcore()
# coredm_kpts = np.asarray([2.0 * (mo_core_kpts[k] @ mo_core_kpts[k].conj().T) for k in range(nkpts)])
# corevhf_kpts = kmf.get_veff(cell, coredm_kpts, hermi=1)

# mo_phase = get_mo_coeff_k2R(kmf, mo_coeff, ncore, ncas)[-1]

# E1 = []
# for k in range(nkpts):
#     e_k = 0
#     corevhf_k = corevhf_kpts[k]
#     fock = h1ao_kpts[k] + 0.5*corevhf_k
#     e_k += np.einsum('ij,ji', coredm_kpts[k], fock)
#     h1ao_k = fock + 0.5 * corevhf_k
#     h1mo_k = mo_coeff[k][:, ncore:ncore+ncas].conj().T @ h1ao_k @ mo_coeff[k][:, ncore:ncore+ncas]
#     casdm1_k = reduce(np.dot, (mo_phase[k], casdm1, mo_phase[k].conj().T))
#     e_k += np.einsum('ij,ji', h1mo_k, casdm1_k)
#     E1.append(e_k)

# E2 = []
# for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
#     k4 = kconserv[k1, k2, k3]
#     h2emo_k = kmf.with_df.ao2mo([mo_coeff[k1][:, ncore:ncore+ncas], 
#                                     mo_coeff[k2][:, ncore:ncore+ncas], 
#                                     mo_coeff[k3][:, ncore:ncore+ncas], 
#                                     mo_coeff[k4][:, ncore:ncore+ncas]],
#                                 [kpts[k1], 
#                                     kpts[k2], 
#                                     kpts[k3], 
#                                     kpts[k4]], compact=False).reshape([ncas]*4)
    
#     casdm2_k = np.einsum('ip, jq, pqrs, kr, ls -> ijkl',
#                         mo_phase[k1].conj(), mo_phase[k2], 
#                         casdm2, mo_phase[k3].conj(), mo_phase[k4])
#     e2 = 0.5*np.einsum('pqrs,pqrs->', h2emo_k, casdm2_k)
#     E2.append(e2)

# E0 = cell.energy_nuc()
# E1_avg = sum(E1)/nkpts
# E2_avg = sum(E2)/nkpts**2

# print("   kHF Energy: ", kmf.e_tot)
# print(" CASCI Energy: ", kmc.e_tot.real)
# print("kCASCI Energy: ", (E0 + E1_avg + E2_avg).real)
# print("Difference between kCASCI and CASCI   : ", (E0 + E1_avg + E2_avg).real - kmc.e_tot.real)
