
import json
import os
import numpy as np
from scipy.linalg import expm

from ase import Atoms
from pyscf import lib
from pyscf.pbc import scf, df
from pyscf.pbc import gto as pgto
from mrh.my_pyscf.pbc.mcscf import avas

from mrh.my_pyscf.pbc import mcscf
from mrh.my_pyscf.pbc.fci import csf_solver

# Example to run the CASCI for kmf object (1D, 2D or 3D)

# TODO: Write one general function to make the H2 chain in 1D, 2D and 3D. 
# The current code is a bit redundant.
def make_h2_1D(intraH=1.1, interH=1.5, nx=1, vacuum=17.5):
    Lx = nx * (intraH + interH)
    cell = np.diag([Lx, vacuum, vacuum])
    y0 = vacuum / 2.0
    z0 = vacuum / 2.0
    positions = []
    symbols = []
    for i in range(nx):
        x0 = i * (intraH + interH)
        positions.append([x0,        y0, z0])   # N1
        positions.append([x0+intraH, y0, z0])   # N2
        symbols += ['Li', 'Li']
    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, )
    atoms.center()
    return atoms


# H2 Molecule in 1D
atoms1D = make_h2_1D(intraH=1.5, interH=3, nx = 1, vacuum=17.5)
cell = pgto.Cell()
cell.a = atoms1D.cell.array
pos = atoms1D.get_positions()
sym = atoms1D.get_chemical_symbols()
cell.atom = [(sym[i], tuple(pos[i])) for i in range(len(sym))]
cell.basis = 'STO-6G'
cell.unit = 'Angstrom'
cell.max_memory = 100000
cell.ke_cutoff = 100
cell.precision = 1e-10
cell.verbose = lib.logger.INFO
# cell.output = 'h2_1D_CASCI.log'
cell.build()

kmesh1D = [2, 1, 1]

kpts = cell.make_kpts(kmesh1D, wrap_around=True)

kmf = scf.KRHF(cell, kpts=kpts).density_fit(auxbasis='def2-svp-jkfit')
kmf.max_cycle=1000
kmf.exxdiv = None
kmf.conv_tol = 1e-10
kmf.kernel()

mo_coeff = avas.kernel(kmf, ['Li 2s'], minao=cell.basis)[2]


def rotate_mos(mo_coeff, kappa, scale):
    '''
    Rotate orbitals with C_new = C exp(scale * kappa)
    '''
    nkpts = len(mo_coeff)
    mo_rot = np.array(mo_coeff, copy=True)
    for k in range(nkpts):
        U = expm(scale * kappa[k])
        mo_rot[k] = mo_rot[k] @ U
    return mo_rot

def compute_mcscf_energy_at_mo(mc_ref, mo_rot):
    '''
    Evaluate the CASSCF energy at a rotated MO coefficient matrix mo_rot.

    A fresh CASSCF object is built with the same active space, but with
    macro orbital optimization disabled, so this acts as an energy evaluation
    at fixed rotated orbitals.
    '''
    ncas = mc_ref.ncas
    nelecas = mc_ref.nelecas
    mf = mc_ref._scf

    mc_temp = mcscf.CASCI(mf, ncas, nelecas)
    mc_temp.fcisolver = csf_solver(cell, smult=1)
    energy = mc_temp.kernel(mo_rot)[0]
    assert mc_temp.converged
    return energy


def unpack_uniq_var(mc, v):
    v = np.asarray(v)
    nmo = mc.mo_coeff[0].shape[1]
    nkpts = mc.nkpts
    dtype = mc.mo_coeff[0].dtype
    idx = mc.uniq_var_indices(nmo, mc.ncore, mc.ncas, mc.frozen)
    uniq_idx = int(np.count_nonzero(idx))

    # Decide whether the input is for a single k-point or for all k-points.
    assert v.size == uniq_idx or v.size == mc.nkpts * uniq_idx

    def _unpack_uniq_var(v):
        # For a single k-point.
        mat = np.zeros((nmo,nmo), dtype=dtype)
        mat[idx] = v
        return mat - mat.conj().T

    if v.size == uniq_idx:
        return _unpack_uniq_var(v)

    elif v.size == nkpts * uniq_idx:
        mats = np.zeros((nkpts, nmo, nmo), dtype=dtype)
        for k in range(nkpts):
            p0 = k * uniq_idx
            p1 = (k + 1) * uniq_idx
            mats[k] = _unpack_uniq_var(v[p0:p1])
        return mats
        

def compute_grad_error(mc, step_sizes, x0, grad_analytical):
    '''
    Compute the error in the gradient for a given set of step sizes and 
    initial direction x0.
    '''
    # Normalize the initial direction
    x0 = x0/np.linalg.norm(x0)

    # Compute reference energy at the current MO coefficients
    E0 = compute_mcscf_energy_at_mo(mc, mc.mo_coeff)
    
    # Store results in a dictionary
    results = {
        'step_sizes': list(step_sizes),
        "scale": [],
        "dE": [],
        "g_dot_x": [],
        "abs_epsilon": [],
    }

    for s in step_sizes:
        X = s * x0.copy()
        kappa = unpack_uniq_var(mc, X)
        mo_rot = rotate_mos(mc.mo_coeff.copy(), kappa, 1.0)
        E = compute_mcscf_energy_at_mo(mc, mo_rot)
        dE = E - E0
        g_dot_x = - np.vdot(grad_analytical, X)
        numerator = E - E0 - g_dot_x
        if abs(dE) < 1e-30: eps = np.nan
        else: eps = numerator / dE
        results["scale"].append(s)
        results["dE"].append(dE)
        results["g_dot_x"].append(g_dot_x)
        results["abs_epsilon"].append(abs(eps) if np.isfinite(eps) else np.nan)

    for k in results:
        results[k] = np.array(results[k])

    return results


kmc2 = mcscf.CASSCF(kmf, 2, (1,1))
kmc2.max_cycle_macro = 0
kmc2.fcisolver = csf_solver(cell, smult=1)
kmc2.kernel(mo_coeff)

kmc2.mo_coeff = mo_coeff.copy()

mo_ref = mo_coeff.copy()

grad = kmc2.get_grad(mo_coeff)

# Note: in pyscf the gradients are stored as the g-g.T that means
# I should multiply by 2 to get the correct gradient for the orbital rotations
# for this testing.
nkpts = kmc2.nkpts
grad_analytical = 2.0 / nkpts * np.asarray(grad).ravel()

# Define step sizes for the finite difference check
step_sizes = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 1e-5, 1e-6]

# Using the analytical gradient as the initial direction for the finite difference check
results = compute_grad_error(kmc2, step_sizes, x0=grad_analytical.copy(), grad_analytical=grad_analytical)

def get_rotation_vec_2():
    '''
    Build one random rotation direction per k-point.
    Each returned X0 has shape (nkpts, nocc, nvir), but only one k-block is nonzero.
    '''
    nocc = np.sum(kmf.mo_occ[0] > 0)
    nvir = kmf.mo_coeff[0].shape[1] - nocc
    rng = np.random.default_rng(7)
    nkpts = len(kmf.mo_coeff)
    
    X0_list = []
    for k in range(nkpts):
        # X0 = np.zeros((nocc, nvir), dtype=np.complex128)
        Xr = np.random.default_rng(7).standard_normal((grad_analytical.shape[0] // nkpts)) #rng.standard_normal((nocc, nvir))
        Xi = np.random.default_rng(7).standard_normal((grad_analytical.shape[0] // nkpts))
        X0_rand = Xr + 1j * Xi
        X0_rand /= np.linalg.norm(X0_rand)
        X0 = X0_rand
        X0_list.append(X0)
    return np.asarray(X0_list).ravel()


X0_rand = get_rotation_vec_2()
results_rand = compute_grad_error(kmc2, step_sizes, x0=X0_rand.copy(), grad_analytical=grad_analytical)

# Print results
for s, dE, gx, eps in zip(results['step_sizes'], results['dE'], results['g_dot_x'], results['abs_epsilon']):
    print(f'{s:9.1e}  dE={dE: .6e}  g·x={gx: .6e}  eps={eps: .6e}')



from matplotlib import pyplot as plt
plt.figure(figsize=(8,6))
plt.loglog(results["scale"], results["abs_epsilon"], marker='o', label=r'$|\epsilon(x)|$ (grad dir)')
plt.loglog(results_rand["scale"], results_rand["abs_epsilon"], marker='o', label=r'$|\epsilon(x)|$ (rand dir)')


# guide-to-eye linear scaling
ref = results["abs_epsilon"][0] * (results["scale"] / results["scale"][0])
plt.loglog(results["scale"], ref, '--', label='linear guide', color='black')
plt.loglog(results["scale"], abs(results["dE"]), '-.', marker='^', label='|dE|')


plt.xlabel(r'$|x|$', fontsize=16)
plt.ylabel(r'$|\epsilon(x)|$', fontsize=16)
plt.title('Sanity Check: Analytical Gradient for kRHF', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, which='both', alpha=0.3)
plt.savefig('n2_krhf_grad.pdf', dpi=300)
plt.close()

# step_sizes = [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5, 1e-6]
# grad_num = compute_mcscf_grad_blocks_step_sizes(kmc2, step_sizes)
# grad_num['step_sizes'] = step_sizes
# grad_num['analytical'] = grad

# import pickle
# with open('li2_orb_hrad_nk2_631g.pkl', 'wb') as f:
#     pickle.dump(grad_num, f)