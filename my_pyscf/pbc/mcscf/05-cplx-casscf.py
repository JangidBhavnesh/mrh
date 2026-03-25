
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


def make_kappa_pair(nmo, p, q, component="real", dtype=np.complex128):
    """
    Build a single-pair anti-Hermitian orbital-rotation generator K
    for orbitals p and q.

    component = "real":
        K_pq = +1, K_qp = -1

    component = "imag":
        K_pq = +1j, K_qp = +1j

    Both are anti-Hermitian:
        K^\dagger = -K
    """
    K = np.zeros((nmo, nmo), dtype=dtype)

    if component == "real":
        K[p, q] = 1.0
        K[q, p] = -1.0
    elif component == "imag":
        K[p, q] = 1.0j
        K[q, p] = 1.0j
    else:
        raise ValueError(f"Unknown component = {component}")

    return K


def rotate_mos_pair(mo_coeff, p, q, scale, kidx, component="real"):
    """
    Rotate orbitals at one k-point using

        C_new = C exp(scale * K)

    where K is the anti-Hermitian generator for pair (p,q).
    """
    mo_rot = [c.copy() for c in mo_coeff]
    nmo = mo_rot[kidx].shape[1]

    K = make_kappa_pair(nmo, p, q, component=component, dtype=mo_rot[kidx].dtype)
    U = expm(scale * K)
    mo_rot[kidx] = mo_rot[kidx] @ U
    return mo_rot


def compute_mcscf_energy_at_mo(mc_ref, mo_rot):
    """
    Evaluate the CASCI energy at fixed rotated orbitals.

    A fresh CASCI object is built from the same SCF reference and active space.
    """
    ncas = mc_ref.ncas
    nelecas = mc_ref.nelecas
    mf = mc_ref._scf

    mc_temp = mcscf.CASCI(mf, ncas, nelecas)

    # Reuse the same FCI solver if possible
    mc_temp.fcisolver = mc_ref.fcisolver

    e = mc_temp.kernel(mo_coeff=mo_rot)[0]
    assert mc_temp.converged
    return e


def compute_grad_pq_numerical_complex(mc, mo_ref, p, q, h, kidx):
    """
    Numerical complex orbital gradient element for orbital pair (p,q)
    at k-point kidx.

    For complex orbital rotations, we finite-difference along two
    independent anti-Hermitian directions:

      1) real generator:
            K^R_pq =  1,   K^R_qp = -1

      2) imaginary generator:
            K^I_pq =  i,   K^I_qp =  i

    Then define the complex gradient element as

        g_pq = dE/dx + i dE/dy

    where x is the real-pair rotation amplitude and y is the imaginary-pair
    rotation amplitude.

    If your analytic convention uses the opposite sign for the imaginary
    component, switch the last line to

        g_pq = dE_dx - 1j * dE_dy
    """
    # --- real-direction finite difference ---
    mo_p_r = rotate_mos_pair(mo_ref, p, q, +h, kidx, component="real")
    mo_m_r = rotate_mos_pair(mo_ref, p, q, -h, kidx, component="real")

    e_p_r = compute_mcscf_energy_at_mo(mc, mo_p_r)
    e_m_r = compute_mcscf_energy_at_mo(mc, mo_m_r)

    dE_dx = (e_p_r - e_m_r) / (2.0 * h)

    # --- imaginary-direction finite difference ---
    mo_p_i = rotate_mos_pair(mo_ref, p, q, +h, kidx, component="imag")
    mo_m_i = rotate_mos_pair(mo_ref, p, q, -h, kidx, component="imag")

    e_p_i = compute_mcscf_energy_at_mo(mc, mo_p_i)
    e_m_i = compute_mcscf_energy_at_mo(mc, mo_m_i)

    dE_dy = (e_p_i - e_m_i) / (2.0 * h)

    # Combine into one complex gradient element
    grad_pq = dE_dx + 1j * dE_dy
    return grad_pq


def compute_grad_pq_numerical(mc, mo_ref, kidx, p, q, h):
    """
    Backward-compatible wrapper returning the complex numerical
    gradient element for pair (p,q) at k-point kidx.
    """
    return compute_grad_pq_numerical_complex(mc, mo_ref, p, q, h, kidx)


def compute_mcscf_grad_blocks_step_sizes(mc, step_sizes):
    """
    Compute numerical MCSCF orbital-gradient blocks for several step sizes.

    Returns
    -------
    grad_blocks : dict
        grad_blocks[h][kidx] is a dictionary with:
          - core_active : shape (ncore, nact)
          - active_vir  : shape (nact, nvir)
          - core_vir    : shape (ncore, nvir)
          - packed      : flat vector in block order
                          [core-active, active-vir, core-vir]
    """
    ncore = mc.ncore
    nact = mc.ncas
    nkpts = mc.nkpts
    nmo = mc.mo_coeff[0].shape[1]
    nvir = nmo - ncore - nact

    core_idx = np.arange(0, ncore)
    act_idx  = np.arange(ncore, ncore + nact)
    vir_idx  = np.arange(ncore + nact, nmo)

    mo_ref = [c.copy() for c in mc.mo_coeff]
    dtype = np.result_type(mc.mo_coeff[0].dtype, np.complex128)

    grad_blocks = {}

    def get_grad_blocks_for_k(h, kidx):
        G_ca = np.zeros((ncore, nact), dtype=dtype)
        G_av = np.zeros((nact, nvir), dtype=dtype)
        G_cv = np.zeros((ncore, nvir), dtype=dtype)

        packed = []

        # core-active block
        for i_core, p in enumerate(core_idx):
            for i_act, q in enumerate(act_idx):
                g_fd = compute_grad_pq_numerical(mc, mo_ref, kidx, p, q, h)
                G_ca[i_core, i_act] = g_fd
                packed.append(g_fd)

        # active-virtual block
        for i_act, p in enumerate(act_idx):
            for i_vir, q in enumerate(vir_idx):
                g_fd = compute_grad_pq_numerical(mc, mo_ref, kidx, p, q, h)
                G_av[i_act, i_vir] = g_fd
                packed.append(g_fd)

        # core-virtual block
        for i_core, p in enumerate(core_idx):
            for i_vir, q in enumerate(vir_idx):
                g_fd = compute_grad_pq_numerical(mc, mo_ref, kidx, p, q, h)
                G_cv[i_core, i_vir] = g_fd
                packed.append(g_fd)

        return {
            # "core_active": G_ca,
            # "active_vir": G_av,
            # "core_vir": G_cv,
            "packed": np.array(packed, dtype=dtype),
        }

    for h in step_sizes:
        grad_blocks[h] = {}
        for kidx in range(nkpts):
            grad_blocks[h][kidx] = get_grad_blocks_for_k(h, kidx)

    return grad_blocks

kmc2 = mcscf.CASSCF(kmf, 2, (1,1))
kmc2.max_cycle_macro = 0
kmc2.fcisolver = csf_solver(cell, smult=1)
kmc2.kernel(mo_coeff)

kmc2.mo_coeff = mo_coeff.copy()

mo_ref = mo_coeff.copy()

# Define step sizes for the finite difference check
step_sizes = [1e-1, 1e-3, 1e-5, 1e-7]

grad_num = compute_mcscf_grad_blocks_step_sizes(kmc2, step_sizes)



import pickle
with open('li2_grad_sto6g.pkl', 'wb') as f:
    pickle.dump(grad_num, f)



grad = kmc2.get_grad(mo_coeff)

# Note: in pyscf the gradients are stored as the g-g.T that means
# I should multiply by 2 to get the correct gradient for the orbital rotations
# for this testing.
nkpts = kmc2.nkpts
grad_analytical = 2.0 / nkpts * np.asarray(grad).ravel()

print(grad_num)


# step_sizes = [5e-1, 1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 1e-4, 1e-5, 1e-6]
# grad_num = compute_mcscf_grad_blocks_step_sizes(kmc2, step_sizes)
# grad_num['step_sizes'] = step_sizes
# grad_num['analytical'] = grad
