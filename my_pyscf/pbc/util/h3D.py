from turtle import color
import numpy as np
from ase import Atoms
from pyscf.pbc import gto, scf
from matplotlib import pyplot as plt

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

def get_cell(atoms):
    cell = gto.Cell()
    cell.a = atoms.cell.array
    pos = atoms.get_positions()
    sym = atoms.get_chemical_symbols()
    cell.atom = [(sym[i], tuple(pos[i])) for i in range(len(sym))]
    cell.basis = '6-31g'
    cell.unit = 'Angstrom'
    cell.max_memory = 100000
    cell.ke_cutoff = 100
    cell.verbose = 3
    cell.build()
    return cell

def run_scf(cell, kmesh=(1, 1, 1)):
    kpts = cell.make_kpts(kmesh, wrap_around=True)
    mf = scf.KRHF(cell, kpts=kpts).density_fit()
    mf.exxdiv = None
    mf.conv_tol = 1e-10
    mf.kernel()
    return mf

def plot_bands(cell, e_kn, filename="h3D_bands.png", title=r"H$_2$ Band Structure"):
    # Restricted HF Only
    au2ev = 27.21139
    homo_band = cell.nelectron // 2 - 1
    lumo_band = homo_band + 1

    vbmax = max(en[homo_band] for en in e_kn)
    e_kn = [en - vbmax for en in e_kn]
    nk = len(e_kn)
    nbands = len(e_kn[0])

    k_idx = list(range(nk))
    plt.figure(figsize=(5, 6))
    for n in range(nbands):
        y = [e_kn[ik][n] * au2ev for ik in range(nk)]
        plt.plot(k_idx, y, lw=1.0, alpha=0.8, color='gray')
        plt.scatter(k_idx, y, s=10, alpha=0.8, color='gray')

    homo_y = [e_kn[ik][homo_band] * au2ev for ik in range(nk)]
    lumo_y = [e_kn[ik][lumo_band] * au2ev for ik in range(nk)]

    plt.plot(k_idx, homo_y, lw=2.0)
    plt.scatter(k_idx, homo_y, s=30, marker="^", zorder=5, label=f"VB")

    plt.plot(k_idx, lumo_y, lw=2.0)
    plt.scatter(k_idx, lumo_y, s=30, marker="v", zorder=5, label=f"CB")

    homo_k = max(range(nk), key=lambda ik: e_kn[ik][homo_band])
    lumo_k = min(range(nk), key=lambda ik: e_kn[ik][lumo_band])
    plt.scatter([homo_k], [homo_y[homo_k]], s=120, marker="*", zorder=6, label="VBM")
    plt.scatter([lumo_k], [lumo_y[lumo_k]], s=120, marker="*", zorder=6, label="CBM")

    emin, emax = min(homo_y), max(lumo_y)
    plt.ylim(emin - 10, emax + 50)
    plt.xlim(-1, nk)
    plt.axhline(0.0, ls="--", lw=2.0, label='Fermi Level', color='black')
    plt.xticks([])
    plt.xlabel("k-vector", fontsize=12)
    plt.ylabel("Energy (eV)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(frameon=False, fontsize=10, loc="best")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

if __name__ == "__main__":
    kmesh = (4, 4, 4)
    h3D = make_h2_3D(intraH=0.74, interH=1.5, nx = 1, ny=1, nz=1, vacuum=17.5)
    cell = get_cell(h3D)
    kmf = run_scf(cell, kmesh=kmesh)

    band_kpts = cell.make_kpts(kmesh, wrap_around=True)
    e_kn = kmf.get_bands(band_kpts)[0]

    plot_bands(cell, e_kn, filename="h3D_bands.png", title=rf"H$_2$ Band Structure {kmesh}")


