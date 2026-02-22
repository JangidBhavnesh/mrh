



# There is LASSCF RDM algorithm, I totally forgot about it. Anyways that won't work for my purposes, but
# here, I will join the DMRG as FCI solver for the LASSCF using the LASSCF RDM algorithm. This will be similar to
# using the SCI or SQD based SCI as the FCI solvers.

# Why don't we just write a wrapper function which points to the any FCIsolver available in PySCF, then just writing the
# name of that solver in the input file, and then we can use that as the FCI solver for LASSCF. 
# That way, we can easily switch between different FCI solvers without having to change the code. My guess is one 
# won't be able to juice couple of paper out of that. Haha....

import os
from pyscf import lib, dmrgscf
from mrh.my_pyscf.mcscf.lasscf_rdm import LASSCF as LASSCFNoSymm, RDMSolver, FCIBox

try:
    from pyscf import dmrgscf
except ImportError:
    print("pyscf dmrgscf module not found")
    
def get_dmrg_kernel (las, ifrag, maxM=512, tol=1e-7, num_thrds=lib.num_threads()):
    from pyscf.dmrgscf import settings
    def make_dmrg_solver(ncas, nelec, h0e, h1e, h2e):
        mol = las.mol
        h1e_sub = h1e[0] + h1e[1]
        h2e_sub = h2e
        assert (ncas == h1e_sub.shape[0])
        assert (ncas == h2e_sub.shape[0])
        solver = dmrgscf.DMRGCI(mol, maxM=maxM)
        solver.tol = tol
        solver.memory = int(mol.max_memory/1000)
        solver.nroots = 1
        solver.scratchDirectory = os.path.abspath(settings.BLOCKSCRATCHDIR) + f'/frag{ifrag}'
        solver.runtimeDir = os.path.abspath(settings.BLOCKSCRATCHDIR)  + f'/frag{ifrag}'
        solver.threads = num_thrds
        solver.spin = nelec[0] - nelec[1]
        solver.dump_flags()
        eci, FCIVec = solver.kernel(h1e_sub, h2e_sub, ncas, nelec)
        dm1a, dm1b = solver.make_rdm1s(FCIVec, ncas, nelec)
        dm2 = solver.make_rdm12(FCIVec, ncas, nelec)[1]
        s2, sm = solver.spin_square(FCIVec, ncas, nelec)
        print(f"DMRG fragment {ifrag} energy: {eci:.8f} Ha, S^2: {s2:.4f}, S: {sm:.4f}")
        return eci+h0e, (dm1a, dm1b), dm2
    return make_dmrg_solver


def make_fcibox (mol, kernel=None, get_init_guess=None, spin=None):
    solver = RDMSolver (mol, kernel=kernel, get_init_guess=get_init_guess)
    solver.spin = spin
    solver.smult = abs(spin) + 1 if spin is not None else None
    return FCIBox ([solver])

def DMRGLASSCF(mf_or_mol, ncas_sub, nelecas_sub, **kwargs):
    las = LASSCFNoSymm (mf_or_mol, ncas_sub, nelecas_sub, **kwargs)
    mol = las.mol
    nthreads = lib.num_threads()
    maxM = 512
    tol = 1e-7
    las.fciboxes = [make_fcibox (mol,
                                kernel=get_dmrg_kernel (las, ifrag, maxM=maxM, tol=tol, num_thrds=nthreads), 
                                spin= (las.nelecas_sub[ifrag][0] - las.nelecas_sub[ifrag][1]))
                                for ifrag in range (las.nfrags)]
    return las

LASSCF = DMRGLASSCF



if __name__ == '__main__':
    from pyscf import gto, scf, mcscf
    from pyscf.mcscf import avas

    mol = gto.M()
    mol.atom='''
    H -8.10574 2.01997 0.00000;
    H -7.29369 2.08633 0.00000; 
    H -3.10185 2.22603 0.00000;
    H -2.29672 2.35095 0.00000''' 
    mol.basis='CC-PVDZ'
    mol.verbose= lib.logger.INFO
    mol.max_memory = 100000
    mol.build()

    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    mo_coeff = avas.kernel(mf, ['H 1s'], minao=mol.basis)[2]

    # Reference values:
    mc = mcscf.CASSCF(mf, 4, (2, 2))
    solver = dmrgscf.DMRGCI(mol, maxM=512)
    solver.spin = 0
    solver.num_thrds = lib.num_threads()
    mc.fcisolver = solver
    mc.kernel(mo_coeff)
    eref = mc.e_tot

    # DMRG-LASSCF with 2 fragments
    las = LASSCF (mf, (4,4), ((1,1), (1, 1)), spin_sub=(1, 1))
    mo = las.localize_init_guess (([0, 1], [2, 3], ), mo_coeff)
    las.kernel (mo)
    elas = las.e_tot

    # Output results
    print("RHF energy: {:.8f} Ha".format(mf.e_tot))
    print(f"Reference DMRG-CASSCF energy: {eref:.8f} Ha")
    print(f"DMRG-LASSCF energy: {elas:.8f} Ha")
    print(f"Energy difference: {elas-eref:.8f} Ha")