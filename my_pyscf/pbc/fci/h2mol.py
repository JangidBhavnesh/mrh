import numpy as np
import os
from functools import reduce
from pyscf import fci
from pyscf import lib
from pyscf.pbc import scf, df, ao2mo
from pyscf.pbc import gto as pgto

'''
Benchmark the FCI kerenl H2 molecule with 2e and 4o: CCSD == FCI
'''
hhdistance = 0.8
Test1, Test2, Test3 = 0, 0, 1

if Test1:
    # Mol check:
    from pyscf import gto as mgto
    from pyscf import scf as mscf
    from pyscf import cc, mcscf

    mol = mgto.M(atom=f'''H 0, 0,0; H {hhdistance}, 0, 0''', 
                basis='631G', 
                charge=0, 
                spin=0, 
                verbose=0)
    mol.build()

    mf = mscf.RHF(mol)
    mf.kernel()

    mycc = cc.CCSD(mf)
    ecc = mycc.kernel()[0]

    mc = mcscf.CASCI(mf, 4, 2)
    ecas = mc.kernel()[0]

    assert abs(ecas - ecc - mf.e_tot) < 1e-6, "FCI and CCSD energies do not match within tolerance"
    ecctot = ecc + mf.e_tot
    eci = ecas
    print("")
    print("Mean-field Energy:", mf.e_tot)
    print("RCCSD energy:", ecctot)
    print("CASCI energy:", eci)
    print("")
    print("Molecular Test passed")

if Test2:
    # Periodic but at the Gamma point:
    cell = pgto.Cell(atom = f'''H 0, 0,0; H {hhdistance}, 0, 0''',
                    a = np.diag([17.5+hhdistance, 17.5, 17.5]),
                    basis = '631G',
                    pseudo = None,
                    precision = 1e-12,
                    verbose = 0, #lib.logger.INFO,
                    max_memory = 10000,
                    ke_cutoff = 400,
    )
    cell.build()

    from pyscf.pbc import scf, cc

    mf = scf.RHF(cell).density_fit()
    mf.exxdiv = None
    mf.conv_tol = 1e-12
    mf.kernel()

    mycc = cc.CCSD(mf)
    mycc.kernel()

    ecctot = mycc.e_tot

    mc = mcscf.CASCI(mf, 4, 2)
    eci = mc.kernel()[0]

    assert abs(eci - ecctot) < 1e-6, f"CASCI and CCSD energies do not match within tolerance: {abs(eci - ecctot)}"

    print("")
    print("Mean-field Energy:", mf.e_tot)
    print("RCCSD energy:", ecctot)
    print("CASCI energy:", eci)
    print("")
    print("Periodic Test at Gamma point: passed")

if Test3:
    # Periodic but at a specified k-point grid:
    cell = pgto.Cell(atom = f'''H 0, 0,0; H {hhdistance}, 0, 0''',
                    a = np.diag([2*hhdistance, 17.5, 17.5]),
                    basis = '631G',
                    pseudo = None,
                    precision = 1e-12,
                    verbose = 3,
                    max_memory = 10000,
                    ke_cutoff = 400,
    )
    cell.build()

    nk = [0, 0, 0]
    kpts = cell.make_kpts(nk)
    # abskpt = cell.get_abs_kpts([0.25, 1, 1])
    # kpts = abskpt

    kmf = scf.RHF(cell,).density_fit()
    kmf.max_cycle = 1000
    # kmf.kpt= kpts
    kmf.exxdiv = None
    kmf.conv_tol = 1e-12
    meanfieldenergy = kmf.kernel()

    mo_coeff = [kmf.mo_coeff, ]
    # print(kmf.mo_coeff.dtype)
    # exit()
    # Time to check the CCSD
    from pyscf.pbc import cc
    mycc = cc.RCCSD(kmf)
    mycc.kernel()

    # dm1 = mycc.make_rdm1()
    # dm2 = mycc.make_rdm2()
    
    # nmo = kmf.mo_coeff.shape[1]
    # eri_mo = kmf.with_df.ao2mo(kmf.mo_coeff, kpts=kmf.kpt).reshape([nmo]*4)
    # h1 = reduce(np.dot, (kmf.mo_coeff.conj().T, kmf.get_hcore(), kmf.mo_coeff))
    # e_tot = np.einsum('ij,ji', h1, dm1) 
    # print(e_tot.real)
    # e_tot2 = np.einsum('ijkl,jilk', eri_mo, dm2)*.5 
    # print(kmf.energy_nuc())
    # print(e_tot2.real)
    # print("RCCSD energy based on CCSD density matrices =", (e_tot + e_tot2 + kmf.energy_nuc()).real)
    # exit()
    # Use the right AVAS, for the active space selection.
    #from mrh.my_pyscf.pbc.mcscf import avas
    #mo_coeff = avas.kernel(kmf, ['H 1s', 'H 2s'], minao=cell.basis, threshold=0.3)[2]
    # mo_coeff = kmf.mo_coeff
    # Generate the CAS Hamiltonian
    def _basis_transformation(mat, mo_coeff):
        return reduce(np.dot, (mo_coeff.conj().T, mat, mo_coeff))

    def _ao2mo(kmf, mo_cas, kpt):
        '''
        Transform the eri into mo basis
        '''
        from pyscf import ao2mo
        ncas = mo_cas.shape[1]
        # eri_ao = kmf.with_df.ao2mo(mo_cas, kpts=[kpt]*4)
        # eri_mo = ao2mo.general(kmf._eri, mo_cas, compact=False).reshape(ncas, ncas, ncas, ncas)

        eri = kmf.with_df.ao2mo(mo_cas, kpts=[kpt]*4)
        eri = ao2mo.restore(1, eri, mo_cas.shape[1])
        return eri

    def get_coredm(mo_c):
        '''
        Basically the cdm is 2* (C @ C.T)
        '''
        return 2 * (mo_c @ mo_c.conj().T)

    def get_fci_ham(kmf, ncore, ncas, mo_coeff):
        '''
        Get the FCI Hamiltonian in the CAS space.
        First compute the one-e Ham and 2e Ham in AO basis, then transform that to MO basis.
        At the same time, you can compute the core energy.
        '''
        cell = kmf.cell
        hcorekpts = [kmf.get_hcore(),]
        coredm = [get_coredm(mo_coeff[i][:, :ncore]) for i in range(len(kmf.kpts))]
        veffkpts = [np.zeros_like(hcorekpts[0]),] # kmf.get_veff(cell, coredm)
        hamlst = {}
        for i, kpt in enumerate(kmf.kpts):
            energy_nuc = cell.energy_nuc()
            mo_cas = mo_coeff[i][:, ncore:ncore+ncas]
            # Core energy
            Fpq = hcorekpts[i] + 0.5 * veffkpts[i]
            ecore = np.einsum('ij, ji', Fpq, coredm[i])
            ecore += energy_nuc

            # Transform the integrals to MO basis
            h1ao = hcorekpts[i] + veffkpts[i]
            h1 = _basis_transformation(h1ao, mo_cas)
            h2 = _ao2mo(kmf, mo_cas, kpt)
            hamlst[i] = (ecore, h1, h2)
        return hamlst


    ncas = 4
    ncore = 0
    nelecas = (1,1)
    hamlst = get_fci_ham(kmf, ncore, ncas, mo_coeff)

    e_tot = 0
    from mrh.my_pyscf.pbc.fci import direct_nosym_spacomp
    for (h0, h1, h2) in hamlst.values():
        efci, fcivec = direct_nosym_spacomp.kernel(h1, h2, ncas, nelecas, ecore=h0)
        print(fcivec.real)
        if abs(efci[0].imag) > 1e-12:
            print("Non-zero imaginary part in FCI energy:", efci[0].imag)
        # Currently FCI is solving the entire Mat, so it will produce all the roots, take the ground state only.
        e_tot += efci[0]
        efci, _ = fci.direct_spin1.kernel(h1, h2, ncas, nelecas, ecore=h0)
        e_tot = efci
        print(_)
    
    print("")
    print("Mean-field Energy:", kmf.e_tot)
    # print("KRCCSD Energy (per unit cell):", mycc.e_tot)
    print("FCI Energy (per k-point):", e_tot.real/len(kmf.kpts))
    # print("Difference between FCI and CCSD:", e_tot.real/len(kmf.kpts) - mycc.e_tot)
    #TODO: Check the same with GTH-PP or some ECP