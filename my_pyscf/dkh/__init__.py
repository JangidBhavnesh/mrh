# Please consider citing
# M. Reiher, A. Wolf, J. Chem. Phys. 121 (2004) 10944-10956           
# A. Wolf, M. Reiher, B. A. Hess, J. Chem. Phys. 117 (2002) 9215-9226 


# Author: Bhavnesh Jangid
# Note, these are just scalar relativistic corrections to the hamiltonian
# the picture change corrections to properties will be added later.

''''
# Benchmarks
# Basis = ANO-RCC-VTZP
# SCF Conv = 1e-14
# OpenMolcas : d8b5cd8618f628876a7e0c692027dc6d (commit)
# Orca : 5.0.4 (Release)
# PySCF: 2.6.2
# Psi4 : 1.9.1 release


# Absolute Energies

# DKH-2
Atom    OpenMolcas                      Orca                    Psi4                    PySCF
Zn      -1794.2309554841        -1794.2309578136        -1794.2309578172        -1794.2309578171
Cd      -5590.2829028569        -5590.2829212740        -5590.2829212659        -5590.2829212659
Hg      -19603.8079195666       -19603.8081169078       -19603.8081169134       -19603.8081169123


# DKH3
Atom            Psi4                    PySCF
Zn      -1794.31745252487       -1794.31745252483
Cd      -5591.51837455907       -5591.51837455902
Hg      -19625.72495753350      -19625.72495753240


# DKH4
Atom            PySCF
Zn      -1794.30866836286
Cd      -5591.33296981271
Hg      -19621.75579187100
'''


from mrh.my_pyscf.dkh._dkh import dkhscalar
dkh = dkhscalar

