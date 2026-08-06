
from pyscf.pbc import scf, dft
from mrh.my_pyscf.pbc.mcscf import casci
from mrh.my_pyscf.pbc.mcscf import kcasci
from mrh.my_pyscf.pbc.mcscf import mc1step
from mrh.my_pyscf.pbc.mcscf.productstate import (
    PBCTransSymmImpureProductStateFCISolver,
)
from mrh.my_pyscf.pbc.mcscf.klasci import (
    PBCLASCINoSymm,
    PBCLASCITransSymm,
    kLASCI,
)

def CASCI(kmf, ncas, nelecas, ncore=None):
    assert isinstance(kmf, scf.hf.SCF),  "CASCI only works with periodic SCF objects"
    # Make sure kdft mean field objects are not passed to kCASCI
    if isinstance(kmf, dft.krks.KRKS) or isinstance(kmf, dft.kuks.KUKS) \
        or isinstance(kmf, dft.rks.RKS) or isinstance(kmf, dft.uks.UKS):
        raise NotImplementedError("CASCI with DFT is not implemented yet.")
    if isinstance(kmf, scf.kuhf.KUHF):
        kmf = scf.addons.convert_to_rhf(kmf)
    kmc = casci.CASCI(kmf, ncas, nelecas, ncore)
    return kmc

def KCASCI(kmf, ncas, nelecas, ncore=None, target_k=None, charge=None,
           charged_spin=None):
    assert isinstance(kmf, scf.hf.SCF),  "KCASCI only works with periodic SCF objects"
    # Make sure kdft mean field objects are not passed to kCASCI
    if isinstance(kmf, dft.krks.KRKS) or isinstance(kmf, dft.kuks.KUKS) \
        or isinstance(kmf, dft.rks.RKS) or isinstance(kmf, dft.uks.UKS):
        raise NotImplementedError("KCASCI with DFT is not implemented yet.")
    if isinstance(kmf, scf.kuhf.KUHF):
        kmf = scf.addons.convert_to_rhf(kmf)
    if charge is None:
        if target_k is None:
            target_k = 0
        kmc = kcasci.KCASCI(kmf, ncas, nelecas, ncore, target_k=target_k)
    else:
        kmc = kcasci.ChargedKCASCI(kmf, ncas, nelecas, ncore,
                                   charge=charge, target_k=target_k,
                                   charged_spin=charged_spin)
    return kmc

def CASSCF(kmf, ncas, nelecas, ncore=None):
    assert isinstance(kmf, scf.hf.SCF),  "CASSCF only works with periodic SCF objects"
    # Make sure kdft mean field objects are not passed to kCASSCF
    if isinstance(kmf, dft.krks.KRKS) or isinstance(kmf, dft.kuks.KUKS) \
        or isinstance(kmf, dft.rks.RKS) or isinstance(kmf, dft.uks.UKS):
        raise NotImplementedError("CASSCF with DFT is not implemented yet.")
    # If the mean-field object is KUHF, convert it to RHF before passing to CASSCF, 
    if isinstance(kmf, scf.kuhf.KUHF):
        kmf = scf.addons.convert_to_rhf(kmf)
    kmc = mc1step.CASSCF(kmf, ncas, nelecas, ncore)
    return kmc

KLASCI = kLASCI
# The kLASCI call function should be added here instead of defining it in kasci.py
