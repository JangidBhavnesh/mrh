#!/bin/bash
import copy

from pyscf import mcscf
from pyscf.pbc import scf, dft

from mrh.my_pyscf.pbc import mcscf as pbc_mcscf
from mrh.my_pyscf.pbc.mcpdft.mcpdft import get_mcpdft_child_class as get_pbc_mcpdft_child_class_gamma
from mrh.my_pyscf.pbc.mcpdft.kmcpdft import get_mcpdft_child_class

# Author: Bhavnesh Jangid
# Implementing MC-PDFT at gamma point and k-MC-PDFT. For initialization, I am using different function,.
# (as sanity checks will be different.) However, I will try to import as much code from molecular PDFT 
# and same code structure.

def _sanity_check_for_kmf(kmf0):
    """Validate that the input is a periodic Hartree-Fock object."""
    assert isinstance(kmf0, scf.hf.SCF),  \
        "PBC MC-PDFT only works with periodic SCF objects"

    if isinstance(kmf0, dft.krks.KRKS) or isinstance(kmf0, dft.kuks.KUKS) \
        or isinstance(kmf0, dft.rks.RKS) or isinstance(kmf0, dft.uks.UKS):
        raise NotImplementedError("PBC MC-PDFT only works with periodic HF objects.")
        # In this case, probably one need to regenerate the 3C integrals.

    if isinstance(kmf0, scf.kuhf.KUHF):
        kmf0 = scf.addons.convert_to_rhf(kmf0)
    
    return kmf0

def _MCPDFT (mc_class, kmc_or_kmf, ot, ncas, nelecas, ncore=None, frozen=None,
            get_mcpdft_child_class=get_mcpdft_child_class,
            allow_frozen=False, **kwargs):
    kmf0 = getattr (kmc_or_kmf, '_scf', None)
    
    # If started with kCASCI or kCASSCF object, 
    if kmf0 is not None:
        kmf0 = _sanity_check_for_kmf(kmf0)
        kmc0 = kmc_or_kmf
    else:
        kmf0 = kmc_or_kmf
        kmf0 = _sanity_check_for_kmf(kmf0)
        kmc0 = None

    if frozen is not None and not allow_frozen:
        raise NotImplementedError("Frozen orbitals are not supported in k-MC-PDFT")
    mc_kwargs = {"ncore": ncore}
    if frozen is not None:
        mc_kwargs["frozen"] = frozen
    kmc = get_mcpdft_child_class(
        mc_class(kmf0, ncas, nelecas, **mc_kwargs), ot, **kwargs,
    )

    if kmc0 is not None:
        from mrh.my_pyscf.pbc.mcscf.casci import PBCCASCI
        if isinstance(kmc0, PBCCASCI):
            kmc.kmesh = kmc0.kmesh
            kmc.kpts = kmc0.kpts
        kmc.verbose = kmc0.verbose
        kmc.stdout = kmc0.stdout
        kmc.mo_coeff = kmc_or_kmf.mo_coeff.copy()
        kmc.ci = copy.deepcopy (kmc_or_kmf.ci)
        kmc.converged = kmc0.converged
    return kmc

# For Gamma-point only.
def CASSCFPDFT(kmc_or_kmf, ot, ncas, nelecas, ncore=None, frozen=None, **kwargs):
    get_mcpdft_child_class = get_pbc_mcpdft_child_class_gamma
    return _MCPDFT(mcscf.CASSCF, kmc_or_kmf, ot, ncas, nelecas, ncore=ncore, frozen=frozen,
                    get_mcpdft_child_class=get_mcpdft_child_class,
                    allow_frozen=True,
                **kwargs)

def CASCIPDFT(kmc_or_kmf, ot, ncas, nelecas, ncore=None, frozen=None, **kwargs):
    get_pbc_mcpdft_child_class = get_pbc_mcpdft_child_class_gamma
    return _MCPDFT(mcscf.CASCI, kmc_or_kmf, ot, ncas, nelecas, ncore=ncore, frozen=frozen,
                    get_mcpdft_child_class=get_pbc_mcpdft_child_class,
                **kwargs)

CASSCF = CASSCFPDFT
CASCI = CASCIPDFT

# For k-points
def kCASSCFPDFT(kmc_or_kmf, ot, ncas, nelecas, ncore=None, frozen=None, **kwargs):
    return _MCPDFT(pbc_mcscf.CASSCF, kmc_or_kmf, ot, ncas, nelecas, ncore=ncore, frozen=frozen,
                **kwargs)

def kCASCIPDFT(kmc_or_kmf, ot, ncas, nelecas, ncore=None, frozen=None, **kwargs):
    return _MCPDFT(pbc_mcscf.CASCI, kmc_or_kmf, ot, ncas, nelecas, ncore=ncore, frozen=frozen,
                **kwargs)

KCASSCF = kCASSCFPDFT
KCASCI = kCASCIPDFT


def _laspdftEnergy(mc_class, mc_or_mf, ot, ncas_sub, nelecas_sub,
                   ncore=None, spin_sub=None, frozen=None, **kwargs):
    from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCFNoSymm, LASSCFSymm
    from mrh.my_pyscf.mcscf.lasscf_async import (
        LASSCFNoSymm as AsyncLASSCFNoSymm,
        LASSCFSymm as AsyncLASSCFSymm,
    )
    from mrh.my_pyscf.pbc.mcpdft.laspdft import get_mcpdft_child_class

    las_classes = (
        LASSCFNoSymm,
        LASSCFSymm,
        AsyncLASSCFNoSymm,
        AsyncLASSCFSymm,
    )
    if isinstance(mc_or_mf, las_classes):
        las = mc_or_mf
        if frozen is not None:
            las.frozen = frozen
    else:
        las_kwargs = {"ncore": ncore, "spin_sub": spin_sub, "frozen": frozen}
        las = mc_class(mc_or_mf, ncas_sub, nelecas_sub, **las_kwargs)
    return get_mcpdft_child_class(las, ot, **kwargs)


def LASSCFPDFT(mc_or_mf, ot, ncas_sub=None, nelecas_sub=None, ncore=None,
               spin_sub=None, frozen=None, **kwargs):
    """
    Create a gamma-point periodic LAS-PDFT solver.
    """
    if ncas_sub is None:
        ncas_sub = getattr(mc_or_mf, "ncas_sub", None)
    if nelecas_sub is None:
        nelecas_sub = getattr(mc_or_mf, "nelecas_sub", None)
    from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF as MolecularLASSCF
    return _laspdftEnergy(
        MolecularLASSCF,
        mc_or_mf,
        ot,
        ncas_sub,
        nelecas_sub,
        ncore=ncore,
        spin_sub=spin_sub,
        frozen=frozen,
        **kwargs,
    )


LASSCF = LASSCFPDFT
