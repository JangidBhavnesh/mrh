#!/bin/bash
import copy

from pyscf import mcscf
from pyscf.pbc import scf

from mrh.my_pyscf.pbc import mcscf as pbc_mcscf
from mrh.my_pyscf.pbc.mcscf import _sanity_check_for_kmf
from mrh.my_pyscf.pbc.mcpdft.mcpdft import get_mcpdft_child_class as get_pbc_mcpdft_child_class_gamma
from mrh.my_pyscf.pbc.mcpdft.kmcpdft import (
    get_charged_kcas_mcpdft_child_class,
    get_kcas_mcpdft_child_class,
    get_mcpdft_child_class,
)

# Author: Bhavnesh Jangid

# Implementing MC-PDFT at gamma point and k-MC-PDFT while reusing the
# periodic MCSCF input validation and as much molecular PDFT code as possible.

def _MCPDFT (mc_class, kmc_or_kmf, ot, ncas, nelecas, ncore=None, frozen=None,
            get_mcpdft_child_class=get_mcpdft_child_class,
            allow_frozen=False, **kwargs):
    # Newton SCF objects also expose ``_scf``.  Identify mean-field objects
    # before using that attribute to distinguish an existing MC calculation.
    kmf0 = getattr(kmc_or_kmf, '_scf', None)
    if isinstance(kmc_or_kmf, scf.hf.SCF) or kmf0 is None:
        kmf0 = _sanity_check_for_kmf(kmc_or_kmf)
        kmc0 = None
    else:
        kmf0 = _sanity_check_for_kmf(kmf0)
        kmc0 = kmc_or_kmf

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

def kCASCIPDFT(kmc_or_kmf, ot, ncas, nelecas, ncore=None, frozen=None,
               momentum_resolved=False, target_k=None, charge=None,
               charged_spin=None, **kwargs):
    """Construct conventional or momentum-resolved k-CASCI-PDFT.

    Existing kCASCI objects select the momentum-resolved route automatically.
    For a mean-field input, set ``momentum_resolved=True`` and optionally pass
    ``target_k`` and ``charge``; otherwise conventional periodic CASCI is used.
    """
    from mrh.my_pyscf.pbc.mcscf.kcasci import PBCKCASCI

    is_kcasci = isinstance(kmc_or_kmf, PBCKCASCI)
    momentum_resolved = momentum_resolved or is_kcasci
    if not momentum_resolved:
        if (target_k is not None or charge not in (None, 0)
                or charged_spin is not None):
            raise ValueError(
                "target_k, charge, and charged_spin require "
                "momentum_resolved=True",
            )
        return _MCPDFT(pbc_mcscf.CASCI, kmc_or_kmf, ot, ncas, nelecas,
                       ncore=ncore, frozen=frozen, **kwargs)
    if frozen is not None:
        raise ValueError("Frozen orbitals are not supported in k-MC-PDFT")

    if not is_kcasci:
        if charged_spin is not None and charge in (None, 0):
            raise ValueError("charged_spin requires charge +1 or -1")
        kcas_kwargs = {"ncore": ncore, "target_k": target_k}
        if charge not in (None, 0):
            kcas_kwargs.update(charge=charge, charged_spin=charged_spin)
        kmc = pbc_mcscf.KCASCI(
            kmc_or_kmf, ncas, nelecas, **kcas_kwargs,
        )
    else:
        kmc = kmc_or_kmf
        for name, value, default in (
            ("charge", charge, 0),
            ("charged_spin", charged_spin, None),
        ):
            if value is not None and value != getattr(kmc, name, default):
                raise ValueError(
                    f"{name} conflicts with the existing PBCKCASCI object",
                )
        existing_target = getattr(kmc, "target_k", None)
        if (target_k is not None and existing_target is not None
                and target_k % kmc.nkpts != int(existing_target) % kmc.nkpts):
            raise ValueError(
                "target_k conflicts with the existing PBCKCASCI CI sector",
            )

    make_pdft = (get_charged_kcas_mcpdft_child_class
                 if getattr(kmc, "charge", 0)
                 else get_kcas_mcpdft_child_class)
    pdft = make_pdft(kmc, ot, **kwargs)
    if getattr(kmc, "charge", 0) and target_k is not None:
        pdft.target_k = int(target_k) % pdft.nkpts
    return pdft

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
