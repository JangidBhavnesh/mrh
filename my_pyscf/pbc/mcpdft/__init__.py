#!/bin/bash
import copy
from numbers import Integral

from pyscf import mcscf
from pyscf.pbc import scf, dft

from mrh.my_pyscf.pbc import mcscf as pbc_mcscf
from mrh.my_pyscf.pbc.mcpdft.mcpdft import get_mcpdft_child_class as get_pbc_mcpdft_child_class_gamma
from mrh.my_pyscf.pbc.mcpdft.kmcpdft import (
    get_charged_kcas_mcpdft_child_class,
    get_kcas_mcpdft_child_class,
    get_mcpdft_child_class,
)

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
    """Construct conventional or total-momentum-resolved k-CASCI-PDFT.

    Set ``momentum_resolved=True`` to use ``PBCKCASCI`` and select its total
    momentum sector with ``target_k``.  Set ``charge`` to ``+1`` or ``-1``
    for electron removal or addition.  Omitting ``target_k`` for a charged
    calculation evaluates every stored momentum sector.  The default retains
    the established Wannier-basis periodic CASCI-PDFT implementation.
    """
    if not isinstance(momentum_resolved, bool):
        raise ValueError("momentum_resolved must be True or False")

    if not momentum_resolved:
        if target_k is not None:
            raise ValueError(
                "target_k requires momentum_resolved=True",
            )
        if charge not in (None, 0) or charged_spin is not None:
            raise ValueError(
                "charge and charged_spin require momentum_resolved=True",
            )
        return _MCPDFT(
            pbc_mcscf.CASCI, kmc_or_kmf, ot, ncas, nelecas,
            ncore=ncore, frozen=frozen, **kwargs,
        )

    if target_k is not None and not isinstance(target_k, Integral):
        raise ValueError("target_k must be an integer or None")
    if charge is not None and not isinstance(charge, Integral):
        raise ValueError("charge must be an integer or None")
    if charge is not None:
        charge = int(charge)
        if charge not in (-1, 0, 1):
            raise ValueError("charge must be -1, 0, +1, or None")
    if charged_spin is not None and not isinstance(charged_spin, Integral):
        raise ValueError("charged_spin must be an integer or None")
    if charged_spin is not None:
        charged_spin = int(charged_spin)
    if charge == 0 and charged_spin is not None:
        raise ValueError("charged_spin requires charge +1 or -1")

    if frozen is not None:
        raise ValueError("Frozen orbitals are not supported in k-MC-PDFT")

    from mrh.my_pyscf.pbc.mcscf.kcasci import PBCKCASCI

    # A second-order SCF wrapper has an ``_scf`` attribute too, so test the
    # public SCF type before treating the input as an existing PBCKCASCI.
    kmf = getattr(kmc_or_kmf, "_scf", None)
    if isinstance(kmc_or_kmf, scf.hf.SCF) or kmf is None:
        if charged_spin is not None and charge not in (-1, 1):
            raise ValueError("charged_spin requires charge +1 or -1")
        kmf = _sanity_check_for_kmf(kmc_or_kmf)
        if charge in (-1, 1):
            kmc = pbc_mcscf.KCASCI(
                kmf, ncas, nelecas, ncore=ncore, charge=charge,
                target_k=target_k, charged_spin=charged_spin,
            )
        else:
            sector = 0 if target_k is None else target_k
            kmc = pbc_mcscf.KCASCI(
                kmf, ncas, nelecas, ncore=ncore, target_k=sector,
            )
    else:
        if not isinstance(kmc_or_kmf, PBCKCASCI):
            raise TypeError(
                "momentum_resolved=True requires a mean-field object or "
                "an existing PBCKCASCI object",
            )
        _sanity_check_for_kmf(kmf)
        kmc = kmc_or_kmf
        existing_charge = int(getattr(kmc, "charge", 0))
        if charge is not None and charge != existing_charge:
            raise ValueError(
                "charge conflicts with the existing PBCKCASCI object",
            )
        if charged_spin is not None:
            existing_spin = getattr(kmc, "charged_spin", None)
            if existing_spin != charged_spin:
                raise ValueError(
                    "charged_spin conflicts with the existing PBCKCASCI "
                    "object",
                )
        if target_k is not None:
            requested_sector = target_k % kmc.nkpts
            existing_target = getattr(kmc, "target_k", None)
            if (existing_target is not None
                    and requested_sector != int(existing_target) % kmc.nkpts):
                raise ValueError(
                    "target_k conflicts with the existing PBCKCASCI CI sector",
                )

    if getattr(kmc, "charge", 0):
        pdft = get_charged_kcas_mcpdft_child_class(kmc, ot, **kwargs)
        if target_k is not None:
            pdft.target_k = int(target_k) % pdft.nkpts
        return pdft
    return get_kcas_mcpdft_child_class(kmc, ot, **kwargs)

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


def _validate_klas_pdft_input(klas, method):
    """Validate an intake-only periodic LASCI or LASSCF PDFT request."""
    from mrh.my_pyscf.pbc.mcscf.klasci import (
        PBCLASCINoSymm,
        PBCLASCITransSymm,
    )
    from mrh.my_pyscf.pbc.mcscf.klasscf import PBCLASSCFNoSymm

    if method == "KLASCI":
        valid = isinstance(klas, PBCLASCINoSymm) and not isinstance(
            klas, PBCLASSCFNoSymm,
        )
    elif method == "KLASSCF":
        valid = isinstance(klas, PBCLASSCFNoSymm)
    else:
        raise ValueError(f"Unknown kLAS-PDFT method: {method}")
    if not valid:
        raise TypeError(
            f"mcpdft.{method} requires an existing {method} object",
        )
    if isinstance(klas, PBCLASCITransSymm) or getattr(
            klas, "trans_sym", False):
        raise NotImplementedError(
            "translation-packed kLAS-PDFT is not implemented",
        )
    if int(getattr(klas, "nroots", 0)) != 1:
        raise NotImplementedError(
            "The initial kLAS-PDFT implementation supports one root",
        )
    if getattr(klas, "mo_coeff", None) is None:
        raise ValueError("The kLAS object has no molecular orbitals")
    if getattr(klas, "ci", None) is None:
        raise ValueError("The kLAS object has no CI vectors")

    nkpts = len(klas.kpts)
    ncastot = nkpts * int(klas.ncas)
    if int(sum(klas.ncas_sub)) != ncastot:
        raise ValueError(
            "sum(ncas_sub) must equal nkpts * ncas for kLAS-PDFT",
        )
    return klas


def kLASCIPDFT(klas, ot, **kwargs):
    """Create a fixed-wavefunction PDFT evaluator from an existing KLASCI."""
    from mrh.my_pyscf.pbc.mcpdft.klaspdft import (
        get_klas_mcpdft_child_class,
    )
    _validate_klas_pdft_input(klas, "KLASCI")
    return get_klas_mcpdft_child_class(klas, ot, **kwargs)


def kLASSCFPDFT(klas, ot, **kwargs):
    """Create a fixed-wavefunction PDFT evaluator from an existing KLASSCF."""
    from mrh.my_pyscf.pbc.mcpdft.klaspdft import (
        get_klas_mcpdft_child_class,
    )
    _validate_klas_pdft_input(klas, "KLASSCF")
    return get_klas_mcpdft_child_class(klas, ot, **kwargs)


KLASCI = kLASCIPDFT
KLASSCF = kLASSCFPDFT
