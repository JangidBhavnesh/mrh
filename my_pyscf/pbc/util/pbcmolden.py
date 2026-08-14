import numpy as np

from pyscf import lib
from pyscf.pbc.tools import k2gamma
from pyscf.pbc.tools.pbc import super_cell
from pyscf.tools import molden

from mrh.my_pyscf.pbc.mcscf.k2R import get_mo_coeff_k2R
from mrh.my_pyscf.pbc.util.wannier import get_wannier_orbs

# Author: Bhavnesh Jangid
# Some wrapper functions to print the orbitals from k-SCF, k-CASSCF, or
# k-LASCI calculations to a Molden file.


def _get_kmesh(kmf, kmesh):
    '''Get and sanity check the k-point mesh.'''
    if kmesh is None:
        kmesh = k2gamma.kpts_to_kmesh(kmf.cell, kmf.kpts)
    kmesh = tuple(np.asarray(kmesh, dtype=int).tolist())
    assert np.prod(kmesh) == len(kmf.kpts), \
        "kmesh and number of kpts in kmf do not match"
    return kmesh


def _get_orbital_property(prop, nkpts, nmo, orb_slice, wannier):
    '''Slice and reshape orbital energies or occupations for Molden.'''
    if prop is None:
        return None

    prop = np.asarray(prop)
    nsel = len(range(*orb_slice.indices(nmo)))
    if prop.ndim == 1 and prop.size == nkpts*nsel:
        return prop
    if prop.shape == (nmo,):
        prop = np.broadcast_to(prop, (nkpts, nmo))
    if prop.shape != (nkpts, nmo):
        msg = (f"orbital property must have shape ({nmo},) or "
               f"({nkpts}, {nmo}); got {prop.shape}")
        raise ValueError(msg)

    prop = prop[:, orb_slice]
    if wannier:
        # A Wannier orbital does not have a unique k-point orbital energy or
        # occupation. Values can be retained only when they are k-independent.
        if not np.allclose(prop, prop[0], atol=1e-10, rtol=0.0):
            return None
        return np.tile(prop[0], nkpts)
    return prop.reshape(-1)


def from_mo(kmf, filename, mo_coeff=None, kmesh=None, wannier=False,
            only_active=False, ncore=None, ncas=None, occ=None, ene=None,
            **kwargs):
    '''
    Print complex k-point molecular orbitals in Molden format.
    First transform the complex k-point orbitals to the Born-von Karman
    supercell. The orbitals can be transformed either to the k-to-R basis or
    to the Wannier basis before the real part is written to the Molden file.

    args:
        kmf: pbc.scf object
            Mean-field object containing the cell and k-points.
        filename: str
            Name of the output Molden file.

    kwargs:
        mo_coeff: list or np.ndarray (nkpts, nao, nmo)
            Complex MO coefficients at each k-point. If None, ``kmf.mo_coeff``
            is used.
        kmesh: tuple
            K-point mesh for the calculation. If None, it is inferred from
            ``kmf.kpts``.
        wannier: bool
            If True, transform the selected orbitals to the Wannier basis. If
            False, use the standard k-to-R transformation.
        only_active: bool
            If True, print only ``mo_coeff[:, :, ncore:ncore+ncas]``.
        ncore: int
            Number of core orbitals per unit cell. Required when
            ``only_active=True``.
        ncas: int
            Number of active orbitals per unit cell. Required when
            ``only_active=True``.
        occ: list or np.ndarray
            Orbital occupations at each k-point.
        ene: list or np.ndarray
            Orbital energies at each k-point.
        kwargs:
            Additional arguments passed to ``pyscf.tools.molden.from_mo``.

    returns:
        None
    '''
    if mo_coeff is None:
        mo_coeff = kmf.mo_coeff
    mo_coeff = np.asarray(mo_coeff)
    if mo_coeff.ndim != 3:
        msg = ("mo_coeff must have shape (nkpts, nao, nmo); "
               f"got {mo_coeff.shape}")
        raise ValueError(msg)

    nkpts, nao, nmo = mo_coeff.shape
    if nkpts != len(kmf.kpts):
        msg = (f"mo_coeff contains {nkpts} k-points; "
               f"expected {len(kmf.kpts)}")
        raise ValueError(msg)
    if nao != kmf.cell.nao_nr():
        msg = f"mo_coeff AO dimension is {nao}; expected {kmf.cell.nao_nr()}"
        raise ValueError(msg)

    kmesh = _get_kmesh(kmf, kmesh)

    # Select the orbitals before doing the k-to-R or k-to-Wannier
    # transformation. This avoids constructing unused supercell orbitals.
    if only_active:
        if ncore is None or ncas is None:
            raise ValueError(
                "ncore and ncas are required when only_active=True"
            )
        orb_slice = slice(ncore, ncore+ncas)
    else:
        orb_slice = slice(0, nmo)
    mo_selected = mo_coeff[:, :, orb_slice]
    nmo_selected = mo_selected.shape[2]

    if wannier:
        wannier_orb = get_wannier_orbs(kmf, kmesh, mo_selected)[0]
        mo_coeff_R = wannier_orb.reshape(
            nkpts*nao, nkpts*nmo_selected,
        )
        scell = super_cell(kmf.cell, kmesh)
    else:
        scell, _, mo_coeff_R = get_mo_coeff_k2R(
            kmf, mo_selected, 0, nmo_selected, kmesh=kmesh,
        )[:3]

    occ_R = _get_orbital_property(
        occ, nkpts, nmo, orb_slice, wannier,
    )
    ene_R = _get_orbital_property(
        ene, nkpts, nmo, orb_slice, wannier,
    )
    if wannier:
        # Wannier orbitals are generally not eigenvectors of the Fock matrix,
        # so the individual k-point orbital energies are not meaningful.
        ene_R = None

    # The Molden format and the PySCF Molden writer support real coefficients
    # only. The imaginary component cannot be represented in this file format.
    if np.max(np.abs(mo_coeff_R.imag), initial=0.0) > 1e-8:
        lib.logger.warn(
            kmf, "Molden does not support complex MO coefficients; "
            "only their real part is written",
        )
    molden.from_mo(
        scell, filename, mo_coeff_R.real, occ=occ_R, ene=ene_R, **kwargs,
    )
    return None


def from_scf(kmf, filename, kmesh=None, wannier=False, only_active=False,
             ncore=None, ncas=None, **kwargs):
    '''
    Print orbitals from a periodic k-point SCF calculation.
    For documentation of the options, see ``from_mo``.
    '''
    return from_mo(
        kmf, filename, mo_coeff=kmf.mo_coeff, kmesh=kmesh,
        wannier=wannier, only_active=only_active, ncore=ncore, ncas=ncas,
        occ=getattr(kmf, 'mo_occ', None),
        ene=getattr(kmf, 'mo_energy', None), **kwargs,
    )


def from_lasscf(las, filename, kmesh=None, wannier=True, only_active=True,
                **kwargs):
    '''
    Print orbitals from a periodic LASCI or LASSCF calculation.
    By default, print only the active orbitals in the Wannier basis.
    For documentation of the options, see ``from_mo``.
    '''
    return from_mo(
        las._scf, filename, mo_coeff=las.mo_coeff,
        kmesh=las.kmesh if kmesh is None else kmesh,
        wannier=wannier, only_active=only_active,
        ncore=las.ncore, ncas=las.ncas,
        occ=getattr(las, 'mo_occ', None),
        ene=getattr(las, 'mo_energy', None), **kwargs,
    )


def from_kcasscf(kmc, filename, kmesh=None, wannier=False,
                 only_active=False, **kwargs):
    '''
    Print orbitals from a k-CASCI or k-CASSCF calculation.
    By default, print all orbitals using the standard k-to-R transformation.
    For documentation of the options, see ``from_mo``.
    '''
    return from_mo(
        kmc._scf, filename, mo_coeff=kmc.mo_coeff,
        kmesh=kmc.kmesh if kmesh is None else kmesh,
        wannier=wannier, only_active=only_active,
        ncore=kmc.ncore, ncas=kmc.ncas,
        occ=getattr(kmc, 'mo_occ', None),
        ene=getattr(kmc, 'mo_energy', None), **kwargs,
    )


# Backward-compatible name for k-CASCI calculations.
from_kcasci = from_kcasscf


def print_molden(kmf, mo_coeff, kmesh, filename, only_occ=False,
                 only_virt=False):
    '''
    Print molecular orbitals in Molden format.
    This function is retained for backward compatibility. New code should use
    ``from_mo`` or ``from_scf``.
    '''
    if only_occ and only_virt:
        raise ValueError("only_occ and only_virt cannot both be True")

    nelec = kmf.cell.nelectron
    nocc = nelec // 2 + nelec % 2
    if only_occ:
        return from_mo(
            kmf, filename, mo_coeff=mo_coeff, kmesh=kmesh,
            only_active=True, ncore=0, ncas=nocc,
        )
    if only_virt:
        mo_coeff = np.asarray(mo_coeff)[:, :, nocc:]
    return from_mo(kmf, filename, mo_coeff=mo_coeff, kmesh=kmesh)


def print_molden_only_as(kmf, mo_coeff, kmesh, filename, ncas, ncore):
    '''
    Print only the active space orbitals in Molden format.
    This function is retained for backward compatibility. New code should use
    ``from_mo(..., only_active=True)``.
    '''
    return from_mo(
        kmf, filename, mo_coeff=mo_coeff, kmesh=kmesh,
        only_active=True, ncore=ncore, ncas=ncas,
    )


def print_molden_natorbs(kmc, kmesh, filename):
    '''
    Print natural orbitals in Molden format.
    args:
        kmc: k-CASSCF object
            k-CASSCF object containing the natural orbitals.
        kmesh: tuple
            k-point mesh for the calculation.
        filename: str
            Name of the output Molden file.
    '''
    scell, _, mo_coeff_R = get_mo_coeff_k2R(
        kmc._scf, kmc.mo_coeff, kmc.ncore, kmc.ncas, kmesh=kmesh,
    )[:3]
    nkpts = np.prod(kmesh)
    ncastot = kmc.ncas * nkpts
    nelecastot = (kmc.nelecas[0] * nkpts, kmc.nelecas[1] * nkpts)
    dm1 = kmc.fcisolver.make_rdm1(kmc.ci, ncastot, nelecastot)
    noon, natorb = np.linalg.eigh(dm1)
    idx = np.argsort(noon)[::-1]
    natorb = natorb[:, idx]
    mo_coeff_R = mo_coeff_R @ natorb
    molden.from_mo(scell, filename, mo_coeff_R.real, occ=noon[idx])
    return None


def print_molden_wannier(wannier_orb, cell, kmesh, filename, occ=None):
    '''
    Print Wannier orbitals in Molden format.
    args:
        wannier_orb: np.ndarray (ncell*nao, nwann)
            Wannier orbital coefficients in real space.
        cell: pyscf.pbc.Cell
            Cell object containing the lattice information.
        kmesh: tuple
            K-point mesh for the calculation.
        filename: str
            Name of the output Molden file.
    '''
    scell = super_cell(cell, kmesh)
    molden.from_mo(scell, filename, wannier_orb.real, occ=occ)
    return None
