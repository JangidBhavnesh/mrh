"""LAS-PDFT support for periodic gamma-point calculations."""

from mrh.my_pyscf.mcpdft import laspdft as molecular_laspdft
from mrh.my_pyscf.pbc.mcpdft.mcpdft import _MCPDFT


class _LASPDFT(_MCPDFT, molecular_laspdft._LASPDFT):
    """Periodic MC-PDFT energy for a localized active-space wavefunction."""

    @property
    def cell(self):
        return self._scf.cell


def get_mcpdft_child_class(las, ot, DoLASSI=False, states=None, **kwargs):
    """Combine a gamma-point LAS solver with periodic LAS-PDFT methods."""
    return molecular_laspdft.get_mcpdft_child_class(
        las,
        ot,
        DoLASSI=DoLASSI,
        states=states,
        _pdft_class=_LASPDFT,
        **kwargs,
    )
