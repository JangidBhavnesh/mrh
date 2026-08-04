import numpy as np

from pyscf.mcpdft.mcpdft import _PDFT
from pyscf.pbc.dft import gen_grid as pbc_gen_grid

from mrh.my_pyscf.pbc.mcpdft.otfnalperiodic import get_pbc_otfnal_gamma

'''
Author: Bhavnesh Jangid
MC-PDFT for periodic systems at the gamma point only.

Note the only difference between this MC-PDFT and the one defined in pyscf/mcpdft/mcpdft.py 
is that the grids are defined for periodic systems.

Previously, this was hosted in the mrh/my_pyscf/mcpdft folder. However, I have moved it to 
this file to make sure periodic MC-PDFT is hosted in right folder.
'''

class _MCPDFT(_PDFT):
    '''
    MC-PDFT for periodic systems at the gamma point only.
    This class is making sure, the functionalities which are not 
    compatible with periodic systems are throwing NotImplementedError. 
    '''

    def _init_ot_grids(self, my_ot, grids_attr=None):
        '''
        Initialization of on-top functional and grids for periodic systems.
        '''
        if grids_attr is None:
            grids_attr = {}

        old_grids = getattr(self, 'grids', None)

        if isinstance(my_ot, (str, np.bytes_)):
            # Note: I have changed the input arg. for below function.
            self.otfnal = get_pbc_otfnal_gamma(self._scf, my_ot)
        else:
            self.otfnal = my_ot

        pbc_grid_types = (
            pbc_gen_grid.UniformGrids,
            pbc_gen_grid.BeckeGrids,
        )

        if isinstance(old_grids, pbc_grid_types):
            self.otfnal.grids = old_grids
        else:
            self.otfnal.grids = pbc_gen_grid.BeckeGrids(self.cell,)

        self.otfnal.grids.__dict__.update(grids_attr)

        for key, value in grids_attr.items():
            assert getattr(self.otfnal.grids, key, None) == value

        self.otfnal.verbose = self.verbose
        self.otfnal.stdout = self.stdout    
    
    def nuc_grad_method(self):
        raise NotImplementedError("Nuclear gradients are not implemented for periodic MC-PDFT yet.")
    
    def dip_moment(self, **kwargs):
        raise NotImplementedError("Dipole moment is not implemented for periodic MC-PDFT yet.")
    
def get_mcpdft_child_class(kmc, ot, **kwargs):
    mc_doc = (kmc.__class__.__doc__ or 'No docstring for MC-SCF parent method')

    class PDFT(_MCPDFT, kmc.__class__):
        __doc__ = mc_doc + '\n\n' + _MCPDFT.__doc__
        _mc_class = kmc.__class__
        # MC-PDFT object requires cell object in ot.reset functions
        _mc_class.cell = kmc._scf.cell
        
    pdft = PDFT(kmc._scf, kmc.ncas, kmc.nelecas, my_ot=ot, **kwargs)
    _keys = pdft._keys.copy()
    pdft.__dict__.update(kmc.__dict__)
    pdft._keys = pdft._keys.union(_keys)
    return pdft
