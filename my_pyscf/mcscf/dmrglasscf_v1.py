# It's time to introduce the DMRG and LASSCF methods to each other.

from pyscf import __config__
import os
from pyscf import gto, lib

try:
    # DMRGSCF is a module which connects the PySCF MCSCF module to the various DMRG implementations.
    # For example: block2, block, CheMPS2, etc.
    from pyscf import dmrgscf
    from pyscf.dmrgscf import settings
    # Based on LASSCF philosophy, we will try to use the spin_adapted DMRGCI code only.
    assert ('spin_adapted' in settings.BLOCKEXE or 'block2main' in settings.BLOCKEXE), \
        "Please recompile the block2 with spin-adapted code.."
    # If the user has not set the BLOCKSCRATCHDIR variable, then we will set it to the PySCF temporary directory.
    settings.BLOCKSCRATCHDIR = getattr(__config__, 'dmrgscf_BLOCKSCRATCHDIR', lib.param.TMPDIR)

except ImportError:
    print("DMRGSCF module not found. \
        Please install that module: https://github.com/pyscf/dmrgscf")

try:
    # I think, block2 is the latest code, and active support is being provided for that code.
    # That's why I will tie the LASSCF to that code only.
    import block2
except ImportError:
    print("Block2 module not found. \
        Please install that module:https://github.com/block-hczhai/block2-preview.git")

from pyscf.mcscf import mc1step
from pyscf.dmrgscf.dmrgci import DMRGCI
from mrh.my_pyscf.mcscf import lasci
from mrh.my_pyscf.mcscf.addons import state_average_n_mix, get_h1e_zipped_fcisolver
from mrh.my_pyscf.mcscf import lasci

class DMRGLASCI(DMRGCI):
    '''
    Wrapper class for DMRGCI to perform LASSCF calculations.
    Only changing variables which are important for the LASSCF, otherwise keep it the same as DMRGCI.
    '''
    def __init__(self, mol, maxM:int=512, tol:float=1e-7, num_thrds:int=lib.num_threads(), memory:int=None, fragno:int=0, **kwargs):
        super().__init__(mol, maxM=maxM, tol=tol, num_thrds=num_thrds, memory=memory, **kwargs)
        self.fragno = fragno
        self.scratchDirectory = os.path.abspath(settings.BLOCKSCRATCHDIR)+f'/frag{self.fragno}'
        self.integralFile = "FCIDUMP" + f"_frag{self.fragno}"
        self.configFile = "dmrg.conf" + f"_frag{self.fragno}"
        self.outputFile = "dmrg.out" + f"_frag{self.fragno}"

    def check_transformer_cache(self,  **kwargs):
        pass

class DMRGLASCINoSymm(lasci.LASCINoSymm):
    def __init__(self, mf, ncas, nelecas, ncore=None, spin_sub=None, frozen=None, frozen_ci=None, **kwargs):
        super().__init__(mf, ncas, nelecas, ncore=ncore, spin_sub=spin_sub, frozen=frozen, frozen_ci=frozen_ci, **kwargs)
        self.maxM = kwargs.get('maxM', 512)
        self.tol = kwargs.get('tol', 1e-7)
        self.num_thrds = kwargs.get('num_thrds', lib.num_threads())
        # I will overwrite the fciboxes attribute of the LASCINoSymm class, which is a list of fci solvers for each fragment. In DMRG-LASSCF, we will use DMRGCI as the fci solver for each fragment, and we will initialize it in the constructor of this class.
        self.fciboxes = []
        if isinstance(spin_sub, int):
            self.fciboxes.append(self._init_dmrgcibox(spin_sub,self.nelecas_sub[0], fragno=0))
        else:
            assert (len (spin_sub) == self.nfrags)
            for fragno, (smult, nel) in enumerate(zip (spin_sub, self.nelecas_sub)):
                self.fciboxes.append (self._init_dmrgcibox (smult, nel, fragno=fragno)) 

    def _init_dmrgcibox(self, smult, nel, fragno):
        solver = DMRGLASCI(self._scf.mol, maxM=self.maxM, tol=self.tol, num_thrds=self.num_thrds, fragno=fragno)
        solver.spin = nel[0] - nel[1] # smult - 1 is also possible, but probably this is more consistent with LAS code.
        return get_h1e_zipped_fcisolver (state_average_n_mix (self, [solver], [1.0]).fcisolver)

    def get_init_guess_ci(self, mo_coeff, ci0=None, eri_cas=None, **kwargs):
        return [[None for i in range (self.nroots)] for j in range (self.nfrags)]

class DMRGLASSCFNoSymm (DMRGLASCINoSymm):
    from mrh.my_pyscf.mcscf.lasscf_sync_o0 import LASSCFNoSymm
    _ugg = LASSCFNoSymm._ugg
    _hop = LASSCFNoSymm._hop
    as_scanner = mc1step.as_scanner    
    split_veff = LASSCFNoSymm.split_veff
    Gradients = NotImplementedError

    def dump_flags (self, verbose=None, _method_name='DMRGLASSCF'):
        lasci.LASCINoSymm.dump_flags (self, verbose=verbose, _method_name=_method_name)
    
def LASSCF (mf_or_mol, ncas_sub, nelecas_sub, **kwargs):
    use_gpu = kwargs.get('use_gpu', None)
    if isinstance(mf_or_mol, gto.Mole):
        mf = scf.RHF(mf_or_mol)
    elif isinstance (mf_or_mol, scf.hf.SCF):
        mf = mf_or_mol
    else:
        raise RuntimeError ("LASSCF constructor requires molecule or SCF instance")
    if mf.mol.symmetry:
        log = lib.logger.new_logger(mf, verbose)
        log.warn("Symmetry is detected in the molecule, but DMRG-LASSCF \
            with symmetry is not yet implemented. Switching to without symm algorithm.")
  
    las = DMRGLASSCFNoSymm (mf, ncas_sub, nelecas_sub, **kwargs)

    if getattr (mf, 'with_df', None):
        las = lasci.density_fit (las, with_df = mf.with_df)

    return las

if __name__ == '__main__':
    from pyscf import scf, lib, tools, mcscf
    from mrh.tests.lasscf.me2n2_struct import structure as struct
    mol = struct (2.0, '6-31g')
    mol.output = 'lasscf_sync_o0.log'
    mol.verbose = lib.logger.DEBUG
    mol.build ()

    mf = scf.RHF (mol).run ()

    las = LASSCF (mf, (4,), ((2,2),), spin_sub=(1,))
    print(las.fciboxes)
    las.lasci_()
    print(las.__class__)