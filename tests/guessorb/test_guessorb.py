import unittest
import numpy as np
from pyscf import gto
from mrh.my_pyscf.guessorb import guessorb

'''
To check the accuracy of the implementation, I am comparing the mo_energy
calculated to the OpenMolcas guessorb module.

That can be read from the file.guessorb.h5 like this:

import h5py
import numpy as np

with h5py.File(file.guessorb.h5, "r") as hdf_file:
    item = hdf_file["MO_ENERGIES"]
    print(np.asarray(item))

Note that I have used OpenMolcas
version: 082a19c42-dirty
commit: 082a19c42dded5cb87a081429237d2937ecec3fd
'''

'''
1. Testing the mo_energy against the OpenMolcas 
and current implementation.
2. Orthonormalization testing is done by default.
3. Model Fock Matrix in AO basis against OpenMolcas
'''

class KnownValues(unittest.TestCase):
    def test_NAtom(self):
        mol = gto.M(atom = '''N 0 0 0''',
        basis = 'STO-3G',
        verbose = 1,
        spin=3)
        mol.output = '/dev/null'
        mol.build()

        mo_energy, mo_coeff = guessorb.get_guessorb(mol)

        # Calculated from OpenMolcas: 082a19c42dded5cb87a081429237d2937ecec3fd
        mo_energy_ref = [-15.6267, -0.9432, -0.5593, -0.5593, -0.5593]
        [self.assertAlmostEqual(energy, energy_ref, 2) \
        for energy, energy_ref in zip(mo_energy, mo_energy_ref)]

        # These values are generated with this code.
        # mrh: 3ddcaf20878b0f6c64518efc42c0f70cb579fa63
        # pyscf: 6f6d3741bf42543e02ccaa1d4ef43d9bf83b3dda
        mo_energy_bench = [-15.62670557,  -0.94322069,  -0.55938518,  -0.55938518,  -0.55938518]
        [self.assertAlmostEqual(energy, energy_ref, 2) \
        for energy, energy_ref in zip(mo_energy, mo_energy_bench)]

        # Model Fock Matrix in AO basis calculated from OpenMolcas.
        fockao_ref = np.array([[-15.51166808, -0.11072152,  0.,  0.,  0.],
            [-0.11072152,  -1.00614293,  0.,  0.,  0.],
            [0.,  0.,  -0.56231464,  0.,  0.],
            [0.,  0.,  0.,  -0.56231464,  0.],
            [0.,  0.,  0.,  0.,  -0.56231464]])

        fockao = guessorb.get_model_fock(mol)

        self.assertTrue(np.allclose(fockao, fockao_ref, 4))


    def test_CO2(self):
        mol = gto.M(atom ='''
        C 0.000000 0.000000 0.000000
        O 0.000000 0.000000 1.155028
        O 0.000000 0.000000 -1.155028
        ''',
        basis = 'STO-3G',
        verbose = 1)
        mol.output = '/dev/null'
        mol.build()

        mo_energy, mo_coeff = guessorb.get_guessorb(mol)
       
        mo_energy_ref = [-20.6800278, -20.67878  , -11.3383716, -1.60247219, -1.45419645,
                        -0.76608563, -0.71940002, -0.71940002, -0.67930921, -0.61218369,
                        -0.61218369, -0.32625724, -0.32625724, -0.21767392, -0.10280624]
        
        [self.assertAlmostEqual(energy, energy_ref, 2) 
        for energy, energy_ref in zip(mo_energy, mo_energy_ref)]

        # These values are generated with this code.
        # mrh: 3ddcaf20878b0f6c64518efc42c0f70cb579fa63
        # pyscf: 6f6d3741bf42543e02ccaa1d4ef43d9bf83b3dda
        mo_energy_bench = [-20.68002789, -20.67878,    -11.3383716, -1.60247216,-1.45419646,
                           -0.76608562,  -0.71940001,  -0.71940001, -0.6793092, -0.61218369,
                           -0.61218369,  -0.32625724,  -0.32625724, -0.21767394,-0.10280624]

        [self.assertAlmostEqual(energy, energy_ref, 2) \
        for energy, energy_ref in zip(mo_energy, mo_energy_bench)]

        fockao_ref = np.array([
            [-11.25544661,  -0.03267994,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
            [-0.03267994,  -0.75369347,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
            [0.,   0.,  -0.42669298,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
            [0.,   0.,   0.,  -0.42669298,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
            [0.,   0.,   0.,   0.,  -0.42669298,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
            [0.,   0.,   0.,   0.,   0., -20.58457027,  -0.02240635,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
            [0.,   0.,   0.,   0.,   0.,  -0.02240635,  -1.32307697,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
            [0.,   0.,   0.,   0.,   0.,   0.,   0.,  -0.61557399,   0.,   0.,   0.,   0.,   0.,   0.,   0.],
            [0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  -0.61557399,   0.,   0.,   0.,   0.,   0.,   0.],
            [0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  -0.61557399,   0.,   0.,   0.,   0.,   0.],
            [0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., -20.58457027,  -0.02240635,   0.,   0.,   0.],
            [0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  -0.02240635,  -1.32307697,   0.,   0.,   0.],
            [0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  -0.61557399,   0.,   0.],
            [0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  -0.61557399,   0.],
            [0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  -0.61557399]])

        fockao = guessorb.get_model_fock(mol)

        self.assertTrue(np.allclose(fockao, fockao_ref, 4))

if __name__ == "__main__":
    print("Full Tests for GuessOrb")
    unittest.main()
