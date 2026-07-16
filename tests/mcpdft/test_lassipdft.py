import unittest
from pyscf import lib, gto, scf
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.lassi import LASSI
from mrh.my_pyscf.lassi.spaces import all_single_excitations
from mrh.my_pyscf.mcscf.lasci import get_space_info

class KnownValues(unittest.TestCase):
    def _test_h4_cascipdft_limit (self):
        xyz='''H 0 0 0
               H 1 0 0
               H 3 0 0
               H 4 0 0'''

        mol = gto.M (atom=xyz, basis='sto3g', symmetry=False, verbose=0, output='/dev/null')
        mf = scf.RHF (mol).run ()
        
        # LASSCF and LASSI
        las = LASSCF (mf, (2,2), (2,2), spin_sub=(1,1))
        las.lasci ()
        las1 = las
        for i in range (2): las1 = all_single_excitations (las1)
        charges, spins, smults, wfnsyms = get_space_info (las1)
        lroots = 4 - smults
        idx = (charges!=0) & (lroots==3)
        lroots[idx] = 1
        las1.conv_tol_grad = las.conv_tol_self = 9e99
        las1.lasci (lroots=lroots.T)
        las1.dump_spaces ()
        # CASCI limit
        from pyscf import mcpdft
        mc = mcpdft.CASCI (mf, 'tPBE', 4, 4).run ()
        for opt in range (2):
            with self.subTest (opt=opt):
                lsi = LASSI (las1)
                lsi.opt = opt
                lsi.kernel ()
                self.assertAlmostEqual (lsi.e_roots[0], mc.e_mcscf, 7)
                from mrh.my_pyscf import mcpdft
                lsipdft = mcpdft.LASSI (lsi, 'tPBE')
                lsipdft.opt = opt
                lsipdft.kernel()
                self.assertAlmostEqual (lsipdft.e_tot[0], mc.e_tot, 7)
    
    # Note passing:
    def test_h4_lpdft_limit (self):
        xyz='''H 0 0 0
               H 1 0 0
               H 3 0 0
               H 4 0 0'''

        mol = gto.M (atom=xyz, basis='sto3g', symmetry=False, verbose=4)
        mf = scf.RHF (mol).run ()
        
        # LASSCF and LASSI
        las = LASSCF (mf, (2,2), (2,2), spin_sub=(1,1))
        las.lasci ()
        las1 = las
        for i in range (2): las1 = all_single_excitations (las1)
        charges, spins, smults, wfnsyms = get_space_info (las1)
        lroots = 4 - smults
        idx = (charges!=0) & (lroots==3)
        lroots[idx] = 1
        las1.conv_tol_grad = las.conv_tol_self = 9e99
        las1.lasci (lroots=lroots.T)
        las1.dump_spaces ()
        # CASCI limit
        # L-PDFT with CASCI requires the dump_chk to be False.
        
        from mrh.my_pyscf.fci import csf_solver
        from pyscf import mcpdft, mcscf

        mc = mcscf.CASCI (mf, 4, 4)
        mc.fcisolver.nroots = 36
        e_mcscf = mc.kernel()[0]

        RDMs = [mc.fcisolver.make_rdm1s(mc.ci[i], mc.ncas, mc.nelecas) for i in range(len(mc.ci))]
        RDM2 = [mc.fcisolver.make_rdm12(mc.ci[i], mc.ncas, mc.nelecas)[1] for i in range(len(mc.ci))]
        
        mc = mcpdft.CASSCF (mf, 'tPBE', 4, 4).state_average_([1/36, ]*36)
        mc = mc.multi_state('lin')
        mc.kernel ()

        RDMsComp = []
        RDM2Comp = []

        import numpy as np

        for opt in range (1):
            with self.subTest (opt=opt):
                lsi = LASSI (las1)
                lsi.opt = opt
                lsi.kernel ()
               
                RDMsComp = [lsi.make_rdm1s(state=i) for i in range(len(lsi.e_roots))]
                RDM2Comp = [lsi.make_casdm2(state=i) for i in range(len(lsi.e_roots))]
                
                # for i in range(len(lsi.e_roots)):
                #     # print(np.max(np.abs(np.array(RDMsComp[i]) - np.array(RDMs[i]))))
                #     print(np.max(np.abs(np.array(RDM2Comp[i]) - np.array(RDM2[i]))))
                #     # np.testing.assert_allclose(np.array(RDMsComp[i]), np.array(RDMs[i]), rtol=1e-7, atol=1e-7)
                #     # np.testing.assert_allclose(np.array(RDM2Comp[i]), np.array(RDM2[i]), rtol=1e-7, atol=1e-7)

                from mrh.my_pyscf import mcpdft
                lsipdft = mcpdft.LASSI (lsi, 'tPBE', verbose=4)
                lsipdft.opt = opt
                lsipdft = lsipdft.multi_state()
                lsipdft.kernel()

                for j, e_root in enumerate(e_mcscf):
                    self.assertAlmostEqual (lsi.e_roots[j], e_root, 7)
                    self.assertAlmostEqual (lsipdft.e_states[0], mc.e_states[0], 7)

                del lsi
if __name__ == "__main__":
    print("Full Tests for LASSI-PDFT")
    unittest.main()


