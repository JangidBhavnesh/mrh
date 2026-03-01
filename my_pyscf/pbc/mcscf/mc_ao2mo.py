import numpy as np
from functools import reduce
from pyscf import lib
from pyscf.pbc.lib import kpts_helper


# The 2e integrals transformation to MO basis for the orbital optimization.mk


def _do_ao2mo_direct(kcasscf, mo_kpts, nkpts, ncore, ncas, nmo):
    cell = kcasscf._scf.cell
    kpts = kcasscf._scf.kpts
    mydf = kcasscf._scf.with_df
    nocc = ncore + ncas
    dtype = mo_kpts[0].dtype
    assert len(mo_kpts) == nkpts

    ppaa = np.empty((nkpts, nkpts, nkpts, nmo, nmo, ncas, ncas), dtype=dtype)
    papa = np.empty((nkpts, nkpts, nkpts, nmo, ncas, nmo, ncas), dtype=dtype)
    kconserv = kpts_helper.get_kconserv(cell, kpts)
    for k1 in range(nkpts):
        for k2 in range(nkpts):
            for k3 in range(nkpts):
                k4 = kconserv[k1, k2, k3]
                mo_ppaa = [mo_kpts[k1], mo_kpts[k2], mo_kpts[k3][:, ncore:nocc], mo_kpts[k4][:, ncore:nocc]]
                kp_tuple = [kpts[i] for i in (k1, k2, k3, k4)]
                ppaa[k1, k2, k3] = mydf.ao2mo(mo_ppaa, kp_tuple, compact=False)
                mo_paaa = [mo_kpts[k1], mo_kpts[k2][:, ncore:nocc], mo_kpts[k3], mo_kpts[k4][:, ncore:nocc]]
                papa[k1, k2, k3] = mydf.ao2mo(mo_paaa, kp_tuple, compact=False)

    # This is very naive implementation, would require a lot of optimization.
    j_pc = np.empty((nkpts, nmo, ncore), dtype=dtype)
    k_pc = np.empty((nkpts, nmo, ncore), dtype=dtype)
    for k in range(nkpts):
        mo_ppaa = [mo_kpts[k], mo_kpts[k], mo_kpts[k][:, :ncore], mo_kpts[k][:, :ncore]]
        temp = mydf.ao2mo(mo_ppaa, [kpts[k]]*4, compact=False)
        j_pc[k] = np.einsum('ppjj->pj', temp)
        mo_papa = [mo_kpts[k], mo_kpts[k][:, :ncore], mo_kpts[k][:, ncore:nocc], mo_kpts[k][:, ncore:nocc]]
        temp = mydf.ao2mo(mo_papa, [kpts[k]]*4, compact=False)
        k_pc[k] = np.einsum('pjpj->pj', temp)
    return ppaa, papa, j_pc, k_pc


def _do_ao2mo_disk(kcasscf, mo_kpts, nkpts, ncore, ncas, nmo):
    ppaa = None
    papa = None
    j_pc = None
    k_pc = None
    return j_pc, k_pc

def _mem_usage(nkpts, ncore, ncas, nmo):
    basic = nkpts**3 * nmo**2 * ncas**2 * 16 / 1e6
    incore = basic + nkpts**3 * (ncore+ncas) * nmo**3 * 16 / 1e6
    return incore

class _ERIS:
    '''
    AO2MO transformation of the 2e integrals for the orbital optimization step.
    Args:
        kcasscf: instance of pbc.mcscf.CASSCF
            The K-CASSCF object.
        mo_kpts: list of numpy arrays 
            The MO coefficients for each k-point.
        method: str (direct or disk) (Default is 'direct')
            The method for the 2e integrals transformation. Basically, we require the ppaa and papa integrals, 
            each for nkpt^3 points. If method is 'direct', we will compute the required integrals on the fly. 
            If method is 'disk', we will save the required integrals on disk and read them when required.
        level: int (Default is 1)
            level-1: ppaa, papa, vhf, jpc and kpc
            level-2: Only ppaa, papa and vhf
        '''

    def __init__(self, kcasscf, mo_kpts, method='direct', level=1):
        self.erifile = None
        cell = kcasscf._scf.cell
        kpts = kcasscf._scf.kpts
        ncore = kcasscf.ncore
        ncas = kcasscf.ncas
        nkpts = len(kpts)
        nao, nmo = mo_kpts[0].shape
        dtype = mo_kpts[0].dtype

        dmcore_kpts = np.asarray(
            [2.0 * (mo_kpts[k][:, :ncore] @ mo_kpts[k][:, :ncore].conj().T) 
             for k in range(nkpts)], 
             dtype=dtype)
        
        vj_kpts, vk_kpts = kcasscf._scf.get_jk(cell, dmcore_kpts, kpts=kpts, hermi=1)
        self.vhf_c = np.array(
            [reduce(np.dot, (mo_kpts[k].conj().T, 2.0 * vj_kpts[k] - vk_kpts[k], mo_kpts[k]))
             for k in range(nkpts)], 
             dtype=dtype)

        mem_incore = _mem_usage(nkpts, ncore, ncas, nmo)
        mem_now = lib.current_memory()[0]
        log = lib.logger.Logger(kcasscf.stdout, kcasscf.verbose)
        log.debug('Memory usage for incore ERI transformation: %.2f MB.Current memory usage: %.2f MB. Max memory: %.2f MB.',
              mem_incore, mem_now, kcasscf.max_memory)
        if (method == 'direct' and mem_now + mem_incore < 0.9 * kcasscf.max_memory):
            log.debug('Using direct ERI transformation.')
            self.ppaa, self.papa, self.j_pc, self.k_pc = _do_ao2mo_direct(kcasscf, mo_kpts, nkpts, ncore, ncas, nmo)
        else:
            log.debug('Using disk ERI transformation.')
            self.j_pc, self.k_pc = _do_ao2mo_disk(kcasscf, mo_kpts, nkpts, ncore, ncas, nmo)