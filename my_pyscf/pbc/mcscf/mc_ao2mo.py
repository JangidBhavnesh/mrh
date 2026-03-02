import numpy as np
from functools import reduce
from pyscf import lib
from pyscf.ao2mo import _ao2mo
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.df import df_ao2mo
from pyscf.pbc.df.df import _load3c


_mo_as_complex = df_ao2mo._mo_as_complex
_conc_mos = df_ao2mo._conc_mos


# The 2e integrals transformation to MO basis for the orbital optimization.mk

def _do_ao2mo_direct(kcasscf, mo_kpts, nkpts, ncore, ncas, nmo, level=1):
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
    if level == 1:
        j_pc = np.empty((nkpts, nmo, ncore), dtype=dtype)
        k_pc = np.empty((nkpts, nmo, ncore), dtype=dtype)
        for k in range(nkpts):
            mo_ppaa = [mo_kpts[k], mo_kpts[k], mo_kpts[k][:, :ncore], mo_kpts[k][:, :ncore]]
            temp = mydf.ao2mo(mo_ppaa, [kpts[k]]*4, compact=False)
            j_pc[k] = np.einsum('ppjj->pj', temp)
            mo_papa = [mo_kpts[k], mo_kpts[k][:, :ncore], mo_kpts[k][:, ncore:nocc], mo_kpts[k][:, ncore:nocc]]
            temp = mydf.ao2mo(mo_papa, [kpts[k]]*4, compact=False)
            k_pc[k] = np.einsum('pjpj->pj', temp)
    else:
        j_pc = None
        k_pc = None
    return ppaa, papa, j_pc, k_pc

def get_nauxlist(mydf, kpts, nkpts):
    nauxlist = {}
    for k1 in range(nkpts):
        for k2 in range(nkpts):
            kpti_kptj = np.vstack((kpts[k1], kpts[k2]))
            assert kpti_kptj.shape == (2, 3)
            with _load3c(mydf._cderi, mydf._dataname, kpti_kptj) as j3c:
                nauxlist[(k1,k2)] = j3c.shape[0]
    return nauxlist

def _do_ao2mo_disk(kcasscf, mo_kpts, nkpts, ncore, ncas, nmo, level=1):
    cell = kcasscf._scf.cell
    kpts = kcasscf._scf.kpts
    nkpts = kcasscf.nkpts
    mydf = kcasscf._scf.with_df
    nocc = ncore + ncas
    dtype = mo_kpts[0].dtype

    assert len(mo_kpts) == nkpts

    erifile = lib.H5TmpFile()
    erifile.require_group("ppaa") # Create the group for storing the ppaa integrals.
    erifile.require_group("papa") # Create the group for storing the papa integrals.

    log = lib.logger.Logger(kcasscf.stdout, kcasscf.verbose)
    # Steps:
    # 1. Loop over k1, k2 pairs and compute the ao2mo for these pairs, and save them on disk.
    

    mem_now = lib.current_memory()[0]
    # I am not sure wheather the naoaux will be same for all k1, k2 pairs. 
    # So I am taking the maximum naoaux among all pairs.
    # naoaux = mydf.get_naoaux()
    nauxlist = get_nauxlist(mydf, kpts, nkpts)
    naoaux = max(nauxlist.values())
    mem_required = naoaux * nmo * nmo * 16 / 1e6
    if mem_now < 2.0 * mem_required:
        raise MemoryError(f"Not enough memory for intermediate arrays for ao2mo transformation. \
                          Required: {mem_required} MB, Current: {mem_now} MB.")
    
    compact = False # For complex integrals: I am using compact as of now to avoid any conj/sign issues.
    fxpp = lib.H5TmpFile()
    grp = fxpp.require_group("xpp") # Create the group for storing the Lpq integrals.
    grp2 = fxpp.require_group("xpp_sign") # Create the group for storing sign of the Lpq integrals.

    # Step-1
    for k1 in range(nkpts):
        for k2 in range(nkpts):
            mo_coeffs = _mo_as_complex(mo_kpts[k1], mo_kpts[k2])
            nij_pair, moij, ijslice = _conc_mos([mo_coeffs[0], mo_coeffs[1]])[1:]
            kptij = np.vstack((kpts[k1], kpts[k2]))
            naux = nauxlist[(k1, k2)]
            for LpqR, LpqI, sign in mydf.sr_loop(kptij, mem_now, compact):
                tao = []
                ao_loc = None
                zij = LpqR + 1j * LpqI
                zij = _ao2mo.r_e2(LpqR, moij, ijslice, tao, ao_loc, out=zij)
                # TODO: Learn whether I should store the 2D or 3D array for better I/O performance.
                grp.create_dataset(f"{k1}_{k2}", data=zij.reshape(naux, nmo, nmo))
                grp2.create_dataset(f"{k1}_{k2}", data=sign)

    # TODO: use the blksize to loop over the nmo to reduce the memory footprint.           
    # Step-2: Construct the papa integrals:
    kconserv = kpts_helper.get_kconserv(cell, kpts)
    # for k1 in range(nkpts):
    #     for k2 in range(nkpts):
    #         for k3 in range(nkpts):
    for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
        k4 = kconserv[k1, k2, k3]
        papa = np.zeros((nmo*ncas, nmo*ncas), dtype=dtype)
        # Read the Lpq integrals for the required pairs from disk.
        zij_12 = grp[f"{k1}_{k2}"][:, :ncas][:] # Lpq
        zkl_34 = grp[f"{k3}_{k4}"][:, :ncas][:] # Lkl
        # Reshape 
        zij_12 = zij_12.reshape(-1, nmo*ncas)
        zkl_34 = zkl_34.reshape(-1, nmo*ncas)
        sign = grp2[f"{k1}_{k2}"][:]
        lib.dot(zij_12.T, zkl_34, sign, papa, 1)
        erifile[f"papa/{k1}_{k2}_{k3}"] = papa.reshape(nmo, ncas, nmo, ncas)

    papa = None
    
    # Step-3: Construct the ppaa integrals:
    for k1, k2, k3 in kpts_helper.loop_kkk(nkpts):
        k4 = kconserv[k1, k2, k3]
        ppaa = np.zeros((nmo*nmo, ncas*ncas), dtype=dtype)
        zij_12 = grp[f"{k1}_{k2}"][:]
        zkl_34 = grp[f"{k3}_{k4}"][:ncas, :ncas][:]
        zij_12 = zij_12.reshape(-1, nmo*nmo)
        zkl_34 = zkl_34.reshape(-1, ncas*ncas)
        sign = grp2[f"{k1}_{k2}"][:]
        lib.dot(zij_12.T, zkl_34, sign, ppaa, 1)
        erifile[f"ppaa/{k1}_{k2}_{k3}"] = ppaa.reshape(nmo, nmo, ncas, ncas)

    ppaa = None

    log.debug1('Initializing disk for Lpq integrals.')

    if level == 1:
        j_pc = np.empty((nkpts, nmo, ncore), dtype=mo_kpts[0].dtype)
        k_pc = np.empty((nkpts, nmo, ncore), dtype=mo_kpts[0].dtype)
        cell = kcasscf._scf.cell
        kpts = kcasscf._scf.kpts
    else:
        j_pc = None
        k_pc = None

    fxpp.close()

    return erifile, j_pc, k_pc

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
        self.ppaa_kpts = None
        self.papa_kpts = None
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
            self.ppaa_kpts, self.papa_kpts, self.j_pc, self.k_pc = _do_ao2mo_direct(kcasscf, mo_kpts, nkpts, ncore, ncas, nmo, level=level)
        else:
            log.debug('Using disk ERI transformation.')
            self.erifile, self.j_pc, self.k_pc = _do_ao2mo_disk(kcasscf, mo_kpts, nkpts, ncore, ncas, nmo, level=level)
            
        # To access the ppaa and papa integrals: I am bifurcating the code based on whether 
        # we are using disk or direct method. If we are using direct method, 
        # we can directly access the ppaa and papa integrals from the attributes.
        # If we are using disk method, we need to read the integrals from the disk. 
        # To avoid writing separate code for accessing the integrals in different methods, 
        # I am defining two lambda functions that will access the integrals based on the method used.
        self.ppaa = lambda k1, k2, k3: self.get_ppaa(k1, k2, k3)
        self.papa = lambda k1, k2, k3: self.get_papa(k1, k2, k3)

    @staticmethod
    def _kkey(k1, k2, k3):
        return f"{int(k1)}_{int(k2)}_{int(k3)}"

    def _get(self, eriname, k1, k2, k3):
        # General wrapper
        arr = getattr(self, eriname + "_kpts", None)
        if arr is not None:return arr[k1, k2, k3]
        assert self.erifile is not None
        data = self.erifile[f"{eriname}/{self._kkey(k1, k2, k3)}"]
        return data[()]

    def get_ppaa(self, k1, k2, k3):
        return self._get("ppaa", k1, k2, k3)

    def get_papa(self, k1, k2, k3):
        return self._get("papa", k1, k2, k3)