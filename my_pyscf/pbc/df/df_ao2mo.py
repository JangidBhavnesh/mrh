import numpy as np

from pyscf import lib
from pyscf.pbc.df.df_ao2mo import ao2mo_7d, gamma_point, unique
from pyscf.pbc.lib import kpts_helper
from pyscf.ao2mo import _ao2mo
from pyscf.ao2mo.incore import iden_coeffs, _conc_mos

# Author: Bhavnesh Jangid

def ao2mo_7d(mydf, mo_coeff_kpts, kpts=None, factor=1, out=None):
    '''
    Optimized function to perform the AO2MO transformation.
    Block AO -> Block MO basis. This is optimized version of the one in
    the pyscf.pbc.df.df_ao2mo module.
    '''
    cell = mydf.cell
    if kpts is None:
        kpts = mydf.kpts
    nkpts = len(kpts)

    if isinstance(mo_coeff_kpts, np.ndarray) and mo_coeff_kpts.ndim == 3:
        mo_coeff_kpts = [mo_coeff_kpts] * 4
    else:
        mo_coeff_kpts = list(mo_coeff_kpts)

    nmoi, nmoj, nmok, nmol = [x.shape[2] for x in mo_coeff_kpts]
    eri_shape = (nkpts, nkpts, nkpts, nmoi, nmoj, nmok, nmol)

    if gamma_point(kpts):
        dtype = np.result_type(*mo_coeff_kpts)
    else:
        dtype = np.complex128

    if out is None:
        out = np.empty(eri_shape, dtype=dtype)
    else:
        assert out.shape == eri_shape

    kptij_lst = np.array([(ki, kj) for ki in kpts for kj in kpts])
    kptis_lst = kptij_lst[:, 0]
    kptjs_lst = kptij_lst[:, 1]
    kpt_ji = kptjs_lst - kptis_lst

    uniq_kpts, uniq_index, uniq_inverse = unique(kpt_ji)

    nao = cell.nao_nr()
    max_memory = max(2000, mydf.max_memory - lib.current_memory()[0] - nao**4 * 16 / 1e6 ) * 0.5

    tao = []
    ao_loc = None
    kconserv = kpts_helper.get_kconserv(cell, kpts)

    # Create a temporary file to store the intermediate zkl arrays for 
    # each unique q = kj - ki.
    # This will save the expensive zkl computation for each (ki, kj) pair in the same q group.
    feri = lib.H5TmpFile()
    for uniq_id, kpt in enumerate(uniq_kpts):
        adapted_ji_idx = np.where(uniq_inverse == uniq_id)[0]

        qgrp_name = f"q{uniq_id}"
        qgrp = feri.create_group(qgrp_name)

        ji0 = adapted_ji_idx[0]
        ki0 = ji0 // nkpts
        kj0 = ji0 % nkpts

        kl_for_kk = []
        nblk_for_kk = []

        for kk in range(nkpts):
            kl = kconserv[ki0, kj0, kk]
            kl_for_kk.append(kl)
            mokl, klslice = _conc_mos( mo_coeff_kpts[2][kk],mo_coeff_kpts[3][kl] )[2:]
            kkgrp = qgrp.create_group(f"kk{kk}")

            signs = []
            iblk = 0

            for LrsR, LrsI, sign in mydf.sr_loop(
                    kpts[[kk, kl]], max_memory, False, mydf.blockdim):
                zkl = _ao2mo.r_e2( LrsR + LrsI * 1j, mokl, klslice, tao, ao_loc )
                kkgrp.create_dataset(f"z{iblk}", data=zkl)
                signs.append(sign)
                iblk += 1

            kkgrp.create_dataset("signs", data=np.asarray(signs))
            nblk_for_kk.append(iblk)

        feri.flush()

        for ji, ji_idx in enumerate(adapted_ji_idx):
            ki = ji_idx // nkpts
            kj = ji_idx % nkpts

            moij, ijslice = _conc_mos( mo_coeff_kpts[0][ki], mo_coeff_kpts[1][kj] )[2:]

            zij = []
            for LpqR, LpqI, sign in mydf.sr_loop(kpts[[ki, kj]], 
                                                    max_memory, False, mydf.blockdim):
                zij.append( _ao2mo.r_e2( LpqR + LpqI * 1j, moij, ijslice, tao, ao_loc ) )

            for kk in range(nkpts):
                kl = kl_for_kk[kk]
                # The cached kl mapping should be valid for all adapted
                # (ki, kj) pairs in the same q = kj - ki group.
                assert kl == kconserv[ki, kj, kk]
                eri_mo = np.zeros( (nmoi * nmoj, nmok * nmol), dtype=np.complex128 )

                kkgrp = qgrp[f"kk{kk}"]
                signs = kkgrp["signs"][()]
                nblk = nblk_for_kk[kk]

                for iblk in range(nblk): 
                    zkl = kkgrp[f"z{iblk}"][()] 
                    lib.dot( zij[iblk].T, zkl, signs[iblk] * factor, eri_mo, 1 )
                if dtype == np.double:
                    eri_mo = eri_mo.real

                out[ki, kj, kk] = eri_mo.reshape(eri_shape[3:])

        del feri[qgrp_name]

    feri.close()

    return out

if __name__ == "__main__":
    # This has to go to the unit tests.
    from pyscf.pbc import gto, scf

    def timer_start():
        return lib.logger.process_clock(), lib.logger.perf_counter()

    def timer_stop(label, t0):
        cpu0, wall0 = t0
        cpu = lib.logger.process_clock() - cpu0
        wall = lib.logger.perf_counter() - wall0
        print(f"{label:35s} CPU = {cpu:10.2f} s   Wall = {wall:10.2f} s")
        return cpu, wall

    cell = gto.Cell()
    cell.atom = """
    C  0.000000  0.000000  0.000000
    C  0.891700  0.891700  0.891700
    """
    cell.a = """
    0.000000  1.783400  1.783400
    1.783400  0.000000  1.783400
    1.783400  1.783400  0.000000
    """
    cell.basis = "gth-szv"
    cell.pseudo = "gth-pade"
    cell.unit = "B"
    cell.verbose = 4
    cell.build()

    # Use a small 1D kmesh first. Increase to [6,1,1] or [8,1,1]
    # if you want a clearer caching advantage.
    kmesh = [4, 4, 4]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)

    # ------------------------------------------------------------
    # Mean-field calculation.
    # ------------------------------------------------------------
    kmf = scf.KRHF(cell, kpts, exxdiv=None).density_fit()
    kmf.with_df._cderi_to_save = "cderi.diamond.h5"
    # kmf.with_df._cderi = "cderi.diamond.h5"
    kmf.conv_tol = 1e-1

    t0 = timer_start()
    kmf.kernel()
    timer_stop("KRHF mean field", t0)
    
    mo_coeff_kpts = np.asarray([kmf.mo_coeff[k][:, :4] for k in range(nkpts)])

    print("mo_coeff_kpts shape =", mo_coeff_kpts.shape)
    print("nkpts =", nkpts)

    # ------------------------------------------------------------
    # Original PySCF ao2mo_7d.
    # ------------------------------------------------------------
    import time
    t = time.perf_counter()
    t0 = timer_start()
    eri_ref = kmf.with_df.ao2mo_7d(mo_coeff_kpts, kpts=kpts)
    timer_stop("Original PySCF ao2mo_7d", t0)
    print("Original PySCF ao2mo_7d wall time =", time.perf_counter() - t, "s")
    # ------------------------------------------------------------
    # Optimized cached ao2mo_7d defined above.
    # ------------------------------------------------------------

    t = time.perf_counter()
    t0 = timer_start()
    eri_new = ao2mo_7d(kmf.with_df, mo_coeff_kpts, kpts=kpts)
    timer_stop("Cached zkl ao2mo_7d", t0)
    print("Cached zkl ao2mo_7d wall time =", time.perf_counter() - t, "s")

    # ------------------------------------------------------------
    # Check correctness.
    # ------------------------------------------------------------
    diff = np.max(np.abs(eri_ref - eri_new))
    norm = np.max(np.abs(eri_ref))
    rel = diff / norm if norm > 0 else diff

    print()
    print("Correctness check:")
    print("  max abs diff =", diff)
    print("  relative diff =", rel)

    assert np.allclose(eri_ref, eri_new, atol=1e-9, rtol=1e-9)
    print("AO2MO cached implementation agrees with original PySCF ao2mo_7d.")