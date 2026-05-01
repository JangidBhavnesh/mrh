import numpy as np

from pyscf.fci.addons import _unpack_nelec
from mrh.my_pyscf.pbc.fci import rdm_helper, kcistrings


from mrh.my_pyscf.pbc.fci.kcistrings import gen_k_sector_linkstr_info, gen_k_sector_maps

# Author: Bhavnesh Jangid

'''
Implementation of k-FCI.
'''

def _unpack(norb, nelec, link_index, nkpts, spin=None):
    assert norb % nkpts == 0
    if link_index is None:
        neleca, nelecb = _unpack_nelec(nelec, spin)
        norb_k = norb // nkpts
        orb_k = (np.arange(norb, dtype=np.int32) // norb_k).astype(np.int32)
        if spin == 0 and neleca == nelecb:
            link_indexa = link_indexb = kcistrings.gen_linkstr_index_k(range(norb), neleca, orb_k, nkpts)
        else:
            link_indexa = kcistrings.gen_linkstr_index_k(range(norb), neleca, orb_k, nkpts)
            link_indexb = kcistrings.gen_linkstr_index_k(range(norb), nelecb, orb_k, nkpts)
        return link_indexa, link_indexb
    else:
        assert link_index[0].shape[2] == link_index[1].shape[2] == 8
        return link_index


def contract_1e_k(h1e, fcivec, norb, nelec, nkpts, kindx, link_index=None):
    '''
    Contract one-electron Hamiltonian with a k-FCI vector in a fixed 
    total momentum sector.
    args:
        h1e : ndarray, shape (nkpts, norb_k, norb_k)
            One-electron integrals in k-space, where norb_k = norb // nkpts.
        fcivec : ndarray, shape (sector_size,)
            k-FCI vector in the target total momentum sector.
        norb : int
            Total number of orbitals.
        nelec : tuple of 2 ints
            Number of alpha and beta electrons.
        nkpts : int
            Number of k-points / momentum sectors.
        kindx : int
            Target total momentum sector. (0<=kindx < nkpts)
        link_index : tuple of 2 ndarrays or None
            Look up tables/link index for alpha and beta strings. 
            If None, it will be generated on the fly.
        Note: these are k-aware link indices, and the link columns are:
            [cre, des, target_address, parity, k0, k_cre, k_des, dK].
            and overall shape is (nstr, nlink, 8) for each spin sector.
    returns:
        sigma_ci : ndarray, shape (sector_size,)
            Result of the Hamiltonian-vector product in the target momentum sector.
    '''

    link_indexa, link_indexb = _unpack(norb, nelec, link_index, nkpts)
    dtype = np.result_type(h1e, fcivec)

    # Sanity checks
    assert link_indexa.ndim == link_indexb.ndim == 3
    assert link_indexa.shape[2] == link_indexb.shape[2] == 8
    assert h1e.ndim == 3
    ncas = norb // nkpts
    assert h1e.shape == (nkpts, ncas, ncas)

    kindx = int(kindx) % nkpts

    # Generate the k-sector blocks and the corresponding alpha/beta string
    # lists and global-to-local (specific k-sector) maps.
    # rows are [ka, kb, na, nb, offset, size]
    blocks = gen_k_sector_linkstr_info(link_indexa, link_indexb, nkpts, kindx)
    sector_size = int(blocks[:, 5].sum())

    assert fcivec.size == sector_size

    straid_k, strbid_k, tota_2k, totb_2k = gen_k_sector_maps(link_indexa, link_indexb, nkpts)

    # Making sure fcivec is in the right dtype and C-contiguous.
    h1e = np.asarray(h1e, dtype=dtype, order="C")
    fcivec = np.asarray(fcivec, dtype=dtype, order="C")
    sigma_ci = np.zeros(fcivec.shape, dtype=dtype, order="C")

    # link columns: [cre, des, target_address, parity, k0, k_cre, k_des, dK]
    CRE = 0
    DES = 1
    TARGET = 2
    SIGN = 3
    K_CRE = 5
    K_DES = 6
    DK = 7

    for ka, kb, na, nb, offset, size in blocks:
        Cblk = fcivec[offset:offset + size].reshape(na, nb)
        Sblk = sigma_ci[offset:offset + size].reshape(na, nb)

        alpha_ids = straid_k[ka]
        beta_ids = strbid_k[kb]

        # h1e contraction for the alpha strings.
        for ia0_local, astr0 in enumerate(alpha_ids):
            astr0 = int(astr0)
            for link in link_indexa[astr0]:
                p = int(link[CRE])
                q = int(link[DES])
                astr1 = int(link[TARGET])
                sign = link[SIGN]

                k_cre = int(link[K_CRE]) % nkpts
                k_des = int(link[K_DES]) % nkpts
                dK = int(link[DK]) % nkpts

                # h1e[k, p, q] is k-diagonal, so only k_cre == k_des contributes.
                # which means only dk=0 contributes.
                if (k_cre != k_des) or (dK != 0): continue

                # Note that p and q are in the global orbital indexing, 
                # but h1e is in the k-space orbital indexing, so we need 
                # to mod by ncas to get the correct orbital indices for h1e.
                hpq = h1e[k_cre, p % ncas, q % ncas]

                # Check if the excitation is out of the momentum sector then skip this.
                ia1_local = tota_2k[ka, astr1]
                if ia1_local < 0: continue
                Sblk[ia1_local, :] += sign * hpq * Cblk[ia0_local, :]

        # h1e contraction for the beta strings.
        for ib0_local, bstr0 in enumerate(beta_ids):
            bstr0 = int(bstr0)
            for link in link_indexb[bstr0]:
                p = int(link[CRE])
                q = int(link[DES])
                bstr1 = int(link[TARGET])
                sign = link[SIGN]
                k_cre = int(link[K_CRE]) % nkpts
                k_des = int(link[K_DES]) % nkpts
                dK = int(link[DK]) % nkpts
                # h1e[k, p, q] is k-diagonal, so only k_cre == k_des contributes.
                # which means only dk=0 contributes.
                if (k_cre != k_des) or (dK != 0): continue

                hpq = h1e[k_cre, p % ncas, q % ncas]
                
                # Check if the excitation is out of the momentum sector then skip this.
                ib1_local = totb_2k[kb, bstr1]
                if ib1_local < 0:
                    continue

                Sblk[:, ib1_local] += sign * hpq * Cblk[:, ib0_local]

    return sigma_ci


if __name__ == '__main__':
    
    TEST1 = False
    TEST2 = True
    TEST3 = False

    if TEST1:
        ncastot = 8
        nelectot = (4, 4)
        nkpts = 2 #(3, 3, 1)
        link_indexa, link_indexb = _unpack(ncastot, nelectot, None, nkpts, spin=None)
        print("link_indexa shape:", link_indexa.shape)
        print("link_indexb shape:", link_indexb.shape)
        print("----")

        # Possible k0 sectors:
        print("Possible k0 sectors: alpha str", np.unique(link_indexa[:, :, 4]))
        print("Possible k0 sectors: beta str", np.unique(link_indexb[:, :, 4])) 
        print("----")
        # Compare the alpha and beta string counts per k0 sector. 
        # They should be the same for spin-0 cases.
        det_count = kcistrings._count_det_per_k((link_indexa, link_indexb))
        print("Determinant count alpha:", det_count[0])
        print("Determinant count  beta:", det_count[1])
        print("----")
        # Now for det_alpha * det_beta, the total number of determinants 
        # would be k0 = (Ka + Kb) % nkpts. 
        # Let's count how many determinants we have in each k0 sector.
        counts_det = {k: 0 for k in range(nkpts)}
        for Ka, Na in det_count[0].items():
            for Kb, Nb in det_count[1].items():
                Kdet = (Ka + Kb) % nkpts
                if Kdet == 0:
                    print(f"Ka={Ka}, Kb={Kb}, Na={Na}, Nb={Nb}, Kdet={Kdet}")
                counts_det[Kdet] += Na * Nb
        
        print(counts_det)
        print("----")
        # When I will solve the kFCI problem, I will be solving it for the one of
        # the k0 sectors. Overall, the total number of determinants would be appro.
        # ntot_det / nkpts. Which is reduction in total number of determinants, but 
        # this will be huge headache to workout the proper vectorization.
        
        # Anyways:
        # Now compare with the total number of determinants from cistring.
        from pyscf.fci import cistring
        na, nb = _unpack_nelec(nelectot, spin=None)
        strsa = cistring.gen_linkstr_index(range(ncastot), na)
        strsb = cistring.gen_linkstr_index(range(ncastot), nb)
        tot_det = len(strsa) * len(strsb)

        print("Total determinants from cistring:", tot_det)
        print("Total determinants from link_index:", sum(counts_det.values()))
        print("Comparison at one of k-pts:", tot_det/list(counts_det.values())[0]) # Almost an order of magnitude.
        print("----")
    

    if TEST2:
        rng = np.random.default_rng(12)

        nkpts = 4
        ncas = 4
        norb = nkpts * ncas

        nelec = (2, 2)
        kindx = 0

        h1e = (
            rng.normal(size=(nkpts, ncas, ncas))
            + 1j * rng.normal(size=(nkpts, ncas, ncas))
        )

        # Optional: make each h1e[k] Hermitian
        for k in range(nkpts):
            h1e[k] = 0.5 * (h1e[k] + h1e[k].conj().T)

        link_index = None
        link_indexa, link_indexb = _unpack(norb, nelec, link_index, nkpts)

        for kindx in range(nkpts):
            blocks = gen_k_sector_linkstr_info(link_indexa, link_indexb, nkpts, kindx)
            sector_size = int(blocks[:, 5].sum())

            fcivec = (
                rng.normal(size=sector_size)
                + 1j * rng.normal(size=sector_size)
            )
            fcivec /= np.linalg.norm(fcivec)

            sigma_ci = contract_1e_k(
                h1e,
                fcivec,
                norb,
                nelec,
                nkpts,
                kindx,
                link_index=None
            )

            print("blocks:")
            print(blocks)
            print("sector_size =", sector_size)
            print("fcivec shape   =", fcivec.shape)
            print("sigma_ci shape =", sigma_ci.shape)
            print("||fcivec||     =", np.linalg.norm(fcivec))
            print("||sigma_ci||   =", np.linalg.norm(sigma_ci))
            # print("sigma_ci[:10]  =", sigma_ci[:10])


