
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <omp.h>
#include <complex.h>
#include "config.h"
#include "np_helper/np_helper.h"
#include "fci.h"

// Author: Bhavnesh Jangid

/*
For the implementation of the k-FCI, I would need to update the construction of link_index to include
the momentum information. The link_index layout would be:
link_index[str_id, link_id, :] =
    [cre, des, target_address, parity, k0, k_cre, k_des, dK]
    Here:
    k0 is the total momentum of the starting string,
    k_cre and k_des are the momentum labels of the creation and annihilation orbitals, respectively
    dk = (k_cre - k_des) mod nkpts is the momentum transfer.
Therefore, the shape of the link_index would change from (na, nlink, 4) to (na, nlink, 8).
    na = number of strings
    nlink = number of links per string = nocc * nvir + nocc
    norb = number of orbitals
    nocc = number of occupied orbitals
    nvir = number of virtual orbitals = norb - nocc
*/


/*
This function returns x modulo n as a non-negative integer in the range [0, n-1].
Required for wrapping k-point/momentum differences such as k_cre - k_des.
*/
static inline int mod_pos(int x, int n)
{
    int r = x % n;
    return (r < 0) ? r + n : r;
}

void FCIstrs2addr(int *addr, uint64_t *strs, int nstr, int norb, int nelec);

// The kFCI Hamiltonian would be hermitian, but the imaginary part would be anti-symmetric, hence
// I can not use the tril index. (!) Additionally, we skipped the triangular packed index because k-FCI 
// needs the ordered creation/destruction labels (cre, des) to compute k_cre, k_des, and dK. 
// Therefore, I have skipped the construction of link_index with the tril symmetry.

// [cre, des, target_address, parity, k0, k_cre, k_des, dK]
void FCIlinkstr_index_k(int *link_index, int norb, int na, int nocc, 
                        uint64_t *strs, int *orb_k, int nkpts)
{
    int occ[norb];
    int vir[norb];
    int nvir = norb - nocc;
    int nlink = nocc * nvir + nocc;
    int str_id, io, iv;
    int i, a, k;
    int cre, des;
    int tempk;
    int k_cre, k_des, dK;
    int K0;
    uint64_t str0, str1;
    uint64_t str1s[nocc * nvir];
    int addrbuf[nocc * nvir];
    int *tab;

    for (str_id = 0; str_id < na; str_id++) {
        str0 = strs[str_id];
        /*
         * First building the occupied and virtual orbital lists,
         * and then computing the total momentum K0 of the spin string.
         */
        K0 = 0;
        io = 0;
        iv = 0;
        for (i = 0; i < norb; i++) {
            if (str0 & (1ULL << i)) {
                occ[io] = i;
                io += 1;
                K0 = mod_pos(K0 + orb_k[i], nkpts);
            } else {
                vir[iv] = i;
                iv += 1;
            }
        }

        tab = link_index + str_id * nlink * 8;

        /*
         * Step-1: Diagonal links for the identity operation (no excitation):
         * a_i^\dagger a_i |D> = |D>
         */
        for (k = 0; k < nocc; k++) {
            cre = occ[k];
            des = occ[k];
            tempk = k * 8;
            k_cre = orb_k[cre];
            k_des = orb_k[des];
            dK = mod_pos(k_cre - k_des, nkpts);

            tab[tempk + 0] = cre;
            tab[tempk + 1] = des;
            tab[tempk + 2] = str_id;
            tab[tempk + 3] = 1;
            tab[tempk + 4] = K0;
            tab[tempk + 5] = k_cre;
            tab[tempk + 6] = k_des;
            tab[tempk + 7] = dK;
        }

        /*
         * Step-2: Single-excitation links:
         * a_a^\dagger a_i |D0> = parity |D1>
         */
        
        k = nocc;
        for (i = 0; i < nocc; i++) {
            des = occ[i];
            for (a = 0; a < nvir; a++, k++) {
                cre = vir[a];
                tempk = k * 8;
                str1 = (str0 ^ (1ULL << des)) | (1ULL << cre);
                str1s[k - nocc] = str1;
                k_cre = orb_k[cre];
                k_des = orb_k[des];
                dK = mod_pos(k_cre - k_des, nkpts);

                tab[tempk + 0] = cre;
                tab[tempk + 1] = des;
                tab[tempk + 2] = -1; // to be filled down with the target address of str1
                tab[tempk + 3] = FCIcre_des_sign(cre, des, str0);
                tab[tempk + 4] = K0;
                tab[tempk + 5] = k_cre;
                tab[tempk + 6] = k_des;
                tab[tempk + 7] = dK;
            }
        }

        /*
         * Fill target addresses for the single excitations.
         */
        FCIstrs2addr(addrbuf, str1s, nocc * nvir, norb, nocc);

        for (k = 0; k < nocc * nvir; k++) {
            tab[(k + nocc) * 8 + 2] = addrbuf[k];
        }
    }
}
