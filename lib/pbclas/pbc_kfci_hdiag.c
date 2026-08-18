#include <complex.h>
#include <omp.h>
#include <stdlib.h>
#include "pbc_kfci_common.h"
#include "vhf/fblas.h"

/*
Author: Bhavnesh Jangid
*/

/*
 * Hamiltonian Diagonal for kFCI.
 *
 * The Python layer builds the momentum-sector determinant blocks and compact
 * contraction structures.  This file only evaluates the diagonal entries from
 * those structures.
 */

/* 
 * Memory management:
 * - Caller-provided integral, diagonal, link, block, and contraction-map arrays
 *   are never freed here.
 * - pbc_kfci_make_block_tables transfers block_offset, block_na, and block_nb
 *   to its caller on success and frees them locally on failure.
 * - FCIhdiag_k frees those block tables before returning.
 * - FCIhdiag_k_stream_ab allocates wmat and per-block amat, bmat, and tmp work
 *   arrays, and frees each array before returning or taking a fallback path.
*/

/** Add diagonal one-electron contributions for all CI blocks. */
static void add_one_electron_hdiag(double complex *hdiag,
                                   double complex *h1e,
                                   int nkpts, int ncas,
                                   int nblocks, int *blocks,
                                   int *linka, int nlinka,
                                   int *linkb, int nlinkb,
                                   int *stra_ids, int *stra_offsets,
                                   int *strb_ids, int *strb_offsets,
                                   int dk_zero)
{
#pragma omp parallel for schedule(static) default(none) if(nblocks > 16) \
        shared(hdiag, h1e, nkpts, ncas, nblocks, blocks, \
               linka, nlinka, linkb, nlinkb, \
               stra_ids, stra_offsets, strb_ids, strb_offsets, dk_zero)
        for (int iblk = 0; iblk < nblocks; iblk++) {
                int *blk = blocks + iblk * 6;
                int ka = blk[BLOCK_KA];
                int kb = blk[BLOCK_KB];
                int na = blk[BLOCK_NA];
                int nb = blk[BLOCK_NB];
                int offset = blk[BLOCK_OFFSET];

                for (int ia = 0; ia < na; ia++) {
                        int astr0 = stra_ids[stra_offsets[ka] + ia];
                        double complex val = 0.0 + 0.0 * I;

                        for (int ilink = 0; ilink < nlinka; ilink++) {
                                int *link = linka + (astr0 * nlinka + ilink)
                                        * NLINK_FIELDS;
                                int sign = link[LINK_SIGN];
                                if (sign == 0) {
                                        break;
                                }
                                if (link[LINK_TARGET] != astr0) {
                                        continue;
                                }

                                int k_cre = link[LINK_K_CRE] % nkpts;
                                int k_des = link[LINK_K_DES] % nkpts;
                                int dk = link[LINK_DK] % nkpts;
                                if (k_cre != k_des || dk != dk_zero) {
                                        continue;
                                }

                                int p = link[LINK_CRE] % ncas;
                                int q = link[LINK_DES] % ncas;
                                val += (double)sign *
                                        h1e[(k_cre * ncas + p) * ncas + q];
                        }

                        for (int ib = 0; ib < nb; ib++) {
                                hdiag[offset + ia * nb + ib] += val;
                        }
                }

                for (int ib = 0; ib < nb; ib++) {
                        int bstr0 = strb_ids[strb_offsets[kb] + ib];
                        double complex val = 0.0 + 0.0 * I;

                        for (int ilink = 0; ilink < nlinkb; ilink++) {
                                int *link = linkb + (bstr0 * nlinkb + ilink)
                                        * NLINK_FIELDS;
                                int sign = link[LINK_SIGN];
                                if (sign == 0) {
                                        break;
                                }
                                if (link[LINK_TARGET] != bstr0) {
                                        continue;
                                }

                                int k_cre = link[LINK_K_CRE] % nkpts;
                                int k_des = link[LINK_K_DES] % nkpts;
                                int dk = link[LINK_DK] % nkpts;
                                if (k_cre != k_des || dk != dk_zero) {
                                        continue;
                                }

                                int p = link[LINK_CRE] % ncas;
                                int q = link[LINK_DES] % ncas;
                                val += (double)sign *
                                        h1e[(k_cre * ncas + p) * ncas + q];
                        }

                        for (int ia = 0; ia < na; ia++) {
                                hdiag[offset + ia * nb + ib] += val;
                        }
                }
        }
}

/** Add mapped alpha-beta two-electron diagonal contributions. */
static void add_ab_hdiag(double complex *hdiag,
                         double complex *eri,
                         int nkpts,
                         int *block_offset,
                         int *ab_group_tab,
                         int *ab_group_offsets,
                         int *ab_src_addr,
                         int *ab_dst_addr,
                         int *ab_sign,
                         long long *ab_eri_idx_ab,
                         long long *ab_eri_idx_ba)
{
        int table_size = nkpts * nkpts;

#pragma omp parallel for schedule(static) default(none) if(table_size > 16) \
        shared(hdiag, eri, table_size, block_offset, ab_group_tab, \
               ab_group_offsets, ab_src_addr, ab_dst_addr, ab_sign, \
               ab_eri_idx_ab, ab_eri_idx_ba)
        for (int src_key = 0; src_key < table_size; src_key++) {
                int src_offset = block_offset[src_key];
                if (src_offset < 0) {
                        continue;
                }

                int group0 = ab_group_offsets[src_key];
                int group1 = ab_group_offsets[src_key + 1];
                for (int ig = group0; ig < group1; ig++) {
                        int *group = ab_group_tab + ig * 3;
                        int dst_offset = group[0];
                        if (dst_offset != src_offset) {
                                continue;
                        }

                        for (int i = group[1]; i < group[2]; i++) {
                                int addr = ab_src_addr[i];
                                if (ab_dst_addr[i] != addr) {
                                        continue;
                                }
                                hdiag[src_offset + addr] +=
                                        (double)ab_sign[i] *
                                        (eri[ab_eri_idx_ab[i]] +
                                         eri[ab_eri_idx_ba[i]]);
                        }
                }
        }
}

/** Add mapped alpha-alpha or beta-beta diagonal contributions. */
static void add_same_spin_hdiag(double complex *hdiag,
                                double complex *eri,
                                int nkpts,
                                int spin_alpha,
                                int *block_offset,
                                int *block_na,
                                int *block_nb,
                                int *group_tab,
                                int *group_offsets,
                                int *src_addr,
                                int *dst_addr,
                                int *sign,
                                long long *eri_idx,
                                int dk_zero)
{
        int table_size = nkpts * nkpts;

#pragma omp parallel for schedule(static) default(none) if(table_size > 16) \
        shared(hdiag, eri, table_size, spin_alpha, block_offset, \
               block_na, block_nb, group_tab, group_offsets, \
               src_addr, dst_addr, sign, eri_idx)
        for (int src_key = 0; src_key < table_size; src_key++) {
                int src_offset = block_offset[src_key];
                if (src_offset < 0) {
                        continue;
                }

                int na = block_na[src_key];
                int nb = block_nb[src_key];
                int group0 = group_offsets[src_key];
                int group1 = group_offsets[src_key + 1];

                for (int ig = group0; ig < group1; ig++) {
                        int *group = group_tab + ig * 4;
                        int dst_offset = group[0];
                        if (dst_offset != src_offset) {
                                continue;
                        }

                        for (int i = group[2]; i < group[3]; i++) {
                                int src = src_addr[i];
                                if (dst_addr[i] != src) {
                                        continue;
                                }

                                double complex val =
                                        (double)sign[i] * eri[eri_idx[i]];
                                if (spin_alpha) {
                                        int ia = src;
                                        for (int ib = 0; ib < nb; ib++) {
                                                hdiag[src_offset + ia * nb +
                                                      ib] += val;
                                        }
                                } else {
                                        int ib = src;
                                        for (int ia = 0; ia < na; ia++) {
                                                hdiag[src_offset + ia * nb +
                                                      ib] += val;
                                        }
                                }
                        }
                }
        }
}

/** Add alpha-beta diagonal terms for one block without work arrays. */
static void add_ab_hdiag_scalar_block(double complex *hdiag,
                                      double complex *eri,
                                      int nkpts, int ncas,
                                      int *blocks,
                                      int iblk,
                                      int *linka, int nlinka,
                                      int *linkb, int nlinkb,
                                      int *stra_ids,
                                      int *stra_offsets,
                                      int *strb_ids,
                                      int *strb_offsets,
                                      int dk_zero)
{
        int *blk = blocks + iblk * 6;
        int ka = blk[BLOCK_KA];
        int kb = blk[BLOCK_KB];
        int na = blk[BLOCK_NA];
        int nb = blk[BLOCK_NB];
        int offset = blk[BLOCK_OFFSET];

        for (int ia = 0; ia < na; ia++) {
                int astr0 = stra_ids[stra_offsets[ka] + ia];

                for (int ib = 0; ib < nb; ib++) {
                        int bstr0 = strb_ids[strb_offsets[kb] + ib];
                        double complex val = 0.0 + 0.0 * I;

                        for (int ilinka = 0; ilinka < nlinka; ilinka++) {
                                int *la = linka + (astr0 * nlinka + ilinka) *
                                        NLINK_FIELDS;
                                int signa = la[LINK_SIGN];
                                if (signa == 0) {
                                        break;
                                }
                                if (la[LINK_TARGET] != astr0 ||
                                    mod_pos(la[LINK_DK], nkpts) != dk_zero ||
                                    mod_pos(la[LINK_K_CRE], nkpts) !=
                                    mod_pos(la[LINK_K_DES], nkpts)) {
                                        continue;
                                }

                                for (int ilinkb = 0; ilinkb < nlinkb;
                                     ilinkb++) {
                                        int *lb = linkb + (bstr0 * nlinkb +
                                                           ilinkb) *
                                                NLINK_FIELDS;
                                        int signb = lb[LINK_SIGN];
                                        long long idx_ab;
                                        long long idx_ba;

                                        if (signb == 0) {
                                                break;
                                        }
                                        if (lb[LINK_TARGET] != bstr0 ||
                                            mod_pos(lb[LINK_DK], nkpts) != dk_zero ||
                                            mod_pos(lb[LINK_K_CRE], nkpts) !=
                                            mod_pos(lb[LINK_K_DES], nkpts)) {
                                                continue;
                                        }

                                        idx_ab = eri_index_k(
                                                la[LINK_K_CRE],
                                                la[LINK_K_DES],
                                                lb[LINK_K_CRE],
                                                la[LINK_CRE] % ncas,
                                                la[LINK_DES] % ncas,
                                                lb[LINK_CRE] % ncas,
                                                lb[LINK_DES] % ncas,
                                                nkpts, ncas);
                                        idx_ba = eri_index_k(
                                                lb[LINK_K_CRE],
                                                lb[LINK_K_DES],
                                                la[LINK_K_CRE],
                                                lb[LINK_CRE] % ncas,
                                                lb[LINK_DES] % ncas,
                                                la[LINK_CRE] % ncas,
                                                la[LINK_DES] % ncas,
                                                nkpts, ncas);
                                        val += (double)(signa * signb) *
                                                (eri[idx_ab] + eri[idx_ba]);
                                }
                        }

                        hdiag[offset + ia * nb + ib] += val;
                }
        }
}

/**
 * Build the k-FCI Hamiltonian diagonal from precomputed contraction maps.
 * @param hdiag Output packed Hamiltonian diagonal.
 * @param h1e One-electron integrals arranged by k-point.
 * @param eri Two-electron integrals in k-point storage order.
 * @param nkpts Number of k-points.
 * @param ncas Number of active orbitals per k-point.
 * @param nblocks Number of packed CI blocks.
 * @param blocks Packed CI block records.
 * @param linka Alpha link table.
 * @param nlinka Number of links per alpha string.
 * @param linkb Beta link table.
 * @param nlinkb Number of links per beta string.
 * @param stra_ids Alpha string IDs grouped by momentum.
 * @param stra_offsets Offsets into stra_ids.
 * @param strb_ids Beta string IDs grouped by momentum.
 * @param strb_offsets Offsets into strb_ids.
 * @param dk_zero Momentum label for zero transfer.
 * @param ab_group_tab Alpha-beta contraction groups.
 * @param ab_group_offsets Alpha-beta group ranges by source block.
 * @param ab_src_addr Alpha-beta source addresses.
 * @param ab_dst_addr Alpha-beta destination addresses.
 * @param ab_sign Alpha-beta fermionic signs.
 * @param ab_eri_idx_ab Alpha-beta ERI indices in AB order.
 * @param ab_eri_idx_ba Alpha-beta ERI indices in BA order.
 * @param aa_group_tab Alpha-alpha contraction groups.
 * @param aa_group_offsets Alpha-alpha group ranges by source block.
 * @param aa_src_addr Alpha-alpha source addresses.
 * @param aa_dst_addr Alpha-alpha destination addresses.
 * @param aa_sign Alpha-alpha fermionic signs.
 * @param aa_eri_idx Alpha-alpha ERI indices.
 * @param bb_group_tab Beta-beta contraction groups.
 * @param bb_group_offsets Beta-beta group ranges by source block.
 * @param bb_src_addr Beta-beta source addresses.
 * @param bb_dst_addr Beta-beta destination addresses.
 * @param bb_sign Beta-beta fermionic signs.
 * @param bb_eri_idx Beta-beta ERI indices.
 * @return Nothing.
 */
void FCIhdiag_k(double complex *hdiag,
                double complex *h1e,
                double complex *eri,
                int nkpts, int ncas,
                int nblocks, int *blocks,
                int *linka, int nlinka,
                int *linkb, int nlinkb,
                int *stra_ids, int *stra_offsets,
                int *strb_ids, int *strb_offsets,
                int dk_zero,
                int *ab_group_tab,
                int *ab_group_offsets,
                int *ab_src_addr,
                int *ab_dst_addr,
                int *ab_sign,
                long long *ab_eri_idx_ab,
                long long *ab_eri_idx_ba,
                int *aa_group_tab,
                int *aa_group_offsets,
                int *aa_src_addr,
                int *aa_dst_addr,
                int *aa_sign,
                long long *aa_eri_idx,
                int *bb_group_tab,
                int *bb_group_offsets,
                int *bb_src_addr,
                int *bb_dst_addr,
                int *bb_sign,
                long long *bb_eri_idx)
{
        int ndet = 0;
        int *block_offset = NULL;
        int *block_na = NULL;
        int *block_nb = NULL;

        if (pbc_kfci_make_block_tables(nkpts, nblocks, blocks,
                              &block_offset, &block_na, &block_nb,
                              &ndet) != 0) {
                return;
        }

        pbc_kfci_zset0(hdiag, (size_t)ndet);
        if (ndet == 0) {
                free(block_offset);
                free(block_na);
                free(block_nb);
                return;
        }

        add_one_electron_hdiag(hdiag, h1e, nkpts, ncas, nblocks, blocks,
                               linka, nlinka, linkb, nlinkb,
                               stra_ids, stra_offsets,
                               strb_ids, strb_offsets, dk_zero);
        add_ab_hdiag(hdiag, eri, nkpts, block_offset,
                     ab_group_tab, ab_group_offsets,
                     ab_src_addr, ab_dst_addr, ab_sign,
                     ab_eri_idx_ab, ab_eri_idx_ba);
        add_same_spin_hdiag(hdiag, eri, nkpts, 1, block_offset,
                            block_na, block_nb,
                            aa_group_tab, aa_group_offsets,
                            aa_src_addr, aa_dst_addr, aa_sign, aa_eri_idx,
                            dk_zero);
        add_same_spin_hdiag(hdiag, eri, nkpts, 0, block_offset,
                            block_na, block_nb,
                            bb_group_tab, bb_group_offsets,
                            bb_src_addr, bb_dst_addr, bb_sign, bb_eri_idx,
                            dk_zero);

        free(block_offset);
        free(block_na);
        free(block_nb);
}

/**
 * Add streamed alpha-beta terms to a packed k-FCI Hamiltonian diagonal.
 * @param hdiag Hamiltonian diagonal updated in place.
 * @param eri Two-electron integrals in k-point storage order.
 * @param nkpts Number of k-points.
 * @param ncas Number of active orbitals per k-point.
 * @param nblocks Number of packed CI blocks.
 * @param blocks Packed CI block records.
 * @param linka Alpha link table.
 * @param nlinka Number of links per alpha string.
 * @param linkb Beta link table.
 * @param nlinkb Number of links per beta string.
 * @param stra_ids Alpha string IDs grouped by momentum.
 * @param stra_offsets Offsets into stra_ids.
 * @param strb_ids Beta string IDs grouped by momentum.
 * @param strb_offsets Offsets into strb_ids.
 * @param dk_zero Momentum label for zero transfer.
 * @return Nothing.
 */
void FCIhdiag_k_stream_ab(double complex *hdiag,
                          double complex *eri,
                          int nkpts, int ncas,
                          int nblocks, int *blocks,
                          int *linka, int nlinka,
                          int *linkb, int nlinkb,
                          int *stra_ids, int *stra_offsets,
                          int *strb_ids, int *strb_offsets,
                          int dk_zero)
{
        const char TRANS_N = 'N';
        const char TRANS_T = 'T';
        const double complex Z0 = 0.0 + 0.0 * I;
        const double complex Z1 = 1.0 + 0.0 * I;
        int norb = nkpts * ncas;
        double complex *wmat = malloc(sizeof(double complex) *
                                      (size_t)norb * norb);

        if (wmat == NULL) {
#pragma omp parallel for schedule(dynamic) default(none) \
        shared(hdiag, eri, nkpts, ncas, nblocks, blocks, linka, nlinka, \
               linkb, nlinkb, stra_ids, stra_offsets, strb_ids, strb_offsets, \
               dk_zero)
                for (int iblk = 0; iblk < nblocks; iblk++) {
                        add_ab_hdiag_scalar_block(
                                hdiag, eri, nkpts, ncas, blocks, iblk,
                                linka, nlinka, linkb, nlinkb,
                                stra_ids, stra_offsets,
                                strb_ids, strb_offsets, dk_zero);
                }
                return;
        }

        for (int gp = 0; gp < norb; gp++) {
                int kp = gp / ncas;
                int p = gp % ncas;
                for (int gb = 0; gb < norb; gb++) {
                        int kb = gb / ncas;
                        int b = gb % ncas;
                        long long idx_ab = eri_index_k(
                                kp, kp, kb, p, p, b, b, nkpts, ncas);
                        long long idx_ba = eri_index_k(
                                kb, kb, kp, b, b, p, p, nkpts, ncas);
                        wmat[gp + (size_t)gb * norb] =
                                eri[idx_ab] + eri[idx_ba];
                }
        }

#pragma omp parallel for schedule(dynamic) default(none) \
        shared(hdiag, eri, wmat, nkpts, ncas, norb, nblocks, blocks, linka, \
               nlinka, linkb, nlinkb, stra_ids, stra_offsets, strb_ids, \
               strb_offsets, TRANS_N, TRANS_T, Z0, Z1, dk_zero)
        for (int iblk = 0; iblk < nblocks; iblk++) {
                int *blk = blocks + iblk * 6;
                int ka = blk[BLOCK_KA];
                int kb = blk[BLOCK_KB];
                int na = blk[BLOCK_NA];
                int nb = blk[BLOCK_NB];
                int offset = blk[BLOCK_OFFSET];
                double complex *amat = malloc(sizeof(double complex) *
                                              (size_t)norb * na);
                double complex *bmat = malloc(sizeof(double complex) *
                                              (size_t)nb * norb);
                double complex *tmp = malloc(sizeof(double complex) *
                                             (size_t)norb * na);

                if (amat == NULL || bmat == NULL || tmp == NULL) {
                        free(tmp);
                        free(bmat);
                        free(amat);
                        add_ab_hdiag_scalar_block(
                                hdiag, eri, nkpts, ncas, blocks, iblk,
                                linka, nlinka, linkb, nlinkb,
                                stra_ids, stra_offsets,
                                strb_ids, strb_offsets, dk_zero);
                        continue;
                }

                pbc_kfci_zset0(amat, (size_t)norb * na);
                pbc_kfci_zset0(bmat, (size_t)nb * norb);

                for (int ia = 0; ia < na; ia++) {
                        int astr0 = stra_ids[stra_offsets[ka] + ia];
                        for (int ilink = 0; ilink < nlinka; ilink++) {
                                int *link = linka + (astr0 * nlinka + ilink)
                                        * NLINK_FIELDS;
                                int sign = link[LINK_SIGN];
                                int k;
                                int p;
                                int g;
                                if (sign == 0) {
                                        break;
                                }
                                if (link[LINK_TARGET] != astr0 ||
                                    mod_pos(link[LINK_DK], nkpts) != dk_zero ||
                                    mod_pos(link[LINK_K_CRE], nkpts) !=
                                    mod_pos(link[LINK_K_DES], nkpts)) {
                                        continue;
                                }
                                k = mod_pos(link[LINK_K_CRE], nkpts);
                                p = link[LINK_CRE] % ncas;
                                g = k * ncas + p;
                                amat[g + (size_t)ia * norb] +=
                                        (double)sign;
                        }
                }

                for (int ib = 0; ib < nb; ib++) {
                        int bstr0 = strb_ids[strb_offsets[kb] + ib];
                        for (int ilink = 0; ilink < nlinkb; ilink++) {
                                int *link = linkb + (bstr0 * nlinkb + ilink)
                                        * NLINK_FIELDS;
                                int sign = link[LINK_SIGN];
                                int k;
                                int p;
                                int g;
                                if (sign == 0) {
                                        break;
                                }
                                if (link[LINK_TARGET] != bstr0 ||
                                    mod_pos(link[LINK_DK], nkpts) != dk_zero ||
                                    mod_pos(link[LINK_K_CRE], nkpts) !=
                                    mod_pos(link[LINK_K_DES], nkpts)) {
                                        continue;
                                }
                                k = mod_pos(link[LINK_K_CRE], nkpts);
                                p = link[LINK_CRE] % ncas;
                                g = k * ncas + p;
                                bmat[ib + (size_t)g * nb] += (double)sign;
                        }
                }

                zgemm_(&TRANS_T, &TRANS_N, &norb, &na, &norb,
                       &Z1, wmat, &norb, amat, &norb,
                       &Z0, tmp, &norb);
                zgemm_(&TRANS_N, &TRANS_N, &nb, &na, &norb,
                       &Z1, bmat, &nb, tmp, &norb,
                       &Z1, hdiag + offset, &nb);

                free(tmp);
                free(bmat);
                free(amat);
        }

        free(wmat);
}
