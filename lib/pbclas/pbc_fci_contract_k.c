#include <complex.h>
#include <omp.h>
#include <stdlib.h>
#include "../fblas.h"
#include "pbc_kfci_common.h"

/*
Author: Bhavnesh Jangid
*/

/*
 * k-point FCI contraction helpers.
 *
 * This file implements the low-level complex 2e contraction for the
 * momentum-sector k-FCI representation used by direct_spin1_kfci.py.  The
 * Python layer owns the k-sector link data and structural contraction maps.
 *
 * Memory management:
 * - Caller-provided integral, CI, link, block, and contraction-map arrays are
 *   never freed here.
 * - LinkOrderK owns its offsets and indices arrays; the calling contraction
 *   routine frees them before returning.
 * - Block lookup tables, task tables, group-key tables, prefixes, locks, and
 *   contraction work buffers are temporary and freed by their allocating
 *   routine.
 * - pbc_kfci_make_block_tables and build_ab_tasks transfer their allocated
 *   arrays to the caller on success and free them locally on failure.
 */

#define AB_TASK_CHUNK 1048576

typedef struct {
        int *offsets;
        int *indices;
        int nlinks_total;
} LinkOrderK;

/**
 * Group links by target string and momentum transfer.
 *
 * offsets[target * nkpts + dK] gives the first link ending at target with
 * transfer dK.  This reverse ordering lets a thread gather all contributions
 * to an output string instead of scattering contributions from source strings.
 * @param link_index Momentum-labelled link table.
 * @param nstr Number of strings.
 * @param nlink Number of links per string.
 * @param nkpts Number of k-points.
 * @param order Output grouped-link table; caller frees offsets and indices.
 * @return 0 on success; 1 on allocation failure.
 */
static int make_link_order_target_k(int *link_index, int nstr, int nlink,
                                    int nkpts, LinkOrderK *order)
{
        int table_size = nstr * nkpts;
        int nlinks_total = nstr * nlink;
        int *counts = calloc((size_t)table_size, sizeof(int));
        int *offsets = calloc((size_t)table_size + 1, sizeof(int));
        int *indices = NULL;

        order->offsets = NULL;
        order->indices = NULL;
        order->nlinks_total = 0;

        if (counts == NULL || offsets == NULL) {
                free(counts);
                free(offsets);
                return 1;
        }

        for (int ilink = 0; ilink < nlinks_total; ilink++) {
                int *row = link_index + ilink * NLINK_FIELDS;
                int target = row[LINK_TARGET];
                if (row[LINK_SIGN] == 0 || target < 0) {
                        continue;
                }
                int dk = mod_pos(row[LINK_DK], nkpts);
                counts[target * nkpts + dk]++;
        }

        offsets[0] = 0;
        for (int i = 0; i < table_size; i++) {
                offsets[i + 1] = offsets[i] + counts[i];
                counts[i] = offsets[i];
        }

        indices = malloc(sizeof(int) * (size_t)(offsets[table_size] > 0
                                                ? offsets[table_size] : 1));
        if (indices == NULL) {
                free(counts);
                free(offsets);
                return 1;
        }

        for (int ilink = 0; ilink < nlinks_total; ilink++) {
                int *row = link_index + ilink * NLINK_FIELDS;
                int target = row[LINK_TARGET];
                if (row[LINK_SIGN] == 0 || target < 0) {
                        continue;
                }
                int dk = mod_pos(row[LINK_DK], nkpts);
                int key = target * nkpts + dk;
                indices[counts[key]++] = ilink;
        }

        order->offsets = offsets;
        order->indices = indices;
        order->nlinks_total = offsets[table_size];

        free(counts);
        return 0;
}

/**
 * Apply the one-electron Hamiltonian to a packed k-sector CI vector.
 * @param h1e One-electron integrals arranged by k-point.
 * @param ci0 Input packed CI vector.
 * @param ci1 Output packed CI vector.
 * @param nkpts Number of k-points.
 * @param ncas Number of active orbitals per k-point.
 * @param nblocks Number of packed CI blocks.
 * @param blocks Packed CI block records.
 * @param linka Alpha link table.
 * @param nstra Number of alpha strings.
 * @param nlinka Number of links per alpha string.
 * @param linkb Beta link table.
 * @param nstrb Number of beta strings.
 * @param nlinkb Number of links per beta string.
 * @param stra_ids Alpha string IDs grouped by momentum.
 * @param stra_offsets Offsets into stra_ids.
 * @param strb_ids Beta string IDs grouped by momentum.
 * @param strb_offsets Offsets into strb_ids.
 * @param str2tot_a Alpha global-to-sector string map.
 * @param str2tot_b Beta global-to-sector string map.
 * @param dk_zero Momentum label for zero transfer.
 * @return Nothing.
 */
void FCIcontract_1e_k(double complex *h1e,
                      double complex *ci0,
                      double complex *ci1,
                      int nkpts, int ncas,
                      int nblocks, int *blocks,
                      int *linka, int nstra, int nlinka,
                      int *linkb, int nstrb, int nlinkb,
                      int *stra_ids, int *stra_offsets,
                      int *strb_ids, int *strb_offsets,
                      int *str2tot_a, int *str2tot_b,
                      int dk_zero)
{
        int ndet = 0;

        for (int iblk = 0; iblk < nblocks; iblk++) {
                int *blk = blocks + iblk * 6;
                int offset = blk[BLOCK_OFFSET];
                int size = blk[BLOCK_SIZE];
                if (offset + size > ndet) {
                        ndet = offset + size;
                }
        }

        pbc_kfci_zset0(ci1, (size_t)ndet);

        int *alpha_prefix = malloc(sizeof(int) * (size_t)(nblocks + 1));
        int *beta_prefix = malloc(sizeof(int) * (size_t)(nblocks + 1));
        if (alpha_prefix == NULL || beta_prefix == NULL) {
                free(alpha_prefix);
                free(beta_prefix);
                return;
        }

        alpha_prefix[0] = 0;
        beta_prefix[0] = 0;
        for (int iblk = 0; iblk < nblocks; iblk++) {
                int *blk = blocks + iblk * 6;
                alpha_prefix[iblk + 1] = alpha_prefix[iblk] + blk[BLOCK_NA];
                beta_prefix[iblk + 1] = beta_prefix[iblk] + blk[BLOCK_NB];
        }
        int alpha_tasks = alpha_prefix[nblocks];
        int beta_tasks = beta_prefix[nblocks];

        /*
         * Target-driven contraction.  For a reverse link
         * q^+ p |target> = |source>, the required matrix element is h[p,q],
         * i.e. h[des,cre] in the reverse link row.  Each OpenMP iteration
         * owns a distinct output alpha row or beta column, avoiding atomics
         * and per-thread full CI buffers.
         */
#pragma omp parallel default(none) \
        shared(h1e, ci0, ci1, nkpts, ncas, nblocks, blocks, \
               linka, nstra, nlinka, stra_ids, stra_offsets, str2tot_a, \
               alpha_prefix, alpha_tasks, linkb, nstrb, nlinkb, \
               strb_ids, strb_offsets, str2tot_b, beta_prefix, beta_tasks, \
               dk_zero)
{
#pragma omp for schedule(dynamic)
        for (int itask = 0; itask < alpha_tasks; itask++) {
                int iblk = 0;
                while (alpha_prefix[iblk + 1] <= itask) {
                        iblk++;
                }

                int *blk = blocks + iblk * 6;
                int ka = blk[BLOCK_KA];
                int nb = blk[BLOCK_NB];
                int offset = blk[BLOCK_OFFSET];
                int ia1 = itask - alpha_prefix[iblk];
                int astr1 = stra_ids[stra_offsets[ka] + ia1];

                for (int ilink = 0; ilink < nlinka; ilink++) {
                        int *link = linka + (astr1 * nlinka + ilink)
                                * NLINK_FIELDS;
                        int k_cre = link[LINK_K_CRE] % nkpts;
                        int k_des = link[LINK_K_DES] % nkpts;
                        int dk = link[LINK_DK] % nkpts;

                        if (k_cre != k_des || dk != dk_zero) {
                                continue;
                        }

                        int astr0 = link[LINK_TARGET];
                        if (astr0 < 0 || astr0 >= nstra) {
                                continue;
                        }

                        int ia0 = str2tot_a[ka * nstra + astr0];
                        if (ia0 < 0) {
                                continue;
                        }

                        int p = link[LINK_DES] % ncas;
                        int q = link[LINK_CRE] % ncas;
                        double sign = (double)link[LINK_SIGN];
                        double complex hpq =
                                h1e[(k_cre * ncas + p) * ncas + q];

                        for (int ib = 0; ib < nb; ib++) {
                                ci1[offset + ia1 * nb + ib] +=
                                        sign * hpq *
                                        ci0[offset + ia0 * nb + ib];
                        }
                }
        }

#pragma omp for schedule(dynamic)
        for (int itask = 0; itask < beta_tasks; itask++) {
                int iblk = 0;
                while (beta_prefix[iblk + 1] <= itask) {
                        iblk++;
                }

                int *blk = blocks + iblk * 6;
                int kb = blk[BLOCK_KB];
                int na = blk[BLOCK_NA];
                int nb = blk[BLOCK_NB];
                int offset = blk[BLOCK_OFFSET];
                int ib1 = itask - beta_prefix[iblk];
                int bstr1 = strb_ids[strb_offsets[kb] + ib1];

                for (int ilink = 0; ilink < nlinkb; ilink++) {
                        int *link = linkb + (bstr1 * nlinkb + ilink)
                                * NLINK_FIELDS;
                        int k_cre = link[LINK_K_CRE] % nkpts;
                        int k_des = link[LINK_K_DES] % nkpts;
                        int dk = link[LINK_DK] % nkpts;

                        if (k_cre != k_des || dk != dk_zero) {
                                continue;
                        }

                        int bstr0 = link[LINK_TARGET];
                        if (bstr0 < 0 || bstr0 >= nstrb) {
                                continue;
                        }

                        int ib0 = str2tot_b[kb * nstrb + bstr0];
                        if (ib0 < 0) {
                                continue;
                        }

                        int p = link[LINK_DES] % ncas;
                        int q = link[LINK_CRE] % ncas;
                        double sign = (double)link[LINK_SIGN];
                        double complex hpq =
                                h1e[(k_cre * ncas + p) * ncas + q];

                        for (int ia = 0; ia < na; ia++) {
                                ci1[offset + ia * nb + ib1] +=
                                        sign * hpq *
                                        ci0[offset + ia * nb + ib0];
                        }
                }
        }
}

        free(alpha_prefix);
        free(beta_prefix);
}

/**
 * Apply the spin-squared operator to a packed k-sector CI vector.
 * @param ci0 Input packed CI vector.
 * @param ci1 Output packed CI vector.
 * @param norb Total number of active orbitals.
 * @param neleca Number of alpha electrons.
 * @param nelecb Number of beta electrons.
 * @param nkpts Number of k-points.
 * @param nblocks Number of packed CI blocks.
 * @param blocks Packed CI block records.
 * @param linka Alpha link table.
 * @param nstra Number of alpha strings.
 * @param nlinka Number of links per alpha string.
 * @param linkb Beta link table.
 * @param nstrb Number of beta strings.
 * @param nlinkb Number of links per beta string.
 * @param stra_ids Alpha string IDs grouped by momentum.
 * @param stra_offsets Offsets into stra_ids.
 * @param strb_ids Beta string IDs grouped by momentum.
 * @param strb_offsets Offsets into strb_ids.
 * @param str2tot_a Alpha global-to-sector string map.
 * @param str2tot_b Beta global-to-sector string map.
 * @return Nothing.
 */
void FCIcontract_ss_k(double complex *ci0,
                      double complex *ci1,
                      int norb, int neleca, int nelecb, int nkpts,
                      int nblocks, int *blocks,
                      int *linka, int nstra, int nlinka,
                      int *linkb, int nstrb, int nlinkb,
                      int *stra_ids, int *stra_offsets,
                      int *strb_ids, int *strb_offsets,
                      int *str2tot_a, int *str2tot_b)
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

        size_t lookup_size = (size_t)nstrb * norb * norb;
        int *beta_target = malloc(sizeof(int) * (lookup_size > 0
                                                  ? lookup_size : 1));
        int *beta_sign = calloc(lookup_size > 0 ? lookup_size : 1,
                                sizeof(int));
        int *strk_a = malloc(sizeof(int) * (size_t)(nstra > 0 ? nstra : 1));
        int *strk_b = malloc(sizeof(int) * (size_t)(nstrb > 0 ? nstrb : 1));
        int *alpha_prefix = malloc(sizeof(int) * (size_t)(nblocks + 1));

        if (beta_target == NULL || beta_sign == NULL ||
            strk_a == NULL || strk_b == NULL || alpha_prefix == NULL) {
                free(alpha_prefix);
                free(strk_b);
                free(strk_a);
                free(beta_sign);
                free(beta_target);
                free(block_offset);
                free(block_na);
                free(block_nb);
                return;
        }

        for (size_t i = 0; i < lookup_size; i++) {
                beta_target[i] = -1;
        }
        for (int bstr = 0; bstr < nstrb; bstr++) {
                for (int ilink = 0; ilink < nlinkb; ilink++) {
                        int *link = linkb + (bstr * nlinkb + ilink)
                                * NLINK_FIELDS;
                        int target = link[LINK_TARGET];
                        int sign = link[LINK_SIGN];
                        if (target < 0 || sign == 0) {
                                continue;
                        }
                        int p = link[LINK_CRE];
                        int q = link[LINK_DES];
                        size_t idx = ((size_t)bstr * norb + p) * norb + q;
                        beta_target[idx] = target;
                        beta_sign[idx] = sign;
                }
        }

        for (int k = 0; k < nkpts; k++) {
                for (int i = stra_offsets[k]; i < stra_offsets[k + 1]; i++) {
                        strk_a[stra_ids[i]] = k;
                }
                for (int i = strb_offsets[k]; i < strb_offsets[k + 1]; i++) {
                        strk_b[strb_ids[i]] = k;
                }
        }

        alpha_prefix[0] = 0;
        for (int iblk = 0; iblk < nblocks; iblk++) {
                alpha_prefix[iblk + 1] = alpha_prefix[iblk]
                        + blocks[iblk * 6 + BLOCK_NA];
        }
        int alpha_tasks = alpha_prefix[nblocks];
        double ss0 = 0.25 * (neleca - nelecb) * (neleca - nelecb)
                + 0.5 * (neleca + nelecb);

#pragma omp parallel for schedule(dynamic) default(none) \
        shared(ci0, ci1, norb, nkpts, nblocks, blocks, \
               linka, nstra, nlinka, nstrb, \
               stra_ids, stra_offsets, strb_ids, strb_offsets, \
               str2tot_a, str2tot_b, block_offset, block_nb, \
               beta_target, beta_sign, strk_a, strk_b, \
               alpha_prefix, alpha_tasks, ss0)
        for (int itask = 0; itask < alpha_tasks; itask++) {
                int iblk = 0;
                while (alpha_prefix[iblk + 1] <= itask) {
                        iblk++;
                }

                int *blk = blocks + iblk * 6;
                int ka1 = blk[BLOCK_KA];
                int kb1 = blk[BLOCK_KB];
                int nb1 = blk[BLOCK_NB];
                int dst_offset = blk[BLOCK_OFFSET];
                int ia1 = itask - alpha_prefix[iblk];
                int astr1 = stra_ids[stra_offsets[ka1] + ia1];

                for (int ib1 = 0; ib1 < nb1; ib1++) {
                        int bstr1 = strb_ids[strb_offsets[kb1] + ib1];
                        int dst = dst_offset + ia1 * nb1 + ib1;
                        double complex value = ss0 * ci0[dst];

                        for (int ilink = 0; ilink < nlinka; ilink++) {
                                int *la = linka + (astr1 * nlinka + ilink)
                                        * NLINK_FIELDS;
                                int astr0 = la[LINK_TARGET];
                                int signa = la[LINK_SIGN];
                                if (astr0 < 0 || signa == 0) {
                                        continue;
                                }

                                int p = la[LINK_CRE];
                                int q = la[LINK_DES];
                                size_t idx = ((size_t)bstr1 * norb + q)
                                        * norb + p;
                                int bstr0 = beta_target[idx];
                                if (bstr0 < 0) {
                                        continue;
                                }

                                int ka0 = strk_a[astr0];
                                int kb0 = strk_b[bstr0];
                                int src_key = ka0 * nkpts + kb0;
                                int src_offset = block_offset[src_key];
                                if (src_offset < 0) {
                                        continue;
                                }

                                int ia0 = str2tot_a[ka0 * nstra + astr0];
                                int ib0 = str2tot_b[kb0 * nstrb + bstr0];
                                if (ia0 < 0 || ib0 < 0) {
                                        continue;
                                }

                                int src = src_offset
                                        + ia0 * block_nb[src_key] + ib0;
                                value -= signa * beta_sign[idx] * ci0[src];
                        }
                        ci1[dst] = value;
                }
        }

        free(alpha_prefix);
        free(strk_b);
        free(strk_a);
        free(beta_sign);
        free(beta_target);
        free(block_offset);
        free(block_na);
        free(block_nb);
}

/**
 * Stream the alpha-beta two-electron contraction in one momentum sector.
 *
 * Alpha and beta links are paired when their momentum transfers are dK and
 * -dK.  Reverse link tables are used to gather contributions for each target
 * determinant directly from the packed source blocks.
 *
 * One OpenMP task owns one alpha-string row of an output block.  All writes
 * made by that task are therefore disjoint from the writes of other tasks, so
 * the contraction does not need atomics, locks, or thread-local CI vectors.
 * @param eri Two-electron integrals in k-point storage order.
 * @param ci0 Input packed CI vector.
 * @param ci1 Output packed CI vector; contributions are accumulated.
 * @param nkpts Number of k-points.
 * @param ncas Number of active orbitals per k-point.
 * @param nblocks Number of packed CI blocks.
 * @param blocks Packed CI block records.
 * @param linka Alpha link table.
 * @param nstra Number of alpha strings.
 * @param nlinka Number of links per alpha string.
 * @param linkb Beta link table.
 * @param nstrb Number of beta strings.
 * @param nlinkb Number of links per beta string.
 * @param stra_ids Alpha string IDs grouped by momentum.
 * @param stra_offsets Offsets into stra_ids.
 * @param strb_ids Beta string IDs grouped by momentum.
 * @param strb_offsets Offsets into strb_ids.
 * @param str2tot_a Alpha global-to-sector string map.
 * @param str2tot_b Beta global-to-sector string map.
 * @param kneg K-point additive-inverse table.
 * @return Nothing.
 */
void FCIcontract_2e_k_stream_ab(double complex *eri,
                                double complex *ci0,
                                double complex *ci1,
                                int nkpts, int ncas,
                                int nblocks, int *blocks,
                                int *linka, int nstra, int nlinka,
                                int *linkb, int nstrb, int nlinkb,
                                int *stra_ids, int *stra_offsets,
                                int *strb_ids, int *strb_offsets,
                                int *str2tot_a, int *str2tot_b,
                                int *kneg)
{
        int ndet = 0;
        int *block_offset = NULL;
        int *block_na = NULL;
        int *block_nb = NULL;
        int *alpha_prefix = NULL;
        LinkOrderK order_a;
        LinkOrderK order_b;

        order_a.offsets = NULL;
        order_a.indices = NULL;
        order_a.nlinks_total = 0;
        order_b.offsets = NULL;
        order_b.indices = NULL;
        order_b.nlinks_total = 0;

        if (pbc_kfci_make_block_tables(nkpts, nblocks, blocks,
                              &block_offset, &block_na, &block_nb,
                              &ndet) != 0) {
                return;
        }
        /* Reverse links are indexed by target string and dK. */
        if (make_link_order_target_k(
                    linka, nstra, nlinka, nkpts, &order_a) != 0 ||
            make_link_order_target_k(
                    linkb, nstrb, nlinkb, nkpts, &order_b) != 0) {
                free(order_a.offsets);
                free(order_a.indices);
                free(order_b.offsets);
                free(order_b.indices);
                free(block_offset);
                free(block_na);
                free(block_nb);
                return;
        }

        alpha_prefix = malloc(sizeof(int) * (size_t)(nblocks + 1));
        if (alpha_prefix == NULL) {
                free(order_a.offsets);
                free(order_a.indices);
                free(order_b.offsets);
                free(order_b.indices);
                free(block_offset);
                free(block_na);
                free(block_nb);
                return;
        }

        /* Flatten the alpha rows from all packed momentum blocks. */
        alpha_prefix[0] = 0;
        for (int iblk = 0; iblk < nblocks; iblk++) {
                alpha_prefix[iblk + 1] = alpha_prefix[iblk]
                        + blocks[iblk * 6 + BLOCK_NA];
        }
        int alpha_tasks = alpha_prefix[nblocks];

#pragma omp parallel for schedule(dynamic) default(none) \
        shared(eri, ci0, ci1, nkpts, ncas, nblocks, blocks, \
               linka, nstra, nlinka, linkb, nstrb, nlinkb, \
               stra_ids, stra_offsets, strb_ids, strb_offsets, \
               str2tot_a, str2tot_b, block_offset, block_nb, \
               order_a, order_b, kneg, alpha_prefix, alpha_tasks)
        for (int itask = 0; itask < alpha_tasks; itask++) {
                /* This task owns the complete output row (ia1, all ib1). */
                int iblk = 0;
                while (alpha_prefix[iblk + 1] <= itask) {
                        iblk++;
                }

                int *blk = blocks + iblk * 6;
                int ka1 = blk[BLOCK_KA];
                int kb1 = blk[BLOCK_KB];
                int nb1 = blk[BLOCK_NB];
                int dst_offset = blk[BLOCK_OFFSET];
                int ia1 = itask - alpha_prefix[iblk];
                int astr1 = stra_ids[stra_offsets[ka1] + ia1];

                for (int ib1 = 0; ib1 < nb1; ib1++) {
                        int bstr1 = strb_ids[strb_offsets[kb1] + ib1];
                        double complex value = 0.0 + 0.0 * I;

                        for (int dka = 0; dka < nkpts; dka++) {
                                /* Total momentum is conserved by dKb = -dKa. */
                                int dkb = kneg[dka];
                                int akey = astr1 * nkpts + dka;
                                int bkey = bstr1 * nkpts + dkb;
                                int a0 = order_a.offsets[akey];
                                int a1 = order_a.offsets[akey + 1];
                                int b0 = order_b.offsets[bkey];
                                int b1 = order_b.offsets[bkey + 1];

                                for (int ia = a0; ia < a1; ia++) {
                                        int aid = order_a.indices[ia];
                                        int astr0 = aid / nlinka;
                                        int *la = linka + aid * NLINK_FIELDS;
                                        int ka0 = mod_pos(
                                                la[LINK_K0], nkpts);
                                        int aloc0 = str2tot_a[
                                                ka0 * nstra + astr0];
                                        if (aloc0 < 0) {
                                                continue;
                                        }

                                        for (int ib = b0; ib < b1; ib++) {
                                                int bid = order_b.indices[ib];
                                                int bstr0 = bid / nlinkb;
                                                int *lb = linkb
                                                        + bid * NLINK_FIELDS;
                                                int kb0 = mod_pos(
                                                        lb[LINK_K0], nkpts);
                                                int src_key = ka0 * nkpts
                                                        + kb0;
                                                int src_offset =
                                                        block_offset[src_key];
                                                if (src_offset < 0) {
                                                        continue;
                                                }

                                                int bloc0 = str2tot_b[
                                                        kb0 * nstrb + bstr0];
                                                if (bloc0 < 0) {
                                                        continue;
                                                }

                                                /*
                                                 * Both spin orderings enter
                                                 * the alpha-beta contraction.
                                                 */
                                                long long eri_idx_ab =
                                                        eri_index_k(
                                                        la[LINK_K_CRE],
                                                        la[LINK_K_DES],
                                                        lb[LINK_K_CRE],
                                                        la[LINK_CRE] % ncas,
                                                        la[LINK_DES] % ncas,
                                                        lb[LINK_CRE] % ncas,
                                                        lb[LINK_DES] % ncas,
                                                        nkpts, ncas);
                                                long long eri_idx_ba =
                                                        eri_index_k(
                                                        lb[LINK_K_CRE],
                                                        lb[LINK_K_DES],
                                                        la[LINK_K_CRE],
                                                        lb[LINK_CRE] % ncas,
                                                        lb[LINK_DES] % ncas,
                                                        la[LINK_CRE] % ncas,
                                                        la[LINK_DES] % ncas,
                                                        nkpts, ncas);
                                                double sign = (double)(
                                                        la[LINK_SIGN]
                                                        * lb[LINK_SIGN]);
                                                /*
                                                 * Address in the packed
                                                 * source block.
                                                 */
                                                int src = src_offset
                                                        + aloc0
                                                        * block_nb[src_key]
                                                        + bloc0;
                                                value += (
                                                        eri[eri_idx_ab]
                                                        + eri[eri_idx_ba])
                                                        * sign * ci0[src];
                                        }
                                }
                        }

                        int dst = dst_offset + ia1 * nb1 + ib1;
                        ci1[dst] += value;
                }
        }

        free(alpha_prefix);
        free(order_a.offsets);
        free(order_a.indices);
        free(order_b.offsets);
        free(order_b.indices);
        free(block_offset);
        free(block_na);
        free(block_nb);
}

/** Return the momentum-block key for a packed CI offset, or -1 if absent. */
static int find_block_key_by_offset(int *block_offset, int table_size,
                                    int offset)
{
        for (int key = 0; key < table_size; key++) {
                if (block_offset[key] == offset) {
                        return key;
                }
        }
        return -1;
}

/**
 * Map each contraction group to its source and destination block keys.
 * @param table_size Number of momentum-pair block slots.
 * @param block_offset Packed offset for each block slot.
 * @param group_tab Contraction group records.
 * @param group_stride Number of integers per group record.
 * @param group_offsets Group ranges for each source block.
 * @param ngroups Number of contraction groups.
 * @param group_src_key Output source-block keys.
 * @param group_dst_key Output destination-block keys.
 * @return 0 on success; 1 if a destination block is absent.
 */
static int fill_group_keys(int table_size,
                           int *block_offset,
                           int *group_tab,
                           int group_stride,
                           int *group_offsets,
                           int ngroups,
                           int *group_src_key,
                           int *group_dst_key)
{
        for (int ig = 0; ig < ngroups; ig++) {
                group_src_key[ig] = -1;
                group_dst_key[ig] = -1;
        }

        for (int src_key = 0; src_key < table_size; src_key++) {
                int group0 = group_offsets[src_key];
                int group1 = group_offsets[src_key + 1];

                for (int ig = group0; ig < group1; ig++) {
                        int *group = group_tab + ig * group_stride;
                        int dst_key = find_block_key_by_offset(
                                block_offset, table_size, group[0]);
                        if (dst_key < 0) {
                                return 1;
                        }
                        group_src_key[ig] = src_key;
                        group_dst_key[ig] = dst_key;
                }
        }
        return 0;
}

/**
 * Split large alpha-beta contraction groups into bounded OpenMP tasks.
 * @param ab_group_tab Alpha-beta group records.
 * @param ab_ngroups Number of alpha-beta groups.
 * @param p_task_group Output group index per task; caller owns the array.
 * @param p_task_entry0 Output first entry per task; caller owns the array.
 * @param p_task_entry1 Output end entry per task; caller owns the array.
 * @param p_ntasks Output number of tasks.
 * @return 0 on success; 1 on allocation failure.
 */
static int build_ab_tasks(int *ab_group_tab,
                          int ab_ngroups,
                          int **p_task_group,
                          int **p_task_entry0,
                          int **p_task_entry1,
                          int *p_ntasks)
{
        int ntasks = 0;

        for (int ig = 0; ig < ab_ngroups; ig++) {
                int *group = ab_group_tab + ig * 3;
                int nentry = group[2] - group[1];
                ntasks += (nentry + AB_TASK_CHUNK - 1) / AB_TASK_CHUNK;
        }

        size_t task_size = (size_t)(ntasks > 0 ? ntasks : 1);
        int *task_group = malloc(sizeof(int) * task_size);
        int *task_entry0 = malloc(sizeof(int) * task_size);
        int *task_entry1 = malloc(sizeof(int) * task_size);
        if (task_group == NULL || task_entry0 == NULL ||
            task_entry1 == NULL) {
                free(task_group);
                free(task_entry0);
                free(task_entry1);
                return 1;
        }

        int itask = 0;
        for (int ig = 0; ig < ab_ngroups; ig++) {
                int *group = ab_group_tab + ig * 3;
                for (int entry0 = group[1]; entry0 < group[2];
                     entry0 += AB_TASK_CHUNK) {
                        int entry1 = entry0 + AB_TASK_CHUNK;
                        if (entry1 > group[2]) {
                                entry1 = group[2];
                        }
                        task_group[itask] = ig;
                        task_entry0[itask] = entry0;
                        task_entry1[itask] = entry1;
                        itask++;
                }
        }

        *p_task_group = task_group;
        *p_task_entry0 = task_entry0;
        *p_task_entry1 = task_entry1;
        *p_ntasks = ntasks;
        return 0;
}

/** Apply all sparse alpha-beta groups for one source CI block. */
static void contract_ab_sparse_struct(double complex *ci0,
                                      double complex *ci1,
                                      double complex *eri,
                                      int src_key,
                                      int src_offset,
                                      int *ab_group_tab,
                                      int *ab_group_offsets,
                                      int *ab_src_addr,
                                      int *ab_dst_addr,
                                      int *ab_sign,
                                      long long *ab_eri_idx_ab,
                                      long long *ab_eri_idx_ba)
{
        int group0 = ab_group_offsets[src_key];
        int group1 = ab_group_offsets[src_key + 1];

        for (int ig = group0; ig < group1; ig++) {
                int *group = ab_group_tab + ig * 3;
                int dst_offset = group[0];

                for (int i = group[1]; i < group[2]; i++) {
                        double complex coef =
                                (eri[ab_eri_idx_ab[i]] +
                                 eri[ab_eri_idx_ba[i]]) *
                                (double)ab_sign[i];
                        ci1[dst_offset + ab_dst_addr[i]] +=
                                coef * ci0[src_offset + ab_src_addr[i]];
                }
        }
}

/** Apply one sparse alpha-beta task and lock its destination block update. */
static void contract_ab_sparse_task(double complex *ci0,
                                    double complex *ci1,
                                    double complex *eri,
                                    double complex *ab_buf,
                                    int src_offset,
                                    int dst_offset,
                                    int dst_size,
                                    int entry0,
                                    int entry1,
                                    omp_lock_t *dst_lock,
                                    int *ab_src_addr,
                                    int *ab_dst_addr,
                                    int *ab_sign,
                                    long long *ab_eri_idx_ab,
                                    long long *ab_eri_idx_ba)
{
        const int inc = 1;
        const double complex Z1 = 1.0 + 0.0 * I;

        pbc_kfci_zset0(ab_buf, (size_t)dst_size);
        for (int i = entry0; i < entry1; i++) {
                double complex coef =
                        (eri[ab_eri_idx_ab[i]] +
                         eri[ab_eri_idx_ba[i]]) * (double)ab_sign[i];
                ab_buf[ab_dst_addr[i]] +=
                        coef * ci0[src_offset + ab_src_addr[i]];
        }

        omp_set_lock(dst_lock);
        zaxpy_(&dst_size, &Z1, ab_buf, &inc,
               ci1 + dst_offset, &inc);
        omp_unset_lock(dst_lock);
}

/** Apply all alpha-alpha groups for one source block with dense GEMM. */
static void contract_aa_zgemm_struct(double complex *eri,
                                     double complex *ci0,
                                     double complex *ci1,
                                     double complex *amat,
                                     int nkpts, int ka, int kb,
                                     int na, int nb,
                                     int src_offset,
                                     int *aa_group_tab,
                                     int *aa_group_offsets,
                                     int *aa_src_addr,
                                     int *aa_dst_addr,
                                     int *aa_sign,
                                     long long *aa_eri_idx)
{
        const char TRANS_N = 'N';
        const double complex Z1 = 1.0 + 0.0 * I;
        int src_key = ka * nkpts + kb;
        int group0 = aa_group_offsets[src_key];
        int group1 = aa_group_offsets[src_key + 1];

        if (na == 0 || nb == 0) {
                return;
        }

        for (int ig = group0; ig < group1; ig++) {
                int *group = aa_group_tab + ig * 4;
                int dst_offset = group[0];
                int dst_na = group[1];
                int entry0 = group[2];
                int entry1 = group[3];

                pbc_kfci_zset0(amat, (size_t)dst_na * na);
                for (int i = entry0; i < entry1; i++) {
                        amat[aa_dst_addr[i] * (size_t)na + aa_src_addr[i]] +=
                                eri[aa_eri_idx[i]] * (double)aa_sign[i];
                }

                zgemm_(&TRANS_N, &TRANS_N, &nb, &dst_na, &na,
                       &Z1, ci0 + src_offset, &nb,
                       amat, &na,
                       &Z1, ci1 + dst_offset, &nb);
        }
}

/** Apply one alpha-alpha GEMM group with a locked destination update. */
static void contract_aa_zgemm_group(double complex *eri,
                                    double complex *ci0,
                                    double complex *ci1,
                                    double complex *amat,
                                    int na, int nb,
                                    int src_offset,
                                    omp_lock_t *dst_lock,
                                    int *group,
                                    int *aa_src_addr,
                                    int *aa_dst_addr,
                                    int *aa_sign,
                                    long long *aa_eri_idx)
{
        const char TRANS_N = 'N';
        const double complex Z1 = 1.0 + 0.0 * I;
        int dst_offset = group[0];
        int dst_na = group[1];
        int entry0 = group[2];
        int entry1 = group[3];

        if (na == 0 || nb == 0) {
                return;
        }

        pbc_kfci_zset0(amat, (size_t)dst_na * na);
        for (int i = entry0; i < entry1; i++) {
                amat[aa_dst_addr[i] * (size_t)na + aa_src_addr[i]] +=
                        eri[aa_eri_idx[i]] * (double)aa_sign[i];
        }

        omp_set_lock(dst_lock);
        zgemm_(&TRANS_N, &TRANS_N, &nb, &dst_na, &na,
               &Z1, ci0 + src_offset, &nb,
               amat, &na,
               &Z1, ci1 + dst_offset, &nb);
        omp_unset_lock(dst_lock);
}

/** Apply all beta-beta groups for one source block with dense GEMM. */
static void contract_bb_zgemm_struct(double complex *eri,
                                     double complex *ci0,
                                     double complex *ci1,
                                     double complex *bmat,
                                     int nkpts, int ka, int kb,
                                     int na, int nb,
                                     int src_offset,
                                     int *bb_group_tab,
                                     int *bb_group_offsets,
                                     int *bb_src_addr,
                                     int *bb_dst_addr,
                                     int *bb_sign,
                                     long long *bb_eri_idx)
{
        const char TRANS_N = 'N';
        const double complex Z1 = 1.0 + 0.0 * I;
        int src_key = ka * nkpts + kb;
        int group0 = bb_group_offsets[src_key];
        int group1 = bb_group_offsets[src_key + 1];

        if (na == 0 || nb == 0) {
                return;
        }

        for (int ig = group0; ig < group1; ig++) {
                int *group = bb_group_tab + ig * 4;
                int dst_offset = group[0];
                int dst_nb = group[1];
                int entry0 = group[2];
                int entry1 = group[3];

                pbc_kfci_zset0(bmat, (size_t)nb * dst_nb);
                for (int i = entry0; i < entry1; i++) {
                        bmat[bb_src_addr[i] * (size_t)dst_nb +
                             bb_dst_addr[i]] +=
                                eri[bb_eri_idx[i]] * (double)bb_sign[i];
                }

                zgemm_(&TRANS_N, &TRANS_N, &dst_nb, &na, &nb,
                       &Z1, bmat, &dst_nb,
                       ci0 + src_offset, &nb,
                       &Z1, ci1 + dst_offset, &dst_nb);
        }
}

/** Apply one beta-beta GEMM group with a locked destination update. */
static void contract_bb_zgemm_group(double complex *eri,
                                    double complex *ci0,
                                    double complex *ci1,
                                    double complex *bmat,
                                    int na, int nb,
                                    int src_offset,
                                    omp_lock_t *dst_lock,
                                    int *group,
                                    int *bb_src_addr,
                                    int *bb_dst_addr,
                                    int *bb_sign,
                                    long long *bb_eri_idx)
{
        const char TRANS_N = 'N';
        const double complex Z1 = 1.0 + 0.0 * I;
        int dst_offset = group[0];
        int dst_nb = group[1];
        int entry0 = group[2];
        int entry1 = group[3];

        if (na == 0 || nb == 0) {
                return;
        }

        pbc_kfci_zset0(bmat, (size_t)nb * dst_nb);
        for (int i = entry0; i < entry1; i++) {
                bmat[bb_src_addr[i] * (size_t)dst_nb + bb_dst_addr[i]] +=
                        eri[bb_eri_idx[i]] * (double)bb_sign[i];
        }

        omp_set_lock(dst_lock);
        zgemm_(&TRANS_N, &TRANS_N, &dst_nb, &na, &nb,
               &Z1, bmat, &dst_nb,
               ci0 + src_offset, &nb,
               &Z1, ci1 + dst_offset, &dst_nb);
        omp_unset_lock(dst_lock);
}

/**
 * Apply the mapped two-electron Hamiltonian to a packed k-sector CI vector.
 * @param eri Two-electron integrals in k-point storage order.
 * @param ci0 Input packed CI vector.
 * @param ci1 Output packed CI vector.
 * @param nkpts Number of k-points.
 * @param ncas Number of active orbitals per k-point.
 * @param nblocks Number of packed CI blocks.
 * @param blocks Packed CI block records.
 * @param ab_group_tab Alpha-beta contraction groups.
 * @param ab_group_offsets Alpha-beta group ranges by source block.
 * @param ab_src_addr Alpha-beta source addresses.
 * @param ab_dst_addr Alpha-beta destination addresses.
 * @param ab_sign Alpha-beta fermionic signs.
 * @param ab_eri_idx_ab Alpha-beta ERI indices in AB order.
 * @param ab_eri_idx_ba Alpha-beta ERI indices in BA order.
 * @param nab_entries Number of alpha-beta entries.
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
void FCIcontract_2e_k(double complex *eri,
                      double complex *ci0,
                      double complex *ci1,
                      int nkpts, int ncas,
                      int nblocks, int *blocks,
                      int *ab_group_tab,
                      int *ab_group_offsets,
                      int *ab_src_addr,
                      int *ab_dst_addr,
                      int *ab_sign,
                      long long *ab_eri_idx_ab,
                      long long *ab_eri_idx_ba,
                      int nab_entries,
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
        (void)nab_entries;

        int ndet = 0;
        int *block_offset = NULL;
        int *block_na = NULL;
        int *block_nb = NULL;

        if (pbc_kfci_make_block_tables(nkpts, nblocks, blocks,
                              &block_offset, &block_na, &block_nb,
                              &ndet) != 0) {
                return;
        }
        if (ndet == 0) {
                free(block_offset);
                free(block_na);
                free(block_nb);
                return;
        }

        int max_na = 0;
        int max_nb = 0;
        int max_block_size = 0;
        for (int iblk = 0; iblk < nblocks; iblk++) {
                int *blk = blocks + iblk * 6;
                if (blk[BLOCK_NA] > max_na) {
                        max_na = blk[BLOCK_NA];
                }
                if (blk[BLOCK_NB] > max_nb) {
                        max_nb = blk[BLOCK_NB];
                }
                if (blk[BLOCK_SIZE] > max_block_size) {
                        max_block_size = blk[BLOCK_SIZE];
                }
        }

        size_t ndet_size = (size_t)ndet;
        size_t ab_work_size = (size_t)max_block_size;
        size_t aa_work_size = (size_t)max_na * max_na;
        size_t bb_work_size = (size_t)max_nb * max_nb;
        if (ab_work_size == 0) {
                ab_work_size = 1;
        }
        if (aa_work_size == 0) {
                aa_work_size = 1;
        }
        if (bb_work_size == 0) {
                bb_work_size = 1;
        }

        int table_size = nkpts * nkpts;
        int ab_ngroups = ab_group_offsets[table_size];
        int aa_ngroups = aa_group_offsets[table_size];
        int bb_ngroups = bb_group_offsets[table_size];
        size_t ab_group_size = (size_t)(ab_ngroups > 0 ? ab_ngroups : 1);
        size_t aa_group_size = (size_t)(aa_ngroups > 0 ? aa_ngroups : 1);
        size_t bb_group_size = (size_t)(bb_ngroups > 0 ? bb_ngroups : 1);

        pbc_kfci_zset0(ci1, (size_t)ndet);
        int status = 0;
        int ab_ntasks = 0;
        int *ab_task_group = NULL;
        int *ab_task_entry0 = NULL;
        int *ab_task_entry1 = NULL;
        int *ab_group_src_key = malloc(sizeof(int) * ab_group_size);
        int *ab_group_dst_key = malloc(sizeof(int) * ab_group_size);
        int *aa_group_src_key = malloc(sizeof(int) * aa_group_size);
        int *aa_group_dst_key = malloc(sizeof(int) * aa_group_size);
        int *bb_group_src_key = malloc(sizeof(int) * bb_group_size);
        int *bb_group_dst_key = malloc(sizeof(int) * bb_group_size);
        omp_lock_t *block_locks = malloc(sizeof(omp_lock_t)
                                         * (size_t)table_size);

        if (ab_group_src_key == NULL || ab_group_dst_key == NULL ||
            aa_group_src_key == NULL || aa_group_dst_key == NULL ||
            bb_group_src_key == NULL || bb_group_dst_key == NULL ||
            block_locks == NULL ||
            build_ab_tasks(ab_group_tab, ab_ngroups,
                           &ab_task_group, &ab_task_entry0,
                           &ab_task_entry1, &ab_ntasks) != 0 ||
            fill_group_keys(table_size, block_offset, ab_group_tab, 3,
                            ab_group_offsets, ab_ngroups,
                            ab_group_src_key, ab_group_dst_key) != 0 ||
            fill_group_keys(table_size, block_offset, aa_group_tab, 4,
                            aa_group_offsets, aa_ngroups,
                            aa_group_src_key, aa_group_dst_key) != 0 ||
            fill_group_keys(table_size, block_offset, bb_group_tab, 4,
                            bb_group_offsets, bb_ngroups,
                            bb_group_src_key, bb_group_dst_key) != 0) {
                status = 1;
        }

        if (status == 0) {
                for (int key = 0; key < table_size; key++) {
                        omp_init_lock(&block_locks[key]);
                }

#pragma omp parallel default(none) \
        shared(eri, ci0, ci1, nkpts, ncas, \
               block_offset, block_na, block_nb, block_locks, \
               ab_group_tab, ab_group_offsets, ab_src_addr, ab_dst_addr, \
               ab_sign, ab_eri_idx_ab, ab_eri_idx_ba, \
               ab_group_src_key, ab_group_dst_key, \
               ab_task_group, ab_task_entry0, ab_task_entry1, ab_ntasks, \
               aa_group_tab, aa_group_offsets, aa_group_src_key, \
               aa_group_dst_key, aa_ngroups, \
               aa_src_addr, aa_dst_addr, aa_sign, aa_eri_idx, \
               bb_group_tab, bb_group_offsets, bb_group_src_key, \
               bb_group_dst_key, bb_ngroups, \
               bb_src_addr, bb_dst_addr, bb_sign, bb_eri_idx, \
               ab_work_size, aa_work_size, bb_work_size, status)
{
        double complex *ab_buf = malloc(sizeof(double complex) * ab_work_size);
        double complex *amat = malloc(sizeof(double complex) * aa_work_size);
        double complex *bmat = malloc(sizeof(double complex) * bb_work_size);
        int ok = (ab_buf != NULL && amat != NULL && bmat != NULL);

        if (!ok) {
#pragma omp atomic write
                status = 1;
        }

#pragma omp barrier

        if (status == 0) {
#pragma omp for schedule(dynamic)
                for (int itask = 0; itask < ab_ntasks; itask++) {
                        int ig = ab_task_group[itask];
                        int src_key = ab_group_src_key[ig];
                        int dst_key = ab_group_dst_key[ig];
                        int src_offset = block_offset[src_key];
                        int *group = ab_group_tab + ig * 3;
                        int dst_size = block_na[dst_key] * block_nb[dst_key];

                        contract_ab_sparse_task(
                                ci0, ci1, eri, ab_buf, src_offset, group[0],
                                dst_size, ab_task_entry0[itask],
                                ab_task_entry1[itask], &block_locks[dst_key],
                                ab_src_addr, ab_dst_addr, ab_sign,
                                ab_eri_idx_ab, ab_eri_idx_ba);
                }

#pragma omp for schedule(dynamic)
                for (int ig = 0; ig < aa_ngroups; ig++) {
                        int src_key = aa_group_src_key[ig];
                        int dst_key = aa_group_dst_key[ig];
                        int na = block_na[src_key];
                        int nb = block_nb[src_key];
                        int src_offset = block_offset[src_key];
                        int *group = aa_group_tab + ig * 4;

                        contract_aa_zgemm_group(
                                eri, ci0, ci1, amat, na, nb, src_offset,
                                &block_locks[dst_key], group, aa_src_addr,
                                aa_dst_addr, aa_sign, aa_eri_idx);
                }

#pragma omp for schedule(dynamic)
                for (int ig = 0; ig < bb_ngroups; ig++) {
                        int src_key = bb_group_src_key[ig];
                        int dst_key = bb_group_dst_key[ig];
                        int na = block_na[src_key];
                        int nb = block_nb[src_key];
                        int src_offset = block_offset[src_key];
                        int *group = bb_group_tab + ig * 4;

                        contract_bb_zgemm_group(
                                eri, ci0, ci1, bmat, na, nb, src_offset,
                                &block_locks[dst_key], group, bb_src_addr,
                                bb_dst_addr, bb_sign, bb_eri_idx);
                }
        }

        free(bmat);
        free(amat);
        free(ab_buf);
}

                for (int key = 0; key < table_size; key++) {
                        omp_destroy_lock(&block_locks[key]);
                }
        }

        if (status != 0) {
                double complex *amat = malloc(sizeof(double complex)
                                              * aa_work_size);
                double complex *bmat = malloc(sizeof(double complex)
                                              * bb_work_size);
                if (amat == NULL || bmat == NULL) {
                        free(amat);
                        free(bmat);
                        free(ab_task_entry1);
                        free(ab_task_entry0);
                        free(ab_task_group);
                        free(block_locks);
                        free(bb_group_dst_key);
                        free(bb_group_src_key);
                        free(aa_group_dst_key);
                        free(aa_group_src_key);
                        free(ab_group_dst_key);
                        free(ab_group_src_key);
                        free(block_offset);
                        free(block_na);
                        free(block_nb);
                        return;
                }

                pbc_kfci_zset0(ci1, ndet_size);
                for (int iblk = 0; iblk < nblocks; iblk++) {
                        int *blk = blocks + iblk * 6;
                        int ka = blk[BLOCK_KA];
                        int kb = blk[BLOCK_KB];
                        int na = blk[BLOCK_NA];
                        int nb = blk[BLOCK_NB];
                        int src_offset = blk[BLOCK_OFFSET];
                        int src_key = ka * nkpts + kb;

                        contract_ab_sparse_struct(ci0, ci1, eri,
                                                  src_key, src_offset,
                                                  ab_group_tab,
                                                  ab_group_offsets,
                                                  ab_src_addr,
                                                  ab_dst_addr,
                                                  ab_sign, ab_eri_idx_ab,
                                                  ab_eri_idx_ba);

                        contract_aa_zgemm_struct(
                                eri, ci0, ci1, amat,
                                nkpts, ka, kb, na, nb, src_offset,
                                aa_group_tab, aa_group_offsets,
                                aa_src_addr, aa_dst_addr,
                                aa_sign, aa_eri_idx);

                        contract_bb_zgemm_struct(
                                eri, ci0, ci1, bmat,
                                nkpts, ka, kb, na, nb, src_offset,
                                bb_group_tab, bb_group_offsets,
                                bb_src_addr, bb_dst_addr,
                                bb_sign, bb_eri_idx);
                }

                free(amat);
                free(bmat);
        }

        free(ab_task_entry1);
        free(ab_task_entry0);
        free(ab_task_group);
        free(block_locks);
        free(bb_group_dst_key);
        free(bb_group_src_key);
        free(aa_group_dst_key);
        free(aa_group_src_key);
        free(ab_group_dst_key);
        free(ab_group_src_key);
        free(block_offset);
        free(block_na);
        free(block_nb);
}
