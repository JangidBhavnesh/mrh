/*
 * k-point FCI contraction helpers.
 *
 * This file implements the low-level complex 2e contraction for the
 * momentum-sector k-FCI representation used by direct_spin1_kfci.py.  The
 * Python layer owns the k-sector link data and structural contraction maps.
 */

#include <complex.h>
#include <omp.h>
#include <stdlib.h>
#include "vhf/fblas.h"

#define BLOCK_KA      0
#define BLOCK_KB      1
#define BLOCK_NA      2
#define BLOCK_NB      3
#define BLOCK_OFFSET  4
#define BLOCK_SIZE    5

#define LINK_CRE    0
#define LINK_DES    1
#define LINK_TARGET 2
#define LINK_SIGN   3
#define LINK_K_CRE  5
#define LINK_K_DES  6
#define LINK_DK     7
#define NLINK_FIELDS 8

static void zset0(double complex *x, size_t n)
{
        for (size_t i = 0; i < n; i++) {
                x[i] = 0.0 + 0.0 * I;
        }
}

static void zadd(double complex *out, double complex *in, size_t n)
{
        for (size_t i = 0; i < n; i++) {
                out[i] += in[i];
        }
}

static int make_block_tables(int nkpts, int nblocks, int *blocks,
                             int **p_block_offset,
                             int **p_block_na,
                             int **p_block_nb,
                             int *p_ndet)
{
        int table_size = nkpts * nkpts;
        int ndet = 0;
        int *block_offset = malloc(sizeof(int) * (size_t)table_size);
        int *block_na = malloc(sizeof(int) * (size_t)table_size);
        int *block_nb = malloc(sizeof(int) * (size_t)table_size);

        if (block_offset == NULL || block_na == NULL || block_nb == NULL) {
                free(block_offset);
                free(block_na);
                free(block_nb);
                return 1;
        }

        for (int i = 0; i < table_size; i++) {
                block_offset[i] = -1;
                block_na[i] = 0;
                block_nb[i] = 0;
        }

        for (int iblk = 0; iblk < nblocks; iblk++) {
                int *blk = blocks + iblk * 6;
                int ka = blk[BLOCK_KA];
                int kb = blk[BLOCK_KB];
                int offset = blk[BLOCK_OFFSET];
                int size = blk[BLOCK_SIZE];
                int key = ka * nkpts + kb;
                block_offset[key] = offset;
                block_na[key] = blk[BLOCK_NA];
                block_nb[key] = blk[BLOCK_NB];
                if (offset + size > ndet) {
                        ndet = offset + size;
                }
        }

        *p_block_offset = block_offset;
        *p_block_na = block_na;
        *p_block_nb = block_nb;
        *p_ndet = ndet;
        return 0;
}

void FCIcontract_1e_k(double complex *h1e,
                      double complex *ci0,
                      double complex *ci1,
                      int nkpts, int ncas,
                      int nblocks, int *blocks,
                      int *linka, int nstra, int nlinka,
                      int *linkb, int nstrb, int nlinkb,
                      int *stra_ids, int *stra_offsets,
                      int *strb_ids, int *strb_offsets,
                      int *str2tot_a, int *str2tot_b)
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

        zset0(ci1, (size_t)ndet);

#pragma omp parallel for schedule(dynamic) default(none) \
        shared(h1e, ci0, ci1, nkpts, ncas, nblocks, blocks, \
               linka, nstra, nlinka, linkb, nstrb, nlinkb, \
               stra_ids, stra_offsets, strb_ids, strb_offsets, \
               str2tot_a, str2tot_b)
        for (int iblk = 0; iblk < nblocks; iblk++) {
                int *blk = blocks + iblk * 6;
                int ka = blk[BLOCK_KA];
                int kb = blk[BLOCK_KB];
                int na = blk[BLOCK_NA];
                int nb = blk[BLOCK_NB];
                int offset = blk[BLOCK_OFFSET];

                for (int ia0 = 0; ia0 < na; ia0++) {
                        int astr0 = stra_ids[stra_offsets[ka] + ia0];

                        for (int ilink = 0; ilink < nlinka; ilink++) {
                                int *link = linka + (astr0 * nlinka + ilink)
                                        * NLINK_FIELDS;
                                int k_cre = link[LINK_K_CRE] % nkpts;
                                int k_des = link[LINK_K_DES] % nkpts;
                                int dk = link[LINK_DK] % nkpts;

                                if (k_cre != k_des || dk != 0) {
                                        continue;
                                }

                                int astr1 = link[LINK_TARGET];
                                if (astr1 < 0 || astr1 >= nstra) {
                                        continue;
                                }

                                int ia1 = str2tot_a[ka * nstra + astr1];
                                if (ia1 < 0) {
                                        continue;
                                }

                                int p = link[LINK_CRE] % ncas;
                                int q = link[LINK_DES] % ncas;
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

                for (int ib0 = 0; ib0 < nb; ib0++) {
                        int bstr0 = strb_ids[strb_offsets[kb] + ib0];

                        for (int ilink = 0; ilink < nlinkb; ilink++) {
                                int *link = linkb + (bstr0 * nlinkb + ilink)
                                        * NLINK_FIELDS;
                                int k_cre = link[LINK_K_CRE] % nkpts;
                                int k_des = link[LINK_K_DES] % nkpts;
                                int dk = link[LINK_DK] % nkpts;

                                if (k_cre != k_des || dk != 0) {
                                        continue;
                                }

                                int bstr1 = link[LINK_TARGET];
                                if (bstr1 < 0 || bstr1 >= nstrb) {
                                        continue;
                                }

                                int ib1 = str2tot_b[kb * nstrb + bstr1];
                                if (ib1 < 0) {
                                        continue;
                                }

                                int p = link[LINK_CRE] % ncas;
                                int q = link[LINK_DES] % ncas;
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
}

static void fill_ab_sparse_coef(double complex *eri,
                                double complex *ab_coef,
                                int *ab_sign,
                                long long *ab_eri_idx_ab,
                                long long *ab_eri_idx_ba,
                                int nab_entries)
{
#pragma omp parallel for schedule(static) default(none) \
        shared(eri, ab_coef, ab_sign, ab_eri_idx_ab, ab_eri_idx_ba, nab_entries)
        for (int i = 0; i < nab_entries; i++) {
                ab_coef[i] = (eri[ab_eri_idx_ab[i]] + eri[ab_eri_idx_ba[i]])
                        * (double)ab_sign[i];
        }
}

static void contract_ab_sparse_struct(double complex *ci0,
                                      double complex *ci1,
                                      int src_key,
                                      int src_offset,
                                      int *ab_group_tab,
                                      int *ab_group_offsets,
                                      int *ab_src_addr,
                                      int *ab_dst_addr,
                                      double complex *ab_coef)
{
        int group0 = ab_group_offsets[src_key];
        int group1 = ab_group_offsets[src_key + 1];

        for (int ig = group0; ig < group1; ig++) {
                int *group = ab_group_tab + ig * 3;
                int dst_offset = group[0];

                for (int i = group[1]; i < group[2]; i++) {
                        ci1[dst_offset + ab_dst_addr[i]] +=
                                ab_coef[i] *
                                ci0[src_offset + ab_src_addr[i]];
                }
        }
}

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

                zset0(amat, (size_t)dst_na * na);
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

                zset0(bmat, (size_t)nb * dst_nb);
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

void FCIcontract_2e_k_zgemm_ab_struct(double complex *eri,
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
        int ndet = 0;
        int *block_offset = NULL;
        int *block_na = NULL;
        int *block_nb = NULL;

        if (make_block_tables(nkpts, nblocks, blocks,
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
        for (int iblk = 0; iblk < nblocks; iblk++) {
                int *blk = blocks + iblk * 6;
                if (blk[BLOCK_NA] > max_na) {
                        max_na = blk[BLOCK_NA];
                }
                if (blk[BLOCK_NB] > max_nb) {
                        max_nb = blk[BLOCK_NB];
                }
        }

        size_t ndet_size = (size_t)ndet;
        size_t aa_work_size = (size_t)max_na * max_na;
        size_t bb_work_size = (size_t)max_nb * max_nb;
        if (aa_work_size == 0) {
                aa_work_size = 1;
        }
        if (bb_work_size == 0) {
                bb_work_size = 1;
        }

        double complex *ab_coef = malloc(sizeof(double complex)
                                         * (size_t)(nab_entries > 0 ?
                                                    nab_entries : 1));
        if (ab_coef == NULL) {
                free(block_offset);
                free(block_na);
                free(block_nb);
                return;
        }
        fill_ab_sparse_coef(eri, ab_coef, ab_sign,
                            ab_eri_idx_ab, ab_eri_idx_ba, nab_entries);

        zset0(ci1, (size_t)ndet);
        int status = 0;
        int nthreads = omp_get_max_threads();
        if (nthreads > nblocks) {
                nthreads = nblocks;
        }
        if (nthreads < 1) {
                nthreads = 1;
        }

#pragma omp parallel default(none) \
        num_threads(nthreads) \
        shared(eri, ci0, ci1, nkpts, ncas, nblocks, blocks, \
               block_offset, block_na, block_nb, \
               ab_group_tab, ab_group_offsets, ab_src_addr, ab_dst_addr, \
               ab_coef, aa_group_tab, aa_group_offsets, \
               aa_src_addr, aa_dst_addr, aa_sign, aa_eri_idx, \
               bb_group_tab, bb_group_offsets, \
               bb_src_addr, bb_dst_addr, bb_sign, bb_eri_idx, \
               ndet_size, aa_work_size, bb_work_size, status)
{
        double complex *ci1buf = malloc(sizeof(double complex) * ndet_size);
        double complex *amat = malloc(sizeof(double complex) * aa_work_size);
        double complex *bmat = malloc(sizeof(double complex) * bb_work_size);
        int ok = (ci1buf != NULL && amat != NULL && bmat != NULL);

        if (!ok) {
#pragma omp atomic write
                status = 1;
        }
        if (ok) {
                zset0(ci1buf, ndet_size);
        }

#pragma omp for schedule(dynamic)
        for (int iblk = 0; iblk < nblocks; iblk++) {
                if (ok) {
                        int *blk = blocks + iblk * 6;
                        int ka = blk[BLOCK_KA];
                        int kb = blk[BLOCK_KB];
                        int na = blk[BLOCK_NA];
                        int nb = blk[BLOCK_NB];
                        int src_offset = blk[BLOCK_OFFSET];
                        int src_key = ka * nkpts + kb;

                        contract_ab_sparse_struct(ci0, ci1buf,
                                                  src_key, src_offset,
                                                  ab_group_tab,
                                                  ab_group_offsets,
                                                  ab_src_addr,
                                                  ab_dst_addr,
                                                  ab_coef);

                        contract_aa_zgemm_struct(
                                eri, ci0, ci1buf, amat,
                                nkpts, ka, kb, na, nb, src_offset,
                                aa_group_tab, aa_group_offsets,
                                aa_src_addr, aa_dst_addr,
                                aa_sign, aa_eri_idx);

                        contract_bb_zgemm_struct(
                                eri, ci0, ci1buf, bmat,
                                nkpts, ka, kb, na, nb, src_offset,
                                bb_group_tab, bb_group_offsets,
                                bb_src_addr, bb_dst_addr,
                                bb_sign, bb_eri_idx);
                }
        }

        if (ok) {
#pragma omp critical
                {
                        zadd(ci1, ci1buf, ndet_size);
                }
        }

        free(bmat);
        free(amat);
        free(ci1buf);
}

        if (status != 0) {
                double complex *amat = malloc(sizeof(double complex)
                                              * aa_work_size);
                double complex *bmat = malloc(sizeof(double complex)
                                              * bb_work_size);
                if (amat == NULL || bmat == NULL) {
                        free(amat);
                        free(bmat);
                        free(ab_coef);
                        free(block_offset);
                        free(block_na);
                        free(block_nb);
                        return;
                }

                zset0(ci1, ndet_size);
                for (int iblk = 0; iblk < nblocks; iblk++) {
                        int *blk = blocks + iblk * 6;
                        int ka = blk[BLOCK_KA];
                        int kb = blk[BLOCK_KB];
                        int na = blk[BLOCK_NA];
                        int nb = blk[BLOCK_NB];
                        int src_offset = blk[BLOCK_OFFSET];
                        int src_key = ka * nkpts + kb;

                        contract_ab_sparse_struct(ci0, ci1,
                                                  src_key, src_offset,
                                                  ab_group_tab,
                                                  ab_group_offsets,
                                                  ab_src_addr,
                                                  ab_dst_addr,
                                                  ab_coef);

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

        free(ab_coef);
        free(block_offset);
        free(block_na);
        free(block_nb);
}
