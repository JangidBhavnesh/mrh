/*
 * k-point FCI contraction helpers.
 *
 * This file implements the low-level complex 2e contraction for the
 * momentum-sector k-FCI representation used by direct_spin1_kfci.py.  The
 * Python layer owns the k-sector link and pair tables for now.
 */

#include <complex.h>
#include <stdlib.h>
#include "vhf/fblas.h"

#define BLOCK_KA      0
#define BLOCK_KB      1
#define BLOCK_NA      2
#define BLOCK_NB      3
#define BLOCK_OFFSET  4
#define BLOCK_SIZE    5

#define AB_A0      0
#define AB_A1      1
#define AB_B0      2
#define AB_B1      3
#define AB_SIGN    4
#define AB_KA1     5
#define AB_KB1     6
#define AB_KPA     7
#define AB_KQA     8
#define AB_KRB     9
#define AB_PA      10
#define AB_QA      11
#define AB_RB      12
#define AB_SB      13
#define AB_KPB     14
#define AB_KQB     15
#define AB_KRA     16
#define AB_PB      17
#define AB_QB      18
#define AB_RA      19
#define AB_SA      20
#define NAB_FIELDS 21

#define SS_0       0
#define SS_1       1
#define SS_SIGN    2
#define SS_K1      3
#define SS_KP      4
#define SS_KQ      5
#define SS_KR      6
#define SS_P       7
#define SS_Q       8
#define SS_R       9
#define SS_S       10
#define NSS_FIELDS 11

static inline size_t eri_index(int kp, int kq, int kr,
                               int p, int q, int r, int s,
                               int nkpts, int ncas)
{
        size_t idx = (size_t)kp;
        idx = idx * (size_t)nkpts + (size_t)kq;
        idx = idx * (size_t)nkpts + (size_t)kr;
        idx = idx * (size_t)ncas + (size_t)p;
        idx = idx * (size_t)ncas + (size_t)q;
        idx = idx * (size_t)ncas + (size_t)r;
        idx = idx * (size_t)ncas + (size_t)s;
        return idx;
}

void FCIcontract_2e_k(double complex *eri,
                      double complex *ci0,
                      double complex *ci1,
                      int nkpts, int ncas,
                      int nblocks, int *blocks,
                      int *ab_pairs, int *ab_offsets,
                      int *aa_pairs, int *aa_offsets,
                      int *bb_pairs, int *bb_offsets)
{
        int ndet = 0;
        int table_size = nkpts * nkpts;
        int *block_offset = malloc(sizeof(int) * (size_t)table_size);
        int *block_na = malloc(sizeof(int) * (size_t)table_size);
        int *block_nb = malloc(sizeof(int) * (size_t)table_size);

        if (block_offset == NULL || block_na == NULL || block_nb == NULL) {
                free(block_offset);
                free(block_na);
                free(block_nb);
                return;
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

        for (int i = 0; i < ndet; i++) {
                ci1[i] = 0.0 + 0.0 * I;
        }

        for (int iblk = 0; iblk < nblocks; iblk++) {
                int *blk = blocks + iblk * 6;
                int ka = blk[BLOCK_KA];
                int kb = blk[BLOCK_KB];
                int na = blk[BLOCK_NA];
                int nb = blk[BLOCK_NB];
                int src_offset = blk[BLOCK_OFFSET];
                int key = ka * nkpts + kb;

                if (block_offset[key] < 0) {
                        continue;
                }

                int ab0 = ab_offsets[key];
                int ab1 = ab_offsets[key + 1];
                for (int i = ab0; i < ab1; i++) {
                        int *row = ab_pairs + i * NAB_FIELDS;
                        int dst_key = row[AB_KA1] * nkpts + row[AB_KB1];
                        int dst_offset = block_offset[dst_key];

                        if (dst_offset < 0) {
                                continue;
                        }

                        int dst_nb = block_nb[dst_key];
                        int a0 = row[AB_A0];
                        int a1 = row[AB_A1];
                        int b0 = row[AB_B0];
                        int b1 = row[AB_B1];
                        double sign = (double)row[AB_SIGN];

                        double complex val_ab = eri[eri_index(
                                row[AB_KPA], row[AB_KQA], row[AB_KRB],
                                row[AB_PA], row[AB_QA],
                                row[AB_RB], row[AB_SB],
                                nkpts, ncas)];
                        double complex val_ba = eri[eri_index(
                                row[AB_KPB], row[AB_KQB], row[AB_KRA],
                                row[AB_PB], row[AB_QB],
                                row[AB_RA], row[AB_SA],
                                nkpts, ncas)];

                        ci1[dst_offset + a1 * dst_nb + b1] +=
                                (val_ab + val_ba) * sign *
                                ci0[src_offset + a0 * nb + b0];
                }

                int aa0 = aa_offsets[ka];
                int aa1 = aa_offsets[ka + 1];
                for (int i = aa0; i < aa1; i++) {
                        int *row = aa_pairs + i * NSS_FIELDS;
                        int dst_key = row[SS_K1] * nkpts + kb;
                        int dst_offset = block_offset[dst_key];

                        if (dst_offset < 0) {
                                continue;
                        }

                        int dst_nb = block_nb[dst_key];
                        int a0 = row[SS_0];
                        int a1 = row[SS_1];
                        double sign = (double)row[SS_SIGN];
                        double complex val = eri[eri_index(
                                row[SS_KP], row[SS_KQ], row[SS_KR],
                                row[SS_P], row[SS_Q], row[SS_R], row[SS_S],
                                nkpts, ncas)];

                        for (int b = 0; b < nb; b++) {
                                ci1[dst_offset + a1 * dst_nb + b] +=
                                        val * sign *
                                        ci0[src_offset + a0 * nb + b];
                        }
                }

                int bb0 = bb_offsets[kb];
                int bb1 = bb_offsets[kb + 1];
                for (int i = bb0; i < bb1; i++) {
                        int *row = bb_pairs + i * NSS_FIELDS;
                        int dst_key = ka * nkpts + row[SS_K1];
                        int dst_offset = block_offset[dst_key];

                        if (dst_offset < 0) {
                                continue;
                        }

                        int dst_nb = block_nb[dst_key];
                        int b0 = row[SS_0];
                        int b1 = row[SS_1];
                        double sign = (double)row[SS_SIGN];
                        double complex val = eri[eri_index(
                                row[SS_KP], row[SS_KQ], row[SS_KR],
                                row[SS_P], row[SS_Q], row[SS_R], row[SS_S],
                                nkpts, ncas)];

                        for (int a = 0; a < na; a++) {
                                ci1[dst_offset + a * dst_nb + b1] +=
                                        val * sign *
                                        ci0[src_offset + a * nb + b0];
                        }
                }
        }

        free(block_offset);
        free(block_na);
        free(block_nb);
}

static void zset0(double complex *x, size_t n)
{
        for (size_t i = 0; i < n; i++) {
                x[i] = 0.0 + 0.0 * I;
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

static void contract_ab_scalar(double complex *eri,
                               double complex *ci0,
                               double complex *ci1,
                               int nkpts, int ncas,
                               int ka, int kb, int nb,
                               int src_offset,
                               int *block_offset,
                               int *block_nb,
                               int *ab_pairs,
                               int *ab_offsets)
{
        int key = ka * nkpts + kb;
        int ab0 = ab_offsets[key];
        int ab1 = ab_offsets[key + 1];

        for (int i = ab0; i < ab1; i++) {
                int *row = ab_pairs + i * NAB_FIELDS;
                int dst_key = row[AB_KA1] * nkpts + row[AB_KB1];
                int dst_offset = block_offset[dst_key];

                if (dst_offset < 0) {
                        continue;
                }

                int dst_nb = block_nb[dst_key];
                int a0 = row[AB_A0];
                int a1 = row[AB_A1];
                int b0 = row[AB_B0];
                int b1 = row[AB_B1];
                double sign = (double)row[AB_SIGN];

                double complex val_ab = eri[eri_index(
                        row[AB_KPA], row[AB_KQA], row[AB_KRB],
                        row[AB_PA], row[AB_QA],
                        row[AB_RB], row[AB_SB],
                        nkpts, ncas)];
                double complex val_ba = eri[eri_index(
                        row[AB_KPB], row[AB_KQB], row[AB_KRA],
                        row[AB_PB], row[AB_QB],
                        row[AB_RA], row[AB_SA],
                        nkpts, ncas)];

                ci1[dst_offset + a1 * dst_nb + b1] +=
                        (val_ab + val_ba) * sign *
                        ci0[src_offset + a0 * nb + b0];
        }
}

static void contract_aa_zgemm(double complex *eri,
                              double complex *ci0,
                              double complex *ci1,
                              int nkpts, int ncas,
                              int ka, int kb, int na, int nb,
                              int src_offset,
                              int *block_offset,
                              int *block_na,
                              int *block_nb,
                              int *aa_pairs,
                              int *aa_offsets)
{
        const char TRANS_N = 'N';
        const double complex Z1 = 1.0 + 0.0 * I;
        int aa0 = aa_offsets[ka];
        int aa1 = aa_offsets[ka + 1];

        if (aa0 == aa1 || na == 0 || nb == 0) {
                return;
        }

        for (int ka1 = 0; ka1 < nkpts; ka1++) {
                int dst_key = ka1 * nkpts + kb;
                int dst_offset = block_offset[dst_key];
                int dst_na = block_na[dst_key];
                int dst_nb = block_nb[dst_key];
                int nnz = 0;

                if (dst_offset < 0 || dst_na == 0 || dst_nb != nb) {
                        continue;
                }

                double complex *amat = calloc((size_t)dst_na * na,
                                              sizeof(double complex));
                if (amat == NULL) {
                        return;
                }

                for (int i = aa0; i < aa1; i++) {
                        int *row = aa_pairs + i * NSS_FIELDS;
                        if (row[SS_K1] != ka1) {
                                continue;
                        }

                        int a0 = row[SS_0];
                        int a1 = row[SS_1];
                        double sign = (double)row[SS_SIGN];
                        double complex val = eri[eri_index(
                                row[SS_KP], row[SS_KQ], row[SS_KR],
                                row[SS_P], row[SS_Q], row[SS_R], row[SS_S],
                                nkpts, ncas)];

                        amat[a1 * (size_t)na + a0] += val * sign;
                        nnz++;
                }

                if (nnz > 0) {
                        /*
                         * Row-major S += A*C is column-major
                         * S.T += C.T*A.T.
                         */
                        zgemm_(&TRANS_N, &TRANS_N, &nb, &dst_na, &na,
                               &Z1, ci0 + src_offset, &nb,
                               amat, &na,
                               &Z1, ci1 + dst_offset, &dst_nb);
                }
                free(amat);
        }
}

static void contract_bb_zgemm(double complex *eri,
                              double complex *ci0,
                              double complex *ci1,
                              int nkpts, int ncas,
                              int ka, int kb, int na, int nb,
                              int src_offset,
                              int *block_offset,
                              int *block_nb,
                              int *bb_pairs,
                              int *bb_offsets)
{
        const char TRANS_N = 'N';
        const double complex Z1 = 1.0 + 0.0 * I;
        int bb0 = bb_offsets[kb];
        int bb1 = bb_offsets[kb + 1];

        if (bb0 == bb1 || na == 0 || nb == 0) {
                return;
        }

        for (int kb1 = 0; kb1 < nkpts; kb1++) {
                int dst_key = ka * nkpts + kb1;
                int dst_offset = block_offset[dst_key];
                int dst_nb = block_nb[dst_key];
                int nnz = 0;

                if (dst_offset < 0 || dst_nb == 0) {
                        continue;
                }

                double complex *bmat = calloc((size_t)nb * dst_nb,
                                              sizeof(double complex));
                if (bmat == NULL) {
                        return;
                }

                for (int i = bb0; i < bb1; i++) {
                        int *row = bb_pairs + i * NSS_FIELDS;
                        if (row[SS_K1] != kb1) {
                                continue;
                        }

                        int b0 = row[SS_0];
                        int b1 = row[SS_1];
                        double sign = (double)row[SS_SIGN];
                        double complex val = eri[eri_index(
                                row[SS_KP], row[SS_KQ], row[SS_KR],
                                row[SS_P], row[SS_Q], row[SS_R], row[SS_S],
                                nkpts, ncas)];

                        bmat[b0 * (size_t)dst_nb + b1] += val * sign;
                        nnz++;
                }

                if (nnz > 0) {
                        /*
                         * Row-major S += C*B is column-major
                         * S.T += B.T*C.T.
                         */
                        zgemm_(&TRANS_N, &TRANS_N, &dst_nb, &na, &nb,
                               &Z1, bmat, &dst_nb,
                               ci0 + src_offset, &nb,
                               &Z1, ci1 + dst_offset, &dst_nb);
                }
                free(bmat);
        }
}

void FCIcontract_2e_k_zgemm(double complex *eri,
                            double complex *ci0,
                            double complex *ci1,
                            int nkpts, int ncas,
                            int nblocks, int *blocks,
                            int *ab_pairs, int *ab_offsets,
                            int *aa_pairs, int *aa_offsets,
                            int *bb_pairs, int *bb_offsets)
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

        zset0(ci1, (size_t)ndet);

        for (int iblk = 0; iblk < nblocks; iblk++) {
                int *blk = blocks + iblk * 6;
                int ka = blk[BLOCK_KA];
                int kb = blk[BLOCK_KB];
                int na = blk[BLOCK_NA];
                int nb = blk[BLOCK_NB];
                int src_offset = blk[BLOCK_OFFSET];

                contract_ab_scalar(eri, ci0, ci1, nkpts, ncas,
                                   ka, kb, nb, src_offset,
                                   block_offset, block_nb,
                                   ab_pairs, ab_offsets);

                contract_aa_zgemm(eri, ci0, ci1, nkpts, ncas,
                                  ka, kb, na, nb, src_offset,
                                  block_offset, block_na, block_nb,
                                  aa_pairs, aa_offsets);

                contract_bb_zgemm(eri, ci0, ci1, nkpts, ncas,
                                  ka, kb, na, nb, src_offset,
                                  block_offset, block_nb,
                                  bb_pairs, bb_offsets);
        }

        free(block_offset);
        free(block_na);
        free(block_nb);
}
