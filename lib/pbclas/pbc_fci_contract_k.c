/*
 * k-point FCI contraction helpers.
 *
 * This file implements the low-level complex 2e contraction for the
 * momentum-sector k-FCI representation used by direct_spin1_kfci.py.  The
 * Python layer owns the k-sector link and pair tables for now.
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

#define LINK_CRE    0
#define LINK_DES    1
#define LINK_TARGET 2
#define LINK_SIGN   3
#define LINK_K_CRE  5
#define LINK_K_DES  6
#define LINK_DK     7
#define NLINK_FIELDS 8

typedef struct {
        int dst_offset;
        int entry0;
        int entry1;
} ABGroup;

typedef struct {
        int src_addr;
        int dst_addr;
        double complex coef;
} ABEntry;

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

static int make_ab_sparse_tables(double complex *eri,
                                 int nkpts, int ncas,
                                 int *block_offset,
                                 int *block_nb,
                                 int *ab_pairs,
                                 int *ab_offsets,
                                 ABGroup **p_groups,
                                 int **p_group_offsets,
                                 ABEntry **p_entries)
{
        int table_size = nkpts * nkpts;
        int ngroups = 0;
        int nentries = 0;
        int *dst_counts = calloc((size_t)table_size, sizeof(int));

        if (dst_counts == NULL) {
                return 1;
        }

        for (int src_key = 0; src_key < table_size; src_key++) {
                if (block_offset[src_key] < 0) {
                        continue;
                }

                for (int dst_key = 0; dst_key < table_size; dst_key++) {
                        dst_counts[dst_key] = 0;
                }

                int ab0 = ab_offsets[src_key];
                int ab1 = ab_offsets[src_key + 1];
                for (int i = ab0; i < ab1; i++) {
                        int *row = ab_pairs + i * NAB_FIELDS;
                        int dst_key = row[AB_KA1] * nkpts + row[AB_KB1];

                        if (block_offset[dst_key] < 0) {
                                continue;
                        }
                        if (dst_counts[dst_key] == 0) {
                                ngroups++;
                        }
                        dst_counts[dst_key]++;
                        nentries++;
                }
        }

        ABGroup *groups = malloc(sizeof(ABGroup)
                                 * (size_t)(ngroups > 0 ? ngroups : 1));
        int *group_offsets = malloc(sizeof(int) * (size_t)(table_size + 1));
        ABEntry *entries = malloc(sizeof(ABEntry)
                                  * (size_t)(nentries > 0 ? nentries : 1));

        if (groups == NULL || group_offsets == NULL || entries == NULL) {
                free(groups);
                free(group_offsets);
                free(entries);
                free(dst_counts);
                return 1;
        }

        int gpos = 0;
        int epos = 0;
        for (int src_key = 0; src_key < table_size; src_key++) {
                group_offsets[src_key] = gpos;

                for (int dst_key = 0; dst_key < table_size; dst_key++) {
                        dst_counts[dst_key] = 0;
                }

                if (block_offset[src_key] < 0) {
                        continue;
                }

                int ab0 = ab_offsets[src_key];
                int ab1 = ab_offsets[src_key + 1];
                for (int i = ab0; i < ab1; i++) {
                        int *row = ab_pairs + i * NAB_FIELDS;
                        int dst_key = row[AB_KA1] * nkpts + row[AB_KB1];

                        if (block_offset[dst_key] >= 0) {
                                dst_counts[dst_key]++;
                        }
                }

                for (int dst_key = 0; dst_key < table_size; dst_key++) {
                        int count = dst_counts[dst_key];
                        if (count > 0) {
                                groups[gpos].dst_offset =
                                        block_offset[dst_key];
                                groups[gpos].entry0 = epos;
                                groups[gpos].entry1 = epos + count;
                                dst_counts[dst_key] = epos;
                                epos += count;
                                gpos++;
                        }
                }

                int src_nb = block_nb[src_key];
                for (int i = ab0; i < ab1; i++) {
                        int *row = ab_pairs + i * NAB_FIELDS;
                        int dst_key = row[AB_KA1] * nkpts + row[AB_KB1];
                        int dst_offset = block_offset[dst_key];

                        if (dst_offset < 0) {
                                continue;
                        }

                        int dst_nb = block_nb[dst_key];
                        int pos = dst_counts[dst_key]++;
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

                        entries[pos].src_addr = a0 * src_nb + b0;
                        entries[pos].dst_addr = a1 * dst_nb + b1;
                        entries[pos].coef = (val_ab + val_ba) * sign;
                }
        }
        group_offsets[table_size] = gpos;

        *p_groups = groups;
        *p_group_offsets = group_offsets;
        *p_entries = entries;

        free(dst_counts);
        return 0;
}

static void contract_ab_sparse(double complex *ci0,
                               double complex *ci1,
                               int src_key,
                               int src_offset,
                               ABGroup *ab_groups,
                               int *ab_group_offsets,
                               ABEntry *ab_entries)
{
        int group0 = ab_group_offsets[src_key];
        int group1 = ab_group_offsets[src_key + 1];

        for (int ig = group0; ig < group1; ig++) {
                ABGroup *group = ab_groups + ig;
                int dst_offset = group->dst_offset;

                for (int i = group->entry0; i < group->entry1; i++) {
                        ABEntry *entry = ab_entries + i;
                        ci1[dst_offset + entry->dst_addr] +=
                                entry->coef *
                                ci0[src_offset + entry->src_addr];
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

static void contract_aa_zgemm(double complex *eri,
                              double complex *ci0,
                              double complex *ci1,
                              double complex *amat,
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

                zset0(amat, (size_t)dst_na * na);

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
        }
}

static void contract_bb_zgemm(double complex *eri,
                              double complex *ci0,
                              double complex *ci1,
                              double complex *bmat,
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

                zset0(bmat, (size_t)nb * dst_nb);

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
        ABGroup *ab_groups = NULL;
        int *ab_group_offsets = NULL;
        ABEntry *ab_entries = NULL;

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

        int use_ab_sparse = (
                make_ab_sparse_tables(eri, nkpts, ncas,
                                      block_offset, block_nb,
                                      ab_pairs, ab_offsets,
                                      &ab_groups, &ab_group_offsets,
                                      &ab_entries) == 0);

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
               ab_pairs, ab_offsets, aa_pairs, aa_offsets, \
               bb_pairs, bb_offsets, ndet_size, aa_work_size, bb_work_size, \
               status, use_ab_sparse, ab_groups, ab_group_offsets, ab_entries)
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

                        if (use_ab_sparse) {
                                contract_ab_sparse(ci0, ci1buf,
                                                   src_key, src_offset,
                                                   ab_groups,
                                                   ab_group_offsets,
                                                   ab_entries);
                        } else {
                                contract_ab_scalar(eri, ci0, ci1buf,
                                                   nkpts, ncas,
                                                   ka, kb, nb, src_offset,
                                                   block_offset, block_nb,
                                                   ab_pairs, ab_offsets);
                        }

                        contract_aa_zgemm(eri, ci0, ci1buf, amat,
                                          nkpts, ncas,
                                          ka, kb, na, nb, src_offset,
                                          block_offset, block_na, block_nb,
                                          aa_pairs, aa_offsets);

                        contract_bb_zgemm(eri, ci0, ci1buf, bmat,
                                          nkpts, ncas,
                                          ka, kb, na, nb, src_offset,
                                          block_offset, block_nb,
                                          bb_pairs, bb_offsets);
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
                        free(block_offset);
                        free(block_na);
                        free(block_nb);
                        free(ab_groups);
                        free(ab_group_offsets);
                        free(ab_entries);
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

                        if (use_ab_sparse) {
                                contract_ab_sparse(ci0, ci1,
                                                   src_key, src_offset,
                                                   ab_groups,
                                                   ab_group_offsets,
                                                   ab_entries);
                        } else {
                                contract_ab_scalar(eri, ci0, ci1,
                                                   nkpts, ncas,
                                                   ka, kb, nb, src_offset,
                                                   block_offset, block_nb,
                                                   ab_pairs, ab_offsets);
                        }

                        contract_aa_zgemm(eri, ci0, ci1, amat,
                                          nkpts, ncas,
                                          ka, kb, na, nb, src_offset,
                                          block_offset, block_na, block_nb,
                                          aa_pairs, aa_offsets);

                        contract_bb_zgemm(eri, ci0, ci1, bmat,
                                          nkpts, ncas,
                                          ka, kb, na, nb, src_offset,
                                          block_offset, block_nb,
                                          bb_pairs, bb_offsets);
                }

                free(amat);
                free(bmat);
        }

        free(block_offset);
        free(block_na);
        free(block_nb);
        free(ab_groups);
        free(ab_group_offsets);
        free(ab_entries);
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
               ab_coef, aa_pairs, aa_offsets, bb_pairs, bb_offsets, \
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

                        contract_aa_zgemm(eri, ci0, ci1buf, amat,
                                          nkpts, ncas,
                                          ka, kb, na, nb, src_offset,
                                          block_offset, block_na, block_nb,
                                          aa_pairs, aa_offsets);

                        contract_bb_zgemm(eri, ci0, ci1buf, bmat,
                                          nkpts, ncas,
                                          ka, kb, na, nb, src_offset,
                                          block_offset, block_nb,
                                          bb_pairs, bb_offsets);
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

                        contract_aa_zgemm(eri, ci0, ci1, amat,
                                          nkpts, ncas,
                                          ka, kb, na, nb, src_offset,
                                          block_offset, block_na, block_nb,
                                          aa_pairs, aa_offsets);

                        contract_bb_zgemm(eri, ci0, ci1, bmat,
                                          nkpts, ncas,
                                          ka, kb, na, nb, src_offset,
                                          block_offset, block_nb,
                                          bb_pairs, bb_offsets);
                }

                free(amat);
                free(bmat);
        }

        free(ab_coef);
        free(block_offset);
        free(block_na);
        free(block_nb);
}
