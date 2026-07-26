/*
 * Diagonal k-point FCI Hamiltonian helper.
 *
 * The Python layer builds the momentum-sector determinant blocks and compact
 * contraction structures.  This file only evaluates the diagonal entries from
 * those structures.
 */

#include <complex.h>
#include <omp.h>
#include <stdlib.h>

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
                int key = blk[BLOCK_KA] * nkpts + blk[BLOCK_KB];
                int offset = blk[BLOCK_OFFSET];
                int size = blk[BLOCK_SIZE];

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

static void add_one_electron_hdiag(double complex *hdiag,
                                   double complex *h1e,
                                   int nkpts, int ncas,
                                   int nblocks, int *blocks,
                                   int *linka, int nlinka,
                                   int *linkb, int nlinkb,
                                   int *stra_ids, int *stra_offsets,
                                   int *strb_ids, int *strb_offsets)
{
#pragma omp parallel for schedule(static) default(none) if(nblocks > 16) \
        shared(hdiag, h1e, nkpts, ncas, nblocks, blocks, \
               linka, nlinka, linkb, nlinkb, \
               stra_ids, stra_offsets, strb_ids, strb_offsets)
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
                                if (k_cre != k_des || dk != 0) {
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
                                if (k_cre != k_des || dk != 0) {
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
                                long long *eri_idx)
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

void FCIhdiag_k(double complex *hdiag,
                double complex *h1e,
                double complex *eri,
                int nkpts, int ncas,
                int nblocks, int *blocks,
                int *linka, int nlinka,
                int *linkb, int nlinkb,
                int *stra_ids, int *stra_offsets,
                int *strb_ids, int *strb_offsets,
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

        if (make_block_tables(nkpts, nblocks, blocks,
                              &block_offset, &block_na, &block_nb,
                              &ndet) != 0) {
                return;
        }

        zset0(hdiag, (size_t)ndet);
        if (ndet == 0) {
                free(block_offset);
                free(block_na);
                free(block_nb);
                return;
        }

        add_one_electron_hdiag(hdiag, h1e, nkpts, ncas, nblocks, blocks,
                               linka, nlinka, linkb, nlinkb,
                               stra_ids, stra_offsets,
                               strb_ids, strb_offsets);
        add_ab_hdiag(hdiag, eri, nkpts, block_offset,
                     ab_group_tab, ab_group_offsets,
                     ab_src_addr, ab_dst_addr, ab_sign,
                     ab_eri_idx_ab, ab_eri_idx_ba);
        add_same_spin_hdiag(hdiag, eri, nkpts, 1, block_offset,
                            block_na, block_nb,
                            aa_group_tab, aa_group_offsets,
                            aa_src_addr, aa_dst_addr, aa_sign, aa_eri_idx);
        add_same_spin_hdiag(hdiag, eri, nkpts, 0, block_offset,
                            block_na, block_nb,
                            bb_group_tab, bb_group_offsets,
                            bb_src_addr, bb_dst_addr, bb_sign, bb_eri_idx);

        free(block_offset);
        free(block_na);
        free(block_nb);
}
