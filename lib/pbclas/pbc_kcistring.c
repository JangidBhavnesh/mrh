#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <omp.h>
#include <complex.h>
#include "config.h"
#include "np_helper/np_helper.h"
#include "fci.h"

// Author: Bhavnesh Jangid

/*
For the implementation of the k-FCI, I would need to update the construction of
link_index to include the momentum information. The link_index layout would be:
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
Memory management
-----------------
The following arrays are allocated dynamically in this file:

  - counts: temporary workspace, freed by the function that allocates it.
  - offsets and indices: stored in LinkOrderK and freed by free_link_order_k.
  - block_offset, block_na, and block_nb: temporary block lookup tables,
    freed before the allocating function returns.
  - ab_counts, aa_counts, and bb_counts: returned to the caller by
    count_contract_map_k on success and freed by that caller; freed
    locally on failure.
  - group_index and cursor arrays: temporary contraction-group workspace,
    freed before the allocating function returns.

The stack arrays in FCIlinkstr_index_k are released automatically on return.
Input and output buffers supplied by the caller are not allocated or freed here.
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

static inline long long eri_index_k(int kp, int kq, int kr,
                                    int p, int q, int r, int s,
                                    int nkpts, int ncas)
{
    long long idx = kp;
    idx = idx * nkpts + kq;
    idx = idx * nkpts + kr;
    idx = idx * ncas + p;
    idx = idx * ncas + q;
    idx = idx * ncas + r;
    idx = idx * ncas + s;
    return idx;
}

/* Column indices for one momentum-labelled link record. */
#define KLINK_CRE      0
#define KLINK_DES      1
#define KLINK_TARGET   2
#define KLINK_SIGN     3
#define KLINK_K0       4
#define KLINK_K_CRE    5
#define KLINK_K_DES    6
#define KLINK_DK       7
#define KLINK_FIELDS   8

/* Column indices for one (alpha momentum, beta momentum) CI block. */
#define BLOCK_KA       0
#define BLOCK_KB       1
#define BLOCK_NA       2
#define BLOCK_NB       3
#define BLOCK_OFFSET   4
#define BLOCK_SIZE     5

typedef struct {
    int *offsets;
    int *indices;
    int nlinks_total;
} LinkOrderK;

/**
 * Build lookup tables for the momentum-sector CI blocks.
 * @param nkpts Number of k-points.
 * @param nblocks Number of CI blocks.
 * @param blocks Input block records.
 * @param block_offset Output offset for each momentum pair.
 * @param block_na Output number of alpha strings per momentum pair.
 * @param block_nb Output number of beta strings per momentum pair.
 * @return 0 on success; 1 if a block momentum is invalid.
 */
static int make_block_tables_k(int nkpts, int nblocks, int *blocks,
                               int *block_offset, int *block_na,
                               int *block_nb)
{
    int table_size = nkpts * nkpts;
    for (int i = 0; i < table_size; i++) {
        block_offset[i] = -1;
        block_na[i] = 0;
        block_nb[i] = 0;
    }

    for (int iblk = 0; iblk < nblocks; iblk++) {
        int *blk = blocks + iblk * 6;
        int key = blk[BLOCK_KA] * nkpts + blk[BLOCK_KB];
        if (key < 0 || key >= table_size) {
            return 1;
        }
        block_offset[key] = blk[BLOCK_OFFSET];
        block_na[key] = blk[BLOCK_NA];
        block_nb[key] = blk[BLOCK_NB];
    }
    return 0;
}

/**
 * Group link indices by starting momentum and momentum transfer.
 * @param link_index Momentum-labelled link table.
 * @param nstr Number of strings.
 * @param nlink Number of links per string.
 * @param nkpts Number of k-points.
 * @param order Output grouped-link lookup; free with free_link_order_k.
 * @return 0 on success; 1 on allocation failure.
 */
static int make_link_order_k(int *link_index, int nstr, int nlink,
                             int nkpts, LinkOrderK *order)
{
    int table_size = nkpts * nkpts;
    int nlinks_total = nstr * nlink;
    /* Allocated here: counts is local; offsets and indices enter order. */
    int *counts = calloc((size_t)table_size, sizeof(int));
    int *offsets = calloc((size_t)table_size + 1, sizeof(int));
    int *indices = NULL;

    order->offsets = NULL;
    order->indices = NULL;
    order->nlinks_total = 0;

    if (counts == NULL || offsets == NULL) {
        /* Deallocate any successfully allocated local arrays. */
        free(counts);
        free(offsets);
        return 1;
    }

    if (nlinks_total > 0) {
        indices = malloc(sizeof(int) * (size_t)nlinks_total);
        if (indices == NULL) {
            /* Deallocate arrays allocated by this function. */
            free(counts);
            free(offsets);
            return 1;
        }
    }

    for (int ilink = 0; ilink < nlinks_total; ilink++) {
        int *row = link_index + ilink * KLINK_FIELDS;
        int k0 = mod_pos(row[KLINK_K0], nkpts);
        int dk = mod_pos(row[KLINK_DK], nkpts);
        counts[k0 * nkpts + dk]++;
    }

    offsets[0] = 0;
    for (int i = 0; i < table_size; i++) {
        offsets[i + 1] = offsets[i] + counts[i];
        counts[i] = offsets[i];
    }

    for (int ilink = 0; ilink < nlinks_total; ilink++) {
        int *row = link_index + ilink * KLINK_FIELDS;
        int k0 = mod_pos(row[KLINK_K0], nkpts);
        int dk = mod_pos(row[KLINK_DK], nkpts);
        int key = k0 * nkpts + dk;
        indices[counts[key]++] = ilink;
    }

    /* Deallocate local workspace; order takes ownership below. */
    free(counts);
    order->offsets = offsets;
    order->indices = indices;
    order->nlinks_total = nlinks_total;
    return 0;
}

static void free_link_order_k(LinkOrderK *order)
{
    /* Deallocate arrays owned by order. */
    free(order->offsets);
    free(order->indices);
    order->offsets = NULL;
    order->indices = NULL;
    order->nlinks_total = 0;
}

/**
 * Count alpha-beta contraction entries between CI momentum blocks.
 * @param linka Alpha link table.
 * @param nlinka Number of links per alpha string.
 * @param linkb Beta link table.
 * @param order_a Alpha links grouped by momentum.
 * @param order_b Beta links grouped by momentum.
 * @param block_offset CI block offsets.
 * @param block_nb Number of beta strings in each CI block.
 * @param nkpts Number of k-points.
 * @param counts Output entry counts for each source/destination block pair.
 * @return Nothing.
 */
static void count_ab_struct_k(int *linka, int nlinka,
                              int *linkb,
                              LinkOrderK *order_a, LinkOrderK *order_b,
                              int *block_offset, int *block_nb,
                              int nkpts, long long *counts)
{
    int table_size = nkpts * nkpts;

#pragma omp parallel for schedule(dynamic) default(none) \
    shared(linka, nlinka, linkb, order_a, order_b, block_offset, block_nb, \
           nkpts, counts, table_size)
    for (int src_key = 0; src_key < table_size; src_key++) {
        int ka = src_key / nkpts;
        int kb = src_key % nkpts;
        int src_nb = block_nb[src_key];

        if (block_offset[src_key] < 0) {
            continue;
        }

        for (int dka = 0; dka < nkpts; dka++) {
            int akey = ka * nkpts + dka;
            int bkey = kb * nkpts + mod_pos(-dka, nkpts);
            int a0 = order_a->offsets[akey];
            int a1 = order_a->offsets[akey + 1];
            int b0 = order_b->offsets[bkey];
            int b1 = order_b->offsets[bkey + 1];

            for (int ia = a0; ia < a1; ia++) {
                int aid = order_a->indices[ia];
                int *la = linka + aid * KLINK_FIELDS;
                int astr1 = la[KLINK_TARGET];
                int ka1 = mod_pos(la[KLINK_K0] + la[KLINK_DK], nkpts);

                if (astr1 < 0) {
                    continue;
                }

                for (int ib = b0; ib < b1; ib++) {
                    int bid = order_b->indices[ib];
                    int *lb = linkb + bid * KLINK_FIELDS;
                    int bstr1 = lb[KLINK_TARGET];
                    int kb1 = mod_pos(lb[KLINK_K0] + lb[KLINK_DK], nkpts);
                    int dst_key = ka1 * nkpts + kb1;

                    if (bstr1 < 0 || block_offset[dst_key] < 0) {
                        continue;
                    }
                    if (src_nb == 0 || block_nb[dst_key] == 0) {
                        continue;
                    }
                    counts[src_key * table_size + dst_key]++;
                }
            }
        }
    }
}

/**
 * Count same-spin contraction entries between CI momentum blocks.
 * @param link_index Link table for the selected spin.
 * @param nstr Number of strings.
 * @param nlink Number of links per string.
 * @param order Links grouped by momentum.
 * @param block_offset CI block offsets.
 * @param block_na Number of alpha strings in each CI block.
 * @param block_nb Number of beta strings in each CI block.
 * @param nkpts Number of k-points.
 * @param spin Spin index: 0 for alpha, 1 for beta.
 * @param counts Output entry counts for each source block/destination momentum.
 * @return Nothing.
 */
static void count_same_spin_struct_k(int *link_index, int nstr, int nlink,
                                     LinkOrderK *order,
                                     int *block_offset, int *block_na,
                                     int *block_nb,
                                     int nkpts, int spin,
                                     long long *counts)
{
    int table_size = nkpts * nkpts;

    if (nlink == 0) {
        return;
    }

#pragma omp parallel for schedule(dynamic) default(none) \
    shared(link_index, nstr, nlink, order, block_offset, block_na, block_nb, \
           nkpts, spin, counts, table_size)
    for (int src_key = 0; src_key < table_size; src_key++) {
        int ka = src_key / nkpts;
        int kb = src_key % nkpts;
        int src_k = (spin == 0) ? ka : kb;
        int src_dim_other = (spin == 0) ? block_nb[src_key] : block_na[src_key];

        if (block_offset[src_key] < 0) {
            continue;
        }

        int link0 = order->offsets[src_k * nkpts];
        int link1 = order->offsets[(src_k + 1) * nkpts];

        for (int i1 = link0; i1 < link1; i1++) {
            int id1 = order->indices[i1];
            int *l1 = link_index + id1 * KLINK_FIELDS;
            int str_mid = l1[KLINK_TARGET];
            int dk1 = mod_pos(l1[KLINK_DK], nkpts);

            if (str_mid < 0 || str_mid >= nstr) {
                continue;
            }

            for (int j2 = 0; j2 < nlink; j2++) {
                int *l2 = link_index + (str_mid * nlink + j2) * KLINK_FIELDS;
                int dk2 = mod_pos(l2[KLINK_DK], nkpts);
                int dst_k = mod_pos(l2[KLINK_K0] + l2[KLINK_DK], nkpts);
                int dst_key;

                if ((dk1 + dk2) % nkpts != 0) {
                    continue;
                }
                if (mod_pos(l2[KLINK_K0], nkpts) !=
                    mod_pos(l1[KLINK_K0] + l1[KLINK_DK], nkpts)) {
                    continue;
                }

                if (spin == 0) {
                    dst_key = dst_k * nkpts + kb;
                    if (block_offset[dst_key] < 0 ||
                        block_na[dst_key] == 0 ||
                        block_nb[dst_key] != src_dim_other) {
                        continue;
                    }
                    counts[src_key * nkpts + dst_k]++;
                } else {
                    dst_key = ka * nkpts + dst_k;
                    if (block_offset[dst_key] < 0 ||
                        block_nb[dst_key] == 0 ||
                        block_na[dst_key] != src_dim_other) {
                        continue;
                    }
                    counts[src_key * nkpts + dst_k]++;
                }
            }
        }
    }
}

/**
 * Sum an array of contraction-entry counts.
 * @param counts Input counts.
 * @param n Number of entries.
 * @return Sum of all entries.
 */
static long long sum_counts_k(long long *counts, int n)
{
    long long total = 0;
    for (int i = 0; i < n; i++) {
        total += counts[i];
    }
    return total;
}

/**
 * Count the number of non-zero elements in the counts array.
 * @param counts The array of counts.
 * @param n The number of elements in the array.
 * @return The number of non-zero elements.
 */
static long long count_groups_k(long long *counts, int n)
{
    long long total = 0;
    for (int i = 0; i < n; i++) {
        if (counts[i] > 0) {
            total++;
        }
    }
    return total;
}

/**
 * Count the groups and entries in the AB, AA, and BB contraction maps.
 * @param linka Alpha link table.
 * @param nstra Number of alpha strings.
 * @param nlinka Number of links per alpha string.
 * @param linkb Beta link table.
 * @param nstrb Number of beta strings.
 * @param nlinkb Number of links per beta string.
 * @param blocks CI block records.
 * @param nblocks Number of CI blocks.
 * @param nkpts Number of k-points.
 * @param dims Output group and entry counts for AB, AA, and BB contractions.
 * @param p_ab_counts Output AB counts; caller owns the allocated array.
 * @param p_aa_counts Output AA counts; caller owns the allocated array.
 * @param p_bb_counts Output BB counts; caller owns the allocated array.
 * @return 0 on success; 1 on allocation or block-table failure.
 */
static int count_contract_map_k(int *linka, int nstra, int nlinka,
                                int *linkb, int nstrb, int nlinkb,
                                int *blocks, int nblocks, int nkpts,
                                long long *dims,
                                long long **p_ab_counts,
                                long long **p_aa_counts,
                                long long **p_bb_counts)
{
    int table_size = nkpts * nkpts;
    /* Temporary arrays allocated here and released at done. */
    int *block_offset = malloc(sizeof(int) * (size_t)table_size);
    int *block_na = malloc(sizeof(int) * (size_t)table_size);
    int *block_nb = malloc(sizeof(int) * (size_t)table_size);
    long long *ab_counts = calloc((size_t)table_size * table_size,
                                  sizeof(long long));
    long long *aa_counts = calloc((size_t)table_size * nkpts,
                                  sizeof(long long));
    long long *bb_counts = calloc((size_t)table_size * nkpts,
                                  sizeof(long long));
    LinkOrderK order_a;
    LinkOrderK order_b;
    int status = 1;

    order_a.offsets = NULL;
    order_a.indices = NULL;
    order_b.offsets = NULL;
    order_b.indices = NULL;

    if (block_offset == NULL || block_na == NULL || block_nb == NULL ||
        ab_counts == NULL || aa_counts == NULL || bb_counts == NULL) {
        goto done;
    }
    if (make_block_tables_k(nkpts, nblocks, blocks,
                            block_offset, block_na, block_nb) != 0) {
        goto done;
    }
    if (make_link_order_k(linka, nstra, nlinka, nkpts, &order_a) != 0) {
        goto done;
    }
    if (make_link_order_k(linkb, nstrb, nlinkb, nkpts, &order_b) != 0) {
        goto done;
    }

    count_ab_struct_k(linka, nlinka, linkb, &order_a, &order_b,
                      block_offset, block_nb, nkpts, ab_counts);
    count_same_spin_struct_k(linka, nstra, nlinka, &order_a,
                             block_offset, block_na, block_nb,
                             nkpts, 0, aa_counts);
    count_same_spin_struct_k(linkb, nstrb, nlinkb, &order_b,
                             block_offset, block_na, block_nb,
                             nkpts, 1, bb_counts);

    dims[0] = count_groups_k(ab_counts, table_size * table_size);
    dims[1] = sum_counts_k(ab_counts, table_size * table_size);
    dims[2] = count_groups_k(aa_counts, table_size * nkpts);
    dims[3] = sum_counts_k(aa_counts, table_size * nkpts);
    dims[4] = count_groups_k(bb_counts, table_size * nkpts);
    dims[5] = sum_counts_k(bb_counts, table_size * nkpts);
    status = 0;

done:
    /* Deallocate local lookup arrays. */
    free(block_offset);
    free(block_na);
    free(block_nb);
    free_link_order_k(&order_a);
    free_link_order_k(&order_b);

    if (status == 0) {
        /* Transfer count-array ownership to the caller. */
        *p_ab_counts = ab_counts;
        *p_aa_counts = aa_counts;
        *p_bb_counts = bb_counts;
    } else {
        /* Deallocate count arrays when ownership cannot be transferred. */
        free(ab_counts);
        free(aa_counts);
        free(bb_counts);
    }
    return status;
}

/**
 * Build alpha-beta contraction groups from entry counts.
 * @param counts Entry counts for each source/destination block pair.
 * @param block_offset CI block offsets.
 * @param nkpts Number of k-points.
 * @param group_tab Output group records.
 * @param group_offsets Output group offsets for each source block.
 * @param group_index Output block-pair to group lookup.
 * @param cursor Output insertion cursor for each group.
 * @return 0.
 */
static int setup_ab_groups_k(long long *counts, int *block_offset,
                             int nkpts, int *group_tab, int *group_offsets,
                             int *group_index, int *cursor)
{
    int table_size = nkpts * nkpts;
    int gpos = 0;
    int epos = 0;

    for (int i = 0; i < table_size * table_size; i++) {
        group_index[i] = -1;
    }

    for (int src_key = 0; src_key < table_size; src_key++) {
        group_offsets[src_key] = gpos;
        for (int dst_key = 0; dst_key < table_size; dst_key++) {
            long long count = counts[src_key * table_size + dst_key];
            if (count <= 0) {
                continue;
            }
            group_index[src_key * table_size + dst_key] = gpos;
            cursor[gpos] = epos;
            group_tab[gpos * 3 + 0] = block_offset[dst_key];
            group_tab[gpos * 3 + 1] = epos;
            group_tab[gpos * 3 + 2] = epos + (int)count;
            epos += (int)count;
            gpos++;
        }
    }
    group_offsets[table_size] = gpos;
    return 0;
}

/**
 * Build same-spin contraction groups from entry counts.
 * @param counts Entry counts for each source block/destination momentum.
 * @param block_offset CI block offsets.
 * @param block_na Number of alpha strings in each CI block.
 * @param block_nb Number of beta strings in each CI block.
 * @param nkpts Number of k-points.
 * @param spin Spin index: 0 for alpha, 1 for beta.
 * @param group_tab Output group records.
 * @param group_offsets Output group offsets for each source block.
 * @param group_index Output block/momentum to group lookup.
 * @param cursor Output insertion cursor for each group.
 * @return 0.
 */
static int setup_same_spin_groups_k(long long *counts, int *block_offset,
                                    int *block_na, int *block_nb,
                                    int nkpts, int spin,
                                    int *group_tab, int *group_offsets,
                                    int *group_index, int *cursor)
{
    int table_size = nkpts * nkpts;
    int gpos = 0;
    int epos = 0;

    for (int i = 0; i < table_size * nkpts; i++) {
        group_index[i] = -1;
    }

    for (int src_key = 0; src_key < table_size; src_key++) {
        int ka = src_key / nkpts;
        int kb = src_key % nkpts;
        group_offsets[src_key] = gpos;

        for (int dst_k = 0; dst_k < nkpts; dst_k++) {
            long long count = counts[src_key * nkpts + dst_k];
            int dst_key = (spin == 0) ? dst_k * nkpts + kb
                                      : ka * nkpts + dst_k;
            int dst_dim = (spin == 0) ? block_na[dst_key]
                                      : block_nb[dst_key];
            if (count <= 0) {
                continue;
            }
            group_index[src_key * nkpts + dst_k] = gpos;
            cursor[gpos] = epos;
            group_tab[gpos * 4 + 0] = block_offset[dst_key];
            group_tab[gpos * 4 + 1] = dst_dim;
            group_tab[gpos * 4 + 2] = epos;
            group_tab[gpos * 4 + 3] = epos + (int)count;
            epos += (int)count;
            gpos++;
        }
    }
    group_offsets[table_size] = gpos;
    return 0;
}

static void fill_ab_struct_k(int *linka, int nstra, int nlinka,
                             int *linkb, int nstrb, int nlinkb,
                             int *str2tot_a, int *str2tot_b,
                             LinkOrderK *order_a, LinkOrderK *order_b,
                             int *block_offset, int *block_nb,
                             int nkpts, int ncas, int *group_index,
                             int *cursor, int *src_addr, int *dst_addr,
                             int *sign, long long *eri_idx_ab,
                             long long *eri_idx_ba)
{
    int table_size = nkpts * nkpts;

#pragma omp parallel for schedule(dynamic) default(none) \
    shared(linka, nstra, nlinka, linkb, nstrb, nlinkb, str2tot_a, str2tot_b, \
           order_a, order_b, block_offset, block_nb, nkpts, ncas, \
           group_index, cursor, src_addr, dst_addr, sign, eri_idx_ab, \
           eri_idx_ba, table_size)
    for (int src_key = 0; src_key < table_size; src_key++) {
        int ka = src_key / nkpts;
        int kb = src_key % nkpts;
        int src_nb = block_nb[src_key];

        if (block_offset[src_key] < 0) {
            continue;
        }

        for (int dka = 0; dka < nkpts; dka++) {
            int akey = ka * nkpts + dka;
            int bkey = kb * nkpts + mod_pos(-dka, nkpts);
            int a0 = order_a->offsets[akey];
            int a1 = order_a->offsets[akey + 1];
            int b0 = order_b->offsets[bkey];
            int b1 = order_b->offsets[bkey + 1];

            for (int ia = a0; ia < a1; ia++) {
                int aid = order_a->indices[ia];
                int astr0 = aid / nlinka;
                int *la = linka + aid * KLINK_FIELDS;
                int astr1 = la[KLINK_TARGET];
                int ka1 = mod_pos(la[KLINK_K0] + la[KLINK_DK], nkpts);
                int aloc0 = str2tot_a[ka * nstra + astr0];
                int aloc1;

                if (astr1 < 0) {
                    continue;
                }
                aloc1 = str2tot_a[ka1 * nstra + astr1];
                if (aloc0 < 0 || aloc1 < 0) {
                    continue;
                }

                for (int ib = b0; ib < b1; ib++) {
                    int bid = order_b->indices[ib];
                    int bstr0 = bid / nlinkb;
                    int *lb = linkb + bid * KLINK_FIELDS;
                    int bstr1 = lb[KLINK_TARGET];
                    int kb1 = mod_pos(lb[KLINK_K0] + lb[KLINK_DK], nkpts);
                    int dst_key = ka1 * nkpts + kb1;
                    int group = group_index[src_key * table_size + dst_key];
                    int bloc0;
                    int bloc1;
                    int pos;

                    if (bstr1 < 0 || group < 0) {
                        continue;
                    }
                    bloc0 = str2tot_b[kb * nstrb + bstr0];
                    bloc1 = str2tot_b[kb1 * nstrb + bstr1];
                    if (bloc0 < 0 || bloc1 < 0) {
                        continue;
                    }

                    pos = cursor[group]++;
                    src_addr[pos] = aloc0 * src_nb + bloc0;
                    dst_addr[pos] = aloc1 * block_nb[dst_key] + bloc1;
                    sign[pos] = la[KLINK_SIGN] * lb[KLINK_SIGN];
                    eri_idx_ab[pos] = eri_index_k(
                            la[KLINK_K_CRE], la[KLINK_K_DES],
                            lb[KLINK_K_CRE],
                            la[KLINK_CRE] % ncas, la[KLINK_DES] % ncas,
                            lb[KLINK_CRE] % ncas, lb[KLINK_DES] % ncas,
                            nkpts, ncas);
                    eri_idx_ba[pos] = eri_index_k(
                            lb[KLINK_K_CRE], lb[KLINK_K_DES],
                            la[KLINK_K_CRE],
                            lb[KLINK_CRE] % ncas, lb[KLINK_DES] % ncas,
                            la[KLINK_CRE] % ncas, la[KLINK_DES] % ncas,
                            nkpts, ncas);
                }
            }
        }
    }
}

static void fill_same_spin_struct_k(int *link_index, int nstr, int nlink,
                                    int *str2tot, LinkOrderK *order,
                                    int *block_offset, int *block_na,
                                    int *block_nb,
                                    int nkpts, int ncas, int spin,
                                    int *group_index, int *cursor,
                                    int *src_addr, int *dst_addr,
                                    int *sign, long long *eri_idx)
{
    int table_size = nkpts * nkpts;

    if (nlink == 0) {
        return;
    }

#pragma omp parallel for schedule(dynamic) default(none) \
    shared(link_index, nstr, nlink, str2tot, order, block_offset, block_na, \
           block_nb, nkpts, ncas, spin, group_index, cursor, src_addr, \
           dst_addr, sign, eri_idx, table_size)
    for (int src_key = 0; src_key < table_size; src_key++) {
        int ka = src_key / nkpts;
        int kb = src_key % nkpts;
        int src_k = (spin == 0) ? ka : kb;

        if (block_offset[src_key] < 0) {
            continue;
        }

        int link0 = order->offsets[src_k * nkpts];
        int link1 = order->offsets[(src_k + 1) * nkpts];

        for (int i1 = link0; i1 < link1; i1++) {
            int id1 = order->indices[i1];
            int str0 = id1 / nlink;
            int *l1 = link_index + id1 * KLINK_FIELDS;
            int str_mid = l1[KLINK_TARGET];
            int dk1 = mod_pos(l1[KLINK_DK], nkpts);
            int loc0 = str2tot[src_k * nstr + str0];

            if (str_mid < 0 || str_mid >= nstr || loc0 < 0) {
                continue;
            }

            for (int j2 = 0; j2 < nlink; j2++) {
                int *l2 = link_index + (str_mid * nlink + j2) * KLINK_FIELDS;
                int dk2 = mod_pos(l2[KLINK_DK], nkpts);
                int dst_k = mod_pos(l2[KLINK_K0] + l2[KLINK_DK], nkpts);
                int group;
                int loc1;
                int pos;

                if ((dk1 + dk2) % nkpts != 0) {
                    continue;
                }
                if (mod_pos(l2[KLINK_K0], nkpts) !=
                    mod_pos(l1[KLINK_K0] + l1[KLINK_DK], nkpts)) {
                    continue;
                }

                group = group_index[src_key * nkpts + dst_k];
                if (group < 0 || l2[KLINK_TARGET] < 0) {
                    continue;
                }
                loc1 = str2tot[dst_k * nstr + l2[KLINK_TARGET]];
                if (loc1 < 0) {
                    continue;
                }

                pos = cursor[group]++;
                src_addr[pos] = loc0;
                dst_addr[pos] = loc1;
                sign[pos] = l1[KLINK_SIGN] * l2[KLINK_SIGN];
                eri_idx[pos] = eri_index_k(
                        l2[KLINK_K_CRE], l2[KLINK_K_DES],
                        l1[KLINK_K_CRE],
                        l2[KLINK_CRE] % ncas, l2[KLINK_DES] % ncas,
                        l1[KLINK_CRE] % ncas, l1[KLINK_DES] % ncas,
                        nkpts, ncas);
            }
        }
    }
}

/**
 * Count the groups and entries needed for the AB, AA, and BB contraction maps.
 *
 * Python uses these counts to allocate the map arrays.  The matching fill
 * function then stores each CI source address, destination address, sign, and
 * ERI index in those arrays.
 *
 * @param linka Alpha link table, shape (nstra, nlinka, KLINK_FIELDS).
 * @param nstra Total number of alpha strings.
 * @param nlinka Number of alpha excitation links per source string.
 * @param linkb Beta link table, shape (nstrb, nlinkb, KLINK_FIELDS).
 * @param nstrb Total number of beta strings.
 * @param nlinkb Number of beta excitation links per source string.
 * @param blocks Packed CI block records, shape (nblocks, 6).
 * @param nblocks Number of packed CI blocks.
 * @param nkpts Number of momentum points.
 * @param dims Output array of six int64 counts: [number of AB groups,
 *        number of AB entries, number of AA groups, number of AA entries,
 *        number of BB groups, number of BB entries].
 * @return 0 on success; 1 on allocation or block-table failure.
 */
int FCIcount_contract_map_k(int *linka, int nstra, int nlinka,
                            int *linkb, int nstrb, int nlinkb,
                            int *blocks, int nblocks, int nkpts,
                            long long *dims)
{
    long long *ab_counts = NULL;
    long long *aa_counts = NULL;
    long long *bb_counts = NULL;
    int status;

    for (int i = 0; i < 6; i++) {
        dims[i] = 0;
    }

    status = count_contract_map_k(
            linka, nstra, nlinka, linkb, nstrb, nlinkb,
            blocks, nblocks, nkpts, dims,
            &ab_counts, &aa_counts, &bb_counts);

    /* Deallocate count arrays returned by count_contract_map_k. */
    free(ab_counts);
    free(aa_counts);
    free(bb_counts);
    return status;
}

int FCIcount_same_spin_contract_map_k(int *link_index, int nstr,
                                     int nlink, int *blocks,
                                     int nblocks, int nkpts,
                                     int spin, long long *dims)
{
    int table_size = nkpts * nkpts;
    /* Temporary arrays allocated here and released at done. */
    int *block_offset = malloc(sizeof(int) * (size_t)table_size);
    int *block_na = malloc(sizeof(int) * (size_t)table_size);
    int *block_nb = malloc(sizeof(int) * (size_t)table_size);
    long long *counts = calloc((size_t)table_size * nkpts,
                               sizeof(long long));
    LinkOrderK order;
    int status = 1;

    order.offsets = NULL;
    order.indices = NULL;
    order.nlinks_total = 0;
    dims[0] = 0;
    dims[1] = 0;

    if (block_offset == NULL || block_na == NULL || block_nb == NULL ||
        counts == NULL) {
        goto done;
    }
    if (make_block_tables_k(nkpts, nblocks, blocks,
                            block_offset, block_na, block_nb) != 0) {
        goto done;
    }
    if (make_link_order_k(link_index, nstr, nlink, nkpts, &order) != 0) {
        goto done;
    }

    count_same_spin_struct_k(link_index, nstr, nlink, &order,
                             block_offset, block_na, block_nb,
                             nkpts, spin, counts);
    dims[0] = count_groups_k(counts, table_size * nkpts);
    dims[1] = sum_counts_k(counts, table_size * nkpts);
    status = 0;

done:
    /* Deallocate all local arrays, including those owned by order. */
    free(order.offsets);
    free(order.indices);
    free(block_offset);
    free(block_na);
    free(block_nb);
    free(counts);
    return status;
}

int FCIfill_same_spin_contract_map_k(int *link_index, int nstr,
                                    int nlink, int *str2tot,
                                    int *blocks, int nblocks,
                                    int nkpts, int ncas, int spin,
                                    int *group_tab,
                                    int *group_offsets,
                                    int *src_addr,
                                    int *dst_addr,
                                    int *sign,
                                    long long *eri_idx)
{
    int table_size = nkpts * nkpts;
    long long dims[2];
    /* Temporary arrays allocated here and released at done. */
    int *block_offset = malloc(sizeof(int) * (size_t)table_size);
    int *block_na = malloc(sizeof(int) * (size_t)table_size);
    int *block_nb = malloc(sizeof(int) * (size_t)table_size);
    int *group_index = malloc(sizeof(int) * (size_t)table_size * nkpts);
    int *cursor = NULL;
    long long *counts = calloc((size_t)table_size * nkpts,
                               sizeof(long long));
    LinkOrderK order;
    int status = 1;

    order.offsets = NULL;
    order.indices = NULL;
    order.nlinks_total = 0;
    dims[0] = 0;
    dims[1] = 0;

    if (block_offset == NULL || block_na == NULL || block_nb == NULL ||
        group_index == NULL || counts == NULL) {
        goto done;
    }
    if (make_block_tables_k(nkpts, nblocks, blocks,
                            block_offset, block_na, block_nb) != 0) {
        goto done;
    }
    if (make_link_order_k(link_index, nstr, nlink, nkpts, &order) != 0) {
        goto done;
    }

    count_same_spin_struct_k(link_index, nstr, nlink, &order,
                             block_offset, block_na, block_nb,
                             nkpts, spin, counts);
    dims[0] = count_groups_k(counts, table_size * nkpts);
    dims[1] = sum_counts_k(counts, table_size * nkpts);
    if (dims[1] > INT_MAX) {
        goto done;
    }

    /* Allocate the per-group insertion cursor; release it at done. */
    cursor = malloc(sizeof(int) * (size_t)(dims[0] > 0 ? dims[0] : 1));
    if (cursor == NULL) {
        goto done;
    }

    setup_same_spin_groups_k(counts, block_offset, block_na, block_nb,
                             nkpts, spin, group_tab, group_offsets,
                             group_index, cursor);
    fill_same_spin_struct_k(link_index, nstr, nlink, str2tot, &order,
                            block_offset, block_na, block_nb,
                            nkpts, ncas, spin, group_index, cursor,
                            src_addr, dst_addr, sign, eri_idx);
    status = 0;

done:
    /* Deallocate all local arrays, including those owned by order. */
    free(order.offsets);
    free(order.indices);
    free(block_offset);
    free(block_na);
    free(block_nb);
    free(group_index);
    free(cursor);
    free(counts);
    return status;
}

int FCIfill_contract_map_k(int *linka, int nstra, int nlinka,
                           int *linkb, int nstrb, int nlinkb,
                           int *str2tot_a, int *str2tot_b,
                           int *blocks, int nblocks,
                           int nkpts, int ncas,
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
    int table_size = nkpts * nkpts;
    long long dims[6];
    long long *ab_counts = NULL;
    long long *aa_counts = NULL;
    long long *bb_counts = NULL;
    /* Temporary arrays allocated here and released at done. */
    int *block_offset = malloc(sizeof(int) * (size_t)table_size);
    int *block_na = malloc(sizeof(int) * (size_t)table_size);
    int *block_nb = malloc(sizeof(int) * (size_t)table_size);
    int *ab_group_index = malloc(sizeof(int) * (size_t)table_size * table_size);
    int *aa_group_index = malloc(sizeof(int) * (size_t)table_size * nkpts);
    int *bb_group_index = malloc(sizeof(int) * (size_t)table_size * nkpts);
    int *ab_cursor = NULL;
    int *aa_cursor = NULL;
    int *bb_cursor = NULL;
    LinkOrderK order_a;
    LinkOrderK order_b;
    int status = 1;

    order_a.offsets = NULL;
    order_a.indices = NULL;
    order_b.offsets = NULL;
    order_b.indices = NULL;

    if (count_contract_map_k(
                linka, nstra, nlinka, linkb, nstrb, nlinkb,
                blocks, nblocks, nkpts, dims,
                &ab_counts, &aa_counts, &bb_counts) != 0) {
        goto done;
    }
    if (dims[1] > INT_MAX || dims[3] > INT_MAX || dims[5] > INT_MAX) {
        goto done;
    }

    /* Allocate one insertion cursor per contraction group. */
    ab_cursor = malloc(sizeof(int) * (size_t)(dims[0] > 0 ? dims[0] : 1));
    aa_cursor = malloc(sizeof(int) * (size_t)(dims[2] > 0 ? dims[2] : 1));
    bb_cursor = malloc(sizeof(int) * (size_t)(dims[4] > 0 ? dims[4] : 1));

    if (block_offset == NULL || block_na == NULL || block_nb == NULL ||
        ab_group_index == NULL || aa_group_index == NULL ||
        bb_group_index == NULL || ab_cursor == NULL ||
        aa_cursor == NULL || bb_cursor == NULL) {
        goto done;
    }
    if (make_block_tables_k(nkpts, nblocks, blocks,
                            block_offset, block_na, block_nb) != 0) {
        goto done;
    }
    if (make_link_order_k(linka, nstra, nlinka, nkpts, &order_a) != 0) {
        goto done;
    }
    if (make_link_order_k(linkb, nstrb, nlinkb, nkpts, &order_b) != 0) {
        goto done;
    }

    setup_ab_groups_k(ab_counts, block_offset, nkpts,
                      ab_group_tab, ab_group_offsets,
                      ab_group_index, ab_cursor);
    setup_same_spin_groups_k(aa_counts, block_offset, block_na, block_nb,
                             nkpts, 0, aa_group_tab, aa_group_offsets,
                             aa_group_index, aa_cursor);
    setup_same_spin_groups_k(bb_counts, block_offset, block_na, block_nb,
                             nkpts, 1, bb_group_tab, bb_group_offsets,
                             bb_group_index, bb_cursor);

    fill_ab_struct_k(linka, nstra, nlinka, linkb, nstrb, nlinkb,
                     str2tot_a, str2tot_b, &order_a, &order_b,
                     block_offset, block_nb, nkpts, ncas,
                     ab_group_index, ab_cursor, ab_src_addr, ab_dst_addr,
                     ab_sign, ab_eri_idx_ab, ab_eri_idx_ba);
    fill_same_spin_struct_k(linka, nstra, nlinka, str2tot_a, &order_a,
                            block_offset, block_na, block_nb,
                            nkpts, ncas, 0, aa_group_index, aa_cursor,
                            aa_src_addr, aa_dst_addr, aa_sign, aa_eri_idx);
    fill_same_spin_struct_k(linkb, nstrb, nlinkb, str2tot_b, &order_b,
                            block_offset, block_na, block_nb,
                            nkpts, ncas, 1, bb_group_index, bb_cursor,
                            bb_src_addr, bb_dst_addr, bb_sign, bb_eri_idx);
    status = 0;

done:
    /* Deallocate all local arrays and grouped-link storage. */
    free(block_offset);
    free(block_na);
    free(block_nb);
    free(ab_group_index);
    free(aa_group_index);
    free(bb_group_index);
    free(ab_cursor);
    free(aa_cursor);
    free(bb_cursor);
    free(ab_counts);
    free(aa_counts);
    free(bb_counts);
    free_link_order_k(&order_a);
    free_link_order_k(&order_b);
    return status;
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
    /* Variable-length stack arrays are released automatically on return. */
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
