/* Direct reduced-density matrices for packed momentum-sector k-FCI vectors. */

#include <complex.h>
#include <omp.h>
#include <stddef.h>
#include <stdlib.h>

#include "pbc_kfci_common.h"


typedef struct {
        int norb;
        int nkpts;
        int nstra;
        int nstrb;
        int *block_offset;
        int *block_na;
        int *block_nb;
        int *stra_k;
        int *stra_local;
        int *strb_k;
        int *strb_local;
} KFCIRDMLayout;


typedef struct {
        int *offsets;
        int *indices;
} KFCITargetLinkOrder;


/* Group excitation links by target string and momentum transfer. */
static int make_target_link_order(int *links, int nstr, int nlink,
                                  int nkpts, KFCITargetLinkOrder *order)
{
        int nkeys = nstr * nkpts;
        int *counts = calloc((size_t)nkeys, sizeof(int));
        int *offsets = calloc((size_t)nkeys + 1, sizeof(int));
        int *indices = NULL;

        order->offsets = NULL;
        order->indices = NULL;
        if (counts == NULL || offsets == NULL) {
                free(counts);
                free(offsets);
                return 1;
        }

        for (int ilink = 0; ilink < nstr * nlink; ilink++) {
                int *link = links + (size_t)ilink * NLINK_FIELDS;
                int target = link[LINK_TARGET];
                if (link[LINK_SIGN] == 0 || target < 0) {
                        continue;
                }
                int dk = mod_pos(link[LINK_DK], nkpts);
                counts[target * nkpts + dk]++;
        }
        for (int key = 0; key < nkeys; key++) {
                offsets[key + 1] = offsets[key] + counts[key];
                counts[key] = offsets[key];
        }

        indices = malloc(sizeof(int) * (size_t)(offsets[nkeys] > 0
                                                 ? offsets[nkeys] : 1));
        if (indices == NULL) {
                free(counts);
                free(offsets);
                return 1;
        }
        for (int ilink = 0; ilink < nstr * nlink; ilink++) {
                int *link = links + (size_t)ilink * NLINK_FIELDS;
                int target = link[LINK_TARGET];
                if (link[LINK_SIGN] == 0 || target < 0) {
                        continue;
                }
                int dk = mod_pos(link[LINK_DK], nkpts);
                int key = target * nkpts + dk;
                indices[counts[key]++] = ilink;
        }

        free(counts);
        order->offsets = offsets;
        order->indices = indices;
        return 0;
}


static int make_string_layout(int nkpts, int nstr, int *str2tot,
                              int **p_string_k, int **p_string_local)
{
        int *string_k = malloc(sizeof(int) * (size_t)nstr);
        int *string_local = malloc(sizeof(int) * (size_t)nstr);

        if (string_k == NULL || string_local == NULL) {
                free(string_k);
                free(string_local);
                return 1;
        }

        for (int istr = 0; istr < nstr; istr++) {
                string_k[istr] = -1;
                string_local[istr] = -1;
                for (int k = 0; k < nkpts; k++) {
                        int local = str2tot[(size_t)k * nstr + istr];
                        if (local >= 0) {
                                string_k[istr] = k;
                                string_local[istr] = local;
                                break;
                        }
                }
                if (string_k[istr] < 0) {
                        free(string_k);
                        free(string_local);
                        return 1;
                }
        }

        *p_string_k = string_k;
        *p_string_local = string_local;
        return 0;
}


static void free_layout(KFCIRDMLayout *layout)
{
        free(layout->block_offset);
        free(layout->block_na);
        free(layout->block_nb);
        free(layout->stra_k);
        free(layout->stra_local);
        free(layout->strb_k);
        free(layout->strb_local);
}


static int make_layout(KFCIRDMLayout *layout, int norb, int nkpts,
                       int nblocks, int *blocks, int nstra, int nstrb,
                       int *str2tot_a, int *str2tot_b)
{
        int ndet;

        layout->norb = norb;
        layout->nkpts = nkpts;
        layout->nstra = nstra;
        layout->nstrb = nstrb;
        layout->block_offset = NULL;
        layout->block_na = NULL;
        layout->block_nb = NULL;
        layout->stra_k = NULL;
        layout->stra_local = NULL;
        layout->strb_k = NULL;
        layout->strb_local = NULL;

        if (pbc_kfci_make_block_tables(
                    nkpts, nblocks, blocks, &layout->block_offset,
                    &layout->block_na, &layout->block_nb, &ndet)) {
                return 1;
        }
        if (make_string_layout(nkpts, nstra, str2tot_a,
                               &layout->stra_k, &layout->stra_local) ||
            make_string_layout(nkpts, nstrb, str2tot_b,
                               &layout->strb_k, &layout->strb_local)) {
                free_layout(layout);
                return 1;
        }
        return 0;
}


static inline size_t dm2_index(int p, int q, int r, int s, int norb)
{
        return (((size_t)p * norb + q) * norb + r) * norb + s;
}


static void reduce_thread_buffers(double complex *out, double complex *work,
                                  int nthreads, size_t size)
{
        pbc_kfci_zset0(out, size);
        for (int ithread = 0; ithread < nthreads; ithread++) {
                double complex *src = work + (size_t)ithread * size;
                for (size_t i = 0; i < size; i++) {
                        out[i] += src[i];
                }
        }
}


/*
 * Accumulate one spin block directly from the packed CI vector.  For the
 * two-body matrix this evaluates the Gram matrix of all one-body-excited CI
 * vectors, matching FCIrdm12kern_[ab]_cplx before reorder_rdm is applied.
 */
static int make_same_spin_rdm(double complex *dm1, double complex *dm2,
                              double complex *ci, KFCIRDMLayout *layout,
                              int *link_index, int nstr, int nlink,
                              int alpha)
{
        int norb = layout->norb;
        int nkpts = layout->nkpts;
        size_t ndm1 = (size_t)norb * norb;
        size_t ndm2 = ndm1 * ndm1;
        int nthreads = omp_get_max_threads();
        double complex *work1 = calloc((size_t)nthreads * ndm1,
                                       sizeof(double complex));
        double complex *work2 = NULL;

        if (dm2 != NULL) {
                work2 = calloc((size_t)nthreads * ndm2,
                               sizeof(double complex));
        }
        if (work1 == NULL || (dm2 != NULL && work2 == NULL)) {
                free(work1);
                free(work2);
                return 1;
        }

#pragma omp parallel
{
        int tid = omp_get_thread_num();
        double complex *local1 = work1 + (size_t)tid * ndm1;
        double complex *local2 = (work2 == NULL) ? NULL
                                : work2 + (size_t)tid * ndm2;

#pragma omp for schedule(dynamic, 8)
        for (int str0 = 0; str0 < nstr; str0++) {
                int source_k = alpha ? layout->stra_k[str0]
                                     : layout->strb_k[str0];
                int source_local = alpha ? layout->stra_local[str0]
                                         : layout->strb_local[str0];
                int *links0 = link_index + (size_t)str0 * nlink * NLINK_FIELDS;

                for (int ilink = 0; ilink < nlink; ilink++) {
                        int *li = links0 + ilink * NLINK_FIELDS;
                        int sign_i = li[LINK_SIGN];
                        int target_i = li[LINK_TARGET];
                        if (sign_i == 0 || target_i < 0) {
                                continue;
                        }

                        int p = li[LINK_CRE];
                        int q = li[LINK_DES];
                        int target_k = alpha ? layout->stra_k[target_i]
                                             : layout->strb_k[target_i];
                        int target_local = alpha
                                ? layout->stra_local[target_i]
                                : layout->strb_local[target_i];

                        /* A one-body RDM element preserves total momentum. */
                        if (target_k == source_k) {
                                for (int other_k = 0; other_k < nkpts;
                                     other_k++) {
                                        int key = alpha
                                                ? source_k * nkpts + other_k
                                                : other_k * nkpts + source_k;
                                        int offset = layout->block_offset[key];
                                        if (offset < 0) {
                                                continue;
                                        }
                                        int na = layout->block_na[key];
                                        int nb = layout->block_nb[key];
                                        double complex value = 0.0;
                                        if (alpha) {
                                                double complex *source = ci + offset
                                                        + (size_t)source_local * nb;
                                                double complex *target = ci + offset
                                                        + (size_t)target_local * nb;
                                                for (int ib = 0; ib < nb; ib++) {
                                                        value += conj(source[ib])
                                                               * target[ib];
                                                }
                                        } else {
                                                for (int ia = 0; ia < na; ia++) {
                                                        double complex *row = ci + offset
                                                                + (size_t)ia * nb;
                                                        value += conj(row[source_local])
                                                               * row[target_local];
                                                }
                                        }
                                        local1[(size_t)p * norb + q]
                                                += (double)sign_i * value;
                                }
                        }

                        if (local2 == NULL) {
                                continue;
                        }

                        for (int jlink = 0; jlink < nlink; jlink++) {
                                int *lj = links0 + jlink * NLINK_FIELDS;
                                int sign_j = lj[LINK_SIGN];
                                int target_j = lj[LINK_TARGET];
                                if (sign_j == 0 || target_j < 0) {
                                        continue;
                                }
                                int target_j_k = alpha
                                        ? layout->stra_k[target_j]
                                        : layout->strb_k[target_j];
                                if (target_j_k != target_k) {
                                        continue;
                                }

                                int r = lj[LINK_CRE];
                                int s = lj[LINK_DES];
                                int target_j_local = alpha
                                        ? layout->stra_local[target_j]
                                        : layout->strb_local[target_j];
                                double complex value = 0.0;

                                for (int other_k = 0; other_k < nkpts;
                                     other_k++) {
                                        int key = alpha
                                                ? target_k * nkpts + other_k
                                                : other_k * nkpts + target_k;
                                        int offset = layout->block_offset[key];
                                        if (offset < 0) {
                                                continue;
                                        }
                                        int na = layout->block_na[key];
                                        int nb = layout->block_nb[key];
                                        if (alpha) {
                                                double complex *row_i = ci + offset
                                                        + (size_t)target_local * nb;
                                                double complex *row_j = ci + offset
                                                        + (size_t)target_j_local * nb;
                                                for (int ib = 0; ib < nb; ib++) {
                                                        value += conj(row_i[ib])
                                                               * row_j[ib];
                                                }
                                        } else {
                                                for (int ia = 0; ia < na; ia++) {
                                                        double complex *row = ci + offset
                                                                + (size_t)ia * nb;
                                                        value += conj(row[target_local])
                                                               * row[target_j_local];
                                                }
                                        }
                                }
                                local2[dm2_index(p, q, s, r, norb)]
                                        += (double)(sign_i * sign_j) * value;
                        }
                }
        }
}

        reduce_thread_buffers(dm1, work1, nthreads, ndm1);
        if (dm2 != NULL) {
                reduce_thread_buffers(dm2, work2, nthreads, ndm2);
        }
        free(work1);
        free(work2);
        return 0;
}


/* Accumulate the alpha-beta RDM from paired, momentum-conserving links. */
static int make_ab_rdm(double complex *dm2ab, double complex *ci,
                       KFCIRDMLayout *layout, int nblocks, int *blocks,
                       int *linka, int nlinka, int *linkb, int nlinkb,
                       int *stra_ids, int *stra_offsets,
                       int *strb_ids, int *strb_offsets, int *kneg)
{
        int norb = layout->norb;
        int nkpts = layout->nkpts;
        size_t ndm2 = (size_t)norb * norb * norb * norb;
        int nthreads = omp_get_max_threads();
        double complex *work = calloc((size_t)nthreads * ndm2,
                                      sizeof(double complex));
        int *alpha_prefix = malloc(sizeof(int) * (size_t)(nblocks + 1));
        KFCITargetLinkOrder order_a;
        KFCITargetLinkOrder order_b;
        order_a.offsets = order_a.indices = NULL;
        order_b.offsets = order_b.indices = NULL;

        if (work == NULL || alpha_prefix == NULL ||
            make_target_link_order(linka, layout->nstra, nlinka, nkpts,
                                   &order_a) ||
            make_target_link_order(linkb, layout->nstrb, nlinkb, nkpts,
                                   &order_b)) {
                free(work);
                free(alpha_prefix);
                free(order_a.offsets);
                free(order_a.indices);
                free(order_b.offsets);
                free(order_b.indices);
                return 1;
        }

        alpha_prefix[0] = 0;
        for (int iblock = 0; iblock < nblocks; iblock++) {
                alpha_prefix[iblock + 1] = alpha_prefix[iblock]
                        + blocks[iblock * 6 + BLOCK_NA];
        }
        int alpha_tasks = alpha_prefix[nblocks];

#pragma omp parallel
{
        int tid = omp_get_thread_num();
        double complex *local = work + (size_t)tid * ndm2;

#pragma omp for schedule(dynamic, 1)
        for (int itask = 0; itask < alpha_tasks; itask++) {
                int iblock = 0;
                while (alpha_prefix[iblock + 1] <= itask) {
                        iblock++;
                }
                int *block = blocks + iblock * 6;
                int ka = block[BLOCK_KA];
                int kb = block[BLOCK_KB];
                int nb = block[BLOCK_NB];
                int offset = block[BLOCK_OFFSET];
                int ia1 = itask - alpha_prefix[iblock];
                int stra1 = stra_ids[stra_offsets[ka] + ia1];

                for (int ib1 = 0; ib1 < nb; ib1++) {
                        int strb1 = strb_ids[strb_offsets[kb] + ib1];
                        double complex target = ci[offset
                                + (size_t)ia1 * nb + ib1];

                        for (int dka = 0; dka < nkpts; dka++) {
                                int dkb = kneg[dka];
                                int akey = stra1 * nkpts + dka;
                                int bkey = strb1 * nkpts + dkb;
                                int a0 = order_a.offsets[akey];
                                int a1 = order_a.offsets[akey + 1];
                                int b0 = order_b.offsets[bkey];
                                int b1 = order_b.offsets[bkey + 1];

                                for (int ai = a0; ai < a1; ai++) {
                                        int aid = order_a.indices[ai];
                                        int stra0 = aid / nlinka;
                                        int *alink = linka
                                                + (size_t)aid * NLINK_FIELDS;
                                        int ka0 = layout->stra_k[stra0];
                                        int ia0 = layout->stra_local[stra0];

                                        for (int bi = b0; bi < b1; bi++) {
                                                int bid = order_b.indices[bi];
                                                int strb0 = bid / nlinkb;
                                                int *blink = linkb
                                                        + (size_t)bid
                                                        * NLINK_FIELDS;
                                                int kb0 = layout->strb_k[strb0];
                                                int source_key = ka0 * nkpts
                                                        + kb0;
                                                int source_offset =
                                                        layout->block_offset[
                                                                source_key];
                                                if (source_offset < 0) {
                                                        continue;
                                                }
                                                int ib0 =
                                                        layout->strb_local[strb0];
                                                int source_nb =
                                                        layout->block_nb[
                                                                source_key];
                                                double complex source = ci[
                                                        source_offset
                                                        + (size_t)ia0 * source_nb
                                                        + ib0];
                                                int p = alink[LINK_CRE];
                                                int q = alink[LINK_DES];
                                                int r = blink[LINK_CRE];
                                                int s = blink[LINK_DES];
                                                local[dm2_index(q, p, s, r,
                                                                norb)] +=
                                                        (double)(alink[LINK_SIGN]
                                                                 * blink[LINK_SIGN])
                                                        * conj(source) * target;
                                        }
                                }
                        }
                }
        }
}

        reduce_thread_buffers(dm2ab, work, nthreads, ndm2);
        free(work);
        free(alpha_prefix);
        free(order_a.offsets);
        free(order_a.indices);
        free(order_b.offsets);
        free(order_b.indices);
        return 0;
}


int FCIkci_make_rdm1s_direct(double complex *dm1a, double complex *dm1b,
                             double complex *ci, int norb, int nkpts,
                             int nblocks, int *blocks,
                             int *linka, int nstra, int nlinka,
                             int *linkb, int nstrb, int nlinkb,
                             int *str2tot_a, int *str2tot_b)
{
        KFCIRDMLayout layout;
        int err = make_layout(&layout, norb, nkpts, nblocks, blocks,
                              nstra, nstrb, str2tot_a, str2tot_b);
        if (err) {
                return err;
        }
        err = make_same_spin_rdm(dm1a, NULL, ci, &layout, linka, nstra,
                                 nlinka, 1);
        if (!err) {
                err = make_same_spin_rdm(dm1b, NULL, ci, &layout, linkb,
                                         nstrb, nlinkb, 0);
        }
        free_layout(&layout);
        return err;
}


int FCIkci_make_rdm12s_direct(
        double complex *dm1a, double complex *dm1b,
        double complex *dm2aa, double complex *dm2ab,
        double complex *dm2bb, double complex *ci,
        int norb, int nkpts, int nblocks, int *blocks,
        int *linka, int nstra, int nlinka,
        int *linkb, int nstrb, int nlinkb,
        int *stra_ids, int *stra_offsets,
        int *strb_ids, int *strb_offsets,
        int *str2tot_a, int *str2tot_b, int *kneg)
{
        KFCIRDMLayout layout;
        int err = make_layout(&layout, norb, nkpts, nblocks, blocks,
                              nstra, nstrb, str2tot_a, str2tot_b);
        if (err) {
                return err;
        }

        err = make_same_spin_rdm(dm1a, dm2aa, ci, &layout, linka, nstra,
                                 nlinka, 1);
        if (!err) {
                err = make_same_spin_rdm(dm1b, dm2bb, ci, &layout, linkb,
                                         nstrb, nlinkb, 0);
        }
        if (!err) {
                err = make_ab_rdm(dm2ab, ci, &layout, nblocks, blocks,
                                  linka, nlinka, linkb, nlinkb,
                                  stra_ids, stra_offsets,
                                  strb_ids, strb_offsets, kneg);
        }
        free_layout(&layout);
        return err;
}
