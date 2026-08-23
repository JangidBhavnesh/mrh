#ifndef PBC_KFCI_COMMON_H
#define PBC_KFCI_COMMON_H

#include <complex.h>
#include <stddef.h>

#define BLOCK_KA      0
#define BLOCK_KB      1
#define BLOCK_NA      2
#define BLOCK_NB      3
#define BLOCK_OFFSET  4
#define BLOCK_SIZE    5

#define LINK_CRE      0
#define LINK_DES      1
#define LINK_TARGET   2
#define LINK_SIGN     3
#define LINK_K0       4
#define LINK_K_CRE    5
#define LINK_K_DES    6
#define LINK_DK       7
#define NLINK_FIELDS  8

/** Return x modulo n in the range [0, n - 1]. */
static inline int mod_pos(int x, int n)
{
        int r = x % n;
        return (r < 0) ? r + n : r;
}

/** Flatten k-point and active-orbital indices into the ERI storage index. */
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

/** Set n complex values to zero. */
void pbc_kfci_zset0(double complex *x, size_t n);

/**
 * Build lookup arrays for packed CI momentum blocks.
 *
 * The caller owns the returned block_offset, block_na, and block_nb arrays.
 *
 * @return 0 on success; 1 on allocation failure.
 */
int pbc_kfci_make_block_tables(int nkpts, int nblocks, int *blocks,
                               int **p_block_offset,
                               int **p_block_na,
                               int **p_block_nb,
                               int *p_ndet);

#endif
