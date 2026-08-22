#include <complex.h>
#include <stddef.h>
#include <stdlib.h>

#include "pbc_kfci_common.h"

void pbc_kfci_zset0(double complex *x, size_t n)
{
        for (size_t i = 0; i < n; i++) {
                x[i] = 0.0 + 0.0 * I;
        }
}

int pbc_kfci_make_block_tables(int nkpts, int nblocks, int *blocks,
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
