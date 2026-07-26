/*
 * k-CASCI spectral-function helpers.
 *
 * The Python layer builds the pole table.  This file broadens those poles onto
 * a real-frequency grid.
 */

#include <math.h>
#include <omp.h>
#include <complex.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define KIND_HOLE 0
#define KIND_PARTICLE 1
#define BROADEN_LORENTZIAN 0
#define BROADEN_GAUSSIAN 1

#define BLOCK_KA      0
#define BLOCK_KB      1
#define BLOCK_NA      2
#define BLOCK_NB      3
#define BLOCK_OFFSET  4

#define OP_CRE    0
#define OP_DES    1
#define OP_TARGET 2
#define OP_SIGN   3

static inline double broaden_delta(double omega, double omega0,
                                   double eta, int broadening)
{
        double x = omega - omega0;
        if (broadening == BROADEN_GAUSSIAN) {
                double y = x / eta;
                return exp(-0.5 * y * y) / (eta * sqrt(2.0 * M_PI));
        }
        return eta / M_PI / (x * x + eta * eta);
}

void FCIspectral_broaden(double *hole, double *particle, double *total,
                         int *kind, int *k_index, int *orbital, int *spin,
                         double *omega0, double *weight,
                         double *omega_grid,
                         int npoles, int nomega,
                         int nkpts, int norb_axis, int spin_axis,
                         int orbital_resolved, int spin_resolved,
                         double eta, int broadening)
{
#pragma omp parallel for schedule(dynamic) default(none) \
        if(npoles * nomega > 10000) \
        shared(hole, particle, total, kind, k_index, orbital, spin, \
               omega0, weight, omega_grid, npoles, nomega, nkpts, \
               norb_axis, spin_axis, orbital_resolved, spin_resolved, \
               eta, broadening)
        for (int ipole = 0; ipole < npoles; ipole++) {
                int k = k_index[ipole];
                int p = orbital_resolved ? orbital[ipole] : 0;
                int s = spin_resolved ? spin[ipole] : 0;

                if (k < 0 || k >= nkpts ||
                    p < 0 || p >= norb_axis ||
                    s < 0 || s >= spin_axis) {
                        continue;
                }

                long base = (((long)k * norb_axis + p) * spin_axis + s)
                        * nomega;
                double pole_omega = omega0[ipole];
                double pole_weight = weight[ipole];

                for (int iw = 0; iw < nomega; iw++) {
                        double value = pole_weight *
                                broaden_delta(omega_grid[iw], pole_omega,
                                              eta, broadening);
                        long idx = base + iw;

#pragma omp atomic
                        total[idx] += value;
                        if (kind[ipole] == KIND_HOLE) {
#pragma omp atomic
                                hole[idx] += value;
                        } else if (kind[ipole] == KIND_PARTICLE) {
#pragma omp atomic
                                particle[idx] += value;
                        }
                }
        }
}

static inline int mod_pos(int x, int n)
{
        int r = x % n;
        return (r < 0) ? r + n : r;
}

void FCIspectral_apply_k_op(double complex *out,
                            double complex *fcivec,
                            int *blocks, int nblocks, int nkpts,
                            int *stra_ids, int *stra_offsets,
                            int *strb_ids, int *strb_offsets,
                            int *target_str2loc_a, int target_nstra,
                            int *target_str2loc_b, int target_nstrb,
                            int *target_block_offset,
                            int *target_block_na,
                            int *target_block_nb,
                            int *op_index, int nlink,
                            int orb, int k_op, int spin, int cre,
                            int beta_phase)
{
#pragma omp parallel for schedule(dynamic) default(none) if(nblocks > 8) \
        shared(out, fcivec, blocks, nblocks, nkpts, stra_ids, stra_offsets, \
               strb_ids, strb_offsets, target_str2loc_a, target_nstra, \
               target_str2loc_b, target_nstrb, target_block_offset, \
               target_block_na, target_block_nb, op_index, nlink, orb, \
               k_op, spin, cre, beta_phase)
        for (int iblk = 0; iblk < nblocks; iblk++) {
                int *blk = blocks + iblk * 6;
                int ka = blk[BLOCK_KA];
                int kb = blk[BLOCK_KB];
                int na = blk[BLOCK_NA];
                int nb = blk[BLOCK_NB];
                int offset = blk[BLOCK_OFFSET];

                if (spin == 0) {
                        int ka1 = cre ? mod_pos(ka + k_op, nkpts)
                                      : mod_pos(ka - k_op, nkpts);
                        int key = ka1 * nkpts + kb;
                        int off1 = target_block_offset[key];
                        if (off1 < 0) {
                                continue;
                        }
                        int nb1 = target_block_nb[key];

                        for (int ia = 0; ia < na; ia++) {
                                int str0 = stra_ids[stra_offsets[ka] + ia];
                                int ia1 = -1;
                                int sign1 = 0;

                                for (int ilink = 0; ilink < nlink; ilink++) {
                                        int *link = op_index +
                                                ((long)str0 * nlink + ilink) * 4;
                                        int sign = link[OP_SIGN];
                                        if (sign == 0) {
                                                break;
                                        }
                                        int op_orb = cre ? link[OP_CRE]
                                                         : link[OP_DES];
                                        if (op_orb != orb) {
                                                continue;
                                        }
                                        int str1 = link[OP_TARGET];
                                        ia1 = target_str2loc_a[
                                                ka1 * target_nstra + str1];
                                        sign1 = sign;
                                        break;
                                }

                                if (ia1 < 0) {
                                        continue;
                                }
                                for (int ib = 0; ib < nb; ib++) {
                                        out[off1 + ia1 * nb1 + ib] +=
                                                (double)sign1 *
                                                fcivec[offset + ia * nb + ib];
                                }
                        }
                } else {
                        int kb1 = cre ? mod_pos(kb + k_op, nkpts)
                                      : mod_pos(kb - k_op, nkpts);
                        int key = ka * nkpts + kb1;
                        int off1 = target_block_offset[key];
                        if (off1 < 0) {
                                continue;
                        }
                        int na1 = target_block_na[key];
                        int nb1 = target_block_nb[key];

                        for (int ib = 0; ib < nb; ib++) {
                                int str0 = strb_ids[strb_offsets[kb] + ib];
                                int ib1 = -1;
                                int sign1 = 0;

                                for (int ilink = 0; ilink < nlink; ilink++) {
                                        int *link = op_index +
                                                ((long)str0 * nlink + ilink) * 4;
                                        int sign = link[OP_SIGN];
                                        if (sign == 0) {
                                                break;
                                        }
                                        int op_orb = cre ? link[OP_CRE]
                                                         : link[OP_DES];
                                        if (op_orb != orb) {
                                                continue;
                                        }
                                        int str1 = link[OP_TARGET];
                                        ib1 = target_str2loc_b[
                                                kb1 * target_nstrb + str1];
                                        sign1 = sign;
                                        break;
                                }

                                if (ib1 < 0) {
                                        continue;
                                }
                                for (int ia = 0; ia < na; ia++) {
                                        out[off1 + ia * nb1 + ib1] +=
                                                (double)(beta_phase * sign1) *
                                                fcivec[offset + ia * nb + ib];
                                }
                        }
                }
        }
}
