/*
 * k-CASCI spectral-function helpers.
 *
 * The Python layer builds the pole table.  This file broadens those poles onto
 * a real-frequency grid.
 */

#include <math.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define KIND_HOLE 0
#define KIND_PARTICLE 1
#define BROADEN_LORENTZIAN 0
#define BROADEN_GAUSSIAN 1

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
