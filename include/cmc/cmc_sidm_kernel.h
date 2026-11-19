#ifndef CMC_SIDM_KERNEL_H
#define CMC_SIDM_KERNEL_H
#include <math.h>

/* Random matching selects a given unordered pair with probability 1/(n-1)
 * for even n and 1/n for odd n (one random particle sits out). */
static inline double sidm_matching_weight(int n) { return n % 2 ? n : n - 1; }

/* Equal-mass elastic scattering, isotropic in the centre-of-mass frame.
 * u_mu and u_phi are independent uniforms in [0,1). Velocities use a common
 * local Cartesian frame; caller projects back to the spherical MC variables. */
static inline void sidm_elastic_equal(double a[3], double b[3], double u_mu, double u_phi)
{
    double w2 = 0.0, cm[3];
    for (int d = 0; d < 3; ++d) {
        cm[d] = 0.5 * (a[d] + b[d]);
        w2 += (a[d] - b[d]) * (a[d] - b[d]);
    }
    double mu = 2.0 * u_mu - 1.0, phi = 6.2831853071795864769 * u_phi;
    double h = 0.5 * sqrt(w2), st = sqrt(fmax(0.0, 1.0 - mu * mu));
    double w[3] = {h * st * cos(phi), h * st * sin(phi), h * mu};
    for (int d = 0; d < 3; ++d) {
        a[d] = cm[d] + w[d];
        b[d] = cm[d] - w[d];
    }
}
#endif
