/* Spherical-shell DSMC for elastic, isotropic, constant-cross-section SIDM.
 * DM uses the ordinary mass/potential/orbit arrays, but no gravitational
 * relaxation or stellar physics. See docs/DM_SIDM.rst for assumptions. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <stdint.h>
#include "cmc.h"
#include "cmc_vars.h"
#include "cmc_sidm_kernel.h"

typedef struct {
    long global_index;
    double r, m, vr, vt;
} dm_sample;

int is_dm(long k) { return star[k].se_k == CMC_DM_TYPE; }

static void dm_fail(const char *message)
{
    fprintf(stderr, "DM/SIDM: %s (rank %d)\n", message, myid);
    MPI_Abort(MPI_COMM_WORLD, 1);
    abort();
}

static void *dm_alloc(size_t n, size_t size)
{
    if (n && size > SIZE_MAX/n) dm_fail("allocation overflow");
    void *p = calloc(n ? n : 1, size);
    if (!p) dm_fail("allocation failed");
    return p;
}

void validate_dm(void)
{
    if (!isfinite(SIDM_SIGMA_OVER_M) || SIDM_SIGMA_OVER_M < 0.0)
        dm_fail("SIDM_SIGMA_OVER_M must be finite and nonnegative");
    if (!isfinite(SIDM_PMAX) || SIDM_PMAX <= 0.0 || SIDM_PMAX > 0.1)
        dm_fail("SIDM_PMAX must be in (0, 0.1]");
    if (SIDM_SHELL_SIZE < 2 || SIDM_SHELL_SIZE > 1000000)
        dm_fail("SIDM_SHELL_SIZE must be in [2, 1000000]");
    long count = 0;
    double mmin = GSL_POSINF, mmax = 0.0;
    for (long k = 1; k <= End[myid]-Start[myid]+1; ++k) {
        long b = star[k].binind;
        if (b && (binary[b].bse_kw[0] == CMC_DM_TYPE || binary[b].bse_kw[1] == CMC_DM_TYPE))
            dm_fail("DM particles cannot be binary components");
        if (!is_dm(k)) continue;
        if (b) dm_fail("DM particles must be single objects");
        long g = get_global_idx(k);
        double m = star_m[g];
        if (!(m > 0.0) || !isfinite(m) || !(star_r[g] > 0.0) || !isfinite(star_r[g]) ||
            !isfinite(star[k].vr) || !isfinite(star[k].vt) || star[k].vt < 0.0)
            dm_fail("invalid DM mass, radius or velocity");
        star[k].rad = 0.0;
        star[k].lifetime = GSL_POSINF;
        mmin = fmin(mmin, m); mmax = fmax(mmax, m); ++count;
    }
    MPI_Allreduce(&count, &DM_N, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &mmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &mmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (SIDM_SIGMA_OVER_M > 0.0 && DM_N && mmax-mmin > 1.e-12*mmax)
        dm_fail("SIDM requires equal-mass DM simulation particles");
    if (DM_N && !(log(GAMMA * clus.N_STAR) > 0.0))
        dm_fail("invalid CMC time normalization: GAMMA*N must exceed one");
    if (myid == 0 && DM_N)
        fprintf(stdout, "DM: %ld particles, type=%d; no gravitational relaxation; SIDM sigma/m=%g cm^2/g\n",
                DM_N, CMC_DM_TYPE, SIDM_SIGMA_OVER_M);
}

/* Only root holds the global DM sample. Other ranks keep their owned DM
 * slice; this preserves the memory benefit of the node-shared gravity arrays.
 * Counts/displacements are in objects, not bytes (the MPI datatype supplies
 * the record extent). No persistent particle/checkpoint state is introduced. */
typedef struct {
    int total, local_n;
    int *counts, *offsets;
    dm_sample *local, *all;
    MPI_Datatype record_type;
} dm_batch;

static dm_batch dm_collect(void)
{
    dm_batch batch = {0};
    for (long k=1; k<=mpiEnd-mpiBegin+1; ++k) if (is_dm(k)) ++batch.local_n;
    batch.local = dm_alloc(batch.local_n, sizeof(dm_sample));
    int j = 0;
    for (long k=1; k<=mpiEnd-mpiBegin+1; ++k) {
        if (!is_dm(k)) continue;
        long g = get_global_idx(k);
        batch.local[j++] = (dm_sample){g, star_r[g], star_m[g], star[k].vr, star[k].vt};
    }
    batch.counts = dm_alloc(procs, sizeof(int));
    batch.offsets = dm_alloc(procs, sizeof(int));
    MPI_Allgather(&batch.local_n, 1, MPI_INT, batch.counts, 1, MPI_INT, MPI_COMM_WORLD);
    long total = 0;
    for (int p=0; p<procs; ++p) {
        if (total + batch.counts[p] > INT_MAX) dm_fail("DM count exceeds MPI int displacement range");
        batch.offsets[p] = total;
        total += batch.counts[p];
    }
    batch.total = total; DM_N = total;
    if (myid == 0) batch.all = dm_alloc(batch.total, sizeof(dm_sample));
    MPI_Type_contiguous(sizeof(dm_sample), MPI_BYTE, &batch.record_type);
    MPI_Type_commit(&batch.record_type);
    MPI_Gatherv(batch.local, batch.local_n, batch.record_type, batch.all,
                batch.counts, batch.offsets, batch.record_type, 0, MPI_COMM_WORLD);
    return batch;
}

static void dm_release(dm_batch *batch)
{
    MPI_Type_free(&batch->record_type);
    free(batch->local); free(batch->all);
    free(batch->counts); free(batch->offsets);
}

/* Merge a last singleton into its preceding shell. Shell boundaries bisect
 * adjacent DM radii; the outer boundary extrapolates half the last spacing. */
static int shell_end(int begin, int n)
{
    int end = begin + SIDM_SHELL_SIZE;
    if (end >= n-1) end = n;
    return end;
}

static double shell_coefficient(const dm_sample *s, int begin, int end, int n)
{
    double lo = begin ? 0.5*(s[begin-1].r + s[begin].r) : 0.0;
    double hi = end < n ? 0.5*(s[end-1].r + s[end].r)
                       : s[n-1].r + 0.5*(s[n-1].r - s[n-2].r);
    double volume = (4.0*PI/3.0)*(hi-lo)*(hi*hi + hi*lo + lo*lo);
    if (!(volume > 0.0) || !isfinite(volume)) dm_fail("invalid SIDM shell volume");
    /* m_code * units.mstar [g], V_code * units.l^3 [cm^3],
     * v_code * units.l/units.t [cm/s],
     * dt_code * units.t*N/log(GAMMA*N) [s]. */
    double c = sidm_matching_weight(end-begin) * s[begin].m * SIDM_SIGMA_OVER_M
             * (units.mstar / (units.l*units.l))
             * (clus.N_STAR / log(GAMMA*clus.N_STAR)) / volume;
    if (!isfinite(c) || c < 0.0) dm_fail("nonfinite SIDM rate coefficient");
    return c;
}

double sidm_timestep(void)
{
    if (!DM_N || SIDM_SIGMA_OVER_M == 0.0) return GSL_POSINF;
    dm_batch batch = dm_collect();
    double limit = GSL_POSINF;
    if (myid == 0) {
        int n = batch.total;
        dm_sample *s = batch.all;
        double bound = 0.0;
        for (int b=0; b<n-1; ) {
            int e = shell_end(b, n);
            double vmax = 0.0;
            for (int i=b; i<e; ++i) vmax = fmax(vmax, hypot(s[i].vr, s[i].vt));
            /* Each particle is used at most once this step, so w <= 2*vmax
             * bounds every pair's incoming relative speed. */
            bound = fmax(bound, shell_coefficient(s,b,e,n) * 2.0*vmax);
            b = e;
        }
        if (bound > 0.0) limit = SIDM_PMAX / bound;
    }
    MPI_Bcast(&limit, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    dm_release(&batch);
    return limit;
}

void sidm_apply(double dt)
{
    if (!DM_N || SIDM_SIGMA_OVER_M == 0.0) return;
    dm_batch batch = dm_collect();
    int n = batch.total;
    dm_sample *s = batch.all;
    if (myid == 0) {
        long tried = 0, events = 0;
        double expected = 0.0, pmax = 0.0, ke_before = 0.0, ke_after = 0.0;
        int *order = dm_alloc(SIDM_SHELL_SIZE+1, sizeof(int));
        for (int i=0; i<n; ++i) ke_before += 0.5*s[i].m*(s[i].vr*s[i].vr+s[i].vt*s[i].vt);
        for (int b=0; b<n-1; ) {
            int e = shell_end(b,n), count = e-b;
            double coefficient = shell_coefficient(s,b,e,n);
            for (int i=0; i<count; ++i) order[i] = b+i;
            for (int i=count-1; i>0; --i) {
                int j = (int)(rng_t113_dbl_new(curr_st)*(i+1));
                int tmp = order[i]; order[i] = order[j]; order[j] = tmp;
            }
            for (int i=0; i+1<count; i+=2) {
                dm_sample *a = &s[order[i]], *z = &s[order[i+1]];
                double azimuth = 2.0*PI*rng_t113_dbl_new(curr_st);
                double va[3] = {a->vt, 0.0, a->vr};
                double vz[3] = {z->vt*cos(azimuth), z->vt*sin(azimuth), z->vr};
                double w2 = 0.0;
                for (int d=0; d<3; ++d) w2 += (va[d]-vz[d])*(va[d]-vz[d]);
                double probability = coefficient * sqrt(w2) * dt;
                if (!isfinite(probability) || probability < 0.0 || probability > SIDM_PMAX*(1.0+1.e-10))
                    dm_fail("SIDM probability exceeds timestep bound");
                ++tried; expected += probability; pmax = fmax(pmax, probability);
                if (rng_t113_dbl_new(curr_st) >= probability) continue;
                double u = rng_t113_dbl_new(curr_st), v = rng_t113_dbl_new(curr_st);
                sidm_elastic_equal(va,vz,u,v);
                a->vr = va[2]; a->vt = hypot(va[0],va[1]);
                z->vr = vz[2]; z->vt = hypot(vz[0],vz[1]);
                ++events;
            }
            b = e;
        }
        for (int i=0; i<n; ++i) ke_after += 0.5*s[i].m*(s[i].vr*s[i].vr+s[i].vt*s[i].vt);
        char path[160];
        snprintf(path,sizeof(path),"%s.sidm.dat",outprefix);
        /* New invocations, including restarts, get an explicit segment header. */
        static int started = 0;
        FILE *fp = fopen(path, !started && RESTART_TCOUNT == 0 ? "w" : "a");
        if (!fp) dm_fail("cannot open SIDM diagnostics");
        if (!started) fprintf(fp,"# tcount time dt N_DM pairs events expected_events max_probability delta_KE/KE (SIDM only)\n");
        started = 1;
        fprintf(fp,"%ld %.17g %.17g %d %ld %ld %.17g %.17g %.17g\n",
                tcount,TotalTime,dt,n,tried,events,expected,pmax,
                ke_before ? (ke_after-ke_before)/ke_before : 0.0);
        fclose(fp); free(order);
    }
    MPI_Scatterv(batch.all, batch.counts, batch.offsets, batch.record_type,
                 batch.local, batch.local_n, batch.record_type, 0, MPI_COMM_WORLD);
    for (int i=0; i<batch.local_n; ++i) {
        long g = batch.local[i].global_index;
        long k = g-mpiBegin+1;
        star[k].vr = batch.local[i].vr; star[k].vt = batch.local[i].vt;
        set_star_EJ(k);
    }
    dm_release(&batch);
}
