/* Rank-local initial-condition input. Serial HDF5 is sufficient: every rank
 * opens the file read-only and selects only its own object/binary rows. */
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cmc.h"
#include "cmc_vars.h"
#include "hdf5.h"

static void input_fail(const char *message)
{
    fprintf(stderr, "rank %d: HDF5 initial conditions: %s\n", myid, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    abort();
}

static void *input_calloc(size_t n, size_t size)
{
    if (size && n > SIZE_MAX / size) input_fail("allocation overflow");
    void *p = calloc(n ? n : 1, size);
    if (!p) input_fail("allocation failed");
    return p;
}

static hid_t open_input(void)
{
    hid_t file = H5Fopen(INPUT_FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) input_fail("cannot open input file");
    return file;
}

static hid_t open_table(hid_t file, const char *name, hsize_t rows, hsize_t columns)
{
    hid_t ds = H5Dopen2(file, name, H5P_DEFAULT);
    if (ds < 0) input_fail("missing object/binary dataset");
    hid_t space = H5Dget_space(ds);
    hsize_t dims[2];
    if (space < 0 || H5Sget_simple_extent_ndims(space) != 2 ||
        H5Sget_simple_extent_dims(space, dims, NULL) < 0 ||
        dims[0] != rows || dims[1] != columns)
        input_fail("unexpected dataset shape (including sentinel rows)");
    H5Sclose(space);
    return ds;
}

static void read_attribute(hid_t ds, const char *name, hid_t type, void *value)
{
    hid_t attr = H5Aopen(ds, name, H5P_DEFAULT);
    if (attr < 0) input_fail("required header attribute missing");
    hid_t space = H5Aget_space(attr);
    if (space < 0 || H5Sget_simple_extent_npoints(space) != 1 ||
        H5Aread(attr, type, value) < 0)
        input_fail("invalid scalar header attribute");
    H5Sclose(space);
    H5Aclose(attr);
}

void cmc_read_local_header(void)
{
    hid_t file = open_input();
    hid_t ds = H5Dopen2(file, "CLUS_OBJ_DATA/block0_values", H5P_DEFAULT);
    if (ds < 0) input_fail("object dataset missing");
    read_attribute(ds, "NOBJ", H5T_NATIVE_LONG, &cfd.NOBJ);
    read_attribute(ds, "NBINARY", H5T_NATIVE_LONG, &cfd.NBINARY);
    read_attribute(ds, "MCLUS", H5T_NATIVE_DOUBLE, &cfd.Mclus);
    read_attribute(ds, "RVIR", H5T_NATIVE_DOUBLE, &cfd.Rvir);
    read_attribute(ds, "RTID", H5T_NATIVE_DOUBLE, &cfd.Rtid);
    read_attribute(ds, "Z", H5T_NATIVE_DOUBLE, &cfd.Z);
    /* Existing sorting and MPI count/displacement arrays use int. */
    if (cfd.NOBJ <= 0 || cfd.NBINARY < 0 ||
        cfd.NOBJ > INT_MAX - 2 || cfd.NBINARY > INT_MAX - 1 ||
        cfd.NOBJ + 2.0*cfd.NBINARY + 2 > INT_MAX/1.1 ||
        !isfinite(cfd.Mclus) || cfd.Mclus <= 0 ||
        !isfinite(cfd.Rvir) || cfd.Rvir <= 0 ||
        !isfinite(cfd.Rtid) || cfd.Rtid <= 0 || !isfinite(cfd.Z))
        input_fail("invalid header or model exceeds existing CMC index limits");
    H5Dclose(ds);
    H5Fclose(file);
}

/* Bounded temporary buffers, even for a single-rank model. */
static void read_rows(hid_t ds, hsize_t first, hsize_t n, hsize_t columns, double *buffer)
{
    if (!n) return;
    hsize_t offset[2] = {first, 0}, count[2] = {n, columns};
    hid_t file_space = H5Dget_space(ds);
    hid_t memory_space = H5Screate_simple(2, count, NULL);
    if (file_space < 0 || memory_space < 0 ||
        H5Sselect_hyperslab(file_space, H5S_SELECT_SET, offset, NULL, count, NULL) < 0 ||
        H5Dread(ds, H5T_NATIVE_DOUBLE, memory_space, file_space, H5P_DEFAULT, buffer) < 0)
        input_fail("cannot read requested rows");
    H5Sclose(memory_space);
    H5Sclose(file_space);
}

static long input_integer(double value, long maximum)
{
    if (!isfinite(value) || value < 0 || value > (double)maximum ||
        value >= 9007199254740992.0 || floor(value) != value)
        input_fail("invalid integer ID, type or binary reference");
    return (long)value;
}

static void unpack_object(long i, const double *x)
{
    for (int j = 0; j < 8; ++j)
        if (!isfinite(x[j])) input_fail("non-finite object value");
    cfd.obj_id[i] = input_integer(x[0], LONG_MAX);
    /* DM is the sole negative physical type accepted by local input. */
    cfd.obj_k[i] = x[1] == CMC_DM_TYPE ? CMC_DM_TYPE : (int)input_integer(x[1], INT_MAX);
    cfd.obj_m[i] = x[2]; cfd.obj_Reff[i] = x[3]; cfd.obj_r[i] = x[4];
    cfd.obj_vr[i] = x[5]; cfd.obj_vt[i] = x[6];
    cfd.obj_binind[i] = input_integer(x[7], cfd.NBINARY);
}

typedef struct { long row, slot; } binary_reference;
static int compare_reference(const void *a, const void *b)
{
    const binary_reference *x = a, *y = b;
    return (x->row > y->row) - (x->row < y->row);
}

static void unpack_binary(long i, long file_row, const double *x)
{
    for (int j = 0; j < 11; ++j)
        if (!isfinite(x[j])) input_fail("non-finite binary value");
    if (input_integer(x[0], cfd.NBINARY) != file_row)
        input_fail("binary index column does not match referenced file row");
    if (x[2] == CMC_DM_TYPE || x[6] == CMC_DM_TYPE)
        input_fail("DM particles cannot be binary components");
    cfd.bs_index[i] = file_row;
    cfd.bs_id1[i] = input_integer(x[1], LONG_MAX);
    cfd.bs_k1[i] = (int)input_integer(x[2], INT_MAX);
    cfd.bs_m1[i] = x[3]; cfd.bs_Reff1[i] = x[4];
    cfd.bs_id2[i] = input_integer(x[5], LONG_MAX);
    cfd.bs_k2[i] = (int)input_integer(x[6], INT_MAX);
    cfd.bs_m2[i] = x[7]; cfd.bs_Reff2[i] = x[8];
    cfd.bs_a[i] = x[9]; cfd.bs_e[i] = x[10];
}

void cmc_read_local_particles(void)
{
    const long n = End[myid] - Start[myid] + 1;
    if (n < 0) input_fail("invalid particle partition");
    hid_t file = open_input();
    hid_t objects = open_table(file, "CLUS_OBJ_DATA/block0_values", cfd.NOBJ+2, 8);
    hid_t binaries = open_table(file, "CLUS_BINARY_DATA/block0_values", cfd.NBINARY+1, 11);
#define ALLOC(field, count) cfd.field = input_calloc((count), sizeof(*cfd.field))
    ALLOC(obj_id,n+2); ALLOC(obj_k,n+2); ALLOC(obj_m,n+2); ALLOC(obj_Reff,n+2);
    ALLOC(obj_r,n+2); ALLOC(obj_vr,n+2); ALLOC(obj_vt,n+2); ALLOC(obj_binind,n+2);
    const long chunk = 4096;
    double *buffer = input_calloc(chunk * 11, sizeof(double));
    /* Preserve central-mass and outer sentinel values independently of ownership. */
    read_rows(objects, 0, 1, 8, buffer); unpack_object(0, buffer);
    read_rows(objects, cfd.NOBJ+1, 1, 8, buffer); unpack_object(n+1, buffer);
    if (cfd.obj_binind[0] || cfd.obj_binind[n+1]) input_fail("binary on a sentinel row");
    for (long first = 1; first <= n; first += chunk) {
        long count = n-first+1 < chunk ? n-first+1 : chunk;
        read_rows(objects, Start[myid]+first-1, count, 8, buffer);
        for (long j = 0; j < count; ++j) unpack_object(first+j, buffer+8*j);
    }
    long nb = 0;
    for (long i = 1; i <= n; ++i) if (cfd.obj_binind[i]) ++nb;
    binary_reference *refs = input_calloc(nb, sizeof(*refs));
    long slot = 0;
    for (long i = 1; i <= n; ++i) if (cfd.obj_binind[i]) {
        refs[slot].row = cfd.obj_binind[i]; refs[slot].slot = slot+1;
        cfd.obj_binind[i] = ++slot; /* remap file binary row to compact local row */
    }
    ALLOC(bs_index,nb+1); ALLOC(bs_id1,nb+1); ALLOC(bs_k1,nb+1);
    ALLOC(bs_m1,nb+1); ALLOC(bs_Reff1,nb+1); ALLOC(bs_id2,nb+1); ALLOC(bs_k2,nb+1);
    ALLOC(bs_m2,nb+1); ALLOC(bs_Reff2,nb+1); ALLOC(bs_a,nb+1); ALLOC(bs_e,nb+1);
#undef ALLOC
    read_rows(binaries, 0, 1, 11, buffer); unpack_binary(0, 0, buffer);
    qsort(refs, nb, sizeof(*refs), compare_reference);
    for (long i = 1; i < nb; ++i)
        if (refs[i].row == refs[i-1].row) input_fail("duplicate local binary reference");
    for (long first = 0; first < nb;) {
        long count = 1;
        while (first+count < nb && count < chunk &&
               refs[first+count].row == refs[first].row+count) ++count;
        read_rows(binaries, refs[first].row, count, 11, buffer);
        for (long j = 0; j < count; ++j)
            unpack_binary(refs[first+j].slot, refs[first+j].row, buffer+11*j);
        first += count;
    }
    free(refs); free(buffer);
    H5Dclose(binaries); H5Dclose(objects); H5Fclose(file);
    /* The stock capacity assumes roughly uniform binary placement. Primordial
     * binaries may all be on one rank. Preserve the actual capacity in checkpoints. */
    if (nb+2 > N_BIN_DIM_OPT) {
        long capacity = nb+2;
        free(binary);
        binary = input_calloc(capacity, sizeof(*binary));
        N_BIN_DIM_OPT = capacity;
    }
    if (n+2 > N_STAR_DIM_OPT) {
        long capacity = n+2;
        free(star); free(sigma_array.r); free(sigma_array.sigma);
        star = input_calloc(capacity, sizeof(*star));
        sigma_array.r = input_calloc(capacity, sizeof(double));
        sigma_array.sigma = input_calloc(capacity, sizeof(double));
        N_STAR_DIM_OPT = capacity;
    }
    fprintf(stderr, "rank %d: local HDF5 input: %ld/%ld objects, %ld/%ld binary rows\n",
            myid, n, cfd.NOBJ, nb, cfd.NBINARY);
}
