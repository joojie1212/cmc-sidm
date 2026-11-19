/* Node-shared, read-mostly global arrays with rank-private COW updates.
 * A writable MPI shared window alone is NOT equivalent to replicated arrays:
 * local encounter/evolution writes must remain invisible to other ranks until
 * the original publication points. Linux MAP_PRIVATE supplies that isolation.
 * MPI identifies physical nodes and publishes snapshots; POSIX shm backs them.
 */
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cmc.h"
#include "cmc_vars.h"
#ifdef __linux__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif

static int enabled;
static MPI_Comm node_comm = MPI_COMM_NULL, leaders_comm = MPI_COMM_NULL;
static int node_rank, node_size;
#ifdef __linux__
static int backing_fd = -1;
static size_t stride, mapping_bytes, page_bytes;
static char *private_mapping, *shared_mapping;
#endif

static void shared_fail(const char *message)
{
    fprintf(stderr, "rank %d: shared global arrays: %s (errno=%d: %s)\n",
            myid, message, errno, strerror(errno));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    abort();
}

int cmc_globals_shared(void) { return enabled; }

static double *view(int which)
{
    if (which == CMC_GLOBAL_R) return star_r;
    if (which == CMC_GLOBAL_M) return star_m;
    if (which == CMC_GLOBAL_PHI) return star_phi;
    shared_fail("invalid array selector");
    return NULL;
}

#ifdef __linux__
static double *published(int which)
{
    return (double *)(shared_mapping + (size_t)which*stride);
}

/* Refresh ONLY the range overwritten by the original communication/computation.
 * Preserve private updates outside it, including new particles above N_MAX and
 * sentinels. Boundary pages are preserved byte-for-byte outside [first,last].
 * Mapping replacement happens at an existing quiescent point, never mid-step. */
static void refresh(int which, long first, long last)
{
    if (last < first) return;
    size_t begin = (size_t)first*sizeof(double);
    size_t end = ((size_t)last+1)*sizeof(double);
    size_t page_begin = begin/page_bytes*page_bytes;
    size_t page_end = (end+page_bytes-1)/page_bytes*page_bytes;
    if (first < 0 || last >= N_STAR_DIM || page_end > stride)
        shared_fail("refresh outside array capacity");
    char *base = (char *)view(which);
    size_t head = begin-page_begin, tail = page_end-end;
    char *edges = malloc(head+tail+1);
    if (!edges) shared_fail("boundary buffer allocation failed");
    memcpy(edges, base+page_begin, head);
    memcpy(edges+head, base+end, tail);
    void *p = mmap(base+page_begin, page_end-page_begin, PROT_READ|PROT_WRITE,
                   MAP_PRIVATE|MAP_FIXED, backing_fd, (off_t)((size_t)which*stride+page_begin));
    if (p == MAP_FAILED) shared_fail("cannot refresh private view");
    memcpy(base+page_begin, edges, head);
    memcpy(base+end, edges+head, tail);
    free(edges);
}
#endif

void cmc_globals_allocate(void)
{
    const char *option = getenv("CMC_SHARED_GLOBALS");
#ifdef __linux__
    enabled = 1;
#else
    enabled = 0;
#endif
    if (option) {
        if (!strcmp(option,"0")) enabled = 0;
        else if (!strcmp(option,"1")) enabled = 1;
        else shared_fail("CMC_SHARED_GLOBALS must be 0 or 1");
    }
    int low, high;
    MPI_Allreduce(&enabled,&low,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    MPI_Allreduce(&enabled,&high,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    if (low != high) shared_fail("CMC_SHARED_GLOBALS differs between ranks");
    if (!enabled) {
        star_r = malloc((size_t)N_STAR_DIM*sizeof(double));
        star_m = malloc((size_t)N_STAR_DIM*sizeof(double));
        star_phi = malloc((size_t)N_STAR_DIM*sizeof(double));
        if (!star_r || !star_m || !star_phi) shared_fail("replicated allocation failed");
        if (!myid) fprintf(stderr,"global arrays: replicated mode\n");
        return;
    }
#ifndef __linux__
    shared_fail("shared COW arrays require Linux; use CMC_SHARED_GLOBALS=0");
#else
    MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,myid,MPI_INFO_NULL,&node_comm);
    MPI_Comm_rank(node_comm,&node_rank);
    /* Testing only: emulate noncontiguous node groups on one physical host. */
    option = getenv("CMC_SHARED_TEST_GROUPS");
    if (option) {
        char *end; long groups = strtol(option,&end,10);
        if (!*option || *end || groups < 1 || groups > INT_MAX)
            shared_fail("invalid CMC_SHARED_TEST_GROUPS");
        MPI_Comm split;
        MPI_Comm_split(node_comm,node_rank % (int)groups,myid,&split);
        MPI_Comm_free(&node_comm); node_comm=split;
        MPI_Comm_rank(node_comm,&node_rank);
    }
    MPI_Comm_size(node_comm,&node_size);
    MPI_Comm_split(MPI_COMM_WORLD,node_rank == 0 ? 0 : MPI_UNDEFINED,myid,&leaders_comm);
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0 || N_STAR_DIM <= 0 ||
        (uint64_t)N_STAR_DIM > (SIZE_MAX/3-(size_t)page)/sizeof(double))
        shared_fail("invalid shared allocation size");
    page_bytes=(size_t)page;
    stride=((size_t)N_STAR_DIM*sizeof(double)+page_bytes-1)/page_bytes*page_bytes;
    mapping_bytes=3*stride;
    if (mapping_bytes > INT64_MAX) shared_fail("backing size exceeds off_t");
    char name[128]={0};
    if (!node_rank) {
        snprintf(name,sizeof(name),"/cmc-globals-%lu-%ld-%d",(unsigned long)getuid(),(long)getpid(),myid);
        backing_fd=shm_open(name,O_CREAT|O_EXCL|O_RDWR,0600);
        if (backing_fd < 0) shared_fail("cannot create POSIX shared memory");
        struct statvfs space;
        if (fstatvfs(backing_fd,&space) ||
            (long double)space.f_bavail*space.f_frsize < mapping_bytes) {
            shm_unlink(name);
            shared_fail("insufficient /dev/shm space; enlarge it or use CMC_SHARED_GLOBALS=0");
        }
        if (ftruncate(backing_fd,(off_t)mapping_bytes)) {
            shm_unlink(name);
            shared_fail("cannot size shared backing");
        }
    }
    MPI_Bcast(name,sizeof(name),MPI_CHAR,0,node_comm);
    if (node_rank) backing_fd=shm_open(name,O_RDWR,0600);
    if (backing_fd < 0) shared_fail("cannot open node shared backing");
    MPI_Barrier(node_comm);
    if (!node_rank && shm_unlink(name)) shared_fail("cannot unlink backing name");
    /* Reserve only after unlink: even an OOM kill during reservation cannot leave
     * a named multi-GB object on the compute node. */
    if (!node_rank) {
        int status=posix_fallocate(backing_fd,0,(off_t)mapping_bytes);
        if (status) {
            errno=status;
            shared_fail("cannot reserve shared backing (check /dev/shm and job memory limits)");
        }
    }
    MPI_Barrier(node_comm);
    /* The unlinked object lives until the last descriptor/mapping is closed. */
    private_mapping=mmap(NULL,mapping_bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE,backing_fd,0);
    if (private_mapping == MAP_FAILED) shared_fail("cannot map private COW view");
    if (!node_rank) {
        shared_mapping=mmap(NULL,mapping_bytes,PROT_READ|PROT_WRITE,MAP_SHARED,backing_fd,0);
        if (shared_mapping == MAP_FAILED) shared_fail("cannot map published snapshot");
        fprintf(stderr,"rank %d: node-shared globals: %d ranks, %.3f MiB backing total; private COW updates\n",
                myid,node_size,mapping_bytes/1048576.0);
    }
    star_r=(double *)private_mapping;
    star_m=(double *)(private_mapping+stride);
    star_phi=(double *)(private_mapping+2*stride);
#endif
}

void cmc_globals_gather(int which, const double *local, const int *counts, const int *displacements)
{
    double *array=view(which);
    if (!enabled) {
        if (local == array+displacements[myid])
            MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DOUBLE,array,counts,displacements,MPI_DOUBLE,MPI_COMM_WORLD);
        else
            MPI_Allgatherv(local,counts[myid],MPI_DOUBLE,array,counts,displacements,MPI_DOUBLE,MPI_COMM_WORLD);
        return;
    }
#ifdef __linux__
    /* Snapshot the send slice before anyone changes shared backing pages. */
    if (counts[myid] < 0) shared_fail("negative local gather count");
    size_t bytes=(size_t)counts[myid]*sizeof(double);
    double *send=malloc(bytes ? bytes : sizeof(double));
    if (!send) shared_fail("gather staging allocation failed");
    memcpy(send,local,bytes);
    long first=LONG_MAX,last=-1;
    for (int rank=0;rank<procs;++rank) {
        if (counts[rank] < 0 || displacements[rank] < 0 ||
            (long)displacements[rank]+counts[rank] > N_STAR_DIM)
            shared_fail("gather range exceeds capacity");
        if (counts[rank]) {
            if (displacements[rank]<first) first=displacements[rank];
            if ((long)displacements[rank]+counts[rank]-1>last)
                last=(long)displacements[rank]+counts[rank]-1;
        }
    }
    /* CMC uses a contiguous global prefix, even when node membership isn't contiguous. */
    long count_sum=0;
    for (int rank=0;rank<procs;++rank) count_sum+=counts[rank];
    if (last >= first && count_sum != last-first+1) shared_fail("noncontiguous publication range");
    MPI_Barrier(node_comm);
    MPI_Gatherv(send,counts[myid],MPI_DOUBLE,myid==0 ? published(which) : NULL,
                counts,displacements,MPI_DOUBLE,0,MPI_COMM_WORLD);
    free(send);
    if (!node_rank && last >= first)
        MPI_Bcast(published(which)+first,(int)(last-first+1),MPI_DOUBLE,0,leaders_comm);
    MPI_Barrier(node_comm);
    if (last >= first) refresh(which,first,last);
    MPI_Barrier(node_comm);
#endif
}

void cmc_globals_potential(void)
{
    if (!enabled) { potential_calculate(); return; }
#ifdef __linux__
    /* No mid-step private values may be published here: mass/radius snapshots
     * have already been collected by initialization/post_sort_comm. */
    MPI_Barrier(node_comm);
    if (!node_rank) {
        double *r=star_r,*m=star_m,*phi=star_phi;
        star_r=published(CMC_GLOBAL_R); star_m=published(CMC_GLOBAL_M);
        star_phi=published(CMC_GLOBAL_PHI);
        potential_calculate();
        star_r=r; star_m=m; star_phi=phi;
    }
    MPI_Bcast(&clus.N_MAX,1,MPI_LONG,0,node_comm);
    double state[4];
    if (!node_rank) {
        state[0]=cenma.m; state[1]=cenma.m_new; state[2]=Mtotal; state[3]=Rtidal;
    }
    MPI_Bcast(state,4,MPI_DOUBLE,0,node_comm);
    cenma.m=state[0]; cenma.m_new=state[1]; Mtotal=state[2]; Rtidal=state[3];
    MPI_Barrier(node_comm);
    refresh(CMC_GLOBAL_PHI,0,clus.N_MAX+1);
    /* Original potential_calculate changes only this radius sentinel. */
    star_r[clus.N_MAX+1]=SF_INFINITY;
    MPI_Barrier(node_comm);
#endif
}

void cmc_globals_free(void)
{
    if (!enabled) {
        free(star_r); free(star_m); free(star_phi);
    }
#ifdef __linux__
    else {
        if (private_mapping && private_mapping != MAP_FAILED) munmap(private_mapping,mapping_bytes);
        if (shared_mapping && shared_mapping != MAP_FAILED) munmap(shared_mapping,mapping_bytes);
        if (backing_fd >= 0) close(backing_fd);
        private_mapping=NULL; shared_mapping=NULL; backing_fd=-1;
    }
#endif
    star_r=star_m=star_phi=NULL;
}

void cmc_globals_finalize(void)
{
    if (leaders_comm != MPI_COMM_NULL) MPI_Comm_free(&leaders_comm);
    if (node_comm != MPI_COMM_NULL) MPI_Comm_free(&node_comm);
}
