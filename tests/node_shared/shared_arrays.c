/* Standalone storage/communication tests, independent of CMC dynamics. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#ifdef __linux__
#include <malloc.h>
#endif
#include "cmc.h"
#define _MAIN_
#include "cmc_vars.h"

static void check(int ok, const char *message)
{
    if (!ok) {
        fprintf(stderr,"rank %d: TEST FAILED: %s\n",myid,message);
        MPI_Abort(MPI_COMM_WORLD,2);
    }
}

/* A deterministic potential stand-in exercises publication and scalar state. */
long potential_calculate(void)
{
    Mtotal=0;
    for (long i=1;i<=clus.N_MAX;++i) {
        Mtotal+=star_m[i];
        star_phi[i]=-(star_r[i]+star_m[i]);
    }
    cenma.m+=cenma.m_new; cenma.m_new=0;
    Rtidal=Mtotal/3;
    star_phi[0]=cenma.m;
    star_phi[clus.N_MAX+1]=0;
    star_r[clus.N_MAX+1]=SF_INFINITY;
    return clus.N_MAX;
}

static long pss_kib(void)
{
    FILE *f=fopen("/proc/self/smaps_rollup","r");
    if (!f) return -1;
    char line[256]; long value=-1;
    while(fgets(line,sizeof(line),f))
        if(sscanf(line,"Pss: %ld kB",&value)==1) break;
    fclose(f);
    return value;
}

int main(int argc,char **argv)
{
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid); MPI_Comm_size(MPI_COMM_WORLD,&procs);
    long n=argc>1 ? atol(argv[1]) : 10003;
    check(n>procs && n < INT_MAX-128,"test size");
    N_STAR_DIM=n+65; clus.N_STAR=n; clus.N_MAX=n;
    int *counts=calloc(procs,sizeof(int)),*disp=calloc(procs,sizeof(int));
    long start=1;
    for(int rank=0;rank<procs;++rank) {
        counts[rank]=n/procs+(rank<n%procs);
        disp[rank]=start; start+=counts[rank];
    }
    cmc_globals_allocate();
    for(long i=disp[myid];i<disp[myid]+counts[myid];++i) {
        star_r[i]=i+1; star_m[i]=i;
    }
    star_r[0]=ZERO; star_m[0]=0;
    cmc_globals_gather(CMC_GLOBAL_R,star_r+disp[myid],counts,disp);
    cmc_globals_gather(CMC_GLOBAL_M,star_m+disp[myid],counts,disp);
    cenma.m=3; cenma.m_new=.5;
    cmc_globals_potential();
    double sum=0;
    for(long i=1;i<=n;++i) {
        check(star_r[i]==i+1 && star_m[i]==i && star_phi[i]==-(2*i+1),"published initial state");
        sum+=star_r[i]+star_m[i]+star_phi[i];
    }
    check(sum==0 && cenma.m==3.5 && cenma.m_new==0,"leader scalar broadcast");
#ifdef __linux__
    malloc_trim(0);
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    long pss=pss_kib(),total=0;
    MPI_Reduce(&pss,&total,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);
    if(!myid) printf("MEMORY shared=%d ranks=%d N=%ld sum_pss_kib=%ld\n",cmc_globals_shared(),procs,n,total);

    /* All ranks write the SAME offsets differently. No other rank may see it. */
    star_m[0]=500+myid; star_m[n+2]=700+myid; star_phi[n+2]=900+myid;
    star_phi[1]=-100-myid;
    for(long i=disp[myid];i<disp[myid]+counts[myid];++i) star_m[i]*=2;
    star_m[1]=100000+myid;
    MPI_Barrier(MPI_COMM_WORLD);
    check(star_m[1]==100000+myid && star_phi[1]==-100-myid,"rank-private COW isolation");
    cmc_globals_gather(CMC_GLOBAL_M,star_m+disp[myid],counts,disp);
    for(long i=1;i<=n;++i) check(star_m[i]==(i==1 ? 100000 : 2*i),"owner publication");
    check(star_m[0]==500+myid && star_m[n+2]==700+myid,"boundary-page preservation");
    check(star_phi[1]==-100-myid && star_phi[n+2]==900+myid,"other arrays unchanged by mass sync");
    cmc_globals_potential();
    check(star_phi[1]==-100002 && star_phi[n+2]==900+myid,"potential refresh and tail preservation");
    /* Model a shrinking global prefix after escape/removal. */
    n-=3; clus.N_MAX=n;
    start=1;
    for(int rank=0;rank<procs;++rank) {
        counts[rank]=n/procs+(rank<n%procs);disp[rank]=start;start+=counts[rank];
    }
    for(long i=disp[myid];i<disp[myid]+counts[myid];++i) {star_m[i]=i;star_r[i]=i+1;}
    star_m[n+2]=777+myid;
    cmc_globals_gather(CMC_GLOBAL_R,star_r+disp[myid],counts,disp);
    cmc_globals_gather(CMC_GLOBAL_M,star_m+disp[myid],counts,disp);
    check(star_m[n+2]==777+myid,"shrinking-prefix tail preservation");
    cmc_globals_potential();
    for(long i=1;i<=n;++i) check(star_phi[i]==-(2*i+1),"potential after repartition");
    cmc_globals_free(); cmc_globals_finalize();
    free(counts);free(disp);
    if(!myid) puts("SHARED ARRAY TESTS PASSED");
    MPI_Finalize();
    return 0;
}
