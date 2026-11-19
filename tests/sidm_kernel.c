#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include "cmc_sidm_kernel.h"
static uint64_t state=17;
static double uniform(void) {
    state=state*6364136223846793005ULL+1442695040888963407ULL;
    return (state>>11)*0x1.0p-53;
}
int main(void) {
    double mean=0, second=0;
    for (int i=0; i<200000; ++i) {
        double a[3]={1.2,-3.1,2.}, b[3]={-.5,.3,-1.};
        double momentum[3], energy=0, after=0;
        for(int d=0;d<3;d++){momentum[d]=a[d]+b[d];energy+=a[d]*a[d]+b[d]*b[d];}
        double u=uniform(),v=uniform();
        sidm_elastic_equal(a,b,u,v);
        double speed2=0;
        for(int d=0;d<3;d++){
            assert(fabs(a[d]+b[d]-momentum[d])<2e-14);
            after+=a[d]*a[d]+b[d]*b[d];
            speed2+=(a[d]-b[d])*(a[d]-b[d]);
        }
        assert(fabs(energy-after)<5e-14);
        double mu=(a[2]-b[2])/sqrt(speed2);mean+=mu;second+=mu*mu;
    }
    assert(fabs(mean/200000)<.005);
    assert(fabs(second/200000-1./3)<.005);
    double zero[3]={1,2,3}, same[3]={1,2,3};
    sidm_elastic_equal(zero,same,.2,.9);
    assert(zero[0]==1 && zero[1]==2 && zero[2]==3);
    /* Expected total sampled rate equals the sum over unordered physical
       pairs, including the odd-cell idle-particle correction. */
    for(int n=2;n<66;n++) assert((n/2)*sidm_matching_weight(n)==n*(n-1)/2.);
    puts("PASS elastic energy/momentum, isotropy, zero speed, even/odd rate normalization");
}
