/* Small CUDA contract test: under 1 MiB of tensors, no model allocation. */
#include "nn.h"
#include "split.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const float xyz[9]={3.5f,2.25f,4.75f,1.1f,1.2f,1.3f,6.1f,4.2f,2.3f};
typedef struct { float *lg[2]; double values[2][6]; } fixture;
static void split_job(int side,void *arg) {
    fixture *f=arg; nn_init(side); nn_split_cfg(side?1:0,side?0:1,5,8);
    nn_sheet_gather(f->lg[side],(shape5){1,2,5,8,8},xyz,3,side?3:0,side?4:0,side?8:4,f->values[side]);
}
int main(void) {
    const int P=8; const size_t S=512;
    assert(!nn_init(0));
    if (nn_mem_free() < ((size_t)128<<20)) { fputs("insufficient CUDA headroom\n",stderr); return 77; }
    float host[1024];
    for (int z=0;z<P;z++) for (int y=0;y<P;y++) for (int x=0;x<P;x++) {
        size_t k=((size_t)z*P+y)*P+x;
        host[k]=2*z+.3f*y-.2f*x; host[S+k]=.01f*z+.1f*y+.05f*x;
    }
    float *lg=nn_malloc(sizeof host); assert(lg); nn_h2d(lg,host,sizeof host); double values[6];
    nn_split_cfg(0,0,0,0); nn_sheet_gather(lg,(shape5){1,2,8,8,8},xyz,3,0,0,8,values);
    for (int i=0;i<3;i++) { assert(fabs(values[2*i]-(2*xyz[3*i]+.3*xyz[3*i+1]-.2*xyz[3*i+2]))<2e-6); }
    uint64_t indices[3]={1,7,S+9}; float additions[3]={.25f,-.1f,.02f};
    float *gl=nn_malloc(sizeof host); nn_zero(gl,sizeof host); nn_sheet_scatter(gl,indices,additions,3,0); nn_d2h(host,gl,sizeof host);
    for (int i=0;i<3;i++) assert(fabs(host[indices[i]]-additions[i])<1e-7);
    nn_set_f16(1); nn_set_grad_scale(1024);
    nn_zero(gl,sizeof host); nn_sheet_scatter(gl,indices,additions,3,1);
    uint16_t half[1024]; nn_d2h(half,gl,sizeof half);
    for (int i=0;i<3;i++) { _Float16 h; memcpy(&h,half+indices[i],2); assert(fabs((float)h/1024-additions[i])<1e-4); }
    fixture f={0};
    for (int side=0;side<2;side++) {
        assert(!nn_init(side)); float local[640];
        for (int z=0;z<5;z++) for (int y=0;y<8;y++) for (int x=0;x<8;x++) {
            int gz=z+(side?3:0); size_t k=((size_t)z*8+y)*8+x;
            local[k]=2*gz+.3f*y-.2f*x; local[320+k]=.01f*gz+.1f*y+.05f*x;
        }
        f.lg[side]=nn_malloc(sizeof local); nn_h2d(f.lg[side],local,sizeof local);
    }
    split_ctx *ctx=split_create(0,1); assert(ctx); split_run(ctx,split_job,&f);
    for (int side=0;side<2;side++) for (int i=0;i<6;i++) assert(fabs(f.values[side][i]-values[i])<2e-6);
    split_free(ctx); for (int side=0;side<2;side++) { nn_init(side); nn_free(f.lg[side]); }
    nn_init(0); nn_free(lg); nn_free(gl); assert(!nn_check());
    float rows[32]; for (int z=0;z<P;z++) { rows[4*z]=40; rows[4*z+1]=50; rows[4*z+2]=.1f; rows[4*z+3]=-2; }
    float *input=nn_malloc(4*S*4); nn_zero(input,4*S*4);
    nn_sheet_input(input,P,rows,3,10,0); nn_d2h(host,input+S,S*4);
    assert(fabs(host[0]-((hypot(40,50)*.1-2)-3)/10)<1e-6);
    uint8_t ct[512]; memset(ct,1,sizeof ct); ct[5]=0;
    uint8_t *ctd=nn_malloc(sizeof ct); nn_h2d(ctd,ct,sizeof ct);
    nn_zero(input,4*S*4); nn_sheet_gate(input+S,input,ctd,P,rows); nn_d2h(host,input+S,S*4);
    assert(isnan(host[5])); assert(fabs(host[0]-(hypot(40,50)*.1-2))<1e-6);
    nn_free(input); nn_free(ctd); assert(!nn_check());
    puts("sheet CUDA: gather, FP32/FP16 scatter and cross-GPU boundary interpolation: ok"); return 0;
}
