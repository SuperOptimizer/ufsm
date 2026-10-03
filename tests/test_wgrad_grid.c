/* FP16 weight gradients with >65535 spatial blocks, plus channel/bias checks.
   Batched tensors reach the launch limit with under 600 MiB of VRAM. */
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int trial(shape5 xs,int co,int stride) {
    shape5 ys=nn_conv3d_out_shape(xs,co,3,stride);
    size_t nx=shape_numel(xs),ny=shape_numel(ys),nw=(size_t)xs.c*co*27;
    _Float16 *ones=malloc((nx>ny?nx:ny)*2);
    if(!ones) return 1;
    for(size_t i=0;i<(nx>ny?nx:ny);i++) ones[i]=1;
    float *x=nn_malloc(nx*2),*gy=nn_malloc(ny*2),*gw=nn_malloc(nw*4),*gb=nn_malloc(co*4);
    if(!x || !gy || !gw || !gb) return 1;
    nn_h2d(x,ones,nx*2);nn_h2d(gy,ones,ny*2);free(ones);
    nn_zero(gw,nw*4);nn_zero(gb,co*4);
    nn_conv3d_bwd_weight(x,xs,gy,ys,3,stride,gw,gb);
    float *weights=malloc(nw*4),*bias=malloc(co*4);
    nn_d2h(weights,gw,nw*4);nn_d2h(bias,gb,co*4);
    const char *error=nn_check();int failed=error!=NULL;
    if(error) fprintf(stderr,"stride %d: %s\n",stride,error);
    for(int c=0;c<co;c++) {
        double expected=(double)ys.n*ys.d*ys.h*ys.w;
        if(fabs(bias[c]/expected-1)>1e-4) failed=1;
        for(int ci=0;ci<xs.c;ci++) for(int z=0;z<3;z++) for(int y=0;y<3;y++) for(int v=0;v<3;v++) {
            double count=(double)ys.n;
            int offsets[3]={z-1,y-1,v-1},lengths[3]={ys.d,ys.h,ys.w};
            for(int d=0;d<3;d++) count*=lengths[d]-(stride==1?abs(offsets[d]):offsets[d]==-1);
            size_t k=((size_t)c*xs.c+ci)*27+z*9+y*3+v;
            if(fabs(weights[k]/count-1)>1e-4) failed=1;
        }
    }
    free(weights);free(bias);nn_free(x);nn_free(gy);nn_free(gw);nn_free(gb);
    printf("FP16 wgrad stride %d, N%d C%d %dx%dx%d -> C%d: %s\n",stride,xs.n,xs.c,xs.d,xs.h,xs.w,co,failed?"FAIL":"ok");
    return failed;
}
int main(void) {
    if(nn_init(0)) return 1;
    nn_set_prec(1);nn_set_f16(1);nn_set_grad_bf16(1);nn_set_grad_scale(1);
    if(nn_set_prec_policy("")) return 1;
    int failed=trial((shape5){1,17,8,8,8},17,1);
    failed|=trial((shape5){1,17,8,8,8},17,2);
    failed|=trial((shape5){48,1,128,128,128},1,1);
    failed|=trial((shape5){128,1,128,128,128},1,2);
    return failed;
}
