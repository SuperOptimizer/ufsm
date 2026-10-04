#include "target_erode.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    enum {P=9,Q=P+2};
    uint8_t input[Q*Q*Q], output[P*P*P];
    /* A three-voxel sheet becomes one voxel and stays at the crop faces. */
    for (int z=0;z<Q;z++) for (int y=0;y<Q;y++) for (int x=0;x<Q;x++)
        input[(z*Q+y)*Q+x]=x>=4 && x<=6 ? 254:0;
    target_erode1(input,output,P);
    for (int z=0;z<P;z++) for (int y=0;y<P;y++) for (int x=0;x<P;x++)
        assert(output[(z*P+y)*P+x]==(x==4?254:0));
    /* Thin bridges disappear while thicker regions retain their centre. */
    memset(input,0,sizeof input);
    for (int z=2;z<=8;z++) for (int y=2;y<=8;y++) for (int x=2;x<=8;x++)
        if (x<=4 || x>=6 || (y==5 && z==5)) input[(z*Q+y)*Q+x]=254;
    target_erode1(input,output,P);
    assert(output[(4*P+4)*P+4]==0);
    assert(output[(4*P+4)*P+2]==254 && output[(4*P+4)*P+6]==254);
    puts("target erosion: sheet centre, crop halo and thin bridges ok");
    return 0;
}
