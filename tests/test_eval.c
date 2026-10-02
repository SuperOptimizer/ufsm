/* Exact dilation against a scatter reference, including volumes thinner than the radius. */
#include "../src/eval.c"
#include <assert.h>
static void reference(const uint8_t *in, uint8_t *out, const int64_t n[3], int tol) {
    memcpy(out, in, (size_t)n[0]*n[1]*n[2]);
    for (int64_t z=0; z<n[0]; z++) for (int64_t y=0; y<n[1]; y++) for (int64_t x=0; x<n[2]; x++) {
        if (!in[(z*n[1]+y)*n[2]+x]) continue;
        for (int dz=-tol; dz<=tol; dz++) for (int dy=-tol; dy<=tol; dy++) for (int dx=-tol; dx<=tol; dx++) {
            if (dz*dz+dy*dy+dx*dx>tol*tol) continue;
            int64_t zz=z+dz, yy=y+dy, xx=x+dx;
            if (zz>=0 && zz<n[0] && yy>=0 && yy<n[1] && xx>=0 && xx<n[2]) out[(zz*n[1]+yy)*n[2]+xx]=1;
        }
    }
}
int main(void) {
    unsigned state=1234; int cases=0;
    for (int z=1; z<=12; z++) for (int y=1; y<=9; y++) for (int x=1; x<=15; x++) for (int tol=0; tol<=3; tol++) {
        int64_t n[3]={z,y,x}; size_t nv=(size_t)z*y*x;
        uint8_t a[nv], b[nv], c[nv];
        for (size_t i=0;i<nv;i++) {state=state*1664525+1013904223; a[i]=(state>>24)<64;}
        reference(a,b,n,tol); dilate(a,c,n,tol); assert(!memcmp(b,c,nv)); cases++;
    }
    printf("dilation: %d boundary, thin-volume and radius cases bit-exact\n",cases);
}
