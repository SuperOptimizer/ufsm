#include "sheet.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc,char **argv) {
    if (argc!=2) return 2;
    sheet_dataset *s=sheet_load(argv[1]); assert(s);
    const int64_t origin[3]={0,0,0}; const int perm[3]={0,2,1},flip[3]={0,1,0};
    sheet_batch *b=sheet_sample(s,origin,32,perm,flip,nullptr,nullptr,nullptr,2); assert(b && b->np && b->nt);
    uint8_t *target=malloc(32*32*32); memset(target,255,32*32*32);
    sheet_batch *blocked=sheet_sample(s,origin,32,perm,flip,nullptr,target,nullptr,2);
    for (size_t t=0;t<blocked->nt;t++) assert(blocked->terms[t].kind!=4);
    sheet_batch_free(blocked); free(target);
    uint8_t *ignore=calloc(32*32*32,1); sheet_mask_contacts(s,origin,32,ignore);
    if (s->nc) { const float *c=s->contacts[0]; assert(ignore[((int)c[0]*32+(int)c[1])*32+(int)c[2]]); }
    uint8_t *mask=malloc(32*32*32); for (int i=0;i<32*32*32;i++) mask[i]=!ignore[i];
    blocked=sheet_sample(s,origin,32,perm,flip,nullptr,nullptr,mask,2);
    assert(blocked->nt<b->nt); sheet_batch_free(blocked);
    memset(mask,0,32*32*32); blocked=sheet_sample(s,origin,32,perm,flip,nullptr,nullptr,mask,2);
    assert(blocked->nt==0); sheet_batch_free(blocked); free(mask); free(ignore);
    sheet_batch *copy=sheet_clone(b); assert(copy && copy->np==b->np);
    double *v=calloc(2*b->np,sizeof *v),*g=calloc(2*b->np,sizeof *g),parts[5];
    for (size_t i=0;i<b->np;i++) { v[2*i]=.3; v[2*i+1]=b->points[i].q+.2; }
    double loss=sheet_loss(b,v,g,parts,1,2); assert(isfinite(loss) && loss>0);
    for (size_t i=0;i<2*b->np;i++) {
        double old=v[i],step=1e-6; v[i]=old+step;
        double *tmp=malloc(2*b->np*sizeof *tmp),p[5]; double hi=sheet_loss(b,v,tmp,p,1,2);
        v[i]=old-step; double lo=sheet_loss(b,v,tmp,p,1,2); v[i]=old;
        assert(fabs((hi-lo)/(2*step)-g[i])<1e-6); free(tmp);
    }
    free(v); free(g); sheet_batch_free(copy); sheet_batch_free(b); sheet_free(s);
    sheet_record rows[2]={0}; sheet_dataset wide={.nr=2,.records=rows,.max_soft_sigma=2};
    for (int i=0;i<2;i++) { rows[i].kind=2; rows[i].count=2; rows[i].points[0][0]=rows[i].points[1][0]=10+i*30; rows[i].points[0][1]=rows[i].points[1][1]=10; rows[i].points[0][2]=10; }
    rows[0].points[1][2]=25; rows[1].points[1][2]=45;
    assert(sheet_widen_bands(&wide,NAN)); assert(sheet_widen_bands(&wide,.5)); assert(sheet_widen_bands(&wide,5));
    assert(sheet_widen_bands(&wide,1)==0 && wide.nc==0);
    assert(sheet_widen_bands(&wide,2.5)==0 && wide.max_soft_sigma==5 && wide.nc==1);
    assert(wide.contacts[0][0]==10 && wide.contacts[0][2]==17.5f && wide.contacts[0][3]==20);
    free(wide.contacts);
    puts("sheet geometry native contract + finite-difference gradients: ok"); return 0;
}
