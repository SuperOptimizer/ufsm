/* Analytic 3D ramps verify interpolation and registration independently of
   learned predictions. Round trips/Jacobians check the sparse supervision map. */
#include "../src/sample.c"
#include <assert.h>
void *nn_host_alloc(size_t n) { return malloc(n); }
void nn_host_free(void *p) { free(p); }

int main(void) {
    const int P=32; const size_t n=(size_t)P*P*P; const int64_t origin[3]={0};
    uint8_t *ct=malloc(n),*target=calloc(NCH*n,1),*mask=malloc(n),*T=malloc(NCH*n),*M=malloc(n);
    float *X=malloc(4*n*4),*scratch=malloc(n*4),cy[32],cx[32],row[32],table[192];
    uint16_t *H=malloc(4*n*2);
    double knots[2][5]={{0,-20,-15,10,-2},{31,-16.9,-8.8,10,-1.69}},params[32][4];
    sheet_dataset sheet={.nk=2,.knots=knots,.center=4,.scale=20};
    for (int z=0;z<P;z++) {
        sheet_parameters(&sheet,z,params[z]); cy[z]=params[z][0]; cx[z]=params[z][1];
        for (int y=0;y<P;y++) for (int x=0;x<P;x++) { size_t k=((size_t)z*P+y)*P+x; ct[k]=60+z+2*y+3*x; target[k]=z+2*y+3*x; mask[k]=z!=0; }
    }
    for (int mode=0;mode<5;mode++) {   /* 0 rotation, 1 elastic, 2 both, 3 both + zoom-in 0.8, 4 both + anisotropic scale/shear */
        spatial_aug a,b; spatial_aug_make(&a,17,P,mode==1?0:5,1,mode==0?0:1,1);
        spatial_aug_make(&b,17,P,mode==1?0:5,1,mode==0?0:1,1); assert(!memcmp(&a,&b,sizeof a));
        if (mode==3) spatial_aug_zoom(&a,.8);
        if (mode==4) { spatial_aug_affine(&a,99,.2,.12); assert(a.det>.5 && a.det<2); }
        spatial_aug_tables(&a,table);
        for (int d=0;d<3;d++) for (int u=0;u<P;u++) assert(fabs(table[(3+d)*P+u])<=.090001);
        int cropped=0,checked=0;
        for (int si=0;si<48;si++) {
            sym sy=sym_of(si); rng r; rseed(&r,2);
            write_spatial(ct,target,mask,T,M,P,origin,0,1,1,cy,cx,sy,&a,1,0,0,&r,row,0,X,nullptr,nullptr,nullptr,nullptr,&sheet,params,scratch);
            rseed(&r,2);
            write_spatial(ct,target,mask,T,M,P,origin,0,1,1,cy,cx,sy,&a,1,0,0,&r,row,1,nullptr,H,nullptr,nullptr,nullptr,&sheet,params,scratch);
            for (int z=0;z<P;z++) for (int y=0;y<P;y++) for (int x=0;x<P;x++) {
                size_t k=((size_t)z*P+y)*P+x; float pulled[3],jac[3]; spatial_aug_pull(&a,table,z,y,x,pulled,jac);
                double world[3]; int inside=1;
                for (int d=0;d<3;d++) world[sy.perm[d]]=sy.flip[d]?P-1-pulled[d]:pulled[d];
                for (int d=0;d<3;d++) inside &= world[d]>=0 && world[d]<=P-1;
                if (!inside) { assert(!M[k] && X[k]==0 && T[k]==0); cropped++; continue; }
                double ramp=world[0]+2*world[1]+3*world[2];
                assert(fabs(X[k]-(60+ramp))<6e-5 && fabs(T[k]-ramp)<.501);
                float prior=(sheet_reference(&sheet,world)-sheet.center)/sheet.scale;
                assert(fabs(X[n+k]-prior)<1e-6);
                _Float16 h; memcpy(&h,H+n+k,2); assert(fabs((float)h-prior)<2e-4);
                /* A source location contributes to all labels and the q0 scalar
                   at the same dense voxel; float loss prior has no sign flip. */
                checked++;
                if (world[0]<1-1e-5) assert(!M[k]);
            }
        }
        double source[3]={11.25,17.2,13.6},out[3]; spatial_aug_forward(&a,source,out);
        double v[3],recovered[3],center=.5*(P-1);
        for (int d=0;d<3;d++) v[d]=out[d]-center-a.amplitude[d]*sin(a.frequency*out[d]+a.phase[d]);
        double w[3]; for (int d=0;d<3;d++) { w[d]=0; for (int j=0;j<3;j++) w[d]+=a.R[j][d]*v[j]; }
        for (int d=0;d<3;d++) { recovered[d]=center; for (int j=0;j<3;j++) recovered[d]+=a.scale*(a.affine?a.A[d][j]:j==d)*w[j]; assert(fabs(recovered[d]-source[d])<1e-10); }
        sheet_point points[4]={{{11.25,17.2,13.6},3,.5},{{13,15,17},3.2,.7},{{11.25,17.2,13.6},3,.5},{{-100,0,0},4,.8}};
        sheet_term terms[2]={{2,2,0,1,.2},{3,2,2,1,0}};
        sheet_batch batch={.np=4,.nt=2,.points=points,.terms=terms}; memset(M,1,n);
        warp_sheet_batch(&batch,&a,M);
        assert(batch.nt==1 && batch.np==2 && batch.points[0].q==3 && batch.points[0].q0==.5);
        for (int d=0;d<3;d++) assert(fabs(batch.points[0].xyz[d]-out[d])<2e-6);
        assert(batch.terms[0].target==.2f && batch.points[1].q==3.2f);
        printf("spatial mode %d: %d registered voxels, %d ignored corners, 48 symmetries, sparse round trip: ok\n",mode,checked,cropped);
    }
    free(ct); free(target); free(mask); free(T); free(M); free(X); free(H); free(scratch);
    puts("spatial augmentation registration: ok"); return 0;
}
