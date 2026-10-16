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
            write_spatial(ct,target,mask,T,M,P,origin,0,1,1,cy,cx,sy,&a,1,0,0,&r,row,0,X,nullptr,nullptr,nullptr,nullptr,&sheet,params,scratch,0,0);
            rseed(&r,2);
            write_spatial(ct,target,mask,T,M,P,origin,0,1,1,cy,cx,sy,&a,1,0,0,&r,row,1,nullptr,H,nullptr,nullptr,nullptr,&sheet,params,scratch,0,0);
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
    {   /* the stored phase through the same warps (warp_phase, task unwrap): a planar phase ramp of 0.02 / 0.03 / 0.025 turns per voxel
           pulled through each mode and every 6th symmetry equals the analytic phase at the pulled source point (phasor interpolation of
           a linear ramp: < 0.01 turn), unknown where the pull leaves the window */
        uint8_t *ph=malloc(n),*out=malloc(n); const double g[3]={.02,.03,.025};
        for (int z=0;z<P;z++) for (int y=0;y<P;y++) for (int x=0;x<P;x++) { double q=g[0]*z+g[1]*y+g[2]*x; q-=floor(q); ph[((size_t)z*P+y)*P+x]=(uint8_t)((int)lround(q*PHASE_PERIOD)%PHASE_PERIOD); }
        for (int mode=0;mode<5;mode++) {
            spatial_aug a; spatial_aug_make(&a,23,P,mode==1?0:5,1,mode==0?0:1,1);
            if (mode==3) spatial_aug_zoom(&a,.8);
            if (mode==4) spatial_aug_affine(&a,99,.2,.12);
            spatial_aug_tables(&a,table);
            double worst=0; int known=0,outside=0;
            for (int si=0;si<48;si+=6) {
                sym sy=sym_of(si); warp_phase(ph,out,P,sy,&a);
                for (int z=0;z<P;z++) for (int y=0;y<P;y++) for (int x=0;x<P;x++) {
                    size_t k=((size_t)z*P+y)*P+x; float pulled[3],jac[3]; spatial_aug_pull(&a,table,z,y,x,pulled,jac);
                    double world[3]; int inside=1;
                    for (int d=0;d<3;d++) world[sy.perm[d]]=sy.flip[d]?P-1-pulled[d]:pulled[d];
                    for (int d=0;d<3;d++) inside &= world[d]>=0 && world[d]<=P-1;
                    if (!inside) { assert(out[k]==PHASE_UNKNOWN); outside++; continue; }
                    assert(out[k]<PHASE_PERIOD);
                    double q=g[0]*world[0]+g[1]*world[1]+g[2]*world[2], e=out[k]/(double)PHASE_PERIOD-q; e-=round(e);
                    if (fabs(e)>worst) worst=fabs(e); known++;
                }
            }
            assert(worst<.01);
            printf("phase warp mode %d: %d voxels, max error %.4f turn, %d outside the window unknown: ok\n",mode,known,worst,outside);
        }
        free(ph); free(out);
    }
    {   /* hard targets (packed side classes): every warped voxel takes the code of its nearest source voxel (0 / 255, never a blend);
           the CT-only writer (--gpu-input) gives channel 0, targets and mask bit for bit as the 4-channel one */
        uint8_t *hard=malloc(NCH*n),*T1=malloc(NCH*n),*M1=malloc(n); uint16_t *H1=malloc(n*2);
        for (size_t k=0;k<n;k++) { int z=(int)(k/(P*P)),y=(int)(k/P%P),x=(int)(k%P); hard[k]=(x+2*y+3*z)%7<3?255:0; hard[n+k]=hard[k]?0:(z+x)%5?255:0; mask[k]=(x*y+z)%11!=0; }
        int blends=0,mismatch=0,checked=0,ctdiff=0;
        for (int mode=0;mode<5;mode++) {
            spatial_aug a; spatial_aug_make(&a,31+mode,P,mode==1?0:8,1,mode==0?0:1,1);
            if (mode==3) spatial_aug_zoom(&a,.6);   /* (zoom-out's 1.2 of the coarser level) */
            if (mode==4) spatial_aug_affine(&a,77,.2,.12);
            spatial_aug_tables(&a,table);
            for (int si=0;si<48;si+=5) {
                sym sy=sym_of(si); rng r;
                rseed(&r,9); write_spatial(ct,hard,mask,T,M,P,origin,80,.03f,1,cy,cx,sy,&a,1,0,.02f,&r,row,1,nullptr,H,nullptr,nullptr,nullptr,nullptr,nullptr,scratch,1,0);
                rseed(&r,9); write_spatial(ct,hard,mask,T1,M1,P,origin,80,.03f,1,cy,cx,sy,&a,1,0,.02f,&r,row,1,nullptr,H1,nullptr,nullptr,nullptr,nullptr,nullptr,scratch,1,1);
                ctdiff+=memcmp(H,H1,n*2)!=0 || memcmp(T,T1,NCH*n)!=0 || memcmp(M,M1,n)!=0;
                for (int z=0;z<P;z++) for (int y=0;y<P;y++) for (int x=0;x<P;x++) {
                    size_t k=((size_t)z*P+y)*P+x; float pulled[3],jac[3]; spatial_aug_pull(&a,table,z,y,x,pulled,jac);
                    float v[3]; int inside=1,tie=0;
                    for (int d=0;d<3;d++) v[sy.perm[d]]=sy.flip[d]?P-1-pulled[d]:pulled[d];
                    for (int d=0;d<3;d++) { inside&=v[d]>=0 && v[d]<=P-1; float f=v[d]-floorf(v[d]); tie|=fabsf(f-.5f)<1e-3f; }
                    for (int ch=0;ch<NCH;ch++) blends+=T[(size_t)ch*n+k]!=0 && T[(size_t)ch*n+k]!=255;
                    if (!inside || tie) continue;
                    size_t j=((size_t)lroundf(v[0])*P+lroundf(v[1]))*P+lroundf(v[2]);
                    for (int ch=0;ch<NCH;ch++) mismatch+=T[(size_t)ch*n+k]!=hard[(size_t)ch*n+j];
                    checked++;
                }
            }
        }
        printf("hard-target warp: %d voxels checked, %d blends, %d not the nearest source code, CT-only writer differs %d times\n",checked,blends,mismatch,ctdiff);
        assert(!blends && !mismatch && !ctdiff && checked>10000);
        free(hard); free(T1); free(M1); free(H1);
    }
    free(ct); free(target); free(mask); free(T); free(M); free(X); free(H); free(scratch);
    puts("spatial augmentation registration: ok"); return 0;
}
