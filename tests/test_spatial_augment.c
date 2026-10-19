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
    {   /* the cache-blocked writer against the per-voxel reference loop (write_spatial_ref), bit for bit: output formats, CT-only,
           hard / blended targets, appearance plans (LUT, filter, shading, correlated noise), the sheet prior, warps, odd sizes */
        int diffs=0,runs=0;
        for (int Pq=0;Pq<2;Pq++) {
            const int Q=Pq?37:32; const size_t nq=(size_t)Q*Q*Q;
            uint8_t *c2=malloc(nq),*t2=malloc(NCH*nq),*m2=malloc(nq),*Ta=malloc(NCH*nq),*Ma=malloc(nq),*Tb=malloc(NCH*nq),*Mb=malloc(nq);
            float *Xa=malloc(4*nq*4),*Xb=malloc(4*nq*4),*app=malloc(nq*4),*nr=malloc(Q*4),*nrs=malloc(Q*4),*nps=malloc(nq*4),cy2[40],cx2[40],tab2[6*40];
            uint16_t *Ha=malloc(4*nq*2),*Hb=malloc(4*nq*2); double par[40][4];
            for (size_t k=0;k<nq;k++) { c2[k]=(uint8_t)(k*2654435761u>>24); t2[k]=k%3?255:0; t2[nq+k]=(uint8_t)(k*40503u>>8); m2[k]=k%13!=0; }
            for (int z=0;z<Q;z++) { cy2[z]=-30.5f+.3f*z; cx2[z]=70.25f-.2f*z; sheet_parameters(&sheet,z,par[z]); }
            for (int mode=0;mode<5;mode++) for (int plan=0;plan<3;plan++) for (int fmt=0;fmt<3;fmt++) for (int si=0;si<48;si+=7) {
                spatial_aug a; spatial_aug_make(&a,90+mode,Q,mode==1?0:9,1,mode==0?0:1,1);
                if (mode==3) spatial_aug_zoom(&a,.6); if (mode==4) spatial_aug_affine(&a,5,.2,.12);
                (void)tab2;
                ct_aug_plan ap; ct_aug_make(&ap,1000+plan*7+mode,plan>0,90,30,Q);
                if (plan==2) { ap.filter_mode=1; ap.filter_strength=.4f; ap.noise_rho=.4f; ap.noise_sigma=.03f; }
                const ct_aug_plan *pp=plan?&ap:nullptr;
                for (int variant=0;variant<4;variant++) {
                    const int hard=variant&1, cto=fmt&&(variant&2), sh=!cto&&mode==2&&plan==0;
                    sym sy=sym_of(si); rng r;
                    memset(nrs,0,Q*4); memset(nps,0,nq*4); rseed(&r,77);
                    write_spatial_ref(c2,t2,m2,Ta,Ma,Q,origin,85,.031f,1,cy2,cx2,sy,&a,1.05f,.01f,.03f,&r,nr,fmt,Xa,Ha,pp,nrs,nps,sh?&sheet:nullptr,par,app,hard,cto);
                    memset(nrs,0,Q*4); memset(nps,0,nq*4); rseed(&r,77);
                    write_spatial(c2,t2,m2,Tb,Mb,Q,origin,85,.031f,1,cy2,cx2,sy,&a,1.05f,.01f,.03f,&r,nr,fmt,Xb,Hb,pp,nrs,nps,sh?&sheet:nullptr,par,app,hard,cto);
                    const size_t nch=cto?1:4;
                    int d=memcmp(Ta,Tb,NCH*nq)||memcmp(Ma,Mb,nq)||(fmt?memcmp(Ha,Hb,nch*nq*2):memcmp(Xa,Xb,nch*nq*4));
                    if (d && diffs<4) fprintf(stderr,"blocked writer differs: P %d mode %d plan %d fmt %d sym %d hard %d ct-only %d\n",Q,mode,plan,fmt,si,hard,cto);
                    diffs+=d; runs++;
                }
            }
            free(c2); free(t2); free(m2); free(Ta); free(Ma); free(Tb); free(Mb); free(Xa); free(Xb); free(app); free(nr); free(nrs); free(nps); free(Ha); free(Hb);
        }
        printf("cache-blocked warp writer vs the per-voxel loop: %d of %d configurations differ\n",diffs,runs);
        assert(!diffs);
    }
#ifdef __AVX2__
    {   /* the vector arctangent: every lane it does not hand to atan2f (need) carries atan2f's code, over random phasors of all sizes
           and phasors on / next to the rounding boundaries and the wrap */
        uint64_t s = 12345; long bad = 0, fall = 0, tot = 0, rfall = 0, rtot = 0;
        for (int it = 0; it < 400000; it++) {
            float fx[8], fy[8];
            for (int l = 0; l < 8; l++) {
                s = s * 6364136223846793005ull + 1442695040888963407ull; const double u = (s >> 11) * 0x1.0p-53;
                s = s * 6364136223846793005ull + 1442695040888963407ull; const double v = (s >> 11) * 0x1.0p-53;
                double ang = 2 * M_PI * u, mag = pow(10, -3.9 + 4.5 * v);
                if (it & 1) { const int k = (int)(u * PHASE_PERIOD); ang = 2 * M_PI * (k + .5 + (v - .5) * 1e-3 * (it & 2 ? 1 : 1e-3)) / PHASE_PERIOD; }
                if (it % 7 == 0) ang = (v - .5) * 1e-5;   /* (the wrap) */
                fx[l] = (float)(mag * cos(ang)); fy[l] = (float)(mag * sin(ang));
            }
            __m256 need; const __m256i q = warp_code8(_mm256_loadu_ps(fx), _mm256_loadu_ps(fy), &need);
            int qi[8]; _mm256_storeu_si256((__m256i *)qi, q); const int nm = _mm256_movemask_ps(need);
            for (int l = 0; l < 8; l++) { tot++; if (nm >> l & 1) fall++; else bad += qi[l] != warp_code_of(fx[l], fy[l]); }
            if (!(it & 1) && it % 7) { rtot += 8; rfall += __builtin_popcount(nm); }
        }
        printf("vector arctangent: %ld phasors (half placed on rounding boundaries), %ld to atan2f (random angles: %.2f %%), %ld codes differ\n", tot, fall, 100.0 * rfall / rtot, bad);
        assert(!bad);
    }
#endif
    {   /* warp_phase_w4 (vector, fused) against the per-voxel reference loops warp_phase_ref / warp_w4_ref, bytes equal: a spiral phase
           with unknowns and its winding mod 4 (band_wind4_tiled components), rotation / elastic / both / zoom / affine, symmetries,
           P 37 (odd: vector lanes and per-voxel tails), 64, 128, 256 and 512 */
        const int Ps[] = {37, 64, 128, 256, 512};
        int diffs = 0, runs = 0;
        for (int pi = 0; pi < 5; pi++) {
            const int Q = Ps[pi]; const size_t nq = (size_t)Q * Q * Q;
            uint8_t *pq = malloc(nq + 64), *pw = malloc(nq), *w4 = malloc(nq), *wc = malloc(nq), *o[6]; uint16_t *wq = malloc(nq * 2);
            for (int i = 0; i < 6; i++) o[i] = malloc(nq);
            for (int z = 0; z < Q; z++) for (int y = 0; y < Q; y++) for (int x = 0; x < Q; x++) {
                const size_t k = ((size_t)z * Q + y) * Q + x; const double r = hypot(y + 40.0, x - .3 * Q) / 9.0 + .01 * z, f = r - floor(r);
                pq[k] = (x * 7 + y * 3 + z) % 53 == 0 ? PHASE_UNKNOWN : (uint8_t)((int)(f * PHASE_PERIOD) % PHASE_PERIOD);
                pw[k] = (x + 2 * y) % 31 == 0 ? SIDE_UNKNOWN : pq[k];
            }
            memset(pq + nq, PHASE_UNKNOWN, 64);
            if (band_wind4_tiled(pw, Q, Q, Q, (Q + 5) / 6, 80, 64, 1, w4, wc, nullptr) < 0) abort();
            warp_w4_pack(w4, wc, wq, nq);
            const int modes = Q == 512 ? 1 : Q == 256 ? 2 : 5;
            for (int mi = 0; mi < modes; mi++) for (int si = 0; si < 48; si += Q == 512 ? 48 : Q == 256 ? 47 : Q == 128 ? 11 : 5) {
                const int mode = Q >= 256 ? (mi ? 4 : 3) : mi;
                spatial_aug a; spatial_aug_make(&a, 400 + mode + Q, Q, mode == 1 ? 0 : 10, 1, mode == 0 ? 0 : 1, 1);
                if (mode == 3) spatial_aug_zoom(&a, .7); if (mode == 4) spatial_aug_affine(&a, 6 + Q, .2, .12);
                sym sy = sym_of(si);
                warp_phase_ref(pq, o[0], Q, sy, &a); warp_w4_ref(w4, wc, o[1], o[2], Q, sy, &a);
                warp_phase_w4(pq, wq, o[3], o[4], o[5], Q, sy, &a);
                int d = memcmp(o[0], o[3], nq) || memcmp(o[1], o[4], nq) || memcmp(o[2], o[5], nq);
                if (Q < 256) {   /* the single-field forms (draw() without --task unwrap2; the tests' warp_phase) */
                    warp_phase(pq, o[3], Q, sy, &a); warp_w4(w4, wc, o[4], o[5], Q, sy, &a);
                    d |= memcmp(o[0], o[3], nq) || memcmp(o[1], o[4], nq) || memcmp(o[2], o[5], nq);
                }
                size_t known = 0; for (size_t k = 0; k < nq; k++) known += o[1][k] < PHASE_PERIOD;
                if (d && diffs < 4) fprintf(stderr, "phase / w4 warp differs: P %d mode %d sym %d\n", Q, mode, si);
                if (known < nq / 4) { fprintf(stderr, "phase / w4 warp: P %d mode %d sym %d only %zu known winding voxels\n", Q, mode, si, known); d = 1; }
                diffs += d; runs++;
            }
            free(pq); free(pw); free(w4); free(wc); free(wq); for (int i = 0; i < 6; i++) free(o[i]);
        }
        printf("fused vector phase / winding warp vs the per-voxel loops: %d of %d configurations differ\n", diffs, runs);
        assert(!diffs);
    }
    {   /* the trainer's CT writer (--gpu-input --pack-targets: hard targets, CT-only, packed codes straight from the warp) at P 128, 256
           and 512 against the per-voxel reference loop + sample_pack_tm, with no appearance plan, the --ct-aug LUT (gamma, shading,
           correlated noise) and the filter plan: fp16 CT and packed codes equal */
        int diffs = 0, runs = 0;
        for (int Q = 128; Q <= 512; Q *= 2) {
            const size_t nq = (size_t)Q * Q * Q;
            uint8_t *c2 = malloc(nq), *t2 = malloc(2 * nq), *m2 = malloc(nq), *Ta = malloc(2 * nq), *Ma = malloc(nq), *TMa = malloc(nq), *TMb = malloc(nq);
            uint16_t *Ha = malloc(nq * 2), *Hb = malloc(nq * 2); float *app = malloc(nq * 4), *nr = malloc(Q * 4), *nrs = malloc(Q * 4), *nps = malloc((size_t)Q * Q * 4), cyq[512], cxq[512];
            for (size_t k = 0; k < nq; k++) { const int x = (int)(k % Q), y = (int)(k / Q % Q); c2[k] = (x + y) % 41 == 0 ? 0 : (uint8_t)(40 + (k * 2654435761u >> 25)); t2[k] = (x / 5 + y / 7) % 3 == 0 ? 255 : 0; t2[nq + k] = (x / 3 + y / 4) % 4 == 1 ? 255 : 0; m2[k] = c2[k] != 0; }
            for (int z = 0; z < Q; z++) { cyq[z] = -30.5f + .3f * z; cxq[z] = 70.25f - .2f * z; }
            for (int plan = 0; plan < 3; plan++) {
                if (Q == 512 && plan != 1) continue;
                spatial_aug a; spatial_aug_make(&a, 70 + plan + Q, Q, 10, 1, 1, 1); spatial_aug_zoom(&a, .75);
                ct_aug_plan ap; ct_aug_make(&ap, 33 + plan, 1, 90, 30, Q);
                ap.filter_mode = plan == 2; ap.filter_strength = .4f; ap.shading[0] = .03f; ap.shading[2] = -.02f; ap.noise_rho = .35f; ap.noise_sigma = .03f;
                const ct_aug_plan *pp = plan ? &ap : nullptr; sym sy = sym_of(7 * plan + 3); rng r; const int64_t org[3] = {0, 0, 0};
                memset(nrs, 0, Q * 4); memset(nps, 0, (size_t)Q * Q * 4); rseed(&r, 5 + plan);
                write_spatial_ref(c2, t2, m2, Ta, Ma, Q, org, 85, .031f, 1, cyq, cxq, sy, &a, 1.05f, .01f, .03f, &r, nr, 1, nullptr, Ha, pp, nrs, nps, nullptr, nullptr, app, 1, 1);
                size_t bad = 0; if (sample_pack_tm(Ta, Ta + nq, Ma, nq, TMa, &bad)) abort();
                memset(nrs, 0, Q * 4); memset(nps, 0, (size_t)Q * Q * 4); rseed(&r, 5 + plan);
                write_spatial_tm(c2, t2, m2, nullptr, nullptr, Q, org, 85, .031f, 1, cyq, cxq, sy, &a, 1.05f, .01f, .03f, &r, nr, 1, nullptr, Hb, pp, nrs, nps, nullptr, nullptr, app, 1, 1, TMb);
                const int d = memcmp(TMa, TMb, nq) || memcmp(Ha, Hb, nq * 2);
                if (d) fprintf(stderr, "trainer CT writer differs: P %d plan %d\n", Q, plan);
                diffs += d; runs++;
            }
            free(c2); free(t2); free(m2); free(Ta); free(Ma); free(TMa); free(TMb); free(Ha); free(Hb); free(app); free(nr); free(nrs); free(nps);
        }
        printf("trainer CT writer (packed, fp16, plans none / LUT / filter) vs the per-voxel loop at P 128-512: %d of %d differ\n", diffs, runs);
        assert(!diffs);
    }
    {   /* band_wind4 (run-pending union-find) vs band_wind4_ref (per-voxel finds and unions): w4, comp and the stats byte for byte on
           spiral phases with noise (joins that break or wrap), label conflicts (a sheared wall whose phase jumps), unknown blobs and air,
           several shapes, step limits, minimum sizes and component counts, whole windows and tiled */
        int diffs = 0, runs = 0; size_t tot_conf = 0, tot_kept = 0, tot_known = 0;
        const int shapes[][3] = {{37, 41, 43}, {64, 64, 64}, {96, 80, 112}, {130, 128, 129}};
        for (int si = 0; si < 4; si++) for (int var = 0; var < 4; var++) {
            const int D = shapes[si][0], Hh = shapes[si][1], Ww = shapes[si][2]; const size_t nn = (size_t)D * Hh * Ww;
            uint8_t *ph = malloc(nn), *wa = malloc(nn), *ca = malloc(nn), *wb = malloc(nn), *cb = malloc(nn);
            const double cy0 = Hh * (var == 3 ? -.8 : .45), cx0 = Ww * .55, pitch = var == 1 ? 5.5 : 9.0;
            for (int z = 0; z < D; z++) for (int y = 0; y < Hh; y++) for (int x = 0; x < Ww; x++) {
                const size_t k = ((size_t)z * Hh + y) * Ww + x; const uint64_t h = mix64(k * 0x9e3779b97f4a7c15ull + (uint64_t)var);
                double w = hypot(y - cy0, x - cx0) / pitch - atan2(y - cy0, x - cx0) / (2 * M_PI) + .01 * z;
                if (var == 2 && x > Ww / 2 + (y - z) / 3) w += .37;   /* a wall of inconsistent labels */
                w += (double)((int)(h >> 60) - 8) * (var == 1 ? .02 : .004);   /* noise: some joins exceed the step limit */
                const double f = w - floor(w); int c = (int)(f * PHASE_PERIOD); if (c >= PHASE_PERIOD) c = PHASE_PERIOD - 1;
                const int blob = ((x / 11 + 3 * (y / 13) + 5 * (z / 9)) % 7) == 0, air = (h >> 20 & 1023) < (var == 1 ? 60u : 6u);
                ph[k] = blob || air ? PHASE_UNKNOWN : (uint8_t)c;
            }
            const int steps[3] = {80, 30, 125}; const size_t mins[3] = {512, 1, 40}; const int mcs[3] = {15, 1, 4};
            for (int pv = 0; pv < 3; pv++) {
                band_wind4_stats sa = {0}, sb = {0};
                const int na = band_wind4_ref(ph, D, Hh, Ww, steps[pv], mins[pv], mcs[pv], wa, ca, &sa);
                const int nb = band_wind4(ph, D, Hh, Ww, steps[pv], mins[pv], mcs[pv], wb, cb, &sb);
                const int d = na != nb || memcmp(wa, wb, nn) || memcmp(ca, cb, nn) || sa.known != sb.known || sa.kept != sb.kept ||
                              sa.dropped_conflict != sb.dropped_conflict || sa.runs != sb.runs || sa.ncomp != sb.ncomp;
                if (d) fprintf(stderr, "band_wind4 differs: shape %d var %d params %d (%d vs %d components)\n", si, var, pv, na, nb);
                diffs += d; runs++; tot_conf += sa.dropped_conflict; tot_kept += sa.kept; tot_known += sa.known;
                const int tiles[2] = {11 + 5 * pv, 40};   /* tiled (draw()'s form; 11-21: over 255 components, the vector scatter's limit) */
                for (int ti = 0; ti < 2; ti++) {
                    const int ta = band_wind4_tiled_ref(ph, D, Hh, Ww, tiles[ti], steps[pv], mins[pv] / 8 + 1, mcs[pv], wa, ca, &sa);
                    const int tb = band_wind4_tiled(ph, D, Hh, Ww, tiles[ti], steps[pv], mins[pv] / 8 + 1, mcs[pv], wb, cb, &sb);
                    const int dt = ta != tb || memcmp(wa, wb, nn) || memcmp(ca, cb, nn) || sa.known != sb.known || sa.kept != sb.kept ||
                                   sa.dropped_conflict != sb.dropped_conflict || sa.runs != sb.runs || sa.ncomp != sb.ncomp;
                    if (dt) fprintf(stderr, "band_wind4_tiled differs: shape %d var %d params %d tile %d\n", si, var, pv, tiles[ti]);
                    diffs += dt; runs++;
                }
            }
            free(ph); free(wa); free(ca); free(wb); free(cb);
        }
        printf("band_wind4 (whole and tiled) vs the per-voxel reference (4 shapes x 4 phases x 3 parameter sets; %zu known, %zu kept, %zu dropped as inconsistent): %d of %d differ\n", tot_known, tot_kept, tot_conf, diffs, runs);
        assert(!diffs);
    }
    free(ct); free(target); free(mask); free(T); free(M); free(X); free(H); free(scratch);
    puts("spatial augmentation registration: ok"); return 0;
}
