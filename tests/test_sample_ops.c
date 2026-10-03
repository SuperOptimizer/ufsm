/* The sampler's fast kernels against the straightforward loops they replaced (chamfer distance, the input writer)
   and a moment check of the table-based Gaussian noise. Includes sample.c to reach its static functions. */
#include "../src/sample.c"
void *nn_host_alloc(size_t n) { return malloc(n); }
void nn_host_free(void *p) { free(p); }
/* the raster-order 3-4-5 chamfer the row-wise passes replaced (master 51125cd) */
static void chamfer_ref(uint16_t *dm, int P) {
    size_t P2 = (size_t)P * P;
#define MINU(v, e) do { uint16_t e_ = (uint16_t)(e); if (e_ < (v)) (v) = e_; } while (0)
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) {
        uint16_t *row = dm + (size_t)z * P2 + (size_t)y * P;
        const uint16_t *ry = y ? row - P : nullptr, *rz = z ? row - P2 : nullptr, *rzm = (z && y) ? row - P2 - P : nullptr, *rzp = (z && y + 1 < P) ? row - P2 + P : nullptr;
        for (int x = 0; x < P; x++) {
            uint16_t v = row[x]; if (!v) continue;
            int xm = x > 0, xp = x + 1 < P;
            if (xm) MINU(v, row[x - 1] + 3);
            if (ry) { MINU(v, ry[x] + 3); if (xm) MINU(v, ry[x - 1] + 4); if (xp) MINU(v, ry[x + 1] + 4); }
            if (rz) { MINU(v, rz[x] + 3); if (xm) MINU(v, rz[x - 1] + 4); if (xp) MINU(v, rz[x + 1] + 4);
                      if (rzm) { MINU(v, rzm[x] + 4); if (xm) MINU(v, rzm[x - 1] + 5); if (xp) MINU(v, rzm[x + 1] + 5); }
                      if (rzp) { MINU(v, rzp[x] + 4); if (xm) MINU(v, rzp[x - 1] + 5); if (xp) MINU(v, rzp[x + 1] + 5); } }
            row[x] = v;
        }
    }
    for (int z = P - 1; z >= 0; z--) for (int y = P - 1; y >= 0; y--) {
        uint16_t *row = dm + (size_t)z * P2 + (size_t)y * P;
        const uint16_t *ry = y + 1 < P ? row + P : nullptr, *rz = z + 1 < P ? row + P2 : nullptr, *rzm = (z + 1 < P && y) ? row + P2 - P : nullptr, *rzp = (z + 1 < P && y + 1 < P) ? row + P2 + P : nullptr;
        for (int x = P - 1; x >= 0; x--) {
            uint16_t v = row[x]; if (!v) continue;
            int xm = x > 0, xp = x + 1 < P;
            if (xp) MINU(v, row[x + 1] + 3);
            if (ry) { MINU(v, ry[x] + 3); if (xm) MINU(v, ry[x - 1] + 4); if (xp) MINU(v, ry[x + 1] + 4); }
            if (rz) { MINU(v, rz[x] + 3); if (xm) MINU(v, rz[x - 1] + 4); if (xp) MINU(v, rz[x + 1] + 4);
                      if (rzm) { MINU(v, rzm[x] + 4); if (xm) MINU(v, rzm[x - 1] + 5); if (xp) MINU(v, rzm[x + 1] + 5); }
                      if (rzp) { MINU(v, rzp[x] + 4); if (xm) MINU(v, rzp[x - 1] + 5); if (xp) MINU(v, rzp[x + 1] + 5); } }
            row[x] = v;
        }
    }
#undef MINU
}
APPLY_SYM(float, sym_f32_ref)
/* the old path: fp32 channels in double, then the symmetry per channel (vector channels permuted and negated) */
static void write_x_ref(const uint8_t *ctu, int P, const int64_t o[3], double mean, double sd, const double *cy, const double *cx, sym y, float a, float b, float *X) {
    size_t p3 = (size_t)P * P * P; float *xt = malloc(4 * p3 * 4);
    for (int z = 0; z < P; z++) for (int yy = 0; yy < P; yy++) { double dy = (double)(o[1] + yy) - cy[z]; for (int x = 0; x < P; x++) {
        size_t k = ((size_t)z * P + yy) * P + x; double dx = (double)(o[2] + x) - cx[z], nn = sqrt(dy * dy + dx * dx) + 1e-6;
        xt[k] = (float)((ctu[k] - mean) / sd); xt[p3 + k] = 0; xt[2 * p3 + k] = (float)(dy / nn); xt[3 * p3 + k] = (float)(dx / nn); } }
    sym_f32_ref(xt, X, P, y);
    for (size_t k = 0; k < p3; k++) X[k] = a * X[k] + b;
    for (int d = 0; d < 3; d++) { sym_f32_ref(xt + (size_t)(1 + y.perm[d]) * p3, X + (size_t)(1 + d) * p3, P, y); if (y.flip[d]) for (size_t k = 0; k < p3; k++) X[(size_t)(1 + d) * p3 + k] = -X[(size_t)(1 + d) * p3 + k]; }
    free(xt);
}
int main(void) {
    int bad = 0; rng r; rseed(&r, 7);
    const int Ps[] = {5, 32, 64, 128};
    for (int pi = 0; pi < 4; pi++) {
        int P = Ps[pi]; size_t p3 = (size_t)P * P * P;
        for (int dens = 0; dens < 3; dens++) {
            double pr = dens == 0 ? 1e-4 : dens == 1 ? 3e-3 : 0.05;
            uint16_t *a = malloc(p3 * 2), *b = malloc(p3 * 2), *tmp = malloc((size_t)P * P * 2);
            uint8_t *n1 = malloc(p3), *n2 = malloc(p3), *t2 = malloc(p3);
            for (size_t k = 0; k < p3; k++) { int on = runif(&r) < pr; a[k] = b[k] = on ? 0 : 60000; n1[k] = n2[k] = (uint8_t)on; }
            chamfer_pass(a, P, 1); chamfer_pass(a, P, -1); chamfer_ref(b, P);
            size_t dif = 0; for (size_t k = 0; k < p3; k++) dif += a[k] != b[k];
            if (dif) { printf("chamfer P %d dens %g: %zu differ  FAIL\n", P, pr, dif); bad++; }
            free(a); free(b); free(tmp); free(n1); free(n2); free(t2);
        }
    }
    {   /* speed at P = 128 (informational) */
        const int P = 128; size_t p3 = (size_t)P * P * P; uint16_t *a = malloc(p3 * 2), *b = malloc(p3 * 2);
        for (size_t k = 0; k < p3; k++) a[k] = b[k] = runif(&r) < 3e-3 ? 0 : 60000;
        struct timespec t0, t1, t2; clock_gettime(CLOCK_MONOTONIC, &t0);
        chamfer_pass(a, P, 1); chamfer_pass(a, P, -1); clock_gettime(CLOCK_MONOTONIC, &t1); chamfer_ref(b, P); clock_gettime(CLOCK_MONOTONIC, &t2);
        printf("chamfer P=128: row-wise %.1f ms, raster %.1f ms\n", (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6, (t2.tv_sec - t1.tv_sec) * 1e3 + (t2.tv_nsec - t1.tv_nsec) / 1e6);
        free(a); free(b);
    }
    {   /* fused input writer vs the old path, every symmetry */
        const int P = 12; size_t p3 = (size_t)P * P * P;
        uint8_t *ct = malloc(p3); for (size_t k = 0; k < p3; k++) ct[k] = (uint8_t)(rnext(&r) & 0xff);
        int64_t o[3] = {1000, 2345, 3456}; double cy[64], cx[64]; float cyf[64], cxf[64], nrow[64];
        for (int z = 0; z < P; z++) { cy[z] = 2350.25 + z * 0.3; cx[z] = 3450.75 - z * 0.2; cyf[z] = (float)cy[z]; cxf[z] = (float)cx[z]; }
        float *A = malloc(4 * p3 * 4), *B = malloc(4 * p3 * 4); double worst = 0;
        for (int si = 0; si < 48; si++) {
            sym y = sym_of(si);
            write_x_ref(ct, P, o, 101.5, 37.25, cy, cx, y, 1.1f, -0.05f, A);
            write_x(ct, P, o, 101.5f, (float)(1.0 / 37.25), 1, cyf, cxf, y, 1, 1.1f, -0.05f, 0.f, &r, nrow, 0, B, nullptr);
            for (size_t k = 0; k < 4 * p3; k++) { double d = fabs((double)A[k] - B[k]); if (d > worst) worst = d; }
        }
        printf("fused input writer vs old path, 48 symmetries: max abs diff %.3g\n", worst);
        if (worst > 2e-4) { printf("fused input writer  FAIL\n"); bad++; }
        free(ct); free(A); free(B);
    }
    { /* Scalar winding reference follows spatial augmentation without a sign flip. */
        const int P=12; size_t p3=(size_t)P*P*P; const int64_t o[3]={1000,2345,3456};
        double knots[2][5]={{1000,2350,3450,10,-2},{1012,2354,3448,12,-1}};
        sheet_dataset sheet={.nk=2,.knots=knots,.center=4,.scale=20};
        double params[12][4]; float cy[12],cx[12],nrow[12];
        for (int z=0;z<P;z++) { sheet_parameters(&sheet,o[0]+z,params[z]); cy[z]=params[z][0]; cx[z]=params[z][1]; }
        uint8_t *ct=calloc(p3,1); float *input=calloc(4*p3,4); uint16_t *half=calloc(4*p3,2);
        int checked=0;
        for (int si=0;si<48;si++) {
            sym y=sym_of(si); if (y.perm[0]!=0 || y.flip[0]) continue;
            write_x_aug(ct,P,o,0,1,1,cy,cx,y,0,1,0,0,&r,nrow,0,input,nullptr,nullptr,nullptr,nullptr,&sheet,params);
            write_x_aug(ct,P,o,0,1,1,cy,cx,y,0,1,0,0,&r,nrow,1,nullptr,half,nullptr,nullptr,nullptr,&sheet,params);
            for (int z=0;z<P;z++) for (int yy=0;yy<P;yy++) for (int x=0;x<P;x++) {
                int out[3]={z,yy,x}; double world[3];
                for (int d=0;d<3;d++) world[y.perm[d]]=o[y.perm[d]]+(y.flip[d]?P-1-out[d]:out[d]);
                float expected=(sheet_reference(&sheet,world)-sheet.center)/sheet.scale;
                size_t k=p3+((size_t)z*P+yy)*P+x; _Float16 h; memcpy(&h,half+k,2);
                if (fabs(input[k]-expected)>1e-6 || fabs((float)h-expected)>2e-4) bad++;
            }
            checked++;
        }
        printf("winding input: %d Z-preserving symmetries, FP32/FP16 scalar reference checked\n",checked);
        free(ct); free(input); free(half);
    }
    size_t n = 1 << 24; float *X = calloc(n, 4);
    pthread_once(&g_ntab_once, ntab_init);
    for (size_t k = 0; k < n; k += 4) { uint64_t u = rnext(&r); X[k] = g_ntab[u & 0xffff]; X[k + 1] = g_ntab[(u >> 16) & 0xffff]; X[k + 2] = g_ntab[(u >> 32) & 0xffff]; X[k + 3] = g_ntab[u >> 48]; }
    double m = 0, v = 0, m4 = 0; for (size_t k = 0; k < n; k++) { m += X[k]; v += (double)X[k] * X[k]; m4 += (double)X[k] * X[k] * X[k] * X[k]; }
    m /= n; v = v / n - m * m; m4 /= n;
    printf("noise: mean %.4f var %.4f kurtosis %.3f\n", m, v, m4 / (v * v));
    if (fabs(m) > 0.01 || fabs(v - 1) > 0.01 || fabs(m4 / (v * v) - 3) > 0.05) { printf("noise moments  FAIL\n"); bad++; }
    printf(bad ? "sample ops FAIL\n" : "sample ops ok\n");
    return bad != 0;
}
