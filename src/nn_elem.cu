/* CUDA ops: elem section of the former nn.cu */
#include "nn_common.cuh"

__global__ void gn_finalize_k(const double *sums, int NG, size_t len, float eps, float *mean, float *rstd) {
    int ng = blockIdx.x * blockDim.x + threadIdx.x;
    if (ng >= NG) return;
    double m = sums[2 * ng] / (double)len, v = sums[2 * ng + 1] / (double)len - m * m;
    mean[ng] = (float)m;
    rstd[ng] = (float)(1.0 / sqrt((v > 0 ? v : 0) + (double)eps));
}
double *gn_dsums(size_t n) {   /* small persistent device scratch for the double sums, per device */
    static double *buf[10]; static size_t cap[10];
    int d = zs_slot();   /* (both split halves on one GPU: per half) */
    if (n > cap[d]) { if (buf[d]) cudaFree(buf[d]); cudaMalloc(&buf[d], n * sizeof(double)); cap[d] = n; }
    return buf[d];
}
extern "C" void nn_gn_fwd(const float *x, shape5 s, int G, float eps, const float *gamma, const float *beta, float *y, float *mean, float *rstd) {
    if (G > s.c) G = s.c;
    size_t S = shape_spatial(s);
    int NG = s.n * G;
    double *sums = gn_dsums((size_t)2 * NG);
    cudaMemsetAsync(sums, 0, (size_t)2 * NG * sizeof(double));
    gn_sums_k<<<dim3(NG, KSLAB), 256>>>(x, s.c, G, S, sums);
    gn_finalize_k<<<nblk(NG, 128), 128>>>(sums, NG, (size_t)(s.c / G) * S, eps, mean, rstd);
    size_t n = shape_numel(s);
    if (y) gn_apply_k<0, float, float><<<dim3(s.n * s.c, KSLAB), 256>>>(x, gamma, beta, mean, rstd, y, s.c, G, S);   /* y == nullptr: statistics only */
    KCHECK();
}
extern "C" int nn_gn_stats(const float *x, shape5 s, int G, float eps, float *mean, float *rstd) {
    if (G > s.c) G = s.c;
    size_t S = shape_spatial(s);
    int NG = s.n * G;
    double *sums = gn_dsums((size_t)2 * NG);
    cudaMemsetAsync(sums, 0, (size_t)2 * NG * sizeof(double));
    int zlo, zhi; zs_range(s.d, &zlo, &zhi);
    const size_t v0 = (size_t)zlo * s.h * s.w, v1 = (size_t)(s.d - zhi) * s.h * s.w;
    if (ISMX(x)) {   /* MX storage: statistics of the dequantised values */
        if (lp_gn_sums_mx(x, MXDT(x), s.n, s.c, G, S, sums, v0, v1)) return -1;
        zs_reduce(sums, 2 * NG);
        gn_finalize_k<<<nblk(NG, 128), 128>>>(sums, NG, zs_len((size_t)(s.c / G) * S, s.d), eps, mean, rstd);
        KCHECK();
        return 0;
    }
    if (ABF && g_h16) gn_sums_k<f16><<<dim3(NG, KSLAB), 256>>>((const f16 *)x, s.c, G, S, sums, v0, v1);
    else if (ABF) gn_sums_k<bf16><<<dim3(NG, KSLAB), 256>>>((const bf16 *)x, s.c, G, S, sums, v0, v1);
    else gn_sums_k<float><<<dim3(NG, KSLAB), 256>>>(x, s.c, G, S, sums, v0, v1);
    zs_reduce(sums, 2 * NG);
    gn_finalize_k<<<nblk(NG, 128), 128>>>(sums, NG, zs_len((size_t)(s.c / G) * S, s.d), eps, mean, rstd);
    KCHECK();
    return 0;
}
extern "C" void nn_gn_fwd_silu(const float *x, shape5 s, int G, float eps, const float *gamma, const float *beta, float *y, float *mean, float *rstd) {
    if (G > s.c) G = s.c;
    size_t S = shape_spatial(s);
    int NG = s.n * G;
    double *sums = gn_dsums((size_t)2 * NG);
    cudaMemsetAsync(sums, 0, (size_t)2 * NG * sizeof(double));
    gn_sums_k<<<dim3(NG, KSLAB), 256>>>(x, s.c, G, S, sums);
    gn_finalize_k<<<nblk(NG, 128), 128>>>(sums, NG, (size_t)(s.c / G) * S, eps, mean, rstd);
    size_t n = shape_numel(s);
    gn_apply_k<1, float, float><<<dim3(s.n * s.c, KSLAB), 256>>>(x, gamma, beta, mean, rstd, y, s.c, G, S);
    KCHECK();
}
extern "C" void nn_gn_apply_silu(const float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, float *g, float *sil) {
    if (G > s.c) G = s.c;
    size_t n = shape_numel(s), S = shape_spatial(s);
    gn_apply_k<0, float, float><<<dim3(s.n * s.c, KSLAB), 256>>>(x, gamma, beta, mean, rstd, g, s.c, G, S);
    silu_f_k<<<nblk(n, 256), 256>>>(g, sil, n);
    KCHECK();
}
__global__ void gn_bwd_stats_k(const float *x, const float *gy, const float *mean, const float *rstd, int C, int G, size_t S, double *ds) {
    int nc = blockIdx.x, slab = blockIdx.y, n = nc / C, c = nc % C, cpg = C / G, ng = n * G + c / cpg;
    const float *xp = x + (size_t)nc * S, *gp = gy + (size_t)nc * S;
    float m = mean[ng], r = rstd[ng];
    size_t per = (S + KSLAB - 1) / KSLAB, lo = (size_t)slab * per, hi = lo + per < S ? lo + per : S;
    double s1 = 0, s2 = 0;
    for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) { double g = gp[i]; s1 += g; s2 += g * ((xp[i] - m) * r); }
    __shared__ double r1[256], r2[256];
    r1[threadIdx.x] = s1; r2[threadIdx.x] = s2;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) { r1[threadIdx.x] += r1[threadIdx.x + o]; r2[threadIdx.x] += r2[threadIdx.x + o]; } __syncthreads(); }
    if (threadIdx.x == 0) { atomicAdd(&ds[2 * nc], r1[0]); atomicAdd(&ds[2 * nc + 1], r2[0]); }
}
__global__ void d2f_k(const double *d, float *f, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) f[i] = (float)d[i]; }
__global__ void gn_bwd_apply_k(const float *x, const float *gy, const float *gamma, const float *mean, const float *rstd, const float *st,
                               float *gx, int N, int C, int G, size_t S) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * C * S) return;
    int c = (int)((i / S) % C), n = (int)(i / (S * C)), cpg = C / G, g = c / cpg, ng = n * G + g;
    /* group sums of gxhat = gy*gamma and gxhat*xhat */
    float a = 0.f, b = 0.f;
    for (int cc = g * cpg; cc < (g + 1) * cpg; cc++) { a += st[2 * (n * C + cc)] * gamma[cc]; b += st[2 * (n * C + cc) + 1] * gamma[cc]; }
    float len = (float)cpg * (float)S;
    float xhat = (x[i] - mean[ng]) * rstd[ng];
    float gxhat = gy[i] * gamma[c];
    gx[i] = rstd[ng] * (gxhat - a / len - xhat * b / len);
}
__global__ void gn_param_grad_k2(const float *st, float *ggamma, float *gbeta, int N, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float sg = 0.f, sb = 0.f;
    for (int n = 0; n < N; n++) { sb += st[2 * (n * C + c)]; sg += st[2 * (n * C + c) + 1]; }
    ggamma[c] += sg;
    gbeta[c] += sb;
}
__global__ void gn_param_grad_k(const float *st, float *ggamma, float *gbeta, int N, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float sg = 0.f, sb = 0.f;
    for (int n = 0; n < N; n++) { sb += st[2 * (n * C + c)]; sg += st[2 * (n * C + c) + 1]; }
    ggamma[c] += sg;
    gbeta[c] += sb;
}
extern "C" size_t nn_gn_scratch(shape5 s) { return (size_t)4 * s.n * s.c * sizeof(float); }
extern "C" void nn_gn_apply(const float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean, const float *rstd, float *y) {
    if (G > s.c) G = s.c;
    size_t n = shape_numel(s);
    gn_apply_k<0, float, float><<<dim3(s.n * s.c, KSLAB), 256>>>(x, gamma, beta, mean, rstd, y, s.c, G, shape_spatial(s));
    KCHECK();
}
extern "C" void nn_gn_bwd(const float *x, shape5 s, int G, const float *gamma, const float *mean, const float *rstd, const float *gy,
                          float *gx, float *ggamma, float *gbeta, float *scratch) {
    if (G > s.c) G = s.c;
    size_t S = shape_spatial(s);
    int NC = s.n * s.c;
    double *ds = gn_dsums((size_t)2 * NC);
    cudaMemsetAsync(ds, 0, (size_t)2 * NC * sizeof(double));
    gn_bwd_stats_k<<<dim3(NC, KSLAB), 256>>>(x, gy, mean, rstd, s.c, G, S, ds);
    d2f_k<<<nblk(2 * NC, 128), 128>>>(ds, scratch, 2 * NC);
    size_t n = shape_numel(s);
    gn_bwd_apply_k<<<nblk(n, 256), 256>>>(x, gy, gamma, mean, rstd, scratch, gx, s.n, s.c, G, S);
    gn_param_grad_k<<<nblk(s.c, 128), 128>>>(scratch, ggamma, gbeta, s.n, s.c);
    KCHECK();
}
__global__ void silu_f_k(const float *x, float *y, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; y[i] = v / (1.f + expf(-v)); }
}
__global__ void silu_b_k(const float *x, const float *gy, float *gx, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i], s = 1.f / (1.f + expf(-v)); gx[i] = gy[i] * (s * (1.f + v * (1.f - s))); }
}
__global__ void axpy_k(float *y, float a, const float *x, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) y[i] += a * x[i]; }
__global__ void scale_k(float *y, float a, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) y[i] *= a; }
__global__ void u8f_k(const uint8_t *x, float s, float *y, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) y[i] = x[i] * s; }
__global__ void sigm_k(const float *x, float *y, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) y[i] = 1.f / (1.f + expf(-x[i])); }
extern "C" void nn_silu_fwd(const float *x, size_t n, float *y) { silu_f_k<<<nblk(n, 256), 256>>>(x, y, n); KCHECK(); }
extern "C" void nn_silu_bwd(const float *x, const float *gy, size_t n, float *gx) { silu_b_k<<<nblk(n, 256), 256>>>(x, gy, gx, n); KCHECK(); }
extern "C" void nn_axpy(float *y, float a, const float *x, size_t n) { axpy_k<<<nblk(n, 256), 256>>>(y, a, x, n); KCHECK(); }
extern "C" void nn_scale(float *y, float a, size_t n) { scale_k<<<nblk(n, 256), 256>>>(y, a, n); KCHECK(); }
extern "C" void nn_u8_to_f32(const uint8_t *x, size_t n, float scale, float *y) { u8f_k<<<nblk(n, 256), 256>>>(x, scale, y, n); KCHECK(); }
extern "C" void nn_f32_to_h16(const float *x, size_t n, void *y, float scale) { if (g_h16) f2h_k<f16><<<nblk(n, 256), 256>>>(x, (f16 *)y, n, scale); else f2h_k<bf16><<<nblk(n, 256), 256>>>(x, (bf16 *)y, n, scale); KCHECK(); }
extern "C" void nn_f32_to_bf16(const float *x, size_t n, void *y) { nn_f32_to_h16(x, n, y, 1.f); }
extern "C" void nn_h16_to_mx(const void *x, shape5 s, void *y) {
    if (ISMX4(y)) lp_h16_to_mx4(x, g_h16 ? 2 : 1, s.n, s.c, shape_spatial(s), y); else lp_h16_to_mx8(x, g_h16 ? 2 : 1, s.n, s.c, shape_spatial(s), y);
    KCHECK();
}
extern "C" void nn_f32_to_act(const float *x, shape5 s, void *y) {
    if (ISMX4(y)) { lp_f32_to_mx4(x, s.n, s.c, shape_spatial(s), y); KCHECK(); return; }
    if (ISMX(y)) { lp_f32_to_mx8(x, s.n, s.c, shape_spatial(s), y); KCHECK(); return; }
    nn_f32_to_h16(x, shape_numel(s), y, 1.f);
}
__global__ void mask24_k(const float *w, float *out, int Co, int Ci, int T) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, ng = (size_t)Co * (Ci / 4) * T;
    if (i >= ng) return;
    int t = (int)(i % T), g4 = (int)((i / T) % (Ci / 4)), co = (int)(i / ((size_t)T * (Ci / 4)));
    size_t base = ((size_t)co * Ci + 4 * g4) * T + t;
    float v[4], a[4];
#pragma unroll
    for (int j = 0; j < 4; j++) { v[j] = w[base + (size_t)j * T]; a[j] = fabsf(v[j]); }
    int i0 = 0; for (int j = 1; j < 4; j++) if (a[j] > a[i0]) i0 = j;
    int i1 = -1; for (int j = 0; j < 4; j++) if (j != i0 && (i1 < 0 || a[j] > a[i1])) i1 = j;
#pragma unroll
    for (int j = 0; j < 4; j++) out[base + (size_t)j * T] = (j == i0 || j == i1) ? v[j] : 0.f;
}
__global__ void srste24_k(float *g, const float *w, int Co, int Ci, int T, float lambda) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, ng = (size_t)Co * (Ci / 4) * T;
    if (i >= ng) return;
    int t = (int)(i % T), g4 = (int)((i / T) % (Ci / 4)), co = (int)(i / ((size_t)T * (Ci / 4)));
    size_t base = ((size_t)co * Ci + 4 * g4) * T + t;
    float a[4];
#pragma unroll
    for (int j = 0; j < 4; j++) a[j] = fabsf(w[base + (size_t)j * T]);
    int i0 = 0; for (int j = 1; j < 4; j++) if (a[j] > a[i0]) i0 = j;
    int i1 = -1; for (int j = 0; j < 4; j++) if (j != i0 && (i1 < 0 || a[j] > a[i1])) i1 = j;
#pragma unroll
    for (int j = 0; j < 4; j++) if (j != i0 && j != i1) g[base + (size_t)j * T] += lambda * w[base + (size_t)j * T];
}
__global__ void wquant_k(float *w, int Co, int Ci, int T, int bits, unsigned seed) {
    size_t nb = (size_t)Co * ((Ci + 31) / 32) * T, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nb) return;
    int t = (int)(i % T), blk = (int)((i / T) % ((Ci + 31) / 32)), co = (int)(i / ((size_t)T * ((Ci + 31) / 32)));
    int c0 = blk * 32, c1 = c0 + 32 < Ci ? c0 + 32 : Ci;
    const float qmax = bits == 8 ? 448.f : 6.f;
    float amax = 0.f;
    for (int c = c0; c < c1; c++) amax = fmaxf(amax, fabsf(w[((size_t)co * Ci + c) * T + t]));
    if (amax == 0.f) return;
    int e = (int)ceilf(log2f(amax / qmax));
    float scale = ldexpf(1.f, e), inv = ldexpf(1.f, -e);
    for (int c = c0; c < c1; c++) {
        size_t k = ((size_t)co * Ci + c) * T + t;
        float x = w[k], a = fabsf(x) * inv;
        if (a > qmax) a = qmax;
        float sp = grid_spacing(a, bits), lo = floorf(a / sp) * sp, hi = lo + sp;
        if (hi > qmax) hi = qmax;
        float u = (float)(hash32((unsigned)k * 2654435761u ^ seed) & 0xffffff) * (1.f / 16777216.f);
        float q = (u < (a - lo) / sp) ? hi : lo;
        w[k] = copysignf(q * scale, x);
    }
}
__global__ void wq_pack_k(const float *w, unsigned char *q, unsigned char *sc, int Co, int Ci, int T, int bits, unsigned seed) {
    int nblk = (Ci + 31) / 32; size_t nb = (size_t)Co * nblk * T, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nb) return;
    int t = (int)(i % T), blk = (int)((i / T) % nblk), co = (int)(i / ((size_t)T * nblk)), c0 = blk * 32, c1 = c0 + 32 < Ci ? c0 + 32 : Ci;
    const float qmax = bits == 8 ? 448.f : 6.f;
    float amax = 0.f;
    for (int c = c0; c < c1; c++) amax = fmaxf(amax, fabsf(w[((size_t)co * Ci + c) * T + t]));
    int e = amax > 0.f ? (int)ceilf(log2f(amax / qmax)) : -40; if (e < -40) e = -40; if (e > 60) e = 60;   /* clamp: keeps 1/scale finite; values below 2^-40 flush to 0 */
    sc[i] = (unsigned char)(e + 127);
    float inv = ldexpf(1.f, -e);
    for (int c = c0; c < c1; c++) {
        size_t k = ((size_t)co * Ci + c) * T + t; float x = w[k];
        float qv = seed ? sr_quant(fabsf(x) * inv, bits, qmax, hash32((unsigned)k * 2654435761u ^ seed)) : fminf(fabsf(x) * inv, qmax);
        if (!seed) { float sp = grid_spacing(qv, bits); qv = rintf(qv / sp) * sp; }
        wq_put(q, wq_sidx(bits, k, i, c - c0), bits, copysignf(qv, x));
    }
}
__global__ void wq_unpack_k(const unsigned char *q, const unsigned char *sc, float *w, int Co, int Ci, int T, int bits) {
    int nblk = (Ci + 31) / 32; size_t n = (size_t)Co * Ci * T, k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= n) return;
    int t = (int)(k % T), ci = (int)((k / T) % Ci), co = (int)(k / ((size_t)T * Ci));
    size_t b = ((size_t)co * nblk + ci / 32) * T + t;
    w[k] = wq_get(q, wq_sidx(bits, k, b, ci % 32), bits) * ldexpf(1.f, (int)sc[b] - 127);
}
__global__ void wq_adamw_k(unsigned char *q, unsigned char *sc, unsigned char *r, unsigned char *rsc, const float *g, float *m, float *v, int Co, int Ci, int T, int bits,
                           float lr, float b1, float b2, float eps, float wd, float c1, float c2, unsigned seed) {
    int nblk = (Ci + 31) / 32; size_t nb = (size_t)Co * nblk * T, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nb) return;
    int t = (int)(i % T), blk = (int)((i / T) % nblk), co = (int)(i / ((size_t)T * nblk)), c0 = blk * 32, cn = c0 + 32 < Ci ? c0 + 32 : Ci;
    const float qmax = bits == 8 ? 448.f : 6.f;
    float scale = ldexpf(1.f, (int)sc[i] - 127), nw[32], amax = 0.f, rs = r ? ldexpf(1.f, (int)rsc[i] - 127) : 0.f;
    for (int c = c0; c < cn; c++) {
        size_t k = ((size_t)co * Ci + c) * T + t;
        float x = wq_get(q, wq_sidx(bits, k, i, c - c0), bits) * scale + (r ? dec_e4m3(r[k]) * rs : 0.f), gi = g[k];
        float mi = m[k] = b1 * m[k] + (1.f - b1) * gi, vi = v[k] = b2 * v[k] + (1.f - b2) * gi * gi;
        x -= lr * ((mi / c1) / (sqrtf(vi / c2) + eps) + wd * x);
        nw[c - c0] = x; amax = fmaxf(amax, fabsf(x));
    }
    int e = amax > 0.f ? (int)ceilf(log2f(amax / qmax)) : -40; if (e < -40) e = -40; if (e > 60) e = 60;   /* clamp: keeps 1/scale finite; values below 2^-40 flush to 0 */
    sc[i] = (unsigned char)(e + 127);
    float inv = ldexpf(1.f, -e);
    if (r) {   /* round to nearest on the fp4 grid, residual carries the rest */
        float res[32]; size_t ks[32];
        for (int c = c0; c < cn; c++) { size_t k = ((size_t)co * Ci + c) * T + t; float x = nw[c - c0], a = fminf(fabsf(x) * inv, qmax), sp = grid_spacing(a, bits), qv = rintf(a / sp) * sp; if (qv > qmax) qv = qmax; wq_put(q, wq_sidx(bits, k, i, c - c0), bits, copysignf(qv, x)); res[c - c0] = x - copysignf(qv, x) * ldexpf(1.f, e); ks[c - c0] = k; }
        res_write(r, rsc, i, res, cn - c0, ks);
        return;
    }
    for (int c = c0; c < cn; c++) { size_t k = ((size_t)co * Ci + c) * T + t; float x = nw[c - c0]; wq_put(q, wq_sidx(bits, k, i, c - c0), bits, copysignf(sr_quant(fabsf(x) * inv, bits, qmax, hash32((unsigned)k * 2654435761u ^ seed)), x)); }
}
__global__ void wq_ema_k(unsigned char *qe, unsigned char *sce, unsigned char *re, unsigned char *rsce, const unsigned char *qp, const unsigned char *scp, const unsigned char *rp, const unsigned char *rscp, int Co, int Ci, int T, int bits, float d, unsigned seed) {
    int nblk = (Ci + 31) / 32; size_t nb = (size_t)Co * nblk * T, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nb) return;
    int t = (int)(i % T), blk = (int)((i / T) % nblk), co = (int)(i / ((size_t)T * nblk)), c0 = blk * 32, cn = c0 + 32 < Ci ? c0 + 32 : Ci;
    const float qmax = bits == 8 ? 448.f : 6.f;
    float se = ldexpf(1.f, (int)sce[i] - 127), sp = ldexpf(1.f, (int)scp[i] - 127), nw[32], amax = 0.f;
    float rse = re ? ldexpf(1.f, (int)rsce[i] - 127) : 0.f, rsp = rp ? ldexpf(1.f, (int)rscp[i] - 127) : 0.f;
    for (int c = c0; c < cn; c++) { size_t k = ((size_t)co * Ci + c) * T + t, si = wq_sidx(bits, k, i, c - c0); float x = d * (wq_get(qe, si, bits) * se + (re ? dec_e4m3(re[k]) * rse : 0.f)) + (1.f - d) * (wq_get(qp, si, bits) * sp + (rp ? dec_e4m3(rp[k]) * rsp : 0.f)); nw[c - c0] = x; amax = fmaxf(amax, fabsf(x)); }
    int e = amax > 0.f ? (int)ceilf(log2f(amax / qmax)) : -40; if (e < -40) e = -40; if (e > 60) e = 60;   /* clamp: keeps 1/scale finite; values below 2^-40 flush to 0 */
    sce[i] = (unsigned char)(e + 127);
    float inv = ldexpf(1.f, -e);
    if (re) {
        float res[32]; size_t ks[32];
        for (int c = c0; c < cn; c++) { size_t k = ((size_t)co * Ci + c) * T + t; float x = nw[c - c0], a = fminf(fabsf(x) * inv, qmax), spc = grid_spacing(a, bits), qv = rintf(a / spc) * spc; if (qv > qmax) qv = qmax; wq_put(qe, wq_sidx(bits, k, i, c - c0), bits, copysignf(qv, x)); res[c - c0] = x - copysignf(qv, x) * ldexpf(1.f, e); ks[c - c0] = k; }
        res_write(re, rsce, i, res, cn - c0, ks);
        return;
    }
    for (int c = c0; c < cn; c++) { size_t k = ((size_t)co * Ci + c) * T + t; float x = nw[c - c0]; wq_put(qe, wq_sidx(bits, k, i, c - c0), bits, copysignf(sr_quant(fabsf(x) * inv, bits, qmax, hash32((unsigned)k * 2654435761u ^ seed)), x)); }
}
extern "C" size_t nn_wq_nblocks(int co, int ci, int taps) { return (size_t)co * ((ci + 31) / 32) * taps; }
extern "C" size_t nn_wq_bytes(int co, int ci, int taps, int bits) { return bits == 8 ? (size_t)co * ci * taps : nn_wq_nblocks(co, ci, taps) * 16; }
__global__ void wq_residual_k(const float *w, const unsigned char *q, const unsigned char *sc, unsigned char *r, unsigned char *rsc, int Co, int Ci, int T, int bits) {
    int nblk = (Ci + 31) / 32; size_t nb = (size_t)Co * nblk * T, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= nb) return;
    int t = (int)(i % T), blk = (int)((i / T) % nblk), co = (int)(i / ((size_t)T * nblk)), c0 = blk * 32, cn = c0 + 32 < Ci ? c0 + 32 : Ci;
    float scale = ldexpf(1.f, (int)sc[i] - 127), res[32]; size_t ks[32];
    for (int c = c0; c < cn; c++) { size_t k = ((size_t)co * Ci + c) * T + t; res[c - c0] = w[k] - wq_get(q, wq_sidx(bits, k, i, c - c0), bits) * scale; ks[c - c0] = k; }
    res_write(r, rsc, i, res, cn - c0, ks);
}
extern "C" void nn_wq_pack(const float *w, void *q, void *sc, int co, int ci, int taps, int bits, unsigned seed) {
    size_t nb = nn_wq_nblocks(co, ci, taps); wq_pack_k<<<nblk(nb, 128), 128>>>(w, (unsigned char *)q, (unsigned char *)sc, co, ci, taps, bits, seed); KCHECK();
}
extern "C" void nn_wq_residual(const float *w, const void *q, const void *sc, void *r, void *rsc, int co, int ci, int taps, int bits) {
    size_t nb = nn_wq_nblocks(co, ci, taps); wq_residual_k<<<nblk(nb, 128), 128>>>(w, (const unsigned char *)q, (const unsigned char *)sc, (unsigned char *)r, (unsigned char *)rsc, co, ci, taps, bits); KCHECK();
}
extern "C" void nn_wq_unpack(const void *q, const void *sc, float *w, int co, int ci, int taps, int bits) {
    size_t n = (size_t)co * ci * taps; wq_unpack_k<<<nblk(n, 256), 256>>>((const unsigned char *)q, (const unsigned char *)sc, w, co, ci, taps, bits); KCHECK();
}
extern "C" void nn_wq_adamw(void *q, void *sc, void *r, void *rsc, const float *g, float *m, float *v, int co, int ci, int taps, int bits, float lr, float b1, float b2, float eps, float wd, int step, unsigned seed) {
    float c1 = 1.f - powf(b1, (float)step), c2 = 1.f - powf(b2, (float)step);
    size_t nb = nn_wq_nblocks(co, ci, taps);
    wq_adamw_k<<<nblk(nb, 128), 128>>>((unsigned char *)q, (unsigned char *)sc, (unsigned char *)r, (unsigned char *)rsc, g, m, v, co, ci, taps, bits, lr, b1, b2, eps, wd, c1, c2, seed); KCHECK();
}
extern "C" void nn_wq_ema(void *qe, void *sce, void *re, void *rsce, const void *qp, const void *scp, const void *rp, const void *rscp, int co, int ci, int taps, int bits, float decay, unsigned seed) {
    size_t nb = nn_wq_nblocks(co, ci, taps);
    wq_ema_k<<<nblk(nb, 128), 128>>>((unsigned char *)qe, (unsigned char *)sce, (unsigned char *)re, (unsigned char *)rsce, (const unsigned char *)qp, (const unsigned char *)scp, (const unsigned char *)rp, (const unsigned char *)rscp, co, ci, taps, bits, decay, seed); KCHECK();
}
extern "C" void nn_wquant(float *w, int co, int ci, int taps, int bits, unsigned seed) {
    size_t nb = (size_t)co * ((ci + 31) / 32) * taps; wquant_k<<<nblk(nb, 128), 128>>>(w, co, ci, taps, bits, seed); KCHECK();
}
extern "C" void nn_mask24(const float *w, float *out, int co, int ci, int taps) {
    if (ci % 4) { if (out != w) cudaMemcpyAsync(out, w, (size_t)co * ci * taps * 4, cudaMemcpyDeviceToDevice); return; }
    size_t ng = (size_t)co * (ci / 4) * taps; mask24_k<<<nblk(ng, 256), 256>>>(w, out, co, ci, taps); KCHECK();
}
extern "C" void nn_srste24(float *g, const float *w, int co, int ci, int taps, float lambda) {
    if (ci % 4) return;
    size_t ng = (size_t)co * (ci / 4) * taps; srste24_k<<<nblk(ng, 256), 256>>>(g, w, co, ci, taps, lambda); KCHECK();
}
extern "C" void nn_sigmoid(const float *x, size_t n, float *y) { sigm_k<<<nblk(n, 256), 256>>>(x, y, n); KCHECK(); }
__global__ void flip32_k(unsigned *x, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) x[i] ^= 0x80000000u; }
__global__ void flip16_k(unsigned short *x, size_t n) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) x[i] ^= (unsigned short)0x8000; }
extern "C" void nn_flip_sign(void *x, size_t n, int h16) {
    if (h16) flip16_k<<<nblk(n, 256), 256>>>((unsigned short *)x, n); else flip32_k<<<nblk(n, 256), 256>>>((unsigned *)x, n);
    KCHECK();
}
template <typename LT> __global__ void prob_u8_k(const LT *lg, size_t n, uint8_t *out, float hard) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float p = 1.f / (1.f + expf(-lgv(lg, i)));
    out[i] = hard > 0 ? (p >= hard ? 255 : 0) : (uint8_t)(255.f * p + 0.5f);
}
extern "C" void nn_prob_u8(const float *logits, size_t n, uint8_t *out, float hard) {   /* logits fp16 when nn_set_logits_h16 */
    if (g_logits_h16) prob_u8_k<f16><<<nblk(n, 256), 256>>>((const f16 *)logits, n, out, hard); else prob_u8_k<float><<<nblk(n, 256), 256>>>(logits, n, out, hard);
    KCHECK();
}
template <typename T> __device__ __forceinline__ float ldin(const T *x, size_t i);
template <> __device__ __forceinline__ float ldin<float>(const float *x, size_t i) { return x[i]; }
template <> __device__ __forceinline__ float ldin<__half>(const __half *x, size_t i) { return __half2float(x[i]); }
template <> __device__ __forceinline__ float ldin<__nv_bfloat16>(const __nv_bfloat16 *x, size_t i) { return __bfloat162float(x[i]); }
__device__ __forceinline__ int pat(const uint8_t *p, int d, int h, int w, float z, float y, float x) {
    int a = __float2int_rn(z), b = __float2int_rn(y), c = __float2int_rn(x);
    if (a < 0 || b < 0 || c < 0 || a >= d || b >= h || c >= w) return -1;
    return p[((size_t)a * h + b) * w + c];
}
__device__ __forceinline__ int is_ridge(const uint8_t *p, int d, int h, int w, float z, float y, float x, float nz, float ny, float nx, int thr) {
    const int v = pat(p, d, h, w, z, y, x);
    if (v < thr) return 0;
    for (int s = 1; s <= 2; s++)
        if (pat(p, d, h, w, z + s * nz, y + s * ny, x + s * nx) > v || pat(p, d, h, w, z - s * nz, y - s * ny, x - s * nx) > v) return 0;
    return 1;
}
template <typename T> __global__ void ridge_k(const uint8_t *p, const T *xin, int d, int h, int w, int thr, uint8_t *out) {
    const size_t n = (size_t)d * h * w, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int x = (int)(i % w), y = (int)((i / w) % h), z = (int)(i / ((size_t)w * h));
    float nz = ldin<T>(xin, n + i), ny = ldin<T>(xin, 2 * n + i), nx = ldin<T>(xin, 3 * n + i);
    const float m = sqrtf(nz * nz + ny * ny + nx * nx);
    if (m < 1e-3f) { out[i] = 0; return; }
    nz /= m; ny /= m; nx /= m;
    int r = 0;
    for (int s = -1; s <= 1 && !r; s++) r = is_ridge(p, d, h, w, z + s * nz, y + s * ny, x + s * nx, nz, ny, nx, thr);
    out[i] = r ? 255 : 0;
}
__global__ void fill_unl_k(uint8_t *t, uint8_t *m, const uint8_t *vm, const uint8_t *src, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n && !m[i] && vm[i]) { t[i] = src[i]; m[i] = 1; }
}
template <typename T> __global__ void add_into_k(T *y, const T *x, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) y[i] = f2h<T>((float)y[i] + (float)x[i]);
}
/* y += x in y's storage: MX when registered (both tensors in the same format; fp4 with stochastic rounding when SR is on),
   else 16-bit (h16 1: fp16 or bf16 by the 16-bit mode) or fp32 */
extern "C" void nn_add_into(void *y, const void *x, shape5 s, int h16) {
    const int dt = nn_storage(y);
    if (dt == 4 || dt == 8) {
        if (nn_storage(x) != dt) { fprintf(stderr, "nn_add_into: operands in different MX formats (%d, %d)\n", dt, nn_storage(x)); abort(); }
        lp_add_mx(y, x, s, dt, dt == 4 && sr_on() ? sr_seed() : 0u); return;
    }
    const size_t n = shape_numel(s);
    if (!h16) add_into_k<float><<<nblk(n, 256), 256>>>((float *)y, (const float *)x, n);
    else if (g_h16) add_into_k<f16><<<nblk(n, 256), 256>>>((f16 *)y, (const f16 *)x, n);
    else add_into_k<bf16><<<nblk(n, 256), 256>>>((bf16 *)y, (const bf16 *)x, n);
    KCHECK();
}
__global__ void side_tg_k(const uint8_t *s, size_t n, uint8_t *t, uint8_t *m) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) { const uint8_t v = s[i]; t[i] = v == 1 ? 255 : 0; m[i] = v <= 1; }
}
extern "C" void nn_side_targets(const uint8_t *side, size_t n, uint8_t *t, uint8_t *m) { side_tg_k<<<nblk(n, 256), 256>>>(side, n, t, m); KCHECK(); }
extern "C" void nn_fill_unlabelled(uint8_t *t, uint8_t *m, const uint8_t *vm, const uint8_t *src, size_t n) { fill_unl_k<<<nblk(n, 256), 256>>>(t, m, vm, src, n); KCHECK(); }
extern "C" void nn_ridge_u8(const uint8_t *p, const void *x, int xfmt, int d, int h, int w, int thr, uint8_t *out) {
    const size_t n = (size_t)d * h * w;
    if (xfmt == 1) ridge_k<__half><<<nblk(n, 256), 256>>>(p, (const __half *)x, d, h, w, thr, out);
    else if (xfmt == 2) ridge_k<__nv_bfloat16><<<nblk(n, 256), 256>>>(p, (const __nv_bfloat16 *)x, d, h, w, thr, out);
    else ridge_k<float><<<nblk(n, 256), 256>>>(p, (const float *)x, d, h, w, thr, out);
    KCHECK();
}
extern "C" void nn_pred_input(const uint8_t *ct, int W, float mean, float isd, const float *dyo, const float *dxo, int axis, void *x, int h16) {
    size_t n = (size_t)W * W * W;
    if (!h16) pred_in_k<float><<<nblk(n, 256), 256>>>(ct, W, mean, isd, dyo, dxo, axis, (float *)x);
    else if (g_h16) pred_in_k<f16><<<nblk(n, 256), 256>>>(ct, W, mean, isd, dyo, dxo, axis, (f16 *)x);
    else pred_in_k<bf16><<<nblk(n, 256), 256>>>(ct, W, mean, isd, dyo, dxo, axis, (bf16 *)x);
    KCHECK();
}
__global__ void pred_out_k(const float *lg, const uint8_t *ct, size_t n, uint8_t *out) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = ct[i] ? (uint8_t)(255.f / (1.f + __expf(-lg[i])) + 0.5f) : 0;
}
extern "C" void nn_pred_output(const float *lg, const uint8_t *ct, size_t n, uint8_t *out) { pred_out_k<<<nblk(n, 256), 256>>>(lg, ct, n, out); KCHECK(); }
__global__ void pred_place_k(const float *lg, const uint8_t *ct, int W, int halo, int oz, int oy, int ox, int ez, int ey, int ex, int shard, uint8_t *dsh) {
    const int I = W - 2 * halo;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)I * I * I) return;
    const int x = halo + (int)(i % I), y = halo + (int)((i / I) % I), z = halo + (int)(i / ((size_t)I * I));
    const int gz = oz + z, gy = oy + y, gx = ox + x;
    if (gz < 0 || gz >= ez || gy < 0 || gy >= ey || gx < 0 || gx >= ex) return;
    const size_t k = ((size_t)z * W + y) * W + x;
    dsh[((size_t)gz * shard + gy) * shard + gx] = ct[k] ? (uint8_t)(255.f / (1.f + __expf(-lg[k])) + 0.5f) : 0;
}
extern "C" void nn_pred_place(const float *lg, const uint8_t *ct, int W, int halo, int oz, int oy, int ox, int ez, int ey, int ex, int shard, uint8_t *dsh) {
    const size_t I = (size_t)(W - 2 * halo);
    pred_place_k<<<nblk(I * I * I, 256), 256>>>(lg, ct, W, halo, oz, oy, ox, ez, ey, ex, shard, dsh); KCHECK();
}
__global__ void pred_stats_k(const uint8_t *ct, size_t n, unsigned long long *acc) {
    unsigned long long nz = 0, sm = 0, sq = 0;
    for (size_t i = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) * 16; i < n; i += (size_t)gridDim.x * blockDim.x * 16) {
        if (i + 16 <= n && !(((uintptr_t)(ct + i)) & 15)) {
            const uint4 u = *(const uint4 *)(ct + i);
            const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
                for (int b = 0; b < 4; b++) { const unsigned v = (w[j] >> (8 * b)) & 255u; nz += v != 0; sm += v; sq += v * v; }
        } else for (size_t j = i; j < n && j < i + 16; j++) { const unsigned v = ct[j]; nz += v != 0; sm += v; sq += v * v; }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) { nz += __shfl_xor_sync(0xffffffffu, nz, o); sm += __shfl_xor_sync(0xffffffffu, sm, o); sq += __shfl_xor_sync(0xffffffffu, sq, o); }
    if ((threadIdx.x & 31) == 0) { atomicAdd(acc, nz); atomicAdd(acc + 1, sm); atomicAdd(acc + 2, sq); }
}
extern "C" void nn_pred_stats(const uint8_t *ct, size_t n, void *scratch, size_t *nz, double *sum, double *sq) {
    unsigned long long *acc = (unsigned long long *)scratch, h[3];
    cudaMemsetAsync(acc, 0, 3 * sizeof *acc);
    pred_stats_k<<<256, 256>>>(ct, n, acc); KCHECK();
    cudaMemcpy(h, acc, sizeof h, cudaMemcpyDeviceToHost);
    *nz = (size_t)h[0]; *sum = (double)h[1]; *sq = (double)h[2];
}
extern "C" void nn_up2_fwd_gn_into(const float *x, shape5 xs, const nn_gn_t *g, float *y, int ctot, int c0) {
    const gnp_t gp = to_gnp(g);
    if (ISMX(x)) { if (!ISMX(y) || ctot != xs.c || c0) { fprintf(stderr, "up2: MX input needs a whole MX output tensor (y MX %d, ctot %d, c %d, c0 %d)\n", MXDT(y), ctot, xs.c, c0); abort(); } lp_up2_fwd_mx(x, MXDT(x), xs, y, MXDT(y), gp); KCHECK(); return; }
    dim3 grid(nblk(2 * xs.w, 32), nblk(2 * xs.h, 8), (unsigned)(nblk(2 * xs.d, 4) * xs.n * xs.c));
    if (ABF && g_h16) up2_f_k<f16, f16><<<grid, 256>>>((const f16 *)x, (f16 *)y, xs.n, xs.c, xs.d, xs.h, xs.w, ctot, c0, gp);
    else if (ABF) up2_f_k<bf16, bf16><<<grid, 256>>>((const bf16 *)x, (bf16 *)y, xs.n, xs.c, xs.d, xs.h, xs.w, ctot, c0, gp);
    else up2_f_k<float, float><<<grid, 256>>>(x, y, xs.n, xs.c, xs.d, xs.h, xs.w, ctot, c0, gp);
    KCHECK();
}
extern "C" void nn_up2_fwd_into(const float *x, shape5 xs, float *y, int ctot, int c0) { nn_up2_fwd_gn_into(x, xs, nullptr, y, ctot, c0); }
extern "C" void nn_up2_fwd(const float *x, shape5 xs, float *y) { nn_up2_fwd_into(x, xs, y, xs.c, 0); }
extern "C" void nn_up2_fwd_mx_range(const float *x, shape5 xs, int c0, int nc, float *y) {
    if (!ISMX(x) || !ISMX(y) || c0 % 32 || nc % 32) { fprintf(stderr, "nn_up2_fwd_mx_range: MX tensors and 32-channel blocks only\n"); abort(); }
    lp_up2_fwd_mx_blocks(x, MXDT(x), xs, c0 / 32, nc / 32, y, MXDT(y)); KCHECK();
}
__global__ void add_rows_k(float *d, size_t dld, const float *s, size_t sld, int rows, size_t cols) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * cols) return;
    const size_t r = i / cols, c = i % cols;
    d[r * dld + c] += s[r * sld + c];
}
extern "C" void nn_add_rows(float *dst, size_t dld, const float *src, size_t sld, int rows, size_t cols) {
    if (rows < 1 || !cols) return;
    add_rows_k<<<nblk((size_t)rows * cols, 256), 256>>>(dst, dld, src, sld, rows, cols); KCHECK();
}
extern "C" void nn_up2_bwd_into(const float *gy, shape5 xs, float *gx, int ctot, int c0) {
    if (ISMX(gy)) { if (!ISMX(gx)) { fprintf(stderr, "up2_bwd: MX gy needs an MX gx\n"); abort(); } const unsigned osr_ = MXDT(gx) == 4 ? sr_seed() : 0u; if (ctot == xs.c && !c0) lp_up2_bwd_mx(gy, xs, gx, MXDT(gy), MXDT(gx), osr_); else lp_up2_bwd_mx_slice(gy, xs, gx, ctot, c0, MXDT(gy), MXDT(gx), osr_); KCHECK(); return; }
    dim3 grid(nblk(xs.w, 8), nblk(xs.h, 8), (unsigned)(nblk(xs.d, 4) * xs.n * xs.c));
    if (GBF && g_h16) up2_b_k<f16, f16><<<grid, 256>>>((const f16 *)gy, (f16 *)gx, xs.n * xs.c, xs.d, xs.h, xs.w, xs.c, ctot, c0);
    else if (GBF) up2_b_k<bf16, bf16><<<grid, 256>>>((const bf16 *)gy, (bf16 *)gx, xs.n * xs.c, xs.d, xs.h, xs.w, xs.c, ctot, c0);
    else up2_b_k<float, float><<<grid, 256>>>(gy, gx, xs.n * xs.c, xs.d, xs.h, xs.w, xs.c, ctot, c0);
    KCHECK();
}
extern "C" void nn_up2_bwd(const float *gy, shape5 xs, float *gx) { nn_up2_bwd_into(gy, xs, gx, xs.c, 0); }
__global__ void concat_k(const float *a, int ca, const float *b, int cb, float *y, int N, size_t S, int fwd) {
    int C = ca + cb;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * C * S) return;
    int c = (int)((i / S) % C), n = (int)(i / (S * C));
    size_t s = i % S;
    float *src = c < ca ? (float *)a + ((size_t)n * ca + c) * S + s : (float *)b + ((size_t)n * cb + c - ca) * S + s;
    if (fwd) y[i] = *src; else *src = y[i];
}
extern "C" void nn_concat_fwd(const float *a, int ca, const float *b, int cb, shape5 s, float *y) {
    size_t S = shape_spatial(s), n = (size_t)s.n * (ca + cb) * S;
    concat_k<<<nblk(n, 256), 256>>>(a, ca, b, cb, y, s.n, S, 1);
    KCHECK();
}
extern "C" void nn_concat_bwd(const float *gy, int ca, int cb, shape5 s, float *ga, float *gb) {
    size_t S = shape_spatial(s), n = (size_t)s.n * (ca + cb) * S;
    concat_k<<<nblk(n, 256), 256>>>(ga, ca, gb, cb, (float *)gy, s.n, S, 0);
    KCHECK();
}
