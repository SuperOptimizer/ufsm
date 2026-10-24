/* Radial axial attention (src/attn.h at_rays): the ray plan on the host, the stencil weights, the gather / scatter between grid token rows
   and ray samples (bilinear in the axis-normal plane, edge-clamped), and the scroll coordinates (r, arc, k) for RoPE. */
#include "attn.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static unsigned nbk(long long n, int t) { return (unsigned)((n + t - 1) / t); }
static int imax3(int a, int b, int c) { int m = a > b ? a : b; return m > c ? m : c; }

extern "C" void at_rays_dims(int D, int Hh, int W, float mult, int *Q, int *Nt, int *Nr) {
    const int P = imax3(D, Hh, W);
    int nt = (int)ceilf(mult * (float)P); if (nt < 1) nt = 1;
    int a = D, b = Hh, c = W, t;   /* the two largest dims: the longest plane diagonal over the axis choices */
    if (a > b) { t = a; a = b; b = t; }
    if (b > c) { t = b; b = c; c = t; }
    if (a > b) { t = a; a = b; b = t; }
    *Nr = (int)ceil(sqrt((double)b * b + (double)c * c)) + 1; *Nt = nt; *Q = P * nt;
}

/* one ray from (cu, cv) at angle th through the rectangle [-0.5, Nu - 0.5] x [-0.5, Nv - 0.5]: unit-spaced samples centred on the chord */
static void ray_polar(float *rr, double cu, double cv, double th, double rc, int Nu, int Nv, int Nr, int k) {
    const double du = cos(th), dv = sin(th), lo[2] = {-0.5, -0.5}, hi[2] = {Nu - 0.5, Nv - 0.5}, c[2] = {cu, cv}, d[2] = {du, dv};
    double tlo = 0, thi = 1e300; int miss = 0;
    for (int e = 0; e < 2; e++) {
        if (fabs(d[e]) < 1e-12) { if (c[e] < lo[e] || c[e] > hi[e]) miss = 1; continue; }
        double t1 = (lo[e] - c[e]) / d[e], t2 = (hi[e] - c[e]) / d[e];
        if (t1 > t2) { const double t = t1; t1 = t2; t2 = t; }
        if (t1 > tlo) tlo = t1;
        if (t2 < thi) thi = t2;
    }
    const double chord = thi - tlo;
    memset(rr, 0, AT_RAY_F * sizeof *rr); rr[7] = (float)k;
    if (miss || !(chord > 1e-6)) return;   /* (a dummy: len 0) */
    int L = chord < 1 ? 1 : (int)floor(chord); if (L > Nr) L = Nr;
    const double t0 = tlo + 0.5 * (chord - (L - 1));
    rr[0] = (float)(cu + t0 * du); rr[1] = (float)(cv + t0 * dv); rr[2] = (float)du; rr[3] = (float)dv;
    rr[4] = (float)t0; rr[5] = (float)(th * rc); rr[6] = (float)L;
}

extern "C" void at_rays_plan(int n, int D, int Hh, int W, float mult, const float *axr, float *ray_h, int *lens_h, int *ad_h, float *cen_h) {
    int Q, Nt, Nr; at_rays_dims(D, Hh, W, mult, &Q, &Nt, &Nr);
    const int P = imax3(D, Hh, W);
    for (int b = 0; b < n; b++) {
        const float *a = axr + (size_t)b * AT_AXR(P);
        int ad = (int)a[0]; if (ad < 0 || ad > 2) ad = 0;
        const int valid = a[1] != 0.f;
        int Np, Nu, Nv; at_ray_plane(ad, D, Hh, W, &Np, &Nu, &Nv);
        ad_h[b] = ad;
        float *rb = ray_h + (size_t)b * Q * AT_RAY_F; int *lb = lens_h + (size_t)b * Q; float *cb = cen_h + (size_t)b * P * 4;
        memset(cb, 0, (size_t)P * 4 * sizeof *cb);
        for (int k = 0; k < P; k++) for (int j = 0; j < Nt; j++) {
            float *rr = rb + ((size_t)k * Nt + j) * AT_RAY_F;
            memset(rr, 0, AT_RAY_F * sizeof *rr); rr[7] = (float)k;
            if (k >= Np) continue;
            if (!valid) {   /* the grid rows along v */
                if (j < Nu) { rr[0] = (float)j; rr[3] = 1.f; rr[5] = (float)j; rr[6] = (float)(Nv < Nr ? Nv : Nr); }
                continue;
            }
            const double cu = a[2 + 2 * k], cv = a[3 + 2 * k], mu = 0.5 * (Nu - 1), mv = 0.5 * (Nv - 1);
            const int inside = cu >= -0.5 && cu <= Nu - 0.5 && cv >= -0.5 && cv <= Nv - 0.5;
            double rc = hypot(mu - cu, mv - cv); if (rc < 0.5 * (Nu > Nv ? Nu : Nv)) rc = 0.5 * (Nu > Nv ? Nu : Nv);
            double phim = 0, th0 = -M_PI, th1 = M_PI;
            if (!inside) {
                phim = atan2(mv - cv, mu - cu); th0 = 1e300; th1 = -1e300;
                for (int q = 0; q < 4; q++) {
                    const double qu = q & 1 ? Nu - 0.5 : -0.5, qv = q & 2 ? Nv - 0.5 : -0.5;
                    const double dl = remainder(atan2(qv - cv, qu - cu) - phim, 2 * M_PI);
                    if (phim + dl < th0) th0 = phim + dl;
                    if (phim + dl > th1) th1 = phim + dl;
                }
            }
            if (j == 0) { cb[4 * k] = (float)cu; cb[4 * k + 1] = (float)cv; cb[4 * k + 2] = (float)rc; cb[4 * k + 3] = (float)phim; }
            ray_polar(rr, cu, cv, th0 + (j + 0.5) * (th1 - th0) / Nt, rc, Nu, Nv, Nr, k);
        }
        if (!valid) for (int k = 0; k < P; k++) cb[4 * k + 2] = -1.f;   /* (no axis: at_scroll_pos uses the grid indices) */
        for (int q = 0; q < Q; q++) { const int L = (int)rb[(size_t)q * AT_RAY_F + 6]; lb[q] = L > 0 ? L : 1; }
    }
}

/* ---- device ---- */
struct rgeo { int Q, Nr, D, Hh, W; long long Sg; };
static rgeo geo_of(const at_rays *R) { return rgeo{R->Q, R->Nr, R->D, R->Hh, R->W, (long long)R->D * R->Hh * R->W}; }
/* the corners of ray slot s: grid rows (b Sg + token) and weights; returns the count */
__device__ __forceinline__ int slot_corners(const rgeo &G, const float *ray, const int *ad, long long s, long long *row, float *w) {
    const long long rq = s / G.Nr; const int i = (int)(s - rq * G.Nr), b = (int)(rq / G.Q);
    const float *rr = ray + rq * AT_RAY_F; const int a_ = ad[b];
    int Np, Nu, Nv; at_ray_plane(a_, G.D, G.Hh, G.W, &Np, &Nu, &Nv);
    int ia[4], ib[4]; const int nc = at_ray_corners_h(rr, i, Nu, Nv, ia, ib, w);
    const int k = (int)rr[7];
    for (int c = 0; c < nc; c++) row[c] = (long long)b * G.Sg + at_ray_tok(a_, k, ia[c], ib[c], G.Hh, G.W);
    return nc;
}
__global__ void k_ray_pos(const float *ray, int Nr, long long n, float *pos) {
    const long long s = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (s >= n) return;
    const float *rr = ray + (s / Nr) * AT_RAY_F; const int i = (int)(s % Nr);
    pos[3 * s] = rr[4] + (float)i; pos[3 * s + 1] = rr[5]; pos[3 * s + 2] = rr[7];
}
__global__ void k_ray_wacc(rgeo G, const float *ray, const int *ad, long long n, float *acc) {
    const long long s = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (s >= n) return;
    long long row[4]; float w[4]; const int nc = slot_corners(G, ray, ad, s, row, w);
    for (int c = 0; c < nc; c++) if (w[c] != 0.f) atomicAdd(acc + row[c], w[c]);
}
__global__ void k_ray_winv(float *acc, long long n) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) { const float a = acc[i]; acc[i] = a > 1e-6f ? 1.f / a : 0.f; }
}
extern "C" void at_rays_setup(const at_rays *R) {
    const rgeo G = geo_of(R); const long long Tr = (long long)R->n * R->Q * R->Nr, Tg = (long long)R->n * G.Sg;
    k_ray_pos<<<nbk(Tr, 256), 256>>>(R->ray, R->Nr, Tr, (float *)R->pos);
    cudaMemsetAsync((void *)R->wsum, 0, (size_t)Tg * sizeof(float), 0);
    k_ray_wacc<<<nbk(Tr, 256), 256>>>(G, R->ray, R->ad, Tr, (float *)R->wsum);
    k_ray_winv<<<nbk(Tg, 256), 256>>>((float *)R->wsum, Tg);
}
template <int NORM>
__global__ void k_ray_gather(rgeo G, const float *ray, const int *ad, const float *__restrict__ wsum, long long n, int C4, const float *__restrict__ x, float *__restrict__ y) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n * C4) return;
    const long long s = i / C4; const int c4 = (int)(i - s * C4);
    long long row[4]; float w[4]; const int nc = slot_corners(G, ray, ad, s, row, w);
    float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int c = 0; c < nc; c++) {
        const float wc = NORM ? w[c] * wsum[row[c]] : w[c];
        if (wc == 0.f) continue;
        const float4 v = ((const float4 *)(x + row[c] * 4 * C4))[c4];
        a.x += wc * v.x; a.y += wc * v.y; a.z += wc * v.z; a.w += wc * v.w;
    }
    ((float4 *)(y + s * 4 * C4))[c4] = a;
}
template <int NORM>
__global__ void k_ray_scatter(rgeo G, const float *ray, const int *ad, const float *__restrict__ wsum, long long n, int C4, const float *__restrict__ y, float *x) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n * C4) return;
    const long long s = i / C4; const int c4 = (int)(i - s * C4);
    long long row[4]; float w[4]; const int nc = slot_corners(G, ray, ad, s, row, w);
    if (!nc) return;
    const float4 v = ((const float4 *)(y + s * 4 * C4))[c4];
    for (int c = 0; c < nc; c++) {
        const float wc = NORM ? w[c] * wsum[row[c]] : w[c];
        if (wc == 0.f) continue;
        float *d = x + row[c] * 4 * C4 + 4 * c4;
        atomicAdd(d, wc * v.x); atomicAdd(d + 1, wc * v.y); atomicAdd(d + 2, wc * v.z); atomicAdd(d + 3, wc * v.w);
    }
}
extern "C" void at_ray_gather(const at_rays *R, const float *x, float *y, int Cc, int norm) {
    const rgeo G = geo_of(R); const long long Tr = (long long)R->n * R->Q * R->Nr; const int C4 = Cc / 4;
    if (norm) k_ray_gather<1><<<nbk(Tr * C4, 256), 256>>>(G, R->ray, R->ad, R->wsum, Tr, C4, x, y);
    else k_ray_gather<0><<<nbk(Tr * C4, 256), 256>>>(G, R->ray, R->ad, R->wsum, Tr, C4, x, y);
}
extern "C" void at_ray_scatter(const at_rays *R, const float *y, float *x, int Cc, int norm) {
    const rgeo G = geo_of(R); const long long Tr = (long long)R->n * R->Q * R->Nr; const int C4 = Cc / 4;
    cudaMemsetAsync(x, 0, (size_t)R->n * G.Sg * Cc * sizeof(float), 0);
    if (norm) k_ray_scatter<1><<<nbk(Tr * C4, 256), 256>>>(G, R->ray, R->ad, R->wsum, Tr, C4, y, x);
    else k_ray_scatter<0><<<nbk(Tr * C4, 256), 256>>>(G, R->ray, R->ad, R->wsum, Tr, C4, y, x);
}

__global__ void k_scroll_pos(long long n, int S, int nreg, int D, int Hh, int W, int P, const int *ad, const float *cen, float *pos) {
    const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int b = (int)(t / S), ts = (int)(t % S) - nreg;
    float *p = pos + 3 * t;
    if (ts < 0) { p[0] = p[1] = p[2] = 0.f; return; }
    const int z = ts / (Hh * W), y = (ts / W) % Hh, x = ts % W, a_ = ad[b];
    const int k = a_ == 0 ? z : a_ == 1 ? y : x, ua = a_ == 0 ? y : z, vb = a_ == 2 ? y : x;
    const float *c = cen + ((size_t)b * P + k) * 4;
    if (c[2] < 0.f) { p[0] = (float)vb; p[1] = (float)ua; p[2] = (float)k; return; }
    const float du = (float)ua - c[0], dv = (float)vb - c[1];
    const float th = c[3] + remainderf(atan2f(dv, du) - c[3], 6.283185307179586f);
    p[0] = sqrtf(du * du + dv * dv); p[1] = th * c[2]; p[2] = (float)k;
}
extern "C" void at_scroll_pos(int n, int nreg, int D, int Hh, int W, const int *ad, const float *cen, float *pos) {
    const int S = nreg + D * Hh * W; const long long N = (long long)n * S;
    k_scroll_pos<<<nbk(N, 256), 256>>>(N, S, nreg, D, Hh, W, imax3(D, Hh, W), ad, cen, pos);
}
