/* recto-sp's per-cell profiles on the GPU (src/rectogpu.h). Mirrors src/rectosp.c (rs_axis, rs_tri, the non --cls part of rs_rows)
   and src/rectoruler.c (rr_peaks without the material mask, which recto-sp sets to all ones, and without the half-prominence widths,
   which only --cls reads) operation by operation in fp64. UFSM_EXACT_CU (CMakeLists.txt): no FMA contraction, IEEE division and
   sqrt in every build, so each code is the CPU's (tests/test_recto_gpu.c compares them byte for byte). One thread per cell over a
   fixed grid (grid-stride), its profile, smoothed profile and hits in global scratch laid out [sample][thread] (coalesced while the
   warp's threads walk their profiles in step). fp64 runs at 1/64 rate on consumer cards; the profiles are ~70 fp64 ops a sample,
   which the 1/64 rate still covers at tens of millions of cells a second. */
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rectogpu.h"

#define RG_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "recto-sp gpu: %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); return -1; } } while (0)
enum { RT = 128 };   /* threads per block */

static struct {
    int dev, ok; int64_t RZ, ny, nx; uint8_t *ring;
    float *scr; long G; int ncap;           /* scratch: G threads x ncap floats (v, tmp, hit pos, hit height) */
    int32_t *zyx; uint8_t *t6, *q; int Mcap; unsigned long long *cnt;
} g;

extern "C" size_t rsg_free_mem(int dev) {
    int n = 0; if (cudaGetDeviceCount(&n) != cudaSuccess || dev < 0 || dev >= n) { cudaGetLastError(); return 0; }
    if (cudaSetDevice(dev) != cudaSuccess) { cudaGetLastError(); return 0; }
    size_t fr = 0, tot = 0; if (cudaMemGetInfo(&fr, &tot) != cudaSuccess) { cudaGetLastError(); return 0; }
    return fr;
}

extern "C" int rsg_open(int dev, int64_t RZ, int64_t ny, int64_t nx) {
    rsg_close();
    RG_CK(cudaSetDevice(dev)); g.dev = dev; g.RZ = RZ; g.ny = ny; g.nx = nx;
    RG_CK(cudaMalloc(&g.ring, (size_t)RZ * ny * nx));
    RG_CK(cudaMalloc(&g.cnt, 2 * sizeof(unsigned long long)));
    int sms = 0; RG_CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    g.G = ((long)sms * 1536 + RT - 1) / RT * RT; g.ok = 1;   /* (full residency; the scratch is G x ncap floats, set at the first run) */
    return 0;
}
extern "C" void rsg_close(void) {
    if (!g.ok) return;
    cudaFree(g.ring); cudaFree(g.scr); cudaFree(g.zyx); cudaFree(g.t6); cudaFree(g.q); cudaFree(g.cnt);
    memset(&g, 0, sizeof g);
}
extern "C" int rsg_put(int64_t z0, int64_t nz, const uint8_t *src) {
    const size_t sl = (size_t)g.ny * g.nx;
    for (int64_t z = z0; z < z0 + nz;) {   /* contiguous slot runs */
        const int64_t s = z % g.RZ, k = nz - (z - z0) < g.RZ - s ? nz - (z - z0) : g.RZ - s;
        RG_CK(cudaMemcpy(g.ring + (size_t)s * sl, src + (size_t)(z - z0) * sl, (size_t)k * sl, cudaMemcpyHostToDevice));
        z += k;
    }
    return 0;
}

/* src/rectosp.c rs_axis */
__device__ static int d_axis(const uint8_t *t, double v[3]) {
    double T[3][3]; const double a[6] = {t[0] / 255.0, t[1] / 255.0, t[2] / 255.0, t[3] / 255.0 - 0.5, t[4] / 255.0 - 0.5, t[5] / 255.0 - 0.5};
    T[0][0] = a[0]; T[1][1] = a[1]; T[2][2] = a[2]; T[0][1] = T[1][0] = a[3]; T[0][2] = T[2][0] = a[4]; T[1][2] = T[2][1] = a[5];
    int k = T[0][0] >= T[1][1] && T[0][0] >= T[2][2] ? 0 : T[1][1] >= T[2][2] ? 1 : 2; v[0] = v[1] = v[2] = 0; v[k] = 1;
    for (int it = 0; it < 12; it++) {
        double w[3] = {0, 0, 0}; for (int p = 0; p < 3; p++) for (int q = 0; q < 3; q++) w[p] += T[p][q] * v[q];
        const double l = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]); if (l < 1e-9) return -1; for (int p = 0; p < 3; p++) v[p] = w[p] / l;
    }
    return 0;
}
/* src/rectosp.c rs_tri: the box bounds of the block (so / sn), the bytes from the ring */
__device__ static double d_tri(const rsg_par &P, const uint8_t *ring, int64_t RZ, double z, double y, double x) {
    z -= P.so[0]; y -= P.so[1]; x -= P.so[2];
    const int64_t z0 = (int64_t)floor(z), y0 = (int64_t)floor(y), x0 = (int64_t)floor(x);
    if (z0 < 0 || y0 < 0 || x0 < 0 || z0 + 1 >= P.sn[0] || y0 + 1 >= P.sn[1] || x0 + 1 >= P.sn[2]) return NAN;
    const double tz = z - z0, ty = y - y0, tx = x - x0; const size_t sy = P.sn[2], sl = (size_t)P.sn[1] * P.sn[2];
    const int64_t gz = P.so[0] + z0;   /* (the ring's y / x box is the block's: rsg_run checks) */
    const uint8_t *p = ring + (size_t)(gz % RZ) * sl + (size_t)y0 * sy + x0, *r = ring + (size_t)((gz + 1) % RZ) * sl + (size_t)y0 * sy + x0;
    const double a = (1 - ty) * ((1 - tx) * p[0] + tx * p[1]) + ty * ((1 - tx) * p[sy] + tx * p[sy + 1]);
    const double c = (1 - ty) * ((1 - tx) * r[0] + tx * r[1]) + ty * ((1 - tx) * r[sy] + tx * r[sy + 1]);
    return (1 - tz) * a + tz * c;
}

#define V(i) scr[(size_t)(i) * G]
#define TM(i) scr[(size_t)(n + (i)) * G]
#define HP(i) scr[(size_t)(2 * n + (i)) * G]
#define HH(i) scr[(size_t)(2 * n + nhc + (i)) * G]

__global__ void k_rsp(const rsg_par P, const uint8_t *ring, int64_t RZ, int M, const int32_t *zyx, const uint8_t *t6, uint8_t *qo, float *scr0, long G,
                      unsigned long long *cnt) {
    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x; if (tid >= G) return;
    float *scr = scr0 + tid; const int n = P.n, c = P.n / 2, nhc = n / 2 + 2, R = P.R;
    unsigned long long na = 0, ne = 0;
    for (long m = tid; m < M; m += G) {
        qo[m] = 0;
        double nv[3]; if (d_axis(t6 + (size_t)m * 6, nv)) continue;
        na++;
        const int zz = zyx[3 * m], yy = zyx[3 * m + 1], xx = zyx[3 * m + 2];
        const double p[3] = {P.org[0] + (zz + 0.5) * P.h, P.org[1] + (yy + 0.5) * P.h, P.org[2] + (xx + 0.5) * P.h};
        const double st = P.st;
        for (int i = 0; i < n; i++) { const double s = (i - c) * st; V(i) = (float)d_tri(P, ring, RZ, (p[0] + s * nv[0] + 0.5) * P.F - 0.5, (p[1] + s * nv[1] + 0.5) * P.F - 0.5, (p[2] + s * nv[2] + 0.5) * P.F - 0.5); }
        /* rr_peaks (mat all ones) */
        for (int i = 0; i < n; i++) {
            if (!R) { TM(i) = V(i); continue; }
            double s = 0, ws = 0;
            for (int d = -R; d <= R; d++) { const int k = i + d; if (k < 0 || k >= n) continue; const float vk = V(k); if (!(vk == vk)) continue; const double gg = P.gk[d + R]; s += gg * vk; ws += gg; }
            const float vi = V(i); TM(i) = ws > 0 && vi == vi ? (float)(s / ws) : NAN;
        }
        int nh = 0; const int W = P.Wn;
        for (int i = 1; i + 1 < n; i++) {
            const float ti = TM(i);
            if (!(ti == ti) || !(ti > TM(i - 1))) continue;
            int e = i; while (e + 1 < n && TM(e + 1) == ti) e++;
            if (e + 1 >= n || !(TM(e + 1) < ti)) { i = e; continue; }
            double ml = ti, mr = ti;
            for (int q = i - 1; q >= 0 && q >= i - W; q--) { const float tq = TM(q); if (!(tq == tq)) break; if (tq > ti) break; if (tq < ml) ml = tq; }
            for (int q = e + 1; q < n && q <= e + W; q++) { const float tq = TM(q); if (!(tq == tq)) break; if (tq > ti) break; if (tq < mr) mr = tq; }
            const double pr = ti - fmax(ml, mr);
            if (pr >= P.prom) {
                double pos = 0.5 * (i + e);
                if (e == i) { const double a = TM(i - 1), b = ti, cc = TM(i + 1), den = a - 2 * b + cc; if (den < 0) { const double d = 0.5 * (a - cc) / den; if (fabs(d) < 1) pos = i + d; } }
                HP(nh) = (float)pos; HH(nh) = ti; nh++;
            }
            i = e;
        }
        if (P.sep > 0) {
            int again = 1;
            while (again && nh > 1) {
                again = 0; int w = 0;
                for (int i = 0; i < nh; i++) {
                    if (w > 0 && HP(i) - HP(w - 1) < P.sep) { if (HH(i) > HH(w - 1)) { HP(w - 1) = HP(i); HH(w - 1) = HH(i); } again = 1; continue; }
                    if (w != i) { HP(w) = HP(i); HH(w) = HH(i); } w++;
                }
                nh = w;
            }
        }
        /* rs_rows: the hit interval around the centre, the median of the up to 5 around it */
        int j = -1; for (int k = 0; k + 1 < nh; k++) if (HP(k) <= c && HP(k + 1) > c) { j = k; break; }
        if (j < 0) continue;
        const double inner = P.half1 ? P.inner : 1e30; int inonly = 1, mi = P.minint;
        { int ni = 0; for (int k = j - 2; k <= j + 2; k++) if (k >= 0 && k + 1 < nh && HP(k) >= c - inner && HP(k + 1) <= c + inner) ni++;
          if (ni < P.minint && P.half1) { inonly = 0; mi = 2; } }
        float wd[5]; int nw = 0;
        for (int k = j - 2; k <= j + 2; k++) if (k >= 0 && k + 1 < nh && (!inonly || (HP(k) >= c - inner && HP(k + 1) <= c + inner))) wd[nw++] = HP(k + 1) - HP(k);
        if (nw < mi) continue;
        for (int a = 1; a < nw; a++) { const float x = wd[a]; int b = a - 1; while (b >= 0 && wd[b] > x) { wd[b + 1] = wd[b]; b--; } wd[b + 1] = x; }
        const double sp = wd[nw / 2] * st * 8 / P.h;
        qo[m] = (uint8_t)(sp < 1 ? 1 : sp > 255 ? 255 : lround(sp));
        ne++;
    }
    if (na) atomicAdd(&cnt[0], na);
    if (ne) atomicAdd(&cnt[1], ne);
}

extern "C" int rsg_run(const rsg_par *P, int M, const int32_t *zyx, const uint8_t *t6, uint8_t *q, long long *nax, long long *nest) {
    if (!g.ok) return -1;
    if (P->R > RSG_RMAX) return -1;
    const int need = 2 * P->n + 2 * (P->n / 2 + 2);
    if (need > g.ncap) { cudaFree(g.scr); g.scr = nullptr; RG_CK(cudaMalloc(&g.scr, (size_t)g.G * need * sizeof(float))); g.ncap = need; }
    if (M > g.Mcap) { cudaFree(g.zyx); cudaFree(g.t6); cudaFree(g.q); g.zyx = nullptr; g.t6 = g.q = nullptr; const int cap = M + M / 4;
                      RG_CK(cudaMalloc(&g.zyx, (size_t)cap * 3 * sizeof(int32_t))); RG_CK(cudaMalloc(&g.t6, (size_t)cap * 6)); RG_CK(cudaMalloc(&g.q, cap)); g.Mcap = cap; }
    if (M <= 0) return 0;
    RG_CK(cudaMemcpy(g.zyx, zyx, (size_t)M * 3 * sizeof(int32_t), cudaMemcpyHostToDevice));
    RG_CK(cudaMemcpy(g.t6, t6, (size_t)M * 6, cudaMemcpyHostToDevice));
    RG_CK(cudaMemset(g.cnt, 0, 2 * sizeof(unsigned long long)));
    if (P->sn[1] != g.ny || P->sn[2] != g.nx) { fprintf(stderr, "recto-sp gpu: block box %lld x %lld is not the ring's %lld x %lld\n", (long long)P->sn[1], (long long)P->sn[2], (long long)g.ny, (long long)g.nx); return -1; }
    const long G = M < g.G ? ((long)(M + RT - 1) / RT) * RT : g.G;   /* (threads = the scratch stride; G <= g.G fits the allocation) */
    k_rsp<<<(unsigned)(G / RT), RT>>>(*P, g.ring, g.RZ, M, g.zyx, g.t6, g.q, g.scr, G, g.cnt);
    RG_CK(cudaGetLastError());
    unsigned long long c[2]; RG_CK(cudaMemcpy(c, g.cnt, sizeof c, cudaMemcpyDeviceToHost));
    RG_CK(cudaMemcpy(q, g.q, M, cudaMemcpyDeviceToHost));
    *nax += (long long)c[0]; *nest += (long long)c[1];
    return 0;
}
