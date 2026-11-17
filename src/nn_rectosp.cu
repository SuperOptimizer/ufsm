/* recto-sp's per-cell profiles on the GPU (src/rectogpu.h). Mirrors src/rectosp.c (rs_axis, rs_tri, the non --cls part of rs_rows)
   and src/rectoruler.c (rr_peaks without the material mask, which recto-sp sets to all ones, and without the half-prominence widths,
   which only --cls reads) operation by operation in fp64. UFSM_EXACT_CU (CMakeLists.txt): no FMA contraction, IEEE division and
   sqrt in every build, so each code is the CPU's (tests/test_recto_gpu.c compares them byte for byte). One warp per cell (k_rsp).
   fp64 runs at 1/64 rate on consumer cards and a sample costs ~90 fp64 operations (no FMA): that is the kernel's floor. */
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "rectogpu.h"

#define RG_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "recto-sp gpu: %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); return -1; } } while (0)

static struct {
    int dev, ok; int64_t RZ, ny, nx; uint8_t *ring;
    long G;                                 /* resident threads (the grid's cap) */
    int32_t *zyx; uint8_t *t6, *q; double *nv; int Mcap, nres; unsigned long long *cnt;   /* nres: k_rsp blocks resident at once */
    cudaStream_t su, sk;                    /* uploads (rsg_put, from the helper thread) beside the kernels */
} g;

extern "C" size_t rsg_free_mem(int dev) {
    int n = 0; if (cudaGetDeviceCount(&n) != cudaSuccess || dev < 0 || dev >= n) { cudaGetLastError(); return 0; }
    if (cudaSetDevice(dev) != cudaSuccess) { cudaGetLastError(); return 0; }
    size_t fr = 0, tot = 0; if (cudaMemGetInfo(&fr, &tot) != cudaSuccess) { cudaGetLastError(); return 0; }
    return fr;
}

extern "C" int rsg_open(int dev, int64_t RZ, int64_t ny, int64_t nx, int maxM) {
    rsg_close();
    RG_CK(cudaSetDevice(dev)); g.dev = dev; g.RZ = RZ; g.ny = ny; g.nx = nx; g.ok = 1;   /* (ok: rsg_close frees a partial open) */
    RG_CK(cudaMalloc(&g.ring, (size_t)RZ * ny * nx));
    RG_CK(cudaMalloc(&g.cnt, 2 * sizeof(unsigned long long)));
    RG_CK(cudaStreamCreateWithFlags(&g.su, cudaStreamNonBlocking)); RG_CK(cudaStreamCreateWithFlags(&g.sk, cudaStreamNonBlocking));
    int sms = 0; RG_CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    g.G = (long)sms * 1536;
    if (maxM > 0) {   /* the cell arrays up front: no device allocation once the run is under way */
        RG_CK(cudaMalloc(&g.zyx, (size_t)maxM * 3 * sizeof(int32_t))); RG_CK(cudaMalloc(&g.t6, (size_t)maxM * 6)); RG_CK(cudaMalloc(&g.q, maxM));
        RG_CK(cudaMalloc(&g.nv, (size_t)maxM * 3 * sizeof(double))); g.Mcap = maxM; }
    return 0;
}
extern "C" void rsg_close(void) {
    if (!g.ok) return;
    cudaSetDevice(g.dev); if (g.su) cudaStreamDestroy(g.su); if (g.sk) cudaStreamDestroy(g.sk);
    cudaFree(g.ring); cudaFree(g.nv); cudaFree(g.zyx); cudaFree(g.t6); cudaFree(g.q); cudaFree(g.cnt);
    cudaGetLastError(); memset(&g, 0, sizeof g);
}
extern "C" uint8_t *rsg_slot(int64_t z) { return g.ok ? g.ring + (size_t)(z % g.RZ) * g.ny * g.nx : nullptr; }
extern "C" int rsg_zero(int64_t z0, int64_t nz) {
    RG_CK(cudaSetDevice(g.dev));
    const size_t sl = (size_t)g.ny * g.nx;
    for (int64_t z = z0; z < z0 + nz;) { const int64_t s = z % g.RZ, k = nz - (z - z0) < g.RZ - s ? nz - (z - z0) : g.RZ - s; RG_CK(cudaMemsetAsync(g.ring + (size_t)s * sl, 0, (size_t)k * sl, g.su)); z += k; }
    RG_CK(cudaStreamSynchronize(g.su));
    return 0;
}
extern "C" int rsg_put(int64_t z0, int64_t nz, const uint8_t *src) {
    RG_CK(cudaSetDevice(g.dev));   /* (any host thread: the helper uploads while the main thread's kernel runs) */
    const size_t sl = (size_t)g.ny * g.nx;
    for (int64_t z = z0; z < z0 + nz;) {   /* contiguous slot runs */
        const int64_t s = z % g.RZ, k = nz - (z - z0) < g.RZ - s ? nz - (z - z0) : g.RZ - s;
        RG_CK(cudaMemcpyAsync(g.ring + (size_t)s * sl, src + (size_t)(z - z0) * sl, (size_t)k * sl, cudaMemcpyHostToDevice, g.su));
        z += k;
    }
    RG_CK(cudaStreamSynchronize(g.su));
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
/* the cells' axes, one thread each (k_rsp's warps would compute each 32 times: 40 % of its fp64 work); NaN: no axis */
__global__ void k_axis(int M, const uint8_t *t6, double *nvo) {
    const long m = (long)blockIdx.x * blockDim.x + threadIdx.x; if (m >= M) return;
    double v[3]; if (d_axis(t6 + (size_t)m * 6, v)) v[0] = v[1] = v[2] = NAN;
    nvo[3 * m] = v[0]; nvo[3 * m + 1] = v[1]; nvo[3 * m + 2] = v[2];
}
/* src/rectosp.c rs_tri, split: d_loc places sample i of the profile (the same doubles: the coordinates, the block box's bounds, the
   floor and fractions) and tells NaN (outside the block's source box: 0), eight zero corners (1: the CPU's sum is +0 exactly, every
   weight >= 0 and every product +0) or a sample to interpolate (2); d_lerp is the trilinear sum. */
struct d_at { double tz, ty, tx; const uint8_t *p, *r; };
__device__ __forceinline__ int d_loc(const rsg_par &P, const uint8_t *ring, int64_t RZ, int i, int c, const double p[3], const double nv[3], d_at &A) {
    const double s = (i - c) * P.st; double z, y, x;
    if (P.F == 1.0) { z = (p[0] + s * nv[0] + 0.5) - 0.5; y = (p[1] + s * nv[1] + 0.5) - 0.5; x = (p[2] + s * nv[2] + 0.5) - 0.5; }   /* (x * 1.0 is x) */
    else { z = (p[0] + s * nv[0] + 0.5) * P.F - 0.5; y = (p[1] + s * nv[1] + 0.5) * P.F - 0.5; x = (p[2] + s * nv[2] + 0.5) * P.F - 0.5; }
    z -= P.so[0]; y -= P.so[1]; x -= P.so[2];
    const int64_t z0 = (int64_t)floor(z), y0 = (int64_t)floor(y), x0 = (int64_t)floor(x);
    if (z0 < 0 || y0 < 0 || x0 < 0 || z0 + 1 >= P.sn[0] || y0 + 1 >= P.sn[1] || x0 + 1 >= P.sn[2]) return 0;
    A.tz = z - z0; A.ty = y - y0; A.tx = x - x0; const size_t sy = P.sn[2], sl = (size_t)P.sn[1] * P.sn[2];
    const int64_t gz = P.so[0] + z0;   /* (the ring's y / x box is the block's: rsg_run checks) */
    A.p = ring + (size_t)(gz % RZ) * sl + (size_t)y0 * sy + x0; A.r = ring + (size_t)((gz + 1) % RZ) * sl + (size_t)y0 * sy + x0;
    const uint8_t *q = A.p, *r = A.r;
    return q[0] | q[1] | q[sy] | q[sy + 1] | r[0] | r[1] | r[sy] | r[sy + 1] ? 2 : 1;
}
__device__ __forceinline__ double d_lerp(const d_at &A, size_t sy) {
    const double tz = A.tz, ty = A.ty, tx = A.tx; const uint8_t *p = A.p, *r = A.r;
    const double a = (1 - ty) * ((1 - tx) * p[0] + tx * p[1]) + ty * ((1 - tx) * p[sy] + tx * p[sy + 1]);
    const double c = (1 - ty) * ((1 - tx) * r[0] + tx * r[1]) + ty * ((1 - tx) * r[sy] + tx * r[sy + 1]);
    return (1 - tz) * a + tz * c;
}
/* the smallest float >= v / the largest float <= v: float x >= v exactly when x >= f_up(v) (and <= likewise), so the interval tests
   run in fp32 */
__device__ __forceinline__ float f_next(float f, int up) {   /* nextafterf toward +inf (up) or -inf, f finite (mcc's device library has no nextafterf) */
    if (f == 0.0f) return __int_as_float(up ? 1 : (int)0x80000001u);
    const int b = __float_as_int(f); return __int_as_float((f > 0) == (up != 0) ? b + 1 : b - 1);
}
__device__ __forceinline__ float f_up(double v) { float f = (float)v; if ((double)f < v) f = f_next(f, 1); return f; }
__device__ __forceinline__ float f_dn(double v) { float f = (float)v; if ((double)f > v) f = f_next(f, 0); return f; }

/* one warp per cell. The fp64 pipe of a consumer card (1/64 rate) runs a warp instruction at the same cost whatever its active lanes,
   so the fp64 work runs on compacted lists in full warps: the samples to interpolate (their zero-corner neighbours are 0 at once), the
   smoothed samples with a nonzero window; rr_peaks' candidates (each i with tmp[i] > tmp[i - 1] stands alone: the samples its plateau
   skip passes over equal tmp[i] and are never candidates) search their prominence in fp32 (minima of floats) and are compacted in
   order by ballot; lane 0 merges the hits and picks the median interval (its interval tests in fp32 against f_up / f_dn bounds).
   Shared per warp: v[n], tmp[n], hit pos and height [n / 2 + 2] each, a work list [n] of shorts. */
__global__ void k_rsp(const rsg_par P, const uint8_t *ring, int64_t RZ, int M, const int32_t *zyx, const double *nvi, uint8_t *qo, unsigned long long *cnt) {
    extern __shared__ float sm[];
    const int n = P.n, c = P.n / 2, nhc = n / 2 + 2, R = P.R, lane = threadIdx.x & 31, wib = threadIdx.x >> 5;
    float *V = sm + (size_t)wib * (2 * n + 2 * nhc + (n + 1) / 2), *TM = V + n, *HP = TM + n, *HH = HP + nhc; short *WL = (short *)(HH + nhc);
    const long nw = (long)gridDim.x * (blockDim.x >> 5); const size_t sy = P.sn[2];
    unsigned long long na = 0, ne = 0; const unsigned FULL = 0xffffffffu, below = (1u << lane) - 1;
    double wsum = 0; for (int d = -R; d <= R && R; d++) wsum += P.gk[d + R];   /* (rr_peaks' ws where every tap counts, in its order; device-side: the host's fast math may reassociate a sum) */
    const double inner = P.half1 ? P.inner : 1e30; const float ilo = f_up(c - inner), ihi = f_dn(c + inner);
    for (long m = (long)blockIdx.x * (blockDim.x >> 5) + wib; m < M; m += nw) {
        const double nv[3] = {nvi[3 * m], nvi[3 * m + 1], nvi[3 * m + 2]}; if (!(nv[0] == nv[0])) { if (!lane) qo[m] = 0; continue; }
        na++;
        const int zz = zyx[3 * m], yy = zyx[3 * m + 1], xx = zyx[3 * m + 2];
        const double p[3] = {P.org[0] + (zz + 0.5) * P.h, P.org[1] + (yy + 0.5) * P.h, P.org[2] + (xx + 0.5) * P.h};
        int nwl = 0;
        for (int b0 = 0; b0 < n; b0 += 32) {   /* the profile: NaN / zero samples at once, the rest listed */
            const int i = b0 + lane; d_at A; int k = 0;
            if (i < n) { k = d_loc(P, ring, RZ, i, c, p, nv, A); if (k < 2) V[i] = k ? 0.0f : NAN; }
            const unsigned bal = __ballot_sync(FULL, k == 2);
            if (k == 2) WL[nwl + __popc(bal & below)] = (short)i;
            nwl += __popc(bal);
        }
        __syncwarp();
        for (int j = lane; j < nwl; j += 32) { const int i = WL[j]; d_at A; d_loc(P, ring, RZ, i, c, p, nv, A); V[i] = (float)d_lerp(A, sy); }
        __syncwarp();
        nwl = 0;
        for (int b0 = 0; b0 < n; b0 += 32) {   /* rr_peaks' smoothing (mat all ones): NaN / all-zero windows at once, the rest listed */
            const int i = b0 + lane; int work = 0;
            if (i < n) {
                if (!R) TM[i] = V[i];
                else {
                    const float vi = V[i]; int nz = vi == vi && vi == 0.0f;   /* (valid samples all 0, v[i] valid: s = +0, ws > 0, tmp = +0) */
                    for (int d = -R; d <= R && nz; d++) { const int k = i + d; if (k < 0 || k >= n) continue; const float vk = V[k]; if (vk == vk && vk != 0.0f) nz = 0; }
                    if (!(vi == vi)) TM[i] = NAN; else if (nz) TM[i] = 0.0f; else work = 1;
                }
            }
            const unsigned bal = __ballot_sync(FULL, work);
            if (work) WL[nwl + __popc(bal & below)] = (short)i;
            nwl += __popc(bal);
        }
        __syncwarp();
        for (int j = lane; j < nwl; j += 32) {
            const int i = WL[j]; double s = 0, ws = 0;
            if (i >= R && i + R < n) {   /* inside: ws is the sum of every tap (wsum, the same additions) unless a sample is NaN */
                int nan = 0; for (int d = -R; d <= R; d++) { const float vk = V[i + d]; if (vk == vk) s += P.gk[d + R] * vk; else nan = 1; }
                if (nan) { s = 0; for (int d = -R; d <= R; d++) { const float vk = V[i + d]; if (!(vk == vk)) continue; const double gg = P.gk[d + R]; s += gg * vk; ws += gg; } }
                else ws = wsum;
            } else
                for (int d = -R; d <= R; d++) { const int k = i + d; if (k < 0 || k >= n) continue; const float vk = V[k]; if (!(vk == vk)) continue; const double gg = P.gk[d + R]; s += gg * vk; ws += gg; }
            TM[i] = ws > 0 ? (float)(s / ws) : NAN;   /* (v[i] is valid here) */
        }
        __syncwarp();
        int nh = 0; const int W = P.Wn;
        for (int b0 = 0; b0 < n; b0 += 32) {   /* the candidates of 32 samples at once */
            const int i = b0 + lane; int hit = 0; float hp = 0, hh = 0;
            if (i >= 1 && i + 1 < n) {
                const float ti = TM[i];
                if (ti == ti && ti > TM[i - 1]) {
                    int e = i; while (e + 1 < n && TM[e + 1] == ti) e++;
                    if (e + 1 < n && TM[e + 1] < ti) {
                        float ml = ti, mr = ti;   /* (the CPU's doubles hold these floats: the same minima) */
                        for (int q = i - 1; q >= 0 && q >= i - W; q--) { const float tq = TM[q]; if (!(tq == tq)) break; if (tq > ti) break; if (tq < ml) ml = tq; }
                        for (int q = e + 1; q < n && q <= e + W; q++) { const float tq = TM[q]; if (!(tq == tq)) break; if (tq > ti) break; if (tq < mr) mr = tq; }
                        const double pr = (double)ti - (double)fmaxf(ml, mr);
                        if (pr >= P.prom) {
                            double pos = 0.5 * (i + e);
                            if (e == i) { const double a = TM[i - 1], bb = ti, cc = TM[i + 1], den = a - 2 * bb + cc; if (den < 0) { const double d = 0.5 * (a - cc) / den; if (fabs(d) < 1) pos = i + d; } }
                            hit = 1; hp = (float)pos; hh = ti;
                        }
                    }
                }
            }
            const unsigned bal = __ballot_sync(FULL, hit);
            if (hit) { const int r = nh + __popc(bal & below); HP[r] = hp; HH[r] = hh; }
            nh += __popc(bal);
        }
        __syncwarp();
        if (!lane) {
            if (P.sep > 0) {
                const float sepf = f_up(P.sep);   /* (HP[i] - HP[w - 1] < sep, a float difference: < f_up(sep) is the same test) */
                int again = 1;
                while (again && nh > 1) {
                    again = 0; int w = 0;
                    for (int i = 0; i < nh; i++) {
                        if (w > 0 && HP[i] - HP[w - 1] < sepf) { if (HH[i] > HH[w - 1]) { HP[w - 1] = HP[i]; HH[w - 1] = HH[i]; } again = 1; continue; }
                        HP[w] = HP[i]; HH[w] = HH[i]; w++;
                    }
                    nh = w;
                }
            }
            qo[m] = 0;
            /* rs_rows: the hit interval around the centre, the median of the up to 5 around it */
            int j = -1; for (int k = 0; k + 1 < nh; k++) if (HP[k] <= c && HP[k + 1] > c) { j = k; break; }
            if (j < 0) goto done;
            int inonly = 1, mi = P.minint;
            { int ni = 0; for (int k = j - 2; k <= j + 2; k++) if (k >= 0 && k + 1 < nh && HP[k] >= ilo && HP[k + 1] <= ihi) ni++;
              if (ni < P.minint && P.half1) { inonly = 0; mi = 2; } }
            float wd[5]; int nwd = 0;
            for (int k = j - 2; k <= j + 2; k++) if (k >= 0 && k + 1 < nh && (!inonly || (HP[k] >= ilo && HP[k + 1] <= ihi))) wd[nwd++] = HP[k + 1] - HP[k];
            if (nwd < mi) goto done;
            for (int a = 1; a < nwd; a++) { const float x = wd[a]; int b = a - 1; while (b >= 0 && wd[b] > x) { wd[b + 1] = wd[b]; b--; } wd[b + 1] = x; }
            const double sp = wd[nwd / 2] * P.st * 8 / P.h;
            qo[m] = (uint8_t)(sp < 1 ? 1 : sp > 255 ? 255 : lround(sp));
            ne++;
        }
    done:
        __syncwarp();
    }
    if (!lane && na) atomicAdd(&cnt[0], na);
    if (!lane && ne) atomicAdd(&cnt[1], ne);
}

extern "C" int rsg_run(const rsg_par *P, int M, const int32_t *zyx, const uint8_t *t6, uint8_t *q, long long *nax, long long *nest) {
    if (!g.ok) return -1;
    RG_CK(cudaSetDevice(g.dev));
    if (P->R > RSG_RMAX) return -1;
    const int wpb = 4; const size_t smem = (size_t)wpb * (2 * P->n + 2 * (P->n / 2 + 2) + (P->n + 1) / 2) * sizeof(float);
    if (smem > 48 * 1024) { fprintf(stderr, "recto-sp gpu: profiles of %d samples do not fit shared memory\n", P->n); return -1; }
    if (M > g.Mcap) { cudaFree(g.zyx); cudaFree(g.t6); cudaFree(g.q); cudaFree(g.nv); g.zyx = nullptr; g.t6 = g.q = nullptr; g.nv = nullptr; const int cap = M + M / 4;
                      RG_CK(cudaMalloc(&g.zyx, (size_t)cap * 3 * sizeof(int32_t))); RG_CK(cudaMalloc(&g.t6, (size_t)cap * 6)); RG_CK(cudaMalloc(&g.q, cap));
                      RG_CK(cudaMalloc(&g.nv, (size_t)cap * 3 * sizeof(double))); g.Mcap = cap; }
    if (M <= 0) return 0;
    RG_CK(cudaMemcpyAsync(g.zyx, zyx, (size_t)M * 3 * sizeof(int32_t), cudaMemcpyHostToDevice, g.sk));
    RG_CK(cudaMemcpyAsync(g.t6, t6, (size_t)M * 6, cudaMemcpyHostToDevice, g.sk));
    RG_CK(cudaMemsetAsync(g.cnt, 0, 2 * sizeof(unsigned long long), g.sk));
    if (P->sn[1] != g.ny || P->sn[2] != g.nx) { fprintf(stderr, "recto-sp gpu: block box %lld x %lld is not the ring's %lld x %lld\n", (long long)P->sn[1], (long long)P->sn[2], (long long)g.ny, (long long)g.nx); return -1; }
    if (!g.nres) {   /* the grid: the blocks resident at once (a grid-stride grid past that ran a partial last wave: up to a third of the time) */
        int nb = 0, sms = 0; RG_CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_rsp, wpb * 32, smem)); RG_CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, g.dev));
        g.nres = (nb > 0 ? nb : 1) * sms; }
    k_axis<<<(unsigned)((M + 255) / 256), 256, 0, g.sk>>>(M, g.t6, g.nv);
    const long nb = (M + wpb - 1) / wpb < g.nres ? (M + wpb - 1) / wpb : g.nres;
    k_rsp<<<(unsigned)nb, wpb * 32, smem, g.sk>>>(*P, g.ring, g.RZ, M, g.zyx, g.nv, g.q, g.cnt);
    RG_CK(cudaGetLastError());
    unsigned long long c[2]; RG_CK(cudaMemcpyAsync(c, g.cnt, sizeof c, cudaMemcpyDeviceToHost, g.sk));
    RG_CK(cudaMemcpyAsync(q, g.q, M, cudaMemcpyDeviceToHost, g.sk));
    RG_CK(cudaStreamSynchronize(g.sk));
    *nax += (long long)c[0]; *nest += (long long)c[1];
    return 0;
}
