/* the on-the-fly ink chunk on the GPU (src/inkflygpu.h): inksplat.c inkfly_chunk's per-unit passes as three kernels.
   A: one thread per label pixel of the unit's candidate tiles: supervised, its mesh point and normal (ssurf_pn on the mesh patch), the
      near test, the face search (the [lo, hi] CT column, trilinear, [1 2 1] in float, face_of) -> a compacted (r, c, face) list.
   B: one thread per (faced pixel, sub-sample): ssurf_pn at the sub-sample, nt band samples -> the ink / no-ink hit counts and the max
      label per voxel (atomics: counts and max commute, so the order of the hits does not matter; the CPU's saturating uint8 counts are
      min(count, 255)).
   C: one thread per chunk voxel: the CPU's finalize (held-out boxes, CT >= iso, the ink share, prob254) merged by known max into the
      chunk; the accumulators are cleared for the next unit.
   Built with FMA contraction off (CMakeLists.txt UFSM_STRICT_UNITS, under UFSM_STRICT_MATH=ON; fast math by default): every double expression rounds as gcc -std=c23's (no contraction), sqrt / division are IEEE, so the
   bytes are the CPU's. Per sampler thread a stream and buffers that grow on demand, all of them under one budget (UFSM_INKFLY_GPU_MB,
   default 1024) and above a free-memory reserve for the trainer on the same device (UFSM_INKFLY_RESERVE_MB, default 1536): a chunk
   that does not fit is done on the CPU (return 1), the GPU path stays on. (2026-11-15 try7: every worker kept its own buffers on the
   training card without a bound; a cudaStreamCreate ran out of memory at start-up and switched the GPU path off for good, and the
   trainer died at step 450 with an unknown CUDA error at its next check.) */
#include "defaulted.h"
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "inkflygpu.h"

enum { TS = 64, FACE_NONE = -128, NSMAX = 64 };
#define IG_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "inkfly gpu: %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); rc = -1; goto out; } } while (0)

typedef struct {
    const uint8_t *lab, *msk; long long rh, rw, rr0, rc0, lh, lw; const int32_t *tiles; int ntiles, ntx;
    const float *X, *Y, *Z; int pu0, pv0, pw, ph, mw, mh; double scale, F; long long glr0, glc0; int nsub, nt; double tstep;
} du_t;
typedef struct { double offr, offc, sign, face_a, thr, V, off, slack; int lo, ns, iso, run, gap; long long a0[3], b0[3], co[3], cn[3], o[3], n[3], shape[3]; int nexcl; long long excl[INKFLY_MAXEXCL][6]; } dc_t;
typedef struct { int32_t r, c; int32_t fo; } frec;

__device__ static inline void d_grid(const du_t &u, const dc_t &c, double r, double cc, double *gu, double *gv) {
    *gu = ((cc + u.glc0 + 0.5) * u.F - 0.5 + c.offc) * u.scale; *gv = ((r + u.glr0 + 0.5) * u.F - 0.5 + c.offr) * u.scale;
}
__device__ static inline int d_pn(const du_t &u, double gu, double gv, double P[3], double N[3], int *bad) {
    if (gu < 0 || gv < 0 || gu > u.mw - 1 || gv > u.mh - 1) return -1;
    int c0 = (int)gu, r0 = (int)gv; if (c0 >= u.mw - 1) c0 = u.mw - 2; if (r0 >= u.mh - 1) r0 = u.mh - 2;
    if (c0 < u.pu0 || c0 + 1 >= u.pu0 + u.pw || r0 < u.pv0 || r0 + 1 >= u.pv0 + u.ph) { *bad = 1; return -1; }
    const double fu = gu - c0, fv = gv - r0; const size_t i00 = (size_t)(r0 - u.pv0) * u.pw + (c0 - u.pu0), i01 = i00 + 1, i10 = i00 + u.pw, i11 = i10 + 1;
    if (u.Z[i00] < 0 || u.Z[i01] < 0 || u.Z[i10] < 0 || u.Z[i11] < 0) return -1;
    const float *A[3] = {u.Z, u.Y, u.X}; double du[3], dv[3];
    for (int d = 0; d < 3; d++) {
        const double a = A[d][i00], b = A[d][i01], cc = A[d][i10], e = A[d][i11];
        P[d] = (1 - fv) * ((1 - fu) * a + fu * b) + fv * ((1 - fu) * cc + fu * e);
        du[d] = (1 - fv) * (b - a) + fv * (e - cc); dv[d] = (1 - fu) * (cc - a) + fu * (e - b);
    }
    N[0] = du[1] * dv[2] - du[2] * dv[1]; N[1] = du[2] * dv[0] - du[0] * dv[2]; N[2] = du[0] * dv[1] - du[1] * dv[0];
    const double nl = sqrt(N[0] * N[0] + N[1] * N[1] + N[2] * N[2]); if (nl < 1e-12) return -1;
    for (int d = 0; d < 3; d++) N[d] /= nl;
    return 0;
}
__device__ static inline float d_tri(const uint8_t *b, const long long n[3], double z, double y, double x) {
    if (z < 0 || y < 0 || x < 0 || z > n[0] - 1 || y > n[1] - 1 || x > n[2] - 1) return -1.f;
    long long z0 = (long long)z, y0 = (long long)y, x0 = (long long)x; if (z0 >= n[0] - 1) z0 = n[0] - 2; if (y0 >= n[1] - 1) y0 = n[1] - 2; if (x0 >= n[2] - 1) x0 = n[2] - 2;
    if (z0 < 0 || y0 < 0 || x0 < 0) return -1.f;
    const double fz = z - z0, fy = y - y0, fx = x - x0; double s = 0;
    for (int c = 0; c < 8; c++) {
        const long long a = z0 + (c >> 2 & 1), bb = y0 + (c >> 1 & 1), e = x0 + (c & 1);
        s += (c >> 2 & 1 ? fz : 1 - fz) * (c >> 1 & 1 ? fy : 1 - fy) * (c & 1 ? fx : 1 - fx) * b[((size_t)a * n[1] + bb) * n[2] + e];
    }
    return (float)s;
}
__device__ static int d_face_of(const float *s, int n, int lo, int iso, int run, int gap) {
    for (int i = 0; i < n;) {
        if (s[i] < iso) { i++; continue; }
        int j = i; while (j < n && s[j] >= iso) j++;
        const int t0 = lo + i, t1 = lo + j - 1;
        if (j - i >= run && t1 >= -4 && t0 <= 8) {
            int g = 0; while (j + g < n && s[j + g] < iso && g < gap) g++;
            return g >= gap && t1 >= -127 && t1 <= 127 ? t1 : FACE_NONE;
        }
        if (t0 > 8) return FACE_NONE;
        i = j;
    }
    return FACE_NONE;
}
/* inksplat.c tri_box_z: rows [s0, s1) of the box in b; -1 outside the box, -2 a corner row outside the slab */
__device__ static inline float d_tri_z(const uint8_t *b, const long long n[3], long long s0, long long s1, double z, double y, double x) {
    if (z < 0 || y < 0 || x < 0 || z > n[0] - 1 || y > n[1] - 1 || x > n[2] - 1) return -1.f;
    long long z0 = (long long)z, y0 = (long long)y, x0 = (long long)x; if (z0 >= n[0] - 1) z0 = n[0] - 2; if (y0 >= n[1] - 1) y0 = n[1] - 2; if (x0 >= n[2] - 1) x0 = n[2] - 2;
    if (z0 < 0 || y0 < 0 || x0 < 0) return -1.f;
    if (z0 < s0 || z0 + 1 >= s1) return -2.f;
    const double fz = z - z0, fy = y - y0, fx = x - x0; double s = 0;
    for (int c = 0; c < 8; c++) {
        const long long a = z0 + (c >> 2 & 1) - s0, bb = y0 + (c >> 1 & 1), e = x0 + (c & 1);
        s += (c >> 2 & 1 ? fz : 1 - fz) * (c >> 1 & 1 ? fy : 1 - fy) * (c & 1 ? fx : 1 - fx) * b[((size_t)a * n[1] + bb) * n[2] + e];
    }
    return (float)s;
}
typedef struct { double off, V, sign; int lo, ns, iso, run, gap; long long co[3], cn[3], z0, z1; } rp_t;
/* the MIL rays' face search (inksplat.c inkfly_rays mode 1): one ray a thread */
__global__ void k_rayface(rp_t p, const double *pn, unsigned n, const uint8_t *ct, uint8_t *drop, int *bad) {
    const unsigned id = blockIdx.x * blockDim.x + threadIdx.x; if (id >= n) return;
    const double *P = pn + 6 * (size_t)id, *N = P + 3; float raw[NSMAX], sm[NSMAX];
    for (int k = 0; k < p.ns; k++) {
        const double tt = (p.lo + k) * p.sign; double q[3];
        for (int d = 0; d < 3; d++) q[d] = (P[d] + tt * N[d] - p.off) / p.V - p.co[d];
        const float v = d_tri_z(ct, p.cn, p.z0, p.z1, q[0], q[1], q[2]); if (v < -1.5f) { atomicExch(bad, 1); return; } raw[k] = v < 0 ? 0 : v;
    }
    for (int k = 0; k < p.ns; k++) sm[k] = 0.25f * (raw[k > 0 ? k - 1 : 0] + 2 * raw[k] + raw[k < p.ns - 1 ? k + 1 : p.ns - 1]);
    drop[id] = d_face_of(sm, p.ns, p.lo, p.iso, p.run, p.gap) != FACE_NONE;
}
typedef struct { int32_t r, c; double P[3], N[3]; } nrec_t;
/* A1: the near pixels of one unit's candidate tiles, with their point / normal, and the level-0 box their samples reach */
__global__ void k_near(du_t u, const dc_t *cp, nrec_t *rec, unsigned *nrec, long long *bb, int *bad) {
    const dc_t &c = *cp;
    const int t = u.tiles[blockIdx.x], tx = t % u.ntx, ty = t / u.ntx;
    const long long r0 = (long long)ty * TS, c0 = (long long)tx * TS, r1 = r0 + TS < u.lh ? r0 + TS : u.lh, c1 = c0 + TS < u.lw ? c0 + TS : u.lw;
    for (int k = threadIdx.x; k < TS * TS; k += blockDim.x) {
        const long long r = r0 + k / TS - u.rr0, cc = c0 + k % TS - u.rc0;
        if (r + u.rr0 >= r1 || cc + u.rc0 >= c1) continue;
        if (u.msk && u.msk[(size_t)r * u.rw + cc] < 128) continue;
        double gu, gv, P[3], N[3]; d_grid(u, c, (double)r, (double)cc, &gu, &gv);
        if (d_pn(u, gu, gv, P, N, bad)) continue;
        int far = 0; for (int d = 0; d < 3 && !far; d++) far = P[d] < c.a0[d] - c.slack || P[d] > c.b0[d] + c.slack;
        if (far) continue;
        const unsigned j = atomicAdd(nrec, 1u); nrec_t *R = &rec[j]; R->r = (int32_t)r; R->c = (int32_t)cc;
        for (int d = 0; d < 3; d++) { R->P[d] = P[d]; R->N[d] = N[d];
            atomicMin(&bb[d], (long long)floor(P[d] - c.slack)); atomicMax(&bb[3 + d], (long long)ceil(P[d] + c.slack)); }
    }
}
/* A2: the face of each near pixel (the CT box now on the device) -> the faced list */
__global__ void k_face(const dc_t *cp, const uint8_t *ct, const nrec_t *rec, const unsigned *nrec, frec *out, unsigned *nout) {
    const dc_t &c = *cp; const unsigned id = blockIdx.x * blockDim.x + threadIdx.x; if (id >= *nrec) return;
    const nrec_t R = rec[id]; float raw[NSMAX], sm[NSMAX];
    for (int i = 0; i < c.ns; i++) {
        const double tt = (c.lo + i) * c.sign; double q[3];
        for (int d = 0; d < 3; d++) q[d] = (R.P[d] + tt * R.N[d] - c.off) / c.V - c.co[d];
        const float v = d_tri(ct, c.cn, q[0], q[1], q[2]); raw[i] = v < 0 ? 0 : v;
    }
    for (int i = 0; i < c.ns; i++) sm[i] = 0.25f * (raw[i > 0 ? i - 1 : 0] + 2 * raw[i] + raw[i < c.ns - 1 ? i + 1 : c.ns - 1]);
    const int fo = d_face_of(sm, c.ns, c.lo, c.iso, c.run, c.gap);
    if (fo == FACE_NONE) return;
    const unsigned j = atomicAdd(nout, 1u); out[j].r = R.r; out[j].c = R.c; out[j].fo = fo;
}
__global__ void k_splat(du_t u, const dc_t *cp, const frec *rec, const unsigned *nrec, unsigned *ni, unsigned *nn, unsigned *pi, unsigned *pn, int *bad) {
    const dc_t &c = *cp; const int ss = u.nsub * u.nsub;
    const long long id = (long long)blockIdx.x * blockDim.x + threadIdx.x; if (id >= (long long)*nrec * ss) return;
    const frec R = rec[id / ss]; const int a = (int)(id % ss) / u.nsub, b = (int)(id % ss) % u.nsub;
    const double ta = R.fo + c.face_a; const unsigned lv = u.lab[(size_t)R.r * u.rw + R.c]; const int ink = lv >= 128;
    double gu, gv, P2[3], N2[3];
    d_grid(u, c, (long long)R.r + (a + 0.5) / u.nsub - 0.5, (long long)R.c + (b + 0.5) / u.nsub - 0.5, &gu, &gv);
    if (d_pn(u, gu, gv, P2, N2, bad)) return;
    for (int k = 0; k < u.nt; k++) {
        const double tt = (ta + k * u.tstep) * c.sign; long long vi[3]; int in = 1;
        for (int d = 0; d < 3; d++) { vi[d] = (long long)floor((P2[d] + tt * N2[d] - c.off) / c.V + 0.5) - c.o[d]; if (vi[d] < 0 || vi[d] >= c.n[d]) { in = 0; break; } }
        if (!in) continue;
        const size_t k2 = ((size_t)vi[0] * c.n[1] + vi[1]) * c.n[2] + vi[2];
        if (ink) { atomicAdd(&ni[k2], 1u); atomicMax(&pi[k2], lv); } else { atomicAdd(&nn[k2], 1u); atomicMax(&pn[k2], lv); }
    }
}
__global__ void k_final(const dc_t *cp, const uint8_t *ct, unsigned *ni, unsigned *nn, unsigned *pi, unsigned *pn, uint8_t *out, int *nonempty) {
    const dc_t &c = *cp; const size_t cv = (size_t)c.n[0] * c.n[1] * c.n[2];
    const size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (k >= cv) return;
    const unsigned a = ni[k] < 255 ? ni[k] : 255, bq = nn[k] < 255 ? nn[k] : 255; const int h = (int)(a + bq); if (!h) return;
    const long long z = c.o[0] + (long long)(k / ((size_t)c.n[1] * c.n[2])), y = c.o[1] + (long long)(k / c.n[2] % c.n[1]), x = c.o[2] + (long long)(k % c.n[2]);
    if (z < c.shape[0] && y < c.shape[1] && x < c.shape[2]) {
        int ex = 0;
        for (int e = 0; e < c.nexcl && !ex; e++) { const long long *bx = c.excl[e], Vi = (long long)c.V;
            ex = (z + 1) * Vi > bx[0] && z * Vi < bx[0] + bx[3] && (y + 1) * Vi > bx[1] && y * Vi < bx[1] + bx[4] && (x + 1) * Vi > bx[2] && x * Vi < bx[2] + bx[5]; }
        const long long cz = z - c.co[0], cy = y - c.co[1], cx = x - c.co[2];
        const uint8_t cv8 = cz >= 0 && cy >= 0 && cx >= 0 && cz < c.cn[0] && cy < c.cn[1] && cx < c.cn[2] ? ct[((size_t)cz * c.cn[1] + cy) * c.cn[2] + cx] : 0;
        if (!ex && cv8 && cv8 >= c.iso) {
            const int isink = a && a >= c.thr * h;
            const uint8_t v = (uint8_t)(((isink ? pi[k] : pn[k]) * 254 + 127) / 255);
            out[k] = out[k] == 255 ? v : (out[k] > v ? out[k] : v); *nonempty = 1;
        }
    }
    ni[k] = nn[k] = pi[k] = pn[k] = 0;
}

/* per calling thread: a stream and its buffers (the sampler's workers run concurrently; the device is the process's choice) */
static int g_dev = 0;
enum { B_CT, B_LAB, B_MSK, B_TIL, B_X, B_Y, B_Z, B_NREC, B_FREC, B_ACC, B_OUT, B_SM, B_DU, B_RPN, B_RDR, NB };
typedef struct { int dev; cudaStream_t st; struct { void *p; size_t cap; } b[NB]; int nu; size_t base[1024]; du_t hu[1024]; } ictx;
typedef struct { dc_t c; long long bb[6]; int nonempty, bad; unsigned cnt[1024], fcnt[1024]; } sm_t;
static __thread ictx *t_ic;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static size_t g_held, g_budget, g_reserve;   /* mgpu: guarded (g_mu; one ink device per process). bytes in every thread's buffers; their bound; the device memory left to the trainer */
static void budget_init(void) {
    if (g_budget) return;
    const char *b = getenv("UFSM_INKFLY_GPU_MB"), *r = getenv("UFSM_INKFLY_RESERVE_MB");
    g_budget = (size_t)(b && atol(b) > 0 ? atol(b) : 1024) << 20; g_reserve = (size_t)(r && atol(r) >= 0 ? atol(r) : 1536) << 20;
}
/* 0: buffer i holds n bytes; 1: not within the budget / reserve / allocator (this chunk on the CPU; the old buffer is kept) */
static int grow(ictx *x, int i, size_t n) {
    if (x->b[i].cap >= n) return 0;
    const size_t c = n + n / 4; int rc = 1;
    pthread_mutex_lock(&g_mu); budget_init();
    size_t fr = 0, tot = 0;
    if (g_held - x->b[i].cap + c <= g_budget && cudaMemGetInfo(&fr, &tot) == cudaSuccess && fr + x->b[i].cap >= c + g_reserve) {
        void *p = nullptr;
        if (cudaMalloc(&p, c) == cudaSuccess) {
            if (x->b[i].p) cudaFree(x->b[i].p);
            g_held = g_held - x->b[i].cap + c; x->b[i].p = p; x->b[i].cap = c; rc = 0;
        } else (void)cudaGetLastError();   /* (the allocator's error is not sticky; clear it for the next check) */
    }
    pthread_mutex_unlock(&g_mu);
    return rc;
}
static void to_dc(const ifg_cfg *cf, dc_t *hc) {
    memset(hc, 0, sizeof *hc);
    hc->offr = cf->offr; hc->offc = cf->offc; hc->sign = cf->sign; hc->face_a = cf->face_a; hc->thr = cf->thr; hc->V = cf->V; hc->off = cf->off; hc->slack = cf->slack;
    hc->lo = cf->lo; hc->ns = cf->ns; hc->iso = cf->iso; hc->run = cf->run; hc->gap = cf->gap; hc->nexcl = cf->nexcl;
    for (int d = 0; d < 3; d++) { hc->a0[d] = cf->a0[d]; hc->b0[d] = cf->b0[d]; hc->shape[d] = cf->shape[d]; }
    for (int e = 0; e < cf->nexcl; e++) for (int d = 0; d < 6; d++) hc->excl[e][d] = cf->excl[e][d];
}
static unsigned long long g_over;   /* mgpu: guarded (g_mu) */
static void atomic_over(void) {
    pthread_mutex_lock(&g_mu); const unsigned long long k = ++g_over; const size_t h = g_held; pthread_mutex_unlock(&g_mu);
    if (k == 1) UFSM_DEFAULTED("inkfly = the CPU for chunks over the GPU budget (%zu of %zu MiB held, %zu MiB reserve: the device is full; the planner keeps "
                               "the budget free when the ink shares a training card, else lower UFSM_INKFLY_GPU_MB)", h >> 20, g_budget >> 20, g_reserve >> 20);
    if (k == 1 || k % 1000 == 0) fprintf(stderr, "inkfly gpu: %llu chunk(s) over the GPU budget (%zu of %zu MiB held, %zu MiB reserve): on the CPU\n", k, h >> 20, g_budget >> 20, g_reserve >> 20);
}
extern "C" unsigned long long inkfly_gpu_over(void) { pthread_mutex_lock(&g_mu); const unsigned long long k = g_over; pthread_mutex_unlock(&g_mu); return k; }
extern "C" void inkfly_gpu_device(int dev) { g_dev = dev; }
/* the trainer's memory planner (2026-11-20): the budget it keeps free on a training card that also runs the ink chunks, and the reserve
   from then on (the plan already holds the trainer's own memory, so the 1536 MiB default guard shrinks to what the planner left); an
   explicit UFSM_INKFLY_RESERVE_MB stays */
extern "C" size_t inkfly_gpu_budget(void) { pthread_mutex_lock(&g_mu); budget_init(); const size_t b = g_budget; pthread_mutex_unlock(&g_mu); return b; }
/* the budget from the device's free memory once the trainer holds its plan (2026-11-20, lead: FR1's P 384 windows hit the fixed 1024 MiB
   with 673 MiB held while the card had room): budget = what the ink holds + the free memory - keep, reserve = keep. An explicit
   UFSM_INKFLY_GPU_MB stays. Returns the budget (0: unchanged) */
extern "C" size_t inkfly_gpu_budget_from_free(size_t keep) {
    pthread_mutex_lock(&g_mu); budget_init(); size_t fr = 0, tot = 0, b = 0; const char *e = getenv("UFSM_INKFLY_GPU_MB");
    int prev = -1; cudaGetDevice(&prev);
    if (!(e && atol(e) > 0) && cudaSetDevice(g_dev) == cudaSuccess && cudaMemGetInfo(&fr, &tot) == cudaSuccess) {
        b = g_held + (fr > keep ? fr - keep : 0); if (b < ((size_t)64 << 20)) b = (size_t)64 << 20;
        g_budget = b; const char *r = getenv("UFSM_INKFLY_RESERVE_MB"); if (!(r && atol(r) >= 0)) g_reserve = keep;
    } else (void)cudaGetLastError();
    if (prev >= 0) cudaSetDevice(prev);
    pthread_mutex_unlock(&g_mu); return b;
}
extern "C" void inkfly_gpu_set_reserve(size_t bytes) {
    pthread_mutex_lock(&g_mu); budget_init(); const char *r = getenv("UFSM_INKFLY_RESERVE_MB"); if (!(r && atol(r) >= 0)) g_reserve = bytes; pthread_mutex_unlock(&g_mu);
}
extern "C" int inkfly_gpu_near(const ifg_unit *U, int nu, const ifg_cfg *cf, int64_t lo[3], int64_t hi[3], uint64_t *nnear) {
    int rc = 0; sm_t *hs = nullptr; ictx *x = t_ic;
    if (cf->ns > NSMAX || cf->nexcl > INKFLY_MAXEXCL || nu > 1024) return -2;
    if (!x) { x = t_ic = (ictx *)calloc(1, sizeof *x); x->dev = -1; }
    if (x->dev != g_dev) {
        IG_CK(cudaSetDevice(g_dev)); if (x->st) { cudaStreamDestroy(x->st); x->st = nullptr; }
        if (cudaStreamCreateWithFlags(&x->st, cudaStreamNonBlocking) != cudaSuccess) { (void)cudaGetLastError(); x->st = nullptr; return 1; }   /* (no stream now: this chunk on the CPU) */
        x->dev = g_dev;
    }
    IG_CK(cudaSetDevice(x->dev));
    {
        size_t nl = 0, nm = 0, nt = 0, np = 0, nr = 0;
        for (int i = 0; i < nu; i++) { const ifg_unit *h = &U[i]; nl += (size_t)h->rh * h->rw; nm += h->msk ? (size_t)h->rh * h->rw : 0; nt += h->ntiles; np += (size_t)h->pw * h->ph; nr += (size_t)h->ntiles * TS * TS; }
        if (grow(x, B_LAB, nl + 1) || grow(x, B_MSK, nm + 1) || grow(x, B_TIL, 4 * nt + 4) || grow(x, B_X, 4 * np + 4) || grow(x, B_Y, 4 * np + 4) || grow(x, B_Z, 4 * np + 4) ||
            grow(x, B_NREC, nr * sizeof(nrec_t) + 64) || grow(x, B_FREC, nr * sizeof(frec) + 64) || grow(x, B_SM, sizeof(sm_t))) { atomic_over(); rc = 1; goto out; }
        hs = (sm_t *)calloc(1, sizeof *hs); to_dc(cf, &hs->c);
        for (int d = 0; d < 3; d++) { hs->bb[d] = INT64_MAX; hs->bb[3 + d] = INT64_MIN; }
        sm_t *ds = (sm_t *)x->b[B_SM].p;
        IG_CK(cudaMemcpyAsync(ds, hs, sizeof *hs, cudaMemcpyHostToDevice, x->st));
        size_t ol = 0, om = 0, ot = 0, op = 0, orr = 0; x->nu = nu;
        for (int i = 0; i < nu; i++) {
            const ifg_unit *h = &U[i]; const size_t l = (size_t)h->rh * h->rw, p = (size_t)h->pw * h->ph; du_t &u = x->hu[i];
            uint8_t *dl = (uint8_t *)x->b[B_LAB].p + ol, *dm = h->msk ? (uint8_t *)x->b[B_MSK].p + om : nullptr; int32_t *dt = (int32_t *)x->b[B_TIL].p + ot;
            float *dx = (float *)x->b[B_X].p + op, *dy = (float *)x->b[B_Y].p + op, *dz = (float *)x->b[B_Z].p + op;
            IG_CK(cudaMemcpyAsync(dl, h->lab, l, cudaMemcpyHostToDevice, x->st)); if (dm) IG_CK(cudaMemcpyAsync(dm, h->msk, l, cudaMemcpyHostToDevice, x->st));
            IG_CK(cudaMemcpyAsync(dt, h->tiles, 4 * (size_t)h->ntiles, cudaMemcpyHostToDevice, x->st));
            IG_CK(cudaMemcpyAsync(dx, h->X, 4 * p, cudaMemcpyHostToDevice, x->st)); IG_CK(cudaMemcpyAsync(dy, h->Y, 4 * p, cudaMemcpyHostToDevice, x->st)); IG_CK(cudaMemcpyAsync(dz, h->Z, 4 * p, cudaMemcpyHostToDevice, x->st));
            u.lab = dl; u.msk = dm; u.rh = h->rh; u.rw = h->rw; u.rr0 = h->rr0; u.rc0 = h->rc0; u.lh = h->lh; u.lw = h->lw; u.tiles = dt; u.ntiles = h->ntiles; u.ntx = h->ntx;
            u.X = dx; u.Y = dy; u.Z = dz; u.pu0 = h->pu0; u.pv0 = h->pv0; u.pw = h->pw; u.ph = h->ph; u.mw = h->mw; u.mh = h->mh; u.scale = h->scale; u.F = h->F; u.glr0 = h->glr0; u.glc0 = h->glc0;
            u.nsub = h->nsub; u.nt = h->nt; u.tstep = h->tstep; x->base[i] = orr;
            if (h->ntiles) k_near<<<h->ntiles, 256, 0, x->st>>>(u, &ds->c, (nrec_t *)x->b[B_NREC].p + orr, &ds->cnt[i], ds->bb, &ds->bad);
            IG_CK(cudaGetLastError());
            ol += l; om += h->msk ? l : 0; ot += h->ntiles; op += p; orr += (size_t)h->ntiles * TS * TS;
        }
        IG_CK(cudaMemcpyAsync(hs, ds, sizeof *hs, cudaMemcpyDeviceToHost, x->st)); IG_CK(cudaStreamSynchronize(x->st));
        if (hs->bad) { rc = 1; goto out; }
        uint64_t s = 0; for (int i = 0; i < nu; i++) s += hs->cnt[i];
        *nnear = s; for (int d = 0; d < 3; d++) { lo[d] = hs->bb[d]; hi[d] = hs->bb[3 + d]; }
    }
out:
    free(hs); return rc;
}
static unsigned long long g_rays;   /* mgpu: guarded (g_mu) */
extern "C" unsigned long long inkfly_gpu_rays(void) { pthread_mutex_lock(&g_mu); const unsigned long long k = g_rays; pthread_mutex_unlock(&g_mu); return k; }
extern "C" int inkfly_gpu_rayface(const ifg_ray *p, const double *pn, size_t n, const uint8_t *ct, uint8_t *drop) {
    int rc = 0; ictx *x = t_ic; int hbad = 0;
    if (!n) return 0;
    if (p->ns > NSMAX || n > 0xffffffffu) return 1;
    if (!x) { x = t_ic = (ictx *)calloc(1, sizeof *x); x->dev = -1; }
    if (x->dev != g_dev) {
        IG_CK(cudaSetDevice(g_dev)); if (x->st) { cudaStreamDestroy(x->st); x->st = nullptr; }
        if (cudaStreamCreateWithFlags(&x->st, cudaStreamNonBlocking) != cudaSuccess) { (void)cudaGetLastError(); x->st = nullptr; return 1; }
        x->dev = g_dev;
    }
    IG_CK(cudaSetDevice(x->dev));
    {
        const size_t ncv = (size_t)(p->z1 - p->z0) * p->cn[1] * p->cn[2], nd = (n + 15) & ~(size_t)15;
        if (grow(x, B_CT, ncv) || grow(x, B_RPN, n * 6 * sizeof(double)) || grow(x, B_RDR, nd + 16)) { atomic_over(); return 1; }
        rp_t h; h.off = p->off; h.V = p->V; h.sign = p->sign; h.lo = p->lo; h.ns = p->ns; h.iso = p->iso; h.run = p->run; h.gap = p->gap;
        for (int d = 0; d < 3; d++) { h.co[d] = p->co[d]; h.cn[d] = p->cn[d]; } h.z0 = p->z0; h.z1 = p->z1;
        uint8_t *dd = (uint8_t *)x->b[B_RDR].p; int *db = (int *)(dd + nd);
        IG_CK(cudaMemcpyAsync(x->b[B_CT].p, ct, ncv, cudaMemcpyHostToDevice, x->st));
        IG_CK(cudaMemcpyAsync(x->b[B_RPN].p, pn, n * 6 * sizeof(double), cudaMemcpyHostToDevice, x->st));
        IG_CK(cudaMemsetAsync(db, 0, sizeof(int), x->st));
        k_rayface<<<(unsigned)((n + 127) / 128), 128, 0, x->st>>>(h, (const double *)x->b[B_RPN].p, (unsigned)n, (const uint8_t *)x->b[B_CT].p, dd, db);
        IG_CK(cudaGetLastError());
        IG_CK(cudaMemcpyAsync(drop, dd, n, cudaMemcpyDeviceToHost, x->st));
        IG_CK(cudaMemcpyAsync(&hbad, db, sizeof(int), cudaMemcpyDeviceToHost, x->st));
        IG_CK(cudaStreamSynchronize(x->st));
        if (hbad) return 2;
        pthread_mutex_lock(&g_mu); g_rays += n; pthread_mutex_unlock(&g_mu);
    }
out:
    return rc;
}
extern "C" int inkfly_gpu_finish(const ifg_cfg *cf, const uint8_t *ctb, const int64_t co[3], const int64_t cn[3], const int64_t o[3], const int64_t n[3], uint8_t *out, int *nonempty) {
    int rc = 0; ictx *x = t_ic; if (!x || x->dev < 0) return -2;
    const size_t cv = (size_t)n[0] * n[1] * n[2], ncv = (size_t)cn[0] * cn[1] * cn[2];
    dc_t *hc = (dc_t *)calloc(1, sizeof *hc); int hb[2] = {0, 0};
    IG_CK(cudaSetDevice(x->dev));
    if (grow(x, B_CT, ncv) || grow(x, B_ACC, 4 * cv * sizeof(unsigned)) || grow(x, B_OUT, cv)) { atomic_over(); rc = 1; goto out; }
    {
        to_dc(cf, hc); for (int d = 0; d < 3; d++) { hc->co[d] = co[d]; hc->cn[d] = cn[d]; hc->o[d] = o[d]; hc->n[d] = n[d]; }
        sm_t *ds = (sm_t *)x->b[B_SM].p; unsigned *acc = (unsigned *)x->b[B_ACC].p, *ni = acc, *nn = acc + cv, *pi = acc + 2 * cv, *pn = acc + 3 * cv; uint8_t *dout = (uint8_t *)x->b[B_OUT].p;
        IG_CK(cudaMemcpyAsync(&ds->c, hc, sizeof *hc, cudaMemcpyHostToDevice, x->st));
        IG_CK(cudaMemcpyAsync(x->b[B_CT].p, ctb, ncv, cudaMemcpyHostToDevice, x->st));
        IG_CK(cudaMemsetAsync(acc, 0, 4 * cv * sizeof(unsigned), x->st)); IG_CK(cudaMemsetAsync(dout, 255, cv, x->st));
        for (int i = 0; i < x->nu; i++) {
            const du_t &u = x->hu[i]; if (!u.ntiles) continue;
            const size_t cap = (size_t)u.ntiles * TS * TS; const int ss = u.nsub * u.nsub;
            k_face<<<(unsigned)((cap + 255) / 256), 256, 0, x->st>>>(&ds->c, (const uint8_t *)x->b[B_CT].p, (const nrec_t *)x->b[B_NREC].p + x->base[i], &ds->cnt[i], (frec *)x->b[B_FREC].p + x->base[i], &ds->fcnt[i]);
            k_splat<<<(unsigned)((cap * ss + 255) / 256), 256, 0, x->st>>>(u, &ds->c, (const frec *)x->b[B_FREC].p + x->base[i], &ds->fcnt[i], ni, nn, pi, pn, &ds->bad);
            k_final<<<(unsigned)((cv + 255) / 256), 256, 0, x->st>>>(&ds->c, (const uint8_t *)x->b[B_CT].p, ni, nn, pi, pn, dout, &ds->nonempty);
            IG_CK(cudaGetLastError());
        }
        IG_CK(cudaMemcpyAsync(hb, &ds->nonempty, 2 * sizeof(int), cudaMemcpyDeviceToHost, x->st));
        IG_CK(cudaMemcpyAsync(out, dout, cv, cudaMemcpyDeviceToHost, x->st));
        IG_CK(cudaStreamSynchronize(x->st));
        if (hb[1]) { rc = 1; goto out; }
        *nonempty = hb[0];
    }
out:
    free(hc); return rc;
}
