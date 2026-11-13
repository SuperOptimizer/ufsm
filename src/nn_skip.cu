/* Empty-space skipping (--skip-empty 1, DESIGN 2026-11-13 "empty-space skipping"): per level a mask over tiles of 8^3 cells of that
   level's grid; a masked 3^3 conv (conv_fwd_f4_k, the network's wide stride-1 convs) skips the main loop of every output tile whose
   tile is inactive, so its outputs there are the bias alone (the epilogue, the MX store and the GroupNorm sums as usual): the network
   is the dense one with those conv outputs replaced by their bias. Not exact: scratch2's input is never spatially constant (the 3
   radial-direction channels, nn_input_latc) and the attention / gfilm stages couple every decoder voxel to the whole window, so no
   tile is provably constant; the margin (tiles of dilation per level) bounds how far the bias fill reaches into material.
   Masks are per device (the host thread's current one) and apply to an MX forward of N 1 whose output grid is exactly a registered
   level grid (no z-window / z-range / slab outputs: those stay dense). */
#include "nn_common.cuh"
#include "nn_lp.h"

#define SK_LEV 8
typedef struct { int on, nl; int D[SK_LEV], H[SK_LEV], W[SK_LEV], mz[SK_LEV], my[SK_LEV], mx[SK_LEV]; uint8_t *m[SK_LEV]; size_t cap[SK_LEV]; uint8_t *t0; size_t t0cap; long used, calls; } skreg_t;
static skreg_t g_sk[8];
static int skdev(void) { int d = 0; cudaGetDevice(&d); return d & 7; }

/* one block per level-0 tile (ts^3 voxels): any voxel above thr */
__global__ void sk_tile_any_k(const uint8_t *ct, int D, int H, int W, int ts, int tz, int ty, int tx, int thr, uint8_t *t0) {
    const int b = blockIdx.x, x0 = (b % tx) * ts, y0 = (b / tx % ty) * ts, z0 = (b / tx / ty) * ts;
    int any = 0;
    for (int i = threadIdx.x; i < ts * ts * ts && !any; i += blockDim.x) {
        const int x = x0 + i % ts, y = y0 + i / ts % ts, z = z0 + i / ts / ts;
        if (x < W && y < H && z < D && ct[((size_t)z * H + y) * W + x] > thr) any = 1;
    }
    any = __syncthreads_or(any);
    if (threadIdx.x == 0) t0[b] = (uint8_t)any;
}
/* level l: tile (a, b, c) of 8^3 level-l cells = 2^l level-0 tiles a side; active when any level-0 tile within r level-l tiles is */
__global__ void sk_level_k(const uint8_t *t0, int tz, int ty, int tx, int l, int r, int mz, int my, int mx, uint8_t *m) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= mz * my * mx) return;
    const int c = i % mx, b = i / mx % my, a = i / mx / my, f = 1 << l;
    const int za = max((a - r) * f, 0), zb = min((a + r + 1) * f, tz), ya = max((b - r) * f, 0), yb = min((b + r + 1) * f, ty), xa = max((c - r) * f, 0), xb = min((c + r + 1) * f, tx);
    uint8_t on = 0;
    for (int z = za; z < zb && !on; z++) for (int y = ya; y < yb && !on; y++) for (int x = xa; x < xb && !on; x++) on = t0[((size_t)z * ty + y) * tx + x];
    m[i] = on;
}
static uint8_t *sk_buf(uint8_t **p, size_t *cap, size_t n) {
    if (*cap < n) { if (*p) cudaFree(*p); *p = nullptr; *cap = 0; if (cudaMalloc((void **)p, n) != cudaSuccess) { cudaGetLastError(); return nullptr; } *cap = n; }
    return *p;
}
extern "C" int nn_skip_build(const uint8_t *ct, int D, int H, int W, int cell, int nlev, int thr, int margin) {
    skreg_t *s = &g_sk[skdev()]; s->on = 0;
    if (nlev > SK_LEV) nlev = SK_LEV;
    const int ts = 8 * cell, tz = (D + ts - 1) / ts, ty = (H + ts - 1) / ts, tx = (W + ts - 1) / ts;
    if (!sk_buf(&s->t0, &s->t0cap, (size_t)tz * ty * tx)) return 1;
    sk_tile_any_k<<<tz * ty * tx, 256>>>(ct, D, H, W, ts, tz, ty, tx, thr, s->t0);
    s->nl = 0;
    for (int l = 0; l < nlev; l++) {
        const int Dc = (D / cell) >> l, Hc = (H / cell) >> l, Wc = (W / cell) >> l; if (Dc < 1 || Hc < 1 || Wc < 1) break;
        const int mz = (Dc + 7) / 8, my = (Hc + 7) / 8, mx = (Wc + 7) / 8; const size_t n = (size_t)mz * my * mx;
        if (!sk_buf(&s->m[l], &s->cap[l], n)) return 1;
        sk_level_k<<<nblk(n, 128), 128>>>(s->t0, tz, ty, tx, l, margin, mz, my, mx, s->m[l]);
        s->D[l] = Dc; s->H[l] = Hc; s->W[l] = Wc; s->mz[l] = mz; s->my[l] = my; s->mx[l] = mx; s->nl = l + 1;
    }
    KCHECK();
    s->on = 1;
    return 0;
}
extern "C" int nn_skip_set(int lev, const uint8_t *hmask, int Dc, int Hc, int Wc) {   /* tests: level lev's tile mask from the host */
    skreg_t *s = &g_sk[skdev()]; if (lev < 0 || lev >= SK_LEV) return 1;
    const int mz = (Dc + 7) / 8, my = (Hc + 7) / 8, mx = (Wc + 7) / 8; const size_t n = (size_t)mz * my * mx;
    if (!sk_buf(&s->m[lev], &s->cap[lev], n)) return 1;
    cudaMemcpy(s->m[lev], hmask, n, cudaMemcpyHostToDevice);
    s->D[lev] = Dc; s->H[lev] = Hc; s->W[lev] = Wc; s->mz[lev] = mz; s->my[lev] = my; s->mx[lev] = mx; if (s->nl <= lev) s->nl = lev + 1;
    s->on = 1;
    return 0;
}
extern "C" void nn_skip_off(void) { skreg_t *s = &g_sk[skdev()]; s->on = 0; for (int l = 0; l < SK_LEV; l++) s->D[l] = 0; s->nl = 0; }
extern "C" void nn_skip_pause(int paused) { skreg_t *s = &g_sk[skdev()]; if (s->nl) s->on = !paused; }
extern "C" double nn_skip_share(int lev) {   /* the active share of level lev's tiles (-1: none) */
    skreg_t *s = &g_sk[skdev()]; if (lev < 0 || lev >= s->nl || !s->D[lev]) return -1;
    const size_t n = (size_t)s->mz[lev] * s->my[lev] * s->mx[lev]; uint8_t *h = (uint8_t *)malloc(n); if (!h) return -1;
    cudaMemcpy(h, s->m[lev], n, cudaMemcpyDeviceToHost); size_t a = 0; for (size_t i = 0; i < n; i++) a += h[i] != 0; free(h);
    return (double)a / n;
}
extern "C" void nn_skip_counts(long *masked, long *calls) { skreg_t *s = &g_sk[skdev()]; *masked = s->used; *calls = s->calls; s->used = s->calls = 0; }
/* the conv launches' lookup: the mask of the level whose grid the output is (MX forward, N 1, whole-tensor output), else none */
extern "C" skipm_t lp_skip_get(shape5 xs, split_t sp, int fwd) {
    skipm_t r = {nullptr, 0, 0};
    skreg_t *s = &g_sk[skdev()]; if (!s->on) return r;
    s->calls++;
    static int dbg = -1; if (dbg < 0) dbg = getenv("UFSM_SKIP_DEBUG") ? atoi(getenv("UFSM_SKIP_DEBUG")) : 0;   /* UFSM_SKIP_DEBUG=N: the first N dense lookups and why */
    if (!fwd || xs.n != 1 || sp.zw1 || sp.Di || sp.yD || sp.accum || sp.sr || sp.osr) {
        if (dbg > 0) { dbg--; fprintf(stderr, "skip: dense %dx%dx%dx%d: fwd %d n %d zw %d-%d zi0 %d Di %d yD %d accum %d sr %u osr %u\n", xs.c, xs.d, xs.h, xs.w, fwd, xs.n, sp.zw0, sp.zw1, sp.zi0, sp.Di, sp.yD, sp.accum, sp.sr, sp.osr); }
        return r;
    }
    for (int l = 0; l < s->nl; l++) if (s->D[l] == xs.d && s->H[l] == xs.h && s->W[l] == xs.w) { r.m = s->m[l]; r.my = s->my[l]; r.mx = s->mx[l]; s->used++; return r; }
    if (dbg > 0) { dbg--; fprintf(stderr, "skip: dense %dx%dx%dx%d: no level grid of these dims (level 0 %dx%dx%d)\n", xs.c, xs.d, xs.h, xs.w, s->D[0], s->H[0], s->W[0]); }
    return r;
}
