/* Empty-space skipping (--skip-empty 1, DESIGN 2026-11-13 / 11-14 "empty-space skipping"): per level a mask over tiles of 8 x 8 x 16 cells
   (z, y, x) of that level's grid. A masked 3^3 conv leaves every output tile whose mask tile is inactive at its bias (the CTA skips its
   main loop; the epilogue, the MX store and the GroupNorm sums run as usual): the network is the dense one with those conv outputs
   replaced by their bias. Not exact against the dense net (scratch2's input is never spatially constant: the radial channels; attention
   and gfilm couple the window), but a well-defined net whose backward is its exact gradient:
     forward       y = mask ? W * x + b : b                       (conv_fwd_f4_k, the output tile's mask)
     backward-data gx = W^T * (mask . gy)                         (conv_fwd_f4_k on gy: staged gy zero in inactive tiles; gx tiles with no
                                                                   active gy tile within one tile skipped, i.e. zero)
     weight grad   gw = sum over active positions of gy (x) x     (the fp4 weight gradient's gy records of inactive tiles zeroed)
     bias grad     gb = sum over all positions of gy              (unchanged: the records' bias sums are taken before the zeroing)
   A conv is masked in its backward only when its forward was (conv id = layer, sub, block: nn_core.cu's thread-local ids), so convs
   whose forward runs another kernel stay dense in both directions. Masks are per device (the host thread's current one) and apply to
   N 1 tensors whose grid is a registered level grid (z-ranges / z-windows located by depth and first plane). */
#include "nn_common.cuh"
#include "nn_lp.h"

#define SK_LEV 8
#define SK_NID 4096
typedef struct {
    int on, nl; int D[SK_LEV], H[SK_LEV], W[SK_LEV], mz[SK_LEV], my[SK_LEV], mx[SK_LEV];
    uint8_t *m[SK_LEV], *md[SK_LEV]; size_t cap[SK_LEV], capd[SK_LEV]; uint8_t *t0; size_t t0cap;
    uint8_t seen[SK_NID];                 /* conv ids masked in this forward */
    long used, calls, bwd, wg;            /* masked forward launches, lookups, masked backward-data launches, masked weight gradients */
} skreg_t;
static skreg_t g_sk[8];
static thread_local int g_sk_wg, g_sk_bwd_expect, g_sk_bwd_took;   /* inside a weight gradient; a masked conv's backward-data pending / taken */
static int skdev(void) { int d = 0; cudaGetDevice(&d); return d & 7; }
static int sk_cid(void) { if (g_layer < 0) return -1; return (((g_layer * 2 + (g_sub > 0 ? g_sub : 0)) & 63) | ((g_block & 63) << 6)) & (SK_NID - 1); }
static int sk_mtz(int D) { return (D + 7) / 8; }
static int sk_mty(int H) { return (H + 7) / 8; }
static int sk_mtx(int W) { return (W + 15) / 16; }

/* one block per level-0 tile (8 x 8 x 16 cells of cell^3 voxels): any voxel above thr. u8 (dt 0) or 16-bit (dt 1 bf16, 2 fp16) */
template <typename TV> __device__ __forceinline__ float sk_ld(const TV *p, size_t i) { return (float)p[i]; }
template <> __device__ __forceinline__ float sk_ld<__half>(const __half *p, size_t i) { return __half2float(p[i]); }
template <> __device__ __forceinline__ float sk_ld<__nv_bfloat16>(const __nv_bfloat16 *p, size_t i) { return __bfloat162float(p[i]); }
template <typename TV> __global__ void sk_tile_any_k(const TV *ct, int D, int H, int W, int cell, int tz, int ty, int tx, float thr, uint8_t *t0) {
    const int b = blockIdx.x, sx = 16 * cell, sy = 8 * cell, sz = 8 * cell;
    const int x0 = (b % tx) * sx, y0 = (b / tx % ty) * sy, z0 = (b / tx / ty) * sz;
    int any = 0;
    for (int i = threadIdx.x; i < sx * sy * sz && !any; i += blockDim.x) {
        const int x = x0 + i % sx, y = y0 + i / sx % sy, z = z0 + i / sx / sy;
        if (x < W && y < H && z < D && sk_ld<TV>(ct, ((size_t)z * H + y) * W + x) > thr) any = 1;
    }
    any = __syncthreads_or(any);
    if (threadIdx.x == 0) t0[b] = (uint8_t)any;
}
/* level l: tile (a, b, c) covers 2^l level-0 tiles a side; active when any level-0 tile within r level-l tiles (Chebyshev) is */
__global__ void sk_level_k(const uint8_t *t0, int tz, int ty, int tx, int l, int r, int mz, int my, int mx, uint8_t *m) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= mz * my * mx) return;
    const int c = i % mx, b = i / mx % my, a = i / mx / my, f = 1 << l;
    const int za = max((a - r) * f, 0), zb = min((a + r + 1) * f, tz), ya = max((b - r) * f, 0), yb = min((b + r + 1) * f, ty), xa = max((c - r) * f, 0), xb = min((c + r + 1) * f, tx);
    uint8_t on = 0;
    for (int z = za; z < zb && !on; z++) for (int y = ya; y < yb && !on; y++) for (int x = xa; x < xb && !on; x++) on = t0[((size_t)z * ty + y) * tx + x];
    m[i] = on;
}
__global__ void sk_dilate_k(const uint8_t *m, int mz, int my, int mx, uint8_t *md) {   /* one tile of dilation: backward-data's gx tiles */
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= mz * my * mx) return;
    const int c = i % mx, b = i / mx % my, a = i / mx / my; uint8_t on = 0;
    for (int z = max(a - 1, 0); z <= min(a + 1, mz - 1) && !on; z++) for (int y = max(b - 1, 0); y <= min(b + 1, my - 1) && !on; y++) for (int x = max(c - 1, 0); x <= min(c + 1, mx - 1) && !on; x++)
        on = m[((size_t)z * my + y) * mx + x];
    md[i] = on;
}
static uint8_t *sk_buf(uint8_t **p, size_t *cap, size_t n) {
    if (*cap < n) { if (*p) cudaFree(*p); *p = nullptr; *cap = 0; if (cudaMalloc((void **)p, n) != cudaSuccess) { cudaGetLastError(); return nullptr; } *cap = n; }
    return *p;
}
static int sk_level(skreg_t *s, int l, int Dc, int Hc, int Wc) {   /* allocate level l's masks (mask, dilated) and record its grid */
    const int mz = sk_mtz(Dc), my = sk_mty(Hc), mx = sk_mtx(Wc); const size_t n = (size_t)mz * my * mx;
    if (!sk_buf(&s->m[l], &s->cap[l], n) || !sk_buf(&s->md[l], &s->capd[l], n)) return 1;
    s->D[l] = Dc; s->H[l] = Hc; s->W[l] = Wc; s->mz[l] = mz; s->my[l] = my; s->mx[l] = mx; if (s->nl <= l) s->nl = l + 1;
    return 0;
}
static int sk_build(const void *ct, int dt, int D, int H, int W, int cell, int nlev, float thr, int margin) {
    skreg_t *s = &g_sk[skdev()]; s->on = 0; s->nl = 0; for (int l = 0; l < SK_LEV; l++) s->D[l] = 0;
    memset(s->seen, 0, sizeof s->seen);
    if (nlev > SK_LEV) nlev = SK_LEV;
    if (nlev < 1) return 0;
    const int tz = sk_mtz((D + cell - 1) / cell), ty = sk_mty((H + cell - 1) / cell), tx = sk_mtx((W + cell - 1) / cell);
    if (!sk_buf(&s->t0, &s->t0cap, (size_t)tz * ty * tx)) return 1;
    if (dt == 0) sk_tile_any_k<uint8_t><<<tz * ty * tx, 256>>>((const uint8_t *)ct, D, H, W, cell, tz, ty, tx, thr, s->t0);
    else if (dt == 2) sk_tile_any_k<__half><<<tz * ty * tx, 256>>>((const __half *)ct, D, H, W, cell, tz, ty, tx, thr, s->t0);
    else sk_tile_any_k<__nv_bfloat16><<<tz * ty * tx, 256>>>((const __nv_bfloat16 *)ct, D, H, W, cell, tz, ty, tx, thr, s->t0);
    for (int l = 0; l < nlev; l++) {
        const int Dc = (D / cell) >> l, Hc = (H / cell) >> l, Wc = (W / cell) >> l; if (Dc < 1 || Hc < 1 || Wc < 1) break;
        if (sk_level(s, l, Dc, Hc, Wc)) return 1;
        const size_t n = (size_t)s->mz[l] * s->my[l] * s->mx[l];
        sk_level_k<<<nblk(n, 128), 128>>>(s->t0, tz, ty, tx, l, margin, s->mz[l], s->my[l], s->mx[l], s->m[l]);
        sk_dilate_k<<<nblk(n, 128), 128>>>(s->m[l], s->mz[l], s->my[l], s->mx[l], s->md[l]);
    }
    KCHECK();
    s->on = 1;
    return 0;
}
extern "C" int nn_skip_build(const uint8_t *ct, int D, int H, int W, int cell, int nlev, int thr, int margin) { return sk_build(ct, 0, D, H, W, cell, nlev, (float)thr, margin); }
extern "C" int nn_skip_build16(const void *ct16, int D, int H, int W, int cell, int nlev, float thr, int margin) { return sk_build(ct16, g_h16 ? 2 : 1, D, H, W, cell, nlev, thr, margin); }
extern "C" int nn_skip_set(int lev, const uint8_t *hmask, int Dc, int Hc, int Wc) {   /* tests: level lev's tile mask (8 x 8 x 16 cells) from the host */
    skreg_t *s = &g_sk[skdev()]; if (lev < 0 || lev >= SK_LEV) return 1;
    if (sk_level(s, lev, Dc, Hc, Wc)) return 1;
    const size_t n = (size_t)s->mz[lev] * s->my[lev] * s->mx[lev];
    cudaMemcpy(s->m[lev], hmask, n, cudaMemcpyHostToDevice);
    sk_dilate_k<<<nblk(n, 128), 128>>>(s->m[lev], s->mz[lev], s->my[lev], s->mx[lev], s->md[lev]); KCHECK();
    if (!s->on) memset(s->seen, 0, sizeof s->seen);
    s->on = 1;
    return 0;
}
extern "C" void nn_skip_off(void) { skreg_t *s = &g_sk[skdev()]; s->on = 0; for (int l = 0; l < SK_LEV; l++) s->D[l] = 0; s->nl = 0; memset(s->seen, 0, sizeof s->seen); }
extern "C" void nn_skip_pause(int paused) { skreg_t *s = &g_sk[skdev()]; if (s->nl) s->on = !paused; if (!paused) memset(s->seen, 0, sizeof s->seen); }
extern "C" int nn_skip_on(void) { return g_sk[skdev()].on; }
extern "C" double nn_skip_share(int lev) {   /* the active share of level lev's tiles (-1: none) */
    skreg_t *s = &g_sk[skdev()]; if (lev < 0 || lev >= s->nl || !s->D[lev]) return -1;
    const size_t n = (size_t)s->mz[lev] * s->my[lev] * s->mx[lev]; uint8_t *h = (uint8_t *)malloc(n); if (!h) return -1;
    cudaMemcpy(h, s->m[lev], n, cudaMemcpyDeviceToHost); size_t a = 0; for (size_t i = 0; i < n; i++) a += h[i] != 0; free(h);
    return (double)a / n;
}
extern "C" void nn_skip_counts(long *masked, long *calls) { skreg_t *s = &g_sk[skdev()]; *masked = s->used; *calls = s->calls; s->used = s->calls = 0; }
extern "C" void nn_skip_counts_bwd(long *bwd, long *wg) { skreg_t *s = &g_sk[skdev()]; *bwd = s->bwd; *wg = s->wg; s->bwd = s->wg = 0; }
static int sk_find(const skreg_t *s, int D, int H, int W) { for (int l = 0; l < s->nl; l++) if (s->D[l] == D && s->H[l] == H && s->W[l] == W) return l; return -1; }
static int sk_dbg(void) { static int d = -1; if (d < 0) d = getenv("UFSM_SKIP_DEBUG") ? atoi(getenv("UFSM_SKIP_DEBUG")) : 0; return d; }
/* conv_fwd_tc: whether this conv will be masked (forward: a level grid; backward-data: its forward was masked). The sparse 2:4 kernels
   do not take the mask, so a masked conv bypasses them (both directions, consistently). Backward-data of a masked conv: arms the check
   that the masked kernel took it (nn_skip_bwd_check) */
extern "C" int nn_skip_claim(shape5 xs) {
    skreg_t *s = &g_sk[skdev()]; if (!s->on || xs.n != 1) return 0;
    const int cid = sk_cid(); if (cid < 0) return 0;
    if (g_pass == 1) { if (!s->seen[cid]) return 0; g_sk_bwd_expect = 1; g_sk_bwd_took = 0; return 1; }
    return 0;   /* (forward: claimed by lp_skip_get's level match; the sparse kernels are bypassed by nn_skip_fwd_level) */
}
extern "C" int nn_skip_fwd_level(shape5 xs, split_t sp) {   /* forward: this conv's output grid is a registered level (lp_skip_get's match) */
    skreg_t *s = &g_sk[skdev()]; if (!s->on || xs.n != 1 || sk_cid() < 0 || g_pass == 1) return 0;
    const int D = sp.yD ? sp.yD : sp.Di ? sp.Di : xs.d, zoff = sp.yD ? sp.yzo : sp.Di ? sp.zi0 : 0;
    if (sp.accum || zoff % 8 || (sp.zw1 && zoff)) return 0;
    return sk_find(s, D, xs.h, xs.w) >= 0;
}
extern "C" void nn_skip_bwd_check(void) {
    if (!g_sk_bwd_expect) return;
    g_sk_bwd_expect = 0;
    if (!g_sk_bwd_took) { fprintf(stderr, "skip-empty: the backward-data of masked conv %d (layer %d sub %d block %d) ran a kernel without the mask: its gradient would not be the masked net's\n", sk_cid(), g_layer, g_sub, g_block); abort(); }
}
extern "C" void nn_skip_wg_begin(void) { g_sk_wg = 1; }
extern "C" void nn_skip_wg_end(void) { g_sk_wg = 0; }
/* lp_f4fwd.cuh launch_f4: the conv's mask. Forward (g_pass 0): the output's level (records the conv id). Backward-data (g_pass 1, the
   conv's forward masked): m = the level's mask dilated by one tile (gx tiles to compute), mi = the mask (gy staged as zero elsewhere).
   A z-range output (planes of a depth-yD output / reading planes z + zi0 of a depth-Di input) is located by that depth and its first
   plane (a multiple of 8: whole tiles); a z-window (zw0 / zw1) has whole-tensor plane numbers */
extern "C" skipm_t lp_skip_get(shape5 xs, split_t sp, int fwd) {
    skipm_t r = {nullptr, nullptr, 0, 0, 0};
    skreg_t *s = &g_sk[skdev()]; if (!s->on) return r;
    s->calls++;
    const int cid = sk_cid(), bwd = g_pass == 1;
    const int D = sp.yD ? sp.yD : sp.Di ? sp.Di : xs.d, zoff = sp.yD ? sp.yzo : sp.Di ? sp.zi0 : 0;
    int dbg = sk_dbg();
    if (cid < 0 || xs.n != 1 || sp.accum || zoff % 8 || (sp.yD && sp.Di && sp.yzo != sp.zi0) || (sp.zw1 && zoff) || (bwd && !s->seen[cid]) || (!bwd && (sp.sr || sp.osr))) {
        if (dbg > 0 && cid >= 0 && (!bwd || s->seen[cid])) { static int left = -1; if (left < 0) left = dbg; if (left-- > 0) fprintf(stderr, "skip: dense %dx%dx%dx%d pass %d cid %d: n %d zw %d-%d zi0 %d Di %d yD %d yzo %d accum %d sr %u osr %u\n", xs.c, xs.d, xs.h, xs.w, g_pass, cid, xs.n, sp.zw0, sp.zw1, sp.zi0, sp.Di, sp.yD, sp.yzo, sp.accum, sp.sr, sp.osr); }
        return r;
    }
    const int l = sk_find(s, D, xs.h, xs.w);
    if (l < 0) {
        if (bwd) { fprintf(stderr, "skip-empty: backward-data of masked conv %d on a %dx%dx%d grid that is no level\n", cid, D, xs.h, xs.w); abort(); }
        return r;
    }
    r.my = s->my[l]; r.mx = s->mx[l]; r.zoff = zoff;
    if (bwd) { r.m = s->md[l]; r.mi = s->m[l]; s->bwd++; g_sk_bwd_took = 1; }
    else { r.m = s->m[l]; s->seen[cid] = 1; s->used++; }
    return r;
}
/* lp_f4wgrad.cuh: the weight gradient of a masked conv (its forward masked; gy's grid a level): the mask whose inactive tiles' gy records
   are zeroed before the kernels read them (m nullptr: dense). A masked conv's weight gradient must take the record path (abort otherwise) */
extern "C" skipm_t lp_skip_wgrad(shape5 ys) {
    skipm_t r = {nullptr, nullptr, 0, 0, 0};
    skreg_t *s = &g_sk[skdev()]; if (!s->on || !g_sk_wg) return r;
    const int cid = sk_cid(); if (cid < 0 || !s->seen[cid]) return r;
    const int l = sk_find(s, ys.d, ys.h, ys.w);
    if (l < 0 || ys.n != 1) { fprintf(stderr, "skip-empty: the weight gradient of masked conv %d on a %dx%dx%dx%d grid that is no level\n", cid, ys.n, ys.d, ys.h, ys.w); abort(); }
    r.m = s->m[l]; r.my = s->my[l]; r.mx = s->mx[l]; s->wg++;
    return r;
}
/* zero the gy records of inactive tiles: records [n][co][z-step zs0 + k][8-row tile][16-x tile] of rec bytes each (gy_pre4_k's layout:
   a record is 2 planes x 8 rows x 16 x, inside one mask tile: z-steps of 2 planes, mask tiles of 8 planes; 16-x tiles = mask x tiles) */
__global__ void sk_rec_zero_k(uint8_t *buf, int C, int nzs, int zs0, int ht, int wt, size_t rec, const uint8_t *m, int my, int mx) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, nr = (size_t)C * nzs * ht * wt;
    if (i >= nr) return;
    const int xt = (int)(i % wt), yt = (int)(i / wt % ht), zs = (int)(i / ((size_t)wt * ht) % nzs) + zs0;
    if (m[((size_t)(zs >> 2) * my + yt) * mx + xt]) return;   /* (2 planes a z-step, 8 planes a mask tile) */
    uint2 *p = (uint2 *)(buf + i * rec); for (size_t k = 0; k < rec / 8; k++) p[k] = make_uint2(0u, 0u);   /* (zero blocks: value 0 at any scale) */
}
extern "C" void lp_skip_rec_zero(uint8_t *buf, shape5 ys, int zs0, int zs1, size_t rec, skipm_t sk) {
    if (!sk.m) return;
    if (rec % 8 || ((uintptr_t)buf & 7)) { fprintf(stderr, "skip-empty: gy records of %zu bytes at %p (8-byte multiples expected)\n", rec, (void *)buf); abort(); }
    const int ht = (ys.h + 7) / 8, wt = (ys.w + 15) / 16; const size_t nr = (size_t)ys.c * (zs1 - zs0) * ht * wt;
    sk_rec_zero_k<<<nblk(nr, 256), 256>>>(buf, ys.c, zs1 - zs0, zs0, ht, wt, rec, sk.m, sk.my, sk.mx); KCHECK();
}
