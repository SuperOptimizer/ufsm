/* CUDA ops: loss section of the former nn.cu */
#include "nn_common.cuh"

float g_posw = 1.f;
extern "C" void nn_set_pos_weight(float w) { g_posw = w; }
int g_tol = 0;
extern "C" void nn_set_loss_tol(int r) { g_tol = r < 0 ? 0 : r > 2 ? 2 : r; }
extern "C" int nn_get_loss_tol(void) { return g_tol; }
int g_logits_h16 = 0;   /* the logits the losses read are fp16 (unet_logits_h16: lean training with the logit gradient beside them) */
extern "C" void nn_set_logits_h16(int on) { g_logits_h16 = on; }
int g_head_h16 = 0;
extern "C" void nn_set_head_out_h16(int on) { g_head_h16 = on; }
template <typename LT> __global__ void loss_stats_k(const LT *lg, const uint8_t *t, const uint8_t *m, const uint8_t *w, int C, size_t S, double *ds, float pw,
                                                   int tol, uint8_t *code, int D, int H, int W) {
    int nc = blockIdx.x, slab = blockIdx.y, n = nc / C;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0;   /* per-thread fp32 partials, fp64 block reduction */
    if (w[nc]) {
        const LT *l = lg + (size_t)nc * S; const uint8_t *tp = t + (size_t)nc * S, *mp = m + (size_t)n * S;
        uint8_t *cp = tol && nc % C == 0 ? code + (size_t)n * S : nullptr;
        size_t per = (S + gridDim.y - 1) / gridDim.y, lo = (size_t)slab * per, hi = lo + per < S ? lo + per : S;
        /* LS_V of the thread's voxels per round, their loads first (the one-voxel loop waited a memory latency per voxel), then
           each voxel in the thread's order: the same partial sums, bit for bit */
        constexpr int LS_V = 4;
        for (size_t i0 = lo + threadIdx.x; i0 < hi; i0 += (size_t)LS_V * blockDim.x) {
            unsigned mb[LS_V], tb[LS_V]; float xl[LS_V];
#pragma unroll
            for (int k = 0; k < LS_V; k++) {
                const size_t i = i0 + (size_t)k * blockDim.x;
                mb[k] = i < hi ? mp[i] : 0u; tb[k] = 0u; xl[k] = 0.f;
                if (mb[k]) { xl[k] = lgv(l, i); tb[k] = tp[i]; }
            }
#pragma unroll
            for (int k = 0; k < LS_V; k++) {
                const size_t i = i0 + (size_t)k * blockDim.x;
                if (!mb[k]) continue;
                const float mw = mb[k];   /* the mask is a weight (1 = plain; nn_packed_mask / nn_unwrap_targets2 write more) */
                float x = xl[k], p = tb[k] * (1.f / 255.f);
                float sg = 1.f / (1.f + __expf(-x));
                float spp = fmaxf(x, 0.f) + log1pf(__expf(-fabsf(x)));   /* softplus(x) */
                float xq = x, spq = spp;
                if (cp && tp[i] >= 128) {   /* offset-tolerant positive term (p >= 0.5): maximum along the sheet normal */
                    const int z = (int)(i / ((size_t)H * W)), y = (int)((i / W) % H), xx = (int)(i % W);
                    uint8_t c8; xq = tol_line_max(l, mp, z, y, xx, D, H, W, tol_normal(tp, z, y, xx, D, H, W), tol, &c8); cp[i] = c8;
                    spq = fmaxf(xq, 0.f) + log1pf(__expf(-fabsf(xq)));
                }
                float bce = pw * p * (spq - xq) + (1.f - p) * spp;          /* softplus(-x) = softplus(x) - x */
                a0 += mw; a1 += mw * bce; a2 += sg * p; a3 += sg; a4 += p;
            }
        }
    }
    __shared__ double r[5][256];
    r[0][threadIdx.x] = a0; r[1][threadIdx.x] = a1; r[2][threadIdx.x] = a2; r[3][threadIdx.x] = a3; r[4][threadIdx.x] = a4;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) for (int k = 0; k < 5; k++) r[k][threadIdx.x] += r[k][threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) for (int k = 0; k < 5; k++) atomicAdd(&ds[nc * 5 + k], r[k][0]);
}
__global__ void loss_d2f_k(const double *d, float *f, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) f[i] = (float)d[i]; }
__global__ void loss_fin_k(const float *st, const uint8_t *w, int N, int C, float *fin) {
    if (threadIdx.x || blockIdx.x) return;
    int active = 0, cnt[16] = {0};
    for (int c = 0; c < 2 * C + 2; c++) fin[c] = 0.f;
    for (int nc = 0; nc < N * C; nc++) {
        if (!w[nc] || st[nc * 5] < 1.f) continue;
        active++;
        int c = nc % C;
        fin[c] += st[nc * 5 + 1] / st[nc * 5];
        fin[C + c] += 1.f - (2.f * st[nc * 5 + 2] + 1.f) / (st[nc * 5 + 3] + st[nc * 5 + 4] + 1.f);
        cnt[c]++;
    }
    for (int c = 0; c < C; c++) if (cnt[c]) { fin[c] /= cnt[c]; fin[C + c] /= cnt[c]; }
    fin[2 * C] = (float)active;
    fin[2 * C + 1] = active ? 1.f / active : 0.f;
}
int g_loss_g16 = 0;
static float g_recall = 0.f;   /* --recall weight (recto channel): 1 - (sum s p + 1) / (sum p + 1), a skeleton-recall term for thin labels */
extern "C" void nn_set_recall(float w) { g_recall = w < 0.f ? 0.f : w; }
extern "C" void nn_set_loss_grad_h16(int on) { g_loss_g16 = on; }
extern "C" int nn_get_loss_grad_h16(void) { return g_loss_g16; }
extern "C" size_t nn_loss_scratch(shape5 s) { return ((size_t)5 * s.n * s.c + 2 * s.c + 2) * sizeof(float); }
extern "C" void nn_loss_async(const float *logits, const uint8_t *t, const uint8_t *m, const uint8_t *w, shape5 s, float dice_w, float *gl, float *scratch) {
    nn_loss_async_tol(logits, t, m, w, s, dice_w, gl, scratch, nullptr);
}
template <typename GT> __global__ void gscale_k(GT *g, size_t n, float a) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) g[i] = f2h<GT>(a * (float)g[i]);
}
/* g *= a over n logit-gradient elements in the losses' storage (fp32, or 16-bit when nn_set_loss_grad_h16) */
extern "C" void nn_loss_grad_scale(void *g, size_t n, float a) {
    if (!g_loss_g16) gscale_k<float><<<nblk(n, 256), 256>>>((float *)g, n, a);
    else if (g_h16) gscale_k<f16><<<nblk(n, 256), 256>>>((f16 *)g, n, a);
    else gscale_k<bf16><<<nblk(n, 256), 256>>>((bf16 *)g, n, a);
    KCHECK();
}
extern "C" void nn_loss_async_tol(const float *logits, const uint8_t *t, const uint8_t *m, const uint8_t *w, shape5 s, float dice_w, float *gl, float *scratch, uint8_t *code) {
    size_t S = shape_spatial(s);
    int NC = s.n * s.c;
    float *fin = scratch + (size_t)5 * NC;
    double *ds = gn_dsums((size_t)5 * NC);
    const int tol = code ? g_tol : 0;
    if (tol) cudaMemsetAsync(code, 255, (size_t)s.n * S);
    else code = nullptr;
    cudaMemsetAsync(ds, 0, (size_t)5 * NC * sizeof(double));
    /* slabs per (n, c): at least KSLAB and ~512 blocks in all (one recto channel with KSLAB slabs left most SMs idle) */
    const int slabs = NC >= 16 ? KSLAB : 512 / NC;
    if (g_logits_h16) loss_stats_k<f16><<<dim3(NC, slabs), 256>>>((const f16 *)logits, t, m, w, s.c, S, ds, g_posw, tol, code, s.d, s.h, s.w);
    else loss_stats_k<float><<<dim3(NC, slabs), 256>>>(logits, t, m, w, s.c, S, ds, g_posw, tol, code, s.d, s.h, s.w);
    zs_reduce(ds, 5 * NC);   /* spatial split: statistics of the whole window (the halo planes are masked out by the caller) */
    loss_d2f_k<<<nblk(5 * NC, 128), 128>>>(ds, scratch, 5 * NC);
    loss_fin_k<<<1, 32>>>(scratch, w, s.n, s.c, fin);
    if (gl) {
        size_t n = shape_numel(s);
        if (g_logits_h16) {
            const f16 *l16 = (const f16 *)logits;
            if (!g_loss_g16) loss_grad_k<float, f16><<<lg_grid(S, s.n * s.c), 256>>>(l16, t, m, w, s.n, s.c, S, scratch, dice_w, fin, gl, g_posw, 1.f, tol, code, s.d, s.h, s.w, g_recall);
            else if (g_h16) loss_grad_k<f16, f16><<<lg_grid(S, s.n * s.c), 256>>>(l16, t, m, w, s.n, s.c, S, scratch, dice_w, fin, (f16 *)gl, g_posw, g_gscale, tol, code, s.d, s.h, s.w, g_recall);
            else loss_grad_k<bf16, f16><<<lg_grid(S, s.n * s.c), 256>>>(l16, t, m, w, s.n, s.c, S, scratch, dice_w, fin, (bf16 *)gl, g_posw, g_gscale, tol, code, s.d, s.h, s.w, g_recall);
        } else
        if (!g_loss_g16) loss_grad_k<float><<<lg_grid(S, s.n * s.c), 256>>>(logits, t, m, w, s.n, s.c, S, scratch, dice_w, fin, gl, g_posw, 1.f, tol, code, s.d, s.h, s.w, g_recall);
        else if (g_h16) loss_grad_k<f16><<<lg_grid(S, s.n * s.c), 256>>>(logits, t, m, w, s.n, s.c, S, scratch, dice_w, fin, (f16 *)gl, g_posw, g_gscale, tol, code, s.d, s.h, s.w, g_recall);
        else loss_grad_k<bf16><<<lg_grid(S, s.n * s.c), 256>>>(logits, t, m, w, s.n, s.c, S, scratch, dice_w, fin, (bf16 *)gl, g_posw, g_gscale, tol, code, s.d, s.h, s.w, g_recall);
    }
    KCHECK();
}
/* --half-res pack: the losses of the full-resolution window read straight from the network's packed logits (lg [n][8 C][d h w],
   channel 8 c + k = sub-voxel k = (z & 1) 4 + (y & 1) 2 + (x & 1) of output c), the gradient written in the same layout (in place
   over the logits allowed): nn_half_output's unpacked fp32 logits, nn_loss_async_tol and nn_half_grad (mode 2) without the
   full-resolution logits and gradient. Same per-element arithmetic and summation order (bit for bit). s: the full-resolution shape */
__device__ __forceinline__ size_t pk_lg_index(size_t i, int n, int c, int C, int H, int W, size_t hv) {   /* full-res voxel i of (n, c) -> packed element */
    const int z = (int)(i / ((size_t)H * W)), y = (int)((i / W) % H), x = (int)(i % W);
    const int k = (z & 1) * 4 + (y & 1) * 2 + (x & 1);
    return ((size_t)n * 8 * C + 8 * (size_t)c + k) * hv + ((size_t)(z >> 1) * (H >> 1) + (y >> 1)) * (W >> 1) + (x >> 1);
}
/* the target byte and mask weight of voxel i: t / m, or (t == nullptr) m holds the packed codes (sample.h SAMPLE_TM_*: bit c target c
   = 255, bit 2 mask), the values nn_unpack_tm would write */
__device__ __forceinline__ void pk_tm(const uint8_t *t, const uint8_t *m, size_t nc, int c, int n, size_t S, size_t i, unsigned *tv, unsigned *mv) {
    if (t) { *tv = t[nc * S + i]; *mv = m[(size_t)n * S + i]; return; }
    const unsigned code = m[(size_t)n * S + i];
    *tv = (code >> c & 1u) * 255u; *mv = code >> 2 & 1u;
}
template <typename LT> __global__ void loss_stats_pk_k(const LT *lg, const uint8_t *t, const uint8_t *m, const uint8_t *w, int C, size_t S, double *ds, float pw, int H, int W) {
    int nc = blockIdx.x, slab = blockIdx.y, n = nc / C;
    const size_t hv = S / 8;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0;
    if (w[nc]) {
        size_t per = (S + gridDim.y - 1) / gridDim.y, lo = (size_t)slab * per, hi = lo + per < S ? lo + per : S;
        for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
            unsigned tv, mv; pk_tm(t, m, nc, nc % C, n, S, i, &tv, &mv);
            if (!mv) continue;
            const float mw = mv;
            float x = lgv(lg, pk_lg_index(i, n, nc % C, C, H, W, hv)), p = tv * (1.f / 255.f);
            float sg = 1.f / (1.f + __expf(-x));
            float spp = fmaxf(x, 0.f) + log1pf(__expf(-fabsf(x)));
            float xq = x, spq = spp;
            float bce = pw * p * (spq - xq) + (1.f - p) * spp;
            a0 += mw; a1 += mw * bce; a2 += sg * p; a3 += sg; a4 += p;
        }
    }
    __shared__ double r[5][256];
    r[0][threadIdx.x] = a0; r[1][threadIdx.x] = a1; r[2][threadIdx.x] = a2; r[3][threadIdx.x] = a3; r[4][threadIdx.x] = a4;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) for (int k = 0; k < 5; k++) r[k][threadIdx.x] += r[k][threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) for (int k = 0; k < 5; k++) atomicAdd(&ds[nc * 5 + k], r[k][0]);
}
template <typename GT, typename LT> __global__ void loss_grad_pk_k(const LT *lg, const uint8_t *t, const uint8_t *m, const uint8_t *w, int N, int C, size_t S, const float *st,
                                                                  float dice_w, const float *fin, GT *gl, float pw, float gscale, int H, int W, float rec_w) {
    const size_t j = blockIdx.x * (size_t)blockDim.x + threadIdx.x, hv = S / 8;   /* one thread per packed logit element */
    if (j >= (size_t)N * C * S) return;
    const size_t ch = j / hv, hvi = j % hv;
    const int n = (int)(ch / (8 * (size_t)C)), c8 = (int)(ch % (8 * (size_t)C)), c = c8 >> 3, k = c8 & 7, nc = n * C + c;
    const int h = H >> 1, w2 = W >> 1;
    const int z = 2 * (int)(hvi / ((size_t)h * w2)) + (k >> 2), y = 2 * (int)((hvi / w2) % h) + (k >> 1 & 1), x0 = 2 * (int)(hvi % w2) + (k & 1);
    const size_t i = ((size_t)z * H + y) * W + x0;   /* the full-resolution voxel */
    unsigned tv, mv; pk_tm(t, m, nc, c, n, S, i, &tv, &mv);
    if (!w[nc] || !mv) { gl[j] = f2h<GT>(0.f); return; }
    float nm = st[nc * 5], Ssp = st[nc * 5 + 2], Ss = st[nc * 5 + 3], Sp = st[nc * 5 + 4];
    float x = lgv(lg, j), p = tv * (1.f / 255.f), s = 1.f / (1.f + expf(-x));
    float den = Ss + Sp + 1.f;
    float pq = p;
    const float mw = mv;
    float g = mw * ((1.f - p) * s - pw * pq * (1.f - s)) / fmaxf(nm, 1.f);
    float ddice_ds = -(2.f * p * den - (2.f * Ssp + 1.f)) / (den * den);
    g += dice_w * ddice_ds * s * (1.f - s);
    if (rec_w > 0.f && nc % C == 0) g -= rec_w * p / (Sp + 1.f) * s * (1.f - s);
    gl[j] = f2h<GT>(g * fin[2 * C + 1] * gscale);
}
extern "C" void nn_loss_pack(const void *logits, const uint8_t *t, const uint8_t *m, const uint8_t *w, shape5 s, float dice_w, void *gl, float *scratch) {
    if ((s.d | s.h | s.w) & 1) { fprintf(stderr, "nn_loss_pack: even window sides\n"); abort(); }
    const size_t S = shape_spatial(s);
    const int NC = s.n * s.c;
    float *fin = scratch + (size_t)5 * NC;
    double *ds = gn_dsums((size_t)5 * NC);
    cudaMemsetAsync(ds, 0, (size_t)5 * NC * sizeof(double));
    const int slabs = NC >= 16 ? KSLAB : 512 / NC;
    if (g_logits_h16) loss_stats_pk_k<f16><<<dim3(NC, slabs), 256>>>((const f16 *)logits, t, m, w, s.c, S, ds, g_posw, s.h, s.w);
    else loss_stats_pk_k<float><<<dim3(NC, slabs), 256>>>((const float *)logits, t, m, w, s.c, S, ds, g_posw, s.h, s.w);
    zs_reduce(ds, 5 * NC);
    loss_d2f_k<<<nblk(5 * NC, 128), 128>>>(ds, scratch, 5 * NC);
    loss_fin_k<<<1, 32>>>(scratch, w, s.n, s.c, fin);
    if (gl) {
        const size_t n = shape_numel(s);
#define LGP(GT, LT, gs) loss_grad_pk_k<GT, LT><<<nblk(n, 256), 256>>>((const LT *)logits, t, m, w, s.n, s.c, S, scratch, dice_w, fin, (GT *)gl, g_posw, gs, s.h, s.w, g_recall)
        if (g_logits_h16) { if (!g_loss_g16) LGP(float, f16, 1.f); else if (g_h16) LGP(f16, f16, g_gscale); else LGP(bf16, f16, g_gscale); }
        else { if (!g_loss_g16) LGP(float, float, 1.f); else if (g_h16) LGP(f16, float, g_gscale); else LGP(bf16, float, g_gscale); }
#undef LGP
    }
    KCHECK();
}
/* the loss in parts, for the trainer's head slabs (train.c): nn_loss_stats_acc adds a slab's statistics (logits and targets of that
   slab, s its shape) to ds (5 doubles per (n, c), zeroed by the caller before the first slab); nn_loss_stats_fin turns them into the
   scratch nn_loss_async_tol leaves (s: the whole window's shape, for N and C); nn_loss_grad_only writes a slab's logit gradient from
   that scratch. No tolerance band. Same per-element arithmetic as nn_loss_async_tol (fp64 sums in another order) */
extern "C" void nn_loss_stats_acc(const void *logits, const uint8_t *t, const uint8_t *m, const uint8_t *w, shape5 s, double *ds) {
    const size_t S = shape_spatial(s); const int NC = s.n * s.c, slabs = NC >= 16 ? KSLAB : 512 / NC;
    if (g_logits_h16) loss_stats_k<f16><<<dim3(NC, slabs), 256>>>((const f16 *)logits, t, m, w, s.c, S, ds, g_posw, 0, nullptr, s.d, s.h, s.w);
    else loss_stats_k<float><<<dim3(NC, slabs), 256>>>((const float *)logits, t, m, w, s.c, S, ds, g_posw, 0, nullptr, s.d, s.h, s.w);
    KCHECK();
}
extern "C" void nn_loss_stats_fin(double *ds, const uint8_t *w, shape5 s, float *scratch) {
    const int NC = s.n * s.c;
    zs_reduce(ds, 5 * NC);
    loss_d2f_k<<<nblk(5 * NC, 128), 128>>>(ds, scratch, 5 * NC);
    loss_fin_k<<<1, 32>>>(scratch, w, s.n, s.c, scratch + (size_t)5 * NC);
    KCHECK();
}
extern "C" void nn_loss_grad_only(const void *logits, const uint8_t *t, const uint8_t *m, const uint8_t *w, shape5 s, float dice_w, void *gl, const float *scratch) {
    const size_t S = shape_spatial(s), n = shape_numel(s);
    const float *fin = scratch + (size_t)5 * s.n * s.c;
    if (g_logits_h16) {
        const f16 *l16 = (const f16 *)logits;
        if (!g_loss_g16) loss_grad_k<float, f16><<<lg_grid(S, s.n * s.c), 256>>>(l16, t, m, w, s.n, s.c, S, scratch, dice_w, fin, (float *)gl, g_posw, 1.f, 0, nullptr, s.d, s.h, s.w, g_recall);
        else if (g_h16) loss_grad_k<f16, f16><<<lg_grid(S, s.n * s.c), 256>>>(l16, t, m, w, s.n, s.c, S, scratch, dice_w, fin, (f16 *)gl, g_posw, g_gscale, 0, nullptr, s.d, s.h, s.w, g_recall);
        else loss_grad_k<bf16, f16><<<lg_grid(S, s.n * s.c), 256>>>(l16, t, m, w, s.n, s.c, S, scratch, dice_w, fin, (bf16 *)gl, g_posw, g_gscale, 0, nullptr, s.d, s.h, s.w, g_recall);
    } else if (!g_loss_g16) loss_grad_k<float><<<lg_grid(S, s.n * s.c), 256>>>((const float *)logits, t, m, w, s.n, s.c, S, scratch, dice_w, fin, (float *)gl, g_posw, 1.f, 0, nullptr, s.d, s.h, s.w, g_recall);
    else if (g_h16) loss_grad_k<f16><<<lg_grid(S, s.n * s.c), 256>>>((const float *)logits, t, m, w, s.n, s.c, S, scratch, dice_w, fin, (f16 *)gl, g_posw, g_gscale, 0, nullptr, s.d, s.h, s.w, g_recall);
    else loss_grad_k<bf16><<<lg_grid(S, s.n * s.c), 256>>>((const float *)logits, t, m, w, s.n, s.c, S, scratch, dice_w, fin, (bf16 *)gl, g_posw, g_gscale, 0, nullptr, s.d, s.h, s.w, g_recall);
    KCHECK();
}
/* ---- count loss (train --count-lambda): turns along sheet-normal segments ----
   K segments per window, each from a random voxel whose ground-truth phase gradient (over +-s voxels) is known, L unit steps along
   that gradient (nearest voxels); a segment counts when every step's phase is known and it crosses at least half a turn. True turns
   Tg = the phase unwrapped along it; predicted Tp = sum of rho over its voxels, rho = exp(logit) / dscale (the density code x / (1 + x),
   x = exp(logit): train --density-code 1). Loss |log((Tp + e) / (Tg + e))| (e 0.05 turn), the mean over valid segments; its
   gradient sign / (Tp + e) rho_k per voxel accumulates in acc (fp32) and is added, times lambda / valid, into the density channel
   of the logit gradient. out2 (device): mean loss, valid segments, mean Tp / Tg (the count ratio). */
__device__ __forceinline__ int cnt_pdiff(int a, int b) { int d = a - b; d -= 252 * (d > 126); d += 252 * (d <= -126); return d; }
__device__ __forceinline__ uint64_t cnt_mix(uint64_t x) { x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 27; x *= 0x94d049bb133111ebull; return x ^ (x >> 31); }
template <typename LT> __global__ void count_seg_k(const LT *dl, const uint8_t *ph, int D, int H, int W, int s, int K, int L, float dscale, uint64_t seed,
                                                   float *acc, float *seg, unsigned *nvalid) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x; if (k >= K) return;
    seg[3 * k] = -1.f;
    const size_t HW = (size_t)H * W, n = (size_t)D * HW;
    uint64_t h = cnt_mix(seed * 0x9e3779b97f4a7c15ull + (uint64_t)k * 0xc2b2ae3d27d4eb4full);
    for (int tries = 0; tries < 16; tries++) {
        h = cnt_mix(h + 0x9e3779b97f4a7c15ull);
        const size_t i0 = (size_t)(h % n); const int z = (int)(i0 / HW), y = (int)(i0 / W % H), x = (int)(i0 % W);
        if (ph[i0] >= 252 || z < s || z >= D - s || y < s || y >= H - s || x < s || x >= W - s) continue;
        const size_t sz = (size_t)s * HW, sy = (size_t)s * W;
        const int a[6] = {ph[i0 + sz], ph[i0 - sz], ph[i0 + sy], ph[i0 - sy], ph[i0 + s], ph[i0 - s]};
        int ok = 1; for (int q = 0; q < 6; q++) ok &= a[q] < 252; if (!ok) continue;
        float g[3] = {(float)cnt_pdiff(a[0], a[1]), (float)cnt_pdiff(a[2], a[3]), (float)cnt_pdiff(a[4], a[5])};
        const float gn = sqrtf(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]); if (gn < 1e-3f) continue;
        for (int q = 0; q < 3; q++) g[q] /= gn;
        float tg = 0.f, tp = 0.f; int prev = ph[i0], valid = 1;
        for (int j = 0; j < L && valid; j++) {
            const int zz = (int)lrintf(z + j * g[0]), yy = (int)lrintf(y + j * g[1]), xx = (int)lrintf(x + j * g[2]);
            if (zz < 0 || zz >= D || yy < 0 || yy >= H || xx < 0 || xx >= W) { valid = 0; break; }
            const size_t i = (size_t)zz * HW + (size_t)yy * W + xx; const int v = ph[i];
            if (v >= 252) { valid = 0; break; }
            if (j) tg += cnt_pdiff(v, prev) * (1.f / 252.f);
            prev = v; if (j < L - 1) tp += __expf(fminf(lgv(dl, i), 20.f)) / dscale;   /* L - 1 unit steps: rho of each step's start */
        }
        if (!valid || tg < .5f) continue;
        const float e = .05f, r = logf((tp + e) / (tg + e)), coef = (r > 0 ? 1.f : -1.f) / (tp + e);
        seg[3 * k] = fabsf(r); seg[3 * k + 1] = tp; seg[3 * k + 2] = tg;
        atomicAdd(nvalid, 1u);
        for (int j = 0; j < L - 1; j++) {
            const int zz = (int)lrintf(z + j * g[0]), yy = (int)lrintf(y + j * g[1]), xx = (int)lrintf(x + j * g[2]);
            const size_t i = (size_t)zz * HW + (size_t)yy * W + xx;
            atomicAdd(&acc[i], coef * __expf(fminf(lgv(dl, i), 20.f)) / dscale);
        }
        return;
    }
}
__global__ void count_sum_k(const float *seg, int K, const unsigned *nvalid, float *out2) {
    if (threadIdx.x || blockIdx.x) return;
    double l = 0, tp = 0, tg = 0; for (int k = 0; k < K; k++) if (seg[3 * k] >= 0) { l += seg[3 * k]; tp += seg[3 * k + 1]; tg += seg[3 * k + 2]; }
    const unsigned nv = *nvalid; out2[0] = nv ? (float)(l / nv) : 0.f; out2[1] = (float)nv; out2[2] = tg > 0 ? (float)(tp / tg) : 0.f;
}
template <typename GT> __global__ void count_apply_k(GT *gl, float *acc, size_t n, const unsigned *nvalid, float scale) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= n) return;
    const float a = acc[i]; if (a == 0.f) return;
    acc[i] = 0.f;
    const unsigned nv = *nvalid; if (!nv) return;
    gl[i] = f2h<GT>((float)gl[i] + scale / (float)nv * a);
}
extern "C" void nn_count_loss(const void *dens_logits, const uint8_t *phase, int d, int h, int w, int s, int K, int L, float dscale, uint64_t seed,
                              float lambda, void *dens_grad, float *acc, float *seg, unsigned *nvalid, float *out2) {
    const size_t n = (size_t)d * h * w;
    cudaMemsetAsync(nvalid, 0, sizeof *nvalid);
    if (g_logits_h16) count_seg_k<f16><<<nblk(K, 128), 128>>>((const f16 *)dens_logits, phase, d, h, w, s, K, L, dscale, seed, acc, seg, nvalid);
    else count_seg_k<float><<<nblk(K, 128), 128>>>((const float *)dens_logits, phase, d, h, w, s, K, L, dscale, seed, acc, seg, nvalid);
    count_sum_k<<<1, 32>>>(seg, K, nvalid, out2);
    if (dens_grad) {
        if (!g_loss_g16) count_apply_k<float><<<nblk(n, 256), 256>>>((float *)dens_grad, acc, n, nvalid, lambda);
        else if (g_h16) count_apply_k<f16><<<nblk(n, 256), 256>>>((f16 *)dens_grad, acc, n, nvalid, lambda * g_gscale);
        else count_apply_k<bf16><<<nblk(n, 256), 256>>>((bf16 *)dens_grad, acc, n, nvalid, lambda * g_gscale);
    } else cudaMemsetAsync(acc, 0, n * sizeof *acc);   /* validation: no gradient */
    KCHECK();
}
extern "C" void nn_loss_fetch(const float *scratch, shape5 s, float *out) {
    CK(cudaMemcpy(out, scratch + (size_t)5 * s.n * s.c, (size_t)(2 * s.c + 1) * sizeof(float), cudaMemcpyDeviceToHost));
}
extern "C" void nn_loss(const float *logits, const uint8_t *t, const uint8_t *m, const uint8_t *w, shape5 s, float dice_w,
                        float *gl, float *out, float *scratch) {
    nn_loss_async(logits, t, m, w, s, dice_w, gl, scratch);
    nn_loss_fetch(scratch, s, out);
}
template <typename LT> __global__ void sheet_gather_k(const LT *lg,shape5 s,const float *xyz,size_t np,int z0,int lo,int hi,double *out) {
    size_t p=blockIdx.x*(size_t)blockDim.x+threadIdx.x; if (p>=np) return;
    int iz=(int)floorf(xyz[3*p]),iy=(int)floorf(xyz[3*p+1]),ix=(int)floorf(xyz[3*p+2]);
    double fz=xyz[3*p]-iz,fy=xyz[3*p+1]-iy,fx=xyz[3*p+2]-ix;
    double a=0,b=0; size_t S=(size_t)s.d*s.h*s.w;
    for (int dz=0;dz<2;dz++) for (int dy=0;dy<2;dy++) for (int dx=0;dx<2;dx++) {
        int gz=iz+dz,z=gz-z0,y=iy+dy,x=ix+dx;
        if (gz<lo || gz>=hi || z<0 || z>=s.d || y<0 || y>=s.h || x<0 || x>=s.w) continue;
        double w=(dz?fz:1-fz)*(dy?fy:1-fy)*(dx?fx:1-fx);
        if (w==0) continue; size_t k=((size_t)z*s.h+y)*s.w+x;
        a+=w*lgv(lg,k); b+=w*lgv(lg,S+k);
    }
    out[2*p]=a; out[2*p+1]=b;
}
extern "C" void nn_sheet_gather(const float *lg,shape5 s,const float *coords,size_t np,int z0,int lo,int hi,double *values) {
    if (!np) return;
    float *xyz; double *v; CK(cudaMalloc(&xyz,3*np*sizeof(float))); CK(cudaMalloc(&v,2*np*sizeof(double)));
    CK(cudaMemcpy(xyz,coords,3*np*sizeof(float),cudaMemcpyHostToDevice));
    if (g_logits_h16) sheet_gather_k<f16><<<nblk(np,128),128>>>((const f16 *)lg,s,xyz,np,z0,lo,hi,v);
    else sheet_gather_k<float><<<nblk(np,128),128>>>(lg,s,xyz,np,z0,lo,hi,v);
    zs_reduce(v,(int)(2*np));
    CK(cudaMemcpy(values,v,2*np*sizeof(double),cudaMemcpyDeviceToHost));
    CK(cudaFree(xyz)); CK(cudaFree(v)); KCHECK();
}
extern "C" void nn_sheet_scatter(float *gl,const uint64_t *indices,const float *values,size_t n,int h16) {
    if (!n) return; uint64_t *ix; float *v;
    CK(cudaMalloc(&ix,n*sizeof(uint64_t))); CK(cudaMalloc(&v,n*sizeof(float)));
    CK(cudaMemcpy(ix,indices,n*sizeof(uint64_t),cudaMemcpyHostToDevice)); CK(cudaMemcpy(v,values,n*sizeof(float),cudaMemcpyHostToDevice));
    if (!h16) sheet_scatter_k<float><<<nblk(n,256),256>>>(gl,ix,v,n,1.f);
    else if (g_h16) sheet_scatter_k<f16><<<nblk(n,256),256>>>((f16 *)gl,ix,v,n,g_gscale);
    else sheet_scatter_k<bf16><<<nblk(n,256),256>>>((bf16 *)gl,ix,v,n,g_gscale);
    CK(cudaFree(ix)); CK(cudaFree(v)); KCHECK();
}
extern "C" void nn_sheet_input(void *input,int W,const float *rows,float center,float scale,int h16) {
    float *r; CK(cudaMalloc(&r,4*(size_t)W*sizeof(float))); CK(cudaMemcpy(r,rows,4*(size_t)W*sizeof(float),cudaMemcpyHostToDevice));
    size_t n=(size_t)W*W*W;
    if (!h16) sheet_input_k<float><<<nblk(n,256),256>>>((float *)input,W,r,center,scale);
    else if (g_h16) sheet_input_k<f16><<<nblk(n,256),256>>>((f16 *)input,W,r,center,scale);
    else sheet_input_k<bf16><<<nblk(n,256),256>>>((bf16 *)input,W,r,center,scale);
    CK(cudaFree(r)); KCHECK();
}
__global__ void sheet_gate_k(float *r,const float *surface,const uint8_t *ct,int W,const float *rows) {
    size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
    if (i>=(size_t)W*W*W) return;
    if (!ct[i] || surface[i]<-2.19722458f) r[i]=nanf("");
    else {
        int z=i/((size_t)W*W),y=(i/W)%W,x=i%W;
        const float *a=rows+4*z;
        r[i]+=hypotf(y+a[0],x+a[1])*a[2]+a[3];
    }
}
extern "C" void nn_sheet_gate(float *r,const float *surface,const uint8_t *ct,int W,const float *rows) {
    float *a; CK(cudaMalloc(&a,4*(size_t)W*sizeof(float))); CK(cudaMemcpy(a,rows,4*(size_t)W*sizeof(float),cudaMemcpyHostToDevice));
    sheet_gate_k<<<nblk((size_t)W*W*W,256),256>>>(r,surface,ct,W,a);
    CK(cudaFree(a)); KCHECK();
}
extern "C" void nn_peer_copy(void *dst, int dst_dev, const void *src, int src_dev, size_t bytes) {
    static char enabled[8][8];
    if (!enabled[dst_dev & 7][src_dev & 7]) {
        int can = 0; cudaDeviceCanAccessPeer(&can, dst_dev, src_dev);
        if (can) { int cur; cudaGetDevice(&cur); cudaSetDevice(dst_dev); cudaDeviceEnablePeerAccess(src_dev, 0); cudaGetLastError(); cudaSetDevice(cur); }
        enabled[dst_dev & 7][src_dev & 7] = 1;
    }
    CK(cudaMemcpyPeer(dst, dst_dev, src, src_dev, bytes));
}
int zsegs(const void *p, shape5 s, int esz, zseg_t *sg) {
    const size_t HW = (size_t)s.h * s.w, S = HW * s.d;
    if (esz) { sg[0] = {0, (size_t)s.n * s.c, HW * esz, S * esz}; return 1; }
    const int dt = nn_storage(p), bw = s.c <= 8 ? 8 : s.c <= 16 ? 16 : 32, nb = (s.c + bw - 1) / bw, rb = dt == 4 ? bw / 2 : bw;   /* (mx_bw) */
    if (dt != 4 && dt != 8) { fprintf(stderr, "split: tensor %p has no MX registration\n", p); abort(); }
    sg[0] = {0, (size_t)s.n * nb, HW * rb, S * rb};
    sg[1] = {(size_t)s.n * nb * S * rb, (size_t)s.n * nb, HW, S};
    return 2;
}
size_t zsegs_plane_bytes(const zseg_t *sg, int ns) { size_t b = 0; for (int i = 0; i < ns; i++) b += sg[i].rows * sg[i].pb; return b; }
void zsegs_copy(void *t, const zseg_t *sg, int ns, int z0, int nz, void *c, int dir) {
    char *cb = (char *)c;
    for (int i = 0; i < ns; i++) {
        char *tp = (char *)t + sg[i].base + (size_t)z0 * sg[i].pb;
        const size_t w = (size_t)nz * sg[i].pb;
        if (dir) CK(cudaMemcpy2DAsync(tp, sg[i].pitch, cb, w, w, sg[i].rows, cudaMemcpyDeviceToDevice, 0));
        else CK(cudaMemcpy2DAsync(cb, w, tp, sg[i].pitch, w, sg[i].rows, cudaMemcpyDeviceToDevice, 0));
        cb += w * sg[i].rows;
    }
}
void zsegs_zero(void *t, const zseg_t *sg, int ns, int z0, int nz) {
    if (nz <= 0) return;
    for (int i = 0; i < ns; i++) CK(cudaMemset2DAsync((char *)t + sg[i].base + (size_t)z0 * sg[i].pb, sg[i].pitch, 0, (size_t)nz * sg[i].pb, sg[i].rows, 0));
}
extern "C" void nn_split_zero(const void *t, shape5 s, int esz, int lo, int hi) {
    zseg_t sg[2]; int ns = zsegs(t, s, esz, sg);
    zsegs_zero((void *)t, sg, ns, 0, lo); zsegs_zero((void *)t, sg, ns, s.d - hi, hi);
    KCHECK();
}
cudaEvent_t zs_ev(int dev, int k) {   /* per-device events for the cross-GPU ordering */
    static cudaEvent_t ev[8][4]; static int init[8];
    if (!init[dev]) { int cur; cudaGetDevice(&cur); cudaSetDevice(dev); for (int i = 0; i < 4; i++) cudaEventCreateWithFlags(&ev[dev][i], cudaEventDisableTiming); cudaSetDevice(cur); init[dev] = 1; }
    return ev[dev][k];
}
void zs_xbar(const int *dev, int k) {
    int cur; cudaGetDevice(&cur);
    for (int i = 0; i < 2; i++) { cudaSetDevice(dev[i]); CK(cudaEventRecord(zs_ev(dev[i], k), 0)); }
    for (int i = 0; i < 2; i++) { cudaSetDevice(dev[i]); CK(cudaStreamWaitEvent(0, zs_ev(dev[1 - i], k), 0)); }
    cudaSetDevice(cur);
}
extern "C" size_t nn_split_halo_bytes(shape5 s, int esz) {   /* bytes of one plane (send / receive buffer size) */
    const size_t HW = (size_t)s.h * s.w;
    if (esz) return (size_t)s.n * s.c * HW * esz;
    const int bw = s.c <= 8 ? 8 : s.c <= 16 ? 16 : 32, nb = (s.c + bw - 1) / bw;
    return (size_t)s.n * nb * HW * (bw + 1);
}
/* both halves on one GPU (split_create(d, d): tests emulate the two-GPU split): the halo streams and events are per side */
static int g_zs_same = 0;
static int zs_key(int dev, int side) { return g_zs_same ? 8 + side : dev & 7; }
cudaStream_t zs_comm(int dev, int side) {
    static cudaStream_t st[10]; const int q = zs_key(dev, side);
    if (!st[q]) { int cur; cudaGetDevice(&cur); cudaSetDevice(dev); cudaStreamCreateWithFlags(&st[q], cudaStreamNonBlocking); cudaSetDevice(cur); }
    return st[q];
}
cudaEvent_t zs_hev(int dev, int side, int slot, int k) {   /* 0 packed, 1 arrived, 2 unpacked */
    static cudaEvent_t ev[10][2][3]; static int init[10]; const int q = zs_key(dev, side);
    if (!init[q]) { int cur; cudaGetDevice(&cur); cudaSetDevice(dev); for (int i = 0; i < 6; i++) cudaEventCreateWithFlags(&ev[q][i / 3][i % 3], cudaEventDisableTiming); cudaSetDevice(cur); init[q] = 1; }
    return ev[q][slot & 1][k];
}
extern "C" void nn_split_halo_begin(void *const *t, const int *dev, shape5 s, int esz, int h, void *const *sb, void *const *rb, int slot) {
    int cur; cudaGetDevice(&cur);
    g_zs_same = dev[0] == dev[1];
    zseg_t sg[2][2]; int ns = 0;
    for (int i = 0; i < 2; i++) { cudaSetDevice(dev[i]); ns = zsegs(t[i], s, esz, sg[i]); }
    const size_t nbytes = zsegs_plane_bytes(sg[0], ns);
    const int D = s.d;
    /* side 0 sends its last own plane D - h - 1, side 1 its first own plane h; the outer halo planes are zeroed */
    const int zsend[2] = {D - h - 1, h}, zlo[2] = {D - h + 1, 0};
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(dev[i]);
        CK(cudaStreamWaitEvent(0, zs_hev(dev[1 - i], 1 - i, slot, 1), 0));   /* the previous exchange has read this side's send buffer */
        zsegs_copy(t[i], sg[i], ns, zsend[i], 1, sb[i], 0);
        zsegs_zero(t[i], sg[i], ns, zlo[i], h - 1);
        CK(cudaEventRecord(zs_hev(dev[i], i, slot, 0), 0));
    }
    for (int i = 0; i < 2; i++) {
        cudaSetDevice(dev[i]);
        cudaStream_t cs = zs_comm(dev[i], i);
        CK(cudaStreamWaitEvent(cs, zs_hev(dev[1 - i], 1 - i, slot, 0), 0));
        CK(cudaStreamWaitEvent(cs, zs_hev(dev[i], i, slot, 2), 0));        /* the previous exchange has unpacked this receive buffer */
        CK(cudaMemcpyPeerAsync(rb[i], dev[i], sb[1 - i], dev[1 - i], nbytes, cs));
        CK(cudaEventRecord(zs_hev(dev[i], i, slot, 1), cs));
    }
    cudaSetDevice(cur);
    KCHECK();
}
extern "C" void nn_split_halo_end(void *t, int side, shape5 s, int esz, int h, void *rb, int slot) {   /* current device = this side's */
    zseg_t sg[2]; int ns = zsegs(t, s, esz, sg);
    const int dev = cur_dev();
    CK(cudaStreamWaitEvent(0, zs_hev(dev, side, slot, 1), 0));
    zsegs_copy(t, sg, ns, side ? h - 1 : s.d - h, 1, rb, 1);
    CK(cudaEventRecord(zs_hev(dev, side, slot, 2), 0));
    KCHECK();
}
__global__ void zs_add_k(double *a, const double *b, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) a[i] += b[i]; }
extern "C" void nn_split_allreduce(double *const *b, const int *dev, int n, double *const *r) {
    int cur; cudaGetDevice(&cur);
    zs_xbar(dev, 2);
    for (int i = 0; i < 2; i++) { cudaSetDevice(dev[i]); CK(cudaMemcpyPeerAsync(r[i], dev[i], b[1 - i], dev[1 - i], (size_t)n * sizeof(double), 0)); }
    zs_xbar(dev, 3);   /* both copies done before either sum overwrites its source */
    for (int i = 0; i < 2; i++) { cudaSetDevice(dev[i]); zs_add_k<<<nblk(n, 128), 128>>>(b[i], r[i], n); }
    cudaSetDevice(cur);
    KCHECK();
}

/* ---- affinity loss on band labels (task band_affinity) ----
   band[n][S]: band value of each voxel in steps of 1/18 turn mod 252 (winding_mod14), 255 = unknown. For output channel
   c0 + j (offset j: d voxels along axis a), voxel i owns the pair (i, i + d e_a) when i lies in this GPU's planes
   [zlo, D - zhi) and both voxels are known (the partner may lie in the split halo). Target 1 = same band (modular
   difference < half a turn), 0 = different. Loss per channel: class-weighted BCE (the "different" class weighted by
   clamp(#same / #different, 1, 10)) + dice_w * dice on the "different" class (prediction 1 - sigmoid). Gradient into the
   owning voxel's channel only. Statistics per channel (double): [n_same, n_diff, sum softplus(-x) same, sum softplus(x)
   diff, sum (1 - s), sum (1 - s) over diff]. */
#define AFF_NS 6
__device__ __forceinline__ int aff_target(uint8_t bi, uint8_t bj) {   /* 1 same band, 0 different */
    int dd = ((int)bi - (int)bj + 126) % 252; if (dd < 0) dd += 252; dd -= 126;
    return abs(dd) < 9;
}
/* row-major traversal: a (z, y) row per block iteration and threads along x, so the voxel coordinates need no 64-bit
   divisions; only this GPU's owned planes are visited */
template <typename LT> __global__ void __launch_bounds__(128) aff_stats_k(const LT *lg, const uint8_t *band, aff_offsets_t off, int C, int c0, int D, int H, int W, int zlo, int zhi, double *ds) {
    const int nj = blockIdx.y, n = nj / off.K, k = nj % off.K;
    const size_t S = (size_t)D * H * W;
    const LT *l = lg + ((size_t)n * C + c0 + k) * S; const uint8_t *bp = band + (size_t)n * S;
    const int a = off.ax[k], dz = a == 0 ? off.d[k] : 0, dy = a == 1 ? off.d[k] : 0, dx = a == 2 ? off.d[k] : 0;
    const long dj = ((long)dz * H + dy) * W + dx;
    float acc[AFF_NS] = {0, 0, 0, 0, 0, 0};
    const int rows = (D - zhi - zlo) * H;
    for (int r = blockIdx.x; r < rows; r += gridDim.x) {
        const int z = zlo + r / H, y = r % H;
        if (z + dz < 0 || z + dz >= D || y + dy < 0 || y + dy >= H) continue;
        const size_t row = ((size_t)z * H + y) * W;
        for (int x = threadIdx.x; x < W; x += blockDim.x) {
            if (x + dx < 0 || x + dx >= W) continue;
            const size_t i = row + x;
            const uint8_t bi = bp[i], bj = bp[i + dj];
            if (bi == 255 || bj == 255) continue;
            const float v = lgv(l, i), s = 1.f / (1.f + __expf(-v)), sp = fmaxf(v, 0.f) + log1pf(__expf(-fabsf(v)));
            if (aff_target(bi, bj)) { acc[0] += 1; acc[2] += sp - v; } else { acc[1] += 1; acc[3] += sp; acc[5] += 1.f - s; }
            acc[4] += 1.f - s;
        }
    }
    __shared__ double r[AFF_NS][128];
    for (int q = 0; q < AFF_NS; q++) r[q][threadIdx.x] = acc[q];
    __syncthreads();
    for (int o = 64; o > 0; o >>= 1) { if (threadIdx.x < o) for (int q = 0; q < AFF_NS; q++) r[q][threadIdx.x] += r[q][threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) for (int q = 0; q < AFF_NS; q++) atomicAdd(&ds[k * AFF_NS + q], r[q][0]);
}
/* fin per channel: [bce, dice, wdiff, 1 / (n_same + w n_diff), den, 2 I + 1] (6 floats per channel) */
__global__ void aff_fin_k(const double *ds, int K, float *fin) {
    const int k = threadIdx.x; if (k >= K) return;
    const double *t = ds + k * AFF_NS;
    const double ns = t[0], nd = t[1], w = nd > 0 ? fmin(10.0, fmax(1.0, ns / nd)) : 1.0, norm = ns + w * nd;
    const double den = t[4] + nd + 1.0, num = 2.0 * t[5] + 1.0;
    float *f = fin + k * 6;
    f[0] = norm > 0 ? (float)((t[2] + w * t[3]) / norm) : 0.f; f[1] = norm > 0 ? (float)(1.0 - num / den) : 0.f;
    f[2] = (float)w; f[3] = norm > 0 ? (float)(1.0 / norm) : 0.f; f[4] = (float)den; f[5] = (float)num;
}
/* block per (z, y) row (blockIdx.x), threads along x; every voxel's channel is written (0 when it owns no pair) */
template <typename GT, typename LT = float> __global__ void __launch_bounds__(128) aff_grad_k(const LT *lg, const uint8_t *band, aff_offsets_t off, int N, int C, int c0, int D, int H, int W,
                                                  int zlo, int zhi, const float *fin, float dice_w, float scale, GT *gl) {
    const size_t S = (size_t)D * H * W;
    const int nk = blockIdx.y, n = nk / off.K, k = nk % off.K;
    const int z = blockIdx.x / H, y = blockIdx.x % H;
    const int a = off.ax[k], dz = a == 0 ? off.d[k] : 0, dy = a == 1 ? off.d[k] : 0, dx = a == 2 ? off.d[k] : 0;
    const long dj = ((long)dz * H + dy) * W + dx;
    const size_t row = ((size_t)z * H + y) * W, ob = ((size_t)n * C + c0 + k) * S + row;
    const uint8_t *bp = band + (size_t)n * S + row;
    const bool rowok = z >= zlo && z < D - zhi && z + dz >= 0 && z + dz < D && y + dy >= 0 && y + dy < H;
    const float *f = fin + k * 6;
    for (int x = threadIdx.x; x < W; x += blockDim.x) {
        float g = 0.f;
        if (rowok && x + dx >= 0 && x + dx < W) {
            const uint8_t bi = bp[x], bj = bp[x + dj];
            if (bi != 255 && bj != 255) {
                const float v = lgv(lg, ob + x), s = 1.f / (1.f + expf(-v));
                const int same = aff_target(bi, bj);
                g = (same ? -(1.f - s) : f[2] * s) * f[3];
                g += dice_w * ((same ? 0.f : 2.f) * f[4] - f[5]) / (f[4] * f[4]) * s * (1.f - s);
            }
        }
        gl[ob + x] = f2h<GT>(g * scale);
    }
}
extern "C" size_t nn_aff_scratch(int K) { return (size_t)K * 6 * sizeof(float); }
extern "C" void nn_aff_loss_async(const float *logits, shape5 s, int c0, const uint8_t *band, aff_offsets_t off, float dice_w, float lambda,
                                  float *gl, float *fin) {
    const int zlo_hi_on = zs_on();
    int zlo = 0, zhi = 0; if (zlo_hi_on) zs_range(s.d, &zlo, &zhi);
    double *ds = gn_dsums((size_t)AFF_NS * off.K);
    cudaMemsetAsync(ds, 0, (size_t)AFF_NS * off.K * sizeof(double));
    if (g_logits_h16) aff_stats_k<f16><<<dim3(512, s.n * off.K), 128>>>((const f16 *)logits, band, off, s.c, c0, s.d, s.h, s.w, zlo, zhi, ds);
    else aff_stats_k<float><<<dim3(512, s.n * off.K), 128>>>(logits, band, off, s.c, c0, s.d, s.h, s.w, zlo, zhi, ds);
    zs_reduce(ds, AFF_NS * off.K);
    aff_fin_k<<<1, 32>>>(ds, off.K, fin);
    if (gl) {
        const dim3 grid((unsigned)(s.d * s.h), s.n * off.K);
        const float scale = lambda / off.K;   /* mean over channels (each channel's loss already covers all samples) */
        if (g_logits_h16) {
            const f16 *l16 = (const f16 *)logits;
            if (!g_loss_g16) aff_grad_k<float, f16><<<grid, 128>>>(l16, band, off, s.n, s.c, c0, s.d, s.h, s.w, zlo, zhi, fin, dice_w, scale, gl);
            else if (g_h16) aff_grad_k<f16, f16><<<grid, 128>>>(l16, band, off, s.n, s.c, c0, s.d, s.h, s.w, zlo, zhi, fin, dice_w, scale * g_gscale, (f16 *)gl);
            else VERIFY_ONLY(aff_grad_k<bf16, f16><<<grid, 128>>>(l16, band, off, s.n, s.c, c0, s.d, s.h, s.w, zlo, zhi, fin, dice_w, scale * g_gscale, (bf16 *)gl));
        } else
        if (!g_loss_g16) aff_grad_k<float><<<grid, 128>>>(logits, band, off, s.n, s.c, c0, s.d, s.h, s.w, zlo, zhi, fin, dice_w, scale, gl);
        else if (g_h16) aff_grad_k<f16><<<grid, 128>>>(logits, band, off, s.n, s.c, c0, s.d, s.h, s.w, zlo, zhi, fin, dice_w, scale * g_gscale, (f16 *)gl);
        else VERIFY_ONLY(aff_grad_k<bf16><<<grid, 128>>>(logits, band, off, s.n, s.c, c0, s.d, s.h, s.w, zlo, zhi, fin, dice_w, scale * g_gscale, (bf16 *)gl));
    }
    KCHECK();
}
/* host copy of the per-channel [bce, dice, wdiff, ...] (6 floats per channel) */
extern "C" void nn_aff_loss_fetch(const float *fin, int K, float *out) { CK(cudaMemcpy(out, fin, (size_t)K * 6 * sizeof(float), cudaMemcpyDeviceToHost)); }

/* ---- train --task unwrap2: the gauge-free winding mod 4 loss (nn.h nn_wind4_*) ----
   Two logits (a, b) per voxel predict 0.5 + 0.5 cos / sin (2 pi W / 4); the target angle is theta = 2 pi w4 / 252 (w4: W mod 4 in 63
   codes per turn). The gauge (W's integer part) is arbitrary per component, so the loss of component c is the soft BCE against the
   target rotated by the best of the four quarter turns g (W + g): bce_g = sum m [sp(a) + sp(b) - 0.5 (a + b) - 0.5 R_g], with
   R_0 = a cos + b sin, R_1 = b cos - a sin, R_2 = -R_0, R_3 = -R_1, so per component only N, S0 = sum m (sp(a) + sp(b)), H = sum m 0.5
   (a + b), X = sum m 0.5 R_0, Y = sum m 0.5 R_1 and the accuracy counts A_k (the predicted angle within 1/8 turn mod 4 of theta + k
   pi / 2) are summed: W4_NS per component. */
#define W4_NC 255   /* components (gauges) per window: comp 0..254, 255 none */
#define W4_NS 9
template <typename LT> __global__ void w4_stats_k(const LT *la, const LT *lb, const uint8_t *w4, const uint8_t *cp, const uint8_t *m, size_t n, double *ds) {
    __shared__ float sh[W4_NC * W4_NS];
    for (int k = threadIdx.x; k < W4_NC * W4_NS; k += blockDim.x) sh[k] = 0.f;
    __syncthreads();
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const unsigned c = cp[i], t = w4[i], mw = m[i];
        if (c >= W4_NC || t >= 252u || !mw) continue;
        const float a = lgv(la, i), b = lgv(lb, i), w = (float)mw;
        float st, ct; sincosf(6.283185307179586f * (float)t * (1.f / 252.f), &st, &ct);
        const float spa = fmaxf(a, 0.f) + log1pf(__expf(-fabsf(a))), spb = fmaxf(b, 0.f) + log1pf(__expf(-fabsf(b)));
        float *s = sh + c * W4_NS;
        atomicAdd(&s[0], w); atomicAdd(&s[1], w * (spa + spb)); atomicAdd(&s[2], w * 0.5f * (a + b));
        atomicAdd(&s[3], w * 0.5f * (a * ct + b * st)); atomicAdd(&s[4], w * 0.5f * (b * ct - a * st));
        const float pa = 2.f / (1.f + __expf(-a)) - 1.f, pb = 2.f / (1.f + __expf(-b)) - 1.f;
        /* the predicted rotation relative to the target in quarter turns, rounded: k = round(4 (phi - theta) / 2 pi) mod 4 */
        const float e = atan2f(pb * ct - pa * st, pa * ct + pb * st);   /* phi - theta in (-pi, pi] */
        const int k = ((int)lrintf(e * (2.f / 3.14159265358979f)) % 4 + 4) % 4;
        atomicAdd(&s[5 + k], w);
    }
    __syncthreads();
    for (int k = threadIdx.x; k < W4_NC * W4_NS; k += blockDim.x) if (sh[k] != 0.f) atomicAdd(&ds[k], (double)sh[k]);
}
/* per component the best gauge g* (largest of X, Y, -X, -Y); out: [0] loss (mean per channel of the best-gauge BCE over the
   supervised voxels), [1] accuracy (fraction with the predicted turn right at g*), [2] N, [3] 1 / N (0 when none), [4] components,
   [5] the same BCE at the worst gauge (a gauge-blind net has [0] == [5]), [8 + c] g* of component c */
__global__ void w4_fin_k(const double *ds, float *out) {
    if (threadIdx.x || blockIdx.x) return;
    double N = 0, L = 0, Lw = 0, A = 0; int nc = 0;
    for (int c = 0; c < W4_NC; c++) {
        const double *s = ds + c * W4_NS;
        out[8 + c] = 0.f;
        if (s[0] <= 0) continue;
        const double R[4] = {s[3], s[4], -s[3], -s[4]};
        int g = 0, gw = 0; for (int k = 1; k < 4; k++) { if (R[k] > R[g]) g = k; if (R[k] < R[gw]) gw = k; }
        N += s[0]; L += s[1] - s[2] - R[g]; Lw += s[1] - s[2] - R[gw]; A += s[5 + g]; nc++;
        out[8 + c] = (float)g;
    }
    out[0] = N > 0 ? (float)(L / (2 * N)) : 0.f; out[1] = N > 0 ? (float)(A / N) : 0.f; out[2] = (float)N; out[3] = N > 0 ? (float)(1 / N) : 0.f;
    out[4] = (float)nc; out[5] = N > 0 ? (float)(Lw / (2 * N)) : 0.f; out[6] = out[7] = 0.f;
}
template <typename GT, typename LT> __global__ void w4_grad_k(const LT *la, const LT *lb, const uint8_t *w4, const uint8_t *cp, const uint8_t *m, size_t n,
                                                             const float *fin, float scale, GT *ga, GT *gb) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned c = cp[i], t = w4[i], mw = m[i];
    float da = 0.f, db = 0.f;
    if (c < W4_NC && t < 252u && mw) {
        const float a = lgv(la, i), b = lgv(lb, i);
        const int g = (int)fin[8 + c];
        float st, ct; sincosf(6.283185307179586f * ((float)t * (1.f / 252.f) + 0.25f * (float)g), &st, &ct);   /* the target rotated by g quarter turns */
        const float k = (float)mw * 0.5f * fin[3] * scale;   /* mean over the two channels and the supervised voxels */
        da = k * (1.f / (1.f + __expf(-a)) - (0.5f + 0.5f * ct)); db = k * (1.f / (1.f + __expf(-b)) - (0.5f + 0.5f * st));
    }
    ga[i] = f2h<GT>(da); gb[i] = f2h<GT>(db);   /* (in place over the logits: each thread reads its two logits first) */
}
extern "C" size_t nn_wind4_scratch(void) { return (8 + W4_NC) * sizeof(float); }
extern "C" size_t nn_wind4_sums(void) { return (size_t)W4_NC * W4_NS * sizeof(double); }
extern "C" void nn_wind4_stats(const void *la, const void *lb, const uint8_t *w4, const uint8_t *comp, const uint8_t *m, size_t n, double *ds) {
    const unsigned nb = (unsigned)(n / 2048 + 1 < 1024 ? n / 2048 + 1 : 1024);
    if (g_logits_h16) w4_stats_k<f16><<<nb, 256>>>((const f16 *)la, (const f16 *)lb, w4, comp, m, n, ds);
    else w4_stats_k<float><<<nb, 256>>>((const float *)la, (const float *)lb, w4, comp, m, n, ds);
    KCHECK();
}
extern "C" void nn_wind4_fin(double *ds, float *out) {
    zs_reduce(ds, W4_NC * W4_NS);   /* spatial split: both halves' sums (halo planes masked by the caller), the same gauges */
    w4_fin_k<<<1, 32>>>(ds, out);
    KCHECK();
}
extern "C" void nn_wind4_grad(const void *la, const void *lb, const uint8_t *w4, const uint8_t *comp, const uint8_t *m, size_t n, const float *fin, float lambda, void *ga, void *gb) {
    const unsigned g = nblk(n, 256);
#define W4G(GT, LT, s) w4_grad_k<GT, LT><<<g, 256>>>((const LT *)la, (const LT *)lb, w4, comp, m, n, fin, (s), (GT *)ga, (GT *)gb)
    if (g_logits_h16) { if (!g_loss_g16) W4G(float, f16, lambda); else if (g_h16) W4G(f16, f16, lambda * g_gscale); else VERIFY_ONLY(W4G(bf16, f16, lambda * g_gscale)); }
    else { if (!g_loss_g16) W4G(float, float, lambda); else if (g_h16) W4G(f16, float, lambda * g_gscale); else VERIFY_ONLY(W4G(bf16, float, lambda * g_gscale)); }
#undef W4G
    KCHECK();
}
/* the whole loss in one call (whole windows): ds (W4_NC * W4_NS doubles) zeroed, stats, gauges, gradient (ga / gb nullptr: none) */
extern "C" void nn_wind4_loss(const void *la, const void *lb, const uint8_t *w4, const uint8_t *comp, const uint8_t *m, size_t n, double *ds, float *fin, float lambda, void *ga, void *gb) {
    cudaMemsetAsync(ds, 0, W4_NC * W4_NS * sizeof(double));
    nn_wind4_stats(la, lb, w4, comp, m, n, ds);
    nn_wind4_fin(ds, fin);
    if (ga) nn_wind4_grad(la, lb, w4, comp, m, n, fin, lambda, ga, gb);
}

/* train --distill: the cosine feature-distillation loss (nn.h nn_distill_loss). One thread per voxel; channel c of sample b at y[(b C + c) S + i] */
__global__ void distill_k(const float *y, const int8_t *t, int N, int C, size_t S, float *g, double *ds) {
    __shared__ float sh[2];
    if (threadIdx.x < 2) sh[threadIdx.x] = 0.f;
    __syncthreads();
    const size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k < (size_t)N * S) {
        const size_t b = k / S, i = k - b * S, base = b * (size_t)C * S + i;
        float yy = 0.f, tt = 0.f, yt = 0.f;
        for (int c = 0; c < C; c++) { const float a = y[base + c * S], q = (float)t[base + c * S]; yy += a * a; tt += q * q; yt += a * q; }
        if (tt > 0.f) {
            const float ny = sqrtf(yy) + 1e-6f, nt = sqrtf(tt), cs = yt / (ny * nt);
            for (int c = 0; c < C; c++) g[base + c * S] = -((float)t[base + c * S] / nt - cs * y[base + c * S] / ny) / ny;
            atomicAdd(&sh[0], 1.f - cs); atomicAdd(&sh[1], 1.f);
        } else for (int c = 0; c < C; c++) g[base + c * S] = 0.f;
    }
    __syncthreads();
    if (threadIdx.x < 2 && sh[threadIdx.x] != 0.f) atomicAdd(&ds[threadIdx.x], (double)sh[threadIdx.x]);
}
__global__ void distill_scale_k(float *g, size_t n, const double *ds, float w) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) g[i] *= ds[1] > 0 ? (float)(w / ds[1]) : 0.f;
}
extern "C" void nn_distill_loss(const float *y, const int8_t *t, int n, int C, size_t S, float w, float *g, double *ds) {
    cudaMemsetAsync(ds, 0, 2 * sizeof(double));
    distill_k<<<nblk((size_t)n * S, 256), 256>>>(y, t, n, C, S, g, ds);
    KCHECK();
    distill_scale_k<<<nblk((size_t)n * C * S, 256), 256>>>(g, (size_t)n * C * S, ds, w);
    KCHECK();
}

/* ---- phase wavenumber loss (train --pgrad-lambda): the predicted phase's rotation between a voxel and its partner --pgrad-dist
   voxels along the ground-truth phase gradient, against the label's rotation there. Per pair i -> j: a = z_i, b = z_j the predicted
   phasors (2 sigmoid(logit) - 1 per channel), u = exp(2 pi i dW) the label's rotation (dW = the phase codes' wrapped difference),
   q = b conj(a); loss 1 - Re(q conj(u)) / (|a| |b| + eps): angular, so a blurred (small, slowly turning) phasor is not cheaper the
   way it is under the per-voxel BCE, and a phase that skips every other sheet turns at half the label's rate and pays 1 - cos(pi dW).
   Weighted by the phase mask (the packed-sheet weight: train --packed-weight), mean over the pairs. Head slabs [z0, z1) of the window:
   both ends inside the slab; the gradient direction from the codes over +-s voxels of the whole window. sums (device doubles): sum
   w loss, sum w (pass 0 accumulates them over the slabs; pass 1 reads sum w). ---- */
template <typename LT> __global__ void pgrad_k(const LT *lc, const LT *ls, const uint8_t *ph, const uint8_t *wm, int D, int H, int W, int z0, int z1,
                                               int s, float dist, double *sums, float *acc) {
    const size_t HW = (size_t)H * W, ns = (size_t)(z1 - z0) * HW;
    const size_t li = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    float wl = 0.f, ww = 0.f;
    if (li < ns) do {
        const float w = wm[li]; if (w <= 0.f) break;
        const int z = z0 + (int)(li / HW), y = (int)(li / W % H), x = (int)(li % W);
        const size_t gi = (size_t)z * HW + (size_t)y * W + x; const int p0 = ph[gi];
        if (p0 >= 252 || z < s || z >= D - s || y < s || y >= H - s || x < s || x >= W - s) break;
        const size_t sz = (size_t)s * HW, sy = (size_t)s * W;
        const int a6[6] = {ph[gi + sz], ph[gi - sz], ph[gi + sy], ph[gi - sy], ph[gi + s], ph[gi - s]};
        int ok = 1; for (int q = 0; q < 6; q++) ok &= a6[q] < 252; if (!ok) break;
        float g[3] = {(float)cnt_pdiff(a6[0], a6[1]), (float)cnt_pdiff(a6[2], a6[3]), (float)cnt_pdiff(a6[4], a6[5])};
        const float gn = sqrtf(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]); if (gn < 1e-3f) break;
        const int zz = (int)lrintf(z + dist * g[0] / gn), yy = (int)lrintf(y + dist * g[1] / gn), xx = (int)lrintf(x + dist * g[2] / gn);
        if (zz < z0 || zz >= z1 || yy < 0 || yy >= H || xx < 0 || xx >= W) break;
        const size_t gj = (size_t)zz * HW + (size_t)yy * W + xx; const int p1 = ph[gj]; if (p1 >= 252) break;
        const size_t lj = gj - (size_t)z0 * HW; if (lj == li) break;
        const float dW = cnt_pdiff(p1, p0) * (1.f / 252.f), ur = cospif(2.f * dW), ui = sinpif(2.f * dW);
        const float sa = 1.f / (1.f + __expf(-lgv(lc, li))), sb = 1.f / (1.f + __expf(-lgv(ls, li)));
        const float sc = 1.f / (1.f + __expf(-lgv(lc, lj))), sd = 1.f / (1.f + __expf(-lgv(ls, lj)));
        const float ar = 2.f * sa - 1.f, ai = 2.f * sb - 1.f, br = 2.f * sc - 1.f, bi = 2.f * sd - 1.f;
        /* R = Re(b conj(a) conj(u)); M = |a| |b| */
        const float kr = ar * ur - ai * ui, ki = -(ar * ui + ai * ur);   /* k = conj(a u) */
        const float R = br * kr - bi * ki;
        const float na = sqrtf(ar * ar + ai * ai) + 1e-6f, nb = sqrtf(br * br + bi * bi) + 1e-6f, M = na * nb, e = .05f, den = M + e;
        wl = w * (1.f - R / den); ww = w;
        if (acc) {
            const float mr = br * ur + bi * ui, mi = bi * ur - br * ui;   /* m = b conj(u): R = Re(conj(a) m) */
            const float c0 = -w / den, c1 = w * R / (den * den);           /* dloss = c0 dR + c1 dM */
            const float gar = c0 * mr + c1 * ar * nb / na, gai = c0 * mi + c1 * ai * nb / na;
            const float gbr = c0 * kr + c1 * br * na / nb, gbi = -c0 * ki + c1 * bi * na / nb;
            atomicAdd(&acc[li], gar * 2.f * sa * (1.f - sa)); atomicAdd(&acc[ns + li], gai * 2.f * sb * (1.f - sb));
            atomicAdd(&acc[lj], gbr * 2.f * sc * (1.f - sc)); atomicAdd(&acc[ns + lj], gbi * 2.f * sd * (1.f - sd));
        }
    } while (0);
    for (int o = 16; o; o >>= 1) { wl += __shfl_down_sync(0xffffffffu, wl, o); ww += __shfl_down_sync(0xffffffffu, ww, o); }
    __shared__ float sh[2][32];
    if ((threadIdx.x & 31) == 0) { sh[0][threadIdx.x >> 5] = wl; sh[1][threadIdx.x >> 5] = ww; }
    __syncthreads();
    if (threadIdx.x == 0) {
        double a = 0, b = 0; for (int k = 0; k < (int)(blockDim.x >> 5); k++) { a += sh[0][k]; b += sh[1][k]; }
        if (b > 0) { atomicAdd(&sums[0], a); atomicAdd(&sums[1], b); }
    }
}
template <typename GT> __global__ void pgrad_apply_k(GT *gc, GT *gs, float *acc, size_t ns, const double *sums, float scale) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= ns) return;
    const float a = acc[i], b = acc[ns + i]; if (a == 0.f && b == 0.f) return;
    acc[i] = acc[ns + i] = 0.f;
    const double sw = sums[1]; if (!(sw > 0)) return;
    const float f = (float)(scale / sw);
    gc[i] = f2h<GT>((float)gc[i] + f * a); gs[i] = f2h<GT>((float)gs[i] + f * b);
}
extern "C" void nn_pgrad_slab(const void *lc, const void *ls, const uint8_t *phase, const uint8_t *wm, int d, int h, int w, int z0, int z1, int s, float dist,
                              double *sums, float *acc) {
    const size_t ns = (size_t)(z1 - z0) * h * w;
    if (g_logits_h16) pgrad_k<f16><<<nblk(ns, 256), 256>>>((const f16 *)lc, (const f16 *)ls, phase, wm, d, h, w, z0, z1, s, dist, sums, acc);
    else pgrad_k<float><<<nblk(ns, 256), 256>>>((const float *)lc, (const float *)ls, phase, wm, d, h, w, z0, z1, s, dist, sums, acc);
    KCHECK();
}
extern "C" void nn_pgrad_apply(void *gc, void *gs, float *acc, size_t ns, const double *sums, float lambda) {
    if (!g_loss_g16) { fprintf(stderr, "nn_pgrad_apply: needs 16-bit logit gradients (head slabs)\n"); abort(); }
    if (g_h16) pgrad_apply_k<f16><<<nblk(ns, 256), 256>>>((f16 *)gc, (f16 *)gs, acc, ns, sums, lambda * g_gscale);
    else pgrad_apply_k<bf16><<<nblk(ns, 256), 256>>>((bf16 *)gc, (bf16 *)gs, acc, ns, sums, lambda * g_gscale);
    KCHECK();
}
