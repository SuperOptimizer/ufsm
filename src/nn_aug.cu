/* train --gpu-aug: a window's CT channel built on the device from its source-grid bytes (sample.h batch.ct8 / batch.aug):
   the scanner ops of its scan_plan (scan_augment.h; the float ops of scan_aug_exec, here in 16-bit storage in the CT channel's
   own buffer), the output bytes and their z-score, then the cube symmetry or the warp with the intensity jitter (sample.c write_x_ct /
   write_spatial_tm channel 0) into the 16-bit CT channel. The CPU made every decision (the plan, the record); the results match the
   CPU path within float tolerance (tests/test_gpu_aug.c), the zero set exactly. Device memory: one P^3 byte staging buffer, the plan, a
   few P^2 planes. */
#include "nn_common.cuh"
#include "scan_augment.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

typedef struct {
    uint8_t *stage; size_t stage_n;   /* the window's bytes: input, then output */
    scan_plan *plan;                  /* the device copy of the plan */
    scan_ct_rec *rec;                 /* the device copy of the record (without its plan) */
    float *pl[4]; size_t pl_n;        /* P^2 planes: correlated-noise rows / state, GIN ring / plane */
    double *dsc;                      /* GIN g1, g2 */
    unsigned long long *usc;          /* GIN gn; output sum, sum of squares; hist count */
    unsigned *hcnt;                   /* hist bins */
    float *hm;                        /* hist map (257) */
    float *scal;                      /* fmean, fisd, gm, gs */
    float *ntab;                      /* the jitter noise table (sample.c g_ntab) */
    float *cssn;                      /* --gpu-aug 2: the phase codes' cos, sin (256 each) */
    cudaEvent_t done; int used;       /* the last window's work (the staging buffer is shared by both streams' calls) */
} aug_dev;
static aug_dev g_ad[8];

static __device__ __forceinline__ float h2f(__half h) { return __half2float(h); }
static __device__ __forceinline__ __half f2h(float f) { return __float2half_rn(f); }
static __device__ __forceinline__ uint64_t sm64(uint64_t s) {   /* SplitMix64's output for state s (scan_augment.c next_u64 after its add) */
    uint64_t v = s; v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ull; v = (v ^ (v >> 27)) * 0x94d049bb133111ebull; return v ^ (v >> 31);
}
static __device__ __forceinline__ float ih4d(uint32_t v) { return (float)((int)(v & 255) + (int)((v >> 8) & 255) + (int)((v >> 16) & 255) + (int)(v >> 24) - 510) * (1.f / 147.80f); }
/* gauss_row's value x of the row that starts at stream call q0 (the stream's seed ns): call q0 + x / 2 (+1: next_u64 adds first) */
static __device__ __forceinline__ float gauss_at(uint64_t ns, uint64_t q0, int x) {
    const uint64_t v = sm64(ns + (q0 + (uint64_t)(x >> 1) + 1) * 0x9e3779b97f4a7c15ull);
    return ih4d(x & 1 ? (uint32_t)(v >> 32) : (uint32_t)v);
}
/* the output's air: the input byte 0 (unless SA_AIR), or a chunk dropout box (with_chunks: from that op on) */
static __device__ __forceinline__ int air_at(const scan_plan *p, const uint8_t *in, size_t k, int z, int y, int x, int with_chunks) {
    int a = !(p->on & 1u << SA_AIR) && !in[k];
    if (with_chunks) for (int i = 0; i < p->nchunk && !a; i++) {
        const int *b0 = p->chunk_b0[i], *b1 = p->chunk_b1[i];
        a = z >= b0[0] && z < b1[0] && y >= b0[1] && y < b1[1] && x >= b0[2] && x < b1[2];
    }
    return a;
}

__global__ void aug_load_k(const uint8_t *in, __half *F, size_t n, int air, float level) {
    const size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= n) return;
    const uint8_t v = in[k];
    F[k] = f2h(air && !v ? level : (float)v);
}
/* a separable operator along the contiguous axis: one block per row, the row in shared memory */
__global__ void aug_bop_x_k(__half *F, const scan_bop *b, int P) {
    extern __shared__ float sh[];
    __half *row = F + (size_t)blockIdx.x * P;
    for (int x = threadIdx.x; x < P; x += blockDim.x) sh[x] = h2f(row[x]);
    __syncthreads();
    for (int i = threadIdx.x; i < P; i += blockDim.x) {
        const float *w = b->w[i]; const int s = b->s[i], n = b->n[i];
        float a = w[0] * sh[s];
        for (int t = 1; t < n; t++) a += w[t] * sh[s + t];
        row[i] = f2h(a);
    }
}
/* along axis 0 (st = P^2, u = y) or 1 (st = P, u = z): 32 lines side by side (consecutive x), P x 32 halves in shared memory */
__global__ void aug_bop_zy_k(__half *F, const scan_bop *b, int P, int axis) {
    extern __shared__ __half shh[];
    const int x = blockIdx.x * 32 + threadIdx.x, u = blockIdx.y;
    const size_t P2 = (size_t)P * P, st = axis == 0 ? P2 : (size_t)P, base = axis == 0 ? (size_t)u * P : (size_t)u * P2;
    if (x < P) for (int i = threadIdx.y; i < P; i += blockDim.y) shh[i * 32 + threadIdx.x] = F[base + i * st + x];
    __syncthreads();
    if (x >= P) return;
    for (int i = threadIdx.y; i < P; i += blockDim.y) {
        const float *w = b->w[i]; const int s = b->s[i], n = b->n[i];
        float a = w[0] * h2f(shh[s * 32 + threadIdx.x]);
        for (int t = 1; t < n; t++) a += w[t] * h2f(shh[(s + t) * 32 + threadIdx.x]);
        F[base + i * st + x] = f2h(a);
    }
}
__global__ void aug_point_k(__half *F, const scan_plan *p, int P, int what) {   /* what: 0 bias, 1 radial, 2 slice */
    const size_t P2 = (size_t)P * P, n = P2 * P, k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= n) return;
    const int z = (int)(k / P2), y = (int)(k / P % P), x = (int)(k % P);
    float f = h2f(F[k]);
    if (what == 0) {
        const int g = p->bias_g; const float sc = (float)(g - 1) / (float)(P - 1);
        auto ixw = [&](int u, int *i, float *w) { float t = u * sc; int j = (int)t; if (j > g - 2) j = g - 2; *i = j; *w = t - j; };
        int iz, iy, ix; float fz, fy, wx; ixw(z, &iz, &fz); ixw(y, &iy, &fy); ixw(x, &ix, &wx);
        const float *G = p->bias_G;
        auto crow = [&](int j) { const float *q = G + ((size_t)iz * g + iy) * g + j; return (1 - fz) * ((1 - fy) * q[0] + fy * q[g]) + fz * ((1 - fy) * q[g * g] + fy * q[g * g + g]); };
        const float c0 = crow(ix), c1 = crow(ix + 1);
        f *= c0 + wx * (c1 - c0);
    } else if (what == 1) {
        const float dy = y - p->ccy[z], dx = x - p->ccx[z];
        float u = (sqrtf(dy * dy + dx * dx) - p->rmin) * p->res; if (u < 0) u = 0; if (u > p->nt - 2) u = (float)(p->nt - 2);
        const int i = (int)u; f += p->tab[i] + (u - i) * (p->tab[i + 1] - p->tab[i]);
    } else f = f * p->slice_gain[z] + p->slice_off[z];
    F[k] = f2h(f);
}
__global__ void aug_noise_k(__half *F, const scan_plan *p, int P, uint64_t q0) {   /* white noise: row r = z P + y from call q0 + r ceil(P / 2) */
    const size_t P2 = (size_t)P * P, n = P2 * P, k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= n) return;
    const uint64_t hp = (uint64_t)(P + 1) / 2, r = k / P; const int x = (int)(k % P);
    const float g = gauss_at(p->ns, q0 + r * hp, x), f = h2f(F[k]);
    F[k] = f2h(f + p->noise_s0 * g * ((1 - p->noise_mix) + p->noise_mix * sqrtf(fmaxf(f, 0.f) * p->noise_im)));
}
/* correlated noise of plane z: rows along x (one thread per row), then y and z (one thread per column x) */
__global__ void aug_cnoise_x_k(const scan_plan *p, int P, int z, uint64_t q0, float *S) {
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= P) return;
    const float rho = p->cnoise_rho, scale = sqrtf(1.f - rho * rho); const uint64_t hp = (uint64_t)(P + 1) / 2, qr = q0 + ((uint64_t)z * P + y) * hp;
    float last = 0;
    for (int x = 0; x < P; x++) { const float g = gauss_at(p->ns, qr, x), vx = x ? rho * last + scale * g : g; last = vx; S[(size_t)y * P + x] = vx; }
}
__global__ void aug_cnoise_yz_k(__half *F, const scan_plan *p, int P, int z, const float *S, float *PP) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= P) return;
    const float rho = p->cnoise_rho, scale = sqrtf(1.f - rho * rho), s0 = p->cnoise_s0;
    float prev = 0;
    __half *pl = F + (size_t)z * P * P;
    for (int y = 0; y < P; y++) {
        const size_t i = (size_t)y * P + x; const float vx = S[i], vy = y ? rho * prev + scale * vx : vx; prev = vy;
        const float vz = z ? rho * PP[i] + scale * vy : vy; PP[i] = vz;
        pl[i] = f2h(h2f(pl[i]) + s0 * vz);
    }
}
/* paganin's two-sided exponential along axis 0 or 1: one thread per line (u, x), coalesced across x */
__global__ void aug_exp_zy_k(__half *F, int P, int axis, float a) {
    const size_t P2 = (size_t)P * P, t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (t >= P2) return;
    const int u = (int)(t / P), x = (int)(t % P); const float b = 1 - a;
    const size_t st = axis == 0 ? P2 : (size_t)P, base = (axis == 0 ? (size_t)u * P : (size_t)u * P2) + x;
    float y = h2f(F[base]); y = b * y + a * y; F[base] = f2h(y);
    for (int i = 1; i < P; i++) { const size_t j = base + i * st; y = b * h2f(F[j]) + a * y; F[j] = f2h(y); }
    for (int i = P - 2; i >= 0; i--) { const size_t j = base + i * st; y = b * h2f(F[j]) + a * y; F[j] = f2h(y); }
}
/* along the contiguous axis: 32 rows per block in shared memory (as floats in the passes), one thread per row */
__global__ void aug_exp_x_k(__half *F, int P, float a) {
    extern __shared__ float shf[];
    const size_t r0 = (size_t)blockIdx.x * 32, nrow = (size_t)P * P;
    for (int j = threadIdx.x; j < 32 * P; j += blockDim.x) { const size_t r = r0 + j / P; shf[j] = r < nrow ? h2f(F[r0 * P + j]) : 0.f; }
    __syncthreads();
    if (threadIdx.x < 32 && r0 + threadIdx.x < nrow) {
        float *v = shf + (size_t)threadIdx.x * P; const float b = 1 - a;
        float y = b * v[0] + a * v[0]; v[0] = y;
        for (int i = 1; i < P; i++) { y = b * v[i] + a * y; v[i] = y; }
        for (int i = P - 2; i >= 0; i--) { y = b * v[i] + a * y; v[i] = y; }
    }
    __syncthreads();
    for (int j = threadIdx.x; j < 32 * P; j += blockDim.x) { const size_t r = r0 + j / P; if (r < nrow) F[r0 * P + j] = f2h(shf[j]); }
}
/* GIN's output at (z, y, x): the normalised CT's k^3 neighbourhood (edges replicated); plane z - 1 from prevp when given (its
   original values, the plane itself already blended) */
static __device__ float gin_at(const __half *F, const float *prevp, const scan_plan *p, int P, int z, int y, int x, float im, float is) {
    const int k = p->gin_k, r = k / 2, C = p->gin_C; float h[4] = {0, 0, 0, 0};
    for (int dz = -r, t = 0; dz <= r; dz++) for (int dy = -r; dy <= r; dy++) for (int dx = -r; dx <= r; dx++, t++) {
        const int zz = min(max(z + dz, 0), P - 1), yy = min(max(y + dy, 0), P - 1), xx = min(max(x + dx, 0), P - 1);
        const size_t pi = (size_t)yy * P + xx;
        const float raw = prevp && zz == z - 1 ? prevp[pi] : h2f(F[(size_t)zz * P * P + pi]);
        const float v = (raw - im) * is;
        for (int ch = 0; ch < C; ch++) h[ch] += p->gin_w1[ch][t] * v;
    }
    float o = 0;
    for (int ch = 0; ch < C; ch++) { const float u = h[ch] + p->gin_b1[ch]; o += p->gin_w2[ch] * (u > 0 ? u : .2f * u); }
    return o;
}
__global__ void aug_gin_stats_k(const __half *F, const uint8_t *in, const scan_plan *p, int P, double *dsc, unsigned long long *usc) {
    const size_t P2 = (size_t)P * P, n = P2 * P, k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    float g1 = 0, g2 = 0; unsigned gn = 0;
    if (k < n) {
        const int z = (int)(k / P2), y = (int)(k / P % P), x = (int)(k % P);
        if (!air_at(p, in, k, z, y, x, 0)) { const float o = gin_at(F, nullptr, p, P, z, y, x, (float)p->mean, (float)(1 / p->sd)); g1 = o; g2 = o * o; gn = 1; }
    }
    for (int s = 16; s; s >>= 1) { g1 += __shfl_down_sync(~0u, g1, s); g2 += __shfl_down_sync(~0u, g2, s); gn += __shfl_down_sync(~0u, gn, s); }
    if ((threadIdx.x & 31) == 0 && gn) { atomicAdd(&dsc[0], (double)g1); atomicAdd(&dsc[1], (double)g2); atomicAdd(&usc[0], (unsigned long long)gn); }
}
__global__ void aug_gin_fin_k(const double *dsc, const unsigned long long *usc, float *scal) {
    const double gn = (double)usc[0], gm = gn > 0 ? dsc[0] / gn : 0, gs = gn > 0 ? sqrt(fmax(dsc[1] / gn - gm * gm, 1e-12)) : 1;
    scal[2] = (float)gm; scal[3] = (float)gs;
}
/* k == 1: every voxel at once (no neighbourhood); k == 3: plane z (G into Gp from the original plane z - 1 in prevp) */
__global__ void aug_gin1_k(__half *F, const scan_plan *p, int P, const float *scal) {
    const size_t P2 = (size_t)P * P, n = P2 * P, k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= n) return;
    const int z = (int)(k / P2), y = (int)(k / P % P), x = (int)(k % P);
    const float mean = (float)p->mean, sd = (float)p->sd, al = (float)p->gin_alpha;
    const float G = gin_at(F, nullptr, p, P, z, y, x, mean, 1.f / sd);
    F[k] = f2h(al * (mean + sd * (G - scal[2]) / scal[3]) + (1 - al) * h2f(F[k]));
}
__global__ void aug_gin3_g_k(const __half *F, const float *prevp, const scan_plan *p, int P, int z, float *Gp) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)P * P) return;
    Gp[i] = gin_at(F, z ? prevp : nullptr, p, P, z, (int)(i / P), (int)(i % P), (float)p->mean, (float)(1 / p->sd));
}
__global__ void aug_gin3_b_k(__half *F, const scan_plan *p, int P, int z, const float *Gp, float *prevp, const float *scal) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)P * P) return;
    __half *q = F + (size_t)z * P * P + i; const float f = h2f(*q);
    prevp[i] = f;   /* (plane z's original values, for plane z + 1's neighbourhood) */
    const float mean = (float)p->mean, sd = (float)p->sd, al = (float)p->gin_alpha;
    *q = f2h(al * (mean + sd * (Gp[i] - scal[2]) / scal[3]) + (1 - al) * f);
}
__global__ void aug_plane_k(__half *F, const scan_plan *p, int P, int z, int src, int mode, uint64_t q0) {   /* dropped slice */
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, P2 = (size_t)P * P;
    if (i >= P2) return;
    __half *pl = F + (size_t)z * P2;
    if (mode == 0) { pl[i] = F[(size_t)src * P2 + i]; return; }
    const uint64_t hp = (uint64_t)(P + 1) / 2; const int r = (int)(i / P), x = (int)(i % P);
    pl[i] = f2h((float)(p->mean + (mode == 2 ? 1.5 * p->sd * gauss_at(p->ns, q0 + r * hp, x) : 0)));
}
__global__ void aug_cut_k(__half *F, const scan_plan *p, int P, int bi, uint64_t q0) {   /* cutout box bi */
    const int *b0 = p->cut_b0[bi], *b1 = p->cut_b1[bi], nx = b1[2] - b0[2], ny = b1[1] - b0[1];
    const size_t n = (size_t)(b1[0] - b0[0]) * ny * nx, t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int dz = (int)(t / ((size_t)ny * nx)), dy = (int)(t / nx % ny), x = b0[2] + (int)(t % nx);
    const uint64_t hp = (uint64_t)(P + 1) / 2;
    const float g = p->cut_noisy[bi] ? gauss_at(p->ns, q0 + ((uint64_t)dz * ny + dy) * hp, x) : 0.f;
    F[((size_t)(b0[0] + dz) * P + b0[1] + dy) * P + x] = f2h((float)(p->mean + (p->cut_noisy[bi] ? p->sd * g : 0)));
}
__global__ void aug_lut_k(__half *F, const float *lut, size_t n) {   /* the tone curve (before the histogram) */
    const size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= n) return;
    const float v = fminf(fmaxf(h2f(F[k]), 0.f), 255.f); const int i = (int)v;
    F[k] = f2h(lut[i] + (v - i) * (lut[i + 1] - lut[i]));
}
__global__ void aug_hist_k(const __half *F, const uint8_t *in, const scan_plan *p, int P, unsigned *cnt, unsigned long long *usc) {
    __shared__ unsigned h[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) h[i] = 0;
    __syncthreads();
    const size_t P2 = (size_t)P * P, n = P2 * P, k = 7 * (blockIdx.x * (size_t)blockDim.x + threadIdx.x);
    if (k < n) {
        const int z = (int)(k / P2), y = (int)(k / P % P), x = (int)(k % P);
        if (!air_at(p, in, k, z, y, x, 1)) { const float v = fminf(fmaxf(h2f(F[k]), 1.f), 255.f); atomicAdd(&h[(int)v], 1u); }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < 256; i += blockDim.x) if (h[i]) { atomicAdd(&cnt[i], h[i]); atomicAdd(&usc[3], (unsigned long long)h[i]); }
}
__global__ void aug_hist_map_k(const unsigned *cnt, const unsigned long long *usc, const scan_plan *p, float *hm) {   /* one thread: scan_aug_exec's map */
    const double nw = (double)usc[3];
    if (nw <= 0) { hm[0] = -1; return; }   /* (no material: no map) */
    const float *ct = p->hist_ct; double acc = 0; int t = 1;
    for (int v = 0; v < 256; v++) {
        acc += cnt[v]; const double q = acc / nw;
        while (t < 255 && ct[t] < q) t++;
        const double lo = ct[t - 1], hi = ct[t], f = hi > lo ? (q - lo) / (hi - lo) : 1;
        hm[v] = (float)(t - 1 + fmin(1., fmax(0., f)));
    }
    hm[256] = hm[255];
}
__global__ void aug_hist_apply_k(__half *F, const float *hm, const scan_plan *p, size_t n) {
    const size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= n || hm[0] < 0) return;
    const float f = h2f(F[k]), v = fminf(fmaxf(f, 0.f), 255.f); const int i = (int)v;
    const float m = hm[i] + (v - i) * (hm[i + 1] - hm[i]), al = (float)p->hist_alpha;
    F[k] = f2h(al * m + (1 - al) * f);
}
/* the output bytes (tone unless the histogram ran, quantisation, rounding to 1..255, air 0) and their sums */
__global__ void aug_final_k(const __half *F, uint8_t *io, const scan_plan *p, int P, unsigned long long *usc) {
    const size_t P2 = (size_t)P * P, n = P2 * P, k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    unsigned long long s1 = 0, s2 = 0;
    if (k < n) {
        const int z = (int)(k / P2), y = (int)(k / P % P), x = (int)(k % P);
        const int air = air_at(p, io, k, z, y, x, 1);
        float f = h2f(F[k]);
        if (p->tone && !p->hist_on) { const float v = fminf(fmaxf(f, 0.f), 255.f); const int i = (int)v; f = p->lut[i] + (v - i) * (p->lut[i + 1] - p->lut[i]); }
        if (p->step > 0) f = (floorf(f * (1 / p->step)) + .5f) * p->step;
        const float v = fminf(fmaxf(f + .5f, 1.f), 255.5f); const unsigned q = air ? 0u : (unsigned)(int)v;
        __syncwarp();
        s1 = q; s2 = (unsigned long long)q * q;
        io[k] = (uint8_t)q;   /* (air_at read io[k] above: each voxel reads only its own input byte) */
    }
    for (int s = 16; s; s >>= 1) { s1 += __shfl_down_sync(~0u, s1, s); s2 += __shfl_down_sync(~0u, s2, s); }
    if ((threadIdx.x & 31) == 0) { atomicAdd(&usc[1], s1); atomicAdd(&usc[2], s2); }
}
__global__ void aug_sums_k(const uint8_t *in, size_t n, unsigned long long *usc) {   /* the sums of unaugmented bytes */
    const size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    unsigned long long s1 = 0, s2 = 0;
    if (k < n) { s1 = in[k]; s2 = s1 * s1; }
    for (int s = 16; s; s >>= 1) { s1 += __shfl_down_sync(~0u, s1, s); s2 += __shfl_down_sync(~0u, s2, s); }
    if ((threadIdx.x & 31) == 0) { atomicAdd(&usc[1], s1); atomicAdd(&usc[2], s2); }
}
__global__ void aug_norm_k(const unsigned long long *usc, size_t n, const scan_ct_rec *rec, float *scal) {   /* draw()'s z-score */
    if (rec->fixed) { scal[0] = rec->fmean; scal[1] = rec->fisd; return; }
    const double mean = (double)usc[1] / (double)n, var = (double)usc[2] / (double)n - mean * mean, sd = sqrt(var > 0 ? var : 0) + 1e-3;
    scal[0] = (float)mean; scal[1] = (float)(1.0 / sd);
}
static __device__ __forceinline__ float jit_noise(const float *ntab, uint64_t seed, size_t k) { return ntab[sm64(seed + (k + 1) * 0x9e3779b97f4a7c15ull) & 0xffff]; }
static __device__ __forceinline__ void put16(void *H, size_t k, float v, int xfmt) {
    if (xfmt == 1) ((__half *)H)[k] = __float2half_rn(v); else ((__nv_bfloat16 *)H)[k] = __float2bfloat16_rn(v);
}
/* the CT channel through the cube symmetry (write_x_ct) or the warp (write_spatial_tm), z-scored, with the jitter; rec: the radial record
   (nn.h NN_RAD_REC: header, axis, then the warp's tables) */
__global__ void aug_write_k(const uint8_t *in, int P, const float *rad, const scan_ct_rec *rec, const float *scal, const float *ntab, void *H) {
    const size_t P2 = (size_t)P * P, n = P2 * P, k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= n) return;
    const int32_t *h = (const int32_t *)rad; const int sb = h[5], warped = h[6] & 1, affine = h[6] >> 1 & 1;
    const int perm[3] = {sb & 3, sb >> 2 & 3, sb >> 4 & 3}, flip[3] = {sb >> 8 & 1, sb >> 9 & 1, sb >> 10 & 1};
    const int io[3] = {(int)(k / P2), (int)(k / P % P), (int)(k % P)};
    const float mean = scal[0], isd = scal[1];
    float v;
    if (!warped) {
        int s[3]; for (int d = 0; d < 3; d++) s[perm[d]] = flip[d] ? P - 1 - io[d] : io[d];
        v = ((float)in[((size_t)s[0] * P + s[1]) * P + s[2]] - mean) * isd;
    } else {
        const float *tb = rad + 8 + 2 * (size_t)P, *R = tb + 6 * (size_t)P, *A = R + 9; const float sc = R[27];
        const float c = .5f * (P - 1);
        float vt[3]; for (int d = 0; d < 3; d++) vt[d] = io[d] - c - tb[d * P + io[d]];
        float U[3], Vv[3];
        for (int d = 0; d < 3; d++) {
            const float t01 = R[0 * 3 + d] * vt[0] + R[1 * 3 + d] * vt[1];
            U[d] = t01 + R[2 * 3 + d] * vt[2];
        }
        if (affine) { float w[3] = {U[0], U[1], U[2]}; for (int d = 0; d < 3; d++) U[d] = c + sc * (A[d * 3 + 0] * w[0] + A[d * 3 + 1] * w[1] + A[d * 3 + 2] * w[2]); }
        else for (int d = 0; d < 3; d++) U[d] = c + sc * U[d];
        for (int d = 0; d < 3; d++) Vv[perm[d]] = flip[d] ? (P - 1) - U[d] : U[d];
        int inside = 1; for (int d = 0; d < 3; d++) inside &= Vv[d] >= 0 && Vv[d] <= P - 1;
        if (!inside) { put16(H, k, 0.f, rec->xfmt); return; }
        int B[3]; float Fr[3];
        for (int d = 0; d < 3; d++) { int b = (int)Vv[d]; if (b > P - 2) b = P - 2; B[d] = b; Fr[d] = Vv[d] - b; }
        float cv = 0;
        for (int cn = 0; cn < 8; cn++) {
            const int bz = cn >> 2, by = cn >> 1 & 1, bx = cn & 1;
            const float w = ((bz ? Fr[0] : 1 - Fr[0]) * (by ? Fr[1] : 1 - Fr[1])) * (bx ? Fr[2] : 1 - Fr[2]);
            if (!(w > 1e-7f)) continue;
            cv += w * (((float)in[((size_t)(B[0] + bz) * P + B[1] + by) * P + B[2] + bx] - mean) * isd);
        }
        v = cv;
    }
    if (rec->jitter) v = rec->ia * v + rec->ib + rec->isg * jit_noise(ntab, rec->nseed, k);
    put16(H, k, v, rec->xfmt);
}
__global__ void aug_ntab_k(float *t) {   /* sample.c ntab_init: the normal quantiles of (i + .5) / 65536 */
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 65536) t[i] = (float)normcdfinv((i + 0.5) / 65536.0);
}

static aug_dev *aug_get(int P, int ns) {   /* ns: the staging buffer in P^3 bytes (1 the CT, 2 with the labels) */
    int dv = 0; cudaGetDevice(&dv); aug_dev *a = &g_ad[dv & 7];
    const size_t p3 = (size_t)P * P * P, p2 = (size_t)P * P;
    if (a->stage_n < ns * p3) { if (a->stage) { CK(cudaDeviceSynchronize()); cudaFree(a->stage); } CK(cudaMalloc(&a->stage, ns * p3)); a->stage_n = ns * p3; }
    if (a->pl_n < p2) { for (int i = 0; i < 4; i++) { if (a->pl[i]) cudaFree(a->pl[i]); CK(cudaMalloc(&a->pl[i], p2 * sizeof(float))); } a->pl_n = p2; }
    if (!a->plan) {
        CK(cudaMalloc(&a->plan, sizeof(scan_plan))); CK(cudaMalloc(&a->rec, sizeof(scan_ct_rec)));
        CK(cudaMalloc(&a->dsc, 2 * sizeof(double))); CK(cudaMalloc(&a->usc, 4 * sizeof(unsigned long long)));
        CK(cudaMalloc(&a->hcnt, 256 * sizeof(unsigned))); CK(cudaMalloc(&a->hm, 257 * sizeof(float))); CK(cudaMalloc(&a->scal, 4 * sizeof(float)));
        CK(cudaMalloc(&a->ntab, 65536 * sizeof(float))); aug_ntab_k<<<256, 256>>>(a->ntab); KCHECK(); CK(cudaDeviceSynchronize());
        CK(cudaFuncSetAttribute(aug_bop_zy_k, cudaFuncAttributeMaxDynamicSharedMemorySize, SA_MAXP * 32 * 2));
        CK(cudaFuncSetAttribute(aug_exp_x_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
        CK(cudaEventCreateWithFlags(&a->done, cudaEventDisableTiming));
    }
    return a;
}
extern "C" size_t nn_aug_device_bytes(int P) { const size_t p3 = (size_t)P * P * P; return p3 + 4 * (size_t)P * P * 4 + sizeof(scan_plan) + sizeof(scan_ct_rec) + 65536 * 4 + 8192; }
extern "C" size_t nn_label_warp_device_bytes(int P) { return (size_t)P * P * P + 2048; }   /* --gpu-aug 2: the second P^3 of staging, the tables */

/* the scanner ops of rec_h's plan on F (P^3 16-bit, the input bytes loaded), the stream s */
static void aug_scan(aug_dev *a, const scan_plan *p, int P, __half *F, cudaStream_t s) {
    const size_t P2 = (size_t)P * P, p3 = P2 * P; const unsigned on = p->on; const unsigned nb = nblk(p3, 256);
    const uint64_t hp = (uint64_t)(P + 1) / 2; uint64_t q = 0;   /* the noise stream's calls so far */
    scan_plan *pd = a->plan;
    aug_load_k<<<nb, 256, 0, s>>>(a->stage, F, p3, (on & 1u << SA_AIR) != 0, p->air_level);
    if (p->bop_on[0]) aug_bop_zy_k<<<dim3(nblk(P, 32), P), dim3(32, 8), (size_t)P * 32 * 2, s>>>(F, &pd->bop[0], P, 0);
    if (p->bop_on[1]) aug_bop_zy_k<<<dim3(nblk(P, 32), P), dim3(32, 8), (size_t)P * 32 * 2, s>>>(F, &pd->bop[1], P, 1);
    if (p->bop_on[2]) aug_bop_x_k<<<(unsigned)P2, 128, (size_t)P * 4, s>>>(F, &pd->bop[2], P);
    if (on & 1u << SA_BIAS) aug_point_k<<<nb, 256, 0, s>>>(F, pd, P, 0);
    if (on & (1u << SA_RING | 1u << SA_CUPPING)) aug_point_k<<<nb, 256, 0, s>>>(F, pd, P, 1);
    if (on & 1u << SA_SLICE) aug_point_k<<<nb, 256, 0, s>>>(F, pd, P, 2);
    if (on & 1u << SA_NOISE) { aug_noise_k<<<nb, 256, 0, s>>>(F, pd, P, q); q += P2 * hp; }
    if (on & 1u << SA_CNOISE) {
        for (int z = 0; z < P; z++) {
            aug_cnoise_x_k<<<nblk(P, 64), 64, 0, s>>>(pd, P, z, q, a->pl[0]);
            aug_cnoise_yz_k<<<nblk(P, 64), 64, 0, s>>>(F, pd, P, z, a->pl[0], a->pl[1]);
        }
        q += P2 * hp;
    }
    if (on & 1u << SA_PAGANIN) {
        aug_exp_zy_k<<<nblk(P2, 128), 128, 0, s>>>(F, P, 0, p->pag_a);
        aug_exp_zy_k<<<nblk(P2, 128), 128, 0, s>>>(F, P, 1, p->pag_a);
        aug_exp_x_k<<<nblk(P2, 32), 256, (size_t)32 * P * 4, s>>>(F, P, p->pag_a);
    }
    if (on & 1u << SA_GIN) {
        CK(cudaMemsetAsync(a->dsc, 0, 2 * sizeof(double), s)); CK(cudaMemsetAsync(a->usc, 0, sizeof(unsigned long long), s));
        aug_gin_stats_k<<<nb, 256, 0, s>>>(F, a->stage, pd, P, a->dsc, a->usc);
        aug_gin_fin_k<<<1, 1, 0, s>>>(a->dsc, a->usc, a->scal);
        if (p->gin_k == 1) aug_gin1_k<<<nb, 256, 0, s>>>(F, pd, P, a->scal);
        else for (int z = 0; z < P; z++) {
            aug_gin3_g_k<<<nblk(P2, 256), 256, 0, s>>>(F, a->pl[2], pd, P, z, a->pl[3]);
            aug_gin3_b_k<<<nblk(P2, 256), 256, 0, s>>>(F, pd, P, z, a->pl[3], a->pl[2], a->scal);
        }
    }
    for (int i = 0; i < p->ndrop; i++) {
        const int z = p->drop_z[i], mode = p->drop_mode[i];
        aug_plane_k<<<nblk(P2, 256), 256, 0, s>>>(F, pd, P, z, z ? z - 1 : 1, mode, q);
        if (mode) q += (uint64_t)P * hp;
    }
    for (int i = 0; i < p->ncut; i++) {
        const int *b0 = p->cut_b0[i], *b1 = p->cut_b1[i]; const size_t n = (size_t)(b1[0] - b0[0]) * (b1[1] - b0[1]) * (b1[2] - b0[2]);
        if (n) aug_cut_k<<<nblk(n, 256), 256, 0, s>>>(F, pd, P, i, q);
        if (p->cut_noisy[i]) q += (uint64_t)(b1[0] - b0[0]) * (b1[1] - b0[1]) * hp;
    }
    if (p->hist_on) {
        if (p->tone) aug_lut_k<<<nb, 256, 0, s>>>(F, pd->lut, p3);
        CK(cudaMemsetAsync(a->hcnt, 0, 256 * sizeof(unsigned), s)); CK(cudaMemsetAsync(a->usc + 3, 0, sizeof(unsigned long long), s));
        aug_hist_k<<<nblk(nblk(p3, 7), 256), 256, 0, s>>>(F, a->stage, pd, P, a->hcnt, a->usc);
        aug_hist_map_k<<<1, 1, 0, s>>>(a->hcnt, a->usc, pd, a->hm);
        aug_hist_apply_k<<<nb, 256, 0, s>>>(F, a->hm, pd, p3);
    }
    CK(cudaMemsetAsync(a->usc + 1, 0, 2 * sizeof(unsigned long long), s));
    aug_final_k<<<nb, 256, 0, s>>>(F, a->stage, pd, P, a->usc);
}
/* one window: ct8_h (P^3 bytes) and rec_h (pinned host; its plan only when rec_h->scan), rad_d (the device copy of its radial record,
   already uploaded on this stream) into ct16 (P^3 16-bit) */
extern "C" void nn_aug_ct(const uint8_t *ct8_h, const void *rec_h_, const void *rad_d, int P, void *ct16, int on_copy_stream) {
    const scan_ct_rec *rec_h = (const scan_ct_rec *)rec_h_;
    if (P > SA_MAXP || (rec_h->scan && rec_h->plan.P != P)) { fprintf(stderr, "nn_aug_ct: P %d (plan %d, max %d)\n", P, rec_h->plan.P, SA_MAXP); abort(); }
    aug_dev *a = aug_get(P, 1);
    cudaStream_t s = on_copy_stream ? copy_stream() : 0;
    const size_t p3 = (size_t)P * P * P;
    if (a->used) CK(cudaStreamWaitEvent(s, a->done, 0));
    CK(cudaMemcpyAsync(a->stage, ct8_h, p3, cudaMemcpyHostToDevice, s));
    CK(cudaMemcpyAsync(a->rec, rec_h, offsetof(scan_ct_rec, plan), cudaMemcpyHostToDevice, s));
    if (rec_h->scan) {
        CK(cudaMemcpyAsync(a->plan, &rec_h->plan, sizeof(scan_plan), cudaMemcpyHostToDevice, s));
        aug_scan(a, &rec_h->plan, P, (__half *)ct16, s);   /* (the CT channel's buffer as the work volume) */
    } else {
        CK(cudaMemsetAsync(a->usc + 1, 0, 2 * sizeof(unsigned long long), s));
        aug_sums_k<<<nblk(p3, 256), 256, 0, s>>>(a->stage, p3, a->usc);
    }
    aug_norm_k<<<1, 1, 0, s>>>(a->usc, p3, a->rec, a->scal);
    aug_write_k<<<nblk(p3, 256), 256, 0, s>>>(a->stage, P, (const float *)rad_d, a->rec, a->scal, a->ntab, ct16);
    CK(cudaEventRecord(a->done, s)); a->used = 1;
    KCHECK();
}
/* ---- train --gpu-aug 2: the labels of warped windows warped on the device (sample.c write_spatial_tm's packed targets / mask,
   warp_phase_w4's phase and winding mod 4), from the source-grid labels the sampler shipped in the batch's own label buffers. The
   coordinates, weights and sums in the CPU's float operations and order (_rn intrinsics: no contraction under -use_fast_math); the
   phase codes' atan2 in double rounded to float (glibc atan2f is within an ulp: codes may differ at rounding ties) ---- */
enum { LW_PERIOD = 252, LW_UNKNOWN = 255 };
static __device__ __forceinline__ float fm(float a, float b) { return __fmul_rn(a, b); }
static __device__ __forceinline__ float fa(float a, float b) { return __fadd_rn(a, b); }
static __device__ __forceinline__ float fs(float a, float b) { return __fsub_rn(a, b); }
/* the source position of output voxel k (sample.c spatial_aug_pull through the symmetry, as write_spatial_tm's fast path): base corner
   index, fractions; 0 outside */
static __device__ __forceinline__ int lw_coords(const float *rad, int P, size_t k, size_t *b0, float F[3]) {
    const int32_t *h = (const int32_t *)rad; const int sb = h[5], affine = h[6] >> 1 & 1;
    const int perm[3] = {sb & 3, sb >> 2 & 3, sb >> 4 & 3}, flip[3] = {sb >> 8 & 1, sb >> 9 & 1, sb >> 10 & 1};
    const size_t P2 = (size_t)P * P;
    const int io[3] = {(int)(k / P2), (int)(k / P % P), (int)(k % P)};
    const float *tb = rad + 8 + 2 * (size_t)P, *R = tb + 6 * (size_t)P, *A = R + 9; const float sc = R[27];
    const float c = .5f * (P - 1);
    float vt[3]; for (int d = 0; d < 3; d++) vt[d] = fs(fs((float)io[d], c), tb[d * P + io[d]]);
    float U[3], V[3];
    if (affine) {
        float w[3];
        for (int d = 0; d < 3; d++) w[d] = fa(fa(fm(R[d], vt[0]), fm(R[3 + d], vt[1])), fm(R[6 + d], vt[2]));
        for (int d = 0; d < 3; d++) U[d] = fa(c, fm(sc, fa(fa(fm(A[d * 3], w[0]), fm(A[d * 3 + 1], w[1])), fm(A[d * 3 + 2], w[2]))));
    } else for (int d = 0; d < 3; d++) U[d] = fa(c, fm(sc, fa(fa(fm(R[d], vt[0]), fm(R[3 + d], vt[1])), fm(R[6 + d], vt[2]))));
    const float fP1 = (float)(P - 1);
    for (int d = 0; d < 3; d++) V[perm[d]] = flip[d] ? fs(fP1, U[d]) : U[d];
    for (int d = 0; d < 3; d++) if (!(V[d] >= 0.f && V[d] <= fP1)) return 0;
    int B[3];
    for (int d = 0; d < 3; d++) { int b = (int)V[d]; if (b > P - 2) b = P - 2; B[d] = b; F[d] = fs(V[d], (float)b); }
    *b0 = ((size_t)B[0] * P + B[1]) * P + B[2];
    return 1;
}
static __device__ __forceinline__ float lw_w(const float F[3], int cn) {   /* corner cn's weight ((wz * wy) * wx) */
    const int bz = cn >> 2, by = cn >> 1 & 1, bx = cn & 1;
    return fm(fm(bz ? F[0] : fs(1.f, F[0]), by ? F[1] : fs(1.f, F[1])), bx ? F[2] : fs(1.f, F[2]));
}
static __device__ __forceinline__ size_t lw_off(int P, int cn) { return ((size_t)(cn >> 2) * P + (cn >> 1 & 1)) * P + (cn & 1); }
static __device__ __forceinline__ uint8_t lw_code(float sx, float sy) {   /* sample.c warp_code_of */
    float t = fm((float)atan2((double)sy, (double)sx), (float)(LW_PERIOD / (2 * 3.14159265358979323846)));
    if (t < 0) t = fa(t, (float)LW_PERIOD);
    const int q = __float2int_rn(t); return (uint8_t)(q >= LW_PERIOD ? q - LW_PERIOD : q);
}
/* packed targets / mask (SAMPLE_TM_T0 1, T1 2, M 4): the nearest corner's targets (fraction > .5), the mask where every corner of
   weight > 1e-7 has it; 0 outside */
__global__ void lw_tm_k(const uint8_t *src, uint8_t *dst, int P, const float *rad) {
    const size_t n = (size_t)P * P * P, k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= n) return;
    size_t b0; float F[3];
    if (!lw_coords(rad, P, k, &b0, F)) { dst[k] = 0; return; }
    int ok = 1;
    for (int cn = 0; cn < 8; cn++) if (lw_w(F, cn) > 1e-7f && !(src[b0 + lw_off(P, cn)] & 4)) ok = 0;
    const uint8_t t = src[b0 + ((size_t)(F[0] > .5f) * P + (F[1] > .5f)) * P + (F[2] > .5f)] & 3;
    dst[k] = (uint8_t)(t | (ok ? 4 : 0));
}
/* the phase (circular mean of the known corners' codes) */
__global__ void lw_phase_k(const uint8_t *src, uint8_t *dst, int P, const float *rad, const float *cs, const float *sn) {
    const size_t n = (size_t)P * P * P, k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= n) return;
    size_t b0; float F[3]; uint8_t out = LW_UNKNOWN;
    if (lw_coords(rad, P, k, &b0, F)) {
        float sx = 0.f, sy = 0.f, wk = 0.f;
        for (int cn = 0; cn < 8; cn++) {
            const float w = lw_w(F, cn); const int p = src[b0 + lw_off(P, cn)];
            if (!(w > 1e-7f) || p >= LW_PERIOD) continue;
            sx = fa(sx, fm(w, cs[p])); sy = fa(sy, fm(w, sn[p])); wk = fa(wk, w);
        }
        if (!(wk < .5f || fa(fm(sx, sx), fm(sy, sy)) < 1e-8f)) out = lw_code(sx, sy);
    }
    dst[k] = out;
}
/* the winding mod 4 and its component: the nearest corner's component (fraction >= .5), the circular mean over the corners of that
   component; 255 / 255 else */
__global__ void lw_w4_k(const uint8_t *w4, const uint8_t *cp, uint8_t *dw, uint8_t *dc, int P, const float *rad, const float *cs, const float *sn) {
    const size_t n = (size_t)P * P * P, k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= n) return;
    size_t b0; float F[3]; uint8_t ow = 255, oc = 255;
    if (lw_coords(rad, P, k, &b0, F)) {
        const int c0 = cp[b0 + ((size_t)(F[0] >= .5f) * P + (F[1] >= .5f)) * P + (F[2] >= .5f)];
        if (c0 != 255) {
            float sx = 0.f, sy = 0.f, wk = 0.f;
            for (int cn = 0; cn < 8; cn++) {
                const float w = lw_w(F, cn); const size_t j = b0 + lw_off(P, cn); const int q = w4[j];
                if (!(w > 1e-7f) || cp[j] != c0 || q >= LW_PERIOD) continue;
                sx = fa(sx, fm(w, cs[q])); sy = fa(sy, fm(w, sn[q])); wk = fa(wk, w);
            }
            if (!(wk < .5f || fa(fm(sx, sx), fm(sy, sy)) < 1e-8f)) { ow = lw_code(sx, sy); oc = (uint8_t)c0; }
        }
    }
    dw[k] = ow; dc[k] = oc;
}
/* one window's labels of one kind (0 packed targets / mask: a_h -> da; 1 phase: a_h -> da; 2 winding mod 4 and component: a_h, b_h ->
   da, db) from the pinned host source labels, warped by rad_d (the window's radial record, on the device); cs_h / sn_h: the phase
   codes' cos / sin (PHASE_PERIOD floats, sample.c g_wcs / g_wsn). Shares nn_aug_ct's staging buffer (2 P^3 with labels) and its order */
extern "C" void nn_label_warp(int kind, const uint8_t *a_h, const uint8_t *b_h, const void *rad_d, int P, uint8_t *da, uint8_t *db,
                              const float *cs_h, const float *sn_h, int on_copy_stream) {
    aug_dev *a = aug_get(P, 2);
    cudaStream_t s = on_copy_stream ? copy_stream() : 0;
    const size_t p3 = (size_t)P * P * P;
    if (!a->cssn) { CK(cudaMalloc(&a->cssn, 2 * 256 * sizeof(float))); CK(cudaMemcpy(a->cssn, cs_h, LW_PERIOD * sizeof(float), cudaMemcpyHostToDevice));
                    CK(cudaMemcpy(a->cssn + 256, sn_h, LW_PERIOD * sizeof(float), cudaMemcpyHostToDevice)); }
    if (a->used) CK(cudaStreamWaitEvent(s, a->done, 0));
    CK(cudaMemcpyAsync(a->stage, a_h, p3, cudaMemcpyHostToDevice, s));
    if (kind == 2) CK(cudaMemcpyAsync(a->stage + p3, b_h, p3, cudaMemcpyHostToDevice, s));
    const unsigned nb = nblk(p3, 256); const float *rad = (const float *)rad_d;
    if (kind == 0) lw_tm_k<<<nb, 256, 0, s>>>(a->stage, da, P, rad);
    else if (kind == 1) lw_phase_k<<<nb, 256, 0, s>>>(a->stage, da, P, rad, a->cssn, a->cssn + 256);
    else lw_w4_k<<<nb, 256, 0, s>>>(a->stage, a->stage + p3, da, db, P, rad, a->cssn, a->cssn + 256);
    CK(cudaEventRecord(a->done, s)); a->used = 1;
    KCHECK();
}
/* tests: the output bytes of the last nn_aug_ct (device -> host, synchronous) */
extern "C" void nn_aug_last_bytes(uint8_t *out, int P) { int dv = 0; cudaGetDevice(&dv); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(out, g_ad[dv & 7].stage, (size_t)P * P * P, cudaMemcpyDeviceToHost)); }
