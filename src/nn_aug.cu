/* train --gpu-aug: a window's CT channel built on the device from its source-grid bytes (sample.h batch.ct8 / batch.aug):
   the scanner ops of its scan_plan (scan_augment.h; the float ops of scan_aug_exec, here in 16-bit storage in the CT channel's
   own buffer), the output bytes and their z-score, then the cube symmetry or the warp with the intensity jitter (sample.c write_x_ct /
   write_spatial_tm channel 0) into the 16-bit CT channel. The CPU made every decision (the plan, the record); the results match the
   CPU path within float tolerance (tests/test_gpu_aug.c), the zero set exactly. Device memory: one P^3 byte staging buffer, the plan, a
   few P^2 planes. */
#include "band_w4dev.h"
#include "nn_common.cuh"
#include "scan_augment.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <time.h>   /* (clock_gettime: nvcc + g++ got it through other headers, clang does not) */

#define AUG_CN_BYTES ((size_t)64 << 20)   /* the correlated noise's chunk of planes */
typedef struct {
    uint8_t *stage; size_t stage_n;   /* the window's bytes: input, then output */
    scan_plan *plan;                  /* the device copy of the plan */
    scan_ct_rec *rec;                 /* the device copy of the record (without its plan) */
    float *uw; uint16_t *ui; int *nu; /* the x operator's distinct weight rows (aug_bop_uniq_k) */
    float *cn; size_t cn_n;           /* correlated noise: a chunk of planes (aug_cnoise_*_ck) */
    float *pl[4]; size_t pl_n;        /* P^2 planes: correlated-noise rows / state, GIN ring / plane */
    double *dsc;                      /* GIN g1, g2 */
    unsigned long long *usc;          /* GIN gn; output sum, sum of squares; hist count */
    unsigned *hcnt;                   /* hist bins */
    float *hm;                        /* hist map (257) */
    float *scal;                      /* fmean, fisd, gm, gs */
    float *ntab;                      /* the jitter noise table (sample.c g_ntab) */
    float *cssn;                      /* --gpu-aug 2: the phase codes' cos, sin (256 each) */
    cudaEvent_t done; int used;       /* the last window's work (the staging buffer is shared by both streams' calls) */
    void *fw; size_t fw_n;            /* the scanner's work volume when the CT channel's planes do not hold it (narrow z ranges) */
    const void *key; int ks0, ks1;  /* key: the host plane whose source planes [ks0, ks1) the staging buffer holds unchanged */
} aug_dev;
static aug_dev g_ad[16];   /* per device, two slots (nn_zcomm.slot: two ranks on one device in the tests) */
static int lw_keyed(const aug_dev *a, const void *h, int s0, int s1) { return a->key == h && a->ks0 == s0 && a->ks1 == s1; }
static void lw_setkey(aug_dev *a, const void *h, int s0, int s1) { a->key = h; a->ks0 = s0; a->ks1 = s1; }

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

/* voxel k's (z, y, x): 32-bit divisions where the index allows (P <= 1625; a 64-bit division costs several times more per voxel) */
static __device__ __forceinline__ void lw_zyx(size_t k, int P, int io[3]) {
    if (k >> 32) { const size_t P2 = (size_t)P * P; io[0] = (int)(k / P2); io[1] = (int)(k / P % P); io[2] = (int)(k % P); return; }
    const unsigned ku = (unsigned)k, P2 = (unsigned)P * (unsigned)P, z = ku / P2, r = ku - z * P2, y = r / (unsigned)P;
    io[0] = (int)z; io[1] = (int)y; io[2] = (int)(r - y * (unsigned)P);
}
__global__ void aug_load_k(const uint8_t *in, __half *F, size_t n, int air, float level) {
    const size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= n) return;
    const uint8_t v = in[k];
    F[k] = f2h(air && !v ? level : (float)v);
}
/* a separable operator along the contiguous axis: one block per row, the row in shared memory */
__global__ void aug_bop_x_k(__half *F, const scan_bop *b, int P, size_t r0) {   /* (rows from r0 = z0 P) */
    extern __shared__ float sh[];
    __half *row = F + (r0 + blockIdx.x) * P;
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
/* z ranges: axis 1 the planes u0 + blockIdx.y; axis 0 the rows [la, lb) in shared memory, the outputs [oa, ob) (the taps of rows oa..ob-1
   inside [la, lb)); the whole window: u0 0, la oa 0, lb ob P */
__global__ void aug_bop_zy_k(__half *F, const scan_bop *b, int P, int axis, int u0, int la, int lb, int oa, int ob) {
    extern __shared__ __half shh[];
    const int x = blockIdx.x * 32 + threadIdx.x, u = u0 + blockIdx.y;
    const size_t P2 = (size_t)P * P, st = axis == 0 ? P2 : (size_t)P, base = axis == 0 ? (size_t)u * P : (size_t)u * P2;
    if (x < P) for (int i = la + threadIdx.y; i < lb; i += blockDim.y) shh[(i - la) * 32 + threadIdx.x] = F[base + i * st + x];
    __syncthreads();
    if (x >= P) return;
    for (int i = oa + threadIdx.y; i < ob; i += blockDim.y) {
        const float *w = b->w[i]; const int s = b->s[i] - la, n = b->n[i];
        float a = w[0] * h2f(shh[s * 32 + threadIdx.x]);
        for (int t = 1; t < n; t++) a += w[t] * h2f(shh[(s + t) * 32 + threadIdx.x]);
        F[base + i * st + x] = f2h(a);
    }
}
/* the same along axis 0 or 1 with one thread per line (u, x), coalesced across x, and no shared memory (aug_bop_zy_k's P x 32 halves
   are 64 KB of the SM's 100 KB at P 1024: one block of 256 threads per SM). In place through a ring of the last BOP_RING original
   values: position c is read once, before any output at or beyond it is written; bop_ring_ok checks on the host that every row's
   taps are still in the ring. The sums and their order are aug_bop_zy_k's */
#define BOP_RING 64
static int bop_ring_ok(const scan_bop *b, int P) {
    int c = 0;
    for (int i = 0; i < P; i++) {
        const int need = b->s[i] + b->n[i] > i + 1 ? b->s[i] + b->n[i] : i + 1;
        if (b->s[i] < 0 || need > P || b->n[i] < 1) return 0;
        if (need > c) c = need;
        if (c - b->s[i] > BOP_RING) return 0;
    }
    return 1;
}
__global__ void aug_bop_line_k(__half *F, const scan_bop *b, int P, int axis, size_t t0, size_t t1) {   /* (lines [t0, t1): axis 1 planes t0 / P ..) */
    const size_t P2 = (size_t)P * P, t = t0 + blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (t >= t1) return;
    const int u = (int)(t / P), x = (int)(t % P);
    const size_t st = axis == 0 ? P2 : (size_t)P, base = (axis == 0 ? (size_t)u * P : (size_t)u * P2) + x;
    float ring[BOP_RING]; int c = 0;
    for (int i = 0; i < P; i++) {
        const float *w = b->w[i]; const int s = b->s[i], n = b->n[i], need = s + n > i + 1 ? s + n : i + 1;
        for (; c < need; c++) ring[c & (BOP_RING - 1)] = h2f(F[base + c * st]);
        float a = w[0] * ring[s & (BOP_RING - 1)];
        for (int k = 1; k < n; k++) a += w[k] * ring[(s + k) & (BOP_RING - 1)];
        F[base + i * st] = f2h(a);
    }
}
/* along x with R rows per block (R P halves in shared memory, the outputs strided over 256 threads) and the rows' weights from a
   table of the operator's distinct rows (uw[ui[i]], built on the device by aug_bop_uniq_k: a blur's interior rows are one row, so a
   warp's taps load a few cache lines instead of one row of b->w per thread; that cost 8 ms per tap and Gvoxel, against 1 in the z / y
   passes where a warp shares its row). *nu == 0 (more than BOP_UNIQ distinct rows): b->w. The sums and their order are aug_bop_x_k's.
   P 1024, 13 taps: 30 ms against 137 */
#define BOP_UNIQ 64
__global__ void aug_bop_uniq_k(const scan_bop *b, int P, float *uw, uint16_t *ui, int *nu) {
    int U = 0, un[BOP_UNIQ];
    for (int i = 0; i < P; i++) {
        const int n = b->n[i]; int u = U - 1;   /* (the last rows first: neighbours repeat) */
        for (; u >= 0; u--) if (un[u] == n) {
            int k = 0; while (k < n && __float_as_uint(uw[(size_t)u * SA_MAXTAP + k]) == __float_as_uint(b->w[i][k])) k++;
            if (k == n) break;
        }
        if (u < 0) { if (U == BOP_UNIQ) { *nu = 0; return; } u = U++; un[u] = n; for (int k = 0; k < n; k++) uw[(size_t)u * SA_MAXTAP + k] = b->w[i][k]; }
        ui[i] = (uint16_t)u;
    }
    *nu = U;
}
__global__ void aug_bop_xu_k(__half *F, const scan_bop *b, int P, int R, const float *uw, const uint16_t *ui, const int *nu, size_t rb, size_t nrow) {   /* (rows [rb, nrow)) */
    extern __shared__ __half shu[];
    const size_t r0 = rb + (size_t)blockIdx.x * R; __half *blk = F + r0 * P;
    const int n = (int)((r0 + R <= nrow ? R : nrow - r0) * P), tab = *nu > 0;
    for (int j = threadIdx.x; j < n; j += blockDim.x) shu[j] = blk[j];
    __syncthreads();
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        const int i = j % P; const __half *row = shu + (j - i);
        const float *w = tab ? uw + (size_t)ui[i] * SA_MAXTAP : b->w[i]; const int s = b->s[i], nt = b->n[i];
        float a = w[0] * h2f(row[s]);
        for (int t = 1; t < nt; t++) a += w[t] * h2f(row[s + t]);
        blk[j] = f2h(a);
    }
}
static int g_bop_line = 1;   /* tests: 0 = the shared-memory kernel */
extern "C" void nn_aug_bop_line(int on) { g_bop_line = on; }
/* tests / tuning: ms of one pass of a taps-wide operator over a P^3 window: out[0] z shared-memory (0 above its default shared-memory
   limit), [1] z line, [2] y shared, [3] y line, [4] x one row a block, [5] x rows + table */
extern "C" void nn_aug_bop_bench(int P, int taps, double *out) {
    const size_t P2 = (size_t)P * P, p3 = P2 * P; __half *F; scan_bop *bd, *bh = (scan_bop *)calloc(1, sizeof *bh); float *uw; uint16_t *ui; int *nu;
    CK(cudaMalloc(&F, p3 * 2)); CK(cudaMalloc(&bd, sizeof *bd)); CK(cudaMemset(F, 0, p3 * 2));
    CK(cudaMalloc(&uw, BOP_UNIQ * SA_MAXTAP * 4)); CK(cudaMalloc(&ui, SA_MAXP * 2)); CK(cudaMalloc(&nu, 4));
    for (int i = 0; i < P; i++) { int s0 = i - taps / 2, n = taps; if (s0 < 0) { n += s0; s0 = 0; } if (s0 + n > P) n = P - s0; bh->s[i] = s0; bh->n[i] = n; for (int k = 0; k < n; k++) bh->w[i][k] = 1.f / n; }
    bh->P = P; CK(cudaMemcpy(bd, bh, sizeof *bh, cudaMemcpyHostToDevice)); CK(cudaDeviceSynchronize());
    const int R = 16384 / P < 1 ? 1 : 16384 / P;
    for (int m = 0; m < 6; m++) {
        double best = 1e30;
        for (int r = 0; r < 4; r++) {
            CK(cudaDeviceSynchronize()); timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
            if (m == 4) aug_bop_x_k<<<(unsigned)P2, 128, (size_t)P * 4>>>(F, bd, P, 0);
            else if (m == 5) { aug_bop_uniq_k<<<1, 1>>>(bd, P, uw, ui, nu); aug_bop_xu_k<<<nblk(P2, R), 256, (size_t)R * P * 2>>>(F, bd, P, R, uw, ui, nu, 0, P2); }
            else if (m & 1) aug_bop_line_k<<<nblk(P2, 128), 128>>>(F, bd, P, m / 2, 0, P2);
            else if ((size_t)P * 32 * 2 <= 48 * 1024) aug_bop_zy_k<<<dim3(nblk(P, 32), P), dim3(32, 8), (size_t)P * 32 * 2>>>(F, bd, P, m / 2, 0, 0, P, 0, P);
            KCHECK(); CK(cudaDeviceSynchronize()); clock_gettime(CLOCK_MONOTONIC, &t1);
            const double ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) * 1e-6; if (ms < best) best = ms;
        }
        out[m] = best;
    }
    cudaFree(F); cudaFree(bd); cudaFree(uw); cudaFree(ui); cudaFree(nu); free(bh);
}
__global__ void aug_point_k(__half *F, const scan_plan *p, int P, int what, size_t k0, size_t k1) {   /* what: 0 bias, 1 radial, 2 slice */
    const size_t P2 = (size_t)P * P, n = k1, k = k0 + blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= n) return;
    int zyx_[3]; lw_zyx(k, P, zyx_); const int z = zyx_[0], y = zyx_[1], x = zyx_[2];
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
__global__ void aug_noise_k(__half *F, const scan_plan *p, int P, uint64_t q0, size_t k0, size_t k1) {   /* white noise: row r = z P + y from call q0 + r ceil(P / 2) */
    const size_t P2 = (size_t)P * P, n = k1, k = k0 + blockIdx.x * (size_t)blockDim.x + threadIdx.x;
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
static int g_cn_z = -1;   /* planes per chunk (-1: what the chunk buffer holds; 0: the per-plane kernels; tests) */
/* the same in chunks of planes [z0, z0 + nzc) (one kernel pair per plane ran 2 P launches of P threads each: 577 ms of a P 1024 window,
   the card nearly idle): the x recurrences of nzc P rows at once, then y per (plane, column), then z per (y, x) through the chunk with
   the carried plane PP. Each element's operations and their order are the per-plane kernels'. S: nzc P^2 floats */
__global__ void aug_cnoise_x_ck(const scan_plan *p, int P, int z0, int nzc, uint64_t q0, float *S) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nzc * P) return;
    const int zr = t / P, y = t % P, z = z0 + zr;
    const float rho = p->cnoise_rho, scale = sqrtf(1.f - rho * rho); const uint64_t hp = (uint64_t)(P + 1) / 2, qr = q0 + ((uint64_t)z * P + y) * hp;
    float last = 0; float *row = S + ((size_t)zr * P + y) * P;
    for (int x = 0; x < P; x++) { const float g = gauss_at(p->ns, qr, x), vx = x ? rho * last + scale * g : g; last = vx; row[x] = vx; }
}
__global__ void aug_cnoise_y_ck(const scan_plan *p, int P, int nzc, float *S) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nzc * P) return;
    const int zr = t / P, x = t % P;
    const float rho = p->cnoise_rho, scale = sqrtf(1.f - rho * rho);
    float prev = 0; float *col = S + (size_t)zr * P * P + x;
    for (int y = 0; y < P; y++) { const float vx = col[(size_t)y * P], vy = y ? rho * prev + scale * vx : vx; prev = vy; col[(size_t)y * P] = vy; }
}
__global__ void aug_cnoise_z_ck(__half *F, const scan_plan *p, int P, int z0, int nzc, const float *S, float *PP) {
    const size_t P2 = (size_t)P * P, i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= P2) return;
    const float rho = p->cnoise_rho, scale = sqrtf(1.f - rho * rho), s0 = p->cnoise_s0;
    float pz = z0 ? PP[i] : 0.f;
    for (int zr = 0; zr < nzc; zr++) {
        const int z = z0 + zr; const float vy = S[(size_t)zr * P2 + i], vz = z ? rho * pz + scale * vy : vy; pz = vz;
        __half *f = F + (size_t)z * P2 + i; *f = f2h(h2f(*f) + s0 * vz);
    }
    PP[i] = pz;
}
/* paganin's two-sided exponential along axis 0 or 1: one thread per line (u, x), coalesced across x */
__global__ void aug_exp_zy_k(__half *F, int P, int axis, float a, size_t t0, size_t t1) {   /* (lines [t0, t1)) */
    const size_t P2 = (size_t)P * P, t = t0 + blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (t >= t1) return;
    const int u = (int)(t / P), x = (int)(t % P); const float b = 1 - a;
    const size_t st = axis == 0 ? P2 : (size_t)P, base = (axis == 0 ? (size_t)u * P : (size_t)u * P2) + x;
    float y = h2f(F[base]); y = b * y + a * y; F[base] = f2h(y);
    for (int i = 1; i < P; i++) { const size_t j = base + i * st; y = b * h2f(F[j]) + a * y; F[j] = f2h(y); }
    for (int i = P - 2; i >= 0; i--) { const size_t j = base + i * st; y = b * h2f(F[j]) + a * y; F[j] = f2h(y); }
}
/* the same along z over planes [za, zb) of one rank (split windows): dir 0 forward (cin nullptr: plane za is the line's start, its
   y = b v + a v), dir 1 backward from plane zb - 1 down to za; cin / cout: the carried float y per line (P^2) */
__global__ void aug_exp_z_part_k(__half *F, int P, float a, int za, int zb, int dir, const float *cin, float *cout) {
    const size_t P2 = (size_t)P * P, t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (t >= P2) return;
    const float b = 1 - a; float y;
    if (!dir) {
        int i = za;
        if (!cin) { const size_t j = (size_t)za * P2 + t; y = h2f(F[j]); y = b * y + a * y; F[j] = f2h(y); i++; } else y = cin[t];
        for (; i < zb; i++) { const size_t j = (size_t)i * P2 + t; y = b * h2f(F[j]) + a * y; F[j] = f2h(y); }
    } else {
        y = cin[t];
        for (int i = zb - 1; i >= za; i--) { const size_t j = (size_t)i * P2 + t; y = b * h2f(F[j]) + a * y; F[j] = f2h(y); }
    }
    cout[t] = y;
}
/* along the contiguous axis: R rows per block (32, fewer when 32 P floats pass the 96 KB of shared memory: P > 768) in shared memory (as floats in the passes), one thread per row */
__global__ void aug_exp_x_k(__half *F, int P, float a, int R, size_t rb, size_t nrow) {   /* (rows [rb, nrow)) */
    extern __shared__ float shf[];
    const size_t r0 = rb + (size_t)blockIdx.x * R;
    for (int j = threadIdx.x; j < R * P; j += blockDim.x) { const size_t r = r0 + j / P; shf[j] = r < nrow ? h2f(F[r0 * P + j]) : 0.f; }
    __syncthreads();
    if (threadIdx.x < R && r0 + threadIdx.x < nrow) {
        float *v = shf + (size_t)threadIdx.x * P; const float b = 1 - a;
        float y = b * v[0] + a * v[0]; v[0] = y;
        for (int i = 1; i < P; i++) { y = b * v[i] + a * y; v[i] = y; }
        for (int i = P - 2; i >= 0; i--) { y = b * v[i] + a * y; v[i] = y; }
    }
    __syncthreads();
    for (int j = threadIdx.x; j < R * P; j += blockDim.x) { const size_t r = r0 + j / P; if (r < nrow) F[r0 * P + j] = f2h(shf[j]); }
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
__global__ void aug_gin_stats_k(const __half *F, const uint8_t *in, const scan_plan *p, int P, double *dsc, unsigned long long *usc, size_t k0, size_t k1) {
    const size_t P2 = (size_t)P * P, n = k1, k = k0 + blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    float g1 = 0, g2 = 0; unsigned gn = 0;
    if (k < n) {
        int zyx_[3]; lw_zyx(k, P, zyx_); const int z = zyx_[0], y = zyx_[1], x = zyx_[2];
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
__global__ void aug_gin1_k(__half *F, const scan_plan *p, int P, const float *scal, size_t k0, size_t k1) {
    const size_t P2 = (size_t)P * P, n = k1, k = k0 + blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= n) return;
    int zyx_[3]; lw_zyx(k, P, zyx_); const int z = zyx_[0], y = zyx_[1], x = zyx_[2];
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
__global__ void aug_plane_f_k(const __half *F, int P, int z, float *pl) {   /* plane z as floats (gin3's prevp at a rank's first plane) */
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, P2 = (size_t)P * P;
    if (i < P2) pl[i] = h2f(F[(size_t)z * P2 + i]);
}
__global__ void aug_plane_k(__half *F, const scan_plan *p, int P, int z, int src, int mode, uint64_t q0) {   /* dropped slice */
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x, P2 = (size_t)P * P;
    if (i >= P2) return;
    __half *pl = F + (size_t)z * P2;
    if (mode == 0) { pl[i] = F[(size_t)src * P2 + i]; return; }
    const uint64_t hp = (uint64_t)(P + 1) / 2; const int r = (int)(i / P), x = (int)(i % P);
    pl[i] = f2h((float)(p->mean + (mode == 2 ? 1.5 * p->sd * gauss_at(p->ns, q0 + r * hp, x) : 0)));
}
__global__ void aug_cut_k(__half *F, const scan_plan *p, int P, int bi, uint64_t q0, int za, int zb) {   /* cutout box bi (its planes in [za, zb)) */
    const int *b0 = p->cut_b0[bi], *b1 = p->cut_b1[bi], nx = b1[2] - b0[2], ny = b1[1] - b0[1], lo = b0[0] > za ? b0[0] : za, hi = b1[0] < zb ? b1[0] : zb;
    const size_t n = hi > lo ? (size_t)(hi - lo) * ny * nx : 0, t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int dz = lo - b0[0] + (int)(t / ((size_t)ny * nx)), dy = (int)(t / nx % ny), x = b0[2] + (int)(t % nx);
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
__global__ void aug_hist_k(const __half *F, const uint8_t *in, const scan_plan *p, int P, unsigned *cnt, unsigned long long *usc, size_t t0, size_t k1) {   /* (every 7th voxel: k = 7 (t0 + t) < k1) */
    __shared__ unsigned h[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) h[i] = 0;
    __syncthreads();
    const size_t P2 = (size_t)P * P, n = k1, k = 7 * (t0 + blockIdx.x * (size_t)blockDim.x + threadIdx.x);
    if (k < n) {
        int zyx_[3]; lw_zyx(k, P, zyx_); const int z = zyx_[0], y = zyx_[1], x = zyx_[2];
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
__global__ void aug_final_k(const __half *F, uint8_t *io, const scan_plan *p, int P, unsigned long long *usc, size_t k0, size_t k1) {
    const size_t P2 = (size_t)P * P, n = k1, k = k0 + blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    unsigned s1 = 0, s2 = 0;   /* (a block's sums fit 32 bits: 256 voxels of at most 255) */
    __shared__ unsigned sh[2][8];
    if (k < n) {
        int zyx[3]; lw_zyx(k, P, zyx); const int z = zyx[0], y = zyx[1], x = zyx[2];
        const int air = air_at(p, io, k, z, y, x, 1);
        float f = h2f(F[k]);
        if (p->tone && !p->hist_on) { const float v = fminf(fmaxf(f, 0.f), 255.f); const int i = (int)v; f = p->lut[i] + (v - i) * (p->lut[i + 1] - p->lut[i]); }
        if (p->step > 0) f = (floorf(f * (1 / p->step)) + .5f) * p->step;
        const float v = fminf(fmaxf(f + .5f, 1.f), 255.5f); const unsigned q = air ? 0u : (unsigned)(int)v;
        __syncwarp();
        s1 = q; s2 = q * q;
        io[k] = (uint8_t)q;   /* (air_at read io[k] above: each voxel reads only its own input byte) */
    }
    for (int s = 16; s; s >>= 1) { s1 += __shfl_down_sync(~0u, s1, s); s2 += __shfl_down_sync(~0u, s2, s); }
    if ((threadIdx.x & 31) == 0) { sh[0][threadIdx.x >> 5] = s1; sh[1][threadIdx.x >> 5] = s2; }
    __syncthreads();
    if (threadIdx.x == 0) {   /* one pair of atomics a block (was one a warp: 8 M contended 64-bit adds at P1024) */
        unsigned long long a1 = 0, a2 = 0; for (int w = 0; w < (int)(blockDim.x >> 5); w++) { a1 += sh[0][w]; a2 += sh[1][w]; }
        atomicAdd(&usc[1], a1); atomicAdd(&usc[2], a2);
    }
}
/* the sums of unaugmented bytes: AUG_SUM_T bytes a thread (four 64-bit loads; in: cudaMalloc-aligned), one pair of atomics a block
   (the first version, a voxel a thread and a pair of atomics a warp, cost more than the CT channel's write) */
enum { AUG_SUM_T = 32 };
__global__ void aug_sums_k(const uint8_t *in, size_t n, unsigned long long *usc) {
    const size_t k0 = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) * AUG_SUM_T;
    unsigned s1 = 0, s2 = 0;   /* (a block: 8192 bytes, sums < 2^30) */
    __shared__ unsigned sh[2][8];
    if (k0 + AUG_SUM_T <= n) {
#pragma unroll
        for (int j = 0; j < AUG_SUM_T / 8; j++) {
            const unsigned long long w = ((const unsigned long long *)(in + k0))[j];
#pragma unroll
            for (int b = 0; b < 8; b++) { const unsigned v = (unsigned)(w >> (8 * b)) & 255u; s1 += v; s2 += v * v; }
        }
    } else for (size_t k = k0; k < n; k++) { const unsigned v = in[k]; s1 += v; s2 += v * v; }
    for (int s = 16; s; s >>= 1) { s1 += __shfl_down_sync(~0u, s1, s); s2 += __shfl_down_sync(~0u, s2, s); }
    if ((threadIdx.x & 31) == 0) { sh[0][threadIdx.x >> 5] = s1; sh[1][threadIdx.x >> 5] = s2; }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long a1 = 0, a2 = 0; for (int w = 0; w < (int)(blockDim.x >> 5); w++) { a1 += sh[0][w]; a2 += sh[1][w]; }
        atomicAdd(&usc[1], a1); atomicAdd(&usc[2], a2);
    }
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
__global__ void aug_write_k(const uint8_t *in, int P, const float *rad, const scan_ct_rec *rec, const float *scal, const float *ntab, void *H, size_t k0, size_t k1) {
    const size_t P2 = (size_t)P * P, n = k1, k = k0 + blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= n) return;
    const int32_t *h = (const int32_t *)rad; const int sb = h[5], warped = h[6] & 1, affine = h[6] >> 1 & 1;
    const int perm[3] = {sb & 3, sb >> 2 & 3, sb >> 4 & 3}, flip[3] = {sb >> 8 & 1, sb >> 9 & 1, sb >> 10 & 1};
    int io[3]; lw_zyx(k, P, io);
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

/* the staging buffer's reuse between the calls of one batch upload: aug_dev.key is the pinned host plane whose bytes it holds unchanged
   (nullptr: none). train calls nn_aug_stage_reset before each batch's uploads (the sampler reuses its host planes for later batches) */
static int g_lw_ref = 0;
extern "C" void nn_label_warp_ref(int on) { g_lw_ref = on; }   /* tests: the reference kernels (double atan2 everywhere, the ink plane written first, no reuse) */
extern "C" void nn_aug_stage_reset(void) { int dv = 0; cudaGetDevice(&dv); g_ad[(dv & 7) * 2].key = g_ad[(dv & 7) * 2 + 1].key = nullptr; }
static aug_dev *aug_get(int P, size_t sb, int slot = 0) {   /* sb: the staging buffer's bytes (P^3 the CT, 2 P^3 with the labels; z ranges: 2 (s1 - s0) P^2) */
    int dv = 0; cudaGetDevice(&dv); aug_dev *a = &g_ad[(dv & 7) * 2 + (slot & 1)];
    const size_t p2 = (size_t)P * P;
    if (a->stage_n < sb) { a->key = nullptr; if (a->stage) { CK(cudaDeviceSynchronize()); cudaFree(a->stage); } CK(cudaMalloc(&a->stage, sb)); a->stage_n = sb; }
    if (a->pl_n < p2) { for (int i = 0; i < 4; i++) { if (a->pl[i]) cudaFree(a->pl[i]); CK(cudaMalloc(&a->pl[i], p2 * sizeof(float))); } a->pl_n = p2; }
    { const size_t zc = AUG_CN_BYTES / (p2 * sizeof(float)) < 1 ? 1 : AUG_CN_BYTES / (p2 * sizeof(float)), want = (zc < (size_t)P ? zc : (size_t)P) * p2;
      if (a->cn_n < want) { if (a->cn) { CK(cudaDeviceSynchronize()); cudaFree(a->cn); } CK(cudaMalloc(&a->cn, want * sizeof(float))); a->cn_n = want; } }
    if (!a->plan) {
        CK(cudaMalloc(&a->plan, sizeof(scan_plan))); CK(cudaMalloc(&a->rec, sizeof(scan_ct_rec)));
        CK(cudaMalloc(&a->uw, 64 * SA_MAXTAP * sizeof(float))); CK(cudaMalloc(&a->ui, SA_MAXP * sizeof(uint16_t))); CK(cudaMalloc(&a->nu, sizeof(int)));
        CK(cudaMalloc(&a->dsc, 2 * sizeof(double))); CK(cudaMalloc(&a->usc, 4 * sizeof(unsigned long long)));
        CK(cudaMalloc(&a->hcnt, 256 * sizeof(unsigned))); CK(cudaMalloc(&a->hm, 257 * sizeof(float))); CK(cudaMalloc(&a->scal, 4 * sizeof(float)));
        CK(cudaMalloc(&a->ntab, 65536 * sizeof(float))); aug_ntab_k<<<256, 256>>>(a->ntab); KCHECK(); CK(cudaDeviceSynchronize());
        CK(cudaFuncSetAttribute(aug_bop_zy_k, cudaFuncAttributeMaxDynamicSharedMemorySize, SA_MAXP * 32 * 2));
        CK(cudaFuncSetAttribute(aug_exp_x_k, cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
        CK(cudaEventCreateWithFlags(&a->done, cudaEventDisableTiming));
    }
    return a;
}
extern "C" void nn_aug_cnoise_chunk(int nz) { g_cn_z = nz; }
extern "C" size_t nn_aug_device_bytes(int P) { const size_t p3 = (size_t)P * P * P, p2 = (size_t)P * P, zc = AUG_CN_BYTES / (p2 * 4) < 1 ? 1 : AUG_CN_BYTES / (p2 * 4); return p3 + 4 * p2 * 4 + (zc < (size_t)P ? zc : (size_t)P) * p2 * 4 + sizeof(scan_plan) + sizeof(scan_ct_rec) + 65536 * 4 + 8192; }
extern "C" size_t nn_label_warp_device_bytes(int P) { return (size_t)P * P * P + 2048; }   /* --gpu-aug 2: the second P^3 of staging, the tables */
/* nn_aug_ct_z: the staging of ns source planes, the work volume when a rank's planes plus 2 halo planes pass its nz output planes */
extern "C" size_t nn_aug_device_bytes_z(int P, int ns, int nz, int halo) {
    const size_t p2 = (size_t)P * P, w = (size_t)(P / 2 + 2 * halo);
    return nn_aug_device_bytes(P) - p2 * P + (size_t)ns * p2 + (w > (size_t)nz ? w * p2 * 2 : 0);
}
/* the z-range builders' staging: 2 ns P^2 (ns: the most source planes s1 - s0 of a call) and the tables */
extern "C" size_t nn_label_warp_device_bytes_z(int P, int ns) { return 2 * (size_t)ns * P * P + 2048; }

/* ---- the scanner ops of a plan on F (16-bit, the input bytes loaded) over the source planes [oa, ob) of the window, stream s. F and
   in (the staged bytes) are virtual whole-window pointers: valid for the planes the caller staged (owned planes plus the halo h on F).
   zc (split windows, nn.h nn_zcomm): rank r of 2 owns [oa, ob); the stencils' halo planes, the recursions' carries (correlated noise
   0 -> 1; paganin forward 0 -> 1, backward 1 -> 0) and the statistics (GIN, histogram, the output sums) cross the ranks, so the bytes
   of the owned planes are the whole-window run's. Every collective is decided by the shared plan: both ranks call them in one order ---- */
static void zc_sum(const nn_zcomm *zc, void *d, int n, int type, cudaStream_t s) { if (zc && zc->nrank > 1) zc->sum(zc->ctx, d, n, type, (void *)s); }
static void zc_halo(const nn_zcomm *zc, __half *F, int P, int oa, int ob, int h, cudaStream_t s) {   /* the halo planes from the peer's owned planes */
    if (!zc || zc->nrank < 2 || h <= 0) return;
    const size_t P2 = (size_t)P * P, nb = (size_t)h * P2 * 2;
    if (zc->rank == 0) zc->xchg(zc->ctx, 1, F + (size_t)(ob - h) * P2, nb, F + (size_t)ob * P2, nb, (void *)s);
    else zc->xchg(zc->ctx, 0, F + (size_t)oa * P2, nb, F + (size_t)(oa - h) * P2, nb, (void *)s);
}
/* the halo of a plan: the planes beyond a rank's own that its stencils read (the z blur's taps, GIN's k^3 neighbourhood, a dropped
   slice's source); the same for every rank */
static int aug_plan_halo(const scan_plan *p, int P) {
    int h = 0;
    if (p->bop_on[0]) for (int i = 0; i < P; i++) { const int lo = i - p->bop[0].s[i], hi = p->bop[0].s[i] + p->bop[0].n[i] - 1 - i; if (lo > h) h = lo; if (hi > h) h = hi; }
    if ((p->on & 1u << SA_GIN) && p->gin_k / 2 > h) h = p->gin_k / 2;
    if (p->ndrop && h < 1) h = 1;
    return h;
}
static int zc_owner(int P, int nrank, int z) { return nrank > 1 && z >= P / 2 ? 1 : 0; }
static void aug_scan(aug_dev *a, const scan_plan *p, int P, __half *F, uint8_t *in, int oa, int ob, int h, const nn_zcomm *zc, cudaStream_t s) {
    const size_t P2 = (size_t)P * P, k0 = (size_t)oa * P2, k1 = (size_t)ob * P2, n = k1 - k0; const unsigned on = p->on; const unsigned nb = nblk(n, 256);
    const int split = zc && zc->nrank > 1, rk = split ? zc->rank : 0, nr = split ? 2 : 1;
    const uint64_t hp = (uint64_t)(P + 1) / 2; uint64_t q = 0;   /* the noise stream's calls so far */
    scan_plan *pd = a->plan;
    aug_load_k<<<nb, 256, 0, s>>>(in + k0, F + k0, n, (on & 1u << SA_AIR) != 0, p->air_level);
    for (int ax = 0; ax < 2; ax++) if (p->bop_on[ax]) {
        if (ax == 0 && split) {   /* z: the taps of the owned rows from the halo */
            zc_halo(zc, F, P, oa, ob, h, s);
            int la = P, lb = 0; for (int i = oa; i < ob; i++) { const int s0 = p->bop[0].s[i], s1 = s0 + p->bop[0].n[i]; if (s0 < la) la = s0; if (s1 > lb) lb = s1; }
            aug_bop_zy_k<<<dim3(nblk(P, 32), P), dim3(32, 8), (size_t)(lb - la) * 32 * 2, s>>>(F, &pd->bop[0], P, 0, 0, la, lb, oa, ob);
        } else if (g_bop_line && bop_ring_ok(&p->bop[ax], P)) aug_bop_line_k<<<nblk(ax == 0 ? P2 : n / P, 128), 128, 0, s>>>(F, &pd->bop[ax], P, ax, ax == 0 ? 0 : k0 / P, ax == 0 ? P2 : k1 / P);
        else if (ax == 0) aug_bop_zy_k<<<dim3(nblk(P, 32), P), dim3(32, 8), (size_t)P * 32 * 2, s>>>(F, &pd->bop[0], P, 0, 0, 0, P, 0, P);
        else aug_bop_zy_k<<<dim3(nblk(P, 32), ob - oa), dim3(32, 8), (size_t)P * 32 * 2, s>>>(F, &pd->bop[1], P, 1, oa, 0, P, 0, P);
    }
    if (p->bop_on[2]) {
        const int R = 16384 / P < 1 ? 1 : 16384 / P;   /* 32 KB of halves */
        const size_t rb = k0 / P, re = k1 / P;
        if (g_bop_line) { aug_bop_uniq_k<<<1, 1, 0, s>>>(&pd->bop[2], P, a->uw, a->ui, a->nu); aug_bop_xu_k<<<nblk(re - rb, R), 256, (size_t)R * P * 2, s>>>(F, &pd->bop[2], P, R, a->uw, a->ui, a->nu, rb, re); }
        else aug_bop_x_k<<<(unsigned)(re - rb), 128, (size_t)P * 4, s>>>(F, &pd->bop[2], P, rb);
    }
    nn_ptick_fine("      y load + blur ops");
    if (on & 1u << SA_BIAS) aug_point_k<<<nb, 256, 0, s>>>(F, pd, P, 0, k0, k1);
    if (on & (1u << SA_RING | 1u << SA_CUPPING)) aug_point_k<<<nb, 256, 0, s>>>(F, pd, P, 1, k0, k1);
    if (on & 1u << SA_SLICE) aug_point_k<<<nb, 256, 0, s>>>(F, pd, P, 2, k0, k1);
    nn_ptick_fine("      y point ops");
    if (on & 1u << SA_NOISE) { aug_noise_k<<<nb, 256, 0, s>>>(F, pd, P, q, k0, k1); q += P2 * hp; }
    nn_ptick_fine("      y noise");
    if (on & 1u << SA_CNOISE) {
        const int zmax = (int)(a->cn_n / P2), zc_ = g_cn_z < 0 || g_cn_z > zmax ? zmax : g_cn_z;
        if (split && zc_ < 1) { fprintf(stderr, "nn_aug_ct_z: split windows need the chunked correlated noise\n"); abort(); }
        if (split && rk == 1) zc->xchg(zc->ctx, 0, nullptr, 0, a->pl[1], P2 * sizeof(float), (void *)s);   /* (the carried plane of plane oa - 1) */
        if (zc_ >= 1) for (int z = oa; z < ob; z += zc_) {
            const int nzc = z + zc_ < ob ? zc_ : ob - z;
            aug_cnoise_x_ck<<<nblk((size_t)nzc * P, 64), 64, 0, s>>>(pd, P, z, nzc, q, a->cn);
            aug_cnoise_y_ck<<<nblk((size_t)nzc * P, 64), 64, 0, s>>>(pd, P, nzc, a->cn);
            aug_cnoise_z_ck<<<nblk(P2, 256), 256, 0, s>>>(F, pd, P, z, nzc, a->cn, a->pl[1]);
        }
        else for (int z = 0; z < P; z++) {
            aug_cnoise_x_k<<<nblk(P, 64), 64, 0, s>>>(pd, P, z, q, a->pl[0]);
            aug_cnoise_yz_k<<<nblk(P, 64), 64, 0, s>>>(F, pd, P, z, a->pl[0], a->pl[1]);
        }
        if (split && rk == 0) zc->xchg(zc->ctx, 1, a->pl[1], P2 * sizeof(float), nullptr, 0, (void *)s);
        q += P2 * hp;
    }
    nn_ptick_fine("      y correlated noise");
    if (on & 1u << SA_PAGANIN) {
        if (split) {   /* z: forward 0 -> 1, then 1 runs its backward from P - 2 and hands the carry to 0 (a->pl[0]: the carried y per line) */
            float *cy = a->pl[0]; const unsigned nl = nblk(P2, 128);
            if (rk == 0) {
                aug_exp_z_part_k<<<nl, 128, 0, s>>>(F, P, p->pag_a, 0, ob, 0, nullptr, cy);
                zc->xchg(zc->ctx, 1, cy, P2 * sizeof(float), nullptr, 0, (void *)s);
                zc->xchg(zc->ctx, 1, nullptr, 0, cy, P2 * sizeof(float), (void *)s);
                aug_exp_z_part_k<<<nl, 128, 0, s>>>(F, P, p->pag_a, 0, ob, 1, cy, cy);
            } else {
                zc->xchg(zc->ctx, 0, nullptr, 0, cy, P2 * sizeof(float), (void *)s);
                aug_exp_z_part_k<<<nl, 128, 0, s>>>(F, P, p->pag_a, oa, ob, 0, cy, cy);
                aug_exp_z_part_k<<<nl, 128, 0, s>>>(F, P, p->pag_a, oa, ob - 1, 1, cy, cy);   /* (ob = P: the backward pass starts at P - 2) */
                zc->xchg(zc->ctx, 0, cy, P2 * sizeof(float), nullptr, 0, (void *)s);
            }
        } else aug_exp_zy_k<<<nblk(P2, 128), 128, 0, s>>>(F, P, 0, p->pag_a, 0, P2);
        aug_exp_zy_k<<<nblk(n / P, 128), 128, 0, s>>>(F, P, 1, p->pag_a, k0 / P, k1 / P);
        const int R = 32 * P * 4 <= 96 * 1024 ? 32 : 96 * 1024 / (P * 4);
        aug_exp_x_k<<<nblk(n / P, R), 256, (size_t)R * P * 4, s>>>(F, P, p->pag_a, R, k0 / P, k1 / P);
    }
    nn_ptick_fine("      y paganin");
    if (on & 1u << SA_GIN) {
        zc_halo(zc, F, P, oa, ob, h, s);   /* (the neighbourhood's planes before GIN) */
        CK(cudaMemsetAsync(a->dsc, 0, 2 * sizeof(double), s)); CK(cudaMemsetAsync(a->usc, 0, sizeof(unsigned long long), s));
        aug_gin_stats_k<<<nb, 256, 0, s>>>(F, in, pd, P, a->dsc, a->usc, k0, k1);
        zc_sum(zc, a->dsc, 2, 1, s); zc_sum(zc, a->usc, 1, 0, s);
        aug_gin_fin_k<<<1, 1, 0, s>>>(a->dsc, a->usc, a->scal);
        if (p->gin_k == 1) aug_gin1_k<<<nb, 256, 0, s>>>(F, pd, P, a->scal, k0, k1);
        else for (int z = oa; z < ob; z++) {
            if (z == oa && z > 0) aug_plane_f_k<<<nblk(P2, 256), 256, 0, s>>>(F, P, z - 1, a->pl[2]);   /* (plane oa - 1 as it was before GIN: the halo) */
            aug_gin3_g_k<<<nblk(P2, 256), 256, 0, s>>>(F, a->pl[2], pd, P, z, a->pl[3]);
            aug_gin3_b_k<<<nblk(P2, 256), 256, 0, s>>>(F, pd, P, z, a->pl[3], a->pl[2], a->scal);
        }
    }
    nn_ptick_fine("      y gin");
    for (int i = 0; i < p->ndrop; i++) {
        const int z = p->drop_z[i], mode = p->drop_mode[i], src = z ? z - 1 : 1;
        if (split && mode == 0 && zc_owner(P, nr, z) != zc_owner(P, nr, src)) zc_halo(zc, F, P, oa, ob, h, s);   /* (the source plane is the peer's: as it is now) */
        if (z >= oa && z < ob) aug_plane_k<<<nblk(P2, 256), 256, 0, s>>>(F, pd, P, z, src, mode, q);
        if (mode) q += (uint64_t)P * hp;
    }
    for (int i = 0; i < p->ncut; i++) {
        const int *b0 = p->cut_b0[i], *b1 = p->cut_b1[i];
        const int lo = b0[0] > oa ? b0[0] : oa, hi = b1[0] < ob ? b1[0] : ob; const size_t nc = hi > lo ? (size_t)(hi - lo) * (b1[1] - b0[1]) * (b1[2] - b0[2]) : 0;
        if (nc) aug_cut_k<<<nblk(nc, 256), 256, 0, s>>>(F, pd, P, i, q, oa, ob);
        if (p->cut_noisy[i]) q += (uint64_t)(b1[0] - b0[0]) * (b1[1] - b0[1]) * hp;
    }
    nn_ptick_fine("      y planes + cuts");
    if (p->hist_on) {
        if (p->tone) aug_lut_k<<<nb, 256, 0, s>>>(F + k0, pd->lut, n);
        CK(cudaMemsetAsync(a->hcnt, 0, 256 * sizeof(unsigned), s)); CK(cudaMemsetAsync(a->usc + 3, 0, sizeof(unsigned long long), s));
        const size_t t0 = (k0 + 6) / 7, nt = k1 > 7 * t0 ? (k1 - 7 * t0 + 6) / 7 : 0;
        if (nt) aug_hist_k<<<nblk(nt, 256), 256, 0, s>>>(F, in, pd, P, a->hcnt, a->usc, t0, k1);
        zc_sum(zc, a->hcnt, 256, 2, s); zc_sum(zc, a->usc + 3, 1, 0, s);
        aug_hist_map_k<<<1, 1, 0, s>>>(a->hcnt, a->usc, pd, a->hm);
        aug_hist_apply_k<<<nb, 256, 0, s>>>(F + k0, a->hm, pd, n);
    }
    nn_ptick_fine("      y tone + histogram");
    CK(cudaMemsetAsync(a->usc + 1, 0, 2 * sizeof(unsigned long long), s));
    aug_final_k<<<nb, 256, 0, s>>>(F, in, pd, P, a->usc, k0, k1);
    zc_sum(zc, a->usc + 1, 2, 0, s);
}
/* one window: ct8_h (P^3 bytes) and rec_h (pinned host; its plan only when rec_h->scan), rad_h / rad_d (its radial record on the host /
   on the device, already uploaded on this stream) into ct16: output planes [z0, z0 + nz) (ct16 holds those planes; the whole window: 0,
   P). The source planes staged: the whole window without zc; with zc (split windows) the rank's own planes [oa, ob) (rank r of 2: the
   low / high half) and the warp's bound for [z0, z0 + nz) (nn_aug_src_range), the scanned bytes of the peer's planes in it exchanged */
extern "C" void nn_aug_ct_z(const uint8_t *ct8_h, const void *rec_h_, const void *rad_h, const void *rad_d, int P, int z0, int nz, void *ct16, const nn_zcomm *zc, int on_copy_stream) {
    const scan_ct_rec *rec_h = (const scan_ct_rec *)rec_h_;
    if (P > SA_MAXP || (rec_h->scan && rec_h->plan.P != P) || z0 < 0 || nz < 1 || z0 + nz > P) { fprintf(stderr, "nn_aug_ct: P %d (plan %d, max %d), z range [%d, %d)\n", P, rec_h->plan.P, SA_MAXP, z0, z0 + nz); abort(); }
    const int split = zc && zc->nrank > 1, rk = split ? zc->rank : 0;
    if (zc && (zc->nrank > 2 || (split && P % 4))) { fprintf(stderr, "nn_aug_ct_z: 2 ranks and P %% 4 == 0\n"); abort(); }
    const size_t P2 = (size_t)P * P, p3 = P2 * P;
    const int oa = split ? rk * (P / 2) : 0, ob = split ? (rk + 1) * (P / 2) : P;
    int s0 = 0, s1 = P; if (split) nn_aug_src_range(rad_h, P, z0, nz, &s0, &s1);
    const int ta = s0 < oa ? s0 : oa, tb = s1 > ob ? s1 : ob;   /* the staged source planes */
    const int h = split && rec_h->scan ? aug_plan_halo(&rec_h->plan, P) : 0, fa = oa - h < 0 ? 0 : oa - h, fb = ob + h > P ? P : ob + h;
    aug_dev *a = aug_get(P, (size_t)(tb - ta) * P2, zc ? zc->slot : 0);
    cudaStream_t s = zc && zc->stream ? (cudaStream_t)zc->stream : on_copy_stream ? copy_stream() : 0;
    if (a->used) CK(cudaStreamWaitEvent(s, a->done, 0));
    if (!lw_keyed(a, ct8_h, ta, tb)) CK(cudaMemcpyAsync(a->stage, ct8_h + (size_t)ta * P2, (size_t)(tb - ta) * P2, cudaMemcpyHostToDevice, s));   /* (else: staged by nn_ink_bits just before) */
    a->key = nullptr;   /* (the scanner operators write their bytes back into the staging buffer) */
    uint8_t *in = a->stage - (size_t)ta * P2;   /* (virtual whole-window pointers: valid on the staged planes) */
    CK(cudaMemcpyAsync(a->rec, rec_h, offsetof(scan_ct_rec, plan), cudaMemcpyHostToDevice, s));
    nn_ptick_fine("    x ct bytes h2d");
    const size_t xb = 2;   /* (a half / bfloat16 channel) */
    if (rec_h->scan) {
        CK(cudaMemcpyAsync(a->plan, &rec_h->plan, sizeof(scan_plan), cudaMemcpyHostToDevice, s));
        __half *F;
        if (fb - fa <= nz) F = (__half *)ct16 - (size_t)fa * P2;   /* (the CT channel's buffer as the work volume, as before) */
        else {   /* (a range narrower than the work planes: its own buffer) */
            const size_t need = (size_t)(fb - fa) * P2 * 2;
            if (a->fw_n < need) { if (a->fw) { CK(cudaStreamSynchronize(s)); cudaFree(a->fw); } CK(cudaMalloc(&a->fw, need)); a->fw_n = need; }
            F = (__half *)a->fw - (size_t)fa * P2;
        }
        aug_scan(a, &rec_h->plan, P, F, in, oa, ob, h, zc, s);
        if (split) {   /* the peer's scanned bytes of the planes this rank's warp reads (and ours of the peer's: its bound from its range) */
            int ps0, ps1; nn_aug_src_range(rad_h, P, zc->peer_z0, zc->peer_nz, &ps0, &ps1);
            const int poa = (1 - rk) * (P / 2), pob = poa + P / 2;
            const int ga = s0 > poa ? s0 : poa, gb = s1 < pob ? s1 : pob, sa = ps0 > oa ? ps0 : oa, sb = ps1 < ob ? ps1 : ob;
            zc->xchg(zc->ctx, 1 - rk, sb > sa ? in + (size_t)sa * P2 : nullptr, sb > sa ? (size_t)(sb - sa) * P2 : 0, gb > ga ? in + (size_t)ga * P2 : nullptr, gb > ga ? (size_t)(gb - ga) * P2 : 0, (void *)s);
        }
        nn_ptick_fine("    x ct scan aug");
    } else {
        CK(cudaMemsetAsync(a->usc + 1, 0, 2 * sizeof(unsigned long long), s));
        const size_t n = (size_t)(ob - oa) * P2;
        aug_sums_k<<<nblk(nblk(n, AUG_SUM_T), 256), 256, 0, s>>>(in + (size_t)oa * P2, n, a->usc);
        zc_sum(zc, a->usc + 1, 2, 0, s);
    }
    aug_norm_k<<<1, 1, 0, s>>>(a->usc, p3, a->rec, a->scal);
    void *H = (char *)ct16 - (size_t)z0 * P2 * xb;
    aug_write_k<<<nblk((size_t)nz * P2, 256), 256, 0, s>>>(in, P, (const float *)rad_d, a->rec, a->scal, a->ntab, H, (size_t)z0 * P2, (size_t)(z0 + nz) * P2);
    nn_ptick_fine("    x ct sums + write");
    CK(cudaEventRecord(a->done, s)); a->used = 1;
    KCHECK();
}
extern "C" void nn_aug_ct(const uint8_t *ct8_h, const void *rec_h_, const void *rad_d, int P, void *ct16, int on_copy_stream) {
    nn_aug_ct_z(ct8_h, rec_h_, nullptr, rad_d, P, 0, P, ct16, nullptr, on_copy_stream);
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
    int io[3]; lw_zyx(k, P, io);
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
/* ref 0: the single-precision atan2 decides wherever its t is more than 2e-3 from a rounding boundary (its error in t is < 1e-4: a few
   ulp of pi times 40, plus the product's); the double one, 1 voxel in 250, only at the boundaries. The same codes as ref 1 (tests), at
   a fraction of the time: double precision runs at 1 / 64 of the single rate on these cards */
template <int REF> static __device__ __forceinline__ uint8_t lw_code(float sx, float sy) {   /* sample.c warp_code_of */
    if (!REF) {
        float tf = atan2f(sy, sx) * (float)(LW_PERIOD / (2 * 3.14159265358979323846));
        if (tf < 0) tf += (float)LW_PERIOD;
        const float fr = tf - floorf(tf);
        if (fabsf(fr - .5f) > 2e-3f && tf == tf) { const int q = (int)(tf + .5f); return (uint8_t)(q >= LW_PERIOD ? q - LW_PERIOD : q); }
    }
    float t = fm((float)atan2((double)sy, (double)sx), (float)(LW_PERIOD / (2 * 3.14159265358979323846)));
    if (t < 0) t = fa(t, (float)LW_PERIOD);
    const int q = __float2int_rn(t); return (uint8_t)(q >= LW_PERIOD ? q - LW_PERIOD : q);
}
/* packed targets / mask (SAMPLE_TM_T0 1, T1 2, M 4): the nearest corner's targets (fraction > .5), the mask where every corner of
   weight > 1e-7 has it; 0 outside. pm (optional): the mask bit as 0 / 1 (sample.c batch.pm of a warped window) */
/* z ranges (all label kernels): output planes [z0, z0 + nz) as k0 = z0 P^2, n = nz P^2 (dst / pm hold those planes only), the source
   planes from s0 on as so = s0 P^2 (the staging holds [s0, s1): nn_aug_src_range's bound); the whole window: 0, P^3, 0 */
__global__ void lw_tm_k(const uint8_t *src, uint8_t *dst, uint8_t *pm, int P, const float *rad, size_t k0, size_t n, size_t so) {
    const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x, k = k0 + t;
    if (t >= n) return;
    dst += t; if (pm) pm += t;
    size_t b0; float F[3];
    if (!lw_coords(rad, P, k, &b0, F)) { *dst = 0; if (pm) *pm = 0; return; }
    b0 -= so;
    int ok = 1;
    for (int cn = 0; cn < 8; cn++) if (lw_w(F, cn) > 1e-7f && !(src[b0 + lw_off(P, cn)] & 4)) ok = 0;
    const uint8_t tt = src[b0 + ((size_t)(F[0] > .5f) * P + (F[1] > .5f)) * P + (F[2] > .5f)] & 3;
    *dst = (uint8_t)(tt | (ok ? 4 : 0)); if (pm) *pm = (uint8_t)ok;
}
/* the phase (circular mean of the known corners' codes) */
template <int REF> __global__ void lw_phase_k(const uint8_t *src, uint8_t *dst, int P, const float *rad, const float *cs, const float *sn, size_t k0, size_t n, size_t so) {
    const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x, k = k0 + t;
    if (t >= n) return;
    size_t b0; float F[3]; uint8_t out = LW_UNKNOWN;
    if (lw_coords(rad, P, k, &b0, F)) {
        b0 -= so;
        float sx = 0.f, sy = 0.f, wk = 0.f;
        for (int cn = 0; cn < 8; cn++) {
            const float w = lw_w(F, cn); const int p = src[b0 + lw_off(P, cn)];
            if (!(w > 1e-7f) || p >= LW_PERIOD) continue;
            sx = fa(sx, fm(w, cs[p])); sy = fa(sy, fm(w, sn[p])); wk = fa(wk, w);
        }
        if (!(wk < .5f || fa(fm(sx, sx), fm(sy, sy)) < 1e-8f)) out = lw_code<REF>(sx, sy);
    }
    dst[t] = out;
}
/* the winding mod 4 and its component: the nearest corner's component (fraction >= .5), the circular mean over the corners of that
   component; 255 / 255 else */
template <int REF> __global__ void lw_w4_k(const uint8_t *w4, const uint8_t *cp, uint8_t *dw, uint8_t *dc, int P, const float *rad, const float *cs, const float *sn,
                                            size_t k0, size_t n, size_t so) {
    const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x, k = k0 + t;
    if (t >= n) return;
    size_t b0; float F[3]; uint8_t ow = 255, oc = 255;
    if (lw_coords(rad, P, k, &b0, F)) {
        b0 -= so;
        const int c0 = cp[b0 + ((size_t)(F[0] >= .5f) * P + (F[1] >= .5f)) * P + (F[2] >= .5f)];
        if (c0 != 255) {
            float sx = 0.f, sy = 0.f, wk = 0.f;
            for (int cn = 0; cn < 8; cn++) {
                const float w = lw_w(F, cn); const size_t j = b0 + lw_off(P, cn); const int q = w4[j];
                if (!(w > 1e-7f) || cp[j] != c0 || q >= LW_PERIOD) continue;
                sx = fa(sx, fm(w, cs[q])); sy = fa(sy, fm(w, sn[q])); wk = fa(wk, w);
            }
            if (!(wk < .5f || fa(fm(sx, sx), fm(sy, sy)) < 1e-8f)) { ow = lw_code<REF>(sx, sy); oc = (uint8_t)c0; }
        }
    }
    dw[t] = ow; dc[t] = oc;
}
/* the ink plane (train --ink-head, prob254, 255 unknown): trilinear over the corners when every corner of weight > 1e-7 is known, else
   unknown; outside unknown (sample.c sample_warp_ink) */
/* bits (optional, --rv ink bits): src is the CT bytes and the plane's value is ink_val's, taken at the corners (= ink_bits_k's plane, without
   writing it: the staged CT bytes stay as nn_aug_ct needs them) */
static __device__ __forceinline__ int ink_val(const uint8_t *src, const uint8_t *bits, int it, size_t j) {
    return !bits ? src[j] : bits[j >> 3] >> (j & 7) & 1 ? (it <= 255 && src[j] >= it ? 254 : 0) : 255;
}
__global__ void lw_ink_k(const uint8_t *src, uint8_t *dst, int P, const float *rad, size_t k0, size_t n, size_t so, const uint8_t *bits = nullptr, int it = 0) {
    const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x, k = k0 + t;
    if (t >= n) return;
    size_t b0; float F[3]; uint8_t out = 255;
    if (lw_coords(rad, P, k, &b0, F)) {
        b0 -= so;
        float acc = 0.f; int ok = 1;
        for (int cn = 0; cn < 8; cn++) {
            const float w = lw_w(F, cn);
            if (!(w > 1e-7f)) continue;
            const int q = ink_val(src, bits, it, b0 + lw_off(P, cn));
            if (q == 255) { ok = 0; break; }
            acc = fa(acc, fm(w, (float)q));
        }
        if (ok) { const int q = __float2int_rn(acc); out = (uint8_t)(q < 0 ? 0 : q > 254 ? 254 : q); }
    }
    dst[t] = out;
}
__global__ void ink_sym_k(const uint8_t *src, uint8_t *dst, int P, int code, size_t k0, size_t n, size_t so, const uint8_t *bits = nullptr, int it = 0) {   /* out[i0 i1 i2] = in[sum_d stride[perm d] (flip d ? P-1-i_d : i_d)]; bits: lw_ink_k's */
    const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x, k = k0 + t;
    if (t >= n) return;
    int i[3]; lw_zyx(k, P, i);
    const int64_t stride[3] = {(int64_t)P * P, P, 1};
    int64_t q = 0;
    for (int d = 0; d < 3; d++) { const int pd = code >> (2 * d) & 3, fd = code >> (8 + d) & 1; q += stride[pd] * (fd ? P - 1 - i[d] : i[d]); }
    dst[t] = (uint8_t)ink_val(src, bits, it, (size_t)q - so);
}
/* lw_coords' source z over output planes [z0, z0 + nz): U = c + sc M vt with vt[d] = i_d - c - tb[d][i_d] (the per-axis tables) is
   linear in vt, so its extremes over the box of vt's per-axis ranges are at the box's corners (double here; the kernel's float
   rounding is far below the plane of margin) */
extern "C" void nn_aug_src_range(const void *rad_h, int P, int z0, int nz, int *s0, int *s1) {
    const int32_t *h = (const int32_t *)rad_h; const float *rad = (const float *)rad_h;
    const int sb = h[5], warped = h[6] & 1, affine = h[6] >> 1 & 1;
    int dz = 0; for (int d = 0; d < 3; d++) if ((sb >> (2 * d) & 3) == 0) dz = d;   /* the output axis that lands on source z */
    const int fz = sb >> (8 + dz) & 1;
    if (!warped) {   /* the cube symmetry: source z is output axis dz (flipped by fz) */
        if (dz != 0) { *s0 = 0; *s1 = P; return; }
        *s0 = fz ? P - z0 - nz : z0; *s1 = fz ? P - z0 : z0 + nz; return;
    }
    const float *tb = rad + 8 + 2 * (size_t)P, *R = tb + 6 * (size_t)P, *A = R + 9; const double sc = R[27], c = .5 * (P - 1);
    double lo[3], hi[3];
    for (int d = 0; d < 3; d++) {
        const int i0 = d == 0 ? z0 : 0, i1 = d == 0 ? z0 + nz : P; lo[d] = 1e30; hi[d] = -1e30;
        for (int i = i0; i < i1; i++) { const double v = i - c - tb[d * (size_t)P + i]; if (v < lo[d]) lo[d] = v; if (v > hi[d]) hi[d] = v; }
    }
    double M[3];   /* row dz of the linear map: U[dz] = c + sc sum_e M[e] vt[e] */
    for (int e = 0; e < 3; e++) M[e] = affine ? A[dz * 3] * R[e * 3] + A[dz * 3 + 1] * R[e * 3 + 1] + A[dz * 3 + 2] * R[e * 3 + 2] : R[e * 3 + dz];
    double umin = 1e30, umax = -1e30;
    for (int k = 0; k < 8; k++) {
        double u = 0; for (int e = 0; e < 3; e++) u += M[e] * (k >> e & 1 ? hi[e] : lo[e]);
        u = c + sc * u; if (fz) u = (P - 1) - u;
        if (u < umin) umin = u; if (u > umax) umax = u;
    }
    long a = (long)floor(umin - 1), b = (long)floor(umax + 1) + 2;   /* (corners B, B + 1 of B = floor(V), a plane of margin each side) */
    if (a < 0) a = 0; if (a > P - 2) a = P - 2; if (b > P) b = P; if (b < a + 2) b = a + 2;
    *s0 = (int)a & ~1; *s1 = (int)b;   /* (s0 even: s0 P^2 a multiple of 8 for the ink bits at any P divisible by 2) */
}
/* ---- z ranges (one window split in z across GPUs, train --split z on the latent path): the builders write output planes [z0, z0 + nz)
   of the global P^3 window (the same records and draws, global coordinates) from the source planes [s0, s1) only (uploaded from the
   host plane's offset s0 P^2). nn_aug_src_range bounds [s0, s1) for a range. The whole-window calls are z0 0, nz P, s0 0, s1 P ---- */
static void lw_range_check(const char *fn, int P, int z0, int nz, int s0, int s1) {
    if (z0 < 0 || nz < 1 || z0 + nz > P || s0 < 0 || s1 > P || s1 <= s0 || (((size_t)s0 * P * P) & 7)) {
        fprintf(stderr, "%s: z range [%d, %d) source [%d, %d) of P %d\n", fn, z0, z0 + nz, s0, s1, P); abort();
    }
}
static void lw_tables(aug_dev *a, const float *cs_h, const float *sn_h) {
    if (a->cssn) return;
    CK(cudaMalloc(&a->cssn, 2 * 256 * sizeof(float))); CK(cudaMemcpy(a->cssn, cs_h, LW_PERIOD * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(a->cssn + 256, sn_h, LW_PERIOD * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaStreamSynchronize(0));   /* (once: a pageable upload of <= 64 KB is only staged on return, its device write queued on the default stream; the kernels below may run on the copy stream) */
}
/* the source planes [s0, s1) of host plane h into the staging buffer at off (in planes), unless they are staged already (key) */
static void lw_stage(aug_dev *a, const uint8_t *h, int P, int s0, int s1, size_t off, cudaStream_t s) {
    const size_t p2 = (size_t)P * P;
    CK(cudaMemcpyAsync(a->stage + off * p2, h + (size_t)s0 * p2, (size_t)(s1 - s0) * p2, cudaMemcpyHostToDevice, s));
}

/* one window's labels of one kind (0 packed targets / mask: a_h -> da, and db (optional) the pmask plane; 1 phase: a_h -> da; 3 ink plane: a_h -> da; 2 winding mod 4 and component: a_h, b_h ->
   da, db) from the pinned host source labels, warped by rad_d (the window's radial record, on the device); cs_h / sn_h: the phase
   codes' cos / sin (PHASE_PERIOD floats, sample.c g_wcs / g_wsn). Shares nn_aug_ct's staging buffer (2 P^3 with labels) and its order */
extern "C" void nn_label_warp_z(int kind, const uint8_t *a_h, const uint8_t *b_h, const void *rad_d, int P, int z0, int nz, int s0, int s1, uint8_t *da, uint8_t *db,
                                const float *cs_h, const float *sn_h, int on_copy_stream) {
    lw_range_check("nn_label_warp_z", P, z0, nz, s0, s1);
    const size_t p2 = (size_t)P * P, ns = (size_t)(s1 - s0), n = (size_t)nz * p2, k0 = (size_t)z0 * p2, so = (size_t)s0 * p2;
    aug_dev *a = aug_get(P, 2 * ns * p2);
    cudaStream_t s = on_copy_stream ? copy_stream() : 0;
    lw_tables(a, cs_h, sn_h);
    if (a->used) CK(cudaStreamWaitEvent(s, a->done, 0));
    lw_stage(a, a_h, P, s0, s1, 0, s);
    if (kind == 2) lw_stage(a, b_h, P, s0, s1, ns, s);
    nn_ptick_fine("      y label h2d");
    const unsigned nb = nblk(n, 256); const float *rad = (const float *)rad_d;
    if (kind == 0) lw_tm_k<<<nb, 256, 0, s>>>(a->stage, da, db, P, rad, k0, n, so);   /* (db: the pmask plane, or nullptr) */
    else if (kind == 1 && g_lw_ref) lw_phase_k<1><<<nb, 256, 0, s>>>(a->stage, da, P, rad, a->cssn, a->cssn + 256, k0, n, so);
    else if (kind == 1) lw_phase_k<0><<<nb, 256, 0, s>>>(a->stage, da, P, rad, a->cssn, a->cssn + 256, k0, n, so);
    else if (kind == 3) lw_ink_k<<<nb, 256, 0, s>>>(a->stage, da, P, rad, k0, n, so);   /* (the ink plane) */
    else if (g_lw_ref) lw_w4_k<1><<<nb, 256, 0, s>>>(a->stage, a->stage + ns * p2, da, db, P, rad, a->cssn, a->cssn + 256, k0, n, so);
    else lw_w4_k<0><<<nb, 256, 0, s>>>(a->stage, a->stage + ns * p2, da, db, P, rad, a->cssn, a->cssn + 256, k0, n, so);
    if (kind == 1 && !g_lw_ref) lw_setkey(a, a_h, s0, s1); else a->key = nullptr;   /* (the phase plane stays staged: nn_w4runs_label's source) */
    CK(cudaEventRecord(a->done, s)); a->used = 1;
    KCHECK();
}
extern "C" void nn_label_warp(int kind, const uint8_t *a_h, const uint8_t *b_h, const void *rad_d, int P, uint8_t *da, uint8_t *db,
                              const float *cs_h, const float *sn_h, int on_copy_stream) {
    nn_label_warp_z(kind, a_h, b_h, rad_d, P, 0, P, 0, P, da, db, cs_h, sn_h, on_copy_stream);
}
/* --rv ink bits (sample.h batch.inkb): the papyrus plane known ? (ct8 >= it ? 254 : 0) : 255 on the source grid, in place over the
   CT bytes (stage), then through the window's warp (lw_ink_k) or its cube symmetry (= sample.c sym_u8) */
__global__ void ink_bits_k(uint8_t *st, const uint8_t *bits, int it, size_t n) {
    const size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= n) return;
    st[k] = bits[k >> 3] >> (k & 7) & 1 ? (it <= 255 && st[k] >= it ? 254 : 0) : 255;
}
/* one window's source-grid label plane through its cube symmetry (symcode: the radial record's h[5]) into da (train --gpu-aug 3 runs
   slots keep the phase on the source grid) */
extern "C" void nn_label_sym_z(const uint8_t *a_h, int symcode, int P, int z0, int nz, int s0, int s1, uint8_t *da, int on_copy_stream) {
    lw_range_check("nn_label_sym_z", P, z0, nz, s0, s1);
    const size_t p2 = (size_t)P * P, ns = (size_t)(s1 - s0), n = (size_t)nz * p2;
    aug_dev *a = aug_get(P, 2 * ns * p2);
    cudaStream_t s = on_copy_stream ? copy_stream() : 0;
    if (a->used) CK(cudaStreamWaitEvent(s, a->done, 0));
    lw_stage(a, a_h, P, s0, s1, 0, s);
    nn_ptick_fine("      y label h2d");
    ink_sym_k<<<nblk(n, 256), 256, 0, s>>>(a->stage, da, P, symcode, (size_t)z0 * p2, n, (size_t)s0 * p2);
    if (g_lw_ref) a->key = nullptr; else lw_setkey(a, a_h, s0, s1);
    CK(cudaEventRecord(a->done, s)); a->used = 1;
    KCHECK();
}
extern "C" void nn_label_sym(const uint8_t *a_h, int symcode, int P, uint8_t *da, int on_copy_stream) { nn_label_sym_z(a_h, symcode, P, 0, P, 0, P, da, on_copy_stream); }
/* --gpu-aug 3 runs slots: one window's winding mod 4 and component from its source-grid phase (side_h) and runs (band_w4dev.h w4runs),
   rebuilt on the device (nn_w4runs_apply_z over the staged planes [s0, s1), in the staging buffer), then warped (lw_w4_k) or through
   the symmetry into da / db */
extern "C" void nn_w4runs_label_z(const uint8_t *side_h, const void *runs, const void *rad_d, int warped, int symcode, int P, int z0, int nz, int s0, int s1,
                                  uint8_t *da, uint8_t *db, const float *cs_h, const float *sn_h, int on_copy_stream) {
    lw_range_check("nn_w4runs_label_z", P, z0, nz, s0, s1);
    const w4runs *r = (const w4runs *)runs;
    cudaStream_t s = on_copy_stream ? copy_stream() : 0;
    const size_t p2 = (size_t)P * P, ns = (size_t)(s1 - s0), n = (size_t)nz * p2, k0 = (size_t)z0 * p2, so = (size_t)s0 * p2; const unsigned nb = nblk(n, 256);
    if (!r->noff) { CK(cudaMemsetAsync(da, 255, n, s)); CK(cudaMemsetAsync(db, 255, n, s)); return; }   /* (no winding: unknown) */
    aug_dev *a = aug_get(P, 2 * ns * p2);
    lw_tables(a, cs_h, sn_h);
    if (a->used) CK(cudaStreamWaitEvent(s, a->done, 0));
    if (!lw_keyed(a, side_h, s0, s1)) lw_stage(a, side_h, P, s0, s1, 0, s);   /* (else: staged by the phase plane's call just before, the same range) */
    a->key = nullptr;
    nn_ptick_fine("      y label h2d");
    if (nn_w4runs_apply_z(a->stage, P, P, P, s0, s1, r->tile, r->ms, r, a->stage, a->stage + ns * p2, (void *)s)) {
        /* no device memory for the rebuild's scratch beside the trainer (2026-11-12: it aborted unissl three times at P 1024 on a
           16 GB card): this window's planes rebuilt on the host from the same phase (w4runs_apply_host: the same bytes), then uploaded */
        static int nfb; const int k = __sync_add_and_fetch(&nfb, 1);
        if (k <= 3 || !(k & (k - 1))) fprintf(stderr, "nn_w4runs_label: no device memory for the runs rebuild: the window rebuilt on the host (%d so far)\n", k);
        const size_t p3 = p2 * P;
        uint8_t *h = (uint8_t *)malloc(2 * p3); if (!h) { fprintf(stderr, "nn_w4runs_label: no host memory for the fallback\n"); abort(); }
        w4runs_apply_host(side_h, P, P, P, r->tile, r->ms, r, h, h + p3);
        CK(cudaMemcpyAsync(a->stage, h + so, ns * p2, cudaMemcpyHostToDevice, s)); CK(cudaMemcpyAsync(a->stage + ns * p2, h + p3 + so, ns * p2, cudaMemcpyHostToDevice, s));
        CK(cudaStreamSynchronize(s)); free(h);
    }
    nn_ptick_fine("      y w4 runs apply");
    if (warped && g_lw_ref) lw_w4_k<1><<<nb, 256, 0, s>>>(a->stage, a->stage + ns * p2, da, db, P, (const float *)rad_d, a->cssn, a->cssn + 256, k0, n, so);
    else if (warped) lw_w4_k<0><<<nb, 256, 0, s>>>(a->stage, a->stage + ns * p2, da, db, P, (const float *)rad_d, a->cssn, a->cssn + 256, k0, n, so);
    else { ink_sym_k<<<nb, 256, 0, s>>>(a->stage, da, P, symcode, k0, n, so); ink_sym_k<<<nb, 256, 0, s>>>(a->stage + ns * p2, db, P, symcode, k0, n, so); }
    CK(cudaEventRecord(a->done, s)); a->used = 1;
    KCHECK();
}
extern "C" void nn_w4runs_label(const uint8_t *side_h, const void *runs, const void *rad_d, int warped, int symcode, int P, uint8_t *da, uint8_t *db,
                                const float *cs_h, const float *sn_h, int on_copy_stream) {
    nn_w4runs_label_z(side_h, runs, rad_d, warped, symcode, P, 0, P, 0, P, da, db, cs_h, sn_h, on_copy_stream);
}
extern "C" void nn_ink_bits_z(const uint8_t *ct8_h, const uint8_t *bits_h, int it, const void *rad_d, int warped, int symcode, int P, int z0, int nz, int s0, int s1,
                              uint8_t *da, int on_copy_stream) {
    lw_range_check("nn_ink_bits_z", P, z0, nz, s0, s1);
    const size_t p2 = (size_t)P * P, ns = (size_t)(s1 - s0), n = (size_t)nz * p2, k0 = (size_t)z0 * p2, so = (size_t)s0 * p2; const unsigned nb = nblk(n, 256);
    aug_dev *a = aug_get(P, 2 * ns * p2);
    cudaStream_t s = on_copy_stream ? copy_stream() : 0;
    if (a->used) CK(cudaStreamWaitEvent(s, a->done, 0));
    lw_stage(a, ct8_h, P, s0, s1, 0, s);
    CK(cudaMemcpyAsync(a->stage + ns * p2, bits_h + so / 8, ns * p2 / 8, cudaMemcpyHostToDevice, s));
    nn_ptick_fine("      y label h2d");
    if (g_lw_ref) {   /* (tests: the plane written over the CT bytes first) */
        ink_bits_k<<<nblk(ns * p2, 256), 256, 0, s>>>(a->stage, a->stage + ns * p2, it, ns * p2);
        if (warped) lw_ink_k<<<nb, 256, 0, s>>>(a->stage, da, P, (const float *)rad_d, k0, n, so);
        else ink_sym_k<<<nb, 256, 0, s>>>(a->stage, da, P, symcode, k0, n, so);
        a->key = nullptr;
    } else {   /* the plane's values taken at the kernels' reads: the CT bytes stay staged for nn_aug_ct, which follows in train's upload */
        if (warped) lw_ink_k<<<nb, 256, 0, s>>>(a->stage, da, P, (const float *)rad_d, k0, n, so, a->stage + ns * p2, it);
        else ink_sym_k<<<nb, 256, 0, s>>>(a->stage, da, P, symcode, k0, n, so, a->stage + ns * p2, it);
        lw_setkey(a, ct8_h, s0, s1);
    }
    CK(cudaEventRecord(a->done, s)); a->used = 1;
    KCHECK();
}
extern "C" void nn_ink_bits(const uint8_t *ct8_h, const uint8_t *bits_h, int it, const void *rad_d, int warped, int symcode, int P, uint8_t *da, int on_copy_stream) {
    nn_ink_bits_z(ct8_h, bits_h, it, rad_d, warped, symcode, P, 0, P, 0, P, da, on_copy_stream);
}
/* tests: the output bytes of the last nn_aug_ct (device -> host, synchronous) */
extern "C" void nn_aug_last_bytes(uint8_t *out, int P) { int dv = 0; cudaGetDevice(&dv); CK(cudaDeviceSynchronize()); CK(cudaMemcpy(out, g_ad[(dv & 7) * 2].stage, (size_t)P * P * P, cudaMemcpyDeviceToHost)); }
