/* dinovol (src/dinovol.h): CUDA kernels of the scrollprize dinovol ViT beyond the xnet layer set (src/xnet.h): channel
   LayerNorm, 2x average pooling, mixed 3D RoPE, a fused flash attention for head dim 48, the SwiGLU gate, the grouped 3^3 FPN
   convolution, the dense-head point gather, and the blend / normalise / quantise steps of ufsm dinovol-embed. Every kernel
   runs on xnet_stream(). fp16 storage, fp32 arithmetic. */
#include "dinovol.h"
#include "xnet.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define ST ((cudaStream_t)xnet_stream())
static inline unsigned nb_of(long long n, int t) { long long b = (n + t - 1) / t; return (unsigned)(b > 2147483647LL ? 2147483647LL : b); }

/* ---- LayerNorm over the last dimension (rows of C), one warp per row: y = (x - mean) / sqrt(var + eps) w + b (+ res) (relu) ---- */
__global__ void k_lnrow(const __half *__restrict__ x, const float *__restrict__ w, const float *__restrict__ b, float eps,
                        const __half *__restrict__ res, int relu, __half *__restrict__ y, long long M, int C) {
    const long long row = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5; const int lane = threadIdx.x & 31;
    if (row >= M) return;
    const __half *xr = x + row * C;
    float v[16]; int n = 0; float s = 0.f;
    for (int c = lane; c < C; c += 32) { v[n] = __half2float(xr[c]); s += v[n]; n++; }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mean = s / C; float q = 0.f;
    for (int i = 0; i < n; i++) { const float d = v[i] - mean; q += d * d; }
    for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    const float rs = rsqrtf(q / C + eps);
    n = 0;
    for (int c = lane; c < C; c += 32, n++) {
        float o = __half2float(__float2half_rn((v[n] - mean) * rs * w[c] + b[c]));   /* the affine output rounds to the activation dtype first */
        if (res) o += __half2float(res[row * C + c]);
        if (relu) o = fmaxf(o, 0.f);
        y[row * C + c] = __float2half_rn(o);
    }
}
/* rows of C = 8 G channels: G threads per row (G 4..32), 16-byte loads */
template <int G> __global__ void k_lnrow_v(const __half *__restrict__ x, const float *__restrict__ w, const float *__restrict__ b, float eps,
                                          const __half *__restrict__ res, int relu, __half *__restrict__ y, long long M) {
    constexpr int C = 8 * G;
    const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x, row = t / G; const int q = (int)(t % G);
    const bool ok = row < M;
    float v[8]; float s = 0.f;
    if (ok) { const uint4 u = *(const uint4 *)(x + row * C + q * 8); const __half *h = (const __half *)&u; for (int k = 0; k < 8; k++) { v[k] = __half2float(h[k]); s += v[k]; } }
    else for (int k = 0; k < 8; k++) v[k] = 0.f;
#pragma unroll
    for (int o = G / 2; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mean = s / C; float qq = 0.f;
    for (int k = 0; k < 8; k++) { const float d = v[k] - mean; qq += d * d; }
#pragma unroll
    for (int o = G / 2; o; o >>= 1) qq += __shfl_xor_sync(0xffffffffu, qq, o);
    if (!ok) return;
    const float rs = rsqrtf(qq / C + eps);
    __align__(16) __half out[8]; __align__(16) __half rr[8];
    if (res) *(uint4 *)rr = *(const uint4 *)(res + row * C + q * 8);
    for (int k = 0; k < 8; k++) {
        const int c = q * 8 + k;
        float o = __half2float(__float2half_rn((v[k] - mean) * rs * w[c] + b[c]));
        if (res) o += __half2float(rr[k]);
        if (relu) o = fmaxf(o, 0.f);
        out[k] = __float2half_rn(o);
    }
    *(uint4 *)(y + row * C + q * 8) = *(const uint4 *)out;
}
extern "C" void dv_layernorm(const void *x, const float *w, const float *b, float eps, const void *res, int relu, void *y, long long M, int C) {
    if (C > 512) { fprintf(stderr, "dv_layernorm: C %d > 512\n", C); return; }
    const unsigned nb = nb_of(M * (C / 8), 256);
    switch (C) {
    case 32: k_lnrow_v<4><<<nb, 256, 0, ST>>>((const __half *)x, w, b, eps, (const __half *)res, relu, (__half *)y, M); return;
    case 64: k_lnrow_v<8><<<nb, 256, 0, ST>>>((const __half *)x, w, b, eps, (const __half *)res, relu, (__half *)y, M); return;
    case 128: k_lnrow_v<16><<<nb, 256, 0, ST>>>((const __half *)x, w, b, eps, (const __half *)res, relu, (__half *)y, M); return;
    case 256: k_lnrow_v<32><<<nb, 256, 0, ST>>>((const __half *)x, w, b, eps, (const __half *)res, relu, (__half *)y, M); return;
    default: break;
    }
    k_lnrow<<<nb_of(M * 32, 256), 256, 0, ST>>>((const __half *)x, w, b, eps, (const __half *)res, relu, (__half *)y, M, C);
}

/* ---- 2x2x2 average pooling, NDHWC ---- */
__global__ void k_avgpool3(const __half *__restrict__ x, __half *__restrict__ y, int D, int H, int W, int C) {
    const int Do = D / 2, Ho = H / 2, Wo = W / 2; const long long total = (long long)Do * Ho * Wo * C;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const int c = (int)(i % C); long long t = i / C; const int xo = (int)(t % Wo); t /= Wo; const int yo = (int)(t % Ho); const int zo = (int)(t / Ho);
        float s = 0.f;
        for (int a = 0; a < 2; a++) for (int e = 0; e < 2; e++) for (int f = 0; f < 2; f++)
            s += __half2float(x[(((size_t)(2 * zo + a) * H + 2 * yo + e) * W + 2 * xo + f) * C + c]);
        y[i] = __float2half_rn(s * 0.125f);
    }
}
extern "C" void dv_avgpool2(const void *x, void *y, int D, int H, int W, int C) {
    k_avgpool3<<<nb_of((long long)D * H * W * C / 8, 256), 256, 0, ST>>>((const __half *)x, (__half *)y, D, H, W, C);
}

/* ---- mixed 3D RoPE on the q and k parts of qkv [N][3 H 48] (patch tokens n >= P, grid g^3 in z y x order): pair p < 24 of head
   h rotates dims p and p + 24 by 2 pi sum_d coord_d mix[h][p][d], coord = 2 (i + 0.5) / g - 1 (normalize_coords "separate").
   Whole turns are removed before the sine / cosine (the fast-math sinf is inexact past pi). ---- */
__global__ void k_rope(__half *__restrict__ qkv, const float *__restrict__ mix /* [H][24][3] */, int N, int P, int g, int H) {
    const long long total = (long long)(N - P) * H * 24;
    const int ld = 3 * H * 48;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const int p = (int)(i % 24); long long t = i / 24; const int h = (int)(t % H); const long long tok = t / H;
        const int ix = (int)(tok % g), iy = (int)((tok / g) % g), iz = (int)(tok / ((long long)g * g));
        const float cz = 2.f * ((iz + 0.5f) / g) - 1.f, cy = 2.f * ((iy + 0.5f) / g) - 1.f, cx = 2.f * ((ix + 0.5f) / g) - 1.f;
        const float *m = mix + ((size_t)h * 24 + p) * 3;
        /* the reference einsum runs in fp32: angle = 2 pi (c . f) with fp32 products */
        const float af = cz * m[0] + cy * m[1] + cx * m[2];
        const float red = af - rintf(af);   /* whole turns removed: the fast sincos is accurate on [-pi, pi] */
        float s, c; __sincosf(6.28318530717958648f * red, &s, &c);
        __half *row = qkv + (size_t)(P + tok) * ld + h * 48;
        for (int part = 0; part < 2; part++) {
            __half *r = row + part * H * 48;
            const float a = __half2float(r[p]), b = __half2float(r[p + 24]);
            r[p] = __float2half_rn(a * c - b * s); r[p + 24] = __float2half_rn(b * c + a * s);
        }
    }
}
extern "C" void dv_rope(void *qkv, const float *mix_dev, int N, int P, int g, int H) {
    k_rope<<<nb_of((long long)(N - P) * H * 24, 256), 256, 0, ST>>>((__half *)qkv, mix_dev, N, P, g, H);
}

/* ---- flash attention, head dim 48: qkv [N][3 H 48] (q | k | v, head h at h 48) -> out [N][H 48]. Block = 64 queries of one head
   (4 warps x 16 rows), key / value tiles of 64 rows double-buffered through cp.async, mma.sync m16n8k16 fp16 -> fp32, online
   softmax in the exp2 domain. ---- */
__device__ __forceinline__ unsigned su32(const void *p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cpa16(void *dst, const void *src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(su32(dst)), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int n> __device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(n)); }
__device__ __forceinline__ void ldm4(unsigned *r, const void *p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(su32(p)));
}
__device__ __forceinline__ void ldm4t(unsigned *r, const void *p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(su32(p)));
}
__device__ __forceinline__ void mma(float *c, const unsigned *a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned pack2(float a, float b) { __half2 h = __floats2half2_rn(a, b); return *(unsigned *)&h; }

enum { FA_BQ = 64, FA_BK = 64, FA_LD = 56 };   /* 48 dims + 8 pad: 112-byte rows, conflict-free ldmatrix */
__global__ void __launch_bounds__(128) k_flash48(const __half *__restrict__ qkv, __half *__restrict__ out, int N, int H, float sl2) {
    __shared__ __align__(128) __half sQ[FA_BQ * FA_LD], sK[2][FA_BK * FA_LD], sV[2][FA_BK * FA_LD];
    const int h = blockIdx.y, q0 = blockIdx.x * FA_BQ, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int ld = 3 * H * 48;
    const __half *Qg = qkv + h * 48, *Kg = qkv + (H + h) * 48, *Vg = qkv + (2 * H + h) * 48;
    for (int i = tid; i < FA_BQ * 6; i += 128) {
        const int r = i / 6, c = i % 6, n = q0 + r;
        cpa16(sQ + r * FA_LD + c * 8, Qg + (size_t)(n < N ? n : N - 1) * ld + c * 8, n < N ? 16 : 0);
    }
    auto load_kv = [&](int buf, int k0) {
        for (int i = tid; i < FA_BK * 6; i += 128) {
            const int r = i / 6, c = i % 6, n = k0 + r; const bool ok = n < N; const size_t o = (size_t)(ok ? n : N - 1) * ld + c * 8;
            cpa16(sK[buf] + r * FA_LD + c * 8, Kg + o, ok ? 16 : 0);
            cpa16(sV[buf] + r * FA_LD + c * 8, Vg + o, ok ? 16 : 0);
        }
    };
    load_kv(0, 0); cpa_commit();
    const int nt = (N + FA_BK - 1) / FA_BK;
    unsigned qf[3][4];
    float o[6][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
#pragma unroll
    for (int a = 0; a < 6; a++) o[a][0] = o[a][1] = o[a][2] = o[a][3] = 0.f;
    const int t4 = lane & 3;
    for (int j = 0; j < nt; j++) {
        if (j + 1 < nt) load_kv((j + 1) & 1, (j + 1) * FA_BK);
        cpa_commit();
        cpa_wait<1>(); __syncthreads();
        if (j == 0) {
#pragma unroll
            for (int kk = 0; kk < 3; kk++) ldm4(qf[kk], sQ + (warp * 16 + (lane & 15)) * FA_LD + kk * 16 + (lane >> 4) * 8);
        }
        const __half *K = sK[j & 1], *V = sV[j & 1];
        float s[8][4];
#pragma unroll
        for (int a = 0; a < 8; a++) s[a][0] = s[a][1] = s[a][2] = s[a][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 3; kk++)
#pragma unroll
            for (int nj = 0; nj < 4; nj++) {
                unsigned r[4]; ldm4(r, K + (nj * 16 + (lane & 7) + ((lane >> 4) << 3)) * FA_LD + kk * 16 + ((lane >> 3) & 1) * 8);
                mma(s[2 * nj], qf[kk], r[0], r[1]); mma(s[2 * nj + 1], qf[kk], r[2], r[3]);
            }
        const int kb = j * FA_BK;
#pragma unroll
        for (int a = 0; a < 8; a++)
#pragma unroll
            for (int e = 0; e < 4; e++) { const int key = kb + a * 8 + t4 * 2 + (e & 1); s[a][e] = key < N ? s[a][e] * sl2 : -INFINITY; }
#pragma unroll
        for (int r = 0; r < 2; r++) {
            float mx = -INFINITY;
#pragma unroll
            for (int a = 0; a < 8; a++) mx = fmaxf(mx, fmaxf(s[a][2 * r], s[a][2 * r + 1]));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1)); mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float mn = fmaxf(m[r], mx), al = exp2f(m[r] - mn); m[r] = mn;
            float sum = 0.f;
#pragma unroll
            for (int a = 0; a < 8; a++) { s[a][2 * r] = exp2f(s[a][2 * r] - mn); s[a][2 * r + 1] = exp2f(s[a][2 * r + 1] - mn); sum += s[a][2 * r] + s[a][2 * r + 1]; }
            l[r] = l[r] * al + sum;
#pragma unroll
            for (int a = 0; a < 6; a++) { o[a][2 * r] *= al; o[a][2 * r + 1] *= al; }
        }
#pragma unroll
        for (int kc = 0; kc < 4; kc++) {
            unsigned pa[4] = {pack2(s[2 * kc][0], s[2 * kc][1]), pack2(s[2 * kc][2], s[2 * kc][3]), pack2(s[2 * kc + 1][0], s[2 * kc + 1][1]), pack2(s[2 * kc + 1][2], s[2 * kc + 1][3])};
#pragma unroll
            for (int dp = 0; dp < 3; dp++) {
                unsigned r[4]; ldm4t(r, V + (kc * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * FA_LD + dp * 16 + (lane >> 4) * 8);
                mma(o[2 * dp], pa, r[0], r[1]); mma(o[2 * dp + 1], pa, r[2], r[3]);
            }
        }
        __syncthreads();
    }
    cpa_wait<0>();
    const int g = lane >> 2;
#pragma unroll
    for (int r = 0; r < 2; r++) {
        float lt = l[r]; lt += __shfl_xor_sync(0xffffffffu, lt, 1); lt += __shfl_xor_sync(0xffffffffu, lt, 2);
        const float inv = 1.f / lt; const int n = q0 + warp * 16 + g + 8 * r;
        if (n < N)
#pragma unroll
            for (int a = 0; a < 6; a++) *(__half2 *)(out + (size_t)n * H * 48 + h * 48 + a * 8 + t4 * 2) = __floats2half2_rn(o[a][2 * r] * inv, o[a][2 * r + 1] * inv);
    }
}
extern "C" void dv_attention48(const void *qkv, void *out, int N, int H) {
    const float sl2 = (float)(1.4426950408889634 / sqrt(48.0));
    dim3 g((unsigned)((N + FA_BQ - 1) / FA_BQ), (unsigned)H);
    k_flash48<<<g, 128, 0, ST>>>((const __half *)qkv, (__half *)out, N, H, sl2);
}

/* ---- SwiGLU (timm GluMlp, gate_last False): g[n][j] = silu(a[n][j]) a[n][F + j], a [N][2F] ---- */
__global__ void k_glu(const __half *__restrict__ a, __half *__restrict__ g, long long N, int F) {
    const long long total = N * F;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const long long n = i / F; const int j = (int)(i % F);
        const float x1 = __half2float(a[n * 2 * F + j]), x2 = __half2float(a[n * 2 * F + F + j]);
        g[i] = __float2half_rn(__half2float(__float2half_rn(x1 / (1.f + expf(-x1)))) * x2);
    }
}
extern "C" void dv_glu(const void *a, void *g, long long N, int F) { k_glu<<<nb_of(N * F, 256), 256, 0, ST>>>((const __half *)a, (__half *)g, N, F); }

/* ---- grouped 3^3 convolution (padding 1, groups of 8 channels) + bias, then exact GELU: x, y [n^3][C] ---- */
__global__ void k_gconv8_gelu(const __half *__restrict__ x, const float *__restrict__ w /* [C][8][27] */, const float *__restrict__ b, __half *__restrict__ y, int n, int C) {
    const long long total = (long long)n * n * n * C;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const int co = (int)(i % C); long long t = i / C; const int xx = (int)(t % n); t /= n; const int yy = (int)(t % n); const int zz = (int)(t / n);
        const int c0 = co / 8 * 8; float acc = b[co];
        for (int a = 0; a < 3; a++) { const int z = zz + a - 1; if (z < 0 || z >= n) continue;
            for (int e = 0; e < 3; e++) { const int yv = yy + e - 1; if (yv < 0 || yv >= n) continue;
                for (int f = 0; f < 3; f++) { const int xv = xx + f - 1; if (xv < 0 || xv >= n) continue;
                    const __half *xp = x + (((size_t)z * n + yv) * n + xv) * C + c0; const float *wp = w + (size_t)co * 216 + (a * 3 + e) * 3 + f;
#pragma unroll
                    for (int ci = 0; ci < 8; ci++) acc += __half2float(xp[ci]) * wp[ci * 27];
                }
            }
        }
        y[i] = __float2half_rn(0.5f * acc * (1.f + erff(acc * 0.70710678118654752f)));
    }
}
extern "C" void dv_gconv8_gelu(const void *x, const float *w_dev, const float *b_dev, void *y, int n, int C) {
    k_gconv8_gelu<<<nb_of((long long)n * n * n * C, 256), 256, 0, ST>>>((const __half *)x, w_dev, b_dev, (__half *)y, n, C);
}

/* ---- dense head point gather (PointDenseHead.readout inputs): one warp per point; coordinates in tile voxel indices (centers at +0.5).
   tok [P][Ct] = trilinear tokens; feat [P][Ct + sum skip + Cf] = tokens | LN(skip0) | LN(skip1) | LN(skip2) | LN(fpn) ---- */
struct dv_grid { const __half *p; int n, c; const float *w, *b; };   /* grid [n^3][c]; LN params (nullptr: none) */
struct dv_gather_args { dv_grid g[5]; int T, ng; float eps; };
__global__ void k_gather(dv_gather_args A, const float *__restrict__ coords, int P, __half *__restrict__ tok, __half *__restrict__ feat, int F) {
    const long long pt = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5; const int lane = threadIdx.x & 31;
    if (pt >= P) return;
    const float cz = coords[pt * 3], cy = coords[pt * 3 + 1], cx = coords[pt * 3 + 2];
    int off = 0;
    for (int gi = 0; gi < A.ng; gi++) {
        const dv_grid G = A.g[gi]; const float sc = (float)G.n / (float)A.T, top = (float)(G.n - 1);
        float pz = fminf(fmaxf((cz + 0.5f) * sc - 0.5f, 0.f), top), py = fminf(fmaxf((cy + 0.5f) * sc - 0.5f, 0.f), top), px = fminf(fmaxf((cx + 0.5f) * sc - 0.5f, 0.f), top);
        const float lz = floorf(pz), ly = floorf(py), lx = floorf(px), fz = pz - lz, fy = py - ly, fx = px - lx;
        int iz[2] = {(int)lz, (int)fminf(lz + 1.f, top)}, iy[2] = {(int)ly, (int)fminf(ly + 1.f, top)}, ix[2] = {(int)lx, (int)fminf(lx + 1.f, top)};
        float wz[2] = {1.f - fz, fz}, wy[2] = {1.f - fy, fy}, wx[2] = {1.f - fx, fx};
        float v[16]; int nv = 0; float s = 0.f;
        for (int c = lane; c < G.c; c += 32) {
            float acc = 0.f;
#pragma unroll
            for (int a = 0; a < 2; a++)
#pragma unroll
                for (int e = 0; e < 2; e++)
#pragma unroll
                    for (int f = 0; f < 2; f++)
                        acc += __half2float(G.p[(((size_t)iz[a] * G.n + iy[e]) * G.n + ix[f]) * G.c + c]) * (wz[a] * wy[e] * wx[f]);
            v[nv++] = acc; s += acc;
        }
        if (G.w) {   /* LayerNorm of the sampled vector */
            for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
            const float mean = s / G.c; float q = 0.f;
            for (int i = 0; i < nv; i++) { const float d = v[i] - mean; q += d * d; }
            for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
            const float rs = rsqrtf(q / G.c + A.eps); nv = 0;
            for (int c = lane; c < G.c; c += 32, nv++) v[nv] = (v[nv] - mean) * rs * G.w[c] + G.b[c];
        } else if (tok) { nv = 0; for (int c = lane; c < G.c; c += 32, nv++) tok[pt * G.c + c] = __float2half_rn(v[nv]); }
        nv = 0;
        for (int c = lane; c < G.c; c += 32, nv++) feat[pt * F + off + c] = __float2half_rn(v[nv]);
        off += G.c;
    }
}
extern "C" void dv_gather(const dv_gather_grid *grids, int ng, int T, float eps, const float *coords_dev, int P, void *tok, void *feat, int F) {
    dv_gather_args A; A.T = T; A.ng = ng; A.eps = eps;
    for (int i = 0; i < ng; i++) { A.g[i].p = (const __half *)grids[i].p; A.g[i].n = grids[i].n; A.g[i].c = grids[i].c; A.g[i].w = grids[i].w; A.g[i].b = grids[i].b; }
    k_gather<<<nb_of((long long)P * 32, 256), 256, 0, ST>>>(A, coords_dev, P, (__half *)tok, (__half *)feat, F);
}

/* ---- dense / ink readout: proj [P][96] -> unit dense (64) and ink (32), accumulated w acc[row][96] (fp32) or written (acc nullptr: out) ---- */
__global__ void k_head_out(const __half *__restrict__ proj, const float *__restrict__ wgt, const int *__restrict__ rows, int P, float *__restrict__ acc, int nd, int ni) {
    const long long pt = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5; const int lane = threadIdx.x & 31;
    if (pt >= P) return;
    const int C = nd + ni; const __half *r = proj + pt * C;
    float sd = 0.f, si = 0.f, v[4]; int n = 0;
    for (int c = lane; c < C; c += 32, n++) { v[n] = __half2float(r[c]); if (c < nd) sd += v[n] * v[n]; else si += v[n] * v[n]; }
    for (int o = 16; o; o >>= 1) { sd += __shfl_xor_sync(0xffffffffu, sd, o); si += __shfl_xor_sync(0xffffffffu, si, o); }
    const float idn = 1.f / fmaxf(sqrtf(sd), 1e-12f), iin = 1.f / fmaxf(sqrtf(si), 1e-12f);
    const float w = wgt ? wgt[pt] : 1.f; float *dst = acc + (size_t)(rows ? rows[pt] : pt) * C;
    n = 0;
    for (int c = lane; c < C; c += 32, n++) { const float u = v[n] * (c < nd ? idn : iin); if (wgt) dst[c] += w * u; else dst[c] = u; }
}
extern "C" void dv_head_out(const void *proj, const float *wgt, const int *rows, int P, float *acc, int nd, int ni) {
    if (nd + ni > 128) { fprintf(stderr, "dv_head_out: %d channels\n", nd + ni); return; }
    k_head_out<<<nb_of((long long)P * 32, 256), 256, 0, ST>>>((const __half *)proj, wgt, rows, P, acc, nd, ni);
}

/* ---- lattice blend weight (embed_volume.AxisLattice.weights) at a continuous position u relative to the tile origin ---- */
__device__ __forceinline__ float lat_w(float u, int tile, int margin, int ramp) {
    const float rise = fminf(fmaxf((u - margin) / ramp, 0.f), 1.f), fall = fminf(fmaxf(((float)(tile - margin) - u) / ramp, 0.f), 1.f);
    return rise * fall;
}

/* ---- column points of one tile: valid blocks (mask [nz][ny][nx] of the column accumulator, step s) with z in [b0, b1) -> coordinates in the
   tile (voxel indices of block centers), blend weights and accumulator rows; count via atomics (order does not matter: rows differ) ---- */
__global__ void k_points(const uint8_t *__restrict__ mask, int b0, int b1, int ny, int nx, int s, float oz, float oy, float ox,
                         int tile, int margin, int ramp, float *__restrict__ coords, float *__restrict__ wgt, int *__restrict__ rows, int *__restrict__ cnt) {
    const long long total = (long long)(b1 - b0) * ny * nx;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const long long row = (long long)b0 * ny * nx + i;
        if (!mask[row]) continue;
        const int x = (int)(row % nx), y = (int)((row / nx) % ny), z = (int)(row / ((long long)nx * ny));
        const float hs = (s - 1) * 0.5f;
        const float pz = oz + z * s + hs, py = oy + y * s + hs, px = ox + x * s + hs;
        const int k = atomicAdd(cnt, 1);
        coords[3 * k] = pz; coords[3 * k + 1] = py; coords[3 * k + 2] = px;
        wgt[k] = lat_w(pz + 0.5f, tile, margin, ramp) * lat_w(py + 0.5f, tile, margin, ramp) * lat_w(px + 0.5f, tile, margin, ramp);
        rows[k] = (int)row;
    }
}
extern "C" int dv_points(const uint8_t *mask_dev, int b0, int b1, int ny, int nx, int s, float oz, float oy, float ox, int tile, int margin, int ramp,
                         float *coords, float *wgt, int *rows, int *cnt_dev) {
    cudaMemsetAsync(cnt_dev, 0, 4, ST);
    const long long total = (long long)(b1 - b0) * ny * nx;
    if (total > 0) k_points<<<nb_of(total, 256), 256, 0, ST>>>(mask_dev, b0, b1, ny, nx, s, oz, oy, ox, tile, margin, ramp, coords, wgt, rows, cnt_dev);
    int n = 0; cudaMemcpyAsync(&n, cnt_dev, 4, cudaMemcpyDeviceToHost, ST); cudaStreamSynchronize(ST);
    return n;
}

/* ---- token blend: acc [nz][ny][nx][C + 1] += w tok[tz][ty][tx] (and w in the last channel) for z rows [b0, b1); tok grid g^3 [C];
   tz = (b + tz0), ty = y + ty0, tx = x + tx0; weights at the token centers relative to the tile origin (oz, oy, ox in voxels) ---- */
__global__ void k_tok_acc(float *__restrict__ acc, int b0, int b1, int ny, int nx, int C, const __half *__restrict__ tok, int g, int tz0, int ty0, int tx0,
                          int ps, int tile, int margin, int ramp, const uint8_t *__restrict__ mask) {
    const long long total = (long long)(b1 - b0) * ny * nx * (C + 1);
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const int c = (int)(i % (C + 1)); const long long cell = (long long)b0 * ny * nx + i / (C + 1);
        if (mask && !mask[cell]) continue;
        const int x = (int)(cell % nx), y = (int)((cell / nx) % ny), z = (int)(cell / ((long long)nx * ny));
        const int tz = z + tz0, ty = y + ty0, tx = x + tx0;
        const float w = lat_w((float)(tz * ps) + ps * 0.5f, tile, margin, ramp) * lat_w((float)(ty * ps) + ps * 0.5f, tile, margin, ramp) * lat_w((float)(tx * ps) + ps * 0.5f, tile, margin, ramp);
        acc[cell * (C + 1) + c] += c < C ? w * __half2float(tok[(((size_t)tz * g + ty) * g + tx) * C + c]) : w;
    }
}
extern "C" void dv_tok_acc(float *acc, int b0, int b1, int ny, int nx, int C, const void *tok, int g, int tz0, int ty0, int tx0, int ps, int tile, int margin, int ramp, const uint8_t *mask) {
    const long long total = (long long)(b1 - b0) * ny * nx * (C + 1);
    if (total > 0) k_tok_acc<<<nb_of(total, 256), 256, 0, ST>>>(acc, b0, b1, ny, nx, C, (const __half *)tok, g, tz0, ty0, tx0, ps, tile, margin, ramp, mask);
}

/* ---- valid-block masks: mask[z][y][x] = any nonzero CT voxel in the s^3 block at (z0 + z s, y0 + y s, x0 + x s) of ct [D][H][W] and needed (y / x
   block range [ylo, yhi) x [xlo, xhi)) ---- */
__global__ void k_blockmask(const uint8_t *__restrict__ ct, int D, int H, int W, int z0, int y0, int x0, int s, int nz, int ny, int nx,
                            int ylo, int yhi, int xlo, int xhi, uint8_t *__restrict__ mask) {
    const long long total = (long long)nz * ny * nx;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const int x = (int)(i % nx), y = (int)((i / nx) % ny), z = (int)(i / ((long long)nx * ny));
        uint8_t any = 0;
        if (y >= ylo && y < yhi && x >= xlo && x < xhi)
            for (int a = 0; a < s && !any; a++) for (int e = 0; e < s && !any; e++) for (int f = 0; f < s; f++) {
                const int zz = z0 + z * s + a, yy = y0 + y * s + e, xx = x0 + x * s + f;
                if (zz >= 0 && zz < D && yy >= 0 && yy < H && xx >= 0 && xx < W && ct[((size_t)zz * H + yy) * W + xx]) { any = 1; break; }
            }
        mask[i] = any;
    }
}
extern "C" void dv_blockmask(const uint8_t *ct, int D, int H, int W, int z0, int y0, int x0, int s, int nz, int ny, int nx, int ylo, int yhi, int xlo, int xhi, uint8_t *mask) {
    const long long total = (long long)nz * ny * nx;
    if (total > 0) k_blockmask<<<nb_of(total, 256), 256, 0, ST>>>(ct, D, H, W, z0, y0, x0, s, nz, ny, nx, ylo, yhi, xlo, xhi, mask);
}

/* ---- region copy / add on [nz][*][*][C] fp32 arrays: dst[z][dy0 + y][dx0 + x] (=|+=) src[z][sy0 + y][sx0 + x] for y < ny, x < nx ---- */
__global__ void k_region(float *__restrict__ dst, int dny, int dnx, int dy0, int dx0, const float *__restrict__ src, int sny, int snx, int sy0, int sx0,
                         int nz, int ny, int nx, int C, int add) {
    const long long total = (long long)nz * ny * nx * C;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const int c = (int)(i % C); long long t = i / C; const int x = (int)(t % nx); t /= nx; const int y = (int)(t % ny); const int z = (int)(t / ny);
        const float v = src[(((size_t)z * sny + sy0 + y) * snx + sx0 + x) * C + c];
        float *d = dst + (((size_t)z * dny + dy0 + y) * dnx + dx0 + x) * C + c;
        if (add) *d += v; else *d = v;
    }
}
extern "C" void dv_region(float *dst, int dny, int dnx, int dy0, int dx0, const float *src, int sny, int snx, int sy0, int sx0, int nz, int ny, int nx, int C, int add) {
    const long long total = (long long)nz * ny * nx * C;
    if (total > 0) k_region<<<nb_of(total, 256), 256, 0, ST>>>(dst, dny, dnx, dy0, dx0, src, sny, snx, sy0, sx0, nz, ny, nx, C, add);
}

/* ---- finish a cell [nz][ny][nx] of an accumulator [nz][ANY][ANX][CA] at (y0, x0): vectors of channels [c0, c0 + C) (tokens: divided by the weight in
   channel C; unit: L2-normalised), zeroed where mask (same layout as acc, 1 byte) is 0; out int8 (per-vector absmax scale -> scale fp16, or
   unit scale 127 when scale is nullptr), fp16 or fp32, [nz][ny][nx][C] ---- */
__global__ void k_finish(const float *__restrict__ acc, int any, int anx, int CA, int y0, int x0, int nz, int ny, int nx, int c0, int C, int wch, int unit,
                         const uint8_t *__restrict__ mask, int dtype, void *__restrict__ out, __half *__restrict__ scale, int unit8) {
    const long long cell = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5; const int lane = threadIdx.x & 31;
    if (cell >= (long long)nz * ny * nx) return;
    const int x = (int)(cell % nx), y = (int)((cell / nx) % ny), z = (int)(cell / ((long long)nx * ny));
    const long long ar = ((long long)z * any + y0 + y) * anx + x0 + x;
    const float *a = acc + ar * CA;
    const bool ok = mask[ar] != 0;
    const float dv = wch >= 0 ? fmaxf(a[wch], 1.17549435e-38f) : 1.f;
    float v[16]; int n = 0; float ss = 0.f, mx = 0.f;
    for (int c = lane; c < C; c += 32, n++) { v[n] = ok ? a[c0 + c] / dv : 0.f; ss += v[n] * v[n]; }
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float inv = unit ? 1.f / fmaxf(sqrtf(ss), 1e-12f) : 1.f;
    for (int i = 0; i < n; i++) { v[i] *= inv; mx = fmaxf(mx, fabsf(v[i])); }
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    n = 0;
    if (dtype == 0) {   /* int8 */
        float div = 1.f;
        if (!unit8) { const __half sc = __float2half_rn(mx / 127.f); if (lane == 0) scale[cell] = sc; const float sf = __half2float(sc); div = sf > 0.f ? sf : 1.f; }
        for (int c = lane; c < C; c += 32, n++) ((int8_t *)out)[cell * C + c] = (int8_t)fminf(fmaxf(rintf(unit8 ? v[n] * 127.f : v[n] / div), -127.f), 127.f);
    } else if (dtype == 1) for (int c = lane; c < C; c += 32, n++) ((__half *)out)[cell * C + c] = __float2half_rn(v[n]);
    else for (int c = lane; c < C; c += 32, n++) ((float *)out)[cell * C + c] = v[n];
}
extern "C" void dv_finish(const float *acc, int any, int anx, int CA, int y0, int x0, int nz, int ny, int nx, int c0, int C, int wch, int unit,
                          const uint8_t *mask, int dtype, void *out, void *scale, int unit8) {
    if (C > 512) { fprintf(stderr, "dv_finish: C %d\n", C); return; }
    const long long n = (long long)nz * ny * nx;
    if (n > 0) k_finish<<<nb_of(n * 32, 256), 256, 0, ST>>>(acc, any, anx, CA, y0, x0, nz, ny, nx, c0, C, wch, unit, mask, dtype, out, (__half *)scale, unit8);
}
extern "C" void dv_copy_dev(void *dst, const void *src, size_t bytes) { cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, ST); }
/* ---- fp32 residual stream: LayerNorm of fp32 rows -> fp16, and x += u (fp16 update) ---- */
__global__ void k_lnrow32(const float *__restrict__ x, const float *__restrict__ w, const float *__restrict__ b, float eps, __half *__restrict__ y, long long M, int C) {
    const long long row = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5; const int lane = threadIdx.x & 31;
    if (row >= M) return;
    const float *xr = x + row * C;
    float v[16]; int n = 0; float s = 0.f;
    for (int c = lane; c < C; c += 32) { v[n] = xr[c]; s += v[n]; n++; }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float mean = s / C; float q = 0.f;
    for (int i = 0; i < n; i++) { const float d = v[i] - mean; q += d * d; }
    for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffffu, q, o);
    const float rs = rsqrtf(q / C + eps);
    n = 0;
    for (int c = lane; c < C; c += 32, n++) y[row * C + c] = __float2half_rn((v[n] - mean) * rs * w[c] + b[c]);
}
extern "C" void dv_layernorm32(const float *x, const float *w, const float *b, float eps, void *y, long long M, int C) {
    k_lnrow32<<<nb_of(M * 32, 256), 256, 0, ST>>>(x, w, b, eps, (__half *)y, M, C);
}
__global__ void k_addh(float *__restrict__ x, const __half *__restrict__ u, long long n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) x[i] += __half2float(u[i]);
}
extern "C" void dv_add_half(float *x, const void *u, long long n) { k_addh<<<nb_of(n, 256), 256, 0, ST>>>(x, (const __half *)u, n); }
__global__ void k_h2f(float *__restrict__ x, const __half *__restrict__ u, long long n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) x[i] = __half2float(u[i]);
}
extern "C" void dv_half_to_float(float *x, const void *u, long long n) { k_h2f<<<nb_of(n, 256), 256, 0, ST>>>(x, (const __half *)u, n); }
extern "C" void dv_zero_dev(void *p, size_t bytes) { cudaMemsetAsync(p, 0, bytes, ST); }
/* dst[z][y][x] = src[z][y][x] for y < ny, x < nx (src rows of snx, planes of sny x snx) */
extern "C" void dv_mask_crop(const uint8_t *src, int sny, int snx, uint8_t *dst, int nz, int ny, int nx) {
    cudaMemcpy2DAsync(dst, (size_t)nx, src, (size_t)snx, (size_t)nx, (size_t)ny, cudaMemcpyDeviceToDevice, ST);   /* plane 0 */
    for (int z = 1; z < nz; z++) cudaMemcpy2DAsync(dst + (size_t)z * ny * nx, (size_t)nx, src + (size_t)z * sny * snx, (size_t)snx, (size_t)nx, (size_t)ny, cudaMemcpyDeviceToDevice, ST);
}
