/* xtsf (xtsf.h): the TimeSformer kernels of ufsm ink-ext (patch rows, token assembly, divided space-time attention with rotary
   embeddings and the cls token, GEGLU). The linear layers are xnet 1x1 convs over the token rows, the LayerNorms dv_layernorm32. */
#include "xtsf.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

#define ST ((cudaStream_t)xnet_stream())
static inline unsigned nb_of(long long n, int t) { long long b = (n + t - 1) / t; return (unsigned)(b > 2147483647LL ? 2147483647LL : b); }

__global__ void k_tsf_patches(const uint8_t *__restrict__ in, __half *__restrict__ out, int B, int F, int S, int P, float clip, float inv) {
    const int G = S / P, PP = P * P; const long long total = (long long)B * F * G * G * PP;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const int e = (int)(i % PP); long long t = i / PP; const int gx = (int)(t % G); t /= G; const int gy = (int)(t % G); t /= G;
        const int f = (int)(t % F); const long long b = t / F;
        const int y = gy * P + e / P, x = gx * P + e % P;
        const float u = in[((b * F + f) * S + y) * S + x];
        out[i] = __float2half_rn(fminf(u, clip) * inv);
    }
}
extern "C" xt xtsf_patches(const uint8_t *dev, int B, int F, int S, int P, float clip, float div) {
    const int G = S / P; xt y = xt_new(B, 1, 1, F * G * G, P * P);
    k_tsf_patches<<<nb_of((long long)xt_numel(y), 256), 256, 0, ST>>>(dev, (__half *)y.p, B, F, S, P, clip, 1.f / div);
    return y;
}

__global__ void k_tsf_tokens(const __half *__restrict__ emb, const float *__restrict__ cls, float *__restrict__ x, int B, int n, int C) {
    const long long total = (long long)B * (n + 1) * C;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total; i += (long long)gridDim.x * blockDim.x) {
        const int c = (int)(i % C); const long long r = i / C; const int t = (int)(r % (n + 1)); const long long b = r / (n + 1);
        x[i] = t ? __half2float(emb[(b * n + t - 1) * C + c]) : cls[c];
    }
}
extern "C" void xtsf_tokens(xt emb, const float *cls_dev, float *x_dev) {
    const int n = emb.w, C = emb.c;
    k_tsf_tokens<<<nb_of((long long)emb.n * (n + 1) * C, 256), 256, 0, ST>>>((const __half *)emb.p, cls_dev, x_dev, emb.n, n, C);
}

/* one warp per (b, h, query token); lane l holds dims 2l, 2l + 1 of the 64; online softmax over the keys */
__global__ void k_tsf_attn(const __half *__restrict__ qkv, __half *__restrict__ out, int B, int H, int F, int NP, int time,
                           const float *__restrict__ rs, const float *__restrict__ rc) {
    const long long wid = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5; const int lane = threadIdx.x & 31;
    const int N = 1 + F * NP, I = H * 64;
    if (wid >= (long long)B * H * N) return;
    const int qi = (int)(wid % N); const int h = (int)((wid / N) % H); const long long b = wid / ((long long)N * H);
    const __half *base = qkv + b * N * 3 * I; const int d0 = 2 * lane;
    const float sc = 0.125f;   /* 64^-0.5 */
    float q0 = __half2float(base[(long long)qi * 3 * I + h * 64 + d0]) * sc, q1 = __half2float(base[(long long)qi * 3 * I + h * 64 + d0 + 1]) * sc;
    int f = 0, p = 0;
    if (qi) {
        f = (qi - 1) / NP; p = (qi - 1) % NP; const int r = time ? f : p;
        const float c0 = rc[r * 64 + d0], c1 = rc[r * 64 + d0 + 1], s0 = rs[r * 64 + d0], s1 = rs[r * 64 + d0 + 1];
        const float a = q0 * c0 - q1 * s0, bb = q1 * c1 + q0 * s1; q0 = a; q1 = bb;
    }
    const int nk = qi ? 1 + (time ? F : NP) : N;
    float m = -INFINITY, l = 0.f, a0 = 0.f, a1 = 0.f;
    for (int j = 0; j < nk; j++) {
        int kt, r = -1;
        if (!qi) kt = j;
        else if (!j) kt = 0;
        else if (time) { kt = 1 + (j - 1) * NP + p; r = j - 1; }
        else { kt = 1 + f * NP + (j - 1); r = j - 1; }
        const __half *kp = base + (long long)kt * 3 * I + I + h * 64 + d0;
        float k0 = __half2float(kp[0]), k1 = __half2float(kp[1]);
        if (r >= 0) {
            const float c0 = rc[r * 64 + d0], c1 = rc[r * 64 + d0 + 1], s0 = rs[r * 64 + d0], s1 = rs[r * 64 + d0 + 1];
            const float a = k0 * c0 - k1 * s0, bb = k1 * c1 + k0 * s1; k0 = a; k1 = bb;
        }
        float s = q0 * k0 + q1 * k1;
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        const __half *vp = kp + I;
        const float v0 = __half2float(vp[0]), v1 = __half2float(vp[1]);
        const float mn = fmaxf(m, s), cor = __expf(m - mn), e = __expf(s - mn);
        l = l * cor + e; a0 = a0 * cor + e * v0; a1 = a1 * cor + e * v1; m = mn;
    }
    __half *o = out + (b * N + qi) * I + h * 64 + d0;
    o[0] = __float2half_rn(a0 / l); o[1] = __float2half_rn(a1 / l);
}
extern "C" void xtsf_attn(xt qkv, xt out, int heads, int F, int NP, int time, const float *rot_sin, const float *rot_cos) {
    const long long warps = (long long)qkv.n * heads * (1 + F * NP);
    k_tsf_attn<<<nb_of(warps * 32, 256), 256, 0, ST>>>((const __half *)qkv.p, (__half *)out.p, qkv.n, heads, F, NP, time, rot_sin, rot_cos);
}

__global__ void k_tsf_geglu(const __half *__restrict__ h, __half *__restrict__ y, long long M, int Fh) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < M * Fh; i += (long long)gridDim.x * blockDim.x) {
        const long long r = i / Fh; const int c = (int)(i % Fh);
        const float x = __half2float(h[r * 2 * Fh + c]), g = __half2float(h[r * 2 * Fh + Fh + c]);
        y[i] = __float2half_rn(x * 0.5f * g * (1.f + erff(g * 0.70710678118654752f)));
    }
}
extern "C" xt xtsf_geglu(xt h) {
    const int Fh = h.c / 2; xt y = xt_new(h.n, h.d, h.h, h.w, Fh); const long long M = (long long)h.n * h.d * h.h * h.w;
    k_tsf_geglu<<<nb_of(M * Fh, 256), 256, 0, ST>>>((const __half *)h.p, (__half *)y.p, M, Fh);
    return y;
}
