/* Kernels of ufsm surf-eval (src/surfeval.c): LeakyReLU, 2x2x2 / stride-2 transposed-convolution re-layouts, average pooling, the
   upstream DC_SkelREC_and_CE_loss (2 classes, ignore label 2, foreground-only soft Dice and skeleton recall per sample, smooth 1e-5,
   masked mean CE) and its validation counts, fp16 -> fp32 copies. All on the default stream (as nn_*). Tensors [N][C][D][H][W] fp32. */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

static inline unsigned nbk(long long n, int b) { return (unsigned)((n + b - 1) / b); }

__global__ void k_lrelu(const float *x, float *y, long long n, float s) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i < n) { const float v = x[i]; y[i] = v > 0.f ? v : s * v; }
}
__global__ void k_lrelu_bwd(const float *x, float *g, long long n, float s) {   /* g *= f'(x) in place (x: the pre-activation) */
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i < n && !(x[i] > 0.f)) g[i] *= s;
}
extern "C" void se_lrelu(const float *x, float *y, long long n, float s) { k_lrelu<<<nbk(n, 256), 256>>>(x, y, n, s); }
extern "C" void se_lrelu_bwd(const float *x, float *g, long long n, float s) { k_lrelu_bwd<<<nbk(n, 256), 256>>>(x, g, n, s); }

/* transposed conv k2 s2 as 8 GEMMs: tap t = (a, b, c) of the 2^3 output sub-lattice. scatter: tmp [N][8][Co][V] (V = d h w of the input)
   -> y [N][Co][2d][2h][2w] + bias; gather (the backward): gy -> tmp layout */
__global__ void k_tc_scatter(const float *tmp, const float *bias, float *y, int N, int Co, int d, int h, int w) {
    const long long V = (long long)d * h * w, tot = (long long)N * 8 * Co * V;
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= tot) return;
    long long r = i; const int v = (int)(r % V); r /= V; const int co = (int)(r % Co); r /= Co; const int t = (int)(r % 8); const int n = (int)(r / 8);
    const int z = v / (h * w), yy = v / w % h, x = v % w, a = t >> 2, b = (t >> 1) & 1, c = t & 1;
    y[(((size_t)n * Co + co) * 2 * d + 2 * z + a) * 4 * (size_t)h * w + (size_t)(2 * yy + b) * 2 * w + 2 * x + c] = tmp[i] + (bias ? bias[co] : 0.f);
}
__global__ void k_tc_gather(const float *gy, float *tmp, int N, int Co, int d, int h, int w) {
    const long long V = (long long)d * h * w, tot = (long long)N * 8 * Co * V;
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= tot) return;
    long long r = i; const int v = (int)(r % V); r /= V; const int co = (int)(r % Co); r /= Co; const int t = (int)(r % 8); const int n = (int)(r / 8);
    const int z = v / (h * w), yy = v / w % h, x = v % w, a = t >> 2, b = (t >> 1) & 1, c = t & 1;
    tmp[i] = gy[(((size_t)n * Co + co) * 2 * d + 2 * z + a) * 4 * (size_t)h * w + (size_t)(2 * yy + b) * 2 * w + 2 * x + c];
}
extern "C" void se_tc_scatter(const float *tmp, const float *bias, float *y, int N, int Co, int d, int h, int w) {
    const long long tot = (long long)N * 8 * Co * d * h * w; k_tc_scatter<<<nbk(tot, 256), 256>>>(tmp, bias, y, N, Co, d, h, w); }
extern "C" void se_tc_gather(const float *gy, float *tmp, int N, int Co, int d, int h, int w) {
    const long long tot = (long long)N * 8 * Co * d * h * w; k_tc_gather<<<nbk(tot, 256), 256>>>(gy, tmp, N, Co, d, h, w); }
/* bias gradient of the transposed conv: gb[co] += sum over n, taps, voxels of tmp [N][8][Co][V] (one block per co) */
__global__ void k_tc_gbias(const float *tmp, float *gb, int N, int Co, long long V) {
    const int co = blockIdx.x; float s = 0.f;
    for (int nt = 0; nt < N * 8; nt++) for (long long v = threadIdx.x; v < V; v += blockDim.x) s += tmp[((size_t)nt * Co + co) * V + v];
    __shared__ float sh[256]; sh[threadIdx.x] = s; __syncthreads();
    for (int o = 128; o; o >>= 1) { if ((int)threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o]; __syncthreads(); }
    if (!threadIdx.x) gb[co] += sh[0];
}
extern "C" void se_tc_gbias(const float *tmp, float *gb, int N, int Co, long long V) { k_tc_gbias<<<Co, 256>>>(tmp, gb, N, Co, V); }

/* fp16 [S][C] (channel-last, S = g^3) -> fp32 [C][S] into channel block c0 of a ctot-channel tensor of sample n */
__global__ void k_h2f_cl(const __half *src, float *dst, long long S, int C) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= S * C) return;
    const long long s = i / C; const int c = (int)(i % C); dst[(size_t)c * S + s] = __half2float(src[i]);
}
extern "C" void se_h2f_cl(const void *src, float *dst, long long S, int C) { k_h2f_cl<<<nbk(S * C, 256), 256>>>((const __half *)src, dst, S, C); }

/* the loss. pass 1 per sample n: sums over its voxels (mask = label != 2): [0] CE sum, [1] mask count, [2] I = sum p y, [3] Sp = sum p,
   [4] Sg = sum y, [5] Ir = sum p skel, [6] Sk = sum skel; p = softmax(l)[1] */
__global__ void k_loss_sums(const float *lg, const uint8_t *lab, const uint8_t *sk, long long V, double *sums) {
    const int n = blockIdx.y; double a[7] = {0, 0, 0, 0, 0, 0, 0};
    for (long long v = blockIdx.x * (long long)blockDim.x + threadIdx.x; v < V; v += (long long)gridDim.x * blockDim.x) {
        const uint8_t L = lab[(size_t)n * V + v]; if (L == 2) continue;
        const float l0 = lg[((size_t)n * 2) * V + v], l1 = lg[((size_t)n * 2 + 1) * V + v];
        const float d = l1 - l0, p = 1.f / (1.f + __expf(-d));
        const float y = L == 1 ? 1.f : 0.f, s = sk[(size_t)n * V + v] ? 1.f : 0.f;
        const float lse = fmaxf(l0, l1) + log1pf(__expf(-fabsf(d)));   /* log(e^l0 + e^l1) */
        a[0] += lse - (L == 1 ? l1 : l0); a[1] += 1; a[2] += p * y; a[3] += p; a[4] += y; a[5] += p * s; a[6] += s;
    }
    for (int k = 0; k < 7; k++) {
        double v = a[k];
        for (int o = 16; o; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if ((threadIdx.x & 31) == 0) atomicAdd(&sums[n * 8 + k], v);
    }
}
/* pass 2: d loss / d logits. L = CE_sum / count_all + mean_n (-dice_n) + mean_n (-rec_n), dice_n = (2 I + s) / (Sp + Sg + s),
   rec_n = (Ir + s) / (Sk + s) */
__global__ void k_loss_grad(const float *lg, const uint8_t *lab, const uint8_t *sk, long long V, int N, const double *sums, double cnt_all, float sm, float *g) {   /* N: the mean's sample count (accumulated crops) */
    const int n = blockIdx.y;
    const double I = sums[n * 8 + 2], Sp = sums[n * 8 + 3], Sg = sums[n * 8 + 4], Sk = sums[n * 8 + 6], D = Sp + Sg + sm;
    for (long long v = blockIdx.x * (long long)blockDim.x + threadIdx.x; v < V; v += (long long)gridDim.x * blockDim.x) {
        const uint8_t L = lab[(size_t)n * V + v]; float g0 = 0.f, g1 = 0.f;
        if (L != 2) {
            const float l0 = lg[((size_t)n * 2) * V + v], l1 = lg[((size_t)n * 2 + 1) * V + v], p = 1.f / (1.f + __expf(-(l1 - l0)));
            const float y = L == 1 ? 1.f : 0.f, s = sk[(size_t)n * V + v] ? 1.f : 0.f;
            const double dp = -(2.0 * y / D - (2.0 * I + sm) / (D * D)) / N - (double)s / (Sk + sm) / N;   /* d L / d p */
            const float dl1 = (float)(dp * p * (1.f - p)) + (p - y) / (float)cnt_all;   /* + CE: (softmax - onehot) / count */
            g1 = dl1; g0 = -dl1;
        }
        g[((size_t)n * 2) * V + v] = g0; g[((size_t)n * 2 + 1) * V + v] = g1;
    }
}
extern "C" void se_loss_sums(const float *lg, const uint8_t *lab, const uint8_t *sk, int N, long long V, double *sums) {
    cudaMemsetAsync(sums, 0, (size_t)N * 8 * sizeof(double), 0); k_loss_sums<<<dim3(256, N), 256>>>(lg, lab, sk, V, sums); }
extern "C" void se_loss_grad(const float *lg, const uint8_t *lab, const uint8_t *sk, int N, int Nmean, long long V, const double *sums, double cnt_all, float sm, float *g) {
    k_loss_grad<<<dim3(256, N), 256>>>(lg, lab, sk, V, Nmean, sums, cnt_all, sm, g); }
/* validation counts (threshold 0.5, valid voxels): [0] pred & truth, [1] pred, [2] truth, [3] pred & medial, [4] medial */
__global__ void k_val_counts(const float *lg, const uint8_t *lab, const uint8_t *sk, long long V, unsigned long long *c) {
    const int n = blockIdx.y; unsigned long long a[5] = {0, 0, 0, 0, 0};
    for (long long v = blockIdx.x * (long long)blockDim.x + threadIdx.x; v < V; v += (long long)gridDim.x * blockDim.x) {
        const uint8_t L = lab[(size_t)n * V + v]; if (L == 2) continue;
        const int pr = lg[((size_t)n * 2 + 1) * V + v] >= lg[((size_t)n * 2) * V + v], tr = L == 1, md = sk[(size_t)n * V + v] != 0;
        a[0] += pr && tr; a[1] += pr; a[2] += tr; a[3] += pr && md; a[4] += md;
    }
    for (int k = 0; k < 5; k++) atomicAdd(&c[k], a[k]);
}
extern "C" void se_val_counts(const float *lg, const uint8_t *lab, const uint8_t *sk, int N, long long V, unsigned long long *c) { k_val_counts<<<dim3(256, N), 256>>>(lg, lab, sk, V, c); }

/* AdamW (torch semantics: decoupled decay p -= lr wd p, bias-corrected moments) with a global gradient-norm clip scale */
__global__ void k_adamw(float *p, const float *g, float *m, float *v, long long n, float lr, float b1, float b2, float eps, float wd, float c1, float c2, float gs) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i >= n) return;
    const float gi = g[i] * gs; p[i] *= 1.f - lr * wd;
    m[i] = b1 * m[i] + (1.f - b1) * gi; v[i] = b2 * v[i] + (1.f - b2) * gi * gi;
    p[i] -= lr * (m[i] / c1) / (sqrtf(v[i] / c2) + eps);
}
extern "C" void se_adamw(float *p, const float *g, float *m, float *v, long long n, float lr, float b1, float b2, float eps, float wd, int step, float gs) {
    k_adamw<<<nbk(n, 256), 256>>>(p, g, m, v, n, lr, b1, b2, eps, wd, 1.f - powf(b1, (float)step), 1.f - powf(b2, (float)step), gs); }
__global__ void k_sumsq(const float *g, long long n, double *out) {
    double s = 0; for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) s += (double)g[i] * g[i];
    for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(out, s);
}
extern "C" void se_sumsq(const float *g, long long n, double *out) { cudaMemsetAsync(out, 0, 8, 0); k_sumsq<<<512, 256>>>(g, n, out); }
/* per-channel bias of [N][C][V]: add, and its gradient (gb += sums; one block per channel) */
__global__ void k_bias_add(float *y, const float *b, int C, long long V, long long tot) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (i < tot) y[i] += b[(i / V) % C];
}
extern "C" void se_bias_add(float *y, const float *b, int N, int C, long long V) { const long long t = (long long)N * C * V; k_bias_add<<<nbk(t, 256), 256>>>(y, b, C, V, t); }
__global__ void k_bias_grad(const float *g, float *gb, int N, int C, long long V) {
    const int c = blockIdx.x; float s = 0.f;
    for (int n = 0; n < N; n++) for (long long v = threadIdx.x; v < V; v += blockDim.x) s += g[((size_t)n * C + c) * V + v];
    __shared__ float sh[256]; sh[threadIdx.x] = s; __syncthreads();
    for (int o = 128; o; o >>= 1) { if ((int)threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o]; __syncthreads(); }
    if (!threadIdx.x) gb[c] += sh[0];
}
extern "C" void se_bias_grad(const float *g, float *gb, int N, int C, long long V) { k_bias_grad<<<C, 256>>>(g, gb, N, C, V); }
