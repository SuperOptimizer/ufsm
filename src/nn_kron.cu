/* Kronecker-factored conv preconditioner feeding Muon (train --muon-kron 1, 2026-11-13; DESIGN.md "Muon variants").
   Each 3^3 conv gradient G [Co][Ci][27] is preconditioned like a third-order tensor in Shampoo (Gupta, Koren & Singer 2018, the
   order-k form: one factor per tensor mode, exponent -1 / (2k) = -1/6) before Muon's momentum and Newton-Schulz:
     L1 (Co x Co) = EMA b2 of G_(1) G_(1)^T, L2 (Ci x Ci) = EMA of G_(2) G_(2)^T, L3 (27 x 27) = EMA of G_(3) G_(3)^T   (mode unfoldings)
     G <- G x1 P1 x2 P2 x3 P3,   Pk = (Lk + eps |Lk|_F I)^(-1/6)                                                          (mode products)
   Preconditioner memory per conv: 2 (Co^2 + Ci^2 + 27^2) floats (factors and their roots) instead of (Co Ci 27)^2 for a full
   matrix, and 2 (Co^2 + (Ci 27)^2) for Shampoo / SOAP on the unfolded [Co][Ci 27]. The roots are refreshed for the convs with
   c % every == step % every (round robin: the cost spreads evenly over the steps) and for every conv at the first step; they are
   coupled Newton iterations for the inverse p-th root (Guo & Higham 2006): X0 = I / s, M0 = A / s^p (s^p = |A|_F: eigenvalues of M0
   in (0, 1]), T = ((p + 1) I - M) / p, X <- X T, M <- T^p M; X -> A^(-1/p) as M -> I (quadratic near the end; NEWTON_IT steps).
   Every product is one strided fp32 tiled kernel (kg_k). The gradient is preconditioned in place (Muon then reads it as its g), one
   conv at a time through one scratch of the largest conv's size. */
#include "nn_common.cuh"

typedef struct {   /* C[m][n] = alpha sum_k A[m][k] B[k][n] + beta C[m][n]; contraction index k addresses (k / kq) h + (k % kq) l */
    const float *A, *B; float *C; int M, N, K, kq;
    long long lam, akh, akl, bkh, bkl, lbn, lcm, lcn;
    long long sza, szb, szc;   /* per blockIdx.z offsets of A, B, C (z-batched entries) */
    float alpha, beta;
} kg_t;
__global__ void __launch_bounds__(256) kg_k(const kg_t g) {
    const float *A = g.A + blockIdx.z * g.sza, *B = g.B + blockIdx.z * g.szb; float *C = g.C + blockIdx.z * g.szc;
    const int m0 = blockIdx.y * 64, n0 = blockIdx.x * 64; if (m0 >= g.M || n0 >= g.N) return;
    __shared__ __align__(16) float sl[16][68], sr[16][68];
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4; float acc[4][4] = {};
    for (int k0 = 0; k0 < g.K; k0 += 16) {
        for (int t = threadIdx.x; t < 1024; t += 256) {
            const int kk = t & 15, r = t >> 4, k = k0 + kk, m = m0 + r, n = n0 + r;
            const long long ka = (long long)(k / g.kq) * g.akh + (long long)(k % g.kq) * g.akl, kb = (long long)(k / g.kq) * g.bkh + (long long)(k % g.kq) * g.bkl;
            sl[kk][r] = m < g.M && k < g.K ? A[(long long)m * g.lam + ka] : 0.f;
            sr[kk][r] = n < g.N && k < g.K ? B[kb + (long long)n * g.lbn] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < 16; kk++) {
            const float4 l = *(const float4 *)&sl[kk][ty * 4], r = *(const float4 *)&sr[kk][tx * 4];
            const float lv[4] = {l.x, l.y, l.z, l.w}, rv[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += lv[i] * rv[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; i++) {
        const int m = m0 + ty * 4 + i; if (m >= g.M) continue;
        for (int j = 0; j < 4; j++) {
            const int n = n0 + tx * 4 + j; if (n >= g.N) continue;
            float *c = C + (long long)m * g.lcm + (long long)n * g.lcn; *c = g.alpha * acc[i][j] + (g.beta != 0.f ? g.beta * *c : 0.f);
        }
    }
}
static void kg(const kg_t &g, int nz = 1) { kg_k<<<dim3(nblk(g.N, 64), nblk(g.M, 64), nz), 256>>>(g); }
/* plain row-major n x n products: C = alpha A B + beta C */
static void sq(const float *A, const float *B, float *C, int n, float alpha = 1.f, float beta = 0.f) {
    kg_t g = {A, B, C, n, n, n, n, n, 0, 1, 0, n, 1, n, 1, 0, 0, 0, alpha, beta}; kg(g);
}
__global__ void kr_eye_k(float *X, int n, float d) { const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n * n) X[i] = i / n == i % n ? d : 0.f; }
__global__ void kr_damp_k(const float *L, float *M, int n, const float *fro, float eps, int p) {   /* M0 = (L + eps |L| I) / s^p, s^p = (1 + eps) |L| */
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n * n) return;
    const float f = fmaxf(*fro, 1e-30f); M[i] = (L[i] + (i / n == i % n ? eps * f : 0.f)) / ((1.f + eps) * f);
}
__global__ void kr_fro_k(const float *L, int n, float *out) {   /* one block: |L|_F */
    __shared__ float r[256]; float a = 0.f; for (int i = threadIdx.x; i < n * n; i += 256) a += L[i] * L[i];
    r[threadIdx.x] = a; __syncthreads(); for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) r[threadIdx.x] += r[threadIdx.x + o]; __syncthreads(); }
    if (!threadIdx.x) *out = sqrtf(r[0]);
}
__global__ void kr_t_k(const float *M, float *T, int n, int p) {   /* T = ((p + 1) I - M) / p */
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n * n) T[i] = ((i / n == i % n ? (float)(p + 1) : 0.f) - M[i]) / (float)p;
}
__global__ void kr_scale_k(float *X, int n, const float *fro, float eps, int p) {   /* X /= s: s = ((1 + eps) |L|)^(1/p) */
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n * n) X[i] /= powf((1.f + eps) * fmaxf(*fro, 1e-30f), 1.f / (float)p);
}
enum { KR_P = 6, NEWTON_IT = 24 };
/* P = (L + eps |L|_F I)^(-1/6); work >= 6 n^2 + 1 floats */
static void inv_root(const float *L, float *P, int n, float eps, float *w) {
    float *X = w, *M = X + n * n, *T = M + n * n, *T2 = T + n * n, *T3 = T2 + n * n, *T6 = T3 + n * n, *fro = T6 + n * n;
    const unsigned b = nblk((size_t)n * n, 256);
    kr_fro_k<<<1, 256>>>(L, n, fro); kr_damp_k<<<b, 256>>>(L, M, n, fro, eps, KR_P); kr_eye_k<<<b, 256>>>(X, n, 1.f);
    for (int it = 0; it < NEWTON_IT; it++) {
        kr_t_k<<<b, 256>>>(M, T, n, KR_P);
        sq(X, T, T2, n); cudaMemcpyAsync(X, T2, (size_t)n * n * 4, cudaMemcpyDeviceToDevice);   /* X <- X T */
        sq(T, T, T2, n); sq(T2, T, T3, n); sq(T3, T3, T6, n);                                   /* T^6 */
        sq(T6, M, T2, n); cudaMemcpyAsync(M, T2, (size_t)n * n * 4, cudaMemcpyDeviceToDevice);  /* M <- T^6 M */
    }
    kr_scale_k<<<b, 256>>>(X, n, fro, eps, KR_P); cudaMemcpyAsync(P, X, (size_t)n * n * 4, cudaMemcpyDeviceToDevice);
}
/* the three Gram matrices of G [Co][Ci][27] into L1 / L2 / L3 as EMAs: L = b2 L + (1 - b2) Gram */
static void grams(const float *G, int Co, int Ci, float *L1, float *L2, float *L3, float b2) {
    const int K = Ci * 27, a = 1.f - b2 > 0.f;
    const float al = a ? 1.f - b2 : 1.f, be = a ? b2 : 0.f;
    kg(kg_t{G, G, L1, Co, Co, K, K, (long long)K, 0, 1, 0, 1, (long long)K, (long long)Co, 1, 0, 0, 0, al, be});   /* L1[o][o'] = sum_k G[o][k] G[o'][k] */
    /* L2[i][i'] = sum_(o, t) G[o][i][t] G[o][i'][t]: contraction k = o 27 + t -> o Ci 27 + t */
    kg(kg_t{G, G, L2, Ci, Ci, Co * 27, 27, 27, (long long)Ci * 27, 1, (long long)Ci * 27, 1, 27, (long long)Ci, 1, 0, 0, 0, al, be});
    /* L3[t][t'] = sum_(o, i) G[(o Ci + i) 27 + t] G[(o Ci + i) 27 + t'] */
    kg(kg_t{G, G, L3, 27, 27, Co * Ci, Co * Ci, 1, 0, 27, 0, 27, 1, 27, 1, 0, 0, 0, al, be});
}
/* G <- G x1 P1 x2 P2 x3 P3 in place; S: Co Ci 27 floats of scratch */
static void precond(float *G, int Co, int Ci, const float *P1, const float *P2, const float *P3, float *S) {
    const int K = Ci * 27; const long long CK = (long long)Ci * 27;
    kg(kg_t{P1, G, S, Co, K, Co, Co, (long long)Co, 0, 1, 0, (long long)K, 1, (long long)K, 1, 0, 0, 0, 1.f, 0.f});   /* S = P1 G (mode 1) */
    /* mode 2, one entry per tap t (z): G[o][i][t] = sum_i' S[o][i'][t] P2[i][i'] (P2 symmetric: B[i'][i] = P2[i'][i]) */
    kg(kg_t{S, P2, G, Co, Ci, Ci, Ci, CK, 0, 27, 0, (long long)Ci, 1, CK, 27, 1, 0, 1, 1.f, 0.f}, 27);
    /* mode 3: S[(o, i)][t] = sum_t' G[(o, i)][t'] P3[t'][t] */
    kg(kg_t{G, P3, S, Co * Ci, 27, 27, 27, 27, 0, 1, 0, 27, 1, 27, 1, 0, 0, 0, 1.f, 0.f});
    cudaMemcpyAsync(G, S, (size_t)Co * K * 4, cudaMemcpyDeviceToDevice);
}

struct nn_kron { int n, maxn; int *Co, *Ci; float **g, **L, **P; float *scr, *work; long long step; };
extern "C" nn_kron *nn_kron_create(int n, float *const *g, const int *Co, const int *Ci) {
    nn_kron *k = (nn_kron *)calloc(1, sizeof *k); k->n = n; size_t mx = 0, st = 0; int mn = 27;
    k->Co = (int *)malloc(n * sizeof(int)); k->Ci = (int *)malloc(n * sizeof(int)); k->g = (float **)malloc(n * sizeof(float *));
    k->L = (float **)malloc(3 * n * sizeof(float *)); k->P = (float **)malloc(3 * n * sizeof(float *));
    for (int i = 0; i < n; i++) {
        k->Co[i] = Co[i]; k->Ci[i] = Ci[i]; k->g[i] = g[i]; const size_t sz = (size_t)Co[i] * Ci[i] * 27; if (sz > mx) mx = sz;
        st += (size_t)Co[i] * Co[i] + (size_t)Ci[i] * Ci[i] + 729; mn = Co[i] > mn ? Co[i] : mn; mn = Ci[i] > mn ? Ci[i] : mn;
    }
    float *L = (float *)nn_malloc(st * 4), *P = (float *)nn_malloc(st * 4); nn_zero(L, st * 4); nn_zero(P, st * 4);
    for (int i = 0, o = 0; i < n; i++) {
        const int d[3] = {Co[i], Ci[i], 27};
        for (int f = 0; f < 3; f++) { k->L[3 * i + f] = L + o; k->P[3 * i + f] = P + o; o += d[f] * d[f]; }
    }
    k->maxn = mn; k->scr = (float *)nn_malloc(mx * 4); k->work = (float *)nn_malloc(((size_t)6 * mn * mn + 1) * 4);
    return k;
}
extern "C" size_t nn_kron_bytes(const nn_kron *k) {
    size_t st = 0, mx = 0; for (int i = 0; i < k->n; i++) { st += (size_t)k->Co[i] * k->Co[i] + (size_t)k->Ci[i] * k->Ci[i] + 729; const size_t s = (size_t)k->Co[i] * k->Ci[i] * 27; if (s > mx) mx = s; }
    return 4 * (2 * st + mx + 6 * (size_t)k->maxn * k->maxn + 1);
}
extern "C" void nn_kron_free(nn_kron *k) {
    if (!k) return; nn_free(k->L[0]); nn_free(k->P[0]); nn_free(k->scr); nn_free(k->work);
    free(k->Co); free(k->Ci); free(k->g); free(k->L); free(k->P); free(k);
}
/* one step: the factor EMAs from the raw gradients, the roots of this step's round-robin share, the gradients preconditioned in place */
extern "C" void nn_kron_step(nn_kron *k, float b2, float eps, int every) {
    const long long s = k->step++;
    for (int i = 0; i < k->n; i++) {
        const int Co = k->Co[i], Ci = k->Ci[i];
        grams(k->g[i], Co, Ci, k->L[3 * i], k->L[3 * i + 1], k->L[3 * i + 2], s ? b2 : 0.f);   /* (the first step: the Grams themselves) */
        if (!s || (every > 0 && i % every == s % every)) {
            const int d[3] = {Co, Ci, 27};
            for (int f = 0; f < 3; f++) inv_root(k->L[3 * i + f], k->P[3 * i + f], d[f], eps, k->work);
        }
        precond(k->g[i], Co, Ci, k->P[3 * i], k->P[3 * i + 1], k->P[3 * i + 2], k->scr);
    }
    KCHECK();
}
/* test hooks: the root / product kernels on caller buffers */
extern "C" void nn_kron_inv_root(const float *L, float *P, int n, float eps, float *work) { inv_root(L, P, n, eps, work); KCHECK(); }
extern "C" void nn_kron_grams(const float *G, int Co, int Ci, float *L1, float *L2, float *L3) { grams(G, Co, Ci, L1, L2, L3, 0.f); KCHECK(); }
extern "C" void nn_kron_precond(float *G, int Co, int Ci, const float *P1, const float *P2, const float *P3, float *S) { precond(G, Co, Ci, P1, P2, P3, S); KCHECK(); }
