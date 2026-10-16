/* CUDA ops: opt section of the former nn.cu */
#include "nn_common.cuh"

__global__ void adamw_k(float *p, const float *g, float *m, float *v, size_t n, float lr, float b1, float b2, float eps, float wd, float c1, float c2) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    float mi = m[i] = b1 * m[i] + (1.f - b1) * gi;
    float vi = v[i] = b2 * v[i] + (1.f - b2) * gi * gi;
    float mh = mi / c1, vh = vi / c2;
    p[i] -= lr * (mh / (sqrtf(vh) + eps) + wd * p[i]);
}
__global__ void mm_xxt_k(const float *X, int Co, int K, float *A) {   /* A = X X^T (Co x Co) */
    int i = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= Co) return;
    const float *a = X + (size_t)i * K, *b = X + (size_t)j * K; float s = 0.f;
    for (int k = 0; k < K; k++) s += a[k] * b[k];
    A[(size_t)i * Co + j] = s;
}
__global__ void mm_sq_k(const float *A, int Co, float b, float c, float *B) {   /* B = b A + c A A (Co x Co) */
    int i = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= Co) return;
    float s = 0.f;
    for (int k = 0; k < Co; k++) s += A[(size_t)i * Co + k] * A[(size_t)k * Co + j];
    B[(size_t)i * Co + j] = b * A[(size_t)i * Co + j] + c * s;
}
__global__ void mm_bx_k(const float *B, const float *X, int Co, int K, float a, float *Y) {   /* Y = a X + B X (Co x K) */
    int i = blockIdx.y; size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= (size_t)K) return;
    float s = 0.f;
    for (int j = 0; j < Co; j++) s += B[(size_t)i * Co + j] * X[(size_t)j * K + k];
    Y[(size_t)i * K + k] = a * X[(size_t)i * K + k] + s;
}
__global__ void muon_mom_k(const float *g, float *mom, float *x, size_t n, float beta) {   /* nesterov momentum: mom = beta mom + g; x = g + beta mom */
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i >= n) return;
    float m = beta * mom[i] + g[i]; mom[i] = m; x[i] = g[i] + beta * m;
}
__global__ void muon_sumsq_k(const float *x, size_t n, double *ss) {   /* block-reduced sum of squares into one double */
    __shared__ float r[256]; float a = 0.f; for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) a += x[i] * x[i];
    r[threadIdx.x] = a; __syncthreads(); for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) r[threadIdx.x] += r[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(ss, (double)r[0]);
}
__global__ void muon_scale_k(float *x, size_t n, const double *ss, float eps) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) x[i] *= (float)(1.0 / (sqrt(*ss) + eps)); }
__global__ void muon_apply_k(float *p, const float *o, size_t n, float lr, float scale, float wd) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) p[i] = p[i] * (1.f - lr * wd) - lr * scale * o[i]; }
extern "C" void nn_muon(float *p, const float *g, float *mom, int Co, int K, float lr, float beta, float wd, float *work) {
    size_t n = (size_t)Co * K;
    float *X = work, *Y = work + n, *A = work + 2 * n, *B = A + (size_t)Co * Co;
    muon_mom_k<<<nblk(n, 256), 256>>>(g, mom, X, n, beta);
    double *ss = gn_dsums(1); cudaMemsetAsync(ss, 0, sizeof(double));
    muon_sumsq_k<<<nblk(n, 256) > 64 ? 64 : nblk(n, 256), 256>>>(X, n, ss);
    muon_scale_k<<<nblk(n, 256), 256>>>(X, n, ss, 1e-7f);
    const float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    dim3 gco(nblk(Co, 128), Co), gk(nblk(K, 256), Co);
    for (int it = 0; it < 5; it++) {
        mm_xxt_k<<<gco, 128>>>(X, Co, K, A);
        mm_sq_k<<<gco, 128>>>(A, Co, b, c, B);
        mm_bx_k<<<gk, 256>>>(B, X, Co, K, a, Y);
        float *t = X; X = Y; Y = t;
    }
    float scale = sqrtf(fmaxf(1.f, (float)Co / (float)K));
    muon_apply_k<<<nblk(n, 256), 256>>>(p, X, n, lr, scale, wd);
    KCHECK();
}
__global__ void bmuon_mom_k(const muon_desc_t *d, float beta, double *ss) {
    const muon_desc_t D = d[blockIdx.z]; size_t n = (size_t)D.Co * D.K;
    __shared__ float r[256]; float a = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { float m = beta * D.mom[i] + D.g[i]; D.mom[i] = m; float x = D.g[i] + beta * m; D.X[i] = x; a += x * x; }
    r[threadIdx.x] = a; __syncthreads(); for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) r[threadIdx.x] += r[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(&ss[blockIdx.z], (double)r[0]);
}
__global__ void bmuon_scale_k(const muon_desc_t *d, const double *ss) {
    const muon_desc_t D = d[blockIdx.z]; size_t n = (size_t)D.Co * D.K; float inv = (float)(1.0 / (sqrt(ss[blockIdx.z]) + 1e-7));
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) D.X[i] *= inv;
}
__global__ void bmuon_apply_k(const muon_desc_t *d, int swap, float lr, float wd) {
    const muon_desc_t D = d[blockIdx.z]; const float *O = swap ? D.Y : D.X; size_t n = (size_t)D.Co * D.K;
    float scale = sqrtf(fmaxf(1.f, (float)D.Co / (float)D.K));
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) D.p[i] = D.p[i] * (1.f - lr * wd) - lr * scale * O[i];
}
/* 5 Newton-Schulz iterations (modded-nanogpt coefficients) on every table entry's X (Co x K); returns the buffer flag of the
   result (1: Y) */
static int ns5_batch(const muon_desc_t *d, int nconv, int maxco, int maxk) {
    dim3 gco(nblk(maxco, 64), nblk(maxco, 64), nconv), gk(nblk(maxk, 64), nblk(maxco, 64), nconv);
    const float a = 3.4445f, b = -4.7750f, c = 2.0315f;
    int swap = 0;
    for (int it = 0; it < 5; it++) {
        gemm_ns_k<0, muon_desc_t><<<gco, 256>>>(d, swap, 0.f, 0.f);
        gemm_ns_k<1, muon_desc_t><<<gco, 256>>>(d, swap, b, c);
        gemm_ns_k<2, muon_desc_t><<<gk, 256>>>(d, swap, a, 0.f);
        swap ^= 1;
    }
    return swap;
}
extern "C" void nn_muon_batch(const void *descs, int nconv, int maxco, int maxk, float lr, float beta, float wd) {
    const muon_desc_t *d = (const muon_desc_t *)descs;
    double *ss = gn_dsums((size_t)nconv); cudaMemsetAsync(ss, 0, (size_t)nconv * sizeof(double));
    dim3 g1(32, 1, nconv);
    bmuon_mom_k<<<g1, 256>>>(d, beta, ss);
    bmuon_scale_k<<<g1, 256>>>(d, ss);
    const int swap = ns5_batch(d, nconv, maxco, maxk);
    bmuon_apply_k<<<g1, 256>>>(d, swap, lr, wd);
    KCHECK();
}
/* ---- Muon-C (Qing & Li 2026, arXiv 2609.09676) and Hyperball (Wen, Dang, Lyu, Ma & Liang 2026, arXiv 2606.16899) ----
   Muon-C: the Nesterov momentum of a k^3 kernel [Co][Ci][k^3] goes through the orthonormal DFT over the three kernel axes
   (the critical Fourier grid: one Co x Ci channel-transfer block per kernel frequency), every block is polarised on its
   own by complex Newton-Schulz (the paper: 5 iterations with Muon's a, b, c, X0 = B / (|B|_F + 1e-8), a block with more
   rows than columns polarised as its conjugate transpose), and the inverse DFT returns a real update on the original
   support. A real kernel has conjugate-symmetric blocks and polar(conj B) = conj polar(B), so a 3^3 kernel polarises 14
   of its 27 frequencies (0 and one of each conjugate pair). A complex block X = Xr + i Xi is iterated as its real
   embedding [[Xr, -Xi], [Xi, Xr]] (2m x 2n) with Muon's batched products: the embedding is a *-homomorphism, so the real
   iteration is the complex one exactly. The applied update is rescaled to the RMS exact-polar unfolded Muon has for the
   conv (sqrt(max(1, Co/K)) sqrt(min(Co, K) / (Co K)), K = Ci k^3): the paper's matched applied-update RMS, so --muon-lr
   keeps its meaning. Frequencies run one at a time over all convs, so the embedded scratch is one frequency's.
   Hyperball: W <- R Normalize(W - eta R Normalize(u)) with R the conv's |W|_F when the wrapper first ran and eta the
   Hyperball learning rate; it replaces the decay. u is the Muon or Muon-C direction (its scale cancels).
   Per-conv sums use 32 fixed partials summed in order (deterministic). */
typedef struct { const float *U; float *O; float *EX, *EY; int Co, Ci, k, m, n, tr; } muonc_desc_t;
__constant__ float c_mc_cs[2][14][27];   /* cos / sin of 2 pi (f . t) / 3: the 14 polarised frequencies f of a 3^3 grid, taps t */
__constant__ float c_mc_w[14];           /* 1: the zero frequency, 2: a conjugate pair */
static void mc_tables(void) {
    static int done[16];
    if (done[cur_dev()]) return;
    float cs[2][14][27], w[14]; int nf = 0;
    for (int f = 0; f < 27; f++) {
        const int p = f / 9, q = f / 3 % 3, r = f % 3, neg = (3 - p) % 3 * 9 + (3 - q) % 3 * 3 + (3 - r) % 3;
        if (f > neg) continue;
        for (int t = 0; t < 27; t++) {
            const double ph = 2.0 * M_PI * ((p * (t / 9) + q * (t / 3 % 3) + r * (t % 3)) % 3) / 3.0;
            cs[0][nf][t] = (float)cos(ph); cs[1][nf][t] = (float)sin(ph);
        }
        w[nf++] = f == neg ? 1.f : 2.f;
    }
    CK(cudaMemcpyToSymbol(c_mc_cs, cs, sizeof cs)); CK(cudaMemcpyToSymbol(c_mc_w, w, sizeof w));
    done[cur_dev()] = 1;
}
__device__ __forceinline__ void mc_part(double a, double *part) {   /* block sum into part[conv * 32 + block] */
    __shared__ double r[256];
    r[threadIdx.x] = a; __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) r[threadIdx.x] += r[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) part[blockIdx.y * 32 + blockIdx.x] = r[0];
}
__device__ __forceinline__ double mc_total(const double *part) { double s = 0; for (int j = 0; j < 32; j++) s += part[blockIdx.y * 32 + j]; return s; }
__global__ void muonc_dft_k(const muonc_desc_t *d, int f, double *part) {   /* the frequency-f block of every conv, embedded */
    const muonc_desc_t D = d[blockIdx.y];
    const int T = D.k * D.k * D.k, ld = 2 * D.n; const size_t np = (size_t)D.Co * D.Ci; const float nrm = rsqrtf((float)T);
    double a = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < np; i += (size_t)gridDim.x * blockDim.x) {
        const float *u = D.U + i * T; float re = 0.f, im = 0.f;
        if (T == 1) re = u[0];
        else for (int t = 0; t < 27; t++) { re += u[t] * c_mc_cs[0][f][t]; im -= u[t] * c_mc_cs[1][f][t]; }
        re *= nrm; im *= nrm;
        int r = (int)(i / D.Ci), c = (int)(i % D.Ci);
        if (D.tr) { const int t = r; r = c; c = t; im = -im; }   /* the conjugate transpose: B~[ci][co] = conj B[co][ci] */
        D.EX[(size_t)r * ld + c] = re; D.EX[(size_t)r * ld + D.n + c] = -im;
        D.EX[(size_t)(D.m + r) * ld + c] = im; D.EX[(size_t)(D.m + r) * ld + D.n + c] = re;
        a += (double)re * re + (double)im * im;
    }
    mc_part(a, part);
}
__global__ void muonc_norm_k(const muonc_desc_t *d, const double *part) {   /* X0 = B / (|B|_F + 1e-8), the complex norm */
    const muonc_desc_t D = d[blockIdx.y]; const float inv = (float)(1.0 / (sqrt(mc_total(part)) + 1e-8));
    const size_t n = (size_t)4 * D.m * D.n;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) D.EX[i] *= inv;
}
__global__ void muonc_idft_k(const muonc_desc_t *d, int f, int swap, int first) {   /* O (+)= the frequency's share of the inverse DFT */
    const muonc_desc_t D = d[blockIdx.y]; const float *E = swap ? D.EY : D.EX;
    const int T = D.k * D.k * D.k, ld = 2 * D.n; const size_t np = (size_t)D.Co * D.Ci;
    const float sc = (T == 1 ? 1.f : c_mc_w[f]) * rsqrtf((float)T);
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < np; i += (size_t)gridDim.x * blockDim.x) {
        int r = (int)(i / D.Ci), c = (int)(i % D.Ci);
        if (D.tr) { const int t = r; r = c; c = t; }
        const float pr = E[(size_t)r * ld + c], pi = D.tr ? -E[(size_t)(D.m + r) * ld + c] : E[(size_t)(D.m + r) * ld + c];
        float *o = D.O + i * T;
        if (T == 1) o[0] = (first ? 0.f : o[0]) + sc * pr;
        else for (int t = 0; t < 27; t++) o[t] = (first ? 0.f : o[t]) + sc * (pr * c_mc_cs[0][f][t] - pi * c_mc_cs[1][f][t]);
    }
}
__global__ void mopt_sumsq_k(const muon_desc_t *d, int which, double *part) {   /* which: 0 X, 1 Y, -1 the weights */
    const muon_desc_t D = d[blockIdx.y]; const float *x = which < 0 ? D.p : which ? D.Y : D.X; const size_t n = (size_t)D.Co * D.K;
    double a = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) a += (double)x[i] * x[i];
    mc_part(a, part);
}
__global__ void hb_rinit_k(const int *hb, float *R, const double *part) {   /* R = |W|_F where not yet set; one block per conv */
    const int c = blockIdx.x; if (threadIdx.x || !hb[c] || R[c] > 0.f) return;
    double s = 0; for (int j = 0; j < 32; j++) s += part[c * 32 + j];
    R[c] = (float)sqrt(s);
}
__global__ void mopt_apply_k(const muon_desc_t *d, int swap, int mc, const int *hb, const float *R, const double *po, double *pw,
                             float lr, float wd, float hb_lr) {
    const muon_desc_t D = d[blockIdx.y]; const float *O = swap ? D.Y : D.X; const size_t n = (size_t)D.Co * D.K;
    const double so = mc_total(po);
    double a = 0;
    if (hb && hb[blockIdx.y]) {   /* trial point W - eta R O / |O|; its norm for the projection */
        const float st = (float)((double)hb_lr * R[blockIdx.y] / fmax(sqrt(so), 1e-30));
        for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { const float w = D.p[i] - st * O[i]; D.p[i] = w; a += (double)w * w; }
    } else {
        float scale = sqrtf(fmaxf(1.f, (float)D.Co / (float)D.K));   /* Muon (as bmuon_apply_k) */
        if (mc) {
            const double nk = (double)D.Co * D.K, tgt = sqrt(fmax(1.0, (double)D.Co / D.K)) * sqrt(fmin((double)D.Co, (double)D.K) / nk);
            scale = (float)(tgt / (sqrt(so / nk) + 1e-8));
        }
        for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) D.p[i] = D.p[i] * (1.f - lr * wd) - lr * scale * O[i];
    }
    mc_part(a, pw);
}
__global__ void hb_proj_k(const muon_desc_t *d, const int *hb, const float *R, const double *pw) {   /* back onto the sphere |W|_F = R */
    if (!hb[blockIdx.y]) return;
    const muon_desc_t D = d[blockIdx.y]; const size_t n = (size_t)D.Co * D.K;
    const float s = (float)(R[blockIdx.y] / fmax(sqrt(mc_total(pw)), 1e-30));
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) D.p[i] *= s;
}
extern "C" void nn_muon_ext_batch(const void *descs, int nconv, int maxco, int maxk, const void *mcd, const void *nsd, int k, int ns_maxm, int ns_maxn,
                                  const int *hb, float *R, int r_init, double *part, float lr, float beta, float wd, float hb_lr) {
    const muon_desc_t *d = (const muon_desc_t *)descs;
    double *pa = part, *pb = part + (size_t)32 * nconv;
    dim3 g1(32, 1, nconv), g2(32, nconv);
    int swap;
    cudaMemsetAsync(pa, 0, (size_t)nconv * sizeof(double));   /* bmuon_mom_k's Frobenius sums (one per conv) */
    bmuon_mom_k<<<g1, 256>>>(d, beta, pa);   /* X = the Nesterov momentum */
    if (!mcd) { bmuon_scale_k<<<g1, 256>>>(d, pa); swap = ns5_batch(d, nconv, maxco, maxk); }
    else {   /* Muon-C: frequency by frequency, the direction accumulates in Y */
        mc_tables();
        const muonc_desc_t *m = (const muonc_desc_t *)mcd; const muon_desc_t *s = (const muon_desc_t *)nsd;
        const int nf = k == 3 ? 14 : 1;   /* one kernel size for the whole table: 3 or 1 */
        for (int f = 0; f < nf; f++) {
            muonc_dft_k<<<g2, 256>>>(m, f, pa);
            muonc_norm_k<<<g2, 256>>>(m, pa);
            muonc_idft_k<<<g2, 256>>>(m, f, ns5_batch(s, nconv, ns_maxm, ns_maxn), f == 0);
        }
        swap = 1;
    }
    if (r_init) { mopt_sumsq_k<<<g2, 256>>>(d, -1, pb); hb_rinit_k<<<nconv, 32>>>(hb, R, pb); }
    mopt_sumsq_k<<<g2, 256>>>(d, swap, pa);
    mopt_apply_k<<<g2, 256>>>(d, swap, mcd != nullptr, hb, R, pa, pb, lr, wd, hb_lr);
    if (hb) hb_proj_k<<<g2, 256>>>(d, hb, R, pb);
    KCHECK();
}
__constant__ float c_anvil_maps[6][3] = {{3.923798038567f, -6.095026865488f, 3.905234618423f}, {3.278126713798f, -3.328923386476f, 0.989127286973f},
    {3.505298394150f, -5.137358782410f, 1.968325560615f}, {2.815058591845f, -3.685181239622f, 1.417196497642f},
    {2.245503932403f, -2.443826979899f, 0.963091710461f}, {2.256537145403f, -2.166840097229f, 0.929501253245f}};
__global__ void anvil_mom_k(const anvil_desc_t *d, float bf, float bs, float w, float mu, double *ss) {
    const anvil_desc_t D = d[blockIdx.z]; size_t n = (size_t)D.Co * D.K;
    __shared__ float r[256]; float a = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float g = D.g[i], v0 = D.v0[i] + (1.f - bf) * (g - D.v0[i]), v1 = D.v1[i] + (1.f - bs) * (g - D.v1[i]);
        D.v0[i] = v0; D.v1[i] = v1;
        float m = w * v0 + (1.f - w) * v1, x = g + mu * (m - g);
        D.X[i] = x; a += x * x;
    }
    r[threadIdx.x] = a; __syncthreads(); for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) r[threadIdx.x] += r[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) atomicAdd(&ss[blockIdx.z], (double)r[0]);
}
__global__ void anvil_scale_k(const anvil_desc_t *d, const double *ss) {
    const anvil_desc_t D = d[blockIdx.z]; size_t n = (size_t)D.Co * D.K; float inv = (float)(1.0 / (sqrt(ss[blockIdx.z]) * 1.05 + 1e-6));
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) D.X[i] *= inv;
}
__global__ void anvil_sq_k(const anvil_desc_t *d, int it) {
    const anvil_desc_t D = d[blockIdx.z]; int Co = D.Co; float b = c_anvil_maps[it][1], c = c_anvil_maps[it][2];
    int i = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x; if (i >= Co || j >= Co) return;
    float s = 0.f; for (int k = 0; k < Co; k++) s += D.A[(size_t)i * Co + k] * D.A[(size_t)k * Co + j];
    D.B[(size_t)i * Co + j] = b * D.A[(size_t)i * Co + j] + c * s;
}
__global__ void anvil_rowpow_k(const anvil_desc_t *d, int swap) {
    const anvil_desc_t D = d[blockIdx.z]; const float *O = swap ? D.Y : D.X; int i = blockIdx.x; if (i >= D.Co) return;
    __shared__ float r[256]; float a = 0.f; for (int k = threadIdx.x; k < D.K; k += blockDim.x) { float v = O[(size_t)i * D.K + k]; a += v * v; }
    r[threadIdx.x] = a; __syncthreads(); for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) r[threadIdx.x] += r[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) D.R[i] = r[0] / (float)D.K;
}
__global__ void anvil_eq_k(const anvil_desc_t *d, float b2) {
    const anvil_desc_t D = d[blockIdx.x]; if (threadIdx.x) return;
    float pre = 0.f, post = 0.f;
    for (int i = 0; i < D.Co; i++) { float pw = D.R[i]; pre += pw; D.E[i] += (1.f - b2) * (pw - D.E[i]); float gn = rsqrtf(fmaxf(D.E[i], 1e-10f)); post += pw * gn * gn; }
    float s = sqrtf(pre) / fmaxf(sqrtf(post), 1e-10f);
    for (int i = 0; i < D.Co; i++) D.R[i] = rsqrtf(fmaxf(D.E[i], 1e-10f)) * s;
}
__global__ void anvil_apply_k(const anvil_desc_t *d, int swap, float lr, float wd) {   /* sign-aligned decay + update */
    const anvil_desc_t D = d[blockIdx.z]; const float *O = swap ? D.Y : D.X; size_t n = (size_t)D.Co * D.K;
    float scale = sqrtf(fmaxf(1.f, (float)D.Co / (float)D.K));
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float u = O[i] * D.R[i / (size_t)D.K] * scale, p = D.p[i];
        float dec = (u * p >= 0.f) ? lr * wd * p : 0.f;
        D.p[i] = p - dec - lr * u;
    }
}
extern "C" void nn_anvil_batch(const void *descs, int nconv, int maxco, int maxk, float lr, float beta_fast, float beta_slow, float w_fast, float mu, float beta2, float wd) {
    const anvil_desc_t *d = (const anvil_desc_t *)descs;
    double *ss = gn_dsums((size_t)nconv); cudaMemsetAsync(ss, 0, (size_t)nconv * sizeof(double));
    dim3 g1(32, 1, nconv), gco(nblk(maxco, 128), maxco, nconv), g64(nblk(maxco, 64), nblk(maxco, 64), nconv), gk(nblk(maxk, 64), nblk(maxco, 64), nconv), grow(maxco, 1, nconv);
    anvil_mom_k<<<g1, 256>>>(d, beta_fast, beta_slow, w_fast, mu, ss);
    anvil_scale_k<<<g1, 256>>>(d, ss);
    int swap = 0;
    static const float maps_a[6] = {3.923798038567f, 3.278126713798f, 3.505298394150f, 2.815058591845f, 2.245503932403f, 2.256537145403f};
    for (int it = 0; it < 6; it++) {
        gemm_ns_k<0, anvil_desc_t><<<g64, 256>>>(d, swap, 0.f, 0.f);
        anvil_sq_k<<<gco, 128>>>(d, it);
        gemm_ns_k<2, anvil_desc_t><<<gk, 256>>>(d, swap, maps_a[it], 0.f);
        swap ^= 1;
    }
    anvil_rowpow_k<<<grow, 256>>>(d, swap);
    anvil_eq_k<<<nconv, 32>>>(d, beta2);
    anvil_apply_k<<<g1, 256>>>(d, swap, lr, wd);
    KCHECK();
}
extern "C" void nn_adamw(float *p, const float *g, float *m, float *v, size_t n, float lr, float b1, float b2, float eps, float wd, int step) {
    float c1 = 1.f - powf(b1, (float)step), c2 = 1.f - powf(b2, (float)step);
    adamw_k<<<nblk(n, 256), 256>>>(p, g, m, v, n, lr, b1, b2, eps, wd, c1, c2);
    KCHECK();
}
__global__ void ema_k(float *e, const float *p, size_t n, float d) { size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; if (i < n) e[i] = d * e[i] + (1.f - d) * p[i]; }
extern "C" void nn_ema(float *ema, const float *p, size_t n, float decay) { ema_k<<<nblk(n, 256), 256>>>(ema, p, n, decay); KCHECK(); }
__global__ void sum_k(const float *x, size_t n, float *out, int sq) {
    double s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { double v = x[i]; s += sq ? v * v : v; }
    __shared__ double r[256];
    r[threadIdx.x] = s;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) { if (threadIdx.x < o) r[threadIdx.x] += r[threadIdx.x + o]; __syncthreads(); }
    if (threadIdx.x == 0) out[blockIdx.x] = (float)r[0];
}
double reduce(const float *x, size_t n, float *scratch, int sq) {
    int nb = 1024;
    sum_k<<<nb, 256>>>(x, n, scratch, sq);
    float h[1024];
    CK(cudaMemcpy(h, scratch, sizeof h, cudaMemcpyDeviceToHost));
    double s = 0;
    for (int i = 0; i < nb; i++) s += h[i];
    return s;
}
extern "C" double nn_sum(const float *x, size_t n, float *scratch) { return reduce(x, n, scratch, 0); }
extern "C" double nn_sumsq(const float *x, size_t n, float *scratch) { return reduce(x, n, scratch, 1); }
extern "C" void nn_fake_quant_affine(void *x, shape5 s, int fmt, const float *mean, const float *rstd, int G) {
    if (fmt <= 0 || ISMX(x) || fmt == 1) return;   /* MX formats only (the tensor scale of NVFP4 is not meaningful after the affine) */
    size_t S = shape_spatial(s); int C = s.c, N = s.n; const int gs = 32, ng = (C + gs - 1) / gs;
    size_t tot = (size_t)N * ng * S;
    if (ABF && g_h16) fq_aff_k<f16><<<nblk(tot, 256), 256>>>((f16 *)x, N, C, S, fmt, mean, rstd, G);
    else if (ABF) fq_aff_k<bf16><<<nblk(tot, 256), 256>>>((bf16 *)x, N, C, S, fmt, mean, rstd, G);
    else fq_aff_k<float><<<nblk(tot, 256), 256>>>((float *)x, N, C, S, fmt, mean, rstd, G);
    KCHECK();
}
extern "C" void nn_fake_quant(void *x, shape5 s, int fmt) {
    if (fmt <= 0 || ISMX(x)) return;
    size_t S = shape_spatial(s), n = shape_numel(s);
    static unsigned *am[8];
    if (!am[cur_dev()]) cudaMalloc(&am[cur_dev()], 4);
    cudaMemsetAsync(am[cur_dev()], 0, 4);
    const int gs = fmt == 1 ? 16 : 32;
    size_t nt = (size_t)s.n * ((s.c + gs - 1) / gs) * S;
    if (ABF && g_h16) { fq_amax_k<f16><<<512, 256>>>((const f16 *)x, n, am[cur_dev()]); fq_k<f16><<<nblk(nt, 256), 256>>>((f16 *)x, s.n, s.c, S, fmt, am[cur_dev()]); }
    else if (ABF) { fq_amax_k<bf16><<<512, 256>>>((const bf16 *)x, n, am[cur_dev()]); fq_k<bf16><<<nblk(nt, 256), 256>>>((bf16 *)x, s.n, s.c, S, fmt, am[cur_dev()]); }
    else { fq_amax_k<float><<<512, 256>>>((const float *)x, n, am[cur_dev()]); fq_k<float><<<nblk(nt, 256), 256>>>((float *)x, s.n, s.c, S, fmt, am[cur_dev()]); }
    KCHECK();
}
