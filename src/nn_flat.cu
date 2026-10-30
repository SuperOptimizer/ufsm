/* SLIM iterations of ufsm flatten on the GPU (flatten.c slim_run, same algorithm, fp64): per triangle the local step (closed-form 2x2
   SVD, symmetric Dirichlet proxy weights, closest rotation), the matrix-free Jacobi-preconditioned CG of the global step (warm start at the
   current map, the same proximal term), the flip-capped halving line search. Reductions run over a fixed grid in a fixed order, so a run is
   deterministic; it differs from the CPU threads' summation order in the last bits only (energies agree to ~1e-12 relative per step).
   Called through flatten.c's weak flat_slim_gpu: a binary without this object (test_flatten, test_strip) stays CPU-only. */
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { FB = 256, FG = 1024 };   /* block size, blocks of every reduction (fixed: deterministic sums) */

__device__ __forceinline__ void fl_jac(const double *g, const double *x, const int *v, double J[4]) {
    J[0] = J[1] = J[2] = J[3] = 0;
    for (int k = 0; k < 3; k++) { const double u = x[2 * v[k]], w = x[2 * v[k] + 1]; J[0] += u * g[2 * k]; J[1] += u * g[2 * k + 1]; J[2] += w * g[2 * k]; J[3] += w * g[2 * k + 1]; }
}
__device__ __forceinline__ double fl_sd(const double J[4]) {
    const double det = J[0] * J[3] - J[1] * J[2], f = J[0] * J[0] + J[1] * J[1] + J[2] * J[2] + J[3] * J[3];
    return det > 0 ? f * (1 + 1 / (det * det)) : INFINITY;
}
/* block sum of (a, b) -> part[2 * blockIdx.x] */
__device__ __forceinline__ void fl_bsum2(double a, double b, double *part) {
    __shared__ double sa[FB], sb[FB];
    sa[threadIdx.x] = a; sb[threadIdx.x] = b; __syncthreads();
    for (int s = FB / 2; s > 0; s >>= 1) { if ((int)threadIdx.x < s) { sa[threadIdx.x] += sa[threadIdx.x + s]; sb[threadIdx.x] += sb[threadIdx.x + s]; } __syncthreads(); }
    if (!threadIdx.x) { part[2 * blockIdx.x] = sa[0]; part[2 * blockIdx.x + 1] = sb[0]; }
}
__global__ void k_fin2(const double *part, int n, double *out) {   /* one block: sums of the n (a, b) partials -> out[0..1] */
    __shared__ double sa[FB], sb[FB];
    double a = 0, b = 0; for (int i = threadIdx.x; i < n; i += FB) { a += part[2 * i]; b += part[2 * i + 1]; }
    sa[threadIdx.x] = a; sb[threadIdx.x] = b; __syncthreads();
    for (int s = FB / 2; s > 0; s >>= 1) { if ((int)threadIdx.x < s) { sa[threadIdx.x] += sa[threadIdx.x + s]; sb[threadIdx.x] += sb[threadIdx.x + s]; } __syncthreads(); }
    if (!threadIdx.x) { out[0] = sa[0]; out[1] = sb[0]; }
}
__global__ void k_finmin(const double *part, int n, double *out) {
    __shared__ double s[FB];
    double m = INFINITY; for (int i = threadIdx.x; i < n; i += FB) m = fmin(m, part[i]);
    s[threadIdx.x] = m; __syncthreads();
    for (int k = FB / 2; k > 0; k >>= 1) { if ((int)threadIdx.x < k) s[threadIdx.x] = fmin(s[threadIdx.x], s[threadIdx.x + k]); __syncthreads(); }
    if (!threadIdx.x) out[0] = s[0];
}

#define GS(i, n) for (long long i = blockIdx.x * (long long)FB + threadIdx.x; i < (n); i += (long long)gridDim.x * FB)

__global__ void k_energy(int nt, const double *g, const double *A, const int *tri, const double *x, double *part) {
    double e = 0, a = 0;
    GS(t, nt) { if (!(A[t] > 0)) continue; double J[4]; fl_jac(g + 6 * t, x, tri + 3 * t, J); e += A[t] * fl_sd(J); a += A[t]; }
    fl_bsum2(e, a, part);
}
__global__ void k_local(int nt, const double *g, const double *A, const int *tri, const double *x, double *Mo, double *MRo) {
    GS(t, nt) {
        double J[4]; fl_jac(g + 6 * t, x, tri + 3 * t, J);
        const double E = (J[0] + J[3]) / 2, F = (J[0] - J[3]) / 2, G = (J[2] + J[1]) / 2, H = (J[2] - J[1]) / 2;
        const double Q = hypot(E, H), R = hypot(F, G), a1 = atan2(G, F), a2 = atan2(H, E);
        const double s1 = Q + R; double s2 = Q - R; const double ph = (a2 + a1) / 2, ro = a2;
        if (!(s2 > 1e-12)) s2 = 1e-12;
        const double w1 = (s1 + 1) * (s1 * s1 + 1) / (s1 * s1 * s1), w2 = (s2 + 1) * (s2 * s2 + 1) / (s2 * s2 * s2);
        const double cp = cos(ph), sp = sin(ph);
        double *M = Mo + 3 * t; const double m0 = cp * cp * w1 + sp * sp * w2, m1 = cp * sp * (w1 - w2), m2 = sp * sp * w1 + cp * cp * w2;
        M[0] = m0; M[1] = m1; M[2] = m2;
        const double cr = cos(ro), sr = sin(ro), a = A[t]; double *B = MRo + 4 * t;
        B[0] = a * (m0 * cr + m1 * sr); B[1] = a * (-m0 * sr + m1 * cr); B[2] = a * (m1 * cr + m2 * sr); B[3] = a * (-m1 * sr + m2 * cr);
    }
}
__global__ void k_triq(int nt, const double *g, const double *A, const int *tri, const double *M3, const double *src, double *Qo) {
    GS(t, nt) {
        double J[4]; fl_jac(g + 6 * t, src, tri + 3 * t, J); const double *M = M3 + 3 * t, a = A[t]; double *Q = Qo + 4 * t;
        Q[0] = a * (M[0] * J[0] + M[1] * J[2]); Q[1] = a * (M[0] * J[1] + M[1] * J[3]); Q[2] = a * (M[1] * J[0] + M[2] * J[2]); Q[3] = a * (M[1] * J[1] + M[2] * J[3]);
    }
}
__global__ void k_gather(int nv, const int *vo, const int *vl, const double *Q4, const double *g, double lam, const double *src, double *dst, double *part) {
    double pq = 0;
    GS(v, nv) {
        double a = lam * src[2 * v], b = lam * src[2 * v + 1];
        for (int i = vo[v]; i < vo[v + 1]; i++) {
            const int t = vl[i] / 3, k = vl[i] % 3; const double *Q = Q4 + 4 * (long long)t, *gg = g + 6 * (long long)t + 2 * k;
            a += Q[0] * gg[0] + Q[1] * gg[1]; b += Q[2] * gg[0] + Q[3] * gg[1];
        }
        dst[2 * v] = a; dst[2 * v + 1] = b; pq += src[2 * v] * a + src[2 * v + 1] * b;
    }
    fl_bsum2(pq, 0, part);
}
__global__ void k_rhs(int nv, const int *vo, const int *vl, const double *MR, const double *M3, const double *A, const double *g, double lam, const double *x, double *rhs, double *diag, double *part) {
    double md = 0;
    GS(v, nv) {
        double a = lam * x[2 * v], b = lam * x[2 * v + 1], da = lam, db = lam;
        for (int i = vo[v]; i < vo[v + 1]; i++) {
            const int t = vl[i] / 3, k = vl[i] % 3; const double *B = MR + 4 * (long long)t, *gg = g + 6 * (long long)t + 2 * k, *M = M3 + 3 * (long long)t, q = gg[0] * gg[0] + gg[1] * gg[1];
            a += B[0] * gg[0] + B[1] * gg[1]; b += B[2] * gg[0] + B[3] * gg[1]; da += A[t] * M[0] * q; db += A[t] * M[2] * q;
        }
        rhs[2 * v] = a; rhs[2 * v + 1] = b; diag[2 * v] = da; diag[2 * v + 1] = db; md += da + db;
    }
    fl_bsum2(md, 0, part);
}
__global__ void k_cginit(long long n2, const double *rhs, const double *q, const double *diag, double *r, double *z, double *p, double *part, double *part2) {
    double rz = 0, rr = 0, bb = 0;
    GS(i, n2) { const double ri = rhs[i] - q[i], zi = ri / diag[i]; r[i] = ri; z[i] = zi; p[i] = zi; rz += ri * zi; rr += ri * ri; bb += rhs[i] * rhs[i]; }
    fl_bsum2(rz, rr, part); fl_bsum2(bb, 0, part2);
}
__global__ void k_update(long long n2, double alpha, double *d, const double *p, double *r, const double *q, double *z, const double *diag, double *part) {
    double rz = 0, rr = 0;
    GS(i, n2) { d[i] += alpha * p[i]; const double ri = r[i] - alpha * q[i], zi = ri / diag[i]; r[i] = ri; z[i] = zi; rz += ri * zi; rr += ri * ri; }
    fl_bsum2(rz, rr, part);
}
__global__ void k_pdir(long long n2, double beta, const double *z, double *p) { GS(i, n2) p[i] = z[i] + beta * p[i]; }
__global__ void k_sub(long long n2, double *d, const double *x) { GS(i, n2) d[i] -= x[i]; }
__global__ void k_axpy(long long n2, double a, const double *x, const double *d, double *xt) { GS(i, n2) xt[i] = x[i] + a * d[i]; }
__global__ void k_maxstep(int nt, const int *tri, const double *x, const double *d, double *part) {
    __shared__ double s[FB];
    double tm = INFINITY;
    GS(t, nt) {
        const int *v = tri + 3 * t;
        const double e1u = x[2 * v[1]] - x[2 * v[0]], e1v = x[2 * v[1] + 1] - x[2 * v[0] + 1], e2u = x[2 * v[2]] - x[2 * v[0]], e2v = x[2 * v[2] + 1] - x[2 * v[0] + 1];
        const double f1u = d[2 * v[1]] - d[2 * v[0]], f1v = d[2 * v[1] + 1] - d[2 * v[0] + 1], f2u = d[2 * v[2]] - d[2 * v[0]], f2v = d[2 * v[2] + 1] - d[2 * v[0] + 1];
        const double cc = e1u * e2v - e1v * e2u, bb = e1u * f2v - e1v * f2u + f1u * e2v - f1v * e2u, aa = f1u * f2v - f1v * f2u;
        double r = INFINITY;
        if (fabs(aa) < 1e-14 * (fabs(bb) + fabs(cc) + 1e-300)) { if (bb < 0) r = -cc / bb; }
        else {
            const double D = bb * bb - 4 * aa * cc;
            if (D >= 0) { const double sq = sqrt(D), q = -0.5 * (bb + (bb >= 0 ? sq : -sq)), r1 = q / aa, r2 = q != 0 ? cc / q : INFINITY;
                          r = fmin(r, r1 > 0 ? r1 : INFINITY); r = fmin(r, r2 > 0 ? r2 : INFINITY); }
        }
        if (r < tm) tm = r;
    }
    s[threadIdx.x] = tm; __syncthreads();
    for (int k = FB / 2; k > 0; k >>= 1) { if ((int)threadIdx.x < k) s[threadIdx.x] = fmin(s[threadIdx.x], s[threadIdx.x + k]); __syncthreads(); }
    if (!threadIdx.x) part[blockIdx.x] = s[0];
}

static double fl_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int fl_grid(long long n) { const long long b = (n + FB - 1) / FB; return (int)(b < 1 ? 1 : b > FG ? FG : b); }

typedef struct {
    int nv, nt; long long n2;
    double *g, *A, *M, *MR, *Q, *x, *d, *rhs, *diag, *r, *z, *p, *q, *xt, *part, *part2, *res;
    int *tri, *vo, *vl;
} flg;
static void flg_free(flg *G) {
    void *p[] = {G->g, G->A, G->M, G->MR, G->Q, G->x, G->d, G->rhs, G->diag, G->r, G->z, G->p, G->q, G->xt, G->part, G->part2, G->res, G->tri, G->vo, G->vl};
    for (size_t i = 0; i < sizeof p / sizeof *p; i++) if (p[i]) cudaFree(p[i]);
}
static double fl_sum2(flg *G, int nb, double *b) {   /* partials of the last kernel -> (a, b) on the host */
    double h[2]; k_fin2<<<1, FB>>>(G->part, nb, G->res); cudaMemcpy(h, G->res, 16, cudaMemcpyDeviceToHost);
    if (b) *b = h[1]; return h[0];
}
static double fl_energy(flg *G, const double *x) {
    const int nb = fl_grid(G->nt); k_energy<<<nb, FB>>>(G->nt, G->g, G->A, G->tri, x, G->part);
    double a; const double e = fl_sum2(G, nb, &a); return a > 0 ? e / a : 0;
}
static double fl_matvec(flg *G, double lam, const double *x, double *y) {
    k_triq<<<fl_grid(G->nt), FB>>>(G->nt, G->g, G->A, G->tri, G->M, x, G->Q);
    const int nb = fl_grid(G->nv); k_gather<<<nb, FB>>>(G->nv, G->vo, G->vl, G->Q, G->g, lam, x, y, G->part);
    return fl_sum2(G, nb, nullptr);
}

/* slim_run on the GPU: g (6 / triangle), A (1 / triangle), tri (3 / triangle), the incidence CSR vo (nv + 1) / vl (3 nt); x (2 / vertex)
   updated in place. Returns 0 and the final energy in *Eout, -1 when the device is missing or out of memory (x untouched: the caller
   runs the CPU solver). */
extern "C" int flat_slim_gpu(const double *g, const double *A, const int *tri, int nt, const int *vo, const int *vl, int nv, double *xh,
                             int iters, int cg_iters, double tol, int verbose, const char *tag, double *Eout) {
    int ndev = 0; if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) { cudaGetLastError(); return -1; }
    static int flags_set = 0;   /* host waits block instead of spinning: the desk's cores are busy with other jobs */
    if (!flags_set) { flags_set = 1; if (cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync) != cudaSuccess) cudaGetLastError(); }
    flg G; memset(&G, 0, sizeof G); G.nv = nv; G.nt = nt; G.n2 = 2LL * nv;
    const size_t T = (size_t)nt, V2 = (size_t)G.n2 * 8;
    int ok = cudaMalloc(&G.g, T * 48) == cudaSuccess && cudaMalloc(&G.A, T * 8) == cudaSuccess && cudaMalloc(&G.M, T * 24) == cudaSuccess &&
             cudaMalloc(&G.MR, T * 32) == cudaSuccess && cudaMalloc(&G.Q, T * 32) == cudaSuccess && cudaMalloc(&G.tri, T * 12) == cudaSuccess &&
             cudaMalloc(&G.vo, ((size_t)nv + 1) * 4) == cudaSuccess && cudaMalloc(&G.vl, T * 12) == cudaSuccess &&
             cudaMalloc(&G.x, V2) == cudaSuccess && cudaMalloc(&G.d, V2) == cudaSuccess && cudaMalloc(&G.rhs, V2) == cudaSuccess &&
             cudaMalloc(&G.diag, V2) == cudaSuccess && cudaMalloc(&G.r, V2) == cudaSuccess && cudaMalloc(&G.z, V2) == cudaSuccess &&
             cudaMalloc(&G.p, V2) == cudaSuccess && cudaMalloc(&G.q, V2) == cudaSuccess && cudaMalloc(&G.xt, V2) == cudaSuccess &&
             cudaMalloc(&G.part, FG * 16) == cudaSuccess && cudaMalloc(&G.part2, FG * 16) == cudaSuccess && cudaMalloc(&G.res, 16) == cudaSuccess;
    if (ok) ok = cudaMemcpy(G.g, g, T * 48, cudaMemcpyHostToDevice) == cudaSuccess && cudaMemcpy(G.A, A, T * 8, cudaMemcpyHostToDevice) == cudaSuccess &&
                 cudaMemcpy(G.tri, tri, T * 12, cudaMemcpyHostToDevice) == cudaSuccess && cudaMemcpy(G.vo, vo, ((size_t)nv + 1) * 4, cudaMemcpyHostToDevice) == cudaSuccess &&
                 cudaMemcpy(G.vl, vl, T * 12, cudaMemcpyHostToDevice) == cudaSuccess && cudaMemcpy(G.x, xh, V2, cudaMemcpyHostToDevice) == cudaSuccess;
    if (!ok) { fprintf(stderr, "flatten: GPU unavailable (%s): CPU solver\n", cudaGetErrorString(cudaGetLastError())); flg_free(&G); return -1; }
    const long long n2 = G.n2; const int bv = fl_grid(n2);
    double E = fl_energy(&G, G.x);
    if (verbose) fprintf(stderr, "flatten: %s %d vertices %d triangles, energy %.5f (GPU)\n", tag, nv, nt, E);
    for (int it = 0; it < iters && isfinite(E); it++) {
        const double t0 = fl_now();
        k_local<<<fl_grid(nt), FB>>>(nt, G.g, G.A, G.tri, G.x, G.M, G.MR);
        const int nbv = fl_grid(nv);
        k_rhs<<<nbv, FB>>>(nv, G.vo, G.vl, G.MR, G.M, G.A, G.g, 0.0, G.x, G.rhs, G.diag, G.part);
        const double lam = 1e-6 * fl_sum2(&G, nbv, nullptr) / (double)n2;
        k_rhs<<<nbv, FB>>>(nv, G.vo, G.vl, G.MR, G.M, G.A, G.g, lam, G.x, G.rhs, G.diag, G.part);
        cudaMemcpy(G.d, G.x, V2, cudaMemcpyDeviceToDevice);
        fl_matvec(&G, lam, G.d, G.q);
        k_cginit<<<bv, FB>>>(n2, G.rhs, G.q, G.diag, G.r, G.z, G.p, G.part, G.part2);
        double rr0, rz = fl_sum2(&G, bv, &rr0), hb[2];
        k_fin2<<<1, FB>>>(G.part2, bv, G.res); cudaMemcpy(hb, G.res, 16, cudaMemcpyDeviceToHost); const double bb = hb[0];
        int k = 0; double rr = rr0;
        for (; k < cg_iters && rr > 1e-14 * bb; k++) {
            const double pq = fl_matvec(&G, lam, G.p, G.q); if (!(pq > 0)) break;
            const double alpha = rz / pq;
            k_update<<<bv, FB>>>(n2, alpha, G.d, G.p, G.r, G.q, G.z, G.diag, G.part);
            const double rz2 = fl_sum2(&G, bv, &rr);
            const double beta = rz2 / rz; rz = rz2;
            k_pdir<<<bv, FB>>>(n2, beta, G.z, G.p);
        }
        k_sub<<<bv, FB>>>(n2, G.d, G.x);
        /* line search: capped at 0.8 of the first inversion, halved until the energy falls */
        const int nbt = fl_grid(nt); k_maxstep<<<nbt, FB>>>(nt, G.tri, G.x, G.d, G.part);
        double tm; k_finmin<<<1, FB>>>(G.part, nbt, G.res); cudaMemcpy(&tm, G.res, 8, cudaMemcpyDeviceToHost);
        double a = fmin(1.0, 0.8 * tm), E2 = E, step = 0;
        for (int s = 0; s < 40 && a > 1e-12; s++, a *= 0.5) {
            k_axpy<<<bv, FB>>>(n2, a, G.x, G.d, G.xt);
            E2 = fl_energy(&G, G.xt);
            if (isfinite(E2) && E2 < E) { double *t = G.x; G.x = G.xt; G.xt = t; step = a; break; }
        }
        if (step == 0) E2 = E;
        if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "flatten: GPU error: CPU solver\n"); flg_free(&G); return -1; }
        if (verbose) fprintf(stderr, "flatten: %s it %d energy %.5f -> %.5f (step %.3g, cg %d, res %.2g, %.2fs GPU)\n", tag, it, E, E2, step, k, sqrt(rr / (rr0 > 0 ? rr0 : 1)), fl_now() - t0);
        const double dE = E - E2; E = E2;
        if (step == 0 || dE < tol * E) break;
    }
    const int okc = cudaMemcpy(xh, G.x, V2, cudaMemcpyDeviceToHost) == cudaSuccess;
    flg_free(&G);
    if (!okc) return -1;
    *Eout = E; return 0;
}
