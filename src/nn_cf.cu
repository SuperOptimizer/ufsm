/* ufsm count-field on the GPU (src/cfgpu.h): pass 1's pooling, the CG solves, the --model-only R 0 model pass. Mirrors
   src/countfield.c element by element: built without fast math and with -fmad=false (Makefile), so every fp32 / fp64 operation
   rounds as the CPU's (gcc -std=c23 does not contract either); the libm calls (atan2, hypot, cos, sin) may differ from glibc in the
   last bit. Each solve cell's sums run in one thread in the CPU's voxel order (z, y, x); the CG's dot products over a fixed grid of
   blocks (deterministic; the CPU threads sum in another order, so CG iterates differ in the last bits). */
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cfgpu.h"

#define CF_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "count-field gpu: %s (%s:%d)\n", cudaGetErrorString(e_), __FILE__, __LINE__); goto fail; } } while (0)
enum { CB = 256, CG = 1024 };   /* block size; blocks of every CG reduction (fixed: deterministic sums) */

extern "C" int cf_gpu_ok(void) { int n = 0; if (cudaGetDeviceCount(&n) != cudaSuccess) { cudaGetLastError(); return 0; } return n > 0; }

/* ---- device copies of countfield.c's helpers ---- */
__device__ __forceinline__ double d_axis_at(const float *v, int n, double z) { if (z <= 0) return v[0]; if (z >= n - 1) return v[n - 1]; const int i = (int)z; const double t = z - i; return v[i] + t * (v[i + 1] - v[i]); }
__device__ __forceinline__ double d_smooth(double x, double a, double b) { const double t = x <= a ? 0 : x >= b ? 1 : (x - a) / (b - a); return t * t * (3 - 2 * t); }
/* the phase of a voxel only takes 65536 values (pc, ps bytes): atan2 and hypot of each, tabled on the host by glibc (the CPU's very
   numbers; a device atan2 a bit off flips wrap(+-0.5) between exactly opposite phasors, a whole half turn in the phase gradient) */
struct ph_tab { const double *a, *c; };   /* a = atan2(b, a), c = hypot(a, b) at index pc * 256 + ps */
static double *g_tab;
static int tab_get(ph_tab *t) {
    if (!g_tab) {
        double *h = (double *)malloc(2 * 65536 * sizeof(double)); if (!h) return 1;
        for (int u = 0; u < 256; u++) for (int v = 0; v < 256; v++) { const double a = (u - 128) / 127.0, b = (v - 128) / 127.0; h[u * 256 + v] = atan2(b, a); h[65536 + u * 256 + v] = hypot(a, b); }
        const int bad = cudaMalloc(&g_tab, 2 * 65536 * sizeof(double)) != cudaSuccess || cudaMemcpy(g_tab, h, 2 * 65536 * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess;
        free(h); if (bad) { cudaFree(g_tab); g_tab = nullptr; cudaGetLastError(); return 1; }
    }
    t->a = g_tab; t->c = g_tab + 65536; return 0;
}
__device__ __forceinline__ double d_ph(const ph_tab &T, const uint8_t *pc, const uint8_t *ps, const uint8_t *w, size_t i, double *c) {
    const int k = pc[i] * 256 + ps[i]; *c = w[i] ? T.c[k] : 0; return T.a[k] / (2 * M_PI);
}
__device__ __forceinline__ double d_wrap(double d) { return d - round(d); }
__device__ double d_axis(const uint8_t *const *t8, size_t i, double v[3]) {
    double T[3][3]; const double t[6] = {t8[0][i] / 255.0, t8[1][i] / 255.0, t8[2][i] / 255.0, t8[3][i] / 255.0 - 0.5, t8[4][i] / 255.0 - 0.5, t8[5][i] / 255.0 - 0.5};
    T[0][0] = t[0]; T[1][1] = t[1]; T[2][2] = t[2]; T[0][1] = T[1][0] = t[3]; T[0][2] = T[2][0] = t[4]; T[1][2] = T[2][1] = t[5];
    int k = T[0][0] >= T[1][1] && T[0][0] >= T[2][2] ? 0 : T[1][1] >= T[2][2] ? 1 : 2; v[0] = v[1] = v[2] = 0; v[k] = 1;
    double l = 0;
    for (int it = 0; it < 16; it++) {
        double w[3] = {0, 0, 0}; for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) w[a] += T[a][b] * v[b];
        l = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]); if (l < 1e-12) return 0; for (int a = 0; a < 3; a++) v[a] = w[a] / l;
    }
    const double tr = t[0] + t[1] + t[2]; return tr > 1e-6 ? l / tr : 0;
}
__device__ double d_tri(const float *U, int d, int hh, int w, double qz, double qy, double qx) {
    const double q[3] = {qz, qy, qx}; const int dm[3] = {d, hh, w}; int c[3]; double t[3];
    for (int k = 0; k < 3; k++) { double a = q[k] < 0 ? 0 : q[k] > dm[k] - 1 ? dm[k] - 1 : q[k]; c[k] = (int)a; if (c[k] > dm[k] - 2) c[k] = dm[k] - 2 < 0 ? 0 : dm[k] - 2; t[k] = dm[k] > 1 ? a - c[k] : 0; }
    double s = 0;
    for (int e = 0; e < 8; e++) {
        const int a = c[0] + (e >> 2 & 1), b = c[1] + (e >> 1 & 1), x = c[2] + (e & 1);
        const double wv = (e >> 2 & 1 ? t[0] : 1 - t[0]) * (e >> 1 & 1 ? t[1] : 1 - t[1]) * (e & 1 ? t[2] : 1 - t[2]);
        if (wv != 0) s += wv * U[((size_t)(a < d ? a : d - 1) * hh + (b < hh ? b : hh - 1)) * w + (x < w ? x : w - 1)];
    }
    return s;
}

/* ---- pass 1: one thread per solve cell of the group (cf_pool_run's sums in its order) ---- */
struct pool_k {
    const uint8_t *pl[10]; int ra;   /* device rows [ra, ..) of the planes pc ps t0..t5 sp w */
    int zoff, nzl, Za, nZ, H, W, P, hh, w, na; double h; const float *cy, *cx; double emin, sheet0, kph, c0; float *S, *G[3]; ph_tab T;
};
__global__ void k_pool(pool_k a) {
    const long long nc = (long long)a.nZ * a.hh * a.w, j = blockIdx.x * (long long)blockDim.x + threadIdx.x; if (j >= nc) return;
    const int X = (int)(j % a.w), Y = (int)(j / a.w % a.hh), Z = a.Za + (int)(j / ((long long)a.w * a.hh));
    const size_t HW = (size_t)a.H * a.W; const uint8_t *pc = a.pl[0], *ps = a.pl[1], *sp = a.pl[8], *wp = a.pl[9];
    float S = 0, G[3] = {0, 0, 0}; unsigned long long ckey = ~0ull; double cv[3] = {0, 0, 0}, clr = 0;
    const int ya = Y * a.P, yb = min(a.H, ya + a.P), xa = X * a.P, xb = min(a.W, xa + a.P);
    for (int dz = 0; dz < a.P; dz++) {
        const int z = Z * a.P + dz, zl = z - a.zoff; if (zl < 0) continue; if (zl >= a.nzl) break;
        const double pz = (z + 0.5) * a.h, cy = d_axis_at(a.cy, a.na, pz), cx = d_axis_at(a.cx, a.na, pz);
        for (int y = ya; y < yb; y++) for (int x = xa; x < xb; x++) {
            const size_t i = (size_t)(z - a.ra) * HW + (size_t)y * a.W + x; if (!wp[i]) continue;
            if (a.kph > 0 && zl > 0 && zl + 1 < a.nzl && y > 0 && y + 1 < a.H && x > 0 && x + 1 < a.W) {
                double c0v, cm, cp; const double p0 = d_ph(a.T, pc, ps, wp, i, &c0v); const size_t st[3] = {HW, (size_t)a.W, 1}; double gph[3], cmin = c0v;
                for (int k = 0; k < 3; k++) { const double pm = d_ph(a.T, pc, ps, wp, i - st[k], &cm), pp = d_ph(a.T, pc, ps, wp, i + st[k], &cp); cmin = fmin(cmin, fmin(cm, cp)); gph[k] = 0.5 * (d_wrap(pp - p0) + d_wrap(p0 - pm)); }
                const double cw = a.kph * d_smooth(cmin, a.c0, a.c0 + 0.3);
                if (cw > 0) { S += (float)cw; for (int k = 0; k < 3; k++) G[k] += (float)(cw * gph[k]); }
            }
            if (!sp[i]) continue;
            const unsigned long long key = (unsigned long long)a.pl[2][i] | (unsigned long long)a.pl[3][i] << 8 | (unsigned long long)a.pl[4][i] << 16 |
                                           (unsigned long long)a.pl[5][i] << 24 | (unsigned long long)a.pl[6][i] << 32 | (unsigned long long)a.pl[7][i] << 40;
            if (key != ckey) { ckey = key; clr = d_axis(a.pl + 2, i, cv); }   /* (the same n n^T bytes as the last voxel: the same axis) */
            double v[3] = {cv[0], cv[1], cv[2]}; const double lr = clr; if (lr <= 0) continue;
            double ry = (y + 0.5) * a.h - cy, rx = (x + 0.5) * a.h - cx; const double rl = hypot(ry, rx); if (rl < 1) continue; ry /= rl; rx /= rl;
            double dot = v[1] * ry + v[2] * rx; if (dot < 0) { v[0] = -v[0]; v[1] = -v[1]; v[2] = -v[2]; dot = -dot; }
            const double c = wp[i] / 255.0 * d_smooth(dot, a.emin, a.emin + 0.25) * d_smooth(lr, a.sheet0, a.sheet0 + 0.3);
            if (c <= 0) continue;
            const double rho = 8.0 / sp[i];
            S += (float)c; for (int k = 0; k < 3; k++) G[k] += (float)(c * rho * v[k]);
        }
    }
    const size_t o = (size_t)(Z - a.Za) * a.hh * a.w + (size_t)Y * a.w + X; a.S[o] = S; for (int k = 0; k < 3; k++) a.G[k][o] = G[k];
}

extern "C" int cf_gpu_pool(const cf_gpu_poolcfg *c, const uint8_t *const pl[10], int zoff, int nzl, float *S, float *const G[3]) {
    const size_t HW = (size_t)c->H * c->W, cells = (size_t)c->hh * c->w; const int P = c->P;
    uint8_t *dpl[2] = {nullptr, nullptr}; float *dax = nullptr, *dsg[2] = {nullptr, nullptr}; int rc = 1; cudaStream_t sk = nullptr, sc = nullptr; ph_tab T;
    /* groups of solve rows: their field rows (+1 halo row each side, inside the chunk) of the 10 planes under ~2 GB, two buffers: the
       next group's upload overlaps this group's kernel */
    size_t fr = 0, tot = 0; if (cudaMemGetInfo(&fr, &tot) != cudaSuccess) { cudaGetLastError(); return 1; }
    const double bud = fmin(2e9, (fr - fmin((double)fr, 512e6)) / 2.2);   /* per buffer: 2 GB, or less of a shared device */
    int gZ = (int)((bud / (10.0 * HW) - 2) / P); if (getenv("UFSM_CF_POOL_GZ")) gZ = atoi(getenv("UFSM_CF_POOL_GZ")); if (gZ < 1) gZ = 1;   /* (env: tests) */
    const int Z0 = zoff / P, Z1 = (zoff + nzl + P - 1) / P; if (gZ > Z1 - Z0) gZ = Z1 - Z0;
    const int maxr = gZ * P + 2, ng = (Z1 - Z0 + gZ - 1) / gZ; const size_t pb = (size_t)maxr * HW;
    if (tab_get(&T)) return 1;
    CF_CK(cudaStreamCreateWithFlags(&sk, cudaStreamNonBlocking)); CF_CK(cudaStreamCreateWithFlags(&sc, cudaStreamNonBlocking));
    for (int q = 0; q < 2 && q < ng; q++) { CF_CK(cudaMalloc(&dpl[q], 10 * pb)); CF_CK(cudaMalloc(&dsg[q], 4 * (size_t)gZ * cells * sizeof(float))); }
    CF_CK(cudaMalloc(&dax, 2 * (size_t)c->na * sizeof(float)));
    CF_CK(cudaMemcpy(dax, c->cy, c->na * sizeof(float), cudaMemcpyHostToDevice)); CF_CK(cudaMemcpy(dax + c->na, c->cx, c->na * sizeof(float), cudaMemcpyHostToDevice));
    for (int g = 0; g <= ng; g++) {   /* step g: upload group g (stream sc) while the kernel of group g - 1 runs (stream sk) */
        if (g >= 1) {   /* launch g - 1 */
            const int Za = Z0 + (g - 1) * gZ, nZ = Za + gZ <= Z1 ? gZ : Z1 - Za, q = (g - 1) & 1;
            int ra = Za * P - 1; if (ra < zoff) ra = zoff;
            pool_k a; for (int k = 0; k < 10; k++) a.pl[k] = dpl[q] + k * pb;
            a.ra = ra; a.zoff = zoff; a.nzl = nzl; a.Za = Za; a.nZ = nZ; a.H = c->H; a.W = c->W; a.P = P; a.hh = c->hh; a.w = c->w; a.na = c->na; a.h = c->h; a.T = T;
            a.cy = dax; a.cx = dax + c->na; a.emin = c->emin; a.sheet0 = c->sheet0; a.kph = c->kph; a.c0 = c->c0;
            const size_t gc = (size_t)nZ * cells; a.S = dsg[q]; for (int k = 0; k < 3; k++) a.G[k] = dsg[q] + (k + 1) * gc;
            k_pool<<<(unsigned)((gc + 127) / 128), 128, 0, sk>>>(a); CF_CK(cudaGetLastError());
        }
        if (g < ng) {   /* upload g (its buffer's last kernel, g - 2, finished at step g - 1) */
            const int Za = Z0 + g * gZ, nZ = Za + gZ <= Z1 ? gZ : Z1 - Za, q = g & 1;
            int ra = Za * P - 1, rb = (Za + nZ) * P + 1; if (ra < zoff) ra = zoff; if (rb > zoff + nzl) rb = zoff + nzl;
            for (int k = 0; k < 10; k++) CF_CK(cudaMemcpyAsync(dpl[q] + k * pb, pl[k] + (size_t)(ra - zoff) * HW, (size_t)(rb - ra) * HW, cudaMemcpyHostToDevice, sc));
            CF_CK(cudaStreamSynchronize(sc));
        }
        if (g >= 1) {   /* the sums of g - 1 */
            const int Za = Z0 + (g - 1) * gZ, nZ = Za + gZ <= Z1 ? gZ : Z1 - Za, q = (g - 1) & 1; const size_t gc = (size_t)nZ * cells;
            CF_CK(cudaStreamSynchronize(sk));
            CF_CK(cudaMemcpy(S + (size_t)Za * cells, dsg[q], gc * sizeof(float), cudaMemcpyDeviceToHost));
            for (int k = 0; k < 3; k++) CF_CK(cudaMemcpy(G[k] + (size_t)Za * cells, dsg[q] + (k + 1) * gc, gc * sizeof(float), cudaMemcpyDeviceToHost));
        }
    }
    rc = 0;
fail:
    if (sk) cudaStreamSynchronize(sk);
    if (sc) cudaStreamSynchronize(sc);
    for (int q = 0; q < 2; q++) { cudaFree(dpl[q]); cudaFree(dsg[q]); }
    cudaFree(dax); if (sk) cudaStreamDestroy(sk); if (sc) cudaStreamDestroy(sc); if (rc) cudaGetLastError();
    return rc;
}

/* ---- CG (cf_cg_solve / cf_cg_run / cf_apply) ---- */
struct cg_k { int d, hh, w; size_t n; double lam; const float *wt[3], *b, *diag; float *U, *r, *p, *Ap; double alpha, beta; double *part; };
__device__ __forceinline__ float d_apply(const cg_k &L, const float *u, size_t i) {
    const size_t st[3] = {(size_t)L.hh * L.w, (size_t)L.w, 1}; const int c0[3] = {(int)(i / st[0]), (int)(i / L.w % L.hh), (int)(i % L.w)}, dm[3] = {L.d, L.hh, L.w};
    const double ui = u[i]; double s = L.lam * ui;
    for (int ax = 0; ax < 3; ax++) {
        if (c0[ax] + 1 < dm[ax]) s += L.wt[ax][i] * (ui - u[i + st[ax]]);
        if (c0[ax] > 0) s += L.wt[ax][i - st[ax]] * (ui - u[i - st[ax]]);
    }
    return (float)s;
}
__device__ __forceinline__ void d_bsum2(double a, double b, double *part) {
    __shared__ double sa[CB], sb[CB];
    sa[threadIdx.x] = a; sb[threadIdx.x] = b; __syncthreads();
    for (int s = CB / 2; s > 0; s >>= 1) { if ((int)threadIdx.x < s) { sa[threadIdx.x] += sa[threadIdx.x + s]; sb[threadIdx.x] += sb[threadIdx.x + s]; } __syncthreads(); }
    if (!threadIdx.x) { part[2 * blockIdx.x] = sa[0]; part[2 * blockIdx.x + 1] = sb[0]; }
}
#define GS(i, n) for (size_t i = blockIdx.x * (size_t)CB + threadIdx.x; i < (n); i += (size_t)gridDim.x * CB)
__global__ void k_cg0(cg_k L) { double s0 = 0, s1 = 0; GS(i, L.n) { const float ap = d_apply(L, L.U, i); L.r[i] = L.b[i] - ap; L.p[i] = L.r[i] / L.diag[i]; s0 += (double)L.r[i] * L.p[i]; s1 += (double)L.b[i] * L.b[i]; } d_bsum2(s0, s1, L.part); }
__global__ void k_cg1(cg_k L) { double s0 = 0; GS(i, L.n) { const float ap = d_apply(L, L.p, i); if (L.Ap) L.Ap[i] = ap; s0 += (double)L.p[i] * ap; } d_bsum2(s0, 0, L.part); }
__global__ void k_cg2(cg_k L) {   /* (L.Ap null, short of device memory: A p again, the same numbers: p is unchanged since k_cg1) */
    double s0 = 0, s1 = 0;
    GS(i, L.n) { const float ap = L.Ap ? L.Ap[i] : d_apply(L, L.p, i); L.U[i] += (float)(L.alpha * L.p[i]); L.r[i] -= (float)(L.alpha * ap); s0 += (double)L.r[i] * L.r[i] / L.diag[i]; s1 += (double)L.r[i] * L.r[i]; }
    d_bsum2(s0, s1, L.part);
}
__global__ void k_cg3(cg_k L) { GS(i, L.n) L.p[i] = (float)(L.r[i] / L.diag[i] + L.beta * L.p[i]); }
static int cg_sum(const cg_k &L, double *hp, double *a, double *b) {   /* the block partials, summed in block order on the host */
    if (cudaMemcpy(hp, L.part, 2 * CG * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    double s = 0, t = 0; for (int k = 0; k < CG; k++) { s += hp[2 * k]; t += hp[2 * k + 1]; }
    if (a) *a = s;
    if (b) *b = t;
    return 0;
}

extern "C" int cf_gpu_cg(int d, int hh, int w, double lam, const float *const wt[3], const float *b, const float *diag, float *U, int iters, double tol, double *relres) {
    cg_k L; memset(&L, 0, sizeof L); L.d = d; L.hh = hh; L.w = w; L.n = (size_t)d * hh * w; L.lam = lam;
    const size_t nb = L.n * sizeof(float); float *buf = nullptr; double *hp = (double *)malloc(2 * CG * sizeof(double)); int it = -1;
    const unsigned grid = (unsigned)(L.n / CB + 1 < CG ? L.n / CB + 1 : CG);
    if (!hp) return -1;
    size_t fr = 0, tot = 0; if (cudaMemGetInfo(&fr, &tot) != cudaSuccess) { cudaGetLastError(); free(hp); return -1; }
    const int na = fr > 9 * nb + ((size_t)384 << 20) && !getenv("UFSM_CF_CG_NOAP") ? 9 : 8;   /* (8: no stored A p, recomputed; env: tests) */
    if (fr < 8 * nb + ((size_t)384 << 20)) { fprintf(stderr, "count-field gpu: %.1f GB free on the device, the CG needs %.1f GB\n", fr / 1e9, 8.0 * nb / 1e9); free(hp); return -1; }
    CF_CK(cudaMalloc(&buf, na * nb)); CF_CK(cudaMalloc(&L.part, 2 * CG * sizeof(double))); CF_CK(cudaMemset(L.part, 0, 2 * CG * sizeof(double)));
    { float *q[9] = {0}; for (int k = 0; k < na; k++) q[k] = buf + k * L.n;
      for (int k = 0; k < 3; k++) { L.wt[k] = q[k]; CF_CK(cudaMemcpy(q[k], wt[k], nb, cudaMemcpyHostToDevice)); }
      L.b = q[3]; L.diag = q[4]; L.U = q[5]; L.r = q[6]; L.p = q[7]; L.Ap = q[8];
      CF_CK(cudaMemcpy(q[3], b, nb, cudaMemcpyHostToDevice)); CF_CK(cudaMemcpy(q[4], diag, nb, cudaMemcpyHostToDevice)); CF_CK(cudaMemcpy(q[5], U, nb, cudaMemcpyHostToDevice)); }
    {   /* (a grid smaller than CG blocks leaves zero partials: the sums are unchanged) */
        double rz, bb, rr = 0; k_cg0<<<grid, CB>>>(L); CF_CK(cudaGetLastError()); if (cg_sum(L, hp, &rz, &bb)) goto fail;
        const double bn = sqrt(bb > 0 ? bb : 1); it = 0;
        for (; it < iters; it++) {
            double pAp; k_cg1<<<grid, CB>>>(L); if (cg_sum(L, hp, &pAp, nullptr)) goto fail; if (pAp <= 0) break;
            L.alpha = rz / pAp; double rz1; k_cg2<<<grid, CB>>>(L); if (cg_sum(L, hp, &rz1, &rr)) goto fail;
            if (sqrt(rr) / bn < tol) { it++; break; }
            L.beta = rz1 / rz; rz = rz1; k_cg3<<<grid, CB>>>(L);
        }
        CF_CK(cudaGetLastError()); CF_CK(cudaMemcpy(U, L.U, nb, cudaMemcpyDeviceToHost));
        *relres = sqrt(rr) / bn;
    }
    cudaFree(buf); cudaFree(L.part); free(hp); return it;
fail:
    cudaFree(buf); cudaFree(L.part); free(hp); cudaGetLastError(); return -1;
}

/* ---- the --model-only R 0 model pass: per solve cell gr / gi (cf_out_run's sums), the global gauge sum ---- */
struct gs_k { ph_tab T; const uint8_t *pc, *ps, *w; int z0, z1, D, H, W, P, d, hh, w_, na, dir; double h; const float *cy, *cx, *U; float *gr, *gi; double *part; };
__device__ __forceinline__ double d_Wcell(const gs_k &M, int z, int y, int x) {   /* cf_Wcell with the gauge 0 and no --gauge ls field */
    const double P = M.P, pz = (z + 0.5) * M.h, py = (y + 0.5) * M.h, px = (x + 0.5) * M.h;
    const double U = d_tri(M.U, M.d, M.hh, M.w_, (z + 0.5) / P - 0.5, (y + 0.5) / P - 0.5, (x + 0.5) / P - 0.5) + 0;
    const double cy = d_axis_at(M.cy, M.na, pz), cx = d_axis_at(M.cx, M.na, pz);
    return U - M.dir * atan2(py - cy, px - cx) / (2 * M_PI) + 0;
}
__global__ void k_gsum(gs_k M) {
    const int Z0 = M.z0 / M.P, nZ = (M.z1 - 1) / M.P - Z0 + 1; const long long nc = (long long)nZ * M.hh * M.w_;
    const long long j = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    double ar = 0, ai = 0;
    if (j < nc) {
        const int X = (int)(j % M.w_), Y = (int)(j / M.w_ % M.hh), Z = Z0 + (int)(j / ((long long)M.w_ * M.hh));
        const size_t HW = (size_t)M.H * M.W; float gr = 0, gi = 0;
        const int ya = Y * M.P, yb = min(M.H, ya + M.P), xa = X * M.P, xb = min(M.W, xa + M.P);
        for (int zo = Z * M.P; zo < Z * M.P + M.P && zo < M.z1; zo++) {
            if (zo < M.z0) continue;
            for (int y = ya; y < yb; y++) for (int x = xa; x < xb; x++) {
                const size_t i = (size_t)(zo - M.z0) * HW + (size_t)y * M.W + x;
                if (!M.w[i]) continue;
                const double Wv = d_Wcell(M, zo, y, x); const float Wf = (float)Wv;
                const int k = M.pc[i] * 256 + M.ps[i]; const double cph = M.T.c[k];   /* (hypot / atan2 of pc, ps: the table) */
                if (cph < 1e-3) continue;
                const double ph = M.T.a[k];
                { const double e = ph - 2 * M_PI * Wf; gr += (float)(cph * cos(e)); gi += (float)(cph * sin(e)); }
                if (i % 7 == 0) { double sn, cs; sincos(ph - 2 * M_PI * Wv, &sn, &cs); ar += cph * cs; ai += cph * sn; }
            }
        }
        const size_t o = ((size_t)(Z - Z0) * M.hh + Y) * M.w_ + X; M.gr[o] = gr; M.gi[o] = gi;   /* (the chunk's solve rows) */
    }
    __shared__ double sa[128], sb[128];
    sa[threadIdx.x] = ar; sb[threadIdx.x] = ai; __syncthreads();
    for (int s = 64; s > 0; s >>= 1) { if ((int)threadIdx.x < s) { sa[threadIdx.x] += sa[threadIdx.x + s]; sb[threadIdx.x] += sb[threadIdx.x + s]; } __syncthreads(); }
    if (!threadIdx.x) { M.part[2 * blockIdx.x] = sa[0]; M.part[2 * blockIdx.x + 1] = sb[0]; }
}
static struct { gs_k M; uint8_t *pl; size_t cap; float *g; size_t gcap; double *hp; size_t pcap; } G_;
extern "C" void cf_gpu_gsum_end(void) { cudaFree(G_.pl); cudaFree((void *)G_.M.U); cudaFree((void *)G_.M.cy); cudaFree(G_.g); cudaFree(G_.M.part); free(G_.hp); memset(&G_, 0, sizeof G_); cudaGetLastError(); }
extern "C" int cf_gpu_gsum_begin(const cf_gpu_gscfg *c) {
    memset(&G_, 0, sizeof G_); gs_k &M = G_.M;
    M.D = c->D; M.H = c->H; M.W = c->W; M.P = c->P; M.d = c->d; M.hh = c->hh; M.w_ = c->w; M.na = c->na; M.dir = c->dir; M.h = c->h;
    const size_t n = (size_t)c->d * c->hh * c->w; float *u = nullptr, *ax = nullptr;
    CF_CK(cudaMalloc(&u, n * sizeof(float))); M.U = u; CF_CK(cudaMemcpy(u, c->U, n * sizeof(float), cudaMemcpyHostToDevice));
    CF_CK(cudaMalloc(&ax, 2 * (size_t)c->na * sizeof(float))); M.cy = ax; M.cx = ax + c->na;
    CF_CK(cudaMemcpy(ax, c->cy, c->na * sizeof(float), cudaMemcpyHostToDevice)); CF_CK(cudaMemcpy(ax + c->na, c->cx, c->na * sizeof(float), cudaMemcpyHostToDevice));
    if (tab_get(&M.T)) goto fail;
    return 0;
fail:
    cf_gpu_gsum_end(); return 1;
}
extern "C" int cf_gpu_gsum_chunk(const uint8_t *pc, const uint8_t *ps, const uint8_t *w, int z0, int z1, float *gr, float *gi, double acc[2]) {
    gs_k &M = G_.M; const size_t HW = (size_t)M.H * M.W, nb = (size_t)(z1 - z0) * HW, cells = (size_t)M.hh * M.w_;
    const int Z0 = z0 / M.P, nZ = (z1 - 1) / M.P - Z0 + 1; const size_t nc = (size_t)nZ * cells; const unsigned nbk = (unsigned)((nc + 127) / 128);
    if (3 * nb > G_.cap) { cudaFree(G_.pl); G_.pl = nullptr; G_.cap = 0; CF_CK(cudaMalloc(&G_.pl, 3 * nb)); G_.cap = 3 * nb; }
    if (2 * nc > G_.gcap) { cudaFree(G_.g); G_.g = nullptr; G_.gcap = 0; CF_CK(cudaMalloc(&G_.g, 2 * nc * sizeof(float))); G_.gcap = 2 * nc; }
    M.gr = G_.g; M.gi = G_.g + nc;
    if (nbk > G_.pcap) { cudaFree(M.part); M.part = nullptr; free(G_.hp); G_.pcap = 0; CF_CK(cudaMalloc(&M.part, 2 * (size_t)nbk * sizeof(double))); G_.hp = (double *)malloc(2 * (size_t)nbk * sizeof(double)); if (!G_.hp) goto fail; G_.pcap = nbk; }
    M.pc = G_.pl; M.ps = G_.pl + nb; M.w = G_.pl + 2 * nb; M.z0 = z0; M.z1 = z1;
    CF_CK(cudaMemcpy(G_.pl, pc, nb, cudaMemcpyHostToDevice)); CF_CK(cudaMemcpy(G_.pl + nb, ps, nb, cudaMemcpyHostToDevice)); CF_CK(cudaMemcpy(G_.pl + 2 * nb, w, nb, cudaMemcpyHostToDevice));
    k_gsum<<<nbk, 128>>>(M); CF_CK(cudaGetLastError());
    CF_CK(cudaMemcpy(gr + (size_t)Z0 * cells, M.gr, nc * sizeof(float), cudaMemcpyDeviceToHost));
    CF_CK(cudaMemcpy(gi + (size_t)Z0 * cells, M.gi, nc * sizeof(float), cudaMemcpyDeviceToHost));
    CF_CK(cudaMemcpy(G_.hp, M.part, 2 * (size_t)nbk * sizeof(double), cudaMemcpyDeviceToHost));
    for (unsigned k = 0; k < nbk; k++) { acc[0] += G_.hp[2 * k]; acc[1] += G_.hp[2 * k + 1]; }
    return 0;
fail:
    cudaGetLastError(); return 1;
}
