/* --muon-ns polar and --normuon 1 (nn_muon_ext_batch ns_mode / nm_rows; DESIGN.md 2026-11-13):
   - both off, the ext path is nn_muon_batch bit for bit (and train without the flags never enters it: unet_muon's mx_any);
   - polar: the update equals a double-precision 5-step Newton-Schulz with the Polar Express maps, and its singular values sit
     closer to 1 than the fixed map's (the direction's O O^T diagonal / off-diagonal spread);
   - normuon: over 3 steps of Nesterov momentum, the update equals Muon's direction with each row scaled by rsqrt(EMA of its mean
     square) and the whole rescaled to Muon's Frobenius norm (a double reference fed the GPU's own Muon direction), and after the first
     step every row has the same RMS. Shapes cover Co < K, Co > K and square. */
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { float *p; const float *g; float *mom, *X, *Y, *A, *B; int Co, K; } muon_desc_t;   /* mirrors nn_common.cuh */

static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); fails++; } } while (0)
static unsigned rs = 77;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return (rs >> 8) / 16777216.f - 0.5f; }

enum { N = 5 };
static const int CO[N] = {16, 48, 64, 96, 40}, KK[N] = {108, 16, 64, 864, 1080};
typedef struct { muon_desc_t h[N]; void *dd; int maxco, maxk; double *part; float *rows; int *off; int tot; float *prev[N]; void *dprev; } batch;
static void batch_new(batch *b) {
    memset(b, 0, sizeof *b); int off[N + 1], t = 0;
    for (int i = 0; i < N; i++) {
        const size_t nw = (size_t)CO[i] * KK[i]; muon_desc_t *d = &b->h[i];
        d->p = nn_malloc(nw * 4); d->g = nn_malloc(nw * 4); d->mom = nn_malloc(nw * 4); d->X = nn_malloc(nw * 4); d->Y = nn_malloc(nw * 4);
        d->A = nn_malloc((size_t)CO[i] * CO[i] * 4); d->B = nn_malloc((size_t)CO[i] * CO[i] * 4); d->Co = CO[i]; d->K = KK[i];
        nn_zero(d->p, nw * 4); nn_zero(d->mom, nw * 4);
        if (CO[i] > b->maxco) b->maxco = CO[i];
        if (KK[i] > b->maxk) b->maxk = KK[i];
        off[i] = t; t += CO[i];
    }
    off[N] = t; b->tot = t;
    b->dd = nn_malloc(sizeof b->h); nn_h2d(b->dd, b->h, sizeof b->h);
    b->part = nn_malloc((size_t)32 * N * 16 * sizeof(double)); b->rows = nn_malloc((size_t)2 * t * 4); nn_zero(b->rows, (size_t)2 * t * 4);
    b->off = nn_malloc((N + 1) * sizeof(int)); nn_h2d(b->off, off, (N + 1) * sizeof(int));
    for (int i = 0; i < N; i++) { const size_t nw = (size_t)CO[i] * KK[i]; b->prev[i] = nn_malloc(nw * 4); nn_zero(b->prev[i], nw * 4); }
    b->dprev = nn_malloc(sizeof b->prev); nn_h2d(b->dprev, b->prev, sizeof b->prev);
}
static void batch_free(batch *b) {
    for (int i = 0; i < N; i++) { muon_desc_t *d = &b->h[i]; nn_free(d->p); nn_free((void *)d->g); nn_free(d->mom); nn_free(d->X); nn_free(d->Y); nn_free(d->A); nn_free(d->B); }
    nn_free(b->dd); nn_free(b->part); nn_free(b->rows); nn_free(b->off); for (int i = 0; i < N; i++) nn_free(b->prev[i]); nn_free(b->dprev);
}
static void runv(batch *b, int ns, int nm, float lr, float beta, float wd, float vr) {
    nn_muon_ext_batch(b->dd, N, b->maxco, b->maxk, nullptr, nullptr, 3, 0, 0, nullptr, nullptr, 0, b->part, lr, beta, wd, 0.f, ns, nm ? b->rows : nullptr, b->off, b->tot, 0.9f,
                      vr >= 0.f ? (float *const *)b->dprev : nullptr, vr < 0.f ? 0.f : vr);
}
static void run(batch *b, int ns, int nm, float lr, float beta, float wd) { runv(b, ns, nm, lr, beta, wd, -1.f); }
static void set_g(batch *a, batch *b, unsigned seed) {
    rs = seed;
    for (int i = 0; i < N; i++) {
        const size_t nw = (size_t)CO[i] * KK[i]; float *g = malloc(nw * 4); for (size_t j = 0; j < nw; j++) g[j] = frand() * (1.f + (j % 7) * 0.3f) * (1.f + (j / KK[i]) % 3);
        nn_h2d((float *)a->h[i].g, g, nw * 4); if (b) nn_h2d((float *)b->h[i].g, g, nw * 4); free(g);
    }
}
static double rel(const float *a, const double *b, size_t n) { double num = 0, den = 0; for (size_t i = 0; i < n; i++) { const double d = a[i] - b[i]; num += d * d; den += b[i] * b[i]; } return sqrt(num / den); }

static const double PE[5][3] = {{8.470329, -25.108079, 18.629279}, {4.182835, -3.108702, 0.580607}, {3.961857, -2.954063, 0.562976}, {3.286586, -2.464720, 0.507358}, {2.273750, -1.644660, 0.416191}};
static void ns_ref(double *X, int m, int n, int polar) {   /* 5 Newton-Schulz steps on m x n, normalised start (as bmuon_scale_k) */
    double f = 0; for (int i = 0; i < m * n; i++) f += X[i] * X[i];
    f = 1.0 / (sqrt(f) + 1e-7); for (int i = 0; i < m * n; i++) X[i] *= f;
    double *A = malloc(sizeof *A * m * m), *B = malloc(sizeof *B * m * m), *Y = malloc(sizeof *Y * m * n);
    for (int it = 0; it < 5; it++) {
        const double a = polar ? PE[it][0] : 3.4445, b = polar ? PE[it][1] : -4.7750, c = polar ? PE[it][2] : 2.0315;
        for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) { double s = 0; for (int q = 0; q < n; q++) s += X[i * n + q] * X[j * n + q]; A[i * m + j] = s; }
        for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) { double s = 0; for (int q = 0; q < m; q++) s += A[i * m + q] * A[q * m + j]; B[i * m + j] = b * A[i * m + j] + c * s; }
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) { double s = 0; for (int q = 0; q < m; q++) s += B[i * m + q] * X[q * n + j]; Y[i * n + j] = a * X[i * n + j] + s; }
        memcpy(X, Y, sizeof *Y * m * n);
    }
    free(A); free(B); free(Y);
}
static void spread(const float *o, int Co, int K, double *wd, double *wo) {   /* O O^T / mean diag (Co <= K) or O^T O (Co > K): max |diag - 1|, max |offdiag| */
    const int m = Co <= K ? Co : K; double *G = calloc((size_t)m * m, sizeof *G), tr = 0;
    for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) {
        double s = 0;
        if (Co <= K) for (int q = 0; q < K; q++) s += (double)o[(size_t)a * K + q] * o[(size_t)b * K + q];
        else for (int q = 0; q < Co; q++) s += (double)o[(size_t)q * K + a] * o[(size_t)q * K + b];
        G[a * m + b] = s; if (a == b) tr += s;
    }
    tr /= m; *wd = *wo = 0;
    for (int a = 0; a < m; a++) for (int b = 0; b < m; b++) { const double v = G[a * m + b] / tr; if (a == b) *wd = fmax(*wd, fabs(v - 1)); else *wo = fmax(*wo, fabs(v)); }
    free(G);
}

static void off_is_muon(void) {   /* ext without any variant == nn_muon_batch, bit for bit, over 3 steps with momentum and decay */
    batch a, b; batch_new(&a); batch_new(&b);
    for (int i = 0; i < N; i++) { const size_t nw = (size_t)CO[i] * KK[i]; float *w = malloc(nw * 4); for (size_t j = 0; j < nw; j++) w[j] = frand() * 0.1f; nn_h2d(a.h[i].p, w, nw * 4); nn_h2d(b.h[i].p, w, nw * 4); free(w); }
    for (int s = 0; s < 3; s++) { set_g(&a, &b, 100 + s); nn_muon_batch(a.dd, N, a.maxco, a.maxk, 0.02f, 0.95f, 0.01f); run(&b, 0, 0, 0.02f, 0.95f, 0.01f); }
    for (int i = 0; i < N; i++) {
        const size_t nw = (size_t)CO[i] * KK[i]; float *x = malloc(nw * 4), *y = malloc(nw * 4); nn_d2h(x, a.h[i].p, nw * 4); nn_d2h(y, b.h[i].p, nw * 4);
        const int same = !memcmp(x, y, nw * 4); printf("  variants off == nn_muon_batch, %3d x %4d: %s\n", CO[i], KK[i], same ? "bit for bit" : "DIFFERENT");
        CHECK(same, "off path %d", i); free(x); free(y);
    }
    batch_free(&a); batch_free(&b);
}

static void polar(void) {
    batch a, f; batch_new(&a); batch_new(&f); set_g(&a, &f, 7);
    run(&a, 1, 0, 1.f, 0.f, 0.f); run(&f, 0, 0, 1.f, 0.f, 0.f);   /* p = -sqrt(max(1, Co / K)) O */
    rs = 7;
    for (int i = 0; i < N; i++) {
        const int Co = CO[i], K = KK[i]; const size_t nw = (size_t)Co * K; const double sc = sqrt(fmax(1.0, (double)Co / K));
        float *g = malloc(nw * 4); for (size_t j = 0; j < nw; j++) g[j] = frand() * (1.f + (j % 7) * 0.3f) * (1.f + (j / K) % 3);
        double *r = malloc(nw * 8); for (size_t j = 0; j < nw; j++) r[j] = g[j];
        ns_ref(r, Co, K, 1); for (size_t j = 0; j < nw; j++) r[j] *= -sc;   /* (the GPU iterates Co x K as given, Co > K too) */
        float *p = malloc(nw * 4), *q = malloc(nw * 4); nn_d2h(p, a.h[i].p, nw * 4); nn_d2h(q, f.h[i].p, nw * 4);
        const double e = rel(p, r, nw); double pd, po, fd, fo; spread(p, Co, K, &pd, &po); spread(q, Co, K, &fd, &fo);
        printf("  polar %3d x %4d: vs double reference %.2e; O O^T spread |diag - 1| %.3f |off| %.3f (fixed maps %.3f / %.3f)\n", Co, K, e, pd, po, fd, fo);
        CHECK(e < 3e-4, "polar direction %d", i);
        CHECK(pd < fd && pd < 0.3, "polar spread %d: %.3f vs fixed %.3f", i, pd, fd);
        free(g); free(r); free(p); free(q);
    }
    batch_free(&a); batch_free(&f);
}

static void normuon(void) {
    batch m, n; batch_new(&m); batch_new(&n);
    double *E[N]; for (int i = 0; i < N; i++) E[i] = calloc(CO[i], sizeof(double));
    const double b2 = 0.9;
    for (int s = 0; s < 3; s++) {
        set_g(&m, &n, 300 + s);
        for (int i = 0; i < N; i++) { const size_t nw = (size_t)CO[i] * KK[i]; nn_zero(m.h[i].p, nw * 4); nn_zero(n.h[i].p, nw * 4); }
        run(&m, 0, 0, 1.f, 0.95f, 0.f); run(&n, 0, 1, 1.f, 0.95f, 0.f);   /* one momentum history each, identical gradients: p = -s O */
        for (int i = 0; i < N; i++) {
            const int Co = CO[i], K = KK[i]; const size_t nw = (size_t)Co * K;
            float *o = malloc(nw * 4), *u = malloc(nw * 4); nn_d2h(o, m.h[i].p, nw * 4); nn_d2h(u, n.h[i].p, nw * 4);
            double pre = 0, post = 0, *pw = malloc(Co * sizeof *pw), *r = malloc(nw * 8);
            for (int a = 0; a < Co; a++) { double t = 0; for (int k = 0; k < K; k++) t += (double)o[(size_t)a * K + k] * o[(size_t)a * K + k]; pw[a] = t / K; pre += pw[a]; E[i][a] += (1 - b2) * (pw[a] - E[i][a]); post += pw[a] / E[i][a]; }
            const double sc = sqrt(pre / post);
            for (int a = 0; a < Co; a++) for (int k = 0; k < K; k++) r[(size_t)a * K + k] = o[(size_t)a * K + k] * sc / sqrt(E[i][a]);
            const double e = rel(u, r, nw); double fo = 0, fu = 0, rmin = 1e30, rmax = 0;
            for (size_t j = 0; j < nw; j++) { fo += (double)o[j] * o[j]; fu += (double)u[j] * u[j]; }
            for (int a = 0; a < Co; a++) { double t = 0; for (int k = 0; k < K; k++) t += (double)u[(size_t)a * K + k] * u[(size_t)a * K + k]; rmin = fmin(rmin, t); rmax = fmax(rmax, t); }
            printf("  normuon step %d %3d x %4d: vs reference %.2e, |U|_F / |O|_F %.7f, row RMS max / min %.4f\n", s, Co, K, e, sqrt(fu / fo), sqrt(rmax / rmin));
            CHECK(e < 2e-5, "normuon step %d conv %d", s, i);
            CHECK(fabs(sqrt(fu / fo) - 1) < 1e-5, "normuon norm step %d conv %d", s, i);
            if (!s) CHECK(sqrt(rmax / rmin) < 1.0001, "first step rows not equalised conv %d", i);
            free(o); free(u); free(pw); free(r);
        }
    }
    for (int i = 0; i < N; i++) free(E[i]);
    batch_free(&m); batch_free(&n);
}

static void vr(void) {
    batch a, b, v; batch_new(&a); batch_new(&b); batch_new(&v);
    for (int i = 0; i < N; i++) { const size_t nw = (size_t)CO[i] * KK[i]; float *w = malloc(nw * 4); for (size_t j = 0; j < nw; j++) w[j] = frand() * 0.1f; nn_h2d(a.h[i].p, w, nw * 4); nn_h2d(b.h[i].p, w, nw * 4); free(w); }
    double *m[N], *gp[N]; for (int i = 0; i < N; i++) { m[i] = calloc((size_t)CO[i] * KK[i], 8); gp[i] = calloc((size_t)CO[i] * KK[i], 8); }
    const double G = 0.025, beta = 0.95, k = G * beta / (1 - beta);
    for (int s = 0; s < 3; s++) {
        set_g(&a, &b, 500 + s); set_g(&v, nullptr, 500 + s);
        nn_muon_batch(a.dd, N, a.maxco, a.maxk, 0.02f, 0.95f, 0.01f); runv(&b, 0, 0, 0.02f, 0.95f, 0.01f, 0.f); runv(&v, 0, 0, 0.02f, 0.95f, 0.01f, (float)G);
        rs = 500 + s;
        for (int i = 0; i < N; i++) {
            const size_t nw = (size_t)CO[i] * KK[i]; double cn = 0; float *g = malloc(nw * 4), *mm = malloc(nw * 4); double *c = malloc(nw * 8), *r = malloc(nw * 8);
            for (size_t j = 0; j < nw; j++) g[j] = frand() * (1.f + (j % 7) * 0.3f) * (1.f + (j / KK[i]) % 3);
            for (size_t j = 0; j < nw; j++) { c[j] = g[j] + k * (g[j] - gp[i][j]); gp[i][j] = g[j]; cn += c[j] * c[j]; }
            const double f = sqrt(cn) > 1 ? 1 / sqrt(cn) : 1;
            for (size_t j = 0; j < nw; j++) { m[i][j] = beta * m[i][j] + c[j] * f; r[j] = m[i][j]; }
            nn_d2h(mm, v.h[i].mom, nw * 4); const double e = rel(mm, r, nw);
            printf("  muon-vr step %d %3d x %4d: momentum vs reference %.2e (|c| %.3g)\n", s, CO[i], KK[i], e, sqrt(cn));
            CHECK(e < 1e-5, "muon-vr momentum step %d conv %d", s, i);
            free(g); free(mm); free(c); free(r);
        }
    }
    for (int i = 0; i < N; i++) {
        const size_t nw = (size_t)CO[i] * KK[i]; float *x = malloc(nw * 4), *y = malloc(nw * 4); nn_d2h(x, a.h[i].p, nw * 4); nn_d2h(y, b.h[i].p, nw * 4);
        const int same = !memcmp(x, y, nw * 4); printf("  muon-vr G 0 == nn_muon_batch, %3d x %4d: %s\n", CO[i], KK[i], same ? "bit for bit" : "DIFFERENT");
        CHECK(same, "vr G 0 %d", i); free(x); free(y); free(m[i]); free(gp[i]);
    }
    batch_free(&a); batch_free(&b); batch_free(&v);
}

static void jacobi_root(const double *A0, double *R, int n, double expo) {   /* R = A^expo for symmetric positive definite A (cyclic Jacobi) */
    double *A = malloc(sizeof *A * n * n), *V = calloc((size_t)n * n, sizeof *V); memcpy(A, A0, sizeof *A * n * n);
    for (int i = 0; i < n; i++) V[i * n + i] = 1;
    for (int sw = 0; sw < 60; sw++) {
        double off = 0; for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) off += A[i * n + j] * A[i * n + j];
        if (off < 1e-30) break;
        for (int p = 0; p < n; p++) for (int q = p + 1; q < n; q++) {
            if (fabs(A[p * n + q]) < 1e-300) continue;
            const double th = (A[q * n + q] - A[p * n + p]) / (2 * A[p * n + q]), t = (th >= 0 ? 1 : -1) / (fabs(th) + sqrt(th * th + 1)), c = 1 / sqrt(t * t + 1), sn = t * c;
            for (int k = 0; k < n; k++) { const double a = A[k * n + p], b = A[k * n + q]; A[k * n + p] = c * a - sn * b; A[k * n + q] = sn * a + c * b; }
            for (int k = 0; k < n; k++) { const double a = A[p * n + k], b = A[q * n + k]; A[p * n + k] = c * a - sn * b; A[q * n + k] = sn * a + c * b; }
            for (int k = 0; k < n; k++) { const double a = V[k * n + p], b = V[k * n + q]; V[k * n + p] = c * a - sn * b; V[k * n + q] = sn * a + c * b; }
        }
    }
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) { double s = 0; for (int k = 0; k < n; k++) s += V[i * n + k] * pow(A[k * n + k], expo) * V[j * n + k]; R[i * n + j] = s; }
    free(A); free(V);
}
static void cautious(void) {   /* one step from p = 0, wd 0: the cautious update = the plain one masked where its sign disagrees with g, norm kept */
    batch a, b; batch_new(&a); batch_new(&b); set_g(&a, &b, 900);
    for (int i = 0; i < N; i++) { const size_t nw = (size_t)CO[i] * KK[i]; nn_zero(a.h[i].p, nw * 4); nn_zero(b.h[i].p, nw * 4); }
    run(&a, 0, 0, 0.02f, 0.95f, 0.f); nn_muon_ext_opts(1); run(&b, 0, 0, 0.02f, 0.95f, 0.f); nn_muon_ext_opts(0);
    rs = 900;
    for (int i = 0; i < N; i++) {
        const size_t nw = (size_t)CO[i] * KK[i]; float *g = malloc(nw * 4), *u = malloc(nw * 4), *c = malloc(nw * 4); double *r = malloc(nw * 8), n0 = 0, n1 = 0;
        for (size_t j = 0; j < nw; j++) g[j] = frand() * (1.f + (j % 7) * 0.3f) * (1.f + (j / KK[i]) % 3);
        nn_d2h(u, a.h[i].p, nw * 4); nn_d2h(c, b.h[i].p, nw * 4);
        for (size_t j = 0; j < nw; j++) { n0 += (double)u[j] * u[j]; r[j] = -u[j] * g[j] > 0 ? -u[j] : 0; n1 += r[j] * r[j]; }   /* (p = -lr s O) */
        for (size_t j = 0; j < nw; j++) r[j] *= -sqrt(n0 / n1);
        const double e = rel(c, r, nw); printf("  muon-cautious %3d x %4d: vs reference %.2e (kept %.0f%% of the norm before rescale)\n", CO[i], KK[i], e, 100 * sqrt(n1 / n0));
        CHECK(e < 1e-5, "cautious %d", i); free(g); free(u); free(c); free(r);
    }
    batch_free(&a); batch_free(&b);
}
static void tc(void) {   /* fp16 tensor-core Newton-Schulz vs the fp32 path: 3 Muon steps, the parameter updates' relative difference */
    batch a, b; batch_new(&a); batch_new(&b);
    for (int i = 0; i < N; i++) { const size_t nw = (size_t)CO[i] * KK[i]; nn_zero(a.h[i].p, nw * 4); nn_zero(b.h[i].p, nw * 4); }
    for (int s = 0; s < 3; s++) { set_g(&a, &b, 700 + s); nn_muon_batch(a.dd, N, a.maxco, a.maxk, 0.02f, 0.95f, 0.f); nn_muon_tc(1); nn_muon_batch(b.dd, N, b.maxco, b.maxk, 0.02f, 0.95f, 0.f); nn_muon_tc(0); }
    for (int i = 0; i < N; i++) {
        const size_t nw = (size_t)CO[i] * KK[i]; float *x = malloc(nw * 4), *y = malloc(nw * 4); double *r = malloc(nw * 8);
        nn_d2h(x, a.h[i].p, nw * 4); nn_d2h(y, b.h[i].p, nw * 4); for (size_t j = 0; j < nw; j++) r[j] = x[j];
        const double e = rel(y, r, nw); printf("  muon-tc %3d x %4d: fp16 tensor cores vs fp32, 3 steps: %.2e\n", CO[i], KK[i], e);
        CHECK(e < 1e-2, "tc %d", i); free(x); free(y); free(r);
    }
    batch_free(&a); batch_free(&b);
    for (int m = 2; m <= 2; m++) {   /* cuBLASLt Newton-Schulz: TF32 tensor cores vs fp32 */
        batch_new(&a); batch_new(&b);
        for (int i = 0; i < N; i++) { const size_t nw = (size_t)CO[i] * KK[i]; nn_zero(a.h[i].p, nw * 4); nn_zero(b.h[i].p, nw * 4); }
        for (int s = 0; s < 3; s++) { set_g(&a, &b, 700 + s); nn_muon_batch(a.dd, N, a.maxco, a.maxk, 0.02f, 0.95f, 0.f); nn_muon_tc(m); nn_muon_batch(b.dd, N, b.maxco, b.maxk, 0.02f, 0.95f, 0.f); nn_muon_tc(0); }
        for (int i = 0; i < N; i++) {
            const size_t nw = (size_t)CO[i] * KK[i]; float *x = malloc(nw * 4), *y = malloc(nw * 4); double *r = malloc(nw * 8);
            nn_d2h(x, a.h[i].p, nw * 4); nn_d2h(y, b.h[i].p, nw * 4); for (size_t j = 0; j < nw; j++) r[j] = x[j];
            const double e = rel(y, r, nw); printf("  muon-tc %d %3d x %4d: cuBLASLt %s vs fp32, 3 steps: %.2e\n", m, CO[i], KK[i], "TF32", e);
            CHECK(e < 1e-2, "tc%d %d", m, i); free(x); free(y); free(r);
        }
        batch_free(&a); batch_free(&b);
    }
}
static void kron(void) {
    enum { NC = 2 }; const int Co[NC] = {12, 24}, Ci[NC] = {20, 8}; const float eps = 1e-3f, b2 = 0.9f;
    float *g[NC]; double *L[NC][3];
    for (int c = 0; c < NC; c++) { const int d[3] = {Co[c], Ci[c], 27}; g[c] = nn_malloc((size_t)Co[c] * Ci[c] * 27 * 4); for (int f = 0; f < 3; f++) L[c][f] = calloc((size_t)d[f] * d[f], 8); }
    struct nn_kron *k = nn_kron_create(NC, g, Co, Ci);
    for (int s = 0; s < 2; s++) {
        for (int c = 0; c < NC; c++) {
            const int n = Co[c] * Ci[c] * 27, d[3] = {Co[c], Ci[c], 27}; float *h = malloc(n * 4), *o = malloc(n * 4); double *G = malloc(n * 8), *P[3], *ref = calloc(n, 8);
            for (int j = 0; j < n; j++) { h[j] = frand() * (1.f + (j % 27) * 0.1f) * (1.f + (j / (Ci[c] * 27)) % 4); G[j] = h[j]; }
            nn_h2d(g[c], h, n * 4);
            /* the Grams and their EMA (step 0: the Grams themselves) */
            for (int f = 0; f < 3; f++) {
                const int m = d[f]; double *gr = calloc((size_t)m * m, 8);
                for (int o1 = 0; o1 < Co[c]; o1++) for (int i1 = 0; i1 < Ci[c]; i1++) for (int t1 = 0; t1 < 27; t1++) {
                    const int idx[3] = {o1, i1, t1}; const double v = G[(o1 * Ci[c] + i1) * 27 + t1];
                    for (int a = 0; a < m; a++) {   /* the other index equal, mode f index a */
                        int jd[3] = {o1, i1, t1}; jd[f] = a; gr[idx[f] * m + a] += v * G[(jd[0] * Ci[c] + jd[1]) * 27 + jd[2]];
                    }
                }
                for (int q = 0; q < m * m; q++) L[c][f][q] = s ? b2 * L[c][f][q] + (1 - b2) * gr[q] : gr[q];
                free(gr);
                double fr = 0; for (int q = 0; q < m * m; q++) fr += L[c][f][q] * L[c][f][q]; fr = sqrt(fr);
                double *A = malloc(sizeof *A * m * m); memcpy(A, L[c][f], sizeof *A * m * m); for (int q = 0; q < m; q++) A[q * m + q] += eps * fr;
                P[f] = malloc(sizeof *P[f] * m * m); jacobi_root(A, P[f], m, -1.0 / 6); free(A);
            }
            for (int o1 = 0; o1 < Co[c]; o1++) for (int i1 = 0; i1 < Ci[c]; i1++) for (int t1 = 0; t1 < 27; t1++) {
                double acc = 0;
                for (int o2 = 0; o2 < Co[c]; o2++) for (int i2 = 0; i2 < Ci[c]; i2++) for (int t2 = 0; t2 < 27; t2++)
                    acc += P[0][o1 * Co[c] + o2] * P[1][i1 * Ci[c] + i2] * P[2][t1 * 27 + t2] * G[(o2 * Ci[c] + i2) * 27 + t2];
                ref[(o1 * Ci[c] + i1) * 27 + t1] = acc;
            }
            free(h); free(G); for (int f = 0; f < 3; f++) free(P[f]);
            if (c == NC - 1) {   /* every conv's gradient uploaded: one step (every = 1: all roots refreshed each step) */
                nn_kron_step(k, b2, eps, 1); nn_sync();
            }
            o = realloc(o, n * 4); (void)o; free(o);
            static double *refs[NC]; refs[c] = ref;
            if (c == NC - 1) for (int cc = 0; cc < NC; cc++) {
                const int nn = Co[cc] * Ci[cc] * 27; float *got = malloc(nn * 4); nn_d2h(got, g[cc], nn * 4);
                const double e = rel(got, refs[cc], nn);
                printf("  muon-kron step %d conv %2d x %2d x 27: vs dense double reference %.2e\n", s, Co[cc], Ci[cc], e);
                CHECK(e < 2e-3, "kron step %d conv %d", s, cc); free(got); free(refs[cc]);
            }
        }
    }
    nn_kron_free(k); for (int c = 0; c < NC; c++) { nn_free(g[c]); for (int f = 0; f < 3; f++) free(L[c][f]); }
}

int main(void) {
    if (nn_init(0)) return 1;
    off_is_muon();
    polar();
    normuon();
    vr();
    cautious();
    tc();
    kron();
    nn_sync(); const char *e = nn_check(); CHECK(!e, "cuda %s", e ? e : "");
    printf(fails ? "muon-ns polar / normuon / muon-vr / muon-kron / cautious FAIL (%d)\n" : "muon-ns polar / normuon / muon-vr / muon-kron / cautious ok\n", fails);
    return fails != 0;
}
