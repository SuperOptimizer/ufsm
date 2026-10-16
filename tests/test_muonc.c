/* Muon-C and Hyperball (nn_muon_ext_batch, train --muon-c / --hyperball):
   - the Muon-C direction equals a double-precision reference (orthonormal 3^3 DFT, an independent complex Newton-Schulz per
     frequency for all 27 frequencies, inverse DFT) and its RMS is the unfolded exact-polar Muon RMS; the reference's inverse
     is real (conjugate symmetry), and every frequency block of the update is near orthogonal;
   - with a 1^3 kernel Muon-C is Muon's orthogonalisation;
   - Hyperball keeps |W|_F at its first-step norm and takes the step R Normalize(W - eta R O / |O|), exempt convs unchanged;
   - both off, the ext path is nn_muon_batch. */
#include "nn.h"
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { float *p; const float *g; float *mom, *X, *Y, *A, *B; int Co, K; } muon_desc_t;                 /* mirrors nn_common.cuh */
typedef struct { const float *U; float *O; float *EX, *EY; int Co, Ci, k, m, n, tr; } muonc_desc_t;   /* mirrors nn_opt.cu */

static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); fails++; } } while (0)
static unsigned rs = 2024;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return (rs >> 8) / 16777216.f - 0.5f; }

typedef struct {
    int n, k, Co[16], Ci[16]; size_t nw[16];
    muon_desc_t h[16]; muonc_desc_t mc[16]; muon_desc_t ns[16];
    void *dd, *dmc, *dns; int maxco, maxk, nsm, nsn;
    int *hbf; float *R; double *part;
} batch;
static void batch_new(batch *b, int n, const int *Co, const int *Ci, int k) {
    memset(b, 0, sizeof *b); b->n = n; b->k = k;
    for (int i = 0; i < n; i++) {
        const int K = Ci[i] * k * k * k, m = Co[i] < Ci[i] ? Co[i] : Ci[i], nx = Co[i] < Ci[i] ? Ci[i] : Co[i];
        b->Co[i] = Co[i]; b->Ci[i] = Ci[i]; b->nw[i] = (size_t)Co[i] * K;
        muon_desc_t *d = &b->h[i];
        d->p = nn_malloc(b->nw[i] * 4); d->g = nn_malloc(b->nw[i] * 4); d->mom = nn_malloc(b->nw[i] * 4);
        d->X = nn_malloc(b->nw[i] * 4); d->Y = nn_malloc(b->nw[i] * 4); d->A = nn_malloc((size_t)Co[i] * Co[i] * 4); d->B = nn_malloc((size_t)Co[i] * Co[i] * 4);
        d->Co = Co[i]; d->K = K; nn_zero(d->p, b->nw[i] * 4); nn_zero(d->mom, b->nw[i] * 4);
        if (Co[i] > b->maxco) b->maxco = Co[i];
        if (K > b->maxk) b->maxk = K;
        muonc_desc_t *c = &b->mc[i];
        *c = (muonc_desc_t){.U = d->X, .O = d->Y, .Co = Co[i], .Ci = Ci[i], .k = k, .m = m, .n = nx, .tr = Co[i] > Ci[i]};
        c->EX = nn_malloc((size_t)4 * m * nx * 4); c->EY = nn_malloc((size_t)4 * m * nx * 4);
        b->ns[i] = (muon_desc_t){.X = c->EX, .Y = c->EY, .A = nn_malloc((size_t)16 * m * m), .B = nn_malloc((size_t)16 * m * m), .Co = 2 * m, .K = 2 * nx};
        if (2 * m > b->nsm) b->nsm = 2 * m;
        if (2 * nx > b->nsn) b->nsn = 2 * nx;
    }
    b->dd = nn_malloc(sizeof(muon_desc_t) * n); nn_h2d(b->dd, b->h, sizeof(muon_desc_t) * n);
    b->dmc = nn_malloc(sizeof(muonc_desc_t) * n); nn_h2d(b->dmc, b->mc, sizeof(muonc_desc_t) * n);
    b->dns = nn_malloc(sizeof(muon_desc_t) * n); nn_h2d(b->dns, b->ns, sizeof(muon_desc_t) * n);
    b->hbf = nn_malloc(4 * n); b->R = nn_malloc(4 * n); nn_zero(b->R, 4 * n); b->part = nn_malloc(64 * n * sizeof(double));
}
static void batch_free(batch *b) {
    for (int i = 0; i < b->n; i++) {
        muon_desc_t *d = &b->h[i];
        nn_free(d->p); nn_free((void *)d->g); nn_free(d->mom); nn_free(d->X); nn_free(d->Y); nn_free(d->A); nn_free(d->B);
        nn_free(b->mc[i].EX); nn_free(b->mc[i].EY); nn_free(b->ns[i].A); nn_free(b->ns[i].B);
    }
    nn_free(b->dd); nn_free(b->dmc); nn_free(b->dns); nn_free(b->hbf); nn_free(b->R); nn_free(b->part);
}
static void run(batch *b, int mc, const int *hb, int r_init, float lr, float beta, float wd, float hb_lr) {
    if (hb) nn_h2d(b->hbf, hb, 4 * b->n);
    nn_muon_ext_batch(b->dd, b->n, b->maxco, b->maxk, mc ? b->dmc : nullptr, b->dns, b->k, b->nsm, b->nsn, hb ? b->hbf : nullptr, b->R, r_init, b->part, lr, beta, wd, hb_lr);
}

/* ---- double-precision reference ---- */
static void ns_ref(double complex *X, int m, int n) {   /* in place: 5 complex Newton-Schulz iterations on m x n (m <= n), normalised start */
    double f = 0; for (int i = 0; i < m * n; i++) f += creal(X[i] * conj(X[i]));
    f = 1.0 / (sqrt(f) + 1e-8); for (int i = 0; i < m * n; i++) X[i] *= f;
    double complex *A = malloc(sizeof *A * m * m), *B = malloc(sizeof *B * m * m), *Y = malloc(sizeof *Y * m * n);
    for (int it = 0; it < 5; it++) {
        for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) { double complex s = 0; for (int q = 0; q < n; q++) s += X[i * n + q] * conj(X[j * n + q]); A[i * m + j] = s; }
        for (int i = 0; i < m; i++) for (int j = 0; j < m; j++) { double complex s = 0; for (int q = 0; q < m; q++) s += A[i * m + q] * A[q * m + j]; B[i * m + j] = -4.7750 * A[i * m + j] + 2.0315 * s; }
        for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) { double complex s = 0; for (int q = 0; q < m; q++) s += B[i * m + q] * X[q * n + j]; Y[i * n + j] = 3.4445 * X[i * n + j] + s; }
        memcpy(X, Y, sizeof *Y * m * n);
    }
    free(A); free(B); free(Y);
}
static double complex dft_w(int f, int t, int sign) {   /* e^{sign i 2 pi (f . t) / 3} over the 3^3 grid */
    const int ph = ((f / 9) * (t / 9) + (f / 3 % 3) * (t / 3 % 3) + (f % 3) * (t % 3)) % 3;
    return cexp(sign * I * 2.0 * M_PI * ph / 3.0);
}
/* the Muon-C direction of g [Co][Ci][27] (all 27 frequencies polarised independently); returns max |Im| / max |Re| of the inverse */
static double muonc_ref(const float *g, int Co, int Ci, double *out) {
    const int m = Co < Ci ? Co : Ci, n = Co < Ci ? Ci : Co, tr = Co > Ci;
    double complex *blk = malloc(sizeof *blk * 27 * Co * Ci), *X = malloc(sizeof *X * m * n);
    for (int f = 0; f < 27; f++) {
        for (int co = 0; co < Co; co++) for (int ci = 0; ci < Ci; ci++) {
            double complex s = 0; for (int t = 0; t < 27; t++) s += g[((size_t)co * Ci + ci) * 27 + t] * dft_w(f, t, -1);
            s /= sqrt(27.0);
            if (tr) X[ci * n + co] = conj(s); else X[co * n + ci] = s;
        }
        ns_ref(X, m, n);
        for (int co = 0; co < Co; co++) for (int ci = 0; ci < Ci; ci++) blk[((size_t)f * Co + co) * Ci + ci] = tr ? conj(X[ci * n + co]) : X[co * n + ci];
    }
    double mi = 0, mr = 0;
    for (int co = 0; co < Co; co++) for (int ci = 0; ci < Ci; ci++) for (int t = 0; t < 27; t++) {
        double complex s = 0; for (int f = 0; f < 27; f++) s += blk[((size_t)f * Co + co) * Ci + ci] * dft_w(f, t, 1);
        s /= sqrt(27.0);
        out[((size_t)co * Ci + ci) * 27 + t] = creal(s);
        mi = fmax(mi, fabs(cimag(s))); mr = fmax(mr, fabs(creal(s)));
    }
    free(blk); free(X);
    return mi / mr;
}
static double rel(const float *a, const double *b, size_t n, double sa, double sb) {   /* | a sa - b sb | / | b sb | */
    double num = 0, den = 0; for (size_t i = 0; i < n; i++) { const double d = a[i] * sa - b[i] * sb; num += d * d; den += b[i] * b[i] * sb * sb; }
    return sqrt(num / den);
}
static double nrm(const float *a, size_t n) { double s = 0; for (size_t i = 0; i < n; i++) s += (double)a[i] * a[i]; return sqrt(s); }

static void direction(void) {
    enum { N = 7 };
    const int Co[N] = {16, 16, 32, 16, 24, 40, 64}, Ci[N] = {4, 16, 16, 32, 40, 24, 64};
    batch b; batch_new(&b, N, Co, Ci, 3);
    float *gh[N];
    for (int i = 0; i < N; i++) { gh[i] = malloc(b.nw[i] * 4); for (size_t j = 0; j < b.nw[i]; j++) gh[i][j] = frand() * (1.f + (j % 5)); nn_h2d((float *)b.h[i].g, gh[i], b.nw[i] * 4); }
    run(&b, 1, nullptr, 0, 1.f, 0.f, 0.f, 0.f);   /* p = 0 - 1 * s O */
    nn_sync(); const char *e = nn_check(); CHECK(!e, "cuda %s", e ? e : "");
    for (int i = 0; i < N; i++) {
        const size_t nw = b.nw[i]; const int K = Ci[i] * 27;
        float *d = malloc(nw * 4); double *r = malloc(nw * 8);
        nn_d2h(d, b.h[i].p, nw * 4);
        const double im = muonc_ref(gh[i], Co[i], Ci[i], r);
        const double tgt = sqrt(fmin(Co[i], K) / ((double)Co[i] * K)), rr = nrm(d, nw) / sqrt((double)nw);
        double sr = 0; for (size_t j = 0; j < nw; j++) sr += r[j] * r[j];
        const double err = rel(d, r, nw, -1.0 / nrm(d, nw), 1.0 / sqrt(sr));
        printf("  muon-c %3d x %3d: direction vs reference %.2e, RMS %.6g (target %.6g), inverse imag / real %.1e\n", Co[i], Ci[i], err, rr, tgt, im);
        CHECK(err < 3e-4, "direction %d x %d", Co[i], Ci[i]);
        CHECK(fabs(rr / tgt - 1) < 1e-5, "RMS %d x %d", Co[i], Ci[i]);
        CHECK(im < 1e-9, "reference inverse not real %d x %d", Co[i], Ci[i]);
        /* every frequency block of the update: P P^* (m x m, m = min(Co, Ci)) ~ c I, c common to all blocks */
        const int m = Co[i] < Ci[i] ? Co[i] : Ci[i]; double c0 = 0, wd = 0, wo = 0;
        double complex *P = malloc(sizeof *P * Co[i] * Ci[i]);
        for (int f = 0; f < 27; f++) {
            for (int co = 0; co < Co[i]; co++) for (int ci = 0; ci < Ci[i]; ci++) {
                double complex s = 0; for (int t = 0; t < 27; t++) s += d[((size_t)co * Ci[i] + ci) * 27 + t] * dft_w(f, t, -1);
                P[co * Ci[i] + ci] = s / sqrt(27.0);
            }
            for (int a = 0; a < m; a++) for (int bb = 0; bb < m; bb++) {
                double complex s = 0;
                if (Co[i] <= Ci[i]) for (int q = 0; q < Ci[i]; q++) s += P[a * Ci[i] + q] * conj(P[bb * Ci[i] + q]);
                else for (int q = 0; q < Co[i]; q++) s += conj(P[q * Ci[i] + a]) * P[q * Ci[i] + bb];
                if (!f && !a && !bb) c0 = creal(s);
                if (a == bb) wd = fmax(wd, fabs(creal(s) / c0 - 1)); else wo = fmax(wo, cabs(s) / c0);
            }
        }
        printf("               per-frequency P P^* / c: max |diag - 1| %.3f, max |offdiag| %.3f\n", wd, wo);
        CHECK(wd < 0.6 && wo < 0.35, "frequency blocks not near orthogonal %d x %d", Co[i], Ci[i]);   /* 5 iterations: singular values ~[0.7, 1.2] */
        free(P); free(d); free(r); free(gh[i]);
    }
    batch_free(&b);
}

static void one_by_one(void) {   /* a 1^3 kernel: Muon-C = Muon's polar factor */
    enum { N = 3 };
    const int Co[N] = {16, 48, 32}, Ci[N] = {48, 16, 32};
    batch b; batch_new(&b, N, Co, Ci, 1);
    for (int i = 0; i < N; i++) { const size_t nw = b.nw[i]; float *g = malloc(nw * 4); for (size_t j = 0; j < nw; j++) g[j] = frand() * (1.f + (j % 3)); nn_h2d((float *)b.h[i].g, g, nw * 4); free(g); }
    run(&b, 1, nullptr, 0, 1.f, 0.f, 0.f, 0.f);
    for (int i = 0; i < N; i++) {
        const size_t nw = b.nw[i];
        float *rp = nn_malloc(nw * 4), *rm = nn_malloc(nw * 4), *w = nn_malloc((2 * nw + 2 * (size_t)Co[i] * Co[i]) * 4);
        nn_zero(rp, nw * 4); nn_zero(rm, nw * 4);
        nn_muon(rp, b.h[i].g, rm, Co[i], Ci[i], 1.f, 0.f, 0.f, w);
        float *a = malloc(nw * 4); double *r = malloc(nw * 8); nn_d2h(a, rp, nw * 4); for (size_t j = 0; j < nw; j++) r[j] = a[j];
        nn_d2h(a, b.h[i].p, nw * 4);
        double sr = 0; for (size_t j = 0; j < nw; j++) sr += r[j] * r[j];
        const double err = rel(a, r, nw, 1.0 / nrm(a, nw), 1.0 / sqrt(sr));
        printf("  1^3 kernel %2d x %2d: muon-c direction vs muon %.2e\n", Co[i], Ci[i], err);
        CHECK(err < 1e-4, "1x1 kernel %d x %d", Co[i], Ci[i]);
        nn_free(rp); nn_free(rm); nn_free(w); free(a); free(r);
    }
    batch_free(&b);
}

static void hyperball(int mc) {
    enum { N = 4 };
    const int Co[N] = {16, 32, 16, 64}, Ci[N] = {4, 16, 16, 64}, hb[N] = {1, 1, 0, 1};   /* conv 2 exempt */
    batch b, r; batch_new(&b, N, Co, Ci, 3); batch_new(&r, N, Co, Ci, 3);
    float *w0[N], *gg[N];
    for (int i = 0; i < N; i++) {
        const size_t nw = b.nw[i]; const double sd = sqrt(2.0 / (Ci[i] * 27));
        w0[i] = malloc(nw * 4); gg[i] = malloc(nw * 4);
        for (size_t j = 0; j < nw; j++) { w0[i][j] = (float)(sd * 3.4641 * frand()); gg[i][j] = frand(); }
        nn_h2d(b.h[i].p, w0[i], nw * 4); nn_h2d((float *)b.h[i].g, gg[i], nw * 4); nn_h2d((float *)r.h[i].g, gg[i], nw * 4);
    }
    const float lr = 0.02f, wd = 0.01f, eta = 0.0141f;
    run(&r, mc, nullptr, 0, 1.f, 0.f, 0.f, 0.f);    /* the direction: r.p = -s O */
    run(&b, mc, hb, 1, lr, 0.f, wd, eta);           /* the first Hyperball step sets R = |W0| */
    float Rh[N]; nn_d2h(Rh, b.R, 4 * N);
    for (int i = 0; i < N; i++) {
        const size_t nw = b.nw[i]; float *o = malloc(nw * 4), *p = malloc(nw * 4); double *ref = malloc(nw * 8);
        nn_d2h(o, r.h[i].p, nw * 4); nn_d2h(p, b.h[i].p, nw * 4);
        const double R0 = nrm(w0[i], nw), no = nrm(o, nw);
        if (hb[i]) {
            double s = 0; for (size_t j = 0; j < nw; j++) { ref[j] = w0[i][j] + eta * R0 * o[j] / no; s += ref[j] * ref[j]; }   /* o = -s O */
            for (size_t j = 0; j < nw; j++) ref[j] *= R0 / sqrt(s);
        } else {   /* exempt: the plain update p (1 - lr wd) - lr s O, s O = -o */
            for (size_t j = 0; j < nw; j++) ref[j] = (double)w0[i][j] * (1.0 - lr * wd) + lr * o[j];
        }
        const double err = rel(p, ref, nw, 1, 1), np = nrm(p, nw);
        printf("  hyperball (%s) conv %d %2d x %2d %s: step vs reference %.2e, |W| / R %.7f, R / |W0| %.7f\n", mc ? "muon-c" : "muon", i, Co[i], Ci[i],
               hb[i] ? "on    " : "exempt", err, np / R0, Rh[i] > 0 ? Rh[i] / R0 : 0.0);
        CHECK(err < 2e-5, "hyperball step %d", i);
        if (hb[i]) { CHECK(fabs(np / R0 - 1) < 1e-6, "norm %d", i); CHECK(fabs(Rh[i] / R0 - 1) < 1e-6, "radius %d", i); }
        else CHECK(Rh[i] == 0.f, "exempt conv got a radius");
        free(o); free(p); free(ref);
    }
    /* a second step keeps R (no re-init) and the norm */
    for (int i = 0; i < N; i++) { for (size_t j = 0; j < b.nw[i]; j++) gg[i][j] = frand(); nn_h2d((float *)b.h[i].g, gg[i], b.nw[i] * 4); }
    run(&b, mc, hb, 0, lr, 0.f, wd, eta);
    float R2[N]; nn_d2h(R2, b.R, 4 * N);
    for (int i = 0; i < N; i++) if (hb[i]) {
        float *p = malloc(b.nw[i] * 4); nn_d2h(p, b.h[i].p, b.nw[i] * 4);
        CHECK(R2[i] == Rh[i] && fabs(nrm(p, b.nw[i]) / Rh[i] - 1) < 1e-6, "second step radius / norm %d", i);
        free(p);
    }
    for (int i = 0; i < N; i++) { free(w0[i]); free(gg[i]); }
    batch_free(&b); batch_free(&r);
}

static void both_off(void) {   /* nn_muon_ext_batch without Muon-C or Hyperball == nn_muon_batch */
    enum { N = 4 };
    const int Co[N] = {16, 40, 96, 128}, Ci[N] = {4, 37, 224, 96};
    batch a, b; batch_new(&a, N, Co, Ci, 3); batch_new(&b, N, Co, Ci, 3);
    for (int i = 0; i < N; i++) {
        const size_t nw = a.nw[i]; float *w = malloc(nw * 4); for (size_t j = 0; j < nw; j++) w[j] = frand() * 0.1f;
        nn_h2d(a.h[i].p, w, nw * 4); nn_h2d(b.h[i].p, w, nw * 4); free(w);
    }
    for (int step = 0; step < 3; step++) {
        for (int i = 0; i < N; i++) { const size_t nw = a.nw[i]; float *g = malloc(nw * 4); for (size_t j = 0; j < nw; j++) g[j] = frand(); nn_h2d((float *)a.h[i].g, g, nw * 4); nn_h2d((float *)b.h[i].g, g, nw * 4); free(g); }
        nn_muon_batch(a.dd, N, a.maxco, a.maxk, 0.02f, 0.95f, 0.01f);
        run(&b, 0, nullptr, 0, 0.02f, 0.95f, 0.01f, 0.f);
    }
    for (int i = 0; i < N; i++) {
        const size_t nw = a.nw[i]; float *x = malloc(nw * 4), *y = malloc(nw * 4); double *yd = malloc(nw * 8);
        nn_d2h(x, a.h[i].p, nw * 4); nn_d2h(y, b.h[i].p, nw * 4); for (size_t j = 0; j < nw; j++) yd[j] = y[j];
        const double err = rel(x, yd, nw, 1, 1);
        printf("  both off == nn_muon_batch, %3d x %3d: rel %.2e\n", Co[i], Ci[i], err);
        CHECK(err < 1e-6, "both off %d", i);
        free(x); free(y); free(yd);
    }
    batch_free(&a); batch_free(&b);
}

int main(void) {
    if (nn_init(0)) return 1;
    direction();
    one_by_one();
    hyperball(0);
    hyperball(1);
    both_off();
    nn_sync(); const char *e = nn_check(); CHECK(!e, "cuda %s", e ? e : "");
    printf(fails ? "muon-c / hyperball FAIL (%d)\n" : "muon-c / hyperball ok\n", fails);
    return fails != 0;
}
