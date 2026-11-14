/* Fully adaptive Muon (DESIGN.md 2026-11-14 "Fully adaptive Muon"):
   - --opt-stats: nn_muon_stats changes no weight (bit for bit against a run without it) and its per-matrix |update|^2, |W|^2 equal the
     host's sums of the actual weight change and the weights before the step (plain batched Muon and the ext path with NorMuon);
   - adaptive Muon (nn_muon_ext_adapt): over 5 steps with a varying schedule and decay, the weights, the (b) ratio EMAs and the (c) DoG
     state equal a double-precision reference fed the GPU's own Muon direction (from a twin batch at lr 1, wd 0, weights reset each
     step: O = -dp / s): (b) with the geometric-mean target, (b) with an absolute target that hits the clamp, (c) alone, and NorMuon +
     (b) + (c) together. */
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
typedef struct { muon_desc_t h[N]; void *dd; int maxco, maxk; double *part; float *rows; int *off; int tot; } batch;
static void batch_new(batch *b, unsigned seed) {
    memset(b, 0, sizeof *b); int off[N + 1], t = 0; rs = seed;
    for (int i = 0; i < N; i++) {
        const size_t nw = (size_t)CO[i] * KK[i]; muon_desc_t *d = &b->h[i];
        d->p = nn_malloc(nw * 4); d->g = nn_malloc(nw * 4); d->mom = nn_malloc(nw * 4); d->X = nn_malloc(nw * 4); d->Y = nn_malloc(nw * 4);
        d->A = nn_malloc((size_t)CO[i] * CO[i] * 4); d->B = nn_malloc((size_t)CO[i] * CO[i] * 4); d->Co = CO[i]; d->K = KK[i];
        float *w = malloc(nw * 4); for (size_t j = 0; j < nw; j++) w[j] = frand() * 0.2f * (1.f + i); nn_h2d(d->p, w, nw * 4); free(w);
        nn_zero(d->mom, nw * 4);
        if (CO[i] > b->maxco) b->maxco = CO[i];
        if (KK[i] > b->maxk) b->maxk = KK[i];
        off[i] = t; t += CO[i];
    }
    off[N] = t; b->tot = t;
    b->dd = nn_malloc(sizeof b->h); nn_h2d(b->dd, b->h, sizeof b->h);
    b->part = nn_malloc((size_t)32 * N * 16 * sizeof(double)); b->rows = nn_malloc((size_t)2 * t * 4); nn_zero(b->rows, (size_t)2 * t * 4);
    b->off = nn_malloc((N + 1) * sizeof(int)); nn_h2d(b->off, off, (N + 1) * sizeof(int));
}
static void batch_free(batch *b) {
    for (int i = 0; i < N; i++) { muon_desc_t *d = &b->h[i]; nn_free(d->p); nn_free((void *)d->g); nn_free(d->mom); nn_free(d->X); nn_free(d->Y); nn_free(d->A); nn_free(d->B); }
    nn_free(b->dd); nn_free(b->part); nn_free(b->rows); nn_free(b->off);
}
static void set_g(batch *a, batch *b, unsigned seed) {
    rs = seed;
    for (int i = 0; i < N; i++) {
        const size_t nw = (size_t)CO[i] * KK[i]; float *g = malloc(nw * 4); for (size_t j = 0; j < nw; j++) g[j] = frand() * (1.f + (j % 7) * 0.3f) * (1.f + (j / KK[i]) % 3);
        nn_h2d((float *)a->h[i].g, g, nw * 4); if (b) nn_h2d((float *)b->h[i].g, g, nw * 4); free(g);
    }
}
static float *get_p(const batch *b, int i) { const size_t nw = (size_t)CO[i] * KK[i]; float *h = malloc(nw * 4); nn_d2h(h, b->h[i].p, nw * 4); return h; }
static void step(batch *b, int ext, int nm, float lr, float wd) {
    if (!ext) nn_muon_batch(b->dd, N, b->maxco, b->maxk, lr, 0.9f, wd);
    else nn_muon_ext_batch(b->dd, N, b->maxco, b->maxk, nullptr, nullptr, 3, 0, 0, nullptr, nullptr, 0, b->part, lr, 0.9f, wd, 0.f, 0, nm ? b->rows : nullptr, b->off, b->tot, 0.9f, nullptr, 0.f);
}

/* stats on A, off on B: identical weights; A's sums equal the host's */
static void test_stats(int ext, int nm) {
    batch a, b; batch_new(&a, 5); batch_new(&b, 5);
    double *st = nn_malloc((size_t)2 * (N + 3) * sizeof(double));
    const float lr = 0.02f, wd = 0.1f; double worst = 0; int same = 1;
    for (int it = 0; it < 3; it++) {
        set_g(&a, &b, 100 + it);
        float *p0[N]; for (int i = 0; i < N; i++) p0[i] = get_p(&a, i);
        nn_zero(st, (size_t)2 * (N + 3) * sizeof(double)); nn_muon_stats(st, 3);   /* base 3: the sums land at [6, 6 + 2 N) */
        step(&a, ext, nm, lr, wd); nn_muon_stats(nullptr, 0); step(&b, ext, nm, lr, wd);
        double h[2 * (N + 3)]; nn_d2h(h, st, sizeof h);
        CHECK(h[0] == 0 && h[5] == 0, "stats outside base");
        for (int i = 0; i < N; i++) {
            const size_t nw = (size_t)CO[i] * KK[i]; float *pa = get_p(&a, i), *pb = get_p(&b, i);
            same &= !memcmp(pa, pb, nw * 4);
            double dd = 0, ww = 0; for (size_t j = 0; j < nw; j++) { const double d = (double)pa[j] - p0[i][j]; dd += d * d; ww += (double)p0[i][j] * p0[i][j]; }
            const double e1 = fabs(h[2 * (3 + i)] - dd) / dd, e2 = fabs(h[2 * (3 + i) + 1] - ww) / ww;
            if (e1 > worst) worst = e1;
            if (e2 > worst) worst = e2;
            free(pa); free(pb); free(p0[i]);
        }
    }
    printf("stats %s%s: weights %s, worst rel err of the sums %.2e\n", ext ? "ext" : "plain", nm ? "+normuon" : "", same ? "identical" : "DIFFER", worst);
    CHECK(same, "stats changed the weights");
    CHECK(worst < 1e-3, "stats sums off: %.3e", worst);
    nn_free(st); batch_free(&a); batch_free(&b);
}

static double mscale(int Co, int K) { return sqrt(fmax(1.0, (double)Co / K)); }
/* lars / dog on batch A against the reference; B gives the direction */
static void test_adapt(int nm, int lars, float tgt, int dog) {
    batch a, b; batch_new(&a, 9); batch_new(&b, 9);
    const float lr0 = 0.02f, wd = 0.1f, sched[5] = {1.f, 1.f, 0.7f, 0.4f, 0.1f}, lb = 0.9f, lc = 2.f, dmax = 3.f;
    float *lrm = nn_malloc(N * 4), *ema = nn_malloc(N * 4), *disp[N]; double *dst = nn_malloc(4 * sizeof(double)), *part = nn_malloc((size_t)64 * N * sizeof(double));
    nn_zero(ema, N * 4); nn_zero(dst, 4 * sizeof(double));
    for (int i = 0; i < N; i++) { const size_t nw = (size_t)CO[i] * KK[i]; disp[i] = nn_malloc(nw * 4); nn_zero(disp[i], nw * 4); }
    void *ddisp = nn_malloc(sizeof disp); nn_h2d(ddisp, disp, sizeof disp);
    const nn_madapt ad = {.lrm = lrm, .ema = ema, .dog = dst, .disp = (float *const *)ddisp, .part = part, .lr_peak = lr0, .lars_tgt = tgt, .lars_b = lb, .lars_c = lc,
                          .dog_eta0 = lr0, .dog_max = dmax * lr0, .on_lars = lars, .on_dog = dog};
    double *P[N], *Dd[N], rema[N] = {0}, rdog[4] = {0}, worst = 0, worst_s = 0; int clamped = 0;
    for (int i = 0; i < N; i++) { float *p = get_p(&a, i); const size_t nw = (size_t)CO[i] * KK[i]; P[i] = malloc(nw * sizeof(double)); Dd[i] = calloc(nw, sizeof(double)); for (size_t j = 0; j < nw; j++) P[i][j] = p[j]; free(p); }
    for (int it = 0; it < 5; it++) {
        set_g(&a, &b, 300 + it); const float lr = lr0 * sched[it];
        for (int i = 0; i < N; i++) nn_zero(b.h[i].p, (size_t)CO[i] * KK[i] * 4);
        nn_muon_ext_adapt(&ad); step(&a, 1, nm, lr, wd); nn_muon_ext_adapt(nullptr);
        step(&b, 1, nm, 1.f, 0.f);
        double *O[N], so[N], sw[N], lsum = 0, m[N], v2 = 0;
        for (int i = 0; i < N; i++) {
            const size_t nw = (size_t)CO[i] * KK[i]; float *pb = get_p(&b, i); O[i] = malloc(nw * sizeof(double)); so[i] = sw[i] = 0;
            for (size_t j = 0; j < nw; j++) { O[i][j] = -(double)pb[j] / (float)mscale(CO[i], KK[i]); so[i] += O[i][j] * O[i][j]; sw[i] += P[i][j] * P[i][j]; }
            free(pb);
            const double rho = mscale(CO[i], KK[i]) * sqrt(so[i]) / sqrt(sw[i]);
            if (lars) { rema[i] = it ? lb * rema[i] + (1 - lb) * rho : rho; lsum += log(rema[i]); }
        }
        const double t = tgt > 0 ? tgt / lr0 : exp(lsum / N);
        for (int i = 0; i < N; i++) { m[i] = lars ? fmin(fmax(t / rema[i], 1 / (double)lc), lc) : 1; if (lars && (m[i] == lc || m[i] == 1 / (double)lc)) clamped++;
            v2 += m[i] * m[i] * mscale(CO[i], KK[i]) * mscale(CO[i], KK[i]) * so[i]; }
        double f = 1;
        if (dog) { const double sc = sched[it], gi = sc * sc * v2;
            if (rdog[1] <= 0) rdog[0] = fmax(rdog[0], lr0 * sqrt(gi));
            rdog[1] += gi; rdog[0] = fmax(rdog[0], rdog[3]); rdog[2] = fmin(rdog[0] / sqrt(rdog[1]), dmax * lr0); f = rdog[2] / lr0; }
        double dd = 0, sn[N];
        for (int i = 0; i < N; i++) {
            const size_t nw = (size_t)CO[i] * KK[i]; const double l = lr * m[i] * f, s = mscale(CO[i], KK[i]); sn[i] = 0;
            for (size_t j = 0; j < nw; j++) { const double u = l * s * O[i][j], np = P[i][j] * (1 - l * wd) - u; sn[i] += (np - P[i][j]) * (np - P[i][j]); P[i][j] = np; Dd[i][j] -= u; dd += Dd[i][j] * Dd[i][j]; }
            free(O[i]);
        }
        rdog[3] = sqrt(dd);
        for (int i = 0; i < N; i++) {   /* weights: relative to the step's size (so the check is not diluted by |W|) */
            const size_t nw = (size_t)CO[i] * KK[i]; float *pa = get_p(&a, i); double e = 0, ref = 0;
            for (size_t j = 0; j < nw; j++) { const double d = pa[j] - P[i][j]; e += d * d; } ref = sn[i];
            const double r = sqrt(e / ref); if (r > worst) worst = r;
            for (size_t j = 0; j < nw; j++) P[i][j] = pa[j];   /* continue from the GPU's weights (float rounding does not accumulate) */
            free(pa);
        }
        float he[N]; double hd[4]; nn_d2h(he, ema, sizeof he); nn_d2h(hd, dst, sizeof hd);
        for (int i = 0; i < N && lars; i++) { const double r = fabs(he[i] - rema[i]) / rema[i]; if (r > worst_s) worst_s = r; }
        for (int k = 0; k < 4 && dog; k++) { const double r = fabs(hd[k] - rdog[k]) / fmax(fabs(rdog[k]), 1e-30); if (r > worst_s) worst_s = r; }
        if (dog) { rdog[3] = hd[3]; }   /* (the GPU's displacement continues from its own float buffer) */
        if (it == 4) printf("  step 5: m [%.3f .. %.3f], eta %.4g (lr0 %.3g), rbar %.4g, G %.4g, dist %.4g\n", lars ? fmin(fmin(m[0], m[1]), fmin(m[2], fmin(m[3], m[4]))) : 1.,
                            lars ? fmax(fmax(m[0], m[1]), fmax(m[2], fmax(m[3], m[4]))) : 1., dog ? hd[2] : lr0, lr0, hd[0], hd[1], hd[3]);
    }
    printf("adapt nm %d lars %d (target %g) dog %d: weights worst rel %.2e, state worst rel %.2e%s\n", nm, lars, tgt, dog, worst, worst_s, clamped ? " (clamp hit)" : "");
    CHECK(worst < 3e-4, "adaptive weights off: %.3e", worst);
    CHECK(worst_s < 1e-4, "adaptive state off: %.3e", worst_s);
    if (tgt > 0) CHECK(clamped, "the absolute target should hit the clamp");
    for (int i = 0; i < N; i++) { free(P[i]); free(Dd[i]); nn_free(disp[i]); }
    nn_free(lrm); nn_free(ema); nn_free(dst); nn_free(part); nn_free(ddisp); batch_free(&a); batch_free(&b);
}

int main(void) {
    nn_init(0);
    test_stats(0, 0);
    test_stats(1, 0);
    test_stats(1, 1);
    test_adapt(0, 1, 0.f, 0);
    test_adapt(0, 1, 0.05f, 0);
    test_adapt(0, 0, 0.f, 1);
    test_adapt(1, 1, 0.f, 1);
    if (fails) { printf("test_admuon: %d FAILED\n", fails); return 1; }
    printf("test_admuon: OK\n");
    return 0;
}
