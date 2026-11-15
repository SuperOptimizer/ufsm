/* --muon-cwd 1, cautious weight decay on the Muon matrices (DESIGN.md 2026-11-15 "Cautious weight decay"): from the same weights,
   momentum and gradient, three twins step once: Z at wd 0 (its weight change is the step u = lr s O, recovered as w0 - w), C at wd with
   cwd on, P at wd with cwd off. C equals the host's u w0 >= 0 ? w0 (1 - lr wd) - u : w0 - u (elements whose recovered u is within
   rounding of 0 skipped), P the plain w0 (1 - lr wd) - u; and cwd on at wd 0 is bit for bit the plain step. Plain batch, ext, ext +
   NorMuon. */
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

static void test_cwd(int ext, int nm) {
    batch z, c, p, q; batch_new(&z, 11); batch_new(&c, 11); batch_new(&p, 11); batch_new(&q, 11);
    const float lr = 0.02f, wd = 0.5f; double worst_c = 0, worst_p = 0; size_t kept = 0, tot = 0, skip = 0; int same0 = 1;
    set_g(&z, &c, 400); set_g(&p, &q, 400);
    float *w0[N]; for (int i = 0; i < N; i++) w0[i] = get_p(&z, i);
    nn_muon_cwd(0); step(&z, ext, nm, lr, 0.f); step(&p, ext, nm, lr, wd);
    nn_muon_cwd(1); step(&c, ext, nm, lr, wd); step(&q, ext, nm, lr, 0.f); nn_muon_cwd(0);
    for (int i = 0; i < N; i++) {
        const size_t nw = (size_t)CO[i] * KK[i]; float *wz = get_p(&z, i), *wc = get_p(&c, i), *wp = get_p(&p, i), *wq = get_p(&q, i);
        same0 &= !memcmp(wz, wq, nw * 4);
        for (size_t j = 0; j < nw; j++) {
            const float w = w0[i][j], u = w - wz[j], ulp = fabsf(w) * 1.2e-7f + 1e-30f;
            const double ep = fabs((double)wp[j] - (w * (1.f - lr * wd) - u)) / (fabs(w) + 1e-6);
            if (ep > worst_p) worst_p = ep;
            tot++;
            if (fabsf(u) < 8.f * ulp) { skip++; continue; }
            const int dec = u * w >= 0.f; kept += dec;
            const double ec = fabs((double)wc[j] - (dec ? w * (1.f - lr * wd) - u : w - u)) / (fabs(w) + 1e-6);
            if (ec > worst_c) worst_c = ec;
        }
        free(wz); free(wc); free(wp); free(wq); free(w0[i]);
    }
    printf("cwd %s%s: decayed %.1f %% of %zu (%zu near-zero steps skipped), cwd vs host worst %.2e, plain vs host worst %.2e, cwd at wd 0 %s\n",
           ext ? "ext" : "plain", nm ? "+normuon" : "", 100.0 * kept / (tot - skip), tot, skip, worst_c, worst_p, same0 ? "identical" : "DIFFERS");
    CHECK(worst_c < 1e-5, "cwd step off the host reference: %.3e", worst_c);
    CHECK(worst_p < 1e-5, "plain step off the host reference: %.3e", worst_p);
    CHECK(same0, "cwd at wd 0 differs from the plain step");
    CHECK(kept > (tot - skip) / 5 && kept < (tot - skip) * 4 / 5, "the mask should split the elements");
    batch_free(&z); batch_free(&c); batch_free(&p); batch_free(&q);
}

int main(void) {
    nn_init(0);
    test_cwd(0, 0);
    test_cwd(1, 0);
    test_cwd(1, 1);
    if (fails) { printf("test_muoncwd: %d FAILED\n", fails); return 1; }
    printf("test_muoncwd: OK\n");
    return 0;
}
