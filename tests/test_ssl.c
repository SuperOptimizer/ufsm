/* train --ssl kernels (src/ssl.h, src/nn_ssl.cu) against host references:
   (1) nn_ssl_mask on 16- and 32-bit CT channels == the host's block decision (every voxel), masked share near the ratio;
   (2) nn_ssl_targets == the host's centred, unit-normalised int8 targets (byte for byte), zero outside the masked blocks;
   (3) nn_ssl_idx: distinct indices for coprime strides (N = S);
   (4) the whole --ssl loss on the side head's outputs y [C][S] (nn_distill_loss at the targeted cells + the variance hinge on the
       unit rows of N sampled cells, dn_gather / dn_varreg / dn_scatter_grad) against central differences of a double host loss. */
#include "ssl.h"
#include "dino.h"
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static double ru(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0 / 9007199254740992.0); }
static double rn(void) { return sqrt(-2 * log(ru() + 1e-12)) * cos(2 * M_PI * ru()); }
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); fails++; } } while (0)

static void test_mask(int es) {
    const int Pz = 32, P = 48, bv = 16; const float ratio = 0.4f; const uint64_t seed = 12345;
    const size_t n = (size_t)Pz * P * P;
    uint8_t *h = malloc(n * es), *g = malloc(n * es);
    for (size_t i = 0; i < n * es; i++) h[i] = (uint8_t)(1 + (i * 37) % 251);   /* no zero bytes */
    void *d = nn_malloc(n * es); nn_h2d(d, h, n * es);
    nn_ssl_mask(d, es, Pz, P, bv, ratio, seed); nn_sync(); nn_d2h(g, d, n * es);
    size_t bad = 0, nm = 0;
    for (int z = 0; z < Pz; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
        const size_t v = ((size_t)z * P + y) * P + x; const int m = ssl_blk_masked(seed, z / bv, y / bv, x / bv, ratio); nm += m;
        for (int b = 0; b < es; b++) bad += g[v * es + b] != (m ? 0 : h[v * es + b]);
    }
    CHECK(!bad, "mask es %d: %zu bytes differ from the host", es, bad);
    printf("mask es %d: %zu of %zu voxels masked (ratio %.2f), %zu bytes differ\n", es, nm, n, ratio, bad);
    /* the block share over many blocks */
    size_t mb = 0, nb = 0; for (int z = 0; z < 64; z++) for (int y = 0; y < 64; y++) for (int x = 0; x < 64; x++) { mb += ssl_blk_masked(777, z, y, x, ratio); nb++; }
    CHECK(fabs((double)mb / nb - ratio) < 0.01, "mask share %.4f vs %.2f", (double)mb / nb, ratio);
    nn_free(d); free(h); free(g);
}

static void host_targets(const float *y, int C, int cd, int ch, int cw, int cv, int bv, float ratio, uint64_t seed, int8_t *t) {
    const size_t S = (size_t)cd * ch * cw; double *mu = calloc(C, sizeof *mu);
    for (int c = 0; c < C; c++) for (size_t s = 0; s < S; s++) mu[c] += y[(size_t)c * S + s];
    for (size_t s = 0; s < S; s++) {
        const int x = (int)(s % cw), yy = (int)((s / cw) % ch), z = (int)(s / ((size_t)cw * ch));
        const int on = ssl_blk_masked(seed, z * cv / bv, yy * cv / bv, x * cv / bv, ratio);
        double n2 = 0; for (int c = 0; c < C; c++) { const double v = y[(size_t)c * S + s] - mu[c] / (double)S; n2 += v * v; }
        const double n = sqrt(n2);
        for (int c = 0; c < C; c++) t[(size_t)c * S + s] = on && n > 1e-6 ? (int8_t)lrint(fmin(127.0, fmax(-127.0, 127.0 * (y[(size_t)c * S + s] - mu[c] / (double)S) / n))) : 0;
    }
    free(mu);
}
static void test_targets(void) {
    const int C = 16, cd = 4, ch = 6, cw = 10, cv = 8, bv = 16; const float ratio = 0.5f; const uint64_t seed = 99;
    const size_t S = (size_t)cd * ch * cw, n = (size_t)C * S;
    float *y = malloc(n * 4); for (size_t i = 0; i < n; i++) y[i] = (float)(rn() + 0.7 * (i / S));   /* channel offsets: the centring matters */
    int8_t *ht = malloc(n), *gt = malloc(n);
    host_targets(y, C, cd, ch, cw, cv, bv, ratio, seed, ht);
    float *dy = nn_malloc(n * 4); int8_t *dt = nn_malloc(n); double *mu = nn_malloc(C * sizeof(double));
    nn_h2d(dy, y, n * 4); nn_ssl_targets(dy, C, cd, ch, cw, cv, bv, ratio, seed, mu, dt); nn_sync(); nn_d2h(gt, dt, n);
    size_t bad = 0, on = 0; for (size_t i = 0; i < n; i++) bad += ht[i] != gt[i];
    for (size_t s = 0; s < S; s++) on += ht[s] != 0 || ht[S + s] != 0;
    CHECK(!bad, "targets: %zu of %zu bytes differ", bad, n);
    CHECK(on > 0 && on < S, "targets: %zu of %zu cells targeted", on, S);
    printf("targets: %zu of %zu cells targeted, %zu bytes differ\n", on, S, bad);
    nn_free(dy); nn_free(dt); nn_free(mu); free(y); free(ht); free(gt);
}
static void test_idx(void) {
    const size_t sizes[3] = {4096, 3 * 5 * 7 * 64, 1000};
    for (int k = 0; k < 3; k++) {
        const size_t S = sizes[k], st = ssl_stride(S, 1234567 + k); int *d = nn_malloc(S * sizeof(int)), *h = malloc(S * sizeof(int)); uint8_t *seen = calloc(S, 1);
        nn_ssl_idx(d, (int)S, S, 17, st); nn_sync(); nn_d2h(h, d, S * sizeof(int));
        size_t dup = 0; for (size_t i = 0; i < S; i++) { if (h[i] < 0 || (size_t)h[i] >= S || seen[h[i]]) dup++; else seen[h[i]] = 1; }
        CHECK(!dup, "idx S %zu stride %zu: %zu out of range or repeated", S, st, dup);
        nn_free(d); free(h); free(seen);
    }
    printf("idx: distinct over 3 sizes\n");
}

/* the --ssl loss in double: w / n_t sum_targeted (1 - cos(y_s, t_s)) + lambda / C sum_c relu(gamma - std_c(u)), u_k = y_{idx_k} / |y| */
static double host_loss(const double *y, const int8_t *t, int C, size_t S, double w, const int *idx, int N, double gamma, double lambda) {
    double L = 0; int nt = 0;
    for (size_t s = 0; s < S; s++) {
        double yy = 0, tt = 0, yt = 0; for (int c = 0; c < C; c++) { const double a = y[c * S + s], q = t[c * S + s]; yy += a * a; tt += q * q; yt += a * q; }
        if (tt > 0) { L += 1 - yt / ((sqrt(yy) + 1e-6) * sqrt(tt)); nt++; }
    }
    L = nt ? w * L / nt : 0;
    double *u = malloc((size_t)N * C * sizeof *u);
    for (int k = 0; k < N; k++) { double n2 = 0; for (int c = 0; c < C; c++) n2 += y[c * S + idx[k]] * y[c * S + idx[k]]; const double n = sqrt(n2); for (int c = 0; c < C; c++) u[k * C + c] = y[c * S + idx[k]] / n; }
    for (int c = 0; c < C; c++) {
        double m = 0, m2 = 0; for (int k = 0; k < N; k++) { m += u[k * C + c]; m2 += u[k * C + c] * u[k * C + c]; }
        m /= N; const double sd = sqrt(fmax(m2 / N - m * m, 0) + 1e-6);
        if (sd < gamma) L += (gamma - sd) * lambda / C;
    }
    free(u); return L;
}
static void test_loss(void) {
    const int C = 16, cd = 4, ch = 4, cw = 6, cv = 8, bv = 16, N = 40; const float ratio = 0.5f, w = 0.5f, lambda = 1.f; const uint64_t seed = 4242;
    const size_t S = (size_t)cd * ch * cw, n = (size_t)C * S;
    const float gamma = 0.6f;   /* above the rows' spread on purpose: the hinge is active in every channel */
    float *y = malloc(n * 4); double *yd = malloc(n * 8);
    for (size_t i = 0; i < n; i++) { y[i] = (float)(rn() * 0.3 + (i % 3 == 0 ? 0.8 : 0.0)); yd[i] = y[i]; }   /* a shared direction: rows partly collapsed */
    float *tf = malloc(n * 4); for (size_t i = 0; i < n; i++) tf[i] = (float)rn();
    int8_t *t = malloc(n); host_targets(tf, C, cd, ch, cw, cv, bv, ratio, seed, t);
    float *dy = nn_malloc(n * 4), *dg = nn_malloc(n * 4), *e = nn_malloc((size_t)N * C * 4), *ge = nn_malloc((size_t)N * C * 4), *nrm = nn_malloc(N * 4), *vo = nn_malloc(8 * 4);
    int8_t *dt = nn_malloc(n); double *ds = nn_malloc(2 * sizeof(double)); int *didx = nn_malloc(N * sizeof(int));
    nn_h2d(dy, y, n * 4); nn_h2d(dt, t, n);
    const size_t st = ssl_stride(S, 77); nn_ssl_idx(didx, N, S, 5, st);
    nn_distill_loss(dy, dt, 1, C, S, w, dg, ds);
    nn_zero(vo, 32); dn_gather(dy, 0, S, C, didx, N, e, nrm); nn_zero(ge, (size_t)N * C * 4);
    dn_varreg(e, N, C, gamma, lambda, ge, vo); dn_scatter_grad(e, ge, nrm, didx, N, C, S, 1.f, dg);
    float *g = malloc(n * 4); int *idx = malloc(N * sizeof(int)); double hs[2]; float hv;
    nn_sync(); nn_d2h(g, dg, n * 4); nn_d2h(idx, didx, N * sizeof(int)); nn_d2h(hs, ds, 16); nn_d2h(&hv, vo, 4);
    const double L0 = host_loss(yd, t, C, S, w, idx, N, gamma, lambda), Lg = (hs[1] > 0 ? w * hs[0] / hs[1] : 0) + hv;
    CHECK(fabs(L0 - Lg) < 1e-4 * (1 + fabs(L0)), "loss: device %.7f vs host %.7f", Lg, L0);
    double worst = 0, gmax = 0;
    for (size_t i = 0; i < n; i++) gmax = fmax(gmax, fabs(g[i]));
    for (size_t i = 0; i < n; i++) {
        const double h = 1e-5, sv = yd[i];
        yd[i] = sv + h; const double Lp = host_loss(yd, t, C, S, w, idx, N, gamma, lambda);
        yd[i] = sv - h; const double Lm = host_loss(yd, t, C, S, w, idx, N, gamma, lambda); yd[i] = sv;
        worst = fmax(worst, fabs((Lp - Lm) / (2 * h) - g[i]));
    }
    CHECK(worst < 2e-3 * gmax, "loss gradient: worst |device - central difference| %.3e of max %.3e", worst, gmax);
    printf("loss %.6f (host %.6f; %0.f targeted cells, hinge %.4f), gradient worst error %.2e of max %.2e\n", Lg, L0, hs[1], hv, worst, gmax);
    nn_free(dy); nn_free(dg); nn_free(e); nn_free(ge); nn_free(nrm); nn_free(vo); nn_free(dt); nn_free(ds); nn_free(didx);
    free(y); free(yd); free(tf); free(t); free(g); free(idx);
}

int main(void) {
    if (nn_init(0)) { printf("SKIP: no GPU\n"); return 0; }
    test_mask(2); test_mask(4); test_targets(); test_idx(); test_loss();
    if (fails) { printf("%d FAILED\n", fails); return 1; }
    printf("PASS\n"); return 0;
}
