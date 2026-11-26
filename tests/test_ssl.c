/* train --ssl kernels (src/ssl.h, src/nn_ssl.cu) against host references:
   (1) nn_ssl_mask on 16- and 32-bit CT channels == the host's block decision (every voxel), masked share near the ratio;
   (2) nn_ssl_targets == the host's centred, unit-normalised int8 targets (byte for byte), zero outside the masked blocks;
   (3) nn_ssl_idx: distinct indices for coprime strides (N = S);
   (4) the whole --ssl loss on the side head's outputs y [C][S] (nn_distill_loss at the targeted cells + the variance hinge on the
       unit rows of N sampled cells, dn_gather / dn_varreg / dn_scatter_grad) against central differences of a double host loss;
   (5) --ssl-views 2 (2026-11-26): ssl2_pull (the host's continuous pull through a radial record) == the device CT warp (nn_aug_ct on
       ramps z / y / x: trilinear of a linear source is exact, so the output voxel IS its source coordinate) for a symmetric, a warped
       (rotation + affine + zoom + elastic + symmetry) and a teacher record; ssl2_map's teacher coordinate pulls back (through the
       teacher record) onto the student cell's source point; nn_ssl2_targets == the host's trilinear, centred, unit int8 targets. */
#include "ssl.h"
#include "dino.h"
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "test_gpu.h"
#include "scan_augment.h"

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static double ru(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0 / 9007199254740992.0); }
static double rn(void) { return sqrt(-2 * log(ru() + 1e-12)) * cos(2 * M_PI * ru()); }
static int fails = 0;
static float h2f_(uint16_t h) { _Float16 v; memcpy(&v, &h, 2); return (float)v; }
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

/* (5) --ssl-heads: the teacher's gated per-cell targets of every mode (BCE per channel with and without the papyrus gate, the normal's
   |N|_F gate, the phase pair's magnitude gate) and the student's soft-BCE gradient (fp16 logits) vs the host: targets byte for byte (a
   rounding step allowed), the gradient against the host formula and against central differences of the host BCE, the per-head sums */
static float h2f(uint16_t h) { _Float16 v; memcpy(&v, &h, 2); return (float)v; }
static uint16_t f2h(float f) { _Float16 v = (_Float16)f; uint16_t h; memcpy(&h, &v, 2); return h; }
static double sgm(double z) { return 1 / (1 + exp(-z)); }
static double bce(double z, double y) { return fmax(z, 0) - z * y + log1p(exp(-fabs(z))); }
static void test_sslh(float ncf) {
    const int nch = 16, rv = 12, cd = 8, ch = 8, cw = 8, cv = 4, bv = 16; const float ratio = 0.4f; const uint64_t seed = 31;
    ssl_heads hs = {.nh = 5, .nslot = 1 + 2 + 6 + 2 + 1};
    hs.h[0] = (ssl_head){SSLH_BCE, 11, 1, 0, 1, 0.3f};     /* ink, on papyrus */
    hs.h[1] = (ssl_head){SSLH_BCE, 12, 2, 1, 0, 0.3f};     /* rv, no gate */
    hs.h[2] = (ssl_head){SSLH_NORM, 2, 6, 3, 1, ncf};      /* normal */
    hs.h[3] = (ssl_head){SSLH_PHASE, 0, 2, 9, 1, 0.7f};    /* phase */
    hs.h[4] = (ssl_head){SSLH_BCE, 8, 1, 11, 1, 0.3f};     /* dens */
    for (int h = 0; h < hs.nh; h++) hs.scale[h] = 0.1f + 0.2f * h;
    const size_t S = (size_t)cd * ch * cw, half = S / 2;   /* two chunks of half the cells */
    uint16_t *lg = malloc(S * nch * 2); for (size_t i = 0; i < S * nch; i++) lg[i] = f2h((float)(rn() * 3));
    uint8_t *ht = malloc(S * hs.nslot); size_t hcnt[32] = {0};
    for (size_t s = 0; s < S; s++) {
        const size_t c = s / half, i = s % half; const uint16_t *L = lg + c * half * nch;   /* chunk layout [nch][half] */
        const int x = (int)(s % cw), y = (int)((s / cw) % ch), z = (int)(s / ((size_t)cw * ch));
        const int seen = !ssl_blk_masked(seed, z * cv / bv, y * cv / bv, x * cv / bv, ratio);
        const int pap = fmax(sgm(h2f(L[(size_t)rv * half + i])), sgm(h2f(L[(size_t)(rv + 1) * half + i]))) >= 0.5;
        for (int h = 0; h < hs.nh; h++) {
            const ssl_head e = hs.h[h]; double p[6]; int on[6];
            for (int k = 0; k < e.nch; k++) p[k] = sgm(h2f(L[(size_t)(e.ch0 + k) * half + i]));
            const int ok = seen && (!e.pap || pap);
            if (e.mode == SSLH_NORM) { double f = 0; for (int k = 0; k < 3; k++) f += p[k] * p[k]; for (int k = 3; k < 6; k++) f += 2 * (p[k] - 0.5) * (p[k] - 0.5); for (int k = 0; k < 6; k++) on[k] = ok && sqrt(f) >= e.conf; }
            else if (e.mode == SSLH_PHASE) { const double u = 2 * p[0] - 1, v = 2 * p[1] - 1; on[0] = on[1] = ok && sqrt(u * u + v * v) >= e.conf; }
            else for (int k = 0; k < e.nch; k++) on[k] = ok && fabs(p[k] - 0.5) >= e.conf;
            for (int k = 0; k < e.nch; k++) { ht[(size_t)(e.slot0 + k) * S + s] = on[k] ? (uint8_t)lrint(254 * p[k]) : 255; hcnt[e.slot0 + k] += on[k]; }
        }
    }
    void *dl = nn_malloc(S * nch * 2), *dg = nn_malloc(half * nch * 2); uint8_t *dt = nn_malloc(S * hs.nslot), *gt = malloc(S * hs.nslot);
    unsigned long long *dc = nn_malloc(32 * 8), hc[32]; double *ds = nn_malloc(2 * SSLH_MAX * 8), hsum[2 * SSLH_MAX];
    nn_h2d(dl, lg, S * nch * 2); nn_zero(dc, 32 * 8); nn_zero(ds, 2 * SSLH_MAX * 8);
    for (int c = 0; c < 2; c++) nn_sslh_targets((char *)dl + c * half * nch * 2, &hs, rv, half, c * half, cd, ch, cw, cv, bv, ratio, seed, dt, dc);
    nn_sync(); nn_d2h(gt, dt, S * hs.nslot); nn_d2h(hc, dc, 32 * 8);
    size_t bad = 0, cbad = 0, tot = 0; for (size_t j = 0; j < S * hs.nslot; j++) bad += abs((int)gt[j] - (int)ht[j]) > (gt[j] == 255 || ht[j] == 255 ? 0 : 1);
    for (int k = 0; k < hs.nslot; k++) { cbad += hc[k] != hcnt[k]; tot += hcnt[k]; }
    CHECK(!bad && !cbad, "heads targets: %zu bytes differ, %zu slot counts differ", bad, cbad);
    CHECK(tot > 0 && hcnt[3] > 0 && hcnt[3] < S && hcnt[9] > 0 && hcnt[9] < S, "heads targets: degenerate gates (normal %zu, phase %zu of %zu)", hcnt[3], hcnt[9], S);
    double worst = 0, gmax = 0, fdw = 0, Lh[SSLH_MAX] = {0}, Nh[SSLH_MAX] = {0}; uint16_t *gg = malloc(half * nch * 2); int stray = 0;
    for (int c = 0; c < 2; c++) {
        nn_zero(dg, half * nch * 2); nn_sslh_grad((char *)dl + c * half * nch * 2, &hs, half, c * half, S, dt, dg, ds); nn_sync(); nn_d2h(gg, dg, half * nch * 2);
        int used[16] = {0};
        for (int h = 0; h < hs.nh; h++) for (int k = 0; k < hs.h[h].nch; k++) {
            const int chn = hs.h[h].ch0 + k; used[chn] = 1;
            for (size_t i = 0; i < half; i++) {
                const uint8_t q = gt[(size_t)(hs.h[h].slot0 + k) * S + c * half + i]; const double z = h2f(lg[c * half * nch + (size_t)chn * half + i]), y = q / 254.0;
                const double want = q == 255 ? 0 : hs.scale[h] * (sgm(z) - y), got = h2f(gg[(size_t)chn * half + i]);
                worst = fmax(worst, fabs(got - want)); gmax = fmax(gmax, fabs(want));
                if (q != 255) { const double e = 1e-4, fd = hs.scale[h] * (bce(z + e, y) - bce(z - e, y)) / (2 * e); fdw = fmax(fdw, fabs(got - fd)); Lh[h] += bce(z, y); Nh[h] += 1; }
            }
        }
        for (int k = 0; k < nch; k++) if (!used[k]) for (size_t i = 0; i < half; i++) stray |= gg[(size_t)k * half + i] != 0;
    }
    nn_d2h(hsum, ds, 2 * SSLH_MAX * 8);
    double sw = 0; for (int h = 0; h < hs.nh; h++) sw = fmax(sw, fabs(hsum[2 * h] - Lh[h]) / (1 + Lh[h]) + fabs(hsum[2 * h + 1] - Nh[h]));
    CHECK(!stray && worst <= 2e-3 * (gmax > 0 ? gmax : 1) && fdw <= 4e-3 * (gmax > 0 ? gmax : 1) && sw < 1e-3, "heads grad: worst %.3g / fd %.3g of %.3g, sums off %.3g, stray %d", worst, fdw, gmax, sw, stray);
    printf("ssl heads (normal conf %.2f): gated ink %zu, rv %zu+%zu, normal %zu, phase %zu, dens %zu of %zu cells; %zu target bytes differ; gradient worst %.2e (central differences %.2e) of %.2e\n",
           ncf, hcnt[0], hcnt[1], hcnt[2], hcnt[3], hcnt[9], hcnt[11], S, bad, worst, fdw, gmax);
    nn_free(dl); nn_free(dg); nn_free(dt); nn_free(dc); nn_free(ds); free(lg); free(ht); free(gt); free(gg);
}

/* (5) --ssl-views 2 */
static void rec_sym(void *r, int P, int sb) { memset(r, 0, NN_RAD_REC(P)); int32_t *h = r; h[2] = P; h[3] = P; h[5] = sb; }
static void rec_warp(void *r, int P, int sb, double ang, int affine, double sc, double amp) {
    rec_sym(r, P, sb); int32_t *h = r; h[6] = 1 | affine << 1;
    float *tb = (float *)r + 8 + 2 * (size_t)P, *R = tb + 6 * (size_t)P;
    for (int d = 0; d < 3; d++) for (int u = 0; u < P; u++) { tb[d * P + u] = (float)(amp * sin(0.13 * u + d)); tb[(3 + d) * P + u] = (float)(amp * 0.13 * cos(0.13 * u + d)); }
    const double c = cos(ang), s_ = sin(ang), Rm[9] = {c, -s_, 0, s_, c, 0, 0, 0, 1};   /* (a rotation about the first axis pair) */
    const double Am[9] = {1.05, 0.03, 0, -0.02, 0.97, 0.04, 0.01, 0, 1.02};
    for (int k = 0; k < 9; k++) { R[k] = (float)Rm[k]; R[9 + k] = (float)(affine ? Am[k] : (k % 4 == 0)); }
    R[27] = (float)sc;
}
static void test_pull(const char *what, const void *rec, int P) {   /* device warp of the ramps vs ssl2_pull at every voxel */
    const size_t p3 = (size_t)P * P * P; uint8_t *in = malloc(p3); uint16_t *o16 = malloc(p3 * 2); float *got[3];
    scan_ct_rec sr; memset(&sr, 0, sizeof sr); sr.fixed = 1; sr.fmean = 0.f; sr.fisd = 1.f; sr.xfmt = 1;
    void *rd = nn_malloc(NN_RAD_REC(P)), *ct = nn_malloc(p3 * 2); nn_h2d(rd, rec, NN_RAD_REC(P));
    for (int a = 0; a < 3; a++) {
        for (size_t k = 0; k < p3; k++) { const size_t z = k / ((size_t)P * P), y = k / P % P, x = k % P; in[k] = (uint8_t)(a == 0 ? z : a == 1 ? y : x); }
        nn_aug_ct(in, &sr, rd, P, ct, 0); nn_sync(); nn_d2h(o16, ct, p3 * 2);
        got[a] = malloc(p3 * 4); for (size_t k = 0; k < p3; k++) got[a][k] = h2f_(o16[k]);
    }
    size_t n = 0, bad = 0; double worst = 0;
    for (size_t k = 0; k < p3; k++) {
        const double io[3] = {(double)(k / ((size_t)P * P)), (double)(k / P % P), (double)(k % P)}; double V[3];
        if (!ssl2_pull(rec, P, io, V)) continue;
        int edge = 0; for (int d = 0; d < 3; d++) edge |= V[d] < 0.5 || V[d] > P - 1.5; if (edge) continue;   /* (the device's 0 outside vs a real 0 at the face) */
        n++; for (int d = 0; d < 3; d++) { const double e = fabs(got[d][k] - V[d]); if (e > worst) worst = e; bad += e > 0.02 + V[d] * 1e-3; }
    }
    CHECK(n > p3 / 4 && !bad, "ssl2_pull %s: %zu of %zu interior voxels off the device warp (worst %.4f)", what, bad, n, worst);
    printf("ssl2_pull %s: %zu voxels, worst |device - host| %.4f\n", what, n, worst);
    for (int a = 0; a < 3; a++) free(got[a]);
    nn_free(rd); nn_free(ct); free(in); free(o16);
}
static void test_views(void) {
    const int P = 64, cv = 8, G = P / cv; uint8_t *rs = malloc(NN_RAD_REC(P)), *rw = malloc(NN_RAD_REC(P)), *rt = malloc(NN_RAD_REC(P));
    rec_sym(rs, P, 2 | 0 << 2 | 1 << 4 | 1 << 8 | 1 << 10);   /* perm (2, 0, 1), flips z and x */
    rec_warp(rw, P, 1 | 2 << 2 | 0 << 4 | 1 << 9, 0.3, 1, 1.15, 1.5);
    const double sh[3] = {5.5, -3.0, 2.25}; ssl2_teacher_rec(rw, rt, P, sh, 0.85);
    test_pull("symmetry", rs, P); test_pull("warp", rw, P); test_pull("teacher", rt, P);
    /* the map: each student cell's teacher coordinate, pushed back through the teacher record, lands on the student's source point */
    const void *st[2] = {rs, rw}; float *map = malloc((size_t)G * G * G * 3 * 4);
    for (int k = 0; k < 2; k++) {
        const long nm = ssl2_map(st[k], rt, P, cv, map); double worst = 0; long nchk = 0;
        for (int z = 0; z < G; z++) for (int y = 0; y < G; y++) for (int x = 0; x < G; x++) {
            const float *m = map + 3 * (((size_t)z * G + y) * G + x); if (isnan(m[0])) continue;
            const double io[3] = {z * cv + 3.5, y * cv + 3.5, x * cv + 3.5}, it[3] = {m[0] * cv + 3.5, m[1] * cv + 3.5, m[2] * cv + 3.5}; double Vs[3], Vt[3];
            ssl2_pull(st[k], P, io, Vs); ssl2_pull(rt, P, it, Vt); for (int d = 0; d < 3; d++) worst = fmax(worst, fabs(Vs[d] - Vt[d])); nchk++; }
        CHECK(nm > 0 && nm == nchk && worst < 1e-3, "ssl2_map %d: %ld mapped, %ld checked, worst round trip %.2e voxels", k, nm, nchk, worst);
        printf("ssl2_map %s: %ld of %d cells mapped, worst round trip %.2e voxels\n", k ? "warp" : "symmetry", nm, G * G * G, worst);
    }
    /* nn_ssl2_targets vs the host */
    const int C = 12, bv = 16; const float ratio = 0.6f; const uint64_t seed = 7; const size_t S = (size_t)G * G * G, n = (size_t)C * S;
    float *y = malloc(n * 4); for (size_t i = 0; i < n; i++) y[i] = (float)(rn() + 0.5 * (i / S));
    ssl2_map(rw, rt, P, cv, map);
    int8_t *ht = calloc(n, 1), *gt = malloc(n); double *mu = calloc(C, sizeof *mu);
    for (int c = 0; c < C; c++) for (size_t s2 = 0; s2 < S; s2++) mu[c] += y[(size_t)c * S + s2];
    const int nn3[3] = {G, G, G}; size_t on = 0;
    for (size_t s2 = 0; s2 < S; s2++) {
        const int x = (int)(s2 % G), yy = (int)(s2 / G % G), z = (int)(s2 / ((size_t)G * G)); const float *m = map + 3 * s2;
        if (!ssl_blk_masked(seed, z * cv / bv, yy * cv / bv, x * cv / bv, ratio) || isnan(m[0])) continue;
        int b[3]; double f[3]; for (int d = 0; d < 3; d++) { int i = (int)floor(m[d]); if (i > G - 2) i = G - 2; if (i < 0) i = 0; double fr = m[d] - i; fr = fr < 0 ? 0 : fr > 1 ? 1 : fr; b[d] = i; f[d] = fr; }
        double n2 = 0, v[64]; for (int c = 0; c < C; c++) { v[c] = ssl2_tri(y + (size_t)c * S, nn3, b, f) - mu[c] / (double)S; n2 += v[c] * v[c]; }
        const double nv = sqrt(n2); if (!(nv > 1e-6)) continue; on++;
        for (int c = 0; c < C; c++) ht[(size_t)c * S + s2] = (int8_t)lrint(fmin(127.0, fmax(-127.0, 127.0 * v[c] / nv)));
    }
    float *dy = nn_malloc(n * 4), *dm = nn_malloc(S * 3 * 4); int8_t *dt = nn_malloc(n); double *dmu = nn_malloc(C * sizeof(double));
    nn_h2d(dy, y, n * 4); nn_h2d(dm, map, S * 3 * 4); nn_ssl2_targets(dy, C, G, G, G, cv, bv, ratio, seed, dm, dmu, dt); nn_sync(); nn_d2h(gt, dt, n);
    size_t bad = 0; for (size_t i = 0; i < n; i++) bad += ht[i] != gt[i];
    CHECK(!bad && on > 0 && on < S, "ssl2 targets: %zu of %zu bytes differ, %zu of %zu cells targeted", bad, n, on, S);
    printf("ssl2 targets: %zu of %zu cells targeted, %zu bytes differ\n", on, S, bad);
    nn_free(dy); nn_free(dm); nn_free(dt); nn_free(dmu); free(y); free(ht); free(gt); free(mu); free(map); free(rs); free(rw); free(rt);
}

int main(void) {
    if (nn_init(0)) return test_no_gpu("test_ssl", "nn_init(0) failed");
    test_mask(2); test_mask(4); test_targets(); test_idx(); test_loss(); test_sslh(0.8f); test_sslh(1.2f); test_views();
    if (fails) { printf("%d FAILED\n", fails); return 1; }
    printf("PASS\n"); return 0;
}
