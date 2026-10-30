/* wtrunk (src/wtrunk.h): the in-slab input kernel and the output transform vs CPU references; with the wind96 checkpoint
   (UFSM_WT_CKPT or ~/ufsm-data/wcomb/d/paris4/ckpt/wind96-wk1-best.ckpt): heads equal with and without the feature channels,
   a batch of 2 equals two single forwards, finite outputs */
#include "wtrunk.h"
#include "nn.h"
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static float h2f(uint16_t h) {   /* IEEE half */
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023; float v;
    if (e == 0) v = ldexpf((float)m, -24); else if (e == 31) v = m ? NAN : INFINITY; else v = ldexpf((float)(m | 1024), (int)e - 25);
    return s ? -v : v;
}
static float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t f2h(float f) {   /* round to nearest half (normal range; tests only) */
    if (f == 0) return 0;
    const int s = f < 0; f = fabsf(f); int e; float m = frexpf(f, &e);   /* f = m 2^e, m in [0.5, 1) */
    int E = e + 14; if (E <= 0) { int q = (int)lrintf(ldexpf(f, 24)); return (uint16_t)(s << 15 | q); }
    int q = (int)lrintf((m * 2 - 1) * 1024); if (q == 1024) { q = 0; E++; }
    return (uint16_t)(s << 15 | E << 10 | q);
}
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)

static void frames(wt_frame *fr, int n, int T) {
    for (int s = 0; s < n; s++) {   /* tilted rays, axis along a slightly tilted z */
        wt_frame *f = fr + s; const double t = 0.4 + 0.9 * s, tl = 0.1;
        double d[3] = {cos(t) * cos(tl), sin(t) * cos(tl), sin(tl)};
        double a[3] = {-sin(t), cos(t), 0}, b[3] = {d[1] * a[2] - d[2] * a[1], d[2] * a[0] - d[0] * a[2], d[0] * a[1] - d[1] * a[0]};
        for (int e = 0; e < 3; e++) { f->d[e] = d[e]; f->a[e] = a[e]; f->b[e] = b[e]; f->o[e] = 300 * d[e] - (T - 1) / 2.0 * (a[e] + b[e]) + (e == 2 ? 500 : 0); }
        f->c[0] = 3; f->c[1] = -5; f->c[2] = 480; f->u[0] = 0.05; f->u[1] = 0; f->u[2] = 0.998;
    }
}
int main(void) {
    if (nn_init(0)) return 1;
    const int T = 64, L = 128, n = 2; const size_t V = (size_t)T * T * L;
    uint8_t *h = malloc(n * V);
    for (size_t i = 0; i < n * V; i++) h[i] = (uint8_t)(i % 7 == 0 ? 0 : (i * 2654435761u >> 13) & 255);
    wt_frame fr[2]; frames(fr, n, T);
    uint8_t *im = nn_malloc(n * V); nn_h2d(im, h, n * V);
    {   /* input kernel */
        float prm[32]; for (int s = 0; s < n; s++) { float *p = prm + 16 * s; const wt_frame *f = fr + s;
            double un = sqrt(f->u[0] * f->u[0] + f->u[1] * f->u[1] + f->u[2] * f->u[2]);
            for (int e = 0; e < 3; e++) { p[e] = (float)(f->o[e] - f->c[e]); p[3 + e] = (float)f->a[e]; p[6 + e] = (float)f->b[e]; p[9 + e] = (float)f->d[e]; p[12 + e] = (float)(f->u[e] / un); } p[15] = 0; }
        float *dp = nn_malloc(sizeof prm); nn_h2d(dp, prm, sizeof prm);
        unsigned long long *st = nn_malloc(4 * 8);
        uint16_t *dx = nn_malloc(n * 4 * V * 2), *hx = malloc(n * 4 * V * 2);
        wt_input(im, dp, n, T, L, st, dx, 1); nn_d2h(hx, dx, n * 4 * V * 2);
        double emax[4] = {0, 0, 0, 0};
        for (int s = 0; s < n; s++) {
            double sm = 0, sq = 0; for (size_t v = 0; v < V; v++) { const double c = h[s * V + v]; sm += c; sq += c * c; }
            const double m = sm / V, var = sq / V - m * m, isd = 1.0 / (sqrt(var) + 1e-3);
            const wt_frame *f = fr + s; const double un = sqrt(f->u[0] * f->u[0] + f->u[1] * f->u[1] + f->u[2] * f->u[2]);
            for (int j = 0; j < T; j++) for (int i = 0; i < T; i++) for (int k = 0; k < L; k++) {
                const size_t q = ((size_t)j * T + i) * L + k;
                double r[3], ru = 0; for (int e = 0; e < 3; e++) { r[e] = f->o[e] - f->c[e] + i * f->a[e] + j * f->b[e] + k * f->d[e]; ru += r[e] * f->u[e] / un; }
                double rn = 0; for (int e = 0; e < 3; e++) { r[e] -= ru * f->u[e] / un; rn += r[e] * r[e]; } rn = sqrt(rn);
                const double ref[4] = {(h[s * V + ((size_t)i * T + j) * L + k] - m) * isd, 0, (r[0] * f->a[0] + r[1] * f->a[1] + r[2] * f->a[2]) / rn, (r[0] * f->d[0] + r[1] * f->d[1] + r[2] * f->d[2]) / rn};
                for (int c = 0; c < 4; c++) { const double e = fabs(h2f(hx[(size_t)s * 4 * V + c * V + q]) - ref[c]) / (1 + fabs(ref[c])); if (e > emax[c]) emax[c] = e; }
            }
        }
        printf("input: max rel err ct %.2e zero %.2e ra %.2e rd %.2e\n", emax[0], emax[1], emax[2], emax[3]);
        for (int c = 0; c < 4; c++) CHECK(emax[c] < 2e-3, "input channel %d err %g", c, emax[c]);
        nn_free(dp); nn_free(st); nn_free(dx); free(hx);
    }
    {   /* output transform (fp16 logits) */
        const int C = 5, co = 3; uint16_t *hl = malloc(n * C * V * 2), *ho = malloc(n * C * V * 2);
        for (size_t i = 0; i < n * C * V; i++) hl[i] = f2h((float)((int)(i * 2246822519u >> 20 & 4095) - 2048) / 400.f);
        uint16_t *dl = nn_malloc(n * C * V * 2), *dout = nn_malloc(n * C * V * 2); nn_h2d(dl, hl, n * C * V * 2);
        wt_output(dl, 1, n, C, co, T, L, dout); nn_d2h(ho, dout, n * C * V * 2);
        double em = 0;
        for (int s = 0; s < n; s++) for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) for (int k = 0; k < L; k++) for (int c = 0; c < C; c++) {
            const float a = h2f(hl[(size_t)(s * C + c) * V + ((size_t)j * T + i) * L + k]), ref = c < co ? 1.f / (1.f + expf(-a)) : a;
            const float o = bf2f(ho[(((size_t)s * T + i) * T + j) * L * C + (size_t)k * C + c]);
            const double e = fabs(o - ref) / (1 + fabs(ref)); if (e > em) em = e;
        }
        printf("output: max rel err %.2e\n", em); CHECK(em < 8e-3, "output err %g", em);
        nn_free(dl); nn_free(dout); free(hl); free(ho);
    }
    char ck[1024]; const char *e = getenv("UFSM_WT_CKPT");
    if (e) snprintf(ck, sizeof ck, "%s", e); else snprintf(ck, sizeof ck, "%s/ufsm-data/wcomb/d/paris4/ckpt/wind96-wk1-best.ckpt", getenv("HOME") ? getenv("HOME") : ".");
    if (access(ck, R_OK)) { printf("no checkpoint %s: network checks skipped\n", ck); }
    else {
        wtrunk *w1 = wt_open(ck, T, L, 2, 1);
        CHECK(w1, "wt_open feat 1");
        if (w1) {
            const int C = wt_channels(w1), co = wt_heads(w1);
            uint16_t *o2 = nn_malloc(n * V * C * 2), *ha = malloc(n * V * C * 2), *hb = malloc(n * V * C * 2);
            CHECK(!wt_forward(w1, im, fr, n, 9.6, o2), "forward batch 2"); nn_d2h(ha, o2, n * V * C * 2);
            wt_close(w1);
            wtrunk *w0 = wt_open(ck, T, L, 1, 1);
            CHECK(!wt_forward(w0, im, fr, n, 9.6, o2), "forward batch 1"); nn_d2h(hb, o2, n * V * C * 2); wt_close(w0);
            double db = 0, fs = 0; long bad = 0;
            for (size_t i = 0; i < n * V * C; i++) { const float a = bf2f(ha[i]), b = bf2f(hb[i]); if (!isfinite(a)) bad++; const double d = fabs(a - b); if (d > db) db = d; if ((int)(i % C) >= co) fs += fabs(a); }
            printf("batch 2 vs 1: max |d| %.3g, non-finite %ld, mean |feature| %.3g (C %d)\n", db, bad, fs / (n * V * (C - co)), C);
            for (int s = 0; s < n; s++) { double dh_ = 0, df_ = 0, mh = 0, mf = 0; for (size_t v = 0; v < V; v++) for (int c = 0; c < C; c++) { const size_t q = ((size_t)s * V + v) * C + c; const double d = fabs(bf2f(ha[q]) - bf2f(hb[q])); if (c < co) { mh += d; if (d > dh_) dh_ = d; } else { mf += d; if (d > df_) df_ = d; } }
                printf("  slab %d: heads max %.3g mean %.3g, features max %.3g mean %.3g\n", s, dh_, mh / V / co, df_, mf / V / (C - co)); }
            CHECK(bad == 0 && db < 0.05 && fs > 0, "batch / finite / features");
            {   /* reference: no in-place level-0 slabs (the whole-window inference path), batch 1 and 2 */
                uint16_t *hr = malloc(n * V * C * 2);
                for (int mb = 1; mb <= 2; mb++) {
                    unet_set_infer_slab(0); wtrunk *wr = wt_open(ck, T, L, mb, 1);
                    CHECK(!wt_forward(wr, im, fr, n, 9.6, o2), "forward no slabs"); nn_d2h(hr, o2, n * V * C * 2); wt_close(wr); unet_set_infer_slab(-1);
                    double d1 = 0, d2 = 0; for (size_t i = 0; i < n * V * C; i++) { const double x = fabs(bf2f(hr[i]) - bf2f(hb[i])), y = fabs(bf2f(hr[i]) - bf2f(ha[i])); if (x > d1) d1 = x; if (y > d2) d2 = y; }
                    printf("no slabs batch %d vs slabs batch 1: max |d| %.3g, vs slabs batch 2: %.3g\n", mb, d1, d2);
                }
                free(hr);
            }
            wtrunk *wh = wt_open(ck, T, L, 2, 0);
            CHECK(wh && wt_channels(wh) == co, "heads-only channels");
            uint16_t *hh = malloc(n * V * co * 2);
            CHECK(!wt_forward(wh, im, fr, n, 9.6, o2), "forward heads only"); nn_d2h(hh, o2, n * V * co * 2); wt_close(wh);
            double dh = 0; for (size_t v = 0; v < n * V; v++) for (int c = 0; c < co; c++) { const double d = fabs(bf2f(hh[v * co + c]) - bf2f(ha[v * C + c])); if (d > dh) dh = d; }
            printf("heads with vs without features: max |d| %.3g\n", dh);
            CHECK(dh < 0.02, "heads differ with features: %g", dh);
            nn_free(o2); free(ha); free(hb); free(hh);
        }
    }
    nn_free(im); free(h);
    if (fails) { printf("test_wtrunk: %d FAILED\n", fails); return 1; }
    printf("test_wtrunk: ok\n");
    return 0;
}
