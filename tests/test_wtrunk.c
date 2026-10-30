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
        wt_output(dl, 1, n, C, co, T, L, C, dout); nn_d2h(ho, dout, n * C * V * 2);
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
    if (!access(ck, R_OK)) {   /* stage 2: the trunk gradient through heads and features. 16-bit mode (UFSM_WT_HI): vs central differences of a
                                  linear loss L = sum r . out along d = g / |g| (r on the head channels only, then on the features only);
                                  production precision (MX-fp4): the cosine of its gradient with the 16-bit one */
        float *g16[3] = {nullptr, nullptr, nullptr};
        for (int mode = 0; mode < 2; mode++) {
            if (!mode) setenv("UFSM_WT_HI", "1", 1); else unsetenv("UFSM_WT_HI");
            wtrunk *wt = wt_open_train(ck, T, L, 0);
            CHECK(wt, "wt_open_train");
            if (!wt) continue;
            const int C = wt_channels(wt), co = wt_heads(wt); unet *u = wt_unet(wt); const size_t np = unet_nparams(u), nv = V * C;
            uint16_t *o = nn_malloc(nv * 2), *ho = malloc(nv * 2), *hr = malloc(nv * 2), *gr = nn_malloc(nv * 2);
            float *th = malloc(np * 4), *gh = malloc(np * 4), *p = (float *)unet_live_params(u);
            nn_d2h(th, p, np * 4);
            double tn = 0; for (size_t i = 0; i < np; i++) tn += (double)th[i] * th[i]; tn = sqrt(tn);
            for (int part = 0; part < (mode ? 2 : 3); part++) {   /* part 2: features with r = W^T G (the head rows' span) */
                if (part < 2) for (size_t q = 0; q < nv; q++) { const int c = (int)(q % C); float r = ((c < co) == !part) ? (float)((int)((q * 2654435761u) >> 20 & 255) - 128) / 128.f : 0.f; uint32_t b; memcpy(&b, &r, 4); hr[q] = (uint16_t)(b >> 16); }
                else {
                    size_t hw_, hb_; unet_head_offsets(u, &hw_, &hb_); const int w0 = C - co; float *W = malloc((size_t)co * w0 * 4); nn_d2h(W, unet_live_params(u) + hw_, (size_t)co * w0 * 4);
                    for (size_t v = 0; v < V; v++) { float Gk[64]; for (int k = 0; k < co; k++) Gk[k] = (float)((int)(((v * 64 + k) * 2654435761u) >> 20 & 255) - 128) / 128.f;
                        for (int c = 0; c < C; c++) { float r = 0; if (c >= co) for (int k = 0; k < co; k++) r += W[(size_t)k * w0 + c - co] * Gk[k]; uint32_t b; memcpy(&b, &r, 4); hr[v * C + c] = (uint16_t)(b >> 16); } }
                    free(W);
                }
                nn_h2d(gr, hr, nv * 2);
                wt_zero_grad(wt);
                CHECK(!wt_forward_train(wt, im, fr, 9.6, o, C), "forward_train");
                CHECK(!wt_backward(wt, gr, C), "backward");
                nn_d2h(gh, unet_grad_ptr(u), np * 4);
                double gn = 0; for (size_t i = 0; i < np; i++) gn += (double)gh[i] * gh[i]; gn = sqrt(gn);
                if (!mode || getenv("WT_FD_ALL")) {
                    if (!mode) { g16[part] = malloc(np * 4); memcpy(g16[part], gh, np * 4); }
                    const double eps = (getenv("WT_EPS") ? atof(getenv("WT_EPS")) : 1e-4) * tn; double Lv[2];
                    for (int sgn = 0; sgn < 2; sgn++) {
                        float *t2 = malloc(np * 4); for (size_t i = 0; i < np; i++) t2[i] = th[i] + (float)((sgn ? -eps : eps) * gh[i] / gn); nn_h2d(p, t2, np * 4); free(t2);
                        CHECK(!wt_forward_train(wt, im, fr, 9.6, o, C), "forward_train eps");
                        nn_d2h(ho, o, nv * 2); double l = 0; for (size_t q = 0; q < nv; q++) l += (double)bf2f(ho[q]) * bf2f(hr[q]);
                        Lv[sgn] = l;
                    }
                    nn_h2d(p, th, np * 4);
                    const double fd = (Lv[0] - Lv[1]) / (2 * eps);
                    printf("16-bit-gradient check (%s): |g| %.4g, central difference along g / |g| %.4g (ratio %.3f)\n", part == 2 ? "features in the head rows' span" : part ? "features" : "heads", gn, fd, fd / gn);
                    CHECK(gn > 0 && fabs(fd / gn - 1) < 0.2, "trunk gradient (part %d) off: fd %g vs |g| %g", part, fd, gn);   /* (the central difference itself moves ~10% between runs) */
                }
                if (mode && g16[part]) {
                    double d = 0, a = 0, b = 0; for (size_t i = 0; i < np; i++) { d += (double)gh[i] * g16[part][i]; a += (double)gh[i] * gh[i]; b += (double)g16[part][i] * g16[part][i]; }
                    printf("MX-fp4 vs 16-bit gradient (%s): cosine %.3f, |g| %.4g vs %.4g\n", part ? "features" : "heads", d / sqrt(a * b + 1e-300), sqrt(a), sqrt(b));
                    CHECK(d / sqrt(a * b + 1e-300) > 0.6, "MX-fp4 trunk gradient (%s) disagrees", part ? "features" : "heads");
                }
            }
            {   /* the training forward's outputs vs the inference build's (the trained weights copied): the same network */
                uint16_t *o2 = nn_malloc(nv * 2), *h1 = malloc(nv * 2), *h2 = malloc(nv * 2);
                CHECK(!wt_forward_train(wt, im, fr, 9.6, o, C), "fwd"); nn_d2h(h1, o, nv * 2);
                CHECK(!wt_forward_ld(wt, im, fr, 1, 9.6, o2, C), "fwd inf"); nn_d2h(h2, o2, nv * 2);
                double dh = 0, df = 0, mf = 0; for (size_t q = 0; q < nv; q++) { const double d = fabs(bf2f(h1[q]) - bf2f(h2[q])); if ((int)(q % C) < co) dh += d; else { df += d; mf += fabs(bf2f(h2[q])); } }
                printf("train vs inference forward (%s): heads mean |d| %.4f, features mean |d| %.4f of mean |f| %.4f\n", mode ? "MX-fp4" : "16-bit", dh / (V * co), df / (V * (C - co)), mf / (V * (C - co)));
                nn_free(o2); free(h1); free(h2);
            }
            {   /* injection identity: a logit gradient G through the head == the same gradient injected at the head input as W^T G (all but the head's
                   own weight / bias gradients equal) */
                size_t hw, hb; unet_head_offsets(u, &hw, &hb); const int w0 = C - co; const size_t S = V;
                float *W = malloc((size_t)co * w0 * 4); nn_d2h(W, unet_live_params(u) + hw, (size_t)co * w0 * 4);
                float *G = malloc(S * co * 4); for (size_t q = 0; q < S * co; q++) G[q] = (float)((int)((q * 2246822519u) >> 20 & 255) - 128) / 1280.f;
                float *dG = nn_malloc(S * co * 4); nn_h2d(dG, G, S * co * 4);
                const float gs = unet_bwd_gscale(); uint16_t *F = malloc(S * w0 * 2);
                for (size_t v = 0; v < S; v++) for (int c = 0; c < w0; c++) { float a = 0; for (int k = 0; k < co; k++) a += W[(size_t)k * w0 + c] * G[(size_t)k * S + v]; a *= gs;
                    uint16_t hbits; if (nn_get_f16()) { hbits = f2h(a); } else { uint32_t b; memcpy(&b, &a, 4); hbits = (uint16_t)(b >> 16); } F[(size_t)c * S + v] = hbits; }
                uint16_t *dF = nn_malloc(S * w0 * 2); nn_h2d(dF, F, S * w0 * 2);
                float *zero = nn_malloc(S * co * 4); nn_zero(zero, S * co * 4);
                float *ga = malloc(np * 4), *gb = malloc(np * 4);
                wt_zero_grad(wt); CHECK(!wt_forward_train(wt, im, fr, 9.6, o, C), "fwd"); unet_backward(u, dG); nn_d2h(ga, unet_grad_ptr(u), np * 4);
                wt_zero_grad(wt); CHECK(!wt_forward_train(wt, im, fr, 9.6, o, C), "fwd"); unet_set_l0_feat_grad(u, dF); unet_backward(u, zero); unet_set_l0_feat_grad(u, nullptr); nn_d2h(gb, unet_grad_ptr(u), np * 4);
                if (!mode) {   /* central difference of sum G . logits (fp16 logits of the forward, before any sigmoid / bf16) along the head path's gradient */
                    double gn2 = 0; for (size_t i = 0; i < np; i++) gn2 += (double)ga[i] * ga[i]; gn2 = sqrt(gn2);
                    double tn2 = 0; for (size_t i = 0; i < np; i++) tn2 += (double)th[i] * th[i]; tn2 = sqrt(tn2);
                    const double eps = 1e-4 * tn2; double Lv[2]; uint16_t *yy = malloc(S * co * 2);
                    for (int sgn = 0; sgn < 2; sgn++) {
                        float *t2 = malloc(np * 4); for (size_t i = 0; i < np; i++) t2[i] = th[i] + (float)((sgn ? -eps : eps) * ga[i] / gn2); nn_h2d(p, t2, np * 4); free(t2);
                        CHECK(!wt_forward_train(wt, im, fr, 9.6, o, C), "fwd eps"); nn_d2h(yy, wt_y16(wt), S * co * 2);
                        double l = 0; for (size_t q = 0; q < S * co; q++) l += (double)G[q] * h2f(yy[q]); Lv[sgn] = l;
                    }
                    nn_h2d(p, th, np * 4); free(yy);
                    printf("logit path: |g| %.4g, central difference of sum G . logits %.4g (ratio %.3f)\n", gn2, (Lv[0] - Lv[1]) / (2 * eps), (Lv[0] - Lv[1]) / (2 * eps) / gn2);
                    CHECK(fabs((Lv[0] - Lv[1]) / (2 * eps) / gn2 - 1) < 0.2, "logit path gradient off");
                }
                double d = 0, a2 = 0, b2 = 0; for (size_t i = 0; i < np; i++) { if (i >= hw && i < hw + (size_t)co * w0) continue; if (i >= hb && i < hb + (size_t)co) continue; d += (double)ga[i] * gb[i]; a2 += (double)ga[i] * ga[i]; b2 += (double)gb[i] * gb[i]; }
                printf("injection identity (%s): cosine %.4f, |g head path| %.4g, |g injected| %.4g\n", mode ? "MX-fp4" : "16-bit", d / sqrt(a2 * b2 + 1e-300), sqrt(a2), sqrt(b2));
                CHECK(d / sqrt(a2 * b2 + 1e-300) > (mode ? 0.95 : 0.999) && fabs(sqrt(b2 / a2) - 1) < 0.1, "injection identity");
                {   /* the forward's features are the head's input: sigmoid^-1(heads) == W . features + b (from the bf16 outputs) */
                    float *B = malloc(co * 4); nn_d2h(B, unet_live_params(u) + hb, co * 4); uint16_t *hoo = malloc(nv * 2);
                    CHECK(!wt_forward_train(wt, im, fr, 9.6, o, C), "fwd"); nn_d2h(hoo, o, nv * 2); double em = 0, sm = 0; long cnt = 0;
                    for (size_t v = 0; v < S; v += 7) for (int k = 0; k < co; k++) {
                        const float sg = bf2f(hoo[v * C + k]); if (sg < 0.05f || sg > 0.95f) continue;
                        double a = B[k]; for (int c = 0; c < w0; c++) a += (double)W[(size_t)k * w0 + c] * bf2f(hoo[v * C + co + c]);
                        const double lg = log(sg / (1 - sg)); em += fabs(lg - a); sm += fabs(lg); cnt++;
                    }
                    printf("heads vs W . features + b (%s): mean |d| %.4f of mean |logit| %.3f over %ld\n", mode ? "MX-fp4" : "16-bit", em / cnt, sm / cnt, cnt);
                    CHECK(em / cnt < 0.05 * (sm / cnt) + 0.02, "features are not the head input");
                    free(B); free(hoo);
                }
                free(W); free(G); free(F); free(ga); free(gb); nn_free(dG); nn_free(dF); nn_free(zero);
            }
            free(ho); free(hr); free(th); free(gh); nn_free(o); nn_free(gr);
            wt_close(wt);
        }
        free(g16[0]); free(g16[1]); free(g16[2]);
    }
    nn_free(im); free(h);
    if (fails) { printf("test_wtrunk: %d FAILED\n", fails); return 1; }
    printf("test_wtrunk: ok\n");
    return 0;
}
