/* --init-widen (unet_init_widen, DESIGN 2026-11-14 "Widening warm start"): a 3-level donor (down_norm, film) widened into a 4-level net
   with gfilm and a bottleneck attention stage (the extra level between the donor's bottom and its first decoder: the +/- pass-through).
   The same window through both nets (inference, the donor's GroupNorm the window's; the widened net's gn_ema with no running statistics
   yet: the window's too):
   (1) widths whole multiples of the donor's (every GroupNorm group holds whole copies): the logits equal the donor's up to the
       arithmetic (16-bit storage: mean |d| / mean |logit| < 1e-2; the MX-fp4 path within twice the donor's own fp4 - 16-bit gap);
   (2) the u1 -> scratch2 shape (96,128,160 -> 256,320,384,96: 12 -> 32 channels per group, uneven copies): reported, the deviation
       of the uneven GroupNorm weighting;
   (3) inherited ranges recorded, optimiser state 0.
   TW_DONOR=ckpt TW_TARGET=ckpt (a checkpoint whose header gives the target net; --down-norm set to the donor's): the real pair (u1 ->
   scratch2) on a TW_P^3-cell latent window (default 64), random inputs, reported only. */
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int bad;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); bad++; } } while (0)
static uint64_t rs = 4242;
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) * 0x1.0p-53 * 2 - 1); }

static void prec(int fp4) {
    nn_set_tf32(1); nn_set_f16(1); nn_set_sr(0);
    if (fp4) { nn_set_prec(3); nn_set_prec_policy("all=fp4:fp4:fp4,enc0.c1=fp16"); unet_set_act_mx4(1); }
    else { nn_set_prec(1); nn_set_prec_policy(""); unet_set_act_mx4(0); }
}
/* mean |a - b| / mean |a| */
static double rel(const float *a, const float *b, size_t n) { double e = 0, m = 0; for (size_t i = 0; i < n; i++) { e += fabs((double)a[i] - b[i]); m += fabs(a[i]); } return e / (m > 0 ? m : 1); }
static const float g_codes[2] = {0.4f, -0.7f};
/* the logits of checkpoint path (net cfg c) on window dx in one precision (a fresh net per precision: the activation storage is built
   with it) */
static float *logits(const unet_cfg *c, const char *path, const float *dx, shape5 xs, int fp4, size_t *no) {
    prec(fp4);
    unet *u = unet_create(c); if (unet_load(u, path) < 0) { printf("cannot load %s\n", path); exit(1); }
    unet_set_film(u, g_codes, 2);
    *no = shape_numel(unet_out_shape(u, xs)); float *h = malloc(*no * 4);
    nn_d2h(h, unet_forward(u, dx, xs, 0), *no * 4); unet_free(u);
    return h;
}
/* the donor and the widened net on one window, 16-bit and MX-fp4: the donor's channels 0 .. nc - 1 against the same of the widened net */
static double pair(const char *tag, const unet_cfg *dc, const char *dp, const unet_cfg *wc, const char *wp, shape5 xs, int nc, double *r4, double *rd4) {
    const size_t nx = shape_numel(xs);
    float *x = malloc(nx * 4); for (size_t i = 0; i < nx; i++) x[i] = frand();
    float *dx = nn_malloc(nx * 4); nn_h2d(dx, x, nx * 4);
    size_t na, nb;
    float *a = logits(dc, dp, dx, xs, 0, &na), *b = logits(wc, wp, dx, xs, 0, &nb), *a4 = logits(dc, dp, dx, xs, 1, &na), *b4 = logits(wc, wp, dx, xs, 1, &nb);
    const size_t n = (size_t)nc * (na / dc->cout);   /* (channel-major logits: the first nc channels) */
    const double r = rel(a, b, n); *r4 = rel(a4, b4, n); *rd4 = rel(a, a4, n);
    double mx = 0; for (size_t i = 0; i < n; i++) { const double e = fabs((double)a[i] - b[i]); if (e > mx) mx = e; }
    printf("%s: 16-bit mean |d| / mean |logit| %.3g (max |d| %.3g); MX-fp4 %.3g (the donor's own fp4 vs 16-bit gap %.3g)\n", tag, r, mx, *r4, *rd4);
    free(x); free(a); free(b); free(a4); free(b4); nn_free(dx);
    return r;
}
static unet *donor_of(const unet_cfg *c, const char *path) {
    unet *d = unet_create(c); unet_init(d, 7);
    {   /* non-trivial GroupNorm affines, biases and FiLM pairs (fresh ones are 1 / 0) */
        const size_t np = unet_nparams(d); float *p = malloc(np * 4); nn_d2h(p, unet_live_params(d), np * 4);
        float *h = malloc(np * 4); memcpy(h, p, np * 4);
        for (size_t i = 0; i < np; i++) if (p[i] == 1.f) h[i] = 1.f + 0.3f * frand(); else if (p[i] == 0.f) h[i] = 0.1f * frand();
        nn_h2d((float *)unet_live_params(d), h, np * 4); free(p); free(h);
    }
    unet_save(d, path, 1, nullptr);
    unet_free(d); d = unet_create(c); unet_load(d, path);
    return d;
}

int main(void) {
    if (nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0)) { printf("no cuda\n"); return 1; }
    char path[256], wpath[256]; snprintf(path, sizeof path, "%s/test_widen.%d.ckpt", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp", (int)getpid());
    snprintf(wpath, sizeof wpath, "%s.w", path);
    for (int cse = 0; cse < 2; cse++) {
        unet_cfg dc = {}; dc.nlev = 3; dc.cin = 4; dc.cout = 3; dc.G = 8; dc.down_norm = 1; dc.film = 2;
        unet_cfg tc = dc; tc.nlev = 4; tc.attn_blocks = 2; tc.attn_pe = 2; tc.gfilm = 1; tc.gn_ema = 1; tc.cout = 5;
        if (!cse) { dc.widths[0] = 32; dc.widths[1] = 64; dc.widths[2] = 96; tc.widths[0] = 64; tc.widths[1] = 128; tc.widths[2] = 192; tc.widths[3] = 96; }
        else { dc.widths[0] = 96; dc.widths[1] = 128; dc.widths[2] = 160; tc.widths[0] = 256; tc.widths[1] = 320; tc.widths[2] = 384; tc.widths[3] = 96; }
        prec(0); unet *d = donor_of(&dc, path), *w = unet_create(&tc); unet_init(w, 11);
        const int rows[5] = {0, 1, 2, -1, -1};
        const int n = unet_init_widen(w, path, rows, nullptr);
        CHECK(n > 0, "unet_init_widen (%d)", n);
        const shape5 xs = {1, 4, 32, 32, 32};
        double r4, rd4;
        unet_save(w, wpath, 1, nullptr);
        const double r = pair(cse ? "(2) u1 shape, uneven copies" : "(1) whole copies", &dc, path, &tc, wpath, xs, 3, &r4, &rd4);
        if (!cse) { CHECK(r < 1e-2, "(1) 16-bit: the donor's function (%.3g)", r); CHECK(r4 < 2 * rd4 + 1e-3, "(1) MX-fp4: within twice the donor's own fp4 gap (%.3g vs %.3g)", r4, rd4); }
        else CHECK(isfinite(r) && r < 0.5, "(2) finite, near the donor (%.3g)", r);
        if (!cse) {
            CHECK(unet_inherited(w) > 0 && unet_inherited(w) < unet_nparams(w), "(3) inherited %zu of %zu", unet_inherited(w), unet_nparams(w));
            CHECK(unet_init_widen(w, "/nonexistent", rows, nullptr) < 0, "(3) a missing donor fails");
            unet_cfg bc = tc; bc.widths[2] = 160; unet *x = unet_create(&bc); unet_init(x, 3);   /* the bottom too narrow for the pass-through */
            CHECK(unet_init_widen(x, path, rows, nullptr) < 0, "(3) a bottom narrower than 2x the donor's is refused"); unet_free(x);
        }
        unet_free(d); unet_free(w);
    }
    remove(path); remove(wpath);
    const char *dp = getenv("TW_DONOR"), *tp = getenv("TW_TARGET");
    if (dp && tp) {   /* the real pair, e.g. u1 -> scratch2's shape */
        unet_cfg dc, tc; int s1, s2;
        if (unet_peek(dp, &dc, &s1) || unet_peek(tp, &tc, &s2)) { printf("cannot read %s / %s\n", dp, tp); return 1; }
        tc.down_norm = dc.down_norm; tc.skip_thr = 0;
        prec(0); unet *w = unet_create(&tc); unet_init(w, 11);
        int rows[512]; for (int co = 0; co < tc.cout; co++) rows[co] = co < 11 ? co : co >= tc.cout - 128 ? dc.cout - 128 + co - (tc.cout - 128) : -1;
        rows[12] = 11; rows[13] = 12;   /* (u1's rv pair 11-12 -> 12-13; ink 11 and the fibre rows fresh) */
        int frows[16]; for (int k = 0; k < 16; k++) frows[k] = k < 4 ? k : k == 5 ? 4 : k == 6 ? 5 : -1;   /* (stem outputs 0 1 9 10 | ink 11 | rv 12 13 | fibre) */
        const int n = unet_init_widen(w, dp, rows, frows);
        CHECK(n > 0, "the real pair widens (%d)", n);
        const int P = getenv("TW_P") ? atoi(getenv("TW_P")) : 64;
        const shape5 xs = {1, dc.cin, P, P, P};
        double r4, rd4;
        /* the donor's channels 0-10 against the widened net's 0-10 (the rows by name; the head's other rows differ in place) */
        unet_save(w, wpath, 1, nullptr); unet_free(w);
        unet_cfg wc; int s3; unet_peek(wpath, &wc, &s3);
        pair("(real) donor rows 0-10", &dc, dp, &wc, wpath, xs, 11, &r4, &rd4);
        remove(wpath);
    }
    if (bad) { printf("test_widen: %d failed\n", bad); return 1; }
    printf("test_widen: ok\n");
    return 0;
}
