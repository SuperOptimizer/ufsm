/* Level-0 normalisation modes (unet_cfg.l0_norm, train --l0-norm), tensor-core path: none (and ws) put mean 0 / rstd 1 at the four level-0
   GroupNorm sites (enc[0] and dec[0]) and leaves the coarser levels' statistics live; ema normalises with the window's statistics
   through its warm-up, then with the running ones (the momentum blend of the earlier windows'), the EMA weights with their own;
   recalibration replaces the EMA weights' set with the mean of windows' own statistics; ws's forward is invariant to an affine
   change of a standardised conv's raw weights per output channel (none's is not); checkpoints keep the mode, the statistics and the
   update count (reloaded: the same constants); the backward stays finite.
   (Finite differences per mode: UFSM_L0_NORM=1 / 2 / 3 test_unet; MX-fp4 gradients and three level-0 buffers: test_grad_mx4.) */
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int bad;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); bad++; } } while (0)
static uint64_t rs = 777;
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) * 0x1.0p-53 * 2 - 1); }
static double var_of(float r) { return 1.0 / ((double)r * r) - 1e-5; }
static double rel(const float *a, const float *b, size_t n) { double d = 0, r = 0; for (size_t i = 0; i < n; i++) { d += ((double)a[i] - b[i]) * ((double)a[i] - b[i]); r += (double)b[i] * b[i]; } return sqrt(d / (r + 1e-300)); }
/* slab (MX storage, the training default): K training steps' parameter gradients (their mean) and the first step's logits */
static void slab_run(int md, int S, const float *x, shape5 xs, const float *gy, unsigned step0, int K, float *g, float *lg) {
    unet_cfg cfg = {3, {16, 16, 32}, 4, 2, 8}; cfg.l0_norm = md; cfg.l0_slab = S;
    unet *u = unet_create(&cfg); unet_init(u, 41);
    const size_t np = unet_nparams(u), no = shape_numel(xs) / 4 * 2; float *t = malloc(np * 4);
    memset(g, 0, np * 4);
    for (int k = 0; k < K; k++) {
        nn_set_sr_step(step0 + k); const float *l = unet_forward(u, x, xs, 1);
        if (!k) nn_d2h(lg, l, no * 4);
        unet_zero_grad(u); unet_backward(u, gy); unet_grad_d2h(u, t);
        for (size_t i = 0; i < np; i++) g[i] += t[i] / K;
    }
    free(t); unet_free(u);
}

int main(void) {
    if (nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0)) { printf("no cuda\n"); return 1; }
    nn_set_tf32(1); nn_set_f16(1);
    const shape5 xs = {1, 4, 32, 32, 32};
    const size_t nx = shape_numel(xs), no = nx / 4 * 2;
    float *h = malloc(nx * 4), *x1 = nn_malloc(nx * 4), *x2 = nn_malloc(nx * 4), *gy = nn_malloc(no * 4);
    for (size_t i = 0; i < nx; i++) h[i] = frand();
    nn_h2d(x1, h, nx * 4);
    for (size_t i = 0; i < nx; i++) h[i] = 0.5f + 2.f * frand();   /* another mean and scale: other level-0 statistics */
    nn_h2d(x2, h, nx * 4);
    for (size_t i = 0; i < no; i++) h[i] = frand();
    nn_h2d(gy, h, no * 4);
    char path[64]; snprintf(path, sizeof path, "/tmp/ufsm_test_l0_norm_%d.ckpt", (int)getpid());
    for (int md = 0; md <= 3; md++) {
        unet_cfg cfg = {3, {16, 32, 64}, 4, 2, 8};
        cfg.l0_norm = md;
        unet *u = unet_create(&cfg); unet_init(u, 5);
        unet_set_l0_ema(u, 0.25f, 2);
        const size_t ns = unet_gn_nstats(u);
        const int G = 8;   /* level-0 sites: enc[0] m1 r1 m2 r2 at [0, 4G), dec[0]'s the last 4G; enc[1]'s at [4G, 8G) */
        float *s[5]; for (int k = 0; k < 5; k++) s[k] = malloc(ns * 4);
        const int bad0 = bad;
        nn_set_sr_step(11); unet_forward(u, x1, xs, 1); unet_gn_stats_get(u, s[0]);   /* (the same seeds for window 1: the statistics agree to ~1e-4, summation order) */
        nn_set_sr_step(12); unet_forward(u, x2, xs, 1); unet_gn_stats_get(u, s[1]);
        nn_set_sr_step(11); unet_forward(u, x1, xs, 1); unet_gn_stats_get(u, s[2]);
        unet_zero_grad(u); unet_backward(u, gy);
        const size_t np = unet_nparams(u); float *g = malloc(np * 4); unet_grad_d2h(u, g);
        int fin = 1; double gn2 = 0; for (size_t i = 0; i < np; i++) { fin &= isfinite(g[i]) != 0; gn2 += (double)g[i] * g[i]; }
        CHECK(fin && gn2 > 0, "mode %d: backward gradients finite and nonzero (|g|^2 %g)", md, gn2);
        const size_t site[2] = {0, ns - 4 * (size_t)G};
        if (md == 0) {
            double d = 0; for (size_t i = 0; i < ns; i++) d = fmax(d, fabs((double)s[2][i] - s[0][i]) / (fabs((double)s[0][i]) + 1e-3));
            CHECK(d < 1e-3 && unet_l0_updates(u) == 0, "gn: the same window's statistics again (%g), no running ones (%ld)", d, unet_l0_updates(u));
        } else if (md == 1 || md == 3) {
            int c = 1; for (int b = 0; b < 2; b++) for (int k = 0; k < 4 * G; k++) c &= s[0][site[b] + k] == (k / G % 2 ? 1.f : 0.f);
            int live = 0; for (int k = 0; k < G; k++) live |= s[0][4 * G + k] != 0.f;
            CHECK(c && live, "none / ws: level-0 statistics mean 0 / rstd 1 (%d), enc[1]'s live (%d)", c, live);
        } else {
            double e = 0, dd = 0;   /* forward 3: 0.75 (window 1's) + 0.25 (window 2's), means and variances */
            for (int b = 0; b < 2; b++) for (int q = 0; q < 2; q++) for (int k = 0; k < G; k++) {
                const size_t im = site[b] + (size_t)(2 * q) * G + k, ir = im + G;
                const double m = 0.75 * s[0][im] + 0.25 * s[1][im], v = 0.75 * var_of(s[0][ir]) + 0.25 * var_of(s[1][ir]);
                e = fmax(e, fabs(s[2][im] - m) / (sqrt(v) + 1e-6));
                e = fmax(e, fabs(var_of(s[2][ir]) - v) / (v + 1e-12));
                dd = fmax(dd, fabs((double)s[0][im] - s[1][im]) / (sqrt(v) + 1e-6));
            }
            CHECK(e < 1e-4 && dd > 1e-2 && unet_l0_updates(u) == 3, "ema: warm-up live, then the running blend (err %g, windows differ by %g, %ld updates)", e, dd, unet_l0_updates(u));
            unet_use_ema(u, 1); unet_forward(u, x2, xs, 0); unet_gn_stats_get(u, s[3]); unet_use_ema(u, 0);
            unet_forward(u, x2, xs, 0); unet_gn_stats_get(u, s[4]);
            float *t = malloc(ns * 4); unet_forward(u, x1, xs, 0); unet_gn_stats_get(u, t);
            int ce = 1, cl = 1;
            for (int b = 0; b < 2; b++) for (int k = 0; k < 4 * G; k++) {   /* (rstd through the variance: an ulp) */
                ce &= fabs((double)s[3][site[b] + k] - s[0][site[b] + k]) <= 1e-6 * fabs((double)s[0][site[b] + k]); cl &= s[4][site[b] + k] == t[site[b] + k];
            }
            double e4 = 0;   /* forward 3 (training) moved them by 0.25 toward its own; its enc[0] conv1 statistics are window 1's */
            for (int k = 0; k < G; k++) {
                const double m = 0.75 * s[2][k] + 0.25 * s[0][k], v = 0.75 * var_of(s[2][G + k]) + 0.25 * var_of(s[0][G + k]);
                e4 = fmax(e4, fmax(fabs(s[4][k] - m) / sqrt(v), fabs(var_of(s[4][G + k]) - v) / v));
            }
            CHECK(ce && cl && e4 < 1e-4 && unet_l0_updates(u) == 3, "ema: the EMA weights' statistics (the first window's: no unet_ema yet) %d, "
                  "inference the running ones, no update (%d, %ld updates), the third window's update (err %g)", ce, cl, unet_l0_updates(u), e4);
            /* recalibration of the EMA weights' set: two windows with their own statistics, then their mean as the constants */
            float *ra = malloc(ns * 4), *rb = malloc(ns * 4);
            unet_use_ema(u, 1); unet_l0_recal(u, 1);
            unet_forward(u, x1, xs, 0); unet_gn_stats_get(u, ra);
            unet_forward(u, x2, xs, 0); unet_gn_stats_get(u, rb);
            unet_l0_recal(u, 0); unet_forward(u, x1, xs, 0); unet_gn_stats_get(u, t); unet_use_ema(u, 0);
            double er = 0, dr = 0;
            for (int b = 0; b < 2; b++) for (int q = 0; q < 2; q++) for (int k = 0; k < G; k++) {
                const size_t im = site[b] + (size_t)(2 * q) * G + k, ir = im + G;
                const double m = 0.5 * ra[im] + 0.5 * rb[im], v = 0.5 * var_of(ra[ir]) + 0.5 * var_of(rb[ir]);
                er = fmax(er, fmax(fabs(t[im] - m) / sqrt(v), fabs(var_of(t[ir]) - v) / v));
                dr = fmax(dr, fabs((double)ra[im] - s[0][im]) / sqrt(v));   /* (window 1 live under the same weights: as forward 1) */
            }
            CHECK(er < 1e-4 && unet_l0_updates(u) == 3, "ema recalibration: the EMA weights' constants = the mean of the windows' statistics (err %g, %ld updates)", er, unet_l0_updates(u));
            CHECK(dr < 1e-3, "ema recalibration: the forwards normalise with their own statistics (window 1 vs forward 1: %g)", dr);
            free(ra); free(rb);
            free(t);
        }
        if (md) {   /* checkpoint round trip: the mode, the update count and the constants (bits) */
            unet_ema(u, 0.5f);
            unet_use_ema(u, 1); unet_forward(u, x2, xs, 0); unet_gn_stats_get(u, s[3]); unet_use_ema(u, 0);
            unet_forward(u, x2, xs, 0); unet_gn_stats_get(u, s[4]);
            CHECK(!unet_save(u, path, 7, nullptr), "save");
            unet_cfg pc; int st = 0;
            CHECK(!unet_peek(path, &pc, &st) && pc.l0_norm == md && st == 7, "peek: l0_norm %d", pc.l0_norm);
            unet *v = unet_create(&pc);
            CHECK(unet_load(v, path) == 7 && unet_l0_updates(v) == unet_l0_updates(u), "load: %ld updates", unet_l0_updates(v));
            float *t = malloc(ns * 4);
            int same = 1;
            unet_use_ema(v, 1); unet_forward(v, x2, xs, 0); unet_gn_stats_get(v, t); unet_use_ema(v, 0);
            for (int b = 0; b < 2; b++) for (int k = 0; k < 4 * G; k++) same &= t[site[b] + k] == s[3][site[b] + k];
            unet_forward(v, x2, xs, 0); unet_gn_stats_get(v, t);
            for (int b = 0; b < 2; b++) for (int k = 0; k < 4 * G; k++) same &= t[site[b] + k] == s[4][site[b] + k];
            CHECK(same, "reloaded: the level-0 constants of the live and the EMA weights bit-identical");
            free(t); unet_free(v); remove(path);
        }
        if (md == 1 || md == 3) {   /* ws: the forward is invariant to an affine change of a standardised conv's raw weights per output
                                       channel (the stem's: first in the layout, 3 w + 0.5); none is not */
            float *l0 = malloc(no * 4), *l1 = malloc(no * 4);
            nn_set_sr_step(21); nn_d2h(l0, unet_forward(u, x1, xs, 0), no * 4);
            CHECK(!unet_save(u, path, 7, nullptr), "save");
            FILE *f = fopen(path, "r+b"); char line[16384];
            fseek(f, 4, SEEK_SET); if (!fgets(line, sizeof line, f)) bad++;
            const long p0 = ftell(f); const size_t nw = (size_t)16 * 4 * 27;
            float *pp = malloc(np * 4);
            for (int a = 0; a < 2; a++) {   /* live and EMA weights */
                fseek(f, p0 + (long)(a * np * 4), SEEK_SET); if (fread(pp, 4, np, f) != np) bad++;
                for (size_t i = 0; i < nw; i++) pp[i] = 3.f * pp[i] + 0.5f;
                fseek(f, p0 + (long)(a * np * 4), SEEK_SET); fwrite(pp, 4, np, f);
            }
            fclose(f);
            CHECK(unet_load(u, path) == 7, "load the changed stem");
            nn_set_sr_step(21); nn_d2h(l1, unet_forward(u, x1, xs, 0), no * 4);
            double d = 0, mx = 0; for (size_t i = 0; i < no; i++) { d = fmax(d, fabs((double)l1[i] - l0[i])); mx = fmax(mx, fabs((double)l0[i])); }
            if (md == 3) CHECK(d < 0.02 * mx, "ws: logits invariant to the stem's raw-weight scale and shift (max diff %g of %g)", d, mx);
            else CHECK(d > 0.1 * mx, "none: logits change with the stem's raw weights (max diff %g of %g)", d, mx);
            free(l0); free(l1); free(pp); remove(path);
        }
        printf("  l0_norm %d: %s\n", md, bad > bad0 ? "FAIL" : "ok");
        for (int k = 0; k < 5; k++) free(s[k]);
        free(g); unet_free(u);
    }
    {   /* slab: GroupNorm statistics per z-slab. One slab (S = the depth) is GroupNorm but for the extra MX-fp4 rounding of the normalised
           tensor (and the fp8 one of the rescaled gradient); 16-plane slabs on an input whose statistics change along z move the logits
           further from GroupNorm than that rounding. MX-fp4 activations, MX-fp8 gradients (less noise than fp4); the gradients' SR noise
           floor from two seed sets of GroupNorm (printed). The slabs' arithmetic: the fp32 finite differences (UFSM_L0_NORM=4 test_unet) */
        const int bad0 = bad;
        nn_set_tf32(1); nn_set_act_bf16(1); nn_set_grad_bf16(1); nn_set_f16(1); nn_set_grad_scale(1024); nn_set_sr(1); nn_set_gn_stored(1);
        nn_set_prec(3); if (nn_set_prec_policy("all=fp4:fp4:fp4,enc0.c1=fp16")) bad++;
        setenv("UFSM_F4_WGRAD", "1", 1);
        unet_set_act_mx4(1); unet_set_grad_mx8(1); unet_set_grad_mx4(0); unet_set_input_prec(8); unet_set_recompute(1); unet_set_chunk_up(2); unet_set_lean(2);
        const shape5 zs = {1, 4, 64, 32, 32};
        const size_t zn = shape_numel(zs), zo = zn / 4 * 2, S3 = (size_t)64 * 32 * 32;
        float *hz = malloc(zn * 4);
        for (size_t i = 0; i < zn; i++) { const int z = (int)(i % S3 / 1024); hz[i] = (1.f + z / 16.f) * frand() + 0.05f * z; }   /* z-dependent mean and scale */
        float *xz = nn_malloc(zn * 4), *gz = nn_malloc(zo * 4);
        nn_h2d(xz, hz, zn * 4); for (size_t i = 0; i < zo; i++) hz[i] = frand(); nn_h2d(gz, hz, zo * 4);
        unet_cfg c0 = {3, {16, 16, 32}, 4, 2, 8}; unet *t0 = unet_create(&c0); const size_t np = unet_nparams(t0); unet_free(t0);
        float *gA = malloc(np * 4), *gB = malloc(np * 4), *gD = malloc(np * 4), *g16 = malloc(np * 4);
        float *lA = malloc(zo * 4), *lB = malloc(zo * 4), *lD = malloc(zo * 4), *l16 = malloc(zo * 4);
        const int K = 4;
        slab_run(0, 0, xz, zs, gz, 100, K, gA, lA); slab_run(0, 0, xz, zs, gz, 200, K, gB, lB);
        slab_run(4, 64, xz, zs, gz, 100, K, gD, lD); slab_run(4, 16, xz, zs, gz, 100, K, g16, l16);
        int fin = 1; for (size_t i = 0; i < np; i++) fin &= isfinite(gD[i]) && isfinite(g16[i]);
        const double fl = rel(gB, gA, np), dD = rel(gD, gA, np), d16 = rel(g16, gA, np), lfl = rel(lB, lA, zo), lD_ = rel(lD, lA, zo), l16_ = rel(l16, lA, zo);
        printf("  slab: grads vs gn: noise floor %.3g, one slab %.3g, 16-plane slabs %.3g; logits: %.3g, %.3g, %.3g\n", fl, dD, d16, lfl, lD_, l16_);
        CHECK(fin && lfl == 0 && lD_ < 0.5, "slab: finite gradients, one slab = GroupNorm up to the extra rounding (logits %g, deterministic gn %g)", lD_, lfl);
        {   /* the fp32 path: one slab is GroupNorm exactly, 16-plane slabs differ on the z-graded input; the MX path tracks its fp32
               reference with 16-plane slabs about as well as GroupNorm's MX path tracks its own */
            float *lf[3], *lm[3], *gf[3], *gm[3]; const int md_[3] = {0, 4, 4}, S_[3] = {0, 64, 16};
            for (int a = 0; a < 3; a++) {
                lf[a] = malloc(zo * 4); lm[a] = malloc(zo * 4); gf[a] = malloc(np * 4); gm[a] = calloc(np, 4);
                unet_cfg c = {3, {16, 16, 32}, 4, 2, 8}; c.l0_norm = md_[a]; c.l0_slab = S_[a];
                unet *u = unet_create(&c); unet_init(u, 41);
                nn_set_tf32(0); nn_set_sr_step(300); nn_d2h(lf[a], unet_forward(u, xz, zs, 1), zo * 4);
                unet_zero_grad(u); unet_backward(u, gz); unet_grad_d2h(u, gf[a]);
                nn_set_tf32(1);
                for (int k = 0; k < K; k++) {   /* (the MX gradients: the mean of K SR seeds) */
                    nn_set_sr_step(300 + k); const float *l = unet_forward(u, xz, zs, 1); if (!k) nn_d2h(lm[a], l, zo * 4);
                    unet_zero_grad(u); unet_backward(u, gz); unet_grad_d2h(u, gA);
                    for (size_t i = 0; i < np; i++) gm[a][i] += gA[i] / K;
                }
                unet_free(u);
            }
            const double e1 = rel(lf[1], lf[0], zo), e16 = rel(lf[2], lf[0], zo), q0 = rel(lm[0], lf[0], zo), q16 = rel(lm[2], lf[2], zo);
            const double ge1 = rel(gf[1], gf[0], np), ge16 = rel(gf[2], gf[0], np), gq0 = rel(gm[0], gf[0], np), gq16 = rel(gm[2], gf[2], np);
            printf("  slab fp32: one slab vs gn %.3g, 16-plane slabs vs gn %.3g; MX vs fp32: gn %.3g, 16-plane slabs %.3g (logits)\n", e1, e16, q0, q16);
            printf("  slab fp32: one slab vs gn %.3g, 16-plane slabs vs gn %.3g; MX vs fp32: gn %.3g, 16-plane slabs %.3g (gradients)\n", ge1, ge16, gq0, gq16);
            CHECK(e1 < 1e-5 && ge1 < 1e-4 && e16 > 0.05 && ge16 > 0.05, "slab fp32: one slab = GroupNorm (%g, %g), 16-plane slabs another function (%g, %g)", e1, ge1, e16, ge16);
            CHECK(q16 < 1.5 * q0 + 0.1 && gq16 < 1.5 * gq0 + 0.1, "slab: the MX path's 16-plane slabs track the fp32 reference (logits %g, gradients %g; GroupNorm %g, %g)", q16, gq16, q0, gq0);
            for (int a = 0; a < 3; a++) { free(lf[a]); free(lm[a]); free(gf[a]); free(gm[a]); }
        }
        {   /* checkpoint: the mode and the slab size */
            unet_cfg c4 = {3, {16, 16, 32}, 4, 2, 8}; c4.l0_norm = 4; c4.l0_slab = 32;
            unet *u = unet_create(&c4); unet_init(u, 3);
            nn_set_sr_step(5); unet_forward(u, xz, zs, 1);
            CHECK(!unet_save(u, path, 9, nullptr), "save");
            unet_cfg pc; int st = 0;
            CHECK(!unet_peek(path, &pc, &st) && pc.l0_norm == 4 && pc.l0_slab == 32, "slab: peek l0_norm %d l0_slab %d", pc.l0_norm, pc.l0_slab);
            unet *v = unet_create(&pc); CHECK(unet_load(v, path) == 9, "slab: load");
            nn_set_sr_step(6); nn_d2h(lA, unet_forward(u, xz, zs, 0), zo * 4);
            nn_set_sr_step(6); nn_d2h(lB, unet_forward(v, xz, zs, 0), zo * 4);
            CHECK(rel(lB, lA, zo) < 1e-6, "slab: reloaded inference logits (rel diff %g)", rel(lB, lA, zo));
            unet_free(u); unet_free(v); remove(path);
        }
        printf("  l0_norm 4 (slab, MX): %s\n", bad > bad0 ? "FAIL" : "ok");
        free(gA); free(gB); free(gD); free(g16); free(lA); free(lB); free(lD); free(l16); free(hz); nn_free(xz); nn_free(gz);
    }
    const char *e = nn_check();
    if (e) { printf("cuda error: %s\n", e); bad++; }
    printf("level-0 norm modes: %s\n", bad ? "FAIL" : "PASS");
    return bad != 0;
}
