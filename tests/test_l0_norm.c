/* Level-0 normalisation modes (unet_cfg.l0_norm, train --l0-norm), tensor-core path: none puts mean 0 / rstd 1 at the four level-0
   GroupNorm sites (enc[0] and dec[0]) and leaves the coarser levels' statistics live; ema normalises with the window's statistics
   through its warm-up, then with the running ones (the momentum blend of the earlier windows'), the EMA weights with their own;
   checkpoints keep the mode, the statistics and the update count (reloaded: the same constants); the backward stays finite.
   (Finite differences per mode: UFSM_L0_NORM=1 / 2 test_unet; MX-fp4 gradients and three level-0 buffers: test_grad_mx4.) */
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
    for (int md = 0; md <= 2; md++) {
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
        } else if (md == 1) {
            int c = 1; for (int b = 0; b < 2; b++) for (int k = 0; k < 4 * G; k++) c &= s[0][site[b] + k] == (k / G % 2 ? 1.f : 0.f);
            int live = 0; for (int k = 0; k < G; k++) live |= s[0][4 * G + k] != 0.f;
            CHECK(c && live, "none: level-0 statistics mean 0 / rstd 1 (%d), enc[1]'s live (%d)", c, live);
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
        printf("  l0_norm %d: %s\n", md, bad > bad0 ? "FAIL" : "ok");
        for (int k = 0; k < 5; k++) free(s[k]);
        free(g); unet_free(u);
    }
    const char *e = nn_check();
    if (e) { printf("cuda error: %s\n", e); bad++; }
    printf("level-0 norm modes: %s\n", bad ? "FAIL" : "PASS");
    return bad != 0;
}
