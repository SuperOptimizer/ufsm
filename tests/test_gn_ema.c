/* Running GroupNorm statistics at every site (unet_cfg.gn_ema, train --gn-ema 1), tensor-core path (MX storage, the training
   default), nets with down norms and with a bottleneck attention stage:
   - warm-up: the first training forwards normalise with the window's own statistics, then every site holds the running blend
     (momentum toward each forward's statistics, [mean | var] per group) and the convs leave it alone;
   - locality: once constant, an input change far outside the receptive field leaves the logits bit-identical (live GroupNorm: every
     logit moves with the window statistics, the coupling the 2026-11-13 ERF probe measured at 58% of the local change in u1);
   - momentum 1: a fresh net's first (live, warm-up) training forward sets the constants to that window's statistics, so its
     inference forward matches live GroupNorm (to the var <-> rstd round trip);
   - the EMA weights keep their own set (unet_ema blends the batch statistics in), EMA-weight forwards do not update;
   - checkpoints keep the flag, the statistics and the update count (reloaded: bit-identical logits), an old checkpoint (no
     statistics) loads into a gn_ema net as a fresh warm-up;
   - the backward with constant statistics stays finite. (Finite differences: UFSM_GN_EMA=1 test_unet, the fp32 path.) */
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
static double var_of(float r) { return 1.0 / ((double)r * r) - 1e-5; }
static double maxdiff(const float *a, const float *b, size_t n) { double d = 0; for (size_t i = 0; i < n; i++) { const double e = fabs((double)a[i] - b[i]); if (e > d) d = e; } return d; }

/* the logits of a forward (train 0 / 1) to the host */
static void fwd(unet *u, const float *dx, shape5 xs, int train, float *h, size_t no) { nn_d2h(h, unet_forward(u, dx, xs, train), no * 4); }

static void run(const char *name, unet_cfg cfg) {
    printf("%s\n", name);
    char path[96]; snprintf(path, sizeof path, "/tmp/ufsm_test_gn_ema_%d.ckpt", (int)getpid());
    const int P = 96; const shape5 xs = {1, 4, P, P, P};
    const size_t nx = shape_numel(xs);
    cfg.gn_ema = 1;
    unet *u = unet_create(&cfg); unet_init(u, 7);
    const shape5 os = unet_out_shape(u, xs); const size_t no = shape_numel(os);
    float *x1 = malloc(nx * 4), *x2 = malloc(nx * 4), *x3 = malloc(nx * 4);
    for (size_t i = 0; i < nx; i++) { x1[i] = frand(); x2[i] = 0.5f * frand() + 0.3f; }
    float *d1 = nn_malloc(nx * 4), *d2 = nn_malloc(nx * 4), *d3 = nn_malloc(nx * 4); nn_h2d(d1, x1, nx * 4); nn_h2d(d2, x2, nx * 4);
    const size_t ns = unet_gn_nstats(u);
    float *s[6]; for (int k = 0; k < 6; k++) s[k] = malloc(ns * 4);
    float *la = malloc(no * 4), *lb = malloc(no * 4), *lc = malloc(no * 4);

    /* (1) warm-up 2, momentum 0.25: forwards 1 and 2 live, 3 the blend (of 1 and 2), and the sites are left as they were */
    unet_set_gn_ema(u, 0.25f, 2);
    nn_set_sr_step(1); unet_forward(u, d1, xs, 1); unet_gn_stats_get(u, s[0]);
    nn_set_sr_step(2); unet_forward(u, d2, xs, 1); unet_gn_stats_get(u, s[1]);
    nn_set_sr_step(3); unet_forward(u, d1, xs, 1); unet_gn_stats_get(u, s[2]);
    double e = 0, dd = 0;
    /* the layout pairs a site's means with its rstds; with no site table here compare every element both ways: as a mean (linear blend)
       and as an rstd (blend of variances), take the better match per element (the two disagree by far more than the tolerance) */
    for (size_t i = 0; i < ns; i++) {
        const double m = 0.75 * s[0][i] + 0.25 * s[1][i];
        const double v = 0.75 * var_of(s[0][i]) + 0.25 * var_of(s[1][i]), r = 1.0 / sqrt((v > 0 ? v : 0) + 1e-5);
        const double em = fabs(s[2][i] - m) / (fabs(m) + 1e-3), er = fabs(s[2][i] - r) / (fabs(r) + 1e-3);
        const double ee = em < er ? em : er; if (ee > e) e = ee;
        const double d = fabs(s[0][i] - s[1][i]) / (fabs(s[0][i]) + 1e-3); if (d > dd) dd = d;
    }
    CHECK(e < 1e-4 && dd > 1e-2 && unet_gn_ema_updates(u) == 3, "warm-up live, then the running blend at every site (err %g, windows differ by %g, %ld updates)", e, dd, unet_gn_ema_updates(u));

    /* (2) locality: x3 = x1 with its first 16^3 corner changed; logits of the opposite 16^3 corner (beyond the receptive field) */
    memcpy(x3, x1, nx * 4);
    for (int c = 0; c < 4; c++) for (int z = 0; z < 16; z++) for (int y = 0; y < 16; y++) for (int xx = 0; xx < 16; xx++) x3[(((size_t)c * P + z) * P + y) * P + xx] += 2.f;
    nn_h2d(d3, x3, nx * 4);
    fwd(u, d1, xs, 0, la, no); fwd(u, d3, xs, 0, lb, no);
    double far = 0, near = 0;
    for (int c = 0; c < os.c; c++) for (int z = 0; z < 16; z++) for (int y = 0; y < 16; y++) for (int xx = 0; xx < 16; xx++) {
        const size_t f = (((size_t)c * P + P - 1 - z) * P + P - 1 - y) * P + P - 1 - xx, n = (((size_t)c * P + z) * P + y) * P + xx;
        const double a = fabs((double)la[f] - lb[f]), b = fabs((double)la[n] - lb[n]); if (a > far) far = a; if (b > near) near = b;
    }
    CHECK(far == 0 && near > 1e-3, "running statistics: a far corner's logits bit-identical (%g), the changed corner's move (%g)", far, near);
    {   /* the same with live GroupNorm (a net without gn_ema, the same weights): the far corner moves */
        unet_cfg c0 = cfg; c0.gn_ema = 0; unet *v = unet_create(&c0); unet_init(v, 7);
        fwd(v, d1, xs, 0, la, no); fwd(v, d3, xs, 0, lb, no);
        double f0 = 0;
        for (int c = 0; c < os.c; c++) for (int z = 0; z < 16; z++) for (int y = 0; y < 16; y++) for (int xx = 0; xx < 16; xx++) {
            const size_t f = (((size_t)c * P + P - 1 - z) * P + P - 1 - y) * P + P - 1 - xx; const double a = fabs((double)la[f] - lb[f]); if (a > f0) f0 = a;
        }
        CHECK(f0 > 1e-4, "live GroupNorm: the far corner moves with the window statistics (%g)", f0);
        /* (3) momentum 1: a fresh net's first (warm-up, live) training forward of x2 sets the constants to x2's own statistics:
           forwarding x2 again (inference, the constants) matches the live-GroupNorm net on x2 (to the var <-> rstd round trip) */
        unet *w = unet_create(&cfg); unet_init(w, 7); unet_set_gn_ema(w, 1.f, 100);
        nn_set_sr_step(4); unet_forward(w, d2, xs, 1);
        fwd(w, d2, xs, 0, la, no); fwd(v, d2, xs, 0, lb, no);
        unet_gn_stats_get(w, s[3]); unet_gn_stats_get(v, s[4]);
        double se = 0; for (size_t i = 0; i < ns; i++) { const double d = fabs((double)s[3][i] - s[4][i]) / (fabs(s[4][i]) + 1e-3); if (d > se) se = d; }
        double mx = 0; for (size_t i = 0; i < no; i++) { const double a = fabs(lb[i]); if (a > mx) mx = a; }
        const double dm = maxdiff(la, lb, no);
        printf("  momentum 1: statistics rel err %.3g, max |logit difference| %.3g of max |logit| %.3g\n", se, dm, mx);
        CHECK(se < 1e-4 && dm <= 1e-2 * mx, "momentum 1: the window's own statistics as constants = live GroupNorm (statistics %g, logits %g of %g)", se, dm, mx);
        unet_free(w);
        unet_free(v);
    }

    /* (4) EMA weights: their own set (the first batch's until unet_ema), EMA-weight training forwards leave the counts alone */
    {
        const long n0 = unet_gn_ema_updates(u);
        unet_use_ema(u, 1); nn_set_sr_step(5); unet_forward(u, d1, xs, 1); unet_use_ema(u, 0);
        CHECK(unet_gn_ema_updates(u) == n0, "EMA-weight forwards do not update (%ld -> %ld)", n0, unet_gn_ema_updates(u));
        unet_ema(u, 0.5f);
        unet_use_ema(u, 1); fwd(u, d1, xs, 0, la, no); unet_gn_stats_get(u, s[3]); unet_use_ema(u, 0);
        fwd(u, d1, xs, 0, lb, no); unet_gn_stats_get(u, s[4]);
        CHECK(memcmp(s[3], s[4], ns * 4), "the EMA weights' constants differ from the live weights' after unet_ema");
    }

    /* (5) checkpoint: flag, statistics, count; reloaded logits bit-identical (live and EMA weights) */
    {
        fwd(u, d2, xs, 0, la, no); unet_use_ema(u, 1); fwd(u, d2, xs, 0, lc, no); unet_use_ema(u, 0);
        CHECK(!unet_save(u, path, 7, nullptr), "save");
        unet_cfg pc; int st; CHECK(!unet_peek(path, &pc, &st) && pc.gn_ema == 1, "peek: gn_ema %d", pc.gn_ema);
        unet *v = unet_create(&pc);
        CHECK(unet_load(v, path) == 7 && unet_gn_ema_updates(v) == unet_gn_ema_updates(u), "load: %ld updates", unet_gn_ema_updates(v));
        fwd(v, d2, xs, 0, lb, no);
        CHECK(!memcmp(la, lb, no * 4), "reloaded: the live weights' logits bit-identical (max diff %g)", maxdiff(la, lb, no));
        unet_use_ema(v, 1); fwd(v, d2, xs, 0, lb, no); unet_use_ema(v, 0);
        CHECK(!memcmp(lc, lb, no * 4), "reloaded: the EMA weights' logits bit-identical (max diff %g)", maxdiff(lc, lb, no));
        unet_free(v);
        /* an old checkpoint (no statistics) into a gn_ema net: a fresh warm-up */
        unet_cfg c0 = cfg; c0.gn_ema = 0; unet *o = unet_create(&c0); unet_init(o, 9);
        CHECK(!unet_save(o, path, 3, nullptr), "save (no gn_ema)");
        unet *w = unet_create(&cfg);
        CHECK(unet_load(w, path) == 3 && unet_gn_ema_updates(w) == 0, "an old checkpoint: %ld updates (a fresh warm-up)", unet_gn_ema_updates(w));
        fwd(w, d1, xs, 1, la, no); fwd(o, d1, xs, 1, lb, no);
        CHECK(!memcmp(la, lb, no * 4), "fresh warm-up: the first training forward is live GroupNorm (max diff %g)", maxdiff(la, lb, no));
        unet_free(w); unet_free(o);
        unlink(path);
    }

    /* (6) the backward with constant statistics: finite gradients */
    {
        unet_set_gn_ema(u, 0.01f, 1);
        float *gy = malloc(no * 4); for (size_t i = 0; i < no; i++) gy[i] = frand();
        float *dg = nn_malloc(no * 4); nn_h2d(dg, gy, no * 4);
        nn_set_sr_step(6); unet_forward(u, d1, xs, 1); unet_zero_grad(u); unet_backward(u, dg);
        const size_t np = unet_nparams(u); float *g = malloc(np * 4); unet_grad_d2h(u, g);
        double n2 = 0; int fin = 1; for (size_t i = 0; i < np; i++) { if (!isfinite(g[i])) fin = 0; n2 += (double)g[i] * g[i]; }
        CHECK(fin && n2 > 0, "constant statistics: the backward's gradients finite and non-zero (norm %g)", sqrt(n2));
        free(g); free(gy); nn_free(dg);
    }
    for (int k = 0; k < 6; k++) free(s[k]);
    free(x1); free(x2); free(x3); free(la); free(lb); free(lc); nn_free(d1); nn_free(d2); nn_free(d3);
    unet_free(u);
}

int main(void) {
    if (nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0)) { printf("no cuda\n"); return 1; }
    nn_set_tf32(1); nn_set_f16(1);
    unet_cfg a = {}; a.nlev = 3; a.widths[0] = 16; a.widths[1] = 16; a.widths[2] = 32; a.cin = 4; a.cout = 2; a.G = 8; a.down_norm = 1;
    run("down norms (16,16,32)", a);
    unet_cfg b = {}; b.nlev = 3; b.widths[0] = 16; b.widths[1] = 32; b.widths[2] = 48; b.cin = 4; b.cout = 2; b.G = 8; b.attn_blocks = 1; b.attn_pe = 0;
    run("bottleneck attention (16,32,48 + 1 block)", b);
    if (bad) { printf("test_gn_ema: %d failed\n", bad); return 1; }
    printf("test_gn_ema: ok\n");
    return 0;
}
