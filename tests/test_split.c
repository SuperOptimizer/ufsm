/* Spatial split (one window across two GPUs along z) against the same window on one GPU: loss and every parameter gradient.
   Modes: fp16 storage (exact up to summation order) and the --fp4 1 training preset (mx4 storage, fp4 forward / backward-data,
   fp8 weight gradient, stochastic rounding), whose split-vs-single difference is compared with the single-GPU run-to-run
   difference under two rounding seeds. Recompute 0 / 1 / 2, down_norm 0 / 1. Skipped (exit 0) with fewer than two GPUs.
   env: UFSM_P (96), UFSM_B (1). */
#include "split.h"
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NCH 2
static int P, B, Dl, H0;
static float *hx; static uint8_t *ht, *hm, *hw;   /* the whole window */

typedef struct { unet *u; float *x, *gl, *scratch; uint8_t *t, *m, *w; } side_t;

static void upload_side(side_t *d, int side) {   /* this side's planes [z0, z0 + Dl) of the window; halo planes masked out */
    const size_t p2 = (size_t)P * P, p3 = p2 * P, l3 = p2 * Dl;
    const int z0 = side ? P / 2 - H0 : 0;
    float *x = malloc((size_t)B * 4 * l3 * 4); uint8_t *t = malloc((size_t)B * NCH * l3), *m = malloc((size_t)B * l3);
    for (int r = 0; r < B * 4; r++) memcpy(x + r * l3, hx + r * p3 + z0 * p2, l3 * 4);
    for (int r = 0; r < B * NCH; r++) memcpy(t + r * l3, ht + r * p3 + z0 * p2, l3);
    for (int n = 0; n < B; n++) {
        memcpy(m + n * l3, hm + n * p3 + z0 * p2, l3);
        const int hz0 = side ? 0 : Dl - H0;
        memset(m + n * l3 + hz0 * p2, 0, H0 * p2);
    }
    nn_h2d(d->x, x, (size_t)B * 4 * l3 * 4); nn_h2d(d->t, t, (size_t)B * NCH * l3); nn_h2d(d->m, m, (size_t)B * l3); nn_h2d(d->w, hw, (size_t)B * NCH);
    free(x); free(t); free(m);
}
static void side_alloc(side_t *d, const unet_cfg *cfg, int depth) {
    const size_t n3 = (size_t)P * P * depth;
    d->u = unet_create(cfg); unet_init(d->u, 7);
    d->x = nn_malloc((size_t)B * 4 * n3 * 4); d->t = nn_malloc((size_t)B * NCH * n3); d->m = nn_malloc((size_t)B * n3); d->w = nn_malloc((size_t)B * NCH);
    d->gl = nn_malloc((size_t)B * NCH * n3 * 2); d->scratch = nn_malloc(nn_loss_scratch((shape5){B, NCH, depth, P, P}) + 64);
}
static void side_free(side_t *d) { unet_free(d->u); nn_free(d->x); nn_free(d->t); nn_free(d->m); nn_free(d->w); nn_free(d->gl); nn_free(d->scratch); }

static double step(side_t *d, int depth, int train) {   /* forward, loss, backward; returns nothing useful on its own */
    shape5 xs = {B, 4, depth, P, P};
    const float *lg = unet_forward(d->u, d->x, xs, train);
    shape5 os = unet_out_shape(d->u, xs);
    nn_loss_async(lg, d->t, d->m, d->w, os, 0.5f, train ? d->gl : nullptr, d->scratch);
    if (train) { unet_zero_grad(d->u); unet_backward_x(d->u, d->gl, 1); }
    return 0;
}
static double fetch(side_t *d, int depth) {
    float out[2 * NCH + 1]; nn_loss_fetch(d->scratch, (shape5){B, NCH, depth, P, P}, out);
    double l = 0; for (int c = 0; c < NCH; c++) l += out[c] + 0.5 * out[NCH + c];
    return l / NCH;
}

static side_t S2[2];
static void job(int side, void *arg) { (void)arg; step(&S2[side], Dl, 1); }

static double single(const unet_cfg *cfg, unsigned sr_step, float *g) {
    nn_init(0); nn_set_sr_step(sr_step);
    side_t d; side_alloc(&d, cfg, P);
    nn_h2d(d.x, hx, (size_t)B * 4 * P * P * P * 4); nn_h2d(d.t, ht, (size_t)B * NCH * P * P * P); nn_h2d(d.m, hm, (size_t)B * P * P * P); nn_h2d(d.w, hw, (size_t)B * NCH);
    step(&d, P, 1);
    double l = fetch(&d, P);
    unet_grad_d2h(d.u, g);
    side_free(&d);
    return l;
}
static double split(split_ctx *c, const unet_cfg *cfg, unsigned sr_step, float *g, size_t np) {
    for (int s = 0; s < 2; s++) { nn_init(s); nn_set_sr_step(sr_step); side_alloc(&S2[s], cfg, Dl); unet_set_split(S2[s].u, s, H0, split_halo); upload_side(&S2[s], s); }
    split_run(c, job, nullptr);
    double l[2]; float *g1 = malloc(np * 4);
    for (int s = 0; s < 2; s++) { nn_init(s); l[s] = fetch(&S2[s], Dl); unet_grad_d2h(S2[s].u, s ? g1 : g); }
    for (size_t i = 0; i < np; i++) g[i] += g1[i];
    if (l[0] != l[1]) printf("    (losses of the two sides differ: %.9g %.9g)\n", l[0], l[1]);
    for (int s = 0; s < 2; s++) { nn_init(s); side_free(&S2[s]); }
    free(g1);
    return l[0];
}

/* relative L2 difference overall and the worst parameter tensor */
typedef struct { const char *name; size_t off, len; } seg_t;
static int nseg; static seg_t segs[128]; static char names[128][24];
static size_t addseg(const char *nm, size_t off, size_t len) { snprintf(names[nseg], 24, "%s", nm); segs[nseg] = (seg_t){names[nseg], off, len}; nseg++; return off + len; }
static size_t addconv(const char *nm, int ci, int co, int k, size_t off) { char b[24]; snprintf(b, 24, "%s.w", nm); off = addseg(b, off, (size_t)co * ci * k * k * k); snprintf(b, 24, "%s.b", nm); return addseg(b, off, co); }
static size_t addgn(const char *nm, int c, size_t off) { char b[24]; snprintf(b, 24, "%s.gn", nm); return addseg(b, off, 2 * (size_t)c); }
static void segs_build(const unet_cfg *cfg) {
    const int *w = cfg->widths, L = cfg->nlev; size_t off = 0; char b[24]; nseg = 0;
    for (int i = 0; i < L; i++) { int ci = i ? w[i - 1] : cfg->cin; snprintf(b, 24, "enc%d.c1", i); off = addconv(b, ci, w[i], 3, off); snprintf(b, 24, "enc%d.n1", i); off = addgn(b, w[i], off); snprintf(b, 24, "enc%d.c2", i); off = addconv(b, w[i], w[i], 3, off); snprintf(b, 24, "enc%d.n2", i); off = addgn(b, w[i], off); }
    for (int i = 0; i < L - 1; i++) { snprintf(b, 24, "down%d", i); off = addconv(b, w[i], w[i], 3, off); }
    for (int i = L - 2; i >= 0; i--) { int ci = w[i] + w[i + 1]; snprintf(b, 24, "dec%d.c1", i); off = addconv(b, ci, w[i], 3, off); snprintf(b, 24, "dec%d.n1", i); off = addgn(b, w[i], off); snprintf(b, 24, "dec%d.c2", i); off = addconv(b, w[i], w[i], 3, off); snprintf(b, 24, "dec%d.n2", i); off = addgn(b, w[i], off); }
    off = addconv("head", w[0], cfg->cout, 1, off);
    if (cfg->down_norm) for (int i = 0; i < L - 1; i++) { snprintf(b, 24, "dn%d", i); off = addgn(b, w[i], off); }
}
static double rel(const float *a, const float *b, size_t off, size_t n) {
    double d2 = 0, r2 = 0;
    for (size_t i = off; i < off + n; i++) { double d = (double)a[i] - b[i]; d2 += d * d; r2 += (double)b[i] * b[i]; }
    return sqrt(d2 / (r2 + 1e-300));
}
static double worst(const float *a, const float *b, const char **nm) {
    double m = 0;
    for (int i = 0; i < nseg; i++) { double r = rel(a, b, segs[i].off, segs[i].len); if (r > m) { m = r; *nm = segs[i].name; } }
    return m;
}

int main(void) {
    if (nn_init(0) || nn_init(1)) { printf("test_split: needs two GPUs, skipped\n"); return 0; }
    P = getenv("UFSM_P") ? atoi(getenv("UFSM_P")) : 96; B = getenv("UFSM_B") ? atoi(getenv("UFSM_B")) : 1;
    unet_cfg cfg = {4, {16, 32, 64, 80}, 4, NCH, 8, 1};
    H0 = 1 << (cfg.nlev - 1); Dl = P / 2 + H0;
    if (P % (2 * H0)) { fprintf(stderr, "P must be a multiple of %d\n", 2 * H0); return 2; }
    const size_t p3 = (size_t)P * P * P;
    hx = malloc((size_t)B * 4 * p3 * 4); ht = malloc((size_t)B * NCH * p3); hm = malloc((size_t)B * p3); hw = malloc((size_t)B * NCH);
    srand(5);
    /* a smooth random field (sum of a few plane waves) so the network sees structure along z */
    for (int n = 0; n < B; n++) for (int c = 0; c < 4; c++) {
        double k[3][3], ph[3]; for (int j = 0; j < 3; j++) { for (int a = 0; a < 3; a++) k[j][a] = (rand() % 1000) / 1000.0 * 0.4; ph[j] = rand() % 628 / 100.0; }
        for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
            double v = 0; for (int j = 0; j < 3; j++) v += sin(k[j][0] * z + k[j][1] * y + k[j][2] * x + ph[j]);
            hx[(((size_t)n * 4 + c) * P + z) * P * P + (size_t)y * P + x] = (float)(v + 0.3 * ((rand() % 2001) / 1000.0 - 1));
        }
    }
    for (size_t i = 0; i < (size_t)B * NCH * p3; i++) { float v = hx[i % ((size_t)B * 4 * p3)]; ht[i] = v > 1.2f ? 255 : v > 0.8f ? 120 : 0; }
    for (size_t i = 0; i < (size_t)B * p3; i++) hm[i] = rand() % 10 != 0;
    for (int i = 0; i < B * NCH; i++) hw[i] = 1;

    nn_set_f16(1); nn_set_grad_scale(1024.f); nn_set_loss_grad_h16(1);
    split_ctx *ctx = split_create(0, 1);
    unet *probe = unet_create(&cfg); size_t np = unet_nparams(probe); unet_free(probe);
    float *g0 = malloc(np * 4), *g1 = malloc(np * 4), *g2 = malloc(np * 4);
    int fails = 0;
    printf("test_split: P %d B %d, level-0 halo %d planes, local depth %d\n", P, B, H0, Dl);
    for (int fp4 = 0; fp4 < 2; fp4++) {
        if (fp4) { unet_set_act_mx4(1); nn_set_sr(1); if (nn_set_prec_policy("all=fp4:fp4:fp8,enc0.c1=fp16")) return 2; }
        for (int dn = 0; dn < 2; dn++)
            for (int rc = 0; rc < 3; rc++) {
                cfg.down_norm = dn; segs_build(&cfg);
                { unet *pr = unet_create(&cfg); np = unet_nparams(pr); unet_free(pr); }
                unet_set_recompute(rc);
                double ls = single(&cfg, 1, g0), lp = split(ctx, &cfg, 1, g1, np);
                const char *wn = "", *wn2 = "";
                double dl = fabs(lp - ls) / fabs(ls), dg = rel(g1, g0, 0, np), wg = worst(g1, g0, &wn);
                if (!fp4) {
                    /* fp16 storage: identical math up to summation order (GroupNorm and loss sums over two halves) */
                    int ok = dl < 1e-4 && dg < 2e-3 && wg < 1e-2;
                    printf("  fp16 down_norm %d recompute %d: loss %.6f vs %.6f (rel %.1e), grad rel %.2e, worst %s %.2e  %s\n", dn, rc, lp, ls, dl, dg, wn, wg, ok ? "ok" : "FAIL");
                    fails += !ok;
                } else {
                    /* fp4: compare with the single-GPU difference between two rounding seeds */
                    double ln = single(&cfg, 2, g2);
                    double nl = fabs(ln - ls) / fabs(ls), ng = rel(g2, g0, 0, np), nw = worst(g2, g0, &wn2);
                    int ok = dl < 3 * nl + 1e-3 && dg < 1.5 * ng && wg < 1.5 * nw + 0.05;
                    printf("  fp4  down_norm %d recompute %d: loss %.6f vs %.6f (rel %.1e; seeds %.1e), grad rel %.3f (seeds %.3f), worst %s %.3f (seeds %s %.3f)  %s\n",
                           dn, rc, lp, ls, dl, nl, dg, ng, wn, wg, wn2, nw, ok ? "ok" : "FAIL");
                    fails += !ok;
                }
                const char *e = nn_check(); if (e) { printf("cuda error: %s\n", e); return 1; }
            }
    }
    split_free(ctx);
    printf("test_split: %s\n", fails ? "FAILED" : "passed");
    return fails ? 1 : 0;
}
