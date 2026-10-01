/* Synthetic training run to compare conv precisions end to end: the (16,32,64,80) U-Net learns to mark a thin
   isosurface "sheet" of a random smooth field (a stand-in for the papyrus surface), BCE + soft Dice, AdamW.
   Every precision sees the same sample sequence; the held-out loss is always evaluated with the fp32 kernels.
   Usage: train_lp [steps]  env UFSM_PRECS (default "012"), UFSM_P (64), UFSM_B (2), UFSM_LR (2e-3). */
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rs;
static double ur(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * 0x1.0p-53; }
/* field f = sum of 6 random plane waves; x = [f + noise, 3 coordinate channels]; t = 255 * (|f - 0.3| < 0.12) */
static void make_sample(int P, float *x, uint8_t *t, uint64_t seed) {
    rs = seed * 0x9e3779b97f4a7c15ull + 1;
    double k[6][3], ph[6], a[6];
    for (int i = 0; i < 6; i++) { for (int j = 0; j < 3; j++) k[i][j] = (ur() * 2 - 1) * 0.25; ph[i] = ur() * 6.283; a[i] = 0.3 + 0.4 * ur(); }
    size_t S = (size_t)P * P * P;
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int xx = 0; xx < P; xx++) {
        size_t v = ((size_t)z * P + y) * P + xx;
        double f = 0; for (int i = 0; i < 6; i++) f += a[i] * sin(k[i][0] * z + k[i][1] * y + k[i][2] * xx + ph[i]);
        x[v] = (float)(f + (ur() - 0.5) * 0.3);
        static int mode = -1; if (mode < 0) mode = getenv("UFSM_INPUT") ? atoi(getenv("UFSM_INPUT")) : 0;
        if (mode == 4) { for (int c = 0; c < 4; c++) x[c * S + v] = (float)(ur() * 2 - 1); }   /* uniform noise in all channels */
        else if (mode == 3) { x[v] = (float)((ur() - 0.5) * 3.4); x[S + v] = x[2 * S + v] = x[3 * S + v] = 0.f; }   /* white-noise input */
        else if (mode == 1) { x[S + v] = x[2 * S + v] = x[3 * S + v] = 0.f; }                                  /* no coordinate channels */
        else if (mode == 2) { x[S + v] = 0.f; x[2 * S + v] = 0.6f; x[3 * S + v] = 0.8f; }                /* constant unit "radial" vector */
        else { x[S + v] = (float)z / P - 0.5f; x[2 * S + v] = (float)y / P - 0.5f; x[3 * S + v] = (float)xx / P - 0.5f; }
        t[v] = fabs(f - 0.3) < 0.12 ? 255 : 0;
    }
}
/* parameter segments in unet.c order (enc blocks, down convs, dec blocks, head) for per-tensor error reports */
typedef struct { char name[24]; size_t off, len; int kind; } seg_t;   /* kind 0 conv w, 1 conv b, 2 gn gamma, 3 gn beta */
static int nseg;
static seg_t segs[128];
static size_t addseg(const char *nm, int kind, size_t off, size_t len) { seg_t *g = &segs[nseg++]; snprintf(g->name, sizeof g->name, "%s", nm); g->off = off; g->len = len; g->kind = kind; return off + len; }
static size_t addconv(const char *nm, int ci, int co, int k, size_t off) { char b[24]; snprintf(b, 24, "%s.w", nm); off = addseg(b, 0, off, (size_t)co * ci * k * k * k); snprintf(b, 24, "%s.b", nm); return addseg(b, 1, off, co); }
static size_t addgn(const char *nm, int c, size_t off) { char b[24]; snprintf(b, 24, "%s.g", nm); off = addseg(b, 2, off, c); snprintf(b, 24, "%s.be", nm); return addseg(b, 3, off, c); }
static size_t addblock(const char *nm, int ci, int co, size_t off) {
    char b[24];
    snprintf(b, 24, "%s.c1", nm); off = addconv(b, ci, co, 3, off); snprintf(b, 24, "%s.n1", nm); off = addgn(b, co, off);
    snprintf(b, 24, "%s.c2", nm); off = addconv(b, co, co, 3, off); snprintf(b, 24, "%s.n2", nm); return addgn(b, co, off);
}
static void build_segs(const int *w, int L, int cin, int cout) {
    size_t off = 0; char b[24];
    for (int i = 0; i < L; i++) { snprintf(b, 24, "enc%d", i); off = addblock(b, i ? w[i - 1] : cin, w[i], off); }
    for (int i = 0; i < L - 1; i++) { snprintf(b, 24, "down%d", i); off = addconv(b, w[i], w[i], 3, off); }
    for (int i = L - 2; i >= 0; i--) { snprintf(b, 24, "dec%d", i); off = addblock(b, w[i] + w[i + 1], w[i], off); }
    addconv("head", w[0], cout, 1, off);
}
static void seg_report(const float *g0, const float *g1) {
    static const char *kn[4] = {"conv w", "conv b", "gn gamma", "gn beta"};
    double kd[4] = {0}, kr[4] = {0};
    printf("      worst tensors:");
    for (int i = 0; i < nseg; i++) {
        double d2 = 0, r2 = 0;
        for (size_t j = segs[i].off; j < segs[i].off + segs[i].len; j++) { double d = (double)g1[j] - g0[j]; d2 += d * d; r2 += (double)g0[j] * g0[j]; }
        kd[segs[i].kind] += d2; kr[segs[i].kind] += r2;
        if (sqrt(d2 / (r2 + 1e-300)) > 0.2) printf(" %s %.2g", segs[i].name, sqrt(d2 / (r2 + 1e-300)));
    }
    printf("\n      by kind:");
    for (int k = 0; k < 4; k++) printf(" %s rel %.3g (|g| %.3g)", kn[k], sqrt(kd[k] / (kr[k] + 1e-300)), sqrt(kr[k]));
    printf("\n");
}
int main(int argc, char **argv) {
    if (nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0)) return 1;
    int steps = argc > 1 ? atoi(argv[1]) : 300;
    int P = getenv("UFSM_P") ? atoi(getenv("UFSM_P")) : 64, B = getenv("UFSM_B") ? atoi(getenv("UFSM_B")) : 2;
    float lr = getenv("UFSM_LR") ? (float)atof(getenv("UFSM_LR")) : 2e-3f;
    const char *precs = getenv("UFSM_PRECS") ? getenv("UFSM_PRECS") : "012";
    size_t S = (size_t)P * P * P;
    shape5 xs = {B, 4, P, P, P}, ls = {B, 1, P, P, P};
    float *hx = malloc(xs.n * 4 * S * 4); uint8_t *ht = malloc(B * S), *hm = malloc(B * S), hw[64];
    memset(hm, 1, B * S); memset(hw, 1, sizeof hw);
    float *dx = nn_malloc(B * 4 * S * 4), *gl = nn_malloc(B * S * 4), *scr = nn_malloc(nn_loss_scratch(ls));
    uint8_t *dt = nn_malloc(B * S), *dm = nn_malloc(B * S), *dw = nn_malloc(64);
    nn_h2d(dm, hm, B * S); nn_h2d(dw, hw, 64);
    /* held-out set: 4 batches with seeds far from the training ones */
    enum { NH = 4 };
    float *hxh[NH]; uint8_t *hth[NH];
    for (int h = 0; h < NH; h++) { hxh[h] = malloc(B * 4 * S * 4); hth[h] = malloc(B * S); for (int b = 0; b < B; b++) make_sample(P, hxh[h] + (size_t)b * 4 * S, hth[h] + (size_t)b * S, 1000000 + h * B + b); }
    printf("synthetic sheet task, P %d, batch %d, %d steps, lr %g\n", P, B, steps, lr);
    for (const char *pc = precs; *pc; pc++) {
        int prec = *pc - '0';
        unet_cfg cfg = {4, {16, 32, 64, 80}, 4, 1, 8};
        unet *u = unet_create(&cfg);
        if (!nseg) build_segs(cfg.widths, cfg.nlev, cfg.cin, cfg.cout);
        unet_init(u, getenv("UFSM_SEED") ? atoi(getenv("UFSM_SEED")) : 7);
        /* unet.c sizes some activation buffers by the precision mode at first build: build them in fp32 mode
           (the largest layout) so that switching to fp32 for the held-out evaluation is safe */
        nn_set_prec(0); unet_forward(u, dx, xs, 1);
        double t0 = now(), run = 0; int nrun = 0;
        printf("prec %d:", prec); fflush(stdout);
        for (int it = 1; it <= steps; it++) {
            for (int b = 0; b < B; b++) make_sample(P, hx + (size_t)b * 4 * S, ht + (size_t)b * S, (uint64_t)it * B + b);
            nn_h2d(dx, hx, B * 4 * S * 4); nn_h2d(dt, ht, B * S);
            nn_set_prec(prec);
            if (getenv("UFSM_QAT")) nn_set_prec_wgrad(1);
            if (getenv("UFSM_SPARSE24") && it >= atoi(getenv("UFSM_SPARSE24")) && !unet_get_sparse24(u)) { unet_set_sparse24(u, 1); printf("  step %d: 2:4 sparsity on\n", it); }
            const float *lg = unet_forward(u, dx, xs, 1);
            float out[3];
            nn_loss(lg, dt, dm, dw, ls, 1.f, gl, out, scr);
            unet_zero_grad(u); unet_backward(u, gl);
            unet_clip_grad(u, 1.0);
            unet_srste24(u, 2e-4f);
            float lrt = it < 30 ? lr * it / 30 : lr * 0.5f * (1 + cosf(3.14159f * (it - 30) / (steps - 30)));
            unet_adamw(u, lrt, 0.9f, 0.999f, 1e-8f, 1e-4f, it);
            if (getenv("UFSM_WQ")) { if (!unet_get_wq(u)) unet_set_wq(u, atoi(getenv("UFSM_WQ"))); unet_wquant(u, (unsigned)it); }
            run += out[0] + out[1]; nrun++;
            if (it % 50 == 0) { printf(" %d:%.4f", it, run / nrun); fflush(stdout); run = 0; nrun = 0; }
            if (getenv("UFSM_GCHECK") && (it % 100 == 0 || it == 1)) {   /* gradient agreement of bf16 / fp8 with fp32 at the current weights */
                size_t np = unet_nparams(u);
                float *g0 = malloc(np * 4), *g1 = malloc(np * 4);
                printf("\n    step %d grad vs fp32:", it);
                for (int q = 0; q <= 2; q++) {
                    nn_set_prec(q);
                    float o2[3];
                    nn_loss(unet_forward(u, dx, xs, 1), dt, dm, dw, ls, getenv("UFSM_DICEW") ? (float)atof(getenv("UFSM_DICEW")) : 1.f, gl, o2, scr);
                    unet_zero_grad(u); unet_backward(u, gl); unet_grad_d2h(u, q ? g1 : g0);
                    if (q) {
                        double d2 = 0, r2 = 0, dot = 0, n1 = 0;
                        for (size_t i = 0; i < np; i++) { double d = (double)g1[i] - g0[i]; d2 += d * d; r2 += (double)g0[i] * g0[i]; dot += (double)g0[i] * g1[i]; n1 += (double)g1[i] * g1[i]; }
                        printf("  prec %d rel %.3g cos %.5f loss %.5f\n", q, sqrt(d2 / r2), dot / sqrt(r2 * n1), o2[0] + o2[1]);
                        seg_report(g0, g1);
                    } else printf(" loss %.5f", o2[0] + o2[1]);
                }
                printf("\n"); fflush(stdout);
                free(g0); free(g1); unet_zero_grad(u);
            }
        }
        nn_sync();
        double dt_s = now() - t0;
        /* held-out evaluation with the exact fp32 kernels */
        nn_set_prec(0);
        double hl = 0, hb = 0, hd = 0;
        for (int h = 0; h < NH; h++) {
            nn_h2d(dx, hxh[h], B * 4 * S * 4); nn_h2d(dt, hth[h], B * S);
            float out[3];
            nn_loss(unet_forward(u, dx, xs, 0), dt, dm, dw, ls, 1.f, gl, out, scr);
            hb += out[0] / NH; hd += out[1] / NH; hl += (out[0] + out[1]) / NH;
        }
        const char *e = nn_check();
        printf("\n  prec %d: held-out loss %.4f (bce %.4f, dice %.4f), %.1f s%s%s\n", prec, hl, hb, hd, dt_s, e ? " cuda: " : "", e ? e : "");
        unet_free(u);
    }
    return 0;
}
