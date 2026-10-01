/* Per-layer precision sweep for the (16,32,64,80) U-Net: for every conv layer and every low precision, lower that
   layer alone (rest bf16) and measure (a) the whole-network parameter-gradient error against the exact fp32 kernels
   (same metric as test_unet: random input, random logit gradient) and (b) the layer's own kernel time (event profiler,
   fwd + bwd_data + bwd_w). Then greedily build the fastest policies whose predicted error (errors add in quadrature)
   stays under each budget, and measure them for real (error + train-step time).
   env: UFSM_P (96), UFSM_B (2), UFSM_SWEEP_PRECS ("234"), UFSM_BUDGETS ("0.05,0.10"), UFSM_POLICIES (extra policies to
   measure, separated by '|'), UFSM_SWEEP_SKIP=1 (only measure UFSM_POLICIES). */
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rs = 12345;
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) * 0x1.0p-53 * 2 - 1); }
/* conv slots (unet_prof_layers order: slot = 2 * layer + conv); "" = no such conv */
#define UNET_NLID UNET_NSLOT
static const char *lname[UNET_NSLOT] = {"enc0.c1", "enc0.c2", "enc1.c1", "enc1.c2", "enc2.c1", "enc2.c2", "enc3.c1", "enc3.c2",
                                        "down0", "", "down1", "", "down2", "", "dec2.c1", "dec2.c2", "dec1.c1", "dec1.c2", "dec0.c1", "dec0.c2", "head", ""};
#define UNET_LID_HEAD 20
static unet *net; static float *dx, *dg, *g0, *g1; static shape5 xs; static size_t np;
static double grad_err(void) {
    unet_forward(net, dx, xs, 1); unet_zero_grad(net); unet_backward(net, dg); unet_grad_d2h(net, g1);
    double d2 = 0, n2 = 0;
    for (size_t i = 0; i < np; i++) { double d = (double)g1[i] - g0[i]; d2 += d * d; n2 += (double)g0[i] * g0[i]; }
    return sqrt(d2 / n2);
}
static double step_ms(int iters) {   /* train step without the optimizer (fwd + bwd), best of 3 */
    double best = 1e9;
    for (int r = 0; r < 3; r++) {
        nn_sync(); double t0 = now();
        for (int i = 0; i < iters; i++) { unet_forward(net, dx, xs, 1); unet_zero_grad(net); unet_backward(net, dg); }
        nn_sync(); double t = (now() - t0) / iters * 1e3; if (t < best) best = t;
    }
    return best;
}
static void layer_ms(double out[UNET_NLID]) {   /* per-layer conv kernel time (fwd + bwd_data + bwd_w), mean of 5 steps */
    double lm[UNET_NLID][3];
    unet_prof_layers_on(1);
    unet_forward(net, dx, xs, 1); unet_zero_grad(net); unet_backward(net, dg); unet_prof_layers(lm);
    for (int i = 0; i < 5; i++) { unet_forward(net, dx, xs, 1); unet_zero_grad(net); unet_backward(net, dg); }
    unet_prof_layers(lm);
    unet_prof_layers_on(0);
    for (int l = 0; l < UNET_NLID; l++) out[l] = (lm[l][0] + lm[l][1] + lm[l][2]) / 5;
}
static double measure_policy(const char *pol, const char *label) {
    if (nn_set_prec_policy(pol)) { printf("bad policy %s\n", pol); return 1e9; }
    double e = grad_err(), t = step_ms(10);
    printf("%-10s err %6.2f%%  fwd+bwd %6.2f ms  policy: %s\n", label, e * 100, t, pol);
    fflush(stdout);
    return e;
}
int main(void) {
    if (nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0)) return 1;
    int P = getenv("UFSM_P") ? atoi(getenv("UFSM_P")) : 96, B = getenv("UFSM_B") ? atoi(getenv("UFSM_B")) : 2;
    unet_cfg cfg = {4, {16, 32, 64, 80}, 4, 1, 8};
    net = unet_create(&cfg); unet_init(net, 2); np = unet_nparams(net);
    xs = (shape5){B, 4, P, P, P};
    size_t nx = shape_numel(xs), nl = (size_t)B * P * P * P;
    float *h = malloc(nx * 4);
    for (size_t i = 0; i < nx; i++) h[i] = frand();
    dx = nn_malloc(nx * 4); nn_h2d(dx, h, nx * 4);
    for (size_t i = 0; i < nl; i++) h[i] = frand();
    dg = nn_malloc(nl * 4); nn_h2d(dg, h, nl * 4); free(h);
    g0 = malloc(np * 4); g1 = malloc(np * 4);
    nn_set_tf32(0); unet_forward(net, dx, xs, 1); unet_zero_grad(net); unet_backward(net, dg); unet_grad_d2h(net, g0);
    nn_set_prec(1); nn_set_prec_policy("");
    double ebase = grad_err(), tbase = step_ms(10), lbase[UNET_NLID];
    layer_ms(lbase);
    printf("P %d batch %d; bf16 everywhere: grad err %.2f%%, fwd+bwd %.2f ms\n", P, B, ebase * 100, tbase);
    if (getenv("UFSM_SWEEP_SHOW")) {   /* per-layer per-pass kernel times and TF/s for each policy in UFSM_SWEEP_SHOW ('|'-separated; "" = bf16) */
        static const int lc[UNET_NSLOT][3] = {{4,16,96},{16,16,96},{16,32,48},{32,32,48},{32,64,24},{64,64,24},{64,80,12},{80,80,12},
                                              {16,16,48},{0,0,0},{32,32,24},{0,0,0},{64,64,12},{0,0,0},{144,64,24},{64,64,24},{96,32,48},{32,32,48},{48,16,96},{16,16,96},{16,1,96},{0,0,0}};
        char *sh = strdup(getenv("UFSM_SWEEP_SHOW")), *sv = nullptr;
        for (char *pol = strtok_r(sh, "|", &sv); pol; pol = strtok_r(nullptr, "|", &sv)) {
            if (!strcmp(pol, "-")) pol = (char *)"";
            nn_set_prec_policy(pol);
            double lm[UNET_NLID][3];
            unet_prof_layers_on(1);
            unet_forward(net, dx, xs, 1); unet_zero_grad(net); unet_backward(net, dg); unet_prof_layers(lm);
            for (int i = 0; i < 5; i++) { unet_forward(net, dx, xs, 1); unet_zero_grad(net); unet_backward(net, dg); }
            unet_prof_layers(lm);
            unet_prof_layers_on(0);
            printf("policy '%s' (ms per step, TF/s; stride-2 flops at the output grid)\n", pol);
            double tot[3] = {0};
            for (int l = 0; l < UNET_NLID; l++) {
                if (!*lname[l]) continue;
                double sp = (double)B * lc[l][2] * lc[l][2] * lc[l][2] * (P / 96.0) * (P / 96.0) * (P / 96.0);
                double fl = 2.0 * sp * lc[l][0] * lc[l][1] * (l == UNET_LID_HEAD ? 1 : 27);
                printf("  %-8s", lname[l]);
                for (int j = 0; j < 3; j++) { double ms = lm[l][j] / 5; tot[j] += ms; printf("  %7.3f ms %5.1f TF", ms, ms > 0 ? fl / ms / 1e9 : 0); }
                printf("\n");
            }
            printf("  total     %7.3f ms           %7.3f ms           %7.3f ms\n", tot[0], tot[1], tot[2]);
        }
        free(sh);
        nn_set_prec_policy("");
    }
    if (getenv("UFSM_POLICIES")) {
        char *s = strdup(getenv("UFSM_POLICIES")), *save = nullptr;
        for (char *t = strtok_r(s, "|", &save); t; t = strtok_r(nullptr, "|", &save)) measure_policy(t, "given");
        free(s);
    }
    if (getenv("UFSM_SWEEP_SKIP")) return 0;
    /* per (layer, pass) candidates: pass 0 forward, 1 backward-data, 2 weight gradient; precisions from UFSM_SWEEP_PRECS
       (default "42": fp16, fp8; '3' adds fp4 for the forward / backward-data) */
    const char *precs = getenv("UFSM_SWEEP_PRECS") ? getenv("UFSM_SWEEP_PRECS") : "42";
    static const char *pn[3] = {"fwd", "bwd_data", "wgrad"};
    int np_ = (int)strlen(precs), pl[8];
    for (int k = 0; k < np_; k++) pl[k] = precs[k] - '0';
    double lb3[UNET_NLID][3];
    {   /* bf16 per-pass baseline */
        double lm[UNET_NLID][3];
        unet_prof_layers_on(1);
        unet_forward(net, dx, xs, 1); unet_zero_grad(net); unet_backward(net, dg); unet_prof_layers(lm);
        for (int i = 0; i < 5; i++) { unet_forward(net, dx, xs, 1); unet_zero_grad(net); unet_backward(net, dg); }
        unet_prof_layers(lm);
        unet_prof_layers_on(0);
        for (int l = 0; l < UNET_NLID; l++) for (int j = 0; j < 3; j++) lb3[l][j] = lm[l][j] / 5;
    }
    static double err[UNET_NLID][3][8], ms[UNET_NLID][3][8];
    static int valid[UNET_NLID][3][8];
    printf("%-8s %-8s %8s", "layer", "pass", "bf16 ms");
    for (int k = 0; k < np_; k++) printf(" | %4s ms  err%%", nn_prec_name(pl[k]));
    printf("\n");
    for (int l = 0; l < UNET_NLID; l++) for (int j = 0; j < 3; j++) {
        if (!*lname[l] || l == UNET_LID_HEAD) continue;
        printf("%-8s %-8s %8.3f", lname[l], pn[j], lb3[l][j]);
        for (int k = 0; k < np_; k++) {
            if ((pl[k] == 3 && j == 2) || (pl[k] == 4 && j == 2 && !getenv("UFSM_F16_WGRAD"))) { printf(" |      -      -"); continue; }
            const char *b1 = nn_prec_name(1), *pp = nn_prec_name(pl[k]);
            char pol[96]; snprintf(pol, sizeof pol, "%s=%s:%s:%s", lname[l], j == 0 ? pp : b1, j == 1 ? pp : b1, j == 2 ? pp : b1);
            nn_set_prec_policy(pol);
            err[l][j][k] = grad_err();
            double lm[UNET_NLID][3];
            unet_prof_layers_on(1);
            unet_forward(net, dx, xs, 1); unet_zero_grad(net); unet_backward(net, dg); unet_prof_layers(lm);
            for (int i = 0; i < 5; i++) { unet_forward(net, dx, xs, 1); unet_zero_grad(net); unet_backward(net, dg); }
            unet_prof_layers(lm);
            unet_prof_layers_on(0);
            ms[l][j][k] = lm[l][j] / 5; valid[l][j][k] = 1;
            printf(" | %7.3f %5.2f", ms[l][j][k], err[l][j][k] * 100);
            fflush(stdout);
        }
        printf("\n");
    }
    nn_set_prec_policy("");
    /* greedy over slots: saving dt = current - candidate time, cost = increase of err^2 over the bf16 baseline */
    const char *bud = getenv("UFSM_BUDGETS") ? getenv("UFSM_BUDGETS") : "0.05,0.10";
    char *bs = strdup(bud), *save = nullptr;
    for (char *tok = strtok_r(bs, ",", &save); tok; tok = strtok_r(nullptr, ",", &save)) {
      const double target = atof(tok);
      double budget = target;
      for (int attempt = 0; attempt < 8; attempt++, budget *= 0.96) {   /* tighten the predicted budget until the measured error fits */
        int choice[UNET_NLID][3];
        for (int l = 0; l < UNET_NLID; l++) for (int j = 0; j < 3; j++) choice[l][j] = -1;
        double e2 = ebase * ebase;
        for (;;) {
            int bl = -1, bj = -1, bk = -1; double best = 0;
            for (int l = 0; l < UNET_NLID; l++) for (int j = 0; j < 3; j++) for (int k = 0; k < np_; k++) {
                if (!valid[l][j][k]) continue;
                int c = choice[l][j];
                double cur_t = c < 0 ? lb3[l][j] : ms[l][j][c], cur_e = c < 0 ? 0 : err[l][j][c] * err[l][j][c] - ebase * ebase;
                double dt = cur_t - ms[l][j][k], de = err[l][j][k] * err[l][j][k] - ebase * ebase - cur_e;
                if (dt <= 0.003) continue;
                if (de < 1e-7) de = 1e-7;
                if (e2 + de > budget * budget) continue;
                if (dt / de > best) { best = dt / de; bl = l; bj = j; bk = k; }
            }
            if (bl < 0) break;
            int c = choice[bl][bj];
            double cur_e = c < 0 ? 0 : err[bl][bj][c] * err[bl][bj][c] - ebase * ebase;
            e2 += err[bl][bj][bk] * err[bl][bj][bk] - ebase * ebase - cur_e;
            choice[bl][bj] = bk;
        }
        char pol[2048]; size_t o = 0; pol[0] = 0;
        for (int l = 0; l < UNET_NLID; l++) {
            if (choice[l][0] < 0 && choice[l][1] < 0 && choice[l][2] < 0) continue;
            const char *q[3];
            for (int j = 0; j < 3; j++) q[j] = nn_prec_name(choice[l][j] < 0 ? 1 : pl[choice[l][j]]);
            if (!strcmp(q[0], q[1]) && !strcmp(q[1], q[2])) o += snprintf(pol + o, sizeof pol - o, "%s%s=%s", o ? "," : "", lname[l], q[0]);
            else o += snprintf(pol + o, sizeof pol - o, "%s%s=%s:%s:%s", o ? "," : "", lname[l], q[0], q[1], q[2]);
        }
        printf("budget %.1f%% (target %.0f%%): predicted err %.2f%%\n", budget * 100, target * 100, sqrt(e2) * 100);
        char label[32]; snprintf(label, sizeof label, "<%.0f%%", target * 100);
        double me = measure_policy(pol, label);
        if (me <= target) break;
      }
    }
    free(bs);
    measure_policy("all=fp8", "all fp8");
    measure_policy("all=fp4", "all fp4");
    measure_policy("all=fp16", "all fp16");
    measure_policy("all=fp16:fp16:fp8", "fp16+f8w");
    const char *e = nn_check(); if (e) printf("cuda: %s\n", e);
    return 0;
}
