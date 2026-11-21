/* bench_isfhead [HP] [CIN] [REPS]: the --ink-surf 2-D head's step cost (inkperf 2026-11-21): per conv forward / data / weight gradient,
   the whole forward + BCE gradient + backward + AdamW, and the host patch AUC (isf_dev_step's insertion sort) on a 19 % labelled patch.
   Defaults: HP 256, CIN 288 (32 trunk channels x 9 layers: S1 / S2), REPS 5. */
#include "inksurf.h"
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rs = 7;
static float frand(void) { rs = rs * 6364136223846793005ull + 1442695040888963407ull; return (float)(((rs >> 11) + 0.5) / 9007199254740992.0 * 2 - 1); }
int main(int argc, char **argv) {
    const int Hp = argc > 1 ? atoi(argv[1]) : 256, cin = argc > 2 ? atoi(argv[2]) : 288, reps = argc > 3 ? atoi(argv[3]) : 5;
    nn_init(0);
    isfhead *h = isfhead_create(cin, Hp, Hp, 1); if (!h) { fprintf(stderr, "bench_isfhead: no head\n"); return 1; }
    const size_t n = (size_t)Hp * Hp, nx = (size_t)cin * n;
    float *hx = malloc(nx * 4), *hl = malloc(n * 4), *hw = malloc(n * 4);
    for (size_t i = 0; i < nx; i++) hx[i] = frand();
    for (size_t i = 0; i < n; i++) { hl[i] = frand() > 0; hw[i] = frand() < -0.62f; }   /* (~19 % labelled, as S1's "patch 19 % unmasked") */
    float *x = nn_malloc(nx * 4), *lab = nn_malloc(n * 4), *wt = nn_malloc(n * 4), *g = nn_malloc(n * 4);
    nn_h2d(x, hx, nx * 4); nn_h2d(lab, hl, n * 4); nn_h2d(wt, hw, n * 4);
    {   /* the head's 13 conv shapes: tiled vs _ref, bits and time */
        static const int sh[13][4] = {{288, 32, 1, 0}, {32, 32, 3, 0}, {32, 48, 3, 1}, {48, 64, 3, 2}, {64, 96, 3, 3}, {96, 96, 3, 4}, {96, 96, 3, 4}, {192, 64, 3, 3},
                                      {128, 48, 3, 2}, {96, 32, 3, 1}, {64, 32, 3, 0}, {32, 1, 1, 0}, {288, 1, 1, 0}};
        double tn[3] = {0, 0, 0}, to[3] = {0, 0, 0}; int bad = 0;
        for (int s = 0; s < 13; s++) {
            const int Ci = sh[s][0], Co = sh[s][1], k = sh[s][2], H = Hp >> sh[s][3]; const size_t hw = (size_t)H * H, nw = (size_t)Co * Ci * k * k;
            float *a = nn_malloc(Ci * hw * 4), *gy = nn_malloc(Co * hw * 4), *w = nn_malloc(nw * 4), *b = nn_malloc(Co * 4), *y0 = nn_malloc(Co * hw * 4), *y1 = nn_malloc(Co * hw * 4),
                  *g0 = nn_malloc(Ci * hw * 4), *g1 = nn_malloc(Ci * hw * 4), *w0 = nn_malloc(nw * 4), *w1 = nn_malloc(nw * 4), *b0 = nn_malloc(Co * 4), *b1 = nn_malloc(Co * 4);
            const size_t mx = (Ci > Co ? Ci : Co) * hw > nw ? (Ci > Co ? Ci : Co) * hw : nw; float *hb = malloc(mx * 4), *hc = malloc(mx * 4);
            for (size_t i = 0; i < Ci * hw; i++) hb[i] = frand(); nn_h2d(a, hb, Ci * hw * 4);
            for (size_t i = 0; i < Co * hw; i++) hb[i] = frand(); nn_h2d(gy, hb, Co * hw * 4);
            for (size_t i = 0; i < nw; i++) hb[i] = 0.1f * frand(); nn_h2d(w, hb, nw * 4);
            for (int i = 0; i < Co; i++) hb[i] = frand(); nn_h2d(b, hb, Co * 4);
            double t;
            for (int r = 0; r < 3; r++) {
                nn_sync(); t = now_s(); nn_isf2_conv_ref(a, Ci, H, H, w, b, Co, k, y0); nn_sync(); if (r) to[0] += now_s() - t;
                t = now_s(); nn_isf2_conv(a, Ci, H, H, w, b, Co, k, y1); nn_sync(); if (r) tn[0] += now_s() - t;
                t = now_s(); nn_isf2_conv_bd_ref(gy, Co, H, H, w, Ci, k, g0, 0); nn_sync(); if (r) to[1] += now_s() - t;
                t = now_s(); nn_isf2_conv_bd(gy, Co, H, H, w, Ci, k, g1, 0); nn_sync(); if (r) tn[1] += now_s() - t;
                t = now_s(); nn_isf2_conv_bw_ref(a, Ci, H, H, gy, Co, k, w0, b0, 0); nn_sync(); if (r) to[2] += now_s() - t;
                t = now_s(); nn_isf2_conv_bw(a, Ci, H, H, gy, Co, k, w1, b1, 0); nn_sync(); if (r) tn[2] += now_s() - t;
            }
            const float *pa[3][2] = {{y0, y1}, {g0, g1}, {w0, w1}}; const size_t pn[3] = {Co * hw, Ci * hw, nw};
            for (int q = 0; q < 3; q++) { nn_d2h(hb, pa[q][0], pn[q] * 4); nn_d2h(hc, pa[q][1], pn[q] * 4);
                if (memcmp(hb, hc, pn[q] * 4)) { size_t d = 0; for (size_t i = 0; i < pn[q]; i++) d += memcmp(hb + i, hc + i, 4) != 0;
                    printf("bench_isfhead: conv %d (%d -> %d, k %d, %d^2) %s: %zu of %zu values differ from _ref\n", s, Ci, Co, k, H, q ? q == 1 ? "data grad" : "weight grad" : "forward", d, pn[q]); bad++; } }
            nn_free(a); nn_free(gy); nn_free(w); nn_free(b); nn_free(y0); nn_free(y1); nn_free(g0); nn_free(g1); nn_free(w0); nn_free(w1); nn_free(b0); nn_free(b1); free(hb); free(hc);
        }
        printf("bench_isfhead: 13 convs x 2: forward %.1f -> %.1f ms, data grad %.1f -> %.1f ms, weight grad %.1f -> %.1f ms; %s\n", 500 * to[0], 500 * tn[0], 500 * to[1], 500 * tn[1],
               500 * to[2], 500 * tn[2], bad ? "BIT DIFFERENCES" : "bit-identical to _ref");
    }
    double tf = 0, tb = 0, ta = 0;
    for (int r = 0; r <= reps; r++) {
        nn_sync(); double t0 = now_s();
        const float *z = isfhead_forward(h, x); nn_sync(); double t1 = now_s();
        isfhead_zero_grad(h); nn_isf2_bce_grad(z, lab, wt, n, 1e-4f, g); isfhead_backward(h, g); nn_sync(); double t2 = now_s();
        isfhead_adamw(h, 1e-3f, 0, r + 1); nn_sync(); double t3 = now_s();
        if (r) { tf += t1 - t0; tb += t2 - t1; ta += t3 - t2; }
    }
    printf("bench_isfhead: Hp %d cin %d: forward %.1f ms, bce + backward %.1f ms, adamw %.2f ms (mean of %d)\n", Hp, cin, 1e3 * tf / reps, 1e3 * tb / reps, 1e3 * ta / reps, reps);
    /* the host AUC as isf_dev_step computes it */
    size_t m = 0; for (size_t p = 0; p < n; p++) m += hw[p] > 0;
    float (*o)[2] = malloc(m * sizeof *o); size_t k = 0; for (size_t p = 0; p < n; p++) if (hw[p] > 0) { o[k][0] = hx[p]; o[k][1] = hl[p]; k++; }
    double t0 = now_s();
    for (size_t i = 1; i < m; i++) { float t0v = o[i][0], t1v = o[i][1]; size_t j = i; while (j > 0 && o[j - 1][0] > t0v) { o[j][0] = o[j - 1][0]; o[j][1] = o[j - 1][1]; j--; } o[j][0] = t0v; o[j][1] = t1v; }
    printf("bench_isfhead: host AUC insertion sort over %zu labelled px: %.1f ms\n", m, 1e3 * (now_s() - t0));
    free(o); free(hx); free(hl); free(hw);
    return 0;
}
