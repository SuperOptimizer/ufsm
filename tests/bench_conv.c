/* Microbenchmark of the conv3d kernels: GFLOP/s for the level-0 shapes of the 1.17M model. */
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static void bench(int cin, int cout, int P, int k, int stride) {
    shape5 xs = {1, cin, P, P, P};
    shape5 ys = nn_conv3d_out_shape(xs, cout, k, stride);
    size_t nw = (size_t)cout * cin * k * k * k;
    float *x = nn_malloc(shape_numel(xs) * 4), *y = nn_malloc(shape_numel(ys) * 4), *w = nn_malloc(nw * 4), *gw = nn_malloc(nw * 4), *gb = nn_malloc(cout * 4), *gx = nn_malloc(shape_numel(xs) * 4), *scr = nn_malloc(nn_conv3d_scratch(xs, cout, k));
    nn_zero(x, shape_numel(xs) * 4); nn_zero(w, nw * 4); nn_zero(y, shape_numel(ys) * 4);
    double flop = 2.0 * shape_spatial(ys) * cout * cin * k * k * k;
    /* correctness of the active path vs the fp32 kernel on random data */
    {
        float *hx = malloc(shape_numel(xs) * 4), *hw = malloc(nw * 4), *y1 = malloc(shape_numel(ys) * 4), *y2 = malloc(shape_numel(ys) * 4);
        for (size_t i = 0; i < shape_numel(xs); i++) hx[i] = (float)rand() / RAND_MAX - 0.5f;
        for (size_t i = 0; i < nw; i++) hw[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
        nn_h2d(x, hx, shape_numel(xs) * 4); nn_h2d(w, hw, nw * 4);
        int mode = nn_get_tf32();
        nn_conv3d_fwd(x, xs, w, nullptr, cout, k, stride, y); nn_d2h(y1, y, shape_numel(ys) * 4);
        nn_set_tf32(0); nn_conv3d_fwd(x, xs, w, nullptr, cout, k, stride, y); nn_d2h(y2, y, shape_numel(ys) * 4); nn_set_tf32(mode);
        double md = 0, mx = 0; for (size_t i = 0; i < shape_numel(ys); i++) { double d = fabs(y1[i] - y2[i]); if (d > md) md = d; if (fabs(y2[i]) > mx) mx = fabs(y2[i]); }
        if (mode) printf("  fwd tc vs fp32: max abs diff %.3g (max |y| %.3g)\n", md, mx);
        if (mode && k == 3) {
            float *g1 = malloc(nw * 4), *g2 = malloc(nw * 4);
            nn_h2d(y, y1, shape_numel(ys) * 4);   /* use the output as a fake gradient */
            float *b1 = malloc(cout * 4), *b2 = malloc(cout * 4);
            nn_zero(gw, nw * 4); nn_zero(gb, cout * 4); nn_conv3d_bwd_weight(x, xs, y, ys, k, stride, gw, gb); nn_d2h(g1, gw, nw * 4); nn_d2h(b1, gb, cout * 4);
            nn_set_tf32(0); nn_zero(gw, nw * 4); nn_zero(gb, cout * 4); nn_conv3d_bwd_weight(x, xs, y, ys, k, stride, gw, gb); nn_d2h(g2, gw, nw * 4); nn_d2h(b2, gb, cout * 4); nn_set_tf32(mode);
            double bd = 0, bm = 0; for (int c = 0; c < cout; c++) { if (fabs(b1[c] - b2[c]) > bd) bd = fabs(b1[c] - b2[c]); if (fabs(b2[c]) > bm) bm = fabs(b2[c]); }
            printf("  bias grad tc vs fp32: max abs diff %.3g (max |gb| %.3g)\n", bd, bm); free(b1); free(b2);
            double gd = 0, gm = 0; for (size_t i = 0; i < nw; i++) { double d = fabs(g1[i] - g2[i]); if (d > gd) gd = d; if (fabs(g2[i]) > gm) gm = fabs(g2[i]); }
            printf("  bwd_w tc vs fp32: max abs diff %.3g (max |gw| %.3g)\n", gd, gm);
            free(g1); free(g2);
        }
        free(hx); free(hw); free(y1); free(y2);
    }
    nn_conv3d_fwd(x, xs, w, nullptr, cout, k, stride, y); nn_sync();
    double t0 = now(); for (int i = 0; i < 5; i++) nn_conv3d_fwd(x, xs, w, nullptr, cout, k, stride, y); nn_sync(); double tf = (now() - t0) / 5;
    nn_conv3d_bwd_data(y, ys, w, xs, k, stride, gx, scr); nn_sync();
    t0 = now(); for (int i = 0; i < 5; i++) nn_conv3d_bwd_data(y, ys, w, xs, k, stride, gx, scr); nn_sync(); double td = (now() - t0) / 5;
    nn_conv3d_bwd_weight(x, xs, y, ys, k, stride, gw, gb); nn_sync();
    t0 = now(); for (int i = 0; i < 5; i++) nn_conv3d_bwd_weight(x, xs, y, ys, k, stride, gw, gb); nn_sync(); double tw = (now() - t0) / 5;
    printf("cin %3d cout %3d P %3d k %d s %d: fwd %6.1f ms %6.0f GF/s | bwd_data %6.1f ms %6.0f GF/s | bwd_w %6.1f ms %6.0f GF/s\n",
           cin, cout, P, k, stride, tf * 1e3, flop / tf / 1e9, td * 1e3, flop / td / 1e9, tw * 1e3, flop / tw / 1e9);
    nn_free(x); nn_free(y); nn_free(w); nn_free(gw); nn_free(gb); nn_free(gx); nn_free(scr);
}
int main(void) {
    nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0);
    if (getenv("UFSM_FP32")) nn_set_tf32(0);
    nn_set_act_bf16(0); nn_set_grad_bf16(0);   /* the harness feeds fp32 activations */
    printf("tf32 %d\n", nn_get_tf32());
    int L[][5] = {{4, 16, 96, 3, 1}, {16, 16, 96, 3, 1}, {48, 16, 96, 3, 1}, {16, 16, 96, 3, 2}, {96, 32, 48, 3, 1}, {144, 64, 24, 3, 1}, {80, 80, 12, 3, 1}};
    int only = getenv("UFSM_LAYER") ? atoi(getenv("UFSM_LAYER")) : -1;   /* UFSM_LAYER=i benchmarks one layer (profiling) */
    for (int i = 0; i < 7; i++) if (only < 0 || only == i) bench(L[i][0], L[i][1], L[i][2], L[i][3], L[i][4]);
    const char *e = nn_check(); if (e) printf("cuda: %s\n", e);
    return 0;
}
