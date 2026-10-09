/* Frozen GroupNorm statistics (unet_gn_freeze): an inference forward with the statistics of the same window reproduces the
   live forward bit for bit (16-bit and MX-fp4 storage); pooling one window gives back its statistics; another window size
   runs with the frozen statistics. */
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

static float *input(int P, unsigned seed) {
    const size_t n = (size_t)4 * P * P * P; float *h = malloc(n * 4);
    for (size_t i = 0; i < n; i++) h[i] = sinf((float)((i * 2654435761u + seed) % 9973) * 0.01f) + (float)((i * 2246822519u + seed) % 1000) / 1000.f;
    float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); free(h); return d;
}

int main(void) {
    nn_init(0); nn_set_tf32(1); nn_set_f16(1);
    for (int mx = 0; mx < 2; mx++) {
        unet_set_act_mx4(mx);
        unet_cfg cfg = {4, {32, 64, 96, 128}, 4, 2, 8, 1, 0};
        unet *u = unet_create(&cfg); unet_init(u, 3);
        const int P = 64; shape5 xs = {1, 4, P, P, P};
        const size_t no = (size_t)2 * P * P * P;
        float *x = input(P, 1), *a = malloc(no * 4), *b = malloc(no * 4);
        nn_d2h(a, unet_forward(u, x, xs, 0), no * 4);
        const size_t ns = unet_gn_nstats(u); float *st = malloc(ns * 4), *pool = malloc(ns * 4);
        CHECK(!unet_gn_stats_get(u, st), "stats_get");
        unet_gn_accumulate(u);
        CHECK(unet_gn_pooled(u, pool) == 1, "one pooled window");
        double md = 0; for (size_t i = 0; i < ns; i++) md = fmax(md, fabs((double)pool[i] - st[i]) / fmax(1e-3, fabs(st[i])));
        CHECK(md < 1e-4, "pooling one window changes its statistics (max rel %.3g)", md);
        unet_gn_freeze(u, st);
        nn_d2h(b, unet_forward(u, x, xs, 0), no * 4);
        CHECK(!memcmp(a, b, no * 4), "mx %d: frozen statistics of the same window do not reproduce the live forward", mx);
        const int P2 = 96; shape5 xs2 = {1, 4, P2, P2, P2}; float *x2 = input(P2, 2);
        const float *l2 = unet_forward(u, x2, xs2, 0); float v; nn_d2h(&v, l2, 4);
        CHECK(isfinite(v), "frozen statistics on another window size");
        const char *e = nn_check(); CHECK(!e, "cuda: %s", e ? e : "");
        unet_gn_freeze(u, nullptr);
        nn_free(x); nn_free(x2); free(a); free(b); free(st); free(pool); unet_free(u);
    }
    if (fails) return 1;
    printf("test_gn_frozen: ok\n");
    return 0;
}
