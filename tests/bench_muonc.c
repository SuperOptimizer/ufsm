/* Optimizer step time of the production net (7 levels 16..256, cout 2, down_norm): Muon, Muon-C, Hyperball, both.
   usage: bench_muonc [steps] — ms per unet_muon call (the whole optimizer: the conv weights plus AdamW on the rest). */
#include "unet.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
int main(int argc, char **argv) {
    const int steps = argc > 1 ? atoi(argv[1]) : 20;
    if (nn_init(0)) return 1;
    unet_cfg cfg = {7, {16, 16, 32, 64, 128, 192, 256}, 4, 2, 8, 1};
    const char *name[4] = {"muon", "muon-c", "hyperball", "muon-c + hyperball"};
    for (int v = 0; v < 4; v++) {
        unet *u = unet_create(&cfg); unet_init(u, 1);
        unet_set_muon_ext(u, v & 1, v >> 1, 0.70710678f);
        const size_t n = unet_nparams(u); float *h = malloc(n * 4); unsigned s = 7;
        for (size_t i = 0; i < n; i++) { s = s * 1664525u + 1013904223u; h[i] = ((s >> 8) / 16777216.f - 0.5f) * 1e-3f; }
        unet_grad_h2d(u, h); free(h);
        for (int i = 0; i < 3; i++) unet_muon(u, 0.02f, 0.95f, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, i + 1);   /* tables, warm-up */
        nn_sync(); const double t0 = now();
        for (int i = 0; i < steps; i++) unet_muon(u, 0.02f, 0.95f, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, i + 4);
        nn_sync(); const double ms = (now() - t0) * 1e3 / steps;
        const char *e = nn_check();
        printf("  %-20s %7.2f ms per optimizer step%s%s\n", name[v], ms, e ? "  cuda: " : "", e ? e : "");
        unet_free(u);
    }
    return 0;
}
