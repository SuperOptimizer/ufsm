/* Optimizer step time of the from-scratch 92 M net (6 levels 256,320,384,448,288,288, 4 attention blocks): Muon fp32 / fp16 tensor-core
   Newton-Schulz, and each variant of DESIGN "Muon variants" (2026-11-13). usage: bench_muonx [steps] — ms per unet_muon call (the
   Muon matrices plus AdamW on the rest; the Kronecker preconditioner's roots refresh round robin every 10 steps, so steps >= 20). */
#include "unet.h"
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
int main(int argc, char **argv) {
    const int steps = argc > 1 ? atoi(argv[1]) : 20;
    if (nn_init(0)) return 1;
    unet_cfg cfg = {6, {256, 320, 384, 448, 288, 288}, 4, 2, 8, 1}; cfg.attn_blocks = 4; cfg.attn_pe = 3;
    enum { NV = 11 };
    const char *name[NV] = {"muon (fp32 NS)", "muon --muon-tc 1", "--muon-ns polar", "--normuon 1", "--muon-vr 0.025", "--muon-kron 1", "--muon-c 1",
                            "--muon-cautious 1", "--muon-ademamix 5", "kron + tc", "muon --muon-tc 2 (TF32)"};
    for (int v = 0; v < NV; v++) {
        unet *u = unet_create(&cfg); unet_init(u, 1);
        nn_muon_tc(v == 10 ? 2 : v == 1 || v == 9);
        unet_set_muon_ext(u, v == 6, 0, 0.70710678f);
        unet_set_muon_variant(u, v == 2, v == 3, 0.95f);
        unet_set_muon_vr(u, v == 4 ? 0.025f : 0.f);
        unet_set_muon_kron(u, v == 5 || v == 9, 0.99f, 1e-3f, 10);
        unet_set_muon_cautious(u, v == 7);
        unet_set_muon_ademamix(u, v == 8 ? 5.f : 0.f, 0.999f, 1000);
        const size_t n = unet_nparams(u); float *h = malloc(n * 4); unsigned s = 7;
        for (size_t i = 0; i < n; i++) { s = s * 1664525u + 1013904223u; h[i] = ((s >> 8) / 16777216.f - 0.5f) * 1e-3f; }
        unet_grad_h2d(u, h); free(h);
        for (int i = 0; i < 3; i++) unet_muon(u, 0.02f, 0.95f, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, i + 1);   /* tables, warm-up */
        nn_sync(); const double t0 = now();
        for (int i = 0; i < steps; i++) unet_muon(u, 0.02f, 0.95f, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.01f, i + 4);
        nn_sync(); const double ms = (now() - t0) * 1e3 / steps;
        const char *e = nn_check();
        printf("  %-22s %8.2f ms per optimizer step (%.1f M params)%s%s\n", name[v], ms, n / 1e6, e ? "  cuda: " : "", e ? e : "");
        unet_free(u);
    }
    nn_muon_tc(0);
    return 0;
}
