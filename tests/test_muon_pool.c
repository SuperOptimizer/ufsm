/* The ext Muon path (--normuon 1) in groups over one shared scratch pool (unet.c mx_groupable; 2026-11-15 try7 one card: one pool for
   every matrix was 0.95 GB at the 104 M net, allocated at the first update outside the memory plan): an attention U-Net stepped 3 times
   with NorMuon at UFSM_MUON_POOL_MB 0.05 (many groups) and at 4096 (one group) from the same weights and gradients gives the same
   weights bit for bit (plain Muon likewise); unet_reserve now builds the ext path's table and pool too. */
#include "unet.h"
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { printf("  FAIL: " __VA_ARGS__); printf("\n"); fails++; } } while (0)

static float *steps(const char *pool_mb, int ext, size_t *n_out, size_t *first_alloc) {
    setenv("UFSM_MUON_POOL_MB", pool_mb, 1);
    unet_cfg cfg = {}; cfg.nlev = 3; cfg.widths[0] = 32; cfg.widths[1] = 64; cfg.widths[2] = 96; cfg.cin = 4; cfg.cout = 2; cfg.G = 8; cfg.attn_blocks = 1; cfg.attn_pe = 0;
    unet *u = unet_create(&cfg); unet_init(u, 3);
    if (ext) unet_set_muon_variant(u, 0, 1, 0.95f);
    unet_reserve(u, 1, 0);
    const size_t n = unet_nparams(u); float *g = malloc(n * 4), *p = malloc(n * 4);
    size_t f0 = 0, f1 = 0, to = 0;
    for (int s = 0; s < 3; s++) {
        unsigned r = 11u + 7u * (unsigned)s; for (size_t i = 0; i < n; i++) { r = r * 1664525u + 1013904223u; g[i] = ((r >> 8) / 16777216.f - 0.5f) * 1e-2f; }
        unet_grad_h2d(u, g);
        if (!s) { nn_sync(); nn_dev_mem(&f0, &to); }
        unet_muon(u, 0.02f, 0.95f, 1e-3f, 0.9f, 0.999f, 1e-8f, 0.1f, s + 1);
        if (!s) { nn_sync(); nn_dev_mem(&f1, &to); }
    }
    unet_params_d2h(u, p); free(g); unet_free(u);
    *n_out = n; *first_alloc = f0 > f1 ? f0 - f1 : 0; return p;
}

int main(void) {
    if (nn_init(0)) return 1;
    for (int ext = 0; ext < 2; ext++) {
        size_t n, n2, a1, a2;
        float *many = steps("0.05", ext, &n, &a1), *one = steps("4096", ext, &n2, &a2);
        const int same = n == n2 && !memcmp(many, one, n * 4);
        printf("%s: 3 steps, pool 0.05 MB (groups) vs 4096 MB (one): %s; device memory taken by the first update %.1f / %.1f MB\n",
               ext ? "ext (normuon)" : "plain", same ? "bit for bit" : "DIFFERENT", a1 / 1048576.0, a2 / 1048576.0);
        CHECK(same, "%s: grouped and one-pool weights differ", ext ? "normuon" : "plain");
        /* (the device's free memory is shared with concurrent suite tests: the first update's share is printed, not checked; train's
           planner check is the "memory:" lines of a UFSM_MEM_DRY run) */
        free(many); free(one);
    }
    if (fails) { printf("test_muon_pool: %d FAILED\n", fails); return 1; }
    printf("test_muon_pool: OK\n");
    return 0;
}
