/* The GroupNorm backward's gy records (lp_gn_silu_bwd_apply_rec) are per host thread; data-parallel training drives every GPU from one
   thread, so the buffers must follow the device (lp_gyrec_select parks one GPU's and takes the next GPU's back). The old code dropped
   the other GPU's buffers at every switch: with 2 GPUs each lost ~one record buffer per step until the cards were full (the desk's
   wind2-wk1: an illegal memory access 1211 steps after every start). On one GPU: apply on device 0, a switch to "GPU 1" and back,
   200 times: the device-0 buffers come back (same pointers, no new allocation, free memory unchanged); the bookkeeping of two
   devices' buffers is kept apart. */
#include "nn.h"
#include "nn_lp.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void lp_gyrec_select(int d);
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

int main(void) {
    nn_init(0);
    const shape5 s = {1, 32, 16, 16, 16};
    const size_t ne = (size_t)s.c * s.d * s.h * s.w, G = 8;
    void *x = nn_malloc(4 * ne), *gy = nn_malloc(4 * ne), *gx = nn_malloc(4 * ne);
    float *gamma = nn_malloc(s.c * 4), *beta = nn_malloc(s.c * 4), *mean = nn_malloc(G * 4), *rstd = nn_malloc(G * 4), *AB = nn_malloc(2 * G * 4);
    nn_zero(x, 4 * ne); nn_zero(gy, 4 * ne); nn_zero(gx, 4 * ne); nn_zero(gamma, s.c * 4); nn_zero(beta, s.c * 4); nn_zero(mean, G * 4); nn_zero(rstd, G * 4); nn_zero(AB, 2 * G * 4);
    if (!lp_gn_silu_bwd_apply_rec(x, 8, s, (int)G, gamma, beta, mean, rstd, gy, gx, AB, 1, 2)) { fprintf(stderr, "test_gyrec: the records path is off here (UFSM_GYREC?): skipped\n"); return 0; }
    nn_sync();
    const uint8_t *rec0 = g_gyrec.rec; const float *gb0 = g_gyrec.gb;
    const size_t f0 = nn_mem_free();
    for (int i = 0; i < 200; i++) {
        lp_gyrec_select(1);
        CHECK(g_gyrec.dev == 1 && (i > 0 || (!g_gyrec.rec && !g_gyrec.gb)), "switch %d: device %d, rec %p", i, g_gyrec.dev, (void *)g_gyrec.rec);
        lp_gyrec_select(0);
        CHECK(g_gyrec.rec == rec0 && g_gyrec.gb == gb0, "switch %d: device 0's buffers not restored", i);
        if (!lp_gn_silu_bwd_apply_rec(x, 8, s, (int)G, gamma, beta, mean, rstd, gy, gx, AB, 1, 2)) { CHECK(0, "apply %d fell back", i); break; }
        CHECK(g_gyrec.rec == rec0, "apply %d reallocated the records", i);
    }
    nn_sync();
    const size_t f1 = nn_mem_free();
    const double lost = ((double)f0 - (double)f1) / 1e6;
    printf("200 device switches: records %p kept, device memory lost %.1f MB (record buffer %.2f MB)\n", (void *)g_gyrec.rec, lost, g_gyrec.cap / 1e6);
    CHECK(lost < 2 + g_gyrec.cap / 1e6, "device memory lost %.1f MB over 200 switches", lost);   /* (the other process on a shared GPU may move it a little) */
    /* the other device's slot holds its own buffers: park a marker as "GPU 1"'s, switch away and back */
    lp_gyrec_select(1);
    uint8_t *marker = (uint8_t *)0x1000; g_gyrec.rec = marker; g_gyrec.cap = 77;
    lp_gyrec_select(0); CHECK(g_gyrec.rec == rec0, "device 0 after parking device 1");
    lp_gyrec_select(1); CHECK(g_gyrec.rec == marker && g_gyrec.cap == 77, "device 1's buffers not kept");
    g_gyrec.rec = nullptr; g_gyrec.cap = 0; lp_gyrec_select(0);
    if (fails) { fprintf(stderr, "test_gyrec: %d failure(s)\n", fails); return 1; }
    printf("test_gyrec: OK\n");
    return 0;
}
