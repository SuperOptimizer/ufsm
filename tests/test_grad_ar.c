/* nn_grad_allreduce2 (perf2 2026-11-13: the split / two-GPU gradient sum in halves, both directions at once, host-staged) against the host
   sum: after the call both GPUs hold a + b (fp32, one rounding: the bytes of train.c's old peer-copy + axpy + broadcast path), and repeat
   calls (the step loop: pinned and receive buffers reused across phases and calls) stay exact. UFSM_TEST_QUICK=1: 9 M floats in ~1 M-float
   chunks (several chunks a phase, odd sizes and offsets); full: the 92 M net's 92.4 M floats at the default 32 MB chunks, timed against the
   old path. Two GPUs if present, else the same device twice. */
#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rs = 88172645463325252ull;
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 40) / 16777216.0 - 0.5) * 1e-3f; }
int main(void) {
    const int quick = getenv("UFSM_TEST_QUICK") && atoi(getenv("UFSM_TEST_QUICK"));
    if (quick) setenv("UFSM_AR_CHUNK", "1000003", 1);
    const size_t n = quick ? 9000017 : 92400000;
    int dv[2] = {0, 1}; if (nn_init(0)) { fprintf(stderr, "no GPU (skipped)\n"); return 0; }
    if (nn_init(1)) { dv[1] = 0; nn_check(); }
    fprintf(stderr, "grad allreduce2: GPUs %d, %d, %zu floats\n", dv[0], dv[1], n);
    float *ha = malloc(n * 4), *hb = malloc(n * 4), *hs = malloc(n * 4), *hr = malloc(n * 4); int fails = 0;
    float *g[2];
    for (int it = 0; it < 3; it++) {
        for (size_t i = 0; i < n; i++) { ha[i] = frand(); hb[i] = frand(); hs[i] = ha[i] + hb[i]; }
        for (int k = 0; k < 2; k++) { nn_init(dv[k]); if (!it) g[k] = nn_malloc(n * 4); nn_h2d(g[k], k ? hb : ha, n * 4); nn_sync(); }
        nn_init(dv[0]);
        if (nn_grad_allreduce2(g, dv, n)) { fprintf(stderr, "FAIL: no buffers\n"); return 1; }
        for (int k = 0; k < 2; k++) { nn_init(dv[k]); nn_d2h(hr, g[k], n * 4); nn_sync();
            if (memcmp(hr, hs, n * 4)) { size_t i = 0; while (hr[i] == hs[i]) i++; fprintf(stderr, "FAIL: call %d GPU %d: element %zu %g vs %g\n", it, k, i, hr[i], hs[i]); fails++; } }
        const char *e = nn_check(); if (e) { fprintf(stderr, "FAIL: %s\n", e); fails++; }
    }
    if (!quick) {   /* timing: the old path (synchronous peer copy to GPU 0, axpy, peer copy back) vs allreduce2 */
        nn_init(dv[0]); float *gp = nn_malloc(n * 4); double t = now();
        for (int r = 0; r < 3; r++) { nn_peer_copy(gp, dv[0], g[1], dv[1], n * 4); nn_axpy(g[0], 1.f, gp, n); nn_peer_copy(g[1], dv[1], g[0], dv[0], n * 4); }
        nn_sync(); const double t0 = (now() - t) / 3; t = now();
        for (int r = 0; r < 3; r++) nn_grad_allreduce2(g, dv, n);
        for (int k = 0; k < 2; k++) { nn_init(dv[k]); nn_sync(); }
        fprintf(stderr, "%zu floats (%.0f MB): old path %.1f ms, allreduce2 %.1f ms\n", n, n * 4e-6, t0 * 1e3, (now() - t) / 3 * 1e3);
    }
    if (fails) { fprintf(stderr, "test_grad_ar: %d failure(s)\n", fails); return 1; }
    fprintf(stderr, "test_grad_ar: ok\n"); return 0;
}
