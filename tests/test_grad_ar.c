/* nn_grad_allreduce2 (perf2 2026-11-13: the split / two-GPU gradient sum in halves, both directions at once, host-staged) against the host
   sum: after the call both GPUs hold a + b (fp32, one rounding: the bytes of train.c's old peer-copy + axpy + broadcast path), and repeat
   calls (the step loop: pinned and receive buffers reused across phases and calls) stay exact. UFSM_TEST_QUICK=1: 9 M floats in ~1 M-float
   chunks (several chunks a phase, odd sizes and offsets); full: the 92 M net's 92.4 M floats at the default 32 MB chunks, timed against the
   old path. Two GPUs if present, else the same device twice.
   Then dp_grad_avg (split.c: train.c's data-parallel average, peer copy + axpy in chunks on GPU 0, scale, broadcast) against the host
   mean, byte for byte (the host does the same float operations in the same order: g0 + g1 (+ g2), then x (1 / ng)): 40 M + 12345 floats
   (try6's ~92 M are ~6 chunks; here 16 M-float chunks give 2 full + an odd tail, 1000003-float chunks 41 with a tail), 2 replicas
   (dev 0, 1) and 3 (dev 0, 1, 0: 1 / 3 is no dyadic scale), avg on and off (--split), every replica checked after the broadcast. */
#include "nn.h"
#include "split.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rs = 88172645463325252ull;
static float frand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 40) / 16777216.0 - 0.5) * 1e-3f; }
/* dp_grad_avg against the host: ng replicas on devices dv[k], n floats, chunk floats a peer copy; 1 on failure */
static int dp_check(const int *dv, int ng, size_t n, size_t chunk, int avg) {
    float *h[3], *g[3], *ref = malloc(n * 4), *got = malloc(n * 4); int bad = 0;
    for (int k = 0; k < ng; k++) { h[k] = malloc(n * 4); for (size_t i = 0; i < n; i++) h[k][i] = frand() * (float)(1 + (i % 5)); }
    for (size_t i = 0; i < n; i++) { float a = h[0][i]; for (int k = 1; k < ng; k++) a = a + h[k][i]; ref[i] = avg ? a * (1.f / ng) : a; }
    for (int k = 0; k < ng; k++) { nn_init(dv[k]); g[k] = nn_malloc(n * 4); nn_h2d(g[k], h[k], n * 4); nn_sync(); }
    nn_init(dv[0]); float *tmp = nn_malloc(chunk * 4);
    dp_grad_avg(g, dv, ng, n, tmp, chunk, avg);
    for (int k = 0; k < ng; k++) {
        nn_init(dv[k]); nn_sync(); nn_d2h(got, g[k], n * 4);
        if (memcmp(got, ref, n * 4)) { size_t i = 0, nd = 0; while (got[i] == ref[i]) i++; for (size_t j = 0; j < n; j++) nd += memcmp(got + j, ref + j, 4) != 0;
            fprintf(stderr, "FAIL: dp_grad_avg %d replicas, chunk %zu, avg %d: replica %d (GPU %d) differs at %zu of %zu floats, first %zu: %g vs %g\n", ng, chunk, avg, k, dv[k], nd, n, i, got[i], ref[i]); bad = 1; }
    }
    const char *e = nn_check(); if (e) { fprintf(stderr, "FAIL: dp_grad_avg: %s\n", e); bad = 1; }
    fprintf(stderr, "dp_grad_avg: %d replicas (GPUs %d, %d%s), %zu floats in %zu-float chunks (%zu), avg %d: %s\n", ng, dv[0], dv[1], ng > 2 ? ", 0" : "", n, chunk,
            (n + chunk - 1) / chunk, avg, bad ? "FAIL" : "== host, every replica");
    for (int k = 0; k < ng; k++) { nn_init(dv[k]); nn_free(g[k]); free(h[k]); }
    nn_init(dv[0]); nn_free(tmp); free(ref); free(got);
    return bad;
}
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
    {   /* train.c's peer path (dp_grad_avg) at several chunks: g[] freed first (the full run's 92 M floats + these 3 x 40 M fit a 16 GB card) */
        for (int k = 0; k < 2; k++) { nn_init(dv[k]); nn_free(g[k]); }
        const size_t na = ((size_t)40 << 20) + 12345; const int d3[3] = {dv[0], dv[1], dv[0]};
        fails += dp_check(dv, 2, na, (size_t)16 << 20, 1);
        fails += dp_check(dv, 2, na, 1000003, 1);
        fails += dp_check(d3, 3, na, (size_t)16 << 20, 1);
        fails += dp_check(dv, 2, na, (size_t)16 << 20, 0);
        for (int k = 0; k < 2; k++) { nn_init(dv[k]); g[k] = nn_malloc(n * 4); nn_h2d(g[k], k ? hb : ha, n * 4); }   /* (the timing below reads them) */
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
