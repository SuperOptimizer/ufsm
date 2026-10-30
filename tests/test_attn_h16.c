/* bf16 transformer blocks (src/nn_attn_h16.cu, at_set_mx(16)) against the fp32 blocks (at_set_mx(0), fp32 cuBLASLt, the same fp16 attention
   core): output y, input gradient and every parameter gradient group, max |h16 - fp32| / rms(fp32) (bf16 activations: 1e-2 .. 3.5e-2
   expected; pass: max < 6e-2 and |d|_2 / |ref|_2 < 1.5e-2). Geometries: the vwt ray (sequences of L tokens along W) and transverse (Hh x W planes) blocks at small sizes, with partial
   64-row tiles; RoPE on; 4 heads x 48, F 256.
   Usage: test_attn_h16 [bench]  (bench: fwd / bwd ms of both at the vwt slab's 98304 tokens, ray and transverse geometry) */
#include "attn.h"
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t rs = 777;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) * 0x1.0p-53; }
static double nrand(void) { double a = urand() + 1e-300, b = urand(); return sqrt(-2 * log(a)) * cos(6.283185307179586 * b); }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static float *dev_from(const float *h, size_t n) { float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); return d; }
/* max |a - b| / rms(b) and |a - b|_2 / |b|_2; pass: max < 6e-2, l2 < 1.5e-2 */
static int cmpv(const char *nm, const float *a, const float *b, size_t n) {
    double ss = 0, sd = 0, mx = 0;
    for (size_t i = 0; i < n; i++) { ss += (double)b[i] * b[i]; const double d = fabs((double)a[i] - b[i]); sd += d * d; if (d > mx || d != d) mx = d != d ? 1e30 : d; }
    const double r = mx / (sqrt(ss / n) + 1e-30), l2 = sqrt(sd / (ss + 1e-300)); const int ok = r < 6e-2 && l2 < 1.5e-2;
    printf("  %-6s max %.3e  l2 %.3e%s\n", nm, r, l2, ok ? "" : "  FAIL");
    return !ok;
}

typedef struct { const char *nm; int nseq, S, Hh, W; } geo;

static int run(const at_dims *d, const geo *q, int bench) {
    const int C = d->C; const long T = (long)q->nseq * q->S;
    at_bp bp; const size_t np = at_block_layout(d, &bp, 0);
    float *hp = malloc(np * 4);
    for (size_t i = 0; i < np; i++) hp[i] = (float)(0.06 * nrand());
    for (int c = 0; c < C; c++) { hp[bp.ln1g + c] = (float)(1 + 0.1 * nrand()); hp[bp.ln2g + c] = (float)(1 + 0.1 * nrand()); hp[bp.qg + c] = (float)(1 + 0.1 * nrand()); hp[bp.kg + c] = (float)(1 + 0.1 * nrand()); }
    float *P = dev_from(hp, np), *G[2] = {nn_malloc(np * 4), nn_malloc(np * 4)};
    float *hx = malloc(T * C * 4), *hg = malloc(T * C * 4);
    for (long i = 0; i < T * C; i++) { hx[i] = (float)nrand(); hg[i] = (float)nrand(); }
    float *x = dev_from(hx, T * C), *y[2] = {nn_malloc(T * C * 4), nn_malloc(T * C * 4)}, *gy[2] = {nn_malloc(T * C * 4), nn_malloc(T * C * 4)};
    at_geom g = {q->nseq, q->S, 1, q->Hh, q->W, NULL, NULL, NULL, {0}, NULL};
    void *wm = nn_malloc(at_ws_bytes(d, T)); at_ws w; at_ws_carve(&w, d, T, wm);
    int fail = 0;
    printf("%s: %d seqs x %d tokens (%ld)\n", q->nm, q->nseq, q->S, T);
    double tm[2][2] = {{0}};
    for (int m = 0; m < 2; m++) {
        at_set_mx(m ? 16 : 0);
        nn_zero(G[m], np * 4); nn_h2d(gy[m], hg, T * C * 4);
        if (at_block_fwd(d, P, &bp, x, y[m], &g, &w) || at_block_bwd(d, P, G[m], &bp, x, gy[m], &g, &w)) { printf("  FAIL: call error (mx %d)\n", m ? 16 : 0); return 1; }
        nn_sync();
        if (nn_check()) { printf("  FAIL: %s\n", nn_check()); return 1; }
        if (bench) {
            double bf = 1e9, bb = 1e9;
            for (int it = 0; it < 5; it++) {
                nn_h2d(gy[m], hg, T * C * 4); nn_sync(); const double t0 = now();
                at_block_fwd(d, P, &bp, x, y[m], &g, &w); nn_sync(); const double t1 = now();
                at_block_bwd(d, P, G[m], &bp, x, gy[m], &g, &w); nn_sync(); const double t2 = now();
                if (t1 - t0 < bf) bf = t1 - t0; if (t2 - t1 < bb) bb = t2 - t1;
            }
            tm[m][0] = bf * 1e3; tm[m][1] = bb * 1e3;
        }
    }
    at_set_mx(0);
    float *a = malloc(np * 4), *b = malloc(np * 4), *ta = malloc(T * C * 4), *tb = malloc(T * C * 4);
    nn_d2h(ta, y[1], T * C * 4); nn_d2h(tb, y[0], T * C * 4); fail += cmpv("y", ta, tb, T * C);
    nn_d2h(ta, gy[1], T * C * 4); nn_d2h(tb, gy[0], T * C * 4); fail += cmpv("dx", ta, tb, T * C);
    nn_d2h(a, G[1], np * 4); nn_d2h(b, G[0], np * 4);
    const struct { const char *nm; size_t off, n; } grp[] = {
        {"ln1g", bp.ln1g, C}, {"ln1b", bp.ln1b, C}, {"wqkv", bp.wqkv, 3 * C * C}, {"bqkv", bp.bqkv, 3 * C}, {"qg", bp.qg, C}, {"kg", bp.kg, C},
        {"wo", bp.wo, C * C}, {"bo", bp.bo, C}, {"ln2g", bp.ln2g, C}, {"ln2b", bp.ln2b, C}, {"w1", bp.w1, 2 * d->F * C}, {"b1", bp.b1, 2 * d->F},
        {"w2", bp.w2, C * d->F}, {"b2", bp.b2, C}};
    for (size_t k = 0; k < sizeof grp / sizeof grp[0]; k++) fail += cmpv(grp[k].nm, a + grp[k].off, b + grp[k].off, grp[k].n);
    if (bench) printf("  fp32: fwd %.2f bwd %.2f ms | h16: fwd %.2f bwd %.2f ms\n", tm[0][0], tm[0][1], tm[1][0], tm[1][1]);
    nn_free(P); nn_free(G[0]); nn_free(G[1]); nn_free(x); nn_free(y[0]); nn_free(y[1]); nn_free(gy[0]); nn_free(gy[1]); nn_free(wm);
    free(hp); free(hx); free(hg); free(a); free(b); free(ta); free(tb);
    return fail;
}

int main(int argc, char **argv) {
    if (nn_init(0)) { printf("SKIP: no GPU\n"); return 0; }
    const int bench = argc > 1 && !strcmp(argv[1], "bench");
    at_set_core(1);
    const at_dims d = {192, 4, 256, 1, 0, 100.f, 0, 0, {0}, 0, 0};
    int fail = 0;
    if (bench) {
        const geo g[2] = {{"ray", 256, 384, 1, 384}, {"transverse", 384, 256, 16, 16}};
        for (int i = 0; i < 2; i++) fail += run(&d, &g[i], 1);
    } else {
        const geo g[3] = {{"ray", 6, 100, 1, 100}, {"transverse", 5, 64, 8, 8}, {"transverse-odd", 3, 90, 9, 10}};
        for (int i = 0; i < 3; i++) fail += run(&d, &g[i], 0);
    }
    printf(fail ? "FAIL (%d)\n" : "PASS\n", fail);
    return fail ? 1 : 0;
}
