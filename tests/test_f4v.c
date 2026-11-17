/* conv_fwd_f4v_k (src/lp_f4v.cu) against conv_fwd_f4_k: the same MMAs in the same order per accumulator, so every output byte and
   every GroupNorm statistic must be identical (lp_set_f4v 0 / 1 on the same operands), over random (not exact) operands:
   - the forward over a GroupNorm input (gn_prenorm_f4_k, then copy staging) with output statistics, depths / heights / widths that
     leave partial tiles, Cop 64 .. 256, three and more input chunks, a split input (decoder concat), a z-window, a z-range input;
   - a plain input (raw block output) without statistics; the backward-data shape (MX-fp4 gy, output SR);
   - stored statistics (sp.stored_stats). Each case also checks that the new kernel actually ran (lp_f4v_calls). */
#include "nn.h"
#include "nn_lp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void lp_set_f4v(int on);
void lp_set_f4_prenorm(int on);
void lp_set_f4v_preslab(int z);
unsigned long lp_f4v_calls(void);
int lp_sw_build(void);
static uint64_t rs = 0xf4f4f4ull;
static unsigned rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (unsigned)(rs >> 32); }
static float frand(void) { return (rnd() >> 8) * (1.f / 16777216.f) * 2.f - 1.f; }
static int bad;
static void check(const char *what, int ok) { printf("  %-84s %s\n", what, ok ? "ok" : "FAIL"); if (!ok) bad++; }
static float *dev_rand(size_t n, float sc, float add) { float *h = malloc(n * 4); for (size_t i = 0; i < n; i++) h[i] = frand() * sc + add; float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4); free(h); return d; }
static void *mx4_rand(shape5 s, float sc) {
    const size_t n = shape_numel(s); float *d = dev_rand(n, sc, 0.f);
    const size_t b = nn_mx4_bytes(s); void *p = nn_malloc(b); nn_zero(p, b); nn_set_storage(p, b, 4);
    lp_f32_to_mx4(d, s.n, s.c, shape_spatial(s), p); nn_free(d); return p;
}
static void *mx4_out(shape5 s, size_t *bytes) { const size_t b = nn_mx4_bytes(s); void *p = nn_malloc(b); nn_zero(p, b); nn_set_storage(p, b, 4); *bytes = b; return p; }
enum { GN = 1, STATS = 2, OSR = 4, SPLIT = 8, ZWIN = 16, ZRANGE = 32, SST = 64, GN2 = 128, PRESLAB = 256 };
static void run_case(const char *name, int Ci, int Co, int D, int H, int W, int fl) {
    const int Dx = fl & ZRANGE ? D + 3 : D;   /* (z-range: output plane z reads input plane z + 1 of a deeper input) */
    shape5 xs = {1, Ci, Dx, H, W}, ys = {1, Co, D, H, W};
    const int split = fl & SPLIT ? Ci - 64 : 0;
    float *w = dev_rand((size_t)Co * Ci * 27, 0.05f, 0.f), *b = dev_rand(Co, 0.1f, 0.f);
    void *x, *x2 = nullptr;
    split_t sp = {0};
    if (split) { shape5 a = xs; a.c = split; x = mx4_rand(a, 2.f); a.c = Ci - split; x2 = mx4_rand(a, 2.f); sp.x2 = x2; sp.c_split = split; }
    else x = mx4_rand(xs, 2.f);
    const int G = 8;
    gnp_t gp = {0};
    float *ga = nullptr, *be = nullptr, *mu = nullptr, *rsd = nullptr;
    if (fl & GN) { ga = dev_rand(Ci, 0.2f, 1.f); be = dev_rand(Ci, 0.1f, 0.f); mu = dev_rand(G, 0.1f, 0.f); rsd = dev_rand(G, 0.1f, 1.f); gp = gnp_mk(ga, be, mu, rsd, G); }
    float *g2a = nullptr, *g2b = nullptr, *g2m = nullptr, *g2r = nullptr;
    if (fl & GN2) {   /* per-segment GroupNorms (decoder concat): gp over the first split channels, gp2 over the rest */
        g2a = dev_rand(Ci - split, 0.3f, 1.f); g2b = dev_rand(Ci - split, 0.2f, 0.f); g2m = dev_rand(G, 0.2f, 0.f); g2r = dev_rand(G, 0.1f, 1.f);
        sp.gp2 = gnp_mk(g2a, g2b, g2m, g2r, G);
    }
    if (fl & OSR) sp.osr = 0x9e3779b9u;
    if (fl & ZWIN) { sp.zw0 = 4; sp.zw1 = D - 4; }
    if (fl & ZRANGE) { sp.zi0 = 1; sp.Di = (int16_t)Dx; }
    if (fl & SST) sp.stored_stats = 1;
    shape5 xc = xs; xc.d = D;   /* the conv's own shape (Di carries the input depth) */
    const int Go = 8;
    const size_t osb = (size_t)2 * Go * 3 * sizeof(double);
    double *os[2] = {nullptr, nullptr};
    size_t yb = 0; void *y[2] = {mx4_out(ys, &yb), mx4_out(ys, &yb)};
    unsigned long c0 = 0;
    for (int k = 0; k < 2; k++) {
        if (fl & STATS) { os[k] = nn_malloc(osb); nn_zero(os[k], osb); }
        lp_set_f4v(k);
        if (fl & PRESLAB) { lp_set_f4_prenorm(k); lp_set_f4v_preslab(k ? 8 : 0); }   /* old: the in-kernel GroupNorm staging; new: the prenorm by 8-plane slabs */
        if (k) c0 = lp_f4v_calls();
        lp_conv_fwd_f4(x, 4, xc, w, b, Co, y[k], 4, gp, os[k], fl & STATS ? Go : 0, sp);
    }
    const int ran = lp_f4v_calls() == c0 + 1;
    lp_set_f4v(0); lp_set_f4_prenorm(1); lp_set_f4v_preslab(0);
    nn_sync();
    const char *e = lp_check();
    unsigned char *h0 = malloc(yb), *h1 = malloc(yb);
    nn_d2h(h0, y[0], yb); nn_d2h(h1, y[1], yb);
    size_t nd = 0; for (size_t i = 0; i < yb; i++) nd += h0[i] != h1[i];
    int sok = 1;
    if (fl & STATS) { unsigned char s0[512], s1[512]; nn_d2h(s0, os[0], osb); nn_d2h(s1, os[1], osb); sok = !memcmp(s0, s1, osb); }
    size_t nz = 0; for (size_t i = 0; i < yb; i++) nz += h0[i] != 0;
    char msg[256];
    snprintf(msg, sizeof msg, "%s: %d->%d %dx%dx%d: %zu of %zu bytes differ%s%s%s", name, Ci, Co, D, H, W, nd, yb, fl & STATS ? (sok ? ", stats equal" : ", STATS DIFFER") : "",
             ran ? "" : ", F4V NOT RUN", e ? e : "");
    check(msg, !e && ran && nd == 0 && sok && nz > yb / 4);
    free(h0); free(h1);
    nn_free(w); nn_free(b); nn_free(x); nn_free(x2); nn_free(y[0]); nn_free(y[1]); nn_free(os[0]); nn_free(os[1]);
    nn_free(ga); nn_free(be); nn_free(mu); nn_free(rsd); nn_free(g2a); nn_free(g2b); nn_free(g2m); nn_free(g2r);
}
int main(void) {
    nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0);
    if (lp_sw_build()) { printf("test_f4v: sm_120a only (ARCH=86 build: conv_fwd_f4v_k not built), skipped\n"); return 0; }
    lp_wmemo_step(1);
    printf("test_f4v: conv_fwd_f4v_k vs conv_fwd_f4_k, byte equality\n");
    run_case("fwd GN + stats", 128, 128, 12, 24, 48, GN | STATS);
    run_case("fwd GN + stats, odd depth / height", 256, 192, 13, 20, 32, GN | STATS);
    run_case("fwd GN + stored stats", 128, 256, 16, 16, 32, GN | STATS | SST);
    run_case("fwd plain + stats (raw block output)", 96, 64, 8, 8, 16, STATS);
    run_case("fwd plain, no stats", 160, 128, 10, 16, 40, 0);
    run_case("bdata (MX-fp4 gy, output SR)", 192, 128, 12, 16, 32, OSR);
    run_case("bdata, odd width, no stats", 128, 64, 9, 11, 37, OSR);
    run_case("split input (128 | 64), stats", 192, 128, 12, 16, 32, SPLIT | STATS);
    run_case("z-window [4, D - 4), stats", 128, 128, 16, 16, 32, STATS | ZWIN);
    run_case("GN staged in the kernel (one output tile: no prenorm)", 128, 64, 12, 16, 32, GN | STATS);
    run_case("GN staged, a partial chunk (80 channels: masked GN)", 80, 64, 10, 16, 32, GN | STATS);
    run_case("GN staged, z-window (no prenorm)", 128, 128, 16, 16, 32, GN | STATS | ZWIN);
    run_case("GN staged, split input, one GN over both", 192, 128, 12, 16, 32, GN | SPLIT | STATS);
    run_case("GN staged, split input, per-segment GNs", 192, 256, 12, 24, 48, GN | GN2 | SPLIT | STATS);
    run_case("slabbed prenorm vs in-kernel GN staging (8-plane slabs)", 128, 128, 20, 16, 32, GN | STATS | PRESLAB);
    run_case("slabbed prenorm, stored stats, odd depth", 256, 192, 13, 24, 32, GN | STATS | SST | PRESLAB);
    run_case("z-range input (zi0 1, Di D + 3)", 128, 64, 12, 16, 32, ZRANGE | OSR);
    if (getenv("UFSM_TEST_QUICK") == nullptr) run_case("level-0 shape", 256, 256, 16, 64, 64, GN | STATS);
    printf("test_f4v: %s\n", bad ? "FAIL" : "ok");
    return bad != 0;
}
