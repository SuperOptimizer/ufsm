/* the 2:4 (pair-wise 4:8) sparse MX-fp4 conv (src/lp_sp24conv.cu) against the dense fp4 kernel on the same masked weights:
   - exact operands (every MX block holds a +-6, so all scales are 1 and every product / partial sum is exact in fp32 whatever the
     summation order): the stored MX-fp4 outputs must be byte-identical (plain, output SR of the gradient path, split input)
   - a GroupNorm + SiLU input (prenorm path): the sums are no longer order-independent, so the outputs agree to fp32 rounding
     (rare one-step e2m1 flips); plus the dequantised output against an fp64 host conv of the dequantised operands
   - the prep's truncation counter: zero for masked weights, non-zero for dense ones */
#include "nn.h"
#include "nn_lp.h"
#include "lp_sp24.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 0x5eed24ull;
static unsigned rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (unsigned)(rs >> 32); }
static float frand(void) { return (rnd() >> 8) * (1.f / 16777216.f) * 2.f - 1.f; }
static const float grid[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
static float rgrid(void) { const unsigned r = rnd(); return (r & 8 ? -1.f : 1.f) * grid[r & 7]; }
int lp_sw_build(void);   /* lp_mxsw.cu: 1 = the software MX paths (ARCH=86) */
static int bad;
static void check(const char *what, int ok) { printf("  %-72s %s\n", what, ok ? "ok" : "FAIL"); if (!ok) bad++; }

/* pair-wise 4:8 mask: per (co, tap, 8 input channels) keep the 2 channel pairs of largest |w0| + |w1| (ties: lower index) */
static void host_mask48(float *w, int Co, int Ci) {
    for (int co = 0; co < Co; co++) for (int t = 0; t < 27; t++) for (int c0 = 0; c0 + 8 <= Ci; c0 += 8) {
        float m[4]; for (int b = 0; b < 4; b++) m[b] = fabsf(w[((size_t)co * Ci + c0 + 2 * b) * 27 + t]) + fabsf(w[((size_t)co * Ci + c0 + 2 * b + 1) * 27 + t]);
        int i0 = m[1] > m[0] ? 1 : 0, i1 = 1 - i0;
        for (int b = 2; b < 4; b++) { if (m[b] > m[i0]) { i1 = i0; i0 = b; } else if (m[b] > m[i1]) i1 = b; }
        for (int b = 0; b < 4; b++) if (b != i0 && b != i1) { w[((size_t)co * Ci + c0 + 2 * b) * 27 + t] = 0.f; w[((size_t)co * Ci + c0 + 2 * b + 1) * 27 + t] = 0.f; }
    }
}
/* exact weights: masked grid values, every (co, tap, 32-channel block) with a kept +-6 (scale 1) */
static void exact_w(float *w, int Co, int Ci) {
    for (size_t i = 0; i < (size_t)Co * Ci * 27; i++) w[i] = rgrid();
    host_mask48(w, Co, Ci);
    for (int co = 0; co < Co; co++) for (int t = 0; t < 27; t++) for (int c0 = 0; c0 < Ci; c0 += 32) {
        int c = -1; for (int k = 0; k < 32 && c < 0; k++) if (w[((size_t)co * Ci + c0 + k) * 27 + t] != 0.f) c = c0 + k;
        if (c < 0) { c = c0; w[((size_t)co * Ci + c + 1) * 27 + t] = 1.f; }   /* (an all-zero block: keep pair 0) */
        w[((size_t)co * Ci + c) * 27 + t] = rnd() & 1 ? 6.f : -6.f;
    }
}
static void exact_x(float *x, shape5 s) {   /* every (position, 32-channel block) holds a +-6 */
    const size_t S = shape_spatial(s);
    for (int c = 0; c < s.c; c++) for (size_t v = 0; v < S; v++) x[(size_t)c * S + v] = rgrid();
    for (int c0 = 0; c0 < s.c; c0 += 32) for (size_t v = 0; v < S; v++) x[(size_t)(c0 + rnd() % 32) * S + v] = rnd() & 1 ? 6.f : -6.f;
}
static void *mx4_dev(const float *h, shape5 s) {
    const size_t n = shape_numel(s); float *d = nn_malloc(n * 4); nn_h2d(d, h, n * 4);
    const size_t b = nn_mx4_bytes(s); void *p = nn_malloc(b); nn_zero(p, b); nn_set_storage(p, b, 4);
    lp_f32_to_mx4(d, s.n, s.c, shape_spatial(s), p); nn_free(d); return p;
}
static void *mx4_out(shape5 s) { const size_t b = nn_mx4_bytes(s); void *p = nn_malloc(b); nn_zero(p, b); nn_set_storage(p, b, 4); return p; }
static float *deq(const void *p, shape5 s) { const size_t n = shape_numel(s); float *d = nn_malloc(n * 4), *h = malloc(n * 4); lp_mx4_to_f32(p, s.n, s.c, shape_spatial(s), d); nn_d2h(h, d, n * 4); nn_free(d); return h; }
static void host_conv(const float *x, shape5 xs, const float *w, int Co, double *y) {   /* N 1, zero padding */
    const int D = xs.d, H = xs.h, W = xs.w, Ci = xs.c; const size_t S = shape_spatial(xs);
    for (int co = 0; co < Co; co++) for (int z = 0; z < D; z++) for (int yy = 0; yy < H; yy++) for (int xx = 0; xx < W; xx++) {
        double s = 0;
        for (int ci = 0; ci < Ci; ci++) for (int t = 0; t < 27; t++) {
            const int iz = z + t / 9 - 1, iy = yy + (t / 3) % 3 - 1, ix = xx + t % 3 - 1;
            if (iz < 0 || iz >= D || iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
            const float wv = w[((size_t)co * Ci + ci) * 27 + t]; if (wv == 0.f) continue;
            s += (double)wv * x[(size_t)ci * S + ((size_t)iz * H + iy) * W + ix];
        }
        y[(size_t)co * S + ((size_t)z * H + yy) * W + xx] = s;
    }
}
/* one case: dense vs sparse on the same masked weights. exact: byte comparison; else tolerance + host reference */
static void run_case(const char *name, int Ci, int Co, int D, int H, int W, int exact, int gn, unsigned osr, int split) {
    shape5 xs = {1, Ci, D, H, W}, ys = {1, Co, D, H, W};
    const size_t nx = shape_numel(xs), ny = shape_numel(ys), nw = (size_t)Co * Ci * 27;
    float *hx = malloc(nx * 4), *hw = malloc(nw * 4);
    if (exact) { exact_x(hx, xs); exact_w(hw, Co, Ci); }
    else { for (size_t i = 0; i < nx; i++) hx[i] = frand() * 2.f; for (size_t i = 0; i < nw; i++) hw[i] = frand() * 0.05f; host_mask48(hw, Co, Ci); }
    float *w = nn_malloc(nw * 4); nn_h2d(w, hw, nw * 4);
    void *x = mx4_dev(hx, xs);
    gnp_t gp = {0}; float *ga = nullptr, *be = nullptr, *mu = nullptr, *rsd = nullptr;
    const int G = 8;
    if (gn) {
        float hg[512], hb[512], hm[8], hr[8];
        for (int c = 0; c < Ci; c++) { hg[c] = 1.f + 0.2f * frand(); hb[c] = 0.1f * frand(); }
        for (int g = 0; g < G; g++) { hm[g] = 0.05f * frand(); hr[g] = 1.2f + 0.1f * frand(); }
        ga = nn_malloc(Ci * 4); be = nn_malloc(Ci * 4); mu = nn_malloc(G * 4); rsd = nn_malloc(G * 4);
        nn_h2d(ga, hg, Ci * 4); nn_h2d(be, hb, Ci * 4); nn_h2d(mu, hm, G * 4); nn_h2d(rsd, hr, G * 4);
        gp = gnp_mk(ga, be, mu, rsd, G);
    }
    split_t sp = {0}; sp.osr = osr;
    void *x2 = nullptr;
    shape5 xs1 = xs;
    if (split) {   /* the input as two MX tensors (decoder concat): x channels [0, split), x2 the rest */
        xs1.c = split; shape5 xs2 = xs; xs2.c = Ci - split;
        const size_t S = shape_spatial(xs);
        x2 = mx4_dev(hx + (size_t)split * S, xs2);
        nn_free(x); x = mx4_dev(hx, xs1);
        sp.x2 = x2; sp.c_split = split;
    }
    void *yd = mx4_out(ys), *ys_ = mx4_out(ys);
    lp_sp24_force(0); lp_wmemo_clear();
    lp_conv_fwd_f4(x, 4, xs, w, nullptr, Co, yd, 4, gp, nullptr, 0, sp);
    lp_sp24_force(1); lp_sp24_err_clear();
    const int took = lp_sp24_conv_fwd(x, 4, xs, w, nullptr, Co, ys_, 4, gp, nullptr, 0, sp);
    lp_sp24_force(-1);
    nn_sync();
    const char *err = lp_check();
    char nm[160];
    snprintf(nm, sizeof nm, "%s: sparse kernel ran (%d -> %d ch, %dx%dx%d)%s%s", name, Ci, Co, D, H, W, err ? ": " : "", err ? err : "");
    check(nm, took == 1 && !err);
    snprintf(nm, sizeof nm, "%s: prep truncated no group of the masked weights", name); check(nm, lp_sp24_err() == 0);
    const size_t nb = nn_mx4_bytes(ys);
    uint8_t *bd = malloc(nb), *bs = malloc(nb); nn_d2h(bd, yd, nb); nn_d2h(bs, ys_, nb);
    size_t diff = 0; for (size_t i = 0; i < nb; i++) diff += bd[i] != bs[i];
    float *fd = deq(yd, ys), *fs = deq(ys_, ys);
    double d2 = 0, r2 = 0; size_t nd = 0;
    for (size_t i = 0; i < ny; i++) { const double d = (double)fs[i] - fd[i]; d2 += d * d; r2 += (double)fd[i] * fd[i]; nd += fs[i] != fd[i]; }
    if (exact) { snprintf(nm, sizeof nm, "%s: MX-fp4 output bytes sparse == dense (%zu of %zu differ)", name, diff, nb); check(nm, diff == 0); }
    else {
        snprintf(nm, sizeof nm, "%s: sparse vs dense %.2g rel, %.3f%% values differ", name, sqrt(d2 / r2), 100.0 * nd / ny);
        check(nm, sqrt(d2 / r2) < 0.02 && nd < ny / 100 + 1);
    }
    if (!gn && D * H * W <= 6 * 20 * 36) {   /* host reference of the dequantised operands (fp4 output rounding ~ 0.1 rel) */
        float *xq = malloc(nx * 4);
        if (split) { shape5 a = xs; a.c = split; float *p = deq(x, a); memcpy(xq, p, (size_t)split * shape_spatial(xs) * 4); free(p);
                     a.c = Ci - split; p = deq(x2, a); memcpy(xq + (size_t)split * shape_spatial(xs), p, (size_t)(Ci - split) * shape_spatial(xs) * 4); free(p); }
        else { float *p = deq(x, xs); memcpy(xq, p, nx * 4); free(p); }
        double *yr = malloc(ny * 8); host_conv(xq, xs, hw, Co, yr);
        double e2 = 0, q2 = 0; for (size_t i = 0; i < ny; i++) { const double d = fs[i] - yr[i]; e2 += d * d; q2 += yr[i] * yr[i]; }
        snprintf(nm, sizeof nm, "%s: sparse vs fp64 host conv %.3g rel (fp4 output)", name, sqrt(e2 / q2)); check(nm, sqrt(e2 / q2) < 0.2);
        free(xq); free(yr);
    }
    free(bd); free(bs); free(fd); free(fs); free(hx); free(hw);
    nn_free(w); nn_free(x); nn_free(x2); nn_free(yd); nn_free(ys_); nn_free(ga); nn_free(be); nn_free(mu); nn_free(rsd);
}

int main(void) {
    if (nn_init(getenv("UFSM_GPU") ? atoi(getenv("UFSM_GPU")) : 0)) { printf("no GPU\nFAIL\n"); return 1; }
    if (lp_sw_build()) { printf("software MX build: no sparse MMA, skipped\nok\n"); return 0; }
    run_case("MT4 exact", 128, 128, 6, 20, 36, 1, 0, 0, 0);
    run_case("MT2 TZ4 exact", 96, 96, 16, 12, 20, 1, 0, 0, 0);
    run_case("MT2 TZ2 exact", 96, 160, 6, 9, 17, 1, 0, 0, 0);
    run_case("output SR exact", 128, 64, 6, 10, 18, 1, 0, 0x1234567u, 0);
    run_case("split input exact", 160, 64, 6, 10, 18, 1, 0, 0, 96);
    run_case("GN input (prenorm)", 128, 128, 6, 12, 20, 0, 1, 0, 0);
    run_case("GN input one tile", 128, 64, 6, 12, 20, 0, 1, 0, 0);
    run_case("random masked", 192, 128, 6, 12, 20, 0, 0, 0, 0);
    {   /* nn_mask24m: mode 2 equals the host pair-wise mask; mode 3 keeps 2 blocks per grid row / column and its flip is pair-wise 4:8 along
           the backward-data's reduction (co); SR-STE adds lambda w on exactly the pruned weights */
        const int Co = 64, Ci = 96; const size_t nw = (size_t)Co * Ci * 27;
        float *hw = malloc(nw * 4), *hm = malloc(nw * 4), *hr = malloc(nw * 4), *hg = malloc(nw * 4);
        for (size_t i = 0; i < nw; i++) hw[i] = frand();
        float *w = nn_malloc(nw * 4), *o = nn_malloc(nw * 4), *g = nn_malloc(nw * 4); nn_h2d(w, hw, nw * 4);
        nn_mask24m(w, o, Co, Ci, 27, 2); nn_d2h(hm, o, nw * 4);
        memcpy(hr, hw, nw * 4); host_mask48(hr, Co, Ci);
        check("nn_mask24m mode 2 == host pair-wise 4:8 mask", !memcmp(hm, hr, nw * 4));
        nn_mask24m(w, o, Co, Ci, 27, 3); nn_d2h(hm, o, nw * 4);
        int okr = 1, okt = 1; size_t kept = 0;
        for (int t = 0; t < 27; t++) for (int ob = 0; ob < Co; ob += 8) for (int cb = 0; cb < Ci; cb += 8) {
            int rc[4] = {0}, cc[4] = {0};
            for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++) {
                int nz = 0; for (int x = 0; x < 2; x++) for (int y = 0; y < 2; y++) nz += hm[((size_t)(ob + 2 * a + x) * Ci + cb + 2 * b + y) * 27 + t] != 0.f;
                if (nz) { rc[a]++; cc[b]++; kept += nz; }
                if (nz && nz != 4) okt = 0;
            }
            for (int a = 0; a < 4; a++) if (rc[a] != 2 || cc[a] != 2) okr = 0;
        }
        check("nn_mask24m mode 3: whole 2x2 blocks, 2 per grid row and column", okr && okt && kept * 2 == nw);
        nn_zero(g, nw * 4); nn_srste24m(g, w, Co, Ci, 27, 0.5f, 3); nn_d2h(hg, g, nw * 4);
        int oks = 1; for (size_t i = 0; i < nw; i++) if (hg[i] != (hm[i] != 0.f ? 0.f : 0.5f * hw[i])) oks = 0;
        check("nn_srste24m mode 3: lambda w on the pruned weights only", oks);
        /* mode 3 backward-data: the flipped weights wt[ci][co][26 - t] through the sparse kernel with the gradient path's output SR */
        float *ht = malloc(nw * 4);
        for (int co = 0; co < Co; co++) for (int ci = 0; ci < Ci; ci++) for (int t = 0; t < 27; t++) ht[((size_t)ci * Co + co) * 27 + 26 - t] = hm[((size_t)co * Ci + ci) * 27 + t];
        float *wt = nn_malloc(nw * 4); nn_h2d(wt, ht, nw * 4);
        shape5 gs = {1, Co, 6, 12, 20}, xs = {1, Ci, 6, 12, 20};
        const size_t ng = shape_numel(gs); float *hgy = malloc(ng * 4); for (size_t i = 0; i < ng; i++) hgy[i] = frand();
        void *gy = mx4_dev(hgy, gs), *gx1 = mx4_out(xs), *gx2 = mx4_out(xs);
        gnp_t none = {0}; split_t sp = {0}; sp.osr = 0xabcdefu;
        lp_sp24_force(0); lp_conv_fwd_f4(gy, 4, gs, wt, nullptr, Ci, gx1, 4, none, nullptr, 0, sp);
        lp_sp24_force(1); lp_sp24_err_clear(); const int took = lp_sp24_conv_fwd(gy, 4, gs, wt, nullptr, Ci, gx2, 4, none, nullptr, 0, sp); lp_sp24_force(-1);
        check("mode 3 backward-data: Co 64 is <= 64 input channels for the flip: dense (not eligible)", took == 0);
        free(ht); free(hgy); nn_free(wt); nn_free(gy); nn_free(gx1); nn_free(gx2);
        free(hw); free(hm); free(hr); free(hg); nn_free(w); nn_free(o); nn_free(g);
    }
    {   /* mode 3 backward-data at > 64 channels: flipped transposable weights carry the pattern (no truncation), sparse ~ dense */
        const int Co = 128, Ci = 96; const size_t nw = (size_t)Co * Ci * 27;
        float *hw = malloc(nw * 4), *hm = malloc(nw * 4), *ht = malloc(nw * 4);
        for (size_t i = 0; i < nw; i++) hw[i] = frand() * 0.05f;
        float *w = nn_malloc(nw * 4), *wt = nn_malloc(nw * 4); nn_h2d(w, hw, nw * 4);
        nn_mask24m(w, w, Co, Ci, 27, 3); nn_d2h(hm, w, nw * 4);
        for (int co = 0; co < Co; co++) for (int ci = 0; ci < Ci; ci++) for (int t = 0; t < 27; t++) ht[((size_t)ci * Co + co) * 27 + 26 - t] = hm[((size_t)co * Ci + ci) * 27 + t];
        nn_h2d(wt, ht, nw * 4);
        shape5 gs = {1, Co, 6, 12, 20}, xs = {1, Ci, 6, 12, 20};
        const size_t ng = shape_numel(gs), nx = shape_numel(xs); float *hgy = malloc(ng * 4); for (size_t i = 0; i < ng; i++) hgy[i] = frand();
        void *gy = mx4_dev(hgy, gs), *gx1 = mx4_out(xs), *gx2 = mx4_out(xs);
        gnp_t none = {0}; split_t sp = {0}; sp.osr = 0xabcdefu;
        lp_sp24_force(0); lp_conv_fwd_f4(gy, 4, gs, wt, nullptr, Ci, gx1, 4, none, nullptr, 0, sp);
        lp_sp24_force(1); lp_sp24_err_clear(); const int took = lp_sp24_conv_fwd(gy, 4, gs, wt, nullptr, Ci, gx2, 4, none, nullptr, 0, sp); lp_sp24_force(-1);
        nn_sync();
        float *a = deq(gx1, xs), *b = deq(gx2, xs); double d2 = 0, r2 = 0; for (size_t i = 0; i < nx; i++) { d2 += (double)(a[i] - b[i]) * (a[i] - b[i]); r2 += (double)a[i] * a[i]; }
        char nm[128]; snprintf(nm, sizeof nm, "mode 3 backward-data (flipped 128 -> 96): sparse ran, no truncation, vs dense %.2g rel", sqrt(d2 / r2));
        check(nm, took == 1 && lp_sp24_err() == 0 && sqrt(d2 / r2) < 0.02);
        /* routing: conv_fwd_tc (nn_conv3d_fwd on MX-fp4 tensors) runs the sparse kernel for registered weights, the dense one otherwise */
        nn_set_prec(3);
        void *x = mx4_dev(hgy, gs), *y1 = mx4_out(xs), *y2 = mx4_out(xs), *y3 = mx4_out(xs);
        lp_sp24_reg_clear(); nn_conv3d_fwd(x, gs, wt, nullptr, Ci, 3, 1, y1);
        lp_sp24_reg(wt, nw, 2); nn_conv3d_fwd(x, gs, wt, nullptr, Ci, 3, 1, y2); lp_sp24_reg_clear();
        lp_sp24_force(1); lp_sp24_conv_fwd(x, 4, gs, wt, nullptr, Ci, y3, 4, none, nullptr, 0, (split_t){0}); lp_sp24_force(-1);
        const size_t nb = nn_mx4_bytes(xs); uint8_t *b1 = malloc(nb), *b2 = malloc(nb), *b3 = malloc(nb);
        nn_d2h(b1, y1, nb); nn_d2h(b2, y2, nb); nn_d2h(b3, y3, nb);
        check("routing: registered weights take the sparse kernel (bytes == direct sparse call)", !memcmp(b2, b3, nb));
        check("routing: unregistered weights take the dense kernel (bytes differ from sparse)", memcmp(b1, b3, nb) != 0);
        free(b1); free(b2); free(b3); nn_free(x); nn_free(y1); nn_free(y2); nn_free(y3);
        free(a); free(b); free(hw); free(hm); free(ht); free(hgy); nn_free(w); nn_free(wt); nn_free(gy); nn_free(gx1); nn_free(gx2);
    }
    {   /* unmasked weights: the prep must count truncated groups */
        shape5 xs = {1, 128, 4, 8, 16}, ys = {1, 64, 4, 8, 16};
        const size_t nx = shape_numel(xs), nw = (size_t)64 * 128 * 27;
        float *hx = malloc(nx * 4), *hw = malloc(nw * 4);
        for (size_t i = 0; i < nx; i++) hx[i] = frand();
        for (size_t i = 0; i < nw; i++) hw[i] = frand();
        float *w = nn_malloc(nw * 4); nn_h2d(w, hw, nw * 4); void *x = mx4_dev(hx, xs), *y = mx4_out(ys);
        gnp_t gp = {0}; split_t sp = {0};
        lp_sp24_force(1); lp_sp24_err_clear(); lp_sp24_conv_fwd(x, 4, xs, w, nullptr, 64, y, 4, gp, nullptr, 0, sp); lp_sp24_force(-1);
        const unsigned e = lp_sp24_err();
        char nm[96]; snprintf(nm, sizeof nm, "dense weights: prep counts truncated groups (%u)", e); check(nm, e > 0);
        free(hx); free(hw); nn_free(w); nn_free(x); nn_free(y);
    }
    printf("%s\n", bad ? "FAIL" : "ok");
    return bad ? 1 : 0;
}
