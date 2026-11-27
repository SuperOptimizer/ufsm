/* train --topo-lambda: the wrap-count (topology) loss of a head slab (src/nn_loss.cu nn_topo_slab; applied by nn_pgrad_apply).
   (1) the device sums (sum w loss, sum w, count-error weight, sum w dT) against a host reference of the same rays (double; the ray voxels
       are rounded with the same IEEE float ops, so the ray sets are identical and sum w is exact), fp32 and fp16 logits;
   (2) the weight-only call (lc = ls = nullptr, the training stats pass) sums exactly the full call's ray weights and no loss;
   (3) the lattice offsets 0 .. k^3 - 1 together take every ray of the k = 1 lattice exactly once;
   (4) the slab gradient against central differences of the host loss (fp32 logits);
   (5) what it measures: a prediction equal to the label up to noise has no count errors, one with an extra turn (an extra sheet) a mean
       dT near +1 on the rays that cross it, one with a turn missing near -1;
   (6) nn_pgrad_apply adds lambda / sums[1] times the gradient into fp16 logit gradients and zeroes the accumulator. */
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)
static uint64_t rs = 0x9e3779b97f4a7c15ull;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0 / 9007199254740992.0); }

enum { D = 40, H = 48, Wd = 52, Z0 = 6, Z1 = 34, S = 1 };
static int pdiff(int a, int b) { int d = a - b; d -= 252 * (d > 126); d += 252 * (d <= -126); return d; }
static long at(int z, int y, int x, const float *n, int t) {   /* the kernel's topo_at: lrint(v + t n) in float, no contraction */
    volatile float pz = (float)t * n[0], py = (float)t * n[1], px = (float)t * n[2];
    volatile float fz = (float)z + pz, fy = (float)y + py, fx = (float)x + px;
    const int zz = (int)lrintf(fz), yy = (int)lrintf(fy), xx = (int)lrintf(fx);
    if (zz < Z0 || zz >= Z1 || yy < 0 || yy >= H || xx < 0 || xx >= Wd) return -1;
    return ((long)(zz - Z0) * H + yy) * Wd + xx;
}
static double sig(double l) { return 1.0 / (1.0 + exp(-l)); }
typedef struct { double l, w, e, d; long nr; } tsum;
/* the host reference: the same lattice, rays and loss in double */
static tsum topo_host(const float *lc, const float *ls, const uint8_t *ph, const uint8_t *wm, int len, int k, unsigned off) {
    tsum o = {0, 0, 0, 0, 0};
    const size_t HW = (size_t)H * Wd; const uint8_t *sp = ph + (size_t)Z0 * HW;
    const int oz = (int)(off % (unsigned)k), oy = (int)(off / (unsigned)k % (unsigned)k), ox = (int)(off / (unsigned)k / (unsigned)k % (unsigned)k);
    for (int z = Z0; z < Z1; z++) for (int y = 0; y < H; y++) for (int x = 0; x < Wd; x++) {
        if ((z + oz) % k || (y + oy) % k || (x + ox) % k) continue;
        const size_t li = (size_t)(z - Z0) * HW + (size_t)y * Wd + x;
        const double w = wm[li]; if (w <= 0 || sp[li] >= 252) continue;
        if (z < S || z >= D - S || y < S || y >= H - S || x < S || x >= Wd - S) continue;
        const size_t gi = (size_t)Z0 * HW + li, sz = (size_t)S * HW, sy = (size_t)S * Wd;
        const int a6[6] = {ph[gi + sz], ph[gi - sz], ph[gi + sy], ph[gi - sy], ph[gi + S], ph[gi - S]};
        int ok = 1; for (int q = 0; q < 6; q++) ok &= a6[q] < 252; if (!ok) continue;
        const int i0 = pdiff(a6[0], a6[1]), i1 = pdiff(a6[2], a6[3]), i2 = pdiff(a6[4], a6[5]);
        if (!i0 && !i1 && !i2) continue;
        const float gn = sqrtf((float)(i0 * i0 + i1 * i1 + i2 * i2)), n[3] = {(float)i0 / gn, (float)i1 / gn, (float)i2 / gn};
        int nb = 0, nf = 0;
        for (int t = 1; t <= len; t++) { const long q = at(z, y, x, n, -t); if (q < 0 || !wm[q] || sp[q] >= 252) break; nb = t; }
        for (int t = 1; t <= len; t++) { const long q = at(z, y, x, n, t); if (q < 0 || !wm[q] || sp[q] >= 252) break; nf = t; }
        if (nb + nf < len) continue;
        int tg = 0, pv = sp[at(z, y, x, n, -nb)];
        for (int t = -nb + 1; t <= nf; t++) { const int v = sp[at(z, y, x, n, t)]; tg += abs(pdiff(v, pv)); pv = v; }
        o.w += w; o.nr++;
        if (!lc) continue;
        double Tp = 0; long q = at(z, y, x, n, -nb); double ar = 2 * sig(lc[q]) - 1, ai = 2 * sig(ls[q]) - 1;
        for (int t = -nb + 1; t <= nf; t++) { q = at(z, y, x, n, t); const double br = 2 * sig(lc[q]) - 1, bi = 2 * sig(ls[q]) - 1;
            Tp += fabs(atan2(ar * bi - ai * br, ar * br + ai * bi)); ar = br; ai = bi; }
        const double dT = Tp / (2 * M_PI) - tg / 252.0, ad = fabs(dT);
        o.l += w * (ad <= 1 ? 0.5 * dT * dT : ad - 0.5); o.e += ad > 0.5 ? w : 0; o.d += w * dT;
    }
    return o;
}
/* the label: concentric sheets ~10 voxels apart around an axis outside the box (normals mostly in the plane, as in a scroll), a little z tilt */
static double wt(int z, int y, int x) { const double r = sqrt((y + 30.0) * (y + 30.0) + (x - 20.0) * (x - 20.0)); return r / 10.0 + 0.02 * z; }
static double bump(int z, int y, int x, double sgn) {   /* in the half y < 24 only, across a radius band: + one extra turn over ~6 voxels (smoothstep);
                                                          - one turn removed over the 10 voxels of a label turn (linear: the phase stands still there, a missed sheet) */
    (void)z; if (y >= 24 || sgn == 0) return 0; const double r = sqrt((y + 30.0) * (y + 30.0) + (x - 20.0) * (x - 20.0)), u = (r - 52.0) / (sgn > 0 ? 6.0 : 10.0);
    return u <= 0 ? 0 : u >= 1 ? sgn : sgn > 0 ? u * u * (3 - 2 * u) : -u;
}
/* logits of a phasor r e^{2 pi i v}: per channel the logit of (1 + component) / 2 */
static void mk_logits(float *lc, float *ls, double sgn, double noise) {
    for (int z = Z0; z < Z1; z++) for (int y = 0; y < H; y++) for (int x = 0; x < Wd; x++) {
        const size_t i = ((size_t)(z - Z0) * H + y) * Wd + x;
        const double v = wt(z, y, x) + bump(z, y, x, sgn) + noise * (2 * urand() - 1), r = 0.55 + 0.4 * urand();
        const double pc = 0.5 + 0.5 * r * cos(2 * M_PI * v), ps = 0.5 + 0.5 * r * sin(2 * M_PI * v);
        lc[i] = (float)log(pc / (1 - pc)); ls[i] = (float)log(ps / (1 - ps));
    }
}

int main(void) {
    nn_init(0);
    nn_set_logits_h16(0);
    const int len = 8, k = 2;
    const size_t n = (size_t)D * H * Wd, ns = (size_t)(Z1 - Z0) * H * Wd;
    uint8_t *ph = malloc(n), *wm = malloc(ns);
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < Wd; x++) {
        const double f = wt(z, y, x) - floor(wt(z, y, x));
        ph[((size_t)z * H + y) * Wd + x] = (x >= 40 && x < 44 && y < 10) ? 255 : (uint8_t)((int)lround(f * 252) % 252);   /* (an unknown block) */
    }
    for (size_t i = 0; i < ns; i++) { const double u = urand(); wm[i] = u < 0.01 ? 0 : u < 0.1 ? 8 : 1; }   /* (a few masked voxels end rays) */
    float *lc = malloc(ns * 4), *ls = malloc(ns * 4);
    uint8_t *dph = nn_malloc(n), *dwm = nn_malloc(ns); float *dlc = nn_malloc(ns * 4), *dls = nn_malloc(ns * 4), *acc = nn_malloc(2 * ns * 4);
    void *dlc16 = nn_malloc(ns * 2), *dls16 = nn_malloc(ns * 2);
    double *dsum = nn_malloc(8 * sizeof(double));
    nn_h2d(dph, ph, n); nn_h2d(dwm, wm, ns); nn_zero(acc, 2 * ns * 4);
#define RUN(LC, LS, K_, OFF, ACC, OUT) do { nn_zero(dsum, 4 * sizeof(double)); nn_topo_slab((LC), (LS), dph, dwm, D, H, Wd, Z0, Z1, S, len, (K_), (OFF), dsum, (ACC)); nn_sync(); nn_d2h((OUT), dsum, 4 * sizeof(double)); } while (0)
    /* (5) + (1): three predictions */
    static const char *const nm[3] = {"clean", "extra turn", "missing turn"};
    const double sg[3] = {0, 1, -1};
    for (int c = 0; c < 3; c++) {
        mk_logits(lc, ls, sg[c], 0.01);
        nn_h2d(dlc, lc, ns * 4); nn_h2d(dls, ls, ns * 4);
        double g4[4]; RUN(dlc, dls, k, 0, nullptr, g4);
        const tsum h = topo_host(lc, ls, ph, wm, len, k, 0);
        printf("topo %-12s: %ld rays, sum w %.0f (host %.0f), loss %.5f (host %.5f), count errors %.4f (host %.4f), mean dT %+.4f (host %+.4f)\n",
               nm[c], h.nr, g4[1], h.w, g4[0] / g4[1], h.l / h.w, g4[2] / g4[1], h.e / h.w, g4[3] / g4[1], h.d / h.w);
        CHECK(h.nr > 2000, "%s: too few rays (%ld)", nm[c], h.nr);
        CHECK(g4[1] == h.w, "%s: sum w %.1f vs host %.1f (the ray sets differ)", nm[c], g4[1], h.w);
        CHECK(fabs(g4[0] - h.l) <= 1e-4 * h.l + 1e-3, "%s: sum w loss %.6f vs host %.6f", nm[c], g4[0], h.l);
        CHECK(fabs(g4[2] - h.e) <= 8.5, "%s: count-error weight %.1f vs host %.1f", nm[c], g4[2], h.e);   /* (one ray at the 0.5 threshold may flip) */
        CHECK(fabs(g4[3] - h.d) <= 1e-4 * fabs(h.d) + 1e-2, "%s: sum w dT %.6f vs host %.6f", nm[c], g4[3], h.d);
        if (c == 0) CHECK(g4[2] == 0 && fabs(g4[3] / g4[1]) < 0.05, "clean: count errors %.4f, mean dT %.4f", g4[2] / g4[1], g4[3] / g4[1]);
        if (c == 1) CHECK(g4[2] / g4[1] > 0.1 && g4[3] / g4[1] > 0.1, "extra turn: count errors %.4f, mean dT %.4f (want > 0.1 both)", g4[2] / g4[1], g4[3] / g4[1]);
        if (c == 2) CHECK(g4[2] / g4[1] > 0.02 && g4[3] / g4[1] < -0.02, "missing turn: count errors %.4f, mean dT %.4f (want > 0.02, < -0.02; clean: 0, ~0)", g4[2] / g4[1], g4[3] / g4[1]);
        /* fp16 logits: the same rays, the loss on the rounded logits */
        if (c == 1) {
            mk_logits(lc, ls, sg[c], 0.01);
            _Float16 *hc = malloc(ns * 2), *hs = malloc(ns * 2);
            for (size_t i = 0; i < ns; i++) { hc[i] = (_Float16)lc[i]; hs[i] = (_Float16)ls[i]; lc[i] = (float)hc[i]; ls[i] = (float)hs[i]; }
            nn_h2d(dlc16, hc, ns * 2); nn_h2d(dls16, hs, ns * 2);
            nn_set_logits_h16(1); double q4[4]; RUN(dlc16, dls16, k, 0, nullptr, q4); nn_set_logits_h16(0);
            const tsum hq = topo_host(lc, ls, ph, wm, len, k, 0);
            printf("topo fp16 logits: loss %.5f (host %.5f)\n", q4[0] / q4[1], hq.l / hq.w);
            CHECK(q4[1] == hq.w && fabs(q4[0] - hq.l) <= 1e-4 * hq.l + 1e-3, "fp16 logits: sums %.6f %.1f vs host %.6f %.1f", q4[0], q4[1], hq.l, hq.w);
            free(hc); free(hs);
        }
    }
    /* (2) weight-only */
    mk_logits(lc, ls, 1, 0.01); nn_h2d(dlc, lc, ns * 4); nn_h2d(dls, ls, ns * 4);
    double full[4], wonly[4];
    RUN(dlc, dls, k, 5, acc, full);
    RUN(nullptr, nullptr, k, 5, nullptr, wonly);
    printf("topo weight-only: sum w %.0f (full call %.0f), loss sum %.3g\n", wonly[1], full[1], wonly[0]);
    CHECK(wonly[0] == 0 && wonly[2] == 0 && wonly[3] == 0 && wonly[1] == full[1], "weight-only sums %.4f %.1f vs the full call's weight %.1f", wonly[0], wonly[1], full[1]);
    /* (3) the offsets cover the k = 1 lattice once */
    { double one[4], tot = 0; RUN(nullptr, nullptr, 1, 0, nullptr, one);
      for (unsigned o = 0; o < (unsigned)(k * k * k); o++) { double q[4]; RUN(nullptr, nullptr, k, o, nullptr, q); tot += q[1]; }
      printf("topo lattice: %u offsets of k %d sum w %.0f, k 1 %.0f\n", (unsigned)(k * k * k), k, tot, one[1]);
      CHECK(tot == one[1], "the k %d offsets sum w %.1f, k 1 %.1f", k, tot, one[1]); }
    /* (4) central differences of the host loss (offset 5, the extra-turn prediction: rays in both huber regimes) */
    float *g = malloc(2 * ns * 4); nn_d2h(g, acc, 2 * ns * 4);
    int checked = 0; double worst = 0;
    for (size_t q = 3; q < ns && checked < 40; q += 37) {
        for (int ch = 0; ch < 2; ch++) {
            float *L = ch ? ls : lc; const float gk = g[(size_t)ch * ns + q];
            if (fabsf(gk) < 1e-3f) continue;
            const float hh = 1e-3f, v0 = L[q];
            L[q] = v0 + hh; const double fp = topo_host(lc, ls, ph, wm, len, k, 5).l;
            L[q] = v0 - hh; const double fm = topo_host(lc, ls, ph, wm, len, k, 5).l;
            L[q] = v0;
            const double fd = (fp - fm) / (2.0 * ((double)(v0 + hh) - (double)v0)), e = fabs(fd - gk) / fmax(fabs(gk), 1e-2);
            if (e > worst) worst = e;
            CHECK(e < 2e-2, "gradient at voxel %zu channel %d: %.6f vs central difference %.6f", q, ch, gk, fd);
            checked++;
        }
    }
    printf("topo: %d gradients against central differences, worst relative error %.4f\n", checked, worst);
    CHECK(checked >= 20, "only %d gradients checked", checked);
    /* (6) apply into fp16 gradients */
    nn_set_f16(1); nn_set_loss_grad_h16(1);
    uint16_t *gz = calloc(ns, 2); void *dgc = nn_malloc(ns * 2), *dgs = nn_malloc(ns * 2);
    nn_h2d(dgc, gz, ns * 2); nn_h2d(dgs, gz, ns * 2);
    nn_h2d(dsum, full, 4 * sizeof(double));
    const float lambda = 0.5f;
    nn_pgrad_apply(dgc, dgs, acc, ns, dsum, lambda);
    nn_sync();
    _Float16 *hc = malloc(ns * 2), *hs = malloc(ns * 2); nn_d2h(hc, dgc, ns * 2); nn_d2h(hs, dgs, ns * 2);
    float *ac = malloc(2 * ns * 4); nn_d2h(ac, acc, 2 * ns * 4);
    double we = 0; size_t nz = 0, nnz = 0;
    for (size_t q = 0; q < ns; q++) for (int ch = 0; ch < 2; ch++) {
        const double want = (double)lambda * nn_get_grad_scale() / full[1] * g[(size_t)ch * ns + q], got = (double)(ch ? hs[q] : hc[q]);
        we = fmax(we, fabs(got - want) / (fabs(want) + 1e-4)); nnz += want != 0;
        nz += ac[(size_t)ch * ns + q] != 0.f;
    }
    printf("topo apply: %zu gradients, worst relative error %.5f (fp16), %zu accumulator entries left\n", nnz, we, nz);
    CHECK(nnz > 1000 && we < 2e-3, "applied gradient off by %.5f (%zu entries)", we, nnz);
    CHECK(nz == 0, "the accumulator was not zeroed (%zu entries)", nz);
    if (fails) { fprintf(stderr, "test_topo: %d FAILED\n", fails); return 1; }
    printf("test_topo: OK\n");
    return 0;
}
