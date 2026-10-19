/* train --task unwrap2: the relative winding mod 4 targets (src/band.c band_wind4) and the gauge-free loss (src/nn_loss.cu
   nn_wind4_*).
   Targets: a planar phase field cut by an unknown wall (two components, each its own gauge), an off-axis spiral window and a window
   around the axis; every kept voxel's w4 must equal W_true mod 4 up to one integer per component (and the 1/63-turn code).
   Loss: the value against a double-precision reference (per component the minimum over the four quarter-turn gauges of the soft BCE),
   gauge invariance (shifting a component's targets by whole turns changes neither the loss nor the gradient), the gradient against
   central differences, the turn accuracy, fp16 gradients, and the masking (unsupervised voxels get a zero gradient). */
#include "band.h"
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

/* phase codes from W (unknown where W is NaN) */
static void phase_of(const double *W, size_t n, uint8_t *ph) {
    for (size_t i = 0; i < n; i++) {
        if (isnan(W[i])) { ph[i] = PHASE_UNKNOWN; continue; }
        const double f = W[i] - floor(W[i]); ph[i] = (uint8_t)((int)lround(f * 252) % 252);
    }
}
/* every kept voxel: (W_true - w4 / 63) mod 4 is one integer per component (|deviation| <= tol turn); returns the kept fraction */
static double check_targets(const char *name, const double *W, const uint8_t *w4, const uint8_t *cp, size_t n, int ncomp, double tol) {
    double g[256]; int seen[256] = {0}; size_t kept = 0, known = 0, bad = 0; double worst = 0;
    for (size_t i = 0; i < n; i++) {
        if (!isnan(W[i])) known++;
        if (cp[i] == 255) { CHECK(w4[i] == 255, "%s: voxel %zu w4 %d without a component", name, i, w4[i]); continue; }
        CHECK(cp[i] < ncomp && w4[i] < 252, "%s: voxel %zu component %d of %d, w4 %d", name, i, cp[i], ncomp, w4[i]);
        if (cp[i] >= 255) continue;
        kept++;
        double e = fmod(W[i] - w4[i] / 63.0, 4.0); if (e < 0) e += 4;
        if (!seen[cp[i]]) { seen[cp[i]] = 1; g[cp[i]] = round(e); }
        double d = e - g[cp[i]]; d -= 4 * round(d / 4);
        if (fabs(d) > worst) worst = fabs(d);
        bad += fabs(d) > tol;
    }
    printf("%s: %d components, %zu of %zu known voxels kept, worst |W - gauge - w4| %.4f turn, %zu beyond %.3f\n", name, ncomp, kept, known, worst, bad, tol);
    CHECK(bad == 0, "%s: %zu voxels off their component's gauge", name, bad);
    return known ? (double)kept / known : 0;
}

/* ---- the loss ---- */
static double sp(double x) { return fmax(x, 0) + log1p(exp(-fabs(x))); }
/* reference: per component min over g of sum m [bce(a, 0.5 + 0.5 cos(t + g pi / 2)) + bce(b, ... sin)], / (2 sum m) */
static double ref_loss(const float *la, const float *lb, const uint8_t *w4, const uint8_t *cp, const uint8_t *m, size_t n, double *acc) {
    double L[255][4] = {{0}}, N[255] = {0}, A[255][4] = {{0}}, Nt = 0, Lt = 0, At = 0;
    for (size_t i = 0; i < n; i++) {
        if (cp[i] >= 255 || w4[i] >= 252 || !m[i]) continue;
        const double t = 2 * M_PI * w4[i] / 252.0, a = la[i], b = lb[i], w = m[i];
        N[cp[i]] += w;
        const double pa = 2 / (1 + exp(-a)) - 1, pb = 2 / (1 + exp(-b)) - 1, e = atan2(pb * cos(t) - pa * sin(t), pa * cos(t) + pb * sin(t));
        const int k = (((int)lround(e / (M_PI / 2))) % 4 + 4) % 4;
        A[cp[i]][k] += w;
        for (int g = 0; g < 4; g++) {
            const double pc = 0.5 + 0.5 * cos(t + g * M_PI / 2), ps = 0.5 + 0.5 * sin(t + g * M_PI / 2);
            L[cp[i]][g] += w * (sp(a) - pc * a + sp(b) - ps * b);
        }
    }
    for (int c = 0; c < 255; c++) {
        if (N[c] <= 0) continue;
        int gb = 0; for (int g = 1; g < 4; g++) if (L[c][g] < L[c][gb]) gb = g;
        Nt += N[c]; Lt += L[c][gb]; At += A[c][gb];
    }
    *acc = Nt > 0 ? At / Nt : 0;
    return Nt > 0 ? Lt / (2 * Nt) : 0;
}
typedef struct { float *la, *lb; uint8_t *w4, *cp, *m; double *ds; float *fin, *ga, *gb; size_t n; } dev_t4;
static dev_t4 dev_alloc(size_t n) {
    dev_t4 d = {nn_malloc(n * 4), nn_malloc(n * 4), nn_malloc(n), nn_malloc(n), nn_malloc(n), nn_malloc(nn_wind4_sums()), nn_malloc(nn_wind4_scratch()), nn_malloc(n * 4), nn_malloc(n * 4), n};
    return d;
}
static void dev_free(dev_t4 *d) { nn_free(d->la); nn_free(d->lb); nn_free(d->w4); nn_free(d->cp); nn_free(d->m); nn_free(d->ds); nn_free(d->fin); nn_free(d->ga); nn_free(d->gb); }
/* the GPU loss (fp32 logits and gradient); fin[8] on the host; ga / gb host (n each) or nullptr */
static void gpu_loss(dev_t4 *d, const float *la, const float *lb, const uint8_t *w4, const uint8_t *cp, const uint8_t *m, float lambda, float *fin, float *ga, float *gb) {
    const size_t n = d->n;
    nn_h2d(d->la, la, n * 4); nn_h2d(d->lb, lb, n * 4); nn_h2d(d->w4, w4, n); nn_h2d(d->cp, cp, n); nn_h2d(d->m, m, n);
    nn_wind4_loss(d->la, d->lb, d->w4, d->cp, d->m, n, d->ds, d->fin, lambda, ga ? d->ga : nullptr, ga ? d->gb : nullptr);
    nn_sync(); nn_d2h(fin, d->fin, 8 * sizeof(float));
    if (ga) { nn_d2h(ga, d->ga, n * 4); nn_d2h(gb, d->gb, n * 4); }
}
static uint64_t rs = 88172645463325252ull;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 11) * (1.0 / 9007199254740992.0); }
static double nrand(void) { const double u = urand() + 1e-12, v = urand(); return sqrt(-2 * log(u)) * cos(2 * M_PI * v); }

int main(void) {
    nn_init(0);
    /* (1) planar W = k . x, an unknown wall at x = 28 (two components), a small island (dropped: under minsize) */
    {
        const int D = 32, H = 40, W = 56; const size_t n = (size_t)D * H * W;
        double *Wt = malloc(n * sizeof *Wt); uint8_t *ph = malloc(n), *w4 = malloc(n), *cp = malloc(n);
        for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            const size_t i = ((size_t)z * H + y) * W + x;
            Wt[i] = 0.013 * z - 0.021 * y + 0.083 * x + 3.7;   /* ~12 voxels per turn along x: 4.6 turns over the window */
            if (x == 28 || (x >= 50 && y < 2)) Wt[i] = NAN;
        }
        phase_of(Wt, n, ph);
        band_wind4_stats st;
        const int nc = band_wind4(ph, D, H, W, 80, 512, 15, w4, cp, &st);
        CHECK(nc == 2, "planar: %d components (2 expected)", nc);
        const double f = check_targets("planar", Wt, w4, cp, n, nc, 1.0 / 63 + 1e-3);
        CHECK(f > 0.999, "planar: kept %.4f", f);
        /* the same field with every winding shifted by 7 turns (another label set's numbering): the phase, so the targets, are identical */
        double *W2 = malloc(n * sizeof *W2); uint8_t *ph2 = malloc(n), *w42 = malloc(n), *cp2 = malloc(n);
        for (size_t i = 0; i < n; i++) W2[i] = Wt[i] + 7;
        phase_of(W2, n, ph2); band_wind4(ph2, D, H, W, 80, 512, 15, w42, cp2, nullptr);
        CHECK(!memcmp(w4, w42, n) && !memcmp(cp, cp2, n), "planar: a global winding offset changed the targets");
        free(W2); free(ph2); free(w42); free(cp2);
        free(Wt); free(ph); free(w4); free(cp);
    }
    /* (2) spiral W = r / p - theta / 2 pi: a window beside the axis (one component), and one around it (the loop around the axis
       gains a turn: inconsistent joins, the component dropped or the voxels still consistent with one gauge where kept) */
    for (int around = 0; around < 2; around++) {
        const int D = 24, H = 64, W = 64; const size_t n = (size_t)D * H * W;
        const double cy = around ? 31.3 : -40.2, cx = around ? 30.7 : 33.1, p = 9.0;
        double *Wt = malloc(n * sizeof *Wt); uint8_t *ph = malloc(n), *w4 = malloc(n), *cp = malloc(n);
        for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            const double dy = y - cy, dx = x - cx, r = hypot(dy, dx);
            Wt[((size_t)z * H + y) * W + x] = r < 4 ? NAN : r / p - atan2(dy, dx) / (2 * M_PI);
        }
        phase_of(Wt, n, ph);
        band_wind4_stats st;
        const int nc = band_wind4(ph, D, H, W, 80, 512, 15, w4, cp, &st);
        printf("spiral %s: %d components, %zu known, %zu kept, %zu dropped as inconsistent, %zu runs\n", around ? "around the axis" : "beside the axis", nc, st.known, st.kept, st.dropped_conflict, st.runs);
        if (!around) {
            CHECK(nc == 1, "spiral beside the axis: %d components", nc);
            const double f = check_targets("spiral beside the axis", Wt, w4, cp, n, nc, 1.0 / 63 + 1e-3);
            CHECK(f > 0.999, "spiral beside the axis: kept %.4f", f);
        } else {
            CHECK(st.dropped_conflict > 0 && st.kept < st.known / 2, "spiral around the axis: the inconsistent loop was not dropped (kept %zu of %zu)", st.kept, st.known);
            /* per tile of 16: only the tiles whose loops enclose the axis are dropped, each kept tile its own gauge */
            const int nt = band_wind4_tiled(ph, D, H, W, 16, 80, 512, 1, w4, cp, &st);
            printf("spiral around the axis, tiles of 16: %d components, %zu of %zu kept, %zu dropped\n", nt, st.kept, st.known, st.dropped_conflict);
            const double f = check_targets("spiral around the axis, tiles of 16", Wt, w4, cp, n, nt, 1.0 / 63 + 1e-3);
            CHECK(nt >= 24 && f > 0.85, "tiles: %d components, kept %.3f", nt, f);
        }
        free(Wt); free(ph); free(w4); free(cp);
    }
    /* (3) the loss: value vs the reference, gauge invariance, gradient vs central differences, accuracy, masking */
    {
        const size_t n = 96; dev_t4 d = dev_alloc(n);
        float *la = malloc(n * 4), *lb = malloc(n * 4), *ga = malloc(n * 4), *gb = malloc(n * 4), *ga2 = malloc(n * 4), *gb2 = malloc(n * 4);
        uint8_t *w4 = malloc(n), *cp = malloc(n), *m = malloc(n), *w4s = malloc(n);
        const int gtrue[3] = {1, 3, 0};   /* the net's gauge per component: its prediction is the target rotated by g quarter turns */
        for (size_t i = 0; i < n; i++) {
            w4[i] = (uint8_t)(urand() * 252); cp[i] = (uint8_t)(i % 3); m[i] = (uint8_t)(1 + (i % 5 == 0));
            if (i % 17 == 3) cp[i] = 255;   /* no component */
            if (i % 13 == 5) m[i] = 0;      /* masked */
            if (i % 19 == 7) w4[i] = 255;   /* unknown */
            const double t = 2 * M_PI * (w4[i] % 252) / 252.0 + (cp[i] < 3 ? gtrue[cp[i]] : 0) * M_PI / 2;
            la[i] = (float)(2.5 * cos(t) + 0.6 * nrand()); lb[i] = (float)(2.5 * sin(t) + 0.6 * nrand());
        }
        const float lambda = 0.7f; float fin[8]; double racc;
        gpu_loss(&d, la, lb, w4, cp, m, lambda, fin, ga, gb);
        const double rl = ref_loss(la, lb, w4, cp, m, n, &racc);
        printf("loss: gpu %.6f reference %.6f (worst gauge %.6f), accuracy gpu %.4f reference %.4f, %g voxels, %g components\n", fin[0], rl, fin[5], fin[1], racc, fin[2], fin[4]);
        CHECK(fabs(fin[0] - rl) < 1e-5 * fmax(1, fabs(rl)), "loss %.7f vs reference %.7f", fin[0], rl);
        CHECK(fabs(fin[1] - racc) < 1e-6, "accuracy %.5f vs reference %.5f", fin[1], racc);
        CHECK(fin[1] > 0.8 && fin[5] > fin[0] + 0.1, "the net's gauge not found: accuracy %.3f, best %.4f worst %.4f", fin[1], fin[0], fin[5]);
        float f8[15 + 8]; nn_d2h(f8, d.fin, sizeof f8);
        for (int c = 0; c < 3; c++) CHECK((int)f8[8 + c] == gtrue[c], "component %d: gauge %d, the net's %d", c, (int)f8[8 + c], gtrue[c]);
        /* gauge invariance: each component's targets shifted by a different number of whole turns (63 codes per turn) */
        const int shift[3] = {2, 1, 3};
        for (size_t i = 0; i < n; i++) w4s[i] = w4[i] < 252 && cp[i] < 3 ? (uint8_t)((w4[i] + 63 * shift[cp[i]]) % 252) : w4[i];
        float fin2[8]; gpu_loss(&d, la, lb, w4s, cp, m, lambda, fin2, ga2, gb2);
        double gd = 0; for (size_t i = 0; i < n; i++) gd = fmax(gd, fmax(fabs(ga[i] - ga2[i]), fabs(gb[i] - gb2[i])));
        printf("gauge shift: loss %.7f -> %.7f, accuracy %.4f -> %.4f, max gradient difference %.2e\n", fin[0], fin2[0], fin[1], fin2[1], gd);
        CHECK(fabs(fin[0] - fin2[0]) < 1e-6 && fabs(fin[1] - fin2[1]) < 1e-6 && gd < 1e-7, "a whole-turn shift of the targets changed the loss / gradient");
        /* masking: unsupervised voxels get exactly 0 */
        int mz = 1; for (size_t i = 0; i < n; i++) if ((cp[i] >= 3 || !m[i] || w4[i] >= 252) && (ga[i] != 0.f || gb[i] != 0.f)) mz = 0;
        CHECK(mz, "unsupervised voxels with a nonzero gradient");
        /* central differences of lambda * loss (double reference: the same gauges away from ties) */
        double worst = 0, gmax = 0;
        for (size_t i = 0; i < n; i++) for (int ch = 0; ch < 2; ch++) {
            float *l = ch ? lb : la; const float x0 = l[i], e = 1e-3f; double a1, lp, lm;
            l[i] = x0 + e; lp = ref_loss(la, lb, w4, cp, m, n, &a1); l[i] = x0 - e; lm = ref_loss(la, lb, w4, cp, m, n, &a1); l[i] = x0;
            const double fd = lambda * (lp - lm) / (2 * e), an = ch ? gb[i] : ga[i];
            worst = fmax(worst, fabs(fd - an)); gmax = fmax(gmax, fabs(fd));
        }
        printf("gradient vs central differences: max |error| %.3e of max |gradient| %.3e\n", worst, gmax);
        CHECK(worst < 1e-3 * gmax + 1e-7, "gradient error %.3e (max %.3e)", worst, gmax);
        /* fp16 gradient storage (the trainer's in-place logit gradient) within fp16 rounding of the fp32 one, times the gradient scale */
        nn_set_f16(1); nn_set_loss_grad_h16(1);
        nn_wind4_loss(d.la, d.lb, d.w4, d.cp, d.m, n, d.ds, d.fin, lambda, d.ga, d.gb); nn_sync();
        _Float16 *h = malloc(n * 2); nn_d2h(h, d.ga, n * 2); nn_set_loss_grad_h16(0); nn_set_f16(0);
        nn_h2d(d.w4, w4, n); gpu_loss(&d, la, lb, w4, cp, m, lambda, fin, ga, gb);
        double rlo = 1e30, rhi = 0; size_t nz16 = 0;   /* g16 = gradient scale x g32: one ratio up to fp16 rounding */
        for (size_t i = 0; i < n; i++) {
            if (ga[i] == 0.f) { nz16 += h[i] != 0; continue; }
            if (fabs(ga[i]) < 1e-3 * gmax) continue;
            const double r = (double)h[i] / ga[i]; rlo = fmin(rlo, r); rhi = fmax(rhi, r);
        }
        printf("fp16 gradient: ratio to fp32 %.4g .. %.4g\n", rlo, rhi);
        CHECK(nz16 == 0 && rlo > 0 && rhi / rlo < 1.01, "fp16 gradient: ratio %.4g .. %.4g, %zu nonzero where fp32 is 0", rlo, rhi, nz16);
        free(la); free(lb); free(ga); free(gb); free(ga2); free(gb2); free(w4); free(cp); free(m); free(w4s); dev_free(&d);
    }
    if (fails) { fprintf(stderr, "test_wind4: %d failure(s)\n", fails); return 1; }
    printf("test_wind4: OK\n");
    return 0;
}
