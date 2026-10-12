/* task papyrus label generation (src/papyrus.h): smoothing, cutoffs, soft targets, ignore band, cleanup, Otsu, checkpoint tag */
#include "papyrus.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)
#define IDX(z, y, x) (((size_t)(z) * P + (y)) * P + (x))

int main(void) {
    enum { P = 24 };
    const size_t p3 = (size_t)P * P * P;
    uint8_t *ct = malloc(p3), *t = malloc(p3), *ign = calloc(p3, 1), *fg = malloc(p3);
    void *scratch = malloc(8 * p3);
    float *A = malloc(4 * p3), *B = malloc(4 * p3);

    /* taps: normalised and symmetric */
    { float w[2 * PAPYRUS_RMAX + 1]; const int r = papyrus_taps(1.f, w); double s = 0; for (int k = 0; k <= 2 * r; k++) s += w[k];
      CHECK(r == 3 && fabs(s - 1) < 1e-6 && fabsf(w[0] - w[6]) < 1e-7 && fabsf(w[3] - 0.39905f) < 1e-4, "taps r %d sum %g centre %g", r, s, w[3]);
      CHECK(papyrus_taps(0.f, w) == 0 && w[0] == 1.f, "sigma 0 is the identity"); }

    /* smoothing: a constant stays constant (clamped edges), an impulse spreads as the separable Gaussian and keeps its mass */
    memset(ct, 77, p3); papyrus_smooth(ct, P, 1.5f, A);
    { double e = 0; for (size_t k = 0; k < p3; k++) e = fmax(e, fabs(A[k] - 77.0)); CHECK(e < 1e-3, "constant cube changed by %g", e); }
    memset(ct, 0, p3); ct[IDX(12, 12, 12)] = 200; papyrus_smooth(ct, P, 1.f, A);
    { float w[2 * PAPYRUS_RMAX + 1]; papyrus_taps(1.f, w); double m = 0; for (size_t k = 0; k < p3; k++) m += A[k];
      CHECK(fabs(m - 200) < 1e-2, "impulse mass %g", m);
      CHECK(fabsf(A[IDX(12, 12, 12)] - 200 * w[3] * w[3] * w[3]) < 1e-3, "impulse peak %g", A[IDX(12, 12, 12)]);
      CHECK(fabsf(A[IDX(12, 12, 13)] - A[IDX(12, 13, 12)]) < 1e-5 && fabsf(A[IDX(11, 12, 12)] - A[IDX(12, 12, 11)]) < 1e-5, "impulse not isotropic"); }

    /* hard cutoff on the raw CT: a step 40 | 120 along x, the outside mask (CT 0) background */
    papyrus_cfg c = {.t_lo = 50, .t_hi = 128};
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) ct[IDX(z, y, x)] = z == 0 ? 0 : x < P / 2 ? 40 : 120;
    papyrus_labels(ct, P, &c, 80.f, 0.5, t, ign, fg, scratch);
    { int bad = 0; for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) bad += t[IDX(z, y, x)] != (z && x >= P / 2 ? 255 : 0);
      CHECK(!bad, "hard cutoff 80: %d wrong voxels", bad); }
    papyrus_labels(ct, P, &c, 130.f, 0.5, t, ign, fg, scratch);
    { int any = 0; for (size_t k = 0; k < p3; k++) any |= t[k]; CHECK(!any, "cutoff above every value labels papyrus"); }
    papyrus_labels(ct, P, &c, 30.f, 0.5, t, ign, fg, scratch);
    { int bad = 0; for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) bad += t[IDX(z, y, x)] != (z ? 255 : 0);
      CHECK(!bad, "cutoff below every value: %d wrong (CT 0 must stay background)", bad); }

    /* soft targets: 255 sigmoid((s - t) / tau), tau from u_tau; ignore band |s - t| < band */
    c.tau_lo = 2; c.tau_hi = 6;   /* u_tau 0.5 -> tau 4 */
    for (size_t k = 0; k < p3; k++) ct[k] = (uint8_t)(70 + k % 21);   /* 70 .. 90 */
    papyrus_labels(ct, P, &c, 80.f, 0.5, t, ign, fg, scratch);
    { int bad = 0; for (size_t k = 0; k < p3; k++) { const double e = 255.0 / (1.0 + exp(-((double)ct[k] - 80.0) / 4.0)); bad += fabs(t[k] - e) > 2.5; }
      CHECK(!bad, "soft targets: %d voxels off the sigmoid", bad);
      size_t k80 = 0; while (ct[k80] != 80) k80++; CHECK(t[k80] == 128, "soft target at the cutoff %d", t[k80]);
      int ig = 0; for (size_t k = 0; k < p3; k++) ig += ign[k]; CHECK(!ig, "no band: %d ignored", ig); }
    c.tau_lo = c.tau_hi = 0; c.band = 5;
    papyrus_labels(ct, P, &c, 80.f, 0.5, t, ign, fg, scratch);
    { int bad = 0; for (size_t k = 0; k < p3; k++) bad += ign[k] != (abs((int)ct[k] - 80) < 5) || t[k] != (ct[k] > 80 ? 255 : 0);
      CHECK(!bad, "band 5: %d voxels with the wrong ignore / target", bad); }
    memset(ign, 0, p3); c.band = 0;

    /* smoothing in the labels: a one-voxel bright plane (200 on 40) is papyrus below the smoothed peak 40 + 160 w0, not above */
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) ct[IDX(z, y, x)] = x == 12 ? 200 : 40;
    c.sigma = 1.f;
    papyrus_labels(ct, P, &c, 100.f, 0.5, t, ign, fg, scratch);
    { int bad = 0; for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) bad += t[IDX(z, y, x)] != (x == 12 ? 255 : 0);
      CHECK(!bad, "smoothed plane at cutoff 100 (peak 103.8): %d wrong", bad); }
    papyrus_labels(ct, P, &c, 105.f, 0.5, t, ign, fg, scratch);
    { int any = 0; for (size_t k = 0; k < p3; k++) any |= t[k]; CHECK(!any, "smoothed plane at cutoff 105 still papyrus"); }
    papyrus_labels(ct, P, &c, 75.f, 0.5, t, ign, fg, scratch);   /* first neighbours 40 + 160 w1 = 78.7 */
    { int bad = 0; for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) bad += t[IDX(z, y, x)] != (abs(x - 12) <= 1 ? 255 : 0);
      CHECK(!bad, "smoothed plane at cutoff 75: %d wrong", bad); }
    c.sigma = 0;

    /* cleanup: a papyrus speck in the void and a void hole in a papyrus block flip; a small component on a face stays */
    memset(ct, 30, p3);
    ct[IDX(5, 5, 5)] = 150;                                                   /* speck (1 voxel) */
    for (int z = 10; z < 18; z++) for (int y = 10; y < 18; y++) for (int x = 10; x < 18; x++) ct[IDX(z, y, x)] = 150;   /* block */
    ct[IDX(13, 13, 13)] = 30; ct[IDX(13, 13, 14)] = 30;                       /* hole (2 voxels) */
    ct[IDX(0, 3, 3)] = 150;                                                   /* speck on the face z = 0 */
    c.clean = 8;
    const size_t nf = papyrus_labels(ct, P, &c, 80.f, 0.5, t, ign, fg, scratch);
    CHECK(nf == 3, "cleanup flipped %zu voxels, expected 3", nf);
    CHECK(t[IDX(5, 5, 5)] == 0 && fg[IDX(5, 5, 5)] == 0, "speck kept");
    CHECK(t[IDX(13, 13, 13)] == 255 && t[IDX(13, 13, 14)] == 255, "hole not filled");
    CHECK(t[IDX(0, 3, 3)] == 255, "face component removed");
    CHECK(t[IDX(12, 12, 12)] == 255 && t[IDX(2, 2, 2)] == 0, "cleanup changed large components");
    c.clean = 600;   /* the block (512 - 2 voxels) is below it now: it goes, the void around it (touching the faces) stays */
    papyrus_labels(ct, P, &c, 80.f, 0.5, t, ign, fg, scratch);
    CHECK(t[IDX(12, 12, 12)] == 0 && t[IDX(13, 13, 13)] == 0 && t[IDX(0, 3, 3)] == 255, "large cleanup threshold");
    c.clean = 0;

    /* Otsu of a two-mode histogram, and the relative cutoff */
    for (size_t k = 0; k < p3; k++) ct[k] = k / ((size_t)P * P) < 8 ? 0 : (k / P) % 2 ? 60 : 160;   /* air planes, two material modes by row */
    { const float o = papyrus_otsu(ct, p3, 1); CHECK(o >= 60 && o < 160, "Otsu %g", o);
      papyrus_cfg r = {.t_lo = 50, .t_hi = 128, .rel_lo = 0.5f, .rel_hi = 1.5f, .rel_p = 0.5f};
      CHECK(fabsf(papyrus_cutoff(&r, ct, p3, 0.9, 0.25) - 69.5f) < 1e-4, "absolute cutoff %g", papyrus_cutoff(&r, ct, p3, 0.9, 0.25));
      const float o3 = papyrus_otsu(ct, p3, 3), rel = papyrus_cutoff(&r, ct, p3, 0.1, 0.5);
      CHECK(fabsf(rel - o3) < 1e-4, "relative cutoff %g vs Otsu %g", rel, o3);
      CHECK(papyrus_cutoff(&r, ct, p3, 0.9, 0.0) == 50.f && papyrus_cutoff(&r, ct, p3, 0.9, 1.0) == 128.f, "cutoff range ends"); }

    /* checkpoint tag */
    { char fn[] = "/tmp/test_papyrus_XXXXXX"; int fd = mkstemp(fn); FILE *f = fd >= 0 ? fdopen(fd, "w") : nullptr; int cond = -1;
      if (f) { fputs("UFSM{\"nlev\":4,\"extra\":{\"task\":\"papyrus\",\"papyrus\":{\"version\":1,\"cond\":1}}}\n\x01\x02", f); fclose(f); }
      CHECK(papyrus_checkpoint(fn, &cond) == 1 && cond == 1, "papyrus checkpoint not recognised (cond %d)", cond);
      f = fopen(fn, "w"); if (f) { fputs("UFSM{\"nlev\":4,\"extra\":{\"task\":\"band_affinity\"}}\n", f); fclose(f); }
      CHECK(papyrus_checkpoint(fn, &cond) == 0, "band_affinity checkpoint taken for papyrus");
      remove(fn); CHECK(papyrus_checkpoint(fn, &cond) == -1, "missing checkpoint"); }

    /* running max / min and the closing vs brute force, windows cut at the faces, radii from 0 to past the cube */
    { enum { Q = 17 }; const size_t q3 = (size_t)Q * Q * Q;
      uint8_t *in = malloc(q3), *o = malloc(q3), *T = malloc(q3), *Cl = malloc(q3), *F = malloc(q3), *dil = malloc(q3);
      uint32_t st = 12345; for (size_t k = 0; k < q3; k++) { st = st * 1664525u + 1013904223u; in[k] = (uint8_t)(st >> 24); }
#define QI(z, y, x) (((size_t)(z) * Q + (y)) * Q + (x))
      const int radii[] = {0, 1, 2, 5, 9, 20};
      for (int ri = 0; ri < 6; ri++) for (int mx = 0; mx <= 1; mx++) {
          const int r = radii[ri]; int bad = 0;
          papyrus_filter(in, o, Q, r, mx, T);
          for (int z = 0; z < Q; z++) for (int y = 0; y < Q; y++) for (int x = 0; x < Q; x++) {
              int e = mx ? 0 : 255;
              for (int c = z - r; c <= z + r; c++) for (int b = y - r; b <= y + r; b++) for (int a = x - r; a <= x + r; a++) {
                  if (c < 0 || b < 0 || a < 0 || c >= Q || b >= Q || a >= Q) continue;
                  const int v = in[QI(c, b, a)]; e = mx ? (v > e ? v : e) : (v < e ? v : e);
              }
              bad += o[QI(z, y, x)] != e;
          }
          CHECK(!bad, "%s filter r %d: %d voxels differ from brute force", mx ? "max" : "min", r, bad);
      }
      { const int r = 2; int bad = 0;   /* closing = min of the max; floor = min */
        papyrus_levels(in, Q, r, Cl, F, T);
        papyrus_filter(in, dil, Q, r, 1, T); papyrus_filter(dil, o, Q, r, 0, T);
        for (size_t k = 0; k < q3; k++) bad += Cl[k] != o[k] || Cl[k] < in[k];
        papyrus_filter(in, o, Q, r, 0, T);
        for (size_t k = 0; k < q3; k++) bad += F[k] != o[k];
        CHECK(!bad, "closing / floor: %d voxels wrong", bad); }
#undef QI
      free(in); free(o); free(T); free(Cl); free(F); free(dil); }

    /* gap labels (no cutoff) on a synthetic scroll varying along x: open air 20 | faint papyrus 70 | sheet 160 | packed gap 80 |
       sheet 160 | sheet 200 with a shallow dip to 150 (above t_hi: texture, not void); then noisy uniform papyrus */
    { enum { Q = 64 }; const size_t q3 = (size_t)Q * Q * Q;
      uint8_t *c = malloc(q3), *tg = malloc(q3), *ig = calloc(q3, 1), *f = malloc(q3), *sc = malloc(8 * q3);
      papyrus_cfg L = {.t_lo = 50, .t_hi = 128, .sigma = 0, .local = 6, .local_sigma = 0, .local_depth = 30, .local_margin = 0};
      int prof[Q], want[Q];
      for (int x = 0; x < Q; x++) {
          prof[x] = x < 10 ? 20 : x < 18 ? 70 : x < 30 ? 160 : x < 33 ? 80 : x < 45 ? 160 : x >= 52 && x < 55 ? 150 : 200;
          want[x] = !(x < 10 || (x >= 30 && x < 33));
      }
      for (size_t k = 0; k < q3; k++) c[k] = (uint8_t)prof[k % Q];
      papyrus_labels(c, Q, &L, 0.f, 0.0, tg, ig, f, sc);
      int bad = 0, first = -1;
      for (size_t k = 0; k < q3; k++) if ((tg[k] == 255) != want[k % Q] || ig[k]) { bad++; if (first < 0) first = (int)(k % Q); }
      CHECK(!bad, "gap labels: %d voxels wrong (first at x %d: CT %d label %d)", bad, first, first >= 0 ? prof[first] : 0, first >= 0 ? tg[first] : 0);
      { papyrus_cfg G = L; G.local = 0; papyrus_labels(c, Q, &G, 55.f, 0.0, tg, ig, f, sc);   /* the old way at cutoff 55 */
        CHECK(tg[31] == 255, "a global cutoff of 55 should have called the packed gap papyrus"); }
      L.local_margin = 0.2f; memset(ig, 0, q3);   /* the margin ignores the half-depth band of a ramp, not the plateaus */
      for (size_t k = 0; k < q3; k++) { const int x = (int)(k % Q); c[k] = (uint8_t)(x == 29 || x == 33 ? 120 : prof[x]); }
      papyrus_labels(c, Q, &L, 0.f, 0.0, tg, ig, f, sc);
      CHECK(ig[29] && ig[33] && !ig[25] && !ig[31] && !ig[40], "margin: ramp voxels %d %d ignored, plateaus %d %d %d not", ig[29], ig[33], ig[25], ig[31], ig[40]);
      uint32_t st = 777; L.local_margin = 0; memset(ig, 0, q3);   /* noise +-12 on uniform papyrus 100 (inside [t_lo, t_hi], so only the
                                                                     depth gate keeps it papyrus) is texture: no void */
      for (size_t k = 0; k < q3; k++) { st = st * 1664525u + 1013904223u; c[k] = (uint8_t)(100 + (int)((st >> 24) % 25) - 12); }
      papyrus_labels(c, Q, &L, 0.f, 0.0, tg, ig, f, sc);
      int nv = 0; for (size_t k = 0; k < q3; k++) nv += tg[k] != 255;
      CHECK(!nv, "noisy uniform papyrus: %d voxels called void", nv);
      free(c); free(tg); free(ig); free(f); free(sc); }

    free(ct); free(t); free(ign); free(fg); free(scratch); free(A); free(B);
    if (fails) { fprintf(stderr, "test_papyrus: %d failures\n", fails); return 1; }
    puts("papyrus labels: smoothing, hard / soft cutoffs, ignore band, cleanup, Otsu, gap labels, checkpoint tag ok");
    return 0;
}
