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

    free(ct); free(t); free(ign); free(fg); free(scratch); free(A); free(B);
    if (fails) { fprintf(stderr, "test_papyrus: %d failures\n", fails); return 1; }
    puts("papyrus labels: smoothing, hard / soft cutoffs, ignore band, cleanup, Otsu, checkpoint tag ok");
    return 0;
}
