/* src/phase_check.h: the outward share of a synthetic stored phase (an Archimedean spiral's band phase frac((r - R0) / P) around an
   axis at (cy, cx) per z) is ~1, its negation ~0, a phase rising with azimuth only (no radial component) ~0.5 (discretisation only); phase_mask_ref
   masks exactly the voxels differing by more than tol and bins the differences. */
#include "phase_check.h"
#include "band.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

int main(void) {
    enum { D = 24, H = 96, W = 96 };
    const int n[3] = {D, H, W}; const int64_t y0 = 300, x0 = 200; const double R0 = 20, P = 9;
    uint8_t *p = malloc((size_t)D * H * W), *q = malloc((size_t)D * H * W), *t = malloc((size_t)D * H * W);
    double cy[D], cx[D];
    for (int z = 0; z < D; z++) { cy[z] = 330 + 0.5 * z; cx[z] = 180 - 0.3 * z; }   /* the axis drifts with z; the box straddles it in y */
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x;
        const double Y = y0 + y - cy[z], X = x0 + x - cx[z], r = hypot(Y, X), u = (r - R0) / P;
        const int code = (int)lround((u - floor(u)) * PHASE_PERIOD) % PHASE_PERIOD;
        p[i] = (x % 17 == 0) ? PHASE_UNKNOWN : (uint8_t)code;   /* some unknown columns */
        q[i] = p[i] < PHASE_PERIOD && p[i] ? (uint8_t)(PHASE_PERIOD - p[i]) : p[i];
        const double a = atan2(Y, X) / (2 * M_PI);
        t[i] = (uint8_t)((int)lround((a - floor(a)) * PHASE_PERIOD) % PHASE_PERIOD);
    }
    size_t up = 0, tot = 0; phase_outward_count(p, n, y0, x0, cy, cx, &up, &tot);
    const double f = tot ? (double)up / tot : NAN;
    size_t up2 = 0, tot2 = 0; phase_outward_count(q, n, y0, x0, cy, cx, &up2, &tot2);
    const double f2 = tot2 ? (double)up2 / tot2 : NAN;
    size_t up3 = 0, tot3 = 0; phase_outward_count(t, n, y0, x0, cy, cx, &up3, &tot3);
    const double f3 = tot3 ? (double)up3 / tot3 : NAN;
    printf("phase-check: outward %.4f of %zu (rising outward), %.4f of %zu (negated), %.4f of %zu for an azimuth-only phase\n", f, tot, f2, tot2, f3, tot3);
    CHECK(tot > 10000 && f > 0.99, "rising outward: %.4f of %zu", f, tot);
    CHECK(tot2 == tot && f2 < 0.01, "negated: %.4f of %zu", f2, tot2);
    CHECK(!(fabs(f3 - 0.5) > 0.1), "azimuth-only phase: outward %.4f of %zu (no radial direction expected)", f3, tot3);
    /* phase_mask_ref */
    const size_t N = (size_t)D * H * W; uint8_t *o = malloc(N), *r = malloc(N);
    size_t exp_mask = 0, exp_both = 0;
    for (size_t i = 0; i < N; i++) {
        o[i] = (uint8_t)(i % 252); const int d = (int)(i % 101) - 50;   /* -50 .. 50 codes: |d| up to 0.198 turn */
        r[i] = (i % 13 == 0) ? PHASE_UNKNOWN : (uint8_t)(((int)o[i] + d + 252) % 252);
        if (i % 29 == 0) o[i] = PHASE_UNKNOWN;
        if (o[i] < 252 && r[i] < 252) { exp_both++; exp_mask += abs(d) / 252.0 > 0.1; }
    }
    size_t hist[PHASE_AGREE_BINS] = {0}, masked = 0; const size_t both = phase_mask_ref(o, r, N, 0.1, hist, &masked);
    size_t left = 0; for (size_t i = 0; i < N; i++) if (o[i] < 252 && r[i] < 252) left++;
    printf("phase_mask_ref: %zu known in both, %zu masked (expected %zu), bins %zu %zu %zu %zu %zu\n", both, masked, exp_mask, hist[0], hist[1], hist[2], hist[3], hist[4]);
    CHECK(both == exp_both && masked == exp_mask && left == both - masked && hist[3] == 0 && hist[4] == 0 && hist[0] + hist[1] + hist[2] == both, "phase_mask_ref counts");
    free(p); free(q); free(t); free(o); free(r);
    printf("test_phase_check: %s\n", fails ? "FAIL" : "OK");
    return fails != 0;
}
