/* sample_phase_outward (the sampler's per-source outward share, train's "phase outward by source" line): a synthetic Archimedean spiral's
   phase around an axis, rising outward, must count (almost) all aligned samples outward; the same phase negated (a label set numbered
   inward) almost all inward; a window off the axis centre too, and a planar phase whose gradient is tangential to the axis counts nothing. */
#include "sample.h"
#include "band.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static void spiral(uint8_t *ph, int D, int P, const int64_t o[3], double cy, double cx, double spacing, int hole) {
    for (int z = 0; z < D; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
        const double dy = o[1] + y - cy, dx = o[2] + x - cx, r = sqrt(dy * dy + dx * dx), th = atan2(dy, dx) / (2 * M_PI);
        double t = r / spacing - th; t -= floor(t);   /* windings rising outward: r = spacing (k + theta) */
        uint8_t *v = &ph[((size_t)z * P + y) * P + x];
        *v = (uint8_t)((int)lrint(t * PHASE_PERIOD) % PHASE_PERIOD);
        if (hole && (x / 7 + y / 5 + z / 3) % 6 == 0) *v = PHASE_UNKNOWN;
    }
}
int main(void) {
    const int D = 48, P = 96; uint8_t *ph = malloc((size_t)D * P * P); float cy[48], cx[48]; int fail = 0;
    const int64_t os[2][3] = {{0, 0, 0}, {10, 150, -40}};   /* centred on the axis; well off it (the axis outside the window) */
    for (int k = 0; k < 2; k++) {
        for (int z = 0; z < D; z++) { cy[z] = 48.f + 0.1f * z; cx[z] = 47.f - 0.05f * z; }
        for (int neg = 0; neg < 2; neg++) {
            spiral(ph, D, P, os[k], 48, 47, 9.0, 1);
            if (neg) for (size_t q = 0; q < (size_t)D * P * P; q++) if (ph[q] < PHASE_PERIOD && ph[q]) ph[q] = (uint8_t)(PHASE_PERIOD - ph[q]);   /* (sample.c phase_negate) */
            uint64_t c[2] = {0, 0}; sample_phase_outward(ph, D, P, os[k], cy, cx, c);
            const double sh = c[0] + c[1] ? (double)c[0] / (double)(c[0] + c[1]) : -1;
            const int ok = c[0] + c[1] > 200 && (neg ? sh < 0.03 : sh > 0.97);
            fprintf(stderr, "window %d %s: %llu out, %llu in, outward share %.4f%s\n", k, neg ? "negated" : "outward", (unsigned long long)c[0], (unsigned long long)c[1], sh, ok ? "" : "  FAIL");
            fail += !ok;
        }
    }
    for (int z = 0; z < D; z++) { cy[z] = 48.f; cx[z] = 48.f; }   /* phase planes z = const: the gradient along z, never radial */
    for (int z = 0; z < D; z++) for (int i = 0; i < P * P; i++) ph[(size_t)z * P * P + i] = (uint8_t)(z * 9 % PHASE_PERIOD);
    { uint64_t c[2] = {0, 0}; const int64_t o0[3] = {0, 0, 0}; sample_phase_outward(ph, D, P, o0, cy, cx, c);
      const int ok = c[0] + c[1] == 0; fprintf(stderr, "axial gradient: %llu + %llu samples (want 0)%s\n", (unsigned long long)c[0], (unsigned long long)c[1], ok ? "" : "  FAIL"); fail += !ok; }
    free(ph);
    fprintf(stderr, "test_phase_outward: %s\n", fail ? "FAIL" : "ok");
    return fail ? 1 : 0;
}
