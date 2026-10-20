/* sample_phase_interp (separable, polynomial atan2: the sampler's 9.6 -> 2.4 um stored-phase path) against the 8-neighbour atan2f
   reference: random phase ramps with unknown holes, windows of several shapes / offsets, shift 2 and 3. Codes must agree exactly in
   >= 99.9% of the known voxels and within 1 code (circular) everywhere; the known / unknown split may differ only at the 0.5-weight cut.
   Prints both paths' times. */
#include "sample.h"
#include "band.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rs = 88172645463325252ull;
static uint64_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }
int main(void) {
    const int cases[][3] = {{64, 64, 2}, {32, 96, 2}, {48, 80, 3}, {192, 256, 2}, {192, 768, 2}};
    double tn = 0, tr = 0; int fail = 0;
    for (int ci = 0; ci < 5; ci++) {
        const int D = cases[ci][0], P = cases[ci][1], shift = cases[ci][2], g = 1 << shift;
        for (int rep = 0; rep < 3; rep++) {
            const int64_t o[3] = {(int64_t)(rnd() % 50), (int64_t)(rnd() % 50), (int64_t)(rnd() % 50)};
            const double c0 = 0.5 * (g - 1); const int wn[3] = {D, P, P}; int64_t lo[3], ln[3];
            for (int d = 0; d < 3; d++) { lo[d] = (int64_t)floor((o[d] - c0) / g); ln[d] = (int64_t)floor((o[d] + wn[d] - 1 - c0) / g) + 2 - lo[d]; }
            const size_t N = (size_t)ln[0] * ln[1] * ln[2];
            uint8_t *ph = malloc(N);
            const double kz = 0.01 * (double)(rnd() % 100) / 100, ky = 0.08 + 0.04 * (double)(rnd() % 100) / 100, kx = 0.03;   /* turns per grid voxel */
            for (int64_t z = 0; z < ln[0]; z++) for (int64_t y = 0; y < ln[1]; y++) for (int64_t x = 0; x < ln[2]; x++) {
                double t = kz * z + ky * y + kx * x + 0.02 * sin(0.3 * x); t -= floor(t);
                int q = (int)lrint(t * PHASE_PERIOD) % PHASE_PERIOD;
                const int hole = (x / 5 + y / 7 + z / 3) % 9 == 0 || rnd() % 40 == 0;
                ph[((size_t)z * ln[1] + y) * ln[2] + x] = hole ? PHASE_UNKNOWN : (uint8_t)q;
            }
            const size_t p3 = (size_t)D * P * P;
            uint8_t *a = malloc(p3), *b = malloc(p3);
            double t0 = now(); sample_phase_interp(ph, ln, lo, o, D, P, shift, a); tn += now() - t0;
            t0 = now(); sample_phase_interp_ref(ph, ln, lo, o, D, P, shift, b); tr += now() - t0;
            size_t known = 0, same = 0, off1 = 0, worse = 0, split = 0;
            for (size_t k = 0; k < p3; k++) {
                if ((a[k] == PHASE_UNKNOWN) != (b[k] == PHASE_UNKNOWN)) { split++; continue; }
                if (a[k] == PHASE_UNKNOWN) continue;
                known++; int d = abs((int)a[k] - (int)b[k]); if (d > PHASE_PERIOD / 2) d = PHASE_PERIOD - d;
                if (!d) same++; else if (d == 1) off1++; else worse++;
            }
            const int bad = worse || same < known * 0.999 || split > p3 / 10000 + 2 || known < p3 / 2;
            if (bad || rep == 0) printf("D %d P %d shift %d: %zu known, %zu same, %zu off by 1, %zu worse, %zu known/unknown splits%s\n", D, P, shift, known, same, off1, worse, split, bad ? "  FAIL" : "");
            fail |= bad;
            free(ph); free(a); free(b);
        }
    }
    printf("separable %.3f s, reference %.3f s (%.1fx)\n", tn, tr, tr / tn);
    printf(fail ? "test_phase_interp: FAIL\n" : "test_phase_interp: ok\n");
    return fail;
}
