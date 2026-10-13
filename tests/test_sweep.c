/* predict --cond-sweep's host half (src/sweep.c sweep_unwrap): on an Archimedean spiral around an axis inside the box, U = r / pitch
   spans more than 14 turns; canvases hold the 14-turn phasor of U (with up to +-0.35 turn of noise) and the 1-turn phase phi = frac(U -
   theta / 2 pi), quantised to bytes as the place kernel writes them. The unwrap must give back U up to one constant everywhere the
   flood reaches (the snapped fraction removes the 14-turn noise), reach every papyrus voxel, and leave air NaN. */
#include "sweep.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)
static uint8_t q8(double v) { v = 127.5 + 127.5 * v + 0.5; return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); }

int main(void) {
    const int64_t bn[3] = {5, 96, 110}; const size_t N = (size_t)bn[0] * bn[1] * bn[2], P2 = (size_t)bn[1] * bn[2];
    const double pitch = 2.7;   /* r up to ~70 voxels: ~26 turns */
    float cy[5], cx[5]; for (int z = 0; z < 5; z++) { cy[z] = 47.3f + 0.4f * z; cx[z] = 51.8f - 0.3f * z; }
    uint8_t *c14 = malloc(2 * N), *p1 = malloc(2 * N), *done = malloc(N); float *U = malloc(N * 4); double *Ut = malloc(N * 8);
    unsigned s = 99; size_t npap = 0;
    for (int dir = -1; dir <= 1; dir += 2) {
        npap = 0;
        for (size_t g = 0; g < N; g++) {
            const int64_t z = (int64_t)(g / P2), y = (int64_t)((g / bn[2]) % bn[1]), x = (int64_t)(g % bn[2]);
            const double r = hypot(y - cy[z], x - cx[z]), th = atan2(y - cy[z], x - cx[z]) / (2 * M_PI), u = r / pitch;
            Ut[g] = u;
            done[g] = r > 4 ? 1 : 2; npap += done[g] == 1;
            s = s * 1664525u + 1013904223u; const double noise = ((s >> 8) / 16777216.0 - 0.5) * 0.7;   /* +-0.35 turn on the 14-turn phasor */
            c14[g] = q8(cos(2 * M_PI * (u + noise) / 14)); c14[N + g] = q8(sin(2 * M_PI * (u + noise) / 14));
            double phi = u - dir * th; phi -= floor(phi);
            p1[g] = q8(cos(2 * M_PI * phi)); p1[N + g] = q8(sin(2 * M_PI * phi));
        }
        const size_t seed = (size_t)2 * P2 + 10 * bn[2] + 12;   /* a papyrus voxel near a corner (far from the axis) */
        const size_t reached = sweep_unwrap(c14, p1, done, bn, cy, cx, dir, seed, U);
        CHECK(reached == npap, "dir %d: the flood reached %zu of %zu papyrus voxels", dir, reached, npap);
        const double off = U[seed] - Ut[seed]; double emax = 0; size_t nan_air = 0, air = 0;
        for (size_t g = 0; g < N; g++) {
            if (done[g] != 1) { air++; nan_air += U[g] != U[g]; continue; }
            const double e = fabs(U[g] - Ut[g] - off); if (e > emax) emax = e;
        }
        CHECK(fabs(off - 14 * round(off / 14)) < 0.02, "dir %d: offset %.3f is not a multiple of 14 turns", dir, off);
        CHECK(emax < 0.02, "dir %d: U off the true spiral coordinate by up to %.3f turn", dir, emax);
        CHECK(nan_air == air, "dir %d: %zu of %zu air voxels got a U", dir, air - nan_air, air);
        printf("dir %+d: %zu papyrus voxels, U range %.1f turns, max |U - U_true - %.0f| %.4f turn\n", dir, npap, 70 / pitch, off, emax);
    }
    free(c14); free(p1); free(done); free(U); free(Ut);
    if (fails) { fprintf(stderr, "test_sweep: %d failures\n", fails); return 1; }
    puts("test_sweep: the conditioned sweep's unwrap recovers U beyond 14 turns ok");
    return 0;
}
