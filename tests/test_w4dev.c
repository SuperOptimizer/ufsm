/* band_wind4_dev (src/nn_w4.cu: the per-voxel work of the winding mod 4 on the GPU, the union-find over run events
   on the host) against band_wind4_tiled, byte for byte: w4, the component plane, the component count and the statistics.
   Fields: a spiral with noise and unknown holes (long runs, several components), vortices (loops that gain a turn: conflicts, dropped
   components), random phases (runs of one voxel), all unknown; windows that are not multiples of the tile, tile 0 (one tile, 15
   components), tiles of one voxel row. */
#include "band_w4dev.h"
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static double ru(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) / 9007199254740992.0; }

static void field(int kind, int D, int H, int W, uint8_t *ph) {
    const double cy = H * (0.3 + 0.4 * ru()), cx = W * (0.3 + 0.4 * ru()), pitch = 9 + 30 * ru(), noise = kind == 1 ? 0.02 : 0.08 * ru(), hole = 0.02 + 0.1 * ru();
    double vz[6], vy[6], vx[6]; for (int k = 0; k < 6; k++) { vz[k] = D * ru(); vy[k] = H * ru(); vx[k] = W * ru(); }
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x;
        double w;
        if (kind == 2) w = ru();   /* random phases */
        else if (kind == 3) { ph[i] = 252 + (int)(4 * ru()) % 4; continue; }
        else {
            const double dy = y - cy, dx = x - cx, r = sqrt(dy * dy + dx * dx);
            w = r / pitch + atan2(dy, dx) / (2 * M_PI) + 0.01 * z;   /* a spiral: one turn per pitch */
            if (kind == 1) for (int k = 0; k < 6; k++) w += atan2(y - vy[k], x - vx[k] + 0.3 * (z - vz[k])) / (2 * M_PI) * (k & 1 ? 1 : -1);   /* vortex lines: inconsistent loops */
            w += noise * (ru() - 0.5);
        }
        w -= floor(w);
        int q = (int)lround(w * 252) % 252;
        if (ru() < hole || (kind == 0 && ((x / 7 + y / 11) % 9 == 0 && z % 5 < 2))) q = 252 + (int)(4 * ru()) % 4;
        ph[i] = (uint8_t)q;
    }
}
#include <time.h>
static double cpu_s(void) { struct timespec t; clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static double wall_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
int main(int argc, char **argv) {
    const int dev = 0; w4dev_init(&dev, 1);
    if (argc >= 4) {   /* test_w4dev PHASE.raw P TILE [REPEATS]: a stored phase window (ufsm read), both paths timed (thread CPU and wall) */
        const int P = atoi(argv[2]), T = atoi(argv[3]), R = argc > 4 ? atoi(argv[4]) : 3; const size_t n = (size_t)P * P * P;
        uint8_t *ph = malloc(n), *w0 = malloc(n), *c0 = malloc(n), *w1 = nn_host_alloc(n), *c1 = nn_host_alloc(n);   /* (the sampler's batch planes are pinned) */
        FILE *f = fopen(argv[1], "rb"); if (!f || fread(ph, 1, n, f) != n) { fprintf(stderr, "test_w4dev: cannot read %s\n", argv[1]); return 2; }
        for (int r = 0; r < R; r++) {
            band_wind4_stats s0 = {0}, s1 = {0};
            double c = cpu_s(), w = wall_s(); const int n0 = band_wind4_tiled(ph, P, P, P, T, 80, 512, T ? 1 : 15, w0, c0, &s0); const double ca = cpu_s() - c, wa = wall_s() - w;
            c = cpu_s(); w = wall_s(); const int n1 = band_wind4_dev(ph, P, P, P, T, 80, 512, T ? 1 : 15, w1, c1, &s1); const double cb = cpu_s() - c, wb = wall_s() - w;
            printf("P %d tile %d: %zu runs, components %d / %d, bytes %s; CPU path %.3f CPU-s (%.3f s wall), device path %.3f CPU-s (%.3f s wall)\n", P, T, s0.runs, n0, n1,
                   memcmp(w0, w1, n) || memcmp(c0, c1, n) ? "DIFFER" : "identical", ca, wa, cb, wb);
        }
        return 0;
    }
    static const int dims[][4] = {{48, 48, 48, 16}, {40, 52, 61, 16}, {33, 47, 70, 24}, {64, 64, 64, 0}, {50, 31, 44, 64}, {20, 20, 300, 7}, {96, 96, 96, 32}, {17, 90, 90, 1}, {130, 140, 150, 48}};
    int fails = 0, cases = 0; size_t runs = 0, dropped = 0;
    for (size_t di = 0; di < sizeof dims / sizeof dims[0]; di++) for (int kind = 0; kind < 4; kind++) for (int v = 0; v < 2; v++) {
        const int D = dims[di][0], H = dims[di][1], W = dims[di][2], T = dims[di][3], ms = v ? 40 : 80; const size_t n = (size_t)D * H * W, minsize = v ? 1 : 64;
        uint8_t *ph = malloc(n), *w0 = malloc(n), *c0 = malloc(n), *w1 = malloc(n), *c1 = malloc(n);
        field(kind, D, H, W, ph); memset(w1, 7, n); memset(c1, 7, n);
        band_wind4_stats s0 = {0}, s1 = {0};
        const int mc = T ? (v ? 3 : 1) : 15;
        const int n0 = band_wind4_tiled(ph, D, H, W, T, ms, minsize, mc, w0, c0, &s0), n1 = band_wind4_dev(ph, D, H, W, T, ms, minsize, mc, w1, c1, &s1);
        size_t dw = 0, dc = 0; for (size_t i = 0; i < n; i++) { dw += w0[i] != w1[i]; dc += c0[i] != c1[i]; }
        /* the runs path (slots without the planes): device and CPU runs, rebuilt on the host and on the device */
        for (int pass = 0; pass < 2; pass++) {
            if (pass == 1) w4dev_init(&dev, 0);   /* (no device: band_wind4_runs takes the CPU loops) */
            w4runs ru = {0}; band_wind4_stats s2 = {0};
            const int n2 = band_wind4_runs(ph, D, H, W, T, ms, minsize, mc, &ru, &s2);
            uint8_t *w2 = malloc(n), *c2 = malloc(n); memset(w2, 7, n); memset(c2, 7, n);
            w4runs_apply_host(ph, D, H, W, T, ms, &ru, w2, c2);
            size_t ew = 0, ec = 0; for (size_t i = 0; i < n; i++) { ew += w0[i] != w2[i]; ec += c0[i] != c2[i]; }
            uint8_t *dp = nn_malloc(n), *dw4 = nn_malloc(n), *dcp = nn_malloc(n); nn_h2d(dp, ph, n);
            const int ra = nn_w4runs_apply(dp, D, H, W, T, ms, &ru, dw4, dcp, nullptr); nn_sync();
            nn_d2h(w2, dw4, n); nn_d2h(c2, dcp, n);
            size_t gw = 0, gc = 0; for (size_t i = 0; i < n; i++) { gw += w0[i] != w2[i]; gc += c0[i] != c2[i]; }
            if (n2 != n0 || ew || ec || ra || gw || gc || s2.runs != s0.runs || s2.kept != s0.kept || ru.ncomp != n0) {
                fprintf(stderr, "FAIL runs (%s): %dx%dx%d tile %d field %d step %d: components %d / %d, host rebuild %zu / %zu bytes differ, device rebuild rc %d %zu / %zu\n",
                        pass ? "cpu" : "device", D, H, W, T, kind, ms, n0, n2, ew, ec, ra, gw, gc);
                fails++;
            }
            nn_free(dp); nn_free(dw4); nn_free(dcp); free(w2); free(c2); free(ru.off); free(ru.ro); free(ru.cb);
            if (pass == 1) w4dev_init(&dev, 1);
        }
        cases++; runs += s0.runs; dropped += s0.dropped_conflict;
        if (n0 != n1 || dw || dc || s0.known != s1.known || s0.kept != s1.kept || s0.dropped_conflict != s1.dropped_conflict || s0.runs != s1.runs || s0.ncomp != s1.ncomp) {
            fprintf(stderr, "FAIL: %dx%dx%d tile %d field %d step %d: components %d / %d, %zu w4 and %zu component bytes differ, known %zu / %zu kept %zu / %zu dropped %zu / %zu runs %zu / %zu\n",
                    D, H, W, T, kind, ms, n0, n1, dw, dc, s0.known, s1.known, s0.kept, s1.kept, s0.dropped_conflict, s1.dropped_conflict, s0.runs, s1.runs);
            fails++;
        }
        free(ph); free(w0); free(c0); free(w1); free(c1);
    }
    if (fails) { printf("test_w4dev: %d of %d cases FAIL\n", fails, cases); return 1; }
    printf("test_w4dev: ok (%d cases byte-identical to band_wind4_tiled: %zu runs, %zu voxels in components dropped for conflicts)\n", cases, runs, dropped);
    return 0;
}
