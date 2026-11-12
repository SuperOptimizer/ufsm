#pragma once
/* The 2x pooling of winding_mod14 codes (codes 1..252 = phase step + 1, 0 none, 255 conflict), one 2^3 block -> one coarse code; shared by
   the host (band_onfly.c codes_pool2_box, phase-raster --gs 4) and the device (nn_band.cu k_codes_pool2), byte-identical.
   Empty when all eight are; 255 when any is 255 (or not a code: 253, 254) or two lie more than a quarter turn apart; else the circular mean of the codes' unit
   vectors (float sums in child order), rounded to the nearest step. The rounding is a bin search on the sign of cross(edge_m, v), edge_m
   the unit vector (as floats) at (m - 0.5) steps: each cross is two float x float products (exact in double) and one subtraction, so the
   sign is exact and the same with or without FMA contraction. (Until 2026-11-12 it was lround(atan2(y, x) / 2 pi x 252): the same bin
   except within float rounding of a bin edge, e.g. two children on code k and two on k + 1; glibc's atan2 has no device twin.) */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846   /* (glibc's value; strict C23 without _GNU_SOURCE has none) */
#endif

#ifdef __CUDACC__
#define CP_HD __host__ __device__
#else
#define CP_HD
#endif
enum { CP_P = 252, CP_QUARTER = 9 };   /* BAND_PERIOD; BAND_STEPS / 2: the conflict threshold 2 |diff| > BAND_STEPS / 2 in band steps */

/* T[0 .. 4 CP_P): cos, sin of code k's angle (2 pi k / P), then of the bin edge below code k (2 pi (k - 0.5) / P), as floats */
static inline void cp_tables_init(float *T) {
    for (int k = 0; k < CP_P; k++) {
        T[k] = (float)cos(2 * M_PI * k / CP_P); T[CP_P + k] = (float)sin(2 * M_PI * k / CP_P);
        T[2 * CP_P + k] = (float)cos(2 * M_PI * (k - 0.5) / CP_P); T[3 * CP_P + k] = (float)sin(2 * M_PI * (k - 0.5) / CP_P);
    }
}
CP_HD static inline float cp_fadd(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}
CP_HD static inline double cp_cross(const float *T, int m, float x, float y) {   /* > 0: v counter-clockwise of edge m (within a half turn) */
    const int j = ((m % CP_P) + CP_P) % CP_P;
    return (double)T[2 * CP_P + j] * (double)y - (double)T[3 * CP_P + j] * (double)x;
}
CP_HD static inline uint8_t cp_pool8(const uint8_t v[8], const float *T) {
    int first = -1, k = 0; float x = 0, y = 0;
    for (int q = 0; q < 8; q++) {
        const int c = v[q];
        if (!c) continue;
        if (c > CP_P) return 255;   /* (255 conflict; 253, 254 are no code: conflict too, the tables end at 252) */
        if (first >= 0) { int d = ((c - 1 - first + 126) % 252 + 252) % 252 - 126; if (d < 0) d = -d; if (2 * d > CP_QUARTER) return 255; }   /* (band_diff) */
        if (first < 0) first = c - 1;
        x = cp_fadd(x, T[c - 1]); y = cp_fadd(y, T[CP_P + c - 1]); k++;
    }
    if (!k) return 0;
    int m = first;   /* the mean lies within a quarter turn of the first code: a few steps of walk */
    for (int it = 0; it < CP_P / 2 && cp_cross(T, m, x, y) < 0; it++) m--;
    for (int it = 0; it < CP_P / 2 && cp_cross(T, m + 1, x, y) >= 0; it++) m++;
    return (uint8_t)(1 + ((m % CP_P) + CP_P) % CP_P);
}
