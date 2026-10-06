/* Band field from a winding_mod14 raster (see band.h). Exact Euclidean feature transform (Felzenszwalb-Huttenlocher lower
   envelopes, separable over x, y, z) carrying the index of the nearest surface voxel. */
#include "band.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define BINF 1e20f

/* 1-D squared-distance lower envelope along one line: f[i] (cost), src[i] (index) -> d[i], s[i] */
static void envelope(const float *f, const int32_t *src, int n, float *d, int32_t *s, int *v, float *z) {
    int k = 0; v[0] = -1;
    for (int q = 0; q < n; q++) {
        if (f[q] >= BINF) continue;
        if (v[0] < 0) { v[0] = q; z[0] = -BINF; z[1] = BINF; k = 0; continue; }
        float sct;
        for (;;) {
            const int r = v[k];
            sct = ((f[q] + (float)q * q) - (f[r] + (float)r * r)) / (2.f * (q - r));
            if (sct <= z[k] && k > 0) k--; else break;
        }
        if (sct <= z[k]) { v[k] = q; z[k + 1] = BINF; }   /* k == 0 and the new parabola dominates */
        else { k++; v[k] = q; z[k] = sct; z[k + 1] = BINF; }
    }
    if (v[0] < 0) { for (int q = 0; q < n; q++) { d[q] = BINF; s[q] = -1; } return; }
    k = 0;
    for (int q = 0; q < n; q++) {
        while (z[k + 1] < q) k++;
        const int r = v[k]; d[q] = (float)(q - r) * (q - r) + f[r]; s[q] = src[r];
    }
}

/* squared distance (label voxels) and flat index of the nearest seed voxel; idx -1 / d BINF where there is none */
static int edt_feature(const uint8_t *seed, const int n[3], float *d2, int32_t *idx) {
    const int nz = n[0], ny = n[1], nx = n[2], m = nz > ny ? (nz > nx ? nz : nx) : (ny > nx ? ny : nx);
    float *f = malloc((size_t)m * 4), *dl = malloc((size_t)m * 4), *z = malloc((size_t)(m + 1) * 4);
    int32_t *sr = malloc((size_t)m * 4), *sl = malloc((size_t)m * 4); int *v = malloc((size_t)m * sizeof(int));
    if (!f || !dl || !z || !sr || !sl || !v) { free(f); free(dl); free(z); free(sr); free(sl); free(v); return -1; }
    const size_t sy = (size_t)nx, sz = (size_t)ny * nx;
    for (size_t i = 0; i < (size_t)nz * sz; i++) { d2[i] = seed[i] ? 0.f : BINF; idx[i] = seed[i] ? (int32_t)i : -1; }
    const size_t str[3] = {1, sy, sz}; const int len[3] = {nx, ny, nz};
    for (int ax = 0; ax < 3; ax++) {   /* x, then y, then z */
        const int L = len[ax];
        const size_t st = str[ax];
        const size_t nlines = (size_t)nz * ny * nx / L;
        for (size_t li = 0; li < nlines; li++) {
            size_t base;
            if (ax == 0) base = li * sy;
            else if (ax == 1) base = (li / nx) * sz + (li % nx);
            else base = li;
            for (int q = 0; q < L; q++) { f[q] = d2[base + q * st]; sr[q] = idx[base + q * st]; }
            envelope(f, sr, L, dl, sl, v, z);
            for (int q = 0; q < L; q++) { d2[base + q * st] = dl[q]; idx[base + q * st] = sl[q]; }
        }
    }
    free(f); free(dl); free(z); free(sr); free(sl); free(v);
    return 0;
}

int band_field(const uint8_t *codes, const int n[3], const int64_t o[3], const double *cy, const double *cx, band_params p, uint8_t *out) {
    const size_t N = (size_t)n[0] * n[1] * n[2], sy = (size_t)n[2], sz = (size_t)n[1] * n[2];
    uint8_t *seed = malloc(N), *flag = malloc(N);
    float *d2 = malloc(N * 4); int32_t *idx = malloc(N * 4);
    if (!seed || !flag || !d2 || !idx) { free(seed); free(flag); free(d2); free(idx); return -1; }
    for (size_t i = 0; i < N; i++) seed[i] = codes[i] > 0 && codes[i] < 255;
    if (edt_feature(seed, n, d2, idx)) { free(seed); free(flag); free(d2); free(idx); return -1; }
    /* band = nearest surface winding +- half a turn; unknown beyond the flood radius */
    const float r2 = (p.radius / 2.f) * (p.radius / 2.f);
    for (size_t i = 0; i < N; i++) {
        if (idx[i] < 0 || d2[i] > r2) { out[i] = BAND_UNKNOWN; continue; }
        const size_t s = (size_t)idx[i];
        const int z = (int)(i / sz), y = (int)((i / sy) % n[1]), x = (int)(i % sy);
        const int zs = (int)(s / sz), ys = (int)((s / sy) % n[1]), xs = (int)(s % sy);
        const double rv = hypot(2.0 * (y + o[1]) + 0.5 - cy[z], 2.0 * (x + o[2]) + 0.5 - cx[z]);
        const double rs = hypot(2.0 * (ys + o[1]) + 0.5 - cy[zs], 2.0 * (xs + o[2]) + 0.5 - cx[zs]);
        const int k = (int)codes[s] - 1 + (rv >= rs ? BAND_HALF : -BAND_HALF);
        out[i] = (uint8_t)((k % BAND_PERIOD + BAND_PERIOD) % BAND_PERIOD);
    }
    /* conflicts (two surfaces > 1/4 turn apart in one voxel): unknown within one voxel */
    memset(flag, 0, N);
    for (int z = 0; z < n[0]; z++) for (int y = 0; y < n[1]; y++) for (int x = 0; x < n[2]; x++) {
        if (codes[z * sz + y * sy + x] != 255) continue;
        for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
            const int a = z + dz, b = y + dy, c = x + dx;
            if (a >= 0 && a < n[0] && b >= 0 && b < n[1] && c >= 0 && c < n[2]) flag[a * sz + b * sy + c] = 1;
        }
    }
    for (size_t i = 0; i < N; i++) if (flag[i]) out[i] = BAND_UNKNOWN;
    /* missing wraps: the band jumps by >= half a turn between neighbours both more than 2 native voxels off the surfaces */
    memset(seed, 0, N);   /* reused: jump voxels */
    int any = 0;
    for (int z = 0; z < n[0]; z++) for (int y = 0; y < n[1]; y++) for (int x = 0; x < n[2]; x++) {
        const size_t i = z * sz + y * sy + x;
        if (out[i] == BAND_UNKNOWN || d2[i] <= 1.f) continue;
        const size_t nb[3] = {z + 1 < n[0] ? i + sz : i, y + 1 < n[1] ? i + sy : i, x + 1 < n[2] ? i + 1 : i};
        for (int a = 0; a < 3; a++) {
            const size_t j = nb[a];
            if (j == i || out[j] == BAND_UNKNOWN || d2[j] <= 1.f) continue;
            if (abs(band_diff(out[i], out[j])) >= BAND_HALF) { seed[i] = seed[j] = 1; any = 1; }
        }
    }
    if (any) {
        if (edt_feature(seed, n, d2, idx)) { free(seed); free(flag); free(d2); free(idx); return -1; }
        const float s2 = (p.span / 2.f) * (p.span / 2.f);
        for (size_t i = 0; i < N; i++) if (d2[i] <= s2) out[i] = BAND_UNKNOWN;
    }
    free(seed); free(flag); free(d2); free(idx);
    return 0;
}
