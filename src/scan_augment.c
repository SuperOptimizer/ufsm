#include "scan_augment.h"
#include "ct_augment.h"
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

const char *const scan_aug_names[SA_N] = {"seam", "air", "lowres", "blur", "sharpen", "bias", "ring", "cupping", "slice",
                                          "noise", "cnoise", "tone", "quant", "dropslice", "cutout", "chunk", "paganin", "gin", "hist", "motion"};
#include <stdio.h>
int scan_aug_load_hist(const char *path, scan_aug_cfg *c) {
    FILE *f = fopen(path, "r"); if (!f) return -1;
    float *h = nullptr; int n = 0; double cnt[256];
    for (;;) {
        int k = 0; for (; k < 256 && fscanf(f, "%lf", &cnt[k]) == 1; k++) {}
        if (k < 256) break;
        float *t = realloc(h, (size_t)(n + 1) * 256 * sizeof *h); if (!t) { free(h); fclose(f); return -1; } h = t;
        double tot = 0; for (int v = 1; v < 256; v++) tot += cnt[v];   /* material only (0 = masked air) */
        if (tot <= 0) continue;
        double acc = 0; float *cdf = h + (size_t)n * 256; cdf[0] = 0;
        for (int v = 1; v < 256; v++) { acc += cnt[v]; cdf[v] = (float)(acc / tot); }
        n++;
    }
    fclose(f);
    if (!n) { free(h); return -1; }
    c->hist = h; c->nhist = n;
    return 0;
}
/* two-sided exponential smoothing along one axis of a P^3 float volume (forward and backward first-order recursions, the kernel
   (1 - a)^2 / (1 - a^2) a^|t| sums to 1), edges replicated; line: P floats */
[[maybe_unused]] static void exp_smooth_ref(float *F, int P, int axis, float a, float *line) {
    const size_t P2 = (size_t)P * P, st = axis == 0 ? P2 : axis == 1 ? (size_t)P : 1;
    for (int u = 0; u < P; u++) for (int v = 0; v < P; v++) {
        const size_t b = axis == 0 ? (size_t)u * P + v : axis == 1 ? (size_t)u * P2 + v : (size_t)u * P2 + (size_t)v * P;
        float y = F[b];
        for (int i = 0; i < P; i++) { y = (1 - a) * F[b + i * st] + a * y; line[i] = y; }
        for (int i = P - 1; i >= 0; i--) { y = i == P - 1 ? line[i] : (1 - a) * line[i] + a * y; F[b + i * st] = y; }
    }
}
static void transpose(const float *a, float *b, int P);
/* exp_smooth's axis-1 sweep over one P x P plane: along the row index, every column side by side */
static void exp_smooth_rows(float *base, int P, float a) {
    const float b = 1 - a;
    for (int v = 0; v < P; v++) base[v] = b * base[v] + a * base[v];   /* (y starts as the first element) */
    for (int i = 1; i < P; i++) { float *ri = base + (size_t)i * P; const float *rp = ri - P; for (int v = 0; v < P; v++) ri[v] = b * ri[v] + a * rp[v]; }
    for (int i = P - 2; i >= 0; i--) { float *ri = base + (size_t)i * P; const float *rn = ri + P; for (int v = 0; v < P; v++) ri[v] = b * ri[v] + a * rn[v]; }
}
/* exp_smooth_ref with the lines of axes 0 and 1 run side by side (rows of the contiguous axis at a time, the forward pass in place):
   every element sees the same float operations in the same order, bit for bit; axis 2 (contiguous lines): with a P^2 plane
   scratch and P % 8 == 0, each plane transposed, swept as axis 1, transposed back (else the per-line loop) */
static void exp_smooth(float *F, int P, int axis, float a, float *line, float *plane) {
    const size_t P2 = (size_t)P * P;
    if (axis == 2) {
        if (!plane || P % 8) { exp_smooth_ref(F, P, axis, a, line); return; }
        for (int u = 0; u < P; u++) { transpose(F + (size_t)u * P2, plane, P); exp_smooth_rows(plane, P, a); transpose(plane, F + (size_t)u * P2, P); }
        return;
    }
    const float b = 1 - a;
    const size_t st = axis == 0 ? P2 : (size_t)P;   /* along the line */
    const int outer = P, inner = axis == 0 ? P * P : P;   /* axis 0: one z sweep over whole planes; axis 1: per plane, rows of x */
    for (int u = 0; u < (axis == 0 ? 1 : outer); u++) {
        float *base = F + (axis == 0 ? 0 : (size_t)u * P2);
        float *r0 = base;
        for (int v = 0; v < inner; v++) r0[v] = b * r0[v] + a * r0[v];   /* (y starts as the first element) */
        for (int i = 1; i < P; i++) { float *ri = base + i * st; const float *rp = ri - st; for (int v = 0; v < inner; v++) ri[v] = b * ri[v] + a * rp[v]; }
        for (int i = P - 2; i >= 0; i--) { float *ri = base + i * st; const float *rn = ri + st; for (int v = 0; v < inner; v++) ri[v] = b * ri[v] + a * rn[v]; }
    }
}
/* exp_smooth axis 1 then axis 2 with each z plane in cache between them (the same operations per plane, bit for bit) */
static void exp_smooth_yx(float *F, int P, float a, float *line, float *plane) {
    if (!plane || P % 8) { exp_smooth(F, P, 1, a, line, plane); exp_smooth(F, P, 2, a, line, plane); return; }
    const size_t P2 = (size_t)P * P;
    for (int u = 0; u < P; u++) {
        float *pl = F + (size_t)u * P2;
        exp_smooth_rows(pl, P, a);   /* (= exp_smooth axis 1 on this plane) */
        transpose(pl, plane, P); exp_smooth_rows(plane, P, a); transpose(plane, pl, P);
    }
}

/* Local SplitMix64, as ct_augment: decisions and per-voxel noise use their own streams. */
static uint64_t next_u64(uint64_t *s) {
    uint64_t v = (*s += UINT64_C(0x9e3779b97f4a7c15));
    v = (v ^ (v >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    v = (v ^ (v >> 27)) * UINT64_C(0x94d049bb133111eb);
    return v ^ (v >> 31);
}
/* fmaxf / fminf against a constant that is not a NaN (glibc's semantics: a NaN x gives the constant), inline: the libm calls
   were ~5% of an --aug all sampler profile and kept the per-voxel loops scalar */
static inline float fmax_c(float x, float c) { return x >= c ? x : c; }
static inline float fmin_c(float x, float c) { return x <= c ? x : c; }
static double uniform(uint64_t *s) { return (double)(next_u64(s) >> 11) * 0x1.0p-53; }
static double sym1(uint64_t *s) { return 2 * uniform(s) - 1; }
/* Irwin-Hall of four bytes: unit variance, |g| < 3.46. Two draws per 64 bits. */
static inline float ih4(uint32_t v) { return (float)((int)(v & 255) + (int)((v >> 8) & 255) + (int)((v >> 16) & 255) + (int)(v >> 24) - 510) * (1.f / 147.80f); }
static void gauss_row(uint64_t *s, float *row, int n) {
    for (int x = 0; x < n; x += 2) { uint64_t v = next_u64(s); row[x] = ih4((uint32_t)v); if (x + 1 < n) row[x + 1] = ih4((uint32_t)(v >> 32)); }
}

/* ---- separable 1D operators: out[i] = sum_t w[i][t] in[s[i] + t], replicate boundary ---- */
#define MAXTAP SA_MAXTAP
typedef scan_bop bop;
static void bop_add(float *dense, int *lo, int *hi, int P, int j, float w) {
    j = j < 0 ? 0 : j >= P ? P - 1 : j;
    dense[j] += w; if (j < *lo) *lo = j; if (j > *hi) *hi = j;
}
static void bop_store(bop *b, int i, float *dense, int lo, int hi) {
    if (hi - lo + 1 > MAXTAP) abort();   /* the magnitude caps keep every composition within MAXTAP */
    b->s[i] = lo; b->n[i] = hi - lo + 1;
    for (int t = lo; t <= hi; t++) { b->w[i][t - lo] = dense[t]; dense[t] = 0; }
}
static void bop_identity(bop *b, int P) { b->P = P; for (int i = 0; i < P; i++) { b->s[i] = i; b->n[i] = 1; b->w[i][0] = 1; } }
/* sharpen a > 0: (1 + a) I - a G (phase-contrast edge overshoot); a = -1 is the plain blur G */
static void bop_gauss(bop *b, int P, double sigma, double a, float *dense) {
    int r = (int)ceil(3 * sigma); if (r > 7) r = 7;
    double g[15], sum = 0;
    for (int t = -r; t <= r; t++) sum += g[t + r] = exp(-.5 * t * t / (sigma * sigma));
    b->P = P;
    for (int i = 0; i < P; i++) {
        int lo = P, hi = -1;
        if (a > -1) bop_add(dense, &lo, &hi, P, i, (float)(1 + a));
        for (int t = -r; t <= r; t++) bop_add(dense, &lo, &hi, P, i + t, (float)((a > -1 ? -a : 1) * g[t + r] / sum));
        bop_store(b, i, dense, lo, hi);
    }
}
/* coarser scan: cell averages over [phase + j f, phase + (j+1) f), linearly interpolated back at the voxel centres */
static void bop_lowres(bop *b, int P, double f, double phase, float *dense) {
    b->P = P;
    for (int i = 0; i < P; i++) {
        double u = (i + .5 - phase) / f - .5; int j0 = (int)floor(u); double t = u - j0;
        int lo = P, hi = -1;
        for (int c = 0; c < 2; c++) {
            double a = phase + (j0 + c) * f, e = a + f, wc = c ? t : 1 - t;
            if (wc <= 0) continue;
            for (int k = (int)floor(a); k < (int)ceil(e); k++) {
                double ov = fmin(e, k + 1) - fmax(a, k);
                if (ov > 0) bop_add(dense, &lo, &hi, P, k, (float)(wc * ov / f));
            }
        }
        bop_store(b, i, dense, lo, hi);
    }
}
/* sample motion along an axis: ghost echoes (mode 0: (1 - sum w) I + sum w_e shift(t_e), fractional shifts linear) or a linear
   motion blur (mode 1: the mean of a box of length L, its end voxels weighted by their fraction) */
static void bop_motion(bop *b, int P, int mode, int ne, const double t[2], const double w[2], double L, float *dense) {
    b->P = P;
    for (int i = 0; i < P; i++) {
        int lo = P, hi = -1;
        if (mode == 0) {
            double w0 = 1; for (int e = 0; e < ne; e++) w0 -= w[e];
            bop_add(dense, &lo, &hi, P, i, (float)w0);
            for (int e = 0; e < ne; e++) {
                const double u = i + t[e]; const int j = (int)floor(u); const double f = u - j;
                if (1 - f > 0) bop_add(dense, &lo, &hi, P, j, (float)(w[e] * (1 - f)));
                if (f > 0) bop_add(dense, &lo, &hi, P, j + 1, (float)(w[e] * f));
            }
        } else {   /* box [i - L / 2, i + L / 2] over unit voxels [k - .5, k + .5) */
            const double a = i - .5 * L, e = i + .5 * L;
            for (int k = (int)floor(a + .5); k <= (int)ceil(e - .5); k++) {
                const double ov = fmin(e, k + .5) - fmax(a, k - .5);
                if (ov > 0) bop_add(dense, &lo, &hi, P, k, (float)(ov / L));
            }
        }
        bop_store(b, i, dense, lo, hi);
    }
}
/* c = a . b (b applied first) */
static void bop_compose(bop *c, const bop *a, const bop *b, float *dense) {
    int P = a->P; c->P = P;
    for (int i = 0; i < P; i++) {
        int lo = P, hi = -1;
        for (int t = 0; t < a->n[i]; t++) {
            int j = a->s[i] + t;
            for (int u = 0; u < b->n[j]; u++) bop_add(dense, &lo, &hi, P, b->s[j] + u, a->w[i][t] * b->w[j][u]);
        }
        bop_store(c, i, dense, lo, hi);
    }
}
#ifdef __AVX2__
/* b (stride sb) = the transpose of the 8 x 8 block at a (stride sa) */
static inline void tr8(const float *a, size_t sa, float *b, size_t sb) {
    __m256 r[8], t[8], u[8];
    for (int k = 0; k < 8; k++) r[k] = _mm256_loadu_ps(a + k * sa);
    for (int k = 0; k < 8; k += 2) { t[k] = _mm256_unpacklo_ps(r[k], r[k + 1]); t[k + 1] = _mm256_unpackhi_ps(r[k], r[k + 1]); }
    for (int k = 0; k < 8; k += 4) {
        u[k] = _mm256_shuffle_ps(t[k], t[k + 2], 0x44); u[k + 1] = _mm256_shuffle_ps(t[k], t[k + 2], 0xEE);
        u[k + 2] = _mm256_shuffle_ps(t[k + 1], t[k + 3], 0x44); u[k + 3] = _mm256_shuffle_ps(t[k + 1], t[k + 3], 0xEE);
    }
    for (int k = 0; k < 4; k++) { _mm256_storeu_ps(b + k * sb, _mm256_permute2f128_ps(u[k], u[k + 4], 0x20)); _mm256_storeu_ps(b + (k + 4) * sb, _mm256_permute2f128_ps(u[k], u[k + 4], 0x31)); }
}
#endif
static void transpose(const float *a, float *b, int P) {
#ifdef __AVX2__
    if (P % 8 == 0) {
        for (int i0 = 0; i0 < P; i0 += 32) for (int j0 = 0; j0 < P; j0 += 32)
            for (int i = i0; i < i0 + 32 && i < P; i += 8) for (int j = j0; j < j0 + 32 && j < P; j += 8) tr8(a + (size_t)i * P + j, P, b + (size_t)j * P + i, P);
        return;
    }
#endif
    for (int i0 = 0; i0 < P; i0 += 32) for (int j0 = 0; j0 < P; j0 += 32)
        for (int i = i0; i < i0 + 32 && i < P; i++) for (int j = j0; j < j0 + 32 && j < P; j++) b[(size_t)j * P + i] = a[(size_t)i * P + j];
}
/* rows i of out = sum_t w[i][t] rows (s[i] + t) of in: vector form along the contiguous axis. Each output element is
   w0 r0, then + w_t r_t for t ascending (no contraction); 8 lanes accumulate all taps in a register before one store (the
   tap-outer loop read and wrote the output row once per tap) */
static void bop_rows(const bop *b, const float *in, float *out, size_t ostride, int P) {
    for (int i = 0; i < P; i++) {
        float *o = out + (size_t)i * ostride;
        const float *r0 = in + (size_t)b->s[i] * P; const float *wi = b->w[i]; const int n = b->n[i];
        int x = 0;
#ifdef __AVX2__
        for (; x + 16 <= P; x += 16) {
            __m256 a0 = _mm256_mul_ps(_mm256_set1_ps(wi[0]), _mm256_loadu_ps(r0 + x)), a1 = _mm256_mul_ps(_mm256_set1_ps(wi[0]), _mm256_loadu_ps(r0 + x + 8));
            for (int t = 1; t < n; t++) {
                const float *r = r0 + (size_t)t * P; const __m256 w = _mm256_set1_ps(wi[t]);
                a0 = _mm256_add_ps(a0, _mm256_mul_ps(w, _mm256_loadu_ps(r + x))); a1 = _mm256_add_ps(a1, _mm256_mul_ps(w, _mm256_loadu_ps(r + x + 8)));
            }
            _mm256_storeu_ps(o + x, a0); _mm256_storeu_ps(o + x + 8, a1);
        }
#endif
        for (; x < P; x++) { float v = wi[0] * r0[x]; for (int t = 1; t < n; t++) v += wi[t] * r0[(size_t)t * P + x]; o[x] = v; }
    }
}
/* tmp: 2 P^2 floats. axis 2 transposes each plane, axis 1 runs on planes, axis 0 on (z, x) slabs. */
static void bop_apply(float *F, int P, int axis, const bop *b, float *tmp) {
    const size_t P2 = (size_t)P * P; float *t2 = tmp + P2;
    for (int a = 0; a < P; a++) {
        if (axis == 2) { transpose(F + a * P2, tmp, P); bop_rows(b, tmp, t2, P, P); transpose(t2, F + a * P2, P); }
        else if (axis == 1) { memcpy(tmp, F + a * P2, P2 * sizeof *tmp); bop_rows(b, tmp, F + a * P2, P, P); }
        else { for (int z = 0; z < P; z++) memcpy(tmp + (size_t)z * P, F + z * P2 + (size_t)a * P, P * sizeof *tmp); bop_rows(b, tmp, F + (size_t)a * P, P2, P); }
    }
}
/* bop_apply axis 1 (by) then axis 2 (bx) with the plane in cache between them: per z plane the same operations as the two full
   passes, bit for bit (either may be null: identity) */
static void bop_apply_yx(float *F, int P, const bop *by, const bop *bx, float *tmp) {
    const size_t P2 = (size_t)P * P; float *t2 = tmp + P2;
    if (!by && !bx) return;
    for (int a = 0; a < P; a++) {
        float *pl = F + a * P2;
        if (by) { memcpy(tmp, pl, P2 * sizeof *tmp); bop_rows(by, tmp, pl, P, P); }
        if (bx) { transpose(pl, tmp, P); bop_rows(bx, tmp, t2, P, P); transpose(t2, pl, P); }
    }
}
static int bop_is_identity(const bop *b) { for (int i = 0; i < b->P; i++) if (b->n[i] != 1 || b->s[i] != i || b->w[i][0] != 1) return 0; return 1; }

/* shift plane z >= zc content by (dy, dx); vacated voxels take fill */
static void shift_planes(void *v, size_t es, int P, int zc, int dy, int dx, const void *fill, void *tmp) {
    const size_t row = (size_t)P * es, plane = row * P;
    const int x0 = dx > 0 ? dx : 0, x1 = dx < 0 ? P + dx : P;   /* destination x range with a source */
    for (int z = zc; z < P; z++) {
        uint8_t *pl = (uint8_t *)v + z * plane; memcpy(tmp, pl, plane);
        for (int y = 0; y < P; y++) {
            uint8_t *d = pl + y * row; int sy = y - dy;
            for (int x = 0; x < P; x++) if (sy < 0 || sy >= P || x < x0 || x >= x1) memcpy(d + x * es, fill, es);
            if (sy >= 0 && sy < P && x1 > x0) memcpy(d + x0 * es, (uint8_t *)tmp + sy * row + (x0 - dx) * es, (size_t)(x1 - x0) * es);
        }
    }
}

/* GIN's random conv net over the float CT F: G = sum_ch w2 leaky(conv_k(normalised F) + b1), and the sums of G over the voxels Z
   leaves material (g1, g2, gn: z, y, x order). Rows of x with replicated edges, taps outer: each voxel accumulates its taps in the
   per-voxel loop's order (tests/test_scan_augment.c compares with that loop bit for bit). */
static void gin_eval(const float *F, const uint8_t *Z, int P, int C, int k, const float w1[4][27], const float b1[4], const float w2[4],
                     float im, float is, float *G, double *g1p, double *g2p, size_t *gnp) {
    const size_t P2 = (size_t)P * P, RP = (size_t)P + 2, PL = (size_t)P * RP; const int r = k / 2;
    double g1 = 0, g2 = 0; size_t gn = 0;
    /* the normalised planes in a ring of k slots (plane zz in slot zz % k: the k planes a z needs are distinct mod k, or repeat at
       the clamped edges), each source plane normalised once (was k^2 times, row by row), rows padded with their replicated ends */
    float *ring = malloc((size_t)k * PL * sizeof *ring); int slot_z[3] = {-1, -1, -1};
    if (!ring) abort();
    for (int z = 0; z < P; z++) {
        for (int dz = -r; dz <= r; dz++) {
            const int zz = z + dz < 0 ? 0 : z + dz >= P ? P - 1 : z + dz, s = zz % k;
            if (slot_z[s] == zz) continue;
            slot_z[s] = zz;
            for (int yy = 0; yy < P; yy++) {
                const float *src = F + zz * P2 + (size_t)yy * P; float *d = ring + (size_t)s * PL + (size_t)yy * RP + 1;
                for (int x = 0; x < P; x++) d[x] = (src[x] - im) * is;
                d[-1] = d[0]; d[P] = d[P - 1];
            }
        }
        for (int y = 0; y < P; y++) {
            const float *rowp[9];   /* (dz, dy) -> the padded row's first voxel */
            for (int dz = -r, q = 0; dz <= r; dz++) for (int dy = -r; dy <= r; dy++, q++) {
                const int zz = z + dz < 0 ? 0 : z + dz >= P ? P - 1 : z + dz, yy = y + dy < 0 ? 0 : y + dy >= P ? P - 1 : y + dy;
                rowp[q] = ring + (size_t)(zz % k) * PL + (size_t)yy * RP + 1;
            }
            float *g = G + z * P2 + (size_t)y * P;
            int x0 = 0;
#ifdef __AVX2__
            /* 8 voxels: each channel's sum in a register, from +0 through the taps in order; then the leaky sum from +0 (no contraction) */
            for (; x0 + 8 <= P; x0 += 8) {
                __m256 h[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
                for (int q = 0, t = 0; q < k * k; q++) for (int dx = -r; dx <= r; dx++, t++) {
                    const __m256 v = _mm256_loadu_ps(rowp[q] + x0 + dx);
                    for (int ch = 0; ch < C; ch++) h[ch] = _mm256_add_ps(h[ch], _mm256_mul_ps(_mm256_set1_ps(w1[ch][t]), v));
                }
                __m256 o = _mm256_setzero_ps();
                for (int ch = 0; ch < C; ch++) {
                    const __m256 u = _mm256_add_ps(h[ch], _mm256_set1_ps(b1[ch]));
                    const __m256 l = _mm256_blendv_ps(_mm256_mul_ps(_mm256_set1_ps(.2f), u), u, _mm256_cmp_ps(u, _mm256_setzero_ps(), _CMP_GT_OQ));
                    o = _mm256_add_ps(o, _mm256_mul_ps(_mm256_set1_ps(w2[ch]), l));
                }
                _mm256_storeu_ps(g + x0, o);
            }
#endif
            for (int x = x0; x < P; x++) {
                float h[4] = {0, 0, 0, 0};
                for (int q = 0, t = 0; q < k * k; q++) for (int dx = -r; dx <= r; dx++, t++) { const float v = rowp[q][x + dx]; for (int ch = 0; ch < C; ch++) h[ch] += w1[ch][t] * v; }
                float o = 0;
                for (int ch = 0; ch < C; ch++) { const float u = h[ch] + b1[ch]; o += w2[ch] * (u > 0 ? u : .2f * u); }
                g[x] = o;
            }
            const uint8_t *zr = Z + z * P2 + (size_t)y * P;
            for (int x = 0; x < P; x++) if (!zr[x]) { const float o = g[x]; g1 += o; g2 += (double)o * o; gn++; }
        }
    }
    free(ring);
    *g1p = g1; *g2p = g2; *gnp = gn;
}

/* SA_GIN's output volume, one per thread (2 MB aligned, huge pages requested), freed at thread exit */
typedef struct { float *g; size_t n; } gin_tls;
static pthread_key_t gin_key;
static pthread_once_t gin_once = PTHREAD_ONCE_INIT;
static void gin_free(void *p) { gin_tls *t = p; free(t->g); free(t); }
static void gin_init(void) { pthread_key_create(&gin_key, gin_free); }
static float *gin_buf(size_t n) {
    pthread_once(&gin_once, gin_init);
    gin_tls *t = pthread_getspecific(gin_key);
    if (!t) { if (!(t = calloc(1, sizeof *t)) || pthread_setspecific(gin_key, t)) abort(); }
    if (t->n < n) {
        const size_t gb = n * sizeof *t->g, H = (size_t)2 << 20;
        free(t->g); t->g = nullptr; t->n = 0;
        if (posix_memalign((void **)&t->g, H, gb)) abort();
#ifdef MADV_HUGEPAGE
        madvise(t->g, (gb + H - 1) & ~(H - 1), MADV_HUGEPAGE);
#endif
        t->n = n;
    }
    return t->g;
}
unsigned scan_aug_plan(const scan_aug_cfg *c, uint64_t seed, uint8_t *ctu, uint8_t *tgt, int nch, uint8_t *ign, int P,
                       const int64_t o[3], float *cy, float *cx, int hasax, const int chunk[3], int allow_seam, scan_plan *p) {
    p->on = 0; p->P = P;
    if (!scan_aug_enabled(c) || P < 4 || P > SA_MAXP) return 0;
    const double S = c->strength;
#define SO(i) (c->s[i] > 0 ? (double)c->s[i] : S)   /* this op's magnitude scale */
    const size_t P2 = (size_t)P * P, p3 = P2 * P;
    uint64_t rs = seed;
    unsigned on = 0;
    for (int i = 0; i < SA_N; i++) if (c->p[i] > 0 && uniform(&rs) < c->p[i]) on |= 1u << i;
    if (!allow_seam) on &= ~(1u << SA_SEAM);
    p->on = on;
    if (!on) return 0;
    p->ns = seed ^ UINT64_C(0x5ca1ab1e0ddba11);
    double sum = 0, sq = 0; size_t nz = 0; size_t hist[256] = {0};
    for (size_t k = 0; k < p3; k += 7) hist[ctu[k]]++;   /* magnitudes only need the statistics of a subsample */
    for (int v = 1; v < 256; v++) { nz += hist[v]; sum += (double)v * hist[v]; sq += (double)v * v * hist[v]; }
    const double mean = nz ? sum / nz : 128, sd = nz ? sqrt(fmax(sq / nz - mean * mean, 1)) : 32;   /* of the material (nonzero) voxels */
    p->mean = mean; p->sd = sd;
    if (on & 1u << SA_SEAM) {   /* stitched scan segments: a z slab moved in y/x, labels and axis with it (the input bytes: vacated voxels 0 = air) */
        int zc = P / 8 + (int)(uniform(&rs) * (P - P / 4)), m = 1 + (int)lround(2 * SO(SA_SEAM)), dy, dx;
        do { dy = (int)lround(sym1(&rs) * m); dx = (int)lround(sym1(&rs) * m); } while (!dy && !dx);
        const uint8_t one = 1, zero = 0; uint8_t *tmp = malloc(P2); if (!tmp) abort();
        shift_planes(ctu, 1, P, zc, dy, dx, &zero, tmp);
        shift_planes(ign, 1, P, zc, dy, dx, &one, tmp);
        for (int ch = 0; ch < nch; ch++) shift_planes(tgt + ch * p3, 1, P, zc, dy, dx, &zero, tmp);
        for (int z = zc; z < P; z++) { cy[z] += dy; cx[z] += dx; }
        free(tmp);
    }
    if (on & 1u << SA_AIR) {   /* unmasked scan: air (masked to 0) becomes dim noisy material, still unsupervised */
        size_t lo = 0, k1 = nz / 100; int v1 = 1; while (v1 < 255 && lo + hist[v1] <= k1) lo += hist[v1++];
        p->air_level = (float)(v1 * (.4 + .6 * uniform(&rs)));
        for (size_t k = 0; k < p3; k++) if (!ctu[k]) ign[k] = 1;
    }
    /* resolution: PSF blur or phase-contrast sharpening, then a coarser sampling grid, per axis */
    p->bop_on[0] = p->bop_on[1] = p->bop_on[2] = 0;
    if (on & (1u << SA_LOWRES | 1u << SA_BLUR | 1u << SA_SHARPEN | 1u << SA_MOTION)) {
        bop *op = malloc(3 * sizeof *op); float *dense = calloc(P, sizeof *dense); if (!op || !dense) abort();
        int iso = uniform(&rs) < .5;
        const double SL = SO(SA_LOWRES), SB = SO(SA_BLUR), SS = SO(SA_SHARPEN);
        double f0 = 1 + uniform(&rs) * 1.5 * SL, sb0 = .3 + uniform(&rs) * .7 * SB, sa0 = .15 + uniform(&rs) * .3 * SS;
        int maxis = -1, mmode = 0, mne = 0; double mt[2] = {0, 0}, mw[2] = {0, 0}, mL = 0;   /* motion (drawn after the old ops' draws) */
        if (on & 1u << SA_MOTION) {
            const double SM = SO(SA_MOTION);
            maxis = (int)(uniform(&rs) * 3); if (maxis > 2) maxis = 2;
            mmode = uniform(&rs) < .5; mne = 1 + (uniform(&rs) < .4);
            for (int e = 0; e < mne; e++) { mt[e] = (1 + 3 * uniform(&rs)) * SM * (uniform(&rs) < .5 ? -1 : 1); mw[e] = .1 + .25 * uniform(&rs); }
            mL = (1.5 + 3.5 * uniform(&rs)) * SM;
            for (int e = 0; e < mne; e++) mt[e] = fmax(-6, fmin(6, mt[e]));
            mL = fmin(mL, 8);
        }
        for (int d = 0; d < 3; d++) {
            double f = iso ? f0 : 1 + uniform(&rs) * 1.5 * SL, sb = iso ? sb0 : .3 + uniform(&rs) * .7 * SB, sa = iso ? sa0 : .15 + uniform(&rs) * .3 * SS;
            double ss = .7 + .6 * uniform(&rs), phase = uniform(&rs) * f;
            f = fmin(f, 4); sb = fmin(sb, 2.3);
            bop_identity(&op[0], P);
            if (on & 1u << SA_BLUR) { bop_gauss(&op[1], P, sb, -1, dense); bop_compose(&op[2], &op[1], &op[0], dense); op[0] = op[2]; }
            if (on & 1u << SA_SHARPEN) { bop_gauss(&op[1], P, ss, sa, dense); bop_compose(&op[2], &op[1], &op[0], dense); op[0] = op[2]; }
            if (d == maxis && (mmode || mne)) { bop_motion(&op[1], P, mmode, mne, mt, mw, mL, dense); bop_compose(&op[2], &op[1], &op[0], dense); op[0] = op[2]; }
            if (on & 1u << SA_LOWRES && f > 1.01) { bop_lowres(&op[1], P, f, phase, dense); bop_compose(&op[2], &op[1], &op[0], dense); op[0] = op[2]; }
            if (bop_is_identity(&op[0])) continue;
            p->bop[d] = op[0]; p->bop_on[d] = 1;
        }
        free(op); free(dense);
    }
    if (on & 1u << SA_BIAS) {   /* smooth multiplicative field: exp(N(0, b)) on a 3..5 grid, linearly interpolated */
        int g = 3 + (int)(uniform(&rs) * 3); double b = (.08 + .12 * uniform(&rs)) * SO(SA_BIAS);
        p->bias_g = g; for (int k = 0; k < g * g * g; k++) p->bias_G[k] = (float)exp(b * sym1(&rs) * 1.7320508);
    }
    if (on & (1u << SA_RING | 1u << SA_CUPPING)) {   /* radial profile about the scan axis: rings + cupping */
        float *ccy = p->ccy, *ccx = p->ccx;
        if (hasax) for (int z = 0; z < P; z++) { ccy[z] = cy[z] - (float)o[1]; ccx[z] = cx[z] - (float)o[2]; }
        else { double t = 2 * M_PI * uniform(&rs), D = P * (1 + 3 * uniform(&rs)); for (int z = 0; z < P; z++) { ccy[z] = (float)(.5 * P + D * sin(t)); ccx[z] = (float)(.5 * P + D * cos(t)); } }
        float rmin = INFINITY, rmax = 0;
        for (int z = 0; z < P; z++) {
            float dy = fmaxf(0, fmaxf(-ccy[z], ccy[z] - (P - 1))), dx = fmaxf(0, fmaxf(-ccx[z], ccx[z] - (P - 1)));
            float fy = fmaxf(fabsf(ccy[z]), fabsf(ccy[z] - (P - 1))), fx = fmaxf(fabsf(ccx[z]), fabsf(ccx[z] - (P - 1)));
            rmin = fminf(rmin, sqrtf(dy * dy + dx * dx)); rmax = fmaxf(rmax, sqrtf(fy * fy + fx * fx));
        }
        const float res = 4; int nt = (int)((rmax - rmin) * res) + 3;
        if (nt > SA_MAXTAB) { fprintf(stderr, "scan_aug: radial table of %d entries (> %d)\n", nt, SA_MAXTAB); abort(); }
        float *tab = p->tab; memset(tab, 0, (size_t)nt * sizeof *tab);
        p->rmin = rmin; p->res = res; p->nt = nt;
        if (on & 1u << SA_RING) {
            int nr = 3 + (int)fmin(512, (rmax - rmin) * (.02 + .1 * uniform(&rs)) * SO(SA_RING));   /* dense: one per miscalibrated detector column */ double amp = (.05 + .15 * uniform(&rs)) * SO(SA_RING) * sd;
            for (int i = 0; i < nr; i++) {
                double r0 = rmin + uniform(&rs) * (rmax - rmin), w = .4 + 1.1 * uniform(&rs), a = amp * sym1(&rs) * 1.7320508;
                for (int t = (int)fmax(0, (r0 - 4 * w - rmin) * res); t < nt && rmin + t / res < r0 + 4 * w; t++) { double d = (rmin + t / res - r0) / w; tab[t] += (float)(a * exp(-.5 * d * d)); }
            }
        }
        if (on & 1u << SA_CUPPING) {   /* beam hardening: a smooth radial bowl, here its local quadratic over the patch */
            double a = (.1 + .3 * uniform(&rs)) * SO(SA_CUPPING) * sd * (uniform(&rs) < .5 ? -1 : 1), lin = sym1(&rs);
            for (int t = 0; t < nt; t++) { double u = 2 * (t / res) / fmax(rmax - rmin, 1) - 1; tab[t] += (float)(a * (u * u + lin * u)); }
        }
    }
    if (on & 1u << SA_SLICE) {   /* per-slice gain/offset along the scan axis (beam current / detector drift), AR(1) in z */
        double rho = .9 * uniform(&rs), q = sqrt(1 - rho * rho), ga = .03 * SO(SA_SLICE), oa = .08 * SO(SA_SLICE) * sd, gz = 0, oz = 0;
        for (int z = 0; z < P; z++) {
            gz = z ? rho * gz + q * sym1(&rs) * 1.7320508 : sym1(&rs) * 1.7320508; oz = z ? rho * oz + q * sym1(&rs) * 1.7320508 : sym1(&rs) * 1.7320508;
            p->slice_gain[z] = (float)exp(ga * gz); p->slice_off[z] = (float)(oa * oz);
        }
    }
    if (on & 1u << SA_NOISE) {   /* white noise, partly signal-dependent (sigma ~ sqrt(value)) */
        p->noise_s0 = (float)((.05 + .2 * uniform(&rs)) * SO(SA_NOISE) * sd); p->noise_mix = (float)uniform(&rs); p->noise_im = (float)(1 / mean);
    }
    if (on & 1u << SA_CNOISE) {   /* correlated noise: unit-variance AR(1) along x, y, z (ct_augment's generator) */
        p->cnoise_rho = (float)(.5 + .35 * uniform(&rs));
        p->cnoise_s0 = (float)((.05 + .15 * uniform(&rs)) * SO(SA_CNOISE) * sd);
    }
    if (on & 1u << SA_PAGANIN) {   /* phase-retrieval-like low-pass after the noise: per axis exp(-|t| / lambda) */
        const double lam = (.4 + 1.2 * uniform(&rs)) * SO(SA_PAGANIN); p->pag_a = (float)exp(-1 / fmax(lam, .05));
    }
    if (on & 1u << SA_GIN) {   /* random shallow conv net (GIN-style), positive-biased centre taps keep the structure */
        const int C = 2 + (int)(uniform(&rs) * 3), k = uniform(&rs) < .5 ? 1 : 3, kk = k * k * k;
        p->gin_C = C; p->gin_k = k;
        for (int ch = 0; ch < C; ch++) {
            for (int t = 0; t < kk; t++) p->gin_w1[ch][t] = (float)(sym1(&rs) * 1.7320508 / sqrt(kk));
            p->gin_w1[ch][kk / 2] += 1.f; p->gin_b1[ch] = (float)(.5 * sym1(&rs)); p->gin_w2[ch] = (float)(.2 + fabs(sym1(&rs)));
        }
        p->gin_alpha = fmin(1, (.3 + .7 * uniform(&rs)) * SO(SA_GIN));
    }
    float *lut = p->lut; p->tone = 0;
    for (int i = 0; i < 257; i++) lut[i] = (float)(i < 256 ? i : 255);
    if (on & 1u << SA_TONE) {   /* reconstruction / conversion to 8 bits: gamma, random monotone remap, narrower window */
        const double ST = SO(SA_TONE); double g = exp(sym1(&rs) * .35 * ST), kn[6] = {0}, inc[5], tot = 0;
        for (int i = 0; i < 5; i++) tot += inc[i] = exp(sym1(&rs) * .5 * ST);
        for (int i = 0; i < 5; i++) kn[i + 1] = kn[i] + 255 * inc[i] / tot;
        double lo = 0, hi = 255;
        if (uniform(&rs) < .5) { lo = fmax(0, mean - sd * (4 - 1.25 * ST * uniform(&rs))); hi = fmin(255, mean + sd * (4 - 1.25 * ST * uniform(&rs))); }
        for (int i = 0; i < 256; i++) {
            double v = 255 * pow(i / 255., g), u = v / 51; int j = (int)u; if (j > 4) j = 4;
            v = kn[j] + (u - j) * (kn[j + 1] - kn[j]);
            lut[i] = (float)(255 * fmin(1, fmax(0, (v - lo) / fmax(hi - lo, 1))));
        }
        lut[256] = lut[255]; p->tone = 1;
    }
    p->step = 0;
    if (on & 1u << SA_QUANT) { int drop = 1 + (int)(uniform(&rs) * 2 * SO(SA_QUANT)); if (drop > 4) drop = 4; p->step = (float)(1 << drop); }
    p->ndrop = 0;
    if (on & 1u << SA_DROPSLICE) {   /* missing / corrupt reconstructed slices: repeated neighbour, constant or pure noise */
        int n = 1 + (int)(uniform(&rs) * 2 * SO(SA_DROPSLICE));
        if (n > SA_MAXBOX) abort();   /* (n <= 5 for strength <= 2) */
        for (int i = 0; i < n; i++) {
            int z = (int)(uniform(&rs) * P), mode = (int)(uniform(&rs) * 3);
            p->drop_z[i] = z; p->drop_mode[i] = mode;
            if (c->drop_ignore) memset(ign + z * P2, 1, P2);
        }
        p->ndrop = n;
    }
    p->ncut = 0;
    if (on & 1u << SA_CUTOUT) {   /* occlusion: boxes of constant or noise */
        int n = 1 + (int)(uniform(&rs) * 2 * SO(SA_CUTOUT));
        if (n > SA_MAXBOX) abort();
        for (int i = 0; i < n; i++) {
            int *b0 = p->cut_b0[i], *b1 = p->cut_b1[i], noisy = uniform(&rs) < .5;
            for (int d = 0; d < 3; d++) { int e = P / 16 + (int)(uniform(&rs) * P / 8 * SO(SA_CUTOUT)); if (e > P / 2) e = P / 2; b0[d] = (int)(uniform(&rs) * (P - e)); b1[d] = b0[d] + e; }
            p->cut_noisy[i] = noisy;
            if (c->drop_ignore) for (int z = b0[0]; z < b1[0]; z++) for (int y = b0[1]; y < b1[1]; y++) memset(ign + z * P2 + (size_t)y * P + b0[2], 1, b1[2] - b0[2]);
        }
        p->ncut = n;
    }
    p->nchunk = 0;
    if (on & 1u << SA_CHUNK) {   /* missing zarr chunks read as the fill value 0: unsupervised through the CT > 0 mask */
        int n = 1 + (int)(uniform(&rs) * 2 * SO(SA_CHUNK)), e[3];
        if (n > SA_MAXBOX) abort();
        for (int d = 0; d < 3; d++) { e[d] = chunk[d] > 0 && chunk[d] <= P / 2 ? chunk[d] : P / 4; if (e[d] < 1) e[d] = 1; }
        for (int i = 0; i < n; i++) {
            int *b0 = p->chunk_b0[i], *b1 = p->chunk_b1[i];
            for (int d = 0; d < 3; d++) {   /* a chunk-grid cell (global grid) that meets the patch */
                int64_t g0 = (o[d] / e[d]) * e[d], cells = (o[d] + P - 1) / e[d] - o[d] / e[d] + 1, a = g0 + (int64_t)(uniform(&rs) * cells) * e[d] - o[d];
                b0[d] = a < 0 ? 0 : (int)a; b1[d] = a + e[d] > P ? P : (int)(a + e[d]);
            }
        }
        p->nchunk = n;
    }
    p->hist_on = (on & 1u << SA_HIST) && c->nhist > 0;
    if (p->hist_on) {   /* the window's material histogram onto a target scan's (quantile mapping), blended */
        const float *ct = c->hist + (size_t)(uniform(&rs) * c->nhist) * 256; p->hist_alpha = fmin(1, (.5 + .5 * uniform(&rs)) * SO(SA_HIST));
        memcpy(p->hist_ct, ct, sizeof p->hist_ct);
    }
#undef SO
    return on;
}
void scan_plan_air(const scan_plan *p, const uint8_t *ctu, uint8_t *air) {
    const int P = p->P; const size_t P2 = (size_t)P * P, p3 = P2 * P;
    if (p->on & 1u << SA_AIR) memset(air, 0, p3); else for (size_t k = 0; k < p3; k++) air[k] = !ctu[k];
    for (int i = 0; i < p->nchunk; i++) {
        const int *b0 = p->chunk_b0[i], *b1 = p->chunk_b1[i];
        for (int z = b0[0]; z < b1[0]; z++) for (int y = b0[1]; y < b1[1]; y++) memset(air + z * P2 + (size_t)y * P + b0[2], 1, b1[2] - b0[2]);
    }
}
void scan_aug_exec(const scan_plan *p, uint8_t *ctu, void *scratch) {
    const unsigned on = p->on;
    if (!on) return;
    const int P = p->P; const size_t P2 = (size_t)P * P, p3 = P2 * P;
    const double mean = p->mean, sd = p->sd;
    uint64_t ns = p->ns;
    float *F = scratch; uint8_t *Z = (uint8_t *)scratch + 4 * p3;   /* float CT; 1 = the output is air (0) */
    float *tmp = malloc(2 * P2 * sizeof *tmp), *row = malloc(P * sizeof *row);
    if (!tmp || !row) abort();
    size_t k0 = 0;
#ifdef __AVX2__
    for (; k0 + 16 <= p3; k0 += 16) {   /* the bytes as floats, and the air flags */
        const __m128i b = _mm_loadu_si128((const __m128i *)(ctu + k0));
        _mm256_storeu_ps(F + k0, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(b)));
        _mm256_storeu_ps(F + k0 + 8, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(b, 8))));
        _mm_storeu_si128((__m128i *)(Z + k0), _mm_and_si128(_mm_cmpeq_epi8(b, _mm_setzero_si128()), _mm_set1_epi8(1)));
    }
#endif
    for (size_t k = k0; k < p3; k++) { F[k] = ctu[k]; Z[k] = !ctu[k]; }
    if (on & 1u << SA_AIR) { const float level = p->air_level; for (size_t k = 0; k < p3; k++) if (Z[k]) { F[k] = level; Z[k] = 0; } }
    if (p->bop_on[0]) bop_apply(F, P, 0, &p->bop[0], tmp);
    bop_apply_yx(F, P, p->bop_on[1] ? &p->bop[1] : nullptr, p->bop_on[2] ? &p->bop[2] : nullptr, tmp);
    if (on & 1u << SA_BIAS) {   /* smooth multiplicative field: exp(N(0, b)) on a 3..5 grid, linearly interpolated */
        const int g = p->bias_g; const float *G = p->bias_G;
        int *ix = malloc(P * sizeof *ix); float *wx = malloc(P * sizeof *wx), crow[5]; if (!ix || !wx) abort();
        const float sc = (float)(g - 1) / (float)(P - 1);
        for (int x = 0; x < P; x++) { float u = x * sc; int i = (int)u; if (i > g - 2) i = g - 2; ix[x] = i; wx[x] = u - i; }
        for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) {
            int iz = ix[z], iy = ix[y]; float fz = wx[z], fy = wx[y];
            for (int j = 0; j < g; j++) {
                const float *q = G + ((size_t)iz * g + iy) * g + j;
                crow[j] = (1 - fz) * ((1 - fy) * q[0] + fy * q[g]) + fz * ((1 - fy) * q[g * g] + fy * q[g * g + g]);
            }
            float *r = F + z * P2 + (size_t)y * P;
            for (int x = 0; x < P; x++) r[x] *= crow[ix[x]] + wx[x] * (crow[ix[x] + 1] - crow[ix[x]]);
        }
        free(ix); free(wx);
    }
    if (on & (1u << SA_RING | 1u << SA_CUPPING)) {   /* radial profile about the scan axis: rings + cupping */
        const float *ccy = p->ccy, *ccx = p->ccx, *tab = p->tab, rmin = p->rmin, res = p->res; const int nt = p->nt;
        for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) {
            float *r = F + z * P2 + (size_t)y * P, dy = y - ccy[z];
            int x = 0;
#ifdef __AVX2__
            {   /* 8 lanes, the same float operations (IEEE sqrt, the clamps as selects) and the table through gathers */
                const __m256 vdy2 = _mm256_mul_ps(_mm256_set1_ps(dy), _mm256_set1_ps(dy)), vcx = _mm256_set1_ps(ccx[z]), vrmin = _mm256_set1_ps(rmin),
                             vres = _mm256_set1_ps(res), zero = _mm256_setzero_ps(), top = _mm256_set1_ps((float)(nt - 2));
                const __m256 step = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
                for (; x + 8 <= P; x += 8) {
                    const __m256 dx = _mm256_sub_ps(_mm256_add_ps(_mm256_set1_ps((float)x), step), vcx);
                    __m256 u = _mm256_mul_ps(_mm256_sub_ps(_mm256_sqrt_ps(_mm256_add_ps(vdy2, _mm256_mul_ps(dx, dx))), vrmin), vres);
                    u = _mm256_blendv_ps(u, zero, _mm256_cmp_ps(u, zero, _CMP_LT_OQ));
                    u = _mm256_blendv_ps(u, top, _mm256_cmp_ps(u, top, _CMP_GT_OQ));
                    const __m256i i = _mm256_cvttps_epi32(u);
                    const __m256 t0 = _mm256_i32gather_ps(tab, i, 4), t1 = _mm256_i32gather_ps(tab + 1, i, 4);
                    const __m256 add = _mm256_add_ps(t0, _mm256_mul_ps(_mm256_sub_ps(u, _mm256_cvtepi32_ps(i)), _mm256_sub_ps(t1, t0)));
                    _mm256_storeu_ps(r + x, _mm256_add_ps(_mm256_loadu_ps(r + x), add));
                }
            }
#endif
            for (; x < P; x++) {
                float dx = x - ccx[z], u = (sqrtf(dy * dy + dx * dx) - rmin) * res; if (u < 0) u = 0; if (u > nt - 2) u = nt - 2;
                int i = (int)u; r[x] += tab[i] + (u - i) * (tab[i + 1] - tab[i]);
            }
        }
    }
    if (on & 1u << SA_SLICE) for (int z = 0; z < P; z++) {   /* per-slice gain/offset along the scan axis */
        const float gain = p->slice_gain[z], off = p->slice_off[z]; float *pl = F + z * P2;
        for (size_t k = 0; k < P2; k++) pl[k] = pl[k] * gain + off;
    }
    if (on & 1u << SA_NOISE) {   /* white noise, partly signal-dependent (sigma ~ sqrt(value)) */
        const float s0 = p->noise_s0, mix = p->noise_mix, im = p->noise_im;
        for (size_t r = 0; r < P2; r++) {
            float *v = F + r * P; gauss_row(&ns, row, P);
            for (int x = 0; x < P; x++) v[x] += s0 * row[x] * ((1 - mix) + mix * sqrtf(fmax_c(v[x], 0) * im));
        }
    }
    if (on & 1u << SA_CNOISE) {   /* correlated noise: unit-variance AR(1) along x, y, z (ct_augment's generator) */
        ct_aug_plan q; memset(&q, 0, sizeof q); q.noise_rho = p->cnoise_rho;
        const float s0 = p->cnoise_s0;
        float *prow = malloc(P * sizeof *prow), *pplane = malloc(P2 * sizeof *pplane); if (!prow || !pplane) abort();
        for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) {
            gauss_row(&ns, row, P); ct_aug_noise_row(&q, row, prow, pplane, P, z, y);
            float *v = F + z * P2 + (size_t)y * P; for (int x = 0; x < P; x++) v[x] += s0 * row[x];
        }
        free(prow); free(pplane);
    }
    if (on & 1u << SA_PAGANIN) { const float a = p->pag_a; exp_smooth(F, P, 0, a, row, tmp); exp_smooth_yx(F, P, a, row, tmp); }
    if (on & 1u << SA_GIN) {   /* random shallow conv net (GIN-style) */
        const double alpha = p->gin_alpha;
        float *G = gin_buf(p3);   /* (P^3 floats kept per thread: a fresh 512 MB at P512 was faulted in and unmapped per use) */
        const float im = (float)mean, is = (float)(1 / sd);
        double g1 = 0, g2 = 0; size_t gn = 0;
        gin_eval(F, Z, P, p->gin_C, p->gin_k, p->gin_w1, p->gin_b1, p->gin_w2, im, is, G, &g1, &g2, &gn);
        const double gm = gn ? g1 / gn : 0, gs = gn ? sqrt(fmax(g2 / gn - gm * gm, 1e-12)) : 1;
        for (size_t i = 0; i < p3; i++) F[i] = (float)(alpha * (mean + sd * (G[i] - gm) / gs) + (1 - alpha) * F[i]);
    }
    const float *lut = p->lut; const int tone = p->tone; const float step = p->step;
    for (int i = 0; i < p->ndrop; i++) {   /* missing / corrupt reconstructed slices: repeated neighbour, constant or pure noise */
        const int z = p->drop_z[i], mode = p->drop_mode[i]; float *pl = F + z * P2;
        if (mode == 0) memcpy(pl, F + (size_t)(z ? z - 1 : 1) * P2, P2 * sizeof *pl);
        else for (int r = 0; r < P; r++) { gauss_row(&ns, row, P); for (int x = 0; x < P; x++) pl[(size_t)r * P + x] = (float)(mean + (mode == 2 ? 1.5 * sd * row[x] : 0)); }
    }
    for (int i = 0; i < p->ncut; i++) {   /* occlusion: boxes of constant or noise */
        const int *b0 = p->cut_b0[i], *b1 = p->cut_b1[i], noisy = p->cut_noisy[i];
        for (int z = b0[0]; z < b1[0]; z++) for (int y = b0[1]; y < b1[1]; y++) {
            float *v = F + z * P2 + (size_t)y * P;
            if (noisy) gauss_row(&ns, row, P);
            for (int x = b0[2]; x < b1[2]; x++) v[x] = (float)(mean + (noisy ? sd * row[x] : 0));
        }
    }
    for (int i = 0; i < p->nchunk; i++) {   /* missing zarr chunks read as the fill value 0 */
        const int *b0 = p->chunk_b0[i], *b1 = p->chunk_b1[i];
        for (int z = b0[0]; z < b1[0]; z++) for (int y = b0[1]; y < b1[1]; y++) memset(Z + z * P2 + (size_t)y * P + b0[2], 1, b1[2] - b0[2]);
    }
    const int hist_on = p->hist_on;
    if (tone && hist_on) for (size_t k = 0; k < p3; k++) { float v = fmin_c(fmax_c(F[k], 0), 255); int i = (int)v; F[k] = lut[i] + (v - i) * (lut[i + 1] - lut[i]); }
    if (hist_on) {   /* the window's material histogram onto a target scan's (quantile mapping), blended */
        const float *ct = p->hist_ct; const double alpha = p->hist_alpha;
        double hw[257] = {0}, nw = 0;
        for (size_t k = 0; k < p3; k += 7) if (!Z[k]) { float v = fmin_c(fmax_c(F[k], 1), 255); hw[(int)v] += 1; nw++; }
        if (nw > 0) {
            float hm[257]; double acc = 0; int t = 1;
            for (int v = 0; v < 256; v++) {
                acc += hw[v]; const double q = acc / nw;
                while (t < 255 && ct[t] < q) t++;
                const double lo = ct[t - 1], hi = ct[t], f = hi > lo ? (q - lo) / (hi - lo) : 1;
                hm[v] = (float)(t - 1 + fmin(1, fmax(0, f)));
            }
            hm[256] = hm[255];
            for (size_t k = 0; k < p3; k++) { float v = fmin_c(fmax_c(F[k], 0), 255); int i = (int)v; const float m = hm[i] + (v - i) * (hm[i + 1] - hm[i]); F[k] = (float)(alpha * m + (1 - alpha) * F[k]); }
        }
    }
    /* the tone curve (unless the histogram needed it first), the quantisation and the byte conversion: pointwise, in that order,
       one pass */
    const int tone_now = tone && !hist_on; const float is = step > 0 ? 1 / step : 0;
    size_t kf = 0;
#ifdef __AVX2__
    {   /* 8 lanes: the clamps as selects with fmax_c / fmin_c's comparisons, the tone table through gathers, floorf as round-down */
        const __m256 zero = _mm256_setzero_ps(), c255 = _mm256_set1_ps(255), half = _mm256_set1_ps(.5f), one = _mm256_set1_ps(1), top = _mm256_set1_ps(255.5f),
                     vis = _mm256_set1_ps(is), vstep = _mm256_set1_ps(step);
        for (; kf + 8 <= p3; kf += 8) {
            __m256 f = _mm256_loadu_ps(F + kf);
            if (tone_now) {
                __m256 v = _mm256_blendv_ps(zero, f, _mm256_cmp_ps(f, zero, _CMP_GE_OQ));
                v = _mm256_blendv_ps(c255, v, _mm256_cmp_ps(v, c255, _CMP_LE_OQ));
                const __m256i i = _mm256_cvttps_epi32(v);
                const __m256 l0 = _mm256_i32gather_ps(lut, i, 4), l1 = _mm256_i32gather_ps(lut + 1, i, 4);
                f = _mm256_add_ps(l0, _mm256_mul_ps(_mm256_sub_ps(v, _mm256_cvtepi32_ps(i)), _mm256_sub_ps(l1, l0)));
            }
            if (step > 0) f = _mm256_mul_ps(_mm256_add_ps(_mm256_floor_ps(_mm256_mul_ps(f, vis)), half), vstep);
            __m256 v = _mm256_add_ps(f, half);
            v = _mm256_blendv_ps(one, v, _mm256_cmp_ps(v, one, _CMP_GE_OQ));
            v = _mm256_blendv_ps(top, v, _mm256_cmp_ps(v, top, _CMP_LE_OQ));
            __m256i q = _mm256_cvttps_epi32(v);   /* 1 .. 255 */
            uint64_t zb; memcpy(&zb, Z + kf, 8);
            const __m256i zm = _mm256_cmpeq_epi32(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)zb)), _mm256_setzero_si256());
            q = _mm256_and_si256(q, zm);   /* Z: air 0 */
            const __m256i p = _mm256_packus_epi16(_mm256_packus_epi32(q, q), _mm256_setzero_si256());
            const uint64_t qb = (uint64_t)(uint32_t)_mm_cvtsi128_si32(_mm256_castsi256_si128(p)) | (uint64_t)(uint32_t)_mm_cvtsi128_si32(_mm256_extracti128_si256(p, 1)) << 32;
            memcpy(ctu + kf, &qb, 8);
        }
    }
#endif
    for (size_t k = kf; k < p3; k++) {
        float f = F[k];
        if (tone_now) { const float v = fmin_c(fmax_c(f, 0), 255); const int i = (int)v; f = lut[i] + (v - i) * (lut[i + 1] - lut[i]); }
        if (step > 0) f = (floorf(f * is) + .5f) * step;
        const float v = fmin_c(fmax_c(f + .5f, 1), 255.5f); const uint8_t q = (uint8_t)(int)v; ctu[k] = Z[k] ? 0 : q;
    }
    free(tmp); free(row);
}
unsigned scan_aug_apply(const scan_aug_cfg *c, uint64_t seed, uint8_t *ctu, uint8_t *tgt, int nch, uint8_t *ign, int P,
                        const int64_t o[3], float *cy, float *cx, int hasax, const int chunk[3], int allow_seam, void *scratch) {
    if (!scan_aug_enabled(c) || P < 4 || P > SA_MAXP) return 0;
    scan_plan *p = malloc(sizeof *p); if (!p) abort();
    const unsigned on = scan_aug_plan(c, seed, ctu, tgt, nch, ign, P, o, cy, cx, hasax, chunk, allow_seam, p);
    if (on) scan_aug_exec(p, ctu, scratch);
    free(p);
    return on;
}
