/* fm_tol.h - comparisons for the checks that are bit-exact only under UFSM_STRICT_MATH (DESIGN 2026-11-14 "Fast math everywhere").
 *
 * The default build is fast math on both sides: host C -ffast-math -fno-finite-math-only (reassociation, contraction, reciprocal
 * division) and every CUDA unit, the former UFSM_STRICT_UNITS included, -use_fast_math. A GPU result and its host reference, or
 * two host code paths that run the same float math in different code, then round differently on a few elements. These helpers
 * take a bound for the default build; under -DUFSM_STRICT_MATH=ON (CMake defines UFSM_STRICT_MATH) every bound collapses to
 * exact, so the strict build keeps the old byte-identity checks unchanged.
 *
 * Bounds are per check, from the arithmetic involved (no measurements were possible when they were written: the 11-14 sweep had
 * the GPUs): a u8 that is round(float) moves by at most 1 when the float moves by a few ulps; how many elements may move is a
 * fraction of n (elements whose pre-rounding value lies within a few ulps of a .5 boundary). Classes cut at a float threshold
 * (inside / outside a radius, known / unknown) flip whole, hence the separate count bound fm_cmp_u8 takes.
 *
 * Every helper prints its worst element and its count of differing elements, and returns 0 when the check passes. */
#ifndef UFSM_FM_TOL_H
#define UFSM_FM_TOL_H
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#ifdef UFSM_STRICT_MATH
#define FM_STRICT 1
#else
#define FM_STRICT 0
#endif
/* a bound under the default fast-math build, 0 under UFSM_STRICT_MATH */
#define FM_TOL(x) (FM_STRICT ? 0 : (x))

/* elements allowed to differ: ceil(frac * n) (frac 0 -> 0) */
static inline size_t fm_allow(size_t n, double frac) { return FM_STRICT || frac <= 0 ? 0 : (size_t)ceil(frac * (double)n); }

/* u8: at most ceil(frac * n) elements differ at all, and of those at most ceil(flip * n) by more than maxd (class flips at a float
   threshold: inside / outside a radius, known / unknown). cyc > 0: values < cyc are cyclic codes (phase mod cyc), compared by
   cyclic distance; a difference into / out of a value >= cyc (e.g. 255 "unknown") is a flip. Exact under UFSM_STRICT_MATH. */
static inline int fm_cmp_u8c(const char *what, const uint8_t *a, const uint8_t *b, size_t n, int maxd, int cyc, double frac, double flip) {
    size_t nd = 0, nbig = 0, first = (size_t)-1; int worst = 0;
    for (size_t i = 0; i < n; i++) {
        if (a[i] == b[i]) continue;
        if (first == (size_t)-1) first = i;
        nd++;
        int d = a[i] > b[i] ? a[i] - b[i] : b[i] - a[i];
        if (cyc > 0) d = a[i] < cyc && b[i] < cyc ? (d < cyc - d ? d : cyc - d) : 256;
        if (d > worst) worst = d;
        if (d > maxd) nbig++;
    }
    const size_t allow = fm_allow(n, frac), allow_big = fm_allow(n, flip);
    const int bad = nd > allow || nbig > allow_big;
    if (nd)
        fprintf(stderr, "%s %s: %zu of %zu differ (allowed %zu), %zu by more than %d (allowed %zu), worst %d, first at %zu%s\n", bad ? "FAIL" : "ok",
                what, nd, n, allow, nbig, maxd, allow_big, worst, first, FM_STRICT ? " [strict]" : "");
    return bad;
}
/* u8 within +-maxd, at most ceil(frac * n) elements differing, none by more than maxd */
static inline int fm_cmp_u8(const char *what, const uint8_t *a, const uint8_t *b, size_t n, int maxd, double frac) {
    return fm_cmp_u8c(what, a, b, n, maxd, 0, frac, 0);
}

/* floats: |a - b| <= abs_tol + rel * max(|a|, |b|) on every element; NaN must meet NaN, inf the same inf. */
static inline int fm_close(double a, double b, double rel, double abs_tol) {
    if (isnan(a) || isnan(b)) return isnan(a) && isnan(b);
    if (isinf(a) || isinf(b)) return a == b;
    return fabs(a - b) <= abs_tol + rel * fmax(fabs(a), fabs(b));
}
static inline int fm_cmp_f32(const char *what, const float *a, const float *b, size_t n, double rel, double abs_tol) {
    if (FM_STRICT) return memcmp(a, b, n * sizeof *a) ? (fprintf(stderr, "FAIL %s: not bit-identical [strict]\n", what), 1) : 0;
    size_t nbad = 0, wi = 0; double worst = 0;
    for (size_t i = 0; i < n; i++) {
        if (!fm_close(a[i], b[i], rel, abs_tol)) nbad++;
        const double d = fabs((double)a[i] - b[i]);
        if (d > worst) worst = d, wi = i;
    }
    if (nbad) fprintf(stderr, "FAIL %s: %zu of %zu beyond rel %g abs %g; worst |d| %g at %zu (%g vs %g)\n", what, nbad, n, rel, abs_tol, worst, wi,
                      (double)a[wi], (double)b[wi]);
    return nbad != 0;
}
static inline int fm_cmp_f64(const char *what, const double *a, const double *b, size_t n, double rel, double abs_tol) {
    if (FM_STRICT) return memcmp(a, b, n * sizeof *a) ? (fprintf(stderr, "FAIL %s: not bit-identical [strict]\n", what), 1) : 0;
    size_t nbad = 0, wi = 0; double worst = 0;
    for (size_t i = 0; i < n; i++) {
        if (!fm_close(a[i], b[i], rel, abs_tol)) nbad++;
        const double d = fabs(a[i] - b[i]);
        if (d > worst) worst = d, wi = i;
    }
    if (nbad) fprintf(stderr, "FAIL %s: %zu of %zu beyond rel %g abs %g; worst |d| %g at %zu (%g vs %g)\n", what, nbad, n, rel, abs_tol, worst, wi, a[wi], b[wi]);
    return nbad != 0;
}

/* IEEE binary16 bit patterns: within `ulps` units in the last place (sign-magnitude order; +0 / -0 equal), NaN meets NaN. */
static inline int32_t fm_h16_ord(uint16_t h) { return (h & 0x8000u) ? -(int32_t)(h & 0x7fffu) : (int32_t)h; }
static inline int fm_cmp_h16(const char *what, const uint16_t *a, const uint16_t *b, size_t n, int ulps) {
    if (FM_STRICT) ulps = 0;
    size_t nbad = 0, nd = 0, wi = 0; int32_t worst = 0;
    for (size_t i = 0; i < n; i++) {
        if (a[i] == b[i]) continue;
        const int na = (a[i] & 0x7c00u) == 0x7c00u && (a[i] & 0x3ffu), nb = (b[i] & 0x7c00u) == 0x7c00u && (b[i] & 0x3ffu);
        if (na || nb) { if (!(na && nb)) nbad++; continue; }
        int32_t d = fm_h16_ord(a[i]) - fm_h16_ord(b[i]); d = d < 0 ? -d : d;
        if (d) nd++;
        if (d > worst) worst = d, wi = i;
        if (d > ulps) nbad++;
    }
    if (nbad || nd)
        fprintf(stderr, "%s %s: %zu of %zu differ, %zu beyond %d ulp; worst %d ulp at %zu (0x%04x vs 0x%04x)%s\n", nbad ? "FAIL" : "ok", what,
                nd, n, nbad, ulps, (int)worst, wi, a[wi], b[wi], FM_STRICT ? " [strict]" : "");
    return nbad != 0;
}
#endif
