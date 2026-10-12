/* Task papyrus (2026-10-11): papyrus (foreground) vs void / air (background) inside the scroll, labels made from the CT itself.
   The CT outside the scroll is masked to 0; inside, low values are void, high values papyrus, with no exact cutoff. Per native
   patch the sampler draws a cutoff t (absolute U(t_lo, t_hi) raw CT units, or with probability rel_p relative to the patch's Otsu
   threshold, Otsu x U(rel_lo, rel_hi)) and labels the CLEAN CT smoothed by a Gaussian (sigma voxels): papyrus where s > t. Scanner
   augmentation then corrupts only the network input, so the net has to use context. Optional: soft targets sigmoid((s - t) / tau)
   with tau ~ U(tau_lo, tau_hi), an ignore band |s - t| < band, and a cleanup that flips 6-connected papyrus components and void
   holes smaller than `clean` voxels (components touching the patch faces are kept). With `cond` the network is conditioned on t, so
   at inference (`predict --papyrus-cut T`) any cutoff can be chosen per run: cond 1 adds the cutoff in input units, (t - window
   mean) / window sd (times the intensity jitter), to the stem bias through a learned vector; cond 2 z-scores the CT channel around
   the cutoff instead of the window mean, (ct - t) / window sd (no offset jitter), so the cutoff is the input's zero. Without it the
   output is P(s > t) marginalised over the cutoffs.
   Header-only (like target_erode.h): the sampler, the trainer, predict and tests/test_papyrus.c include it. */
#pragma once
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct papyrus_cfg {
    float t_lo, t_hi;            /* absolute cutoff range, raw CT units */
    float rel_lo, rel_hi, rel_p; /* with probability rel_p: t = Otsu threshold of the patch's nonzero CT x U(rel_lo, rel_hi) */
    float sigma;                 /* Gaussian smoothing of the label CT, voxels (0: the raw CT) */
    float tau_lo, tau_hi;        /* soft targets 255 sigmoid((s - t) / tau), tau ~ U(tau_lo, tau_hi) raw units; tau_hi 0: hard targets */
    float band;                  /* ignore |s - t| < band (raw units) */
    int clean;                   /* > 0: papyrus components and void holes (6-connected) below this many voxels flip */
    int cond;                    /* the network is conditioned on t: 1 through the stem bias (unet_set_cond), 2 as the input's zero */
    int local;                   /* > 0: gap labels, no cutoff (papyrus_local): radius of the closing / floor window, voxels */
    float local_sigma;           /* Gaussian smoothing of the CT the closing and floor come from, voxels */
    float local_depth;           /* a gap: closing - floor at least this deep (raw units); shallower is papyrus texture or noise */
    float local_margin;          /* in a gap, ignore |s - t| < local_margin (closing - floor) */
} papyrus_cfg;

enum { PAPYRUS_RMAX = 8 };

/* normalised Gaussian taps w[0 .. 2r], r = ceil(3 sigma) capped at PAPYRUS_RMAX; returns r (0: no smoothing, w[0] = 1) */
static inline int papyrus_taps(float sigma, float *w) {
    if (!(sigma > 0)) { w[0] = 1.f; return 0; }
    int r = (int)ceilf(3.f * sigma); if (r > PAPYRUS_RMAX) r = PAPYRUS_RMAX;
    double s = 0, e[2 * PAPYRUS_RMAX + 1];
    for (int k = -r; k <= r; k++) s += e[k + r] = exp(-0.5 * k * k / ((double)sigma * sigma));
    for (int k = 0; k <= 2 * r; k++) w[k] = (float)(e[k] / s);
    return r;
}

/* Streaming separable Gaussian smoothing, one z plane at a time with edges clamped: the planes smoothed in x and y wait in a ring of
   2r + 1, so the cube is read once and nothing larger than a few planes leaves the cache. Call papyrus_plane for z = 0, 1, .. P-1. */
typedef struct { int P, r, have; float w[2 * PAPYRUS_RMAX + 1]; float *ring, *tmp, *row; } papyrus_smoother;
static inline void papyrus_smoother_init(papyrus_smoother *s, int P, float sigma) {
    s->P = P; s->r = papyrus_taps(sigma, s->w); s->have = -1;
    const size_t P2 = (size_t)P * P;
    s->ring = malloc((2 * (size_t)s->r + 2) * P2 * sizeof(float)); s->tmp = s->ring + (2 * (size_t)s->r + 1) * P2;
    s->row = malloc(((size_t)P + 2 * PAPYRUS_RMAX) * sizeof(float));
    if (!s->ring || !s->row) abort();
}
static inline void papyrus_smoother_free(papyrus_smoother *s) { free(s->ring); free(s->row); }
/* plane p smoothed along x (through a clamped padded row) then y (row combinations) into ring slot p % (2r + 1) */
static inline void papyrus_xy(papyrus_smoother *s, const uint8_t *ct, int p) {
    const int P = s->P, r = s->r; const size_t P2 = (size_t)P * P;
    float *restrict t = s->tmp, *restrict row = s->row, *restrict d = s->ring + (size_t)(p % (2 * r + 1)) * P2;
    for (int y = 0; y < P; y++) {
        const uint8_t *src = ct + (size_t)p * P2 + (size_t)y * P; float *restrict o = t + (size_t)y * P;
        for (int x = -r; x < P + r; x++) row[x + r] = src[x < 0 ? 0 : x >= P ? P - 1 : x];
        for (int x = 0; x < P; x++) o[x] = 0.f;
        for (int k = 0; k <= 2 * r; k++) { const float wk = s->w[k]; const float *restrict q = row + k; for (int x = 0; x < P; x++) o[x] += wk * q[x]; }
    }
    for (int y = 0; y < P; y++) {
        float *restrict o = d + (size_t)y * P;
        for (int x = 0; x < P; x++) o[x] = 0.f;
        for (int k = 0; k <= 2 * r; k++) {
            int yy = y + k - r; yy = yy < 0 ? 0 : yy >= P ? P - 1 : yy;
            const float wk = s->w[k]; const float *restrict q = t + (size_t)yy * P;
            for (int x = 0; x < P; x++) o[x] += wk * q[x];
        }
    }
}
/* the smoothed plane z (P^2 floats) into out */
static inline void papyrus_plane(papyrus_smoother *s, const uint8_t *ct, int z, float *restrict out) {
    const int P = s->P, r = s->r, R = 2 * r + 1; const size_t P2 = (size_t)P * P;
    if (!r) { for (size_t j = 0; j < P2; j++) out[j] = ct[(size_t)z * P2 + j]; return; }
    const int need = z + r < P - 1 ? z + r : P - 1;
    if (s->have < z - r - 1 || s->have > need) s->have = (z - r > 0 ? z - r : 0) - 1;   /* out of order: restart the ring */
    while (s->have < need) papyrus_xy(s, ct, ++s->have);
    for (size_t j = 0; j < P2; j++) out[j] = 0.f;
    for (int k = 0; k <= 2 * r; k++) {
        int zz = z + k - r; zz = zz < 0 ? 0 : zz >= P ? P - 1 : zz;
        const float wk = s->w[k]; const float *restrict q = s->ring + (size_t)(zz % R) * P2;
        for (size_t j = 0; j < P2; j++) out[j] += wk * q[j];
    }
}
/* the whole cube smoothed into A (P^3 floats) */
static inline void papyrus_smooth(const uint8_t *ct, int P, float sigma, float *A) {
    papyrus_smoother s; papyrus_smoother_init(&s, P, sigma);
    for (int z = 0; z < P; z++) papyrus_plane(&s, ct, z, A + (size_t)z * P * P);
    papyrus_smoother_free(&s);
}

/* Running max (mx 1) or min over 2r + 1 values, the window cut at the ends (van Herk / Gil-Werman: 3 comparisons per value whatever
   r), along an axis of n elements that are rows of m contiguous bytes, element k at in + k es; out likewise (out != in). g, h:
   (n + 2r) m bytes of scratch. */
static inline void papyrus_ext(const uint8_t *in, uint8_t *out, int n, size_t es, int m, int r, int mx, uint8_t *g, uint8_t *h) {
    const int w = 2 * r + 1, L = n + 2 * r; const uint8_t pad = mx ? 0 : 255;   /* pad: neutral for the extremum */
    if (m == 1) {   /* scalar elements (the contiguous x lines) */
#define PAP_EXT(a, b) (mx ? ((a) > (b) ? (a) : (b)) : ((a) < (b) ? (a) : (b)))
        for (int k = 0; k < L; k++) { const uint8_t v = k >= r && k < n + r ? in[(size_t)(k - r) * es] : pad; g[k] = k % w == 0 ? v : PAP_EXT(g[k - 1], v); }
        for (int k = L - 1; k >= 0; k--) { const uint8_t v = k >= r && k < n + r ? in[(size_t)(k - r) * es] : pad; h[k] = k % w == w - 1 || k == L - 1 ? v : PAP_EXT(h[k + 1], v); }
        for (int i = 0; i < n; i++) out[(size_t)i * es] = PAP_EXT(h[i], g[i + 2 * r]);
#undef PAP_EXT
        return;
    }
    for (int k = 0; k < L; k++) {   /* g: running extremum from the start of each block of w */
        uint8_t *restrict gk = g + (size_t)k * m;
        const uint8_t *restrict s = k >= r && k < n + r ? in + (size_t)(k - r) * es : nullptr;
        if (k % w == 0) { if (s) memcpy(gk, s, m); else memset(gk, pad, m); continue; }
        const uint8_t *restrict gp = gk - m;
        if (!s) memcpy(gk, gp, m);
        else if (mx) for (int j = 0; j < m; j++) gk[j] = gp[j] > s[j] ? gp[j] : s[j];
        else for (int j = 0; j < m; j++) gk[j] = gp[j] < s[j] ? gp[j] : s[j];
    }
    for (int k = L - 1; k >= 0; k--) {   /* h: running extremum to the end of each block */
        uint8_t *restrict hk = h + (size_t)k * m;
        const uint8_t *restrict s = k >= r && k < n + r ? in + (size_t)(k - r) * es : nullptr;
        if (k % w == w - 1 || k == L - 1) { if (s) memcpy(hk, s, m); else memset(hk, pad, m); continue; }
        const uint8_t *restrict hn = hk + m;
        if (!s) memcpy(hk, hn, m);
        else if (mx) for (int j = 0; j < m; j++) hk[j] = hn[j] > s[j] ? hn[j] : s[j];
        else for (int j = 0; j < m; j++) hk[j] = hn[j] < s[j] ? hn[j] : s[j];
    }
    for (int i = 0; i < n; i++) {   /* the window of out[i] is padded [i, i + 2r]: h from its start, g to its end */
        const uint8_t *restrict a = h + (size_t)i * m, *restrict b = g + (size_t)(i + 2 * r) * m; uint8_t *restrict o = out + (size_t)i * es;
        if (mx) for (int j = 0; j < m; j++) o[j] = a[j] > b[j] ? a[j] : b[j];
        else for (int j = 0; j < m; j++) o[j] = a[j] < b[j] ? a[j] : b[j];
    }
}
/* Max (mx 1) or min of the P^3 cube in over (2r + 1)^3 windows cut at the faces, into out (out != in); T: P^3 scratch */
static inline void papyrus_filter(const uint8_t *in, uint8_t *out, int P, int r, int mx, uint8_t *T) {
    const size_t P2 = (size_t)P * P; const size_t L = (size_t)P + 2 * (size_t)r;
    uint8_t *g = malloc(2 * L * (size_t)P), *h = g + L * (size_t)P;
    if (!g) abort();
    for (size_t l = 0; l < P2; l++) papyrus_ext(in + l * P, T + l * P, P, 1, 1, r, mx, g, h);                  /* x: one line at a time */
    for (int z = 0; z < P; z++) papyrus_ext(T + (size_t)z * P2, out + (size_t)z * P2, P, (size_t)P, P, r, mx, g, h);   /* y: rows */
    for (int y = 0; y < P; y++) papyrus_ext(out + (size_t)y * P, T + (size_t)y * P, P, P2, P, r, mx, g, h);           /* z: rows */
    memcpy(out, T, P2 * P);
    free(g);
}
/* The closing (min of the max: dark features narrower than the window filled up to the brightness around them) and the floor (min)
   of S over (2r + 1)^3 windows, into Cl and F; T: P^3 scratch */
static inline void papyrus_levels(const uint8_t *S, int P, int r, uint8_t *Cl, uint8_t *F, uint8_t *T) {
    papyrus_filter(S, F, P, r, 1, T);    /* the dilation, in F for now */
    papyrus_filter(F, Cl, P, r, 0, T);   /* its erosion: the closing */
    papyrus_filter(S, F, P, r, 0, T);
}

/* Otsu threshold of the nonzero values (every `step`-th voxel): papyrus is v > the returned value (x.5); 0 without material */
static inline float papyrus_otsu(const uint8_t *ct, size_t n, size_t step) {
    uint64_t h[256] = {0};
    for (size_t k = 0; k < n; k += step) h[ct[k]]++;
    double tot = 0, sum = 0;
    for (int v = 1; v < 256; v++) { tot += (double)h[v]; sum += (double)v * h[v]; }
    if (tot <= 0) return 0.f;
    double w0 = 0, s0 = 0, best = -1; int bt = 1;
    for (int t = 1; t < 255; t++) {   /* class 0: values 1..t */
        w0 += (double)h[t]; s0 += (double)t * h[t];
        const double w1 = tot - w0; if (w0 <= 0 || w1 <= 0) continue;
        const double m0 = s0 / w0, m1 = (sum - s0) / w1, b = w0 * w1 * (m0 - m1) * (m0 - m1);
        if (b > best) { best = b; bt = t; }
    }
    return (float)bt + 0.5f;
}

static inline uint32_t papyrus_find(uint32_t *p, uint32_t k) { while (p[k] != k) { p[k] = p[p[k]]; k = p[k]; } return k; }
static inline void papyrus_union(uint32_t *p, uint32_t a, uint32_t b) {
    a = papyrus_find(p, a); b = papyrus_find(p, b);
    if (a < b) p[b] = a; else if (b < a) p[a] = b;
}
/* 6-connected components of the voxels with m[k] == val: those below minsize voxels that do not touch a face of the cube flip
   (m[k] = !val, flip[k] = 1). par, cnt: P^3 uint32 scratch each. Returns the number of flipped voxels. */
static inline size_t papyrus_flip_small(uint8_t *m, int P, uint8_t val, int minsize, uint32_t *par, uint32_t *cnt, uint8_t *flip) {
    const size_t P2 = (size_t)P * P, p3 = P2 * P;
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
        const size_t k = ((size_t)z * P + y) * P + x;
        if (m[k] != val) continue;
        par[k] = (uint32_t)k;
        if (x && m[k - 1] == val) papyrus_union(par, (uint32_t)k, (uint32_t)(k - 1));
        if (y && m[k - P] == val) papyrus_union(par, (uint32_t)k, (uint32_t)(k - P));
        if (z && m[k - P2] == val) papyrus_union(par, (uint32_t)k, (uint32_t)(k - P2));
    }
    memset(cnt, 0, p3 * sizeof *cnt);
    const uint32_t EDGE = 0x80000000u;
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
        const size_t k = ((size_t)z * P + y) * P + x;
        if (m[k] != val) continue;
        const uint32_t r = papyrus_find(par, (uint32_t)k);
        const int face = !x || !y || !z || x == P - 1 || y == P - 1 || z == P - 1;
        cnt[r] = ((cnt[r] & ~EDGE) + 1) | (cnt[r] & EDGE) | (face ? EDGE : 0);
    }
    size_t n = 0;
    for (size_t k = 0; k < p3; k++) {   /* (the components are found through par, so flipping m on the way is safe) */
        if (m[k] != val) continue;
        const uint32_t c = cnt[papyrus_find(par, (uint32_t)k)];
        if (!(c & EDGE) && c < (uint32_t)minsize) { m[k] = !val; flip[k] = 1; n++; }
    }
    return n;
}

/* The cutoff of one patch from three uniforms (the sampler draws them; the function itself is deterministic) */
static inline float papyrus_cutoff(const papyrus_cfg *c, const uint8_t *ct, size_t n, double u_mode, double u_t) {
    float t;
    if (c->rel_p > 0 && u_mode < c->rel_p) {
        const float o = papyrus_otsu(ct, n, 3);
        t = o * (c->rel_lo + (float)u_t * (c->rel_hi - c->rel_lo));
    } else t = c->t_lo + (float)u_t * (c->t_hi - c->t_lo);
    return t < 0.5f ? 0.5f : t > 254.5f ? 254.5f : t;
}

/* The CT smoothed by sigma, rounded into the P^3 bytes S */
static inline void papyrus_smooth_u8(const uint8_t *ct, int P, float sigma, uint8_t *S) {
    const size_t P2 = (size_t)P * P;
    papyrus_smoother sm; papyrus_smoother_init(&sm, P, sigma);
    float *plane = malloc(P2 * sizeof *plane);
    if (!plane) abort();
    for (int z = 0; z < P; z++) {
        papyrus_plane(&sm, ct, z, plane);
        uint8_t *restrict o = S + (size_t)z * P2;
        for (size_t j = 0; j < P2; j++) { const float v = plane[j] + 0.5f; o[j] = v <= 0.f ? 0 : v >= 255.f ? 255 : (uint8_t)v; }
    }
    free(plane); papyrus_smoother_free(&sm);
}

/* Gap labels (cfg local > 0; no cutoff, t and u_tau unused). The CT is not bimodal: open void is dark, but a gap narrower than the
   blur never reaches the void level (packed sheets at ~160 with gaps dipping to ~80), and papyrus can be faint (~70 beside air at
   ~20). A voxel is void when it is dark (s <= t_lo) or in the lower half of a gap: the closing Cl (of the CT smoothed by
   local_sigma, window radius `local`) fills only dark features with brighter material around them, so where Cl - F >= local_depth
   (F the window's floor) the voxel sits in a gap and its threshold is the gap's half depth t = (Cl + F) / 2 (blur puts a boundary at
   half the step), clamped to [t_lo, t_hi] (nothing above t_hi is void); a step (bright on one side only) or a faint plateau is not
   filled, so it keeps t = t_lo. s: the CT smoothed by sigma. Ignored: within local_margin (Cl - F) of t in a gap, and |s - t| < band.
   scratch: 5 P^3 bytes. */
static inline void papyrus_local(const uint8_t *ct, int P, const papyrus_cfg *c, uint8_t *target, uint8_t *ign, uint8_t *fg, uint8_t *scratch) {
    const size_t p3 = (size_t)P * P * P;
    uint8_t *S = scratch, *S2 = S + p3, *Cl = S2 + p3, *F = Cl + p3, *T = F + p3;
    papyrus_smooth_u8(ct, P, c->sigma, S);
    if (c->local_sigma != c->sigma) papyrus_smooth_u8(ct, P, c->local_sigma, S2); else S2 = S;
    papyrus_levels(S2, P, c->local, Cl, F, T);
    const float lo = c->t_lo, hi = c->t_hi, dmin = c->local_depth, mg = c->local_margin, band = c->band;
    for (size_t k = 0; k < p3; k++) {
        const float depth = (float)Cl[k] - (float)F[k], s = S[k];
        const int gap = depth >= dmin;
        float t = gap ? 0.5f * ((float)Cl[k] + (float)F[k]) : lo;
        t = t < lo ? lo : t > hi ? hi : t;
        const float d = fabsf(s - t);
        fg[k] = s > t; target[k] = ct[k] && fg[k] ? 255 : 0;
        ign[k] |= (gap && d < mg * depth) || (band > 0 && d < band);
    }
}

/* Labels of one native patch at cutoff t: target[k] = 255 sigmoid((s - t) / tau) (tau = tau_lo + u_tau (tau_hi - tau_lo); tau_hi 0:
   255 [s > t]) with s the smoothed CT; ign[k] = 1 where |s - t| < band (ign is or-ed into, not cleared); CT 0 (outside the scroll
   mask) has target 0. local > 0: papyrus_local instead (no cutoff). clean > 0: small components flip (target 0 / 255, not ignored).
   scratch (cleanup and local): 8 P^3 bytes, 4-byte aligned; fg: P^3 bytes (the papyrus mask). Returns the number of voxels the
   cleanup flipped. */
static inline size_t papyrus_labels(const uint8_t *ct, int P, const papyrus_cfg *c, float t, double u_tau,
                                    uint8_t *target, uint8_t *ign, uint8_t *fg, void *scratch) {
    const size_t P2 = (size_t)P * P, p3 = P2 * P;
    if (c->local > 0) { papyrus_local(ct, P, c, target, ign, fg, scratch); goto cleanup; }
    const float tau = c->tau_hi > 0 ? c->tau_lo + (float)u_tau * (c->tau_hi - c->tau_lo) : 0.f;
    uint8_t lut[1025];   /* sigmoid over d = (s - t) / tau in [-8, 8], 64 steps per unit */
    if (tau > 0) for (int i = 0; i <= 1024; i++) { const double d = (i - 512) / 64.0; lut[i] = (uint8_t)lrint(255.0 / (1.0 + exp(-d))); }
    const float itau = tau > 0 ? 1.f / tau : 0.f, band = c->band;
    papyrus_smoother sm; papyrus_smoother_init(&sm, P, c->sigma);
    float *plane = malloc(P2 * sizeof *plane);
    if (!plane) abort();
    for (int z = 0; z < P; z++) {   /* plane by plane: the smoothed CT never exists as a cube */
        papyrus_plane(&sm, ct, z, plane);
        const size_t z0 = (size_t)z * P2;
        const uint8_t *restrict cz = ct + z0; uint8_t *restrict fz = fg + z0, *restrict tz = target + z0, *restrict iz = ign + z0;
        for (size_t j = 0; j < P2; j++) fz[j] = plane[j] > t;
        if (band > 0) for (size_t j = 0; j < P2; j++) iz[j] |= fabsf(plane[j] - t) < band;
        if (tau > 0) for (size_t j = 0; j < P2; j++) {
            float u = (plane[j] - t) * itau * 64.f + 512.5f; u = u < 0.f ? 0.f : u > 1024.f ? 1024.f : u;
            tz[j] = cz[j] ? lut[(int)u] : 0;
        } else for (size_t j = 0; j < P2; j++) tz[j] = cz[j] && fz[j] ? 255 : 0;
    }
    free(plane); papyrus_smoother_free(&sm);
cleanup:;
    size_t flipped = 0;
    if (c->clean > 0) {   /* components through two P^3 uint32 arrays */
        uint32_t *par = scratch, *cnt = par + p3;
        uint8_t *flip = (uint8_t *)malloc(p3);
        if (!flip) abort();
        memset(flip, 0, p3);
        flipped = papyrus_flip_small(fg, P, 1, c->clean, par, cnt, flip);   /* specks of papyrus in the void */
        flipped += papyrus_flip_small(fg, P, 0, c->clean, par, cnt, flip);  /* holes of void in papyrus */
        for (size_t k = 0; k < p3; k++) if (flip[k]) { target[k] = ct[k] && fg[k] ? 255 : 0; ign[k] = 0; }
        free(flip);
    }
    return flipped;
}

/* 1 when a checkpoint's header names the papyrus task (*cond = its conditioning), 0 when not, -1 when unreadable */
static inline int papyrus_checkpoint(const char *path, int *cond) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char *h = malloc(1 << 16);
    if (!h) { fclose(f); return -1; }
    const size_t n = fread(h, 1, (1 << 16) - 1, f); fclose(f); h[n] = 0;
    char *e = memchr(h, '\n', n); if (e) *e = 0;
    const char *p = strstr(h, "\"papyrus\":{");
    int rc = strstr(h, "\"task\":\"papyrus\"") && p ? 1 : 0;
    if (rc && cond) { const char *q = strstr(p, "\"cond\":"); *cond = q ? atoi(q + 7) : 0; }
    free(h);
    return rc;
}
