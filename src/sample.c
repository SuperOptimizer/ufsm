#include "sample.h"
#include "nn.h"
#include <math.h>
#include <stddef.h>
#include <pthread.h>
#include <unistd.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- rng (xoshiro256**) ---- */
typedef struct { uint64_t s[4]; } rng;
static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static uint64_t rnext(rng *r) {
    uint64_t *s = r->s, res = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return res;
}
static void rseed(rng *r, uint64_t seed) {
    for (int i = 0; i < 4; i++) { seed += 0x9e3779b97f4a7c15ull; uint64_t z = seed; z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; r->s[i] = z ^ (z >> 31); }
}
static double runif(rng *r) { return (double)(rnext(r) >> 11) * 0x1.0p-53; }
static int64_t rint_below(rng *r, int64_t n) { return n <= 1 ? 0 : (int64_t)(runif(r) * (double)n); }
static double rnorm(rng *r) { double u = runif(r) + 1e-300, v = runif(r); return sqrt(-2 * log(u)) * cos(6.283185307179586 * v); }

sample_cfg sample_cfg_default(void) {
    sample_cfg c = {0};
    c.P = 128; c.B = 2; c.nworkers = 8; c.nbuf = 6; c.seed = 0;
    c.level_p[0] = 0.5; c.level_p[1] = 0.25; c.level_p[2] = 0.15; c.level_p[3] = 0.1;
    c.min_fg = 0.3; c.empty_keep = 0.2; c.augment = 1;
    return c;
}

void pool2_u8(const uint8_t *in, int n, uint8_t *out) {
    int h = n / 2;
    for (int z = 0; z < h; z++)
        for (int y = 0; y < h; y++)
            for (int x = 0; x < h; x++) {
                const uint8_t *p = in + ((size_t)(2 * z) * n + 2 * y) * n + 2 * x;
                unsigned s = p[0] + p[1] + p[n] + p[n + 1];
                p += (size_t)n * n;
                s += p[0] + p[1] + p[n] + p[n + 1];
                out[((size_t)z * h + y) * h + x] = (uint8_t)((s + 4) >> 3);
            }
}

/* label pooling: 0..254 = surface fraction, 255 = ignore (ignore when at least half the children are) */
static void pool2_labels(const uint8_t *in, int n, uint8_t *out) {
    int h = n / 2;
    for (int z = 0; z < h; z++) for (int y = 0; y < h; y++) for (int x = 0; x < h; x++) {
        const uint8_t *p = in + ((size_t)(2 * z) * n + 2 * y) * n + 2 * x;
        const uint8_t v[8] = {p[0], p[1], p[n], p[n + 1], p[(size_t)n * n], p[(size_t)n * n + 1], p[(size_t)n * n + n], p[(size_t)n * n + n + 1]};
        int nig = 0, sum = 0, cnt = 0;
        for (int i = 0; i < 8; i++) { if (v[i] == 255) nig++; else { sum += v[i]; cnt++; } }
        out[((size_t)z * h + y) * h + x] = nig >= 4 ? 255 : (uint8_t)((sum + cnt / 2) / cnt);
    }
}

/* ---- cube symmetries: out[z,y,x] = in[q] with q[perm[d]] = (flip[d] ? P-1-i_d : i_d) ---- */
typedef struct { int perm[3]; int flip[3]; } sym;

static sym sym_of(int s) { /* s in [0, 48) */
    static const int perms[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    sym y;
    memcpy(y.perm, perms[s / 8], sizeof y.perm);
    for (int d = 0; d < 3; d++) y.flip[d] = (s >> d) & 1;
    return y;
}
/* proper rotation (det +1)? parity of the permutation times the number of flips must be even: reflections change the
   handedness that distinguishes the recto from the verso face of a winding */
static int sym_is_rotation(sym y) {
    static const int odd[6] = {0, 1, 1, 0, 0, 1};   /* permutation parity in perms[] order */
    int p = (y.perm[0] == 0 && y.perm[1] == 1) ? 0 : (y.perm[0] == 0) ? 1 : (y.perm[0] == 1 && y.perm[1] == 0) ? 2 : (y.perm[0] == 1) ? 3 : (y.perm[0] == 2 && y.perm[1] == 0) ? 4 : 5;
    return ((odd[p] + y.flip[0] + y.flip[1] + y.flip[2]) & 1) == 0;
}

#define APPLY_SYM(T, name)                                                                        \
    static void name(const T *in, T *out, int P, sym y) {                                         \
        int64_t stride[3] = {(int64_t)P * P, P, 1};                                                \
        for (int z = 0; z < P; z++)                                                               \
            for (int yy = 0; yy < P; yy++)                                                        \
                for (int x = 0; x < P; x++) {                                                     \
                    int i[3] = {z, yy, x};                                                        \
                    int64_t q = 0;                                                                \
                    for (int d = 0; d < 3; d++) q += stride[y.perm[d]] * (y.flip[d] ? P - 1 - i[d] : i[d]); \
                    out[((int64_t)z * P + yy) * P + x] = in[q];                                   \
                }                                                                                 \
    }
APPLY_SYM(float, sym_f32)
APPLY_SYM(uint8_t, sym_u8)

/* ---- sampler ---- */

typedef enum { FREE, FILLING, READY, TAKEN } slot_state;

struct sampler {
    sources *S;
    sample_cfg cfg;
    batch *slots;
    slot_state *state;
    int nslots;
    pthread_mutex_t mu;
    pthread_cond_t cv_free, cv_ready;
    pthread_t *th;
    atomic_int stop;
    atomic_uint_fast64_t produced, rejected;
    double *cum;   /* cumulative source weights */
    struct { uint32_t *idx; size_t n; int lev; int64_t shape[3]; } *occ;   /* per source: coarse label cells containing papyrus (guides the position draw) */
};

static size_t P3(const sampler *sp) { return (size_t)sp->cfg.P * sp->cfg.P * sp->cfg.P; }

static batch alloc_batch(const sample_cfg *c) {
    size_t p3 = (size_t)c->P * c->P * c->P;
    batch b;
    b.x = nn_host_alloc((size_t)c->B * 4 * p3 * sizeof(float));   /* pinned: the trainer uploads asynchronously */
    b.t = nn_host_alloc((size_t)c->B * NCH * p3);
    b.m = nn_host_alloc((size_t)c->B * p3);
    b.w = nn_host_alloc((size_t)c->B * NCH);
    b.src = malloc((size_t)c->B * sizeof(int16_t));
    b.level = malloc((size_t)c->B * sizeof(int8_t));
    b.corner = malloc((size_t)c->B * sizeof *b.corner);
    return b;
}

static void free_batch(batch *b) { nn_host_free(b->x); nn_host_free(b->t); nn_host_free(b->m); nn_host_free(b->w); free(b->src); free(b->level); free(b->corner); }

/* Level choice for a source: restrict cfg.level_p to levels the CT has and every target of the source
   can provide (pyramid: same level; regions: levels 0..1). Returns -1 if nothing is usable. */
static int pick_level(sampler *sp, source *s, rng *r, int region_ch) {
    double p[MAXLEV], tot = 0;
    for (int l = 0; l < MAXLEV; l++) {
        p[l] = sp->cfg.level_p[l];
        if (p[l] <= 0) continue;
        if (region_ch >= 0 && l > 1) { p[l] = 0; continue; }
        if (!source_ct(s, l)) { p[l] = 0; continue; }
        int any = 0;
        for (int c = 0; c < NCH; c++) {
            if (c == region_ch) { any = 1; continue; }
            if (s->tgt_key[c] && source_tgt(s, c, l)) any = 1;
        }
        if (!any) p[l] = 0;
        tot += p[l];
    }
    if (tot <= 0) return -1;
    double u = runif(r) * tot;
    for (int l = 0; l < MAXLEV; l++) { u -= p[l]; if (p[l] > 0 && u <= 0) return l; }
    return -1;
}

/* Fill patch i of batch b. Returns 0 on success, 1 if rejected (try again), -1 on I/O error. */
static int draw(sampler *sp, batch *b, int i, rng *r, float *xtmp, uint8_t *ttmp, uint8_t *big) {
    const sample_cfg *c = &sp->cfg;
    const int P = c->P;
    const size_t p3 = P3(sp);
    /* source by weight */
    double u = runif(r) * sp->cum[sp->S->n - 1];
    int si = 0;
    while (si < sp->S->n - 1 && sp->cum[si] < u) si++;
    source *s = &sp->S->src[si];
    if (c->holdout && !s->hold_n[0]) return 1;
    /* regions target? pick one region, then a level in {0,1} */
    int region_ch = -1, ri = -1;
    for (int ch = 0; ch < NCH; ch++) if (s->reg[ch] && s->reg[ch]->n) region_ch = ch;
    int l = pick_level(sp, s, r, region_ch);
    if (l < 0) return 1;
    z3 *ct = source_ct(s, l);
    const z3_meta *m = z3_meta_of(ct);
    int64_t o[3], n[3] = {P, P, P};
    if (region_ch >= 0) {
        regions *R = s->reg[region_ch];
        ri = (int)rint_below(r, R->n);
        int64_t span = (int64_t)P << l;              /* level-0 extent of the patch */
        int rs = R->rsize ? R->rsize[ri] : R->size;
        if (span > rs) return 1;
        for (int d = 0; d < 3; d++) o[d] = (R->origin[ri][d] + rint_below(r, rs - span + 1)) >> l;
    } else if (c->holdout) {
        int64_t span = (int64_t)P << l;
        for (int d = 0; d < 3; d++) { if (s->hold_n[d] < span) return 1; o[d] = (s->hold_o[d] + rint_below(r, s->hold_n[d] - span + 1)) >> l; }
    } else if (sp->occ[si].n) {   /* draw around a random coarse cell that contains papyrus */
        size_t k = (size_t)(runif(r) * (double)sp->occ[si].n); if (k >= sp->occ[si].n) k = sp->occ[si].n - 1;
        uint32_t id = sp->occ[si].idx[k];
        int64_t cz = id / (sp->occ[si].shape[1] * sp->occ[si].shape[2]), cy = (id / sp->occ[si].shape[2]) % sp->occ[si].shape[1], cx = id % sp->occ[si].shape[2];
        int64_t cc[3] = {cz, cy, cx};
        int dl = sp->occ[si].lev - l;    /* coarse cell -> level-l voxels */
        for (int d = 0; d < 3; d++) {
            if (m->shape[d] < P) return 1;
            int64_t f = dl >= 0 ? (int64_t)1 << dl : 1;
            int64_t v = dl >= 0 ? cc[d] * f + rint_below(r, f) : cc[d] >> (-dl);
            o[d] = v - P / 2; if (o[d] < 0) o[d] = 0; if (o[d] > m->shape[d] - P) o[d] = m->shape[d] - P;
        }
    } else {
        for (int d = 0; d < 3; d++) { if (m->shape[d] < P) return 1; o[d] = rint_below(r, m->shape[d] - P + 1); }
    }
    if (!c->holdout && s->hold_n[0]) {   /* training: reject patches touching the held-out box */
        int64_t span = (int64_t)P << l, hit = 1;
        for (int d = 0; d < 3; d++) { int64_t a = o[d] << l; if (a + span <= s->hold_o[d] || a >= s->hold_o[d] + s->hold_n[d]) hit = 0; }
        if (hit) return 1;
    }
    /* cheap occupancy test on a coarse level before fetching the fine cube */
    int lc = l + 3;
    z3 *cct = lc < MAXLEV ? source_ct(s, lc) : nullptr;
    if (cct) {
        int pc = P >> 3;
        int64_t oc[3] = {o[0] >> 3, o[1] >> 3, o[2] >> 3}, nc[3] = {pc, pc, pc};
        if (z3_read(cct, oc, nc, big, 1)) return -1;
        size_t nz = 0, tot = (size_t)pc * pc * pc;
        for (size_t k = 0; k < tot; k++) nz += big[k] != 0;
        if ((double)nz / (double)tot < c->min_fg) { atomic_fetch_add(&sp->rejected, 1); return 1; }
    }
    /* CT */
    uint8_t *ctu = big;
    if (z3_read(ct, o, n, ctu, 1)) return -1;
    size_t nz = 0;
    double sum = 0, sq = 0;
    for (size_t k = 0; k < p3; k++) { nz += ctu[k] != 0; sum += ctu[k]; sq += (double)ctu[k] * ctu[k]; }
    if ((double)nz / (double)p3 < c->min_fg) { atomic_fetch_add(&sp->rejected, 1); return 1; }
    /* targets */
    uint8_t w[NCH] = {0};
    int tmax = 0;
    for (int ch = 0; ch < NCH; ch++) {
        uint8_t *dst = ttmp + (size_t)ch * p3;
        if (ch == region_ch) {
            z3 *rz = source_region(s, ch, ri);
            if (!rz) return -1;
            regions *R = s->reg[ch];
            int64_t span = (int64_t)P << l;
            int64_t ro[3], rn[3] = {span, span, span};
            int shared = source_region_shared(s, ch);
            for (int d = 0; d < 3; d++) ro[d] = (o[d] << l) - (shared ? 0 : R->origin[ri][d]);
            uint8_t *tmp = big + p3;   /* big has room for a (2P)^3 cube after the CT */
            if (l == 0) { if (z3_read(rz, ro, rn, dst, 1)) return -1; }
            else { if (z3_read(rz, ro, rn, tmp, 1)) return -1; pool2_labels(tmp, (int)span, dst); }
            w[ch] = 1;
        } else if (s->tgt_key[ch]) {
            z3 *tz = source_tgt(s, ch, l);
            if (!tz) { memset(dst, 0, p3); continue; }
            if (z3_read(tz, o, n, dst, 1)) return -1;
            w[ch] = 1;
        } else memset(dst, 0, p3);
        if (w[ch]) for (size_t k = 0; k < p3; k++) if (dst[k] != 255 && dst[k] > tmax) tmax = dst[k];
    }
    if (tmax == 0 && runif(r) > c->empty_keep) { atomic_fetch_add(&sp->rejected, 1); return 1; }
    /* curriculum: thicken the surface band by max-filtering the target (hard labels only, pyramid sources) */
    if (c->dilate > 0 && region_ch < 0) {
        int D = c->dilate >> l; if (D < 1) D = 1;
        uint8_t *tmp2 = big + 2 * p3;
        for (int ch = 0; ch < NCH; ch++) {
            if (!w[ch]) continue;
            uint8_t *dst = ttmp + (size_t)ch * p3;
            for (int pass = 0; pass < 3; pass++) {
                size_t str = pass == 0 ? 1 : pass == 1 ? (size_t)P : (size_t)P * P;
                for (size_t k = 0; k < p3; k++) {
                    int idx = pass == 0 ? (int)(k % P) : pass == 1 ? (int)((k / P) % P) : (int)(k / ((size_t)P * P));
                    uint8_t v = dst[k];
                    if (v != 255) { int a = idx - D < 0 ? -idx : -D, bnd = idx + D >= P ? P - 1 - idx : D; for (int d = a; d <= bnd; d++) { uint8_t u = dst[k + (ptrdiff_t)d * (ptrdiff_t)str]; if (u != 255 && u > v) v = u; } }
                    tmp2[k] = v;
                }
                memcpy(dst, tmp2, p3);
            }
        }
    }
    /* soft ridge target: background voxels near the surface get 254 * exp(-(d / sigma)^2 / 2), d = 3-4-5 chamfer distance / 3 */
    if (c->soft > 0 && region_ch < 0) {
        float sigma = c->soft / (float)(1 << l); if (sigma < 0.75f) sigma = 0.75f;
        uint16_t *dm = (uint16_t *)(big + 2 * p3);   /* distance map (needs 2 P^3 bytes: big has room) */
        const uint16_t INF = 60000;
        for (int ch = 0; ch < NCH; ch++) {
            if (!w[ch]) continue;
            uint8_t *dst = ttmp + (size_t)ch * p3;
            for (size_t k = 0; k < p3; k++) dm[k] = (dst[k] != 255 && dst[k] > 0) ? 0 : INF;
            /* two-pass chamfer 3-4-5 */
            for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
                size_t k = ((size_t)z * P + y) * P + x; uint16_t v = dm[k]; if (!v) continue;
                if (x) v = dm[k - 1] + 3 < v ? dm[k - 1] + 3 : v;
                if (y) { v = dm[k - P] + 3 < v ? dm[k - P] + 3 : v; if (x) v = dm[k - P - 1] + 4 < v ? dm[k - P - 1] + 4 : v; if (x + 1 < P) v = dm[k - P + 1] + 4 < v ? dm[k - P + 1] + 4 : v; }
                if (z) { size_t kz = k - (size_t)P * P; v = dm[kz] + 3 < v ? dm[kz] + 3 : v; if (x) v = dm[kz - 1] + 4 < v ? dm[kz - 1] + 4 : v; if (x + 1 < P) v = dm[kz + 1] + 4 < v ? dm[kz + 1] + 4 : v; if (y) v = dm[kz - P] + 4 < v ? dm[kz - P] + 4 : v; if (y + 1 < P) v = dm[kz + P] + 4 < v ? dm[kz + P] + 4 : v; }
                dm[k] = v;
            }
            for (int z = P - 1; z >= 0; z--) for (int y = P - 1; y >= 0; y--) for (int x = P - 1; x >= 0; x--) {
                size_t k = ((size_t)z * P + y) * P + x; uint16_t v = dm[k]; if (!v) continue;
                if (x + 1 < P) v = dm[k + 1] + 3 < v ? dm[k + 1] + 3 : v;
                if (y + 1 < P) { v = dm[k + P] + 3 < v ? dm[k + P] + 3 : v; if (x) v = dm[k + P - 1] + 4 < v ? dm[k + P - 1] + 4 : v; if (x + 1 < P) v = dm[k + P + 1] + 4 < v ? dm[k + P + 1] + 4 : v; }
                if (z + 1 < P) { size_t kz = k + (size_t)P * P; v = dm[kz] + 3 < v ? dm[kz] + 3 : v; if (x) v = dm[kz - 1] + 4 < v ? dm[kz - 1] + 4 : v; if (x + 1 < P) v = dm[kz + 1] + 4 < v ? dm[kz + 1] + 4 : v; if (y) v = dm[kz - P] + 4 < v ? dm[kz - P] + 4 : v; if (y + 1 < P) v = dm[kz + P] + 4 < v ? dm[kz + P] + 4 : v; }
                dm[k] = v;
            }
            for (size_t k = 0; k < p3; k++) if (dst[k] == 0 && dm[k] < INF) { float d = dm[k] / 3.f; int t = (int)(254.f * expf(-0.5f * d * d / (sigma * sigma)) + 0.5f); if (t > 0) dst[k] = (uint8_t)t; }
        }
    }
    /* partial annotations: trust background only within R voxels of an annotated surface, ignore the rest */
    if (s->trust_band > 0) {
        int R = s->trust_band >> l; if (R < 1) R = 1;
        uint8_t *near = big + p3, *tmp2 = big + 2 * p3;   /* scratch cubes (big holds room for 10 P^3) */
        for (int ch = 0; ch < NCH; ch++) {
            if (!w[ch]) continue;
            uint8_t *dst = ttmp + (size_t)ch * p3;
            for (size_t k = 0; k < p3; k++) near[k] = dst[k] != 255 && dst[k] > 0;
            /* separable max filter of radius R along x, y, z */
            for (int pass = 0; pass < 3; pass++) {
                size_t str = pass == 0 ? 1 : pass == 1 ? (size_t)P : (size_t)P * P;
                for (size_t k = 0; k < p3; k++) {
                    int idx = pass == 0 ? (int)(k % P) : pass == 1 ? (int)((k / P) % P) : (int)(k / ((size_t)P * P));
                    uint8_t v = 0;
                    int a = idx - R < 0 ? -idx : -R, bnd = idx + R >= P ? P - 1 - idx : R;
                    for (int d = a; d <= bnd && !v; d++) v = near[k + (ptrdiff_t)d * (ptrdiff_t)str];
                    tmp2[k] = v;
                }
                memcpy(near, tmp2, p3);
            }
            for (size_t k = 0; k < p3; k++) if (dst[k] == 0 && !near[k]) dst[k] = 255;
        }
    }
    /* label encoding -> target probability * 255 and the ignore mask (255 = ignore) */
    uint8_t *ign = big + 9 * p3;
    memset(ign, 0, p3);
    for (int ch = 0; ch < NCH; ch++) {
        if (!w[ch]) continue;
        uint8_t *dst = ttmp + (size_t)ch * p3;
        for (size_t k = 0; k < p3; k++) { uint8_t v = dst[k]; if (v == 255) { ign[k] = 1; dst[k] = 0; } else dst[k] = (uint8_t)((v * 255 + 127) / 254); }
    }
    /* z-score + radial into xtmp (4 channels) */
    double mean = sum / (double)p3, var = sq / (double)p3 - mean * mean, sd = sqrt(var > 0 ? var : 0) + 1e-3;
    float *xc = xtmp, *rz_ = xtmp + p3, *ry = xtmp + 2 * p3, *rx = xtmp + 3 * p3;
    double scale = (double)(1 << l);
    for (int z = 0; z < P; z++) {
        double cy, cx;
        axis_at(&s->ax, (double)(o[0] + z) * scale, &cy, &cx);
        cy /= scale; cx /= scale;
        for (int y = 0; y < P; y++) {
            double dy = (double)(o[1] + y) - cy;
            for (int x = 0; x < P; x++) {
                size_t k = ((size_t)z * P + y) * P + x;
                double dx = (double)(o[2] + x) - cx, nn = sqrt(dy * dy + dx * dx) + 1e-6;
                xc[k] = (float)((ctu[k] - mean) / sd);
                rz_[k] = 0;
                ry[k] = s->ax.n ? (float)(dy / nn) : 0;
                rx[k] = s->ax.n ? (float)(dx / nn) : 0;
            }
        }
    }
    uint8_t *mask = big + 9 * p3;   /* ign lives here: fold CT > 0 into it */
    for (size_t k = 0; k < p3; k++) mask[k] = ctu[k] != 0 && !mask[k];
    /* augment: symmetry (permute/flip cube, rotate the radial vector) + intensity */
    float *X = b->x + (size_t)i * 4 * p3;
    uint8_t *T = b->t + (size_t)i * NCH * p3, *M = b->m + (size_t)i * p3;
    if (c->augment) {
        sym y = sym_of((int)rint_below(r, 48));
        if (c->augment == 4) y = sym_of(0);   /* intensity jitter only */
        if (c->augment == 2) while (!sym_is_rotation(y)) y = sym_of((int)rint_below(r, 48));   /* rotations only */
        if (c->augment == 3) { y = sym_of((int)rint_below(r, 48)); while (y.perm[0] != 0 || y.flip[0]) y = sym_of((int)rint_below(r, 48)); }   /* z fixed: y/x swaps and flips only */
        sym_f32(xc, X, P, y);
        /* vector channels: output axis d takes input axis perm[d], negated when flipped */
        for (int d = 0; d < 3; d++) {
            sym_f32(xtmp + (size_t)(1 + y.perm[d]) * p3, X + (size_t)(1 + d) * p3, P, y);
            if (y.flip[d]) for (size_t k = 0; k < p3; k++) X[(size_t)(1 + d) * p3 + k] = -X[(size_t)(1 + d) * p3 + k];
        }
        for (int ch = 0; ch < NCH; ch++) sym_u8(ttmp + (size_t)ch * p3, T + (size_t)ch * p3, P, y);
        sym_u8(mask, M, P, y);
        double a = exp((runif(r) * 2 - 1) * 0.22), bb = (runif(r) * 2 - 1) * 0.2, sg = runif(r) * 0.1;
        for (size_t k = 0; k < p3; k++) X[k] = (float)(a * X[k] + bb + sg * rnorm(r));
    } else {
        memcpy(X, xtmp, 4 * p3 * sizeof(float));
        memcpy(T, ttmp, NCH * p3);
        memcpy(M, mask, p3);
    }
    memcpy(b->w + (size_t)i * NCH, w, NCH);
    b->src[i] = (int16_t)si;
    b->level[i] = (int8_t)l;
    memcpy(b->corner[i], o, sizeof o);
    for (int d = 0; d < 3; d++) b->corner[i][d] <<= l;
    atomic_fetch_add(&sp->produced, 1);
    return 0;
}

typedef struct { sampler *sp; int id; } warg;

static void *worker(void *arg) {
    warg *wa = arg;
    sampler *sp = wa->sp;
    rng r;
    rseed(&r, sp->cfg.seed * 1000003ull + (uint64_t)wa->id);
    size_t p3 = P3(sp);
    float *xtmp = malloc(4 * p3 * sizeof(float));
    uint8_t *ttmp = malloc(NCH * p3);
    uint8_t *big = malloc(10 * p3);   /* [0,P^3) CT (also the coarse probe), [P^3, 9P^3) (2P)^3 region scratch, [9P^3, 10P^3) mask */
    while (!atomic_load(&sp->stop)) {
        pthread_mutex_lock(&sp->mu);
        int k = -1;
        while (k < 0 && !atomic_load(&sp->stop)) {
            for (int j = 0; j < sp->nslots; j++) if (sp->state[j] == FREE) { k = j; break; }
            if (k < 0) pthread_cond_wait(&sp->cv_free, &sp->mu);
        }
        if (k >= 0) sp->state[k] = FILLING;
        pthread_mutex_unlock(&sp->mu);
        if (k < 0) break;
        batch *b = &sp->slots[k];
        int fail = 0; unsigned spin = 0;
        for (int i = 0; i < sp->cfg.B && !atomic_load(&sp->stop);) {
            int rc = draw(sp, b, i, &r, xtmp, ttmp, big);
            if (rc == 0) { i++; fail = 0; spin = 0; }
            else if (rc > 0 && ++spin == (1u << 22)) fprintf(stderr, "sampler: %u consecutive draws rejected or impossible (P=%d too large for the sources' regions, holdout boxes or levels?)\n", spin, sp->cfg.P);
            else if (rc < 0) { fprintf(stderr, "sampler: %s\n", z3_error()); if (++fail > 20) { fprintf(stderr, "sampler: 20 consecutive read failures, stopping\n"); atomic_store(&sp->stop, 1); } }
        }
        pthread_mutex_lock(&sp->mu);
        sp->state[k] = READY;
        pthread_cond_broadcast(&sp->cv_ready);
        pthread_mutex_unlock(&sp->mu);
    }
    free(xtmp); free(ttmp); free(big);
    free(wa);
    return nullptr;
}

/* coarsest recto level that fits in memory: the cells containing any surface fraction (fallback: any labelled cell) */
occ_index source_occupancy(source *s) {
    occ_index o = {nullptr, 0, -1, {0, 0, 0}};
    if (s->reg[0] && s->reg[0]->n) return o;
    int lc = -1; z3 *tz = nullptr;
    for (int l = MAXLEV - 1; l >= 3; l--) { z3 *t = source_tgt(s, 0, l); if (t) { const z3_meta *m = z3_meta_of(t); if ((double)m->shape[0] * m->shape[1] * m->shape[2] <= 3e8) { lc = l; tz = t; break; } } }
    if (!tz) return o;
    const z3_meta *m = z3_meta_of(tz);
    size_t tot = (size_t)m->shape[0] * m->shape[1] * m->shape[2];
    uint8_t *buf = malloc(tot);
    int64_t o0[3] = {0, 0, 0};
    if (!buf || z3_read(tz, o0, m->shape, buf, 8)) { free(buf); return o; }
    size_t n = 0; for (size_t k = 0; k < tot; k++) n += buf[k] != 255 && buf[k] > 0;
    int surf = n > 0; if (!n) for (size_t k = 0; k < tot; k++) n += buf[k] != 255;
    if (!n) { free(buf); return o; }
    uint32_t *idx = malloc(n * sizeof *idx); size_t j = 0;
    for (size_t k = 0; k < tot; k++) if (buf[k] != 255 && (surf ? buf[k] > 0 : 1)) idx[j++] = (uint32_t)k;
    free(buf);
    o.idx = idx; o.n = n; o.lev = lc; for (int d = 0; d < 3; d++) o.shape[d] = m->shape[d];
    fprintf(stderr, "sampler: %s: %zu of %zu level-%d cells %s\n", s->name, n, tot, lc, surf ? "contain surface" : "are labelled");
    return o;
}
/* Pull the CT of every occupied cell at levels 0..maxlev through the chunk cache (training then reads from local disk).
   Cells are visited in a shuffled order so a partial prefetch is still useful; one thread per in-flight region. */
typedef struct { source *s; occ_index *o; int level; size_t from, to; size_t *done, *fail; } pf_arg;
static void *pf_worker(void *p) {
    pf_arg *a = p; z3 *ct = source_ct(a->s, a->level); if (!ct) return nullptr;
    const z3_meta *m = z3_meta_of(ct);
    int dl = a->o->lev - a->level; int64_t f = dl >= 0 ? (int64_t)1 << dl : 1;
    size_t cap = (size_t)f * f * f; uint8_t *buf = malloc(cap);
    for (size_t k = a->from; k < a->to; k++) {
        uint32_t id = a->o->idx[(k * 2654435761u) % a->o->n];
        int64_t cz = id / (a->o->shape[1] * a->o->shape[2]), cy = (id / a->o->shape[2]) % a->o->shape[1], cx = id % a->o->shape[2];
        int64_t cc[3] = {cz, cy, cx}, org[3], n[3];
        for (int d = 0; d < 3; d++) { org[d] = cc[d] * f; n[d] = f; if (org[d] + n[d] > m->shape[d]) n[d] = m->shape[d] - org[d]; if (n[d] <= 0) n[d] = 0; }
        if (n[0] > 0 && n[1] > 0 && n[2] > 0 && z3_read(ct, org, n, buf, 1)) __atomic_fetch_add(a->fail, 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(a->done, 1, __ATOMIC_RELAXED);
    }
    free(buf); return nullptr;
}
int sources_prefetch(sources *S, int maxlev, int nthreads, double fraction) {
    for (int i = 0; i < S->n; i++) {
        source *s = &S->src[i];
        occ_index o = source_occupancy(s);
        if (!o.n) { fprintf(stderr, "prefetch: %s: no occupancy index (regions source?), skipped\n", s->name); continue; }
        for (int l = 0; l <= maxlev; l++) {
            if (!source_ct(s, l)) continue;
            size_t ncell = (size_t)(o.n * fraction); if (ncell < 1) ncell = 1; if (ncell > o.n) ncell = o.n;
            size_t done = 0, fail = 0; pthread_t *th = malloc(nthreads * sizeof *th); pf_arg *args = malloc(nthreads * sizeof *args);
            for (int t = 0; t < nthreads; t++) { args[t] = (pf_arg){s, &o, l, ncell * t / nthreads, ncell * (t + 1) / nthreads, &done, &fail}; pthread_create(&th[t], nullptr, pf_worker, &args[t]); }
            size_t last = 0;
            while (last < ncell) { usleep(2000000); last = __atomic_load_n(&done, __ATOMIC_RELAXED); fprintf(stderr, "prefetch: %s level %d: %zu / %zu cells\r", s->name, l, last, ncell); }
            for (int t = 0; t < nthreads; t++) pthread_join(th[t], nullptr);
            free(th); free(args);
            fprintf(stderr, "\nprefetch: %s level %d done (%zu cells, %zu read failures)\n", s->name, l, ncell, fail);
        }
        free(o.idx);
    }
    return 0;
}
sampler *sampler_start(sources *S, const sample_cfg *cfg) {
    sampler *sp = calloc(1, sizeof *sp);
    sp->S = S;
    sp->cfg = *cfg;
    sp->nslots = cfg->nbuf;
    sp->slots = calloc((size_t)sp->nslots, sizeof *sp->slots);
    sp->state = calloc((size_t)sp->nslots, sizeof *sp->state);
    for (int i = 0; i < sp->nslots; i++) sp->slots[i] = alloc_batch(cfg);
    pthread_mutex_init(&sp->mu, nullptr);
    pthread_cond_init(&sp->cv_free, nullptr);
    pthread_cond_init(&sp->cv_ready, nullptr);
    sp->occ = calloc((size_t)S->n, sizeof *sp->occ);
    for (int i = 0; i < S->n; i++) {
        occ_index o = source_occupancy(&S->src[i]);
        sp->occ[i].idx = o.idx; sp->occ[i].n = o.n; sp->occ[i].lev = o.lev; for (int d = 0; d < 3; d++) sp->occ[i].shape[d] = o.shape[d];
    }
    sp->cum = malloc((size_t)S->n * sizeof(double));
    double acc = 0;
    for (int i = 0; i < S->n; i++) { acc += S->src[i].weight; sp->cum[i] = acc; }
    sp->th = calloc((size_t)cfg->nworkers, sizeof(pthread_t));
    for (int i = 0; i < cfg->nworkers; i++) {
        warg *wa = malloc(sizeof *wa);
        wa->sp = sp; wa->id = i;
        pthread_create(&sp->th[i], nullptr, worker, wa);
    }
    return sp;
}

batch *sampler_next(sampler *sp) {
    pthread_mutex_lock(&sp->mu);
    int k = -1;
    while (k < 0) {
        for (int j = 0; j < sp->nslots; j++) if (sp->state[j] == READY) { k = j; break; }
        if (k < 0) {
            if (atomic_load(&sp->stop)) { pthread_mutex_unlock(&sp->mu); return nullptr; }
            pthread_cond_wait(&sp->cv_ready, &sp->mu);
        }
    }
    sp->state[k] = TAKEN;
    pthread_mutex_unlock(&sp->mu);
    return &sp->slots[k];
}

void sampler_release(sampler *sp, batch *b) {
    pthread_mutex_lock(&sp->mu);
    sp->state[b - sp->slots] = FREE;
    pthread_cond_signal(&sp->cv_free);
    pthread_mutex_unlock(&sp->mu);
}

void sampler_stop(sampler *sp) {
    atomic_store(&sp->stop, 1);
    pthread_mutex_lock(&sp->mu);
    pthread_cond_broadcast(&sp->cv_free);
    pthread_cond_broadcast(&sp->cv_ready);
    pthread_mutex_unlock(&sp->mu);
    for (int i = 0; i < sp->cfg.nworkers; i++) pthread_join(sp->th[i], nullptr);
    for (int i = 0; i < sp->nslots; i++) free_batch(&sp->slots[i]);
    for (int i = 0; i < sp->S->n; i++) free(sp->occ[i].idx);
    free(sp->occ); free(sp->slots); free(sp->state); free(sp->th); free(sp->cum);
    free(sp);
}

void sampler_stats(const sampler *sp, uint64_t *produced, uint64_t *rejected) {
    *produced = atomic_load(&sp->produced);
    *rejected = atomic_load(&sp->rejected);
}
