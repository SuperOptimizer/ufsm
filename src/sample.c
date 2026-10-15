#include "sample.h"
#include "band.h"
#include "target_erode.h"
#include "papyrus.h"
#include "nn.h"
#include "ct_augment.h"
#include "spatial_augment.h"
#include "edt.h"
#include <math.h>
#include <stddef.h>
#include <pthread.h>
#include <unistd.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __F16C__
#include <immintrin.h>
#endif

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

sample_cfg sample_cfg_default(void) {
    sample_cfg c = {0};
    c.P = 128; c.B = 2; c.nworkers = 8; c.nbuf = 6; c.seed = 0;
    c.level_p[0] = 0.5; c.level_p[1] = 0.25; c.level_p[2] = 0.15; c.level_p[3] = 0.1;
    c.min_fg = 0.3; c.empty_keep = 0.2; c.augment = 1; c.snap = 1; c.symmetry_p = 1;
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

/* ---- cube symmetries: out[z,y,x] = in[q] with q[perm[d]] = (flip[d] ? P-1-i_d : i_d); slabs (D x P x P, D != P) only
   take the symmetries that keep z (perm[0] == 0) ---- */
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
    static void name(const T *in, T *out, int D, int P, sym y) {                                  \
        if (y.perm[0] == 0 && y.perm[1] == 1 && !y.flip[0] && !y.flip[1] && !y.flip[2]) { memcpy(out, in, (size_t)D * P * P * sizeof(T)); return; } \
        if (D != P && y.perm[0] != 0) abort();                                                    \
        int64_t stride[3] = {(int64_t)P * P, P, 1}; const int n[3] = {D, P, P};                    \
        /* the source index is affine along x: a row is a copy, a reversed copy or a fixed-stride gather (was 3 terms per voxel) */ \
        const int64_t sx = y.flip[2] ? -stride[y.perm[2]] : stride[y.perm[2]];                     \
        if (sx != 1 && sx != -1) {   /* x reads a strided source axis: 64 x 64 tiles over x and the output axis that is      \
                                        contiguous in the source (one byte per cache line per row otherwise: the costliest   \
                                        stage of a sample), the third axis outermost; the same copy, tile by tile */          \
            int64_t sd[3], q0 = 0;                                                                \
            for (int d = 0; d < 3; d++) { sd[d] = y.flip[d] ? -stride[y.perm[d]] : stride[y.perm[d]]; if (y.flip[d]) q0 += stride[y.perm[d]] * (n[d] - 1); } \
            const int dc = (sd[0] == 1 || sd[0] == -1) ? 0 : 1, du = 1 - dc;   /* tiled with x; the remaining axis */      \
            for (int u = 0; u < n[du]; u++)                                                       \
                for (int c0 = 0; c0 < n[dc]; c0 += 64)                                            \
                    for (int x0 = 0; x0 < P; x0 += 64) {                                          \
                        const int c1 = c0 + 64 < n[dc] ? c0 + 64 : n[dc], x1 = x0 + 64 < P ? x0 + 64 : P; \
                        for (int c = c0; c < c1; c++) {                                           \
                            const int z = dc ? u : c, yy = dc ? c : u;                            \
                            const T *src = in + q0 + sd[0] * z + sd[1] * yy;                      \
                            T *o = out + ((int64_t)z * P + yy) * P;                               \
                            for (int x = x0; x < x1; x++) o[x] = src[(int64_t)x * sx];            \
                        }                                                                         \
                    }                                                                             \
            return;                                                                               \
        }                                                                                         \
        for (int z = 0; z < D; z++)                                                               \
            for (int yy = 0; yy < P; yy++) {                                                      \
                const int i[3] = {z, yy, 0};                                                      \
                int64_t q = 0;                                                                    \
                for (int d = 0; d < 3; d++) q += stride[y.perm[d]] * (y.flip[d] ? n[d] - 1 - i[d] : i[d]); \
                T *o = out + ((int64_t)z * P + yy) * P;                                           \
                const T *src = in + q;                                                            \
                if (sx == 1) memcpy(o, src, (size_t)P * sizeof(T));                               \
                else if (sx == -1) for (int x = 0; x < P; x++) o[x] = src[-x];                    \
                else for (int x = 0; x < P; x++) o[x] = src[(int64_t)x * sx];                     \
            }                                                                                     \
    }
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
    atomic_int stop, failed;
    atomic_uint_fast64_t produced, rejected;
    atomic_uint soft_bits;   /* current soft-target sigma (float bits), set by sampler_set_soft */
    int64_t *slot_j;         /* deterministic mode: batch index held by each slot */
    int64_t next_claim, next_take;   /* deterministic mode: next batch index to fill / to hand out (under mu) */
    atomic_uint_fast64_t prof_ns[16];
    unsigned char (*lvl_ok)[2][MAXLEV];   /* per source: cached level availability (0 unknown, 1 no, 2 yes) */   /* per-stage nanoseconds (env UFSM_SAMPLER_PROF): read, targets, dilate, soft, trust, encode, zscore, augment, x16, other */
    int prof;
    double *cum;   /* cumulative source weights */
    struct { uint32_t *idx; size_t n; int lev; int64_t shape[3]; } *occ;   /* per source: coarse label cells containing papyrus (guides the position draw) */
};

static int PZ(const sample_cfg *c) { return c->Pz > 0 ? c->Pz : c->P; }   /* window depth (z) */
static size_t P3(const sampler *sp) { return (size_t)PZ(&sp->cfg) * sp->cfg.P * sp->cfg.P; }

static batch alloc_batch(const sample_cfg *c) {
    size_t p3 = (size_t)PZ(c) * c->P * c->P;
    batch b;
    const size_t xc = 4 + (c->aux != 0) + (c->cond ? 3 : 0);   /* input channels per sample */
    b.x = c->xfmt ? nullptr : nn_host_alloc((size_t)c->B * xc * p3 * sizeof(float));   /* pinned: the trainer uploads asynchronously */
    b.x16 = c->xfmt ? nn_host_alloc((size_t)c->B * xc * p3 * 2) : nullptr;
    b.t = nn_host_alloc((size_t)c->B * NCH * p3);
    b.m = nn_host_alloc((size_t)c->B * p3);
    b.w = nn_host_alloc((size_t)c->B * NCH);
    b.src = malloc((size_t)c->B * sizeof(int16_t));
    b.level = malloc((size_t)c->B * sizeof(int8_t));
    b.um = malloc((size_t)c->B * sizeof(float));
    b.corner = malloc((size_t)c->B * sizeof *b.corner);
    b.iso = calloc((size_t)c->B, sizeof *b.iso);
    b.sheet = calloc((size_t)c->B, sizeof *b.sheet);
    b.band = c->band ? nn_host_alloc((size_t)c->B * p3) : nullptr;
    b.side = c->side ? nn_host_alloc((size_t)c->B * p3) : nullptr;
    b.pm = c->pmask ? nn_host_alloc((size_t)c->B * p3) : nullptr;
    b.wind = c->cond ? nn_host_alloc((size_t)c->B * p3) : nullptr;
    b.pap = c->pap_head ? nn_host_alloc((size_t)c->B * p3) : nullptr;
    return b;
}

static void free_batch(batch *b) { if (b->pap) nn_host_free(b->pap); if (b->wind) nn_host_free(b->wind); if (b->band) nn_host_free(b->band); if (b->side) nn_host_free(b->side); if (b->pm) nn_host_free(b->pm); if (b->x) nn_host_free(b->x); if (b->x16) nn_host_free(b->x16); nn_host_free(b->t); nn_host_free(b->m); nn_host_free(b->w); free(b->src); free(b->level); free(b->um); free(b->corner); free(b->iso); if (b->sheet) { sheet_batch_free(b->sheet[0]); free(b->sheet); } }

/* Level choice for a source: restrict cfg.level_p to levels the CT has and every target of the source
   can provide (pyramid: same level; regions: levels 0..1). Returns -1 if nothing is usable. */
/* which levels a source can serve (CT and at least one target open): computed once per (source, region channel) since
   source_ct / source_tgt take the global open lock and may open remote stores; before this every draw re-checked every level */
/* Inclusive level-l origins whose whole patch lies in the CT, region and validation box.
   Round starts up and ends down: flooring a non-aligned holdout start leaks outside it. */
static int patch_bounds(const source *s, const z3_meta *m, const int64_t wn[3], int l, const regions *r, int ri, int validation, int64_t lo[3], int64_t hi[3]) {
    int64_t f = (int64_t)1 << l;
    for (int d = 0; d < 3; d++) {
        const int64_t P = wn[d];
        lo[d] = 0; hi[d] = m->shape[d] - P;
        if (r) {
            int64_t a = r->origin[ri][d], end = a + (r->rsize ? r->rsize[ri] : r->size);
            int64_t rl = (a + f - 1) / f, rh = end / f - P;
            if (rl > lo[d]) lo[d] = rl;
            if (rh < hi[d]) hi[d] = rh;
        }
        if (validation) {
            if (s->hold_n[d] <= 0) return 0;
            int64_t hl = (s->hold_o[d] + f - 1) / f, hh = (s->hold_o[d] + s->hold_n[d]) / f - P;
            if (hl > lo[d]) lo[d] = hl;
            if (hh < hi[d]) hi[d] = hh;
        }
        if (hi[d] < lo[d]) return 0;
    }
    return 1;
}
static int same_ct(const source *a, const source *b) {
    return a == b || (a->um == b->um && !strcmp(a->ct_key, b->ct_key) && !strcmp(store_root(a->s), store_root(b->s)));
}
static int training_hits_holdout(const sampler *sp, const source *s, const int64_t o[3], int l) {
    const int64_t span[3] = {(int64_t)PZ(&sp->cfg) << l, (int64_t)sp->cfg.P << l, (int64_t)sp->cfg.P << l};
    for (int i = 0; i < sp->S->n; i++) {
        const source *h = &sp->S->src[i]; if (!h->hold_n[0] || !same_ct(s, h)) continue;
        int hit = 1;
        for (int d = 0; d < 3; d++) { int64_t a = o[d] << l; if (a + span[d] <= h->hold_o[d] || a >= h->hold_o[d] + h->hold_n[d]) hit = 0; }
        if (hit) return 1;
    }
    return 0;
}
/* Any free cell of the origin grid begins at lo or just after a forbidden interval.
   Check these corners to reject a source covered by the union of same-CT holdouts. */
static int training_corner_exists(const sampler *sp, const source *s, int l, const int64_t lo[3], const int64_t hi[3]) {
    int n[3] = {1, 1, 1}; int64_t v[3][sp->S->n + 1], f = (int64_t)1 << l;
    for (int d = 0; d < 3; d++) v[d][0] = lo[d];
    if (!training_hits_holdout(sp, s, lo, l)) return 1;
    for (int i = 0; i < sp->S->n; i++) {
        const source *h = &sp->S->src[i]; if (!h->hold_n[0] || !same_ct(s, h)) continue;
        for (int d = 0; d < 3; d++) { int64_t end = (h->hold_o[d] + h->hold_n[d] + f - 1) / f; if (end > lo[d] && end <= hi[d]) v[d][n[d]++] = end; }
    }
    for (int z = 0; z < n[0]; z++) for (int y = 0; y < n[1]; y++) for (int x = 0; x < n[2]; x++) {
        int64_t o[3] = {v[0][z], v[1][y], v[2][x]}; if (!training_hits_holdout(sp, s, o, l)) return 1;
    }
    return 0;
}
static int level_ok(sampler *sp, source *s, int si, int l, int region_ch) {
    int key = region_ch >= 0 ? 1 : 0;
    unsigned char *cache = sp->lvl_ok[si][key];
    if (cache[l] == 0) {   /* 0 unknown, 1 no, 2 yes */
        int ok = source_ct(s, l) != nullptr;
        if (ok) {
            const z3_meta *m = z3_meta_of(source_ct(s, l));
            regions *r = region_ch >= 0 ? s->reg[region_ch] : nullptr;
            int fits = 0; int64_t lo[3], hi[3]; const int64_t wn[3] = {PZ(&sp->cfg), sp->cfg.P, sp->cfg.P};
            for (int i = 0; i < (r ? r->n : 1); i++) if (patch_bounds(s, m, wn, l, r, i, sp->cfg.holdout, lo, hi) &&
                (sp->cfg.holdout || training_corner_exists(sp, s, l, lo, hi))) { fits = 1; break; }
            ok = fits;
        }
        if (ok) {
            int any = 0;
            for (int c = 0; c < NCH; c++) {
                if (c == region_ch) { any = 1; continue; }
                if (s->tgt_key[c] && source_tgt_for_level(s, c, l, nullptr)) any = 1;
            }
            ok = any;
        }
        cache[l] = ok ? 2 : 1;
    }
    return cache[l] == 2;
}
static int pick_level(sampler *sp, source *s, rng *r, int region_ch) {
    double p[MAXLEV], tot = 0;
    int si = (int)(s - sp->S->src);
    for (int l = 0; l < MAXLEV; l++) {
        p[l] = sp->cfg.level_p[l];
        if (p[l] <= 0) continue;
        if (region_ch >= 0 && l > 1) { p[l] = 0; continue; }
        if (l < s->min_level || (s->max_level >= 0 && l > s->max_level)) { p[l] = 0; continue; }
        if (!level_ok(sp, s, si, l, region_ch)) { p[l] = 0; continue; }
        tot += p[l];
    }
    if (tot <= 0) return -1;
    double u = runif(r) * tot;
    for (int l = 0; l < MAXLEV; l++) { u -= p[l]; if (p[l] > 0 && u <= 0) return l; }
    return -1;
}

/* Fill patch i of batch b. Returns 0 on success, 1 if rejected (try again), -1 on I/O error. */
static double tnow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
#define PROF_MARK(stage) do { if (sp->prof) { double t_ = tnow(); atomic_fetch_add(&sp->prof_ns[stage], (uint_fast64_t)((t_ - pt) * 1e9)); pt = t_; } } while (0)
enum { PS_SRC, PS_LEVEL, PS_PICK, PS_PROBE, PS_READ, PS_STATS, PS_TARGETS, PS_DILATE, PS_SOFT, PS_TRUST, PS_ENCODE, PS_ZSCORE, PS_AUGMENT, PS_X16, PS_SCAN, PS_N };
static const char *prof_names[PS_N] = {"src", "level", "pick", "probe", "ctread", "ctstats", "targets", "dilate", "soft", "trust", "encode", "zscore", "augment", "x16", "scan"};
/* 3-4-5 chamfer distance (3 per voxel step) from the voxels of t that are annotated surface (0 < t < 255): two raster
   passes. Returns 0 (dm all CH_INF) when the patch has no surface. */
#define CH_INF 60000
static inline uint16_t mn16(uint16_t a, uint16_t b) { return a < b ? a : b; }
/* one raster pass (dir +1 forward, -1 backward). Per row, the neighbours in the previous row and the previous plane are
   final for the pass, so they are folded in for the whole row at once (vectorisable); the in-row dependency is then a
   scalar scan. Same result as visiting the voxels in raster order. */
static void chamfer_pass(uint16_t *dm, int D, int P, int dir) {   /* a D x P x P window */
    const size_t P2 = (size_t)P * P;
    for (int zi = 0; zi < D; zi++) {
        const int z = dir > 0 ? zi : D - 1 - zi, zp = z - dir;
        for (int yi = 0; yi < P; yi++) {
            const int y = dir > 0 ? yi : P - 1 - yi, yp = y - dir;
            uint16_t *row = dm + (size_t)z * P2 + (size_t)y * P;
            /* (row, weight face / edge) pairs: previous row of this plane (3, 4); previous plane rows y (3, 4), y-1 and y+1 (4, 5) */
            const uint16_t *nb[4] = {nullptr, nullptr, nullptr, nullptr}; uint16_t wf[4] = {3, 3, 4, 4}, we[4] = {4, 4, 5, 5};
            if (yp >= 0 && yp < P) nb[0] = dm + (size_t)z * P2 + (size_t)yp * P;
            if (zp >= 0 && zp < D) {
                nb[1] = dm + (size_t)zp * P2 + (size_t)y * P;
                if (y > 0) nb[2] = nb[1] - P;
                if (y + 1 < P) nb[3] = nb[1] + P;
            }
            for (int q = 0; q < 4; q++) {
                const uint16_t *r = nb[q]; if (!r) continue;
                const uint16_t f = wf[q], e = we[q];
                row[0] = mn16(row[0], r[0] + f); if (P > 1) row[0] = mn16(row[0], r[1] + e);
                for (int x = 1; x < P - 1; x++) { uint16_t v = row[x]; v = mn16(v, r[x] + f); v = mn16(v, r[x - 1] + e); v = mn16(v, r[x + 1] + e); row[x] = v; }
                if (P > 1) { row[P - 1] = mn16(row[P - 1], r[P - 1] + f); row[P - 1] = mn16(row[P - 1], r[P - 2] + e); }
            }
            if (dir > 0) for (int x = 1; x < P; x++) row[x] = mn16(row[x], row[x - 1] + 3);
            else for (int x = P - 2; x >= 0; x--) row[x] = mn16(row[x], row[x + 1] + 3);
        }
    }
}
static int chamfer345(const uint8_t *t, uint16_t *dm, int D, int P) {
    size_t p3 = (size_t)D * P * P; int any = 0;
    for (size_t k = 0; k < p3; k++) { int s = t[k] != 255 && t[k] > 0; dm[k] = s ? 0 : CH_INF; any |= s; }
    if (!any) return 0;
    chamfer_pass(dm, D, P, 1);
    chamfer_pass(dm, D, P, -1);
    return 1;
}
/* The same 3-4-5 chamfer distance saturated at 255 in bytes (dm8 = min(distance, 255): every path to a distance below 255
   only passes values below it, so the saturating adds and mins are exact there). The in-row scans run as log-step
   min-plus scans (offsets 1..64; 3 * 85 >= 255 saturates) in vectorisable loops. Callers use it when every threshold
   they compare with is below 255. */
static inline uint8_t sadd8(uint8_t a, uint8_t b) { const uint8_t v = (uint8_t)(a + b); return v < a ? 255 : v; }   /* vpaddusb */
static inline uint8_t mn8(uint8_t a, uint8_t b) { return a < b ? a : b; }
static void chamfer_pass8(uint8_t *dm, int D, int P, int dir, uint8_t *tmp) {
    const size_t P2 = (size_t)P * P;
    for (int zi = 0; zi < D; zi++) {
        const int z = dir > 0 ? zi : D - 1 - zi, zp = z - dir;
        for (int yi = 0; yi < P; yi++) {
            const int y = dir > 0 ? yi : P - 1 - yi, yp = y - dir;
            uint8_t *row = dm + (size_t)z * P2 + (size_t)y * P;
            const uint8_t *nb[4] = {nullptr, nullptr, nullptr, nullptr}; const uint8_t wf[4] = {3, 3, 4, 4}, we[4] = {4, 4, 5, 5};
            if (yp >= 0 && yp < P) nb[0] = dm + (size_t)z * P2 + (size_t)yp * P;
            if (zp >= 0 && zp < D) {
                nb[1] = dm + (size_t)zp * P2 + (size_t)y * P;
                if (y > 0) nb[2] = nb[1] - P;
                if (y + 1 < P) nb[3] = nb[1] + P;
            }
            for (int q = 0; q < 4; q++) {
                const uint8_t *r = nb[q]; if (!r) continue;
                const uint8_t f = wf[q], e = we[q];
                row[0] = mn8(row[0], sadd8(r[0], f)); if (P > 1) row[0] = mn8(row[0], sadd8(r[1], e));
                for (int x = 1; x < P - 1; x++) row[x] = mn8(mn8(row[x], sadd8(r[x], f)), mn8(sadd8(r[x - 1], e), sadd8(r[x + 1], e)));
                if (P > 1) { row[P - 1] = mn8(row[P - 1], sadd8(r[P - 1], f)); row[P - 1] = mn8(row[P - 1], sadd8(r[P - 2], e)); }
            }
            for (int st = 1; st < P && st <= 64; st <<= 1) {   /* row[x] = min over j of row[j] + 3 |x - j|, j before x in the pass direction */
                const uint8_t a = (uint8_t)(3 * st);
                memcpy(tmp, row, (size_t)P);
                if (dir > 0) for (int x = st; x < P; x++) row[x] = mn8(row[x], sadd8(tmp[x - st], a));
                else for (int x = 0; x < P - st; x++) row[x] = mn8(row[x], sadd8(tmp[x + st], a));
            }
        }
    }
}
static int chamfer345_8(const uint8_t *t, uint8_t *dm, int D, int P) {
    size_t p3 = (size_t)D * P * P; unsigned any = 0;
    for (size_t k = 0; k < p3; k++) { const unsigned sv = (t[k] != 255) & (t[k] > 0); dm[k] = sv ? 0 : 255; any |= sv; }
    if (!any) return 0;
    uint8_t *tmp = malloc((size_t)P); if (!tmp) abort();
    chamfer_pass8(dm, D, P, 1, tmp);
    chamfer_pass8(dm, D, P, -1, tmp);
    free(tmp);
    return 1;
}
/* per-voxel Gaussian noise: 65536 N(0,1) quantiles (Acklam's inverse normal CDF), indexed by 16 random bits */
static float g_ntab[65536];
static pthread_once_t g_ntab_once = PTHREAD_ONCE_INIT;
static double inv_ncdf(double p) {
    static const double a[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
    static const double d[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00};
    if (p < 0.02425) { double q = sqrt(-2 * log(p)); return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    if (p > 1 - 0.02425) { double q = sqrt(-2 * log(1 - p)); return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1); }
    double q = p - 0.5, t = q * q;
    return (((((a[0] * t + a[1]) * t + a[2]) * t + a[3]) * t + a[4]) * t + a[5]) * q / (((((b[0] * t + b[1]) * t + b[2]) * t + b[3]) * t + b[4]) * t + 1);
}
static void ntab_init(void) { for (int i = 0; i < 65536; i++) g_ntab[i] = (float)inv_ncdf((i + 0.5) / 65536.0); }
static inline uint16_t bf16_rn(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }
/* The network input of one patch in one pass: channel 0 = z-scored CT with the intensity jitter (gain ia, offset ib,
   noise isg N(0,1)), channels 1..3 = the radial unit vector (source z component 0), all read through the cube
   symmetry y (output axis d reads source axis perm[d], reversed when flipped; vector channel 1 + d = (-1)^flip[d] *
   source component perm[d]). cyz / cxz: axis centre per source z. Output fp32 X (xfmt 0) or 16-bit H (1 fp16,
   2 bf16). One rng draw per 4 voxels, as the old separate jitter pass. nrow: P floats of scratch. */
static void write_x_aug(const uint8_t *ctu, int P, const int64_t o[3], float fmean, float fisd, int hasax, const float *cyz, const float *cxz,
                    sym y, int jitter, float ia, float ib, float isg, rng *r, float *nrow, int xfmt, float *X, uint16_t *H, const ct_aug_plan *ap, float *noise_row_state, float *noise_plane_state,
                    const sheet_dataset *sheet,const double (*sheet_params)[4]) {
    pthread_once(&g_ntab_once, ntab_init);
    const size_t p3 = (size_t)P * P * P;
    const float sgn[3] = {y.flip[0] ? -1.f : 1.f, y.flip[1] ? -1.f : 1.f, y.flip[2] ? -1.f : 1.f};
    for (int z = 0; z < P; z++)
        for (int yy = 0; yy < P; yy++) {
            const int io[3] = {z, yy, 0};
            int s0[3], ds[3] = {0, 0, 0};
            for (int d = 0; d < 3; d++) s0[y.perm[d]] = y.flip[d] ? P - 1 - io[d] : io[d];
            ds[y.perm[2]] = y.flip[2] ? -1 : 1;   /* the source coordinate that moves with output x */
            if (jitter) {
                for (int x = 0; x + 4 <= P; x += 4) { uint64_t u = rnext(r); nrow[x] = g_ntab[u & 0xffff]; nrow[x + 1] = g_ntab[(u >> 16) & 0xffff]; nrow[x + 2] = g_ntab[(u >> 32) & 0xffff]; nrow[x + 3] = g_ntab[u >> 48]; }
                for (int x = P & ~3; x < P; x++) nrow[x] = g_ntab[rnext(r) & 0xffff];
                /* Isotropic correlated noise is generated in output coordinates; cube symmetries preserve its distribution. */
                if (ap && ap->noise_rho > 0) ct_aug_noise_row(ap, nrow, noise_row_state, noise_plane_state, P, z, yy);
            }
            const size_t ko = ((size_t)z * P + yy) * P;
            for (int x = 0; x < P; x++) {
                const int sz_ = s0[0] + ds[0] * x, sy_ = s0[1] + ds[1] * x, sx_ = s0[2] + ds[2] * x;
                const size_t q = ((size_t)sz_ * P + sy_) * P + sx_;
                float v0 = ap ? ct_aug_value(ap, ctu, P, sz_, sy_, sx_) : ((float)ctu[q] - fmean) * fisd;
                if (jitter) v0 = ia * v0 + ib + isg * nrow[x];
                float comp[3] = {0.f, 0.f, 0.f};
                if (hasax) {
                    const float dyv = (float)(o[1] + sy_) - cyz[sz_], dxv = (float)(o[2] + sx_) - cxz[sz_];
                    const float inv = 1.f / (sqrtf(dyv * dyv + dxv * dxv) + 1e-6f);
                    comp[1] = dyv * inv; comp[2] = dxv * inv;
                }
                float v1 = sgn[0] * comp[y.perm[0]];
                const float v2 = sgn[1] * comp[y.perm[1]], v3 = sgn[2] * comp[y.perm[2]];
                if (sheet) {
                    const double *a=sheet_params[sz_];
                    double prior=hypot((double)(o[1]+sy_)-a[0],(double)(o[2]+sx_)-a[1])/a[2]+a[3];
                    v1=(float)((prior-sheet->center)/sheet->scale);
                }
                const size_t k = ko + x;
                if (xfmt == 1) { _Float16 h0 = (_Float16)v0, h1 = (_Float16)v1, h2 = (_Float16)v2, h3 = (_Float16)v3; memcpy(&H[k], &h0, 2); memcpy(&H[p3 + k], &h1, 2); memcpy(&H[2 * p3 + k], &h2, 2); memcpy(&H[3 * p3 + k], &h3, 2); }
                else if (xfmt == 2) { H[k] = bf16_rn(v0); H[p3 + k] = bf16_rn(v1); H[2 * p3 + k] = bf16_rn(v2); H[3 * p3 + k] = bf16_rn(v3); }
                else { X[k] = v0; X[p3 + k] = v1; X[2 * p3 + k] = v2; X[3 * p3 + k] = v3; }
            }
        }
}
/* fp16 (round to nearest even, as the (_Float16) cast) of n floats: F16C 8 at a time, which the cast loop does not vectorise to */
static void f16_row(const float *v, uint16_t *h, int n) {
    int x = 0;
#ifdef __F16C__
    for (; x + 8 <= n; x += 8) _mm_storeu_si128((__m128i *)(h + x), _mm256_cvtps_ph(_mm256_loadu_ps(v + x), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#endif
    for (; x < n; x++) { _Float16 t = (_Float16)v[x]; memcpy(h + x, &t, 2); }
}
/* write_x_aug without an appearance plan or sheet prior, one output row at a time in vectorisable loops (no math errno, so
   sqrtf vectorises); bit-identical to it: the same expressions, rng draws and signed zeros. D x P x P windows (D != P: slabs,
   z-keeping symmetries only). */
[[gnu::optimize("no-math-errno")]]
static void write_x(const uint8_t *ctu, int D, int P, const int64_t o[3], float fmean, float fisd, int hasax, const float *cyz, const float *cxz,
                    sym y, int jitter, float ia, float ib, float isg, rng *r, float *nrow, int xfmt, float *X, uint16_t *H,
                    const ct_aug_plan *ap, float *noise_row_state, float *noise_plane_state) {   /* ap: the CT appearance plan (cubes, D == P), else nullptr */
    pthread_once(&g_ntab_once, ntab_init);
    const size_t p3 = (size_t)D * P * P; const int nd[3] = {D, P, P};
    const float sgn[3] = {y.flip[0] ? -1.f : 1.f, y.flip[1] ? -1.f : 1.f, y.flip[2] ? -1.f : 1.f};
    float *a = malloc(9 * (size_t)P * sizeof *a);   /* rows: v0..v3, dy, dx, radial y, radial x, zero */
    if (!a) abort();
    float *v[4] = {a, a + P, a + 2 * P, a + 3 * P}, *dy = a + 4 * P, *dx = a + 5 * P, *comp[3] = {a + 8 * P, a + 6 * P, a + 7 * P};
    memset(comp[0], 0, (size_t)P * sizeof *a);
    if (!hasax) { memset(comp[1], 0, (size_t)P * sizeof *a); memset(comp[2], 0, (size_t)P * sizeof *a); }
    const int o1 = (int)o[1], o2 = (int)o[2];
    for (int z = 0; z < D; z++)
        for (int yy = 0; yy < P; yy++) {
            const int io[3] = {z, yy, 0};
            int s0[3], ds[3] = {0, 0, 0};
            for (int d = 0; d < 3; d++) s0[y.perm[d]] = y.flip[d] ? nd[d] - 1 - io[d] : io[d];
            ds[y.perm[2]] = y.flip[2] ? -1 : 1;
            if (jitter) {
                for (int x = 0; x + 4 <= P; x += 4) { uint64_t u = rnext(r); nrow[x] = g_ntab[u & 0xffff]; nrow[x + 1] = g_ntab[(u >> 16) & 0xffff]; nrow[x + 2] = g_ntab[(u >> 32) & 0xffff]; nrow[x + 3] = g_ntab[u >> 48]; }
                for (int x = P & ~3; x < P; x++) nrow[x] = g_ntab[rnext(r) & 0xffff];
                if (ap && ap->noise_rho > 0) ct_aug_noise_row(ap, nrow, noise_row_state, noise_plane_state, P, z, yy);   /* correlated noise (write_x_aug's order) */
            }
            const ptrdiff_t q0 = ((ptrdiff_t)s0[0] * P + s0[1]) * P + s0[2], qs = ((ptrdiff_t)ds[0] * P + ds[1]) * P + ds[2];
            const uint8_t *src = ctu + q0;
            float *v0 = v[0];
            if (ap) for (int x = 0; x < P; x++) v0[x] = ct_aug_value(ap, ctu, P, s0[0] + ds[0] * x, s0[1] + ds[1] * x, s0[2] + ds[2] * x);   /* LUT, filter, shading */
            else if (qs == 1) for (int x = 0; x < P; x++) v0[x] = ((float)src[x] - fmean) * fisd;
            else for (int x = 0; x < P; x++) v0[x] = ((float)src[qs * x] - fmean) * fisd;
            if (jitter) for (int x = 0; x < P; x++) v0[x] = ia * v0[x] + ib + isg * nrow[x];
            if (hasax) {
                float *ry = comp[1], *rx = comp[2];
                if (!ds[0]) {   /* the source z is fixed along the row */
                    const float cy = cyz[s0[0]], cx = cxz[s0[0]];
                    for (int x = 0; x < P; x++) { dy[x] = (float)(o1 + s0[1] + ds[1] * x) - cy; dx[x] = (float)(o2 + s0[2] + ds[2] * x) - cx; }
                } else for (int x = 0; x < P; x++) { const int sz_ = s0[0] + ds[0] * x; dy[x] = (float)(o1 + s0[1]) - cyz[sz_]; dx[x] = (float)(o2 + s0[2]) - cxz[sz_]; }
                for (int x = 0; x < P; x++) { const float inv = 1.f / (sqrtf(dy[x] * dy[x] + dx[x] * dx[x]) + 1e-6f); ry[x] = dy[x] * inv; rx[x] = dx[x] * inv; }
            }
            for (int d = 0; d < 3; d++) { const float g = sgn[d], *cp = comp[y.perm[d]]; float *vd = v[1 + d]; for (int x = 0; x < P; x++) vd[x] = g * cp[x]; }
            const size_t ko = ((size_t)z * P + yy) * P;
            for (int ch = 0; ch < 4; ch++) {
                const float *vc = v[ch];
                if (xfmt == 1) f16_row(vc, H + (size_t)ch * p3 + ko, P);
                else if (xfmt == 2) { uint16_t *hd = H + (size_t)ch * p3 + ko; for (int x = 0; x < P; x++) hd[x] = bf16_rn(vc[x]); }
                else memcpy(X + (size_t)ch * p3 + ko, vc, (size_t)P * sizeof *vc);
            }
        }
    free(a);
}
/* Continuous geometry uses source coordinates for CT, labels, q0 and the axis.
   Only valid source voxels are supervised: rotated cube corners are ignored.
   The four input channels stay unchanged; a winding task uses q0 and the two
   Y/X projections of the transported radial vector. */
static void write_spatial(const uint8_t *ctu,const uint8_t *target,const uint8_t *mask,
                         uint8_t *T,uint8_t *M,int P,const int64_t o[3],float mean,float isd,
                         int hasax,const float *cy,const float *cx,sym sy,const spatial_aug *a,
                         float gain,float offset,float noise,rng *r,float *nrow,int xfmt,float *X,uint16_t *H,
                         const ct_aug_plan *ap,float *noise_row,float *noise_plane,
                         const sheet_dataset *sheet,const double (*params)[4],float *appearance) {
    pthread_once(&g_ntab_once,ntab_init);
    size_t p3=(size_t)P*P*P; float *table=malloc(6*(size_t)P*sizeof *table);
    if (!table) abort();
    spatial_aug_tables(a,table);
    if (ap && ap->filter_mode) {
        for (int z=0;z<P;z++) for (int y=0;y<P;y++) for (int x=0;x<P;x++)
            appearance[((size_t)z*P+y)*P+x]=ct_aug_value(ap,ctu,P,z,y,x);
    }
    const float sign[3]={sy.flip[0]?-1:1,sy.flip[1]?-1:1,sy.flip[2]?-1:1};
    for (int z=0;z<P;z++) for (int y=0;y<P;y++) {
        for (int x=0;x+4<=P;x+=4) { uint64_t v=rnext(r); for (int j=0;j<4;j++) nrow[x+j]=g_ntab[(v>>(16*j))&65535]; }
        for (int x=P&~3;x<P;x++) nrow[x]=g_ntab[rnext(r)&65535];
        if (ap && ap->noise_rho>0) ct_aug_noise_row(ap,nrow,noise_row,noise_plane,P,z,y);
        for (int x=0;x<P;x++) {
            size_t k=((size_t)z*P+y)*P+x;
            float u[3],jac[3],v[3]; spatial_aug_pull(a,table,z,y,x,u,jac);
            for (int d=0;d<3;d++) v[sy.perm[d]]=sy.flip[d]?P-1-u[d]:u[d];
            float out[4]={0}; int inside=1;
            for (int d=0;d<3;d++) if (v[d]<0 || v[d]>P-1) inside=0;
            for (int ch=0;ch<NCH;ch++) T[(size_t)ch*p3+k]=0;
            M[k]=0;
            if (inside) {
                int base[3]; float f[3];
                for (int d=0;d<3;d++) { base[d]=(int)v[d]; if (base[d]>=P-1) base[d]=P-2; f[d]=v[d]-base[d]; }
                float tv[NCH]={0},cv=0; int valid=1;
                for (int bz=0;bz<2;bz++) for (int by=0;by<2;by++) for (int bx=0;bx<2;bx++) {
                    float weight=(bz?f[0]:1-f[0])*(by?f[1]:1-f[1])*(bx?f[2]:1-f[2]);
                    if (weight<=1e-7f) continue;
                    size_t j=((size_t)(base[0]+bz)*P+base[1]+by)*P+base[2]+bx;
                    valid &= mask[j]!=0;
                    float val=ap && ap->filter_mode?appearance[j]:ap?ap->lut[ctu[j]]:((float)ctu[j]-mean)*isd;
                    cv+=weight*val;
                    for (int ch=0;ch<NCH;ch++) tv[ch]+=weight*target[(size_t)ch*p3+j];
                }
                if (ap && !ap->filter_mode && P>1) for (int d=0;d<3;d++) cv+=ap->shading[d]*(2*v[d]/(P-1)-1);
                out[0]=gain*cv+offset+noise*nrow[x]; M[k]=(uint8_t)valid;
                for (int ch=0;ch<NCH;ch++) T[(size_t)ch*p3+k]=(uint8_t)lroundf(tv[ch]);
                float frac=f[0],ayy=cy[base[0]]*(1-frac)+cy[base[0]+1]*frac;
                float axx=cx[base[0]]*(1-frac)+cx[base[0]+1]*frac;
                float dy=(float)o[1]+v[1]-ayy,dx=(float)o[2]+v[2]-axx;
                float radius=sqrtf(dy*dy+dx*dx)+1e-6f;
                float radial[3]={0,hasax?dy/radius:0,hasax?dx/radius:0},discrete[3],transported[3];
                for (int d=0;d<3;d++) discrete[d]=sign[d]*radial[sy.perm[d]];
                if (a->affine) { float q[3]; for (int d=0;d<3;d++) q[d]=(float)a->Ainv[d][0]*discrete[0]+(float)a->Ainv[d][1]*discrete[1]+(float)a->Ainv[d][2]*discrete[2]; memcpy(discrete,q,sizeof q); }
                float norm=1e-12f;
                for (int d=0;d<3;d++) { transported[d]=jac[d]*((float)a->R[d][0]*discrete[0]+(float)a->R[d][1]*discrete[1]+(float)a->R[d][2]*discrete[2]); norm+=transported[d]*transported[d]; }
                norm=sqrtf(norm); for (int d=0;d<3;d++) out[1+d]=transported[d]/norm;
                if (sheet) {
                    double q[4]; for (int d=0;d<4;d++) q[d]=params[base[0]][d]*(1-(double)frac)+params[base[0]+1][d]*frac;
                    double prior=hypot(o[1]+(double)v[1]-q[0],o[2]+(double)v[2]-q[1])/q[2]+q[3];
                    out[1]=(float)((prior-sheet->center)/sheet->scale);
                }
            }
            for (int ch=0;ch<4;ch++) {
                size_t j=(size_t)ch*p3+k;
                if (xfmt==1) { _Float16 h=(_Float16)out[ch]; memcpy(H+j,&h,2); }
                else if (xfmt==2) H[j]=bf16_rn(out[ch]); else X[j]=out[ch];
            }
        }
    }
    free(table);
}
/* Keep complete sparse terms only. q/q0 are material scalar coordinates and
   are invariant under reflection/deformation; coordinates use the same warp
   as the dense targets. This also removes paths leaving the valid crop. */
static void warp_sheet_batch(sheet_batch *b,const spatial_aug *a,const uint8_t *mask) {
    size_t nt=0,np=0; int P=a->P;
    for (size_t t=0;t<b->nt;t++) {
        sheet_term term=b->terms[t]; sheet_point points[SHEET_PATH]; int valid=1;
        for (unsigned i=0;i<term.count;i++) {
            points[i]=b->points[term.first+i]; double source[3],output[3];
            for (int d=0;d<3;d++) source[d]=points[i].xyz[d];
            spatial_aug_forward(a,source,output);
            int v[3]; for (int d=0;d<3;d++) { if (output[d]<0 || output[d]>=P-1) valid=0; points[i].xyz[d]=(float)output[d]; v[d]=(int)lround(output[d]); }
            if (valid && !mask[((size_t)v[0]*P+v[1])*P+v[2]]) valid=0;
        }
        if (!valid) continue;
        term.first=(uint32_t)np; b->terms[nt++]=term;
        memcpy(b->points+np,points,term.count*sizeof *points); np+=term.count;
    }
    b->np=np; b->nt=nt;
}
/* band of a native window (task band_affinity): the band field (src/band.h) on the label grid of the window plus a halo,
   nearest-upsampled like the binary labels (native g -> label (g + 1) >> 1). out: P^3; side (optional): P^3 SIDE_* labels, or
   with phase the winding phase (band.h PHASE_*). dense > 0 (phase): a phase voxel whose windings lie more than dense label voxels
   apart is kept with probability dense / spacing (a hash of its global position and seed, no RNG draws), so the loss weighs
   closely packed windings up, as upstream lasagna's density weight. */
static uint64_t mix64(uint64_t x) { x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 27; x *= 0x94d049bb133111ebull; return x ^ (x >> 31); }
/* per-worker pool of the per-sample window-sized scratch (draw()'s P^3 buffers, 56 MB at P 384; the band / phase / aux window
   builders' label and read buffers, several MB each): glibc serves blocks above its mmap threshold (<= 32 MB) with mmap / munmap,
   so every sample paid ~14k page faults per buffer under the process-wide mmap lock that all workers share. Sizes of 1 MB and
   more are rounded up to whole MB (windows clipped at a volume edge reuse the same buffers); zero = calloc semantics over n. */
#define TP_N 32
static _Thread_local struct { void *b[TP_N]; size_t n[TP_N]; int k; } tpool;
static size_t tp_class(size_t n) { return n >= ((size_t)1 << 20) ? (n + ((size_t)1 << 20) - 1) & ~(((size_t)1 << 20) - 1) : n; }
static void *tp_get(size_t n, int zero) {
    const size_t c = tp_class(n);
    for (int i = 0; i < tpool.k; i++)
        if (tpool.n[i] == c) { void *p = tpool.b[i]; tpool.b[i] = tpool.b[--tpool.k]; tpool.n[i] = tpool.n[tpool.k]; if (zero) memset(p, 0, n); return p; }
    return zero ? calloc(c, 1) : malloc(c);
}
static void tp_put(void *p, size_t n) {
    if (!p) return;
    if (tpool.k < TP_N) { tpool.b[tpool.k] = p; tpool.n[tpool.k] = tp_class(n); tpool.k++; } else free(p);
}
static void tp_drain(void) { while (tpool.k) free(tpool.b[--tpool.k]); }
/* --iso-aug: the foreground of a D x P x P window for a void cutoff iso, 1 within R voxels (3-4-5 chamfer) of papyrus (clean CT >= iso),
   whatever the voxel's own CT; pooled buffer (tp_put it), nullptr when out of memory. Labels outside it (lifted off the sheets into
   the void) become ignored, labels inside are always kept, even where they cross void. */
static uint8_t *iso_foreground(const uint8_t *ctu, int D, int P, int iso, int R) {
    const size_t p3 = (size_t)D * P * P;
    uint8_t *fg = tp_get(p3, 0), *pap = tp_get(p3, 0);
    if (!fg || !pap) { tp_put(fg, p3); tp_put(pap, p3); return nullptr; }
    for (size_t k = 0; k < p3; k++) pap[k] = ctu[k] >= iso;
    const int any = chamfer345_8(pap, fg, D, P), tr = 3 * R > 254 ? 254 : 3 * R;
    for (size_t k = 0; k < p3; k++) fg[k] = any && fg[k] <= tr;
    tp_put(pap, p3);
    return fg;
}
static int band_window(source *s, const int64_t o[3], int P, uint8_t *out, uint8_t *side, int phase, int dense, uint64_t seed) {
    band_params bp = {s->band_radius, s->band_span, 0};
    const int halo = (int)ceilf((bp.radius + bp.span) / 2.f) + 2;
    int64_t lo[3], ln[3], ro[3], rn[3]; int ni[3];
    const z3_meta *m = z3_meta_of(s->band);
    for (int d = 0; d < 3; d++) {
        lo[d] = ((o[d] + 1) >> 1) - halo; ln[d] = ((o[d] + P) >> 1) - ((o[d] + 1) >> 1) + 1 + 2 * halo; ni[d] = (int)ln[d];
        ro[d] = lo[d] < 0 ? 0 : lo[d]; const int64_t e = lo[d] + ln[d] < m->shape[d] ? lo[d] + ln[d] : m->shape[d]; rn[d] = e - ro[d];
    }
    const size_t N = (size_t)ln[0] * ln[1] * ln[2];
    uint8_t *codes = tp_get(N, 1), *bl = tp_get(N, 0), *sl = side ? tp_get(N, 0) : nullptr, *sp = side && phase && dense > 0 ? tp_get(N, 0) : nullptr;
    double *cy = malloc(ln[0] * sizeof(double)), *cx = malloc(ln[0] * sizeof(double));
    int rc = !codes || !bl || !cy || !cx || (side && !sl) || (side && phase && dense > 0 && !sp);
    if (!rc && rn[0] > 0 && rn[1] > 0 && rn[2] > 0) {
        const size_t rnn = (size_t)rn[0] * rn[1] * rn[2];
        uint8_t *tmp = tp_get(rnn, 0);
        rc = !tmp || z3_read(s->band, ro, rn, tmp, 1);
        if (!rc) for (int64_t z = 0; z < rn[0]; z++) for (int64_t y = 0; y < rn[1]; y++)
            memcpy(codes + ((size_t)(z + ro[0] - lo[0]) * ln[1] + (y + ro[1] - lo[1])) * ln[2] + (ro[2] - lo[2]), tmp + ((size_t)z * rn[1] + y) * rn[2], (size_t)rn[2]);
        tp_put(tmp, rnn);
    }
    if (!rc) {
        for (int64_t z = 0; z < ln[0]; z++) axis_at(&s->ax, 2.0 * (z + lo[0]) + 0.5, &cy[z], &cx[z]);
        rc = phase ? band_field_phase(codes, ni, lo, cy, cx, bp, bl, sl, sp) : band_field_side(codes, ni, lo, cy, cx, bp, bl, sl);
    }
    if (!rc) for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) {
        const size_t r0 = ((size_t)(((o[0] + z + 1) >> 1) - lo[0]) * ln[1] + (((o[1] + y + 1) >> 1) - lo[1])) * ln[2];
        const uint8_t *row = bl + r0;
        uint8_t *dst = out + ((size_t)z * P + y) * P;
        for (int x = 0; x < P; x++) dst[x] = row[((o[2] + x + 1) >> 1) - lo[2]];
        if (side) { const uint8_t *sr = sl + r0; uint8_t *sd = side + ((size_t)z * P + y) * P; for (int x = 0; x < P; x++) sd[x] = sr[((o[2] + x + 1) >> 1) - lo[2]]; }
        if (sp) {
            const uint8_t *spr = sp + r0; uint8_t *sd = side + ((size_t)z * P + y) * P;
            const uint64_t hz = mix64(seed ^ (uint64_t)(o[0] + z) * 0x9e3779b97f4a7c15ull ^ (uint64_t)(o[1] + y) * 0xc2b2ae3d27d4eb4full);
            for (int x = 0; x < P; x++) {
                const int v = spr[((o[2] + x + 1) >> 1) - lo[2]];
                if (sd[x] >= PHASE_PERIOD || v <= dense) continue;
                if ((mix64(hz ^ (uint64_t)(o[2] + x)) >> 40) * (uint64_t)v >= ((uint64_t)dense << 24)) sd[x] = PHASE_UNKNOWN;   /* keep with p = dense / v */
            }
        }
    }
    tp_put(codes, N); tp_put(bl, N); tp_put(sl, N); tp_put(sp, N); free(cy); free(cx);
    return rc ? -1 : 0;
}

/* nonzero count, sum and sum of squares of a uint8 array: integer partial sums per 4096 values (vectorise), exact as the
   double accumulation of the integers they replace */
static void u8_stats(const uint8_t *a, size_t n, size_t *nz, double *sum, double *sq) {
    uint64_t z = 0, s1 = 0, s2 = 0;
    for (size_t i = 0; i < n; i += 4096) {
        const size_t e = n - i < 4096 ? n - i : 4096;
        uint32_t cz = 0, c1 = 0, c2 = 0;
        for (size_t k = 0; k < e; k++) { const uint32_t v = a[i + k]; cz += v != 0; c1 += v; c2 += v * v; }
        z += cz; s1 += c1; s2 += c2;
    }
    *nz = (size_t)z; *sum = (double)s1; *sq = (double)s2;
}
/* the stored phase (src/sources.h source.phase) of a window of the sample's level: shift 1 nearest-upsampled from the grid one
   level coarser (the label grid for native windows) as band_window does, shift 0 read directly */
/* shift >= 2 (a 9.6 um phase for 2.4 um windows): circular trilinear interpolation. Grid voxel j of g = 2^shift native voxels sits at
   g j + (g - 1) / 2 (src/band.c); each window voxel gets the weighted mean phasor of its 8 grid neighbours with a known phase,
   unknown when they hold under half the weight. Between two rectos the phase is a linear ramp, which this reproduces; the 1 -> 0
   wrap at a recto is continuous on the circle. */
static int phase_window_interp(z3 *a, const int64_t o[3], int D, int P, int shift, uint8_t *side) {   /* a D x P x P window, D <= P */
    static float cs[PHASE_PERIOD], sn[PHASE_PERIOD]; static _Atomic int init;
    if (!atomic_load(&init)) { for (int k = 0; k < PHASE_PERIOD; k++) { cs[k] = (float)cos(2 * M_PI * k / PHASE_PERIOD); sn[k] = (float)sin(2 * M_PI * k / PHASE_PERIOD); } atomic_store(&init, 1); }
    const int g = 1 << shift; const double c0 = 0.5 * (g - 1); const int wn[3] = {D, P, P};
    int64_t lo[3], ln[3], ro[3], rn[3]; const z3_meta *m = z3_meta_of(a);
    for (int d = 0; d < 3; d++) {
        lo[d] = (int64_t)floor((o[d] - c0) / g); ln[d] = (int64_t)floor((o[d] + wn[d] - 1 - c0) / g) + 2 - lo[d];
        ro[d] = lo[d] < 0 ? 0 : lo[d]; const int64_t e = lo[d] + ln[d] < m->shape[d] ? lo[d] + ln[d] : m->shape[d]; rn[d] = e - ro[d];
    }
    const size_t N = (size_t)ln[0] * ln[1] * ln[2];
    uint8_t *ph = tp_get(N, 0); int *j0 = malloc(3 * (size_t)P * sizeof *j0); float *tf = malloc(3 * (size_t)P * sizeof *tf);
    if (!ph || !j0 || !tf) { tp_put(ph, N); free(j0); free(tf); return -1; }
    memset(ph, PHASE_UNKNOWN, N);
    if (rn[0] > 0 && rn[1] > 0 && rn[2] > 0) {
        const size_t rnn = (size_t)rn[0] * rn[1] * rn[2];
        uint8_t *tmp = tp_get(rnn, 0);
        if (!tmp || z3_read(a, ro, rn, tmp, 1)) { tp_put(tmp, rnn); tp_put(ph, N); free(j0); free(tf); return -1; }
        for (int64_t z = 0; z < rn[0]; z++) for (int64_t y = 0; y < rn[1]; y++)
            memcpy(ph + ((size_t)(z + ro[0] - lo[0]) * ln[1] + (y + ro[1] - lo[1])) * ln[2] + (ro[2] - lo[2]), tmp + ((size_t)z * rn[1] + y) * rn[2], (size_t)rn[2]);
        tp_put(tmp, rnn);
    }
    for (int d = 0; d < 3; d++) for (int k = 0; k < wn[d]; k++) {   /* per axis: the lower grid neighbour (in the read region) and the fraction */
        const double u = (o[d] + k - c0) / g; const int64_t j = (int64_t)floor(u);
        j0[d * P + k] = (int)(j - lo[d]); tf[d * P + k] = (float)(u - j);
    }
    for (int z = 0; z < D; z++) for (int y = 0; y < P; y++) {
        const int az = j0[z], ay = j0[P + y]; const float tz = tf[z], ty = tf[P + y];
        uint8_t *dst = side + ((size_t)z * P + y) * P;
        for (int x = 0; x < P; x++) {
            const int ax = j0[2 * P + x]; const float tx = tf[2 * P + x];
            float sx = 0.f, sy = 0.f, wk = 0.f;
            for (int c = 0; c < 8; c++) {
                const int v = ph[((size_t)(az + (c >> 2 & 1)) * ln[1] + ay + (c >> 1 & 1)) * ln[2] + ax + (c & 1)];
                if (v >= PHASE_PERIOD) continue;
                const float wv = (c & 4 ? tz : 1.f - tz) * (c & 2 ? ty : 1.f - ty) * (c & 1 ? tx : 1.f - tx);
                sx += wv * cs[v]; sy += wv * sn[v]; wk += wv;
            }
            if (wk < 0.5f || sx * sx + sy * sy < 1e-8f) { dst[x] = PHASE_UNKNOWN; continue; }
            float t = atan2f(sy, sx) * (float)(PHASE_PERIOD / (2 * M_PI)); if (t < 0) t += PHASE_PERIOD;
            const int q = (int)lrintf(t); dst[x] = (uint8_t)(q >= PHASE_PERIOD ? q - PHASE_PERIOD : q);
        }
    }
    tp_put(ph, N); free(j0); free(tf);
    return 0;
}
static int phase_window(z3 *a, const int64_t o[3], int D, int P, int shift, uint8_t *side) {   /* a: s->phase, or s->wind (same grid); D x P x P */
    if (shift >= 2) return phase_window_interp(a, o, D, P, shift, side);
    int64_t lo[3], ln[3], ro[3], rn[3]; const int wn[3] = {D, P, P};
    const z3_meta *m = z3_meta_of(a);
    for (int d = 0; d < 3; d++) {
        lo[d] = (o[d] + shift) >> shift; ln[d] = ((o[d] + wn[d] - 1 + shift) >> shift) - lo[d] + 1;
        ro[d] = lo[d] < 0 ? 0 : lo[d]; const int64_t e = lo[d] + ln[d] < m->shape[d] ? lo[d] + ln[d] : m->shape[d]; rn[d] = e - ro[d];
    }
    const size_t N = (size_t)ln[0] * ln[1] * ln[2];
    uint8_t *ph = tp_get(N, 0);
    if (!ph) return -1;
    memset(ph, PHASE_UNKNOWN, N);
    if (rn[0] > 0 && rn[1] > 0 && rn[2] > 0) {
        const size_t rnn = (size_t)rn[0] * rn[1] * rn[2];
        uint8_t *tmp = tp_get(rnn, 0);
        if (!tmp || z3_read(a, ro, rn, tmp, 1)) { tp_put(tmp, rnn); tp_put(ph, N); return -1; }
        for (int64_t z = 0; z < rn[0]; z++) for (int64_t y = 0; y < rn[1]; y++)
            memcpy(ph + ((size_t)(z + ro[0] - lo[0]) * ln[1] + (y + ro[1] - lo[1])) * ln[2] + (ro[2] - lo[2]), tmp + ((size_t)z * rn[1] + y) * rn[2], (size_t)rn[2]);
        tp_put(tmp, rnn);
    }
    for (int z = 0; z < D; z++) for (int y = 0; y < P; y++) {
        const uint8_t *row = ph + ((size_t)(((o[0] + z + shift) >> shift) - lo[0]) * ln[1] + (((o[1] + y + shift) >> shift) - lo[1])) * ln[2];
        uint8_t *dst = side + ((size_t)z * P + y) * P;
        for (int x = 0; x < P; x++) dst[x] = row[((o[2] + x + shift) >> shift) - lo[2]];
    }
    tp_put(ph, N);
    return 0;
}

/* the source's aux volume (src/sources.h) over a window of its level; outside the volume 0 */
static int aux_window(source *s, const int64_t o[3], int P, uint8_t *dst) {
    const z3_meta *m = z3_meta_of(s->aux); int64_t ro[3], rn[3];
    for (int d = 0; d < 3; d++) { ro[d] = o[d] < 0 ? 0 : o[d]; const int64_t e = o[d] + P < m->shape[d] ? o[d] + P : m->shape[d]; rn[d] = e - ro[d]; if (rn[d] <= 0) return 0; }
    const size_t rnn = (size_t)rn[0] * rn[1] * rn[2];
    uint8_t *tmp = tp_get(rnn, 0);
    if (!tmp || z3_read(s->aux, ro, rn, tmp, 1)) { tp_put(tmp, rnn); return -1; }
    for (int64_t z = 0; z < rn[0]; z++) for (int64_t y = 0; y < rn[1]; y++)
        memcpy(dst + ((size_t)(z + ro[0] - o[0]) * P + (y + ro[1] - o[1])) * P + (ro[2] - o[2]), tmp + ((size_t)z * rn[1] + y) * rn[2], (size_t)rn[2]);
    tp_put(tmp, rnn);
    return 0;
}

static double rnorm(rng *r) { double u1 = runif(r); const double u2 = runif(r); if (u1 < 1e-300) u1 = 1e-300; return sqrt(-2 * log(u1)) * cos(2 * M_PI * u2); }
static inline void put_x(float *X, uint16_t *H, int xfmt, size_t idx, float v) {
    if (X) X[idx] = v; else if (xfmt == 1) { _Float16 hv = (_Float16)v; memcpy(&H[idx], &hv, 2); } else H[idx] = bf16_rn(v);
}
/* --cond: the condition of one sample, drawn in the output frame (what predict --cond-sweep hands a window: the outputs of earlier
   windows where they exist). w: the sample's winding mod 14 after the symmetry (band_winding encoding, 255 unknown), shifted by a
   random gauge of k turns into wout (the targets). Pattern: none (p 0.1; wout all unknown, so the gauge-dependent heads are not
   supervised), a seed cube of 4..12 voxels (p 0.15), else slabs at the window faces (each axis with p 0.5, at least one; thickness
   U(P/8, P/2), the low face with p 0.75), then with p 0.3 one or two dropout cubes (4..P/4). Inputs 4-6: cos, sin of 2 pi q' / 14 with
   a smooth angle noise (a 16-voxel grid of N(0, 0.03 turn), trilinear) and 1, where the pattern covers a known voxel; 0 elsewhere. */
static void cond_draw(rng *r, int P, const uint8_t *w, uint8_t *wout, float *X, uint16_t *H, int xfmt, int outside) {
    const size_t p2 = (size_t)P * P, p3 = p2 * P;
    const int gauge = (int)rint_below(r, 14);
    uint8_t *cm = calloc(p3, 1); if (!cm) abort();
    const double u = runif(r); int any = u >= 0.10;
#define CUBE(v, s, c0, c1, c2) for (int z_ = (c0); z_ < (c0) + (s); z_++) for (int y_ = (c1); y_ < (c1) + (s); y_++) memset(cm + ((size_t)z_ * P + y_) * P + (c2), (v), (size_t)(s))
    if (any && u < 0.25) { const int s = 4 + (int)rint_below(r, 9), a = (int)rint_below(r, P - s + 1), b = (int)rint_below(r, P - s + 1), c = (int)rint_below(r, P - s + 1); CUBE(1, s, a, b, c); }
    else if (any) {
        int axes[3], na = 0;
        for (int d = 0; d < 3; d++) if (runif(r) < 0.5) axes[na++] = d;
        if (!na) axes[na++] = (int)rint_below(r, 3);
        for (int a = 0; a < na; a++) {
            const int d = axes[a], t = P / 8 + (int)rint_below(r, P / 2 - P / 8 + 1), high = runif(r) < 0.25, lo = high ? P - t : 0, hi = high ? P : t;
            for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) {
                uint8_t *row = cm + ((size_t)z * P + y) * P;
                if (d == 2) memset(row + lo, 1, (size_t)(hi - lo));
                else if ((d == 0 ? z : y) >= lo && (d == 0 ? z : y) < hi) memset(row, 1, (size_t)P);
            }
        }
    }
    if (any && runif(r) < 0.3) for (int nb = 1 + (int)rint_below(r, 2); nb > 0; nb--) {
        const int s = 4 + (int)rint_below(r, P / 4 - 3 > 1 ? P / 4 - 3 : 1), a = (int)rint_below(r, P - s + 1), b = (int)rint_below(r, P - s + 1), c = (int)rint_below(r, P - s + 1);
        CUBE(0, s, a, b, c);
    }
#undef CUBE
    for (size_t k = 0; k < p3; k++) wout[k] = !any || w[k] >= 252 ? 255 : (uint8_t)((w[k] + 18 * gauge) % 252);
    const int G = P / 16 + 2; float *nf = malloc((size_t)G * G * G * sizeof *nf); if (!nf) abort();
    for (size_t k = 0; k < (size_t)G * G * G; k++) nf[k] = (float)(0.03 * rnorm(r));
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
        const size_t k = ((size_t)z * P + y) * P + x;
        float cv = 0.f, sv = 0.f, mv = 0.f;
        if (cm[k] && wout[k] < 252) {
            const float fz = z / 16.f, fy = y / 16.f, fx = x / 16.f; const int iz = (int)fz, iy = (int)fy, ix = (int)fx;
            const float tz = fz - iz, ty = fy - iy, tx = fx - ix; float nz = 0.f;
            for (int c = 0; c < 8; c++) nz += ((c & 4) ? tz : 1 - tz) * ((c & 2) ? ty : 1 - ty) * ((c & 1) ? tx : 1 - tx) * nf[((size_t)(iz + (c >> 2 & 1)) * G + iy + (c >> 1 & 1)) * G + ix + (c & 1)];
            const double a = 2 * M_PI * (wout[k] / 18.0 + nz) / 14.0;
            cv = (float)cos(a); sv = (float)sin(a); mv = 1.f;
        }
        put_x(X, H, xfmt, 4 * p3 + k, cv); put_x(X, H, xfmt, 5 * p3 + k, sv); put_x(X, H, xfmt, 6 * p3 + k, mv);
    }
    if (outside) for (size_t k = 0; k < p3; k++) if (cm[k]) wout[k] = 255;   /* --cond-loss outside: supervise only where the net must count */
    free(nf); free(cm);
}

void sample_cond_draw(uint64_t seed, int P, const uint8_t *w, uint8_t *wout, float *X, int outside) { rng r; rseed(&r, seed); cond_draw(&r, P, w, wout, X, nullptr, 0, outside); }

/* a stored-phase window (source grid, PHASE codes) through the sample's continuous warp and cube symmetry, as write_spatial pulls the CT:
   each output voxel the phasor mean of its 8 source neighbours with a known phase (trilinear weights), unknown outside the window or
   where the known neighbours hold under half the weight. The normal / density targets follow from the warped phase on the GPU. */
static void warp_phase(const uint8_t *src, uint8_t *dst, int P, sym sy, const spatial_aug *a) {
    static float cs[PHASE_PERIOD], sn[PHASE_PERIOD]; static _Atomic int init;
    if (!atomic_load(&init)) { for (int k = 0; k < PHASE_PERIOD; k++) { cs[k] = (float)cos(2 * M_PI * k / PHASE_PERIOD); sn[k] = (float)sin(2 * M_PI * k / PHASE_PERIOD); } atomic_store(&init, 1); }
    float *table = malloc(6 * (size_t)P * sizeof *table); if (!table) abort();
    spatial_aug_tables(a, table);
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
        const size_t k = ((size_t)z * P + y) * P + x;
        float u[3], jac[3], v[3]; spatial_aug_pull(a, table, z, y, x, u, jac);
        for (int d = 0; d < 3; d++) v[sy.perm[d]] = sy.flip[d] ? P - 1 - u[d] : u[d];
        int inside = 1; for (int d = 0; d < 3; d++) if (v[d] < 0 || v[d] > P - 1) inside = 0;
        dst[k] = PHASE_UNKNOWN;
        if (!inside) continue;
        int base[3]; float f[3];
        for (int d = 0; d < 3; d++) { base[d] = (int)v[d]; if (base[d] >= P - 1) base[d] = P - 2; f[d] = v[d] - base[d]; }
        float sx = 0.f, s_y = 0.f, wk = 0.f;
        for (int c = 0; c < 8; c++) {
            const int bz = c >> 2 & 1, by = c >> 1 & 1, bx = c & 1;
            const float w = (bz ? f[0] : 1 - f[0]) * (by ? f[1] : 1 - f[1]) * (bx ? f[2] : 1 - f[2]);
            const int ph = src[((size_t)(base[0] + bz) * P + base[1] + by) * P + base[2] + bx];
            if (w <= 1e-7f || ph >= PHASE_PERIOD) continue;
            sx += w * cs[ph]; s_y += w * sn[ph]; wk += w;
        }
        if (wk < .5f || sx * sx + s_y * s_y < 1e-8f) continue;
        float t = atan2f(s_y, sx) * (float)(PHASE_PERIOD / (2 * M_PI)); if (t < 0) t += PHASE_PERIOD;
        const int q = (int)lrintf(t); dst[k] = (uint8_t)(q >= PHASE_PERIOD ? q - PHASE_PERIOD : q);
    }
    free(table);
}

static int draw(sampler *sp, batch *b, int i, rng *r, float *xtmp, uint8_t *ttmp, uint8_t *big, uint64_t tile_index, float *noise_state) {
    double pt = sp->prof ? tnow() : 0;
    const sample_cfg *c = &sp->cfg;
    const int P = c->P, Pz = PZ(c);   /* window Pz x P x P (Pz < P: a slab) */
    const size_t p3 = P3(sp);
    /* source by weight */
    double u = runif(r) * sp->cum[sp->S->n - 1];
    int si = 0;
    if (c->cover) si = (int)c->cover->tiles[tile_index][0];
    else while (si < sp->S->n - 1 && sp->cum[si] < u) si++;
    source *s = &sp->S->src[si];
    if (c->holdout && !s->hold_n[0]) return 1;
    /* regions target? pick one region, then a level in {0,1} */
    int region_ch = -1, ri = -1;
    for (int ch = 0; ch < NCH; ch++) if (s->reg[ch] && s->reg[ch]->n) region_ch = ch;
    PROF_MARK(PS_SRC);
    int l = c->cover ? 0 : pick_level(sp, s, r, region_ch);
    if (l < 0) return 1;
    PROF_MARK(PS_LEVEL);
    z3 *ct = source_ct(s, l);
    const z3_meta *m = z3_meta_of(ct);
    int64_t o[3], n[3] = {Pz, P, P};
    if (c->cover) memcpy(o, c->cover->tiles[tile_index] + 1, sizeof o);
    else if (region_ch >= 0) {
        regions *R = s->reg[region_ch];
        ri = (int)rint_below(r, R->n);
        int64_t lo[3], hi[3]; if (!patch_bounds(s, m, n, l, R, ri, c->holdout, lo, hi)) return 1;
        for (int d = 0; d < 3; d++) o[d] = lo[d] + rint_below(r, hi[d] - lo[d] + 1);
    } else if (c->holdout) {
        int64_t lo[3], hi[3]; if (!patch_bounds(s, m, n, l, nullptr, 0, 1, lo, hi)) return 1;
        for (int d = 0; d < 3; d++) o[d] = lo[d] + rint_below(r, hi[d] - lo[d] + 1);
    } else if (sp->occ[si].n) {   /* draw around a random coarse cell that contains papyrus */
        size_t k = (size_t)(runif(r) * (double)sp->occ[si].n); if (k >= sp->occ[si].n) k = sp->occ[si].n - 1;
        uint32_t id = sp->occ[si].idx[k];
        int64_t cz = id / (sp->occ[si].shape[1] * sp->occ[si].shape[2]), cy = (id / sp->occ[si].shape[2]) % sp->occ[si].shape[1], cx = id % sp->occ[si].shape[2];
        int64_t cc[3] = {cz, cy, cx};
        int dl = sp->occ[si].lev - l;    /* coarse cell -> level-l voxels */
        for (int d = 0; d < 3; d++) {
            if (m->shape[d] < n[d]) return 1;
            int64_t f = dl >= 0 ? (int64_t)1 << dl : 1;
            int64_t v = dl >= 0 ? cc[d] * f + rint_below(r, f) : cc[d] >> (-dl);
            o[d] = v - n[d] / 2; if (o[d] < 0) o[d] = 0; if (o[d] > m->shape[d] - n[d]) o[d] = m->shape[d] - n[d];
        }
    } else {
        for (int d = 0; d < 3; d++) { if (m->shape[d] < n[d]) return 1; o[d] = rint_below(r, m->shape[d] - n[d] + 1); }
    }
    if (!c->cover && c->snap && !c->holdout && region_ch < 0) {   /* training only: snapping a validation origin can leave its holdout */
        for (int d = 0; d < 3; d++) {
            int64_t cs = m->chunk[d]; if (cs <= 0) continue;
            o[d] -= o[d] % cs;
            if (o[d] > m->shape[d] - n[d]) o[d] = m->shape[d] - n[d];   /* the last window of an axis may stay unaligned */
            if (o[d] < 0) o[d] = 0;
        }
    }
    if (!c->holdout && training_hits_holdout(sp, s, o, l)) return 1;
    PROF_MARK(PS_PICK);
    /* cheap occupancy test on a coarse level before fetching the fine cube */
    int lc = l + 3;
    z3 *cct = lc < MAXLEV ? source_ct(s, lc) : nullptr;
    if (cct && !c->cover && Pz >= 8) {
        int pc = P >> 3;
        int64_t oc[3] = {o[0] >> 3, o[1] >> 3, o[2] >> 3}, nc[3] = {Pz >> 3, pc, pc};
        if (z3_read(cct, oc, nc, big, 1)) return -1;
        size_t nz = 0, tot = (size_t)nc[0] * pc * pc;
        for (size_t k = 0; k < tot; k++) nz += big[k] != 0;
        if ((double)nz / (double)tot < c->min_fg) { atomic_fetch_add(&sp->rejected, 1); return 1; }
    }
    PROF_MARK(PS_PROBE);
    /* CT */
    uint8_t *ctu = big;
    if (z3_read(ct, o, n, ctu, 1)) return -1;
    PROF_MARK(PS_READ);
    size_t nz = 0;
    double sum = 0, sq = 0;
    u8_stats(ctu, p3, &nz, &sum, &sq);
    if (!c->cover && (double)nz / (double)p3 < c->min_fg) { atomic_fetch_add(&sp->rejected, 1); return 1; }
    PROF_MARK(PS_STATS);
    /* targets */
    uint8_t w[NCH] = {0};
    int tmax = 0;
    const int trust_r = s->trust_band > 0 ? (s->trust_band >> l < 1 ? 1 : s->trust_band >> l) : 0;
    int iso_t = 0, iso_r = trust_r;   /* --iso-aug: this sample's papyrus cutoff (0 = off) and lifted-label radius */
    if (c->augment && c->iso_hi > 0 && runif(r) < c->iso_p) {
        iso_t = c->iso_lo + (int)rint_below(r, c->iso_hi - c->iso_lo + 1);
        if (c->iso_rhi > 0) iso_r = c->iso_rlo + (int)rint_below(r, c->iso_rhi - c->iso_rlo + 1);
    }
    uint8_t *fg = nullptr;   /* the foreground for iso_t (iso_foreground), radius iso_r */
    for (int ch = 0; ch < NCH; ch++) {
        uint8_t *dst = ttmp + (size_t)ch * p3;
        if (c->papyrus) { memset(dst, 0, p3); continue; }   /* task papyrus: labels from the CT below */
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
            z3 *tz = source_tgt_for_level(s, ch, l, nullptr);
            if (!tz) { if (c->cover) return -1; memset(dst, 0, p3); continue; }
            if (c->erode && !s->tgt_prob[ch] && l == 0) {
                int64_t eo[3] = {o[0]-1, o[1]-1, o[2]-1}, en[3] = {P+2, P+2, P+2};
                uint8_t *haloed = big + p3;   /* existing (2P)^3 scratch, P >= 2 */
                if (source_read_target(s, ch, l, eo, en, haloed, 1)) return -1;
                target_erode1(haloed, dst, P);
            } else if (source_read_target(s, ch, l, o, n, dst, 1)) return -1;
            w[ch] = 1;
        } else memset(dst, 0, p3);
        if (iso_t > 0 && w[ch] && !s->tgt_prob[ch]) {   /* lifted labels (farther than iso_r from CT >= iso_t) ignored */
            if (!fg && !(fg = iso_foreground(ctu, Pz, P, iso_t, iso_r))) return -1;
            for (size_t k = 0; k < p3; k++) if (dst[k] && dst[k] != 255 && !fg[k]) dst[k] = 255;   /* ignored, not negative: the sheet may be real but faint */
        }
        if (w[ch] && !c->cover) {   /* largest known target value (only the empty-patch rejection reads it) */
            uint8_t mx = 0;
            for (size_t k = 0; k < p3; k++) { const uint8_t v = dst[k] == 255 ? 0 : dst[k]; mx = v > mx ? v : mx; }
            if (mx > tmax) tmax = mx;
        }
    }
    tp_put(fg, p3);
    if (!c->cover && !c->papyrus && tmax == 0 && runif(r) > c->empty_keep) { atomic_fetch_add(&sp->rejected, 1); return 1; }
    PROF_MARK(PS_TARGETS);
    uint8_t *near = nullptr;   /* --phase-near-um: voxels within the distance of a labelled wrap voxel (raw labels, before softening) */
    if (c->phase_near_um > 0 && c->side == 2 && w[0]) {
        const float rn = c->phase_near_um / (float)(s->um * (double)(1 << l)); float *d2 = tp_get(p3 * sizeof *d2, 0); uint8_t *feat = tp_get(p3, 0); near = tp_get(p3, 0);
        const int64_t en[3] = {Pz, P, P};
        if (!d2 || !feat || !near) { tp_put(d2, p3 * sizeof *d2); tp_put(feat, p3); tp_put(near, p3); return -1; }
        for (size_t k = 0; k < p3; k++) feat[k] = ttmp[k] == 254;
        if (edt_sq(feat, en, d2, 1)) { tp_put(d2, p3 * sizeof *d2); tp_put(feat, p3); tp_put(near, p3); return -1; }
        for (size_t k = 0; k < p3; k++) near[k] = d2[k] <= rn * rn;
        tp_put(d2, p3 * sizeof *d2); tp_put(feat, p3);
    }
    /* curriculum: thicken the surface band by max-filtering the target (hard labels only, pyramid sources) */
    if (c->dilate > 0 && region_ch < 0) {
        int D = c->dilate >> l; if (D < 1) D = 1;
        uint8_t *tmp2 = big + 2 * p3;
        for (int ch = 0; ch < NCH; ch++) {
            if (!w[ch] || s->tgt_prob[ch]) continue;
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
    PROF_MARK(PS_DILATE);
    /* soft ridge target: background voxels near the surface get 254 * exp(-(d / sigma)^2 / 2), d = 3-4-5 chamfer distance / 3 */
    float soft0; { unsigned sb = atomic_load(&sp->soft_bits); memcpy(&soft0, &sb, 4); }
    float morph=0;
    if (c->augment && c->label_morph>0 && runif(r)<c->label_morph_p) morph=(float)((2*runif(r)-1)*c->label_morph);
    uint16_t *dm = (uint16_t *)(big + 2 * p3);   /* chamfer distance to the annotated surface, per channel (3 per voxel); big has room */
    uint8_t *dm8 = big + 2 * p3;                  /* the same saturated at 255, when every threshold below is under 255 */
    int dm_ch = -1, dm_any = 0;                   /* channel the map currently holds, and whether that channel had any surface */
    const int soft_on = soft0 > 0 && region_ch < 0;
    float sigma = 0, shift = 0; int dcut = 0;
    if (soft_on) {
        sigma = (c->soft_um ? soft0 / (float)s->um : soft0) / (float)(1 << l); if (sigma < 0.75f) sigma = 0.75f;
        shift=morph/(float)(1<<l);
        /* exp(-(d/3)^2 / (2 sigma^2)) tabulated over the chamfer distance (3 per voxel); beyond dcut it rounds to 0 */
        dcut = (int)(3.f * (sigma * 4.5f+fmaxf(0,shift))) + 1; if (dcut > 4000) dcut = 4000;
    }
    const int use8 = (!soft_on || dcut <= 254) && 3 * trust_r <= 254;
#define CHAMFER_FOR(ch_) do { if (dm_ch != (ch_)) { dm_ch = (ch_); dm_any = use8 ? chamfer345_8(ttmp + (size_t)(ch_) * p3, dm8, Pz, P) : chamfer345(ttmp + (size_t)(ch_) * p3, dm, Pz, P); } } while (0)
    if (soft_on) {
        uint8_t *etab = big + 4 * p3;
        for (int d = 0; d <= dcut; d++) { float df = fmaxf(0,d / 3.f-shift); int t = (int)(254.f * expf(-0.5f * df * df / (sigma * sigma)) + 0.5f); etab[d] = (uint8_t)(t > 254 ? 254 : t); }
        if (use8) for (int d = dcut + 1; d < 256; d++) etab[d] = 0;   /* indexed by the saturated distance */
        for (int ch = 0; ch < NCH; ch++) {
            if (!w[ch] || s->tgt_prob[ch]) continue;
            uint8_t *dst = ttmp + (size_t)ch * p3;
            CHAMFER_FOR(ch);
            if (!dm_any) continue;   /* no surface in the patch: nothing to soften */
            if (use8) for (size_t k = 0; k < p3; k++) dst[k] = dst[k] == 0 ? etab[dm8[k]] : dst[k];
            else for (size_t k = 0; k < p3; k++) if (dst[k] == 0 && dm[k] <= dcut) { uint8_t t = etab[dm[k]]; if (t) dst[k] = t; }
        }
    }
    PROF_MARK(PS_SOFT);
    /* partial annotations: trust background only within R voxels of an annotated surface, ignore the rest */
    if (trust_r > 0) {
        const int R = trust_r;
        for (int ch = 0; ch < NCH; ch++) {
            if (!w[ch] || s->tgt_prob[ch]) continue;
            uint8_t *dst = ttmp + (size_t)ch * p3;
            CHAMFER_FOR(ch);   /* measured from the hard surface (the soft values added above are not surface) */
            if (!dm_any) { for (size_t k = 0; k < p3; k++) if (dst[k] == 0) dst[k] = 255; continue; }   /* nothing annotated: ignore all background */
            const int tr = 3 * R;
            if (use8) for (size_t k = 0; k < p3; k++) dst[k] = dst[k] == 0 && dm8[k] > tr ? 255 : dst[k];
            else for (size_t k = 0; k < p3; k++) dst[k] = dst[k] == 0 && dm[k] > tr ? 255 : dst[k];
        }
    }
    /* label encoding -> target probability * 255 and the ignore mask (255 = ignore) */
    PROF_MARK(PS_TRUST);
    uint8_t *ign = big + 9 * p3;
    memset(ign, 0, p3);
    for (int ch = 0; ch < NCH; ch++) {
        if (!w[ch]) continue;
        uint8_t *dst = ttmp + (size_t)ch * p3;
        for (size_t k = 0; k < p3; k++) { const uint8_t v = dst[k]; ign[k] |= v == 255; dst[k] = v == 255 ? 0 : (uint8_t)(v + (v >= 127)); }   /* (v * 255 + 127) / 254 for v < 255 */
    }
    float pap_t = 0;
    if (c->papyrus) {   /* task papyrus: channel 0 (probability x 255) and its ignore band from the clean CT, before the scanner augmentation */
        const double u_mode = runif(r), u_t = runif(r), u_tau = runif(r);
        pap_t = papyrus_cutoff(c->papyrus, ctu, p3, u_mode, u_t);
        papyrus_labels(ctu, P, c->papyrus, pap_t, u_tau, ttmp, ign, ttmp + p3, big + p3);   /* channel 1 as scratch, scratch 8 P^3 */
        memset(ttmp + p3, 0, p3);
        w[0] = 1;
    }
    uint8_t *papc = nullptr;   /* --pap-head: the gap labels of the clean CT, 0 void / 1 papyrus / 255 ignored (the symmetry below, with pm) */
    if (c->pap_head) {
        uint8_t *pt = tp_get(p3, 0), *pg = tp_get(p3, 1), *pf = tp_get(p3, 0); papc = tp_get(p3, 0);
        if (!pt || !pg || !pf || !papc) { tp_put(pt, p3); tp_put(pg, p3); tp_put(pf, p3); tp_put(papc, p3); return -1; }
        papyrus_labels(ctu, P, c->pap_head, 0.f, 0.0, pt, pg, pf, big + p3);   /* (scratch 8 P^3: free until the scanner augmentation) */
        for (size_t k = 0; k < p3; k++) papc[k] = pg[k] ? 255 : pt[k] ? 1 : 0;
        tp_put(pt, p3); tp_put(pg, p3); tp_put(pf, p3);
    }
    PROF_MARK(PS_ENCODE);
    if (c->sheet) sheet_mask_contacts(c->sheet,o,P,ign);
    /* input channels (z-scored CT, radial unit vector) written in one pass straight into the batch with the symmetry
       and the jitter applied on the way (16-bit when cfg.xfmt, else fp32): no fp32 intermediates */
    double mean = sum / (double)p3, var = sq / (double)p3 - mean * mean, sd = sqrt(var > 0 ? var : 0) + 1e-3;
    double scale = (double)(1 << l);
    float *cyz = xtmp, *cxz = xtmp + P, *nrow = xtmp + 2 * P;
    for (int z = 0; z < Pz; z++) { double cy, cx; axis_at(&s->ax, (double)(o[0] + z) * scale, &cy, &cx); cyz[z] = (float)(cy / scale); cxz[z] = (float)(cx / scale); }
    if (c->augment && scan_aug_enabled(&c->scan)) {   /* scanner domains on the source grid; the per-patch z-score follows them, as at inference */
        PROF_MARK(PS_ZSCORE);
        if (scan_aug_apply(&c->scan, rnext(r), ctu, ttmp, NCH, ign, P, o, cyz, cxz, s->ax.n > 0, m->chunk, !c->sheet && !c->band && !c->side, big + p3)) {
            sum = sq = 0;
            for (size_t k = 0; k < p3; k++) { sum += ctu[k]; sq += (double)ctu[k] * ctu[k]; }
            mean = sum / (double)p3; var = sq / (double)p3 - mean * mean; sd = sqrt(var > 0 ? var : 0) + 1e-3;
        }
        PROF_MARK(PS_SCAN);
    }
    uint8_t *mask = big + 9 * p3;   /* ign lives here: fold CT > 0 into it */
    for (size_t k = 0; k < p3; k++) mask[k] = (ctu[k] != 0) & (mask[k] == 0);
    PROF_MARK(PS_ZSCORE);
    uint8_t *T = b->t + (size_t)i * NCH * p3, *M = b->m + (size_t)i * p3;
    sym y = sym_of(0);
    float ia = 1.f, ib = 0.f, isg = 0.f;
    if (c->augment) {
        y = sym_of((int)rint_below(r, 48));
        if (c->augment == 4) y = sym_of(0);   /* intensity jitter only */
        if (c->augment == 2) while (!sym_is_rotation(y)) y = sym_of((int)rint_below(r, 48));   /* rotations only */
        if (c->augment == 3) { y = sym_of((int)rint_below(r, 48)); while (y.perm[0] != 0 || y.flip[0]) y = sym_of((int)rint_below(r, 48)); }   /* z fixed: y/x swaps and flips only */
        if (Pz != P && y.perm[0] != 0) return -1;   /* (sampler_start allows slabs only with augment 0, 3, 4) */
        double a = exp((runif(r) * 2 - 1) * 0.22), bb = (runif(r) * 2 - 1) * 0.2, sg = runif(r) * 0.1;
        ia = (float)a; ib = (float)bb; isg = (float)sg;
    }
    if (c->augment && c->symmetry_p < 1 && runif(r) >= c->symmetry_p) y = sym_of(0);
    if (c->augment && c->axis_jitter > 0 && s->ax.n > 0 && runif(r) < (c->axis_jitter_p > 0 ? c->axis_jitter_p : .1)) {
        float jy = (float)((2 * runif(r) - 1) * c->axis_jitter / scale);
        float jx = (float)((2 * runif(r) - 1) * c->axis_jitter / scale);
        /* Bound the auxiliary radial-direction perturbation throughout the patch. CT/targets remain registered. */
        float radius = INFINITY;
        for (int z = 0; z < Pz; z++) {
            float dy = fmaxf(0, fmaxf((float)o[1] - cyz[z], cyz[z] - (float)(o[1] + P - 1)));
            float dx = fmaxf(0, fmaxf((float)o[2] - cxz[z], cxz[z] - (float)(o[2] + P - 1)));
            float rr = sqrtf(dy * dy + dx * dx); if (rr < radius) radius = rr;
        }
        float mag = sqrtf(jy * jy + jx * jx), cap = radius * (c->axis_jitter_deg > 0 ? sinf(c->axis_jitter_deg * 0.0174532925f) : .0348995f);
        if (mag > cap && mag > 0) { jy *= cap / mag; jx *= cap / mag; }
        for (int z = 0; z < Pz; z++) { cyz[z] += jy; cxz[z] += jx; }
    }
    ct_aug_plan ap; const ct_aug_plan *app = nullptr;
    if (c->augment && c->ct_augment) {
        ct_aug_make(&ap, rnext(r), 1, mean, sd, P); app = &ap;
        if (ap.noise_rho > 0) isg = ap.noise_sigma;
    }
    spatial_aug spatial={0};
    /* band-labelled native samples keep the exact voxel grid (the band field is not interpolated) */
    const int pshift = s->phase ? s->phase_level - l : -1;   /* the stored phase: its grid relative to this level's (0 direct, 1 upsampled) */
    const int stored_phase = !c->band && c->side == 2 && s->phase && !c->phase_dense && (pshift == 1 || ((pshift == 0 || pshift == 2) && s->phase_direct));   /* (2: a 9.6 um phase interpolated into 2.4 um windows) */
    const int band_sample = (c->band || c->side) && s->band && (l == 0 || stored_phase);   /* (c->side without c->band: task unwrap, the phase only) */
    const int warp_ok = !band_sample || (stored_phase && !c->band && !c->cond);   /* a stored phase warps (warp_phase); band fields do not */
    if (c->augment && c->geometry_augment && s->geometry && warp_ok) {
        spatial_aug_make(&spatial,rnext(r),P,c->rotate_degrees,c->rotate_p,c->elastic,c->elastic_p);
        if (c->zoom_p > 0 && c->zoom_min > 0 && c->zoom_min < 1 && runif(r) < c->zoom_p) spatial_aug_zoom(&spatial, exp(runif(r) * log(c->zoom_min)));
        if (c->affine_p > 0 && c->scan.strength > 0 && runif(r) < c->affine_p)
            spatial_aug_affine(&spatial, rnext(r), c->affine_aniso * c->scan.strength, c->affine_shear * c->scan.strength);
    }
    b->um[i] = (float)(s->um * (double)(1 << l) * (spatial.active ? spatial.scale * cbrt(spatial.det) : 1.0));   /* affine: the volume-preserving voxel size */
    if (c->papyrus) b->um[i] = c->papyrus->cond == 1 ? (float)(ia * ((pap_t - mean) / sd) + ib) : 0.f;   /* the cutoff where the input puts it */
    if (c->papyrus && c->papyrus->cond == 2) { mean = pap_t; ib = 0.f; }   /* the CT channel (ct - t) / sd: the cutoff at zero */
    if (!spatial.active) {
        for (int ch = 0; ch < NCH; ch++) sym_u8(ttmp + (size_t)ch * p3, T + (size_t)ch * p3, Pz, P, y);
        sym_u8(mask, M, Pz, P, y);
    }
    if (c->pmask && !spatial.active) {   /* CT > 0 through the same symmetry (big + 2 P^3, the soft / trust distance map, is free here) */
        uint8_t *pt = big + 2 * p3;
        for (size_t k = 0; k < p3; k++) pt[k] = ctu[k] != 0;
        sym_u8(pt, b->pm + (size_t)i * p3, Pz, P, y);
    }
    if (c->pap_head) {   /* the gap labels through the same symmetry; warped samples: all ignored */
        if (spatial.active) memset(b->pap + (size_t)i * p3, 255, p3); else sym_u8(papc, b->pap + (size_t)i * p3, Pz, P, y);
        tp_put(papc, p3);
    }
    if (c->band || c->side) {   /* band labels: computed on the source grid, unknown on CT air, then the same symmetry as CT and targets */
        if (!band_sample || (spatial.active && !stored_phase)) {   /* no band labels: affinities (and sides) unsupervised */
            if (c->band) memset(b->band + (size_t)i * p3, BAND_UNKNOWN, p3);
            if (c->side) memset(b->side + (size_t)i * p3, SIDE_UNKNOWN, p3);
        } else if (stored_phase) {   /* the stored phase (ufsm phase-raster, ufsm downsample --mode phase) */
            uint8_t *sw = tp_get(p3, 0);
            if (!sw || phase_window(s->phase, o, Pz, P, pshift, sw)) { tp_put(sw, p3); return -1; }
            for (size_t k = 0; k < p3; k++) if (ctu[k] <= c->side_air || !ctu[k]) sw[k] = SIDE_UNKNOWN;
            if (c->phase_near_um > 0) for (size_t k = 0; k < p3; k++) if (!near || !near[k]) sw[k] = SIDE_UNKNOWN;   /* no labelled wrap: none */
            if (spatial.active) warp_phase(sw, b->side + (size_t)i * p3, P, y, &spatial);   /* (slabs never warp: sampler_start) */
            else sym_u8(sw, b->side + (size_t)i * p3, Pz, P, y);
            tp_put(sw, p3);
        } else {
            if (Pz != P) { fprintf(stderr, "sampler: slab windows take the phase from a stored raster (sources \"phase\" at this level or one coarser)\n"); return -1; }
            uint8_t *bw = tp_get(p3, 0), *sw = c->side ? tp_get(p3, 0) : nullptr;
            if (!bw || (c->side && !sw) || band_window(s, o, P, bw, sw, c->side == 2, c->phase_dense, tile_index ^ c->seed)) { tp_put(bw, p3); tp_put(sw, p3); return -1; }
            for (size_t k = 0; k < p3; k++) if (!ctu[k]) bw[k] = BAND_UNKNOWN;
            if (c->band) sym_u8(bw, b->band + (size_t)i * p3, Pz, P, y);
            if (sw) {   /* sides (phases) above CT side_air only: air (CT <= side_air) stays unsupervised */
                for (size_t k = 0; k < p3; k++) if (ctu[k] <= c->side_air || !ctu[k]) sw[k] = SIDE_UNKNOWN;
                sym_u8(sw, b->side + (size_t)i * p3, Pz, P, y);
            }
            tp_put(bw, p3); tp_put(sw, p3);
        }
    }
    PROF_MARK(PS_AUGMENT);
    const size_t xc = 4 + (c->aux != 0);
    float *X = c->xfmt ? nullptr : b->x + (size_t)i * xc * p3;
    uint16_t *H = c->xfmt ? b->x16 + (size_t)i * xc * p3 : nullptr;
    double (*params)[4]=nullptr;
    if (c->sheet) {
        params=malloc((size_t)P*sizeof *params);
        for (int sz=0;sz<P;sz++) sheet_parameters(c->sheet,o[0]+sz,params[sz]);
    }
    if (spatial.active) write_spatial(ctu,ttmp,mask,T,M,P,o,(float)mean,(float)(1.0/sd),s->ax.n>0,cyz,cxz,y,&spatial,
            ia,ib,isg,r,nrow,c->xfmt,X,H,app,noise_state,noise_state?noise_state+P:nullptr,c->sheet,params,(float *)(big+5*p3));
    else if (c->sheet || (app && Pz != P)) write_x_aug(ctu, P, o, (float)mean, (float)(1.0 / sd), s->ax.n > 0, cyz, cxz, y, c->augment != 0, ia, ib, isg, r, nrow, c->xfmt,
            X, H, app, noise_state, noise_state ? noise_state + P : nullptr,c->sheet,params);
    else write_x(ctu, Pz, P, o, (float)mean, (float)(1.0 / sd), s->ax.n > 0, cyz, cxz, y, c->augment != 0, ia, ib, isg, r, nrow, c->xfmt, X, H,
                 app, noise_state, noise_state ? noise_state + P : nullptr);   /* the CT appearance plan in the row-vectorised writer (cubes) */
    free(params);
    if (c->aux) {   /* the fifth channel: the source's aux volume at this level (zeros elsewhere), the CT's symmetry, v / 127.5 - 1 */
        uint8_t *aw = tp_get(p3, 1), *as = tp_get(p3, 0);
        if (!aw || !as || (s->aux && s->aux_level == l && aux_window(s, o, P, aw))) { tp_put(aw, p3); tp_put(as, p3); return -1; }
        sym_u8(aw, as, Pz, P, y);
        for (size_t k = 0; k < p3; k++) {
            const float v = as[k] / 127.5f - 1.f;
            if (X) X[4 * p3 + k] = v; else if (c->xfmt == 1) { _Float16 hv = (_Float16)v; memcpy(&H[4 * p3 + k], &hv, 2); } else H[4 * p3 + k] = bf16_rn(v);
        }
        tp_put(aw, p3); tp_put(as, p3);
    }
    tp_put(near, p3);
    if (c->cond) {   /* inputs 4-6 and batch.wind from the source's winding raster (on the stored phase's grid; air unknown like the phase) */
        uint8_t *ww = tp_get(p3, 0), *ws = tp_get(p3, 0);
        if (!ww || !ws) { tp_put(ww, p3); tp_put(ws, p3); return -1; }
        if (stored_phase && s->wind) {
            if (phase_window(s->wind, o, Pz, P, pshift, ww)) { tp_put(ww, p3); tp_put(ws, p3); return -1; }
            for (size_t k = 0; k < p3; k++) if (ctu[k] <= c->side_air || !ctu[k]) ww[k] = 255;
        } else memset(ww, 255, p3);
        sym_u8(ww, ws, Pz, P, y);
        cond_draw(r, P, ws, b->wind + (size_t)i * p3, X, H, c->xfmt, c->cond_outside);
        tp_put(ww, p3); tp_put(ws, p3);
    }
    if (c->sheet) {
        sheet_batch_free(b->sheet[i]);
        b->sheet[i] = sheet_sample(c->sheet, o, P, y.perm, y.flip, ctu,ttmp,mask,rnext(r));
        if (!b->sheet[i]) return -1;
        if (spatial.active) warp_sheet_batch(b->sheet[i],&spatial,M);
        w[1]=0; /* winding is supervised by the sparse regression objective */
    }
    if (spatial.active && c->pmask) memcpy(b->pm + (size_t)i * p3, M, p3);   /* warped samples: the warped loss mask */
    PROF_MARK(PS_X16);
    memcpy(b->w + (size_t)i * NCH, w, NCH);
    b->src[i] = (int16_t)si;
    b->level[i] = (int8_t)l;
    b->iso[i][0] = (uint8_t)iso_t; b->iso[i][1] = (uint8_t)(iso_t ? (iso_r > 255 ? 255 : iso_r) : 0);
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
    float *xtmp = malloc(3 * (size_t)sp->cfg.P * sizeof(float));   /* per-z axis centres and a row of noise */
    uint8_t *ttmp = malloc(NCH * p3);
    uint8_t *big = malloc(10 * p3);   /* [0,P^3) CT (also the coarse probe), [P^3, 9P^3) (2P)^3 region scratch, [9P^3, 10P^3) mask */
    const int det = sp->cfg.deterministic;
    float *noise_state = sp->cfg.ct_augment ? malloc(((size_t)sp->cfg.P * sp->cfg.P + sp->cfg.P) * sizeof(float)) : nullptr;
    if (!xtmp || !ttmp || !big || (sp->cfg.ct_augment && !noise_state)) {
        atomic_store(&sp->failed, 1); atomic_store(&sp->stop, 1);
        pthread_mutex_lock(&sp->mu); pthread_cond_broadcast(&sp->cv_ready); pthread_cond_broadcast(&sp->cv_free); pthread_mutex_unlock(&sp->mu);
    }
    while (!atomic_load(&sp->stop)) {
        pthread_mutex_lock(&sp->mu);
        int k = -1;
        int64_t jb = -1;
        if (det) {   /* claim the next batch index; its slot is fixed (jb % nslots) and must be free */
            if (sp->cfg.cover && (uint64_t)sp->next_claim >= sp->cfg.cover->count - sp->cfg.cover_start) {
                pthread_mutex_unlock(&sp->mu); break;
            }
            jb = sp->next_claim++;
            k = (int)(jb % sp->nslots);
            while ((sp->state[k] != FREE || jb - sp->next_take >= sp->nslots) && !atomic_load(&sp->stop)) pthread_cond_wait(&sp->cv_free, &sp->mu);
            if (atomic_load(&sp->stop)) k = -1;
            else sp->slot_j[k] = jb;
        } else
        while (k < 0 && !atomic_load(&sp->stop)) {
            for (int j = 0; j < sp->nslots; j++) if (sp->state[j] == FREE) { k = j; break; }
            if (k < 0) pthread_cond_wait(&sp->cv_free, &sp->mu);
        }
        if (k >= 0) sp->state[k] = FILLING;
        pthread_mutex_unlock(&sp->mu);
        if (k < 0) break;
        batch *b = &sp->slots[k];
        int fail = 0, filled = 0; unsigned spin = 0;
        for (int i = 0; i < sp->cfg.B && !atomic_load(&sp->stop);) {
            if (det && (sp->cfg.cover || (spin == 0 && fail == 0))) rseed(&r, sp->cfg.seed * 0x9e3779b97f4a7c15ull + (sp->cfg.cover_start + (uint64_t)(jb * sp->cfg.B + i)) * 1000003ull + 1);   /* per-sample stream */
            int rc = draw(sp, b, i, &r, xtmp, ttmp, big, sp->cfg.cover_start + (uint64_t)(jb * sp->cfg.B + i), noise_state);
            if (rc == 0) { i++; filled = i; fail = 0; spin = 0; }
            else if (rc > 0 && ++spin == (1u << 22)) { fprintf(stderr, "sampler: %u consecutive draws rejected or impossible (P=%d), stopping\n", spin, sp->cfg.P); atomic_store(&sp->failed, 1); atomic_store(&sp->stop, 1); }
            else if (rc < 0) { fprintf(stderr, "sampler: %s\n", z3_error()); if (++fail > 20) { fprintf(stderr, "sampler: 20 consecutive read failures, stopping\n"); atomic_store(&sp->failed, 1); atomic_store(&sp->stop, 1); } }
        }
        pthread_mutex_lock(&sp->mu);
        sp->state[k] = filled == sp->cfg.B && !atomic_load(&sp->stop) ? READY : FREE;
        pthread_cond_broadcast(&sp->cv_ready);
        pthread_cond_broadcast(&sp->cv_free);
        pthread_mutex_unlock(&sp->mu);
    }
    free(xtmp); free(ttmp); free(big); free(noise_state); tp_drain();
    free(wa);
    return nullptr;
}

/* coarsest recto level that fits in memory: the cells containing any surface fraction (fallback: any labelled cell) */
occ_index source_occupancy(source *s) {
    occ_index o = {nullptr, 0, -1, {0, 0, 0}};
    if (s->reg[0] && s->reg[0]->n) return o;
    /* the coarsest label level that exists (level >= 3); large scans (Paris 4: 2.5e9 cells at level 5) are read in z slabs,
       so there is no size cap besides the 32-bit cell index (a 3e8 cap used to leave such sources without an index, i.e.
       sampled uniformly over the whole volume) */
    int lc = -1; z3 *tz = nullptr;
    for (int l = MAXLEV - 1; l >= 3; l--) { z3 *t = source_tgt(s, 0, l); if (t) { const z3_meta *m = z3_meta_of(t); if ((double)m->shape[0] * m->shape[1] * m->shape[2] < 4.29e9) { lc = l; tz = t; break; } } }
    if (!tz) return o;
    const z3_meta *m = z3_meta_of(tz);
    const int binary = s->tgt_binary[0] || m->label_binary;
    const size_t plane = (size_t)m->shape[1] * m->shape[2], tot = plane * (size_t)m->shape[0];
    const int64_t SZ = m->chunk[0] > 0 ? m->chunk[0] : 64;
    uint8_t *buf = malloc(plane * (size_t)SZ);
    if (!buf) return o;
    size_t cap = 1 << 20, n = 0; uint32_t *idx = malloc(cap * sizeof *idx);
    int surf = 1;
    for (int pass = 0; pass < (binary ? 1 : 2) && !n; pass++) {   /* binary: 255 is a surface, never ignore */
        surf = pass == 0;
        for (int64_t z0 = 0; z0 < m->shape[0]; z0 += SZ) {
            int64_t o0[3] = {z0, 0, 0}, nn[3] = {SZ < m->shape[0] - z0 ? SZ : m->shape[0] - z0, m->shape[1], m->shape[2]};
            if (z3_read(tz, o0, nn, buf, 8)) { free(buf); free(idx); return o; }
            const size_t cnt = (size_t)nn[0] * plane, base = (size_t)z0 * plane;
            for (size_t k = 0; k < cnt; k++) {
                const uint8_t v = buf[k];
                if (binary ? v == 0 : v == 255 || (surf ? v == 0 : 0)) continue;
                if (n == cap) { cap *= 2; uint32_t *t2 = realloc(idx, cap * sizeof *idx); if (!t2) { free(buf); free(idx); return o; } idx = t2; }
                idx[n++] = (uint32_t)(base + k);
            }
        }
    }
    free(buf);
    if (!n) { free(idx); return o; }
    o.idx = idx; o.n = n; o.lev = lc; for (int d = 0; d < 3; d++) o.shape[d] = m->shape[d];
    fprintf(stderr, "sampler: %s: %zu of %zu level-%d cells %s\n", s->name, n, tot, lc, surf ? "contain surface" : "are labelled");
    return o;
}
/* Pull the CT of every occupied cell at levels 0..maxlev through the chunk cache (training then reads from local disk).
   Cells are visited in a shuffled order so a partial prefetch is still useful; one thread per in-flight region. */
/* Prefetch the CT chunks that training touches into the chunk cache: for every coarse cell that contains surface,
   at every level the window around the cell (P wide, P/2 random offset, then snapped to the chunk grid) and the coarse
   probe window at level + 3; chunks are collected in a bitmap per level so each is fetched once, without decoding. */
typedef struct { z3 *ct; uint8_t *bits; int64_t ng[3]; size_t from, to; size_t *done, *fetched, *fail; } pf_arg;
static void *pf_worker(void *p) {
    pf_arg *a = p;
    size_t total = (size_t)a->ng[0] * a->ng[1] * a->ng[2];
    for (size_t k = a->from; k < a->to && k < total; k++) {
        if (!(a->bits[k >> 3] & (1u << (k & 7)))) continue;
        int64_t cz = (int64_t)(k / ((size_t)a->ng[1] * a->ng[2])), cy = (int64_t)((k / (size_t)a->ng[2]) % (size_t)a->ng[1]), cx = (int64_t)(k % (size_t)a->ng[2]);
        int rc = z3_prefetch_chunk(a->ct, cz, cy, cx);
        if (rc < 0) __atomic_fetch_add(a->fail, 1, __ATOMIC_RELAXED); else if (rc > 0) __atomic_fetch_add(a->fetched, 1, __ATOMIC_RELAXED);
        __atomic_fetch_add(a->done, 1, __ATOMIC_RELAXED);
    }
    return nullptr;
}
static void pf_mark(uint8_t *bits, const int64_t *ng, const int64_t *lo, const int64_t *hi) {   /* chunk index box, inclusive, clamped */
    int64_t c0[3], c1[3];
    for (int d = 0; d < 3; d++) { c0[d] = lo[d] < 0 ? 0 : lo[d]; c1[d] = hi[d] >= ng[d] ? ng[d] - 1 : hi[d]; if (c0[d] > c1[d]) return; }
    for (int64_t z = c0[0]; z <= c1[0]; z++) for (int64_t y = c0[1]; y <= c1[1]; y++) for (int64_t x = c0[2]; x <= c1[2]; x++) {
        size_t k = ((size_t)z * (size_t)ng[1] + (size_t)y) * (size_t)ng[2] + (size_t)x; bits[k >> 3] |= (uint8_t)(1u << (k & 7));
    }
}
int sources_prefetch(sources *S, int maxlev, int nthreads, double fraction) {
    const int P = 128;   /* training window (level voxels); windows up to this size are covered */
    for (int i = 0; i < S->n; i++) {
        source *s = &S->src[i];
        occ_index o = source_occupancy(s);
        if (!o.n) { fprintf(stderr, "prefetch: %s: no occupancy index (regions source?), skipped\n", s->name); continue; }
        size_t ncell = (size_t)(o.n * fraction); if (ncell < 1) ncell = 1; if (ncell > o.n) ncell = o.n;
        for (int l = 0; l < MAXLEV; l++) {
            z3 *ct = source_ct(s, l); if (!ct) continue;
            const z3_meta *m = z3_meta_of(ct);
            int want_window = l <= maxlev && l >= s->min_level, want_probe = l >= 3 && l - 3 <= maxlev && l - 3 >= s->min_level;   /* probe reads at level l serve training levels l-3 */
            int whole = l >= 4;   /* the occupancy index reads the coarsest level in full at every start: cache it entirely */
            if (!want_window && !want_probe && !whole) continue;
            int64_t ng[3]; for (int d = 0; d < 3; d++) ng[d] = (m->shape[d] + m->chunk[d] - 1) / m->chunk[d];
            size_t total = (size_t)ng[0] * ng[1] * ng[2];
            uint8_t *bits = calloc((total + 7) / 8, 1);
            int dl = o.lev - l; int64_t f = dl >= 0 ? (int64_t)1 << dl : 1;
            if (whole) { int64_t lo[3] = {0, 0, 0}, hi[3] = {ng[0] - 1, ng[1] - 1, ng[2] - 1}; pf_mark(bits, ng, lo, hi); }
            for (size_t k = 0; k < ncell && !whole; k++) {
                uint32_t id = o.idx[k];
                int64_t cc[3] = {(int64_t)(id / (o.shape[1] * o.shape[2])), (int64_t)((id / o.shape[2]) % o.shape[1]), (int64_t)(id % o.shape[2])};
                int64_t lo[3], hi[3];
                for (int d = 0; d < 3; d++) {
                    int64_t v0 = dl >= 0 ? cc[d] * f : cc[d] >> (-dl), v1 = dl >= 0 ? (cc[d] + 1) * f - 1 : v0;   /* cell extent at this level */
                    int64_t wlo = v0 - P / 2 - m->chunk[d], whi = v1 - P / 2 + P;                               /* window origin range, snapped down, plus the window */
                    if (want_probe && !want_window) { wlo = v0 - (P >> 3) / 2; whi = v1 + (P >> 3); }           /* probe window (P/8 wide) */
                    else if (want_probe) { int64_t plo = v0 - (P >> 3) / 2; if (plo < wlo) wlo = plo; }
                    lo[d] = (wlo < 0 ? 0 : wlo) / m->chunk[d]; hi[d] = (whi >= m->shape[d] ? m->shape[d] - 1 : whi) / m->chunk[d];
                }
                pf_mark(bits, ng, lo, hi);
            }
            size_t nmark = 0; for (size_t k = 0; k < total; k++) nmark += (bits[k >> 3] >> (k & 7)) & 1;
            size_t done = 0, fetched = 0, fail = 0; pthread_t *th = malloc(nthreads * sizeof *th); pf_arg *args = malloc(nthreads * sizeof *args);
            for (int t = 0; t < nthreads; t++) { args[t] = (pf_arg){ct, bits, {ng[0], ng[1], ng[2]}, total * t / nthreads, total * (t + 1) / nthreads, &done, &fetched, &fail}; pthread_create(&th[t], nullptr, pf_worker, &args[t]); }
            size_t last = 0;
            while (last < nmark) { usleep(1000000); last = __atomic_load_n(&done, __ATOMIC_RELAXED); fprintf(stderr, "prefetch: %s level %d: %zu / %zu chunks checked, %zu fetched\r", s->name, l, last, nmark, __atomic_load_n(&fetched, __ATOMIC_RELAXED)); }
            for (int t = 0; t < nthreads; t++) pthread_join(th[t], nullptr);
            free(th); free(args); free(bits);
            fprintf(stderr, "\nprefetch: %s level %d done (%zu chunks needed, %zu fetched now, %zu failures)\n", s->name, l, nmark, fetched, fail);
        }
        free(o.idx);
    }
    return 0;
}
sampler *sampler_start(sources *S, const sample_cfg *cfg) {
    if (!S || S->n <= 0 || cfg->P <= 0 || cfg->B <= 0 || cfg->nworkers <= 0 || cfg->nbuf <= 0) { fprintf(stderr, "sampler: invalid sources, patch, batch, workers or buffer count\n"); return nullptr; }
    if (cfg->erode < 0 || cfg->erode > 1 || (cfg->erode && (cfg->P < 2 || cfg->dilate || cfg->sheet))) {
        fprintf(stderr, "sampler: erode must be 0 or 1, with P >= 2, no dilation and no winding task\n"); return nullptr;
    }
    if (cfg->erode) {
        /* erosion applies to binary targets at native level 0 only (coarser levels and "prob" targets are read as stored) */
        for (int i = 0; i < S->n; i++) for (int ch = 0; ch < NCH; ch++) {
            if (S->src[i].reg[ch] || (S->src[i].tgt_key[ch] && !S->src[i].tgt_binary[ch] && !S->src[i].tgt_prob[ch])) {
                fprintf(stderr, "sampler: erosion requires binary pyramid targets\n"); return nullptr;
            }
        }
    }
    if (cfg->Pz > 0 && cfg->Pz != cfg->P) {   /* slab windows: the paths that take a Pz x P x P window */
        const char *why = cfg->Pz > cfg->P ? "Pz above P" : cfg->cover ? "a cover plan" : cfg->sheet ? "winding geometry" : cfg->band ? "band affinities" :
            cfg->aux ? "an aux input" : cfg->cond ? "conditioning" : cfg->papyrus || cfg->pap_head ? "papyrus labels" : cfg->dilate ? "dilation" :
            cfg->ct_augment ? "CT augmentation" : cfg->geometry_augment ? "geometry augmentation" : scan_aug_enabled(&cfg->scan) ? "scanner augmentation" :
            (cfg->augment == 1 || cfg->augment == 2) ? "symmetries that move z (use --zfix 1)" : (cfg->erode && cfg->level_p[0] > 0) ? "native-level erosion" : nullptr;
        for (int i = 0; !why && i < S->n; i++) for (int ch = 0; ch < NCH; ch++) if (S->src[i].reg[ch]) why = "region targets";
        if (why) { fprintf(stderr, "sampler: slab windows (Pz %d < P %d) do not support %s\n", cfg->Pz, cfg->P, why); return nullptr; }
    }
    sampler *sp = calloc(1, sizeof *sp);
    sp->S = S;
    sp->cfg = *cfg;
    if (cfg->cover) {
        if (cfg->holdout || cfg->B != 1 || cfg->cover->P != cfg->P || cfg->cover_start >= cfg->cover->count) { free(sp); return nullptr; }
        sp->cfg.deterministic = 1; sp->cfg.snap = 0;
        memset(sp->cfg.level_p, 0, sizeof sp->cfg.level_p); sp->cfg.level_p[0] = 1;
    }
    { unsigned sb; memcpy(&sb, &cfg->soft, 4); atomic_store(&sp->soft_bits, sb); }
    sp->prof = ufsm_env_on("UFSM_SAMPLER_PROF");
    sp->lvl_ok = calloc((size_t)S->n, sizeof *sp->lvl_ok);
    sources_open_all(S, 32);   /* every (source, level, channel) store opened up front in parallel: lazily they serialise on one lock (254 opens, 17 s) */
    sp->cum = malloc((size_t)S->n * sizeof(double));
    double acc = 0;
    for (int i = 0; i < S->n; i++) {
        source *s = &S->src[i]; int region_ch = -1, eligible = 0;
        for (int c = 0; c < NCH; c++) if (s->reg[c] && s->reg[c]->n) region_ch = c;
        for (int l = s->min_level < 0 ? 0 : s->min_level; l < MAXLEV; l++) if (sp->cfg.level_p[l] > 0 && (region_ch < 0 || l <= 1) && level_ok(sp, s, i, l, region_ch)) eligible = 1;
        if (!isfinite(s->weight) || s->weight < 0) { fprintf(stderr, "sampler: invalid source weight: %s\n", s->name); free(sp->cum); free(sp->lvl_ok); free(sp); return nullptr; }
        if (!eligible && s->weight > 0) fprintf(stderr, "sampler: %s excluded: no %s patch fits %dx%dx%d and the enabled levels\n", s->name, cfg->holdout ? "validation" : "training", PZ(cfg), cfg->P, cfg->P);
        acc += eligible ? s->weight : 0; sp->cum[i] = acc;
    }
    if (acc <= 0) { fprintf(stderr, "sampler: no eligible sources; check window, holdout boxes, regions and levels\n"); free(sp->cum); free(sp->lvl_ok); free(sp); return nullptr; }
    if (cfg->cover) for (uint64_t k = 0; k < cfg->cover->count; k++) {
        const int64_t *t = cfg->cover->tiles[k]; source *s = &S->src[t[0]];
        z3 *ct = source_ct(s, 0); const z3_meta *m = ct ? z3_meta_of(ct) : nullptr;
        int ok = m && s->min_level == 0 && !s->reg[0] && !s->reg[1] && s->tgt_key[0] && source_tgt_for_level(s, 0, 0, nullptr);
        for (int d = 0; ok && d < 3; d++) ok &= t[d + 1] >= 0 && t[d + 1] + cfg->P <= m->shape[d];
        if (!ok || training_hits_holdout(sp, s, t + 1, 0)) {
            fprintf(stderr, "sampler: cover tile %llu is out of bounds or intersects a holdout\n", (unsigned long long)k);
            free(sp->cum); free(sp->lvl_ok); free(sp); return nullptr;
        }
    }
    sp->nslots = cfg->nbuf;
    sp->slot_j = calloc((size_t)sp->nslots, sizeof *sp->slot_j);
    sp->slots = calloc((size_t)sp->nslots, sizeof *sp->slots);
    sp->state = calloc((size_t)sp->nslots, sizeof *sp->state);
    for (int i = 0; i < sp->nslots; i++) sp->slots[i] = alloc_batch(cfg);
    pthread_mutex_init(&sp->mu, nullptr);
    pthread_cond_init(&sp->cv_free, nullptr);
    pthread_cond_init(&sp->cv_ready, nullptr);
    sp->occ = calloc((size_t)S->n, sizeof *sp->occ);
    for (int i = 0; !cfg->cover && i < S->n; i++) {
        occ_index o = source_occupancy(&S->src[i]);
        sp->occ[i].idx = o.idx; sp->occ[i].n = o.n; sp->occ[i].lev = o.lev; for (int d = 0; d < 3; d++) sp->occ[i].shape[d] = o.shape[d];
    }
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
        if (atomic_load(&sp->stop)) { pthread_mutex_unlock(&sp->mu); return nullptr; }
        if (sp->cfg.cover && (uint64_t)sp->next_take >= sp->cfg.cover->count - sp->cfg.cover_start) { pthread_mutex_unlock(&sp->mu); return nullptr; }
        if (sp->cfg.deterministic) {   /* in batch order */
            int s = (int)(sp->next_take % sp->nslots);
            if (sp->state[s] == READY && sp->slot_j[s] == sp->next_take) { k = s; sp->next_take++; pthread_cond_broadcast(&sp->cv_free); }
        } else
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
    if (sp->cfg.deterministic) pthread_cond_broadcast(&sp->cv_free); else pthread_cond_signal(&sp->cv_free);
    pthread_mutex_unlock(&sp->mu);
}

void sampler_stop(sampler *sp) {
    if (!sp) return;
    atomic_store(&sp->stop, 1);
    pthread_mutex_lock(&sp->mu);
    pthread_cond_broadcast(&sp->cv_free);
    pthread_cond_broadcast(&sp->cv_ready);
    pthread_mutex_unlock(&sp->mu);
    for (int i = 0; i < sp->cfg.nworkers; i++) pthread_join(sp->th[i], nullptr);
    for (int i = 0; i < sp->nslots; i++) free_batch(&sp->slots[i]);
    for (int i = 0; i < sp->S->n; i++) free(sp->occ[i].idx);
    free(sp->occ); free(sp->slot_j); free(sp->slots); free(sp->state); free(sp->th); free(sp->cum); free(sp->lvl_ok);
    free(sp);
}

void sampler_set_soft(sampler *sp, float sigma) { unsigned sb; memcpy(&sb, &sigma, 4); atomic_store(&sp->soft_bits, sb); }
int sampler_failed(const sampler *sp) { return atomic_load(&sp->failed); }
void sampler_prof_print(const sampler *sp) {
    if (!sp->prof) return;
    uint64_t n = atomic_load(&sp->produced); if (!n) n = 1;
    fprintf(stderr, "sampler profile (ms per produced patch, cpu time summed over workers):");
    for (int k = 0; k < PS_N; k++) fprintf(stderr, " %s %.1f", prof_names[k], atomic_load(&sp->prof_ns[k]) / 1e6 / (double)n);
    { uint64_t h, r; z3_io_stats(&h, &r); fprintf(stderr, " | chunk reads: %llu cached, %llu from the store", (unsigned long long)h, (unsigned long long)r); }
    fprintf(stderr, "\n");
}
void sampler_stats(const sampler *sp, uint64_t *produced, uint64_t *rejected) {
    *produced = atomic_load(&sp->produced);
    *rejected = atomic_load(&sp->rejected);
}
