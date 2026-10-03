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
#include <time.h>

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
    c.min_fg = 0.3; c.empty_keep = 0.2; c.augment = 1; c.snap = 1;
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
        if (y.perm[0] == 0 && y.perm[1] == 1 && !y.flip[0] && !y.flip[1] && !y.flip[2]) { memcpy(out, in, (size_t)P * P * P * sizeof(T)); return; } \
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

static size_t P3(const sampler *sp) { return (size_t)sp->cfg.P * sp->cfg.P * sp->cfg.P; }

static batch alloc_batch(const sample_cfg *c) {
    size_t p3 = (size_t)c->P * c->P * c->P;
    batch b;
    b.x = c->xfmt ? nullptr : nn_host_alloc((size_t)c->B * 4 * p3 * sizeof(float));   /* pinned: the trainer uploads asynchronously */
    b.x16 = c->xfmt ? nn_host_alloc((size_t)c->B * 4 * p3 * 2) : nullptr;
    b.t = nn_host_alloc((size_t)c->B * NCH * p3);
    b.m = nn_host_alloc((size_t)c->B * p3);
    b.w = nn_host_alloc((size_t)c->B * NCH);
    b.src = malloc((size_t)c->B * sizeof(int16_t));
    b.level = malloc((size_t)c->B * sizeof(int8_t));
    b.corner = malloc((size_t)c->B * sizeof *b.corner);
    return b;
}

static void free_batch(batch *b) { if (b->x) nn_host_free(b->x); if (b->x16) nn_host_free(b->x16); nn_host_free(b->t); nn_host_free(b->m); nn_host_free(b->w); free(b->src); free(b->level); free(b->corner); }

/* Level choice for a source: restrict cfg.level_p to levels the CT has and every target of the source
   can provide (pyramid: same level; regions: levels 0..1). Returns -1 if nothing is usable. */
/* which levels a source can serve (CT and at least one target open): computed once per (source, region channel) since
   source_ct / source_tgt take the global open lock and may open remote stores; before this every draw re-checked every level */
/* Inclusive level-l origins whose whole patch lies in the CT, region and validation box.
   Round starts up and ends down: flooring a non-aligned holdout start leaks outside it. */
static int patch_bounds(const source *s, const z3_meta *m, int P, int l, const regions *r, int ri, int validation, int64_t lo[3], int64_t hi[3]) {
    int64_t f = (int64_t)1 << l;
    for (int d = 0; d < 3; d++) {
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
    int64_t span = (int64_t)sp->cfg.P << l;
    for (int i = 0; i < sp->S->n; i++) {
        const source *h = &sp->S->src[i]; if (!h->hold_n[0] || !same_ct(s, h)) continue;
        int hit = 1;
        for (int d = 0; d < 3; d++) { int64_t a = o[d] << l; if (a + span <= h->hold_o[d] || a >= h->hold_o[d] + h->hold_n[d]) hit = 0; }
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
            int fits = 0; int64_t lo[3], hi[3];
            for (int i = 0; i < (r ? r->n : 1); i++) if (patch_bounds(s, m, sp->cfg.P, l, r, i, sp->cfg.holdout, lo, hi) &&
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
        if (l < s->min_level) { p[l] = 0; continue; }
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
enum { PS_SRC, PS_LEVEL, PS_PICK, PS_PROBE, PS_READ, PS_STATS, PS_TARGETS, PS_DILATE, PS_SOFT, PS_TRUST, PS_ENCODE, PS_ZSCORE, PS_AUGMENT, PS_X16, PS_N };
static const char *prof_names[PS_N] = {"src", "level", "pick", "probe", "ctread", "ctstats", "targets", "dilate", "soft", "trust", "encode", "zscore", "augment", "x16"};
/* 3-4-5 chamfer distance (3 per voxel step) from the voxels of t that are annotated surface (0 < t < 255): two raster
   passes. Returns 0 (dm all CH_INF) when the patch has no surface. */
#define CH_INF 60000
static inline uint16_t mn16(uint16_t a, uint16_t b) { return a < b ? a : b; }
/* one raster pass (dir +1 forward, -1 backward). Per row, the neighbours in the previous row and the previous plane are
   final for the pass, so they are folded in for the whole row at once (vectorisable); the in-row dependency is then a
   scalar scan. Same result as visiting the voxels in raster order. */
static void chamfer_pass(uint16_t *dm, int P, int dir) {
    const size_t P2 = (size_t)P * P;
    for (int zi = 0; zi < P; zi++) {
        const int z = dir > 0 ? zi : P - 1 - zi, zp = z - dir;
        for (int yi = 0; yi < P; yi++) {
            const int y = dir > 0 ? yi : P - 1 - yi, yp = y - dir;
            uint16_t *row = dm + (size_t)z * P2 + (size_t)y * P;
            /* (row, weight face / edge) pairs: previous row of this plane (3, 4); previous plane rows y (3, 4), y-1 and y+1 (4, 5) */
            const uint16_t *nb[4] = {nullptr, nullptr, nullptr, nullptr}; uint16_t wf[4] = {3, 3, 4, 4}, we[4] = {4, 4, 5, 5};
            if (yp >= 0 && yp < P) nb[0] = dm + (size_t)z * P2 + (size_t)yp * P;
            if (zp >= 0 && zp < P) {
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
static int chamfer345(const uint8_t *t, uint16_t *dm, int P) {
    size_t p3 = (size_t)P * P * P; int any = 0;
    for (size_t k = 0; k < p3; k++) { int s = t[k] != 255 && t[k] > 0; dm[k] = s ? 0 : CH_INF; any |= s; }
    if (!any) return 0;
    chamfer_pass(dm, P, 1);
    chamfer_pass(dm, P, -1);
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
static void write_x(const uint8_t *ctu, int P, const int64_t o[3], float fmean, float fisd, int hasax, const float *cyz, const float *cxz,
                    sym y, int jitter, float ia, float ib, float isg, rng *r, float *nrow, int xfmt, float *X, uint16_t *H) {
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
            }
            const size_t ko = ((size_t)z * P + yy) * P;
            for (int x = 0; x < P; x++) {
                const int sz_ = s0[0] + ds[0] * x, sy_ = s0[1] + ds[1] * x, sx_ = s0[2] + ds[2] * x;
                const size_t q = ((size_t)sz_ * P + sy_) * P + sx_;
                float v0 = ((float)ctu[q] - fmean) * fisd;
                if (jitter) v0 = ia * v0 + ib + isg * nrow[x];
                float comp[3] = {0.f, 0.f, 0.f};
                if (hasax) {
                    const float dyv = (float)(o[1] + sy_) - cyz[sz_], dxv = (float)(o[2] + sx_) - cxz[sz_];
                    const float inv = 1.f / (sqrtf(dyv * dyv + dxv * dxv) + 1e-6f);
                    comp[1] = dyv * inv; comp[2] = dxv * inv;
                }
                const float v1 = sgn[0] * comp[y.perm[0]], v2 = sgn[1] * comp[y.perm[1]], v3 = sgn[2] * comp[y.perm[2]];
                const size_t k = ko + x;
                if (xfmt == 1) { _Float16 h0 = (_Float16)v0, h1 = (_Float16)v1, h2 = (_Float16)v2, h3 = (_Float16)v3; memcpy(&H[k], &h0, 2); memcpy(&H[p3 + k], &h1, 2); memcpy(&H[2 * p3 + k], &h2, 2); memcpy(&H[3 * p3 + k], &h3, 2); }
                else if (xfmt == 2) { H[k] = bf16_rn(v0); H[p3 + k] = bf16_rn(v1); H[2 * p3 + k] = bf16_rn(v2); H[3 * p3 + k] = bf16_rn(v3); }
                else { X[k] = v0; X[p3 + k] = v1; X[2 * p3 + k] = v2; X[3 * p3 + k] = v3; }
            }
        }
}
static int draw(sampler *sp, batch *b, int i, rng *r, float *xtmp, uint8_t *ttmp, uint8_t *big) {
    double pt = sp->prof ? tnow() : 0;
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
    PROF_MARK(PS_SRC);
    int l = pick_level(sp, s, r, region_ch);
    if (l < 0) return 1;
    PROF_MARK(PS_LEVEL);
    z3 *ct = source_ct(s, l);
    const z3_meta *m = z3_meta_of(ct);
    int64_t o[3], n[3] = {P, P, P};
    if (region_ch >= 0) {
        regions *R = s->reg[region_ch];
        ri = (int)rint_below(r, R->n);
        int64_t lo[3], hi[3]; if (!patch_bounds(s, m, P, l, R, ri, c->holdout, lo, hi)) return 1;
        for (int d = 0; d < 3; d++) o[d] = lo[d] + rint_below(r, hi[d] - lo[d] + 1);
    } else if (c->holdout) {
        int64_t lo[3], hi[3]; if (!patch_bounds(s, m, P, l, nullptr, 0, 1, lo, hi)) return 1;
        for (int d = 0; d < 3; d++) o[d] = lo[d] + rint_below(r, hi[d] - lo[d] + 1);
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
    if (c->snap && !c->holdout && region_ch < 0) {   /* training only: snapping a validation origin can leave its holdout */
        for (int d = 0; d < 3; d++) {
            int64_t cs = m->chunk[d]; if (cs <= 0) continue;
            o[d] -= o[d] % cs;
            if (o[d] > m->shape[d] - P) o[d] = m->shape[d] - P;   /* the last window of an axis may stay unaligned */
            if (o[d] < 0) o[d] = 0;
        }
    }
    if (!c->holdout && training_hits_holdout(sp, s, o, l)) return 1;
    PROF_MARK(PS_PICK);
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
    PROF_MARK(PS_PROBE);
    /* CT */
    uint8_t *ctu = big;
    if (z3_read(ct, o, n, ctu, 1)) return -1;
    PROF_MARK(PS_READ);
    size_t nz = 0;
    double sum = 0, sq = 0;
    for (size_t k = 0; k < p3; k++) { nz += ctu[k] != 0; sum += ctu[k]; sq += (double)ctu[k] * ctu[k]; }
    if ((double)nz / (double)p3 < c->min_fg) { atomic_fetch_add(&sp->rejected, 1); return 1; }
    PROF_MARK(PS_STATS);
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
            z3 *tz = source_tgt_for_level(s, ch, l, nullptr);
            if (!tz) { memset(dst, 0, p3); continue; }
            if (source_read_target(s, ch, l, o, n, dst, 1)) return -1;
            w[ch] = 1;
        } else memset(dst, 0, p3);
        if (w[ch]) for (size_t k = 0; k < p3; k++) if (dst[k] != 255 && dst[k] > tmax) tmax = dst[k];
    }
    if (tmax == 0 && runif(r) > c->empty_keep) { atomic_fetch_add(&sp->rejected, 1); return 1; }
    PROF_MARK(PS_TARGETS);
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
    PROF_MARK(PS_DILATE);
    /* soft ridge target: background voxels near the surface get 254 * exp(-(d / sigma)^2 / 2), d = 3-4-5 chamfer distance / 3 */
    float soft0; { unsigned sb = atomic_load(&sp->soft_bits); memcpy(&soft0, &sb, 4); }
    uint16_t *dm = (uint16_t *)(big + 2 * p3);   /* chamfer distance to the annotated surface, per channel (3 per voxel); big has room */
    int dm_ch = -1, dm_any = 0;                   /* channel the map currently holds, and whether that channel had any surface */
#define CHAMFER_FOR(ch_) do { if (dm_ch != (ch_)) { dm_ch = (ch_); dm_any = chamfer345(ttmp + (size_t)(ch_) * p3, dm, P); } } while (0)
    if (soft0 > 0 && region_ch < 0) {
        float sigma = soft0 / (float)(1 << l); if (sigma < 0.75f) sigma = 0.75f;
        /* exp(-(d/3)^2 / (2 sigma^2)) tabulated over the chamfer distance (3 per voxel); beyond dcut it rounds to 0 */
        int dcut = (int)(3.f * sigma * 4.5f) + 1; if (dcut > 4000) dcut = 4000;
        uint8_t *etab = big + 4 * p3;
        for (int d = 0; d <= dcut; d++) { float df = d / 3.f; int t = (int)(254.f * expf(-0.5f * df * df / (sigma * sigma)) + 0.5f); etab[d] = (uint8_t)(t > 254 ? 254 : t); }
        for (int ch = 0; ch < NCH; ch++) {
            if (!w[ch]) continue;
            uint8_t *dst = ttmp + (size_t)ch * p3;
            CHAMFER_FOR(ch);
            if (!dm_any) continue;   /* no surface in the patch: nothing to soften */
            for (size_t k = 0; k < p3; k++) if (dst[k] == 0 && dm[k] <= dcut) { uint8_t t = etab[dm[k]]; if (t) dst[k] = t; }
        }
    }
    PROF_MARK(PS_SOFT);
    /* partial annotations: trust background only within R voxels of an annotated surface, ignore the rest */
    if (s->trust_band > 0) {
        int R = s->trust_band >> l; if (R < 1) R = 1;
        for (int ch = 0; ch < NCH; ch++) {
            if (!w[ch]) continue;
            uint8_t *dst = ttmp + (size_t)ch * p3;
            CHAMFER_FOR(ch);   /* measured from the hard surface (the soft values added above are not surface) */
            if (!dm_any) { for (size_t k = 0; k < p3; k++) if (dst[k] == 0) dst[k] = 255; continue; }   /* nothing annotated: ignore all background */
            for (size_t k = 0; k < p3; k++) if (dst[k] == 0 && dm[k] > 3 * R) dst[k] = 255;
        }
    }
    /* label encoding -> target probability * 255 and the ignore mask (255 = ignore) */
    PROF_MARK(PS_TRUST);
    uint8_t *ign = big + 9 * p3;
    memset(ign, 0, p3);
    for (int ch = 0; ch < NCH; ch++) {
        if (!w[ch]) continue;
        uint8_t *dst = ttmp + (size_t)ch * p3;
        for (size_t k = 0; k < p3; k++) { uint8_t v = dst[k]; if (v == 255) { ign[k] = 1; dst[k] = 0; } else dst[k] = (uint8_t)((v * 255 + 127) / 254); }
    }
    PROF_MARK(PS_ENCODE);
    /* input channels (z-scored CT, radial unit vector) written in one pass straight into the batch with the symmetry
       and the jitter applied on the way (16-bit when cfg.xfmt, else fp32): no fp32 intermediates */
    double mean = sum / (double)p3, var = sq / (double)p3 - mean * mean, sd = sqrt(var > 0 ? var : 0) + 1e-3;
    double scale = (double)(1 << l);
    float *cyz = xtmp, *cxz = xtmp + P, *nrow = xtmp + 2 * P;
    for (int z = 0; z < P; z++) { double cy, cx; axis_at(&s->ax, (double)(o[0] + z) * scale, &cy, &cx); cyz[z] = (float)(cy / scale); cxz[z] = (float)(cx / scale); }
    uint8_t *mask = big + 9 * p3;   /* ign lives here: fold CT > 0 into it */
    for (size_t k = 0; k < p3; k++) mask[k] = ctu[k] != 0 && !mask[k];
    PROF_MARK(PS_ZSCORE);
    uint8_t *T = b->t + (size_t)i * NCH * p3, *M = b->m + (size_t)i * p3;
    sym y = sym_of(0);
    float ia = 1.f, ib = 0.f, isg = 0.f;
    if (c->augment) {
        y = sym_of((int)rint_below(r, 48));
        if (c->augment == 4) y = sym_of(0);   /* intensity jitter only */
        if (c->augment == 2) while (!sym_is_rotation(y)) y = sym_of((int)rint_below(r, 48));   /* rotations only */
        if (c->augment == 3) { y = sym_of((int)rint_below(r, 48)); while (y.perm[0] != 0 || y.flip[0]) y = sym_of((int)rint_below(r, 48)); }   /* z fixed: y/x swaps and flips only */
        double a = exp((runif(r) * 2 - 1) * 0.22), bb = (runif(r) * 2 - 1) * 0.2, sg = runif(r) * 0.1;
        ia = (float)a; ib = (float)bb; isg = (float)sg;
    }
    for (int ch = 0; ch < NCH; ch++) sym_u8(ttmp + (size_t)ch * p3, T + (size_t)ch * p3, P, y);
    sym_u8(mask, M, P, y);
    PROF_MARK(PS_AUGMENT);
    write_x(ctu, P, o, (float)mean, (float)(1.0 / sd), s->ax.n > 0, cyz, cxz, y, c->augment != 0, ia, ib, isg, r, nrow, c->xfmt,
            c->xfmt ? nullptr : b->x + (size_t)i * 4 * p3, c->xfmt ? b->x16 + (size_t)i * 4 * p3 : nullptr);
    PROF_MARK(PS_X16);
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
    float *xtmp = malloc(3 * (size_t)sp->cfg.P * sizeof(float));   /* per-z axis centres and a row of noise */
    uint8_t *ttmp = malloc(NCH * p3);
    uint8_t *big = malloc(10 * p3);   /* [0,P^3) CT (also the coarse probe), [P^3, 9P^3) (2P)^3 region scratch, [9P^3, 10P^3) mask */
    const int det = sp->cfg.deterministic;
    while (!atomic_load(&sp->stop)) {
        pthread_mutex_lock(&sp->mu);
        int k = -1;
        int64_t jb = -1;
        if (det) {   /* claim the next batch index; its slot is fixed (jb % nslots) and must be free */
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
            if (det && spin == 0 && fail == 0) rseed(&r, sp->cfg.seed * 0x9e3779b97f4a7c15ull + (uint64_t)(jb * sp->cfg.B + i) * 1000003ull + 1);   /* per-sample stream */
            int rc = draw(sp, b, i, &r, xtmp, ttmp, big);
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
    free(xtmp); free(ttmp); free(big);
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
    sampler *sp = calloc(1, sizeof *sp);
    sp->S = S;
    sp->cfg = *cfg;
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
        if (!eligible && s->weight > 0) fprintf(stderr, "sampler: %s excluded: no %s patch fits P=%d and the enabled levels\n", s->name, cfg->holdout ? "validation" : "training", cfg->P);
        acc += eligible ? s->weight : 0; sp->cum[i] = acc;
    }
    if (acc <= 0) { fprintf(stderr, "sampler: no eligible sources; check window, holdout boxes, regions and levels\n"); free(sp->cum); free(sp->lvl_ok); free(sp); return nullptr; }
    sp->nslots = cfg->nbuf;
    sp->slot_j = calloc((size_t)sp->nslots, sizeof *sp->slot_j);
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
