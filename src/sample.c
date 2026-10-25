#include "sample.h"
#include "band.h"
#include "target_erode.h"
#include "papyrus.h"
#include "nn.h"
#include "ct_augment.h"
#include "spatial_augment.h"
#include "edt.h"
#include "srpair.h"
#include "view.h"
#include "scrollaug.h"
#include "sample_emb.h"
#include <math.h>
#include <stddef.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
static void *wbig_alloc(size_t n, int zero);
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__F16C__) || defined(__AVX2__)
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
    c.min_fg = 0.3; c.empty_keep = 0.2; c.augment = 1; c.snap = 1; c.symmetry_p = 1; c.chain_len = 8; c.chain_step = 0.5f;
    c.crush = crush_cfg_default();
    return c;
}

/* train --aug presets (sample.h). "all": every op at a moderate probability (nnU-Net-like p 0.05 .. 0.3 each, about 1 window in 20
   untouched by the scanner ops, 2 in 5 unwarped), magnitudes at strength 1 (DESIGN.md "Generalising to other scrolls" used the same
   ops at up to 1.5 for blur); dropped slices / cutouts keep their labels (drop_ignore 0: the net infers occluded papyrus from context);
   the hist op needs target histograms (train --aug-hist FILE). pscale multiplies every op probability of the preset. */
int sample_aug_preset(sample_cfg *c, const char *name) {
    if (!strcmp(name, "none")) {
        c->ct_augment = 0; c->geometry_augment = 0; c->zoom_p = c->zoom_out_p = 0; c->affine_p = 0; c->axis_jitter = 0;
        c->axis_jitter_p = c->axis_jitter_deg = 0; memset(c->scan.p, 0, sizeof c->scan.p); c->scan.strength = 1; c->scan.drop_ignore = 1;
        c->rotate_degrees = 5; c->rotate_p = .2f; c->elastic = 1; c->elastic_p = .15f; c->zoom_min = .7071f; c->zoom_max = 1;
        c->affine_aniso = .1f; c->affine_shear = .06f;
        return 0;
    }
    if (strcmp(name, "all")) return -1;
    c->ct_augment = 0;   /* (its gamma / filter / shading / correlated noise are scanner ops here) */
    c->geometry_augment = 1; c->rotate_degrees = 10; c->rotate_p = .25f; c->elastic = 1; c->elastic_p = .15f;
    c->zoom_min = .8f; c->zoom_p = .2f; c->zoom_max = 1.25f; c->zoom_out_p = .15f;
    c->affine_p = .1f; c->affine_aniso = .1f; c->affine_shear = .06f;
    c->axis_jitter = 800; c->axis_jitter_p = .1f; c->axis_jitter_deg = 2;   /* native voxels (~2 mm at 2.4 um): an estimated umbilicus */
    c->scan.strength = 1; c->scan.drop_ignore = 0;
    static const struct { int op; float p; } P_ALL[] = {
        {SA_SEAM, .05f}, {SA_AIR, .05f}, {SA_LOWRES, .25f}, {SA_BLUR, .3f}, {SA_SHARPEN, .05f}, {SA_BIAS, .15f}, {SA_RING, .1f},
        {SA_CUPPING, .1f}, {SA_SLICE, .1f}, {SA_NOISE, .15f}, {SA_CNOISE, .1f}, {SA_TONE, .25f}, {SA_QUANT, .1f}, {SA_DROPSLICE, .05f},
        {SA_CUTOUT, .1f}, {SA_CHUNK, .05f}, {SA_PAGANIN, .15f}, {SA_GIN, .1f}, {SA_MOTION, .1f}};
    memset(c->scan.p, 0, sizeof c->scan.p);
    for (size_t i = 0; i < sizeof P_ALL / sizeof P_ALL[0]; i++) c->scan.p[P_ALL[i].op] = P_ALL[i].p;
    return 0;
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
    atomic_uint_fast64_t rej_why[4];   /* draws dropped: coarse CT probe, CT read (min_fg), empty targets, holdout overlap (not in rejected) */
    atomic_uint_fast64_t crushed;      /* --aug-crush windows drawn */
    atomic_uint_fast64_t augn[AUGN_N]; /* scroll augmentations applied (sampler_aug_count) */
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

/* batch buffers: pinned (the trainer uploads asynchronously), or plain malloc with UFSM_SAMPLER_NOPIN=1 (CPU-only benchmarks) */
static int nopin(void) { static int v = -1; if (v < 0) { const char *e = getenv("UFSM_SAMPLER_NOPIN"); v = e && atoi(e); } return v; }
static void *halloc(size_t n) { if (nopin()) { void *p = malloc(n ? n : 1); if (!p) abort(); return p; } return nn_host_alloc(n); }
static void hfree(void *p) { if (nopin()) free(p); else nn_host_free(p); }
static batch alloc_batch(const sample_cfg *c) {
    size_t p3 = (size_t)PZ(c) * c->P * c->P;
    batch b;
    const size_t xc = c->gpu_input ? 1 : 4 + (c->aux != 0) + (c->cond ? 3 : 0);   /* input channels per sample (--gpu-input: the CT alone) */
    b.xch = (int)xc;
    b.rad = c->gpu_input ? halloc((size_t)c->B * NN_RAD_REC(PZ(c))) : nullptr;
    b.axr = c->axis_rec > 0 ? halloc((size_t)c->B * SAMPLE_AXR(PZ(c) > c->P ? PZ(c) : c->P) * sizeof(float)) : nullptr;
    b.chid = c->chain_p > 0 ? malloc((size_t)c->B * sizeof *b.chid) : nullptr; b.chidx = c->chain_p > 0 ? malloc((size_t)c->B * sizeof *b.chidx) : nullptr;
    b.choff = c->chain_p > 0 ? malloc((size_t)c->B * 3 * sizeof *b.choff) : nullptr;
    b.xv = c->view_P > 0 ? halloc((size_t)c->B * c->view_P * c->view_P * c->view_P) : nullptr; b.vok = c->view_P > 0 ? calloc((size_t)c->B, 1) : nullptr;
    b.x = c->xfmt || c->restore ? nullptr : halloc((size_t)c->B * xc * p3 * sizeof(float));   /* pinned: the trainer uploads asynchronously */
    b.x16 = c->xfmt && !c->restore ? halloc((size_t)c->B * xc * p3 * 2) : nullptr;
    b.t = c->pack_tm ? nullptr : halloc((size_t)c->B * (c->restore == 1 ? 1 : NCH) * p3);   /* (task restore: the clean CT alone; real pairs: target + input crop) */
    b.m = c->pack_tm || c->restore ? nullptr : halloc((size_t)c->B * p3);
    b.tm = c->pack_tm ? halloc((size_t)c->B * p3) : nullptr;   /* one code byte per voxel instead of NCH + 1 */
    b.w = halloc((size_t)c->B * NCH);
    b.src = malloc((size_t)c->B * sizeof(int16_t));
    b.level = malloc((size_t)c->B * sizeof(int8_t));
    b.um = malloc((size_t)c->B * sizeof(float));
    b.corner = malloc((size_t)c->B * sizeof *b.corner);
    b.sheet = calloc((size_t)c->B, sizeof *b.sheet);
    b.band = c->band ? halloc((size_t)c->B * p3) : nullptr;
    b.side = c->side ? halloc((size_t)c->B * p3) : nullptr;
    b.pm = c->pmask ? halloc((size_t)c->B * p3) : nullptr;
    b.wind = c->cond ? halloc((size_t)c->B * p3) : nullptr;
    b.pap = c->pap_head ? halloc((size_t)c->B * p3) : nullptr;
    b.w4 = c->wind4 ? halloc((size_t)c->B * p3) : nullptr;
    b.wcomp = c->wind4 ? halloc((size_t)c->B * p3) : nullptr;
    b.emb = c->emb_ch ? halloc((size_t)c->B * c->emb_ch * (size_t)(PZ(c) / c->emb_div) * (c->P / c->emb_div) * (c->P / c->emb_div)) : nullptr;
    return b;
}

static void free_batch(batch *b) { if (b->emb) hfree(b->emb); if (b->xv) hfree(b->xv); free(b->vok); free(b->chid); free(b->chidx); free(b->choff); if (b->w4) hfree(b->w4); if (b->wcomp) hfree(b->wcomp); if (b->tm) hfree(b->tm); if (b->rad) hfree(b->rad); if (b->axr) hfree(b->axr); if (b->pap) hfree(b->pap); if (b->wind) hfree(b->wind); if (b->band) hfree(b->band); if (b->side) hfree(b->side); if (b->pm) hfree(b->pm); if (b->x) hfree(b->x); if (b->x16) hfree(b->x16); if (b->t) hfree(b->t); if (b->m) hfree(b->m); hfree(b->w); free(b->src); free(b->level); free(b->um); free(b->corner); if (b->sheet) { sheet_batch_free(b->sheet[0]); free(b->sheet); } }

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
/* held-out boxes of a source: box 0 = hold_o / hold_n (the validation box), 1.. = the further "holdout" list / holdout_file boxes */
static inline int src_nhold(const source *h) { return h->hold_n[0] > 0 ? (h->nhold > 1 ? h->nhold : 1) : 0; }
static inline int64_t hb_o(const source *h, int k, int d) { return k ? h->hold[k][d] : h->hold_o[d]; }
static inline int64_t hb_n(const source *h, int k, int d) { return k ? h->hold[k][3 + d] : h->hold_n[d]; }
static int same_ct(const source *a, const source *b) {
    return a == b || (a->um == b->um && !strcmp(a->ct_key, b->ct_key) && !strcmp(store_root(a->s), store_root(b->s)));
}
static int training_hits_holdout(const sampler *sp, const source *s, const int64_t o[3], int l) {
    const int64_t span[3] = {(int64_t)PZ(&sp->cfg) << l, (int64_t)sp->cfg.P << l, (int64_t)sp->cfg.P << l};
    for (int i = 0; i < sp->S->n; i++) {
        const source *h = &sp->S->src[i]; if (!src_nhold(h) || !same_ct(s, h)) continue;
        for (int k = 0; k < src_nhold(h); k++) {
            int hit = 1;
            for (int d = 0; d < 3; d++) { int64_t a = o[d] << l; if (a + span[d] <= hb_o(h, k, d) || a >= hb_o(h, k, d) + hb_n(h, k, d)) hit = 0; }
            if (hit) return 1;
        }
    }
    return 0;
}
/* Any free cell of the origin grid begins at lo or just after a forbidden interval.
   Check these corners to reject a source covered by the union of same-CT holdouts. */
static int training_corner_exists(const sampler *sp, const source *s, int l, const int64_t lo[3], const int64_t hi[3]) {
    int nh = 0; for (int i = 0; i < sp->S->n; i++) nh += src_nhold(&sp->S->src[i]);
    int n[3] = {1, 1, 1}; int64_t v[3][nh + 1], f = (int64_t)1 << l;
    for (int d = 0; d < 3; d++) v[d][0] = lo[d];
    if (!training_hits_holdout(sp, s, lo, l)) return 1;
    for (int i = 0; i < sp->S->n; i++) {
        const source *h = &sp->S->src[i]; if (!src_nhold(h) || !same_ct(s, h)) continue;
        for (int k = 0; k < src_nhold(h); k++)
            for (int d = 0; d < 3; d++) { int64_t end = (hb_o(h, k, d) + hb_n(h, k, d) + f - 1) / f; if (end > lo[d] && end <= hi[d]) v[d][n[d]++] = end; }
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
        if (ok && !sp->cfg.restore) {   /* (task restore: no targets) */
            int any = 0;
            for (int c = 0; c < NCH; c++) {
                if (c == region_ch) { any = 1; continue; }
                if (s->tgt_key[c] && source_tgt_for_level(s, c, l, nullptr)) any = 1;
            }
            if (s->band && (s->phase || l == 0)) any = 1;   /* band codes (+ stored phase) alone label a source (task unwrap2 sources have no recto / ink) */
            if (s->ct_only) any = 1;   /* an unlabelled scan ("ct_only": 1): windows with no target, for label-free terms */
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
#ifdef __AVX2__
/* eight 32-bit lanes holding 0..255 as eight bytes (lane order) */
static inline uint64_t pack8_u32(__m256i v) {
    const __m256i p = _mm256_packus_epi16(_mm256_packus_epi32(v, v), _mm256_setzero_si256());
    return (uint64_t)(uint32_t)_mm_cvtsi128_si32(_mm256_castsi256_si128(p)) | (uint64_t)(uint32_t)_mm_cvtsi128_si32(_mm256_extracti128_si256(p, 1)) << 32;
}
#endif
/* write_x_aug without an appearance plan or sheet prior, one output row at a time in vectorisable loops (no math errno, so
   sqrtf vectorises); bit-identical to it: the same expressions, rng draws and signed zeros. D x P x P windows (D != P: slabs,
   z-keeping symmetries only). */
[[gnu::optimize("no-math-errno")]]
/* write_x's CT channel alone (--gpu-input, no appearance plan), cache-blocked: when the output row runs along a strided source axis
   (2 of 3 symmetries) the row-by-row gather touched one cache line (and often one page) per voxel, ~2/3 of a P768 window's 7.3 cpu-s.
   Rows whose source rows are adjacent are gathered together: XB output rows at once when output y runs along source x, XB output
   planes when output z does (their noise rows buffered as table indices, drawn in write_x's row order). The same rng draws, float
   operations and rounding as write_x: bit for bit (tests/test_gpu_input.c compares the two over all 48 symmetries). */
enum { XB = 16 };
static void write_x_ct(const uint8_t *ctu, int D, int P, float fmean, float fisd, sym y, int jitter, float ia, float ib, float isg, rng *r,
                       int xfmt, uint16_t *H) {
    const size_t p2 = (size_t)P * P; const int nd[3] = {D, P, P};
    const int byz = y.perm[0] == 2, by = !byz && y.perm[1] == 2;   /* output z / output y runs along source x */
    const int nb = byz ? XB : by ? XB : 1;
    float *v = malloc((size_t)XB * P * sizeof *v);
    uint16_t *ni = jitter && byz ? malloc((size_t)XB * p2 * sizeof *ni) : nullptr;   /* byz: the slab's noise as table indices */
    float *nr = jitter && !byz ? malloc((size_t)XB * P * sizeof *nr) : nullptr;
    if (!v || (jitter && byz && !ni) || (jitter && !byz && !nr)) abort();
    const int ztop = byz ? D : D, ystep = by ? XB : 1, zstep = byz ? XB : 1;
    for (int z0 = 0; z0 < ztop; z0 += zstep) {
        const int nz = byz ? (D - z0 < XB ? D - z0 : XB) : 1;
        if (ni) for (int zz = 0; zz < nz; zz++) for (int yy = 0; yy < P; yy++) {   /* write_x's draws, in its row order */
            uint16_t *q = ni + ((size_t)zz * P + yy) * P;
            for (int x = 0; x + 4 <= P; x += 4) { const uint64_t u = rnext(r); q[x] = (uint16_t)(u & 0xffff); q[x + 1] = (uint16_t)(u >> 16 & 0xffff); q[x + 2] = (uint16_t)(u >> 32 & 0xffff); q[x + 3] = (uint16_t)(u >> 48); }
            for (int x = P & ~3; x < P; x++) q[x] = (uint16_t)(rnext(r) & 0xffff);
        }
        for (int y0 = 0; y0 < P; y0 += ystep) {
            const int ny = by ? (P - y0 < XB ? P - y0 : XB) : 1, nrow = byz ? nz : ny;   /* rows gathered together */
            if (nr) for (int k = 0; k < ny; k++) {   /* (by / plain: the block's rows are consecutive in write_x's order) */
                float *q = nr + (size_t)k * P;
                for (int x = 0; x + 4 <= P; x += 4) { const uint64_t u = rnext(r); q[x] = g_ntab[u & 0xffff]; q[x + 1] = g_ntab[(u >> 16) & 0xffff]; q[x + 2] = g_ntab[(u >> 32) & 0xffff]; q[x + 3] = g_ntab[(u >> 48) & 0xffff]; }
                for (int x = P & ~3; x < P; x++) q[x] = g_ntab[rnext(r) & 0xffff];
            }
            /* source offset of row k's voxel x: q0 + k * kstep + x * qs */
            const int io[3] = {z0, y0, 0}; int s0[3], ds[3] = {0, 0, 0}, dk[3] = {0, 0, 0};
            for (int d = 0; d < 3; d++) s0[y.perm[d]] = y.flip[d] ? nd[d] - 1 - io[d] : io[d];
            ds[y.perm[2]] = y.flip[2] ? -1 : 1;
            const int kd = byz ? 0 : 1; dk[y.perm[kd]] = y.flip[kd] ? -1 : 1;
            const ptrdiff_t q0 = ((ptrdiff_t)s0[0] * P + s0[1]) * P + s0[2], qs = ((ptrdiff_t)ds[0] * P + ds[1]) * P + ds[2], ks = ((ptrdiff_t)dk[0] * P + dk[1]) * P + dk[2];
            if (nrow == 1 && qs == 1) { const uint8_t *src = ctu + q0; for (int x = 0; x < P; x++) v[x] = ((float)src[x] - fmean) * fisd; }
            else if (nrow == 1) { const uint8_t *src = ctu + q0; for (int x = 0; x < P; x++) v[x] = ((float)src[qs * x] - fmean) * fisd; }
            else if (ks == 1) for (int x = 0; x < P; x++) { const uint8_t *src = ctu + q0 + qs * x; for (int k = 0; k < nrow; k++) v[(size_t)k * P + x] = ((float)src[k] - fmean) * fisd; }
            else for (int x = 0; x < P; x++) { const uint8_t *src = ctu + q0 + qs * x; for (int k = 0; k < nrow; k++) v[(size_t)k * P + x] = ((float)src[-k] - fmean) * fisd; }
            for (int k = 0; k < nrow; k++) {
                float *vk = v + (size_t)k * P;
                const int z = byz ? z0 + k : z0, yy = byz ? y0 : y0 + k;
                if (jitter) {
                    if (ni) { const uint16_t *q = ni + ((size_t)k * P + yy) * P; for (int x = 0; x < P; x++) vk[x] = ia * vk[x] + ib + isg * g_ntab[q[x]]; }
                    else { const float *q = nr + (size_t)k * P; for (int x = 0; x < P; x++) vk[x] = ia * vk[x] + ib + isg * q[x]; }
                }
                uint16_t *hd = H + (size_t)z * p2 + (size_t)yy * P;
                if (xfmt == 1) f16_row(vk, hd, P); else for (int x = 0; x < P; x++) hd[x] = bf16_rn(vk[x]);
            }
        }
    }
    free(v); free(ni); free(nr);
}
static void write_x(const uint8_t *ctu, int D, int P, const int64_t o[3], float fmean, float fisd, int hasax, const float *cyz, const float *cxz,
                    sym y, int jitter, float ia, float ib, float isg, rng *r, float *nrow, int xfmt, float *X, uint16_t *H,
                    const ct_aug_plan *ap, float *noise_row_state, float *noise_plane_state, int ct_only) {   /* ap: the CT appearance plan (cubes, D == P), else nullptr;
                                                                                                    ct_only (--gpu-input): channel 0 only (the same rng draws) */
    pthread_once(&g_ntab_once, ntab_init);
    if (ct_only && !ap && xfmt && !getenv("UFSM_WRITE_X_ROWS")) { write_x_ct(ctu, D, P, fmean, fisd, y, jitter, ia, ib, isg, r, xfmt, H); return; }
    if (ct_only) hasax = 0;   /* (the radial rows are not written) */
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
            for (int ch = 0; ch < (ct_only ? 1 : 4); ch++) {
                const float *vc = v[ch];
                if (xfmt == 1) f16_row(vc, H + (size_t)ch * p3 + ko, P);
                else if (xfmt == 2) { uint16_t *hd = H + (size_t)ch * p3 + ko; for (int x = 0; x < P; x++) hd[x] = bf16_rn(vc[x]); }
                else memcpy(X + (size_t)ch * p3 + ko, vc, (size_t)P * sizeof *vc);
            }
        }
    free(a);
}
/* --gpu-input: the radial record of a write_x window (nn.h NN_RAD_REC: header, then the axis centre per source z) */
static void rad_record(void *rec, const int64_t o[3], int D, int P, int hasax, sym y, const float *cyz, const float *cxz, const spatial_aug *a) {
    const int warped = a && a->active;   /* write_spatial windows (cubes): the warp follows the axis centres (nn.h NN_RAD_REC) */
    int32_t h[8] = {(int32_t)o[1], (int32_t)o[2], D, P, hasax != 0,
                    y.perm[0] | y.perm[1] << 2 | y.perm[2] << 4 | (y.flip[0] != 0) << 8 | (y.flip[1] != 0) << 9 | (y.flip[2] != 0) << 10,
                    warped ? 1 | (a->affine != 0) << 1 : 0, 0};
    memset(rec, 0, NN_RAD_REC(D));   /* (unused words zero: records hash and upload deterministically) */
    memcpy(rec, h, sizeof h);
    memcpy((char *)rec + sizeof h, cyz, (size_t)D * sizeof *cyz);
    memcpy((char *)rec + sizeof h + (size_t)D * sizeof *cyz, cxz, (size_t)D * sizeof *cxz);
    if (!warped) return;
    if (D != P || a->P != P) { fprintf(stderr, "sampler: warped --gpu-input windows are cubes\n"); abort(); }
    float *w = (float *)((char *)rec + sizeof h + 2 * (size_t)D * sizeof *cyz);
    spatial_aug_tables(a, w); w += 6 * (size_t)P;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { w[3 * i + j] = (float)a->R[i][j]; w[9 + 3 * i + j] = (float)a->A[i][j]; w[18 + 3 * i + j] = (float)a->Ainv[i][j]; }
    w[27] = (float)a->scale;
}
/* the scroll-axis record (sample.h SAMPLE_AXR, unet.h UNET_AXR) of a window: the axis points (source z k, cyz[k], cxz[k]) through the
   inverse symmetry and the warp's forward map into window coordinates; the record's dim ad is the window dim the axis runs along most,
   its centres per window plane along ad interpolated (extrapolated past the ends) along that polyline */
static void axis_record(float *r, const int64_t o[3], int D, int P, int hasax, sym y, const float *cyz, const float *cxz, const spatial_aug *a,
                        int div) {
    const int warped = a && a->active, se[3] = {D, P, P}, np0 = D > P ? D : P;
    memset(r, 0, SAMPLE_AXR(np0) * sizeof *r);
    r[3] = (float)(div > 0 ? div : 1);
    int ext[3]; for (int d = 0; d < 3; d++) ext[d] = se[y.perm[d]];
    r[2] = (float)ext[0];
    if (!hasax || D < 2) return;
    double (*w)[3] = malloc((size_t)D * sizeof *w);
    if (!w) abort();
    for (int k = 0; k < D; k++) {
        const double v[3] = {k, (double)cyz[k] - (double)o[1], (double)cxz[k] - (double)o[2]};
        double u[3];
        for (int d = 0; d < 3; d++) u[d] = y.flip[d] ? se[y.perm[d]] - 1 - v[y.perm[d]] : v[y.perm[d]];
        if (warped) spatial_aug_forward(a, u, w[k]); else memcpy(w[k], u, sizeof u);
    }
    int ad = 0; double best = -1, q[2][3];   /* ad: the window dim nearest the image of the source z direction (not the axis polyline,
                                                whose in-plane drift may be anything) */
    for (int e = 0; e < 2; e++) {
        const double v[3] = {e ? D - 1 : 0, 0.5 * (P - 1), 0.5 * (P - 1)}; double u[3];
        for (int d = 0; d < 3; d++) u[d] = y.flip[d] ? se[y.perm[d]] - 1 - v[y.perm[d]] : v[y.perm[d]];
        if (warped) spatial_aug_forward(a, u, q[e]); else memcpy(q[e], u, sizeof u);
    }
    for (int d = 0; d < 3; d++) { const double e = fabs(q[1][d] - q[0][d]); if (e > best) { best = e; ad = d; } }
    const int du = ad == 0 ? 1 : 0, dv = ad == 2 ? 1 : 2, np = ext[ad], dir = w[D - 1][ad] >= w[0][ad] ? 1 : -1;
    r[0] = (float)ad; r[1] = 1.f; r[2] = (float)np;
    int k = 0;   /* the polyline segment k..k+1 holding plane t (k walks forward: the planes in the polyline's direction along ad) */
    for (int i = 0; i < np; i++) {
        const int t = dir > 0 ? i : np - 1 - i; const double tt = dir > 0 ? t : -(double)t;
        while (k + 2 < D && (dir > 0 ? w[k + 1][ad] : -w[k + 1][ad]) < tt) k++;
        const double a0 = w[k][ad], a1 = w[k + 1][ad], f = a1 != a0 ? (t - a0) / (a1 - a0) : 0;
        r[4 + 2 * t] = (float)(w[k][du] + f * (w[k + 1][du] - w[k][du]));
        r[5 + 2 * t] = (float)(w[k][dv] + f * (w[k + 1][dv] - w[k][dv]));
    }
    free(w);
}
void sample_axis_record(float *r, const int64_t o[3], int D, int P, int hasax, const int perm[3], const int flip[3], const float *cyz,
                        const float *cxz, int div) {
    sym y; for (int d = 0; d < 3; d++) { y.perm[d] = perm[d]; y.flip[d] = flip[d]; }
    axis_record(r, o, D, P, hasax, y, cyz, cxz, nullptr, div);
}
void sample_axis_record_spatial_test(float *r, const int64_t o[3], int P, int hasax, const int perm[3], const int flip[3], const float *cyz,
                                     const float *cxz, uint64_t aug_seed, float zoom, int affine) {
    sym y; for (int d = 0; d < 3; d++) { y.perm[d] = perm[d]; y.flip[d] = flip[d]; }
    spatial_aug a; spatial_aug_make(&a, aug_seed, P, 10, 1, 1, 1);   /* (as sample_write_spatial_test) */
    if (zoom > 0 && zoom != 1) spatial_aug_zoom(&a, zoom);
    if (affine) spatial_aug_affine(&a, aug_seed ^ 0x9e37u, .1, .06);
    axis_record(r, o, P, P, hasax, y, cyz, cxz, &a, 1);
}
int sample_gpu_input_ok(const sample_cfg *c) {
    const int Pz = c->Pz > 0 ? c->Pz : c->P;
    return c->xfmt != 0 && !c->aux && !c->cond && !c->sheet && !(c->augment && c->ct_augment && Pz != c->P);   /* warps: the CT channel of
                                                                                   write_spatial, the transported radial vector on the device */
}
void sample_write_x_test(const uint8_t *ctu, int D, int P, const int64_t o[3], float mean, float isd, int hasax, const float *cyz,
                         const float *cxz, const int perm[3], const int flip[3], int xfmt, uint64_t seed, uint16_t *H4, uint16_t *H1, void *rec) {
    sym y; for (int d = 0; d < 3; d++) { y.perm[d] = perm[d]; y.flip[d] = flip[d]; }
    float *nrow = malloc((size_t)P * sizeof *nrow);
    rng r;
    rseed(&r, seed); write_x(ctu, D, P, o, mean, isd, hasax, cyz, cxz, y, 1, 1.1f, 0.05f, 0.02f, &r, nrow, xfmt, nullptr, H4, nullptr, nullptr, nullptr, 0);
    rseed(&r, seed); write_x(ctu, D, P, o, mean, isd, hasax, cyz, cxz, y, 1, 1.1f, 0.05f, 0.02f, &r, nrow, xfmt, nullptr, H1, nullptr, nullptr, nullptr, 1);
    rad_record(rec, o, D, P, hasax, y, cyz, cxz, nullptr);
    free(nrow);
}
int sample_write_spatial_test(const uint8_t *ctu, const uint8_t *tgt, const uint8_t *mask, int P, const int64_t o[3], float mean, float isd,
                              int hasax, const float *cyz, const float *cxz, const int perm[3], const int flip[3], int xfmt, uint64_t seed,
                              uint64_t aug_seed, float zoom, int affine, int hard, uint16_t *H4, uint16_t *H1, void *rec, uint8_t *T, uint8_t *M);
/* Continuous geometry uses source coordinates for CT, labels, q0 and the axis.
   Only valid source voxels are supervised: rotated cube corners are ignored.
   The four input channels stay unchanged; a winding task uses q0 and the two
   Y/X projections of the transported radial vector. hard: every target takes
   the code of the nearest source voxel (the trilinear corner of largest
   weight; hard classes such as the side encoding stay 0 / 255, packable).
   ct_only (--gpu-input): channel 0 alone into H (1 channel per sample), the
   same rng draws; the radial vector follows on the device (rad_record). */
[[maybe_unused]] static void write_spatial_ref(const uint8_t *ctu,const uint8_t *target,const uint8_t *mask,
                         uint8_t *T,uint8_t *M,int P,const int64_t o[3],float mean,float isd,
                         int hasax,const float *cy,const float *cx,sym sy,const spatial_aug *a,
                         float gain,float offset,float noise,rng *r,float *nrow,int xfmt,float *X,uint16_t *H,
                         const ct_aug_plan *ap,float *noise_row,float *noise_plane,
                         const sheet_dataset *sheet,const double (*params)[4],float *appearance,int hard,int ct_only) {
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
                const size_t bj=((size_t)(base[0]+(f[0]>.5f))*P+base[1]+(f[1]>.5f))*P+base[2]+(f[2]>.5f);   /* hard: the nearest source voxel */
                for (int bz=0;bz<2;bz++) for (int by=0;by<2;by++) for (int bx=0;bx<2;bx++) {
                    float weight=(bz?f[0]:1-f[0])*(by?f[1]:1-f[1])*(bx?f[2]:1-f[2]);
                    if (weight<=1e-7f) continue;
                    size_t j=((size_t)(base[0]+bz)*P+base[1]+by)*P+base[2]+bx;
                    valid &= mask[j]!=0;
                    float val=ap && ap->filter_mode?appearance[j]:ap?ap->lut[ctu[j]]:((float)ctu[j]-mean)*isd;
                    cv+=weight*val;
                    if (!hard) for (int ch=0;ch<NCH;ch++) tv[ch]+=weight*target[(size_t)ch*p3+j];
                }
                if (ap && !ap->filter_mode && P>1) for (int d=0;d<3;d++) cv+=ap->shading[d]*(2*v[d]/(P-1)-1);
                out[0]=gain*cv+offset+noise*nrow[x]; M[k]=(uint8_t)valid;
                if (hard) for (int ch=0;ch<NCH;ch++) T[(size_t)ch*p3+k]=target[(size_t)ch*p3+bj];
                else for (int ch=0;ch<NCH;ch++) T[(size_t)ch*p3+k]=(uint8_t)lroundf(tv[ch]);
                if (!ct_only) {
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
            }
            for (int ch=0;ch<(ct_only?1:4);ch++) {
                size_t j=(size_t)ch*p3+k;
                if (xfmt==1) { _Float16 h=(_Float16)out[ch]; memcpy(H+j,&h,2); }
                else if (xfmt==2) H[j]=bf16_rn(out[ch]); else X[j]=out[ch];
            }
        }
    }
    free(table);
}
static void *tp_get(size_t n, int zero);
static void tp_put(void *p, size_t n);
/* TM (packed targets, hard and CT-only only; T, M unused): the code bytes of sample_pack_tm written straight from the warp (no
   T / M window buffers); when the targets turn out not to be 0 / 255 or the writer's fast path does not apply, T / M go to pooled
   scratch and are packed after, aborting as draw() did */
static void write_spatial_tm(const uint8_t *ctu,const uint8_t *target,const uint8_t *mask,
                         uint8_t *T,uint8_t *M,int P,const int64_t o[3],float mean,float isd,
                         int hasax,const float *cy,const float *cx,sym sy,const spatial_aug *a,
                         float gain,float offset,float noise,rng *r,float *nrow,int xfmt,float *X,uint16_t *H,
                         const ct_aug_plan *ap,float *noise_row,float *noise_plane,
                         const sheet_dataset *sheet,const double (*params)[4],float *appearance,int hard,int ct_only,uint8_t *TM) {
    pthread_once(&g_ntab_once,ntab_init);
    size_t p3=(size_t)P*P*P; float *table=malloc(6*(size_t)P*sizeof *table);
    if (!table) abort();
    spatial_aug_tables(a,table);
    if (ap && ap->filter_mode) {
        for (int z=0;z<P;z++) for (int y=0;y<P;y++) for (int x=0;x<P;x++)
            appearance[((size_t)z*P+y)*P+x]=ct_aug_value(ap,ctu,P,z,y,x);
    }
    const float sign[3]={sy.flip[0]?-1:1,sy.flip[1]?-1:1,sy.flip[2]?-1:1};
    enum { ZB = 16, TB = 32 };   /* (row tiles of 32: the per-row coordinate setup amortised over 4 vectors) */
    const float c=.5f*(a->P-1),sc=(float)a->scale;
    float Rf[3][3],Af[3][3],Aif[3][3];
    for (int i=0;i<3;i++) for (int j=0;j<3;j++) { Rf[i][j]=(float)a->R[i][j]; Af[i][j]=(float)a->A[i][j]; Aif[i][j]=(float)a->Ainv[i][j]; }
    /* (the z-block scratch pooled per worker: a fresh 16 + 4 MB per window was an mmap / munmap pair and a TLB shootdown) */
    float *vt=malloc(6*(size_t)P*sizeof *vt), *cvb=tp_get((size_t)ZB*P*P*sizeof *cvb,0); uint8_t *inb=tp_get((size_t)ZB*P*P,0);
    if (!vt || !cvb || !inb) abort();
    for (int d=0;d<3;d++) for (int u=0;u<P;u++) {   /* spatial_aug_pull's per-axis terms */
        vt[d*P+u]=u-c-table[d*P+u];
        vt[(3+d)*P+u]=1.f/(1.f-table[(3+d)*P+u]);
    }
    /* ((float)ctu - mean) * isd (or the plan's LUT) tabulated per value: the same floats */
    float lutv[256]; for (int i=0;i<256;i++) lutv[i]=ap?(ap->filter_mode?0:ap->lut[i]):((float)i-mean)*isd;
#ifdef __AVX2__
    /* the trainer's case (hard targets, CT-only, P a multiple of 16): 8 voxels at a time with AVX2 gathers of the CT values (as floats in
       the appearance scratch), mask and target bytes (aligned 32-bit words, in bounds: P^3 % 4 == 0); the same float operations in the
       same order (no contraction), bit for bit */
    const int fast=hard && ct_only && P%16==0 && NCH==2 && !((uintptr_t)mask&3) && !((uintptr_t)target&3) && !((uintptr_t)ctu&3);
    const int filt_fast=ap && ap->filter_mode;   /* (CT values: the appearance floats; else the CT byte through lutv, L1-resident) */
    /* no appearance plan (the trainer's default): the CT byte and the mask bit of each voxel as one 16-bit word in the (otherwise
       unused) appearance scratch, so one 32-bit gather serves an x pair of corners (4 gathers per 8 voxels instead of 24), and the
       CT value is lutv's float ((float) v - mean) * isd computed in the lanes (the same two float operations) */
    /* with an appearance plan (--ct-aug) the same words: the CT value is its LUT entry (a gather from lutv, L1-resident) or, with a
       filter, the appearance float (the words then in pooled scratch beside it) */
    const int cmf=fast && appearance;
    uint16_t *cm=cmf?(filt_fast?tp_get(2*p3,0):(uint16_t *)appearance):nullptr;
    if (cmf && !cm) abort();
    /* hard 0 / 255 targets (what the packed batch requires) ride along as bits 9 and 10: the nearest corner's targets come from the
       same pair words (else two more gathers per 8 voxels) */
    int cmt=cmf;
    if (cmf) {
        uint8_t odd=0;
        for (size_t k=0;k<p3;k++) {
            const uint8_t t0=target[k],t1=target[p3+k];
            odd|=(uint8_t)((t0+1)&0xfe)|(uint8_t)((t1+1)&0xfe);   /* (nonzero unless both are 0 or 255) */
            cm[k]=(uint16_t)(ctu[k]|(mask[k]?0x100:0)|(t0?0x200:0)|(t1?0x400:0));
        }
        cmt=!odd;
    }
    const int direct=TM && cmt;   /* (cmt implies hard, CT-only, the fast path) */
#else
    const int fast=0, direct=0;
#endif
    uint8_t *Tloc=nullptr;
    if (TM && !direct) { if (!(Tloc=tp_get((NCH+1)*p3,0))) abort(); T=Tloc; M=Tloc+NCH*p3; }
    for (int z0=0;z0<P;z0+=ZB) {
        const int nz=P-z0<ZB?P-z0:ZB;
        for (int y0=0;y0<P;y0+=TB) for (int x0=0;x0<P;x0+=TB) {
            const int ny=P-y0<TB?P-y0:TB, nx=P-x0<TB?P-x0:TB;
            for (int z=z0;z<z0+nz;z++) for (int y=y0;y<y0+ny;y++) {
                const float v0=vt[z],v1=vt[P+y],j0=vt[3*P+z],j1=vt[4*P+y];
                float t01[3];
                for (int d=0;d<3;d++) t01[d]=Rf[0][d]*v0+Rf[1][d]*v1;   /* (the first two terms of the pull's sum) */
#ifdef __AVX2__
                if (fast) {
                    const __m256 one=_mm256_set1_ps(1.f),fP1=_mm256_set1_ps((float)(P-1)),zero=_mm256_setzero_ps(),thr=_mm256_set1_ps(1e-7f),half=_mm256_set1_ps(.5f);
                    const __m256i three=_mm256_set1_epi32(3),bytem=_mm256_set1_epi32(255),iP=_mm256_set1_epi32(P);
                    for (int h=0;h<nx;h+=8) {
                        const int xs=x0+h;
                        const __m256 v2=_mm256_loadu_ps(vt+2*P+xs);
                        __m256 U[3],Vv[3],F[3],G[3];
                        if (a->affine) {
                            __m256 w[3];
                            for (int d=0;d<3;d++) w[d]=_mm256_add_ps(_mm256_set1_ps(t01[d]),_mm256_mul_ps(_mm256_set1_ps(Rf[2][d]),v2));
                            for (int d=0;d<3;d++) U[d]=_mm256_add_ps(_mm256_set1_ps(c),_mm256_mul_ps(_mm256_set1_ps(sc),_mm256_add_ps(_mm256_add_ps(
                                _mm256_mul_ps(_mm256_set1_ps(Af[d][0]),w[0]),_mm256_mul_ps(_mm256_set1_ps(Af[d][1]),w[1])),_mm256_mul_ps(_mm256_set1_ps(Af[d][2]),w[2]))));
                        } else for (int d=0;d<3;d++) U[d]=_mm256_add_ps(_mm256_set1_ps(c),_mm256_mul_ps(_mm256_set1_ps(sc),_mm256_add_ps(_mm256_set1_ps(t01[d]),_mm256_mul_ps(_mm256_set1_ps(Rf[2][d]),v2))));
                        for (int d=0;d<3;d++) Vv[sy.perm[d]]=sy.flip[d]?_mm256_sub_ps(fP1,U[d]):U[d];
                        __m256 in=_mm256_and_ps(_mm256_cmp_ps(Vv[0],zero,_CMP_GE_OQ),_mm256_cmp_ps(Vv[0],fP1,_CMP_LE_OQ));
                        for (int d=1;d<3;d++) in=_mm256_and_ps(in,_mm256_and_ps(_mm256_cmp_ps(Vv[d],zero,_CMP_GE_OQ),_mm256_cmp_ps(Vv[d],fP1,_CMP_LE_OQ)));
                        const __m256i ini=_mm256_castps_si256(in);
                        __m256i Bv[3];
                        for (int d=0;d<3;d++) {
                            Bv[d]=_mm256_and_si256(_mm256_min_epi32(_mm256_cvttps_epi32(Vv[d]),_mm256_set1_epi32(P-2)),ini);
                            F[d]=_mm256_sub_ps(Vv[d],_mm256_cvtepi32_ps(Bv[d])); G[d]=_mm256_sub_ps(one,F[d]);
                        }
                        const __m256i b0=_mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(Bv[0],iP),Bv[1]),iP),Bv[2]);
                        __m256 cv=zero; __m256i bad=_mm256_setzero_si256(), wdp[4];
                        if (cmf) for (int pr=0;pr<4;pr++) {   /* corners (bz, by, 0) and (bz, by, 1): the per-voxel order cn = 0..7 */
                            const int bz=pr>>1,by=pr&1;
                            const __m256i wd=wdp[pr]=_mm256_mask_i32gather_epi32(_mm256_setzero_si256(),(const int *)cm,_mm256_add_epi32(b0,_mm256_set1_epi32((bz*P+by)*P)),ini,2);
                            const __m256 wzy=_mm256_mul_ps(bz?F[0]:G[0],by?F[1]:G[1]);   /* (the weight's first product, shared by the pair) */
                            const __m256i zm[2]={_mm256_cmpeq_epi32(_mm256_and_si256(wd,_mm256_set1_epi32(0x100)),_mm256_setzero_si256()),
                                                 _mm256_cmpeq_epi32(_mm256_and_si256(wd,_mm256_set1_epi32(0x1000000)),_mm256_setzero_si256())};   /* mask bit 0 */
                            for (int bx=0;bx<2;bx++) {
                                const __m256 w=_mm256_mul_ps(wzy,bx?F[2]:G[2]);
                                const __m256 use=_mm256_and_ps(in,_mm256_cmp_ps(w,thr,_CMP_GT_OQ));
                                const __m256i h16=bx?_mm256_srli_epi32(wd,16):wd;
                                const __m256 val=filt_fast?_mm256_mask_i32gather_ps(zero,appearance,_mm256_add_epi32(b0,_mm256_set1_epi32((bz*P+by)*P+bx)),use,4)
                                                :ap?_mm256_mask_i32gather_ps(zero,lutv,_mm256_and_si256(h16,bytem),use,4)
                                                :_mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_and_si256(h16,bytem)),_mm256_set1_ps(mean)),_mm256_set1_ps(isd));
                                cv=_mm256_add_ps(cv,_mm256_and_ps(_mm256_mul_ps(w,val),use));
                                bad=_mm256_or_si256(bad,_mm256_and_si256(zm[bx],_mm256_castps_si256(use)));
                            }
                        }
                        else for (int cn=0;cn<8;cn++) {
                            const int bz=cn>>2,by=cn>>1&1,bx=cn&1;
                            const __m256 w=_mm256_mul_ps(_mm256_mul_ps(bz?F[0]:G[0],by?F[1]:G[1]),bx?F[2]:G[2]);
                            const __m256 use=_mm256_and_ps(in,_mm256_cmp_ps(w,thr,_CMP_GT_OQ));
                            const __m256i j=_mm256_add_epi32(b0,_mm256_set1_epi32((bz*P+by)*P+bx));
                            __m256 val;
                            if (filt_fast) val=_mm256_mask_i32gather_ps(zero,appearance,j,use,4);
                            else {
                                const __m256i cw=_mm256_mask_i32gather_epi32(_mm256_setzero_si256(),(const int *)ctu,_mm256_srli_epi32(j,2),_mm256_castps_si256(use),4);
                                const __m256i cb=_mm256_and_si256(_mm256_srlv_epi32(cw,_mm256_slli_epi32(_mm256_and_si256(j,three),3)),bytem);
                                val=_mm256_mask_i32gather_ps(zero,lutv,cb,use,4);
                            }
                            cv=_mm256_add_ps(cv,_mm256_and_ps(_mm256_mul_ps(w,val),use));
                            const __m256i word=_mm256_mask_i32gather_epi32(_mm256_setzero_si256(),(const int *)mask,_mm256_srli_epi32(j,2),_mm256_castps_si256(use),4);
                            const __m256i byte=_mm256_and_si256(_mm256_srlv_epi32(word,_mm256_slli_epi32(_mm256_and_si256(j,three),3)),bytem);
                            bad=_mm256_or_si256(bad,_mm256_and_si256(_mm256_cmpeq_epi32(byte,_mm256_setzero_si256()),_mm256_castps_si256(use)));
                        }
                        if (ap && !ap->filter_mode) for (int d=0;d<3;d++)
                            cv=_mm256_add_ps(cv,_mm256_mul_ps(_mm256_set1_ps(ap->shading[d]),_mm256_sub_ps(_mm256_div_ps(_mm256_mul_ps(_mm256_set1_ps(2.f),Vv[d]),fP1),one)));
                        const __m256 nz_=_mm256_cmp_ps(F[0],half,_CMP_GT_OQ),ny_=_mm256_cmp_ps(F[1],half,_CMP_GT_OQ),nx_=_mm256_cmp_ps(F[2],half,_CMP_GT_OQ);
                        __m256i t0,t1;
                        if (cmt) {   /* the nearest corner's pair word (bz, by from the fractions), its half by fx, bits 9 / 10 as 0 / 255 */
                            const __m256i w0=_mm256_blendv_epi8(wdp[0],wdp[1],_mm256_castps_si256(ny_)),w1=_mm256_blendv_epi8(wdp[2],wdp[3],_mm256_castps_si256(ny_));
                            const __m256i ws=_mm256_blendv_epi8(w0,w1,_mm256_castps_si256(nz_));
                            const __m256i hw=_mm256_blendv_epi8(ws,_mm256_srli_epi32(ws,16),_mm256_castps_si256(nx_));
                            const __m256i one1=_mm256_set1_epi32(1);
                            t0=_mm256_mullo_epi32(_mm256_and_si256(_mm256_srli_epi32(hw,9),one1),bytem);
                            t1=_mm256_mullo_epi32(_mm256_and_si256(_mm256_srli_epi32(hw,10),one1),bytem);
                        } else {
                            const __m256i jn=_mm256_add_epi32(b0,_mm256_add_epi32(_mm256_add_epi32(
                                _mm256_and_si256(_mm256_castps_si256(nz_),_mm256_set1_epi32(P*P)),
                                _mm256_and_si256(_mm256_castps_si256(ny_),iP)),
                                _mm256_and_si256(_mm256_castps_si256(nx_),_mm256_set1_epi32(1))));
                            const __m256i sh=_mm256_slli_epi32(_mm256_and_si256(jn,three),3), wj=_mm256_srli_epi32(jn,2);
                            t0=_mm256_and_si256(_mm256_srlv_epi32(_mm256_mask_i32gather_epi32(_mm256_setzero_si256(),(const int *)target,wj,ini,4),sh),bytem);
                            t1=_mm256_and_si256(_mm256_srlv_epi32(_mm256_mask_i32gather_epi32(_mm256_setzero_si256(),(const int *)(target+p3),wj,ini,4),sh),bytem);
                        }
                        const size_t k=((size_t)z*P+y)*P+xs, kb=((size_t)(z-z0)*P+y)*P+xs;
                        _mm256_storeu_ps(cvb+kb,cv);
                        /* inside 0 / 1, mask inside and no zero corner, targets (the gathers are 0 outside): 8 bytes each */
                        const __m256i inv=_mm256_and_si256(ini,_mm256_set1_epi32(1)), mv=_mm256_andnot_si256(bad,inv);
                        const uint64_t q0=pack8_u32(inv); memcpy(inb+kb,&q0,8);
                        if (direct) {   /* sample_pack_tm's code: t0 >> 7 & T0 | t1 >> 6 & T1 | m << 2 & M */
                            const __m256i code=_mm256_or_si256(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi32(t0,7),_mm256_set1_epi32(SAMPLE_TM_T0)),
                                _mm256_and_si256(_mm256_srli_epi32(t1,6),_mm256_set1_epi32(SAMPLE_TM_T1))),_mm256_and_si256(_mm256_slli_epi32(mv,2),_mm256_set1_epi32(SAMPLE_TM_M)));
                            const uint64_t qc=pack8_u32(code); memcpy(TM+k,&qc,8);
                        } else {
                            const uint64_t q1=pack8_u32(mv),q2=pack8_u32(t0),q3=pack8_u32(t1);
                            memcpy(M+k,&q1,8); memcpy(T+k,&q2,8); memcpy(T+p3+k,&q3,8);
                        }
                    }
                    continue;
                }
#endif
                /* the row segment in stages: coordinates, corners and weights over its lanes (vectorisable), then the corner
                   loads corner by corner (each lane accumulates its corners in the per-voxel order), then the outputs */
                float V[3][TB],Fq[3][TB],W[8][TB],CV[TB],TVc[NCH][TB]; size_t B0[TB]; uint8_t IN[TB],VA[TB];
                {
                    float U[3][TB];
                    if (a->affine) for (int i=0;i<nx;i++) {
                        const float v2=vt[2*P+x0+i]; float w[3];
                        for (int d=0;d<3;d++) w[d]=t01[d]+Rf[2][d]*v2;
                        for (int d=0;d<3;d++) U[d][i]=c+sc*(Af[d][0]*w[0]+Af[d][1]*w[1]+Af[d][2]*w[2]);
                    } else for (int d=0;d<3;d++) { const float t=t01[d],r2=Rf[2][d]; for (int i=0;i<nx;i++) U[d][i]=c+sc*(t+r2*vt[2*P+x0+i]); }
                    for (int d=0;d<3;d++) { const int pd=sy.perm[d]; if (sy.flip[d]) for (int i=0;i<nx;i++) V[pd][i]=P-1-U[d][i]; else for (int i=0;i<nx;i++) V[pd][i]=U[d][i]; }
                    int Bi[3][TB];
                    for (int i=0;i<nx;i++) IN[i]=V[0][i]>=0 && V[0][i]<=P-1 && V[1][i]>=0 && V[1][i]<=P-1 && V[2][i]>=0 && V[2][i]<=P-1;
                    for (int d=0;d<3;d++) for (int i=0;i<nx;i++) { int b=IN[i]?(int)V[d][i]:0; if (b>=P-1) b=P-2; Bi[d][i]=b; Fq[d][i]=V[d][i]-b; }
                    for (int i=0;i<nx;i++) B0[i]=((size_t)Bi[0][i]*P+Bi[1][i])*P+Bi[2][i];
                    for (int cn=0;cn<8;cn++) { const int bz=cn>>2,by=cn>>1&1,bx=cn&1; for (int i=0;i<nx;i++) W[cn][i]=(bz?Fq[0][i]:1-Fq[0][i])*(by?Fq[1][i]:1-Fq[1][i])*(bx?Fq[2][i]:1-Fq[2][i]); }
                }
                for (int i=0;i<nx;i++) { CV[i]=0; VA[i]=1; for (int ch=0;ch<NCH;ch++) TVc[ch][i]=0; }
                const int filt=ap && ap->filter_mode;
                for (int cn=0;cn<8;cn++) {
                    const size_t off=((size_t)(cn>>2)*P+(cn>>1&1))*P+(cn&1);
                    for (int i=0;i<nx;i++) {
                        const float weight=W[cn][i];
                        if (!IN[i] || weight<=1e-7f) continue;
                        const size_t j=B0[i]+off;
                        VA[i]&=mask[j]!=0;
                        CV[i]+=weight*(filt?appearance[j]:lutv[ctu[j]]);
                        if (!hard) for (int ch=0;ch<NCH;ch++) TVc[ch][i]+=weight*target[(size_t)ch*p3+j];
                    }
                }
                for (int i=0;i<nx;i++) {
                    const int x=x0+i;
                    const size_t k=((size_t)z*P+y)*P+x, kb=((size_t)(z-z0)*P+y)*P+x;
                    const int inside=IN[i];
                    inb[kb]=(uint8_t)inside;
                    if (!inside) {
                        for (int ch=0;ch<NCH;ch++) T[(size_t)ch*p3+k]=0;
                        M[k]=0;
                        if (!ct_only) for (int ch=1;ch<4;ch++) { size_t j=(size_t)ch*p3+k; if (xfmt==1) { _Float16 h=(_Float16)0.f; memcpy(H+j,&h,2); } else if (xfmt==2) H[j]=bf16_rn(0.f); else X[j]=0.f; }
                        continue;
                    }
                    const float v[3]={V[0][i],V[1][i],V[2][i]},f[3]={Fq[0][i],Fq[1][i],Fq[2][i]};
                    int base[3]; for (int d=0;d<3;d++) { base[d]=(int)v[d]; if (base[d]>=P-1) base[d]=P-2; }
                    float cv=CV[i],tv[NCH]; const int valid=VA[i]; for (int ch=0;ch<NCH;ch++) tv[ch]=TVc[ch][i];
                    const size_t b0=B0[i];
                    if (ap && !ap->filter_mode && P>1) for (int d=0;d<3;d++) cv+=ap->shading[d]*(2*v[d]/(P-1)-1);
                    cvb[kb]=cv; M[k]=(uint8_t)valid;
                    if (hard) { const size_t bj=b0+((size_t)(f[0]>.5f)*P+(f[1]>.5f))*P+(f[2]>.5f); for (int ch=0;ch<NCH;ch++) T[(size_t)ch*p3+k]=target[(size_t)ch*p3+bj]; }
                    else for (int ch=0;ch<NCH;ch++) T[(size_t)ch*p3+k]=(uint8_t)lroundf(tv[ch]);
                    if (ct_only) continue;
                    const float jac[3]={j0,j1,vt[5*P+x]};
                    float out[4]={0};
                    float frac=f[0],ayy=cy[base[0]]*(1-frac)+cy[base[0]+1]*frac;
                    float axx=cx[base[0]]*(1-frac)+cx[base[0]+1]*frac;
                    float dy=(float)o[1]+v[1]-ayy,dx=(float)o[2]+v[2]-axx;
                    float radius=sqrtf(dy*dy+dx*dx)+1e-6f;
                    float radial[3]={0,hasax?dy/radius:0,hasax?dx/radius:0},discrete[3],transported[3];
                    for (int d=0;d<3;d++) discrete[d]=sign[d]*radial[sy.perm[d]];
                    if (a->affine) { float q[3]; for (int d=0;d<3;d++) q[d]=Aif[d][0]*discrete[0]+Aif[d][1]*discrete[1]+Aif[d][2]*discrete[2]; memcpy(discrete,q,sizeof q); }
                    float norm=1e-12f;
                    for (int d=0;d<3;d++) { transported[d]=jac[d]*(Rf[d][0]*discrete[0]+Rf[d][1]*discrete[1]+Rf[d][2]*discrete[2]); norm+=transported[d]*transported[d]; }
                    norm=sqrtf(norm); for (int d=0;d<3;d++) out[1+d]=transported[d]/norm;
                    if (sheet) {
                        double q[4]; for (int d=0;d<4;d++) q[d]=params[base[0]][d]*(1-(double)frac)+params[base[0]+1][d]*frac;
                        double prior=hypot(o[1]+(double)v[1]-q[0],o[2]+(double)v[2]-q[1])/q[2]+q[3];
                        out[1]=(float)((prior-sheet->center)/sheet->scale);
                    }
                    for (int ch=1;ch<4;ch++) {
                        size_t j=(size_t)ch*p3+k;
                        if (xfmt==1) { _Float16 h=(_Float16)out[ch]; memcpy(H+j,&h,2); }
                        else if (xfmt==2) H[j]=bf16_rn(out[ch]); else X[j]=out[ch];
                    }
                }
            }
        }
        for (int z=z0;z<z0+nz;z++) for (int y=0;y<P;y++) {   /* channel 0: the noise rows in the per-voxel loop's order */
            for (int x=0;x+4<=P;x+=4) { uint64_t v=rnext(r); for (int j=0;j<4;j++) nrow[x+j]=g_ntab[(v>>(16*j))&65535]; }
            for (int x=P&~3;x<P;x++) nrow[x]=g_ntab[rnext(r)&65535];
            if (ap && ap->noise_rho>0) ct_aug_noise_row(ap,nrow,noise_row,noise_plane,P,z,y);
            const size_t k0=((size_t)z*P+y)*P, kb0=((size_t)(z-z0)*P+y)*P;
            int x=0;   /* (the row in place, then one store pass) */
#ifdef __AVX2__
            {
                const __m256 vg=_mm256_set1_ps(gain),vo=_mm256_set1_ps(offset),vn=_mm256_set1_ps(noise);
                for (;x+8<=P;x+=8) {
                    uint64_t ib; memcpy(&ib,inb+kb0+x,8);
                    const __m256 keep=_mm256_castsi256_ps(_mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long)ib)),_mm256_setzero_si256()),_mm256_set1_epi32(-1)));
                    const __m256 o=_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(vg,_mm256_loadu_ps(cvb+kb0+x)),vo),_mm256_mul_ps(vn,_mm256_loadu_ps(nrow+x)));
                    _mm256_storeu_ps(nrow+x,_mm256_and_ps(o,keep));
                }
            }
#endif
            for (;x<P;x++) nrow[x]=inb[kb0+x]?gain*cvb[kb0+x]+offset+noise*nrow[x]:0.f;
            if (xfmt==1) f16_row(nrow,H+k0,P);
            else if (xfmt==2) for (int x=0;x<P;x++) H[k0+x]=bf16_rn(nrow[x]);
            else memcpy(X+k0,nrow,(size_t)P*sizeof *X);
        }
    }
    free(vt); tp_put(cvb,(size_t)ZB*P*P*sizeof *cvb); tp_put(inb,(size_t)ZB*P*P);
#ifdef __AVX2__
    if (cmf && filt_fast) tp_put(cm,2*p3);
#endif
    free(table);
    if (Tloc) {
        size_t bad=0;
        if (sample_pack_tm(T,T+p3,M,p3,TM,&bad)) {
            fprintf(stderr,"sampler: packed targets need hard 0 / 255 targets and a 0 / 1 mask after the warp (voxel %zu); train --pack-targets 0\n",bad);
            abort();
        }
        tp_put(Tloc,(NCH+1)*p3);
    }
}
static void write_spatial(const uint8_t *ctu,const uint8_t *target,const uint8_t *mask,
                         uint8_t *T,uint8_t *M,int P,const int64_t o[3],float mean,float isd,
                         int hasax,const float *cy,const float *cx,sym sy,const spatial_aug *a,
                         float gain,float offset,float noise,rng *r,float *nrow,int xfmt,float *X,uint16_t *H,
                         const ct_aug_plan *ap,float *noise_row,float *noise_plane,
                         const sheet_dataset *sheet,const double (*params)[4],float *appearance,int hard,int ct_only) {
    write_spatial_tm(ctu,target,mask,T,M,P,o,mean,isd,hasax,cy,cx,sy,a,gain,offset,noise,r,nrow,xfmt,X,H,ap,noise_row,noise_plane,sheet,params,appearance,hard,ct_only,nullptr);
}
int sample_write_spatial_test(const uint8_t *ctu, const uint8_t *tgt, const uint8_t *mask, int P, const int64_t o[3], float mean, float isd,
                              int hasax, const float *cyz, const float *cxz, const int perm[3], const int flip[3], int xfmt, uint64_t seed,
                              uint64_t aug_seed, float zoom, int affine, int hard, uint16_t *H4, uint16_t *H1, void *rec, uint8_t *T, uint8_t *M) {
    sym y; for (int d = 0; d < 3; d++) { y.perm[d] = perm[d]; y.flip[d] = flip[d]; }
    spatial_aug a; spatial_aug_make(&a, aug_seed, P, 10, 1, 1, 1);
    if (zoom > 0 && zoom != 1) spatial_aug_zoom(&a, zoom);
    if (affine) spatial_aug_affine(&a, aug_seed ^ 0x9e37u, .1, .06);
    const size_t p3 = (size_t)P * P * P;
    float *nrow = malloc((size_t)P * sizeof *nrow), *app = malloc(p3 * sizeof *app);
    uint8_t *T4 = malloc(NCH * p3), *M4 = malloc(p3);
    if (!nrow || !app || !T4 || !M4) abort();
    rng r;
    rseed(&r, seed); write_spatial(ctu, tgt, mask, T4, M4, P, o, mean, isd, hasax, cyz, cxz, y, &a, 1.1f, .05f, .02f, &r, nrow, xfmt, nullptr, H4,
                                   nullptr, nullptr, nullptr, nullptr, nullptr, app, hard, 0);
    rseed(&r, seed); write_spatial(ctu, tgt, mask, T, M, P, o, mean, isd, hasax, cyz, cxz, y, &a, 1.1f, .05f, .02f, &r, nrow, xfmt, nullptr, H1,
                                   nullptr, nullptr, nullptr, nullptr, nullptr, app, hard, 1);
    int same = !memcmp(T4, T, NCH * p3) && !memcmp(M4, M, p3);
    if (hard) {   /* the trainer's packed form straight from the warp: sample_pack_tm's codes of T / M, and the same CT channel */
        uint8_t *tm = malloc(p3), *tp = malloc(p3); uint16_t *h1b = malloc(p3 * 2); size_t bad = 0;
        if (!tm || !tp || !h1b) abort();
        rseed(&r, seed); write_spatial_tm(ctu, tgt, mask, nullptr, nullptr, P, o, mean, isd, hasax, cyz, cxz, y, &a, 1.1f, .05f, .02f, &r, nrow, xfmt, nullptr, h1b,
                                          nullptr, nullptr, nullptr, nullptr, nullptr, app, hard, 1, tm);
        same = same && !sample_pack_tm(T, T + p3, M, p3, tp, &bad) && !memcmp(tm, tp, p3) && !memcmp(h1b, H1, p3 * 2);
        free(tm); free(tp); free(h1b);
    }
    rad_record(rec, o, P, P, hasax, y, cyz, cxz, &a);
    free(nrow); free(app); free(T4); free(M4);
    return same ? a.active : -1;
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
/* window-sized worker scratch (>= 4 MB): 2 MB aligned with transparent huge pages requested (the system's THP mode is often
   "madvise"): the warp writer's gathers and the volume passes otherwise walk 4 KB page tables over hundreds of MB per window. Pages
   are still only backed when touched. free() releases it as any malloc block. */
static void *wbig_alloc(size_t n, int zero) {
    const size_t H = (size_t)2 << 20;
    if (n < ((size_t)4 << 20)) return zero ? calloc(n ? n : 1, 1) : malloc(n ? n : 1);
    void *p = nullptr;
    if (posix_memalign(&p, H, n)) return nullptr;
#ifdef MADV_HUGEPAGE
    madvise(p, (n + H - 1) & ~(H - 1), MADV_HUGEPAGE);
#endif
    if (zero) memset(p, 0, n);
    return p;
}
static void *tp_get(size_t n, int zero) {   /* the smallest pooled buffer of n's class up to 1.25 times it (window sizes vary by a few MB) */
    const size_t c = tp_class(n), cmax = c >= ((size_t)1 << 20) ? c + c / 4 : c;
    int best = -1;
    for (int i = 0; i < tpool.k; i++) if (tpool.n[i] >= c && tpool.n[i] <= cmax && (best < 0 || tpool.n[i] < tpool.n[best])) best = i;
    if (best >= 0) {   /* (put back under n's class later: a pooled size never exceeds the block's) */
        void *p = tpool.b[best]; tpool.k--; tpool.b[best] = tpool.b[tpool.k]; tpool.n[best] = tpool.n[tpool.k];
        if (zero) memset(p, 0, n);
        return p;
    }
    return wbig_alloc(c, zero);
}
static void tp_put(void *p, size_t n) {
    if (!p) return;
    if (tpool.k < TP_N) { tpool.b[tpool.k] = p; tpool.n[tpool.k] = tp_class(n); tpool.k++; } else free(p);
}
static void tp_drain(void) { while (tpool.k) free(tpool.b[--tpool.k]); }
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
   wrap at a recto is continuous on the circle.
   ph: the grid region read (ln[0] x ln[1] x ln[2] codes, PHASE_UNKNOWN outside the array), its first voxel lo; o: the window corner. */
static void phase_interp_axes(const int64_t o[3], const int64_t lo[3], int D, int P, int shift, int *j0, float *tf) {   /* per axis: the lower grid neighbour, the fraction */
    const int g = 1 << shift; const double c0 = 0.5 * (g - 1); const int wn[3] = {D, P, P};
    for (int d = 0; d < 3; d++) for (int k = 0; k < wn[d]; k++) {
        const double u = (o[d] + k - c0) / g; const int64_t j = (int64_t)floor(u);
        j0[d * P + k] = (int)(j - lo[d]); tf[d * P + k] = (float)(u - j);
    }
}
static float ph_cs[PHASE_PERIOD], ph_sn[PHASE_PERIOD]; static _Atomic int ph_init;
static void ph_tables(void) {
    if (atomic_load(&ph_init)) return;
    for (int k = 0; k < PHASE_PERIOD; k++) { ph_cs[k] = (float)cos(2 * M_PI * k / PHASE_PERIOD); ph_sn[k] = (float)sin(2 * M_PI * k / PHASE_PERIOD); }
    atomic_store(&ph_init, 1);
}
/* the reference: 8 neighbours per window voxel, atan2f (sample_phase_interp_ref; the test's yardstick for the separable path) */
void sample_phase_interp_ref(const uint8_t *ph, const int64_t ln[3], const int64_t lo[3], const int64_t o[3], int D, int P, int shift, uint8_t *side) {
    ph_tables();
    int *j0 = malloc(3 * (size_t)P * sizeof *j0); float *tf = malloc(3 * (size_t)P * sizeof *tf); if (!j0 || !tf) abort();
    phase_interp_axes(o, lo, D, P, shift, j0, tf);
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
                sx += wv * ph_cs[v]; sy += wv * ph_sn[v]; wk += wv;
            }
            if (wk < 0.5f || sx * sx + sy * sy < 1e-8f) { dst[x] = PHASE_UNKNOWN; continue; }
            float t = atan2f(sy, sx) * (float)(PHASE_PERIOD / (2 * M_PI)); if (t < 0) t += PHASE_PERIOD;
            const int q = (int)lrintf(t); dst[x] = (uint8_t)(q >= PHASE_PERIOD ? q - PHASE_PERIOD : q);
        }
    }
    free(j0); free(tf);
}
/* atan2(y, x) in turns, [0, 1): octant reduction + a degree-11 odd minimax atan on [0, 1] (|error| < 2e-6 rad; a phase code is 0.025 rad),
   branch-free so the voxel loop vectorises */
static inline float atan2_turns(float y, float x) {
    const float ax = fabsf(x), ay = fabsf(y), m0 = ax > ay ? ax : ay, mx = m0 > 1e-30f ? m0 : 1e-30f, mn = ax < ay ? ax : ay;   /* (not fmaxf: it does not vectorise) */
    const float a = mn / mx, s = a * a;
    float r = a * (0.99997726f + s * (-0.33262347f + s * (0.19354346f + s * (-0.11643287f + s * (0.05265332f + s * -0.01172120f)))));
    const float sw = (float)(ay > ax), nx = (float)(x < 0.f), ny = (float)(y < 0.f);   /* (arithmetic selects: the loop vectorises) */
    r += sw * (1.57079633f - 2.f * r);
    r += nx * (3.14159265f - 2.f * r);
    r += ny * (6.28318531f - 2.f * r);
    return r * 0.159154943f;
}
/* one window plane: the z blend of two cached grid planes (C, S, W each n floats) and the code */
static void phase_blend_row(const float *restrict A, const float *restrict Bp, size_t n, float t, uint8_t *restrict dst) {
    const float u = 1.f - t;
    for (size_t k = 0; k < n; k++) {
        const float c = u * A[k] + t * Bp[k], sn = u * A[n + k] + t * Bp[n + k], w = u * A[2 * n + k] + t * Bp[2 * n + k];
        const float q = atan2_turns(sn, c) * (float)PHASE_PERIOD;
        int qi = (int)(q + 0.5f); qi -= (qi >= PHASE_PERIOD) * PHASE_PERIOD;
        const int bad = (w < 0.5f) | (c * c + sn * sn < 1e-8f);
        dst[k] = (uint8_t)(qi | bad * 255);   /* (qi < 252: OR 255 = PHASE_UNKNOWN) */
    }
}
/* the same interpolation, separable (trilinear = x, then y, then z): the x-interpolated phasor and weight sums of the grid z planes in use
   (two cached), then per run of window z planes sharing their grid neighbours and per window row the y-interpolated rows of both planes
   (in L1) blended in z for each plane of the run, and a polynomial atan2. Codes equal the reference's but for float rounding at a code's
   half-way point (+-1 code, rare) and the 0.5-weight / zero-phasor cut. */
void sample_phase_interp(const uint8_t *ph, const int64_t ln[3], const int64_t lo[3], const int64_t o[3], int D, int P, int shift, uint8_t *side) {
    ph_tables();
    const size_t PP = (size_t)P * P, RX = 3 * (size_t)ln[1] * P;
    int *j0 = malloc(3 * (size_t)P * sizeof *j0); float *tf = malloc(3 * (size_t)P * sizeof *tf);
    float *rx = malloc(2 * RX * sizeof *rx), *rows = malloc(6 * (size_t)P * sizeof *rows);
    if (!j0 || !tf || !rx || !rows) abort();
    phase_interp_axes(o, lo, D, P, shift, j0, tf);
    const int *jy = j0 + P, *jx = j0 + 2 * P; const float *ty = tf + P, *tx = tf + 2 * P;
    int have[2] = {-1, -1};   /* the grid z plane whose x stage each rx slot holds */
    const int y0 = jy[0], y1 = jy[P - 1] + 1;   /* the grid rows any window row reads */
    for (int z = 0; z < D;) {
        const int az = j0[z]; int ze = z + 1; while (ze < D && j0[ze] == az) ze++;   /* window planes z..ze-1 blend grid planes az, az + 1 */
        int slot[2];
        for (int e = 0; e < 2; e++) {
            const int gz = az + e;
            if (have[0] == gz) { slot[e] = 0; continue; }
            if (have[1] == gz) { slot[e] = 1; continue; }
            const int sl = e == 0 ? (have[0] == az + 1 ? 1 : 0) : (slot[0] == 0 ? 1 : 0);
            for (int gy = y0; gy <= y1; gy++) {   /* x stage: rows of grid plane gz */
                const uint8_t *row = ph + ((size_t)gz * ln[1] + gy) * ln[2];
                float *rc = rx + sl * RX + (size_t)3 * gy * P, *rs = rc + P, *rw = rs + P;
                for (int x = 0; x < P; x++) {
                    const int v0 = row[jx[x]], v1 = row[jx[x] + 1]; const float t = tx[x];
                    const float w0 = v0 < PHASE_PERIOD ? 1.f - t : 0.f, w1 = v1 < PHASE_PERIOD ? t : 0.f;
                    const int k0 = v0 < PHASE_PERIOD ? v0 : 0, k1 = v1 < PHASE_PERIOD ? v1 : 0;
                    rc[x] = w0 * ph_cs[k0] + w1 * ph_cs[k1]; rs[x] = w0 * ph_sn[k0] + w1 * ph_sn[k1]; rw[x] = w0 + w1;
                }
            }
            have[sl] = gz; slot[e] = sl;
        }
        for (int y = 0; y < P; y++) {   /* y stage of both planes for this row, then the z blend of each window plane of the run */
            const float t = ty[y], u = 1.f - t;
            for (int e = 0; e < 2; e++) {
                const float *ra = rx + slot[e] * RX + (size_t)3 * jy[y] * P, *rb = ra + (size_t)3 * P;
                float *r = rows + (size_t)e * 3 * P;
                for (int x = 0; x < 3 * P; x++) r[x] = u * ra[x] + t * rb[x];
            }
            for (int zz = z; zz < ze; zz++) phase_blend_row(rows, rows + (size_t)3 * P, (size_t)P, tf[zz], side + (size_t)zz * PP + (size_t)y * P);
        }
        z = ze;
    }
    free(j0); free(tf); free(rx); free(rows);
}
static int phase_window_interp(z3 *a, const int64_t o[3], int D, int P, int shift, uint8_t *side) {   /* a D x P x P window, D <= P */
    const int g = 1 << shift; const double c0 = 0.5 * (g - 1); const int wn[3] = {D, P, P};
    int64_t lo[3], ln[3], ro[3], rn[3]; const z3_meta *m = z3_meta_of(a);
    for (int d = 0; d < 3; d++) {
        lo[d] = (int64_t)floor((o[d] - c0) / g); ln[d] = (int64_t)floor((o[d] + wn[d] - 1 - c0) / g) + 2 - lo[d];
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
    if (ufsm_env_on("UFSM_PHASE_INTERP_REF")) sample_phase_interp_ref(ph, ln, lo, o, D, P, shift, side);   /* (A/B: the 8-neighbour atan2f path) */
    else sample_phase_interp(ph, ln, lo, o, D, P, shift, side);
    tp_put(ph, N);
    return 0;
}
/* a stored phase / winding read against the convention (sources "phase_sign": -1): codes 0..251 negated mod 252, unknown kept */
static void phase_negate(uint8_t *p, size_t n) { for (size_t k = 0; k < n; k++) if (p[k] < PHASE_PERIOD && p[k]) p[k] = (uint8_t)(PHASE_PERIOD - p[k]); }
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
[[maybe_unused]] static void warp_phase_ref(const uint8_t *src, uint8_t *dst, int P, sym sy, const spatial_aug *a) {
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

/* --task unwrap2: the winding mod 4 (w4, 63 codes per turn) and its component through the warp as warp_phase pulls the phase: each
   output voxel takes the component of its nearest source voxel and the phasor mean (period 4 turns) of the 8 neighbours in that
   component (unknown under half the weight), so two gauges never mix */
[[maybe_unused]] static void warp_w4_ref(const uint8_t *w4, const uint8_t *cp, uint8_t *dw, uint8_t *dc, int P, sym sy, const spatial_aug *a) {
    static float cs[PHASE_PERIOD], sn[PHASE_PERIOD]; static _Atomic int init;
    if (!atomic_load(&init)) { for (int k = 0; k < PHASE_PERIOD; k++) { cs[k] = (float)cos(2 * M_PI * k / PHASE_PERIOD); sn[k] = (float)sin(2 * M_PI * k / PHASE_PERIOD); } atomic_store(&init, 1); }
    float *table = malloc(6 * (size_t)P * sizeof *table); if (!table) abort();
    spatial_aug_tables(a, table);
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
        const size_t k = ((size_t)z * P + y) * P + x;
        float u[3], jac[3], v[3]; spatial_aug_pull(a, table, z, y, x, u, jac);
        for (int d = 0; d < 3; d++) v[sy.perm[d]] = sy.flip[d] ? P - 1 - u[d] : u[d];
        int inside = 1; for (int d = 0; d < 3; d++) if (v[d] < 0 || v[d] > P - 1) inside = 0;
        dw[k] = 255; dc[k] = 255;
        if (!inside) continue;
        int base[3]; float f[3];
        for (int d = 0; d < 3; d++) { base[d] = (int)v[d]; if (base[d] >= P - 1) base[d] = P - 2; f[d] = v[d] - base[d]; }
        const int c0 = cp[((size_t)(base[0] + (f[0] >= .5f)) * P + base[1] + (f[1] >= .5f)) * P + base[2] + (f[2] >= .5f)];
        if (c0 == 255) continue;
        float sx = 0.f, s_y = 0.f, wk = 0.f;
        for (int c = 0; c < 8; c++) {
            const int bz = c >> 2 & 1, by = c >> 1 & 1, bx = c & 1;
            const float w = (bz ? f[0] : 1 - f[0]) * (by ? f[1] : 1 - f[1]) * (bx ? f[2] : 1 - f[2]);
            const size_t j = ((size_t)(base[0] + bz) * P + base[1] + by) * P + base[2] + bx;
            if (w <= 1e-7f || cp[j] != c0 || w4[j] >= PHASE_PERIOD) continue;
            sx += w * cs[w4[j]]; s_y += w * sn[w4[j]]; wk += w;
        }
        if (wk < .5f || sx * sx + s_y * s_y < 1e-8f) continue;
        float t = atan2f(s_y, sx) * (float)(PHASE_PERIOD / (2 * M_PI)); if (t < 0) t += PHASE_PERIOD;
        const int q = (int)lrintf(t); dw[k] = (uint8_t)(q >= PHASE_PERIOD ? q - PHASE_PERIOD : q); dc[k] = (uint8_t)c0;
    }
    free(table);
}

/* warp_phase_ref and warp_w4_ref in one pass, vectorised (draw() and the tests call this): the pull coordinates, corners and trilinear
   weights computed once for the phase and the winding, cache-blocked tiles, AVX2 lanes of 8 voxels along x. Bit for bit the
   reference: the same float operations in the same order per lane (no contraction; a skipped corner adds +0, which changes no sum
   that survives the unknown tests), the corner order c = 0..7, and the code atan2f gives: a vector arctangent (Cephes atanf
   reduction, ~2e-7 rad) decides each voxel's rounded code unless the scaled angle lies within 1e-3 code of a rounding boundary or
   of the wrap, where the voxel falls back to atan2f (~0.2 % of random phasors).
   ph: the phase window, PHASE codes, P^3 + 3 readable bytes (byte pairs along x are gathered as 32-bit words); wq (nullable): the
   winding mod 4 and its component per voxel as w4 | comp << 8 (P^3 words). dph / dw / dc (each nullable): the warped phase, winding
   and component. P^3 cubes (slabs never warp). */
static inline uint8_t warp_code_of(float sx, float sy) {
    float t = atan2f(sy, sx) * (float)(PHASE_PERIOD / (2 * M_PI)); if (t < 0) t += PHASE_PERIOD;
    const int q = (int)lrintf(t); return (uint8_t)(q >= PHASE_PERIOD ? q - PHASE_PERIOD : q);
}
static void warp_pw_voxel(const uint8_t *ph, const uint16_t *wq, uint8_t *dph, uint8_t *dw, uint8_t *dc, int P, sym sy, const spatial_aug *a,
                          const float *table, const float *cs, const float *sn, int z, int y, int x) {
    const size_t k = ((size_t)z * P + y) * P + x;
    float u[3], jac[3], v[3]; spatial_aug_pull(a, table, z, y, x, u, jac);
    for (int d = 0; d < 3; d++) v[sy.perm[d]] = sy.flip[d] ? P - 1 - u[d] : u[d];
    int inside = 1; for (int d = 0; d < 3; d++) if (v[d] < 0 || v[d] > P - 1) inside = 0;
    if (dph) dph[k] = PHASE_UNKNOWN;
    if (dw) { dw[k] = 255; dc[k] = 255; }
    if (!inside) return;
    int base[3]; float f[3];
    for (int d = 0; d < 3; d++) { base[d] = (int)v[d]; if (base[d] >= P - 1) base[d] = P - 2; f[d] = v[d] - base[d]; }
    if (dph) {
        float sx = 0.f, s_y = 0.f, wk = 0.f;
        for (int c = 0; c < 8; c++) {
            const int bz = c >> 2 & 1, by = c >> 1 & 1, bx = c & 1;
            const float w = (bz ? f[0] : 1 - f[0]) * (by ? f[1] : 1 - f[1]) * (bx ? f[2] : 1 - f[2]);
            const int p = ph[((size_t)(base[0] + bz) * P + base[1] + by) * P + base[2] + bx];
            if (w <= 1e-7f || p >= PHASE_PERIOD) continue;
            sx += w * cs[p]; s_y += w * sn[p]; wk += w;
        }
        if (!(wk < .5f || sx * sx + s_y * s_y < 1e-8f)) dph[k] = warp_code_of(sx, s_y);
    }
    if (dw) {
        const int c0 = wq[((size_t)(base[0] + (f[0] >= .5f)) * P + base[1] + (f[1] >= .5f)) * P + base[2] + (f[2] >= .5f)] >> 8;
        if (c0 == 255) return;
        float sx = 0.f, s_y = 0.f, wk = 0.f;
        for (int c = 0; c < 8; c++) {
            const int bz = c >> 2 & 1, by = c >> 1 & 1, bx = c & 1;
            const float w = (bz ? f[0] : 1 - f[0]) * (by ? f[1] : 1 - f[1]) * (bx ? f[2] : 1 - f[2]);
            const int q = wq[((size_t)(base[0] + bz) * P + base[1] + by) * P + base[2] + bx], qw = q & 255;
            if (w <= 1e-7f || q >> 8 != c0 || qw >= PHASE_PERIOD) continue;
            sx += w * cs[qw]; s_y += w * sn[qw]; wk += w;
        }
        if (wk < .5f || sx * sx + s_y * s_y < 1e-8f) return;
        dw[k] = warp_code_of(sx, s_y); dc[k] = (uint8_t)c0;
    }
}
static float g_wcs[PHASE_PERIOD], g_wsn[PHASE_PERIOD];
static pthread_once_t g_wtab_once = PTHREAD_ONCE_INIT;
static float g_wcssn[512] __attribute__((aligned(64)));   /* (cos, sin) pairs of the 256 codes, 0 above PHASE_PERIOD (vector gathers) */
static void wtab_init(void) {
    for (int k = 0; k < PHASE_PERIOD; k++) { g_wcs[k] = (float)cos(2 * M_PI * k / PHASE_PERIOD); g_wsn[k] = (float)sin(2 * M_PI * k / PHASE_PERIOD); g_wcssn[2 * k] = g_wcs[k]; g_wcssn[2 * k + 1] = g_wsn[k]; }
}
#ifdef __AVX2__
/* cos and sin of 8 codes (0..255) from the pair table: two 4 x 64-bit gathers (half the loads of two 8 x 32-bit ones); lanes
   0, 1, 4, 5 go to the first, 2, 3, 6, 7 to the second, so the even / odd shuffles come out in lane order */
static inline void warp_cssn8(__m256i code, __m256 *c, __m256 *s) {
    const __m256i q = _mm256_permutevar8x32_epi32(code, _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7));
    const __m256 g0 = _mm256_castpd_ps(_mm256_i32gather_pd((const double *)g_wcssn, _mm256_castsi256_si128(q), 8));
    const __m256 g1 = _mm256_castpd_ps(_mm256_i32gather_pd((const double *)g_wcssn, _mm256_extracti128_si256(q, 1), 8));
    *c = _mm256_shuffle_ps(g0, g1, 0x88); *s = _mm256_shuffle_ps(g0, g1, 0xdd);
}
/* the rounded code of 8 phasors (sx, sy) as 32-bit lanes; *need: lanes within 1e-3 code of a rounding boundary (caller: atan2f) */
static inline __m256i warp_code8(__m256 sx, __m256 sy, __m256 *need) {
    const __m256 sgn = _mm256_set1_ps(-0.f), ax = _mm256_andnot_ps(sgn, sx), ay = _mm256_andnot_ps(sgn, sy);
    const __m256 mn = _mm256_min_ps(ax, ay), mx = _mm256_max_ps(ax, ay);
    __m256 t = _mm256_div_ps(mn, _mm256_max_ps(mx, _mm256_set1_ps(1e-30f)));   /* [0, 1] */
    const __m256 big = _mm256_cmp_ps(t, _mm256_set1_ps(.4142135623730950f), _CMP_GT_OQ);   /* tan(pi / 8): t -> (t - 1) / (t + 1), + pi / 4 */
    t = _mm256_blendv_ps(t, _mm256_div_ps(_mm256_sub_ps(t, _mm256_set1_ps(1.f)), _mm256_add_ps(t, _mm256_set1_ps(1.f))), big);
    const __m256 z = _mm256_mul_ps(t, t);
    __m256 p = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(8.05374449538e-2f), z), _mm256_set1_ps(-1.38776856032e-1f));
    p = _mm256_add_ps(_mm256_mul_ps(p, z), _mm256_set1_ps(1.99777106478e-1f));
    p = _mm256_add_ps(_mm256_mul_ps(p, z), _mm256_set1_ps(-3.33329491539e-1f));
    __m256 r = _mm256_add_ps(_mm256_mul_ps(_mm256_mul_ps(p, z), t), t);
    r = _mm256_add_ps(r, _mm256_and_ps(big, _mm256_set1_ps((float)(M_PI / 4))));
    r = _mm256_blendv_ps(r, _mm256_sub_ps(_mm256_set1_ps((float)(M_PI / 2)), r), _mm256_cmp_ps(ay, ax, _CMP_GT_OQ));
    r = _mm256_blendv_ps(r, _mm256_sub_ps(_mm256_set1_ps((float)M_PI), r), sx);   /* (sign bit of sx) */
    r = _mm256_xor_ps(r, _mm256_and_ps(sy, sgn));
    __m256 c = _mm256_mul_ps(r, _mm256_set1_ps((float)(PHASE_PERIOD / (2 * M_PI))));
    c = _mm256_add_ps(c, _mm256_and_ps(_mm256_cmp_ps(c, _mm256_setzero_ps(), _CMP_LT_OQ), _mm256_set1_ps((float)PHASE_PERIOD)));
    const __m256 rd = _mm256_round_ps(c, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 d = _mm256_andnot_ps(sgn, _mm256_sub_ps(c, rd));
    /* near .5 (rounding) or near the wrap at 0 / PHASE_PERIOD (where the reference's t < 0 test and our sign may disagree) */
    *need = _mm256_or_ps(_mm256_cmp_ps(d, _mm256_set1_ps(.5f - 1e-3f), _CMP_GT_OQ),
                         _mm256_cmp_ps(_mm256_andnot_ps(sgn, c), _mm256_set1_ps(1e-3f), _CMP_LT_OQ));
    __m256i q = _mm256_cvtps_epi32(rd);
    q = _mm256_sub_epi32(q, _mm256_and_si256(_mm256_cmpgt_epi32(q, _mm256_set1_epi32(PHASE_PERIOD - 1)), _mm256_set1_epi32(PHASE_PERIOD)));
    return q;
}
#endif
static void warp_phase_w4(const uint8_t *ph, const uint16_t *wq, uint8_t *dph, uint8_t *dw, uint8_t *dc, int P, sym sy, const spatial_aug *a) {
    pthread_once(&g_wtab_once, wtab_init);
    const float *cs = g_wcs, *sn = g_wsn;
    float *table = malloc(6 * (size_t)P * sizeof *table); if (!table) abort();
    spatial_aug_tables(a, table);
#ifdef __AVX2__
    const float c = .5f * (a->P - 1), sc = (float)a->scale;
    float Rf[3][3], Af[3][3], *vt = malloc(3 * (size_t)P * sizeof *vt); if (!vt) abort();
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) { Rf[i][j] = (float)a->R[i][j]; Af[i][j] = (float)a->A[i][j]; }
    for (int d = 0; d < 3; d++) for (int u = 0; u < P; u++) vt[d * P + u] = u - c - table[d * P + u];   /* spatial_aug_pull's per-axis terms */
    enum { ZB = 8, YB = 16, XB = 64 };
    const __m256 one = _mm256_set1_ps(1.f), fP1 = _mm256_set1_ps((float)(P - 1)), zero = _mm256_setzero_ps(), thr = _mm256_set1_ps(1e-7f), half = _mm256_set1_ps(.5f);
    const __m256i iP = _mm256_set1_epi32(P), b255 = _mm256_set1_epi32(255), per = _mm256_set1_epi32(PHASE_PERIOD);
    const int xv = P & ~7;   /* vector lanes; the rest per voxel */
    for (int z0 = 0; z0 < P; z0 += ZB) for (int y0 = 0; y0 < P; y0 += YB) for (int x0 = 0; x0 < P; x0 += XB)
    for (int z = z0; z < z0 + ZB && z < P; z++) for (int y = y0; y < y0 + YB && y < P; y++) {
        const float v0 = vt[z], v1 = vt[P + y]; float t01[3];
        for (int d = 0; d < 3; d++) t01[d] = Rf[0][d] * v0 + Rf[1][d] * v1;
        const int xe = x0 + XB < xv ? x0 + XB : xv;
        for (int xs = x0; xs < xe; xs += 8) {
            const __m256 v2 = _mm256_loadu_ps(vt + 2 * P + xs);
            __m256 U[3], Vv[3], F[3], G[3];
            if (a->affine) {
                __m256 w[3];
                for (int d = 0; d < 3; d++) w[d] = _mm256_add_ps(_mm256_set1_ps(t01[d]), _mm256_mul_ps(_mm256_set1_ps(Rf[2][d]), v2));
                for (int d = 0; d < 3; d++) U[d] = _mm256_add_ps(_mm256_set1_ps(c), _mm256_mul_ps(_mm256_set1_ps(sc), _mm256_add_ps(_mm256_add_ps(
                    _mm256_mul_ps(_mm256_set1_ps(Af[d][0]), w[0]), _mm256_mul_ps(_mm256_set1_ps(Af[d][1]), w[1])), _mm256_mul_ps(_mm256_set1_ps(Af[d][2]), w[2]))));
            } else for (int d = 0; d < 3; d++) U[d] = _mm256_add_ps(_mm256_set1_ps(c), _mm256_mul_ps(_mm256_set1_ps(sc), _mm256_add_ps(_mm256_set1_ps(t01[d]), _mm256_mul_ps(_mm256_set1_ps(Rf[2][d]), v2))));
            for (int d = 0; d < 3; d++) Vv[sy.perm[d]] = sy.flip[d] ? _mm256_sub_ps(fP1, U[d]) : U[d];
            __m256 in = _mm256_and_ps(_mm256_cmp_ps(Vv[0], zero, _CMP_GE_OQ), _mm256_cmp_ps(Vv[0], fP1, _CMP_LE_OQ));
            for (int d = 1; d < 3; d++) in = _mm256_and_ps(in, _mm256_and_ps(_mm256_cmp_ps(Vv[d], zero, _CMP_GE_OQ), _mm256_cmp_ps(Vv[d], fP1, _CMP_LE_OQ)));
            const size_t k = ((size_t)z * P + y) * P + xs;
            const __m256i ini = _mm256_castps_si256(in);
            if (_mm256_testz_si256(ini, ini)) {
                if (dph) memset(dph + k, PHASE_UNKNOWN, 8);
                if (dw) { memset(dw + k, 255, 8); memset(dc + k, 255, 8); }
                continue;
            }
            __m256i Bv[3];
            for (int d = 0; d < 3; d++) {
                Bv[d] = _mm256_and_si256(_mm256_min_epi32(_mm256_cvttps_epi32(Vv[d]), _mm256_set1_epi32(P - 2)), ini);
                F[d] = _mm256_sub_ps(Vv[d], _mm256_cvtepi32_ps(Bv[d])); G[d] = _mm256_sub_ps(one, F[d]);
            }
            const __m256i b0 = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_add_epi32(_mm256_mullo_epi32(Bv[0], iP), Bv[1]), iP), Bv[2]);
            /* per corner (the reference's order c = 0..7) its weight ((wz * wy) * wx), shared by the phase and the winding sums */
            __m256 sxp = zero, syp = zero, wkp = zero, sxw = zero, syw = zero, wkw = zero, cok = zero;
            __m256i c0 = b255, wdp[4] = {}, pdp[4] = {};
            for (int pr = 0; pr < 4; pr++) {
                const __m256i ix = _mm256_add_epi32(b0, _mm256_set1_epi32(((pr >> 1) * P + (pr & 1)) * P));   /* (in bounds: outside lanes index 0) */
                if (dph) pdp[pr] = _mm256_i32gather_epi32((const int *)ph, ix, 1);
                if (dw) wdp[pr] = _mm256_i32gather_epi32((const int *)wq, ix, 2);
            }
            if (dw) {   /* the nearest corner (fraction >= .5: the upper one) and its component */
                const __m256 nz_ = _mm256_cmp_ps(F[0], half, _CMP_GE_OQ), ny_ = _mm256_cmp_ps(F[1], half, _CMP_GE_OQ), nx_ = _mm256_cmp_ps(F[2], half, _CMP_GE_OQ);
                const __m256i w0 = _mm256_blendv_epi8(wdp[0], wdp[1], _mm256_castps_si256(ny_)), w1 = _mm256_blendv_epi8(wdp[2], wdp[3], _mm256_castps_si256(ny_));
                const __m256i ws = _mm256_blendv_epi8(w0, w1, _mm256_castps_si256(nz_));
                c0 = _mm256_and_si256(_mm256_srli_epi32(_mm256_blendv_epi8(ws, _mm256_srli_epi32(ws, 16), _mm256_castps_si256(nx_)), 8), b255);
                cok = _mm256_andnot_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(c0, b255)), in);
            }
            for (int pr = 0; pr < 4; pr++) {
                const __m256 wzy = _mm256_mul_ps(pr >> 1 ? F[0] : G[0], pr & 1 ? F[1] : G[1]);
                for (int bx = 0; bx < 2; bx++) {
                    const __m256 w = _mm256_mul_ps(wzy, bx ? F[2] : G[2]), wg = _mm256_cmp_ps(w, thr, _CMP_GT_OQ);
                    if (dph) {
                        const __m256i code = _mm256_and_si256(bx ? _mm256_srli_epi32(pdp[pr], 8) : pdp[pr], b255);
                        const __m256 use = _mm256_and_ps(_mm256_and_ps(in, wg), _mm256_castsi256_ps(_mm256_cmpgt_epi32(per, code)));
                        __m256 cv, sv; warp_cssn8(code, &cv, &sv);
                        sxp = _mm256_add_ps(sxp, _mm256_and_ps(_mm256_mul_ps(w, cv), use));
                        syp = _mm256_add_ps(syp, _mm256_and_ps(_mm256_mul_ps(w, sv), use));
                        wkp = _mm256_add_ps(wkp, _mm256_and_ps(w, use));
                    }
                    if (dw) {
                        const __m256i h = bx ? _mm256_srli_epi32(wdp[pr], 16) : wdp[pr];
                        const __m256i code = _mm256_and_si256(h, b255), comp = _mm256_and_si256(_mm256_srli_epi32(h, 8), b255);
                        const __m256 use = _mm256_and_ps(_mm256_and_ps(cok, wg), _mm256_castsi256_ps(_mm256_and_si256(_mm256_cmpgt_epi32(per, code), _mm256_cmpeq_epi32(comp, c0))));
                        __m256 cv, sv; warp_cssn8(code, &cv, &sv);
                        sxw = _mm256_add_ps(sxw, _mm256_and_ps(_mm256_mul_ps(w, cv), use));
                        syw = _mm256_add_ps(syw, _mm256_and_ps(_mm256_mul_ps(w, sv), use));
                        wkw = _mm256_add_ps(wkw, _mm256_and_ps(w, use));
                    }
                }
            }
            for (int f = 0; f < 2; f++) {
                if (!(f ? dw != nullptr : dph != nullptr)) continue;
                const __m256 sx = f ? sxw : sxp, s_y = f ? syw : syp, wk = f ? wkw : wkp;
                const __m256 ok = _mm256_andnot_ps(_mm256_or_ps(_mm256_cmp_ps(wk, half, _CMP_LT_OQ),
                    _mm256_cmp_ps(_mm256_add_ps(_mm256_mul_ps(sx, sx), _mm256_mul_ps(s_y, s_y)), _mm256_set1_ps(1e-8f), _CMP_LT_OQ)), f ? cok : in);
                __m256 need; const __m256i q = warp_code8(sx, s_y, &need), oki = _mm256_castps_si256(ok);
                uint64_t qb = pack8_u32(_mm256_blendv_epi8(b255, q, oki));
                const int nm = _mm256_movemask_ps(_mm256_and_ps(need, ok));
                if (nm) {
                    float fx[8], fy[8]; _mm256_storeu_ps(fx, sx); _mm256_storeu_ps(fy, s_y);
                    for (int l = 0; l < 8; l++) if (nm >> l & 1) { qb &= ~((uint64_t)255 << 8 * l); qb |= (uint64_t)warp_code_of(fx[l], fy[l]) << 8 * l; }
                }
                if (f) { const uint64_t cb = pack8_u32(_mm256_blendv_epi8(b255, c0, oki)); memcpy(dw + k, &qb, 8); memcpy(dc + k, &cb, 8); }
                else memcpy(dph + k, &qb, 8);
            }
        }
        if (x0 + XB >= P) for (int x = xv; x < P; x++) warp_pw_voxel(ph, wq, dph, dw, dc, P, sy, a, table, cs, sn, z, y, x);
    }
    free(vt);
#else
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) warp_pw_voxel(ph, wq, dph, dw, dc, P, sy, a, table, cs, sn, z, y, x);
#endif
    free(table);
}
/* the phase alone / the winding alone through warp_phase_w4 (tests; draw() calls it directly on padded / interleaved scratch) */
[[maybe_unused]] static void warp_phase(const uint8_t *src, uint8_t *dst, int P, sym sy, const spatial_aug *a) {
    const size_t p3 = (size_t)P * P * P; uint8_t *pp = malloc(p3 + 64); if (!pp) abort();
    memcpy(pp, src, p3); memset(pp + p3, PHASE_UNKNOWN, 64);
    warp_phase_w4(pp, nullptr, dst, nullptr, nullptr, P, sy, a);
    free(pp);
}
static void warp_w4_pack(const uint8_t *w4, const uint8_t *cp, uint16_t *wq, size_t n) { for (size_t k = 0; k < n; k++) wq[k] = (uint16_t)(w4[k] | cp[k] << 8); }
[[maybe_unused]] static void warp_w4(const uint8_t *w4, const uint8_t *cp, uint8_t *dw, uint8_t *dc, int P, sym sy, const spatial_aug *a) {
    const size_t p3 = (size_t)P * P * P; uint16_t *wq = malloc(p3 * 2); if (!wq) abort();
    warp_w4_pack(w4, cp, wq, p3);
    warp_phase_w4(nullptr, wq, nullptr, dw, dc, P, sy, a);
    free(wq);
}

/* task restore: one clean CT window (the source's min_level) inside a "boxes" entry (training) or the holdout box (validation), the tone
   LUT 255 gain (v / 255)^gamma (gain, gamma exp(U(-0.18, 0.18)), 0 kept) and a cube symmetry, into batch.t */
static int draw_restore(sampler *sp, batch *b, int i, rng *r, uint8_t *big) {
    const sample_cfg *c = &sp->cfg;
    const int P = c->P, Pz = PZ(c);
    const size_t p3 = P3(sp);
    double u = runif(r) * sp->cum[sp->S->n - 1];
    int si = 0; while (si < sp->S->n - 1 && sp->cum[si] < u) si++;
    source *s = &sp->S->src[si];
    if (c->holdout ? !s->hold_n[0] : s->hold_n[0] && !s->nbox) return 1;   /* (a holdout source without boxes is validation only) */
    const int l = s->min_level > 0 ? s->min_level : 0;
    if (!level_ok(sp, s, si, l, -1)) return 1;
    z3 *ct = source_ct(s, l);
    const z3_meta *m = z3_meta_of(ct);
    int64_t o[3], n[3] = {Pz, P, P}, lo[3], hi[3];
    if (c->holdout) { if (!patch_bounds(s, m, n, l, nullptr, 0, 1, lo, hi)) return 1; }
    else if (s->nbox) {   /* a box by volume, the window inside it (level-l voxels) */
        double tot = 0; for (int k = 0; k < s->nbox; k++) tot += (double)s->box[k][3] * s->box[k][4] * s->box[k][5];
        double v = runif(r) * tot; int k = 0;
        for (; k < s->nbox - 1; k++) { v -= (double)s->box[k][3] * s->box[k][4] * s->box[k][5]; if (v < 0) break; }
        for (int d = 0; d < 3; d++) {
            lo[d] = (s->box[k][d] + ((int64_t)1 << l) - 1) >> l; hi[d] = ((s->box[k][d] + s->box[k][3 + d]) >> l) - n[d];
            if (hi[d] > m->shape[d] - n[d]) hi[d] = m->shape[d] - n[d];
            if (lo[d] < 0) lo[d] = 0;
            if (hi[d] < lo[d]) return 1;
        }
    } else for (int d = 0; d < 3; d++) { lo[d] = 0; hi[d] = m->shape[d] - n[d]; if (hi[d] < 0) return 1; }
    for (int d = 0; d < 3; d++) o[d] = lo[d] + rint_below(r, hi[d] - lo[d] + 1);
    if (!c->holdout && training_hits_holdout(sp, s, o, l)) return 1;
    z3 *cct = l + 3 < MAXLEV ? source_ct(s, l + 3) : nullptr;   /* coarse occupancy probe */
    if (cct && Pz >= 8) {
        const int pc = P >> 3; int64_t oc[3] = {o[0] >> 3, o[1] >> 3, o[2] >> 3}, nc[3] = {Pz >> 3, pc, pc};
        if (z3_read(cct, oc, nc, big, 1)) return -1;
        size_t nz = 0, tot = (size_t)nc[0] * pc * pc; for (size_t k = 0; k < tot; k++) nz += big[k] != 0;
        if ((double)nz / (double)tot < c->min_fg) { atomic_fetch_add(&sp->rejected, 1); return 1; }
    }
    if (z3_read(ct, o, n, big, 1)) return -1;
    size_t nz = 0; for (size_t k = 0; k < p3; k++) nz += big[k] != 0;
    if ((double)nz / (double)p3 < c->min_fg) { atomic_fetch_add(&sp->rejected, 1); return 1; }
    sym y = sym_of(0);
    if (c->augment) {
        y = sym_of((int)rint_below(r, 48));
        const double gain = exp((runif(r) * 2 - 1) * 0.18), gam = exp((runif(r) * 2 - 1) * 0.18);
        uint8_t lut[256]; lut[0] = 0;
        for (int v = 1; v < 256; v++) { const double t = 255 * gain * pow(v / 255.0, gam); lut[v] = (uint8_t)(t < 1 ? 1 : t > 255 ? 255 : lrint(t)); }
        for (size_t k = 0; k < p3; k++) big[k] = lut[big[k]];
        if (Pz != P && y.perm[0] != 0) y = sym_of(0);
    }
    sym_u8(big, b->t + (size_t)i * p3, Pz, P, y);
    memset(b->w + (size_t)i * NCH, 0, NCH); b->w[(size_t)i * NCH] = 1;
    b->src[i] = (int16_t)si; b->level[i] = (int8_t)l; b->um[i] = (float)(s->um * (double)(1 << l));
    for (int d = 0; d < 3; d++) b->corner[i][d] = o[d] << l;
    atomic_fetch_add(&sp->produced, 1);
    return 0;
}
/* task restore on real pairs (restore 2, src/srpair.h): a box of a pair source (training: its "train" boxes, validation: its "test" boxes or
   every box of a "val" pair), a P^3 target crop at a fine offset 4 oc and the native (P / 4)^3 crop at oc; a cube symmetry and (training) a
   tone LUT on both; batch.t row i = [target P^3 | input (P / 4)^3, rest of the row 0] */
static int draw_restore_pair(sampler *sp, batch *b, int i, rng *r, uint8_t *big) {
    const sample_cfg *c = &sp->cfg;
    const int P = c->P, q = P / 4;
    const size_t p3 = P3(sp), q3 = (size_t)q * q * q;
    double u = runif(r) * sp->cum[sp->S->n - 1];
    int si = 0; while (si < sp->S->n - 1 && sp->cum[si] < u) si++;
    source *s = &sp->S->src[si]; const srpair *p = s->srp;
    if (!p) return 1;
    const int nb = c->holdout ? p->nte : p->ntr; if (nb <= 0 || q > p->B) return 1;
    const int k = (c->holdout ? p->te : p->tr)[rint_below(r, nb)];
    int oc[3]; for (int d = 0; d < 3; d++) oc[d] = (int)rint_below(r, p->B - q + 1);
    uint8_t *co = malloc(q3); if (!co) return -1;
    if (srpair_read(p, k, oc, P, big, co)) { free(co); return -1; }
    size_t nz = 0; for (size_t v = 0; v < p3; v++) nz += big[v] != 0;
    if ((double)nz / (double)p3 < c->min_fg) { free(co); atomic_fetch_add(&sp->rejected, 1); return 1; }
    sym y = sym_of(0);
    if (c->augment) {
        y = sym_of((int)rint_below(r, 48));
        const double gain = exp((runif(r) * 2 - 1) * 0.12), gam = exp((runif(r) * 2 - 1) * 0.12);
        uint8_t lut[256]; lut[0] = 0;
        for (int v = 1; v < 256; v++) { const double t = 255 * gain * pow(v / 255.0, gam); lut[v] = (uint8_t)(t < 1 ? 1 : t > 255 ? 255 : lrint(t)); }
        for (size_t v = 0; v < p3; v++) big[v] = lut[big[v]];
        for (size_t v = 0; v < q3; v++) co[v] = lut[co[v]];
    }
    uint8_t *row = b->t + (size_t)i * 2 * p3;
    sym_u8(big, row, P, P, y);
    memset(row + p3, 0, p3); sym_u8(co, row + p3, q, q, y);
    free(co);
    memset(b->w + (size_t)i * NCH, 0, NCH); b->w[(size_t)i * NCH] = 1;
    b->src[i] = (int16_t)si; b->level[i] = 0; b->um[i] = (float)(s->um / 4);
    b->corner[i][0] = (int64_t)k * p->B + oc[0]; b->corner[i][1] = oc[1]; b->corner[i][2] = oc[2];
    atomic_fetch_add(&sp->produced, 1);
    return 0;
}
/* window chains (sample_cfg.chain_p): per worker and sample slot, the chain's last window and its walk */
typedef struct { int left, si, l, symi, mode, dir; int64_t o[3]; int64_t id; int idx; } chain_st;
/* the next corner of a chain (level-l voxels of source s): mode 0 along the scroll axis (z +- step, y / x following the axis centre), mode 1
   around it (the window centre rotated by step / radius about the axis at the window's middle plane; no axis: x +- step). 0: inside the scan */
int sample_chain_next(const axis *ax, int l, const int64_t shape[3], int Pz, int P, int mode, int dir, float step_frac, const int64_t o[3], int64_t on[3]) {
    const double scale = (double)(1 << l), step = fmax(1.0, round((double)step_frac * P));
    const int64_t n[3] = {Pz, P, P};
    memcpy(on, o, 3 * sizeof *on);
    if (mode == 0) {
        on[0] = o[0] + (int64_t)(dir * step);
        if (ax->n > 0) {
            double cy0, cx0, cy1, cx1;
            axis_at(ax, (o[0] + 0.5 * Pz) * scale, &cy0, &cx0); axis_at(ax, (on[0] + 0.5 * Pz) * scale, &cy1, &cx1);
            on[1] = o[1] + (int64_t)llround((cy1 - cy0) / scale); on[2] = o[2] + (int64_t)llround((cx1 - cx0) / scale);
        }
    } else if (ax->n > 0) {
        double cy, cx; axis_at(ax, (o[0] + 0.5 * Pz) * scale, &cy, &cx); cy /= scale; cx /= scale;
        const double vy = o[1] + 0.5 * P - cy, vx = o[2] + 0.5 * P - cx, R = fmax(hypot(vy, vx), (double)P), a = dir * step / R;
        const double ny = cy + vy * cos(a) - vx * sin(a), nx = cx + vy * sin(a) + vx * cos(a);
        on[1] = (int64_t)llround(ny - 0.5 * P); on[2] = (int64_t)llround(nx - 0.5 * P);
    } else on[2] = o[2] + (int64_t)(dir * step);
    for (int d = 0; d < 3; d++) if (on[d] < 0 || on[d] > shape[d] - n[d]) return 1;
    return 0;
}
/* --aug-crush (src/crush.h): the window pulled through a crush map instead of the plain read. The tiles are decoded into big's
   (2P)^3 scratch (free until the targets are built), so a footprint of several P^3 costs no RAM; the pulled phase (and its ignore
   mask) wait in pooled buffers for the stored-phase step (draw() returns them after every draw). */
/* the window's pulled labels (--aug-crush, --aug-tear): ph / ig the phase and its ignore mask; --aug-tear: w4 / wc the winding mod 4 and
   its component unwrapped before the tear (ph then already masked by ig: pre) */
static _Thread_local struct { uint8_t *ph, *ig, *w4, *wc; int pre; size_t n; double k, j0; int fine; uint64_t tiles; sa_tear tear; sa_delam delam;
                              int cax; double ca[3], cA[3]; double r0, r1; } g_crush;   /* (cax: --aug-curv moved the axis: ca a point on it, cA its direction, window coordinates) */   /* (k .. tear: the last map, diagnostics) */
static void crush_release(void) {
    tp_put(g_crush.ph, g_crush.n + 64); tp_put(g_crush.ig, g_crush.n); tp_put(g_crush.w4, g_crush.n); tp_put(g_crush.wc, g_crush.n);
    g_crush.ph = g_crush.ig = g_crush.w4 = g_crush.wc = nullptr; g_crush.pre = 0; g_crush.cax = 0;
}
typedef struct { crush_cache *cc; const crush_layer *L; const int64_t *o; } crush_ctu;
static double crush_ct_at(void *ud, const double x[3]) {
    const crush_ctu *u = ud; double p[3];
    for (int d = 0; d < 3; d++) p[d] = u->L->a * ((double)u->o[d] + x[d]) + u->L->b[d];
    return crush_linear_at(u->cc, u->L, p);
}
static int training_hits_holdout_box(const sampler *sp, const source *s, const int64_t lo[3], const int64_t n[3], int l) {   /* level-l box */
    for (int i = 0; i < sp->S->n; i++) {
        const source *h = &sp->S->src[i]; if (!src_nhold(h) || !same_ct(s, h)) continue;
        for (int k = 0; k < src_nhold(h); k++) {
            int hit = 1;
            for (int d = 0; d < 3; d++) { const int64_t a = lo[d] * ((int64_t)1 << l), e = (lo[d] + n[d]) * ((int64_t)1 << l); if (e <= hb_o(h, k, d) || a >= hb_o(h, k, d) + hb_n(h, k, d)) hit = 0; }
            if (hit) return 1;
        }
    }
    return 0;
}
/* --aug-xscan: the window's CT (n zyx at origin o, level l) replaced by one of the source's registered partner scans warped onto it
   (src/view.h: a finer partner box-downsampled in the same pass), its intensities mapped onto this window's own CT by quantile matching
   over the voxels both scans cover; where the partner has no data the CT is 0 (outside the scan: no labels there). Labels are not
   touched. 1 swapped, 0 kept (the partner covers under half of the window's scan voxels), -1 read error. */
static int xscan_draw(source *s, int l, const int64_t o[3], const int64_t n[3], rng *r, uint8_t *ctu, uint8_t *tmp) {
    const sview *v = s->xs_h[(int)(rnext(r) % (uint64_t)s->nxs)];
    const size_t N = (size_t)n[0] * n[1] * n[2];
    if (sview_read_level(v, l, o, n, tmp)) return -1;
    uint64_t ha[256] = {0}, hb[256] = {0}, both = 0, own = 0;
    for (size_t k = 0; k < N; k++) if (ctu[k]) { own++; if (tmp[k]) { both++; ha[ctu[k]]++; hb[tmp[k]]++; } }
    if (!own || both * 2 < own) return 0;
    uint8_t lut[256]; lut[0] = 0;
    uint64_t cb = 0, ca = 0; int a = 1;
    for (int b = 1; b < 256; b++) {   /* partner value b -> the window's value at the same quantile (the bin's midpoint) */
        const double q = (double)cb + .5 * (double)hb[b];
        while (a < 255 && (double)(ca + ha[a]) <= q) ca += ha[a++];
        lut[b] = (uint8_t)a; cb += hb[b];
    }
    for (size_t k = 0; k < N; k++) ctu[k] = lut[tmp[k]];
    return 1;
}
/* 1: ctu (P^3) and g_crush hold the crushed window; 0: not crushed (its footprint may reach a holdout box: the plain read follows); -1 error */
static int crush_draw_(sampler *sp, source *s, int l, const int64_t o[3], int P, int pshift, rng *r, uint8_t *arena, size_t abytes, uint8_t *ctu) {
    const sample_cfg *c = &sp->cfg; const size_t p3 = (size_t)P * P * P;
    int fine = s->crush_fine >= 0 ? s->crush_fine : c->crush.fine;
    z3 *fz = fine > 0 && l - fine >= 0 ? source_ct(s, l - fine) : nullptr;
    if (!fz) fine = 0;   /* (no finer level stored: the map's own level) */
    crush_cache cc;
    if (crush_cache_init(&cc, arena, abytes)) return 0;
    crush_layer Lct = {source_ct(s, l), CRUSH_LINEAR, 0, 0, 1, {0, 0, 0}, ctu}, Lph = {s->phase, CRUSH_PHASE, 0, PHASE_UNKNOWN, 0, {0, 0, 0}, nullptr};
    crush_layer_phase(&Lph, pshift);
    double hint[3] = {0, 0, 0}, n[3]; const double *hp = nullptr, cw = .5 * (P - 1);
    if (s->ax.n > 0) {   /* the radial direction at the window centre: the power iteration's seed, the normal when the phase is too sparse */
        const double sc = (double)(1 << l); double cy, cx; axis_at(&s->ax, ((double)o[0] + cw) * sc, &cy, &cx);
        hint[1] = (double)o[1] + cw - cy / sc; hint[2] = (double)o[2] + cw - cx / sc;
        if (hint[1] * hint[1] + hint[2] * hint[2] > 1e-6) hp = hint;
    }
    if (crush_phase_normal(&cc, &Lph, o, P, hp, n) < 32) {
        if (cc.err) { crush_cache_free(&cc); return -1; }
        if (hp) memcpy(n, hint, sizeof n); else { const double z = 2 * runif(r) - 1, a = 2 * M_PI * runif(r), q = sqrt(fmax(0, 1 - z * z)); n[0] = z; n[1] = q * cos(a); n[2] = q * sin(a); }
    }
    crush_map m; crush_map_init(&m, &c->crush, rnext(r), P, n);
    /* every read below (columns and pull) lies in the footprint box */
    double ext[3]; crush_footprint(&m, ext);
    int64_t lo[3], nn[3];
    for (int d = 0; d < 3; d++) { lo[d] = o[d] + (int64_t)floor(cw - ext[d]); nn[d] = (int64_t)ceil(2 * ext[d]) + 2; }
    if (training_hits_holdout_box(sp, s, lo, nn, l)) { crush_cache_free(&cc); return 0; }
    /* the voidness columns from the next coarser level when stored (1/8 of the tile decodes; the voidness is blurred anyway) */
    z3 *cz = source_ct(s, l + 1); crush_layer Lcol = Lct;
    if (cz) { Lcol.z = cz; crush_layer_level(&Lcol, -1); }
    crush_ctu u = {&cc, &Lcol, o};
    crush_map_columns(&m, c->crush.jlo, c->crush.jhi, rnext(r), crush_ct_at, &u);   /* (1: too little CT for a papyrus / void split: no collapse) */
    if (cc.err) { crush_map_free(&m); crush_cache_free(&cc); return -1; }
    uint8_t *ph = tp_get(p3 + 64, 0), *ig = s->phase_ign ? tp_get(p3, 0) : nullptr;
    g_crush.ph = ph; g_crush.ig = ig; g_crush.n = p3;
    if (!ph || (s->phase_ign && !ig)) { crush_map_free(&m); crush_cache_free(&cc); return -1; }
    crush_layer L[3]; int nl = 0;
    if (fine) { crush_layer f = {fz, CRUSH_LINEAR, 1, 0, 1, {0, 0, 0}, ctu}; crush_layer_level(&f, fine); L[nl++] = f; }
    else L[nl++] = Lct;
    Lph.out = ph; L[nl++] = Lph;
    if (ig) { crush_layer g = {s->phase_ign, CRUSH_NEAREST, 0, 0, 1, {0, 0, 0}, ig}; crush_layer_phase(&g, pshift); L[nl++] = g; }
    const int rc = crush_window(&m, &cc, o, L, nl);
    memset(ph + p3, PHASE_UNKNOWN, 64);
    g_crush.k = m.sq_k; g_crush.j0 = m.tab ? m.j0 : 1; g_crush.fine = fine; g_crush.tiles = cc.decoded;
    crush_map_free(&m); crush_cache_free(&cc);
    if (!rc) atomic_fetch_add(&sp->crushed, 1);
    return rc ? -1 : 1;
}
/* the tile cache lives in the window scratch (8 P^3: 512 slots at P 512); small windows (P < 192) borrow a pooled 48 MiB instead */
static int crush_draw(sampler *sp, source *s, int l, const int64_t o[3], int P, int pshift, rng *r, uint8_t *arena, size_t abytes, uint8_t *ctu) {
    const size_t own = (size_t)48 << 20;
    if (abytes >= own) return crush_draw_(sp, s, l, o, P, pshift, r, arena, abytes, ctu);
    uint8_t *a = tp_get(own, 0); if (!a) return -1;
    const int rc = crush_draw_(sp, s, l, o, P, pshift, r, a, own, ctu);
    tp_put(a, own);
    return rc;
}
/* --aug-tear: ctu (P^3) and g_crush (the phase, the winding mod 4 and its component) from a padded read cut and shifted (src/scrollaug.h).
   The labels are prepared on the padded box as the stored-phase path below would (sign, ignore mask, air, band_wind4_tiled), so the
   winding mod 4 keeps the true windings of each side: it jumps across the cut. 1 torn, 0 not (too little phase for the sheet normal,
   or the padded box reaches a holdout box: the plain read follows), -1 error. */
static int tear_draw(sampler *sp, source *s, int l, const int64_t o[3], int P, int pshift, rng *r, uint8_t *ctu) {
    const sample_cfg *c = &sp->cfg; const size_t p3 = (size_t)P * P * P;
    const double vox = s->um * (double)(1 << l), maxd = (c->tear_um > 0 ? c->tear_um : 480.f) / vox;
    const int M = (int)ceil(.5 * maxd) + 2, Pp = P + 2 * M; const size_t pp3 = (size_t)Pp * Pp * Pp;
    const int64_t po[3] = {o[0] - M, o[1] - M, o[2] - M}, pn[3] = {Pp, Pp, Pp};
    const uint64_t seed = rnext(r);
    if (training_hits_holdout_box(sp, s, po, pn, l)) return 0;
    uint8_t *ctp = tp_get(pp3, 0), *php = tp_get(pp3 + 64, 0);
    if (!ctp || !php) { tp_put(ctp, pp3); tp_put(php, pp3 + 64); return -1; }
    if (z3_read(source_ct(s, l), po, pn, ctp, 1) || phase_window(s->phase, po, Pp, Pp, pshift, php)) { tp_put(ctp, pp3); tp_put(php, pp3 + 64); return -1; }
    double n[3], spacing = 0;
    if (sa_phase_normal(php, Pp, Pp, Pp, M, P / 32 > 1 ? P / 32 : 1, n, &spacing) < 32 || !(spacing > 2)) { tp_put(ctp, pp3); tp_put(php, pp3 + 64); return 0; }
    sa_tear *t = &g_crush.tear; sa_tear_make(t, seed, P, n, spacing, maxd);
    if (sa_tear_margin(t) > M) { tp_put(ctp, pp3); tp_put(php, pp3 + 64); return 0; }   /* (cannot happen: |dp|, |dm| <= maxd / 2 + 1) */
    /* the labels as the stored-phase path prepares them, on the padded box */
    if (s->phase_sign < 0) phase_negate(php, pp3);
    if (s->phase_ign && pshift < 2) {
        uint8_t *ig = tp_get(pp3, 0);
        if (!ig || phase_window(s->phase_ign, po, Pp, Pp, pshift, ig)) { tp_put(ig, pp3); tp_put(ctp, pp3); tp_put(php, pp3 + 64); return -1; }
        for (size_t k = 0; k < pp3; k++) if (ig[k] && ig[k] != PHASE_UNKNOWN) php[k] = PHASE_UNKNOWN;
        tp_put(ig, pp3);
    }
    for (size_t k = 0; k < pp3; k++) if (!ctp[k]) php[k] = PHASE_UNKNOWN;   /* (the window's air test after the pull: the same voxels) */
    uint8_t *wwp = nullptr, *wcp = nullptr;
    if (c->wind4) {
        wwp = tp_get(pp3, 0); wcp = tp_get(pp3, 0);
        const int wt0 = c->wind4_tile < 0 ? 0 : c->wind4_tile > (P + 5) / 6 ? c->wind4_tile : (P + 5) / 6, wt = wt0 && wt0 < (Pp + 5) / 6 ? (Pp + 5) / 6 : wt0;
        if (!wwp || !wcp || band_wind4_tiled(php, Pp, Pp, Pp, wt, c->wind4_step > 0 ? c->wind4_step : 80, c->wind4_min > 0 ? (size_t)c->wind4_min : 512, wt ? 1 : 15, wwp, wcp, nullptr) < 0) {
            tp_put(wwp, pp3); tp_put(wcp, pp3); tp_put(ctp, pp3); tp_put(php, pp3 + 64); return -1;
        }
    }
    g_crush.n = p3; g_crush.pre = 1;
    g_crush.ph = tp_get(p3 + 64, 0);
    if (c->wind4) { g_crush.w4 = tp_get(p3, 0); g_crush.wc = tp_get(p3, 0); }
    if (!g_crush.ph || (c->wind4 && (!g_crush.w4 || !g_crush.wc))) { tp_put(wwp, pp3); tp_put(wcp, pp3); tp_put(ctp, pp3); tp_put(php, pp3 + 64); return -1; }
    sa_tear_apply(t, P, M, ctp, php, wwp, wcp, sa_ct_quantile(ctp, pp3, .05), ctu, g_crush.ph, g_crush.w4, g_crush.wc);
    memset(g_crush.ph + p3, PHASE_UNKNOWN, 64);
    if (s->phase_sign < 0) phase_negate(g_crush.ph, p3);   /* (the stored-phase path negates it again) */
    tp_put(wwp, pp3); tp_put(wcp, pp3); tp_put(ctp, pp3); tp_put(php, pp3 + 64);
    atomic_fetch_add(&sp->augn[AUGN_TEAR], 1);
    return 1;
}
/* --aug-delam: ctu (P^3) and g_crush.ph from a padded read with 1-3 peels / holes on papyrus (src/scrollaug.h); the labels (sign, ignore
   mask) prepared on the padded box and pulled with the peels, the winding mod 4 unwrapped by the ordinary path after (the map is
   continuous, the gaps keep the split's phase). 1 damaged, 0 not (no site found / holdout), -1 error. */
static int delam_draw(sampler *sp, source *s, int l, const int64_t o[3], int P, int pshift, rng *r, uint8_t *ctu) {
    const sample_cfg *c = &sp->cfg; const size_t p3 = (size_t)P * P * P;
    const int M = sa_delam_margin(0), Pp = P + 2 * M; const size_t pp3 = (size_t)Pp * Pp * Pp;
    const int64_t po[3] = {o[0] - M, o[1] - M, o[2] - M}, pn[3] = {Pp, Pp, Pp};
    const uint64_t seed = rnext(r);
    if (training_hits_holdout_box(sp, s, po, pn, l)) return 0;
    uint8_t *ctp = tp_get(pp3, 0), *php = tp_get(pp3 + 64, 0);
    if (!ctp || !php) { tp_put(ctp, pp3); tp_put(php, pp3 + 64); return -1; }
    if (z3_read(source_ct(s, l), po, pn, ctp, 1) || phase_window(s->phase, po, Pp, Pp, pshift, php)) { tp_put(ctp, pp3); tp_put(php, pp3 + 64); return -1; }
    double n[3], spacing = 0;
    const int iso = s->iso > 0 ? s->iso : c->iso > 0 ? c->iso : sa_ct_quantile(ctp, pp3, .5);
    sa_delam *d = &g_crush.delam;
    if (sa_phase_normal(php, Pp, Pp, Pp, M, P / 32 > 1 ? P / 32 : 1, n, &spacing) < 32 || !(spacing > 2) ||
        !sa_delam_make(d, seed, P, M, ctp, php, spacing, c->delam_hole, (uint8_t)(iso > 255 ? 255 : iso))) { tp_put(ctp, pp3); tp_put(php, pp3 + 64); return 0; }
    if (s->phase_sign < 0) phase_negate(php, pp3);
    if (s->phase_ign && pshift < 2) {
        uint8_t *ig = tp_get(pp3, 0);
        if (!ig || phase_window(s->phase_ign, po, Pp, Pp, pshift, ig)) { tp_put(ig, pp3); tp_put(ctp, pp3); tp_put(php, pp3 + 64); return -1; }
        for (size_t k = 0; k < pp3; k++) if (ig[k] && ig[k] != PHASE_UNKNOWN) php[k] = PHASE_UNKNOWN;
        tp_put(ig, pp3);
    }
    g_crush.n = p3; g_crush.pre = 1;
    if (!(g_crush.ph = tp_get(p3 + 64, 0))) { tp_put(ctp, pp3); tp_put(php, pp3 + 64); return -1; }
    sa_delam_apply(d, 1, P, M, ctp, php, ctu, g_crush.ph);
    memset(g_crush.ph + p3, PHASE_UNKNOWN, 64);
    if (s->phase_sign < 0) phase_negate(g_crush.ph, p3);   /* (the stored-phase path negates it again) */
    tp_put(ctp, pp3); tp_put(php, pp3 + 64);
    atomic_fetch_add(&sp->augn[AUGN_DELAM], 1);
    return 1;
}
/* --aug-curv: ctu (P^3) and g_crush (the phase, its ignore mask) pulled through a re-bend about a moved axis (crush.h
   crush_map_rebend: stage R alone, through the crush tile cache, so the footprint may be wider than the window). The source's axis
   gives r0 and the radial direction; without one the sheets are taken as flat, normal to the window's phase normal. Tighter (r1
   log-uniform P / 4 .. 0.9 r0, or 8 P when flat) 2 in 3, else flatter (r0 .. 4 r0); skipped when the source axis lies within
   0.75 P of the window centre. Voxels near the seam behind an output axis inside the window: CT 0, phase unknown. The moved axis
   feeds the radial input channels (g_crush.cax). 1 bent, 0 not, -1 error. */
/* the --aug-curv map of window o (stage R alone) from the draws u_kind / u_mag / u_sign: 1 made, 0 skipped (axis near the window, too
   little phase), -1 read error. cc: an initialised cache (the flat case reads the phase normal through it). */
static int curv_map(const crush_cfg *base, source *s, int l, const int64_t o[3], int P, crush_cache *cc, const crush_layer *Lph, uint64_t seed,
                    double u_kind, double u_mag, double u_sign, crush_map *m) {
    const double cw = .5 * (P - 1);
    double U[3], A[3] = {1, 0, 0}, r0 = 0;
    if (s->ax.n > 0) {
        const double sc = (double)(1 << l); double cy, cx, ya, xa, yb, xb;
        axis_at(&s->ax, ((double)o[0] + cw) * sc, &cy, &cx);
        /* the tilt over +-4 windows (hand-placed control points wiggle by ~30 deg over one window on Paris 4) */
        axis_at(&s->ax, ((double)o[0] + cw - 4. * P) * sc, &ya, &xa); axis_at(&s->ax, ((double)o[0] + cw + 4. * P) * sc, &yb, &xb);
        A[0] = 8. * P; A[1] = (yb - ya) / sc; A[2] = (xb - xa) / sc;
        const double la = sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]); for (int d = 0; d < 3; d++) A[d] /= la;
        double w[3] = {0, (double)o[1] + cw - cy / sc, (double)o[2] + cw - cx / sc};
        const double wa = w[0] * A[0] + w[1] * A[1] + w[2] * A[2]; for (int d = 0; d < 3; d++) w[d] -= wa * A[d];
        r0 = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
        if (r0 < .75 * P) return 0;   /* (the axis in or near the window: its own spiral) */
        for (int d = 0; d < 3; d++) U[d] = w[d] / r0;
        /* the local sheets' normal (outward) when the phase gives one: the rings stay tangent to the sheets through the centre (the
           hand-placed axis direction is ~30 deg off the sheet normal on Paris 4, and the rings then cross the sheets: spirals) */
        double n[3];
        if (crush_phase_normal(cc, Lph, o, P, U, n) >= 32) {
            const double sg = n[0] * U[0] + n[1] * U[1] + n[2] * U[2] < 0 ? -1 : 1, na = n[0] * A[0] + n[1] * A[1] + n[2] * A[2];
            double q[3]; for (int d = 0; d < 3; d++) q[d] = sg * (n[d] - na * A[d]);
            const double lq = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
            if (lq > .5) for (int d = 0; d < 3; d++) U[d] = q[d] / lq;
        } else if (cc->err) return -1;
    } else {   /* flat: the phase normal, either sign; the bend axis the scan axis (z) made perpendicular to it */
        double n[3];
        if (crush_phase_normal(cc, Lph, o, P, nullptr, n) < 32) return cc->err ? -1 : 0;
        for (int d = 0; d < 3; d++) U[d] = u_sign < .5 ? -n[d] : n[d];
        if (fabs(U[0]) > .9) { A[0] = 0; A[2] = 1; }
        const double ua = U[0] * A[0] + U[1] * A[1] + U[2] * A[2]; for (int d = 0; d < 3; d++) A[d] -= ua * U[d];
    }
    double r1;
    if (r0 == 0 || u_kind < 2. / 3) { const double lo = .6 * P, hi = r0 > 0 ? .9 * r0 : 8. * P; r1 = lo * pow(hi / lo, u_mag); }
    else r1 = r0 * pow(4., u_mag);
    crush_cfg z = *base; z.squash_p = z.bend_p = z.kink_p = 0; z.rotate_deg = 0;
    crush_map_init(m, &z, seed, P, U); crush_map_rebend(m, U, A, r0, r1);
    return 1;
}
static int curv_draw_(sampler *sp, source *s, int l, const int64_t o[3], int P, int pshift, rng *r, uint8_t *arena, size_t abytes, uint8_t *ctu) {
    const sample_cfg *c = &sp->cfg; const size_t p3 = (size_t)P * P * P; const double cw = .5 * (P - 1);
    const uint64_t seed = rnext(r); const double u_kind = runif(r), u_mag = runif(r), u_sign = runif(r);
    crush_cache cc;
    crush_layer Lct = {source_ct(s, l), CRUSH_LINEAR, 0, 0, 1, {0, 0, 0}, ctu}, Lph = {s->phase, CRUSH_PHASE, 0, PHASE_UNKNOWN, 0, {0, 0, 0}, nullptr};
    crush_layer_phase(&Lph, pshift);
    if (crush_cache_init(&cc, arena, abytes)) return 0;
    crush_map m;
    const int mk = curv_map(&c->crush, s, l, o, P, &cc, &Lph, seed, u_kind, u_mag, u_sign, &m);
    if (mk <= 0) { crush_cache_free(&cc); return mk; }
    const double r0 = m.r0, r1 = m.r1;
    double ext[3]; crush_footprint(&m, ext);
    int64_t lo[3], nn[3];
    for (int d = 0; d < 3; d++) { lo[d] = o[d] + (int64_t)floor(cw - ext[d]); nn[d] = (int64_t)ceil(2 * ext[d]) + 2; }
    if (training_hits_holdout_box(sp, s, lo, nn, l)) { crush_cache_free(&cc); return 0; }
    uint8_t *ph = tp_get(p3 + 64, 0), *ig = s->phase_ign ? tp_get(p3, 0) : nullptr;
    g_crush.ph = ph; g_crush.ig = ig; g_crush.n = p3;
    if (!ph || (s->phase_ign && !ig)) { crush_cache_free(&cc); return -1; }
    crush_layer L[3]; int nl = 0;
    L[nl++] = Lct; Lph.out = ph; L[nl++] = Lph;
    if (ig) { crush_layer g = {s->phase_ign, CRUSH_NEAREST, 0, 0, 1, {0, 0, 0}, ig}; crush_layer_phase(&g, pshift); L[nl++] = g; }
    const int rc = crush_window(&m, &cc, o, L, nl);
    g_crush.tiles = cc.decoded; crush_cache_free(&cc);
    if (rc) return -1;
    memset(ph + p3, PHASE_UNKNOWN, 64);
    if (r1 < .9 * P) {   /* (an output axis within reach of the window: blank the seam behind it, crush_rebend_masked(.85) row by row:
                            pu, pv linear along x, masked where pu < 0 and |pv| < tan(.15 pi) (-pu)) */
        const double tq = tan(.15 * M_PI), mc = m.c;
        for (int zz = 0; zz < P; zz++) for (int y = 0; y < P; y++) {
            const double bu = (zz - mc) * m.rU[0] + (y - mc) * m.rU[1] - mc * m.rU[2] + r1, bv = (zz - mc) * m.rV[0] + (y - mc) * m.rV[1] - mc * m.rV[2];
            if (bu >= 0 && bu + m.rU[2] * (P - 1) >= 0) continue;
            const size_t k0 = ((size_t)zz * P + y) * P;
            for (int x = 0; x < P; x++) {
                const double pu = bu + m.rU[2] * x, pv = bv + m.rV[2] * x;
                if (pu < 0 && fabs(pv) < -tq * pu) { ctu[k0 + x] = 0; ph[k0 + x] = PHASE_UNKNOWN; }
            }
        }
    }
    g_crush.r0 = r0; g_crush.r1 = r1;
    if (s->ax.n > 0) { g_crush.cax = 1; for (int d = 0; d < 3; d++) { g_crush.ca[d] = cw - r1 * m.rU[d]; g_crush.cA[d] = m.rA[d]; } }
    atomic_fetch_add(&sp->augn[AUGN_CURV], 1);
    return 1;
}
static int curv_draw(sampler *sp, source *s, int l, const int64_t o[3], int P, int pshift, rng *r, uint8_t *arena, size_t abytes, uint8_t *ctu) {
    const size_t own = (size_t)48 << 20;
    if (abytes >= own) return curv_draw_(sp, s, l, o, P, pshift, r, arena, abytes, ctu);
    uint8_t *a = tp_get(own, 0); if (!a) return -1;
    const int rc = curv_draw_(sp, s, l, o, P, pshift, r, a, own, ctu);
    tp_put(a, own);
    return rc;
}
/* --aug-sheet-density: the window's CT (Pz x P x P) re-lit per wrap from its stored phase (the pulled one after crush / tear / delam);
   returns 1 applied, 0 no usable phase, -1 error */
/* --aug-inclusions: grains, bubbles and debris pasted into the window's CT (Pz x P x P), labels unchanged; 1 applied, 0 none placed, -1 error */
static int incl_draw(sampler *sp, source *s, int Pz, int P, rng *r, uint8_t *ctu) {
    const sample_cfg *c = &sp->cfg; const size_t p3 = (size_t)Pz * P * P;
    const uint64_t seed = rnext(r);
    sa_incl *in = malloc(sizeof *in); if (!in) return -1;
    const int iso = s->iso > 0 ? s->iso : c->iso > 0 ? c->iso : sa_ct_quantile(ctu, p3, .5);
    const int n = sa_incl_make(in, seed, ctu, Pz, P, P, (uint8_t)(iso > 255 ? 255 : iso), sa_ct_quantile(ctu, p3, .05), 1);
    if (n) { sa_incl_apply(in, 1, ctu, Pz, P, P, ctu); atomic_fetch_add(&sp->augn[AUGN_INCL], 1); }
    free(in);
    return n > 0;
}
static int density_draw(sampler *sp, source *s, const int64_t o[3], int Pz, int P, int pshift, rng *r, uint8_t *ctu) {
    const sample_cfg *c = &sp->cfg; const size_t p3 = (size_t)Pz * P * P;
    const uint64_t seed = rnext(r);
    uint8_t *ph = tp_get(p3 + 64, 0);
    if (!ph) return -1;
    if (!g_crush.ph) {   /* the window's stored phase, kept for the label path (g_crush.pre 0: its conflict mask still applies) */
        if (!(g_crush.ph = tp_get(p3 + 64, 0))) { tp_put(ph, p3 + 64); return -1; }
        g_crush.n = p3; g_crush.pre = 0;
        if (phase_window(s->phase, o, Pz, P, pshift, g_crush.ph)) { tp_put(ph, p3 + 64); return -1; }
        memset(g_crush.ph + p3, PHASE_UNKNOWN, 64);
    }
    memcpy(ph, g_crush.ph, p3);
    if (s->phase_sign < 0) phase_negate(ph, p3);
    const int iso = s->iso > 0 ? s->iso : c->iso > 0 ? c->iso : sa_ct_quantile(ctu, p3, .5);
    sa_density d;
    const int ok = sa_density_make(&d, seed, ctu, ph, Pz, P, P, (uint8_t)(iso > 255 ? 255 : iso));
    if (ok) { sa_density_apply(&d, 1, ctu, ph, Pz, P, P, ctu); atomic_fetch_add(&sp->augn[AUGN_DENSITY], 1); }
    tp_put(ph, p3 + 64);
    return ok;
}
static int draw_(sampler *sp, batch *b, int i, rng *r, float *xtmp, uint8_t *ttmp, uint8_t *big, uint64_t tile_index, float *noise_state, chain_st *cs);
static int draw(sampler *sp, batch *b, int i, rng *r, float *xtmp, uint8_t *ttmp, uint8_t *big, uint64_t tile_index, float *noise_state, chain_st *cs) {
    const int rc = draw_(sp, b, i, r, xtmp, ttmp, big, tile_index, noise_state, cs);
    if (g_crush.ph || g_crush.ig || g_crush.w4 || g_crush.wc || g_crush.cax) crush_release();
    if (rc == 0 && b->chid && (!cs || cs->id < 0)) { b->chid[i] = -1; b->chidx[i] = 0; b->choff[3 * i] = b->choff[3 * i + 1] = b->choff[3 * i + 2] = 0.f; }
    return rc;
}
static int draw_(sampler *sp, batch *b, int i, rng *r, float *xtmp, uint8_t *ttmp, uint8_t *big, uint64_t tile_index, float *noise_state, chain_st *cs) {
    if (sp->cfg.restore == 2) return draw_restore_pair(sp, b, i, r, big);
    if (sp->cfg.restore) return draw_restore(sp, b, i, r, big);
    double pt = sp->prof ? tnow() : 0;
    const sample_cfg *c = &sp->cfg;
    const int P = c->P, Pz = PZ(c);   /* window Pz x P x P (Pz < P: a slab) */
    const size_t p3 = P3(sp);
    /* a window chain: continue (cont: the chain's next window; any rejection ends it) or start one (the decision drawn only with chains on) */
    const int chain_on = c->chain_p > 0 && cs && b->chid && !c->cover && !c->holdout && !c->deterministic;
    const int cont = chain_on && cs->left > 0, left0 = cont ? cs->left : 0;
    if (cs) { if (cont) cs->left = 0; else cs->id = -1; }
    const int starting = chain_on && !cont && runif(r) < c->chain_p;
    int chained = cont || starting;
    /* source by weight */
    double u = cont ? 0 : runif(r) * sp->cum[sp->S->n - 1];
    int si = 0;
    if (c->cover) si = (int)c->cover->tiles[tile_index][0];
    else if (cont) si = cs->si;
    else while (si < sp->S->n - 1 && sp->cum[si] < u) si++;
    source *s = &sp->S->src[si];
    if (c->holdout && !s->hold_n[0]) return 1;
    /* regions target? pick one region, then a level in {0,1} */
    int region_ch = -1, ri = -1;
    for (int ch = 0; ch < NCH; ch++) if (s->reg[ch] && s->reg[ch]->n) region_ch = ch;
    if (region_ch >= 0) chained = 0;
    if (cont && !chained) return 1;
    PROF_MARK(PS_SRC);
    int l = c->cover ? 0 : cont ? cs->l : pick_level(sp, s, r, region_ch);
    if (l < 0) return 1;
    /* zoom-out: the next coarser level, of which the warp below shows the central 1/2 zoom_out (an effective voxel size of this level's
       times zoom_out, from 2x coarser data); the dense-target tasks only (no band / phase / sheet / papyrus / condition / aux rasters) */
    double zoom_out = 0;
    if (!chained && !c->view_P && c->augment && c->geometry_augment && c->zoom_out_p > 0 && c->zoom_max > 1 && !c->cover && region_ch < 0 && s->geometry &&
        !c->band && !c->side && !c->sheet && !c->papyrus && !c->cond && !c->aux && runif(r) < c->zoom_out_p) {
        const double zf = exp(runif(r) * log(c->zoom_max > 2 ? 2 : c->zoom_max));
        if (l + 1 < MAXLEV && (s->max_level < 0 || l + 1 <= s->max_level) && level_ok(sp, s, si, l + 1, -1)) { l++; zoom_out = zf; }
    }
    PROF_MARK(PS_LEVEL);
    z3 *ct = source_ct(s, l);
    const z3_meta *m = z3_meta_of(ct);
    int64_t o[3], n[3] = {Pz, P, P};
    if (c->cover) memcpy(o, c->cover->tiles[tile_index] + 1, sizeof o);
    else if (cont) { if (sample_chain_next(&s->ax, l, m->shape, Pz, P, cs->mode, cs->dir, c->chain_step, cs->o, o)) return 1; }
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
    if (!c->cover && c->snap && !c->holdout && region_ch < 0 && !chained) {   /* training only: snapping a validation origin can leave its holdout */
        for (int d = 0; d < 3; d++) {
            int64_t cs = m->chunk[d]; if (cs <= 0) continue;
            o[d] -= o[d] % cs;
            if (o[d] > m->shape[d] - n[d]) o[d] = m->shape[d] - n[d];   /* the last window of an axis may stay unaligned */
            if (o[d] < 0) o[d] = 0;
        }
    }
    {   /* debugging / tests: UFSM_SAMPLE_AT="z,y,x[;z,y,x...]" (this level's voxels, cycled): replay given windows of the drawn source */
        static const char *at; static _Atomic unsigned nat; static int nc = -1; static int64_t ac[16][3];
        if (nc < 0) { at = getenv("UFSM_SAMPLE_AT"); int k = 0; for (const char *q = at; q && *q && k < 16; k++) {
                          long long a, b2, c2; if (sscanf(q, "%lld,%lld,%lld", &a, &b2, &c2) != 3) break; ac[k][0] = a; ac[k][1] = b2; ac[k][2] = c2;
                          q = strchr(q, ';'); if (q) q++; } nc = k; }
        if (nc > 0) { const unsigned k = atomic_fetch_add(&nat, 1) % (unsigned)nc;
                      for (int d = 0; d < 3; d++) { o[d] = ac[k][d]; if (o[d] > m->shape[d] - n[d]) o[d] = m->shape[d] - n[d]; if (o[d] < 0) o[d] = 0; } }
    }
    if (!c->holdout && training_hits_holdout(sp, s, o, l)) { atomic_fetch_add(&sp->rej_why[3], 1); return 1; }
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
        if ((double)nz / (double)tot < c->min_fg) { atomic_fetch_add(&sp->rejected, 1); atomic_fetch_add(&sp->rej_why[0], 1); return 1; }
    }
    PROF_MARK(PS_PROBE);
    /* CT (--aug-crush: a crushed window of a stored-phase source without target channels, pulled through the map instead) */
    uint8_t *ctu = big;
    int crushed = 0, torn = 0, delam = 0, curved = 0;
    if ((c->crush.p > 0 || c->tear_p > 0 || c->delam_p > 0 || c->curv_p > 0) && c->augment && c->geometry_augment && s->geometry && !chained && !c->view_P && !c->cover && !c->holdout && Pz == P &&
        region_ch < 0 && zoom_out == 0 && c->side == 2 && !c->band && !c->cond && !c->sheet && !c->papyrus && !c->pap_head && !c->aux &&
        !c->phase_dense && s->phase) {
        const int psh = s->phase_level - l;
        int ok = psh == 1 || (psh == 0 && s->phase_direct);
        for (int ch = 0; ch < NCH; ch++) ok &= !s->tgt_key[ch];
        if (ok && c->crush.p > 0 && runif(r) < c->crush.p && (crushed = crush_draw(sp, s, l, o, P, psh, r, big + p3, 8 * p3, ctu)) < 0) return -1;
        if (ok && !crushed && c->tear_p > 0 && runif(r) < c->tear_p && (torn = tear_draw(sp, s, l, o, P, psh, r, ctu)) < 0) return -1;
        if (ok && !crushed && !torn && c->delam_p > 0 && runif(r) < c->delam_p && (delam = delam_draw(sp, s, l, o, P, psh, r, ctu)) < 0) return -1;
        if (ok && !crushed && !torn && !delam && c->curv_p > 0 && runif(r) < c->curv_p && (curved = curv_draw(sp, s, l, o, P, psh, r, big + p3, 8 * p3, ctu)) < 0) return -1;
    }
    if (!crushed && !torn && !delam && !curved && z3_read(ct, o, n, ctu, 1)) return -1;
    PROF_MARK(PS_READ);
    size_t nz = 0;
    double sum = 0, sq = 0;
    u8_stats(ctu, p3, &nz, &sum, &sq);
    if (!c->cover && (double)nz / (double)p3 < c->min_fg) { atomic_fetch_add(&sp->rejected, 1); atomic_fetch_add(&sp->rej_why[1], 1); return 1; }
    if (c->xscan_p > 0 && s->nxs && !crushed && !torn && !delam && !curved && c->augment && !chained && !c->view_P && !c->cover && !c->holdout && runif(r) < c->xscan_p) {
        const int xs = xscan_draw(s, l, o, n, r, ctu, big + p3);   /* (big + p3: free scratch until the scanner augmentation) */
        if (xs < 0) return -1;
        if (xs) { atomic_fetch_add(&sp->augn[AUGN_XSCAN], 1); nz = 0; sum = sq = 0; u8_stats(ctu, p3, &nz, &sum, &sq); }
    }
    if (c->density_p > 0 && s->phase && c->augment && !chained && !c->view_P && !c->cover && !c->holdout && zoom_out == 0) {
        const int psh = s->phase_level - l;
        if ((psh == 1 || (psh == 0 && s->phase_direct)) && runif(r) < c->density_p) {
            const int dn = density_draw(sp, s, o, Pz, P, psh, r, ctu);
            if (dn < 0) return -1;
            if (dn) { nz = 0; sum = sq = 0; u8_stats(ctu, p3, &nz, &sum, &sq); }
        }
    }
    if (c->incl_p > 0 && c->augment && !chained && !c->view_P && !c->cover && !c->holdout && runif(r) < c->incl_p) {
        const int ni = incl_draw(sp, s, Pz, P, r, ctu);
        if (ni < 0) return -1;
        if (ni) { nz = 0; sum = sq = 0; u8_stats(ctu, p3, &nz, &sum, &sq); }
    }
    PROF_MARK(PS_STATS);
    /* --iso-void / --iso-lift: chamfer distance (3 per voxel, saturated at 255) of every voxel to papyrus (clean CT >= the cutoff) */
    const int iso_t = s->iso > 0 ? s->iso : c->iso;
    uint8_t *dpap = nullptr;
    if (iso_t > 0 && (c->iso_void || c->iso_lift > 0) && region_ch < 0) {
        uint8_t *pm = tp_get(p3, 0); dpap = tp_get(p3, 0);
        if (!pm || !dpap) { tp_put(pm, p3); tp_put(dpap, p3); return -1; }
        for (size_t k = 0; k < p3; k++) pm[k] = ctu[k] >= iso_t;
        chamfer345_8(pm, dpap, Pz, P);   /* no papyrus: all 255 */
        tp_put(pm, p3);
    }
    /* targets */
    uint8_t w[NCH] = {0};
    int tmax = 0;
    for (int ch = 0; ch < NCH; ch++) {
        uint8_t *dst = ttmp + (size_t)ch * p3;
        if (c->papyrus) { memset(dst, 0, p3); continue; }   /* task papyrus: labels from the CT below */
        if (ch == CH_SHEET && s->tgt_side[CH_RECTO]) continue;   /* the verso side: written with the recto side below */
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
            if (c->erode && !s->tgt_prob[ch] && !s->tgt_side[ch] && l == 0) {
                int64_t eo[3] = {o[0]-1, o[1]-1, o[2]-1}, en[3] = {P+2, P+2, P+2};
                uint8_t *haloed = big + p3;   /* existing (2P)^3 scratch, P >= 2 */
                if (source_read_target(s, ch, l, eo, en, haloed, 1)) return -1;
                target_erode1(haloed, dst, P);
            } else if (source_read_target(s, ch, l, o, n, dst, 1)) return -1;
            w[ch] = 1;
            if (s->tgt_side[ch]) {   /* side labels (band.h RV_*) -> channel 0 recto-side papyrus, channel 1 verso-side papyrus (clean CT >= the
                                        source's cutoff); void 0 in both; papyrus of unknown side ignored in both; CT 0 masked below */
                uint8_t *vs = ttmp + (size_t)CH_SHEET * p3; int any = 0;
                if (s->tgt_votes_soft[ch]) {   /* side votes kept as codes ("soft": 1): recto-side papyrus r / (r + v), verso the rest */
                    const uint8_t *lut = s->tgt_vote_lut[ch];
                    for (size_t k = 0; k < p3; k++) {
                        const uint8_t p = lut[dst[k]];
                        if (ctu[k] < iso_t) dst[k] = vs[k] = 0;
                        else if (p != 255) { dst[k] = p; vs[k] = (uint8_t)(254 - p); any = 1; }
                        else dst[k] = vs[k] = 255;
                    }
                } else {
                size_t k0 = 0;
#ifdef __AVX2__
                {   /* 32 voxels at a time: the same codes (CT below the cutoff first, then recto, then verso, else ignored) */
                    const __m256i t1 = _mm256_set1_epi8((char)(iso_t > 0 && iso_t <= 256 ? iso_t - 1 : 0)), rc = _mm256_set1_epi8(RV_RECTO), vc = _mm256_set1_epi8(RV_VERSO),
                                  c254 = _mm256_set1_epi8((char)254), c255 = _mm256_set1_epi8((char)255), z = _mm256_setzero_si256();
                    __m256i anyv = z;
                    for (; k0 + 32 <= p3; k0 += 32) {
                        const __m256i cv = _mm256_loadu_si256((const __m256i *)(ctu + k0)), sv = _mm256_loadu_si256((const __m256i *)(dst + k0));
                        const __m256i lt = iso_t <= 0 ? z : iso_t > 256 ? c255 : _mm256_cmpeq_epi8(_mm256_min_epu8(cv, t1), cv);   /* ctu < iso_t */
                        const __m256i re = _mm256_cmpeq_epi8(sv, rc), ve = _mm256_cmpeq_epi8(sv, vc);
                        const __m256i d = _mm256_blendv_epi8(_mm256_blendv_epi8(_mm256_blendv_epi8(c255, z, ve), c254, re), z, lt);
                        const __m256i v = _mm256_blendv_epi8(_mm256_blendv_epi8(_mm256_blendv_epi8(c255, c254, ve), z, re), z, lt);
                        anyv = _mm256_or_si256(anyv, _mm256_andnot_si256(lt, _mm256_or_si256(re, ve)));
                        _mm256_storeu_si256((__m256i *)(dst + k0), d); _mm256_storeu_si256((__m256i *)(vs + k0), v);
                    }
                    any = _mm256_movemask_epi8(anyv) != 0;
                }
#endif
                for (size_t k = k0; k < p3; k++) {
                    const uint8_t sv = dst[k];
                    if (ctu[k] < iso_t) dst[k] = vs[k] = 0;
                    else if (sv == RV_RECTO) { dst[k] = 254; vs[k] = 0; any = 1; }
                    else if (sv == RV_VERSO) { dst[k] = 0; vs[k] = 254; any = 1; }
                    else dst[k] = vs[k] = 255;
                }
                }
                w[CH_SHEET] = 1;
                if (any && !c->cover) tmax = 254;
            }
        } else memset(dst, 0, p3);
        if ((s->tgt_erode[ch] || s->tgt_dilate[ch]) && w[ch] && ch != region_ch) {   /* source band thickness: erode / dilate N voxels (3-4-5 chamfer) */
            uint8_t *mk = tp_get(p3, 0), *dm2 = tp_get(p3, 0);
            if (!mk || !dm2) { tp_put(mk, p3); tp_put(dm2, p3); tp_put(dpap, p3); return -1; }
            if (s->tgt_erode[ch]) {   /* band voxels within N of a non-band voxel (outside the window counts as band) */
                const int er = 3 * s->tgt_erode[ch] > 255 ? 255 : 3 * s->tgt_erode[ch];
                for (size_t k = 0; k < p3; k++) mk[k] = dst[k] == 0;
                if (chamfer345_8(mk, dm2, Pz, P)) for (size_t k = 0; k < p3; k++) if (dst[k] && dst[k] != 255 && dm2[k] <= er) dst[k] = 0;
            }
            if (s->tgt_dilate[ch]) {
                const int dr = 3 * s->tgt_dilate[ch] > 255 ? 255 : 3 * s->tgt_dilate[ch];
                for (size_t k = 0; k < p3; k++) mk[k] = dst[k] && dst[k] != 255;
                if (chamfer345_8(mk, dm2, Pz, P)) for (size_t k = 0; k < p3; k++) if (!dst[k] && dm2[k] <= dr) dst[k] = 254;
            }
            tp_put(mk, p3); tp_put(dm2, p3);
        }
        if (dpap && c->iso_void && w[ch] && s->tgt_prob[ch]) {   /* a teacher's probabilities: deep void is 0 here too (its noise floor) */
            const int vr = 3 * c->iso_pap > 255 ? 255 : 3 * c->iso_pap;
            for (size_t k = 0; k < p3; k++) if (ctu[k] && ctu[k] < iso_t && dpap[k] > vr) dst[k] = 0;
        }
        if (dpap && c->iso_lift > 0 && w[ch] && !s->tgt_prob[ch]) {   /* labels lifted off the sheets into the void: ignored */
            const int lr = 3 * c->iso_lift > 255 ? 255 : 3 * c->iso_lift;
            for (size_t k = 0; k < p3; k++) if (dst[k] && dst[k] != 255 && dpap[k] > lr) dst[k] = 255;
        }
        if (w[ch] && !c->cover) {   /* largest known target value (only the empty-patch rejection reads it) */
            uint8_t mx = 0; size_t k0 = 0;
#ifdef __AVX2__
            {   /* (an integer maximum: any order) */
                const __m256i c255 = _mm256_set1_epi8((char)255); __m256i m = _mm256_setzero_si256();
                for (; k0 + 32 <= p3; k0 += 32) { const __m256i d = _mm256_loadu_si256((const __m256i *)(dst + k0)); m = _mm256_max_epu8(m, _mm256_andnot_si256(_mm256_cmpeq_epi8(d, c255), d)); }
                uint8_t mb[32]; _mm256_storeu_si256((__m256i *)mb, m); for (int j = 0; j < 32; j++) mx = mb[j] > mx ? mb[j] : mx;
            }
#endif
            for (size_t k = k0; k < p3; k++) { const uint8_t v = dst[k] == 255 ? 0 : dst[k]; mx = v > mx ? v : mx; }
            if (mx > tmax) tmax = mx;
        }
    }
    /* empty windows (no target value above 0) kept with p empty_keep; a source without target channels (band codes + stored phase only:
       the --task unwrap2 sources) has none to test, so its windows are all kept (they took the same coin flip, 4 of 5 draws wasted) */
    int tgt_any = region_ch >= 0; for (int ch = 0; ch < NCH; ch++) tgt_any |= s->tgt_key[ch] != nullptr;
    if (!c->cover && !c->papyrus && tgt_any && tmax == 0 && runif(r) > c->empty_keep) { tp_put(dpap, p3); atomic_fetch_add(&sp->rejected, 1); atomic_fetch_add(&sp->rej_why[2], 1); return 1; }
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
    if (c->augment && c->label_morph>0 && !s->tgt_side[CH_RECTO] && runif(r)<c->label_morph_p) morph=(float)((2*runif(r)-1)*c->label_morph);
    uint16_t *dm = (uint16_t *)(big + 2 * p3);   /* chamfer distance to the annotated surface, per channel (3 per voxel); big has room */
    uint8_t *dm8 = big + 2 * p3;                  /* the same saturated at 255, when every threshold below is under 255 */
    int dm_ch = -1, dm_any = 0;                   /* channel the map currently holds, and whether that channel had any surface */
    const int soft_on = soft0 > 0 && region_ch < 0 && !s->tgt_side[CH_RECTO];   /* side labels: hard classes */
    float sigma = 0, shift = 0; int dcut = 0;
    if (soft_on) {
        sigma = (c->soft_um ? soft0 / (float)s->um : soft0) / (float)(1 << l); if (sigma < 0.75f) sigma = 0.75f;
        shift=morph/(float)(1<<l);
        /* exp(-(d/3)^2 / (2 sigma^2)) tabulated over the chamfer distance (3 per voxel); beyond dcut it rounds to 0 */
        dcut = (int)(3.f * (sigma * 4.5f+fmaxf(0,shift))) + 1; if (dcut > 4000) dcut = 4000;
    }
    const int trust_r = s->trust_band > 0 ? (s->trust_band >> l < 1 ? 1 : s->trust_band >> l) : 0;
    const int shell_r = c->shell > 0 && region_ch < 0 ? (c->shell >> l < 1 ? 1 : c->shell >> l) : 0;
    /* --dist-head: channel 1 = 254 exp(-(d / sigma)^2 / 2) of channel 0's hard labels (a wide distance ridge beside the sharp recto) */
    const int dist_on = c->dist_sigma > 0 && region_ch < 0 && w[0] && !s->tgt_prob[0] && NCH > 1;
    float dsig = 0; int dcut2 = 0;
    if (dist_on) {
        dsig = (c->soft_um ? c->dist_sigma / (float)s->um : c->dist_sigma) / (float)(1 << l); if (dsig < 0.75f) dsig = 0.75f;
        dcut2 = (int)(3.f * dsig * 4.5f) + 1; if (dcut2 > 4000) dcut2 = 4000;
    }
    const int use8 = (!soft_on || dcut <= 254) && 3 * trust_r <= 254 && 3 * shell_r <= 254 && (!dist_on || dcut2 <= 254);
#define CHAMFER_FOR(ch_) do { if (dm_ch != (ch_)) { dm_ch = (ch_); dm_any = use8 ? chamfer345_8(ttmp + (size_t)(ch_) * p3, dm8, Pz, P) : chamfer345(ttmp + (size_t)(ch_) * p3, dm, Pz, P); } } while (0)
    uint8_t *etab = big + 4 * p3, *dtab = big + 4 * p3 + 8192;   /* distance -> target tables (indexed by the chamfer distance, 3 per voxel) */
    if (soft_on) {
        for (int d = 0; d <= dcut; d++) { float df = fmaxf(0,d / 3.f-shift); int t = (int)(254.f * expf(-0.5f * df * df / (sigma * sigma)) + 0.5f); etab[d] = (uint8_t)(t > 254 ? 254 : t); }
        if (use8) for (int d = dcut + 1; d < 256; d++) etab[d] = 0;   /* indexed by the saturated distance */
    }
    if (dist_on) {
        for (int d = 0; d <= dcut2; d++) { float df = d / 3.f; int t = (int)(254.f * expf(-0.5f * df * df / (dsig * dsig)) + 0.5f); dtab[d] = (uint8_t)(t > 254 ? 254 : t); }
        if (use8) for (int d = dcut2 + 1; d < 256; d++) dtab[d] = 0;
    }
    /* per channel, all from that channel's hard labels (one distance map, computed before any of these modify the channel):
       --shell: background within R voxels of a label ignored (labels a voxel or two off the sheet); the soft ridge; the trust band
       (partial annotations: background trusted only within R voxels of a label, the rest ignored) */
    const int need_dm = soft_on || trust_r > 0 || shell_r > 0 || dist_on;
    for (int ch = 0; need_dm && ch < NCH; ch++) {
        if (dist_on && ch == 1) continue;   /* filled from channel 0 */
        if (!w[ch] || s->tgt_prob[ch]) continue;
        uint8_t *dst = ttmp + (size_t)ch * p3, *dd = dist_on && ch == 0 ? ttmp + p3 : nullptr;
        CHAMFER_FOR(ch);
        if (dd) {
            if (!dm_any) memset(dd, 0, p3);
            else if (use8) for (size_t k = 0; k < p3; k++) dd[k] = dst[k] == 255 ? 255 : dtab[dm8[k]];
            else for (size_t k = 0; k < p3; k++) dd[k] = dst[k] == 255 ? 255 : dm[k] <= dcut2 ? dtab[dm[k]] : 0;
            w[1] = 1;
        }
        if (dm_any && shell_r > 0) {
            const int sr = 3 * shell_r;
            if (use8) for (size_t k = 0; k < p3; k++) dst[k] = dst[k] == 0 && dm8[k] <= sr ? 255 : dst[k];
            else for (size_t k = 0; k < p3; k++) dst[k] = dst[k] == 0 && dm[k] <= sr ? 255 : dst[k];
        }
        if (dm_any && soft_on) {
            if (use8) for (size_t k = 0; k < p3; k++) dst[k] = dst[k] == 0 ? etab[dm8[k]] : dst[k];
            else for (size_t k = 0; k < p3; k++) if (dst[k] == 0 && dm[k] <= dcut) { uint8_t t = etab[dm[k]]; if (t) dst[k] = t; }
        }
        if (trust_r > 0) {
            const int vr = 3 * c->iso_pap > 255 ? 255 : 3 * c->iso_pap;
            const uint8_t *vp = c->iso_void ? dpap : nullptr;   /* --iso-void: deep void stays background beyond the band */
#define DEEP_VOID(k) (vp && ctu[k] && ctu[k] < iso_t && vp[k] > vr)
            for (int pass = 0; pass < 1 + (dd != nullptr); pass++) {   /* the distance channel takes channel 0's trust band */
                uint8_t *t = pass ? dd : dst;
                if (!dm_any) { for (size_t k = 0; k < p3; k++) if (t[k] == 0 && !DEEP_VOID(k)) t[k] = 255; continue; }   /* nothing annotated: ignore the background */
                const int tr = 3 * trust_r;
                if (use8) for (size_t k = 0; k < p3; k++) t[k] = t[k] == 0 && dm8[k] > tr && !DEEP_VOID(k) ? 255 : t[k];
                else for (size_t k = 0; k < p3; k++) t[k] = t[k] == 0 && dm[k] > tr && !DEEP_VOID(k) ? 255 : t[k];
            }
#undef DEEP_VOID
        }
    }
    tp_put(dpap, p3);
    PROF_MARK(PS_SOFT);
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
    if (g_crush.cax) for (int z = 0; z < Pz; z++) {   /* --aug-curv: the moved axis */
        const double t = (z - g_crush.ca[0]) / g_crush.cA[0];
        cyz[z] = (float)((double)o[1] + g_crush.ca[1] + t * g_crush.cA[1]); cxz[z] = (float)((double)o[2] + g_crush.ca[2] + t * g_crush.cA[2]);
    }
    if (c->augment && scan_aug_enabled(&c->scan)) {   /* scanner domains on the source grid; the per-patch z-score follows them, as at inference */
        PROF_MARK(PS_ZSCORE);
        if (scan_aug_apply(&c->scan, rnext(r), ctu, ttmp, NCH, ign, P, o, cyz, cxz, s->ax.n > 0, m->chunk, !c->sheet && !c->band && !c->side, big + p3)) {
            size_t nz2;   /* (integer sums: the same doubles as accumulating the bytes in double, exact below 2^53) */
            u8_stats(ctu, p3, &nz2, &sum, &sq);
            mean = sum / (double)p3; var = sq / (double)p3 - mean * mean; sd = sqrt(var > 0 ? var : 0) + 1e-3;
        }
        PROF_MARK(PS_SCAN);
    }
    if (s->norm_s > 0) { mean = s->norm_c; sd = s->norm_s; }   /* "norm": the scan's fixed normalisation (ufsm scanprep median / scale) instead of the window z-score */
    uint8_t *mask = big + 9 * p3;   /* ign lives here: fold CT > 0 into it */
    for (size_t k = 0; k < p3; k++) mask[k] = (ctu[k] != 0) & (mask[k] == 0);
    PROF_MARK(PS_ZSCORE);
    /* packed targets (cfg.pack_tm): unwarped samples pack the code in the source frame over the mask (one symmetry copy into the batch);
       warped ones warp into pooled scratch and pack after */
    uint8_t *T = c->pack_tm ? nullptr : b->t + (size_t)i * NCH * p3, *M = c->pack_tm ? nullptr : b->m + (size_t)i * p3;
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
    int symi = 0;   /* (a chain keeps its start's symmetry: the windows share one frame) */
    if (chained) {
        if (cont) y = sym_of(cs->symi);
        else for (int k = 0; k < 48; k++) { const sym t = sym_of(k); if (!memcmp(&t, &y, sizeof t)) { symi = k; break; } }
    }
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
    if (c->augment && c->geometry_augment && s->geometry && warp_ok && !chained && !c->view_P && !crushed) {   /* (crushed: already warped) */
        spatial_aug_make(&spatial,rnext(r),P,c->rotate_degrees,c->rotate_p,c->elastic,c->elastic_p);
        if (zoom_out > 0) spatial_aug_zoom(&spatial, .5 * zoom_out);   /* (the coarser level read above) */
        else if (c->zoom_p > 0 && c->zoom_min > 0 && c->zoom_min < 1 && runif(r) < c->zoom_p) spatial_aug_zoom(&spatial, exp(runif(r) * log(c->zoom_min)));
        if (c->affine_p > 0 && c->scan.strength > 0 && runif(r) < c->affine_p)
            spatial_aug_affine(&spatial, rnext(r), c->affine_aniso * c->scan.strength, c->affine_shear * c->scan.strength);
    }
    b->um[i] = (float)(s->um * (double)(1 << l) * (spatial.active ? spatial.scale * cbrt(spatial.det) : 1.0));   /* affine: the volume-preserving voxel size */
    if (c->papyrus) b->um[i] = c->papyrus->cond == 1 ? (float)(ia * ((pap_t - mean) / sd) + ib) : 0.f;   /* the cutoff where the input puts it */
    if (c->papyrus && c->papyrus->cond == 2) { mean = pap_t; ib = 0.f; }   /* the CT channel (ct - t) / sd: the cutoff at zero */
    uint8_t *Tw = nullptr;   /* (packed and warped: the warped targets and mask before packing) */
    uint8_t *TMd = nullptr;   /* packed, warped, CT-only, nothing else reading T / M: the warp writes the codes into the batch */
    if (c->pack_tm && spatial.active && c->gpu_input && !c->sheet && !c->pmask) TMd = b->tm + (size_t)i * p3;
    else if (c->pack_tm && spatial.active) {
        if (!(Tw = tp_get((NCH + 1) * p3, 0))) return -1;
        T = Tw; M = Tw + NCH * p3;
    }
    if (!spatial.active && c->pack_tm) {
        size_t bad = 0;
        if (sample_pack_tm(ttmp, ttmp + p3, mask, p3, mask, &bad)) {
            fprintf(stderr, "sampler: packed targets need hard 0 / 255 targets and a 0 / 1 mask (voxel %zu: %d %d %d); train --pack-targets 0\n",
                    bad, ttmp[bad], ttmp[p3 + bad], mask[bad]);
            abort();
        }
        sym_u8(mask, b->tm + (size_t)i * p3, Pz, P, y);
    } else if (!spatial.active) {
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
    /* --task unwrap2: the winding mod 4 and its component come from a stored phase below (warped or through the symmetry: every
       voxel written); else unknown (255) */
    int w4_set = 0;
    if (c->band || c->side) {   /* band labels: computed on the source grid, unknown on CT air, then the same symmetry as CT and targets */
        if (!band_sample || (spatial.active && !stored_phase)) {   /* no band labels: affinities (and sides) unsupervised */
            if (c->band) memset(b->band + (size_t)i * p3, BAND_UNKNOWN, p3);
            if (c->side) memset(b->side + (size_t)i * p3, SIDE_UNKNOWN, p3);
        } else if (stored_phase) {   /* the stored phase (ufsm phase-raster, ufsm downsample --mode phase) */
            uint8_t *sw = tp_get(p3 + 64, 0);   /* (+ 64: warp_phase_w4 reads byte pairs as 32-bit words) */
            uint16_t *wq = nullptr;   /* warped --task unwrap2: the winding and its component for warp_phase_w4 (one pass with the phase) */
            if (!sw || (g_crush.ph ? (memcpy(sw, g_crush.ph, p3), 0) : phase_window(s->phase, o, Pz, P, pshift, sw))) { tp_put(sw, p3 + 64); return -1; }
            if (s->phase_sign < 0) phase_negate(sw, p3);   /* (before the winding mod 4: it unwraps this phase) */
            if (s->phase_ign && pshift < 2 && !g_crush.pre) {   /* (--aug-tear: applied before the cut) (interpolated 2.4 um windows: no mask) "ignore": label sets that disagree there (ufsm phase-conflict): unknown, not trained on */
                uint8_t *ig = tp_get(p3, 0);
                if (!ig || (g_crush.ig ? (memcpy(ig, g_crush.ig, p3), 0) : phase_window(s->phase_ign, o, Pz, P, pshift, ig))) { tp_put(ig, p3); tp_put(sw, p3 + 64); return -1; }
                for (size_t k = 0; k < p3; k++) if (ig[k] && ig[k] != PHASE_UNKNOWN) sw[k] = PHASE_UNKNOWN;   /* (outside the mask array: 255 = keep) */
                tp_put(ig, p3);
            }
            if (c->wind4) {   /* the winding mod 4: the phase unwrapped before the air test (gaps join the windings), outside the scroll (CT 0) not */
                uint8_t *pw = tp_get(p3, 0), *ww = tp_get(p3, 0), *wc = tp_get(p3, 0);
                if (!pw || !ww || !wc) { tp_put(pw, p3); tp_put(ww, p3); tp_put(wc, p3); tp_put(sw, p3 + 64); return -1; }
                for (size_t k = 0; k < p3; k++) pw[k] = ctu[k] ? sw[k] : SIDE_UNKNOWN;
                const int wt = c->wind4_tile < 0 ? 0 : c->wind4_tile > (P + 5) / 6 ? c->wind4_tile : (P + 5) / 6;   /* (<= 6^3 tiles: one gauge each fits the loss's 255) */
                if (g_crush.w4) { memcpy(ww, g_crush.w4, p3); memcpy(wc, g_crush.wc, p3); }   /* (--aug-tear: unwrapped before the cut) */
                else if (band_wind4_tiled(pw, Pz, P, P, wt, c->wind4_step > 0 ? c->wind4_step : 80, c->wind4_min > 0 ? (size_t)c->wind4_min : 512, wt ? 1 : 15, ww, wc, nullptr) < 0) {
                    tp_put(pw, p3); tp_put(ww, p3); tp_put(wc, p3); tp_put(sw, p3 + 64); return -1;
                }
                if (spatial.active) {
                    if (!(wq = tp_get(2 * p3, 0))) { tp_put(pw, p3); tp_put(ww, p3); tp_put(wc, p3); tp_put(sw, p3 + 64); return -1; }
                    warp_w4_pack(ww, wc, wq, p3);
                }
                else { sym_u8(ww, b->w4 + (size_t)i * p3, Pz, P, y); sym_u8(wc, b->wcomp + (size_t)i * p3, Pz, P, y); }
                w4_set = 1;   /* (warped: warp_phase_w4 below writes every voxel) */
                tp_put(pw, p3); tp_put(ww, p3); tp_put(wc, p3);
            }
            for (size_t k = 0; k < p3; k++) if (ctu[k] <= c->side_air || !ctu[k]) sw[k] = SIDE_UNKNOWN;
            if (c->phase_near_um > 0) for (size_t k = 0; k < p3; k++) if (!near || !near[k]) sw[k] = SIDE_UNKNOWN;   /* no labelled wrap: none */
            if (spatial.active) {   /* (slabs never warp: sampler_start) */
                memset(sw + p3, PHASE_UNKNOWN, 64);
                warp_phase_w4(sw, wq, b->side + (size_t)i * p3, wq ? b->w4 + (size_t)i * p3 : nullptr, wq ? b->wcomp + (size_t)i * p3 : nullptr, P, y, &spatial);
                tp_put(wq, 2 * p3);
            } else sym_u8(sw, b->side + (size_t)i * p3, Pz, P, y);
            tp_put(sw, p3 + 64);
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
    if (c->wind4 && !w4_set) { memset(b->w4 + (size_t)i * p3, 255, p3); memset(b->wcomp + (size_t)i * p3, 255, p3); }
    if (c->emb_ch) {   /* train --distill: the feature targets through the same symmetry and warp (none on crushed / torn / re-bent windows) */
        const size_t ge = (size_t)(Pz / c->emb_div) * (P / c->emb_div) * (P / c->emb_div);
        int8_t *eo = b->emb + (size_t)i * c->emb_ch * ge;
        if (crushed || torn || curved) memset(eo, 0, ge * c->emb_ch);
        else if (sample_emb_window(s, c->emb_ch, c->emb_div, l, o, Pz, P, y.perm, y.flip, &spatial, eo)) return -1;
    }
    PROF_MARK(PS_AUGMENT);
    const size_t xc = c->gpu_input ? 1 : 4 + (c->aux != 0);
    if (c->gpu_input && (c->sheet || (app && Pz != P))) { fprintf(stderr, "sampler: --gpu-input needs write_x / write_spatial samples (sample_gpu_input_ok)\n"); abort(); }
    float *X = c->xfmt ? nullptr : b->x + (size_t)i * xc * p3;
    uint16_t *H = c->xfmt ? b->x16 + (size_t)i * xc * p3 : nullptr;
    double (*params)[4]=nullptr;
    if (c->sheet) {
        params=malloc((size_t)P*sizeof *params);
        for (int sz=0;sz<P;sz++) sheet_parameters(c->sheet,o[0]+sz,params[sz]);
    }
    if (spatial.active) write_spatial_tm(ctu,ttmp,mask,T,M,P,o,(float)mean,(float)(1.0/sd),s->ax.n>0,cyz,cxz,y,&spatial,
            ia,ib,isg,r,nrow,c->xfmt,X,H,app,noise_state,noise_state?noise_state+P:nullptr,c->sheet,params,(float *)(big+p3),c->pack_tm,c->gpu_input,TMd);
    else if (c->sheet || (app && Pz != P)) write_x_aug(ctu, P, o, (float)mean, (float)(1.0 / sd), s->ax.n > 0, cyz, cxz, y, c->augment != 0, ia, ib, isg, r, nrow, c->xfmt,
            X, H, app, noise_state, noise_state ? noise_state + P : nullptr,c->sheet,params);
    else write_x(ctu, Pz, P, o, (float)mean, (float)(1.0 / sd), s->ax.n > 0, cyz, cxz, y, c->augment != 0, ia, ib, isg, r, nrow, c->xfmt, X, H,
                 app, noise_state, noise_state ? noise_state + P : nullptr, c->gpu_input);   /* the CT appearance plan in the row-vectorised writer (cubes) */
    if (c->gpu_input) rad_record((char *)b->rad + (size_t)i * NN_RAD_REC(Pz), o, Pz, P, s->ax.n > 0, y, cyz, cxz, &spatial);   /* the radial channels: on the device */
    if (c->axis_rec > 0) axis_record(b->axr + (size_t)i * SAMPLE_AXR(Pz > P ? Pz : P), o, Pz, P, s->ax.n > 0, y, cyz, cxz, &spatial, c->axis_rec);
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
    if (c->side == 2 && c->min_phase > 0) {   /* --min-phase: windows with (almost) no known phase supervise nothing (sources with band / phase
                                                 targets only draw anywhere in the scroll): rejected */
        const uint8_t *sd = b->side + (size_t)i * p3; size_t kn = 0;
        for (size_t k = 0; k < p3; k++) kn += sd[k] < PHASE_PERIOD;
        if ((double)kn < c->min_phase * (double)p3) { if (Tw) tp_put(Tw, (NCH + 1) * p3); atomic_fetch_add(&sp->rejected, 1); atomic_fetch_add(&sp->rej_why[2], 1); return 1; }   /* (counted as empty targets) */
    }
    if (c->cond) {   /* inputs 4-6 and batch.wind from the source's winding raster (on the stored phase's grid; air unknown like the phase) */
        uint8_t *ww = tp_get(p3, 0), *ws = tp_get(p3, 0);
        if (!ww || !ws) { tp_put(ww, p3); tp_put(ws, p3); return -1; }
        if (stored_phase && s->wind) {
            if (phase_window(s->wind, o, Pz, P, pshift, ww)) { tp_put(ww, p3); tp_put(ws, p3); return -1; }
            if (s->phase_sign < 0) phase_negate(ww, p3);   /* the winding mod 14 turns in the same encoding: W -> -W */
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
    if (Tw) {
        size_t bad = 0;
        if (sample_pack_tm(Tw, Tw + p3, Tw + NCH * p3, p3, b->tm + (size_t)i * p3, &bad)) {
            fprintf(stderr, "sampler: packed targets need hard 0 / 255 targets and a 0 / 1 mask after the warp (voxel %zu); train --pack-targets 0\n", bad);
            abort();
        }
        tp_put(Tw, (NCH + 1) * p3);
    }
    PROF_MARK(PS_X16);
    {   /* a non-finite input value (an augmentation dividing by a degenerate window's statistics): the window is rejected. The network
           would not show it (the MX input conversion's amax ignores NaN) but the stem's weight gradient reads the 16-bit CT channel, so
           the step's gradient goes non-finite (skipped, the gradient scale halved) on every replay of the window */
        size_t bad = 0, k0 = 0; const size_t nx = xc * p3;
        if (X) { for (size_t k = 0; k < nx; k++) if (!isfinite(X[k])) { if (!bad) k0 = k; bad++; } }
        else { const uint16_t e = c->xfmt == 1 ? 0x7c00 : 0x7f80; for (size_t k = 0; k < nx; k++) if ((H[k] & e) == e) { if (!bad) k0 = k; bad++; } }
        if (bad) {
            static _Atomic int nwarn;
            if (atomic_fetch_add(&nwarn, 1) < 20)
                fprintf(stderr, "sampler: %s level %d corner %lld,%lld,%lld: %zu non-finite input values (first at channel %zu voxel %zu): window rejected\n",
                        s->name, l, (long long)(o[0] << l), (long long)(o[1] << l), (long long)(o[2] << l), bad, k0 / p3, k0 % p3);
            return 1;
        }
    }
    memcpy(b->w + (size_t)i * NCH, w, NCH);
    b->src[i] = (int16_t)si;
    b->level[i] = (int8_t)l;
    memcpy(b->corner[i], o, sizeof o);
    for (int d = 0; d < 3; d++) b->corner[i][d] <<= l;
    if (c->view_P) {   /* scan fusion: the view crop centred on the window, the window's symmetry (unwarped windows: no spatial aug with views) */
        const int Pv = c->view_P; const size_t pv3 = (size_t)Pv * Pv * Pv; uint8_t *dst = b->xv + (size_t)i * pv3;
        b->vok[i] = 0;
        const sview *vw = s->view_h;
        if (vw && l >= vw->lv) {
            uint8_t *tmp = malloc(pv3); int64_t vo[3]; const int64_t wn[3] = {Pz, P, P};
            sview_origin(l, vw->lv, o, wn, Pv, vo);
            if (!tmp || sview_read(vw, vw->has_map ? nullptr : source_ct(s, vw->lv), vo, Pv, tmp)) { free(tmp); return 1; }
            sym_u8(tmp, dst, Pv, Pv, y); free(tmp); b->vok[i] = 1;
        } else memset(dst, 0, pv3);
    }
    if (chained) {   /* the chain's bookkeeping and the window's offset from its previous one in the window frame */
        float off[3] = {0.f, 0.f, 0.f};
        if (cont) {
            for (int d = 0; d < 3; d++) off[d] = (float)((y.flip[d] ? -1 : 1) * (double)(o[y.perm[d]] - cs->o[y.perm[d]]) / P);
            cs->idx++; cs->left = left0 - 1;
        } else {
            static _Atomic int64_t chain_ids;
            cs->id = atomic_fetch_add(&chain_ids, 1); cs->idx = 0; cs->left = c->chain_len - 1;
            cs->si = si; cs->l = l; cs->symi = symi; cs->mode = runif(r) < 0.5 ? 0 : 1; cs->dir = runif(r) < 0.5 ? -1 : 1;
        }
        memcpy(cs->o, o, sizeof o);
        b->chid[i] = cs->id; b->chidx[i] = cs->idx; memcpy(b->choff + 3 * i, off, sizeof off);
    }
    atomic_fetch_add(&sp->produced, 1);
    return 0;
}

typedef struct { sampler *sp; int id; } warg;

int sample_pack_tm(const uint8_t *t0, const uint8_t *t1, const uint8_t *m, size_t n, uint8_t *out, size_t *bad) {
    uint8_t acc = 0;   /* any value outside the code (checked once after the loop: the loop stays branch-free) */
    for (size_t k = 0; k < n; k++) {
        const uint8_t a = t0[k], b = t1[k], mk = m[k];
        acc |= (uint8_t)((a != 0 && a != 255) | (b != 0 && b != 255) | (mk > 1));
        out[k] = (uint8_t)((a >> 7 & SAMPLE_TM_T0) | (b >> 6 & SAMPLE_TM_T1) | (mk << 2 & SAMPLE_TM_M));
    }
    if (!acc) return 0;
    for (size_t k = 0; k < n; k++) if ((t0[k] != 0 && t0[k] != 255) || (t1[k] != 0 && t1[k] != 255) || m[k] > 1) { if (bad) *bad = k; break; }
    return -1;
}
static void *worker(void *arg) {
    warg *wa = arg;
    sampler *sp = wa->sp;
    rng r;
    rseed(&r, sp->cfg.seed * 1000003ull + (uint64_t)wa->id);
    size_t p3 = P3(sp);
    float *xtmp = malloc(3 * (size_t)sp->cfg.P * sizeof(float));   /* per-z axis centres and a row of noise */
    uint8_t *ttmp = wbig_alloc(NCH * p3, 0);
    uint8_t *big = wbig_alloc(10 * p3, 0);   /* [0,P^3) CT (also the coarse probe), [P^3, 9P^3) (2P)^3 region scratch, [9P^3, 10P^3) mask */
    const int det = sp->cfg.deterministic;
    float *noise_state = sp->cfg.ct_augment ? malloc(((size_t)sp->cfg.P * sp->cfg.P + sp->cfg.P) * sizeof(float)) : nullptr;
    chain_st *chs = calloc((size_t)sp->cfg.B, sizeof *chs);   /* (window chains: one per sample slot) */
    for (int k = 0; chs && k < sp->cfg.B; k++) chs[k].id = -1;
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
            int rc = draw(sp, b, i, &r, xtmp, ttmp, big, sp->cfg.cover_start + (uint64_t)(jb * sp->cfg.B + i), noise_state, chs ? chs + i : nullptr);
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
    free(xtmp); free(ttmp); free(big); free(noise_state); free(chs); tp_drain();
    free(wa);
    return nullptr;
}

/* band codes + stored phase sources (no recto target; the --task unwrap2 sources): the stored phase's chunks that hold any known
   phase, from its shard indices (all-unknown chunks are not stored); a cell is one chunk, i.e. a voxel of level phase_level +
   log2(chunk). Windows are then drawn around labelled chunks instead of uniformly over the scan (on the desk's sparsely labelled
   scrolls ~93% of uniform draws were rejected, many after the CT read), and ufsm prefetch covers these sources. */
static occ_index phase_occupancy(source *s) {
    occ_index o = {nullptr, 0, -1, {0, 0, 0}};
    const z3_meta *m = z3_meta_of(s->phase); const int64_t c = m->chunk[0]; int e = 0;
    while (((int64_t)1 << e) < c) e++;
    if (c < 1 || ((int64_t)1 << e) != c || m->chunk[1] != c || m->chunk[2] != c) return o;
    int64_t g[3]; uint8_t *bits = nullptr; const int64_t np = z3_chunk_map(s->phase, g, &bits);
    if (np <= 0 || (double)g[0] * g[1] * g[2] >= 4.29e9) { free(bits); return o; }
    uint32_t *idx = malloc((size_t)np * sizeof *idx); size_t n = 0;
    if (!idx) { free(bits); return o; }
    for (int64_t k = 0; k < g[0] * g[1] * g[2]; k++) if (bits[k]) idx[n++] = (uint32_t)k;
    free(bits);
    o.idx = idx; o.n = n; o.lev = s->phase_level + e; for (int d = 0; d < 3; d++) o.shape[d] = g[d];
    fprintf(stderr, "sampler: %s: %zu of %lld stored-phase chunks (level %d cells) hold labels\n", s->name, n, (long long)(g[0] * g[1] * g[2]), o.lev);
    return o;
}
/* coarsest recto level that fits in memory: the cells containing any surface fraction (fallback: any labelled cell) */
static occ_index label_occupancy(source *s) {
    occ_index o = {nullptr, 0, -1, {0, 0, 0}};
    if (s->reg[0] && s->reg[0]->n) return o;
    /* the coarsest label level that exists (level >= 3); large scans (Paris 4: 2.5e9 cells at level 5) are read in z slabs,
       so there is no size cap besides the 32-bit cell index (a 3e8 cap used to leave such sources without an index, i.e.
       sampled uniformly over the whole volume) */
    int lc = -1; z3 *tz = nullptr;
    for (int l = MAXLEV - 1; l >= 3; l--) { z3 *t = source_tgt(s, 0, l); if (t) { const z3_meta *m = z3_meta_of(t); if ((double)m->shape[0] * m->shape[1] * m->shape[2] < 4.29e9) { lc = l; tz = t; break; } } }
    if (!tz && s->band && s->phase) return phase_occupancy(s);
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
/* an unlabelled scan's ufsm scanprep mask: the cells with code >= mask_min (2: valid window centres) */
static occ_index mask_occupancy(source *s) {
    occ_index o = {nullptr, 0, -1, {0, 0, 0}};
    const size_t tot = (size_t)s->mask_grid[0] * s->mask_grid[1] * s->mask_grid[2];
    size_t n = 0; for (size_t k = 0; k < tot; k++) n += s->mask[k] >= s->mask_min;
    if (!n || !(o.idx = malloc(n * sizeof *o.idx))) return o;
    for (size_t k = 0; k < tot; k++) if (s->mask[k] >= s->mask_min) o.idx[o.n++] = (uint32_t)k;
    o.lev = s->mask_lev; for (int d = 0; d < 3; d++) o.shape[d] = s->mask_grid[d];
    fprintf(stderr, "sampler: %s: %zu of %zu mask cells (level-%d cells) with code >= %d\n", s->name, n, tot, o.lev, s->mask_min);
    return o;
}
occ_index source_occupancy(source *s) {
    occ_index o = label_occupancy(s);
    if (!o.n && s->mask) { free(o.idx); o = mask_occupancy(s); }
    return o;
}
/* Pull the CT of every occupied cell at levels 0..maxlev through the chunk cache (training then reads from local disk).
   Cells are visited in a shuffled order so a partial prefetch is still useful; one thread per in-flight region. */
/* Prefetch the CT chunks that training touches into the chunk cache: for every coarse cell that contains surface,
   at every level the window around the cell (P wide, P/2 random offset, then snapped to the chunk grid) and the coarse
   probe window at level + 3; chunks are collected in a bitmap per level so each is fetched once, without decoding. */
typedef struct { z3 *ct; const uint32_t *list; size_t n; int64_t ng[3]; size_t *next, *done, *fetched, *fail; } pf_arg;
static void *pf_worker(void *p) {   /* the marked chunks, from one shared list: they cluster around the labels, so per-thread index ranges
                                       left most threads idle (2 MB/s from the mirror at 56 threads) */
    pf_arg *a = p;
    for (;;) {
        const size_t i = __atomic_fetch_add(a->next, 1, __ATOMIC_RELAXED); if (i >= a->n) break;
        const size_t k = a->list[i];
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
int sources_prefetch(sources *S, int maxlev, int nthreads, double fraction, int P, int dry) {
    if (P < 8) P = 128;   /* training window (level voxels); windows up to this size are covered */
    size_t all = 0;
    for (int i = 0; i < S->n; i++) {
        source *s = &S->src[i];
        occ_index o = source_occupancy(s);
        if (!o.n) { fprintf(stderr, "prefetch: %s: no occupancy index (regions source?), skipped\n", s->name); continue; }
        size_t ncell = (size_t)(o.n * fraction); if (ncell < 1) ncell = 1; if (ncell > o.n) ncell = o.n;
        for (int l = 0; l < MAXLEV; l++) {
            z3 *ct = source_ct(s, l); if (!ct) continue;
            const z3_meta *m = z3_meta_of(ct);
            const int top = s->max_level >= 0 && s->max_level < maxlev ? s->max_level : maxlev;   /* the levels this source trains on */
            int want_window = l <= top && l >= s->min_level, want_probe = l >= 3 && l - 3 <= top && l - 3 >= s->min_level;   /* probe reads at level l serve training levels l-3 */
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
            all += nmark;
            if (dry) { fprintf(stderr, "prefetch: %s level %d: %zu chunks needed (window %d)\n", s->name, l, nmark, P); free(bits); continue; }
            uint32_t *list = malloc((nmark ? nmark : 1) * sizeof *list); size_t nl = 0;
            for (size_t k = 0; k < total; k++) if ((bits[k >> 3] >> (k & 7)) & 1) list[nl++] = (uint32_t)k;
            size_t done = 0, fetched = 0, fail = 0, next = 0; pthread_t *th = malloc(nthreads * sizeof *th); pf_arg *args = malloc(nthreads * sizeof *args);
            for (int t = 0; t < nthreads; t++) { args[t] = (pf_arg){ct, list, nl, {ng[0], ng[1], ng[2]}, &next, &done, &fetched, &fail}; pthread_create(&th[t], nullptr, pf_worker, &args[t]); }
            size_t last = 0;
            while (last < nmark) { usleep(1000000); last = __atomic_load_n(&done, __ATOMIC_RELAXED); fprintf(stderr, "prefetch: %s level %d: %zu / %zu chunks checked, %zu fetched\r", s->name, l, last, nmark, __atomic_load_n(&fetched, __ATOMIC_RELAXED)); }
            for (int t = 0; t < nthreads; t++) pthread_join(th[t], nullptr);
            free(list);
            free(th); free(args); free(bits);
            fprintf(stderr, "\nprefetch: %s level %d done (%zu chunks needed, %zu fetched now, %zu failures)\n", s->name, l, nmark, fetched, fail);
        }
        free(o.idx);
    }
    fprintf(stderr, "prefetch: %zu chunks over all sources (window %d)\n", all, P);
    return 0;
}
sampler *sampler_start(sources *S, const sample_cfg *cfg) {
    if (!S || S->n <= 0 || cfg->P <= 0 || cfg->B <= 0 || cfg->nworkers <= 0 || cfg->nbuf <= 0) { fprintf(stderr, "sampler: invalid sources, patch, batch, workers or buffer count\n"); return nullptr; }
    if (cfg->gpu_input && !sample_gpu_input_ok(cfg)) { fprintf(stderr, "sampler: --gpu-input needs 16-bit batches and no warps, sheet prior, slab appearance plan, aux or condition inputs\n"); return nullptr; }
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
    if (cfg->view_P) {   /* scan fusion: open the sources' views (kept for the process: train / val samplers share them) */
        if (cfg->view_P < 0 || cfg->view_P & 1 || cfg->restore || cfg->sheet) { fprintf(stderr, "sampler: view_P must be even, without restore / winding-geometry tasks\n"); return nullptr; }
        int nv = 0;
        for (int i = 0; i < S->n; i++) {
            source *s = &S->src[i];
            if (!s->view) continue;
            if (!s->view_h && !(s->view_h = sview_open(s->view_map, s->view_level, s->um, S->cache))) { fprintf(stderr, "sampler: %s: cannot open the view\n", s->name); return nullptr; }
            if (!s->view_map && !source_ct(s, s->view_level)) { fprintf(stderr, "sampler: %s: no CT level %d for the view\n", s->name, s->view_level); return nullptr; }
            nv++;
        }
        if (!nv) fprintf(stderr, "sampler: warning: view_P %d but no source has a \"view\" (zero crops)\n", cfg->view_P);
    }
    if (cfg->xscan_p > 0) {   /* --aug-xscan: open the partner scans (kept for the process, as the views) */
        int nx = 0;
        for (int i = 0; i < S->n; i++) {
            source *s = &S->src[i];
            for (int k = 0; k < s->nxs; k++, nx++)
                if (!s->xs_h[k] && !(s->xs_h[k] = sview_open(s->xs_map[k], 0, s->um, S->cache))) { fprintf(stderr, "sampler: %s: cannot open xscan map %s\n", s->name, s->xs_map[k]); return nullptr; }
        }
        if (!nx) fprintf(stderr, "sampler: warning: --aug-xscan %g but no source has \"xscan\" partner scans\n", cfg->xscan_p);
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
        if (cfg->restore == 2) {   /* real SR pairs: the pair's boxes for this mode */
            eligible = s->srp && (cfg->holdout ? s->srp->nte : s->srp->ntr) > 0 && cfg->P % 4 == 0 && cfg->P / 4 <= s->srp->B && PZ(cfg) == cfg->P;
            if (!(s->weight >= 0)) eligible = 0;
            acc += eligible ? s->weight : 0; sp->cum[i] = acc; continue;
        }
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
    for (int i = 0; !cfg->cover && cfg->restore != 2 && i < S->n; i++) {
        occ_index o = source_occupancy(&S->src[i]);
        const source *fs = &S->src[i];
        if (o.n && fs->focus && fs->focus_w > 0) {   /* "focus": cell k drawn 1 + w x packed share times as often (repeated in the index) */
            if (fs->focus_grid[0] != o.shape[0] || fs->focus_grid[1] != o.shape[1] || fs->focus_grid[2] != o.shape[2])
                fprintf(stderr, "sampler: %s: the focus map grid %lldx%lldx%lld differs from the occupancy grid %lldx%lldx%lld: focus off\n", fs->name,
                        (long long)fs->focus_grid[0], (long long)fs->focus_grid[1], (long long)fs->focus_grid[2], (long long)o.shape[0], (long long)o.shape[1], (long long)o.shape[2]);
            else {
                size_t nn = 0; int *rep = malloc(o.n * sizeof *rep);
                for (size_t k = 0; k < o.n; k++) { const uint8_t v = fs->focus[o.idx[k]]; rep[k] = 1 + (v ? (int)lround(fs->focus_w * (v - 1) / 254.0) : 0); nn += (size_t)rep[k]; }
                uint32_t *ix = malloc(nn * sizeof *ix); size_t q = 0;
                if (ix && rep) {
                    for (size_t k = 0; k < o.n; k++) for (int t = 0; t < rep[k]; t++) ix[q++] = o.idx[k];
                    fprintf(stderr, "sampler: %s: focus w %g: %zu cells -> %zu draws (x%.2f)\n", fs->name, fs->focus_w, o.n, nn, (double)nn / o.n);
                    free(o.idx); o.idx = ix; o.n = nn;
                } else free(ix);
                free(rep);
            }
        }
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
void sampler_reject_reasons(const sampler *sp, uint64_t why[4]) { for (int k = 0; k < 4; k++) why[k] = atomic_load(&sp->rej_why[k]); }
uint64_t sampler_crushed(const sampler *sp) { return atomic_load(&sp->crushed); }
uint64_t sampler_aug_count(const sampler *sp, int which) { return which >= 0 && which < AUGN_N ? atomic_load(&sp->augn[which]) : 0; }
void sampler_stats(const sampler *sp, uint64_t *produced, uint64_t *rejected) {
    *produced = atomic_load(&sp->produced);
    *rejected = atomic_load(&sp->rejected);
}
