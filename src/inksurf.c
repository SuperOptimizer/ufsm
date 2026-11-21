/* The surface-aligned ink head's host side (src/inksurf.h; DESIGN 2026-11-20 "inksurf"): the resample's inverse lists (the deterministic
   adjoint: a per-voxel gather instead of atomics) and the small fp32 2-D U-Net on the flattened feature patch. */
#include "inksurf.h"
#include "nn.h"
#include "unet.h"
#include "inksplat.h"
#include <unistd.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- inverse lists: every (sample, corner) with its weight as the forward computes it, sorted by voxel (stable: the sample order kept) ---- */
typedef struct { int v, s; float w; } sl_e;
int isf_lists_build(const float *pn, int n, int K, float step, int D, int H, int W, isf_lists *L) {
    memset(L, 0, sizeof *L);
    const size_t m = (size_t)n * K * 8; sl_e *e = malloc(m * sizeof *e), *t = malloc(m * sizeof *t); size_t ne = 0;
    if (!e || !t) { free(e); free(t); return -1; }
    for (int j = 0; j < K; j++) for (int p = 0; p < n; p++) {   /* (j n + p ascending: the order a voxel's entries keep) */
        const float tt = (j - 0.5f * (K - 1)) * step; float z, y, x; isf_pos(pn + 6 * (size_t)p, tt, &z, &y, &x);
        int iz, iy, ix; float wz, wy, wx; isf_cell(z, &iz, &wz); isf_cell(y, &iy, &wy); isf_cell(x, &ix, &wx);
        for (int k = 0; k < 8; k++) {
            const int cz = iz + (k >> 2 & 1), cy = iy + (k >> 1 & 1), cx = ix + (k & 1);
            if (cz < 0 || cy < 0 || cx < 0 || cz >= D || cy >= H || cx >= W) continue;
            const float w = (k >> 2 & 1 ? wz : 1.f - wz) * (k >> 1 & 1 ? wy : 1.f - wy) * (k & 1 ? wx : 1.f - wx);
            if (w == 0.f) continue;
            e[ne++] = (sl_e){(int)(((size_t)cz * H + cy) * W + cx), j * n + p, w};
        }
    }
    /* stable LSD radix sort on the voxel index (3 x 11 bits covers 2^33 > D H W) */
    size_t *cnt = malloc(2049 * sizeof *cnt); if (!cnt) { free(e); free(t); return -1; }
    for (int sh = 0; sh < 33; sh += 11) {
        memset(cnt, 0, 2049 * sizeof *cnt);
        for (size_t i = 0; i < ne; i++) cnt[((unsigned)e[i].v >> sh & 2047) + 1]++;
        for (int b = 0; b < 2048; b++) cnt[b + 1] += cnt[b];
        for (size_t i = 0; i < ne; i++) t[cnt[(unsigned)e[i].v >> sh & 2047]++] = e[i];
        sl_e *q = e; e = t; t = q;
    }
    free(cnt); free(t);
    int nv = 0; for (size_t i = 0; i < ne; i++) nv += !i || e[i].v != e[i - 1].v;
    L->vox = malloc((size_t)(nv ? nv : 1) * sizeof(int)); L->off = malloc((size_t)(nv + 1) * sizeof(int)); L->ent = malloc((ne ? ne : 1) * sizeof(int)); L->ew = malloc((ne ? ne : 1) * sizeof(float));
    if (!L->vox || !L->off || !L->ent || !L->ew) { free(e); isf_lists_free(L); return -1; }
    int u = -1; for (size_t i = 0; i < ne; i++) { if (!i || e[i].v != e[i - 1].v) { u++; L->vox[u] = e[i].v; L->off[u] = (int)i; } L->ent[i] = e[i].s; L->ew[i] = e[i].w; }
    L->off[nv] = (int)ne; L->nv = nv; L->ne = (int)ne;
    free(e); return 0;
}
void isf_lists_free(isf_lists *L) { free(L->vox); free(L->off); free(L->ent); free(L->ew); memset(L, 0, sizeof *L); }

/* ---- the 2-D head ---- */
/* (2026-11-21 S3) depth L (pools; 4 = the S1 / S2 head, its parameter layout unchanged): convs 0 stem 1x1, 1 (level 0), 2 .. L+1 the encoder
   at levels 1 .. L (pool, 3x3), B = L+2 the bottleneck (level L), B+1 .. B+L the decoder at levels L-1 .. 0 ([up(previous) ; skip] -> 3x3),
   O = 2L+3 the 1x1 output, R = 2L+4 the linear readout (1x1 from the patch straight to the logit, added: a direct gradient into the trunk).
   Channels e = 32 48 64 96 96 (128 at level 5, 6); the decoder at level l outputs e[l-1] (32 at level 0). */
enum { SH_MAXL = 6, SH_MAXC = 2 * SH_MAXL + 5 };
enum { R_STEM, R_C1, R_ENC, R_BOT, R_DEC, R_OUT, R_LIN };
typedef struct { int L, nc, co[SH_MAXC], ci[SH_MAXC], k[SH_MAXC], lv[SH_MAXC], role[SH_MAXC]; size_t woff[SH_MAXC], boff[SH_MAXC], np; } sh_lay;
static int sh_e(int l) { static const int e[SH_MAXL + 1] = {32, 48, 64, 96, 96, 128, 128}; return e[l]; }
static int sh_enc(int l) { return l ? l + 1 : 1; }   /* the conv whose output is the level-l skip */
static void sh_layout(sh_lay *y, int L, int cin) {
    memset(y, 0, sizeof *y); y->L = L; y->nc = 2 * L + 5; int i = 0;
    y->co[i] = 32; y->ci[i] = cin; y->k[i] = 1; y->lv[i] = 0; y->role[i++] = R_STEM;
    y->co[i] = 32; y->ci[i] = 32; y->k[i] = 3; y->lv[i] = 0; y->role[i++] = R_C1;
    for (int l = 1; l <= L; l++) { y->co[i] = sh_e(l); y->ci[i] = sh_e(l - 1); y->k[i] = 3; y->lv[i] = l; y->role[i++] = R_ENC; }
    y->co[i] = sh_e(L); y->ci[i] = sh_e(L); y->k[i] = 3; y->lv[i] = L; y->role[i++] = R_BOT;
    for (int l = L - 1; l >= 0; l--) { y->ci[i] = y->co[i - 1] + sh_e(l); y->co[i] = l ? sh_e(l - 1) : 32; y->k[i] = 3; y->lv[i] = l; y->role[i++] = R_DEC; }
    y->co[i] = 1; y->ci[i] = 32; y->k[i] = 1; y->lv[i] = 0; y->role[i++] = R_OUT;
    y->co[i] = 1; y->ci[i] = cin; y->k[i] = 1; y->lv[i] = 0; y->role[i++] = R_LIN;
    for (int c = 0; c < y->nc; c++) { y->woff[c] = y->np; y->np += (size_t)y->co[c] * y->ci[c] * y->k[c] * y->k[c]; y->boff[c] = y->np; y->np += (size_t)y->co[c]; }
}
struct isfhead {
    int cin, H, W; sh_lay y;
    float *p, *g, *m, *v;                        /* parameters, gradients, Adam moments */
    float *z[SH_MAXC], *a[SH_MAXC], *in[SH_MAXC]; /* pre-activations, SiLU outputs, conv inputs (pools / concatenations; in[0] = x, borrowed) */
    float *gz, *ga[SH_MAXC], *gin, *gx;          /* gradients: one gz / gin scratch of the largest size, per-node ga, the input's gx */
    size_t bytes;
};
static int hw(const isfhead *h, int lv) { return (h->H >> lv) * (h->W >> lv); }
static float *dalloc(isfhead *h, size_t n) { float *q = nn_malloc(n * 4); h->bytes += n * 4; return q; }
static uint64_t sm64(uint64_t *s) { uint64_t z = (*s += 0x9e3779b97f4a7c15ull); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; return z ^ (z >> 31); }
static double gauss(uint64_t *s) { const double u1 = ((sm64(s) >> 11) + 0.5) / 9007199254740992.0, u2 = ((sm64(s) >> 11) + 0.5) / 9007199254740992.0; return sqrt(-2 * log(u1)) * cos(6.283185307179586 * u2); }
/* the convs whose input is another node's activation directly (not an own buffer): 1 (a[0]), B (a[L+1]), O (a[last decoder]), R (x) */
static int sh_borrow(const sh_lay *y, int i) { return i == 1 || y->role[i] == R_BOT || y->role[i] == R_OUT || y->role[i] == R_LIN; }
isfhead *isfhead_create_d(int cin, int Hp, int Wp, int L, uint64_t seed) {
    if (L < 1 || L > SH_MAXL || cin < 1 || Hp < 16 || Wp < 16 || Hp % (1 << L) || Wp % (1 << L) || Hp % 16 || Wp % 16) {
        fprintf(stderr, "isfhead: patch %d x %d must be a multiple of 16 and of 2^depth (depth %d, 1 .. %d)\n", Hp, Wp, L, SH_MAXL); return nullptr; }
    isfhead *h = calloc(1, sizeof *h); h->cin = cin; h->H = Hp; h->W = Wp; sh_layout(&h->y, L, cin); const sh_lay *y = &h->y;
    float *hp = calloc(y->np, 4); uint64_t s = seed * 0x2545f4914f6cdd1dull + 1;
    for (int i = 0; i < y->nc; i++) {   /* He normal for the SiLU convs; the output convs small, bias -1 (ink is the minority) */
        const size_t nw = (size_t)y->co[i] * y->ci[i] * y->k[i] * y->k[i]; const int lin = y->role[i] == R_OUT || y->role[i] == R_LIN;
        const double sd = lin ? 0.01 : sqrt(2.0 / (y->ci[i] * y->k[i] * y->k[i]));
        for (size_t q = 0; q < nw; q++) hp[y->woff[i] + q] = (float)(sd * gauss(&s));
        if (y->role[i] == R_OUT) hp[y->boff[i]] = -1.f;
    }
    h->p = dalloc(h, y->np); h->g = dalloc(h, y->np); h->m = dalloc(h, y->np); h->v = dalloc(h, y->np);
    if (!h->p || !h->g || !h->m || !h->v) { free(hp); isfhead_free(h); return nullptr; }
    nn_h2d(h->p, hp, y->np * 4); free(hp); nn_zero(h->m, y->np * 4); nn_zero(h->v, y->np * 4); nn_zero(h->g, y->np * 4);
    size_t mx = 0; const int O = y->nc - 2;
    for (int i = 0; i < y->nc; i++) {
        const size_t o = (size_t)y->co[i] * hw(h, y->lv[i]), in = (size_t)y->ci[i] * hw(h, y->lv[i]);
        h->z[i] = dalloc(h, o); h->a[i] = i >= O ? nullptr : dalloc(h, o); h->ga[i] = dalloc(h, o);
        if (i && !sh_borrow(y, i)) h->in[i] = dalloc(h, in);
        if (o > mx) mx = o; if (in > mx) mx = in;
    }
    h->in[1] = h->a[0]; h->in[L + 2] = h->a[L + 1]; h->in[O] = h->a[O - 1];
    h->gz = dalloc(h, mx); h->gin = dalloc(h, mx); h->gx = dalloc(h, (size_t)cin * hw(h, 0));
    for (int i = 0; i < y->nc; i++) if (!h->z[i] || !h->ga[i] || (i < O && !h->a[i]) || (i && i != O + 1 && !h->in[i])) { isfhead_free(h); return nullptr; }
    if (!h->gz || !h->gin || !h->gx) { isfhead_free(h); return nullptr; }
    return h;
}
isfhead *isfhead_create(int cin, int Hp, int Wp, uint64_t seed) { return isfhead_create_d(cin, Hp, Wp, 4, seed); }
/* the device bytes of a head (parameters, gradients, moments, activations: isfhead_create_d's allocations) without allocating */
static size_t isfhead_est(int cin, int Hp, int Wp, int L) {
    sh_lay y; sh_layout(&y, L, cin); size_t b = 4 * y.np * 4, mx = 0; const int O = y.nc - 2;
    for (int i = 0; i < y.nc; i++) { const size_t n = (size_t)(Hp >> y.lv[i]) * (Wp >> y.lv[i]), o = (size_t)y.co[i] * n, in = (size_t)y.ci[i] * n;
        b += (o * (i >= O ? 2 : 3) + (i && !sh_borrow(&y, i) ? in : 0)) * 4; if (o > mx) mx = o; if (in > mx) mx = in; }
    return b + (2 * mx + (size_t)cin * Hp * Wp) * 4;
}
/* train --ink-surf's device footprint, for the memory planner: the head, the patch + its labels, a 32-plane feature slab of the D x H x W
   window, the compact touched-voxel features + gradients (bounded by 2 voxels a sample: measured ~1.3 on S1 patches) and their lists */
size_t isf_reserve_bytes(int C, int Hp, int K, int depth, int D, int H, int W) {
    const size_t n = (size_t)Hp * Hp, nvx = n * K * 2 < (size_t)D * H * W ? n * K * 2 : (size_t)D * H * W;
    return isfhead_est(C * K, Hp, Hp, depth) + ((size_t)C * K + 4) * n * 4 + (size_t)C * 32 * H * W * 2 + nvx * C * 4 * 2 + nvx * 4 * 2 + n * K * 8 * 8 * 2;
}
void isfhead_free(isfhead *h) {
    if (!h) return;
    nn_free(h->p); nn_free(h->g); nn_free(h->m); nn_free(h->v); nn_free(h->gz); nn_free(h->gin); nn_free(h->gx);
    for (int i = 0; i < h->y.nc; i++) { nn_free(h->z[i]); nn_free(h->a[i]); nn_free(h->ga[i]); if (i && !sh_borrow(&h->y, i)) nn_free(h->in[i]); }
    free(h);
}
size_t isfhead_nparams(const isfhead *h) { return h->y.np; }
int isfhead_depth(const isfhead *h) { return h->y.L; }
int isfhead_nconv(const isfhead *h) { return h->y.nc; }
size_t isfhead_conv_span(const isfhead *h, int i, size_t *n) { *n = h->y.boff[i] + (size_t)h->y.co[i] - h->y.woff[i]; return h->y.woff[i]; }
float *isfhead_params(isfhead *h) { return h->p; }
float *isfhead_grads(isfhead *h) { return h->g; }
float *isfhead_moments(isfhead *h, int k) { return k ? h->v : h->m; }
void isfhead_zero_grad(isfhead *h) { nn_zero(h->g, h->y.np * 4); }
size_t isfhead_bytes(const isfhead *h) { return h->bytes; }
static void conv(isfhead *h, int i) {
    const sh_lay *y = &h->y; const int lv = y->lv[i];
    nn_isf2_conv(h->in[i], y->ci[i], h->H >> lv, h->W >> lv, h->p + y->woff[i], h->p + y->boff[i], y->co[i], y->k[i], h->z[i]);
    if (h->a[i]) nn_isf2_silu(h->z[i], (size_t)y->co[i] * hw(h, lv), h->a[i]);
}
static void cat(isfhead *h, int i, const float *lo, int clo, const float *skip, int cs) {   /* in[i] = [up(lo) ; skip] at conv i's level */
    const int lv = h->y.lv[i], H = h->H >> lv, W = h->W >> lv;
    nn_isf2_up(lo, clo, H / 2, W / 2, h->in[i]);
    nn_d2d(h->in[i] + (size_t)clo * H * W, skip, (size_t)cs * H * W * 4);
}
const float *isfhead_forward(isfhead *h, const float *x) {
    const sh_lay *y = &h->y; const int L = y->L, B = L + 2, O = y->nc - 2, R = y->nc - 1;
    h->in[0] = (float *)x;
    conv(h, 0); conv(h, 1);
    for (int i = 2; i <= L + 1; i++) { const int lv = y->lv[i - 1]; nn_isf2_pool(h->a[i - 1], y->co[i - 1], h->H >> lv, h->W >> lv, h->in[i]); conv(h, i); }   /* (the encoder reads pool(a[i-1])) */
    conv(h, B);
    for (int d = B + 1; d < O; d++) { const int l = y->lv[d]; cat(h, d, h->a[d - 1], y->co[d - 1], h->a[sh_enc(l)], sh_e(l)); conv(h, d); }
    conv(h, O);
    h->in[R] = (float *)x; conv(h, R); nn_isf2_add(h->z[O], h->z[R], hw(h, 0));
    return h->z[O];
}
/* conv i backward from ga[i] (or the logit gradient): parameter gradients, d loss / d in[i] into dst (acc 0) */
static void conv_bwd_a(isfhead *h, int i, const float *gout, float *dst, int acc) {
    const sh_lay *y = &h->y; const int lv = y->lv[i], H = h->H >> lv, W = h->W >> lv; const size_t o = (size_t)y->co[i] * H * W;
    const float *gz = gout;
    if (h->a[i]) { nn_isf2_silu_bwd(h->z[i], gout, o, h->gz); gz = h->gz; }
    nn_isf2_conv_bw(h->in[i], y->ci[i], H, W, gz, y->co[i], y->k[i], h->g + y->woff[i], h->g + y->boff[i], 1);
    nn_isf2_conv_bd(gz, y->co[i], H, W, h->p + y->woff[i], y->ci[i], y->k[i], dst, acc);
}
static void conv_bwd(isfhead *h, int i, const float *gout, float *dst) { conv_bwd_a(h, i, gout, dst, 0); }
static void uncat(isfhead *h, int i, int lo, int clo, int skip, int cs) {   /* gin = d in[i]: ga[lo] = up_bwd(first part), ga[skip] = the rest */
    const int lv = h->y.lv[i], H = h->H >> lv, W = h->W >> lv;
    nn_isf2_up_bwd(h->gin, clo, H / 2, W / 2, h->ga[lo]);
    nn_d2d(h->ga[skip], h->gin + (size_t)clo * H * W, (size_t)cs * H * W * 4);
}
const float *isfhead_backward(isfhead *h, const float *glogit) {
    const sh_lay *y = &h->y; const int L = y->L, B = L + 2, O = y->nc - 2, R = y->nc - 1;
    conv_bwd(h, O, glogit, h->ga[O - 1]);
    for (int d = O - 1; d > B; d--) { const int l = y->lv[d]; conv_bwd(h, d, h->ga[d], h->gin); uncat(h, d, d - 1, y->co[d - 1], sh_enc(l), sh_e(l)); }
    conv_bwd(h, B, h->ga[B], h->ga[L + 1]);
    for (int i = L + 1; i >= 2; i--) {   /* in[i] = pool(a[i - 1]): ga[i - 1] += pool_bwd */
        conv_bwd(h, i, h->ga[i], h->gin); const int lv = y->lv[i - 1];
        nn_isf2_pool_bwd(h->gin, y->co[i - 1], h->H >> lv, h->W >> lv, h->ga[i - 1], 1);
    }
    conv_bwd(h, 1, h->ga[1], h->ga[0]);
    conv_bwd(h, 0, h->ga[0], h->gx);
    conv_bwd_a(h, R, glogit, h->gx, 1);
    return h->gx;
}
void isfhead_adamw(isfhead *h, float lr, float wd, int step) { nn_isf2_adamw(h->p, h->g, h->m, h->v, h->y.np, lr, 0.9f, 0.999f, 1e-8f, wd, step); }

/* ---- train --ink-surf: the window's patch from its MIL rays, the device side of a window (DESIGN 2026-11-20 "inksurf") ---- */
/* the forward table: per sample s its <= 8 (compact voxel, weight) pairs (from the inverse lists; slots in voxel order) */
static int lists_fwd(const isf_lists *L, size_t ns, int **fidx, float **fw, size_t *cap) {
    if (*cap < ns) { free(*fidx); free(*fw); *fidx = malloc(ns * 8 * sizeof(int)); *fw = malloc(ns * 8 * sizeof(float)); *cap = ns; if (!*fidx || !*fw) { *cap = 0; return -1; } }
    for (size_t i = 0; i < ns * 8; i++) (*fidx)[i] = -1;
    for (int u = 0; u < L->nv; u++) for (int e = L->off[u]; e < L->off[u + 1]; e++) {
        const size_t s = (size_t)L->ent[e]; int k = 0; while (k < 8 && (*fidx)[s * 8 + k] >= 0) k++;
        if (k == 8) return -1;
        (*fidx)[s * 8 + k] = u; (*fw)[s * 8 + k] = L->ew[e];
    }
    return 0;
}
int isf_win_build(isf_win *w, const inkray *r, int nr, int Hp, int step, int K, float kstep, int D, int H, int W, unsigned aug, int soft, float pvox) {   /* soft: y / 255 (teacher rays), else y != 0 */
    const size_t n = (size_t)Hp * Hp;
    if (w->cap < n) { free(w->pn); free(w->lab); free(w->wt); w->pn = malloc(n * 6 * 4); w->lab = malloc(n * 4); w->wt = malloc(n * 4); w->cap = n; if (!w->pn || !w->lab || !w->wt) { w->cap = 0; return -1; } }
    w->Hp = Hp; w->K = K; w->kstep = kstep; w->np = 0; w->frac = 0; w->unit = -1; w->nouv = 0; w->step = step; w->dvox = 0;
    isf_lists_free(&w->L);
    /* the centre: the on-grid ray nearest the window's centre (one pass of one segment: a unit's other wraps through the window lie far away on
       its label grid, outside the patch) */
    int best = -1; double bd = 1e300; const double cz = 0.5 * (D - 1), cy = 0.5 * (H - 1), cx = 0.5 * (W - 1);
    for (int k = 0; k < nr; k++) { if (r[k].lr < 0) { w->nouv++; continue; } if (!(pvox > 0) && (r[k].lr % step || r[k].lc % step)) continue;
        const double dd = (r[k].p[0] - cz) * (r[k].p[0] - cz) + (r[k].p[1] - cy) * (r[k].p[1] - cy) + (r[k].p[2] - cx) * (r[k].p[2] - cx); if (dd < bd) { bd = dd; best = k; } }
    for (size_t p = 0; p < n; p++) { float *q = w->pn + 6 * p; q[0] = q[1] = q[2] = -1e6f; q[3] = 1; q[4] = q[5] = 0; w->lab[p] = 0; w->wt[p] = 0; }
    if (best < 0) return 0;
    if (pvox > 0) {   /* (2026-11-21) the physical patch: the rays' grid g (label px) and spacing d (window voxels per label px) next to the centre ray, the
                         step the multiple of g nearest pvox / d (S1 / S2 took every 2nd 2.4 um px: 0.5 window voxel, predict sampled 1) */
        int g = 0; double d = 0;
        for (int pass = 0; pass < 2 && !g; pass++) for (int k = 0; k < nr; k++) {
            if (r[k].unit != r[best].unit || r[k].lr < 0 || k == best) continue;
            const int dl = pass ? r[k].lr - r[best].lr : r[k].lc - r[best].lc, same = pass ? r[k].lc == r[best].lc : r[k].lr == r[best].lr;
            if (!same || !dl || (g && abs(dl) >= g)) continue;
            double s2 = 0; for (int q = 0; q < 3; q++) s2 += ((double)r[k].p[q] - r[best].p[q]) * ((double)r[k].p[q] - r[best].p[q]);
            g = abs(dl); d = sqrt(s2) / g; }
        if (!g || !(d > 1e-6)) return 0;
        const int m = (int)lround(pvox / d / g); step = (m > 1 ? m : 1) * g; w->step = step; w->dvox = (float)d;
    }
    const int gr = r[best].lr - Hp / 2 * step, gc = r[best].lc - Hp / 2 * step; best = r[best].unit;
    const float ext = 0.5f * (K - 1) * kstep;
    for (int k = 0; k < nr; k++) {
        const inkray *q = &r[k]; if (q->unit != best || q->lr < 0 || (q->lr - gr) % step || (q->lc - gc) % step) continue;
        int i = (q->lr - gr) / step, j = (q->lc - gc) / step; if (i < 0 || j < 0 || i >= Hp || j >= Hp) continue;
        if (aug & 1) { const int t = i; i = j; j = t; } if (aug & 2) i = Hp - 1 - i; if (aug & 4) j = Hp - 1 - j;   /* (the patch's dihedral draw: the head sees any orientation) */
        const double nl = sqrt((double)q->n[0] * q->n[0] + (double)q->n[1] * q->n[1] + (double)q->n[2] * q->n[2]) * (aug & 8 ? -1 : 1); if (!(fabs(nl) > 1e-6)) continue;   /* (8: the normal reversed) */
        float N[3] = {(float)(q->n[0] / nl), (float)(q->n[1] / nl), (float)(q->n[2] / nl)};
        const float lim[3] = {(float)(D - 1), (float)(H - 1), (float)(W - 1)}; int in = 1;   /* both ends of the column inside the window: else masked */
        for (int d = 0; d < 3 && in; d++) { const float a = q->p[d] - ext * N[d], b = q->p[d] + ext * N[d]; in = a >= 0 && b >= 0 && a <= lim[d] && b <= lim[d]; }
        if (!in) continue;
        float *o = w->pn + 6 * ((size_t)i * Hp + j); for (int d = 0; d < 3; d++) { o[d] = q->p[d]; o[3 + d] = N[d]; }
        w->lab[(size_t)i * Hp + j] = soft ? q->y / 255.f : q->y ? 1.f : 0.f; w->wt[(size_t)i * Hp + j] = 1.f; w->np++;
    }
    w->unit = best; w->frac = (double)w->np / n;
    return isf_win_lists(w, D, H, W);
}
int isf_win_lists(isf_win *w, int D, int H, int W) {
    const size_t n = (size_t)w->Hp * w->Hp; isf_lists_free(&w->L);
    if (!w->np) return 0;
    if (isf_lists_build(w->pn, (int)n, w->K, w->kstep, D, H, W, &w->L)) return -1;
    return lists_fwd(&w->L, n * w->K, &w->fidx, &w->fw, &w->fcap);
}
int isf_win_alloc(isf_win *w, int Hp, int K, float kstep) {
    const size_t n = (size_t)Hp * Hp;
    if (w->cap < n) { free(w->pn); free(w->lab); free(w->wt); w->pn = malloc(n * 6 * 4); w->lab = malloc(n * 4); w->wt = malloc(n * 4); w->cap = n; if (!w->pn || !w->lab || !w->wt) { w->cap = 0; return -1; } }
    w->Hp = Hp; w->K = K; w->kstep = kstep; w->np = 0; return 0;
}
void isf_win_free(isf_win *w) { free(w->pn); free(w->lab); free(w->wt); free(w->fidx); free(w->fw); isf_lists_free(&w->L); memset(w, 0, sizeof *w); }

static int grow(void **p, size_t *cap, size_t need) { if (*cap >= need) return 0; nn_free(*p); *p = nn_malloc(need); *cap = *p ? need : 0; return *p ? 0 : -1; }
int isf_dev_init(isf_dev *g, int C, int Hp, int K, uint64_t seed) { return isf_dev_init_d(g, C, Hp, K, 4, seed); }
int isf_dev_init_d(isf_dev *g, int C, int Hp, int K, int depth, uint64_t seed) {
    memset(g, 0, sizeof *g); g->C = C; g->Hp = Hp; g->K = K;
    g->h = isfhead_create_d(C * K, Hp, Hp, depth, seed); if (!g->h) return -1;
    const size_t n = (size_t)Hp * Hp;
    g->patch = nn_malloc((size_t)C * K * n * 4); g->lab = nn_malloc(n * 4); g->wt = nn_malloc(n * 4); g->glog = nn_malloc(n * 4); g->zh = malloc(n * 4);
    return g->patch && g->lab && g->wt && g->glog && g->zh ? 0 : -1;
}
void isf_dev_free(isf_dev *g) {
    isfhead_free(g->h); nn_free(g->patch); nn_free(g->lab); nn_free(g->wt); nn_free(g->glog); free(g->zh);
    nn_free(g->vox); nn_free(g->off); nn_free(g->ent); nn_free(g->ew); nn_free(g->fidx); nn_free(g->fw); nn_free(g->fc); nn_free(g->gc); nn_free(g->feat); nn_free(g->rows);
    memset(g, 0, sizeof *g);
}
size_t isf_dev_bytes(const isf_dev *g) { return isfhead_bytes(g->h) + ((size_t)g->C * g->K + 4) * g->Hp * g->Hp * 4 + g->c_vox + g->c_off + g->c_ent + g->c_ew + g->c_fidx + g->c_fw + g->c_fc + g->c_gc + g->c_feat + g->c_rows; }
int isf_dev_upload(isf_dev *g, const isf_win *w) {
    g->nv = w->np ? w->L.nv : 0; if (!g->nv) return 0;
    const size_t n = (size_t)g->Hp * g->Hp, ne = (size_t)w->L.ne, nv = (size_t)g->nv, nsk = n * g->K;
    if (grow((void **)&g->vox, &g->c_vox, nv * 4) || grow((void **)&g->off, &g->c_off, (nv + 1) * 4) || grow((void **)&g->ent, &g->c_ent, ne * 4) || grow((void **)&g->ew, &g->c_ew, ne * 4) ||
        grow((void **)&g->fidx, &g->c_fidx, nsk * 8 * 4) || grow((void **)&g->fw, &g->c_fw, nsk * 8 * 4) || grow((void **)&g->fc, &g->c_fc, (size_t)g->C * nv * 4) || grow((void **)&g->gc, &g->c_gc, (size_t)g->C * nv * 4)) return -1;
    nn_h2d(g->vox, w->L.vox, nv * 4); nn_h2d(g->off, w->L.off, (nv + 1) * 4); nn_h2d(g->ent, w->L.ent, ne * 4); nn_h2d(g->ew, w->L.ew, ne * 4);
    nn_h2d(g->fidx, w->fidx, nsk * 8 * 4); nn_h2d(g->fw, w->fw, nsk * 8 * 4); nn_h2d(g->lab, w->lab, n * 4); nn_h2d(g->wt, w->wt, n * 4);
    return 0;
}
static void slab_range(const isf_win *w, size_t vb, size_t ns, int *ua, int *ub) {   /* the touched voxels in [vb, vb + ns): vox is ascending */
    int lo = 0, hi = w->L.nv; while (lo < hi) { const int m = (lo + hi) / 2; if ((size_t)w->L.vox[m] < vb) lo = m + 1; else hi = m; } *ua = lo;
    hi = w->L.nv; while (lo < hi) { const int m = (lo + hi) / 2; if ((size_t)w->L.vox[m] < vb + ns) lo = m + 1; else hi = m; } *ub = lo;
}
int isf_dev_take(isf_dev *g, unet *u, const isf_win *w, int z0, int z1, size_t hw) {
    if (!g->nv) return 0;
    const size_t ns = (size_t)(z1 - z0) * hw, vb = (size_t)z0 * hw; int ua, ub; slab_range(w, vb, ns, &ua, &ub);
    if (ub <= ua) return 0;
    if (grow(&g->feat, &g->c_feat, (size_t)g->C * ns * 2)) return -1;
    if (unet_head_feat_z(u, z0, z1, g->feat)) return -1;
    nn_isf_take(g->feat, g->C, ns, vb, g->vox, ua, ub, g->fc, g->nv);
    return 0;
}
/* o[m] (value, label) sorted by value, ties in their input order: what the stable insertion sort gave (no NaN values) */
static void auc_sort(float (*o)[2], size_t m) {
    float (*t)[2] = malloc((m ? m : 1) * sizeof *t); if (!t) { for (size_t i = 1; i < m; i++) { float a0 = o[i][0], a1 = o[i][1]; size_t j = i;
        while (j > 0 && o[j - 1][0] > a0) { o[j][0] = o[j - 1][0]; o[j][1] = o[j - 1][1]; j--; } o[j][0] = a0; o[j][1] = a1; } return; }
    for (size_t w = 1; w < m; w *= 2) {   /* bottom-up merge sort: left run first on ties = stable */
        for (size_t lo = 0; lo < m; lo += 2 * w) { const size_t mid = lo + w < m ? lo + w : m, hi = lo + 2 * w < m ? lo + 2 * w : m; size_t i = lo, j = mid, k = lo;
            while (i < mid && j < hi) { if (o[j][0] < o[i][0]) { t[k][0] = o[j][0]; t[k][1] = o[j][1]; j++; } else { t[k][0] = o[i][0]; t[k][1] = o[i][1]; i++; } k++; }
            while (i < mid) { t[k][0] = o[i][0]; t[k][1] = o[i][1]; i++; k++; } while (j < hi) { t[k][0] = o[j][0]; t[k][1] = o[j][1]; j++; k++; } }
        memcpy(o, t, m * sizeof *t);
    }
    free(t);
}
double isf_dev_step(isf_dev *g, const isf_win *w, float lambda, int train, float gscale, const float *dscale, double *auc) {
    if (auc) *auc = -1;
    if (!g->nv) return -1;
    const size_t n = (size_t)g->Hp * g->Hp;
    nn_isf_resample_c(g->fc, g->C, g->nv, g->fidx, g->fw, (int)n, g->K, g->patch);
    const float *z = isfhead_forward(g->h, g->patch);
    nn_d2h(g->zh, z, n * 4);
    double l = 0, sw = 0; for (size_t p = 0; p < n; p++) { if (!(w->wt[p] > 0)) continue; const double v = g->zh[p], t = w->lab[p];
        l += w->wt[p] * (fmax(v, 0) - v * t + log1p(exp(-fabs(v)))); sw += w->wt[p]; }
    if (auc) {   /* (the window's patch AUC on its labelled pixels; ties by rank order) */
        double a = 0, np = 0, nn = 0; size_t m = 0; for (size_t p = 0; p < n; p++) m += w->wt[p] > 0;
        float (*o)[2] = malloc(m * sizeof *o); size_t k = 0; for (size_t p = 0; p < n; p++) if (w->wt[p] > 0) { o[k][0] = g->zh[p]; o[k][1] = w->lab[p]; k++; }
        int nan = 0; for (size_t i = 0; i < m; i++) nan |= isnan(o[i][0]);
        if (!nan) auc_sort(o, m);   /* (inkperf 11-21: the insertion sort's order, stable by value, in O(m log m): 15 ms -> ~1 ms at 12.5 k px) */
        else for (size_t i = 1; i < m; i++) { float t0 = o[i][0], t1 = o[i][1]; size_t j = i; while (j > 0 && o[j - 1][0] > t0) { o[j][0] = o[j - 1][0]; o[j][1] = o[j - 1][1]; j--; } o[j][0] = t0; o[j][1] = t1; }
        for (size_t i = 0; i < m; i++) { if (o[i][1] > 0.5f) { a += nn; np++; } else nn++; }
        free(o); *auc = np && nn ? a / (np * nn) : -1;
    }
    if (!(sw > 0)) return -1;
    if (train) {
        nn_isf2_bce_grad(z, g->lab, g->wt, n, (float)(lambda / sw), g->glog);
        const float *gp = isfhead_backward(g->h, g->glog);
        nn_isf_gather(gp, g->C, g->K, (int)n, nullptr, g->off, g->ent, g->ew, g->nv, g->gc, ISF_F32, (size_t)g->nv, gscale, dscale, 0);
    }
    return l / sw;
}
int isf_dev_put(isf_dev *g, unet *u, const isf_win *w, int z0, int z1, size_t hw) {
    unet_set_head_feat_grad_z(u, nullptr);
    if (!g->nv) return 0;
    const size_t ns = (size_t)(z1 - z0) * hw, vb = (size_t)z0 * hw; int ua, ub; slab_range(w, vb, ns, &ua, &ub);
    if (ub <= ua) return 0;
    if (grow(&g->rows, &g->c_rows, (size_t)g->C * ns * 2)) return -1;
    nn_zero(g->rows, (size_t)g->C * ns * 2);
    nn_isf_put(g->gc, g->C, g->nv, g->vox, ua, ub, vb, ns, g->rows);
    unet_set_head_feat_grad_z(u, g->rows);
    return 0;
}
/* the head's parameters + Adam moments + step: "ISF1", int cin, Hp, step, then np floats x 3 */
/* the .isf sidecar: "ISF1" {magic, cin, Hp, step} (depth 4: S1 / S2) or "ISF2" {magic, cin, Hp, step, depth}, then params, Adam m, v */
int isf_save(const isf_dev *g, const char *path, int step) {
    char tmp[4200]; snprintf(tmp, sizeof tmp, "%s.tmp", path); FILE *f = fopen(tmp, "wb"); if (!f) return -1;
    const size_t np = isfhead_nparams(g->h); float *h = malloc(np * 4); int ok = h != nullptr; const int L = isfhead_depth(g->h);
    const int hd[5] = {L == 4 ? 0x31465349 : 0x32465349, g->C * g->K, g->Hp, step, L}; ok = ok && fwrite(hd, 4, L == 4 ? 4 : 5, f) == (size_t)(L == 4 ? 4 : 5);
    const float *src[3] = {isfhead_params(g->h), isfhead_moments(g->h, 0), isfhead_moments(g->h, 1)};
    for (int k = 0; k < 3 && ok; k++) { nn_d2h(h, src[k], np * 4); ok = fwrite(h, 4, np, f) == np; }
    free(h); if (fclose(f) || !ok || rename(tmp, path)) { unlink(tmp); return -1; }
    return 0;
}
int isf_peek(const char *path, int *cin, int *depth, int *step) {
    FILE *f = fopen(path, "rb"); if (!f) return -1; int hd[5] = {0}; const size_t n = fread(hd, 4, 5, f); fclose(f);
    if (n < 4 || (hd[0] != 0x31465349 && (hd[0] != 0x32465349 || n < 5))) return -1;
    if (cin) *cin = hd[1]; if (depth) *depth = hd[0] == 0x31465349 ? 4 : hd[4]; if (step) *step = hd[3]; return 0;
}
/* a file of the head's depth: parameters + moments. Another depth (S3 from S2: --isf-depth 5 from a depth-4 head): the convs of the same role
   (stem, level-0 conv, encoder / decoder at level l, bottleneck, output, readout) and shape take the file's parameters, the rest stay fresh;
   moments zero (*step = 0). *mapped (optional): convs taken */
int isf_load_x(isf_dev *g, const char *path, int *step, int *mapped) {
    int cin, L0, st; if (isf_peek(path, &cin, &L0, &st) || cin != g->C * g->K) return -1;
    FILE *f = fopen(path, "rb"); if (!f) return -1; if (fseek(f, L0 == 4 ? 16 : 20, SEEK_SET)) { fclose(f); return -1; }
    const isfhead *H = g->h; const sh_lay *y = &H->y; sh_lay x; sh_layout(&x, L0, cin);
    float *h = malloc(x.np * 4); int ok = h != nullptr;
    if (L0 == y->L) {
        float *dst[3] = {isfhead_params(g->h), isfhead_moments(g->h, 0), isfhead_moments(g->h, 1)};
        for (int k = 0; k < 3 && ok; k++) { ok = fread(h, 4, x.np, f) == x.np; if (ok) nn_h2d(dst[k], h, x.np * 4); }
        if (mapped) *mapped = y->nc;
    } else {
        ok = ok && fread(h, 4, x.np, f) == x.np; int nm = 0;
        for (int i = 0; ok && i < y->nc; i++) for (int j = 0; j < x.nc; j++)
            if (x.role[j] == y->role[i] && x.lv[j] == y->lv[i] && x.co[j] == y->co[i] && x.ci[j] == y->ci[i] && x.k[j] == y->k[i]
                && (y->role[i] != R_BOT || x.L == y->L)) {   /* (a bottleneck only at the same depth) */
                nn_h2d(H->p + y->woff[i], h + x.woff[j], (x.boff[j] + (size_t)x.co[j] - x.woff[j]) * 4); nm++; break; }
        if (mapped) *mapped = nm; st = 0;
    }
    free(h); fclose(f); if (ok && step) *step = st;
    return ok ? 0 : -1;
}
int isf_load(isf_dev *g, const char *path, int *step) { return isf_load_x(g, path, step, nullptr); }
