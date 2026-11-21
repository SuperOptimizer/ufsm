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
enum { SH_NC = 13 };   /* (conv 12: the linear readout, 1x1 from the patch straight to the logit, added to the U-Net's: a direct gradient into the trunk) */
static const int sh_co[SH_NC] = {32, 32, 48, 64, 96, 96, 96, 64, 48, 32, 32, 1, 1}, sh_k[SH_NC] = {1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1}, sh_lv[SH_NC] = {0, 0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 0, 0};
struct isfhead {
    int cin, H, W, ci[SH_NC]; size_t woff[SH_NC], boff[SH_NC], np;
    float *p, *g, *m, *v;                      /* parameters, gradients, Adam moments */
    float *z[SH_NC], *a[SH_NC], *in[SH_NC];    /* pre-activations, SiLU outputs, conv inputs (pools / concatenations; in[0] = x, borrowed) */
    float *gz, *ga[SH_NC], *gin, *gx;          /* gradients: one gz / gin scratch of the largest size, per-node ga, the input's gx */
    size_t bytes;
};
static int hw(const isfhead *h, int lv) { return (h->H >> lv) * (h->W >> lv); }
static float *dalloc(isfhead *h, size_t n) { float *q = nn_malloc(n * 4); h->bytes += n * 4; return q; }
static uint64_t sm64(uint64_t *s) { uint64_t z = (*s += 0x9e3779b97f4a7c15ull); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; return z ^ (z >> 31); }
static double gauss(uint64_t *s) { const double u1 = ((sm64(s) >> 11) + 0.5) / 9007199254740992.0, u2 = ((sm64(s) >> 11) + 0.5) / 9007199254740992.0; return sqrt(-2 * log(u1)) * cos(6.283185307179586 * u2); }
isfhead *isfhead_create(int cin, int Hp, int Wp, uint64_t seed) {
    if (cin < 1 || Hp < 16 || Wp < 16 || Hp % 16 || Wp % 16) { fprintf(stderr, "isfhead: patch %d x %d must be a multiple of 16\n", Hp, Wp); return nullptr; }
    isfhead *h = calloc(1, sizeof *h); h->cin = cin; h->H = Hp; h->W = Wp;
    const int ci[SH_NC] = {cin, 32, 32, 48, 64, 96, 96, 96 + 96, 64 + 64, 48 + 48, 32 + 32, 32, cin};
    memcpy(h->ci, ci, sizeof ci);
    for (int i = 0; i < SH_NC; i++) { h->woff[i] = h->np; h->np += (size_t)sh_co[i] * ci[i] * sh_k[i] * sh_k[i]; h->boff[i] = h->np; h->np += (size_t)sh_co[i]; }
    float *hp = calloc(h->np, 4); uint64_t s = seed * 0x2545f4914f6cdd1dull + 1;
    for (int i = 0; i < SH_NC; i++) {   /* He normal for the SiLU convs; the output conv small, bias -1 (ink is the minority) */
        const size_t nw = (size_t)sh_co[i] * ci[i] * sh_k[i] * sh_k[i]; const double sd = i >= 11 ? 0.01 : sqrt(2.0 / (ci[i] * sh_k[i] * sh_k[i]));
        for (size_t q = 0; q < nw; q++) hp[h->woff[i] + q] = (float)(sd * gauss(&s));
        if (i == 11) hp[h->boff[i]] = -1.f;
    }
    h->p = dalloc(h, h->np); h->g = dalloc(h, h->np); h->m = dalloc(h, h->np); h->v = dalloc(h, h->np);
    if (!h->p || !h->g || !h->m || !h->v) { free(hp); isfhead_free(h); return nullptr; }
    nn_h2d(h->p, hp, h->np * 4); free(hp); nn_zero(h->m, h->np * 4); nn_zero(h->v, h->np * 4); nn_zero(h->g, h->np * 4);
    size_t mx = 0;
    for (int i = 0; i < SH_NC; i++) {
        const size_t o = (size_t)sh_co[i] * hw(h, sh_lv[i]), in = (size_t)ci[i] * hw(h, sh_lv[i]);
        h->z[i] = dalloc(h, o); h->a[i] = i >= 11 ? nullptr : dalloc(h, o); h->ga[i] = dalloc(h, o);
        if (i) h->in[i] = (i == 1 || i == 6 || i == 11 || i == 12) ? nullptr : dalloc(h, in);   /* (convs 1, 6, 11 read a[0], a[5], a[10] directly) */
        if (o > mx) mx = o; if (in > mx) mx = in;
    }
    h->in[1] = h->a[0]; h->in[6] = h->a[5]; h->in[11] = h->a[10];
    h->gz = dalloc(h, mx); h->gin = dalloc(h, mx); h->gx = dalloc(h, (size_t)cin * hw(h, 0));
    for (int i = 0; i < SH_NC; i++) if (!h->z[i] || !h->ga[i] || (i < 11 && !h->a[i]) || (i && i != 12 && !h->in[i])) { isfhead_free(h); return nullptr; }
    if (!h->gz || !h->gin || !h->gx) { isfhead_free(h); return nullptr; }
    return h;
}
void isfhead_free(isfhead *h) {
    if (!h) return;
    nn_free(h->p); nn_free(h->g); nn_free(h->m); nn_free(h->v); nn_free(h->gz); nn_free(h->gin); nn_free(h->gx);
    for (int i = 0; i < SH_NC; i++) { nn_free(h->z[i]); nn_free(h->a[i]); nn_free(h->ga[i]); if (i != 1 && i != 6 && i != 11 && i != 12 && i) nn_free(h->in[i]); }
    free(h);
}
size_t isfhead_nparams(const isfhead *h) { return h->np; }
int isfhead_nconv(void) { return SH_NC; }
size_t isfhead_conv_span(const isfhead *h, int i, size_t *n) { *n = h->boff[i] + (size_t)sh_co[i] - h->woff[i]; return h->woff[i]; }
float *isfhead_params(isfhead *h) { return h->p; }
float *isfhead_grads(isfhead *h) { return h->g; }
float *isfhead_moments(isfhead *h, int k) { return k ? h->v : h->m; }
void isfhead_zero_grad(isfhead *h) { nn_zero(h->g, h->np * 4); }
size_t isfhead_bytes(const isfhead *h) { return h->bytes; }
static void conv(isfhead *h, int i) {
    const int lv = sh_lv[i]; nn_isf2_conv(h->in[i], h->ci[i], h->H >> lv, h->W >> lv, h->p + h->woff[i], h->p + h->boff[i], sh_co[i], sh_k[i], h->z[i]);
    if (h->a[i]) nn_isf2_silu(h->z[i], (size_t)sh_co[i] * hw(h, lv), h->a[i]);
}
static void cat(isfhead *h, int i, const float *lo, int clo, const float *skip, int cs) {   /* in[i] = [up(lo) ; skip] at conv i's level */
    const int lv = sh_lv[i], H = h->H >> lv, W = h->W >> lv;
    nn_isf2_up(lo, clo, H / 2, W / 2, h->in[i]);
    nn_d2d(h->in[i] + (size_t)clo * H * W, skip, (size_t)cs * H * W * 4);
}
const float *isfhead_forward(isfhead *h, const float *x) {
    h->in[0] = (float *)x;
    conv(h, 0); conv(h, 1);
    for (int i = 2; i <= 5; i++) { const int lv = sh_lv[i - 1]; nn_isf2_pool(h->a[i - 1], sh_co[i - 1], h->H >> lv, h->W >> lv, h->in[i]); conv(h, i); }   /* (2..5 read pool(a[i-1])) */
    conv(h, 6);
    cat(h, 7, h->a[6], 96, h->a[4], 96); conv(h, 7);
    cat(h, 8, h->a[7], 64, h->a[3], 64); conv(h, 8);
    cat(h, 9, h->a[8], 48, h->a[2], 48); conv(h, 9);
    cat(h, 10, h->a[9], 32, h->a[1], 32); conv(h, 10);
    conv(h, 11);
    h->in[12] = (float *)x; conv(h, 12); nn_isf2_add(h->z[11], h->z[12], hw(h, 0));
    return h->z[11];
}
/* conv i backward from ga[i] (or the logit gradient): parameter gradients, d loss / d in[i] into dst (acc 0) */
static void conv_bwd_a(isfhead *h, int i, const float *gout, float *dst, int acc) {
    const int lv = sh_lv[i], H = h->H >> lv, W = h->W >> lv; const size_t o = (size_t)sh_co[i] * H * W;
    const float *gz = gout;
    if (h->a[i]) { nn_isf2_silu_bwd(h->z[i], gout, o, h->gz); gz = h->gz; }
    nn_isf2_conv_bw(h->in[i], h->ci[i], H, W, gz, sh_co[i], sh_k[i], h->g + h->woff[i], h->g + h->boff[i], 1);
    nn_isf2_conv_bd(gz, sh_co[i], H, W, h->p + h->woff[i], h->ci[i], sh_k[i], dst, acc);
}
static void conv_bwd(isfhead *h, int i, const float *gout, float *dst) { conv_bwd_a(h, i, gout, dst, 0); }
static void uncat(isfhead *h, int i, int lo, int clo, int skip, int cs) {   /* gin = d in[i]: ga[lo] = up_bwd(first part), ga[skip] = the rest */
    const int lv = sh_lv[i], H = h->H >> lv, W = h->W >> lv;
    nn_isf2_up_bwd(h->gin, clo, H / 2, W / 2, h->ga[lo]);
    nn_d2d(h->ga[skip], h->gin + (size_t)clo * H * W, (size_t)cs * H * W * 4);
}
const float *isfhead_backward(isfhead *h, const float *glogit) {
    conv_bwd(h, 11, glogit, h->ga[10]);
    conv_bwd(h, 10, h->ga[10], h->gin); uncat(h, 10, 9, 32, 1, 32);
    conv_bwd(h, 9, h->ga[9], h->gin); uncat(h, 9, 8, 48, 2, 48);
    conv_bwd(h, 8, h->ga[8], h->gin); uncat(h, 8, 7, 64, 3, 64);
    conv_bwd(h, 7, h->ga[7], h->gin); uncat(h, 7, 6, 96, 4, 96);
    conv_bwd(h, 6, h->ga[6], h->ga[5]);
    for (int i = 5; i >= 2; i--) {   /* in[i] = pool(a[i - 1]): ga[i - 1] += pool_bwd */
        conv_bwd(h, i, h->ga[i], h->gin); const int lv = sh_lv[i - 1];
        nn_isf2_pool_bwd(h->gin, sh_co[i - 1], h->H >> lv, h->W >> lv, h->ga[i - 1], 1);
    }
    conv_bwd(h, 1, h->ga[1], h->ga[0]);
    conv_bwd(h, 0, h->ga[0], h->gx);
    conv_bwd_a(h, 12, glogit, h->gx, 1);
    return h->gx;
}
void isfhead_adamw(isfhead *h, float lr, float wd, int step) { nn_isf2_adamw(h->p, h->g, h->m, h->v, h->np, lr, 0.9f, 0.999f, 1e-8f, wd, step); }

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
int isf_win_build(isf_win *w, const inkray *r, int nr, int Hp, int step, int K, float kstep, int D, int H, int W, unsigned aug) {
    const size_t n = (size_t)Hp * Hp;
    if (w->cap < n) { free(w->pn); free(w->lab); free(w->wt); w->pn = malloc(n * 6 * 4); w->lab = malloc(n * 4); w->wt = malloc(n * 4); w->cap = n; if (!w->pn || !w->lab || !w->wt) { w->cap = 0; return -1; } }
    w->Hp = Hp; w->K = K; w->kstep = kstep; w->np = 0; w->frac = 0; w->unit = -1; w->nouv = 0;
    isf_lists_free(&w->L);
    /* the centre: the on-grid ray nearest the window's centre (one pass of one segment: a unit's other wraps through the window lie far away on
       its label grid, outside the patch) */
    int best = -1; double bd = 1e300; const double cz = 0.5 * (D - 1), cy = 0.5 * (H - 1), cx = 0.5 * (W - 1);
    for (int k = 0; k < nr; k++) { if (r[k].lr < 0) { w->nouv++; continue; } if (r[k].lr % step || r[k].lc % step) continue;
        const double dd = (r[k].p[0] - cz) * (r[k].p[0] - cz) + (r[k].p[1] - cy) * (r[k].p[1] - cy) + (r[k].p[2] - cx) * (r[k].p[2] - cx); if (dd < bd) { bd = dd; best = k; } }
    for (size_t p = 0; p < n; p++) { float *q = w->pn + 6 * p; q[0] = q[1] = q[2] = -1e6f; q[3] = 1; q[4] = q[5] = 0; w->lab[p] = 0; w->wt[p] = 0; }
    if (best < 0) return 0;
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
        w->lab[(size_t)i * Hp + j] = q->y ? 1.f : 0.f; w->wt[(size_t)i * Hp + j] = 1.f; w->np++;
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
int isf_dev_init(isf_dev *g, int C, int Hp, int K, uint64_t seed) {
    memset(g, 0, sizeof *g); g->C = C; g->Hp = Hp; g->K = K;
    g->h = isfhead_create(C * K, Hp, Hp, seed); if (!g->h) return -1;
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
int isf_save(const isf_dev *g, const char *path, int step) {
    char tmp[4200]; snprintf(tmp, sizeof tmp, "%s.tmp", path); FILE *f = fopen(tmp, "wb"); if (!f) return -1;
    const size_t np = isfhead_nparams(g->h); float *h = malloc(np * 4); int ok = h != nullptr;
    const int hd[4] = {0x31465349, g->C * g->K, g->Hp, step}; ok = ok && fwrite(hd, 4, 4, f) == 4;
    const float *src[3] = {isfhead_params(g->h), isfhead_moments(g->h, 0), isfhead_moments(g->h, 1)};
    for (int k = 0; k < 3 && ok; k++) { nn_d2h(h, src[k], np * 4); ok = fwrite(h, 4, np, f) == np; }
    free(h); if (fclose(f) || !ok || rename(tmp, path)) { unlink(tmp); return -1; }
    return 0;
}
int isf_load(isf_dev *g, const char *path, int *step) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    int hd[4]; const size_t np = isfhead_nparams(g->h); float *h = malloc(np * 4); int ok = h && fread(hd, 4, 4, f) == 4 && hd[0] == 0x31465349 && hd[1] == g->C * g->K;   /* (the patch side may differ: the head is convolutional) */
    float *dst[3] = {isfhead_params(g->h), isfhead_moments(g->h, 0), isfhead_moments(g->h, 1)};
    for (int k = 0; k < 3 && ok; k++) { ok = fread(h, 4, np, f) == np; if (ok) nn_h2d(dst[k], h, np * 4); }
    free(h); fclose(f); if (ok && step) *step = hd[3];
    return ok ? 0 : -1;
}
