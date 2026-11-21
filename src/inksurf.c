/* The surface-aligned ink head's host side (src/inksurf.h; DESIGN 2026-11-20 "inksurf"): the resample's inverse lists (the deterministic
   adjoint: a per-voxel gather instead of atomics) and the small fp32 2-D U-Net on the flattened feature patch. */
#include "inksurf.h"
#include "nn.h"
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
    nn_isf2_conv_bw(h->in[i], h->ci[i], H, W, gz, sh_co[i], sh_k[i], h->g + h->woff[i], h->g + h->boff[i]);
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
