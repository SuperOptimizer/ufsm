/* vwt model (src/vwt.h): flat fp32 parameters / gradients on the device, the forward of one slab keeping every activation the backward
   needs, the backward (gradients accumulate), init, save / load.

   Network (after villa winding_model.py; channels-last [1][T][T][L][C], D = i, H = j, W = k ray):
     x [2] -> 3 x stage s (conv3 stride (2,2,1) -> LN + GELU -> conv3 -> LN + GELU; ch[s]) = skip s
     -> trunk_in 1x1 -> LN + GELU (trunk) -> stem x (conv3 -> LN + GELU -> conv3 -> LN, + residual)
     -> blocks x (at_block over the ray (nseq (T/8)^2, S L, RoPE on k) ; at_block over the transverse plane (nseq L, S (T/8)^2, RoPE on i, j))
     -> up2 + skip 1 -> conv3 (dec) LN GELU -> conv3 LN GELU -> up2 + skip 0 -> 1x1 (ch0) LN GELU -> 1x1 4 -> pixel shuffle, softplus, cumsum. */
#include "vwt.h"
#include "attn.h"
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>

typedef struct { size_t w, b, g, be; int ci, co, k, s; } pconv;   /* conv (+ LN affine g / be when g) */
#define MAXB 16
typedef struct {   /* stored activations of one conv + LN */
    vt_t in, c, a; float *mean, *rstd;
} cact;
struct vwt {
    vwt_cfg cfg; size_t np; float *P, *G;
    pconv enc[3][2], tin, stem[MAXB][2], dec[2], fh0, fh3;
    at_dims ad; at_bp bp[2 * MAXB];
    /* activations */
    vt_t x0; cact e[3][2], ti, st[MAXB][2]; vt_t stout[MAXB];   /* stem block outputs (= next input) */
    float *tokin[2 * MAXB]; float *tok, *tmp; vt_t tokv;           /* block inputs (ray: R order, transverse: T order); final tokens */
    vt_t up1, cat1; cact d0, d1; vt_t up0, cat0; cact h0; vt_t lg; float *inc, *ph;
    void *scratch; void *ws_mem; at_ws ws; at_geom gr, gt;
    int built; const uint16_t *xf; uint16_t *xb; int xC;   /* extra features (cfg.cx); the bf16 input [T][T][L][xC] (CT, valid, features, zero
                                                           channels to xC % 32 == 0) the input conv reads directly */
};

/* env UFSM_VWT_PROF=1: synchronising section timers (forward / backward: enc, trunk_in + stem, blocks, decoder + head) printed per call */
#include <time.h>
static int prof_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("UFSM_VWT_PROF"); v = e && atoi(e); } return v; }
static double prof_t(void) { nn_sync(); struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
#define PT(i) do { if (prof_on()) tp[i] = prof_t(); } while (0)

static uint64_t rng_next(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }
static double unif(uint64_t *s) { return (rng_next(s) >> 11) * (1.0 / 9007199254740992.0); }
static double gauss(uint64_t *s) { double a = unif(s) + 1e-300, b = unif(s); return sqrt(-2 * log(a)) * cos(6.283185307179586 * b); }

static size_t lay_conv(pconv *c, size_t off, int ci, int co, int k, int s, int ln) {
    c->ci = ci; c->co = co; c->k = k; c->s = s;
    c->w = off; off += (size_t)co * k * k * k * ci; c->b = off; off += co;
    if (ln) { c->g = off; off += co; c->be = off; off += co; } else c->g = c->be = 0;
    return off;
}
static void init_conv(float *h, const pconv *c, uint64_t *seed) {   /* PyTorch Conv3d default: U(+-1/sqrt(fan_in)) for w and b */
    const size_t n = (size_t)c->co * c->k * c->k * c->k * c->ci; const double bd = 1.0 / sqrt((double)c->k * c->k * c->k * c->ci);
    for (size_t i = 0; i < n; i++) h[c->w + i] = (float)((2 * unif(seed) - 1) * bd);
    for (int i = 0; i < c->co; i++) h[c->b + i] = (float)((2 * unif(seed) - 1) * bd);
    if (c->g) for (int i = 0; i < c->co; i++) { h[c->g + i] = 1.f; h[c->be + i] = 0.f; }
}

vwt *vwt_create(const vwt_cfg *cfg, unsigned seed) {
    vwt *m = calloc(1, sizeof *m); m->cfg = *cfg;
    const vwt_cfg *c = cfg; if (c->stem > MAXB || c->blocks > MAXB) { free(m); return nullptr; }
    size_t off = 0; int prev = 2 + (c->cx > 0 ? c->cx : 0);
    for (int s = 0; s < 3; s++) { off = lay_conv(&m->enc[s][0], off, prev, c->ch[s], 3, 2, 1); off = lay_conv(&m->enc[s][1], off, c->ch[s], c->ch[s], 3, 1, 1); prev = c->ch[s]; }
    off = lay_conv(&m->tin, off, c->ch[2], c->trunk, 1, 1, 1);
    for (int k = 0; k < c->stem; k++) { off = lay_conv(&m->stem[k][0], off, c->trunk, c->trunk, 3, 1, 1); off = lay_conv(&m->stem[k][1], off, c->trunk, c->trunk, 3, 1, 1); }
    m->ad = (at_dims){c->trunk, c->heads, c->mlp, 1, 0, 100.f, 0, 0, {0}, 0, 0};
    for (int k = 0; k < 2 * c->blocks; k++) off = at_block_layout(&m->ad, &m->bp[k], off);
    off = lay_conv(&m->dec[0], off, c->trunk + c->ch[1], c->dec, 3, 1, 1); off = lay_conv(&m->dec[1], off, c->dec, c->dec, 3, 1, 1);
    off = lay_conv(&m->fh0, off, c->dec + c->ch[0], c->ch[0], 1, 1, 1); off = lay_conv(&m->fh3, off, c->ch[0], 4, 1, 1, 0);
    m->np = off;
    float *h = calloc(off, 4); uint64_t sd = 0x9e3779b97f4a7c15ull ^ seed; rng_next(&sd);
    for (int s = 0; s < 3; s++) { init_conv(h, &m->enc[s][0], &sd); init_conv(h, &m->enc[s][1], &sd); }
    init_conv(h, &m->tin, &sd);
    for (int k = 0; k < c->stem; k++) { init_conv(h, &m->stem[k][0], &sd); init_conv(h, &m->stem[k][1], &sd); }
    for (int k = 0; k < 2 * c->blocks; k++) {   /* as unet's attention stage: LN / QK gains 1, QKV and MLP-in N(0, 1 / C), out projections 0 */
        const at_bp *bp = &m->bp[k]; const int C = m->ad.C, F = m->ad.F; const double s = 1.0 / sqrt((double)C);
        for (int i = 0; i < C; i++) h[bp->ln1g + i] = h[bp->ln2g + i] = h[bp->qg + i] = h[bp->kg + i] = 1.f;
        for (size_t i = 0; i < (size_t)3 * C * C; i++) h[bp->wqkv + i] = (float)(s * gauss(&sd));
        for (size_t i = 0; i < (size_t)2 * F * C; i++) h[bp->w1 + i] = (float)(s * gauss(&sd));
    }
    init_conv(h, &m->dec[0], &sd); init_conv(h, &m->dec[1], &sd); init_conv(h, &m->fh0, &sd);
    init_conv(h, &m->fh3, &sd);   /* phase head: weights 0, bias softplus^-1(phase_init) */
    memset(h + m->fh3.w, 0, (size_t)4 * c->ch[0] * 4);
    for (int i = 0; i < 4; i++) h[m->fh3.b + i] = (float)log(expm1(c->phase_init));
    m->P = nn_malloc(off * 4); m->G = nn_malloc(off * 4); nn_h2d(m->P, h, off * 4); nn_zero(m->G, off * 4); free(h);
    return m;
}
size_t vwt_nparams(const vwt *m) { return m->np; }
float *vwt_params(vwt *m) { return m->P; }
float *vwt_grads(vwt *m) { return m->G; }
void vwt_zero_grad(vwt *m) { nn_zero(m->G, m->np * 4); }

static vt_t vnew(int d, int h, int w, int c) { vt_t t = {nn_malloc((size_t)d * h * w * c * 4), 1, d, h, w, c}; return t; }
static void cact_alloc(cact *a, vt_t in, int co, int s) {
    a->in = in; const int d = (in.d + s - 1) / s, h = (in.h + s - 1) / s;
    a->c = vnew(d, h, in.w, co); a->a = vnew(d, h, in.w, co);
    a->mean = nn_malloc(vt_rows(a->c) * 4); a->rstd = nn_malloc(vt_rows(a->c) * 4);
}
/* the activation buffers for slab T x L (allocated once) */
static void build(vwt *m) {
    if (m->built) return;
    const vwt_cfg *c = &m->cfg; const int T = c->T, L = c->L;
    if (c->cx) {   /* (x0: the shape only) */
        m->xC = (2 + c->cx + 31) / 32 * 32; m->xb = nn_malloc((size_t)T * T * L * m->xC * 2); nn_zero(m->xb, (size_t)T * T * L * m->xC * 2);
        m->x0 = (vt_t){nullptr, 1, T, T, L, 2 + c->cx};
    } else m->x0 = vnew(T, T, L, 2);
    vt_t in = m->x0;
    for (int s = 0; s < 3; s++) { cact_alloc(&m->e[s][0], in, c->ch[s], 2); cact_alloc(&m->e[s][1], m->e[s][0].a, c->ch[s], 1); in = m->e[s][1].a; }
    cact_alloc(&m->ti, in, c->trunk, 1); in = m->ti.a;
    for (int k = 0; k < c->stem; k++) { cact_alloc(&m->st[k][0], in, c->trunk, 1); cact_alloc(&m->st[k][1], m->st[k][0].a, c->trunk, 1); m->stout[k] = vnew(in.d, in.h, in.w, c->trunk); in = m->stout[k]; }
    const long NT = (long)vt_rows(in);
    for (int k = 0; k < 2 * c->blocks; k++) m->tokin[k] = nn_malloc((size_t)NT * c->trunk * 4);
    m->tok = nn_malloc((size_t)NT * c->trunk * 4); m->tmp = nn_malloc((size_t)NT * c->trunk * 4);
    m->tokv = (vt_t){m->tok, 1, in.d, in.h, in.w, c->trunk};
    m->up1 = vnew(2 * in.d, 2 * in.h, L, c->trunk); m->cat1 = vnew(2 * in.d, 2 * in.h, L, c->trunk + c->ch[1]);
    cact_alloc(&m->d0, m->cat1, c->dec, 1); cact_alloc(&m->d1, m->d0.a, c->dec, 1);
    m->up0 = vnew(T / 2, T / 2, L, c->dec); m->cat0 = vnew(T / 2, T / 2, L, c->dec + c->ch[0]);
    cact_alloc(&m->h0, m->cat0, c->ch[0], 1);
    m->lg = vnew(T / 2, T / 2, L, 4);
    m->inc = nn_malloc((size_t)T * T * L * 4); m->ph = nn_malloc((size_t)T * T * L * 4);
    m->scratch = nn_malloc(vt_conv_scratch(m->x0, 0, 3, 1, 1, 1));
    const int t8 = in.d;
    m->gr = (at_geom){t8 * t8, L, 1, 1, L, nullptr, nullptr, nullptr, {0}, nullptr};
    m->gt = (at_geom){L, t8 * t8, 1, t8, t8, nullptr, nullptr, nullptr, {0}, nullptr};
    m->ws_mem = nn_malloc(at_ws_bytes(&m->ad, NT)); at_ws_carve(&m->ws, &m->ad, NT, m->ws_mem);
    m->built = 1;
}

static int conv_ln_fwd(vwt *m, const pconv *p, cact *a, int gelu) {
    const int sd = p->s, sh = p->s;
    if (vt_conv_fwd(a->in, m->P + p->w, m->P + p->b, p->k, sd, sh, 1, a->c, m->scratch)) return -1;
    vt_ln_fwd(a->c, m->P + p->g, m->P + p->be, gelu, a->a, a->mean, a->rstd);
    return 0;
}
/* gy (d/d a->a, may be overwritten) -> gx (d/d a->in; gx.p nullptr: skipped) */
static int conv_ln_bwd(vwt *m, const pconv *p, cact *a, int gelu, vt_t gy, vt_t gc, vt_t gx) {
    vt_ln_bwd(a->c, m->P + p->g, m->P + p->be, gelu, a->mean, a->rstd, gy, gc, m->G + p->g, m->G + p->be);
    return vt_conv_bwd(a->in, gc, m->P + p->w, p->k, p->s, p->s, 1, gx, m->G + p->w, m->G + p->b, m->scratch);
}

int vwt_forward(vwt *m, const uint8_t *im, const uint8_t *va, float *ph, float *inc, int train) {
    build(m); const vwt_cfg *c = &m->cfg; const int T = c->T, L = c->L;
    double tp[6] = {0}; PT(0);
    if (c->cx) {   /* [CT, valid | extra features] */
        if (!m->xf) { fprintf(stderr, "vwt_forward: cfg.cx %d but no features (vwt_set_xfeat)\n", c->cx); return -1; }
        vt_input_bf(im, va, (long long)T * T * L, m->xC, m->xb);
        if (m->xf != m->xb + 2) vt_feat_bf(m->xf, c->cx, (long long)T * T * L, m->xC, 2, m->xb);   /* (vwt_xfeat_buf: written there already) */
        const pconv *p = &m->enc[0][0]; cact *a = &m->e[0][0];
        if (vt_conv_fwd_bfin(m->xb, 1, T, T, L, m->xC, 2 + c->cx, m->P + p->w, m->P + p->b, p->k, p->s, p->s, 1, a->c, m->scratch)) return -1;
        vt_ln_fwd(a->c, m->P + p->g, m->P + p->be, 1, a->a, a->mean, a->rstd);
    } else vt_input(im, va, (long long)T * T * L, m->x0.p);
    for (int s = 0; s < 3; s++) { if ((!(s == 0 && c->cx) && conv_ln_fwd(m, &m->enc[s][0], &m->e[s][0], 1)) || conv_ln_fwd(m, &m->enc[s][1], &m->e[s][1], 1)) return -1; }
    PT(1);
    if (conv_ln_fwd(m, &m->tin, &m->ti, 1)) return -1;
    for (int k = 0; k < c->stem; k++) {
        if (conv_ln_fwd(m, &m->stem[k][0], &m->st[k][0], 1) || conv_ln_fwd(m, &m->stem[k][1], &m->st[k][1], 0)) return -1;
        nn_d2d(m->stout[k].p, m->st[k][0].in.p, vt_numel(m->stout[k]) * 4); vt_add(m->stout[k].p, m->st[k][1].a.p, vt_numel(m->stout[k]));
    }
    const vt_t sin_ = c->stem ? m->stout[c->stem - 1] : m->ti.a; const long NT = (long)vt_rows(sin_); const int t8 = sin_.d, C = c->trunk;
    nn_d2d(m->tok, sin_.p, (size_t)NT * C * 4);
    PT(2);
    for (int k = 0; k < c->blocks; k++) {
        nn_d2d(m->tokin[2 * k], m->tok, (size_t)NT * C * 4);
        if (at_block_fwd(&m->ad, m->P, &m->bp[2 * k], m->tokin[2 * k], m->tmp, &m->gr, &m->ws)) return -1;
        vt_transpose(m->tmp, m->tokin[2 * k + 1], (long long)t8 * t8, L, C);
        if (at_block_fwd(&m->ad, m->P, &m->bp[2 * k + 1], m->tokin[2 * k + 1], m->tmp, &m->gt, &m->ws)) return -1;
        vt_transpose(m->tmp, m->tok, L, (long long)t8 * t8, C);
    }
    PT(3);
    vt_up2_fwd(m->tokv, m->up1); vt_cat_fwd(m->up1, m->e[1][1].a, m->cat1);
    if (conv_ln_fwd(m, &m->dec[0], &m->d0, 1) || conv_ln_fwd(m, &m->dec[1], &m->d1, 1)) return -1;
    vt_up2_fwd(m->d1.a, m->up0); vt_cat_fwd(m->up0, m->e[0][1].a, m->cat0);
    if (conv_ln_fwd(m, &m->fh0, &m->h0, 1)) return -1;
    if (vt_conv_fwd(m->h0.a, m->P + m->fh3.w, m->P + m->fh3.b, 1, 1, 1, 1, m->lg, m->scratch)) return -1;
    vt_phase_fwd(m->lg, inc ? inc : m->inc, ph ? ph : m->ph);
    (void)train;
    PT(4); if (prof_on()) fprintf(stderr, "vwt prof fwd: enc %.0f  tin+stem %.0f  blocks %.0f  dec+head %.0f ms\n", (tp[1] - tp[0]) * 1e3, (tp[2] - tp[1]) * 1e3, (tp[3] - tp[2]) * 1e3, (tp[4] - tp[3]) * 1e3);
    const char *e = nn_check(); if (e) { fprintf(stderr, "vwt_forward: %s\n", e); return -1; }
    return 0;
}

int vwt_backward(vwt *m, const float *gph, const float *ginc) {
    const vwt_cfg *c = &m->cfg; const int L = c->L, C = c->trunk;
    /* scratch gradients: one buffer per resolution class, reused */
    static float *gA = nullptr, *gB = nullptr, *gC = nullptr; static size_t gcap = 0;
    const size_t big = vt_numel(m->cat0) > vt_numel(m->e[0][0].c) * 2 ? vt_numel(m->cat0) : vt_numel(m->e[0][0].c) * 2;
    if (gcap < big) { nn_free(gA); nn_free(gB); nn_free(gC); gA = nn_malloc(big * 4); gB = nn_malloc(big * 4); gC = nn_malloc(big * 4); gcap = big; }
#define LIKE(t, p) ((vt_t){(p), (t).n, (t).d, (t).h, (t).w, (t).c})
    double tp[6] = {0}; PT(0);
    vt_t glg = LIKE(m->lg, gA); vt_phase_bwd(m->lg, gph, ginc, glg);
    vt_t gh = LIKE(m->h0.a, gB); if (vt_conv_bwd(m->h0.a, glg, m->P + m->fh3.w, 1, 1, 1, 1, gh, m->G + m->fh3.w, m->G + m->fh3.b, m->scratch)) return -1;
    vt_t gcat0 = LIKE(m->cat0, gA); if (conv_ln_bwd(m, &m->fh0, &m->h0, 1, gh, LIKE(m->h0.c, gC), gcat0)) return -1;
    vt_t gup0 = LIKE(m->up0, gB), gsk0 = LIKE(m->e[0][1].a, nn_malloc(vt_numel(m->e[0][1].a) * 4));
    vt_cat_bwd(gcat0, gup0, gsk0);
    vt_t gd1 = LIKE(m->d1.a, gC); vt_up2_bwd(gup0, gd1);
    vt_t gd0 = LIKE(m->d0.a, gA); if (conv_ln_bwd(m, &m->dec[1], &m->d1, 1, gd1, LIKE(m->d1.c, gB), gd0)) return -1;
    vt_t gcat1 = LIKE(m->cat1, gC); if (conv_ln_bwd(m, &m->dec[0], &m->d0, 1, gd0, LIKE(m->d0.c, gB), gcat1)) return -1;
    vt_t gup1 = LIKE(m->up1, gA), gsk1 = LIKE(m->e[1][1].a, nn_malloc(vt_numel(m->e[1][1].a) * 4));
    vt_cat_bwd(gcat1, gup1, gsk1);
    vt_t gtok = LIKE(m->tokv, gB); vt_up2_bwd(gup1, gtok);
    const int t8 = m->tokv.d; const long NT = (long)vt_rows(m->tokv);
    PT(1);
    float *gt = gtok.p, *gt2 = gC;
    for (int k = c->blocks - 1; k >= 0; k--) {
        vt_transpose(gt, gt2, (long long)t8 * t8, L, C);   /* to transverse order */
        if (at_block_bwd(&m->ad, m->P, m->G, &m->bp[2 * k + 1], m->tokin[2 * k + 1], gt2, &m->gt, &m->ws)) return -1;
        vt_transpose(gt2, gt, L, (long long)t8 * t8, C);
        if (at_block_bwd(&m->ad, m->P, m->G, &m->bp[2 * k], m->tokin[2 * k], gt, &m->gr, &m->ws)) return -1;
    }
    (void)NT;
    PT(2);
    /* stem: gt = d/d stem output */
    vt_t g = LIKE(m->tokv, gt);
    for (int k = c->stem - 1; k >= 0; k--) {
        vt_t ga1 = LIKE(m->st[k][0].a, gC), gin = LIKE(m->st[k][0].in, gA);
        nn_d2d(gA, g.p, vt_numel(g) * 4);   /* residual path copy (conv_ln_bwd overwrites its gy) */
        vt_t gy2 = LIKE(g, g.p);
        if (conv_ln_bwd(m, &m->stem[k][1], &m->st[k][1], 0, gy2, LIKE(m->st[k][1].c, g.p), ga1)) return -1;
        vt_t gx1 = LIKE(gin, g.p);
        if (conv_ln_bwd(m, &m->stem[k][0], &m->st[k][0], 1, ga1, LIKE(m->st[k][0].c, gC), gx1)) return -1;
        vt_add(g.p, gA, vt_numel(g));   /* + identity */
        (void)gin;
    }
    /* trunk_in */
    vt_t gs2 = LIKE(m->e[2][1].a, gA); if (conv_ln_bwd(m, &m->tin, &m->ti, 1, g, LIKE(m->ti.c, gC), gs2)) return -1;
    PT(3);
    /* stages, adding the skip gradients */
    vt_t gcur = gs2;
    for (int s = 2; s >= 0; s--) {
        if (s == 1) vt_add(gcur.p, gsk1.p, vt_numel(gcur));
        if (s == 0) vt_add(gcur.p, gsk0.p, vt_numel(gcur));
        float *o1 = gcur.p == gA ? gB : gA;
        vt_t ga = LIKE(m->e[s][0].a, o1);
        if (conv_ln_bwd(m, &m->enc[s][1], &m->e[s][1], 1, gcur, LIKE(m->e[s][1].c, gC), ga)) return -1;
        float *o2 = o1 == gA ? gB : gA;
        vt_t gin = s ? LIKE(m->e[s][0].in, o2) : (vt_t){nullptr, 0, 0, 0, 0, 0};
        if (s == 0 && c->cx) {   /* the bf16 padded input conv: weight + bias gradients only */
            const pconv *p = &m->enc[0][0]; cact *a = &m->e[0][0]; const vt_t gc = LIKE(a->c, gC);
            vt_ln_bwd(a->c, m->P + p->g, m->P + p->be, 1, a->mean, a->rstd, ga, gc, m->G + p->g, m->G + p->be);
            if (vt_conv_wgrad_bfin(m->xb, 1, c->T, c->T, L, m->xC, 2 + c->cx, gc, m->P + p->w, p->k, p->s, p->s, 1, m->G + p->w, m->G + p->b, m->scratch)) return -1;
        } else if (conv_ln_bwd(m, &m->enc[s][0], &m->e[s][0], 1, ga, LIKE(m->e[s][0].c, gC), gin)) return -1;
        gcur = gin;
    }
    nn_free(gsk0.p); nn_free(gsk1.p);
    PT(4); if (prof_on()) fprintf(stderr, "vwt prof bwd: dec+head %.0f  blocks %.0f  stem+tin %.0f  enc %.0f ms\n", (tp[1] - tp[0]) * 1e3, (tp[2] - tp[1]) * 1e3, (tp[3] - tp[2]) * 1e3, (tp[4] - tp[3]) * 1e3);
#undef LIKE
    const char *e = nn_check(); if (e) { fprintf(stderr, "vwt_backward: %s\n", e); return -1; }
    return 0;
}

void vwt_set_xfeat(vwt *m, const uint16_t *xf) { m->xf = xf; }
const vwt_cfg *vwt_cfg_of(const vwt *m) { return &m->cfg; }
uint16_t *vwt_xfeat_buf(vwt *m, int *ld) { build(m); if (!m->cfg.cx) return nullptr; if (ld) *ld = m->xC; m->xf = m->xb + 2; return m->xb + 2; }

/* checkpoint header: "VWT1" (cfg without cx) or "VWT2" (full cfg) */
static int read_hdr(FILE *f, vwt_cfg *cfg) {
    char mg[4]; memset(cfg, 0, sizeof *cfg);
    if (fread(mg, 1, 4, f) != 4) return -1;
    if (!memcmp(mg, "VWT1", 4)) return fread(cfg, offsetof(vwt_cfg, cx), 1, f) == 1 ? 0 : -1;
    if (!memcmp(mg, "VWT2", 4)) return fread(cfg, sizeof *cfg, 1, f) == 1 ? 0 : -1;
    return -1;
}
int vwt_ckpt_header(FILE *f, vwt_cfg *cfg) { return read_hdr(f, cfg); }

int vwt_init_from(vwt *m, const char *path, int use_ema) {
    vwt *s = vwt_load(path, use_ema); if (!s) return -1;
    const vwt_cfg *a = &s->cfg, *b = &m->cfg;
    if (a->T != b->T || a->L != b->L || a->trunk != b->trunk || a->stem != b->stem || a->blocks != b->blocks || a->heads != b->heads || a->mlp != b->mlp || a->dec != b->dec ||
        memcmp(a->ch, b->ch, sizeof a->ch) || m->np - m->enc[0][0].b != s->np - s->enc[0][0].b) { fprintf(stderr, "vwt_init_from: %s: architecture differs\n", path); vwt_free(s); return -1; }
    float *hs = malloc(s->np * 4), *hm = malloc(m->np * 4); nn_d2h(hs, s->P, s->np * 4); nn_d2h(hm, m->P, m->np * 4);
    const pconv *ps = &s->enc[0][0], *pm = &m->enc[0][0]; const int taps = 27;
    for (int co = 0; co < pm->co; co++) for (int t = 0; t < taps; t++) for (int ci = 0; ci < pm->ci; ci++)
        hm[pm->w + ((size_t)co * taps + t) * pm->ci + ci] = ci < ps->ci ? hs[ps->w + ((size_t)co * taps + t) * ps->ci + ci] : 0.f;
    memcpy(hm + pm->b, hs + ps->b, (s->np - ps->b) * 4);
    nn_h2d(m->P, hm, m->np * 4); free(hs); free(hm); vwt_free(s);
    fprintf(stderr, "vwt: warm start from %s (%s weights), %d extra input channels zero-initialised\n", path, use_ema ? "EMA" : "live", b->cx);
    return 0;
}

void vwt_free(vwt *m) { if (!m) return; /* (process-lifetime buffers) */ nn_free(m->P); nn_free(m->G); free(m); }

/* checkpoint: "VWT1", cfg, np, params (fp32), has_ema + ema, extra bytes */
int vwt_save(const vwt *m, const char *path, const float *ema, const void *extra, size_t extra_bytes) {
    char tmp[1100]; snprintf(tmp, sizeof tmp, "%s.tmp", path); FILE *f = fopen(tmp, "wb"); if (!f) return -1;
    float *h = malloc(m->np * 4); int rc = 0; const int has = ema != nullptr; const uint64_t np = m->np, eb = extra_bytes;
    rc |= fwrite("VWT2", 1, 4, f) != 4; rc |= fwrite(&m->cfg, sizeof m->cfg, 1, f) != 1; rc |= fwrite(&np, 8, 1, f) != 1;
    nn_d2h(h, m->P, m->np * 4); rc |= fwrite(h, 4, m->np, f) != m->np;
    rc |= fwrite(&has, 4, 1, f) != 1; if (has) { nn_d2h(h, ema, m->np * 4); rc |= fwrite(h, 4, m->np, f) != m->np; }
    rc |= fwrite(&eb, 8, 1, f) != 1; if (eb) rc |= fwrite(extra, 1, eb, f) != eb;
    free(h); rc |= fclose(f) != 0;
    if (!rc) rc = rename(tmp, path);
    return rc ? -1 : 0;
}
vwt *vwt_load(const char *path, int use_ema) {
    FILE *f = fopen(path, "rb"); if (!f) { fprintf(stderr, "vwt: cannot read %s\n", path); return nullptr; }
    vwt_cfg cfg; uint64_t np; int has = 0;
    if (read_hdr(f, &cfg) || fread(&np, 8, 1, f) != 1) { fclose(f); fprintf(stderr, "vwt: %s: bad header\n", path); return nullptr; }
    vwt *m = vwt_create(&cfg, 1); if (!m || m->np != np) { fclose(f); fprintf(stderr, "vwt: %s: parameter count mismatch\n", path); return nullptr; }
    float *h = malloc(np * 4); int rc = fread(h, 4, np, f) != np;
    if (!rc && fread(&has, 4, 1, f) == 1 && has && use_ema) rc = fread(h, 4, np, f) != np;
    fclose(f); if (rc) { free(h); fprintf(stderr, "vwt: %s: short read\n", path); return nullptr; }
    nn_h2d(m->P, h, np * 4); free(h);
    return m;
}
