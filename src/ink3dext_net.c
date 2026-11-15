/* ink3d-ext network (src/ink3dext.h): the gambiarra ResEnc U-Net on xnet (fp16 NDHWC, fp32 epilogues / norms). */
#include "ink3dext.h"
#include "wload.h"
#include "xnet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { GB_MAXST = 8, GB_MAXBLK = 8 };

typedef struct { xconv c; float *g, *b; } cin;               /* conv + InstanceNorm affine (device) */
typedef struct { cin c1, c2, sk; int has_sk, pool; xconv skp; int has_skp; } rblock;   /* skp: the pooled skip as one conv (2x2x2 stride 2, w / 8), MX path */
struct gb_net {
    int nst, nblk[GB_MAXST], feat[GB_MAXST], has_stem;
    i3_geom g;
    cin stem;
    rblock enc[GB_MAXST][GB_MAXBLK];
    xconv up[GB_MAXST - 1]; cin dec[GB_MAXST - 1];
    float *hw; float hb;
    float *w1, *b1, *wsk;   /* MX path: the one-channel first conv (stem, or stage 0 block 0 conv1) [32][27] + bias, and block 0's 1x1 skip [32] (fp32, device) */
    float *logits, *pooled; int cap;
};

static const wl_tensor *need(const wl_dict *d, const char *name) {
    const wl_tensor *t = wl_get(d, name);
    if (!t) fprintf(stderr, "ink3d-ext: tensor %s missing\n", name);
    return t;
}
static float *devf(const wl_dict *d, const char *name) { const wl_tensor *t = need(d, name); return t ? xdev_floats(t->data, t->numel) : nullptr; }
/* <pfx>.conv.weight / .conv.bias (when bias) / .norm.weight / .norm.bias */
static int cin_of(const wl_dict *d, cin *c, const char *pfx, int k, int s, int bias) {
    char w[256], b[256], g[256], be[256];
    snprintf(w, sizeof w, "%s.conv.weight", pfx); snprintf(b, sizeof b, "%s.conv.bias", pfx);
    snprintf(g, sizeof g, "%s.norm.weight", pfx); snprintf(be, sizeof be, "%s.norm.bias", pfx);
    const wl_tensor *wt = need(d, w), *bt = bias ? need(d, b) : nullptr; if (!wt || (bias && !bt)) return -1;
    if (wt->ndim != 5 || wt->shape[2] != k) { fprintf(stderr, "ink3d-ext: %s has an unexpected shape\n", w); return -1; }
    if (xconv_init(&c->c, wt->data, (int)wt->shape[0], (int)wt->shape[1], k, k, k, s, s, s, k / 2, k / 2, k / 2, nullptr, bt ? bt->data : nullptr)) return -1;
    c->g = devf(d, g); c->b = devf(d, be);
    return c->g && c->b ? 0 : -1;
}
static void cin_free(cin *c) { xconv_free(&c->c); xfree_dev(c->g); xfree_dev(c->b); }

/* model geometry (src/ink3dext.h) */
static int geom_of(const char *model, i3_geom *g) {
    if (!strcmp(model, "gambiarra")) { *g = (i3_geom){.crop = 128, .pool_in = 2, .net_in = 64, .pool_out = 1, .out = 64, .parent = 256, .poff = 64}; return 0; }
    if (!strcmp(model, "dino3d")) { *g = (i3_geom){.crop = 256, .pool_in = 1, .net_in = 256, .pool_out = 2, .out = 128, .parent = 256, .poff = 0}; return 0; }
    fprintf(stderr, "ink3d-ext: unknown --model %s (gambiarra | dino3d)\n", model); return -1;
}

gb_net *gb_load(const char *weights, const char *model, i3_geom *geo) {
    i3_geom G; if (geom_of(model, &G)) return nullptr;
    wl_dict *d = wl_load(weights, nullptr);   /* .pth: ema_model first (the recommended inference weights) */
    if (!d) { fprintf(stderr, "ink3d-ext: %s: %s\n", weights, wl_error()); return nullptr; }
    gb_net *n = calloc(1, sizeof *n); int rc = 0; n->g = G;
    char p[256];
    /* stages / blocks / widths from the tensor names */
    while (n->nst < GB_MAXST) { snprintf(p, sizeof p, "shared_encoder.stages.%d.blocks.0.conv1.conv.weight", n->nst); if (!wl_get(d, p)) break; n->nst++; }
    if (n->nst < 2) { fprintf(stderr, "ink3d-ext: %s: no encoder stages\n", weights); rc = -1; }
    for (int s = 0; s < n->nst && !rc; s++) {
        while (n->nblk[s] < GB_MAXBLK) { snprintf(p, sizeof p, "shared_encoder.stages.%d.blocks.%d.conv1.conv.weight", s, n->nblk[s]); if (!wl_get(d, p)) break; n->nblk[s]++; }
        snprintf(p, sizeof p, "shared_encoder.stages.%d.blocks.%d.conv1.conv.weight", s, n->nblk[s]);
        if (wl_get(d, p)) { fprintf(stderr, "ink3d-ext: stage %d has more than %d blocks\n", s, GB_MAXBLK); rc = -1; }
    }
    n->has_stem = wl_get(d, "shared_encoder.stem.convs.0.conv.weight") != nullptr;
    if (!rc && n->has_stem) {
        rc |= cin_of(d, &n->stem, "shared_encoder.stem.convs.0", 3, 1, 1);
        if (wl_get(d, "shared_encoder.stem.convs.1.conv.weight")) { fprintf(stderr, "ink3d-ext: stem has more than one conv\n"); rc = -1; }
    }
    for (int s = 0; s < n->nst && !rc; s++) for (int k = 0; k < n->nblk[s] && !rc; k++) {
        rblock *R = &n->enc[s][k]; const int st = k == 0 && s > 0 ? 2 : 1;
        snprintf(p, sizeof p, "shared_encoder.stages.%d.blocks.%d.conv1", s, k); rc |= cin_of(d, &R->c1, p, 3, st, 1);
        snprintf(p, sizeof p, "shared_encoder.stages.%d.blocks.%d.conv2", s, k); rc |= cin_of(d, &R->c2, p, 3, 1, 1);
        R->pool = st == 2;
        snprintf(p, sizeof p, "shared_encoder.stages.%d.blocks.%d.skip.%d.conv.weight", s, k, R->pool);
        R->has_sk = wl_get(d, p) != nullptr;
        if (R->has_sk) { snprintf(p, sizeof p, "shared_encoder.stages.%d.blocks.%d.skip.%d", s, k, R->pool); rc |= cin_of(d, &R->sk, p, 1, 1, 0); }
        else if (k == 0 && s == 0 && !n->has_stem) { fprintf(stderr, "ink3d-ext: stage %d block %d has no skip projection\n", s, k); rc = -1; }
        /* (a strided block keeping its width: skip = AvgPool3d(2) alone) */
        if (!rc && k == 0) n->feat[s] = R->c1.c.co;
    }
    for (int s = 0; s < n->nst - 1 && !rc; s++) {
        char q[256];
        snprintf(p, sizeof p, "shared_decoder.transpconvs.%d.weight", s); snprintf(q, sizeof q, "shared_decoder.transpconvs.%d.bias", s);
        const wl_tensor *w = need(d, p), *b = need(d, q);
        if (!w || !b || w->ndim != 5 || w->shape[2] != 2) { rc = -1; break; }
        rc |= xtconv3_init(&n->up[s], w->data, (int)w->shape[0], (int)w->shape[1], b->data);
        snprintf(p, sizeof p, "shared_decoder.stages.%d.convs.0", s); rc |= cin_of(d, &n->dec[s], p, 3, 1, 1);
        snprintf(p, sizeof p, "shared_decoder.stages.%d.convs.1.conv.weight", s);
        if (wl_get(d, p)) { fprintf(stderr, "ink3d-ext: decoder stage %d has more than one conv\n", s); rc = -1; }
    }
    if (!rc && xnet_mx()) {   /* the MX path's fused pieces */
        const char *pf = n->has_stem ? "shared_encoder.stem.convs.0" : "shared_encoder.stages.0.blocks.0.conv1";
        snprintf(p, sizeof p, "%s.conv.weight", pf); const wl_tensor *w = wl_get(d, p);
        snprintf(p, sizeof p, "%s.conv.bias", pf); const wl_tensor *b = wl_get(d, p);
        if (w && w->ndim == 5 && w->shape[0] == 32 && w->shape[1] == 1 && w->shape[2] == 3 && w->shape[3] == 3 && w->shape[4] == 3) {
            n->w1 = xdev_floats(w->data, 32 * 27); n->b1 = b ? xdev_floats(b->data, 32) : nullptr;
            const rblock *R = &n->enc[0][0];
            if (!n->has_stem && R->has_sk && !R->pool) {
                const wl_tensor *ws = wl_get(d, "shared_encoder.stages.0.blocks.0.skip.0.conv.weight");
                if (ws && ws->numel == 32) n->wsk = xdev_floats(ws->data, 32);
            }
        }
        for (int s = 1; s < n->nst && !rc; s++) {   /* pooled 1x1 skips -> 2x2x2 stride-2 convs with the weights / 8 */
            rblock *R = &n->enc[s][0]; if (!R->has_sk || !R->pool) continue;
            snprintf(p, sizeof p, "shared_encoder.stages.%d.blocks.0.skip.1.conv.weight", s); const wl_tensor *ws = wl_get(d, p);
            if (!ws || ws->ndim != 5 || ws->shape[2] != 1) continue;
            const int co = (int)ws->shape[0], ci = (int)ws->shape[1]; float *w8 = malloc(sizeof(float) * (size_t)co * ci * 8);
            for (int o = 0; o < co; o++) for (int i = 0; i < ci; i++) for (int t = 0; t < 8; t++) w8[((size_t)o * ci + i) * 8 + t] = ws->data[(size_t)o * ci + i] / 8.f;
            if (!xconv_init(&R->skp, w8, co, ci, 2, 2, 2, 2, 2, 2, 0, 0, 0, nullptr, nullptr)) R->has_skp = 1;
            free(w8);
        }
    }
    if (!rc) {
        const wl_tensor *w = need(d, "task_heads.ink.weight"), *b = need(d, "task_heads.ink.bias");
        if (!w || !b || w->numel != 32) rc = -1; else { n->hw = xdev_floats(w->data, 32); n->hb = b->data[0]; rc |= !n->hw; }
    }
    if (!rc && G.net_in % (1 << (n->nst - 1))) { fprintf(stderr, "ink3d-ext: input %d not divisible by 2^%d\n", G.net_in, n->nst - 1); rc = -1; }
    if (!rc) {
        fprintf(stderr, "ink3d-ext: %s: %d stages (", model, n->nst);
        for (int s = 0; s < n->nst; s++) fprintf(stderr, "%s%d x%d", s ? ", " : "", n->feat[s], n->nblk[s]);
        fprintf(stderr, ")%s; crop %d^3 level 0 -> net %d^3 -> out %d^3 at 4.8 um\n", n->has_stem ? ", stem" : "", G.crop, G.net_in, G.out);
    }
    wl_free(d);
    if (rc || xnet_check("ink3d-ext load")) { fprintf(stderr, "ink3d-ext: cannot load %s\n", weights); gb_free(n); return nullptr; }
    if (geo) *geo = G;
    return n;
}

void gb_free(gb_net *n) {
    if (!n) return;
    if (n->has_stem) cin_free(&n->stem);
    for (int s = 0; s < n->nst; s++) for (int k = 0; k < n->nblk[s]; k++) {
        rblock *R = &n->enc[s][k]; cin_free(&R->c1); cin_free(&R->c2); if (R->has_sk) cin_free(&R->sk); if (R->has_skp) xconv_free(&R->skp);
    }
    for (int s = 0; s < n->nst - 1; s++) { xconv_free(&n->up[s]); cin_free(&n->dec[s]); }
    xfree_dev(n->hw); xfree_dev(n->logits); xfree_dev(n->pooled); xfree_dev(n->w1); xfree_dev(n->b1); xfree_dev(n->wsk);
    free(n);
}

/* BasicBlockD: LReLU(IN(conv2(LReLU(IN(conv1 x)))) + skip(x)), skip = IN(conv1x1([avgpool2] x)) or x. */
static xt rblock_fwd(const rblock *R, xt x) {
    xt h = xconv_fwd(&R->c1.c, x, nullptr, XA_NONE);
    xgn(h, h.c, R->c1.g, R->c1.b, 1e-5f, nullptr, XA_LRELU, h);
    xt h2 = xconv_fwd(&R->c2.c, h, nullptr, XA_NONE); xt_free(&h);
    xt r = x, pooled = {0};
    if (R->has_sk) {
        xt src = x; if (R->pool) { pooled = xavgpool3d_k2(x); src = pooled; }
        r = xconv_fwd(&R->sk.c, src, nullptr, XA_NONE); xgn(r, r.c, R->sk.g, R->sk.b, 1e-5f, nullptr, XA_NONE, r);
        if (pooled.p) xt_free(&pooled);
    } else if (R->pool) r = xavgpool3d_k2(x);   /* same width, strided: the pooled input */
    xgn(h2, h2.c, R->c2.g, R->c2.b, 1e-5f, &r, XA_LRELU, h2);
    if (R->has_sk || R->pool) xt_free(&r);
    return h2;
}

/* ---- the MX-fp4 path (xnet_mx() != 0): every conv with >= 32 input channels on the fp4 tensor cores, InstanceNorm sums from the conv
   epilogues, one fused apply per norm (normalise, residual, LeakyReLU, the next conv's MX copy); fp16 kept only where a later op reads it
   (block outputs: the identity / pooled residual; the last decoder stage: the head) ---- */
static xt cin_mx(const cin *c, xt x, float **coef) {   /* conv + its IN coefficients; the raw fp16 output */
    double *st; xt y = xconv_fwd_st(&c->c, x, &st);
    *coef = xin_coef(st, y.n, y.c, (long long)y.d * y.h * y.w, c->g, c->b, 1e-5f);
    xfree_dev(st);
    return y;
}
static xt rblock_mx(const rblock *R, xt x, int need_half) {   /* x: q (+ fp16 when it is an identity residual); returns q (+ fp16 if need_half) */
    float *c1, *c2, *cs = nullptr;
    xt h = cin_mx(&R->c1, x, &c1); xin_apply(&h, c1, nullptr, nullptr, XA_LRELU, 0, 1); xfree_dev(c1);
    xt h2 = cin_mx(&R->c2, h, &c2); xt_free(&h);
    xt r = x, pooled = {0};
    if (R->has_skp) r = cin_mx(&(cin){.c = R->skp, .g = R->sk.g, .b = R->sk.b}, x, &cs);
    else if (R->has_sk) {
        xt src = x; if (R->pool) { pooled = xavgpool3d_k2(x); src = pooled; }
        r = cin_mx(&R->sk, src, &cs);
        if (pooled.p) xt_free(&pooled);
    } else if (R->pool) r = xavgpool3d_k2(x);
    xin_apply(&h2, c2, &r, cs, XA_LRELU, need_half, 1); xfree_dev(c2); xfree_dev(cs);
    if (R->has_sk || R->pool) xt_free(&r);
    return h2;
}
/* the first block of a stem-less net on a one-channel input (gambiarra stage 0): conv1 fused (xstem1), the 1x1 skip analytic */
static xt rblock0_mx(const gb_net *n, xt x, int need_half) {
    const rblock *R = &n->enc[0][0]; float *rc = nullptr, *c2;
    xt h = xstem1(x, n->w1, n->b1, R->c1.g, R->c1.b, 1e-5f, n->wsk, R->sk.g, R->sk.b, &rc, XA_LRELU, 0, 1);
    xt h2 = cin_mx(&R->c2, h, &c2); xt_free(&h);
    xin_apply(&h2, c2, &x, rc, XA_LRELU, need_half, 1); xfree_dev(c2); xfree_dev(rc);
    return h2;
}
static int gb_forward_mx(gb_net *n, xt x, int B, const float **logits_dev) {
    const int N = n->g.net_in;
    if (n->has_stem) {
        xt y;
        if (n->w1) y = xstem1(x, n->w1, n->b1, n->stem.g, n->stem.b, 1e-5f, nullptr, nullptr, nullptr, nullptr, XA_LRELU, !n->enc[0][0].has_sk, 1);
        else { float *c; y = cin_mx(&n->stem, x, &c); xin_apply(&y, c, nullptr, nullptr, XA_LRELU, 1, 1); xfree_dev(c); }
        xt_free(&x); x = y;
    }
    xt skip[GB_MAXST];
    for (int s = 0; s < n->nst; s++) {
        for (int k = 0; k < n->nblk[s]; k++) {
            /* fp16 of a block's output only for an identity residual (the next block of the stage, or a pooled skip without a conv) */
            const int nh = k + 1 < n->nblk[s] ? !n->enc[s][k + 1].has_sk : s + 1 < n->nst && !n->enc[s + 1][0].has_sk && !n->enc[s + 1][0].has_skp;
            const int nh2 = nh || (s + 1 < n->nst && n->enc[s + 1][0].has_sk && !n->enc[s + 1][0].has_skp && k + 1 == n->nblk[s]);
            xt y = s == 0 && k == 0 && !n->has_stem && n->w1 && n->wsk ? rblock0_mx(n, x, nh2) : rblock_mx(&n->enc[s][k], x, nh2);
            if (!(s > 0 && k == 0)) xt_free(&x);
            x = y;
        }
        skip[s] = x;
    }
    xt lo = skip[n->nst - 1]; float *lcoef = nullptr;
    for (int s = 0; s < n->nst - 1; s++) {
        xt c = xtconv_cat_q(&n->up[s], lo, skip[n->nst - 2 - s]); xt_free(&lo); xt_free(&skip[n->nst - 2 - s]);
        float *cf; xt h = cin_mx(&n->dec[s], c, &cf); xt_free(&c);
        const int last = s == n->nst - 2;
        if (last) { lo = h; lcoef = cf; break; }
        xin_apply(&h, cf, nullptr, nullptr, XA_LRELU, 0, 1); xfree_dev(cf);
        lo = h;
    }
    const size_t nv = (size_t)N * N * N, ov = (size_t)n->g.out * n->g.out * n->g.out;
    if (n->cap < B) {
        xfree_dev(n->logits); xfree_dev(n->pooled); n->pooled = nullptr;
        n->logits = xmalloc_dev((size_t)B * nv * 4);
        if (n->g.pool_out > 1) n->pooled = xmalloc_dev((size_t)B * ov * 4);
        n->cap = n->logits && (n->g.pool_out == 1 || n->pooled) ? B : 0;
    }
    if (!n->cap) { xt_free(&lo); xfree_dev(lcoef); fprintf(stderr, "ink3d-ext: out of device memory (logits)\n"); return -1; }
    if (xin_head(&lo, lcoef, XA_LRELU, n->hw, n->hb, n->logits)) { xin_apply(&lo, lcoef, nullptr, nullptr, XA_LRELU, 1, 0); xhead(lo, n->hw, n->hb, n->logits); }
    xfree_dev(lcoef); xt_free(&lo);
    if (n->g.pool_out == 2) { i3_pool2(n->logits, B, N, n->pooled); *logits_dev = n->pooled; }
    else *logits_dev = n->logits;
    return xnet_check("ink3d-ext forward (MX)");
}

static int gb_forward_t(gb_net *n, xt x, int B, const float **logits_dev);
int gb_forward(gb_net *n, const uint8_t *u8_dev, const float *lut_dev, int B, const float **logits_dev) {
    const int N = n->g.net_in;
    xnet_seq_reset();
    xt x = xt_new(B, N, N, N, 1);
    if (!x.p) { fprintf(stderr, "ink3d-ext: out of device memory (input)\n"); return -1; }
    i3_prep(u8_dev, lut_dev, B, n->g.crop, n->g.pool_in, x.p);
    return gb_forward_t(n, x, B, logits_dev);
}
int gb_forward_x(gb_net *n, void *x_half_dev, int B, const float **logits_dev) {
    const int N = n->g.net_in; xnet_seq_reset();
    xt x = {x_half_dev, B, N, N, N, 1, nullptr};
    return gb_forward_t(n, x, B, logits_dev);
}
static int gb_forward_t(gb_net *n, xt x, int B, const float **logits_dev) {
    const int N = n->g.net_in;
    if (xnet_mx() && !getenv("UFSM_XNET_MX_UNFUSED")) return gb_forward_mx(n, x, B, logits_dev);
    if (n->has_stem) {
        xt y = xconv_fwd(&n->stem.c, x, nullptr, XA_NONE); xt_free(&x);
        xgn(y, y.c, n->stem.g, n->stem.b, 1e-5f, nullptr, XA_LRELU, y);
        x = y;
    }
    xt skip[GB_MAXST];
    for (int s = 0; s < n->nst; s++) {
        for (int k = 0; k < n->nblk[s]; k++) {
            xt y = rblock_fwd(&n->enc[s][k], x);
            if (!(s > 0 && k == 0)) xt_free(&x);   /* the previous stage's output stays as its skip */
            x = y;
        }
        skip[s] = x;
    }
    xt lo = skip[n->nst - 1];
    for (int s = 0; s < n->nst - 1; s++) {
        xt u = xconv_fwd(&n->up[s], lo, nullptr, XA_NONE);
        xt_free(&lo);
        xt c = xconcat(u, skip[n->nst - 2 - s]); xt_free(&u); xt_free(&skip[n->nst - 2 - s]);
        xt h = xconv_fwd(&n->dec[s].c, c, nullptr, XA_NONE); xt_free(&c);
        xgn(h, h.c, n->dec[s].g, n->dec[s].b, 1e-5f, nullptr, XA_LRELU, h);
        lo = h;
    }
    const size_t nv = (size_t)N * N * N, ov = (size_t)n->g.out * n->g.out * n->g.out;
    if (n->cap < B) {
        xfree_dev(n->logits); xfree_dev(n->pooled); n->pooled = nullptr;
        n->logits = xmalloc_dev((size_t)B * nv * 4);
        if (n->g.pool_out > 1) n->pooled = xmalloc_dev((size_t)B * ov * 4);
        n->cap = n->logits && (n->g.pool_out == 1 || n->pooled) ? B : 0;
    }
    if (!n->cap) { xt_free(&lo); fprintf(stderr, "ink3d-ext: out of device memory (logits)\n"); return -1; }
    xhead(lo, n->hw, n->hb, n->logits);
    xt_free(&lo);
    if (n->g.pool_out == 2) { i3_pool2(n->logits, B, N, n->pooled); *logits_dev = n->pooled; }
    else *logits_dev = n->logits;
    return xnet_check("ink3d-ext forward");
}

void gb_set_tile(gb_net *n, int T) { n->g.net_in = T; n->g.out = T / n->g.pool_out; xfree_dev(n->logits); xfree_dev(n->pooled); n->logits = n->pooled = nullptr; n->cap = 0; }
size_t gb_crop_bytes_gpu(const gb_net *n) {
    /* N^3 voxels at the network input: the input, stage-0 tensors (32 ch: conv1 / conv2 / skip / output), the 64-channel concat, the
       decoder output; the coarser levels add ~1 / 4 */
    const size_t v = (size_t)n->g.net_in * n->g.net_in * n->g.net_in;
    return v * 2 * (1 + 8 + 4 * 32 + 64 + 32 + 32) * 5 / 4 + v * 4;
}

/* ---- normalisation (host) ---- */
void i3_percentiles(const uint64_t hist[256], uint64_t n, double *lo, double *hi) {
    const double q[2] = {1.0 / 100.0, 99.0 / 100.0};   /* np.percentile: q / 100, virtual index (n - 1) q, numpy _lerp */
    double out[2];
    for (int k = 0; k < 2; k++) {
        const double virt = (double)(n - 1) * q[k], below = floor(virt), t = virt - below;
        uint64_t ia = (uint64_t)below, ib = ia + 1; if (ib > n - 1) ib = n - 1;
        int a = -1, b = -1; uint64_t cum = 0;
        for (int v = 0; v < 256 && b < 0; v++) { cum += hist[v]; if (a < 0 && cum > ia) a = v; if (cum > ib) b = v; }
        if (a < 0) a = 255; if (b < 0) b = 255;
        const double fa = a, fb = b, diff = fb - fa;
        out[k] = t >= 0.5 ? fb - diff * (1.0 - t) : fa + diff * t;
    }
    *lo = out[0]; *hi = out[1];
}
void i3_lut(double lo, double hi, float lut[256]) {
    for (int v = 0; v < 256; v++) {
        if (hi - lo < 1e-6) { lut[v] = 0.f; continue; }
        double c = v < lo ? lo : v > hi ? hi : (double)v;
        float x = (float)c; x = (float)((double)x - lo); x = (float)((double)x / (hi - lo));
        lut[v] = x;
    }
}
