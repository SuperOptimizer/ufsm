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
typedef struct { cin c1, c2, sk; int has_sk, pool; } rblock;
struct gb_net {
    int nst, nblk[GB_MAXST], feat[GB_MAXST], has_stem;
    i3_geom g;
    cin stem;
    rblock enc[GB_MAXST][GB_MAXBLK];
    xconv up[GB_MAXST - 1]; cin dec[GB_MAXST - 1];
    float *hw; float hb;
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
        rblock *R = &n->enc[s][k]; cin_free(&R->c1); cin_free(&R->c2); if (R->has_sk) cin_free(&R->sk);
    }
    for (int s = 0; s < n->nst - 1; s++) { xconv_free(&n->up[s]); cin_free(&n->dec[s]); }
    xfree_dev(n->hw); xfree_dev(n->logits); xfree_dev(n->pooled);
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

int gb_forward(gb_net *n, const uint8_t *u8_dev, const float *lut_dev, int B, const float **logits_dev) {
    const int N = n->g.net_in;
    xt x = xt_new(B, N, N, N, 1);
    if (!x.p) { fprintf(stderr, "ink3d-ext: out of device memory (input)\n"); return -1; }
    i3_prep(u8_dev, lut_dev, B, n->g.crop, n->g.pool_in, x.p);
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
