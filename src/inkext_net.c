/* ufsm ink-ext networks (inkext.h): v8in (ResNet3D-50 + 2D decoder), Hecate (ResNet-152 3D + 3D U-Net decoder + depth
   attention), Reader V3 (3D depth-fusion stem + 2D residual U-Net) on xnet. Architectures follow the published code:
   v8in ink8um/modeling.py, Hecate hecate.py, Reader V3 villa ink-detection koine_machines (hybrid_3d2d.py) + vesuvius
   build_network_from_config (merge-ink-pipelines). */
#include "inkext.h"
#include "wload.h"
#include "xnet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const ie_spec ie_specs[IE_NMODELS] = {
    {"v8in", 24, 64, 7.91, 21, 12},
    {"hecate24", 64, 256, 2.399, 128, 32},
    {"hecate96", 16, 64, 9.596, 32, 8},
    {"reader3", 17, 384, 9.596, 192, 8},
    {"ink9um", 17, 128, 9.596, 64, 8},   /* scrollprize/ink_9um: 17 of 21 (9.6 um pooled) / 28 (native 9.362) layers, 128 px, overlap 0.5 */
};
int ie_model_kind(const char *name) {
    for (int k = 0; k < IE_NMODELS; k++) if (!strcmp(name, ie_specs[k].name)) return k;
    return -1;
}

typedef struct { xconv c1, c2, c3, ds; int has_ds; } bneck;
typedef struct { xconv c1, c2; xconv sk; int has_sk, stride; float *g1, *b1, *g2, *b2, *gs, *bs; } bbd;   /* Reader BasicBlockD */
typedef struct { xconv c1, c2, sc; float *g1, *b1, *g2, *b2; } rcb;   /* Hecate ResConvBlock3D */

struct ie_net {
    int kind; ie_spec sp;
    /* input normalisation */
    float in_w, in_b, in_rm, in_rv;   /* v8in InstanceNorm affine (w, b) / Hecate BatchNorm3d(1) */
    /* ResNet3D encoder (v8in, Hecate) */
    xconv stem; int nlayer[4]; bneck *layer[4];
    /* v8in decoder */
    xconv dec[3]; float *logit_w; float logit_b;
    /* Hecate decoder */
    xconv red[4]; float *red_g[4], *red_b[4]; rcb blk[3]; float *attn_w; float attn_b, zscale; int margin, xys;
    /* Reader */
    xconv f1, f2; float *f1g, *f1b, *f2g, *f2b; float *att_w; float att_b;
    xconv rstem; float *rstem_g, *rstem_b; int nst; int nblk[6]; bbd *st[6];
    xconv tconv[5], dconv[5]; float *dg[5], *db[5];
};

static const wl_tensor *need(const wl_dict *W, const char *name) {
    const wl_tensor *t = wl_get(W, name);
    if (!t) fprintf(stderr, "ink-ext: weight %s missing\n", name);
    return t;
}
/* conv from W[wname] (3D [co][ci][kd][kh][kw] or 2D [co][ci][kh][kw]) with an optional BatchNorm (prefix bn: .weight .bias
   .running_mean .running_var) and conv bias (cbias) folded into the epilogue scale / bias */
static int mkconv(xconv *c, const wl_dict *W, const char *wname, const char *bn, const char *cbias, int sd, int sh, int sw, int pd, int ph, int pw) {
    const wl_tensor *w = need(W, wname); if (!w) return -1;
    int co = (int)w->shape[0], ci = (int)w->shape[1], kd = 1, kh, kw;
    if (w->ndim == 5) { kd = (int)w->shape[2]; kh = (int)w->shape[3]; kw = (int)w->shape[4]; }
    else if (w->ndim == 4) { kh = (int)w->shape[2]; kw = (int)w->shape[3]; }
    else { fprintf(stderr, "ink-ext: %s is not a conv weight\n", wname); return -1; }
    float *sc = malloc(co * sizeof *sc), *bi = malloc(co * sizeof *bi); const float *cb = nullptr;
    if (cbias) { const wl_tensor *b = need(W, cbias); if (!b) { free(sc); free(bi); return -1; } cb = b->data; }
    if (bn) {
        char n[4][300]; static const char *suf[4] = {"weight", "bias", "running_mean", "running_var"}; const wl_tensor *t[4];
        for (int k = 0; k < 4; k++) { snprintf(n[k], sizeof n[k], "%s.%s", bn, suf[k]); if (!(t[k] = need(W, n[k]))) { free(sc); free(bi); return -1; } }
        xbn_fold(co, t[0]->data, t[1]->data, t[2]->data, t[3]->data, 1e-5f, cb, sc, bi);
    } else for (int k = 0; k < co; k++) { sc[k] = 1.f; bi[k] = cb ? cb[k] : 0.f; }
    const int rc = xconv_init(c, w->data, co, ci, kd, kh, kw, sd, sh, sw, pd, ph, pw, sc, bi);
    free(sc); free(bi);
    return rc;
}
static float *devvec(const wl_dict *W, const char *name, int n) {
    const wl_tensor *t = need(W, name); if (!t) return nullptr;
    if (t->numel != n) { fprintf(stderr, "ink-ext: %s has %lld values, expected %d\n", name, (long long)t->numel, n); return nullptr; }
    return xdev_floats(t->data, (size_t)n);
}
static float scal(const wl_dict *W, const char *name, int *bad) {
    const wl_tensor *t = need(W, name); if (!t || t->numel < 1) { *bad = 1; return 0.f; }
    return t->data[0];
}

/* ---- ResNet3D bottleneck encoder (kenshohara 3D-ResNets, shortcut B) ---- */
static int load_resnet(ie_net *n, const wl_dict *W, const char *pre, const int *nb) {
    char a[300], b[300], c[300];
    int inpl = 64;
    for (int L = 0; L < 4; L++) {
        const int planes = 64 << L, stride = L ? 2 : 1;
        n->nlayer[L] = nb[L]; n->layer[L] = calloc(nb[L], sizeof(bneck));
        for (int i = 0; i < nb[L]; i++) {
            bneck *B = &n->layer[L][i]; const int s = i ? 1 : stride;
            snprintf(a, sizeof a, "%slayer%d.%d.conv1.weight", pre, L + 1, i); snprintf(b, sizeof b, "%slayer%d.%d.bn1", pre, L + 1, i);
            if (mkconv(&B->c1, W, a, b, nullptr, 1, 1, 1, 0, 0, 0)) return -1;
            snprintf(a, sizeof a, "%slayer%d.%d.conv2.weight", pre, L + 1, i); snprintf(b, sizeof b, "%slayer%d.%d.bn2", pre, L + 1, i);
            if (mkconv(&B->c2, W, a, b, nullptr, s, s, s, 1, 1, 1)) return -1;
            snprintf(a, sizeof a, "%slayer%d.%d.conv3.weight", pre, L + 1, i); snprintf(b, sizeof b, "%slayer%d.%d.bn3", pre, L + 1, i);
            if (mkconv(&B->c3, W, a, b, nullptr, 1, 1, 1, 0, 0, 0)) return -1;
            B->has_ds = i == 0 && (stride != 1 || inpl != planes * 4);
            if (B->has_ds) {
                snprintf(a, sizeof a, "%slayer%d.%d.downsample.0.weight", pre, L + 1, i); snprintf(c, sizeof c, "%slayer%d.%d.downsample.1", pre, L + 1, i);
                if (mkconv(&B->ds, W, a, c, nullptr, s, s, s, 0, 0, 0)) return -1;
            }
            inpl = planes * 4;
        }
    }
    return 0;
}
static xt bneck_fwd(const bneck *B, xt x) {
    xt o1 = xconv_fwd(&B->c1, x, nullptr, XA_RELU);
    xt o2 = xconv_fwd(&B->c2, o1, nullptr, XA_RELU); xt_free(&o1);
    xt r = B->has_ds ? xconv_fwd(&B->ds, x, nullptr, XA_NONE) : x;
    xt o3 = xconv_fwd(&B->c3, o2, &r, XA_RELU); xt_free(&o2);
    if (B->has_ds) xt_free(&r);
    return o3;
}
/* stem output -> the four layer outputs (feats[0..3]); x is consumed */
static void resnet_fwd(const ie_net *n, xt x, xt feats[4]) {
    for (int L = 0; L < 4; L++) {
        for (int i = 0; i < n->nlayer[L]; i++) { xt y = bneck_fwd(&n->layer[L][i], x); if (!(L > 0 && i == 0)) xt_free(&x); x = y; }
        feats[L] = x;
    }
}

static int load_v8in(ie_net *n, const wl_dict *W) {
    int bad = 0;
    n->in_w = scal(W, "normalization.weight", &bad); n->in_b = scal(W, "normalization.bias", &bad);
    if (bad || mkconv(&n->stem, W, "backbone.conv1.weight", "backbone.bn1", nullptr, 1, 2, 2, 3, 3, 3)) return -1;
    static const int nb[4] = {3, 4, 6, 3};
    if (load_resnet(n, W, "backbone.", nb)) return -1;
    for (int i = 0; i < 3; i++) {
        char a[100], b[100]; snprintf(a, sizeof a, "decoder.convs.%d.0.weight", i); snprintf(b, sizeof b, "decoder.convs.%d.1", i);
        if (mkconv(&n->dec[i], W, a, b, nullptr, 1, 1, 1, 0, 1, 1)) return -1;
    }
    n->logit_w = devvec(W, "decoder.logit.weight", 256); n->logit_b = scal(W, "decoder.logit.bias", &bad);
    return n->logit_w && !bad ? 0 : -1;
}

static int load_hecate(ie_net *n, const wl_dict *W) {
    int bad = 0; const char *P = "canonical.";
    n->in_w = scal(W, "canonical.normalization.weight", &bad); n->in_b = scal(W, "canonical.normalization.bias", &bad);
    n->in_rm = scal(W, "canonical.normalization.running_mean", &bad); n->in_rv = scal(W, "canonical.normalization.running_var", &bad);
    n->zscale = scal(W, "depth_coordinate_scale", &bad);
    if (bad) return -1;
    if (n->kind == IE_HEC24) { if (mkconv(&n->stem, W, "canonical.backbone.conv1.weight", "canonical.backbone.bn1", nullptr, 1, 2, 2, 3, 3, 3)) return -1; n->margin = 1; n->xys = 4; }
    else { if (mkconv(&n->stem, W, "canonical.backbone.conv1.weight", "canonical.backbone.bn1", nullptr, 1, 1, 1, 1, 1, 1)) return -1; n->margin = 0; n->xys = 1; }
    static const int nb[4] = {3, 8, 36, 3};
    char pre[64]; snprintf(pre, sizeof pre, "%sbackbone.", P);
    if (load_resnet(n, W, pre, nb)) return -1;
    static const int dd[4] = {64, 128, 256, 512};
    for (int i = 0; i < 4; i++) {
        char a[120], g[120], b[120];
        snprintf(a, sizeof a, "canonical.decoder.channel_reduce.%d.0.weight", i); snprintf(g, sizeof g, "canonical.decoder.channel_reduce.%d.1.weight", i); snprintf(b, sizeof b, "canonical.decoder.channel_reduce.%d.1.bias", i);
        if (mkconv(&n->red[i], W, a, nullptr, nullptr, 1, 1, 1, 0, 0, 0) || !(n->red_g[i] = devvec(W, g, dd[i])) || !(n->red_b[i] = devvec(W, b, dd[i]))) return -1;
    }
    for (int j = 0; j < 3; j++) {
        char a[120], b[120]; rcb *R = &n->blk[j]; const int oc = dd[2 - j];
        snprintf(a, sizeof a, "canonical.decoder.decoder_blocks.%d.conv1.weight", j); if (mkconv(&R->c1, W, a, nullptr, nullptr, 1, 1, 1, 1, 1, 1)) return -1;
        snprintf(a, sizeof a, "canonical.decoder.decoder_blocks.%d.conv2.weight", j); if (mkconv(&R->c2, W, a, nullptr, nullptr, 1, 1, 1, 1, 1, 1)) return -1;
        snprintf(a, sizeof a, "canonical.decoder.decoder_blocks.%d.shortcut.weight", j); if (mkconv(&R->sc, W, a, nullptr, nullptr, 1, 1, 1, 0, 0, 0)) return -1;
        snprintf(a, sizeof a, "canonical.decoder.decoder_blocks.%d.gn1.weight", j); snprintf(b, sizeof b, "canonical.decoder.decoder_blocks.%d.gn1.bias", j);
        if (!(R->g1 = devvec(W, a, oc)) || !(R->b1 = devvec(W, b, oc))) return -1;
        snprintf(a, sizeof a, "canonical.decoder.decoder_blocks.%d.gn2.weight", j); snprintf(b, sizeof b, "canonical.decoder.decoder_blocks.%d.gn2.bias", j);
        if (!(R->g2 = devvec(W, a, oc)) || !(R->b2 = devvec(W, b, oc))) return -1;
    }
    n->attn_w = devvec(W, "canonical.decoder.depth_collapse.attn_conv.weight", 64); n->attn_b = scal(W, "canonical.decoder.depth_collapse.attn_conv.bias", &bad);
    n->logit_w = devvec(W, "canonical.decoder.logit.weight", 64); n->logit_b = scal(W, "canonical.decoder.logit.bias", &bad);
    return n->attn_w && n->logit_w && !bad ? 0 : -1;
}

static int load_reader(ie_net *n, const wl_dict *W) {
    int bad = 0;
    if (mkconv(&n->f1, W, "depth_fusion.features.0.weight", nullptr, nullptr, 1, 1, 1, 1, 1, 1) || mkconv(&n->f2, W, "depth_fusion.features.3.weight", nullptr, nullptr, 1, 1, 1, 1, 1, 1)) return -1;
    if (!(n->f1g = devvec(W, "depth_fusion.features.1.weight", 16)) || !(n->f1b = devvec(W, "depth_fusion.features.1.bias", 16)) ||
        !(n->f2g = devvec(W, "depth_fusion.features.4.weight", 16)) || !(n->f2b = devvec(W, "depth_fusion.features.4.bias", 16))) return -1;
    n->att_w = devvec(W, "depth_fusion.attention_logits.weight", 16); n->att_b = scal(W, "depth_fusion.attention_logits.bias", &bad);
    if (!n->att_w || bad) return -1;
    const char *E = "network.shared_encoder.";
    char a[200], b[200], g[200];
    snprintf(a, sizeof a, "%sstem.convs.0.conv.weight", E); snprintf(b, sizeof b, "%sstem.convs.0.conv.bias", E);
    if (mkconv(&n->rstem, W, a, nullptr, b, 1, 1, 1, 0, 1, 1)) return -1;
    const wl_tensor *sw = need(W, a); const int stem_c = (int)sw->shape[0];
    snprintf(g, sizeof g, "%sstem.convs.0.norm.weight", E); snprintf(b, sizeof b, "%sstem.convs.0.norm.bias", E);
    if (!(n->rstem_g = devvec(W, g, stem_c)) || !(n->rstem_b = devvec(W, b, stem_c))) return -1;
    /* stages: as many as present; blocks per stage by probing */
    n->nst = 0; int inc = stem_c;
    for (int s = 0; s < 6; s++) {
        snprintf(a, sizeof a, "%sstages.%d.blocks.0.conv1.conv.weight", E, s); if (!wl_get(W, a)) break;
        int nb = 0; for (;; nb++) { snprintf(a, sizeof a, "%sstages.%d.blocks.%d.conv1.conv.weight", E, s, nb); if (!wl_get(W, a)) break; }
        n->nblk[s] = nb; n->st[s] = calloc(nb, sizeof(bbd)); n->nst = s + 1;
        for (int i = 0; i < nb; i++) {
            bbd *B = &n->st[s][i]; const int stride = (s > 0 && i == 0) ? 2 : 1; B->stride = stride;
            char p[160]; snprintf(p, sizeof p, "%sstages.%d.blocks.%d.", E, s, i);
            snprintf(a, sizeof a, "%sconv1.conv.weight", p); snprintf(b, sizeof b, "%sconv1.conv.bias", p);
            if (mkconv(&B->c1, W, a, nullptr, b, 1, stride, stride, 0, 1, 1)) return -1;
            const int oc = (int)wl_get(W, a)->shape[0];
            snprintf(a, sizeof a, "%sconv2.conv.weight", p); snprintf(b, sizeof b, "%sconv2.conv.bias", p);
            if (mkconv(&B->c2, W, a, nullptr, b, 1, 1, 1, 0, 1, 1)) return -1;
            snprintf(a, sizeof a, "%sconv1.norm.weight", p); snprintf(b, sizeof b, "%sconv1.norm.bias", p); if (!(B->g1 = devvec(W, a, oc)) || !(B->b1 = devvec(W, b, oc))) return -1;
            snprintf(a, sizeof a, "%sconv2.norm.weight", p); snprintf(b, sizeof b, "%sconv2.norm.bias", p); if (!(B->g2 = devvec(W, a, oc)) || !(B->b2 = devvec(W, b, oc))) return -1;
            snprintf(a, sizeof a, "%sskip.1.conv.weight", p); B->has_sk = wl_get(W, a) != nullptr;
            if ((inc != oc) != B->has_sk) { fprintf(stderr, "ink-ext: reader stage %d block %d: skip projection mismatch\n", s, i); return -1; }
            if (B->has_sk) {
                if (mkconv(&B->sk, W, a, nullptr, nullptr, 1, 1, 1, 0, 0, 0)) return -1;
                snprintf(a, sizeof a, "%sskip.1.norm.weight", p); snprintf(b, sizeof b, "%sskip.1.norm.bias", p); if (!(B->gs = devvec(W, a, oc)) || !(B->bs = devvec(W, b, oc))) return -1;
            }
            inc = oc;
        }
    }
    if (n->nst < 2) { fprintf(stderr, "ink-ext: reader: no encoder stages\n"); return -1; }
    for (int s = 0; s < n->nst - 1; s++) {
        const char *D = "network.shared_decoder.";
        snprintf(a, sizeof a, "%stranspconvs.%d.weight", D, s); snprintf(b, sizeof b, "%stranspconvs.%d.bias", D, s);
        const wl_tensor *tw = need(W, a), *tb = need(W, b); if (!tw || !tb) return -1;
        if (xtconv_init(&n->tconv[s], tw->data, (int)tw->shape[0], (int)tw->shape[1], tb->data)) return -1;
        snprintf(a, sizeof a, "%sstages.%d.convs.0.conv.weight", D, s); snprintf(b, sizeof b, "%sstages.%d.convs.0.conv.bias", D, s);
        if (mkconv(&n->dconv[s], W, a, nullptr, b, 1, 1, 1, 0, 1, 1)) return -1;
        const int oc = (int)wl_get(W, a)->shape[0];
        snprintf(a, sizeof a, "%sstages.%d.convs.0.norm.weight", D, s); snprintf(b, sizeof b, "%sstages.%d.convs.0.norm.bias", D, s);
        if (!(n->dg[s] = devvec(W, a, oc)) || !(n->db[s] = devvec(W, b, oc))) return -1;
    }
    const wl_tensor *hw = need(W, "network.task_heads.ink.weight"); if (!hw) return -1;
    n->logit_w = xdev_floats(hw->data, (size_t)hw->numel); n->logit_b = scal(W, "network.task_heads.ink.bias", &bad);
    return bad ? -1 : 0;
}

ie_net *ie_load(int kind, const char *weights) {
    if (kind < 0 || kind >= IE_NMODELS) return nullptr;
    const char *sub = kind == IE_HEC24 || kind == IE_HEC96 ? "ema_model" : ie_is_koine(kind) ? "model" : nullptr;
    wl_dict *W = nullptr;
    if (ie_is_koine(kind)) W = wl_load(weights, "ema_model");   /* koine infer.py prefers the EMA weights when a checkpoint has them */
    if (!W) W = wl_load(weights, sub);
    if (!W) { fprintf(stderr, "ink-ext: %s\n", wl_error()); return nullptr; }
    ie_net *n = calloc(1, sizeof *n); n->kind = kind; n->sp = ie_specs[kind];
    int rc;
    if (kind == IE_V8IN) rc = load_v8in(n, W);
    else if (ie_is_koine(kind)) rc = load_reader(n, W);
    else {
        const double um = wl_num(W, "sampling_um", 0);
        if (um > 0 && fabs(um - (kind == IE_HEC24 ? 2.4 : 9.6)) > 0.05) { fprintf(stderr, "ink-ext: %s is a %.1f um Hecate checkpoint\n", weights, um); rc = -1; }
        else rc = load_hecate(n, W);
    }
    wl_free(W);
    if (rc || xnet_check("load")) { fprintf(stderr, "ink-ext: cannot build %s from %s\n", n->sp.name, weights); free(n); return nullptr; }
    xnet_sync();
    return n;
}
void ie_free(ie_net *n) { if (n) free(n); }   /* device memory stays in the xnet cache until exit */

size_t ie_tile_bytes_gpu(int kind) {
    switch (kind) {
    case IE_V8IN: return (size_t)700 << 20;
    case IE_HEC24: return (size_t)700 << 20;
    case IE_HEC96: return (size_t)40 << 20;
    case IE_INK9UM: return (size_t)56 << 20;
    default: return (size_t)450 << 20;
    }
}

/* ---- host-side input statistics ---- */
static void hist_u8(const uint8_t *x, size_t n, uint64_t h[256]) { memset(h, 0, 256 * sizeof *h); for (size_t i = 0; i < n; i++) h[x[i]]++; }
static double order_stat(const uint64_t h[256], uint64_t k) {   /* k-th smallest (0-based) */
    uint64_t c = 0; for (int v = 0; v < 256; v++) { c += h[v]; if (c > k) return v; }
    return 255;
}
static double np_percentile_hist(const uint64_t h[256], uint64_t n, double q) {   /* numpy linear method */
    const double idx = q / 100.0 * (double)(n - 1); const uint64_t i0 = (uint64_t)floor(idx); const double f = idx - (double)i0;
    const double a = order_stat(h, i0), b = i0 + 1 < n ? order_stat(h, i0 + 1) : a;
    return a + (b - a) * f;
}
typedef struct { double v; uint64_t c; } vc;
static int vc_cmp(const void *a, const void *b) { const double x = ((const vc *)a)->v, y = ((const vc *)b)->v; return (x > y) - (x < y); }
static double median_vc(vc *a, int m, uint64_t n) {   /* np.median of a multiset (n even: mean of the middle two) */
    qsort(a, (size_t)m, sizeof *a, vc_cmp);
    const uint64_t k0 = (n - 1) / 2, k1 = n / 2; double r0 = 0, r1 = 0; uint64_t c = 0;
    for (int i = 0; i < m; i++) { const uint64_t c2 = c + a[i].c; if (k0 >= c && k0 < c2) r0 = a[i].v; if (k1 >= c && k1 < c2) { r1 = a[i].v; break; } c = c2; }
    return 0.5 * (r0 + r1);
}
void ie_robust_params(const uint8_t *x, size_t n, float *lo, float *hi, float *a, float *b) {
    uint64_t h[256]; hist_u8(x, n, h);
    double s = 0, s2 = 0; int vmin = 255, vmax = 0;
    for (int v = 0; v < 256; v++) if (h[v]) { s += (double)v * h[v]; s2 += (double)v * v * h[v]; if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
    const double mean = s / n, std = sqrt(fmax(0, s2 / n - mean * mean));
    const double l = np_percentile_hist(h, n, 1.0), u = np_percentile_hist(h, n, 99.0);
    vc vals[260]; int m = 0;
    uint64_t below = 0, above = 0; for (int v = 0; v < 256; v++) { if (v <= l) below += h[v]; else if (v >= u) above += h[v]; }
    if (below) vals[m++] = (vc){l, below};
    for (int v = 0; v < 256; v++) if (h[v] && v > l && v < u) vals[m++] = (vc){v, h[v]};
    if (above) vals[m++] = (vc){u, above};
    const double med = median_vc(vals, m, n);
    for (int i = 0; i < m; i++) vals[i].v = fabs(vals[i].v - med);
    const double mad = median_vc(vals, m, n);
    double sm = 1.4826 * mad;
    if (!(sm >= 1e-6)) {   /* normalize_robust fallbacks: pre-clip std, else half the percentile span, else 1 */
        if (std >= 1e-6) sm = std; else if (fabs(u - l) / 2 >= 1e-6) sm = fabs(u - l) / 2; else sm = 1;
    }
    *lo = (float)l; *hi = (float)u; *a = (float)(1.0 / sm); *b = (float)(-med / sm);
    (void)vmin; (void)vmax;
}

/* ---- forwards ---- */
static xt upload_tiles(const uint8_t *tiles, int B, int D, int H, int W, uint8_t **dev) {
    const size_t bytes = (size_t)B * D * H * W; *dev = xmalloc_dev(bytes);
    xh2d(*dev, tiles, bytes);
    return (xt){0};
}

static void v8in_fwd(const ie_net *n, const uint8_t *tiles, int B, float *out) {
    const int D = 24, T = 64; const size_t per = (size_t)D * T * T;
    float lo[B], hi[B], a[B], b[B];
    for (int i = 0; i < B; i++) {   /* InstanceNorm3d(1) of clip(x, 0, 200) / 255: exact statistics from the histogram */
        uint64_t h[256]; hist_u8(tiles + i * per, per, h); double s = 0, s2 = 0;
        for (int v = 0; v < 256; v++) { const double f = (v > 200 ? 200 : v) / 255.0; s += f * h[v]; s2 += f * f * h[v]; }
        const double m = s / per, var = fmax(0, s2 / per - m * m), r = 1.0 / sqrt(var + 1e-5);
        lo[i] = 0; hi[i] = 200; a[i] = (float)(r * n->in_w / 255.0); b[i] = (float)(n->in_b - m * r * n->in_w);
    }
    uint8_t *dev; upload_tiles(tiles, B, D, T, T, &dev);
    xt x0 = xinput_u8(dev, B, D, T, T, lo, hi, a, b); xfree_dev(dev);
    xt x1 = xresize(x0, 96, 256, 256); xt_free(&x0);
    xt s = xconv_fwd(&n->stem, x1, nullptr, XA_RELU); xt_free(&x1);
    xt p = xmaxpool2d_k3s2(s); xt_free(&s);
    xt f[4]; resnet_fwd(n, p, f);
    xt m[4]; for (int i = 0; i < 4; i++) { m[i] = xdepthmax(f[i]); xt_free(&f[i]); }
    for (int i = 3; i > 0; i--) {
        xt up = xresize(m[i], 1, 2 * m[i].h, 2 * m[i].w); xt_free(&m[i]);
        xt c = xconcat(m[i - 1], up); xt_free(&up); xt_free(&m[i - 1]);
        m[i - 1] = xconv_fwd(&n->dec[i - 1], c, nullptr, XA_RELU); xt_free(&c);
    }
    float *lg = xmalloc_dev((size_t)B * T * T * 4);
    xhead(m[0], n->logit_w, n->logit_b, lg); xt_free(&m[0]);
    xd2h(out, lg, (size_t)B * T * T * 4); xfree_dev(lg);
}

static xt rcb_fwd(const rcb *R, xt x) {
    xt id = xconv_fwd(&R->sc, x, nullptr, XA_NONE);
    xt h = xconv_fwd(&R->c1, x, nullptr, XA_NONE); xgn(h, R->c1.co >= 32 ? 32 : R->c1.co, R->g1, R->b1, 1e-5f, nullptr, XA_RELU, h);
    xt h2 = xconv_fwd(&R->c2, h, nullptr, XA_NONE); xt_free(&h);
    xgn(h2, R->c2.co >= 32 ? 32 : R->c2.co, R->g2, R->b2, 1e-5f, &id, XA_RELU, h2); xt_free(&id);
    return h2;
}
/* out: logits on the T x T grid (logit of the bilinearly upsampled probability) or probabilities */
static void hecate_fwd(const ie_net *n, const uint8_t *tiles, int B, float *out, int logits) {
    const int D = n->sp.D, T = n->sp.T, mg = n->margin, Dn = D - 2 * mg, s = n->xys, F = T / s; const size_t per = (size_t)D * T * T;
    /* the columns with data (raw.any over the full depth) at feature resolution (max-pooled over s x s) */
    uint8_t *sup = calloc((size_t)B * F * F, 1), *full = calloc((size_t)B * T * T, 1);
    for (int i = 0; i < B; i++) for (int d = 0; d < D; d++) for (int y = 0; y < T; y++) for (int x = 0; x < T; x++)
        if (tiles[i * per + ((size_t)d * T + y) * T + x]) { full[((size_t)i * T + y) * T + x] = 1; sup[((size_t)i * F + y / s) * F + x / s] = 1; }
    uint8_t *crop = malloc((size_t)B * Dn * T * T);
    for (int i = 0; i < B; i++) memcpy(crop + (size_t)i * Dn * T * T, tiles + i * per + (size_t)mg * T * T, (size_t)Dn * T * T);
    float lo[B], hi[B], a[B], b[B]; const double div = n->kind == IE_HEC24 ? 200.0 : 255.0, r = 1.0 / sqrt((double)n->in_rv + 1e-5);
    for (int i = 0; i < B; i++) { lo[i] = 0; hi[i] = 255; a[i] = (float)(r * n->in_w / div); b[i] = (float)(n->in_b - n->in_rm * r * n->in_w); }
    uint8_t *dev; upload_tiles(crop, B, Dn, T, T, &dev); free(crop);
    xt x0 = xinput_u8(dev, B, Dn, T, T, lo, hi, a, b); xfree_dev(dev);
    xt st = xconv_fwd(&n->stem, x0, nullptr, XA_RELU); xt_free(&x0);
    if (n->kind == IE_HEC24) { xt p = xmaxpool2d_k3s2(st); xt_free(&st); st = p; }
    xt f[4]; resnet_fwd(n, st, f);
    xt r4[4];
    for (int i = 0; i < 4; i++) {
        r4[i] = xconv_fwd(&n->red[i], f[i], nullptr, XA_NONE); xt_free(&f[i]);
        xgn(r4[i], 32, n->red_g[i], n->red_b[i], 1e-5f, nullptr, XA_RELU, r4[i]);
    }
    xt x = r4[3];
    for (int j = 0; j < 3; j++) {
        xt sk = r4[2 - j]; xt up = xresize(x, sk.d, sk.h, sk.w); xt_free(&x);
        xt c = xconcat(up, sk); xt_free(&up); xt_free(&r4[2 - j]);
        x = rcb_fwd(&n->blk[j], c); xt_free(&c);
    }
    const size_t nv = (size_t)x.n * x.d * x.h * x.w;
    float *sc = xmalloc_dev(nv * 4), *lg = xmalloc_dev(nv * 4), *o2 = xmalloc_dev((size_t)B * x.h * x.w * 4); uint8_t *sd = xmalloc_dev((size_t)B * F * F);
    xhead(x, n->attn_w, n->attn_b, sc); xhead(x, n->logit_w, n->logit_b, lg); xt_free(&x);
    xh2d(sd, sup, (size_t)B * F * F);
    xhecate_collapse(sc, lg, sd, B, (int)(nv / ((size_t)B * F * F)), F, F, n->zscale, o2);
    float *h2 = malloc((size_t)B * F * F * 4); xd2h(h2, o2, (size_t)B * F * F * 4);
    xfree_dev(sc); xfree_dev(lg); xfree_dev(o2); xfree_dev(sd);
    /* sigmoid, bilinear (align_corners False) F -> T, logit of the clamped probability */
    const float eps = 1.1920929e-07f;
    for (int i = 0; i < B; i++) {
        const float *g = h2 + (size_t)i * F * F; float *o = out + (size_t)i * T * T;
        for (int y = 0; y < T; y++) {
            float sy = F == T ? (float)y : ((float)F / T) * (y + 0.5f) - 0.5f; if (sy < 0) sy = 0;
            const int y0 = (int)sy, y1 = y0 < F - 1 ? y0 + 1 : y0; const float ly = sy - y0;
            for (int xx = 0; xx < T; xx++) {
                float sx = F == T ? (float)xx : ((float)F / T) * (xx + 0.5f) - 0.5f; if (sx < 0) sx = 0;
                const int x0 = (int)sx, x1 = x0 < F - 1 ? x0 + 1 : x0; const float lx = sx - x0;
                const float p00 = 1 / (1 + expf(-g[y0 * F + x0])), p01 = 1 / (1 + expf(-g[y0 * F + x1])), p10 = 1 / (1 + expf(-g[y1 * F + x0])), p11 = 1 / (1 + expf(-g[y1 * F + x1]));
                float p = (1 - ly) * ((1 - lx) * p00 + lx * p01) + ly * ((1 - lx) * p10 + lx * p11);
                if (logits) { p = p < eps ? eps : p > 1 - eps ? 1 - eps : p; p = logf(p / (1 - p)); }
                o[(size_t)y * T + xx] = p;
            }
        }
    }
    free(h2); free(sup); free(full);
}

static xt bbd_fwd(const bbd *B, xt x) {
    xt h = xconv_fwd(&B->c1, x, nullptr, XA_NONE); xgn(h, h.c, B->g1, B->b1, 1e-5f, nullptr, XA_LRELU, h);
    xt h2 = xconv_fwd(&B->c2, h, nullptr, XA_NONE); xt_free(&h);
    xt sk = x; int own = 0;
    if (B->stride != 1) { sk = xavgpool2d_k2(x); own = 1; }
    if (B->has_sk) { xt p = xconv_fwd(&B->sk, sk, nullptr, XA_NONE); if (own) xt_free(&sk); xgn(p, p.c, B->gs, B->bs, 1e-5f, nullptr, XA_NONE, p); sk = p; own = 1; }
    xgn(h2, h2.c, B->g2, B->b2, 1e-5f, &sk, XA_LRELU, h2);
    if (own) xt_free(&sk);
    return h2;
}
static void reader_fwd(const ie_net *n, const uint8_t *tiles, int B, float *out) {
    const int D = n->sp.D, T = n->sp.T; const size_t per = (size_t)D * T * T;
    float lo[B], hi[B], a[B], b[B];
    for (int i = 0; i < B; i++) ie_robust_params(tiles + i * per, per, &lo[i], &hi[i], &a[i], &b[i]);
    uint8_t *dev; upload_tiles(tiles, B, D, T, T, &dev);
    xt x0 = xinput_u8(dev, B, D, T, T, lo, hi, a, b); xfree_dev(dev);
    xt f1 = xconv_fwd(&n->f1, x0, nullptr, XA_NONE); xt_free(&x0); xgn(f1, f1.c, n->f1g, n->f1b, 1e-5f, nullptr, XA_LRELU, f1);
    xt f2 = xconv_fwd(&n->f2, f1, nullptr, XA_NONE); xt_free(&f1); xgn(f2, f2.c, n->f2g, n->f2b, 1e-5f, nullptr, XA_LRELU, f2);
    float *at = xmalloc_dev((size_t)B * D * T * T * 4); xhead(f2, n->att_w, n->att_b, at);
    xt z = xdepth_attn_max(f2, at); xt_free(&f2); xfree_dev(at);
    xt x = xconv_fwd(&n->rstem, z, nullptr, XA_NONE); xt_free(&z); xgn(x, x.c, n->rstem_g, n->rstem_b, 1e-5f, nullptr, XA_LRELU, x);
    xt skip[6];
    for (int s = 0; s < n->nst; s++) {
        for (int i = 0; i < n->nblk[s]; i++) { xt y = bbd_fwd(&n->st[s][i], x); if (!(s > 0 && i == 0)) xt_free(&x); x = y; }
        skip[s] = x;
    }
    xt l = skip[n->nst - 1];
    for (int s = 0; s < n->nst - 1; s++) {
        xt up = xconv_fwd(&n->tconv[s], l, nullptr, XA_NONE); xt_free(&l);
        xt c = xconcat(up, skip[n->nst - 2 - s]); xt_free(&up); xt_free(&skip[n->nst - 2 - s]);
        l = xconv_fwd(&n->dconv[s], c, nullptr, XA_NONE); xt_free(&c);
        xgn(l, l.c, n->dg[s], n->db[s], 1e-5f, nullptr, XA_LRELU, l);
    }
    float *lg = xmalloc_dev((size_t)B * T * T * 4);
    xhead(l, n->logit_w, n->logit_b, lg); xt_free(&l);
    xd2h(out, lg, (size_t)B * T * T * 4); xfree_dev(lg);
}

int ie_forward(ie_net *n, const uint8_t *tiles, int B, float *out, int logits) {
    const size_t TT = (size_t)n->sp.T * n->sp.T;
    if (n->kind == IE_V8IN) v8in_fwd(n, tiles, B, out);
    else if (ie_is_koine(n->kind)) reader_fwd(n, tiles, B, out);
    else hecate_fwd(n, tiles, B, out, logits);
    if (xnet_check("forward")) return -1;
    if (!logits && (n->kind == IE_V8IN || ie_is_koine(n->kind))) for (size_t i = 0; i < (size_t)B * TT; i++) out[i] = 1.f / (1.f + expf(-out[i]));
    return 0;
}
