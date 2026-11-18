/* ufsm ink-ext networks (inkext.h): v8in (ResNet3D-50 + 2D decoder), Hecate (ResNet-152 3D + 3D U-Net decoder + depth
   attention), Reader V3 (3D depth-fusion stem + 2D residual U-Net) on xnet. Architectures follow the published code:
   v8in ink8um/modeling.py, Hecate hecate.py, Reader V3 villa ink-detection koine_machines (hybrid_3d2d.py) + vesuvius
   build_network_from_config (merge-ink-pipelines). */
#include "inkext.h"
#include "wload.h"
#include "xnet.h"
#include "xtsf.h"
#include "dinovol.h"
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
    /* ink-sdk profiles (src/ink_sdk/profiles/ink_9um-*.toml): the spacing is each profile's scale_um, the tile its checkpoint's patch_size */
    {"reader2", 17, 128, 9.596, 64, 8},   /* domenicor046/reader-v2 step 40000 (9.6 um) */
    {"dense9", 17, 128, 9.362, 64, 8},    /* Nieuwlaar/ink9um-dense-native step 16000 (native 9.362 um; weights/dense_native_016000.safetensors) */
    {"rectina", 17, 384, 9.362, 192, 8},  /* evilaliv3/rectina m4 ckpt_020000 (9.362 um, 384 px; CC-BY-NC-4.0) */
    {"trace3b", 17, 128, 9.362, 64, 8},   /* Jashann TRACE v3B_best (9.362 um, tolerance 0.1; CC-BY-NC-SA-4.0) */
    {"canon2", 62, 256, 2.399, 128, 31},  /* canonical_2um r152_3ddec_v2_l5_epoch13 (Lightning .ckpt state_dict; divide 255, no clip) */
    {"studio", 16, 128, 9.362, 64, 7},    /* studio-softloss-20261023-ema (CC BY-NC 4.0 weights): 128 px windows, the central 96 written, Hann stride 64 */
    {"tsf", 26, 64, 7.91, 16, 13},        /* scrollprize TimeSformers (GP / large scroll1, scroll5): 4 x 4 logits per 64 px tile, stride 16, hann */
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
    int gnorm; float g_lo, g_hi, g_a, g_b;   /* Reader: one robust normalisation for every tile (ie_set_norm, ink-ext --norm sheet) */
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
    /* Studio DeepResidualUNet: widths 48 96 192 384 640, encoder blocks 2 3 4 5 6, decoder blocks 3 3 2 2 */
    struct srb { xconv c1, c2; float *g1, *b1, *g2, *b2; } *senc[5], *sdec[4];
    int nenc[5], ndec[4];
    xconv sdown[4], sup[4], sfuse[4]; float *sdg[4], *sdb[4], *sfg[4], *sfb[4], *shg, *shb; float sdw[16];
    /* TimeSformer: depth layers of (time attention, space attention, GEGLU feed-forward), each pre-LayerNorm */
    struct tsl { xconv qkv[2], ao[2], f0, f3; float *lg[3], *lb[3]; } *tl; int tdepth, theads;
    xconv tpe; float *tcls, *trs[2], *trc[2]; float *thw, *thb, *tow, *tob;   /* final LN + Linear(512, 16) on the host */
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

/* Hecate (keys under "canonical.") and canonical_2um (the same RegressionModel at the top level: backbone. decoder. normalization.) */
static int load_hecate(ie_net *n, const wl_dict *W) {
    int bad = 0; const char *P = n->kind == IE_CANON2 ? "" : "canonical."; char k[200];
    if (n->kind == IE_CANON2 && !wl_get(W, "normalization.weight")) { n->in_w = 1; n->in_b = 0; n->in_rm = 0; n->in_rv = 1 - 1e-5f; }   /* with_norm False: identity */
    else {
        snprintf(k, sizeof k, "%snormalization.weight", P); n->in_w = scal(W, k, &bad); snprintf(k, sizeof k, "%snormalization.bias", P); n->in_b = scal(W, k, &bad);
        snprintf(k, sizeof k, "%snormalization.running_mean", P); n->in_rm = scal(W, k, &bad); snprintf(k, sizeof k, "%snormalization.running_var", P); n->in_rv = scal(W, k, &bad);
    }
    n->zscale = n->kind == IE_CANON2 ? 0.f : scal(W, "depth_coordinate_scale", &bad);
    if (bad) return -1;
    char cw[120], cb[120]; snprintf(cw, sizeof cw, "%sbackbone.conv1.weight", P); snprintf(cb, sizeof cb, "%sbackbone.bn1", P);
    if (n->kind == IE_HEC24 || n->kind == IE_CANON2) { if (mkconv(&n->stem, W, cw, cb, nullptr, 1, 2, 2, 3, 3, 3)) return -1; n->margin = n->kind == IE_HEC24; n->xys = 4; }
    else { if (mkconv(&n->stem, W, cw, cb, nullptr, 1, 1, 1, 1, 1, 1)) return -1; n->margin = 0; n->xys = 1; }
    static const int nb[4] = {3, 8, 36, 3};
    char pre[64]; snprintf(pre, sizeof pre, "%sbackbone.", P);
    if (n->kind == IE_CANON2) {   /* the Lightning checkpoint may prefix model. / module. (the loader strips them): refuse anything else than the plain names */
        snprintf(k, sizeof k, "%slayer3.35.conv3.weight", pre); if (!wl_get(W, k)) { fprintf(stderr, "ink-ext: canon2: no %s (not a ResNet-152 canonical checkpoint)\n", k); return -1; }
    }
    if (load_resnet(n, W, pre, nb)) return -1;
    static const int dd[4] = {64, 128, 256, 512};
    for (int i = 0; i < 4; i++) {
        char a[120], g[120], b[120];
        snprintf(a, sizeof a, "%sdecoder.channel_reduce.%d.0.weight", P, i); snprintf(g, sizeof g, "%sdecoder.channel_reduce.%d.1.weight", P, i); snprintf(b, sizeof b, "%sdecoder.channel_reduce.%d.1.bias", P, i);
        if (mkconv(&n->red[i], W, a, nullptr, nullptr, 1, 1, 1, 0, 0, 0) || !(n->red_g[i] = devvec(W, g, dd[i])) || !(n->red_b[i] = devvec(W, b, dd[i]))) return -1;
    }
    for (int j = 0; j < 3; j++) {
        char a[120], b[120]; rcb *R = &n->blk[j]; const int oc = dd[2 - j];
        snprintf(a, sizeof a, "%sdecoder.decoder_blocks.%d.conv1.weight", P, j); if (mkconv(&R->c1, W, a, nullptr, nullptr, 1, 1, 1, 1, 1, 1)) return -1;
        snprintf(a, sizeof a, "%sdecoder.decoder_blocks.%d.conv2.weight", P, j); if (mkconv(&R->c2, W, a, nullptr, nullptr, 1, 1, 1, 1, 1, 1)) return -1;
        snprintf(a, sizeof a, "%sdecoder.decoder_blocks.%d.shortcut.weight", P, j); if (mkconv(&R->sc, W, a, nullptr, nullptr, 1, 1, 1, 0, 0, 0)) return -1;
        snprintf(a, sizeof a, "%sdecoder.decoder_blocks.%d.gn1.weight", P, j); snprintf(b, sizeof b, "%sdecoder.decoder_blocks.%d.gn1.bias", P, j);
        if (!(R->g1 = devvec(W, a, oc)) || !(R->b1 = devvec(W, b, oc))) return -1;
        snprintf(a, sizeof a, "%sdecoder.decoder_blocks.%d.gn2.weight", P, j); snprintf(b, sizeof b, "%sdecoder.decoder_blocks.%d.gn2.bias", P, j);
        if (!(R->g2 = devvec(W, a, oc)) || !(R->b2 = devvec(W, b, oc))) return -1;
    }
    snprintf(k, sizeof k, "%sdecoder.depth_collapse.attn_conv.weight", P); n->attn_w = devvec(W, k, 64); snprintf(k, sizeof k, "%sdecoder.depth_collapse.attn_conv.bias", P); n->attn_b = scal(W, k, &bad);
    snprintf(k, sizeof k, "%sdecoder.logit.weight", P); n->logit_w = devvec(W, k, 64); snprintf(k, sizeof k, "%sdecoder.logit.bias", P); n->logit_b = scal(W, k, &bad);
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

static xt upload_tiles(const uint8_t *tiles, int B, int D, int H, int W, uint8_t **dev);
static void up_prob(const float *h2, int B, int F, int T, int logits, float *out);
/* ---- Studio (vesuvius-studio-multiscroll-cpt hf/model.py DeepResidualUNet(deeper=False)) ---- */
static const int st_w[5] = {48, 96, 192, 384, 640}, st_ne[5] = {2, 3, 4, 5, 6}, st_nd[4] = {3, 3, 2, 2};
static int st_groups(int c) { int a = 32, b = c; while (b) { const int t = a % b; a = b; b = t; } return a; }   /* GroupNorm(gcd(32, c), c) */
static int load_srb(struct srb *R, const wl_dict *W, const char *p, int c) {
    char a[160], b[160];
    snprintf(a, sizeof a, "%s.branch.2.weight", p); if (mkconv(&R->c1, W, a, nullptr, nullptr, 1, 1, 1, 1, 1, 1)) return -1;
    snprintf(a, sizeof a, "%s.branch.6.weight", p); if (mkconv(&R->c2, W, a, nullptr, nullptr, 1, 1, 1, 1, 1, 1)) return -1;
    snprintf(a, sizeof a, "%s.branch.0.weight", p); snprintf(b, sizeof b, "%s.branch.0.bias", p); if (!(R->g1 = devvec(W, a, c)) || !(R->b1 = devvec(W, b, c))) return -1;
    snprintf(a, sizeof a, "%s.branch.3.weight", p); snprintf(b, sizeof b, "%s.branch.3.bias", p); if (!(R->g2 = devvec(W, a, c)) || !(R->b2 = devvec(W, b, c))) return -1;
    return 0;
}
static int load_studio(ie_net *n, const wl_dict *W) {
    char a[160], b[160];
    if (mkconv(&n->stem, W, "stem.weight", nullptr, nullptr, 1, 1, 1, 1, 1, 1)) return -1;
    for (int i = 0; i < 5; i++) {
        n->nenc[i] = st_ne[i]; n->senc[i] = calloc(st_ne[i], sizeof(struct srb));
        for (int j = 0; j < st_ne[i]; j++) { snprintf(a, sizeof a, "enc.%d.%d", i, j); if (load_srb(&n->senc[i][j], W, a, st_w[i])) return -1; }
        snprintf(a, sizeof a, "enc.%d.%d.branch.2.weight", i, st_ne[i]); if (wl_get(W, a)) { fprintf(stderr, "ink-ext: studio: more encoder blocks than DeepResidualUNet(deeper=False)\n"); return -1; }
    }
    for (int i = 0; i < 4; i++) {
        snprintf(a, sizeof a, "down.%d.2.weight", i); if (mkconv(&n->sdown[i], W, a, nullptr, nullptr, i ? 2 : 1, 2, 2, 1, 1, 1)) return -1;
        snprintf(a, sizeof a, "down.%d.0.weight", i); snprintf(b, sizeof b, "down.%d.0.bias", i); if (!(n->sdg[i] = devvec(W, a, st_w[i])) || !(n->sdb[i] = devvec(W, b, st_w[i]))) return -1;
        const int c = st_w[3 - i];   /* decoder stage k works at width w[3 - k] */
        snprintf(a, sizeof a, "up.%d.weight", i); if (mkconv(&n->sup[i], W, a, nullptr, nullptr, 1, 1, 1, 0, 0, 0)) return -1;
        snprintf(a, sizeof a, "fuse.%d.2.weight", i); if (mkconv(&n->sfuse[i], W, a, nullptr, nullptr, 1, 1, 1, 0, 0, 0)) return -1;
        snprintf(a, sizeof a, "fuse.%d.0.weight", i); snprintf(b, sizeof b, "fuse.%d.0.bias", i); if (!(n->sfg[i] = devvec(W, a, 2 * c)) || !(n->sfb[i] = devvec(W, b, 2 * c))) return -1;
        n->ndec[i] = st_nd[i]; n->sdec[i] = calloc(st_nd[i], sizeof(struct srb));
        for (int j = 0; j < st_nd[i]; j++) { snprintf(a, sizeof a, "dec.%d.%d", i, j); if (load_srb(&n->sdec[i][j], W, a, c)) return -1; }
    }
    if (!(n->shg = devvec(W, "head.0.weight", 48)) || !(n->shb = devvec(W, "head.0.bias", 48)) || !(n->logit_w = devvec(W, "head.2.weight", 48))) return -1;
    int bad = 0; n->logit_b = scal(W, "head.2.bias", &bad);
    const wl_tensor *dl = need(W, "depth_logits"); if (!dl || dl->numel != 16 || bad) return -1;
    double mx = -1e300, sm = 0; for (int d = 0; d < 16; d++) mx = fmax(mx, dl->data[d]);
    for (int d = 0; d < 16; d++) sm += exp(dl->data[d] - mx);
    for (int d = 0; d < 16; d++) n->sdw[d] = (float)(exp(dl->data[d] - mx) / sm);   /* depth_logits.softmax(0) */
    return 0;
}
static xt srb_fwd(const struct srb *R, xt x) {   /* x + conv(silu(gn(conv(silu(gn(x)))))) */
    xt h = xt_new(x.n, x.d, x.h, x.w, x.c); xgn(x, st_groups(x.c), R->g1, R->b1, 1e-5f, nullptr, XA_SILU, h);
    xt h1 = xconv_fwd(&R->c1, h, nullptr, XA_NONE); xt_free(&h);
    xgn(h1, st_groups(h1.c), R->g2, R->b2, 1e-5f, nullptr, XA_SILU, h1);
    xt y = xconv_fwd(&R->c2, h1, &x, XA_NONE); xt_free(&h1);
    return y;
}
static void studio_fwd(const ie_net *n, const uint8_t *tiles, int B, float *out) {
    const int D = n->sp.D, T = n->sp.T;
    float lo[B], hi[B], a[B], b[B]; for (int i = 0; i < B; i++) { lo[i] = 0; hi[i] = 200; a[i] = 1.f / 255; b[i] = 0; }   /* clamp(0, 200) / 255 */
    uint8_t *dev; upload_tiles(tiles, B, D, T, T, &dev);
    xt x0 = xinput_u8(dev, B, D, T, T, lo, hi, a, b); xfree_dev(dev);
    xt x = xconv_fwd(&n->stem, x0, nullptr, XA_NONE); xt_free(&x0);
    xt skip[4];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < n->nenc[i]; j++) { xt y = srb_fwd(&n->senc[i][j], x); xt_free(&x); x = y; }
        if (i < 4) {
            skip[i] = x;
            xt h = xt_new(x.n, x.d, x.h, x.w, x.c); xgn(x, st_groups(x.c), n->sdg[i], n->sdb[i], 1e-5f, nullptr, XA_SILU, h);
            x = xconv_fwd(&n->sdown[i], h, nullptr, XA_NONE); xt_free(&h);
        }
    }
    for (int k = 0; k < 4; k++) {
        xt sk = skip[3 - k];
        /* up(interpolate(x)): the bias-free 1x1 conv commutes with the (linear, per-channel) trilinear resize, so it runs first at the low resolution */
        xt u = xconv_fwd(&n->sup[k], x, nullptr, XA_NONE); xt_free(&x);
        xt r = xresize(u, sk.d, sk.h, sk.w); xt_free(&u);
        xt c = xconcat(r, sk); xt_free(&r); xt_free(&skip[3 - k]);
        xgn(c, st_groups(c.c), n->sfg[k], n->sfb[k], 1e-5f, nullptr, XA_SILU, c);
        x = xconv_fwd(&n->sfuse[k], c, nullptr, XA_NONE); xt_free(&c);
        for (int j = 0; j < n->ndec[k]; j++) { xt y = srb_fwd(&n->sdec[k][j], x); xt_free(&x); x = y; }
    }
    xgn(x, st_groups(x.c), n->shg, n->shb, 1e-5f, nullptr, XA_SILU, x);
    const size_t TT = (size_t)T * T; float *lg = xmalloc_dev((size_t)B * D * TT * 4);
    xhead(x, n->logit_w, n->logit_b, lg); xt_free(&x);
    float *h3 = malloc((size_t)B * D * TT * 4); xd2h(h3, lg, (size_t)B * D * TT * 4); xfree_dev(lg);
    for (int i = 0; i < B; i++) for (size_t p = 0; p < TT; p++) {   /* (head(x) * softmax(depth_logits)).sum(depth) */
        double s = 0; for (int d = 0; d < D; d++) s += (double)n->sdw[d] * h3[((size_t)i * D + d) * TT + p];
        out[(size_t)i * TT + p] = (float)s;
    }
    free(h3);
}

/* ---- TimeSformer (lucidrains TimeSformer-pytorch 0.4.1, dim 512, 26 frames of 64 px, 16 px patches, rotary, num_classes 16) ---- */
static int lin(xconv *c, const wl_dict *W, const char *wname, const char *bname) {   /* nn.Linear as a 1x1x1 conv over the token rows */
    const wl_tensor *w = need(W, wname); if (!w || w->ndim != 2) return -1;
    const float *b = nullptr; if (bname) { const wl_tensor *t = need(W, bname); if (!t) return -1; b = t->data; }
    return xconv_init(c, w->data, (int)w->shape[0], (int)w->shape[1], 1, 1, 1, 1, 1, 1, 0, 0, 0, nullptr, b);
}
static float *hostvec(const wl_dict *W, const char *name, int n) {
    const wl_tensor *t = need(W, name); if (!t || t->numel != n) return nullptr;
    float *v = malloc((size_t)n * sizeof *v); memcpy(v, t->data, (size_t)n * sizeof *v); return v;
}
static int load_tsf(ie_net *n, const wl_dict *W) {
    char P[64] = "", a[200], b[200];   /* key prefix: "backbone." (safetensors) or "model.backbone." (Lightning) */
    for (int i = 0; i < wl_count(W); i++) { const char *k = wl_at(W, i)->name; const char *e = strstr(k, "cls_token"); if (e && !e[9] && e - k < 60) { memcpy(P, k, (size_t)(e - k)); P[e - k] = 0; break; } }
    snprintf(a, sizeof a, "%scls_token", P); const wl_tensor *cls = wl_get(W, a);
    if (!cls || cls->numel != 512) { fprintf(stderr, "ink-ext: tsf: no 512-wide cls_token\n"); return -1; }
    n->tcls = xdev_floats(cls->data, 512);
    snprintf(a, sizeof a, "%sto_patch_embedding.weight", P); snprintf(b, sizeof b, "%sto_patch_embedding.bias", P); if (lin(&n->tpe, W, a, b)) return -1;
    if (n->tpe.ci != 256) { fprintf(stderr, "ink-ext: tsf: patch dim %d, expected 256 (16 x 16 x 1)\n", n->tpe.ci); return -1; }
    n->tdepth = 0; for (;; n->tdepth++) { snprintf(a, sizeof a, "%slayers.%d.0.fn.to_qkv.weight", P, n->tdepth); if (!wl_get(W, a)) break; }
    snprintf(a, sizeof a, "%slayers.0.0.fn.to_qkv.weight", P); const wl_tensor *q = wl_get(W, a);
    if (!n->tdepth || !q || q->shape[0] % (3 * 64)) { fprintf(stderr, "ink-ext: tsf: no attention layers (dim_head 64)\n"); return -1; }
    n->theads = (int)(q->shape[0] / (3 * 64));
    n->tl = calloc((size_t)n->tdepth, sizeof *n->tl);
    for (int L = 0; L < n->tdepth; L++) {
        struct tsl *T = &n->tl[L];
        for (int k = 0; k < 2; k++) {
            snprintf(a, sizeof a, "%slayers.%d.%d.fn.to_qkv.weight", P, L, k); if (lin(&T->qkv[k], W, a, nullptr)) return -1;
            snprintf(a, sizeof a, "%slayers.%d.%d.fn.to_out.0.weight", P, L, k); snprintf(b, sizeof b, "%slayers.%d.%d.fn.to_out.0.bias", P, L, k); if (lin(&T->ao[k], W, a, b)) return -1;
        }
        snprintf(a, sizeof a, "%slayers.%d.2.fn.net.0.weight", P, L); snprintf(b, sizeof b, "%slayers.%d.2.fn.net.0.bias", P, L); if (lin(&T->f0, W, a, b)) return -1;
        snprintf(a, sizeof a, "%slayers.%d.2.fn.net.3.weight", P, L); snprintf(b, sizeof b, "%slayers.%d.2.fn.net.3.bias", P, L); if (lin(&T->f3, W, a, b)) return -1;
        for (int k = 0; k < 3; k++) {
            snprintf(a, sizeof a, "%slayers.%d.%d.norm.weight", P, L, k); snprintf(b, sizeof b, "%slayers.%d.%d.norm.bias", P, L, k);
            if (!(T->lg[k] = devvec(W, a, 512)) || !(T->lb[k] = devvec(W, b, 512))) return -1;
        }
    }
    snprintf(a, sizeof a, "%sto_out.0.weight", P); n->thw = hostvec(W, a, 512); snprintf(a, sizeof a, "%sto_out.0.bias", P); n->thb = hostvec(W, a, 512);
    snprintf(a, sizeof a, "%sto_out.1.weight", P); n->tow = hostvec(W, a, 16 * 512); snprintf(a, sizeof a, "%sto_out.1.bias", P); n->tob = hostvec(W, a, 16);
    if (!n->thw || !n->thb || !n->tow || !n->tob) return -1;
    /* rotary tables from the checkpoint's buffers: frames (RotaryEmbedding: t inv_freqs[j mod 32], the halves concatenated) and the 4 x 4
       patch grid (AxialRotaryEmbedding: linspace(-1, 1) x scales x pi for h then w, each value repeated twice) */
    snprintf(a, sizeof a, "%sframe_rot_emb.inv_freqs", P); const wl_tensor *iv = wl_get(W, a);
    snprintf(a, sizeof a, "%simage_rot_emb.scales", P); const wl_tensor *scl = wl_get(W, a);
    if (!iv || iv->numel != 32 || !scl || scl->numel != 16) { fprintf(stderr, "ink-ext: tsf: rotary buffers missing\n"); return -1; }
    float ts[26 * 64], tc[26 * 64], is[16 * 64], ic[16 * 64];
    for (int t = 0; t < 26; t++) for (int j = 0; j < 64; j++) { const float fr = (float)t * iv->data[j % 32]; ts[t * 64 + j] = sinf(fr); tc[t * 64 + j] = cosf(fr); }
    for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) for (int k = 0; k < 64; k++) {
        const int e = k / 2; const float hs = -1.f + 2.f * i / 3, ws = -1.f + 2.f * j / 3;
        const float v = e < 16 ? hs * scl->data[e] * (float)M_PI : ws * scl->data[e - 16] * (float)M_PI;
        is[(i * 4 + j) * 64 + k] = sinf(v); ic[(i * 4 + j) * 64 + k] = cosf(v);
    }
    n->trs[0] = xdev_floats(is, 16 * 64); n->trc[0] = xdev_floats(ic, 16 * 64); n->trs[1] = xdev_floats(ts, 26 * 64); n->trc[1] = xdev_floats(tc, 26 * 64);
    return 0;
}
static void tsf_fwd(const ie_net *n, const uint8_t *tiles, int B, float *out, int logits) {
    const int F = 26, S = 64, NP = 16, N = 1 + F * NP, C = 512, I = n->theads * 64;
    uint8_t *dev; upload_tiles(tiles, B, F, S, S, &dev);
    xt pin = xtsf_patches(dev, B, F, S, 16, 200.f, 200.f); xfree_dev(dev);   /* clip(0, 200) / 200 */
    xt emb = xconv_fwd(&n->tpe, pin, nullptr, XA_NONE); xt_free(&pin);
    float *x = xmalloc_dev((size_t)B * N * C * 4); xtsf_tokens(emb, n->tcls, x); xt_free(&emb);
    const long long M = (long long)B * N;
    for (int L = 0; L < n->tdepth; L++) {
        const struct tsl *T = &n->tl[L];
        for (int k = 0; k < 2; k++) {   /* time (k 0, frame rotary), then space (patch rotary) */
            xt ln = xt_new(B, 1, 1, N, C); dv_layernorm32(x, T->lg[k], T->lb[k], 1e-5f, ln.p, M, C);
            xt qkv = xconv_fwd(&T->qkv[k], ln, nullptr, XA_NONE); xt_free(&ln);
            xt ao = xt_new(B, 1, 1, N, I); xtsf_attn(qkv, ao, n->theads, F, NP, k == 0, n->trs[k == 0], n->trc[k == 0]); xt_free(&qkv);
            xt y = xconv_fwd(&T->ao[k], ao, nullptr, XA_NONE); xt_free(&ao);
            dv_add_half(x, y.p, M * C); xt_free(&y);
        }
        xt ln = xt_new(B, 1, 1, N, C); dv_layernorm32(x, T->lg[2], T->lb[2], 1e-5f, ln.p, M, C);
        xt h = xconv_fwd(&T->f0, ln, nullptr, XA_NONE); xt_free(&ln);
        xt g = xtsf_geglu(h); xt_free(&h);
        xt y = xconv_fwd(&T->f3, g, nullptr, XA_NONE); xt_free(&g);
        dv_add_half(x, y.p, M * C); xt_free(&y);
    }
    float *hx = malloc((size_t)B * N * C * 4); xd2h(hx, x, (size_t)B * N * C * 4); xfree_dev(x);   /* (only the cls rows are needed) */
    float *lg = malloc((size_t)B * 16 * sizeof *lg);
    for (int b = 0; b < B; b++) {   /* to_out: LayerNorm(512) + Linear(512, 16) of the cls token -> 4 x 4 logits */
        const float *c = hx + (size_t)b * N * C; double m = 0, v = 0; for (int i = 0; i < C; i++) m += c[i];
        m /= C; for (int i = 0; i < C; i++) v += (c[i] - m) * (c[i] - m);
        const double r = 1 / sqrt(v / C + 1e-5); double z[512];
        for (int i = 0; i < C; i++) z[i] = (c[i] - m) * r * n->thw[i] + n->thb[i];
        for (int o = 0; o < 16; o++) { double s = n->tob[o]; for (int i = 0; i < C; i++) s += z[i] * n->tow[(size_t)o * C + i]; lg[b * 16 + o] = (float)s; }
    }
    free(hx);
    up_prob(lg, B, 4, S, logits, out);
    free(lg);
}

ie_net *ie_load(int kind, const char *weights) {
    if (kind < 0 || kind >= IE_NMODELS) return nullptr;
    const char *sub = kind == IE_HEC24 || kind == IE_HEC96 ? "ema_model" : ie_is_koine(kind) ? "model" : kind == IE_CANON2 || kind == IE_TSF ? "state_dict" : nullptr;
    wl_dict *W = nullptr;
    if (ie_is_koine(kind)) W = wl_load(weights, "ema_model");   /* koine infer.py prefers the EMA weights when a checkpoint has them */
    if (!W) W = wl_load(weights, sub);
    if (!W) { fprintf(stderr, "ink-ext: %s\n", wl_error()); return nullptr; }
    ie_net *n = calloc(1, sizeof *n); n->kind = kind; n->sp = ie_specs[kind];
    int rc; xnet_wseq_reset();   /* conv indices per net (UFSM_XNET_MX_SKIP) */
    if (kind == IE_V8IN) rc = load_v8in(n, W);
    else if (ie_is_koine(kind)) rc = load_reader(n, W);
    else if (kind == IE_CANON2) rc = load_hecate(n, W);
    else if (kind == IE_STUDIO) rc = load_studio(n, W);
    else if (kind == IE_TSF) rc = load_tsf(n, W);
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
    case IE_HEC24: case IE_CANON2: return (size_t)700 << 20;
    case IE_STUDIO: return (size_t)400 << 20;
    case IE_TSF: return (size_t)24 << 20;
    case IE_HEC96: return (size_t)40 << 20;
    case IE_INK9UM: case IE_READER2: case IE_DENSE9: case IE_TRACE3B: return (size_t)56 << 20;
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
/* B maps of F x F logits -> T x T: sigmoid, bilinear (align_corners False), the logit of the clamped probability (logits) or it */
static void up_prob(const float *h2, int B, int F, int T, int logits, float *out) {
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
}
/* out: logits on the T x T grid (logit of the bilinearly upsampled probability) or probabilities */
static void hecate_fwd(const ie_net *n, const uint8_t *tiles, int B, float *out, int logits) {
    const int D = n->sp.D, T = n->sp.T, mg = n->margin, Dn = D - 2 * mg, s = n->xys, F = T / s; const size_t per = (size_t)D * T * T;
    /* the columns with data (raw.any over the full depth) at feature resolution (max-pooled over s x s) */
    uint8_t *sup = calloc((size_t)B * F * F, 1), *full = calloc((size_t)B * T * T, 1);
    if (n->kind == IE_CANON2) memset(sup, 1, (size_t)B * F * F);   /* canonical: the plain softmax over depth, no support mask */
    else for (int i = 0; i < B; i++) for (int d = 0; d < D; d++) for (int y = 0; y < T; y++) for (int x = 0; x < T; x++)
        if (tiles[i * per + ((size_t)d * T + y) * T + x]) { full[((size_t)i * T + y) * T + x] = 1; sup[((size_t)i * F + y / s) * F + x / s] = 1; }
    uint8_t *crop = malloc((size_t)B * Dn * T * T);
    for (int i = 0; i < B; i++) memcpy(crop + (size_t)i * Dn * T * T, tiles + i * per + (size_t)mg * T * T, (size_t)Dn * T * T);
    float lo[B], hi[B], a[B], b[B]; const double div = n->kind == IE_HEC24 ? 200.0 : 255.0, r = 1.0 / sqrt((double)n->in_rv + 1e-5);
    for (int i = 0; i < B; i++) { lo[i] = 0; hi[i] = 255; a[i] = (float)(r * n->in_w / div); b[i] = (float)(n->in_b - n->in_rm * r * n->in_w); }
    uint8_t *dev; upload_tiles(crop, B, Dn, T, T, &dev); free(crop);
    xt x0 = xinput_u8(dev, B, Dn, T, T, lo, hi, a, b); xfree_dev(dev);
    xt st = xconv_fwd(&n->stem, x0, nullptr, XA_RELU); xt_free(&x0);
    if (n->kind == IE_HEC24 || n->kind == IE_CANON2) { xt p = xmaxpool2d_k3s2(st); xt_free(&st); st = p; }
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
    up_prob(h2, B, F, T, logits, out);
    free(h2); free(sup); free(full);
}

/* wq: an MX conv reads the block's output next (xgn_q writes its MX copy then, UFSM_XNET_GNQ=1) */
static xt bbd_fwd(const bbd *B, xt x, int wq) {
    xt h = xconv_fwd(&B->c1, x, nullptr, XA_NONE); xgn_q(h, h.c, B->g1, B->b1, 1e-5f, nullptr, XA_LRELU, B->c2.mxq != nullptr, &h);
    xt h2 = xconv_fwd(&B->c2, h, nullptr, XA_NONE); xt_free(&h);
    xt sk = x; int own = 0;
    if (B->stride != 1) { sk = xavgpool2d_k2(x); own = 1; }
    if (B->has_sk) { xt p = xconv_fwd(&B->sk, sk, nullptr, XA_NONE); if (own) xt_free(&sk); xgn(p, p.c, B->gs, B->bs, 1e-5f, nullptr, XA_NONE, p); sk = p; own = 1; }
    xgn_q(h2, h2.c, B->g2, B->b2, 1e-5f, &sk, XA_LRELU, wq, &h2);
    if (own) xt_free(&sk);
    return h2;
}
static void reader_fwd(const ie_net *n, const uint8_t *tiles, int B, float *out) {
    const int D = n->sp.D, T = n->sp.T; const size_t per = (size_t)D * T * T; xnet_seq_reset();
    float lo[B], hi[B], a[B], b[B];
    for (int i = 0; i < B; i++) {
        if (n->gnorm) { lo[i] = n->g_lo; hi[i] = n->g_hi; a[i] = n->g_a; b[i] = n->g_b; }
        else ie_robust_params(tiles + i * per, per, &lo[i], &hi[i], &a[i], &b[i]);
    }
    uint8_t *dev; upload_tiles(tiles, B, D, T, T, &dev);
    xt x0 = xinput_u8(dev, B, D, T, T, lo, hi, a, b); xfree_dev(dev);
    xt f1 = xconv_fwd(&n->f1, x0, nullptr, XA_NONE); xt_free(&x0);
    float *g1 = xgn_stats(f1, f1.c, 1e-5f); xt f2 = xconv_fwd_gnin(&n->f2, f1, g1, g1 + (size_t)f1.n * f1.c, n->f1g, n->f1b, XA_LRELU); xfree_dev(g1); xt_free(&f1);
    static int fused = -1; if (fused < 0) fused = getenv("UFSM_IE_DATTN") ? atoi(getenv("UFSM_IE_DATTN")) : 1;   /* 0: the separate GN / head / attention kernels; mgpu: env */
    xt z;
    if (fused && f2.c == 16 && D <= 32) { z = xdepth_collapse16(f2, f2.c, n->f2g, n->f2b, 1e-5f, XA_LRELU, n->att_w, n->att_b); xt_free(&f2); }
    else {
        xgn(f2, f2.c, n->f2g, n->f2b, 1e-5f, nullptr, XA_LRELU, f2);
        float *at = xmalloc_dev((size_t)B * D * T * T * 4); xhead(f2, n->att_w, n->att_b, at);
        z = xdepth_attn_max(f2, at); xt_free(&f2); xfree_dev(at);
    }
    /* the MX conv that reads each block's output next: the next block's c1, after the last block the first tconv */
#define R3_NEXTQ(s, i) ((i) + 1 < n->nblk[s] ? n->st[s][(i) + 1].c1.mxq != nullptr : (s) + 1 < n->nst ? n->st[(s) + 1][0].c1.mxq != nullptr : n->nst > 1 && n->tconv[0].mxq != nullptr)
    xt x = xconv_fwd(&n->rstem, z, nullptr, XA_NONE); xt_free(&z); xgn_q(x, x.c, n->rstem_g, n->rstem_b, 1e-5f, nullptr, XA_LRELU, n->st[0][0].c1.mxq != nullptr, &x);
    xt skip[6];
    for (int s = 0; s < n->nst; s++) {
        for (int i = 0; i < n->nblk[s]; i++) { xt y = bbd_fwd(&n->st[s][i], x, R3_NEXTQ(s, i)); if (!(s > 0 && i == 0)) xt_free(&x); x = y; }
        skip[s] = x;
    }
    xt l = skip[n->nst - 1];
    for (int s = 0; s < n->nst - 1; s++) {
        xt up = xconv_fwd(&n->tconv[s], l, nullptr, XA_NONE); xt_free(&l);
        xt c = xconcat(up, skip[n->nst - 2 - s]); xt_free(&up); xt_free(&skip[n->nst - 2 - s]);
        l = xconv_fwd(&n->dconv[s], c, nullptr, XA_NONE); xt_free(&c);
        xgn_q(l, l.c, n->dg[s], n->db[s], 1e-5f, nullptr, XA_LRELU, s + 1 < n->nst - 1 && n->tconv[s + 1].mxq != nullptr, &l);
    }
#undef R3_NEXTQ
    float *lg = xmalloc_dev((size_t)B * T * T * 4);
    xhead(l, n->logit_w, n->logit_b, lg); xt_free(&l);
    xd2h(out, lg, (size_t)B * T * T * 4); xfree_dev(lg);
}

int ie_forward(ie_net *n, const uint8_t *tiles, int B, float *out, int logits) {
    const size_t TT = (size_t)n->sp.T * n->sp.T;
    if (n->kind == IE_V8IN) v8in_fwd(n, tiles, B, out);
    else if (ie_is_koine(n->kind)) reader_fwd(n, tiles, B, out);
    else if (n->kind == IE_STUDIO) studio_fwd(n, tiles, B, out);
    else if (n->kind == IE_TSF) tsf_fwd(n, tiles, B, out, logits);
    else hecate_fwd(n, tiles, B, out, logits);
    if (xnet_check("forward")) return -1;
    if (!logits && (n->kind == IE_V8IN || ie_is_koine(n->kind) || n->kind == IE_STUDIO)) for (size_t i = 0; i < (size_t)B * TT; i++) out[i] = 1.f / (1.f + expf(-out[i]));
    return 0;
}

void ie_set_norm(ie_net *n, int on, float lo, float hi, float a, float b) { n->gnorm = on; n->g_lo = lo; n->g_hi = hi; n->g_a = a; n->g_b = b; }
