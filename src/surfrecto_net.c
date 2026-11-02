/* surfrecto network (src/surfrecto.h): the recto-surface ResEnc U-Net with scSE on xnet (fp16 NDHWC, fp32 epilogues / norms). */
#include "surfrecto.h"
#include "wload.h"
#include "xnet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { SR_ST = 7, SR_MAXB = 6 };
static const int sr_feat[SR_ST] = {32, 64, 128, 256, 320, 320, 320}, sr_nblk[SR_ST] = {1, 3, 4, 6, 6, 6, 6};

typedef struct { xconv c; float *g, *b; } cin;               /* conv + InstanceNorm affine (device) */
typedef struct { cin c1, c2, sk; int has_sk, pool; float *sw, sb, *cg; } rblock;
struct sr_net {
    cin stem;
    rblock enc[SR_ST][SR_MAXB];
    xconv up[SR_ST - 1]; cin dec[SR_ST - 1];
    float *hw; float hb;
    float *margin; size_t cap;
    int keep; xt kept[2 * SR_ST - 1];
    int oldgn; float *mr; double *acc;   /* oldgn (UFSM_SURFRECTO_XGN=1): xnet xgn instead of the one-pass IN kernels */
};
static sr_net *g_n;   /* the net of the running forward (IN scratch) */
void sr_in_stats(const void *x, int C, long long S, float eps, float *mr_dev, double *acc_dev);
void sr_in_apply(void *x, int C, size_t nvox, const float *mr_dev, const float *g_dev, const float *b_dev, int lrelu);
void sr_in_scse(void *h, const void *res, size_t nvox, int C, const float *mr_dev, const float *g_dev, const float *b_dev,
                const float *sw_dev, float sb, const float *cg_dev);
/* InstanceNorm (affine, eps 1e-5) + optional LeakyReLU, in place */
static void inorm(xt t, const cin *c, int lrelu) {
    if (g_n->oldgn) { xgn(t, t.c, c->g, c->b, 1e-5f, nullptr, lrelu ? XA_LRELU : XA_NONE, t); return; }
    const size_t S = (size_t)t.d * t.h * t.w;
    sr_in_stats(t.p, t.c, (long long)S, 1e-5f, g_n->mr, g_n->acc);
    sr_in_apply(t.p, t.c, S, g_n->mr, c->g, c->b, lrelu);
}

static const wl_tensor *need(const wl_dict *d, const char *name) {
    const wl_tensor *t = wl_get(d, name);
    if (!t) fprintf(stderr, "surfrecto: tensor %s missing\n", name);
    return t;
}
static float *devf(const wl_dict *d, const char *name) { const wl_tensor *t = need(d, name); return t ? xdev_floats(t->data, t->numel) : nullptr; }
/* <pfx>.conv.weight / .conv.bias (when bias) / .norm.weight / .norm.bias */
static int cin_of(const wl_dict *d, cin *c, const char *pfx, int k, int s, int bias) {
    char w[256], b[256], g[256], be[256];
    snprintf(w, sizeof w, "%s.conv.weight", pfx); snprintf(b, sizeof b, "%s.conv.bias", pfx);
    snprintf(g, sizeof g, "%s.norm.weight", pfx); snprintf(be, sizeof be, "%s.norm.bias", pfx);
    const wl_tensor *wt = need(d, w), *bt = bias ? need(d, b) : nullptr; if (!wt || (bias && !bt)) return -1;
    if (wt->ndim != 5 || wt->shape[2] != k) { fprintf(stderr, "surfrecto: %s has an unexpected shape\n", w); return -1; }
    if (xconv_init(&c->c, wt->data, (int)wt->shape[0], (int)wt->shape[1], k, k, k, s, s, s, k / 2, k / 2, k / 2, nullptr, bt ? bt->data : nullptr)) return -1;
    c->g = devf(d, g); c->b = devf(d, be);
    return c->g && c->b ? 0 : -1;
}
static void cin_free(cin *c) { xconv_free(&c->c); xfree_dev(c->g); xfree_dev(c->b); c->g = c->b = nullptr; }

/* scSE of block pfx: spatial 1x1 conv (device weights + scalar bias) and the constant channel gate sigmoid(fc2(relu(fc1(beta)))) */
static int scse_of(const wl_dict *d, rblock *R, const char *pfx, int C) {
    const wl_tensor *f1w = wl_getf(d, "%s.squeeze_excitation.cSE.fc1.weight", pfx), *f1b = wl_getf(d, "%s.squeeze_excitation.cSE.fc1.bias", pfx),
                    *f2w = wl_getf(d, "%s.squeeze_excitation.cSE.fc2.weight", pfx), *f2b = wl_getf(d, "%s.squeeze_excitation.cSE.fc2.bias", pfx),
                    *sw = wl_getf(d, "%s.squeeze_excitation.sSE.conv.weight", pfx), *sb = wl_getf(d, "%s.squeeze_excitation.sSE.conv.bias", pfx),
                    *beta = wl_getf(d, "%s.conv2.norm.bias", pfx);
    if (!f1w || !f1b || !f2w || !f2b || !sw || !sb || !beta) { fprintf(stderr, "surfrecto: %s: scSE tensors missing\n", pfx); return -1; }
    const int r = (int)f1w->shape[0];
    if (f1w->shape[1] != C || f2w->shape[0] != C || f2w->shape[1] != r || sw->numel != C || beta->numel != C) { fprintf(stderr, "surfrecto: %s: scSE shapes\n", pfx); return -1; }
    double z[64]; float cg[512];
    if (r > 64 || C > 512) return -1;
    for (int j = 0; j < r; j++) { double a = f1b->data[j]; for (int c = 0; c < C; c++) a += (double)f1w->data[(size_t)j * C + c] * beta->data[c]; z[j] = a > 0 ? a : 0; }
    for (int c = 0; c < C; c++) { double a = f2b->data[c]; for (int j = 0; j < r; j++) a += (double)f2w->data[(size_t)c * r + j] * z[j]; cg[c] = (float)(1.0 / (1.0 + exp(-a))); }
    R->cg = xdev_floats(cg, C); R->sw = xdev_floats(sw->data, C); R->sb = sb->data[0];
    return R->cg && R->sw ? 0 : -1;
}

sr_net *sr_load(const char *pth) {
    wl_dict *d = wl_load(pth, "model");
    if (!d) { fprintf(stderr, "surfrecto: %s: %s\n", pth, wl_error()); return nullptr; }
    sr_net *n = calloc(1, sizeof *n); int rc = 0;
    { const char *e = getenv("UFSM_SURFRECTO_XGN"); n->oldgn = e && atoi(e); }
    char p[256];
    rc |= cin_of(d, &n->stem, "shared_encoder.stem.convs.0", 3, 1, 1);
    if (!rc && wl_get(d, "shared_encoder.stem.convs.1.conv.weight")) { fprintf(stderr, "surfrecto: more than one stem conv\n"); rc = -1; }
    int cin_ch = 32;
    for (int s = 0; s < SR_ST && !rc; s++) for (int k = 0; k < sr_nblk[s] && !rc; k++) {
        rblock *R = &n->enc[s][k]; const int st = k == 0 && s > 0 ? 2 : 1;
        snprintf(p, sizeof p, "shared_encoder.stages.%d.blocks.%d.conv1", s, k); rc |= cin_of(d, &R->c1, p, 3, st, 1);
        snprintf(p, sizeof p, "shared_encoder.stages.%d.blocks.%d.conv2", s, k); rc |= cin_of(d, &R->c2, p, 3, 1, 1);
        if (rc) break;
        if (R->c1.c.co != sr_feat[s] || R->c1.c.ci != cin_ch) { fprintf(stderr, "surfrecto: stage %d block %d is %d -> %d (expected %d -> %d)\n", s, k, R->c1.c.ci, R->c1.c.co, cin_ch, sr_feat[s]); rc = -1; break; }
        R->pool = st == 2;
        snprintf(p, sizeof p, "shared_encoder.stages.%d.blocks.%d", s, k); rc |= scse_of(d, R, p, sr_feat[s]);
        const int proj = cin_ch != sr_feat[s];
        if (proj) {
            snprintf(p, sizeof p, "shared_encoder.stages.%d.blocks.%d.skip.%d", s, k, R->pool);
            R->has_sk = 1; rc |= cin_of(d, &R->sk, p, 1, 1, 0);
        }   /* strided without a width change: the skip is AvgPool3d(2) alone */
        cin_ch = sr_feat[s];
    }
    for (int s = 0; s < SR_ST && !rc; s++) {
        snprintf(p, sizeof p, "shared_encoder.stages.%d.blocks.%d.conv1.conv.weight", s, sr_nblk[s]);
        if (wl_get(d, p)) { fprintf(stderr, "surfrecto: stage %d has more than %d blocks\n", s, sr_nblk[s]); rc = -1; }
    }
    if (!rc && wl_get(d, "shared_encoder.stages.7.blocks.0.conv1.conv.weight")) { fprintf(stderr, "surfrecto: more than %d stages\n", SR_ST); rc = -1; }
    for (int s = 0; s < SR_ST - 1 && !rc; s++) {
        const wl_tensor *w = wl_getf(d, "task_decoders.surface.transpconvs.%d.weight", s), *b = wl_getf(d, "task_decoders.surface.transpconvs.%d.bias", s);
        if (!w || !b || w->ndim != 5 || w->shape[2] != 2) { fprintf(stderr, "surfrecto: decoder transpconv %d missing / unexpected\n", s); rc = -1; break; }
        rc |= xtconv3_init(&n->up[s], w->data, (int)w->shape[0], (int)w->shape[1], b->data);
        snprintf(p, sizeof p, "task_decoders.surface.stages.%d.convs.0", s); rc |= cin_of(d, &n->dec[s], p, 3, 1, 1);
        if (wl_getf(d, "task_decoders.surface.stages.%d.convs.1.conv.weight", s)) { fprintf(stderr, "surfrecto: decoder stage %d has more than one conv\n", s); rc = -1; }
    }
    if (!rc) {
        const wl_tensor *w = need(d, "task_decoders.surface.seg_layers.5.weight"), *b = need(d, "task_decoders.surface.seg_layers.5.bias");
        if (!w || !b || w->numel != 64 || b->numel != 2) rc = -1;
        else {
            float wd[32]; for (int c = 0; c < 32; c++) wd[c] = w->data[32 + c] - w->data[c];
            n->hw = xdev_floats(wd, 32); n->hb = b->data[1] - b->data[0]; rc |= !n->hw;
        }
    }
    wl_free(d);
    if (rc || xnet_check("surfrecto load")) { fprintf(stderr, "surfrecto: cannot load %s\n", pth); sr_free(n); return nullptr; }
    return n;
}

void sr_free(sr_net *n) {
    if (!n) return;
    cin_free(&n->stem);
    for (int s = 0; s < SR_ST; s++) for (int k = 0; k < SR_MAXB; k++) {
        rblock *R = &n->enc[s][k]; cin_free(&R->c1); cin_free(&R->c2); if (R->has_sk) cin_free(&R->sk);
        xfree_dev(R->sw); xfree_dev(R->cg);
    }
    for (int s = 0; s < SR_ST - 1; s++) { xconv_free(&n->up[s]); cin_free(&n->dec[s]); }
    for (int k = 0; k < 2 * SR_ST - 1; k++) xt_free(&n->kept[k]);
    xfree_dev(n->hw); xfree_dev(n->margin); xfree_dev(n->mr); xfree_dev(n->acc);
    free(n);
}

void sr_keep(sr_net *n, int on) { n->keep = on; }
const void *sr_kept(const sr_net *n, int k, int *edge, int *ch) {
    if (k < 0 || k >= 2 * SR_ST - 1 || !n->kept[k].p) return nullptr;
    *edge = n->kept[k].d; *ch = n->kept[k].c; return n->kept[k].p;
}
void sr_copy_dev(void *dst, const void *src, size_t bytes);   /* src/nn_surfrecto.cu */
static void keep_of(sr_net *n, int k, xt t) {
    if (!n->keep) return;
    xt_free(&n->kept[k]); n->kept[k] = xt_new(t.n, t.d, t.h, t.w, t.c);
    if (n->kept[k].p) sr_copy_dev(n->kept[k].p, t.p, xt_numel(t) * 2);
}

/* BasicBlockD + scSE; x stays owned by the caller */
static xt rblock_fwd(const rblock *R, xt x) {
    xt h = xconv_fwd(&R->c1.c, x, nullptr, XA_NONE);
    inorm(h, &R->c1, 1);
    xt h2 = xconv_fwd(&R->c2.c, h, nullptr, XA_NONE); xt_free(&h);
    xt r = x, pooled = {0};
    if (R->pool) { pooled = xavgpool3d_k2(x); r = pooled; }
    if (R->has_sk) {
        r = xconv_fwd(&R->sk.c, r, nullptr, XA_NONE); inorm(r, &R->sk, 0);
        if (pooled.p) xt_free(&pooled);
    }
    const size_t nv = (size_t)h2.n * h2.d * h2.h * h2.w;
    if (g_n->oldgn) {
        xgn(h2, h2.c, R->c2.g, R->c2.b, 1e-5f, nullptr, XA_NONE, h2);
        sr_scse(h2.p, r.p, nv, h2.c, R->sw, R->sb, R->cg);
    } else {
        sr_in_stats(h2.p, h2.c, (long long)nv, 1e-5f, g_n->mr, g_n->acc);
        sr_in_scse(h2.p, r.p, nv, h2.c, g_n->mr, R->c2.g, R->c2.b, R->sw, R->sb, R->cg);
    }
    if (R->has_sk || R->pool) xt_free(&r);
    return h2;
}

const float *sr_forward(sr_net *n, const uint8_t *u8_dev, int P, float mean, float stdv) {
    if (P % 64) { fprintf(stderr, "surfrecto: window %d is not a multiple of 64\n", P); return nullptr; }
    const float lo = 0.f, hi = 255.f, a = 1.f / fmaxf(stdv, 1e-8f), b = -mean * a;
    xt x = xinput_u8(u8_dev, 1, P, P, P, &lo, &hi, &a, &b);
    if (!x.p) { fprintf(stderr, "surfrecto: out of device memory (input)\n"); return nullptr; }
    g_n = n;
    if (!n->mr) { n->mr = xmalloc_dev(sizeof(float) * 2 * 512); n->acc = xmalloc_dev(sizeof(double) * 2 * 512); }
    xt h = xconv_fwd(&n->stem.c, x, nullptr, XA_NONE); xt_free(&x);
    if (!h.p || !n->mr || !n->acc) { fprintf(stderr, "surfrecto: out of device memory (stem)\n"); return nullptr; }
    inorm(h, &n->stem, 1);
    x = h;
    xt skip[SR_ST];
    for (int s = 0; s < SR_ST; s++) {
        for (int k = 0; k < sr_nblk[s]; k++) {
            xt y = rblock_fwd(&n->enc[s][k], x);
            if (!y.p) { fprintf(stderr, "surfrecto: out of device memory (stage %d)\n", s); return nullptr; }
            if (!(s > 0 && k == 0)) xt_free(&x);   /* the previous stage's output stays as its skip */
            x = y;
        }
        skip[s] = x; keep_of(n, s, x);
    }
    xt lo_t = skip[SR_ST - 1];
    for (int s = 0; s < SR_ST - 1; s++) {
        xt u = xconv_fwd(&n->up[s], lo_t, nullptr, XA_NONE);
        xt_free(&lo_t);
        xt c = xconcat(u, skip[SR_ST - 2 - s]); xt_free(&u); xt_free(&skip[SR_ST - 2 - s]);
        xt hh = xconv_fwd(&n->dec[s].c, c, nullptr, XA_NONE); xt_free(&c);
        if (!hh.p) { fprintf(stderr, "surfrecto: out of device memory (decoder %d)\n", s); return nullptr; }
        inorm(hh, &n->dec[s], 1);
        lo_t = hh; keep_of(n, SR_ST + s, hh);
    }
    const size_t nv = (size_t)P * P * P;
    if (n->cap < nv) { xfree_dev(n->margin); n->margin = xmalloc_dev(nv * 4); n->cap = n->margin ? nv : 0; }
    if (!n->margin) { xt_free(&lo_t); fprintf(stderr, "surfrecto: out of device memory (margin)\n"); return nullptr; }
    xhead(lo_t, n->hw, n->hb, n->margin);
    xt_free(&lo_t);
    return xnet_check("surfrecto forward") ? nullptr : n->margin;
}

void sr_zstats(const uint8_t *u, size_t n, double *mean, double *stdv, int *vmin, int *vmax) {
    uint64_t h[4][256]; memset(h, 0, sizeof h);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) { h[0][u[i]]++; h[1][u[i + 1]]++; h[2][u[i + 2]]++; h[3][u[i + 3]]++; }
    for (; i < n; i++) h[0][u[i]]++;
    double s = 0, s2 = 0; int lo = 256, hi = -1;
    for (int v = 0; v < 256; v++) {
        const uint64_t c = h[0][v] + h[1][v] + h[2][v] + h[3][v];
        if (!c) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
        s += (double)c * v;
    }
    const double m = n ? s / (double)n : 0.0;
    for (int v = 0; v < 256; v++) { const uint64_t c = h[0][v] + h[1][v] + h[2][v] + h[3][v]; if (c) s2 += (double)c * (v - m) * (v - m); }
    *mean = m; *stdv = n ? sqrt(s2 / (double)n) : 0.0;
    if (vmin) *vmin = lo;
    if (vmax) *vmax = hi;
}
