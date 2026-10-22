/* dinovol network (src/dinovol.h): weights from the torch zip checkpoint (teacher branch), the tile forward and the dense head. */
#include "dinovol.h"
#include "wload.h"
#include "xnet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { DV_E = 480, DV_H = 10, DV_HD = 48, DV_PRE = 5, DV_MAXBLK = 32, DV_MAXST = 8, DV_PATCH = 8, DV_HCHUNK = 131072 };

typedef struct { xconv c; float *w, *b; } cln;   /* conv + channel LayerNorm (eps 1e-5) */
typedef struct { cln c1, c2, sk; int has_sk, stride; } bblock;
typedef struct { float *ln1w, *ln1b, *ln2w, *ln2b, *mix; xconv qkv, proj, fc1, fc2; } vblock;

struct dv_net {
    cln stem; bblock bb[DV_MAXST * 4]; int nbb, stage_end[DV_MAXST], nstage, stage_ch[DV_MAXST]; xconv fproj;
    void *prefix;                              /* fp16 [5][480]: cls, 4 registers */
    vblock vb[DV_MAXBLK]; int depth; float *nw, *nb;
    int F;                                     /* SwiGLU hidden (3456) */
    /* dense head */
    xconv fpn_in, lat, ref2, mlp0, mlp2, shortcut, proj; float *refw, *refb, *fnw, *fnb, *skw[3], *skb[3], *tw, *tb; int fpn, hid, nd, ni;
    /* tile state */
    int T; xt tok, sk[3], fpng; void *X;
};

static const wl_tensor *need(const wl_dict *d, const char *name) {
    const wl_tensor *t = wl_get(d, name);
    if (!t) fprintf(stderr, "dinovol: tensor %s missing\n", name);
    return t;
}
static float *devf(const wl_dict *d, const char *name) { const wl_tensor *t = need(d, name); return t ? xdev_floats(t->data, t->numel) : nullptr; }
static int conv_of(const wl_dict *d, xconv *c, const char *wname, const char *bname, int stride, int pad, const float *scale, const float *bias_override) {
    const wl_tensor *w = need(d, wname); if (!w) return -1;
    const int co = (int)w->shape[0], ci = (int)w->shape[1], k = w->ndim == 5 ? (int)w->shape[2] : 1;
    const float *bias = bias_override;
    if (!bias && bname) { const wl_tensor *b = need(d, bname); if (!b) return -1; bias = b->data; }
    return xconv_init(c, w->data, co, ci, k, k, k, stride, stride, stride, pad, pad, pad, scale, bias);
}
static int cln_of(const wl_dict *d, cln *c, const char *pfx, int stride, int k, int has_bias) {
    char wn[256], bn[256], nw[256], nb[256];
    snprintf(wn, sizeof wn, "%s.conv.weight", pfx); snprintf(bn, sizeof bn, "%s.conv.bias", pfx);
    snprintf(nw, sizeof nw, "%s.norm.weight", pfx); snprintf(nb, sizeof nb, "%s.norm.bias", pfx);
    if (conv_of(d, &c->c, wn, has_bias ? bn : nullptr, stride, k / 2, nullptr, nullptr)) return -1;
    c->w = devf(d, nw); c->b = devf(d, nb);
    return c->w && c->b ? 0 : -1;
}

dv_net *dv_load(const char *ckpt) {
    wl_dict *d = wl_load(ckpt, "teacher");
    if (!d) { fprintf(stderr, "dinovol: %s: %s\n", ckpt, wl_error()); return nullptr; }
    dv_net *n = calloc(1, sizeof *n); int rc = 0;
    rc |= cln_of(d, &n->stem, "backbone.down_projection.stem.convs.0", 1, 3, 1);
    for (int s = 0; s < DV_MAXST; s++) {
        char p[256]; snprintf(p, sizeof p, "backbone.down_projection.stages.%d.blocks.0.conv1.conv.weight", s);
        if (!wl_get(d, p)) break;
        for (int b = 0;; b++) {
            snprintf(p, sizeof p, "backbone.down_projection.stages.%d.blocks.%d.conv1.conv.weight", s, b);
            const wl_tensor *w = wl_get(d, p); if (!w) break;
            bblock *B = &n->bb[n->nbb++];
            snprintf(p, sizeof p, "backbone.down_projection.stages.%d.blocks.%d.skip.1.conv.weight", s, b);
            B->has_sk = wl_get(d, p) != nullptr;
            B->stride = s > 0 && b == 0 ? 2 : 1;
            char q[256];
            snprintf(q, sizeof q, "backbone.down_projection.stages.%d.blocks.%d.conv1", s, b); rc |= cln_of(d, &B->c1, q, B->stride, 3, 0);
            snprintf(q, sizeof q, "backbone.down_projection.stages.%d.blocks.%d.conv2", s, b); rc |= cln_of(d, &B->c2, q, 1, 3, 0);
            if (B->has_sk) { snprintf(q, sizeof q, "backbone.down_projection.stages.%d.blocks.%d.skip.1", s, b); rc |= cln_of(d, &B->sk, q, 1, 1, 0); }
            n->stage_ch[s] = (int)w->shape[0];
        }
        n->stage_end[s] = n->nbb; n->nstage = s + 1;
    }
    rc |= conv_of(d, &n->fproj, "backbone.down_projection.final_proj.weight", "backbone.down_projection.final_proj.bias", 1, 0, nullptr, nullptr);
    {   /* prefix tokens: cls then the registers (Eva._pos_embed) */
        const wl_tensor *c = need(d, "backbone.cls_token"), *r = need(d, "backbone.reg_token");
        if (!c || !r || r->numel != (DV_PRE - 1) * DV_E) rc = -1;
        else {
            float tmp[DV_PRE * DV_E]; memcpy(tmp, c->data, DV_E * 4); memcpy(tmp + DV_E, r->data, (DV_PRE - 1) * DV_E * 4);
            xt t = xt_new(1, 1, 1, DV_PRE, DV_E); n->prefix = t.p;
            /* float -> fp16 on the host via the input path: upload floats, convert with a 1x1 identity-free route: use a host conversion */
            uint16_t h[DV_PRE * DV_E];
            for (int i = 0; i < DV_PRE * DV_E; i++) {   /* round to nearest even fp16 */
                const float f = tmp[i]; uint32_t x; memcpy(&x, &f, 4);
                const uint32_t sign = (x >> 16) & 0x8000u; int e = (int)((x >> 23) & 0xff) - 127 + 15; uint32_t m = x & 0x7fffffu;
                if (e <= 0) { if (e < -10) h[i] = (uint16_t)sign; else { m |= 0x800000u; const int sh = 14 - e; uint32_t v = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1); if (rem > half || (rem == half && (v & 1))) v++; h[i] = (uint16_t)(sign | v); } }
                else if (e >= 31) h[i] = (uint16_t)(sign | 0x7c00u);
                else { uint32_t v = ((uint32_t)e << 10) | (m >> 13); const uint32_t rem = m & 0x1fffu; if (rem > 0x1000u || (rem == 0x1000u && (v & 1))) v++; h[i] = (uint16_t)(sign | v); }
            }
            xh2d(n->prefix, h, sizeof h);
        }
    }
    for (int b = 0; b < DV_MAXBLK; b++) {
        char p[256]; snprintf(p, sizeof p, "backbone.blocks.%d.attn.qkv.weight", b);
        if (!wl_get(d, p)) break;
        vblock *V = &n->vb[b]; n->depth = b + 1;
#define BN(s) (snprintf(p, sizeof p, "backbone.blocks.%d." s, b), p)
        V->ln1w = devf(d, BN("norm1.weight")); V->ln1b = devf(d, BN("norm1.bias"));
        V->ln2w = devf(d, BN("norm2.weight")); V->ln2b = devf(d, BN("norm2.bias"));
        V->mix = devf(d, BN("rope_embed.mix_frequencies"));
        rc |= !V->ln1w || !V->ln1b || !V->ln2w || !V->ln2b || !V->mix;
        const wl_tensor *mixt = wl_get(d, BN("rope_embed.mix_frequencies"));
        if (mixt && (mixt->shape[0] != DV_H || mixt->shape[1] != DV_HD / 2 || mixt->shape[2] != 3)) { fprintf(stderr, "dinovol: unexpected RoPE shape\n"); rc = -1; }
        rc |= conv_of(d, &V->qkv, BN("attn.qkv.weight"), nullptr, 1, 0, nullptr, nullptr);
        /* x + gamma (W o + b): epilogue scale gamma, bias gamma b */
        const wl_tensor *g1 = need(d, BN("gamma_1")), *pb = need(d, BN("attn.proj.bias")), *g2 = need(d, BN("gamma_2")), *f2b = need(d, BN("mlp.fc2.bias"));
        if (!g1 || !pb || !g2 || !f2b) { rc = -1; break; }
        float sb[DV_E];
        for (int i = 0; i < DV_E; i++) sb[i] = g1->data[i] * pb->data[i];
        rc |= conv_of(d, &V->proj, BN("attn.proj.weight"), nullptr, 1, 0, g1->data, sb);
        {   /* fc1 rows interleaved (x1_j, x2_j) for the fused SwiGLU epilogue (XA_GLU) */
            char wq[256]; snprintf(wq, sizeof wq, "backbone.blocks.%d.mlp.fc1.weight", b);
            const wl_tensor *w1 = need(d, wq), *b1 = need(d, BN("mlp.fc1.bias"));
            if (!w1 || !b1) { rc = -1; break; }
            const int F2 = (int)w1->shape[0], Fh = F2 / 2, K1 = (int)w1->shape[1];
            float *wi = malloc((size_t)F2 * K1 * 4), *bi = malloc((size_t)F2 * 4);
            for (int jj = 0; jj < Fh; jj++) {
                memcpy(wi + (size_t)(2 * jj) * K1, w1->data + (size_t)jj * K1, (size_t)K1 * 4); memcpy(wi + (size_t)(2 * jj + 1) * K1, w1->data + (size_t)(Fh + jj) * K1, (size_t)K1 * 4);
                bi[2 * jj] = b1->data[jj]; bi[2 * jj + 1] = b1->data[Fh + jj];
            }
            rc |= xconv_init(&V->fc1, wi, F2, K1, 1, 1, 1, 1, 1, 1, 0, 0, 0, nullptr, bi);
            free(wi); free(bi);
        }
        for (int i = 0; i < DV_E; i++) sb[i] = g2->data[i] * f2b->data[i];
        rc |= conv_of(d, &V->fc2, BN("mlp.fc2.weight"), nullptr, 1, 0, g2->data, sb);
        const wl_tensor *f1 = wl_get(d, BN("mlp.fc1.weight")); if (f1) n->F = (int)f1->shape[0] / 2;
#undef BN
    }
    n->nw = devf(d, "backbone.norm.weight"); n->nb = devf(d, "backbone.norm.bias");
    /* dense head */
    rc |= conv_of(d, &n->fpn_in, "dense_head.fpn_in.weight", "dense_head.fpn_in.bias", 1, 0, nullptr, nullptr);
    rc |= conv_of(d, &n->lat, "dense_head.fpn_laterals.0.weight", "dense_head.fpn_laterals.0.bias", 1, 0, nullptr, nullptr);
    rc |= conv_of(d, &n->ref2, "dense_head.fpn_refines.0.2.weight", "dense_head.fpn_refines.0.2.bias", 1, 0, nullptr, nullptr);
    n->refw = devf(d, "dense_head.fpn_refines.0.0.weight"); n->refb = devf(d, "dense_head.fpn_refines.0.0.bias");
    n->fnw = devf(d, "dense_head.fpn_norm.weight"); n->fnb = devf(d, "dense_head.fpn_norm.bias");
    for (int k = 0; k < 3; k++) { char p[128]; snprintf(p, sizeof p, "dense_head.skip_norms.%d.weight", k); n->skw[k] = devf(d, p); snprintf(p, sizeof p, "dense_head.skip_norms.%d.bias", k); n->skb[k] = devf(d, p); }
    rc |= conv_of(d, &n->mlp0, "dense_head.mlp.0.weight", "dense_head.mlp.0.bias", 1, 0, nullptr, nullptr);
    rc |= conv_of(d, &n->mlp2, "dense_head.mlp.2.weight", "dense_head.mlp.2.bias", 1, 0, nullptr, nullptr);
    rc |= conv_of(d, &n->shortcut, "dense_head.token_shortcut.weight", "dense_head.token_shortcut.bias", 1, 0, nullptr, nullptr);
    n->tw = devf(d, "dense_head.trunk_norm.weight"); n->tb = devf(d, "dense_head.trunk_norm.bias");
    {   /* obj_proj and ink_proj as one GEMM (64 + 32 outputs) */
        const wl_tensor *ow = need(d, "dense_head.obj_proj.weight"), *ob = need(d, "dense_head.obj_proj.bias"), *iw = need(d, "dense_head.ink_proj.weight"), *ib = need(d, "dense_head.ink_proj.bias");
        if (!ow || !ob || !iw || !ib) rc = -1;
        else {
            n->nd = (int)ow->shape[0]; n->ni = (int)iw->shape[0]; n->hid = (int)ow->shape[1];
            const int co = n->nd + n->ni; float *w = malloc((size_t)co * n->hid * 4), *bb = malloc((size_t)co * 4);
            memcpy(w, ow->data, (size_t)n->nd * n->hid * 4); memcpy(w + (size_t)n->nd * n->hid, iw->data, (size_t)n->ni * n->hid * 4);
            memcpy(bb, ob->data, n->nd * 4); memcpy(bb + n->nd, ib->data, n->ni * 4);
            rc |= xconv_init(&n->proj, w, co, n->hid, 1, 1, 1, 1, 1, 1, 0, 0, 0, nullptr, bb);
            free(w); free(bb);
        }
    }
    { const wl_tensor *f = wl_get(d, "dense_head.fpn_in.weight"); n->fpn = f ? (int)f->shape[0] : 0; }
    rc |= !n->nw || !n->nb || !n->refw || !n->refb || !n->fnw || !n->fnb || !n->tw || !n->tb || !n->skw[0] || !n->skw[1] || !n->skw[2];
    if (n->nstage != 4 || n->stage_ch[n->nstage - 1] <= 0 || n->depth < 1 || n->fpn % 8) { fprintf(stderr, "dinovol: unexpected architecture (%d stages, %d blocks)\n", n->nstage, n->depth); rc = -1; }
    wl_free(d);
    if (rc || xnet_check("dinovol load")) { fprintf(stderr, "dinovol: cannot load %s\n", ckpt); dv_free(n); return nullptr; }
    return n;
}
static void cln_free(cln *c) { xconv_free(&c->c); xfree_dev(c->w); xfree_dev(c->b); }
void dv_free(dv_net *n) {
    if (!n) return;
    cln_free(&n->stem);
    for (int i = 0; i < n->nbb; i++) { cln_free(&n->bb[i].c1); cln_free(&n->bb[i].c2); if (n->bb[i].has_sk) cln_free(&n->bb[i].sk); }
    xconv_free(&n->fproj); xfree_dev(n->prefix);
    for (int b = 0; b < n->depth; b++) { vblock *V = &n->vb[b]; xfree_dev(V->ln1w); xfree_dev(V->ln1b); xfree_dev(V->ln2w); xfree_dev(V->ln2b); xfree_dev(V->mix);
        xconv_free(&V->qkv); xconv_free(&V->proj); xconv_free(&V->fc1); xconv_free(&V->fc2); }
    xfree_dev(n->nw); xfree_dev(n->nb);
    xconv_free(&n->fpn_in); xconv_free(&n->lat); xconv_free(&n->ref2); xconv_free(&n->mlp0); xconv_free(&n->mlp2); xconv_free(&n->shortcut); xconv_free(&n->proj);
    xfree_dev(n->refw); xfree_dev(n->refb); xfree_dev(n->fnw); xfree_dev(n->fnb); xfree_dev(n->tw); xfree_dev(n->tb);
    for (int k = 0; k < 3; k++) { xfree_dev(n->skw[k]); xfree_dev(n->skb[k]); }
    xt_free(&n->tok); for (int k = 0; k < 3; k++) xt_free(&n->sk[k]); xt_free(&n->fpng); xfree_dev(n->X);
    free(n);
}
int dv_embed_dim(const dv_net *n) { (void)n; return DV_E; }
int dv_dense_dims(const dv_net *n, int *nd, int *ni) { *nd = n->nd; *ni = n->ni; return n->nd + n->ni; }
int dv_patch(const dv_net *n) { (void)n; return DV_PATCH; }
int dv_train_tile(const dv_net *n) { (void)n; return 192; }
const void *dv_tokens(const dv_net *n) { return n->tok.p; }
const void *dv_skip(const dv_net *n, int k, int *edge, int *ch) { *edge = n->sk[k].d; *ch = n->sk[k].c; return n->sk[k].p; }
const void *dv_fpn(const dv_net *n, int *edge, int *ch) { *edge = n->fpng.d; *ch = n->fpng.c; return n->fpng.p; }

/* conv + channel LayerNorm (in place) (+ res) (ReLU) */
static xt cln_fwd(const cln *c, xt x, const xt *res, int relu) {
    xt y = xconv_fwd(&c->c, x, nullptr, XA_NONE);
    dv_layernorm(y.p, c->w, c->b, 1e-5f, res ? res->p : nullptr, relu, y.p, (long long)y.n * y.d * y.h * y.w, y.c);
    return y;
}
static xt bblock_fwd(const bblock *B, xt x) {
    xt r = x, pooled = {0};
    if (B->has_sk) {
        xt src = x;
        if (B->stride == 2) { pooled = xt_new(1, x.d / 2, x.h / 2, x.w / 2, x.c); dv_avgpool2(x.p, pooled.p, x.d, x.h, x.w, x.c); src = pooled; }
        r = cln_fwd(&B->sk, src, nullptr, 0);
        xt_free(&pooled);
    }
    xt h = cln_fwd(&B->c1, x, nullptr, 1);
    xt y = cln_fwd(&B->c2, h, &r, 1);
    xt_free(&h);
    if (B->has_sk) xt_free(&r);
    return y;
}

int dv_forward(dv_net *n, const uint8_t *u8_dev, int T, float center, float scale, float hi) {
    if (T % (DV_PATCH * 2) || T < 32) { fprintf(stderr, "dinovol: tile %d must be a multiple of 16\n", T); return -1; }
    xt_free(&n->tok); for (int k = 0; k < 3; k++) xt_free(&n->sk[k]); xt_free(&n->fpng);
    n->T = T;
    const float lo = -1e30f, a = 1.f / scale, b = -center / scale;
    xt in = xinput_u8(u8_dev, 1, T, T, T, &lo, &hi, &a, &b);
    xt x = cln_fwd(&n->stem, in, nullptr, 1); xt_free(&in);
    int bi = 0, owned = 1;   /* owned: x is not one of the kept skips */
    for (int s = 0; s < n->nstage; s++) {
        for (; bi < n->stage_end[s]; bi++) { xt y = bblock_fwd(&n->bb[bi], x); if (owned) xt_free(&x); x = y; owned = 1; }
        if (s < 3) { n->sk[s] = x; owned = 0; }   /* the stage output is both a skip and the next input */
    }
    xt tk = xconv_fwd(&n->fproj, x, nullptr, XA_NONE); xt_free(&x);
    const int g = T / DV_PATCH; const long long Np = (long long)g * g * g, N = Np + DV_PRE;
    /* token sequence X [N][480] (fp32 residual stream, as under autocast): prefix then patches */
    float *X = xmalloc_dev((size_t)N * DV_E * 4);
    dv_half_to_float(X, n->prefix, (long long)DV_PRE * DV_E);
    dv_half_to_float(X + (size_t)DV_PRE * DV_E, tk.p, Np * DV_E);
    xt_free(&tk);
    xt h = xt_new(1, 1, 1, (int)N, DV_E), o = xt_new(1, 1, 1, (int)N, DV_E);
    for (int b = 0; b < n->depth; b++) {
        const vblock *V = &n->vb[b];
        dv_layernorm32(X, V->ln1w, V->ln1b, 1e-5f, h.p, N, DV_E);
        xt qkv = xconv_fwd(&V->qkv, h, nullptr, XA_NONE);
        dv_rope(qkv.p, V->mix, (int)N, DV_PRE, g, DV_H);
        dv_attention48(qkv.p, o.p, (int)N, DV_H);
        xt_free(&qkv);
        xt u = xconv_fwd(&V->proj, o, nullptr, XA_NONE); dv_add_half(X, u.p, N * DV_E); xt_free(&u);
        dv_layernorm32(X, V->ln2w, V->ln2b, 1e-5f, h.p, N, DV_E);
        xt G = xconv_fwd(&V->fc1, h, nullptr, XA_GLU);   /* [N][F]: silu(x1) x2 fused */
        u = xconv_fwd(&V->fc2, G, nullptr, XA_NONE); xt_free(&G); dv_add_half(X, u.p, N * DV_E); xt_free(&u);
    }
    dv_layernorm32(X, n->nw, n->nb, 1e-5f, h.p, N, DV_E);
    n->tok = xt_new(1, g, g, g, DV_E);
    {
        extern void dv_copy_dev(void *dst, const void *src, size_t bytes);
        dv_copy_dev(n->tok.p, (const char *)h.p + (size_t)DV_PRE * DV_E * 2, (size_t)Np * DV_E * 2);
    }
    xt_free(&h); xt_free(&o); xfree_dev(X);
    /* FPN grid at the 128-channel skip's resolution */
    xt f = xconv_fwd(&n->fpn_in, n->tok, nullptr, XA_NONE);
    const xt s2 = n->sk[2];
    xt up = xresize(f, s2.d, s2.h, s2.w); xt_free(&f);
    xt f2 = xconv_fwd(&n->lat, s2, &up, XA_NONE); xt_free(&up);
    xt r = xt_new(1, f2.d, f2.h, f2.w, f2.c); dv_gconv8_gelu(f2.p, n->refw, n->refb, r.p, f2.d, f2.c);
    n->fpng = xconv_fwd(&n->ref2, r, &f2, XA_NONE); xt_free(&r); xt_free(&f2);
    return xnet_check("dinovol forward");
}

int dv_net_points(dv_net *n, const float *coords_dev, const float *wgt_dev, const int *rows_dev, int P, float *acc_dev) {
    const int C = n->nd + n->ni, Ft = DV_E + n->sk[0].c + n->sk[1].c + n->sk[2].c + n->fpn;
    for (int p0 = 0; p0 < P; p0 += DV_HCHUNK) {
        const int np = P - p0 < DV_HCHUNK ? P - p0 : DV_HCHUNK;
        xt tok = xt_new(1, 1, 1, np, DV_E), feat = xt_new(1, 1, 1, np, Ft);
        dv_gather_grid G[5] = {{n->tok.p, n->tok.d, DV_E, nullptr, nullptr},
                               {n->sk[0].p, n->sk[0].d, n->sk[0].c, n->skw[0], n->skb[0]}, {n->sk[1].p, n->sk[1].d, n->sk[1].c, n->skw[1], n->skb[1]},
                               {n->sk[2].p, n->sk[2].d, n->sk[2].c, n->skw[2], n->skb[2]}, {n->fpng.p, n->fpng.d, n->fpng.c, n->fnw, n->fnb}};
        dv_gather(G, 5, n->T, 1e-5f, coords_dev + (size_t)p0 * 3, np, tok.p, feat.p, Ft);
        xt S = xconv_fwd(&n->shortcut, tok, nullptr, XA_NONE); xt_free(&tok);
        xt h1 = xconv_fwd(&n->mlp0, feat, nullptr, XA_GELU); xt_free(&feat);
        xt h2 = xconv_fwd(&n->mlp2, h1, &S, XA_GELU | XA_POSTRES); xt_free(&h1); xt_free(&S);
        dv_layernorm(h2.p, n->tw, n->tb, 1e-5f, nullptr, 0, h2.p, np, n->hid);
        xt pr = xconv_fwd(&n->proj, h2, nullptr, XA_NONE); xt_free(&h2);
        dv_head_out(pr.p, wgt_dev ? wgt_dev + p0 : nullptr, rows_dev ? rows_dev + p0 : nullptr, np, wgt_dev ? acc_dev : acc_dev + (size_t)p0 * C, n->nd, n->ni);
        xt_free(&pr);
    }
    return xnet_check("dinovol points");
}
