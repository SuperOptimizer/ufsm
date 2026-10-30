/* villa winding model network (src/villawind.h): weights from ckpt_final.pth ("model"), the slab forward. Reference:
   villa winding_models/winding_model.py @37d1897 (WindingModel with encoder 32/64/96, trunk 192, 2 stem blocks, 6 axial blocks,
   6 heads, max_relative_distance 128, decoder 96, decoder_levels 1, full_resolution_head). Layout: villa [B, C, H, W, L] =
   xnet NDHWC with D = H (axis a), H = W (axis b), W = L (ray). The trunk residual stream is fp32; convs / linears fp16 x fp16 -> fp32. */
#include "villawind.h"
#include "wload.h"
#include "xnet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { VD = 192, VH = 6, VHD = 32, VMLP = 768, VDEC = 96, VNB = 6, VRMAX = 128 };
static const float EPS = 1e-5f;

typedef struct { xconv c; float *g, *b; } cln;   /* conv + channel LayerNorm (device affine) */
typedef struct {
    float *rnw, *rnb, *tnw, *tnb, *mnw, *mnb, *rbias, *tbias;   /* device */
    xconv rqkv, rproj, tqkv, tproj, fc1, fc2;
} ablock;
struct vw_net {
    cln st[3][2], tin, stem[2][2], dec[2], fh0;
    ablock B[VNB];
    float *fh3w[4]; float fh3b[4];
    int tside;   /* transverse bias table side 2 md + 1 */
};

static const wl_tensor *need(const wl_dict *d, const char *name) {
    const wl_tensor *t = wl_get(d, name);
    if (!t) fprintf(stderr, "villawind: tensor %s missing\n", name);
    return t;
}
static float *devf(const wl_dict *d, const char *name) { const wl_tensor *t = need(d, name); return t ? xdev_floats(t->data, t->numel) : nullptr; }
/* conv (k 3 or 1) + LN: names <cw>.weight/bias, <lw>.weight/bias; stride (s, s, 1); ci padded to cip with zero weights */
static int cln_of(const wl_dict *d, cln *c, const char *cw, const char *lw, int s, int cip) {
    char a[256], b[256]; snprintf(a, sizeof a, "%s.weight", cw); snprintf(b, sizeof b, "%s.bias", cw);
    const wl_tensor *wt = need(d, a), *bt = need(d, b); if (!wt || !bt) return -1;
    const int co = (int)wt->shape[0], ci = (int)wt->shape[1], k = (int)wt->shape[2], kk = k * k * k;
    const float *w = wt->data; float *wp = nullptr;
    if (cip > ci) {
        wp = calloc((size_t)co * cip * kk, 4);
        for (int o = 0; o < co; o++) for (int i = 0; i < ci; i++) memcpy(wp + ((size_t)o * cip + i) * kk, w + ((size_t)o * ci + i) * kk, (size_t)kk * 4);
        w = wp;
    }
    const int rc = xconv_init(&c->c, w, co, cip > ci ? cip : ci, k, k, k, s, s, 1, k / 2, k / 2, k / 2, nullptr, bt->data);
    free(wp); if (rc) return -1;
    if (!lw) { c->g = c->b = nullptr; return 0; }
    snprintf(a, sizeof a, "%s.weight", lw); snprintf(b, sizeof b, "%s.bias", lw);
    c->g = devf(d, a); c->b = devf(d, b);
    return c->g && c->b ? 0 : -1;
}
static int lin_of(const wl_dict *d, xconv *c, const char *pfx) {
    char a[256], b[256]; snprintf(a, sizeof a, "%s.weight", pfx); snprintf(b, sizeof b, "%s.bias", pfx);
    const wl_tensor *wt = need(d, a), *bt = need(d, b); if (!wt || !bt) return -1;
    return xconv_init(c, wt->data, (int)wt->shape[0], (int)wt->shape[1], 1, 1, 1, 1, 1, 1, 0, 0, 0, nullptr, bt->data);
}

vw_net *vw_load(const char *pth) {
    wl_dict *d = wl_load(pth, "model"); if (!d) { fprintf(stderr, "villawind: %s: %s\n", pth, wl_error()); return nullptr; }
    vw_net *n = calloc(1, sizeof *n); int rc = 0; char a[256], b[256];
    for (int s = 0; s < 3; s++) {
        snprintf(a, sizeof a, "stages.%d.layers.0", s); snprintf(b, sizeof b, "stages.%d.layers.1", s); rc |= cln_of(d, &n->st[s][0], a, b, 2, s ? 0 : 8);
        snprintf(a, sizeof a, "stages.%d.layers.3", s); snprintf(b, sizeof b, "stages.%d.layers.4", s); rc |= cln_of(d, &n->st[s][1], a, b, 1, 0);
    }
    rc |= cln_of(d, &n->tin, "trunk_in.0", "trunk_in.1", 1, 0);
    for (int k = 0; k < 2; k++) {
        snprintf(a, sizeof a, "stem_blocks.%d.conv1", k); snprintf(b, sizeof b, "stem_blocks.%d.norm1", k); rc |= cln_of(d, &n->stem[k][0], a, b, 1, 0);
        snprintf(a, sizeof a, "stem_blocks.%d.conv2", k); snprintf(b, sizeof b, "stem_blocks.%d.norm2", k); rc |= cln_of(d, &n->stem[k][1], a, b, 1, 0);
    }
    for (int k = 0; k < VNB && !rc; k++) {
        ablock *B = &n->B[k];
#define DV(f, nm) (snprintf(a, sizeof a, "attention_blocks.%d." nm, k), B->f = devf(d, a), rc |= !B->f)
#define LN(f, nm) (snprintf(a, sizeof a, "attention_blocks.%d." nm, k), rc |= lin_of(d, &B->f, a))
        DV(rnw, "ray_norm.weight"); DV(rnb, "ray_norm.bias"); DV(tnw, "transverse_norm.weight"); DV(tnb, "transverse_norm.bias");
        DV(mnw, "mlp_norm.weight"); DV(mnb, "mlp_norm.bias"); DV(rbias, "ray_attention.relative_bias"); DV(tbias, "transverse_attention.relative_bias");
        LN(rqkv, "ray_attention.qkv"); LN(rproj, "ray_attention.proj"); LN(tqkv, "transverse_attention.qkv"); LN(tproj, "transverse_attention.proj");
        LN(fc1, "mlp.0"); LN(fc2, "mlp.2");
#undef DV
#undef LN
        snprintf(a, sizeof a, "attention_blocks.%d.ray_attention.relative_bias", k); const wl_tensor *rb = wl_get(d, a);
        if (rb && rb->shape[1] != 2 * VRMAX + 1) { fprintf(stderr, "villawind: ray bias %lld, expected %d\n", (long long)rb->shape[1], 2 * VRMAX + 1); rc = 1; }
        snprintf(a, sizeof a, "attention_blocks.%d.transverse_attention.relative_bias", k); const wl_tensor *tb = wl_get(d, a);
        if (tb) n->tside = (int)tb->shape[1];
    }
    rc |= cln_of(d, &n->dec[0], "decoder.0", "decoder.1", 1, 0);
    rc |= cln_of(d, &n->dec[1], "decoder.3", "decoder.4", 1, 0);
    rc |= cln_of(d, &n->fh0, "full_resolution_head.0", "full_resolution_head.1", 1, 0);
    const wl_tensor *hw = need(d, "full_resolution_head.3.weight"), *hb = need(d, "full_resolution_head.3.bias");
    if (!hw || !hb || hw->shape[0] != 4 || hw->shape[1] != 32) rc = 1;
    else for (int c = 0; c < 4; c++) { n->fh3w[c] = xdev_floats(hw->data + c * 32, 32); n->fh3b[c] = hb->data[c]; }
    wl_free(d);
    if (rc) { fprintf(stderr, "villawind: %s: unexpected checkpoint layout\n", pth); free(n); return nullptr; }
    return n;
}
void vw_free(vw_net *n) {
    if (!n) return;
    /* device weights live in the xnet cache for the process lifetime */
    free(n);
}

static void dump_xt(const char *dir, const char *name, xt t) {
    if (!dir) return;
    const size_t n = xt_numel(t); uint16_t *h = malloc(n * 2); float *f = malloc(n * 4);
    xd2h(h, t.p, n * 2);
    for (size_t i = 0; i < n; i++) { const uint16_t u = h[i]; const uint32_t s = (u & 0x8000u) << 16; int e = (u >> 10) & 31; uint32_t m = u & 0x3ff, v;
        if (e == 0) { if (!m) v = s; else { e = 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3ff; v = s | (uint32_t)(e + 112) << 23 | m << 13; } }
        else if (e == 31) v = s | 0x7f800000u | m << 13; else v = s | (uint32_t)(e + 112) << 23 | m << 13;
        memcpy(&f[i], &v, 4); }
    char p[1400]; snprintf(p, sizeof p, "%s/%s.f32", dir, name); FILE *fo = fopen(p, "wb");
    if (fo) { fwrite(f, 4, n, fo); fclose(fo); }
    snprintf(p, sizeof p, "%s/%s.shape", dir, name); fo = fopen(p, "w"); if (fo) { fprintf(fo, "%d %d %d %d\n", t.d, t.h, t.w, t.c); fclose(fo); }
    free(h); free(f);
}
static void dump_f32(const char *dir, const char *name, const float *dev, size_t n) {
    if (!dir) return;
    float *f = malloc(n * 4); xd2h(f, dev, n * 4);
    char p[1400]; snprintf(p, sizeof p, "%s/%s.f32", dir, name); FILE *fo = fopen(p, "wb"); if (fo) { fwrite(f, 4, n, fo); fclose(fo); }
    free(f);
}

/* y = act(LN(conv(x))) (+ res) */
static xt cln_fwd(const cln *c, xt x, int gelu, const xt *res) {
    xt y = xconv_fwd(&c->c, x, nullptr, XA_NONE);
    vw_ln(y.p, 0, c->g, c->b, EPS, gelu, res ? res->p : nullptr, y.p, 0, (long long)y.n * y.d * y.h * y.w, y.c);
    return y;
}

/* multi-head attention over qkv fp16 [T][576] (rows: seq s, token t -> s ss + t ts), output fp16 [T][192] */
static int attention(const void *qkv, void *out, int nseq, int Lt, long long ss, long long ts, int mode, int md, int side, const float *bias_all, int nbias, float *S, void *P) {
    const int Q3 = 3 * VD; const float scale = 1.f / sqrtf((float)VHD);
    for (int h = 0; h < VH; h++) {
        const void *q = (const char *)qkv + (size_t)h * VHD * 2, *k = (const char *)qkv + (size_t)(VD + h * VHD) * 2, *v = (const char *)qkv + (size_t)(2 * VD + h * VHD) * 2;
        /* S^T (Lk x Lq, col-major) = K (Lk x 32) Q^T: A = K stored col-major 32 x Lk (op T), B = Q col-major 32 x Lq */
        if (vw_gemm(1, 0, Lt, Lt, VHD, k, (int)(Q3 * ts), Q3 * ss, q, (int)(Q3 * ts), Q3 * ss, S, Lt, (long long)Lt * Lt, 1, nseq)) return -1;
        vw_softmax(S, P, (long long)nseq * Lt, Lt, mode, md, side, bias_all + (size_t)h * nbias, scale);
        /* O^T (32 x Lq) = V^T (32 x Lk) P^T (Lk x Lq) */
        if (vw_gemm(0, 0, VHD, Lt, Lt, v, (int)(Q3 * ts), Q3 * ss, P, Lt, (long long)Lt * Lt, (char *)out + (size_t)h * VHD * 2, (int)(VD * ts), VD * ss, 0, nseq)) return -1;
    }
    return 0;
}

int vw_forward(vw_net *n, const uint8_t *im, const uint8_t *va, int T, int L, float mean, float inv, float *phase_dev, const char *dump) {
    if (T % 8 || T < 8 || T / 8 > (n->tside + 1) / 2) { fprintf(stderr, "villawind: transverse size %d unsupported\n", T); return -1; }
    xt x = xt_new(1, T, T, L, 8); vw_input(im, va, mean, inv, x.p, (long long)T * T * L);
    xt sk[3];
    for (int s = 0; s < 3; s++) {
        xt a = cln_fwd(&n->st[s][0], x, 1, nullptr); xt_free(&x);
        x = cln_fwd(&n->st[s][1], a, 1, nullptr); xt_free(&a);
        sk[s] = x; char nm[32]; snprintf(nm, sizeof nm, "stage%d", s); dump_xt(dump, nm, x);
    }
    xt f = cln_fwd(&n->tin, sk[2], 1, nullptr); dump_xt(dump, "trunk_in", f);
    for (int k = 0; k < 2; k++) {
        xt a = cln_fwd(&n->stem[k][0], f, 1, nullptr);
        xt b = cln_fwd(&n->stem[k][1], a, 0, &f); xt_free(&a); xt_free(&f); f = b;
    }
    dump_xt(dump, "stem", f);
    const int t8 = T / 8; const long long NT = (long long)t8 * t8 * L;
    float *tok = xmalloc_dev((size_t)NT * VD * 4); vw_h2f(tok, f.p, NT * VD); xt_free(&f);
    xt ln = xt_new(1, 1, 1, (int)NT, VD), att = xt_new(1, 1, 1, (int)NT, VD);
    const int Lmax = L > t8 * t8 ? L : t8 * t8; const long long smax = (long long)(t8 * t8) * L * L > (long long)L * (t8 * t8) * (t8 * t8) ? (long long)(t8 * t8) * L * L : (long long)L * (t8 * t8) * (t8 * t8);
    (void)Lmax;
    float *S = xmalloc_dev((size_t)smax * 4); void *P = xmalloc_dev((size_t)smax * 2);
    if (!tok || !S || !P) { fprintf(stderr, "villawind: out of device memory\n"); return -1; }
    const int md = (n->tside - 1) / 2;
    for (int k = 0; k < VNB; k++) {
        ablock *B = &n->B[k];
        /* ray attention: sequences = columns (t8 t8), tokens along L (contiguous rows) */
        vw_ln(tok, 1, B->rnw, B->rnb, EPS, 0, nullptr, ln.p, 0, NT, VD);
        xt qkv = xconv_fwd(&B->rqkv, ln, nullptr, XA_NONE);
        if (attention(qkv.p, att.p, t8 * t8, L, L, 1, 0, VRMAX, 0, B->rbias, 2 * VRMAX + 1, S, P)) return -1;
        xt_free(&qkv);
        xt pr = xconv_fwd(&B->rproj, att, nullptr, XA_NONE); vw_add_half(tok, pr.p, NT * VD); xt_free(&pr);
        /* transverse attention: sequences = ray positions, tokens = the plane (row stride L) */
        vw_ln(tok, 1, B->tnw, B->tnb, EPS, 0, nullptr, ln.p, 0, NT, VD);
        qkv = xconv_fwd(&B->tqkv, ln, nullptr, XA_NONE);
        if (attention(qkv.p, att.p, L, t8 * t8, 1, L, 1, md, t8, B->tbias, n->tside * n->tside, S, P)) return -1;
        xt_free(&qkv);
        pr = xconv_fwd(&B->tproj, att, nullptr, XA_NONE); vw_add_half(tok, pr.p, NT * VD); xt_free(&pr);
        /* MLP */
        vw_ln(tok, 1, B->mnw, B->mnb, EPS, 0, nullptr, ln.p, 0, NT, VD);
        xt h1 = xconv_fwd(&B->fc1, ln, nullptr, XA_GELU);
        xt h2 = xconv_fwd(&B->fc2, h1, nullptr, XA_NONE); xt_free(&h1);
        vw_add_half(tok, h2.p, NT * VD); xt_free(&h2);
        if (dump) { char nm[32]; snprintf(nm, sizeof nm, "block%d", k); dump_f32(dump, nm, tok, (size_t)NT * VD); }
    }
    xfree_dev(S); xfree_dev(P); xt_free(&ln); xt_free(&att);
    f = xt_new(1, t8, t8, L, VD); vw_f2h(f.p, tok, NT * VD); xfree_dev(tok);
    xt up = xresize(f, 2 * t8, 2 * t8, L); xt_free(&f);
    xt cat = xconcat(up, sk[1]); xt_free(&up);
    xt d0 = cln_fwd(&n->dec[0], cat, 1, nullptr); xt_free(&cat);
    xt d1 = cln_fwd(&n->dec[1], d0, 1, nullptr); xt_free(&d0); dump_xt(dump, "decoder", d1);
    up = xresize(d1, T / 2, T / 2, L); xt_free(&d1);
    cat = xconcat(up, sk[0]); xt_free(&up);
    xt h0 = cln_fwd(&n->fh0, cat, 1, nullptr); xt_free(&cat); dump_xt(dump, "head0", h0);
    const size_t nh = (size_t)(T / 2) * (T / 2) * L;
    float *lg = xmalloc_dev(nh * 4 * 4); if (!lg) return -1;
    for (int c = 0; c < 4; c++) xhead(h0, n->fh3w[c], n->fh3b[c], lg + c * nh);
    xt_free(&h0); for (int s = 0; s < 3; s++) xt_free(&sk[s]);
    vw_phase(lg, T / 2, T / 2, L, phase_dev, nullptr);
    if (dump) dump_f32(dump, "logits", lg, nh * 4);
    xfree_dev(lg);
    if (dump) dump_f32(dump, "phase", phase_dev, (size_t)T * T * L);
    return xnet_check("villawind forward");
}
