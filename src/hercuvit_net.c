/* HercuViT network (src/hercuvit.h): weights from checkpoint_latest.pth ("model" entry), the window forward, the host-side
   Mask2Former query decoder. Reference: hercunet/hercuvit/model/{decoupled_pixel_encoder,decoupled_mask_decoder,decoder_common,
   resblocks}.py (jimmylomro/hercUNet de4acb7). */
#include "hercuvit.h"
#include "dinovol.h"
#include "wload.h"
#include "xnet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { HD = 32, HH = 4, HHD = 8, HMLP = 128, HQ = 64, HL = 6, HMAXN = 16 * 16 * 16 };

typedef struct { xconv c; float *g, *b; } cin;                 /* conv + InstanceNorm affine (device) */
typedef struct { cin c1, c2, sk; int has_sk; } rblock;
typedef struct { float qw[HD * HD], qb[HD], kw[HD * HD], kb[HD], vw[HD * HD], vb[HD], ow[HD * HD], ob[HD], qnw[HHD], qnb[HHD], knw[HHD], knb[HHD]; } hattn;
typedef struct { hattn cross, self; float n1w[HD], n1b[HD], n2w[HD], n2b[HD], n3w[HD], n3b[HD], m0w[HMLP * HD], m0b[HMLP], m2w[HD * HMLP], m2b[HD]; } hlayer;

struct hv_net {
    dv_net *enc;
    xconv up[3]; float *lnw[3], *lnb[3];
    rblock rb[3][2];
    float *luw, *lub;                          /* LayerNorm on U1 */
    float *sw; float sb;                       /* surface margin head: w1 - w0, b1 - b0 */
    /* query decoder (host fp32) */
    float qfeat[HQ * HD], qpos[HQ * HD], lvl[3 * HD];
    hlayer L[HL];
    float e0w[HD * HD], e0b[HD], e2w[HD * HD], e2b[HD], e4w[HD * HD], e4b[HD], menw[HD], menb[HD], dnw[HD], dnb[HD];
    /* window state */
    int T; xt u1; float *margin, *me_dev, *pool; float me[HQ * HD];
};

static const wl_tensor *need(const wl_dict *d, const char *name) {
    const wl_tensor *t = wl_get(d, name);
    if (!t) fprintf(stderr, "hercuvit: tensor %s missing\n", name);
    return t;
}
static float *devf(const wl_dict *d, const char *name) { const wl_tensor *t = need(d, name); return t ? xdev_floats(t->data, t->numel) : nullptr; }
static int hostf(const wl_dict *d, const char *name, float *dst, int64_t n) {
    const wl_tensor *t = need(d, name); if (!t) return -1;
    if (t->numel != n) { fprintf(stderr, "hercuvit: %s has %lld values, expected %lld\n", name, (long long)t->numel, (long long)n); return -1; }
    memcpy(dst, t->data, (size_t)n * 4); return 0;
}
static int cin_of(const wl_dict *d, cin *c, const char *pfx, int k, int bias) {
    char w[256], b[256], g[256], be[256];
    snprintf(w, sizeof w, "%s.0.weight", pfx); snprintf(b, sizeof b, "%s.0.bias", pfx); snprintf(g, sizeof g, "%s.1.weight", pfx); snprintf(be, sizeof be, "%s.1.bias", pfx);
    const wl_tensor *wt = need(d, w), *bt = bias ? need(d, b) : nullptr; if (!wt || (bias && !bt)) return -1;
    if (xconv_init(&c->c, wt->data, (int)wt->shape[0], (int)wt->shape[1], k, k, k, 1, 1, 1, k / 2, k / 2, k / 2, nullptr, bt ? bt->data : nullptr)) return -1;
    c->g = devf(d, g); c->b = devf(d, be);
    return c->g && c->b ? 0 : -1;
}
static int attn_of(const wl_dict *d, hattn *a, const char *pfx) {
    char p[256]; int rc = 0;
#define AF(field, nm, n) (snprintf(p, sizeof p, "%s." nm, pfx), rc |= hostf(d, p, a->field, n))
    AF(qw, "q_proj.weight", HD * HD); AF(qb, "q_proj.bias", HD); AF(kw, "k_proj.weight", HD * HD); AF(kb, "k_proj.bias", HD);
    AF(vw, "v_proj.weight", HD * HD); AF(vb, "v_proj.bias", HD); AF(ow, "out_proj.weight", HD * HD); AF(ob, "out_proj.bias", HD);
    AF(qnw, "q_norm.weight", HHD); AF(qnb, "q_norm.bias", HHD); AF(knw, "k_norm.weight", HHD); AF(knb, "k_norm.bias", HHD);
#undef AF
    return rc;
}

hv_net *hv_load(const char *pth) {
    wl_dict *d = wl_load(pth, "model");
    if (!d) { fprintf(stderr, "hercuvit: %s: %s\n", pth, wl_error()); return nullptr; }
    hv_net *n = calloc(1, sizeof *n); int rc = 0;
    n->enc = dv_load_dict(d, pth);
    if (!n->enc) rc = -1;
    else if (dv_embed_dim(n->enc) != 288) { fprintf(stderr, "hercuvit: encoder width %d (expected 288)\n", dv_embed_dim(n->enc)); rc = -1; }
    for (int s = 0; s < 3 && !rc; s++) {
        char p[256], q[256];
        snprintf(p, sizeof p, "pixel_decoder.up%d.weight", s); snprintf(q, sizeof q, "pixel_decoder.up%d.bias", s);
        const wl_tensor *w = need(d, p), *b = need(d, q);
        if (!w || !b || w->ndim != 5) { rc = -1; break; }
        rc |= xtconv3_init(&n->up[s], w->data, (int)w->shape[0], (int)w->shape[1], b->data);
        snprintf(p, sizeof p, "pixel_decoder.ln%d.weight", s); n->lnw[s] = devf(d, p);
        snprintf(p, sizeof p, "pixel_decoder.ln%d.bias", s); n->lnb[s] = devf(d, p);
        rc |= !n->lnw[s] || !n->lnb[s];
        for (int k = 0; k < 2; k++) {
            rblock *R = &n->rb[s][k];
            snprintf(p, sizeof p, "pixel_decoder.block%d.blocks.%d.conv1", s, k); rc |= cin_of(d, &R->c1, p, 3, 1);
            snprintf(p, sizeof p, "pixel_decoder.block%d.blocks.%d.conv2", s, k); rc |= cin_of(d, &R->c2, p, 3, 1);
            snprintf(p, sizeof p, "pixel_decoder.block%d.blocks.%d.skip.0.0.weight", s, k);
            R->has_sk = wl_get(d, p) != nullptr;
            if (R->has_sk) { snprintf(p, sizeof p, "pixel_decoder.block%d.blocks.%d.skip.0", s, k); rc |= cin_of(d, &R->sk, p, 1, 0); }
        }
    }
    n->luw = devf(d, "pixel_decoder.ln_u1.weight"); n->lub = devf(d, "pixel_decoder.ln_u1.bias"); rc |= !n->luw || !n->lub;
    {
        const wl_tensor *w = need(d, "pixel_decoder.surf_head.weight"), *b = need(d, "pixel_decoder.surf_head.bias");
        if (!w || !b || w->shape[0] != 2 || w->shape[1] != HD) rc = -1;
        else { float dw[HD]; for (int c = 0; c < HD; c++) dw[c] = w->data[HD + c] - w->data[c]; n->sw = xdev_floats(dw, HD); n->sb = b->data[1] - b->data[0]; rc |= !n->sw; }
    }
    rc |= hostf(d, "instance_head.query_feat.weight", n->qfeat, HQ * HD) | hostf(d, "instance_head.query_pos.weight", n->qpos, HQ * HD) |
          hostf(d, "instance_head.mem_level_embed", n->lvl, 3 * HD);
    for (int l = 0; l < HL; l++) {
        char p[256]; hlayer *Y = &n->L[l];
        snprintf(p, sizeof p, "instance_head.layers.%d.cross", l); rc |= attn_of(d, &Y->cross, p);
        snprintf(p, sizeof p, "instance_head.layers.%d.self_attn", l); rc |= attn_of(d, &Y->self, p);
#define LF(field, nm, cnt) (snprintf(p, sizeof p, "instance_head.layers.%d." nm, l), rc |= hostf(d, p, Y->field, cnt))
        LF(n1w, "n1.weight", HD); LF(n1b, "n1.bias", HD); LF(n2w, "n2.weight", HD); LF(n2b, "n2.bias", HD); LF(n3w, "n3.weight", HD); LF(n3b, "n3.bias", HD);
        LF(m0w, "mlp.0.weight", HMLP * HD); LF(m0b, "mlp.0.bias", HMLP); LF(m2w, "mlp.2.weight", HD * HMLP); LF(m2b, "mlp.2.bias", HD);
#undef LF
    }
    if (wl_get(d, "instance_head.layers.6.n1.weight")) { fprintf(stderr, "hercuvit: more than %d decoder layers\n", HL); rc = -1; }
    rc |= hostf(d, "instance_head.mask_embed.0.weight", n->e0w, HD * HD) | hostf(d, "instance_head.mask_embed.0.bias", n->e0b, HD) |
          hostf(d, "instance_head.mask_embed.2.weight", n->e2w, HD * HD) | hostf(d, "instance_head.mask_embed.2.bias", n->e2b, HD) |
          hostf(d, "instance_head.mask_embed.4.weight", n->e4w, HD * HD) | hostf(d, "instance_head.mask_embed.4.bias", n->e4b, HD) |
          hostf(d, "instance_head.me_norm.weight", n->menw, HD) | hostf(d, "instance_head.me_norm.bias", n->menb, HD) |
          hostf(d, "instance_head.decoder_norm.weight", n->dnw, HD) | hostf(d, "instance_head.decoder_norm.bias", n->dnb, HD);
    wl_free(d);
    n->me_dev = xmalloc_dev(HQ * HD * 4); n->pool = xmalloc_dev((size_t)HQ * HMAXN * 4);
    if (rc || !n->me_dev || !n->pool || xnet_check("hercuvit load")) { fprintf(stderr, "hercuvit: cannot load %s\n", pth); hv_free(n); return nullptr; }
    return n;
}
static void cin_free(cin *c) { xconv_free(&c->c); xfree_dev(c->g); xfree_dev(c->b); }
void hv_free(hv_net *n) {
    if (!n) return;
    dv_free(n->enc);
    for (int s = 0; s < 3; s++) {
        xconv_free(&n->up[s]); xfree_dev(n->lnw[s]); xfree_dev(n->lnb[s]);
        for (int k = 0; k < 2; k++) { cin_free(&n->rb[s][k].c1); cin_free(&n->rb[s][k].c2); if (n->rb[s][k].has_sk) cin_free(&n->rb[s][k].sk); }
    }
    xfree_dev(n->luw); xfree_dev(n->lub); xfree_dev(n->sw);
    xt_free(&n->u1); xfree_dev(n->margin); xfree_dev(n->me_dev); xfree_dev(n->pool);
    free(n);
}
int hv_queries(const hv_net *n) { (void)n; return HQ; }
const float *hv_margin(const hv_net *n) { return n->margin; }
const void *hv_u1(const hv_net *n) { return n->u1.p; }
const float *hv_embed(const hv_net *n) { return n->me_dev; }
const float *hv_embed_host(const hv_net *n) { return n->me; }
const void *hv_tokens(const hv_net *n) { return dv_tokens(n->enc); }
const void *hv_skip(const hv_net *n, int k, int *edge, int *ch) { return dv_skip(n->enc, k, edge, ch); }

/* BasicBlockD: LReLU(IN(conv2(LReLU(IN(conv1 x)))) + skip(x)), skip = IN(conv1x1 x) or x */
static xt rblock_fwd(const rblock *R, xt x) {
    xt h = xconv_fwd(&R->c1.c, x, nullptr, XA_NONE);
    xgn(h, h.c, R->c1.g, R->c1.b, 1e-5f, nullptr, XA_LRELU, h);
    xt h2 = xconv_fwd(&R->c2.c, h, nullptr, XA_NONE); xt_free(&h);
    xt r = x;
    if (R->has_sk) { r = xconv_fwd(&R->sk.c, x, nullptr, XA_NONE); xgn(r, r.c, R->sk.g, R->sk.b, 1e-5f, nullptr, XA_NONE, r); }
    xgn(h2, h2.c, R->c2.g, R->c2.b, 1e-5f, &r, XA_LRELU, h2);
    if (R->has_sk) xt_free(&r);
    return h2;
}

/* ---- host query decoder ---- */
static void ln(const float *x, const float *w, const float *b, int C, float *y) {
    double m = 0, v = 0; for (int c = 0; c < C; c++) m += x[c]; m /= C;
    for (int c = 0; c < C; c++) { v += (x[c] - m) * (x[c] - m); }
    v /= C;
    const double r = 1.0 / sqrt(v + 1e-5);
    for (int c = 0; c < C; c++) y[c] = (float)((x[c] - m) * r) * w[c] + b[c];
}
static void lin(const float *x, const float *W, const float *b, int in, int out, float *y) {   /* W [out][in] */
    for (int o = 0; o < out; o++) { float s = b ? b[o] : 0.f; const float *w = W + (size_t)o * in; for (int i = 0; i < in; i++) s += w[i] * x[i]; y[o] = s; }
}
/* QKNormAttention: q [Lq][32], k / v [Lk][32] (already the inputs of the projections); keep [Lq][Lk] (nullptr: all) */
static void attention(const hattn *A, const float *qin, int Lq, const float *kin, const float *vin, int Lk, const uint8_t *keep, float *out, float *K, float *V, float *sc) {
    float Qp[HQ * HD];
    for (int i = 0; i < Lq; i++) { lin(qin + i * HD, A->qw, A->qb, HD, HD, Qp + i * HD); for (int h = 0; h < HH; h++) ln(Qp + i * HD + h * HHD, A->qnw, A->qnb, HHD, Qp + i * HD + h * HHD); }
    for (int j = 0; j < Lk; j++) {
        lin(kin + (size_t)j * HD, A->kw, A->kb, HD, HD, K + (size_t)j * HD); for (int h = 0; h < HH; h++) ln(K + (size_t)j * HD + h * HHD, A->knw, A->knb, HHD, K + (size_t)j * HD + h * HHD);
        lin(vin + (size_t)j * HD, A->vw, A->vb, HD, HD, V + (size_t)j * HD);
    }
    const float scale = 1.f / sqrtf((float)HHD);
    for (int i = 0; i < Lq; i++) {
        float o[HD] = {0};
        const uint8_t *kp = keep ? keep + (size_t)i * Lk : nullptr;
        for (int h = 0; h < HH; h++) {
            const float *q = Qp + i * HD + h * HHD; float mx = -INFINITY;
            for (int j = 0; j < Lk; j++) {
                if (kp && !kp[j]) { sc[j] = -INFINITY; continue; }
                const float *k = K + (size_t)j * HD + h * HHD; float s = 0.f; for (int c = 0; c < HHD; c++) s += q[c] * k[c];
                sc[j] = s * scale; if (sc[j] > mx) mx = sc[j];
            }
            double den = 0; for (int j = 0; j < Lk; j++) { const float e = sc[j] == -INFINITY ? 0.f : expf(sc[j] - mx); sc[j] = e; den += e; }
            for (int j = 0; j < Lk; j++) if (sc[j] != 0.f) { const float p = (float)(sc[j] / den); const float *v = V + (size_t)j * HD + h * HHD; for (int c = 0; c < HHD; c++) o[h * HHD + c] += p * v[c]; }
        }
        lin(o, A->ow, A->ob, HD, HD, out + i * HD);
    }
}
static void predict_me(const hv_net *n, const float *q, float *me) {
    for (int i = 0; i < HQ; i++) {
        float a[HD], b[HD];
        ln(q + i * HD, n->dnw, n->dnb, HD, a);
        lin(a, n->e0w, n->e0b, HD, HD, b); for (int c = 0; c < HD; c++) b[c] = fmaxf(b[c], 0.f);
        lin(b, n->e2w, n->e2b, HD, HD, a); for (int c = 0; c < HD; c++) a[c] = fmaxf(a[c], 0.f);
        lin(a, n->e4w, n->e4b, HD, HD, b);
        ln(b, n->menw, n->menb, HD, me + i * HD);
    }
}
static void sine_pe(int s, float *pe) {   /* _sine_pe_3d((s,s,s), 32): per axis 10 = 5 sin + 5 cos, 2 zero pad; raster z y x */
    const int per = HD / 3, nf = per / 2;
    for (int z = 0; z < s; z++) for (int y = 0; y < s; y++) for (int x = 0; x < s; x++) {
        float *p = pe + ((size_t)(z * s + y) * s + x) * HD; const int cc[3] = {z, y, x};
        for (int a = 0; a < 3; a++) {
            const float co = (float)((cc[a] + 0.5f) / s * 2 - 1);
            for (int f = 0; f < nf; f++) { const float fr = (float)(1.0 / pow(10000.0, 2.0 * f / per)), ang = co * fr; p[a * per + f] = sinf(ang); p[a * per + nf + f] = cosf(ang); }
        }
        for (int c = 3 * per; c < HD; c++) p[c] = 0.f;
    }
}
static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu, x;
    if (e == 0) { if (!m) x = s; else { while (!(m & 0x400u)) { m <<= 1; e--; } e++; m &= 0x3ffu; x = s | ((uint32_t)(e + 112) << 23) | (m << 13); } }
    else if (e == 31) x = s | 0x7f800000u | (m << 13);
    else x = s | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &x, 4); return f;
}

static int decode(hv_net *n) {
    const int T = n->T, sides[3] = {T / 32, T / 16, T / 8};
    int N[3], Ntot = 0;
    for (int l = 0; l < 3; l++) { N[l] = sides[l] * sides[l] * sides[l]; Ntot += N[l]; }
    if (N[2] > HMAXN) { fprintf(stderr, "hercuvit: window %d too large for the query decoder (max 128)\n", T); return -1; }
    float *mem = malloc((size_t)Ntot * HD * 4), *memk = malloc((size_t)Ntot * HD * 4), *K = malloc((size_t)N[2] * HD * 4), *V = malloc((size_t)N[2] * HD * 4),
          *sc = malloc((size_t)N[2] * 4), *pool = malloc((size_t)HQ * N[2] * 4);
    uint8_t *keep = malloc((size_t)HQ * N[2]); uint16_t *hb = malloc((size_t)N[2] * HD * 2);
    if (!mem || !memk || !K || !V || !sc || !pool || !keep || !hb) { fprintf(stderr, "hercuvit: out of memory\n"); return -1; }
    int base[3]; for (int l = 0, o = 0; l < 3; l++) { base[l] = o; o += N[l]; }
    for (int l = 0; l < 3; l++) {   /* memory = trilinear U1 at 1/32, 1/16, 1/8; keys + sine PE + level embedding */
        xt r = xresize(n->u1, sides[l], sides[l], sides[l]);
        xd2h(hb, r.p, (size_t)N[l] * HD * 2); xt_free(&r);
        float *m = mem + (size_t)base[l] * HD, *mk = memk + (size_t)base[l] * HD;
        for (size_t i = 0; i < (size_t)N[l] * HD; i++) m[i] = h2f(hb[i]);
        sine_pe(sides[l], mk);
        for (int j = 0; j < N[l]; j++) for (int c = 0; c < HD; c++) mk[(size_t)j * HD + c] += m[(size_t)j * HD + c] + n->lvl[l * HD + c];
    }
    float q[HQ * HD], qp[HQ * HD], tmp[HQ * HD];
    memcpy(q, n->qfeat, sizeof q);
    predict_me(n, q, n->me);
    for (int li = 0; li < HL; li++) {
        const int lv = li % 3, s = sides[lv], Nl = N[lv];
        xh2d(n->me_dev, n->me, sizeof n->me);
        hv_pool_max(n->u1.p, T, n->me_dev, HQ, s, n->pool);
        xd2h(pool, n->pool, (size_t)HQ * Nl * 4);
        for (int i = 0; i < HQ; i++) {   /* keep = pooled logit > 0; a query with nothing kept attends everywhere */
            int any = 0; for (int j = 0; j < Nl; j++) any |= (keep[(size_t)i * Nl + j] = pool[(size_t)i * Nl + j] > 0.f);
            if (!any) memset(keep + (size_t)i * Nl, 1, Nl);
        }
        const hlayer *Y = &n->L[li];
        for (int i = 0; i < HQ; i++) { ln(q + i * HD, Y->n1w, Y->n1b, HD, tmp + i * HD); for (int c = 0; c < HD; c++) qp[i * HD + c] = tmp[i * HD + c] + n->qpos[i * HD + c]; }
        float o[HQ * HD];
        attention(&Y->cross, qp, HQ, memk + (size_t)base[lv] * HD, mem + (size_t)base[lv] * HD, Nl, keep, o, K, V, sc);
        for (int i = 0; i < HQ * HD; i++) q[i] += o[i];
        for (int i = 0; i < HQ; i++) { ln(q + i * HD, Y->n2w, Y->n2b, HD, tmp + i * HD); for (int c = 0; c < HD; c++) qp[i * HD + c] = tmp[i * HD + c] + n->qpos[i * HD + c]; }
        attention(&Y->self, qp, HQ, qp, tmp, HQ, nullptr, o, K, V, sc);
        for (int i = 0; i < HQ * HD; i++) q[i] += o[i];
        for (int i = 0; i < HQ; i++) {
            float a[HD], h[HMLP], b[HD];
            ln(q + i * HD, Y->n3w, Y->n3b, HD, a);
            lin(a, Y->m0w, Y->m0b, HD, HMLP, h); for (int c = 0; c < HMLP; c++) h[c] = 0.5f * h[c] * (1.f + erff(h[c] * 0.70710678118654752f));
            lin(h, Y->m2w, Y->m2b, HMLP, HD, b);
            for (int c = 0; c < HD; c++) q[i * HD + c] += b[c];
        }
        predict_me(n, q, n->me);
    }
    xh2d(n->me_dev, n->me, sizeof n->me);
    free(mem); free(memk); free(K); free(V); free(sc); free(pool); free(keep); free(hb);
    return xnet_check("hercuvit decoder");
}

int hv_forward(hv_net *n, const uint8_t *u8_dev, int T, float lo, float hi, float center, float scale, int inst) {
    if (T % 32 || T < 64) { fprintf(stderr, "hercuvit: window %d must be a multiple of 32 (>= 64)\n", T); return -1; }
    xt_free(&n->u1);
    if (n->T != T) { xfree_dev(n->margin); n->margin = xmalloc_dev((size_t)T * T * T * 4); n->T = T; if (!n->margin) return -1; }
    if (dv_forward_norm(n->enc, u8_dev, T, lo, hi, center, scale)) return -1;
    const int g = T / 8;
    xt x = {(void *)dv_tokens(n->enc), 1, g, g, g, dv_embed_dim(n->enc)};
    int owned = 0;
    for (int s = 0; s < 3; s++) {
        int e, ch; const void *sp = dv_skip(n->enc, 2 - s, &e, &ch);
        xt sk = {(void *)sp, 1, e, e, e, ch};
        xt u = xconv_fwd(&n->up[s], x, nullptr, XA_NONE);
        if (owned) xt_free(&x);
        if (u.d != e || u.c != ch) { fprintf(stderr, "hercuvit: decoder stage %d: %d^3 x %d vs skip %d^3 x %d\n", s, u.d, u.c, e, ch); return -1; }
        dv_layernorm(u.p, n->lnw[s], n->lnb[s], 1e-6f, nullptr, 0, u.p, (long long)e * e * e, ch);
        xt c = xconcat(u, sk); xt_free(&u);
        xt y = rblock_fwd(&n->rb[s][0], c); xt_free(&c);
        x = rblock_fwd(&n->rb[s][1], y); xt_free(&y); owned = 1;
    }
    dv_layernorm(x.p, n->luw, n->lub, 1e-6f, nullptr, 0, x.p, (long long)T * T * T, x.c);
    n->u1 = x;
    xhead(n->u1, n->sw, n->sb, n->margin);
    if (xnet_check("hercuvit forward")) return -1;
    return inst ? decode(n) : 0;
}

/* ---- villa normalize_robust on a uint8 window, from its histogram (NumPy linear percentiles, even-count medians) ---- */
static double os_at(const int64_t *cum, int64_t k) { int v = 0; while (cum[v] <= k) v++; return v; }   /* k-th smallest (0-based) */
void hv_robust_stats(const uint8_t *u, size_t n, double *lo, double *hi, double *center, double *scale) {
    int64_t h[256] = {0}, cum[256];
    for (size_t i = 0; i < n; i++) h[u[i]]++;
    int64_t c = 0; double s1 = 0, s2 = 0; int vmin = 255, vmax = 0;
    for (int v = 0; v < 256; v++) { c += h[v]; cum[v] = c; s1 += (double)v * h[v]; s2 += (double)v * v * h[v]; if (h[v]) { if (v < vmin) vmin = v; if (v > vmax) vmax = v; } }
    const int64_t N = (int64_t)n;
    double pct[2]; const double ps[2] = {1.0, 99.0};
    for (int k = 0; k < 2; k++) {
        const double idx = ps[k] / 100.0 * (double)(N - 1); const int64_t i0 = (int64_t)floor(idx); const double t = idx - i0;
        const double a = os_at(cum, i0), b = os_at(cum, i0 + 1 < N ? i0 + 1 : i0);
        pct[k] = a + (b - a) * t;
    }
    const double L = pct[0], H = pct[1];
#define CL(v) ((v) < L ? L : (v) > H ? H : (v))
    const double med = N % 2 ? CL(os_at(cum, N / 2)) : 0.5 * (CL(os_at(cum, N / 2 - 1)) + CL(os_at(cum, N / 2)));
    /* MAD: the deviations |clip(v) - med| per value, sorted (with counts) */
    double dv[256]; int64_t dc[256]; int nd = 0;
    for (int v = 0; v < 256; v++) if (h[v]) { dv[nd] = fabs(CL((double)v) - med); dc[nd] = h[v]; nd++; }
#undef CL
    for (int i = 1; i < nd; i++) { const double kv = dv[i]; const int64_t kc = dc[i]; int j = i - 1; while (j >= 0 && dv[j] > kv) { dv[j + 1] = dv[j]; dc[j + 1] = dc[j]; j--; } dv[j + 1] = kv; dc[j + 1] = kc; }
    double mad;
    {
        int64_t k1 = N % 2 ? N / 2 : N / 2 - 1, k2 = N / 2; double a = 0, b = 0; int64_t acc = 0;
        for (int i = 0; i < nd; i++) { if (acc <= k1 && k1 < acc + dc[i]) a = dv[i]; if (acc <= k2 && k2 < acc + dc[i]) b = dv[i]; acc += dc[i]; }
        mad = 0.5 * (a + b);
    }
    double sm = 1.4826 * mad;
    if (!(sm >= 1e-6)) {   /* fallback ladder: pre-clip std, half the percentile span, 1 */
        const double mean = s1 / (double)N, sd = sqrt(fmax(s2 / (double)N - mean * mean, 0.0));
        sm = sd >= 1e-6 ? sd : fabs(H - L) / 2.0 >= 1e-6 ? fabs(H - L) / 2.0 : 1.0;
    }
    (void)vmin; (void)vmax;
    *lo = L; *hi = H; *center = med; *scale = sm;
}
