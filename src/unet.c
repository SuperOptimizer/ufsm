#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_prof = -1;
static const char *g_names[8] = {"conv_fwd", "conv_bwd_data", "conv_bwd_w", "gn", "elementwise", "up/concat", "upload+loss+opt", ""};
#define PROF(k, call) do { if (g_prof < 0) g_prof = getenv("UFSM_PROF") != nullptr; if (g_prof) { nn_prof_begin(k); call; nn_prof_end(); } else { call; } } while (0)
void unet_prof_report(void) { if (g_prof > 0) { double ms[8]; nn_prof_collect(ms, 8); double tot = 0; for (int i = 0; i < 7; i++) tot += ms[i]; for (int i = 0; i < 7; i++) if (ms[i] > 0) fprintf(stderr, "  %-14s %8.1f ms  %4.1f%%\n", g_names[i], ms[i], 100 * ms[i] / tot); } }

typedef struct { int cin, cout, k, stride; size_t w, b; } convp;      /* offsets into the flat param array */
typedef struct { int c; size_t gamma, beta; } gnp;

typedef struct {
    convp c1, c2;
    gnp n1, n2;
    /* stored activations: conv outputs a1, a2 (GroupNorm inputs), GN stats, and the block output s2.
       gn(a) and silu(gn(a1)) are recomputed in backward. */
    const float *in, *in2;      /* in2: second input tensor for channels >= c_split (decoder skip), tensor-core path only */
    int c_split;
    float *a1, *a2, *s2;
    float *m1, *r1, *m2, *r2;
    shape5 xs, ys;
} block;

struct unet {
    unet_cfg cfg;
    size_t np;
    float *p, *g, *m, *v, *ema;   /* device, flat */
    float *live;                  /* p while training; swapped with ema in unet_use_ema */
    float *fw, *wm;               /* weights the forward/backward read: live, or wm = 2:4-masked copy of live when sparse24 */
    int sparse24, wq;             /* wq: 0, 8 or 4 = the 3^3 conv weights live in packed fp8 / fp4 storage; p/ema hold dequantized shadows */
    unsigned char *wq_q, *wq_sc, *wq_qe, *wq_sce;   /* packed weights + block scales for live params and for the EMA (byte offsets below) */
    size_t wq_qoff[64], wq_soff[64]; int wq_n; unsigned ema_step;   /* per 3^3 conv (for_each_conv3 order): offsets into the packed arrays */
    int using_ema;
    block enc[UNET_MAXLEV], dec[UNET_MAXLEV];
    convp down[UNET_MAXLEV], head;
    /* activations */
    shape5 xs;                    /* shape the buffers were built for */
    int built, train;
    shape5 ls[UNET_MAXLEV];       /* spatial shape at each level (n, -, d, h, w) */
    float *downo[UNET_MAXLEV];    /* down conv outputs */
    float *cat[UNET_MAXLEV];                 /* decoder input: [upsampled w[i+1] | skip w[i]] */
    float *xin;                              /* bf16 copy of the network input (act-bf16 mode) */
    float *glb;                              /* bf16 copy of the logit gradient (grad-bf16 mode) */
    int mode;                                /* kernel/storage mode the activations were built for */
    float *logits;
    float *gA[UNET_MAXLEV], *gB[UNET_MAXLEV];        /* grad buffers, widest tensor at the level */
    float *t1[UNET_MAXLEV], *t2[UNET_MAXLEV];        /* recompute scratch, w[i] wide */
    float *gskip[UNET_MAXLEV], *gout[UNET_MAXLEV];
    float *gn_scratch, *conv_scratch, *red_scratch;
    size_t conv_scratch_n, act_bytes;
};

static int G_of(const unet *u, int c) { return u->cfg.G < c ? u->cfg.G : c; }

/* ---- parameter layout ---- */
static size_t add_conv(unet *u, convp *c, int cin, int cout, int k, int stride, size_t off) {
    c->cin = cin; c->cout = cout; c->k = k; c->stride = stride;
    c->w = off; off += (size_t)cout * cin * k * k * k;
    c->b = off; off += (size_t)cout;
    return off;
}
static size_t add_gn(unet *u, gnp *n, int c, size_t off) { n->c = c; n->gamma = off; off += (size_t)c; n->beta = off; off += (size_t)c; return off; }
static size_t add_block(unet *u, block *b, int cin, int cout, size_t off) {
    off = add_conv(u, &b->c1, cin, cout, 3, 1, off);
    off = add_gn(u, &b->n1, cout, off);
    off = add_conv(u, &b->c2, cout, cout, 3, 1, off);
    off = add_gn(u, &b->n2, cout, off);
    return off;
}

unet *unet_create(const unet_cfg *cfg) {
    unet *u = calloc(1, sizeof *u);
    u->cfg = *cfg;
    int L = cfg->nlev;
    const int *w = cfg->widths;
    size_t off = 0;
    for (int i = 0; i < L; i++) off = add_block(u, &u->enc[i], i == 0 ? cfg->cin : w[i - 1], w[i], off);
    for (int i = 0; i < L - 1; i++) off = add_conv(u, &u->down[i], w[i], w[i], 3, 2, off);
    for (int i = L - 2; i >= 0; i--) off = add_block(u, &u->dec[i], w[i] + w[i + 1], w[i], off);
    off = add_conv(u, &u->head, w[0], cfg->cout, 1, 1, off);
    u->np = off;
    if (getenv("UFSM_DEBUG")) {
        for (int i = 0; i < L; i++) fprintf(stderr, "enc%d c1.w %zu c1.b %zu n1 %zu c2.w %zu c2.b %zu n2 %zu\n", i, u->enc[i].c1.w, u->enc[i].c1.b, u->enc[i].n1.gamma, u->enc[i].c2.w, u->enc[i].c2.b, u->enc[i].n2.gamma);
        for (int i = 0; i < L - 1; i++) fprintf(stderr, "down%d w %zu b %zu\n", i, u->down[i].w, u->down[i].b);
        for (int i = L - 2; i >= 0; i--) fprintf(stderr, "dec%d c1.w %zu c1.b %zu n1 %zu c2.w %zu c2.b %zu n2 %zu\n", i, u->dec[i].c1.w, u->dec[i].c1.b, u->dec[i].n1.gamma, u->dec[i].c2.w, u->dec[i].c2.b, u->dec[i].n2.gamma);
        fprintf(stderr, "head w %zu b %zu total %zu\n", u->head.w, u->head.b, off);
    }
    u->p = nn_malloc(off * 4); u->g = nn_malloc(off * 4); u->m = nn_malloc(off * 4); u->v = nn_malloc(off * 4); u->ema = nn_malloc(off * 4);
    nn_zero(u->g, off * 4); nn_zero(u->m, off * 4); nn_zero(u->v, off * 4);
    u->live = u->p; u->fw = u->p; u->wm = nullptr; u->sparse24 = 0; u->wq = 0; u->wq_q = u->wq_sc = u->wq_qe = u->wq_sce = nullptr; u->wq_n = 0;
    u->gn_scratch = nn_malloc(1 << 20);
    u->red_scratch = nn_malloc(4096 * 4);
    return u;
}

static void free_acts(unet *u);
void unet_free(unet *u) {
    if (!u) return;
    free_acts(u);
    nn_free(u->p); nn_free(u->g); nn_free(u->m); nn_free(u->v); nn_free(u->ema);
    nn_free(u->gn_scratch); nn_free(u->red_scratch);
    free(u);
}

static void free_acts(unet *u);
const unet_cfg *unet_cfg_of(const unet *u) { return &u->cfg; }
size_t unet_nparams(const unet *u) { return u->np; }

/* ---- init ---- */
static uint64_t sm64(uint64_t *s) { uint64_t z = (*s += 0x9e3779b97f4a7c15ull); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; return z ^ (z >> 31); }
static double unif(uint64_t *s) { return (double)(sm64(s) >> 11) * 0x1.0p-53; }
static double gauss(uint64_t *s) { double a = unif(s) + 1e-300, b = unif(s); return sqrt(-2 * log(a)) * cos(6.283185307179586 * b); }

static void init_conv(float *h, const convp *c, uint64_t *seed, double bias) {
    size_t nw = (size_t)c->cout * c->cin * c->k * c->k * c->k;
    double std = sqrt(2.0 / ((double)c->cin * c->k * c->k * c->k));
    for (size_t i = 0; i < nw; i++) h[c->w + i] = (float)(std * gauss(seed));
    for (int i = 0; i < c->cout; i++) h[c->b + i] = (float)bias;
}
static void init_gn(float *h, const gnp *n) { for (int i = 0; i < n->c; i++) { h[n->gamma + i] = 1.f; h[n->beta + i] = 0.f; } }

void unet_init(unet *u, uint64_t seed) {
    float *h = calloc(u->np, 4);
    int L = u->cfg.nlev;
    for (int i = 0; i < L; i++) { init_conv(h, &u->enc[i].c1, &seed, 0); init_gn(h, &u->enc[i].n1); init_conv(h, &u->enc[i].c2, &seed, 0); init_gn(h, &u->enc[i].n2); }
    for (int i = 0; i < L - 1; i++) init_conv(h, &u->down[i], &seed, 0);
    for (int i = 0; i < L - 1; i++) { init_conv(h, &u->dec[i].c1, &seed, 0); init_gn(h, &u->dec[i].n1); init_conv(h, &u->dec[i].c2, &seed, 0); init_gn(h, &u->dec[i].n2); }
    init_conv(h, &u->head, &seed, -2.0);
    nn_h2d(u->p, h, u->np * 4);
    nn_d2d(u->ema, u->p, u->np * 4);
    nn_zero(u->m, u->np * 4); nn_zero(u->v, u->np * 4);
    free(h);
}

/* ---- activation buffers ---- */
static float *dalloc(unet *u, size_t n) { u->act_bytes += n * 4; return nn_malloc(n * 4); }
#define ABF (nn_get_tf32() && nn_get_act_bf16())
#define GBF (ABF && nn_get_grad_bf16())
static float *dalloc_act(unet *u, size_t n) { size_t b = ABF ? n * 2 : n * 4; u->act_bytes += b; return nn_malloc(b); }   /* activation storage (bf16 in act-bf16 mode) */
static float *dalloc_grad(unet *u, size_t n) { size_t b = GBF ? n * 2 : n * 4; u->act_bytes += b; return nn_malloc(b); }  /* activation-gradient storage */

static void free_acts(unet *u) {
    for (int i = 0; i < u->cfg.nlev; i++) {
        block *bs[2] = {&u->enc[i], &u->dec[i]};
        for (int k = 0; k < 2; k++) {
            block *b = bs[k];
            nn_free(b->a1); nn_free(b->a2); nn_free(b->s2); nn_free(b->m1); nn_free(b->r1); nn_free(b->m2); nn_free(b->r2);
            b->a1 = b->a2 = b->s2 = b->m1 = b->r1 = b->m2 = b->r2 = nullptr;
        }
        nn_free(u->downo[i]); nn_free(u->cat[i]); nn_free(u->gskip[i]); nn_free(u->gout[i]); nn_free(u->gA[i]); nn_free(u->gB[i]); nn_free(u->t1[i]); nn_free(u->t2[i]);
        u->downo[i] = u->cat[i] = u->gskip[i] = u->gout[i] = u->gA[i] = u->gB[i] = u->t1[i] = u->t2[i] = nullptr;
    }
    nn_free(u->logits); u->logits = nullptr;
    nn_free(u->conv_scratch); u->conv_scratch = nullptr; u->conv_scratch_n = 0;
    u->built = 0; u->act_bytes = 0;
}

static void build_block_acts(unet *u, block *b, shape5 xs, int train) {
    b->xs = xs;
    b->ys = xs; b->ys.c = b->c1.cout;
    size_t n = shape_numel(b->ys);
    int G = G_of(u, b->c1.cout);
    b->a1 = dalloc_act(u, n); b->a2 = dalloc_act(u, n); b->s2 = dalloc_act(u, n);
    b->m1 = dalloc(u, (size_t)xs.n * G); b->r1 = dalloc(u, (size_t)xs.n * G); b->m2 = dalloc(u, (size_t)xs.n * G); b->r2 = dalloc(u, (size_t)xs.n * G);
}

static void build_acts(unet *u, shape5 xs, int train) {
    free_acts(u);
    int L = u->cfg.nlev;
    const int *w = u->cfg.widths;
    shape5 s = xs;
    for (int i = 0; i < L; i++) {
        u->ls[i] = s;
        shape5 bin = s; bin.c = i == 0 ? u->cfg.cin : w[i - 1];
        build_block_acts(u, &u->enc[i], bin, train);
        if (i < L - 1) {
            shape5 ds = nn_conv3d_out_shape(s, w[i], 3, 2);
            u->downo[i] = dalloc_act(u, shape_numel(ds));
            s = ds;
        }
    }
    for (int i = L - 2; i >= 0; i--) {
        shape5 li = u->ls[i];
        size_t S = shape_spatial(li);
        u->cat[i] = dalloc_act(u, (size_t)li.n * (nn_get_tf32() ? w[i + 1] : w[i] + w[i + 1]) * S);
        shape5 cin = li; cin.c = w[i] + w[i + 1];
        build_block_acts(u, &u->dec[i], cin, train);
    }
    shape5 os = u->ls[0]; os.c = u->cfg.cout;
    u->logits = dalloc(u, shape_numel(os));
    if (train) {
        for (int i = 0; i < L; i++) {
            shape5 li = u->ls[i];
            size_t S = shape_spatial(li);
            int wmax = i < L - 1 ? w[i] + w[i + 1] : w[i];
            if (i == 0 && u->cfg.cin > wmax) wmax = u->cfg.cin;
            size_t nb = (size_t)li.n * wmax * S, nw = (size_t)li.n * w[i] * S;
            u->gA[i] = dalloc_grad(u, nb); u->gB[i] = dalloc_grad(u, nb);
            if (!nn_get_tf32()) { u->t1[i] = dalloc(u, nw); u->t2[i] = dalloc(u, nw); }
            u->gskip[i] = dalloc_grad(u, nw); u->gout[i] = dalloc_grad(u, nw);
        }
        size_t cs = 0;
        for (int i = 0; i < L; i++) {
            size_t a = nn_conv3d_scratch(u->enc[i].xs, w[i], 3), b = nn_conv3d_scratch(u->enc[i].ys, w[i], 3);
            if (a > cs) cs = a;
            if (b > cs) cs = b;
            if (i < L - 1) { size_t c = nn_conv3d_scratch(u->dec[i].xs, w[i], 3); if (c > cs) cs = c; }
        }
        u->conv_scratch = nn_malloc(cs); u->conv_scratch_n = cs; u->act_bytes += cs;
    }
    u->xin = ABF ? dalloc_act(u, shape_numel(xs)) : nullptr;
    u->glb = (train && GBF) ? dalloc_grad(u, shape_numel(os)) : nullptr;
    u->xs = xs; u->built = 1; u->train = train; u->mode = nn_get_tf32() * 4 + ABF * 2 + GBF;
}

size_t unet_activation_bytes(const unet *u) { return u->act_bytes; }
shape5 unet_out_shape(const unet *u, shape5 xs) { xs.c = u->cfg.cout; return xs; }

/* ---- forward ---- */
static const float *P(const unet *u, size_t off) { return u->fw + off; }
void unet_apply_sparse24(unet *u);

static void block_fwd(unet *u, block *b, int level, const float *x) {
    b->in = x;
    int G = G_of(u, b->c1.cout);
    float *t1 = u->t1[level] ? u->t1[level] : b->s2;    /* inference: no scratch, s2 doubles as temp */
    if (nn_get_tf32()) {   /* conv1 yields the stats of a1; conv2 applies gn + silu to a1 while staging and yields the stats of a2 */
        if (b->in2) PROF(0, nn_conv3d_fwd_split(x, b->in2, b->c_split, b->xs, 0, nullptr, nullptr, nullptr, nullptr, P(u, b->c1.w), P(u, b->c1.b), b->c1.cout, b->a1, G, 1e-5f, b->m1, b->r1));
        else PROF(0, nn_conv3d_fwd_gn_stats(x, b->xs, 0, nullptr, nullptr, nullptr, nullptr, P(u, b->c1.w), P(u, b->c1.b), b->c1.cout, b->a1, G, 1e-5f, b->m1, b->r1));
        PROF(0, nn_conv3d_fwd_gn_stats(b->a1, b->ys, G, P(u, b->n1.gamma), P(u, b->n1.beta), b->m1, b->r1, P(u, b->c2.w), P(u, b->c2.b), b->c2.cout, b->a2, G, 1e-5f, b->m2, b->r2));
        PROF(3, nn_gn_silu_apply(b->a2, b->ys, G, P(u, b->n2.gamma), P(u, b->n2.beta), b->m2, b->r2, b->s2));
        return;
    }
    PROF(0, nn_conv3d_fwd(x, b->xs, P(u, b->c1.w), P(u, b->c1.b), b->c1.cout, 3, 1, b->a1));
    PROF(3, nn_gn_fwd_silu(b->a1, b->ys, G, 1e-5f, P(u, b->n1.gamma), P(u, b->n1.beta), t1, b->m1, b->r1));
    PROF(0, nn_conv3d_fwd(t1, b->ys, P(u, b->c2.w), P(u, b->c2.b), b->c2.cout, 3, 1, b->a2));
    PROF(3, nn_gn_fwd_silu(b->a2, b->ys, G, 1e-5f, P(u, b->n2.gamma), P(u, b->n2.beta), b->s2, b->m2, b->r2));
}

const float *unet_forward(unet *u, const float *x, shape5 xs, int train) {
    int div = 1 << (u->cfg.nlev - 1);
    if (xs.d % div || xs.h % div || xs.w % div) { fprintf(stderr, "unet: spatial size %dx%dx%d must be divisible by %d\n", xs.d, xs.h, xs.w, div); abort(); }
    if (!u->built || memcmp(&u->xs, &xs, sizeof xs) || (train && !u->train) || u->mode != nn_get_tf32() * 4 + ABF * 2 + GBF) build_acts(u, xs, train);
    unet_apply_sparse24(u);
    int L = u->cfg.nlev;
    const int *w = u->cfg.widths;
    const float *cur = x;
    if (ABF) { nn_f32_to_bf16(x, shape_numel(xs), u->xin); cur = u->xin; }
    for (int i = 0; i < L; i++) {
        nn_set_layer(i); block_fwd(u, &u->enc[i], i, cur);
        cur = u->enc[i].s2;
        if (i < L - 1) {
            nn_set_layer(4 + i); PROF(0, nn_conv3d_fwd(cur, u->enc[i].ys, P(u, u->down[i].w), P(u, u->down[i].b), w[i], 3, 2, u->downo[i]));
            cur = u->downo[i];
        }
    }
    for (int i = L - 2; i >= 0; i--) {
        shape5 src = u->ls[i + 1]; src.c = w[i + 1];
        shape5 li = u->ls[i];
        if (nn_get_tf32()) {   /* cat holds only the upsampled part; the skip is read in place by the split conv */
            PROF(5, nn_up2_fwd_into(cur, src, u->cat[i], w[i + 1], 0));
            u->dec[i].in2 = u->enc[i].s2; u->dec[i].c_split = w[i + 1];
        } else {
            PROF(5, nn_up2_fwd_into(cur, src, u->cat[i], w[i + 1] + w[i], 0));
            size_t S = shape_spatial(li);
            for (int n = 0; n < li.n; n++)
                nn_d2d(u->cat[i] + ((size_t)n * (w[i + 1] + w[i]) + w[i + 1]) * S, u->enc[i].s2 + (size_t)n * w[i] * S, (size_t)w[i] * S * 4);
            u->dec[i].in2 = nullptr;
        }
        nn_set_layer(9 - i); block_fwd(u, &u->dec[i], i, u->cat[i]);
        cur = u->dec[i].s2;
    }
    nn_set_layer(10); PROF(0, nn_conv3d_fwd(cur, u->dec[0].ys, P(u, u->head.w), P(u, u->head.b), u->cfg.cout, 1, 1, u->logits));
    if (getenv("UFSM_DEBUG") && !ABF) {
        for (int i = 0; i < L; i++) fprintf(stderr, "enc%d a1 %.4g a2 %.4g s2 %.4g%s\n", i, nn_sumsq(u->enc[i].a1, shape_numel(u->enc[i].ys), u->red_scratch), nn_sumsq(u->enc[i].a2, shape_numel(u->enc[i].ys), u->red_scratch), nn_sumsq(u->enc[i].s2, shape_numel(u->enc[i].ys), u->red_scratch), i < L - 1 ? "" : " (bottom)");
        for (int i = L - 2; i >= 0; i--) fprintf(stderr, "dec%d cat %.4g a1 %.4g a2 %.4g s2 %.4g\n", i, nn_sumsq(u->cat[i], shape_numel(u->dec[i].xs), u->red_scratch), nn_sumsq(u->dec[i].a1, shape_numel(u->dec[i].ys), u->red_scratch), nn_sumsq(u->dec[i].a2, shape_numel(u->dec[i].ys), u->red_scratch), nn_sumsq(u->dec[i].s2, shape_numel(u->dec[i].ys), u->red_scratch));
        fprintf(stderr, "logits %.4g\n", nn_sumsq(u->logits, shape_numel(u->ls[0]) / u->ls[0].c * u->cfg.cout, u->red_scratch));
    }
    return u->logits;
}

/* ---- backward ---- */
/* gy: grad wrt block output (s2). Returns gB of the level holding grad wrt block input. */
static float *block_bwd(unet *u, block *b, int level, const float *gy, float *gx2) {
    int G = G_of(u, b->c1.cout);
    size_t n = shape_numel(b->ys);
    float *A = u->gA[level], *B = u->gB[level], *t1 = u->t1[level], *t2 = u->t2[level];
    float *g = u->g;
    if (nn_get_tf32()) {   /* fused: no materialised gn / silu activations */
        PROF(3, nn_gn_silu_bwd(b->a2, b->ys, G, P(u, b->n2.gamma), P(u, b->n2.beta), b->m2, b->r2, gy, B, g + b->n2.gamma, g + b->n2.beta, u->gn_scratch));   /* B = d/d a2 */
        PROF(2, nn_conv3d_bwd_weight_gn(b->a1, b->ys, G, P(u, b->n1.gamma), P(u, b->n1.beta), b->m1, b->r1, B, b->ys, g + b->c2.w, g + b->c2.b));
        PROF(1, nn_conv3d_bwd_data(B, b->ys, P(u, b->c2.w), b->ys, 3, 1, A, u->conv_scratch));                              /* A = d/d s1 */
        PROF(3, nn_gn_silu_bwd(b->a1, b->ys, G, P(u, b->n1.gamma), P(u, b->n1.beta), b->m1, b->r1, A, A, g + b->n1.gamma, g + b->n1.beta, u->gn_scratch));    /* A = d/d a1 (in place) */
    } else {
    PROF(3, nn_gn_apply(b->a2, b->ys, G, P(u, b->n2.gamma), P(u, b->n2.beta), b->m2, b->r2, t1));            /* t1 = g2 */
    PROF(4, nn_silu_bwd(t1, gy, n, A));                                                                        /* A = d/d g2 */
    PROF(3, nn_gn_bwd(b->a2, b->ys, G, P(u, b->n2.gamma), b->m2, b->r2, A, B, g + b->n2.gamma, g + b->n2.beta, u->gn_scratch));   /* B = d/d a2 */
    PROF(3, nn_gn_apply(b->a1, b->ys, G, P(u, b->n1.gamma), P(u, b->n1.beta), b->m1, b->r1, t1));            /* t1 = g1 */
    PROF(4, nn_silu_fwd(t1, n, t2));                                                                           /* t2 = s1 */
    PROF(2, nn_conv3d_bwd_weight(t2, b->ys, B, b->ys, 3, 1, g + b->c2.w, g + b->c2.b));
    PROF(1, nn_conv3d_bwd_data(B, b->ys, P(u, b->c2.w), b->ys, 3, 1, A, u->conv_scratch));                    /* A = d/d s1 */
    PROF(4, nn_silu_bwd(t1, A, n, B));                                                                         /* B = d/d g1 */
    PROF(3, nn_gn_bwd(b->a1, b->ys, G, P(u, b->n1.gamma), b->m1, b->r1, B, A, g + b->n1.gamma, g + b->n1.beta, u->gn_scratch));   /* A = d/d a1 */
    }
    if (b->in2 && nn_get_tf32()) {
        PROF(2, nn_conv3d_bwd_weight_split(b->in, b->in2, b->c_split, b->xs, 0, nullptr, nullptr, nullptr, nullptr, A, b->ys, g + b->c1.w, g + b->c1.b));
        PROF(1, nn_conv3d_bwd_data_split(A, b->ys, P(u, b->c1.w), b->xs, B, gx2, b->c_split, u->conv_scratch));   /* B = d/d up part, gx2 = d/d skip */
        return B;
    }
    PROF(2, nn_conv3d_bwd_weight(b->in, b->xs, A, b->ys, 3, 1, g + b->c1.w, g + b->c1.b));
    PROF(1, nn_conv3d_bwd_data(A, b->ys, P(u, b->c1.w), b->xs, 3, 1, B, u->conv_scratch));            /* B = d/d in */
    return B;
}

void unet_backward(unet *u, const float *glogits) {
    int L = u->cfg.nlev;
    const float gscale = GBF ? nn_get_grad_scale() : 1.f;
    if (GBF) { shape5 os = u->ls[0]; os.c = u->cfg.cout; nn_f32_to_h16(glogits, shape_numel(os), u->glb, gscale); glogits = u->glb; }
    const int *w = u->cfg.widths;
    float *g = u->g;
    /* head */
    block *d0 = &u->dec[0];
    shape5 os = u->ls[0]; os.c = u->cfg.cout;
    nn_set_layer(10); PROF(2, nn_conv3d_bwd_weight(d0->s2, d0->ys, glogits, os, 1, 1, g + u->head.w, g + u->head.b));
    PROF(1, nn_conv3d_bwd_data(glogits, os, P(u, u->head.w), d0->ys, 1, 1, u->gout[0], u->conv_scratch));
    /* decoder, bottom-up in the graph = i from 0 to L-2 */
    for (int i = 0; i < L - 1; i++) {
        shape5 li = u->ls[i];
        shape5 src = u->ls[i + 1]; src.c = w[i + 1];
        if (nn_get_tf32()) {
            nn_set_layer(9 - i); float *gup = block_bwd(u, &u->dec[i], i, u->gout[i], u->gskip[i]);   /* gB[i]: grad wrt the upsampled part; skip grad written in place */
            PROF(5, nn_up2_bwd(gup, src, u->gout[i + 1]));
        } else {
            nn_set_layer(9 - i); float *gcat = block_bwd(u, &u->dec[i], i, u->gout[i], nullptr);       /* gB[i]: grad wrt concat */
            PROF(5, nn_concat_bwd(gcat, w[i + 1], w[i], li, u->gA[i], u->gskip[i]));
            PROF(5, nn_up2_bwd(u->gA[i], src, u->gout[i + 1]));
        }
    }
    /* encoder, from the bottom */
    for (int i = L - 1; i >= 0; i--) {
        float *gy = i < L - 1 ? u->gskip[i] : u->gout[i];   /* enc[i].s2 receives skip + down-conv grads; the bottom gets the up-path grad */
        nn_set_layer(i); float *gin = block_bwd(u, &u->enc[i], i, gy, nullptr);            /* grad wrt enc[i] input */
        if (i > 0) {
            /* enc[i] input = downo[i-1] = down[i-1](enc[i-1].s2): propagate to enc[i-1].s2, accumulate into gskip[i-1] */
            shape5 ds = u->enc[i].xs;                                     /* == down output shape */
            nn_set_layer(4 + i - 1); PROF(2, nn_conv3d_bwd_weight(u->enc[i - 1].s2, u->enc[i - 1].ys, gin, ds, 3, 2, g + u->down[i - 1].w, g + u->down[i - 1].b));
            int acc; PROF(1, acc = nn_conv3d_bwd_data_acc(gin, ds, P(u, u->down[i - 1].w), u->enc[i - 1].ys, 3, 2, u->gskip[i - 1], u->conv_scratch));   /* gskip += */
            if (acc) {
                PROF(1, nn_conv3d_bwd_data(gin, ds, P(u, u->down[i - 1].w), u->enc[i - 1].ys, 3, 2, u->gA[i - 1], u->conv_scratch));
                PROF(4, nn_axpy(u->gskip[i - 1], 1.f, u->gA[i - 1], shape_numel(u->enc[i - 1].ys)));
            }
        }
    }
    if (gscale != 1.f) nn_scale(g, 1.f / gscale, u->np);   /* activation gradients were scaled for 16-bit storage */
}

void unet_zero_grad(unet *u) { nn_zero(u->g, u->np * 4); }
float *unet_grad_ptr(unet *u) { return u->g; }
void unet_grad_d2h(unet *u, float *host) { nn_d2h(host, u->g, u->np * 4); }
void unet_grad_h2d(unet *u, const float *host) { nn_h2d(u->g, host, u->np * 4); }
double unet_grad_norm(unet *u) { return sqrt(nn_sumsq(u->g, u->np, u->red_scratch)); }
void unet_clip_grad(unet *u, double max_norm) {
    double n = unet_grad_norm(u);
    if (n > max_norm && n > 0) nn_scale(u->g, (float)(max_norm / n), u->np);
}
static void wq_adamw(unet *u, float lr, float b1, float b2, float eps, float wd, int step);
static void wq_ema(unet *u, float decay);
void unet_adamw(unet *u, float lr, float b1, float b2, float eps, float wd, int step) { if (u->wq) wq_adamw(u, lr, b1, b2, eps, wd, step); else nn_adamw(u->p, u->g, u->m, u->v, u->np, lr, b1, b2, eps, wd, step); }
void unet_ema(unet *u, float decay) { if (u->wq) wq_ema(u, decay); else nn_ema(u->ema, u->p, u->np, decay); }
void unet_use_ema(unet *u, int on) { u->live = on ? u->ema : u->p; u->using_ema = on; if (!u->sparse24) u->fw = u->live; }
/* ---- 2:4 structured sparsity: the forward reads a masked copy of the live weights; SR-STE keeps the master weights dense ---- */
static void for_each_conv3(unet *u, void (*fn)(unet *, const convp *, void *), void *arg) {
    int L = u->cfg.nlev;
    for (int i = 0; i < L; i++) { fn(u, &u->enc[i].c1, arg); fn(u, &u->enc[i].c2, arg); }
    for (int i = 0; i < L - 1; i++) { fn(u, &u->down[i], arg); fn(u, &u->dec[i].c1, arg); fn(u, &u->dec[i].c2, arg); }
}
static void mask_conv(unet *u, const convp *c, void *arg) { (void)arg; nn_mask24(u->wm + c->w, u->wm + c->w, c->cout, c->cin, c->k * c->k * c->k); }
static void srste_conv(unet *u, const convp *c, void *arg) { nn_srste24(u->g + c->w, u->live + c->w, c->cout, c->cin, c->k * c->k * c->k, *(float *)arg); }
void unet_set_sparse24(unet *u, int on) {
    u->sparse24 = on;
    if (on && !u->wm) u->wm = nn_malloc(u->np * 4);
    u->fw = on ? u->wm : u->live;
}
int unet_get_sparse24(const unet *u) { return u->sparse24; }
void unet_apply_sparse24(unet *u) {
    if (!u->sparse24) { u->fw = u->live; return; }
    nn_d2d(u->wm, u->live, u->np * 4);
    for_each_conv3(u, mask_conv, nullptr);
    u->fw = u->wm;
}
void unet_srste24(unet *u, float lambda) { if (u->sparse24) for_each_conv3(u, srste_conv, &lambda); }
/* ---- true fp8 / fp4 weights: packed storage, optimizer and EMA act on the packed values ---- */
void unet_set_wq(unet *u, int bits) {
    u->wq = bits;
    if (!bits) return;
    if (!u->wq_q) {
        /* offsets: packed weights use the element offset (fp8: bytes; fp4: element/2, offsets are even), scales are cumulative */
        int L = u->cfg.nlev; const convp *cs[64]; int n = 0;
        for (int i = 0; i < L; i++) { cs[n++] = &u->enc[i].c1; cs[n++] = &u->enc[i].c2; }
        for (int i = 0; i < L - 1; i++) { cs[n++] = &u->down[i]; cs[n++] = &u->dec[i].c1; cs[n++] = &u->dec[i].c2; }
        size_t so = 0;
        for (int i = 0; i < n; i++) { u->wq_qoff[i] = cs[i]->w; u->wq_soff[i] = so; so += nn_wq_nblocks(cs[i]->cout, cs[i]->cin, cs[i]->k * cs[i]->k * cs[i]->k); }
        u->wq_n = n;
        u->wq_q = nn_malloc(u->np); u->wq_sc = nn_malloc(so + 64); u->wq_qe = nn_malloc(u->np); u->wq_sce = nn_malloc(so + 64);
        /* initial packing of the fp32 values (round to nearest), then the shadows take the grid values */
        for (int i = 0; i < n; i++) {
            const convp *c = cs[i]; int T = c->k * c->k * c->k; size_t qo = bits == 8 ? c->w : c->w / 2;
            nn_wq_pack(u->p + c->w, u->wq_q + qo, u->wq_sc + u->wq_soff[i], c->cout, c->cin, T, bits, 0);
            nn_wq_pack(u->ema + c->w, u->wq_qe + qo, u->wq_sce + u->wq_soff[i], c->cout, c->cin, T, bits, 0);
            nn_wq_unpack(u->wq_q + qo, u->wq_sc + u->wq_soff[i], u->p + c->w, c->cout, c->cin, T, bits);
            nn_wq_unpack(u->wq_qe + qo, u->wq_sce + u->wq_soff[i], u->ema + c->w, c->cout, c->cin, T, bits);
        }
    }
}
int unet_get_wq(const unet *u) { return u->wq; }
static const convp *wq_conv(unet *u, int i) {   /* i-th 3^3 conv in for_each_conv3 order */
    int L = u->cfg.nlev, n = 0;
    for (int j = 0; j < L; j++) { if (n++ == i) return &u->enc[j].c1; if (n++ == i) return &u->enc[j].c2; }
    for (int j = 0; j < L - 1; j++) { if (n++ == i) return &u->down[j]; if (n++ == i) return &u->dec[j].c1; if (n++ == i) return &u->dec[j].c2; }
    return nullptr;
}
/* optimizer step with packed weights: AdamW on the packed conv weights, the plain fp32 AdamW on everything in between */
static void wq_adamw(unet *u, float lr, float b1, float b2, float eps, float wd, int step) {
    size_t pos = 0;
    for (int i = 0; i < u->wq_n; i++) {
        const convp *c = wq_conv(u, i); int T = c->k * c->k * c->k; size_t n = (size_t)c->cout * c->cin * T, qo = u->wq == 8 ? c->w : c->w / 2;
        if (c->w > pos) nn_adamw(u->p + pos, u->g + pos, u->m + pos, u->v + pos, c->w - pos, lr, b1, b2, eps, wd, step);
        nn_wq_adamw(u->wq_q + qo, u->wq_sc + u->wq_soff[i], u->g + c->w, u->m + c->w, u->v + c->w, c->cout, c->cin, T, u->wq, lr, b1, b2, eps, wd, step, (unsigned)step * 7919u + (unsigned)i);
        nn_wq_unpack(u->wq_q + qo, u->wq_sc + u->wq_soff[i], u->p + c->w, c->cout, c->cin, T, u->wq);
        pos = c->w + n;
    }
    if (pos < u->np) nn_adamw(u->p + pos, u->g + pos, u->m + pos, u->v + pos, u->np - pos, lr, b1, b2, eps, wd, step);
}
static void wq_ema(unet *u, float decay) {
    size_t pos = 0;
    for (int i = 0; i < u->wq_n; i++) {
        const convp *c = wq_conv(u, i); int T = c->k * c->k * c->k; size_t n = (size_t)c->cout * c->cin * T, qo = u->wq == 8 ? c->w : c->w / 2;
        if (c->w > pos) nn_ema(u->ema + pos, u->p + pos, c->w - pos, decay);
        nn_wq_ema(u->wq_qe + qo, u->wq_sce + u->wq_soff[i], u->wq_q + qo, u->wq_sc + u->wq_soff[i], c->cout, c->cin, T, u->wq, decay, (unsigned)(u->ema_step++) * 104729u + (unsigned)i);
        nn_wq_unpack(u->wq_qe + qo, u->wq_sce + u->wq_soff[i], u->ema + c->w, c->cout, c->cin, T, u->wq);
        pos = c->w + n;
    }
    if (pos < u->np) nn_ema(u->ema + pos, u->p + pos, u->np - pos, decay);
}
void unet_wquant(unet *u, unsigned seed) { (void)u; (void)seed; }   /* kept for API compatibility: packed storage makes it unnecessary */

/* ---- checkpoints ---- */
static int g_loaded_sparse = 0, g_loaded_wq = 0;   /* set by the header parser, applied by unet_load */
int unet_save(const unet *u, const char *path, int step, const char *extra) {
    char tmp[1400];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return -1;
    fprintf(f, "UFSM{\"nlev\":%d,\"widths\":[", u->cfg.nlev);
    for (int i = 0; i < u->cfg.nlev; i++) fprintf(f, "%s%d", i ? "," : "", u->cfg.widths[i]);
    fprintf(f, "],\"cin\":%d,\"cout\":%d,\"G\":%d,\"nparams\":%zu,\"step\":%d,\"sparse24\":%d,\"wq\":%d,\"extra\":%s}\n", u->cfg.cin, u->cfg.cout, u->cfg.G, u->np, step, u->sparse24, u->wq, extra ? extra : "{}");
    float *h = malloc(u->np * 4);
    const float *arrs[4] = {u->p, u->ema, u->m, u->v};
    for (int a = 0; a < 4; a++) { nn_d2h(h, arrs[a], u->np * 4); if (fwrite(h, 4, u->np, f) != u->np) { fclose(f); free(h); return -1; } }
    free(h);
    fclose(f);
    return rename(tmp, path);
}

static int read_header(FILE *f, unet_cfg *cfg, int *step, size_t *np) {
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "UFSM", 4)) return -1;
    char line[4096];
    if (!fgets(line, sizeof line, f)) return -1;
    memset(cfg, 0, sizeof *cfg);
    const char *p;
    if ((p = strstr(line, "\"nlev\":"))) cfg->nlev = atoi(p + 7);
    if ((p = strstr(line, "\"widths\":["))) { p += 10; for (int i = 0; i < cfg->nlev && i < UNET_MAXLEV; i++) { cfg->widths[i] = atoi(p); p = strchr(p, ','); if (!p) break; p++; } }
    if ((p = strstr(line, "\"cin\":"))) cfg->cin = atoi(p + 6);
    if ((p = strstr(line, "\"cout\":"))) cfg->cout = atoi(p + 7);
    if ((p = strstr(line, "\"G\":"))) cfg->G = atoi(p + 4);
    if ((p = strstr(line, "\"nparams\":"))) *np = (size_t)atoll(p + 10);
    if ((p = strstr(line, "\"step\":"))) *step = atoi(p + 7);
    if ((p = strstr(line, "\"sparse24\":"))) g_loaded_sparse = atoi(p + 11);
    if ((p = strstr(line, "\"wq\":"))) g_loaded_wq = atoi(p + 5);
    return 0;
}

int unet_peek(const char *path, unet_cfg *cfg, int *step) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    size_t np;
    int rc = read_header(f, cfg, step, &np);
    fclose(f);
    return rc;
}

int unet_load(unet *u, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    unet_cfg cfg; int step = 0; size_t np = 0;
    g_loaded_sparse = 0; g_loaded_wq = 0;
    if (read_header(f, &cfg, &step, &np) || np != u->np) { fclose(f); return -1; }
    float *h = malloc(u->np * 4);
    float *arrs[4] = {u->p, u->ema, u->m, u->v};
    for (int a = 0; a < 4; a++) {
        if (fread(h, 4, u->np, f) != u->np) { if (a < 2) { fclose(f); free(h); return -1; } break; }
        nn_h2d(arrs[a], h, u->np * 4);
    }
    free(h);
    fclose(f);
    if (g_loaded_sparse) unet_set_sparse24(u, 1);
    if (g_loaded_wq) unet_set_wq(u, g_loaded_wq);
    return step;
}
