#include "unet.h"
#include "checkpoint.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_prof = -1;
static const char *g_names[8] = {"conv_fwd", "conv_bwd_data", "conv_bwd_w", "gn", "elementwise", "up/concat", "loss", "optimizer"};
/* UFSM_PROF=1: per-category event timing; UFSM_PROF=layers (or unet_prof_layers_on): also per conv, event category
   k + 8 * (slot + 1) with slot = 2 * layer + conv (layer ids as in nn_set_layer, conv 0 = c1 / single, 1 = c2) */
#define PROF_INIT() do { if (g_prof < 0) { const char *e_ = getenv("UFSM_PROF"); g_prof = !e_ || !*e_ || !strcmp(e_, "0") ? 0 : !strcmp(e_, "layers") ? 2 : 1; } } while (0)
#define PROF_SLOT() (nn_get_layer() < 0 ? -1 : 2 * nn_get_layer() + (nn_get_conv() > 0 ? 1 : 0))
#define PROF(k, call) do { PROF_INIT(); if (g_prof) { nn_prof_begin((k) + (g_prof == 2 ? 8 * (PROF_SLOT() + 1) : 0)); call; nn_prof_end(); } else { call; } } while (0)
#define PROF_NK (8 * (UNET_NSLOT + 1))
static double g_prof_slot[UNET_NSLOT][3];
static void prof_collect(double *cat) {
    double ms[PROF_NK];
    nn_prof_collect(ms, PROF_NK);
    for (int i = 0; i < 8; i++) cat[i] = 0;
    for (int k = 0; k < PROF_NK; k++) { cat[k % 8] += ms[k]; if (k >= 8 && k % 8 < 3) g_prof_slot[k / 8 - 1][k % 8] += ms[k]; }
}
void unet_prof_report(void) {
    if (g_prof > 0) {
        double ms[8]; prof_collect(ms); double tot = 0;
        for (int i = 0; i < 8; i++) tot += ms[i];
        for (int i = 0; i < 8; i++) if (ms[i] > 0) fprintf(stderr, "  %-14s %8.1f ms  %4.1f%%\n", g_names[i], ms[i], 100 * ms[i] / tot);
        if (g_prof == 2) for (int i = 0; i < UNET_NSLOT; i++) {
            if (g_prof_slot[i][0] + g_prof_slot[i][1] + g_prof_slot[i][2] > 0)
            {
                char nm[16]; nn_layer_name(i / 2, nm);
                fprintf(stderr, "  slot %2d fwd %.1f bwd_data %.1f bwd_w %.1f ms  %s%s\n", i, g_prof_slot[i][0], g_prof_slot[i][1], g_prof_slot[i][2], nm,
                        !strncmp(nm, "down", 4) || !strcmp(nm, "head") ? "" : i % 2 ? ".c2" : ".c1");
            }
            for (int j = 0; j < 3; j++) g_prof_slot[i][j] = 0;
        }
    }
}
void unet_prof_layers_on(int on) { PROF_INIT(); g_prof = on ? 2 : 0; }
static double g_prof_cat[8];
void unet_prof_layers(double out[][3]) {
    for (int i = 0; i < UNET_NSLOT; i++) for (int j = 0; j < 3; j++) g_prof_slot[i][j] = 0;
    prof_collect(g_prof_cat);
    for (int i = 0; i < UNET_NSLOT; i++) for (int j = 0; j < 3; j++) out[i][j] = g_prof_slot[i][j];
}
void unet_prof_cats(double out[8]) { for (int i = 0; i < 8; i++) out[i] = g_prof_cat[i]; }   /* categories of the last unet_prof_layers call */

typedef struct { int cin, cout, k, stride; size_t w, b; } convp;      /* offsets into the flat param array */
typedef struct { int c; size_t gamma, beta; } gnp;

typedef struct {
    convp c1, c2;
    gnp n1, n2;
    int keep_a1;                    /* recompute 2 can retain selected smaller activations */
    int a1_shared;                  /* shared encoder a1 (training): a1 lives in the same-level decoder block's a1 buffer and backward re-runs conv1 */
    int a1_lent;                    /* decoder whose a1 buffer an encoder shares: re-run conv1 in a repeated backward (no forward between) */
    /* stored activations: conv outputs a1, a2 (GroupNorm inputs), GN stats, and the block output s2.
       gn(a) and silu(gn(a1)) are recomputed in backward. */
    const float *in, *in2;      /* in2: second input tensor for channels >= c_split (decoder skip), tensor-core path only */
    int c_split;
    const float *inx; const gnp *ing; const float *inm, *inr;   /* down_norm: conv1 input = silu(gn(inx)) (pre-norm down-conv output) */
    const void *xb, *x2b;       /* recompute mode, decoder: input [up2(silu(gn(xb.a2))) (transient) | silu(gn(x2b.a2)) (in staging)] */
    float *a1, *a2, *s2;
    float *m1, *r1, *m2, *r2;
    float *pm1, *pr1, *pm2, *pr2;   /* previous-step stats (UFSM_FAKEQ_WHERE=2 study) */
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
    unsigned char *wq_r, *wq_rsc, *wq_re, *wq_rsce;  /* fp4: fp8 error-feedback residuals (live, EMA) */
    size_t wq_qoff[64], wq_soff[64]; int wq_n; unsigned ema_step;   /* per 3^3 conv (for_each_conv3 order): offsets into the packed arrays */
    int using_ema;
    block enc[UNET_MAXLEV], dec[UNET_MAXLEV];
    convp down[UNET_MAXLEV], head;
    gnp dn[UNET_MAXLEV];          /* down_norm: GroupNorm after down[i] */
    float *dm[UNET_MAXLEV], *dr[UNET_MAXLEV], *downs[UNET_MAXLEV];   /* its statistics; fp32 path: the normalised output */
    /* activations */
    shape5 xs;                    /* shape the buffers were built for */
    int built, train;
    shape5 ls[UNET_MAXLEV];       /* spatial shape at each level (n, -, d, h, w) */
    float *downo[UNET_MAXLEV];    /* down conv outputs */
    float *cat[UNET_MAXLEV];                 /* decoder input: [upsampled w[i+1] | skip w[i]] */
    float *xin;                              /* bf16 copy of the network input (act-bf16 mode) */
    float *muon_mom, *muon_work; size_t muon_work_n;   /* Muon momentum (np) and Newton-Schulz scratch */
    void *muon_descs; int muon_nconv, muon_maxco, muon_maxk;   /* batched Muon descriptor table (device) */
    float *anvil_v1, *anvil_pool; void *anvil_descs; int anvil_nconv;   /* ANVIL slow rail, scratch pool, descriptors */
    float *glb;                              /* bf16 copy of the logit gradient (grad-bf16 mode) */
    int mode;                                /* kernel/storage mode the activations were built for */
    float *logits;
    float *gA[UNET_MAXLEV], *gB[UNET_MAXLEV];        /* grad buffers, widest tensor at the level */
    float *t1[UNET_MAXLEV], *t2[UNET_MAXLEV];        /* recompute scratch, w[i] wide */
    float *gskip[UNET_MAXLEV], *gout[UNET_MAXLEV];
    float *gn_scratch, *conv_scratch, *red_scratch;
    float *wg_tmp; size_t wg_tmp_n;   /* chunked decoder conv1 weight gradient: one slice's [cout][cin][27] */
    size_t conv_scratch_n, act_bytes, grad_bytes;
    size_t gB_bytes, gA_bytes;               /* capacity of the shared gradient buffers B and A */
    size_t gB_cap[UNET_MAXLEV];              /* capacity of gB[i] (noB: gout[i]) */
    size_t gout_cap[UNET_MAXLEV];            /* bytes of gout[i] */
    int rc_ok; size_t rc_cap;                /* lean 2: a transient in gB[level] is allowed now; capacity of the transient's buffer */
    int nob;                                 /* lean 2: no buffer B; a block's own gout / gskip is its B (in-place GroupNorm backward) */
    int last_a1_live;                        /* final decoder a1 survives the head until the first backward */
    int sea_stale;                           /* shared encoder a1: a backward left the encoders' a1 in the decoders' buffers */
    size_t logits_bytes;                     /* lean: logits at the start of A, the 16-bit logit gradient after them */
    int logits_h16;                          /* lean 2: fp16 logits, so the 16-bit logit gradient fits after them in A */
    float *rc_extra; size_t rc_extra_bytes;  /* training transient for the upsampled decoder input when B is too small */
    int split_side, split_h0;                /* spatial split (unet_set_split): side 0 / 1, level-0 halo planes; h0 0 = off */
    unet_halo_fn split_halo, split_begin, split_end;
};

static int G_of(const unet *u, int c) { return u->cfg.G < c ? u->cfg.G : c; }

/* ---- spatial split along z (unet_set_split): this GPU's tensors are its z planes plus 2^(L-1-l) halo planes at level l on the
   side facing the other GPU (h0 = 2^(L-1) at level 0, so every level keeps the stride-2 parity and the shapes the U-Net needs).
   Every stencil consumer (3^3 conv, stride-2 conv, upsample) needs only the innermost halo plane: a produced tensor gets it
   from the other GPU (sp_halo; the outer halo planes are zeroed), and a gradient that feeds a reduction (GroupNorm backward,
   weight / bias gradient) gets its halo planes zeroed first (sp_zero) so only owned voxels contribute. GroupNorm statistics
   and the loss sums are restricted and summed across the GPUs inside nn (nn_split_cfg). */
void unet_set_split_async(unet *u, unet_halo_fn begin, unet_halo_fn end) { u->split_begin = end ? begin : nullptr; u->split_end = end; }
void unet_set_split(unet *u, int side, int h0, unet_halo_fn fn) { u->split_side = side; u->split_h0 = fn ? h0 : 0; u->split_halo = fn; }
static int sp_on(const unet *u) { return u->split_h0 > 0; }
static int sp_h(const unet *u, shape5 s) { return (int)((long)u->split_h0 * s.d / u->xs.d); }
static int sp_esz(const void *p, int grad);
static void sp_halo(const unet *u, const void *p, shape5 s, int grad) { if (sp_on(u)) u->split_halo(p, s, sp_esz(p, grad), sp_h(u, s)); }
/* a gradient whose halo is zeroed for a weight gradient and then needed by a backward-data: the exchange starts before the
   weight gradient and completes after it (sp_halo semantics when no asynchronous hooks are set) */
static void sp_begin(const unet *u, const void *p, shape5 s, int grad) { if (sp_on(u) && u->split_begin) u->split_begin(p, s, sp_esz(p, grad), sp_h(u, s)); }
static void sp_end(const unet *u, const void *p, shape5 s, int grad) { if (sp_on(u)) { if (u->split_begin) u->split_end(p, s, sp_esz(p, grad), sp_h(u, s)); else sp_halo(u, p, s, grad); } }
static void sp_zero(const unet *u, const void *p, shape5 s, int grad) { if (sp_on(u)) { int h = sp_h(u, s); nn_split_zero(p, s, sp_esz(p, grad), u->split_side ? h : 0, u->split_side ? 0 : h); } }
static void sp_cfg(const unet *u) {   /* the device's nn split state follows the network that runs on it */
    if (!sp_on(u)) { nn_split_cfg(0, 0, 0, 0); return; }
    const int h = u->split_h0;
    nn_split_cfg(u->split_side ? h : 0, u->split_side ? 0 : h, u->xs.d, 2 * (u->xs.d - h));
}

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
    if (cfg->down_norm) for (int i = 0; i < L - 1; i++) off = add_gn(u, &u->dn[i], w[i], off);
    u->np = off;
    if (ufsm_env_on("UFSM_DEBUG")) {
        for (int i = 0; i < L; i++) fprintf(stderr, "enc%d c1.w %zu c1.b %zu n1 %zu c2.w %zu c2.b %zu n2 %zu\n", i, u->enc[i].c1.w, u->enc[i].c1.b, u->enc[i].n1.gamma, u->enc[i].c2.w, u->enc[i].c2.b, u->enc[i].n2.gamma);
        for (int i = 0; i < L - 1; i++) fprintf(stderr, "down%d w %zu b %zu\n", i, u->down[i].w, u->down[i].b);
        for (int i = L - 2; i >= 0; i--) fprintf(stderr, "dec%d c1.w %zu c1.b %zu n1 %zu c2.w %zu c2.b %zu n2 %zu\n", i, u->dec[i].c1.w, u->dec[i].c1.b, u->dec[i].n1.gamma, u->dec[i].c2.w, u->dec[i].c2.b, u->dec[i].n2.gamma);
        fprintf(stderr, "head w %zu b %zu total %zu\n", u->head.w, u->head.b, off);
    }
    u->p = nn_malloc(off * 4); u->g = nn_malloc(off * 4); u->m = nn_malloc(off * 4); u->v = nn_malloc(off * 4); u->ema = nn_malloc(off * 4);
    nn_zero(u->g, off * 4); nn_zero(u->m, off * 4); nn_zero(u->v, off * 4);
    u->live = u->p; u->fw = u->p; u->wm = nullptr; u->sparse24 = 0; u->wq = 0; u->wq_q = u->wq_sc = u->wq_qe = u->wq_sce = nullptr; u->wq_r = u->wq_rsc = u->wq_re = u->wq_rsce = nullptr; u->wq_n = 0;
    u->gn_scratch = nn_malloc(1 << 20);
    u->red_scratch = nn_malloc(4096 * 4);
    return u;
}

static void free_acts(unet *u);
void unet_free(unet *u) {
    if (!u) return;
    free_acts(u);
    nn_free(u->p); nn_free(u->g); nn_free(u->m); nn_free(u->v); nn_free(u->ema);
    nn_free(u->gn_scratch); nn_free(u->red_scratch); nn_free(u->wg_tmp);
    nn_free(u->wm); nn_free(u->muon_mom); nn_free(u->muon_work); nn_free(u->muon_descs);
    nn_free(u->anvil_v1); nn_free(u->anvil_pool); nn_free(u->anvil_descs);
    nn_free(u->wq_q); nn_free(u->wq_sc); nn_free(u->wq_qe); nn_free(u->wq_sce);
    nn_free(u->wq_r); nn_free(u->wq_rsc); nn_free(u->wq_re); nn_free(u->wq_rsce);
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
    if (u->cfg.down_norm) for (int i = 0; i < L - 1; i++) init_gn(h, &u->dn[i]);
    for (int i = 0; i < L - 1; i++) { init_conv(h, &u->dec[i].c1, &seed, 0); init_gn(h, &u->dec[i].n1); init_conv(h, &u->dec[i].c2, &seed, 0); init_gn(h, &u->dec[i].n2); }
    init_conv(h, &u->head, &seed, -2.0);
    nn_h2d(u->p, h, u->np * 4);
    nn_d2d(u->ema, u->p, u->np * 4);
    nn_zero(u->m, u->np * 4); nn_zero(u->v, u->np * 4);
    free(h);
}

/* ---- activation buffers ---- */
/* dry run (unet_train_bytes): the activation allocators only count bytes and hand out a dummy pointer */
static int g_dry = 0;
/* UFSM_MEM_REPORT=1: unet_train_bytes prints the dry build's bytes per buffer group (g_mlab, set in build_acts) */
static const char *g_mlab = "other";
static struct { const char *lab; size_t b; int n; } g_mrep[64]; static int g_nmrep = 0;
static void mrep_add(size_t b) {
    int i = 0; while (i < g_nmrep && strcmp(g_mrep[i].lab, g_mlab)) i++;
    if (i == g_nmrep) { if (g_nmrep == 64) return; g_mrep[g_nmrep].lab = g_mlab; g_mrep[g_nmrep].b = 0; g_mrep[g_nmrep].n = 0; g_nmrep++; }
    g_mrep[i].b += b; g_mrep[i].n++;
}
static float *dmalloc(size_t b) { if (g_dry) mrep_add(b); return g_dry ? (float *)(uintptr_t)4096 : nn_malloc(b); }
static void dstorage(const void *p, size_t b, int dt) { if (!g_dry) nn_set_storage(p, b, dt); }
static float *dalloc(unet *u, size_t n) { u->act_bytes += n * 4; return dmalloc(n * 4); }
#define ABF (nn_get_tf32() && nn_get_act_bf16())
#define GBF (ABF && nn_get_grad_bf16())
static float *dalloc_oom(float *p, size_t b) { if (!p && b) { fprintf(stderr, "unet: out of device memory for a %.2f GB activation buffer\n", b / 1e9); abort(); } return p; }
static float *dalloc_act(unet *u, size_t n) { size_t b = ABF ? n * 2 : n * 4; u->act_bytes += b; return dalloc_oom(dmalloc(b), b); }   /* activation storage (bf16 in act-bf16 mode) */
/* activation tensor of shape s: MX-fp8 (registered, channel-blocked bytes + scales) in act-MX8 mode, else as dalloc_act */
static int g_act_mx8 = -1, g_act_mx4 = -1;
/* MX activation storage: fp8 (registry 8) or packed fp4 (registry 4); mx4 wins when both are requested */
static int act_mx4(void) { if (g_act_mx4 < 0) g_act_mx4 = ufsm_env_on("UFSM_ACT_MX4"); return g_act_mx4 && nn_get_tf32(); }
static int act_mx8(void) { if (g_act_mx8 < 0) g_act_mx8 = ufsm_env_on("UFSM_ACT_MX8"); return (g_act_mx8 || act_mx4()) && nn_get_tf32(); }   /* any MX format */
static int act_dt(void) { return act_mx4() ? 4 : 8; }   /* registry dt of the MX activations */
void unet_set_act_mx8(int on) { g_act_mx8 = on; }
void unet_set_act_mx4(int on) { g_act_mx4 = on; }
static int g_grad_mx8 = -1;   /* MX-fp8 activation gradients (env UFSM_GRAD_MX8=1; needs the MX activations) */
static int g_input_mx = -1, g_input_prec = -1;
static int grad_mx8(void) { if (g_grad_mx8 < 0) g_grad_mx8 = ufsm_env_on("UFSM_GRAD_MX8"); return g_grad_mx8 && act_mx8(); }
/* MX-fp4 activation gradients (opt-in, env UFSM_GRAD_MX4=1 with the MX gradient mode): every gradient tensor stored as packed e2m1
   with 32-element block scales, written with exact stochastic rounding (the backward kernels' MX-fp4 paths). Level-0 voxel:
   gradient buffer A 33 -> 17 B; the logit gradient moves to the end of gout[0] */
static int g_grad_mx4 = -1;
static int grad_mx4(void) { if (g_grad_mx4 < 0) g_grad_mx4 = ufsm_env_on("UFSM_GRAD_MX4"); return g_grad_mx4 && grad_mx8(); }
static int grad_dt(void) { return grad_mx4() ? 4 : 8; }   /* registry dt of the MX gradients */
void unet_set_grad_mx4(int on) { g_grad_mx4 = on; }
/* decoder conv1 weight gradient with the upsampled part in 32-channel slices: silu(gn(a2)) of the coarse block once into
   gout[level + 1], each slice upsampled into gout[level] and weight-gradiented into a small buffer added into its columns, so
   gout[level] holds one slice, not the whole up part (MX-fp4 gradients: level 0's gout 34 -> 17 B per voxel). The MX up-mode
   wgrad (no fine transient at all) cannot use the warp-specialised kernel: +6% step. Default with MX-fp4 gradients; env
   UFSM_UP_WG_CHUNK=0/1 overrides */
static int g_up_wg_chunk = -2;
static int up_wg_chunk(void) {
    if (g_up_wg_chunk == -2) { const char *v = getenv("UFSM_UP_WG_CHUNK"); g_up_wg_chunk = v ? atoi(v) : -1; }
    return act_mx8() && (g_up_wg_chunk >= 0 ? g_up_wg_chunk : grad_mx4());
}
void unet_set_up_wg_chunk(int on) { g_up_wg_chunk = on; }   /* -1: default (with MX-fp4 gradients) */
int unet_grad_mx4(void) { return grad_mx4(); }
static int input_policy(void) {
    if (g_input_prec < 0) g_input_prec = getenv("UFSM_XIN_PREC") ? atoi(getenv("UFSM_XIN_PREC")) : 0;
    if (g_input_prec != 0 && g_input_prec != 4 && g_input_prec != 8) { fprintf(stderr, "stem input precision must be 0, 4 or 8\n"); abort(); }
    return g_input_prec;
}
static int input_mx(void) { if (g_input_mx < 0) g_input_mx = ufsm_env_on("UFSM_XIN_MX"); return act_mx8() && (g_input_mx || input_policy() || grad_mx8()); }
int unet_input_prec(void) { return input_mx() ? input_policy() ? input_policy() : act_dt() : 0; }
int unet_set_input_prec(int bits) { if (bits != 0 && bits != 4 && bits != 8) return -1; g_input_prec = bits; return 0; }
void unet_set_input_mx(int on) { g_input_mx = on; }
void unet_set_grad_mx8(int on) { g_grad_mx8 = on; }
int unet_grad_mx8(void) { return grad_mx8(); }
static float *dalloc_grad_mx(unet *u, size_t bytes) { u->act_bytes += bytes; u->grad_bytes += bytes; float *p = dalloc_oom(dmalloc(bytes), bytes); dstorage(p, bytes, grad_dt()); return p; }
static float *dalloc_act_s(unet *u, shape5 s) {
    if (!act_mx8()) return dalloc_act(u, shape_numel(s));
    size_t b = nn_mx_bytes(s, act_dt());
    u->act_bytes += b;
    float *p = dalloc_oom(dmalloc(b), b);
    dstorage(p, b, act_dt());
    return p;
}
static float *dalloc_input(unet *u, shape5 s) {
    int dt = unet_input_prec(); if (!dt) return dalloc_act(u, shape_numel(s));
    size_t b = nn_mx_bytes(s, dt); u->act_bytes += b;
    float *p = dalloc_oom(dmalloc(b), b); dstorage(p, b, dt); return p;
}
/* recompute mode (env UFSM_RECOMPUTE=1, tensor-core path): no block outputs s2 and no upsampled decoder inputs are
   stored at full resolution. Convs reading a block output apply silu(gn(a2)) while staging; the outputs of the blocks
   that get upsampled (levels >= 1, ~2% of the bytes) stay stored, and the decoder's upsampled part is rebuilt from them
   into a transient buffer right before each use: the gradient buffer B, free at both points
   (forward: before any gradient; backward: between dec conv2's backward-data and dec conv1's), or in inference one
   buffer sized for the largest level. */
static int g_recompute = -1;
static int recompute(void) { if (g_recompute < 0) { const char *e = getenv("UFSM_RECOMPUTE"); g_recompute = e ? atoi(e) : 1; }   /* default: recompute 1 (a third less memory, same error) */ return nn_get_tf32() ? g_recompute : 0; }
/* level 2: also no stored a1 in training. Every block's a1 lives in one shared buffer; backward re-runs conv1 into it
   (the GN statistics of a1 are kept from the forward) */
static int recompute_a1(void) { return recompute() >= 2; }
/* Optional retained coarse decoder activations. The finest shared buffer is unchanged.
   This exchanges some available memory for fewer conv1 reruns. */
static int keep_coarse_a1(void) {
    return recompute_a1() && ufsm_env_on("UFSM_RC_KEEP_COARSE");
}
void unet_set_recompute(int on) { g_recompute = on; }
/* shared encoder a1 (recompute 1, training): an encoder block's conv1 output has its same-level decoder block's a1 shape; it is
   dead once the encoder's conv2 has run (forward) and the decoder writes its own a1 there later, while in backward every
   decoder block finishes before the encoder ones. So encoder blocks above the bottom keep no a1 of their own: backward re-runs
   their conv1 into the shared buffer (as recompute 2, same kernel, precision and rounding keys). Level-0 voxel: -17 B of the
   ~178 (all levels ~-22 B) for one encoder conv1 forward per level (the stem, 32 -> 64 @ P / 2, ...: ~2.5% of a step).
   env UFSM_RC_ENC_A1=1 or the memory planner. */
static int g_share_enc_a1 = -1;
static int share_enc_a1(void) { if (g_share_enc_a1 < 0) g_share_enc_a1 = ufsm_env_on("UFSM_RC_ENC_A1"); return g_share_enc_a1 && recompute() == 1 && nn_get_tf32(); }
void unet_set_share_enc_a1(int on) { g_share_enc_a1 = on; }
int unet_share_enc_a1(void) { return share_enc_a1(); }
#define UMODE() (nn_get_tf32() * 4 + ABF * 2 + GBF + 8 * act_mx8() + 16 * grad_mx8() + 32 * recompute() + 128 * act_mx4() + 256 * chunk_up() + 1024 * lean() + 4096 * input_mx() + 8192 * (unet_input_prec() == 8) + 16384 * unet_wide_up_grad() + 32768 * keep_coarse_a1() + 65536 * share_enc_a1() + 131072 * grad_mx4() + 262144 * up_wg_chunk())
/* chunk mode (recompute, 16-bit activations and gradients, fused upsample): the decoder's up-part input gradient is
   produced in w[i]-channel chunks, each upsample-backwarded straight into its slice of gout[i + 1], so the shared
   gradient buffer B only needs w[i] channels (env UFSM_CHUNK_UP=0 turns it off) */
static int g_chunk = -1, g_lean = -1, g_wide_up = -1;
void unet_set_wide_up_grad(int on) { g_wide_up = on != 0; }
int unet_wide_up_grad(void) { return g_wide_up >= 0 ? g_wide_up : ufsm_env_on("UFSM_WIDE_UP_GRAD"); }
void unet_set_lean(int on) { g_lean = on; }
static int lean(void) { if (g_lean < 0) g_lean = getenv("UFSM_LEAN") ? atoi(getenv("UFSM_LEAN")) : 0; return g_lean; }
void unet_set_chunk_up(int on) { g_chunk = on; }
static int chunk_up(void) {
    if (g_chunk < 0) { const char *e = getenv("UFSM_CHUNK_UP"); g_chunk = e ? atoi(e) : 1; }
    const int f = g_chunk;
    const char *fu = getenv("UFSM_FUSED_UP");
    /* MX activation storage: only with UFSM_CHUNK_UP=2 (halves the gradient buffer B: 32 B / level-0 voxel less, ~-15% training
       memory, for the largest windows; costs ~0.9 ms per 96^3 B2 step since dec0.c1's backward-data runs in three launches) */
    return f && recompute() && (!act_mx8() || f >= 2) && (!grad_mx8() || f >= 2) && (!fu || atoi(fu) >= 2);   /* MX-fp8 gradients: 16-channel chunks merged into the MX gout block (lp_up2_bwd_mx_slice) */
}
static int sp_esz(const void *p, int grad) { return nn_storage(p) ? 0 : (grad ? GBF : ABF) ? 2 : 4; }
static float *dalloc_grad(unet *u, size_t n) { size_t b = GBF ? n * 2 : n * 4; u->act_bytes += b; u->grad_bytes += b; return dalloc_oom(dmalloc(b), b); }  /* activation-gradient storage */

/* buffers may be shared (gradient buffers across levels, gskip == gout, inference scratch): free each pointer once */
static void free_once(float **ptrs, int n) {
    if (g_dry) return;
    for (int i = 0; i < n; i++) {
        if (!ptrs[i]) continue;
        int dup = 0;
        for (int j = 0; j < i; j++) if (ptrs[j] == ptrs[i]) { dup = 1; break; }
        if (!dup) nn_free(ptrs[i]);
    }
}
static void free_acts(unet *u) {
    float *pp[UNET_MAXLEV * 30 + 8]; int np = 0;
    for (int i = 0; i < u->cfg.nlev; i++) {
        block *bs[2] = {&u->enc[i], &u->dec[i]};
        for (int k = 0; k < 2; k++) {
            block *b = bs[k];
            float *q[7] = {b->a1, b->a2, b->s2, b->m1, b->r1, b->m2, b->r2};
            for (int j = 0; j < 7; j++) pp[np++] = q[j];
            b->a1 = b->a2 = b->s2 = b->m1 = b->r1 = b->m2 = b->r2 = nullptr;
        }
        float *q[11] = {u->gskip[i], u->gA[i], u->gB[i], u->downo[i], u->cat[i], u->gout[i], u->t1[i], u->t2[i], u->dm[i], u->dr[i], u->downs[i]};
        for (int j = 0; j < 11; j++) pp[np++] = q[j];
        u->dm[i] = u->dr[i] = u->downs[i] = nullptr;
        u->downo[i] = u->cat[i] = u->gskip[i] = u->gout[i] = u->gA[i] = u->gB[i] = u->t1[i] = u->t2[i] = nullptr;
    }
    pp[np++] = u->logits; pp[np++] = u->conv_scratch; pp[np++] = u->rc_extra;
    if (u->rc_extra && !g_dry) nn_storage_forget(u->rc_extra);
    u->rc_extra = nullptr; u->rc_extra_bytes = 0; u->gB_bytes = 0; pp[np++] = u->xin; pp[np++] = u->glb;
    free_once(pp, np);
    u->logits = nullptr; u->conv_scratch = nullptr; u->xin = nullptr; u->glb = nullptr; u->conv_scratch_n = 0;
    u->built = 0; u->act_bytes = 0; u->grad_bytes = 0;
    u->last_a1_live = 0;
}

/* a1 / a2: nullptr = allocate, else a shared buffer (inference or recompute 2) */
static void build_block_acts(unet *u, block *b, shape5 xs, int train, int keep_s2, float *a1, float *a2, float *s2) {
    b->xs = xs;
    b->ys = xs; b->ys.c = b->c1.cout;
    int G = G_of(u, b->c1.cout);
    b->keep_a1 = train && recompute_a1() && !a1;
    b->a1_shared = b->a1_lent = 0;
    b->a1 = a1 ? a1 : dalloc_act_s(u, b->ys); b->a2 = a2 ? a2 : dalloc_act_s(u, b->ys);
    b->s2 = recompute() && (!keep_s2 || act_mx8()) ? nullptr : s2 ? s2 : dalloc_act_s(u, b->ys);   /* MX: the up staging normalises a2 itself */
    b->m1 = dalloc(u, (size_t)xs.n * G); b->r1 = dalloc(u, (size_t)xs.n * G); b->m2 = dalloc(u, (size_t)xs.n * G); b->r2 = dalloc(u, (size_t)xs.n * G);
}
static size_t act_bytes_of(shape5 s) { return act_mx8() ? nn_mx_bytes(s, act_dt()) : shape_numel(s) * (ABF ? 2 : 4); }

static void build_acts(unet *u, shape5 xs, int train) {
    free_acts(u);
    int L = u->cfg.nlev;
    const int *w = u->cfg.widths;
    /* inference on the tensor-core path: conv outputs that die within the forward share two scratch buffers. T1 holds
       every a1 (dead after conv2). T2 holds the down-conv outputs (dead after the next block's conv1) and the a2 that are
       not kept (decoder; encoder too unless recompute mode, where the encoder a2 are the skips); the upsampled decoder
       inputs share one buffer. Only the skips and the small upsample sources stay per tensor. */
    const int share = !train && nn_get_tf32(), rc = recompute();
    const int reuse_skip = share && rc && L > 1 && !ufsm_env_on("UFSM_INFER_SCRATCH_OLD");
    float *T1 = nullptr, *T2 = nullptr, *TC = nullptr;
    {
        shape5 s = xs; for (int i = 0; i < L; i++) { u->ls[i] = s; if (i < L - 1) s = nn_conv3d_out_shape(s, w[i], 3, 2); }
    }
    g_mlab = "T1 (shared a1, recompute 2)";
    if (train && recompute_a1()) {
        shape5 m1 = u->ls[0]; m1.c = 0;
        for (int i = 0; i < L; i++) { shape5 y = u->ls[i]; y.c = w[i]; if (act_bytes_of(y) > act_bytes_of(m1)) m1 = y; }
        T1 = dalloc_act_s(u, m1);
    }
    if (share) {
        shape5 m1 = u->ls[0], m2, mc = u->ls[0]; m1.c = 0; mc.c = 0;
        for (int i = 0; i < L; i++) { shape5 y = u->ls[i]; y.c = w[i]; if (act_bytes_of(y) > act_bytes_of(m1)) m1 = y; }
        m2 = m1;
        if (reuse_skip) {
            /* Full-resolution decoder c2 is the only large T2 user. Its encoder skip is dead after
               decoder c1, so c2 can write there. Earlier decoder outputs need only a smaller T2. */
            m2.c = 0;
            for (int i = 1; i < L - 1; i++) { shape5 y = u->ls[i]; y.c = w[i]; if (act_bytes_of(y) > act_bytes_of(m2)) m2 = y; }
        }
        for (int i = 0; i < L - 1; i++) { shape5 d = u->ls[i + 1]; d.c = w[i]; if (act_bytes_of(d) > act_bytes_of(m2)) m2 = d; }
        for (int i = 0; i < L - 1; i++) { shape5 c = u->ls[i]; c.c = w[i + 1]; if (act_bytes_of(c) > act_bytes_of(mc)) mc = c; }
        T1 = dalloc_act_s(u, m1); T2 = dalloc_act_s(u, m2); TC = rc ? nullptr : dalloc_act_s(u, mc);   /* recompute: rc_tmp allocates on demand */
    }
    g_mlab = "encoder a1 / a2 / GN stats + down outputs";
    const int sea = train && share_enc_a1() && L > 1;
    for (int i = 0; i < L; i++) {
        shape5 bin = u->ls[i]; bin.c = i == 0 ? u->cfg.cin : w[i - 1];
        float *ea1 = T1;
        if (sea && i < L - 1) { shape5 y = u->ls[i]; y.c = w[i]; ea1 = dalloc_act_s(u, y); }   /* shared with dec[i].a1 (below) */
        build_block_acts(u, &u->enc[i], bin, train, i == L - 1, ea1,
                         rc ? nullptr : T2, nullptr);   /* recompute mode keeps the (small) outputs that get upsampled */
        if (sea && i < L - 1) u->enc[i].a1_shared = 1;
        if (i < L - 1) {
            shape5 ds = u->ls[i + 1]; ds.c = w[i];
            u->downo[i] = share ? T2 : dalloc_act_s(u, ds);
            if (u->cfg.down_norm) {
                int G = G_of(u, w[i]);
                u->dm[i] = dalloc(u, (size_t)ds.n * G); u->dr[i] = dalloc(u, (size_t)ds.n * G);
                if (!nn_get_tf32()) u->downs[i] = dalloc(u, shape_numel(ds));   /* fp32 path: materialised silu(gn(.)) */
            }
        }
    }
    g_mlab = "decoder a1 / a2 / GN stats";
    for (int i = L - 2; i >= 0; i--) {
        shape5 li = u->ls[i];
        shape5 cs_ = li; cs_.c = nn_get_tf32() ? w[i + 1] : w[i] + w[i + 1];
        if (share && !rc) u->cat[i] = TC;
        else if (!rc) u->cat[i] = dalloc_act_s(u, cs_);
        shape5 cin = li; cin.c = w[i] + w[i + 1];
        /* A 16-bit decoder's kept SiLU may also use its consumed encoder skip. In MX modes s2
           is absent and the next decoder normalizes a2 on the fly. Training keeps separate buffers. */
        build_block_acts(u, &u->dec[i], cin, train, i > 0,
                         sea ? u->enc[i].a1 : train && i > 0 && keep_coarse_a1() ? nullptr : T1,
                         reuse_skip && i == 0 ? u->enc[0].a2 : T2,
                         reuse_skip && i > 0 ? u->enc[i].a2 : nullptr);
        if (sea) u->dec[i].a1_lent = 1;
    }
    shape5 os = u->ls[0]; os.c = u->cfg.cout;
    u->logits = nullptr; u->logits_h16 = 0;
    int lg_gout = 0;   /* fp16 logits at the start of gout[0] (MX-fp4 gradients, below) */
    if (train) {
        if (nn_get_tf32()) {
            /* tensor-core path: A only holds the level width and B the widest of {width, decoder up part, encoder block
               input}; both are block-local, so one pair sized for the largest level serves every level. gout[i] is dead
               once dec[i]'s backward has read it, before that block writes gskip[i]: they share a buffer. */
            g_mlab = "gradient buffer A";
            const int gmx = grad_mx8(), chunk = chunk_up();
            size_t na = 0, nbb = 0;   /* elements (16-bit / fp32) or bytes (MX) */
            for (int i = 0; i < L; i++) {
                /* chunk mode: the up-part gradient goes through B in w[i]-channel chunks, so B needs no w[i+1] */
                int cb[3] = {w[i], i < L - 1 && !chunk ? w[i + 1] : 0, i ? w[i - 1] : u->cfg.cin};
                for (int k = 0; k < 3; k++) {   /* B holds each of these tensors at this level */
                    shape5 sb = u->ls[i]; sb.c = cb[k];
                    size_t e = !cb[k] ? 0 : gmx ? nn_mx_bytes(sb, grad_dt()) : shape_numel(sb);
                    if (e > nbb) nbb = e;
                }
                shape5 sa = u->ls[i]; sa.c = w[i];
                size_t e = gmx ? nn_mx_bytes(sa, grad_dt()) : shape_numel(sa);
                if (e > na) na = e;
            }
            u->nob = lean() >= 2 && chunk;   /* lean 2: no B (needs the chunked up-part gradient: the up part goes through gout[i]) */
            float *A = gmx ? dalloc_grad_mx(u, na) : dalloc_grad(u, na), *B = u->nob ? nullptr : gmx ? dalloc_grad_mx(u, nbb) : dalloc_grad(u, nbb);
            u->gB_bytes = u->nob ? 0 : gmx ? nbb : nbb * (GBF ? 2 : 4);
            u->gA_bytes = gmx ? na : na * (GBF ? 2 : 4);
            /* lean mode: the fp32 logits live in A (dead from the end of the forward until the first block backward, the loss
               reads them before; A is unregistered for the plane-major fp32 writes of the head). Costs: logits are not
               readable after the backward (the trainer's non-finite diagnosis then reports post-backward values) */
            if (lean() && (gmx ? na : na * (GBF ? 2 : 4)) >= shape_numel(os) * 4) u->logits = A;
            /* lean 2 with the MX head: fp16 logits when the 16-bit logit gradient then fits after them in A and not after fp32
               logits (desk widths: A 33 B / level-0 voxel, logits 28 B + gradient 14 B; else a separate 14 B / voxel buffer) */
            {
                const size_t ab = u->gA_bytes, l32 = (shape_numel(os) * 4 + 255) & ~(size_t)255, l16 = (shape_numel(os) * 2 + 255) & ~(size_t)255, g16 = shape_numel(os) * 2;
                u->logits_h16 = u->logits == A && u->nob && act_mx8() && ab < l32 + g16 && ab >= l16 + g16 && !ufsm_env_on("UFSM_LOGITS32");
                /* MX-fp4 gradients (A 17 B / level-0 voxel at 32 ch): fp16 logits at the start of gout[0] (dead before the head's
                   backward-data writes it; the trainer's batch goes to its end), the 16-bit logit gradient in A */
                lg_gout = lean() && u->nob && grad_mx4() && act_mx8() && ab < l16 + g16 && ab >= g16 && !ufsm_env_on("UFSM_LOGITS32");
                if (lg_gout) { u->logits = nullptr; u->logits_h16 = 1; }
            }
            u->logits_bytes = shape_numel(os) * (u->logits_h16 ? 2 : 4);
            g_mlab = "gradient buffers gout / gskip (per level)";
            for (int i = 0; i < L; i++) {
                shape5 so = u->ls[i]; so.c = w[i];
                size_t gob = gmx ? nn_mx_bytes(so, grad_dt()) : shape_numel(so) * (GBF ? 2 : 4);
                if (u->nob && i < L - 1 && !up_wg_chunk()) { shape5 up = u->ls[i]; up.c = w[i + 1]; if (act_bytes_of(up) > gob) gob = act_bytes_of(up); }   /* also holds the MX up transient (level 1: 34 vs 33 B) */
                if (up_wg_chunk() && i > 0 && act_bytes_of(so) > gob) gob = act_bytes_of(so);   /* holds the coarse silu(gn(a2)) the level below's weight gradient reads */
                if (up_wg_chunk() && i < L - 1) { shape5 sl = u->ls[i]; sl.c = 32; if (act_bytes_of(sl) > gob) gob = act_bytes_of(sl); }   /* one upsampled 32-channel slice */
                if (u->nob && gmx && i == 0 && i < L - 1 && unet_wide_up_grad()) {
                    shape5 up = u->ls[i]; up.c = w[i + 1];
                    size_t ub = nn_mx_bytes(up, grad_dt()); if (ub > gob) gob = ub;
                }   /* optional full finest-level up gradient; the chunk loop still defers the aliased skip write */
                if (lg_gout && i == 0) gob += 4096;   /* alignment slack: [fp16 logits | ... | batch] (unet_batch_scratch) */
                u->gA[i] = A;
                u->gout[i] = gmx ? dalloc_grad_mx(u, gob) : dalloc_grad(u, gob / (GBF ? 2 : 4) + 1);
                u->gskip[i] = i < L - 1 ? u->gout[i] : nullptr;
                u->gout_cap[i] = gmx ? gob : (gob / (GBF ? 2 : 4) + 1) * (GBF ? 2 : 4);
                if (u->nob) { u->gB[i] = u->gout[i]; u->gB_cap[i] = gob; }   /* no B: the level's own gradient buffer serves as B */
                else { u->gB[i] = B; u->gB_cap[i] = u->gB_bytes; }
            }
            if (u->nob) u->gB_bytes = u->gB_cap[0];
            if (lg_gout) {
                if (u->gout_cap[0] >= (u->logits_bytes + 255) / 256 * 256) u->logits = u->gout[0];
                else u->logits = A;   /* fp16 logits in A, the logit gradient separate */
            }
            g_mlab = "decoder up-part transient (rc_extra)";
            if (act_mx8() && recompute() && !up_wg_chunk()) {   /* the MX weight-gradient transient of the decoder up part where it does not fit its level's
                                                 buffer: allocated now so the dry build (memory planner) counts it */
                shape5 m = u->ls[0]; m.c = 0;
                for (int i = 0; i < L - 1; i++) { shape5 c = u->ls[i]; c.c = w[i + 1]; if (act_bytes_of(c) > u->gB_cap[i] && act_bytes_of(c) > act_bytes_of(m)) m = c; }
                if (m.c) { u->rc_extra = dalloc_act_s(u, m); u->rc_extra_bytes = act_bytes_of(m); }
            }
        } else
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
        g_mlab = "conv scratch";
        size_t cs = 0;
        for (int i = 0; i < L; i++) {
            size_t a = nn_conv3d_scratch(u->enc[i].xs, w[i], 3), b = nn_conv3d_scratch(u->enc[i].ys, w[i], 3);
            if (a > cs) cs = a;
            if (b > cs) cs = b;
            if (i < L - 1) { size_t c = nn_conv3d_scratch(u->dec[i].xs, w[i], 3); if (c > cs) cs = c; }
        }
        u->conv_scratch = dmalloc(cs); u->conv_scratch_n = cs; u->act_bytes += cs;
    }
    g_mlab = "logits (fp32) / network input";
    if (!u->logits) u->logits = dalloc(u, shape_numel(os));
    u->xin = ABF && input_mx() ? dalloc_input(u, xs) : nullptr;   /* include mandatory input storage in the dry planner */
    u->glb = nullptr;   /* allocated on first use for a fp32 logit gradient */
    u->xs = xs; u->built = 1; u->train = train; u->mode = UMODE();
}

size_t unet_activation_bytes(const unet *u) { return u->act_bytes; }
/* build the activation / gradient buffers for xs now (as the first forward would) */
void unet_build(unet *u, shape5 xs, int train) {
    if (!u->built || memcmp(&u->xs, &xs, sizeof xs) || (train && !u->train) || u->mode != UMODE()) build_acts(u, xs, train);
}
int unet_act_mx(void) { return act_mx8(); }   /* MX activation storage (fp8 or fp4) */
int unet_act_mx4(void) { return act_mx4(); }
int unet_input_converted(void) { return input_mx(); }   /* the 16-bit network input is copied into MX storage at the forward's start */
/* lean mode: the gradient buffer B as scratch for the trainer's 16-bit logit gradient (free from the end of the forward until
   the head backward has read it); nullptr when not built for training, not lean, or too small */
/* lean: where the trainer's 16-bit logit gradient goes: lean 2 after the logits in A (B is gout[0], which the head backward
   writes while reading the logit gradient), lean 1 at the start of B; nullptr if it does not fit */
void *unet_logit_grad_scratch(unet *u, size_t bytes) {
    if (!(u->built && u->train && lean())) return nullptr;
    if (u->nob) {
        const size_t off = (u->logits_bytes + 255) & ~(size_t)255;
        if (u->logits == u->gA[0] && u->gA_bytes >= off + bytes) return (void *)((char *)u->gA[0] + off);
        if (u->logits == u->gout[0] && u->gA_bytes >= bytes) return (void *)u->gA[0];   /* logits in gout[0]: the whole of A */
        /* MX-fp4 gradients: at the end of gout[0] (the batch sits at its start; the head's fp4 backward-data output, written
           while the logit gradient is read, ends at 17 B of the 34 per level-0 voxel). The trainer checks the batch overlap. */
        if (grad_mx4() && u->gout_cap[0] >= 2 * bytes) return (void *)((char *)u->gout[0] + ((u->gout_cap[0] - bytes) & ~(size_t)255));
        return nullptr;
    }
    return u->gB[0] && u->gB_bytes >= bytes ? (void *)u->gB[0] : nullptr;
}
int unet_lean_nob(const unet *u) { return u->nob; }
int unet_logits_h16(const unet *u) { return u->built && u->logits_h16; }
void *unet_grad_scratch(unet *u, size_t bytes) { return u->built && u->train && lean() && u->gB[0] && u->gB_bytes >= bytes ? (void *)u->gB[0] : nullptr; }
/* lean 2: the trainer's batch as [head | tail] in gout[0]; with the fp16 logits at its start (MX-fp4 gradients) at the buffer's
   end, the tail (targets, masks: read by the loss) clear of the logits, the head (the input: read only by the forward's
   conversion, before the head writes the logits) allowed over them */
void *unet_batch_scratch(unet *u, size_t head, size_t tail) {
    if (!(u->built && u->train && lean() && u->nob && u->gB[0])) return nullptr;
    if (u->logits != u->gout[0]) return u->gB_bytes >= head + tail ? (void *)u->gB[0] : nullptr;
    if (u->gB_bytes < head + tail) return nullptr;
    const size_t p = (u->gB_bytes - head - tail) & ~(size_t)255, lg = (u->logits_bytes + 255) & ~(size_t)255;
    return p + head >= lg ? (void *)((char *)u->gB[0] + p) : nullptr;
}
/* device bytes of the training activations / gradients at input shape xs under the current storage modes, without allocating
   (dry build); the model is left unbuilt */
size_t unet_train_bytes(unet *u, shape5 xs) {
    if (u->built) free_acts(u);
    g_dry = 1; g_nmrep = 0; g_mlab = "other";
    build_acts(u, xs, 1);
    const size_t b = u->act_bytes;
    if (ufsm_env_on("UFSM_MEM_REPORT")) {
        const double nv = (double)xs.n * xs.d * xs.h * xs.w;
        fprintf(stderr, "memory report (dry build %dx%dx%dx%d, mode %d): %.3f GB, %.1f B per input voxel\n", xs.n, xs.d, xs.h, xs.w, UMODE(), b / 1e9, b / nv);
        for (int i = 0; i < g_nmrep; i++) fprintf(stderr, "  %-46s %8.3f GB %6.1f B/voxel (%d allocations)\n", g_mrep[i].lab, g_mrep[i].b / 1e9, g_mrep[i].b / nv, g_mrep[i].n);
    }
    free_acts(u);
    g_dry = 0;
    return b;
}
size_t unet_grad_bytes(const unet *u) { return u->grad_bytes; }
shape5 unet_out_shape(const unet *u, shape5 xs) { xs.c = u->cfg.cout; return xs; }

/* ---- forward ---- */
static const float *P(const unet *u, size_t off) { return u->fw + off; }
void unet_apply_sparse24(unet *u);

/* the GroupNorm (params + stats) of a block's output: silu(gn(a2)) = s2 */
static nn_gn_t gn_out(const unet *u, const block *b) {
    nn_gn_t g = {P(u, b->n2.gamma), P(u, b->n2.beta), b->m2, b->r2, G_of(u, b->c2.cout)};
    return g;
}
/* accuracy study (env UFSM_FAKEQ = nvfp4 | mxfp4 | mxfp6e2m3 | mxfp6e3m2 | mxfp8): stored activations (conv outputs a1,
   a2, down-conv outputs, kept block outputs) are rounded in place to that format right after they are produced */
static int fakeq(void) {
    static int f = -1;
    if (f < 0) { const char *e = getenv("UFSM_FAKEQ"); f = !e ? 0 : !strcmp(e, "nvfp4") ? 1 : !strcmp(e, "mxfp4") ? 2 : !strcmp(e, "mxfp6e2m3") ? 3 : !strcmp(e, "mxfp6e3m2") ? 4 : !strcmp(e, "mxfp8") ? 5 : 0; }
    return nn_get_tf32() ? f : 0;
}
#define FQ(p, s) do { if (fakeq() && fakeq_where() == 0) nn_fake_quant((p), (s), fakeq()); } while (0)
/* UFSM_FAKEQ_WHERE: 0 raw pre-GN a1/a2 (default, the earlier study), 1 post-GN+SiLU operand s2 only, 2 pre-GN a1/a2 after the
   affine (x - mean) * rstd with the PREVIOUS step's GN stats (the "delayed-stats affine" storage design) */
static int fakeq_where(void) { static int w = -1; if (w < 0) { const char *e = getenv("UFSM_FAKEQ_WHERE"); w = e ? atoi(e) : 0; } return w; }
#define FQ_POST(p, s) do { if (fakeq() && fakeq_where() == 1) nn_fake_quant((p), (s), fakeq()); } while (0)
#define FQ_AFF(p, s, m, r, G) do { if (fakeq() && fakeq_where() == 2) nn_fake_quant_affine((p), (s), fakeq(), (m), (r), (G)); } while (0)
/* same study for the stored activation gradients (env UFSM_FAKEQ_GRAD, same format names; 16-bit gradient storage) */
static int fakeq_grad(void) {
    static int f = -1;
    if (f < 0) { const char *e = getenv("UFSM_FAKEQ_GRAD"); f = !e ? 0 : !strcmp(e, "nvfp4") ? 1 : !strcmp(e, "mxfp4") ? 2 : !strcmp(e, "mxfp6e2m3") ? 3 : !strcmp(e, "mxfp6e3m2") ? 4 : !strcmp(e, "mxfp8") ? 5 : 0; }
    return nn_get_tf32() && GBF ? f : 0;
}
#define FQG(p, s) do { if (fakeq_grad()) nn_fake_quant((p), (s), fakeq_grad()); } while (0)
static void rc_fail(const char *what) { fprintf(stderr, "unet: recompute mode: %s unsupported\n", what); abort(); }
/* transient buffer for the upsampled decoder input at level i (shape s), typed as activation storage */
static int g_in_bwd = 0;   /* inside unet_backward_x (lean mode: B holds the trainer's batch until then) */
static float *rc_tmp(unet *u, int level, shape5 s, int *reg) {
    if (u->train && lean() && !g_in_bwd) rc_fail("lean mode: a forward transient in the gradient buffer (it holds the batch)");
    if (u->train && u->nob && !u->rc_ok) rc_fail("lean 2: a transient while the level's gradient buffer is live");
    if (u->train && act_bytes_of(s) > u->gB_cap[level]) {   /* chunk mode left B too small for this (fp8 / MX) transient */
        if (act_bytes_of(s) > u->rc_extra_bytes) {
            if (u->rc_extra) { nn_storage_forget(u->rc_extra); nn_free(u->rc_extra); u->act_bytes -= u->rc_extra_bytes; }
            const int *w = u->cfg.widths; shape5 m = u->ls[0]; m.c = 0;   /* the largest transient that does not fit its level's buffer */
            for (int i = 0; i < u->cfg.nlev - 1; i++) { shape5 c = u->ls[i]; c.c = w[i + 1]; if (act_bytes_of(c) > u->gB_cap[i] && act_bytes_of(c) > act_bytes_of(m)) m = c; }
            u->rc_extra = dalloc_act_s(u, m); u->rc_extra_bytes = act_bytes_of(m);
        }
        *reg = 0;
        return u->rc_extra;
    }
    if (!u->train && !u->cat[0]) {   /* inference: the transient is only allocated if the fused path is unavailable */
        const int *w = u->cfg.widths; shape5 m = u->ls[0]; m.c = 0;
        for (int i = 0; i < u->cfg.nlev - 1; i++) { shape5 c = u->ls[i]; c.c = w[i + 1]; if (act_bytes_of(c) > act_bytes_of(m)) m = c; }
        u->cat[0] = dalloc_act_s(u, m);
    }
    float *p = u->train ? u->gB[level] : u->cat[0];
    *reg = 0;
    u->rc_cap = u->train ? u->gB_cap[level] : 0;
    if (act_mx8() && nn_storage(p) != act_dt()) { *reg = 1 + nn_storage(p); nn_set_storage(p, nn_mx_bytes(s, act_dt()), act_dt()); }   /* gradient buffer (16-bit or MX-fp8) used as activation storage for now */
    return p;
}
/* reg: 0 nothing changed, 1 the buffer was unregistered, 1 + dt it was registered as dt (an MX-fp8 gradient buffer under mx4 activations) */
static void rc_done(unet *u, float *p, int reg) { if (reg == 1) nn_storage_forget(p); else if (reg > 1) nn_set_storage(p, u->rc_cap, reg - 1); }
static float *rc_up(unet *u, block *b, int level, int *reg) {
    const block *xb = (const block *)b->xb;
    shape5 us = b->xs; us.c = b->c_split;
    float *p = rc_tmp(u, level, us, reg);
    nn_gn_t g = gn_out(u, xb);
    if (xb->s2) PROF(5, nn_up2_fwd_into(xb->s2, xb->ys, p, b->c_split, 0));
    else if (u->train && act_mx8() && g_in_bwd && level + 1 < u->cfg.nlev && u->gout_cap[level + 1] >= act_bytes_of(xb->ys)) {
        /* no kept s2 under MX storage: silu(gn(a2)) of the coarse block once into gout[level + 1] (free until this block's up-part
           gradient chunks write it), then the plain upsample (the GN inside up2_mx_k ran 8x per fine voxel: 3.6 vs 0.8 ms) */
        float *t = u->gout[level + 1];
        const int dt0 = nn_storage(t);
        nn_set_storage(t, act_bytes_of(xb->ys), act_dt());
        PROF(3, nn_gn_silu_apply(xb->a2, xb->ys, g.G, g.gamma, g.beta, g.mean, g.rstd, t));
        PROF(5, nn_up2_fwd_into(t, xb->ys, p, b->c_split, 0));
        if (dt0) nn_set_storage(t, u->gout_cap[level + 1], dt0); else nn_storage_forget(t);
    }
    else PROF(5, nn_up2_fwd_gn_into(xb->a2, xb->ys, &g, p, b->c_split, 0));
    sp_halo(u, p, us, 0);
    return p;
}
/* decoder conv1 forward (y with GN stats when G) or weight gradient (gy != nullptr) in recompute mode. Fused: the
   upsampled part is read from the coarse block output inside the conv staging (16-bit kernels), so inference needs no
   full-resolution transient at all (-33% inference memory at the same speed) and training skips the upsample kernel
   (~1 ms per step at 96^3 B2; the gradient buffer B is still sized for the up-part gradient). The fp8 / MX kernels
   take the transient. env UFSM_FUSED_UP: 0 never, 1 inference only, 2 always (default). */
static int fused_up(const unet *u) { static int f = -1; if (f < 0) { const char *e = getenv("UFSM_FUSED_UP"); f = e ? atoi(e) : 2; } return f >= 2 || (f == 1 && !u->train); }
static int dec_conv1(unet *u, block *b, int level, float *y, int G, float *m, float *r, const float *gy, float *gw, float *gbias) {
    const block *xb = (const block *)b->xb, *x2b = (const block *)b->x2b;
    nn_gn_t g2 = gn_out(u, x2b);
    int rv = -1;
    if (fused_up(u) && xb->s2) {
        if (gy) PROF(2, rv = nn_conv3d_bwd_weight_x(xb->s2, nullptr, x2b->a2, &g2, b->c_split, 1, b->xs, gy, b->ys, 3, 1, gw, gbias));
        else PROF(0, rv = nn_conv3d_fwd_x(xb->s2, nullptr, x2b->a2, &g2, b->c_split, 1, b->xs, P(u, b->c1.w), P(u, b->c1.b), b->c1.cout, 3, 1, y, G, 1e-5f, m, r));
        if (!rv) return 0;
    } else if (fused_up(u) && !gy && act_mx8()) {   /* no kept s2 (MX storage): the up staging applies the coarse block's GN+SiLU to its a2 */
        nn_gn_t gxb = gn_out(u, xb);
        PROF(0, rv = nn_conv3d_fwd_x(xb->a2, &gxb, x2b->a2, &g2, b->c_split, 1, b->xs, P(u, b->c1.w), P(u, b->c1.b), b->c1.cout, 3, 1, y, G, 1e-5f, m, r));
        if (!rv) return 0;
    }
    if (gy && up_wg_chunk() && !xb->s2 && u->train && g_in_bwd && level + 1 < u->cfg.nlev && u->gout_cap[level + 1] >= act_bytes_of(xb->ys)
        && b->c_split % 32 == 0 && act_dt() == nn_storage(x2b->a2)) {
        float *t = u->gout[level + 1], *p = u->gB[level];
        const int dt0 = nn_storage(t), dp0 = nn_storage(p), cin = b->xs.c, cs = b->c_split, cout = b->c1.cout, sk = cin - cs;
        shape5 sl = b->xs; int cw = 32;
        for (int c = 64; c <= cs; c += 32) { sl.c = c; if (cs % c == 0 && act_bytes_of(sl) <= u->gB_cap[level]) cw = c; }
        sl.c = cw;
        if (act_bytes_of(sl) <= u->gB_cap[level]) {
            const size_t need = (size_t)cout * (cw + sk) * 27;
            if (need > u->wg_tmp_n) { nn_free(u->wg_tmp); u->wg_tmp = nn_malloc(need * 4); u->wg_tmp_n = need; }
            nn_gn_t g = gn_out(u, xb);
            nn_set_storage(t, act_bytes_of(xb->ys), act_dt());
            PROF(3, nn_gn_silu_apply(xb->a2, xb->ys, g.G, g.gamma, g.beta, g.mean, g.rstd, t));
            nn_set_storage(p, act_bytes_of(sl), act_dt());
            rv = 0;
            for (int c0 = 0; c0 < cs && !rv; c0 += cw) {   /* the first slice carries the skip part and the bias */
                PROF(5, nn_up2_fwd_mx_range(t, xb->ys, c0, cw, p));
                sp_halo(u, p, sl, 0);
                const int first = c0 == 0, nl = cw + (first ? sk : 0);
                shape5 xs1 = b->xs; xs1.c = nl;
                nn_zero(u->wg_tmp, (size_t)cout * nl * 27 * 4);
                PROF(2, rv = nn_conv3d_bwd_weight_x(p, nullptr, first ? x2b->a2 : nullptr, first ? &g2 : nullptr, cw, 0, xs1, gy, b->ys, 3, 1, u->wg_tmp, first ? gbias : nullptr));
                if (rv) break;
                nn_add_rows(gw + (size_t)c0 * 27, (size_t)cin * 27, u->wg_tmp, (size_t)nl * 27, cout, (size_t)cw * 27);
                if (first) nn_add_rows(gw + (size_t)cs * 27, (size_t)cin * 27, u->wg_tmp + (size_t)cw * 27, (size_t)nl * 27, cout, (size_t)sk * 27);
            }
            if (dt0) nn_set_storage(t, u->gout_cap[level + 1], dt0); else nn_storage_forget(t);
            if (dp0) nn_set_storage(p, u->gB_cap[level], dp0); else nn_storage_forget(p);
            if (!rv) return 0;
            rc_fail("chunked decoder conv1 weight gradient");
        }
    }
    int reg; float *up = rc_up(u, b, level, &reg);
    if (gy) PROF(2, rv = nn_conv3d_bwd_weight_x(up, nullptr, x2b->a2, &g2, b->c_split, 0, b->xs, gy, b->ys, 3, 1, gw, gbias));
    else PROF(0, rv = nn_conv3d_fwd_x(up, nullptr, x2b->a2, &g2, b->c_split, 0, b->xs, P(u, b->c1.w), P(u, b->c1.b), b->c1.cout, 3, 1, y, G, 1e-5f, m, r));
    rc_done(u, up, reg);
    return rv;
}
/* down_norm: GroupNorm (params + stats) of a block's input, G = 0 when there is none */
static nn_gn_t gn_in(const unet *u, const block *b) {
    nn_gn_t g = {nullptr, nullptr, nullptr, nullptr, 0};
    if (b->ing) { g.gamma = P(u, b->ing->gamma); g.beta = P(u, b->ing->beta); g.mean = b->inm; g.rstd = b->inr; g.G = G_of(u, b->ing->c); }
    return g;
}
static void block_fwd(unet *u, block *b, int level, const float *x) {
    b->in = x;
    int G = G_of(u, b->c1.cout);
    float *t1 = u->t1[level] ? u->t1[level] : b->s2;    /* inference: no scratch, s2 doubles as temp */
    if (nn_get_tf32()) {   /* conv1 yields the stats of a1; conv2 applies gn + silu to a1 while staging and yields the stats of a2 */
        nn_set_conv(0);
        if (fakeq() && fakeq_where() == 2) {   /* keep the previous step's stats for the affine study */
            if (!b->pm1) { b->pm1 = nn_malloc((size_t)b->ys.n * G * 4 * 4); b->pr1 = b->pm1 + (size_t)b->ys.n * G; b->pm2 = b->pr1 + (size_t)b->ys.n * G; b->pr2 = b->pm2 + (size_t)b->ys.n * G; { size_t m = (size_t)b->ys.n * G; float *one = malloc(m * 4); for (size_t k = 0; k < m; k++) one[k] = 1.f; nn_zero(b->pm1, m * 4 * 4); nn_h2d(b->pr1, one, m * 4); nn_h2d(b->pr2, one, m * 4); free(one); } }
            nn_d2d(b->pm1, b->m1, (size_t)b->ys.n * G * 4); nn_d2d(b->pr1, b->r1, (size_t)b->ys.n * G * 4); nn_d2d(b->pm2, b->m2, (size_t)b->ys.n * G * 4); nn_d2d(b->pr2, b->r2, (size_t)b->ys.n * G * 4);
        }
        if (b->xb) { if (dec_conv1(u, b, level, b->a1, G, b->m1, b->r1, nullptr, nullptr, nullptr)) rc_fail("decoder conv1"); }   /* [up2(coarse) | silu(gn(skip a2))] */
        else if (b->in2) PROF(0, nn_conv3d_fwd_split(x, b->in2, b->c_split, b->xs, 0, nullptr, nullptr, nullptr, nullptr, P(u, b->c1.w), P(u, b->c1.b), b->c1.cout, b->a1, G, 1e-5f, b->m1, b->r1));
        else { nn_gn_t gi = gn_in(u, b); PROF(0, nn_conv3d_fwd_gn_stats(x, b->xs, gi.G, gi.gamma, gi.beta, gi.mean, gi.rstd, P(u, b->c1.w), P(u, b->c1.b), b->c1.cout, b->a1, G, 1e-5f, b->m1, b->r1)); }   /* down_norm: gn + silu of the input in staging */
        FQ(b->a1, b->ys); FQ_AFF(b->a1, b->ys, b->pm1, b->pr1, G);
        sp_halo(u, b->a1, b->ys, 0);
        nn_set_conv(1);
        PROF(0, nn_conv3d_fwd_gn_stats(b->a1, b->ys, G, P(u, b->n1.gamma), P(u, b->n1.beta), b->m1, b->r1, P(u, b->c2.w), P(u, b->c2.b), b->c2.cout, b->a2, G, 1e-5f, b->m2, b->r2));
        nn_set_conv(-1);
        FQ(b->a2, b->ys); FQ_AFF(b->a2, b->ys, b->pm2, b->pr2, G);
        sp_halo(u, b->a2, b->ys, 0);
        if (b->s2) { PROF(3, nn_gn_silu_apply(b->a2, b->ys, G, P(u, b->n2.gamma), P(u, b->n2.beta), b->m2, b->r2, b->s2)); FQ(b->s2, b->ys); FQ_POST(b->s2, b->ys); }
        return;
    }
    PROF(0, nn_conv3d_fwd(x, b->xs, P(u, b->c1.w), P(u, b->c1.b), b->c1.cout, 3, 1, b->a1));
    PROF(3, nn_gn_fwd_silu(b->a1, b->ys, G, 1e-5f, P(u, b->n1.gamma), P(u, b->n1.beta), t1, b->m1, b->r1));
    PROF(0, nn_conv3d_fwd(t1, b->ys, P(u, b->c2.w), P(u, b->c2.b), b->c2.cout, 3, 1, b->a2));
    PROF(3, nn_gn_fwd_silu(b->a2, b->ys, G, 1e-5f, P(u, b->n2.gamma), P(u, b->n2.beta), b->s2, b->m2, b->r2));
}

const float *unet_forward(unet *u, const float *x, shape5 xs, int train) { return unet_forward_x(u, x, xs, train, 0); }
const float *unet_forward_x(unet *u, const void *xv, shape5 xs, int train, int x_h16) {
    const float *x = (const float *)xv;
    nn_set_nlev(u->cfg.nlev);   /* layer ids / names of the precision policy, manifests and profiles (a no-op unless the depth changed) */
    int div = 1 << (u->cfg.nlev - 1);
    if (xs.d % div || xs.h % div || xs.w % div) { fprintf(stderr, "unet: spatial size %dx%dx%d must be divisible by %d\n", xs.d, xs.h, xs.w, div); abort(); }
    if (!u->built || memcmp(&u->xs, &xs, sizeof xs) || (train && !u->train) || u->mode != UMODE()) build_acts(u, xs, train);
    if (sp_on(u) && (!nn_get_tf32() || !ABF)) { fprintf(stderr, "unet: the spatial split needs the tensor-core path with 16-bit or MX storage\n"); abort(); }
    sp_cfg(u);
    unet_apply_sparse24(u);
    int L = u->cfg.nlev;
    const int *w = u->cfg.widths;
    const float *cur = x;
    if (x_h16 && !ABF) { fprintf(stderr, "unet_forward_x: a 16-bit input needs the 16-bit activation storage\n"); abort(); }
    /* Without MX gradients the four-channel input can stay 16-bit; the tap-packed FP8 kernel writes MX a1.
       MX gradients require a packed input for the available weight-gradient kernels (eight-wide MX rows). */
    const int xin_mx = input_mx();   /* MX-fp8 gradients require an MX x for the available weight-gradient kernels */
    if (ABF && !x_h16) { if (!u->xin) u->xin = dalloc_input(u, xs); nn_f32_to_act(x, xs, u->xin); cur = u->xin; }
    else if (x_h16 && act_mx8() && xin_mx) { if (!u->xin) u->xin = dalloc_input(u, xs); nn_h16_to_mx(x, xs, u->xin); cur = u->xin; }
    for (int i = 0; i < L; i++) {
        nn_set_layer(i); block_fwd(u, &u->enc[i], i, cur);
        cur = u->enc[i].s2;
        if (i < L - 1) {
            nn_set_layer(L + i);
            if (recompute()) { nn_gn_t g = gn_out(u, &u->enc[i]); int r; PROF(0, r = nn_conv3d_fwd_x(u->enc[i].a2, &g, nullptr, nullptr, 0, 0, u->enc[i].ys, P(u, u->down[i].w), P(u, u->down[i].b), w[i], 3, 2, u->downo[i], 0, 0.f, nullptr, nullptr)); if (r) rc_fail("down conv"); }
            else PROF(0, nn_conv3d_fwd(cur, u->enc[i].ys, P(u, u->down[i].w), P(u, u->down[i].b), w[i], 3, 2, u->downo[i]));
            { shape5 ds = u->ls[i + 1]; ds.c = w[i]; sp_halo(u, u->downo[i], ds, 0); }
            if (u->cfg.down_norm) {   /* GroupNorm + SiLU after the down conv: stats now, the transform in the next conv1's staging */
                shape5 ds = u->ls[i + 1]; ds.c = w[i];
                const int G = G_of(u, w[i]);
                block *nb = &u->enc[i + 1];
                nb->inx = u->downo[i]; nb->ing = &u->dn[i]; nb->inm = u->dm[i]; nb->inr = u->dr[i];
                if (nn_get_tf32()) { if (nn_gn_stats(u->downo[i], ds, G, 1e-5f, u->dm[i], u->dr[i])) { fprintf(stderr, "unet: down_norm GroupNorm statistics unsupported for this storage\n"); abort(); } }
                else { PROF(3, nn_gn_fwd_silu(u->downo[i], ds, G, 1e-5f, P(u, u->dn[i].gamma), P(u, u->dn[i].beta), u->downs[i], u->dm[i], u->dr[i])); cur = u->downs[i]; continue; }
            }
            { shape5 ds = u->ls[i + 1]; ds.c = w[i]; FQ(u->downo[i], ds); }
            cur = u->downo[i];
        }
    }
    for (int i = L - 2; i >= 0; i--) {
        shape5 src = u->ls[i + 1]; src.c = w[i + 1];
        shape5 li = u->ls[i];
        u->dec[i].xb = u->dec[i].x2b = nullptr;
        if (recompute()) {   /* nothing materialised: dec conv1 stages up2(silu(gn(a2))) of the coarse block and silu(gn(a2)) of the skip */
            u->dec[i].xb = i == L - 2 ? (const void *)&u->enc[L - 1] : (const void *)&u->dec[i + 1];
            u->dec[i].x2b = &u->enc[i]; u->dec[i].in2 = nullptr; u->dec[i].c_split = w[i + 1];
        } else if (nn_get_tf32()) {   /* cat holds only the upsampled part; the skip is read in place by the split conv */
            PROF(5, nn_up2_fwd_into(cur, src, u->cat[i], w[i + 1], 0));
            { shape5 us = li; us.c = w[i + 1]; sp_halo(u, u->cat[i], us, 0); }
            u->dec[i].in2 = u->enc[i].s2; u->dec[i].c_split = w[i + 1];
        } else {
            PROF(5, nn_up2_fwd_into(cur, src, u->cat[i], w[i + 1] + w[i], 0));
            size_t S = shape_spatial(li);
            for (int n = 0; n < li.n; n++)
                nn_d2d(u->cat[i] + ((size_t)n * (w[i + 1] + w[i]) + w[i + 1]) * S, u->enc[i].s2 + (size_t)n * w[i] * S, (size_t)w[i] * S * 4);
            u->dec[i].in2 = nullptr;
        }
        nn_set_layer(3 * L - 3 - i); block_fwd(u, &u->dec[i], i, u->cat[i]);
        cur = u->dec[i].s2;
    }
    nn_set_layer(3 * L - 2);
    nn_set_head_out_h16(u->logits_h16);
    if (recompute()) { nn_gn_t g = gn_out(u, &u->dec[0]); int r; PROF(0, r = nn_conv3d_fwd_x(u->dec[0].a2, &g, nullptr, nullptr, 0, 0, u->dec[0].ys, P(u, u->head.w), P(u, u->head.b), u->cfg.cout, 1, 1, u->logits, 0, 0.f, nullptr, nullptr)); if (r) rc_fail("head"); }
    else PROF(0, nn_conv3d_fwd(cur, u->dec[0].ys, P(u, u->head.w), P(u, u->head.b), u->cfg.cout, 1, 1, u->logits));
    nn_set_head_out_h16(0);
    if (ufsm_env_on("UFSM_DEBUG") && !ABF && !recompute()) {
        for (int i = 0; i < L; i++) fprintf(stderr, "enc%d a1 %.4g a2 %.4g s2 %.4g%s\n", i, nn_sumsq(u->enc[i].a1, shape_numel(u->enc[i].ys), u->red_scratch), nn_sumsq(u->enc[i].a2, shape_numel(u->enc[i].ys), u->red_scratch), nn_sumsq(u->enc[i].s2, shape_numel(u->enc[i].ys), u->red_scratch), i < L - 1 ? "" : " (bottom)");
        for (int i = L - 2; i >= 0; i--) fprintf(stderr, "dec%d cat %.4g a1 %.4g a2 %.4g s2 %.4g\n", i, nn_sumsq(u->cat[i], shape_numel(u->dec[i].xs), u->red_scratch), nn_sumsq(u->dec[i].a1, shape_numel(u->dec[i].ys), u->red_scratch), nn_sumsq(u->dec[i].a2, shape_numel(u->dec[i].ys), u->red_scratch), nn_sumsq(u->dec[i].s2, shape_numel(u->dec[i].ys), u->red_scratch));
        fprintf(stderr, "logits %.4g\n", nn_sumsq(u->logits, shape_numel(u->ls[0]) / u->ls[0].c * u->cfg.cout, u->red_scratch));
    }
    /* Recompute 2 shares a1 across blocks. The last decoder wrote it last, and the head/loss
       only consume a2/logits, so its first backward can consume the original a1. */
    u->last_a1_live = train && recompute_a1() && !ufsm_env_on("UFSM_RC_REDO_LAST");
    u->sea_stale = 0;
    return u->logits;
}

/* ---- backward ---- */
/* gy: grad wrt block output (s2). Returns gB of the level holding grad wrt block input. */
static float *block_bwd(unet *u, block *b, int level, const float *gy, float *gx2) {
    int G = G_of(u, b->c1.cout);
    size_t n = shape_numel(b->ys);
    float *A = u->gA[level], *B = u->nob ? (float *)gy : u->gB[level], *t1 = u->t1[level], *t2 = u->t2[level];   /* no B: in place on gy */
    float *g = u->g;
    const int live = u->last_a1_live && b == &u->dec[0];
    u->last_a1_live = 0;   /* repeated backward without a new forward must rebuild the shared buffer */
    if ((recompute_a1() && !b->keep_a1 && !live) || b->a1_shared || (b->a1_lent && u->sea_stale)) {   /* a1 was overwritten by later blocks: re-run conv1 (same kernel and precision as the forward) */
        nn_set_conv(0);
        int r;
        if (b->xb) r = dec_conv1(u, b, level, b->a1, G, nullptr, nullptr, nullptr, nullptr, nullptr);
        else { nn_gn_t gi = gn_in(u, b); PROF(0, r = nn_conv3d_fwd_x(b->in, gi.G ? &gi : nullptr, nullptr, nullptr, 0, 0, b->xs, P(u, b->c1.w), P(u, b->c1.b), b->c1.cout, 3, 1, b->a1, G, 0.f, nullptr, nullptr)); }
        if (r) rc_fail("conv1 recompute");
        FQ(b->a1, b->ys);
        sp_halo(u, b->a1, b->ys, 0);
    }
    nn_set_conv(1);   /* precision-policy tag: c2 first, then c1 */
    if (nn_get_tf32()) {   /* fused: no materialised gn / silu activations */
        sp_zero(u, gy, b->ys, 1);   /* split: only owned voxels enter the reductions below */
        PROF(3, nn_gn_silu_bwd(b->a2, b->ys, G, P(u, b->n2.gamma), P(u, b->n2.beta), b->m2, b->r2, gy, B, g + b->n2.gamma, g + b->n2.beta, u->gn_scratch));   /* B = d/d a2 */
        FQG(B, b->ys);
        sp_zero(u, B, b->ys, 1);
        sp_begin(u, B, b->ys, 1);   /* split: the halo crosses during the weight gradient (which reads it as zero) */
        PROF(2, nn_conv3d_bwd_weight_gn(b->a1, b->ys, G, P(u, b->n1.gamma), P(u, b->n1.beta), b->m1, b->r1, B, b->ys, g + b->c2.w, g + b->c2.b));
        sp_end(u, B, b->ys, 1);
        PROF(1, nn_conv3d_bwd_data(B, b->ys, P(u, b->c2.w), b->ys, 3, 1, A, u->conv_scratch));                              /* A = d/d s1 */
        FQG(A, b->ys);
        sp_zero(u, A, b->ys, 1);
        PROF(3, nn_gn_silu_bwd(b->a1, b->ys, G, P(u, b->n1.gamma), P(u, b->n1.beta), b->m1, b->r1, A, A, g + b->n1.gamma, g + b->n1.beta, u->gn_scratch));    /* A = d/d a1 (in place) */
        FQG(A, b->ys);
        sp_zero(u, A, b->ys, 1);
        if (b != &u->enc[0]) sp_begin(u, A, b->ys, 1);   /* for conv1's backward-data (enc0 has none) */
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
    nn_set_conv(0);
    if (b->xb) {   /* recompute: weight gradient with the upsampled part rebuilt into B (free until backward-data below) */
        u->rc_ok = 1;   /* gy / B consumed: the level's gradient buffer is free until the skip gradient below */
        if (dec_conv1(u, b, level, nullptr, 0, nullptr, nullptr, A, g + b->c1.w, g + b->c1.b)) rc_fail("decoder conv1 weight gradient");
        u->rc_ok = 0;
        sp_end(u, A, b->ys, 1);
        if (chunk_up()) {   /* skip gradient, then the up-part gradient in chunks through B, upsample-backwarded into gout[level + 1] */
            const block *xb = (const block *)b->xb;
            const size_t per_c = shape_numel(b->ys) / b->ys.c * (GBF ? 2 : 4);
            const size_t cap = u->gB_cap[level];
            int cs = (int)(cap / per_c); if (cs > b->c_split) cs = b->c_split;
            if (grad_mx8()) {   /* MX-fp8 chunks: 16 channels or whole 32-channel blocks that fit B */
                shape5 t = b->xs; cs = 16;
                for (int c = 32; c <= b->c_split; c += 32) { t.c = c; if (nn_mx_bytes(t, grad_dt()) <= cap) cs = c; }
                t.c = cs; if (nn_mx_bytes(t, grad_dt()) > cap) rc_fail("MX chunk does not fit the gradient buffer");
            }
            if (cs < 1) rc_fail("chunk does not fit the gradient buffer");
            int r;
            if (!u->nob) {   /* skip gradient first (no B: last, it lands in the buffer the chunks pass through) */
                PROF(1, r = nn_conv3d_bwd_data_range(A, b->ys, P(u, b->c1.w), b->xs, b->c_split, b->xs.c - b->c_split, gx2, u->conv_scratch));
                if (r) rc_fail("skip backward-data");
                { shape5 s2 = b->xs; s2.c = b->xs.c - b->c_split; FQG(gx2, s2); }
            }
            for (int c0 = 0; c0 < b->c_split; c0 += cs) {
                const int nc = b->c_split - c0 < cs ? b->c_split - c0 : cs;
                PROF(1, r = nn_conv3d_bwd_data_range(A, b->ys, P(u, b->c1.w), b->xs, c0, nc, B, u->conv_scratch));
                if (r) rc_fail("up-part backward-data");
                { shape5 fs = b->xs; fs.c = nc; sp_halo(u, B, fs, 1); }
                shape5 cs5 = xb->ys; cs5.c = nc;
                PROF(5, nn_up2_bwd_into(B, cs5, u->gout[level + 1], b->c_split, c0));
            }
            if (u->nob) {
                PROF(1, r = nn_conv3d_bwd_data_range(A, b->ys, P(u, b->c1.w), b->xs, b->c_split, b->xs.c - b->c_split, gx2, u->conv_scratch));
                if (r) rc_fail("skip backward-data");
            }
            nn_set_conv(-1);
            return nullptr;   /* gout[level + 1] already written */
        }
        PROF(1, nn_conv3d_bwd_data_split(A, b->ys, P(u, b->c1.w), b->xs, B, gx2, b->c_split, u->conv_scratch));   /* B = d/d up part, gx2 = d/d skip */
        { shape5 s1 = b->xs, s2 = b->xs; s1.c = b->c_split; s2.c = b->xs.c - b->c_split; FQG(B, s1); FQG(gx2, s2); }
        nn_set_conv(-1);
        return B;
    }
    if (b->in2 && nn_get_tf32()) {
        PROF(2, nn_conv3d_bwd_weight_split(b->in, b->in2, b->c_split, b->xs, 0, nullptr, nullptr, nullptr, nullptr, A, b->ys, g + b->c1.w, g + b->c1.b));
        sp_end(u, A, b->ys, 1);
        PROF(1, nn_conv3d_bwd_data_split(A, b->ys, P(u, b->c1.w), b->xs, B, gx2, b->c_split, u->conv_scratch));   /* B = d/d up part, gx2 = d/d skip */
        { shape5 s1 = b->xs, s2 = b->xs; s1.c = b->c_split; s2.c = b->xs.c - b->c_split; FQG(B, s1); FQG(gx2, s2); }
        nn_set_conv(-1);
        return B;
    }
    const nn_gn_t gi = gn_in(u, b);
    if (gi.G && nn_get_tf32()) PROF(2, nn_conv3d_bwd_weight_gn(b->in, b->xs, gi.G, gi.gamma, gi.beta, gi.mean, gi.rstd, A, b->ys, g + b->c1.w, g + b->c1.b));
    else PROF(2, nn_conv3d_bwd_weight(b->in, b->xs, A, b->ys, 3, 1, g + b->c1.w, g + b->c1.b));
    if (b != &u->enc[0] || !nn_get_tf32())   /* the network-input gradient of enc0 is never used */
    { sp_end(u, A, b->ys, 1);
      PROF(1, nn_conv3d_bwd_data(A, b->ys, P(u, b->c1.w), b->xs, 3, 1, B, u->conv_scratch));            /* B = d/d in */
        FQG(B, b->xs); }
    if (gi.G) {   /* down_norm: back through silu(gn(.)) of the down-conv output -> B = d/d inx */
        sp_zero(u, B, b->xs, 1);
        if (nn_get_tf32()) PROF(3, nn_gn_silu_bwd(b->inx, b->xs, gi.G, gi.gamma, gi.beta, gi.mean, gi.rstd, B, B, g + b->ing->gamma, g + b->ing->beta, u->gn_scratch));
        else {
            const size_t ni = shape_numel(b->xs);
            PROF(3, nn_gn_apply(b->inx, b->xs, gi.G, gi.gamma, gi.beta, gi.mean, gi.rstd, t1));   /* t1 = g */
            PROF(4, nn_silu_bwd(t1, B, ni, A));                                                     /* A = d/d g */
            PROF(3, nn_gn_bwd(b->inx, b->xs, gi.G, gi.gamma, gi.mean, gi.rstd, A, B, g + b->ing->gamma, g + b->ing->beta, u->gn_scratch));   /* B = d/d inx */
        }
    }
    nn_set_conv(-1);
    return B;
}

void unet_backward(unet *u, const float *glogits) { unet_backward_x(u, glogits, 0); }
void unet_backward_x(unet *u, const void *gv, int g_h16) {
    const float *glogits = (const float *)gv;
    int L = u->cfg.nlev;
    nn_set_nlev(L);
    g_in_bwd = 1;
    const float gscale = GBF ? nn_get_grad_scale() : 1.f;
    sp_cfg(u);
    if (g_h16 && !GBF) { fprintf(stderr, "unet_backward_x: a 16-bit logit gradient needs the 16-bit gradient storage\n"); abort(); }
    if (GBF && !g_h16) { shape5 os = u->ls[0]; os.c = u->cfg.cout; if (!u->glb) u->glb = dalloc_grad(u, shape_numel(os)); nn_f32_to_h16(glogits, shape_numel(os), u->glb, gscale); glogits = u->glb; }
    const int *w = u->cfg.widths;
    float *g = u->g;
    /* head; lean mode: the 16-bit logit gradient may sit in B (unet_grad_scratch), whose MX-fp8 registration must not make the
       head ops read it as MX */
    int ghide = 0; float *gbuf = nullptr; size_t gcap = 0;
    {   /* lean: the logit gradient sits in B (lean 1) or after the logits in A (lean 2) */
        float *cand[2] = {u->gB[0], u->gA[0]}; size_t caps[2] = {u->gB_bytes, u->gA_bytes};
        for (int k = 0; k < 2 && !gbuf; k++) if (cand[k] && (const char *)glogits >= (const char *)cand[k] && (const char *)glogits < (const char *)cand[k] + caps[k]) { gbuf = cand[k]; gcap = caps[k]; }
        /* the logit gradient must read as plain 16-bit: the buffer's MX registration ends where it starts (or is dropped when it
           starts the buffer); the part below stays MX, which matters when the buffer is gout[0], the head's backward-data output */
        if (gbuf) { ghide = nn_storage(gbuf); if (ghide) { const size_t below = (size_t)((const char *)glogits - (const char *)gbuf); if (below) nn_set_storage(gbuf, below, ghide); else nn_storage_forget(gbuf); } }
    }
    block *d0 = &u->dec[0];
    shape5 os = u->ls[0]; os.c = u->cfg.cout;
    nn_set_layer(3 * L - 2);
    if (recompute()) { nn_gn_t gg = gn_out(u, d0); int r; PROF(2, r = nn_conv3d_bwd_weight_x(d0->a2, &gg, nullptr, nullptr, 0, 0, d0->ys, glogits, os, 1, 1, g + u->head.w, g + u->head.b)); if (r) rc_fail("head weight gradient"); }
    else PROF(2, nn_conv3d_bwd_weight(d0->s2, d0->ys, glogits, os, 1, 1, g + u->head.w, g + u->head.b));
    PROF(1, nn_conv3d_bwd_data(glogits, os, P(u, u->head.w), d0->ys, 1, 1, u->gout[0], u->conv_scratch));
    if (ghide) dstorage(gbuf, gcap, ghide);
    FQG(u->gout[0], d0->ys);
    /* decoder, bottom-up in the graph = i from 0 to L-2 */
    for (int i = 0; i < L - 1; i++) {
        shape5 li = u->ls[i];
        shape5 src = u->ls[i + 1]; src.c = w[i + 1];
        if (nn_get_tf32()) {
            nn_set_layer(3 * L - 3 - i); float *gup = block_bwd(u, &u->dec[i], i, u->gout[i], u->gskip[i]);   /* gB[i]: grad wrt the upsampled part; skip grad written in place */
            if (gup) { shape5 us = li; us.c = w[i + 1]; sp_halo(u, gup, us, 1); }
            if (gup) PROF(5, nn_up2_bwd(gup, src, u->gout[i + 1]));   /* nullptr: chunk mode wrote gout[i + 1] */
            FQG(u->gout[i + 1], src);
        } else {
            nn_set_layer(3 * L - 3 - i); float *gcat = block_bwd(u, &u->dec[i], i, u->gout[i], nullptr);       /* gB[i]: grad wrt concat */
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
            nn_set_layer(L + i - 1);
            sp_zero(u, gin, ds, 1);
            sp_begin(u, gin, ds, 1);
            if (recompute()) { nn_gn_t gg = gn_out(u, &u->enc[i - 1]); int r; PROF(2, r = nn_conv3d_bwd_weight_x(u->enc[i - 1].a2, &gg, nullptr, nullptr, 0, 0, u->enc[i - 1].ys, gin, ds, 3, 2, g + u->down[i - 1].w, g + u->down[i - 1].b)); if (r) rc_fail("down weight gradient"); }
            else PROF(2, nn_conv3d_bwd_weight(u->enc[i - 1].s2, u->enc[i - 1].ys, gin, ds, 3, 2, g + u->down[i - 1].w, g + u->down[i - 1].b));
            sp_end(u, gin, ds, 1);
            int acc; PROF(1, acc = nn_conv3d_bwd_data_acc(gin, ds, P(u, u->down[i - 1].w), u->enc[i - 1].ys, 3, 2, u->gskip[i - 1], u->conv_scratch));   /* gskip += */
            if (acc == 0) FQG(u->gskip[i - 1], u->enc[i - 1].ys);
            if (acc) {
                PROF(1, nn_conv3d_bwd_data(gin, ds, P(u, u->down[i - 1].w), u->enc[i - 1].ys, 3, 2, u->gA[i - 1], u->conv_scratch));
                PROF(4, nn_axpy(u->gskip[i - 1], 1.f, u->gA[i - 1], shape_numel(u->enc[i - 1].ys)));
            }
        }
    }
    if (gscale != 1.f) nn_scale(g, 1.f / gscale, u->np);   /* activation gradients were scaled for 16-bit storage */
    if (share_enc_a1()) u->sea_stale = 1;   /* the decoders' a1 buffers now hold the encoders' */
    g_in_bwd = 0;
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
static void for_each_conv3(unet *u, void (*fn)(unet *, const convp *, void *), void *arg);
/* Conv traversal follows graph order, not parameter offsets. Sort weight intervals before
   updating the complementary biases, norms and head, so each parameter has one owner. */
typedef struct { const convp *c[64]; int n; } conv_list;
static void collect_conv(unet *u, const convp *c, void *arg) {
    (void)u; conv_list *a = arg; a->c[a->n++] = c;
}
static void plain_ranges(unet *u, void (*fn)(unet *, size_t, size_t, void *), void *arg) {
    conv_list a = {0}; for_each_conv3(u, collect_conv, &a);
    for (int i = 1; i < a.n; i++) {
        const convp *c = a.c[i]; int j = i;
        while (j && a.c[j - 1]->w > c->w) { a.c[j] = a.c[j - 1]; j--; }
        a.c[j] = c;
    }
    size_t pos = 0;
    for (int i = 0; i < a.n; i++) {
        const convp *c = a.c[i]; size_t n = (size_t)c->cout * c->cin * c->k * c->k * c->k;
        if (c->w < pos || c->w > u->np || n > u->np - c->w) { fprintf(stderr, "invalid optimizer parameter partition\n"); abort(); }
        if (c->w > pos) fn(u, pos, c->w - pos, arg);
        pos = c->w + n;
    }
    if (pos < u->np) fn(u, pos, u->np - pos, arg);
}
typedef struct { float lr, b1, b2, eps, wd; int step; } adam_arg;
static void plain_adamw(unet *u, size_t off, size_t n, void *arg) {
    adam_arg *a = arg;
    nn_adamw(u->p + off, u->g + off, u->m + off, u->v + off, n, a->lr, a->b1, a->b2, a->eps, a->wd, a->step);
}
static void adamw_nonconv(unet *u, float lr, float b1, float b2, float eps, float wd, int step) {
    adam_arg a = {lr, b1, b2, eps, wd, step}; plain_ranges(u, plain_adamw, &a);
}
static void plain_ema(unet *u, size_t off, size_t n, void *arg) { nn_ema(u->ema + off, u->p + off, n, *(float *)arg); }
/* Muon for the 3^3 conv weights (viewed as [Co][Ci * 27]); biases, GroupNorm parameters and the head stay on AdamW.
   Exclude these weights from AdamW: zero gradients still allow decay and stale moments to change them. */
typedef struct { float lr, beta, wd; } muon_arg;
static void muon_conv(unet *u, const convp *c, void *arg) {
    muon_arg *a = arg; int K = c->cin * c->k * c->k * c->k; size_t n = (size_t)c->cout * K;
    size_t need = 2 * n + 2 * (size_t)c->cout * c->cout;
    if (u->muon_work_n < need) { if (u->muon_work) nn_free(u->muon_work); u->muon_work = nn_malloc(need * 4); u->muon_work_n = need; }
    nn_muon(u->p + c->w, u->g + c->w, u->muon_mom + c->w, c->cout, K, a->lr, a->beta, a->wd, u->muon_work);
}
typedef struct { float *p; const float *g; float *mom, *X, *Y, *A, *B; int Co, K; } muon_desc_t;   /* mirrors nn.cu */
typedef struct { muon_desc_t h[64]; int n; size_t pool; int maxco, maxk; } muon_build_t;
static void muon_collect(unet *u, const convp *c, void *arg) {
    muon_build_t *b = arg; int K = c->cin * c->k * c->k * c->k; size_t n = (size_t)c->cout * K;
    muon_desc_t *d = &b->h[b->n++]; d->p = u->p + c->w; d->g = u->g + c->w; d->mom = u->muon_mom + c->w; d->Co = c->cout; d->K = K;
    d->X = (float *)(uintptr_t)b->pool; b->pool += n; d->Y = (float *)(uintptr_t)b->pool; b->pool += n;
    d->A = (float *)(uintptr_t)b->pool; b->pool += (size_t)c->cout * c->cout; d->B = (float *)(uintptr_t)b->pool; b->pool += (size_t)c->cout * c->cout;
    if (c->cout > b->maxco) b->maxco = c->cout;
    if (K > b->maxk) b->maxk = K;
}
void unet_muon(unet *u, float lr_muon, float beta, float lr_adam, float b1, float b2, float eps, float wd, int step) {
    if (u->wq) { unet_adamw(u, lr_adam, b1, b2, eps, wd, step); return; }   /* packed weights: AdamW only */
    if (!u->muon_mom) { u->muon_mom = nn_malloc(u->np * 4); nn_zero(u->muon_mom, u->np * 4); }
    if (ufsm_env_on("UFSM_MUON_UNBATCHED")) {   /* reference path: one conv at a time */
        muon_arg a = {lr_muon, beta, wd};
        for_each_conv3(u, muon_conv, &a);
    } else {
        if (!u->muon_descs) {   /* build the descriptor table and the scratch pool once (offsets first, then pointers) */
            muon_build_t b = {0};
            for_each_conv3(u, muon_collect, &b);
            float *pool = nn_malloc(b.pool * 4);
            for (int i = 0; i < b.n; i++) { muon_desc_t *d = &b.h[i]; d->X = pool + (uintptr_t)d->X; d->Y = pool + (uintptr_t)d->Y; d->A = pool + (uintptr_t)d->A; d->B = pool + (uintptr_t)d->B; }
            u->muon_descs = nn_malloc(sizeof(muon_desc_t) * (size_t)b.n); nn_h2d(u->muon_descs, b.h, sizeof(muon_desc_t) * (size_t)b.n);
            u->muon_nconv = b.n; u->muon_maxco = b.maxco; u->muon_maxk = b.maxk; u->muon_work = pool;
        }
        nn_muon_batch(u->muon_descs, u->muon_nconv, u->muon_maxco, u->muon_maxk, lr_muon, beta, wd);
    }
    adamw_nonconv(u, lr_adam, b1, b2, eps, wd, step);
}
/* ANVIL II on the 3^3 conv weights (see nn.cu); AdamW elsewhere. step counts from 1; rail schedule as in the nanogpt record. */
typedef struct { float *p; const float *g; float *v0, *X, *Y, *A, *B, *v1, *E, *R; int Co, K; } anvil_desc_t;
typedef struct { anvil_desc_t h[64]; int n; size_t pool; int maxco, maxk; } anvil_build_t;
static void anvil_collect(unet *u, const convp *c, void *arg) {
    anvil_build_t *b = arg; int K = c->cin * c->k * c->k * c->k; size_t n = (size_t)c->cout * K;
    anvil_desc_t *d = &b->h[b->n++]; d->p = u->p + c->w; d->g = u->g + c->w; d->v0 = u->muon_mom + c->w; d->v1 = u->anvil_v1 + c->w; d->Co = c->cout; d->K = K;
    d->X = (float *)(uintptr_t)b->pool; b->pool += n; d->Y = (float *)(uintptr_t)b->pool; b->pool += n;
    d->A = (float *)(uintptr_t)b->pool; b->pool += (size_t)c->cout * c->cout; d->B = (float *)(uintptr_t)b->pool; b->pool += (size_t)c->cout * c->cout;
    d->E = (float *)(uintptr_t)b->pool; b->pool += (size_t)c->cout; d->R = (float *)(uintptr_t)b->pool; b->pool += (size_t)c->cout;
    if (c->cout > b->maxco) b->maxco = c->cout;
    if (K > b->maxk) b->maxk = K;
}
void unet_anvil(unet *u, float lr, float wd, int step, int steps, float lr_adam, float b1, float b2, float eps, float wd_adam) {
    if (u->wq) { unet_adamw(u, lr_adam, b1, b2, eps, wd_adam, step); return; }
    if (!u->muon_mom) { u->muon_mom = nn_malloc(u->np * 4); nn_zero(u->muon_mom, u->np * 4); }
    if (!u->anvil_v1) { u->anvil_v1 = nn_malloc(u->np * 4); nn_zero(u->anvil_v1, u->np * 4); }
    if (!u->anvil_descs) {
        anvil_build_t b = {0};
        for_each_conv3(u, anvil_collect, &b);
        float *pool = nn_malloc(b.pool * 4); nn_zero(pool, b.pool * 4);
        for (int i = 0; i < b.n; i++) { anvil_desc_t *d = &b.h[i]; d->X = pool + (uintptr_t)d->X; d->Y = pool + (uintptr_t)d->Y; d->A = pool + (uintptr_t)d->A; d->B = pool + (uintptr_t)d->B; d->E = pool + (uintptr_t)d->E; d->R = pool + (uintptr_t)d->R; }
        u->anvil_descs = nn_malloc(sizeof(anvil_desc_t) * (size_t)b.n); nn_h2d(u->anvil_descs, b.h, sizeof(anvil_desc_t) * (size_t)b.n);
        u->anvil_nconv = b.n; u->muon_maxco = b.maxco; u->muon_maxk = b.maxk; u->anvil_pool = pool;
        /* lane energy starts at 1 so the first equalisation is a no-op */
        for (int i = 0; i < b.n; i++) { float *one = malloc((size_t)b.h[i].Co * 4); for (int k = 0; k < b.h[i].Co; k++) one[k] = 1.f; nn_h2d(b.h[i].E, one, (size_t)b.h[i].Co * 4); free(one); }
    }
    /* fast-rail beta: 0.85 -> 0.93 over the first 240 steps, 0.93 until the slow rail engages at step 514 (then 0.85, blend 0.4385) */
    const int engage = 514, bwarm = 240; float bf, w;
    if (step >= engage) { bf = 0.85f; w = 0.4385f; }
    else { bf = step < bwarm ? 0.85f + (0.93f - 0.85f) * (float)step / bwarm : 0.93f; w = 1.f; }
    (void)steps;
    nn_anvil_batch(u->anvil_descs, u->anvil_nconv, u->muon_maxco, u->muon_maxk, lr, bf, 0.98f, w, 0.95f, 0.9f, wd);
    adamw_nonconv(u, lr_adam, b1, b2, eps, wd_adam, step);
}
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
        size_t so = 0, qo = 0;
        for (int i = 0; i < n; i++) { int T = cs[i]->k * cs[i]->k * cs[i]->k; u->wq_qoff[i] = qo; qo += nn_wq_bytes(cs[i]->cout, cs[i]->cin, T, bits); u->wq_soff[i] = so; so += nn_wq_nblocks(cs[i]->cout, cs[i]->cin, T); }
        u->wq_n = n;
        u->wq_q = nn_malloc(qo + 64); u->wq_sc = nn_malloc(so + 64); u->wq_qe = nn_malloc(qo + 64); u->wq_sce = nn_malloc(so + 64);
        if (bits == 4) { u->wq_r = nn_malloc(u->np); u->wq_rsc = nn_malloc(so + 64); u->wq_re = nn_malloc(u->np); u->wq_rsce = nn_malloc(so + 64); }
        /* initial packing of the fp32 values (round to nearest), then the shadows take the grid values */
        for (int i = 0; i < n; i++) {
            const convp *c = cs[i]; int T = c->k * c->k * c->k; size_t qo = u->wq_qoff[i];
            nn_wq_pack(u->p + c->w, u->wq_q + qo, u->wq_sc + u->wq_soff[i], c->cout, c->cin, T, bits, 0);
            nn_wq_pack(u->ema + c->w, u->wq_qe + qo, u->wq_sce + u->wq_soff[i], c->cout, c->cin, T, bits, 0);
            if (bits == 4) {   /* residual = what the fp4 grid dropped at initial packing */
                nn_wq_residual(u->p + c->w, u->wq_q + qo, u->wq_sc + u->wq_soff[i], u->wq_r + c->w, u->wq_rsc + u->wq_soff[i], c->cout, c->cin, T, bits);
                nn_wq_residual(u->ema + c->w, u->wq_qe + qo, u->wq_sce + u->wq_soff[i], u->wq_re + c->w, u->wq_rsce + u->wq_soff[i], c->cout, c->cin, T, bits);
            }
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
    adamw_nonconv(u, lr, b1, b2, eps, wd, step);
    for (int i = 0; i < u->wq_n; i++) {
        const convp *c = wq_conv(u, i); int T = c->k * c->k * c->k; size_t qo = u->wq_qoff[i];
        nn_wq_adamw(u->wq_q + qo, u->wq_sc + u->wq_soff[i], u->wq_r ? u->wq_r + c->w : nullptr, u->wq_r ? u->wq_rsc + u->wq_soff[i] : nullptr, u->g + c->w, u->m + c->w, u->v + c->w, c->cout, c->cin, T, u->wq, lr, b1, b2, eps, wd, step, (unsigned)step * 7919u + (unsigned)i);
        nn_wq_unpack(u->wq_q + qo, u->wq_sc + u->wq_soff[i], u->p + c->w, c->cout, c->cin, T, u->wq);
    }
}
static void wq_ema(unet *u, float decay) {
    plain_ranges(u, plain_ema, &decay);
    for (int i = 0; i < u->wq_n; i++) {
        const convp *c = wq_conv(u, i); int T = c->k * c->k * c->k; size_t qo = u->wq_qoff[i];
        nn_wq_ema(u->wq_qe + qo, u->wq_sce + u->wq_soff[i], u->wq_re ? u->wq_re + c->w : nullptr, u->wq_re ? u->wq_rsce + u->wq_soff[i] : nullptr, u->wq_q + qo, u->wq_sc + u->wq_soff[i], u->wq_r ? u->wq_r + c->w : nullptr, u->wq_r ? u->wq_rsc + u->wq_soff[i] : nullptr, c->cout, c->cin, T, u->wq, decay, (unsigned)(u->ema_step++) * 104729u + (unsigned)i);
        nn_wq_unpack(u->wq_qe + qo, u->wq_sce + u->wq_soff[i], u->ema + c->w, c->cout, c->cin, T, u->wq);
    }
}
void unet_wquant(unet *u, unsigned seed) { (void)u; (void)seed; }   /* kept for API compatibility: packed storage makes it unnecessary */

/* ---- checkpoints ---- */
static int g_loaded_sparse = 0, g_loaded_wq = 0, g_loaded_muon = 0;   /* set by the header parser, applied by unet_load */
/* debug: per block, the GroupNorm statistics of the last forward (mean / rstd of a1 and a2 per sample and group:
   non-finite counts and extremes) and the largest |weight| of each conv. Localises a non-finite forward. */
void unet_debug_stats(const unet *u) {
    int L = u->cfg.nlev;
    for (int i = 0; i < 2 * L - 1; i++) {
        const block *b = i < L ? &u->enc[i] : &u->dec[2 * L - 2 - i];
        const char *nm = i < L ? "enc" : "dec"; int li = i < L ? i : 2 * L - 2 - i;
        int G = G_of(u, b->c1.cout); size_t n = (size_t)b->xs.n * G;
        const float *arr[4] = {b->m1, b->r1, b->m2, b->r2}; const char *an[4] = {"m1", "r1", "m2", "r2"};
        fprintf(stderr, "%s%d:", nm, li);
        for (int a = 0; a < 4; a++) {
            if (!arr[a]) { fprintf(stderr, " %s -", an[a]); continue; }
            float *h = malloc(n * 4); nn_d2h(h, arr[a], n * 4);
            fprintf(stderr, " %s[", an[a]);
            for (int s_ = 0; s_ < b->xs.n; s_++) { size_t nf = 0; double mx = 0; for (int g = 0; g < G; g++) { float v = h[(size_t)s_ * G + g]; if (!isfinite(v)) nf++; else if (fabs(v) > mx) mx = fabs(v); } fprintf(stderr, "%s%zu nf, max %.3g", s_ ? "; " : "", nf, mx); }
            fprintf(stderr, "]"); free(h);
        }
        size_t nw1 = (size_t)b->c1.cout * b->c1.cin * 27, nw2 = (size_t)b->c2.cout * b->c2.cin * 27;
        float *h = malloc((nw1 > nw2 ? nw1 : nw2) * 4); double w1 = 0, w2 = 0;
        nn_d2h(h, P(u, b->c1.w), nw1 * 4); for (size_t k = 0; k < nw1; k++) if (fabs(h[k]) > w1) w1 = fabs(h[k]);
        nn_d2h(h, P(u, b->c2.w), nw2 * 4); for (size_t k = 0; k < nw2; k++) if (fabs(h[k]) > w2) w2 = fabs(h[k]);
        fprintf(stderr, " max|w| %.3g %.3g\n", w1, w2); free(h);
    }
}
/* debug: per sample, non-finite count and largest |value| of a stored activation tensor (16-bit or fp32 storage; not MX) */
static void dbg_tensor(const char *nm, const float *t, shape5 s) {
    if (!t) { fprintf(stderr, " %s -", nm); return; }
    size_t per = (size_t)s.c * shape_spatial(s), n = (size_t)s.n * per;
    int h16 = ABF; void *h = malloc(n * (h16 ? 2 : 4)); nn_d2h(h, t, n * (h16 ? 2 : 4));
    fprintf(stderr, " %s[", nm);
    for (int b = 0; b < s.n; b++) {
        size_t nf = 0; double mx = 0;
        for (size_t k = (size_t)b * per; k < (size_t)(b + 1) * per; k++) {
            float v;
            if (!h16) v = ((float *)h)[k];
            else if (nn_get_f16()) { _Float16 x; memcpy(&x, (uint16_t *)h + k, 2); v = (float)x; }
            else { uint32_t u32 = (uint32_t)((uint16_t *)h)[k] << 16; memcpy(&v, &u32, 4); }
            if (!isfinite(v)) nf++; else if (fabs(v) > mx) mx = fabs(v);
        }
        fprintf(stderr, "%s%zu nf, max %.3g", b ? "; " : "", nf, mx);
    }
    fprintf(stderr, "]"); free(h);
}
void unet_debug_acts(const unet *u) {
    if (act_mx8()) { fprintf(stderr, "unet_debug_acts: MX storage not supported\n"); return; }   /* TODO dequantise via lp_mx*_to_f32 */
    int L = u->cfg.nlev;
    for (int i = 0; i < L; i++) {
        const block *b = &u->enc[i];
        fprintf(stderr, "enc%d:", i); dbg_tensor("a1", b->a1, b->ys); dbg_tensor("a2", b->a2, b->ys); dbg_tensor("s2", b->s2, b->ys);
        if (i < L - 1) { shape5 ds = u->ls[i + 1]; ds.c = u->cfg.widths[i]; dbg_tensor("down", u->downo[i], ds); }
        fprintf(stderr, "\n");
    }
    for (int i = L - 2; i >= 0; i--) { const block *b = &u->dec[i]; fprintf(stderr, "dec%d:", i); dbg_tensor("a1", b->a1, b->ys); dbg_tensor("a2", b->a2, b->ys); fprintf(stderr, "\n"); }
}
int unet_save(const unet *u, const char *path, int step, const char *extra) {
    char tmp[1400];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f) return -1;
    fprintf(f, "UFSM{\"nlev\":%d,\"widths\":[", u->cfg.nlev);
    for (int i = 0; i < u->cfg.nlev; i++) fprintf(f, "%s%d", i ? "," : "", u->cfg.widths[i]);
    fprintf(f, "],\"cin\":%d,\"cout\":%d,\"G\":%d,\"down_norm\":%d,\"nparams\":%zu,\"step\":%d,\"sparse24\":%d,\"wq\":%d,\"muon_mom\":%d,\"extra\":%s}\n", u->cfg.cin, u->cfg.cout, u->cfg.G, u->cfg.down_norm, u->np, step, u->sparse24, u->wq, u->muon_mom != nullptr, extra ? extra : "{}");
    float *h = malloc(u->np * 4);
    const float *arrs[4] = {u->p, u->ema, u->m, u->v};
    for (int a = 0; a < 4; a++) { nn_d2h(h, arrs[a], u->np * 4); if (fwrite(h, 4, u->np, f) != u->np) { fclose(f); free(h); return -1; } }
    if (u->muon_mom) {
        nn_d2h(h, u->muon_mom, u->np * 4);
        if (fwrite(h, 4, u->np, f) != u->np) { fclose(f); free(h); return -1; }
    }
    free(h);
    if (fclose(f)) return -1;
    return rename(tmp, path);
}

static int read_header(FILE *f, unet_cfg *cfg, int *step, size_t *np) {
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "UFSM", 4)) return -1;
    char line[UFSM_CHECKPOINT_HEADER];
    if (!fgets(line, sizeof line, f) || !strchr(line, '\n')) return -1;
    memset(cfg, 0, sizeof *cfg);
    const char *p;
    if ((p = strstr(line, "\"nlev\":"))) cfg->nlev = atoi(p + 7);
    if ((p = strstr(line, "\"widths\":["))) { p += 10; for (int i = 0; i < cfg->nlev && i < UNET_MAXLEV; i++) { cfg->widths[i] = atoi(p); p = strchr(p, ','); if (!p) break; p++; } }
    if ((p = strstr(line, "\"cin\":"))) cfg->cin = atoi(p + 6);
    if ((p = strstr(line, "\"cout\":"))) cfg->cout = atoi(p + 7);
    if ((p = strstr(line, "\"G\":"))) cfg->G = atoi(p + 4);
    if ((p = strstr(line, "\"down_norm\":"))) cfg->down_norm = atoi(p + 12);
    if ((p = strstr(line, "\"nparams\":"))) *np = (size_t)atoll(p + 10);
    if ((p = strstr(line, "\"step\":"))) *step = atoi(p + 7);
    if ((p = strstr(line, "\"sparse24\":"))) g_loaded_sparse = atoi(p + 11);
    if ((p = strstr(line, "\"wq\":"))) g_loaded_wq = atoi(p + 5);
    if ((p = strstr(line, "\"muon_mom\":"))) g_loaded_muon = atoi(p + 11);
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
    g_loaded_sparse = 0; g_loaded_wq = 0; g_loaded_muon = 0;
    if (read_header(f, &cfg, &step, &np) || np != u->np) { fclose(f); return -1; }
    float *h = malloc(u->np * 4);
    float *arrs[4] = {u->p, u->ema, u->m, u->v};
    for (int a = 0; a < 4; a++) {
        if (fread(h, 4, u->np, f) != u->np) { if (a < 2) { fclose(f); free(h); return -1; } break; }
        nn_h2d(arrs[a], h, u->np * 4);
    }
    if (g_loaded_muon) {
        if (fread(h, 4, u->np, f) != u->np) { fclose(f); free(h); return -1; }
        if (!u->muon_mom) u->muon_mom = nn_malloc(u->np * 4);
        nn_h2d(u->muon_mom, h, u->np * 4);
    } else if (u->muon_mom) nn_zero(u->muon_mom, u->np * 4);   /* old checkpoints have no Muon state */
    free(h);
    fclose(f);
    if (g_loaded_sparse) unet_set_sparse24(u, 1);
    if (g_loaded_wq) unet_set_wq(u, g_loaded_wq);
    return step;
}

/* Warm start into a wider head (band affinity outputs): load a checkpoint of the same net with fewer outputs. The body is
   copied as is, head rows < keep (and their biases) are kept, new rows get zero weights and bias new_bias (zero optimiser
   state), and the parameters after the head (down_norm GroupNorms) shift. Returns the checkpoint step or -1. */
int unet_load_grow(unet *u, const char *path, int keep, float new_bias) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    unet_cfg oc; int step = 0; size_t onp = 0;
    g_loaded_sparse = 0; g_loaded_wq = 0; g_loaded_muon = 0;
    if (read_header(f, &oc, &step, &onp)) { fclose(f); return -1; }
    const unet_cfg *c = &u->cfg;
    int same = oc.nlev == c->nlev && oc.cin == c->cin && oc.G == c->G && oc.down_norm == c->down_norm && oc.cout <= c->cout && keep <= oc.cout;
    for (int i = 0; same && i < c->nlev; i++) same = oc.widths[i] == c->widths[i];
    const size_t hw = u->head.w, ci = (size_t)u->head.cin, tail_new = u->head.b + (size_t)c->cout, tail = u->np - tail_new;
    if (!same || onp != hw + ci * oc.cout + oc.cout + tail || g_loaded_sparse || g_loaded_wq) {
        fprintf(stderr, "grow-head: %s is not the same network with <= %d outputs\n", path, c->cout); fclose(f); return -1;
    }
    float *h = malloc(onp * 4), *n = malloc(u->np * 4);
    float *arrs[5] = {u->p, u->ema, u->m, u->v, u->muon_mom};
    int na = g_loaded_muon ? 5 : 4;
    if (g_loaded_muon && !u->muon_mom) { u->muon_mom = nn_malloc(u->np * 4); arrs[4] = u->muon_mom; }
    for (int a = 0; a < na; a++) {
        if (fread(h, 4, onp, f) != onp) { free(h); free(n); fclose(f); return -1; }
        const int weights = a < 2;   /* p and ema: new biases = new_bias; optimiser state: zero */
        memcpy(n, h, hw * 4);
        for (int co = 0; co < c->cout; co++)
            for (size_t k = 0; k < ci; k++) n[hw + (size_t)co * ci + k] = co < keep ? h[hw + (size_t)co * ci + k] : 0.f;
        for (int co = 0; co < c->cout; co++) n[u->head.b + co] = co < keep ? h[hw + ci * oc.cout + co] : weights ? new_bias : 0.f;
        memcpy(n + tail_new, h + hw + ci * oc.cout + oc.cout, tail * 4);
        nn_h2d(arrs[a], n, u->np * 4);
    }
    if (!g_loaded_muon && u->muon_mom) nn_zero(u->muon_mom, u->np * 4);
    free(h); free(n); fclose(f);
    return step;
}

/* Partial warm start across architectures (e.g. a deeper / wider net): every conv and GroupNorm whose shape matches the
   same layer of the checkpoint's net (enc / down / dec by level, head, down_norm) is copied into the weights and the EMA;
   everything else keeps its fresh initialisation, optimiser state starts at zero. Returns the number of copied tensors or -1. */
static int copy_conv(unet *u, const unet *o, const convp *a, const convp *b) {
    if (a->cin != b->cin || a->cout != b->cout || a->k != b->k || a->stride != b->stride) return 0;
    const size_t nw = (size_t)a->cout * a->cin * a->k * a->k * a->k;
    nn_d2d(u->p + a->w, o->p + b->w, nw * 4); nn_d2d(u->ema + a->w, o->ema + b->w, nw * 4);
    nn_d2d(u->p + a->b, o->p + b->b, (size_t)a->cout * 4); nn_d2d(u->ema + a->b, o->ema + b->b, (size_t)a->cout * 4);
    return 1;
}
static int copy_gn(unet *u, const unet *o, const gnp *a, const gnp *b) {
    if (a->c != b->c) return 0;
    nn_d2d(u->p + a->gamma, o->p + b->gamma, (size_t)a->c * 4); nn_d2d(u->ema + a->gamma, o->ema + b->gamma, (size_t)a->c * 4);
    nn_d2d(u->p + a->beta, o->p + b->beta, (size_t)a->c * 4); nn_d2d(u->ema + a->beta, o->ema + b->beta, (size_t)a->c * 4);
    return 1;
}
int unet_init_from(unet *u, const char *path) {
    unet_cfg oc; int step;
    if (unet_peek(path, &oc, &step) || oc.cin != u->cfg.cin) return -1;
    unet *o = unet_create(&oc);
    if (unet_load(o, path) < 0) { unet_free(o); return -1; }
    int n = 0;
    const int L = u->cfg.nlev < oc.nlev ? u->cfg.nlev : oc.nlev;
    for (int i = 0; i < L; i++) {
        const block *a = &u->enc[i], *b = &o->enc[i];
        n += copy_conv(u, o, &a->c1, &b->c1) + copy_gn(u, o, &a->n1, &b->n1) + copy_conv(u, o, &a->c2, &b->c2) + copy_gn(u, o, &a->n2, &b->n2);
    }
    for (int i = 0; i < L - 1; i++) {
        n += copy_conv(u, o, &u->down[i], &o->down[i]);
        if (u->cfg.down_norm && oc.down_norm) n += copy_gn(u, o, &u->dn[i], &o->dn[i]);
        const block *a = &u->dec[i], *b = &o->dec[i];   /* dec[i] works at level i in both nets */
        n += copy_conv(u, o, &a->c1, &b->c1) + copy_gn(u, o, &a->n1, &b->n1) + copy_conv(u, o, &a->c2, &b->c2) + copy_gn(u, o, &a->n2, &b->n2);
    }
    n += copy_conv(u, o, &u->head, &o->head);
    unet_free(o);
    nn_zero(u->m, u->np * 4); nn_zero(u->v, u->np * 4);
    if (u->muon_mom) nn_zero(u->muon_mom, u->np * 4);
    return n;
}

int unet_start_sheet(unet *u) {
    if (u->cfg.cin!=4 || u->cfg.cout!=2 || u->wq) {
        fputs("winding initialization needs a four-input/two-output donor with FP32 master weights\n",stderr); return -1;
    }
    float *arrays[5]={u->p,u->ema,u->m,u->v,u->muon_mom};
    const convp *stem=&u->enc[0].c1;
    for (int a=0;a<5;a++) if (arrays[a]) {
        for (int c=0;c<stem->cout;c++) nn_zero(arrays[a]+stem->w+((size_t)c*4+1)*27,27*4);
        nn_zero(arrays[a]+u->head.w+u->head.cin,(size_t)u->head.cin*4);
        nn_zero(arrays[a]+u->head.b+1,4);
    }
    return 0;
}
