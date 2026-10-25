#include "unet.h"
#include "attn.h"
#include "checkpoint.h"
#include "nn_lp.h"
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
    int single;                     /* cfg.dec_convs 1: conv1 writes the output a2 (stats m2 / r2, norm n2 = n1); no c2, a1 unused */
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
    size_t wq_qoff[UNET_MAXCONV], wq_soff[UNET_MAXCONV]; int wq_n; unsigned ema_step;   /* per 3^3 conv (for_each_conv3 order): offsets into the packed arrays */
    int using_ema;
    block *rb[UNET_MAXLEV]; int nrb_any; float *gres[UNET_MAXLEV]; size_t gres_cap[UNET_MAXLEV];   /* cfg.enc_blocks: residual blocks, the identity gradient per level */
    float *gn_frz; size_t gn_frz_n;   /* unet_gn_freeze: per-sample GroupNorm statistics used by inference forwards instead of the window's */
    double *gn_acc; long gn_acc_n;     /* unet_gn_accumulate: sums of the window statistics (mean, mean^2 per mean slot, variance per rstd slot) */
    float *l0run, *l0live; size_t l0live_n;   /* cfg.l0_norm ema: [mean G | var G] per level-0 site (l0_site order), three sets of 8 G: the
                                                 running statistics of the live weights, of the EMA weights (unet_ema blends the last
                                                 batch's in with the weights' decay: first order in the weight change), the last training
                                                 forward's; l0live: the convs' statistics while constant ([mean | rstd] x n G per site) */
    long l0_n; float l0_mom; int l0_warm, l0_c, l0_rc;   /* ema: updates so far, momentum, warm-up; l0_c: this forward's level-0 statistics are
                                                          constants; l0_rc: unet_l0_recal's forwards so far (-1 off) */
    float *wsp, ws_gain;          /* cfg.l0_norm 3 (ws): the weights the forward reads (live, the level-0 convs standardised); SiLU's gain */
    float *l0s_st, *l0s_tmp; size_t l0s_n, l0s_nng, l0s_tmpn; unsigned l0s_seed[4], l0s_fwd;   /* cfg.l0_norm 4 (slab): per site [mean | rstd]
                                  per (sample, slab, group) (nng each), then zeros, ones (nng), ones, zeros (widths[0]: gamma, beta of the fp32
                                  path's plain normalisation); the fp32 path's three slab-layout temporaries; the sites' SR seeds */
    const float *l0s_reg[4];      /* slab kernel path: the statistics arrays this net registered (lp_gn_slab_reg) */
    float *l0st_g2, *l0st_up; size_t l0st_g2n, l0st_upn;   /* level-0 slab stream (stage S3): dec[0]'s d/d a2, its up transient */
    float *l0st_poison;           /* UFSM_L0_STREAM_POISON: a NaN plane source */
    int l0st, l0st_dir;           /* this forward / backward runs the level-0 slab stream (2: recomputing); its backward phase's direction (+1 down, -1 up) */
    unsigned l0st_rc[4]; int l0st_rk[4];   /* the forward's level-0 ops (stem, enc[0] conv2, dec[0] conv1, conv2): first seed draw, draws per window */
    int l0ring;                   /* this training build keeps level-0 tensors in VMM slab rings (stage S4, UFSM_L0_RING with stream 2) */
    float *l0r_ups; size_t l0r_upsn;   /* ring mode: silu(gn(dec[1].a2)) at level 1, the per-slab up transient's source */
    int (*l0h_fwd)(void *, int, int, const void *); int (*l0h_bwd)(void *, int, int, void *); void *l0h_ctx;   /* unet_set_l0_head */
    void *l0h_lg; size_t l0h_lgn;      /* the stream head: one slab's fp16 logits [cout][planes h w] */
    unsigned l0h_c; int l0h_k;         /* the forward head's first seed draw and draws per slab (the backward's recomputes replay them) */
    int l0hb;                          /* this ring build runs the head per slab: dec[0].a2 and gout[0] in rings, the skip gradient its own */
    int l0rx; const void *l0x_ct; int l0x_held[16];   /* the network input in a ring (--gpu-input: rebuilt per slab from the CT channel); slot -> slab */
    void *l0r_hsk; size_t l0r_hskn;    /* the skip gradient's pinned host copy (stage S4's host mode: the device holds a ring) */
    size_t sc_w; float sc_c; float *stem_b;   /* scale_cond: offset of the conditioning vector, current scale code, effective stem bias */
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
    int mx_c, mx_hb; float mx_hb_mult;       /* Muon variants: Muon-C, Hyperball (eta = mx_hb_mult x the Muon lr) */
    void *mx_mcd, *mx_nsd; float *mx_pool; int mx_ns_maxm, mx_ns_maxn;   /* Muon-C tables and one frequency's scratch */
    int *mx_hbf; float *mx_R; double *mx_part; int mx_R_ok;   /* Hyperball flags, radii (device), sums; R valid */
    float mx_R_load[UNET_MAXCONV]; int mx_R_nload;           /* radii read from a checkpoint, uploaded at the next step */
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
    void *h_a1[UNET_MAXLEV];                 /* shared encoder a1 offload: pinned host copies of the encoders' a1 */
    void *ev_a1[UNET_MAXLEV][4];             /* (start, done) of the copy out after the encoder's forward, of the copy back */
    int a1_out[UNET_MAXLEV];                 /* a copy out is pending (the decoder's conv1 must wait for it) / has a host copy */
    int xin_shared, xin_out, a2_stale, fwd_train;       /* training: the network input lives in dec[0].a2's buffer (host copy in between); the
                                                buffer holds the input since a backward (a repeated backward re-runs dec[0].c2) */
    size_t xin_bytes; void *h_xin, *ev_xin[4];
    const void *rad;                         /* unet_set_input_rad (train --gpu-input): the 16-bit input is the CT alone, these radial records
                                                build the 4 MX input channels (nn_input_rad) */
    int a0_share, a2_out;                    /* level-0 A in dec[0].a2's buffer; enc[0].a2 on the host from its forward to dec[0]'s conv1 weight
                                                gradient, its buffer holding the logit gradient in between (and the input in enc[0]'s backward) */
    void *h_a2, *ev_a2[4];
    void *ev_head;                           /* l3b: recorded in enc[0]'s backward once the batch head (X2) and the radial records are read for the last time */
    int xin_inf, lg_t1; size_t t1_bytes; float *lg_p; size_t lg_bytes;   /* lg_t1: the logits in lg_p (T1, or Y with level-1 slabs), lg_bytes registered */
    int lg_x;                                /* lg_t1 in X itself (slabs, one fp16 logit channel, a 16-wide level 0 whose Y cannot hold them): the head
                                                runs per z-slab into the slab temp, each slab copied to the logits' planes at X's start, X data planes
                                                that slab and the earlier ones have consumed (unet_forward_x) */
    int slab, slab1, slab_s; float *slab_tmp, *slab_carry, *t1p;   /* slab1: level 1 in place too (Y = enc[1].a2's buffer; UFSM_INFER_SLAB_L1=0 off) */
    int ct_mode, ct_stem; float *slab_xin;    /* unet_set_ct_input: the caller passes the uint8 CT window (unet_forward_ct); with slabs and an MX-fp8 stem
                                                 input the stem builds its input per slab from it (no full-window input) */
    /* coarse offload (UFSM_INFER_OFFLOAD, with level-1 slabs): the last r_bytes of X's data region (R, from plane zr on) go to pinned host
       memory (h_r) from enc[0]'s end to dec[0]'s conv1; R holds every coarse buffer meanwhile (T1, T2, the skips below level 1, the
       level-1 slab temp and carry), Y the level-0 slab temp and the stem input (free there) and the down-conv output that enc[1]'s conv1
       reads in place; dec[0]'s conv1 slabs use R as temp, then Y's consumed coarse planes (slab_tmpd: its own temp if that schedule
       does not fit) */
    int p3, zr, r_out, r_in; float *R, *Xp, *Yp; size_t r_bytes, x_bytes, y_bytes; void *h_r, *ev_r[4];
    float *slab_tmp1, *slab_carry1, *slab_tmpd;
    struct { const uint8_t *ct; float mean, isd; const float *dyo, *dxo; int axis, h16, on; } ctin;   /* in-place z-slab convs at level 0 (UFSM_INFER_SLAB): enc[0] / dec[0] a1 and a2 all in
                                                         X = enc[0].a2's buffer; slab_s output planes per launch into slab_tmp (+ one carry plane) */     /* inference aliasing (UFSM_INFER_ALIAS, default on): the MX network input in enc[0].a2's buffer
                                                (dead from the stem to enc[0].c2), the fp32 logits in T1 (dec[0].a1, dead after dec[0].c2) */
    int gco;                                 /* the coarse levels' gout in dec[0]'s buffers: at the end of dec[0].a2's (dead from dec[0]'s
                                                backward to enc[0]'s), gout[1] at the end of dec[0].a1's during dec[0]'s backward */
    float *gout_x1, *gout_home1;             /* gco: gout[1] during dec[0]'s backward / afterwards */
    int dno, dn_out;                         /* the down-conv outputs in dec[0].a2's buffer, on the host from the next encoder's conv1 to its
                                                backward (dn_out: the copies out were issued) */
    void *h_dn[UNET_MAXLEV], *ev_dn[UNET_MAXLEV][4];
    int sko, sk_out;                         /* the coarse skips enc[1 .. L-2].a2 at the start of dec[0].a2's buffer, on the host from their
                                                decoder's conv1 to its backward */
    void *h_sk[UNET_MAXLEV], *ev_sk[UNET_MAXLEV][4];
    /* three level-0 buffers (UFSM_L0_3BUF, the chain above at an 8- or 16-channel level 0; l3_want): X1 = enc[0].a1 = dec[0].a1 = gout[0] =
       gskip[0] = gB[0] (the input in enc[0]'s backward), X2 = enc[0].a2 = dec[0].a2 (the batch head at the forward's start, the coarse
       backward homes of gco / dno / sko, enc[0]'s a1 in its backward), X3 = gA[0] = the input in the forward (then the coarse forward homes)
       = the fp16 logits and their in-place gradient, the batch tail at its end, the skip again from dec[0]'s backward to enc[0]'s. The
       head's input gradient is never stored (nn_gn_silu_bwd_head recomputes it inside dec[0]'s GroupNorm backward); the skip comes back
       twice (after dec[0]'s up transient into X2, after dec[0]'s backward into X3), the encoder's a1 into X2 after down0's backward */
    int l3b, l3s, l3_no, l3_skip_in;         /* l3s: the copy schedule (l3_sched, latched at the build); l3_no: unavailable (a host buffer), the four-buffer layout; l3_skip_in: dec[0]'s conv1 weight gradient
                                                brings the skip back after building its up transient */
    int l3_ct;                               /* this training forward's input is rebuilt in enc[0]'s backward from its CT channel (--gpu-input: the 16-bit
                                                CT in X2 and the radial records), the CT on the host meanwhile (2 B per voxel instead of the 5 B input) */
    float *l3x[4]; size_t l3xb[4], l3_lo;    /* X1..X3 and their bytes; X3's bytes below the batch tail */
    float *skf[UNET_MAXLEV], *skb[UNET_MAXLEV], *dnf[UNET_MAXLEV], *dnb[UNET_MAXLEV];   /* the coarse skips / down-conv outputs: forward homes (X3),
                                                backward homes (X2) */
    size_t bat_head, bat_tail;               /* unet_set_batch_bytes: the trainer's batch [head | tail] (the three-buffer layout sizes X3 for the tail) */
    size_t logits_bytes;                    /* lean: logits at the start of A, the 16-bit logit gradient after them */
    int logits_h16;                          /* lean 2: fp16 logits, so the 16-bit logit gradient fits after them in A */
    int lh16_req;                            /* unet_set_fp16_logits: fp16 logits where the MX head writes them into their own buffer */
    int head_ext;                            /* unet_set_head_ext: training runs the head per z-slab outside (no whole-window logits) */
    convp ds[UNET_MAXLEV];                   /* cfg.ds_levels: the deep-supervision 1^3 heads of decoder levels 1..ds_levels */
    float *ds_lg[UNET_MAXLEV];               /* their fp32 logits (training builds) */
    const float *ds_gl[UNET_MAXLEV];         /* their fp32 logit gradients for the next backward (unet_set_ds_grad) */
    void *ds_g16, *ds_tmp; size_t ds_g16_n, ds_tmp_n;   /* backward scratch: 16-bit logit gradient, the head's input gradient in gout's storage */
    float wd_nonconv;                        /* unet_set_wd_nonconv: weight decay multiplier for the non-3^3 parameters (GroupNorm, biases,
                                                the head) under Muon / ANVIL; 0 (default): only the conv weights decay */
    void (*head_cb)(void *); void *head_ctx; /* ... and its backward at the head's step (unet_head_fwd_z / unet_head_bwd_z per slab) */
    float *rc_extra; size_t rc_extra_bytes;  /* training transient for the upsampled decoder input when B is too small */
    int head_c0, head_nc;                    /* inference builds: only logit channels [c0, c0 + nc) (nc 0: all) */
    int split_side, split_h0;                /* spatial split (unet_set_split): side 0 / 1, level-0 halo planes; h0 0 = off */
    unet_halo_fn split_halo, split_begin, split_end;
    /* cfg.attn_blocks: the bottleneck transformer stage (src/attn.h). tb is the stage's output block (a2 = the new raw output, m2 / r2 its
       GroupNorm statistics, n2 = the bottleneck's n2 parameters; no convs of its own); the token rows: tok slot 0 the converted input,
       slot 1 after the PE (block 0's input), slot k + 1 block k's input, slot tbn + 1 the output */
    int tbn; at_dims tbd; at_bp tbp[UNET_MAXTB]; convp tbm[UNET_MAXTB][4]; size_t peg_w, peg_b, reg_off;
    block tb; at_ws tbw; at_geom tbg;
    float *tb_cs, *tb_tok, *tb_wsb, *tb_g, *tb_gx, *tb_dt, *tb_dt2, *tb_zero;
    float *tb_vr; size_t vr_off;   /* value residual (cfg.attn_vres): [v1 | dv1 | own per block application] token rows; the per-head lerps (blocks 1.., H each) */
    float *tb_keep; at_ws tbk[UNET_MAXTA];   /* unet_set_attn_keep: each block application's forward tensors of the last training forward (tbk[a],
                                                the backward scratch shared with tbw), read by its backward instead of a recompute */
    int tba;                                 /* block applications: tbn x cfg.attn_repeat (application a runs block a % tbn) */
    /* cfg.attn_axis 1 / attn_pe 3: the scroll geometry. tb_rb: one device buffer for the ray table, lengths, axis dims, ray positions, the
       stencil weight sums, the grid tokens' scroll positions and the axis frames (at_rays_plan); the host side of a plan; the axis records */
    at_rays tbr; float *tb_rb, *tb_rayh, *tb_cenh, *tb_axb, *tb_axh; int *tb_lensh, *tb_adh; int tb_Q, tb_Nt, tb_Nr, tb_axn, tb_axs, tb_axdirty, tb_axwarn;
    int nrg, tb_memn; size_t mem_gate, mem_woff; float *tb_memin, *tb_memf, *tb_memout, *tb_memfh; const float *tb_memsrc;   /* register rows (attn_reg + attn_mem); window memory */
    int tb_nv, tb_vn; size_t view_w, view_b; float *tb_vxp, *tb_ve; const uint8_t *tb_vsrc;   /* scan fusion: view rows per sample, the crops' count; patches, embeddings */
    float *tb_gpos; float *tb_cen; long tb_Tr;
    float *tb_dfx[AT_DFM_MAXL], *tb_dfcs;    /* deformable attention: the finer levels' LayerNorm'd tokens (at_geom.dfx), their channel-major staging */
    float *tb_dfr;                           /* ... cfg.attn_dinit 1: the grid tokens' frames (at_geom.dfr; inside tb_rb, filled by tb_geom) */
    size_t tb_p0;                            /* the attention stage's parameters: [tb_p0, np) (0 without a stage) */
    float trunk_lr;                          /* unet_set_trunk_lr: lr multiplier outside the stage (0 = 1) */
    const float *tb_xg;                      /* unet_set_token_grad: extra gradient at the stage output tokens */
};

static int G_of(const unet *u, int c) { return u->cfg.G < c ? u->cfg.G : c; }
/* output channels of the coarse block at level i that the decoder upsamples into level i - 1: dec[i] (cfg.dec_widths, 0 = the
   encoder width), the bottom level's enc[L-1] */
static int dw_of(const unet_cfg *c, int i) { return i >= c->nlev - 1 || !c->dec_widths[i] ? c->widths[i] : c->dec_widths[i]; }
#define UPW(i) dw_of(&u->cfg, (i))
/* the block whose output is level i's encoder output (skip, down-conv input, bottom): the last residual block, else enc[i] */
static block *stage_out0(unet *u, int i) { return u->cfg.enc_blocks[i] > 0 ? &u->rb[i][u->cfg.enc_blocks[i] - 1] : &u->enc[i]; }
/* (with the attention stage: its output block at the bottom level) */
static block *stage_out(unet *u, int i) { return u->tbn && i == u->cfg.nlev - 1 ? &u->tb : stage_out0(u, i); }

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
static size_t add_block(unet *u, block *b, int cin, int cout, size_t off);
static size_t add_single(unet *u, block *b, int cin, int cout, size_t off) {   /* one-conv decoder block: c1 + n1; c2 / n2 alias them */
    off = add_conv(u, &b->c1, cin, cout, 3, 1, off);
    off = add_gn(u, &b->n1, cout, off);
    b->c2 = b->c1; b->c2.cin = cout; b->n2 = b->n1; b->single = 1;
    return off;
}
static int any_single(const unet *u) { for (int i = 0; i < u->cfg.nlev - 1; i++) if (u->dec[i].single) return 1; return 0; }
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
    for (int i = 0; i < L - 1; i++) {
        const int d = dw_of(cfg, i);
        if (d <= 0 || d > w[i] || (i == 0 && d != w[0])) { fprintf(stderr, "unet: decoder width %d at level %d must be in 1..%d and the encoder width at level 0 (MX storage: multiples of 32)\n", d, i, w[i]); abort(); }
    }
    for (int i = L - 2; i >= 0; i--) off = cfg->dec_convs[i] == 1 ? add_single(u, &u->dec[i], w[i] + dw_of(cfg, i + 1), dw_of(cfg, i), off)
                                                                   : add_block(u, &u->dec[i], w[i] + dw_of(cfg, i + 1), dw_of(cfg, i), off);
    off = add_conv(u, &u->head, w[0], cfg->cout, 1, 1, off);
    if (cfg->down_norm) for (int i = 0; i < L - 1; i++) off = add_gn(u, &u->dn[i], w[i], off);
    if (cfg->scale_cond) { u->sc_w = off; off += (size_t)w[0]; }
    for (int i = 0; i < L; i++) if (cfg->enc_blocks[i] > 0) {   /* residual blocks after encoder level i (params last: older layouts unchanged) */
        if (i == 0) { fprintf(stderr, "unet: extra residual blocks need level >= 1\n"); abort(); }
        u->rb[i] = calloc((size_t)cfg->enc_blocks[i], sizeof(block)); u->nrb_any = 1;
        for (int k = 0; k < cfg->enc_blocks[i]; k++) off = add_block(u, &u->rb[i][k], w[i], w[i], off);
    }
    if (cfg->ds_levels < 0 || (cfg->ds_levels > 0 && cfg->ds_levels > L - 2)) { fprintf(stderr, "unet: deep supervision takes decoder levels 1..%d\n", L - 2); abort(); }
    for (int j = 1; j <= cfg->ds_levels; j++) off = add_conv(u, &u->ds[j], dw_of(cfg, j), cfg->cout, 1, 1, off);   /* (params last: older layouts unchanged) */
    if (cfg->attn_blocks > 0) {   /* the bottleneck transformer stage (params last) */
        const int C = w[L - 1];
        const int rep = cfg->attn_repeat > 1 ? cfg->attn_repeat : 1;
        if (C % AT_HD || cfg->attn_blocks > UNET_MAXTB || cfg->attn_pe < 0 || cfg->attn_pe > 3 || cfg->attn_reg < 0 || cfg->attn_axis < 0 || cfg->attn_axis > 1 ||
            cfg->attn_blocks * rep > UNET_MAXTA || (cfg->attn_axis == 1 && (cfg->attn_reg || cfg->attn_mem || cfg->view_P)) || cfg->attn_mem < 0 ||
            cfg->view_P < 0 || (cfg->view_P && (cfg->view_patch <= 0 || cfg->view_P % cfg->view_patch)) || cfg->attn_deform < 0 || cfg->attn_dlev < 0 ||
            (cfg->attn_deform && (cfg->attn_reg || cfg->attn_mem || cfg->view_P || cfg->attn_axis || (cfg->attn_dlev ? cfg->attn_dlev : 2) > AT_DFM_MAXL ||
                                  (cfg->attn_dlev ? cfg->attn_dlev : 2) > L)) ||
            cfg->attn_dco < 0 || cfg->attn_dco > 1 || cfg->attn_dinit < 0 || cfg->attn_dinit > 1 || (!cfg->attn_deform && (cfg->attn_dco || cfg->attn_dinit))) {
            fprintf(stderr, "unet: attention blocks need a bottleneck width that is a multiple of %d (got %d), at most %d blocks (%d applications with --attn-repeat), "
                            "--attn-pe peg|rope|none|scroll, --attn-axis full|radial (radial: no registers, memory or view tokens), view_P a multiple of view_patch, "
                            "--attn-deform: no registers / memory / view / radial, at most %d levels\n", AT_HD, C, UNET_MAXTB, UNET_MAXTA, AT_DFM_MAXL); abort(); }
        const int F = cfg->attn_mlp > 0 ? cfg->attn_mlp : (8 * C / 3 + 31) / 32 * 32;
        u->tb_nv = cfg->view_P ? (cfg->view_P / cfg->view_patch) * (cfg->view_P / cfg->view_patch) * (cfg->view_P / cfg->view_patch) : 0;
        u->nrg = cfg->attn_reg + cfg->attn_mem + u->tb_nv;
        u->tbd = (at_dims){C, C / AT_HD, F, cfg->attn_pe == 1 ? 1 : cfg->attn_pe == 3 ? 2 : 0, u->nrg, 100.f};
        if (cfg->attn_deform) {   /* deformable attention: the bottleneck and the next finer encoder levels' outputs */
            u->tbd.rope = 0; u->tbd.dK = cfg->attn_deform; u->tbd.dL = cfg->attn_dlev ? cfg->attn_dlev : 2;
            for (int l = 0; l < u->tbd.dL; l++) u->tbd.dC[l] = w[L - 1 - l];
            u->tbd.dCo = cfg->attn_dco; u->tbd.dInit = cfg->attn_dinit;
        }
        u->tba = cfg->attn_blocks * rep;
        u->tb_p0 = off;
        if (cfg->attn_pe == 0) { u->peg_w = off; off += (size_t)C * 27; u->peg_b = off; off += (size_t)C; }
        u->reg_off = off; off += (size_t)u->nrg * C;
        if (cfg->attn_mem) { u->mem_gate = off; off += (size_t)C; u->mem_woff = off; off += (size_t)4 * C; }   /* (the memory rows follow the registers) */
        if (cfg->view_P) { const int k = cfg->view_patch; u->view_w = off; off += (size_t)C * k * k * k; u->view_b = off; off += (size_t)C; }   /* (the view rows follow the memory rows) */
        for (int k = 0; k < cfg->attn_blocks; k++) {
            at_bp *bp = &u->tbp[k];
            off = at_block_layout(&u->tbd, bp, off);
            u->tbm[k][0] = (convp){C, u->tbd.dK ? C : 3 * C, 1, 1, bp->wqkv, bp->bqkv};   /* (deformable: the value projection; the offset / logit and
                                                                                         finer-level projections train with Adam) */ u->tbm[k][1] = (convp){C, C, 1, 1, bp->wo, bp->bo};
            u->tbm[k][2] = (convp){C, 2 * F, 1, 1, bp->w1, bp->b1}; u->tbm[k][3] = (convp){F, C, 1, 1, bp->w2, bp->b2};
        }
        u->tbn = cfg->attn_blocks;
        if (cfg->attn_vres && cfg->attn_deform) { fprintf(stderr, "unet: --attn-vres needs query-key attention (not --attn-deform)\n"); abort(); }
        if (cfg->attn_vres) { u->vr_off = off; off += (size_t)cfg->attn_blocks * (C / AT_HD); }
        block *so = stage_out0(u, L - 1);
        u->tb.c1 = u->tb.c2 = (convp){C, C, 3, 1, 0, 0};   /* (shapes only: no convs) */
        u->tb.n1 = u->tb.n2 = so->n2;                      /* the level's output normalisation moves after the stage */
    }
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
    if (cfg->l0_norm < 0 || cfg->l0_norm > 4) { fprintf(stderr, "unet: l0_norm %d (0 gn, 1 none, 2 ema, 3 ws, 4 slab)\n", cfg->l0_norm); abort(); }
    if (cfg->l0_norm == 4 && u->cfg.l0_slab <= 0) u->cfg.l0_slab = 64;
    if (cfg->l0_norm == 3) {   /* 1 / std(silu(z)), z ~ N(0, 1): the NF-net gain of the convs reading a SiLU output */
        double s1 = 0, s2 = 0;
        for (int i = -24000; i <= 24000; i++) { const double z = i * 5e-4, p = exp(-0.5 * z * z) * 5e-4 / sqrt(2 * M_PI), y = z / (1 + exp(-z)); s1 += p * y; s2 += p * y * y; }
        u->ws_gain = (float)(1.0 / sqrt(s2 - s1 * s1));
        u->wsp = nn_malloc(u->np * 4);
    }
    u->l0_mom = 0.01f; u->l0_warm = 100; u->l0_rc = -1;
    if (cfg->l0_norm == 2) { const size_t n = (size_t)24 * G_of(u, w[0]); u->l0run = nn_malloc(n * 4); nn_zero(u->l0run, n * 4); }
    return u;
}

static void free_acts(unet *u);
/* the fp4 kernels memoise prepared weights per conv id, weight pointer and step counter (lp_f4fwd.cuh wmemo): every weight change
   outside a training step (init, load, free: another unet's weights may land at the same address under the same step) clears it */
void unet_free(unet *u) {
    nn_wmemo_clear();
    if (!u) return;
    for (int i = 0; i < 4; i++) if (u->l0s_reg[i]) { lp_gn_slab_reg(u->l0s_reg[i], 0, 0, 0, 0); u->l0s_reg[i] = nullptr; }
    free_acts(u);
    nn_free(u->p); nn_free(u->g); nn_free(u->m); nn_free(u->v); nn_free(u->ema);
    nn_free(u->gn_scratch); nn_free(u->red_scratch); nn_free(u->wg_tmp);
    nn_free(u->wm); nn_free(u->muon_mom); nn_free(u->muon_work); nn_free(u->muon_descs); nn_free(u->stem_b);
    nn_free(u->mx_mcd); nn_free(u->mx_nsd); nn_free(u->mx_pool); nn_free(u->mx_hbf); nn_free(u->mx_R); nn_free(u->mx_part); free(u->gn_frz); free(u->gn_acc); nn_free(u->l0run); nn_free(u->l0live); nn_free(u->wsp); nn_free(u->l0s_st); nn_free(u->l0s_tmp); nn_free(u->l0st_g2); nn_free(u->l0st_up); nn_free(u->l0st_poison); nn_free(u->l0h_lg); for (int i = 0; i < UNET_MAXLEV; i++) free(u->rb[i]);
    nn_free(u->anvil_v1); nn_free(u->anvil_pool); nn_free(u->anvil_descs); nn_free(u->ds_g16); nn_free(u->ds_tmp);
    nn_free(u->wq_q); nn_free(u->wq_sc); nn_free(u->wq_qe); nn_free(u->wq_sce);
    nn_free(u->wq_r); nn_free(u->wq_rsc); nn_free(u->wq_re); nn_free(u->wq_rsce);
    free(u->tb_rayh); free(u->tb_cenh); free(u->tb_lensh); free(u->tb_adh); free(u->tb_axb); free(u->tb_axh); free(u->tb_memfh);
    free(u);
}

static void free_acts(unet *u);
const unet_cfg *unet_cfg_of(const unet *u) { return &u->cfg; }
void unet_head_offsets(const unet *u, size_t *w, size_t *b) { *w = u->head.w; *b = u->head.b; }
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
    nn_wmemo_clear();
    u->mx_R_nload = 0; u->mx_R_ok = 0;   /* Hyperball radii: the new weights' norms */
    float *h = calloc(u->np, 4);
    int L = u->cfg.nlev;
    for (int i = 0; i < L; i++) { init_conv(h, &u->enc[i].c1, &seed, 0); init_gn(h, &u->enc[i].n1); init_conv(h, &u->enc[i].c2, &seed, 0); init_gn(h, &u->enc[i].n2); }
    for (int i = 0; i < L - 1; i++) init_conv(h, &u->down[i], &seed, 0);
    if (u->cfg.down_norm) for (int i = 0; i < L - 1; i++) init_gn(h, &u->dn[i]);
    for (int i = 0; i < L - 1; i++) { init_conv(h, &u->dec[i].c1, &seed, 0); init_gn(h, &u->dec[i].n1); if (!u->dec[i].single) { init_conv(h, &u->dec[i].c2, &seed, 0); init_gn(h, &u->dec[i].n2); } }
    init_conv(h, &u->head, &seed, -2.0);
    for (int i = 0; i < L; i++) for (int k = 0; k < u->cfg.enc_blocks[i]; k++) {   /* residual blocks: identity at start (conv2 zero) */
        block *b = &u->rb[i][k];
        init_conv(h, &b->c1, &seed, 0); init_gn(h, &b->n1); init_conv(h, &b->c2, &seed, 0); init_gn(h, &b->n2);
        if (!ufsm_env_on("UFSM_RB_NONZERO")) memset(h + b->c2.w, 0, (size_t)b->c2.cout * b->c2.cin * 27 * 4);   /* (tests: random conv2) */
    }
    for (int j = 1; j <= u->cfg.ds_levels; j++) init_conv(h, &u->ds[j], &seed, -2.0);   /* deep-supervision heads, as the main head */
    if (u->tbn) {   /* attention stage: LN gains / QK-norm gains 1, QKV and MLP-in N(0, 1 / C), out projections, PEG and biases 0 (identity) */
        const int C = u->tbd.C, F = u->tbd.F;
        for (int k = 0; k < u->tbn; k++) {
            const at_bp *bp = &u->tbp[k];
            for (int c = 0; c < C; c++) h[bp->ln1g + c] = h[bp->ln2g + c] = 1.f;
            if (!u->tbd.dK) for (int c = 0; c < C; c++) h[bp->qg + c] = h[bp->kg + c] = 1.f;
            const double sd = 1.0 / sqrt((double)C);
            for (size_t i = 0; i < (size_t)(u->tbd.dK ? 1 : 3) * C * C; i++) h[bp->wqkv + i] = (float)(sd * gauss(&seed));
            if (u->tbd.dK) {   /* deformable: offset / logit weights 0 (UFSM_ATTN_NONZERO: small random), the offsets' bias the grid prior, logits 0;
                                  the finer levels' value projections N(0, 1 / C_l) */
                const size_t NOA = (size_t)at_dfm_noa(&u->tbd);
                const int nz = ufsm_env_on("UFSM_ATTN_NONZERO");
                for (size_t i = 0; i < NOA * C; i++) h[bp->woa + i] = nz ? (float)(0.02 * gauss(&seed)) : 0.f;
                for (size_t i = 0; i < NOA; i++) h[bp->boa + i] = 0.f;
                at_dfm_prior(&u->tbd, h + bp->boa);
                for (int l = 1; l < u->tbd.dL; l++) {
                    const int Cl = u->tbd.dC[l];
                    for (size_t i = 0; i < (size_t)C * Cl; i++) h[bp->wvl[l] + i] = (float)(gauss(&seed) / sqrt((double)Cl));
                }
            }
            for (size_t i = 0; i < (size_t)2 * F * C; i++) h[bp->w1 + i] = (float)(sd * gauss(&seed));
            /* out projections: 0 (identity blocks), random with UFSM_ATTN_NONZERO (tests), or UFSM_ATTN_OUT_INIT=s: s x the natural std (SSL: a
               zero stage passes the register token through unchanged, the same CLS for every sample until the projections grow) */
            const double os = ufsm_env_on("UFSM_ATTN_NONZERO") ? 1.0 : getenv("UFSM_ATTN_OUT_INIT") ? atof(getenv("UFSM_ATTN_OUT_INIT")) : 0.0;
            if (os > 0) {
                for (size_t i = 0; i < (size_t)C * C; i++) h[bp->wo + i] = (float)(os * sd * gauss(&seed));
                for (size_t i = 0; i < (size_t)C * F; i++) h[bp->w2 + i] = (float)(os * gauss(&seed) / sqrt((double)F));
            }
        }
        for (size_t i = 0; i < (size_t)u->nrg * C; i++) h[u->reg_off + i] = (float)(0.02 * gauss(&seed));
        if (u->cfg.attn_mem) for (size_t i = 0; i < (size_t)5 * C; i++) h[u->mem_gate + i] = ufsm_env_on("UFSM_ATTN_NONZERO") ? (float)(0.5 * gauss(&seed)) : 0.f;   /* gate, W_off */
        if (u->cfg.attn_vres) for (int k = 1; k < u->tbn; k++) for (int hh = 0; hh < C / AT_HD; hh++) h[u->vr_off + (size_t)k * (C / AT_HD) + hh] = 0.5f;
        if (u->cfg.attn_pe == 0 && ufsm_env_on("UFSM_ATTN_NONZERO")) for (size_t i = 0; i < (size_t)C * 27; i++) h[u->peg_w + i] = (float)(0.1 * gauss(&seed));
        if (u->cfg.view_P) {   /* the view patch embedding: unit-variance rows for unit inputs, zero bias */
            const int k3 = u->cfg.view_patch * u->cfg.view_patch * u->cfg.view_patch;
            for (size_t i = 0; i < (size_t)C * k3; i++) h[u->view_w + i] = (float)(gauss(&seed) / sqrt((double)k3));
            for (int i = 0; i < C; i++) h[u->view_b + i] = 0.f;
        }
    }
    nn_h2d(u->p, h, u->np * 4);
    nn_d2d(u->ema, u->p, u->np * 4);
    nn_zero(u->m, u->np * 4); nn_zero(u->v, u->np * 4);
    free(h);
}

/* ---- activation buffers ---- */
/* dry run (unet_train_bytes): the activation allocators only count bytes and hand out a dummy pointer */
static int g_dry = 0;
static size_t g_hoff;   /* the last build's (dry or real) pinned host offload bytes; each is copied out and back once per step */
/* UFSM_MEM_REPORT=1: unet_train_bytes prints the dry build's bytes per buffer group (g_mlab, set in build_acts) */
static const char *g_mlab = "other";
static struct { const char *lab; size_t b; int n; } g_mrep[64]; static int g_nmrep = 0;
static double g_mrep_nv;   /* UFSM_MEM_REPORT=2: every allocation of the dry build, in bytes per input voxel */
static void mrep_add(size_t b) {
    if (g_mrep_nv > 0 && b >= 0.05 * g_mrep_nv) fprintf(stderr, "    alloc %-40s %7.2f B/voxel\n", g_mlab, b / g_mrep_nv);
    int i = 0; while (i < g_nmrep && strcmp(g_mrep[i].lab, g_mlab)) i++;
    if (i == g_nmrep) { if (g_nmrep == 64) return; g_mrep[g_nmrep].lab = g_mlab; g_mrep[g_nmrep].b = 0; g_mrep[g_nmrep].n = 0; g_nmrep++; }
    g_mrep[i].b += b; g_mrep[i].n++;
}
static float *dmalloc(size_t b) { if (g_dry) mrep_add(b); return g_dry ? (float *)(uintptr_t)4096 : nn_malloc(b); }
static void dstorage(const void *p, size_t b, int dt) { if (!g_dry) nn_set_storage(p, b, dt); }
static float *dalloc(unet *u, size_t n) { u->act_bytes += n * 4; return dmalloc(n * 4); }
/* the attention stage keeps every block's forward tensors of a training forward for its backward (no recompute: + at_ws_fwd_bytes per
   block); env UFSM_ATTN_KEEP, unet_set_attn_keep (before the build) */
static int g_attn_keep = -1;
void unet_set_attn_keep(int on) { g_attn_keep = on; }
static int attn_keep(void) { if (g_attn_keep < 0) g_attn_keep = ufsm_env_on("UFSM_ATTN_KEEP"); return g_attn_keep; }
/* radial attention: rays per axis-normal plane / the grid's largest dim (env UFSM_ATTN_RAY_MULT, default 2: about 0.7 voxel apart where the
   axis lies outside the window) */
static float attn_ray_mult(void) { static float m = 0; if (m <= 0) { const char *e = getenv("UFSM_ATTN_RAY_MULT"); m = e && atof(e) > 0 ? (float)atof(e) : 2.f; } return m; }
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
/* the smallest up-part chunk the MX gradient kernels take for an up part of c channels: MX-fp4 whole blocks (an 8- or 16-channel
   up part at once, else 32), MX-fp8 16-channel halves of a block (or the whole up part when it is narrower) */
static int mx_chunk_min(int c) { return grad_dt() == 4 ? (c <= 16 ? c : 32) : (c < 16 ? c : 16); }
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
static int g_infer_alias = -1;
static int infer_alias(void) { if (g_infer_alias < 0) g_infer_alias = getenv("UFSM_INFER_ALIAS") ? ufsm_env_on("UFSM_INFER_ALIAS") : 1; return g_infer_alias; }
void unet_set_infer_alias(int on) { g_infer_alias = on; }
static int g_infer_off = -1;
static int infer_offload(void) { if (g_infer_off < 0) g_infer_off = getenv("UFSM_INFER_OFFLOAD") ? ufsm_env_on("UFSM_INFER_OFFLOAD") : 1; return g_infer_off; }
void unet_set_infer_offload(int on) { g_infer_off = on; }
static int g_infer_slab = -1;
static int infer_slab(void) { if (g_infer_slab < 0) g_infer_slab = getenv("UFSM_INFER_SLAB") ? atoi(getenv("UFSM_INFER_SLAB")) : 64; return g_infer_slab; }   /* planes per slab (default 64), 0 off */
void unet_set_infer_slab(int planes) { g_infer_slab = planes; }
int unet_ct_stem(const unet *u) { return u->built && !u->train && u->ct_stem; }
void unet_set_ct_input(unet *u, int on) { if (u->ct_mode != on && u->built && !u->train) free_acts(u); u->ct_mode = on; }
static int share_enc_a1(void) { if (g_share_enc_a1 < 0) g_share_enc_a1 = ufsm_env_on("UFSM_RC_ENC_A1"); return g_share_enc_a1 && recompute() == 1 && nn_get_tf32(); }
void unet_set_share_enc_a1(int on) { g_share_enc_a1 = on; }
/* shared encoder a1 without the re-run: each encoder's a1 is copied to pinned host memory once its block's forward is done
   and copied back after the same-level decoder's backward (the backward of the coarser levels hides both copies; desk
   slab ~1.7 GB each way per step at ~28 GB/s). Default on; env UFSM_A1_OFFLOAD=0 re-runs the encoder conv1 instead */
static int g_a1_offload = -1;
static int a1_offload(void) { if (g_a1_offload < 0) g_a1_offload = getenv("UFSM_A1_OFFLOAD") ? ufsm_env_on("UFSM_A1_OFFLOAD") : 1; return g_a1_offload && share_enc_a1(); }
void unet_set_a1_offload(int on) { g_a1_offload = on; }
int unet_a1_offload(void) { return a1_offload(); }
/* training: the MX network input (read by the stem conv's forward and weight gradient only) shares dec[0].a2's buffer, which is
   free before the decoder's forward and after its backward; pinned host copy in between (desk slab 0.68 GB each way).
   -9 B per level-0 voxel at fp8 input. Default on; env UFSM_XIN_OFFLOAD=0 keeps a separate buffer */
static int g_xin_offload = -1;
static int xin_offload(void) { if (g_xin_offload < 0) g_xin_offload = getenv("UFSM_XIN_OFFLOAD") ? ufsm_env_on("UFSM_XIN_OFFLOAD") : 1; return g_xin_offload; }
void unet_set_xin_offload(int on) { g_xin_offload = on; }
/* lean 2 with MX-fp4 activations and gradients: each decoder level's gradient buffer A is its dec[i].a2's buffer (dead after the
   first op of dec[i]'s backward), the separate A only serves the bottom level (desk: 17 -> ~0 B per level-0 voxel). For that the skip enc[0].a2 goes
   to pinned host memory after its forward (it is idle from dec[0]'s conv1 to that conv's weight gradient) and its buffer takes
   the 16-bit logit gradient; it comes back after the head's backward. With the input offload the input returns into it once
   enc[0]'s backward has read a2. Default on; env UFSM_A0_SHARE=0 off */
static int g_a0_share = -1;
static int a0_share_on(void) { if (g_a0_share < 0) g_a0_share = getenv("UFSM_A0_SHARE") ? ufsm_env_on("UFSM_A0_SHARE") : 1; return g_a0_share; }
void unet_set_a0_share(int on) { g_a0_share = on; }
/* lean 2 with the level-0 A share, recompute 1: the coarse levels' gradient buffers gout[1..] (the up-part gradients, the
   skip gradients) are only live from the end of dec[0]'s backward to enc[1]'s; they go to the end of dec[0].a2's buffer, dead
   in that span (it is A again from enc[0]'s conv2 backward-data on). gout[1] is also written in dec[0]'s backward itself (the
   coarse silu(gn(a2)) of the sliced weight gradient, the up-part gradient): there it lives at the end of dec[0].a1's buffer,
   dead after dec[0]'s GN1 backward, and is copied over afterwards (desk slab 0.32 GB of device copy). Desk mode -5.2 B per
   level-0 voxel. Default on; env UFSM_COARSE_GRAD_SHARE=0 off */
static int g_gco = -1;
static int gco_on(void) { if (g_gco < 0) g_gco = getenv("UFSM_COARSE_GRAD_SHARE") ? ufsm_env_on("UFSM_COARSE_GRAD_SHARE") : 1; return g_gco; }
void unet_set_coarse_grad_share(int on) { g_gco = on; }
/* with the coarse gradient share: the down-conv outputs downo[i] (read by enc[i + 1]'s conv1 in the forward and its backward, idle
   in between) live below the coarse gout in dec[0].a2's buffer: free in the forward from the input's copy out to dec[0]'s conv2,
   and in the backward from dec[0]'s to enc[0]'s; a pinned host copy from the forward's read to the backward (desk slab 0.21 GB
   each way). Desk mode -2.8 B per level-0 voxel. Needs the input below them. Default on; env UFSM_DOWN_OFFLOAD=0 off */
static int g_dno = -1;
static int dno_on(void) { if (g_dno < 0) g_dno = getenv("UFSM_DOWN_OFFLOAD") ? ufsm_env_on("UFSM_DOWN_OFFLOAD") : 1; return g_dno; }
void unet_set_down_offload(int on) { g_dno = on; }
/* with the down-conv output offload: the coarse skips stage_out(i).a2 (1 <= i <= L-2: enc[i].a2, or with residual blocks the last
   block's a2; written by its conv2, read by the down conv and dec[i]'s conv1, idle until dec[i]'s backward) at the start of dec[0].a2's
   buffer (after the input's copy out: enc[1]'s conv2 waits for it), with a pinned host copy from dec[i]'s conv1 to dec[0]'s backward
   (desk slab 0.39 GB each way).
   Desk mode -5.2 B per level-0 voxel. Default on; env UFSM_SKIP_OFFLOAD=0 off */
static int g_sko = -1;
static int sko_on(void) { if (g_sko < 0) g_sko = getenv("UFSM_SKIP_OFFLOAD") ? ufsm_env_on("UFSM_SKIP_OFFLOAD") : 1; return g_sko; }
void unet_set_skip_offload(int on) { g_sko = on; }
#define SKO_DEFER ((float *)(uintptr_t)1)   /* build: an encoder a2 placed later */
int unet_share_enc_a1(void) { return share_enc_a1(); }
/* the planner's --mem fast: every training host offload (encoder a1, input, level-0 skip / A share, down outputs, coarse skips) off
   (0), or back to its default / env switch (1) */
void unet_set_host_offloads(int on) { const int v = on ? -1 : 0; g_a1_offload = v; g_xin_offload = v; g_a0_share = v; g_dno = v; g_sko = v; }
size_t unet_last_offload_bytes(void) { return g_hoff; }
/* three level-0 buffers instead of four (unet.l3b): -9 B per level-0 voxel at 16 channels (36 -> 27 B at the backward's peak), for
   three exposed host copies (the skip into X2 in dec[0]'s conv1 weight gradient, the encoder's a1 before enc[0]'s backward, the input
   before the stem's weight gradient). Opt-in (env UFSM_L0_3BUF=1) until its paired checks pass */
static int g_l0_3buf = -1;
static int l0_3buf_on(void) { if (g_l0_3buf < 0) g_l0_3buf = getenv("UFSM_L0_3BUF") ? ufsm_env_on("UFSM_L0_3BUF") : 0; return g_l0_3buf; }
void unet_set_l0_3buf(int on) { g_l0_3buf = on; }   /* -1: the env / default */
static int l0ring_mode(void);   /* (stage S4: the slab rings requested; a build input) */
#define UMODE() (nn_get_tf32() * 4 + ABF * 2 + GBF + 8 * act_mx8() + 16 * grad_mx8() + 32 * recompute() + 128 * act_mx4() + 256 * chunk_up() + 1024 * lean() + 4096 * input_mx() + 8192 * (unet_input_prec() == 8) + 16384 * unet_wide_up_grad() + 32768 * keep_coarse_a1() + 65536 * share_enc_a1() + 131072 * grad_mx4() + 262144 * up_wg_chunk() + 524288 * a1_offload() + 1048576 * xin_offload() + 2097152 * a0_share_on() + 4194304 * gco_on() + 8388608 * dno_on() + 16777216 * sko_on() + 33554432 * infer_alias() + 67108864 * (infer_slab() > 0) + 134217728 * infer_offload() + 268435456 * l0_3buf_on() + 536870912 * l3_sched_on() + 1073741824 * l0ring_mode())
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
/* coarse-offload layout (unet.p3): a buffer placed inside X or Y is no allocation of its own */
static int interior(const unet *u, const void *p) {
    const char *c = (const char *)p;
    return u->p3 && p && ((u->Xp && c > (const char *)u->Xp && c < (const char *)u->Xp + u->x_bytes) || (u->Yp && c > (const char *)u->Yp && c < (const char *)u->Yp + u->y_bytes));
}
static void free_acts(unet *u) {
    float *pp[UNET_MAXLEV * 30 + 8]; int np = 0;
    if (u->l3b) {   /* the level-0 tensors and the coarse homes back where they were allocated / placed first */
        if (u->a2_out || u->a1_out[0] || u->xin_out) nn_sync();
        u->enc[0].a1 = u->l3x[1]; u->enc[0].a2 = u->l3x[2];
        for (int i = 0; i < u->cfg.nlev && !g_dry; i++) {
            float *q[4] = {u->skf[i], u->skb[i], u->dnf[i], u->dnb[i]};
            for (int j = 0; j < 4; j++) if (q[j]) nn_storage_forget(q[j]);
        }
        for (int i = 0; i < UNET_MAXLEV; i++) u->skf[i] = u->skb[i] = u->dnf[i] = u->dnb[i] = nullptr;
    }
    for (int i = 0; i < u->cfg.nlev; i++) {
        block *bs[2] = {&u->enc[i], &u->dec[i]};
        for (int k = 0; k < 2; k++) {
            block *b = bs[k];
            if (k == 0 && u->sko && i > 0 && i < u->cfg.nlev - 1 && !u->cfg.enc_blocks[i]) { if (!g_dry && b->a2) nn_storage_forget(b->a2); b->a2 = nullptr; }   /* (inside dec[0].a2's) */
            if (k == 0 && u->h_sk[i]) { if (u->sk_out) nn_sync(); nn_host_free(u->h_sk[i]); u->h_sk[i] = nullptr; }
            float *q[7] = {b->a1, b->a2, b->s2, b->m1, b->r1, b->m2, b->r2};
            for (int j = 0; j < 7; j++) pp[np++] = q[j];
            b->a1 = b->a2 = b->s2 = b->m1 = b->r1 = b->m2 = b->r2 = nullptr;
        }
        const int inner = u->gco && i > 0;   /* gout / gskip / gB inside dec[0]'s buffers */
        if (u->dno && i < u->cfg.nlev - 1) { if (!g_dry && u->downo[i]) nn_storage_forget(u->downo[i]); u->downo[i] = nullptr; }   /* (inside dec[0].a2's) */
        if (u->h_dn[i]) { if (u->dn_out) nn_sync(); nn_host_free(u->h_dn[i]); u->h_dn[i] = nullptr; }
        float *q[11] = {inner ? nullptr : u->gskip[i], u->gA[i], inner ? nullptr : u->gB[i], u->downo[i], u->cat[i], inner ? nullptr : u->gout[i], u->t1[i], u->t2[i], u->dm[i], u->dr[i], u->downs[i]};
        for (int j = 0; j < 11; j++) pp[np++] = q[j];
        if (inner && !g_dry) for (int k = 0; k < 2; k++) { float *r = k ? u->gout_x1 : u->gout[i]; if (r && (k == 0 || i == 1)) nn_storage_forget(r); }
        u->dm[i] = u->dr[i] = u->downs[i] = nullptr;
        u->downo[i] = u->cat[i] = u->gskip[i] = u->gout[i] = u->gA[i] = u->gB[i] = u->t1[i] = u->t2[i] = nullptr;
    }
    if (u->l0r_hsk) { nn_sync(); nn_host_free(u->l0r_hsk); u->l0r_hsk = nullptr; u->l0r_hskn = 0; }
    if (u->l0ring) {   /* (stage S4: built here, unlike the lazy stream buffers of S3) */
        float *q[3] = {u->l0st_g2, u->l0st_up, u->l0r_upsn ? u->l0r_ups : nullptr};   /* (l0r_upsn 0: gout[1]'s, freed with it) */
        free_once(q, 3);
        u->l0st_g2 = u->l0st_up = u->l0r_ups = nullptr; u->l0st_g2n = u->l0st_upn = u->l0r_upsn = 0;
        u->l0ring = 0;
    }
    for (int i = 0; i < u->cfg.nlev; i++) {   /* residual blocks: own buffers */
        float *q[8] = {u->gres[i]}; int nq = 1;
        free_once(q, nq); u->gres[i] = nullptr; u->gres_cap[i] = 0;
        for (int k = 0; k < u->cfg.enc_blocks[i]; k++) {
            block *b = &u->rb[i][k];
            if (k == u->cfg.enc_blocks[i] - 1 && u->sko && i > 0 && i < u->cfg.nlev - 1) { if (!g_dry && b->a2) nn_storage_forget(b->a2); b->a2 = nullptr; }   /* (the stage output: inside dec[0].a2's) */
            float *r[7] = {b->a1, b->a2, b->s2, b->m1, b->r1, b->m2, b->r2};
            free_once(r, 7); b->a1 = b->a2 = b->s2 = b->m1 = b->r1 = b->m2 = b->r2 = nullptr;
        }
    }
    if (u->tbn) {   /* attention stage */
        block *b = &u->tb;
        float *r[18] = {b->a1, b->a2, b->s2, b->m1, b->r1, b->m2, b->r2, u->tb_cs, u->tb_tok, u->tb_wsb, u->tb_zero, u->tb_g, u->tb_gx, u->tb_dt, u->tb_dt2, u->tb_keep, u->tb_rb, u->tb_vr};
        free_once(r, 18);
        b->a1 = b->a2 = b->s2 = b->m1 = b->r1 = b->m2 = b->r2 = nullptr;
        u->tb_cs = u->tb_tok = u->tb_wsb = u->tb_zero = u->tb_g = u->tb_gx = u->tb_dt = u->tb_dt2 = u->tb_keep = u->tb_rb = u->tb_vr = nullptr;
    }
    pp[np++] = u->logits; pp[np++] = u->conv_scratch; pp[np++] = u->rc_extra;
    if (u->rc_extra && !g_dry) nn_storage_forget(u->rc_extra);
    u->rc_extra = nullptr; u->rc_extra_bytes = 0; u->gB_bytes = 0; pp[np++] = u->xin; pp[np++] = u->glb;
    for (int j = 0; j < np; j++) if (interior(u, pp[j])) pp[j] = nullptr;
    free_once(pp, np);
    u->logits = nullptr; u->conv_scratch = nullptr; u->xin = nullptr; u->glb = nullptr; u->conv_scratch_n = 0;
    for (int i = 0; i < UNET_MAXLEV; i++) {
        if (u->h_a1[i]) { if (u->a1_out[i]) nn_sync(); nn_host_free(u->h_a1[i]); }   /* (a copy may still be in flight) */
        u->h_a1[i] = nullptr; u->a1_out[i] = 0;
    }
    if (u->h_xin) { if (u->xin_out) nn_sync(); nn_host_free(u->h_xin); }
    u->h_xin = nullptr; u->xin_shared = u->xin_out = u->a2_stale = 0; u->xin_bytes = 0;
    if (u->h_a2) { if (u->a2_out) nn_sync(); nn_host_free(u->h_a2); }
    u->h_a2 = nullptr; u->a0_share = u->a2_out = 0;
    u->xin_inf = u->lg_t1 = u->lg_x = 0; u->t1_bytes = 0; u->lg_p = nullptr; u->lg_bytes = 0;
    {
        float *q[6] = {u->slab_tmp, u->slab_carry, u->slab_xin, u->slab_tmp1, u->slab_carry1, u->slab_tmpd};
        for (int j = 0; j < 6; j++) if (interior(u, q[j])) { if (!g_dry && j == 2) nn_storage_forget(q[j]); q[j] = nullptr; }
        free_once(q, 6); u->slab_tmp = u->slab_carry = u->slab_xin = u->slab_tmp1 = u->slab_carry1 = u->slab_tmpd = nullptr; u->slab = u->slab1 = u->slab_s = u->ct_stem = 0;
    }
    if (u->h_r) { if (u->r_out || u->r_in) nn_sync(); nn_host_free(u->h_r); }
    u->h_r = nullptr; u->p3 = u->r_out = u->r_in = 0; u->R = u->Xp = u->Yp = nullptr; u->r_bytes = u->x_bytes = u->y_bytes = 0; u->zr = 0;
    u->gco = 0; u->gout_x1 = u->gout_home1 = nullptr; u->dno = u->dn_out = 0; u->sko = u->sk_out = 0;
    u->l3b = u->l3s = u->l3_skip_in = 0; u->l3_lo = 0; for (int k = 0; k < 4; k++) { u->l3x[k] = nullptr; u->l3xb[k] = 0; }
    u->built = 0; u->act_bytes = 0; u->grad_bytes = 0;
    u->last_a1_live = 0;
}

/* cfg.l0_norm slab, kernel path (UFSM_L0_SLAB_KERN=1, MX tensor cores; level-0 slab streaming plan stage S1): the level-0 blocks'
   m1 / r1 / m2 / r2 hold per-(sample, z-slab) statistics [n][slab][G], registered with lp_gn_slab_reg so every reader's gnp_t carries
   the slab geometry and applies the slab of each plane (no normalised copy, no extra rounding: D's prototype is the reference);
   the producer convs' statistics come from a pass over the pseudo-sample shape ((sample, slab) of one MX channel block is a sample
   of the same layout), the GroupNorm backwards run over that shape on the raw tensor with the true statistics. Uniform slabs. */
static int g_l0s_kern = -1;   /* unet_set_l0_slab_kern; -1: env UFSM_L0_SLAB_KERN */
void unet_set_l0_slab_kern(int on) { g_l0s_kern = on; }
static int l0s_kern(const unet *u) {
    static int k = -1;
    if (k < 0) { const char *e = getenv("UFSM_L0_SLAB_KERN"); k = e ? atoi(e) : 0; }
    return (g_l0s_kern >= 0 ? g_l0s_kern : k) && u->cfg.l0_norm == 4 && nn_get_tf32();
}
static int l0s_nslab(const unet *u, int d) { const int S = u->cfg.l0_slab; return (d + S - 1) / S; }
/* ---- stage S4: level-0 slab rings (UFSM_L0_RING=1 with UFSM_L0_STREAM=2; plan /home/forrest/.claude/plans/level0-slab-streaming.md).
   The level-0 tensors of the slab stream become VMM rings (nn_ring_alloc): whole-tensor virtual addresses (S2's windows, S3's schedule
   unchanged), slab s backed by physical chunk s mod R. R per tensor is the schedule's live span (S3's DESIGN.md table, verified by the
   bit-for-bit test: a ring too short changes the bits). Indices: 0 enc[0].a1, 1 the skip enc[0].a2, 2 dec[0].a1, 3 dec[0].a2,
   4 d/d a2 (the stream's B2), 5 level 0's A, 6 the up-part gradient (level 0's B), 7 dec[0]'s up transient, 8 gout[0], 9 the input,
   10 the skip gradient in the host mode (UFSM_L0_RING_SKIP=1, the default; 0: whole on the device) */
static int l0ring_skip_host(void) { static int v = -1; if (v < 0) { const char *e = getenv("UFSM_L0_RING_SKIP"); v = e ? atoi(e) : 1; } return v; }
static int g_l0ring = -1;
void unet_set_l0_ring(int on) { g_l0ring = on; }
static int l0ring_req(void) { if (g_l0ring < 0) { const char *e = getenv("UFSM_L0_RING"); g_l0ring = e ? atoi(e) : 0; } return g_l0ring; }
static int l0ring_R(int k) {
    static int R[10] = {0}, R10 = 1; static int set = 0;
    if (!set) {
        const int def[11] = {3, 5, 3, 1, 3, 3, 3, 3, 1, 3, 1};   /* (the shortest that keep the bits: test_l0_stream with each one shorter fails) */
        for (int i = 0; i < 10; i++) R[i] = def[i];
        R10 = def[10];
        const char *e = getenv("UFSM_L0_RING_R");   /* "a,b,c,...": overrides in index order (the test's too-short controls) */
        for (int i = 0; e && *e && i < 11; i++) { if (i < 10) R[i] = atoi(e); else R10 = atoi(e); const char *c = strchr(e, ','); e = c ? c + 1 : nullptr; }
        set = 1;
    }
    return k >= 0 && k < 10 ? R[k] : k == 10 ? R10 : 3;
}
static int l0st_req(void);
static int l0ring_mode(void) { return l0ring_req() && l0st_req() >= 2; }
/* the rings fit this build: stream 2 requested, slab kernel path, uniform slabs whose MX data and scale planes lie on the mapping
   granularity (16-channel level 0: S h w a multiple of 2 MB, i.e. P a multiple of 256 at 32-plane slabs) */
static int l0ring_want(const unet *u, int train, shape5 xs) {
    if (!train || !l0ring_req() || l0st_req() < 2 || !l0s_kern(u) || xs.n != 1 || u->l3b) return 0;
    const size_t G = g_dry ? (size_t)2 << 20 : nn_ring_gran(), hw = (size_t)xs.h * xs.w, S = (size_t)u->cfg.l0_slab;
    const char *why = !G ? "CUDA virtual memory management" : xs.d % u->cfg.l0_slab || l0s_nslab(u, xs.d) < 2 ? "several uniform slabs"
        : (S * hw) % G ? "slabs of l0_slab h w bytes on the 2 MB mapping granularity (P a multiple of 256 at 32-plane slabs)" : nullptr;
    if (why) { fprintf(stderr, "unet: UFSM_L0_RING needs %s\n", why); abort(); }
    return 1;
}
/* a level-0 tensor of shape s (one sample, depth a multiple of the slab), MX storage dt, as a ring of R slabs */
static float *dalloc_ring(unet *u, shape5 s, int dt, int R, int grad) {
    const int bw = s.c <= 8 ? 8 : s.c <= 16 ? 16 : 32, nb = (s.c + bw - 1) / bw, rb = dt == 4 ? bw / 2 : bw;
    const size_t hw = (size_t)s.h * s.w, D = (size_t)s.d, S = (size_t)u->cfg.l0_slab, ns = D / S, rr = (size_t)R < ns ? (size_t)R : ns;
    size_t reg[16], sl[16], phys = 0; int n = 0;
    for (int k = 0; k < nb; k++) { reg[n] = D * hw * rb; sl[n] = S * hw * rb; n++; }
    for (int k = 0; k < nb; k++) { reg[n] = D * hw; sl[n] = S * hw; n++; }
    for (int k = 0; k < n; k++) phys += rr * sl[k];
    const size_t tot = nn_mx_bytes(s, dt);
    u->act_bytes += phys; if (grad) u->grad_bytes += phys;
    if (g_dry) { mrep_add(phys); return (float *)(uintptr_t)4096; }
    size_t ph = 0;
    float *p = (float *)nn_ring_alloc(n, reg, sl, R, &ph);
    if (!p) { fprintf(stderr, "unet: a level-0 slab ring of %.2f GB (%zu of %zu slabs; shape %d %d %d %d %d, dt %d) failed (VMM, granularity or memory)\n", phys / 1e9, rr, ns, s.n, s.c, s.d, s.h, s.w, dt); abort(); }
    dstorage(p, tot, dt);
    return p;
}
/* a1 / a2: nullptr = allocate, else a shared buffer (inference or recompute 2) */
static void build_block_acts(unet *u, block *b, shape5 xs, int train, int keep_s2, float *a1, float *a2, float *s2) {
    b->xs = xs;
    b->ys = xs; b->ys.c = b->c1.cout;
    int G = G_of(u, b->c1.cout);
    if (l0s_kern(u) && (b == &u->enc[0] || (b == &u->dec[0] && u->cfg.nlev > 1))) G *= l0s_nslab(u, xs.d);   /* (slab kernel path: [n][slab][G] statistics) */
    b->keep_a1 = train && recompute_a1() && !a1;
    b->a1_shared = b->a1_lent = 0;
    b->a1 = a1 ? a1 : dalloc_act_s(u, b->ys); b->a2 = a2 ? a2 : dalloc_act_s(u, b->ys);
    b->s2 = recompute() && (!keep_s2 || act_mx8()) ? nullptr : s2 ? s2 : dalloc_act_s(u, b->ys);   /* MX: the up staging normalises a2 itself */
    b->m1 = dalloc(u, (size_t)xs.n * G); b->r1 = dalloc(u, (size_t)xs.n * G); b->m2 = dalloc(u, (size_t)xs.n * G); b->r2 = dalloc(u, (size_t)xs.n * G);
}
static size_t act_bytes_of(shape5 s) { return act_mx8() ? nn_mx_bytes(s, act_dt()) : shape_numel(s) * (ABF ? 2 : 4); }
/* MX storage: channels per block of a c-channel tensor (nn_mx4_bytes), and the bytes per voxel of one block's data row */
static int mx_bw_of(int c) { return c <= 8 ? 8 : c <= 16 ? 16 : 32; }
static int mx_rb_of(int c) { return act_mx4() ? mx_bw_of(c) / 2 : mx_bw_of(c); }
/* in-place inference slabs at a level of c channels: whole MX blocks of 16 or a multiple of 32 channels (8: the z-range needs the
   fused output statistics, cout >= 16) */
static int slab_w_ok(int c) { return c % 32 == 0 || c == 16; }

/* coarse offload: enc[1]'s conv1 reads down0's output (w[0] channels) in place in Y (w[1]): placed at the data rows of Y's last
   w[0] / 32 blocks, so plane z of each of its blocks (data rows, then scales) sits where Y's plane z of a block goes */
static size_t p3_down_off(const unet *u) {   /* (build_acts: one block width for w[0] and w[1]) */
    const int bw = mx_bw_of(u->cfg.widths[1]), nb0 = (u->cfg.widths[0] + bw - 1) / bw, nb1 = (u->cfg.widths[1] + bw - 1) / bw, rb = mx_rb_of(u->cfg.widths[1]);
    return (size_t)(nb1 - nb0) * shape_spatial(u->ls[1]) * rb;
}
/* dec[0]'s conv1 under the coarse offload: planes of the slab at z0 that fit its temp in Y's consumed coarse planes (block 0's
   data rows below (z0 >> 1) - 3; the up staging reads from (z0 >> 1) - 2 on), from Y + 256, a multiple of 4 */
static int dc1_fity(const unet *u, int z0) {
    const int cz = (z0 >> 1) - 3, rb = mx_rb_of(u->cfg.widths[1]);
    if (cz <= 0) return 0;
    const long long avail = (long long)cz * u->ls[1].h * u->ls[1].w * rb - 256;
    shape5 p = u->ls[0]; p.c = u->cfg.widths[0]; p.d = 1;
    const long long n = avail > 0 ? avail / (long long)act_bytes_of(p) : 0;
    return (int)(n / 4 * 4);
}
/* the slab at z0 (0: none fits); *B: 0 phase A (temp in R, the slab's input planes below zr), 1 phase B (temp in Y). The switch:
   once Y holds min(s, 16) planes, or when phase A cannot go on */
static int dc1_next(const unet *u, int z0, int *B) {
    const int D = u->ls[0].d, s = u->slab_s, fy = dc1_fity(u, z0);
    if (!*B) {
        int sa = s < D - z0 ? s : D - z0;
        if (z0 + sa >= u->zr) sa = (u->zr - 1 - z0) / 4 * 4;
        if (sa >= 4 && fy < (s < 16 ? s : 16)) return sa;
        *B = 1;
    }
    if (D - z0 <= (s < fy ? s : fy)) return D - z0;
    const int sk = s < fy ? s : fy;
    return sk >= 4 ? sk : 0;
}
static int fakeq(void);
static int fakeq_grad(void);
/* the three level-0 buffers (unet.l3b) apply: a training build of the production chain (MX-fp4 activations and gradients, lean 2 chunked,
   recompute 1, the shared encoder a1 offloaded, the level-0 A share, the input / coarse gradient / down-output / skip offloads) with an
   8- or 16-channel level 0 the head and the decoder above it fit (UPW(1) <= w[0]), whole-window logits of <= 4 channels from the recomputing
   GN backward's kernels, no split / single-conv decoders / level-0 residual blocks / deep supervision (residual blocks at levels >= 1: the
   coarse skip homes hold each level's stage output); build_acts checks the rest (host buffers) */
static int l3_want(const unet *u, int train) {
    const int L = u->cfg.nlev, *w = u->cfg.widths;
    if (!train || !l0_3buf_on() || u->l3_no || L < 3) return 0;
    if (!(nn_get_tf32() && ABF && GBF && act_mx4() && grad_mx4() && grad_dt() == act_dt() && lean() >= 2 && chunk_up() && recompute() == 1 && !recompute_a1())) return 0;
    if (!(share_enc_a1() && a1_offload() && a0_share_on() && gco_on() && dno_on() && sko_on() && xin_offload() && input_mx())) return 0;
    if ((w[0] != 8 && w[0] != 16) || UPW(1) > w[0] || UPW(0) != w[0] || unet_wide_up_grad()) return 0;
    if (any_single(u) || u->cfg.enc_blocks[0] || u->head_ext || sp_on(u) || u->cfg.ds_levels || fakeq() || fakeq_grad() || (u->cfg.l0_norm == 4 && !l0s_kern(u))) return 0;   /* (slab prototype: no head-recomputing GN backward;
                                                                                                      kernel path: one slab, else l0_begin stops) */
    return nn_gn_silu_bwd_head_ok(w[0], u->cfg.cout);
}
static int l3_sched_on(void);
static void build_acts(unet *u, shape5 xs, int train) {
    free_acts(u);
    g_hoff = 0;
    int L = u->cfg.nlev;
    const int *w = u->cfg.widths;
    u->l3b = l3_want(u, train);
    u->l3s = u->l3b && l3_sched_on();
    u->l0ring = l0ring_want(u, train, xs);   /* (stage S4: the level-0 tensors of the slab stream in VMM rings) */
    u->l0hb = u->l0ring && u->l0h_fwd;        /* (with the stream head: dec[0].a2 and gout[0] in rings too, no whole logits) */
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
    /* in-place slab convs at level 0 (see unet.slab): X holds every level-0 tensor; T1 only serves the coarser levels. GroupNorm level 0
       only: the slab convs reduce whole-window GroupNorm statistics themselves, the cfg.l0_norm modes (none / ema / ws / slab) are not
       wired into them yet and run the whole-window inference path */
    const int slab = share && reuse_skip && infer_alias() && act_mx8() && infer_slab() > 0 && u->cfg.enc_blocks[0] == 0 && slab_w_ok(w[0]) && !any_single(u)
                     && u->cfg.l0_norm == 0;
    const int slab1 = slab && L > 2 && slab_w_ok(w[1]) && u->cfg.enc_blocks[1] == 0 && (!getenv("UFSM_INFER_SLAB_L1") || ufsm_env_on("UFSM_INFER_SLAB_L1"));
    float *X = nullptr, *Y = nullptr;
    u->slab = slab; u->slab1 = slab1; u->slab_s = slab ? (infer_slab() + 3) / 4 * 4 : 0;
    float *SK[UNET_MAXLEV] = {0};   /* coarse offload: the skips below level 1 inside R */
    if (share) {
        shape5 m1 = u->ls[0], m2, mc = u->ls[0]; m1.c = 0; mc.c = 0;
        for (int i = slab1 ? 2 : slab ? 1 : 0; i < L; i++) { shape5 y = u->ls[i]; y.c = w[i]; if (act_bytes_of(y) > act_bytes_of(m1)) m1 = y; }
        m2 = m1;
        if (reuse_skip) {
            /* Full-resolution decoder c2 is the only large T2 user. Its encoder skip is dead after
               decoder c1, so c2 can write there. Earlier decoder outputs need only a smaller T2. */
            m2.c = 0;
            for (int i = slab1 ? 2 : 1; i < L - 1; i++) { shape5 y = u->ls[i]; y.c = w[i]; if (act_bytes_of(y) > act_bytes_of(m2)) m2 = y; }
        }
        const int off_ok = slab1 && infer_offload() && !u->nrb_any && w[1] >= w[0] && mx_bw_of(w[0]) == mx_bw_of(w[1]);   /* (enc[1]'s conv1 in place over the down output in Y: its blocks nest in Y's) */
        for (int i = off_ok ? 1 : 0; i < L - 1; i++) { shape5 d = u->ls[i + 1]; d.c = w[i]; if (act_bytes_of(d) > act_bytes_of(m2)) m2 = d; }
        for (int i = 0; i < L - 1; i++) { shape5 c = u->ls[i]; c.c = UPW(i + 1); if (act_bytes_of(c) > act_bytes_of(mc)) mc = c; }
        /* the coarse offload (unet.p3): R's contents, the slab size that fits Y's level-0 temps, the dec[0] conv1 schedule */
        shape5 y0 = u->ls[0]; y0.c = w[0];
        shape5 y1 = u->ls[1]; y1.c = w[1];
        const int ip = unet_input_prec(), ct_ok = u->ct_mode && input_mx() && (ip == 8 || ip == 4) && xs.h == xs.w && xs.n == 1 && u->cfg.cin == 4;
        int p3 = 0;
        if (off_ok) {
            const size_t xb = act_bytes_of(y0), yb = act_bytes_of(y1), A = 256;
            const int bw0 = mx_bw_of(w[0]), nb0 = (w[0] + bw0 - 1) / bw0, rb0 = mx_rb_of(w[0]);
            const size_t S0 = shape_spatial(y0), blk_data = S0 * rb0;
            int s = u->slab_s;
            for (; s >= 4; s -= 4) {
                shape5 t0 = y0; t0.d = s < y0.d ? s : y0.d;
                shape5 xi = xs; xi.d = t0.d + 2; if (xi.d > xs.d) xi.d = xs.d;
                if (A + ((act_bytes_of(t0) + 255) & ~(size_t)255) + (ct_ok ? nn_mx_bytes(xi, ip) : 0) <= yb) break;
            }
            if (s >= 4) {
                shape5 t0 = y0; t0.d = s < y0.d ? s : y0.d;
                shape5 t1 = y1; t1.d = s < y1.d ? s : y1.d;
                shape5 c1 = y1; c1.d = 1;
                size_t rs = A;
                rs += (act_bytes_of(m1) + 255) & ~(size_t)255; rs += (act_bytes_of(m2) + 255) & ~(size_t)255;
                for (int i = 2; i < L; i++) { shape5 y = u->ls[i]; y.c = w[i]; rs += (act_bytes_of(y) + 255) & ~(size_t)255; }
                rs += (act_bytes_of(t1) + 255) & ~(size_t)255; rs += (act_bytes_of(c1) + 255) & ~(size_t)255;
                const size_t rt = A + ((act_bytes_of(t0) + 255) & ~(size_t)255);   /* dec[0]'s conv1 phase-A temp (R alone then) */
                if (rt > rs) rs = rt;
                if (rs + A <= blk_data) {
                    p3 = 1; u->slab_s = s; u->r_bytes = rs; u->x_bytes = xb; u->y_bytes = yb;
                    const size_t data_end = (size_t)nb0 * blk_data, off = (data_end - rs) & ~(size_t)255;
                    u->zr = (int)((off - (size_t)(nb0 - 1) * blk_data) / ((size_t)y0.h * y0.w * rb0));
                    u->r_bytes = data_end - off;
                }
            }
        }
        if (p3 && !g_dry) {
            u->h_r = nn_host_alloc_try(u->r_bytes);
            if (!u->h_r) { fprintf(stderr, "unet: cannot pin %.2f GB for the coarse offload; separate coarse buffers\n", u->r_bytes / 1e9); p3 = 0; u->slab_s = slab ? (infer_slab() + 3) / 4 * 4 : 0; }
            else for (int k = 0; k < 4; k++) if (!u->ev_r[k]) u->ev_r[k] = nn_event_create();
        }
        if (!p3 && off_ok) { shape5 d = u->ls[1]; d.c = w[0]; if (act_bytes_of(d) > act_bytes_of(m2)) m2 = d; }   /* down0's output back in T2 */
        if (slab) {
            g_mlab = "X (level 0, in-place slabs) + slab temp";
            X = dalloc_act_s(u, y0);
            if (slab1) Y = dalloc_act_s(u, y1);
        }
        u->p3 = p3;
        if (p3) {   /* R = the end of X's data region; every coarse buffer inside it, the level-0 temps inside Y */
            u->Xp = X; u->Yp = Y;
            {   /* (MX layout: data rows of every block, then the scales; R ends where X's data ends) */
                const int bw0 = mx_bw_of(w[0]), nb0 = (w[0] + bw0 - 1) / bw0, rb0 = mx_rb_of(w[0]);
                u->R = (float *)((char *)X + (size_t)nb0 * shape_spatial(y0) * rb0 - u->r_bytes);
            }
            char *r = (char *)u->R + 256;
            T1 = (float *)r; r += (act_bytes_of(m1) + 255) & ~(size_t)255;
            T2 = (float *)r; r += (act_bytes_of(m2) + 255) & ~(size_t)255;
            for (int i = 2; i < L; i++) { shape5 y = u->ls[i]; y.c = w[i]; SK[i] = (float *)r; r += (act_bytes_of(y) + 255) & ~(size_t)255; }
            u->slab_tmp1 = (float *)r; { shape5 t1 = y1; t1.d = u->slab_s < y1.d ? u->slab_s : y1.d; r += (act_bytes_of(t1) + 255) & ~(size_t)255; }
            u->slab_carry1 = (float *)r;
            u->slab_tmp = (float *)((char *)Y + 256);
            { shape5 t0 = y0; t0.d = u->slab_s < y0.d ? u->slab_s : y0.d; u->slab_xin = (float *)((char *)u->slab_tmp + ((act_bytes_of(t0) + 255) & ~(size_t)255)); }
            g_mlab = "X (level 0, in-place slabs) + slab temp";
            { shape5 c0 = y0; c0.d = 1; u->slab_carry = dalloc_act_s(u, c0); }
            /* dec[0]'s conv1: R, then Y's consumed coarse planes as temp; its own temp if that schedule does not reach the end */
            int B = 0, ok = 1;
            for (int z0 = 0, sk; z0 < y0.d && ok; z0 += sk) { sk = dc1_next(u, z0, &B); if (!sk) ok = 0; }
            if (!ok || infer_offload() == 2) { shape5 t0 = y0; t0.d = u->slab_s < y0.d ? u->slab_s : y0.d; u->slab_tmpd = dalloc_act_s(u, t0); }   /* (2: always, a test hook) */
            u->ct_stem = ct_ok;
            u->t1_bytes = act_bytes_of(m1); u->t1p = T1;
            TC = nullptr;
        } else {
            T1 = dalloc_act_s(u, m1); T2 = dalloc_act_s(u, m2); TC = rc ? nullptr : dalloc_act_s(u, mc);   /* recompute: rc_tmp allocates on demand */
            u->t1_bytes = act_bytes_of(m1); u->t1p = T1;
            if (slab) {
                g_mlab = "X (level 0, in-place slabs) + slab temp";
                shape5 ts = y0; ts.d = u->slab_s < y0.d ? u->slab_s : y0.d; u->slab_tmp = dalloc_act_s(u, ts);
                ts.d = 1; u->slab_carry = dalloc_act_s(u, ts);
                u->slab_tmp1 = u->slab_tmp; u->slab_carry1 = u->slab_carry;
                u->ct_stem = ct_ok;
                if (u->ct_stem) {   /* the stem's MX input (fp8, or fp4: the trainer's --gpu-input precision) of one slab and its halo planes */
                    g_mlab = "logits (fp32) / network input";
                    shape5 xi = xs; xi.d = (u->slab_s < xs.d ? u->slab_s : xs.d) + 2; if (xi.d > xs.d) xi.d = xs.d;
                    const size_t xb_ = nn_mx_bytes(xi, ip); u->act_bytes += xb_;
                    u->slab_xin = dalloc_oom(dmalloc(xb_), xb_); dstorage(u->slab_xin, xb_, ip);
                }
            }
        }
    }
    g_mlab = "encoder a1 / a2 / GN stats + down outputs";
    const int sea = train && share_enc_a1() && L > 1;
    for (int i = 0; i < L; i++) {
        shape5 bin = u->ls[i]; bin.c = i == 0 ? u->cfg.cin : w[i - 1];
        float *ea1 = T1;
        if (sea && i < L - 1 && u->l3b && i == 0) {   /* X1 (unet.l3b): also the up transient, level 0's B / gskip, the input in enc[0]'s backward */
            shape5 y = u->ls[0]; y.c = w[0];
            size_t b = act_bytes_of(y), xb = nn_mx_bytes(xs, unet_input_prec());
            b = ((xb > b ? xb : b) + 255) & ~(size_t)255;
            g_mlab = "level-0 X1 (a1 / up transient / gskip / input)";
            ea1 = dalloc_oom(dmalloc(b), b); u->act_bytes += b; dstorage(ea1, b, act_dt());
            u->l3x[1] = ea1; u->l3xb[1] = b;
            g_mlab = "encoder a1 / a2 / GN stats + down outputs";
        } else if (sea && i < L - 1) { shape5 y = u->ls[i]; y.c = w[i]; ea1 = dalloc_act_s(u, y); }   /* shared with dec[i].a1 (below) */
        if (slab && i == 0) ea1 = X;
        if (slab1 && i == 1) ea1 = Y;
        if (u->l3b && i == 0) g_mlab = "level-0 X2 (skip / dec[0].a2 / coarse bwd homes)";
        float *ra2 = nullptr;
        if (u->l0ring && i == 0) {   /* (stage S4: enc[0].a1 and the skip in slab rings) */
            shape5 y = u->ls[0]; y.c = w[0];
            g_mlab = "level-0 slab rings (stage S4)";
            ea1 = dalloc_ring(u, y, act_dt(), l0ring_R(0), 0); ra2 = dalloc_ring(u, y, act_dt(), l0ring_R(1), 0);
            g_mlab = "encoder a1 / a2 / GN stats + down outputs";
        }
        build_block_acts(u, &u->enc[i], bin, train, i == L - 1, ea1,
                         ra2 ? ra2 : slab && i == 0 ? X : slab1 && i == 1 ? Y : u->p3 ? SK[i] : rc && train && sko_on() && !u->cfg.enc_blocks[i] && i > 0 && i < L - 1 ? SKO_DEFER : rc || u->cfg.enc_blocks[i] ? nullptr : T2, nullptr);   /* recompute mode keeps the (small) outputs that
                                                                                                                         get upsampled; training: the coarse skips placed later */
        if (u->l3b && i == 0) { u->l3x[2] = u->enc[0].a2; u->l3xb[2] = act_bytes_of(u->enc[0].ys); g_mlab = "encoder a1 / a2 / GN stats + down outputs"; }
        if (sea && i < L - 1) {
            u->enc[i].a1_shared = 1;
            const int off = a1_offload() && !(i > 0 && u->l3s);   /* (l3_sched: the coarse levels re-run conv1) */
            if (off) { shape5 y = u->ls[i]; y.c = w[i]; g_hoff += act_bytes_of(y); }
            if (off && !g_dry) {
                shape5 y = u->ls[i]; y.c = w[i];
                u->h_a1[i] = nn_host_alloc_try(act_bytes_of(y));   /* nullptr: this level re-runs conv1 */
                if (!u->h_a1[i]) fprintf(stderr, "unet: cannot pin %.2f GB for the encoder a1 offload; level %d re-runs conv1\n", act_bytes_of(y) / 1e9, i);
                for (int k = 0; k < 4; k++) if (!u->ev_a1[i][k]) u->ev_a1[i][k] = nn_event_create();
            }
        }
        if (i < L - 1) {
            shape5 ds = u->ls[i + 1]; ds.c = w[i];
            u->downo[i] = u->p3 && i == 0 ? (float *)((char *)Y + p3_down_off(u)) : share ? T2 : train && dno_on() ? nullptr : dalloc_act_s(u, ds);   /* (training: placed with the gradient buffers;
                                                                                         coarse offload: down0's output in Y, read in place by enc[1]'s conv1) */
            if (u->cfg.down_norm) {
                int G = G_of(u, w[i]);
                u->dm[i] = dalloc(u, (size_t)ds.n * G); u->dr[i] = dalloc(u, (size_t)ds.n * G);
                if (!nn_get_tf32()) u->downs[i] = dalloc(u, shape_numel(ds));   /* fp32 path: materialised silu(gn(.)) */
            }
        }
    }
    g_mlab = "residual blocks";
    for (int i = 0; i < L; i++) for (int k = 0; k < u->cfg.enc_blocks[i]; k++) {   /* own buffers; conv1 reads silu(gn(previous a2)) in staging */
        block *b = &u->rb[i][k], *prev = k ? &u->rb[i][k - 1] : &u->enc[i];
        shape5 y = u->ls[i]; y.c = w[i];
        const int last = k == u->cfg.enc_blocks[i] - 1;   /* the stage output: a coarse skip placed with the skip offload */
        build_block_acts(u, b, y, train, i == L - 1 && last, nullptr, last && rc && train && sko_on() && i > 0 && i < L - 1 ? SKO_DEFER : nullptr, nullptr);
        b->inx = prev->a2; b->ing = nn_get_tf32() ? &prev->n2 : nullptr; b->inm = prev->m2; b->inr = prev->r2;
    }
    if (u->tbn) {   /* attention stage: its output block, fp32 token rows (one set per block input), one block's scratch, gradients */
        g_mlab = "attention stage";
        if (!nn_get_tf32() || sp_on(u)) { fprintf(stderr, "unet: attention blocks need the tensor-core path and no spatial split\n"); abort(); }
        shape5 y = u->ls[L - 1]; y.c = w[L - 1];
        build_block_acts(u, &u->tb, y, train, 1, nullptr, nullptr, nullptr);
        const long Sg = (long)shape_spatial(y), T = (long)y.n * (Sg + u->nrg); const int C = w[L - 1];
        u->tbg = (at_geom){y.n, (int)(Sg + u->nrg), y.d, y.h, y.w, nullptr, nullptr, nullptr};
        long Tr = 0;
        if (unet_cfg_axis(&u->cfg)) {   /* the scroll geometry's buffers (tb_geom fills them per forward) */
            at_rays_dims(y.d, y.h, y.w, attn_ray_mult(), &u->tb_Q, &u->tb_Nt, &u->tb_Nr);
            const int Pm = y.d > y.h ? (y.d > y.w ? y.d : y.w) : (y.h > y.w ? y.h : y.w);
            Tr = u->cfg.attn_axis == 1 ? (long)y.n * u->tb_Q * u->tb_Nr : 0;
            const size_t nq = (size_t)y.n * u->tb_Q, al = 64;
            const size_t o_ray = 0, o_len = o_ray + (nq * AT_RAY_F + al - 1) / al * al, o_ad = o_len + (nq + al - 1) / al * al, o_pos = o_ad + al,
                         o_ws = o_pos + ((size_t)Tr * 3 + al - 1) / al * al, o_gp = o_ws + ((size_t)y.n * Sg + al - 1) / al * al,
                         o_cen = o_gp + ((size_t)T * 3 + al - 1) / al * al, o_fr = o_cen + ((size_t)y.n * Pm * 4 + al - 1) / al * al,
                         o_end = o_fr + (u->tbd.dInit ? ((size_t)T * 9 + al - 1) / al * al : 0);
            u->tb_rb = dalloc(u, o_end);
            u->tbr = (at_rays){y.n, u->tb_Q, u->tb_Nt, u->tb_Nr, y.d, y.h, y.w, (const int *)(u->tb_rb + o_ad), u->tb_rb + o_ray, (const int *)(u->tb_rb + o_len),
                               u->tb_rb + o_pos, u->tb_rb + o_ws};
            u->tb_gpos = u->tb_rb + o_gp; u->tb_cen = u->tb_rb + o_cen;
            free(u->tb_rayh); free(u->tb_cenh); free(u->tb_lensh); free(u->tb_adh); free(u->tb_axb);
            u->tb_rayh = malloc(nq * AT_RAY_F * sizeof(float)); u->tb_cenh = malloc((size_t)y.n * Pm * 4 * sizeof(float));
            u->tb_lensh = malloc(nq * sizeof(int)); u->tb_adh = malloc((size_t)y.n * sizeof(int)); u->tb_axb = malloc((size_t)y.n * AT_AXR(Pm) * sizeof(float));
            if (!u->tb_rayh || !u->tb_cenh || !u->tb_lensh || !u->tb_adh || !u->tb_axb) { fprintf(stderr, "unet: no host memory for the ray plan\n"); abort(); }
            u->tb_axdirty = 1;
            if (u->cfg.attn_axis == 1) u->tbg.rays = &u->tbr;
            else if (u->cfg.attn_pe == 3) u->tbg.pos = u->tb_gpos;
            u->tb_dfr = u->tbd.dInit ? u->tb_rb + o_fr : nullptr; u->tbg.dfr = u->tb_dfr;
        }
        u->tb_Tr = Tr;
        u->tb_cs = dalloc(u, (size_t)y.n * C * Sg);
        if (u->tbd.dK) {   /* deformable attention: the finer levels' token rows (filled per forward) */
            size_t mx = 0;
            for (int l = 1; l < u->tbd.dL; l++) {
                const shape5 s = stage_out0(u, L - 1 - l)->ys;
                if (s.d != y.d << l || s.h != y.h << l || s.w != y.w << l || s.c != u->tbd.dC[l]) {
                    fprintf(stderr, "unet: deformable attention: level %d is %dx%dx%d, not 2^%d x the bottleneck %dx%dx%d\n", L - 1 - l, s.d, s.h, s.w, l, y.d, y.h, y.w); abort(); }
                const size_t ne = shape_numel(s); u->tb_dfx[l] = dalloc(u, ne); if (ne > mx) mx = ne;
                u->tbg.dfx[l] = u->tb_dfx[l];
            }
            u->tb_dfcs = mx ? dalloc(u, mx) : nullptr;
        }
        u->tb_tok = dalloc(u, (size_t)(u->tba + 2) * T * C);
        const size_t wsb = at_ws_bytes_r(&u->tbd, T, Tr);
        u->tb_wsb = dalloc(u, wsb / 4 + 1);
        if (!g_dry) at_ws_carve_r(&u->tbw, &u->tbd, T, Tr, u->tb_wsb);
        if (train && attn_keep()) {   /* one forward set per block application (unet_set_attn_keep) */
            const size_t fb = at_ws_fwd_bytes_r(&u->tbd, T, Tr) / 4;
            u->tb_keep = dalloc(u, (size_t)u->tba * fb + 1);
            for (int k = 0; k < u->tba; k++) { u->tbk[k] = u->tbw; if (!g_dry) at_ws_set_fwd_r(&u->tbk[k], &u->tbd, T, Tr, u->tb_keep + (size_t)k * fb); }
        }
        u->tb_zero = dalloc(u, (size_t)(u->nrg + 1) * C);
        if (!g_dry) nn_zero(u->tb_zero, (size_t)(u->nrg + 1) * C * 4);
        if (u->cfg.attn_mem) {   /* window memory: its input (copied at the forward), the offset features, the output rows */
            const size_t nm = (size_t)y.n * u->cfg.attn_mem * C;
            u->tb_memin = dalloc(u, nm); u->tb_memf = dalloc(u, (size_t)y.n * 4 + 64); u->tb_memout = dalloc(u, nm);
            if (!g_dry) { nn_zero(u->tb_memin, nm * 4); nn_zero(u->tb_memf, ((size_t)y.n * 4 + 64) * 4); nn_zero(u->tb_memout, nm * 4); }
        }
        if (u->cfg.view_P) {   /* scan fusion: the view patches (kept for the weight gradient) and their embeddings / row gradients */
            const int k = u->cfg.view_patch; const size_t nr = (size_t)y.n * u->tb_nv;
            u->tb_vxp = dalloc(u, nr * k * k * k); u->tb_ve = dalloc(u, nr * C);
        }
        if (u->cfg.attn_vres) u->tb_vr = dalloc(u, (size_t)(u->tba + 2) * T * C);   /* (v1, dv1, one own V per block application) */
        if (train) {
            const int gmx = grad_mx8();
            u->tb_g = gmx ? dalloc_grad_mx(u, nn_mx_bytes(y, grad_dt())) : dalloc_grad(u, shape_numel(y));
            shape5 yi = y; yi.c = L > 1 ? w[L - 2] : u->cfg.cin;   /* (the bottleneck block's backward leaves d/d its input in this buffer too) */
            const size_t bx = gmx ? (nn_mx_bytes(y, grad_dt()) > nn_mx_bytes(yi, grad_dt()) ? nn_mx_bytes(y, grad_dt()) : nn_mx_bytes(yi, grad_dt())) : 0,
                         nx = shape_numel(y) > shape_numel(yi) ? shape_numel(y) : shape_numel(yi);
            u->tb_gx = gmx ? dalloc_grad_mx(u, bx) : dalloc_grad(u, nx);
            u->tb_dt = dalloc(u, (size_t)T * C); u->tb_dt2 = dalloc(u, (size_t)T * C);
        }
    }
    g_mlab = "decoder a1 / a2 / GN stats";
    for (int i = L - 2; i >= 0; i--) {
        shape5 li = u->ls[i];
        shape5 cs_ = li; cs_.c = nn_get_tf32() ? UPW(i + 1) : w[i] + UPW(i + 1);
        if (share && !rc) u->cat[i] = TC;
        else if (!rc) u->cat[i] = dalloc_act_s(u, cs_);
        shape5 cin = li; cin.c = w[i] + UPW(i + 1);
        /* A 16-bit decoder's kept SiLU may also use its consumed encoder skip. In MX modes s2
           is absent and the next decoder normalizes a2 on the fly. Training keeps separate buffers. */
        float *rd1 = nullptr, *rd2 = nullptr;
        if (u->l0ring && i == 0) {
            shape5 y = u->ls[0]; y.c = w[0];
            g_mlab = "level-0 slab rings (stage S4)";
            rd1 = dalloc_ring(u, y, act_dt(), l0ring_R(2), 0);
            if (u->l0hb) rd2 = dalloc_ring(u, y, act_dt(), l0ring_R(3), 0);   /* (the stream head reads it per slab) */
            g_mlab = "decoder a1 / a2 / GN stats";
        }
        build_block_acts(u, &u->dec[i], cin, train, i > 0,
                         rd1 ? rd1 : slab && i == 0 ? X : slab1 && i == 1 ? Y : sea ? u->enc[i].a1 : train && i > 0 && keep_coarse_a1() ? nullptr : T1,
                         rd2 ? rd2 : u->dec[i].single ? nullptr : (reuse_skip || u->l3b) && i == 0 ? u->enc[0].a2 : slab1 && i == 1 ? Y : T2,   /* (l3b: X2) */
                         reuse_skip && i > 0 ? stage_out(u, i)->a2 : nullptr);
        if (sea) u->dec[i].a1_lent = 1;
    }
    shape5 os = u->ls[0]; os.c = u->cfg.cout;
    u->logits = nullptr; u->logits_h16 = 0;
    int lg_gout = 0, lg_fit = 1, l3_ok = 1;   /* fp16 logits at the start of gout[0] (MX-fp4 gradients, below); lg_fit: they and their gradient fit
                                      the level-0 buffers */
    if (train) {
        if (nn_get_tf32()) {
            /* tensor-core path: A only holds the level width and B the widest of {width, decoder up part, encoder block
               input}; both are block-local, so one pair sized for the largest level serves every level. gout[i] is dead
               once dec[i]'s backward has read it, before that block writes gskip[i]: they share a buffer. */
            g_mlab = "gradient buffer A";
            const int gmx = grad_mx8(), chunk = chunk_up();
            {   /* level-0 A in dec[0].a2's buffer (see a0_share_on) */
                shape5 d0 = u->ls[0]; d0.c = w[0];
                const int idt = input_mx() ? unet_input_prec() : 0;
                const int xin_will = ABF && idt && xin_offload() && act_bytes_of(d0) >= nn_mx_bytes(xs, idt);
                const int enc0_rerun = recompute_a1() || (share_enc_a1() && !(a1_offload() && (g_dry || u->h_a1[0])));
                /* the level-0 offload chain (A in dec[0].a2, enc[0].a2 on the host, gco / dno / sko below) no longer needs the fp16
                   logits and their gradient to fit level-0 buffers (11-channel heads: 22 > 17 B per voxel turned the whole chain off,
                   ~+28 B per voxel): wider heads keep both in buffers of their own (lg_fit 0) */
                u->a0_share = a0_share_on() && L > 1 && act_mx4() && grad_mx4() && grad_dt() == act_dt() && lean() >= 2 && chunk && !any_single(u)
                              && !(xin_will && enc0_rerun);
                lg_fit = act_bytes_of(d0) >= shape_numel(os) * 2 + 4096;
                if (u->a0_share) g_hoff += act_bytes_of(d0);
                if (u->a0_share && !g_dry) {
                    u->h_a2 = nn_host_alloc_try(act_bytes_of(d0));
                    if (!u->h_a2) { fprintf(stderr, "unet: cannot pin %.2f GB for the skip offload; level-0 A keeps its own buffer\n", act_bytes_of(d0) / 1e9); u->a0_share = 0; }
                    else for (int k = 0; k < 4; k++) if (!u->ev_a2[k]) u->ev_a2[k] = nn_event_create();
                }
            }
            size_t na = 0, nbb = 0;   /* elements (16-bit / fp32) or bytes (MX) */
            for (int i = u->l0ring ? 1 : 0; i < L; i++) {   /* (stage S4: level 0's A and B are rings of their own) */
                /* chunk mode: the up-part gradient goes through B in w[i]-channel chunks, so B needs no w[i+1] */
                int cb[3] = {w[i], i < L - 1 && !chunk ? UPW(i + 1) : 0, i ? w[i - 1] : u->cfg.cin};
                for (int k = 0; k < 3; k++) {   /* B holds each of these tensors at this level */
                    shape5 sb = u->ls[i]; sb.c = cb[k];
                    size_t e = !cb[k] ? 0 : gmx ? nn_mx_bytes(sb, grad_dt()) : shape_numel(sb);
                    if (e > nbb) nbb = e;
                }
                shape5 sa = u->ls[i]; sa.c = w[i];
                size_t e = gmx ? nn_mx_bytes(sa, grad_dt()) : shape_numel(sa);
                if (e > na && !(u->a0_share && i < L - 1 && UPW(i) >= w[i])) na = e;   /* a0_share: a decoder level's A is its dec[i].a2's buffer
                                                                                     (a narrower decoder's is too small for the encoder's gradient) */
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
                if (u->a0_share && lg_fit) lg_gout = 1;   /* the logit gradient goes to enc[0].a2's buffer */
                if (u->a0_share && !lg_fit) lg_gout = 0;   /* (and the logits to a buffer of their own: fp16 with unet_set_fp16_logits) */
                if (lg_gout) { u->logits = nullptr; u->logits_h16 = 1; }
                if (u->l3b) { lg_gout = 0; u->logits = nullptr; u->logits_h16 = 1; }   /* (l3b: the fp16 logits in X3, below) */
            }
            u->logits_bytes = shape_numel(os) * (u->logits_h16 ? 2 : 4);
            g_mlab = "gradient buffers gout / gskip (per level)";
            size_t gobs[UNET_MAXLEV];
            for (int i = 0; i < L; i++) {
                shape5 so = u->ls[i]; so.c = w[i];
                size_t gob = gmx ? nn_mx_bytes(so, grad_dt()) : shape_numel(so) * (GBF ? 2 : 4);
                if (u->nob && i < L - 1 && !up_wg_chunk()) { shape5 up = u->ls[i]; up.c = UPW(i + 1); if (act_bytes_of(up) > gob) gob = act_bytes_of(up); }   /* also holds the MX up transient (level 1: 34 vs 33 B) */
                if (up_wg_chunk() && i > 0 && act_bytes_of(so) > gob) gob = act_bytes_of(so);   /* holds the coarse silu(gn(a2)) the level below's weight gradient reads */
                if (up_wg_chunk() && i < L - 1) { shape5 sl = u->ls[i]; sl.c = UPW(i + 1) % 32 ? UPW(i + 1) : 32; if (act_bytes_of(sl) > gob) gob = act_bytes_of(sl); }   /* one upsampled 32-channel slice (a narrower or
                                                                                                     non-32-multiple up part: the whole up part, unsliced) */
                if (u->nob && gmx && chunk && i < L - 1) {   /* the smallest legal up-part chunk passes through this level's gradient buffer */
                    shape5 t = u->ls[i]; t.c = mx_chunk_min(UPW(i + 1)); const size_t cb = nn_mx_bytes(t, grad_dt()); if (cb > gob) gob = cb;
                }
                if (u->nob && gmx && i == 0 && i < L - 1 && unet_wide_up_grad()) {
                    shape5 up = u->ls[i]; up.c = UPW(i + 1);
                    size_t ub = nn_mx_bytes(up, grad_dt()); if (ub > gob) gob = ub;
                }   /* optional full finest-level up gradient; the chunk loop still defers the aliased skip write */
                if (lg_gout && i == 0) gob += 4096;   /* alignment slack: [fp16 logits | ... | batch] (unet_batch_scratch) */
                gobs[i] = gob;
            }
            {   /* the coarse gout in dec[0]'s buffers (see gco_on) */
                const size_t hb = act_bytes_of(u->dec[0].ys);
                size_t need = 256;
                for (int i = 1; i < L; i++) need += (gobs[i] + 255) & ~(size_t)255;
                u->gco = gco_on() && gmx && u->nob && u->a0_share && recompute() == 1 && !recompute_a1() && L > 1 && grad_dt() == act_dt()
                         && need <= hb && ((gobs[1] + 255) & ~(size_t)255) + 256 <= act_bytes_of(u->dec[0].ys);
            }
            if (u->l3b) {   /* X3 (unet.l3b): level 0's A, the input (5 B per voxel), then the coarse forward homes from 256 on, then the fp16 logits
                               and their gradient: all below the trainer's batch tail at its end (unet_batch_scratch2) */
                size_t lo = 256;
                for (int k = 1; k < L - 1; k++) lo += (act_bytes_of(u->enc[k].ys) + 255) & ~(size_t)255;
                for (int k = 0; k < L - 1; k++) { shape5 ds = u->ls[k + 1]; ds.c = w[k]; lo += (act_bytes_of(ds) + 255) & ~(size_t)255; }
                const size_t xb = nn_mx_bytes(xs, unet_input_prec()), lb = shape_numel(os) * 2;
                if (xb > lo) lo = xb;
                if (lb > lo) lo = lb;
                lo = (lo + 255) & ~(size_t)255;
                shape5 d0 = u->ls[0]; d0.c = w[0];
                size_t b = act_bytes_of(d0);
                if (lo + ((u->bat_tail + 255) & ~(size_t)255) > b) b = lo + ((u->bat_tail + 255) & ~(size_t)255);
                b = (b + 255) & ~(size_t)255;
                if (gobs[0] > u->l3xb[1]) l3_ok = 0;   /* (X1 is level 0's B too) */
                g_mlab = "level-0 X3 (A / input / logits / batch tail)";
                u->l3x[3] = dalloc_oom(dmalloc(b), b); u->act_bytes += b; u->grad_bytes += b; dstorage(u->l3x[3], b, grad_dt());
                u->l3xb[3] = b; u->l3_lo = lo;
                g_mlab = "gradient buffers gout / gskip (per level)";
            }
            size_t gco_off = act_bytes_of(u->dec[0].ys) & ~(size_t)255;
            for (int i = 0; i < L; i++) {
                const size_t gob = u->l3b && i == 0 ? u->l3xb[1] : gobs[i];
                u->gA[i] = u->l3b && i == 0 ? u->l3x[3] : u->a0_share && i < L - 1 && UPW(i) >= w[i] ? u->dec[i].a2 : A;   /* dead after the first op of dec[i]'s backward */
                if (u->l3b && i == 0) u->gout[0] = u->l3x[1];   /* (X1: B and gskip of level 0 as well) */
                else if (u->l0hb && i == 0) {   /* (stage S4, the stream head: gout[0] a ring, the skip gradient whole on its own below) */
                    shape5 a0 = u->ls[0]; a0.c = w[0];
                    g_mlab = "level-0 slab rings (stage S4)";
                    u->gout[0] = dalloc_ring(u, a0, grad_dt(), l0ring_R(8), 1);
                    g_mlab = "gradient buffers gout / gskip (per level)";
                } else if (u->gco && i > 0) {
                    gco_off -= (gob + 255) & ~(size_t)255;
                    u->gout[i] = (float *)((char *)u->dec[0].a2 + gco_off);
                    dstorage(u->gout[i], gob, grad_dt());
                    if (i == 1) {
                        u->gout_home1 = u->gout[1];
                        if (!u->l3b) {   /* (l3b: X1 takes dec[0]'s up transient; gout[1] stays home, its skip comes back later) */
                            u->gout_x1 = (float *)((char *)u->dec[0].a1 + ((act_bytes_of(u->dec[0].ys) - gob) & ~(size_t)255));
                            dstorage(u->gout_x1, gob, grad_dt());
                        }
                    }
                } else
                u->gout[i] = gmx ? dalloc_grad_mx(u, gob) : dalloc_grad(u, gob / (GBF ? 2 : 4) + 1);
                u->gskip[i] = i < L - 1 ? u->gout[i] : nullptr;
                u->gout_cap[i] = gmx ? gob : (gob / (GBF ? 2 : 4) + 1) * (GBF ? 2 : 4);
                if (u->l0hb && i == 0) {   /* (the skip gradient: from dec[0]'s conv1 backward-data until down0's -- in a ring with each slab copied to the
                                                 pinned host as dec[0]'s backward completes it and back before down0's accumulates into it (the host mode),
                                                 or whole on the device) */
                    shape5 a0 = u->ls[0]; a0.c = w[0];
                    u->gout_cap[0] = nn_mx_bytes(a0, grad_dt());
                    if (l0ring_skip_host() && !g_dry) { u->l0r_hsk = nn_host_alloc_try(u->gout_cap[0]); u->l0r_hskn = u->gout_cap[0]; }
                    if (l0ring_skip_host() && !g_dry && !u->l0r_hsk) fprintf(stderr, "unet: cannot pin %.2f GB for the skip gradient's host copy; it stays whole on the device\n", u->gout_cap[0] / 1e9);
                    if (l0ring_skip_host() && (g_dry || u->l0r_hsk)) {
                        g_hoff += u->gout_cap[0];
                        g_mlab = "level-0 slab rings (stage S4)";
                        u->gskip[0] = dalloc_ring(u, a0, grad_dt(), l0ring_R(10), 1);
                    } else {
                        g_mlab = "level-0 skip gradient (whole)";
                        u->gskip[0] = dalloc_grad_mx(u, u->gout_cap[0]);
                    }
                    g_mlab = "gradient buffers gout / gskip (per level)";
                }
                if (u->nob) { u->gB[i] = u->gout[i]; u->gB_cap[i] = gob; }   /* no B: the level's own gradient buffer serves as B */
                else { u->gB[i] = B; u->gB_cap[i] = u->gB_bytes; }
            }
            if (u->nob) u->gB_bytes = u->gB_cap[0];
            if (u->l0ring) {   /* (stage S4: level 0's A, its B = the up-part gradient, the stream's d/d a2 and dec[0]'s up transient in rings;
                                  silu(gn(dec[1].a2)) at level 1, the up transient's per-slab source) */
                shape5 a0 = u->ls[0]; a0.c = w[0];
                shape5 b0 = u->ls[0]; b0.c = UPW(1);   /* (dec[0].c_split as the forward sets it) */
                g_mlab = "level-0 slab rings (stage S4)";
                u->gA[0] = dalloc_ring(u, a0, grad_dt(), l0ring_R(5), 1);
                u->gB[0] = dalloc_ring(u, b0, grad_dt(), l0ring_R(6), 1); u->gB_cap[0] = nn_mx_bytes(b0, grad_dt());
                u->l0st_g2 = dalloc_ring(u, a0, grad_dt(), l0ring_R(4), 1); u->l0st_g2n = nn_mx_bytes(a0, grad_dt());
                u->l0st_up = dalloc_ring(u, b0, act_dt(), l0ring_R(7), 0); u->l0st_upn = act_bytes_of(b0);
                const block *xb = L == 2 ? stage_out(u, L - 1) : &u->dec[1];   /* (dec[0].xb as the forward sets it) */
                if (act_dt() == grad_dt() && u->gout_cap[1] >= act_bytes_of(xb->ys)) { u->l0r_ups = u->gout[1]; u->l0r_upsn = 0; }   /* (gout[1]: free until B1's
                    upsample backward writes it, top down behind the up windows' reads; one storage type: no re-registration) */
                else { g_mlab = "level-1 up source (stage S4)"; u->l0r_ups = dalloc_act_s(u, xb->ys); u->l0r_upsn = act_bytes_of(xb->ys); }
                g_mlab = "gradient buffers gout / gskip (per level)";
            }
            for (int i = 0; i < L; i++) if (u->cfg.enc_blocks[i] > 0) {   /* residual blocks: the identity gradient of the stream */
                shape5 sg = u->ls[i]; sg.c = w[i];
                u->gres_cap[i] = gmx ? nn_mx_bytes(sg, grad_dt()) : shape_numel(sg) * (GBF ? 2 : 4);
                u->gres[i] = gmx ? dalloc_grad_mx(u, u->gres_cap[i]) : dalloc_grad(u, shape_numel(sg));
            }
            {   /* the down-conv outputs below the coarse gout in dec[0].a2's buffer (see dno_on), above the input it holds at first */
                size_t dn = 0;
                for (int i = 0; i < L - 1; i++) { shape5 ds = u->ls[i + 1]; ds.c = w[i]; dn += (act_bytes_of(ds) + 255) & ~(size_t)255; }
                const int idt = ABF && input_mx() ? unet_input_prec() : 0;
                const size_t xlo = idt && xin_offload() && !u->l3b ? (nn_mx_bytes(xs, idt) + 255) & ~(size_t)255 : 0;   /* (if the input will share it; l3b: X3) */
                u->dno = u->gco && dno_on() && gco_off >= dn + xlo + 256;
                if (u->dno) g_hoff += dn;
                for (int i = 0; u->dno && !g_dry && i < L - 1; i++) {
                    shape5 ds = u->ls[i + 1]; ds.c = w[i];
                    if (!(u->h_dn[i] = nn_host_alloc_try(act_bytes_of(ds)))) { fprintf(stderr, "unet: cannot pin %.2f GB for the down-conv output offload; own buffers\n", act_bytes_of(ds) / 1e9); u->dno = 0; }
                    for (int k = 0; k < 4; k++) if (!u->ev_dn[i][k]) u->ev_dn[i][k] = nn_event_create();
                }
                if (!u->dno) for (int i = 0; i < L - 1; i++) if (u->h_dn[i]) { nn_host_free(u->h_dn[i]); u->h_dn[i] = nullptr; }
                const char *lab = g_mlab; g_mlab = "encoder a1 / a2 / GN stats + down outputs";
                for (int i = 0; i < L - 1; i++) {
                    if (u->downo[i]) continue;
                    shape5 ds = u->ls[i + 1]; ds.c = w[i];
                    if (u->dno) { gco_off -= (act_bytes_of(ds) + 255) & ~(size_t)255; u->downo[i] = (float *)((char *)u->dec[0].a2 + gco_off); dstorage(u->downo[i], act_bytes_of(ds), act_dt()); }
                    else u->downo[i] = dalloc_act_s(u, ds);
                }
                /* the coarse skips at the start of the buffer (from 256 B on: a registration of its own, not the buffer's) */
                size_t sk = 256;
                for (int i = 1; i < L - 1; i++) sk += (act_bytes_of(u->enc[i].ys) + 255) & ~(size_t)255;   /* (a stage output: enc[i]'s shape) */
                u->sko = u->dno && sko_on() && sk <= gco_off;
                if (u->sko) g_hoff += sk - 256;
                for (int i = 1; u->sko && !g_dry && i < L - 1; i++) {
                    if (!(u->h_sk[i] = nn_host_alloc_try(act_bytes_of(u->enc[i].ys)))) { fprintf(stderr, "unet: cannot pin %.2f GB for the skip offload; own buffers\n", act_bytes_of(u->enc[i].ys) / 1e9); u->sko = 0; }
                    for (int k = 0; k < 4; k++) if (!u->ev_sk[i][k]) u->ev_sk[i][k] = nn_event_create();
                }
                if (!u->sko) for (int i = 1; i < L - 1; i++) if (u->h_sk[i]) { nn_host_free(u->h_sk[i]); u->h_sk[i] = nullptr; }
                size_t so = 256;
                for (int i = 1; i < L - 1; i++) {
                    block *b = stage_out(u, i);
                    if (b->a2 != SKO_DEFER) continue;
                    if (u->sko) { b->a2 = (float *)((char *)u->dec[0].a2 + so); dstorage(b->a2, act_bytes_of(b->ys), act_dt()); so += (act_bytes_of(b->ys) + 255) & ~(size_t)255; }
                    else b->a2 = dalloc_act_s(u, b->ys);
                }
                g_mlab = lab;
            }
            if (u->l3b && u->dno && u->sko) {   /* the coarse homes: backward as placed above (X2), forward in X3 from 256 on (unet.l3b) */
                size_t o = 256;
                for (int k = 1; k < L - 1; k++) { u->skb[k] = stage_out(u, k)->a2; u->skf[k] = (float *)((char *)u->l3x[3] + o); o += (act_bytes_of(u->enc[k].ys) + 255) & ~(size_t)255; }
                for (int k = 0; k < L - 1; k++) { shape5 ds = u->ls[k + 1]; ds.c = w[k]; u->dnb[k] = u->downo[k]; u->dnf[k] = (float *)((char *)u->l3x[3] + o); o += (act_bytes_of(ds) + 255) & ~(size_t)255; }
            }
            if (lg_gout) {
                if (u->gout_cap[0] >= (u->logits_bytes + 255) / 256 * 256) u->logits = u->gout[0];
                else u->logits = A;   /* fp16 logits in A, the logit gradient separate */
            }
            if (u->l3b) u->logits = u->l3x[3];
            g_mlab = "decoder up-part transient (rc_extra)";
            if (act_mx8() && recompute() && !up_wg_chunk()) {   /* the MX weight-gradient transient of the decoder up part where it does not fit its level's
                                                 buffer: allocated now so the dry build (memory planner) counts it */
                shape5 m = u->ls[0]; m.c = 0;
                for (int i = 0; i < L - 1; i++) { shape5 c = u->ls[i]; c.c = UPW(i + 1); if (act_bytes_of(c) > u->gB_cap[i] && act_bytes_of(c) > act_bytes_of(m)) m = c; }
                if (m.c) { u->rc_extra = dalloc_act_s(u, m); u->rc_extra_bytes = act_bytes_of(m); }
            }
        } else
        for (int i = 0; i < L; i++) {
            shape5 li = u->ls[i];
            size_t S = shape_spatial(li);
            int wmax = i < L - 1 ? w[i] + UPW(i + 1) : w[i];
            if (i == 0 && u->cfg.cin > wmax) wmax = u->cfg.cin;
            size_t nb = (size_t)li.n * wmax * S, nw = (size_t)li.n * w[i] * S;
            u->gA[i] = dalloc_grad(u, nb); u->gB[i] = dalloc_grad(u, nb);
            if (!nn_get_tf32()) { u->t1[i] = dalloc(u, nw); u->t2[i] = dalloc(u, nw); }
            u->gskip[i] = dalloc_grad(u, nw); u->gout[i] = dalloc_grad(u, nw);
            if (u->cfg.enc_blocks[i] > 0) { u->gres[i] = dalloc_grad(u, nw); u->gres_cap[i] = nw * (GBF ? 2 : 4); }
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
    g_mlab = "encoder a1 / a2 / GN stats + down outputs";
    for (int i = 0; i < L - 1; i++) if (!u->downo[i]) { shape5 ds = u->ls[i + 1]; ds.c = w[i]; u->downo[i] = dalloc_act_s(u, ds); }   /* (deferred, not placed) */
    for (int i = 0; i < L; i++) { block *so = stage_out(u, i); if (so->a2 == SKO_DEFER) so->a2 = dalloc_act_s(u, so->ys); }
    g_mlab = "logits (fp32 / fp16) / network input";
    if (!train && u->head_nc) os.c = u->head_nc;   /* inference: the requested logit channels only */
    if (u->lh16_req && !u->logits && act_mx8() && nn_get_tf32()) {   /* fp16 logits (the MX head writes them: lp_conv1_fwd_mx_h16) in their own buffer */
        u->logits_h16 = 1;
        if (train) u->logits_bytes = shape_numel(os) * 2;
    }
    const size_t lgb = shape_numel(os) * (u->logits_h16 ? 2 : 4);
    const int ialias = share && reuse_skip && infer_alias() && act_mx8();   /* inference aliasing (see unet.xin_inf / lg_t1) */
    u->lg_t1 = u->xin_inf = u->lg_x = 0;
    if (!u->logits && ialias && !u->p3 && lgb <= u->t1_bytes) { u->logits = T1; u->lg_t1 = 1; u->lg_p = T1; u->lg_bytes = u->t1_bytes; }   /* (coarse offload: T1 is in X) */
    else if (!u->logits && slab1 && lgb <= act_bytes_of(u->enc[1].ys)) {   /* Y: dec[1].a2, dead once dec[0]'s conv1 has upsampled it */
        u->logits = Y; u->lg_t1 = 1; u->lg_p = Y; u->lg_bytes = act_bytes_of(u->enc[1].ys);
    } else if (!u->logits && slab && !train && ialias && u->logits_h16 && os.c == 1 && (w[0] + mx_bw_of(w[0]) - 1) / mx_bw_of(w[0]) == 1 && rc
               && lgb <= shape_spatial(u->ls[0]) * (size_t)mx_rb_of(w[0]) && !(getenv("UFSM_INFER_LGX") && !ufsm_env_on("UFSM_INFER_LGX"))) {
        /* X (dec[0].a2) itself: the fp16 plane-major logits over its first data rows, written per z-slab after the slab's head has read
           its planes (lg_x; 16-wide level 0: Y holds 1.1 of the 2 B per voxel; UFSM_INFER_LGX=0: their own buffer) */
        shape5 y0 = u->ls[0]; y0.c = w[0];
        u->logits = X; u->lg_t1 = u->lg_x = 1; u->lg_p = X; u->lg_bytes = act_bytes_of(y0);
    }
    if (train && (u->head_ext || u->l0hb)) { u->logits = nullptr; u->lg_t1 = 0; }   /* head slabs (or the stream head, stage S4): no whole-window logits */
    else if (!u->logits) u->logits = dalloc(u, (lgb + 3) / 4);
    for (int j = 0; j < UNET_MAXLEV; j++) { u->ds_lg[j] = nullptr; u->ds_gl[j] = nullptr; }
    if (train) for (int j = 1; j <= u->cfg.ds_levels; j++) {   /* deep supervision: fp32 logits per supervised decoder level */
        g_mlab = "deep-supervision logits"; shape5 os = u->ls[j]; os.c = u->cfg.cout; u->ds_lg[j] = dalloc(u, shape_numel(os));
    }
    u->xin = nullptr;
    if (ABF && input_mx()) {   /* include mandatory input storage in the dry planner */
        const int idt = unet_input_prec(); const size_t xb_ = nn_mx_bytes(xs, idt);
        if (ialias && !slab && act_bytes_of(u->enc[0].ys) >= xb_) { u->xin = u->enc[0].a2; u->xin_inf = 1; u->xin_bytes = xb_; }   /* (slabs: the stem writes X) */

        if (train && nn_get_tf32() && act_mx8() && xin_offload() && L > 1 && (u->l3b || act_bytes_of(u->dec[0].ys) >= xb_)) {
            void *h = g_dry ? nullptr : nn_host_alloc_try(xb_);
            if (g_dry || h) {
                g_hoff += xb_;
                u->xin = u->l3b ? u->l3x[3] : u->dec[0].a2; u->xin_shared = 1; u->h_xin = h; u->xin_bytes = xb_;
                if (!g_dry) for (int k = 0; k < 4; k++) if (!u->ev_xin[k]) u->ev_xin[k] = nn_event_create();
            } else fprintf(stderr, "unet: cannot pin %.2f GB for the input offload; a separate input buffer\n", xb_ / 1e9);
        }
        u->l0rx = u->l0hb && u->rad && unet_input_prec() && !u->cfg.half_res;   /* (stage S4: --gpu-input records set before the build) */
        if (u->l0rx && !u->xin) { g_mlab = "level-0 slab rings (stage S4)"; u->xin = dalloc_ring(u, xs, unet_input_prec(), l0ring_R(9), 0); }
        if (!u->xin && !u->ct_stem) u->xin = dalloc_input(u, xs);   /* (CT stem: none; unet_forward_x allocates one on demand) */
    }
    u->glb = nullptr;   /* allocated on first use for a fp32 logit gradient */
    if (u->l3b && !(l3_ok && u->a0_share && u->gco && u->dno && u->sko && u->xin_shared && u->xin == u->l3x[3] && u->logits == u->l3x[3]
                    && (g_dry || (u->h_a1[0] && u->h_a2 && u->h_xin)))) {   /* (a size, a host buffer): the four-buffer layout */
        fprintf(stderr, "unet: three level-0 buffers unavailable (A share %d, coarse shares %d %d %d, input %d, host %d); the four-buffer layout\n",
                u->a0_share, u->gco, u->dno, u->sko, u->xin_shared, g_dry || (u->h_a1[0] && u->h_a2 && u->h_xin));
        u->l3_no = 1;
        free_acts(u);
        if (g_dry) g_nmrep = 0;
        build_acts(u, xs, train);
        return;
    }
    u->xs = xs; u->built = 1; u->train = train; u->mode = UMODE();
}

size_t unet_activation_bytes(const unet *u) { return u->act_bytes; }
/* build the activation / gradient buffers for xs now (as the first forward would) */
/* inference: compute and keep only logit channels [c0, c0 + nc) (nc 0: all); the forward's logits then hold those channels */
void unet_set_head_channels(unet *u, int c0, int nc) {
    if (c0 < 0 || nc < 0 || c0 + nc > u->cfg.cout) { fprintf(stderr, "unet_set_head_channels: [%d, %d) outside %d outputs\n", c0, c0 + nc, u->cfg.cout); abort(); }
    if (nc == u->cfg.cout) nc = 0;
    if (u->head_c0 == c0 && u->head_nc == nc) return;
    if (u->built && !u->train) free_acts(u);
    u->head_c0 = c0; u->head_nc = nc;
}
/* inference build: the shared a1 scratch, free from the forward's start until the stem conv writes it (after the input
   conversion has read the network input), so a caller's 16-bit input may live there; nullptr if too small or not built so */
void *unet_input_scratch(unet *u, size_t bytes) {
    if (!u->built || u->train || !nn_get_tf32() || !input_mx() || !u->enc[0].a1) return nullptr;
    const block *b = &u->enc[0]; shape5 s = b->ys;
    return act_bytes_of(s) >= bytes && u->dec[0].a1 == b->a1 ? (void *)b->a1 : nullptr;
}
/* inference build with the reused level-0 skip: enc[0].a2's buffer, free from the forward's start until the input conversion writes
   the network input there (or the stem writes it), so a caller's window data read before the forward (the uint8 CT) may live there */
void *unet_input_scratch2(unet *u, size_t bytes) {
    if (!u->built || u->train || !nn_get_tf32() || !u->enc[0].a2 || u->dec[0].a2 != u->enc[0].a2 || !infer_alias()) return nullptr;
    const size_t ab = act_bytes_of(u->enc[0].ys);
    if (u->slab) {   /* X also holds the 16-bit input (unet_input_scratch, at its start, up to 8 B per voxel): the CT at its end */
        const size_t x16 = shape_spatial(u->enc[0].ys) * 8, off = ab >= bytes ? (ab - bytes) & ~(size_t)255 : 0;
        return ab >= bytes && off >= x16 ? (void *)((char *)u->enc[0].a2 + off) : nullptr;
    }
    return ab >= bytes ? (void *)u->enc[0].a2 : nullptr;
}
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
    if (u->l3b) return nullptr;   /* (three level-0 buffers: the gradient over the fp16 logits in X3) */
    if (u->nob && u->a0_share) return u->logits == u->gout[0] && act_bytes_of(u->enc[0].ys) >= bytes ? (void *)u->enc[0].a2 : nullptr;
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
/* the logits sit in the buffer the head's backward-data writes (gout[0]: fp16 logits of the lean 2 MX-fp4 layout) */
int unet_logits_in_gx(const unet *u) { return u->built && u->train && u->logits && u->logits == u->gout[0]; }
void unet_set_input_rad(unet *u, const void *rec) { u->rad = rec; }
int unet_logits_h16(const unet *u) { return u->built && u->logits_h16; }
void unet_set_fp16_logits(unet *u, int on) {   /* takes effect at the next build */
    if (u->lh16_req == !!on) return;
    if (u->built && !u->train) free_acts(u);
    u->lh16_req = !!on;
}
void unet_set_wd_nonconv(unet *u, float mul) { u->wd_nonconv = mul; }
void unet_set_head_ext(unet *u, int on) {   /* takes effect at the next build */
    if (u->head_ext == !!on) return;
    if (u->built) free_acts(u);
    u->head_ext = !!on;
}
void unet_set_head_cb(unet *u, void (*cb)(void *), void *ctx) { u->head_cb = cb; u->head_ctx = ctx; }
void unet_set_l0_head(unet *u, int (*fwd)(void *, int, int, const void *), int (*bwd)(void *, int, int, void *), void *ctx) {
    if (u->built && (!fwd) != (!u->l0h_fwd)) free_acts(u);   /* (a ring build places dec[0].a2, gout[0] and the logits by it) */
    u->l0h_fwd = fwd; u->l0h_bwd = bwd; u->l0h_ctx = ctx;
}
void *unet_grad_scratch(unet *u, size_t bytes) { return u->built && u->train && lean() && u->gB[0] && u->gB_bytes >= bytes ? (void *)u->gB[0] : nullptr; }
/* lean 2: the trainer's batch as [head | tail] in gout[0]; with the fp16 logits at its start (MX-fp4 gradients) at the buffer's
   end, the tail (targets, masks: read by the loss) clear of the logits, the head (the input: read only by the forward's
   conversion, before the head writes the logits) allowed over them */
void *unet_batch_scratch(unet *u, size_t head, size_t tail) {
    if (!(u->built && u->train && lean() && u->nob && u->gB[0])) return nullptr;
    if (u->l3b) return nullptr;   /* (unet_batch_scratch2) */
    if (u->logits != u->gout[0]) return u->gB_bytes >= head + tail ? (void *)u->gB[0] : nullptr;
    if (u->gB_bytes < head + tail) return nullptr;
    const size_t p = (u->gB_bytes - head - tail) & ~(size_t)255, lg = (u->logits_bytes + 255) & ~(size_t)255;
    return p + head >= lg ? (void *)((char *)u->gB[0] + p) : nullptr;
}
/* three level-0 buffers (unet.l3b): the batch's head (the input, read by the forward's conversion before enc[0]'s conv2) in X2 (X1 if
   larger), its tail (targets, masks: read by the loss after the forward) at X3's end, *tailp; nullptr if not this layout or too large.
   unet_set_batch_bytes before the build sizes X3 for the tail */
void *unet_batch_scratch2(unet *u, size_t head, size_t tail, void **tailp) {
    *tailp = nullptr;
    if (!(u->built && u->train && u->l3b)) return nullptr;
    const size_t p = u->l3xb[3] >= tail ? (u->l3xb[3] - tail) & ~(size_t)255 : 0;
    if (!tail || p < u->l3_lo) return nullptr;
    void *h = head <= u->l3xb[2] ? (void *)u->l3x[2] : head <= u->l3xb[1] ? (void *)u->l3x[1] : nullptr;
    if (h) *tailp = (char *)u->l3x[3] + p;
    return h;
}
void unet_set_batch_bytes(unet *u, size_t head, size_t tail) {
    u->bat_head = head;   /* (placed by unet_batch_scratch2; X3's size follows the tail) */
    if (u->bat_tail == tail) return;
    if (u->built && u->l3b) free_acts(u);
    u->bat_tail = tail;
}
int unet_l0_3buf(const unet *u) { return u->built && u->l3b; }
int unet_l0_3buf_ct(const unet *u) { return u->built && u->l3_ct; }   /* the last training forward kept the CT for the input's rebuild */
void *unet_l0_3buf_head_free(unet *u) {   /* (created here so a copy stream can wait for it before the first backward: a no-op wait) */
    if (!(u->built && u->l3b)) return nullptr;
    if (!u->ev_head) u->ev_head = nn_event_create();
    return u->ev_head;
}
static int g_last_dry_l3b;
int unet_last_dry_l3b(void) { return g_last_dry_l3b; }   /* the last dry build (unet_train_bytes) used the three level-0 buffers */
/* device bytes of the training activations / gradients at input shape xs under the current storage modes, without allocating
   (dry build); the model is left unbuilt */
static size_t dry_bytes(unet *u, shape5 xs, int train) {
    if (u->built) free_acts(u);
    g_dry = 1; g_nmrep = 0; g_mlab = "other";
    g_mrep_nv = getenv("UFSM_MEM_REPORT") && atoi(getenv("UFSM_MEM_REPORT")) >= 2 ? (double)xs.n * xs.d * xs.h * xs.w : 0;
    build_acts(u, xs, train);
    g_mrep_nv = 0;
    g_last_dry_l3b = u->l3b;
    const size_t b = u->act_bytes;
    if (ufsm_env_on("UFSM_MEM_REPORT")) {
        const double nv = (double)xs.n * xs.d * xs.h * xs.w;
        fprintf(stderr, "memory report (dry %s build %dx%dx%dx%d, mode %d): %.3f GB, %.1f B per input voxel\n", train ? "training" : "inference", xs.n, xs.d, xs.h, xs.w, UMODE(), b / 1e9, b / nv);
        for (int i = 0; i < g_nmrep; i++) fprintf(stderr, "  %-46s %8.3f GB %6.1f B/voxel (%d allocations)\n", g_mrep[i].lab, g_mrep[i].b / 1e9, g_mrep[i].b / nv, g_mrep[i].n);
    }
    free_acts(u);
    g_dry = 0;
    return b;
}
size_t unet_train_bytes(unet *u, shape5 xs) { return dry_bytes(u, xs, 1); }
size_t unet_infer_bytes(unet *u, shape5 xs) { return dry_bytes(u, xs, 0); }
size_t unet_grad_bytes(const unet *u) { return u->grad_bytes; }
shape5 unet_out_shape(const unet *u, shape5 xs) { xs.c = u->cfg.cout; return xs; }

/* ---- forward ---- */
static const float *P(const unet *u, size_t off) { return u->fw + off; }
/* conv1 bias of an encoder block: the scale-conditioned effective bias for the stem */
static const float *B1(const unet *u, const block *b) { return b == &u->enc[0] && u->cfg.scale_cond ? u->stem_b : P(u, b->c1.b); }
void unet_set_scale(unet *u, double um) { u->sc_c = unet_scale_code(um); }
void unet_set_cond(unet *u, float c) { u->sc_c = c; }
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
            shape5 m = u->ls[0]; m.c = 0;   /* the largest transient that does not fit its level's buffer */
            for (int i = 0; i < u->cfg.nlev - 1; i++) { shape5 c = u->ls[i]; c.c = UPW(i + 1); if (act_bytes_of(c) > u->gB_cap[i] && act_bytes_of(c) > act_bytes_of(m)) m = c; }
            u->rc_extra = dalloc_act_s(u, m); u->rc_extra_bytes = act_bytes_of(m);
        }
        *reg = 0;
        return u->rc_extra;
    }
    if (!u->train && !u->cat[0]) {   /* inference: the transient is only allocated if the fused path is unavailable */
        shape5 m = u->ls[0]; m.c = 0;
        for (int i = 0; i < u->cfg.nlev - 1; i++) { shape5 c = u->ls[i]; c.c = UPW(i + 1); if (act_bytes_of(c) > act_bytes_of(m)) m = c; }
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
/* l3b, the skip's way back into X2 for dec[0]'s conv1 weight gradient: bytes [lo, hi) of it (the host copy's layout is the buffer's);
   wait: stream 0 waits for it and every part before it (one copy stream) */
static void l3_skip_copy(unet *u, size_t lo, size_t hi, int wait) {
    if (hi > lo) nn_offload_copy((char *)u->enc[0].a2 + lo, (const char *)u->h_a2 + lo, hi - lo, 0, u->ev_a2[2], u->ev_a2[3]);
    if (wait) nn_stream_wait(0, u->ev_a2[3]);
}
/* l3b, the copy schedule (env UFSM_L3_SCHED, default on): every copy on the one copy stream in the order it is due. Forward: the encoder's
   a1 out once the stem wrote it (dec[0]'s conv1 waits), the skip once enc[0] finished (dec[0]'s conv2), then the down outputs and the coarse
   skips (their forward homes in X3 are only overwritten by the logits: the head waits for them, not dec[0]'s conv2). Backward: after
   dec[0]'s, the coarse tensors back, the skip into X3 (down0's weight gradient) and the encoder's a1 into X2 but for the coarse homes'
   bytes, which follow after down0's backward. The coarse encoders re-run conv1 instead of their a1 copies (a few ms of small convs, the
   same kernel and rounding keys), which their decoders waited for behind level 0's: one copy engine (the laptop under WSL runs every copy
   one at a time, H2D and D2H too, in the order they became ready) cannot let a small copy pass a big one. Same bits */
static int g_l3_sched = -1;
static int l3_sched_on(void) { if (g_l3_sched < 0) g_l3_sched = getenv("UFSM_L3_SCHED") ? ufsm_env_on("UFSM_L3_SCHED") : 1; return g_l3_sched; }
static int l3_sched(const unet *u) { return u->l3s; }
void unet_set_l3_sched(int on) { g_l3_sched = on; }   /* -1: the env / default (the next build) */
/* (l3_sched) the skip for down0's weight gradient: the part of it dec[0]'s backward left in X2 copied into X3 on the device, the host
   copy only for the rest (gout[1]'s bytes): -1.05 GB of host-to-device copy per P512 step for a ~5 ms copy kernel; env
   UFSM_L3_SKIP_D2D, default on (0: the whole skip from the host). Same bits */
static int g_l3_skip_d2d = -1;
static int l3_skip_d2d(void) { if (g_l3_skip_d2d < 0) g_l3_skip_d2d = getenv("UFSM_L3_SKIP_D2D") ? ufsm_env_on("UFSM_L3_SKIP_D2D") : 1; return g_l3_skip_d2d; }
void unet_set_l3_skip_d2d(int on) { g_l3_skip_d2d = on; }   /* -1: the env / default */
/* (l3_sched) X2's bytes [*lo, *hi) that no coarse tensor uses in the coarse levels' backward: between the coarse skips' homes at its start
   and the down outputs' and coarse gradients' at its end (the encoder's a1 comes back there early) */
static void l3_x2_free(const unet *u, size_t *lo, size_t *hi) {
    const char *x = (const char *)u->l3x[2], *xe = x + u->l3xb[2];
    const int L = u->cfg.nlev;
    size_t a = 256, b = u->l3xb[2];
    for (int k = 1; k < L - 1; k++) if (u->skb[k]) { const size_t e = (size_t)((const char *)u->skb[k] - x) + act_bytes_of(u->enc[k].ys); if (e > a) a = e; }
    for (int k = 0; k < L - 1; k++) if (u->dnb[k]) { const size_t s = (size_t)((const char *)u->dnb[k] - x); if (s < b) b = s; }
    for (int i = 1; i < L; i++) if ((const char *)u->gout[i] >= x && (const char *)u->gout[i] < xe) { const size_t s = (size_t)((const char *)u->gout[i] - x); if (s < b) b = s; }
    a = (a + 255) & ~(size_t)255; b &= ~(size_t)255;
    *lo = a; *hi = b > a ? b : a;
}
/* the offload copy order (env UFSM_A1_OUT_SPLIT, default on): level 0's shared encoder a1 out in two halves, the second after the encoders,
   the down-conv outputs out after the encoders too, level 0's a1 back after the decoders. On the one copy stream the coarse encoders' own a1 copies, which their decoders wait for
   before conv1, queued behind the whole level-0 one (1.2 GB, 42 ms at P512 on the laptop's 28 GB/s) and the down outputs: dec[L-2]'s conv1
   waited ~27 ms per step. dec[0]'s conv1, long after, waits for the level-0 halves; its conv2 / the head for the down outputs. Backward:
   the coarse encoders' a1 queued behind level 0's (enc[1]'s backward waited ~6 ms) */
static int a1_out_split(void) { static int v = -1; if (v < 0) v = getenv("UFSM_A1_OUT_SPLIT") ? ufsm_env_on("UFSM_A1_OUT_SPLIT") : 1; return v; }
/* l3b with the GPU-built input (unet_set_input_rad, --gpu-input): the 16-bit CT channel the input was built from goes to the host instead of
   the MX input and comes back into X1's end for the stem's weight gradient, the input rebuilt below it by nn_input_rad (the forward's
   bits); env UFSM_L0_3BUF_CT, default on. l3_ct_off: the CT's offset in X1, 0 if it does not fit beside the input */
static int l3_ct_on(void) { static int v = -1; if (v < 0) v = getenv("UFSM_L0_3BUF_CT") ? ufsm_env_on("UFSM_L0_3BUF_CT") : 1; return v; }
static size_t l3_ct_bytes(const unet *u) { return (size_t)u->xs.n * u->xs.d * u->xs.h * u->xs.w * 2; }
static size_t l3_ct_off(const unet *u) {
    const size_t cb = l3_ct_bytes(u), lo = (u->xin_bytes + 255) & ~(size_t)255;
    if (cb > u->l3xb[1]) return 0;
    const size_t off = (u->l3xb[1] - cb) & ~(size_t)255;
    return off >= lo ? off : 0;
}
/* l3b: enc[0]'s conv2 backward-data before its weight gradient (env UFSM_L0_3BUF_REORDER, default on), so the encoder's a1 comes back into
   X2 during its GroupNorm backward and that backward-data instead of exposed */
static int g_l3_reorder = -1;
static int l3_reorder(void) { if (g_l3_reorder < 0) g_l3_reorder = getenv("UFSM_L0_3BUF_REORDER") ? ufsm_env_on("UFSM_L0_3BUF_REORDER") : 1; return g_l3_reorder; }
void unet_set_l0_3buf_reorder(int on) { g_l3_reorder = on; }   /* -1: the env / default */
/* l3b: the skip's bytes outside the up transient's coarse temp (silu(gn(dec[1].a2)) at the start of gout[1]'s home) come back from dec[0]'s
   conv2 backward-data on, those after the up transient. The split of dec[0]'s conv1 weight gradient (env UFSM_L0_3BUF_SPLIT=1): the up part's (X1, X3) during that
   copy, then the skip part's. Off by default: the two 16-channel weight gradients cost more than the copy they hide (laptop 5080, P512:
   +13 ms vs ~-5 ms); default: one weight gradient after the copy */
static int g_l3_split = -1;
static int l3_split(void) { if (g_l3_split < 0) g_l3_split = getenv("UFSM_L0_3BUF_SPLIT") ? ufsm_env_on("UFSM_L0_3BUF_SPLIT") : 0; return g_l3_split; }
void unet_set_l0_3buf_split(int on) { g_l3_split = on; }   /* -1: the env / default */
static size_t l3_skip_lo(const unet *u) { return (size_t)((const char *)u->gout_home1 - (const char *)u->l3x[2]); }
static size_t l3_skip_hi(const unet *u) { return l3_skip_lo(u) + ((act_bytes_of(u->dec[1].ys) + 255) & ~(size_t)255); }   /* (the coarse temp's end) */
/* the two parts accumulate into gw through wg_tmp (gw holds only this conv's gradient: zeroed before the backward); the seeds of one call
   (the skip part replays the up part's draws: the gy SR is keyed by element, so both parts see the same gy) */
static int l3_wgrad_split(unet *u, block *b, const float *up, const float *gy, float *gw, float *gbias) {
    const block *x2b = (const block *)b->x2b;
    nn_gn_t g2 = gn_out(u, x2b);
    const int cin = b->xs.c, cs = b->c_split, sk = cin - cs, cout = b->c1.cout;
    const size_t need = (size_t)cout * (cs > sk ? cs : sk) * 27;
    if (need > u->wg_tmp_n) { nn_free(u->wg_tmp); u->wg_tmp = nn_malloc(need * 4); u->wg_tmp_n = need; }
    shape5 xu = b->xs, xk = b->xs; xu.c = cs; xk.c = sk;
    const unsigned c0 = nn_sr_ctr();
    int rv;
    nn_zero(u->wg_tmp, (size_t)cout * cs * 27 * 4);
    PROF(2, rv = nn_conv3d_bwd_weight_x(up, nullptr, nullptr, nullptr, 0, 0, xu, gy, b->ys, 3, 1, u->wg_tmp, gbias));
    if (rv) return rv;
    nn_add_rows(gw, (size_t)cin * 27, u->wg_tmp, (size_t)cs * 27, cout, (size_t)cs * 27);
    const unsigned c1 = nn_sr_ctr();
    l3_skip_copy(u, 0, 0, 1);   /* the skip back */
    nn_sr_set_ctr(c0);
    nn_zero(u->wg_tmp, (size_t)cout * sk * 27 * 4);
    PROF(2, rv = nn_conv3d_bwd_weight_x(x2b->a2, &g2, nullptr, nullptr, 0, 0, xk, gy, b->ys, 3, 1, u->wg_tmp, nullptr));
    if (rv) return rv;
    nn_add_rows(gw + (size_t)cs * 27, (size_t)cin * 27, u->wg_tmp, (size_t)sk * 27, cout, (size_t)sk * 27);
    if (nn_sr_ctr() != c1) rc_fail("three level-0 buffers: the split weight gradient's seed draws");
    return 0;
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
    if (gy && u->l3_skip_in && b == &u->dec[0]) {   /* l3b: the up transient built from the coarse temp in gout[1]'s home (X2), dead now */
        u->l3_skip_in = 0;
        l3_skip_copy(u, l3_skip_lo(u), l3_skip_hi(u), !l3_split());   /* the coarse temp's bytes (the rest came back since dec[0]'s conv2 backward-data) */
        if (l3_split()) {
            rv = l3_wgrad_split(u, b, up, gy, gw, gbias);
            rc_done(u, up, reg);
            return rv;
        }
    }
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
static int gn_frozen(const unet *u) { return u->gn_frz && !u->fwd_train && nn_get_tf32(); }
static void gn_frozen_load(unet *u);
/* cfg.l0_norm: the level-0 GroupNorm sites, l0_site = 2 * (dec[0]) + conv - 1 (conv 1: m1 / r1, 2: m2 / r2; a one-conv dec[0] has only
   2). While a forward's level-0 statistics are constants (u->l0_c: none, or ema after its warm-up; latched by l0_begin for the
   backward) they sit in m1 / r1 / m2 / r2 before the convs run, the convs write none (none, ema outside training) or write them to
   l0live for the running update after the block (ema), and the block's GroupNorm backwards drop the statistics' terms (nn_set_gn_const) */
static int l0_blk(const unet *u, const block *b) { return b == &u->enc[0] ? 0 : b == &u->dec[0] && u->cfg.nlev > 1 ? 1 : -1; }
static int l0_fixed(const unet *u, const block *b) { return u->l0_c && l0_blk(u, b) >= 0; }
static int l0_G(const unet *u) { return G_of(u, u->cfg.widths[0]); }
static float *l0_live(unet *u, const block *b, int conv, int r) {   /* where conv's statistics go while constant (r 0 mean, 1 rstd) */
    if (u->cfg.l0_norm != 2 || !u->fwd_train || l0_blk(u, b) < 0) return nullptr;
    const size_t ng = (size_t)u->xs.n * l0_G(u);
    return u->l0live + ((size_t)(2 * l0_blk(u, b) + conv - 1) * 2 + r) * ng;
}
static void l0s_begin(unet *u);
static void l0s_kern_begin(unet *u) {   /* slab kernel path: the level-0 statistics arrays and their slab geometry into the registry */
    const int S = u->cfg.l0_slab, D = u->ls[0].d, ns = l0s_nslab(u, D), hw = u->ls[0].h * u->ls[0].w;
    if (ns > 1 && D % S) { fprintf(stderr, "unet: --l0-norm slab (kernel path): the level-0 depth %d is not a multiple of the slab (%d planes)\n", D, S); abort(); }
    if (ns > 1 && (sp_on(u) || u->gn_frz)) { fprintf(stderr, "unet: --l0-norm slab (kernel path): no split / frozen statistics yet\n"); abort(); }
    if (ns > 1 && S % 8) {   /* (the MX readers stage at most 9 planes per coefficient table: two banks, i.e. two slabs, need S >= 8) */
        fprintf(stderr, "unet: --l0-norm slab (kernel path): slabs of %d planes; multiples of 8 only\n", S);
        abort();
    }
    for (int i = 0; i < 4; i++) if (u->l0s_reg[i]) { lp_gn_slab_reg(u->l0s_reg[i], 0, 0, 0, 0); u->l0s_reg[i] = nullptr; }
    if (ns <= 1) return;   /* one slab: the statistics are the sample's ([n][1][G] = [n][G]): every reader as GroupNorm, its usual kernels */
    for (int i = 0; i < (u->cfg.nlev > 1 ? 2 : 1); i++) for (int k = 1; k <= 2; k++) {
        block *b = i ? &u->dec[0] : &u->enc[0];
        if (k == 1 && b->single) continue;
        const float *m = k == 1 ? b->m1 : b->m2;
        lp_gn_slab_reg(m, S, 0, ns, hw); u->l0s_reg[2 * i + k - 1] = m;
    }
}
static void l0_begin(unet *u) {   /* forward start: latch l0_c, the constant statistics into the sites */
    const int md = u->cfg.l0_norm, G = l0_G(u), N = u->xs.n;
    if (md == 4 && l0s_kern(u)) { l0s_kern_begin(u); u->l0_c = 0; return; }   /* (live statistics, the readers apply each plane's slab) */
    if (md == 4) l0s_begin(u);
    u->l0_c = md == 1 || md == 3 || md == 4 || (md == 2 && u->l0_n > 0 && u->l0_n >= u->l0_warm && u->l0_rc < 0);   /* (recalibration: the window's own) */
    if (!u->l0_c) return;
    if (md == 2 && u->fwd_train && u->l0live_n < (size_t)8 * N * G) { nn_free(u->l0live); u->l0live_n = (size_t)8 * N * G; u->l0live = nn_malloc(u->l0live_n * 4); }
    for (int i = 0; i < (u->cfg.nlev > 1 ? 2 : 1); i++) for (int k = 1; k <= 2; k++) {
        block *b = i ? &u->dec[0] : &u->enc[0];
        if (k == 1 && b->single) continue;
        const float *run = md == 2 ? u->l0run + (u->using_ema ? (size_t)8 * G : 0) + (size_t)(2 * i + k - 1) * 2 * G : nullptr;   /* (EMA weights: their own) */
        nn_gn_const_fill(k == 1 ? b->m1 : b->m2, k == 1 ? b->r1 : b->r2, run, N, G, 1e-5f);
    }
}
/* cfg.l0_norm 4 (slab): GroupNorm statistics per z-slab of l0_slab planes. The convs write the raw level-0 tensor (no statistics), then
   l0_slab_fwd normalises it in place with its slabs' statistics (the readers apply gamma, beta + SiLU with mean 0 / rstd 1, as none);
   the GroupNorm backwards run over the pseudo-sample shape (one channel block: (sample, slab) is a sample of the same layout) on the
   normalised tensor with mean 0 / rstd 1 -- the slab's statistics terms -- and d/dx = rstd(slab) d/dx' follows (exact SR for MX-fp4
   gradients). MX storage (the training default: one MX channel block, <= 32 channels) or the fp32 path (through the slab layout).
   A prototype: the normalised tensor is rounded to the storage format once more (round to nearest) and the gradient once more */
static int l0_slab_on(const unet *u, const block *b) { return u->cfg.l0_norm == 4 && l0_blk(u, b) >= 0; }
static shape5 l0s_shape(const unet *u, const block *b) { shape5 p = b->ys; const int S = u->cfg.l0_slab; p.n *= p.d / S; p.d = S; return p; }
/* a per-sample GroupNorm op (nn_gn_silu_apply) on block b's level-0 tensor x (shape s; output y) on the slab kernel path with several
   slabs: the pseudo-sample shape ((sample, slab) of an MX tensor of one channel block is a sample of the same layout); s otherwise */
static shape5 l0s_view(const unet *u, const block *b, shape5 s, const void *x, const void *y) {
    if (!(l0_slab_on(u, b) && l0s_kern(u) && l0s_nslab(u, s.d) > 1)) return s;
    if (!nn_storage(x) || (y && !nn_storage(y)) || s.c > 32) {
        fprintf(stderr, "unet: --l0-norm slab (kernel path): a materialised level-0 GN+SiLU needs MX tensors of one channel block\n"); abort();
    }
    return l0s_shape(u, b);
}
static void l0s_begin(unet *u) {
    const int S = u->cfg.l0_slab, D = u->ls[0].d, G = l0_G(u), w0 = u->cfg.widths[0];
    if (D % S) { fprintf(stderr, "unet: --l0-norm slab: the level-0 depth %d is not a multiple of the slab (%d planes)\n", D, S); abort(); }
    if (u->cfg.widths[0] > 32 && nn_get_tf32()) { fprintf(stderr, "unet: --l0-norm slab: at most 32 level-0 channels (one MX channel block)\n"); abort(); }
    const size_t nng = (size_t)u->xs.n * (D / S) * G, need = 10 * nng + 2 * (size_t)w0;
    if (u->l0s_n < need) { nn_free(u->l0s_st); u->l0s_st = nn_malloc(need * 4); u->l0s_n = need; }
    u->l0s_nng = nng;
    nn_gn_const_fill(u->l0s_st + 8 * nng, u->l0s_st + 9 * nng, nullptr, (int)nng, 1, 1e-5f);   /* zeros, ones */
    nn_gn_const_fill(u->l0s_st + 10 * nng + w0, u->l0s_st + 10 * nng, nullptr, w0, 1, 1e-5f);   /* gamma ones, beta zeros */
    if (!nn_get_tf32()) {
        shape5 t = u->ls[0]; t.c = w0; const size_t n3 = 3 * shape_numel(t);
        if (u->l0s_tmpn < n3) { nn_free(u->l0s_tmp); u->l0s_tmp = nn_malloc(n3 * 4); u->l0s_tmpn = n3; }
    }
    u->l0s_fwd++;
}
static float *l0s_stat(unet *u, const block *b, int k, int r) { return u->l0s_st + ((size_t)(2 * l0_blk(u, b) + k - 1) * 2 + r) * u->l0s_nng; }
static void l0s_check(const unet *u, const float *x) {   /* tensor cores: MX level-0 tensors; else the fp32 reference path */
    (void)u; if (!nn_storage(x) && nn_get_tf32()) { fprintf(stderr, "unet: --l0-norm slab: the tensor-core path needs MX level-0 storage\n"); abort(); }
}
static void l0_slab_fwd(unet *u, block *b, int k, float *x, int stats) {   /* x: conv k's raw output; stats 0: the forward's (a recompute) */
    if (!l0_slab_on(u, b)) return;
    if (l0s_kern(u)) {   /* kernel path: the slabs' statistics over the pseudo-sample shape (one slab: the conv's own, as GroupNorm) */
        if (stats && l0s_nslab(u, b->ys.d) > 1) {
            float *m = k == 1 ? b->m1 : b->m2, *r = k == 1 ? b->r1 : b->r2;
            if (!nn_storage(x) || nn_gn_stats(x, l0s_shape(u, b), l0_G(u), 1e-5f, m, r)) { fprintf(stderr, "unet: --l0-norm slab (kernel path): MX level-0 storage only\n"); abort(); }
        }
        return;
    }
    const int G = l0_G(u), site = 2 * l0_blk(u, b) + k - 1, S = u->cfg.l0_slab;
    const shape5 ps = l0s_shape(u, b);
    float *m = l0s_stat(u, b, k, 0), *r = l0s_stat(u, b, k, 1);
    l0s_check(u, x);
    if (stats) u->l0s_seed[site] = ((u->l0s_fwd * 0x9e3779b9u) ^ (0x7f4a7c15u + 0x632be5abu * (unsigned)site)) | 1u;
    if (nn_storage(x)) {
        if (stats && nn_gn_stats(x, ps, G, 1e-5f, m, r)) { fprintf(stderr, "unet: --l0-norm slab: statistics unsupported for this storage\n"); abort(); }
        nn_l0_affine(x, ps, G, m, r, 0u);   /* round to nearest, as the convs store activations (a recompute: the same bits) */
        return;
    }
    float *t = u->l0s_tmp, *one = u->l0s_st + 10 * u->l0s_nng, *zero = one + u->cfg.widths[0];
    nn_slab_perm(t, x, b->ys, S, 0);
    if (stats) (void)nn_gn_stats(t, ps, G, 1e-5f, m, r);
    nn_gn_apply(t, ps, G, one, zero, m, r, t);
    nn_slab_perm(x, t, b->ys, S, 1);
}
static void l0_after(unet *u, block *b) {   /* ema, training forward: the block's batch statistics, after dec[0] into the running ones */
    const int i = l0_blk(u, b), G = l0_G(u), N = u->xs.n;
    if (i < 0 || u->cfg.l0_norm != 2 || (!u->fwd_train && u->l0_rc < 0)) return;
    float *run = u->l0run, *rune = run + (size_t)8 * G, *bat = run + (size_t)16 * G;
    for (int k = 1; k <= 2; k++) {
        const float *m = u->l0_c ? l0_live(u, b, k, 0) : k == 1 ? b->m1 : b->m2, *r = u->l0_c ? l0_live(u, b, k, 1) : k == 1 ? b->r1 : b->r2;
        if (k == 1 && b->single) m = u->l0_c ? l0_live(u, b, 2, 0) : b->m2, r = u->l0_c ? l0_live(u, b, 2, 1) : b->r2;   /* (one conv: n1 = n2) */
        nn_gn_ema_update(bat + (size_t)(2 * i + k - 1) * 2 * G, m, r, N, G, 1.f, 1e-5f);
    }
    if (i < 1 && u->cfg.nlev > 1) return;   /* (after the last level-0 block) */
    if (u->l0_rc >= 0) {   /* recalibration: the cumulative mean of the windows' statistics replaces the set of the weights in use */
        float *t = u->using_ema ? rune : run;
        if (!u->l0_rc) nn_d2d(t, bat, (size_t)8 * G * 4); else nn_ema(t, bat, (size_t)8 * G, (float)u->l0_rc / (float)(u->l0_rc + 1));
        u->l0_rc++;
        return;
    }
    if (!u->l0_n) { nn_d2d(run, bat, (size_t)8 * G * 4); nn_d2d(rune, bat, (size_t)8 * G * 4); }
    else nn_ema(run, bat, (size_t)8 * G, 1.f - u->l0_mom);
    u->l0_n++;
}
void unet_set_l0_ema(unet *u, float mom, int warm) { u->l0_mom = mom; u->l0_warm = warm; }
static void ws_backward(unet *u);
void unet_l0_recal(unet *u, int on) { if (u->cfg.l0_norm == 2) u->l0_rc = on ? 0 : -1; }
long unet_l0_updates(const unet *u) { return u->l0_n; }
/* MX tensor planes [s0, s0 + n) of src (depth Ds) -> planes [d0, d0 + n) of dst (depth Dd); one sample, s.c channels, s.h x s.w
   planes: per channel block its data rows and its scales are each plane-contiguous */
static void mx_copy_planes(void *dst, int Dd, int d0, const void *src, int Ds, int s0, int n, shape5 s) {
    if (n <= 0) return;
    const int bw = s.c <= 8 ? 8 : s.c <= 16 ? 16 : 32, nb = (s.c + bw - 1) / bw, rb = act_mx4() ? bw / 2 : bw;
    const size_t hw = (size_t)s.h * s.w, Sd = (size_t)Dd * hw, Ss = (size_t)Ds * hw;
    for (int k = 0; k < nb; k++) {
        nn_d2d_async((char *)dst + ((size_t)k * Sd + (size_t)d0 * hw) * rb, (const char *)src + ((size_t)k * Ss + (size_t)s0 * hw) * rb, (size_t)n * hw * rb);
        nn_d2d_async((char *)dst + (size_t)nb * Sd * rb + (size_t)k * Sd + (size_t)d0 * hw, (const char *)src + (size_t)nb * Ss * rb + (size_t)k * Ss + (size_t)s0 * hw, (size_t)n * hw);
    }
}
static int dec_conv1(unet *u, block *b, int level, float *y, int G, float *m, float *r, const float *gy, float *gw, float *gbias);
/* in-place z-slab conv (unet.slab): conv 1 (decoder: [up2(coarse) | skip X] -> X) or 2 (X -> X) of a level-0 block whose a1 and a2
   are both X. Each launch computes slab_s output planes into slab_tmp through the z-range (nn_set_zrange: input planes read from X,
   the output GroupNorm sums accumulated over the slabs and finalised after the last). After slab k, its planes but the last go into X:
   the input planes below the next slab's first input plane (its first output plane - 1) are dead; the last waits in the carry plane
   until the next slab is computed. Per voxel the same arithmetic as one launch (tile starts stay multiples of 4); the statistics
   differ only in the summation order. */
/* the stem of a CT forward (unet_forward_ct, unet.ct_stem): per slab its MX input planes (fp8 or fp4: unet_input_prec; one halo plane each side) built from
   the CT (nn_pred_input_mx8, bit-identical to the 16-bit input + conversion), the stem through the z-range into slab_tmp, then into X
   as slab_run (the caller's CT may live at the end of X: X's last scale planes, written plane for plane only once dead) */
static void stem_slab_run(unet *u, block *b, int G, float *om, float *orr) {
    float *X = b->a1;
    const shape5 ys = b->ys; const int D = ys.d, W = ys.w, s = u->slab_s;
    const int ip = unet_input_prec();   /* the stem input's MX format (8 or 4, build_acts ct_ok) */
    const size_t xb_ = nn_mx_bytes((shape5){1, 4, s + 2 < D ? s + 2 : D, ys.h, ys.w}, ip);
    nn_gn_t gi = gn_in(u, b);
    for (int z0 = 0, k = 0; z0 < D; z0 += s, k++) {
        const int sk = D - z0 < s ? D - z0 : s, last = z0 + sk >= D;
        const int tlo = z0 > 0 ? z0 - 1 : 0, thi = z0 + sk + 1 < D ? z0 + sk + 1 : D, nt = thi - tlo;
        nn_set_storage(u->slab_xin, xb_, ip);
        nn_pred_input_mx(u->ctin.ct, W, u->ctin.mean, u->ctin.isd, u->ctin.dyo, u->ctin.dxo, u->ctin.axis, tlo, nt, u->slab_xin, u->ctin.h16, ip);
        nn_set_zrange(z0 - tlo, nt, k == 0, last, D, z0);
        shape5 xs = {1, 4, sk, ys.h, ys.w};
        PROF(0, nn_conv3d_fwd_gn_stats(u->slab_xin, xs, gi.G, gi.gamma, gi.beta, gi.mean, gi.rstd, P(u, b->c1.w), B1(u, b), b->c1.cout, u->slab_tmp, G, 1e-5f, om, orr));
        nn_set_zrange(0, 0, 0, 0, 0, 0);
        if (k) mx_copy_planes(X, D, z0 - 1, u->slab_carry, 1, 0, 1, ys);
        mx_copy_planes(X, D, z0, u->slab_tmp, sk, 0, last ? sk : sk - 1, ys);
        if (!last) mx_copy_planes(u->slab_carry, 1, 0, u->slab_tmp, sk, sk - 1, 1, ys);
    }
    if (u->p3) nn_storage_forget(u->slab_xin);   /* (inside Y: its fp8 registration would outlive the stem) */
}
/* conv 1 of an encoder block reads x (coarse offload: enc[1]'s conv1 over down0's output, which sits in Y plane for plane, see
   p3_down_off); dec[0]'s conv1 under the coarse offload takes its temp from R, then from Y's consumed coarse planes (dc1_next), and
   brings R back from the host between the two (waited before a slab reaches plane zr) */
static void slab_run(unet *u, block *b, int level, int conv, const float *x, int G, float *om, float *orr) {
    float *X = b->a2;
    const shape5 ys = b->ys; const int D = ys.d, s = u->slab_s;
    float *carry = level ? u->slab_carry1 : u->slab_carry;
    const int dc1 = u->p3 && conv == 1 && b == &u->dec[0];
    int B = 0, waited = 0;
    if (dc1 && u->slab_tmpd && u->r_out) { nn_offload_copy(u->R, u->h_r, u->r_bytes, 0, u->ev_r[2], u->ev_r[3]); u->r_in = 1; }   /* (own temp: R comes back at once) */
    for (int z0 = 0, k = 0, sk; z0 < D; z0 += sk, k++) {
        float *tmp = level ? u->slab_tmp1 : u->slab_tmp;
        if (dc1 && !u->slab_tmpd) {
            const int B0 = B;
            sk = dc1_next(u, z0, &B);
            if (!sk) { fprintf(stderr, "unet: dec[0] conv1 slab schedule does not fit\n"); abort(); }
            if (B && !B0 && u->r_out) { nn_offload_copy(u->R, u->h_r, u->r_bytes, 0, u->ev_r[2], u->ev_r[3]); u->r_in = 1; }   /* R no longer the temp: back from the host */
            tmp = B ? (float *)((char *)u->Yp + 256) : u->R;
        } else {
            sk = D - z0 < s ? D - z0 : s;
            if (dc1) tmp = u->slab_tmpd;
        }
        if (dc1 && u->r_in && !waited && z0 + sk >= u->zr) { nn_stream_wait(0, u->ev_r[3]); waited = 1; }   /* this slab reads (and writes) R's planes */
        const int last = z0 + sk >= D;
        nn_set_zrange(z0, D, k == 0, last, D, z0);
        if (conv == 1 && b->xb) {
            const shape5 full = b->xs; b->xs.d = sk;
            const int rv = dec_conv1(u, b, level, tmp, G, om, orr, nullptr, nullptr, nullptr);
            b->xs = full;
            if (rv) rc_fail("decoder conv1 (slab)");
        } else if (conv == 1) {
            shape5 xs = b->xs; xs.d = sk;
            nn_gn_t gi = gn_in(u, b);
            PROF(0, nn_conv3d_fwd_gn_stats(x, xs, gi.G, gi.gamma, gi.beta, gi.mean, gi.rstd, P(u, b->c1.w), B1(u, b), b->c1.cout, tmp, G, 1e-5f, om, orr));
        } else {
            shape5 xs = ys; xs.d = sk;
            PROF(0, nn_conv3d_fwd_gn_stats(X, xs, G, P(u, b->n1.gamma), P(u, b->n1.beta), b->m1, b->r1, P(u, b->c2.w), P(u, b->c2.b), b->c2.cout, tmp, G, 1e-5f, om, orr));
        }
        nn_set_zrange(0, 0, 0, 0, 0, 0);
        if (k) mx_copy_planes(X, D, z0 - 1, carry, 1, 0, 1, ys);
        mx_copy_planes(X, D, z0, tmp, sk, 0, last ? sk : sk - 1, ys);
        if (!last) mx_copy_planes(carry, 1, 0, tmp, sk, sk - 1, 1, ys);
    }
    if (dc1 && u->r_in && !waited) nn_stream_wait(0, u->ev_r[3]);
    if (dc1) u->r_out = u->r_in = 0;
}
/* l3b: the coarse skips enc[1 .. L-2].a2 and down-conv outputs downo[0 .. L-2] at their forward homes (X3: fwd 1, the forward's start)
   or backward homes (X2: fwd 0, after dec[0]'s backward, before their copies back), registered there; the readers' input pointers move
   with the down-conv outputs (enc[k + 1]'s conv1 weight gradient, its down_norm GroupNorm backward) */
static void l3_homes(unet *u, int fwd) {
    const int L = u->cfg.nlev, *w = u->cfg.widths;
    for (int k = 0; k < L - 1; k++) {
        shape5 ds = u->ls[k + 1]; ds.c = w[k];
        nn_storage_forget(fwd ? u->dnb[k] : u->dnf[k]);
        u->downo[k] = fwd ? u->dnf[k] : u->dnb[k];
        nn_set_storage(u->downo[k], act_bytes_of(ds), act_dt());
        if (!fwd) { u->enc[k + 1].in = u->downo[k]; if (u->cfg.down_norm) u->enc[k + 1].inx = u->downo[k]; }   /* (the forward sets them) */
        if (k >= 1) {   /* (the level's stage output: enc[k], or its last residual block) */
            block *so = stage_out(u, k);
            nn_storage_forget(fwd ? u->skb[k] : u->skf[k]);
            so->a2 = fwd ? u->skf[k] : u->skb[k];
            nn_set_storage(so->a2, act_bytes_of(so->ys), act_dt());
        }
    }
}
static void block_fwd(unet *u, block *b, int level, const float *x) {
    b->in = x;
    int G = G_of(u, b->c1.cout);
    const int frz = gn_frozen(u), lc = l0_fixed(u, b);   /* frozen statistics already sit in m1/r1/m2/r2: the convs only replay the stores */
    float *om1 = frz ? nullptr : lc ? l0_live(u, b, 1, 0) : b->m1, *or1 = frz ? nullptr : lc ? l0_live(u, b, 1, 1) : b->r1;   /* (constant level-0 statistics: l0_live) */
    float *om2 = frz ? nullptr : lc ? l0_live(u, b, 2, 0) : b->m2, *or2 = frz ? nullptr : lc ? l0_live(u, b, 2, 1) : b->r2;
    float *t1 = u->t1[level] ? u->t1[level] : b->s2;    /* inference: no scratch, s2 doubles as temp */
    if (nn_get_tf32()) {   /* conv1 yields the stats of a1; conv2 applies gn + silu to a1 while staging and yields the stats of a2 */
        nn_set_conv(0);
        if (fakeq() && fakeq_where() == 2) {   /* keep the previous step's stats for the affine study */
            if (!b->pm1) { b->pm1 = nn_malloc((size_t)b->ys.n * G * 4 * 4); b->pr1 = b->pm1 + (size_t)b->ys.n * G; b->pm2 = b->pr1 + (size_t)b->ys.n * G; b->pr2 = b->pm2 + (size_t)b->ys.n * G; { size_t m = (size_t)b->ys.n * G; float *one = malloc(m * 4); for (size_t k = 0; k < m; k++) one[k] = 1.f; nn_zero(b->pm1, m * 4 * 4); nn_h2d(b->pr1, one, m * 4); nn_h2d(b->pr2, one, m * 4); free(one); } }
            nn_d2d(b->pm1, b->m1, (size_t)b->ys.n * G * 4); nn_d2d(b->pr1, b->r1, (size_t)b->ys.n * G * 4); nn_d2d(b->pm2, b->m2, (size_t)b->ys.n * G * 4); nn_d2d(b->pr2, b->r2, (size_t)b->ys.n * G * 4);
        }
        const int inplace = u->slab && (b == &u->enc[0] || b == &u->dec[0] || (u->slab1 && (b == &u->enc[1] || b == &u->dec[1])));   /* a1 == a2 == X (Y): in-place slab convs */
        const int sg = b->single;   /* one-conv decoder: conv1 writes the block output a2 and its statistics (m2 / r2) */
        float *o1 = sg ? b->a2 : b->a1, *sm = sg ? om2 : om1, *sr = sg ? or2 : or1;
        if (sg && b == &u->dec[0]) {   /* the copies out of dec[0].a2's buffer finish before conv1 writes it (two convs: before conv2) */
            if (u->xin_shared && u->xin_out) nn_stream_wait(0, u->ev_xin[1]);
            if (u->dno && u->dn_out) for (int k = 0; k < u->cfg.nlev - 1; k++) nn_stream_wait(0, u->ev_dn[k][1]);
            if (u->sko && u->sk_out) for (int k = 1; k < u->cfg.nlev - 1; k++) nn_stream_wait(0, u->ev_sk[k][1]);
        }
        if (b->xb && inplace) slab_run(u, b, level, 1, nullptr, G, om1, or1);
        else if (inplace && u->p3 && b == &u->enc[1]) slab_run(u, b, level, 1, x, G, om1, or1);   /* down0's output, in Y */
        else if (b->xb) { if (dec_conv1(u, b, level, o1, G, sm, sr, nullptr, nullptr, nullptr)) rc_fail("decoder conv1"); }   /* [up2(coarse) | silu(gn(skip a2))] */
        else if (b == &u->enc[0] && u->ctin.on) stem_slab_run(u, b, G, om1, or1);
        else if (b->in2) PROF(0, nn_conv3d_fwd_split(x, b->in2, b->c_split, b->xs, 0, nullptr, nullptr, nullptr, nullptr, P(u, b->c1.w), B1(u, b), b->c1.cout, o1, G, 1e-5f, sm, sr));
        else { nn_gn_t gi = gn_in(u, b); PROF(0, nn_conv3d_fwd_gn_stats(x, b->xs, gi.G, gi.gamma, gi.beta, gi.mean, gi.rstd, P(u, b->c1.w), B1(u, b), b->c1.cout, o1, G, 1e-5f, sm, sr)); }   /* down_norm: gn + silu of the input in staging */
        l0_slab_fwd(u, b, sg ? 2 : 1, o1, 1);   /* slab: normalised in place with its slabs' statistics */
        if (sg) {   /* one conv: a2 is the output */
            FQ(b->a2, b->ys); FQ_AFF(b->a2, b->ys, b->pm2, b->pr2, G);
            sp_halo(u, b->a2, b->ys, 0);
            if (b->s2) { PROF(3, nn_gn_silu_apply(b->a2, l0s_view(u, b, b->ys, b->a2, b->s2), G, P(u, b->n2.gamma), P(u, b->n2.beta), b->m2, b->r2, b->s2)); FQ(b->s2, b->ys); FQ_POST(b->s2, b->ys); }
            nn_set_conv(-1);
            return;
        }
        FQ(b->a1, b->ys); FQ_AFF(b->a1, b->ys, b->pm1, b->pr1, G);
        sp_halo(u, b->a1, b->ys, 0);
        if (b == &u->enc[0] && u->xin_inf) nn_set_storage(b->a2, act_bytes_of(b->ys), act_dt());   /* the stem has read the input: conv2 writes a2 over it */
        if (b == &u->enc[0] && l3_sched(u) && u->fwd_train && b->a1_shared && u->h_a1[0]) {   /* (l3_sched) the encoder's a1 out while conv2 reads it */
            nn_offload_copy(u->h_a1[0], b->a1, act_bytes_of(b->ys), 1, u->ev_a1[0][0], u->ev_a1[0][1]);
            u->a1_out[0] = 1;
        }
        if (b == &u->enc[0] && u->l3_ct) nn_stream_wait(0, u->ev_xin[1]);   /* (conv2 writes X2: the CT out) */
        else if (b == &u->enc[0] && u->xin_shared && u->fwd_train && u->h_xin) {   /* the stem has read the input: to the host during conv2 and the coarser levels */
            nn_offload_copy(u->h_xin, u->xin, u->xin_bytes, 1, u->ev_xin[0], u->ev_xin[1]); u->xin_out = 1;
        }
        if (b == &u->dec[0]) {   /* conv2 writes a2 over what dec[0].a2's buffer held: the input, the down-conv outputs, the coarse skips,
                                    all copied out by now (waited here, not before conv1: the last skip's copy out overlaps it) */
            if (!l3_sched(u)) {   /* (l3_sched: their forward homes are in X3, the head waits) */
                if (u->xin_shared && u->xin_out) nn_stream_wait(0, u->ev_xin[1]);
                if (u->dno && u->dn_out) for (int k = 0; k < u->cfg.nlev - 1; k++) nn_stream_wait(0, u->ev_dn[k][1]);
                if (u->sko && u->sk_out) for (int k = 1; k < u->cfg.nlev - 1; k++) nn_stream_wait(0, u->ev_sk[k][1]);
            }
            if (u->l3b && u->a2_out) nn_stream_wait(0, u->ev_a2[1]);   /* (l3b: a2 over the skip in X2) */
        }
        if (b == &u->enc[1] && u->sko) {   /* the first coarse skip goes over the input in dec[0].a2's buffer: its copy out done */
            if (u->xin_shared && u->xin_out) nn_stream_wait(0, u->ev_xin[1]);
            nn_set_storage(u->dec[0].a2, act_bytes_of(u->dec[0].ys), act_dt());
        }
        nn_set_conv(1);
        if (inplace) slab_run(u, b, level, 2, nullptr, G, om2, or2);
        else PROF(0, nn_conv3d_fwd_gn_stats(b->a1, b->ys, G, P(u, b->n1.gamma), P(u, b->n1.beta), b->m1, b->r1, P(u, b->c2.w), P(u, b->c2.b), b->c2.cout, b->a2, G, 1e-5f, om2, or2));
        nn_set_conv(-1);
        l0_slab_fwd(u, b, 2, b->a2, 1);
        FQ(b->a2, b->ys); FQ_AFF(b->a2, b->ys, b->pm2, b->pr2, G);
        sp_halo(u, b->a2, b->ys, 0);
        if (b->s2) { PROF(3, nn_gn_silu_apply(b->a2, l0s_view(u, b, b->ys, b->a2, b->s2), G, P(u, b->n2.gamma), P(u, b->n2.beta), b->m2, b->r2, b->s2)); FQ(b->s2, b->ys); FQ_POST(b->s2, b->ys); }
        return;
    }
    /* constant level-0 statistics (they sit in m / r): live ones into l0_live for the running update, then the affine + SiLU */
#define GN_FWD_SILU(a, k, nrm, y, m, r) do { l0_slab_fwd(u, b, k, a, 1); if (!lc) PROF(3, nn_gn_fwd_silu(a, b->ys, G, 1e-5f, P(u, nrm.gamma), P(u, nrm.beta), y, m, r)); \
        else { if (l0_live(u, b, k, 0)) (void)nn_gn_stats(a, b->ys, G, 1e-5f, l0_live(u, b, k, 0), l0_live(u, b, k, 1)); \
               PROF(3, nn_gn_silu_apply(a, b->ys, G, P(u, nrm.gamma), P(u, nrm.beta), m, r, y)); } } while (0)
    if (b->single) {   /* one conv: a2 = conv1's output, s2 its silu(gn) */
        PROF(0, nn_conv3d_fwd(x, b->xs, P(u, b->c1.w), B1(u, b), b->c1.cout, 3, 1, b->a2));
        GN_FWD_SILU(b->a2, 2, b->n2, b->s2, b->m2, b->r2);
        return;
    }
    PROF(0, nn_conv3d_fwd(x, b->xs, P(u, b->c1.w), B1(u, b), b->c1.cout, 3, 1, b->a1));
    GN_FWD_SILU(b->a1, 1, b->n1, t1, b->m1, b->r1);
    PROF(0, nn_conv3d_fwd(t1, b->ys, P(u, b->c2.w), P(u, b->c2.b), b->c2.cout, 3, 1, b->a2));
    GN_FWD_SILU(b->a2, 2, b->n2, b->s2, b->m2, b->r2);
#undef GN_FWD_SILU
}

/* residual block (cfg.enc_blocks): x_k = x_{k-1} + conv2(silu(gn1(conv1(silu(gn(x_{k-1})))))), gn(x_{k-1}) = the previous block's output
   normalisation (applied in conv1's staging on the tensor-core path); a2 holds x_k, its statistics are those of the sum */
static void rblock_fwd(unet *u, block *b, int level, block *prev) {
    const int G = G_of(u, b->c1.cout), frz = gn_frozen(u);
    if (nn_get_tf32()) {
        b->in = prev->a2;
        nn_gn_t gi = gn_in(u, b);
        nn_set_conv(0);
        PROF(0, nn_conv3d_fwd_gn_stats(prev->a2, b->xs, gi.G, gi.gamma, gi.beta, gi.mean, gi.rstd, P(u, b->c1.w), P(u, b->c1.b), b->c1.cout, b->a1, G, 1e-5f, frz ? nullptr : b->m1, frz ? nullptr : b->r1));
        sp_halo(u, b->a1, b->ys, 0);
        nn_set_conv(1);
        PROF(0, nn_conv3d_fwd_gn_stats(b->a1, b->ys, G, P(u, b->n1.gamma), P(u, b->n1.beta), b->m1, b->r1, P(u, b->c2.w), P(u, b->c2.b), b->c2.cout, b->a2, G, 1e-5f, nullptr, nullptr));
        nn_set_conv(-1);
        PROF(4, nn_add_into_act(b->a2, prev->a2, b->ys, ABF));
        sp_halo(u, b->a2, b->ys, 0);
        if (!frz && nn_gn_stats(b->a2, b->ys, G, 1e-5f, b->m2, b->r2)) { fprintf(stderr, "unet: residual block GroupNorm statistics unsupported for this storage\n"); abort(); }
        if (b->s2) PROF(3, nn_gn_silu_apply(b->a2, b->ys, G, P(u, b->n2.gamma), P(u, b->n2.beta), b->m2, b->r2, b->s2));
        return;
    }
    b->in = prev->s2;
    float *t1 = u->t1[level] ? u->t1[level] : b->s2;
    PROF(0, nn_conv3d_fwd(prev->s2, b->xs, P(u, b->c1.w), P(u, b->c1.b), b->c1.cout, 3, 1, b->a1));
    PROF(3, nn_gn_fwd_silu(b->a1, b->ys, G, 1e-5f, P(u, b->n1.gamma), P(u, b->n1.beta), t1, b->m1, b->r1));
    PROF(0, nn_conv3d_fwd(t1, b->ys, P(u, b->c2.w), P(u, b->c2.b), b->c2.cout, 3, 1, b->a2));
    nn_add_into_act(b->a2, prev->a2, b->ys, 0);
    PROF(3, nn_gn_fwd_silu(b->a2, b->ys, G, 1e-5f, P(u, b->n2.gamma), P(u, b->n2.beta), b->s2, b->m2, b->r2));
}

/* ---- the attention stage (cfg.attn_blocks, src/attn.h): stored tensors <-> fp32 (grad: the gradient storage), forward, backward ---- */
static void tb_to_f32(const void *x, shape5 s, int grad, float *y) {
    const int dt = nn_storage(x); const size_t S = shape_spatial(s);
    if (dt == 8) lp_mx8_to_f32(x, s.n, s.c, S, y);
    else if (dt == 4) lp_mx4_to_f32(x, s.n, s.c, S, y);
    else if (grad ? GBF : ABF) at_h16_to_f32(x, (long long)shape_numel(s), nn_get_f16(), y);
    else nn_d2d(y, x, shape_numel(s) * 4);
}
static void tb_from_f32(const float *x, shape5 s, int grad, void *y) {
    const int dt = nn_storage(y); const size_t S = shape_spatial(s);
    if (dt == 8) lp_f32_to_mx8(x, s.n, s.c, S, y);
    else if (dt == 4) lp_f32_to_mx4(x, s.n, s.c, S, y);
    else if (grad ? GBF : ABF) nn_f32_to_h16(x, shape_numel(s), y, 1.f);
    else nn_d2d(y, x, shape_numel(s) * 4);
}
#define TB_TOK(u, k) ((u)->tb_tok + (size_t)(k) * (size_t)(u)->tbg.nseq * (u)->tbg.S * (u)->tbd.C)
/* value residual hooks of block application k (at_ws.vr): block 0's first application keeps v1, the later ones lerp to it with their block's lambdas */
static void tb_vr_set(unet *u, at_ws *ws, int k) {
    if (!u->tb_vr) { ws->vr = 0; return; }
    const size_t TC = (size_t)u->tbg.nseq * u->tbg.S * u->tbd.C, H = (size_t)u->tbd.H;
    ws->vr = k ? 2 : 1; ws->v1 = u->tb_vr; ws->dv1 = u->tb_vr + TC; ws->own = u->tb_vr + (2 + (size_t)k) * TC;
    ws->lam = P(u, u->vr_off + (size_t)(k % u->tbn) * H); ws->glam = u->g + u->vr_off + (size_t)(k % u->tbn) * H;
}
void unet_set_attn_mem(unet *u, const float *mem, const float *feat, int n) {
    if (n <= 0) { u->tb_memn = 0; u->tb_memsrc = nullptr; return; }
    free(u->tb_memfh); u->tb_memfh = calloc((size_t)n * 4, sizeof(float));
    if (!u->tb_memfh) { fprintf(stderr, "unet: no memory for the window memory features\n"); abort(); }
    if (feat) memcpy(u->tb_memfh, feat, (size_t)n * 4 * sizeof(float));
    u->tb_memn = n; u->tb_memsrc = mem;
}
const float *unet_attn_mem_out(const unet *u, int *M, int *C) {
    if (M) *M = u->cfg.attn_mem;
    if (C) *C = u->tbd.C;
    return u->tbn && u->cfg.attn_mem && u->built ? u->tb_memout : nullptr;
}
/* the forward's memory input: the set rows / features for this batch size, else zero */
static void tb_mem_load(unet *u, int n) {
    const int C = u->tbd.C; const size_t nm = (size_t)n * u->cfg.attn_mem * C;
    const int have = u->tb_memn == n;
    if (have && u->tb_memsrc) nn_d2d(u->tb_memin, u->tb_memsrc, nm * 4); else nn_zero(u->tb_memin, nm * 4);
    if (have) nn_h2d(u->tb_memf, u->tb_memfh, (size_t)n * 4 * sizeof(float)); else nn_zero(u->tb_memf, (size_t)n * 4 * sizeof(float));
}
void unet_set_attn_view(unet *u, const uint8_t *view, int n) { u->tb_vsrc = n > 0 ? view : nullptr; u->tb_vn = n > 0 ? n : 0; }
/* scan fusion: the view rows of this forward (the set crops for this batch size, else zero crops): patches -> E = Xp W^T -> rows += E + b */
static void tb_view_fwd(unet *u, int n, int S) {
    const int C = u->tbd.C, Pv = u->cfg.view_P, k = u->cfg.view_patch, k3 = k * k * k;
    at_view_patch(u->tb_vn == n ? u->tb_vsrc : nullptr, n, Pv, k, u->tb_vxp);
    if (at_gemm(0, 1, n * u->tb_nv, C, k3, u->tb_vxp, P(u, u->view_w), u->tb_ve, 0.f)) { fprintf(stderr, "unet: view embedding GEMM failed\n"); abort(); }
    at_view_add(TB_TOK(u, 0), n, S, u->cfg.attn_reg + u->cfg.attn_mem, u->tb_nv, C, u->tb_ve, P(u, u->view_b));
}
void unet_set_attn_axis(unet *u, const float *rec, int n, int stride) {
    if (n <= 0 || stride < 4 || !rec) { u->tb_axn = 0; u->tb_axdirty = 1; return; }
    if (n * stride > u->tb_axn * u->tb_axs || !u->tb_axh) { free(u->tb_axh); u->tb_axh = malloc((size_t)n * stride * sizeof(float)); if (!u->tb_axh) { fprintf(stderr, "unet: no memory for axis records\n"); abort(); } }
    memcpy(u->tb_axh, rec, (size_t)n * stride * sizeof(float));
    u->tb_axn = n; u->tb_axs = stride; u->tb_axdirty = 1;
}
/* the scroll geometry of this forward (cfg.attn_axis 1 / attn_pe 3) from the axis records (unet_set_attn_axis): each sample's window
   axis centres averaged over the window planes of a bottleneck plane, in bottleneck voxels; then the ray plan, its device tables, the
   ray positions and stencil weight sums (radial) or the grid tokens' scroll positions (full attention) */
static void tb_geom(unet *u) {
    if (!u->tb_rb || !u->tb_axdirty) return;
    const shape5 ys = u->tb.ys; const int n = ys.n, D = ys.d, Hh = ys.h, W = ys.w, Pm = D > Hh ? (D > W ? D : W) : (Hh > W ? Hh : W);
    const int fp = 1 << (u->cfg.nlev - 1), have = u->tb_axh && u->tb_axn == n;
    if (!have && !u->tb_axwarn) { fprintf(stderr, "unet: the attention stage has no scroll axis records for %d samples (unet_set_attn_axis): rays along the grid rows\n", n); u->tb_axwarn = 1; }
    for (int b = 0; b < n; b++) {
        float *o = u->tb_axb + (size_t)b * AT_AXR(Pm);
        memset(o, 0, AT_AXR(Pm) * sizeof *o);
        if (!have) continue;
        const float *r = u->tb_axh + (size_t)b * u->tb_axs;
        const int ad = (int)r[0], np = (int)r[2], div = (int)r[3] > 0 ? (int)r[3] : 1, F = div * fp;
        if (ad < 0 || ad > 2) continue;
        int Np, Nu, Nv; at_ray_plane(ad, D, Hh, W, &Np, &Nu, &Nv);
        o[0] = (float)ad;
        if (r[1] == 0.f || np != Np * F || u->tb_axs < UNET_AXR(np)) continue;
        o[1] = 1.f;
        for (int k = 0; k < Np; k++) {
            double su = 0, sv = 0;
            for (int z = k * F; z < (k + 1) * F; z++) { su += r[4 + 2 * z]; sv += r[5 + 2 * z]; }
            o[2 + 2 * k] = (float)((su / F + 0.5) / F - 0.5); o[3 + 2 * k] = (float)((sv / F + 0.5) / F - 0.5);
        }
    }
    at_rays_plan(n, D, Hh, W, attn_ray_mult(), u->tb_axb, u->tb_rayh, u->tb_lensh, u->tb_adh, u->tb_cenh);
    nn_h2d((void *)u->tbr.ray, u->tb_rayh, (size_t)n * u->tb_Q * AT_RAY_F * sizeof(float));
    nn_h2d((void *)u->tbr.lens, u->tb_lensh, (size_t)n * u->tb_Q * sizeof(int));
    nn_h2d((void *)u->tbr.ad, u->tb_adh, (size_t)n * sizeof(int));
    nn_h2d(u->tb_cen, u->tb_cenh, (size_t)n * Pm * 4 * sizeof(float));
    if (u->cfg.attn_axis == 1) at_rays_setup(&u->tbr);
    else if (u->cfg.attn_pe == 3) at_scroll_pos(n, u->nrg, D, Hh, W, u->tbr.ad, u->tb_cen, u->tb_gpos);
    if (u->tb_dfr) at_dfm_frame(n, D, Hh, W, u->tbr.ad, u->tb_cen, u->tb_dfr);   /* deformable offsets in the normal frames */
    u->tb_axdirty = 0;
}
int unet_attn_rays(const unet *u, int *Q, int *Nr, double *coverage) {
    if (!u->tb_rb || u->cfg.attn_axis != 1) return -1;
    if (Q) *Q = u->tb_Q;
    if (Nr) *Nr = u->tb_Nr;
    if (coverage) {
        const size_t ng = (size_t)u->tbr.n * u->tbr.D * u->tbr.Hh * u->tbr.W; float *w = malloc(ng * sizeof *w); size_t c = 0;
        nn_sync(); nn_d2h(w, u->tbr.wsum, ng * sizeof *w);
        for (size_t i = 0; i < ng; i++) c += w[i] > 0.f;
        *coverage = (double)c / (double)ng; free(w);
    }
    return 0;
}
/* the bottleneck's raw output x -> PE -> blocks -> tb.a2 (the level's output; GroupNorm statistics of the new tensor) */
static void tb_fwd(unet *u) {
    const int C = u->tbd.C, nr = u->nrg;
    block *x = stage_out0(u, u->cfg.nlev - 1), *b = &u->tb;
    const shape5 ys = b->ys; const long Sg = (long)shape_spatial(ys), T = (long)ys.n * (Sg + nr);
    tb_geom(u);
    PROF(4, tb_to_f32(x->a2, ys, 0, u->tb_cs));
    at_cs_to_tok(u->tb_cs, TB_TOK(u, 0), P(u, u->reg_off), ys.n, C, Sg, nr);
    if (u->cfg.attn_mem) { tb_mem_load(u, ys.n); at_mem_add(TB_TOK(u, 0), ys.n, (int)(Sg + nr), u->cfg.attn_reg, u->cfg.attn_mem, C, P(u, u->mem_gate), P(u, u->mem_woff), u->tb_memin, u->tb_memf); }
    if (u->cfg.view_P) tb_view_fwd(u, ys.n, (int)(Sg + nr));
    if (u->cfg.attn_pe == 0) at_peg_fwd(TB_TOK(u, 0), TB_TOK(u, 1), P(u, u->peg_w), P(u, u->peg_b), &u->tbg, nr, C);
    else nn_d2d(TB_TOK(u, 1), TB_TOK(u, 0), (size_t)T * C * 4);
    for (int l = 1; l < u->tbd.dL; l++) {   /* deformable attention: the finer levels' outputs as LayerNorm'd token rows */
        const block *f = stage_out0(u, u->cfg.nlev - 1 - l); const shape5 fs = f->ys; const long Sf = (long)shape_spatial(fs);
        PROF(4, tb_to_f32(f->a2, fs, 0, u->tb_dfcs));
        at_cs_to_tok(u->tb_dfcs, u->tb_dfx[l], nullptr, fs.n, fs.c, Sf, 0);
        at_ln_plain(u->tb_dfx[l], u->tb_dfx[l], (long)fs.n * Sf, fs.c);
    }
    for (int k = 0; k < u->tba; k++) {   /* (application k runs block k % tbn: --attn-repeat) */
        at_ws *ws = u->tb_keep && u->fwd_train ? &u->tbk[k] : &u->tbw;   /* (keep: a training forward fills the application's own set) */
        tb_vr_set(u, ws, k);
        int r; PROF(0, r = at_block_fwd(&u->tbd, u->fw, &u->tbp[k % u->tbn], TB_TOK(u, k + 1), TB_TOK(u, k + 2), &u->tbg, ws));
        ws->kept = ws != &u->tbw;
        if (r) { fprintf(stderr, "unet: attention block forward failed\n"); abort(); }
    }
    if (u->cfg.attn_mem) at_mem_out(TB_TOK(u, u->tba + 1), ys.n, (int)(Sg + nr), u->cfg.attn_reg, u->cfg.attn_mem, C, u->tb_memout);
    at_tok_to_cs(TB_TOK(u, u->tba + 1), u->tb_cs, ys.n, C, Sg, nr);
    PROF(4, tb_from_f32(u->tb_cs, ys, 0, b->a2));
    const int G = G_of(u, C);
    if (!gn_frozen(u) && nn_gn_stats(b->a2, ys, G, 1e-5f, b->m2, b->r2)) { fprintf(stderr, "unet: attention stage GroupNorm statistics unsupported for this storage\n"); abort(); }
    if (b->s2) PROF(3, nn_gn_silu_apply(b->a2, ys, G, P(u, b->n2.gamma), P(u, b->n2.beta), b->m2, b->r2, b->s2));
}

static int l0st_want(const unet *u);   /* the level-0 slab stream (stage S3; after the backward) */
static int l0h_on(const unet *u);     /* its head per slab (stage S4) */
static const float *l0st_fwd_enc(unet *u, const float *x);
static void l0st_fwd_dec(unet *u);
const float *unet_forward(unet *u, const float *x, shape5 xs, int train) { return unet_forward_x(u, x, xs, train, 0); }
/* inference from the uint8 CT window (predict): needs a CT-stem build (unet_set_ct_input before the build, slabs on, MX-fp8 stem
   input): the network input is never materialised for the whole window */
const float *unet_forward_ct(unet *u, const uint8_t *ct, float mean, float isd, const float *dyo, const float *dxo, int axis, shape5 xs, int h16) {
    unet_build(u, xs, 0);
    if (!u->ct_stem) { fprintf(stderr, "unet_forward_ct: not a CT-stem build (unet_set_ct_input, UFSM_INFER_SLAB, an MX-fp8 or MX-fp4 stem input)\n"); abort(); }
    u->ctin.ct = ct; u->ctin.mean = mean; u->ctin.isd = isd; u->ctin.dyo = dyo; u->ctin.dxo = dxo; u->ctin.axis = axis; u->ctin.h16 = h16; u->ctin.on = 1;
    const float *lg = unet_forward_x(u, nullptr, xs, 0, h16);
    u->ctin.on = 0;
    return lg;
}
const float *unet_forward_x(unet *u, const void *xv, shape5 xs, int train, int x_h16) {
    const float *x = (const float *)xv;
    nn_set_nlev(u->cfg.nlev);   /* layer ids / names of the precision policy, manifests and profiles (a no-op unless the depth changed) */
    int div = 1 << (u->cfg.nlev - 1);
    if (xs.d % div || xs.h % div || xs.w % div) { fprintf(stderr, "unet: spatial size %dx%dx%d must be divisible by %d\n", xs.d, xs.h, xs.w, div); abort(); }
    if (!u->built || memcmp(&u->xs, &xs, sizeof xs) || (train && !u->train) || u->mode != UMODE()) build_acts(u, xs, train);
    if (sp_on(u) && (!nn_get_tf32() || !ABF)) { fprintf(stderr, "unet: the spatial split needs the tensor-core path with 16-bit or MX storage\n"); abort(); }
    sp_cfg(u);
    unet_apply_sparse24(u);
    if (u->cfg.scale_cond) {   /* stem bias b + c v for this forward (and the recompute passes of its backward) */
        const int c0 = u->enc[0].c1.cout;
        if (!u->stem_b) u->stem_b = nn_malloc((size_t)c0 * 4);
        nn_d2d(u->stem_b, P(u, u->enc[0].c1.b), (size_t)c0 * 4);
        nn_axpy(u->stem_b, u->sc_c, P(u, u->sc_w), (size_t)c0);
    }
    int L = u->cfg.nlev;
    const int *w = u->cfg.widths;
    const float *cur = x;
    if (x_h16 && !ABF) { fprintf(stderr, "unet_forward_x: a 16-bit input needs the 16-bit activation storage\n"); abort(); }
    /* Without MX gradients the four-channel input can stay 16-bit; the tap-packed FP8 kernel writes MX a1.
       MX gradients require a packed input for the available weight-gradient kernels (eight-wide MX rows). */
    const int xin_mx = input_mx();   /* MX-fp8 gradients require an MX x for the available weight-gradient kernels */
    if (u->l3b) {   /* the backward left the input in X1, the encoder's a1 in X2, the skip in X3, the coarse tensors at their backward homes */
        u->enc[0].a1 = u->l3x[1]; u->enc[0].a2 = u->l3x[2];
        nn_set_storage(u->l3x[1], u->l3xb[1], act_dt()); nn_set_storage(u->l3x[2], u->l3xb[2], act_dt());
        l3_homes(u, 1);
    }
    if (u->xin_shared) { nn_set_storage(u->xin, u->xin_bytes, unet_input_prec()); u->a2_stale = 1; }   /* dec[0].a2's buffer holds the input until the decoder */
    u->fwd_train = train;
    if (u->a0_share) nn_set_storage(u->enc[0].a2, act_bytes_of(u->enc[0].ys), act_dt());   /* a backward left the input in it */
    if (gn_frozen(u)) gn_frozen_load(u);
    l0_begin(u);
    u->l0st = l0st_want(u);
    if (u->lg_t1) nn_set_storage(u->lg_p, u->lg_bytes, act_dt());   /* the last head wrote fp32 logits there (unregistered) */
    if (u->xin_inf) nn_set_storage(u->xin, u->xin_bytes, unet_input_prec());   /* enc[0].a2's buffer holds the input until enc[0].c2 */
    if (u->ctin.on) cur = nullptr;   /* unet_forward_ct: the stem builds its input per slab from the CT */
    else if (ABF && !x_h16) { if (!u->xin) u->xin = dalloc_input(u, xs); nn_f32_to_act(x, xs, u->xin); cur = u->xin; }
    else if (x_h16 && act_mx8() && xin_mx) {
        if (!u->xin) u->xin = dalloc_input(u, xs);
        if (u->rad && u->cfg.half_res == 2) nn_input_rad_pack(x, xs, u->rad, u->xin);   /* --half-res pack: x is the full-resolution CT channel */
        else if (u->rad && u->l0rx) { u->l0x_ct = x; for (int k = 0; k < 16; k++) u->l0x_held[k] = -1; }   /* (stage S4: per slab in the stream, l0x_need) */
        else if (u->rad) nn_input_rad(x, xs, u->rad, u->xin);   /* --gpu-input: x is the CT channel, the radial channels from the records */
        else nn_h16_to_mx(x, xs, u->xin);
        cur = u->xin;
        if (getenv("UFSM_XIN_HASH")) {   /* diagnostic: FNV-1a of the converted input (the --gpu-input A/B compares these) */
            const size_t nb = nn_mx_bytes(xs, unet_input_prec()); uint8_t *h = malloc(nb); nn_sync(); nn_d2h(h, u->xin, nb);
            uint64_t f = 1469598103934665603ull; for (size_t k = 0; k < nb; k++) f = (f ^ h[k]) * 1099511628211ull;
            fprintf(stderr, "xin hash %016llx (%zu bytes)\n", (unsigned long long)f, nb); free(h);
        }
    }
    if (u->rad && !(x_h16 && act_mx8() && xin_mx && !u->ctin.on)) { fprintf(stderr, "unet_forward_x: radial input records need the 16-bit -> MX input conversion\n"); abort(); }
    u->l3_ct = u->l3b && train && u->rad && u->cfg.half_res != 2 && x_h16 && xin_mx && u->h_xin && l3_ct_on() && l3_ct_off(u) && l3_ct_bytes(u) <= u->xin_bytes
               && (const char *)x >= (const char *)u->l3x[2] && (const char *)x + l3_ct_bytes(u) <= (const char *)u->l3x[2] + u->l3xb[2];
    if (u->l3_ct) {   /* (l3b, GPU-built input) the CT to the host: enc[0]'s conv2 writes X2 after it */
        nn_offload_copy(u->h_xin, x, l3_ct_bytes(u), 1, u->ev_xin[0], u->ev_xin[1]);
        u->xin_out = 1;
    }
    size_t a1_rest = 0;   /* (a1_out_split: level 0's a1 bytes still to copy out after the encoders) */
    for (int i = 0; i < L; i++) {
        if (i == 1 && u->r_out) nn_stream_wait(0, u->ev_r[1]);   /* the coarse levels write R: its copy out done */
        if (i == 0 && u->l0st) { cur = l0st_fwd_enc(u, cur); continue; }   /* (level-0 slab stream: the stem, conv2, down0 per slab) */
        nn_set_layer(i); block_fwd(u, &u->enc[i], i, cur); l0_after(u, &u->enc[i]);
        if (i == 0 && u->p3) {   /* X's planes from zr on: to the host while down0 reads them (R takes the coarse levels after that) */
            nn_offload_copy(u->h_r, u->R, u->r_bytes, 1, u->ev_r[0], u->ev_r[1]);
            u->r_out = 1;
        }
        if (u->h_a1[i] && u->enc[i].a1_shared && train && !(i == 0 && l3_sched(u))) {   /* a1 is dead until the encoder's backward: to the host while
                                                                the coarser levels run. level 0's in two halves, the second after the encoders (a1_out_split; l3_sched: block_fwd) */
            const size_t ab = act_bytes_of(u->enc[i].ys), h = i == 0 && a1_out_split() && L > 2 ? (ab / 2) & ~(size_t)255 : ab;
            nn_offload_copy(u->h_a1[i], u->enc[i].a1, h, 1, u->ev_a1[i][0], u->ev_a1[i][1]);
            u->a1_out[i] = 1; if (i == 0) a1_rest = ab - h;
        }
        if (i == 0 && l3_sched(u) && u->a0_share && train) {   /* (l3_sched) the skip out next (dec[0]'s conv2 waits for it) */
            nn_offload_copy(u->h_a2, u->enc[0].a2, act_bytes_of(u->enc[0].ys), 1, u->ev_a2[0], u->ev_a2[1]);
            u->a2_out = 1;
        }
        if (u->dno && train && i > 0 && !a1_out_split()) {   /* the down-conv output this block read: to the host until the block's backward
                                                                (a1_out_split: after the encoders) */
            shape5 ds = u->ls[i]; ds.c = w[i - 1];
            nn_offload_copy(u->h_dn[i - 1], u->downo[i - 1], act_bytes_of(ds), 1, u->ev_dn[i - 1][0], u->ev_dn[i - 1][1]);
            u->dn_out = 1;
        }
        for (int k = 0; k < u->cfg.enc_blocks[i]; k++) { nn_set_layer(i); nn_set_block(1 + k); rblock_fwd(u, &u->rb[i][k], i, k ? &u->rb[i][k - 1] : &u->enc[i]); nn_set_block(0); }
        if (i == L - 1 && u->tbn) tb_fwd(u);
        block *so = stage_out(u, i);
        cur = so->s2;
        if (i < L - 1) {
            nn_set_layer(L + i);
            if (i == 0 && u->l3b && u->xin_out) nn_stream_wait(0, u->ev_xin[1]);   /* (l3b: down0's output over the input in X3: its copy out done) */
            if (recompute()) { nn_gn_t g = gn_out(u, so); int r; PROF(0, r = nn_conv3d_fwd_x(so->a2, &g, nullptr, nullptr, 0, 0, so->ys, P(u, u->down[i].w), P(u, u->down[i].b), w[i], 3, 2, u->downo[i], 0, 0.f, nullptr, nullptr)); if (r) rc_fail("down conv"); }
            else PROF(0, nn_conv3d_fwd(cur, so->ys, P(u, u->down[i].w), P(u, u->down[i].b), w[i], 3, 2, u->downo[i]));
            { shape5 ds = u->ls[i + 1]; ds.c = w[i]; sp_halo(u, u->downo[i], ds, 0); }
            if (u->cfg.down_norm) {   /* GroupNorm + SiLU after the down conv: stats now, the transform in the next conv1's staging */
                shape5 ds = u->ls[i + 1]; ds.c = w[i];
                const int G = G_of(u, w[i]);
                block *nb = &u->enc[i + 1];
                nb->inx = u->downo[i]; nb->ing = &u->dn[i]; nb->inm = u->dm[i]; nb->inr = u->dr[i];
                if (nn_get_tf32()) { if (!gn_frozen(u) && nn_gn_stats(u->downo[i], ds, G, 1e-5f, u->dm[i], u->dr[i])) { fprintf(stderr, "unet: down_norm GroupNorm statistics unsupported for this storage\n"); abort(); } }
                else { PROF(3, nn_gn_fwd_silu(u->downo[i], ds, G, 1e-5f, P(u, u->dn[i].gamma), P(u, u->dn[i].beta), u->downs[i], u->dm[i], u->dr[i])); cur = u->downs[i]; continue; }
            }
            { shape5 ds = u->ls[i + 1]; ds.c = w[i]; FQ(u->downo[i], ds); }
            cur = u->downo[i];
        }
    }
    if (a1_rest) {   /* the rest of level 0's a1 (dec[0]'s conv1 waits for it, long after the coarse decoders' waits for their own a1) */
        const size_t ab = act_bytes_of(u->enc[0].ys), h = ab - a1_rest;
        nn_offload_copy((char *)u->h_a1[0] + h, (const char *)u->enc[0].a1 + h, a1_rest, 1, u->ev_a1[0][0], u->ev_a1[0][1]);
    }
    if (u->dno && train && a1_out_split()) for (int i = 1; i < L; i++) {   /* the down-conv outputs, read by their encoders' conv1: to the host until
                                                                           the blocks' backward (needed out by dec[0]'s conv2 / the head) */
        shape5 ds = u->ls[i]; ds.c = w[i - 1];
        nn_offload_copy(u->h_dn[i - 1], u->downo[i - 1], act_bytes_of(ds), 1, u->ev_dn[i - 1][0], u->ev_dn[i - 1][1]);
        u->dn_out = 1;
    }
    if (u->a0_share && train && !l3_sched(u)) {   /* the skip: a copy on the host (its buffer takes the logit gradient after the forward); queued after
                                     the encoders' a1, which the decoders wait for (needed only at the forward's end) */
        nn_offload_copy(u->h_a2, u->enc[0].a2, act_bytes_of(u->enc[0].ys), 1, u->ev_a2[0], u->ev_a2[1]);
        u->a2_out = 1;
    }
    for (int i = L - 2; i >= 0; i--) {
        const int uw = UPW(i + 1);   /* channels upsampled into this level */
        shape5 src = u->ls[i + 1]; src.c = uw;
        shape5 li = u->ls[i];
        u->dec[i].xb = u->dec[i].x2b = nullptr;
        if (recompute()) {   /* nothing materialised: dec conv1 stages up2(silu(gn(a2))) of the coarse block and silu(gn(a2)) of the skip */
            u->dec[i].xb = i == L - 2 ? (const void *)stage_out(u, L - 1) : (const void *)&u->dec[i + 1];
            u->dec[i].x2b = stage_out(u, i); u->dec[i].in2 = nullptr; u->dec[i].c_split = uw;
        } else if (nn_get_tf32()) {   /* cat holds only the upsampled part; the skip is read in place by the split conv */
            PROF(5, nn_up2_fwd_into(cur, src, u->cat[i], uw, 0));
            { shape5 us = li; us.c = uw; sp_halo(u, u->cat[i], us, 0); }
            u->dec[i].in2 = stage_out(u, i)->s2; u->dec[i].c_split = uw;
        } else {
            PROF(5, nn_up2_fwd_into(cur, src, u->cat[i], uw + w[i], 0));
            size_t S = shape_spatial(li);
            for (int n = 0; n < li.n; n++)
                nn_d2d(u->cat[i] + ((size_t)n * (uw + w[i]) + uw) * S, stage_out(u, i)->s2 + (size_t)n * w[i] * S, (size_t)w[i] * S * 4);
            u->dec[i].in2 = nullptr;
        }
        if (u->a1_out[i]) nn_stream_wait(0, u->ev_a1[i][1]);   /* the decoder's conv1 overwrites the shared buffer */
        if (i == 0 && u->xin_shared) nn_set_storage(u->dec[0].a2, act_bytes_of(u->dec[0].ys), act_dt());   /* its conv2 writes a2 over the input
                                                                                                                (block_fwd waits for the copies out) */
        if (i == 0 && (u->xin_shared || u->a0_share)) u->a2_stale = 0;
        nn_set_layer(3 * L - 3 - i);
        if (i == 0 && u->l0st) { u->dec[0].in = u->cat[0]; l0st_fwd_dec(u); }   /* (level-0 slab stream) */
        else block_fwd(u, &u->dec[i], i, u->cat[i]);
        l0_after(u, &u->dec[i]);
        if (u->sko && train && i > 0) {   /* its skip: to the host until the decoder's backward */
            nn_offload_copy(u->h_sk[i], stage_out(u, i)->a2, act_bytes_of(u->enc[i].ys), 1, u->ev_sk[i][0], u->ev_sk[i][1]);
            u->sk_out = 1;
        }
        cur = u->dec[i].s2;
    }
    if (train && u->train && u->cfg.ds_levels) {   /* deep supervision: each supervised decoder level's own 1^3 head, from its output */
        if (sp_on(u)) { fprintf(stderr, "unet: deep supervision with the spatial split is not supported\n"); abort(); }
        nn_set_head_out_h16(0);
        for (int j = 1; j <= u->cfg.ds_levels; j++) {
            const block *d = &u->dec[j]; const float *dw = P(u, u->ds[j].w), *db = P(u, u->ds[j].b);
            if (recompute()) { nn_gn_t g = gn_out(u, d); int r; PROF(0, r = nn_conv3d_fwd_x(d->a2, &g, nullptr, nullptr, 0, 0, d->ys, dw, db, u->cfg.cout, 1, 1, u->ds_lg[j], 0, 0.f, nullptr, nullptr)); if (r) rc_fail("deep-supervision head"); }
            else PROF(0, nn_conv3d_fwd(d->s2, d->ys, dw, db, u->cfg.cout, 1, 1, u->ds_lg[j]));
        }
    }
    nn_set_layer(3 * L - 2);
    if (!(u->head_ext && !u->logits) && !l0h_on(u)) {   /* (the stream head ran per slab in F2; head slabs: the trainer runs the head per z-slab, unet_head_fwd_z; a training build has no
                                            whole-window logits, so its validation forwards (train 0, no rebuild) skip the head too) */
        if (u->lg_t1 && !u->lg_x) nn_set_storage(u->logits, u->lg_bytes, 0);   /* plain fp32 plane-major logits over dec[0].a1 (or the level-1 buffer) */
        if (l3_sched(u)) {   /* (the coarse forward homes and the input in X3 copied out; without l3_sched dec[0]'s conv2 waited) */
            if (u->xin_shared && u->xin_out) nn_stream_wait(0, u->ev_xin[1]);
            if (u->dno && u->dn_out) for (int k = 0; k < L - 1; k++) nn_stream_wait(0, u->ev_dn[k][1]);
            if (u->sko && u->sk_out) for (int k = 1; k < L - 1; k++) nn_stream_wait(0, u->ev_sk[k][1]);
        }
        if (u->l3b) nn_storage_forget(u->l3x[3]);   /* fp16 logits over X3 (the coarse forward homes copied out: dec[0]'s conv2 / above waited) */
        nn_set_head_out_h16(u->logits_h16);
        const int hc0 = !u->train && u->head_nc ? u->head_c0 : 0, hnc = !u->train && u->head_nc ? u->head_nc : u->cfg.cout;   /* rows of the 1^3 head */
        const float *hw = P(u, u->head.w) + (size_t)hc0 * w[0], *hb = P(u, u->head.b) + hc0;
        if (u->lg_x) {   /* the logits over X (dec[0].a2): per z-slab into the slab temp, then to their planes at X's start (data planes
                            [z0 / 4, (z0 + s) / 4) at 8 B per 16-channel MX-fp4 voxel: read by this slab or before) */
            nn_gn_t g = gn_out(u, &u->dec[0]);
            const size_t hwp = (size_t)u->ls[0].h * u->ls[0].w;
            const int D = u->ls[0].d, s = u->slab_s;
            void *tmp = u->p3 ? (void *)((char *)u->Yp + 256) : (void *)u->slab_tmp;   /* (p3: Y, free once dec[0]'s conv1 has upsampled dec[1]) */
            for (int z0 = 0; z0 < D; z0 += s) {
                const int sk = D - z0 < s ? D - z0 : s;
                int r; PROF(0, r = nn_head_fwd_z(u->dec[0].a2, &g, u->dec[0].ys, hw, hb, hnc, tmp, (size_t)z0 * hwp, (size_t)sk * hwp));
                if (r) rc_fail("head (slabs over X)");
                nn_d2d_async((char *)u->logits + (size_t)z0 * hwp * 2, tmp, (size_t)sk * hwp * 2);
            }
            nn_set_storage(u->logits, u->lg_bytes, 0);   /* plain fp16 planes for the readers (re-registered at the next forward) */
        }
        else if (recompute()) { nn_gn_t g = gn_out(u, &u->dec[0]); int r; PROF(0, r = nn_conv3d_fwd_x(u->dec[0].a2, &g, nullptr, nullptr, 0, 0, u->dec[0].ys, hw, hb, hnc, 1, 1, u->logits, 0, 0.f, nullptr, nullptr)); if (r) rc_fail("head"); }
        else PROF(0, nn_conv3d_fwd(cur, u->dec[0].ys, hw, hb, hnc, 1, 1, u->logits));
        nn_set_head_out_h16(0);
    }
    if (u->a0_share && train && u->a2_out) nn_stream_wait(0, u->ev_a2[1]);   /* the loss writes the logit gradient over the skip's buffer next */
    if (ufsm_env_on("UFSM_DEBUG") && !ABF && !recompute() && u->logits) {
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

/* head slabs (unet_set_head_ext): the head on output planes [z0, z1) of the window, from dec[0]'s output as the whole-window head
   reads it (recompute: a2 with its GroupNorm + SiLU on the fly; else s2). fwd: fp16 logits [cout][(z1 - z0) h w] into y16 (after the
   forward, or inside the head callback); bwd (inside the callback only): from the fp16 logit gradient in the same layout, weight /
   bias gradients accumulated, the head input's gradient rows of those planes into gout[0]; first: the backward's first slab */
static int head_src(unet *u, const float **x, nn_gn_t *g) {
    block *d0 = &u->dec[0];
    if (recompute()) { *g = gn_out(u, d0); *x = d0->a2; return 1; }
    *x = d0->s2; return 0;
}
int unet_head_fwd_z(unet *u, int z0, int z1, void *y16) {
    const float *x; nn_gn_t g; const int gn = head_src(u, &x, &g);
    const size_t hw = (size_t)u->ls[0].h * u->ls[0].w;
    nn_set_layer(3 * u->cfg.nlev - 2);
    return nn_head_fwd_z(x, gn ? &g : nullptr, u->dec[0].ys, P(u, u->head.w), P(u, u->head.b), u->cfg.cout, y16, (size_t)z0 * hw, (size_t)(z1 - z0) * hw);
}
int unet_head_bwd_z(unet *u, int z0, int z1, const void *gy16, int first) {
    const float *x; nn_gn_t g; const int gn = head_src(u, &x, &g);
    const size_t hw = (size_t)u->ls[0].h * u->ls[0].w;
    return nn_head_bwd_z(x, gn ? &g : nullptr, u->dec[0].ys, gy16, u->cfg.cout, P(u, u->head.w), u->g + u->head.w, u->g + u->head.b, u->gout[0],
                         (size_t)z0 * hw, (size_t)(z1 - z0) * hw, first);
}

/* ---- backward ---- */
/* gy: grad wrt block output (s2). Returns gB of the level holding grad wrt block input. */
/* pre: gy is already d/d a2 (its normalisation was backpropagated by the next residual block's input GroupNorm); gid: added to d/d a2
   (the residual identity gradient, fp32 path); save: receives d/d a2 (the identity gradient for the block before) */
/* the block's GroupNorm backwards (k: its norm 1 / 2); slab: over the pseudo-sample shape with mean 0 / rstd 1, then the slabs' rstd */
static void gn_silu_bwd_b(unet *u, const block *b, int k, const float *x, shape5 s, int G, const float *gamma, const float *beta, const float *mean,
                          const float *rstd, const float *gy, float *gx, float *gg, float *gb, float *scratch) {
    if (l0_slab_on(u, b) && l0s_kern(u)) {   /* kernel path: GroupNorm backward per (sample, slab) on the raw tensor, the true statistics */
        const int ms = l0s_nslab(u, s.d) > 1;
        if (ms && (!nn_storage(x) || !nn_storage(gy) || !nn_storage(gx))) { fprintf(stderr, "unet: --l0-norm slab (kernel path): MX tensors and gradients only\n"); abort(); }
        nn_gn_silu_bwd(x, ms ? l0s_shape(u, b) : s, G, gamma, beta, mean, rstd, gy, gx, gg, gb, scratch);
        return;
    }
    if (!l0_slab_on(u, b) || !nn_storage(x)) { if (l0_slab_on(u, b)) l0s_check(u, x); nn_gn_silu_bwd(x, s, G, gamma, beta, mean, rstd, gy, gx, gg, gb, scratch); return; }
    const shape5 ps = l0s_shape(u, b);
    if (!nn_storage(gx)) { fprintf(stderr, "unet: --l0-norm slab: MX gradients only\n"); abort(); }
    nn_gn_silu_bwd(x, ps, G, gamma, beta, u->l0s_st + 8 * u->l0s_nng, u->l0s_st + 9 * u->l0s_nng, gy, gx, gg, gb, scratch);
    nn_l0_affine(gx, ps, G, nullptr, l0s_stat(u, b, k, 1), (u->l0s_seed[2 * l0_blk(u, b) + k - 1] * 0x2545f491u) | 1u);
}
static void gn_bwd_b(unet *u, const block *b, int k, const float *x, shape5 s, int G, const float *gamma, const float *mean, const float *rstd,
                     const float *gy, float *gx, float *gg, float *gb, float *scratch) {
    if (!l0_slab_on(u, b)) { nn_gn_bwd(x, s, G, gamma, mean, rstd, gy, gx, gg, gb, scratch); return; }
    const shape5 ps = l0s_shape(u, b); const int S = u->cfg.l0_slab; const size_t n = shape_numel(s);
    float *t0 = u->l0s_tmp, *t1 = t0 + n, *t2 = t1 + n, *one = u->l0s_st + 10 * u->l0s_nng, *zero = one + u->cfg.widths[0];
    nn_slab_perm(t0, x, s, S, 0); nn_slab_perm(t1, gy, s, S, 0);
    nn_gn_bwd(t0, ps, G, gamma, u->l0s_st + 8 * u->l0s_nng, u->l0s_st + 9 * u->l0s_nng, t1, t2, gg, gb, scratch);
    nn_gn_apply(t2, ps, G, one, zero, u->l0s_st + 8 * u->l0s_nng, l0s_stat(u, (block *)b, k, 1), t2);   /* d/dx = rstd(slab) d/dx' */
    nn_slab_perm(gx, t2, s, S, 1);
}
/* level k's stage output (the stored block output silu(gn(a2)), s2) after a training-mode forward, and its shape; nullptr when the
   build does not keep it (recompute / lean modes) */
const float *unet_stage_out(unet *u, int k, shape5 *s) {
    if (!u->built || k < 0 || k >= u->cfg.nlev) return nullptr;
    const block *b = stage_out(u, k); if (s) *s = b->ys;
    return b->s2;
}
/* UFSM_VIRTUAL_TOKENS=1 (dino-train with --attn-blocks 0): a net without the attention stage reports a zero token buffer of the bottleneck's
   shape (attn_reg + grid rows of its width), so the CLS paths run (on zeros) and their token gradients are dropped */
static float *g_vtok; static size_t g_vtok_n;
const float *unet_tokens(const unet *u, int *nseq, int *S, int *C, int *nreg) {
    if (!u->tbn && u->built && getenv("UFSM_VIRTUAL_TOKENS")) {
        const block *b = stage_out((unet *)u, u->cfg.nlev - 1); const size_t Sv = (size_t)b->ys.d * b->ys.h * b->ys.w + (size_t)u->cfg.attn_reg, n = Sv * b->ys.c;
        if (n > g_vtok_n) { if (g_vtok) nn_free(g_vtok); g_vtok = nn_malloc(n * 4); nn_zero(g_vtok, n * 4); g_vtok_n = n; }
        if (nseq) *nseq = 1;
        if (S) *S = (int)Sv;
        if (C) *C = b->ys.c;
        if (nreg) *nreg = u->cfg.attn_reg;
        return g_vtok;
    }
    if (!u->tbn || !u->built || !u->tb_tok) return nullptr;
    if (nseq) { *nseq = u->tbg.nseq; }
    if (S) { *S = u->tbg.S; }
    if (C) { *C = u->tbd.C; }
    if (nreg) { *nreg = u->nrg; }
    return TB_TOK(u, u->tba + 1);
}
void unet_set_token_grad(unet *u, const float *g) { u->tb_xg = g; }
/* d/d (the level output silu(gn(tb.a2))) -> d/d (the bottleneck's raw output x), in the gradient storage (block_bwd's pre = 1 input) */
static float *tb_bwd(unet *u, const float *gy) {
    const int C = u->tbd.C, nr = u->nrg, G = G_of(u, C);
    block *b = &u->tb; float *g = u->g;
    const shape5 ys = b->ys; const long Sg = (long)shape_spatial(ys);
    PROF(3, gn_silu_bwd_b(u, b, 2, b->a2, ys, G, P(u, b->n2.gamma), P(u, b->n2.beta), b->m2, b->r2, gy, u->tb_g, g + b->n2.gamma, g + b->n2.beta, u->gn_scratch));
    PROF(4, tb_to_f32(u->tb_g, ys, 1, u->tb_cs));
    at_cs_to_tok(u->tb_cs, u->tb_dt, u->tb_zero, ys.n, C, Sg, nr);
    if (u->tb_xg) nn_axpy(u->tb_dt, 1.f, u->tb_xg, (size_t)ys.n * (Sg + nr) * C);   /* (unet_set_token_grad: e.g. a pooled CLS loss) */
    if (u->tb_vr) nn_zero(u->tb_vr + (size_t)u->tbg.nseq * u->tbg.S * C, (size_t)u->tbg.nseq * u->tbg.S * C * 4);   /* (dv1) */
    for (int k = u->tba - 1; k >= 0; k--) {
        at_ws *ws = u->tb_keep && u->tbk[k].kept ? &u->tbk[k] : &u->tbw;   /* (kept: no recompute) */
        tb_vr_set(u, ws, k);
        int r; PROF(1, r = at_block_bwd(&u->tbd, u->fw, g, &u->tbp[k % u->tbn], TB_TOK(u, k + 1), u->tb_dt, &u->tbg, ws));
        if (r) { fprintf(stderr, "unet: attention block backward failed\n"); abort(); }
    }
    float *d = u->tb_dt;
    if (u->cfg.attn_pe == 0) { at_peg_bwd(TB_TOK(u, 0), u->tb_dt, u->tb_dt2, P(u, u->peg_w), g + u->peg_w, g + u->peg_b, &u->tbg, nr, C); d = u->tb_dt2; }
    at_reg_grad(d, g + u->reg_off, ys.n, C, Sg, nr);
    if (u->cfg.attn_mem) at_mem_grad(d, ys.n, (int)(Sg + nr), u->cfg.attn_reg, u->cfg.attn_mem, C, u->tb_memin, u->tb_memf, g + u->mem_gate, g + u->mem_woff);
    if (u->cfg.view_P) {   /* the view rows: gather their gradient, then the embedding's weight gradient GE^T Xp */
        const int k = u->cfg.view_patch, k3 = k * k * k, R = ys.n * u->tb_nv;
        at_view_gather(d, ys.n, (int)(Sg + nr), u->cfg.attn_reg + u->cfg.attn_mem, u->tb_nv, C, u->tb_ve, g + u->view_b);
        if (at_gemm(1, 0, C, k3, R, u->tb_ve, u->tb_vxp, g + u->view_w, 1.f)) { fprintf(stderr, "unet: view embedding gradient GEMM failed\n"); abort(); }
    }
    at_tok_to_cs(d, u->tb_cs, ys.n, C, Sg, nr);
    PROF(4, tb_from_f32(u->tb_cs, ys, 1, u->tb_gx));
    return u->tb_gx;
}
static float *block_bwd_(unet *u, block *b, int level, const float *gy, float *gx2, int pre, const float *gid, float *save);
static float *block_bwd(unet *u, block *b, int level, const float *gy, float *gx2, int pre, const float *gid, float *save) {
    nn_set_gn_const(l0_fixed(u, b) && u->cfg.l0_norm != 4);   /* constant level-0 statistics: no mean / variance terms in the block's GroupNorm
                                                              backwards (slab: the slabs' terms, gn_silu_bwd_b) */
    float *r = block_bwd_(u, b, level, gy, gx2, pre, gid, save);
    nn_set_gn_const(0);
    return r;
}
static float *block_bwd_(unet *u, block *b, int level, const float *gy, float *gx2, int pre, const float *gid, float *save) {
    int G = G_of(u, b->c1.cout);
    size_t n = shape_numel(b->ys);
    float *A = u->gA[level], *B = u->nob || pre ? (float *)gy : u->gB[level], *t1 = u->t1[level], *t2 = u->t2[level];   /* no B: in place on gy */
    const size_t gbytes = nn_storage(B) ? nn_mx_bytes(b->ys, nn_storage(B)) : n * (GBF ? 2 : 4);
    float *g = u->g;
    const int live = u->last_a1_live && b == &u->dec[0];
    u->last_a1_live = 0;   /* repeated backward without a new forward must rebuild the shared buffer */
    const int lvl_off = b->a1_shared && level < UNET_MAXLEV && u->a1_out[level];
    const int l3ro = b == &u->enc[0] && u->l3b && l3_reorder() && nn_get_tf32() && !b->single && !pre;   /* (its a1 waited for below) */
    if (lvl_off) { if (!l3ro) nn_stream_wait(0, u->ev_a1[level][3]); }   /* offloaded encoder a1: back from the host */
    else if (u->l3b && b->a1_shared && (b == &u->enc[0] || !l3_sched(u))) rc_fail("three level-0 buffers: an encoder a1 without its host copy");   /* (a re-run
                                                                                                                 would race the copies; l3_sched: the coarse levels re-run) */
    else if (!b->single && ((recompute_a1() && !b->keep_a1 && !live) || b->a1_shared || (b->a1_lent && u->sea_stale))) {   /* a1 was overwritten by later blocks: re-run conv1 (same kernel and precision as the forward) */
        nn_set_conv(0);
        int r;
        if (b->xb) r = dec_conv1(u, b, level, b->a1, G, nullptr, nullptr, nullptr, nullptr, nullptr);
        else { nn_gn_t gi = gn_in(u, b); PROF(0, r = nn_conv3d_fwd_x(b->in, gi.G ? &gi : nullptr, nullptr, nullptr, 0, 0, b->xs, P(u, b->c1.w), B1(u, b), b->c1.cout, 3, 1, b->a1, G, 0.f, nullptr, nullptr)); }
        if (r) rc_fail("conv1 recompute");
        l0_slab_fwd(u, b, 1, b->a1, 0);   /* (slab: normalised as in the forward) */
        FQ(b->a1, b->ys);
        sp_halo(u, b->a1, b->ys, 0);
    }
    nn_set_conv(1);   /* precision-policy tag: c2 first, then c1 */
    if (nn_get_tf32() && b->single) {   /* one conv: A = d/d a2 (conv1's output) */
        if (pre || gid || save) { fprintf(stderr, "unet: a one-conv decoder block takes no residual gradient\n"); abort(); }
        sp_zero(u, gy, b->ys, 1);
        PROF(3, gn_silu_bwd_b(u, b, 2, b->a2, b->ys, G, P(u, b->n2.gamma), P(u, b->n2.beta), b->m2, b->r2, gy, A, g + b->n2.gamma, g + b->n2.beta, u->gn_scratch));
        FQG(A, b->ys);
        sp_zero(u, A, b->ys, 1);
        sp_begin(u, A, b->ys, 1);
    } else if (!nn_get_tf32() && b->single) {
        PROF(3, nn_gn_apply(b->a2, b->ys, G, P(u, b->n2.gamma), P(u, b->n2.beta), b->m2, b->r2, t1));            /* t1 = g2 */
        PROF(4, nn_silu_bwd(t1, gy, n, B));                                                                        /* B = d/d g2 */
        PROF(3, gn_bwd_b(u, b, 2, b->a2, b->ys, G, P(u, b->n2.gamma), b->m2, b->r2, B, A, g + b->n2.gamma, g + b->n2.beta, u->gn_scratch));   /* A = d/d a2 */
    } else if (nn_get_tf32()) {   /* fused: no materialised gn / silu activations */
        sp_zero(u, gy, b->ys, 1);   /* split: only owned voxels enter the reductions below */
        if (!pre) PROF(3, gn_silu_bwd_b(u, b, 2, b->a2, b->ys, G, P(u, b->n2.gamma), P(u, b->n2.beta), b->m2, b->r2, gy, B, g + b->n2.gamma, g + b->n2.beta, u->gn_scratch));   /* B = d/d a2 */
        if (gid) nn_add_into(B, gid, b->ys, GBF);
        if (save) nn_d2d(save, B, gbytes);
        if (b == &u->enc[0] && u->a0_share && u->xin_shared && !u->l3b) {   /* a2 read for the last time: the input into its buffer for conv1's weight gradient */
            nn_set_storage(b->a2, u->xin_bytes, unet_input_prec());
            nn_offload_copy(b->a2, u->h_xin, u->xin_bytes, 0, u->ev_xin[2], u->ev_xin[3]);
            b->in = b->a2;
        }
        FQG(B, b->ys);
        sp_zero(u, B, b->ys, 1);
        unsigned c0 = 0, c1 = 0, kw = 0;
        if (l3ro) {   /* l3b: the backward-data (no a1) first while the encoder's a1 comes back into X2, with the seeds of the usual order: the
                         weight gradient's (one draw when SR is on: launch_bwd_w_tc) skipped, then replayed for it */
            c0 = nn_sr_ctr(); kw = nn_sr_on() ? 1u : 0u;
            nn_sr_set_ctr(c0 + kw);
            PROF(1, nn_conv3d_bwd_data(B, b->ys, P(u, b->c2.w), b->ys, 3, 1, A, u->conv_scratch));                          /* A = d/d s1 */
            c1 = nn_sr_ctr();
            nn_sr_set_ctr(c0);
            if (lvl_off) nn_stream_wait(0, u->ev_a1[level][3]);
        }
        sp_begin(u, B, b->ys, 1);   /* split: the halo crosses during the weight gradient (which reads it as zero) */
        PROF(2, nn_conv3d_bwd_weight_gn(b->a1, b->ys, G, P(u, b->n1.gamma), P(u, b->n1.beta), b->m1, b->r1, B, b->ys, g + b->c2.w, g + b->c2.b));
        sp_end(u, B, b->ys, 1);
        if (l3ro) {
            if (nn_sr_ctr() != c0 + kw) rc_fail("three level-0 buffers: enc[0]'s conv2 weight gradient drew other seeds than predicted (UFSM_L0_3BUF_REORDER=0)");
            nn_sr_set_ctr(c1);
        } else PROF(1, nn_conv3d_bwd_data(B, b->ys, P(u, b->c2.w), b->ys, 3, 1, A, u->conv_scratch));                       /* A = d/d s1 */
        if (b == &u->dec[0] && u->l3b) {   /* d/d a2 (X2) read for the last time: the skip back but for the coarse temp's bytes during GN1's backward
                                             and the up transient (dec_conv1 brings those) */
            nn_set_storage(u->enc[0].a2, act_bytes_of(u->enc[0].ys), act_dt());
            l3_skip_copy(u, 0, l3_skip_lo(u), 0);
            l3_skip_copy(u, l3_skip_hi(u), act_bytes_of(u->enc[0].ys), 0);
        }
        if (b == &u->enc[0] && u->l3b) {   /* B (X1) read for the last time: the input back into it for the stem's weight gradient (l3_ct: its CT
                                             into X1's end, the input rebuilt below) */
            nn_set_storage(u->l3x[1], u->xin_bytes, unet_input_prec());
            if (u->l3_ct) nn_offload_copy((char *)u->l3x[1] + l3_ct_off(u), u->h_xin, l3_ct_bytes(u), 0, u->ev_xin[2], u->ev_xin[3]);
            else nn_offload_copy(u->l3x[1], u->h_xin, u->xin_bytes, 0, u->ev_xin[2], u->ev_xin[3]);
            b->in = u->l3x[1];
        }
        FQG(A, b->ys);
        sp_zero(u, A, b->ys, 1);
        PROF(3, gn_silu_bwd_b(u, b, 1, b->a1, b->ys, G, P(u, b->n1.gamma), P(u, b->n1.beta), b->m1, b->r1, A, A, g + b->n1.gamma, g + b->n1.beta, u->gn_scratch));    /* A = d/d a1 (in place) */
        FQG(A, b->ys);
        sp_zero(u, A, b->ys, 1);
        if (b != &u->enc[0]) sp_begin(u, A, b->ys, 1);   /* for conv1's backward-data (enc0 has none) */
    } else {
    PROF(3, nn_gn_apply(b->a2, b->ys, G, P(u, b->n2.gamma), P(u, b->n2.beta), b->m2, b->r2, t1));            /* t1 = g2 */
    PROF(4, nn_silu_bwd(t1, gy, n, A));                                                                        /* A = d/d g2 */
    PROF(3, gn_bwd_b(u, b, 2, b->a2, b->ys, G, P(u, b->n2.gamma), b->m2, b->r2, A, B, g + b->n2.gamma, g + b->n2.beta, u->gn_scratch));   /* B = d/d a2 */
    if (gid) nn_add_into(B, gid, b->ys, GBF);
    if (save) nn_d2d(save, B, gbytes);
    PROF(3, nn_gn_apply(b->a1, b->ys, G, P(u, b->n1.gamma), P(u, b->n1.beta), b->m1, b->r1, t1));            /* t1 = g1 */
    PROF(4, nn_silu_fwd(t1, n, t2));                                                                           /* t2 = s1 */
    PROF(2, nn_conv3d_bwd_weight(t2, b->ys, B, b->ys, 3, 1, g + b->c2.w, g + b->c2.b));
    PROF(1, nn_conv3d_bwd_data(B, b->ys, P(u, b->c2.w), b->ys, 3, 1, A, u->conv_scratch));                    /* A = d/d s1 */
    PROF(4, nn_silu_bwd(t1, A, n, B));                                                                         /* B = d/d g1 */
    PROF(3, gn_bwd_b(u, b, 1, b->a1, b->ys, G, P(u, b->n1.gamma), b->m1, b->r1, B, A, g + b->n1.gamma, g + b->n1.beta, u->gn_scratch));   /* A = d/d a1 */
    }
    nn_set_conv(0);
    if (b->xb) {   /* recompute: weight gradient with the upsampled part rebuilt into B (free until backward-data below) */
        if (b == &u->dec[0] && u->a0_share) { if (u->l3b) u->l3_skip_in = 1; else nn_stream_wait(0, u->ev_a2[3]); }   /* the skip back from the host
                                                                                                 (l3b: dec_conv1 brings it into X2 after its up transient) */
        else if (u->sko && level > 0 && b == &u->dec[level]) nn_stream_wait(0, u->ev_sk[level][3]);   /* a coarse skip back from the host */
        u->rc_ok = 1;   /* gy / B consumed: the level's gradient buffer is free until the skip gradient below */
        if (dec_conv1(u, b, level, nullptr, 0, nullptr, nullptr, A, g + b->c1.w, g + b->c1.b)) rc_fail("decoder conv1 weight gradient");
        if (u->l3_skip_in) rc_fail("three level-0 buffers: a dec[0] conv1 weight gradient without its up transient");
        u->rc_ok = 0;
        sp_end(u, A, b->ys, 1);
        if (chunk_up()) {   /* skip gradient, then the up-part gradient in chunks through B, upsample-backwarded into gout[level + 1] */
            const block *xb = (const block *)b->xb;
            const size_t per_c = shape_numel(b->ys) / b->ys.c * (GBF ? 2 : 4);
            const size_t cap = u->gB_cap[level];
            int cs = (int)(cap / per_c); if (cs > b->c_split) cs = b->c_split;
            if (grad_mx8()) {   /* MX chunks: the smallest legal chunk (mx_chunk_min) or whole 32-channel blocks that fit B */
                shape5 t = b->xs; cs = mx_chunk_min(b->c_split);
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
            float *Bc = u->l3b && b == &u->dec[0] ? u->gB[0] : B;   /* (l3b: gy was dec[0]'s a2; the chunks through X1) */
            for (int c0 = 0; c0 < b->c_split; c0 += cs) {
                const int nc = b->c_split - c0 < cs ? b->c_split - c0 : cs;
                PROF(1, r = nn_conv3d_bwd_data_range(A, b->ys, P(u, b->c1.w), b->xs, c0, nc, Bc, u->conv_scratch));
                if (r) rc_fail("up-part backward-data");
                { shape5 fs = b->xs; fs.c = nc; sp_halo(u, Bc, fs, 1); }
                shape5 cs5 = xb->ys; cs5.c = nc;
                PROF(5, nn_up2_bwd_into(Bc, cs5, u->gout[level + 1], b->c_split, c0));
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
    if (b == &u->enc[0] && u->a0_share && u->xin_shared) nn_stream_wait(0, u->ev_xin[3]);
    if (b == &u->enc[0] && u->l3_ct) PROF(4, nn_input_rad((const char *)u->l3x[1] + l3_ct_off(u), u->xs, u->rad, u->l3x[1]));   /* (the forward's input) */
    if (b == &u->enc[0] && u->l3b) { if (!u->ev_head) u->ev_head = nn_event_create(); nn_event_record(u->ev_head, 0); }   /* X2 (its a1 read by
                                                                    GN1's backward) and the records are free: the next batch's head may go up */
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

/* ---- level-0 slab stream (stage S3 of /home/forrest/.claude/plans/level0-slab-streaming.md; UFSM_L0_STREAM=1 or unet_set_l0_stream).
   On the slab kernel path with several slabs the level-0 ops run per slab -- S2's z-windows on the WHOLE tensors -- in a skewed
   schedule: the forward lags one slab (a 3^3 conv's window reads one plane of the next slab, whose statistics close when that slab is
   written: F1 stem / conv2 / down0, F2 conv1 / conv2 of dec[0]); the backward streams down through dec[0] and its up-part gradient into
   gout[1] (B1: GN2 + conv2's weight gradient at slab t, conv2's backward-data + GN1 + conv1's weight gradient at t + 1, conv1's
   backward-data at t + 2, the upsample backward at t + 3) and up through down0 and enc[0] (B2': down0's weight gradient and
   backward-data into the skip gradient, GN2 and conv2's weight gradient at slab t, conv2's backward-data, GN1 and the stem's weight
   gradient at t - 1). Every op draws the stochastic-rounding seeds of the unstreamed order (its first draw replayed for each window:
   nn_sr_set_ctr; the draws per launch asserted equal), the statistics of a slab are the slab pass's bits (nn_gn_stats_n), so the
   activations, the logits, gout[1], d/d downo[0] and the skip gradient equal the unstreamed slab path's bit for bit and the parameter
   gradients up to the atomic summation order. Whole tensors and no recompute yet: stage S4 replaces the level-0 tensors by slab rings
   (then recomputing what is no longer resident). UFSM_L0_STREAM_POISON=1 overwrites the planes of the transient gradients (dec[0]'s
   d/d a2, A, the up-part gradient; enc[0]'s d/d a2, A) after their last reader in the schedule (NaN scales): the liveness S4 needs.
   Supported: one sample, the plain layout (no split / three level-0 buffers / offloads / lean 2 / head slabs), recompute 1, MX. */
static int g_l0st = -1;
void unet_set_l0_stream(int on) { g_l0st = on; }   /* -1: env UFSM_L0_STREAM */
static int l0st_req(void) { if (g_l0st < 0) { const char *e = getenv("UFSM_L0_STREAM"); g_l0st = e ? atoi(e) : 0; } return g_l0st; }
static int l0st_poison_on(void) { static int v = -1; if (v < 0) { const char *e = getenv("UFSM_L0_STREAM_POISON"); v = e ? atoi(e) : 0; } return v; }   /* 2: one slab early (the test's control) */
static int l0st_want(const unet *u) {   /* the stream requested, and supported for this net (abort when requested but not) */
    if (!l0st_req() || u->cfg.l0_norm != 4) return 0;
    const block *d0 = &u->dec[0], *e0 = &u->enc[0];
    const char *why = !nn_get_tf32() || !act_mx8() || !grad_mx8() ? "MX activations and gradients on the tensor cores"
        : recompute() != 1 ? "recompute 1" : !l0s_kern(u) ? "the slab kernel path (UFSM_L0_SLAB_KERN=1)"
        : u->cfg.l0_slab % 8 || u->ls[0].d % u->cfg.l0_slab || l0s_nslab(u, u->ls[0].d) < 2 ? "several uniform slabs of a multiple of 8 planes"
        : u->xs.n != 1 ? "one sample" : u->cfg.nlev < 2 ? "two levels or more"
        : sp_on(u) ? "no spatial split" : u->l3b ? "no three level-0 buffers" : u->p3 || u->slab ? "no in-place slabs"
        : u->xin_shared || u->a0_share || u->dno || u->sko ? "no activation offloads" : u->nob || u->gco ? "no lean 2" : u->lg_x || u->head_ext ? "no head slabs"
        : u->ctin.on ? "no CT stem" : chunk_up() ? "no chunked up-part gradient"
        : u->cfg.ds_levels || u->cfg.enc_blocks[0] || e0->single || d0->single || e0->a1_shared || d0->a1_shared || e0->a1_lent || d0->a1_lent
          ? "two-conv level-0 blocks with their own a1, no deep supervision"
        : up_wg_chunk() && UPW(1) % 32 == 0 ? "an unchunked decoder conv1 weight gradient (UFSM_UP_WG_CHUNK=0 or a 16-wide level 1)"
        : nullptr;   /* (UPW(1): dec[0].c_split, set by the forward after this check on a first forward) */
    if (why) { fprintf(stderr, "unet: UFSM_L0_STREAM needs %s\n", why); abort(); }
    return l0st_req() >= 2 ? 2 : 1;
}
/* the z-window of slab s, the i-th of an op's n windows in execution order (the first zeroes, the last finalises output sums) */
static void l0st_win(const unet *u, int s, int i, int n) {
    const int S = u->cfg.l0_slab, D = u->ls[0].d, z0 = s * S, z1 = z0 + S < D ? z0 + S : D;
    nn_set_zwin(z0, z1, D, i == 0, i == n - 1);
}
static void l0st_nowin(void) { nn_set_zwin(0, 0, 0, 0, 0); }
typedef struct { unsigned c; int k; } l0op;   /* a streamed op: its first seed draw in the unstreamed order, its draws per launch (-1: not run) */
static void l0op_go(const l0op *o) { nn_sr_set_ctr(o->c); }
static void l0op_end(l0op *o, l0op *next) {
    const int k = (int)(nn_sr_ctr() - o->c);
    if (o->k < 0) { o->k = k; if (next) next->c = o->c + (unsigned)k; }
    else if (o->k != k) rc_fail("level-0 slab stream: an op drew another number of seeds than in its first window");
}
static void l0op_after(const l0op *o) { nn_sr_set_ctr(o->c + (unsigned)o->k); }   /* the counter after the op, as unstreamed */
/* a whole level-0 buffer of the stream (S3: whole tensors), registered as storage dt */
static float *l0st_buf(float **p, size_t *n, size_t bytes, int dt) {
    if (*n < bytes) { nn_free(*p); *p = nn_malloc(bytes); *n = bytes; }
    nn_set_storage(*p, bytes, dt);
    return *p;
}
/* slab s's statistics of conv k's output: the slab pass's bits (l0_slab_fwd: nn_gn_stats over the pseudo-sample shape) */
static void l0st_stats(unet *u, block *b, int k, int s) {
    float *x = k == 1 ? b->a1 : b->a2, *m = k == 1 ? b->m1 : b->m2, *r = k == 1 ? b->r1 : b->r2;
    if (nn_gn_stats_n(x, l0s_shape(u, b), l0_G(u), 1e-5f, m, r, s, 1)) rc_fail("level-0 slab stream: slab statistics (MX level-0 storage)");
}
/* UFSM_L0_STREAM_POISON: slab s of the MX level-0 tensor t (shape s5) overwritten with 0xff bytes (NaN scales): dead for the schedule */
static void l0st_poison_raw(unet *u, void *t, shape5 s5, int s) {
    const int S = u->cfg.l0_slab, D = s5.d;
    if (!l0st_poison_on() || s < 0 || s * S >= D) return;
    const size_t hw = (size_t)s5.h * s5.w, need = (size_t)2 * S * hw * 33;
    if (!u->l0st_poison) { uint8_t *h = malloc(need); memset(h, 0xff, need); u->l0st_poison = nn_malloc(need); nn_h2d(u->l0st_poison, h, need); free(h); }
    const int n = D - s * S < S ? D - s * S : S;
    mx_copy_planes(t, D, s * S, u->l0st_poison, n, 0, n, s5);
}
static void l0st_poison(unet *u, void *t, shape5 s5, int s) {   /* a transient slab after its last reader */
    if (l0st_poison_on() == 2) s -= u->l0st_dir;   /* (control: a slab the schedule still reads) */
    l0st_poison_raw(u, t, s5, s);
}
/* stream 2 (recomputing): a phase's level-0 activations are dead on entry (S4 keeps none of them whole): poisoned all through */
static void l0st_poison_all(unet *u, void *t, shape5 s5) { if (u->l0st >= 2) for (int s = 0; s < l0s_nslab(u, s5.d); s++) l0st_poison_raw(u, t, s5, s); }
/* stage S4, the input ring (--gpu-input): slabs [lo, hi] of the network input built from the CT channel and the records (input_rad
   windows) unless the ring holds them -- the stem's windows read its slab and one plane of each neighbour */
static void l0x_need(unet *u, int lo, int hi) {
    if (!u->l0rx) return;
    const int ns = l0s_nslab(u, u->ls[0].d), R = l0ring_R(9) < 16 ? l0ring_R(9) : 16;
    if (!u->l0x_ct || !u->rad) rc_fail("level-0 slab rings: the input ring needs the CT channel and its records (unet_set_input_rad before the build)");
    for (int s = lo < 0 ? 0 : lo; s <= hi && s < ns; s++) {
        if (u->l0x_held[s % R] == s) continue;
        l0st_win(u, s, 1, 3);
        PROF(4, nn_input_rad(u->l0x_ct, u->xs, u->rad, u->xin));
        l0st_nowin();
        u->l0x_held[s % R] = s;
    }
}
/* stream 2: slab s of a forward level-0 op's output recomputed (0 the stem -> enc[0].a1, 1 enc[0]'s conv2 -> a2, 2 dec[0]'s conv1 ->
   a1, 3 its conv2 -> a2): the forward's seeds (replayed, the draws asserted), its statistics (kept: no statistics pass). The layer and
   conv tags are the forward's; the caller restores its own and its seed counter is kept */
static void l0st_re(unet *u, int op, int s) {
    const int ns = l0s_nslab(u, u->ls[0].d), L = u->cfg.nlev;
    if (u->l0st < 2 || s < 0 || s >= ns || (l0st_poison_on() == 3 && op == 1)) return;   /* (3: the test's control, no skip recompute) */
    block *b = op < 2 ? &u->enc[0] : &u->dec[0];
    const int G = G_of(u, b->c1.cout);
    const unsigned c0 = nn_sr_ctr();
    nn_set_layer(op < 2 ? 0 : 3 * L - 3); nn_set_conv(op & 1);
    l0st_win(u, s, 1, 3);   /* (a middle window: no output sums zeroed or finalised; there are none) */
    nn_sr_set_ctr(u->l0st_rc[op]);
    int r = 0;
    if (op == 0) { l0x_need(u, s - 1, s + 1); l0st_win(u, s, 1, 3); nn_sr_set_ctr(u->l0st_rc[op]);
                   const nn_gn_t gi = gn_in(u, b); PROF(0, nn_conv3d_fwd_gn_stats(b->in, b->xs, gi.G, gi.gamma, gi.beta, gi.mean, gi.rstd, P(u, b->c1.w), B1(u, b), b->c1.cout, b->a1, G, 1e-5f, nullptr, nullptr)); }
    else if (op == 2) r = dec_conv1(u, b, 0, b->a1, G, nullptr, nullptr, nullptr, nullptr, nullptr);
    else PROF(0, nn_conv3d_fwd_gn_stats(b->a1, b->ys, G, P(u, b->n1.gamma), P(u, b->n1.beta), b->m1, b->r1, P(u, b->c2.w), P(u, b->c2.b), b->c2.cout, b->a2, G, 1e-5f, nullptr, nullptr));
    if (r || (int)(nn_sr_ctr() - u->l0st_rc[op]) != u->l0st_rk[op]) rc_fail("level-0 slab stream: a recompute (or its seed draws)");
    l0st_nowin(); nn_set_conv(-1);
    nn_sr_set_ctr(c0);
}
/* test hook (unet_l0_stream_dump): device copies of gout[1] after dec[0]'s backward (0), d/d downo[0] (1), the skip gradient (2) */
static int g_l0st_dump; static void *g_l0st_dumped[4]; static size_t g_l0st_dumpn[4];
void unet_l0_stream_dump(int on) { g_l0st_dump = on; }
const void *unet_l0_stream_dumped(int k, size_t *bytes) { if (bytes) *bytes = k >= 0 && k < 4 ? g_l0st_dumpn[k] : 0; return k >= 0 && k < 4 ? g_l0st_dumped[k] : nullptr; }
static void l0st_dump(int k, const void *p, shape5 s) {
    if (!g_l0st_dump) return;
    const int dt = nn_storage(p);
    const size_t b = dt ? nn_mx_bytes(s, dt) : shape_numel(s) * (GBF ? 2 : 4);
    if (g_l0st_dumpn[k] < b) { nn_free(g_l0st_dumped[k]); g_l0st_dumped[k] = nn_malloc(b); }
    g_l0st_dumpn[k] = b;
    nn_d2d(g_l0st_dumped[k], p, b);
}
/* ---- the stream head (stage S4, unet_set_l0_head): no whole-window logits. The forward runs the 1^3 head on each slab of dec[0]'s
   output as soon as its statistics close (unet_head_fwd_z: fp16 [cout][planes h w]) and hands the slab to fwd (the loss statistics);
   the backward's dec[0] step at slab t computes the slab's logits again, bwd writes the fp16 logit gradient over them, and the head's
   backward (unet_head_bwd_z) writes gout[0]'s rows of the slab, read at once by GroupNorm 2's backward. Seeds: each forward head slab
   replays the forward record (the whole head's first draw), the backward head draws its one seed at its first slab (nn_head_bwd_z),
   so the counter runs as with the whole head. */
static int l0h_on(const unet *u) { return u->l0st && u->l0h_fwd; }
static void *l0h_slab(unet *u, int s, int *z0, int *z1) {
    const int S = u->cfg.l0_slab, D = u->ls[0].d;
    *z0 = s * S; *z1 = *z0 + S < D ? *z0 + S : D;
    const size_t need = (size_t)u->cfg.cout * (*z1 - *z0) * u->ls[0].h * u->ls[0].w * 2;
    if (u->l0h_lgn < need) { nn_free(u->l0h_lg); u->l0h_lg = nn_malloc(need); u->l0h_lgn = need; }
    return u->l0h_lg;
}
static void l0h_fwd_slab(unet *u, int s, int replay) {   /* replay: the backward's recompute (the forward's seeds; the counter kept) */
    int z0, z1; void *lg = l0h_slab(u, s, &z0, &z1);
    const unsigned c0 = nn_sr_ctr();
    nn_set_layer(3 * u->cfg.nlev - 2); nn_set_conv(-1);
    nn_sr_set_ctr(u->l0h_c);
    if (unet_head_fwd_z(u, z0, z1, lg)) rc_fail("level-0 slab stream: the head per slab (MX activations)");
    const int k = (int)(nn_sr_ctr() - u->l0h_c);
    if (u->l0h_k < 0) u->l0h_k = k; else if (k != u->l0h_k) rc_fail("level-0 slab stream: a head slab drew another number of seeds");
    if (replay) nn_sr_set_ctr(c0);
    else if (u->l0h_fwd(u->l0h_ctx, z0, z1, lg)) rc_fail("level-0 slab stream: the head's forward callback");
}
/* stage S4's host mode: slab s of the skip gradient between its ring and the pinned host copy (MX: each block's data rows, then its
   scales), on the compute stream -- out as dec[0]'s backward completes it, in before down0's backward-data accumulates into it */
static void l0r_skip_copy(unet *u, int s, int to_host) {
    shape5 a0 = u->ls[0]; a0.c = u->cfg.widths[0];
    const int dt = grad_dt(), bw = a0.c <= 8 ? 8 : a0.c <= 16 ? 16 : 32, nb = (a0.c + bw - 1) / bw, rb = dt == 4 ? bw / 2 : bw;
    const size_t hw = (size_t)a0.h * a0.w, D = (size_t)a0.d, S = (size_t)u->cfg.l0_slab, z0 = (size_t)s * S, n = (z0 + S < D ? S : D - z0);
    char *dev = (char *)u->gskip[0], *host = (char *)u->l0r_hsk;
    for (int k = 0; k < nb; k++) {
        const size_t od = ((size_t)k * D + z0) * hw * rb, os = (size_t)nb * D * hw * rb + ((size_t)k * D + z0) * hw;
        if (to_host) { nn_copy_async(host + od, dev + od, n * hw * rb); nn_copy_async(host + os, dev + os, n * hw); }
        else { nn_copy_async(dev + od, host + od, n * hw * rb); nn_copy_async(dev + os, host + os, n * hw); }
    }
}
static void l0st_dump_host(int k, const void *h, size_t b) {   /* (the test hook, the skip gradient in the host mode) */
    if (!g_l0st_dump) return;
    if (g_l0st_dumpn[k] < b) { nn_free(g_l0st_dumped[k]); g_l0st_dumped[k] = nn_malloc(b); }
    g_l0st_dumpn[k] = b;
    nn_sync(); nn_h2d(g_l0st_dumped[k], h, b);
}
/* F1: the stem, enc[0]'s conv2 and down0 per slab (conv2 and down0 one slab behind the stem); down_norm's level-1 statistics after */
static const float *l0st_fwd_enc(unet *u, const float *x) {
    block *b = &u->enc[0];
    const int L = u->cfg.nlev, ns = l0s_nslab(u, b->ys.d), G = G_of(u, b->c1.cout), *w = u->cfg.widths;
    b->in = x;
    const nn_gn_t gi = gn_in(u, b), go = gn_out(u, b);
    l0op st = {nn_sr_ctr(), -1}, c2 = {0, -1}, dn = {0, -1};
    for (int t = 0; t <= ns; t++) {
        if (t < ns) {   /* the stem, slab t (no output sums: the slab pass gives the statistics) */
            l0x_need(u, t - 1, t + 1);
            nn_set_layer(0); nn_set_conv(0); l0st_win(u, t, t, ns); l0op_go(&st);
            PROF(0, nn_conv3d_fwd_gn_stats(x, b->xs, gi.G, gi.gamma, gi.beta, gi.mean, gi.rstd, P(u, b->c1.w), B1(u, b), b->c1.cout, b->a1, G, 1e-5f, nullptr, nullptr));
            l0op_end(&st, &c2); l0st_nowin();
            l0st_stats(u, b, 1, t);
        }
        if (t >= 1) {   /* conv2 (its window reads a1's first plane of slab t, whose statistics just closed) and down0, slab t - 1 */
            const int s = t - 1;
            nn_set_layer(0); nn_set_conv(1); l0st_win(u, s, s, ns); l0op_go(&c2);
            PROF(0, nn_conv3d_fwd_gn_stats(b->a1, b->ys, G, P(u, b->n1.gamma), P(u, b->n1.beta), b->m1, b->r1, P(u, b->c2.w), P(u, b->c2.b), b->c2.cout, b->a2, G, 1e-5f, nullptr, nullptr));
            l0op_end(&c2, &dn); l0st_nowin(); nn_set_conv(-1);
            l0st_stats(u, b, 2, s);
            nn_set_layer(L); l0st_win(u, s, s, ns); l0op_go(&dn);
            int r; PROF(0, r = nn_conv3d_fwd_x(b->a2, &go, nullptr, nullptr, 0, 0, b->ys, P(u, u->down[0].w), P(u, u->down[0].b), w[0], 3, 2, u->downo[0], 0, 0.f, nullptr, nullptr));
            if (r) rc_fail("level-0 slab stream: down0");
            l0op_end(&dn, nullptr); l0st_nowin();
        }
    }
    l0op_after(&dn);
    u->l0st_rc[0] = st.c; u->l0st_rk[0] = st.k; u->l0st_rc[1] = c2.c; u->l0st_rk[1] = c2.k;   /* (the recomputes replay them) */
    if (u->cfg.down_norm) {   /* (as unet_forward_x: the statistics now, the transform in enc[1]'s conv1 staging) */
        shape5 ds = u->ls[1]; ds.c = w[0];
        block *nb = &u->enc[1];
        nb->inx = u->downo[0]; nb->ing = &u->dn[0]; nb->inm = u->dm[0]; nb->inr = u->dr[0];
        if (!gn_frozen(u) && nn_gn_stats(u->downo[0], ds, G_of(u, w[0]), 1e-5f, u->dm[0], u->dr[0])) rc_fail("level-0 slab stream: down_norm statistics");
    }
    return u->downo[0];
}
/* F2: dec[0]'s conv1 (the up part from dec[1]'s whole output, the skip enc[0].a2) and conv2 one slab behind */
static void l0st_fwd_dec(unet *u) {
    block *b = &u->dec[0], *e = &u->enc[0];
    const int ns = l0s_nslab(u, b->ys.d), G = G_of(u, b->c1.cout), L = u->cfg.nlev, lag = u->l0st >= 2 ? 2 : 0;
    l0op c1 = {nn_sr_ctr(), -1}, c2 = {0, -1}, hd = {0, -1};
    const int hs = l0h_on(u);   /* (the stream head: each slab's logits as its statistics close) */
    u->l0h_k = -1;
    l0st_poison_all(u, e->a1, e->ys); l0st_poison_all(u, e->a2, e->ys);   /* (stream 2: the skip and enc[0]'s a1 recomputed here) */
    for (int t = 0; t <= ns + lag; t++) {
        l0st_re(u, 0, t); l0st_re(u, 1, t - 1);   /* (stream 2: the skip's slab t - 1 for conv1 two slabs behind) */
        nn_set_layer(3 * L - 3);
        if (t - lag >= 0 && t - lag < ns) {
            const int s = t - lag;
            nn_set_conv(0); l0st_win(u, s, s, ns); l0op_go(&c1);
            if (dec_conv1(u, b, 0, b->a1, G, nullptr, nullptr, nullptr, nullptr, nullptr)) rc_fail("level-0 slab stream: decoder conv1");
            l0op_end(&c1, &c2); l0st_nowin();
            l0st_stats(u, b, 1, s);
        }
        if (t - lag >= 1 && t - lag <= ns) {
            const int s = t - lag - 1;
            nn_set_conv(1); l0st_win(u, s, s, ns); l0op_go(&c2);
            PROF(0, nn_conv3d_fwd_gn_stats(b->a1, b->ys, G, P(u, b->n1.gamma), P(u, b->n1.beta), b->m1, b->r1, P(u, b->c2.w), P(u, b->c2.b), b->c2.cout, b->a2, G, 1e-5f, nullptr, nullptr));
            l0op_end(&c2, &hd); l0st_nowin();
            l0st_stats(u, b, 2, s);
            if (hs) { u->l0h_c = hd.c; l0h_fwd_slab(u, s, 0); hd.k = u->l0h_k; nn_set_layer(3 * L - 3); }
        }
    }
    nn_set_conv(-1);
    l0op_after(hs ? &hd : &c2);
    u->l0st_rc[2] = c1.c; u->l0st_rk[2] = c1.k; u->l0st_rc[3] = c2.c; u->l0st_rk[3] = c2.k;
}
/* dec[0]'s up transient up2(silu(gn(dec[1].a2))) into up (rc_up's arithmetic, whole: S3 keeps the whole tensors) */
static void l0st_up_build(unet *u, block *b, float *up) {
    const block *xb = (const block *)b->xb;
    nn_gn_t g = gn_out(u, xb);
    if (u->gout_cap[1] >= act_bytes_of(xb->ys)) {
        float *t = u->gout[1];
        const int dt0 = nn_storage(t);
        nn_set_storage(t, act_bytes_of(xb->ys), act_dt());
        PROF(3, nn_gn_silu_apply(xb->a2, xb->ys, g.G, g.gamma, g.beta, g.mean, g.rstd, t));
        PROF(5, nn_up2_fwd_into(t, xb->ys, up, b->c_split, 0));
        if (dt0) nn_set_storage(t, u->gout_cap[1], dt0); else nn_storage_forget(t);
    } else PROF(5, nn_up2_fwd_gn_into(xb->a2, xb->ys, &g, up, b->c_split, 0));
}
/* ring mode (S4): dec[0]'s up transient per slab -- the up2 windows from silu(gn(dec[1].a2)), computed once into the level-1 buffer
   (l0r_ups; rc_up's arithmetic as l0st_up_build's first branch: GroupNorm + SiLU stored, then upsampled) -- so that before conv1's
   weight gradient of slab s the ring holds slabs s - 1 .. s + 1 (its window reads one plane beyond each end). *lo: the lowest slab
   built. The op's seeds: the apply's draws, then each window replays the whole up2's first draw (o->k = both, as the whole op) */
static void l0st_up_ring(unet *u, block *b, float *up, int s, l0op *o, l0op *next, int *lo, unsigned *kap) {
    const block *xb = (const block *)b->xb;
    const int ns = l0s_nslab(u, b->ys.d);
    if (u->gout_cap[1] < act_bytes_of(xb->ys)) { fprintf(stderr, "unet: UFSM_L0_RING: dec[0]'s up transient needs the stored-source path (gout[1] holds dec[1]'s output)\n"); abort(); }
    int kup = -1;
    if (o->k < 0) {
        const nn_gn_t g = gn_out(u, xb);
        l0op_go(o);
        if (u->l0r_upsn) nn_set_storage(u->l0r_ups, u->l0r_upsn, act_dt());   /* (else gout[1], registered as the same type) */
        PROF(3, nn_gn_silu_apply(xb->a2, xb->ys, g.G, g.gamma, g.beta, g.mean, g.rstd, u->l0r_ups));
        *kap = nn_sr_ctr() - o->c;
        *lo = ns;
    }
    for (int k = *lo - 1; k >= s - 1 && k >= 0; k--) {
        const unsigned c0 = o->c + *kap;
        nn_sr_set_ctr(c0);
        l0st_win(u, k, ns - 1 - k, ns);
        PROF(5, nn_up2_fwd_into(u->l0r_ups, xb->ys, up, b->c_split, 0));
        l0st_nowin();
        const int d = (int)(nn_sr_ctr() - c0);
        if (kup < 0) kup = d; else if (kup != d) rc_fail("level-0 slab rings: an up2 window drew another number of seeds");
        if (o->k >= 0 && (int)*kap + d != o->k) rc_fail("level-0 slab rings: an up2 window drew another number of seeds");
        *lo = k;
    }
    if (o->k < 0) { o->k = (int)*kap + (kup < 0 ? 0 : kup); if (next) next->c = o->c + (unsigned)o->k; }
}
/* B1: dec[0]'s backward from gout[0] (the head's input gradient) and the upsample backward into gout[1], down the slabs */
static void l0st_bwd_dec(unet *u) {
    block *b = &u->dec[0];
    const int L = u->cfg.nlev, ns = l0s_nslab(u, b->ys.d), G = G_of(u, b->c1.cout);
    float *g = u->g, *A = u->gA[0], *Bu = u->gB[0], *gx2 = u->gskip[0];
    const float *gy = u->gout[0];
    float *B2 = l0st_buf(&u->l0st_g2, &u->l0st_g2n, nn_mx_bytes(b->ys, grad_dt()), grad_dt());   /* d/d a2 (the unstreamed path: in gB[0]) */
    shape5 us = b->xs; us.c = b->c_split;
    float *up = l0st_buf(&u->l0st_up, &u->l0st_upn, act_bytes_of(us), act_dt());
    shape5 src = u->ls[1]; src.c = UPW(1);
    const block *x2b = (const block *)b->x2b;
    const nn_gn_t g2 = gn_out(u, x2b);
    l0op o_g2 = {nn_sr_ctr(), -1}, o_c2w = {0, -1}, o_c2d = {0, -1}, o_g1 = {0, -1}, o_up = {0, -1}, o_c1w = {0, -1}, o_c1d = {0, -1}, o_u2b = {0, -1};
    const int hs = l0h_on(u);   /* (the stream head: its backward per slab, before GroupNorm 2's; its one seed first) */
    const unsigned hb_c = nn_sr_ctr();
    if (hs) o_g2.c = 0;
    int n1 = 0, n2 = 0, n3 = 0;   /* windows run of the ops one, two and three slabs behind */
    int up_lo = ns; unsigned up_kap = 0;   /* (ring mode: the up transient's lowest built slab, its source's draws) */
    u->l0st_dir = 1;
    /* stream 2: every level-0 activation recomputed ahead of its readers, down the slabs: at tick t enc[0]'s a1 of slab t - 3, the
       skip of t - 2 (dec[0]'s conv1 reads its planes around t - 1), dec[0]'s a1 of t - 1 (conv2's weight gradient at t reads its
       planes around t), dec[0]'s a2 of t */
    block *e = &u->enc[0];
    l0st_poison_all(u, e->a1, e->ys); l0st_poison_all(u, e->a2, e->ys); l0st_poison_all(u, b->a1, b->ys); l0st_poison_all(u, b->a2, b->ys);
    for (int t = ns - 1 + (u->l0st >= 2 ? 3 : 0); t >= -3; t--) {
        l0st_re(u, 0, t - 3); l0st_re(u, 1, t - 2); l0st_re(u, 2, t - 1); l0st_re(u, 3, t);
        nn_set_layer(3 * L - 3);
        if (hs && t >= 0 && t < ns) {   /* the head: the slab's logits again, the callback's gradient over them, gout[0]'s rows of the slab */
            l0h_fwd_slab(u, t, 1);
            int z0, z1; void *lg = l0h_slab(u, t, &z0, &z1);
            if (u->l0h_bwd(u->l0h_ctx, z0, z1, lg)) rc_fail("level-0 slab stream: the head's backward callback");
            nn_set_layer(3 * L - 2); nn_set_conv(-1);
            if (t == ns - 1) nn_sr_set_ctr(hb_c);
            if (unet_head_bwd_z(u, z0, z1, lg, t == ns - 1)) rc_fail("level-0 slab stream: the head's backward per slab (MX, 16-bit gradients)");
            if (t == ns - 1) o_g2.c = nn_sr_ctr();   /* (the next op's first draw: after the head's one seed) */
            nn_set_layer(3 * L - 3);
        }
        if (t >= 0 && t < ns) {   /* GN2's backward and conv2's weight gradient, slab t */
            nn_set_conv(1); l0st_win(u, t, ns - 1 - t, ns); l0op_go(&o_g2);
            PROF(3, gn_silu_bwd_b(u, b, 2, b->a2, b->ys, G, P(u, b->n2.gamma), P(u, b->n2.beta), b->m2, b->r2, gy, B2, g + b->n2.gamma, g + b->n2.beta, u->gn_scratch));
            l0op_end(&o_g2, &o_c2w); l0op_go(&o_c2w);
            PROF(2, nn_conv3d_bwd_weight_gn(b->a1, b->ys, G, P(u, b->n1.gamma), P(u, b->n1.beta), b->m1, b->r1, B2, b->ys, g + b->c2.w, g + b->c2.b));
            l0op_end(&o_c2w, &o_c2d); l0st_nowin();
        }
        const int s1 = t + 1, s2 = t + 2, s3 = t + 3;
        if (s1 >= 0 && s1 < ns) {   /* conv2's backward-data (reads d/d a2's last plane of slab t), GN1's backward, conv1's weight gradient: slab t + 1 */
            nn_set_conv(1); l0st_win(u, s1, n1, ns); l0op_go(&o_c2d);
            PROF(1, nn_conv3d_bwd_data(B2, b->ys, P(u, b->c2.w), b->ys, 3, 1, A, u->conv_scratch));
            l0op_end(&o_c2d, &o_g1); l0op_go(&o_g1);
            PROF(3, gn_silu_bwd_b(u, b, 1, b->a1, b->ys, G, P(u, b->n1.gamma), P(u, b->n1.beta), b->m1, b->r1, A, A, g + b->n1.gamma, g + b->n1.beta, u->gn_scratch));
            l0op_end(&o_g1, &o_up); l0st_nowin();
            nn_set_conv(0);
            if (u->l0ring) l0st_up_ring(u, b, up, s1, &o_up, &o_c1w, &up_lo, &up_kap);   /* (ring: per slab, s1 - 1 .. s1 + 1 held) */
            else if (o_up.k < 0) { l0op_go(&o_up); l0st_up_build(u, b, up); l0op_end(&o_up, &o_c1w); }   /* the up transient, once */
            l0st_win(u, s1, n1, ns); l0op_go(&o_c1w);
            int r; PROF(2, r = nn_conv3d_bwd_weight_x(up, nullptr, x2b->a2, &g2, b->c_split, 0, b->xs, A, b->ys, 3, 1, g + b->c1.w, g + b->c1.b));
            if (r) rc_fail("level-0 slab stream: decoder conv1 weight gradient");
            l0op_end(&o_c1w, &o_c1d); l0st_nowin(); n1++;
            l0st_poison(u, B2, b->ys, s1 + 1);   /* (d/d a2 of slab t + 2: its last reader was this backward-data) */
        }
        if (s2 >= 0 && s2 < ns) {   /* conv1's backward-data (reads A's last plane of slab t + 1): the up-part and the skip gradient, slab t + 2 */
            nn_set_conv(0); l0st_win(u, s2, n2++, ns); l0op_go(&o_c1d);
            PROF(1, nn_conv3d_bwd_data_split(A, b->ys, P(u, b->c1.w), b->xs, Bu, gx2, b->c_split, u->conv_scratch));
            l0op_end(&o_c1d, &o_u2b); l0st_nowin();
            if (u->l0r_hsk) PROF(4, l0r_skip_copy(u, s2, 1));   /* (host mode: the skip gradient's slab out) */
            l0st_poison(u, A, b->ys, s2 + 1);
        }
        if (s3 >= 0 && s3 < ns) {   /* the upsample backward into gout[1]'s planes of slab t + 3 (reads the up-part gradient's last plane of slab t + 2) */
            nn_set_conv(-1); l0st_win(u, s3, n3++, ns); l0op_go(&o_u2b);
            PROF(5, nn_up2_bwd(Bu, src, u->gout[1]));
            l0op_end(&o_u2b, nullptr); l0st_nowin();
            l0st_poison(u, Bu, us, s3 + 1);
        }
    }
    nn_set_conv(-1);
    l0op_after(&o_u2b);
}
/* B2': down0's backward (gin = d/d downo[0]: weight gradient, backward-data accumulated into the skip gradient) and enc[0]'s, up the slabs */
static void l0st_bwd_enc(unet *u, float *gin) {
    block *b = &u->enc[0];
    const int L = u->cfg.nlev, ns = l0s_nslab(u, b->ys.d), G = G_of(u, b->c1.cout);
    float *g = u->g, *A = u->gA[0], *gsk = u->gskip[0];
    /* d/d a2 in the stream's own buffer: the levels share one B, and gin (level 1's) is it -- read by down0's windows while enc[0]'s
       windows write theirs (unstreamed, down0 finishes first) */
    float *B = l0st_buf(&u->l0st_g2, &u->l0st_g2n, nn_mx_bytes(b->ys, grad_dt()), grad_dt());
    if ((const char *)gin < (const char *)u->gA[0] + u->gB_bytes && (const char *)gin >= (const char *)u->gA[0]) rc_fail("level-0 slab stream: d/d downo[0] in level 0's A");
    const shape5 ds = u->enc[1].xs;
    const nn_gn_t gg = gn_out(u, b);
    l0op o_dw = {nn_sr_ctr(), -1}, o_dd = {0, -1}, o_g2 = {0, -1}, o_c2w = {0, -1}, o_c2d = {0, -1}, o_g1 = {0, -1}, o_sw = {0, -1};
    u->l0st_dir = -1;
    /* stream 2: enc[0]'s a1 recomputed one slab ahead (conv2's weight gradient at t reads its planes around t), the skip at t (down0's
       weight gradient reads its planes up to t, GN2 slab t) */
    l0st_poison_all(u, b->a1, b->ys); l0st_poison_all(u, b->a2, b->ys);
    l0st_re(u, 0, 0);
    for (int t = 0; t <= ns; t++) {
        l0st_re(u, 0, t + 1); l0st_re(u, 1, t);
        if (t < ns && u->l0r_hsk) PROF(4, l0r_skip_copy(u, t, 0));   /* (host mode: the skip gradient's slab in, before down0's backward-data adds to it) */
        if (t < ns) {   /* down0 (weight gradient; backward-data into the skip gradient's slab t: complete now), GN2 and conv2's weight gradient */
            nn_set_layer(L); nn_set_conv(-1); l0st_win(u, t, t, ns); l0op_go(&o_dw);
            int r; PROF(2, r = nn_conv3d_bwd_weight_x(b->a2, &gg, nullptr, nullptr, 0, 0, b->ys, gin, ds, 3, 2, g + u->down[0].w, g + u->down[0].b));
            if (r) rc_fail("level-0 slab stream: down0 weight gradient");
            l0op_end(&o_dw, &o_dd); l0op_go(&o_dd);
            PROF(1, r = nn_conv3d_bwd_data_acc(gin, ds, P(u, u->down[0].w), b->ys, 3, 2, gsk, u->conv_scratch));
            if (r) rc_fail("level-0 slab stream: down0's backward-data does not accumulate");
            l0op_end(&o_dd, &o_g2);
            nn_set_layer(0); nn_set_conv(1); l0op_go(&o_g2);
            PROF(3, gn_silu_bwd_b(u, b, 2, b->a2, b->ys, G, P(u, b->n2.gamma), P(u, b->n2.beta), b->m2, b->r2, gsk, B, g + b->n2.gamma, g + b->n2.beta, u->gn_scratch));
            if (u->l0r_hsk && g_l0st_dump) l0r_skip_copy(u, t, 1);   /* (the test hook: the accumulated slab, as the device mode's whole tensor ends) */
            l0op_end(&o_g2, &o_c2w); l0op_go(&o_c2w);
            PROF(2, nn_conv3d_bwd_weight_gn(b->a1, b->ys, G, P(u, b->n1.gamma), P(u, b->n1.beta), b->m1, b->r1, B, b->ys, g + b->c2.w, g + b->c2.b));
            l0op_end(&o_c2w, &o_c2d); l0st_nowin();
        }
        if (t >= 1) {   /* conv2's backward-data (reads d/d a2's first plane of slab t), GN1, the stem's weight gradient: slab t - 1 */
            const int s = t - 1;
            nn_set_layer(0); nn_set_conv(1); l0st_win(u, s, s, ns); l0op_go(&o_c2d);
            PROF(1, nn_conv3d_bwd_data(B, b->ys, P(u, b->c2.w), b->ys, 3, 1, A, u->conv_scratch));
            l0op_end(&o_c2d, &o_g1); l0op_go(&o_g1);
            PROF(3, gn_silu_bwd_b(u, b, 1, b->a1, b->ys, G, P(u, b->n1.gamma), P(u, b->n1.beta), b->m1, b->r1, A, A, g + b->n1.gamma, g + b->n1.beta, u->gn_scratch));
            l0op_end(&o_g1, &o_sw);
            nn_set_conv(0);
            l0x_need(u, s - 1, s + 1); l0st_win(u, s, s, ns);   /* (the input ring: its window rebuilt around slab s) */
            l0op_go(&o_sw);
            PROF(2, nn_conv3d_bwd_weight(b->in, b->xs, A, b->ys, 3, 1, g + b->c1.w, g + b->c1.b));
            l0op_end(&o_sw, nullptr); l0st_nowin();
            l0st_poison(u, B, b->ys, s - 1);   /* d/d a2 of slab t - 2: its last reader (conv2's backward-data of slab t - 1) ran */
            l0st_poison(u, A, b->ys, s);
        }
    }
    nn_set_conv(-1);
    l0op_after(&o_sw);
}

void unet_backward(unet *u, const float *glogits) { unet_backward_x(u, glogits, 0); }
/* deep supervision, before dec[j]'s backward: the head's weight / bias gradients (accumulating kernels: zeroed first) from the level's
   output (recompute: a2 with its GroupNorm + SiLU on the fly, else s2) and its input gradient, written in gout[j]'s storage (MX by
   registration, 16-bit or fp32) and added to the up-path gradient already there */
static void ds_bwd(unet *u, int j, float *g) {
    const block *d = &u->dec[j]; const convp *c = &u->ds[j];
    shape5 xs = d->ys, os = xs; os.c = u->cfg.cout;
    const size_t no = shape_numel(os), nx = shape_numel(xs);
    const float *gl = u->ds_gl[j];
    if (GBF) {   /* the 16-bit logit gradient the 1^3 kernels read, with the gradient scale (unscaled with the rest at the end) */
        if (u->ds_g16_n < no * 2) { if (u->ds_g16) nn_free(u->ds_g16); u->ds_g16 = nn_malloc(no * 2); u->ds_g16_n = no * 2; }
        nn_f32_to_h16(u->ds_gl[j], no, u->ds_g16, nn_get_grad_scale()); gl = (const float *)u->ds_g16;
    }
    nn_zero(g + c->w, (size_t)c->cout * c->cin * 4); nn_zero(g + c->b, (size_t)c->cout * 4);
    if (recompute()) { nn_gn_t gg = gn_out(u, d); int r = nn_conv3d_bwd_weight_x(d->a2, &gg, nullptr, nullptr, 0, 0, xs, gl, os, 1, 1, g + c->w, g + c->b); if (r) rc_fail("deep-supervision head weight gradient"); }
    else nn_conv3d_bwd_weight(d->s2, xs, gl, os, 1, 1, g + c->w, g + c->b);
    const int dt = nn_storage(u->gout[j]);
    const size_t tb = dt ? nn_mx_bytes(xs, dt) : nx * (GBF ? 2 : 4);
    if (u->ds_tmp_n < tb) { if (u->ds_tmp) nn_free(u->ds_tmp); u->ds_tmp = nn_malloc(tb); u->ds_tmp_n = tb; }
    nn_set_storage(u->ds_tmp, tb, dt);
    nn_conv3d_bwd_data(gl, os, P(u, c->w), xs, 1, 1, (float *)u->ds_tmp, u->conv_scratch);
    nn_add_into(u->gout[j], u->ds_tmp, xs, GBF);
}
const float *unet_ds_logits(const unet *u, int j, shape5 *s) {
    if (j < 1 || j > u->cfg.ds_levels || !u->ds_lg[j]) return nullptr;
    if (s) { *s = u->ls[j]; s->c = u->cfg.cout; }
    return u->ds_lg[j];
}
void unet_set_ds_grad(unet *u, int j, const float *gl) { if (j >= 1 && j <= u->cfg.ds_levels) u->ds_gl[j] = gl; }
void unet_backward_x(unet *u, const void *gv, int g_h16) {
    const float *glogits = (const float *)gv;
    int L = u->cfg.nlev;
    nn_set_nlev(L);
    g_in_bwd = 1;
    const float gscale = GBF ? nn_get_grad_scale() : 1.f;
    sp_cfg(u);
    if (g_h16 && !GBF) { fprintf(stderr, "unet_backward_x: a 16-bit logit gradient needs the 16-bit gradient storage\n"); abort(); }
    if (GBF && !g_h16 && !u->head_ext && !l0h_on(u)) { shape5 os = u->ls[0]; os.c = u->cfg.cout; if (!u->glb) u->glb = dalloc_grad(u, shape_numel(os)); nn_f32_to_h16(glogits, shape_numel(os), u->glb, gscale); glogits = u->glb; }
    const int *w = u->cfg.widths;
    float *g = u->g;
    if ((u->xin_shared || u->a0_share) && u->a2_stale) {   /* repeated backward: dec[0].a2's buffer holds the input, the decoders' a2 were
                                                             their levels' A -> re-run their conv2 (and conv1 where the shared encoder a1
                                                             overwrote it) from the bottom up, the forward's kernels and precision */
        if ((u->xin_shared && !u->xin_out) || (u->a0_share && !u->a2_out)) { fprintf(stderr, "unet: a backward needs a training forward first (the activation offload)\n"); abort(); }
        if (u->l3b) { fprintf(stderr, "unet: a repeated backward with three level-0 buffers (UFSM_L0_3BUF=0 for one)\n"); abort(); }
        if (u->sko) {   /* the coarse skips (the decoders' conv1 re-runs read them; dec[0]'s conv2 then overwrites them, and they come back again) */
            if (!u->sk_out) { fprintf(stderr, "unet: a backward needs a training forward first (the skip offload)\n"); abort(); }
            for (int k = 1; k < L - 1; k++) { nn_offload_copy(stage_out(u, k)->a2, u->h_sk[k], act_bytes_of(u->enc[k].ys), 0, u->ev_sk[k][2], u->ev_sk[k][3]); nn_stream_wait(0, u->ev_sk[k][3]); }
        }
        if (u->a0_share) {   /* the skip (read by dec[0]'s conv1 re-run) back from the host first */
            nn_set_storage(u->enc[0].a2, act_bytes_of(u->enc[0].ys), act_dt());
            nn_offload_copy(u->enc[0].a2, u->h_a2, act_bytes_of(u->enc[0].ys), 0, u->ev_a2[2], u->ev_a2[3]);
            nn_stream_wait(0, u->ev_a2[3]);
        }
        for (int i = u->a0_share ? L - 2 : 0; i >= 0; i--) {
            block *b = &u->dec[i];
            const int G = G_of(u, b->c1.cout);
            nn_set_layer(3 * L - 3 - i);
            if (b->a1_lent && u->sea_stale) { nn_set_conv(0); if (dec_conv1(u, b, i, b->a1, G, nullptr, nullptr, nullptr, nullptr, nullptr)) rc_fail("decoder conv1 recompute"); sp_halo(u, b->a1, b->ys, 0); }
            nn_set_storage(b->a2, act_bytes_of(b->ys), act_dt());
            nn_set_conv(1);
            PROF(0, nn_conv3d_fwd_gn_stats(b->a1, b->ys, G, P(u, b->n1.gamma), P(u, b->n1.beta), b->m1, b->r1, P(u, b->c2.w), P(u, b->c2.b), b->c2.cout, b->a2, G, 1e-5f, b->m2, b->r2));
            nn_set_conv(-1);
            sp_halo(u, b->a2, b->ys, 0);
        }
        u->a2_stale = 0;
    }
    /* head; lean mode: the 16-bit logit gradient may sit in B (unet_grad_scratch), whose MX-fp8 registration must not make the
       head ops read it as MX */
    int ghide = 0; float *gbuf = nullptr; size_t gcap = 0;
    {   /* lean: the logit gradient sits in B (lean 1) or after the logits in A (lean 2) */
        float *cand[3] = {u->gB[0], u->gA[0], u->a0_share ? u->enc[0].a2 : nullptr};
        size_t caps[3] = {u->gB_bytes, u->a0_share ? 0 : u->gA_bytes, u->a0_share ? act_bytes_of(u->enc[0].ys) : 0};
        for (int k = 0; k < 3 && !gbuf; k++) if (cand[k] && (const char *)glogits >= (const char *)cand[k] && (const char *)glogits < (const char *)cand[k] + caps[k]) { gbuf = cand[k]; gcap = caps[k]; }
        /* the logit gradient must read as plain 16-bit: the buffer's MX registration ends where it starts (or is dropped when it
           starts the buffer); the part below stays MX, which matters when the buffer is gout[0], the head's backward-data output */
        if (gbuf) { ghide = nn_storage(gbuf); if (ghide) { const size_t below = (size_t)((const char *)glogits - (const char *)gbuf); if (below) nn_set_storage(gbuf, below, ghide); else nn_storage_forget(gbuf); } }
    }
    block *d0 = &u->dec[0];
    shape5 os = u->ls[0]; os.c = u->cfg.cout;
    nn_set_layer(3 * L - 2);
    int hf = -1;   /* the fused head backward (weight, bias and input gradient in one pass over act(x) and the logit gradient) */
    if (l0h_on(u)) hf = 0;   /* (the stream head: per slab in dec[0]'s stream step) */
    else if (u->head_ext) {   /* head slabs: the trainer's callback runs head, loss gradient and head backward per z-slab (unet_head_*_z) */
        if (!u->head_cb) { fprintf(stderr, "unet_backward_x: head slabs without a callback\n"); abort(); }
        u->head_cb(u->head_ctx);
        hf = 0;
    } else if (u->l3b) {   /* three level-0 buffers: no buffer for the head's input gradient. The head's weight gradient, then dec[0]'s GroupNorm
                              backward recomputing that gradient from the logit gradient (nn_gn_silu_bwd_head: the bits of the head's
                              backward-data into gout[0] followed by block_bwd's GroupNorm backward, the same seeds drawn), d/d a2 in place in X2 */
        nn_gn_t gg = gn_out(u, d0); int r;
        PROF(2, r = nn_conv3d_bwd_weight_x(d0->a2, &gg, nullptr, nullptr, 0, 0, d0->ys, glogits, os, 1, 1, g + u->head.w, g + u->head.b));
        if (r) rc_fail("head weight gradient");
        if (nn_head_bwd_x_draws()) (void)nn_sr_seed();   /* (nn_head_bwd_x's draw before it declines 8 / 16 inputs) */
        const unsigned hs = nn_sr_seed();               /* (the head backward-data's) */
        nn_exec_prec_note(1, 0);
        nn_set_layer(3 * L - 3); nn_set_conv(1);
        nn_set_gn_const(l0_fixed(u, d0));
        const int hns = l0_slab_on(u, d0) && l0s_kern(u) ? l0s_nslab(u, d0->ys.d) : 1;   /* slab kernel path: over the pseudo-sample shape */
        PROF(3, r = nn_gn_silu_bwd_head_ns(d0->a2, hns > 1 ? l0s_shape(u, d0) : d0->ys, G_of(u, d0->c2.cout), P(u, d0->n2.gamma), P(u, d0->n2.beta), d0->m2, d0->r2,
                                           glogits, u->cfg.cout, P(u, u->head.w), hs, g + d0->n2.gamma, g + d0->n2.beta, u->gn_scratch, hns));
        nn_set_gn_const(0);
        if (r) rc_fail("three level-0 buffers: the head-recomputing GroupNorm backward");
        nn_set_conv(-1); nn_set_layer(3 * L - 2);
        nn_set_storage(u->l3x[3], u->l3xb[3], grad_dt());   /* X3: level 0's A from dec[0]'s conv2 backward-data on */
        hf = 0;
    } else if (recompute()) { nn_gn_t gg = gn_out(u, d0); PROF(2, hf = nn_head_bwd_x(d0->a2, &gg, d0->ys, glogits, os, P(u, u->head.w), g + u->head.w, g + u->head.b, u->gout[0])); }
    else PROF(2, hf = nn_head_bwd_x(d0->s2, nullptr, d0->ys, glogits, os, P(u, u->head.w), g + u->head.w, g + u->head.b, u->gout[0]));
    if (hf) {
        if (recompute()) { nn_gn_t gg = gn_out(u, d0); int r; PROF(2, r = nn_conv3d_bwd_weight_x(d0->a2, &gg, nullptr, nullptr, 0, 0, d0->ys, glogits, os, 1, 1, g + u->head.w, g + u->head.b)); if (r) rc_fail("head weight gradient"); }
        else PROF(2, nn_conv3d_bwd_weight(d0->s2, d0->ys, glogits, os, 1, 1, g + u->head.w, g + u->head.b));
        PROF(1, nn_conv3d_bwd_data(glogits, os, P(u, u->head.w), d0->ys, 1, 1, u->gout[0], u->conv_scratch));
    }
    if (ghide) dstorage(gbuf, gcap, ghide);
    if (u->a0_share && !u->l3b) nn_offload_copy(u->enc[0].a2, u->h_a2, act_bytes_of(u->enc[0].ys), 0, u->ev_a2[2], u->ev_a2[3]);   /* the skip back for dec[0]'s conv1 weight gradient
                                                                                                                                (l3b: inside it, l3_skip_return) */
    if (!u->l3b) FQG(u->gout[0], d0->ys);
    if (u->gco && !u->l3b) { u->gout[1] = u->gB[1] = u->gout_x1; if (L > 2) u->gskip[1] = u->gout_x1; }   /* dec[0]'s backward: gout[1] in dec[0].a1's buffer */
    /* decoder, bottom-up in the graph = i from 0 to L-2 */
    for (int i = 0; i < L - 1; i++) {
        if (i >= 1 && i <= u->cfg.ds_levels && u->ds_gl[i]) ds_bwd(u, i, g);   /* deep supervision: gout[i] += the level head's input gradient */
        shape5 li = u->ls[i];
        shape5 src = u->ls[i + 1]; src.c = UPW(i + 1);
        if (i == 0 && u->l0st) {   /* (level-0 slab stream: B1) */
            l0st_bwd_dec(u); l0st_dump(0, u->gout[1], src);
            shape5 g0 = li; g0.c = w[0];   /* (the whole 16-channel tensor: ls[0] carries the input's channels) */
            if (u->l0r_hsk) l0st_dump_host(3, u->l0r_hsk, u->l0r_hskn); else l0st_dump(3, u->gskip[0], g0);
            continue;
        }
        if (nn_get_tf32()) {
            if (i > 0 && u->sko && u->sea_stale) nn_stream_wait(0, u->ev_sk[i][3]);   /* its skip back from the host (a conv1 re-run reads it at once) */
            const int l3 = u->l3b && i == 0;   /* (l3b: dec[0]'s d/d a2 already in place in its a2) */
            nn_set_layer(3 * L - 3 - i); float *gup = block_bwd(u, &u->dec[i], i, l3 ? u->dec[0].a2 : u->gout[i], u->gskip[i], l3, nullptr, nullptr);   /* gB[i]: grad wrt the upsampled part; skip grad written in place */
            if (i == 0 && u->gco && !u->l3b) {   /* gout[1] (the up-part gradient) to dec[0].a2's buffer before dec[0].a1's takes the encoder's a1 back */
                if (gup) rc_fail("coarse gradient share: an unchunked up-part gradient");
                shape5 g1 = u->ls[1]; g1.c = UPW(1);
                nn_d2d(u->gout_home1, u->gout_x1, nn_mx_bytes(g1, grad_dt()));
                u->gout[1] = u->gB[1] = u->gout_home1; if (L > 2) u->gskip[1] = u->gout_home1;
            }
            /* (l3_sched) the skip below gout[1]'s home is still whole in X2 (dec[0]'s conv1 weight gradient read it): into X3 on the device
               before anything is copied over it, so the host copy into X3 below brings only the bytes gout[1] took (UFSM_L3_SKIP_D2D) */
            const size_t skd = i == 0 && l3_sched(u) && l3_skip_d2d() && u->gout_home1 && u->enc[0].a2 == u->l3x[2] ? l3_skip_lo(u) : 0;
            if (skd) nn_d2d_kernel(u->l3x[3], u->l3x[2], skd);
            if (i == 0 && u->l3b) l3_homes(u, 0);   /* (the coarse tensors come back to X2) */
            if (i == 0 && u->sko) {   /* the coarse skips back (the next decoder's first) */
                if (!u->sk_out) { fprintf(stderr, "unet: a backward needs a training forward first (the skip offload)\n"); abort(); }
                for (int k = 1; k < L - 1; k++) nn_offload_copy(stage_out(u, k)->a2, u->h_sk[k], act_bytes_of(u->enc[k].ys), 0, u->ev_sk[k][2], u->ev_sk[k][3]);
            }
            if (i == 0 && u->dno) {   /* the down-conv outputs back below them (ahead of the encoders' a1 on the copy stream) */
                if (!u->dn_out) { fprintf(stderr, "unet: a backward needs a training forward first (the down-conv output offload)\n"); abort(); }
                for (int k = L - 2; k >= 0; k--) { shape5 ds = u->ls[k + 1]; ds.c = w[k]; nn_offload_copy(u->downo[k], u->h_dn[k], act_bytes_of(ds), 0, u->ev_dn[k][2], u->ev_dn[k][3]); }
            }
            if (i == 0 && l3_sched(u)) {   /* (l3_sched) after the coarse tensors: the skip into X3 (A is dead) for down0's weight gradient, then the encoder's
                                       a1 into X2 but for the coarse homes' bytes */
                u->enc[0].a2 = u->l3x[3];
                nn_set_storage(u->l3x[3], act_bytes_of(u->enc[0].ys), act_dt());
                nn_offload_copy((char *)u->l3x[3] + skd, (const char *)u->h_a2 + skd, act_bytes_of(u->enc[0].ys) - skd, 0, u->ev_a2[2], u->ev_a2[3]);
                size_t flo, fhi; l3_x2_free(u, &flo, &fhi);
                if (u->a1_out[0] && fhi > flo) nn_offload_copy((char *)u->l3x[2] + flo, (const char *)u->h_a1[0] + flo, fhi - flo, 0, u->ev_a1[0][2], u->ev_a1[0][3]);
            }
            if (u->a1_out[i] && !(i == 0 && (u->l3b || a1_out_split()))) nn_offload_copy(u->enc[i].a1, u->h_a1[i], act_bytes_of(u->enc[i].ys), 0, u->ev_a1[i][2], u->ev_a1[i][3]);   /* the decoder's
                                                                         a1 is dead (level 0's after the decoders, l3b: into X2 after down0's backward) */

            if (i == 0 && u->xin_shared && !u->a0_share) {   /* and its a2: the input back for the stem's weight gradient */
                if (!u->xin_out) { fprintf(stderr, "unet: a backward needs a training forward first (the input offload)\n"); abort(); }
                nn_set_storage(u->xin, u->xin_bytes, unet_input_prec());
                nn_offload_copy(u->xin, u->h_xin, u->xin_bytes, 0, u->ev_xin[2], u->ev_xin[3]);
            }
            if (i == 0 && (u->xin_shared || u->a0_share)) u->a2_stale = 1;
            if (gup) { shape5 us = li; us.c = UPW(i + 1); sp_halo(u, gup, us, 1); }
            if (gup) PROF(5, nn_up2_bwd(gup, src, u->gout[i + 1]));   /* nullptr: chunk mode wrote gout[i + 1] */
            FQG(u->gout[i + 1], src);
            if (i == 0) { shape5 g0 = li; g0.c = w[0]; l0st_dump(0, u->gout[1], src); l0st_dump(3, u->gskip[0], g0); }
        } else {
            nn_set_layer(3 * L - 3 - i); float *gcat = block_bwd(u, &u->dec[i], i, u->gout[i], nullptr, 0, nullptr, nullptr);       /* gB[i]: grad wrt concat */
            PROF(5, nn_concat_bwd(gcat, UPW(i + 1), w[i], li, u->gA[i], u->gskip[i]));
            PROF(5, nn_up2_bwd(u->gA[i], src, u->gout[i + 1]));
        }
    }
    if (u->a1_out[0] && !u->l3b && a1_out_split()) nn_offload_copy(u->enc[0].a1, u->h_a1[0], act_bytes_of(u->enc[0].ys), 0, u->ev_a1[0][2], u->ev_a1[0][3]);   /* (level 0's a1,
                    behind the coarse ones on the copy stream: enc[0]'s backward, the last, waits for it) */
    if (u->l3b && !l3_sched(u)) {   /* the skip again, into X3 (A dead since dec[0]'s backward), for down0's weight gradient and enc[0]'s GroupNorm
                       backward; after the decoders (their encoders' a1 come back first on the one copy stream; l3_sched: after dec[0]'s) */
        u->enc[0].a2 = u->l3x[3];
        nn_set_storage(u->l3x[3], act_bytes_of(u->enc[0].ys), act_dt());
        nn_offload_copy(u->l3x[3], u->h_a2, act_bytes_of(u->enc[0].ys), 0, u->ev_a2[2], u->ev_a2[3]);
    }
    /* encoder, from the bottom */
    for (int i = L - 1; i >= 0; i--) {
        float *gy = i < L - 1 ? u->gskip[i] : u->gout[i];   /* enc[i].s2 receives skip + down-conv grads; the bottom gets the up-path grad */
        if (i == 0 && u->xin_shared && !u->a0_share) nn_stream_wait(0, u->ev_xin[3]);
        if (i > 0 && u->dno) nn_stream_wait(0, u->ev_dn[i - 1][3]);   /* its input back from the host */
        int pre = 0; const float *gid = nullptr;
        if (i == L - 1 && u->tbn) { nn_set_layer(i); nn_set_conv(1); gy = tb_bwd(u, gy); pre = 1; }   /* the attention stage: gy = d/d (the bottleneck's raw output) */
        if (u->cfg.enc_blocks[i] > 0) {   /* back through the residual blocks (last first); each saves d/d x_k, the identity gradient of x_{k-1} */
            shape5 ys = u->ls[i]; ys.c = w[i];
            for (int k = u->cfg.enc_blocks[i] - 1; k >= 0; k--) {
                nn_set_layer(i); nn_set_block(1 + k); float *r = block_bwd(u, &u->rb[i][k], i, gy, nullptr, pre, gid, u->gres[i]); nn_set_block(0);
                if (nn_get_tf32()) { nn_add_into(r, u->gres[i], ys, GBF); pre = 1; gid = nullptr; }   /* r = d/d x_{k-1} through conv1 and its input GroupNorm */
                else { pre = 0; gid = u->gres[i]; }   /* r = d/d silu(gn(x_{k-1})): the identity joins after the GroupNorm backward of the block before */
                gy = r;
            }
        }
        nn_set_layer(i); float *gin = block_bwd(u, &u->enc[i], i, gy, nullptr, pre, gid, nullptr);            /* grad wrt enc[i] input */
        if (i == 1) l0st_dump(1, gin, u->enc[1].xs);
        if (i == 1 && u->l0st) { l0st_bwd_enc(u, gin); break; }   /* (level-0 slab stream: B2', down0 and enc[0]) */
        if (i > 0) {
            /* enc[i] input = downo[i-1] = down[i-1](level i-1's output): propagate to it, accumulate into gskip[i-1] */
            block *so = stage_out(u, i - 1);
            shape5 ds = u->enc[i].xs;                                     /* == down output shape */
            nn_set_layer(L + i - 1);
            sp_zero(u, gin, ds, 1);
            sp_begin(u, gin, ds, 1);
            if (i == 1 && u->l3b) nn_stream_wait(0, u->ev_a2[3]);   /* (l3b: the skip back in X3) */
            if (recompute()) { nn_gn_t gg = gn_out(u, so); int r; PROF(2, r = nn_conv3d_bwd_weight_x(so->a2, &gg, nullptr, nullptr, 0, 0, so->ys, gin, ds, 3, 2, g + u->down[i - 1].w, g + u->down[i - 1].b)); if (r) rc_fail("down weight gradient"); }
            else PROF(2, nn_conv3d_bwd_weight(so->s2, so->ys, gin, ds, 3, 2, g + u->down[i - 1].w, g + u->down[i - 1].b));
            sp_end(u, gin, ds, 1);
            int acc; PROF(1, acc = nn_conv3d_bwd_data_acc(gin, ds, P(u, u->down[i - 1].w), so->ys, 3, 2, u->gskip[i - 1], u->conv_scratch));   /* gskip += */
            if (acc && u->gco && i == 1) rc_fail("coarse gradient share: a down-conv backward without accumulation (gA[0] holds its input)");
            if (acc == 0) FQG(u->gskip[i - 1], so->ys);
            if (i == 1 && u->l3b) {   /* X2's coarse tensors are dead: the encoder's a1 back into it for enc[0]'s backward (exposed; l3_sched: the
                                         coarse homes' bytes only, the rest came back during the coarse levels' backward) */
                u->enc[0].a1 = u->l3x[2];
                nn_set_storage(u->l3x[2], act_bytes_of(u->enc[0].ys), act_dt());
                const size_t ab = act_bytes_of(u->enc[0].ys);
                if (l3_sched(u)) {
                    size_t flo, fhi; l3_x2_free(u, &flo, &fhi);
                    if (fhi <= flo) flo = fhi = ab;
                    nn_offload_copy(u->l3x[2], u->h_a1[0], flo, 0, u->ev_a1[0][2], u->ev_a1[0][3]);
                    if (ab > fhi) nn_offload_copy((char *)u->l3x[2] + fhi, (const char *)u->h_a1[0] + fhi, ab - fhi, 0, u->ev_a1[0][2], u->ev_a1[0][3]);
                } else nn_offload_copy(u->l3x[2], u->h_a1[0], ab, 0, u->ev_a1[0][2], u->ev_a1[0][3]);
            }
            if (acc) {
                PROF(1, nn_conv3d_bwd_data(gin, ds, P(u, u->down[i - 1].w), so->ys, 3, 2, u->gA[i - 1], u->conv_scratch));
                PROF(4, nn_axpy(u->gskip[i - 1], 1.f, u->gA[i - 1], shape_numel(so->ys)));
            }
        }
    }
    if (u->l0r_hsk) l0st_dump_host(2, u->l0r_hsk, u->l0r_hskn); else l0st_dump(2, u->gskip[0], u->enc[0].ys);
    if (gscale != 1.f) nn_scale(g, 1.f / gscale, u->np);   /* activation gradients were scaled for 16-bit storage */
    if (share_enc_a1()) u->sea_stale = 1;   /* the decoders' a1 buffers now hold the encoders' */
    /* d/dv of the stem bias b + c v is c d/db (exact when the gradient was zeroed before this backward: one backward per step) */
    if (u->cfg.scale_cond) nn_axpy(g + u->sc_w, u->sc_c, g + u->enc[0].c1.b, (size_t)u->enc[0].c1.cout);
    if (u->cfg.l0_norm == 3) ws_backward(u);
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
static int trunk_lr_on(const unet *u);
void unet_adamw(unet *u, float lr, float b1, float b2, float eps, float wd, int step) {
    if (u->wq) wq_adamw(u, lr, b1, b2, eps, wd, step);
    else if (trunk_lr_on(u)) {   /* the trunk [0, tb_p0) at trunk_lr x lr, the stage at lr */
        nn_adamw(u->p, u->g, u->m, u->v, u->tb_p0, lr * u->trunk_lr, b1, b2, eps, wd, step);
        nn_adamw(u->p + u->tb_p0, u->g + u->tb_p0, u->m + u->tb_p0, u->v + u->tb_p0, u->np - u->tb_p0, lr, b1, b2, eps, wd, step);
    } else nn_adamw(u->p, u->g, u->m, u->v, u->np, lr, b1, b2, eps, wd, step);
}
static void for_each_conv3(unet *u, void (*fn)(unet *, const convp *, void *), void *arg);
static void for_each_mat(unet *u, void (*fn)(unet *, const convp *, void *), void *arg);   /* the Muon matrices: the 3^3 convs + the attention linears */
/* Conv traversal follows graph order, not parameter offsets. Sort weight intervals before
   updating the complementary biases, norms and head, so each parameter has one owner. */
typedef struct { const convp *c[UNET_MAXCONV]; int n; } conv_list;
static void collect_conv(unet *u, const convp *c, void *arg) {
    (void)u; conv_list *a = arg; a->c[a->n++] = c;
}
static void plain_ranges(unet *u, void (*fn)(unet *, size_t, size_t, void *), void *arg) {
    conv_list a = {0}; if (u->wq) for_each_conv3(u, collect_conv, &a); else for_each_mat(u, collect_conv, &a);   /* (packed weights: the 3^3 convs only) */
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
static int trunk_lr_on(const unet *u) { return u->tbn && u->trunk_lr > 0.f && u->trunk_lr != 1.f; }
static void plain_adamw(unet *u, size_t off, size_t n, void *arg) {
    adam_arg *a = arg;
    if (trunk_lr_on(u) && off < u->tb_p0) {   /* the trunk part at trunk_lr x lr, the stage's part (if the range reaches it) at lr */
        const size_t nt = off + n <= u->tb_p0 ? n : u->tb_p0 - off;
        nn_adamw(u->p + off, u->g + off, u->m + off, u->v + off, nt, a->lr * u->trunk_lr, a->b1, a->b2, a->eps, a->wd, a->step);
        off += nt; n -= nt;
        if (!n) return;
    }
    nn_adamw(u->p + off, u->g + off, u->m + off, u->v + off, n, a->lr, a->b1, a->b2, a->eps, a->wd, a->step);
}
void unet_set_trunk_lr(unet *u, float mult) { u->trunk_lr = mult; }
static void adamw_nonconv(unet *u, float lr, float b1, float b2, float eps, float wd, int step) {
    adam_arg a = {lr, b1, b2, eps, wd * u->wd_nonconv, step}; plain_ranges(u, plain_adamw, &a);   /* GroupNorm gains decay toward 0, not 1 */
}
static void plain_ema(unet *u, size_t off, size_t n, void *arg) { nn_ema(u->ema + off, u->p + off, n, *(float *)arg); }
/* Muon for the 3^3 conv weights (viewed as [Co][Ci * 27]); biases, GroupNorm parameters and the head stay on AdamW.
   Exclude these weights from AdamW: zero gradients still allow decay and stale moments to change them. */
typedef struct { float lr, beta, wd; } muon_arg;
static void muon_conv(unet *u, const convp *c, void *arg) {
    muon_arg *a = arg; int K = c->cin * c->k * c->k * c->k; size_t n = (size_t)c->cout * K;
    size_t need = 2 * n + 2 * (size_t)c->cout * c->cout;
    if (u->muon_work_n < need) { if (u->muon_work) nn_free(u->muon_work); u->muon_work = nn_malloc(need * 4); u->muon_work_n = need; }
    const float lr = trunk_lr_on(u) && c->w < u->tb_p0 ? a->lr * u->trunk_lr : a->lr;
    nn_muon(u->p + c->w, u->g + c->w, u->muon_mom + c->w, c->cout, K, lr, a->beta, a->wd, u->muon_work);
}
typedef struct { float *p; const float *g; float *mom, *X, *Y, *A, *B; int Co, K; } muon_desc_t;   /* mirrors nn.cu */
typedef struct { muon_desc_t h[UNET_MAXCONV]; int n; size_t pool; int maxco, maxk; } muon_build_t;
static void muon_collect(unet *u, const convp *c, void *arg) {
    muon_build_t *b = arg; int K = c->cin * c->k * c->k * c->k; size_t n = (size_t)c->cout * K;
    muon_desc_t *d = &b->h[b->n++]; d->p = u->p + c->w; d->g = u->g + c->w; d->mom = u->muon_mom + c->w; d->Co = c->cout; d->K = K;
    d->X = (float *)(uintptr_t)b->pool; b->pool += n; d->Y = (float *)(uintptr_t)b->pool; b->pool += n;
    d->A = (float *)(uintptr_t)b->pool; b->pool += (size_t)c->cout * c->cout; d->B = (float *)(uintptr_t)b->pool; b->pool += (size_t)c->cout * c->cout;
    if (c->cout > b->maxco) b->maxco = c->cout;
    if (K > b->maxk) b->maxk = K;
}
static void muon_table(unet *u, muon_build_t *b) {   /* the descriptor table and its scratch pool, built once (offsets first, then pointers) */
    for_each_mat(u, muon_collect, b);
    if (u->muon_descs) return;
    float *pool = nn_malloc(b->pool * 4);
    for (int i = 0; i < b->n; i++) { muon_desc_t *d = &b->h[i]; d->X = pool + (uintptr_t)d->X; d->Y = pool + (uintptr_t)d->Y; d->A = pool + (uintptr_t)d->A; d->B = pool + (uintptr_t)d->B; }
    u->muon_descs = nn_malloc(sizeof(muon_desc_t) * (size_t)b->n); nn_h2d(u->muon_descs, b->h, sizeof(muon_desc_t) * (size_t)b->n);
    u->muon_nconv = b->n; u->muon_maxco = b->maxco; u->muon_maxk = b->maxk; u->muon_work = pool;
}
/* ---- Muon variants (train --muon-c 1 / --hyperball 1; see nn_muon_ext_batch in nn_opt.cu) ----
   Muon-C covers every Muon conv (all are 3^3; the paper routes every kernel larger than 1x1 through it, strided ones
   included): the stem's 16 x 4 and an encoder conv1's (2C) x C blocks are polarised as their conjugate transposes.
   Hyperball covers the convs whose output feeds a GroupNorm (the loss is then nearly invariant to their scale, as for the
   paper's pre-norm matrices): all but the down convs of a net without down_norm, which keep Muon's decay. Its learning rate
   is mx_hb_mult x the scheduled Muon lr; the default 1/sqrt(2) gives the first step Muon's relative length: Kaiming-
   initialised [Co][K] weights have |W|_F = sqrt(2 Co) and a Muon step lr sqrt(Co) (Co <= K for every conv here). */
typedef struct { const float *U; float *O; float *EX, *EY; int Co, Ci, k, m, n, tr; } muonc_desc_t;   /* mirrors nn_opt.cu */
void unet_set_muon_ext(unet *u, int muonc, int hyperball, float hb_mult) { u->mx_c = muonc; u->mx_hb = hyperball; u->mx_hb_mult = hb_mult; }
typedef struct { int i; int hb[UNET_MAXCONV]; int on; } hb_build_t;
static void hb_collect(unet *u, const convp *c, void *arg) {
    hb_build_t *b = arg; int is_down = 0;
    for (int i = 0; i < u->cfg.nlev - 1; i++) is_down |= c == &u->down[i];
    b->hb[b->i++] = b->on && !(is_down && !u->cfg.down_norm);
}
static void muon_ext(unet *u, const muon_build_t *b, float lr, float beta, float wd) {
    const int n = b->n;
    if (!u->mx_part) {
        u->mx_part = nn_malloc((size_t)32 * n * (2 + 14) * sizeof(double));
        hb_build_t h = {.on = u->mx_hb}; for_each_conv3(u, hb_collect, &h);
        u->mx_hbf = nn_malloc((size_t)n * sizeof(int)); nn_h2d(u->mx_hbf, h.hb, (size_t)n * sizeof(int));
        u->mx_R = nn_malloc((size_t)n * 4); nn_zero(u->mx_R, (size_t)n * 4); u->mx_R_ok = 0;
    }
    if (u->mx_c && !u->mx_mcd) {   /* Muon-C: per conv its 14 complex frequency blocks, per conv and frequency a Newton-Schulz entry */
        enum { NF = 14 };
        muonc_desc_t mc[UNET_MAXCONV]; muon_desc_t *ns = calloc((size_t)n * NF, sizeof *ns); size_t pool = 0;
        for (int i = 0; i < n; i++) {
            const muon_desc_t *d = &b->h[i]; const int Ci = d->K / 27, m = d->Co < Ci ? d->Co : Ci, nx = d->Co < Ci ? Ci : d->Co;
            if (Ci * 27 != d->K) { fprintf(stderr, "muon-c: conv %d is not 3^3\n", i); abort(); }
            mc[i] = (muonc_desc_t){.U = nullptr, .O = nullptr, .Co = d->Co, .Ci = Ci, .k = 3, .m = m, .n = nx, .tr = d->Co > Ci};
            mc[i].EX = (float *)(uintptr_t)pool; pool += (size_t)NF * 2 * m * nx; mc[i].EY = (float *)(uintptr_t)pool; pool += (size_t)NF * 2 * m * nx;
            for (int f = 0; f < NF; f++) {
                muon_desc_t *e = &ns[(size_t)i * NF + f];
                const uintptr_t fo = (uintptr_t)f * 2 * m * nx;   /* offsets in floats until the pool exists */
                *e = (muon_desc_t){.X = (float *)((uintptr_t)mc[i].EX + fo), .Y = (float *)((uintptr_t)mc[i].EY + fo), .Co = m, .K = nx};
                e->A = (float *)(uintptr_t)pool; pool += (size_t)2 * m * m; e->B = (float *)(uintptr_t)pool; pool += (size_t)2 * m * m;
            }
            if (m > u->mx_ns_maxm) u->mx_ns_maxm = m;
            if (nx > u->mx_ns_maxn) u->mx_ns_maxn = nx;
        }
        u->mx_pool = nn_malloc(pool * 4);
        const muon_desc_t *dev = u->muon_descs; muon_desc_t host[UNET_MAXCONV]; nn_d2h(host, dev, sizeof(muon_desc_t) * (size_t)n);
        for (int i = 0; i < n; i++) {
            mc[i].EX = u->mx_pool + (uintptr_t)mc[i].EX; mc[i].EY = u->mx_pool + (uintptr_t)mc[i].EY; mc[i].U = host[i].X; mc[i].O = host[i].Y;
            for (int f = 0; f < NF; f++) {
                muon_desc_t *e = &ns[(size_t)i * NF + f];
                e->X = u->mx_pool + (uintptr_t)e->X; e->Y = u->mx_pool + (uintptr_t)e->Y; e->A = u->mx_pool + (uintptr_t)e->A; e->B = u->mx_pool + (uintptr_t)e->B;
            }
        }
        u->mx_mcd = nn_malloc(sizeof(muonc_desc_t) * (size_t)n); nn_h2d(u->mx_mcd, mc, sizeof(muonc_desc_t) * (size_t)n);
        u->mx_nsd = nn_malloc(sizeof(muon_desc_t) * (size_t)n * NF); nn_h2d(u->mx_nsd, ns, sizeof(muon_desc_t) * (size_t)n * NF);
        free(ns);
    }
    int r_init = 0;
    if (u->mx_hb) {
        if (u->mx_R_nload) {   /* radii from the checkpoint */
            if (u->mx_R_nload == n) { nn_h2d(u->mx_R, u->mx_R_load, (size_t)n * 4); u->mx_R_ok = 1; }
            else fprintf(stderr, "hyperball: the checkpoint has %d radii for %d convs; radii from the current weights\n", u->mx_R_nload, n);
            u->mx_R_nload = 0;
        }
        if (!u->mx_R_ok) { nn_zero(u->mx_R, (size_t)n * 4); r_init = 1; u->mx_R_ok = 1; }
    }
    nn_muon_ext_batch(u->muon_descs, n, u->muon_maxco, u->muon_maxk, u->mx_c ? u->mx_mcd : nullptr, u->mx_nsd, 3, u->mx_ns_maxm, u->mx_ns_maxn,
                      u->mx_hb ? u->mx_hbf : nullptr, u->mx_R, r_init, u->mx_part, lr, beta, wd, u->mx_hb_mult * lr);
}
void unet_muon(unet *u, float lr_muon, float beta, float lr_adam, float b1, float b2, float eps, float wd, int step) {
    if (u->wq) { unet_adamw(u, lr_adam, b1, b2, eps, wd, step); return; }   /* packed weights: AdamW only */
    if (!u->muon_mom) { u->muon_mom = nn_malloc(u->np * 4); nn_zero(u->muon_mom, u->np * 4); }
    if (u->tbn && (u->mx_c || u->mx_hb)) { fprintf(stderr, "unet: Muon-C / Hyperball do not cover the attention blocks' linear layers\n"); abort(); }
    if (u->mx_c || u->mx_hb) {
        muon_build_t b = {0}; muon_table(u, &b);
        muon_ext(u, &b, lr_muon, beta, wd);
    } else if (ufsm_env_on("UFSM_MUON_UNBATCHED")) {   /* reference path: one conv at a time */
        muon_arg a = {lr_muon, beta, wd};
        for_each_mat(u, muon_conv, &a);
    } else {
        if (!u->muon_descs) { muon_build_t b = {0}; muon_table(u, &b); }
        if (trunk_lr_on(u)) {   /* the table is the trunk's convs, then the stage's 4 x tbn linears (for_each_mat): two batches */
            const int na = 4 * u->tbn, nt = u->muon_nconv - na;
            nn_muon_batch(u->muon_descs, nt, u->muon_maxco, u->muon_maxk, lr_muon * u->trunk_lr, beta, wd);
            nn_muon_batch((const char *)u->muon_descs + sizeof(muon_desc_t) * (size_t)nt, na, u->muon_maxco, u->muon_maxk, lr_muon, beta, wd);
        } else nn_muon_batch(u->muon_descs, u->muon_nconv, u->muon_maxco, u->muon_maxk, lr_muon, beta, wd);
    }
    adamw_nonconv(u, lr_adam, b1, b2, eps, wd, step);
}
/* ANVIL II on the 3^3 conv weights (see nn.cu); AdamW elsewhere. step counts from 1; rail schedule as in the nanogpt record. */
typedef struct { float *p; const float *g; float *v0, *X, *Y, *A, *B, *v1, *E, *R; int Co, K; } anvil_desc_t;
typedef struct { anvil_desc_t h[UNET_MAXCONV]; int n; size_t pool; int maxco, maxk; } anvil_build_t;
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
        for_each_mat(u, anvil_collect, &b);
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
void unet_ema(unet *u, float decay) {
    if (u->wq) wq_ema(u, decay); else nn_ema(u->ema, u->p, u->np, decay);
    if (u->cfg.l0_norm == 2 && u->l0_n) { const size_t G = (size_t)G_of(u, u->cfg.widths[0]); nn_ema(u->l0run + 8 * G, u->l0run + 16 * G, 8 * G, decay); }   /* the EMA weights' statistics */
}
const float *unet_live_params(const unet *u) { return u->p; }
/* zero every gradient but the output head's (weights and biases): train only the head on frozen features */
void unet_grad_head_only(unet *u, int blocks) {   /* blocks: also train the last decoder blocks dec[blocks-1] .. dec[0] */
    const int k = blocks < 0 ? 0 : blocks > u->cfg.nlev - 1 ? u->cfg.nlev - 1 : blocks;
    const size_t hb = k ? u->dec[k - 1].c1.w : u->head.w, he = u->head.b + (size_t)u->cfg.cout;
    if (hb) nn_zero(u->g, hb * 4);
    if (he < u->np) nn_zero(u->g + he, (u->np - he) * 4);
}
/* the forward reads live (unet_apply_sparse24 resets fw from it every call), so the swap goes through live */
void unet_use_params(unet *u, const float *w) {
    nn_wmemo_clear();
    u->live = w ? (float *)w : u->using_ema ? u->ema : u->p;
    if (!u->sparse24) u->fw = u->live;
}
/* GroupNorm statistic sites in a fixed order: enc[0..L-1] (m1 r1 m2 r2), down norms (dm dr), dec[L-2..0] (m1 r1 m2 r2); each
   holds n x G floats. fn gets the device array and its group count. */
static size_t gn_sites(unet *u, void (*fn)(unet *, float *, int, size_t, void *), void *arg) {
    size_t off = 0; const int L = u->cfg.nlev;
#define GN_SITE(ptr, g) do { if (fn) fn(u, ptr, g, off, arg); off += (size_t)(g); } while (0)
    for (int i = 0; i < L; i++) { block *b = &u->enc[i]; GN_SITE(b->m1, G_of(u, b->c1.cout)); GN_SITE(b->r1, G_of(u, b->c1.cout)); GN_SITE(b->m2, G_of(u, b->c2.cout)); GN_SITE(b->r2, G_of(u, b->c2.cout)); }
    if (u->cfg.down_norm) for (int i = 0; i < L - 1; i++) { GN_SITE(u->dm[i], G_of(u, u->cfg.widths[i])); GN_SITE(u->dr[i], G_of(u, u->cfg.widths[i])); }
    for (int i = L - 2; i >= 0; i--) { block *b = &u->dec[i]; GN_SITE(b->m1, G_of(u, b->c1.cout)); GN_SITE(b->r1, G_of(u, b->c1.cout)); GN_SITE(b->m2, G_of(u, b->c2.cout)); GN_SITE(b->r2, G_of(u, b->c2.cout)); }
    for (int i = 0; i < L; i++) for (int k = 0; k < u->cfg.enc_blocks[i]; k++) { block *b = &u->rb[i][k]; GN_SITE(b->m1, G_of(u, b->c1.cout)); GN_SITE(b->r1, G_of(u, b->c1.cout)); GN_SITE(b->m2, G_of(u, b->c2.cout)); GN_SITE(b->r2, G_of(u, b->c2.cout)); }
    if (u->tbn) { GN_SITE(u->tb.m2, G_of(u, u->tbd.C)); GN_SITE(u->tb.r2, G_of(u, u->tbd.C)); }   /* (the attention stage's output) */
#undef GN_SITE
    return off;
}
size_t unet_gn_nstats(unet *u) { return gn_sites(u, nullptr, nullptr); }
static void gn_get1(unet *u, float *d, int g, size_t off, void *h) { (void)u; nn_d2h((float *)h + off, d, (size_t)g * 4); }
int unet_gn_stats_get(unet *u, float *h) {   /* sample 0 of the last forward */
    if (!u->built || !nn_get_tf32()) return -1;
    gn_sites(u, gn_get1, h);
    return 0;
}
static void gn_put1(unet *u, float *d, int g, size_t off, void *arg) {
    (void)arg;
    for (int n = 0; n < u->xs.n; n++) nn_h2d(d + (size_t)n * g, u->gn_frz + off, (size_t)g * 4);
}
static void gn_frozen_load(unet *u) { gn_sites(u, gn_put1, nullptr); }
/* pooling over windows: sites come in (mean, rstd) pairs of equal group count; the pooled variance is the mean within-window
   variance plus the variance of the window means */
typedef struct { const float *t; double *acc; float *out; double nw; int k; } gn_pool_t;
static void gn_pool1(unet *u, float *d, int g, size_t off, void *arg) {
    (void)u; (void)d; gn_pool_t *q = arg;
    if (q->k++ & 1) return;   /* the rstd site of the pair: handled with its mean */
    const size_t n = unet_gn_nstats(u);
    for (int i = 0; i < g; i++) {
        const size_t im = off + (size_t)i, ir = off + (size_t)g + i;
        if (q->t) {
            const double m = q->t[im], r = q->t[ir];
            q->acc[im] += m; q->acc[n + im] += m * m; q->acc[ir] += 1.0 / (r * r) - 1e-5;
        } else {
            const double m = q->acc[im] / q->nw, v = q->acc[ir] / q->nw + q->acc[n + im] / q->nw - m * m;
            q->out[im] = (float)m; q->out[ir] = (float)(1.0 / sqrt((v > 0 ? v : 0) + 1e-5));
        }
    }
}
void unet_gn_accumulate(unet *u) {
    const size_t n = unet_gn_nstats(u);
    if (!u->gn_acc) { u->gn_acc = calloc(2 * n, sizeof(double)); u->gn_acc_n = 0; }
    float *t = malloc(n * 4);
    if (!unet_gn_stats_get(u, t)) { gn_pool_t q = {t, u->gn_acc, nullptr, 0, 0}; gn_sites(u, gn_pool1, &q); u->gn_acc_n++; }
    free(t);
}
long unet_gn_pooled(unet *u, float *h) {
    if (!u->gn_acc || !u->gn_acc_n) return 0;
    gn_pool_t q = {nullptr, u->gn_acc, h, (double)u->gn_acc_n, 0}; gn_sites(u, gn_pool1, &q);
    return u->gn_acc_n;
}
void unet_gn_freeze(unet *u, const float *h) {
    free(u->gn_frz); u->gn_frz = nullptr; u->gn_frz_n = 0;
    if (!h) return;
    u->gn_frz_n = unet_gn_nstats(u);
    u->gn_frz = malloc(u->gn_frz_n * 4);
    memcpy(u->gn_frz, h, u->gn_frz_n * 4);
}
void unet_blend_ema(unet *u, float a) { nn_scale(u->ema, 1.f - a, u->np); nn_axpy(u->ema, a, u->p, u->np); nn_wmemo_clear(); }
const float *unet_ema_params(const unet *u) { return u->ema; }
void unet_use_ema(unet *u, int on) { nn_wmemo_clear(); u->live = on ? u->ema : u->p; u->using_ema = on; if (!u->sparse24) u->fw = u->live; }
const float *unet_fwd_params(const unet *u) { return u->fw; }
/* ---- 2:4 structured sparsity: the forward reads a masked copy of the live weights; SR-STE keeps the master weights dense ---- */
static void for_each_conv3(unet *u, void (*fn)(unet *, const convp *, void *), void *arg) {
    int L = u->cfg.nlev;
    for (int i = 0; i < L; i++) { fn(u, &u->enc[i].c1, arg); fn(u, &u->enc[i].c2, arg); }
    for (int i = 0; i < L - 1; i++) { fn(u, &u->down[i], arg); fn(u, &u->dec[i].c1, arg); if (!u->dec[i].single) fn(u, &u->dec[i].c2, arg); }
    for (int i = 0; i < L; i++) for (int k = 0; k < u->cfg.enc_blocks[i]; k++) { fn(u, &u->rb[i][k].c1, arg); fn(u, &u->rb[i][k].c2, arg); }
}
static void for_each_mat(unet *u, void (*fn)(unet *, const convp *, void *), void *arg) {
    for_each_conv3(u, fn, arg);
    for (int k = 0; k < u->tbn; k++) for (int j = 0; j < 4; j++) fn(u, &u->tbm[k][j], arg);
}
static void mask_conv(unet *u, const convp *c, void *arg) { (void)arg; nn_mask24(u->wm + c->w, u->wm + c->w, c->cout, c->cin, c->k * c->k * c->k); }
static void srste_conv(unet *u, const convp *c, void *arg) { nn_srste24(u->g + c->w, u->live + c->w, c->cout, c->cin, c->k * c->k * c->k, *(float *)arg); }
void unet_set_sparse24(unet *u, int on) {
    u->sparse24 = on;
    if (on && !u->wm) u->wm = nn_malloc(u->np * 4);
    u->fw = on ? u->wm : u->live;
}
int unet_get_sparse24(const unet *u) { return u->sparse24; }
static void fq2d_conv(unet *u, const convp *c, void *arg) {   /* cfg.w4_2d: the fp4 stride-1 convs (not the stem, not the downs) of the
                                                                  forward copy arg (u->wm, or u->wsp after the ws standardisation) */
    if (c == &u->enc[0].c1 || c->stride != 1) return;
    nn_fakeq_w4_2d((float *)arg + c->w, c->cout, c->cin, c->k * c->k * c->k, 32);
}
/* cfg.l0_norm 3 (ws): the level-0 3^3 convs (the stem; the others read SiLU outputs: NF-net gain) */
static int ws_convs(const unet *u, const convp **c, float *gain) {
    int n = 0;
    c[n] = &u->enc[0].c1; gain[n++] = 1.f;
    c[n] = &u->enc[0].c2; gain[n++] = u->ws_gain;
    if (u->cfg.nlev > 1) { c[n] = &u->dec[0].c1; gain[n++] = u->ws_gain; if (!u->dec[0].single) { c[n] = &u->dec[0].c2; gain[n++] = u->ws_gain; } }
    return n;
}
#define WS_EPS 1e-5f
static void ws_backward(unet *u) {   /* d/dw through the standardisation (the gradient was zeroed before this backward) */
    const convp *c[4]; float ga[4]; const int n = ws_convs(u, c, ga);
    for (int i = 0; i < n; i++) nn_ws_bwd(u->g + c[i]->w, nullptr, u->live + c[i]->w, c[i]->cout, c[i]->cin * c[i]->k * c[i]->k * c[i]->k, ga[i], WS_EPS);
}
void unet_apply_sparse24(unet *u) {   /* (and cfg.w4_2d / cfg.l0_norm ws: the forward copy of the live weights, transformed every forward) */
    if (u->cfg.l0_norm == 3) {   /* ws: a copy of the live weights with the level-0 convs' standardised (every forward: the live / EMA switch) */
        if (u->sparse24 || u->wq) { fprintf(stderr, "unet: --l0-norm ws needs dense fp32 master weights (no 2:4 sparsity, no packed weights)\n"); abort(); }
        nn_d2d(u->wsp, u->live, u->np * 4);
        const convp *c[4]; float ga[4]; const int n = ws_convs(u, c, ga);
        for (int i = 0; i < n; i++) nn_ws_fwd(u->wsp + c[i]->w, u->live + c[i]->w, c[i]->cout, c[i]->cin * c[i]->k * c[i]->k * c[i]->k, ga[i], WS_EPS);
        if (u->cfg.w4_2d) for_each_conv3(u, fq2d_conv, u->wsp);
        u->fw = u->wsp;
        return;
    }
    if (!u->sparse24 && !u->cfg.w4_2d) { u->fw = u->live; return; }
    if (!u->wm) u->wm = nn_malloc(u->np * 4);
    nn_d2d(u->wm, u->live, u->np * 4);
    if (u->sparse24) for_each_conv3(u, mask_conv, nullptr);
    if (u->cfg.w4_2d) for_each_conv3(u, fq2d_conv, u->wm);
    u->fw = u->wm;
}
void unet_srste24(unet *u, float lambda) { if (u->sparse24) for_each_conv3(u, srste_conv, &lambda); }
/* ---- true fp8 / fp4 weights: packed storage, optimizer and EMA act on the packed values ---- */
void unet_set_wq(unet *u, int bits) {
    u->wq = bits;
    if (!bits) return;
    if (!u->wq_q) {
        /* offsets: packed weights use the element offset (fp8: bytes; fp4: element/2, offsets are even), scales are cumulative */
        int L = u->cfg.nlev; const convp *cs[UNET_MAXCONV]; int n = 0;
        for (int i = 0; i < L; i++) { cs[n++] = &u->enc[i].c1; cs[n++] = &u->enc[i].c2; }
        for (int i = 0; i < L - 1; i++) { cs[n++] = &u->down[i]; cs[n++] = &u->dec[i].c1; cs[n++] = &u->dec[i].c2; }
        for (int i = 0; i < L; i++) for (int k = 0; k < u->cfg.enc_blocks[i]; k++) { cs[n++] = &u->rb[i][k].c1; cs[n++] = &u->rb[i][k].c2; }
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
    for (int j = 0; j < L; j++) for (int k = 0; k < u->cfg.enc_blocks[j]; k++) { if (n++ == i) return &u->rb[j][k].c1; if (n++ == i) return &u->rb[j][k].c2; }
    return nullptr;
}
/* optimizer step with packed weights: AdamW on the packed conv weights, the plain fp32 AdamW on everything in between */
static void wq_adamw(unet *u, float lr, float b1, float b2, float eps, float wd, int step) {
    adamw_nonconv(u, lr, b1, b2, eps, wd, step);
    for (int i = 0; i < u->wq_n; i++) {
        const convp *c = wq_conv(u, i); int T = c->k * c->k * c->k; size_t qo = u->wq_qoff[i];
        nn_wq_adamw(u->wq_q + qo, u->wq_sc + u->wq_soff[i], u->wq_r ? u->wq_r + c->w : nullptr, u->wq_r ? u->wq_rsc + u->wq_soff[i] : nullptr, u->g + c->w, u->m + c->w, u->v + c->w, c->cout, c->cin, T, u->wq,
                    trunk_lr_on(u) ? lr * u->trunk_lr : lr, b1, b2, eps, wd, step, (unsigned)step * 7919u + (unsigned)i);   /* (the packed convs: all trunk) */
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
static float g_loaded_hbR[UNET_MAXCONV]; static int g_loaded_hbR_n = 0;   /* Hyperball radii of the header */
static long g_loaded_l0_n; static int g_loaded_l0_warm, g_loaded_l0_cnt; static float g_loaded_l0_mom, g_loaded_l0_run[16 * 256];   /* cfg.l0_norm ema */
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
/* debug: per block, the gradient norm of each 3^3 conv weight (c1, c2) and of the attention blocks' linears; the head */
void unet_debug_grads(const unet *u, const char *tag) {
    const int L = u->cfg.nlev; float *h = malloc(u->np * 4); nn_d2h(h, u->g, u->np * 4);
    #define GN_(off, n) ({ double a_ = 0; for (size_t k_ = 0; k_ < (size_t)(n); k_++) a_ += (double)h[(off) + k_] * h[(off) + k_]; sqrt(a_); })
    fprintf(stderr, "grads %s:", tag);
    for (int i = 0; i < 2 * L - 1; i++) {
        const block *b = i < L ? &u->enc[i] : &u->dec[2 * L - 2 - i]; const int li = i < L ? i : 2 * L - 2 - i;
        const size_t n1 = (size_t)b->c1.cout * b->c1.cin * 27, n2 = (size_t)b->c2.cout * b->c2.cin * 27;
        fprintf(stderr, " %s%d %.2e/%.2e", i < L ? "e" : "d", li, GN_(b->c1.w, n1), GN_(b->c2.w, n2));
    }
    for (int k = 0; k < u->tbn; k++) for (int m = 0; m < 4; m++) { const convp *c = &u->tbm[k][m]; fprintf(stderr, " t%d.%d %.2e", k, m, GN_(c->w, (size_t)c->cin * c->cout)); }
    fprintf(stderr, " head %.2e\n", GN_(u->head.w, (size_t)u->head.cin * u->head.cout));
    #undef GN_
    free(h);
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
    fputc(']', f);
    { int any = 0; for (int i = 0; i < u->cfg.nlev; i++) any |= u->cfg.dec_widths[i] != 0;   /* only when set: older readers see the old header */
      if (any) { fputs(",\"dec_widths\":[", f); for (int i = 0; i < u->cfg.nlev; i++) fprintf(f, "%s%d", i ? "," : "", u->cfg.dec_widths[i]); fputc(']', f); } }
    { int any = 0; for (int i = 0; i < u->cfg.nlev; i++) any |= u->cfg.enc_blocks[i] != 0;
      if (any) { fputs(",\"enc_blocks\":[", f); for (int i = 0; i < u->cfg.nlev; i++) fprintf(f, "%s%d", i ? "," : "", u->cfg.enc_blocks[i]); fputc(']', f); } }
    if (u->cfg.half_res) fprintf(f, ",\"half_res\":%d", u->cfg.half_res);
    if (u->cfg.w4_2d) fprintf(f, ",\"w4_2d\":%d", u->cfg.w4_2d);
    if (u->cfg.ds_levels) fprintf(f, ",\"ds_levels\":%d", u->cfg.ds_levels);
    if (u->cfg.attn_blocks) fprintf(f, ",\"attn_blocks\":%d,\"attn_pe\":%d,\"attn_reg\":%d,\"attn_mlp\":%d", u->cfg.attn_blocks, u->cfg.attn_pe, u->cfg.attn_reg, u->tbd.F);
    if (u->cfg.attn_vres) fprintf(f, ",\"attn_vres\":%d", u->cfg.attn_vres);
    if (u->cfg.attn_blocks && u->cfg.attn_axis) fprintf(f, ",\"attn_axis\":%d", u->cfg.attn_axis);
    if (u->cfg.attn_blocks && u->cfg.attn_repeat > 1) fprintf(f, ",\"attn_repeat\":%d", u->cfg.attn_repeat);
    if (u->cfg.attn_blocks && u->cfg.attn_mem) fprintf(f, ",\"attn_mem\":%d", u->cfg.attn_mem);
    if (u->cfg.attn_blocks && u->cfg.view_P) fprintf(f, ",\"view_P\":%d,\"view_patch\":%d", u->cfg.view_P, u->cfg.view_patch);
    if (u->cfg.attn_blocks && u->cfg.attn_deform) fprintf(f, ",\"attn_deform\":%d,\"attn_dlev\":%d", u->cfg.attn_deform, u->tbd.dL);
    if (u->cfg.attn_blocks && u->cfg.attn_deform && u->cfg.attn_dco) fprintf(f, ",\"attn_dco\":%d", u->cfg.attn_dco);
    if (u->cfg.attn_blocks && u->cfg.attn_deform && u->cfg.attn_dinit) fprintf(f, ",\"attn_dinit\":%d", u->cfg.attn_dinit);
    if (u->cfg.l0_norm) fprintf(f, ",\"l0_norm\":%d", u->cfg.l0_norm);
    if (u->cfg.l0_norm == 4) fprintf(f, ",\"l0_slab\":%d", u->cfg.l0_slab);
    if (u->cfg.l0_norm == 2) {   /* the running statistics (live weights, EMA weights; l0_site order, [mean G | var G] each) and their state */
        const int n = 16 * G_of(u, u->cfg.widths[0]); float *r = malloc((size_t)n * 4); nn_d2h(r, u->l0run, (size_t)n * 4);
        fprintf(f, ",\"l0_n\":%ld,\"l0_warm\":%d,\"l0_mom\":%.9g,\"l0_run\":[", u->l0_n, u->l0_warm, u->l0_mom);
        for (int i = 0; i < n; i++) fprintf(f, "%s%.9g", i ? "," : "", r[i]);
        fputc(']', f); free(r);
    }
    { int any = 0; for (int i = 0; i < u->cfg.nlev; i++) any |= u->cfg.dec_convs[i] == 1;
      if (any) { fputs(",\"dec_convs\":[", f); for (int i = 0; i < u->cfg.nlev; i++) fprintf(f, "%s%d", i ? "," : "", u->cfg.dec_convs[i]); fputc(']', f); } }
    fprintf(f, ",\"cin\":%d,\"cout\":%d,\"G\":%d,\"down_norm\":%d,\"scale_cond\":%d,\"nparams\":%zu,\"step\":%d,\"sparse24\":%d,\"wq\":%d,\"muon_mom\":%d", u->cfg.cin, u->cfg.cout, u->cfg.G, u->cfg.down_norm, u->cfg.scale_cond, u->np, step, u->sparse24, u->wq, u->muon_mom != nullptr);
    if (u->mx_hb && (u->mx_R_nload || (u->mx_R && u->mx_R_ok))) {   /* Hyperball radii (Muon table order): a resumed run keeps its spheres */
        float r[UNET_MAXCONV]; const int nr = u->mx_R_nload ? u->mx_R_nload : u->muon_nconv;
        if (u->mx_R_nload) memcpy(r, u->mx_R_load, (size_t)nr * 4); else nn_d2h(r, u->mx_R, (size_t)nr * 4);
        fputs(",\"hb_R\":[", f); for (int i = 0; i < nr; i++) fprintf(f, "%s%.9g", i ? "," : "", r[i]); fputc(']', f);
    }
    fprintf(f, ",\"extra\":%s}\n", extra ? extra : "{}");
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
    if ((p = strstr(line, "\"scale_cond\":"))) cfg->scale_cond = atoi(p + 13);
    memset(cfg->dec_widths, 0, sizeof cfg->dec_widths);
    if ((p = strstr(line, "\"dec_widths\":["))) { p += 14; for (int i = 0; i < UNET_MAXLEV && *p && *p != ']'; i++) { cfg->dec_widths[i] = (int)strtol(p, (char **)&p, 10); if (*p == ',') p++; } }
    if ((p = strstr(line, "\"enc_blocks\":["))) { p += 14; for (int i = 0; i < UNET_MAXLEV && *p && *p != ']'; i++) { cfg->enc_blocks[i] = (int)strtol(p, (char **)&p, 10); if (*p == ',') p++; } }
    if ((p = strstr(line, "\"half_res\":"))) cfg->half_res = atoi(p + 11);
    cfg->w4_2d = (p = strstr(line, "\"w4_2d\":")) ? atoi(p + 8) : 0;
    if ((p = strstr(line, "\"ds_levels\":"))) cfg->ds_levels = atoi(p + 12);
    if ((p = strstr(line, "\"attn_blocks\":"))) cfg->attn_blocks = atoi(p + 14);
    if ((p = strstr(line, "\"attn_pe\":"))) cfg->attn_pe = atoi(p + 10);
    if ((p = strstr(line, "\"attn_reg\":"))) cfg->attn_reg = atoi(p + 11);
    if ((p = strstr(line, "\"attn_mlp\":"))) cfg->attn_mlp = atoi(p + 11);
    cfg->attn_vres = (p = strstr(line, "\"attn_vres\":")) ? atoi(p + 12) : 0;
    if ((p = strstr(line, "\"attn_axis\":"))) cfg->attn_axis = atoi(p + 12);
    if ((p = strstr(line, "\"attn_repeat\":"))) cfg->attn_repeat = atoi(p + 14);
    if ((p = strstr(line, "\"attn_mem\":"))) cfg->attn_mem = atoi(p + 11);
    if ((p = strstr(line, "\"view_P\":"))) cfg->view_P = atoi(p + 9);
    if ((p = strstr(line, "\"view_patch\":"))) cfg->view_patch = atoi(p + 13);
    if ((p = strstr(line, "\"attn_deform\":"))) cfg->attn_deform = atoi(p + 14);
    if ((p = strstr(line, "\"attn_dlev\":"))) cfg->attn_dlev = atoi(p + 12);
    cfg->attn_dco = (p = strstr(line, "\"attn_dco\":")) ? atoi(p + 11) : 0;
    cfg->attn_dinit = (p = strstr(line, "\"attn_dinit\":")) ? atoi(p + 13) : 0;
    if ((p = strstr(line, "\"dec_convs\":["))) { p += 13; for (int i = 0; i < UNET_MAXLEV && *p && *p != ']'; i++) { cfg->dec_convs[i] = (int)strtol(p, (char **)&p, 10); if (*p == ',') p++; } }
    if ((p = strstr(line, "\"nparams\":"))) *np = (size_t)atoll(p + 10);
    if ((p = strstr(line, "\"step\":"))) *step = atoi(p + 7);
    if ((p = strstr(line, "\"sparse24\":"))) g_loaded_sparse = atoi(p + 11);
    if ((p = strstr(line, "\"wq\":"))) g_loaded_wq = atoi(p + 5);
    if ((p = strstr(line, "\"muon_mom\":"))) g_loaded_muon = atoi(p + 11);
    g_loaded_hbR_n = 0;
    if ((p = strstr(line, "\"hb_R\":["))) { p += 8; while (g_loaded_hbR_n < UNET_MAXCONV && *p && *p != ']') { g_loaded_hbR[g_loaded_hbR_n++] = strtof(p, (char **)&p); if (*p == ',') p++; } }
    if ((p = strstr(line, "\"l0_norm\":"))) cfg->l0_norm = atoi(p + 10);
    if ((p = strstr(line, "\"l0_slab\":"))) cfg->l0_slab = atoi(p + 10);
    g_loaded_l0_cnt = 0;
    if ((p = strstr(line, "\"l0_n\":"))) g_loaded_l0_n = atol(p + 7);
    if ((p = strstr(line, "\"l0_warm\":"))) g_loaded_l0_warm = atoi(p + 10);
    if ((p = strstr(line, "\"l0_mom\":"))) g_loaded_l0_mom = strtof(p + 9, nullptr);
    if ((p = strstr(line, "\"l0_run\":["))) { p += 10; while (g_loaded_l0_cnt < 16 * 256 && *p && *p != ']') { g_loaded_l0_run[g_loaded_l0_cnt++] = strtof(p, (char **)&p); if (*p == ',') p++; } }
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
    nn_wmemo_clear();
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
    u->mx_R_nload = g_loaded_hbR_n; memcpy(u->mx_R_load, g_loaded_hbR, (size_t)g_loaded_hbR_n * 4); u->mx_R_ok = 0;   /* none: from the weights */
    if (g_loaded_sparse) unet_set_sparse24(u, 1);
    if (g_loaded_wq) unet_set_wq(u, g_loaded_wq);
    if (u->cfg.l0_norm == 2) {   /* the running statistics (a checkpoint without them: a fresh warm-up) */
        const int n = 16 * G_of(u, u->cfg.widths[0]);
        if (cfg.l0_norm == 2 && g_loaded_l0_cnt == n) { nn_h2d(u->l0run, g_loaded_l0_run, (size_t)n * 4); u->l0_n = g_loaded_l0_n; u->l0_warm = g_loaded_l0_warm; u->l0_mom = g_loaded_l0_mom; }
        else { nn_zero(u->l0run, (size_t)n * 4); u->l0_n = 0; }
    }
    return step;
}

/* Warm start into a wider head (band affinity outputs): load a checkpoint of the same net with fewer outputs. The body is
   copied as is, head rows < keep (and their biases) are kept, new rows get zero weights and bias new_bias (zero optimiser
   state), and the parameters after the head (down_norm GroupNorms) shift. Returns the checkpoint step or -1. */
int unet_load_grow(unet *u, const char *path, int keep, float new_bias) {
    nn_wmemo_clear();
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
    u->mx_R_nload = 0; u->mx_R_ok = 0;   /* a grown head: Hyperball radii from the weights */
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
/* deformable block k of a donor with another coarse-level / offset-frame setting (same C, F, dK, dL): every tensor but the offset / logit
   projection as is; of that the logit rows (and bias) of the common levels, the offset rows only with the same frames (else the fresh
   prior); the new coarse level's rows keep their initialisation */
static void dfm_copy_block(unet *u, const unet *o, int k, float *dst, const float *sp) {
    const at_dims *a = &u->tbd, *b = &o->tbd; const at_bp *pa = &u->tbp[k], *pb = &o->tbp[k];
    const size_t C = a->C, end_a = pa->ln1g + at_block_nparams(a), end_b = pb->ln1g + at_block_nparams(b);
    nn_d2d(dst + pa->ln1g, sp + pb->ln1g, (pa->woa - pa->ln1g) * 4);   /* LN 1, value projection */
    for (int l = 1; l < a->dL; l++) { nn_d2d(dst + pa->wvl[l], sp + pb->wvl[l], C * a->dC[l] * 4); nn_d2d(dst + pa->bvl[l], sp + pb->bvl[l], C * 4); }
    nn_d2d(dst + pa->wo, sp + pb->wo, (end_a - pa->wo) * 4);           /* out projection, LN 2, MLP */
    if (end_a - pa->wo != end_b - pb->wo) { fprintf(stderr, "unet: deformable donor layout mismatch\n"); abort(); }
    const int H = a->H, K = a->dK, La = at_dfm_nl(a), Lb = at_dfm_nl(b), Lc = a->dL;
    for (int h = 0; h < H; h++) for (int l = 0; l < Lc; l++) for (int q = 0; q < K; q++) {
        const size_t ia = ((size_t)h * La + l) * K + q, ib = ((size_t)h * Lb + l) * K + q, ga = (size_t)H * La * K * 3 + ia, gb = (size_t)H * Lb * K * 3 + ib;
        if (a->dInit == b->dInit) { nn_d2d(dst + pa->woa + ia * 3 * C, sp + pb->woa + ib * 3 * C, 3 * C * 4); nn_d2d(dst + pa->boa + ia * 3, sp + pb->boa + ib * 3, 3 * 4); }
        nn_d2d(dst + pa->woa + ga * C, sp + pb->woa + gb * C, C * 4); nn_d2d(dst + pa->boa + ga, sp + pb->boa + gb, 4);
    }
}
int unet_init_from(unet *u, const char *path) { return unet_init_from_map(u, path, nullptr); }
/* head_rows (u's cout entries, or nullptr = shape-matched copy): output row co of u's head takes the checkpoint's head row
   head_rows[co] (weights and bias), -1 = keep the fresh row. E.g. a verso warm start {0, 0, 1, 2, ...}: recto copied into the
   new verso channel 1, the old rows 1.. shifted up by one. */
/* the stem when the inputs grew (e.g. train --cond: 4 -> 7): the donor's weights for its inputs, zero for the new ones, so the net
   computes the donor's function whatever the new inputs hold */
static int copy_conv_in(unet *u, const unet *o, const convp *a, const convp *b) {
    if (a->cin < b->cin || a->cout != b->cout || a->k != b->k || a->stride != b->stride) return 0;
    const size_t k3 = (size_t)a->k * a->k * a->k, ra = (size_t)a->cin * k3, rb = (size_t)b->cin * k3;
    for (int e = 0; e < 2; e++) {
        float *dst = e ? u->ema : u->p; const float *sp = e ? o->ema : o->p;
        nn_zero(dst + a->w, (size_t)a->cout * ra * 4);
        for (int co = 0; co < a->cout; co++) nn_d2d(dst + a->w + co * ra, sp + b->w + co * rb, rb * 4);
        nn_d2d(dst + a->b, sp + b->b, (size_t)a->cout * 4);
    }
    return 1;
}
int unet_init_from_map(unet *u, const char *path, const int *head_rows) {
    unet_cfg oc; int step;
    if (unet_peek(path, &oc, &step) || oc.cin > u->cfg.cin) return -1;   /* (more inputs: the stem grows, copy_conv_in) */
    unet *o = unet_create(&oc);
    if (unet_load(o, path) < 0) { unet_free(o); return -1; }
    int n = 0;
    const int L = u->cfg.nlev < oc.nlev ? u->cfg.nlev : oc.nlev;
    for (int i = 0; i < L; i++) {
        const block *a = &u->enc[i], *b = &o->enc[i];
        n += (i == 0 && oc.cin < u->cfg.cin ? copy_conv_in(u, o, &a->c1, &b->c1) : copy_conv(u, o, &a->c1, &b->c1)) + copy_gn(u, o, &a->n1, &b->n1) +
             copy_conv(u, o, &a->c2, &b->c2) + copy_gn(u, o, &a->n2, &b->n2);
    }
    for (int i = 0; i < L - 1; i++) {
        n += copy_conv(u, o, &u->down[i], &o->down[i]);
        if (u->cfg.down_norm && oc.down_norm) n += copy_gn(u, o, &u->dn[i], &o->dn[i]);
        const block *a = &u->dec[i], *b = &o->dec[i];   /* dec[i] works at level i in both nets */
        n += copy_conv(u, o, &a->c1, &b->c1) + copy_gn(u, o, &a->n1, &b->n1) + copy_conv(u, o, &a->c2, &b->c2) + copy_gn(u, o, &a->n2, &b->n2);
    }
    if (!head_rows) n += copy_conv(u, o, &u->head, &o->head);
    else if (u->head.cin == o->head.cin && u->head.k == o->head.k) {
        const size_t row = (size_t)u->head.cin * u->head.k * u->head.k * u->head.k;
        for (int co = 0; co < u->cfg.cout; co++) {
            const int src = head_rows[co];
            if (src < 0) continue;
            if (src >= oc.cout) { unet_free(o); return -1; }
            for (int e = 0; e < 2; e++) {
                float *dst = e ? u->ema : u->p; const float *sp = e ? o->ema : o->p;
                nn_d2d(dst + u->head.w + co * row, sp + o->head.w + src * row, row * 4);
                nn_d2d(dst + u->head.b + co, sp + o->head.b + src, 4);
            }
        }
        n++;
    } else { unet_free(o); return -1; }
    if (u->cfg.scale_cond && oc.scale_cond && u->enc[0].c1.cout == o->enc[0].c1.cout) {
        nn_d2d(u->p + u->sc_w, o->p + o->sc_w, (size_t)u->enc[0].c1.cout * 4); nn_d2d(u->ema + u->sc_w, o->ema + o->sc_w, (size_t)u->enc[0].c1.cout * 4); n++;
    }
    for (int i = 0; i < L; i++) for (int k = 0; k < u->cfg.enc_blocks[i]; k++) {   /* residual blocks: the donor's, or identity blocks whose output
                                                                                    normalisation is the level's (the net then computes the donor's function) */
        const block *a = &u->rb[i][k];
        if (k < oc.enc_blocks[i]) { const block *b = &o->rb[i][k]; n += copy_conv(u, o, &a->c1, &b->c1) + copy_gn(u, o, &a->n1, &b->n1) + copy_conv(u, o, &a->c2, &b->c2) + copy_gn(u, o, &a->n2, &b->n2); }
        else if (a->n2.c == u->enc[i].n2.c) for (int e = 0; e < 2; e++) {
            float *arr = e ? u->ema : u->p;
            nn_d2d(arr + a->n2.gamma, arr + u->enc[i].n2.gamma, (size_t)a->n2.c * 4); nn_d2d(arr + a->n2.beta, arr + u->enc[i].n2.beta, (size_t)a->n2.c * 4);
        }
    }
    if (u->tbn && o->tbn && u->tbd.C == o->tbd.C && u->tbd.F == o->tbd.F && u->tbd.dK == o->tbd.dK && u->tbd.dL == o->tbd.dL && (u->cfg.attn_pe == o->cfg.attn_pe || (u->cfg.attn_pe && o->cfg.attn_pe))) {   /* the attention stage: PEG, registers
                                                     (the common count) and the common blocks (the donor's first ones) */
        for (int e = 0; e < 2; e++) {
            float *dst = e ? u->ema : u->p; const float *sp = e ? o->ema : o->p; const size_t C = u->tbd.C;
            if (u->cfg.attn_pe == 0) { nn_d2d(dst + u->peg_w, sp + o->peg_w, C * 27 * 4); nn_d2d(dst + u->peg_b, sp + o->peg_b, C * 4); }
            const int nr = u->cfg.attn_reg < o->cfg.attn_reg ? u->cfg.attn_reg : o->cfg.attn_reg;
            if (nr) nn_d2d(dst + u->reg_off, sp + o->reg_off, (size_t)nr * C * 4);
            if (u->cfg.attn_mem && u->cfg.attn_mem == o->cfg.attn_mem) {   /* window memory rows, gate, W_off */
                nn_d2d(dst + u->reg_off + (size_t)u->cfg.attn_reg * C, sp + o->reg_off + (size_t)o->cfg.attn_reg * C, (size_t)u->cfg.attn_mem * C * 4);
                nn_d2d(dst + u->mem_gate, sp + o->mem_gate, 5 * C * 4);
            }
            if (u->cfg.view_P && u->cfg.view_P == o->cfg.view_P && u->cfg.view_patch == o->cfg.view_patch) {   /* view rows, embedding */
                const size_t k3 = (size_t)u->cfg.view_patch * u->cfg.view_patch * u->cfg.view_patch;
                nn_d2d(dst + u->reg_off + (size_t)(u->cfg.attn_reg + u->cfg.attn_mem) * C, sp + o->reg_off + (size_t)(o->cfg.attn_reg + o->cfg.attn_mem) * C, (size_t)u->tb_nv * C * 4);
                nn_d2d(dst + u->view_w, sp + o->view_w, (k3 + 1) * C * 4);
            }
            const int nb = u->tbn < o->tbn ? u->tbn : o->tbn;
            if (u->tbd.dCo == o->tbd.dCo && u->tbd.dInit == o->tbd.dInit)
                for (int k = 0; k < nb; k++) nn_d2d(dst + u->tbp[k].ln1g, sp + o->tbp[k].ln1g, at_block_nparams(&u->tbd) * 4);   /* (one contiguous block) */
            else for (int k = 0; k < nb; k++) dfm_copy_block(u, o, k, dst, sp);   /* (another deformable layout: tensor by tensor) */
        }
        n += (u->tbn < o->tbn ? u->tbn : o->tbn) + 1;
    }
    unet_free(o);
    nn_zero(u->m, u->np * 4); nn_zero(u->v, u->np * 4);
    if (u->muon_mom) nn_zero(u->muon_mom, u->np * 4);
    u->mx_R_nload = 0; u->mx_R_ok = 0;   /* Hyperball radii from the copied weights */
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

/* task restore: a zero output head (live, EMA and optimiser state), so a fresh residual net outputs its input */
void unet_zero_head(unet *u) {
    nn_wmemo_clear();
    float *arr[] = {u->p, u->ema, u->m, u->v};
    for (int i = 0; i < 4; i++) { nn_zero(arr[i] + u->head.w, (size_t)u->head.cin * u->head.cout * u->head.k * u->head.k * u->head.k * sizeof(float)); nn_zero(arr[i] + u->head.b, (size_t)u->head.cout * sizeof(float)); }
}
void unet_head_bias_zero(unet *u) {   /* the head bias (live, EMA, moments) to 0: embedding heads (unet_init's -2 bias would align every output) */
    nn_wmemo_clear();
    float *arr[] = {u->p, u->ema, u->m, u->v};
    for (int i = 0; i < 4; i++) nn_zero(arr[i] + u->head.b, (size_t)u->head.cout * sizeof(float));
}
