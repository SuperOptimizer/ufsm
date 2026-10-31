/* ufsm train: sampler -> pinned batches -> one or more GPUs (data parallel, gradients averaged through the
   host) -> UNet -> BCE+Dice -> AdamW/EMA, with held-out validation batches, CSV log and checkpoints. */
#include "nn.h"
#include "sysinfo.h"
#include "checkpoint.h"
#include "sample.h"
#include "papyrus.h"
#include "restore.h"
#include "srpair.h"
#include "cover.h"
#include <limits.h>
#include "sources.h"
#include "split.h"
#include "unet.h"
#include "nn_lp.h"
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include "embz.h"
#include "sample_emb.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
static double optd(int argc, char **argv, const char *name, double dflt) { const char *v = opt(argc, argv, name, nullptr); return v ? atof(v) : dflt; }
static const char *g_aug_name = "none";   /* train --aug preset */

static volatile sig_atomic_t g_stop;
static void on_sig(int s) { (void)s; g_stop = 1; }

typedef struct {
    int dev;
    unet *u;
    float *xb[2], *gl, *scratch; /* device batch inputs (double-buffered), loss gradient, loss scratch */
    uint8_t *tolc;               /* --tol: per-voxel cube offset of the tolerant maximum (nullptr when off) */
    const float *lg;             /* logits of the last run_batch */
    uint8_t *tb[2], *mb[2], *wb[2], *hwh[2];   /* hwh: --heads, the channel weights in device-row order (pinned) */
    float hl[8]; int hon[8];                    /* --heads: the last batch's loss per head and whether the head had a target */
    float *x; uint8_t *t, *m, *w; /* the buffers of the batch being computed */
    void *radb[2], *rad;         /* --gpu-input: the batch's radial records on the device (nn.h NN_RAD_REC), per buffer / current */
    int64_t *chh[2]; int *chih[2]; float *choh[2]; int chv[2];   /* --attn-mem: per buffer the batch's chain ids, indices, offsets (host copies) */
    uint8_t *vh[2], *vd; size_t vhn[2], vdn; int vv[2];   /* --view-P: per buffer the batch's view crops (host copies) and whether it has them; the device copy */
    float *axh[2]; int axv[2], axn[2], axs[2];   /* --attn-axis radial / --attn-pe scroll: per buffer the batch's scroll-axis records (host copies, sample.h
                                    SAMPLE_AXR) and whether it has them (restore / slab batches: none) */
    void *ev_up[2], *ev_done[2]; /* upload into buffer i finished; the compute that last used buffer i finished */
    batch *pending;              /* host batch whose upload into buffer (cur ^ 1) is in flight */
    batch *tail_due;             /* upsplit: the host batch whose tail (targets, mask, weights) goes up after the loss fetch */
    int upsplit;                 /* lean, three level-0 buffers: the next batch's head goes up in the stem's weight gradient (X2 free), its tail
                                    after the step's loss fetch (X3 is the stem's gradient until the step's end) */
    int hsplit;                  /* lean, head slabs: the next batch's derived-target planes (side / winding) go up once the backward's head step
                                    read them (ev_hs), the input and packed targets (inside gout[0]) after the step's loss fetch (upload_tail) */
    void *ev_hs;                 /* hsplit: the head slabs' backward pass of the current step finished (stream 0) */
    int lean;                    /* one batch buffer; logit gradient in the model's gradient buffer B */
    int gl_inside;               /* lean: the logit gradient lives in the model's gradient buffers (setup checked the batch overlap) */
    char *bat_lo, *bat_hi;       /* lean: the batch's bytes inside the model's buffers (a logit gradient placed later must avoid them) */
    int gl_inplace;              /* the logit gradient over the fp16 logits (g_gl_inplace) */
    float *pg_acc; double pg_h[2], pg_loss;   /* --pgrad-lambda: slab logit gradient (2 planes x slab, fp32), sums w loss / w (hs_ds + 56: + 20..54 hold the normal loss stats), loss */
    void *hs_lg; double *hs_ds; float hs_dice; int hs_z, hs_P, hs_train;   /* head slabs (g_head_slab): fp16 logits of one slab, loss sums, recto dice, planes per slab, window width, a training pass (the backward follows) */
    int64_t cornb[2][3]; int lvlb[2];   /* level-0 corner / level of the batch in each buffer (verso dump) */
    int srcb[2], last;           /* the batch's source in each buffer (-1 unknown); the buffer of the last training step's batch (diagnostics) */
    uint8_t *vsoft, *vanch;      /* --verso-ridge: the soft teacher output of the window; --verso-anchor ridge target */
    uint8_t *vmask, *vw; float *vscratch, vout[3]; double verso_loss;   /* verso loss: own mask (every owned voxel), weight byte, scratch, parts */
    float *teacher;              /* --verso frozen teacher: the starting weights, read by the verso target forward */
    int cur;
    int side;                    /* --split z: 0 = low z half, 1 = high z half of every window; -1 = whole windows */
    const sheet_batch *sheet;
    double sheet_loss, sheet_parts[5];
    float umb[2], um;            /* input voxel size of the batch in each buffer / the current one (scale_cond) */
    uint8_t *bandb[2], *band;    /* task band_affinity: band per voxel (double-buffered like t / m), current */
    uint8_t *sideb[2], *sdb, *sidet, *sidem, *sidew; float *sscratch, sout[5]; double side_loss;   /* --side: SIDE_* / PHASE_* labels, loss target / mask / weight */
    float *aff_fin, aff_out[48]; double aff_loss;
    uint8_t *pmb[2], *pm, *nt, *nm, *nw; float *iscratch, *nscratch, iout[3], nout[15]; double ink_loss, norm_loss;
    float *cnt_acc, *cnt_seg, *cnt_out, cnt_h[3]; unsigned *cnt_nv; uint64_t cnt_step; double count_loss;   /* --count-lambda */   /* task unwrap: papyrus mask (ink loss), normal / density targets, mask, weights; scratch and parts of the ink and normal losses */
    uint8_t *w4b[2], *wcb[2], *w4, *wc; double *w4ds; float *w4fin, w4h[8]; double w4_loss;   /* --task unwrap2: winding mod 4 and its component per voxel (double-buffered), loss sums, scratch, host parts */
    uint8_t *wdb[2], *wd, *ct4, *cm4, *cw4; float *cscratch, cout4[9]; double cond_loss;   /* --cond: the gauge-shifted winding per voxel, its 4 targets, mask, weights; scratch and parts */
    unsigned long long *cacc, cacc_h[2];   /* --cond: known / right-turn voxel counts of the 14-turn head (device, last batch on the host) */
    uint8_t *papb[2], *pap, *pt1, *pmk1, *pw1; float *pscr, pout3[3]; double pap_loss;   /* --pap-head: labels (double-buffered), target, mask, weight; scratch, parts */
    void *x2, *glh, *lrt; float *lgf;  /* --half-res: the network's input and logit gradient, the intermediate grid (div 4), the full-resolution fp32 logits */
    uint8_t *ds_t[UNET_MAXLEV], *ds_m[UNET_MAXLEV]; float *ds_gl[UNET_MAXLEV], *ds_scr[UNET_MAXLEV], ds_out[UNET_MAXLEV][2 * 8 + 2]; double ds_loss; int ds_fresh;   /* --deep-sup (fresh: the last batch computed them) */
    int8_t *embb[2], *embt; float *embg; double *embds, embh[2], emb_loss, emb_cells; int emb_fresh;   /* --distill: feature targets (double-buffered), head gradient, loss sums (sum 1 - cos, cells) */
    uint8_t *rlow; void *rscr; size_t rscr_b; double *rst, rst_h[RP_NST]; float *rrd; restore_recipe rcp;   /* task restore: degraded input (uint8), degradation scratch,
                                                                                      loss sums (device / last fetched), the recipe of the next forward */
} gpu_state;
/* task band_affinity: output channel 0 = recto (dense loss on target channel 0), channels 1..K = affinities for g_aff_off */
static int g_half, g_hdiv = 1;   /* --half-res: 1 pool, 2 pack, 3 pool4, 4 pack4 (unet_cfg.half_res), g_hdiv its factor; the losses see g_cout
                                    full-resolution channels */
static int g_pkf;   /* --half-res pack, the direct path: the losses read the network's packed logits (nn_loss_pack) and write its packed
                       gradient (in place), the input built from the CT channel on the device (--gpu-input, nn_input_rad_pack) or by
                       nn_lowres_input; no full-resolution logits / gradient; the lean memory modes. UFSM_PACK_DIRECT=0: the old path */
#define TCMAX 8   /* surface-task target / output channels at most (--heads) */
static int g_heads, g_tc = NCH;   /* --heads H (>= 2): recto heads in output channels 0 and 2..H, sheet in 1; g_tc: target channels on the device */
static const sources *g_S;        /* --heads: each sample's source names its head */
static int g_aff, g_cout = NCH; static aff_offsets_t g_aff_off; static float g_aff_lambda = 1.f;
/* --verso 1: output channel 1 is the verso face. Its target is the model's own recto channel (eval forward, no gradient) with
   the radial input reversed and the CT unchanged: a recto model marks the other face of the sheet when the direction flips
   (the usrm2 radial-sign trick). Band affinities then start at channel 2. g_verso_on: from --verso-start on. */
/* --side 1 (task band_affinity): one more output channel, the last, classifies every papyrus voxel near labelled rectos as on the
   recto side (outward of the nearest recto) or the verso side of it (src/band.h SIDE_*): BCE over its own mask, times g_side_lambda.
   --side 2: two channels, the winding phase between labelled consecutive rectos (src/band.h PHASE_*) as soft targets
   0.5 + 0.5 cos(2 pi phase) and 0.5 + 0.5 sin(2 pi phase) (BCE: minimised at the target, so an unsure net shrinks towards 0.5). */
static int g_side, g_side_ch, g_side_nc; static float g_side_lambda = 1.f;
/* --task unwrap (B == 1): channel 0 recto (target 0, the label mask), 1 ink (target slot 1 = the source's "ink" prob teacher, BCE over
   all papyrus with dice g_ink_dice, times g_ink_lambda), 2-3 the winding phase (the --side 2 path), 4-9 the sign-free normal n n^T and
   10 the sheet density, both from the phase gradient over +-g_norm_s voxels (nn_unwrap_targets; BCE, times g_norm_lambda) */
static int g_xc = 4;   /* input channels per sample: CT + radial (3), + the aux volume with --aux-input, + 3 with --cond */
/* --cond 1 (task unwrap): conditioned inference (predict --cond-sweep). Inputs 4-6 hold an earlier prediction of the winding mod 14
   turns where one exists: cos and sin of 2 pi q' / 14 and a validity mask, q' = q + k with a random integer gauge k per sample;
   outputs g_cond_ch.. +3 are cos / sin of 2 pi q' / 14 and 2 pi q' / 2 (soft BCE like the phase), so the net carries the gauge it
   is handed across the window. The sampler draws the condition from the sources' "winding" raster (src/sample.c cond patterns) */
static int g_cond, g_cond_ch, g_cond_outside; static float g_cond_lambda = 1.f;
/* --pap-head 1 (task unwrap): one more output channel, papyrus vs void (src/papyrus.h gap labels of the clean CT, no cutoff input; the
   sampler's batch.pap), BCE over voxels inside the scroll (CT > 0) that are not ignored, times g_pap_lambda */
static int g_pap_head, g_pap_ch; static float g_pap_lambda = 1.f; static papyrus_cfg g_paph;
/* --task unwrap2 (B == 1): the spiral fit's heads only: 0-1 the winding phase, 2-7 the normal n n^T, 8 the density (as task unwrap's
   2-10), 9-10 the relative winding mod 4 (cos / sin of 2 pi W / 4, W = the stored phase unwrapped over the window: sample.c band_wind4)
   with the gauge-free loss nn_wind4_* (per component the best of the four integer gauges), times g_w4_lambda. No recto or ink head */
static int g_unwrap2, g_w4_ch = -1; static float g_w4_lambda = 1.f;
static int g_unwrap, g_norm_ch, g_norm_s = 4; static float g_ink_lambda = 1.f, g_ink_dice = 0.f, g_norm_lambda = 1.f, g_dens_scale = 32.f;
static int g_dens_code; static float g_packed_w, g_packed_um = 120.f;   /* --density-code, --packed-weight, --packed-um (task unwrap) */
static float g_ptrunc_tau, g_ptrunc_kappa = 0.5f; static int g_ptrunc_from = 5000, g_ptrunc_on;   /* --phase-trunc TAU [--phase-trunc-conf K] [--phase-trunc-from STEP]: the robust phase loss (nn_phase_trunc), on from STEP */
static int g_gl_inplace;   /* the logit gradient over the fp16 logits (set before the memory planner) */
static int g_head_slab;    /* unwrap: head, losses and head backward per z-slab (no whole-window logits or derived targets; set before the planner) */
static int pm_used(void);   /* the papyrus-mask plane (pm) is read: unwrap, not unwrap2 (no ink plane) unless --pap-head */
static int g_tm_unused;    /* unwrap2 head slabs: the packed recto target / mask (tm -> t, m) is never read: not uploaded, not unpacked, no 3 B per voxel (UFSM_TM_KEEP=1 keeps them) */
static int g_hs_z;         /* its planes per slab */
static int g_ds; static float g_ds_w = 0.5f, g_ds_frac = 1.f; static int g_ds_anneal = 1;   /* --deep-sup K: decoder levels 1..K, weight w^j x frac */
static int g_distill, g_distill_lev = 3, g_distill_div; static float g_distill_w = 1.f; static float g_distill_pap;   /* --distill-papyrus F: targets only on head cells >= F papyrus */ static size_t g_distill_n;   /* --distill C: a C-channel 1^3 head on decoder level --distill-level
                                                                                                    (cells of g_distill_div^3 window voxels; g_distill_n int8 targets per batch), cosine loss x --distill-weight */
static float g_count_lambda; static int g_count_K = 1024, g_count_L = 64;   /* --count-lambda / -segments / -length: the count loss (nn_count_loss) */
static float g_pg_lambda, g_pg_dist[4] = {2.f}; static int g_pg_nd = 1, g_pg_angle, g_pg_mask;   /* --pgrad-lambda / --pgrad-dist (D[,D...], up to 4): the phase wavenumber loss (nn_pgrad_slab; head slabs only) */
/* --task papyrus: papyrus vs void inside the scroll, labels made by the sampler from the clean CT at a random cutoff per patch
   (src/papyrus.h) in target channel 0; --pap-cond 1 conditions the stem bias on that cutoff in input units (batch.um, unet_set_cond) */
static int g_papyrus; static papyrus_cfg g_pap;
static int papyrus_args(int argc, char **argv, int B) {
    papyrus_cfg *c = &g_pap; memset(c, 0, sizeof *c);
    const char *rel = opt(argc, argv, "--pap-rel", nullptr);
    int bad = sscanf(opt(argc, argv, "--pap-cut", "50,128"), "%f,%f", &c->t_lo, &c->t_hi) != 2 ||
              sscanf(opt(argc, argv, "--pap-tau", "0,0"), "%f,%f", &c->tau_lo, &c->tau_hi) != 2 ||
              (rel && sscanf(rel, "%f,%f", &c->rel_lo, &c->rel_hi) != 2);
    c->rel_p = rel ? (float)atof(opt(argc, argv, "--pap-rel-p", "0.5")) : 0.f;
    c->sigma = (float)atof(opt(argc, argv, "--pap-sigma", "1")); c->band = (float)atof(opt(argc, argv, "--pap-band", "0"));
    c->local = atoi(opt(argc, argv, "--pap-local", "0"));   /* gap labels (papyrus_local): no cutoff, so no conditioning by default */
    c->local_sigma = (float)atof(opt(argc, argv, "--pap-local-sigma", "1.5")); c->local_depth = (float)atof(opt(argc, argv, "--pap-local-depth", "30"));
    c->local_margin = (float)atof(opt(argc, argv, "--pap-local-margin", "0.1"));
    c->clean = atoi(opt(argc, argv, "--pap-clean", "0")); c->cond = atoi(opt(argc, argv, "--pap-cond", c->local > 0 ? "0" : "1"));   /* 1 stem bias, 2 input zero */
    if (c->local < 0 || (c->local > 0 && (c->cond || rel || c->tau_hi > 0 || !(c->local_sigma >= 0 && c->local_sigma <= PAPYRUS_RMAX / 3.f) ||
                                          !(c->local_depth > 0) || !(c->local_margin >= 0 && c->local_margin < 0.5f)))) {
        fprintf(stderr, "--task papyrus --pap-local R (voxels): gap labels with no cutoff, so no --pap-cond, --pap-rel or --pap-tau; [--pap-local-sigma 0..%.3g] [--pap-local-depth D > 0] [--pap-local-margin 0..0.5)\n", PAPYRUS_RMAX / 3.0);
        return 1;
    }
    bad |= !(c->t_lo >= 0 && c->t_lo <= c->t_hi && c->t_hi <= 255) || !(c->rel_p >= 0 && c->rel_p <= 1) || (rel && !(c->rel_lo > 0 && c->rel_lo <= c->rel_hi)) ||
           !(c->sigma >= 0 && c->sigma <= PAPYRUS_RMAX / 3.f) || !(c->tau_lo >= 0 && c->tau_lo <= c->tau_hi) || (c->tau_hi > 0 && !(c->tau_lo > 0)) ||
           !(c->band >= 0) || c->clean < 0 || c->cond < 0 || c->cond > 2;
    if (bad) { fprintf(stderr, "--task papyrus: --pap-cut LO,HI (0..255) [--pap-rel LO,HI --pap-rel-p P] [--pap-sigma 0..%.3g] [--pap-tau LO,HI (> 0)] [--pap-band B] [--pap-clean N] [--pap-cond 0|1|2]\n", PAPYRUS_RMAX / 3.0); return 1; }
    if (c->cond == 1 && B != 1) { fprintf(stderr, "--task papyrus --pap-cond 1 needs --B 1 (one cutoff per forward)\n"); return 1; }
    if (atoi(opt(argc, argv, "--ct-aug", "0"))) { fprintf(stderr, "--task papyrus: not with --ct-aug (its input is not the z-scored CT the cutoff code assumes)\n"); return 1; }
    if (c->local > 0) {
        fprintf(stderr, "papyrus: gap labels (no cutoff): void where the CT smoothed by sigma %g is <= %g, or in the lower half of a gap at least %g deep "
                        "(closing - floor of the CT smoothed by sigma %g, radius %d voxels; threshold clamped to [%g, %g]); margin %g, ignore band %g, cleanup < %d voxels\n",
                (double)c->sigma, (double)c->t_lo, (double)c->local_depth, (double)c->local_sigma, c->local, (double)c->t_lo, (double)c->t_hi, (double)c->local_margin, (double)c->band, c->clean);
        return 0;
    }
    fprintf(stderr, "papyrus: cutoff U(%g, %g)", c->t_lo, c->t_hi);
    if (c->rel_p > 0) fprintf(stderr, ", with p %g Otsu x U(%g, %g)", c->rel_p, c->rel_lo, c->rel_hi);
    fprintf(stderr, " of the CT smoothed by sigma %g; %s targets", c->sigma, c->tau_hi > 0 ? "soft" : "hard");
    if (c->tau_hi > 0) fprintf(stderr, " (tau U(%g, %g))", c->tau_lo, c->tau_hi);
    fprintf(stderr, ", ignore band %g, cleanup < %d voxels, %s\n", c->band, c->clean, c->cond == 1 ? "conditioned on the cutoff (stem bias)" :
            c->cond == 2 ? "conditioned on the cutoff (the CT input's zero)" : "marginalised over cutoffs");
    return 0;
}
/* --task restore (src/restore.h): target channel 0 the clean CT (1 channel, no mask: CT > 0), the input degraded from it on the device per
   step (a family drawn from --restore-mix, its parameters from the ranges), residual output input + 64 logit, Charbonnier + edge + MSE */
static int g_restore, g_rsplit = 1; static float g_reps = 0.01f, g_rmse = 0.1f, g_redge = 0.5f; static restore_ranges g_rr;
static int g_rpairs; static restore_pair_w g_rpw;   /* --sr-pairs 1: real pairs (sources with "sr_pair", src/srpair.h) and their loss weights */
static uint64_t g_rseed; static int g_rP; static int g_Pz;   /* (the window edge, for the recipes) */
static int restore_args(int argc, char **argv) {
    restore_ranges_default(&g_rr);
    if (*opt(argc, argv, "--restore-mix", "") && restore_parse_mix(&g_rr, opt(argc, argv, "--restore-mix", ""))) { fprintf(stderr, "--restore-mix identity:P,pyramid:P,scanner:P\n"); return 1; }
    const char *in = opt(argc, argv, "--restore-input", "split");
    if (strcmp(in, "split") && strcmp(in, "plain")) { fprintf(stderr, "--restore-input split|plain\n"); return 1; }
    g_rsplit = !strcmp(in, "split");
    g_reps = (float)atof(opt(argc, argv, "--restore-eps", "0.01")); g_rmse = (float)atof(opt(argc, argv, "--mse-weight", "0.1")); g_redge = (float)atof(opt(argc, argv, "--edge-weight", "0.5"));
    g_rr.pyr_q = (float)atof(opt(argc, argv, "--pyramid-q", "2")); g_rr.sc_q = (float)atof(opt(argc, argv, "--scanner-q", "8"));
#define RANGE(flag, dst) do { const char *v_ = opt(argc, argv, flag, nullptr); if (v_ && sscanf(v_, "%f,%f", &(dst)[0], &(dst)[1]) != 2) { fprintf(stderr, "%s LO,HI\n", flag); return 1; } } while (0)
    RANGE("--scanner-step", g_rr.sc_step); RANGE("--scanner-psf", g_rr.sc_psf); RANGE("--scanner-noise", g_rr.sc_noise);
    RANGE("--scanner-sigvar", g_rr.sc_sigvar); RANGE("--scanner-corr", g_rr.sc_corr); RANGE("--scanner-edge", g_rr.sc_edge);
#undef RANGE
    g_rr.sc_edge_p = (float)atof(opt(argc, argv, "--scanner-edge-p", "0.3")); g_rr.sc_aniso = (float)atof(opt(argc, argv, "--scanner-aniso", "0.15"));
    g_rr.sc_gain = (float)atof(opt(argc, argv, "--scanner-gain", "0.06")); g_rr.sc_gamma = (float)atof(opt(argc, argv, "--scanner-gamma", "0.06")); g_rr.sc_bias = (float)atof(opt(argc, argv, "--scanner-bias", "3"));
    g_rpairs = atoi(opt(argc, argv, "--sr-pairs", "0")) != 0;
    g_rpw = (restore_pair_w){g_reps, g_rmse, g_redge, (float)atof(opt(argc, argv, "--sr-shift-w", "0.5")), (float)atof(opt(argc, argv, "--sr-ds", "0.5")), atoi(opt(argc, argv, "--sr-shift", "1")), (float)atof(opt(argc, argv, "--sr-unc", "0"))};
    if (g_rpw.unc < 0) { fprintf(stderr, "--sr-unc >= 0\n"); return 1; }
    if (g_rpairs && (!(g_rpw.shift_w >= 0 && g_rpw.shift_w <= 1) || g_rpw.ds < 0 || (g_rpw.shift != 0 && g_rpw.shift != 1))) { fprintf(stderr, "--sr-shift-w 0..1, --sr-ds >= 0, --sr-shift 0|1\n"); return 1; }
    if (g_rpairs) fprintf(stderr, "restore: real pairs: loss (1 - %g) Charbonnier + %g shift-min Charbonnier (shift %d) + %g edge + %g MSE + %g downsample consistency\n", g_rpw.shift_w, g_rpw.shift_w, g_rpw.shift, g_redge, g_rmse, g_rpw.ds);
    if (!(g_reps > 0) || g_rmse < 0 || g_redge < 0 || !(g_rr.sc_step[0] >= 2 && g_rr.sc_step[1] >= g_rr.sc_step[0]) || g_rr.pyr_q < 0 || g_rr.sc_q < 0) { fprintf(stderr, "restore: eps > 0, weights >= 0, --scanner-step >= 2\n"); return 1; }
    fprintf(stderr, "restore: mix identity %.3g pyramid %.3g (4x block mean, q %g) scanner %.3g (step %g-%g, psf %g-%g coarse voxels, noise %g-%g DN, sigvar %g-%g, corr %g-%g, edge %g-%g p %g, q %g); "
                    "input %s (x - 128) / 64, residual out = input + 64 logit; loss Charbonnier (eps %g) + %g edge + %g MSE\n",
            g_rr.mix[0], g_rr.mix[1], g_rr.pyr_q, g_rr.mix[2], g_rr.sc_step[0], g_rr.sc_step[1], g_rr.sc_psf[0], g_rr.sc_psf[1], g_rr.sc_noise[0], g_rr.sc_noise[1],
            g_rr.sc_sigvar[0], g_rr.sc_sigvar[1], g_rr.sc_corr[0], g_rr.sc_corr[1], g_rr.sc_edge[0], g_rr.sc_edge[1], g_rr.sc_edge_p, g_rr.sc_q, g_rsplit ? "split into two MX-fp8-exact channels" : "one channel", g_reps, g_redge, g_rmse);
    return 0;
}
static uint64_t rmix(uint64_t x) { x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 27; x *= 0x94d049bb133111ebull; return x ^ (x >> 31); }
/* the training recipe of this GPU's next forward: family and parameters from (seed, step, GPU) */
static void restore_train_recipe(gpu_state *d, int step, int g) {
    if (!g_restore) return;
    const uint64_t h = rmix(g_rseed ^ rmix((uint64_t)step * 0x9E3779B97F4A7C15ull + (uint64_t)g));
    restore_recipe_draw(&d->rcp, restore_pick_family(&g_rr, (double)(h >> 11) * 0x1.0p-53), rmix(h), g_Pz, g_rP, g_rP, &g_rr);
}
static float g_verso_swap, g_verso_lambda = 1.f;   /* lambda: verso loss weight (its gradient and its share of the reported loss) */ static int g_verso_flip, g_head_only;   /* swap: probability of a step with the direction reversed and recto/verso targets exchanged */
static int g_verso, g_verso_start, g_verso_on, g_verso_self, g_verso_zero, g_verso_ridge, g_verso_anchor;   /* anchor: recto voxels without a trusted label take the teacher's normal-direction output */ static float g_verso_hard;   /* ridge: thin verso targets, the teacher's ridge along the direction at >= this byte */   /* hard: binary verso targets at this teacher probability; zero: fresh verso head row */   /* self: targets from the live weights (collapses: the net learns to ignore the direction) */
static sheet_dataset *g_sheet;
static float g_sheet_ramp;
static int g_sheet_variant=2;
/* --split z: both GPUs work on the same window, each on its z half plus g_h0 halo planes (depth g_Dl); the mask is zeroed on
   the halo planes (from a pinned zero buffer) so each voxel counts once */
static int g_Dl, g_h0; static uint8_t *g_zeros;
static int g_bench_same;   /* UFSM_BENCH_SAME */
static int g_Pz;   /* window depth: --Pz (a Pz x P x P slab), else P */
static int g_valP;   /* --val-P N (!= P): validation windows of N^3 (the same voxels for every run with this N and --val-seed), scored by an
                        inference-only copy of the network on GPU 0 that reads the training weights; 0: windows of the training size */
static size_t g_val_reserve;   /* --val-P: device bytes the planner keeps free for that copy's inference build */
static unet *g_val_u;          /* --val-P: that copy */

static int g_xfmt = 0, g_g16 = 0;   /* input batches uploaded as 16-bit (1 fp16, 2 bf16); loss gradient written as 16-bit */
static size_t xbytes(void) { return g_xfmt ? 2 : 4; }
static const void *bx(const batch *b) { return g_xfmt ? (const void *)b->x16 : (const void *)b->x; }
/* --gpu-input (default when the configuration allows it, sample_gpu_input_ok): batches carry the CT channel only (g_xs 1) and per sample a
   radial record; the network's 4 MX input channels are built on the device (nn_input_rad). g_xs: input channels per sample in the batches */
static int g_gpu_in, g_xs = 4, g_gpu_want, g_gpu_asked, g_pack, g_gpu_aug, g_gpu_aug_want, g_gpu_aug_asked;   /* g_gpu_aug: --gpu-aug (sample.h gpu_aug: the device builds the CT channel, nn_aug_ct) */
static int g_xs_unused_;   /* g_pack: --pack-targets (batch.tm, nn_unpack_tm on upload) */
static size_t rad_bytes(int B) { return g_gpu_in ? (size_t)B * NN_RAD_REC(g_Pz) : 0; }
/* the attention stage's scroll axis (unet_set_attn_axis): the sampler writes each window's record (sample_cfg.axis_rec) */
static int g_axr;
static size_t axr_floats(int P) { return (size_t)SAMPLE_AXR(g_Pz > P ? g_Pz : P); }
static size_t axr_bytes(int B, int P) { return g_axr ? (size_t)B * axr_floats(P) * sizeof(float) : 0; }
static void axr_take(gpu_state *d, const batch *b, int i, int B, int P) {
    if (!g_axr) return;
    d->axv[i] = b->axr != nullptr;
    if (!b->axr) return;
    if (!d->axh[i]) { d->axh[i] = malloc((size_t)B * SAMPLE_AXR(4096) * sizeof(float)); if (!d->axh[i]) abort(); }
    memcpy(d->axh[i], b->axr, axr_bytes(B, P));
    d->axn[i] = B; d->axs[i] = (int)axr_floats(P);
}
/* --attn-mem: window memory carried along the sampler's window chains (sample_cfg.chain_p). Per GPU a bank of the latest memory output
   per chain (device rows, a constant: stop-gradient); a chain window whose predecessor's entry is there gets it with its offset, the rest
   (chain starts, other GPUs' predecessors, evicted entries, validation) zero memory with valid 0 */
static int g_mem;
#define MEMBANK 64
typedef struct { int64_t id[MEMBANK]; int idx[MEMBANK]; float *dev[MEMBANK]; int next; float *stage; size_t stage_n; long hits, misses; } mem_bank;
static mem_bank g_mb[8];
static void chain_take(gpu_state *d, const batch *b, int i, int B) {
    if (!g_mem) return;
    d->chv[i] = b->chid != nullptr;
    if (!b->chid) return;
    if (!d->chh[i]) { d->chh[i] = malloc(64 * sizeof(int64_t)); d->chih[i] = malloc(64 * sizeof(int)); d->choh[i] = malloc(64 * 3 * sizeof(float)); if (!d->chh[i] || !d->chih[i] || !d->choh[i] || B > 64) abort(); }
    memcpy(d->chh[i], b->chid, (size_t)B * sizeof(int64_t)); memcpy(d->chih[i], b->chidx, (size_t)B * sizeof(int)); memcpy(d->choh[i], b->choff, (size_t)B * 3 * sizeof(float));
}
static void mem_pre(gpu_state *d, int B, int train) {
    if (!g_mem) return;
    if (!train || !d->chv[d->cur]) { unet_set_attn_mem(d->u, nullptr, nullptr, 0); return; }
    mem_bank *mb = &g_mb[d->dev & 7]; const int M = unet_cfg_of(d->u)->attn_mem, C = unet_cfg_of(d->u)->widths[unet_cfg_of(d->u)->nlev - 1];
    const size_t row = (size_t)M * C;
    if (mb->stage_n < (size_t)B * row) { if (mb->stage) nn_free(mb->stage); mb->stage = nn_malloc((size_t)B * row * 4); mb->stage_n = (size_t)B * row; }
    float feat[64 * 4];
    for (int s = 0; s < B; s++) {
        const int64_t id = d->chh[d->cur][s]; const int idx = d->chih[d->cur][s]; int e = -1;
        if (id >= 0 && idx > 0) for (int k = 0; k < MEMBANK; k++) if (mb->dev[k] && mb->id[k] == id && mb->idx[k] == idx - 1) { e = k; break; }
        if (id >= 0 && idx > 0) { if (e >= 0) mb->hits++; else mb->misses++; }
        if (e >= 0) nn_d2d(mb->stage + s * row, mb->dev[e], row * 4); else nn_zero(mb->stage + s * row, row * 4);
        for (int k = 0; k < 3; k++) feat[4 * s + k] = e >= 0 ? d->choh[d->cur][3 * s + k] : 0.f;
        feat[4 * s + 3] = e >= 0 ? 1.f : 0.f;
    }
    unet_set_attn_mem(d->u, mb->stage, feat, B);
}
static void mem_post(gpu_state *d, int B, int train) {   /* after a training forward: its memory rows into the bank under (chain, index) */
    if (!g_mem || !train || !d->chv[d->cur]) return;
    mem_bank *mb = &g_mb[d->dev & 7]; int M, C; const float *out = unet_attn_mem_out(d->u, &M, &C);
    if (!out) return;
    const size_t row = (size_t)M * C;
    for (int s = 0; s < B; s++) {
        const int64_t id = d->chh[d->cur][s]; if (id < 0) continue;
        int e = -1;
        for (int k = 0; k < MEMBANK; k++) if (mb->dev[k] && mb->id[k] == id) { e = k; break; }
        if (e < 0) { e = mb->next; mb->next = (mb->next + 1) % MEMBANK; }
        if (!mb->dev[e]) mb->dev[e] = nn_malloc(row * 4);
        nn_d2d(mb->dev[e], out + s * row, row * 4); mb->id[e] = id; mb->idx[e] = d->chih[d->cur][s];
    }
}
/* --view-P (scan fusion): the batch's view crops (sample.h batch.xv) copied for buffer i (the batch goes back to the sampler); view_set
   uploads them before the forward (the device copy is read by the forward alone) */
static int g_view;
static void view_take(gpu_state *d, const batch *b, int i, int B, int async) {
    (void)async;
    if (!g_view) return;
    d->vv[i] = b->xv != nullptr;
    if (!b->xv) return;
    const size_t n = (size_t)B * g_view * g_view * g_view;
    if (d->vhn[i] < n) { free(d->vh[i]); d->vh[i] = malloc(n); d->vhn[i] = n; if (!d->vh[i]) abort(); }
    memcpy(d->vh[i], b->xv, n);
}
static void view_set(gpu_state *d, int B) {
    if (!g_view) return;
    if (!d->vv[d->cur]) { unet_set_attn_view(d->u, nullptr, 0); return; }
    const size_t n = (size_t)B * g_view * g_view * g_view;
    if (d->vdn < n) { if (d->vd) nn_free(d->vd); d->vd = nn_malloc(n); d->vdn = n; }
    nn_h2d(d->vd, d->vh[d->cur], n);
    unet_set_attn_view(d->u, d->vd, B);
}
static void axr_set(gpu_state *d, int i) { if (g_axr) unet_set_attn_axis(d->u, d->axv[i] ? d->axh[i] : nullptr, d->axn[i], d->axs[i]); }
/* once the memory mode is set: the GPU-built input needs the 16-bit -> MX input conversion (unet_input_converted: MX activations with
   MX gradients or an input policy) */
static void gpu_input_settle(sample_cfg *sc) {
    g_gpu_in = g_gpu_want && unet_input_converted();
    g_xs = g_gpu_in ? 1 : g_xc; sc->gpu_input = g_gpu_in;
    g_gpu_aug = g_gpu_aug_want && g_gpu_in && sample_gpu_aug_ok(sc) ? 1 : 0;
    if (g_gpu_aug && g_gpu_aug_want == 2) { if (sample_gpu_lwarp_ok(sc)) g_gpu_aug = 2; else fprintf(stderr, "--gpu-aug 2: the labels' warp needs packed targets and no sheet / pmask / gap / distill / cond / band / aux labels; --gpu-aug 1\n"); }
    sc->gpu_aug = g_gpu_aug;
    if (g_gpu_aug) fprintf(stderr, "input: the scanner augmentation, z-score, symmetry / warp and jitter of the CT channel on the GPU (--gpu-aug 0: on the CPU)\n");
    if (g_gpu_aug == 2) fprintf(stderr, "input: the warped windows' packed targets, phase and winding mod 4 warped on the GPU (--gpu-aug 1: on the CPU)\n");
    if (g_gpu_aug_asked && g_gpu_aug_want && !g_gpu_aug) fprintf(stderr, "--gpu-aug: not available here (needs --gpu-input, cubes, no --ct-aug, --side-air 0, no --overfit / --split); CPU augmentation\n");
    if (g_gpu_in) fprintf(stderr, "input: the CT channel uploaded, the radial channels built on the GPU (--gpu-input 0: host-built)\n");
    else if (g_gpu_asked) fprintf(stderr, "--gpu-input: not available here (needs the 16-bit -> MX input conversion, whole windows, the 4 plain inputs, no warps / sheet prior); host-built input\n");
}
static void select_buf(gpu_state *d, int i) {
    d->rad = d->radb[i]; if (g_gpu_in) unet_set_input_rad(d->u, d->rad);
    axr_set(d, i);
    d->cur = i; d->embt = d->embb[i]; d->x = d->xb[i]; d->t = d->tb[i]; d->m = d->mb[i]; d->w = d->wb[i]; d->band = d->bandb[i]; d->sdb = d->sideb[i]; d->pm = d->pmb[i]; d->wd = d->wdb[i]; d->pap = d->papb[i]; d->w4 = d->w4b[i]; d->wc = d->wcb[i]; d->um = d->umb[i]; }
/* --heads: the device target row of host row r (n * NCH + c): a sample's recto target trains its source's head (channel 0, or 1 + head),
   the sheet target channel 1; the other heads' rows keep stale bytes under weight 0 */
static int trow(const batch *b, int r) {
    if (!g_heads) return r;
    const int n = r / NCH, c = r % NCH, h = g_S->src[b->src[n]].head;
    return n * g_tc + (c == CH_SHEET ? CH_SHEET : h ? 1 + h : CH_RECTO);
}
static const uint8_t *trow_w(const batch *b, int B, uint8_t *hw) {   /* the channel weights in device-row order (hw: B x g_tc) */
    if (!g_heads) return b->w;
    memset(hw, 0, (size_t)B * g_tc);
    for (int r = 0; r < B * NCH; r++) hw[trow(b, r)] = b->w[r];
    return hw;
}
static size_t host_mem_available(void) { return ufsm_mem_available(); }   /* (src/sysinfo.h: MemAvailable under every cgroup memory.max) */
/* the host arrays of a batch in this run's layout: the addresses of its members and their sizes (validation copies, --val-store) */
enum { VAL_DISK, VAL_RAM, VAL_REGEN, VAL_NF = 20 };
static int val_fields(batch *b, int B, size_t p3, void **f[VAL_NF], size_t sz[VAL_NF]) {
    int n = 0;
#define VF(p_, s_) do { f[n] = (void **)&(p_); sz[n++] = (s_); } while (0)
    if (g_restore) { VF(b->t, (size_t)B * g_tc * p3); VF(b->w, (size_t)B * NCH); VF(b->src, (size_t)B * sizeof *b->src); VF(b->um, (size_t)B * sizeof *b->um); return n; }   /* (the clean CT alone) */
    if (g_xfmt) VF(b->x16, (size_t)B * g_xs * p3 * 2); else VF(b->x, (size_t)B * g_xs * p3 * 4);
    if (g_gpu_in) VF(b->rad, rad_bytes(B));
    if (g_axr) VF(b->axr, (size_t)B * SAMPLE_AXR((int)llround(sqrt((double)p3 / g_Pz)) > g_Pz ? (int)llround(sqrt((double)p3 / g_Pz)) : g_Pz) * sizeof(float));
    if (g_pack) VF(b->tm, (size_t)B * p3); else { VF(b->t, (size_t)B * NCH * p3); VF(b->m, (size_t)B * p3); }
    VF(b->w, (size_t)B * NCH); VF(b->src, (size_t)B * sizeof *b->src); VF(b->um, (size_t)B * sizeof *b->um);
    if (g_aff) VF(b->band, (size_t)B * p3);
    if (g_side) VF(b->side, (size_t)B * p3);
    if (g_unwrap) VF(b->pm, (size_t)B * p3);
    if (g_unwrap2) { VF(b->w4, (size_t)B * p3); VF(b->wcomp, (size_t)B * p3); }
    if (g_cond) VF(b->wind, (size_t)B * p3);
    if (g_pap_head) VF(b->pap, (size_t)B * p3);
    if (g_view) { VF(b->xv, (size_t)B * g_view * g_view * g_view); VF(b->vok, (size_t)B); }   /* (after the fields of runs without views: their hash unchanged) */
    if (g_distill) VF(b->emb, g_distill_n);
#undef VF
    return n;
}
typedef struct { int fd; size_t bytes; batch buf; } valdisk;   /* --val-store disk: the windows back to back in an unlinked file */
static int pwrite_all(int fd, const void *p, size_t n, size_t off) {
    for (size_t k = 0; k < n;) { const ssize_t w = pwrite(fd, (const char *)p + k, n - k, (off_t)(off + k)); if (w <= 0) return -1; k += (size_t)w; }
    return 0;
}
static uint64_t hash_bytes(uint64_t h, const void *p, size_t n) {   /* 64-bit words, multiply-xorshift; the tail bytewise */
    const uint8_t *c = p; size_t k = 0;
    for (; k + 8 <= n; k += 8) { uint64_t w; memcpy(&w, c + k, 8); h = (h ^ w) * 0x100000001B3ull; h ^= h >> 29; }
    for (; k < n; k++) { h = (h ^ c[k]) * 0x100000001B3ull; }
    return h ^ (h >> 31);
}
static int val_read(valdisk *v, int i, int B, size_t p3) {   /* window i into v->buf (its pages then leave the page cache) */
    void **f[VAL_NF]; size_t sz[VAL_NF];
    const int nf = val_fields(&v->buf, B, p3, f, sz);
    size_t wb = 0;
    for (int k = 0; k < nf; k++) wb += sz[k];
    size_t off = (size_t)i * wb;
    for (int k = 0; k < nf; k++) {
        for (size_t d = 0; d < sz[k];) { const ssize_t r = pread(v->fd, (char *)*f[k] + d, sz[k] - d, (off_t)(off + d)); if (r <= 0) return -1; d += (size_t)r; }
        off += sz[k];
    }
    posix_fadvise(v->fd, (off_t)((size_t)i * wb), (off_t)wb, POSIX_FADV_DONTNEED);
    return 0;
}
/* this GPU's z slab of a batch: rows of the [rows][P][P][P] arrays, planes [z0, z0 + g_Dl); mask halo planes from zeros */
static void upload_slab(gpu_state *d, const batch *b, int i, int B, int P, int async) {
    void (*cp)(void *, const void *, size_t) = async ? nn_h2d_copy_stream : nn_h2d;
    const size_t p2 = (size_t)P * P, p3 = p2 * P, l3 = p2 * g_Dl, own = l3 - g_h0 * p2;
    const size_t z0 = d->side ? (size_t)(P / 2 - g_h0) : 0, xb = xbytes();
    for (int r = 0; r < B * g_xs; r++) cp((char *)d->xb[i] + r * l3 * xb, (const char *)bx(b) + (r * p3 + z0 * p2) * xb, l3 * xb);   /* (--gpu-input excludes --split) */
    for (int r = 0; r < B * NCH; r++) cp(d->tb[i] + trow(b, r) * l3, b->t + r * p3 + z0 * p2, l3);
    for (int n = 0; n < B; n++) {
        uint8_t *m = d->mb[i] + n * l3;
        if (d->side) { cp(m, g_zeros, g_h0 * p2); cp(m + g_h0 * p2, b->m + n * p3 + (z0 + g_h0) * p2, own); }
        else { cp(m, b->m + n * p3, own); cp(m + own, g_zeros, g_h0 * p2); }
    }
    cp(d->wb[i], trow_w(b, B, d->hwh[i]), (size_t)B * g_tc);
    if (g_aff) for (int n = 0; n < B; n++) cp(d->bandb[i] + n * l3, b->band + n * p3 + z0 * p2, l3);   /* halo planes too: partners of owned voxels */
    if (g_side) for (int n = 0; n < B; n++) cp(d->sideb[i] + n * l3, b->side + n * p3 + z0 * p2, l3);   /* halo planes: masked at the loss */
    if (pm_used()) for (int n = 0; n < B; n++) {   /* (unwrap2: no ink plane) the papyrus mask, halo planes zeroed like the label mask */
        uint8_t *pm = d->pmb[i] + n * l3;
        if (d->side) { cp(pm, g_zeros, g_h0 * p2); cp(pm + g_h0 * p2, b->pm + n * p3 + (z0 + g_h0) * p2, own); }
        else { cp(pm, b->pm + n * p3, own); cp(pm + own, g_zeros, g_h0 * p2); }
    }
    if (g_cond) for (int n = 0; n < B; n++) cp(d->wdb[i] + n * l3, b->wind + n * p3 + z0 * p2, l3);   /* (--cond excludes --split: whole windows) */
    if (g_unwrap2) for (int n = 0; n < B; n++) { cp(d->w4b[i] + n * l3, b->w4 + n * p3 + z0 * p2, l3); cp(d->wcb[i] + n * l3, b->wcomp + n * p3 + z0 * p2, l3); }   /* halo planes: masked at the loss (the phase mask) */
    if (g_pap_head) for (int n = 0; n < B; n++) cp(d->papb[i] + n * l3, b->pap + n * p3 + z0 * p2, l3);   /* (--pap-head excludes --split) */
}
/* task restore: the clean window (target channel 0) and its weight byte; the input is built on the device */
static void restore_upload(gpu_state *d, const batch *b, int i, int B, int P, int async) {
    void (*cp)(void *, const void *, size_t) = async ? nn_h2d_copy_stream : nn_h2d;
    cp(d->tb[i], b->t, (size_t)B * g_tc * g_Pz * P * P); cp(d->wb[i], b->w, (size_t)B * g_tc);
}
/* synchronous upload (pageable host memory: validation batches) */
/* --gpu-aug 2 with --task unwrap(2): the papyrus mask of a warped window is its warped packed mask bit, written by the targets' warp (lab_upload
   kind 0's db); the batch's pm planes then go with the packed targets */
static int lw_pm(const batch *b) { return g_unwrap && g_pack && g_gpu_aug == 2 && b->ct8; }
static int pm_used(void) { return g_unwrap && (!g_unwrap2 || g_pap_head); }
/* one label kind of a batch (nn_label_warp's: 0 packed targets / mask, 1 phase, 2 winding mod 4 + component) into da (db): --gpu-aug 2
   batches (batch.ct8) carry source-grid labels for windows with a warped radial record, warped on the device (rad: the records' device
   copy, uploaded first by ct_upload); the rest are copies */
static void lab_upload(int kind, const batch *b, uint8_t *da, uint8_t *db, const uint8_t *ha, const uint8_t *hb, const void *rad, int B, int P, int async) {
    void (*cp)(void *, const void *, size_t) = async ? nn_h2d_copy_stream : nn_h2d;
    const size_t p3 = (size_t)g_Pz * P * P;
    if (g_gpu_aug != 2 || !b->ct8) { cp(da, ha, (size_t)B * p3); if (db) cp(db, hb, (size_t)B * p3); return; }
    const float *cs, *sn; sample_phase_cssn(&cs, &sn);
    for (int s = 0; s < B; s++) {
        int32_t h[8]; memcpy(h, (const char *)b->rad + (size_t)s * NN_RAD_REC(g_Pz), sizeof h);
        if (h[6] & 1) nn_label_warp(kind, ha + (size_t)s * p3, hb ? hb + (size_t)s * p3 : nullptr, (const char *)rad + (size_t)s * NN_RAD_REC(g_Pz), P,
                                    da + (size_t)s * p3, db ? db + (size_t)s * p3 : nullptr, cs, sn, async);
        else { cp(da + (size_t)s * p3, ha + (size_t)s * p3, p3); if (db) cp(db + (size_t)s * p3, hb + (size_t)s * p3, p3); }
    }
}
/* the input channel(s) and radial records of a batch into the device buffers x / rad (async: on the copy stream). --gpu-aug batches
   (batch.ct8): the records first, then each window's CT channel built on the device from its bytes (nn_aug_ct) */
static void ct_upload(const batch *b, void *x, void *rad, int B, int P, int async) {
    const size_t p3 = (size_t)g_Pz * P * P;
    if (b->ct8) {
        if (async) nn_h2d_copy_stream(rad, b->rad, rad_bytes(B)); else nn_h2d(rad, b->rad, rad_bytes(B));
        for (int s = 0; s < B; s++) nn_aug_ct(b->ct8 + (size_t)s * p3, (const scan_ct_rec *)b->aug + s, (const char *)rad + (size_t)s * NN_RAD_REC(g_Pz), P, (char *)x + (size_t)s * p3 * xbytes(), async);
        return;
    }
    if (async) nn_h2d_copy_stream(x, bx(b), (size_t)B * g_xs * p3 * xbytes()); else nn_h2d(x, bx(b), (size_t)B * g_xs * p3 * xbytes());
    if (g_gpu_in) { if (async) nn_h2d_copy_stream(rad, b->rad, rad_bytes(B)); else nn_h2d(rad, b->rad, rad_bytes(B)); }
}
/* diagnostic UFSM_DUMP_WIN=PREFIX (with UFSM_SYNC_UPLOAD=1 and --gpu-input 0, cubes): the first UFSM_DUMP_N (8) host windows as
   PREFIX-NN.jpg, rows z-mid / y-mid / x-mid, columns CT channel | phase hue over the CT | winding mod 4 hue over the CT (unknown: CT only) */
int jpeg_write_rgb(const char *path, int w, int h, const uint8_t *rgb, int quality);   /* (imgops.h) */
static void dump_win(const batch *b, int B, int P) {
    static int nd = 0; const char *pre = getenv("UFSM_DUMP_WIN"); const int N = getenv("UFSM_DUMP_N") ? atoi(getenv("UFSM_DUMP_N")) : 8;
    if (!pre || nd >= N || g_Pz != P || !b->x16 || !g_xfmt) return;
    const size_t p3 = (size_t)P * P * P;
    uint8_t *img = malloc((size_t)9 * P * P * 3); if (!img) return;
    for (int s = 0; s < B && nd < N; s++, nd++) {
        const uint16_t *x = b->x16 + (size_t)s * g_xs * p3;
        for (int row = 0; row < 3 * P; row++) {
            const int a = row % P, pl = row / P;
            for (int c = 0; c < 3; c++) for (int k = 0; k < P; k++) {
                const size_t vi = pl == 0 ? ((size_t)(P / 2) * P + a) * P + k : pl == 1 ? ((size_t)a * P + P / 2) * P + k : ((size_t)a * P + k) * P + P / 2;   /* z-mid (y = a), y-mid (z = a), x-mid (z = a, y = k) */
                float v; if (g_xfmt == 1) { _Float16 h; memcpy(&h, x + vi, 2); v = (float)h; } else { uint32_t u = (uint32_t)x[vi] << 16; memcpy(&v, &u, 4); }
                const int g = (int)(128 + 40 * v), gc = g < 0 ? 0 : g > 255 ? 255 : g;
                uint8_t *o = img + ((size_t)row * 3 * P + (size_t)c * P + k) * 3; o[0] = o[1] = o[2] = (uint8_t)gc;
                const uint8_t *L = c == 1 ? b->side : c == 2 ? b->w4 : nullptr;
                if (L && L[(size_t)s * p3 + vi] < 252) {
                    const double h = L[(size_t)s * p3 + vi] / 252.0 * (c == 2 ? 4 : 1);   /* (w4: 63 codes a turn: the hue cycles every turn) */
                    const double cr[3] = {.5 + .5 * cos(2 * M_PI * h), .5 + .5 * cos(2 * M_PI * (h - 1. / 3)), .5 + .5 * cos(2 * M_PI * (h - 2. / 3))};
                    for (int d = 0; d < 3; d++) o[d] = (uint8_t)((gc + 255 * cr[d]) / 2);
                }
            }
        }
        char path[1100]; snprintf(path, sizeof path, "%s-%02d.jpg", pre, nd);
        if (jpeg_write_rgb(path, 3 * P, 3 * P, img, 88)) fprintf(stderr, "cannot write %s\n", path);
        else fprintf(stderr, "dumped window %s (source %d, corner %lld %lld %lld)\n", path, b->src ? b->src[s] : -1,
                     b->corner ? (long long)b->corner[s][0] : -1LL, b->corner ? (long long)b->corner[s][1] : -1LL, b->corner ? (long long)b->corner[s][2] : -1LL);
    }
    free(img);
}
static void upload(gpu_state *d, const batch *b, int B, int P) {
    dump_win(b, B, P);
    if (d->upsplit || d->hsplit) nn_event_sync(d->ev_up[d->cur]);   /* (an asynchronous head / tail into the same buffer may still be in flight) */
    d->sheet=b->sheet?b->sheet[0]:nullptr;
    d->um = d->umb[d->cur] = b->um ? b->um[0] : 0.f;
    if (b->corner && b->level) { memcpy(d->cornb[d->cur], b->corner[0], sizeof d->cornb[0]); d->lvlb[d->cur] = b->level[0]; }   /* validation copies carry none */
    d->srcb[d->cur] = b->src ? b->src[0] : -1;
    axr_take(d, b, d->cur, B, P); if (g_restore || d->side >= 0) d->axv[d->cur] = 0; axr_set(d, d->cur); chain_take(d, b, d->cur, B); view_take(d, b, d->cur, B, 0);
    if (g_restore) { restore_upload(d, b, d->cur, B, P, 0); return; }
    if (d->side >= 0) { upload_slab(d, b, d->cur, B, P, 0); return; }
    size_t p3 = (size_t)g_Pz * P * P;
    ct_upload(b, d->x, d->rad, B, P, 0);
    if (g_tm_unused) {}
    else if (g_pack) { lab_upload(0, b, d->m, lw_pm(b) ? d->pm : nullptr, b->tm, b->pm, d->rad, B, P, 0); if (!g_pkf) nn_unpack_tm(d->t, d->m, B, p3, 0); }   /* (--pack-targets excludes --heads; the
                                                                                                       direct pack loss reads the codes) */
    else {
        if (!g_heads) nn_h2d(d->t, b->t, (size_t)B * NCH * p3);
        else for (int r = 0; r < B * NCH; r++) nn_h2d(d->t + (size_t)trow(b, r) * p3, b->t + (size_t)r * p3, p3);
        nn_h2d(d->m, b->m, (size_t)B * p3);
    }
    nn_h2d(d->w, trow_w(b, B, d->hwh[d->cur]), (size_t)B * g_tc);
    if (g_aff) nn_h2d(d->band, b->band, (size_t)B * p3);
    if (g_side) lab_upload(1, b, d->sdb, nullptr, b->side, nullptr, d->rad, B, P, 0);
    if (pm_used() && !lw_pm(b)) nn_h2d(d->pm, b->pm, (size_t)B * p3);   /* (lw_pm: with the packed targets; unwrap2: no ink plane read) */
    if (g_cond) nn_h2d(d->wd, b->wind, (size_t)B * p3);
    if (g_unwrap2) lab_upload(2, b, d->w4, d->wc, b->w4, b->wcomp, d->rad, B, P, 0);
    if (g_distill) nn_h2d(d->embt, b->emb, g_distill_n);
    if (g_pap_head) nn_h2d(d->pap, b->pap, (size_t)B * p3);
    if (b->ct8) nn_sync();   /* (--gpu-aug: nn_aug_ct / nn_label_warp copy from the pinned batch asynchronously; the caller may release it now) */
}
/* asynchronous upload of a pinned sampler batch into buffer i on the copy stream, after the compute that last used it */
static void upload_async(gpu_state *d, const batch *b, int i, int B, int P) {
    d->sheet=b->sheet?b->sheet[0]:nullptr;
    d->umb[i] = b->um ? b->um[0] : 0.f;
    if (b->corner && b->level) { memcpy(d->cornb[i], b->corner[0], sizeof d->cornb[0]); d->lvlb[i] = b->level[0]; }
    d->srcb[i] = b->src ? b->src[0] : -1;
    axr_take(d, b, i, B, P); if (g_restore || d->side >= 0) d->axv[i] = 0; chain_take(d, b, i, B); view_take(d, b, i, B, 1);
    size_t p3 = (size_t)g_Pz * P * P;
    if (d->upsplit) {   /* the head now, after the backward's last read of X2 and the records; the tail with upload_tail */
        nn_stream_wait(1, unet_l0_3buf_head_free(d->u));
        ct_upload(b, d->xb[i], d->radb[i], B, P, 1);
        d->tail_due = b;
        return;
    }
    if (d->hsplit) {   /* the derived-target planes now (read last by this step's head slabs, ev_hs); input and packed targets with upload_tail */
        nn_stream_wait(1, d->ev_hs);
        if (!b->ct8) {   /* (--gpu-aug batches: the label warps read the radial records, which go with the input in the tail) */
            if (g_side) nn_h2d_copy_stream(d->sideb[i], b->side, (size_t)B * p3);
            if (pm_used() && !lw_pm(b)) nn_h2d_copy_stream(d->pmb[i], b->pm, (size_t)B * p3);   /* (unwrap2: no ink plane read) */
            if (g_unwrap2) { nn_h2d_copy_stream(d->w4b[i], b->w4, (size_t)B * p3); nn_h2d_copy_stream(d->wcb[i], b->wcomp, (size_t)B * p3); }
        }
        d->tail_due = b;
        return;
    }
    nn_stream_wait(1, d->ev_done[i]);   /* the step two back: this upload overlaps the current step */
    if (d->lean) nn_stream_wait(1, d->ev_done[i ^ 1]);   /* lean: one buffer (inside the gradient buffers), free after the current step */
    if (g_restore) restore_upload(d, b, i, B, P, 1);
    else if (d->side >= 0) upload_slab(d, b, i, B, P, 1);
    else {
        ct_upload(b, d->xb[i], d->radb[i], B, P, 1);
        if (g_tm_unused) {}
        else if (g_pack) { lab_upload(0, b, d->mb[i], lw_pm(b) ? d->pmb[i] : nullptr, b->tm, b->pm, d->radb[i], B, P, 1); if (!g_pkf) nn_unpack_tm(d->tb[i], d->mb[i], B, p3, 1); }   /* (copy stream) */
        else {
            if (!g_heads) nn_h2d_copy_stream(d->tb[i], b->t, (size_t)B * NCH * p3);
            else for (int r = 0; r < B * NCH; r++) nn_h2d_copy_stream(d->tb[i] + (size_t)trow(b, r) * p3, b->t + (size_t)r * p3, p3);
            nn_h2d_copy_stream(d->mb[i], b->m, (size_t)B * p3);
        }
        nn_h2d_copy_stream(d->wb[i], trow_w(b, B, d->hwh[i]), (size_t)B * g_tc);
        if (g_aff) nn_h2d_copy_stream(d->bandb[i], b->band, (size_t)B * p3);
        if (g_side) lab_upload(1, b, d->sideb[i], nullptr, b->side, nullptr, d->radb[i], B, P, 1);
        if (pm_used() && !lw_pm(b)) nn_h2d_copy_stream(d->pmb[i], b->pm, (size_t)B * p3);   /* (lw_pm: with the packed targets) */
        if (g_cond) nn_h2d_copy_stream(d->wdb[i], b->wind, (size_t)B * p3);
        if (g_unwrap2) lab_upload(2, b, d->w4b[i], d->wcb[i], b->w4, b->wcomp, d->radb[i], B, P, 1);
        if (g_distill) nn_h2d_copy_stream(d->embb[i], b->emb, g_distill_n);
        if (g_pap_head) nn_h2d_copy_stream(d->papb[i], b->pap, (size_t)B * p3);
    }
    nn_event_record(d->ev_up[i], 1);
}
/* upsplit: the tail of the batch upload_async left (the step's end: the stem's gradient in X3 read; issued with the optimiser, after the
   loss and gradient-norm fetches, whose small copies would otherwise wait behind it where copies run one at a time); ev_up then covers
   head and tail */
static void upload_tail(gpu_state *d, int B, int P) {
    const batch *b = d->tail_due;
    if (!b) return;
    const int i = d->cur;   /* (lean: both buffer indices alias) */
    const size_t p3 = (size_t)g_Pz * P * P;
    nn_stream_wait(1, d->ev_done[i]); nn_stream_wait(1, d->ev_done[i ^ 1]);
    if (d->hsplit) {   /* the input (inside gout[0]: the stem's weight gradient read it until the step's end); --gpu-aug batches: then their
                          label planes, warped on the device with the radial records just uploaded (upload_async left them) */
        ct_upload(b, d->xb[i], d->radb[i], B, P, 1);
        if (b->ct8) {
            if (g_side) lab_upload(1, b, d->sideb[i], nullptr, b->side, nullptr, d->radb[i], B, P, 1);
            if (pm_used() && !lw_pm(b)) nn_h2d_copy_stream(d->pmb[i], b->pm, (size_t)B * p3);
            if (g_unwrap2) lab_upload(2, b, d->w4b[i], d->wcb[i], b->w4, b->wcomp, d->radb[i], B, P, 1);
        }
    }
    if (g_tm_unused) {}
    else if (g_pack) { lab_upload(0, b, d->mb[i], lw_pm(b) ? d->pmb[i] : nullptr, b->tm, b->pm, d->radb[i], B, P, 1); nn_unpack_tm(d->tb[i], d->mb[i], B, p3, 1); }
    else {
        if (!g_heads) nn_h2d_copy_stream(d->tb[i], b->t, (size_t)B * NCH * p3);
        else for (int r = 0; r < B * NCH; r++) nn_h2d_copy_stream(d->tb[i] + (size_t)trow(b, r) * p3, b->t + (size_t)r * p3, p3);
        nn_h2d_copy_stream(d->mb[i], b->m, (size_t)B * p3);
    }
    nn_h2d_copy_stream(d->wb[i], trow_w(b, B, d->hwh[i]), (size_t)B * g_tc);
    if (g_distill) nn_h2d_copy_stream(d->embb[i], b->emb, g_distill_n);
    nn_event_record(d->ev_up[i], 1);
    d->tail_due = nullptr;
}

/* a non-finite forward: statistics of the input, targets and logits of the batch just computed (device buffer cur^1,
   since select_buf already moved on) and a raw dump <out>/nan_step<N>.bin (x16|x32, t, m, w, logits) */
static void diagnose_nan(gpu_state *d, int B, int P, int step, const char *out) {
    nn_init(d->dev); nn_sync();
    size_t p3 = (size_t)g_Pz * P * P, nx = (size_t)B * g_xs * p3, nl = (size_t)B * g_tc * p3;   /* (--gpu-input: the CT channel) */
    int cb = d->cur ^ 1;
    size_t xbytes_ = nx * xbytes();
    void *hx = malloc(xbytes_); nn_d2h(hx, d->xb[cb], xbytes_);
    uint8_t *ht = malloc((size_t)B * g_tc * p3), *hm = malloc((size_t)B * p3), *hw = malloc((size_t)B * g_tc);
    if (g_tm_unused) { memset(ht, 0, (size_t)B * g_tc * p3); memset(hm, 0, (size_t)B * p3); }   /* (unwrap2: no target / mask on the device) */
    else { nn_d2h(ht, d->tb[cb], (size_t)B * g_tc * p3); nn_d2h(hm, d->mb[cb], (size_t)B * p3); } nn_d2h(hw, d->wb[cb], (size_t)B * g_tc);
    float *hl = malloc(nl * 4);
    if (!d->lg) memset(hl, 0, nl * 4);   /* head slabs: no whole-window logits (reported as 0) */
    else if (unet_logits_h16(d->u)) { _Float16 *h16 = malloc(nl * 2); nn_d2h(h16, d->lg, nl * 2); for (size_t k = 0; k < nl; k++) hl[k] = (float)h16[k]; free(h16); }
    else nn_d2h(hl, d->lg, nl * 4);
    double xmax = 0; size_t xnf = 0;
    for (size_t k = 0; k < nx; k++) {
        float v;
        if (!g_xfmt) v = ((float *)hx)[k];
        else if (g_xfmt == 1) { _Float16 h; memcpy(&h, (uint16_t *)hx + k, 2); v = (float)h; }
        else { uint32_t u = (uint32_t)((uint16_t *)hx)[k] << 16; memcpy(&v, &u, 4); }
        if (!isfinite(v)) xnf++; else if (fabs(v) > xmax) xmax = fabs(v);
    }
    double lmax = 0; size_t lnf = 0, mcount = 0, tpos = 0;
    for (size_t k = 0; k < nl; k++) { if (!isfinite(hl[k])) lnf++; else if (fabs(hl[k]) > lmax) lmax = fabs(hl[k]); }
    for (size_t k = 0; k < (size_t)B * p3; k++) mcount += hm[k] != 0;
    for (size_t k = 0; k < (size_t)B * g_tc * p3; k++) tpos += ht[k] != 0 && ht[k] != 255;
    fprintf(stderr, "step %d: non-finite forward: input non-finite %zu max|x| %.3g; logits non-finite %zu max|logit| %.3g; mask %zu of %zu; target>0 %zu; w", step, xnf, xmax, lnf, lmax, mcount, (size_t)B * p3, tpos);
    for (int i = 0; i < B * g_tc; i++) fprintf(stderr, " %d", hw[i]);
    fprintf(stderr, "\n");
    char fn[1500]; snprintf(fn, sizeof fn, "%s/nan_step%d.bin", out, step);
    FILE *f = fopen(fn, "wb");
    if (f) { int hdr[4] = {B, P, g_xfmt, g_tc}; fwrite(hdr, 4, 4, f); fwrite(hx, 1, xbytes_, f); fwrite(ht, 1, (size_t)B * g_tc * p3, f); fwrite(hm, 1, (size_t)B * p3, f); fwrite(hw, 1, (size_t)B * g_tc, f); fwrite(hl, 4, nl, f); fclose(f); fprintf(stderr, "  batch dumped to %s\n", fn); }
    free(hx); free(ht); free(hm); free(hw); free(hl);
}
typedef struct { uint64_t index; double value; } sheet_coo;
static int coo_cmp(const void *a,const void *b) { uint64_t x=((const sheet_coo *)a)->index,y=((const sheet_coo *)b)->index; return (x>y)-(x<y); }
static void sheet_objective(gpu_state *d,shape5 os,int P,int train) {
    const sheet_batch *b=d->sheet; d->sheet_loss=0; memset(d->sheet_parts,0,sizeof d->sheet_parts);
    if (!g_sheet || !b || !b->np) return;
    float *xyz=malloc(3*b->np*sizeof *xyz); double *v=malloc(2*b->np*sizeof *v),*grad=malloc(2*b->np*sizeof *grad);
    for (size_t p=0;p<b->np;p++) memcpy(xyz+3*p,b->points[p].xyz,3*sizeof(float));
    int z0=d->side==1?P/2-g_h0:0,lo=d->side==1?P/2:0,hi=d->side==0?P/2:P;
    nn_sheet_gather(d->lg,os,xyz,b->np,z0,lo,hi,v);
    for (size_t p=0;p<b->np;p++) v[2*p+1]+=b->points[p].q0;
    d->sheet_loss=sheet_loss(b,v,grad,d->sheet_parts,g_sheet_ramp,g_sheet_variant);
    if (train) {
        sheet_coo *coo=malloc(16*b->np*sizeof *coo); size_t count=0,S=(size_t)os.d*P*P;
        for (size_t p=0;p<b->np;p++) {
            int base[3]; double f[3]; for (int a=0;a<3;a++) { base[a]=(int)floor(xyz[3*p+a]); f[a]=xyz[3*p+a]-base[a]; }
            for (int dz=0;dz<2;dz++) for (int dy=0;dy<2;dy++) for (int dx=0;dx<2;dx++) {
                int gz=base[0]+dz,z=gz-z0,y=base[1]+dy,x=base[2]+dx;
                if (gz<lo || gz>=hi || z<0 || z>=os.d || y<0 || y>=P || x<0 || x>=P) continue;
                double w=(dz?f[0]:1-f[0])*(dy?f[1]:1-f[1])*(dx?f[2]:1-f[2]); if (w==0) continue;
                size_t idx=((size_t)z*P+y)*P+x;
                for (int c=0;c<2;c++) if (grad[2*p+c]!=0) coo[count++]=(sheet_coo){idx+c*S,w*grad[2*p+c]};
            }
        }
        qsort(coo,count,sizeof *coo,coo_cmp); uint64_t *idx=malloc((count?count:1)*sizeof *idx); float *dg=malloc((count?count:1)*sizeof *dg); size_t n=0;
        for (size_t i=0;i<count;) { uint64_t k=coo[i].index; double value=0; do { value+=coo[i++].value; } while (i<count && coo[i].index==k); idx[n]=k; dg[n++]=(float)value; }
        nn_sheet_scatter(d->gl,idx,dg,n,g_g16); free(coo); free(idx); free(dg);
    }
    free(xyz); free(v); free(grad);
}
/* forward + loss kernels on the uploaded batch of this GPU; fills gl when train. */
/* the verso teacher labels every sheet: supervise every voxel this GPU owns (split: not the halo planes) */
#define VMASK_INIT() do { if (!d->vmask) { \
        const size_t p2_ = (size_t)xs.h * xs.w, hz_ = d->side >= 0 ? (size_t)g_h0 * p2_ : 0; \
        uint8_t *h_ = malloc(l3); memset(h_, 1, l3); \
        if (d->side == 0) memset(h_ + l3 - hz_, 0, hz_); else if (d->side == 1) memset(h_, 0, hz_); \
        d->vmask = nn_malloc(l3); nn_h2d(d->vmask, h_, l3); free(h_); \
        d->vw = nn_malloc(1); static const uint8_t one_ = 1; nn_h2d(d->vw, &one_, 1); \
        d->vscratch = nn_malloc(nn_loss_scratch((shape5){1, 1, xs.d, xs.h, xs.w}) + 64); } } while (0)
static void verso_target(gpu_state *d, shape5 xs) {
    nn_set_logits_h16(unet_logits_h16(d->u));   /* nn_prob_u8 reads the logits in the network's storage format */
    const size_t l3 = (size_t)xs.d * xs.h * xs.w;   /* B == 1: x is [4][l3], t is [NCH][l3] (this GPU's slab, halo planes included) */
    const int h16 = g_xfmt != 0, check = ufsm_env_on("UFSM_VERSO_CHECK");
    const size_t xb = 4 * l3 * (h16 ? 2 : 4);
    uint8_t *h0 = nullptr;
    if (check) { h0 = malloc(xb + 2 * l3); nn_d2h(h0, d->x, xb); nn_d2h(h0 + xb, d->t, l3); nn_d2h(h0 + xb + l3, d->m, l3); }
    nn_flip_sign((uint8_t *)d->x + l3 * (h16 ? 2 : 4), 3 * l3, h16);   /* radial channels 1..3 */
    if (d->teacher) unet_use_params(d->u, d->teacher);
    const float *lg = unet_forward_x(d->u, d->x, xs, 0, h16);
    if (d->teacher) unet_use_params(d->u, nullptr);
    shape5 os = unet_out_shape(d->u, xs);
    if (os.d != xs.d || os.h != xs.h || os.w != xs.w) { fprintf(stderr, "verso: output grid differs from the input grid\n"); abort(); }
    if (g_verso_ridge) {   /* soft teacher output into the (not yet used) logit-gradient buffer, then its thin ridge into the verso target */
        if (!d->vsoft) d->vsoft = nn_malloc(l3);
        uint8_t *soft = d->vsoft; nn_prob_u8(lg, l3, soft, 0.f);
        nn_ridge_u8(soft, d->x, g_xfmt, xs.d, xs.h, xs.w, g_verso_ridge, d->t + l3);
    } else nn_prob_u8(lg, l3, d->t + l3, g_verso_hard);
    nn_flip_sign((uint8_t *)d->x + l3 * (h16 ? 2 : 4), 3 * l3, h16);
    if (g_verso_anchor) {   /* the same teacher with the direction as is: recto pseudo-labels where the recto target is not trusted */
        if (d->teacher) unet_use_params(d->u, d->teacher);
        const float *la = unet_forward_x(d->u, d->x, xs, 0, h16);
        if (d->teacher) unet_use_params(d->u, nullptr);
        uint8_t *soft = d->vsoft ? d->vsoft : (d->vsoft = nn_malloc(l3)), *dst = d->vsoft;
        if (g_verso_ridge && g_verso_anchor == 1) {   /* 1: thin ridge where unlabelled; 2: soft where unlabelled; 3: soft everywhere (recto distilled) */
            if (!d->vanch) d->vanch = nn_malloc(l3);
            nn_prob_u8(la, l3, soft, 0.f); nn_ridge_u8(soft, d->x, g_xfmt, xs.d, xs.h, xs.w, g_verso_ridge, d->vanch); dst = d->vanch;
        } else nn_prob_u8(la, l3, dst, 0.f);
        VMASK_INIT();
        if (g_verso_anchor == 3) { nn_d2d(d->t, dst, l3); nn_d2d(d->m, d->vmask, l3); static const uint8_t one = 1; nn_h2d(d->w, &one, 1); }
        else nn_fill_unlabelled(d->t, d->m, d->vmask, dst, l3);
    }
    VMASK_INIT();
    if (g_verso_flip) nn_flip_sign((uint8_t *)d->x + l3 * (h16 ? 2 : 4), 3 * l3, h16);   /* swap step: train on the reversed direction */
    if (check && !g_verso_flip) {   /* the eval forward must leave the batch buffers (inside the gradient buffers in lean modes) intact */
        uint8_t *h1 = malloc(xb + 2 * l3); nn_d2h(h1, d->x, xb); nn_d2h(h1 + xb, d->t, l3); nn_d2h(h1 + xb + l3, d->m, l3);
        fprintf(stderr, "verso check (GPU %d): input %s, recto target %s, mask %s\n", d->dev, memcmp(h0, h1, xb) ? "CHANGED" : "intact",
                memcmp(h0 + xb, h1 + xb, l3) ? "CHANGED" : "intact", memcmp(h0 + xb + l3, h1 + xb + l3, l3) ? "CHANGED" : "intact");
        free(h1); free(h0);
        /* the same eval forward without the reversal: how far the verso target is from the recto output on this window */
        uint8_t *vt = malloc(2 * l3); nn_d2h(vt, d->t + l3, l3);
        if (d->teacher) unet_use_params(d->u, d->teacher);
        const float *lr = unet_forward_x(d->u, d->x, xs, 0, h16);
        if (d->teacher) unet_use_params(d->u, nullptr); nn_prob_u8(lr, l3, d->t + l3, 0.f); nn_d2h(vt + l3, d->t + l3, l3);
        double ad = 0, sa = 0, sb = 0; for (size_t k = 0; k < l3; k++) { ad += abs((int)vt[k] - (int)vt[l3 + k]); sa += vt[k]; sb += vt[l3 + k]; }
        fprintf(stderr, "verso check (GPU %d): mean verso target %.1f, recto output %.1f, mean |difference| %.1f (of 255)\n", d->dev, sa / l3, sb / l3, ad / l3);
        const char *dump = getenv("UFSM_VERSO_DUMP");   /* dir: mid-z planes of CT input, verso target, recto output, recto label */
        if (dump) {
            static int nd; char fn[1200]; snprintf(fn, sizeof fn, "%s/verso_%d_%d.raw", dump, d->dev, nd++);
            const size_t pl = (size_t)xs.h * xs.w, zm = (size_t)xs.d / 2;
            uint8_t *lab = malloc(l3); nn_d2h(lab, d->t, l3);
            uint8_t *xh = malloc(xb); nn_d2h(xh, d->x, xb);
            FILE *f = fopen(fn, "wb");
            if (f) { int hdr[3] = {xs.h, xs.w, h16}; fwrite(hdr, 4, 3, f); fwrite(d->cornb[d->cur], 8, 3, f); fwrite(&d->lvlb[d->cur], 4, 1, f); fwrite(xh + zm * pl * (h16 ? 2 : 4), h16 ? 2 : 4, pl, f);
                     fwrite(vt + zm * pl, 1, pl, f); fwrite(vt + l3 + zm * pl, 1, pl, f); fwrite(lab + zm * pl, 1, pl, f);
                     for (int ch = 1; ch < 4; ch++) fwrite(xh + (ch * l3 + zm * pl) * (h16 ? 2 : 4), h16 ? 2 : 4, pl, f); fclose(f); }
            free(lab); free(xh);
        }
        /* train-mode forward (live weights), direction as is vs reversed: does the training forward see the direction? */
        { uint8_t *a = malloc(2 * l3);
          const float *f1 = unet_forward_x(d->u, d->x, xs, 1, h16); nn_prob_u8(f1, l3, d->t + l3, 0.f); nn_d2h(a, d->t + l3, l3);
          nn_flip_sign((uint8_t *)d->x + l3 * (h16 ? 2 : 4), 3 * l3, h16);
          const float *f2 = unet_forward_x(d->u, d->x, xs, 1, h16); nn_prob_u8(f2, l3, d->t + l3, 0.f); nn_d2h(a + l3, d->t + l3, l3);
          nn_flip_sign((uint8_t *)d->x + l3 * (h16 ? 2 : 4), 3 * l3, h16);
          const float *e1 = unet_forward_x(d->u, d->x, xs, 0, h16); nn_prob_u8(e1, l3, d->t + l3, 0.f); uint8_t *e = malloc(l3); nn_d2h(e, d->t + l3, l3);
          double dt = 0, de = 0, dte = 0; for (size_t k = 0; k < l3; k++) { dt += abs((int)a[k] - (int)a[l3 + k]); de += abs((int)vt[k] - (int)vt[l3 + k]); dte += abs((int)a[k] - (int)e[k]); }
          fprintf(stderr, "verso check (GPU %d): |normal - reversed| train-mode %.1f, eval-mode teacher %.1f; |train - eval| normal %.1f\n", d->dev, dt / l3, de / l3, dte / l3);
          free(a); free(e); }
        nn_h2d(d->t + l3, vt, l3); free(vt);
    }
}
/* head slabs (unwrap, B 1, no split; channels 0 recto, 1 ink, 2-3 phase, 4-10 normal / density): the 1^3 head, the targets derived
   from the phase (sidet / sidem / nt / nm, slab-sized) and the four losses over z-slabs of g_hs_z planes, so neither the window's
   11 fp16 logit planes (22 B per voxel) nor its derived targets (11 B) exist whole. pass 0 (after the forward): every loss's
   statistics, summed over the slabs and finished into its scratch (fetch_loss reads them as before); pass 1 (the backward's head
   step, head_slab_cb): the head again, each loss's gradient in place over the slab's logits, the head backward of the slab. The
   per-voxel arithmetic is the whole-window path's (fp64 sums in another order; the head backward's SR seed is the backward's) */
static void head_slab_pass(gpu_state *d, int pass) {
    const int D = g_Pz, H = d->hs_P, W = d->hs_P, Z = d->hs_z;
    const size_t hw = (size_t)H * W, l3 = (size_t)D * hw;
    nn_set_logits_h16(1);
    if (!pass) { nn_zero(d->hs_ds, 64 * sizeof(double)); if (g_unwrap2) nn_zero(d->w4ds, nn_wind4_sums()); }
    for (int z0 = 0; z0 < D; z0 += Z) {
        const int z1 = z0 + Z < D ? z0 + Z : D, nz = z1 - z0;
        const size_t ns = (size_t)nz * hw, vb = (size_t)z0 * hw;
#define LGC(c) ((char *)d->hs_lg + (size_t)(c) * ns * 2)
        if (unet_head_fwd_z(d->u, z0, z1, d->hs_lg)) { fprintf(stderr, "head slabs: the head is not covered (MX activations; UFSM_HEAD_SLAB=0)\n"); abort(); }
        const shape5 one = {1, 1, nz, H, W}, two = {1, 2, nz, H, W}, seven = {1, 7, nz, H, W};
        nn_phase_targets(d->sdb + vb, ns, d->sidet, d->sidem);
        if (g_ptrunc_on) nn_phase_trunc(LGC(g_side_ch), LGC(g_side_ch + 1), d->sidet, d->sidem, ns, g_ptrunc_tau, g_ptrunc_kappa);   /* robust phase loss (both passes: the same mask) */
        if (g_packed_w > 1) nn_packed_mask_z(d->sdb, D, H, W, g_norm_s, d->um, g_packed_um, g_packed_w, d->sidem, z0, z1);   /* packed sheets weigh more */
        nn_unwrap_targets2_z(d->sdb, D, H, W, g_norm_s, g_dens_scale, g_dens_code, d->um, g_packed_um, g_packed_w, d->nt, d->nm, z0, z1);
        if (!pass) {
            if (!g_unwrap2) {
                nn_loss_stats_acc(LGC(0), d->t + vb, d->m + vb, d->w, one, d->hs_ds);
                nn_loss_stats_acc(LGC(1), d->t + l3 + vb, d->pm + vb, d->w + 1, one, d->hs_ds + 5);
            } else nn_wind4_stats(LGC(g_w4_ch), LGC(g_w4_ch + 1), d->w4 + vb, d->wc + vb, d->sidem, ns, d->w4ds);
            nn_loss_stats_acc(LGC(g_side_ch), d->sidet, d->sidem, d->sidew, two, d->hs_ds + 10);
            nn_loss_stats_acc(LGC(g_norm_ch), d->nt, d->nm, d->nw, seven, d->hs_ds + 20);
            if (g_pg_lambda > 0) nn_pgrad_slab(d->hs_train ? nullptr : LGC(g_side_ch), d->hs_train ? nullptr : LGC(g_side_ch + 1), d->sdb, d->sidem, D, H, W, z0, z1, g_norm_s, g_pg_dist, g_pg_nd, g_pg_angle, g_pg_mask, d->hs_ds + 56, nullptr);   /* (training: the pair weights only; the loss sum comes from the gradient pass, + 58) */
            continue;
        }
        if (!g_unwrap2) {
            nn_loss_grad_only(LGC(0), d->t + vb, d->m + vb, d->w, one, d->hs_dice, LGC(0), d->scratch);
            nn_loss_grad_only(LGC(1), d->t + l3 + vb, d->pm + vb, d->w + 1, one, g_ink_dice, LGC(1), d->iscratch);
            if (g_ink_lambda != 1.f) nn_loss_grad_scale(LGC(1), ns, g_ink_lambda);
        } else nn_wind4_grad(LGC(g_w4_ch), LGC(g_w4_ch + 1), d->w4 + vb, d->wc + vb, d->sidem, ns, d->w4fin, g_w4_lambda, LGC(g_w4_ch), LGC(g_w4_ch + 1));
        if (g_pg_lambda > 0) nn_pgrad_slab(LGC(g_side_ch), LGC(g_side_ch + 1), d->sdb, d->sidem, D, H, W, z0, z1, g_norm_s, g_pg_dist, g_pg_nd, g_pg_angle, g_pg_mask, d->hs_ds + 58, d->pg_acc);   /* (before the BCE gradient overwrites the logits; its sums go to + 58: the logged loss of a training step) */
        nn_loss_grad_only(LGC(g_side_ch), d->sidet, d->sidem, d->sidew, two, 0.f, LGC(g_side_ch), d->sscratch);
        if (g_side_lambda != 1.f) nn_loss_grad_scale(LGC(g_side_ch), 2 * ns, g_side_lambda);
        if (g_pg_lambda > 0) nn_pgrad_apply(LGC(g_side_ch), LGC(g_side_ch + 1), d->pg_acc, ns, d->hs_ds + 56, g_pg_lambda);
        nn_loss_grad_only(LGC(g_norm_ch), d->nt, d->nm, d->nw, seven, 0.f, LGC(g_norm_ch), d->nscratch);
        if (g_norm_lambda != 1.f) nn_loss_grad_scale(LGC(g_norm_ch), 7 * ns, g_norm_lambda);
        if (unet_head_bwd_z(d->u, z0, z1, d->hs_lg, z0 == 0)) { fprintf(stderr, "head slabs: the fused head backward is not covered (UFSM_HEAD_SLAB=0)\n"); abort(); }
#undef LGC
    }
    if (!pass) {
        const shape5 w1 = {1, 1, D, H, W}, w2 = {1, 2, D, H, W}, w7 = {1, 7, D, H, W};
        if (!g_unwrap2) {
            nn_loss_stats_fin(d->hs_ds, d->w, w1, d->scratch);
            nn_loss_stats_fin(d->hs_ds + 5, d->w + 1, w1, d->iscratch);
        } else nn_wind4_fin(d->w4ds, d->w4fin);
        nn_loss_stats_fin(d->hs_ds + 10, d->sidew, w2, d->sscratch);
        nn_loss_stats_fin(d->hs_ds + 20, d->nw, w7, d->nscratch);
    }
}
static void head_slab_cb(void *ctx) {
    gpu_state *d = ctx;
    head_slab_pass(d, 1);
    if (d->hsplit) nn_event_record(d->ev_hs, 0);   /* the derived-target planes are free for the next batch (upload_async) */
}
/* tests (env UFSM_TEST_NAN=MODE:STEP[,STEP...]): force a non-finite step on GPU 0 at those steps (where: 0 before the forward, 1 after the
   forward and loss, 2 after the backward). MODE logit: +inf into the first logit-gradient element before the backward (an overflowing loss
   gradient through the whole backward); grad: NaN into the first parameter gradient after it (the skip path alone); input: NaN into the
   first input voxel (a non-finite forward); big: the first 1/8 of the input's CT channel at 60000 (an overflowing forward); zero: every
   parameter gradient 0 after the backward (a window with nothing supervised) */
static void test_nan_inject(gpu_state *d, int step, int where, int P) {
    static int mode = -1; static char steps[256];
    if (mode < 0) {
        const char *e = getenv("UFSM_TEST_NAN"), *names[7] = {"logit:", "grad:", "input:", "big:", "zero:", "zerologit:", "weight:"}; mode = 0;
        for (int k = 0; e && k < 7; k++) if (!strncmp(e, names[k], strlen(names[k]))) { mode = k + 1; snprintf(steps, sizeof steps, ",%s,", e + strlen(names[k])); }
    }
    static const int at[8] = {-1, 1, 2, 0, 0, 2, 1, 0};
    if (!mode || d->dev != 0 || at[mode] != where) return;
    char k[24]; snprintf(k, sizeof k, ",%d,", step);
    if (!strstr(steps, k)) return;
    if (mode == 1) {
        if (!d->gl) return;   /* (head slabs: no whole-window logit gradient) */
        const size_t n = getenv("UFSM_TEST_NAN_N") ? (size_t)atoll(getenv("UFSM_TEST_NAN_N")) : 1;   /* elements (a region: many overflowing blocks) */
        if (g_g16) { uint16_t *h = malloc(2 * n); for (size_t i = 0; i < n; i++) h[i] = g_xfmt == 2 ? 0x7f80 : 0x7c00;   /* (bf16 0x7c00 is still ~5e36) */ nn_h2d(d->gl, h, 2 * n); free(h); }
        else { float *h = malloc(4 * n); for (size_t i = 0; i < n; i++) h[i] = INFINITY; nn_h2d(d->gl, h, 4 * n); free(h); }
    } else if (mode == 2) { const float nan = NAN; nn_h2d(unet_grad_ptr(d->u), &nan, 4); }
    else if (mode == 5) nn_zero(unet_grad_ptr(d->u), unet_nparams(d->u) * 4);
    else if (mode == 6) { if (!d->gl) return; nn_zero(d->gl, (size_t)g_cout * g_Pz * P * P * (g_g16 ? 2 : 4)); }
    else if (mode == 7) { const float nan = NAN; const size_t at_ = getenv("UFSM_TEST_NAN_N") ? (size_t)atoll(getenv("UFSM_TEST_NAN_N")) : unet_nparams(d->u) / 2; nn_h2d((float *)unet_live_params(d->u) + at_, &nan, 4); }
    else {
        const size_t n = mode == 3 && !getenv("UFSM_TEST_NAN_N") ? 1 : mode == 3 ? (size_t)atoll(getenv("UFSM_TEST_NAN_N")) : (size_t)g_Pz * P * P / 8 + 1, eb = xbytes();
        uint8_t *h = malloc(n * eb);
        for (size_t i = 0; i < n; i++) {
            if (eb == 4) { const float v = mode == 3 ? NAN : 60000.f; memcpy(h + 4 * i, &v, 4); }
            else { const uint16_t v = g_xfmt == 2 ? (mode == 3 ? 0x7fc0 : 0x476a) : (mode == 3 ? 0x7e00 : 0x7b53); memcpy(h + 2 * i, &v, 2); }
        }
        nn_h2d(d->x, h, n * eb); free(h);
    }
    static const char *what[8] = {"", "+inf logit gradient", "NaN parameter gradient", "NaN input voxel", "input CT at 60000", "all-zero parameter gradient", "all-zero logit gradient (nothing supervised)", "NaN weight (UFSM_TEST_NAN_N: its index)"};
    fprintf(stderr, "test: step %d: %s injected\n", step, what[mode]);
}
/* the window of the last training step on this GPU (source, level, level-0 corner): which window of a deterministic stream misbehaved */
static void window_note(const gpu_state *d, const sources *S, const char *what) {
    const int b = d->last & 1, si = d->srcb[b];
    fprintf(stderr, "%s on GPU %d: source %d (%s), level %d, level-0 corner %lld,%lld,%lld\n", what, d->dev, si, S && si >= 0 && si < S->n ? S->src[si].name : "?",
            d->lvlb[b], (long long)d->cornb[b][0], (long long)d->cornb[b][1], (long long)d->cornb[b][2]);
}
/* UFSM_SYNC_CHECK=1: synchronise after the forward / losses and after the backward and report a CUDA error there with the window (slow;
   for locating an asynchronous fault, e.g. with CUDA_LAUNCH_BLOCKING=1) */
static const sources *g_Sdiag;
static int sync_check(gpu_state *d, int step, const char *what) {
    static int on = -1; if (on < 0) on = ufsm_env_on("UFSM_SYNC_CHECK");
    if (!on) return 0;
    nn_sync();
    const char *e = nn_check();
    if (!e) return 0;
    fprintf(stderr, "cuda error at step %d after the %s: %s\n", step, what, e);
    window_note(d, g_Sdiag, "  window");
    return 1;
}
/* UFSM_SYNC_CHECK=1 also counts non-finite values: the logits right after the forward (a window whose voxels are all unsupervised hides
   non-finite logits from the losses) and the parameter gradient after the backward */
static void diag_nonfinite(gpu_state *d, const char *what, const void *p, size_t n, int h16) {
    static int on = -1; if (on < 0) on = ufsm_env_on("UFSM_SYNC_CHECK");
    if (!on || !p) return;
    const unsigned long long c = nn_count_nonfinite(p, n, h16);
    if (!c) return;
    fprintf(stderr, "diagnostics: %llu non-finite %s of %zu\n", c, what, n);
    window_note(d, g_Sdiag, "  window");
}
static void run_batch(gpu_state *d, int B, int P, float dice_w, int train) {
    shape5 xs = {B, g_xc, d->side >= 0 ? g_Dl : g_Pz, P, P};
    if (unet_cfg_of(d->u)->scale_cond) { if (g_papyrus) unet_set_cond(d->u, d->um); else unet_set_scale(d->u, d->um); }   /* B == 1 (checked at
                                                                                                    startup); papyrus: um is the cutoff code */
    if (g_verso_on) verso_target(d, xs);
    const int q = P / g_hdiv, hm = unet_hr_mode(g_half);
    const shape5 hs = {B, g_cout, q, q, q};   /* --half-res: the network's grid, the full-resolution output channels */
    mem_pre(d, B, train);
    view_set(d, B);
    const float *lg;
    if (g_restore && g_rpairs) {   /* real pairs: the input crop (target row 1) upsampled x4 onto the window, then (x - 128) / 64 */
        const float st4[3] = {4.f, 4.f, 4.f}, ph0[3] = {0.f, 0.f, 0.f}; const size_t p3_ = (size_t)g_Pz * P * P;
        nn_restore_upsample(d->t + p3_, P / 4, P / 4, P / 4, d->rlow, g_Pz, P, P, st4, ph0, nullptr);
        nn_restore_input(d->rlow, d->x, p3_, g_rsplit, g_xfmt == 1);
    } else if (g_restore) {   /* the input from the clean window: the recipe's degradation, then (x - 128) / 64 */
        if (nn_restore_degrade(d->t, d->rlow, g_Pz, P, P, &d->rcp, d->rscr, d->rscr_b, nullptr)) { fprintf(stderr, "restore: the degradation scratch is too small for this recipe\n"); abort(); }
        nn_restore_input(d->rlow, d->x, (size_t)g_Pz * P * P, g_rsplit, g_xfmt == 1);
    }
    if (g_pkf) {   /* --half-res pack, direct: the network's input from the CT channel (--gpu-input) or the host-built window; its packed
                      logits go to the losses as they are (nn_loss_pack) */
        const shape5 ns = {B, unet_cfg_of(d->u)->cin, q, q, q};
        if (!g_gpu_in) nn_lowres_input(d->x, xs, g_hdiv, hm, d->lrt, d->x2, g_xfmt != 0);
        lg = unet_forward_x(d->u, g_gpu_in ? d->x : d->x2, ns, train, g_xfmt != 0);
    } else if (g_half) {   /* the network on blocks of the window: input and logits through nn_lowres_*, fp32 full-resolution logits for the losses */
        nn_lowres_input(d->x, xs, g_hdiv, hm, d->lrt, d->x2, g_xfmt != 0);
        const float *l2 = unet_forward_x(d->u, d->x2, (shape5){B, unet_cfg_of(d->u)->cin, q, q, q}, train, g_xfmt != 0);
        nn_lowres_output(l2, unet_logits_h16(d->u), hs, g_hdiv, hm, (float *)d->lrt, d->lgf);
        lg = d->lgf;
    } else lg = unet_forward_x(d->u, d->x, xs, train, g_xfmt != 0);
    d->lg = lg;
    mem_post(d, B, train);
    if (train && !g_half && !g_head_slab && !g_pkf) diag_nonfinite(d, "logits", lg, (size_t)B * g_cout * xs.d * P * P, unet_logits_h16(d->u) ? 1 : 0);   /* (UFSM_SYNC_CHECK) */
    d->ds_fresh = g_ds && train;   /* validation batches have no deep-supervision losses */
    if (g_ds && train) {   /* deep supervision: each supervised decoder level's logits against the targets / mask max- / any-pooled to its grid */
        const int save = nn_get_loss_grad_h16(); nn_set_loss_grad_h16(0);   /* fp32 gradients: unet_backward_x converts them with the gradient scale */
        for (int j = 1; j <= g_ds; j++) {
            shape5 sj; const float *lj = unet_ds_logits(d->u, j, &sj);
            if (!lj) { fprintf(stderr, "deep supervision: no level-%d logits\n", j); abort(); }
            nn_pool_targets(d->t, d->m, B, g_tc, xs.d, P, P, 1 << j, d->ds_t[j], d->ds_m[j]);
            nn_loss_async(lj, d->ds_t[j], d->ds_m[j], d->w, sj, dice_w, d->ds_gl[j], d->ds_scr[j]);
            nn_scale(d->ds_gl[j], powf(g_ds_w, (float)j) * g_ds_frac, shape_numel(sj));
            unet_set_ds_grad(d->u, j, d->ds_gl[j]);
        }
        nn_set_loss_grad_h16(save);
    }
    d->emb_fresh = g_distill && train;
    if (g_distill && train) {   /* --distill: the head's cosine loss against the feature targets; the fp32 gradient joins its decoder level in the backward */
        shape5 se; const float *ye = unet_emb_logits(d->u, &se);
        if (!ye) { fprintf(stderr, "distill: no head outputs after the forward\n"); abort(); }
        nn_distill_loss(ye, d->embt, se.n, se.c, (size_t)se.d * se.h * se.w, g_distill_w, d->embg, d->embds);
        unet_set_emb_grad(d->u, d->embg);
    }
    if (g_head_slab) {   /* the head per z-slab: the loss statistics now, the gradients in the backward (head_slab_cb) */
        int prof = ufsm_env_on("UFSM_PROF");
        if (prof) nn_prof_begin(6);
        d->gl = nullptr; d->hs_dice = dice_w; d->hs_train = train;
        head_slab_pass(d, 0);
        if (prof) nn_prof_end();
        return;
    }
    if (d->gl_inplace && train && !unet_logits_h16(d->u)) {   /* this layout kept fp32 logits (non-lean or narrow nets): a separate gradient */
        d->gl_inplace = 0;
        if (!d->gl) d->gl = nn_malloc((size_t)B * g_cout * xs.d * P * P * (g_g16 ? 2 : 4));
        fprintf(stderr, "logit gradient: fp32 logits in this layout, separate gradient buffer\n");
    }
    if (d->gl_inplace && train && unet_logits_in_gx(d->u)) {   /* the fp16 logits sit at the start of gout[0], which the head's backward-data writes
                                                              (MX-fp4 gradients, lean 2): an in-place gradient would be overwritten as it is read, and
                                                              the head would see gout[0] as plain 16-bit storage. The layout's own scratch (the parked
                                                              skip's buffer with the level-0 A share) or a separate buffer instead */
        d->gl_inplace = 0;
        const size_t gb = (size_t)B * g_cout * xs.d * P * P * (g_g16 ? 2 : 4);
        void *pg = d->lean ? unet_logit_grad_scratch(d->u, gb) : nullptr;
        if (pg && !(d->bat_lo && (char *)pg < d->bat_hi && (char *)pg + gb > d->bat_lo)) { d->gl = pg; d->gl_inside = 1; }
        else if (!d->gl) d->gl = nn_malloc(gb);
        fprintf(stderr, "logit gradient: the logits share the head's input-gradient buffer, gradient %s\n", d->gl_inside ? "in the layout's scratch" : "in a separate buffer");
    }
    if (d->gl_inplace && train) {   /* the losses write each element's gradient over its own logit (read first) */
        if (!g_g16) { fprintf(stderr, "in-place logit gradient: the gradient is not 16-bit (UFSM_GL_SEPARATE=1)\n"); abort(); }
        d->gl = (float *)lg;
    }
    if (d->lean && train && !d->gl_inplace) {   /* the logit gradient lives in the model's gradient buffer B (built by the forward above) */
        const size_t gb = (size_t)B * g_cout * xs.d * P * P * (g_g16 ? 2 : 4);
        void *pg = d->gl_inside ? unet_logit_grad_scratch(d->u, gb) : nullptr;
        if (pg) d->gl = pg;   /* (re)built model: its scratch moved; otherwise the separate buffer allocated at setup */
    }
    shape5 os = g_half ? (shape5){B, g_cout, P, P, P} : unet_out_shape(d->u, xs);
    int prof = ufsm_env_on("UFSM_PROF");
    if (prof) nn_prof_begin(6);
    shape5 ts = os; if (g_aff) ts.c = NCH;   /* B == 1: the target channels are the leading output channels */
    nn_set_logits_h16((!g_half || g_pkf) && unet_logits_h16(d->u));   /* lean 2: fp16 logits in the gradient buffer */
    if (g_restore && g_rpairs) nn_restore_pair_loss(lg, unet_logits_h16(d->u), d->rlow, d->t, d->t + (size_t)g_Pz * P * P, g_Pz, P, P, &g_rpw, train ? d->gl : nullptr, g_g16, nn_get_f16(),
                                                    nn_get_grad_scale(), d->rst, d->rrd, !train);
    else if (g_restore) nn_restore_loss(lg, unet_logits_h16(d->u), d->rlow, d->t, g_Pz, P, P, g_reps, g_rmse, g_redge, train ? d->gl : nullptr, g_g16, nn_get_f16(), nn_get_grad_scale(), d->rst);
    else if (g_unwrap2) {}   /* no recto / ink heads */
    else if (g_unwrap) {   /* recto on target 0 with the label mask; ink on target 1 over all papyrus, each its own loss */
        const size_t l3 = (size_t)os.d * os.h * os.w, eb = g_g16 ? 2 : 4, le = unet_logits_h16(d->u) ? 2 : 4;
        const shape5 one = {1, 1, os.d, os.h, os.w};
        nn_loss_async_tol(lg, d->t, d->m, d->w, one, dice_w, train ? d->gl : nullptr, d->scratch, d->tolc);
        nn_loss_async_tol((const float *)((const char *)lg + l3 * le), d->t + l3, d->pm, d->w + 1, one, g_ink_dice, train ? (float *)((char *)d->gl + l3 * eb) : nullptr, d->iscratch, nullptr);
        if (train && g_ink_lambda != 1.f) nn_loss_grad_scale((char *)d->gl + l3 * eb, l3, g_ink_lambda);
    } else if (g_pkf) nn_loss_pack(lg, g_pack ? nullptr : d->t, d->m, d->w, ts, dice_w, train ? d->gl : nullptr, d->scratch);   /* (ts: the full-resolution window;
                                                                                                       --pack-targets: the codes in m) */
    else if (!g_verso_on) nn_loss_async_tol(lg, d->t, d->m, d->w, ts, dice_w, train ? d->gl : nullptr, d->scratch, d->tolc);
    else {   /* recto labels (trust-band mask) and verso teacher targets (every owned voxel), each its own loss. A swap step
                (direction reversed) exchanges the output channels they train: reversing the direction exchanges the faces */
        const size_t l3 = (size_t)os.d * os.h * os.w, eb = g_g16 ? 2 : 4, le = unet_logits_h16(d->u) ? 2 : 4;
        const int cr = g_verso_flip ? 1 : 0, cv = 1 - cr;
        const shape5 one = {1, 1, os.d, os.h, os.w};
        nn_loss_async_tol((const float *)((const char *)lg + cr * l3 * le), d->t, d->m, d->w, one, dice_w, train ? (float *)((char *)d->gl + cr * l3 * eb) : nullptr, d->scratch, d->tolc);
        nn_loss_async_tol((const float *)((const char *)lg + cv * l3 * le), d->t + l3, d->vmask, d->vw, one, dice_w, train ? (float *)((char *)d->gl + cv * l3 * eb) : nullptr, d->vscratch, nullptr);
        if (train && g_verso_lambda != 1.f) nn_loss_grad_scale((char *)d->gl + cv * l3 * eb, l3, g_verso_lambda);
    }
    if (g_verso_on && ufsm_env_on("UFSM_VERSO_CHECK")) {   /* what the losses actually read */
        const size_t l3 = (size_t)os.d * os.h * os.w; uint8_t *h = malloc(2 * l3); nn_sync(); nn_d2h(h, d->t, 2 * l3);
        double a = 0, b = 0, ab = 0; for (size_t k = 0; k < l3; k++) { a += h[k]; b += h[l3 + k]; ab += abs((int)h[k] - (int)h[l3 + k]); }
        fprintf(stderr, "verso check (GPU %d) at the loss: mean recto target %.1f, verso target %.1f, |difference| %.1f\n", d->dev, a / l3, b / l3, ab / l3);
        free(h);
    }
    if (g_aff) nn_aff_loss_async(lg, os, 1 + g_verso, d->band, g_aff_off, dice_w, g_aff_lambda, train ? d->gl : nullptr, d->aff_fin);
    if (g_side) {   /* recto / verso side: BCE over known papyrus voxels (B == 1); split: halo planes are the other GPU's */
        const size_t l3 = (size_t)os.d * os.h * os.w, eb = g_g16 ? 2 : 4, le = unet_logits_h16(d->u) ? 2 : 4;
        if (g_ptrunc_on) { fprintf(stderr, "--phase-trunc needs the head-slab loss path (task unwrap / unwrap2, B 1, no --split; UFSM_HEAD_SLAB on)\n"); abort(); }
        if (g_side == 2) nn_phase_targets(d->sdb, l3, d->sidet, d->sidem); else nn_side_targets(d->sdb, l3, d->sidet, d->sidem);
        if (g_side == 2 && g_unwrap && g_packed_w > 1) nn_packed_mask(d->sdb, os.d, os.h, os.w, g_norm_s, d->um, g_packed_um, g_packed_w, d->sidem);   /* packed sheets weigh more */
        if (ufsm_env_on("UFSM_SIDE_CHECK")) {   /* what the side / phase loss reads: known labels and the mask */
            uint8_t *h = malloc(2 * l3); nn_sync(); nn_d2h(h, d->sdb, l3); nn_d2h(h + l3, d->sidem, l3);
            size_t kn = 0, mk = 0; for (size_t k = 0; k < l3; k++) { kn += h[k] != 255; mk += h[l3 + k] != 0; }
            fprintf(stderr, "side check (GPU %d): %zu of %zu voxels known, %zu in the mask\n", d->dev, kn, l3, mk); free(h);
        }
        if (d->side == 0) nn_zero(d->sidem + l3 - (size_t)g_h0 * P * P, (size_t)g_h0 * P * P);
        else if (d->side == 1) nn_zero(d->sidem, (size_t)g_h0 * P * P);
        const shape5 one = {1, g_side_nc, os.d, os.h, os.w};
        nn_loss_async_tol((const float *)((const char *)lg + g_side_ch * l3 * le), d->sidet, d->sidem, d->sidew, one, 0.f,
                          train ? (float *)((char *)d->gl + g_side_ch * l3 * eb) : nullptr, d->sscratch, nullptr);
        if (train && g_side_lambda != 1.f) nn_loss_grad_scale((char *)d->gl + g_side_ch * l3 * eb, g_side_nc * l3, g_side_lambda);
    }
    if (g_unwrap) {   /* normal n n^T and density from the phase gradient; split: halo planes are the other GPU's */
        const size_t l3 = (size_t)os.d * os.h * os.w, eb = g_g16 ? 2 : 4, le = unet_logits_h16(d->u) ? 2 : 4;
        nn_unwrap_targets2(d->sdb, os.d, os.h, os.w, g_norm_s, g_dens_scale, g_dens_code, d->um, g_packed_um, g_packed_w, d->nt, d->nm);
        if (d->side == 0) nn_zero(d->nm + l3 - (size_t)g_h0 * P * P, (size_t)g_h0 * P * P);
        else if (d->side == 1) nn_zero(d->nm, (size_t)g_h0 * P * P);
        const shape5 seven = {1, 7, os.d, os.h, os.w};
        nn_loss_async_tol((const float *)((const char *)lg + g_norm_ch * l3 * le), d->nt, d->nm, d->nw, seven, 0.f,
                          train ? (float *)((char *)d->gl + g_norm_ch * l3 * eb) : nullptr, d->nscratch, nullptr);
        if (train && g_norm_lambda != 1.f) nn_loss_grad_scale((char *)d->gl + g_norm_ch * l3 * eb, 7 * l3, g_norm_lambda);
        if (g_count_lambda > 0) {   /* turns along sheet-normal segments: the density integral against the true count */
            const size_t dch = (size_t)(g_norm_ch + 6);
            nn_count_loss((const char *)lg + dch * l3 * le, d->sdb, os.d, os.h, os.w, g_norm_s, g_count_K, g_count_L, g_dens_scale,
                          ((uint64_t)d->dev << 48) ^ d->cnt_step++, g_count_lambda, train ? (char *)d->gl + dch * l3 * eb : nullptr, d->cnt_acc, d->cnt_seg, d->cnt_nv, d->cnt_out);
        }
        if (g_unwrap2)   /* the winding mod 4, gauge-free, over the phase mask (its halo planes zeroed above) */
            nn_wind4_loss((const char *)lg + (size_t)g_w4_ch * l3 * le, (const char *)lg + (size_t)(g_w4_ch + 1) * l3 * le, d->w4, d->wc, d->sidem, l3, d->w4ds, d->w4fin, g_w4_lambda,
                          train ? (char *)d->gl + (size_t)g_w4_ch * l3 * eb : nullptr, train ? (char *)d->gl + (size_t)(g_w4_ch + 1) * l3 * eb : nullptr);
    }
    if (g_cond) {   /* the winding mod 14 / mod 2 heads (unknown winding, and samples drawn without a condition, are masked) */
        const size_t l3 = (size_t)os.d * os.h * os.w, eb = g_g16 ? 2 : 4, le = unet_logits_h16(d->u) ? 2 : 4;
        nn_cond_targets(d->wd, l3, d->ct4, d->cm4);
        const shape5 four = {1, 4, os.d, os.h, os.w};
        nn_loss_async_tol((const float *)((const char *)lg + g_cond_ch * l3 * le), d->ct4, d->cm4, d->cw4, four, 0.f,
                          train ? (float *)((char *)d->gl + g_cond_ch * l3 * eb) : nullptr, d->cscratch, nullptr);
        if (train && g_cond_lambda != 1.f) nn_loss_grad_scale((char *)d->gl + g_cond_ch * l3 * eb, 4 * l3, g_cond_lambda);
        nn_zero(d->cacc, 2 * sizeof(unsigned long long)); nn_cond_acc((const char *)lg + g_cond_ch * l3 * le, le == 2, d->wd, l3, d->cacc);
    }
    if (g_pap_head) {   /* papyrus / void: BCE over the scroll's voxels (CT > 0) that the gap labels do not ignore */
        const size_t l3 = (size_t)os.d * os.h * os.w, eb = g_g16 ? 2 : 4, le = unet_logits_h16(d->u) ? 2 : 4;
        nn_pap_targets(d->pap, d->pm, l3, d->pt1, d->pmk1);
        const shape5 one = {1, 1, os.d, os.h, os.w};
        nn_loss_async_tol((const float *)((const char *)lg + g_pap_ch * l3 * le), d->pt1, d->pmk1, d->pw1, one, 0.f,
                          train ? (float *)((char *)d->gl + g_pap_ch * l3 * eb) : nullptr, d->pscr, nullptr);
        if (train && g_pap_lambda != 1.f) nn_loss_grad_scale((char *)d->gl + g_pap_ch * l3 * eb, l3, g_pap_lambda);
    }
    sheet_objective(d,os,P,train);
    if (g_half && !g_pkf && train) nn_lowres_grad(d->gl, g_g16, hs, g_hdiv, hm, d->lrt, d->glh);   /* the logit gradient on the network's grid */
    if (prof) nn_prof_end();
}
/* task restore: the loss sums of the last forward (rst_h); out[0] the Charbonnier term, out[NCH] the edge term */
static double restore_fetch(gpu_state *d, float *out) {
    nn_d2h(d->rst_h, d->rst, (g_rpairs ? RP_NST : 8) * sizeof(double));
    if (g_rpairs) {   /* real pairs: main + edge + MSE + downsample consistency */
        const double *s = d->rst_h, n = s[0] > 0 ? s[0] : 1, edge = s[5] > 0 ? s[6] / s[5] : 0, ds = s[8] > 0 ? s[7] / s[8] : 0;
        for (int c = 0; c < 2 * NCH + 1; c++) out[c] = 0;
        out[0] = (float)(s[1] / n); out[NCH] = (float)edge; out[2 * NCH] = 1;
        return s[1] / n + 0.5 * g_rmse * s[3] / 4096.0 / n + g_redge * edge + g_rpw.ds * ds + g_rpw.unc * s[15] / n;
    }
    const double *s = d->rst_h, n = s[0] > 0 ? s[0] : 1, edge = s[5] > 0 ? s[6] / s[5] : 0;
    for (int c = 0; c < 2 * NCH + 1; c++) out[c] = 0;
    out[0] = (float)(s[1] / n); out[NCH] = (float)edge; out[2 * NCH] = 1;
    return s[1] / n + 0.5 * g_rmse * s[3] / 4096.0 / n + g_redge * edge;
}
static double psnr_of(double sse, double n) { return n > 0 && sse > 0 ? 10.0 * log10(255.0 * 255.0 * n / sse) : 99.0; }
/* real pairs validation (g_rpv: per source the RP_NST sums of its validation windows): per pair and per role (training pairs' held-out boxes,
   "val" pairs) PSNR of output / input against the target (after the intensity map), SSIM over 8^3 blocks, gradient PSNR; restore_val.csv;
   the criterion (best.ckpt) is minus the mean PSNR gain over every validated pair */
static double *g_rpv;
static void restore_pair_report(const sources *S, const char *out, int step, double *vcrit) {
    char cp_[1400]; snprintf(cp_, sizeof cp_, "%s/restore_val.csv", out);
    FILE *rf = fopen(cp_, "a");
    if (rf) { fseek(rf, 0, SEEK_END); if (!ftell(rf)) fprintf(rf, "step,pair,role,voxels,psnr,input_psnr,gain,ssim,input_ssim,grad_psnr,input_grad_psnr\n"); }
    double gs = 0; int ng = 0, nr[2] = {0, 0}; double rs[2][4] = {{0}};
    for (int i = 0; i < S->n; i++) {
        const double *v = g_rpv + (size_t)i * RP_NST; if (!(v[0] > 0) || !S->src[i].srp) continue;
        const int role = S->src[i].srp->role;
        const double p = psnr_of(v[3], v[0]), pb = psnr_of(v[4], v[0]), ss = v[13] > 0 ? v[11] / v[13] : 0, sb = v[13] > 0 ? v[12] / v[13] : 0,
                     gp = psnr_of(v[9], v[5]), gb = psnr_of(v[10], v[5]);
        fprintf(stderr, "val sr %s (%s): psnr %.3f (input %.3f, gain %+.3f) ssim %.4f (input %.4f) grad psnr %.3f (input %.3f)\n", S->src[i].name, role ? "val" : "test",
                p, pb, p - pb, ss, sb, gp, gb);
        if (rf) fprintf(rf, "%d,%s,%s,%.0f,%.4f,%.4f,%.4f,%.5f,%.5f,%.4f,%.4f\n", step, S->src[i].name, role ? "val" : "test", v[0], p, pb, p - pb, ss, sb, gp, gb);
        gs += p - pb; ng++; nr[role]++; rs[role][0] += p - pb; rs[role][1] += ss - sb; rs[role][2] += gp - gb; rs[role][3] += ss;
    }
    for (int r = 0; r < 2; r++) if (nr[r]) fprintf(stderr, "val sr step %d %s pairs (%d): mean gain psnr %+.4f dB ssim %+.5f (ssim %.4f) grad psnr %+.4f dB\n", step, r ? "val" : "held-out test",
                                                    nr[r], rs[r][0] / nr[r], rs[r][1] / nr[r], rs[r][3] / nr[r], rs[r][2] / nr[r]);
    if (rf) fclose(rf);
    if (ng) *vcrit = -gs / ng;
}
/* fetch the loss parts of the last run_batch on this GPU (synchronous); returns the scalar loss */
static double fetch_loss(gpu_state *d, int B, int P, float dice_w, float *out) {
    if (g_restore) return restore_fetch(d, out);
    shape5 os = g_half ? (shape5){B, g_cout, P, P, P} : unet_out_shape(d->u, (shape5){B, g_xc, d->side >= 0 ? g_Dl : g_Pz, P, P});
    if (g_aff) os.c = NCH;
    if (!g_verso_on && !g_unwrap) nn_loss_fetch(d->scratch, os, out);
    d->ink_loss = d->norm_loss = d->w4_loss = 0;
    if (g_unwrap) {   /* recto parts laid out as the NCH-channel fetch does; the ink and normal / density losses */
        for (int c = 0; c < 2 * NCH + 1; c++) out[c] = 0;
        if (!g_unwrap2) {
            float r[3]; nn_loss_fetch(d->scratch, (shape5){1, 1, 1, 1, 1}, r);
            out[0] = r[0]; out[NCH] = r[1]; out[2 * NCH] = r[2];
            nn_loss_fetch(d->iscratch, (shape5){1, 1, 1, 1, 1}, d->iout); d->ink_loss = g_ink_lambda * (d->iout[0] + g_ink_dice * d->iout[1]);
        } else { d->iout[0] = d->iout[1] = d->iout[2] = 0; nn_d2h(d->w4h, d->w4fin, 8 * sizeof(float)); }
        nn_loss_fetch(d->nscratch, (shape5){1, 7, 1, 1, 1}, d->nout);
        float nb = 0; for (int c = 0; c < 7; c++) nb += d->nout[c];
        d->norm_loss = g_norm_lambda * nb / 7;
        if (g_unwrap2) d->w4_loss = g_w4_lambda * d->w4h[0];
    }
    d->emb_loss = d->emb_cells = 0;
    if (g_distill && d->emb_fresh) {   /* --distill: weight x mean 1 - cos over the cells with a target */
        nn_d2h(d->embh, d->embds, 2 * sizeof(double)); d->emb_cells = d->embh[1];
        d->emb_loss = d->embh[1] > 0 ? g_distill_w * d->embh[0] / d->embh[1] : 0;
    }
    d->ds_loss = 0;
    if (g_ds && d->ds_fresh) for (int j = 1; j <= g_ds; j++) {   /* weight x mean over the active channels of bce + dice_w dice */
        shape5 sj; unet_ds_logits(d->u, j, &sj); float *o = d->ds_out[j]; nn_loss_fetch(d->ds_scr[j], sj, o);
        double s = 0; int nc = 0;
        for (int c = 0; c < sj.c && c < g_tc; c++) if (o[c] != 0.f || o[sj.c + c] != 0.f) { s += o[c] + dice_w * o[sj.c + c]; nc++; }
        d->ds_loss += powf(g_ds_w, (float)j) * g_ds_frac * (nc ? s / nc : 0);
    }
    d->pg_loss = 0;
    if (g_pg_lambda > 0 && g_head_slab) { nn_d2h(d->pg_h, d->hs_ds + 56, 2 * sizeof(double)); if (d->hs_train) nn_d2h(d->pg_h, d->hs_ds + 58, sizeof(double));   /* (training: the loss sum of the gradient pass) */
        d->pg_loss = d->pg_h[1] > 0 ? g_pg_lambda * d->pg_h[0] / d->pg_h[1] : 0; }
    d->count_loss = 0;
    if (g_count_lambda > 0) { nn_d2h(d->cnt_h, d->cnt_out, 3 * sizeof(float)); d->count_loss = g_count_lambda * d->cnt_h[0]; }
    d->cond_loss = 0;
    if (g_cond) {   /* mean bce over the 4 winding channels, times lambda */
        nn_loss_fetch(d->cscratch, (shape5){1, 4, 1, 1, 1}, d->cout4);
        d->cond_loss = g_cond_lambda * (d->cout4[0] + d->cout4[1] + d->cout4[2] + d->cout4[3]) / 4;
        nn_d2h(d->cacc_h, d->cacc, 2 * sizeof(unsigned long long));
    }
    d->pap_loss = 0;
    if (g_pap_head) {
        nn_loss_fetch(d->pscr, (shape5){1, 1, 1, 1, 1}, d->pout3); d->pap_loss = g_pap_lambda * d->pout3[0];
    }
    d->aff_loss = 0;
    if (g_aff) {   /* mean over the affinity channels of bce + dice_w dice, times lambda */
        nn_aff_loss_fetch(d->aff_fin, g_aff_off.K, d->aff_out);
        for (int k = 0; k < g_aff_off.K; k++) d->aff_loss += d->aff_out[k * 6] + dice_w * d->aff_out[k * 6 + 1];
        d->aff_loss *= g_aff_lambda / g_aff_off.K;
    }
    d->side_loss = 0;
    if (g_side) {   /* sout[0] := mean bce over the side channels */
        nn_loss_fetch(d->sscratch, (shape5){1, g_side_nc, 1, 1, 1}, d->sout);
        if (ufsm_env_on("UFSM_SIDE_CHECK")) fprintf(stderr, "side check (GPU %d): bce %g %g dice %g %g active %g\n", d->dev, d->sout[0], d->sout[1], d->sout[2], d->sout[3], d->sout[4]);
        if (g_side_nc == 2) d->sout[0] = 0.5f * (d->sout[0] + d->sout[1]);
        d->side_loss = g_side_lambda * d->sout[0];
    }
    d->verso_loss = 0;
    if (g_verso_on) {   /* the recto loss ran as a single-channel call: lay its parts out as the NCH-channel fetch does */
        float r[3]; nn_loss_fetch(d->scratch, (shape5){1, 1, 1, 1, 1}, r);
        for (int c = 0; c < 2 * NCH + 1; c++) out[c] = 0;
        out[0] = r[0]; out[NCH] = r[1]; out[2 * NCH] = r[2];
        nn_loss_fetch(d->vscratch, (shape5){1, 1, 1, 1, 1}, d->vout); d->verso_loss = g_verso_lambda * (d->vout[0] + dice_w * d->vout[1]);
    }
    int cout = g_half ? g_cout : unet_cfg_of(d->u)->cout;
    double loss = 0; int active = 0;
    if (g_aff || g_unwrap) cout = NCH;
    for (int c = 0; c < cout; c++) if (out[c] > 0 || out[cout + c] > 0) { loss += out[c] + dice_w * out[cout + c]; active++; }
    if (g_heads) {   /* each head's loss for the validation table; the logs see the mean over the active recto heads in the NCH layout */
        float rb = 0, rd = 0; int na = 0;
        for (int c = 0; c < cout; c++) {
            if (c == CH_SHEET) continue;
            const int h = c ? c - 1 : 0;
            d->hon[h] = out[c] > 0 || out[cout + c] > 0; d->hl[h] = out[c] + dice_w * out[cout + c];
            if (d->hon[h]) { rb += out[c]; rd += out[cout + c]; na++; }
        }
        const float sb = out[CH_SHEET], sd = out[cout + CH_SHEET], act = out[2 * cout];
        out[CH_RECTO] = na ? rb / na : 0; out[CH_SHEET] = sb; out[NCH + CH_RECTO] = na ? rd / na : 0; out[NCH + CH_SHEET] = sd; out[2 * NCH] = act;
    }
    return (active ? loss / active : 0)+d->sheet_loss+d->aff_loss+d->verso_loss+d->side_loss+d->ink_loss+d->norm_loss+d->cond_loss+d->pap_loss+d->count_loss+d->pg_loss+d->ds_loss+d->w4_loss+d->emb_loss;
}

/* --split z: forward, loss and backward of one window on both GPUs (src/split.h) */
typedef struct { gpu_state *G; int B, P, train; float dice_w; } split_arg;
static void split_job(int side, void *a) {
    split_arg *s = a;
    gpu_state *d = &s->G[side];
    run_batch(d, s->B, s->P, s->dice_w, s->train);
    if (s->train) unet_backward_x(d->u, g_half && !g_pkf ? d->glh : d->gl, g_g16);
}

int cmd_train(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: ufsm train <sources.json> --out DIR [--P 512] [--B 1] [--steps 20000] [--lr 1e-3] [--warmup 500] [--wd 0.01]\n"
                        "       [--dice 0.5] [--ema 0.999] [--widths 16,32,64,80] [--down-norm 1] [--gpus 0,1] [--workers 12 (new runs with --gpu-aug 2: 3)] [--gpu-aug 0|1|2 (new runs: 2)] [--sampler-slots N] [--seed 0] [--det 1] [--resume CKPT] [--finetune 1] [--fp4 0|1|2 (default 2)] [--mem auto|auto16|default|wide] [--opt adamw|muon|anvil] [--muon-lr 0.02] [--muon-beta 0.95] [--anvil-lr 0.023] [--anvil-wd 2.25] [--sched cos|wsd] [--cooldown 0.2]\n"
                        "       [--val-batches 8] [--val-P N (validation windows of N^3 whatever P: the same voxels across window sizes)] [--log-every 20] [--val-every 500] [--ckpt-every 1000] [--clip 5] [--levels 0.5,0.25,0.15,0.1] [--fp32] [--f16 1] [--gscale 1024] [--prec 1|2|3|4] [--input-prec 0|4|8] [--policy enc0=1,dec0.c1=fp16:fp16:fp8,...] [--qat 2|3] [--wq 8|4] [--sparse24 STEP] [--srste 2e-4] [--pos-weight 1] [--tol 0..2] [--sr 1]\n"
                        "       [--seconds S] [--warmup-seconds S (default 5%% of time budget)] (time-based schedule and final checkpoint)\n"
                        "       [--schedule-seconds S --schedule-elapsed S]   wall-time LR horizon across finite-cover evaluation pauses\n"
                        "       [--cover PLAN.json] [--cover-extend-from OLD_PLAN.json] [--cover-next-pass 1] [--stop-at STEP] [--ct-aug 0|1] [--symmetry-p 1] [--axis-jitter 0] [--limit-seconds 0]\n"
                        "       [--lr-floor 0 (the schedule ends at this fraction of the peak; modded-nanogpt 0.3)] [--wd-nonconv 0 (decay multiplier for GroupNorm, biases, head under Muon / ANVIL)]\n"
                        "       [--muon-c 1 (Muon-C: conv kernels polarised per Fourier frequency)] [--hyperball 1 (fixed weight / update norms)] [--hb-lr-mult 0.707 (Hyperball lr / Muon lr)]\n"
                        "       [--trunk-lr-mult 1 (lr x F outside the attention stage)] [--attn-dcoarse 0|1 (deformable: + the 2x pooled level)] [--attn-deform-init grid|normal]\n"
                        "       [--recall W (recto: + W (1 - soft recall), gradient only)]\n"
                        "       [--dec-convs 2 (convs per decoder level, one value or a list; 1 = one conv, nnU-Net style)]\n"
                        "       [--deep-sup K (1^3 heads on decoder levels 1..K)] [--ds-weight 0.5 (level j: w^j)] [--ds-anneal 1 (weights -> 0 over the run)]\n"
                        "       [--distill C (1^3 head predicting the sources' \"emb\" feature vectors, cosine loss)] [--distill-level 3] [--distill-weight 1]\n"
                        "       [--iso T (void cutoff of sources without \"iso\")] [--iso-void 0|1] [--iso-pap 2] [--iso-lift 0 (voxels)]   void (CT < cutoff, > iso-pap voxels\n"
                        "                                     from papyrus) is background beyond the trust band; labels with no papyrus within iso-lift voxels ignored\n"
                        "       [--shell R (level-0 voxels: background beside a label ignored)] [--dist-head SIGMA (channel 1 = distance ridge of channel 0)]\n"
                        "       [--geometry-aug 0|1] [--rotate-deg 5] [--rotate-p 0.2] [--elastic 1 (amplitude in voxels, 0..10; capped at 0.09 P / 2 pi: invertible)] [--elastic-p 0.15] [--label-morph 0] [--label-morph-p 0.2]\n"
                        "       [--aug-strength 1 (0..2)] [--aug-p 0 (default p of every scanner op)] [--aug-OP-p P] [--aug-drop-ignore 1] [--aug-affine-p 0] [--aug-aniso 0.1] [--aug-shear 0.06]\n"
                        "       [--aug-crush 0 (p: crushed windows of stored-phase sources, src/crush.h)] [--aug-crush-void 0.15,0.5] [--aug-crush-squash 0.5] [--aug-crush-k 1.5,3]\n"
                        "       [--aug-crush-bend 0.5] [--aug-crush-bend-amp 0.06] [--aug-crush-kink 0.25] [--aug-crush-rot 10] [--aug-crush-fine 0 (CT N levels finer, sources \"crush_fine\")] [--aug-crush-warp 0 (1: crushed windows also take the rotation / elastic / affine / zoom warp)]\n"
                        "       [--aug-xscan 0 (p: the CT from a registered partner scan, sources \"xscan\": [{\"map\": REG.json}])]\n"
                        "       [--aug-tear 0 (p: crush-type windows cut across the sheets, one side shifted up to ~1+ turns, src/scrollaug.h)] [--aug-tear-um 480 (largest shift)]\n"
                        "       [--aug-delam 0 (p: crush-type windows with 1-3 peels (thin voids inside sheets) / burned-out holes, labels keep the winding)] [--aug-delam-hole 0.5 (per site)]\n"
                        "       [--aug-curv 0 (p: crush-type windows re-bent about a moved axis: tighter inner core / flatter outer wraps, labels pulled)]\n"
                        "       [--aug-inclusions 0 (p: bright mineral grains, bubbles in papyrus and debris in the gaps pasted into the CT, labels unchanged)]\n"
                        "       [--aug-sheet-density 0 (p: stored-phase windows re-lit per wrap from the phase + a slow carbonisation field, CT only)]\n"
                        "         scanner ops (source grid, z = scan axis): seam air lowres blur sharpen bias ring cupping slice noise cnoise tone quant dropslice cutout chunk\n"
                        "       [--sheet-band-scale 1]   widen audited soft-band cap and close-gap exclusions together\n"
                        "       [--aux-input 1]   a fifth input channel from the sources' \"aux\" volumes (e.g. the papyrus void fraction)\n"
                        "       [--erode 0|1] [--soft SIGMA] [--soft-unit vox|um]   native binary-core erosion before softening (--levels 1,0,0,0); sigma in level-0 voxels or um\n"
                        "       [--gn-stats stored|legacy]   fresh training uses stored activations; resume preserves the saved contract\n"
                        "       [--schedule-start STEP]   restart the LR schedule at this saved step, preserving optimizer state\n"
                        "       [--half-res none|pool|pack|pool4|pack4]   the network on 2^3 (4^3) blocks of the window: pool = block means in, logits\n"
                        "                                     upsampled; pack = means + the CT's sub-voxel gradients / contrast in, 8 sub-voxel logits out\n"
                        "                                     (pack4: of the 2x-pooled window, logits upsampled 2x after); task surface\n"
                        "       [--task surface_winding --geometry geometry.json] [--sheet-init 1] [--sheet-variant 0|1|2] [--task band_affinity --grow-head 1 --aff-lambda 1] [--init-from CKPT (shape-matching layers)]\n"
                        "       [--task papyrus [--pap-cut 50,128] [--pap-rel 0.6,1.6 --pap-rel-p 0.5] [--pap-sigma 1] [--pap-tau LO,HI] [--pap-band 0] [--pap-clean 0] [--pap-cond 1|2|0]]\n"
                        "                                     papyrus vs void labels from the clean CT at a random cutoff per patch (src/papyrus.h); --pap-cond 1|2:\n"
                        "                                     the net is conditioned on the cutoff (1 stem bias, 2 the CT input's zero; predict --papyrus-cut T)\n"
                        "       [--warm-start 1]   start a new cover from a legacy checkpoint's weights/state\n"
                        "  B is the per-GPU batch; gradients are averaged across GPUs every step (effective batch B x ngpus).\n"
                        "  --split z --gpus 0,1: every window is split along z across the two GPUs instead (batch B; P a multiple of 2^nlev).\n"
                        "  --Pz N: slab windows N x P x P (N <= P, a multiple of 2^(nlev-1)): pyramid targets at levels >= 1 with a stored phase, --zfix 1\n"
                        "  env UFSM_PROF=1 prints per-op GPU time every log interval (categories 'loss' and 'optimizer').\n");
        return 2;
    }
    const char *src = argv[2], *out = opt(argc, argv, "--out", "runs/run");
    int P = atoi(opt(argc, argv, "--P", "512")), B = atoi(opt(argc, argv, "--B", "1")), steps = atoi(opt(argc, argv, "--steps", "20000"));   /* 512^3 batch 1 per 16 GB GPU is the standard window (user, 2026-10-01) */
    float lr0 = (float)atof(opt(argc, argv, "--lr", "1e-3")), wd = (float)atof(opt(argc, argv, "--wd", "0.01")), dice_w = (float)atof(opt(argc, argv, "--dice", "0.5"));
    float ema = (float)atof(opt(argc, argv, "--ema", "0.999")), clip = (float)atof(opt(argc, argv, "--clip", "5"));
    int warmup = atoi(opt(argc, argv, "--warmup", "500")), workers = atoi(opt(argc, argv, "--workers", "12"));
    const int workers_asked = opt(argc, argv, "--workers", nullptr) != nullptr;
    double seconds = atof(opt(argc, argv, "--seconds", "0"));
    double schedule_seconds=atof(opt(argc,argv,"--schedule-seconds","0"));
    double schedule_elapsed=atof(opt(argc,argv,"--schedule-elapsed","0"));
    if (!isfinite(schedule_seconds) || schedule_seconds<0 || !isfinite(schedule_elapsed) || schedule_elapsed<0 ||
        (schedule_seconds && (seconds || schedule_elapsed>=schedule_seconds)) || (!schedule_seconds && schedule_elapsed)) {
        fprintf(stderr,"invalid wall-time schedule horizon/elapsed or conflicting --seconds\n"); return 2;
    }
    double time_horizon=schedule_seconds?schedule_seconds:seconds;
    double limit_seconds = atof(opt(argc, argv, "--limit-seconds", "0"));
    if (!isfinite(limit_seconds) || limit_seconds < 0) { fprintf(stderr, "invalid limit-seconds\n"); return 2; }
    int stop_at = atoi(opt(argc, argv, "--stop-at", "0"));
    if (stop_at < 0) { fprintf(stderr, "stop-at must be nonnegative\n"); return 2; }
    double warmup_seconds = atof(opt(argc, argv, "--warmup-seconds", "-1"));
    if (!isfinite(seconds) || seconds < 0 || !isfinite(warmup_seconds) || warmup_seconds < -1) { fprintf(stderr, "invalid time budget\n"); return 2; }
    if (warmup_seconds < 0) warmup_seconds = time_horizon * 0.05;
    if (time_horizon && warmup_seconds >= time_horizon) { fprintf(stderr, "warmup-seconds must be less than time horizon\n"); return 2; }
    int nval = atoi(opt(argc, argv, "--val-batches", "8")), log_every = atoi(opt(argc, argv, "--log-every", "20"));
    int val_every = atoi(opt(argc, argv, "--val-every", "500")), ckpt_every = atoi(opt(argc, argv, "--ckpt-every", "1000"));
    const int val_live = atoi(opt(argc, argv, "--val-live", "0"));   /* validate the live weights instead of the EMA (e.g. resume a last.ckpt for one lr-0 step) */
    g_valP = atoi(opt(argc, argv, "--val-P", "0"));   /* (checked once the window and the task are known) */
    const int l0_recal = atoi(opt(argc, argv, "--l0-recal", "4"));   /* --l0-norm ema: windows recalibrating the EMA weights' level-0 statistics before each validation */
    uint64_t seed = (uint64_t)atoll(opt(argc, argv, "--seed", "0"));
    const char *resume = opt(argc, argv, "--resume", nullptr);
    /* run defaults (2026-10-29): a NEW run (no --resume) defaults to --gpu-aug 2 and 3 sampler workers (with the labels' warp on the GPU
       the sampler keeps a P512 trainer fed with 2-3 workers; CPU augmentation needed ~10 at ~3.2 GB host each) and records that in
       <out>/run-defaults.txt; a resumed run keeps the old defaults (--gpu-aug 0, 12 workers: its augmentation stream unchanged) unless
       its run directory holds that record. Explicit --gpu-aug / --workers always win. */
    int newdef = !resume;
    if (resume) { char rp[1400]; snprintf(rp, sizeof rp, "%s/run-defaults.txt", out); FILE *rf = fopen(rp, "r"); if (rf) { char ln[128] = {0}; if (fgets(ln, sizeof ln, rf) && strstr(ln, "gpu_aug_default 2")) newdef = 1; fclose(rf); } }
    const char *geometry=opt(argc,argv,"--geometry",nullptr),*task=opt(argc,argv,"--task",geometry?"surface_winding":"surface");
    int sheet_init=atoi(opt(argc,argv,"--sheet-init","0")); char old_geometry[65],old_reference[65];
    int saved_variant=2,saved_sheet_start=0;
    int sheet_ck=resume?sheet_checkpoint(resume,old_geometry,old_reference):0;
    g_aff = !strcmp(task,"band_affinity");
    g_unwrap2 = !strcmp(task,"unwrap2");   /* the spiral fit's heads: phase, normal / density, winding mod 4 (no recto, ink) */
    g_unwrap = !strcmp(task,"unwrap") || g_unwrap2;
    g_restore = !strcmp(task, "restore");
    if (g_restore) {
        if (restore_args(argc, argv)) return 2;
        g_cout = g_rpairs && g_rpw.unc > 0 ? 2 : 1; g_rP = P;   /* (--sr-unc: channel 1 the log scale) */ g_rseed = (uint64_t)atoll(opt(argc, argv, "--seed", "0")) * 0x2545F4914F6CDD1Dull + 0x7E57;
        const char *bad = B != 1 ? "--B 1" : geometry ? "no geometry" : !strcmp(opt(argc, argv, "--split", "0"), "z") ? "no --split" : strcmp(opt(argc, argv, "--half-res", "none"), "none") ? "no --half-res" :
                          atoi(opt(argc, argv, "--overfit", "0")) ? "no --overfit" : *opt(argc, argv, "--cover", "") ? "no --cover" : atoi(opt(argc, argv, "--verso", "0")) || atoi(opt(argc, argv, "--side", "0")) || atoi(opt(argc, argv, "--cond", "0")) ||
                          atoi(opt(argc, argv, "--pap-head", "0")) || atoi(opt(argc, argv, "--heads", "0")) > 1 || *opt(argc, argv, "--deep-sup", "") || atoi(opt(argc, argv, "--aux-input", "0")) ? "no verso / side / cond / pap-head / heads / deep-sup / aux" :
                          !strcmp(opt(argc, argv, "--mem", "auto"), "wide") ? "no --mem wide" : nullptr;
        if (bad) { fprintf(stderr, "--task restore: %s\n", bad); return 2; }
    }
    if (g_unwrap) {   /* recto, ink (task unwrap), then phase (2) and normal / density (7) below */
        if (B != 1 || geometry || atoi(opt(argc,argv,"--overfit","0")) || strcmp(opt(argc,argv,"--mem","auto"),"wide")==0) { fprintf(stderr,"unwrap: B=1, no geometry / overfit / wide\n"); return 2; }
        g_cout = g_unwrap2 ? 0 : 2;
        if (g_unwrap2) {
            g_w4_lambda = (float)atof(opt(argc,argv,"--wind4-lambda","1"));
            if (!(g_w4_lambda >= 0) || opt(argc,argv,"--ink-lambda",nullptr) || opt(argc,argv,"--ink-dice",nullptr) || atoi(opt(argc,argv,"--cond","0")) || atoi(opt(argc,argv,"--pap-head","0"))) {
                fprintf(stderr,"--task unwrap2: no ink head, --cond or --pap-head; --wind4-lambda >= 0\n"); return 2;
            }
        }
        g_ink_lambda = (float)atof(opt(argc,argv,"--ink-lambda","1")); g_ink_dice = (float)atof(opt(argc,argv,"--ink-dice","0"));
        g_norm_lambda = (float)atof(opt(argc,argv,"--normal-lambda","1")); g_norm_s = atoi(opt(argc,argv,"--normal-step","4"));
        if (g_norm_s < 1 || g_norm_s > 16) { fprintf(stderr,"--normal-step 1..16\n"); return 2; }
        g_dens_scale = (float)atof(opt(argc,argv,"--density-scale","32"));   /* density target min(1, scale rho): 32 at 2.4 um, 8 at 9.6 um (the same physical code) */
        if (!(g_dens_scale > 0)) { fprintf(stderr,"--density-scale > 0\n"); return 2; }
        /* --density-code 1: density target x / (1 + x), x = scale rho (never saturates; 0: min(1, x), which clips every spacing under
           scale x voxel size, 77 um at the default codes); --packed-weight W (> 1): the phase and normal / density losses weigh a voxel
           1 at a sheet spacing of --packed-um (120) or more, rising to W at half of it (the nets under-count packed sheets) */
        g_dens_code = atoi(opt(argc,argv,"--density-code","0")); g_packed_w = (float)atof(opt(argc,argv,"--packed-weight","0")); g_packed_um = (float)atof(opt(argc,argv,"--packed-um","120"));
        g_ptrunc_tau = (float)atof(opt(argc,argv,"--phase-trunc","0")); g_ptrunc_kappa = (float)atof(opt(argc,argv,"--phase-trunc-conf","0.5")); g_ptrunc_from = atoi(opt(argc,argv,"--phase-trunc-from","5000"));
        if (!(g_ptrunc_tau >= 0 && g_ptrunc_tau < 0.5f) || !(g_ptrunc_kappa >= 0 && g_ptrunc_kappa <= 1)) { fprintf(stderr,"--phase-trunc 0..0.5 turns, --phase-trunc-conf 0..1\n"); return 2; }
        if (g_ptrunc_tau > 0) fprintf(stderr, "robust phase loss: confident (|phasor| >= %g) predictions more than %g turn off their label drop out of the phase mask from step %d\n", g_ptrunc_kappa, g_ptrunc_tau, g_ptrunc_from);
        if (g_dens_code < 0 || g_dens_code > 1 || g_packed_w < 0 || g_packed_w > 255 || !(g_packed_um > 0)) { fprintf(stderr,"--density-code 0|1, --packed-weight 0..255, --packed-um > 0\n"); return 2; }
        {   /* --density-packed 0: the density plane's loss without the packed weight (mask 0 / 1; the normal planes keep it). The weight pulled
               voxels the net cannot classify (gaps beside packed sheets) toward the dense target: integrated density over-counted gaps */
            const int dp = atoi(opt(argc,argv,"--density-packed","1"));
            if (dp < 0 || dp > 1) { fprintf(stderr,"--density-packed 0|1\n"); return 2; }
            if (!dp) { nn_set_loss_unit_ch7(6); fprintf(stderr, "density loss: packed weight off (mask 0 / 1; normal planes keep --packed-weight %g)\n", g_packed_w); }
        }
        /* --pgrad-lambda L (> 0): the phase wavenumber loss (nn_pgrad_slab): the predicted phase's rotation over --pgrad-dist D voxels (2)
           along the true sheet normal against the label's, angular, weighted by the phase mask (packed sheets x --packed-weight);
           punishes a phase that skips sheets, which the per-voxel BCE lets blur. Head-slab path only (task unwrap2, B 1) */
        g_pg_lambda = (float)atof(opt(argc,argv,"--pgrad-lambda","0")); g_pg_angle = atoi(opt(argc,argv,"--pgrad-angle","0")) > 0; g_pg_mask = atoi(opt(argc,argv,"--pgrad-mask","0")) > 0;   /* --pgrad-mask 1: reliable label pairs only (nn_loss.cu pg_dists) */   /* --pgrad-angle 1: the angular gradient only (pgrad otherwise raises every phasor magnitude where the rotation is right: 10-25 pg1 / s1p median confidence 0.2 -> 0.55, gross errors at conf > 0.6 2-7% -> 21-41%) */
        { const char *pd = opt(argc,argv,"--pgrad-dist","2"); g_pg_nd = 0; int bad = 0;   /* several distances: each voxel pairs with a partner at each */
          while (*pd && g_pg_nd < 4) { char *e; g_pg_dist[g_pg_nd] = strtof(pd, &e); if (e == pd) { bad = 1; break; } bad |= !(g_pg_dist[g_pg_nd] >= 1 && g_pg_dist[g_pg_nd] <= 16); g_pg_nd++; pd = *e == ',' ? e + 1 : e; if (*e && *e != ',') { bad = 1; break; } }
          if (*pd || !g_pg_nd) bad = 1;
          if (!(g_pg_lambda >= 0) || bad) { fprintf(stderr,"--pgrad-lambda >= 0, --pgrad-dist D[,D...] (1..16, up to 4)\n"); return 2; } }
        /* --count-lambda L (> 0): the count loss: per step --count-segments K segments of --count-length voxels along the true sheet
           normal, |log| of the predicted density integral over the true turns (nn_count_loss); needs --density-code 1 */
        g_count_lambda = (float)atof(opt(argc,argv,"--count-lambda","0")); g_count_K = atoi(opt(argc,argv,"--count-segments","1024")); g_count_L = atoi(opt(argc,argv,"--count-length","64"));
        if (g_count_lambda < 0 || (g_count_lambda > 0 && (g_dens_code != 1 || g_count_K < 1 || g_count_K > 65536 || g_count_L < 4 || g_count_L > 1024 || !strcmp(opt(argc,argv,"--split","0"),"z")))) {
            fprintf(stderr,"--count-lambda: needs --density-code 1, --count-segments 1..65536, --count-length 4..1024, no --split\n"); return 2;
        }
    }
    if (g_aff) {   /* affinities at offsets 1 and 8 along z, y, x (8 = the split halo at four levels) */
        g_aff_off = (aff_offsets_t){6, {0, 1, 2, 0, 1, 2}, {1, 1, 1, 8, 8, 8}};
        g_cout = 1 + g_aff_off.K; g_aff_lambda = (float)atof(opt(argc,argv,"--aff-lambda","1"));
        if (B != 1 || geometry || atoi(opt(argc,argv,"--overfit","0")) || strcmp(opt(argc,argv,"--mem","auto"),"wide")==0) { fprintf(stderr,"band_affinity: B=1, no geometry / overfit / wide\n"); return 2; }
    }
    g_verso = atoi(opt(argc,argv,"--verso","0")); g_verso_start = atoi(opt(argc,argv,"--verso-start","0"));
    { const char *vt = opt(argc,argv,"--verso-teacher","frozen"); if (strcmp(vt,"frozen") && strcmp(vt,"self")) { fprintf(stderr,"--verso-teacher frozen|self\n"); return 2; } g_verso_self = !strcmp(vt,"self"); }
    g_verso_hard = (float)atof(opt(argc,argv,"--verso-hard","0")); g_verso_zero = !strcmp(opt(argc,argv,"--verso-init","copy"),"zero");
    g_verso_ridge = atoi(opt(argc,argv,"--verso-ridge","0")); g_verso_anchor = atoi(opt(argc,argv,"--verso-anchor","0"));
    g_verso_swap = (float)atof(opt(argc,argv,"--verso-swap","0")); g_verso_lambda = (float)atof(opt(argc,argv,"--verso-lambda","1")); g_head_only = atoi(opt(argc,argv,"--train-head-only","0"));
    if (g_verso) fprintf(stderr, "verso: channel 1, teacher %s, targets %s (hard %g, ridge %d), head %s, from step %d\n", g_verso_self ? "live" : "frozen",
                         g_verso_ridge ? "ridge" : g_verso_hard > 0 ? "hard" : "soft", g_verso_hard, g_verso_ridge, g_verso_zero ? "fresh" : "copy of recto", g_verso_start);
    if (g_verso) {
        if (B != 1 || geometry || strcmp(task,"surface_winding")==0 || g_unwrap) { fprintf(stderr,"--verso: B=1, not with the winding or unwrap tasks\n"); return 2; }
        if (g_aff) g_cout = 2 + g_aff_off.K;   /* recto, verso, affinities */
    }
    g_side = atoi(opt(argc,argv,"--side","0")); g_side_lambda = (float)atof(opt(argc,argv,"--side-lambda","1"));
    if (g_unwrap) { if (g_side && g_side != 2) { fprintf(stderr,"unwrap: the phase head is --side 2\n"); return 2; } g_side = 2; }
    if (g_side) {
        if (!g_aff && !g_unwrap) { fprintf(stderr,"--side needs --task band_affinity (side labels come from the band sources)\n"); return 2; }
        if (g_side != 1 && g_side != 2) { fprintf(stderr,"--side 1 (recto / verso side) or 2 (winding phase)\n"); return 2; }
        g_side_nc = g_side == 2 ? 2 : 1; g_side_ch = g_cout; g_cout += g_side_nc;   /* the last output channel(s) */
        fprintf(stderr, "%s: channel%s %d%s, lambda %g, air <= %s\n", g_side == 2 ? "phase" : "side", g_side == 2 ? "s" : "", g_side_ch,
                g_side == 2 ? " (cos), +1 (sin)" : "", g_side_lambda, opt(argc,argv,"--side-air","0"));
    }
    if (g_unwrap) {
        g_norm_ch = g_cout; g_cout += 7;
        if (g_unwrap2) { g_w4_ch = g_cout; g_cout += 2; }
        if (g_unwrap2) fprintf(stderr, "unwrap2: channels %d-%d phase, %d-%d normal n n^T, %d density (lambda %g, step %d), %d-%d winding mod 4 (gauge-free, lambda %g)\n",
                               g_side_ch, g_side_ch + 1, g_norm_ch, g_norm_ch + 5, g_norm_ch + 6, g_norm_lambda, g_norm_s, g_w4_ch, g_w4_ch + 1, g_w4_lambda);
        else fprintf(stderr, "unwrap: channels 0 recto, 1 ink (lambda %g, dice %g), %d-%d phase, %d-%d normal n n^T, %d density (lambda %g, step %d)\n",
                     g_ink_lambda, g_ink_dice, g_side_ch, g_side_ch + 1, g_norm_ch, g_norm_ch + 5, g_norm_ch + 6, g_norm_lambda, g_norm_s);
    }
    g_cond = atoi(opt(argc, argv, "--cond", "0")); g_cond_lambda = (float)atof(opt(argc, argv, "--cond-lambda", "1"));
    const char *closs = opt(argc, argv, "--cond-loss", "all");   /* all: the winding heads supervised everywhere known; outside: only where the condition does not reach */
    if (strcmp(closs, "all") && strcmp(closs, "outside")) { fprintf(stderr, "--cond-loss all|outside\n"); return 2; }
    g_cond_outside = !strcmp(closs, "outside");
    if (g_cond) {
        if (!g_unwrap || B != 1) { fprintf(stderr, "--cond: task unwrap, B 1\n"); return 2; }
        g_cond_ch = g_cout; g_cout += 4;
        fprintf(stderr, "cond: inputs 4-6 the winding mod 14 condition (cos, sin, mask), outputs %d-%d cos / sin of 2 pi q / 14 and 2 pi q / 2 (lambda %g)\n",
                g_cond_ch, g_cond_ch + 3, g_cond_lambda);
    }
    g_pap_head = atoi(opt(argc, argv, "--pap-head", "0"));
    if (g_pap_head) {   /* the gap-label settings validated on block B (DESIGN.md "Papyrus gap labels") */
        if (!g_unwrap || !strcmp(opt(argc, argv, "--split", "0"), "z")) { fprintf(stderr, "--pap-head: task unwrap, whole windows (no --split)\n"); return 2; }
        papyrus_cfg *c = &g_paph; memset(c, 0, sizeof *c);
        int bad = sscanf(opt(argc, argv, "--pap-cut", "55,128"), "%f,%f", &c->t_lo, &c->t_hi) != 2;
        c->sigma = (float)atof(opt(argc, argv, "--pap-sigma", "1")); c->band = (float)atof(opt(argc, argv, "--pap-band", "5"));
        c->local = atoi(opt(argc, argv, "--pap-local", "8")); c->local_sigma = (float)atof(opt(argc, argv, "--pap-local-sigma", "1.5"));
        c->local_depth = (float)atof(opt(argc, argv, "--pap-local-depth", "30")); c->local_margin = (float)atof(opt(argc, argv, "--pap-local-margin", "0.1"));
        g_pap_lambda = (float)atof(opt(argc, argv, "--pap-lambda", "1"));
        bad |= !(c->t_lo >= 0 && c->t_lo <= c->t_hi && c->t_hi <= 255) || !(c->sigma >= 0 && c->sigma <= PAPYRUS_RMAX / 3.f) || !(c->band >= 0) || c->local < 1 ||
               !(c->local_sigma >= 0 && c->local_sigma <= PAPYRUS_RMAX / 3.f) || !(c->local_depth > 0) || !(c->local_margin >= 0 && c->local_margin < 0.5f);
        if (bad) { fprintf(stderr, "--pap-head 1 [--pap-cut 55,128] [--pap-sigma 1] [--pap-band 5] [--pap-local 8] [--pap-local-sigma 1.5] [--pap-local-depth 30] [--pap-local-margin 0.1] [--pap-lambda 1]\n"); return 2; }
        g_pap_ch = g_cout; g_cout += 1;
        fprintf(stderr, "papyrus head: channel %d, gap labels of the clean CT (void <= %g or in the lower half of a gap >= %g deep, radius %d; band %g, margin %g), lambda %g\n",
                g_pap_ch, (double)c->t_lo, (double)c->local_depth, c->local, (double)c->band, (double)c->local_margin, (double)g_pap_lambda);
    }
    g_papyrus = !strcmp(task, "papyrus");
    if (g_papyrus && (geometry || g_verso || g_side || g_unwrap)) { fprintf(stderr, "--task papyrus: no geometry, verso, side or unwrap\n"); return 2; }
    if (g_papyrus && papyrus_args(argc, argv, B)) return 2;
    if (sheet_ck<0 || (strcmp(task,"surface") && strcmp(task,"surface_winding") && !g_aff && !g_unwrap && !g_papyrus && !g_restore) || (!strcmp(task,"surface_winding") != (geometry!=nullptr)) ||
        (sheet_ck && (!geometry || sheet_init)) || (geometry && resume && !sheet_ck && !sheet_init) || (sheet_init && (!geometry || !resume))) {
        fprintf(stderr,"task mismatch: surface_winding needs --geometry; legacy warm-start needs --sheet-init 1; winding resumes preserve their geometry\n"); return 2;
    }
    if (geometry) {
        g_sheet=sheet_load(geometry); if (!g_sheet) return 2;
        if (sheet_ck && (strcmp(old_geometry,g_sheet->manifest_sha) || strcmp(old_reference,g_sheet->reference_sha))) { fprintf(stderr,"winding geometry/reference changed at resume\n"); return 2; }
        if (B!=1 || strcmp(opt(argc,argv,"--mem","auto"),"wide")==0 || atoi(opt(argc,argv,"--overfit","0")) || getenv("UFSM_SYNC_UPLOAD")) { fprintf(stderr,"winding task requires B=1, no wide/overfit/sync-upload\n"); return 2; }
        if (sheet_ck && sheet_checkpoint_options(resume,&saved_variant,&saved_sheet_start)) return 2;
        const char *variant=opt(argc,argv,"--sheet-variant",nullptr);
        g_sheet_variant=variant?atoi(variant):saved_variant;
        if (g_sheet_variant<0 || g_sheet_variant>2 || (sheet_ck && g_sheet_variant!=saved_variant)) {
            fprintf(stderr,"winding resume must preserve its loss variant\n"); return 2;
        }
        unet_set_input_mx(0); unet_set_input_prec(0); unet_set_grad_mx8(0);
    }
    checkpoint_runtime resumed_runtime; int has_runtime = resume ? checkpoint_runtime_read(resume, &resumed_runtime) : 0;
    if (has_runtime < 0) { fprintf(stderr, "cannot read checkpoint settings: %s\n", resume); return 1; }
    const char *gn_stats = opt(argc, argv, "--gn-stats", resume && (!has_runtime || !resumed_runtime.gn_stored) ? "legacy" : "stored");
    if (strcmp(gn_stats, "stored") && strcmp(gn_stats, "legacy")) { fprintf(stderr, "--gn-stats must be stored or legacy\n"); return 2; }
    nn_set_gn_stored(!strcmp(gn_stats, "stored"));
    int f16 = atoi(opt(argc, argv, "--f16", "1"));       /* 16-bit storage/operands as fp16 (8x finer than bf16, same speed); 0 = bf16 */
    nn_set_prec(atoi(opt(argc, argv, "--prec", "1")));   /* 1 bf16, 2 fp8, 3 fp4 fwd + fp8 wgrad (2/3 force bf16 storage) */
    if (has_runtime && resumed_runtime.input_prec) unet_set_input_prec(resumed_runtime.input_prec);
    if (*opt(argc, argv, "--input-prec", "") && unet_set_input_prec(atoi(opt(argc, argv, "--input-prec", "0")))) { fprintf(stderr, "--input-prec must be 0, 4 or 8\n"); return 2; }
    if (g_sheet) { unet_set_input_mx(0); unet_set_input_prec(0); }
    if (g_restore && !*opt(argc, argv, "--input-prec", "")) unet_set_input_prec(8);   /* restore: MX-fp8 stem input (the split input is exact there) */
    if (f16) { nn_set_f16(1); nn_set_grad_scale((float)atof(opt(argc, argv, "--gscale", "1024"))); }
    const char *optname = opt(argc, argv, "--opt", "adamw");   /* adamw | muon (3^3 conv weights: nesterov momentum + Newton-Schulz orthogonalisation; rest AdamW) */
    float muon_lr = (float)atof(opt(argc, argv, "--muon-lr", "0.02")), muon_beta = (float)atof(opt(argc, argv, "--muon-beta", "0.95"));
    int use_muon = !strcmp(optname, "muon"), use_anvil = !strcmp(optname, "anvil");
    /* Muon variants: --muon-c 1 (Muon-C, per-frequency polarisation of the conv kernels), --hyperball 1 (fixed weight and update
       norms; eta = --hb-lr-mult x the Muon lr, default 1/sqrt(2): Muon's first relative step; replaces the decay) */
    const int muon_c = atoi(opt(argc, argv, "--muon-c", "0")), hyperball = atoi(opt(argc, argv, "--hyperball", "0"));
    const float hb_mult = (float)atof(opt(argc, argv, "--hb-lr-mult", "0.70710678"));
    if ((muon_c || hyperball) && !use_muon) { fprintf(stderr, "--muon-c / --hyperball need --opt muon\n"); return 2; }
    if (muon_c < 0 || muon_c > 1 || hyperball < 0 || hyperball > 1 || !(hb_mult > 0.f)) { fprintf(stderr, "--muon-c / --hyperball: 0 or 1; --hb-lr-mult > 0\n"); return 2; }
    if (muon_c || hyperball) optname = muon_c && hyperball ? "muon-c-hb" : muon_c ? "muon-c" : "muon-hb";   /* the manifest / checkpoint label */
    /* --trunk-lr-mult F: every parameter outside the attention stage (the conv trunk, heads, norms; Muon and AdamW) at F x the scheduled
       rates, the stage at the full rates (unet_set_trunk_lr; default 1) */
    const float trunk_lr = (float)atof(opt(argc, argv, "--trunk-lr-mult", "1"));
    if (!(trunk_lr >= 0.f) || (trunk_lr != 1.f && use_anvil)) { fprintf(stderr, "--trunk-lr-mult: >= 0, not with --opt anvil\n"); return 2; }
    float anvil_lr = (float)atof(opt(argc, argv, "--anvil-lr", "0.023")), anvil_wd = (float)atof(opt(argc, argv, "--anvil-wd", "2.25"));
    const char *sched = opt(argc, argv, "--sched", "cos");   /* cos | wsd (warmup, constant, linear cooldown over the last --cooldown fraction; extendable runs) */
    int schedule_start = atoi(opt(argc, argv, "--schedule-start", "0"));
    if (sheet_ck) {
        if (*opt(argc,argv,"--schedule-start","") && schedule_start!=saved_sheet_start) {
            fprintf(stderr,"winding resume must preserve its ramp/schedule origin\n"); return 2;
        }
        schedule_start=saved_sheet_start;
    }
    if (schedule_start < 0) { fprintf(stderr, "schedule-start must be nonnegative\n"); return 2; }
    float cooldown = (float)atof(opt(argc, argv, "--cooldown", "0.2"));
    const float lr_floor = (float)atof(opt(argc, argv, "--lr-floor", "0"));   /* the schedule ends at this fraction of the peak lr (modded-nanogpt: 0.3) */
    if (!(lr_floor >= 0.f && lr_floor < 1.f)) { fprintf(stderr, "--lr-floor: 0 <= F < 1\n"); return 2; }
    int qat = atoi(opt(argc, argv, "--qat", "0"));
    /* Save the requested policy and the compute paths observed after dispatch, alongside storage modes. */
#define WRITE_MANIFEST(path) do { \
    char mf_[8192]; size_t mn_ = (size_t)nn_prec_manifest(mf_, sizeof mf_); \
    int add_ = snprintf(mf_ + mn_, sizeof mf_ - mn_, " act_mx4 %d act_mx8 %d grad_mx8 %d grad_mx4 %d input_prec %d gn_stored %d f16 %d opt %s\n", unet_act_mx4(), unet_act_mx() && !unet_act_mx4(), unet_grad_mx8(), unet_grad_mx4(), unet_input_prec(), nn_get_gn_stored(), nn_get_f16(), optname); \
    mn_ += add_ > 0 && (size_t)add_ < sizeof mf_ - mn_ ? (size_t)add_ : sizeof mf_ - mn_ - 1; \
    nn_exec_manifest(mf_ + mn_, sizeof mf_ - mn_); \
    FILE *mff_ = fopen(path, "w"); if (mff_) { fputs(mf_, mff_); fputc('\n', mff_); fclose(mff_); } \
} while (0)
    if (qat) { nn_set_prec(qat); nn_set_prec_wgrad(1); }
    if (atoi(opt(argc, argv, "--sr", "0"))) nn_set_sr(1);                                   /* stochastic rounding of fp8 gradient operands */
    int wq = atoi(opt(argc, argv, "--wq", "0"));            /* 8 or 4: true fp8 / fp4 weights (stochastic rounding after each update) */
    int sparse_at = atoi(opt(argc, argv, "--sparse24", "0"));   /* step from which the 3^3 conv weights are 2:4 sparse (0 = dense) */
    float srste = (float)atof(opt(argc, argv, "--srste", "2e-4"));
    nn_set_pos_weight((float)atof(opt(argc, argv, "--pos-weight", "1")));   /* BCE weight of surface voxels */
    int overfit = atoi(opt(argc, argv, "--overfit", "0"));   /* diagnostic: train on the first batch forever */
    g_bench_same = ufsm_env_on("UFSM_BENCH_SAME");   /* benchmark: every step on the first batch (step time without the loader or the upload) */
    int noaug = atoi(opt(argc, argv, "--noaug", "0"));       /* diagnostic: no augmentation */
    /* --fp4 1: packed fp4 activations, fp4 stride-1 compute, fp8 weight gradients and stochastic rounding.
       --fp4 2: fp4 stride-1 weight gradients as well. Small inputs and stride-2 operations have FP8 / FP32
       fallbacks; a requested 16-bit policy does not promote MX storage. The executed manifest records these.
       Explicit --policy changes the requested precisions; --fp4 0 uses 16-bit storage. */
    int expl = *opt(argc, argv, "--prec", "") || *opt(argc, argv, "--wq", "");
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--fp32")) expl = 1;
    const int fp8 = atoi(opt(argc, argv, "--fp8", "0"));   /* --fp8 1: MX-fp8 activations and fp8 compute everywhere but the fp16 stem (the fp4 A/B) */
    const int fp4 = fp8 ? 0 : atoi(opt(argc, argv, "--fp4", expl ? "0" : "2"));   /* --fp4 2 since 2026-10-01 night: passed the stair (last.ckpt 0.299 / 0.284 vs --fp4 1 0.307 / 0.284) and the 40k confirmation (r11c), 1-4% faster */
    { int nl = 1; for (const char *q = opt(argc, argv, "--widths", "16,32,64,80"); *q; q++) nl += *q == ','; nn_set_nlev(nl < UNET_MAXLEV ? nl : UNET_MAXLEV); }   /* layer names of the policy */
    if (fp4) {
        unet_set_act_mx4(1); setenv("UFSM_ACT_MX4", "1", 0); nn_set_sr(1);
        if (nn_set_prec_policy(fp4 >= 2 ? "all=fp4:fp4:fp4,enc0.c1=fp16" : "all=fp4:fp4:fp8,enc0.c1=fp16")) return 2;
        if (fp4 >= 2) setenv("UFSM_F4_WGRAD", "1", 0);
    }
    if (fp8) { unet_set_act_mx8(1); nn_set_sr(1); if (nn_set_prec_policy("all=fp8:fp8:fp8,enc0.c1=fp16")) return 2; }
    if (*opt(argc, argv, "--policy", "") && nn_set_prec_policy(opt(argc, argv, "--policy", ""))) return 2;
    { char mf[4096]; nn_prec_manifest(mf, sizeof mf); fprintf(stderr, "%s\n", mf); char mp[1400]; snprintf(mp, sizeof mp, "mkdir -p '%s'", out); if (system(mp)) {} if (!resume) { char rp[1400]; snprintf(rp, sizeof rp, "%s/run-defaults.txt", out); FILE *rf = fopen(rp, "w"); if (rf) { fprintf(rf, "gpu_aug_default 2 workers_default 3\n"); fclose(rf); } } snprintf(mp, sizeof mp, "%s/precision.txt", out); WRITE_MANIFEST(mp); }   /* per-layer: "enc0=1,enc1=2,..." */
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--fp32")) nn_set_tf32(0);
    int devs[8], ng = 0;
    { char *t = strdup(opt(argc, argv, "--gpus", opt(argc, argv, "--gpu", "0"))); for (char *q = strtok(t, ","); q && ng < 8; q = strtok(nullptr, ",")) devs[ng++] = atoi(q); free(t); }
    nn_init(devs[0]);   /* env precision knobs are read here; the storage modes decide the batch formats */
    g_xfmt = nn_get_tf32() && nn_get_act_bf16() && !ufsm_env_on("UFSM_ACT_MX8") ? (nn_get_f16() ? 1 : 2) : 0;
    g_g16 = nn_get_tf32() && nn_get_act_bf16() && nn_get_grad_bf16();
    if (ufsm_env_on("UFSM_X32")) g_xfmt = g_g16 = 0;   /* diagnostic: fp32 batches and logit gradient, converted on the device (the old path) */
    nn_set_loss_grad_h16(g_g16);
    if (g_restore && g_xfmt != 1) { fprintf(stderr, "--task restore: needs 16-bit (fp16) batches: tensor cores, --f16 1, 16-bit activation storage\n"); return 2; }
    if (g_restore) g_tc = g_rpairs ? 2 : 1;   /* the clean CT (real pairs: the target and the input crop) */
    g_heads = atoi(opt(argc, argv, "--heads", "0"));   /* recto heads, one per teacher: sources name theirs ("head": k); channel 0 is head 0, then sheet, then heads 1.. */
    if (g_heads == 1) g_heads = 0;
    if (g_heads) {
        if (g_heads < 0 || g_heads > TCMAX - 1 || g_aff || g_unwrap || g_verso || g_side || g_cond || g_pap_head || geometry || g_cout != NCH) {
            fprintf(stderr, "--heads 2..%d: task surface (no affinities, unwrap, verso, side, winding, papyrus head)\n", TCMAX - 1); return 2;
        }
        g_cout = g_tc = g_heads + 1;
        fprintf(stderr, "heads: %d recto heads (output channels 0, 2..%d; the sheet channel 1); each sample trains its source's head\n", g_heads, g_heads);
    }
    unet_cfg cfg = {4, {16, 32, 64, 80}, 4, g_cout, 8, 0, 0};
    cfg.down_norm = atoi(opt(argc, argv, "--down-norm", "0"));
    { const char *ln = opt(argc, argv, "--l0-norm", "gn");   /* level-0 normalisation (unet_cfg.l0_norm): gn, none (affine + SiLU), ema (running statistics), ws (none + weight standardisation), slab (GroupNorm per z-slab) */
      cfg.l0_norm = !strcmp(ln, "none") ? 1 : !strcmp(ln, "ema") ? 2 : !strcmp(ln, "ws") ? 3 : !strcmp(ln, "slab") ? 4 : 0;
      if (!cfg.l0_norm && strcmp(ln, "gn")) { fprintf(stderr, "--l0-norm gn|none|ema|ws|slab\n"); return 2; }
      cfg.l0_slab = atoi(opt(argc, argv, "--l0-slab", "64")); }   /* slab: planes per GroupNorm slab */
    cfg.scale_cond = atoi(opt(argc, argv, "--scale-cond", "0"));   /* voxel-size conditioning of the stem (unet_set_scale per batch) */
    if (g_papyrus) cfg.scale_cond = g_pap.cond == 1;   /* task papyrus: the stem-bias conditioning carries the cutoff instead */
    if (cfg.scale_cond && B != 1) { fprintf(stderr, "--scale-cond needs --B 1 (one voxel size per forward)\n"); return 2; }
    { const char *dw = opt(argc, argv, "--dec-widths", nullptr);   /* narrower coarse decoder levels (0 = encoder width), e.g. 0,32 */
      if (dw) { char *t = strdup(dw); int l = 0; for (char *q = strtok(t, ","); q && l < UNET_MAXLEV; q = strtok(nullptr, ",")) cfg.dec_widths[l++] = atoi(q); free(t); } }
    { const char *dc = opt(argc, argv, "--dec-convs", nullptr);   /* convs per decoder level: one value for every level, or a list (1 = one conv) */
      if (dc) { char *t = strdup(dc); int l = 0, last = 2; for (char *q = strtok(t, ","); q && l < UNET_MAXLEV; q = strtok(nullptr, ",")) cfg.dec_convs[l++] = last = atoi(q);
                if (l == 1) for (int k = 1; k < UNET_MAXLEV; k++) cfg.dec_convs[k] = last;
                free(t); } }
    { const char *eb = opt(argc, argv, "--enc-blocks", nullptr);   /* extra residual blocks per encoder level (levels >= 1), e.g. 0,0,2,3,3,3 */
      if (eb) { char *t = strdup(eb); int l = 0; for (char *q = strtok(t, ","); q && l < UNET_MAXLEV; q = strtok(nullptr, ",")) cfg.enc_blocks[l++] = atoi(q); free(t); } }
    {   /* the bottleneck transformer stage (unet_cfg.attn_*, src/attn.h) */
        cfg.attn_blocks = atoi(opt(argc, argv, "--attn-blocks", "0")); cfg.attn_reg = atoi(opt(argc, argv, "--attn-reg", "0")); cfg.attn_mlp = atoi(opt(argc, argv, "--attn-mlp", "0")); cfg.attn_vres = atoi(opt(argc, argv, "--attn-vres", "0"));
        const char *pe = opt(argc, argv, "--attn-pe", "peg");
        cfg.attn_pe = !strcmp(pe, "peg") ? 0 : !strcmp(pe, "rope") ? 1 : !strcmp(pe, "none") ? 2 : !strcmp(pe, "scroll") ? 3 : -1;
        if (cfg.attn_pe < 0) { fprintf(stderr, "--attn-pe peg|rope|none|scroll\n"); return 2; }
        const char *ax = opt(argc, argv, "--attn-axis", "full");   /* radial: attention along the rays of the scroll coordinates (unet_cfg.attn_axis) */
        cfg.attn_axis = !strcmp(ax, "full") ? 0 : !strcmp(ax, "radial") ? 1 : -1;
        if (cfg.attn_axis < 0) { fprintf(stderr, "--attn-axis full|radial\n"); return 2; }
        cfg.attn_repeat = atoi(opt(argc, argv, "--attn-repeat", "0"));   /* weight-tied repetitions of the blocks */
        cfg.attn_mem = atoi(opt(argc, argv, "--attn-mem", "0"));   /* window memory tokens (unet_cfg.attn_mem; trained on --chain-p window chains) */
        cfg.view_P = atoi(opt(argc, argv, "--view-P", "0")); cfg.view_patch = atoi(opt(argc, argv, "--view-patch", "8"));   /* scan fusion: view crops (sources "view", src/view.h) as view rows */
        if (!cfg.view_P) cfg.view_patch = 0;
        cfg.attn_deform = atoi(opt(argc, argv, "--attn-deform", "0"));   /* deformable attention: points per head and level (unet_cfg.attn_deform) */
        cfg.attn_dlev = atoi(opt(argc, argv, "--attn-dlev", "2"));        /* ... sampled levels: the bottleneck + the next finer ones */
        if (!cfg.attn_deform) cfg.attn_dlev = 0;
        cfg.attn_dco = atoi(opt(argc, argv, "--attn-dcoarse", "0"));     /* ... 1: also the 2x average-pooled bottleneck values (unet_cfg.attn_dco) */
        const char *di = opt(argc, argv, "--attn-deform-init", "grid");    /* ... offsets: grid (sphere prior) | normal (per-token sheet-normal frame) */
        cfg.attn_dinit = !strcmp(di, "grid") ? 0 : !strcmp(di, "normal") ? 1 : -1;
        if (cfg.attn_dinit < 0 || cfg.attn_dco < 0 || cfg.attn_dco > 1) { fprintf(stderr, "--attn-deform-init grid|normal, --attn-dcoarse 0|1\n"); return 2; }
        if (!cfg.attn_deform && (cfg.attn_dco || cfg.attn_dinit)) { fprintf(stderr, "--attn-dcoarse / --attn-deform-init: need --attn-deform\n"); return 2; }
    }
    cfg.w4_2d = atoi(opt(argc, argv, "--w4-2d", "0"));   /* consistent fp4 weights (unet_cfg.w4_2d): one scale per 32 x 32 block and tap */
    if (resume) { unet_cfg pc; int st; if (!unet_peek(resume, &pc, &st)) { cfg.w4_2d = pc.w4_2d; cfg.down_norm = pc.down_norm; cfg.l0_norm = pc.l0_norm; cfg.l0_slab = pc.l0_slab; cfg.scale_cond = pc.scale_cond; memcpy(cfg.dec_widths, pc.dec_widths, sizeof cfg.dec_widths); cfg.ds_levels = pc.ds_levels; memcpy(cfg.dec_convs, pc.dec_convs, sizeof cfg.dec_convs); memcpy(cfg.enc_blocks, pc.enc_blocks, sizeof cfg.enc_blocks);
                  cfg.attn_blocks = pc.attn_blocks; cfg.attn_pe = pc.attn_pe; cfg.attn_reg = pc.attn_reg; cfg.attn_mlp = pc.attn_mlp; cfg.attn_vres = pc.attn_vres; cfg.attn_axis = pc.attn_axis; cfg.attn_repeat = pc.attn_repeat; cfg.attn_mem = pc.attn_mem; cfg.view_P = pc.view_P; cfg.view_patch = pc.view_patch;
                  cfg.attn_deform = pc.attn_deform; cfg.attn_dlev = pc.attn_dlev; cfg.attn_dco = pc.attn_dco; cfg.attn_dinit = pc.attn_dinit; } }   /* the checkpoint decides */
    if (cfg.w4_2d) fprintf(stderr, "weights: consistent MX-fp4 (--w4-2d 1): one scale per 32 x 32 block and tap, the same values in the forward and backward-data\n");
    { char *t = strdup(opt(argc, argv, "--widths", "16,32,64,80")); cfg.nlev = 0; for (char *q = strtok(t, ","); q && cfg.nlev < UNET_MAXLEV; q = strtok(nullptr, ",")) cfg.widths[cfg.nlev++] = atoi(q); free(t); }
    if (P % (1 << (cfg.nlev - 1))) { fprintf(stderr, "P must be divisible by %d\n", 1 << (cfg.nlev - 1)); return 2; }
    g_Pz = atoi(opt(argc, argv, "--Pz", "0")); if (g_Pz <= 0) g_Pz = P;   /* --Pz: Pz x P x P slab windows (src/sample.h sample_cfg.Pz) */
    if (g_Pz != P) {
        if (g_Pz > P || g_Pz % (1 << (cfg.nlev - 1))) { fprintf(stderr, "--Pz: at most P and divisible by %d\n", 1 << (cfg.nlev - 1)); return 2; }
        if (!strcmp(opt(argc, argv, "--split", "0"), "z") || strcmp(opt(argc, argv, "--half-res", "none"), "none") || *opt(argc, argv, "--cover", "")) {
            fprintf(stderr, "--Pz: whole windows at full resolution (no --split, --half-res or --cover)\n"); return 2;
        }
    }
    const int split = !strcmp(opt(argc, argv, "--split", "0"), "z");
    {   /* --deep-sup K: 1^3 heads on decoder levels 1..K, loss weight w^j, annealed linearly to 0 over the run (--ds-anneal 0: constant) */
        const char *ds = opt(argc, argv, "--deep-sup", nullptr);
        g_ds = ds ? atoi(ds) : 0; g_ds_w = (float)atof(opt(argc, argv, "--ds-weight", "0.5")); g_ds_anneal = atoi(opt(argc, argv, "--ds-anneal", "1"));
        if (ds && !resume) cfg.ds_levels = g_ds;
        if (g_ds < 0 || g_ds > cfg.nlev - 2 || (resume && g_ds > cfg.ds_levels)) { fprintf(stderr, "--deep-sup: decoder levels 1..%d (a resumed net: at most its %d heads)\n", cfg.nlev - 2, cfg.ds_levels); return 2; }
        if (g_ds && (g_unwrap || g_aff || g_sheet || g_half || split || g_papyrus || g_verso || g_side || g_cond || g_cout != g_tc)) {
            fprintf(stderr, "--deep-sup: task surface (whole windows, no --half-res / --split / --verso / --side)\n"); return 2;
        }
        if (g_ds) fprintf(stderr, "deep supervision: decoder levels 1..%d, weights %g^j%s\n", g_ds, g_ds_w, g_ds_anneal ? ", annealed to 0 over the run" : "");
    }
    if (split) {
        g_h0 = 1 << (cfg.nlev - 1); g_Dl = P / 2 + g_h0;
        if (ng != 2) { fprintf(stderr, "--split z needs --gpus with two devices\n"); return 2; }
        if (P % (2 * g_h0)) { fprintf(stderr, "--split z: P must be divisible by %d\n", 2 * g_h0); return 2; }
        if (overfit || getenv("UFSM_SYNC_UPLOAD") || !nn_get_tf32()) { fprintf(stderr, "--split z: not with --overfit, UFSM_SYNC_UPLOAD or --fp32\n"); return 2; }
        g_zeros = nn_host_alloc((size_t)g_h0 * P * P); memset(g_zeros, 0, (size_t)g_h0 * P * P);
    }
    {   /* --half-res pool|pack: the network runs on 2x2x2 blocks of the window (nn.h nn_half_*); the checkpoint decides on resume */
        const char *hr = opt(argc, argv, "--half-res", "none");
        g_half = !strcmp(hr, "pool") ? 1 : !strcmp(hr, "pack") ? 2 : !strcmp(hr, "pool4") ? 3 : !strcmp(hr, "pack4") ? 4 : 0;
        if (!g_half && strcmp(hr, "none")) { fprintf(stderr, "--half-res none|pool|pack|pool4|pack4\n"); return 2; }
        if (resume) { unet_cfg pc; int st; if (!unet_peek(resume, &pc, &st)) g_half = pc.half_res; }
        g_hdiv = unet_hr_div(g_half);
        if (g_half) {
            const int pk = unet_hr_mode(g_half) == 2;
            if (g_aff || g_unwrap || g_verso || g_side || geometry || split || overfit || 8 * g_cout > 16) { fprintf(stderr, "--half-res: task surface only (no affinities, verso, side, winding, split or overfit)\n"); return 2; }
            if (P % (g_hdiv << (cfg.nlev - 1))) { fprintf(stderr, "--half-res: P must be divisible by %d\n", g_hdiv << (cfg.nlev - 1)); return 2; }
            cfg.half_res = g_half; cfg.cin = pk ? 8 : 4; cfg.cout = pk ? 8 * g_cout : g_cout;
            g_pkf = g_half == 2 && !g_heads && !g_pap_head && !g_papyrus && !g_ds && !atoi(opt(argc, argv, "--tol", "0"))   /* (no tolerance band in nn_loss_pack) */
                    && !(getenv("UFSM_PACK_DIRECT") && !atoi(getenv("UFSM_PACK_DIRECT")));
            fprintf(stderr, "low resolution (%s): the network runs on %d^3 blocks of %d^3 voxels, %d inputs, %d outputs\n", hr, P / g_hdiv, g_hdiv, cfg.cin, cfg.cout);
        }
    }
    {   /* --distill C: a 1^3 head on decoder level --distill-level (default 3, stride 8) predicts the sources' "emb" feature vectors
           (C channels, e.g. PCA-64 dinovol tokens) under a cosine loss of weight --distill-weight; the targets go through the window's
           symmetry and warp like the labels (src/sample_emb.h); a resumed net keeps its head */
        g_distill = atoi(opt(argc, argv, "--distill", "0")); g_distill_lev = atoi(opt(argc, argv, "--distill-level", "3"));
        g_distill_w = (float)atof(opt(argc, argv, "--distill-weight", "1"));
        g_distill_pap = (float)atof(opt(argc, argv, "--distill-papyrus", "0"));
        if (resume) { unet_cfg pc; int st; if (!unet_peek(resume, &pc, &st)) { cfg.emb_lev = pc.emb_lev; cfg.emb_ch = pc.emb_ch; if (pc.emb_lev && g_distill) { g_distill = pc.emb_ch; g_distill_lev = pc.emb_lev; } } }
        if (g_distill) {
            if (g_distill < 0 || g_distill > 512 || g_distill_lev < 1 || g_distill_lev > cfg.nlev - 2 || !(g_distill_w > 0)) {
                fprintf(stderr, "--distill C (1..512 channels), --distill-level 1..%d, --distill-weight > 0\n", cfg.nlev - 2); return 2;
            }
            if (split || g_half) { fprintf(stderr, "--distill: whole windows at full resolution (no --split z or --half-res)\n"); return 2; }
            g_distill_div = 1 << g_distill_lev;
            if (P % g_distill_div || g_Pz % g_distill_div) { fprintf(stderr, "--distill: P and Pz must be divisible by %d\n", g_distill_div); return 2; }
            cfg.emb_lev = g_distill_lev; cfg.emb_ch = g_distill;
            g_distill_n = (size_t)B * g_distill * (size_t)(g_Pz / g_distill_div) * (P / g_distill_div) * (P / g_distill_div);
            fprintf(stderr, "distill: %d channels on decoder level %d (%d^3 voxels per cell), cosine weight %g\n", g_distill, g_distill_lev, g_distill_div, g_distill_w);
        }
    }
    {   /* --aux-input 1: a fifth input channel from each source's aux volume (src/sources.h; e.g. the papyrus void fraction pooled
           from 2.4 um, which keeps the packed gaps a 9.6 um CT averages away); the checkpoint's cin decides on resume */
        g_xc = atoi(opt(argc, argv, "--aux-input", "0")) ? 5 : g_cond ? 7 : 4;
        if (resume) { unet_cfg pc; int st; if (!unet_peek(resume, &pc, &st) && !pc.half_res) g_xc = pc.cin == 5 || pc.cin == 7 ? pc.cin : 4; }
        if (g_xc == 5) {
            if (g_half || split || g_sheet || g_cond) { fprintf(stderr, "--aux-input: not with --half-res, --split, --cond or winding geometry\n"); return 2; }
            cfg.cin = 5; fprintf(stderr, "inputs: CT, radial (3), aux (sources' \"aux\" volumes)\n");
        }
        if (g_restore) {   /* 2 (split) or 1 input channels; the checkpoint decides on resume */
            g_xc = g_rsplit ? 2 : 1;
            if (resume) { unet_cfg pc; int st; if (!unet_peek(resume, &pc, &st)) { g_xc = pc.cin; g_rsplit = pc.cin == 2; } }
            cfg.cin = g_xc;
        }
        if ((g_xc == 7) != (g_cond != 0)) { fprintf(stderr, "--cond: the checkpoint's inputs (%d) and --cond %d disagree\n", g_xc, g_cond); return 2; }
        if (g_cond) {
            if (g_half || split || g_sheet) { fprintf(stderr, "--cond: not with --half-res, --split or winding geometry\n"); return 2; }
            cfg.cin = 7; fprintf(stderr, "inputs: CT, radial (3), winding condition (cos, sin, mask)\n");
        }
    }
    const int nl = split ? 1 : ng;   /* windows per step (loss averaging, samples / s) */
    if (g_sheet && nl!=1) { fprintf(stderr,"winding currently supports one whole or spatially split window\n"); return 2; }
    char cmd[1400]; snprintf(cmd, sizeof cmd, "mkdir -p '%s'", out); if (system(cmd)) return 1;

    sources *S = sources_load(src);
    g_Sdiag = S;
    if (!S) return 1;
    g_S = S;
    for (int i = 0; g_restore && i < S->n; i++) if (!S->src[i].srp != !g_rpairs) { fprintf(stderr, "--task restore: %s: \"sr_pair\" sources need --sr-pairs 1 and --sr-pairs 1 needs them all\n", S->src[i].name); return 2; }
    if (g_distill) {   /* --distill: which sources have feature targets, and whether their groups open with the head's channel count */
        int ns = 0, bad = 0;
        for (int i = 0; i < S->n; i++) {
            const source *s = &S->src[i]; int ok = 0;
            for (int k = 0; k < s->nemb; k++) {
                emb e; if (emb_open(&e, s->emb_grp[k], s->emb_name)) { fprintf(stderr, "distill: %s: %s/%s does not open (no targets from it)\n", s->name, s->emb_grp[k], s->emb_name); continue; }
                if (e.a.C != g_distill || e.a.dt != 0) { fprintf(stderr, "distill: %s: %s/%s: %d channels / dtype %d, the head %d int8\n", s->name, s->emb_grp[k], s->emb_name, e.a.C, e.a.dt, g_distill); bad++; }
                else ok++;
                emb_close(&e);
            }
            ns += ok > 0;
            if (s->nemb) fprintf(stderr, "distill: %s: %d of %d \"emb\" groups (%s, level %d)\n", s->name, ok, s->nemb, s->emb_name, s->emb_level);
            if (s->nemb && g_distill_pap > 0) { if (s->pap_level > 0) fprintf(stderr, "distill: %s: targets on cells >= %.0f%% papyrus (CT >= %d)\n", s->name, 100. * g_distill_pap, s->pap_level);
                                                else fprintf(stderr, "distill: WARNING: %s: no \"papyrus\" / \"iso\" level: its targets are not gated\n", s->name); }
        }
        if (bad || !ns) { fprintf(stderr, "distill: %s\n", bad ? "groups with the wrong channel count / type" : "no source has an \"emb\" group that opens"); return 2; }
    }
    for (int i = 0; i < S->n; i++) {   /* --heads: every source's head exists; sources naming heads need --heads */
        const int h = S->src[i].head;
        if (h && (!g_heads || h >= g_heads)) { fprintf(stderr, "sources: %s trains head %d; train --heads %d or more\n", S->src[i].name, h, h + 1); return 2; }
    }
    if (g_heads) {
        for (int h = 0; h < g_heads; h++) {
            fprintf(stderr, "head %d (channel %d):", h, h ? 1 + h : 0); int k = 0;
            for (int i = 0; i < S->n; i++) if (S->src[i].head == h) { fprintf(stderr, " %s", S->src[i].name); k++; }
            fprintf(stderr, "%s\n", k ? "" : " no source (untrained)");
        }
    }
    sample_cfg sc = sample_cfg_default();
    sc.P = P; sc.Pz = g_Pz != P ? g_Pz : 0; sc.B = B; sc.nworkers = workers; sc.nbuf = atoi(opt(argc, argv, "--sampler-slots", "0")) > 0 ? atoi(opt(argc, argv, "--sampler-slots", "0")) : 4 * ng + 2;   /* batches in flight (each worker fills one; pinned memory per slot) */ sc.seed = seed; sc.xfmt = g_xfmt;
    g_axr = unet_cfg_axis(&cfg); sc.axis_rec = g_axr ? g_hdiv : 0;   /* the scroll axis per window */
    if (g_axr && split) { fprintf(stderr, "--attn-axis radial / --attn-pe scroll / --attn-deform-init normal: not with --split z\n"); return 2; }
    if (g_axr && !cfg.attn_deform) fprintf(stderr, "attention: %s, %s positions around each window's scroll axis\n", cfg.attn_axis ? "radial (along the rays from the axis)" : "full", cfg.attn_pe == 3 ? "scroll RoPE" : "grid");
    if (g_axr && cfg.attn_deform) fprintf(stderr, "attention: deformable offsets in each token's normal / tangent / axis frame around the window's scroll axis\n");
    sc.deterministic = atoi(opt(argc, argv, "--det", "0"));   /* same batches for the same seed regardless of thread timing (paired comparisons; off: a slow worker can stall the ring) */
    g_mem = cfg.attn_blocks > 0 && cfg.attn_mem > 0;   /* window memory tokens, trained along window chains */
    g_view = cfg.attn_blocks > 0 ? cfg.view_P : 0; sc.view_P = g_view;   /* scan fusion: the sampler's view crops */
    if (cfg.view_P && (cfg.attn_blocks <= 0 || cfg.attn_axis || split || g_restore || cfg.view_patch <= 0 || cfg.view_P % cfg.view_patch || cfg.view_P & 1)) {
        fprintf(stderr, "--view-P: needs --attn-blocks, full attention, no --split / restore, an even size that --view-patch divides\n"); return 2; }
    if (g_view) { int nv = 0; for (int i = 0; i < S->n; i++) nv += S->src[i].view; fprintf(stderr, "attention: scan fusion: %d^3 view crops, %d^3 patches -> %d view rows per window; %d of %d sources with a view\n", cfg.view_P, cfg.view_patch, (cfg.view_P / cfg.view_patch) * (cfg.view_P / cfg.view_patch) * (cfg.view_P / cfg.view_patch), nv, S->n); }
    sc.chain_p = (float)atof(opt(argc, argv, "--chain-p", "0")); sc.chain_len = atoi(opt(argc, argv, "--chain-len", "8")); sc.chain_step = (float)atof(opt(argc, argv, "--chain-step", "0.5"));
    if (cfg.attn_deform > 0) {
        if (cfg.attn_blocks <= 0 || split) { fprintf(stderr, "--attn-deform: needs --attn-blocks, no --split\n"); return 2; }
        fprintf(stderr, "attention: deformable, %d points per head on each of %d levels%s, %s offsets\n", cfg.attn_deform, cfg.attn_dlev + cfg.attn_dco,
                cfg.attn_dco ? " (incl. the 2x pooled coarse level)" : "", cfg.attn_dinit ? "normal-frame" : "grid");
    }
    if (cfg.attn_mem > 0 && (cfg.attn_blocks <= 0 || cfg.attn_axis)) { fprintf(stderr, "--attn-mem: needs --attn-blocks and full attention\n"); return 2; }
    if (sc.chain_p > 0 && (!g_mem || sc.deterministic || split || sc.chain_len < 2 || !(sc.chain_step > 0))) { fprintf(stderr, "--chain-p: needs --attn-mem, --det 0, no --split, --chain-len >= 2, --chain-step > 0\n"); return 2; }
    if (g_mem) fprintf(stderr, "attention: %d window memory tokens; window chains: %s\n", cfg.attn_mem, sc.chain_p > 0 ? "on" : "off (--chain-p 0: the memory input stays zero)");
    if (noaug) sc.augment = 0;
    if (atoi(opt(argc, argv, "--rotonly", "0"))) sc.augment = 2;   /* proper rotations only (no reflections) */
    if (atoi(opt(argc, argv, "--zfix", "0"))) sc.augment = 3;      /* diagnostic: symmetries that keep the z axis */
    if (atoi(opt(argc, argv, "--intonly", "0"))) sc.augment = 4;   /* diagnostic: intensity jitter, no symmetry */
    sc.aux = g_xc == 5; sc.cond = g_cond; sc.cond_outside = g_cond_outside; sc.pap_head = g_pap_head ? &g_paph : nullptr;
    sc.band = g_aff; sc.side = g_side; sc.pmask = g_unwrap;   /* unwrap: the phase only (band codes per window, or the stored phase) */ sc.side_air = atoi(opt(argc, argv, "--side-air", "0"));
    sc.wind4 = g_unwrap2; sc.wind4_step = atoi(opt(argc, argv, "--wind4-step", "80")); sc.wind4_min = atoi(opt(argc, argv, "--wind4-min", "512"));   /* unwrap2: band_wind4's join limit (phase codes) and smallest component */
    sc.emb_ch = g_distill; sc.emb_div = g_distill_div; sc.emb_pap = g_distill_pap;   /* --distill: the feature targets per head cell */
    { const int t = atoi(opt(argc, argv, "--wind4-tile", "64")); sc.wind4_tile = t > 0 ? t : -1; }   /* unwrap2: one gauge per tile of T voxels (>= P / 6), 0 = per window */
    sc.min_phase = atof(opt(argc, argv, "--min-phase", g_unwrap2 ? "0.001" : "0"));   /* side 2: windows with less known phase than this fraction are redrawn (unwrap2: 0.1%: band / phase-only sources draw anywhere) */
    if (!(sc.min_phase >= 0 && sc.min_phase < 1) || (sc.min_phase > 0 && g_side != 2)) { fprintf(stderr, "--min-phase: 0 .. 1, with the phase head (task unwrap / unwrap2)\n"); return 2; }
    if (g_unwrap2 && (sc.wind4_step < 1 || sc.wind4_step > 125 || sc.wind4_min < 1)) { fprintf(stderr, "--wind4-step 1..125 (phase codes, 252 per turn), --wind4-min >= 1\n"); return 2; }
    sc.papyrus = g_papyrus ? &g_pap : nullptr;
    sc.restore = g_restore ? 1 + g_rpairs : 0; if (g_restore) sc.min_fg = optd(argc, argv, "--min-fg", 0.5);
    sc.shell = atoi(opt(argc, argv, "--shell", "0"));   /* background within R level-0 voxels of a label ignored */
    sc.iso = atoi(opt(argc, argv, "--iso", "0"));          /* void cutoff of sources without "iso" */
    sc.iso_void = atoi(opt(argc, argv, "--iso-void", "0")); sc.iso_pap = atoi(opt(argc, argv, "--iso-pap", "2")); sc.iso_lift = atoi(opt(argc, argv, "--iso-lift", "0"));
    if (sc.iso < 0 || sc.iso > 255 || sc.iso_pap < 0 || sc.iso_pap > 84 || sc.iso_lift < 0 || sc.iso_lift > 84) { fprintf(stderr, "--iso 0..255, --iso-pap / --iso-lift 0..84\n"); return 2; }
    if (sc.iso_void || sc.iso_lift) {
        int any = sc.iso > 0; for (int i = 0; i < S->n; i++) any |= S->src[i].iso > 0;
        if (!any) { fprintf(stderr, "--iso-void / --iso-lift: no void cutoff (sources \"iso\" or --iso T)\n"); return 2; }
        fprintf(stderr, "void cutoff: the sources' \"iso\", else %d (papyrus = CT >= the cutoff)\n", sc.iso);
        if (sc.iso_void) fprintf(stderr, "  void rule: 0 < CT < cutoff and > %d voxels from papyrus -> background everywhere (CT 0 stays ignored)\n", sc.iso_pap);
        if (sc.iso_lift) fprintf(stderr, "  lift rule: label voxels with no papyrus within %d voxels -> ignored\n", sc.iso_lift);
    }
    sc.dist_sigma = (float)atof(opt(argc, argv, "--dist-head", "0"));   /* channel 1: a distance ridge of channel 0 (sigma, --soft-unit) */
    if (sc.shell < 0 || sc.shell > 84 || !(sc.dist_sigma >= 0.f)) { fprintf(stderr, "--shell 0..84, --dist-head >= 0\n"); return 2; }
    sc.phase_dense = atoi(opt(argc, argv, "--phase-dense", "0"));   /* side 2: weigh windings closer than this (label voxels) up */
    if (sc.phase_dense && g_side != 2) { fprintf(stderr, "--phase-dense needs --side 2\n"); return 2; }
    if (sc.phase_dense) fprintf(stderr, "phase: windings wider than %d label voxels supervised at a rate dense / spacing\n", sc.phase_dense);
    /* --phase-near-um X: the stored phase supervised only within X um of a labelled wrap voxel (sample.h phase_near_um): the phase
       raster interpolates between published wraps, wrongly across a missing one (on a Paris4 strip 82% of the phase target is
       interpolated; the network's phase is grossly off in 20% of the cells on labelled wraps, 30% between them) */
    sc.phase_near_um = (float)atof(opt(argc, argv, "--phase-near-um", "0"));
    if (sc.phase_near_um < 0 || (sc.phase_near_um > 0 && g_side != 2)) { fprintf(stderr, "--phase-near-um: >= 0, task unwrap (the phase)\n"); return 2; }
    if (sc.phase_near_um > 0) fprintf(stderr, "phase: supervised within %g um of a labelled wrap voxel\n", sc.phase_near_um);
    /* --aug all|none: the augmentation preset (sample.c sample_aug_preset); the flags below override its fields. Default "all" for the
       dense-target tasks on cube windows (every scanner op, continuous geometry, zoom in / out, axis jitter: training for other scrolls,
       scanners and damage), "none" (symmetries + intensity jitter, the recipe before 2026-10-16) for the other tasks and for slabs.
       --aug-p-scale X multiplies every op probability of the preset (an A/B dial); resume an older run with --aug none to keep its data */
    const int dense_task = !g_sheet && !g_papyrus && !g_aff && !g_side && !g_cond && !g_unwrap && g_xc == 4 && g_Pz == P;
    g_aug_name = opt(argc, argv, "--aug", dense_task ? "all" : "none");
    if (sample_aug_preset(&sc, g_aug_name)) { fprintf(stderr, "--aug: all | none\n"); return 2; }
    if (!dense_task && !strcmp(g_aug_name, "all")) { fprintf(stderr, "--aug all: the dense-target tasks on cube windows only\n"); return 2; }
    {
        const float ps = (float)atof(opt(argc, argv, "--aug-p-scale", "1"));
        if (!(ps >= 0 && ps <= 4)) { fprintf(stderr, "--aug-p-scale: 0..4\n"); return 2; }
        for (int i = 0; i < SA_N; i++) sc.scan.p[i] = fminf(1, sc.scan.p[i] * ps);
        sc.rotate_p = fminf(1, sc.rotate_p * ps); sc.elastic_p = fminf(1, sc.elastic_p * ps); sc.zoom_p = fminf(1, sc.zoom_p * ps);
        sc.zoom_out_p = fminf(1, sc.zoom_out_p * ps); sc.affine_p = fminf(1, sc.affine_p * ps); sc.axis_jitter_p = fminf(1, sc.axis_jitter_p * ps);
    }
    sc.ct_augment = (int)optd(argc, argv, "--ct-aug", sc.ct_augment);
    sc.symmetry_p = (float)atof(opt(argc, argv, "--symmetry-p", "1"));
    sc.axis_jitter = (float)optd(argc, argv, "--axis-jitter", sc.axis_jitter);
    /* --axis-jitter-p P (default 0.1), --axis-jitter-deg D (default 2): how often the axis moves and the cap on the radial input's
       direction change; an estimated umbilicus on another scroll is off by ~1-3 mm (on block B, 3 mm costs phase confidence .65 -> .58) */
    sc.axis_jitter_p = (float)optd(argc, argv, "--axis-jitter-p", sc.axis_jitter_p); sc.axis_jitter_deg = (float)optd(argc, argv, "--axis-jitter-deg", sc.axis_jitter_deg);
    sc.geometry_augment = (int)optd(argc,argv,"--geometry-aug",sc.geometry_augment);
    sc.rotate_degrees=(float)optd(argc,argv,"--rotate-deg",sc.rotate_degrees);
    sc.rotate_p=(float)optd(argc,argv,"--rotate-p",sc.rotate_p);
    sc.elastic=(float)optd(argc,argv,"--elastic",sc.elastic);
    sc.elastic_p=(float)optd(argc,argv,"--elastic-p",sc.elastic_p);
    sc.zoom_min=(float)optd(argc,argv,"--zoom-min",sc.zoom_min);   /* zoom-in down to this scale (needs --geometry-aug 1) */
    sc.zoom_p=(float)optd(argc,argv,"--zoom-p",sc.zoom_p);
    sc.zoom_max=(float)optd(argc,argv,"--zoom-max",sc.zoom_max);   /* zoom-out up to this factor (1..2: the next coarser level, needs --geometry-aug 1) */
    sc.zoom_out_p=(float)optd(argc,argv,"--zoom-out-p",sc.zoom_out_p);
    sc.label_morph=(float)atof(opt(argc,argv,"--label-morph","0"));
    sc.label_morph_p=(float)atof(opt(argc,argv,"--label-morph-p","0.2"));
    sc.scan.strength=(float)optd(argc,argv,"--aug-strength",sc.scan.strength);
    sc.scan.drop_ignore=(int)optd(argc,argv,"--aug-drop-ignore",sc.scan.drop_ignore);
    {
        const char *all=opt(argc,argv,"--aug-p",nullptr);   /* every op at this probability (overrides the preset's) */
        for (int i=0;i<SA_N;i++) { char f[48]; snprintf(f,sizeof f,"--aug-%s-p",scan_aug_names[i]); sc.scan.p[i]=(float)optd(argc,argv,f,all?atof(all):sc.scan.p[i]); }
        for (int i=0;i<SA_N;i++) { char f[48]; snprintf(f,sizeof f,"--aug-%s-s",scan_aug_names[i]); sc.scan.s[i]=(float)atof(opt(argc,argv,f,"0")); }   /* per-op strength (0: --aug-strength) */
        const char *hf=opt(argc,argv,"--aug-hist",nullptr);   /* target histograms for the hist op (ufsm ct-hist) */
        if (hf && scan_aug_load_hist(hf,&sc.scan)) { fprintf(stderr,"--aug-hist: cannot read %s\n",hf); return 2; }
        if (hf && !strcmp(g_aug_name,"all") && !all && !opt(argc,argv,"--aug-hist-p",nullptr)) sc.scan.p[SA_HIST]=.15f;   /* the preset's, when targets are given */
        if (sc.scan.p[SA_HIST]>0 && !sc.scan.nhist) { fprintf(stderr,"--aug-hist-p needs --aug-hist FILE\n"); return 2; }
        for (int i=0;i<SA_N;i++) if (!isfinite(sc.scan.s[i]) || sc.scan.s[i]<0 || sc.scan.s[i]>3) { fprintf(stderr,"--aug-%s-s: 0..3\n",scan_aug_names[i]); return 2; }
    }
    sc.affine_p=(float)optd(argc,argv,"--aug-affine-p",sc.affine_p);
    sc.affine_aniso=(float)optd(argc,argv,"--aug-aniso",sc.affine_aniso);
    sc.affine_shear=(float)optd(argc,argv,"--aug-shear",sc.affine_shear);
    {
        int bad=!isfinite(sc.scan.strength) || sc.scan.strength<0 || sc.scan.strength>2 || sc.scan.drop_ignore<0 || sc.scan.drop_ignore>1 ||
                !isfinite(sc.affine_p) || sc.affine_p<0 || sc.affine_p>1 || !isfinite(sc.affine_aniso) || sc.affine_aniso<0 || sc.affine_aniso>.3f ||
                !isfinite(sc.affine_shear) || sc.affine_shear<0 || sc.affine_shear>.2f || (sc.affine_p>0 && !sc.geometry_augment);
        for (int i=0;i<SA_N;i++) bad|=!isfinite(sc.scan.p[i]) || sc.scan.p[i]<0 || sc.scan.p[i]>1;
        if (bad) { fprintf(stderr,"invalid scanner/affine augmentation bounds (--aug-affine-p needs --geometry-aug 1)\n"); return 2; }
    }
    if (crush_cfg_parse(&sc.crush,argc,argv)) return 2;
    if (sc.crush.p>0) {
        if (!sc.geometry_augment || noaug) { fprintf(stderr,"--aug-crush needs --geometry-aug 1 (and augmentation on)\n"); return 2; }
        crush_cfg_print(&sc.crush);
    }
    sc.xscan_p = (float)atof(opt(argc, argv, "--aug-xscan", "0"));
    if (!(sc.xscan_p >= 0 && sc.xscan_p <= 1)) { fprintf(stderr, "--aug-xscan: a probability 0 .. 1\n"); return 2; }
    if (sc.xscan_p > 0 && noaug) { fprintf(stderr, "--aug-xscan needs augmentation on\n"); return 2; }
    sc.tear_p = (float)atof(opt(argc, argv, "--aug-tear", "0")); sc.tear_um = (float)atof(opt(argc, argv, "--aug-tear-um", "480"));
    if (!(sc.tear_p >= 0 && sc.tear_p <= 1) || !(sc.tear_um > 0)) { fprintf(stderr, "--aug-tear: a probability 0 .. 1, --aug-tear-um > 0\n"); return 2; }
    if (sc.tear_p > 0 && noaug) { fprintf(stderr, "--aug-tear needs augmentation on\n"); return 2; }
    sc.delam_p = (float)atof(opt(argc, argv, "--aug-delam", "0")); sc.delam_hole = (float)atof(opt(argc, argv, "--aug-delam-hole", "0.5"));
    if (!(sc.delam_p >= 0 && sc.delam_p <= 1) || !(sc.delam_hole >= 0 && sc.delam_hole <= 1)) { fprintf(stderr, "--aug-delam / --aug-delam-hole: probabilities 0 .. 1\n"); return 2; }
    if (sc.delam_p > 0 && noaug) { fprintf(stderr, "--aug-delam needs augmentation on\n"); return 2; }
    sc.incl_p = (float)atof(opt(argc, argv, "--aug-inclusions", "0"));
    if (!(sc.incl_p >= 0 && sc.incl_p <= 1)) { fprintf(stderr, "--aug-inclusions: a probability 0 .. 1\n"); return 2; }
    if (sc.incl_p > 0 && noaug) { fprintf(stderr, "--aug-inclusions needs augmentation on\n"); return 2; }
    sc.curv_p = (float)atof(opt(argc, argv, "--aug-curv", "0"));
    if (!(sc.curv_p >= 0 && sc.curv_p <= 1)) { fprintf(stderr, "--aug-curv: a probability 0 .. 1\n"); return 2; }
    if (sc.curv_p > 0 && noaug) { fprintf(stderr, "--aug-curv needs augmentation on\n"); return 2; }
    sc.density_p = (float)atof(opt(argc, argv, "--aug-sheet-density", "0"));
    if (!(sc.density_p >= 0 && sc.density_p <= 1)) { fprintf(stderr, "--aug-sheet-density: a probability 0 .. 1\n"); return 2; }
    if (sc.density_p > 0 && noaug) { fprintf(stderr, "--aug-sheet-density needs augmentation on\n"); return 2; }
    sc.sheet=g_sheet;
    if (g_sheet) { sc.augment=noaug?0:sc.geometry_augment?1:3; memset(sc.level_p,0,sizeof sc.level_p); sc.level_p[0]=1; }
    if (!isfinite(sc.symmetry_p) || sc.symmetry_p < 0 || sc.symmetry_p > 1 || !isfinite(sc.axis_jitter) || sc.axis_jitter < 0 ||
        !(sc.axis_jitter_p >= 0 && sc.axis_jitter_p <= 1) || !(sc.axis_jitter_deg >= 0 && sc.axis_jitter_deg <= 60)) { fprintf(stderr, "invalid augmentation bounds\n"); return 2; }
    if (sc.geometry_augment<0 || sc.geometry_augment>1 || !isfinite(sc.rotate_degrees) || sc.rotate_degrees<0 || sc.rotate_degrees>15 || !isfinite(sc.elastic) || sc.elastic<0 || sc.elastic>10 ||
        !isfinite(sc.rotate_p) || sc.rotate_p<0 || sc.rotate_p>1 || !isfinite(sc.elastic_p) || sc.elastic_p<0 || sc.elastic_p>1 ||
        !isfinite(sc.zoom_min) || sc.zoom_min<0.25f || sc.zoom_min>1 || !isfinite(sc.zoom_p) || sc.zoom_p<0 || sc.zoom_p>1 ||
        !isfinite(sc.zoom_max) || sc.zoom_max<1 || sc.zoom_max>2 || !isfinite(sc.zoom_out_p) || sc.zoom_out_p<0 || sc.zoom_out_p>1 ||
        !isfinite(sc.label_morph) || sc.label_morph<0 || sc.label_morph>1 || !isfinite(sc.label_morph_p) || sc.label_morph_p<0 || sc.label_morph_p>1) {
        fprintf(stderr,"invalid continuous geometry/morphology augmentation bounds\n"); return 2;
    }
    {   /* --gpu-input 1 (default where possible): the batches carry the CT channel alone plus a radial record per sample, the 4 MX input
           channels are built on the device (nn_input_rad): 6 B per voxel less per sampler slot, upload and device batch, no CPU x16
           radial pass. Needs the 16-bit -> MX input conversion, whole windows, the plain 4 inputs and write_x samples; --gpu-input 0 or
           UFSM_DUMP_BATCH (it shows the 4 channels): the host-built 4 channels as before */
        const int want = atoi(opt(argc, argv, "--gpu-input", "1"));
        g_gpu_want = want && g_xfmt && unet_act_mx() && !split && (!g_half || g_pkf) && g_xc == 4 && sample_gpu_input_ok(&sc) && !getenv("UFSM_DUMP_BATCH");
        g_gpu_asked = atoi(opt(argc, argv, "--gpu-input", "0"));   /* (an explicit request: say when it cannot apply) */
        g_gpu_aug_asked = opt(argc, argv, "--gpu-aug", nullptr) != nullptr;
        g_gpu_aug_want = atoi(opt(argc, argv, "--gpu-aug", newdef ? "2" : "0"));   /* --gpu-aug 1: the CT channel built on the device (nn_aug_ct); 2: also the warped windows' labels (nn_label_warp) */
        if (g_gpu_aug_want < 0 || g_gpu_aug_want > 2) { fprintf(stderr, "--gpu-aug: 0, 1 or 2\n"); return 2; }
        if (atoi(opt(argc, argv, "--overfit", "0")) || split) g_gpu_aug_want = 0;
        /* the MX input conversion it needs depends on the memory mode (MX gradients): settled after the planner (gpu_input_settle) */
    }
    {   /* --pack-targets 1 (default where it applies): hard two-class targets (every source's recto target side-encoded: 0 / 255 in both
           channels) travel as one code byte per voxel (sample.h SAMPLE_TM_*) through the sampler slots, the validation batches and the
           upload, unpacked on the device (nn_unpack_tm): 3 -> 1 B per voxel on the host and over PCIe. Not with --heads (per-source
           target rows), --split, the papyrus task, --dist-head, the sheet prior or --overfit (geometry warps: nearest-code targets);
           --pack-targets 0: target and mask bytes */
        int hard = S->n > 0;
        for (int i = 0; i < S->n; i++) {   /* soft side votes: fractions; a source without target channels (band codes + stored phase only, the
                                              --task unwrap2 sources) has all-zero targets, which pack (warps: the nearest code, also 0) */
            int none = 1; for (int c = 0; c < NCH; c++) none &= !S->src[i].tgt_key[c] && !S->src[i].reg[c];
            hard &= (S->src[i].tgt_side[CH_RECTO] && !S->src[i].tgt_votes_soft[CH_RECTO]) || none;
        }
        g_pack = atoi(opt(argc, argv, "--pack-targets", "1")) && hard && !g_heads && !split && !sc.papyrus && sc.dist_sigma <= 0 && !sc.sheet && !overfit && !g_restore;   /* (warped windows take the nearest source voxel's code) */
        sc.pack_tm = g_pack;
        if (g_pack) fprintf(stderr, "targets: packed, one byte per voxel on the host and in the upload (--pack-targets 0: target and mask bytes)\n");
        else if (atoi(opt(argc, argv, "--pack-targets", "0"))) fprintf(stderr, "--pack-targets: not available here (needs side-encoded targets in every source, no --heads, --split, papyrus task, --dist-head, sheet prior or --overfit)\n");
    }
    cover_plan *plan = nullptr; cover_progress coverage = {0}, prior_cover = {0};
    const char *cover_path = opt(argc, argv, "--cover", nullptr);
    int warm_start=atoi(opt(argc,argv,"--warm-start","0"));
    if (warm_start && (!resume || sheet_ck)) { fprintf(stderr,"--warm-start requires a legacy donor checkpoint\n"); return 2; }
    int has_cover = resume && !sheet_init && !warm_start ? cover_checkpoint_read(resume, &prior_cover) : 0;
    if (has_cover < 0 || (has_cover && !cover_path)) { fprintf(stderr, "finite-cover resume requires its original plan\n"); return 2; }
    int next_pass = atoi(opt(argc, argv, "--cover-next-pass", "0"));
    if (next_pass < 0 || next_pass > 1 || (next_pass && (!has_cover || !cover_path || prior_cover.cursor != prior_cover.count))) {
        fprintf(stderr, "--cover-next-pass requires a completed finite-cover checkpoint and the next plan\n"); return 2;
    }
    if (next_pass) has_cover = 0; /* Keep model, EMA, optimizer and global step; reset only the tile cursor. */
    const char *extend_from = opt(argc, argv, "--cover-extend-from", nullptr);
    if (extend_from && (!has_cover || !cover_path)) { fprintf(stderr, "cover extension requires a matching finite-cover checkpoint\n"); return 2; }
    if (cover_path) {
        if (B != 1 || nl != 1 || overfit || seconds || atoi(opt(argc, argv, "--finetune", "0"))) { fprintf(stderr, "cover requires B=1, one window per step, step-based schedule and no overfit/finetune\n"); return 2; }
        plan = cover_load(cover_path, S, P); if (!plan) return 2;
        const char *pinned = opt(argc, argv, "--cover-sha256", plan->sha256);
        if (extend_from) {
            cover_plan *previous = cover_load(extend_from, S, P);
            int valid = cover_validate_extension(previous, plan, &prior_cover);
            cover_free(previous);
            if (valid) { fprintf(stderr, "cover extension must preserve the saved plan as an unchanged prefix\n"); return 2; }
            fprintf(stderr, "extending cover: %llu -> %llu tiles, preserving cursor and schedule origin\n",
                (unsigned long long)prior_cover.count, (unsigned long long)plan->count);
            strcpy(prior_cover.sha256, plan->sha256); prior_cover.count = plan->count;
        }
        if (strcmp(pinned, plan->sha256) || (has_cover && (strcmp(prior_cover.sha256, plan->sha256) || prior_cover.count != plan->count))) { fprintf(stderr, "cover SHA256/count mismatch\n"); return 2; }
        coverage = next_pass ? (cover_progress){0} : prior_cover; strcpy(coverage.sha256, plan->sha256); coverage.count = plan->count;
        sc.cover = plan; sc.cover_start = coverage.cursor; sc.deterministic = 1; sc.snap = 0;
        memset(sc.level_p, 0, sizeof sc.level_p); sc.level_p[0] = 1;
    }
    sc.dilate = atoi(opt(argc, argv, "--dilate", "0"));   /* thicken surface targets by D level-0 voxels (curriculum) */
    sc.erode = atoi(opt(argc, argv, "--erode", "0"));
    nn_set_loss_tol(atoi(opt(argc, argv, "--tol", "0")));   /* offset-tolerant positives, radius in output voxels (<= 2) */    nn_set_recall((float)atof(opt(argc, argv, "--recall", "0")));   /* skeleton-recall term on the recto (thin labels: the label is its skeleton) */
    if (sc.erode < 0 || sc.erode > 1 || (sc.erode && (sc.dilate || g_sheet))) {
        fprintf(stderr, "--erode requires 0 or 1, no --dilate and no winding task\n"); return 2;
    }
    sc.soft = (float)atof(opt(argc, argv, "--soft", "0"));  /* soft ridge target with this sigma (level-0 voxels) */
    { const char *u = opt(argc, argv, "--soft-unit", "vox");   /* um: --soft/--soft-end in um for every source and level */
      if (strcmp(u, "vox") && strcmp(u, "um")) { fprintf(stderr, "--soft-unit must be vox or um\n"); return 2; }
      sc.soft_um = !strcmp(u, "um");
      if (sc.soft_um && g_sheet) { fprintf(stderr, "--soft-unit um is not supported with the winding task\n"); return 2; } }
    double sheet_band_scale=atof(opt(argc,argv,"--sheet-band-scale","1"));
    if (!isfinite(sheet_band_scale) || sheet_band_scale<1 || (!g_sheet && sheet_band_scale!=1) ||
        (g_sheet && sheet_widen_bands(g_sheet,sheet_band_scale))) {
        fprintf(stderr,"invalid sheet band scale or close-gap radius exceeds 32 native voxels\n"); return 2;
    }
    if (g_sheet && (!isfinite(sc.soft) || sc.soft<=0 || sc.soft+sc.label_morph>g_sheet->max_soft_sigma)) {
        fprintf(stderr,"winding soft sigma must be positive; sigma plus morphology expansion must be <= audited spacing cap %g\n",g_sheet->max_soft_sigma); return 2;
    }
    float soft_end = (float)atof(opt(argc, argv, "--soft-end", "-1")); if (soft_end < 0) soft_end = sc.soft;   /* sigma annealed linearly to this value at the last step */
    if (plan && soft_end != sc.soft) { fprintf(stderr, "finite cover requires a fixed soft-target sigma for deterministic resume\n"); return 2; }
    if (soft_end != sc.soft) fprintf(stderr, "soft target sigma annealed %g -> %g; the fixed validation batches keep sigma %g, so the validation loss (and best.ckpt) is not comparable across the run: score last.ckpt\n", sc.soft, soft_end, sc.soft);
    { const char *lv = opt(argc, argv, "--levels", nullptr); if (lv) { char *t = strdup(lv); int l = 0; memset(sc.level_p, 0, sizeof sc.level_p); for (char *q = strtok(t, ","); q && l < MAXLEV; q = strtok(nullptr, ",")) sc.level_p[l++] = atof(q); free(t); } }
    if (g_sheet) { memset(sc.level_p,0,sizeof sc.level_p); sc.level_p[0]=1; if (S->n!=1) { fprintf(stderr,"winding v1 requires one registered source\n"); return 2; } }

    size_t p3 = (size_t)g_Pz * P * P, p3l = split ? (size_t)P * P * g_Dl : p3;   /* per-GPU voxels of a window */
    gpu_state G[8];
    int step0 = 0;
    for (int g = 0; g < ng; g++) {
        gpu_state *d = &G[g];
        d->dev = devs[g]; d->side = split ? g : -1;
        d->axh[0] = d->axh[1] = nullptr; d->axv[0] = d->axv[1] = d->axn[0] = d->axn[1] = d->axs[0] = d->axs[1] = 0;
        d->chh[0] = d->chh[1] = nullptr; d->chih[0] = d->chih[1] = nullptr; d->choh[0] = d->choh[1] = nullptr; d->chv[0] = d->chv[1] = 0;
        d->vh[0] = d->vh[1] = d->vd = nullptr; d->vhn[0] = d->vhn[1] = d->vdn = 0; d->vv[0] = d->vv[1] = 0;
        if (nn_init(d->dev)) { fprintf(stderr, "cannot select GPU %d\n", d->dev); return 1; }
        d->u = unet_create(&cfg);
        unet_set_l0_ema(d->u, (float)atof(opt(argc, argv, "--l0-mom", "0.01")), atoi(opt(argc, argv, "--l0-warm", "100")));   /* --l0-norm ema */
        unet_set_wd_nonconv(d->u, (float)atof(opt(argc, argv, "--wd-nonconv", "0")));   /* Muon / ANVIL: no decay on GroupNorm, biases, head */
        unet_set_muon_ext(d->u, muon_c, hyperball, hb_mult);
        if (trunk_lr != 1.f) unet_set_trunk_lr(d->u, trunk_lr);
        /* fp16 logits in their own buffer (MX activations; the losses read them as unet_logits_h16 says): 11 unwrap heads at P 384
           are 2.5 GB in fp32. Not for half-resolution nets (the losses read the fp32 upsampled copy) or sheet nets; UFSM_LOGITS32=1 */
        unet_set_fp16_logits(d->u, (!g_half || g_pkf) && !g_sheet && !ufsm_env_on("UFSM_LOGITS32"));
        if (resume) {
            const int grow = atoi(opt(argc, argv, "--grow-head", "0"));   /* band_affinity warm start from a recto checkpoint */
            step0 = grow ? unet_load_grow(d->u, resume, 1, 1.f) : unet_load(d->u, resume);
            if (step0 < 0) { fprintf(stderr, "cannot load %s\n", resume); return 1; }
            if (atoi(opt(argc, argv, "--finetune", "0"))) step0 = 0;   /* loaded weights/state, fresh schedule */
            if (sheet_init && unet_start_sheet(d->u)) return 2;
        } else {
            unet_init(d->u, seed + 1); /* deterministic: every GPU starts identical */
            if (g_restore) unet_zero_head(d->u);   /* residual net: output == input at step 0 */
            const char *init_from = opt(argc, argv, "--init-from", nullptr);   /* partial warm start from another architecture */
            if (init_from) {
                int rows[16];   /* --verso: recto copied into the verso channel, the donor's other rows (affinities) shifted up */
                if (g_verso) { unet_cfg dc; int ds; if (unet_peek(init_from, &dc, &ds)) { fprintf(stderr, "cannot read %s\n", init_from); return 1; }
                    rows[0] = 0; rows[1] = g_verso_zero ? -1 : 0;
                    for (int co = 2; co < cfg.cout; co++) rows[co] = dc.cout == cfg.cout ? co : co - 1 < dc.cout ? co - 1 : -1;
                    if (dc.cout == cfg.cout && g_aff) { fprintf(stderr, "--verso --init-from: %s already has %d outputs; use --resume\n", init_from, dc.cout); return 1; } }
                const char *hm = opt(argc, argv, "--head-map", nullptr);   /* explicit donor row per output channel (-1 fresh), e.g. 0,2,3,4,5,6,7,-1 */
                int w4map = 0;   /* unwrap2 from a task unwrap donor (recto, ink, phase 2-3, normal / density 4-10): its rows 2..10 into 0..8, the winding mod 4 fresh */
                if (g_unwrap2 && !hm && !g_verso) {
                    unet_cfg dc; int ds;
                    if (unet_peek(init_from, &dc, &ds)) { fprintf(stderr, "cannot read %s\n", init_from); return 1; }
                    if (checkpoint_unwrap_version(init_from) == 1 && dc.cout >= 11) {
                        for (int co = 0; co < cfg.cout; co++) rows[co] = co < 9 ? co + 2 : -1;
                        w4map = 1;
                        if (g == 0) fprintf(stderr, "init-from: task unwrap donor: head rows 2-10 (phase, normal, density) -> 0-8, rows 9-10 (winding mod 4) fresh\n");
                    }
                }
                if (hm) { int k = 0; char *t = strdup(hm); for (char *q = strtok(t, ","); q && k < 16; q = strtok(nullptr, ",")) rows[k++] = atoi(q); free(t);
                    if (k != cfg.cout) { fprintf(stderr, "--head-map has %d rows, the network %d outputs\n", k, cfg.cout); return 2; } }
                const int nt = g_verso || hm || w4map ? unet_init_from_map(d->u, init_from, rows) : unet_init_from(d->u, init_from);
                if (nt < 0) { fprintf(stderr, "cannot initialise from %s\n", init_from); return 1; }
                if (g == 0) fprintf(stderr, "initialised %d layer tensors from %s (the rest fresh)\n", nt, init_from);
            }
            if (g_sheet && unet_start_sheet(d->u)) return 2;
        }
        if (wq) unet_set_wq(d->u, wq);
        d->teacher = nullptr; d->vsoft = d->vanch = nullptr; d->vmask = d->vw = nullptr; d->vscratch = nullptr; d->verso_loss = 0;
        if (g_verso && !g_verso_self) {   /* frozen teacher: --verso-teacher-ckpt CKPT (warm-start mapping as --init-from), else the starting weights */
            const size_t np = unet_nparams(d->u); d->teacher = nn_malloc(np * 4);
            const char *tp = opt(argc, argv, "--verso-teacher-ckpt", nullptr);
            if (tp) {
                unet *tu = unet_create(&cfg); unet_cfg tc; int ts;
                if (unet_peek(tp, &tc, &ts)) { fprintf(stderr, "cannot read %s\n", tp); return 1; }
                int trows[16]; trows[0] = 0; trows[1] = 0;
                for (int co = 2; co < cfg.cout; co++) trows[co] = tc.cout == cfg.cout ? co : co - 1 < tc.cout ? co - 1 : -1;
                if (tc.cout == cfg.cout) for (int co = 0; co < cfg.cout; co++) trows[co] = co;
                if (unet_init_from_map(tu, tp, trows) < 0) { fprintf(stderr, "cannot load the verso teacher %s\n", tp); return 1; }
                nn_d2d(d->teacher, unet_live_params(tu), np * 4); unet_free(tu);
                if (g == 0) fprintf(stderr, "verso: frozen teacher from %s\n", tp);
            } else nn_d2d(d->teacher, unet_live_params(d->u), np * 4);
        }
        if (split) { unet_set_split(d->u, g, g_h0, split_halo); if (!getenv("UFSM_SPLIT_SYNC")) unet_set_split_async(d->u, split_halo_begin, split_halo_end); }
        const char *e = nn_check(); if (e) { fprintf(stderr, "GPU %d: %s\n", d->dev, e); return 1; }
    }
    int lean = getenv("UFSM_LEAN") ? atoi(getenv("UFSM_LEAN")) : 0;   /* lean: one device batch buffer (no upload overlap), the logit
                                                                     gradient in the model's gradient buffer, logits in A */
    if (resume) fprintf(stderr, "resumed %s at step %d\n", resume, step0);
    if (plan) {
        if (!has_cover) coverage.base_step = step0;
        if ((uint64_t)coverage.base_step + coverage.count > INT_MAX || step0 != coverage.base_step + (int)coverage.cursor) { fprintf(stderr, "cover checkpoint step/cursor mismatch\n"); return 2; }
        steps = coverage.base_step + (int)coverage.count; schedule_start = coverage.base_step;
        if (coverage.cursor == coverage.count) { fprintf(stderr, "cover already complete\n"); return 2; }
        fprintf(stderr, "finite cover: %llu/%llu committed tiles, SHA256 %s, final step %d\n", (unsigned long long)coverage.cursor, (unsigned long long)coverage.count, coverage.sha256, steps);
    }
    if (schedule_start > step0) { fprintf(stderr, "schedule-start cannot follow the resumed step\n"); return 2; }
    if (stop_at && (stop_at <= step0 || stop_at > steps)) { fprintf(stderr, "stop-at must follow the resumed step and not exceed the final step\n"); return 2; }
    if (sheet_init && !schedule_start) schedule_start=step0;
    if (schedule_start>=steps || (sheet_ck && schedule_start!=saved_sheet_start)) {
        fprintf(stderr,"invalid or changed winding schedule origin\n"); return 2;
    }
    {   /* --mem auto (default): the cheapest storage mode whose training buffers fit next to what is already allocated on every
           GPU; an explicit UFSM_CHUNK_UP / UFSM_RECOMPUTE / UFSM_GRAD_MX8 or --mem default keeps the env / built-in modes */
        /* the logit gradient in place over the fp16 logits: every loss's gradient pass reads an element's logit (and stats of an earlier
           pass) before writing its gradient there; not with readers of other channels' logits after a loss (count loss, cond accuracy),
           the tolerance band, affinities, half-resolution or sheet nets. -22 B per voxel at 11 outputs. UFSM_GL_SEPARATE=1: own buffer */
        g_gl_inplace = !g_restore && (!g_half || g_pkf) && !g_sheet && !g_aff && !g_cond && g_count_lambda == 0.f && !nn_get_loss_tol() && g_g16 && !ufsm_env_on("UFSM_LOGITS32") && !ufsm_env_on("UFSM_GL_SEPARATE");
        /* head slabs (head_slab_pass): the production unwrap layout (11 outputs, every channel trained by one of its four losses), B 1,
           whole windows, MX activations; -22 B (logits) - 11 B (derived targets) per voxel. UFSM_HEAD_SLAB=0: whole-window head and losses;
           UFSM_HEAD_SLAB_Z: planes per slab (32) */
        g_head_slab = g_gl_inplace && g_unwrap && B == 1 && !split && !g_verso && !g_pap_head && g_side == 2 && g_side_nc == 2
                      && (g_unwrap2 ? g_side_ch == 0 && g_norm_ch == 2 && g_w4_ch == 9 : g_side_ch == 2 && g_norm_ch == 4) && g_cout == 11 && (unet_act_mx() || nn_get_act_bf16()) && cfg.widths[0] == 32 && !(getenv("UFSM_HEAD_SLAB") && !atoi(getenv("UFSM_HEAD_SLAB")));   /* (the fused
                                                                                                     slab head backward takes exactly 32 inputs) */
        g_tm_unused = g_unwrap2 && g_head_slab && !g_ds && !ufsm_env_on("UFSM_TM_KEEP");   /* (the unwrap2 head slabs read only the side / winding planes) */
        g_hs_z = getenv("UFSM_HEAD_SLAB_Z") ? atoi(getenv("UFSM_HEAD_SLAB_Z")) : 32;
        if (g_hs_z < 1) g_hs_z = 1;
        if (g_hs_z > g_Pz) g_hs_z = g_Pz;
        for (int g = 0; g < ng; g++) unet_set_head_ext(G[g].u, g_head_slab);
        if (g_pg_lambda > 0 && !g_head_slab) { fprintf(stderr, "--pgrad-lambda needs the head-slab loss path (task unwrap2, B 1, no --split / --verso)\n"); return 2; }
        if (g_valP == P && !getenv("UFSM_VALP_FORCE")) g_valP = 0;   /* (UFSM_VALP_FORCE=1: the separate validation network at the training size, a test) */
        if (g_valP) {   /* --val-P: the plain recto-style losses on whole cube windows; validation on GPU 0 */
            if (g_valP < 0 || g_valP > P || g_valP % (1 << (cfg.nlev - 1))) { fprintf(stderr, "--val-P: a positive multiple of %d, at most P (the windows go up through "
                                                                                       "the training batch buffers)\n", 1 << (cfg.nlev - 1)); return 2; }
            if (split || g_Pz != P || g_half || g_aff || g_side || g_unwrap || g_heads || g_ds || g_cond || g_pap_head || g_verso || g_head_slab || g_sheet || g_restore
                || cfg.l0_norm == 2 || nn_get_loss_tol()) {
                fprintf(stderr, "--val-P: whole cube windows and the plain losses only (no --split, --Pz, --half-res, heads, deep supervision, verso, "
                                "side / unwrap / restore / cond / papyrus / band tasks, --tol, --l0-norm ema)\n"); return 2;
            }
            nn_init(G[0].dev);
            g_val_u = unet_create(&cfg);   /* its own buffers (an inference build at N^3); the weights are the training net's (unet_use_params) */
            unet_set_fp16_logits(g_val_u, !ufsm_env_on("UFSM_LOGITS32"));   /* (as the training nets) */
        }
        const char *mm = opt(argc, argv, "--mem", "auto");   /* MX-fp8 gradients passed their stair (3 seeds, mean 0.293 vs 0.294) */
        const int auto16 = g_sheet || !strcmp(mm, "auto16") || !unet_act_mx();   /* sheet reference keeps FP16 stem, hence 16-bit gradients */
        /* MX gradient modes store MX-fp4 activation gradients by default (paired 500-step runs: same step time, -2.9 GB per desk GPU,
           validation loss +0.1..0.8%) where the fp4 kernels apply: widths in 32-channel blocks, an even coarsest level;
           UFSM_GRAD_MX4=0 keeps MX-fp8 */
        {
            /* any P the net takes (a multiple of 2^(nlev-1): P 640 / 704 with 7 levels); the stride-2 weight gradient's gy scales no
               longer need the coarsest width a multiple of 4 */
            int g4ok = P % (1 << (cfg.nlev - 1)) == 0 && g_Pz % (1 << (cfg.nlev - 1)) == 0 && fp4 >= 2 && !*opt(argc, argv, "--policy", "");   /* the fp8 weight gradients of --fp4 1
                                                                                                     (or a --policy) take no MX-fp4 gy */
            /* 16- and 8-channel levels too (paired 2000-step desk check, production recipe: val 0.6506 vs 0.6534 MX-fp8, +3% speed, 5.8 vs 9.7 GB) */
            const int g4n = getenv("UFSM_G4_NARROW") ? ufsm_env_on("UFSM_G4_NARROW") : 1;   /* default on since the paired check (2026-10-15); =0 keeps MX-fp8 there */
            for (int i = 0; i < cfg.nlev; i++) if (cfg.widths[i] % 32 && !(g4n && (cfg.widths[i] == 16 || cfg.widths[i] == 8))) g4ok = 0;
            for (int i = 0; i < cfg.nlev - 1; i++) if (cfg.dec_widths[i] > 0 && cfg.dec_widths[i] % 32 && !(g4n && (cfg.dec_widths[i] == 16 || cfg.dec_widths[i] == 8))) g4ok = 0;
            const int g4 = getenv("UFSM_GRAD_MX4") ? ufsm_env_on("UFSM_GRAD_MX4") : g4ok;
            if (g4 && !g4ok) { fprintf(stderr, "UFSM_GRAD_MX4: MX-fp4 gradients need --fp4 2 (fp4 weight gradients), widths that are multiples of 32 and P divisible by %d\n", 1 << (cfg.nlev - 1)); return 2; }
            unet_set_grad_mx4(g4);
            /* shared encoder a1 (recompute 1) with its host offload by default: ~0.6% step, -22 B per level-0 voxel; UFSM_RC_ENC_A1=0 off */
            unet_set_share_enc_a1(getenv("UFSM_RC_ENC_A1") ? ufsm_env_on("UFSM_RC_ENC_A1") : 1);
        }
        if (!strcmp(mm, "wide")) {
            if (!nn_get_tf32() || !unet_act_mx()) { fprintf(stderr, "--mem wide requires tensor cores and MX activation storage\n"); return 2; }
            unet_set_chunk_up(2); unet_set_recompute(1); unet_set_grad_mx8(1); unet_set_lean(2); unet_set_wide_up_grad(1);
            lean = 2;
            fprintf(stderr, "memory: MX-fp8 gradients, chunked, lean 2, wide finest-level up gradient (explicit mode)\n");
            { char mp[1400]; snprintf(mp, sizeof mp, "%s/precision.txt", out); WRITE_MANIFEST(mp); }
        } else if ((!strcmp(mm, "auto") || !strcmp(mm, "fast") || auto16) && nn_get_tf32() && !getenv("UFSM_CHUNK_UP") && !getenv("UFSM_RECOMPUTE") && !getenv("UFSM_GRAD_MX8")) {
            static const struct { int chunk, rc, gmx, lean, sea; const char *what; int l3; } cand[] = {
                /* by step cost (MX-fp8 gradients passed their 3-seed stair and cost nothing; lean costs ~nothing, the upload is
                   ~3% of a large-window step; recompute 2 ~10%), the 16-bit-gradient modes only for --mem auto16 / 16-bit activations.
                   Model bytes per level-0 voxel / step at 96^3 B2 (--fp4 1): MX-fp8 164 / 26.2 ms, + chunked 148 / 27.2, + recompute 2
                   133 / 29.8; 16-bit default 219 / 26.4, chunked 186 / 27.4, + recompute 2 171 / 29.8. lean 1: one batch buffer, logits
                   in A, logit gradient / targets / mask (/ input) in B; lean 2: no gradient buffer B (each block's incoming gradient
                   buffer serves as B), -17 B */
                {1, 1, 1, 0, 0, "MX gradients"}, {2, 1, 1, 0, 0, "MX gradients, chunked up-part gradient"},
                {2, 1, 1, 1, 0, "MX gradients, chunked, lean"}, {2, 1, 1, 2, 0, "MX gradients, chunked, lean 2"},
                {2, 1, 1, 2, 1, "MX gradients, chunked, lean 2, shared encoder a1"},   /* ~-22 B / level-0 voxel, ~2.5% (encoder conv1 re-run) */
                {2, 1, 1, 2, 1, "MX gradients, chunked, lean 2, shared encoder a1, three level-0 buffers", 1},   /* (unet.l3b: 8- / 16-channel level 0)
                                                                                     -9 B / level-0 voxel at 16 channels, +0.8% step (P512, laptop) */
                {2, 2, 1, 0, 0, "MX gradients, chunked, recompute 2"}, {2, 2, 1, 1, 0, "MX gradients, chunked, recompute 2, lean"},
                {2, 2, 1, 2, 0, "MX gradients, chunked, recompute 2, lean 2"},
                {1, 1, 0, 0, 0, "16-bit gradients"}, {2, 1, 0, 0, 0, "16-bit gradients, chunked up-part gradient (UFSM_CHUNK_UP=2)"},
                {2, 2, 0, 0, 0, "16-bit gradients, chunked, recompute 2"}, {2, 2, 0, 1, 0, "16-bit gradients, chunked, recompute 2, lean"},
                {2, 2, 0, 2, 0, "16-bit gradients, chunked, recompute 2, lean 2"}};
            /* shared encoder a1 in every mode where it applies (recompute 1): with its host offload ~0.6% step for -22 B per
               level-0 voxel (desk slab -1.66 GB per GPU); default on, UFSM_RC_ENC_A1=0 off */
            const int sea_force = getenv("UFSM_RC_ENC_A1") ? ufsm_env_on("UFSM_RC_ENC_A1") : 1;
            const int nc = (int)(sizeof cand / sizeof cand[0]);
            /* the search order: --mem auto takes the smallest recompute-1 mode that fits (lean 2, chunked, shared a1: production
               P 384 12.75 -> 10.47 GB per GPU for ~2% throughput, the serialised batch upload; the user: RAM savings by default unless
               they cost massive speed), then the larger recompute-1 ones, then recompute 2 (~10% step for ~0.3 GB more); --mem fast
               keeps the list order (fastest first) and is PCIe-aware: it first tries every mode with no pinned host offloads (encoder
               a1, input, level-0 skip / A share, down outputs, coarse skips: ~2 x 19 B per level-0 voxel over PCIe each step with the
               production net, the bound from ~800 Mvox/s) and no shared encoder a1 (unless the mode is one), then the modes as --mem auto
               builds them */
            static const int ord_small[] = {5, 4, 3, 2, 1, 0, 8, 7, 6, 9, 10, 11, 12, 13};   /* (16-bit modes: list order, no recompute-1 lean ones) */
            const int l3_off = getenv("UFSM_L0_3BUF") && !ufsm_env_on("UFSM_L0_3BUF");   /* (UFSM_L0_3BUF=0: never the three level-0 buffers) */
            const int fastest = !strcmp(mm, "fast");
            size_t fmin = (size_t)-1;
            for (int g = 0; g < ng; g++) { nn_init(G[g].dev); size_t f = nn_mem_free(); if (f < fmin) fmin = f; }
            nn_init(G[0].dev);
            if (getenv("UFSM_MEM_FREE_GB") && atof(getenv("UFSM_MEM_FREE_GB")) > 0) {   /* plan for another card's free memory (UFSM_MEM_DRY sizing for a busy GPU) */
                fmin = (size_t)(atof(getenv("UFSM_MEM_FREE_GB")) * 1e9); fprintf(stderr, "memory: UFSM_MEM_FREE_GB: planning for %.2f GB free\n", fmin / 1e9);
            }
            if (g_val_u) {   /* --val-P: the validation copy's inference build at N^3 stays out of the training budget */
                g_val_reserve = unet_infer_bytes(g_val_u, (shape5){B, g_xc, g_valP, g_valP, g_valP});
                g_val_reserve += g_val_reserve * 2 / 5 + ((size_t)256 << 20);   /* (the dry build counts ~3/4 of what it takes: 1.08 GB at 384^3) */
                fmin = fmin > g_val_reserve ? fmin - g_val_reserve : 0;
                fprintf(stderr, "memory: --val-P %d keeps %.2f GB free for the validation network\n", g_valP, g_val_reserve / 1e9);
            }
            const shape5 xs = g_half ? (shape5){B, cfg.cin, P / g_hdiv, P / g_hdiv, P / g_hdiv} : (shape5){B, cfg.cin, split ? g_Dl : g_Pz, P, P};   /* split: this GPU's slab of the window */
            const size_t p3 = p3l;
            int pick = -1, pick_hoff = 1, minc = -1, min_hoff = 1; size_t need = 0, pick_off = 0, pick_need = 0, min_need = (size_t)-1;
            {   /* the batch tail [targets | mask] the three-level-0-buffer layout holds at X3's end (its dry builds size X3 for it) */
                const size_t a256 = 255;
                unet_set_batch_bytes(G[0].u, 0, (((size_t)B * g_tc * p3 + a256) & ~a256) + (((size_t)B * p3 + a256) & ~a256));
            }
            const int pick0 = getenv("UFSM_MEM_PICK") ? atoi(getenv("UFSM_MEM_PICK")) : 0;   /* A/B of the modes: start the search at this candidate */
            _Static_assert(sizeof ord_small / sizeof ord_small[0] == sizeof cand / sizeof cand[0], "every mode in the search order");
            for (int pass = fastest ? 0 : 1; pass < 2 && pick < 0; pass++)   /* pass 0 (--mem fast): no host offloads */
            for (int oi = 0; oi < nc && pick < 0; oi++) {
                const int c = pick0 > 0 && pick0 < nc ? (oi ? nc : pick0) : fastest ? oi : ord_small[oi];   /* UFSM_MEM_PICK: that mode only */
                if (c >= nc) break;
                if (cand[c].gmx == auto16) continue;   /* auto: MX gradient modes; auto16: the 16-bit ones */
                if (g_half && !g_pkf && cand[c].lean) continue;   /* (the batch buffers are full resolution, the network's half) */
                if (cand[c].l3 && (l3_off || !pass)) continue;   /* (the three level-0 buffers rely on their host copies) */
                unet_set_host_offloads(pass);
                unet_set_chunk_up(cand[c].chunk); unet_set_recompute(cand[c].rc); unet_set_grad_mx8(cand[c].gmx); unet_set_lean(cand[c].lean); unet_set_share_enc_a1(cand[c].sea || (pass && sea_force));
                unet_set_l0_3buf(cand[c].l3);
                const size_t tb = unet_train_bytes(G[0].u, xs), nbuf = cand[c].lean ? 1 : 2;
                if (cand[c].l3 && !unet_last_dry_l3b()) continue;   /* (not this net: the mode above) */
                const int gin = g_gpu_want && unet_input_converted();   /* this mode's input layout (gpu_input_settle after the pick) */
                const size_t xsc = gin ? 1 : (size_t)g_xc, radb = gin ? (size_t)B * NN_RAD_REC(g_Pz) : 0;
                size_t trainer = nbuf * ((size_t)B * xsc * p3 * xbytes() + (size_t)B * g_tc * p3 + (size_t)B * p3) + (cand[c].lean || g_gl_inplace ? 0 : (size_t)B * g_cout * p3 * (g_g16 ? 2 : 4)) + nbuf * radb;
                /* lean 2 (no buffer B): [input | targets | mask] at the end of gout[0] when they fit beside its level-0 gradient (at least w0
                   channels of the MX gradient type; the lean placement below), otherwise the input keeps a buffer of its own ("input separate":
                   8 B per voxel for the 4 fp16 channels, uncounted before) */
                const size_t xinb = (size_t)B * xsc * p3 * xbytes(), batb = (g_pkf && g_pack ? 0 : (size_t)B * g_tc * p3) + (size_t)B * p3;
                const shape5 g0s = {B, cfg.widths[0], (int)(p3 / ((size_t)P * P)), P, P};
                const size_t g0b = cand[c].gmx ? nn_mx_bytes(g0s, unet_grad_mx4() ? 4 : 8) : (size_t)B * cfg.widths[0] * p3 * 2;
                const int l3 = unet_last_dry_l3b();   /* three level-0 buffers: the tail inside X3 (counted), the input in X2 when it fits */
                const int xin_sep = g_restore || (cand[c].lean >= 2 && cand[c].gmx && (l3 ? xinb > g0b : !(unet_input_converted() && g_xfmt && xinb + batb + 8192 <= g0b)));
                if (cand[c].lean) trainer = (cand[c].gmx ? (xin_sep ? xinb : 0) : xinb) + radb   /* batch buffers inside the gradient buffer B / gout[0] (estimate) */
                                            + (g_cout > NCH && !g_gl_inplace ? (size_t)B * g_cout * p3 * (g_g16 ? 2 : 4) : 0)
                                            + (g_head_slab && !g_tm_unused ? (size_t)B * (NCH + 1) * p3 : 0);   /* head slabs: targets and mask outside the gradient buffers */   /* > 2 outputs (affinities, verso): the logit gradient may not fit beside the logits */
                if (g_pkf) trainer = nbuf * (xinb + batb + radb)   /* --half-res pack: the full-resolution batch beside the network (its buffers are
                                                                        the half grid's); the host-built input's half-grid copy */
                                     + (gin ? 0 : (size_t)B * cfg.cin * (p3 / 8) * xbytes());
                /* the fp4 weight gradient's gy records of a whole level-0 layer (lp_gn_silu_bwd_apply_rec, MX-fp4 gradients): 136 B per
                   channel x 256 voxels (~0.5 B per voxel and channel), outside the dry build. 8 / 16-channel levels only with UFSM_GYREC=2
                   (8.5 B per level-0 voxel at 16 channels for ~0 step time there; 8 channels: -2%) */
                /* deferred GroupNorm applies (UFSM_GN_DEFER, default; not with split / three level-0 buffers): the records of one ~96 MB
                   pre-pass slab at a time instead of a whole layer's (-17 B per level-0 voxel at 32 channels) */
                const int gdef = (getenv("UFSM_GN_DEFER") ? atoi(getenv("UFSM_GN_DEFER")) : 1) && !split && !cand[c].l3 && cfg.widths[0] % 32 == 0;
                const size_t grec = unet_grad_mx4() && (cfg.widths[0] % 32 == 0 || lp_gyrec_narrow()) && (getenv("UFSM_GYREC") ? atoi(getenv("UFSM_GYREC")) : 1)
                                    ? (gdef ? (size_t)100 << 20 : (size_t)B * cfg.widths[0] * p3 * 136 / 256) : 0;
                need = grec + tb + tb / 14 + trainer + (nn_get_loss_tol() ? (size_t)B * p3 : 0) + (g_aff ? (size_t)2 * B * p3 : 0) + (g_unwrap ? (g_head_slab ? (size_t)4 * B * p3 + (size_t)g_hs_z * P * P * (2 * g_cout + 11) + 4096 : (size_t)15 * B * p3) + (g_unwrap2 ? (size_t)4 * B * p3 : 0) : 0) + (g_cond ? (size_t)7 * B * p3 : 0) + (g_pap_head ? (size_t)4 * B * p3 : 0) + (g_restore ? (size_t)B * p3 * (cand[c].lean ? 5 : 1) + restore_scratch_bytes(g_Pz, P, P, fminf(g_rr.sc_step[0], 4.f)) + 256 : 0) + ((size_t)550 << 20) + (split ? (size_t)4 * B * 32 * P * P * 2 : 0) + (g_gpu_aug_want && gin ? nn_aug_device_bytes(P) + (g_gpu_aug_want == 2 ? nn_label_warp_device_bytes(P) : 0) : 0);   /* split: two slots of halo send / receive planes */   /* kernel workspaces: measured device growth / tracked ~1.07 + 0.3 GB, + 0.2 GB margin for other processes */
                if (ufsm_env_on("UFSM_MEM_REPORT"))   /* the planner's terms of this mode */
                    fprintf(stderr, "memory plan: mode %d (%s%s): network %.2f + workspace %.2f + gy records %.2f + trainer %.2f + task %.2f + fixed %.2f = %.2f GB\n", c, cand[c].what,
                            pass ? "" : ", no host offloads", tb / 1e9, tb / 14 / 1e9, grec / 1e9, trainer / 1e9, (need - grec - tb - tb / 14 - trainer - ((size_t)550 << 20)) / 1e9, (double)((size_t)550 << 20) / 1e9, need / 1e9);
                if (need < min_need) { min_need = need; minc = c; min_hoff = pass; }   /* (no fit: the smallest mode evaluated) */
                if (need <= fmin) { pick = c; pick_hoff = pass; pick_off = unet_last_offload_bytes(); pick_need = need; }
            }
            if (pick < 0) { pick = minc >= 0 ? minc : auto16 ? nc - 1 : 8; pick_hoff = minc >= 0 ? min_hoff : 1; fprintf(stderr, "memory: no mode fits %.2f GB free (smallest needs %.2f GB); trying %s\n", fmin / 1e9, (minc >= 0 ? min_need : need) / 1e9, cand[pick].what); }
            else {   /* the mode, its device bytes per level-0 voxel of the window and its PCIe traffic per step */
                const size_t vox = (size_t)B * p3, upb = (size_t)B * (g_gpu_want && unet_input_converted() ? 1 : (size_t)g_xc) * p3 * xbytes()
                                   + (g_pack ? (size_t)B * p3 : (size_t)B * (g_tc + 1) * p3) + (g_gpu_want ? (size_t)B * NN_RAD_REC(g_Pz) : 0);
                fprintf(stderr, "memory: %s%s, %.2f of %.2f GB free per GPU (%.1f B per voxel); per step %.2f GB batch upload, %.2f GB host offloads "
                        "each way (%.1f B per voxel%s)\n", cand[pick].what, fastest ? (pick_hoff ? " (--mem fast: with host offloads)" : " (--mem fast: no host offloads)") : "",
                        pick_need / 1e9, fmin / 1e9, (double)pick_need / (double)vox, upb / 1e9, pick_off / 1e9, (double)pick_off / (double)vox,
                        fastest || !pick_off ? "" : "; --mem fast skips them where the window fits without");
            }
            unet_set_host_offloads(pick_hoff);
            unet_set_chunk_up(cand[pick].chunk); unet_set_recompute(cand[pick].rc); unet_set_grad_mx8(cand[pick].gmx); unet_set_lean(cand[pick].lean); unet_set_share_enc_a1(cand[pick].sea || (pick_hoff && sea_force));
            unet_set_l0_3buf(cand[pick].l3);
            if (unet_share_enc_a1()) fprintf(stderr, "memory: shared encoder a1, %s (default; UFSM_RC_ENC_A1=0 off)\n", unet_a1_offload() ? "offloaded to pinned host memory between its uses" : "conv1 re-run in the backward");
            if (unet_grad_mx4()) fprintf(stderr, "memory: MX gradients stored as MX-fp4 (default; UFSM_GRAD_MX4=0: MX-fp8)\n");
            lean = cand[pick].lean;
            { char mp[1400]; snprintf(mp, sizeof mp, "%s/precision.txt", out); WRITE_MANIFEST(mp); }   /* the planner's modes */
            if (ufsm_env_on("UFSM_MEM_DRY")) { fprintf(stderr, "memory: UFSM_MEM_DRY, exiting after the planner\n"); return 0; }   /* footprints in seconds (with UFSM_MEM_REPORT=1) */
        }
    }
    if (g_half && !g_pkf && lean) { fprintf(stderr, "--half-res: not with the lean memory modes\n"); return 2; }
    gpu_input_settle(&sc);   /* (the memory mode is set: the input layout of the batches) */
    if (newdef && !workers_asked && g_gpu_aug == 2) { workers = 3; sc.nworkers = workers; fprintf(stderr, "sampler: 3 workers (the new-run default with --gpu-aug 2; --workers N sets it)\n"); }
    for (int g = 0; g < ng; g++) {   /* per-GPU batch buffers. lean: one, shared by both slots, inside the model's gradient buffers when
                                        they are large enough: [targets | mask | input if the forward copies it] in B (lean 1: after the
                                        logit gradient) or in gout[0] (lean 2); the logit gradient in B (lean 1) or after the logits in A */
        gpu_state *d = &G[g];
        nn_init(d->dev);
        char *bs = nullptr;
        const size_t p3 = p3l;   /* this GPU's voxels of a window */
        const size_t a256 = 255, glb = ((size_t)B * g_cout * p3 * (g_g16 ? 2 : 4) + a256) & ~a256, tbb = ((size_t)B * g_tc * p3 + a256) & ~a256,
                     mbb = ((size_t)B * p3 + a256) & ~a256, xbb = ((size_t)B * g_xs * p3 * xbytes() + a256) & ~a256;
        const int xin_b = lean && unet_input_converted() && g_xfmt && !g_restore;   /* (restore: the device-built input in its own buffer) */
        char *tail = nullptr;   /* three level-0 buffers: [targets | mask] apart from the input */
        size_t goff = 0;
        int xin_in = 0;
        d->gl = nullptr; d->gl_inside = 0; d->gl_inplace = g_gl_inplace;
        if (lean && !g_restore) {   /* (restore: target, mask, input and logit gradient in their own buffers: the MX chunk transient of a
                                       16-channel level 0 does not fit beside them in the gradient buffer) */
            unet_set_batch_bytes(d->u, xin_b ? xbb : 0, tbb + mbb);
            unet_build(d->u, g_half ? (shape5){B, cfg.cin, P / g_hdiv, P / g_hdiv, P / g_hdiv} : (shape5){B, cfg.cin, split ? g_Dl : g_Pz, P, P}, 1);
            goff = unet_lean_nob(d->u) ? 0 : glb;
            void *l3t = nullptr;
            if (unet_l0_3buf(d->u) && (bs = unet_batch_scratch2(d->u, xin_b ? xbb : 0, tbb + mbb, &l3t))) {   /* three level-0 buffers: the input in X2,
                                                                                                         [targets | mask] at X3's end */
                xin_in = xin_b; tail = (char *)l3t;
            } else if (unet_lean_nob(d->u)) {   /* lean 2: [input | targets | mask] (the input may lie over the logits, see unet_batch_scratch) */
                if (xin_b && (bs = unet_batch_scratch(d->u, xbb, tbb + mbb))) xin_in = 1;
                else bs = unet_batch_scratch(d->u, 0, tbb + mbb);
            } else if (xin_b && (bs = unet_grad_scratch(d->u, goff + tbb + mbb + xbb))) xin_in = 1;
            else bs = unet_grad_scratch(d->u, goff + tbb + mbb);
            if (d->gl_inplace && !unet_logits_h16(d->u)) {   /* the layout kept fp32 logits (narrow nets: A holds them and a 16-bit gradient) */
                d->gl_inplace = 0;
                if (g == 0) fprintf(stderr, "logit gradient: fp32 logits in this layout, gradient beside them instead of in place\n");
            }
            d->gl = d->gl_inplace ? nullptr : unet_logit_grad_scratch(d->u, glb);
            if (d->gl && bs && (char *)d->gl < bs + goff + tbb + mbb + (xin_in ? xbb : 0) && (char *)d->gl + glb > bs) d->gl = nullptr;   /* never over the batch */
            d->bat_lo = bs; d->bat_hi = bs ? bs + goff + tbb + mbb + (xin_in ? xbb : 0) : nullptr;
            if (tail) d->bat_lo = d->bat_hi = nullptr;   /* (three level-0 buffers: no logit gradient scratch to keep off it) */
            d->gl_inside = d->gl != nullptr;
        }
        for (int i = 0; i < 2; i++) {
            if (i && lean) { d->xb[1] = d->xb[0]; d->tb[1] = d->tb[0]; d->mb[1] = d->mb[0]; d->wb[1] = d->wb[0]; d->radb[1] = d->radb[0]; }
            else if (bs) {
                const size_t toff = goff + (unet_lean_nob(d->u) && xin_in ? xbb : 0), xoff = unet_lean_nob(d->u) ? 0 : goff + tbb + mbb;
                char *tp = tail ? tail : bs + toff;
                d->tb[i] = (uint8_t *)tp; d->mb[i] = (uint8_t *)(tp + tbb);
                d->xb[i] = xin_in ? (void *)(bs + xoff) : nn_malloc((size_t)B * g_xs * p3 * xbytes());
                d->wb[i] = nn_malloc((size_t)B * g_tc);
                d->radb[i] = g_gpu_in ? nn_malloc(rad_bytes(B)) : nullptr;
            }
            else { d->xb[i] = nn_malloc((size_t)B * g_xs * p3 * xbytes()); d->tb[i] = nn_malloc(g_pkf && g_pack ? 256 : (size_t)B * g_tc * p3);   /* (the direct pack
                                                                                                       loss reads the packed codes in mb) */ d->mb[i] = nn_malloc((size_t)B * p3); d->wb[i] = nn_malloc((size_t)B * g_tc); d->radb[i] = g_gpu_in ? nn_malloc(rad_bytes(B)) : nullptr; }
            d->hwh[i] = g_heads ? nn_host_alloc((size_t)B * g_tc) : nullptr;
            d->ev_up[i] = nn_event_create();
        }
        if (g_head_slab && bs) {   /* head slabs read the targets and mask in the backward (head_slab_cb), after the head's backward has
                                      started writing the gradient buffers: their own buffers (+3 B per voxel), the input stays inside */
            const size_t tb1 = g_tm_unused ? 256 : tbb, mb1 = g_tm_unused ? 256 : mbb;   /* (unwrap2: no recto target / mask is read) */
            d->tb[0] = d->tb[1] = nn_malloc(tb1); d->mb[0] = d->mb[1] = nn_malloc(mb1);
            if (!lean) { d->tb[1] = nn_malloc(tb1); d->mb[1] = nn_malloc(mb1); }
        }
        if (lean && g == 0) fprintf(stderr, "lean %d: batch buffers %s, logit gradient %s\n", lean, !bs ? "separate" : tail ? (xin_in ? "in the three level-0 buffers (input in X2, targets / mask at X3's end)" : "at X3's end (three level-0 buffers), input separate")
                                       : xin_in ? "and input inside the gradient buffers" : "inside the gradient buffers, input separate", tail && d->gl_inplace ? "over the logits" : d->gl ? "inside" : "separate");
        for (int i = 0; i < 2; i++) { d->ev_done[i] = nn_event_create(); nn_event_record(d->ev_done[i], 0); }
        d->pending = nullptr; select_buf(d, 0);
        if (!d->gl && !d->gl_inplace) d->gl = nn_malloc((size_t)B * g_cout * p3 * (g_g16 ? 2 : 4));
        if (g_distill) {   /* --distill: int8 targets (double-buffered unless lean), the head's fp32 gradient, the loss sums */
            d->embb[0] = nn_malloc(g_distill_n); d->embb[1] = lean ? d->embb[0] : nn_malloc(g_distill_n); d->embt = d->embb[d->cur];
            d->embg = nn_malloc(g_distill_n * 4); d->embds = nn_malloc(2 * sizeof(double));
        }
        for (int j = 1; j <= g_ds; j++) {   /* deep supervision: pooled targets / mask, fp32 logit gradient and loss scratch per level */
            const shape5 sj = {B, g_cout, g_Pz >> j, P >> j, P >> j}; const size_t sv = (size_t)sj.d * sj.h * sj.w;
            d->ds_t[j] = nn_malloc((size_t)B * g_tc * sv); d->ds_m[j] = nn_malloc((size_t)B * sv);
            d->ds_gl[j] = nn_malloc(shape_numel(sj) * 4); d->ds_scr[j] = nn_malloc(nn_loss_scratch(sj));
        }
        if (g_pkf) {   /* (the direct pack path: the host-built input's half-grid copy only; logits and gradient on the network's grid) */
            d->x2 = g_gpu_in ? nullptr : nn_malloc((size_t)B * cfg.cin * (p3 / 8) * xbytes());
            d->lgf = nullptr; d->glh = nullptr; d->lrt = nullptr;
        } else if (g_half) {
            const size_t q3 = p3 / ((size_t)g_hdiv * g_hdiv * g_hdiv), t1 = (size_t)B * (p3 / 8) * (4 * xbytes() > 4 * (size_t)g_cout ? 4 * xbytes() : 4 * (size_t)g_cout);
            d->x2 = nn_malloc((size_t)B * cfg.cin * q3 * xbytes()); d->lgf = nn_malloc((size_t)B * g_cout * p3 * 4);
            d->glh = nn_malloc((size_t)B * cfg.cout * q3 * (g_g16 ? 2 : 4)); d->lrt = g_hdiv > 2 ? nn_malloc(t1) : nullptr;
        }
        if (g_aff) {
            d->bandb[0] = nn_malloc((size_t)B * p3); d->bandb[1] = lean ? d->bandb[0] : nn_malloc((size_t)B * p3);
            d->aff_fin = nn_malloc(nn_aff_scratch(g_aff_off.K)); d->band = d->bandb[d->cur];
        }
        if (g_side) {
            d->sideb[0] = nn_malloc((size_t)B * p3); d->sideb[1] = lean ? d->sideb[0] : nn_malloc((size_t)B * p3); d->sdb = d->sideb[d->cur];
            const size_t tp3 = g_head_slab ? (size_t)g_hs_z * P * P : (size_t)B * p3;   /* derived targets: one head slab, or the window */
            d->sidet = nn_malloc((size_t)g_side_nc * tp3); d->sidem = nn_malloc(tp3);
            d->sidew = nn_malloc(2); static const uint8_t one_w[2] = {1, 1}; nn_h2d(d->sidew, one_w, 2);
            d->sscratch = nn_malloc(nn_loss_scratch((shape5){1, g_side_nc, g_Pz, P, P}) + 64);
        }
        if (g_unwrap) {
            if (pm_used()) { d->pmb[0] = nn_malloc((size_t)B * p3); d->pmb[1] = lean ? d->pmb[0] : nn_malloc((size_t)B * p3); }   /* (unwrap2: no ink plane) */ d->pm = d->pmb[d->cur];
            const size_t tp3 = g_head_slab ? (size_t)g_hs_z * P * P : (size_t)B * p3;
            d->nt = nn_malloc((size_t)7 * tp3); d->nm = nn_malloc(tp3);
            if (g_head_slab) {
                d->hs_z = g_hs_z; d->hs_P = P;
                d->hs_lg = nn_malloc((size_t)g_cout * tp3 * 2); d->hs_ds = nn_malloc(64 * sizeof(double));
                if (g_pg_lambda > 0) { d->pg_acc = nn_malloc(2 * tp3 * sizeof(float)); nn_zero(d->pg_acc, 2 * tp3 * sizeof(float)); }
                unet_set_head_cb(d->u, head_slab_cb, d);
                if (g == 0) fprintf(stderr, "head slabs: %d planes per slab (%.0f MB of logits and targets instead of %.2f GB)\n", g_hs_z,
                                    (double)tp3 * (2 * g_cout + 11) / 1e6, (double)B * p3 * (2 * g_cout + 11) / 1e9);
            }
            d->nw = nn_malloc(7); static const uint8_t seven_w[7] = {1, 1, 1, 1, 1, 1, 1}; nn_h2d(d->nw, seven_w, 7);
            d->iscratch = nn_malloc(nn_loss_scratch((shape5){1, 1, g_Pz, P, P}) + 64);
            d->nscratch = nn_malloc(nn_loss_scratch((shape5){1, 7, g_Pz, P, P}) + 64);
            if (g_unwrap2) {
                d->w4b[0] = nn_malloc((size_t)B * p3); d->w4b[1] = lean ? d->w4b[0] : nn_malloc((size_t)B * p3); d->w4 = d->w4b[d->cur];
                d->wcb[0] = nn_malloc((size_t)B * p3); d->wcb[1] = lean ? d->wcb[0] : nn_malloc((size_t)B * p3); d->wc = d->wcb[d->cur];
                d->w4ds = nn_malloc(nn_wind4_sums()); d->w4fin = nn_malloc(nn_wind4_scratch() + 64);
            }
            if (g_count_lambda > 0) {
                d->cnt_acc = nn_malloc(p3 * sizeof(float)); nn_zero(d->cnt_acc, p3 * sizeof(float));
                d->cnt_seg = nn_malloc((size_t)3 * g_count_K * sizeof(float)); d->cnt_nv = nn_malloc(sizeof(unsigned)); d->cnt_out = nn_malloc(4 * sizeof(float));
            }
        }
        if (g_cond) {
            d->wdb[0] = nn_malloc((size_t)B * p3); d->wdb[1] = lean ? d->wdb[0] : nn_malloc((size_t)B * p3); d->wd = d->wdb[d->cur];
            d->ct4 = nn_malloc((size_t)4 * B * p3); d->cm4 = nn_malloc((size_t)B * p3);
            d->cw4 = nn_malloc(4); static const uint8_t four_w[4] = {1, 1, 1, 1}; nn_h2d(d->cw4, four_w, 4);
            d->cscratch = nn_malloc(nn_loss_scratch((shape5){1, 4, g_Pz, P, P}) + 64);
            d->cacc = nn_malloc(2 * sizeof(unsigned long long));
        }
        if (g_pap_head) {
            d->papb[0] = nn_malloc((size_t)B * p3); d->papb[1] = lean ? d->papb[0] : nn_malloc((size_t)B * p3); d->pap = d->papb[d->cur];
            d->pt1 = nn_malloc((size_t)B * p3); d->pmk1 = nn_malloc((size_t)B * p3);
            d->pw1 = nn_malloc(1); static const uint8_t one_w1 = 1; nn_h2d(d->pw1, &one_w1, 1);
            d->pscr = nn_malloc(nn_loss_scratch((shape5){1, 1, g_Pz, P, P}) + 64);
        }
        d->scratch = nn_malloc(nn_loss_scratch((shape5){B, g_tc, g_Pz, P, P}) + 64);
        if (g_restore) { d->rlow = nn_malloc(p3); d->rscr_b = restore_scratch_bytes(g_Pz, P, P, fminf(g_rr.sc_step[0], 4.f)); d->rscr = nn_malloc(d->rscr_b); d->rst = nn_malloc(RP_NST * sizeof(double)); d->rrd = g_rpairs ? nn_malloc((size_t)(P / 4) * (P / 4) * (P / 4) * sizeof(float) + 256) : nullptr; }
        d->tolc = nn_get_loss_tol() ? nn_malloc((size_t)B * p3) : nullptr;
        d->lean = lean;
        d->upsplit = lean && tail && !split && !g_restore && d->side < 0 && !g_head_slab && !g_aff && !g_side && !g_unwrap && !g_cond && !g_pap_head
                     && (!getenv("UFSM_UPSPLIT") || ufsm_env_on("UFSM_UPSPLIT")) && unet_l0_3buf_head_free(d->u);   /* (env UFSM_UPSPLIT=0 off) */
        if (d->upsplit && g == 0) fprintf(stderr, "batch upload: the head during the stem's weight gradient, the tail after the loss fetch\n");
        /* lean head slabs (the production unwrap layout): the per-voxel target planes outside the gradient buffers are read only by the
           head slabs, so the next batch's go up during the backward; the input and packed targets in gout[0] after the loss fetch, with
           the optimiser (the whole upload used to sit on the step's end, P512 laptop ~35 ms, and the loss fetch's small copies waited
           behind it: ~20 ms of idle GPU more). UFSM_HSPLIT=0 off */
        d->hsplit = lean && g_head_slab && !d->upsplit && !split && !g_restore && d->side < 0 && !g_aff && !g_cond && !g_pap_head && !g_distill && !g_heads
                    && (!getenv("UFSM_HSPLIT") || ufsm_env_on("UFSM_HSPLIT"));
        if (d->hsplit) { d->ev_hs = nn_event_create(); nn_event_record(d->ev_hs, 0); if (g == 0) fprintf(stderr, "batch upload: target planes during the backward, input and packed targets after the loss fetch\n"); }
        const char *e = nn_check(); if (e) { fprintf(stderr, "GPU %d: %s\n", d->dev, e); return 1; }
    }
    nn_init(G[0].dev);
    size_t np = unet_nparams(G[0].u);
    checkpoint_runtime runtime = {.version = 1, .train_window = P, .prec = nn_get_prec(), .f16 = nn_get_f16(),
        .act_mx4 = unet_act_mx4(), .act_mx8 = unet_act_mx() && !unet_act_mx4(), .grad_mx8 = unet_grad_mx8(), .input_mx = unet_input_converted(), .input_prec = unet_input_prec(), .gn_stored = nn_get_gn_stored()};
    const char *saved_policy = opt(argc, argv, "--policy", "");
    if (!*saved_policy) saved_policy = fp4 >= 2 ? "all=fp4:fp4:fp4,enc0.c1=fp16" : fp4 ? "all=fp4:fp4:fp8,enc0.c1=fp16" : fp8 ? "all=fp8:fp8:fp8,enc0.c1=fp16" : "";
    if (strlen(saved_policy) >= sizeof runtime.policy || strlen(optname) >= sizeof runtime.optimizer) { fprintf(stderr, "checkpoint settings too long\n"); return 2; }
    strcpy(runtime.policy, saved_policy); strcpy(runtime.optimizer, optname);
    char checkpoint_extra[8192], runtime_extra[8192];
    if (checkpoint_runtime_json(&runtime, runtime_extra, sizeof runtime_extra)) { fprintf(stderr, "invalid checkpoint settings\n"); return 2; }
    {
        size_t n = strlen(runtime_extra); runtime_extra[n-1] = 0;
        snprintf(runtime_extra+n-1, sizeof runtime_extra-n+1,
            ",\"target\":{\"version\":1,\"erode_native_voxels\":%d,\"erosion_kernel\":\"face6\",\"soft_sigma\":%.9g,\"soft_unit\":\"%s\",\"loss_tolerance_voxels\":%d}}", sc.erode, sc.soft, sc.soft_um ? "um" : "vox", nn_get_loss_tol());
    }
    if (g_heads) {   /* the recto heads' output channels and their sources (predict --channel picks one) */
        char hj[2048]; size_t k = (size_t)snprintf(hj, sizeof hj, ",\"heads\":{\"version\":1,\"recto_channels\":[");
        for (int h = 0; h < g_heads; h++) k += (size_t)snprintf(hj + k, sizeof hj - k, "%s%d", h ? "," : "", h ? 1 + h : 0);
        k += (size_t)snprintf(hj + k, sizeof hj - k, "],\"sources\":[");
        for (int h = 0; h < g_heads && k < sizeof hj - 64; h++) {
            k += (size_t)snprintf(hj + k, sizeof hj - k, "%s[", h ? "," : ""); int f = 1;
            for (int i = 0; i < S->n && k < sizeof hj - 64; i++) if (S->src[i].head == h) { k += (size_t)snprintf(hj + k, sizeof hj - k, "%s\"%s\"", f ? "" : ",", S->src[i].name); f = 0; }
            k += (size_t)snprintf(hj + k, sizeof hj - k, "]");
        }
        snprintf(hj + k, sizeof hj - k, "]}}");
        size_t n = strlen(runtime_extra); runtime_extra[n-1] = 0;
        snprintf(runtime_extra+n-1, sizeof runtime_extra-n+1, "%s", hj);
    }
    fprintf(stderr, "surface targets: binary-core erosion %d native voxel(s), then soft sigma %g %s; loss offset tolerance %d voxel(s)\n", sc.erode, sc.soft, sc.soft_um ? "um" : "level-0 voxels", nn_get_loss_tol());
    if (g_aff) {
        size_t n=strlen(runtime_extra); runtime_extra[n-1]=0;
        snprintf(runtime_extra+n-1,sizeof runtime_extra-n+1,",\"task\":\"band_affinity\",\"affinity\":{\"version\":1,\"channel0\":\"recto\",\"offsets_axis_d\":[[0,1],[1,1],[2,1],[0,8],[1,8],[2,8]],\"band\":\"winding_mod14 nearest-recto +- half turn\",\"lambda\":%.9g}}",g_aff_lambda);
    }
    if (g_papyrus) {   /* channel 0 = papyrus (src/papyrus.h); predict reads "cond" (papyrus_checkpoint) */
        size_t n=strlen(runtime_extra); runtime_extra[n-1]=0;
        snprintf(runtime_extra+n-1,sizeof runtime_extra-n+1,",\"task\":\"papyrus\",\"papyrus\":{\"version\":1,\"cond\":%d,\"code\":\"%s\",\"cut\":[%.9g,%.9g],\"rel\":[%.9g,%.9g],\"rel_p\":%.9g,\"sigma\":%.9g,\"tau\":[%.9g,%.9g],\"band\":%.9g,\"clean\":%d}}",
                 g_pap.cond,g_pap.cond==2?"input (ct - cutoff) / window sd":g_pap.cond?"stem bias (cutoff - window mean) / window sd":"none",g_pap.t_lo,g_pap.t_hi,g_pap.rel_lo,g_pap.rel_hi,g_pap.rel_p,g_pap.sigma,g_pap.tau_lo,g_pap.tau_hi,g_pap.band,g_pap.clean);
        if (g_pap.local > 0) {   /* gap labels (papyrus_local): inside the papyrus object */
            n=strlen(runtime_extra); runtime_extra[n-2]=0;
            snprintf(runtime_extra+n-2,sizeof runtime_extra-n+2,",\"labels\":\"gap\",\"local\":%d,\"local_sigma\":%.9g,\"local_depth\":%.9g,\"local_margin\":%.9g}}",
                     g_pap.local,g_pap.local_sigma,g_pap.local_depth,g_pap.local_margin);
        }
    }
    if (g_restore) {   /* predict reads "restore" (residual output, the input layout, the fine voxel size) */
        size_t n=strlen(runtime_extra); runtime_extra[n-1]=0;
        snprintf(runtime_extra+n-1,sizeof runtime_extra-n+1,",\"task\":\"restore\",\"restore\":{\"version\":1,\"input\":\"%s\",\"center\":128,\"scale\":64,\"residual\":true,\"output\":\"input + 64 logit\",\"upsampling\":\"trilinear\","
                 "\"fine_um\":%.9g,%s\"eps\":%.9g,\"mse_weight\":%.9g,\"edge_weight\":%.9g,\"mix\":{\"identity\":%.9g,\"pyramid\":%.9g,\"scanner\":%.9g},\"pyramid\":{\"factor\":4,\"q\":%.9g},"
                 "\"scanner\":{\"step\":[%.9g,%.9g],\"psf\":[%.9g,%.9g],\"aniso\":%.9g,\"noise\":[%.9g,%.9g],\"sigvar\":[%.9g,%.9g],\"corr\":[%.9g,%.9g],\"edge\":[%.9g,%.9g],\"edge_p\":%.9g,\"gain\":%.9g,\"gamma\":%.9g,\"bias\":%.9g,\"q\":%.9g}}}",
                 g_rsplit?"split":"plain",g_rpairs?S->src[0].um/4:S->src[0].um*(double)(1<<(S->src[0].min_level>0?S->src[0].min_level:0)),g_rpairs?"\"sr_scale\":4,\"sr_pairs\":1,":"",g_reps,g_rmse,g_redge,g_rr.mix[0],g_rr.mix[1],g_rr.mix[2],g_rr.pyr_q,
                 g_rr.sc_step[0],g_rr.sc_step[1],g_rr.sc_psf[0],g_rr.sc_psf[1],g_rr.sc_aniso,g_rr.sc_noise[0],g_rr.sc_noise[1],g_rr.sc_sigvar[0],g_rr.sc_sigvar[1],g_rr.sc_corr[0],g_rr.sc_corr[1],
                 g_rr.sc_edge[0],g_rr.sc_edge[1],g_rr.sc_edge_p,g_rr.sc_gain,g_rr.sc_gamma,g_rr.sc_bias,g_rr.sc_q);
    }
    if (g_verso) {   /* channel 1 = verso; band affinities (if any) follow at 2.. */
        size_t n=strlen(runtime_extra); runtime_extra[n-1]=0;
        snprintf(runtime_extra+n-1,sizeof runtime_extra-n+1,",\"verso\":{\"version\":1,\"channel\":1,\"target\":\"%s recto channel with the radial input reversed\",\"start_step\":%d}}",g_verso_self?"live":"frozen starting weights",g_verso_start);
    }
    if (g_side == 1) {   /* the last channel: recto / verso side of the nearest labelled recto (src/band.h) */
        size_t n=strlen(runtime_extra); runtime_extra[n-1]=0;
        snprintf(runtime_extra+n-1,sizeof runtime_extra-n+1,",\"side\":{\"version\":1,\"channel\":%d,\"positive\":\"recto side (outward of the nearest labelled recto)\",\"air\":%d,\"lambda\":%.9g}}",g_side_ch,sc.side_air,g_side_lambda);
    }
    if (g_side == 2) {   /* the last two channels: winding phase between consecutive rectos (src/band.h PHASE_*) */
        size_t n=strlen(runtime_extra); runtime_extra[n-1]=0;
        snprintf(runtime_extra+n-1,sizeof runtime_extra-n+1,",\"phase\":{\"version\":1,\"channel\":%d,\"channels\":[\"0.5+0.5cos(2pi phase)\",\"0.5+0.5sin(2pi phase)\"],\"phase\":\"fraction of the way outward from a recto to the next\",\"air\":%d,\"lambda\":%.9g,\"dense\":%d}}",g_side_ch,sc.side_air,g_side_lambda,sc.phase_dense);
    }
    if (g_unwrap2) {   /* channel layout of the unwrap2 heads (predict --fit-fields reads "version":2) */
        size_t n=strlen(runtime_extra); runtime_extra[n-1]=0;
        snprintf(runtime_extra+n-1,sizeof runtime_extra-n+1,",\"unwrap\":{\"version\":2,\"channels\":{\"phase\":[%d,%d],\"normal\":[%d,%d,%d,%d,%d,%d],\"density\":%d,\"wind4\":[%d,%d]},\"normal\":\"n_z^2, n_y^2, n_x^2, 0.5+n_z n_y, 0.5+n_z n_x, 0.5+n_y n_x\",\"density\":\"%s, rho = |grad phase| turns per voxel\",\"wind4\":\"0.5+0.5cos / sin(2pi W/4), W the stored phase unwrapped over the window, gauge-free\",\"density_scale\":%.9g,\"density_code\":%d,\"packed_weight\":%.9g,\"packed_um\":%.9g,\"normal_lambda\":%.9g,\"normal_step\":%d,\"wind4_lambda\":%.9g,\"wind4_step\":%d,\"wind4_min\":%d,\"wind4_tile\":%d}}",
                 g_side_ch,g_side_ch+1,g_norm_ch,g_norm_ch+1,g_norm_ch+2,g_norm_ch+3,g_norm_ch+4,g_norm_ch+5,g_norm_ch+6,g_w4_ch,g_w4_ch+1,
                 g_dens_code ? "x / (1 + x), x = density_scale rho" : "min(1, density_scale rho)",g_dens_scale,g_dens_code,g_packed_w,g_packed_um,g_norm_lambda,g_norm_s,g_w4_lambda,sc.wind4_step,sc.wind4_min,sc.wind4_tile);
    } else if (g_unwrap) {   /* channel layout of the unwrap heads */
        size_t n=strlen(runtime_extra); runtime_extra[n-1]=0;
        snprintf(runtime_extra+n-1,sizeof runtime_extra-n+1,",\"unwrap\":{\"version\":1,\"channels\":{\"recto\":0,\"ink\":1,\"phase\":[%d,%d],\"normal\":[%d,%d,%d,%d,%d,%d],\"density\":%d},\"normal\":\"n_z^2, n_y^2, n_x^2, 0.5+n_z n_y, 0.5+n_z n_x, 0.5+n_y n_x\",\"density\":\"%s, rho = |grad phase| turns per voxel\",\"density_scale\":%.9g,\"density_code\":%d,\"packed_weight\":%.9g,\"packed_um\":%.9g,\"ink_lambda\":%.9g,\"ink_dice\":%.9g,\"normal_lambda\":%.9g,\"normal_step\":%d}}",
                 g_side_ch,g_side_ch+1,g_norm_ch,g_norm_ch+1,g_norm_ch+2,g_norm_ch+3,g_norm_ch+4,g_norm_ch+5,g_norm_ch+6,
                 g_dens_code ? "x / (1 + x), x = density_scale rho" : "min(1, density_scale rho)",g_dens_scale,g_dens_code,g_packed_w,g_packed_um,g_ink_lambda,g_ink_dice,g_norm_lambda,g_norm_s);
        if (g_cond) {   /* conditioned inference (predict --cond-sweep): inside the unwrap object */
            n=strlen(runtime_extra); runtime_extra[n-2]=0;
            snprintf(runtime_extra+n-2,sizeof runtime_extra-n+2,",\"cond\":{\"version\":1,\"inputs\":[4,5,6],\"input\":\"cos, sin of 2 pi q / 14, mask\",\"channels\":[%d,%d,%d,%d],\"periods\":[14,2],\"lambda\":%.9g,\"loss\":\"%s\"}}}",
                     g_cond_ch,g_cond_ch+1,g_cond_ch+2,g_cond_ch+3,g_cond_lambda,g_cond_outside?"outside":"all");
        }
        if (g_pap_head) {   /* the papyrus / void head: inside the unwrap object */
            n=strlen(runtime_extra); runtime_extra[n-2]=0;
            snprintf(runtime_extra+n-2,sizeof runtime_extra-n+2,",\"papyrus\":{\"version\":1,\"channel\":%d,\"labels\":\"gap\",\"cut\":[%.9g,%.9g],\"sigma\":%.9g,\"band\":%.9g,\"local\":%d,\"local_sigma\":%.9g,\"local_depth\":%.9g,\"local_margin\":%.9g,\"lambda\":%.9g}}}",
                     g_pap_ch,g_paph.t_lo,g_paph.t_hi,g_paph.sigma,g_paph.band,g_paph.local,g_paph.local_sigma,g_paph.local_depth,g_paph.local_margin,g_pap_lambda);
        }
    }
    if (sc.phase_near_um > 0) {   /* the phase supervision radius */
        size_t n=strlen(runtime_extra); runtime_extra[n-1]=0;
        snprintf(runtime_extra+n-1,sizeof runtime_extra-n+1,",\"phase_near_um\":%.9g}",sc.phase_near_um);
    }
    if (g_count_lambda > 0) {   /* the count loss */
        size_t n=strlen(runtime_extra); runtime_extra[n-1]=0;
        snprintf(runtime_extra+n-1,sizeof runtime_extra-n+1,",\"count_loss\":{\"lambda\":%.9g,\"segments\":%d,\"length\":%d}}",g_count_lambda,g_count_K,g_count_L);
    }
    if (g_Pz != P) {   /* slab windows: the training window's depth (predict --window-z) */
        size_t n=strlen(runtime_extra); runtime_extra[n-1]=0;
        snprintf(runtime_extra+n-1,sizeof runtime_extra-n+1,",\"train_window_z\":%d}",g_Pz);
    }
    if (g_sheet) {
        size_t n=strlen(runtime_extra); runtime_extra[n-1]=0;
        snprintf(runtime_extra+n-1,sizeof runtime_extra-n+1,",\"task\":\"surface_winding\",\"sheet\":{\"version\":1,\"units\":\"turns\",\"outputs\":[\"surface_logit\",\"winding_residual\"],\"geometry_sha256\":\"%s\",\"reference_sha256\":\"%s\",\"variant\":%d,\"schedule_start\":%d}}",g_sheet->manifest_sha,g_sheet->reference_sha,g_sheet_variant,schedule_start);
    }
    {
        size_t n=strlen(runtime_extra); runtime_extra[n-1]=0;
        snprintf(runtime_extra+n-1,sizeof runtime_extra-n+1,",\"augmentation\":{\"version\":1,\"mode\":%d,\"ct\":%d,\"symmetry_p\":%.9g,\"axis_jitter\":%.9g,\"axis_jitter_p\":%.9g,\"axis_jitter_deg\":%.9g,\"geometry\":%d,\"rotate_degrees\":%.9g,\"rotate_p\":%.9g,\"elastic_voxels\":%.9g,\"elastic_p\":%.9g,\"label_morph_voxels\":%.9g,\"label_morph_p\":%.9g,\"soft_sigma\":%.9g,\"sheet_band_scale\":%.9g}}",
            sc.augment,sc.ct_augment,sc.symmetry_p,sc.axis_jitter,sc.axis_jitter_p,sc.axis_jitter_deg,sc.geometry_augment,sc.rotate_degrees,sc.rotate_p,sc.elastic,sc.elastic_p,sc.label_morph,sc.label_morph_p,sc.soft,sheet_band_scale);
    }
    if (scan_aug_enabled(&sc.scan) || (sc.affine_p>0 && sc.scan.strength>0)) {   /* recorded only when on: older runs keep their metadata */
        size_t n=strlen(runtime_extra); runtime_extra[n-1]=0;
        int k=snprintf(runtime_extra+n-1,sizeof runtime_extra-n+1,",\"scan_augmentation\":{\"version\":1,\"strength\":%.9g,\"drop_ignore\":%d,\"affine_p\":%.9g,\"aniso\":%.9g,\"shear\":%.9g,\"p\":{",
            sc.scan.strength,sc.scan.drop_ignore,sc.affine_p,sc.affine_aniso,sc.affine_shear);
        for (int i=0;i<SA_N;i++) k+=snprintf(runtime_extra+n-1+k,sizeof runtime_extra-n+1-k,"%s\"%s\":%.9g",i?",":"",scan_aug_names[i],sc.scan.p[i]);
        k+=snprintf(runtime_extra+n-1+k,sizeof runtime_extra-n+1-k,"}");
        { int any=0; for (int i=0;i<SA_N;i++) if (sc.scan.s[i]>0) { k+=snprintf(runtime_extra+n-1+k,sizeof runtime_extra-n+1-k,"%s\"%s\":%.9g",any?",":",\"s\":{",scan_aug_names[i],sc.scan.s[i]); any=1; } if (any) k+=snprintf(runtime_extra+n-1+k,sizeof runtime_extra-n+1-k,"}"); }
        if (sc.scan.nhist) k+=snprintf(runtime_extra+n-1+k,sizeof runtime_extra-n+1-k,",\"hist_targets\":%d",sc.scan.nhist);
        snprintf(runtime_extra+n-1+k,sizeof runtime_extra-n+1-k,"}}");
        fprintf(stderr,"scanner augmentation: strength %.3g, drop-ignore %d, affine p %.3g (aniso %.3g shear %.3g); p",sc.scan.strength,sc.scan.drop_ignore,sc.affine_p,sc.affine_aniso,sc.affine_shear);
        for (int i=0;i<SA_N;i++) if (sc.scan.p[i]>0) fprintf(stderr," %s %.3g%s",scan_aug_names[i],sc.scan.p[i],sc.scan.s[i]>0?"*":"");
        { int any=0; for (int i=0;i<SA_N;i++) if (sc.scan.s[i]>0) { fprintf(stderr,"%s%s %.3g",any?", ":"; strength ",scan_aug_names[i],sc.scan.s[i]); any=1; } }
        fprintf(stderr,"\n");
    }
    if (strcmp(g_aug_name,"none") || sc.zoom_out_p>0) {   /* recorded when not the old default (older runs keep their metadata) */
        size_t n=strlen(runtime_extra); runtime_extra[n-1]=0;
        snprintf(runtime_extra+n-1,sizeof runtime_extra-n+1,",\"aug\":{\"preset\":\"%s\",\"zoom_min\":%.9g,\"zoom_p\":%.9g,\"zoom_max\":%.9g,\"zoom_out_p\":%.9g}}",
                 g_aug_name,sc.zoom_min,sc.zoom_p,sc.zoom_max,sc.zoom_out_p);
    }
    fprintf(stderr,"augmentation preset %s: zoom in to %.3g p %.3g, out to %.3g p %.3g\n",g_aug_name,sc.zoom_min,sc.zoom_p,sc.zoom_max,sc.zoom_out_p);
    fprintf(stderr,"augmentation: mode %d, CT %d, symmetry p %.3g, axis jitter %.3g voxels (p %.3g, <= %.3g degrees), geometry %d (rotation +/-%.3g degrees p %.3g; elastic <=%.3g voxels p %.3g), soft-band dilation/erosion +/-%.3g voxels p %.3g\n",
        sc.augment,sc.ct_augment,sc.symmetry_p,sc.axis_jitter,sc.axis_jitter_p>0?sc.axis_jitter_p:.1,sc.axis_jitter_deg>0?sc.axis_jitter_deg:2,sc.geometry_augment,sc.rotate_degrees,sc.rotate_p,sc.elastic,sc.elastic_p,sc.label_morph,sc.label_morph_p);
    if (schedule_seconds) {
        size_t n=strlen(runtime_extra); runtime_extra[n-1]=0;
        snprintf(runtime_extra+n-1,sizeof runtime_extra-n+1,",\"time_schedule\":{\"version\":1,\"seconds\":%.9g,\"elapsed_at_launch\":%.9g,\"warmup_seconds\":%.9g,\"cooldown\":%.9g}}",
            schedule_seconds,schedule_elapsed,warmup_seconds,cooldown);
    }
    strcpy(checkpoint_extra, runtime_extra);
    if (plan && cover_checkpoint_extra(runtime_extra, &coverage, checkpoint_extra, sizeof checkpoint_extra)) return 2;
    fprintf(stderr, "model widths"); for (int i = 0; i < cfg.nlev; i++) fprintf(stderr, " %d", cfg.widths[i]);
    if (cfg.l0_norm) fprintf(stderr, " (level-0 norm %s)", cfg.l0_norm == 1 ? "none" : cfg.l0_norm == 2 ? "ema" : cfg.l0_norm == 3 ? "ws" : "slab");
    if (g_Pz != P) fprintf(stderr, ": %zu params; window %dx%dx%d (slab) B=%d x %d GPU(s) [", np, g_Pz, P, P, B, ng); else fprintf(stderr, ": %zu params; P=%d B=%d x %d GPU(s) [", np, P, B, ng); for (int g = 0; g < ng; g++) fprintf(stderr, "%s%d", g ? "," : "", devs[g]); fprintf(stderr, "] %s%s\n", nn_get_tf32() ? "tensor cores (see precision.txt for compute and storage)" : "fp32", split ? ", each window split along z" : "");
    split_ctx *sctx = split ? split_create(devs[0], devs[1]) : nullptr;
    nn_init(G[0].dev);
    float *gpeer = ng > 1 ? nn_malloc(np * 4) : nullptr;   /* on GPU 0: incoming gradients of the other GPUs */

    /* validation set: fixed batches from the held-out boxes (unaugmented), scored on GPU 0 */
    sample_cfg vc = sc; vc.gpu_aug = 0; vc.cover = nullptr; vc.cover_start = 0; vc.ct_augment = 0; vc.axis_jitter = 0; vc.geometry_augment=0; vc.label_morph=0; vc.scan.strength=0; vc.affine_p=0; vc.crush.p=0; vc.xscan_p=0; vc.tear_p=0; vc.delam_p=0; vc.curv_p=0; vc.incl_p=0; vc.density_p=0; { const char *vs = opt(argc, argv, "--val-seed", nullptr); vc.seed = (vs ? (uint64_t)atoll(vs) : seed) + 777; }   /* --val-seed: other training seeds on the same windows */ vc.augment = 0; vc.nworkers = 4; vc.nbuf = 2; vc.holdout = 1;
    { int any = 0; for (int i = 0; i < S->n; i++) any |= S->src[i].hold_n[0] > 0 || (S->src[i].srp && S->src[i].srp->nte > 0); if (!any) { vc.holdout = 0; fprintf(stderr, "no holdout boxes in the sources: validation batches come from the training distribution\n"); } }
    /* the validation windows are the first nval batches of a deterministic sampler (each window from its own seeded rng: the same
       windows in every run with this seed), drawn once here. --val-store disk (default): kept in an unlinked file in the run directory
       (3 B per voxel each with --gpu-input and packed targets; its space returns when the trainer exits) and read back at every pass
       through one host batch; ram: copies in host RAM (the former layout); regen: drawn again at every pass (no copies; the pass waits
       for 3 sampler workers). The sheet prior's per-window records keep ram. */
    vc.deterministic = 1; vc.nworkers = 3; vc.nbuf = 3;
    /* --val-P N: N^3 windows (vc.P), whatever the training window; the batch layout helpers see the validation size while they run */
    const int vP = g_valP ? g_valP : P, Pz_tr = g_Pz;
    const size_t p3v = g_valP ? (size_t)vP * vP * vP : p3;
    if (g_valP) { vc.P = vP; vc.Pz = 0; g_Pz = vP; }
    uint64_t vhash = 0x9E3779B97F4A7C15ull;   /* the validation windows' bytes (the same windows across runs: the same hash) */
    const char *vmode_s = opt(argc, argv, "--val-store", "disk");
    int vmode = !strcmp(vmode_s, "ram") ? VAL_RAM : !strcmp(vmode_s, "regen") ? VAL_REGEN : !strcmp(vmode_s, "disk") ? VAL_DISK : -1;
    if (vmode < 0) { fprintf(stderr, "--val-store: disk, ram or regen\n"); return 2; }
    if (vmode == VAL_DISK && sc.sheet) vmode = VAL_RAM;
    batch *val = calloc((size_t)nval, sizeof *val);
    valdisk vdk = {.fd = -1};
    if (nval > 0 && vmode != VAL_REGEN) {
        if (vmode == VAL_DISK) {
            char vp[1400]; snprintf(vp, sizeof vp, "%s/.val-windows", out);
            vdk.fd = open(vp, O_RDWR | O_CREAT | O_TRUNC, 0600);
            if (vdk.fd < 0 || unlink(vp)) { fprintf(stderr, "cannot create %s (--val-store ram or regen avoid it)\n", vp); return 1; }
        }
        sampler *vs = sampler_start(S, &vc);   /* --val-batches 0: no validation (e.g. windows wider than the holdout) */
        if (!vs) { fprintf(stderr, "cannot construct validation sampler\n"); return 1; }
        for (int i = 0; i < nval; i++) {
            batch *b = sampler_next(vs);
            if (!b) { nval = i; break; }
            void **fb[VAL_NF], **fv[VAL_NF]; size_t sz[VAL_NF];
            const int nf = val_fields(b, B, p3v, fb, sz); val_fields(&val[i], B, p3v, fv, sz);
            for (int k = 0; k < nf; k++) vhash = hash_bytes(vhash, *fb[k], sz[k]);
            for (int k = 0; k < nf; k++)
                if (vmode == VAL_RAM) { *fv[k] = malloc(sz[k]); memcpy(*fv[k], *fb[k], sz[k]); }
                else if (pwrite_all(vdk.fd, *fb[k], sz[k], vdk.bytes)) { fprintf(stderr, "cannot write the validation windows\n"); return 1; }
                else vdk.bytes += sz[k];
            if (vmode == VAL_RAM) { val[i].sheet=calloc((size_t)B,sizeof *val[i].sheet); val[i].sheet[0]=sheet_clone(b->sheet?b->sheet[0]:nullptr); }
            sampler_release(vs, b);
        }
        int validation_failed = sampler_failed(vs);
        sampler_stop(vs);
        if (validation_failed) { fprintf(stderr, "validation sampler failed\n"); return 1; }
        if (vmode == VAL_DISK) {   /* one host batch to read them back into; the written pages leave the page cache */
            void **fv[VAL_NF]; size_t sz[VAL_NF];
            const int nf = val_fields(&vdk.buf, B, p3v, fv, sz);
            for (int k = 0; k < nf; k++) if (!(*fv[k] = malloc(sz[k]))) { fprintf(stderr, "out of host memory (validation batch)\n"); return 1; }
            fdatasync(vdk.fd); posix_fadvise(vdk.fd, 0, 0, POSIX_FADV_DONTNEED);
        }
        fprintf(stderr, "%d validation batches of %d^3, %s (--val-store %s), windows hash %016llx\n", nval, vP, vmode == VAL_RAM ? "held in host RAM" : "in an unlinked file in the run directory, read back at each pass", vmode_s, (unsigned long long)vhash);
    } else if (nval > 0) fprintf(stderr, "%d validation batches, drawn again at every validation pass (--val-store regen)\n", nval);
    else fprintf(stderr, "0 validation batches\n");
    g_Pz = Pz_tr;

    if (!sc.cover && step0 > 0) {   /* resumed: a fresh sample stream (the per-sample seeds count from batch 0 at every sampler start, so a
                                       resume under --det replayed the windows and augmentations of steps 1, 2, ...); validation keeps its seed */
        sc.seed = seed + 0x9E3779B97F4A7C15ull * (uint64_t)step0;
        fprintf(stderr, "resumed at step %d: training sample stream reseeded\n", step0);
    }
    if (!atoi(opt(argc, argv, "--sampler-slots", "0"))) {   /* the default slot count against the host memory available now */
        const size_t p3s = (size_t)g_Pz * P * P, slotb = (size_t)B * p3s * ((size_t)g_xs * xbytes() + (g_pack ? 1 : NCH + 1) + (g_aff != 0) + (g_side != 0)
                           + (g_unwrap != 0) + 2 * (g_unwrap2 != 0) + (g_cond != 0) + (g_pap_head != 0)) + rad_bytes(B), win = slotb + 4 * p3s;   /* + a worker's scratch */
        const size_t avail = host_mem_available(), budget = avail / 10 * 7;
        int cap = win ? (int)(budget / win) : sc.nbuf;
        if (cap < 3) cap = 3;   /* (one uploading, two drawing) */
        if (avail && cap < sc.nbuf) {
            fprintf(stderr, "sampler: %d slots (not %d): %.2f GB per window in flight (slot + worker scratch), %.1f GB host memory available; --sampler-slots sets it\n",
                    cap, sc.nbuf, win / 1e9, avail / 1e9);
            sc.nbuf = cap;
        }
        /* a worker fills a whole slot, so with fewer free slots than workers the rest park on the slot condition variable (the desk's
           unwrap2 run: 28 workers, 2 GPUs, B 1 -> 10 slots, the trainer holding some, ~6-10 drawing): by default every worker a slot
           plus 2 per GPU with the trainer and 2 ready, as far as the memory left beside the workers' own scratch (~16 P^3 bytes each:
           draw()'s ttmp / big and pooled window buffers) allows. --det 1 hands batches out in order, so a slow window holds every
           finished one behind it in its slot (head-of-line): with draw times varying 2x (warped or not, scanner ops) a slot per
           worker kept ~60% of them drawing (a queue simulation reproduces the desk's 15 of 28); 1.5 slots per worker ~90% */
        const int want = ((sc.deterministic ? 3 * workers / 2 : workers) + B - 1) / B + 2 * ng + 2;
        if (avail && want > sc.nbuf && sc.nbuf == 4 * ng + 2) {
            const long long room = (long long)budget - (long long)workers * 16 * (long long)p3s - (long long)sc.nbuf * (long long)win;
            long long extra = room > 0 && slotb ? room / (long long)slotb : 0;
            if (extra > want - sc.nbuf) extra = want - sc.nbuf;
            sc.nbuf += (int)extra;
            fprintf(stderr, "sampler: %d slots for %d workers (B %d)%s\n", sc.nbuf, workers, B,
                    sc.nbuf < want ? ": host memory caps them, workers beyond the free slots wait (--sampler-slots sets it)" : "");
        }
    }
    if (sc.deterministic && sc.nbuf < (3 * workers / 2 + B - 1) / B + 2 * ng + 2)
        fprintf(stderr, "sampler: %d slots for %d workers under --det 1: batches leave in order, so finished windows wait in their slots behind "
                "slower ones and workers without a free slot idle; ~1.5 slots per worker + %d keep them drawing (fewer workers and more slots "
                "beat more workers: each idle worker holds ~16 P^3 bytes of scratch, a slot %.2f GB)\n", sc.nbuf, workers, 2 * ng + 2,
                (double)B * g_Pz * P * P * ((size_t)g_xs * xbytes() + (g_pack ? 1 : NCH + 1) + (g_side != 0) + 2 * (g_unwrap2 != 0)) / 1e9);
    sampler *sp = sampler_start(S, &sc);
    if (!sp) return 1;
    char logp[1400]; snprintf(logp, sizeof logp, "%s/log.csv", out);
    FILE *log = fopen(logp, step0 ? "a" : "w");
    FILE *sheet_log=nullptr;
    if (g_sheet) {
        snprintf(logp,sizeof logp,"%s/geometry.csv",out); sheet_log=fopen(logp,step0&&!sheet_init?"a":"w");
        if (sheet_log) { fseek(sheet_log,0,SEEK_END); if (!ftell(sheet_log)) fputs("step,coordinate,continuity,ordering,path,gap,weighted,ramp\n",sheet_log); }
    }
    if (log) {
        fseek(log, 0, SEEK_END);
        if (ftell(log) == 0) fprintf(log, "step,lr,loss,bce,dice,active,gnorm,val_loss,val_bce,val_dice,samples_per_s,wait_s\n");
    }
    signal(SIGINT, on_sig); signal(SIGTERM, on_sig);
    int prof = ufsm_env_on("UFSM_PROF");
    double t0 = now(), tlog = t0, wait = 0, acc_loss = 0, acc_bce = 0, acc_dice = 0, acc_g = 0; int nacc = 0;
    if (seconds) fprintf(stderr, "training budget %.1fs, time-based warmup %.1fs\n", seconds, warmup_seconds);
    if (schedule_seconds) fprintf(stderr,"wall-time LR horizon %.1fs, elapsed %.1fs, warmup %.1fs; finite cover/ramp origin preserved\n",schedule_seconds,schedule_elapsed,warmup_seconds);
    int committed_step = step0, training_failed = 0;
    double best_val = 1e30, vcrit = 0; int nskip = 0; (void)nskip;
    double acc_rn = 0, acc_re = 0, acc_rb = 0;   /* restore: training voxels, error^2, input error^2 since the last log line */
    float parts[8][2 * TCMAX + 1];
    double acc_emb = 0, acc_embc = 0, acc_ds = 0, acc_vbce = 0, acc_vdice = 0, acc_side = 0, acc_ink = 0, acc_norm = 0, acc_cond = 0, acc_ck = 0, acc_cr = 0, acc_pap = 0, acc_cnt = 0, acc_cnr = 0, acc_w4 = 0, acc_w4w = 0, acc_w4a = 0, acc_pg = 0;   /* (acc_pg: --pgrad-lambda loss; acc_cnt / acc_cnr: --count-lambda loss, Tp / Tg) verso channel, side channel; --cond (loss, known / right counts); --pap-head */
    for (int step = step0 + 1; step <= steps && !g_stop; step++) {
        g_verso_on = g_verso && step >= g_verso_start;
        g_ptrunc_on = g_ptrunc_tau > 0 && step >= g_ptrunc_from;   /* (off again for the validation passes below) */
        { uint64_t h = (uint64_t)step * 0x9E3779B97F4A7C15ull + (uint64_t)seed; h ^= h >> 31; h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 29;
          g_verso_flip = g_verso_on && g_verso_swap > 0 && (double)(h >> 11) * 0x1p-53 < g_verso_swap; }   /* same on every GPU */
        g_sheet_ramp=fminf(1.f,(float)(step-schedule_start)/500.f);
        nn_set_sr_step((unsigned)step);
        double loss = 0, active = 0; int want_next[64] = {0};   /* (per GPU: fetch its next batch after all launches) */
        /* forward + backward on every GPU; kernel launches are asynchronous so the GPUs overlap */
        if (split) {   /* one window on both GPUs: upload both slabs of a batch, then one job with the halo exchanges */
            if (!G[0].pending) {
                double tw = now(); batch *b = sampler_next(sp); wait += now() - tw;
                if (!b) { fprintf(stderr, "sampler stopped\n"); g_stop = 1; break; }
                for (int g = 0; g < 2; g++) { nn_init(G[g].dev); upload_async(&G[g], b, G[g].cur, B, P); }
                G[0].pending = b;
            }
            for (int g = 0; g < 2; g++) {
                gpu_state *d = &G[g];
                nn_init(d->dev);
                nn_stream_wait(0, d->ev_up[d->cur]);
                if (sparse_at && step >= sparse_at && !unet_get_sparse24(d->u)) { unet_set_sparse24(d->u, 1); if (g == 0) fprintf(stderr, "step %d: 2:4 sparsity on (SR-STE lambda %g)\n", step, srste); }
                unet_zero_grad(d->u);
            }
            split_arg sa = {G, B, P, 1, dice_w};
            split_run(sctx, split_job, &sa);
            for (int g = 0; g < 2; g++) { nn_init(G[g].dev); nn_event_record(G[g].ev_done[G[g].cur], 0); nn_event_sync(G[g].ev_up[G[g].cur]); }
            sampler_release(sp, G[0].pending);
            G[0].pending = nullptr;
            if (step < steps && !g_stop) {
                double tw = now(); batch *nb = sampler_next(sp); wait += now() - tw;
                if (!nb) { fprintf(stderr, "sampler stopped before final step\n"); training_failed = 1; break; }
                for (int g = 0; g < 2; g++) { nn_init(G[g].dev); int nxt = G[g].cur ^ 1; upload_async(&G[g], nb, nxt, B, P); select_buf(&G[g], nxt); }
                G[0].pending = nb;
            }
        } else
        for (int g = 0; g < ng; g++) {
            gpu_state *d = &G[g];
            nn_init(d->dev);
            want_next[g] = 0;
            if (overfit > 1) {                              /* diagnostic: cycle through the first N batches (host copies, synchronous upload) */
                static batch *fixed[256]; static int nfixed = 0;
                int k = (step - step0 - 1) % overfit;
                if (k >= nfixed) {
                    double tw = now(); batch *b = sampler_next(sp); wait += now() - tw;
                    if (!b) { fprintf(stderr, "sampler stopped\n"); g_stop = 1; break; }
                    batch *cp = calloc(1, sizeof *cp); size_t p3b = (size_t)g_Pz * P * P;
                    if (g_xfmt) { cp->x16 = malloc((size_t)B * g_xs * p3b * 2); memcpy(cp->x16, b->x16, (size_t)B * g_xs * p3b * 2); }
                    else { cp->x = malloc((size_t)B * g_xs * p3b * 4); memcpy(cp->x, b->x, (size_t)B * g_xs * p3b * 4); }
                    if (g_gpu_in) { cp->rad = malloc(rad_bytes(B)); memcpy(cp->rad, b->rad, rad_bytes(B)); }
                    if (g_axr) { cp->axr = malloc(axr_bytes(B, P)); memcpy(cp->axr, b->axr, axr_bytes(B, P)); }
                    if (g_view) { const size_t nv_ = (size_t)B * g_view * g_view * g_view; cp->xv = malloc(nv_); memcpy(cp->xv, b->xv, nv_); }
                    cp->t = malloc((size_t)B * NCH * p3b); memcpy(cp->t, b->t, (size_t)B * NCH * p3b);
                    cp->m = malloc((size_t)B * p3b); memcpy(cp->m, b->m, (size_t)B * p3b);
                    cp->w = malloc((size_t)B * NCH); memcpy(cp->w, b->w, (size_t)B * NCH);
                    cp->src = malloc((size_t)B * sizeof *b->src); memcpy(cp->src, b->src, (size_t)B * sizeof *b->src);
                    cp->um = malloc((size_t)B * sizeof(float)); memcpy(cp->um, b->um, (size_t)B * sizeof(float));
                    sampler_release(sp, b); fixed[nfixed++] = cp;
                }
                upload(d, fixed[k], B, P);
                unet_zero_grad(d->u); restore_train_recipe(d, step, g), run_batch(d, B, P, dice_w, 1); unet_backward_x(d->u, g_half && !g_pkf ? d->glh : d->gl, g_g16);
                continue;
            }
            if (ufsm_env_on("UFSM_SYNC_UPLOAD")) {               /* diagnostic: plain synchronous upload, single buffer */
                double tw = now(); batch *b = sampler_next(sp); wait += now() - tw;
                if (!b) { fprintf(stderr, "sampler stopped\n"); g_stop = 1; break; }
                upload(d, b, B, P); sampler_release(sp, b);
                if (getenv("UFSM_DUMP_BATCH") && step == step0 + 5) {   /* diagnostic: what the GPU sees (mid slice of patch 0: CT | target | mask) */
                    size_t p3 = (size_t)g_Pz * P * P; const size_t zm = (size_t)(g_Pz / 2); float *hx = malloc(4 * p3 * 4); uint8_t *ht = malloc(g_tc * p3), *hm = malloc(p3);
                    if (!g_xfmt) nn_d2h(hx, d->x, 4 * p3 * 4);
                    else { uint16_t *h16 = malloc(4 * p3 * 2); nn_d2h(h16, d->x, 4 * p3 * 2);
                        for (size_t k = 0; k < 4 * p3; k++) { if (g_xfmt == 1) { _Float16 h; memcpy(&h, &h16[k], 2); hx[k] = (float)h; } else { uint32_t u = (uint32_t)h16[k] << 16; memcpy(&hx[k], &u, 4); } }
                        free(h16); }
                    nn_d2h(ht, d->t, g_tc * p3); nn_d2h(hm, d->m, p3);
                    FILE *f = fopen(getenv("UFSM_DUMP_BATCH"), "wb"); fprintf(f, "P5\n%d %d\n255\n", 3 * P, P);
                    for (int y = 0; y < P; y++) {
                        for (int x = 0; x < P; x++) { float v = hx[(zm * P + y) * P + x]; int g = (int)(128 + 40 * v); fputc(g < 0 ? 0 : g > 255 ? 255 : g, f); }
                        for (int x = 0; x < P; x++) fputc(ht[(zm * P + y) * P + x], f);
                        for (int x = 0; x < P; x++) fputc(hm[(zm * P + y) * P + x] ? 255 : 0, f);
                    }
                    fclose(f); free(hx); free(ht); free(hm); fprintf(stderr, "dumped batch to %s\n", getenv("UFSM_DUMP_BATCH"));
                }
                unet_zero_grad(d->u); restore_train_recipe(d, step, g), run_batch(d, B, P, dice_w, 1); unet_backward_x(d->u, g_half && !g_pkf ? d->glh : d->gl, g_g16);
                continue;
            }
            /* the batch for this step was uploaded during the previous step (first step: upload now) */
            if (!d->pending) {
                double tw = now();
                batch *b = sampler_next(sp);
                wait += now() - tw;
                if (!b) { fprintf(stderr, "sampler stopped\n"); g_stop = 1; break; }
                upload_async(d, b, d->cur, B, P);
                axr_set(d, d->cur);
                upload_tail(d, B, P);                          /* (upsplit: its tail too) */
                d->pending = b;
            }
            nn_stream_wait(0, d->ev_up[d->cur]);           /* compute waits for the upload */
            if (sparse_at && step >= sparse_at && !unet_get_sparse24(d->u)) { unet_set_sparse24(d->u, 1); if (g == 0) fprintf(stderr, "step %d: 2:4 sparsity on (SR-STE lambda %g)\n", step, srste); }
            unet_zero_grad(d->u);
            d->last = d->cur;
            test_nan_inject(d, step, 0, P);   /* tests: UFSM_TEST_NAN */
            restore_train_recipe(d, step, g), run_batch(d, B, P, dice_w, 1);
            test_nan_inject(d, step, 1, P);
            if (sync_check(d, step, "forward and losses")) return 1;
            unet_backward_x(d->u, g_half && !g_pkf ? d->glh : d->gl, g_g16);
            test_nan_inject(d, step, 2, P);
            if (sync_check(d, step, "backward")) return 1;
            diag_nonfinite(d, "parameter gradients", unet_grad_ptr(d->u), unet_nparams(d->u), 0);
            nn_event_record(d->ev_done[d->cur], 0);
            nn_event_sync(d->ev_up[d->cur]);                 /* host buffer of this batch is free again */
            if (overfit || g_bench_same) continue;           /* keep computing on the same device buffer (UFSM_BENCH_SAME: GPU step time without the loader) */
            sampler_release(sp, d->pending);
            /* prefetch the next batch into the other buffer while this step computes; several GPUs: after every GPU's step is
               launched (waiting here for GPU 0's next batch held GPU 1's launch back whenever the sampler was behind) */
            d->pending = nullptr;
            want_next[g] = step < steps && !g_stop;
        }
        for (int g = 0; g < ng && !g_stop && !training_failed; g++) {
            if (!want_next[g]) continue;
            gpu_state *d = &G[g]; nn_init(d->dev);
            double tw = now(); batch *nb = sampler_next(sp); wait += now() - tw;
            if (!nb) { fprintf(stderr, "sampler stopped before final step\n"); training_failed = 1; break; }
            int nxt = d->cur ^ 1; upload_async(d, nb, nxt, B, P); d->pending = nb; select_buf(d, nxt);
        }
        if (g_stop || training_failed) break;
        /* average gradients across GPUs: sum on GPU 0 (peer copies), scale, broadcast */
        if (ng > 1) {
            nn_init(G[0].dev);
            for (int g = 1; g < ng; g++) { nn_peer_copy(gpeer, G[0].dev, unet_grad_ptr(G[g].u), G[g].dev, np * 4); nn_axpy(unet_grad_ptr(G[0].u), 1.f, gpeer, np); }
            if (!split) nn_scale(unet_grad_ptr(G[0].u), 1.f / ng, np);   /* split: the halves' gradients add up to the window's */
            for (int g = 1; g < ng; g++) nn_peer_copy(unet_grad_ptr(G[g].u), G[g].dev, unet_grad_ptr(G[0].u), G[0].dev, np * 4);
        }
        for (int g = 0; g < nl; g++) { nn_init(G[g].dev); loss += fetch_loss(&G[g], B, P, dice_w, parts[g]); active += parts[g][2 * NCH]; }   /* split: both sides hold the window's loss */
        if (sheet_log) { fprintf(sheet_log,"%d",step); for (int k=0;k<5;k++) fprintf(sheet_log,",%.8g",G[0].sheet_parts[k]); fprintf(sheet_log,",%.8g,%.8g\n",G[0].sheet_loss,g_sheet_ramp); fflush(sheet_log); }
        int fwd_nan = 0;   /* non-finite loss parts: the forward itself produced non-finite logits (not a gradient-scale overflow) */
        if (g_sheet && !isfinite(G[0].sheet_loss)) fwd_nan=1;
        for (int g = 0; g < nl; g++) for (int c = 0; c < 2 * NCH; c++) if (!isfinite(parts[g][c])) fwd_nan = 1;
        for (int g = 0; g < nl; g++)   /* the heads' own losses (task unwrap / unwrap2 have no recto parts in parts[]) */
            if (!isfinite(G[g].side_loss) || !isfinite(G[g].norm_loss) || !isfinite(G[g].ink_loss) || !isfinite(G[g].w4_loss) || !isfinite(G[g].count_loss)) fwd_nan = 1;
        if (fwd_nan && !overfit && !split && !g_restore) diagnose_nan(&G[0], B, P, step, out);
        float lr;
        if (time_horizon) {
            double elapsed = fmin(schedule_elapsed + now() - t0, time_horizon);
            if (elapsed < warmup_seconds) lr = lr0 * (float)(elapsed / warmup_seconds);
            else if (!strcmp(sched, "wsd")) {
                double cd0 = time_horizon * (1.0 - cooldown);
                lr = elapsed < cd0 ? lr0 : lr0 * (lr_floor + (1.f - lr_floor) * (float)((time_horizon - elapsed) / (time_horizon - cd0)));
            } else lr = lr0 * (lr_floor + (1.f - lr_floor) * 0.5f * (1.f + cosf(3.14159265f * (float)((elapsed - warmup_seconds) / (time_horizon - warmup_seconds)))));
        }
        else {
            int ss = step - schedule_start, total = steps - schedule_start;
            if (ss <= warmup) lr = lr0 * (float)ss / warmup;
            else if (!strcmp(sched, "wsd")) { int cd0 = (int)(total * (1.f - cooldown)); lr = ss < cd0 ? lr0 : lr0 * (lr_floor + (1.f - lr_floor) * (float)(total - ss) / (float)(total - cd0 > 0 ? total - cd0 : 1)); }
            else lr = lr0 * (lr_floor + (1.f - lr_floor) * 0.5f * (1.f + cosf(3.14159265f * (float)(ss - warmup) / (float)(total - warmup))));
        }
        if (g_ds && g_ds_anneal) {   /* deep supervision weights: linearly to 0 over the run (the time horizon when there is one) */
            const double pr = time_horizon ? fmin(1.0, (schedule_elapsed + now() - t0) / time_horizon)
                                           : (double)(step - schedule_start) / (double)(steps - schedule_start > 0 ? steps - schedule_start : 1);
            g_ds_frac = (float)fmax(0.0, 1.0 - pr);
        }
        if (soft_end != sc.soft) sampler_set_soft(sp, sc.soft + (soft_end - sc.soft) * (step < steps ? (float)step / (float)steps : 1.f));
        double gn = 0;
        for (int g = 0; g < ng; g++) {
            gpu_state *d = &G[g];
            nn_init(d->dev);
            if (prof) nn_prof_begin(7);
            if (g == 0) gn = unet_grad_norm(d->u);          /* identical on every GPU after averaging */
            if (plan && (!isfinite(gn) || fwd_nan)) {
                fprintf(stderr, "cover tile %llu has non-finite forward/gradients; aborting without advancing its cursor\n", (unsigned long long)coverage.cursor);
                training_failed = 1; break;
            }
            if (!isfinite(gn) || fwd_nan) {                   /* skip the step; a 16-bit gradient overflow also halves the gradient scale */
                if (g == 0) {
                    if (fwd_nan) fprintf(stderr, "step %d: non-finite forward, skipping (grad scale kept at %g)\n", step, nn_get_grad_scale());
                    else { float sc = nn_get_grad_scale(); nn_set_grad_scale(sc > 1.f ? sc * 0.5f : 1.f); fprintf(stderr, "step %d: non-finite gradient norm, skipping; grad scale -> %g\n", step, nn_get_grad_scale()); }
                    for (int q = 0; q < nl; q++) window_note(&G[q], S, "  window");
                    nskip++;
                }
                continue;
            }
            if (g_head_only) unet_grad_head_only(d->u, g_head_only - 1);   /* --train-head-only N: head + last N-1 decoder blocks (frozen rest: --wd 0) */
            if (clip > 0 && gn > clip) unet_clip_grad(d->u, clip);
            unet_srste24(d->u, srste);                       /* no-op unless sparse */
            if (use_anvil) unet_anvil(d->u, anvil_lr * (lr / lr0), anvil_wd, step, steps, lr, 0.9f, 0.999f, 1e-8f, wd);
            else if (use_muon) unet_muon(d->u, muon_lr * (lr / lr0), muon_beta, lr, 0.9f, 0.999f, 1e-8f, wd, step);   /* Muon lr follows the same schedule shape */
            else unet_adamw(d->u, lr, 0.9f, 0.999f, 1e-8f, wd, step);
            unet_ema(d->u, step < 100 ? 0.9f : ema);
            unet_wquant(d->u, (unsigned)step);              /* no-op unless --wq */
            if (prof) nn_prof_end();
        }
        for (int g = 0; g < ng; g++) if (G[g].tail_due) { nn_init(G[g].dev); upload_tail(&G[g], B, P); }   /* (upsplit: after the loss / gradient-norm
                                                                                                           fetches' small copies, during the optimiser) */
        if (training_failed) break;
        committed_step = step;
        if (plan) { coverage.cursor = (uint64_t)(step - coverage.base_step); if (cover_checkpoint_extra(runtime_extra, &coverage, checkpoint_extra, sizeof checkpoint_extra)) return 2; }
        const char *e = nn_check();
        if (e) { fprintf(stderr, "cuda error at step %d: %s\n", step, e); for (int q = 0; q < nl; q++) window_note(&G[q], S, "  window"); return 1; }
        if ((seconds && now() - t0 >= seconds) || (schedule_seconds && schedule_elapsed+now()-t0>=schedule_seconds) ||
            (limit_seconds && now() - t0 >= limit_seconds) || (stop_at && step >= stop_at)) g_stop = 1;
        acc_loss += loss / nl; acc_bce += parts[0][0]; acc_dice += parts[0][g_aff || g_unwrap || g_heads ? NCH : g_half ? g_cout : cfg.cout]; acc_g += gn; nacc++;
        if (g_verso_on) { acc_vbce += G[0].vout[0]; acc_vdice += G[0].vout[1]; }
        if (g_side) acc_side += G[0].sout[0];
        if (g_unwrap) { acc_ink += G[0].iout[0]; acc_norm += G[0].norm_loss / (g_norm_lambda ? g_norm_lambda : 1.f); }
        if (g_unwrap2) { acc_w4 += G[0].w4h[0]; acc_w4w += G[0].w4h[5]; acc_w4a += G[0].w4h[1]; } acc_pg += G[0].pg_loss;
        if (g_ds) acc_ds += G[0].ds_loss;
        if (g_distill) { acc_emb += G[0].emb_loss; acc_embc += G[0].emb_cells; }
        if (g_cond) { acc_cond += G[0].cond_loss / (g_cond_lambda ? g_cond_lambda : 1.f); acc_ck += (double)G[0].cacc_h[0]; acc_cr += (double)G[0].cacc_h[1]; }
        if (g_pap_head) acc_pap += G[0].pout3[0];
        if (g_restore) { acc_rn += G[0].rst_h[0]; acc_re += G[0].rst_h[3]; acc_rb += G[0].rst_h[4]; }
        if (g_count_lambda > 0) { acc_cnt += G[0].cnt_h[0]; acc_cnr += G[0].cnt_h[2]; }
        if (step % log_every == 0 || step == steps || g_stop) {
            double dt = now() - tlog;
            double vl = -1, vb = 0, vd = 0, vh[TCMAX] = {0}; int vhn[TCMAX] = {0}; double vw4[6] = {0};   /* unwrap2: wind4 bce, worst-gauge bce, right turns, voxels, phase, normal */
            char vstr[128] = ",,";
            double rv_n[RF_N] = {0}, rv_e[RF_N] = {0}, rv_b[RF_N] = {0}, rv_l = 0;   /* restore validation: per family voxels, error^2, input error^2 */
            if (g_rpairs) { free(g_rpv); g_rpv = calloc((size_t)S->n * RP_NST, sizeof *g_rpv); }
            if (step % val_every == 0 || step == steps || g_stop) {
                g_ptrunc_on = 0;   /* (validation: the plain phase loss) */
                const int nv = split ? 2 : 1;   /* split: validation windows are split too */
                for (int g = 0; g < nv; g++) { nn_init(G[g].dev); unet_use_ema(G[g].u, !val_live); }   /* (--val-live 1: the live weights) */
                {   /* validation on the buffer not holding the prefetched batch (its upload has completed: ev_up synced above) */
                    int keep = G[0].cur;
                    for (int g = 0; g < nv; g++) { if (G[g].lean) { nn_init(G[g].dev); nn_event_sync(G[g].ev_up[keep]); } select_buf(&G[g], keep ^ 1); }   /* lean: one buffer, the prefetch must have landed */
                    g_verso_flip = 0;   /* validation: the direction as is */
                    if (cfg.l0_norm == 2 && !val_live && l0_recal > 0 && unet_l0_updates(G[0].u) > 0) {   /* ema: the EMA weights' level-0 statistics from
                                                         l0_recal training windows (own deterministic sampler: the training sequence unchanged) */
                        sample_cfg rc = vc; rc.holdout = 0; rc.seed = seed + 999; rc.nworkers = 2;
                        sampler *rq = sampler_start(S, &rc);
                        if (!rq) { fprintf(stderr, "cannot construct the recalibration sampler\n"); return 1; }
                        for (int g = 0; g < nv; g++) { nn_init(G[g].dev); unet_l0_recal(G[g].u, 1); }
                        for (int i = 0; i < l0_recal; i++) {
                            batch *rb = sampler_next(rq);
                            if (!rb) break;
                            for (int g = 0; g < nv; g++) { nn_init(G[g].dev); upload(&G[g], rb, B, P); }
                            if (split) { split_arg sa = {G, B, P, 0, dice_w}; split_run(sctx, split_job, &sa); nn_init(G[0].dev); }
                            else run_batch(&G[0], B, P, dice_w, 0);
                            sampler_release(rq, rb);
                        }
                        sampler_stop(rq);
                        for (int g = 0; g < nv; g++) { nn_init(G[g].dev); unet_l0_recal(G[g].u, 0); }
                        nn_init(G[0].dev);
                    }
                    sampler *vq = nval > 0 && vmode == VAL_REGEN ? sampler_start(S, &vc) : nullptr;   /* --val-store regen: the same windows drawn again */
                    if (nval > 0 && vmode == VAL_REGEN && !vq) { fprintf(stderr, "cannot construct validation sampler\n"); return 1; }
                    /* --val-P: the validation copy of the network (its own N^3 inference build, the training net's current weights) on GPU 0's
                       batch buffers (larger than N^3 needs; the lean layouts' prefetched batch goes up again below, as for the plain pass) */
                    gpu_state Gv;
                    if (g_val_u) {
                        nn_init(G[0].dev);
                        nn_event_sync(G[0].ev_up[0]); nn_event_sync(G[0].ev_up[1]); nn_sync();   /* (no upload into the shared buffers in flight) */
                        Gv = G[0]; Gv.u = g_val_u; Gv.lean = 0; Gv.upsplit = 0; Gv.hsplit = 0; Gv.gl_inplace = 0; Gv.gl_inside = 0; Gv.side = -1; Gv.pending = nullptr; Gv.tail_due = nullptr;
                        unet_use_params(g_val_u, unet_fwd_params(G[0].u));
                        select_buf(&Gv, Gv.cur);   /* (the radial records of its forward) */
                        g_Pz = vP;
                    }
                    int got = 0;
                    for (int i = 0; i < nval; i++) {
                        float vp[2 * TCMAX + 1];
                        batch *vbt = vq ? sampler_next(vq) : vmode == VAL_DISK ? (val_read(&vdk, i, B, p3v) ? nullptr : &vdk.buf) : &val[i];
                        if (!vbt) { if (vmode == VAL_DISK) { fprintf(stderr, "cannot read the validation windows\n"); return 1; } break; }
                        double l;
                        if (g_val_u) {   /* --val-P: the validation copy on GPU 0 (plain cube windows and losses only) */
                            static int reported;
                            const size_t f0 = reported ? 0 : nn_mem_free();
                            upload(&Gv, vbt, B, vP); run_batch(&Gv, B, vP, dice_w, 0); l = fetch_loss(&Gv, B, vP, dice_w, vp);
                            if (!reported++) fprintf(stderr, "memory: the --val-P validation network's build took %.2f GB (%.2f reserved)\n",
                                                     f0 > nn_mem_free() ? (f0 - nn_mem_free()) / 1e9 : 0.0, g_val_reserve / 1e9);
                        } else {
                            for (int g = 0; g < nv; g++) { nn_init(G[g].dev); upload(&G[g], vbt, B, P); }
                            if (split) { split_arg sa = {G, B, P, 0, dice_w}; split_run(sctx, split_job, &sa); nn_init(G[0].dev); }
                            else if (g_restore && g_rpairs) {   /* real pairs: per source sums */
                                run_batch(&G[0], B, P, dice_w, 0); rv_l = restore_fetch(&G[0], vp);
                                const int si_ = vbt->src ? vbt->src[0] : 0;
                                if (si_ >= 0 && si_ < S->n) for (int k = 0; k < RP_NST; k++) g_rpv[(size_t)si_ * RP_NST + k] += G[0].rst_h[k];
                            } else if (g_restore) {   /* every family on the same window, with fixed recipes */
                                double ls = 0;
                                for (int f = 0; f < RF_N; f++) {
                                    restore_recipe_draw(&G[0].rcp, f, rmix(0xA11CE5ull + vc.seed * 1315423911ull + (uint64_t)i * 64 + (uint64_t)f), g_Pz, P, P, &g_rr);
                                    run_batch(&G[0], B, P, dice_w, 0); ls += restore_fetch(&G[0], vp);
                                    rv_n[f] += G[0].rst_h[0]; rv_e[f] += G[0].rst_h[3]; rv_b[f] += G[0].rst_h[4];
                                }
                                rv_l = ls / RF_N;
                            } else run_batch(&G[0], B, P, dice_w, 0);
                            l = g_restore ? rv_l : fetch_loss(&G[0], B, P, dice_w, vp);
                        }
                        vl = (vl < 0 ? 0 : vl) + l; vb += vp[0];
                        if (g_unwrap2) { vw4[0] += G[0].w4h[0]; vw4[1] += G[0].w4h[5]; vw4[2] += G[0].w4h[1] * G[0].w4h[2]; vw4[3] += G[0].w4h[2]; vw4[4] += G[0].side_loss / (g_side_lambda ? g_side_lambda : 1.f); vw4[5] += G[0].norm_loss / (g_norm_lambda ? g_norm_lambda : 1.f); } vd += vp[g_aff || g_unwrap || g_heads ? NCH : g_half ? g_cout : cfg.cout];
                        if (g_heads) for (int h = 0; h < g_heads; h++) if (G[0].hon[h]) { vh[h] += G[0].hl[h]; vhn[h]++; }
                        if (vq) sampler_release(vq, vbt);
                        got++;
                    }
                    if (g_val_u) { g_Pz = Pz_tr; unet_use_params(g_val_u, nullptr); }
                    if (vq) {
                        const int vfail = sampler_failed(vq);
                        sampler_stop(vq);
                        if (vfail) { fprintf(stderr, "validation sampler failed\n"); return 1; }
                        if (got < nval) { fprintf(stderr, "validation: %d of %d batches drawn\n", got, nval); nval = got; }
                    }
                    for (int g = 0; g < nv; g++) {
                        select_buf(&G[g], keep);
                        batch *pb = split ? G[0].pending : G[g].pending;
                        if (G[g].lean && pb) { nn_init(G[g].dev); upload(&G[g], pb, B, P); }   /* one buffer: the prefetched batch was overwritten */
                        G[g].sheet=pb && pb->sheet?pb->sheet[0]:nullptr;
                    }
                }
                for (int g = 0; g < nv; g++) { nn_init(G[g].dev); unet_use_ema(G[g].u, 0); }
                nn_init(G[0].dev);
                if (nval) { vl /= nval; vb /= nval; vd /= nval; }
                if (g_unwrap2 && nval) fprintf(stderr, "val unwrap2: phase bce %.5f normal bce %.5f wind4 bce %.5f (worst gauge %.5f) turn acc %.4f over %.0f voxels\n",
                                               vw4[4] / nval, vw4[5] / nval, vw4[0] / nval, vw4[1] / nval, vw4[3] > 0 ? vw4[2] / vw4[3] : 0.0, vw4[3]);
                vcrit = vl;
                if (g_restore && g_rpairs && nval) restore_pair_report(S, out, step, &vcrit);
                else if (g_restore && nval) {   /* PSNR of output and input against the clean window per family; best.ckpt on the mean gain of the degraded families */
                    double gs = 0; int ng_ = 0; char rl[600]; size_t k = 0; rl[0] = 0;
                    char cp_[1400]; snprintf(cp_, sizeof cp_, "%s/restore_val.csv", out);
                    FILE *rf = fopen(cp_, "a");
                    if (rf) { fseek(rf, 0, SEEK_END); if (!ftell(rf)) { fprintf(rf, "step"); for (int f = 0; f < RF_N; f++) fprintf(rf, ",%s_psnr,%s_input_psnr,%s_gain", restore_family_names[f], restore_family_names[f], restore_family_names[f]); fprintf(rf, ",mean_gain\n"); } fprintf(rf, "%d", step); }
                    for (int f = 0; f < RF_N; f++) {
                        const double p = psnr_of(rv_e[f], rv_n[f]), pb = psnr_of(rv_b[f], rv_n[f]);
                        if (f != RF_IDENTITY && g_rr.mix[f] > 0) { gs += p - pb; ng_++; }
                        k += (size_t)snprintf(rl + k, sizeof rl - k, " %s %.3f dB (input %.3f, gain %+.3f)", restore_family_names[f], p, pb, p - pb);
                        if (rf) fprintf(rf, ",%.4f,%.4f,%.4f", p, pb, p - pb);
                    }
                    if (ng_) gs /= ng_;
                    vcrit = -gs;
                    if (rf) { fprintf(rf, ",%.4f\n", gs); fclose(rf); }
                    fprintf(stderr, "val restore step %d:%s; mean gain %+.4f dB\n", step, rl, gs);
                }
                snprintf(vstr, sizeof vstr, "%.5f,%.5f,%.5f", vl, vb, vd);
                if (g_heads) {   /* validation loss per head over the windows of its own sources */
                    fprintf(stderr, "val heads:");
                    for (int h = 0; h < g_heads; h++) fprintf(stderr, " %d %.5f (%d)", h, vhn[h] ? vh[h] / vhn[h] : 0.0, vhn[h]);
                    fprintf(stderr, "\n");
                }
                if (nval && vcrit < best_val) { best_val = vcrit; char bp[1400]; snprintf(bp, sizeof bp, "%s/best.ckpt", out); if (unet_save(G[0].u, bp, step, checkpoint_extra)) { fprintf(stderr, "cannot save %s\n", bp); return 1; } }
            }
            char affs[320] = "";
            if (g_verso) snprintf(affs, 64, " verso bce %.4f dice %.4f", acc_vbce / nacc, acc_vdice / nacc);
            if (g_side) { size_t na = strlen(affs); snprintf(affs + na, sizeof affs - na, " %s bce %.4f", g_side == 2 ? "phase" : "side", acc_side / nacc); }
            if (g_unwrap2) { size_t na = strlen(affs); snprintf(affs + na, sizeof affs - na, " normal bce %.4f wind4 bce %.4f (worst gauge %.4f) acc %.3f", acc_norm / nacc, acc_w4 / nacc, acc_w4w / nacc, acc_w4a / nacc); if (g_pg_lambda > 0) { na = strlen(affs); snprintf(affs + na, sizeof affs - na, " pgrad %.4f", acc_pg / nacc); } }
            if (g_ptrunc_tau > 0) { unsigned long long pc[2]; nn_phase_trunc_counts(pc); size_t na = strlen(affs); snprintf(affs + na, sizeof affs - na, " trunc %.4f", pc[0] ? (double)pc[1] / pc[0] : 0.0); }
            else if (g_unwrap) { size_t na = strlen(affs); snprintf(affs + na, sizeof affs - na, " ink bce %.4f normal bce %.4f", acc_ink / nacc, acc_norm / nacc); }
            if (g_ds) { size_t na = strlen(affs); snprintf(affs + na, sizeof affs - na, " ds %.4f", acc_ds / nacc); }
            if (g_distill) { size_t na = strlen(affs); snprintf(affs + na, sizeof affs - na, " distill %.4f (%.1f%% cells)", acc_emb / nacc, 100. * acc_embc / nacc / ((double)g_distill_n / g_distill));
                if (g_distill_pap > 0) { static size_t pc0, pk0; const size_t pc = atomic_load(&g_emb_pap_cleared), pk = atomic_load(&g_emb_pap_kept); na = strlen(affs);
                    if (pc + pk > pc0 + pk0) snprintf(affs + na, sizeof affs - na, " (papyrus gate kept %.1f%% of the targeted)", 100. * (double)(pk - pk0) / (double)(pc + pk - pc0 - pk0)); pc0 = pc; pk0 = pk; } }
            if (g_cond) { size_t na = strlen(affs); snprintf(affs + na, sizeof affs - na, " cond bce %.4f acc %.3f", acc_cond / nacc, acc_ck > 0 ? acc_cr / acc_ck : -1.0); }
            if (g_pap_head) { size_t na = strlen(affs); snprintf(affs + na, sizeof affs - na, " papyrus bce %.4f", acc_pap / nacc); }
            if (g_restore) { size_t na = strlen(affs); const double p = psnr_of(acc_re, acc_rn), pb = psnr_of(acc_rb, acc_rn);
                             snprintf(affs + na, sizeof affs - na, " psnr %.2f input %.2f gain %+.3f (%.0f Mvox/s)", p, pb, p - pb, (double)nacc * g_Pz * P * P / 1e6 / dt); acc_rn = acc_re = acc_rb = 0; }
            if (g_count_lambda > 0) { size_t na = strlen(affs); snprintf(affs + na, sizeof affs - na, " count %.4f (turns pred/true %.3f)", acc_cnt / nacc, acc_cnr / nacc); }
            if (g_aff) { const float *a = G[0].aff_out; size_t na = strlen(affs); snprintf(affs + na, sizeof affs - na, " aff %.4f (d1 bce %.3f dice %.3f | d8 bce %.3f dice %.3f)", G[0].aff_loss,
                                    (a[0] + a[6] + a[12]) / 3, (a[1] + a[7] + a[13]) / 3, (a[18] + a[24] + a[30]) / 3, (a[19] + a[25] + a[31]) / 3); }
            fprintf(stderr, "step %6d lr %.2e loss %.4f bce %.4f dice %.4f%s gn %.2f %.2f samp/s (wait %.0f%%)%s%s\n", step, lr, acc_loss / nacc, acc_bce / nacc, acc_dice / nacc, affs, acc_g / nacc,
                    (double)nacc * B * nl / dt, 100 * wait / dt, vl >= 0 ? " val " : "", vl >= 0 ? vstr : "");
            { static int memlog = -1; if (memlog < 0) memlog = ufsm_env_on("UFSM_SYNC_CHECK");   /* device memory free per GPU at each log line (a leak) */
              if (memlog) { fprintf(stderr, "  device memory free:"); for (int g = 0; g < ng; g++) { nn_init(G[g].dev); fprintf(stderr, " GPU %d %.1f MB", G[g].dev, nn_mem_free() / 1e6); } fprintf(stderr, "\n"); nn_init(G[0].dev); } }
            if (log) { fprintf(log, "%d,%.3e,%.5f,%.5f,%.5f,%.0f,%.4f,%s,%.3f,%.3f\n", step, lr, acc_loss / nacc, acc_bce / nacc, acc_dice / nacc, active, acc_g / nacc, vstr, (double)nacc * B * nl / dt, wait); fflush(log); }
            if (prof) { fprintf(stderr, "per-op GPU ms over the last %d steps (all GPUs):\n", log_every); unet_prof_report(); }
            acc_emb = acc_embc = 0; acc_ds = 0; acc_loss = acc_bce = acc_dice = acc_g = acc_vbce = acc_vdice = acc_side = acc_ink = acc_norm = acc_cond = acc_ck = acc_cr = acc_pap = acc_cnt = acc_cnr = acc_w4 = acc_w4w = acc_w4a = acc_pg = 0; nacc = 0; tlog = now(); wait = 0;
        }
        if (step % ckpt_every == 0 || step == steps || g_stop) {
            nn_init(G[0].dev);
            char cp[1400]; snprintf(cp, sizeof cp, "%s/last.ckpt", out);
            if (unet_save(G[0].u, cp, step, checkpoint_extra)) { fprintf(stderr, "cannot save %s\n", cp); return 1; }
        }
        if (step == step0 + 1 || step == steps || g_stop) {
            char mp[1400], em[4096]; snprintf(mp, sizeof mp, "%s/precision.txt", out); WRITE_MANIFEST(mp);
            if (step == step0 + 1) { nn_exec_manifest(em, sizeof em); fprintf(stderr, "%s\n", em); }
        }
    }
    /* Signals can arrive between log/checkpoint boundaries; save only completed optimizer updates. */
    nn_init(G[0].dev);
    { char cp[1400]; snprintf(cp, sizeof cp, "%s/last.ckpt", out); if (unet_save(G[0].u, cp, committed_step, checkpoint_extra)) return 1; }
    if (plan) fprintf(stderr, "cover progress %llu/%llu committed tiles\n", (unsigned long long)coverage.cursor, (unsigned long long)coverage.count);
    uint64_t prod, rej; sampler_stats(sp, &prod, &rej);
    uint64_t why[4]; sampler_reject_reasons(sp, why);
    fprintf(stderr, "done in %.0fs; sampler produced %llu, rejected %llu (coarse probe %llu, after the CT read %llu, empty targets %llu; holdout overlaps %llu)\n",
            now() - t0, (unsigned long long)prod, (unsigned long long)rej, (unsigned long long)why[0], (unsigned long long)why[1], (unsigned long long)why[2], (unsigned long long)why[3]);
    if (sc.crush.p > 0) fprintf(stderr, "crush augmentation: %llu crushed windows\n", (unsigned long long)sampler_crushed(sp));
    {   static const char *an[AUGN_N] = {"xscan", "tear", "delam", "sheet-density", "curv", "inclusions"};
        for (int k = 0; k < AUGN_N; k++) if (sampler_aug_count(sp, k)) fprintf(stderr, "%s augmentation: %llu windows\n", an[k], (unsigned long long)sampler_aug_count(sp, k)); }
    if (g_mem) { long h = 0, mi = 0; for (int k = 0; k < 8; k++) { h += g_mb[k].hits; mi += g_mb[k].misses; }
                 fprintf(stderr, "window memory: %ld chain windows with their predecessor's memory, %ld without (evicted / out of order)\n", h, mi); }
    sampler_prof_print(sp);   /* UFSM_SAMPLER_PROF=1: per-stage cpu ms per patch */
    int sampling_failed = sampler_failed(sp);
    sampler_stop(sp);
    if (log) fclose(log);
    if (sheet_log) fclose(sheet_log);
    cover_free(plan);
    sheet_free(g_sheet); g_sheet=nullptr;
    return sampling_failed || training_failed ? 1 : 0;
}
