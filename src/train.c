/* ufsm train: sampler -> pinned batches -> one or more GPUs (data parallel, gradients averaged through the
   host) -> UNet -> BCE+Dice -> AdamW/EMA, with held-out validation batches, CSV log and checkpoints. */
#include "nn.h"
#include "checkpoint.h"
#include "sample.h"
#include "papyrus.h"
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

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}

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
    void *ev_up[2], *ev_done[2]; /* upload into buffer i finished; the compute that last used buffer i finished */
    batch *pending;              /* host batch whose upload into buffer (cur ^ 1) is in flight */
    int lean;                    /* one batch buffer; logit gradient in the model's gradient buffer B */
    int gl_inside;               /* lean: the logit gradient lives in the model's gradient buffers (setup checked the batch overlap) */
    char *bat_lo, *bat_hi;       /* lean: the batch's bytes inside the model's buffers (a logit gradient placed later must avoid them) */
    int gl_inplace;              /* the logit gradient over the fp16 logits (g_gl_inplace) */
    void *hs_lg; double *hs_ds; float hs_dice; int hs_z, hs_P;   /* head slabs (g_head_slab): fp16 logits of one slab, loss sums, recto dice, planes per slab, window width */
    int64_t cornb[2][3]; int lvlb[2];   /* level-0 corner / level of the batch in each buffer (verso dump) */
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
    uint8_t *wdb[2], *wd, *ct4, *cm4, *cw4; float *cscratch, cout4[9]; double cond_loss;   /* --cond: the gauge-shifted winding per voxel, its 4 targets, mask, weights; scratch and parts */
    unsigned long long *cacc, cacc_h[2];   /* --cond: known / right-turn voxel counts of the 14-turn head (device, last batch on the host) */
    uint8_t *papb[2], *pap, *pt1, *pmk1, *pw1; float *pscr, pout3[3]; double pap_loss;   /* --pap-head: labels (double-buffered), target, mask, weight; scratch, parts */
    void *x2, *glh, *lrt; float *lgf;  /* --half-res: the network's input and logit gradient, the intermediate grid (div 4), the full-resolution fp32 logits */
    uint8_t *ds_t[UNET_MAXLEV], *ds_m[UNET_MAXLEV]; float *ds_gl[UNET_MAXLEV], *ds_scr[UNET_MAXLEV], ds_out[UNET_MAXLEV][2 * 8 + 2]; double ds_loss; int ds_fresh;   /* --deep-sup (fresh: the last batch computed them) */
} gpu_state;
/* task band_affinity: output channel 0 = recto (dense loss on target channel 0), channels 1..K = affinities for g_aff_off */
static int g_half, g_hdiv = 1;   /* --half-res: 1 pool, 2 pack, 3 pool4, 4 pack4 (unet_cfg.half_res), g_hdiv its factor; the losses see g_cout
                                    full-resolution channels */
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
static int g_unwrap, g_norm_ch, g_norm_s = 4; static float g_ink_lambda = 1.f, g_ink_dice = 0.f, g_norm_lambda = 1.f, g_dens_scale = 32.f;
static int g_dens_code; static float g_packed_w, g_packed_um = 120.f;   /* --density-code, --packed-weight, --packed-um (task unwrap) */
static int g_gl_inplace;   /* the logit gradient over the fp16 logits (set before the memory planner) */
static int g_head_slab;    /* unwrap: head, losses and head backward per z-slab (no whole-window logits or derived targets; set before the planner) */
static int g_hs_z;         /* its planes per slab */
static int g_ds; static float g_ds_w = 0.5f, g_ds_frac = 1.f; static int g_ds_anneal = 1;   /* --deep-sup K: decoder levels 1..K, weight w^j x frac */
static float g_count_lambda; static int g_count_K = 1024, g_count_L = 64;   /* --count-lambda / -segments / -length: the count loss (nn_count_loss) */
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
static float g_verso_swap, g_verso_lambda = 1.f;   /* lambda: verso loss weight (its gradient and its share of the reported loss) */ static int g_verso_flip, g_head_only;   /* swap: probability of a step with the direction reversed and recto/verso targets exchanged */
static int g_verso, g_verso_start, g_verso_on, g_verso_self, g_verso_zero, g_verso_ridge, g_verso_anchor;   /* anchor: recto voxels without a trusted label take the teacher's normal-direction output */ static float g_verso_hard;   /* ridge: thin verso targets, the teacher's ridge along the direction at >= this byte */   /* hard: binary verso targets at this teacher probability; zero: fresh verso head row */   /* self: targets from the live weights (collapses: the net learns to ignore the direction) */
static sheet_dataset *g_sheet;
static float g_sheet_ramp;
static int g_sheet_variant=2;
/* --split z: both GPUs work on the same window, each on its z half plus g_h0 halo planes (depth g_Dl); the mask is zeroed on
   the halo planes (from a pinned zero buffer) so each voxel counts once */
static int g_Dl, g_h0; static uint8_t *g_zeros;
static int g_Pz;   /* window depth: --Pz (a Pz x P x P slab), else P */

static int g_xfmt = 0, g_g16 = 0;   /* input batches uploaded as 16-bit (1 fp16, 2 bf16); loss gradient written as 16-bit */
static size_t xbytes(void) { return g_xfmt ? 2 : 4; }
static const void *bx(const batch *b) { return g_xfmt ? (const void *)b->x16 : (const void *)b->x; }
/* --gpu-input (default when the configuration allows it, sample_gpu_input_ok): batches carry the CT channel only (g_xs 1) and per sample a
   radial record; the network's 4 MX input channels are built on the device (nn_input_rad). g_xs: input channels per sample in the batches */
static int g_gpu_in, g_xs = 4, g_gpu_want, g_gpu_asked, g_pack;   /* g_pack: --pack-targets (batch.tm, nn_unpack_tm on upload) */
static size_t rad_bytes(int B) { return g_gpu_in ? (size_t)B * NN_RAD_REC(g_Pz) : 0; }
/* once the memory mode is set: the GPU-built input needs the 16-bit -> MX input conversion (unet_input_converted: MX activations with
   MX gradients or an input policy) */
static void gpu_input_settle(sample_cfg *sc) {
    g_gpu_in = g_gpu_want && unet_input_converted();
    g_xs = g_gpu_in ? 1 : g_xc; sc->gpu_input = g_gpu_in;
    if (g_gpu_in) fprintf(stderr, "input: the CT channel uploaded, the radial channels built on the GPU (--gpu-input 0: host-built)\n");
    else if (g_gpu_asked) fprintf(stderr, "--gpu-input: not available here (needs the 16-bit -> MX input conversion, whole windows, the 4 plain inputs, no warps / sheet prior); host-built input\n");
}
static void select_buf(gpu_state *d, int i) {
    d->rad = d->radb[i]; if (g_gpu_in) unet_set_input_rad(d->u, d->rad);
    d->cur = i; d->x = d->xb[i]; d->t = d->tb[i]; d->m = d->mb[i]; d->w = d->wb[i]; d->band = d->bandb[i]; d->sdb = d->sideb[i]; d->pm = d->pmb[i]; d->wd = d->wdb[i]; d->pap = d->papb[i]; d->um = d->umb[i]; }
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
/* host memory this process may still take: MemAvailable, or less under a cgroup v2 memory.max (capped.sh's slice / scope); 0 unknown */
static size_t host_mem_available(void) {
    size_t avail = 0; char line[512];
    FILE *f = fopen("/proc/meminfo", "r");
    if (f) { while (fgets(line, sizeof line, f)) if (!strncmp(line, "MemAvailable:", 13)) { avail = (size_t)strtoull(line + 13, nullptr, 10) << 10; break; } fclose(f); }
    char cg[400] = "";
    if ((f = fopen("/proc/self/cgroup", "r"))) { while (fgets(line, sizeof line, f)) if (!strncmp(line, "0::", 3)) { snprintf(cg, sizeof cg, "%s", line + 3); cg[strcspn(cg, "\n")] = 0; } fclose(f); }
    for (char *e = cg + strlen(cg); *cg; ) {   /* this cgroup and its parents: the tightest memory.max - memory.current */
        char p[600]; unsigned long long mx = 0, cur = 0; int ok = 1;
        snprintf(p, sizeof p, "/sys/fs/cgroup%s/memory.max", cg);
        if ((f = fopen(p, "r"))) { ok = fscanf(f, "%llu", &mx) == 1; fclose(f); } else ok = 0;
        snprintf(p, sizeof p, "/sys/fs/cgroup%s/memory.current", cg);
        if (ok && (f = fopen(p, "r"))) { ok = fscanf(f, "%llu", &cur) == 1; fclose(f); } else ok = 0;
        if (ok && mx > cur && (!avail || mx - cur < avail)) avail = (size_t)(mx - cur);
        while (e > cg && *e != '/') e--;
        if (e == cg) break;
        *e = 0;
    }
    return avail;
}
/* the host arrays of a batch in this run's layout: the addresses of its members and their sizes (validation copies, --val-store) */
enum { VAL_DISK, VAL_RAM, VAL_REGEN, VAL_NF = 16 };
static int val_fields(batch *b, int B, size_t p3, void **f[VAL_NF], size_t sz[VAL_NF]) {
    int n = 0;
#define VF(p_, s_) do { f[n] = (void **)&(p_); sz[n++] = (s_); } while (0)
    if (g_xfmt) VF(b->x16, (size_t)B * g_xs * p3 * 2); else VF(b->x, (size_t)B * g_xs * p3 * 4);
    if (g_gpu_in) VF(b->rad, rad_bytes(B));
    if (g_pack) VF(b->tm, (size_t)B * p3); else { VF(b->t, (size_t)B * NCH * p3); VF(b->m, (size_t)B * p3); }
    VF(b->w, (size_t)B * NCH); VF(b->src, (size_t)B * sizeof *b->src); VF(b->um, (size_t)B * sizeof *b->um);
    if (g_aff) VF(b->band, (size_t)B * p3);
    if (g_side) VF(b->side, (size_t)B * p3);
    if (g_unwrap) VF(b->pm, (size_t)B * p3);
    if (g_cond) VF(b->wind, (size_t)B * p3);
    if (g_pap_head) VF(b->pap, (size_t)B * p3);
#undef VF
    return n;
}
typedef struct { int fd; size_t bytes; batch buf; } valdisk;   /* --val-store disk: the windows back to back in an unlinked file */
static int pwrite_all(int fd, const void *p, size_t n, size_t off) {
    for (size_t k = 0; k < n;) { const ssize_t w = pwrite(fd, (const char *)p + k, n - k, (off_t)(off + k)); if (w <= 0) return -1; k += (size_t)w; }
    return 0;
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
    if (g_unwrap) for (int n = 0; n < B; n++) {   /* the papyrus mask, halo planes zeroed like the label mask */
        uint8_t *pm = d->pmb[i] + n * l3;
        if (d->side) { cp(pm, g_zeros, g_h0 * p2); cp(pm + g_h0 * p2, b->pm + n * p3 + (z0 + g_h0) * p2, own); }
        else { cp(pm, b->pm + n * p3, own); cp(pm + own, g_zeros, g_h0 * p2); }
    }
    if (g_cond) for (int n = 0; n < B; n++) cp(d->wdb[i] + n * l3, b->wind + n * p3 + z0 * p2, l3);   /* (--cond excludes --split: whole windows) */
    if (g_pap_head) for (int n = 0; n < B; n++) cp(d->papb[i] + n * l3, b->pap + n * p3 + z0 * p2, l3);   /* (--pap-head excludes --split) */
}
/* synchronous upload (pageable host memory: validation batches) */
static void upload(gpu_state *d, const batch *b, int B, int P) {
    d->sheet=b->sheet?b->sheet[0]:nullptr;
    d->um = d->umb[d->cur] = b->um ? b->um[0] : 0.f;
    if (b->corner && b->level) { memcpy(d->cornb[d->cur], b->corner[0], sizeof d->cornb[0]); d->lvlb[d->cur] = b->level[0]; }   /* validation copies carry none */
    if (d->side >= 0) { upload_slab(d, b, d->cur, B, P, 0); return; }
    size_t p3 = (size_t)g_Pz * P * P;
    nn_h2d(d->x, bx(b), (size_t)B * g_xs * p3 * xbytes());
    if (g_gpu_in) nn_h2d(d->rad, b->rad, rad_bytes(B));
    if (g_pack) { nn_h2d(d->m, b->tm, (size_t)B * p3); nn_unpack_tm(d->t, d->m, B, p3, 0); }   /* (--pack-targets excludes --heads) */
    else {
        if (!g_heads) nn_h2d(d->t, b->t, (size_t)B * NCH * p3);
        else for (int r = 0; r < B * NCH; r++) nn_h2d(d->t + (size_t)trow(b, r) * p3, b->t + (size_t)r * p3, p3);
        nn_h2d(d->m, b->m, (size_t)B * p3);
    }
    nn_h2d(d->w, trow_w(b, B, d->hwh[d->cur]), (size_t)B * g_tc);
    if (g_aff) nn_h2d(d->band, b->band, (size_t)B * p3);
    if (g_side) nn_h2d(d->sdb, b->side, (size_t)B * p3);
    if (g_unwrap) nn_h2d(d->pm, b->pm, (size_t)B * p3);
    if (g_cond) nn_h2d(d->wd, b->wind, (size_t)B * p3);
    if (g_pap_head) nn_h2d(d->pap, b->pap, (size_t)B * p3);
}
/* asynchronous upload of a pinned sampler batch into buffer i on the copy stream, after the compute that last used it */
static void upload_async(gpu_state *d, const batch *b, int i, int B, int P) {
    d->sheet=b->sheet?b->sheet[0]:nullptr;
    d->umb[i] = b->um ? b->um[0] : 0.f;
    if (b->corner && b->level) { memcpy(d->cornb[i], b->corner[0], sizeof d->cornb[0]); d->lvlb[i] = b->level[0]; }
    size_t p3 = (size_t)g_Pz * P * P;
    nn_stream_wait(1, d->ev_done[i]);   /* the step two back: this upload overlaps the current step */
    if (d->lean) nn_stream_wait(1, d->ev_done[i ^ 1]);   /* lean: one buffer (inside the gradient buffers), free after the current step */
    if (d->side >= 0) upload_slab(d, b, i, B, P, 1);
    else {
        nn_h2d_copy_stream(d->xb[i], bx(b), (size_t)B * g_xs * p3 * xbytes());
        if (g_gpu_in) nn_h2d_copy_stream(d->radb[i], b->rad, rad_bytes(B));
        if (g_pack) { nn_h2d_copy_stream(d->mb[i], b->tm, (size_t)B * p3); nn_unpack_tm(d->tb[i], d->mb[i], B, p3, 1); }   /* (copy stream) */
        else {
            if (!g_heads) nn_h2d_copy_stream(d->tb[i], b->t, (size_t)B * NCH * p3);
            else for (int r = 0; r < B * NCH; r++) nn_h2d_copy_stream(d->tb[i] + (size_t)trow(b, r) * p3, b->t + (size_t)r * p3, p3);
            nn_h2d_copy_stream(d->mb[i], b->m, (size_t)B * p3);
        }
        nn_h2d_copy_stream(d->wb[i], trow_w(b, B, d->hwh[i]), (size_t)B * g_tc);
        if (g_aff) nn_h2d_copy_stream(d->bandb[i], b->band, (size_t)B * p3);
        if (g_side) nn_h2d_copy_stream(d->sideb[i], b->side, (size_t)B * p3);
        if (g_unwrap) nn_h2d_copy_stream(d->pmb[i], b->pm, (size_t)B * p3);
        if (g_cond) nn_h2d_copy_stream(d->wdb[i], b->wind, (size_t)B * p3);
        if (g_pap_head) nn_h2d_copy_stream(d->papb[i], b->pap, (size_t)B * p3);
    }
    nn_event_record(d->ev_up[i], 1);
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
    nn_d2h(ht, d->tb[cb], (size_t)B * g_tc * p3); nn_d2h(hm, d->mb[cb], (size_t)B * p3); nn_d2h(hw, d->wb[cb], (size_t)B * g_tc);
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
    if (!pass) nn_zero(d->hs_ds, 64 * sizeof(double));
    for (int z0 = 0; z0 < D; z0 += Z) {
        const int z1 = z0 + Z < D ? z0 + Z : D, nz = z1 - z0;
        const size_t ns = (size_t)nz * hw, vb = (size_t)z0 * hw;
#define LGC(c) ((char *)d->hs_lg + (size_t)(c) * ns * 2)
        if (unet_head_fwd_z(d->u, z0, z1, d->hs_lg)) { fprintf(stderr, "head slabs: the head is not covered (MX activations; UFSM_HEAD_SLAB=0)\n"); abort(); }
        const shape5 one = {1, 1, nz, H, W}, two = {1, 2, nz, H, W}, seven = {1, 7, nz, H, W};
        nn_phase_targets(d->sdb + vb, ns, d->sidet, d->sidem);
        if (g_packed_w > 1) nn_packed_mask_z(d->sdb, D, H, W, g_norm_s, d->um, g_packed_um, g_packed_w, d->sidem, z0, z1);   /* packed sheets weigh more */
        nn_unwrap_targets2_z(d->sdb, D, H, W, g_norm_s, g_dens_scale, g_dens_code, d->um, g_packed_um, g_packed_w, d->nt, d->nm, z0, z1);
        if (!pass) {
            nn_loss_stats_acc(LGC(0), d->t + vb, d->m + vb, d->w, one, d->hs_ds);
            nn_loss_stats_acc(LGC(1), d->t + l3 + vb, d->pm + vb, d->w + 1, one, d->hs_ds + 5);
            nn_loss_stats_acc(LGC(g_side_ch), d->sidet, d->sidem, d->sidew, two, d->hs_ds + 10);
            nn_loss_stats_acc(LGC(g_norm_ch), d->nt, d->nm, d->nw, seven, d->hs_ds + 20);
            continue;
        }
        nn_loss_grad_only(LGC(0), d->t + vb, d->m + vb, d->w, one, d->hs_dice, LGC(0), d->scratch);
        nn_loss_grad_only(LGC(1), d->t + l3 + vb, d->pm + vb, d->w + 1, one, g_ink_dice, LGC(1), d->iscratch);
        if (g_ink_lambda != 1.f) nn_loss_grad_scale(LGC(1), ns, g_ink_lambda);
        nn_loss_grad_only(LGC(g_side_ch), d->sidet, d->sidem, d->sidew, two, 0.f, LGC(g_side_ch), d->sscratch);
        if (g_side_lambda != 1.f) nn_loss_grad_scale(LGC(g_side_ch), 2 * ns, g_side_lambda);
        nn_loss_grad_only(LGC(g_norm_ch), d->nt, d->nm, d->nw, seven, 0.f, LGC(g_norm_ch), d->nscratch);
        if (g_norm_lambda != 1.f) nn_loss_grad_scale(LGC(g_norm_ch), 7 * ns, g_norm_lambda);
        if (unet_head_bwd_z(d->u, z0, z1, d->hs_lg, z0 == 0)) { fprintf(stderr, "head slabs: the fused head backward is not covered (UFSM_HEAD_SLAB=0)\n"); abort(); }
#undef LGC
    }
    if (!pass) {
        const shape5 w1 = {1, 1, D, H, W}, w2 = {1, 2, D, H, W}, w7 = {1, 7, D, H, W};
        nn_loss_stats_fin(d->hs_ds, d->w, w1, d->scratch);
        nn_loss_stats_fin(d->hs_ds + 5, d->w + 1, w1, d->iscratch);
        nn_loss_stats_fin(d->hs_ds + 10, d->sidew, w2, d->sscratch);
        nn_loss_stats_fin(d->hs_ds + 20, d->nw, w7, d->nscratch);
    }
}
static void head_slab_cb(void *ctx) { head_slab_pass((gpu_state *)ctx, 1); }
static void run_batch(gpu_state *d, int B, int P, float dice_w, int train) {
    shape5 xs = {B, g_xc, d->side >= 0 ? g_Dl : g_Pz, P, P};
    if (unet_cfg_of(d->u)->scale_cond) { if (g_papyrus) unet_set_cond(d->u, d->um); else unet_set_scale(d->u, d->um); }   /* B == 1 (checked at
                                                                                                    startup); papyrus: um is the cutoff code */
    if (g_verso_on) verso_target(d, xs);
    const int q = P / g_hdiv, hm = unet_hr_mode(g_half);
    const shape5 hs = {B, g_cout, q, q, q};   /* --half-res: the network's grid, the full-resolution output channels */
    const float *lg;
    if (g_half) {   /* the network on blocks of the window: input and logits through nn_lowres_*, fp32 full-resolution logits for the losses */
        nn_lowres_input(d->x, xs, g_hdiv, hm, d->lrt, d->x2, g_xfmt != 0);
        const float *l2 = unet_forward_x(d->u, d->x2, (shape5){B, unet_cfg_of(d->u)->cin, q, q, q}, train, g_xfmt != 0);
        nn_lowres_output(l2, unet_logits_h16(d->u), hs, g_hdiv, hm, (float *)d->lrt, d->lgf);
        lg = d->lgf;
    } else lg = unet_forward_x(d->u, d->x, xs, train, g_xfmt != 0);
    d->lg = lg;
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
    if (g_head_slab) {   /* the head per z-slab: the loss statistics now, the gradients in the backward (head_slab_cb) */
        int prof = ufsm_env_on("UFSM_PROF");
        if (prof) nn_prof_begin(6);
        d->gl = nullptr; d->hs_dice = dice_w;
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
    nn_set_logits_h16(!g_half && unet_logits_h16(d->u));   /* lean 2: fp16 logits in the gradient buffer */
    if (g_unwrap) {   /* recto on target 0 with the label mask; ink on target 1 over all papyrus, each its own loss */
        const size_t l3 = (size_t)os.d * os.h * os.w, eb = g_g16 ? 2 : 4, le = unet_logits_h16(d->u) ? 2 : 4;
        const shape5 one = {1, 1, os.d, os.h, os.w};
        nn_loss_async_tol(lg, d->t, d->m, d->w, one, dice_w, train ? d->gl : nullptr, d->scratch, d->tolc);
        nn_loss_async_tol((const float *)((const char *)lg + l3 * le), d->t + l3, d->pm, d->w + 1, one, g_ink_dice, train ? (float *)((char *)d->gl + l3 * eb) : nullptr, d->iscratch, nullptr);
        if (train && g_ink_lambda != 1.f) nn_loss_grad_scale((char *)d->gl + l3 * eb, l3, g_ink_lambda);
    } else if (!g_verso_on) nn_loss_async_tol(lg, d->t, d->m, d->w, ts, dice_w, train ? d->gl : nullptr, d->scratch, d->tolc);
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
    if (g_half && train) nn_lowres_grad(d->gl, g_g16, hs, g_hdiv, hm, d->lrt, d->glh);   /* the logit gradient on the network's grid */
    if (prof) nn_prof_end();
}
/* fetch the loss parts of the last run_batch on this GPU (synchronous); returns the scalar loss */
static double fetch_loss(gpu_state *d, int B, int P, float dice_w, float *out) {
    shape5 os = g_half ? (shape5){B, g_cout, P, P, P} : unet_out_shape(d->u, (shape5){B, g_xc, d->side >= 0 ? g_Dl : g_Pz, P, P});
    if (g_aff) os.c = NCH;
    if (!g_verso_on && !g_unwrap) nn_loss_fetch(d->scratch, os, out);
    d->ink_loss = d->norm_loss = 0;
    if (g_unwrap) {   /* recto parts laid out as the NCH-channel fetch does; the ink and normal / density losses */
        float r[3]; nn_loss_fetch(d->scratch, (shape5){1, 1, 1, 1, 1}, r);
        for (int c = 0; c < 2 * NCH + 1; c++) out[c] = 0;
        out[0] = r[0]; out[NCH] = r[1]; out[2 * NCH] = r[2];
        nn_loss_fetch(d->iscratch, (shape5){1, 1, 1, 1, 1}, d->iout); d->ink_loss = g_ink_lambda * (d->iout[0] + g_ink_dice * d->iout[1]);
        nn_loss_fetch(d->nscratch, (shape5){1, 7, 1, 1, 1}, d->nout);
        float nb = 0; for (int c = 0; c < 7; c++) nb += d->nout[c];
        d->norm_loss = g_norm_lambda * nb / 7;
    }
    d->ds_loss = 0;
    if (g_ds && d->ds_fresh) for (int j = 1; j <= g_ds; j++) {   /* weight x mean over the active channels of bce + dice_w dice */
        shape5 sj; unet_ds_logits(d->u, j, &sj); float *o = d->ds_out[j]; nn_loss_fetch(d->ds_scr[j], sj, o);
        double s = 0; int nc = 0;
        for (int c = 0; c < sj.c && c < g_tc; c++) if (o[c] != 0.f || o[sj.c + c] != 0.f) { s += o[c] + dice_w * o[sj.c + c]; nc++; }
        d->ds_loss += powf(g_ds_w, (float)j) * g_ds_frac * (nc ? s / nc : 0);
    }
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
    return (active ? loss / active : 0)+d->sheet_loss+d->aff_loss+d->verso_loss+d->side_loss+d->ink_loss+d->norm_loss+d->cond_loss+d->pap_loss+d->count_loss+d->ds_loss;
}

/* --split z: forward, loss and backward of one window on both GPUs (src/split.h) */
typedef struct { gpu_state *G; int B, P, train; float dice_w; } split_arg;
static void split_job(int side, void *a) {
    split_arg *s = a;
    gpu_state *d = &s->G[side];
    run_batch(d, s->B, s->P, s->dice_w, s->train);
    if (s->train) unet_backward_x(d->u, g_half ? d->glh : d->gl, g_g16);
}

int cmd_train(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: ufsm train <sources.json> --out DIR [--P 512] [--B 1] [--steps 20000] [--lr 1e-3] [--warmup 500] [--wd 0.01]\n"
                        "       [--dice 0.5] [--ema 0.999] [--widths 16,32,64,80] [--down-norm 1] [--gpus 0,1] [--workers 12] [--sampler-slots N] [--seed 0] [--det 1] [--resume CKPT] [--finetune 1] [--fp4 0|1|2 (default 2)] [--mem auto|auto16|default|wide] [--opt adamw|muon|anvil] [--muon-lr 0.02] [--muon-beta 0.95] [--anvil-lr 0.023] [--anvil-wd 2.25] [--sched cos|wsd] [--cooldown 0.2]\n"
                        "       [--val-batches 8] [--log-every 20] [--val-every 500] [--ckpt-every 1000] [--clip 5] [--levels 0.5,0.25,0.15,0.1] [--fp32] [--f16 1] [--gscale 1024] [--prec 1|2|3|4] [--input-prec 0|4|8] [--policy enc0=1,dec0.c1=fp16:fp16:fp8,...] [--qat 2|3] [--wq 8|4] [--sparse24 STEP] [--srste 2e-4] [--pos-weight 1] [--tol 0..2] [--sr 1]\n"
                        "       [--seconds S] [--warmup-seconds S (default 5%% of time budget)] (time-based schedule and final checkpoint)\n"
                        "       [--schedule-seconds S --schedule-elapsed S]   wall-time LR horizon across finite-cover evaluation pauses\n"
                        "       [--cover PLAN.json] [--cover-extend-from OLD_PLAN.json] [--cover-next-pass 1] [--stop-at STEP] [--ct-aug 0|1] [--symmetry-p 1] [--axis-jitter 0] [--limit-seconds 0]\n"
                        "       [--lr-floor 0 (the schedule ends at this fraction of the peak; modded-nanogpt 0.3)] [--wd-nonconv 0 (decay multiplier for GroupNorm, biases, head under Muon / ANVIL)]\n"
                        "       [--recall W (recto: + W (1 - soft recall), gradient only)]\n"
                        "       [--dec-convs 2 (convs per decoder level, one value or a list; 1 = one conv, nnU-Net style)]\n"
                        "       [--deep-sup K (1^3 heads on decoder levels 1..K)] [--ds-weight 0.5 (level j: w^j)] [--ds-anneal 1 (weights -> 0 over the run)]\n"
                        "       [--iso T (void cutoff of sources without \"iso\")] [--iso-void 0|1] [--iso-pap 2] [--iso-lift 0 (voxels)]   void (CT < cutoff, > iso-pap voxels\n"
                        "                                     from papyrus) is background beyond the trust band; labels with no papyrus within iso-lift voxels ignored\n"
                        "       [--shell R (level-0 voxels: background beside a label ignored)] [--dist-head SIGMA (channel 1 = distance ridge of channel 0)]\n"
                        "       [--geometry-aug 0|1] [--rotate-deg 5] [--rotate-p 0.2] [--elastic 1] [--elastic-p 0.15] [--label-morph 0] [--label-morph-p 0.2]\n"
                        "       [--aug-strength 1 (0..2)] [--aug-p 0 (default p of every scanner op)] [--aug-OP-p P] [--aug-drop-ignore 1] [--aug-affine-p 0] [--aug-aniso 0.1] [--aug-shear 0.06]\n"
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
    uint64_t seed = (uint64_t)atoll(opt(argc, argv, "--seed", "0"));
    const char *resume = opt(argc, argv, "--resume", nullptr);
    const char *geometry=opt(argc,argv,"--geometry",nullptr),*task=opt(argc,argv,"--task",geometry?"surface_winding":"surface");
    int sheet_init=atoi(opt(argc,argv,"--sheet-init","0")); char old_geometry[65],old_reference[65];
    int saved_variant=2,saved_sheet_start=0;
    int sheet_ck=resume?sheet_checkpoint(resume,old_geometry,old_reference):0;
    g_aff = !strcmp(task,"band_affinity");
    g_unwrap = !strcmp(task,"unwrap");
    if (g_unwrap) {   /* recto, ink, then phase (2) and normal / density (7) below */
        if (B != 1 || geometry || atoi(opt(argc,argv,"--overfit","0")) || strcmp(opt(argc,argv,"--mem","auto"),"wide")==0) { fprintf(stderr,"unwrap: B=1, no geometry / overfit / wide\n"); return 2; }
        g_cout = 2;
        g_ink_lambda = (float)atof(opt(argc,argv,"--ink-lambda","1")); g_ink_dice = (float)atof(opt(argc,argv,"--ink-dice","0"));
        g_norm_lambda = (float)atof(opt(argc,argv,"--normal-lambda","1")); g_norm_s = atoi(opt(argc,argv,"--normal-step","4"));
        if (g_norm_s < 1 || g_norm_s > 16) { fprintf(stderr,"--normal-step 1..16\n"); return 2; }
        g_dens_scale = (float)atof(opt(argc,argv,"--density-scale","32"));   /* density target min(1, scale rho): 32 at 2.4 um, 8 at 9.6 um (the same physical code) */
        if (!(g_dens_scale > 0)) { fprintf(stderr,"--density-scale > 0\n"); return 2; }
        /* --density-code 1: density target x / (1 + x), x = scale rho (never saturates; 0: min(1, x), which clips every spacing under
           scale x voxel size, 77 um at the default codes); --packed-weight W (> 1): the phase and normal / density losses weigh a voxel
           1 at a sheet spacing of --packed-um (120) or more, rising to W at half of it (the nets under-count packed sheets) */
        g_dens_code = atoi(opt(argc,argv,"--density-code","0")); g_packed_w = (float)atof(opt(argc,argv,"--packed-weight","0")); g_packed_um = (float)atof(opt(argc,argv,"--packed-um","120"));
        if (g_dens_code < 0 || g_dens_code > 1 || g_packed_w < 0 || g_packed_w > 255 || !(g_packed_um > 0)) { fprintf(stderr,"--density-code 0|1, --packed-weight 0..255, --packed-um > 0\n"); return 2; }
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
        fprintf(stderr, "unwrap: channels 0 recto, 1 ink (lambda %g, dice %g), %d-%d phase, %d-%d normal n n^T, %d density (lambda %g, step %d)\n",
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
    if (sheet_ck<0 || (strcmp(task,"surface") && strcmp(task,"surface_winding") && !g_aff && !g_unwrap && !g_papyrus) || (!strcmp(task,"surface_winding") != (geometry!=nullptr)) ||
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
    if (f16) { nn_set_f16(1); nn_set_grad_scale((float)atof(opt(argc, argv, "--gscale", "1024"))); }
    const char *optname = opt(argc, argv, "--opt", "adamw");   /* adamw | muon (3^3 conv weights: nesterov momentum + Newton-Schulz orthogonalisation; rest AdamW) */
    float muon_lr = (float)atof(opt(argc, argv, "--muon-lr", "0.02")), muon_beta = (float)atof(opt(argc, argv, "--muon-beta", "0.95"));
    int use_muon = !strcmp(optname, "muon"), use_anvil = !strcmp(optname, "anvil");
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
    int noaug = atoi(opt(argc, argv, "--noaug", "0"));       /* diagnostic: no augmentation */
    /* --fp4 1: packed fp4 activations, fp4 stride-1 compute, fp8 weight gradients and stochastic rounding.
       --fp4 2: fp4 stride-1 weight gradients as well. Small inputs and stride-2 operations have FP8 / FP32
       fallbacks; a requested 16-bit policy does not promote MX storage. The executed manifest records these.
       Explicit --policy changes the requested precisions; --fp4 0 uses 16-bit storage. */
    int expl = *opt(argc, argv, "--prec", "") || *opt(argc, argv, "--wq", "");
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--fp32")) expl = 1;
    const int fp4 = atoi(opt(argc, argv, "--fp4", expl ? "0" : "2"));   /* --fp4 2 since 2026-10-01 night: passed the stair (last.ckpt 0.299 / 0.284 vs --fp4 1 0.307 / 0.284) and the 40k confirmation (r11c), 1-4% faster */
    { int nl = 1; for (const char *q = opt(argc, argv, "--widths", "16,32,64,80"); *q; q++) nl += *q == ','; nn_set_nlev(nl < UNET_MAXLEV ? nl : UNET_MAXLEV); }   /* layer names of the policy */
    if (fp4) {
        unet_set_act_mx4(1); setenv("UFSM_ACT_MX4", "1", 0); nn_set_sr(1);
        if (nn_set_prec_policy(fp4 >= 2 ? "all=fp4:fp4:fp4,enc0.c1=fp16" : "all=fp4:fp4:fp8,enc0.c1=fp16")) return 2;
        if (fp4 >= 2) setenv("UFSM_F4_WGRAD", "1", 0);
    }
    if (*opt(argc, argv, "--policy", "") && nn_set_prec_policy(opt(argc, argv, "--policy", ""))) return 2;
    { char mf[4096]; nn_prec_manifest(mf, sizeof mf); fprintf(stderr, "%s\n", mf); char mp[1400]; snprintf(mp, sizeof mp, "mkdir -p '%s'", out); if (system(mp)) {} snprintf(mp, sizeof mp, "%s/precision.txt", out); WRITE_MANIFEST(mp); }   /* per-layer: "enc0=1,enc1=2,..." */
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--fp32")) nn_set_tf32(0);
    int devs[8], ng = 0;
    { char *t = strdup(opt(argc, argv, "--gpus", opt(argc, argv, "--gpu", "0"))); for (char *q = strtok(t, ","); q && ng < 8; q = strtok(nullptr, ",")) devs[ng++] = atoi(q); free(t); }
    nn_init(devs[0]);   /* env precision knobs are read here; the storage modes decide the batch formats */
    g_xfmt = nn_get_tf32() && nn_get_act_bf16() && !ufsm_env_on("UFSM_ACT_MX8") ? (nn_get_f16() ? 1 : 2) : 0;
    g_g16 = nn_get_tf32() && nn_get_act_bf16() && nn_get_grad_bf16();
    if (ufsm_env_on("UFSM_X32")) g_xfmt = g_g16 = 0;   /* diagnostic: fp32 batches and logit gradient, converted on the device (the old path) */
    nn_set_loss_grad_h16(g_g16);
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
    if (resume) { unet_cfg pc; int st; if (!unet_peek(resume, &pc, &st)) { cfg.down_norm = pc.down_norm; cfg.scale_cond = pc.scale_cond; memcpy(cfg.dec_widths, pc.dec_widths, sizeof cfg.dec_widths); cfg.ds_levels = pc.ds_levels; memcpy(cfg.dec_convs, pc.dec_convs, sizeof cfg.dec_convs); memcpy(cfg.enc_blocks, pc.enc_blocks, sizeof cfg.enc_blocks); } }   /* the checkpoint decides */
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
            fprintf(stderr, "low resolution (%s): the network runs on %d^3 blocks of %d^3 voxels, %d inputs, %d outputs\n", hr, P / g_hdiv, g_hdiv, cfg.cin, cfg.cout);
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
    if (!S) return 1;
    g_S = S;
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
    sc.deterministic = atoi(opt(argc, argv, "--det", "0"));   /* same batches for the same seed regardless of thread timing (paired comparisons; off: a slow worker can stall the ring) */
    if (noaug) sc.augment = 0;
    if (atoi(opt(argc, argv, "--rotonly", "0"))) sc.augment = 2;   /* proper rotations only (no reflections) */
    if (atoi(opt(argc, argv, "--zfix", "0"))) sc.augment = 3;      /* diagnostic: symmetries that keep the z axis */
    if (atoi(opt(argc, argv, "--intonly", "0"))) sc.augment = 4;   /* diagnostic: intensity jitter, no symmetry */
    sc.aux = g_xc == 5; sc.cond = g_cond; sc.cond_outside = g_cond_outside; sc.pap_head = g_pap_head ? &g_paph : nullptr;
    sc.band = g_aff; sc.side = g_side; sc.pmask = g_unwrap;   /* unwrap: the phase only (band codes per window, or the stored phase) */ sc.side_air = atoi(opt(argc, argv, "--side-air", "0"));
    sc.papyrus = g_papyrus ? &g_pap : nullptr;
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
    sc.ct_augment = atoi(opt(argc, argv, "--ct-aug", "0"));
    sc.symmetry_p = (float)atof(opt(argc, argv, "--symmetry-p", "1"));
    sc.axis_jitter = (float)atof(opt(argc, argv, "--axis-jitter", "0"));
    /* --axis-jitter-p P (default 0.1), --axis-jitter-deg D (default 2): how often the axis moves and the cap on the radial input's
       direction change; an estimated umbilicus on another scroll is off by ~1-3 mm (on block B, 3 mm costs phase confidence .65 -> .58) */
    sc.axis_jitter_p = (float)atof(opt(argc, argv, "--axis-jitter-p", "0")); sc.axis_jitter_deg = (float)atof(opt(argc, argv, "--axis-jitter-deg", "0"));
    sc.geometry_augment = atoi(opt(argc,argv,"--geometry-aug","0"));
    sc.rotate_degrees=(float)atof(opt(argc,argv,"--rotate-deg","5"));
    sc.rotate_p=(float)atof(opt(argc,argv,"--rotate-p","0.2"));
    sc.elastic=(float)atof(opt(argc,argv,"--elastic","1"));
    sc.elastic_p=(float)atof(opt(argc,argv,"--elastic-p","0.15"));
    sc.zoom_min=(float)atof(opt(argc,argv,"--zoom-min","0.7071"));   /* zoom-in down to this scale (needs --geometry-aug 1) */
    sc.zoom_p=(float)atof(opt(argc,argv,"--zoom-p","0"));
    sc.label_morph=(float)atof(opt(argc,argv,"--label-morph","0"));
    sc.label_morph_p=(float)atof(opt(argc,argv,"--label-morph-p","0.2"));
    sc.scan.strength=(float)atof(opt(argc,argv,"--aug-strength","1"));
    sc.scan.drop_ignore=atoi(opt(argc,argv,"--aug-drop-ignore","1"));
    {
        const char *all=opt(argc,argv,"--aug-p","0");
        for (int i=0;i<SA_N;i++) { char f[48]; snprintf(f,sizeof f,"--aug-%s-p",scan_aug_names[i]); sc.scan.p[i]=(float)atof(opt(argc,argv,f,all)); }
        for (int i=0;i<SA_N;i++) { char f[48]; snprintf(f,sizeof f,"--aug-%s-s",scan_aug_names[i]); sc.scan.s[i]=(float)atof(opt(argc,argv,f,"0")); }   /* per-op strength (0: --aug-strength) */
        const char *hf=opt(argc,argv,"--aug-hist",nullptr);   /* target histograms for the hist op (ufsm ct-hist) */
        if (hf && scan_aug_load_hist(hf,&sc.scan)) { fprintf(stderr,"--aug-hist: cannot read %s\n",hf); return 2; }
        if (sc.scan.p[SA_HIST]>0 && !sc.scan.nhist) { fprintf(stderr,"--aug-hist-p needs --aug-hist FILE\n"); return 2; }
        for (int i=0;i<SA_N;i++) if (!isfinite(sc.scan.s[i]) || sc.scan.s[i]<0 || sc.scan.s[i]>3) { fprintf(stderr,"--aug-%s-s: 0..3\n",scan_aug_names[i]); return 2; }
    }
    sc.affine_p=(float)atof(opt(argc,argv,"--aug-affine-p","0"));
    sc.affine_aniso=(float)atof(opt(argc,argv,"--aug-aniso","0.1"));
    sc.affine_shear=(float)atof(opt(argc,argv,"--aug-shear","0.06"));
    {
        int bad=!isfinite(sc.scan.strength) || sc.scan.strength<0 || sc.scan.strength>2 || sc.scan.drop_ignore<0 || sc.scan.drop_ignore>1 ||
                !isfinite(sc.affine_p) || sc.affine_p<0 || sc.affine_p>1 || !isfinite(sc.affine_aniso) || sc.affine_aniso<0 || sc.affine_aniso>.3f ||
                !isfinite(sc.affine_shear) || sc.affine_shear<0 || sc.affine_shear>.2f || (sc.affine_p>0 && !sc.geometry_augment);
        for (int i=0;i<SA_N;i++) bad|=!isfinite(sc.scan.p[i]) || sc.scan.p[i]<0 || sc.scan.p[i]>1;
        if (bad) { fprintf(stderr,"invalid scanner/affine augmentation bounds (--aug-affine-p needs --geometry-aug 1)\n"); return 2; }
    }
    sc.sheet=g_sheet;
    if (g_sheet) { sc.augment=noaug?0:sc.geometry_augment?1:3; memset(sc.level_p,0,sizeof sc.level_p); sc.level_p[0]=1; }
    if (!isfinite(sc.symmetry_p) || sc.symmetry_p < 0 || sc.symmetry_p > 1 || !isfinite(sc.axis_jitter) || sc.axis_jitter < 0 ||
        !(sc.axis_jitter_p >= 0 && sc.axis_jitter_p <= 1) || !(sc.axis_jitter_deg >= 0 && sc.axis_jitter_deg <= 60)) { fprintf(stderr, "invalid augmentation bounds\n"); return 2; }
    if (sc.geometry_augment<0 || sc.geometry_augment>1 || !isfinite(sc.rotate_degrees) || sc.rotate_degrees<0 || sc.rotate_degrees>15 || !isfinite(sc.elastic) || sc.elastic<0 || sc.elastic>2 ||
        !isfinite(sc.rotate_p) || sc.rotate_p<0 || sc.rotate_p>1 || !isfinite(sc.elastic_p) || sc.elastic_p<0 || sc.elastic_p>1 ||
        !isfinite(sc.zoom_min) || sc.zoom_min<0.25f || sc.zoom_min>1 || !isfinite(sc.zoom_p) || sc.zoom_p<0 || sc.zoom_p>1 ||
        !isfinite(sc.label_morph) || sc.label_morph<0 || sc.label_morph>1 || !isfinite(sc.label_morph_p) || sc.label_morph_p<0 || sc.label_morph_p>1) {
        fprintf(stderr,"invalid continuous geometry/morphology augmentation bounds\n"); return 2;
    }
    {   /* --gpu-input 1 (default where possible): the batches carry the CT channel alone plus a radial record per sample, the 4 MX input
           channels are built on the device (nn_input_rad): 6 B per voxel less per sampler slot, upload and device batch, no CPU x16
           radial pass. Needs the 16-bit -> MX input conversion, whole windows, the plain 4 inputs and write_x samples; --gpu-input 0 or
           UFSM_DUMP_BATCH (it shows the 4 channels): the host-built 4 channels as before */
        const int want = atoi(opt(argc, argv, "--gpu-input", "1"));
        g_gpu_want = want && g_xfmt && unet_act_mx() && !split && !g_half && g_xc == 4 && sample_gpu_input_ok(&sc) && !getenv("UFSM_DUMP_BATCH");
        g_gpu_asked = atoi(opt(argc, argv, "--gpu-input", "0"));   /* (an explicit request: say when it cannot apply) */
        /* the MX input conversion it needs depends on the memory mode (MX gradients): settled after the planner (gpu_input_settle) */
    }
    {   /* --pack-targets 1 (default where it applies): hard two-class targets (every source's recto target side-encoded: 0 / 255 in both
           channels) travel as one code byte per voxel (sample.h SAMPLE_TM_*) through the sampler slots, the validation batches and the
           upload, unpacked on the device (nn_unpack_tm): 3 -> 1 B per voxel on the host and over PCIe. Not with --heads (per-source
           target rows), --split, the papyrus task, --dist-head, the sheet prior, geometry warps (trilinear targets) or --overfit;
           --pack-targets 0: target and mask bytes */
        int hard = S->n > 0;
        for (int i = 0; i < S->n; i++) hard &= S->src[i].tgt_side[CH_RECTO];
        g_pack = atoi(opt(argc, argv, "--pack-targets", "1")) && hard && !g_heads && !split && !sc.papyrus && sc.dist_sigma <= 0 && !sc.sheet && !sc.geometry_augment && !overfit;
        sc.pack_tm = g_pack;
        if (g_pack) fprintf(stderr, "targets: packed, one byte per voxel on the host and in the upload (--pack-targets 0: target and mask bytes)\n");
        else if (atoi(opt(argc, argv, "--pack-targets", "0"))) fprintf(stderr, "--pack-targets: not available here (needs side-encoded targets in every source, no --heads, --split, papyrus task, --dist-head, sheet prior, geometry warps or --overfit)\n");
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
    nn_set_loss_tol(atoi(opt(argc, argv, "--tol", "0")));   /* offset-tolerant positives, radius in output voxels (<= 2) */
    nn_set_recall((float)atof(opt(argc, argv, "--recall", "0")));   /* skeleton-recall term on the recto (thin labels: the label is its skeleton) */
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
        if (nn_init(d->dev)) { fprintf(stderr, "cannot select GPU %d\n", d->dev); return 1; }
        d->u = unet_create(&cfg);
        unet_set_wd_nonconv(d->u, (float)atof(opt(argc, argv, "--wd-nonconv", "0")));   /* Muon / ANVIL: no decay on GroupNorm, biases, head */
        /* fp16 logits in their own buffer (MX activations; the losses read them as unet_logits_h16 says): 11 unwrap heads at P 384
           are 2.5 GB in fp32. Not for half-resolution nets (the losses read the fp32 upsampled copy) or sheet nets; UFSM_LOGITS32=1 */
        unet_set_fp16_logits(d->u, !g_half && !g_sheet && !ufsm_env_on("UFSM_LOGITS32"));
        if (resume) {
            const int grow = atoi(opt(argc, argv, "--grow-head", "0"));   /* band_affinity warm start from a recto checkpoint */
            step0 = grow ? unet_load_grow(d->u, resume, 1, 1.f) : unet_load(d->u, resume);
            if (step0 < 0) { fprintf(stderr, "cannot load %s\n", resume); return 1; }
            if (atoi(opt(argc, argv, "--finetune", "0"))) step0 = 0;   /* loaded weights/state, fresh schedule */
            if (sheet_init && unet_start_sheet(d->u)) return 2;
        } else {
            unet_init(d->u, seed + 1); /* deterministic: every GPU starts identical */
            const char *init_from = opt(argc, argv, "--init-from", nullptr);   /* partial warm start from another architecture */
            if (init_from) {
                int rows[16];   /* --verso: recto copied into the verso channel, the donor's other rows (affinities) shifted up */
                if (g_verso) { unet_cfg dc; int ds; if (unet_peek(init_from, &dc, &ds)) { fprintf(stderr, "cannot read %s\n", init_from); return 1; }
                    rows[0] = 0; rows[1] = g_verso_zero ? -1 : 0;
                    for (int co = 2; co < cfg.cout; co++) rows[co] = dc.cout == cfg.cout ? co : co - 1 < dc.cout ? co - 1 : -1;
                    if (dc.cout == cfg.cout && g_aff) { fprintf(stderr, "--verso --init-from: %s already has %d outputs; use --resume\n", init_from, dc.cout); return 1; } }
                const char *hm = opt(argc, argv, "--head-map", nullptr);   /* explicit donor row per output channel (-1 fresh), e.g. 0,2,3,4,5,6,7,-1 */
                if (hm) { int k = 0; char *t = strdup(hm); for (char *q = strtok(t, ","); q && k < 16; q = strtok(nullptr, ",")) rows[k++] = atoi(q); free(t);
                    if (k != cfg.cout) { fprintf(stderr, "--head-map has %d rows, the network %d outputs\n", k, cfg.cout); return 2; } }
                const int nt = g_verso || hm ? unet_init_from_map(d->u, init_from, rows) : unet_init_from(d->u, init_from);
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
        g_gl_inplace = !g_half && !g_sheet && !g_aff && !g_cond && g_count_lambda == 0.f && !nn_get_loss_tol() && g_g16 && !ufsm_env_on("UFSM_LOGITS32") && !ufsm_env_on("UFSM_GL_SEPARATE");
        /* head slabs (head_slab_pass): the production unwrap layout (11 outputs, every channel trained by one of its four losses), B 1,
           whole windows, MX activations; -22 B (logits) - 11 B (derived targets) per voxel. UFSM_HEAD_SLAB=0: whole-window head and losses;
           UFSM_HEAD_SLAB_Z: planes per slab (32) */
        g_head_slab = g_gl_inplace && g_unwrap && B == 1 && !split && !g_verso && !g_pap_head && g_side == 2 && g_side_nc == 2 && g_side_ch == 2 && g_norm_ch == 4
                      && g_cout == 11 && unet_act_mx() && cfg.widths[0] == 32 && !(getenv("UFSM_HEAD_SLAB") && !atoi(getenv("UFSM_HEAD_SLAB")));   /* (the fused
                                                                                                     slab head backward takes exactly 32 inputs) */
        g_hs_z = getenv("UFSM_HEAD_SLAB_Z") ? atoi(getenv("UFSM_HEAD_SLAB_Z")) : 32;
        if (g_hs_z < 1) g_hs_z = 1;
        if (g_hs_z > g_Pz) g_hs_z = g_Pz;
        for (int g = 0; g < ng; g++) unet_set_head_ext(G[g].u, g_head_slab);
        const char *mm = opt(argc, argv, "--mem", "auto");   /* MX-fp8 gradients passed their stair (3 seeds, mean 0.293 vs 0.294) */
        const int auto16 = g_sheet || !strcmp(mm, "auto16") || !unet_act_mx();   /* sheet reference keeps FP16 stem, hence 16-bit gradients */
        /* MX gradient modes store MX-fp4 activation gradients by default (paired 500-step runs: same step time, -2.9 GB per desk GPU,
           validation loss +0.1..0.8%) where the fp4 kernels apply: widths in 32-channel blocks, an even coarsest level;
           UFSM_GRAD_MX4=0 keeps MX-fp8 */
        {
            /* the stride-2 down convs' MX-fp4 weight gradients need every downsampled width a multiple of 4 (lp_bwd_w_s2_f8): the coarsest,
               P / 2^(nlev-1), included; P % 2^nlev let P 320 / 448 through to an abort */
            int g4ok = P % (4 << (cfg.nlev - 1)) == 0 && g_Pz % (4 << (cfg.nlev - 1)) == 0 && fp4 >= 2 && !*opt(argc, argv, "--policy", "");   /* the fp8 weight gradients of --fp4 1
                                                                                                     (or a --policy) take no MX-fp4 gy */
            /* 16- and 8-channel levels too (paired 2000-step desk check, production recipe: val 0.6506 vs 0.6534 MX-fp8, +3% speed, 5.8 vs 9.7 GB) */
            const int g4n = getenv("UFSM_G4_NARROW") ? ufsm_env_on("UFSM_G4_NARROW") : 1;   /* default on since the paired check (2026-10-15); =0 keeps MX-fp8 there */
            for (int i = 0; i < cfg.nlev; i++) if (cfg.widths[i] % 32 && !(g4n && (cfg.widths[i] == 16 || cfg.widths[i] == 8))) g4ok = 0;
            for (int i = 0; i < cfg.nlev - 1; i++) if (cfg.dec_widths[i] > 0 && cfg.dec_widths[i] % 32 && !(g4n && (cfg.dec_widths[i] == 16 || cfg.dec_widths[i] == 8))) g4ok = 0;
            const int g4 = getenv("UFSM_GRAD_MX4") ? ufsm_env_on("UFSM_GRAD_MX4") : g4ok;
            if (g4 && !g4ok) { fprintf(stderr, "UFSM_GRAD_MX4: MX-fp4 gradients need --fp4 2 (fp4 weight gradients), widths that are multiples of 32 and P divisible by %d\n", 4 << (cfg.nlev - 1)); return 2; }
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
            const shape5 xs = g_half ? (shape5){B, cfg.cin, P / g_hdiv, P / g_hdiv, P / g_hdiv} : (shape5){B, cfg.cin, split ? g_Dl : g_Pz, P, P};   /* split: this GPU's slab of the window */
            const size_t p3 = p3l;
            int pick = -1, pick_hoff = 1; size_t need = 0, pick_off = 0, pick_need = 0;
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
                if (g_half && cand[c].lean) continue;   /* (the batch buffers are full resolution, the network's half) */
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
                const size_t xinb = (size_t)B * xsc * p3 * xbytes(), batb = (size_t)B * g_tc * p3 + (size_t)B * p3;
                const shape5 g0s = {B, cfg.widths[0], (int)(p3 / ((size_t)P * P)), P, P};
                const size_t g0b = cand[c].gmx ? nn_mx_bytes(g0s, unet_grad_mx4() ? 4 : 8) : (size_t)B * cfg.widths[0] * p3 * 2;
                const int l3 = unet_last_dry_l3b();   /* three level-0 buffers: the tail inside X3 (counted), the input in X2 when it fits */
                const int xin_sep = cand[c].lean >= 2 && cand[c].gmx && (l3 ? xinb > g0b : !(unet_input_converted() && g_xfmt && xinb + batb + 8192 <= g0b));
                if (cand[c].lean) trainer = (cand[c].gmx ? (xin_sep ? xinb : 0) : xinb) + radb   /* batch buffers inside the gradient buffer B / gout[0] (estimate) */
                                            + (g_cout > NCH && !g_gl_inplace ? (size_t)B * g_cout * p3 * (g_g16 ? 2 : 4) : 0)
                                            + (g_head_slab ? (size_t)B * (NCH + 1) * p3 : 0);   /* head slabs: targets and mask outside the gradient buffers */   /* > 2 outputs (affinities, verso): the logit gradient may not fit beside the logits */
                /* the fp4 weight gradient's gy records of a whole level-0 layer (lp_gn_silu_bwd_apply_rec, MX-fp4 gradients): 136 B per
                   channel x 256 voxels (~0.5 B per voxel and channel), outside the dry build. 8 / 16-channel levels only with UFSM_GYREC=2
                   (8.5 B per level-0 voxel at 16 channels for ~0 step time there; 8 channels: -2%) */
                const size_t grec = unet_grad_mx4() && (cfg.widths[0] % 32 == 0 || lp_gyrec_narrow()) && (getenv("UFSM_GYREC") ? atoi(getenv("UFSM_GYREC")) : 1) ? (size_t)B * cfg.widths[0] * p3 * 136 / 256 : 0;
                need = grec + tb + tb / 14 + trainer + (nn_get_loss_tol() ? (size_t)B * p3 : 0) + (g_aff ? (size_t)2 * B * p3 : 0) + (g_unwrap ? (g_head_slab ? (size_t)4 * B * p3 + (size_t)g_hs_z * P * P * (2 * g_cout + 11) + 4096 : (size_t)15 * B * p3) : 0) + (g_cond ? (size_t)7 * B * p3 : 0) + (g_pap_head ? (size_t)4 * B * p3 : 0) + ((size_t)550 << 20) + (split ? (size_t)4 * B * 32 * P * P * 2 : 0);   /* split: two slots of halo send / receive planes */   /* kernel workspaces: measured device growth / tracked ~1.07 + 0.3 GB, + 0.2 GB margin for other processes */
                if (need <= fmin) { pick = c; pick_hoff = pass; pick_off = unet_last_offload_bytes(); pick_need = need; }
            }
            if (pick < 0) { pick = auto16 ? nc - 1 : 8; /* the smallest mode of the list */ pick_hoff = 1; fprintf(stderr, "memory: no mode fits %.2f GB free (smallest needs %.2f GB); trying %s\n", fmin / 1e9, need / 1e9, cand[pick].what); }
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
    if (g_half && lean) { fprintf(stderr, "--half-res: not with the lean memory modes\n"); return 2; }
    gpu_input_settle(&sc);   /* (the memory mode is set: the input layout of the batches) */
    for (int g = 0; g < ng; g++) {   /* per-GPU batch buffers. lean: one, shared by both slots, inside the model's gradient buffers when
                                        they are large enough: [targets | mask | input if the forward copies it] in B (lean 1: after the
                                        logit gradient) or in gout[0] (lean 2); the logit gradient in B (lean 1) or after the logits in A */
        gpu_state *d = &G[g];
        nn_init(d->dev);
        char *bs = nullptr;
        const size_t p3 = p3l;   /* this GPU's voxels of a window */
        const size_t a256 = 255, glb = ((size_t)B * g_cout * p3 * (g_g16 ? 2 : 4) + a256) & ~a256, tbb = ((size_t)B * g_tc * p3 + a256) & ~a256,
                     mbb = ((size_t)B * p3 + a256) & ~a256, xbb = ((size_t)B * g_xs * p3 * xbytes() + a256) & ~a256;
        const int xin_b = lean && unet_input_converted() && g_xfmt;
        char *tail = nullptr;   /* three level-0 buffers: [targets | mask] apart from the input */
        size_t goff = 0;
        int xin_in = 0;
        d->gl = nullptr; d->gl_inside = 0; d->gl_inplace = g_gl_inplace;
        if (lean) {
            unet_set_batch_bytes(d->u, xin_b ? xbb : 0, tbb + mbb);
            unet_build(d->u, (shape5){B, cfg.cin, split ? g_Dl : g_Pz, P, P}, 1);
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
            else { d->xb[i] = nn_malloc((size_t)B * g_xs * p3 * xbytes()); d->tb[i] = nn_malloc((size_t)B * g_tc * p3); d->mb[i] = nn_malloc((size_t)B * p3); d->wb[i] = nn_malloc((size_t)B * g_tc); d->radb[i] = g_gpu_in ? nn_malloc(rad_bytes(B)) : nullptr; }
            d->hwh[i] = g_heads ? nn_host_alloc((size_t)B * g_tc) : nullptr;
            d->ev_up[i] = nn_event_create();
        }
        if (g_head_slab && bs) {   /* head slabs read the targets and mask in the backward (head_slab_cb), after the head's backward has
                                      started writing the gradient buffers: their own buffers (+3 B per voxel), the input stays inside */
            d->tb[0] = d->tb[1] = nn_malloc(tbb); d->mb[0] = d->mb[1] = nn_malloc(mbb);
            if (!lean) { d->tb[1] = nn_malloc(tbb); d->mb[1] = nn_malloc(mbb); }
        }
        if (lean && g == 0) fprintf(stderr, "lean %d: batch buffers %s, logit gradient %s\n", lean, !bs ? "separate" : tail ? (xin_in ? "in the three level-0 buffers (input in X2, targets / mask at X3's end)" : "at X3's end (three level-0 buffers), input separate")
                                       : xin_in ? "and input inside the gradient buffers" : "inside the gradient buffers, input separate", tail && d->gl_inplace ? "over the logits" : d->gl ? "inside" : "separate");
        for (int i = 0; i < 2; i++) { d->ev_done[i] = nn_event_create(); nn_event_record(d->ev_done[i], 0); }
        d->pending = nullptr; select_buf(d, 0);
        if (!d->gl && !d->gl_inplace) d->gl = nn_malloc((size_t)B * g_cout * p3 * (g_g16 ? 2 : 4));
        for (int j = 1; j <= g_ds; j++) {   /* deep supervision: pooled targets / mask, fp32 logit gradient and loss scratch per level */
            const shape5 sj = {B, g_cout, g_Pz >> j, P >> j, P >> j}; const size_t sv = (size_t)sj.d * sj.h * sj.w;
            d->ds_t[j] = nn_malloc((size_t)B * g_tc * sv); d->ds_m[j] = nn_malloc((size_t)B * sv);
            d->ds_gl[j] = nn_malloc(shape_numel(sj) * 4); d->ds_scr[j] = nn_malloc(nn_loss_scratch(sj));
        }
        if (g_half) {
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
            d->pmb[0] = nn_malloc((size_t)B * p3); d->pmb[1] = lean ? d->pmb[0] : nn_malloc((size_t)B * p3); d->pm = d->pmb[d->cur];
            const size_t tp3 = g_head_slab ? (size_t)g_hs_z * P * P : (size_t)B * p3;
            d->nt = nn_malloc((size_t)7 * tp3); d->nm = nn_malloc(tp3);
            if (g_head_slab) {
                d->hs_z = g_hs_z; d->hs_P = P;
                d->hs_lg = nn_malloc((size_t)g_cout * tp3 * 2); d->hs_ds = nn_malloc(64 * sizeof(double));
                unet_set_head_cb(d->u, head_slab_cb, d);
                if (g == 0) fprintf(stderr, "head slabs: %d planes per slab (%.0f MB of logits and targets instead of %.2f GB)\n", g_hs_z,
                                    (double)tp3 * (2 * g_cout + 11) / 1e6, (double)B * p3 * (2 * g_cout + 11) / 1e9);
            }
            d->nw = nn_malloc(7); static const uint8_t seven_w[7] = {1, 1, 1, 1, 1, 1, 1}; nn_h2d(d->nw, seven_w, 7);
            d->iscratch = nn_malloc(nn_loss_scratch((shape5){1, 1, g_Pz, P, P}) + 64);
            d->nscratch = nn_malloc(nn_loss_scratch((shape5){1, 7, g_Pz, P, P}) + 64);
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
        d->tolc = nn_get_loss_tol() ? nn_malloc((size_t)B * p3) : nullptr;
        d->lean = lean;
        const char *e = nn_check(); if (e) { fprintf(stderr, "GPU %d: %s\n", d->dev, e); return 1; }
    }
    nn_init(G[0].dev);
    size_t np = unet_nparams(G[0].u);
    checkpoint_runtime runtime = {.version = 1, .train_window = P, .prec = nn_get_prec(), .f16 = nn_get_f16(),
        .act_mx4 = unet_act_mx4(), .act_mx8 = unet_act_mx() && !unet_act_mx4(), .grad_mx8 = unet_grad_mx8(), .input_mx = unet_input_converted(), .input_prec = unet_input_prec(), .gn_stored = nn_get_gn_stored()};
    const char *saved_policy = opt(argc, argv, "--policy", "");
    if (!*saved_policy) saved_policy = fp4 >= 2 ? "all=fp4:fp4:fp4,enc0.c1=fp16" : fp4 ? "all=fp4:fp4:fp8,enc0.c1=fp16" : "";
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
    if (g_unwrap) {   /* channel layout of the unwrap heads */
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
    if (g_Pz != P) fprintf(stderr, ": %zu params; window %dx%dx%d (slab) B=%d x %d GPU(s) [", np, g_Pz, P, P, B, ng); else fprintf(stderr, ": %zu params; P=%d B=%d x %d GPU(s) [", np, P, B, ng); for (int g = 0; g < ng; g++) fprintf(stderr, "%s%d", g ? "," : "", devs[g]); fprintf(stderr, "] %s%s\n", nn_get_tf32() ? "tensor cores (see precision.txt for compute and storage)" : "fp32", split ? ", each window split along z" : "");
    split_ctx *sctx = split ? split_create(devs[0], devs[1]) : nullptr;
    nn_init(G[0].dev);
    float *gpeer = ng > 1 ? nn_malloc(np * 4) : nullptr;   /* on GPU 0: incoming gradients of the other GPUs */

    /* validation set: fixed batches from the held-out boxes (unaugmented), scored on GPU 0 */
    sample_cfg vc = sc; vc.cover = nullptr; vc.cover_start = 0; vc.ct_augment = 0; vc.axis_jitter = 0; vc.geometry_augment=0; vc.label_morph=0; vc.scan.strength=0; vc.affine_p=0; vc.seed = seed + 777; vc.augment = 0; vc.nworkers = 4; vc.nbuf = 2; vc.holdout = 1;
    { int any = 0; for (int i = 0; i < S->n; i++) any |= S->src[i].hold_n[0] > 0; if (!any) { vc.holdout = 0; fprintf(stderr, "no holdout boxes in the sources: validation batches come from the training distribution\n"); } }
    /* the validation windows are the first nval batches of a deterministic sampler (each window from its own seeded rng: the same
       windows in every run with this seed), drawn once here. --val-store disk (default): kept in an unlinked file in the run directory
       (3 B per voxel each with --gpu-input and packed targets; its space returns when the trainer exits) and read back at every pass
       through one host batch; ram: copies in host RAM (the former layout); regen: drawn again at every pass (no copies; the pass waits
       for 3 sampler workers). The sheet prior's per-window records keep ram. */
    vc.deterministic = 1; vc.nworkers = 3; vc.nbuf = 3;
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
            const int nf = val_fields(b, B, p3, fb, sz); val_fields(&val[i], B, p3, fv, sz);
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
            const int nf = val_fields(&vdk.buf, B, p3, fv, sz);
            for (int k = 0; k < nf; k++) if (!(*fv[k] = malloc(sz[k]))) { fprintf(stderr, "out of host memory (validation batch)\n"); return 1; }
            fdatasync(vdk.fd); posix_fadvise(vdk.fd, 0, 0, POSIX_FADV_DONTNEED);
        }
        fprintf(stderr, "%d validation batches, %s (--val-store %s)\n", nval, vmode == VAL_RAM ? "held in host RAM" : "in an unlinked file in the run directory, read back at each pass", vmode_s);
    } else if (nval > 0) fprintf(stderr, "%d validation batches, drawn again at every validation pass (--val-store regen)\n", nval);
    else fprintf(stderr, "0 validation batches\n");

    if (!sc.cover && step0 > 0) {   /* resumed: a fresh sample stream (the per-sample seeds count from batch 0 at every sampler start, so a
                                       resume under --det replayed the windows and augmentations of steps 1, 2, ...); validation keeps its seed */
        sc.seed = seed + 0x9E3779B97F4A7C15ull * (uint64_t)step0;
        fprintf(stderr, "resumed at step %d: training sample stream reseeded\n", step0);
    }
    if (!atoi(opt(argc, argv, "--sampler-slots", "0"))) {   /* the default slot count against the host memory available now */
        const size_t p3s = (size_t)g_Pz * P * P, slotb = (size_t)B * p3s * ((size_t)g_xs * xbytes() + (g_pack ? 1 : NCH + 1) + (g_aff != 0) + (g_side != 0)
                           + (g_unwrap != 0) + (g_cond != 0) + (g_pap_head != 0)) + rad_bytes(B), win = slotb + 4 * p3s;   /* + a worker's scratch */
        const size_t avail = host_mem_available(), budget = avail / 10 * 7;
        int cap = win ? (int)(budget / win) : sc.nbuf;
        if (cap < 3) cap = 3;   /* (one uploading, two drawing) */
        if (avail && cap < sc.nbuf) {
            fprintf(stderr, "sampler: %d slots (not %d): %.2f GB per window in flight (slot + worker scratch), %.1f GB host memory available; --sampler-slots sets it\n",
                    cap, sc.nbuf, win / 1e9, avail / 1e9);
            sc.nbuf = cap;
        }
    }
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
    double best_val = 1e30; int nskip = 0; (void)nskip;
    float parts[8][2 * TCMAX + 1];
    double acc_ds = 0, acc_vbce = 0, acc_vdice = 0, acc_side = 0, acc_ink = 0, acc_norm = 0, acc_cond = 0, acc_ck = 0, acc_cr = 0, acc_pap = 0, acc_cnt = 0, acc_cnr = 0;   /* (acc_cnt / acc_cnr: --count-lambda loss, Tp / Tg) verso channel, side channel; --cond (loss, known / right counts); --pap-head */
    for (int step = step0 + 1; step <= steps && !g_stop; step++) {
        g_verso_on = g_verso && step >= g_verso_start;
        { uint64_t h = (uint64_t)step * 0x9E3779B97F4A7C15ull + (uint64_t)seed; h ^= h >> 31; h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 29;
          g_verso_flip = g_verso_on && g_verso_swap > 0 && (double)(h >> 11) * 0x1p-53 < g_verso_swap; }   /* same on every GPU */
        g_sheet_ramp=fminf(1.f,(float)(step-schedule_start)/500.f);
        nn_set_sr_step((unsigned)step);
        double loss = 0, active = 0;
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
                    cp->t = malloc((size_t)B * NCH * p3b); memcpy(cp->t, b->t, (size_t)B * NCH * p3b);
                    cp->m = malloc((size_t)B * p3b); memcpy(cp->m, b->m, (size_t)B * p3b);
                    cp->w = malloc((size_t)B * NCH); memcpy(cp->w, b->w, (size_t)B * NCH);
                    cp->src = malloc((size_t)B * sizeof *b->src); memcpy(cp->src, b->src, (size_t)B * sizeof *b->src);
                    cp->um = malloc((size_t)B * sizeof(float)); memcpy(cp->um, b->um, (size_t)B * sizeof(float));
                    sampler_release(sp, b); fixed[nfixed++] = cp;
                }
                upload(d, fixed[k], B, P);
                unet_zero_grad(d->u); run_batch(d, B, P, dice_w, 1); unet_backward_x(d->u, g_half ? d->glh : d->gl, g_g16);
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
                unet_zero_grad(d->u); run_batch(d, B, P, dice_w, 1); unet_backward_x(d->u, g_half ? d->glh : d->gl, g_g16);
                continue;
            }
            /* the batch for this step was uploaded during the previous step (first step: upload now) */
            if (!d->pending) {
                double tw = now();
                batch *b = sampler_next(sp);
                wait += now() - tw;
                if (!b) { fprintf(stderr, "sampler stopped\n"); g_stop = 1; break; }
                upload_async(d, b, d->cur, B, P);
                d->pending = b;
            }
            nn_stream_wait(0, d->ev_up[d->cur]);           /* compute waits for the upload */
            if (sparse_at && step >= sparse_at && !unet_get_sparse24(d->u)) { unet_set_sparse24(d->u, 1); if (g == 0) fprintf(stderr, "step %d: 2:4 sparsity on (SR-STE lambda %g)\n", step, srste); }
            unet_zero_grad(d->u);
            run_batch(d, B, P, dice_w, 1);
            unet_backward_x(d->u, g_half ? d->glh : d->gl, g_g16);
            nn_event_record(d->ev_done[d->cur], 0);
            nn_event_sync(d->ev_up[d->cur]);                 /* host buffer of this batch is free again */
            if (overfit) continue;                           /* keep computing on the same device buffer */
            sampler_release(sp, d->pending);
            /* prefetch the next batch into the other buffer while this step computes */
            d->pending = nullptr;
            if (step < steps && !g_stop) {
                double tw = now(); batch *nb = sampler_next(sp); wait += now() - tw;
                if (!nb) { fprintf(stderr, "sampler stopped before final step\n"); training_failed = 1; break; }
                int nxt = d->cur ^ 1; upload_async(d, nb, nxt, B, P); d->pending = nb; select_buf(d, nxt);
            }
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
        if (fwd_nan && !overfit && !split) diagnose_nan(&G[0], B, P, step, out);
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
        if (training_failed) break;
        committed_step = step;
        if (plan) { coverage.cursor = (uint64_t)(step - coverage.base_step); if (cover_checkpoint_extra(runtime_extra, &coverage, checkpoint_extra, sizeof checkpoint_extra)) return 2; }
        const char *e = nn_check();
        if (e) { fprintf(stderr, "cuda error at step %d: %s\n", step, e); return 1; }
        if ((seconds && now() - t0 >= seconds) || (schedule_seconds && schedule_elapsed+now()-t0>=schedule_seconds) ||
            (limit_seconds && now() - t0 >= limit_seconds) || (stop_at && step >= stop_at)) g_stop = 1;
        acc_loss += loss / nl; acc_bce += parts[0][0]; acc_dice += parts[0][g_aff || g_unwrap || g_heads ? NCH : g_half ? g_cout : cfg.cout]; acc_g += gn; nacc++;
        if (g_verso_on) { acc_vbce += G[0].vout[0]; acc_vdice += G[0].vout[1]; }
        if (g_side) acc_side += G[0].sout[0];
        if (g_unwrap) { acc_ink += G[0].iout[0]; acc_norm += G[0].norm_loss / (g_norm_lambda ? g_norm_lambda : 1.f); }
        if (g_ds) acc_ds += G[0].ds_loss;
        if (g_cond) { acc_cond += G[0].cond_loss / (g_cond_lambda ? g_cond_lambda : 1.f); acc_ck += (double)G[0].cacc_h[0]; acc_cr += (double)G[0].cacc_h[1]; }
        if (g_pap_head) acc_pap += G[0].pout3[0];
        if (g_count_lambda > 0) { acc_cnt += G[0].cnt_h[0]; acc_cnr += G[0].cnt_h[2]; }
        if (step % log_every == 0 || step == steps || g_stop) {
            double dt = now() - tlog;
            double vl = -1, vb = 0, vd = 0, vh[TCMAX] = {0}; int vhn[TCMAX] = {0};
            char vstr[128] = ",,";
            if (step % val_every == 0 || step == steps || g_stop) {
                const int nv = split ? 2 : 1;   /* split: validation windows are split too */
                for (int g = 0; g < nv; g++) { nn_init(G[g].dev); unet_use_ema(G[g].u, 1); }
                {   /* validation on the buffer not holding the prefetched batch (its upload has completed: ev_up synced above) */
                    int keep = G[0].cur;
                    for (int g = 0; g < nv; g++) { if (G[g].lean) { nn_init(G[g].dev); nn_event_sync(G[g].ev_up[keep]); } select_buf(&G[g], keep ^ 1); }   /* lean: one buffer, the prefetch must have landed */
                    g_verso_flip = 0;   /* validation: the direction as is */
                    sampler *vq = nval > 0 && vmode == VAL_REGEN ? sampler_start(S, &vc) : nullptr;   /* --val-store regen: the same windows drawn again */
                    if (nval > 0 && vmode == VAL_REGEN && !vq) { fprintf(stderr, "cannot construct validation sampler\n"); return 1; }
                    int got = 0;
                    for (int i = 0; i < nval; i++) {
                        float vp[2 * TCMAX + 1];
                        batch *vbt = vq ? sampler_next(vq) : vmode == VAL_DISK ? (val_read(&vdk, i, B, p3) ? nullptr : &vdk.buf) : &val[i];
                        if (!vbt) { if (vmode == VAL_DISK) { fprintf(stderr, "cannot read the validation windows\n"); return 1; } break; }
                        for (int g = 0; g < nv; g++) { nn_init(G[g].dev); upload(&G[g], vbt, B, P); }
                        if (split) { split_arg sa = {G, B, P, 0, dice_w}; split_run(sctx, split_job, &sa); nn_init(G[0].dev); }
                        else run_batch(&G[0], B, P, dice_w, 0);
                        double l = fetch_loss(&G[0], B, P, dice_w, vp); vl = (vl < 0 ? 0 : vl) + l; vb += vp[0]; vd += vp[g_aff || g_unwrap || g_heads ? NCH : g_half ? g_cout : cfg.cout];
                        if (g_heads) for (int h = 0; h < g_heads; h++) if (G[0].hon[h]) { vh[h] += G[0].hl[h]; vhn[h]++; }
                        if (vq) sampler_release(vq, vbt);
                        got++;
                    }
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
                snprintf(vstr, sizeof vstr, "%.5f,%.5f,%.5f", vl, vb, vd);
                if (g_heads) {   /* validation loss per head over the windows of its own sources */
                    fprintf(stderr, "val heads:");
                    for (int h = 0; h < g_heads; h++) fprintf(stderr, " %d %.5f (%d)", h, vhn[h] ? vh[h] / vhn[h] : 0.0, vhn[h]);
                    fprintf(stderr, "\n");
                }
                if (nval && vl < best_val) { best_val = vl; char bp[1400]; snprintf(bp, sizeof bp, "%s/best.ckpt", out); if (unet_save(G[0].u, bp, step, checkpoint_extra)) { fprintf(stderr, "cannot save %s\n", bp); return 1; } }
            }
            char affs[320] = "";
            if (g_verso) snprintf(affs, 64, " verso bce %.4f dice %.4f", acc_vbce / nacc, acc_vdice / nacc);
            if (g_side) { size_t na = strlen(affs); snprintf(affs + na, sizeof affs - na, " %s bce %.4f", g_side == 2 ? "phase" : "side", acc_side / nacc); }
            if (g_unwrap) { size_t na = strlen(affs); snprintf(affs + na, sizeof affs - na, " ink bce %.4f normal bce %.4f", acc_ink / nacc, acc_norm / nacc); }
            if (g_ds) { size_t na = strlen(affs); snprintf(affs + na, sizeof affs - na, " ds %.4f", acc_ds / nacc); }
            if (g_cond) { size_t na = strlen(affs); snprintf(affs + na, sizeof affs - na, " cond bce %.4f acc %.3f", acc_cond / nacc, acc_ck > 0 ? acc_cr / acc_ck : -1.0); }
            if (g_pap_head) { size_t na = strlen(affs); snprintf(affs + na, sizeof affs - na, " papyrus bce %.4f", acc_pap / nacc); }
            if (g_count_lambda > 0) { size_t na = strlen(affs); snprintf(affs + na, sizeof affs - na, " count %.4f (turns pred/true %.3f)", acc_cnt / nacc, acc_cnr / nacc); }
            if (g_aff) { const float *a = G[0].aff_out; size_t na = strlen(affs); snprintf(affs + na, sizeof affs - na, " aff %.4f (d1 bce %.3f dice %.3f | d8 bce %.3f dice %.3f)", G[0].aff_loss,
                                    (a[0] + a[6] + a[12]) / 3, (a[1] + a[7] + a[13]) / 3, (a[18] + a[24] + a[30]) / 3, (a[19] + a[25] + a[31]) / 3); }
            fprintf(stderr, "step %6d lr %.2e loss %.4f bce %.4f dice %.4f%s gn %.2f %.2f samp/s (wait %.0f%%)%s%s\n", step, lr, acc_loss / nacc, acc_bce / nacc, acc_dice / nacc, affs, acc_g / nacc,
                    (double)nacc * B * nl / dt, 100 * wait / dt, vl >= 0 ? " val " : "", vl >= 0 ? vstr : "");
            if (log) { fprintf(log, "%d,%.3e,%.5f,%.5f,%.5f,%.0f,%.4f,%s,%.3f,%.3f\n", step, lr, acc_loss / nacc, acc_bce / nacc, acc_dice / nacc, active, acc_g / nacc, vstr, (double)nacc * B * nl / dt, wait); fflush(log); }
            if (prof) { fprintf(stderr, "per-op GPU ms over the last %d steps (all GPUs):\n", log_every); unet_prof_report(); }
            acc_ds = 0; acc_loss = acc_bce = acc_dice = acc_g = acc_vbce = acc_vdice = acc_side = acc_ink = acc_norm = acc_cond = acc_ck = acc_cr = acc_pap = acc_cnt = acc_cnr = 0; nacc = 0; tlog = now(); wait = 0;
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
    fprintf(stderr, "done in %.0fs; sampler produced %llu, rejected %llu\n", now() - t0, (unsigned long long)prod, (unsigned long long)rej);
    sampler_prof_print(sp);   /* UFSM_SAMPLER_PROF=1: per-stage cpu ms per patch */
    int sampling_failed = sampler_failed(sp);
    sampler_stop(sp);
    if (log) fclose(log);
    if (sheet_log) fclose(sheet_log);
    cover_free(plan);
    sheet_free(g_sheet); g_sheet=nullptr;
    return sampling_failed || training_failed ? 1 : 0;
}
