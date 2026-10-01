/* ufsm train: sampler -> pinned batches -> one or more GPUs (data parallel, gradients averaged through the
   host) -> UNet -> BCE+Dice -> AdamW/EMA, with held-out validation batches, CSV log and checkpoints. */
#include "nn.h"
#include "sample.h"
#include "sources.h"
#include "split.h"
#include "unet.h"
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

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
    const float *lg;             /* logits of the last run_batch */
    uint8_t *tb[2], *mb[2], *wb[2];
    float *x; uint8_t *t, *m, *w; /* the buffers of the batch being computed */
    void *ev_up[2], *ev_done[2]; /* upload into buffer i finished; the compute that last used buffer i finished */
    batch *pending;              /* host batch whose upload into buffer (cur ^ 1) is in flight */
    int lean;                    /* one batch buffer; logit gradient in the model's gradient buffer B */
    int cur;
    int side;                    /* --split z: 0 = low z half, 1 = high z half of every window; -1 = whole windows */
} gpu_state;
/* --split z: both GPUs work on the same window, each on its z half plus g_h0 halo planes (depth g_Dl); the mask is zeroed on
   the halo planes (from a pinned zero buffer) so each voxel counts once */
static int g_Dl, g_h0; static uint8_t *g_zeros;

static int g_xfmt = 0, g_g16 = 0;   /* input batches uploaded as 16-bit (1 fp16, 2 bf16); loss gradient written as 16-bit */
static size_t xbytes(void) { return g_xfmt ? 2 : 4; }
static const void *bx(const batch *b) { return g_xfmt ? (const void *)b->x16 : (const void *)b->x; }
static void select_buf(gpu_state *d, int i) { d->cur = i; d->x = d->xb[i]; d->t = d->tb[i]; d->m = d->mb[i]; d->w = d->wb[i]; }
/* this GPU's z slab of a batch: rows of the [rows][P][P][P] arrays, planes [z0, z0 + g_Dl); mask halo planes from zeros */
static void upload_slab(gpu_state *d, const batch *b, int i, int B, int P, int async) {
    void (*cp)(void *, const void *, size_t) = async ? nn_h2d_copy_stream : nn_h2d;
    const size_t p2 = (size_t)P * P, p3 = p2 * P, l3 = p2 * g_Dl, own = l3 - g_h0 * p2;
    const size_t z0 = d->side ? (size_t)(P / 2 - g_h0) : 0, xb = xbytes();
    for (int r = 0; r < B * 4; r++) cp((char *)d->xb[i] + r * l3 * xb, (const char *)bx(b) + (r * p3 + z0 * p2) * xb, l3 * xb);
    for (int r = 0; r < B * NCH; r++) cp(d->tb[i] + r * l3, b->t + r * p3 + z0 * p2, l3);
    for (int n = 0; n < B; n++) {
        uint8_t *m = d->mb[i] + n * l3;
        if (d->side) { cp(m, g_zeros, g_h0 * p2); cp(m + g_h0 * p2, b->m + n * p3 + (z0 + g_h0) * p2, own); }
        else { cp(m, b->m + n * p3, own); cp(m + own, g_zeros, g_h0 * p2); }
    }
    cp(d->wb[i], b->w, (size_t)B * NCH);
}
/* synchronous upload (pageable host memory: validation batches) */
static void upload(gpu_state *d, const batch *b, int B, int P) {
    if (d->side >= 0) { upload_slab(d, b, d->cur, B, P, 0); return; }
    size_t p3 = (size_t)P * P * P;
    nn_h2d(d->x, bx(b), (size_t)B * 4 * p3 * xbytes());
    nn_h2d(d->t, b->t, (size_t)B * NCH * p3);
    nn_h2d(d->m, b->m, (size_t)B * p3);
    nn_h2d(d->w, b->w, (size_t)B * NCH);
}
/* asynchronous upload of a pinned sampler batch into buffer i on the copy stream, after the compute that last used it */
static void upload_async(gpu_state *d, const batch *b, int i, int B, int P) {
    size_t p3 = (size_t)P * P * P;
    nn_stream_wait(1, d->ev_done[i]);   /* the step two back: this upload overlaps the current step */
    if (d->lean) nn_stream_wait(1, d->ev_done[i ^ 1]);   /* lean: one buffer (inside the gradient buffers), free after the current step */
    if (d->side >= 0) upload_slab(d, b, i, B, P, 1);
    else {
        nn_h2d_copy_stream(d->xb[i], bx(b), (size_t)B * 4 * p3 * xbytes());
        nn_h2d_copy_stream(d->tb[i], b->t, (size_t)B * NCH * p3);
        nn_h2d_copy_stream(d->mb[i], b->m, (size_t)B * p3);
        nn_h2d_copy_stream(d->wb[i], b->w, (size_t)B * NCH);
    }
    nn_event_record(d->ev_up[i], 1);
}

/* a non-finite forward: statistics of the input, targets and logits of the batch just computed (device buffer cur^1,
   since select_buf already moved on) and a raw dump <out>/nan_step<N>.bin (x16|x32, t, m, w, logits) */
static void diagnose_nan(gpu_state *d, int B, int P, int step, const char *out) {
    nn_init(d->dev); nn_sync();
    size_t p3 = (size_t)P * P * P, nx = (size_t)B * 4 * p3, nl = (size_t)B * NCH * p3;
    int cb = d->cur ^ 1;
    size_t xbytes_ = nx * xbytes();
    void *hx = malloc(xbytes_); nn_d2h(hx, d->xb[cb], xbytes_);
    uint8_t *ht = malloc((size_t)B * NCH * p3), *hm = malloc((size_t)B * p3), *hw = malloc((size_t)B * NCH);
    nn_d2h(ht, d->tb[cb], (size_t)B * NCH * p3); nn_d2h(hm, d->mb[cb], (size_t)B * p3); nn_d2h(hw, d->wb[cb], (size_t)B * NCH);
    float *hl = malloc(nl * 4); nn_d2h(hl, d->lg, nl * 4);
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
    for (size_t k = 0; k < (size_t)B * NCH * p3; k++) tpos += ht[k] != 0 && ht[k] != 255;
    fprintf(stderr, "step %d: non-finite forward: input non-finite %zu max|x| %.3g; logits non-finite %zu max|logit| %.3g; mask %zu of %zu; target>0 %zu; w", step, xnf, xmax, lnf, lmax, mcount, (size_t)B * p3, tpos);
    for (int i = 0; i < B * NCH; i++) fprintf(stderr, " %d", hw[i]);
    fprintf(stderr, "\n");
    char fn[1500]; snprintf(fn, sizeof fn, "%s/nan_step%d.bin", out, step);
    FILE *f = fopen(fn, "wb");
    if (f) { int hdr[4] = {B, P, g_xfmt, NCH}; fwrite(hdr, 4, 4, f); fwrite(hx, 1, xbytes_, f); fwrite(ht, 1, (size_t)B * NCH * p3, f); fwrite(hm, 1, (size_t)B * p3, f); fwrite(hw, 1, (size_t)B * NCH, f); fwrite(hl, 4, nl, f); fclose(f); fprintf(stderr, "  batch dumped to %s\n", fn); }
    free(hx); free(ht); free(hm); free(hw); free(hl);
}
/* forward + loss kernels on the uploaded batch of this GPU (asynchronous); fills gl when train. */
static void run_batch(gpu_state *d, int B, int P, float dice_w, int train) {
    shape5 xs = {B, 4, d->side >= 0 ? g_Dl : P, P, P};
    const float *lg = unet_forward_x(d->u, d->x, xs, train, g_xfmt != 0);
    d->lg = lg;
    if (d->lean && train) {   /* the logit gradient lives in the model's gradient buffer B (built by the forward above) */
        const size_t gb = (size_t)B * NCH * xs.d * P * P * (g_g16 ? 2 : 4);
        void *pg = unet_logit_grad_scratch(d->u, gb);
        if (pg) d->gl = pg;   /* (re)built model: its scratch moved; otherwise the separate buffer allocated at setup */
    }
    shape5 os = unet_out_shape(d->u, xs);
    int prof = ufsm_env_on("UFSM_PROF");
    if (prof) nn_prof_begin(6);
    nn_loss_async(lg, d->t, d->m, d->w, os, dice_w, train ? d->gl : nullptr, d->scratch);
    if (prof) nn_prof_end();
}
/* fetch the loss parts of the last run_batch on this GPU (synchronous); returns the scalar loss */
static double fetch_loss(gpu_state *d, int B, int P, float dice_w, float *out) {
    shape5 os = unet_out_shape(d->u, (shape5){B, 4, d->side >= 0 ? g_Dl : P, P, P});
    nn_loss_fetch(d->scratch, os, out);
    int cout = unet_cfg_of(d->u)->cout;
    double loss = 0; int active = 0;
    for (int c = 0; c < cout; c++) if (out[c] > 0 || out[cout + c] > 0) { loss += out[c] + dice_w * out[cout + c]; active++; }
    return active ? loss / active : 0;
}

/* --split z: forward, loss and backward of one window on both GPUs (src/split.h) */
typedef struct { gpu_state *G; int B, P, train; float dice_w; } split_arg;
static void split_job(int side, void *a) {
    split_arg *s = a;
    gpu_state *d = &s->G[side];
    run_batch(d, s->B, s->P, s->dice_w, s->train);
    if (s->train) unet_backward_x(d->u, d->gl, g_g16);
}

int cmd_train(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: ufsm train <sources.json> --out DIR [--P 96] [--B 2] [--steps 20000] [--lr 1e-3] [--warmup 500] [--wd 0.01]\n"
                        "       [--dice 0.5] [--ema 0.999] [--widths 16,32,64,80] [--down-norm 1] [--gpus 0,1] [--workers 12] [--seed 0] [--det 1] [--resume CKPT] [--finetune 1] [--fp4 0|1|2 (default 1)] [--mem auto|auto16|default] [--opt adamw|muon|anvil] [--muon-lr 0.02] [--muon-beta 0.95] [--anvil-lr 0.023] [--anvil-wd 2.25] [--sched cos|wsd] [--cooldown 0.2]\n"
                        "       [--val-batches 8] [--log-every 20] [--val-every 500] [--ckpt-every 1000] [--clip 5] [--levels 0.5,0.25,0.15,0.1] [--fp32] [--f16 1] [--gscale 1024] [--prec 1|2|3|4] [--policy enc0=1,dec0.c1=fp16:fp16:fp8,...] [--qat 2|3] [--wq 8|4] [--sparse24 STEP] [--srste 2e-4] [--pos-weight 1] [--sr 1]\n"
                        "  B is the per-GPU batch; gradients are averaged across GPUs every step (effective batch B x ngpus).\n"
                        "  --split z --gpus 0,1: every window is split along z across the two GPUs instead (batch B; P a multiple of 2^nlev).\n"
                        "  env UFSM_PROF=1 prints per-op GPU time every log interval (category 'upload+loss+opt').\n");
        return 2;
    }
    const char *src = argv[2], *out = opt(argc, argv, "--out", "runs/run");
    int P = atoi(opt(argc, argv, "--P", "96")), B = atoi(opt(argc, argv, "--B", "2")), steps = atoi(opt(argc, argv, "--steps", "20000"));
    float lr0 = (float)atof(opt(argc, argv, "--lr", "1e-3")), wd = (float)atof(opt(argc, argv, "--wd", "0.01")), dice_w = (float)atof(opt(argc, argv, "--dice", "0.5"));
    float ema = (float)atof(opt(argc, argv, "--ema", "0.999")), clip = (float)atof(opt(argc, argv, "--clip", "5"));
    int warmup = atoi(opt(argc, argv, "--warmup", "500")), workers = atoi(opt(argc, argv, "--workers", "12"));
    int nval = atoi(opt(argc, argv, "--val-batches", "8")), log_every = atoi(opt(argc, argv, "--log-every", "20"));
    int val_every = atoi(opt(argc, argv, "--val-every", "500")), ckpt_every = atoi(opt(argc, argv, "--ckpt-every", "1000"));
    uint64_t seed = (uint64_t)atoll(opt(argc, argv, "--seed", "0"));
    const char *resume = opt(argc, argv, "--resume", nullptr);
    int f16 = atoi(opt(argc, argv, "--f16", "1"));       /* 16-bit storage/operands as fp16 (8x finer than bf16, same speed); 0 = bf16 */
    nn_set_prec(atoi(opt(argc, argv, "--prec", "1")));   /* 1 bf16, 2 fp8, 3 fp4 fwd + fp8 wgrad (2/3 force bf16 storage) */
    if (f16) { nn_set_f16(1); nn_set_grad_scale((float)atof(opt(argc, argv, "--gscale", "1024"))); }
    const char *optname = opt(argc, argv, "--opt", "adamw");   /* adamw | muon (3^3 conv weights: nesterov momentum + Newton-Schulz orthogonalisation; rest AdamW) */
    float muon_lr = (float)atof(opt(argc, argv, "--muon-lr", "0.02")), muon_beta = (float)atof(opt(argc, argv, "--muon-beta", "0.95"));
    int use_muon = !strcmp(optname, "muon"), use_anvil = !strcmp(optname, "anvil");
    float anvil_lr = (float)atof(opt(argc, argv, "--anvil-lr", "0.023")), anvil_wd = (float)atof(opt(argc, argv, "--anvil-wd", "2.25"));
    const char *sched = opt(argc, argv, "--sched", "cos");   /* cos | wsd (warmup, constant, linear cooldown over the last --cooldown fraction; extendable runs) */
    float cooldown = (float)atof(opt(argc, argv, "--cooldown", "0.2"));
    int qat = atoi(opt(argc, argv, "--qat", "0"));
    /* the effective precision manifest is printed after all precision options are applied and saved next to the checkpoints */
#define WRITE_MANIFEST(path) do { char mf_[4096]; int mn_ = nn_prec_manifest(mf_, sizeof mf_); snprintf(mf_ + mn_, sizeof mf_ - mn_, " act_mx4 %d act_mx8 %d grad_mx8 %d f16 %d opt %s\n", ufsm_env_on("UFSM_ACT_MX4"), ufsm_env_on("UFSM_ACT_MX8"), ufsm_env_on("UFSM_GRAD_MX8"), f16, optname); FILE *mff_ = fopen(path, "w"); if (mff_) { fputs(mf_, mff_); fclose(mff_); } } while (0)         /* quantization-aware training: forward/backward-data at precision 2 (fp8) or 3 (fp4), weight gradients at 16-bit */
    if (qat) { nn_set_prec(qat); nn_set_prec_wgrad(1); }
    if (atoi(opt(argc, argv, "--sr", "0"))) nn_set_sr(1);                                   /* stochastic rounding of fp8 gradient operands */
    int wq = atoi(opt(argc, argv, "--wq", "0"));            /* 8 or 4: true fp8 / fp4 weights (stochastic rounding after each update) */
    int sparse_at = atoi(opt(argc, argv, "--sparse24", "0"));   /* step from which the 3^3 conv weights are 2:4 sparse (0 = dense) */
    float srste = (float)atof(opt(argc, argv, "--srste", "2e-4"));
    nn_set_pos_weight((float)atof(opt(argc, argv, "--pos-weight", "1")));   /* BCE weight of surface voxels */
    int overfit = atoi(opt(argc, argv, "--overfit", "0"));   /* diagnostic: train on the first batch forever */
    int noaug = atoi(opt(argc, argv, "--noaug", "0"));       /* diagnostic: no augmentation */
    /* --fp4 1: the fastest configuration that passed the paired stairs on two seeds: packed fp4 activation storage, fp4 forward
       and backward-data, fp8 weight gradients, stochastic rounding of gradient operands, first conv 16-bit.
       --fp4 2: fp4 weight gradients as well (also passed; slower than fp8 until its kernel is optimised). An explicit --policy
       still overrides the precisions. */
    /* default since the r11b 40k confirmation (2026-10-01): --fp4 1 unless a precision is chosen explicitly
       (--prec, --wq, --fp32); --fp4 0 trains 16-bit */
    int expl = *opt(argc, argv, "--prec", "") || *opt(argc, argv, "--wq", "");
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--fp32")) expl = 1;
    const int fp4 = atoi(opt(argc, argv, "--fp4", expl ? "0" : "1"));
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
    unet_cfg cfg = {4, {16, 32, 64, 80}, 4, NCH, 8};
    cfg.down_norm = atoi(opt(argc, argv, "--down-norm", "0"));
    if (resume) { unet_cfg pc; int st; if (!unet_peek(resume, &pc, &st)) cfg.down_norm = pc.down_norm; }   /* the checkpoint decides */
    { char *t = strdup(opt(argc, argv, "--widths", "16,32,64,80")); cfg.nlev = 0; for (char *q = strtok(t, ","); q && cfg.nlev < UNET_MAXLEV; q = strtok(nullptr, ",")) cfg.widths[cfg.nlev++] = atoi(q); free(t); }
    if (P % (1 << (cfg.nlev - 1))) { fprintf(stderr, "P must be divisible by %d\n", 1 << (cfg.nlev - 1)); return 2; }
    const int split = !strcmp(opt(argc, argv, "--split", "0"), "z");
    if (split) {
        g_h0 = 1 << (cfg.nlev - 1); g_Dl = P / 2 + g_h0;
        if (ng != 2) { fprintf(stderr, "--split z needs --gpus with two devices\n"); return 2; }
        if (P % (2 * g_h0)) { fprintf(stderr, "--split z: P must be divisible by %d\n", 2 * g_h0); return 2; }
        if (overfit || getenv("UFSM_SYNC_UPLOAD") || !nn_get_tf32()) { fprintf(stderr, "--split z: not with --overfit, UFSM_SYNC_UPLOAD or --fp32\n"); return 2; }
        g_zeros = nn_host_alloc((size_t)g_h0 * P * P); memset(g_zeros, 0, (size_t)g_h0 * P * P);
    }
    const int nl = split ? 1 : ng;   /* windows per step (loss averaging, samples / s) */
    char cmd[1400]; snprintf(cmd, sizeof cmd, "mkdir -p '%s'", out); if (system(cmd)) return 1;

    sources *S = sources_load(src);
    if (!S) return 1;
    sample_cfg sc = sample_cfg_default();
    sc.P = P; sc.B = B; sc.nworkers = workers; sc.nbuf = 4 * ng + 2; sc.seed = seed; sc.xfmt = g_xfmt;
    sc.deterministic = atoi(opt(argc, argv, "--det", "0"));   /* same batches for the same seed regardless of thread timing (paired comparisons; off: a slow worker can stall the ring) */
    if (noaug) sc.augment = 0;
    if (atoi(opt(argc, argv, "--rotonly", "0"))) sc.augment = 2;   /* proper rotations only (no reflections) */
    if (atoi(opt(argc, argv, "--zfix", "0"))) sc.augment = 3;      /* diagnostic: symmetries that keep the z axis */
    if (atoi(opt(argc, argv, "--intonly", "0"))) sc.augment = 4;   /* diagnostic: intensity jitter, no symmetry */
    sc.dilate = atoi(opt(argc, argv, "--dilate", "0"));   /* thicken surface targets by D level-0 voxels (curriculum) */
    sc.soft = (float)atof(opt(argc, argv, "--soft", "0"));  /* soft ridge target with this sigma (level-0 voxels) */
    float soft_end = (float)atof(opt(argc, argv, "--soft-end", "-1")); if (soft_end < 0) soft_end = sc.soft;   /* sigma annealed linearly to this value at the last step */
    if (soft_end != sc.soft) fprintf(stderr, "soft target sigma annealed %g -> %g; the fixed validation batches keep sigma %g, so the validation loss (and best.ckpt) is not comparable across the run: score last.ckpt\n", sc.soft, soft_end, sc.soft);
    { const char *lv = opt(argc, argv, "--levels", nullptr); if (lv) { char *t = strdup(lv); int l = 0; memset(sc.level_p, 0, sizeof sc.level_p); for (char *q = strtok(t, ","); q && l < MAXLEV; q = strtok(nullptr, ",")) sc.level_p[l++] = atof(q); free(t); } }

    size_t p3 = (size_t)P * P * P, p3l = split ? (size_t)P * P * g_Dl : p3;   /* per-GPU voxels of a window */
    gpu_state G[8];
    int step0 = 0;
    for (int g = 0; g < ng; g++) {
        gpu_state *d = &G[g];
        d->dev = devs[g]; d->side = split ? g : -1;
        if (nn_init(d->dev)) { fprintf(stderr, "cannot select GPU %d\n", d->dev); return 1; }
        d->u = unet_create(&cfg);
        if (resume) { step0 = unet_load(d->u, resume); if (step0 < 0) { fprintf(stderr, "cannot load %s\n", resume); return 1; } }
        if (resume && atoi(opt(argc, argv, "--finetune", "0"))) step0 = 0;   /* weights from the checkpoint, fresh schedule (QAT / low-precision fine-tuning) */
        else unet_init(d->u, seed + 1);                   /* deterministic: every GPU starts identical */
        if (wq) unet_set_wq(d->u, wq);
        if (split) { unet_set_split(d->u, g, g_h0, split_halo); if (!getenv("UFSM_SPLIT_SYNC")) unet_set_split_async(d->u, split_halo_begin, split_halo_end); }
        const char *e = nn_check(); if (e) { fprintf(stderr, "GPU %d: %s\n", d->dev, e); return 1; }
    }
    int lean = getenv("UFSM_LEAN") ? atoi(getenv("UFSM_LEAN")) : 0;   /* lean: one device batch buffer (no upload overlap), the logit
                                                                     gradient in the model's gradient buffer, logits in A */
    if (resume) fprintf(stderr, "resumed %s at step %d\n", resume, step0);
    {   /* --mem auto (default): the cheapest storage mode whose training buffers fit next to what is already allocated on every
           GPU; an explicit UFSM_CHUNK_UP / UFSM_RECOMPUTE / UFSM_GRAD_MX8 or --mem default keeps the env / built-in modes */
        const char *mm = opt(argc, argv, "--mem", "auto");   /* MX-fp8 gradients passed their stair (3 seeds, mean 0.293 vs 0.294) */
        const int auto16 = !strcmp(mm, "auto16") || !unet_act_mx();   /* auto16 (or 16-bit activations): 16-bit gradients only */
        if ((!strcmp(mm, "auto") || auto16) && nn_get_tf32() && !getenv("UFSM_CHUNK_UP") && !getenv("UFSM_RECOMPUTE") && !getenv("UFSM_GRAD_MX8")) {
            static const struct { int chunk, rc, gmx, lean; const char *what; } cand[] = {
                /* by step cost (MX-fp8 gradients passed their 3-seed stair and cost nothing; lean costs ~nothing, the upload is
                   ~3% of a large-window step; recompute 2 ~10%), the 16-bit-gradient modes only for --mem auto16 / 16-bit activations.
                   Model bytes per level-0 voxel / step at 96^3 B2 (--fp4 1): MX-fp8 164 / 26.2 ms, + chunked 148 / 27.2, + recompute 2
                   133 / 29.8; 16-bit default 219 / 26.4, chunked 186 / 27.4, + recompute 2 171 / 29.8. lean 1: one batch buffer, logits
                   in A, logit gradient / targets / mask (/ input) in B; lean 2: no gradient buffer B (each block's incoming gradient
                   buffer serves as B), -17 B */
                {1, 1, 1, 0, "MX-fp8 gradients"}, {2, 1, 1, 0, "MX-fp8 gradients, chunked up-part gradient"},
                {2, 1, 1, 1, "MX-fp8 gradients, chunked, lean"}, {2, 1, 1, 2, "MX-fp8 gradients, chunked, lean 2"},
                {2, 2, 1, 0, "MX-fp8 gradients, chunked, recompute 2"}, {2, 2, 1, 1, "MX-fp8 gradients, chunked, recompute 2, lean"},
                {2, 2, 1, 2, "MX-fp8 gradients, chunked, recompute 2, lean 2"},
                {1, 1, 0, 0, "16-bit gradients"}, {2, 1, 0, 0, "16-bit gradients, chunked up-part gradient (UFSM_CHUNK_UP=2)"},
                {2, 2, 0, 0, "16-bit gradients, chunked, recompute 2"}, {2, 2, 0, 1, "16-bit gradients, chunked, recompute 2, lean"},
                {2, 2, 0, 2, "16-bit gradients, chunked, recompute 2, lean 2"}};
            const int nc = (int)(sizeof cand / sizeof cand[0]);
            size_t fmin = (size_t)-1;
            for (int g = 0; g < ng; g++) { nn_init(G[g].dev); size_t f = nn_mem_free(); if (f < fmin) fmin = f; }
            nn_init(G[0].dev);
            const shape5 xs = {B, cfg.cin, split ? g_Dl : P, P, P};   /* split: this GPU's slab of the window */
            const size_t p3 = p3l;
            int pick = -1; size_t need = 0;
            for (int c = 0; c < nc && pick < 0; c++) {
                if (cand[c].gmx == auto16) continue;   /* auto: MX-fp8 gradient modes; auto16: the 16-bit ones */
                unet_set_chunk_up(cand[c].chunk); unet_set_recompute(cand[c].rc); unet_set_grad_mx8(cand[c].gmx); unet_set_lean(cand[c].lean);
                const size_t tb = unet_train_bytes(G[0].u, xs), nbuf = cand[c].lean ? 1 : 2;
                size_t trainer = nbuf * ((size_t)B * 4 * p3 * xbytes() + (size_t)B * NCH * p3 + (size_t)B * p3) + (cand[c].lean ? 0 : (size_t)B * NCH * p3 * (g_g16 ? 2 : 4));
                if (cand[c].lean) trainer = cand[c].gmx ? 0 : (size_t)B * 4 * p3 * xbytes();   /* batch buffers inside the gradient buffer B (estimate) */
                need = tb + tb / 14 + trainer + ((size_t)550 << 20) + (split ? (size_t)4 * B * 32 * P * P * 2 : 0);   /* split: two slots of halo send / receive planes */   /* kernel workspaces: measured device growth / tracked ~1.07 + 0.3 GB, + 0.2 GB margin for other processes */
                if (need <= fmin) pick = c;
            }
            if (pick < 0) { pick = auto16 ? nc - 1 : 6; /* the smallest mode of the list */ fprintf(stderr, "memory: no mode fits %.2f GB free (smallest needs %.2f GB); trying %s\n", fmin / 1e9, need / 1e9, cand[pick].what); }
            else fprintf(stderr, "memory: %s, %.2f of %.2f GB free per GPU\n", cand[pick].what, need / 1e9, fmin / 1e9);
            unet_set_chunk_up(cand[pick].chunk); unet_set_recompute(cand[pick].rc); unet_set_grad_mx8(cand[pick].gmx); unet_set_lean(cand[pick].lean);
            lean = cand[pick].lean;
        }
    }
    for (int g = 0; g < ng; g++) {   /* per-GPU batch buffers. lean: one, shared by both slots, inside the model's gradient buffers when
                                        they are large enough: [targets | mask | input if the forward copies it] in B (lean 1: after the
                                        logit gradient) or in gout[0] (lean 2); the logit gradient in B (lean 1) or after the logits in A */
        gpu_state *d = &G[g];
        nn_init(d->dev);
        char *bs = nullptr;
        const size_t p3 = p3l;   /* this GPU's voxels of a window */
        const size_t a256 = 255, glb = ((size_t)B * NCH * p3 * (g_g16 ? 2 : 4) + a256) & ~a256, tbb = ((size_t)B * NCH * p3 + a256) & ~a256,
                     mbb = ((size_t)B * p3 + a256) & ~a256, xbb = ((size_t)B * 4 * p3 * xbytes() + a256) & ~a256;
        const int xin_b = lean && unet_input_converted() && g_xfmt;
        size_t goff = 0;
        int xin_in = 0;
        d->gl = nullptr;
        if (lean) {
            unet_build(d->u, (shape5){B, cfg.cin, split ? g_Dl : P, P, P}, 1);
            goff = unet_lean_nob(d->u) ? 0 : glb;
            if (xin_b && (bs = unet_grad_scratch(d->u, goff + tbb + mbb + xbb))) xin_in = 1;
            else bs = unet_grad_scratch(d->u, goff + tbb + mbb);
            d->gl = unet_logit_grad_scratch(d->u, glb);
        }
        for (int i = 0; i < 2; i++) {
            if (i && lean) { d->xb[1] = d->xb[0]; d->tb[1] = d->tb[0]; d->mb[1] = d->mb[0]; d->wb[1] = d->wb[0]; }
            else if (bs) {
                d->tb[i] = (uint8_t *)(bs + goff); d->mb[i] = (uint8_t *)(bs + goff + tbb);
                d->xb[i] = xin_in ? (void *)(bs + goff + tbb + mbb) : nn_malloc((size_t)B * 4 * p3 * xbytes());
                d->wb[i] = nn_malloc((size_t)B * NCH);
            }
            else { d->xb[i] = nn_malloc((size_t)B * 4 * p3 * xbytes()); d->tb[i] = nn_malloc((size_t)B * NCH * p3); d->mb[i] = nn_malloc((size_t)B * p3); d->wb[i] = nn_malloc((size_t)B * NCH); }
            d->ev_up[i] = nn_event_create();
        }
        if (lean && g == 0) fprintf(stderr, "lean %d: batch buffers %s, logit gradient %s\n", lean, !bs ? "separate" : xin_in ? "and input inside the gradient buffers" : "inside the gradient buffers, input separate", d->gl ? "inside" : "separate");
        for (int i = 0; i < 2; i++) { d->ev_done[i] = nn_event_create(); nn_event_record(d->ev_done[i], 0); }
        d->pending = nullptr; select_buf(d, 0);
        if (!d->gl) d->gl = nn_malloc((size_t)B * NCH * p3 * (g_g16 ? 2 : 4));
        d->scratch = nn_malloc(nn_loss_scratch((shape5){B, NCH, P, P, P}) + 64);
        d->lean = lean;
        const char *e = nn_check(); if (e) { fprintf(stderr, "GPU %d: %s\n", d->dev, e); return 1; }
    }
    nn_init(G[0].dev);
    size_t np = unet_nparams(G[0].u);
    fprintf(stderr, "model widths"); for (int i = 0; i < cfg.nlev; i++) fprintf(stderr, " %d", cfg.widths[i]);
    fprintf(stderr, ": %zu params; P=%d B=%d x %d GPU(s) [", np, P, B, ng); for (int g = 0; g < ng; g++) fprintf(stderr, "%s%d", g ? "," : "", devs[g]); fprintf(stderr, "] %s%s\n", nn_get_tf32() ? "bf16 tensor cores" : "fp32", split ? ", each window split along z" : "");
    split_ctx *sctx = split ? split_create(devs[0], devs[1]) : nullptr;
    nn_init(G[0].dev);
    float *gpeer = ng > 1 ? nn_malloc(np * 4) : nullptr;   /* on GPU 0: incoming gradients of the other GPUs */

    /* validation set: fixed batches from the held-out boxes (unaugmented), scored on GPU 0 */
    sample_cfg vc = sc; vc.seed = seed + 777; vc.augment = 0; vc.nworkers = 4; vc.nbuf = 2; vc.holdout = 1;
    { int any = 0; for (int i = 0; i < S->n; i++) any |= S->src[i].hold_n[0] > 0; if (!any) { vc.holdout = 0; fprintf(stderr, "no holdout boxes in the sources: validation batches come from the training distribution\n"); } }
    sampler *vs = sampler_start(S, &vc);
    batch *val = calloc((size_t)nval, sizeof *val);
    for (int i = 0; i < nval; i++) {
        batch *b = sampler_next(vs);
        if (!b) { nval = i; break; }
        if (g_xfmt) { val[i].x16 = malloc((size_t)B * 4 * p3 * 2); memcpy(val[i].x16, b->x16, (size_t)B * 4 * p3 * 2); }
        else { val[i].x = malloc((size_t)B * 4 * p3 * 4); memcpy(val[i].x, b->x, (size_t)B * 4 * p3 * 4); }
        val[i].t = malloc((size_t)B * NCH * p3); memcpy(val[i].t, b->t, (size_t)B * NCH * p3);
        val[i].m = malloc((size_t)B * p3); memcpy(val[i].m, b->m, (size_t)B * p3);
        val[i].w = malloc((size_t)B * NCH); memcpy(val[i].w, b->w, (size_t)B * NCH);
        sampler_release(vs, b);
    }
    sampler_stop(vs);
    fprintf(stderr, "%d validation batches\n", nval);

    sampler *sp = sampler_start(S, &sc);
    char logp[1400]; snprintf(logp, sizeof logp, "%s/log.csv", out);
    FILE *log = fopen(logp, step0 ? "a" : "w");
    if (log && !step0) fprintf(log, "step,lr,loss,bce,dice,active,gnorm,val_loss,val_bce,val_dice,samples_per_s,wait_s\n");
    signal(SIGINT, on_sig); signal(SIGTERM, on_sig);
    int prof = ufsm_env_on("UFSM_PROF");
    double t0 = now(), tlog = t0, wait = 0, acc_loss = 0, acc_bce = 0, acc_dice = 0, acc_g = 0; int nacc = 0;
    double best_val = 1e30; int nskip = 0; (void)nskip;
    float parts[8][2 * NCH + 1];
    for (int step = step0 + 1; step <= steps && !g_stop; step++) {
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
            double tw = now(); batch *nb = sampler_next(sp); wait += now() - tw;
            if (!nb) { fprintf(stderr, "sampler stopped\n"); g_stop = 1; G[0].pending = nullptr; break; }
            for (int g = 0; g < 2; g++) { nn_init(G[g].dev); int nxt = G[g].cur ^ 1; upload_async(&G[g], nb, nxt, B, P); select_buf(&G[g], nxt); }
            G[0].pending = nb;
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
                    batch *cp = calloc(1, sizeof *cp); size_t p3b = (size_t)P * P * P;
                    if (g_xfmt) { cp->x16 = malloc((size_t)B * 4 * p3b * 2); memcpy(cp->x16, b->x16, (size_t)B * 4 * p3b * 2); }
                    else { cp->x = malloc((size_t)B * 4 * p3b * 4); memcpy(cp->x, b->x, (size_t)B * 4 * p3b * 4); }
                    cp->t = malloc((size_t)B * NCH * p3b); memcpy(cp->t, b->t, (size_t)B * NCH * p3b);
                    cp->m = malloc((size_t)B * p3b); memcpy(cp->m, b->m, (size_t)B * p3b);
                    cp->w = malloc((size_t)B * NCH); memcpy(cp->w, b->w, (size_t)B * NCH);
                    sampler_release(sp, b); fixed[nfixed++] = cp;
                }
                upload(d, fixed[k], B, P);
                unet_zero_grad(d->u); run_batch(d, B, P, dice_w, 1); unet_backward_x(d->u, d->gl, g_g16);
                continue;
            }
            if (ufsm_env_on("UFSM_SYNC_UPLOAD")) {               /* diagnostic: plain synchronous upload, single buffer */
                double tw = now(); batch *b = sampler_next(sp); wait += now() - tw;
                if (!b) { fprintf(stderr, "sampler stopped\n"); g_stop = 1; break; }
                upload(d, b, B, P); sampler_release(sp, b);
                if (getenv("UFSM_DUMP_BATCH") && step == step0 + 5) {   /* diagnostic: what the GPU sees (mid slice of patch 0: CT | target | mask) */
                    size_t p3 = (size_t)P * P * P; float *hx = malloc(4 * p3 * 4); uint8_t *ht = malloc(NCH * p3), *hm = malloc(p3);
                    if (!g_xfmt) nn_d2h(hx, d->x, 4 * p3 * 4);
                    else { uint16_t *h16 = malloc(4 * p3 * 2); nn_d2h(h16, d->x, 4 * p3 * 2);
                        for (size_t k = 0; k < 4 * p3; k++) { if (g_xfmt == 1) { _Float16 h; memcpy(&h, &h16[k], 2); hx[k] = (float)h; } else { uint32_t u = (uint32_t)h16[k] << 16; memcpy(&hx[k], &u, 4); } }
                        free(h16); }
                    nn_d2h(ht, d->t, NCH * p3); nn_d2h(hm, d->m, p3);
                    FILE *f = fopen(getenv("UFSM_DUMP_BATCH"), "wb"); fprintf(f, "P5\n%d %d\n255\n", 3 * P, P);
                    for (int y = 0; y < P; y++) {
                        for (int x = 0; x < P; x++) { float v = hx[((size_t)(P / 2) * P + y) * P + x]; int g = (int)(128 + 40 * v); fputc(g < 0 ? 0 : g > 255 ? 255 : g, f); }
                        for (int x = 0; x < P; x++) fputc(ht[((size_t)(P / 2) * P + y) * P + x], f);
                        for (int x = 0; x < P; x++) fputc(hm[((size_t)(P / 2) * P + y) * P + x] ? 255 : 0, f);
                    }
                    fclose(f); free(hx); free(ht); free(hm); fprintf(stderr, "dumped batch to %s\n", getenv("UFSM_DUMP_BATCH"));
                }
                unet_zero_grad(d->u); run_batch(d, B, P, dice_w, 1); unet_backward_x(d->u, d->gl, g_g16);
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
            unet_backward_x(d->u, d->gl, g_g16);
            nn_event_record(d->ev_done[d->cur], 0);
            nn_event_sync(d->ev_up[d->cur]);                 /* host buffer of this batch is free again */
            if (overfit) continue;                           /* keep computing on the same device buffer */
            sampler_release(sp, d->pending);
            /* prefetch the next batch into the other buffer while this step computes */
            double tw = now();
            batch *nb = sampler_next(sp);
            wait += now() - tw;
            if (!nb) { fprintf(stderr, "sampler stopped\n"); g_stop = 1; d->pending = nullptr; break; }
            int nxt = d->cur ^ 1;
            upload_async(d, nb, nxt, B, P);
            d->pending = nb;
            select_buf(d, nxt);
        }
        if (g_stop) break;
        /* average gradients across GPUs: sum on GPU 0 (peer copies), scale, broadcast */
        if (ng > 1) {
            nn_init(G[0].dev);
            for (int g = 1; g < ng; g++) { nn_peer_copy(gpeer, G[0].dev, unet_grad_ptr(G[g].u), G[g].dev, np * 4); nn_axpy(unet_grad_ptr(G[0].u), 1.f, gpeer, np); }
            if (!split) nn_scale(unet_grad_ptr(G[0].u), 1.f / ng, np);   /* split: the halves' gradients add up to the window's */
            for (int g = 1; g < ng; g++) nn_peer_copy(unet_grad_ptr(G[g].u), G[g].dev, unet_grad_ptr(G[0].u), G[0].dev, np * 4);
        }
        for (int g = 0; g < nl; g++) { nn_init(G[g].dev); loss += fetch_loss(&G[g], B, P, dice_w, parts[g]); active += parts[g][2 * NCH]; }   /* split: both sides hold the window's loss */
        int fwd_nan = 0;   /* non-finite loss parts: the forward itself produced non-finite logits (not a gradient-scale overflow) */
        for (int g = 0; g < nl; g++) for (int c = 0; c < 2 * NCH; c++) if (!isfinite(parts[g][c])) fwd_nan = 1;
        if (fwd_nan && !overfit && !split) diagnose_nan(&G[0], B, P, step, out);
        float lr;
        if (step <= warmup) lr = lr0 * (float)step / warmup;
        else if (!strcmp(sched, "wsd")) { int cd0 = (int)(steps * (1.f - cooldown)); lr = step < cd0 ? lr0 : lr0 * (float)(steps - step) / (float)(steps - cd0); }
        else lr = lr0 * 0.5f * (1.f + cosf(3.14159265f * (float)(step - warmup) / (float)(steps - warmup)));
        if (soft_end != sc.soft) sampler_set_soft(sp, sc.soft + (soft_end - sc.soft) * (step < steps ? (float)step / (float)steps : 1.f));
        double gn = 0;
        for (int g = 0; g < ng; g++) {
            gpu_state *d = &G[g];
            nn_init(d->dev);
            if (prof) nn_prof_begin(6);
            if (g == 0) gn = unet_grad_norm(d->u);          /* identical on every GPU after averaging */
            if (!isfinite(gn) || fwd_nan) {                   /* skip the step; a 16-bit gradient overflow also halves the gradient scale */
                if (g == 0) {
                    if (fwd_nan) fprintf(stderr, "step %d: non-finite forward, skipping (grad scale kept at %g)\n", step, nn_get_grad_scale());
                    else { float sc = nn_get_grad_scale(); nn_set_grad_scale(sc > 1.f ? sc * 0.5f : 1.f); fprintf(stderr, "step %d: non-finite gradient norm, skipping; grad scale -> %g\n", step, nn_get_grad_scale()); }
                    nskip++;
                }
                continue;
            }
            if (clip > 0 && gn > clip) unet_clip_grad(d->u, clip);
            unet_srste24(d->u, srste);                       /* no-op unless sparse */
            if (use_anvil) unet_anvil(d->u, anvil_lr * (lr / lr0), anvil_wd, step, steps, lr, 0.9f, 0.999f, 1e-8f, wd);
            else if (use_muon) unet_muon(d->u, muon_lr * (lr / lr0), muon_beta, lr, 0.9f, 0.999f, 1e-8f, wd, step);   /* Muon lr follows the same schedule shape */
            else unet_adamw(d->u, lr, 0.9f, 0.999f, 1e-8f, wd, step);
            unet_ema(d->u, step < 100 ? 0.9f : ema);
            unet_wquant(d->u, (unsigned)step);              /* no-op unless --wq */
            if (prof) nn_prof_end();
        }
        const char *e = nn_check();
        if (e) { fprintf(stderr, "cuda error at step %d: %s\n", step, e); return 1; }
        acc_loss += loss / nl; acc_bce += parts[0][0]; acc_dice += parts[0][cfg.cout]; acc_g += gn; nacc++;
        if (step % log_every == 0 || step == steps) {
            double dt = now() - tlog;
            double vl = -1, vb = 0, vd = 0;
            char vstr[128] = ",,,";
            if (step % val_every == 0 || step == steps) {
                const int nv = split ? 2 : 1;   /* split: validation windows are split too */
                for (int g = 0; g < nv; g++) { nn_init(G[g].dev); unet_use_ema(G[g].u, 1); }
                {   /* validation on the buffer not holding the prefetched batch (its upload has completed: ev_up synced above) */
                    int keep = G[0].cur;
                    for (int g = 0; g < nv; g++) { if (G[g].lean) { nn_init(G[g].dev); nn_event_sync(G[g].ev_up[keep]); } select_buf(&G[g], keep ^ 1); }   /* lean: one buffer, the prefetch must have landed */
                    for (int i = 0; i < nval; i++) {
                        float vp[2 * NCH + 1];
                        for (int g = 0; g < nv; g++) { nn_init(G[g].dev); upload(&G[g], &val[i], B, P); }
                        if (split) { split_arg sa = {G, B, P, 0, dice_w}; split_run(sctx, split_job, &sa); nn_init(G[0].dev); }
                        else run_batch(&G[0], B, P, dice_w, 0);
                        double l = fetch_loss(&G[0], B, P, dice_w, vp); vl = (vl < 0 ? 0 : vl) + l; vb += vp[0]; vd += vp[cfg.cout];
                    }
                    for (int g = 0; g < nv; g++) {
                        select_buf(&G[g], keep);
                        batch *pb = split ? G[0].pending : G[g].pending;
                        if (G[g].lean && pb) { nn_init(G[g].dev); upload(&G[g], pb, B, P); }   /* one buffer: the prefetched batch was overwritten */
                    }
                }
                for (int g = 0; g < nv; g++) { nn_init(G[g].dev); unet_use_ema(G[g].u, 0); }
                nn_init(G[0].dev);
                if (nval) { vl /= nval; vb /= nval; vd /= nval; }
                snprintf(vstr, sizeof vstr, "%.5f,%.5f,%.5f", vl, vb, vd);
                if (nval && vl < best_val) { best_val = vl; char bp[1400]; snprintf(bp, sizeof bp, "%s/best.ckpt", out); unet_save(G[0].u, bp, step, nullptr); }
            }
            fprintf(stderr, "step %6d lr %.2e loss %.4f bce %.4f dice %.4f gn %.2f %s %.2f samp/s (wait %.0f%%)%s\n", step, lr, acc_loss / nacc, acc_bce / nacc, acc_dice / nacc, acc_g / nacc,
                    vl >= 0 ? "val" : "", (double)nacc * B * nl / dt, 100 * wait / dt, vl >= 0 ? vstr : "");
            if (log) { fprintf(log, "%d,%.3e,%.5f,%.5f,%.5f,%.0f,%.4f,%s,%.3f,%.3f\n", step, lr, acc_loss / nacc, acc_bce / nacc, acc_dice / nacc, active, acc_g / nacc, vstr, (double)nacc * B * nl / dt, wait); fflush(log); }
            if (prof) { fprintf(stderr, "per-op GPU ms over the last %d steps (all GPUs):\n", log_every); unet_prof_report(); }
            acc_loss = acc_bce = acc_dice = acc_g = 0; nacc = 0; tlog = now(); wait = 0;
        }
        if (step % ckpt_every == 0 || step == steps || g_stop) {
            nn_init(G[0].dev);
            char cp[1400]; snprintf(cp, sizeof cp, "%s/last.ckpt", out);
            unet_save(G[0].u, cp, step, nullptr);
        }
    }
    uint64_t prod, rej; sampler_stats(sp, &prod, &rej);
    fprintf(stderr, "done in %.0fs; sampler produced %llu, rejected %llu\n", now() - t0, (unsigned long long)prod, (unsigned long long)rej);
    sampler_prof_print(sp);   /* UFSM_SAMPLER_PROF=1: per-stage cpu ms per patch */
    sampler_stop(sp);
    if (log) fclose(log);
    return 0;
}
