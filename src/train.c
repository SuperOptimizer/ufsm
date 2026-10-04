/* ufsm train: sampler -> pinned batches -> one or more GPUs (data parallel, gradients averaged through the
   host) -> UNet -> BCE+Dice -> AdamW/EMA, with held-out validation batches, CSV log and checkpoints. */
#include "nn.h"
#include "checkpoint.h"
#include "sample.h"
#include "cover.h"
#include <limits.h>
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
    const sheet_batch *sheet;
    double sheet_loss, sheet_parts[5];
} gpu_state;
static sheet_dataset *g_sheet;
static float g_sheet_ramp;
static int g_sheet_variant=2;
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
    d->sheet=b->sheet?b->sheet[0]:nullptr;
    if (d->side >= 0) { upload_slab(d, b, d->cur, B, P, 0); return; }
    size_t p3 = (size_t)P * P * P;
    nn_h2d(d->x, bx(b), (size_t)B * 4 * p3 * xbytes());
    nn_h2d(d->t, b->t, (size_t)B * NCH * p3);
    nn_h2d(d->m, b->m, (size_t)B * p3);
    nn_h2d(d->w, b->w, (size_t)B * NCH);
}
/* asynchronous upload of a pinned sampler batch into buffer i on the copy stream, after the compute that last used it */
static void upload_async(gpu_state *d, const batch *b, int i, int B, int P) {
    d->sheet=b->sheet?b->sheet[0]:nullptr;
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
    sheet_objective(d,os,P,train);
    if (prof) nn_prof_end();
}
/* fetch the loss parts of the last run_batch on this GPU (synchronous); returns the scalar loss */
static double fetch_loss(gpu_state *d, int B, int P, float dice_w, float *out) {
    shape5 os = unet_out_shape(d->u, (shape5){B, 4, d->side >= 0 ? g_Dl : P, P, P});
    nn_loss_fetch(d->scratch, os, out);
    int cout = unet_cfg_of(d->u)->cout;
    double loss = 0; int active = 0;
    for (int c = 0; c < cout; c++) if (out[c] > 0 || out[cout + c] > 0) { loss += out[c] + dice_w * out[cout + c]; active++; }
    return (active ? loss / active : 0)+d->sheet_loss;
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
        fprintf(stderr, "usage: ufsm train <sources.json> --out DIR [--P 512] [--B 1] [--steps 20000] [--lr 1e-3] [--warmup 500] [--wd 0.01]\n"
                        "       [--dice 0.5] [--ema 0.999] [--widths 16,32,64,80] [--down-norm 1] [--gpus 0,1] [--workers 12] [--seed 0] [--det 1] [--resume CKPT] [--finetune 1] [--fp4 0|1|2 (default 2)] [--mem auto|auto16|default|wide] [--opt adamw|muon|anvil] [--muon-lr 0.02] [--muon-beta 0.95] [--anvil-lr 0.023] [--anvil-wd 2.25] [--sched cos|wsd] [--cooldown 0.2]\n"
                        "       [--val-batches 8] [--log-every 20] [--val-every 500] [--ckpt-every 1000] [--clip 5] [--levels 0.5,0.25,0.15,0.1] [--fp32] [--f16 1] [--gscale 1024] [--prec 1|2|3|4] [--input-prec 0|4|8] [--policy enc0=1,dec0.c1=fp16:fp16:fp8,...] [--qat 2|3] [--wq 8|4] [--sparse24 STEP] [--srste 2e-4] [--pos-weight 1] [--sr 1]\n"
                        "       [--seconds S] [--warmup-seconds S (default 5%% of time budget)] (time-based schedule and final checkpoint)\n"
                        "       [--schedule-seconds S --schedule-elapsed S]   wall-time LR horizon across finite-cover evaluation pauses\n"
                        "       [--cover PLAN.json] [--cover-extend-from OLD_PLAN.json] [--stop-at STEP] [--ct-aug 0|1] [--symmetry-p 1] [--axis-jitter 0] [--limit-seconds 0]\n"
                        "       [--geometry-aug 0|1] [--rotate-deg 5] [--rotate-p 0.2] [--elastic 1] [--elastic-p 0.15] [--label-morph 0] [--label-morph-p 0.2]\n"
                        "       [--gn-stats stored|legacy]   fresh training uses stored activations; resume preserves the saved contract\n"
                        "       [--schedule-start STEP]   restart the LR schedule at this saved step, preserving optimizer state\n"
                        "       [--task surface_winding --geometry geometry.json] [--sheet-init 1] [--sheet-variant 0|1|2]\n"
                        "       [--warm-start 1]   start a new cover from a legacy checkpoint's weights/state\n"
                        "  B is the per-GPU batch; gradients are averaged across GPUs every step (effective batch B x ngpus).\n"
                        "  --split z --gpus 0,1: every window is split along z across the two GPUs instead (batch B; P a multiple of 2^nlev).\n"
                        "  env UFSM_PROF=1 prints per-op GPU time every log interval (category 'upload+loss+opt').\n");
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
    if (sheet_ck<0 || (strcmp(task,"surface") && strcmp(task,"surface_winding")) || (!strcmp(task,"surface_winding") != (geometry!=nullptr)) ||
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
    int qat = atoi(opt(argc, argv, "--qat", "0"));
    /* Save the requested policy and the compute paths observed after dispatch, alongside storage modes. */
#define WRITE_MANIFEST(path) do { \
    char mf_[8192]; size_t mn_ = (size_t)nn_prec_manifest(mf_, sizeof mf_); \
    int add_ = snprintf(mf_ + mn_, sizeof mf_ - mn_, " act_mx4 %d act_mx8 %d grad_mx8 %d input_prec %d gn_stored %d f16 %d opt %s\n", unet_act_mx4(), unet_act_mx() && !unet_act_mx4(), unet_grad_mx8(), unet_input_prec(), nn_get_gn_stored(), nn_get_f16(), optname); \
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
    unet_cfg cfg = {4, {16, 32, 64, 80}, 4, NCH, 8, 0};
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
    if (g_sheet && nl!=1) { fprintf(stderr,"winding currently supports one whole or spatially split window\n"); return 2; }
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
    sc.ct_augment = atoi(opt(argc, argv, "--ct-aug", "0"));
    sc.symmetry_p = (float)atof(opt(argc, argv, "--symmetry-p", "1"));
    sc.axis_jitter = (float)atof(opt(argc, argv, "--axis-jitter", "0"));
    sc.geometry_augment = atoi(opt(argc,argv,"--geometry-aug","0"));
    sc.rotate_degrees=(float)atof(opt(argc,argv,"--rotate-deg","5"));
    sc.rotate_p=(float)atof(opt(argc,argv,"--rotate-p","0.2"));
    sc.elastic=(float)atof(opt(argc,argv,"--elastic","1"));
    sc.elastic_p=(float)atof(opt(argc,argv,"--elastic-p","0.15"));
    sc.label_morph=(float)atof(opt(argc,argv,"--label-morph","0"));
    sc.label_morph_p=(float)atof(opt(argc,argv,"--label-morph-p","0.2"));
    sc.sheet=g_sheet;
    if (g_sheet) { sc.augment=noaug?0:sc.geometry_augment?1:3; memset(sc.level_p,0,sizeof sc.level_p); sc.level_p[0]=1; }
    if (!isfinite(sc.symmetry_p) || sc.symmetry_p < 0 || sc.symmetry_p > 1 || !isfinite(sc.axis_jitter) || sc.axis_jitter < 0) { fprintf(stderr, "invalid augmentation bounds\n"); return 2; }
    if (sc.geometry_augment<0 || sc.geometry_augment>1 || !isfinite(sc.rotate_degrees) || sc.rotate_degrees<0 || sc.rotate_degrees>15 || !isfinite(sc.elastic) || sc.elastic<0 || sc.elastic>2 ||
        !isfinite(sc.rotate_p) || sc.rotate_p<0 || sc.rotate_p>1 || !isfinite(sc.elastic_p) || sc.elastic_p<0 || sc.elastic_p>1 ||
        !isfinite(sc.label_morph) || sc.label_morph<0 || sc.label_morph>1 || !isfinite(sc.label_morph_p) || sc.label_morph_p<0 || sc.label_morph_p>1) {
        fprintf(stderr,"invalid continuous geometry/morphology augmentation bounds\n"); return 2;
    }
    cover_plan *plan = nullptr; cover_progress coverage = {0}, prior_cover = {0};
    const char *cover_path = opt(argc, argv, "--cover", nullptr);
    int warm_start=atoi(opt(argc,argv,"--warm-start","0"));
    if (warm_start && (!resume || sheet_ck)) { fprintf(stderr,"--warm-start requires a legacy donor checkpoint\n"); return 2; }
    int has_cover = resume && !sheet_init && !warm_start ? cover_checkpoint_read(resume, &prior_cover) : 0;
    if (has_cover < 0 || (has_cover && !cover_path)) { fprintf(stderr, "finite-cover resume requires its original plan\n"); return 2; }
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
        coverage = prior_cover; strcpy(coverage.sha256, plan->sha256); coverage.count = plan->count;
        sc.cover = plan; sc.cover_start = coverage.cursor; sc.deterministic = 1; sc.snap = 0;
        memset(sc.level_p, 0, sizeof sc.level_p); sc.level_p[0] = 1;
    }
    sc.dilate = atoi(opt(argc, argv, "--dilate", "0"));   /* thicken surface targets by D level-0 voxels (curriculum) */
    sc.soft = (float)atof(opt(argc, argv, "--soft", "0"));  /* soft ridge target with this sigma (level-0 voxels) */
    if (g_sheet && (!isfinite(sc.soft) || sc.soft<=0 || sc.soft+sc.label_morph>g_sheet->max_soft_sigma)) {
        fprintf(stderr,"winding soft sigma must be positive; sigma plus morphology expansion must be <= audited spacing cap %g\n",g_sheet->max_soft_sigma); return 2;
    }
    float soft_end = (float)atof(opt(argc, argv, "--soft-end", "-1")); if (soft_end < 0) soft_end = sc.soft;   /* sigma annealed linearly to this value at the last step */
    if (plan && soft_end != sc.soft) { fprintf(stderr, "finite cover requires a fixed soft-target sigma for deterministic resume\n"); return 2; }
    if (soft_end != sc.soft) fprintf(stderr, "soft target sigma annealed %g -> %g; the fixed validation batches keep sigma %g, so the validation loss (and best.ckpt) is not comparable across the run: score last.ckpt\n", sc.soft, soft_end, sc.soft);
    { const char *lv = opt(argc, argv, "--levels", nullptr); if (lv) { char *t = strdup(lv); int l = 0; memset(sc.level_p, 0, sizeof sc.level_p); for (char *q = strtok(t, ","); q && l < MAXLEV; q = strtok(nullptr, ",")) sc.level_p[l++] = atof(q); free(t); } }
    if (g_sheet) { memset(sc.level_p,0,sizeof sc.level_p); sc.level_p[0]=1; if (S->n!=1) { fprintf(stderr,"winding v1 requires one registered source\n"); return 2; } }

    size_t p3 = (size_t)P * P * P, p3l = split ? (size_t)P * P * g_Dl : p3;   /* per-GPU voxels of a window */
    gpu_state G[8];
    int step0 = 0;
    for (int g = 0; g < ng; g++) {
        gpu_state *d = &G[g];
        d->dev = devs[g]; d->side = split ? g : -1;
        if (nn_init(d->dev)) { fprintf(stderr, "cannot select GPU %d\n", d->dev); return 1; }
        d->u = unet_create(&cfg);
        if (resume) {
            step0 = unet_load(d->u, resume);
            if (step0 < 0) { fprintf(stderr, "cannot load %s\n", resume); return 1; }
            if (atoi(opt(argc, argv, "--finetune", "0"))) step0 = 0;   /* loaded weights/state, fresh schedule */
            if (sheet_init && unet_start_sheet(d->u)) return 2;
        } else {
            unet_init(d->u, seed + 1); /* deterministic: every GPU starts identical */
            if (g_sheet && unet_start_sheet(d->u)) return 2;
        }
        if (wq) unet_set_wq(d->u, wq);
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
        const char *mm = opt(argc, argv, "--mem", "auto");   /* MX-fp8 gradients passed their stair (3 seeds, mean 0.293 vs 0.294) */
        const int auto16 = g_sheet || !strcmp(mm, "auto16") || !unet_act_mx();   /* sheet reference keeps FP16 stem, hence 16-bit gradients */
        if (!strcmp(mm, "wide")) {
            if (!nn_get_tf32() || !unet_act_mx()) { fprintf(stderr, "--mem wide requires tensor cores and MX activation storage\n"); return 2; }
            unet_set_chunk_up(2); unet_set_recompute(1); unet_set_grad_mx8(1); unet_set_lean(2); unet_set_wide_up_grad(1);
            lean = 2;
            fprintf(stderr, "memory: MX-fp8 gradients, chunked, lean 2, wide finest-level up gradient (explicit mode)\n");
            { char mp[1400]; snprintf(mp, sizeof mp, "%s/precision.txt", out); WRITE_MANIFEST(mp); }
        } else if ((!strcmp(mm, "auto") || auto16) && nn_get_tf32() && !getenv("UFSM_CHUNK_UP") && !getenv("UFSM_RECOMPUTE") && !getenv("UFSM_GRAD_MX8")) {
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
            { char mp[1400]; snprintf(mp, sizeof mp, "%s/precision.txt", out); WRITE_MANIFEST(mp); }   /* the planner's modes */
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
    checkpoint_runtime runtime = {.version = 1, .train_window = P, .prec = nn_get_prec(), .f16 = nn_get_f16(),
        .act_mx4 = unet_act_mx4(), .act_mx8 = unet_act_mx() && !unet_act_mx4(), .grad_mx8 = unet_grad_mx8(), .input_mx = unet_input_converted(), .input_prec = unet_input_prec(), .gn_stored = nn_get_gn_stored()};
    const char *saved_policy = opt(argc, argv, "--policy", "");
    if (!*saved_policy) saved_policy = fp4 >= 2 ? "all=fp4:fp4:fp4,enc0.c1=fp16" : fp4 ? "all=fp4:fp4:fp8,enc0.c1=fp16" : "";
    if (strlen(saved_policy) >= sizeof runtime.policy || strlen(optname) >= sizeof runtime.optimizer) { fprintf(stderr, "checkpoint settings too long\n"); return 2; }
    strcpy(runtime.policy, saved_policy); strcpy(runtime.optimizer, optname);
    char checkpoint_extra[8192], runtime_extra[8192];
    if (checkpoint_runtime_json(&runtime, runtime_extra, sizeof runtime_extra)) { fprintf(stderr, "invalid checkpoint settings\n"); return 2; }
    if (g_sheet) {
        size_t n=strlen(runtime_extra); runtime_extra[n-1]=0;
        snprintf(runtime_extra+n-1,sizeof runtime_extra-n+1,",\"task\":\"surface_winding\",\"sheet\":{\"version\":1,\"units\":\"turns\",\"outputs\":[\"surface_logit\",\"winding_residual\"],\"geometry_sha256\":\"%s\",\"reference_sha256\":\"%s\",\"variant\":%d,\"schedule_start\":%d}}",g_sheet->manifest_sha,g_sheet->reference_sha,g_sheet_variant,schedule_start);
    }
    {
        size_t n=strlen(runtime_extra); runtime_extra[n-1]=0;
        snprintf(runtime_extra+n-1,sizeof runtime_extra-n+1,",\"augmentation\":{\"version\":1,\"mode\":%d,\"ct\":%d,\"symmetry_p\":%.9g,\"axis_jitter\":%.9g,\"geometry\":%d,\"rotate_degrees\":%.9g,\"rotate_p\":%.9g,\"elastic_voxels\":%.9g,\"elastic_p\":%.9g,\"label_morph_voxels\":%.9g,\"label_morph_p\":%.9g,\"soft_sigma\":%.9g}}",
            sc.augment,sc.ct_augment,sc.symmetry_p,sc.axis_jitter,sc.geometry_augment,sc.rotate_degrees,sc.rotate_p,sc.elastic,sc.elastic_p,sc.label_morph,sc.label_morph_p,sc.soft);
    }
    fprintf(stderr,"augmentation: mode %d, CT %d, symmetry p %.3g, axis jitter %.3g voxels, geometry %d (rotation +/-%.3g degrees p %.3g; elastic <=%.3g voxels p %.3g), soft-band dilation/erosion +/-%.3g voxels p %.3g\n",
        sc.augment,sc.ct_augment,sc.symmetry_p,sc.axis_jitter,sc.geometry_augment,sc.rotate_degrees,sc.rotate_p,sc.elastic,sc.elastic_p,sc.label_morph,sc.label_morph_p);
    if (schedule_seconds) {
        size_t n=strlen(runtime_extra); runtime_extra[n-1]=0;
        snprintf(runtime_extra+n-1,sizeof runtime_extra-n+1,",\"time_schedule\":{\"version\":1,\"seconds\":%.9g,\"elapsed_at_launch\":%.9g,\"warmup_seconds\":%.9g,\"cooldown\":%.9g}}",
            schedule_seconds,schedule_elapsed,warmup_seconds,cooldown);
    }
    strcpy(checkpoint_extra, runtime_extra);
    if (plan && cover_checkpoint_extra(runtime_extra, &coverage, checkpoint_extra, sizeof checkpoint_extra)) return 2;
    fprintf(stderr, "model widths"); for (int i = 0; i < cfg.nlev; i++) fprintf(stderr, " %d", cfg.widths[i]);
    fprintf(stderr, ": %zu params; P=%d B=%d x %d GPU(s) [", np, P, B, ng); for (int g = 0; g < ng; g++) fprintf(stderr, "%s%d", g ? "," : "", devs[g]); fprintf(stderr, "] %s%s\n", nn_get_tf32() ? "tensor cores (see precision.txt for compute and storage)" : "fp32", split ? ", each window split along z" : "");
    split_ctx *sctx = split ? split_create(devs[0], devs[1]) : nullptr;
    nn_init(G[0].dev);
    float *gpeer = ng > 1 ? nn_malloc(np * 4) : nullptr;   /* on GPU 0: incoming gradients of the other GPUs */

    /* validation set: fixed batches from the held-out boxes (unaugmented), scored on GPU 0 */
    sample_cfg vc = sc; vc.cover = nullptr; vc.cover_start = 0; vc.ct_augment = 0; vc.axis_jitter = 0; vc.geometry_augment=0; vc.label_morph=0; vc.seed = seed + 777; vc.augment = 0; vc.nworkers = 4; vc.nbuf = 2; vc.holdout = 1;
    { int any = 0; for (int i = 0; i < S->n; i++) any |= S->src[i].hold_n[0] > 0; if (!any) { vc.holdout = 0; fprintf(stderr, "no holdout boxes in the sources: validation batches come from the training distribution\n"); } }
    sampler *vs = sampler_start(S, &vc);
    if (!vs) { fprintf(stderr, "cannot construct validation sampler\n"); return 1; }
    batch *val = calloc((size_t)nval, sizeof *val);
    for (int i = 0; i < nval; i++) {
        batch *b = sampler_next(vs);
        if (!b) { nval = i; break; }
        if (g_xfmt) { val[i].x16 = malloc((size_t)B * 4 * p3 * 2); memcpy(val[i].x16, b->x16, (size_t)B * 4 * p3 * 2); }
        else { val[i].x = malloc((size_t)B * 4 * p3 * 4); memcpy(val[i].x, b->x, (size_t)B * 4 * p3 * 4); }
        val[i].t = malloc((size_t)B * NCH * p3); memcpy(val[i].t, b->t, (size_t)B * NCH * p3);
        val[i].m = malloc((size_t)B * p3); memcpy(val[i].m, b->m, (size_t)B * p3);
        val[i].w = malloc((size_t)B * NCH); memcpy(val[i].w, b->w, (size_t)B * NCH);
        val[i].sheet=calloc((size_t)B,sizeof *val[i].sheet); val[i].sheet[0]=sheet_clone(b->sheet?b->sheet[0]:nullptr);
        sampler_release(vs, b);
    }
    int validation_failed = sampler_failed(vs);
    sampler_stop(vs);
    if (validation_failed) { fprintf(stderr, "validation sampler failed\n"); return 1; }
    fprintf(stderr, "%d validation batches\n", nval);

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
    float parts[8][2 * NCH + 1];
    for (int step = step0 + 1; step <= steps && !g_stop; step++) {
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
                lr = elapsed < cd0 ? lr0 : lr0 * (float)((time_horizon - elapsed) / (time_horizon - cd0));
            } else lr = lr0 * 0.5f * (1.f + cosf(3.14159265f * (float)((elapsed - warmup_seconds) / (time_horizon - warmup_seconds))));
        }
        else {
            int ss = step - schedule_start, total = steps - schedule_start;
            if (ss <= warmup) lr = lr0 * (float)ss / warmup;
            else if (!strcmp(sched, "wsd")) { int cd0 = (int)(total * (1.f - cooldown)); lr = ss < cd0 ? lr0 : lr0 * (float)(total - ss) / (float)(total - cd0); }
            else lr = lr0 * 0.5f * (1.f + cosf(3.14159265f * (float)(ss - warmup) / (float)(total - warmup)));
        }
        if (soft_end != sc.soft) sampler_set_soft(sp, sc.soft + (soft_end - sc.soft) * (step < steps ? (float)step / (float)steps : 1.f));
        double gn = 0;
        for (int g = 0; g < ng; g++) {
            gpu_state *d = &G[g];
            nn_init(d->dev);
            if (prof) nn_prof_begin(6);
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
        acc_loss += loss / nl; acc_bce += parts[0][0]; acc_dice += parts[0][cfg.cout]; acc_g += gn; nacc++;
        if (step % log_every == 0 || step == steps || g_stop) {
            double dt = now() - tlog;
            double vl = -1, vb = 0, vd = 0;
            char vstr[128] = ",,";
            if (step % val_every == 0 || step == steps || g_stop) {
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
                        G[g].sheet=pb && pb->sheet?pb->sheet[0]:nullptr;
                    }
                }
                for (int g = 0; g < nv; g++) { nn_init(G[g].dev); unet_use_ema(G[g].u, 0); }
                nn_init(G[0].dev);
                if (nval) { vl /= nval; vb /= nval; vd /= nval; }
                snprintf(vstr, sizeof vstr, "%.5f,%.5f,%.5f", vl, vb, vd);
                if (nval && vl < best_val) { best_val = vl; char bp[1400]; snprintf(bp, sizeof bp, "%s/best.ckpt", out); if (unet_save(G[0].u, bp, step, checkpoint_extra)) { fprintf(stderr, "cannot save %s\n", bp); return 1; } }
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
