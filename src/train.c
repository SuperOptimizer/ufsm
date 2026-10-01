/* ufsm train: sampler -> pinned batches -> one or more GPUs (data parallel, gradients averaged through the
   host) -> UNet -> BCE+Dice -> AdamW/EMA, with held-out validation batches, CSV log and checkpoints. */
#include "nn.h"
#include "sample.h"
#include "sources.h"
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
    uint8_t *tb[2], *mb[2], *wb[2];
    float *x; uint8_t *t, *m, *w; /* the buffers of the batch being computed */
    void *ev_up[2], *ev_done;    /* upload of buffer i finished; compute of the previous step finished */
    batch *pending;              /* host batch whose upload into buffer (cur ^ 1) is in flight */
    int cur;
} gpu_state;

static void select_buf(gpu_state *d, int i) { d->cur = i; d->x = d->xb[i]; d->t = d->tb[i]; d->m = d->mb[i]; d->w = d->wb[i]; }
/* synchronous upload (pageable host memory: validation batches) */
static void upload(gpu_state *d, const batch *b, int B, int P) {
    size_t p3 = (size_t)P * P * P;
    nn_h2d(d->x, b->x, (size_t)B * 4 * p3 * 4);
    nn_h2d(d->t, b->t, (size_t)B * NCH * p3);
    nn_h2d(d->m, b->m, (size_t)B * p3);
    nn_h2d(d->w, b->w, (size_t)B * NCH);
}
/* asynchronous upload of a pinned sampler batch into buffer i on the copy stream, after the compute that last used it */
static void upload_async(gpu_state *d, const batch *b, int i, int B, int P) {
    size_t p3 = (size_t)P * P * P;
    nn_stream_wait(1, d->ev_done);
    nn_h2d_copy_stream(d->xb[i], b->x, (size_t)B * 4 * p3 * 4);
    nn_h2d_copy_stream(d->tb[i], b->t, (size_t)B * NCH * p3);
    nn_h2d_copy_stream(d->mb[i], b->m, (size_t)B * p3);
    nn_h2d_copy_stream(d->wb[i], b->w, (size_t)B * NCH);
    nn_event_record(d->ev_up[i], 1);
}

/* forward + loss kernels on the uploaded batch of this GPU (asynchronous); fills gl when train. */
static void run_batch(gpu_state *d, int B, int P, float dice_w, int train) {
    shape5 xs = {B, 4, P, P, P};
    const float *lg = unet_forward(d->u, d->x, xs, train);
    shape5 os = unet_out_shape(d->u, xs);
    int prof = getenv("UFSM_PROF") != nullptr;
    if (prof) nn_prof_begin(6);
    nn_loss_async(lg, d->t, d->m, d->w, os, dice_w, train ? d->gl : nullptr, d->scratch);
    if (prof) nn_prof_end();
}
/* fetch the loss parts of the last run_batch on this GPU (synchronous); returns the scalar loss */
static double fetch_loss(gpu_state *d, int B, int P, float dice_w, float *out) {
    shape5 os = unet_out_shape(d->u, (shape5){B, 4, P, P, P});
    nn_loss_fetch(d->scratch, os, out);
    int cout = unet_cfg_of(d->u)->cout;
    double loss = 0; int active = 0;
    for (int c = 0; c < cout; c++) if (out[c] > 0 || out[cout + c] > 0) { loss += out[c] + dice_w * out[cout + c]; active++; }
    return active ? loss / active : 0;
}

int cmd_train(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: ufsm train <sources.json> --out DIR [--P 96] [--B 2] [--steps 20000] [--lr 1e-3] [--warmup 500] [--wd 0.01]\n"
                        "       [--dice 0.5] [--ema 0.999] [--widths 16,32,64,80] [--gpus 0,1] [--workers 12] [--seed 0] [--resume CKPT]\n"
                        "       [--val-batches 8] [--log-every 20] [--val-every 500] [--ckpt-every 1000] [--clip 5] [--levels 0.5,0.25,0.15,0.1] [--fp32] [--f16 1] [--gscale 1024] [--prec 1|2|3] [--policy enc0=1,...] [--qat 2|3] [--wq 8|4] [--sparse24 STEP] [--srste 2e-4] [--pos-weight 1]\n"
                        "  B is the per-GPU batch; gradients are averaged across GPUs every step (effective batch B x ngpus).\n"
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
    int qat = atoi(opt(argc, argv, "--qat", "0"));         /* quantization-aware training: forward/backward-data at precision 2 (fp8) or 3 (fp4), weight gradients at 16-bit */
    if (qat) { nn_set_prec(qat); nn_set_prec_wgrad(1); }
    int wq = atoi(opt(argc, argv, "--wq", "0"));            /* 8 or 4: true fp8 / fp4 weights (stochastic rounding after each update) */
    int sparse_at = atoi(opt(argc, argv, "--sparse24", "0"));   /* step from which the 3^3 conv weights are 2:4 sparse (0 = dense) */
    float srste = (float)atof(opt(argc, argv, "--srste", "2e-4"));
    nn_set_pos_weight((float)atof(opt(argc, argv, "--pos-weight", "1")));   /* BCE weight of surface voxels */
    int overfit = atoi(opt(argc, argv, "--overfit", "0"));   /* diagnostic: train on the first batch forever */
    int noaug = atoi(opt(argc, argv, "--noaug", "0"));       /* diagnostic: no augmentation */
    if (nn_set_prec_policy(opt(argc, argv, "--policy", ""))) return 2;   /* per-layer: "enc0=1,enc1=2,..." */
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--fp32")) nn_set_tf32(0);
    int devs[8], ng = 0;
    { char *t = strdup(opt(argc, argv, "--gpus", opt(argc, argv, "--gpu", "0"))); for (char *q = strtok(t, ","); q && ng < 8; q = strtok(nullptr, ",")) devs[ng++] = atoi(q); free(t); }
    unet_cfg cfg = {4, {16, 32, 64, 80}, 4, NCH, 8};
    { char *t = strdup(opt(argc, argv, "--widths", "16,32,64,80")); cfg.nlev = 0; for (char *q = strtok(t, ","); q && cfg.nlev < UNET_MAXLEV; q = strtok(nullptr, ",")) cfg.widths[cfg.nlev++] = atoi(q); free(t); }
    if (P % (1 << (cfg.nlev - 1))) { fprintf(stderr, "P must be divisible by %d\n", 1 << (cfg.nlev - 1)); return 2; }
    char cmd[1400]; snprintf(cmd, sizeof cmd, "mkdir -p '%s'", out); if (system(cmd)) return 1;

    sources *S = sources_load(src);
    if (!S) return 1;
    sample_cfg sc = sample_cfg_default();
    sc.P = P; sc.B = B; sc.nworkers = workers; sc.nbuf = 4 * ng + 2; sc.seed = seed;
    if (noaug) sc.augment = 0;
    if (atoi(opt(argc, argv, "--rotonly", "0"))) sc.augment = 2;   /* proper rotations only (no reflections) */
    sc.dilate = atoi(opt(argc, argv, "--dilate", "0"));   /* thicken surface targets by D level-0 voxels (curriculum) */
    sc.soft = (float)atof(opt(argc, argv, "--soft", "0"));  /* soft ridge target with this sigma (level-0 voxels) */
    { const char *lv = opt(argc, argv, "--levels", nullptr); if (lv) { char *t = strdup(lv); int l = 0; memset(sc.level_p, 0, sizeof sc.level_p); for (char *q = strtok(t, ","); q && l < MAXLEV; q = strtok(nullptr, ",")) sc.level_p[l++] = atof(q); free(t); } }

    size_t p3 = (size_t)P * P * P;
    gpu_state G[8];
    int step0 = 0;
    for (int g = 0; g < ng; g++) {
        gpu_state *d = &G[g];
        d->dev = devs[g];
        if (nn_init(d->dev)) { fprintf(stderr, "cannot select GPU %d\n", d->dev); return 1; }
        d->u = unet_create(&cfg);
        if (resume) { step0 = unet_load(d->u, resume); if (step0 < 0) { fprintf(stderr, "cannot load %s\n", resume); return 1; } }
        else unet_init(d->u, seed + 1);                   /* deterministic: every GPU starts identical */
        if (wq) unet_set_wq(d->u, wq);
        for (int i = 0; i < 2; i++) { d->xb[i] = nn_malloc((size_t)B * 4 * p3 * 4); d->tb[i] = nn_malloc((size_t)B * NCH * p3); d->mb[i] = nn_malloc((size_t)B * p3); d->wb[i] = nn_malloc((size_t)B * NCH); d->ev_up[i] = nn_event_create(); }
        d->ev_done = nn_event_create(); nn_event_record(d->ev_done, 0); d->pending = nullptr; select_buf(d, 0);
        d->gl = nn_malloc((size_t)B * NCH * p3 * 4); d->scratch = nn_malloc(nn_loss_scratch((shape5){B, NCH, P, P, P}) + 64);
        const char *e = nn_check(); if (e) { fprintf(stderr, "GPU %d: %s\n", d->dev, e); return 1; }
    }
    if (resume) fprintf(stderr, "resumed %s at step %d\n", resume, step0);
    size_t np = unet_nparams(G[0].u);
    fprintf(stderr, "model widths"); for (int i = 0; i < cfg.nlev; i++) fprintf(stderr, " %d", cfg.widths[i]);
    fprintf(stderr, ": %zu params; P=%d B=%d x %d GPU(s) [", np, P, B, ng); for (int g = 0; g < ng; g++) fprintf(stderr, "%s%d", g ? "," : "", devs[g]); fprintf(stderr, "] %s\n", nn_get_tf32() ? "bf16 tensor cores" : "fp32");
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
        val[i].x = malloc((size_t)B * 4 * p3 * 4); memcpy(val[i].x, b->x, (size_t)B * 4 * p3 * 4);
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
    int prof = getenv("UFSM_PROF") != nullptr;
    double t0 = now(), tlog = t0, wait = 0, acc_loss = 0, acc_bce = 0, acc_dice = 0, acc_g = 0; int nacc = 0;
    double best_val = 1e30; int nskip = 0; (void)nskip;
    float parts[8][2 * NCH + 1];
    for (int step = step0 + 1; step <= steps && !g_stop; step++) {
        double loss = 0, active = 0;
        /* forward + backward on every GPU; kernel launches are asynchronous so the GPUs overlap */
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
                    cp->x = malloc((size_t)B * 4 * p3b * 4); memcpy(cp->x, b->x, (size_t)B * 4 * p3b * 4);
                    cp->t = malloc((size_t)B * NCH * p3b); memcpy(cp->t, b->t, (size_t)B * NCH * p3b);
                    cp->m = malloc((size_t)B * p3b); memcpy(cp->m, b->m, (size_t)B * p3b);
                    cp->w = malloc((size_t)B * NCH); memcpy(cp->w, b->w, (size_t)B * NCH);
                    sampler_release(sp, b); fixed[nfixed++] = cp;
                }
                upload(d, fixed[k], B, P);
                unet_zero_grad(d->u); run_batch(d, B, P, dice_w, 1); unet_backward(d->u, d->gl);
                continue;
            }
            if (getenv("UFSM_SYNC_UPLOAD")) {               /* diagnostic: plain synchronous upload, single buffer */
                double tw = now(); batch *b = sampler_next(sp); wait += now() - tw;
                if (!b) { fprintf(stderr, "sampler stopped\n"); g_stop = 1; break; }
                upload(d, b, B, P); sampler_release(sp, b);
                if (getenv("UFSM_DUMP_BATCH") && step == step0 + 5) {   /* diagnostic: what the GPU sees (mid slice of patch 0: CT | target | mask) */
                    size_t p3 = (size_t)P * P * P; float *hx = malloc(4 * p3 * 4); uint8_t *ht = malloc(NCH * p3), *hm = malloc(p3);
                    nn_d2h(hx, d->x, 4 * p3 * 4); nn_d2h(ht, d->t, NCH * p3); nn_d2h(hm, d->m, p3);
                    FILE *f = fopen(getenv("UFSM_DUMP_BATCH"), "wb"); fprintf(f, "P5\n%d %d\n255\n", 3 * P, P);
                    for (int y = 0; y < P; y++) {
                        for (int x = 0; x < P; x++) { float v = hx[((size_t)(P / 2) * P + y) * P + x]; int g = (int)(128 + 40 * v); fputc(g < 0 ? 0 : g > 255 ? 255 : g, f); }
                        for (int x = 0; x < P; x++) fputc(ht[((size_t)(P / 2) * P + y) * P + x], f);
                        for (int x = 0; x < P; x++) fputc(hm[((size_t)(P / 2) * P + y) * P + x] ? 255 : 0, f);
                    }
                    fclose(f); free(hx); free(ht); free(hm); fprintf(stderr, "dumped batch to %s\n", getenv("UFSM_DUMP_BATCH"));
                }
                unet_zero_grad(d->u); run_batch(d, B, P, dice_w, 1); unet_backward(d->u, d->gl);
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
            unet_backward(d->u, d->gl);
            nn_event_record(d->ev_done, 0);
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
            nn_scale(unet_grad_ptr(G[0].u), 1.f / ng, np);
            for (int g = 1; g < ng; g++) nn_peer_copy(unet_grad_ptr(G[g].u), G[g].dev, unet_grad_ptr(G[0].u), G[0].dev, np * 4);
        }
        for (int g = 0; g < ng; g++) { nn_init(G[g].dev); loss += fetch_loss(&G[g], B, P, dice_w, parts[g]); active += parts[g][2 * NCH]; }
        float lr = step <= warmup ? lr0 * (float)step / warmup : lr0 * 0.5f * (1.f + cosf(3.14159265f * (float)(step - warmup) / (float)(steps - warmup)));
        double gn = 0;
        for (int g = 0; g < ng; g++) {
            gpu_state *d = &G[g];
            nn_init(d->dev);
            if (prof) nn_prof_begin(6);
            if (g == 0) gn = unet_grad_norm(d->u);          /* identical on every GPU after averaging */
            if (!isfinite(gn)) {                              /* 16-bit gradient overflow: skip the step, halve the gradient scale */
                if (g == 0) { float sc = nn_get_grad_scale(); nn_set_grad_scale(sc > 1.f ? sc * 0.5f : 1.f); fprintf(stderr, "step %d: non-finite gradient norm, skipping; grad scale -> %g\n", step, nn_get_grad_scale()); nskip++; }
                continue;
            }
            if (clip > 0 && gn > clip) unet_clip_grad(d->u, clip);
            unet_srste24(d->u, srste);                       /* no-op unless sparse */
            unet_adamw(d->u, lr, 0.9f, 0.999f, 1e-8f, wd, step);
            unet_ema(d->u, step < 100 ? 0.9f : ema);
            unet_wquant(d->u, (unsigned)step);              /* no-op unless --wq */
            if (prof) nn_prof_end();
        }
        const char *e = nn_check();
        if (e) { fprintf(stderr, "cuda error at step %d: %s\n", step, e); return 1; }
        acc_loss += loss / ng; acc_bce += parts[0][0]; acc_dice += parts[0][cfg.cout]; acc_g += gn; nacc++;
        if (step % log_every == 0 || step == steps) {
            double dt = now() - tlog;
            double vl = -1, vb = 0, vd = 0;
            char vstr[128] = ",,,";
            if (step % val_every == 0 || step == steps) {
                nn_init(G[0].dev);
                unet_use_ema(G[0].u, 1);
                {   /* validation on the buffer not holding the prefetched batch (its upload has completed: ev_up synced above) */
                    int keep = G[0].cur; select_buf(&G[0], keep ^ 1);
                    for (int i = 0; i < nval; i++) { upload(&G[0], &val[i], B, P); float vp[2 * NCH + 1]; run_batch(&G[0], B, P, dice_w, 0); double l = fetch_loss(&G[0], B, P, dice_w, vp); vl = (vl < 0 ? 0 : vl) + l; vb += vp[0]; vd += vp[cfg.cout]; }
                    select_buf(&G[0], keep);
                }
                unet_use_ema(G[0].u, 0);
                if (nval) { vl /= nval; vb /= nval; vd /= nval; }
                snprintf(vstr, sizeof vstr, "%.5f,%.5f,%.5f", vl, vb, vd);
                if (nval && vl < best_val) { best_val = vl; char bp[1400]; snprintf(bp, sizeof bp, "%s/best.ckpt", out); unet_save(G[0].u, bp, step, nullptr); }
            }
            fprintf(stderr, "step %6d lr %.2e loss %.4f bce %.4f dice %.4f gn %.2f %s %.2f samp/s (wait %.0f%%)%s\n", step, lr, acc_loss / nacc, acc_bce / nacc, acc_dice / nacc, acc_g / nacc,
                    vl >= 0 ? "val" : "", (double)nacc * B * ng / dt, 100 * wait / dt, vl >= 0 ? vstr : "");
            if (log) { fprintf(log, "%d,%.3e,%.5f,%.5f,%.5f,%.0f,%.4f,%s,%.3f,%.3f\n", step, lr, acc_loss / nacc, acc_bce / nacc, acc_dice / nacc, active, acc_g / nacc, vstr, (double)nacc * B * ng / dt, wait); fflush(log); }
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
    sampler_stop(sp);
    if (log) fclose(log);
    return 0;
}
