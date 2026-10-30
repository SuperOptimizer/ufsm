/* wtrunk: the frozen wind96 U-Net in-slab for the ray model (src/wtrunk.h), set up as predict sets up a task-unwrap checkpoint
   (embedded runtime: precision, policy, fp16, MX-fp4 activations, input MX precision, stored GN statistics; EMA weights; fp16
   logits), plus ufsm wtrunk-bench */
#include "wtrunk.h"
#include "unet.h"
#include "nn.h"
#include "checkpoint.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct wtrunk {
    unet *u; unet_cfg cfg;
    int T, L, maxb, feat, C, h16;
    void *x; float *prm; unsigned long long *st;
    /* training (wt_open_train): the trained network ut (live weights), the inference build u above for no-grad passes (its weights copied from
       ut lazily after each step), the head + identity rows, the forward's fp16 outputs (sigmoid derivative), the backward's inputs */
    unet *ut; float *wext, *bext; void *y16; float *glog; void *gf; int stale, levels, step; char *extra;
};
void wt_grad_in(const void *g, int ld, void *y16, int C, int C0, int T, int L, float gs, int f16);
static void prm_upload(wtrunk *w, const wt_frame *f);

wtrunk *wt_open(const char *ckpt, int T, int L, int maxbatch, int feat) {
    checkpoint_runtime rt; unet_cfg cfg; int step = 0;
    const int emb = checkpoint_runtime_read(ckpt, &rt);
    if (emb != 1 || unet_peek(ckpt, &cfg, &step)) { fprintf(stderr, "wt_open: %s: not a checkpoint with embedded inference settings\n", ckpt); return nullptr; }
    if (cfg.half_res || cfg.cin != 4 || cfg.cout < 11 || cfg.view_P || checkpoint_unwrap_version(ckpt) < 1) {
        fprintf(stderr, "wt_open: %s is not a full-resolution 4-input task-unwrap checkpoint\n", ckpt); return nullptr;
    }
    const int div = 1 << (cfg.nlev - 1);
    if (T % div || L % div || maxbatch < 1) { fprintf(stderr, "wt_open: slab %dx%dx%d must be divisible by %d\n", T, T, L, div); return nullptr; }
    nn_set_nlev(cfg.nlev);
    if (getenv("UFSM_WT_HI") && atoi(getenv("UFSM_WT_HI"))) {   /* tests: bf16 compute, MX-fp8 activations, 16-bit gradients (gradient checks: MX-fp8
                                                                    gradients alone move a central difference to 0.53-0.6 of |g| on random linear losses) */
        rt.prec = 1; rt.policy[0] = 0; rt.f16 = 1; rt.act_mx4 = 0; rt.act_mx8 = 1;   /* (bf16 compute on MX-fp8 storage: the finest mode the trainer's
                                                                                     MX head paths cover) */
    }
    nn_set_gn_stored(rt.gn_stored);
    nn_set_prec(rt.prec);
    if (nn_set_prec_policy(rt.policy)) return nullptr;
    nn_set_f16(rt.f16);
    unet_set_act_mx4(rt.act_mx4); unet_set_act_mx8(rt.act_mx8);
    unet_set_grad_mx8(0);
    if (unet_set_input_prec(rt.input_prec)) return nullptr;
    unet_set_input_mx(rt.input_mx || rt.input_prec);
    unet *u = unet_create(&cfg);
    if (unet_load(u, ckpt) < 0) { fprintf(stderr, "wt_open: cannot load %s\n", ckpt); unet_free(u); return nullptr; }
    unet_use_ema(u, 1);
    unet_set_fp16_logits(u, 1);
    unet_set_head_feat(u, feat);
    wtrunk *w = calloc(1, sizeof *w);
    w->u = u; w->cfg = cfg; w->T = T; w->L = L; w->maxb = maxbatch; w->feat = !!feat;
    w->C = cfg.cout + (feat ? cfg.widths[0] : 0);
    w->h16 = nn_get_f16() ? 1 : 2;
    const size_t V = (size_t)T * T * L;
    w->x = nn_malloc((size_t)maxbatch * 4 * V * 2);
    w->prm = nn_malloc((size_t)maxbatch * 16 * sizeof(float));
    w->st = nn_malloc((size_t)maxbatch * 2 * sizeof(unsigned long long));
    if (!w->x || !w->prm || !w->st) { fprintf(stderr, "wt_open: out of GPU memory\n"); wt_close(w); return nullptr; }
    return w;
}
int wt_channels(const wtrunk *w) { return w->C; }
int wt_heads(const wtrunk *w) { return w->cfg.cout; }

int wt_forward(wtrunk *w, const uint8_t *im, const wt_frame *fr, int n, double um, void *out) { return wt_forward_ld(w, im, fr, n, um, out, w->C); }
int wt_forward_ld(wtrunk *w, const uint8_t *im, const wt_frame *fr, int n, double um, void *out, int ld) {
    if (ld < w->C) { fprintf(stderr, "wt_forward_ld: row stride %d < %d channels\n", ld, w->C); return 1; }
    if (w->ut) {   /* training handle: no-grad passes on the training build in evaluation mode (no rebuild, no second network) */
        const int T = w->T, L = w->L; const size_t V = (size_t)T * T * L; const int co = w->cfg.cout, w0 = w->cfg.widths[0];
        if (w->cfg.scale_cond) unet_set_scale(w->ut, um);
        size_t hw, hb; unet_head_offsets(w->ut, &hw, &hb);
        nn_d2d(w->wext, unet_live_params(w->ut) + hw, (size_t)co * w0 * 4); nn_d2d(w->bext, unet_live_params(w->ut) + hb, (size_t)co * 4);
        for (int s = 0; s < n; s++) {
            prm_upload(w, fr + s);
            wt_input(im + (size_t)s * V, w->prm, 1, T, L, w->st, w->x, w->h16);
            const shape5 xs = {1, 4, T, T, L};
            if (!unet_forward_x(w->ut, w->x, xs, 0, 1)) return 1;
            if (unet_head_ext_fwd(w->ut, w->wext, w->bext, w->C, w->y16)) return 1;
            wt_output(w->y16, 1, 1, w->C, co, T, L, ld, (char *)out + (size_t)s * V * ld * 2);
        }
        const char *e = nn_check(); if (e) { fprintf(stderr, "wt_forward_ld: %s\n", e); return 1; }
        return 0;
    }
    const int T = w->T, L = w->L; const size_t V = (size_t)T * T * L;
    if (w->cfg.scale_cond) unet_set_scale(w->u, um);
    for (int s0 = 0; s0 < n; s0 += w->maxb) {
        const int nb = n - s0 < w->maxb ? n - s0 : w->maxb;
        float h[16 * 64];
        if (nb > 64) { fprintf(stderr, "wt_forward: at most 64 slabs per forward\n"); return 1; }
        for (int s = 0; s < nb; s++) {
            const wt_frame *f = fr + s0 + s; float *p = h + 16 * s;
            double un = sqrt(f->u[0] * f->u[0] + f->u[1] * f->u[1] + f->u[2] * f->u[2]); if (!(un > 0)) un = 1;
            for (int e = 0; e < 3; e++) { p[e] = (float)(f->o[e] - f->c[e]); p[3 + e] = (float)f->a[e]; p[6 + e] = (float)f->b[e]; p[9 + e] = (float)f->d[e]; p[12 + e] = (float)(f->u[e] / un); }
            p[15] = 0.f;
        }
        nn_h2d(w->prm, h, (size_t)nb * 16 * sizeof(float));
        wt_input(im + (size_t)s0 * V, w->prm, nb, T, L, w->st, w->x, w->h16);
        const shape5 xs = {nb, 4, T, T, L};
        const float *lg = unet_forward_x(w->u, w->x, xs, 0, 1);
        if (!lg) return 1;
        wt_output(lg, unet_logits_h16(w->u), nb, w->C, w->cfg.cout, T, L, ld, (char *)out + (size_t)s0 * V * ld * 2);
    }
    const char *e = nn_check(); if (e) { fprintf(stderr, "wt_forward: %s\n", e); return 1; }
    return 0;
}
void wt_close(wtrunk *w) {
    if (!w) return;
    if (w->ut) { unet_free(w->ut); nn_free(w->wext); nn_free(w->y16); nn_free(w->glog); free(w->extra); }
    unet_free(w->u); nn_free(w->x); nn_free(w->prm); nn_free(w->st); free(w);
}

/* ---- training (stage 2) ---- */
static void prm_upload(wtrunk *w, const wt_frame *f) {
    float h[16]; double un = sqrt(f->u[0] * f->u[0] + f->u[1] * f->u[1] + f->u[2] * f->u[2]); if (!(un > 0)) un = 1;
    for (int e = 0; e < 3; e++) { h[e] = (float)(f->o[e] - f->c[e]); h[3 + e] = (float)f->a[e]; h[6 + e] = (float)f->b[e]; h[9 + e] = (float)f->d[e]; h[12 + e] = (float)(f->u[e] / un); }
    h[15] = 0.f; nn_h2d(w->prm, h, sizeof h);
}
wtrunk *wt_open_train(const char *ckpt, int T, int L, int levels) {
    wtrunk *w = wt_open(ckpt, T, L, 1, 1);
    if (!w) return nullptr;
    checkpoint_runtime rt; checkpoint_runtime_read(ckpt, &rt);
    /* the production training precision of the checkpoint (train --fp4 2): fp16 storage with gradient scaling, MX-fp4 activations and
       gradients, fp4 weight gradients, stochastic rounding, the shared encoder a1 */
    const int hi = getenv("UFSM_WT_HI") && atoi(getenv("UFSM_WT_HI"));
    nn_set_grad_scale(1024.f);
    if (!hi) {
        nn_set_sr(1);
        /* MX-fp8 activation gradients (not MX-fp4: with them the head backward hands dec[0] a gradient past its GroupNorm + SiLU, so the
           injected feature gradient lands in the wrong place; test_wtrunk's injection identity, cosine 0.14 -> 0.996); env WT_G4=1 */
        if (rt.act_mx4) { setenv("UFSM_F4_WGRAD", "1", 0); unet_set_grad_mx4(getenv("WT_G4") ? atoi(getenv("WT_G4")) : 0); }
        unet_set_grad_mx8(1);
        unet_set_share_enc_a1(getenv("WT_SEA") ? atoi(getenv("WT_SEA")) : 1);
        /* train --mem auto's smallest recompute-1 mode (lean 2, chunked up-part gradient): beside the ray model on a 16 GB card; env WT_LEAN */
        unet_set_chunk_up(2); unet_set_recompute(1); unet_set_lean(getenv("WT_LEAN") ? atoi(getenv("WT_LEAN")) : 2);
    } else { nn_set_sr(getenv("WT_SR") ? atoi(getenv("WT_SR")) : 0); unet_set_grad_mx4(0); unet_set_grad_mx8(getenv("WT_GMX8") ? atoi(getenv("WT_GMX8")) : 0); unet_set_share_enc_a1(0); }
    char *hl = checkpoint_header_line(ckpt);
    const int resumed = hl && strstr(hl, "\"wtrunk\":");
    {   /* the checkpoint's extra object, marked as a wtrunk save (so wt_open reads RUN/latest.wt as the original) */
        const char *e = hl ? strstr(hl, "\"extra\":{") : nullptr; size_t n = 0;
        if (e) { e += 8; n = strlen(e); while (n && (e[n - 1] == '\n' || e[n - 1] == '\r')) n--; if (n) n--; }   /* (the outer object's '}') */
        w->extra = malloc(n + 32);
        if (e && n > 2) snprintf(w->extra, n + 32, "{\"wtrunk\":1,%.*s", (int)(n - 1), e + 1); else strcpy(w->extra, "{\"wtrunk\":1}");
    }
    unet_cfg cfg = w->cfg; int step = 0; unet_peek(ckpt, &cfg, &step);
    w->ut = unet_create(&w->cfg);
    if (unet_load(w->ut, ckpt) < 0) { fprintf(stderr, "wt_open_train: cannot load %s\n", ckpt); free(hl); wt_close(w); return nullptr; }
    if (!resumed) {   /* fine-tuning start: the EMA weights (the frozen trunk's) as the live ones, a fresh optimiser */
        unet_use_ema(w->ut, 1); nn_d2d((float *)unet_live_params(w->ut), unet_fwd_params(w->ut), unet_nparams(w->ut) * 4); unet_reset_opt(w->ut);
        step = 0;
    }
    unet_use_ema(w->ut, 0);
    unet_set_fp16_logits(w->ut, 1);
    unet_free(w->u); w->u = nullptr;   /* (no-grad passes run on the training build) */
    free(hl);
    w->step = step; w->levels = levels; w->stale = 1;
    const size_t V = (size_t)T * T * L; const int co = w->cfg.cout, w0 = w->cfg.widths[0], C = co + w0;
    if (w->C != C) { fprintf(stderr, "wt_open_train: needs the feature channels\n"); wt_close(w); return nullptr; }
    float *h = calloc((size_t)C * (w0 + 1), 4); for (int c = 0; c < w0; c++) h[(size_t)(co + c) * w0 + c] = 1.f;
    w->wext = nn_malloc((size_t)C * (w0 + 1) * 4); w->bext = w->wext + (size_t)C * w0; nn_h2d(w->wext, h, (size_t)C * (w0 + 1) * 4); free(h);
    /* gf (the features' 16-bit gradient) in y16's feature rows: only the head rows' logits are read after the output (the sigmoid derivative) */
    w->y16 = nn_malloc(V * C * 2); w->glog = nullptr; w->gf = (char *)w->y16 + V * co * 2;   /* (the logit gradient in place of the head rows) */
    if (!w->y16) { fprintf(stderr, "wt_open_train: out of GPU memory\n"); wt_close(w); return nullptr; }
    { const shape5 xs = {1, 4, T, T, L}; fprintf(stderr, "wtrunk: training build %.2f GB (activations + gradients), + %.2f GB outputs / logit gradient\n", unet_train_bytes(w->ut, xs) / 1e9, (double)V * C * 2 / 1e9); }
    fprintf(stderr, "wtrunk: training %s from %s (step %d)%s\n", resumed ? "resumed" : "starts", ckpt, step, levels ? " (head + last decoder blocks only)" : "");
    return w;
}
int wt_step_of(const wtrunk *w) { return w->step; }
struct unet *wt_unet(wtrunk *w) { return w->ut; }
const void *wt_y16(wtrunk *w) { return w->y16; }
int wt_forward_train(wtrunk *w, const uint8_t *im, const wt_frame *fr, double um, void *out, int ld) {
    const int T = w->T, L = w->L, co = w->cfg.cout, w0 = w->cfg.widths[0];
    if (!w->ut || ld < w->C) return 1;
    if (w->cfg.scale_cond) unet_set_scale(w->ut, um);
    prm_upload(w, fr);
    wt_input(im, w->prm, 1, T, L, w->st, w->x, w->h16);
    nn_set_sr_step((unsigned)w->step);
    const shape5 xs = {1, 4, T, T, L};
    if (!unet_forward_x(w->ut, w->x, xs, 1, 1)) return 1;
    size_t hw, hb; unet_head_offsets(w->ut, &hw, &hb);
    nn_d2d(w->wext, unet_live_params(w->ut) + hw, (size_t)co * w0 * 4); nn_d2d(w->bext, unet_live_params(w->ut) + hb, (size_t)co * 4);
    if (unet_head_ext_fwd(w->ut, w->wext, w->bext, w->C, w->y16)) { fprintf(stderr, "wt_forward_train: the head extension is not covered\n"); return 1; }
    wt_output(w->y16, 1, 1, w->C, co, T, L, ld, out);
    const char *e = nn_check(); if (e) { fprintf(stderr, "wt_forward_train: %s\n", e); return 1; }
    return 0;
}
int wt_backward(wtrunk *w, const void *g, int ld) {
    if (!w->ut) return 1;
    wt_grad_in(g, ld, w->y16, w->C, w->cfg.cout, w->T, w->L, unet_bwd_gscale(), nn_get_f16());
    unet_set_l0_feat_grad(w->ut, w->gf);
    unet_backward_x(w->ut, w->y16, 1);   /* (the 16-bit logit gradient, scaled: requires the 16-bit gradient storage) */
    unet_set_l0_feat_grad(w->ut, nullptr);
    const char *e = nn_check(); if (e) { fprintf(stderr, "wt_backward: %s\n", e); return 1; }
    return 0;
}
void wt_zero_grad(wtrunk *w) { if (w->ut) unet_zero_grad(w->ut); }
int wt_step(wtrunk *w, float lr_muon, float lr_adam, float wd, float clip, float ema) {
    if (!w->ut) return -1;
    if (w->levels > 0) unet_grad_head_only(w->ut, w->levels);
    const double gn = unet_grad_norm(w->ut);
    if (!isfinite(gn)) {   /* (fp16 gradient overflow: a smaller scale, the step skipped) */
        const float sc = nn_get_grad_scale(); nn_set_grad_scale(sc > 1.f ? sc * 0.5f : 1.f);
        fprintf(stderr, "wtrunk: non-finite gradient, step skipped, grad scale -> %g\n", nn_get_grad_scale());
        unet_zero_grad(w->ut); return 1;
    }
    if (clip > 0) unet_clip_grad(w->ut, clip);
    w->step++;
    unet_muon(w->ut, lr_muon, 0.95f, lr_adam, 0.9f, 0.999f, 1e-8f, wd, w->step);
    unet_ema(w->ut, ema);
    w->stale = 1;
    return 0;
}
double wt_last_gnorm(wtrunk *w) { return w->ut ? unet_grad_norm(w->ut) : 0; }
int wt_save(wtrunk *w, const char *path) { return w->ut ? unet_save(w->ut, path, w->step, w->extra) : -1; }

/* ufsm wtrunk-bench CKPT [--n 1] [--batch N] [--feat 1] [--iters 10] [--T 128] [--L 384] [--um 9.6]: synthetic sheets (concentric
   shells around an axis at the slab's side), forward time per slab, GPU memory, output statistics per channel */
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static const char *opt_(int argc, char **argv, const char *k, const char *d) { for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
int cmd_wtrunk_bench(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: ufsm wtrunk-bench CKPT [--n 1] [--batch N] [--feat 1] [--iters 10] [--T 128] [--L 384] [--um 9.6] [--gpu 0]\n"); return 2; }
    const int n = atoi(opt_(argc, argv, "--n", "1")), nb = atoi(opt_(argc, argv, "--batch", opt_(argc, argv, "--n", "1"))), feat = atoi(opt_(argc, argv, "--feat", "1"));
    const int it = atoi(opt_(argc, argv, "--iters", "10")), T = atoi(opt_(argc, argv, "--T", "128")), L = atoi(opt_(argc, argv, "--L", "384"));
    const double um = atof(opt_(argc, argv, "--um", "9.6"));
    if (nn_init(atoi(opt_(argc, argv, "--gpu", "0")))) return 1;
    const double f0 = nn_mem_free() / 1e9;
    wtrunk *w = wt_open(argv[2], T, L, nb, feat);
    if (!w) return 1;
    const int C = wt_channels(w); const size_t V = (size_t)T * T * L;
    uint8_t *h = malloc((size_t)n * V); wt_frame *fr = calloc(n, sizeof *fr);
    for (int s = 0; s < n; s++) {   /* slab s: ray along +x from radius 200 + 100 s, a = y, b = z; shells every 14 voxels */
        wt_frame *f = fr + s; const double r0 = 200 + 100 * s;
        f->d[0] = 1; f->a[1] = 1; f->b[2] = 1; f->u[2] = 1;
        f->o[0] = r0; f->o[1] = -(T - 1) / 2.0; f->o[2] = -(T - 1) / 2.0;
        for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) for (int k = 0; k < L; k++) {
            const double y = f->o[1] + i, x = f->o[0] + k, r = sqrt(x * x + y * y), ph = r / 14.0 - floor(r / 14.0);
            const double v = 60 + 120 * exp(-(ph - 0.5) * (ph - 0.5) / 0.01) + 10 * sin(0.37 * i + 0.11 * j + 0.05 * k);
            h[((size_t)s * T + i) * T * L + (size_t)j * L + k] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
        }
    }
    uint8_t *im = nn_malloc((size_t)n * V); nn_h2d(im, h, (size_t)n * V);
    void *out = nn_malloc((size_t)n * V * C * 2);
    if (!out) { fprintf(stderr, "wtrunk-bench: no memory for the output\n"); return 1; }
    if (wt_forward(w, im, fr, n, um, out)) return 1;
    nn_sync();
    const double f1 = nn_mem_free() / 1e9;
    double best = 1e30, sum = 0;
    for (int k = 0; k < it; k++) { nn_sync(); const double t0 = now_s(); if (wt_forward(w, im, fr, n, um, out)) return 1; nn_sync(); const double t = now_s() - t0; sum += t; if (t < best) best = t; }
    printf("wtrunk %dx%dx%d n %d batch %d feat %d C %d: %.1f ms/slab (min), %.1f mean; %.2f Mvox/s; GPU used %.2f GB (output %.2f GB)\n",
           T, T, L, n, nb, feat, C, 1e3 * best / n, 1e3 * sum / it / n, n * V / best / 1e6, f0 - f1, (double)n * V * C * 2 / 1e9);
    uint16_t *ho = malloc((size_t)V * C * 2); nn_d2h(ho, out, (size_t)V * C * 2);
    for (int c = 0; c < C; c++) {   /* slab 0 per-channel mean / sd / NaN count */
        double a = 0, q = 0; long nan = 0;
        for (size_t v = 0; v < V; v++) { uint32_t b = (uint32_t)ho[v * C + c] << 16; float x; memcpy(&x, &b, 4); if (x != x) { nan++; continue; } a += x; q += (double)x * x; }
        a /= V; q = q / V - a * a;
        printf("  ch %2d mean %8.4f sd %8.4f%s\n", c, a, sqrt(q > 0 ? q : 0), nan ? " NaN!" : "");
    }
    free(ho); free(h); free(fr); nn_free(im); nn_free(out); wt_close(w);
    return 0;
}
