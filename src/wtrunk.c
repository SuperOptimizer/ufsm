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
};

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

int wt_forward(wtrunk *w, const uint8_t *im, const wt_frame *fr, int n, double um, void *out) {
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
        wt_output(lg, unet_logits_h16(w->u), nb, w->C, w->cfg.cout, T, L, (char *)out + (size_t)s0 * V * w->C * 2);
    }
    const char *e = nn_check(); if (e) { fprintf(stderr, "wt_forward: %s\n", e); return 1; }
    return 0;
}
void wt_close(wtrunk *w) {
    if (!w) return;
    unet_free(w->u); nn_free(w->x); nn_free(w->prm); nn_free(w->st); free(w);
}

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
