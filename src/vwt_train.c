/* ufsm vwt-train: training loop of the ray-attention winding model (src/vwt.h).

   ufsm vwt-train SOURCES RUNDIR [--gpu 0] [--steps 50000] [--batch 2] [--lr 5e-4] [--wd 0.01] [--warmup 500] [--ema 0.999] [--clip 1]
                  [--tilt 15] [--threads 12] [--log 50] [--ckpt-every 2000] [--seed 1] [--tf32 1] [--core 1] [--mx 16]
                  [--dp 0.25] [--dd 0.05] [--lam-p 1] [--lam-d 1] [--T 128] [--L 384]
                  [--shell 0|1 --lam-shell 0.2 --snap-gate 0.25]   (position-only shell loss on samples with shell masks)
   ufsm vwt-train --bench N [--T 128 --L 384]   forward + backward of N random slabs (no sampler): ms / slab, Mvox/s, device memory

   AdamW (all parameters, as villa's torch AdamW), cosine schedule with linear warm-up, global-norm clipping, EMA of the weights.
   RUNDIR: latest.vwt (+ step_NNNNNN.vwt every --ckpt-every; the EMA weights and the optimiser state ride along), log.tsv; resumes
   from latest.vwt when present; a file RUNDIR/STOP saves and exits. */
#include "vwt.h"
#include "attn.h"
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>

static volatile sig_atomic_t g_term = 0;
static void on_term(int sig) { (void)sig; g_term = 1; }

/* the LAST occurrence wins (launch scripts put their defaults first) */
static const char *opt(int argc, char **argv, const char *k, const char *d) { for (int i = argc - 2; i >= 1; i--) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static vwt_cfg default_cfg(int T, int L) { return (vwt_cfg){T, L, {32, 64, 96}, 192, 2, 6, 4, 256, 96, 0.04f}; }

static int bench(int argc, char **argv) {
    const int N = atoi(opt(argc, argv, "--bench", "3")), T = atoi(opt(argc, argv, "--T", "128")), L = atoi(opt(argc, argv, "--L", "384"));
    vwt_cfg c = default_cfg(T, L); vwt *m = vwt_create(&c, 1); if (!m) return 1;
    const size_t n = (size_t)T * T * L; uint8_t *im = malloc(n), *va = malloc(n);
    for (size_t i = 0; i < n; i++) { im[i] = (uint8_t)(120 + 60 * sin((i % L) * 0.5) + (rand() & 15)); va[i] = 1; }
    uint8_t *imd = nn_malloc(n), *vad = nn_malloc(n); nn_h2d(imd, im, n); nn_h2d(vad, va, n);
    float *ph = nn_malloc(n * 4), *inc = nn_malloc(n * 4), *g1 = nn_malloc(n * 4), *g2 = nn_malloc(n * 4); nn_zero(g1, n * 4); nn_zero(g2, n * 4);
    double tf = 0, tb = 0;
    for (int r = 0; r <= N; r++) {
        nn_sync(); const double t0 = now_s();
        if (vwt_forward(m, imd, vad, ph, inc, 1)) return 1;
        nn_sync(); const double t1 = now_s();
        if (vwt_backward(m, g1, g2)) return 1;
        nn_sync(); const double t2 = now_s();
        if (r) { tf += t1 - t0; tb += t2 - t1; }
    }
    size_t fr = nn_mem_free();
    printf("vwt bench: slab %dx%dx%d, %zu params: forward %.0f ms, backward %.0f ms, step %.0f ms = %.1f Mvox/s; device free %.1f GB\n",
           T, T, L, vwt_nparams(m), tf / N * 1e3, tb / N * 1e3, (tf + tb) / N * 1e3, n / ((tf + tb) / N) / 1e6, fr / 1e9);
    return 0;
}

int cmd_vwt_train(int argc, char **argv) {
    if (nn_init(atoi(opt(argc, argv, "--gpu", "0")))) return 1;
    if (atoi(opt(argc, argv, "--tf32", "1"))) setenv("UFSM_ATTN_TF32", "1", 0);
    at_set_core(atoi(opt(argc, argv, "--core", "1")));
    at_set_mx(atoi(opt(argc, argv, "--mx", "16")));   /* attention-block linears / activations: 16 = bf16 (src/nn_attn_h16.cu), 0 fp32, 8 / 4 MX */
    if (opt(argc, argv, "--bench", nullptr)) return bench(argc, argv);
    if (argc < 4) { fprintf(stderr, "usage: ufsm vwt-train SOURCES RUNDIR [options]  |  ufsm vwt-train --bench N\n"); return 2; }
    const char *src = argv[2], *run = argv[3];
    const int T = atoi(opt(argc, argv, "--T", "128")), L = atoi(opt(argc, argv, "--L", "384")), B = atoi(opt(argc, argv, "--batch", "2"));
    const int steps = atoi(opt(argc, argv, "--steps", "50000")), warm = atoi(opt(argc, argv, "--warmup", "500")), logf = atoi(opt(argc, argv, "--log", "50"));
    const int ckev = atoi(opt(argc, argv, "--ckpt-every", "2000"));
    const float lr0 = (float)atof(opt(argc, argv, "--lr", "5e-4")), wd = (float)atof(opt(argc, argv, "--wd", "0.01")), emad = (float)atof(opt(argc, argv, "--ema", "0.999"));
    const float clip = (float)atof(opt(argc, argv, "--clip", "1")), dp = (float)atof(opt(argc, argv, "--dp", "0.25")), dd = (float)atof(opt(argc, argv, "--dd", "0.05"));
    const float lp = (float)atof(opt(argc, argv, "--lam-p", "1")), ld = (float)atof(opt(argc, argv, "--lam-d", "1"));
    const int use_shell = atoi(opt(argc, argv, "--shell", "0"));   /* position-only shell loss (samples carrying shell masks) */
    const float lsh = (float)atof(opt(argc, argv, "--lam-shell", "0.2")), gate = (float)atof(opt(argc, argv, "--snap-gate", "0.25"));
    mkdir(run, 0755);
    char path[1200]; snprintf(path, sizeof path, "%s/latest.vwt", run);
    vwt_cfg c = default_cfg(T, L);
    vwt *m = access(path, F_OK) == 0 ? vwt_load(path, 0) : vwt_create(&c, (unsigned)atoi(opt(argc, argv, "--seed", "1")));
    if (!m) return 1;
    const size_t np = vwt_nparams(m);
    float *ema = nn_malloc(np * 4), *mo = nn_malloc(np * 4), *vo = nn_malloc(np * 4), *red = nn_malloc(1 << 20);
    nn_d2d(ema, vwt_params(m), np * 4); nn_zero(mo, np * 4); nn_zero(vo, np * 4);
    int step = 0, nbad = 0;
    if (access(path, F_OK) == 0) {   /* resume: the EMA and the optimiser state (extra = step, m, v) */
        FILE *f = fopen(path, "rb"); char mg[4]; vwt_cfg cc; uint64_t n2; int has; uint64_t eb;
        float *h = malloc(np * 4); int ok = f && fread(mg, 1, 4, f) == 4 && fread(&cc, sizeof cc, 1, f) == 1 && fread(&n2, 8, 1, f) == 1 && n2 == np && fread(h, 4, np, f) == np;
        if (ok && fread(&has, 4, 1, f) == 1 && has && fread(h, 4, np, f) == np) nn_h2d(ema, h, np * 4);
        if (ok && fread(&eb, 8, 1, f) == 1 && eb == 4 + 8 * np && fread(&step, 4, 1, f) == 1 && fread(h, 4, np, f) == np) { nn_h2d(mo, h, np * 4); if (fread(h, 4, np, f) == np) nn_h2d(vo, h, np * 4); }
        if (f) fclose(f); free(h);
        fprintf(stderr, "vwt-train: resumed %s at step %d\n", path, step);
    }
    signal(SIGTERM, on_term); signal(SIGINT, on_term);
    vws_set_args(argc, argv);
    vws *S = vws_open(src, T, L, (float)atof(opt(argc, argv, "--tilt", "15")), atoi(opt(argc, argv, "--threads", "12")), (unsigned)(atoi(opt(argc, argv, "--seed", "1")) * 7919 + step), 1);
    if (!S) { fprintf(stderr, "vwt-train: cannot open sources %s\n", src); return 1; }
    const size_t n = (size_t)T * T * L;
    uint8_t *imd = nn_malloc(n), *vad = nn_malloc(n); float *tg = nn_malloc(n * 4), *pw = nn_malloc(n * 4), *gph = nn_malloc(n * 4), *ginc = nn_malloc(n * 4);
    float *phd = nn_malloc(n * 4), *incd = nn_malloc(n * 4); uint16_t *cd = nn_malloc(n * 2); float *ones = malloc(n * 4); uint8_t *shd = nn_malloc(n);
    double acc_h = 0, acc_hc = 0;
    for (size_t i = 0; i < n; i++) ones[i] = 1.f;
    snprintf(path, sizeof path, "%s/log.tsv", run); FILE *lf = fopen(path, step ? "a" : "w"); if (!lf) return 1;
    if (!step) fprintf(lf, "step\tlr\tphase\tdensity\tgnorm\tsec_step\tsample_wait\tlabelled\tmono\tshell\tshell_cols\n");
    double acc_p = 0, acc_d = 0, acc_w = 0, acc_mono = 0, t_step = 0, t_wait = 0, acc_g = 0; int acc_n = 0;
    while (step < steps) {
        const double t0 = now_s(); double wait = 0;
        vwt_zero_grad(m);
        for (int b = 0; b < B; b++) {
            vws_sample s; const double tw = now_s(); if (vws_next(S, &s)) { fprintf(stderr, "vwt-train: sampler failed\n"); return 1; } wait += now_s() - tw;
            nn_h2d(imd, s.im, n); nn_h2d(vad, s.va, n); nn_h2d(tg, s.ph, n * 4); nn_h2d(cd, s.comp, n * 2); nn_h2d(pw, s.pw ? s.pw : ones, n * 4);
            /* monotone convention check on the host: share of labelled ray steps where the target rises */
            { size_t up = 0, tot = 0; for (size_t i = 1; i < n; i += 7) if (i % L && s.comp[i] && s.comp[i] == s.comp[i - 1]) { tot++; up += s.ph[i] > s.ph[i - 1]; } acc_mono += tot ? (double)up / tot : 0; }
            if (vwt_forward(m, imd, vad, phd, incd, 1)) return 1;
            vt_loss lo; if (vt_losses(phd, incd, tg, cd, pw, T, L, dp, dd, lp, ld, 1.f / B, gph, ginc, &lo)) return 1;
            acc_p += lo.phase; acc_d += lo.density; acc_w += lo.wsum;
            if (use_shell && s.shell) {
                double hh, ss, nc; nn_h2d(shd, s.shell, n);
                if (vt_shell_loss(phd, shd, cd, T, L, gate, lsh, lsh, 1.f / B, gph, &hh, &ss, &nc)) return 1;
                acc_h += hh + ss; acc_hc += nc;
            }
            if (vwt_backward(m, gph, ginc)) return 1;
        }
        /* clip, schedule, AdamW, EMA */
        double gn = sqrt(nn_sumsq(vwt_grads(m), np, red));
        if (!isfinite(gn)) { fprintf(stderr, "vwt-train: non-finite gradient at step %d, skipped\n", step); if (++nbad > 20) return 1; continue; }
        nbad = 0;
        if (clip > 0 && gn > clip) nn_scale(vwt_grads(m), (float)(clip / gn), np);
        step++;
        const double lr = step < warm ? lr0 * step / warm : lr0 * 0.5 * (1 + cos(M_PI * (double)(step - warm) / fmax(1, steps - warm)));
        nn_adamw(vwt_params(m), vwt_grads(m), mo, vo, np, (float)lr, 0.9f, 0.999f, 1e-8f, wd, step);
        nn_ema(ema, vwt_params(m), np, emad);
        nn_sync(); t_step += now_s() - t0; t_wait += wait; acc_g += gn; acc_n++;
        if (step % logf == 0 || step == steps) {
            const double k = acc_n * B;
            fprintf(lf, "%d\t%.3g\t%.5f\t%.6f\t%.3f\t%.3f\t%.3f\t%.0f\t%.3f\t%.5f\t%.0f\n", step, lr, acc_p / k, acc_d / k, acc_g / acc_n, t_step / acc_n, t_wait / acc_n, acc_w / k, acc_mono / k, acc_h / k, acc_hc / k);
            fflush(lf);
            fprintf(stderr, "vwt-train: step %d lr %.2e phase %.4f density %.5f |g| %.2f  %.2f s/step (wait %.2f)  %.1f Mvox/s  mono %.2f\n", step, lr, acc_p / k, acc_d / k, acc_g / acc_n,
                    t_step / acc_n, t_wait / acc_n, (double)n * B / (t_step / acc_n) / 1e6, acc_mono / k);
            acc_p = acc_d = acc_w = acc_mono = t_step = t_wait = acc_g = acc_h = acc_hc = 0; acc_n = 0;
        }
        snprintf(path, sizeof path, "%s/STOP", run); const int stop = access(path, F_OK) == 0 || g_term;
        if (step % ckev == 0 || step == steps || stop) {
            const size_t eb = 4 + 8 * np; uint8_t *ex = malloc(eb); memcpy(ex, &step, 4); nn_d2h(ex + 4, mo, np * 4); nn_d2h(ex + 4 + 4 * np, vo, np * 4);
            snprintf(path, sizeof path, "%s/latest.vwt", run);
            if (vwt_save(m, path, ema, ex, eb)) fprintf(stderr, "vwt-train: cannot save %s\n", path);
            if (step % ckev == 0) { char p2[1200]; snprintf(p2, sizeof p2, "%s/step_%06d.vwt", run, step); if (vwt_save(m, p2, ema, nullptr, 0)) fprintf(stderr, "vwt-train: cannot save %s\n", p2); }
            free(ex);
        }
        if (stop) { fprintf(stderr, "vwt-train: STOP at step %d\n", step); break; }
    }
    fclose(lf); vws_close(S);
    return 0;
}

/* until src/vwt_sample.c lands (WM): weak stubs so the binary links; the real sampler overrides them */
__attribute__((weak)) vws *vws_open(const char *s, int T, int L, float t, int n, unsigned sd, int tr) { (void)s; (void)T; (void)L; (void)t; (void)n; (void)sd; (void)tr; fprintf(stderr, "vws_open: no sampler built in\n"); return nullptr; }
__attribute__((weak)) int vws_next(vws *s, vws_sample *o) { (void)s; (void)o; return -1; }
__attribute__((weak)) void vws_close(vws *s) { (void)s; }
__attribute__((weak)) void vws_set_args(int argc, char **argv) { (void)argc; (void)argv; }
