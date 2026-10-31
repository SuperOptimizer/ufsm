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
#include "sysinfo.h"
#include "attn.h"
#include "nn.h"
#if __has_include("wtrunk.h")
#include "wtrunk.h"
#define HAVE_WTRUNK 1
#else   /* the interface of src/wtrunk.h (PF) until it lands */
typedef struct wtrunk wtrunk;
typedef struct { double o[3], a[3], b[3], d[3], c[3], u[3]; } wt_frame;
wtrunk *wt_open(const char *ckpt, int T, int L, int maxbatch, int feat);
int wt_forward(wtrunk *w, const uint8_t *im, const wt_frame *fr, int n, double um, uint16_t *out);
int wt_forward_ld(wtrunk *w, const uint8_t *im, const wt_frame *fr, int n, double um, void *out, int ld);
int wt_channels(const wtrunk *w);
void wt_close(wtrunk *w);
#endif
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

static vwt_cfg default_cfg(int T, int L) { return (vwt_cfg){T, L, {32, 64, 96}, 192, 2, 6, 4, 256, 96, 0.04f, 0}; }
int vwt_ckpt_header(FILE *f, vwt_cfg *cfg);   /* (vwt_net.c) */

/* --bench N [--trunk CKPT] [--mv 1]: one training sample's step as vwt-train runs it, synchronised section times (ms): with --mv the
   pair slab B (trunk + eval forward), A's trunk, forward, loss, the multiview (d2h + host + h2d) and the backward; synthetic slabs */
static int bench(int argc, char **argv) {
    const int N = atoi(opt(argc, argv, "--bench", "3")), T = atoi(opt(argc, argv, "--T", "128")), L = atoi(opt(argc, argv, "--L", "384"));
    const char *trunk = opt(argc, argv, "--trunk", nullptr); const int mv = atoi(opt(argc, argv, "--mv", "0")), mvh = atoi(opt(argc, argv, "--mv-host", "0"));
    wtrunk *WT = nullptr; int cx = 0;
    const int ttr = trunk && atoi(opt(argc, argv, "--trunk-train", "0"));
    if (trunk) { WT = ttr ? wt_open_train(trunk, T, L, 0) : wt_open(trunk, T, L, 1, atoi(opt(argc, argv, "--trunk-feat", "1"))); if (!WT) return 1; cx = wt_channels(WT); }
    vwt_cfg c = default_cfg(T, L); c.cx = cx; vwt *m = vwt_create(&c, 1); if (!m) return 1;
    const size_t n = (size_t)T * T * L; uint8_t *im = malloc(n), *va = malloc(n); float *tgh = malloc(n * 4); uint16_t *ch = malloc(n * 2);
    for (size_t i = 0; i < n; i++) { im[i] = (uint8_t)(120 + 60 * sin((i % L) * 0.5) + (rand() & 15)); va[i] = 1; tgh[i] = 0.07f * (float)(i % L); ch[i] = 1; }
    uint8_t *imd = nn_malloc(n), *vad = nn_malloc(n); nn_h2d(imd, im, n); nn_h2d(vad, va, n);
    float *tg = nn_malloc(n * 4), *pw = nn_malloc(n * 4); uint16_t *cd = nn_malloc(n * 2); nn_h2d(tg, tgh, n * 4); nn_h2d(cd, ch, n * 2);
    { float *o = malloc(n * 4); for (size_t i = 0; i < n; i++) o[i] = 1.f; nn_h2d(pw, o, n * 4); free(o); }
    float *ph = nn_malloc(n * 4), *inc = nn_malloc(n * 4), *g1 = nn_malloc(n * 4), *g2 = nn_malloc(n * 4), *phb = nn_malloc(n * 4);
    float *pha_h = malloc(n * 4), *phb_h = malloc(n * 4), *gmv = malloc(n * 4), *gmvd = nn_malloc(n * 4);
    int xld = cx; uint16_t *xfd = cx ? vwt_xfeat_buf(m, &xld) : nullptr;
    uint16_t *xgd = ttr ? xfd : nullptr; if (xgd) vwt_set_xgrad(m, xgd, xld);
    /* frames: A along +x at radius 600; B its copy rotated 20 degrees about the slab centre's vertical axis (overlapping) */
    const double r0 = 600, th = 20 * M_PI / 180;
    wt_frame fa = {{r0, -(T - 1) / 2.0, 500 - (T - 1) / 2.0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}, {0, 0, 500}, {0, 0, 1}}, fb = fa;
    { const double cxp = r0 + L / 2.0, cs = cos(th), sn = sin(th); double d[3] = {cs, sn, 0}, a[3] = {-sn, cs, 0};
      for (int e = 0; e < 3; e++) { fb.d[e] = d[e]; fb.a[e] = a[e]; }
      fb.o[0] = cxp - L / 2.0 * cs + (T - 1) / 2.0 * sn; fb.o[1] = -L / 2.0 * sn - (T - 1) / 2.0 * cs; }
    const double FA[4][3] = {{fa.o[0], fa.o[1], fa.o[2]}, {fa.a[0], fa.a[1], fa.a[2]}, {fa.b[0], fa.b[1], fa.b[2]}, {fa.d[0], fa.d[1], fa.d[2]}};
    const double FB[4][3] = {{fb.o[0], fb.o[1], fb.o[2]}, {fb.a[0], fb.a[1], fb.a[2]}, {fb.b[0], fb.b[1], fb.b[2]}, {fb.d[0], fb.d[1], fb.d[2]}};
    double t[9] = {0}; long ns = 0;
    const char *nm[9] = {"B trunk", "B forward", "A trunk", "A forward", "loss", "multiview", "A backward", "trunk bwd+step", "total"};
#define TS(k, stmt) do { nn_sync(); const double t0_ = now_s(); stmt; nn_sync(); if (r) t[k] += now_s() - t0_; } while (0)
    for (int r = 0; r <= N; r++) {
        nn_sync(); const double t00 = now_s();
        vwt_zero_grad(m);
        if (mv) {
            if (WT) TS(0, if (wt_forward_ld(WT, imd, &fb, 1, 9.6, xfd, xld)) return 1);
            TS(1, if (vwt_forward(m, imd, vad, phb, nullptr, 0)) return 1);
        }
        if (WT) TS(2, if (ttr ? wt_forward_train(WT, imd, &fa, 9.6, xfd, xld) : wt_forward_ld(WT, imd, &fa, 1, 9.6, xfd, xld)) return 1);
        TS(3, if (vwt_forward(m, imd, vad, ph, inc, 1)) return 1);
        vt_loss lo; TS(4, if (vt_losses(ph, inc, tg, cd, pw, T, L, 0.25f, 0.05f, 1, 1, 0.5f, g1, g2, &lo)) return 1);
        if (mv && mvh) TS(5, nn_d2h(pha_h, ph, n * 4); nn_d2h(phb_h, phb, n * 4); memset(gmv, 0, n * 4);
                      vt_multiview(pha_h, phb_h, va, va, FA, FB, T, L, 4, 0.5f, 0.25f, 0.5f, gmv, &ns); nn_h2d(gmvd, gmv, n * 4); vt_add(g1, gmvd, n));
        else if (mv) TS(5, vt_multiview_dev(ph, phb, vad, vad, FA, FB, T, L, 4, 0.5f, 0.25f, 0.5f, g1, &ns));
        TS(6, if (vwt_backward(m, g1, g2)) return 1);
        if (xgd) TS(7, wt_zero_grad(WT); if (wt_backward(WT, xgd, xld)) return 1; wt_step(WT, 1e-4f, 1e-5f, 0.f, 1.f, 0.999f));
        nn_sync(); if (r) t[8] += now_s() - t00;
    }
#undef TS
    printf("vwt bench: slab %dx%dx%d, cx %d, mv %d (%ld shared points): per sample", T, T, L, cx, mv, ns);
    for (int k = 0; k < 9; k++) if (t[k] > 0) printf("  %s %.1f", nm[k], t[k] / N * 1e3);
    printf(" ms = %.1f Mvox/s; device free %.1f GB\n", n / (t[8] / N) / 1e6, nn_mem_free() / 1e9);
    return 0;
}

/* --diag CKPT DUMPDIR N: the model (EMA weights) on N dumped slabs (ufsm vwt-sample: sNNN.{im,va}.u8, .comp.u16, .ph/.pw.f32) against a
   trivial predictor (the slab's best constant increment): phase / density losses and the share of 32-voxel in-component ray segments whose
   predicted turn count misses the target's by >= 0.5 */
static int rd(const char *dir, int i, const char *ext, void *p, size_t n) {
    char f[1200]; snprintf(f, sizeof f, "%s/s%03d.%s", dir, i, ext); FILE *fp = fopen(f, "rb"); if (!fp) return -1;
    const int ok = fread(p, 1, n, fp) == n; fclose(fp); return ok ? 0 : -1;
}
static void seg_miss(const float *ph, const float *tg, const uint16_t *cp, int T, int L, double *miss, double *tot) {
    for (long c = 0; c < (long)T * T; c++) for (int k = 0; k + 32 < L; k += 16) {
        const long a = c * L + k, b = a + 32; if (!cp[a] || cp[a] != cp[b]) continue;
        int same = 1; for (long q = a; q <= b; q += 4) same &= cp[q] == cp[a]; if (!same) continue;
        *tot += 1; *miss += fabs((ph[b] - ph[a]) - (tg[b] - tg[a])) >= 0.5;
    }
}
static int diag(int argc, char **argv) {
    const char *ck = opt(argc, argv, "--diag", nullptr), *dir = opt(argc, argv, "--dump", nullptr); const int N = atoi(opt(argc, argv, "--n", "8"));
    if (!dir) { fprintf(stderr, "vwt-train --diag CKPT --dump DIR [--n 8] [--ema 1]\n"); return 2; }
    vwt *m = vwt_load(ck, atoi(opt(argc, argv, "--ema", "1"))); if (!m) return 1;
    const int T = 128, L = 384; const size_t n = (size_t)T * T * L;
    uint8_t *im = malloc(n), *va = malloc(n); uint16_t *cp = malloc(n * 2); float *tg = malloc(n * 4), *pw = malloc(n * 4), *ph = malloc(n * 4), *pc = malloc(n * 4), *ic = malloc(n * 4);
    uint8_t *imd = nn_malloc(n), *vad = nn_malloc(n); uint16_t *cd = nn_malloc(n * 2); float *tgd = nn_malloc(n * 4), *pwd = nn_malloc(n * 4), *phd = nn_malloc(n * 4), *incd = nn_malloc(n * 4);
    double sm[6] = {0};
    for (int i = 0; i < N; i++) {
        if (rd(dir, i, "im.u8", im, n) || rd(dir, i, "va.u8", va, n) || rd(dir, i, "comp.u16", cp, n * 2) || rd(dir, i, "ph.f32", tg, n * 4) || rd(dir, i, "pw.f32", pw, n * 4)) { fprintf(stderr, "diag: cannot read slab %d\n", i); break; }
        for (size_t q = 0; q < n; q++) if (!cp[q]) tg[q] = 0.f;
        nn_h2d(imd, im, n); nn_h2d(vad, va, n); nn_h2d(cd, cp, n * 2); nn_h2d(tgd, tg, n * 4); nn_h2d(pwd, pw, n * 4);
        if (vwt_forward(m, imd, vad, phd, incd, 0)) return 1;
        vt_loss a, b; vt_losses(phd, incd, tgd, cd, pwd, T, L, 0.25f, 0.05f, 1, 1, 1, nullptr, nullptr, &a);
        nn_d2h(ph, phd, n * 4);
        double ds = 0, dn = 0; for (size_t q = 1; q < n; q++) if (q % L && cp[q] && cp[q] == cp[q - 1]) { ds += tg[q] - tg[q - 1]; dn++; }
        const float s0 = (float)(dn ? ds / dn : 0.05);
        for (size_t q = 0; q < n; q++) { pc[q] = s0 * (float)(q % L + 1); ic[q] = s0; }
        nn_h2d(phd, pc, n * 4); nn_h2d(incd, ic, n * 4); vt_losses(phd, incd, tgd, cd, pwd, T, L, 0.25f, 0.05f, 1, 1, 1, nullptr, nullptr, &b);
        double mm = 0, mt = 0, cm = 0, ct = 0; seg_miss(ph, tg, cp, T, L, &mm, &mt); seg_miss(pc, tg, cp, T, L, &cm, &ct);
        {   /* the model's mean increment on labelled steps and the correlation of 16-voxel turn counts (model vs target) */
            double sp = 0, np2 = 0, sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, nn2 = 0;
            for (long c0 = 0; c0 < (long)T * T; c0++) for (int k = 16; k < L; k += 8) {
                const long q = c0 * L + k, q0 = q - 16; if (!cp[q] || cp[q] != cp[q0]) continue;
                const double x = ph[q] - ph[q0], y = tg[q] - tg[q0]; sp += x; np2 += 16; sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y; nn2++;
            }
            const double cov = sxy / nn2 - sx / nn2 * sy / nn2, vx = sxx / nn2 - sx * sx / nn2 / nn2, vy = syy / nn2 - sy * sy / nn2 / nn2;
            printf("   model mean increment %.4f (target %.4f), corr of 16-vx counts %.3f\n", sp / np2, s0, cov / sqrt(vx * vy + 1e-30));
        }
        printf("slab %d: model phase %.4f density %.5f miss32 %.3f | constant %.4f: phase %.4f density %.5f miss32 %.3f\n", i, a.phase, a.density, mt ? mm / mt : 0, s0, b.phase, b.density, ct ? cm / ct : 0);
        sm[0] += a.phase; sm[1] += a.density; sm[2] += mt ? mm / mt : 0; sm[3] += b.phase; sm[4] += b.density; sm[5] += ct ? cm / ct : 0;
    }
    printf("mean: model phase %.4f density %.5f miss32 %.3f | constant phase %.4f density %.5f miss32 %.3f\n", sm[0] / N, sm[1] / N, sm[2] / N, sm[3] / N, sm[4] / N, sm[5] / N);
    return 0;
}

int cmd_vwt_train(int argc, char **argv) {
    if (nn_init(atoi(opt(argc, argv, "--gpu", "0")))) return 1;
    if (atoi(opt(argc, argv, "--tf32", "1"))) setenv("UFSM_ATTN_TF32", "1", 0);
    at_set_core(atoi(opt(argc, argv, "--core", "1")));
    at_set_mx(atoi(opt(argc, argv, "--mx", "16")));   /* attention-block linears / activations: 16 = bf16 (src/nn_attn_h16.cu), 0 fp32, 8 / 4 MX */
    if (opt(argc, argv, "--bench", nullptr)) return bench(argc, argv);
    if (opt(argc, argv, "--diag", nullptr)) return diag(argc, argv);
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
    const float lmv = (float)atof(opt(argc, argv, "--lam-mv", "0.25")), mvd = (float)atof(opt(argc, argv, "--mv-delta", "0.5")); const int mvs = atoi(opt(argc, argv, "--mv-stride", "4"));
    const int use_mv = atof(opt(argc, argv, "--mv-p", "0")) > 0 && lmv > 0;
    mkdir(run, 0755);
    char path[1200]; snprintf(path, sizeof path, "%s/latest.vwt", run);
    vwt_cfg c = default_cfg(T, L);
    /* --trunk CKPT: a frozen feature trunk (wind96 in-slab, src/wtrunk) feeding cfg.cx extra input channels; --cx-test N: N zero channels
       (no trunk; smoke tests of the wide input) */
    const char *trunk = opt(argc, argv, "--trunk", nullptr); wtrunk *WT = nullptr; int cx = atoi(opt(argc, argv, "--cx-test", "0"));
    /* --trunk-train 1 (stage 2): the trunk trains too (src/wtrunk.h wt_open_train): Muon at --trunk-muon-lr (default 0.002 = 0.1 x wind96's
       0.02) on its 3^3 convs, AdamW at --trunk-lr-mult (0.1) x the vwt lr elsewhere, both warmed up from 0 over --trunk-warmup (500) steps and
       then the vwt schedule's shape; its own EMA (--ema); RUN/latest.wt + step_N.wt (resumed from RUN/latest.wt before --trunk);
       --trunk-levels k: only the head + the last k decoder blocks; --trunk-wd (0) */
    const int trunk_train = atoi(opt(argc, argv, "--trunk-train", "0")), trunk_warm = atoi(opt(argc, argv, "--trunk-warmup", "500"));
    const float trunk_mlr = (float)atof(opt(argc, argv, "--trunk-muon-lr", "0.002")), trunk_lrm = (float)atof(opt(argc, argv, "--trunk-lr-mult", "0.1")), trunk_wd = (float)atof(opt(argc, argv, "--trunk-wd", "0"));
    uint16_t *xgd = nullptr;
    char wtp[1200]; snprintf(wtp, sizeof wtp, "%s/latest.wt", run);
    if (trunk && trunk_train) {
        const char *src_ = access(wtp, F_OK) == 0 ? wtp : trunk;
        WT = wt_open_train(src_, T, L, atoi(opt(argc, argv, "--trunk-levels", "0"))); if (!WT) { fprintf(stderr, "vwt-train: cannot open the trunk %s for training\n", src_); return 1; } cx = wt_channels(WT);
        xgd = (uint16_t *)1;   /* (set below: the features' gradient over their own input slots, written after the input conv's weight gradient read them) */
    } else if (trunk) { WT = wt_open(trunk, T, L, 1, atoi(opt(argc, argv, "--trunk-feat", "1"))); if (!WT) { fprintf(stderr, "vwt-train: cannot open the trunk %s\n", trunk); return 1; } cx = wt_channels(WT);
                 fprintf(stderr, "vwt-train: frozen trunk %s, %d feature channels\n", trunk, cx); }
    c.cx = cx;
    vwt *m = access(path, F_OK) == 0 ? vwt_load(path, 0) : vwt_create(&c, (unsigned)atoi(opt(argc, argv, "--seed", "1")));
    if (!m) return 1;
    if (access(path, F_OK) != 0 && opt(argc, argv, "--init-from", nullptr) && vwt_init_from(m, opt(argc, argv, "--init-from", nullptr), atoi(opt(argc, argv, "--init-ema", "1")))) return 1;
    int xld = cx; uint16_t *xfd = nullptr;   /* the trunk writes its features straight into the network's padded bf16 input */
    if (WT) xfd = vwt_xfeat_buf(m, &xld);
    if (xgd) { xgd = xfd; vwt_set_xgrad(m, xgd, xld); }   /* (stage 2: d loss / d features for the trunk's backward) */
    else if (cx) { xfd = nn_malloc((size_t)T * T * L * cx * 2); nn_zero(xfd, (size_t)T * T * L * cx * 2); vwt_set_xfeat(m, xfd); }
    const size_t np = vwt_nparams(m);
    float *ema = nn_malloc(np * 4), *mo = nn_malloc(np * 4), *vo = nn_malloc(np * 4), *red = nn_malloc(1 << 20);
    nn_d2d(ema, vwt_params(m), np * 4); nn_zero(mo, np * 4); nn_zero(vo, np * 4);
    int step = 0, nbad = 0;
    if (access(path, F_OK) == 0) {   /* resume: the EMA and the optimiser state (extra = step, m, v) */
        FILE *f = fopen(path, "rb"); vwt_cfg cc; uint64_t n2; int has; uint64_t eb;
        float *h = malloc(np * 4); int ok = f && !vwt_ckpt_header(f, &cc) && fread(&n2, 8, 1, f) == 1 && n2 == np && fread(h, 4, np, f) == np;
        if (ok && fread(&has, 4, 1, f) == 1 && has && fread(h, 4, np, f) == np) nn_h2d(ema, h, np * 4);
        if (ok && fread(&eb, 8, 1, f) == 1 && eb == 4 + 8 * np && fread(&step, 4, 1, f) == 1 && fread(h, 4, np, f) == np) { nn_h2d(mo, h, np * 4); if (fread(h, 4, np, f) == np) nn_h2d(vo, h, np * 4); }
        if (f) fclose(f); free(h);
        fprintf(stderr, "vwt-train: resumed %s at step %d\n", path, step);
    }
    signal(SIGTERM, on_term); signal(SIGINT, on_term);
    vws_set_args(argc, argv);
    /* --overfit DIR --n N: cycle over N dumped slabs instead of the sampler (correctness check: the loss must collapse) */
    const char *ofd = opt(argc, argv, "--overfit", nullptr); const int ofn = atoi(opt(argc, argv, "--n", "4"));
    vws_sample *OF = nullptr;
    if (ofd) {
        const size_t nn = (size_t)T * T * L; OF = calloc(ofn, sizeof *OF);
        for (int i = 0; i < ofn; i++) {
            vws_sample *q = &OF[i]; q->T = T; q->L = L; q->im = malloc(nn); q->va = malloc(nn); q->ph = malloc(nn * 4); q->comp = malloc(nn * 2); q->pw = malloc(nn * 4);
            if (rd(ofd, i, "im.u8", q->im, nn) || rd(ofd, i, "va.u8", q->va, nn) || rd(ofd, i, "ph.f32", q->ph, nn * 4) || rd(ofd, i, "comp.u16", q->comp, nn * 2) || rd(ofd, i, "pw.f32", q->pw, nn * 4)) { fprintf(stderr, "vwt-train: cannot read overfit slab %d\n", i); return 1; }
            for (size_t k = 0; k < nn; k++) if (!q->comp[k]) q->ph[k] = 0.f;
        }
    }
    vws *S = ofd ? nullptr : vws_open(src, T, L, (float)atof(opt(argc, argv, "--tilt", "15")), ufsm_threads(atoi(opt(argc, argv, "--threads", "0")), 12), (unsigned)(atoi(opt(argc, argv, "--seed", "1")) * 7919 + step), 1);
    if (!S && !ofd) { fprintf(stderr, "vwt-train: cannot open sources %s\n", src); return 1; }
    const size_t n = (size_t)T * T * L;
    uint8_t *imd = nn_malloc(n), *vad = nn_malloc(n); float *tg = nn_malloc(n * 4), *pw = nn_malloc(n * 4), *gph = nn_malloc(n * 4), *ginc = nn_malloc(n * 4);
    float *phd = nn_malloc(n * 4), *incd = nn_malloc(n * 4); uint16_t *cd = nn_malloc(n * 2); float *ones = malloc(n * 4); uint8_t *shd = nn_malloc(n);
    double acc_h = 0, acc_hc = 0, acc_mv = 0, acc_mvn = 0;
    float *phb = use_mv ? nn_malloc(n * 4) : nullptr, *pha_h = use_mv ? malloc(n * 4) : nullptr, *phb_h = use_mv ? malloc(n * 4) : nullptr, *gmv = use_mv ? malloc(n * 4) : nullptr;
    const int mvhost = atoi(opt(argc, argv, "--mv-host", "0"));
    float *gmvd = use_mv && mvhost ? nn_malloc(n * 4) : nullptr; uint8_t *vbd = use_mv && !mvhost ? nn_malloc(n) : nullptr;
    for (size_t i = 0; i < n; i++) ones[i] = 1.f;
    snprintf(path, sizeof path, "%s/log.tsv", run); FILE *lf = fopen(path, step ? "a" : "w"); if (!lf) return 1;
    if (!step) fprintf(lf, "step\tlr\tphase\tdensity\tgnorm\tsec_step\tsample_wait\tlabelled\tmono\tshell\tshell_cols\tmv\tmv_pairs\n");
    double acc_p = 0, acc_d = 0, acc_w = 0, acc_mono = 0, t_step = 0, t_wait = 0, acc_g = 0; int acc_n = 0;
    while (step < steps) {
        const double t0 = now_s(); double wait = 0;
        vwt_zero_grad(m); if (xgd) wt_zero_grad(WT);
        for (int b = 0; b < B; b++) {
            vws_sample s; const double tw = now_s(); if (OF) s = OF[(step * B + b) % ofn]; else if (vws_next(S, &s)) { fprintf(stderr, "vwt-train: sampler failed\n"); return 1; } wait += now_s() - tw;
            nn_h2d(imd, s.im, n); nn_h2d(vad, s.va, n); nn_h2d(tg, s.ph, n * 4); nn_h2d(cd, s.comp, n * 2); nn_h2d(pw, s.pw ? s.pw : ones, n * 4);
            /* monotone convention check on the host: share of labelled ray steps where the target rises */
            { size_t up = 0, tot = 0; for (size_t i = 1; i < n; i += 7) if (i % L && s.comp[i] && s.comp[i] == s.comp[i - 1]) { tot++; up += s.ph[i] > s.ph[i - 1]; } acc_mono += tot ? (double)up / tot : 0; }
            int pair = use_mv && s.has_pair;
            if (pair) {   /* B first (no activations kept), then A */
                nn_h2d(imd, s.pim, n); nn_h2d(vad, s.pva, n);
                if (WT) { const wt_frame fr = {{s.po[0], s.po[1], s.po[2]}, {s.pa[0], s.pa[1], s.pa[2]}, {s.pb[0], s.pb[1], s.pb[2]}, {s.pd[0], s.pd[1], s.pd[2]}, {s.axp[0], s.axp[1], s.axp[2]}, {0, 0, 1}};
                          if (wt_forward_ld(WT, imd, &fr, 1, s.um > 0 ? s.um : 9.6, xfd, xld)) return 1; }
                if (vwt_forward(m, imd, vad, phb, nullptr, 0)) return 1;
                if (!mvhost) nn_d2d(vbd, vad, n);   /* (B's valid for the device multiview) */
                nn_h2d(imd, s.im, n); nn_h2d(vad, s.va, n);
            }
            if (WT) { const wt_frame fr = {{s.o[0], s.o[1], s.o[2]}, {s.a[0], s.a[1], s.a[2]}, {s.b[0], s.b[1], s.b[2]}, {s.d[0], s.d[1], s.d[2]}, {s.axp[0], s.axp[1], s.axp[2]}, {0, 0, 1}};
                      if (xgd ? wt_forward_train(WT, imd, &fr, s.um > 0 ? s.um : 9.6, xfd, xld) : wt_forward_ld(WT, imd, &fr, 1, s.um > 0 ? s.um : 9.6, xfd, xld)) return 1; }
            if (vwt_forward(m, imd, vad, phd, incd, 1)) return 1;
            vt_loss lo; if (vt_losses(phd, incd, tg, cd, pw, T, L, dp, dd, lp, ld, 1.f / B, gph, ginc, &lo)) return 1;
            acc_p += lo.phase; acc_d += lo.density; acc_w += lo.wsum;
            if (use_shell && s.shell) {
                double hh, ss, nc; nn_h2d(shd, s.shell, n);
                if (vt_shell_loss(phd, shd, cd, T, L, gate, lsh, lsh, 1.f / B, gph, &hh, &ss, &nc)) return 1;
                acc_h += hh + ss; acc_hc += nc;
            }
            if (pair) {
                const double fa[4][3] = {{s.o[0], s.o[1], s.o[2]}, {s.a[0], s.a[1], s.a[2]}, {s.b[0], s.b[1], s.b[2]}, {s.d[0], s.d[1], s.d[2]}};
                const double fb[4][3] = {{s.po[0], s.po[1], s.po[2]}, {s.pa[0], s.pa[1], s.pa[2]}, {s.pb[0], s.pb[1], s.pb[2]}, {s.pd[0], s.pd[1], s.pd[2]}};
                long ns2 = 0; double lmvv;
                if (mvhost) {   /* --mv-host 1: the host reference */
                    nn_d2h(pha_h, phd, n * 4); nn_d2h(phb_h, phb, n * 4); memset(gmv, 0, n * 4);
                    lmvv = vt_multiview(pha_h, phb_h, s.va, s.pva, fa, fb, T, L, mvs, mvd, lmv, 1.f / B, gmv, &ns2);
                    if (ns2 >= 64) { nn_h2d(gmvd, gmv, n * 4); vt_add(gph, gmvd, n); }
                } else lmvv = vt_multiview_dev(phd, phb, vad, vbd, fa, fb, T, L, mvs, mvd, lmv, 1.f / B, gph, &ns2);
                if (ns2 >= 64) { acc_mv += lmvv; acc_mvn += 1; }
            }
            if (vwt_backward(m, gph, ginc)) return 1;
            if (xgd && wt_backward(WT, xgd, xld)) return 1;
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
        if (xgd) {   /* the trunk: warmup from 0, then the vwt schedule's shape */
            const double ts = wt_step_of(WT) + 1, sh = ts < trunk_warm ? ts / trunk_warm : (step < warm ? 1.0 : lr / lr0);
            wt_step(WT, (float)(trunk_mlr * sh), (float)(trunk_lrm * lr0 * sh), trunk_wd, clip, emad);
        }
        nn_sync(); t_step += now_s() - t0; t_wait += wait; acc_g += gn; acc_n++;
        if (step % logf == 0 || step == steps) {
            const double k = acc_n * B;
            fprintf(lf, "%d\t%.3g\t%.5f\t%.6f\t%.3f\t%.3f\t%.3f\t%.0f\t%.3f\t%.5f\t%.0f\t%.5f\t%.3f\n", step, lr, acc_p / k, acc_d / k, acc_g / acc_n, t_step / acc_n, t_wait / acc_n, acc_w / k, acc_mono / k, acc_h / k, acc_hc / k,
                    acc_mvn ? acc_mv / acc_mvn : 0, acc_mvn / k);
            fflush(lf);
            fprintf(stderr, "vwt-train: step %d lr %.2e phase %.4f density %.5f |g| %.2f  %.2f s/step (wait %.2f)  %.1f Mvox/s  mono %.2f\n", step, lr, acc_p / k, acc_d / k, acc_g / acc_n,
                    t_step / acc_n, t_wait / acc_n, (double)n * B / (t_step / acc_n) / 1e6, acc_mono / k);
            if (acc_mvn) fprintf(stderr, "   multiview %.4f on %.0f%% of the slabs\n", acc_mv / acc_mvn, 100 * acc_mvn / k);
            acc_p = acc_d = acc_w = acc_mono = t_step = t_wait = acc_g = acc_h = acc_hc = acc_mv = acc_mvn = 0; acc_n = 0;
        }
        snprintf(path, sizeof path, "%s/STOP", run); const int stop = access(path, F_OK) == 0 || g_term;
        if (step % ckev == 0 || step == steps || stop) {
            const size_t eb = 4 + 8 * np; uint8_t *ex = malloc(eb); memcpy(ex, &step, 4); nn_d2h(ex + 4, mo, np * 4); nn_d2h(ex + 4 + 4 * np, vo, np * 4);
            snprintf(path, sizeof path, "%s/latest.vwt", run);
            if (vwt_save(m, path, ema, ex, eb)) fprintf(stderr, "vwt-train: cannot save %s\n", path);
            if (step % ckev == 0) { char p2[1200]; snprintf(p2, sizeof p2, "%s/step_%06d.vwt", run, step); if (vwt_save(m, p2, ema, nullptr, 0)) fprintf(stderr, "vwt-train: cannot save %s\n", p2); }
            if (xgd) {
                if (wt_save(WT, wtp)) fprintf(stderr, "vwt-train: cannot save %s\n", wtp);
                if (step % ckev == 0) { char p2[1200]; snprintf(p2, sizeof p2, "%s/step_%06d.wt", run, step); if (wt_save(WT, p2)) fprintf(stderr, "vwt-train: cannot save %s\n", p2); }
            }
            free(ex);
        }
        if (stop) { fprintf(stderr, "vwt-train: STOP at step %d\n", step); break; }
    }
    fclose(lf); if (S) vws_close(S);
    return 0;
}

/* until src/vwt_sample.c lands (WM): weak stubs so the binary links; the real sampler overrides them */
__attribute__((weak)) vws *vws_open(const char *s, int T, int L, float t, int n, unsigned sd, int tr) { (void)s; (void)T; (void)L; (void)t; (void)n; (void)sd; (void)tr; fprintf(stderr, "vws_open: no sampler built in\n"); return nullptr; }
__attribute__((weak)) int vws_next(vws *s, vws_sample *o) { (void)s; (void)o; return -1; }
__attribute__((weak)) void vws_close(vws *s) { (void)s; }
__attribute__((weak)) void vws_set_args(int argc, char **argv) { (void)argc; (void)argv; }
#ifndef HAVE_WTRUNK   /* until src/wtrunk lands (PF) */
__attribute__((weak)) wtrunk *wt_open(const char *c, int T, int L, int mb, int feat) { (void)c; (void)T; (void)L; (void)mb; (void)feat; fprintf(stderr, "wt_open: no trunk built in\n"); return nullptr; }
__attribute__((weak)) int wt_forward(wtrunk *w, const uint8_t *im, const wt_frame *fr, int n, double um, uint16_t *out) { (void)w; (void)im; (void)fr; (void)n; (void)um; (void)out; return -1; }
__attribute__((weak)) int wt_forward_ld(wtrunk *w, const uint8_t *im, const wt_frame *fr, int n, double um, void *out, int ld) { (void)w; (void)im; (void)fr; (void)n; (void)um; (void)out; (void)ld; return -1; }
__attribute__((weak)) int wt_channels(const wtrunk *w) { (void)w; return 0; }
__attribute__((weak)) void wt_close(wtrunk *w) { (void)w; }
#endif
