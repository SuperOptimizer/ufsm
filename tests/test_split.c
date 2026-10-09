/* Spatial split (one window across two GPUs along z) against the same window on one GPU: loss and every parameter gradient,
   recompute 0 / 1 / 2, down_norm 0 / 1, three precision modes:
   - fp16 storage: the same math up to summation order; reference = the single-GPU run repeated (atomics);
   - the --fp4 2 (train default; make test also runs --fp4 1 via UFSM_TEST_POLICY) preset with round-to-nearest gradient operands: bit-identical up to fp4 roundings flipped by last-bit
     differences, which fp4 amplifies; reference = the single-GPU run on a 1e-6-perturbed input;
   - the --fp4 1 preset as trained (stochastic rounding): reference = the single-GPU difference between two rounding seeds.
   Also the memory modes of train --mem auto (chunked up-part gradient, MX-fp8 gradients, lean 1 / 2).
   With one GPU both halves run on it (the halves take turns on the host; UFSM_SPLIT_ONE_GPU=1 forces it). env: UFSM_P (96), UFSM_B (1), UFSM_ONLY=case (one case),
   UFSM_SEGS=1 (per-tensor errors), UFSM_TEST_POLICY (precision policy of the fp4 modes). */
#include "split.h"
#include "unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NCH 2
static int P, B, Dl, H0, DEV[2] = {0, 1};   /* one GPU: both halves on it (emulated split) */
static float *hx; static uint8_t *ht, *hm, *hw;   /* the whole window */
static _Float16 *hg; static int fixed_gl;   /* a fixed 16-bit logit gradient (zero on the halo planes) instead of the loss's */

typedef struct { unet *u; float *x, *gl, *scratch; uint8_t *t, *m, *w; } side_t;

static void upload_side(side_t *d, int side) {   /* this side's planes [z0, z0 + Dl) of the window; halo planes masked out */
    const size_t p2 = (size_t)P * P, p3 = p2 * P, l3 = p2 * Dl;
    const int z0 = side ? P / 2 - H0 : 0;
    float *x = malloc((size_t)B * 4 * l3 * 4); uint8_t *t = malloc((size_t)B * NCH * l3), *m = malloc((size_t)B * l3);
    for (int r = 0; r < B * 4; r++) memcpy(x + r * l3, hx + r * p3 + z0 * p2, l3 * 4);
    for (int r = 0; r < B * NCH; r++) memcpy(t + r * l3, ht + r * p3 + z0 * p2, l3);
    for (int n = 0; n < B; n++) {
        memcpy(m + n * l3, hm + n * p3 + z0 * p2, l3);
        const int hz0 = side ? 0 : Dl - H0;
        memset(m + n * l3 + hz0 * p2, 0, H0 * p2);
    }
    nn_h2d(d->x, x, (size_t)B * 4 * l3 * 4); nn_h2d(d->t, t, (size_t)B * NCH * l3); nn_h2d(d->m, m, (size_t)B * l3); nn_h2d(d->w, hw, (size_t)B * NCH);
    if (fixed_gl) {
        _Float16 *g = malloc((size_t)B * NCH * l3 * 2);
        for (int r = 0; r < B * NCH; r++) { memcpy(g + r * l3, hg + r * p3 + z0 * p2, l3 * 2); memset(g + r * l3 + (side ? 0 : Dl - H0) * p2, 0, H0 * p2 * 2); }
        nn_h2d(d->gl, g, (size_t)B * NCH * l3 * 2); free(g);
    }
    free(x); free(t); free(m);
}
static void side_alloc(side_t *d, const unet_cfg *cfg, int depth) {
    const size_t n3 = (size_t)P * P * depth;
    d->u = unet_create(cfg); unet_init(d->u, 7);
    d->x = nn_malloc((size_t)B * 4 * n3 * 4); d->t = nn_malloc((size_t)B * NCH * n3); d->m = nn_malloc((size_t)B * n3); d->w = nn_malloc((size_t)B * NCH);
    d->gl = nn_malloc((size_t)B * NCH * n3 * 2); d->scratch = nn_malloc(nn_loss_scratch((shape5){B, NCH, depth, P, P}) + 64);
}
static void side_free(side_t *d) { unet_free(d->u); nn_free(d->x); nn_free(d->t); nn_free(d->m); nn_free(d->w); nn_free(d->gl); nn_free(d->scratch); }

static double step(side_t *d, int depth, int train) {   /* forward, loss, backward; returns nothing useful on its own */
    shape5 xs = {B, 4, depth, P, P};
    const float *lg = unet_forward(d->u, d->x, xs, train);
    shape5 os = unet_out_shape(d->u, xs);
    nn_set_logits_h16(unet_logits_h16(d->u));   /* lean 2: fp16 logits in a gradient buffer */
    nn_loss_async(lg, d->t, d->m, d->w, os, 0.5f, train && !fixed_gl ? d->gl : nullptr, d->scratch);
    if (train) { unet_zero_grad(d->u); unet_backward_x(d->u, d->gl, 1); }
    return 0;
}
static double fetch(side_t *d, int depth) {
    float out[2 * NCH + 1]; nn_loss_fetch(d->scratch, (shape5){B, NCH, depth, P, P}, out);
    double l = 0; for (int c = 0; c < NCH; c++) l += out[c] + 0.5 * out[NCH + c];
    return l / NCH;
}

static side_t S2[2];
static void job(int side, void *arg) { (void)arg; step(&S2[side], Dl, 1); }

static double single(const unet_cfg *cfg, unsigned sr_step, float *g) {
    nn_init(0); nn_set_sr_step(sr_step);
    side_t d; side_alloc(&d, cfg, P);
    nn_h2d(d.x, hx, (size_t)B * 4 * P * P * P * 4); nn_h2d(d.t, ht, (size_t)B * NCH * P * P * P); nn_h2d(d.m, hm, (size_t)B * P * P * P); nn_h2d(d.w, hw, (size_t)B * NCH);
    if (fixed_gl) nn_h2d(d.gl, hg, (size_t)B * NCH * P * P * P * 2);
    step(&d, P, 1);
    double l = fetch(&d, P);
    unet_grad_d2h(d.u, g);
    side_free(&d);
    return l;
}
static double split(split_ctx *c, const unet_cfg *cfg, unsigned sr_step, float *g, size_t np) {
    for (int s = 0; s < 2; s++) { nn_init(DEV[s]); nn_set_sr_step(sr_step); side_alloc(&S2[s], cfg, Dl); unet_set_split(S2[s].u, s, H0, split_halo); if (!getenv("UFSM_SPLIT_SYNC")) unet_set_split_async(S2[s].u, split_halo_begin, split_halo_end); upload_side(&S2[s], s); }
    split_run(c, job, nullptr);
    double l[2]; float *g1 = malloc(np * 4);
    for (int s = 0; s < 2; s++) { nn_init(DEV[s]); l[s] = fetch(&S2[s], Dl); unet_grad_d2h(S2[s].u, s ? g1 : g); }
    for (size_t i = 0; i < np; i++) g[i] += g1[i];
    if (l[0] != l[1]) printf("    (losses of the two sides differ: %.9g %.9g)\n", l[0], l[1]);
    for (int s = 0; s < 2; s++) { nn_init(DEV[s]); side_free(&S2[s]); }
    free(g1);
    return l[0];
}

/* relative L2 difference overall and the worst parameter tensor */
typedef struct { const char *name; size_t off, len; } seg_t;
static int nseg; static seg_t segs[128]; static char names[128][24];
static size_t addseg(const char *nm, size_t off, size_t len) { snprintf(names[nseg], 24, "%s", nm); segs[nseg] = (seg_t){names[nseg], off, len}; nseg++; return off + len; }
static size_t addconv(const char *nm, int ci, int co, int k, size_t off) { char b[24]; snprintf(b, 24, "%s.w", nm); off = addseg(b, off, (size_t)co * ci * k * k * k); snprintf(b, 24, "%s.b", nm); return addseg(b, off, co); }
static size_t addgn(const char *nm, int c, size_t off) { char b[24]; snprintf(b, 24, "%s.gn", nm); return addseg(b, off, 2 * (size_t)c); }
static void segs_build(const unet_cfg *cfg) {
    const int *w = cfg->widths, L = cfg->nlev; size_t off = 0; char b[24]; nseg = 0;
    for (int i = 0; i < L; i++) { int ci = i ? w[i - 1] : cfg->cin; snprintf(b, 24, "enc%d.c1", i); off = addconv(b, ci, w[i], 3, off); snprintf(b, 24, "enc%d.n1", i); off = addgn(b, w[i], off); snprintf(b, 24, "enc%d.c2", i); off = addconv(b, w[i], w[i], 3, off); snprintf(b, 24, "enc%d.n2", i); off = addgn(b, w[i], off); }
    for (int i = 0; i < L - 1; i++) { snprintf(b, 24, "down%d", i); off = addconv(b, w[i], w[i], 3, off); }
#define DW(i) ((i) < L - 1 && cfg->dec_widths[i] ? cfg->dec_widths[i] : w[i])   /* decoder widths (narrow coarse levels) */
    for (int i = L - 2; i >= 0; i--) { int ci = w[i] + DW(i + 1), co = DW(i); snprintf(b, 24, "dec%d.c1", i); off = addconv(b, ci, co, 3, off); snprintf(b, 24, "dec%d.n1", i); off = addgn(b, co, off); snprintf(b, 24, "dec%d.c2", i); off = addconv(b, co, co, 3, off); snprintf(b, 24, "dec%d.n2", i); off = addgn(b, co, off); }
#undef DW
    off = addconv("head", w[0], cfg->cout, 1, off);
    if (cfg->down_norm) for (int i = 0; i < L - 1; i++) { snprintf(b, 24, "dn%d", i); off = addgn(b, w[i], off); }
}
static double rel(const float *a, const float *b, size_t off, size_t n) {
    double d2 = 0, r2 = 0;
    for (size_t i = off; i < off + n; i++) { double d = (double)a[i] - b[i]; d2 += d * d; r2 += (double)b[i] * b[i]; }
    return sqrt(d2 / (r2 + 1e-300));
}
static double worst(const float *a, const float *b, const char **nm) {
    double m = 0;
    for (int i = 0; i < nseg; i++) { double r = rel(a, b, segs[i].off, segs[i].len); if (r > m) { m = r; *nm = segs[i].name; } }
    return m;
}

int main(void) {
    if (nn_init(0)) { printf("test_split: no GPU, skipped\n"); return 0; }
    if (ufsm_env_on("UFSM_SPLIT_ONE_GPU") || nn_init(1)) { DEV[1] = 0; nn_init(0); nn_check(); }   /* (clears the failed probe's error) */
    P = getenv("UFSM_P") ? atoi(getenv("UFSM_P")) : 96; B = getenv("UFSM_B") ? atoi(getenv("UFSM_B")) : 1;
    unet_cfg cfg = {4, {16, 32, 64, 80}, 4, NCH, 8, 1};
    H0 = 1 << (cfg.nlev - 1); Dl = P / 2 + H0;
    if (P % (2 * H0)) { fprintf(stderr, "P must be a multiple of %d\n", 2 * H0); return 2; }
    const size_t p3 = (size_t)P * P * P;
    hx = malloc((size_t)B * 4 * p3 * 4); ht = malloc((size_t)B * NCH * p3); hm = malloc((size_t)B * p3); hw = malloc((size_t)B * NCH);
    srand(5);
    /* a smooth random field (sum of a few plane waves) so the network sees structure along z */
    for (int n = 0; n < B; n++) for (int c = 0; c < 4; c++) {
        double k[3][3], ph[3]; for (int j = 0; j < 3; j++) { for (int a = 0; a < 3; a++) k[j][a] = (rand() % 1000) / 1000.0 * 0.4; ph[j] = rand() % 628 / 100.0; }
        for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
            double v = 0; for (int j = 0; j < 3; j++) v += sin(k[j][0] * z + k[j][1] * y + k[j][2] * x + ph[j]);
            hx[(((size_t)n * 4 + c) * P + z) * P * P + (size_t)y * P + x] = (float)(v + 0.3 * ((rand() % 2001) / 1000.0 - 1));
        }
    }
    for (size_t i = 0; i < (size_t)B * NCH * p3; i++) { float v = hx[i % ((size_t)B * 4 * p3)]; ht[i] = v > 1.2f ? 255 : v > 0.8f ? 120 : 0; }
    for (size_t i = 0; i < (size_t)B * p3; i++) hm[i] = rand() % 10 != 0;
    for (int i = 0; i < B * NCH; i++) hw[i] = 1;
    hg = malloc((size_t)B * NCH * p3 * 2);
    for (size_t i = 0; i < (size_t)B * NCH * p3; i++) hg[i] = (_Float16)((float)((i * 2246822519u) % 2001) / 1000.f - 1.f);   /* x grad scale 1024: ~1e-3 */

    nn_set_f16(1); nn_set_grad_scale(1024.f); nn_set_loss_grad_h16(1);
    nn_set_gn_stored(ufsm_env_on("UFSM_TEST_GN_STORED"));
    if (getenv("UFSM_TEST_INPUT_PREC") && unet_set_input_prec(atoi(getenv("UFSM_TEST_INPUT_PREC")))) return 2;
    setenv("UFSM_F4_WGRAD", "1", 0);   /* as train --fp4 2: fp4 weight gradients where the policy asks for them */
    split_ctx *ctx = split_create(DEV[0], DEV[1]);
    size_t np = 0;   /* host gradients sized for the widest case (the trainer-default widths, down_norm) */
    { unet_cfg wc = {4, {32, 64, 96, 128}, 4, NCH, 8, 1}; unet *probe = unet_create(&wc); np = unet_nparams(probe); unet_free(probe); }
    float *g0 = malloc(np * 4), *g1 = malloc(np * 4), *g2 = malloc(np * 4);
    int fails = 0;
    printf("test_split: P %d B %d, level-0 halo %d planes, local depth %d%s\n", P, B, H0, Dl, DEV[0] == DEV[1] ? ", both halves on one GPU" : "");
    /* configurations: (precision, down_norm, recompute, chunk, MX-fp8 gradients, lean); the memory modes are those train --mem
       auto picks from (chunked up-part gradient, MX-fp8 gradients, lean 1 / 2) */
    typedef struct { int fp4, dn, rc, chunk, gmx, lean, desk; } tcfg;
    tcfg cases[64]; int ncase = 0;
    for (int fp4 = 0; fp4 < 3; fp4++) {   /* 0 fp16, 1 --fp4 1 with round-to-nearest gradient operands, 2 --fp4 1 (stochastic rounding) */
        for (int dn = 0; dn < 2; dn++) for (int rc = 0; rc < 3; rc++) cases[ncase++] = (tcfg){fp4, dn, rc, 1, 0, 0, 0};
        cases[ncase++] = (tcfg){fp4, 1, 1, 2, 0, 0, 0}; cases[ncase++] = (tcfg){fp4, 1, 2, 2, 0, 2, 0};
        if (fp4) { static const int mm[5][4] = {{1, 1, 1, 0}, {2, 1, 1, 0}, {2, 1, 1, 1}, {2, 1, 1, 2}, {2, 2, 1, 2}};
                   for (int k = 0; k < 5; k++) cases[ncase++] = (tcfg){fp4, 1, mm[k][1], mm[k][0], mm[k][2], mm[k][3], 0}; }
    }
    /* the trainer's defaults (widths multiples of 32): MX-fp4 gradients, shared encoder a1 with its host offload, the input
       and skip offloads, every decoder level's gradient buffer in its a2's buffer */
    cases[ncase++] = (tcfg){2, 1, 1, 2, 1, 2, 1};
    cases[ncase++] = (tcfg){2, 1, 1, 2, 1, 1, 1};
    cases[ncase++] = (tcfg){2, 1, 1, 2, 1, 2, 2};   /* the same with narrower decoder levels 1 and 2 (unet_cfg.dec_widths) */
    for (int ci = 0; ci < ncase; ci++) {
        const tcfg c = cases[ci];
        if (getenv("UFSM_ONLY") && ci != atoi(getenv("UFSM_ONLY"))) continue;
        if (c.desk && getenv("UFSM_TEST_POLICY") && strstr(getenv("UFSM_TEST_POLICY"), "fp4:fp4:fp8")) continue;   /* MX-fp4 gradients need fp4 weight gradients */
        const int fp4 = c.fp4, dn = c.dn, rc = c.rc;
        if (fp4) { unet_set_act_mx4(1); nn_set_sr(fp4 == 2); if (nn_set_prec_policy(getenv("UFSM_TEST_POLICY") ? getenv("UFSM_TEST_POLICY") : "all=fp4:fp4:fp4,enc0.c1=fp16")) return 2; }
        else { unet_set_act_mx4(0); nn_set_sr(0); nn_set_prec_policy(""); }
        unet_set_chunk_up(c.chunk); unet_set_grad_mx8(c.gmx); unet_set_lean(c.lean);
        unet_set_grad_mx4(c.desk != 0); unet_set_share_enc_a1(c.desk != 0);
        static const unet_cfg base = {4, {16, 32, 64, 80}, 4, NCH, 8, 1}, desk = {4, {32, 64, 96, 128}, 4, NCH, 8, 1};
        cfg = c.desk ? desk : base;
        if (c.desk == 2) { cfg.dec_widths[1] = 32; cfg.dec_widths[2] = 64; }
        char tag[128]; snprintf(tag, sizeof tag, "%2d %s down_norm %d recompute %d%s%s%s%s", ci, fp4 == 0 ? "fp16  " : fp4 == 1 ? "fp4 rn" : "fp4 sr", dn, rc,
                                c.chunk == 2 ? " chunk 2" : "", c.gmx ? " grad-mx8" : "", c.lean == 2 ? " lean 2" : c.lean ? " lean 1" : "",
                                c.desk == 2 ? " +fp4 grads, offloads, decoder 32/64 at levels 1/2" : c.desk ? " +fp4 grads, offloads (train defaults)" : "");
        {
                cfg.down_norm = dn; segs_build(&cfg);
                { unet *pr = unet_create(&cfg); np = unet_nparams(pr); unet_free(pr); }
                unet_set_recompute(rc);
                double ls = single(&cfg, 1, g0), lp = split(ctx, &cfg, 1, g1, np);
                const char *wn = "", *wn2 = "";
                double dl = fabs(lp - ls) / fabs(ls), dg = rel(g1, g0, 0, np), wg = worst(g1, g0, &wn);
                if (fp4 < 2) {
                    /* fp16 storage: identical math up to summation order (GroupNorm and loss sums over two halves); fp4 with
                       round-to-nearest: the same up to roundings flipped by those last-bit differences. Reference: the single-GPU
                       run repeated (atomics make it nondeterministic in the last bits). */
                    /* fp4 reference: the single-GPU run on an input perturbed by ~1e-6 (relative), the size of the
                       summation-order differences: a flipped fp4 rounding is amplified downstream */
                    float *hx0 = nullptr; const size_t nx = (size_t)B * 4 * P * P * P;
                    if (fp4) { hx0 = malloc(nx * 4); memcpy(hx0, hx, nx * 4); for (size_t i = 0; i < nx; i++) hx[i] *= 1.f + 1e-6f * (float)((int)(i * 2654435761u >> 16 & 1023) - 512) / 512.f; }
                    double lr_ = single(&cfg, 1, g2), rg = rel(g2, g0, 0, np);
                    if (fp4) { memcpy(hx, hx0, nx * 4); free(hx0); }
                    int ok = fp4 ? dl < 1e-3 && dg < 2 * rg + 1e-3 : dl < 1e-4 && dg < 2e-3 && wg < 1e-2;
                    printf("  %s: loss %.6f vs %.6f (rel %.1e; %s %.1e), grad rel %.2e (%s %.2e), worst %s %.2e  %s\n", tag,
                           lp, ls, dl, fp4 ? "perturbed" : "rerun", fabs(lr_ - ls) / fabs(ls), dg, fp4 ? "perturbed" : "rerun", rg, wn, wg, ok ? "ok" : "FAIL");
                    fails += !ok;
                    if (getenv("UFSM_SEGS")) for (int i = 0; i < nseg; i++) printf("      %-10s %.2e\n", segs[i].name, rel(g1, g0, segs[i].off, segs[i].len));
                } else {
                    /* fp4: compare with the single-GPU difference between two rounding seeds */
                    double ln = single(&cfg, 2, g2);
                    double nl = fabs(ln - ls) / fabs(ls), ng = rel(g2, g0, 0, np), nw = worst(g2, g0, &wn2);
                    int ok = dl < 3 * nl + 1e-3 && dg < 1.5 * ng + 1e-3 && wg < 1.5 * nw + 0.05;   /* + floor: some modes do not round stochastically (seeds identical) */
                    printf("  %s: loss %.6f vs %.6f (rel %.1e; seeds %.1e), grad rel %.3g (seeds %.3g), worst %s %.3g (seeds %s %.3g)  %s\n",
                           tag, lp, ls, dl, nl, dg, ng, wn, wg, wn2, nw, ok ? "ok" : "FAIL");
                    fails += !ok;
                    if (c.desk) {   /* the memory layout (shared a1 + offload, input / skip offloads, A in a2) against separate
                                       buffers with the same rounding keys: the same arithmetic. A fixed logit gradient: without the
                                       level-0 A share a 2-channel head keeps fp32 logits, whose loss gradient differs in the last
                                       bits. MX-fp4 gradients are not bitwise reproducible: a last-bit difference of an atomic sum
                                       now and then flips a stochastic rounding, and the change spreads (an equally valid sample, as
                                       far from MX-fp8 gradients); so up to three tries, a layout error fails every one. */
                        fixed_gl = 1;
                        double dsp = 1, d1 = 1, la = 0, lq = 0;
                        float *h1 = malloc(np * 4);
                        for (int t = 0; t < 3 && (dsp >= 1e-5 || d1 >= 1e-5); t++) {
                            la = split(ctx, &cfg, 1, g1, np); single(&cfg, 1, h1);
                            unet_set_share_enc_a1(0); unet_set_a0_share(0); unet_set_xin_offload(0);
                            lq = split(ctx, &cfg, 1, g2, np); single(&cfg, 1, g0);
                            unet_set_share_enc_a1(1); unet_set_a0_share(-1); unet_set_xin_offload(-1);
                            const double a = rel(g1, g2, 0, np), b = rel(h1, g0, 0, np);
                            if (a < dsp) dsp = a;
                            if (b < d1) d1 = b;
                        }
                        free(h1); fixed_gl = 0;
                        const int ok2 = fabs(lq - la) < 1e-5 * fabs(la) && dsp < 1e-5 && d1 < 1e-5;   /* (the loss reads fp16 or fp32 logits) */
                        printf("  %s: memory layout vs separate buffers: split grad rel %.3g, one GPU %.3g  %s\n", tag, dsp, d1, ok2 ? "ok" : "FAIL");
                        fails += !ok2;
                    }
                }
                const char *e = nn_check(); if (e) { printf("cuda error: %s\n", e); return 1; }
        }
    }
    split_free(ctx);
    printf("test_split: %s\n", fails ? "FAILED" : "passed");
    return fails ? 1 : 0;
}
