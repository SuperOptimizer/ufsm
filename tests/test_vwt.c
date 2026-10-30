/* vwt (src/vwt_net.c + src/nn_vwt.cu reference kernels): gradient check of the whole model on a small slab (T 16, L 24, trunk 48 =
   one head of 48, every layer type: strided / 1x1 / 3^3 convs, LN with and without GELU, stem residual, ray + transverse at_blocks,
   up2, concat, pixel-shuffle phase head) by central finite differences of J = sum a ph + sum b inc (fp32 forward, attention core fp32),
   sampled coordinates of every parameter group; and of the losses (vt_losses: per-component shift-invariant phase Huber + density
   Huber) w.r.t. ph / inc. Pass: per group max |fd - an| / max(rms(an), 1e-3 max|an|) < 6e-2 (fp32 forward: the FD noise floor at eps 0.05 is ~2-3e-2). */
#include "vwt.h"
#include "attn.h"
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 777;
static double urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) * 0x1.0p-53; }

static int T = 16, L = 24;
static double *A, *B;   /* functional weights */
static uint8_t *imd, *vad; static float *phd, *incd, *phh, *inch;

static double J(vwt *m) {
    const size_t n = (size_t)T * T * L;
    if (vwt_forward(m, imd, vad, phd, incd, 1)) { fprintf(stderr, "forward failed\n"); exit(1); }
    nn_d2h(phh, phd, n * 4); nn_d2h(inch, incd, n * 4);
    double s = 0; for (size_t i = 0; i < n; i++) s += A[i] * phh[i] + B[i] * inch[i];
    return s;
}

int main(void) {
    setenv("UFSM_VWT_REF", "1", 1);   /* the fp32 reference convs (bf16 would sit under the FD noise floor) */
    if (nn_init(0)) { printf("SKIP: no GPU\n"); return 0; }
    at_set_core(0); at_set_mx(0);
    vwt_cfg c = {T, L, {8, 8, 16}, 48, 1, 1, 1, 32, 8, 0.04f};
    vwt *m = vwt_create(&c, 5); if (!m) { printf("FAIL: create\n"); return 1; }
    const size_t np = vwt_nparams(m), n = (size_t)T * T * L;
    /* non-trivial weights: the out projections / phase head start at 0 -> randomise all parameters a little */
    float *P = malloc(np * 4); nn_d2h(P, vwt_params(m), np * 4);
    for (size_t i = 0; i < np; i++) if (P[i] == 0.f) P[i] = (float)(0.05 * (urand() - 0.5));
    nn_h2d(vwt_params(m), P, np * 4);
    uint8_t *im = malloc(n), *va = malloc(n);
    for (size_t i = 0; i < n; i++) { const int k = (int)(i % L); im[i] = (uint8_t)(100 + 80 * sin(k * 0.7 + (i / L) * 0.13) + 20 * urand()); va[i] = (i / L) % 13 != 0; }
    imd = nn_malloc(n); vad = nn_malloc(n); nn_h2d(imd, im, n); nn_h2d(vad, va, n);
    phd = nn_malloc(n * 4); incd = nn_malloc(n * 4); phh = malloc(n * 4); inch = malloc(n * 4);
    A = malloc(n * 8); B = malloc(n * 8); float *Af = malloc(n * 4), *Bf = malloc(n * 4);
    for (size_t i = 0; i < n; i++) { A[i] = urand() - 0.5; B[i] = urand() - 0.5; Af[i] = (float)A[i]; Bf[i] = (float)B[i]; }
    float *gad = nn_malloc(n * 4), *gbd = nn_malloc(n * 4); nn_h2d(gad, Af, n * 4); nn_h2d(gbd, Bf, n * 4);
    vwt_zero_grad(m); J(m);
    if (vwt_backward(m, gad, gbd)) { printf("FAIL: backward\n"); return 1; }
    float *G = malloc(np * 4); nn_d2h(G, vwt_grads(m), np * 4);
    /* parameter groups by index range: sample 24 coordinates per eighth of the vector + the last 300 (head) */
    int fails = 0; const int NG = 10;
    for (int gi = 0; gi < NG; gi++) {
        size_t lo = np * gi / NG, hi = np * (gi + 1) / NG; if (gi == NG - 1) lo = np > 400 ? np - 400 : 0;
        double ms = 0, mx = 0, err = 0; int ns = 16;
        double *an = malloc(ns * 8), *fd = malloc(ns * 8);
        for (int s = 0; s < ns; s++) {
            const size_t i = lo + (size_t)(urand() * (hi - lo)); const float p0 = P[i]; const float eps = (getenv("VWT_GC_EPS") ? (float)atof(getenv("VWT_GC_EPS")) : 5e-2f) * fmaxf(fabsf(p0), 0.1f);
            P[i] = p0 + eps; nn_h2d(vwt_params(m) + i, &P[i], 4); const double jp = J(m);
            P[i] = p0 - eps; nn_h2d(vwt_params(m) + i, &P[i], 4); const double jm = J(m);
            P[i] = p0; nn_h2d(vwt_params(m) + i, &P[i], 4);
            fd[s] = (jp - jm) / (2.0 * eps); an[s] = G[i]; ms += an[s] * an[s]; mx = fmax(mx, fabs(an[s]));
        }
        const double rms = sqrt(ms / ns), den = fmax(rms, 1e-3 * mx) + 1e-12;
        for (int s = 0; s < ns; s++) { err = fmax(err, fabs(fd[s] - an[s]) / den); if (getenv("VWT_GC_V")) printf("   an %+.5e fd %+.5e\n", an[s], fd[s]); }
        printf("params [%zu, %zu): rms grad %.3e  max |fd - an| / rms %.2e\n", lo, hi, rms, err);
        fails += err > 6e-2 && rms > 1e-9;
        free(an); free(fd);
    }
    /* losses */
    {
        uint16_t *comp = malloc(n * 2); float *tg = malloc(n * 4), *pw = malloc(n * 4);
        for (size_t i = 0; i < n; i++) { const int k = (int)(i % L); comp[i] = (i / L) % 7 == 0 ? 0 : (uint16_t)(1 + (i / L) % 3 + (k > 12)); tg[i] = 0.1f * k + 0.3f * (float)urand(); pw[i] = (float)(0.5 + 0.5 * urand()); }
        uint16_t *cd = nn_malloc(n * 2); float *tgd = nn_malloc(n * 4), *pwd = nn_malloc(n * 4), *gp = nn_malloc(n * 4), *gi = nn_malloc(n * 4);
        nn_h2d(cd, comp, n * 2); nn_h2d(tgd, tg, n * 4); nn_h2d(pwd, pw, n * 4);
        J(m);   /* ph / inc */
        vt_loss lo; vt_losses(phd, incd, tgd, cd, pwd, T, L, 0.25f, 0.05f, 1.f, 1.f, 1.f, gp, gi, &lo);
        float *gph = malloc(n * 4), *ginc = malloc(n * 4); nn_d2h(gph, gp, n * 4); nn_d2h(ginc, gi, n * 4);
        double e1 = 0, s1 = 0, e2 = 0, s2 = 0;
        for (int s = 0; s < 40; s++) {
            const size_t i = (size_t)(urand() * n); const float eps = 1e-3f; vt_loss a, b; float v0;
            nn_d2h(&v0, phd + i, 4); float v = v0 + eps; nn_h2d(phd + i, &v, 4); vt_losses(phd, incd, tgd, cd, pwd, T, L, 0.25f, 0.05f, 1.f, 1.f, 1.f, nullptr, nullptr, &a);
            v = v0 - eps; nn_h2d(phd + i, &v, 4); vt_losses(phd, incd, tgd, cd, pwd, T, L, 0.25f, 0.05f, 1.f, 1.f, 1.f, nullptr, nullptr, &b); nn_h2d(phd + i, &v0, 4);
            const double f = ((a.phase + a.density) - (b.phase + b.density)) / (2 * eps); e1 = fmax(e1, fabs(f - gph[i])); s1 = fmax(s1, fabs(gph[i]));
            nn_d2h(&v0, incd + i, 4); v = v0 + eps; nn_h2d(incd + i, &v, 4); vt_losses(phd, incd, tgd, cd, pwd, T, L, 0.25f, 0.05f, 1.f, 1.f, 1.f, nullptr, nullptr, &a);
            v = v0 - eps; nn_h2d(incd + i, &v, 4); vt_losses(phd, incd, tgd, cd, pwd, T, L, 0.25f, 0.05f, 1.f, 1.f, 1.f, nullptr, nullptr, &b); nn_h2d(incd + i, &v0, 4);
            const double f2 = ((a.phase + a.density) - (b.phase + b.density)) / (2 * eps); e2 = fmax(e2, fabs(f2 - ginc[i])); s2 = fmax(s2, fabs(ginc[i]));
        }
        printf("losses: phase %.4f density %.5f; d/d ph max err %.2e (max |g| %.2e), d/d inc max err %.2e (max |g| %.2e)\n", lo.phase, lo.density, e1, s1, e2, s2);
        fails += e1 > 3e-2 * s1 + 1e-6 || e2 > 3e-2 * s2 + 1e-6;
    }
    {   /* shell loss: monotone ph, shell runs every ~5 voxels in half the columns, FD vs the gradient */
        float *ph = malloc(n * 4); uint8_t *sh = calloc(n, 1); uint16_t *cz = calloc(n, 2);
        for (size_t c0 = 0; c0 < (size_t)T * T; c0++) { float a = 0; for (int k = 0; k < L; k++) { a += (float)(0.12 + 0.15 * urand()); ph[c0 * L + k] = a; if (c0 % 2 && k % 5 == 2) sh[c0 * L + k] = 1; } if (c0 % 6 == 1) cz[c0 * L + 3] = 1; }
        float *pd = nn_malloc(n * 4), *gd = nn_malloc(n * 4); uint8_t *sd = nn_malloc(n); uint16_t *czd = nn_malloc(n * 2);
        nn_h2d(pd, ph, n * 4); nn_h2d(sd, sh, n); nn_h2d(czd, cz, n * 2); nn_zero(gd, n * 4);
        double h0, s0, nc; vt_shell_loss(pd, sd, czd, T, L, 0.25f, 1.f, 1.f, 1.f, gd, &h0, &s0, &nc);
        float *g = malloc(n * 4); nn_d2h(g, gd, n * 4); double e = 0, gm = 0;
        for (int t = 0; t < 40; t++) {
            size_t i = (size_t)(urand() * n); i = (i / L) * L + 2 + 5 * (size_t)(urand() * 4); const float eps = 1e-3f, v0 = ph[i]; double hp, sp, hm, sm;
            float v = v0 + eps; nn_h2d(pd + i, &v, 4); vt_shell_loss(pd, sd, czd, T, L, 0.25f, 1.f, 1.f, 1.f, nullptr, &hp, &sp, &nc);
            v = v0 - eps; nn_h2d(pd + i, &v, 4); vt_shell_loss(pd, sd, czd, T, L, 0.25f, 1.f, 1.f, 1.f, nullptr, &hm, &sm, &nc); nn_h2d(pd + i, &v0, 4);
            const double f = ((hp + sp) - (hm + sm)) / (2 * eps); e = fmax(e, fabs(f - g[i])); gm = fmax(gm, fabs(g[i]));
        }
        printf("shell loss: hinge %.4f snap %.5f over %.0f columns; d/d ph max err %.2e (max |g| %.2e)\n", h0, s0, nc, e, gm);
        fails += e > 3e-2 * gm + 1e-6 || nc < 1;
    }
    printf(fails ? "FAIL (%d)\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
