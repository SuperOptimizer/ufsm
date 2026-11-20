/* nn_latdec_cells / nn_latdec_fwd / nn_latdec_bwd (the learned sub-cell decoder, unet_cfg.lat_dec) against a double host reference on a 12 x 8 x 160
   fp16 window (3 x 2 x 40 cells: two 32-cell row tiles, every face of the 3^3 neighbourhood clipped somewhere), for Hd 16 / 32 / 64, Cl 29
   (a random basis, step 0.25) and raw (Cl 64), 2 sub-cell groups (128 rows) and 7 (--lat-sub rv,fib,ink: 448 rows):
   - the latent cells (fp16): quantised as input_latc_k (a rounding tie may flip one step: at most 0.2 % of the values);
   - forward, fp16 and fp32 rows / pre rows: rows += W2 silu(pre + B * L) to fp16 rounding;
   - W2 = 0: the rows byte-identical (+-0 kept: a zero term is never added);
   - backward: the pre rows' gradient (fp16) and B / W2's gradients against the reference's analytic ones, which are checked by central
     differences on a few entries; the head plane scale (nn_set_head_gscale 4) divides the parameter gradients exactly;
   - two runs bit-identical (no atomics). The tensor-core path by default (fp16 operands: tolerances 1e-3 forward fp32 rows, 6e-3 gradients),
     UFSM_LATDEC_FFMA=1 the fp32 FFMA kernels (1e-4, 1e-3);
   - the MX-fp4 weight variant (nn_latdec_wq): the same checks against a reference on the host-quantised weights (wq_ref). Usage: test_latdec */
#include "nn.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
enum { D = 12, H = 8, W = 160, d = 3, h = 2, w = 40 };
static int NG = 2, NO = 128;   /* sub-cell groups (--lat-sub: up to 16) */
static double rnd(uint64_t *s) { *s = *s * 6364136223846793005ull + 1442695040888963407ull; return (double)(*s >> 11) * 0x1p-53 - 0.5; }
static float h2f(uint16_t v) { _Float16 f; memcpy(&f, &v, 2); return (float)f; }
static uint16_t f2h(float v) { _Float16 f = (_Float16)v; uint16_t r; memcpy(&r, &f, 2); return r; }
static int fails, WQ;   /* WQ: the MX-fp4 weight variant (nn_latdec_wq 1): the reference runs on the host-quantised weights (straight through) */
/* the kernels' MX-fp4 rounding (nn_latdec.cu latdec_wq_k): a block of fp16 values onto e2m1 x 2^e, e the smallest with amax 2^-e <= 6, ties to even */
static double e2m1(double x) { const double a = fabs(x);
    const double q = a <= 0.25 ? 0 : a < 0.75 ? 0.5 : a <= 1.25 ? 1 : a < 1.75 ? 1.5 : a <= 2.5 ? 2 : a < 3.5 ? 3 : a <= 5 ? 4 : 6; return x < 0 ? -q : q; }
static void wq_block(double **v, int n) {
    double am = 0; for (int i = 0; i < n; i++) { *v[i] = (_Float16)*v[i]; if (fabs(*v[i]) > am) am = fabs(*v[i]); }
    const float t = (float)am * (1.f / 6.f); uint32_t u; memcpy(&u, &t, 4); int e = (int)((u >> 23) & 0xff) - 127 + ((u & 0x7fffff) != 0); e = e < -24 ? -24 : e > 15 ? 15 : e;
    for (int i = 0; i < n; i++) *v[i] = (double)(_Float16)(e2m1(*v[i] * ldexp(1, -e)) * ldexp(1, e));
}
static void wq_ref(double *B, double *W2, int Hd, int Cl, int no) {   /* B [Cl][27][Hd] in the packed blocks (n, r, dx, 32 channels); W2 [no][Hd] along H */
    const int CP = Cl <= 32 ? 32 : 64; double zero[64], *v[64];
    for (int n = 0; n < Hd; n++) for (int t = 0; t < 9 * 3; t++) for (int c0 = 0; c0 < CP; c0 += 32) {
        for (int i = 0; i < 32; i++) { const int c = c0 + i; zero[i] = 0; v[i] = c < Cl ? &B[((size_t)c * 27 + t) * Hd + n] : &zero[i]; }
        wq_block(v, 32); }
    const int bs = Hd < 32 ? Hd : 32;
    for (int o = 0; o < no; o++) for (int k0 = 0; k0 < Hd; k0 += bs) { for (int i = 0; i < bs; i++) v[i] = &W2[(size_t)o * Hd + k0 + i]; wq_block(v, bs); }
}
static double TF = 1e-3, TG = 6e-3;   /* the tensor-core path (fp16 operands, fp32 sums); UFSM_LATDEC_FFMA=1 (fp32 throughout): 1e-4, 1e-3 */
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)
enum { NC = d * h * w };
/* L as a (planes -1 .. d) x h x w field: Lp(c, q, y, x), zero outside */
static double Lv(const float *L, int c, int q, int y, int x) { return q < -1 || q > d || y < 0 || y >= h || x < 0 || x >= w ? 0.0 : L[((size_t)c * (d + 2) + q + 1) * h * w + (size_t)y * w + x]; }
/* forward: y[o][n] (double) from pre (double [Hd][NC]), L, B, W2; also h, pre-activation a */
static void ref_fwd(const double *pre, const float *L, const double *B, const double *W2, int Hd, int Cl, double *y, double *a, double *hh) {
    for (int q = 0; q < d; q++) for (int yy = 0; yy < h; yy++) for (int x = 0; x < w; x++) {
        const size_t n = ((size_t)q * h + yy) * w + x;
        for (int k = 0; k < Hd; k++) {
            double s = pre[(size_t)k * NC + n];
            for (int c = 0; c < Cl; c++) for (int t = 0; t < 27; t++) s += B[((size_t)c * 27 + t) * Hd + k] * Lv(L, c, q + t / 9 - 1, yy + t / 3 % 3 - 1, x + t % 3 - 1);
            a[(size_t)k * NC + n] = s; hh[(size_t)k * NC + n] = s / (1 + exp(-s));
        }
        for (int o = 0; o < NO; o++) { double v = 0; for (int k = 0; k < Hd; k++) v += W2[(size_t)o * Hd + k] * hh[(size_t)k * NC + n]; y[(size_t)o * NC + n] = v; }
    }
}
static double ref_loss(const double *pre, const float *L, const double *B, const double *W2, int Hd, int Cl, const double *gy, double *y, double *a, double *hh) {
    ref_fwd(pre, L, B, W2, Hd, Cl, y, a, hh); double s = 0; for (size_t i = 0; i < (size_t)NO * NC; i++) s += gy[i] * y[i]; return s;
}
static void run(int Hd, int raw, int ng, uint64_t seed) {
    const int Cl = raw ? 64 : 29; uint64_t s = seed; NG = ng; NO = 64 * ng;
    const size_t V = (size_t)D * H * W, nB = (size_t)Cl * 27 * Hd, nW2 = (size_t)NO * Hd, NP = nB + nW2, nL = (size_t)Cl * (d + 2) * h * w;
    uint16_t *x = malloc(V * 2); float *bas = malloc(29 * 64 * 4), gain[29];
    for (size_t i = 0; i < V; i++) x[i] = f2h((float)(4 * rnd(&s)));
    for (int i = 0; i < 29 * 64; i++) bas[i] = (float)(0.25 * rnd(&s));
    for (int c = 0; c < 29; c++) gain[c] = (float)(0.5 + rnd(&s));
    void *xd = nn_malloc(V * 2); float *basd = nn_malloc(29 * 64 * 4), *gaind = nn_malloc(29 * 4), *Ld = nn_malloc(nL * 2);
    nn_h2d(xd, x, V * 2); nn_h2d(basd, bas, 29 * 64 * 4); nn_h2d(gaind, gain, 29 * 4);
    /* the latent cells, planes -1 .. d */
    const float step = 0.25f;
    nn_latdec_cells(xd, D, H, W, -1, d + 2, raw ? nullptr : basd, gaind, Cl, raw ? 0.f : step, Ld);
    float *L = malloc(nL * 4); { uint16_t *t = malloc(nL * 2); nn_sync(); nn_d2h(t, Ld, nL * 2); for (size_t i = 0; i < nL; i++) L[i] = h2f(t[i]); free(t); }   /* (fp16 cells) */
    size_t flip = 0, bad = 0;
    for (int c = 0; c < Cl; c++) for (int q = -1; q <= d; q++) for (int yy = 0; yy < h; yy++) for (int xx = 0; xx < w; xx++) {
        double r = 0;
        if (q >= 0 && q < d) {
            double cv[64]; for (int k = 0; k < 64; k++) cv[k] = h2f(x[((size_t)(4 * q + k / 16) * H + 4 * yy + k / 4 % 4) * W + 4 * xx + k % 4]);
            if (raw) r = cv[c];
            else { double t = 0; for (int k = 0; k < 64; k++) t += (double)bas[c * 64 + k] * cv[k];
                   const double lim = c ? 127 : 32767; t = fmin(lim, fmax(-lim, rint(t / step))) * step; r = t * gain[c]; }
        }
        const double g = Lv(L, c, q, yy, xx), e = fabs(g - r);
        if (e > 1e-3 * (1 + fabs(r))) { if (!raw && fabs(e - step * gain[c]) < 2e-3 * (1 + fabs(r))) flip++; else bad++; }   /* (fp16: 2^-11 relative) */
    }
    CHECK(bad == 0 && flip * 500 <= nL, "Hd %d raw %d: latent cells: %zu wrong, %zu quantiser flips of %zu", Hd, raw, bad, flip, nL);
    /* parameters, rows, pre rows, the rows' gradient */
    float *p = malloc(NP * 4); double *pB = malloc(nB * 8), *pW = malloc(nW2 * 8);
    for (size_t i = 0; i < nB; i++) pB[i] = p[i] = (float)(rnd(&s) * 2 / sqrt(27.0 * Cl));
    for (size_t i = 0; i < nW2; i++) pW[i] = p[nB + i] = (float)(rnd(&s) * 2 / sqrt((double)Hd));
    if (WQ) wq_ref(pB, pW, Hd, Cl, NO);
    nn_latdec_wq(WQ);
    uint16_t *rows = malloc((size_t)NO * NC * 2), *pre = malloc((size_t)Hd * NC * 2), *gy = malloc((size_t)(NO + Hd) * NC * 2);
    double *pred = malloc((size_t)Hd * NC * 8), *gyd = malloc((size_t)NO * NC * 8);
    for (size_t i = 0; i < (size_t)NO * NC; i++) { rows[i] = f2h((float)(4 * rnd(&s))); if (i % 7 == 0) rows[i] = i % 14 ? 0x8000 : 0; }   /* (+-0 entries) */
    for (size_t i = 0; i < (size_t)Hd * NC; i++) { pre[i] = f2h((float)(2 * rnd(&s))); pred[i] = h2f(pre[i]); }
    for (size_t i = 0; i < (size_t)NO * NC; i++) { gy[i] = f2h((float)(rnd(&s))); gyd[i] = h2f(gy[i]); }
    float *pd = nn_malloc(NP * 4); void *rd = nn_malloc((size_t)NO * NC * 4), *prd = nn_malloc((size_t)Hd * NC * 4), *gyd_ = nn_malloc((size_t)(NO + Hd) * NC * 2);
    nn_h2d(pd, p, NP * 4);
    double *y = malloc((size_t)NO * NC * 8), *a = malloc((size_t)Hd * NC * 8), *hh = malloc((size_t)Hd * NC * 8);
    ref_fwd(pred, L, pB, pW, Hd, Cl, y, a, hh);
    /* forward, the four type combinations */
    for (int r32 = 0; r32 < 2; r32++) for (int p32 = 0; p32 < 2; p32++) {
        float *rf = malloc((size_t)NO * NC * 4), *pf = malloc((size_t)Hd * NC * 4);
        if (r32) { for (size_t i = 0; i < (size_t)NO * NC; i++) rf[i] = h2f(rows[i]); nn_h2d(rd, rf, (size_t)NO * NC * 4); } else nn_h2d(rd, rows, (size_t)NO * NC * 2);
        if (p32) { for (size_t i = 0; i < (size_t)Hd * NC; i++) pf[i] = h2f(pre[i]); nn_h2d(prd, pf, (size_t)Hd * NC * 4); } else nn_h2d(prd, pre, (size_t)Hd * NC * 2);
        CHECK(!nn_latdec_fwd(rd, r32, NC, prd, p32, NC, Ld, d, h, w, pd, Hd, Cl, NG), "Hd %d raw %d: forward refused", Hd, raw);
        nn_sync();
        double me = 0; size_t nb = 0;
        if (r32) nn_d2h(rf, rd, (size_t)NO * NC * 4); else { uint16_t *t = malloc((size_t)NO * NC * 2); nn_d2h(t, rd, (size_t)NO * NC * 2); for (size_t i = 0; i < (size_t)NO * NC; i++) rf[i] = h2f(t[i]); free(t); }
        for (size_t i = 0; i < (size_t)NO * NC; i++) {
            const double r = h2f(rows[i]) + y[i], e = fabs(rf[i] - r), tol = r32 ? TF * (1 + fabs(r)) : 2e-3 * (1 + fabs(r));
            if (e > me) me = e; nb += e > tol;
        }
        CHECK(nb == 0, "Hd %d raw %d: forward r32 %d p32 %d: %zu values off, max |e| %.3g", Hd, raw, r32, p32, nb, me);
        free(rf); free(pf);
    }
    /* W2 = 0: byte-identical rows */
    { float *pz = malloc(NP * 4); memcpy(pz, p, NP * 4); memset(pz + nB, 0, nW2 * 4); float *pzd = nn_malloc(NP * 4); nn_h2d(pzd, pz, NP * 4);
      nn_h2d(rd, rows, (size_t)NO * NC * 2); nn_h2d(prd, pre, (size_t)Hd * NC * 2);
      nn_latdec_fwd(rd, 0, NC, prd, 0, NC, Ld, d, h, w, pzd, Hd, Cl, NG);
      uint16_t *t = malloc((size_t)NO * NC * 2); nn_sync(); nn_d2h(t, rd, (size_t)NO * NC * 2);
      CHECK(!memcmp(t, rows, (size_t)NO * NC * 2), "Hd %d raw %d: W2 = 0 changed the rows", Hd, raw);
      free(t); free(pz); nn_free(pzd); }
    /* backward: analytic reference */
    double *gB = calloc(nB, 8), *gW = calloc(nW2, 8), *gpre = malloc((size_t)Hd * NC * 8);
    for (size_t n = 0; n < NC; n++) {
        const int q = (int)(n / (h * w)), yy = (int)(n / w % h), xx = (int)(n % w);
        for (int k = 0; k < Hd; k++) {
            double dh = 0; for (int o = 0; o < NO; o++) { dh += pW[(size_t)o * Hd + k] * gyd[(size_t)o * NC + n]; gW[(size_t)o * Hd + k] += gyd[(size_t)o * NC + n] * hh[(size_t)k * NC + n]; }
            const double av = a[(size_t)k * NC + n], sg = 1 / (1 + exp(-av)), dp = dh * sg * (1 + av * (1 - sg));
            gpre[(size_t)k * NC + n] = dp;
            for (int c = 0; c < Cl; c++) for (int t = 0; t < 27; t++) gB[((size_t)c * 27 + t) * Hd + k] += dp * Lv(L, c, q + t / 9 - 1, yy + t / 3 % 3 - 1, xx + t % 3 - 1);
        }
    }
    /* the reference's gradient by central differences (a few B / W2 / pre entries) */
    { double *y2 = malloc((size_t)NO * NC * 8), *a2 = malloc((size_t)Hd * NC * 8), *h2 = malloc((size_t)Hd * NC * 8); const double e = 1e-4;
      for (int j = 0; j < 4; j++) {
          const size_t ib = (size_t)((rnd(&s) + 0.5) * (nB - 1)), iw = (size_t)((rnd(&s) + 0.5) * (nW2 - 1)), ip = (size_t)((rnd(&s) + 0.5) * (Hd * NC - 1));
          double *P[3] = {pB, pW, pred}; const size_t I[3] = {ib, iw, ip}; const double G[3] = {gB[ib], gW[iw], gpre[ip]};
          for (int k = 0; k < 3; k++) { const double v = P[k][I[k]];
              P[k][I[k]] = v + e; const double lp = ref_loss(pred, L, pB, pW, Hd, Cl, gyd, y2, a2, h2);
              P[k][I[k]] = v - e; const double lm = ref_loss(pred, L, pB, pW, Hd, Cl, gyd, y2, a2, h2); P[k][I[k]] = v;
              const double fd = (lp - lm) / (2 * e);
              CHECK(fabs(fd - G[k]) <= 1e-5 * (1 + fabs(G[k])) + 1e-6, "Hd %d raw %d: reference gradient %d entry %zu: analytic %.8g, central difference %.8g", Hd, raw, k, I[k], G[k], fd); }
      }
      free(y2); free(a2); free(h2); }
    /* the kernels, twice (bit-identical), then at head plane scale 4 */
    float *gpd = nn_malloc(NP * 4); void *scr = nn_malloc(nn_latdec_scratch(NC, Hd, Cl, NG));
    float *g1 = malloc(NP * 4), *g2 = malloc(NP * 4), *g4 = malloc(NP * 4); uint16_t *gp1 = malloc((size_t)Hd * NC * 2), *gp2 = malloc((size_t)Hd * NC * 2);
    nn_h2d(prd, pre, (size_t)Hd * NC * 2);
    for (int r = 0; r < 3; r++) {
        if (r == 2) nn_set_head_gscale(4.f);
        nn_zero(gpd, NP * 4); nn_h2d(gyd_, gy, (size_t)NO * NC * 2);
        CHECK(!nn_latdec_bwd(gyd_, NC, (char *)gyd_ + (size_t)NO * NC * 2, prd, NC, Ld, d, h, w, pd, Hd, Cl, NG, gpd, scr), "Hd %d raw %d: backward refused", Hd, raw);
        nn_sync(); nn_d2h(r == 0 ? g1 : r == 1 ? g2 : g4, gpd, NP * 4);
        if (r < 2) { uint16_t *t = malloc((size_t)(NO + Hd) * NC * 2); nn_d2h(t, gyd_, (size_t)(NO + Hd) * NC * 2); memcpy(r ? gp2 : gp1, t + (size_t)NO * NC, (size_t)Hd * NC * 2);
                     CHECK(!memcmp(t, gy, (size_t)NO * NC * 2), "Hd %d raw %d: the backward changed the rows' gradient", Hd, raw); free(t); }
        if (r == 2) nn_set_head_gscale(1.f);
    }
    CHECK(!memcmp(g1, g2, NP * 4) && !memcmp(gp1, gp2, (size_t)Hd * NC * 2), "Hd %d raw %d: two backward runs differ", Hd, raw);
    size_t ns4 = 0; for (size_t i = 0; i < NP; i++) ns4 += g4[i] != g1[i] * 0.25f;
    CHECK(ns4 == 0, "Hd %d raw %d: head plane scale 4: %zu parameter gradients not exactly / 4", Hd, raw, ns4);
    double mg = 0, mgB = 0, mgW = 0; size_t nbp = 0, nbB = 0, nbW = 0;
    for (size_t i = 0; i < (size_t)Hd * NC; i++) { const double e = fabs(h2f(gp1[i]) - gpre[i]); if (e > mg) mg = e; nbp += e > 2e-3 * (1 + fabs(gpre[i])); }
    double sB = 0, sW = 0; for (size_t i = 0; i < nB; i++) sB += fabs(gB[i]); for (size_t i = 0; i < nW2; i++) sW += fabs(gW[i]); sB /= nB; sW /= nW2;
    for (size_t i = 0; i < nB; i++) { const double e = fabs(g1[i] - gB[i]); if (e > mgB) mgB = e; nbB += e > TG * (sB + fabs(gB[i])); }
    for (size_t i = 0; i < nW2; i++) { const double e = fabs(g1[nB + i] - gW[i]); if (e > mgW) mgW = e; nbW += e > TG * (sW + fabs(gW[i])); }
    CHECK(nbp == 0 && nbB == 0 && nbW == 0, "Hd %d raw %d: backward: pre %zu off (max %.3g), B %zu off (max %.3g, mean |g| %.3g), W2 %zu off (max %.3g, mean |g| %.3g)", Hd, raw, nbp, mg, nbB, mgB, sB, nbW, mgW, sW);
    if (WQ) {   /* the quantisation is real: the unquantised reference differs */
        size_t nd = 0; for (size_t i = 0; i < nW2; i++) nd += pW[i] != p[nB + i]; CHECK(nd * 2 > nW2, "Hd %d raw %d: fp4: only %zu of %zu W2 values quantised away", Hd, raw, nd, nW2); }
    nn_latdec_wq(0);
    printf("test_latdec%s: Hd %2d Cl %2d ng %d: forward ok, W2 0 identical, backward max |e| pre %.2g B %.2g W2 %.2g, latent flips %zu\n", WQ ? " fp4" : "", Hd, Cl, NG, mg, mgB, mgW, flip);
    nn_free(xd); nn_free(basd); nn_free(gaind); nn_free(Ld); nn_free(pd); nn_free(rd); nn_free(prd); nn_free(gyd_); nn_free(gpd); nn_free(scr);
    free(x); free(bas); free(L); free(p); free(pB); free(pW); free(rows); free(pre); free(gy); free(pred); free(gyd); free(y); free(a); free(hh);
    free(gB); free(gW); free(gpre); free(g1); free(g2); free(g4); free(gp1); free(gp2);
}
int main(void) {
    nn_init(0);
    if (getenv("UFSM_LATDEC_FFMA") && atoi(getenv("UFSM_LATDEC_FFMA"))) { TF = 1e-4; TG = 1e-3; }
    run(16, 0, 2, 1); run(32, 0, 2, 2); run(64, 0, 2, 3); run(64, 1, 2, 4); run(32, 0, 7, 5);   /* (7: phase + rv + fibre + ink, 448 rows: the gradient in 4 chunks) */
    if (TF < 1e-3) {   /* FFMA: the fp4 weights are refused */
        nn_latdec_wq(1); CHECK(nn_latdec_fwd(nullptr, 0, 1, nullptr, 0, 1, nullptr, 1, 1, 2, nullptr, 32, 29, 2) == 1, "fp4 weights: the FFMA path did not refuse"); nn_latdec_wq(0);
    } else { WQ = 1; run(16, 0, 2, 6); run(64, 0, 7, 7); run(64, 1, 2, 8); WQ = 0; }
    if (fails) { fprintf(stderr, "test_latdec: %d failures\n", fails); return 1; }
    printf("test_latdec: ok\n");
    return 0;
}
