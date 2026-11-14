/* gns.h (the trainer's gradient noise scale): synthetic per-window gradients g_i = G + e_i with a known mean G and i.i.d. noise of known
   covariance Sigma (diagonal, uneven) give the true B_noise = tr(Sigma) / |G|^2. The trainer's two pairings are replayed:
     accum: K steps of W windows each (the step's averaged gradient: b = W) against the update's mean (B = K W), |G_b|^2 the mean over the K steps;
     DP, no accum: each replica's own window (b = 1, the replicas' mean of |g|^2) against the all-reduced mean (B = W).
   The EMA ratio must land within 8 % of the truth over many updates; the per-update estimates' means (unbiased) within 3 %; unusable pairs
   (b >= B, NaN) leave the state alone and B_noise is NaN before the first pair. CPU only. */
#include "../src/gns.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static double unif(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return ((rs >> 11) + 0.5) * (1.0 / 9007199254740992.0); }
static double gauss(void) { return sqrt(-2 * log(unif())) * cos(6.283185307179586 * unif()); }

enum { D = 256 };
static double G[D], sd[D];

static double sq(const double *v) { double s = 0; for (int i = 0; i < D; i++) s += v[i] * v[i]; return s; }

/* one update: K steps of W windows. pair 0: (W, mean_k |step_k|^2) vs (K W, |mean|^2); pair 1: (1, mean |g_i|^2) vs (W, mean_k |step_k|^2) */
static void update(int K, int W, double *n2small, double *n2step, double *n2big) {
    double big[D] = {0}, ssm = 0, sst = 0;
    for (int k = 0; k < K; k++) {
        double st[D] = {0};
        for (int w = 0; w < W; w++) { double g[D]; for (int i = 0; i < D; i++) { g[i] = G[i] + sd[i] * gauss(); st[i] += g[i] / W; } ssm += sq(g); }
        for (int i = 0; i < D; i++) big[i] += st[i] / K;
        sst += sq(st);
    }
    *n2small = ssm / (K * W); *n2step = sst / K; *n2big = sq(big);
}

int main(void) {
    int fail = 0;
    double g2 = 0, tr = 0;
    for (int i = 0; i < D; i++) { G[i] = 0.06 * (unif() - 0.5); sd[i] = 0.43 * (0.2 + 1.5 * unif() * unif()); g2 += G[i] * G[i]; tr += sd[i] * sd[i]; }
    for (int i = 0; i < D; i++) G[i] /= sqrt(g2);   /* |G|^2 = 1 */
    g2 = 1;
    const double truth = tr / g2;
    printf("synthetic: |G|^2 %.4f, tr(Sigma) %.2f, true B_noise %.2f windows\n", g2, tr, truth);
    { gns_ema e = gns_make(0.9);
      if (!isnan(gns_bnoise(&e))) { printf("FAIL: B_noise before any pair is not NaN\n"); fail = 1; }
      if (gns_add(&e, 2, 1, 2, 1) || gns_add(&e, 3, 1, 2, 1) || gns_add(&e, 1, NAN, 2, 1) || e.n) { printf("FAIL: an unusable pair was taken\n"); fail = 1; } }
    const struct { int K, W, pair; const char *name; } cs[] = {{5, 2, 0, "accum 5 x 2 GPUs (b 2, B 10)"}, {4, 1, 0, "accum 4, 1 GPU (b 1, B 4)"},
                                                              {1, 2, 1, "2 GPUs, no accum (b 1, B 2)"}, {1, 8, 1, "8 replicas (b 1, B 8)"}};
    for (int c = 0; c < 4; c++) {
        const int U = 20000;
        gns_ema e = gns_make(0.999); double mg = 0, ms = 0;
        for (int u = 0; u < U; u++) {
            double a, st, b; update(cs[c].K, cs[c].W, &a, &st, &b);
            const double bs = cs[c].pair ? 1 : cs[c].W, Bb = cs[c].pair ? cs[c].W : cs[c].K * cs[c].W, ns = cs[c].pair ? a : st, nb = cs[c].pair ? st : b;
            gns_add(&e, bs, ns, Bb, nb);
            mg += (Bb * nb - bs * ns) / (Bb - bs) / U; ms += (ns - nb) / (1 / bs - 1 / Bb) / U;
        }
        const double bn = gns_bnoise(&e), r = bn / truth;
        const int ok = fabs(r - 1) < 0.08 && fabs(mg / g2 - 1) < 0.03 && fabs(ms / tr - 1) < 0.03;
        printf("  %-30s B_noise %.2f (x%.3f of the truth); |G|^2 %.4f, S %.2f (EMA); per-update means |G|^2 %.4f, S %.2f %s\n",
               cs[c].name, bn, r, gns_g2(&e), gns_s(&e), mg, ms, ok ? "ok" : "FAIL");
        fail |= !ok;
    }
    printf(fail ? "test_gns: FAIL\n" : "test_gns: ok\n");
    return fail;
}
