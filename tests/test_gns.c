/* gns.h (the trainer's gradient noise scale): synthetic per-window gradients g_i = G + e_i with a known mean G and i.i.d. noise of known
   covariance Sigma (diagonal, uneven) give the true B_noise = tr(Sigma) / |G|^2. The trainer's two pairings are replayed:
     accum: K steps of W windows each (the step's averaged gradient: b = W) against the update's mean (B = K W), |G_b|^2 the mean over the K steps;
     DP, no accum: each replica's own window (b = 1, the replicas' mean of |g|^2) against the all-reduced mean (B = W).
   The EMA ratio must land within 8 % of the truth over many updates; the per-update estimates' means (unbiased) within 3 %; unusable pairs
   (b >= B, NaN) leave the state alone and B_noise is NaN before the first pair. Then the --accum auto controller (accum_ctl) on synthetic
   B_noise traces: no change in the warm-in, the hold (hysteresis), the gap between changes, x2 steps, the cap, never down. CPU only. */
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

/* runs TRACE(u) (u = 1..U) through the controller from accum a0 at 2 windows per step; returns the final accum, the changes' update numbers in
   at[] and their accums in to[] (n); fails on a decrease or a step above x2 */
typedef double (*trace_fn)(long u);
static int run_ctl(trace_fn tr, long U, int a0, int amax, long *at, int *to, int *n, int *bad) {
    const accum_cfg f = {.k = 1, .margin = 1.25, .wps = 2, .warm = 1000, .hold = 200, .gap = 500, .amax = amax};
    accum_ctl c = accum_ctl_make(a0, &f); *n = 0;
    for (long u = 1; u <= U; u++) {
        const int prev = c.accum, a = accum_ctl_update(&c, &f, tr(u));
        if (a < prev || a > 2 * prev || a > amax) *bad = 1;
        if (a != prev && *n < 32) { at[*n] = u; to[(*n)++] = a; }
    }
    return c.accum;
}
static double tr_const11(long u) { (void)u; return 11; }          /* below 1.25 x 10 windows: never */
static double tr_const40(long u) { (void)u; return 40; }          /* 10 -> 20 at 1200 (x2 cap), -> 40 at 1700 (gap 500), then 40 !> 50 */
static double tr_flip(long u) { return (u / 150) % 2 ? 40 : 5; }  /* above for 150 updates at a time: the hold never fills */
static double tr_huge(long u) { (void)u; return 1e4; }            /* x2 per gap up to the cap */
static double tr_updown(long u) { return u < 3000 ? 60 : 2; }     /* never down */
static double tr_nan(long u) { return u % 150 == 0 ? NAN : 40; }  /* a NaN estimate restarts the hold */
static int test_ctl(void) {
    long at[32]; int to[32], n, bad = 0, fail = 0, a;
    a = run_ctl(tr_const11, 5000, 5, 64, at, to, &n, &bad);
    if (a != 5 || n) { printf("FAIL ctl: B_noise 11 at batch 10 changed accum (%d, %d changes)\n", a, n); fail = 1; }
    a = run_ctl(tr_const40, 5000, 5, 64, at, to, &n, &bad);
    if (n != 2 || at[0] != 1200 || to[0] != 10 || at[1] != 1700 || to[1] != 20 || a != 20) { printf("FAIL ctl: B_noise 40: %d changes (%ld -> %d, %ld -> %d), final %d\n", n, n > 0 ? at[0] : 0, n > 0 ? to[0] : 0, n > 1 ? at[1] : 0, n > 1 ? to[1] : 0, a); fail = 1; }
    else printf("  ctl B_noise 40 from accum 5 x 2 windows: accum 10 at update %ld, 20 at %ld, then held ok\n", at[0], at[1]);
    a = run_ctl(tr_flip, 20000, 5, 64, at, to, &n, &bad);
    if (n) { printf("FAIL ctl: 150-update excursions changed accum (first at %ld)\n", at[0]); fail = 1; }
    a = run_ctl(tr_huge, 10000, 5, 64, at, to, &n, &bad);
    if (a != 64 || n != 4 || to[0] != 10 || to[1] != 20 || to[2] != 40 || to[3] != 64 || at[1] - at[0] != 500) { printf("FAIL ctl: B_noise 1e4: final %d after %d changes\n", a, n); fail = 1; }
    else printf("  ctl B_noise 1e4: 10, 20, 40, 64 (cap) at updates %ld %ld %ld %ld ok\n", at[0], at[1], at[2], at[3]);
    a = run_ctl(tr_updown, 10000, 1, 64, at, to, &n, &bad);
    if (a != to[n - 1] || n < 1) { printf("FAIL ctl: up/down trace ended at %d\n", a); fail = 1; }
    else printf("  ctl B_noise 60 then 2 from accum 1: up to %d, kept after the drop ok\n", a);
    a = run_ctl(tr_nan, 10000, 5, 64, at, to, &n, &bad);
    if (n) { printf("FAIL ctl: NaN estimates every 150 updates did not restart the hold\n"); fail = 1; }
    if (bad) { printf("FAIL ctl: a decrease, a step above x2 or above the cap\n"); fail = 1; }
    return fail;
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
    fail |= test_ctl();
    printf(fail ? "test_gns: FAIL\n" : "test_gns: ok\n");
    return fail;
}
