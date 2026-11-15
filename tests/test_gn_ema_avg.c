/* net#5 (audit 2026-11-14): --gn-ema under data parallel, on the CPU. train.c gn_ema_average copies the replicas' running GroupNorm
   statistics to the host, averages them with unet_gn_ema_mean_host and writes the mean back; unet_gn_ema_average_due says on which steps.
   Checks: (1) after an average every replica holds the same floats, the double mean of the inputs rounded once, whatever the replica
   order; (2) the schedule: every step from the warm-up on, before it only at validation / checkpoint / last / stopping steps;
   (3) the claim the schedule rests on: the running update (ge_after: run = a run + (1 - a) batch, the first update a copy) is linear, so
   replicas averaged only on the due steps hold, on every due step, the statistics of replicas averaged every step (float rounding),
   and from the warm-up on (averaged every step) they stay equal, so both training forwards read the same statistics. */
#include "../src/unet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

static uint64_t rs = 0x9e3779b97f4a7c15ull;
static float rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (float)((rs >> 11) * 0x1.0p-53); }

static void ema(float *run, const float *bat, size_t n, long nupd, float mom) {   /* ge_after's update (nn_ema_finite on finite input) */
    if (!nupd) { memcpy(run, bat, n * 4); return; }
    const float a = 1.f - mom;
    for (size_t i = 0; i < n; i++) run[i] = a * run[i] + (1.f - a) * bat[i];
}

int main(void) {
    /* (1) the mean: same floats everywhere, the double mean rounded once, order-independent */
    for (int ng = 2; ng <= 4; ng++) {
        const size_t n = 1000; float *h[4], *r[4]; double *ref = calloc(n, sizeof *ref);
        for (int g = 0; g < ng; g++) { h[g] = malloc(n * 4); r[g] = malloc(n * 4); for (size_t i = 0; i < n; i++) { h[g][i] = (rnd() - .5f) * (i % 7 ? 3.f : 3e4f); ref[i] += h[g][i]; } }
        for (int g = 0; g < ng; g++) memcpy(r[g], h[ng - 1 - g], n * 4);   /* the replicas in reverse order */
        unet_gn_ema_mean_host(h, ng, n); unet_gn_ema_mean_host(r, ng, n);
        size_t bad = 0;
        for (size_t i = 0; i < n; i++) { const float m = (float)(ref[i] / ng);
            for (int g = 0; g < ng; g++) bad += memcmp(&h[g][i], &m, 4) != 0 || memcmp(&r[g][i], &m, 4) != 0; }
        CHECK(bad == 0, "mean of %d replicas: %zu entries not the rounded double mean", ng, bad);
        for (int g = 0; g < ng; g++) { free(h[g]); free(r[g]); } free(ref);
    }
    { float a[3] = {1, 2, 3}, *h[1] = {a}; unet_gn_ema_mean_host(h, 1, 3); CHECK(a[0] == 1 && a[1] == 2 && a[2] == 3, "one replica: left as is"); }

    /* (2) the schedule */
    CHECK(!unet_gn_ema_average_due(10, 100, 7, 50, 100, 1000, 0), "before warm-up, step 7: no average");
    CHECK(unet_gn_ema_average_due(10, 100, 50, 50, 100, 1000, 0), "before warm-up, validation step: average");
    CHECK(unet_gn_ema_average_due(10, 100, 300, 70, 100, 1000, 0), "before warm-up, checkpoint step: average");
    CHECK(unet_gn_ema_average_due(10, 100, 999, 50, 100, 999, 0), "last step: average");
    CHECK(unet_gn_ema_average_due(10, 100, 7, 50, 100, 1000, 1), "stopping step: average");
    CHECK(unet_gn_ema_average_due(100, 100, 7, 50, 100, 1000, 0), "warm-up reached: every step");
    CHECK(!unet_gn_ema_average_due(10, 100, 7, 0, 0, 1000, 0), "no validation / checkpoint cadence: no division by zero, no average");

    /* (3) two (and three) replicas over a run that crosses the warm-up: sparse averaging (the schedule) against averaging every step */
    for (int ng = 2; ng <= 3; ng++) {
        const size_t n = 512; const int steps = 400, warm = 230, val_every = 50, ckpt_every = 120; const float mom = 0.01f;
        float *s[3], *e[3], *bat = malloc(n * 4);
        for (int g = 0; g < ng; g++) { s[g] = calloc(n, 4); e[g] = calloc(n, 4); }
        double worst = 0; int ndue = 0, neq = 0;
        for (int step = 1; step <= steps; step++) {
            const long nupd = step - 1;   /* (one training forward per step per replica) */
            for (int g = 0; g < ng; g++) {   /* each replica's own batch statistics (different windows) */
                for (size_t i = 0; i < n; i++) bat[i] = (i & 1) ? 0.5f + rnd() : (rnd() - .5f) * 0.2f + 0.05f * g;
                ema(s[g], bat, n, nupd, mom); ema(e[g], bat, n, nupd, mom);
            }
            unet_gn_ema_mean_host(e, ng, n);   /* the reference: averaged every step */
            if (unet_gn_ema_average_due(nupd + 1, warm, step, val_every, ckpt_every, steps, 0)) {
                unet_gn_ema_mean_host(s, ng, n); ndue++;
                int eq = 1; for (int g = 1; g < ng; g++) eq &= memcmp(s[g], s[0], n * 4) == 0; neq += eq;
                for (size_t i = 0; i < n; i++) { const double d = fabs((double)s[0][i] - e[0][i]) / (fabs((double)e[0][i]) + 1e-3); if (d > worst) worst = d; }
            }
            if (nupd + 1 >= warm) { int eq = 1; for (int g = 1; g < ng; g++) eq &= memcmp(s[g], s[0], n * 4) == 0; CHECK(eq, "%d replicas, step %d (warm-up reached): replicas differ", ng, step); }
        }
        const int want = (warm - 1) / val_every + (warm - 1) / ckpt_every - (warm - 1) / 600 + (steps - warm + 1);   /* (lcm(50, 120) = 600) */
        CHECK(ndue == want, "%d replicas: %d averaging steps, expected %d", ng, ndue, want);
        CHECK(neq == ndue, "%d replicas: after %d of %d averages the replicas differ", ng, ndue - neq, ndue);
        CHECK(worst < 2e-5, "%d replicas: the scheduled average is %.3g (relative) from the every-step average", ng, worst);
        fprintf(stderr, "%d replicas: %d averaging steps of %d, equal after each; max relative difference from every-step averaging %.2g\n", ng, ndue, steps, worst);
        for (int g = 0; g < ng; g++) { free(s[g]); free(e[g]); } free(bat);
    }
    if (fails) { fprintf(stderr, "test_gn_ema_avg: %d failures\n", fails); return 1; }
    fprintf(stderr, "test_gn_ema_avg: ok\n");
    return 0;
}
