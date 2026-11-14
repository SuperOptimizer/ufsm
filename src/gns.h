/* Gradient noise scale (critical batch size), McCandlish et al. 2018 "An Empirical Model of Large-Batch Training", appendix A.1.
   Two squared gradient norms of the same parameters at two batch sizes b < B (in windows) give unbiased estimates of the true gradient's
   squared norm |G|^2 and of the per-window noise S = tr(Sigma), from E|G_b|^2 = |G|^2 + S / b:
     |G|^2 ~ (B |G_B|^2 - b |G_b|^2) / (B - b),   S ~ (|G_b|^2 - |G_B|^2) / (1/b - 1/B).
   Both are noisy per update (either can go negative), so EMAs of each are kept and B_noise = EMA(S) / EMA(|G|^2) (the ratio of the
   means, not the mean of the ratios: the paper's choice; the ratio of two noisy unbiased estimates is biased and heavy-tailed). The EMAs
   start at 0 and are bias-corrected (the correction cancels in the ratio but not in the logged |G|^2 and S). Header-only: the CPU test
   (tests/test_gns.c) and the trainer share it. */
#pragma once
#include <math.h>

typedef struct { double beta, g2, s, w; long n; } gns_ema;   /* beta: EMA decay per update; w = 1 - beta^n (the bias correction) */

static inline gns_ema gns_make(double beta) { return (gns_ema){.beta = beta}; }

/* one update's pair: n2b = |G_b|^2 at b windows, n2B = |G_B|^2 at B windows (b < B). Returns 0 when the pair is unusable (b >= B or a
   non-finite norm: the estimate is left as it was) */
static inline int gns_add(gns_ema *e, double b, double n2b, double B, double n2B) {
    if (!(b > 0 && B > b) || !isfinite(n2b) || !isfinite(n2B)) return 0;
    const double g2 = (B * n2B - b * n2b) / (B - b), s = (n2b - n2B) / (1.0 / b - 1.0 / B);
    e->g2 = e->beta * e->g2 + (1 - e->beta) * g2; e->s = e->beta * e->s + (1 - e->beta) * s;
    e->w = e->beta * e->w + (1 - e->beta); e->n++;
    return 1;
}
static inline double gns_g2(const gns_ema *e) { return e->w > 0 ? e->g2 / e->w : NAN; }
static inline double gns_s(const gns_ema *e) { return e->w > 0 ? e->s / e->w : NAN; }
/* B_noise in windows; NAN before the first pair or while EMA(|G|^2) <= 0 (noise-dominated: the gradient's mean is not resolved yet) */
static inline double gns_bnoise(const gns_ema *e) { const double g = gns_g2(e); return e->n && g > 0 ? gns_s(e) / g : NAN; }
