/* --win-weight (audit 2026-11-14 train #4): nn_loss_stats_fin_w / nn_wind4_fin_w scale a head's gradient multiplier by min(1, nm / ref),
   ref the slot's running median of nm (the geometric mean of the first 16 labelled windows, then sign steps of 2^0.05). Checks, over a
   sequence of windows with labelled counts from a few voxels to the whole window:
     - slot -1 leaves the scratch bit for bit as nn_loss_stats_fin (the default path);
     - slot k: fin[2 C + 1] = 1 / active x the host reference tracker's weight, the logged loss values fin[0 .. 2 C] unchanged;
     - nn_win_weight_take reports the interval's windows, mean weight and the count below 0.1 ref, then resets */
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static double href[2], hcnt;   /* the host tracker: log2 ref, windows */
static float hw(float nm) {
    const double l = log2((double)nm);
    double w = hcnt > 0 ? fmin(1.0, exp2(l - href[0])) : 1.0;
    if (hcnt < 16) href[0] = hcnt > 0 ? (href[0] * hcnt + l) / (hcnt + 1) : l; else href[0] += l > href[0] ? 0.05 : l < href[0] ? -0.05 : 0;
    hcnt += 1;
    return (float)w;
}
int main(void) {
    nn_init(0);
    const int C = 2, D = 8, H = 16, W = 16; const size_t S = (size_t)D * H * W;
    shape5 s = {1, C, D, H, W};
    float *lg = malloc(S * C * 4); uint8_t *t = malloc(S * C), *m = malloc(S * C), wch[2] = {1, 1};
    float *dl = nn_malloc(S * C * 4); uint8_t *dt = nn_malloc(S * C), *dm = nn_malloc(S * C), *dw = nn_malloc(2);
    double *ds = nn_malloc(5 * C * sizeof(double));
    const size_t sb = nn_loss_scratch(s); float *sc0 = nn_malloc(sb), *sc1 = nn_malloc(sb);
    float *h0 = malloc(sb), *h1 = malloc(sb);
    nn_h2d(dw, wch, 2);
    int bad = 0, nlow = 0; double wsum = 0; uint64_t r = 88172645463325252ull;
    const int NWIN = 60;
    for (int win = 0; win < NWIN; win++) {
        /* labelled voxels: mostly ~1/2 of the window, every 7th window a sliver (4 .. 40 voxels) */
        const size_t nl = win % 7 == 3 ? 4 + (size_t)(win % 37) : S / 2 + (size_t)(win * 13 % 200);
        for (size_t i = 0; i < S * C; i++) {
            r ^= r << 13; r ^= r >> 7; r ^= r << 17;
            lg[i] = (float)((r >> 40) % 2000) / 250.f - 4.f; t[i] = (uint8_t)((r >> 20) & 255); m[i] = (i % S) < nl;
        }
        nn_h2d(dl, lg, S * C * 4); nn_h2d(dt, t, S * C); nn_h2d(dm, m, S * C);
        for (int pass = 0; pass < 3; pass++) {   /* 0: nn_loss_stats_fin, 1: slot -1, 2: slot 3 */
            nn_zero(ds, 5 * C * sizeof(double));
            nn_loss_stats_acc(dl, dt, dm, dw, s, ds);
            if (pass == 0) nn_loss_stats_fin(ds, dw, s, sc0); else nn_loss_stats_fin_w(ds, dw, s, sc1, pass == 1 ? -1 : 3);
            nn_sync();
            if (pass == 0) { nn_d2h(h0, sc0, sb); continue; }
            nn_d2h(h1, sc1, sb);
            const float *f0 = h0 + 5 * C, *f1 = h1 + 5 * C;
            if (pass == 1) { if (memcmp(h0, h1, sb)) { printf("  window %d: slot -1 differs from nn_loss_stats_fin  FAIL\n", win); bad++; } continue; }
            const float w = hw((float)nl), want = f0[2 * C + 1] * w;
            wsum += w; if (hcnt > 1 && w < 0.1f) nlow++;
            if (memcmp(h0, h1, (5 * C + 2 * C + 1) * sizeof(float))) { printf("  window %d: the statistics / logged losses changed  FAIL\n", win); bad++; }
            if (!(fabsf(f1[2 * C + 1] - want) <= 1e-4f * want))   /* (the device tracker in fp32, the host one in fp64) */ { printf("  window %d (nm %zu): multiplier %.9g, want %.9g (w %.6f)  FAIL\n", win, nl, f1[2 * C + 1], want, w); bad++; }
        }
    }
    float tk[5 * 4]; nn_win_weight_take(tk, 4);
    const float *q = tk + 5 * 3;
    printf("  %d windows: mean weight %.4f (host %.4f), below 0.1 ref %.0f (host %d), ref %.1f (host %.1f)\n", NWIN, q[1], wsum / NWIN, q[2], nlow, q[4], exp2(href[0]));
    if (!(q[0] == NWIN && fabsf(q[1] - (float)(wsum / NWIN)) < 1e-4f && (int)q[2] == nlow && fabsf(q[4] - (float)exp2(href[0])) < 1e-3f * q[4])) { printf("  take: wrong interval summary  FAIL\n"); bad++; }
    if (!(nlow >= 5)) { printf("  the sliver windows should weigh < 0.1 (%d)  FAIL\n", nlow); bad++; }
    nn_win_weight_take(tk, 4);
    if (tk[15] != 0.f) { printf("  take did not reset the interval  FAIL\n"); bad++; }
    const char *e = nn_check(); if (e) { printf("cuda: %s  FAIL\n", e); bad++; }
    printf(bad ? "test_win_weight: %d FAILED\n" : "test_win_weight: ok\n", bad);
    return bad != 0;
}
