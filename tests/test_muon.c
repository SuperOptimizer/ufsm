/* nn_muon: with p = 0, mom = 0, wd = 0 the update is -lr * scale * O where O is the orthogonalised gradient: check O O^T ~ I */
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
int main(void) {
    nn_init(0);
    int Co = 48, K = 32 * 27; size_t n = (size_t)Co * K;
    float *g = malloc(n * 4); unsigned s = 1234567; for (size_t i = 0; i < n; i++) { s = s * 1664525u + 1013904223u; g[i] = ((s >> 8) / 16777216.f - 0.5f) * (1.f + (i % 7)); }
    float *dp = nn_malloc(n * 4), *dg = nn_malloc(n * 4), *dm = nn_malloc(n * 4), *work = nn_malloc((2 * n + 2 * (size_t)Co * Co) * 4);
    nn_zero(dp, n * 4); nn_zero(dm, n * 4); nn_h2d(dg, g, n * 4);
    float lr = 1.f, beta = 0.f;
    nn_muon(dp, dg, dm, Co, K, lr, beta, 0.f, work);
    nn_sync(); const char *e = nn_check(); if (e) { printf("cuda: %s\n", e); return 1; }
    float *o = malloc(n * 4); nn_d2h(o, dp, n * 4);
    double scale = sqrt(fmax(1.0, (double)Co / K)), maxoff = 0, maxdiag = 0;
    for (int i = 0; i < Co; i++) for (int j = 0; j < Co; j++) {
        double d = 0; for (int k = 0; k < K; k++) d += (double)o[(size_t)i * K + k] * o[(size_t)j * K + k];
        d /= scale * scale;   /* O = -p / (lr scale) */
        if (i == j) maxdiag = fmax(maxdiag, fabs(d - 1.0)); else maxoff = fmax(maxoff, fabs(d));
    }
    printf("O O^T: max |diag - 1| %.3g, max |offdiag| %.3g\n", maxdiag, maxoff);
    int ok = maxdiag < 0.35 && maxoff < 0.35;   /* 5 Newton-Schulz iterations give singular values in ~[0.7, 1.2] by design */
    printf(ok ? "muon ok\n" : "muon FAIL\n");
    return !ok;
}
