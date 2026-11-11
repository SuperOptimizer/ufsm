/* nn_gram16 (train --grad-cos) against a double host reference: K planes of fp16 / bf16 with a row stride, two calls accumulating, the
   upper triangle only. Usage: test_gram */
#include "nn.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static float h2f(uint16_t h, int bf) {
    if (bf) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023; float v = e ? ldexpf(1.f + m / 1024.f, (int)e - 15) : ldexpf(m / 1024.f, -14);
    return s ? -v : v;
}
static uint16_t f2h(float f, int bf) {   /* (round toward zero is enough: the reference reads the stored bits back) */
    uint32_t u; memcpy(&u, &f, 4);
    if (bf) return (uint16_t)(u >> 16);
    const uint32_t s = u >> 31; int e = (int)((u >> 23) & 255) - 127 + 15; const uint32_t m = (u >> 13) & 1023;
    return (uint16_t)(s << 15 | (e <= 0 ? 0 : e >= 31 ? 31u << 10 : (uint32_t)e << 10 | m));
}
int main(void) {
    nn_init(0); int fails = 0;
    for (int bf = 0; bf < 2; bf++) for (int K = 1; K <= 13; K += 6) {
        const size_t N = 100003, ld = N + 17; uint16_t *x = malloc(K * ld * 2); srand(7 + K);
        for (size_t i = 0; i < K * ld; i++) x[i] = f2h(((float)rand() / RAND_MAX - 0.5f) * 0.02f, bf);
        double *S = nn_malloc((size_t)K * K * 8), *Sh = malloc((size_t)K * K * 8); nn_zero(S, (size_t)K * K * 8);
        void *xd = nn_malloc(K * ld * 2); nn_h2d(xd, x, K * ld * 2);
        nn_gram16(xd, bf, K, N / 2, ld, S); nn_gram16((uint16_t *)xd + N / 2, bf, K, N - N / 2, ld, S);
        nn_sync(); nn_d2h(Sh, S, (size_t)K * K * 8);
        double worst = 0;
        for (int i = 0; i < K; i++) for (int j = 0; j < K; j++) {
            if (j < i) { if (Sh[i * K + j] != 0) { fprintf(stderr, "FAIL: lower triangle written\n"); fails++; } continue; }
            double r = 0, ni = 0, nj = 0;
            for (size_t n = 0; n < N; n++) { const double a = h2f(x[i * ld + n], bf), b = h2f(x[j * ld + n], bf); r += a * b; ni += a * a; nj += b * b; }
            const double e = fabs(Sh[i * K + j] - r) / sqrt(ni * nj); worst = e > worst ? e : worst;
        }
        printf("gram %s K %d: worst error / |x_i||x_j| %.2e\n", bf ? "bf16" : "fp16", K, worst);
        if (!(worst < 1e-5)) { fprintf(stderr, "FAIL: gram %s K %d\n", bf ? "bf16" : "fp16", K); fails++; }
        nn_free(S); nn_free(xd); free(Sh); free(x);
    }
    if (fails) return 1;
    printf("test_gram: ok\n"); return 0;
}
