/* src/pairacc.c: pair winding accuracy on synthetic boxes (no slip = 1; a planar one-turn slip at the box centre costs the
   straddling pairs ~ D / (2 L) of uniform-direction pairs; the gauge does not matter; subsampling keeps a deterministic hash set) */
#include "../src/pairacc.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int failed;
static void check(const char *n, int ok) { printf("pairacc %-52s %s\n", n, ok ? "ok" : "FAIL"); failed += !ok; }
static void fill(pa_buf *b, int L, int slip, double gauge) {
    for (int z = 0; z < L; z++) for (int y = 0; y < L; y++) for (int x = 0; x < L; x++) {
        const int q = x / 7;   /* "sheets" 7 voxels apart along x */
        double d = gauge + (slip && x >= L / 2 ? 1.0 : 0.0) + 0.05 * sin(x + 3 * y); d = fmod(d + 14, 14);
        pa_add(b, (float)z, (float)y, (float)x, (float)d, (uint8_t)q, ((uint64_t)z << 42) ^ ((uint64_t)y << 21) ^ (uint64_t)x);
    }
}
int main(void) {
    const int L = 120; const double D[2] = {6, 20}; double a[2], ac[2]; size_t n[2], nc[2];
    pa_buf b; pa_init(&b, 1 << 18); fill(&b, L, 0, 3.3);
    check("subsample bounded by the buffer", b.n <= (1u << 18) && b.n > (1u << 16));
    pa_score(&b, D, 2, 100000, 7, a, ac, n, nc);
    check("no slip: every pair right", a[0] == 1 && a[1] == 1 && ac[1] == 1 && n[1] > 50000 && nc[1] > 10000);
    pa_free(&b); pa_init(&b, 1 << 18); fill(&b, L, 1, 13.9);
    pa_score(&b, D, 2, 200000, 7, a, ac, n, nc);
    const double e0 = 1 - D[0] / (2.0 * L), e1 = 1 - D[1] / (2.0 * L);
    printf("  slip: D %g acc %.4f (expect ~%.4f), D %g acc %.4f (expect ~%.4f)\n", D[0], a[0], e0, D[1], a[1], e1);
    check("planar slip costs the straddling pairs", fabs(a[0] - e0) < 0.01 && fabs(a[1] - e1) < 0.015 && a[1] < a[0]);
    pa_buf c, s1, s2; pa_init(&c, 1 << 18); pa_init(&s1, 1 << 17); pa_init(&s2, 1 << 17);
    for (int z = 0; z < L; z++) for (int y = 0; y < L; y++) for (int x = 0; x < L; x++)
        pa_add(z < L / 2 ? &s1 : &s2, (float)z, (float)y, (float)x, 0, 0, ((uint64_t)z << 42) ^ ((uint64_t)y << 21) ^ (uint64_t)x);
    pa_merge(&c, &s1); pa_merge(&c, &s2);
    int sub = 1; for (size_t i = 0; i < c.n; i++) sub &= c.r[i].h < c.T;
    check("merge keeps one threshold", sub && c.n > 0 && c.T <= s1.T && c.T <= s2.T);
    pa_free(&b); pa_free(&c); pa_free(&s1); pa_free(&s2);
    return failed != 0;
}
