/* src/edt.c against brute force: random sparse masks (several shapes, 1 and 8 threads), a single feature, an empty mask;
   then the speed on a 256^3 mask. */
#include "edt.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

static void brute(const uint8_t *m, const int64_t n[3], float *d) {
    const size_t N = (size_t)n[0] * n[1] * n[2];
    for (size_t i = 0; i < N; i++) {
        const int64_t z = i / (n[1] * n[2]), y = (i / n[2]) % n[1], x = i % n[2]; float best = EDT_INF;
        for (size_t j = 0; j < N; j++) if (m[j]) {
            const int64_t a = j / (n[1] * n[2]) - z, b = (j / n[2]) % n[1] - y, c = j % n[2] - x;
            const float s = (float)(a * a + b * b + c * c); if (s < best) best = s;
        }
        d[i] = best;
    }
}

int main(void) {
    srand(7);
    const int64_t shapes[4][3] = {{20, 17, 23}, {1, 40, 33}, {31, 1, 9}, {12, 12, 12}};
    for (int s = 0; s < 4; s++) for (int pass = 0; pass < 3; pass++) {
        const int64_t *n = shapes[s]; const size_t N = (size_t)n[0] * n[1] * n[2];
        uint8_t *m = calloc(N, 1); float *a = malloc(N * 4), *b = malloc(N * 4);
        if (pass == 0) for (size_t i = 0; i < N; i++) m[i] = rand() % 50 == 0;   /* sparse */
        else if (pass == 1) m[N / 3] = 1;                                         /* one feature */
        brute(m, n, a);
        for (int t = 1; t <= 8; t += 7) {
            CHECK(!edt_sq(m, n, b, t), "edt_sq failed");
            size_t bad = 0; for (size_t i = 0; i < N; i++) bad += a[i] != b[i];
            CHECK(bad == 0, "shape %lld x %lld x %lld, mask %d, %d threads: %zu of %zu voxels differ", (long long)n[0], (long long)n[1], (long long)n[2], pass, t, bad, N);
        }
        free(m); free(a); free(b);
    }
    const int64_t n[3] = {256, 256, 256}; const size_t N = (size_t)n[0] * n[1] * n[2];
    uint8_t *m = calloc(N, 1); float *d = malloc(N * 4);
    for (int k = 0; k < 2000; k++) m[(size_t)rand() * rand() % N] = 1;
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    CHECK(!edt_sq(m, n, d, 16), "edt_sq 256^3 failed");
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("edt 256^3, 16 threads: %.3f s\n", (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec));
    free(m); free(d);
    if (fails) { fprintf(stderr, "test_edt: %d failure(s)\n", fails); return 1; }
    printf("test_edt: OK\n");
    return 0;
}
