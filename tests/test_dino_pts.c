/* dino-train GT SupCon points (src/dino_pts.h): the papyrus gate keeps only labelled voxels with raw CT above the level, the reservoir
   holds min(seen, cap) of them, each candidate is equally likely, and with no gate the points (and rng sequence) are the old sampler's. */
#include "../src/dino_pts.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint64_t xs(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)
/* the sampler before the gate (src/dino.c 10-25), for the no-gate equivalence */
static long old_res(const uint8_t *codes, int P, int m, int cap, uint64_t *rs, int *sel, int *code) {
    long seen = 0;
    for (int z = m; z < P - m; z++) for (int y = m; y < P - m; y++) for (int x = m; x < P - m; x++) {
        const int c = codes[((size_t)z * P + y) * P + x]; if (!c || c == 255) continue;
        long k = seen++;
        if (k >= cap) { k = (long)(xs(rs) % (uint64_t)seen); if (k >= cap) continue; }
        sel[k] = (z * P + y) * P + x; code[k] = c;
    }
    return seen;
}
int main(void) {
    enum { P = 40, M = 4, CAP = 300 };
    const size_t n = (size_t)P * P * P; uint8_t *codes = malloc(n), *ct = malloc(n); uint64_t g = 99;
    for (size_t i = 0; i < n; i++) {   /* sheets of papyrus (CT 150) with voids (CT 40) inside the labelled band; code 255 / 0 elsewhere */
        const int z = (int)(i / (P * P)); codes[i] = (z % 8) < 5 ? (uint8_t)(1 + z / 8) : (z % 8) == 5 ? 255 : 0;
        ct[i] = (xs(&g) % 3) ? 150 : 40;
    }
    long npap = 0, nlab = 0;
    for (int z = M; z < P - M; z++) for (int y = M; y < P - M; y++) for (int x = M; x < P - M; x++) {
        const size_t i = ((size_t)z * P + y) * P + x; const int c = codes[i]; if (!c || c == 255) continue; nlab++; npap += ct[i] > 100; }
    int sel[CAP], code[CAP], sel2[CAP], code2[CAP];
    uint64_t r1 = 7, r2 = 7;
    long s1 = dn_gt_reservoir(codes, ct, P, M, -1, CAP, &r1, xs, sel, code), s2 = old_res(codes, P, M, CAP, &r2, sel2, code2);
    CHECK(s1 == nlab && s2 == nlab, "no gate: seen %ld / %ld, %ld labelled", s1, s2, nlab);
    CHECK(!memcmp(sel, sel2, sizeof sel) && !memcmp(code, code2, sizeof code) && r1 == r2, "no gate: points differ from the old sampler");
    long hits = 0, bad = 0, wrong = 0; double frac_void_old = 0;
    for (int k = 0; k < CAP; k++) frac_void_old += ct[sel2[k]] <= 100;
    for (int rep = 0; rep < 200; rep++) {
        uint64_t r = 1000 + rep; long s = dn_gt_reservoir(codes, ct, P, M, 100, CAP, &r, xs, sel, code);
        CHECK(s == npap, "gate: seen %ld, %ld papyrus labelled", s, npap);
        for (int k = 0; k < CAP; k++) { bad += ct[sel[k]] <= 100; wrong += codes[sel[k]] != code[k]; const int z = sel[k] / (P * P), y = sel[k] / P % P, x = sel[k] % P;
            bad += z < M || y < M || x < M || z >= P - M || y >= P - M || x >= P - M; hits += sel[k] < (int)(n / 2); }
    }
    printf("labelled %ld, on papyrus %ld; ungated sample %.1f%% in voids, gated: %ld void / outside points, %ld code mismatches; first half %.3f\n",
           nlab, npap, 100 * frac_void_old / CAP, bad, wrong, (double)hits / (200.0 * CAP));
    CHECK(frac_void_old > 0.2 * CAP, "the fixture has too few void labels");
    CHECK(bad == 0 && wrong == 0, "gated points: %ld in voids / outside, %ld wrong codes", bad, wrong);
    long half = 0; for (int z = M; z < P - M; z++) for (int y = M; y < P - M; y++) for (int x = M; x < P - M; x++) { const size_t i = ((size_t)z * P + y) * P + x;
        half += codes[i] && codes[i] != 255 && ct[i] > 100 && i < n / 2; }
    const double want = (double)half / npap, got = (double)hits / (200.0 * CAP);
    CHECK(got > want - 0.02 && got < want + 0.02, "reservoir not uniform: first-half share %.3f, expected %.3f", got, want);
    { uint64_t r = 5; long s = dn_gt_reservoir(codes, ct, P, M, 255, CAP, &r, xs, sel, code); CHECK(s == 0, "gate 255: %ld points", s); }
    free(codes); free(ct);
    if (fails) { fprintf(stderr, "test_dino_pts: %d failure(s)\n", fails); return 1; }
    printf("test_dino_pts: ok\n"); return 0;
}
