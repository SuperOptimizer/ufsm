/* --cond condition draws (src/sample.c cond_draw via sample_cond_draw): one integer gauge per sample shifts the winding targets,
   the condition channels hold cos / sin of 2 pi q' / 14 (q' the shifted winding, plus a small smooth noise) and a mask exactly where
   the pattern covers a known voxel, zeros elsewhere; about one draw in ten has no condition and no targets; patterns are a small
   seed cube or slabs touching a window face. */
#include "../src/sample.c"   /* (like test_sampler_safety: the sampler compiled in, no CUDA objects) */
void *nn_host_alloc(size_t n) { return malloc(n); }   /* (pinned memory stand-ins, as in test_sampler_safety) */
void nn_host_free(void *p) { free(p); }
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

int main(void) {
    enum { P = 32, N = 400 };
    const size_t p2 = (size_t)P * P, p3 = p2 * P;
    uint8_t *w = malloc(p3), *wo = malloc(p3); float *X = malloc(7 * p3 * sizeof *X);
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {   /* a winding ramp crossing the 14-turn wrap, some unknown */
        const size_t k = ((size_t)z * P + y) * P + x;
        w[k] = k % 7 == 3 ? 255 : (uint8_t)((5 * x + 3 * y + z + 200) % 252);
    }
    int none = 0, cube = 0, slab = 0, bad_gauge = 0, bad_mask = 0, bad_zero = 0, bad_shape = 0; double emax = 0;
    for (int s = 0; s < N; s++) {
        memset(X, 0x7f, 7 * p3 * sizeof *X);
        sample_cond_draw(1000 + s, P, w, wo, X, 0);
        const float *C = X + 4 * p3, *S = X + 5 * p3, *M = X + 6 * p3;
        size_t nm = 0, known = 0; for (size_t k = 0; k < p3; k++) { nm += M[k] == 1.f; known += wo[k] != 255; }
        if (!known) {   /* no condition: no targets, all condition inputs zero */
            none++;
            for (size_t k = 0; k < p3; k++) if (C[k] != 0.f || S[k] != 0.f || M[k] != 0.f) { bad_zero++; break; }
            continue;
        }
        int g = -1;   /* one gauge for the whole sample: wo = w + 18 g (mod 252) where known, unknown where w is */
        for (size_t k = 0; k < p3; k++) {
            if ((w[k] == 255) != (wo[k] == 255)) { bad_gauge++; break; }
            if (w[k] == 255) continue;
            const int d = ((int)wo[k] - (int)w[k] + 252) % 252;
            if (d % 18 || (g >= 0 && d / 18 != g)) { bad_gauge++; break; }
            g = d / 18;
        }
        int lo[3] = {P, P, P}, hi[3] = {-1, -1, -1};
        for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
            const size_t k = ((size_t)z * P + y) * P + x;
            if (M[k] == 0.f) { if (C[k] != 0.f || S[k] != 0.f) bad_zero++; continue; }
            if (M[k] != 1.f || wo[k] == 255) { bad_mask++; continue; }
            const int c[3] = {z, y, x}; for (int d = 0; d < 3; d++) { if (c[d] < lo[d]) lo[d] = c[d]; if (c[d] > hi[d]) hi[d] = c[d]; }
            double e = atan2(S[k], C[k]) / (2 * M_PI) * 14 - wo[k] / 18.0;   /* turns */
            e -= 14 * floor(e / 14 + 0.5); e = fabs(e); if (e > emax) emax = e;
            if (fabs(hypot(C[k], S[k]) - 1) > 1e-5) bad_mask++;
        }
        if (!nm) continue;   /* (a pattern fully dropped out) */
        int face = 0; for (int d = 0; d < 3; d++) face |= lo[d] == 0 || hi[d] == P - 1;
        const int ext = (hi[0] - lo[0] + 1) > (hi[1] - lo[1] + 1) ? hi[0] - lo[0] + 1 : hi[1] - lo[1] + 1, ext2 = ext > hi[2] - lo[2] + 1 ? ext : hi[2] - lo[2] + 1;
        if (ext2 <= 12 && nm <= 12 * 12 * 12) cube++;
        else if (face) slab++;
        else bad_shape++;
    }
    {   /* --cond-loss outside: the same draws, targets only where the condition mask is 0 (the inputs unchanged) */
        int bad_out = 0, kept = 0;
        float *X2 = malloc(7 * p3 * sizeof *X2); uint8_t *wo2 = malloc(p3);
        for (int s = 0; s < 40; s++) {
            sample_cond_draw(1000 + s, P, w, wo, X, 0); sample_cond_draw(1000 + s, P, w, wo2, X2, 1);
            for (size_t k = 0; k < p3; k++) {
                const int masked = X2[6 * p3 + k] == 1.f;
                if (X2[4 * p3 + k] != X[4 * p3 + k] || X2[6 * p3 + k] != X[6 * p3 + k]) bad_out++;   /* inputs identical */
                if (masked ? wo2[k] != 255 : wo2[k] != wo[k]) bad_out++;
                kept += wo2[k] != 255;
            }
        }
        CHECK(!bad_out && kept > 0, "outside-only targets: %d voxels wrong (%d kept)", bad_out, kept);
        free(X2); free(wo2);
    }
    CHECK(none >= N / 20 && none <= N / 6, "%d of %d draws without a condition (want ~10%%)", none, N);
    CHECK(cube >= N / 20 && slab >= N / 2, "patterns: %d seed cubes, %d face slabs of %d", cube, slab, N);
    CHECK(!bad_gauge, "%d draws with a mixed or non-integer gauge", bad_gauge);
    CHECK(!bad_mask && !bad_zero, "%d mask / unit-vector errors, %d nonzero inputs outside the mask", bad_mask, bad_zero);
    CHECK(!bad_shape, "%d patterns neither a seed cube nor touching a face", bad_shape);
    CHECK(emax < 0.2, "condition angle off the shifted winding by up to %.3f turn (noise 0.03 turn)", emax);
    free(w); free(wo); free(X);
    if (fails) { fprintf(stderr, "test_cond: %d failures\n", fails); return 1; }
    printf("cond draws: %d none, %d seed cubes, %d face slabs; one gauge per sample; max angle noise %.3f turn ok\n", none, cube, slab, emax);
    return 0;
}
