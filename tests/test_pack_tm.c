/* train --pack-targets: the packed target / mask code (sample.h SAMPLE_TM_*, sample_pack_tm) unpacked on the device (nn_unpack_tm)
   must equal the target and mask bytes it replaces. Part 1 (always): every code over 1-3 samples per batch, on both streams, and the
   refusal of soft targets / mask weights. Part 2 (with a sources.json of side-encoded targets): deterministic windows drawn with and
   without packing (same seed, so the same windows) compared byte for byte after the device unpack.
   Usage: test_pack_tm [sources.json P nwindows level] */
#include "nn.h"
#include "sample.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rs = 0x9e3779b97f4a7c15ull;
static uint64_t rn(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return rs; }

static int unit(void) {
    int fails = 0;
    for (int nb = 1; nb <= 3; nb++)
        for (int stream = 0; stream < 2; stream++) {
            const size_t p3 = (size_t)12 * 20 * 20;   /* (a multiple of 4, not of 32: the tail of the last block) */
            uint8_t *t = malloc((size_t)nb * 2 * p3), *m = malloc((size_t)nb * p3), *code = malloc((size_t)nb * p3);
            uint8_t *gt = malloc((size_t)nb * 2 * p3), *gm = malloc((size_t)nb * p3);
            for (int n = 0; n < nb; n++)
                for (size_t k = 0; k < p3; k++) {
                    const unsigned c = k < 8 ? (unsigned)k : (unsigned)(rn() % 8);   /* every code first */
                    t[(2 * n) * p3 + k] = c & 1 ? 255 : 0; t[(2 * n + 1) * p3 + k] = c & 2 ? 255 : 0; m[n * p3 + k] = c >> 2 & 1;
                }
            size_t bad = 0;
            for (int n = 0; n < nb; n++)
                if (sample_pack_tm(t + (2 * n) * p3, t + (2 * n + 1) * p3, m + n * p3, p3, code + n * p3, &bad)) { fprintf(stderr, "pack refused a hard batch at %zu\n", bad); fails++; }
            uint8_t *dt = nn_malloc((size_t)nb * 2 * p3), *dm = nn_malloc((size_t)nb * p3);
            memset(gt, 0x5a, (size_t)nb * 2 * p3); nn_h2d(dt, gt, (size_t)nb * 2 * p3);   /* (stale bytes: the unpack writes every one) */
            if (stream) { nn_h2d_copy_stream(dm, code, (size_t)nb * p3); nn_unpack_tm(dt, dm, nb, p3, 1); void *ev = nn_event_create(); nn_event_record(ev, 1); nn_event_sync(ev); }
            else { nn_h2d(dm, code, (size_t)nb * p3); nn_unpack_tm(dt, dm, nb, p3, 0); }
            nn_d2h(gt, dt, (size_t)nb * 2 * p3); nn_d2h(gm, dm, (size_t)nb * p3);
            if (memcmp(gt, t, (size_t)nb * 2 * p3) || memcmp(gm, m, (size_t)nb * p3)) { fprintf(stderr, "unpack mismatch: %d samples, stream %d\n", nb, stream); fails++; }
            /* a soft target value and a mask weight of 2 are refused, at their index */
            t[p3 + 77] = 200;
            if (!sample_pack_tm(t, t + p3, m, p3, code, &bad) || bad != 77) { fprintf(stderr, "soft target not refused (%zu)\n", bad); fails++; }
            t[p3 + 77] = 0; m[91] = 2;
            if (!sample_pack_tm(t, t + p3, m, p3, code, &bad) || bad != 91) { fprintf(stderr, "mask weight not refused (%zu)\n", bad); fails++; }
            nn_free(dt); nn_free(dm); free(t); free(m); free(code); free(gt); free(gm);
        }
    printf("pack / unpack codes: %s\n", fails ? "FAIL" : "ok");
    return fails;
}

static int windows(const char *path, int P, int nw, int level) {
    sources *S = sources_load(path);
    if (!S) { fprintf(stderr, "cannot load %s\n", path); return 1; }
    sample_cfg c = sample_cfg_default();
    c.P = P; c.B = 1; c.nworkers = 3; c.nbuf = 3; c.seed = 4242; c.deterministic = 1; c.augment = 1; c.xfmt = 1; c.holdout = 0;
    memset(c.level_p, 0, sizeof c.level_p); c.level_p[level] = 1;
    sampler *sa = sampler_start(S, &c);
    c.pack_tm = 1;
    sampler *sb = sampler_start(S, &c);
    if (!sa || !sb) { fprintf(stderr, "cannot start the samplers\n"); return 1; }
    const size_t p3 = (size_t)P * P * P;
    uint8_t *dt = nn_malloc(2 * p3), *dm = nn_malloc(p3), *gt = malloc(2 * p3), *gm = malloc(p3);
    int fails = 0; size_t lab = 0;
    for (int i = 0; i < nw; i++) {
        batch *a = sampler_next(sa), *b = sampler_next(sb);
        if (!a || !b) { fprintf(stderr, "sampler stopped at window %d\n", i); fails++; break; }
        if (b->t || b->m || !b->tm || a->tm) { fprintf(stderr, "batch layout: t %p m %p tm %p\n", (void *)b->t, (void *)b->m, (void *)b->tm); fails++; }
        if (memcmp(a->x16, b->x16, p3 * 2) || memcmp(a->w, b->w, NCH) || memcmp(a->corner, b->corner, sizeof *a->corner)) { fprintf(stderr, "window %d: not the same window\n", i); fails++; }
        nn_h2d(dm, b->tm, p3); nn_unpack_tm(dt, dm, 1, p3, 0); nn_d2h(gt, dt, 2 * p3); nn_d2h(gm, dm, p3);
        if (memcmp(gt, a->t, 2 * p3) || memcmp(gm, a->m, p3)) { fprintf(stderr, "window %d: unpacked targets differ\n", i); fails++; }
        for (size_t k = 0; k < p3; k++) lab += a->m[k] && (a->t[k] | a->t[p3 + k]);
        sampler_release(sa, a); sampler_release(sb, b);
    }
    sampler_stop(sa); sampler_stop(sb);
    printf("packed windows (%s, P %d, level %d): %d windows, %zu labelled papyrus voxels: %s\n", path, P, level, nw, lab, fails ? "FAIL" : "ok");
    nn_free(dt); nn_free(dm); free(gt); free(gm);
    return fails;
}

int main(int argc, char **argv) {
    nn_init(0);
    int fails = unit();
    if (argc >= 2) fails += windows(argv[1], argc > 2 ? atoi(argv[2]) : 128, argc > 3 ? atoi(argv[3]) : 6, argc > 4 ? atoi(argv[4]) : 1);
    return fails != 0;
}
