/* Exercise the real sampler without initializing CUDA. Outputs three orthogonal
   CT | soft-target | validity slices and counts for each drawn training cube. */
#include "sample.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void *nn_host_alloc(size_t n) { return malloc(n); }
void nn_host_free(void *p) { free(p); }

static float input16(const uint16_t *p) {
    _Float16 h; memcpy(&h, p, sizeof h); return (float)h;
}

static int slices(const char *dir, int bi, const batch *b, int P) {
    uint8_t *img = malloc((size_t)3 * P * P);
    if (!img) return 1;
    for (int axis = 0; axis < 3; axis++) {
        for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
            int zc = axis == 0 ? P / 2 : y;
            int yc = axis == 0 ? y : axis == 1 ? P / 2 : x;
            int xc = axis == 2 ? P / 2 : x;
            size_t k = ((size_t)zc * P + yc) * P + xc;
            float ct = input16(b->x16 + k);
            img[(size_t)y * 3 * P + x] = (uint8_t)fminf(255, fmaxf(0, 128 + 40 * ct));
            img[(size_t)y * 3 * P + P + x] = b->t[k];
            img[(size_t)y * 3 * P + 2 * P + x] = b->m[k] ? 255 : 0;
        }
        char path[1400]; snprintf(path, sizeof path, "%s/sample-%02d-axis-%d.pgm", dir, bi, axis);
        FILE *f = fopen(path, "wb");
        if (!f) { free(img); return 1; }
        fprintf(f, "P5\n%d %d\n255\n", 3 * P, P);
        int ok = fwrite(img, 1, (size_t)3 * P * P, f) == (size_t)3 * P * P;
        ok &= fclose(f) == 0;
        if (!ok) { free(img); return 1; }
    }
    free(img); return 0;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        fprintf(stderr, "usage: check_surface_samples <sources.json> <output-dir> [P=704] [batches=3]\n");
        return 2;
    }
    int P = argc > 3 ? atoi(argv[3]) : 704, nb = argc > 4 ? atoi(argv[4]) : 3;
    if (P <= 0 || P > 1024 || nb <= 0) return 2;
    sources *S = sources_load(argv[1]);
    if (!S || S->n != 1) { fprintf(stderr, "expected exactly one source\n"); sources_free(S); return 1; }
    sample_cfg c = sample_cfg_default();
    c.P = P; c.B = 1; c.nworkers = 1; c.nbuf = 1; c.xfmt = 1;
    c.seed = 704; c.deterministic = 1; c.augment = 0; c.soft = 3;
    memset(c.level_p, 0, sizeof c.level_p); c.level_p[0] = 1;
    sampler *sp = sampler_start(S, &c);
    if (!sp) { sources_free(S); return 1; }
    char path[1400]; snprintf(path, sizeof path, "%s/samples.json.tmp", argv[2]);
    FILE *f = fopen(path, "w");
    if (!f) { sampler_stop(sp); sources_free(S); return 1; }
    fprintf(f, "{\"P\":%d,\"B\":1,\"level\":0,\"soft\":3,\"seed\":704,\"samples\":[\n", P);
    size_t n = (size_t)P * P * P; int failed = 0;
    for (int bi = 0; bi < nb; bi++) {
        batch *b = sampler_next(sp);
        if (!b) { failed = 1; break; }
        size_t valid = 0, fg = 0, bg = 0, soft = 0, nonfinite = 0;
        for (size_t k = 0; k < n; k++) if (b->m[k]) {
            valid++; fg += b->t[k] == 255; bg += b->t[k] < 128;
            soft += b->t[k] > 0 && b->t[k] < 255;
        }
        for (size_t k = 0; k < 4 * n; k++) nonfinite += !isfinite(input16(b->x16 + k));
        failed |= b->src[0] != 0 || b->level[0] != 0 || b->w[0] != 1 || b->w[1] != 0;
        failed |= valid == 0 || fg == 0 || bg == 0 || soft == 0 || nonfinite != 0;
        failed |= slices(argv[2], bi, b, P);
        fprintf(f, "%s{\"corner_zyx\":[%lld,%lld,%lld],\"voxels\":%zu,\"valid\":%zu,"
                   "\"surface\":%zu,\"target_below_half\":%zu,\"soft\":%zu,\"nonfinite_input\":%zu}",
                bi ? ",\n" : "", (long long)b->corner[0][0], (long long)b->corner[0][1],
                (long long)b->corner[0][2], n, valid, fg, bg, soft, nonfinite);
        fprintf(stderr, "sample %d P=%d: valid %zu surface %zu background %zu soft %zu nonfinite %zu\n",
                bi, P, valid, fg, bg, soft, nonfinite);
        sampler_release(sp, b);
        if (failed) break;
    }
    failed |= sampler_failed(sp);
    fprintf(f, "\n],\"passed\":%s}\n", failed ? "false" : "true");
    failed |= fclose(f) != 0;
    sampler_stop(sp); sources_free(S);
    char final[1400]; snprintf(final, sizeof final, "%s/samples.json", argv[2]);
    if (rename(path, final)) failed = 1;
    return failed != 0;
}
