/* z3_read decodes only the volcomp substreams (16 z x 32 y slabs of a 128^3 chunk) under the request box (auto_submask, src/zarr3.c):
   lossy and lossless-coded arrays read through odd boxes give the same bytes as whole-chunk decoding (UFSM_Z3_AUTOSUB=0). */
#include "store.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    char dir[] = "/tmp/test_autosub_XXXXXX"; if (!mkdtemp(dir)) return 1;
    enum { N = 256 };
    uint8_t *d = malloc((size_t)N * N * N);
    for (size_t i = 0; i < (size_t)N * N * N; i++) {
        const int z = (int)(i / (N * N)), y = (int)(i / N % N), x = (int)(i % N);
        d[i] = (uint8_t)(100 + 60 * sin(0.13 * x + 0.05 * y) * cos(0.07 * z) + (int)((i * 2654435761u) >> 28));
    }
    int fails = 0;
    const float qs[2] = {8.f, 0.f};   /* lossy, lossless */
    for (int qi = 0; qi < 2; qi++) {
        char p[512]; snprintf(p, sizeof p, "%s/v%d/0", dir, qi);
        const int64_t sh[3] = {N, N, N};
        z3w *w = z3w_create(p, sh, 256, qs[qi], 0, nullptr);
        if (!w || z3w_write_shard(w, 0, 0, 0, d, 2) || z3w_close(w)) { fprintf(stderr, "write failed\n"); return 1; }
        snprintf(p, sizeof p, "%s/v%d", dir, qi);
        store *s = store_open(p); z3 *z = s ? z3_open(s, "0", nullptr) : nullptr;
        if (!z) { fprintf(stderr, "open failed: %s\n", z3_error()); return 1; }
        const int64_t boxes[][6] = {{96, 100, 3, 64, 40, 200}, {0, 0, 0, 256, 256, 256}, {127, 31, 64, 2, 2, 2}, {16, 32, 0, 16, 32, 128}, {100, 200, 50, 150, 56, 77}};
        for (int b = 0; b < 5; b++) {
            const int64_t *o = boxes[b], *n = boxes[b] + 3; const size_t nv = (size_t)n[0] * n[1] * n[2];
            uint8_t *a = malloc(nv), *f = malloc(nv);
            setenv("UFSM_Z3_AUTOSUB", "0", 1); const int ra = z3_read(z, o, n, f, 2);
            unsetenv("UFSM_Z3_AUTOSUB"); const int rb = z3_read(z, o, n, a, 2);
            if (ra || rb || memcmp(a, f, nv)) { fprintf(stderr, "FAIL q %g box %d: substream read differs from the whole-chunk read\n", qs[qi], b); fails++; }
            free(a); free(f);
        }
        z3_close(z); store_close(s);
    }
    free(d);
    if (!fails) { char cmd[600]; snprintf(cmd, sizeof cmd, "rm -rf %s", dir); if (system(cmd)) {} }
    if (fails) { fprintf(stderr, "test_z3_autosub: %d FAILED (kept %s)\n", fails, dir); return 1; }
    printf("test_z3_autosub: ok\n");
    return 0;
}
