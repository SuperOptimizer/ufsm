/* decode cost of cached CT windows: bench_read <sources.json> [level] [W] [n]: times z3_read of chunk-aligned W^3 windows
   around occupied cells (single thread), so the per-chunk decode cost is visible without the sampler. */
#include "sample.h"
#include "sources.h"
#include "zarr3.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: bench_read <sources.json> [level] [W] [n]\n"); return 2; }
    int level = argc > 2 ? atoi(argv[2]) : 0, W = argc > 3 ? atoi(argv[3]) : 128, n = argc > 4 ? atoi(argv[4]) : 20;
    sources *S = sources_load(argv[1]); if (!S) return 1;
    source *s = &S->src[0];
    occ_index o = source_occupancy(s);
    z3 *ct = source_ct(s, level); if (!ct) { fprintf(stderr, "no level %d\n", level); return 1; }
    const z3_meta *m = z3_meta_of(ct);
    printf("%s level %d: shape %lld %lld %lld chunk %d %d %d, %zu occupied cells at level %d\n", s->name, level, (long long)m->shape[0], (long long)m->shape[1], (long long)m->shape[2], m->chunk[0], m->chunk[1], m->chunk[2], o.n, o.lev);
    uint8_t *buf = malloc((size_t)W * W * W);
    double tot = 0; size_t nz = 0; int done = 0;
    for (size_t i = 0; i < o.n && done < n; i += o.n / (size_t)n + 1) {
        uint32_t id = o.idx[i];
        int64_t cz = id / (o.shape[1] * o.shape[2]), cy = (id / o.shape[2]) % o.shape[1], cx = id % o.shape[2];
        int64_t cc[3] = {cz, cy, cx}, org[3], nn[3] = {W, W, W};
        int dl = o.lev - level;
        for (int d = 0; d < 3; d++) { int64_t v = dl >= 0 ? cc[d] << dl : cc[d] >> (-dl); v -= v % m->chunk[d]; if (v > m->shape[d] - W) v = m->shape[d] - W; if (v < 0) v = 0; org[d] = v; }
        double t0 = now();
        if (z3_read(ct, org, nn, buf, 1)) { fprintf(stderr, "read: %s\n", z3_error()); return 1; }
        tot += now() - t0; done++;
        for (size_t k = 0; k < (size_t)W * W * W; k += 997) nz += buf[k] != 0;
    }
    printf("%d windows of %d^3 at level %d: %.1f ms each (%.1f MB/s decoded), nonzero sample %zu\n", done, W, level, tot / done * 1e3, (double)W * W * W / 1e6 / (tot / done), nz);
    return 0;
}
