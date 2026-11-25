/* bench_chunk_map ROOT KEY [THREADS...]: z3_chunk_map's wall time per UFSM_CHUNKMAP_THREADS value on a real array (each on a fresh open:
   no in-memory index), and whether every map is the same. perf 11-25: R11d's teacher zarrs over rclone ~/nas2. */
#include "zarr3.h"
#include "store.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: bench_chunk_map ROOT KEY [THREADS ...]\n"); return 2; }
    uint8_t *ref = nullptr; size_t nref = 0; int bad = 0;
    for (int a = argc > 3 ? 3 : 2; a < argc; a++) {
        if (argc > 3) setenv("UFSM_CHUNKMAP_THREADS", argv[a], 1);
        store *s = store_open(argv[1]); z3 *z = s ? z3_open(s, argv[2], nullptr) : nullptr; if (!z) { fprintf(stderr, "open: %s\n", z3_error()); return 1; }
        int64_t g[3]; uint8_t *b = nullptr; const double t0 = now(); const int64_t n = z3_chunk_map(z, g, &b); const double t1 = now();
        const size_t nb = (size_t)(g[0] * g[1] * g[2]);
        if (!ref) { ref = b; nref = nb; b = nullptr; } else { bad |= nb != nref || memcmp(ref, b, nb); free(b); }
        printf("threads %s: %lld chunks stored of %zu, %.2f s\n", argc > 3 ? argv[a] : "default", (long long)n, nb, t1 - t0);
        z3_close(z); store_close(s);
    }
    printf("maps %s\n", bad ? "DIFFER" : "identical"); free(ref); return bad;
}
