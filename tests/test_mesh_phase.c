/* sources "phase": {"onfly": {"meshes": {dir, value, level}, rmax}}: the phase from codes rasterised on read from q-assigned meshes,
   behind the derived-array cache (tests/test_mesh_phase.sh makes the meshes and the stored rasters).
   test_mesh_phase MESH.json STORED.json STORED1.json: MESH.json's phase (level 0) == STORED.json's (the same rule on `ufsm raster`'s
   level-0 codes), its next level (phase_pl[1], codes rasterised at level 1) == STORED1.json's (level-1 raster), over windows not
   aligned to the cache cells; a second load of MESH.json reads them back from the disk cache (no recompute) with the same bytes. */
#include "dcache.h"
#include "sources.h"
#include "zarr3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int cmp(z3 *a, z3 *b, const char *what, size_t *known) {
    const z3_meta *ma = z3_meta_of(a), *mb = z3_meta_of(b);
    for (int d = 0; d < 3; d++) if (ma->shape[d] != mb->shape[d]) { fprintf(stderr, "FAIL %s: shapes differ\n", what); return 1; }
    int bad = 0; size_t nd = 0, nk = 0;
    for (int w = 0; w < 6; w++) {
        int64_t o[3], n[3];
        for (int d = 0; d < 3; d++) { n[d] = ma->shape[d] / 3 + 5; o[d] = (int64_t)((w * 37 + d * 53) % (int)(ma->shape[d] - n[d] + 1)); }
        const size_t v = (size_t)n[0] * n[1] * n[2]; uint8_t *x = malloc(v), *y = malloc(v);
        if (z3_read(a, o, n, x, 4) || z3_read(b, o, n, y, 4)) { fprintf(stderr, "FAIL %s: read\n", what); bad = 1; }
        for (size_t k = 0; k < v; k++) { nd += x[k] != y[k]; nk += x[k] != 255; }
        free(x); free(y);
    }
    fprintf(stderr, "%s: %zu voxels differ, %zu known\n", what, nd, nk);
    if (known) *known = nk;
    return bad || nd;
}
int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: test_mesh_phase MESH.json STORED.json STORED1.json\n"); return 2; }
    store_global_init();
    sources *M = sources_load(argv[1]), *S0 = sources_load(argv[2]), *S1 = sources_load(argv[3]);
    if (!M || !S0 || !S1) { fprintf(stderr, "FAIL: sources_load\n"); return 1; }
    int fails = 0; size_t k0 = 0, k1 = 0;
    source *m = &M->src[0];
    fails += cmp(m->phase, S0->src[0].phase, "level 0: meshes vs stored codes", &k0);
    fails += !m->phase_pl[1] || cmp(m->phase_pl[1], S1->src[0].phase, "level 1: meshes at level 1 vs the level-1 raster", &k1);
    if (!k0 || !k1) { fprintf(stderr, "FAIL: no known phase (%zu / %zu)\n", k0, k1); fails++; }
    dcache_stats a; dcache_get_stats(&a);
    sources *M2 = sources_load(argv[1]);
    fails += !M2 || cmp(M2->src[0].phase, S0->src[0].phase, "level 0 again (the disk cache)", nullptr);
    dcache_stats b; dcache_get_stats(&b);
    fprintf(stderr, "dcache: first load %llu cells computed, second load %llu more computed, %llu from disk\n", (unsigned long long)a.cells_computed,
            (unsigned long long)(b.cells_computed - a.cells_computed), (unsigned long long)(b.cells_disk - a.cells_disk));
    if (!a.cells_computed || b.cells_computed != a.cells_computed || b.cells_disk == a.cells_disk) { fprintf(stderr, "FAIL: the second load did not read the disk cache\n"); fails++; }
    sources_free(M); sources_free(M2); sources_free(S0); sources_free(S1);
    fprintf(stderr, "%s\n", fails ? "test_mesh_phase: FAILED" : "test_mesh_phase: ok");
    return fails != 0;
}
