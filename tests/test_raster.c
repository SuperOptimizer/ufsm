/* CPU-only: indexed multi-surface raster and row transform against the original
   implementation, including tile/shard boundaries, holes and pooled geometry. */
#include "../src/ingest.c"
#include <unistd.h>

static int failures;
static uint32_t state = 17;
static uint32_t random_u32(void) { state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state; }

static void distances(void) {
    for (int N = 1; N <= 35; N += N < 5 ? 1 : 5) for (int T = 1; T <= 70; T = T == 1 ? 3 : T == 3 ? 8 : T == 8 ? 70 : 71) {
        size_t n = (size_t)N * N * N;
        uint8_t *a = malloc(n), *b = malloc(n);
        for (int pattern = 0; pattern < 4; pattern++) {
            for (size_t i = 0; i < n; i++) a[i] = pattern == 0 ? 255 : pattern == 1 ? 0 : random_u32() % 29 == 0 ? 0 : 255;
            if (pattern == 3) { memset(a, 255, n); a[n - 1] = 0; }
            memcpy(b, a, n);
            chamfer_reference(a, N, 3 * T + 3); raster_distance(b, N, 3 * T + 3);
            if (memcmp(a, b, n)) { fprintf(stderr, "distance mismatch N=%d T=%d pattern=%d\n", N, T, pattern); failures++; }
        }
        free(a); free(b);
    }
}

static mesh *fixture(int which) {
    mesh *m = calloc(1, sizeof *m); m->w = 66; m->h = 35;
    size_t n = (size_t)m->w * m->h;
    m->xyz = malloc(n * 3 * sizeof(float)); m->valid = malloc(n);
    for (int d = 0; d < 3; d++) { m->lo[d] = 1e30; m->hi[d] = -1e30; }
    for (int r = 0; r < m->h; r++) for (int c = 0; c < m->w; c++) {
        size_t k = (size_t)r * m->w + c;
        float p[3] = {3.0f * c + 25.0f, 3.0f * r + 10.0f, 127.0f + which * 12.0f + 5.0f * sinf(c * 0.09f) + 0.17f * r};
        if (which == 1) { p[2] = p[0] + 83.0f; p[0] = 128.0f + 0.2f * r; }
        if (which == 2) { p[0] += 280.0f; p[1] += 40.0f; p[2] += 210.0f; }
        m->valid[k] = which == 2 || !(r >= 12 && r < 19 && c >= 29 && c < 35);
        for (int a = 0; a < 3; a++) {
            m->xyz[k * 3 + a] = p[a]; if (!m->valid[k]) continue;
            int d = 2 - a;
            if (p[a] < m->lo[d]) m->lo[d] = p[a];
            if (p[a] > m->hi[d]) m->hi[d] = p[a];
        }
    }
    return m;
}

static void volumes(const char *root) {
    mesh *meshes[3] = {fixture(0), fixture(1), fixture(2)};
    for (int ci = 0; ci < 4; ci++) {
        double scale = ci & 1 ? 0.5 : 1.0; int T = ci < 2 ? 3 : 8;
        int64_t shape[3] = {257, 129, 257};
        char path[2][1024];
        for (int mode = 0; mode < 2; mode++) {
            snprintf(path[mode], sizeof path[mode], "%s/case%d-%s", root, ci, mode ? "indexed" : "reference");
            z3w *writer = z3w_create(path[mode], shape, 128, 0.f, 255, "{}");
            rjob j = {.meshes = meshes, .nm = 3, .scale = scale, .shard = 128, .margin = T + 3, .T = T,
                      .ns = {3, 2, 3}, .shape = {257,129,257}, .w = writer,
                      .indexed = mode, .reference_distance = !mode, .nthreads = 1};
            if (mode && raster_index(&j)) { failures++; return; }
            for (int si = 0; si < 18; si++) raster_shard(si, 0, &j);
            if (atomic_load(&j.failed)) failures++;
            free(j.tiles); free(j.offset); free(j.refs); z3w_close(writer);
        }
        store *sa = store_open(path[0]), *sb = store_open(path[1]);
        z3 *za = z3_open(sa, ".", nullptr), *zb = z3_open(sb, ".", nullptr);
        const int64_t origin[3] = {0,0,0}; size_t n = (size_t)shape[0] * shape[1] * shape[2];
        uint8_t *a = malloc(n), *b = malloc(n);
        if (!za || !zb || z3_read(za, origin, shape, a, 1) || z3_read(zb, origin, shape, b, 1) || memcmp(a,b,n)) {
            fprintf(stderr, "raster mismatch scale=%g T=%d\n", scale, T); failures++;
        } else printf("multi-surface raster scale=%g T=%d: %zu voxels identical\n", scale, T, n);
        free(a); free(b); z3_close(za); z3_close(zb); store_close(sa); store_close(sb);
    }
    for (int i = 0; i < 3; i++) { free(meshes[i]->xyz); free(meshes[i]->valid); free(meshes[i]); }
}

int main(void) {
    char root[] = "/tmp/ufsm-raster-test-XXXXXX";
    if (!mkdtemp(root)) return 2;
    distances(); volumes(root);
    printf("CPU raster tests: %s (%s)\n", failures ? "FAIL" : "PASS", root);
    return failures != 0;
}
