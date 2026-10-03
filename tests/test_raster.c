/* CPU-only: indexed multi-surface raster and row transform against the original
   implementation, including tile/shard boundaries, holes and pooled geometry. */
#include "../src/ingest.c"
#include "sources.h"
#include "volcomp.h"
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
    for (int ci = 0; ci < 8; ci++) {
        double scale = ci & 1 ? 0.5 : 1.0; int T = ci < 2 ? 3 : 8;
        int binary = ci >= 4;
        int64_t shape[3] = {257, 129, 257};
        char path[2][1024];
        for (int mode = 0; mode < 2; mode++) {
            snprintf(path[mode], sizeof path[mode], "%s/case%d-%s", root, ci, mode ? "indexed" : "reference");
            z3w *writer = binary ? z3w_create_mask(path[mode], shape, 128, "{\"ufsm\":{\"encoding\":\"binary\"}}") : z3w_create(path[mode], shape, 128, 0.f, 255, "{}");
            rjob j = {.meshes = meshes, .nm = 3, .scale = scale, .shard = 128, .margin = binary ? 2 : T + 3, .T = T, .binary = binary,
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
            fprintf(stderr, "raster mismatch scale=%g T=%d binary=%d\n", scale, T, binary); failures++;
        } else printf("multi-surface raster scale=%g T=%d binary=%d: %zu voxels identical\n", scale, T, binary, n);
        if (binary && za && zb) {
            if (!z3_meta_of(zb)->label_binary || z3_meta_of(zb)->fill != 0) failures++;
            for (size_t k = 0; k < n; k++) if (b[k] != 0 && b[k] != 255) { failures++; break; }
            /* Verify actual shards use the full 128-cubed mask codec, without its lossy 2x pooling mode. */
            char shard_path[1200]; snprintf(shard_path, sizeof shard_path, "%s/c/0/0/0", path[1]);
            FILE *f = fopen(shard_path, "rb");
            if (!f) failures++;
            else {
                fseek(f, 0, SEEK_END); long bytes = ftell(f); rewind(f);
                uint8_t *enc = malloc((size_t)bytes); uint32_t dim = 0;
                if (fread(enc, 1, (size_t)bytes, f) != (size_t)bytes ||
                    volcomp_mask_info(enc, (size_t)bytes - 20, &dim) != VOLCOMP_OK || dim != 128) failures++;
                fclose(f); free(enc);
            }
            /* Global nearest-neighbour coordinates, including odd origins and volume padding. */
            for (int shift = 1; shift <= 2; shift++) for (int edge = 0; edge < 3; edge++) {
                int64_t o[3] = {edge == 0 ? -3 : edge == 1 ? 127 : shape[0] * (1 << shift) - 3, 123, 251};
                int64_t nn[3] = {9, 7, 30}; uint8_t got[9 * 7 * 30];
                if (z3_read_label_grid(zb, 1, shift, o, nn, got, 1)) { failures++; continue; }
                for (int z = 0; z < 9; z++) for (int y = 0; y < 7; y++) for (int x = 0; x < 30; x++) {
                    int64_t g[3] = {o[0] + z, o[1] + y, o[2] + x}, ix[3]; int inside = 1;
                    for (int d = 0; d < 3; d++) {
                        inside &= g[d] >= 0 && g[d] < shape[d] * (1 << shift);
                        ix[d] = (int64_t)floor((double)g[d] / (1 << shift) + 0.5);
                        if (ix[d] < 0) ix[d] = 0;
                        if (ix[d] >= shape[d]) ix[d] = shape[d] - 1;
                    }
                    uint8_t expected = inside && a[((size_t)ix[0] * shape[1] + ix[1]) * shape[2] + ix[2]] ? 254 : 0;
                    if (got[(z * 7 + y) * 30 + x] != expected) { failures++; goto end_upsample; }
                }
                end_upsample:;
            }
        }
        free(a); free(b); z3_close(za); z3_close(zb); store_close(sa); store_close(sb);
    }
    for (int i = 0; i < 3; i++) { free(meshes[i]->xyz); free(meshes[i]->valid); free(meshes[i]); }
}

static void binary_pool(void) {
    uint8_t in[8 * 8 * 8], out[4 * 4 * 4];
    for (int pattern = 0; pattern < 6; pattern++) {
        for (size_t k = 0; k < sizeof in; k++) in[k] = pattern == 0 ? 0 : pattern == 1 ? 255 : random_u32() % 11 == 0 ? 255 : 0;
        pool2_mask(in, 8, out);
        for (int z = 0; z < 4; z++) for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++) {
            int hit = 0;
            for (int dz = 0; dz < 2; dz++) for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++)
                hit |= in[(((2 * z + dz) * 8 + 2 * y + dy) * 8 + 2 * x + dx)] != 0;
            if (out[(z * 4 + y) * 4 + x] != (hit ? 255 : 0)) failures++;
        }
    }
}

int main(void) {
    char root[] = "/tmp/ufsm-raster-test-XXXXXX";
    if (!mkdtemp(root)) return 2;
    distances(); binary_pool(); volumes(root);
    printf("CPU raster tests: %s (%s)\n", failures ? "FAIL" : "PASS", root);
    return failures != 0;
}
