/* CPU-only: indexed multi-surface raster and row transform against the original
   implementation, including tile/shard boundaries, holes and pooled geometry. */
#include "../src/ingest.c"
#include "sources.h"
#include "volcomp.h"
#include <unistd.h>
#include <sys/stat.h>

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
    for (int ci = 0; ci < 10; ci++) {
        double scale = ci & 1 ? 0.5 : 1.0; int T = ci < 2 ? 3 : 8;
        int binary = ci >= 4;
        int64_t shape[3] = {257, 129, 257};
        char path[2][1024];
        for (int mode = 0; mode < 2; mode++) {
            snprintf(path[mode], sizeof path[mode], "%s/case%d-%s", root, ci, mode ? "indexed" : "reference");
            z3w *writer = binary ? z3w_create_mask(path[mode], shape, 128, "{\"ufsm\":{\"encoding\":\"binary\"}}") : z3w_create(path[mode], shape, 128, 0.f, 255, "{}");
            rjob j = {.meshes = meshes, .nm = 3, .scale = scale, .shard = 128, .margin = binary ? 2 : T + 3, .T = T, .binary = binary, .band=ci>=8?0:4,
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
            /* binary 2 (side labels): raw codes, the pooled cell floor(g / 2^shift) (not rounded) */
            for (int shift = 1; shift <= 2; shift++) for (int edge = 0; edge < 3; edge++) {
                int64_t o[3] = {edge == 0 ? -3 : edge == 1 ? 127 : shape[0] * (1 << shift) - 3, 123, 251};
                int64_t nn[3] = {9, 7, 30}; uint8_t got[9 * 7 * 30];
                if (z3_read_label_grid(zb, 2, shift, o, nn, got, 1)) { failures++; continue; }
                for (int z = 0; z < 9; z++) for (int y = 0; y < 7; y++) for (int x = 0; x < 30; x++) {
                    int64_t g[3] = {o[0] + z, o[1] + y, o[2] + x}, ix[3]; int inside = 1;
                    for (int d = 0; d < 3; d++) {
                        inside &= g[d] >= 0 && g[d] < shape[d] * (1 << shift);
                        ix[d] = (int64_t)floor((double)g[d] / (1 << shift));
                        if (ix[d] < 0) ix[d] = 0;
                        if (ix[d] >= shape[d]) ix[d] = shape[d] - 1;
                    }
                    uint8_t expected = inside ? b[((size_t)ix[0] * shape[1] + ix[1]) * shape[2] + ix[2]] : 0;
                    if (got[(z * 7 + y) * 30 + x] != expected) { fprintf(stderr, "side upsample mismatch shift %d\n", shift); failures++; goto end_side; }
                }
                end_side:;
            }
        }
        free(a); free(b); z3_close(za); z3_close(zb); store_close(sa); store_close(sb);
    }
    for (int i = 0; i < 3; i++) { free(meshes[i]->xyz); free(meshes[i]->valid); free(meshes[i]); }
}

/* raster --dist 1 (arm K's surface-proximity input): indexed + row transform == the reference, and against the plain label raster of
   the same meshes: its surface (254, chamfer <= 4) has codes >= 254 (cap - 4) / cap, its ignore band edge (> 3T) codes <= 254 * 2 / cap */
static void dist_volumes(const char *root) {
    mesh *meshes[3] = {fixture(0), fixture(1), fixture(2)};
    for (int ci = 0; ci < 2; ci++) {
        const double scale = ci ? 0.5 : 1.0; const int T = 6, cap = 3 * T + 3;
        int64_t shape[3] = {257, 129, 257}; char path[3][1024];
        for (int mode = 0; mode < 3; mode++) {   /* 0 reference distance, 1 indexed, 2 the label raster */
            snprintf(path[mode], sizeof path[mode], "%s/dist%d-%d", root, ci, mode);
            z3w *writer = z3w_create(path[mode], shape, 128, 0.f, mode < 2 ? 0 : 255, "{}");
            rjob j = {.meshes = meshes, .nm = 3, .scale = scale, .shard = 128, .margin = T + 3, .T = T, .ns = {3, 2, 3}, .shape = {257,129,257}, .w = writer,
                      .indexed = mode >= 1, .reference_distance = mode == 0, .dist = mode < 2, .nthreads = 1};
            if (j.indexed && raster_index(&j)) { failures++; return; }
            for (int si = 0; si < 18; si++) raster_shard(si, 0, &j);
            if (atomic_load(&j.failed)) failures++;
            free(j.tiles); free(j.offset); free(j.refs); z3w_close(writer);
        }
        const int64_t origin[3] = {0,0,0}; const size_t n = (size_t)shape[0] * shape[1] * shape[2]; uint8_t *v[3]; int ok = 1;
        for (int mode = 0; mode < 3; mode++) {
            store *s = store_open(path[mode]); z3 *z = s ? z3_open(s, ".", nullptr) : nullptr; v[mode] = malloc(n);
            if (!z || z3_read(z, origin, shape, v[mode], 1)) ok = 0;
            z3_close(z); store_close(s);
        }
        size_t on = 0, near = 0, bad = 0;
        if (ok && memcmp(v[0], v[1], n)) { fprintf(stderr, "dist raster: indexed != reference (scale %g)\n", scale); failures++; }
        for (size_t k = 0; ok && k < n; k++) {
            const int c = v[1][k], l = v[2][k];
            if (c > 254) bad++;
            if (l == 254) { on++; if (c < (254 * (cap - 4) + cap / 2) / cap) bad++; }
            else if (l == 255 && c > (254 * 2 + cap / 2) / cap && c) bad++;
            else if (c) near++;
        }
        if (!ok || bad || !on || !near) { fprintf(stderr, "dist raster scale %g: read %d, %zu codes off the label raster (%zu on, %zu near)\n", scale, ok, bad, on, near); failures++; }
        else printf("dist raster scale=%g: indexed == reference, %zu surface / %zu band voxels agree with the label raster\n", scale, on, near);
        for (int mode = 0; mode < 3; mode++) free(v[mode]);
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

/* audit mini/coords #1: a mask pyramid (point-sampled base + pool2_mask levels) read finer than a pooled level. Level l (2^l above the base)
   is the any-of over base blocks of 2^l, so a voxel q at read level r reads level l's cell exactly when it rounds into a base cell of that
   block: expected = any over base block (round(q / 2^-r) >> l); read below the base, round then floor. The old rounding through the
   pooled levels missed set voxels in the upper half of each block. */
static void pooled_pyramid(const char *root) {
    char gdir[1100], base[1200], lv[32]; snprintf(gdir, sizeof gdir, "%s/pyr", root);
    const int64_t shape[3] = {128, 128, 128}; const size_t n = 128 * 128 * 128;
    uint8_t *m = calloc(n, 1); for (size_t k = 0; k < n; k++) m[k] = random_u32() % 97 == 0 ? 255 : 0;
    const char *attrs = "{\"ufsm\":{\"encoding\":\"binary\"}}";
    mkdir(gdir, 0755); z3w_level_name(1.0, lv, sizeof lv); snprintf(base, sizeof base, "%s/%s", gdir, lv);
    z3w *w = z3w_create_mask(base, shape, 128, attrs);
    if (!w || z3w_write_shard(w, 0, 0, 0, m, 1)) { fprintf(stderr, "pooled pyramid: base write failed\n"); failures++; free(m); return; }
    z3w_close(w);
    for (int l = 1; l <= 2; l++) if (pyramid_build_level(gdir, 1.0, l, shape, 128, 0.f, 2, 1, attrs)) { failures++; free(m); return; }
    if (pyramid_write_group(gdir, 1.0, 3, "pooled-test", attrs)) { failures++; free(m); return; }
    store *s = store_open(gdir);
    for (int l = 0; l <= 2; l++) {
        z3w_level_name((double)(1 << l), lv, sizeof lv); z3 *z = z3_open(s, lv, nullptr);
        if (!z || z3_meta_of(z)->label_pooled != l) { fprintf(stderr, "pooled pyramid: level %d label_pooled %d\n", l, z ? z3_meta_of(z)->label_pooled : -9); failures++; z3_close(z); continue; }
        /* read levels: the base (shift l) and one finer than the base (shift l + 1), whole extent plus a margin */
        for (int sub = 0; sub <= 1 && l > 0; sub++) {
            const int shift = l + sub, f = 1 << sub;   /* f: read voxels per base voxel */
            const int64_t o[3] = {-2, 5, 0}, nn[3] = {24, 20, 128 * f + 3};
            uint8_t *got = malloc((size_t)nn[0] * nn[1] * nn[2]); int bad = 0, set = 0;
            if (z3_read_label_grid(z, 1, shift, o, nn, got, 1)) { failures++; free(got); continue; }
            for (int64_t zz = 0; zz < nn[0] && !bad; zz++) for (int64_t y = 0; y < nn[1] && !bad; y++) for (int64_t x = 0; x < nn[2]; x++) {
                const int64_t g[3] = {o[0] + zz, o[1] + y, o[2] + x}; int64_t c[3]; int inside = 1;
                for (int d = 0; d < 3; d++) {
                    inside &= g[d] >= 0 && g[d] < (shape[d] >> l) * ((int64_t)1 << shift);
                    int64_t j = sub ? (int64_t)floor((double)g[d] / f + 0.5) : g[d];   /* base cell */
                    c[d] = j >> l; if (c[d] < 0) c[d] = 0; if (c[d] > (shape[d] >> l) - 1) c[d] = (shape[d] >> l) - 1;
                }
                int hit = 0;
                for (int64_t bz = c[0] << l; bz < (c[0] + 1) << l; bz++) for (int64_t by = c[1] << l; by < (c[1] + 1) << l; by++)
                    for (int64_t bx = c[2] << l; bx < (c[2] + 1) << l; bx++) hit |= m[((size_t)bz * 128 + by) * 128 + bx] != 0;
                const uint8_t e = inside && hit ? 254 : 0, v = got[((size_t)zz * nn[1] + y) * nn[2] + x];
                set += v != 0;
                if (v != e) { fprintf(stderr, "pooled pyramid: level %d shift %d at %lld,%lld,%lld: %d, expected %d\n", l, shift, (long long)g[0], (long long)g[1], (long long)g[2], v, e); bad = 1; failures++; break; }
            }
            if (!bad) printf("pooled mask pyramid: level %d read %d finer: %d set voxels as the any-of blocks\n", l, shift, set);
            free(got);
        }
        z3_close(z);
    }
    store_close(s); free(m);
}

/* (2026-11-26 review) a prob254 target read 2x finer (read_prob_trilinear): 255 is the unknown code, not a value: a fine voxel with any
   weight > 0 on a 255 corner reads 255, the rest the trilinear of the known codes (within 1 of a double reference). It used to blend
   255 into its neighbours (255 next to 0 ~128, "known"). */
static void prob_unknown(const char *root) {
    char dir[1100]; snprintf(dir, sizeof dir, "%s/prob", root);
    const int64_t shape[3] = {128, 128, 128}; const size_t n = 128 * 128 * 128;
    uint8_t *m = malloc(n); for (size_t k = 0; k < n; k++) m[k] = random_u32() % 20 == 0 ? 255 : (uint8_t)(random_u32() % 255);
    z3w *w = z3w_create(dir, shape, 128, 0.f, 0, nullptr);
    if (!w || z3w_write_shard(w, 0, 0, 0, m, 1) || z3w_close(w)) { fprintf(stderr, "prob unknown: write failed\n"); failures++; free(m); return; }
    store *st = store_open(dir); z3 *z = st ? z3_open(st, ".", nullptr) : nullptr;
    const int64_t o[3] = {-1, 40, 100}, nn[3] = {20, 24, 70}; uint8_t *got = malloc((size_t)nn[0] * nn[1] * nn[2]); int bad = 0, unk = 0, kn = 0;
    if (!z || z3_read_label_grid(z, 0, 1, o, nn, got, 1)) { fprintf(stderr, "prob unknown: read failed\n"); failures++; bad = 1; }
    for (int64_t a = 0; a < nn[0] && !bad; a++) for (int64_t b = 0; b < nn[1] && !bad; b++) for (int64_t c = 0; c < nn[2] && !bad; c++) {
        const int64_t g[3] = {o[0] + a, o[1] + b, o[2] + c}; int64_t i0[3]; double t[3];
        for (int d = 0; d < 3; d++) { const double u = (g[d] + 0.5) / 2 - 0.5; int64_t q = (int64_t)floor(u); double f = u - q;
            if (q < 0) { q = 0; f = 0; } if (q >= shape[d] - 1) { q = shape[d] - 1; f = 0; } i0[d] = q; t[d] = f; }
        double v = 0; int u255 = 0;
        for (int k = 0; k < 8; k++) { const int dz = k >> 2 & 1, dy = k >> 1 & 1, dx = k & 1;
            const double wgt = (dz ? t[0] : 1 - t[0]) * (dy ? t[1] : 1 - t[1]) * (dx ? t[2] : 1 - t[2]); if (wgt <= 0) continue;
            const uint8_t cv = m[((size_t)(i0[0] + dz) * 128 + (i0[1] + dy)) * 128 + (i0[2] + dx)]; if (cv == 255) u255 = 1; v += wgt * cv; }
        const uint8_t e = u255 ? 255 : (uint8_t)lround(v), gv = got[((size_t)a * nn[1] + b) * nn[2] + c];
        if (u255 ? gv != 255 : (gv == 255 || abs((int)gv - (int)e) > 1)) {
            fprintf(stderr, "prob unknown: at %lld,%lld,%lld read %d, expected %d (%s)\n", (long long)g[0], (long long)g[1], (long long)g[2], gv, e, u255 ? "a 255 corner" : "known");
            failures++; bad = 1; }
        unk += u255; kn += !u255;
    }
    if (!bad) printf("prob254 read 2x finer: %d voxels on a 255 corner unknown, %d known within 1 of the trilinear\n", unk, kn);
    free(got); free(m); z3_close(z); if (st) store_close(st);
}

int main(void) {
    char root[] = "/tmp/ufsm-raster-test-XXXXXX";
    if (!mkdtemp(root)) return 2;
    distances(); binary_pool(); volumes(root); dist_volumes(root); pooled_pyramid(root); prob_unknown(root);
    printf("CPU raster tests: %s (%s)\n", failures ? "FAIL" : "PASS", root);
    return failures != 0;
}
