/* rvol_open / rvol_read on a zarr v2 pyramid (src/reg.c): levels "0", "1" of a random volume written as raw zarr v2 chunks; the box reads
   (inside, across the edge, fully outside) must equal the in-memory reference, and a level that does not halve the previous stops the pyramid. */
#include "reg.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static void write_level(const char *dir, int l, const uint8_t *v, const int64_t s[3], int ch, char sep) {
    char p[1024]; snprintf(p, sizeof p, "%s/%d", dir, l); mkdir(p, 0755);
    char m[1100]; snprintf(m, sizeof m, "%s/.zarray", p); FILE *f = fopen(m, "w");
    fprintf(f, "{\"chunks\":[%d,%d,%d],\"compressor\":null,\"dimension_separator\":\"%c\",\"dtype\":\"|u1\",\"fill_value\":0,\"filters\":null,\"order\":\"C\","
               "\"shape\":[%lld,%lld,%lld],\"zarr_format\":2}", ch, ch, ch, sep, (long long)s[0], (long long)s[1], (long long)s[2]); fclose(f);
    uint8_t *c = malloc((size_t)ch * ch * ch);
    for (int64_t cz = 0; cz * ch < s[0]; cz++) for (int64_t cy = 0; cy * ch < s[1]; cy++) for (int64_t cx = 0; cx * ch < s[2]; cx++) {
        memset(c, 0, (size_t)ch * ch * ch);
        for (int z = 0; z < ch; z++) for (int y = 0; y < ch; y++) for (int x = 0; x < ch; x++) {
            const int64_t zz = cz * ch + z, yy = cy * ch + y, xx = cx * ch + x;
            if (zz < s[0] && yy < s[1] && xx < s[2]) c[((size_t)z * ch + y) * ch + x] = v[((size_t)zz * s[1] + yy) * s[2] + xx];
        }
        char k[1200]; if (sep == '/') { snprintf(k, sizeof k, "%s/%lld", p, (long long)cz); mkdir(k, 0755); snprintf(k, sizeof k, "%s/%lld/%lld", p, (long long)cz, (long long)cy); mkdir(k, 0755);
                                         snprintf(k, sizeof k, "%s/%lld/%lld/%lld", p, (long long)cz, (long long)cy, (long long)cx); }
        else snprintf(k, sizeof k, "%s/%lld.%lld.%lld", p, (long long)cz, (long long)cy, (long long)cx);
        f = fopen(k, "wb"); fwrite(c, 1, (size_t)ch * ch * ch, f); fclose(f);
    }
    free(c);
}
static uint8_t ref_at(const uint8_t *v, const int64_t s[3], int64_t z, int64_t y, int64_t x) {
    return z < 0 || y < 0 || x < 0 || z >= s[0] || y >= s[1] || x >= s[2] ? 0 : v[((size_t)z * s[1] + y) * s[2] + x];
}
static void check_box(rvol *rv, int l, const uint8_t *v, const int64_t s[3], int64_t oz, int64_t oy, int64_t ox, int64_t n) {
    const int64_t o[3] = {oz, oy, ox}, nn[3] = {n, n, n}; uint8_t *b = malloc((size_t)n * n * n);
    CHECK(!rvol_read_nt(rv, l, o, nn, b, 2), "read level %d box %lld %lld %lld", l, (long long)oz, (long long)oy, (long long)ox);
    int bad = 0;
    for (int64_t z = 0; z < n; z++) for (int64_t y = 0; y < n; y++) for (int64_t x = 0; x < n; x++) bad += b[((size_t)z * n + y) * n + x] != ref_at(v, s, oz + z, oy + y, ox + x);
    CHECK(!bad, "level %d box %lld %lld %lld: %d voxels differ", l, (long long)oz, (long long)oy, (long long)ox, bad);
    free(b);
}

int main(void) {
    char dir[] = "/tmp/test_regz2_XXXXXX"; if (!mkdtemp(dir)) { printf("FAIL mkdtemp\n"); return 1; }
    char vol[1100]; snprintf(vol, sizeof vol, "%s/vol.zarr", dir); mkdir(vol, 0755);
    const int64_t s0[3] = {37, 29, 41}, s1[3] = {19, 15, 21}, sbad[3] = {5, 5, 5};
    uint8_t *v0 = malloc((size_t)s0[0] * s0[1] * s0[2]), *v1 = malloc((size_t)s1[0] * s1[1] * s1[2]), vb[125] = {0};
    srand(3); for (size_t i = 0; i < (size_t)s0[0] * s0[1] * s0[2]; i++) v0[i] = (uint8_t)(1 + rand() % 255);
    for (size_t i = 0; i < (size_t)s1[0] * s1[1] * s1[2]; i++) v1[i] = (uint8_t)(1 + rand() % 255);
    write_level(vol, 0, v0, s0, 8, '/'); write_level(vol, 1, v1, s1, 8, '.'); write_level(vol, 2, vb, sbad, 8, '/');   /* level 2 does not halve level 1 */
    rvol *rv = rvol_open(dir, "vol.zarr", nullptr, 7.91);
    CHECK(rv != nullptr, "rvol_open on a zarr v2 pyramid");
    if (rv) {
        CHECK(rv->nl == 2, "levels %d (want 2: level 2 does not halve level 1)", rv->nl);
        CHECK(rv->um == 7.91, "um %g", rv->um);
        CHECK(rv->shape[0][0] == 37 && rv->shape[0][2] == 41 && rv->shape[1][1] == 15, "shapes");
        check_box(rv, 0, v0, s0, 3, 5, 7, 16);      /* inside, across chunks */
        check_box(rv, 0, v0, s0, 30, -4, 35, 12);   /* across the edges */
        check_box(rv, 0, v0, s0, 50, 50, 50, 4);    /* outside: zeros */
        check_box(rv, 1, v1, s1, 2, 1, 10, 9);
        CHECK(rvol_level_for(rv, 15.82) == 1, "level for 15.82 um");
        rvol_close(rv);
    }
    CHECK(rvol_open(dir, "missing.zarr", nullptr, 1.0) == nullptr, "a missing volume fails cleanly");
    free(v0); free(v1);
    char cmd[1200]; snprintf(cmd, sizeof cmd, "rm -r %s", dir); if (system(cmd)) {}
    printf(fails ? "test_regz2: %d FAILED\n" : "test_regz2: OK\n", fails);
    return fails != 0;
}
