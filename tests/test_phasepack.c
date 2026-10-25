/* phase-pack round trip: a 2x3x3-shard phase array (partial edge shards, unknown regions, values near the 251 -> 0 wrap) is packed
   per z row of shards, unpacked into a directory that already holds a zarr.json with custom attributes; checks: known voxels within
   cyclic error 2 of 252, unknown (255) exactly preserved, absent shards stay absent, the existing zarr.json is kept byte for byte. */
#include "zarr3.h"
#include "z3w.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cmd_phase_pack(int argc, char **argv);
void phase_pack_lines(uint8_t *v, size_t nlines, int nx);
void phase_unpack_lines(uint8_t *v, size_t nlines, int nx);

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

static char *slurp(const char *p, size_t *n) {
    FILE *f = fopen(p, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); long L = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)L + 1); *n = fread(b, 1, (size_t)L, f); b[*n] = 0; fclose(f); return b;
}
static int cyc(int a, int b) { int e = abs(a - b) % 252; return e > 126 ? 252 - e : e; }

int main(void) {
    /* line codec alone: every value, unknown runs */
    {
        uint8_t v[512], w[512];
        for (int i = 0; i < 512; i++) v[i] = (i % 97 == 5 || (i > 300 && i < 320)) ? 255 : (uint8_t)((i * 37) % 252);
        memcpy(w, v, sizeof v); phase_pack_lines(w, 2, 256);
        for (int i = 0; i < 512; i++) CHECK(w[i] <= 63, "code %d at %d", w[i], i);
        phase_unpack_lines(w, 2, 256);
        for (int i = 0; i < 512; i++) {
            if (v[i] == 255) CHECK(w[i] == 255, "unknown lost at %d", i);
            else CHECK(w[i] != 255 && cyc(v[i], w[i]) <= 2, "value %d -> %d at %d", v[i], w[i], i);
        }
    }
    char dir[] = "/tmp/test_phasepack_XXXXXX"; CHECK(mkdtemp(dir) != nullptr, "mkdtemp");
    char src[600], dst[600], pk[600], zj[700];
    snprintf(src, sizeof src, "%s/a/9.6", dir); snprintf(dst, sizeof dst, "%s/b/9.6", dir); snprintf(pk, sizeof pk, "%s/row", dir);
    const int S = 128; const int64_t shape[3] = {200, 300, 330};
    z3w *w = z3w_create(src, shape, S, 0, 255, "{\"ufsm\":{\"content\":\"phase test\"}}");
    CHECK(w != nullptr, "z3w_create %s", z3w_error());
    uint8_t *buf = malloc((size_t)S * S * S);
    for (int sz = 0; sz < 2; sz++) for (int sy = 0; sy < 3; sy++) for (int sx = 0; sx < 3; sx++) {
        if (sy == 1 && sx == 2) continue;   /* an absent shard */
        for (int z = 0; z < S; z++) for (int y = 0; y < S; y++) for (int x = 0; x < S; x++) {
            const int gz = sz * S + z, gy = sy * S + y, gx = sx * S + x;
            uint8_t v = (uint8_t)((gx * 3 + gy + gz * 2) % 252);   /* wraps 251 -> 0 along x */
            if ((gx / 17 + gy / 13 + gz / 11) % 4 == 0) v = 255;
            buf[((size_t)z * S + y) * S + x] = v;
        }
        CHECK(!z3w_write_shard(w, sz, sy, sx, buf, 2), "write shard %s", z3w_error());
    }
    CHECK(!z3w_close(w), "close");
    /* the destination already has its metadata (as on the pod): custom attributes that must survive */
    z3w *d = z3w_create(dst, shape, S, 0, 255, "{\"keep\":\"me\"}"); CHECK(d && !z3w_close(d), "dst");
    snprintf(zj, sizeof zj, "%s/zarr.json", dst); size_t n0; char *before = slurp(zj, &n0);
    for (int row = 0; row < 2; row++) {
        char rs[8]; snprintf(rs, sizeof rs, "%d", row);
        char *pa[] = {"ufsm", "phase-pack", dir, "a/9.6", pk, "--row", rs, "--level", "3", "--threads", "2", nullptr};
        CHECK(cmd_phase_pack(11, pa) == 0, "pack row %d", row);
        char *ua[] = {"ufsm", "phase-unpack", pk, dst, "--threads", "2", nullptr};
        CHECK(cmd_phase_pack(6, ua) == 0, "unpack row %d", row);
        char *ca[] = {"ufsm", "phase-cmp", dir, "a/9.6", dir, "b/9.6", "--row", rs, nullptr};
        CHECK(cmd_phase_pack(8, ca) == 0, "cmp row %d", row);
    }
    size_t n1; char *after = slurp(zj, &n1);
    CHECK(before && after && n0 == n1 && !memcmp(before, after, n0), "zarr.json changed");
    char ab[800]; snprintf(ab, sizeof ab, "%s/c/0/1/2", dst); CHECK(access(ab, F_OK) != 0, "absent shard was written");
    /* explicit read-back of an edge region through the reader */
    store *s = store_open(dir); z3 *za = z3_open(s, "a/9.6", nullptr), *zb = z3_open(s, "b/9.6", nullptr);
    CHECK(za && zb, "open");
    if (za && zb) {
        const int64_t o[3] = {100, 200, 250}, n[3] = {100, 100, 80};
        uint8_t *ra = malloc(100 * 100 * 80), *rb = malloc(100 * 100 * 80); int bad = 0, known = 0;
        CHECK(!z3_read(za, o, n, ra, 2) && !z3_read(zb, o, n, rb, 2), "read");
        for (int k = 0; k < 100 * 100 * 80; k++) {
            if (ra[k] == 255) bad += rb[k] != 255; else { known++; bad += rb[k] == 255 || cyc(ra[k], rb[k]) > 2; }
        }
        CHECK(bad == 0 && known > 0, "%d bad of %d known", bad, known);
        free(ra); free(rb); z3_close(za); z3_close(zb);
    }
    store_close(s);
    char rm[700]; snprintf(rm, sizeof rm, "rm -rf %s", dir); if (system(rm)) {}
    free(buf); free(before); free(after);
    if (fails) { fprintf(stderr, "test_phasepack: %d failures\n", fails); return 1; }
    printf("test_phasepack: ok\n");
    return 0;
}
