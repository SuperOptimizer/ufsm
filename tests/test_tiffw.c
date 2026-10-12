/* src/tiffw.c: uint8 and float32 images written (uncompressed and deflate, one and several strips) read back bit-exact by
   src/tiff.c; a tifxyz directory has x / y / z float32 grids with -1 for invalid vertices and a meta.json with the bbox of the
   valid ones and area_vx2 = valid quads x step^2. */
#include "tiff.h"
#include "tiffw.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

static void roundtrip(int w, int h, int fmt, int deflate) {
    const int bps = fmt == 1 ? 1 : 4; const size_t n = (size_t)w * h;
    uint8_t *a = malloc(n * bps), *b = malloc(n * bps);
    for (size_t i = 0; i < n; i++) {
        if (fmt == 1) a[i] = (uint8_t)(i * 37 % 251);
        else { float v = sinf((float)i * 0.01f) * 1000.f - (i % 7 == 0 ? 1.f : 0.f); memcpy(a + 4 * i, &v, 4); }
    }
    const char *p = "/tmp/ufsm_test_tiffw.tif";
    CHECK(!tiffw_write(p, w, h, fmt, a, deflate), "write %dx%d fmt %d deflate %d", w, h, fmt, deflate);
    tiff *t = tiff_open_file(p); tiff_page pg = {0};
    CHECK(t && !tiff_page_info(t, 0, &pg) && pg.w == w && pg.h == h && pg.bits == 8 * bps && pg.fmt == fmt && pg.spp == 1, "header %dx%d fmt %d", w, h, fmt);
    CHECK(t && !tiff_read_page(t, 0, b) && !memcmp(a, b, n * bps), "pixels %dx%d fmt %d deflate %d", w, h, fmt, deflate);
    if (t) tiff_close(t);
    remove(p); free(a); free(b);
}

int main(void) {
    for (int fmt = 1; fmt <= 3; fmt += 2) for (int d = 0; d < 2; d++) { roundtrip(37, 5, fmt, d); roundtrip(301, 200, fmt, d); }
    const int w = 6, h = 4; float x[24], y[24], z[24];
    for (int i = 0; i < w * h; i++) { x[i] = 100.f + i; y[i] = 200.f + 2 * i; z[i] = i == 7 ? -1.f : 300.f + i; }
    if (tifxyz_write("/tmp/ufsm_test_tifxyz/w007", w, h, x, y, z, 20, 2.4, 7, "test")) { fprintf(stderr, "FAIL: tifxyz write\n"); return 1; }
    tiff *t = tiff_open_file("/tmp/ufsm_test_tifxyz/w007/z.tif"); float zb[24];
    CHECK(t && !tiff_read_page(t, 0, zb) && !memcmp(z, zb, sizeof z), "tifxyz z grid");
    if (t) tiff_close(t);
    FILE *f = fopen("/tmp/ufsm_test_tifxyz/w007/meta.json", "r"); char buf[1024] = {0};
    CHECK(f && fread(buf, 1, sizeof buf - 1, f) > 0, "meta.json");
    if (f) fclose(f);
    /* valid quads: 5 x 3 = 15 minus the 4 around vertex 7 (row 1, col 1) = 11 */
    CHECK(strstr(buf, "\"area_vx2\": 4400.0") && strstr(buf, "\"bbox\": [[100.000, 200.000, 300.000], [123.000, 246.000, 323.000]]") &&
          strstr(buf, "\"scale\": [0.05, 0.05]") && strstr(buf, "\"winding\": 7"), "meta.json content: %s", buf);
    if (system("rm -rf /tmp/ufsm_test_tifxyz")) fprintf(stderr, "cannot remove /tmp/ufsm_test_tifxyz\n");
    if (fails) { fprintf(stderr, "test_tiffw: %d failure(s)\n", fails); return 1; }
    printf("test_tiffw: OK\n");
    return 0;
}
