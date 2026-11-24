/* test_pyramid_rel: pyramid_open_level picks the right level of an OME group (sources.c). The mirror's groups list relative scales
   (1, 2, 4 .. 32): PHerc0172 at 7.91 um has scale 8 on level "3", which is 7.91 um x 2^0 within the 2 % tolerance; before 11-24 that
   read the group as absolute and opened level 3 for level 0 (DINO round 1: "level 3 listed but cannot be opened"). Checked: relative
   groups at 7.91 / 2.4 / 8.64 um, and an absolute group (scales 7.91 x 2^l) at 7.91 um. Levels are told apart by their shapes. CPU only. */
#include "sources.h"
#include "store.h"
#include "zarr3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int fails;
static void wr(const char *p, const char *s) { FILE *f = fopen(p, "w"); if (!f) { perror(p); exit(2); } fputs(s, f); fclose(f); }
static void group(const char *dir, const char *name, double s0) {   /* levels "0".."5", shape 64 >> l, scales s0 x 2^l */
    char p[1024], ds[2048] = "", e[256];
    snprintf(p, sizeof p, "%s/%s", dir, name); mkdir(p, 0755);
    for (int l = 0; l < 6; l++) {
        const double s = s0 * (1 << l); const int n = 64 >> l;
        snprintf(e, sizeof e, "%s{\"path\":\"%d\",\"coordinateTransformations\":[{\"scale\":[%g,%g,%g],\"type\":\"scale\"}]}", l ? "," : "", l, s, s, s);
        strcat(ds, e);
        snprintf(p, sizeof p, "%s/%s/%d", dir, name, l); mkdir(p, 0755);
        snprintf(p, sizeof p, "%s/%s/%d/zarr.json", dir, name, l);
        char a[1024]; snprintf(a, sizeof a, "{\"zarr_format\":3,\"node_type\":\"array\",\"shape\":[%d,%d,%d],\"data_type\":\"uint8\",\"chunk_grid\":{\"name\":\"regular\","
            "\"configuration\":{\"chunk_shape\":[%d,%d,%d]}},\"chunk_key_encoding\":{\"name\":\"default\",\"configuration\":{\"separator\":\"/\"}},\"fill_value\":0,"
            "\"codecs\":[{\"name\":\"bytes\",\"configuration\":{}}],\"attributes\":{},\"dimension_names\":[\"z\",\"y\",\"x\"]}", n, n, n, n, n, n);
        wr(p, a);
    }
    char g[4096]; snprintf(g, sizeof g, "{\"zarr_format\":3,\"node_type\":\"group\",\"attributes\":{\"ome\":{\"version\":\"0.5\",\"multiscales\":[{\"version\":\"0.5\","
        "\"axes\":[{\"name\":\"z\",\"type\":\"space\"},{\"name\":\"y\",\"type\":\"space\"},{\"name\":\"x\",\"type\":\"space\"}],\"datasets\":[%s]}]}}}", ds);
    snprintf(p, sizeof p, "%s/%s/zarr.json", dir, name); wr(p, g);
}
static void check(store *st, const char *name, double um0, int level) {
    z3 *z = pyramid_open_level(st, name, level, um0, nullptr);
    const long got = z ? (long)z3_meta_of(z)->shape[0] : -1, want = 64 >> level;
    printf("%-4s %s um0 %.3f level %d: shape %ld (want %ld)\n", got == want ? "ok" : "FAIL", name, um0, level, got, want);
    fails += got != want;
    if (z) z3_close(z);
}
int main(int argc, char **argv) {
    char dir[1024]; snprintf(dir, sizeof dir, "%s", argc > 1 ? argv[1] : "/tmp/test_pyramid_rel"); mkdir(dir, 0755);
    group(dir, "rel", 1.0); group(dir, "abs", 7.91);
    store *st = store_open(dir); if (!st) { fprintf(stderr, "store_open %s\n", dir); return 2; }
    for (int l = 0; l < 3; l++) { check(st, "rel", 7.91, l); check(st, "rel", 2.4, l); check(st, "rel", 8.64, l); check(st, "abs", 7.91, l); }
    printf("test_pyramid_rel: %s\n", fails ? "FAIL" : "ok");
    return fails != 0;
}
