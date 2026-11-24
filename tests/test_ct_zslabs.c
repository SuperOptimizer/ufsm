/* test_ct_zslabs: a ct_only source's "ct_zslabs" keep only the draw cells whose level-0 z range meets a slab (sample.c label_occupancy;
   ufsm prefetch reads the same index). Fixture: a local 2.4 um group, levels 0..5 (512 >> l), level 5 (16^3) all papyrus, the finer levels
   metadata only. No slabs: 4096 cells; [[128, 256]]: level-5 rows 4..7 = 1024; [[130, 140], [480, 512]]: rows 4 and 15 = 512. Bad entries
   ([z0 >= z1], no ct_only) are load errors. CPU only. */
#include "sample.h"
#include "sources.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static int fails;
static void wr(const char *p, const char *s) { FILE *f = fopen(p, "w"); if (!f) { perror(p); exit(2); } fputs(s, f); fclose(f); }
static void mkd(const char *p) { char b[1024]; snprintf(b, sizeof b, "%s", p); for (char *c = b + 1; *c; c++) if (*c == '/') { *c = 0; mkdir(b, 0755); *c = '/'; } mkdir(b, 0755); }
static void fixture(const char *dir) {
    char p[1024], ds[2048] = "", e[256];
    for (int l = 0; l < 6; l++) {
        const int n = 512 >> l, ch = n < 128 ? n : 128; const double s = 2.4 * (1 << l);
        snprintf(e, sizeof e, "%s{\"path\":\"%d\",\"coordinateTransformations\":[{\"scale\":[%g,%g,%g],\"type\":\"scale\"}]}", l ? "," : "", l, s, s, s); strcat(ds, e);
        snprintf(p, sizeof p, "%s/ct/s.zarr/%d", dir, l); mkd(p);
        snprintf(p, sizeof p, "%s/ct/s.zarr/%d/zarr.json", dir, l);
        char a[1024]; snprintf(a, sizeof a, "{\"zarr_format\":3,\"node_type\":\"array\",\"shape\":[%d,%d,%d],\"data_type\":\"uint8\",\"chunk_grid\":{\"name\":\"regular\","
            "\"configuration\":{\"chunk_shape\":[%d,%d,%d]}},\"chunk_key_encoding\":{\"name\":\"default\",\"configuration\":{\"separator\":\"/\"}},\"fill_value\":0,"
            "\"codecs\":[{\"name\":\"bytes\",\"configuration\":{}}],\"attributes\":{},\"dimension_names\":[\"z\",\"y\",\"x\"]}", n, n, n, ch, ch, ch);
        wr(p, a);
    }
    snprintf(p, sizeof p, "%s/ct/s.zarr/5/c/0/0", dir); mkd(p);   /* level 5: one 16^3 chunk of papyrus (200 >= iso 64) */
    snprintf(p, sizeof p, "%s/ct/s.zarr/5/c/0/0/0", dir);
    FILE *f = fopen(p, "wb"); static uint8_t v[16 * 16 * 16]; memset(v, 200, sizeof v); fwrite(v, 1, sizeof v, f); fclose(f);
    char g[4096]; snprintf(g, sizeof g, "{\"zarr_format\":3,\"node_type\":\"group\",\"attributes\":{\"ome\":{\"version\":\"0.5\",\"multiscales\":[{\"version\":\"0.5\","
        "\"axes\":[{\"name\":\"z\",\"type\":\"space\"},{\"name\":\"y\",\"type\":\"space\"},{\"name\":\"x\",\"type\":\"space\"}],\"datasets\":[%s]}]}}}", ds);
    snprintf(p, sizeof p, "%s/ct/s.zarr/zarr.json", dir); wr(p, g);
}
static long cells(const char *dir, const char *extra, int expect_load) {
    char p[1024], j[2048]; snprintf(p, sizeof p, "%s/src.json", dir);
    snprintf(j, sizeof j, "{\"sources\":[{\"name\":\"s\",\"root\":\"%s/ct\",\"ct\":\"s.zarr\",\"um\":2.4,\"iso\":64,\"min_level\":0,\"max_level\":1,\"weight\":1%s}]}", dir, extra);
    wr(p, j);
    sources *S = sources_load(p);
    if (!S) { if (expect_load) { printf("FAIL load %s\n", extra); fails++; } else printf("ok   refused %s\n", extra); return -1; }
    if (!expect_load) { printf("FAIL accepted %s\n", extra); fails++; sources_free(S); return -1; }
    occ_index o = source_occupancy(&S->src[0]); const long n = (long)o.n; free(o.idx); sources_free(S);
    return n;
}
static void want(const char *dir, const char *extra, long w) {
    const long n = cells(dir, extra, 1); printf("%-4s %-40s %ld cells (want %ld)\n", n == w ? "ok" : "FAIL", extra, n, w); fails += n != w;
}
int main(int argc, char **argv) {
    char dir[1024]; snprintf(dir, sizeof dir, "%s", argc > 1 ? argv[1] : "/tmp/test_ct_zslabs"); mkd(dir); fixture(dir);
    want(dir, ",\"ct_only\":1", 4096);
    want(dir, ",\"ct_only\":1,\"ct_zslabs\":[[128,256]]", 1024);
    want(dir, ",\"ct_only\":1,\"ct_zslabs\":[[130,140],[480,512]]", 512);
    cells(dir, ",\"ct_only\":1,\"ct_zslabs\":[[256,128]]", 0);
    cells(dir, ",\"ct_zslabs\":[[0,128]]", 0);
    printf("test_ct_zslabs: %s\n", fails ? "FAIL" : "ok");
    return fails != 0;
}
