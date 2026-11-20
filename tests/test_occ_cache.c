/* tests/test_occ_cache.c: the sampler's occupancy cache for on-the-fly fibre sources (src/occcache.c): a saved index loads back exactly;
   any input change (a fibre parameter, the polyline file's size / mtime, an extra file, the CT key) is a miss; a truncated file is a miss;
   sources without a fibre target are not cached. CPU only. */
#include "occcache.h"
#include "dcache.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", m); fails++; } } while (0)
static void put(const char *p, const char *txt) { FILE *f = fopen(p, "wb"); fputs(txt, f); fclose(f); }

int main(int argc, char **argv) {
    char d[512], cache[600], afv[600], ext[600], key[8192], key2[8192];
    snprintf(d, sizeof d, "%s/occ.XXXXXX", argc > 1 ? argv[1] : "/tmp"); if (!mkdtemp(d)) { perror("mkdtemp"); return 1; }
    snprintf(cache, sizeof cache, "%s/cache", d); snprintf(afv, sizeof afv, "%s/a.afv", d); snprintf(ext, sizeof ext, "%s/x.afv", d);
    put(afv, "fibres v1"); put(ext, "extra v1");
    source s = {0}; s.name = "src"; s.ct_key = "S/v.zarr"; s.um = 2.4; s.min_level = 2; s.max_level = 2; s.iso = 64;
    struct fib_spec f = {.afv = afv, .extra = "", .rad = 2, .guard = 2, .ring = 3, .iso = 64, .piso = 64, .plane = 0};
    char xs[700]; snprintf(xs, sizeof xs, "%s@map", ext); f.extra = xs;
    s.tgt_fib[0] = &f;
    CHECK(!occ_cache_key(&s, key, sizeof key), "a fibre source has a key");
    uint32_t idx[5] = {3, 17, 99, 1000, 4095}; occ_index o = {idx, 5, 5, {16, 16, 16}}, r = {0};
    CHECK(occ_cache_load(cache, key, &r), "empty cache: a miss");
    CHECK(!occ_cache_save(cache, key, &o), "save");
    CHECK(!occ_cache_load(cache, key, &r) && r.n == 5 && r.lev == 5 && r.shape[0] == 16 && r.shape[2] == 16 && !memcmp(r.idx, idx, sizeof idx), "load = saved");
    free(r.idx); r.idx = nullptr;
    /* each input change: another key */
    f.rad = 2.5; CHECK(!occ_cache_key(&s, key2, sizeof key2) && strcmp(key, key2) && occ_cache_load(cache, key2, &r), "radius changes the key"); f.rad = 2;
    f.plane = 1; occ_cache_key(&s, key2, sizeof key2); CHECK(strcmp(key, key2), "plane changes the key"); f.plane = 0;
    s.ct_key = "S/w.zarr"; occ_cache_key(&s, key2, sizeof key2); CHECK(strcmp(key, key2), "CT changes the key"); s.ct_key = "S/v.zarr";
    put(ext, "extra v2, longer"); occ_cache_key(&s, key2, sizeof key2); CHECK(strcmp(key, key2), "an extra file's content changes the key");
    put(afv, "fibres v2, longer"); occ_cache_key(&s, key2, sizeof key2); CHECK(strcmp(key, key2) && occ_cache_load(cache, key2, &r), "the polyline file changes the key");
    /* a truncated index file: a miss */
    CHECK(!occ_cache_save(cache, key2, &o), "save 2");
    { char kd[1200], p[1300]; dcache_key_dir(cache, key2, kd, sizeof kd); snprintf(p, sizeof p, "%s/occ.bin", kd); CHECK(!truncate(p, 4 + 4 + 24 + 8 + 8), "truncate"); }
    CHECK(occ_cache_load(cache, key2, &r), "a truncated file is a miss");
    /* no fibre target: not cached; no cache dir: a miss / no-op */
    s.tgt_fib[0] = nullptr; CHECK(occ_cache_key(&s, key2, sizeof key2), "no fibre target: no key");
    CHECK(occ_cache_load(nullptr, key, &r) && occ_cache_save("", key, &o), "no dir");
    char cmd[700]; snprintf(cmd, sizeof cmd, "rm -rf -- '%s'", d); if (strstr(d, "/occ.")) (void)!system(cmd);
    if (fails) { fprintf(stderr, "test_occ_cache: %d failed\n", fails); return 1; }
    printf("test_occ_cache: ok\n"); return 0;
}
