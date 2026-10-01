/* TIFF reader vs tifffile references, zarr v3 writer round-trip through the zarr3 reader, and (with
   UFSM_NET=1 + ~/huggingfacetoken) a zarr v2 chunk read from the HF bucket vs a numcodecs reference. */
#include "store.h"
#include "tiff.h"
#include "z3w.h"
#include "zarr2.h"
#include "zarr3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(cond, ...) do { if (!(cond)) { fails++; printf("FAIL: " __VA_ARGS__); printf("\n"); } } while (0)

static uint8_t *slurp(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb");
    if (!f) return nullptr;
    fseek(f, 0, SEEK_END); *n = (size_t)ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *b = malloc(*n + 1);
    if (fread(b, 1, *n, f) != *n) { free(b); fclose(f); return nullptr; }
    fclose(f);
    return b;
}

static void test_tiff(const char *tif, const char *ref, int expect_pages) {
    size_t rn;
    uint8_t *r = slurp(ref, &rn);
    tiff *t = tiff_open_file(tif);
    if (!t || !r) { printf("skip %s (%s)\n", tif, t ? "no ref" : tiff_error()); tiff_close(t); free(r); return; }
    tiff_page p;
    tiff_page_info(t, 0, &p);
    int np = tiff_npages(t);
    printf("%s: %d pages %dx%d bits %d spp %d fmt %d\n", tif, np, p.w, p.h, p.bits, p.spp, p.fmt);
    CHECK(np == expect_pages, "pages %d != %d", np, expect_pages);
    size_t page_bytes = (size_t)p.w * p.h * p.spp * (p.bits / 8);
    CHECK(rn == page_bytes * np, "ref size %zu != %zu", rn, page_bytes * np);
    uint8_t *buf = malloc(page_bytes);
    size_t bad = 0;
    for (int i = 0; i < np && rn == page_bytes * np; i++) {
        if (tiff_read_page(t, i, buf)) { CHECK(0, "read page %d: %s", i, tiff_error()); break; }
        for (size_t k = 0; k < page_bytes; k++) bad += buf[k] != r[(size_t)i * page_bytes + k];
    }
    printf("  mismatching bytes: %zu\n", bad);
    CHECK(bad == 0, "tiff content %s", tif);
    free(buf); free(r); tiff_close(t);
}

static void test_writer(const char *dir) {
    printf("z3w round trip\n");
    int64_t shape[3] = {200, 300, 260};
    int S = 256;
    z3w *w = z3w_create(dir, shape, S, 0.f, 0, "{\"test\":1}");
    if (!w) { CHECK(0, "create: %s", z3w_error()); return; }
    size_t sv = (size_t)S * S * S;
    uint8_t *data = malloc(sv);
    srand(7);
    /* 2 x 2 x 2 shards: fill with a pattern, leave one all-zero */
    for (int sz = 0; sz < 2; sz++) for (int sy = 0; sy < 2; sy++) for (int sx = 0; sx < 2; sx++) {
        for (size_t i = 0; i < sv; i++) data[i] = (sz == 0 && sy == 1 && sx == 1) ? 0 : (uint8_t)((i * 7 + sz * 13 + sy * 17 + sx * 19) % 251);
        CHECK(z3w_write_shard(w, sz, sy, sx, data, 4) == 0, "write shard: %s", z3w_error());
    }
    z3w_close(w);
    store *s = store_open(dir);
    z3 *z = z3_open(s, ".", nullptr);
    if (!z) { CHECK(0, "reopen: %s", z3_error()); return; }
    const z3_meta *m = z3_meta_of(z);
    CHECK(m->shape[0] == 200 && m->shard[0] == 256 && m->chunk[0] == 128 && m->q == 0, "meta");
    int64_t o[3] = {100, 200, 100}, n[3] = {100, 100, 160};
    uint8_t *got = malloc((size_t)n[0] * n[1] * n[2]);
    CHECK(z3_read(z, o, n, got, 4) == 0, "read: %s", z3_error());
    size_t bad = 0;
    for (int64_t zz = 0; zz < n[0]; zz++) for (int64_t yy = 0; yy < n[1]; yy++) for (int64_t xx = 0; xx < n[2]; xx++) {
        int64_t gz = o[0] + zz, gy = o[1] + yy, gx = o[2] + xx;
        int sz = (int)(gz / S), sy = (int)(gy / S), sx = (int)(gx / S);
        size_t i = ((size_t)(gz % S) * S + (gy % S)) * S + (gx % S);
        uint8_t exp = (sz == 0 && sy == 1 && sx == 1) ? 0 : (uint8_t)((i * 7 + sz * 13 + sy * 17 + sx * 19) % 251);
        bad += got[((size_t)zz * n[1] + yy) * n[2] + xx] != exp;
    }
    printf("  mismatching voxels: %zu\n", bad);
    CHECK(bad == 0, "round trip");
    /* missing shard reads as zero */
    int64_t o2[3] = {150, 260, 256}, n2[3] = {8, 8, 4};
    CHECK(z3_read(z, o2, n2, got, 1) == 0, "read missing shard");
    int nz = 0; for (int i = 0; i < 256; i++) nz += got[i] != 0;
    CHECK(nz == 0, "missing shard nonzero");
    z3_close(z); store_close(s); free(data); free(got);
}

static void test_zarr2_net(const char *ref) {
    size_t rn; uint8_t *r = slurp(ref, &rn);
    size_t tn; uint8_t *tok;
    char tp[512]; snprintf(tp, sizeof tp, "%s/huggingfacetoken", getenv("HOME"));
    tok = slurp(tp, &tn);
    if (!r || !tok) { printf("skip zarr2 net test\n"); free(r); free(tok); return; }
    while (tn && (tok[tn - 1] == '\n' || tok[tn - 1] == '\r' || tok[tn - 1] == ' ')) tn--;
    tok[tn] = 0;
    store *s = store_open("https://huggingface.co/buckets/scrollprize/datasets/resolve");
    store_set_bearer(s, (char *)tok);
    z2 *z = z2_open(s, "surfaces/2um_032726/0500p2_5217.zarr/0", nullptr);
    if (!z) { CHECK(0, "z2_open: %s", z2_error()); return; }
    const z2_meta *m = z2_meta_of(z);
    printf("zarr2 net: shape %lld x %lld x %lld chunk %d comp %d sep '%c'\n", (long long)m->shape[0], (long long)m->shape[1], (long long)m->shape[2], m->chunk[0], m->comp, m->sep);
    int64_t o[3] = {100 * 128, 61 * 128, 77 * 128}, n[3] = {128, 128, 128};
    uint8_t *got = malloc(128 * 128 * 128);
    CHECK(z2_read(z, o, n, got, 1) == 0, "z2_read: %s", z2_error());
    CHECK(rn == 128 * 128 * 128 && !memcmp(got, r, rn), "zarr2 chunk content vs numcodecs");
    printf("  chunk 100.61.77 matches reference: %s\n", rn == 128 * 128 * 128 && !memcmp(got, r, rn) ? "yes" : "NO");
    z2_close(z); store_close(s); free(got); free(r); free(tok);
}

int main(int argc, char **argv) {
    const char *S = argc > 1 ? argv[1] : "/tmp";
    char a[1024], b[1024];
    snprintf(a, sizeof a, "%s/lab1.tif", S); snprintf(b, sizeof b, "%s/lab1.raw", S); test_tiff(a, b, 320);
    snprintf(a, sizeof a, "%s/img1.tif", S); snprintf(b, sizeof b, "%s/img1.raw", S); test_tiff(a, b, 320);
    snprintf(a, sizeof a, "%s/x.tif", S); snprintf(b, sizeof b, "%s/x.raw", S); test_tiff(a, b, 1);
    snprintf(a, sizeof a, "%s/z3w_test.zarr", S); test_writer(a);
    if (getenv("UFSM_NET")) { snprintf(b, sizeof b, "%s/c_blosc.raw", S); test_zarr2_net(b); }
    printf(fails ? "%d FAILURES\n" : "formats ok\n", fails);
    return fails != 0;
}
