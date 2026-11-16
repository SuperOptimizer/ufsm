/* test_labcomp: the labcomp label codec (src/labcomp.{c,h}) and ufsm labcomp-pack (src/labcomp_cli.c).
   (a) kind 1 (winding codes) round trips byte for byte on synthetic 128^3 chunks: sheets with one code each (the codec's case), sheets
       plus scattered foreign codes, every voxel a random nonzero byte (the literal path), single voxels at the corners, all zero;
   (b) truncated / bit-flipped streams decode to an error or to some bytes, never past their buffers (the loop ends), lbc_kind refuses
       foreign streams;
   (c) labcomp-pack: a volcomp-lossless sharded array (partial edge chunks, an all-fill shard) re-encoded; z3_read of the packed array
       equals the source everywhere, the zarr.json names the codec, an existing OUT is refused, an unknown kind fails to open. CPU only. */
#include "labcomp.h"
#include "store.h"
#include "z3w.h"
#include "zarr3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

int cmd_labcomp_pack(int argc, char **argv);
static int fails = 0;
static void check(const char *what, int ok) { printf("labcomp: %-104s %s\n", what, ok ? "ok" : "FAIL"); fails += !ok; }
static uint64_t lcg(uint64_t *s) { *s = *s * 6364136223846793005ull + 1442695040888963407ull; return *s >> 17; }
enum { C = LBC_CHUNK, V = LBC_VOXELS };

/* sheets: curved surfaces z ~ f(y, x) + k * pitch, 2 voxels thick, code 1 + k (the codes rasters' layout: one code per sheet patch) */
static void sheets(uint8_t *v, int64_t oz, int64_t oy, int64_t ox, int nz, int ny, int nx, int foreign, uint64_t seed) {
    uint64_t s = seed;
    for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++) {
        const double gz = (double)(oz + z), gy = (double)(oy + y), gx = (double)(ox + x);
        const double t = gz + 6 * sin(gy * 0.05) + 4 * cos(gx * 0.04), k = floor(t / 9.0), f = t - 9.0 * k;
        uint8_t c = f < 2 ? (uint8_t)(1 + ((int)k % 250 + 250) % 250) : 0;
        if (foreign && c && lcg(&s) % 1000 < 5) c = (uint8_t)(1 + lcg(&s) % 255);   /* 0.5 %: another code (patch seams, label conflicts) */
        v[((size_t)z * ny + y) * nx + x] = c;
    }
}
static int roundtrip(const char *what, const uint8_t *src, size_t *bytes) {
    uint8_t *enc = nullptr, *dec = malloc(V); size_t n = 0; int ok = 0;
    if (dec && !lbc_codes_encode(src, &enc, &n)) { memset(dec, 0xAB, V); ok = !lbc_decode(enc, n, dec) && !memcmp(dec, src, V) && lbc_kind(enc, n) == LBC_KIND_CODES; }
    size_t nz = 0; for (size_t i = 0; i < V; i++) nz += src[i] != 0;
    char line[200]; snprintf(line, sizeof line, "%s: exact (%zu B, %.2fx vs raw, %.3f bits / nonzero voxel)", what, n, n ? (double)V / n : 0, nz ? 8.0 * n / nz : 0);
    check(line, ok); if (bytes) *bytes = n;
    free(enc); free(dec); return ok;
}

int main(void) {
    uint8_t *v = malloc(V); uint64_t s = 11;
    /* (a) */
    size_t nsh = 0;
    sheets(v, 0, 0, 0, C, C, C, 0, 1); roundtrip("sheets, one code each", v, &nsh);
    check("sheets cost < 1/40 of their raw bytes", nsh > 0 && nsh < V / 40);
    {   /* speed (printed, not checked): the readers decode a chunk on every miss */
        uint8_t *e = nullptr, *d = malloc(V); size_t en = 0; struct timespec t0, t1, t2;
        clock_gettime(CLOCK_MONOTONIC, &t0); for (int r = 0; r < 4; r++) { free(e); e = nullptr; lbc_codes_encode(v, &e, &en); }
        clock_gettime(CLOCK_MONOTONIC, &t1); for (int r = 0; r < 4; r++) lbc_decode(e, en, d);
        clock_gettime(CLOCK_MONOTONIC, &t2);
        const double te = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec), td = (t2.tv_sec - t1.tv_sec) + 1e-9 * (t2.tv_nsec - t1.tv_nsec);
        printf("labcomp: sheets chunk, one thread: encode %.0f Mvox/s, decode %.0f Mvox/s\n", 4.0 * V / te / 1e6, 4.0 * V / td / 1e6);
        free(e); free(d);
    }
    sheets(v, 300, 1000, 50, C, C, C, 1, 2); roundtrip("sheets + 0.5 % foreign codes", v, nullptr);
    for (size_t i = 0; i < V; i++) v[i] = (uint8_t)(1 + lcg(&s) % 255); roundtrip("every voxel a random nonzero byte", v, nullptr);
    memset(v, 0, V); v[0] = 7; v[V - 1] = 255; v[(size_t)64 * C * C + 5] = 1; roundtrip("three single voxels (corners)", v, nullptr);
    memset(v, 0, V); roundtrip("all zero", v, nullptr);
    memset(v, 200, V); roundtrip("all one code", v, nullptr);
    /* (b) */
    sheets(v, 0, 0, 0, C, C, C, 1, 3);
    uint8_t *enc = nullptr, *dec = malloc(V); size_t n = 0;
    if (lbc_codes_encode(v, &enc, &n)) { check("encode for the corruption checks", 0); return 1; }
    int ret = 0, errs = 0, wrong = 0;
    for (int k = 0; k < 60; k++) {
        uint8_t *b = malloc(n); memcpy(b, enc, n); size_t m = n;
        if (k < 20) m = (size_t)(lcg(&s) % n);   /* truncated */
        else { const int nf = 1 + k % 4; for (int j = 0; j < nf; j++) b[8 + lcg(&s) % (n - 8)] ^= (uint8_t)(1u << (lcg(&s) % 8)); }   /* bit flips after the header */
        const int r = lbc_decode(b, m, dec); ret++; errs += r != 0; wrong += !r && memcmp(dec, v, V);
        free(b);
    }
    char line[200]; snprintf(line, sizeof line, "60 truncated / bit-flipped streams return (%d errors, %d decoded other bytes)", errs, wrong); check(line, ret == 60);
    const char junk[16] = "VCMPxxxxxxxxxxxx"; check("lbc_kind / lbc_decode refuse a foreign stream", lbc_kind(junk, sizeof junk) == -1 && lbc_decode(junk, sizeof junk, dec) == -1);
    check("lbc_kind refuses a short stream", lbc_kind(enc, 5) == -1);
    free(enc);
    /* (c) */
    char tmpl[] = "/tmp/test_labcomp_XXXXXX"; char *dir = mkdtemp(tmpl); if (!dir) return 1;
    char in[300], out[300]; snprintf(in, sizeof in, "%s/in.zarr", dir); snprintf(out, sizeof out, "%s/out.zarr", dir);
    const int64_t sh[3] = {300, 260, 520}; const int SH = 256;   /* shards 2 x 2 x 3; edge chunks partial */
    z3w *w = z3w_create(in, sh, SH, 0, 0, "{\"ufsm\":{\"kind\":\"codes\"}}"); check("z3w_create (volcomp lossless)", w != nullptr); if (!w) return 1;
    uint8_t *shard = malloc((size_t)SH * SH * SH);
    for (int64_t sz = 0; sz < 2; sz++) for (int64_t sy = 0; sy < 2; sy++) for (int64_t sx = 0; sx < 3; sx++) {
        if (sz == 1 && sy == 1 && sx == 2) continue;   /* an absent shard */
        sheets(shard, sz * SH, sy * SH, sx * SH, SH, SH, SH, 1, (uint64_t)(sz * 9 + sy * 3 + sx + 5));
        if (sz == 0 && sy == 1 && sx == 0) memset(shard, 0, (size_t)SH * SH * 128);   /* fill chunks inside a stored shard */
        if (z3w_write_shard(w, sz, sy, sx, shard, 2)) { check("z3w_write_shard", 0); return 1; }
    }
    z3w_close(w); free(shard);
    char *av[] = {"ufsm", "labcomp-pack", in, out, "--threads", "3", nullptr};
    check("labcomp-pack (verified)", cmd_labcomp_pack(6, av) == 0);
    check("labcomp-pack refuses an existing OUT", cmd_labcomp_pack(6, av) == 1);
    store *st = store_open(dir); z3 *za = st ? z3_open(st, "in.zarr", nullptr) : nullptr, *zb = st ? z3_open(st, "out.zarr", nullptr) : nullptr;
    check("open both arrays", za && zb);
    if (za && zb) {
        check("the packed array's meta: labcomp kind codes, no volcomp, same shape", z3_meta_of(zb)->lbc == LBC_KIND_CODES && z3_meta_of(zb)->q < 0
              && !memcmp(z3_meta_of(za)->shape, z3_meta_of(zb)->shape, sizeof z3_meta_of(za)->shape));
        const size_t nv = (size_t)sh[0] * sh[1] * sh[2]; uint8_t *ra = malloc(nv), *rb = malloc(nv); const int64_t o[3] = {0, 0, 0};
        struct timespec t0, t1, t2; clock_gettime(CLOCK_MONOTONIC, &t0);
        const int ra_ok = z3_read(za, o, sh, ra, 1) == 0; clock_gettime(CLOCK_MONOTONIC, &t1);
        const int rr = ra_ok && z3_read(zb, o, sh, rb, 1) == 0; clock_gettime(CLOCK_MONOTONIC, &t2);
        printf("labcomp: z3_read, one thread: volcomp lossless %.0f Mvox/s, labcomp %.0f Mvox/s\n", nv / ((t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec)) / 1e6,
               nv / ((t2.tv_sec - t1.tv_sec) + 1e-9 * (t2.tv_nsec - t1.tv_nsec)) / 1e6);
        size_t nz = 0, diff = 0; for (size_t i = 0; i < nv; i++) { nz += ra[i] != 0; diff += ra[i] != rb[i]; }
        snprintf(line, sizeof line, "z3_read of the packed array = the source (%zu voxels, %zu nonzero, %zu differ)", nv, nz, diff);
        check(line, rr && diff == 0 && nz > nv / 10);
        const int64_t o2[3] = {37, 129, 255}, n2[3] = {200, 100, 140}; const size_t n2v = (size_t)n2[0] * n2[1] * n2[2];
        const int r2 = z3_read(za, o2, n2, ra, 1) == 0 && z3_read(zb, o2, n2, rb, 1) == 0;
        check("an unaligned box across chunk and shard borders: equal", r2 && !memcmp(ra, rb, n2v));
        free(ra); free(rb);
    }
    if (za) z3_close(za);
    if (zb) z3_close(zb);
    char zp[400], cmd[900]; snprintf(zp, sizeof zp, "%s/zarr.json", out);
    FILE *f = fopen(zp, "rb"); char zj[1 << 14] = {0}; if (f) { (void)!fread(zj, 1, sizeof zj - 1, f); fclose(f); }
    check("zarr.json: the labcomp codec and the labcomp record, the source attributes kept", strstr(zj, "\"name\":\"labcomp\"") && strstr(zj, "\"labcomp\":{\"kind\":\"codes\"") && strstr(zj, "\"ufsm\""));
    char *k = strstr(zj, "\"kind\":\"codes\"}}"); if (k) memcpy(k, "\"kind\":\"phzzz\"}}", 16);
    f = fopen(zp, "wb"); if (f) { fputs(zj, f); fclose(f); }
    z3 *zc = st ? z3_open(st, "out.zarr", nullptr) : nullptr; check("an unknown labcomp kind fails to open", zc == nullptr); if (zc) z3_close(zc);
    if (st) store_close(st);
    snprintf(cmd, sizeof cmd, "rm -rf '%s'", dir); if (strncmp(dir, "/tmp/test_labcomp_", 18) == 0 && system(cmd)) {}
    free(v); free(dec);
    if (fails) { printf("test_labcomp: %d FAILED\n", fails); return 1; }
    printf("test_labcomp: ok\n");
    return 0;
}
