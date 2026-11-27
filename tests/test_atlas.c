/* src/parquet.c + src/atlas.c (ATLAS-15k import / audit, 2026-11-27): a Parquet file written here in the ATLAS schema (pyarrow's nesting
   for list<list<float32>>: optional group / repeated list / optional element, twice), with every path the reader takes on the real
   files and the others it claims: two row groups, v1 data pages (SNAPPY: x, GZIP: y), a DATA_PAGE_V2 (ZSTD: z), dictionary-encoded strings
   (RLE_DICTIONARY), a ZSTD float scalar, an ignored list<int32> column, no area_cm2 (optional), a null float element (-> -1), a node with -1 on one axis only
   (-> invalid on all three, DEFAULTED). atlas-import's tifxyz must hold exactly the written grids; snappy's copy elements are checked on a
   hand-made stream. atlas-audit: two flat GT sheets (q 0 at z 100, q 1 at z 114): a patch on one sheet passes, a patch that steps
   from one sheet to the other fails with its switch edges counted, a patch far from both is "nocover", the holdout box is listed. */
#include "parquet.h"
#include "imgops.h"
#include "tiffw.h"
#include "z3w.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <zlib.h>
#include <zstd.h>

int cmd_atlas_import(int argc, char **argv);
int cmd_atlas_audit(int argc, char **argv);
int cmd_atlas_rvscore(int argc, char **argv);

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

/* ---- a byte buffer + Thrift compact writer ---- */
typedef struct { uint8_t *b; size_t n, cap; } bb;
static void put(bb *o, const void *p, size_t n) { if (o->n + n > o->cap) { o->cap = (o->n + n) * 2 + 256; o->b = realloc(o->b, o->cap); } memcpy(o->b + o->n, p, n); o->n += n; }
static void put8(bb *o, uint8_t v) { put(o, &v, 1); }
static void put32le(bb *o, uint32_t v) { uint8_t t[4] = {v, v >> 8, v >> 16, v >> 24}; put(o, t, 4); }
static void uvar(bb *o, uint64_t v) { while (v >= 128) { put8(o, (uint8_t)(v | 128)); v >>= 7; } put8(o, (uint8_t)v); }
static void zz(bb *o, int64_t v) { uvar(o, ((uint64_t)v << 1) ^ (uint64_t)(v >> 63)); }
typedef struct { int last[16]; int d; } tw;   /* field-id stack */
static void fh(bb *o, tw *t, int id, int type) { const int dl = id - t->last[t->d]; if (dl > 0 && dl <= 15) put8(o, (uint8_t)(dl << 4 | type)); else { put8(o, (uint8_t)type); zz(o, id); } t->last[t->d] = id; }
static void fi32(bb *o, tw *t, int id, int64_t v) { fh(o, t, id, 5); zz(o, v); }
static void fi64(bb *o, tw *t, int id, int64_t v) { fh(o, t, id, 6); zz(o, v); }
static void fbin(bb *o, tw *t, int id, const char *s) { fh(o, t, id, 8); uvar(o, strlen(s)); put(o, s, strlen(s)); }
static void fbool(bb *o, tw *t, int id, int v) { fh(o, t, id, v ? 1 : 2); }
static void sbeg(bb *o, tw *t, int id) { fh(o, t, id, 12); t->last[++t->d] = 0; }
static void send(bb *o, tw *t) { put8(o, 0); t->d--; }
static void lhead(bb *o, int n, int et) { if (n < 15) put8(o, (uint8_t)(n << 4 | et)); else { put8(o, (uint8_t)(0xf0 | et)); uvar(o, (uint64_t)n); } }
static void lbeg_struct(bb *o, tw *t, int id, int n) { fh(o, t, id, 9); lhead(o, n, 12); }
static void elem_beg(tw *t) { t->last[++t->d] = 0; }

/* ---- RLE hybrid of small values (runs only) and codecs ---- */
static void rle(bb *o, const uint8_t *v, size_t n, int bw) {
    const int vb = (bw + 7) / 8;
    for (size_t i = 0; i < n;) { size_t j = i; while (j < n && v[j] == v[i]) j++; uvar(o, (uint64_t)(j - i) << 1); for (int k = 0; k < vb; k++) put8(o, k ? 0 : v[i]); i = j; }
}
static void rle_bitpacked(bb *o, const uint32_t *v, size_t n, int bw) {   /* one bit-packed run (n padded to 8) */
    const size_t g = (n + 7) / 8; uvar(o, g << 1 | 1);
    const size_t nb = g * bw; uint8_t *t = calloc(nb, 1);
    for (size_t k = 0; k < n; k++) for (int b = 0; b < bw; b++) if (v[k] >> b & 1) { const size_t q = k * bw + b; t[q >> 3] |= (uint8_t)(1 << (q & 7)); }
    put(o, t, nb); free(t);
}
static void snappy_lit(bb *o, const uint8_t *p, size_t n) {   /* literal-only snappy */
    uvar(o, n);
    for (size_t i = 0; i < n;) { size_t l = n - i < 65536 ? n - i : 65536; put8(o, 61 << 2); put8(o, (uint8_t)(l - 1)); put8(o, (uint8_t)((l - 1) >> 8)); put(o, p + i, l); i += l; }
}
static void gz(bb *o, const uint8_t *p, size_t n) {
    z_stream z; memset(&z, 0, sizeof z); deflateInit2(&z, 6, Z_DEFLATED, 16 + MAX_WBITS, 8, Z_DEFAULT_STRATEGY);
    const size_t cap = deflateBound(&z, n) + 64; uint8_t *t = malloc(cap);
    z.next_in = (Bytef *)p; z.avail_in = (uInt)n; z.next_out = t; z.avail_out = (uInt)cap; deflate(&z, Z_FINISH); put(o, t, z.total_out); deflateEnd(&z); free(t);
}
static void zs(bb *o, const uint8_t *p, size_t n) { const size_t cap = ZSTD_compressBound(n); uint8_t *t = malloc(cap); const size_t r = ZSTD_compress(t, cap, p, n, 3); put(o, t, r); free(t); }
static void pq_compress(int codec, bb *o, const uint8_t *p, size_t n) { if (codec == 1) snappy_lit(o, p, n); else if (codec == 2) gz(o, p, n); else if (codec == 6) zs(o, p, n); else put(o, p, n); }

/* ---- the test data: 3 patches (2 + 1 per row group) ---- */
enum { NP = 3 };
static const char *ids[NP] = {"PHercT-x10-y20-z30", "PHercT-x11-y21-z31", "PHercU-x1-y2-z3"}, *scrolls[NP] = {"PHercT", "PHercT", "PHercU"};
static const char *scans[NP] = {"https://example.invalid/vol/PHercT/20250101000000-8.640um-x.zarr/0", "https://example.invalid/vol/PHercT/20250101000000-8.640um-x.zarr/0",
                                "https://example.invalid/vol/PHercU/20250202000000-9.362um-y.zarr/"};
static const float ums[NP] = {8.64f, 8.64f, 9.362f};
static const int R[NP] = {3, 2, 4}, C[NP] = {4, 5, 3};
static float val(int p, int a, int r, int c) {
    if (p == 0 && r == 1 && c == 2) return -1;                 /* a hole on all axes */
    if (p == 1 && r == 0 && c == 1 && a == 1) return -1;       /* -1 on y only: mixed */
    return 100.f * (a + 1) + 10.f * r + c + 0.25f * p;
}
static int is_null(int p, int a, int r, int c) { return p == 2 && a == 0 && r == 3 && c == 2; }   /* a null element of x */

typedef struct { int64_t off, nval, usz, csz, dict_off; int codec, type; const char *path[5]; int np; } chunkinfo;

/* one column chunk of patches [p0, p1) */
static void write_col(bb *f, int col, int p0, int p1, chunkinfo *ci) {
    const int codecs[8] = {0, 0, 0, 6, 0, 1, 2, 6}; const int codec = codecs[col];
    ci->codec = codec; ci->off = (int64_t)f->n; ci->dict_off = -1;
    static const char *names[8] = {"id", "scroll", "scan", "voxel_um", "centre_xyz", "x", "y", "z"};
    bb lev = {0}, vals = {0}; uint8_t *rep = malloc(4096), *def = malloc(4096); size_t nl = 0;
    if (col <= 2) {   /* dictionary-encoded strings: a dictionary page, then RLE_DICTIONARY indices */
        ci->type = 6; ci->path[0] = names[col]; ci->np = 1;
        const char **S = col == 0 ? ids : col == 1 ? scrolls : scans;
        const char *dict[NP]; int nd = 0; uint32_t idx[NP];
        for (int p = p0; p < p1; p++) { int k = 0; while (k < nd && strcmp(dict[k], S[p])) k++; if (k == nd) dict[nd++] = S[p]; idx[p - p0] = (uint32_t)k; def[nl++] = 1; }
        bb d = {0}; for (int k = 0; k < nd; k++) { put32le(&d, (uint32_t)strlen(dict[k])); put(&d, dict[k], strlen(dict[k])); }
        bb hd = {0}; tw t = {0}; ci->dict_off = (int64_t)f->n;
        fi32(&hd, &t, 1, 2); fi32(&hd, &t, 2, (int64_t)d.n); fi32(&hd, &t, 3, (int64_t)d.n); sbeg(&hd, &t, 7); fi32(&hd, &t, 1, nd); fi32(&hd, &t, 2, 0); send(&hd, &t); put8(&hd, 0);
        put(f, hd.b, hd.n); put(f, d.b, d.n); free(hd.b); free(d.b);
        ci->off = (int64_t)f->n;
        put8(&vals, 2); rle_bitpacked(&vals, idx, (size_t)(p1 - p0), 2);
    } else if (col == 3) {
        ci->type = 4; ci->path[0] = names[col]; ci->np = 1;
        for (int p = p0; p < p1; p++) { put(&vals, &ums[p], 4); def[nl++] = 1; }
    } else if (col == 4) {   /* list<int32>, ignored by the import */
        ci->type = 1; ci->path[0] = names[col]; ci->path[1] = "list"; ci->path[2] = "element"; ci->np = 3;
        for (int p = p0; p < p1; p++) for (int k = 0; k < 3; k++) { rep[nl] = k ? 1 : 0; def[nl++] = 3; const int32_t v = p * 10 + k; put(&vals, &v, 4); }
    } else {
        const int a = col - 5; ci->type = 4; ci->path[0] = names[col]; ci->path[1] = "list"; ci->path[2] = "element"; ci->path[3] = "list"; ci->path[4] = "element"; ci->np = 5;
        for (int p = p0; p < p1; p++) for (int r = 0; r < R[p]; r++) for (int c = 0; c < C[p]; c++) {
            rep[nl] = !r && !c ? 0 : !c ? 1 : 2;
            if (is_null(p, a, r, c)) def[nl++] = 4; else { def[nl++] = 5; const float v = val(p, a, r, c); put(&vals, &v, 4); }
        }
    }
    const int maxdef = col <= 3 ? 1 : col == 4 ? 3 : 5, maxrep = col <= 3 ? 0 : col == 4 ? 1 : 2;
    const int bwd = maxdef >= 4 ? 3 : maxdef >= 2 ? 2 : 1, bwr = maxrep >= 2 ? 2 : 1;
    bb rl = {0}, dl = {0}; if (maxrep) rle(&rl, rep, nl, bwr); rle(&dl, def, nl, bwd);
    const int v2 = col == 7; const int enc = col <= 2 ? 8 : 0;
    bb body = {0}, hd = {0}; tw t = {0};
    if (!v2) {
        if (maxrep) { put32le(&lev, (uint32_t)rl.n); put(&lev, rl.b, rl.n); }
        put32le(&lev, (uint32_t)dl.n); put(&lev, dl.b, dl.n); put(&lev, vals.b, vals.n);
        pq_compress(codec, &body, lev.b, lev.n);
        fi32(&hd, &t, 1, 0); fi32(&hd, &t, 2, (int64_t)lev.n); fi32(&hd, &t, 3, (int64_t)body.n);
        sbeg(&hd, &t, 5); fi32(&hd, &t, 1, (int64_t)nl); fi32(&hd, &t, 2, enc); fi32(&hd, &t, 3, 3); fi32(&hd, &t, 4, 3); send(&hd, &t); put8(&hd, 0);
        ci->usz = (int64_t)lev.n;
    } else {
        bb cv = {0}; pq_compress(codec, &cv, vals.b, vals.n);
        put(&body, rl.b, rl.n); put(&body, dl.b, dl.n); put(&body, cv.b, cv.n);
        int nnull = 0; for (size_t k = 0; k < nl; k++) nnull += def[k] != maxdef;
        fi32(&hd, &t, 1, 3); fi32(&hd, &t, 2, (int64_t)(rl.n + dl.n + vals.n)); fi32(&hd, &t, 3, (int64_t)body.n);
        sbeg(&hd, &t, 8); fi32(&hd, &t, 1, (int64_t)nl); fi32(&hd, &t, 2, nnull); fi32(&hd, &t, 3, p1 - p0); fi32(&hd, &t, 4, enc);
        fi32(&hd, &t, 5, (int64_t)dl.n); fi32(&hd, &t, 6, (int64_t)rl.n); fbool(&hd, &t, 7, 1); send(&hd, &t); put8(&hd, 0);
        ci->usz = (int64_t)(rl.n + dl.n + vals.n); free(cv.b);
    }
    put(f, hd.b, hd.n); put(f, body.b, body.n);
    ci->nval = (int64_t)nl; ci->csz = (int64_t)(f->n - (size_t)(ci->dict_off >= 0 ? ci->dict_off : ci->off));
    free(hd.b); free(body.b); free(rl.b); free(dl.b); free(lev.b); free(vals.b); free(rep); free(def);
}

static void schema_elem(bb *o, int type, int rep, const char *name, int nchild) {
    tw t = {0}; elem_beg(&t); t.d = 0;
    if (type >= 0) fi32(o, &t, 1, type);
    if (rep >= 0) fi32(o, &t, 3, rep);
    fbin(o, &t, 4, name);
    if (nchild > 0) fi32(o, &t, 5, nchild);
    put8(o, 0);
}

static int write_parquet(const char *path) {
    bb f = {0}; put(&f, "PAR1", 4);
    const int rg[2][2] = {{0, 2}, {2, 3}}; chunkinfo ci[2][8];
    for (int g = 0; g < 2; g++) for (int c = 0; c < 8; c++) write_col(&f, c, rg[g][0], rg[g][1], &ci[g][c]);
    bb m = {0}; tw t = {0};
    fi32(&m, &t, 1, 1);
    /* schema: root, 3 strings, voxel_um, centre_xyz (3 elements), x / y / z (5 each) */
    fh(&m, &t, 2, 9); lhead(&m, 1 + 4 + 3 + 15, 12);
    schema_elem(&m, -1, -1, "schema", 8);
    schema_elem(&m, 6, 1, "id", 0); schema_elem(&m, 6, 1, "scroll", 0); schema_elem(&m, 6, 1, "scan", 0); schema_elem(&m, 4, 1, "voxel_um", 0);
    schema_elem(&m, -1, 1, "centre_xyz", 1); schema_elem(&m, -1, 2, "list", 1); schema_elem(&m, 1, 1, "element", 0);
    for (int a = 0; a < 3; a++) { const char nm[2] = {(char)('x' + a), 0};
        schema_elem(&m, -1, 1, nm, 1); schema_elem(&m, -1, 2, "list", 1); schema_elem(&m, -1, 1, "element", 1); schema_elem(&m, -1, 2, "list", 1); schema_elem(&m, 4, 1, "element", 0); }
    fi64(&m, &t, 3, NP);
    lbeg_struct(&m, &t, 4, 2);
    for (int g = 0; g < 2; g++) {
        tw r = {0};
        lbeg_struct(&m, &r, 1, 8);
        for (int c = 0; c < 8; c++) {
            tw cc = {0}; const chunkinfo *k = &ci[g][c];
            fi64(&m, &cc, 2, k->off);
            sbeg(&m, &cc, 3);
            fi32(&m, &cc, 1, k->type);
            fh(&m, &cc, 2, 9); lhead(&m, 2, 5); zz(&m, 0); zz(&m, 3);
            fh(&m, &cc, 3, 9); lhead(&m, k->np, 8); for (int q = 0; q < k->np; q++) { uvar(&m, strlen(k->path[q])); put(&m, k->path[q], strlen(k->path[q])); }
            fi32(&m, &cc, 4, k->codec); fi64(&m, &cc, 5, k->nval); fi64(&m, &cc, 6, k->usz); fi64(&m, &cc, 7, k->csz); fi64(&m, &cc, 9, k->off);
            if (k->dict_off >= 0) fi64(&m, &cc, 11, k->dict_off);
            send(&m, &cc);
            put8(&m, 0);
        }
        fi64(&m, &r, 2, 1000); fi64(&m, &r, 3, rg[g][1] - rg[g][0]);
        put8(&m, 0);
    }
    fbin(&m, &t, 6, "ufsm test_atlas");
    put8(&m, 0);
    put(&f, m.b, m.n); put32le(&f, (uint32_t)m.n); put(&f, "PAR1", 4);
    FILE *o = fopen(path, "wb"); if (!o) return -1;
    const int ok = fwrite(f.b, 1, f.n, o) == f.n; fclose(o); free(f.b); free(m.b); return ok ? 0 : -1;
}

static void test_snappy(void) {
    /* "abcd" literal, copy-1 offset 4 length 8 (overlapping), literal "Z", copy-2 offset 9 length 3 */
    const uint8_t s[] = {16, 3 << 2, 'a', 'b', 'c', 'd', (uint8_t)(((8 - 4) << 2) | 1), 4, 0, 'Z', (uint8_t)(((3 - 1) << 2) | 2), 9, 0};
    uint8_t o[32]; size_t n = 0;
    CHECK(!pq_snappy_decode(s, sizeof s, o, sizeof o, &n) && n == 16 && !memcmp(o, "abcdabcdabcdZabc", 16), "snappy copy elements (%zu: %.16s)", n, o);
    const uint8_t bad[] = {8, 0 << 2, 'a', (uint8_t)(((8 - 4) << 2) | 1), 5};   /* offset past the output */
    CHECK(pq_snappy_decode(bad, sizeof bad, o, sizeof o, &n), "snappy: a copy before the start is refused");
}

static void write_plane(const char *dir, double z, float q, int n) {
    float *X = malloc(sizeof(float) * n * n), *Y = malloc(sizeof(float) * n * n), *Z = malloc(sizeof(float) * n * n), *Q = malloc(sizeof(float) * n * n);
    for (int r = 0; r < n; r++) for (int c = 0; c < n; c++) { const int i = r * n + c; X[i] = (float)c; Y[i] = (float)r; Z[i] = (float)z; Q[i] = q + 0.001f * c; }
    tifxyz_write(dir, n, n, X, Y, Z, 1, 8.64, -1, "test"); char p[1024]; snprintf(p, sizeof p, "%s/q.tif", dir); tiffw_write(p, n, n, 3, Q, 1);
    free(X); free(Y); free(Z); free(Q);
}
static void write_patch(const char *root, const char *scroll, const char *id, double z0, double z1, double xo) {
    enum { W = 40, H = 30 }; float X[W * H], Y[W * H], Z[W * H];
    for (int r = 0; r < H; r++) for (int c = 0; c < W; c++) { const int i = r * W + c; X[i] = (float)(xo + 2 * c); Y[i] = (float)(20 + 2 * r); Z[i] = (float)(c < W / 2 ? z0 : z1); }
    char d[1024]; snprintf(d, sizeof d, "%s/%s/%s.tifxyz", root, scroll, id); tifxyz_write(d, W, H, X, Y, Z, 2, 8.64, -1, "test");
}

static void write_vol(const char *dir, uint8_t (*f)(int z, int y, int x)) {
    const int64_t sh[3] = {128, 128, 128}; z3w *w = z3w_create(dir, sh, 128, 0, 0, nullptr); uint8_t *d = malloc(128 * 128 * 128);
    for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) for (int x = 0; x < 128; x++) d[((size_t)z * 128 + y) * 128 + x] = f(z, y, x);
    if (!w || z3w_write_shard(w, 0, 0, 0, d, 1)) fprintf(stderr, "FAIL: z3w %s\n", dir), fails++;
    z3w_close(w); free(d);
}
static uint8_t v_ct(int z, int y, int x) { (void)z; (void)y; (void)x; return 100; }
static uint8_t v_ph(int z, int y, int x) { (void)y; (void)x; return z < 4 ? 255 : z < 64 ? 50 : 200; }
static uint8_t v_rvz(int z, int y, int x) { (void)y; (void)x; return z < 60 ? 200 : 50; }
static uint8_t v_rvx(int z, int y, int x) { (void)z; (void)y; return x > 64 ? 200 : 50; }
static uint8_t v_codes(int z, int y, int x) { (void)z; (void)y; return x == 65 ? 7 : 0; }
/* atlas-rvscore: a GT box (side: recto recall 56 / 60, four recto rows predicted verso; F1 of rvx's recto face x = 65..66 vs the GT sheet
   x = 65: 1; a probability npy with its sheet at x = 67: F1 1 at the best threshold, no surface above 200) and an ATLAS patch on the plane x = 64,
   the axis at x = 0 */
static void test_rvscore(const char *tmp) {
    char ct[600], ph[600], rz[600], rx[600], cd[600]; snprintf(ct, sizeof ct, "%s/ct", tmp); snprintf(ph, sizeof ph, "%s/ph", tmp); snprintf(rz, sizeof rz, "%s/rvz", tmp); snprintf(rx, sizeof rx, "%s/rvx", tmp); snprintf(cd, sizeof cd, "%s/codes", tmp);
    write_vol(ct, v_ct); write_vol(ph, v_ph); write_vol(rz, v_rvz); write_vol(rx, v_rvx); write_vol(cd, v_codes);
    char out[700]; snprintf(out, sizeof out, "%s/rvs.txt", tmp);
    char ad[700], ix[700], ids[700], axf[700], npy[700]; snprintf(ad, sizeof ad, "%s/rv/PHercR/p1.tifxyz", tmp); snprintf(ix, sizeof ix, "%s/rv/index.tsv", tmp); snprintf(ids, sizeof ids, "%s/rv/ids.txt", tmp); snprintf(axf, sizeof axf, "%s/rv/axis.json", tmp); snprintf(npy, sizeof npy, "%s/p.npy", tmp);
    enum { W = 40, H = 40 }; static float X[W * H], Y[W * H], Z[W * H];
    for (int r = 0; r < H; r++) for (int c = 0; c < W; c++) { X[r * W + c] = 64; Y[r * W + c] = (float)(10 + 2 * c); Z[r * W + c] = (float)(10 + 2 * r); }
    tifxyz_write(ad, W, H, X, Y, Z, 2, 8.64, -1, "test");
    FILE *f = fopen(ix, "w"); fprintf(f, "id\tscroll\tscan\tum\trows\tcols\tvalid\tquads\tarea_cm2\tarea_cm2_atlas\tx0\ty0\tz0\tx1\ty1\tz1\np1\tPHercR\tR.zarr\t8.64\t40\t40\t1600\t1521\t0\t0\t64\t10\t10\t64\t88\t88\n"); fclose(f);
    f = fopen(ids, "w"); fprintf(f, "p1\n"); fclose(f);
    f = fopen(axf, "w"); fprintf(f, "{\"control_points\": [{\"z\": 0, \"y\": 50, \"x\": 0}, {\"z\": 200, \"y\": 50, \"x\": 0}]}\n"); fclose(f);
    {   /* the npy: 128^3 uint8, 200 on x = 67, 1 elsewhere (0 = no prediction) */
        f = fopen(npy, "wb"); char hd[128]; int hl = snprintf(hd, sizeof hd, "{'descr': '|u1', 'fortran_order': False, 'shape': (128, 128, 128), }");
        while ((10 + hl + 1) % 64) hd[hl++] = ' ';
        hd[hl++] = '\n'; const unsigned char mg[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, (unsigned char)(hl & 255), (unsigned char)(hl >> 8)};
        fwrite(mg, 1, 10, f); fwrite(hd, 1, (size_t)hl, f);
        uint8_t *d = malloc(128 * 128 * 128); for (size_t i = 0; i < 128u * 128 * 128; i++) d[i] = i % 128 == 67 ? 200 : 1;
        fwrite(d, 1, 128u * 128 * 128, f); fclose(f); free(d);
    }
    char *a1[] = {"ufsm", "atlas-rvscore", "--rv", (char *)tmp, "rvz", "--box", "16,16,16,96,96,96", "--phase", (char *)tmp, "ph", "--ct", (char *)tmp, "ct", "--axis", axf, "--tag", "box", "--out", out};
    const int r1 = cmd_atlas_rvscore((int)(sizeof a1 / sizeof *a1), a1);
    char *a2[] = {"ufsm", "atlas-rvscore", "--rv", (char *)tmp, "rvx", "--patches", ix, "--ids", ids, "--axis", axf, "--tag", "atlas", "--out", out,
                  "--box", "16,16,16,96,96,96", "--phase", (char *)tmp, "ph", "--codes", (char *)tmp, "codes"};
    const int r2 = cmd_atlas_rvscore((int)(sizeof a2 / sizeof *a2), a2);
    char *a3[] = {"ufsm", "atlas-rvscore", "--prob", npy, "--prob-origin", "0,0,0", "--box", "16,16,16,96,96,96", "--phase", (char *)tmp, "ph", "--codes", (char *)tmp, "codes",
                  "--patches", ix, "--ids", ids, "--axis", axf, "--tag", "npy", "--out", out};
    const int r3 = cmd_atlas_rvscore((int)(sizeof a3 / sizeof *a3), a3);
    char *a4[] = {"ufsm", "atlas-rvscore", "--prob", npy, "--prob-origin", "0,0,0", "--box", "16,16,16,96,96,96", "--phase", (char *)tmp, "ph", "--codes", (char *)tmp, "codes", "--thr", "0.9", "--tag", "npy9", "--out", out};
    const int r4 = cmd_atlas_rvscore((int)(sizeof a4 / sizeof *a4), a4);
    CHECK(r1 == 0 && r2 == 0 && r3 == 0 && r4 == 1, "atlas-rvscore rc %d %d %d %d", r1, r2, r3, r4);
    f = fopen(out, "r"); char ln[2048]; int ok1 = 0, ok2 = 0, ok3 = 0, ok4 = 0, ok5 = 0, ok6 = 0;
    while (f && fgets(ln, sizeof ln, f)) {
        double rr, vr, bal, o, iv, b, dm, rc2, P, R, F;
        const char *p = strstr(ln, "recto_recall\t"); if (p && sscanf(p, "recto_recall\t%lf\tverso_recall\t%lf\tbalanced\t%lf", &rr, &vr, &bal) == 3) ok1 = fabs(rr - 44.0 / 48) < 1e-3 && vr == 1;
        p = strstr(ln, "outer_recto\t"); if (p && sscanf(p, "outer_recto\t%lf\tinner_verso\t%lf\tboth\t%lf\tmean_out_minus_in\t%lf\trecall2\t%lf", &o, &iv, &b, &dm, &rc2) == 5) ok2 = o == 1 && iv == 1 && b == 1 && fabs(dm - 150) < 1e-6 && rc2 == 1;
        p = strstr(ln, "\tP2\t");
        if (p && sscanf(p, "\tP2\t%lf\tR2\t%lf\tF2\t%lf", &P, &R, &F) == 3) { if (strstr(ln, "\tatlas\t")) ok3 = F == 1; if (strstr(ln, "\tnpy\t")) ok4 = F == 1; }
        if (strstr(ln, "\tnpy\tpatches\t") && (p = strstr(ln, "recall2\t")) && sscanf(p, "recall2\t%lf", &rc2) == 1) ok5 = rc2 == 0;   /* x = 67 is 3 voxels from the nodes */
        if (strstr(ln, "\tnpy9\tf1\t") && strstr(ln, "pred_vox\t0")) ok6 = 1;
    }
    if (f) fclose(f);
    CHECK(ok1 && ok2 && ok3 && ok4 && ok5 && ok6, "atlas-rvscore rows (%d %d %d %d %d %d), see %s", ok1, ok2, ok3, ok4, ok5, ok6, out);
}

int main(void) {
    char tmp[] = "/tmp/test_atlas_XXXXXX"; if (!mkdtemp(tmp)) { perror("mkdtemp"); return 1; }
    test_snappy();
    test_rvscore(tmp);
    char pq[512], out[512]; snprintf(pq, sizeof pq, "%s/t.parquet", tmp); snprintf(out, sizeof out, "%s/out", tmp);
    CHECK(!write_parquet(pq), "write the test parquet");
    pq_file *f = pq_open(pq);
    CHECK(f, "pq_open: %s", pq_error());
    if (f) {
        CHECK(pq_num_rows(f) == NP && pq_num_row_groups(f) == 2 && pq_num_leaves(f) == 8, "metadata: %lld rows, %d groups, %d leaves", (long long)pq_num_rows(f), pq_num_row_groups(f), pq_num_leaves(f));
        const pq_leaf *L = pq_leaf_at(f, pq_find_leaf(f, "x"));
        CHECK(L->max_def == 5 && L->max_rep == 2 && L->rep_def[0] == 2 && L->rep_def[1] == 4 && !strcmp(L->path, "x.list.element.list.element"), "x leaf levels %d %d %d %d %s", L->max_def, L->max_rep, L->rep_def[0], L->rep_def[1], L->path);
        pq_col c; CHECK(!pq_read_col(f, 0, pq_find_leaf(f, "centre_xyz"), &c) && c.n == 6 && c.nv == 6 && ((int32_t *)c.vals)[4] == 11, "list<int32> column");
        pq_col_free(&c);
        CHECK(!pq_read_col(f, 1, pq_find_leaf(f, "x"), &c) && c.n == 12 && c.nv == 11, "x of row group 1 with a null: %lld / %lld", (long long)c.n, (long long)c.nv);
        pq_col_free(&c);
        pq_close(f);
    }
    char *av[] = {"ufsm", "atlas-import", out, pq, "--threads", "2"};
    CHECK(cmd_atlas_import(6, av) == 0, "atlas-import rc");
    for (int p = 0; p < NP; p++) {
        char d[1024]; float *X[3]; int w, h;
        for (int a = 0; a < 3; a++) { snprintf(d, sizeof d, "%s/%s/%s.tifxyz/%c.tif", out, scrolls[p], ids[p], 'x' + a); X[a] = img_read_f32(d, &w, &h); CHECK(X[a] && w == C[p] && h == R[p], "%s axis %c: %dx%d", ids[p], 'x' + a, w, h); }
        if (!X[0] || !X[1] || !X[2]) continue;
        int bad = 0;
        for (int r = 0; r < R[p]; r++) for (int c2 = 0; c2 < C[p]; c2++) {
            int hole = 0; float e[3];
            for (int a = 0; a < 3; a++) { e[a] = is_null(p, a, r, c2) ? -1 : val(p, a, r, c2); if (e[a] < 0) hole = 1; }
            for (int a = 0; a < 3; a++) if (X[a][r * C[p] + c2] != (hole ? -1.f : e[a])) bad++;
        }
        CHECK(!bad, "%s: %d node values differ", ids[p], bad);
        for (int a = 0; a < 3; a++) free(X[a]);
        snprintf(d, sizeof d, "%s/%s/%s.tifxyz/meta.json", out, scrolls[p], ids[p]);
        FILE *m = fopen(d, "r"); char buf[4096] = ""; if (m) { buf[fread(buf, 1, sizeof buf - 1, m)] = 0; fclose(m); }
        const char *sn = p < 2 ? "\"scan\": \"20250101000000-8.640um-x.zarr\"" : "\"scan\": \"20250202000000-9.362um-y.zarr\"";
        CHECK(strstr(buf, sn) && strstr(buf, "\"scale\": [0.5, 0.5]") && strstr(buf, p < 2 ? "\"voxel_size_um\": 8.64" : "\"voxel_size_um\": 9.362"), "%s meta: %s", ids[p], buf);
    }
    char idx[600]; snprintf(idx, sizeof idx, "%s/index.tsv", out);
    FILE *ix = fopen(idx, "r"); int lines = 0; char ln[4096]; while (ix && fgets(ln, sizeof ln, ix)) lines++; if (ix) fclose(ix);
    CHECK(lines == NP + 1, "index.tsv: %d lines", lines);

    /* audit: GT sheets q 0 (z 100) and q 1 (z 114) */
    char gt[600], ar[600], ho[600], res[600]; snprintf(gt, sizeof gt, "%s/gt", tmp); snprintf(ar, sizeof ar, "%s/at", tmp);
    char p1[700]; snprintf(p1, sizeof p1, "%s/s0.tifxyz", gt); write_plane(p1, 100, 0, 160); snprintf(p1, sizeof p1, "%s/s1.tifxyz", gt); write_plane(p1, 114, 1, 160);
    snprintf(p1, sizeof p1, "%s/PHercV", ar); char mk[800]; snprintf(mk, sizeof mk, "mkdir -p '%s'", p1); CHECK(!system(mk), "mkdir");
    write_patch(ar, "PHercV", "good", 100.6, 100.6, 0); write_patch(ar, "PHercV", "step", 100.4, 113.7, 0); write_patch(ar, "PHercV", "far", 160, 160, 0); write_patch(ar, "PHercV", "boxed", 100.6, 100.6, 80);
    snprintf(idx, sizeof idx, "%s/index.tsv", ar); ix = fopen(idx, "w");
    fprintf(ix, "id\tscroll\tscan\tum\trows\tcols\tvalid\tquads\tarea_cm2\tarea_cm2_atlas\tx0\ty0\tz0\tx1\ty1\tz1\n");
    fprintf(ix, "good\tPHercV\tV.zarr\t8.64\t30\t40\t1200\t1131\t0\t0\t0\t20\t100.6\t78\t78\t100.6\n");
    fprintf(ix, "step\tPHercV\tV.zarr\t8.64\t30\t40\t1200\t1131\t0\t0\t0\t20\t100.4\t78\t78\t113.7\n");
    fprintf(ix, "far\tPHercV\tW.zarr\t8.64\t30\t40\t1200\t1131\t0\t0\t0\t20\t160\t78\t78\t160\n");
    fprintf(ix, "boxed\tPHercV\tV.zarr\t8.64\t30\t40\t1200\t1131\t0\t0\t80\t20\t100.6\t158\t78\t100.6\n");
    fclose(ix);
    snprintf(ho, sizeof ho, "%s/ho.json", tmp); FILE *hf = fopen(ho, "w");
    fprintf(hf, "{\"boxes\": [{\"name\": \"B1\", \"ct\": \"PHercV/V.zarr\", \"l0\": [90, 0, 120, 20, 64, 64]}, {\"name\": \"B2\", \"ct\": \"PHercV/other.zarr\", \"l0\": [0, 0, 0, 999, 999, 999]}]}\n"); fclose(hf);
    snprintf(res, sizeof res, "%s/audit.tsv", tmp);
    char *aa[] = {"ufsm", "atlas-audit", idx, "--ref", gt, "--value", "q.tif", "--holdouts", ho, "--out", res, "--threads", "2"};
    CHECK(cmd_atlas_audit(13, aa) == 0, "atlas-audit rc");
    FILE *rf = fopen(res, "r"); int seen = 0;
    while (rf && fgets(ln, sizeof ln, rf)) {
        char id[64], sc[64], gate[32], hold[64]; size_t nodes, matched, onsheet, amb, edges, sw, s5; double cover, p50, p90, le2, le4, swr, big; int set;
        if (sscanf(ln, "%63s %63s %zu %zu %lf %lf %lf %lf %lf %zu %zu %zu %zu %lf %zu %lf %d %31s %63s", id, sc, &nodes, &matched, &cover, &p50, &p90, &le2, &le4, &onsheet, &amb, &edges, &sw, &swr, &s5, &big, &set, gate, hold) != 19) continue;
        if (!strcmp(id, "good")) { seen++; CHECK(!strcmp(gate, "pass") && fabs(p50 - 0.6) < 0.01 && sw == 0 && s5 == 1 && big > 0.999 && !strcmp(hold, "-"), "good: %s", ln); }
        if (!strcmp(id, "step")) { seen++; CHECK(!strcmp(gate, "fail") && sw == 30 && s5 == 2 && fabs(big - 0.5) < 0.01 && edges == 30 * 39 + 29 * 40, "step: %s", ln); }
        if (!strcmp(id, "boxed")) { seen++; CHECK(!strcmp(gate, "held") && !strcmp(hold, "B1"), "boxed: %s", ln); }
        if (!strcmp(id, "far")) { seen++; CHECK(!strcmp(gate, "nocover") && matched == 0 && !strcmp(hold, "-"), "far: %s", ln); }
    }
    if (rf) fclose(rf);
    CHECK(seen == 4, "audit rows: %d", seen);
    /* the gate as a filter: --allow (far dropped), --write-q: the kept patch with q = K + theta / 2 pi in the GT gauge */
    char ax[600], al[600], qd[600]; snprintf(ax, sizeof ax, "%s/axis.json", tmp); snprintf(al, sizeof al, "%s/allow.txt", tmp); snprintf(qd, sizeof qd, "%s/kept", tmp);
    FILE *af = fopen(ax, "w"); fprintf(af, "{\"control_points\": [{\"z\": 0, \"y\": 50, \"x\": -100000}, {\"z\": 1000, \"y\": 50, \"x\": -100000}]}\n"); fclose(af);
    af = fopen(al, "w"); fprintf(af, "good\nstep\nboxed\n"); fclose(af);
    char *ab[] = {"ufsm", "atlas-audit", idx, "--ref", gt, "--value", "q.tif", "--holdouts", ho, "--out", res, "--threads", "2", "--allow", al, "--write-q", qd, "--axis", ax, "--gate-amb", "0.05"};
    CHECK(cmd_atlas_audit(21, ab) == 0, "atlas-audit --write-q rc");
    rf = fopen(res, "r"); int dropped = 0; while (rf && fgets(ln, sizeof ln, rf)) if (!strncmp(ln, "far\t", 4) && strstr(ln, "\tdropped\t")) dropped = 1; if (rf) fclose(rf);
    CHECK(dropped, "far: not dropped by --allow");
    snprintf(p1, sizeof p1, "%s/step.tifxyz/q.tif", qd); CHECK(access(p1, F_OK), "a failing patch was written");
    snprintf(p1, sizeof p1, "%s/boxed.tifxyz/q.tif", qd); CHECK(access(p1, F_OK), "a held patch was written");
    snprintf(p1, sizeof p1, "%s/good.tifxyz/q.tif", qd); int qw = 0, qh = 0; float *Qg = img_read_f32(p1, &qw, &qh);
    snprintf(p1, sizeof p1, "%s/good.tifxyz/x.tif", qd); int xw = 0, xh = 0; float *Xg = img_read_f32(p1, &xw, &xh);
    CHECK(Qg && Xg && qw == 40 && qh == 30 && xw == 40, "good: q.tif / x.tif");
    if (Qg && Xg) { double e = 0; for (int i = 0; i < qw * qh; i++) e = fmax(e, fabs(Qg[i] - 0.001 * Xg[i])); CHECK(e < 0.05, "good: q off the GT gauge by %.3f turns", e); }
    free(Qg); free(Xg);
    if (!fails) { snprintf(mk, sizeof mk, "rm -rf '%s'", tmp); if (system(mk)) fprintf(stderr, "cleanup failed\n"); }
    if (fails) { fprintf(stderr, "test_atlas: %d failures (kept %s)\n", fails, tmp); return 1; }
    printf("test_atlas: OK\n");
    return 0;
}
