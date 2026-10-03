/* ufsm ingest-*: re-export upstream ground truth into volcomp zarr v3 pyramids and surfcomp .sfc files.

   Exported label encoding (uint8, lossless q=0): 0 = background, 254 = surface (recto), 255 = ignore;
   coarser levels hold round(254 * surface fraction) over the non-ignore children, or 255 when at least
   half of the children are ignore. */
#include "hf.h"
#include "json.h"
#include "store.h"
#include "tiff.h"
#include "z3w.h"
#include "zarr2.h"
#include "zarr3.h"
#include "zipr.h"
#include "pyramid.h"
#include <math.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "surfcomp.h"
#define MAXLEV_PYR 12

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}

/* ---- tiny parallel-for ---- */
typedef void (*pf_fn)(int i, int tid, void *ud);
typedef struct { pf_fn fn; void *ud; int n; atomic_int next; int tid; } pf_job;
typedef struct { pf_job *j; int tid; } pf_arg;
static void *pf_worker(void *a) {
    pf_arg *pa = a;
    for (;;) { int i = atomic_fetch_add(&pa->j->next, 1); if (i >= pa->j->n) break; pa->j->fn(i, pa->tid, pa->j->ud); }
    return nullptr;
}
static void parallel_for(int n, int nthreads, pf_fn fn, void *ud) {
    if (nthreads > n) nthreads = n;
    if (nthreads < 1) nthreads = 1;
    pf_job j = {fn, ud, n, 0, 0};
    pthread_t th[256]; pf_arg args[256];
    if (nthreads > 256) nthreads = 256;
    for (int t = 0; t < nthreads; t++) { args[t] = (pf_arg){&j, t}; pthread_create(&th[t], nullptr, pf_worker, &args[t]); }
    for (int t = 0; t < nthreads; t++) pthread_join(th[t], nullptr);
}

/* ---- label pooling: 2x2x2 -> 1 ---- */
static uint8_t pool_label8(const uint8_t *v) {
    int nig = 0, sum = 0, cnt = 0;
    for (int i = 0; i < 8; i++) { if (v[i] == 255) nig++; else { sum += v[i]; cnt++; } }
    if (nig >= 4) return 255;
    return (uint8_t)((sum + cnt / 2) / cnt);
}
/* in: n^3, out: (n/2)^3 */
static void pool2_labels(const uint8_t *in, int n, uint8_t *out) {
    int h = n / 2;
    for (int z = 0; z < h; z++) for (int y = 0; y < h; y++) for (int x = 0; x < h; x++) {
        const uint8_t *p = in + ((size_t)(2 * z) * n + 2 * y) * n + 2 * x;
        uint8_t v[8] = {p[0], p[1], p[n], p[n + 1], p[(size_t)n * n], p[(size_t)n * n + 1], p[(size_t)n * n + n], p[(size_t)n * n + n + 1]};
        out[((size_t)z * h + y) * h + x] = pool_label8(v);
    }
}
static void pool2_mean(const uint8_t *in, int n, uint8_t *out) {
    int h = n / 2;
    for (int z = 0; z < h; z++) for (int y = 0; y < h; y++) for (int x = 0; x < h; x++) {
        const uint8_t *p = in + ((size_t)(2 * z) * n + 2 * y) * n + 2 * x;
        unsigned s = p[0] + p[1] + p[n] + p[n + 1] + p[(size_t)n * n] + p[(size_t)n * n + 1] + p[(size_t)n * n + n] + p[(size_t)n * n + n + 1];
        out[((size_t)z * h + y) * h + x] = (uint8_t)((s + 4) >> 3);
    }
}

/* Preserve every surface touched by a 2x2x2 cell. All pyramid levels stay binary. */
static void pool2_mask(const uint8_t *in, int n, uint8_t *out) {
    int h = n / 2; size_t plane = (size_t)n * n;
    for (int z = 0; z < h; z++) for (int y = 0; y < h; y++) for (int x = 0; x < h; x++) {
        const uint8_t *p = in + ((size_t)(2 * z) * n + 2 * y) * n + 2 * x;
        out[((size_t)z * h + y) * h + x] = (p[0] | p[1] | p[n] | p[n + 1] |
                                          p[plane] | p[plane + 1] | p[plane + n] | p[plane + n + 1]) ? 255 : 0;
    }
}

/* ---- pyramid: build level l (>= 1) of `dir` from level l-1 (both our own zarr3 stores) ---- */
typedef struct { z3 *zp; z3w *w; int shard, labels, nthreads; int64_t ns[3]; int64_t shape[3]; atomic_int failed, done; } bl_job;
static void bl_shard(int si, int tid, void *ud) {
    bl_job *j = ud;
    int shard = j->shard;
    int64_t sx = si % j->ns[2], sy = (si / j->ns[2]) % j->ns[1], sz = si / (j->ns[2] * j->ns[1]);
    size_t sv = (size_t)shard * shard * shard;
    const int fill = j->labels == 1 ? 255 : 0;
    uint8_t *out = malloc(sv), *in = malloc(sv), *piece = malloc(sv / 8);
    memset(out, fill, sv);
    int any = 0;
    const z3_meta *pm = z3_meta_of(j->zp);
    for (int oz = 0; oz < 2; oz++) for (int oy = 0; oy < 2; oy++) for (int ox = 0; ox < 2; ox++) {
        int64_t o[3] = {(sz * 2 + oz) * shard, (sy * 2 + oy) * shard, (sx * 2 + ox) * shard}, n[3] = {shard, shard, shard};
        if (o[0] >= pm->shape[0] || o[1] >= pm->shape[1] || o[2] >= pm->shape[2]) continue;
        int pres = z3_shard_present(j->zp, sz * 2 + oz, sy * 2 + oy, sx * 2 + ox);
        if (pres < 0) { atomic_store(&j->failed, 1); break; }
        if (!pres) continue;
        if (z3_read(j->zp, o, n, in, j->nthreads)) { fprintf(stderr, "build_level: read: %s\n", z3_error()); atomic_store(&j->failed, 1); break; }
        if (j->labels == 2) pool2_mask(in, shard, piece);
        else if (j->labels == 1) pool2_labels(in, shard, piece); else pool2_mean(in, shard, piece);
        int h = shard / 2;
        for (int z = 0; z < h; z++) for (int y = 0; y < h; y++)
            memcpy(out + ((size_t)(oz * h + z) * shard + (oy * h + y)) * shard + ox * h, piece + ((size_t)z * h + y) * h, (size_t)h);
        any = 1;
    }
    if (any && !atomic_load(&j->failed) && z3w_write_shard(j->w, sz, sy, sx, out, j->nthreads)) { fprintf(stderr, "build_level: %s\n", z3w_error()); atomic_store(&j->failed, 1); }
    free(out); free(in); free(piece);
    atomic_fetch_add(&j->done, 1);
}

int pyramid_build_level(const char *group_dir, double um0, int l, const int64_t shape0[3], int shard, float q, int labels, int nthreads, const char *attrs) {
    const int fill = labels == 1 ? 255 : 0;
    char cur[1400], lv[32];
    z3w_level_name(um0 * (1 << l), lv, sizeof lv);
    snprintf(cur, sizeof cur, "%s/%s", group_dir, lv);
    int64_t shape[3];
    for (int d = 0; d < 3; d++) shape[d] = (shape0[d] + (1 << l) - 1) >> l;
    store *s = store_open(group_dir);
    char pk[64]; z3w_level_name(um0 * (1 << (l - 1)), pk, sizeof pk);
    z3 *zp = z3_open(s, pk, nullptr);
    if (!zp) { fprintf(stderr, "build_level: cannot open %s/%s: %s\n", group_dir, pk, z3_error()); return -1; }
    z3w *w = labels == 2 ? z3w_create_mask(cur, shape, shard, attrs) : z3w_create(cur, shape, shard, q, fill, attrs);
    if (!w) { fprintf(stderr, "build_level: %s\n", z3w_error()); return -1; }
    bl_job j = {zp, w, shard, labels, 4, {0, 0, 0}, {shape[0], shape[1], shape[2]}, 0, 0};
    for (int d = 0; d < 3; d++) j.ns[d] = (shape[d] + shard - 1) / shard;
    int par = nthreads / 4; if (par < 1) par = 1; if (par > 8) par = 8;   /* 8 shards in flight x 4 decode threads */
    double t0 = now();
    parallel_for((int)(j.ns[0] * j.ns[1] * j.ns[2]), par, bl_shard, &j);
    fprintf(stderr, "level %d: %lld x %lld x %lld in %.0fs\n", l, (long long)shape[0], (long long)shape[1], (long long)shape[2], now() - t0);
    z3w_close(w); z3_close(zp); store_close(s);
    return atomic_load(&j.failed) ? -1 : 0;
}

int pyramid_write_group(const char *dir, double um0, int nlev, const char *name, const char *attrs) {
    double um[MAXLEV_PYR];
    for (int l = 0; l < nlev; l++) um[l] = um0 * (1 << l);
    return z3w_write_group(dir, um, nlev, name, attrs);
}

/* ================= ingest-labels ================= */
typedef struct { int64_t (*c)[3]; int n, cap; char sep; char prefix[1024]; } chunklist;
static int on_chunk(const char *name, int is_dir, int64_t size, void *ud) {
    chunklist *cl = ud;
    long long z, y, x;
    if (is_dir || sscanf(name, "%lld.%lld.%lld", &z, &y, &x) != 3) return 0;
    if (cl->n == cl->cap) { cl->cap = cl->cap ? cl->cap * 2 : 1024; cl->c = realloc(cl->c, (size_t)cl->cap * sizeof *cl->c); }
    cl->c[cl->n][0] = z; cl->c[cl->n][1] = y; cl->c[cl->n][2] = x; cl->n++;
    return 0;
}
/* nested '/' layout: list z dirs, y dirs, x files */
typedef struct { chunklist *cl; store *s; const char *key; long long z, y; int depth; } nested;
static int on_nested(const char *name, int is_dir, int64_t size, void *ud) {
    nested *nd = ud;
    long long v;
    if (sscanf(name, "%lld", &v) != 1) return 0;
    if (nd->depth == 2) { chunklist *cl = nd->cl; if (cl->n == cl->cap) { cl->cap = cl->cap ? cl->cap * 2 : 1024; cl->c = realloc(cl->c, (size_t)cl->cap * sizeof *cl->c); } cl->c[cl->n][0] = nd->z; cl->c[cl->n][1] = nd->y; cl->c[cl->n][2] = v; cl->n++; return 0; }
    nested sub = *nd; sub.depth++;
    if (nd->depth == 0) sub.z = v; else sub.y = v;
    char k[1400];
    if (nd->depth == 0) snprintf(k, sizeof k, "%s/%lld", nd->key, v); else snprintf(k, sizeof k, "%s/%lld/%lld", nd->key, nd->z, v);
    store_list(nd->s, k, on_nested, &sub);
    return 0;
}

typedef struct { z2 *z; const int64_t (*chunks)[3]; int nchunks; uint8_t *shard; int S; int64_t so[3]; int labels; atomic_int failed; } fill_job;
static void fill_chunk(int i, int tid, void *ud) {
    fill_job *j = ud;
    const z2_meta *m = z2_meta_of(j->z);
    int64_t o[3] = {j->chunks[i][0] * m->chunk[0], j->chunks[i][1] * m->chunk[1], j->chunks[i][2] * m->chunk[2]}, n[3] = {m->chunk[0], m->chunk[1], m->chunk[2]};
    uint8_t *buf = malloc((size_t)n[0] * n[1] * n[2]);
    if (z2_read(j->z, o, n, buf, 1)) { fprintf(stderr, "chunk %lld,%lld,%lld: %s\n", (long long)j->chunks[i][0], (long long)j->chunks[i][1], (long long)j->chunks[i][2], z2_error()); atomic_store(&j->failed, 1); free(buf); return; }
    for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) {
        int64_t gz = o[0] + z - j->so[0], gy = o[1] + y - j->so[1], gx = o[2] - j->so[2];
        if (gz < 0 || gz >= j->S || gy < 0 || gy >= j->S) continue;
        uint8_t *dst = j->shard + ((size_t)gz * j->S + gy) * j->S + gx;
        const uint8_t *src = buf + ((size_t)z * n[1] + y) * n[2];
        for (int64_t x = 0; x < n[2] && gx + x < j->S; x++) {
            uint8_t v = src[x];
            dst[x] = j->labels ? (v == 2 ? 255 : v ? 254 : 0) : v;
        }
    }
    free(buf);
}

static int cmp3(const void *a, const void *b) { const int64_t *p = a, *q = b; for (int d = 0; d < 3; d++) if (p[d] != q[d]) return p[d] < q[d] ? -1 : 1; return 0; }

int cmd_ingest_labels(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm ingest-labels <root> <zarr-key> <out-dir> --um U [--levels L] [--token FILE] [--cache DIR] [--threads N] [--raw] [--shard 1024] [--zmax Z]\n"); return 2; }
    const char *root = argv[2], *key = argv[3], *out = argv[4];
    double um = atof(opt(argc, argv, "--um", "0"));
    int nlev = atoi(opt(argc, argv, "--levels", "6"));
    int nthreads = atoi(opt(argc, argv, "--threads", "16"));
    int labels = !strcmp(opt(argc, argv, "--raw", "0"), "0");
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--raw")) labels = 0;
    int shard = atoi(opt(argc, argv, "--shard", "1024"));
    int64_t zmax = atoll(opt(argc, argv, "--zmax", "0"));
    if (um <= 0) { fprintf(stderr, "--um required\n"); return 2; }
    store *s = store_open(root);
    char *tok = hf_token(opt(argc, argv, "--token", nullptr));
    if (tok && strstr(root, "huggingface")) store_set_bearer(s, tok);
    char akey[1200];
    snprintf(akey, sizeof akey, "%s/0", key);
    z2 *z = z2_open(s, akey, opt(argc, argv, "--cache", nullptr));
    if (!z) { fprintf(stderr, "%s\n", z2_error()); return 1; }
    z2_assume_present(z, 1);
    const z2_meta *m = z2_meta_of(z);
    fprintf(stderr, "source %s: %lld x %lld x %lld chunk %d sep '%c' comp %d\n", akey, (long long)m->shape[0], (long long)m->shape[1], (long long)m->shape[2], m->chunk[0], m->sep, m->comp);
    if (shard % m->chunk[0] || shard % m->chunk[1] || shard % m->chunk[2]) { fprintf(stderr, "shard must be a multiple of the source chunk\n"); return 1; }
    chunklist cl = {0};
    double t0 = now();
    if (m->sep == '/') { nested nd = {&cl, s, akey, 0, 0, 0}; store_list(s, akey, on_nested, &nd); }
    else store_list(s, akey, on_chunk, &cl);
    qsort(cl.c, (size_t)cl.n, sizeof *cl.c, cmp3);
    fprintf(stderr, "%d chunks listed in %.0fs\n", cl.n, now() - t0);
    if (!cl.n) return 1;
    int64_t shape[3]; memcpy(shape, m->shape, sizeof shape);
    if (zmax > 0 && zmax < shape[0]) shape[0] = zmax;
    char lv[32], ldir[1400];
    z3w_level_name(um, lv, sizeof lv);
    snprintf(ldir, sizeof ldir, "%s/%s", out, lv);
    char attrs[512];
    snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"%s\",\"source\":\"%s/%s\",\"encoding\":\"%s\"}}", labels ? "labels" : "raw", root, key, labels ? "0=bg,254=surface,255=ignore" : "uint8");
    z3w *w = z3w_create(ldir, shape, shard, 0.f, labels ? 255 : 0, attrs);
    if (!w) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    size_t sv = (size_t)shard * shard * shard;
    uint8_t *buf = malloc(sv);
    int cpp = shard / m->chunk[0];
    int i = 0, nshards = 0;
    uint8_t *used = calloc((size_t)cl.n, 1);
    while (i < cl.n) {
        if (used[i]) { i++; continue; }
        int64_t sz = cl.c[i][0] / cpp, sy = cl.c[i][1] / cpp, sx = cl.c[i][2] / cpp;
        if (sz * shard >= shape[0]) break;
        /* sorted by z, y, x: all chunks of this shard lie inside the run of chunk-rows with the same sz */
        int cnt = 0;
        int64_t (*sel)[3] = malloc((size_t)cpp * cpp * cpp * sizeof *sel);
        for (int j = i; j < cl.n && cl.c[j][0] / cpp == sz; j++)
            if (!used[j] && cl.c[j][1] / cpp == sy && cl.c[j][2] / cpp == sx) { memcpy(sel[cnt++], cl.c[j], sizeof *sel); used[j] = 1; }
        if (!cnt) { free(sel); i++; continue; }
        memset(buf, labels ? 255 : 0, sv);
        fill_job fj = {z, sel, cnt, buf, shard, {sz * shard, sy * shard, sx * shard}, labels, 0};
        parallel_for(cnt, nthreads, fill_chunk, &fj);
        if (atomic_load(&fj.failed)) return 1;
        if (z3w_write_shard(w, sz, sy, sx, buf, nthreads)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
        nshards++;
        fprintf(stderr, "\rshard %lld,%lld,%lld (%d chunks)  %d shards, %d/%d chunks, %.0fs   ", (long long)sz, (long long)sy, (long long)sx, cnt, nshards, i, cl.n, now() - t0);
        free(sel);
    }
    fprintf(stderr, "\nlevel 0 done: %d shards\n", nshards);
    z3w_close(w);
    free(buf); free(used);
    for (int l = 1; l < nlev; l++) if (pyramid_build_level(out, um, l, shape, shard, 0.f, labels, nthreads, attrs)) return 1;
    const char *slash = strrchr(key, '/');
    pyramid_write_group(out, um, nlev, slash ? slash + 1 : key, attrs);
    z2_close(z); store_close(s); free(tok);
    return 0;
}

/* ================= ls ================= */
static int on_print(const char *name, int is_dir, int64_t size, void *ud) { printf("%s%s\t%lld\n", name, is_dir ? "/" : "", (long long)size); return 0; }
int cmd_ls(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm ls <root> <key> [--token FILE]\n"); return 2; }
    store *s = store_open(argv[2]);
    char *tok = hf_token(opt(argc, argv, "--token", nullptr));
    if (tok && strstr(argv[2], "huggingface")) store_set_bearer(s, tok);
    int n = store_list(s, argv[3], on_print, nullptr);
    fprintf(stderr, "%d entries\n", n);
    return n < 0;
}

/* ================= ingest-kaggle ================= */
typedef struct { char **names; int n, cap; } namelist;
static int on_name(const char *name, int is_dir, int64_t size, void *ud) {
    namelist *nl = ud;
    if (is_dir || !strstr(name, ".tif")) return 0;
    if (nl->n == nl->cap) { nl->cap = nl->cap ? nl->cap * 2 : 64; nl->names = realloc(nl->names, (size_t)nl->cap * sizeof *nl->names); }
    nl->names[nl->n++] = strdup(name);
    return 0;
}
static int cmpstr(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

/* reads a cubic uint8 stack of edge <= Cmax into cube (edge^3, tightly packed); returns the edge or -1 */
static int read_cube(store *s, const char *key, uint8_t *cube, int Cmax) {
    size_t n;
    uint8_t *b = store_read_all(s, key, &n);
    if (!b) { fprintf(stderr, "download %s failed\n", key); return -1; }
    tiff *t = tiff_open_mem(b, n);
    if (!t) { fprintf(stderr, "%s: %s\n", key, tiff_error()); free(b); return -1; }
    tiff_page p; tiff_page_info(t, 0, &p);
    int C = tiff_npages(t);
    if (p.w != C || p.h != C || C > Cmax || p.bits != 8 || p.spp != 1) { fprintf(stderr, "%s: unexpected geometry %d pages %dx%d\n", key, C, p.w, p.h); tiff_close(t); free(b); return -1; }
    for (int i = 0; i < C; i++) if (tiff_read_page(t, i, cube + (size_t)i * C * C)) { fprintf(stderr, "%s: %s\n", key, tiff_error()); tiff_close(t); free(b); return -1; }
    tiff_close(t); free(b);
    return C;
}

int cmd_ingest_kaggle(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm ingest-kaggle <hf-root> <out-dir> [--key surfaces/kaggle] [--n N] [--token FILE] [--q 8] [--threads N]\n"); return 2; }
    const char *root = argv[2], *out = argv[3], *key = opt(argc, argv, "--key", "surfaces/kaggle");
    int limit = atoi(opt(argc, argv, "--n", "0")), nthreads = atoi(opt(argc, argv, "--threads", "16"));
    float q = (float)atof(opt(argc, argv, "--q", "8"));
    const int C = 320, SH = 640;   /* two cubes per shard along z */
    store *s = store_open(root);
    char *tok = hf_token(opt(argc, argv, "--token", nullptr));
    if (tok) store_set_bearer(s, tok);
    namelist nl = {0};
    char k[1200];
    snprintf(k, sizeof k, "%s/images", key);
    if (store_list(s, k, on_name, &nl) < 0) return 1;
    qsort(nl.names, (size_t)nl.n, sizeof *nl.names, cmpstr);
    int N = limit > 0 && limit < nl.n ? limit : nl.n;
    fprintf(stderr, "%d cubes (%d listed)\n", N, nl.n);
    int64_t shape[3] = {(int64_t)N * C, C, C};
    char attrs[256], d1[1400], d2[1400];
    snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"kaggle-images\",\"cube\":%d,\"n\":%d}}", C, N);
    snprintf(d1, sizeof d1, "%s/images.zarr/1", out);
    snprintf(d2, sizeof d2, "%s/labels.zarr/1", out);
    z3w *wi = z3w_create(d1, shape, SH, q, 0, attrs);
    snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"labels\",\"cube\":%d,\"n\":%d,\"encoding\":\"0=bg,254=surface,255=ignore\"}}", C, N);
    z3w *wl = z3w_create(d2, shape, SH, 0.f, 255, attrs);
    if (!wi || !wl) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    size_t sv = (size_t)SH * SH * SH;
    uint8_t *bi = malloc(sv), *bl = malloc(sv), *cube = malloc((size_t)C * C * C);
    const char *skipped[4096]; int nskipped = 0;
    int *sizes = calloc((size_t)N, sizeof(int));
    double t0 = now();
    for (int sz = 0; sz * 2 < N; sz++) {
        memset(bi, 0, sv); memset(bl, 255, sv);
        for (int j = 0; j < 2 && sz * 2 + j < N; j++) {
            const char *name = nl.names[sz * 2 + j];
            snprintf(k, sizeof k, "%s/images/%s", key, name);
            int ci = read_cube(s, k, cube, C);
            if (ci < 0) { fprintf(stderr, "skipping %s\n", name); if (nskipped < 4096) skipped[nskipped++] = name; continue; }
            for (int z = 0; z < ci; z++) for (int y = 0; y < ci; y++) memcpy(bi + ((size_t)(j * C + z) * SH + y) * SH, cube + ((size_t)z * ci + y) * ci, (size_t)ci);
            snprintf(k, sizeof k, "%s/labels/%s", key, name);
            int cl = read_cube(s, k, cube, C);
            if (cl != ci) { fprintf(stderr, "skipping %s (label %d vs image %d)\n", name, cl, ci); if (nskipped < 4096) skipped[nskipped++] = name; for (int z = 0; z < ci; z++) for (int y = 0; y < ci; y++) memset(bi + ((size_t)(j * C + z) * SH + y) * SH, 0, (size_t)ci); continue; }
            for (int z = 0; z < cl; z++) for (int y = 0; y < cl; y++) {
                uint8_t *dst = bl + ((size_t)(j * C + z) * SH + y) * SH; const uint8_t *src = cube + ((size_t)z * cl + y) * cl;
                for (int x = 0; x < cl; x++) dst[x] = src[x] == 2 ? 255 : src[x] ? 254 : 0;
            }
            sizes[sz * 2 + j] = cl;
        }
        if (z3w_write_shard(wi, sz, 0, 0, bi, nthreads) || z3w_write_shard(wl, sz, 0, 0, bl, nthreads)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
        fprintf(stderr, "\r%d/%d cubes, %.0fs   ", (sz + 1) * 2 < N ? (sz + 1) * 2 : N, N, now() - t0);
    }
    fprintf(stderr, "\n");
    z3w_close(wi); z3w_close(wl);
    double um[1] = {1.0};
    snprintf(d1, sizeof d1, "%s/images.zarr", out); z3w_write_group(d1, um, 1, "kaggle-images", nullptr);
    snprintf(d2, sizeof d2, "%s/labels.zarr", out); z3w_write_group(d2, um, 1, "kaggle-labels", nullptr);
    /* index of cube names */
    snprintf(d1, sizeof d1, "%s/cubes.json", out);
    FILE *f = fopen(d1, "w");
    if (f) {
        fprintf(f, "{\"cube\":%d,\"n\":%d,\"names\":[", C, N);
        for (int i = 0; i < N; i++) fprintf(f, "%s\"%s\"", i ? "," : "", nl.names[i]);
        fprintf(f, "],\"origins\":[");
        int first = 1;
        for (int i = 0; i < N; i++) if (sizes[i]) { fprintf(f, "%s[%d,0,0,%d]", first ? "" : ",", i * C, sizes[i]); first = 0; }
        fprintf(f, "],\"skipped\":[");
        for (int i = 0; i < nskipped; i++) fprintf(f, "%s\"%s\"", i ? "," : "", skipped[i]);
        fprintf(f, "]}\n");
        fclose(f);
    }
    fprintf(stderr, "%d cubes skipped (bad geometry)\n", nskipped);
    free(bi); free(bl); free(cube);
    return 0;
}

/* ================= ingest-mesh: tifxyz -> .sfc ================= */
static float *load_axis_tif(store *s, const char *key, int *w, int *h) {
    size_t n;
    uint8_t *b = store_read_all(s, key, &n);
    if (!b) { fprintf(stderr, "download %s failed\n", key); return nullptr; }
    tiff *t = tiff_open_mem(b, n);
    if (!t) { fprintf(stderr, "%s: %s\n", key, tiff_error()); free(b); return nullptr; }
    tiff_page p; tiff_page_info(t, 0, &p);
    if (p.bits != 32 || p.fmt != 3 || p.spp != 1) { fprintf(stderr, "%s: expected float32\n", key); tiff_close(t); free(b); return nullptr; }
    float *a = malloc((size_t)p.w * p.h * 4);
    if (tiff_read_page(t, 0, a)) { fprintf(stderr, "%s: %s\n", key, tiff_error()); free(a); a = nullptr; }
    *w = p.w; *h = p.h;
    tiff_close(t); free(b);
    return a;
}

int cmd_ingest_mesh(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm ingest-mesh <root> <tifxyz-dir-key> <out.sfc> [--error 0.1]\n"); return 2; }
    const char *root = argv[2], *key = argv[3], *out = argv[4];
    double err = atof(opt(argc, argv, "--error", "0.1"));
    store *s = store_open(root);
    char k[1400];
    float *ax[3]; int w = 0, h = 0;
    const char *names[3] = {"x.tif", "y.tif", "z.tif"};
    for (int a = 0; a < 3; a++) {
        int aw, ah;
        snprintf(k, sizeof k, "%s/%s", key, names[a]);
        ax[a] = load_axis_tif(s, k, &aw, &ah);
        if (!ax[a]) return 1;
        if (a == 0) { w = aw; h = ah; } else if (aw != w || ah != h) { fprintf(stderr, "axis size mismatch\n"); return 1; }
    }
    snprintf(k, sizeof k, "%s/meta.json", key);
    size_t ml = 0;
    uint8_t *meta = store_read_all(s, k, &ml);
    if (!meta) { fprintf(stderr, "no meta.json under %s\n", key); return 1; }
    /* optional mask.tif */
    snprintf(k, sizeof k, "%s/mask.tif", key);
    uint8_t *mask = nullptr; int mw = 0, mh = 0;
    { size_t n; uint8_t *b = store_read_all(s, k, &n);
      if (b) { tiff *t = tiff_open_mem(b, n); tiff_page p; if (t && !tiff_page_info(t, 0, &p) && p.bits == 8 && p.spp == 1) { mask = malloc((size_t)p.w * p.h); mw = p.w; mh = p.h; if (tiff_read_page(t, 0, mask)) { free(mask); mask = nullptr; } } tiff_close(t); free(b); } }
    sfc_channel descs[4];
    memset(descs, 0, sizeof descs);
    const char *cn[3] = {"x", "y", "z"};
    for (int a = 0; a < 3; a++) {
        snprintf(descs[a].name, sizeof descs[a].name, "%s", cn[a]);
        descs[a].width = (uint64_t)w; descs[a].height = (uint64_t)h; descs[a].dtype = SFC_F32;
        descs[a].flags = SFC_COORDINATE | SFC_XYZ; descs[a].components = 3; descs[a].component = (uint32_t)a; descs[a].tolerance = err;
    }
    int nc = 3;
    if (mask) { snprintf(descs[3].name, sizeof descs[3].name, "mask"); descs[3].width = (uint64_t)mw; descs[3].height = (uint64_t)mh; descs[3].dtype = SFC_U8; descs[3].flags = SFC_EXACT; descs[3].tolerance = 0; nc = 4; }
    char tmp[1500]; snprintf(tmp, sizeof tmp, "%s.tmp", out);
    sfc_writer *wr = nullptr;
    if (sfc_create(tmp, descs, (uint32_t)nc, meta, ml, &wr)) { fprintf(stderr, "sfc_create failed\n"); return 1; }
    static float joint[4096][3]; static uint8_t valid[4096];
    size_t nvalid = 0;
    for (uint64_t by = 0; by < ((uint64_t)h + 63) / 64; by++) for (uint64_t bx = 0; bx < ((uint64_t)w + 63) / 64; bx++) {
        memset(joint, 0, sizeof joint); memset(valid, 0, sizeof valid);
        for (unsigned q = 0; q < 4096; q++) {
            uint64_t x = bx * 64 + q % 64, y = by * 64 + q / 64;
            if (x >= (uint64_t)w || y >= (uint64_t)h) continue;
            int ok = 1;
            for (int a = 0; a < 3; a++) { float v = ax[a][y * (uint64_t)w + x]; joint[q][a] = v; if (!isfinite(v) || (a == 2 && v <= 0)) ok = 0; }
            if (ok && mask && mw % w == 0 && mh % h == 0) {
                int sx = mw / w, sy = mh / h;
                for (int yy = 0; ok && yy < sy; yy++) for (int xx = 0; xx < sx; xx++) if (mask[(y * sy + yy) * (uint64_t)mw + x * sx + xx] < 255) { ok = 0; break; }
            }
            valid[q] = (uint8_t)ok;
            nvalid += ok;
        }
        if (sfc_write_xyz_block(wr, joint, 12, 768, valid)) { fprintf(stderr, "sfc_write_xyz_block failed\n"); sfc_cancel(wr); return 1; }
    }
    if (mask) {
        static uint8_t data[4096];
        for (uint64_t by = 0; by < ((uint64_t)mh + 63) / 64; by++) for (uint64_t bx = 0; bx < ((uint64_t)mw + 63) / 64; bx++) {
            memset(data, 0, sizeof data); memset(valid, 0, sizeof valid);
            for (unsigned q = 0; q < 4096; q++) { uint64_t x = bx * 64 + q % 64, y = by * 64 + q / 64; if (x >= (uint64_t)mw || y >= (uint64_t)mh) continue; data[q] = mask[y * (uint64_t)mw + x]; valid[q] = 1; }
            if (sfc_write_block(wr, data, 1, 64, valid)) { fprintf(stderr, "sfc_write_block failed\n"); sfc_cancel(wr); return 1; }
        }
    }
    if (sfc_finish(wr)) { fprintf(stderr, "sfc_finish failed\n"); return 1; }
    if (rename(tmp, out)) { fprintf(stderr, "rename %s failed\n", out); return 1; }
    fprintf(stderr, "%s: %dx%d grid, %zu valid points%s -> %s\n", key, w, h, nvalid, mask ? " (masked)" : "", out);
    for (int a = 0; a < 3; a++) free(ax[a]);
    free(meta); free(mask); store_close(s);
    return 0;
}

/* ================= ingest-zip: label zarr straight out of labels.zip ================= */
typedef struct { int64_t c[3]; size_t idx; } zchunk;
static int cmp_zchunk(const void *a, const void *b) { const zchunk *p = a, *q = b; for (int d = 0; d < 3; d++) if (p->c[d] != q->c[d]) return p->c[d] < q->c[d] ? -1 : 1; return 0; }
typedef struct { const zipr *zip; const zchunk *chunks; int n; uint8_t *shard; int S; int64_t so[3]; int chunk[3]; int comp; int labels; atomic_int failed; } zfill;
static void zfill_chunk(int i, int tid, void *ud) {
    zfill *j = ud;
    size_t n, cv = (size_t)j->chunk[0] * j->chunk[1] * j->chunk[2];
    uint8_t *b = zipr_read(j->zip, j->chunks[i].idx, &n);
    if (!b) { fprintf(stderr, "%s\n", zipr_error()); atomic_store(&j->failed, 1); return; }
    uint8_t *buf = malloc(cv);
    if (z2_decode(j->comp, b, n, buf, cv)) { fprintf(stderr, "chunk %lld,%lld,%lld: %s\n", (long long)j->chunks[i].c[0], (long long)j->chunks[i].c[1], (long long)j->chunks[i].c[2], z2_error()); atomic_store(&j->failed, 1); free(b); free(buf); return; }
    free(b);
    int64_t o[3] = {j->chunks[i].c[0] * j->chunk[0], j->chunks[i].c[1] * j->chunk[1], j->chunks[i].c[2] * j->chunk[2]};
    for (int64_t z = 0; z < j->chunk[0]; z++) for (int64_t y = 0; y < j->chunk[1]; y++) {
        int64_t gz = o[0] + z - j->so[0], gy = o[1] + y - j->so[1], gx = o[2] - j->so[2];
        if (gz < 0 || gz >= j->S || gy < 0 || gy >= j->S) continue;
        uint8_t *dst = j->shard + ((size_t)gz * j->S + gy) * j->S + gx;
        const uint8_t *src = buf + ((size_t)z * j->chunk[1] + y) * j->chunk[2];
        for (int64_t x = 0; x < j->chunk[2] && gx + x < j->S; x++) { uint8_t v = src[x]; dst[x] = j->labels ? (v == 2 ? 255 : v ? 254 : 0) : v; }
    }
    free(buf);
}

int cmd_ingest_zip(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm ingest-zip <labels.zip> <zarr-name> <out-dir> --um U [--levels L] [--threads N] [--shard 1024] [--zmax Z] [--raw]\n"); return 2; }
    const char *zpath = argv[2], *name = argv[3], *out = argv[4];
    double um = atof(opt(argc, argv, "--um", "0"));
    int nlev = atoi(opt(argc, argv, "--levels", "6")), nthreads = atoi(opt(argc, argv, "--threads", "32")), shard = atoi(opt(argc, argv, "--shard", "1024"));
    int64_t zmax = atoll(opt(argc, argv, "--zmax", "0"));
    int labels = 1;
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], "--raw")) labels = 0;
    if (um <= 0) { fprintf(stderr, "--um required\n"); return 2; }
    double t0 = now();
    zipr *zip = zipr_open(zpath);
    if (!zip) { fprintf(stderr, "%s\n", zipr_error()); return 1; }
    fprintf(stderr, "%zu entries in %s (%.0fs)\n", zipr_count(zip), zpath, now() - t0);
    char key[512];
    snprintf(key, sizeof key, "%s/0/.zarray", name);
    long zi = zipr_find(zip, key);
    if (zi < 0) { fprintf(stderr, "no %s in archive\n", key); return 1; }
    size_t n;
    uint8_t *txt = zipr_read(zip, (size_t)zi, &n);
    json *j = txt ? json_parse((char *)txt, n) : nullptr;
    free(txt);
    if (!j) { fprintf(stderr, "bad .zarray\n"); return 1; }
    int64_t shape[3]; int chunk[3];
    for (int d = 0; d < 3; d++) { shape[d] = (int64_t)json_num(json_at(json_get(j, "shape"), (size_t)d), 0); chunk[d] = (int)json_num(json_at(json_get(j, "chunks"), (size_t)d), 0); }
    const json *comp = json_get(j, "compressor");
    int cc = comp && comp->type != J_NULL ? z2_comp_of(json_str(json_get(comp, "id"), "")) : 0;
    const char *dt = json_str(json_get(j, "dtype"), "");
    if (cc < 0 || (strcmp(dt, "|u1") && strcmp(dt, "<u1") && strcmp(dt, ">u1") && strcmp(dt, "u1"))) { fprintf(stderr, "unsupported .zarray (compressor %d / dtype %s)\n", cc, dt); return 1; }
    json_free(j);
    fprintf(stderr, "%s: %lld x %lld x %lld chunk %d comp %d\n", name, (long long)shape[0], (long long)shape[1], (long long)shape[2], chunk[0], cc);
    if (shard % chunk[0] || shard % chunk[1] || shard % chunk[2]) { fprintf(stderr, "shard must be a multiple of the chunk\n"); return 1; }
    /* collect level-0 chunk entries */
    snprintf(key, sizeof key, "%s/0/", name);
    size_t kl = strlen(key), cap = 1 << 16, cnt = 0;
    zchunk *cl = malloc(cap * sizeof *cl);
    for (size_t i = 0; i < zipr_count(zip); i++) {
        zip_entry e; zipr_entry(zip, i, &e);
        if (strncmp(e.name, key, kl) || e.name[kl] == '.') continue;
        long long z, y, x;
        if (sscanf(e.name + kl, "%lld.%lld.%lld", &z, &y, &x) != 3 && sscanf(e.name + kl, "%lld/%lld/%lld", &z, &y, &x) != 3) continue;
        if (e.size == 0) continue;
        if (cnt == cap) { cap *= 2; cl = realloc(cl, cap * sizeof *cl); }
        cl[cnt++] = (zchunk){{z, y, x}, i};
    }
    qsort(cl, cnt, sizeof *cl, cmp_zchunk);
    fprintf(stderr, "%zu level-0 chunks\n", cnt);
    if (zmax > 0 && zmax < shape[0]) shape[0] = zmax;
    char lv[32], ldir[1400], attrs[512];
    z3w_level_name(um, lv, sizeof lv);
    snprintf(ldir, sizeof ldir, "%s/%s", out, lv);
    snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"%s\",\"source\":\"labels.zip/%s\",\"encoding\":\"%s\"}}", labels ? "labels" : "raw", name, labels ? "0=bg,254=surface,255=ignore" : "uint8");
    z3w *w = z3w_create(ldir, shape, shard, 0.f, labels ? 255 : 0, attrs);
    if (!w) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    size_t sv = (size_t)shard * shard * shard;
    uint8_t *buf = malloc(sv), *used = calloc(cnt, 1);
    int cpp = shard / chunk[0];
    zchunk *sel = malloc((size_t)cpp * cpp * cpp * sizeof *sel);
    size_t i = 0, done = 0; int nshards = 0;
    while (i < cnt) {
        if (used[i]) { i++; continue; }
        int64_t sz = cl[i].c[0] / cpp, sy = cl[i].c[1] / cpp, sx = cl[i].c[2] / cpp;
        if (sz * shard >= shape[0]) break;
        int k = 0;
        for (size_t jj = i; jj < cnt && cl[jj].c[0] / cpp == sz; jj++)
            if (!used[jj] && cl[jj].c[1] / cpp == sy && cl[jj].c[2] / cpp == sx) { sel[k++] = cl[jj]; used[jj] = 1; }
        if (!k) { i++; continue; }
        memset(buf, labels ? 255 : 0, sv);
        zfill fj = {zip, sel, k, buf, shard, {sz * shard, sy * shard, sx * shard}, {chunk[0], chunk[1], chunk[2]}, cc, labels, 0};
        parallel_for(k, nthreads, zfill_chunk, &fj);
        if (atomic_load(&fj.failed)) return 1;
        if (z3w_write_shard(w, sz, sy, sx, buf, nthreads)) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
        nshards++; done += (size_t)k;
        fprintf(stderr, "\rshard %lld,%lld,%lld  %d shards, %zu/%zu chunks, %.0fs   ", (long long)sz, (long long)sy, (long long)sx, nshards, done, cnt, now() - t0);
    }
    fprintf(stderr, "\nlevel 0 done: %d shards\n", nshards);
    z3w_close(w);
    free(buf); free(used); free(sel); free(cl);
    for (int l = 1; l < nlev; l++) if (pyramid_build_level(out, um, l, shape, shard, 0.f, labels, nthreads, attrs)) return 1;
    pyramid_write_group(out, um, nlev, name, attrs);
    zipr_close(zip);
    return 0;
}

/* ================= raster: segment meshes -> label pyramid ================= */
typedef struct { int w, h; float *xyz; uint8_t *valid; double lo[3], hi[3]; } mesh;

static mesh *load_mesh(const char *path) {
    mesh *m = calloc(1, sizeof *m);
    size_t n = strlen(path);
    if (n > 4 && !strcmp(path + n - 4, ".sfc")) {
        sfc_reader *r = nullptr;
        if (sfc_open_file(path, &r)) { fprintf(stderr, "cannot open %s\n", path); free(m); return nullptr; }
        int cx = sfc_find_channel(r, "x");
        if (cx < 0) cx = 0;
        const sfc_channel *c = sfc_channel_info(r, (uint32_t)cx);
        m->w = (int)c->width; m->h = (int)c->height;
        m->xyz = malloc((size_t)m->w * m->h * 3 * sizeof(float));
        m->valid = malloc((size_t)m->w * m->h);
        sfc_cache *cache = nullptr;
        sfc_cache_create(r, (size_t)256 << 20, &cache);
        int rc = sfc_cache_read_xyz_region(cache, (uint32_t)cx, 0, 0, (uint32_t)m->w, (uint32_t)m->h, m->xyz, 12, (size_t)m->w * 12, m->valid);
        sfc_cache_destroy(cache);
        sfc_close(r);
        if (rc) { fprintf(stderr, "%s: read xyz failed (%d)\n", path, rc); free(m->xyz); free(m->valid); free(m); return nullptr; }
    } else {   /* tifxyz directory */
        const char *names[3] = {"x.tif", "y.tif", "z.tif"};
        float *ax[3] = {nullptr, nullptr, nullptr};
        for (int a = 0; a < 3; a++) {
            char p[1400]; snprintf(p, sizeof p, "%s/%s", path, names[a]);
            tiff *t = tiff_open_file(p);
            if (!t) {
                fprintf(stderr, "%s: %s\n", p, tiff_error());
                for (int k = 0; k < a; k++) free(ax[k]);
                free(m); return nullptr;
            }
            tiff_page pg;
            if (tiff_page_info(t, 0, &pg) || pg.w < 2 || pg.h < 2 || pg.bits != 32 || pg.fmt != 3 || pg.spp != 1 ||
                (a && (pg.w != m->w || pg.h != m->h))) {
                fprintf(stderr, "%s: tifxyz axes must be matching 2-D float32 single-channel images\n", p);
                tiff_close(t); for (int k = 0; k < a; k++) free(ax[k]); free(m); return nullptr;
            }
            if (a == 0) { m->w = pg.w; m->h = pg.h; }
            ax[a] = malloc((size_t)pg.w * pg.h * 4);
            if (!ax[a] || tiff_read_page(t, 0, ax[a])) {
                fprintf(stderr, "%s: TIFF decode failed\n", p);
                tiff_close(t); for (int k = 0; k <= a; k++) free(ax[k]); free(m); return nullptr;
            }
            tiff_close(t);
        }
        m->xyz = malloc((size_t)m->w * m->h * 3 * sizeof(float));
        m->valid = malloc((size_t)m->w * m->h);
        for (size_t i = 0; i < (size_t)m->w * m->h; i++) {
            for (int a = 0; a < 3; a++) m->xyz[i * 3 + a] = ax[a][i];
            m->valid[i] = isfinite(ax[0][i]) && isfinite(ax[1][i]) && isfinite(ax[2][i]) && ax[2][i] > 0 && ax[0][i] >= 0;
        }
        for (int a = 0; a < 3; a++) free(ax[a]);
    }
    for (int d = 0; d < 3; d++) { m->lo[d] = 1e30; m->hi[d] = -1e30; }
    for (size_t i = 0; i < (size_t)m->w * m->h; i++) {
        if (!m->valid[i]) continue;
        for (int a = 0; a < 3; a++) { double v = m->xyz[i * 3 + a]; int d = 2 - a; if (v < m->lo[d]) m->lo[d] = v; if (v > m->hi[d]) m->hi[d] = v; }
    }
    return m;
}

/* chamfer 3-4-5 distance transform on a uint8 grid where 0 = surface, 255 = far; capped at `cap` */
static void chamfer_reference(uint8_t *d, int N, int cap) {
    static const int off[13][4] = {{-1,-1,-1,5},{-1,-1,0,4},{-1,-1,1,5},{-1,0,-1,4},{-1,0,0,3},{-1,0,1,4},{-1,1,-1,5},{-1,1,0,4},{-1,1,1,5},{0,-1,-1,4},{0,-1,0,3},{0,-1,1,4},{0,0,-1,3}};
    size_t NN = (size_t)N * N;
    for (int pass = 0; pass < 2; pass++)
        for (int zi = 0; zi < N; zi++) for (int yi = 0; yi < N; yi++) for (int xi = 0; xi < N; xi++) {
            int z = pass ? N - 1 - zi : zi, y = pass ? N - 1 - yi : yi, x = pass ? N - 1 - xi : xi;
            uint8_t *p = d + (size_t)z * NN + (size_t)y * N + x;
            int best = *p;
            if (best == 0) continue;
            for (int k = 0; k < 13; k++) {
                int nz = z + (pass ? -off[k][0] : off[k][0]), ny = y + (pass ? -off[k][1] : off[k][1]), nx = x + (pass ? -off[k][2] : off[k][2]);
                if (nz < 0 || nz >= N || ny < 0 || ny >= N || nx < 0 || nx >= N) continue;
                int v = d[(size_t)nz * NN + (size_t)ny * N + nx] + off[k][3];
                if (v < best) best = v;
            }
            *p = (uint8_t)(best > cap ? cap : best);
        }
}

typedef struct { int mi, r0, r1, c0, c1; double lo[3], hi[3]; } rtile;
typedef struct {
    mesh **meshes; int nm; double scale; int shard, margin, T;
    int64_t ns[3]; int64_t shape[3]; z3w *w; atomic_int failed, done; int nthreads;
    rtile *tiles; size_t ntile; size_t *offset; uint32_t *refs;
    int indexed, reference_distance, binary;
} rjob;

#include "raster_cpu.h"

static void raster_shard(int si, int tid, void *ud) {
    rjob *j = ud;
    int64_t sx = si % j->ns[2], sy = (si / j->ns[2]) % j->ns[1], sz = si / (j->ns[2] * j->ns[1]);
    int S = j->shard, M = j->margin, N = S + 2 * M;
    double lo[3] = {(double)(sz * S - M), (double)(sy * S - M), (double)(sx * S - M)}, hi[3] = {lo[0] + N, lo[1] + N, lo[2] + N};
    /* meshes touching this box */
    int any = j->indexed && j->offset[si + 1] > j->offset[si];
    for (int mi = 0; !j->indexed && mi < j->nm; mi++) {
        mesh *m = j->meshes[mi];
        int hit = 1;
        for (int d = 0; d < 3; d++) if (m->lo[d] * j->scale > hi[d] || m->hi[d] * j->scale < lo[d]) hit = 0;
        if (hit) { any = 1; break; }
    }
    if (!any) { atomic_fetch_add(&j->done, 1); return; }   /* missing shard = fill 255 = ignore */
    size_t NN = (size_t)N * N * N;
    uint8_t *d = malloc(NN);
    if (!d) { atomic_store(&j->failed, 1); return; }
    memset(d, 255, NN);
    size_t begin = j->indexed ? j->offset[si] : 0, end = j->indexed ? j->offset[si + 1] : (size_t)j->nm;
    for (size_t it = begin; it < end; it++) {
        const rtile *tile = j->indexed ? &j->tiles[j->refs[it]] : nullptr;
        int mi = tile ? tile->mi : (int)it;
        mesh *m = j->meshes[mi];
        int hit = 1;
        for (int dd = 0; dd < 3; dd++) if (m->lo[dd] * j->scale > hi[dd] || m->hi[dd] * j->scale < lo[dd]) hit = 0;
        if (!hit) continue;
        int r0 = tile ? tile->r0 : 0, r1 = tile ? tile->r1 : m->h - 1;
        int c0 = tile ? tile->c0 : 0, c1 = tile ? tile->c1 : m->w - 1;
        for (int r = r0; r < r1; r++) for (int c = c0; c < c1; c++) {
            size_t i00 = (size_t)r * m->w + c, i01 = i00 + 1, i10 = i00 + (size_t)m->w, i11 = i10 + 1;
            if (!(m->valid[i00] && m->valid[i01] && m->valid[i10] && m->valid[i11])) continue;
            double p[4][3];
            const size_t idx[4] = {i00, i01, i10, i11};
            double cl[3] = {1e30, 1e30, 1e30}, ch[3] = {-1e30, -1e30, -1e30};
            for (int k = 0; k < 4; k++) for (int a = 0; a < 3; a++) { double v = m->xyz[idx[k] * 3 + a] * j->scale; p[k][2 - a] = v; }
            for (int k = 0; k < 4; k++) for (int dd = 0; dd < 3; dd++) { if (p[k][dd] < cl[dd]) cl[dd] = p[k][dd]; if (p[k][dd] > ch[dd]) ch[dd] = p[k][dd]; }
            int out = 0;
            for (int dd = 0; dd < 3; dd++) if (cl[dd] > hi[dd] || ch[dd] < lo[dd]) out = 1;
            if (out) continue;
            double e1 = 0, e2 = 0;
            for (int dd = 0; dd < 3; dd++) { e1 += (p[1][dd] - p[0][dd]) * (p[1][dd] - p[0][dd]); e2 += (p[2][dd] - p[0][dd]) * (p[2][dd] - p[0][dd]); }
            int nu = (int)ceil(sqrt(e1) / 0.5) + 1, nv = (int)ceil(sqrt(e2) / 0.5) + 1;
            if (nu > 256) nu = 256;
            if (nv > 256) nv = 256;
            for (int a = 0; a < nu; a++) for (int b = 0; b < nv; b++) {
                double u = nu > 1 ? (double)a / (nu - 1) : 0, v = nv > 1 ? (double)b / (nv - 1) : 0;
                int vox[3], in = 1;
                for (int dd = 0; dd < 3; dd++) {
                    double q = (1 - u) * (1 - v) * p[0][dd] + u * (1 - v) * p[1][dd] + (1 - u) * v * p[2][dd] + u * v * p[3][dd];
                    vox[dd] = (int)floor(q - lo[dd] + 0.5);
                    if (vox[dd] < 0 || vox[dd] >= N) in = 0;
                }
                if (in) d[((size_t)vox[0] * N + vox[1]) * N + vox[2]] = 0;
            }
        }
    }
    if (!j->binary || j->reference_distance) {
        if (j->reference_distance) chamfer_reference(d, N, 3 * j->T + 3);
        else raster_distance(d, N, 3 * j->T + 3);
    }
    uint8_t *buf = malloc((size_t)S * S * S);
    if (!buf) { free(d); atomic_store(&j->failed, 1); return; }
    if (j->binary && !j->reference_distance) raster_expand_mask(d, N, buf, S, M);
    else for (int z = 0; z < S; z++) for (int y = 0; y < S; y++) {
        const uint8_t *src = d + ((size_t)(z + M) * N + (y + M)) * N + M;
        uint8_t *dst = buf + ((size_t)z * S + y) * S;
        for (int x = 0; x < S; x++) dst[x] = j->binary ? (src[x] <= 4 ? 255 : 0) : src[x] <= 4 ? 254 : src[x] <= 3 * j->T ? 0 : 255;
    }
    free(d);
    if (z3w_write_shard(j->w, sz, sy, sx, buf, 1)) { fprintf(stderr, "%s\n", z3w_error()); atomic_store(&j->failed, 1); }
    free(buf);
    int done = atomic_fetch_add(&j->done, 1) + 1;
    if (done % 16 == 0) fprintf(stderr, "\r%d/%lld shards   ", done, (long long)(j->ns[0] * j->ns[1] * j->ns[2]));
}

int cmd_raster(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm raster <out-dir> --shape Z,Y,X --um U [--level L] [--binary 0|1] [--T 3] [--levels 6] [--threads 8] [--shard 1024] [--raster-index 1] [--reference-distance 0] <mesh.sfc|tifxyz-dir>...\n"); return 2; }
    const char *out = argv[2];
    long long Z = 0, Y = 0, X = 0;
    if (sscanf(opt(argc, argv, "--shape", "0,0,0"), "%lld,%lld,%lld", &Z, &Y, &X) != 3 || Z <= 0 || Y <= 0 || X <= 0) { fprintf(stderr, "--shape Z,Y,X (positive level-0 voxels) required\n"); return 2; }
    double um = atof(opt(argc, argv, "--um", "0"));
    int level = atoi(opt(argc, argv, "--level", "0")), T = atoi(opt(argc, argv, "--T", "3")), nlev = atoi(opt(argc, argv, "--levels", "6"));
    int nthreads = atoi(opt(argc, argv, "--threads", "8")), shard = atoi(opt(argc, argv, "--shard", "1024"));
    int binary = atoi(opt(argc, argv, "--binary", "0"));
    if (!isfinite(um) || um <= 0 || level < 0 || level > 20 || T < 1 || T > 84 ||
        nlev < 1 || nlev > MAXLEV_PYR || nthreads < 1 || nthreads > 256 || shard < 128 || shard % 128 || (binary != 0 && binary != 1)) {
        fprintf(stderr, "raster: require positive um, level 0..20, T 1..84, levels 1..12, threads 1..256 and shard a positive multiple of 128\n"); return 2;
    }
    mesh *meshes[4096]; int nm = 0;
    for (int i = 3; i < argc && nm < 4096; i++) {
        if (argv[i][0] == '-') { i++; continue; }
        mesh *m = load_mesh(argv[i]);
        if (!m) return 1;
        meshes[nm++] = m;
        fprintf(stderr, "%s: %dx%d, bbox z %.0f-%.0f y %.0f-%.0f x %.0f-%.0f\n", argv[i], m->w, m->h, m->lo[0], m->hi[0], m->lo[1], m->hi[1], m->lo[2], m->hi[2]);
    }
    if (!nm) { fprintf(stderr, "no meshes\n"); return 2; }
    double scale = 1.0 / (1 << level), um_l = um * (1 << level);
    int64_t shape[3] = {(Z + (1 << level) - 1) >> level, (Y + (1 << level) - 1) >> level, (X + (1 << level) - 1) >> level};
    char lv[32], ldir[1400], attrs[512];
    z3w_level_name(um_l, lv, sizeof lv);
    snprintf(ldir, sizeof ldir, "%s/%s", out, lv);
    if (binary) snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"labels\",\"source\":\"raster of %d meshes\",\"encoding\":\"binary\",\"surface_band_chamfer\":4,\"codec\":\"volcomp-mask-lossless\",\"background\":\"all non-surface voxels\"}}", nm);
    else snprintf(attrs, sizeof attrs, "{\"ufsm\":{\"content\":\"labels\",\"source\":\"raster of %d meshes\",\"T\":%d,\"encoding\":\"0=bg,254=surface,255=ignore\"}}", nm, T);
    z3w *w = binary ? z3w_create_mask(ldir, shape, shard, attrs) : z3w_create(ldir, shape, shard, 0.f, 255, attrs);
    if (!w) { fprintf(stderr, "%s\n", z3w_error()); return 1; }
    rjob j = {.meshes = meshes, .nm = nm, .scale = scale, .shard = shard, .margin = binary ? 2 : T + 3, .T = T, .binary = binary,
              .ns = {(shape[0] + shard - 1) / shard, (shape[1] + shard - 1) / shard, (shape[2] + shard - 1) / shard},
              .shape = {shape[0], shape[1], shape[2]}, .w = w, .nthreads = nthreads,
              .indexed = atoi(opt(argc, argv, "--raster-index", "1")),
              .reference_distance = atoi(opt(argc, argv, "--reference-distance", "0"))};
    double t0 = now();
    int result = 0;
    long double nshards = (long double)j.ns[0] * j.ns[1] * j.ns[2];
    if (nshards > INT_MAX) { fprintf(stderr, "raster: too many shards\n"); result = 1; }
    else if (j.indexed && raster_index(&j)) { fprintf(stderr, "raster: spatial index allocation failed\n"); result = 1; }
    else parallel_for((int)nshards, nthreads, raster_shard, &j);
    fprintf(stderr, "\nlevel %d rasterized in %.0fs\n", level, now() - t0);
    z3w_close(w);
    result |= atomic_load(&j.failed);
    for (int l = 1; !result && l < nlev; l++) result = pyramid_build_level(out, um_l, l, shape, shard, 0.f, binary ? 2 : 1, nthreads, attrs) != 0;
    if (!result) result = pyramid_write_group(out, um_l, nlev, "raster-labels", attrs) != 0;
    free(j.tiles); free(j.offset); free(j.refs);
    for (int mi = 0; mi < nm; mi++) { free(meshes[mi]->xyz); free(meshes[mi]->valid); free(meshes[mi]); }
    return result;
}
