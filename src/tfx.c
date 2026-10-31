/* .sfc-backed tifxyz wraps (src/tiffw.h): surface-compressor (third_party/surfcomp, format 4) files holding one wrap's x / y / z grid
   as a joint XYZ coordinate triple (Euclidean tolerance in voxels) and its meta.json bytes as the container metadata. Installs the
   tiffw.h hooks at load time, so every tifxyz reader that goes through tfx_axis_read / tfx_meta_fopen reads <wrap>.sfc directly.
   The last decoded file is cached per thread (a reader asks for x, y, z of the same wrap in turn). */
#include "tiffw.h"
#include "surfcomp.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

/* the .sfc behind a path <wrap>/<leaf>: <wrap> itself (a .sfc file) or <wrap>.sfc; leaf in *leaf. 0 when none */
static int tfx_resolve(const char *path, char *sfc, size_t n, const char **leaf) {
    const char *sl = strrchr(path, '/'); if (!sl) return 0;
    *leaf = sl + 1;
    const size_t wl = (size_t)(sl - path); if (wl + 5 >= n) return 0;
    struct stat st;
    memcpy(sfc, path, wl); sfc[wl] = 0;
    if (wl > 4 && !strcmp(sfc + wl - 4, ".sfc") && !stat(sfc, &st) && S_ISREG(st.st_mode)) return 1;
    memcpy(sfc + wl, ".sfc", 5);
    return !stat(sfc, &st) && S_ISREG(st.st_mode);
}
typedef struct { char path[1400]; int w, h; float *xyz; uint8_t *valid; char *meta; } tfx_cache;
static _Thread_local tfx_cache g_c;
static int tfx_load(const char *sfc) {   /* decode sfc into the thread's cache (0 ok) */
    if (!strcmp(g_c.path, sfc) && g_c.xyz) return 0;
    free(g_c.xyz); free(g_c.valid); free(g_c.meta); memset(&g_c, 0, sizeof g_c);
    sfc_reader *r = nullptr; if (sfc_open_file(sfc, &r)) return -1;
    int cx = sfc_find_channel(r, "x"); if (cx < 0) cx = 0;
    const sfc_channel *c = sfc_channel_info(r, (uint32_t)cx);
    const int w = (int)c->width, h = (int)c->height; const size_t n = (size_t)w * h;
    float *xyz = malloc(n * 12); uint8_t *valid = malloc(n); const uint64_t ml = sfc_metadata_size(r); char *meta = malloc(ml + 1);
    sfc_cache *ca = nullptr; int rc = !xyz || !valid || !meta || sfc_cache_create(r, (size_t)64 << 20, &ca);
    if (!rc) rc = sfc_cache_read_xyz_region(ca, (uint32_t)cx, 0, 0, (uint32_t)w, (uint32_t)h, xyz, 12, (size_t)w * 12, valid);
    if (!rc && ml) rc = sfc_read_metadata(r, meta, ml);
    if (ca) sfc_cache_destroy(ca);
    sfc_close(r);
    if (rc) { free(xyz); free(valid); free(meta); return -1; }
    meta[ml] = 0;
    snprintf(g_c.path, sizeof g_c.path, "%s", sfc); g_c.w = w; g_c.h = h; g_c.xyz = xyz; g_c.valid = valid; g_c.meta = meta;
    return 0;
}
static float *tfx_axis(const char *tifpath, int *w, int *h) {
    char sfc[1400]; const char *leaf;
    if (!tfx_resolve(tifpath, sfc, sizeof sfc, &leaf)) return nullptr;
    const int k = !strcmp(leaf, "x.tif") ? 0 : !strcmp(leaf, "y.tif") ? 1 : !strcmp(leaf, "z.tif") ? 2 : -1;
    if (k < 0 || tfx_load(sfc)) return nullptr;
    const size_t n = (size_t)g_c.w * g_c.h; float *a = malloc(n * 4); if (!a) return nullptr;
    for (size_t i = 0; i < n; i++) a[i] = g_c.valid[i] ? g_c.xyz[3 * i + k] : -1.f;
    *w = g_c.w; *h = g_c.h; return a;
}
static char *tfx_meta(const char *metapath) {
    char sfc[1400]; const char *leaf;
    if (!tfx_resolve(metapath, sfc, sizeof sfc, &leaf) || (strcmp(leaf, "meta.json") && strcmp(leaf, "x.tif")) || tfx_load(sfc)) return nullptr;
    return strdup(g_c.meta ? g_c.meta : "{}");
}
/* the threaded joint-XYZ writer (src/tiffw.h tfx_sfc_write): batches of patches in file order; workers fill + encode, the caller appends */
typedef struct {
    int w, h, nbx; long long b0, nb; double err; tfx_fill_fn fill; void *ctx;
    float (*joint)[4096][3]; uint8_t (*valid)[4096]; uint8_t **pk; size_t *pn; _Atomic long long next; _Atomic int fail;
} tfx_batch;
static void tfx_patch(tfx_batch *B, long long q, sfc_xyz_encoder *e) {
    const long long b = B->b0 + q; const int bx = (int)(b % B->nbx), by = (int)(b / B->nbx);
    float (*J)[3] = B->joint[q]; uint8_t *v = B->valid[q];
    memset(J, 0, 4096 * 12); memset(v, 0, 4096);
    B->fill(B->ctx, bx, by, J, v);
    for (int i = 0; i < 4096; i++) if (bx * 64 + i % 64 >= B->w || by * 64 + i / 64 >= B->h) v[i] = 0;   /* (the writer's in-surface mask) */
    const uint8_t *pk; size_t n;
    if (sfc_xyz_encoder_run(e, J, 12, 768, v, B->err, &pk, &n) || !(B->pk[q] = malloc(n))) { atomic_store(&B->fail, 1); return; }
    memcpy(B->pk[q], pk, n); B->pn[q] = n;
}
static void *tfx_worker(void *arg) {
    tfx_batch *B = arg; sfc_xyz_encoder *e = sfc_xyz_encoder_create(); if (!e) { atomic_store(&B->fail, 1); return nullptr; }
    for (long long q; (q = atomic_fetch_add(&B->next, 1)) < B->nb && !atomic_load(&B->fail);) tfx_patch(B, q, e);
    sfc_xyz_encoder_destroy(e);
    return nullptr;
}
static _Atomic int g_writers;   /* tfx_sfc_write calls running now (concurrent wraps share the threads) */
static int g_sfc_threads = 1;
void tfx_set_threads(int n) { g_sfc_threads = n > 0 ? n : 1; }
int tfx_sfc_write(const char *sfcpath, int w, int h, double err, const char *meta, tfx_fill_fn fill, void *ctx, int nthreads) {
    if (nthreads <= 0) nthreads = g_sfc_threads;
    const int active = atomic_fetch_add(&g_writers, 1) + 1; int nt = nthreads / active; if (nt < 1) nt = 1; if (nt > 64) nt = 64;
    sfc_channel d[3]; memset(d, 0, sizeof d); static const char *cn[3] = {"x", "y", "z"};
    for (int a = 0; a < 3; a++) { snprintf(d[a].name, sizeof d[a].name, "%s", cn[a]); d[a].width = (uint64_t)w; d[a].height = (uint64_t)h; d[a].dtype = SFC_F32;
                                  d[a].flags = SFC_COORDINATE | SFC_XYZ; d[a].components = 3; d[a].component = (uint32_t)a; d[a].tolerance = err; }
    char tmp[1500]; int rc = -1;
    { const char *sl = strrchr(sfcpath, '/'); if (sl) { snprintf(tmp, sizeof tmp, "mkdir -p '%.*s'", (int)(sl - sfcpath), sfcpath); if (system(tmp)) goto out; } }
    snprintf(tmp, sizeof tmp, "%s.tmp%d", sfcpath, (int)getpid()); unlink(tmp);
    sfc_writer *wr = nullptr; if (sfc_create(tmp, d, 3, meta, strlen(meta), &wr)) goto out;
    const int nbx = (w + 63) / 64; const long long nball = (long long)nbx * ((h + 63) / 64);
    const long long cap = nt > 1 ? 32LL * nt : 1;   /* patches per batch (52 KB of input each) */
    tfx_batch B = {.w = w, .h = h, .nbx = nbx, .err = err, .fill = fill, .ctx = ctx};
    B.joint = malloc(cap * sizeof *B.joint); B.valid = malloc(cap * sizeof *B.valid); B.pk = calloc(cap, sizeof *B.pk); B.pn = calloc(cap, sizeof *B.pn);
    sfc_xyz_encoder *e1 = nt > 1 ? nullptr : sfc_xyz_encoder_create();
    rc = !B.joint || !B.valid || !B.pk || !B.pn || (nt == 1 && !e1);
    for (long long b0 = 0; b0 < nball && !rc; b0 += cap) {
        B.b0 = b0; B.nb = nball - b0 < cap ? nball - b0 : cap; atomic_store(&B.next, 0);
        if (nt > 1) {
            pthread_t th[64]; const int k = B.nb < nt ? (int)B.nb : nt;
            for (int t = 0; t < k; t++) pthread_create(&th[t], nullptr, tfx_worker, &B);
            for (int t = 0; t < k; t++) pthread_join(th[t], nullptr);
        } else for (long long q = 0; q < B.nb && !atomic_load(&B.fail); q++) tfx_patch(&B, q, e1);
        rc = atomic_load(&B.fail);
        for (long long q = 0; q < B.nb; q++) {
            if (!rc) rc = sfc_write_xyz_encoded(wr, B.pk[q], B.pn[q], B.joint[q], 12, 768, B.valid[q]);
            free(B.pk[q]); B.pk[q] = nullptr;
        }
    }
    sfc_xyz_encoder_destroy(e1); free(B.joint); free(B.valid); free(B.pk); free(B.pn);
    if (rc) { sfc_cancel(wr); rc = -1; goto out; }
    if (sfc_finish(wr) || rename(tmp, sfcpath)) { unlink(tmp); rc = -1; goto out; }
    rc = 0;
out:
    atomic_fetch_sub(&g_writers, 1);
    return rc;
}
typedef struct { int w, h; const float *x, *y, *z; } tfx_grid;
static void tfx_fill_grid(void *ctx, int bx, int by, float (*joint)[3], uint8_t *valid) {
    const tfx_grid *G = ctx;
    for (int q = 0; q < 4096; q++) {
        const int c = bx * 64 + q % 64, r = by * 64 + q / 64; if (c >= G->w || r >= G->h) continue;
        const size_t i = (size_t)r * G->w + c; joint[q][0] = G->x[i]; joint[q][1] = G->y[i]; joint[q][2] = G->z[i];
        valid[q] = G->z[i] > 0 && isfinite(G->x[i]) && isfinite(G->y[i]) && isfinite(G->z[i]);   /* (sfc: valid samples need z > 0) */
    }
}
static int tfx_write(const char *sfcpath, int w, int h, const float *x, const float *y, const float *z, const char *meta, double err) {
    tfx_grid G = {w, h, x, y, z};
    return tfx_sfc_write(sfcpath, w, h, err, meta, tfx_fill_grid, &G, 0);
}
__attribute__((constructor)) static void tfx_install(void) { tfx_axis_hook = tfx_axis; tfx_meta_hook = tfx_meta; tfx_write_hook = tfx_write; }
