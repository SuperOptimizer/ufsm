/* .sfc-backed tifxyz wraps (src/tiffw.h): surface-compressor (third_party/surfcomp, format 4) files holding one wrap's x / y / z grid
   as a joint XYZ coordinate triple (Euclidean tolerance in voxels) and its meta.json bytes as the container metadata. Installs the
   tiffw.h hooks at load time, so every tifxyz reader that goes through tfx_axis_read / tfx_meta_fopen reads <wrap>.sfc directly.
   The last decoded file is cached per thread (a reader asks for x, y, z of the same wrap in turn). */
#include "tiffw.h"
#include "surfcomp.h"
#include <math.h>
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
static int tfx_write(const char *sfcpath, int w, int h, const float *x, const float *y, const float *z, const char *meta, double err) {
    sfc_channel d[3]; memset(d, 0, sizeof d); static const char *cn[3] = {"x", "y", "z"};
    for (int a = 0; a < 3; a++) { snprintf(d[a].name, sizeof d[a].name, "%s", cn[a]); d[a].width = (uint64_t)w; d[a].height = (uint64_t)h; d[a].dtype = SFC_F32;
                                  d[a].flags = SFC_COORDINATE | SFC_XYZ; d[a].components = 3; d[a].component = (uint32_t)a; d[a].tolerance = err; }
    char tmp[1500];
    { const char *sl = strrchr(sfcpath, '/'); if (sl) { snprintf(tmp, sizeof tmp, "mkdir -p '%.*s'", (int)(sl - sfcpath), sfcpath); if (system(tmp)) return -1; } }
    snprintf(tmp, sizeof tmp, "%s.tmp%d", sfcpath, (int)getpid()); unlink(tmp);
    sfc_writer *wr = nullptr; if (sfc_create(tmp, d, 3, meta, strlen(meta), &wr)) return -1;
    float (*joint)[3] = malloc(4096 * 12); uint8_t *valid = malloc(4096); if (!joint || !valid) { sfc_cancel(wr); free(joint); free(valid); return -1; }
    int rc = 0;
    for (int by = 0; by < (h + 63) / 64 && !rc; by++) for (int bx = 0; bx < (w + 63) / 64 && !rc; bx++) {
        memset(joint, 0, 4096 * 12); memset(valid, 0, 4096);
        for (int q = 0; q < 4096; q++) {
            const int c = bx * 64 + q % 64, r = by * 64 + q / 64; if (c >= w || r >= h) continue;
            const size_t i = (size_t)r * w + c; joint[q][0] = x[i]; joint[q][1] = y[i]; joint[q][2] = z[i];
            valid[q] = z[i] > 0 && isfinite(x[i]) && isfinite(y[i]) && isfinite(z[i]);   /* (sfc: valid samples need z > 0) */
        }
        rc = sfc_write_xyz_block(wr, joint, 12, 768, valid);
    }
    free(joint); free(valid);
    if (rc) { sfc_cancel(wr); return -1; }
    if (sfc_finish(wr) || rename(tmp, sfcpath)) { unlink(tmp); return -1; }
    return 0;
}
__attribute__((constructor)) static void tfx_install(void) { tfx_axis_hook = tfx_axis; tfx_meta_hook = tfx_meta; tfx_write_hook = tfx_write; }
