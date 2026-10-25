#include "tiffw.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>

static double g_sfc_err;   /* (tifxyz_set_sfc, below) */
static void put16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void put32(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); p[2] = (uint8_t)(v >> 16); p[3] = (uint8_t)(v >> 24); }

int tiffw_write(const char *path, int w, int h, int fmt, const void *data, int deflate) {
    if (w <= 0 || h <= 0 || (fmt != 1 && fmt != 3)) return -1;
    const int bps = fmt == 1 ? 1 : 4, rps = 64, ns = (h + rps - 1) / rps;
    const size_t row = (size_t)w * bps;
    uint8_t **strip = calloc(ns, sizeof *strip); uint32_t *len = calloc(ns, sizeof *len);
    if (!strip || !len) { free(strip); free(len); return -1; }
    int rc = 0;
    for (int s = 0; s < ns && !rc; s++) {   /* strips (deflate: zlib stream per strip, no predictor) */
        const int r0 = s * rps, nr = r0 + rps <= h ? rps : h - r0;
        const size_t raw = row * nr; const uint8_t *src = (const uint8_t *)data + row * r0;
        if (!deflate) { strip[s] = malloc(raw); if (!strip[s]) { rc = -1; break; } memcpy(strip[s], src, raw); len[s] = (uint32_t)raw; continue; }
        uLongf cap = compressBound(raw); strip[s] = malloc(cap);
        if (!strip[s] || compress2(strip[s], &cap, src, raw, 6) != Z_OK) { rc = -1; break; }
        len[s] = (uint32_t)cap;
    }
    FILE *f = rc ? nullptr : fopen(path, "wb");
    if (!f) rc = -1;
    if (!rc) {
        /* header, strips, then the IFD (12 entries) and the offset / byte-count arrays */
        uint8_t hd[8] = {'I', 'I', 42, 0}; size_t off = 8;
        uint64_t data_bytes = 0; for (int s = 0; s < ns; s++) data_bytes += len[s];
        const uint64_t ifd = 8 + data_bytes + ((8 + data_bytes) & 1);
        if (ifd + 2 + 12 * 12 + 4 + 8 * (uint64_t)ns > 0xffffffffull) rc = -1;   /* classic TIFF limit */
        put32(hd + 4, (uint32_t)ifd);
        if (!rc && fwrite(hd, 1, 8, f) != 8) rc = -1;
        uint32_t *soff = calloc(ns, sizeof *soff);
        for (int s = 0; s < ns && !rc; s++) { soff[s] = (uint32_t)off; if (fwrite(strip[s], 1, len[s], f) != len[s]) rc = -1; off += len[s]; }
        if (!rc && (off & 1)) { uint8_t z = 0; fwrite(&z, 1, 1, f); off++; }
        const uint32_t arr = (uint32_t)(ifd + 2 + 12 * 12 + 4);   /* strip offsets, then byte counts */
        uint8_t e[2 + 12 * 12 + 4]; memset(e, 0, sizeof e); put16(e, 12);
        const struct { uint16_t tag, type; uint32_t count, value; } ent[12] = {
            {256, 4, 1, (uint32_t)w}, {257, 4, 1, (uint32_t)h}, {258, 3, 1, (uint32_t)(bps * 8)}, {259, 3, 1, deflate ? 8u : 1u},
            {262, 3, 1, 1}, {273, 4, (uint32_t)ns, ns == 1 ? soff[0] : arr}, {277, 3, 1, 1}, {278, 4, 1, (uint32_t)rps},
            {279, 4, (uint32_t)ns, ns == 1 ? len[0] : arr + 4u * (uint32_t)ns}, {284, 3, 1, 1}, {317, 3, 1, 1}, {339, 3, 1, (uint32_t)fmt}};
        for (int k = 0; k < 12; k++) {
            uint8_t *p = e + 2 + 12 * k; put16(p, ent[k].tag); put16(p + 2, ent[k].type); put32(p + 4, ent[k].count);
            if (ent[k].type == 3 && ent[k].count == 1) put16(p + 8, (uint16_t)ent[k].value); else put32(p + 8, ent[k].value);
        }
        if (!rc && fwrite(e, 1, sizeof e, f) != sizeof e) rc = -1;
        if (!rc && ns > 1) {
            uint8_t *a = malloc((size_t)8 * ns);
            for (int s = 0; s < ns; s++) { put32(a + 4 * s, soff[s]); put32(a + 4 * (ns + s), len[s]); }
            if (fwrite(a, 1, (size_t)8 * ns, f) != (size_t)8 * ns) rc = -1;
            free(a);
        }
        free(soff);
    }
    if (f && fclose(f)) rc = -1;
    for (int s = 0; s < ns; s++) free(strip[s]);
    free(strip); free(len);
    return rc;
}

int tifxyz_write(const char *dir, int w, int h, const float *x, const float *y, const float *z, double step, double voxel_um, int winding,
                 const char *source) {
    char p[1400]; const float *ax[3] = {x, y, z}; static const char *nm[3] = {"x.tif", "y.tif", "z.tif"};
    const int sfc = g_sfc_err > 0 && tfx_write_hook;   /* tifxyz_set_sfc: <dir>.sfc instead of the directory */
    if (!sfc) {
        snprintf(p, sizeof p, "mkdir -p '%s'", dir); if (system(p)) return -1;
        for (int k = 0; k < 3; k++) { snprintf(p, sizeof p, "%s/%s", dir, nm[k]); if (tiffw_write(p, w, h, 3, ax[k], 1)) return -1; }
    }
    double lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY}; size_t quads = 0;
    for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) {
        const size_t i = (size_t)r * w + c;
        if (z[i] < 0) continue;
        for (int k = 0; k < 3; k++) { if (ax[k][i] < lo[k]) lo[k] = ax[k][i]; if (ax[k][i] > hi[k]) hi[k] = ax[k][i]; }
        if (r + 1 < h && c + 1 < w && z[i + 1] >= 0 && z[i + w] >= 0 && z[i + w + 1] >= 0) quads++;
    }
    if (!isfinite(lo[0])) for (int k = 0; k < 3; k++) lo[k] = hi[k] = -1;
    const char *uuid = strrchr(dir, '/'); uuid = uuid ? uuid + 1 : dir;
    const double area = (double)quads * step * step, cm2 = area * voxel_um * voxel_um * 1e-8;
    char meta[2048];
    snprintf(meta, sizeof meta, "{\"format\": \"tifxyz\", \"type\": \"seg\", \"uuid\": \"%s\", \"scale\": [%.9g, %.9g], \"bbox\": [[%.3f, %.3f, %.3f], [%.3f, %.3f, %.3f]], "
             "\"area_vx2\": %.1f, \"area_cm2\": %.6f, \"voxel_size_um\": %.6g, \"winding\": %d, \"source\": \"%s\"}\n",
             uuid, 1 / step, 1 / step, lo[0], lo[1], lo[2], hi[0], hi[1], hi[2], area, cm2, voxel_um, winding, source ? source : "ufsm");
    if (sfc) { const size_t n = strlen(dir); snprintf(p, sizeof p, "%s%s", dir, n > 4 && !strcmp(dir + n - 4, ".sfc") ? "" : ".sfc");
               return tfx_write_hook(p, w, h, x, y, z, meta, g_sfc_err); }
    snprintf(p, sizeof p, "%s/meta.json", dir);
    FILE *f = fopen(p, "w"); if (!f) return -1;
    fputs(meta, f);
    return fclose(f) ? -1 : 0;
}

float *(*tfx_axis_hook)(const char *, int *, int *) = nullptr;
char *(*tfx_meta_hook)(const char *) = nullptr;
int (*tfx_write_hook)(const char *, int, int, const float *, const float *, const float *, const char *, double) = nullptr;
static double g_sfc_err = 0;   /* the tentative definition above */
void tifxyz_set_sfc(double err) { g_sfc_err = err; }
#include <unistd.h>
float *tfx_axis_read(const char *tifpath, int *w, int *h, float *(*plain)(const char *, int *, int *)) {
    if (tfx_axis_hook && access(tifpath, F_OK)) { float *a = tfx_axis_hook(tifpath, w, h); if (a) return a; }
    return plain(tifpath, w, h);
}
FILE *tfx_meta_fopen(const char *metapath) {
    FILE *f = fopen(metapath, "rb"); if (f || !tfx_meta_hook) return f;
    char *m = tfx_meta_hook(metapath); if (!m) return nullptr;
    f = fmemopen(nullptr, strlen(m) + 1, "w+"); if (!f) { free(m); return nullptr; }
    fputs(m, f); rewind(f); free(m); return f;
}
char *tfx_meta_read(const char *metapath) {
    FILE *f = tfx_meta_fopen(metapath); if (!f) return nullptr;
    size_t cap = 16384, n = 0; char *b = malloc(cap);
    while (b) { n += fread(b + n, 1, cap - 1 - n, f); if (n < cap - 1) break; char *nb = realloc(b, cap *= 2); if (!nb) { free(b); b = nullptr; } else b = nb; }
    fclose(f); if (b) b[n] = 0; return b;
}
int tfx_exists(const char *path) {
    if (!access(path, F_OK)) return 1;
    if (!tfx_meta_hook) return 0;
    char *m = tfx_meta_hook(path); if (!m) return 0;   /* (a meta.json or x.tif path whose wrap is sfc-backed) */
    free(m); return 1;
}
