/* ufsm ink-place (2026-11-23, hfink): 2-D ink label crops on someone else's surface grid (e.g. the public 9.362 um herculaneum-ink-labels
   squares, values 0 background / 1 ink / 2 ignore / 255 outside) placed onto our meshes' label canvas, as on-the-fly ink label windows.

   PLACE.tsv, one line per crop (tab separated, '#' comments; earlier lines win where crops overlap):
     SEG  TIFXYZ  CROP.png|tif  Y0  X0  SY  SX  [REF|-  [REFMASK|-  [TAG]]]
   Y0 / X0: the canvas edge coordinate (mesh level-0 voxels; canvas px k spans [k, k + 1), inksplat's px k centre is k + 0.5 here) of the
   crop's top-left corner; SY / SX: canvas px per crop px. Per SEG one label raster at label_scale F = max(1, floor(--px-um / mesh voxel um))
   canvas px per label px (the ink-teach-up rule: no label px coarser than 4.8 um on a 2.4 um mesh), the crop resampled bilinear over its
   known px (0 / 1): a label px is supervised only when its whole bilinear footprint is known (2 / 255 / outside the crop stay unlabelled),
   ink where the interpolated ink share >= 0.5. OUT/<SEG>/labels.zarr + mask.zarr (zarr v3 2-D, the full canvas / F, empty chunks not
   written) and OUT/segs.json ({"segments": [{name, tifxyz, labels, mask, label_scale, crop}]}, one window per SEG over its placed bbox).

   REF (optional): a raster on the same mesh canvas (labels / a teacher map; its scale from its size vs the canvas) to check the placement:
   per crop the Pearson r between the placed 0/1 label and REF over the supervised px (REFMASK >= 128 when given) at offset 0 and the best
   offset within +-R label px (--search R), plus IoU (REF >= 128) and AUC (REF as the score) at offset 0. A placement whose best offset is
   not ~0 is off; r ~ 0 everywhere: wrong surface / grid. REF "ct:ROOT:KEY:UM:LEVEL": the CT sampled on the mesh (with --gray crops, e.g.
   a surface render layer: the grid check of a converted mesh).
   Options: --skip-ref-sup (px the REF mask supervises are not written: a set purely additive to labels we have), --gray (crops are images,
   0 = none; nothing written unless --map-tif), --tri / --bin T (crop decoding: 255 ink / 128 background / else unknown, a resampled
   --map-tif; or an ink map thresholded at T, 0 unknown), --map-tif SUFFIX (OUT/<SEG>SUFFIX: the label raster as one uint8 TIFF of the
   render's size, 255 / 128 / 0, the input of surf-resample), --crop-win r0,c0,h,w (every crop cut to that window), --ref-tri (REF is
   such a map). Crop "LABELS|MASK": our own label + supervision rasters as one crop. The pipeline (DESIGN 2026-11-23 "HF ink labels"). */
#include "imgops.h"
#include "inksplat.h"
#include "json.h"
#include "optcheck.h"
#include "sources.h"
#include "store.h"
#include "zarr3.h"
#include "tiffw.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <zstd.h>

static const char *sopt(int argc, char **argv, const char *k, const char *d) { for (int i = 4; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(k))) return argv[i + 1]; return d; }
static int ip_mkdir_p(const char *path) {
    char p[1400]; snprintf(p, sizeof p, "%s", path);
    for (char *q = p + 1; *q; q++) if (*q == '/') { *q = 0; if (mkdir(p, 0775) && errno != EEXIST) return -1; *q = '/'; }
    return mkdir(p, 0775) && errno != EEXIST ? -1 : 0;
}
/* the bbox a [r0, r0 + h) x [c0, c0 + w) of a H x W uint8 raster as a zarr v3 array (1024^2 zstd chunks, fill 0, chunks without a non-zero
   byte not written), OME pixel size um */
static int ip_write_zarr(const char *dir, const uint8_t *a, int64_t r0, int64_t c0, int64_t h, int64_t w, int64_t H, int64_t W, double um) {
    enum { CK = 1024 }; char p[1400];
    snprintf(p, sizeof p, "%s/0", dir); if (ip_mkdir_p(p)) return -1;
    snprintf(p, sizeof p, "%s/zarr.json", dir); FILE *f = fopen(p, "wb"); if (!f) return -1;
    fprintf(f, "{\"zarr_format\": 3, \"node_type\": \"group\", \"attributes\": {\"ufsm\": {\"content\": \"ink-place\"}, \"ome\": {\"version\": \"0.5\", \"multiscales\": [{\"axes\": "
               "[{\"name\": \"y\", \"type\": \"space\", \"unit\": \"micrometer\"}, {\"name\": \"x\", \"type\": \"space\", \"unit\": \"micrometer\"}], \"datasets\": [{\"path\": \"0\", "
               "\"coordinateTransformations\": [{\"type\": \"scale\", \"scale\": [%.9g, %.9g]}]}]}]}}}\n", um, um); fclose(f);
    snprintf(p, sizeof p, "%s/0/zarr.json", dir); f = fopen(p, "wb"); if (!f) return -1;
    fprintf(f, "{\"zarr_format\": 3, \"node_type\": \"array\", \"shape\": [%lld, %lld], \"data_type\": \"uint8\", \"chunk_grid\": {\"name\": \"regular\", \"configuration\": "
               "{\"chunk_shape\": [%d, %d]}}, \"chunk_key_encoding\": {\"name\": \"default\", \"configuration\": {\"separator\": \"/\"}}, \"fill_value\": 0, \"codecs\": "
               "[{\"name\": \"bytes\"}, {\"name\": \"zstd\", \"configuration\": {\"level\": 5, \"checksum\": false}}]}\n", (long long)H, (long long)W, CK, CK); fclose(f);
    uint8_t *c = malloc((size_t)CK * CK); const size_t cb = ZSTD_compressBound((size_t)CK * CK); uint8_t *z = malloc(cb); int rc = 0;
    for (int64_t q0 = r0 / CK * CK; q0 < r0 + h && !rc; q0 += CK) for (int64_t p0 = c0 / CK * CK; p0 < c0 + w && !rc; p0 += CK) {
        memset(c, 0, (size_t)CK * CK); int any = 0;
        for (int64_t r = 0; r < CK; r++) { const int64_t R = q0 + r; if (R < r0 || R >= r0 + h) continue;
            for (int64_t k = 0; k < CK; k++) { const int64_t C = p0 + k; if (C < c0 || C >= c0 + w) continue;
                const uint8_t v = a[(size_t)(R - r0) * w + (C - c0)]; c[(size_t)r * CK + k] = v; any |= v; } }
        if (!any) continue;
        const size_t zn = ZSTD_compress(z, cb, c, (size_t)CK * CK, 5); if (ZSTD_isError(zn)) { rc = -1; break; }
        snprintf(p, sizeof p, "%s/0/c/%lld", dir, (long long)(q0 / CK)); if (ip_mkdir_p(p)) { rc = -1; break; }
        snprintf(p, sizeof p, "%s/0/c/%lld/%lld", dir, (long long)(q0 / CK), (long long)(p0 / CK)); f = fopen(p, "wb");
        if (!f || fwrite(z, 1, zn, f) != zn) rc = -1;
        if (f) fclose(f);
    }
    free(c); free(z); return rc;
}
static int ip_write_map(const char *out, const char *seg, const char *suf, const uint8_t *LB, const uint8_t *MK, int64_t b0, int64_t a0, int64_t bh, int64_t bw, int64_t RH, int64_t RW) {
    uint8_t *m = calloc((size_t)RH * RW, 1); if (!m) return -1;   /* (RH x RW: render's size, (grid - 1) / scale + 1, as surf-resample reads maps) */
    for (int64_t r = 0; r < bh; r++) for (int64_t q = 0; q < bw; q++) { const size_t k = (size_t)r * bw + q; if (MK[k] && r + b0 < RH && q + a0 < RW) m[(size_t)(r + b0) * RW + q + a0] = LB[k] ? 255 : 128; }
    char p[1600]; snprintf(p, sizeof p, "%s/%s%s", out, seg, suf); const int rc = tiffw_write(p, (int)RW, (int)RH, 1, m, 1); free(m); return rc;
}
typedef struct { char seg[256], tx[1024], img[1024], ref[1024], refm[1024], tag[256]; double y0, x0, sy, sx, rfy, rfx; int gy, gx, ay, ax, o; } ipline;
static void ip_orient_dims(int o, int h, int w, int *oh, int *ow);
static inline size_t ip_orient_src(int o, int h, int w, int u, int v);
/* the canvas (level-0 px) of a tifxyz: grid / scale; voxel um from meta.json voxel_size_um (0 if absent); su / sv: the mean 3-D length
   (voxels) of one canvas px along the columns / rows (vertex steps over valid neighbours x scale), the physical px of a flattening */
typedef struct { float *A[3]; int w, h; double sc; } ipmesh;   /* z, y, x level-0 voxels per grid vertex (-1 invalid) */
static int ip_canvas(const char *tx, double *H, double *W, double *vum, double *su, double *sv, ipmesh *M) {
    char p[1400]; snprintf(p, sizeof p, "%s/meta.json", tx); FILE *f = tfx_meta_fopen(p); if (!f) { fprintf(stderr, "ink-place: cannot read %s\n", p); return -1; }
    char m[65536] = {0}; const size_t mn = fread(m, 1, sizeof m - 1, f); fclose(f);
    json *j = json_parse(m, mn); if (!j) return -1;
    const double sc = json_num(json_at(json_get(j, "scale"), 0), 0); *vum = json_num(json_get(j, "voxel_size_um"), 0); json_free(j);
    static const char *nm[3] = {"z", "y", "x"}; float *A[3] = {0}; int w = 0, h = 0, rc = 0;
    for (int k = 0; k < 3; k++) { int w2, h2; snprintf(p, sizeof p, "%s/%s.tif", tx, nm[k]); A[k] = img_read_f32(p, &w2, &h2);
        if (!A[k] || (k && (w2 != w || h2 != h))) { fprintf(stderr, "ink-place: cannot read %s\n", p); rc = -1; break; } w = w2; h = h2; }
    if (!rc && !(sc > 0)) { fprintf(stderr, "ink-place: %s: no scale\n", tx); rc = -1; }
    if (!rc) {
        *H = h / sc; *W = w / sc; double lu = 0, lv = 0; uint64_t nu = 0, nv = 0;
        for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) { const size_t i = (size_t)r * w + c; if (A[0][i] < 0) continue;
            if (c + 1 < w && A[0][i + 1] >= 0) { double d = 0; for (int k = 0; k < 3; k++) d += (A[k][i + 1] - A[k][i]) * (double)(A[k][i + 1] - A[k][i]); lu += sqrt(d); nu++; }
            if (r + 1 < h && A[0][i + w] >= 0) { double d = 0; for (int k = 0; k < 3; k++) d += (A[k][i + w] - A[k][i]) * (double)(A[k][i + w] - A[k][i]); lv += sqrt(d); nv++; } }
        *su = nu ? lu / nu * sc : 1; *sv = nv ? lv / nv * sc : 1;
    }
    if (M && !rc) { for (int k = 0; k < 3; k++) M->A[k] = A[k]; M->w = w; M->h = h; M->sc = sc; }
    else for (int k = 0; k < 3; k++) free(A[k]);
    return rc;
}
static uint8_t *ip_read_crop(const char *path, int *w, int *h) {
    const char *bar = strchr(path, '|');
    if (bar) {   /* "LABELS|MASK" (e.g. our own label + supervision rasters): 1 ink / 0 background where the mask is set, else 2 */
        char a[1100]; snprintf(a, sizeof a, "%.*s", (int)(bar - path), path); int w2, h2;
        uint8_t *L = ip_read_crop(a, w, h), *M = ip_read_crop(bar + 1, &w2, &h2);
        if (!L || !M || w2 != *w || h2 != *h) { free(L); free(M); return nullptr; }
        int lm = 0, mm = 0; for (size_t k = 0; k < (size_t)*w * *h; k++) { if (L[k] > lm) lm = L[k]; if (M[k] > mm) mm = M[k]; }
        for (size_t k = 0; k < (size_t)*w * *h; k++) L[k] = (mm == 1 ? M[k] : M[k] >= 128) ? (lm == 1 ? L[k] : L[k] >= 128) : 2;
        free(M); return L;
    }
    const size_t l = strlen(path);
    if (l > 4 && !strcasecmp(path + l - 4, ".png")) { uint8_t *rgb = png_read_rgb(path, w, h); if (!rgb) return nullptr;
        uint8_t *g = malloc((size_t)*w * *h); for (size_t k = 0; k < (size_t)*w * *h; k++) g[k] = rgb[3 * k]; free(rgb); return g; }
    if ((l > 4 && !strcasecmp(path + l - 4, ".jpg")) || (l > 5 && !strcasecmp(path + l - 5, ".jpeg"))) { uint8_t *rgb = jpeg_read_rgb(path, w, h); if (!rgb) return nullptr;
        uint8_t *g = malloc((size_t)*w * *h); for (size_t k = 0; k < (size_t)*w * *h; k++) g[k] = rgb[3 * k]; free(rgb); return g; }
    if ((l > 4 && !strcasecmp(path + l - 4, ".tif")) || (l > 5 && !strcasecmp(path + l - 5, ".tiff"))) return img_read_u8(path, w, h);
    int64_t H = 0, W = 0; uint8_t *t = inksplat_label_window(path, 0, 0, 1, 1, &H, &W); if (!t) return nullptr; free(t);
    *w = (int)W; *h = (int)H; return inksplat_label_window(path, 0, 0, H, W, nullptr, nullptr);   /* (a zarr v2 3-D: its middle layer) */
}
/* REF "ct:ROOT:KEY:UM:LEVEL": the CT on the mesh surface (bilinear vertex -> trilinear level-L voxel) at the label grid px centres:
   gh x gw px from canvas edge (cy0, cx0), F canvas px each; NaN off the mesh. 0 ok */
static int ip_ct_ref(const char *spec, const ipmesh *M, double cy0, double cx0, int64_t gh, int64_t gw, int F, float *G) {
    char b[2048]; snprintf(b, sizeof b, "%s", spec + 3); char *f[4] = {0}; int nf = 0; for (char *s = b, *t; nf < 4 && (t = strsep(&s, ":"));) f[nf++] = t;
    if (nf < 4) { fprintf(stderr, "ink-place: REF ct:ROOT:KEY:UM:LEVEL expected (%s)\n", spec); return -1; }
    const int lev = atoi(f[3]); const double V = (double)(1 << lev); store *st = store_open(f[0]); z3 *z = st ? pyramid_open_level(st, f[1], lev, atof(f[2]), nullptr) : nullptr;
    if (!z) { fprintf(stderr, "ink-place: CT %s %s level %d: %s\n", f[0], f[1], lev, z3_error()); return -1; }
    const int64_t *shp = z3_meta_of(z)->shape; double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30};
    float *P3 = malloc((size_t)gh * gw * 3 * sizeof *P3);
    for (int64_t r = 0; r < gh; r++) for (int64_t q = 0; q < gw; q++) {
        float *p = P3 + ((size_t)r * gw + q) * 3; p[0] = NAN;
        const double gv = (cy0 + (r + 0.5) * F - 0.5) * M->sc, gu = (cx0 + (q + 0.5) * F - 0.5) * M->sc;
        if (gu < 0 || gv < 0 || gu > M->w - 1 || gv > M->h - 1) continue;
        int c0 = (int)gu, r0 = (int)gv; if (c0 >= M->w - 1) c0 = M->w - 2; if (r0 >= M->h - 1) r0 = M->h - 2;
        const double fu = gu - c0, fv = gv - r0; const size_t i00 = (size_t)r0 * M->w + c0;
        if (M->A[0][i00] < 0 || M->A[0][i00 + 1] < 0 || M->A[0][i00 + M->w] < 0 || M->A[0][i00 + M->w + 1] < 0) continue;
        for (int d = 0; d < 3; d++) { const float *A = M->A[d];
            p[d] = (float)(((1 - fv) * ((1 - fu) * A[i00] + fu * A[i00 + 1]) + fv * ((1 - fu) * A[i00 + M->w] + fu * A[i00 + M->w + 1])) / V - (V - 1) / (2 * V));
            if (p[d] < lo[d]) lo[d] = p[d]; if (p[d] > hi[d]) hi[d] = p[d]; } }
    int64_t o[3], n[3]; size_t nv = 1; int rc = 0;
    for (int d = 0; d < 3; d++) { o[d] = (int64_t)floor(lo[d]) - 1; if (o[d] < 0) o[d] = 0; int64_t e = (int64_t)ceil(hi[d]) + 2; if (e > shp[d]) e = shp[d]; n[d] = e - o[d]; if (n[d] < 2) { n[d] = 0; } nv *= (size_t)(n[d] > 0 ? n[d] : 0); }
    uint8_t *B = nv && nv < ((size_t)3 << 30) ? malloc(nv) : nullptr;
    if (!B || z3_read(z, o, n, B, 6)) { fprintf(stderr, "ink-place: CT box %lld x %lld x %lld not read\n", (long long)n[0], (long long)n[1], (long long)n[2]); rc = -1; }
    for (size_t k = 0; !rc && k < (size_t)gh * gw; k++) {
        const float *p = P3 + k * 3; G[k] = NAN; if (p[0] != p[0]) continue;
        double c[3]; int64_t i[3]; int ok = 1; for (int d = 0; d < 3; d++) { c[d] = p[d] - o[d]; i[d] = (int64_t)floor(c[d]); c[d] -= i[d]; ok &= i[d] >= 0 && i[d] + 1 < n[d]; }
        if (!ok) continue;
        double v = 0; for (int q = 0; q < 8; q++) { const int a = q >> 2 & 1, bb = q >> 1 & 1, cc = q & 1;
            v += (a ? c[0] : 1 - c[0]) * (bb ? c[1] : 1 - c[1]) * (cc ? c[2] : 1 - c[2]) * B[((size_t)(i[0] + a) * n[1] + i[1] + bb) * n[2] + i[2] + cc]; }
        G[k] = (float)v; }
    free(B); free(P3); z3_close(z); return rc;
}
static long long g_ipwin[4];
static int g_refgray;   /* --ref-gray (ink-find): REF an ink probability map, 0 = unknown */
static int g_reftri;   /* --ref-tri: REF is a 255 ink / 128 background / 0 unknown map (a --map-tif, resampled) */   /* --crop-win r0,c0,h,w (h 0: off) */
static uint8_t *ip_get_crop(const ipline *e, int *w, int *h, double *y0, double *x0) {   /* the crop (cut to --crop-win) and its corner */
    uint8_t *c = ip_read_crop(e->img, w, h); *y0 = e->y0; *x0 = e->x0; if (!c || g_ipwin[2] <= 0) return c;
    const long long r0 = g_ipwin[0], c0 = g_ipwin[1], hh = g_ipwin[2], ww = g_ipwin[3]; uint8_t *o = calloc((size_t)hh * ww, 1);
    for (long long r = 0; r < hh; r++) for (long long q = 0; q < ww; q++) if (r0 + r < *h && c0 + q < *w && r0 + r >= 0 && c0 + q >= 0) o[r * ww + q] = c[(size_t)(r0 + r) * *w + c0 + q];
    free(c); *h = (int)hh; *w = (int)ww; *y0 = e->y0 + r0 * e->sy; *x0 = e->x0 + c0 * e->sx; return o;
}
static double ip_auc(const uint64_t hp[256], const uint64_t hn[256]) {
    double P = 0, N = 0, a = 0, cn = 0; for (int b = 0; b < 256; b++) { P += hp[b]; N += hn[b]; }
    for (int b = 0; b < 256; b++) { a += hp[b] * (cn + 0.5 * hn[b]); cn += hn[b]; }
    return P > 0 && N > 0 ? a / (P * N) : NAN;
}
int cmd_ink_place(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: ufsm ink-place PLACE.tsv OUT-DIR [--px-um 4.8] [--search 12] [--vox-um 2.4 (meshes without voxel_size_um)] [--skip-ref-sup] [--gray] [--tri] [--map-tif SUFFIX] [--crop-win r0,c0,h,w]\n"
                        "  PLACE.tsv: SEG TIFXYZ CROP.png Y0 X0 SY SX [REF [REFMASK [TAG]]] (canvas edge coords of the crop corner, canvas px per crop px;\n"
                        "  12th field fy,fx: REF px k centre = canvas centre k f (default: the size ratio, edge-aligned);\n"
                        "  11th field O: the crop oriented first (bit 1 transpose, 2 flip rows, 4 flip cols; ink-find's o);\n"
                        "  Y0 / X0 as gV: V source-grid px (x SY / SX); SY / SX as aU: U um source px over the mesh's mean 3-D px length per axis;\n"
                        "  crop values 0 background / 1 ink / else unlabelled) -> OUT/<SEG>/labels.zarr + mask.zarr, OUT/segs.json (on-the-fly ink windows)\n");
        return 2;
    }
    const char *out = argv[3]; const double pxum = opt_atof("--px-um", sopt(argc, argv, "--px-um", "4.8")), dvum = opt_atof("--vox-um", sopt(argc, argv, "--vox-um", "0"));
    const int SR = opt_atoi("--search", sopt(argc, argv, "--search", "12"));
    { const char *cw = sopt(argc, argv, "--crop-win", nullptr); if (cw && sscanf(cw, "%lld,%lld,%lld,%lld", &g_ipwin[0], &g_ipwin[1], &g_ipwin[2], &g_ipwin[3]) != 4) return 2; }
    int gray = 0, skipsup = 0, tri = 0; for (int i = 4; i < argc; i++) { g_reftri |= !strcmp(argv[i], opt_n("--ref-tri")); gray |= !strcmp(argv[i], opt_n("--gray")); skipsup |= !strcmp(argv[i], opt_n("--skip-ref-sup")); tri |= !strcmp(argv[i], opt_n("--tri")); }
    const int binT = opt_atoi("--bin", sopt(argc, argv, "--bin", "0"));
    const char *mapsuf = sopt(argc, argv, "--map-tif", nullptr);   /* also OUT/<SEG>SUFFIX: the whole canvas / F as one uint8 TIFF, 255 ink / 128 background / 0 unknown */   /* crops are images (e.g. a CT layer): check only, nothing written */
    FILE *fp = fopen(argv[2], "rb"); if (!fp) { fprintf(stderr, "ink-place: cannot open %s\n", argv[2]); return 1; }
    int n = 0, cap = 64; ipline *L = calloc(cap, sizeof *L); char line[8192];
    while (fgets(line, sizeof line, fp)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        char *f[12] = {0}; int nf = 0; for (char *s = line, *t; nf < 12 && (t = strsep(&s, "\t\n"));) if (*t || nf < 7) f[nf++] = t;
        if (nf < 7) { fprintf(stderr, "ink-place: bad line (need 7 tab-separated fields): %s", line); fclose(fp); return 1; }
        if (n == cap) { cap *= 2; L = realloc(L, cap * sizeof *L); }
        ipline *e = &L[n++]; memset(e, 0, sizeof *e);
        snprintf(e->seg, sizeof e->seg, "%s", f[0]); snprintf(e->tx, sizeof e->tx, "%s", f[1]); snprintf(e->img, sizeof e->img, "%s", f[2]);
        /* Y0 / X0 "gV": V in crop px of the source grid (x SY / SX); SY / SX "aU": a source grid of U um px over the mesh's mean 3-D px length per axis */
        e->gy = f[3][0] == 'g'; e->gx = f[4][0] == 'g'; e->ay = f[5][0] == 'a'; e->ax = f[6][0] == 'a';
        e->y0 = atof(f[3] + e->gy); e->x0 = atof(f[4] + e->gx); e->sy = atof(f[5] + e->ay); e->sx = atof(f[6] + e->ax);
        if (nf > 7 && strcmp(f[7], "-")) snprintf(e->ref, sizeof e->ref, "%s", f[7]);
        if (nf > 8 && strcmp(f[8], "-")) snprintf(e->refm, sizeof e->refm, "%s", f[8]);
        snprintf(e->tag, sizeof e->tag, "%s", nf > 9 ? f[9] : ""); e->o = nf > 10 ? atoi(f[10]) & 7 : 0;
        if (nf > 11 && sscanf(f[11], "%lf,%lf", &e->rfy, &e->rfx) != 2) e->rfy = e->rfx = 0;   /* REF px k centre = canvas centre k f (per axis; 0: the size ratio) */
        if (!(e->sy > 0 && e->sx > 0)) { fprintf(stderr, "ink-place: %s: SY / SX must be > 0\n", e->img); fclose(fp); return 1; }
    }
    fclose(fp);
    if (ip_mkdir_p(out)) { fprintf(stderr, "ink-place: cannot create %s\n", out); return 1; }
    char outabs[1200]; if (!realpath(out, outabs)) { fprintf(stderr, "ink-place: cannot resolve %s\n", out); return 1; }
    out = outabs;   /* (absolute label / mask paths in segs.json: train reads them as given, b43e5eae) */
    char jp[1400]; snprintf(jp, sizeof jp, "%s/segs.json", out); FILE *jf = fopen(jp, "wb"); if (!jf) return 1;
    fprintf(jf, "{\"note\": \"ink-place %s: 2-D crops placed on the mesh canvas (bilinear over known px, label px <= %g um)\",\n \"segments\": [\n", argv[2], pxum);
    uint8_t *done = calloc(n, 1); int nw = 0, rc = 0; double tot_sup = 0, tot_ink = 0, tot_skip = 0;
    printf("#seg\ttag\tcrop\tsup_mm2\tink_mm2\tconflict_px\tref_px\tr0\tbest_dy\tbest_dx\tr_best\tiou0\tauc0\tcentre_zyx\n");
    for (int i0 = 0; i0 < n && !rc; i0++) {
        if (done[i0]) continue;
        const char *seg = L[i0].seg; double CH, CW, vum, su, sv;
        ipmesh MS = {0}; if (ip_canvas(L[i0].tx, &CH, &CW, &vum, &su, &sv, &MS)) { rc = 1; break; }
        if (!(vum > 0)) { if (!(dvum > 0)) { fprintf(stderr, "ink-place: %s: no voxel_size_um in meta.json: pass --vox-um\n", L[i0].tx); rc = 1; break; }
            vum = dvum; }
        for (int i = i0; i < n; i++) { if (strcmp(L[i].seg, seg)) continue; ipline *e = &L[i];
            if (e->ay) { e->sy = e->sy / (vum * sv); e->ay = 0; } if (e->ax) { e->sx = e->sx / (vum * su); e->ax = 0; }
            if (e->gy) { e->y0 *= e->sy; e->gy = 0; } if (e->gx) { e->x0 *= e->sx; e->gx = 0; } }
        const int F = pxum / vum >= 1 ? (int)floor(pxum / vum + 1e-6) : 1; const double pum = F * vum, mm2 = 1e-6 * pum * pum;
        const int64_t RH = (int64_t)ceil(CH / F), RW = (int64_t)ceil(CW / F);
        const int64_t MH = (int64_t)fmin(RH, lround((MS.h - 1) / (MS.sc * F)) + 1), MW = (int64_t)fmin(RW, lround((MS.w - 1) / (MS.sc * F)) + 1);   /* --map-tif size */
        /* the union bbox (label px) of this SEG's crops */
        int64_t b0 = INT64_MAX, b1 = INT64_MIN, a0 = INT64_MAX, a1 = INT64_MIN;
        for (int i = i0; i < n; i++) { if (strcmp(L[i].seg, seg)) continue;
            int w0, h0, w, h; double ey0, ex0; uint8_t *c = ip_get_crop(&L[i], &w0, &h0, &ey0, &ex0); if (!c) { fprintf(stderr, "ink-place: cannot read %s\n", L[i].img); rc = 1; break; } free(c);
            ip_orient_dims(L[i].o, h0, w0, &h, &w);
            const int64_t r0 = (int64_t)floor(ey0 / F), r1 = (int64_t)ceil((ey0 + h * L[i].sy) / F), q0 = (int64_t)floor(ex0 / F), q1 = (int64_t)ceil((ex0 + w * L[i].sx) / F);
            if (r0 < b0) b0 = r0; if (r1 > b1) b1 = r1; if (q0 < a0) a0 = q0; if (q1 > a1) a1 = q1; }
        if (rc) break;
        if (b0 < 0) b0 = 0; if (a0 < 0) a0 = 0; if (b1 > RH) b1 = RH; if (a1 > RW) a1 = RW;
        if (b1 <= b0 || a1 <= a0) { fprintf(stderr, "ink-place: %s: every crop lies outside the canvas %.0f x %.0f\n", seg, CH, CW); for (int i = i0; i < n; i++) if (!strcmp(L[i].seg, seg)) done[i] = 1; continue; }
        const int64_t bh = b1 - b0, bw = a1 - a0;
        if ((double)bh * bw > 3e9) { fprintf(stderr, "ink-place: %s: placed bbox %lld x %lld too large\n", seg, (long long)bh, (long long)bw); rc = 1; break; }
        uint8_t *LB = calloc((size_t)bh * bw, 1), *MK = calloc((size_t)bh * bw, 1);
        if (!LB || !MK) { fprintf(stderr, "ink-place: out of memory (%lld x %lld)\n", (long long)bh, (long long)bw); rc = 1; break; }
        uint64_t seg_sup = 0;
        for (int i = i0; i < n && !rc; i++) {
            if (strcmp(L[i].seg, seg)) continue; done[i] = 1; const ipline *e = &L[i];
            int w0, h0, w, h; double ey0, ex0; uint8_t *c = ip_get_crop(e, &w0, &h0, &ey0, &ex0); ip_orient_dims(e->o, h0, w0, &h, &w);
            const int64_t r0 = (int64_t)floor(ey0 / F) - b0, r1 = (int64_t)ceil((ey0 + h * e->sy) / F) - b0, q0 = (int64_t)floor(ex0 / F) - a0, q1 = (int64_t)ceil((ex0 + w * e->sx) / F) - a0;
            const int64_t fh = r1 - r0, fw = q1 - q0; float *P = malloc((size_t)fh * fw * sizeof *P);   /* the placed crop on the label grid: NaN unknown, 0 / 1 (--gray: the value) */
            uint64_t ns = 0, ni = 0, ncf = 0;
            for (int64_t r = 0; r < fh; r++) for (int64_t q = 0; q < fw; q++) {
                float v = NAN; const int64_t R = r0 + r, Q = q0 + q;
                if (R >= 0 && R < bh && Q >= 0 && Q < bw) {
                    const double u = ((R + b0 + 0.5) * F - ey0) / e->sy - 0.5, t = ((Q + a0 + 0.5) * F - ex0) / e->sx - 0.5;
                    const int y0 = (int)floor(u), x0 = (int)floor(t); const double fy = u - y0, fx = t - x0; double s = 0, sw = 0; int bad = 0;
                    for (int dy = 0; dy < 2 && !bad; dy++) for (int dx = 0; dx < 2; dx++) {
                        const double g = (dy ? fy : 1 - fy) * (dx ? fx : 1 - fx); if (g <= 0) continue;
                        const int yy = y0 + dy, xx = x0 + dx;
                        if (yy < 0 || yy >= h || xx < 0 || xx >= w) { bad = 1; break; }
                        uint8_t cv = c[ip_orient_src(e->o, h0, w0, yy, xx)];
                        if (tri && !gray) cv = cv >= 192 ? 1 : cv >= 96 && cv < 160 ? 0 : 2;
                        else if (binT > 0 && !gray) cv = !cv ? 2 : cv >= binT;   /* --bin T: an ink map (0 unknown) thresholded */   /* --tri: 255 ink / 128 background / else unknown (a resampled --map-tif) */
                        if (gray ? cv == 0 : cv > 1) { bad = 1; break; }
                        s += g * cv; sw += g; }
                    if (!bad && sw > 0) v = gray ? (float)(s / sw) : s / sw >= 0.5;
                }
                P[(size_t)r * fw + q] = v;
            }
            free(c);
            uint8_t *skip = nullptr;   /* --skip-ref-sup: px the REF mask supervises (labels we already have) stay out */
            /* the placement check against REF */
            double r00 = NAN, rb = NAN, iou0 = NAN, auc0 = NAN; int bdy = 0, bdx = 0; uint64_t nref = 0;
            if (e->ref[0]) {
              const int64_t gh = fh + 2 * SR, gw = fw + 2 * SR;   /* REF on the label grid over the crop footprint + the search margin */
              const double cy0 = (r0 + b0 - SR) * (double)F, cx0 = (q0 + a0 - SR) * (double)F;
              float *G = malloc((size_t)gh * gw * sizeof *G);   /* NaN: REF unknown */
              if (!strncmp(e->ref, "ct:", 3)) { if (ip_ct_ref(e->ref, &MS, cy0, cx0, gh, gw, F, G)) { free(G); G = nullptr; } }
              else {
                int64_t RHr = 0, RWr = 0; uint8_t *t1 = inksplat_label_window(e->ref, 0, 0, 1, 1, &RHr, &RWr); const int ok1 = t1 != nullptr; free(t1);
                if (!ok1 || RHr < 1 || RWr < 1) { fprintf(stderr, "ink-place: cannot read REF %s\n", e->ref); rc = 1; free(P); break; }
                const double Fr = 0.5 * (CH / RHr + CW / RWr);   /* canvas px per REF px */
                const double fy = e->rfy > 0 ? e->rfy : Fr, fx = e->rfx > 0 ? e->rfx : Fr, hy = e->rfy > 0 ? 0.5 : 0, hx = e->rfx > 0 ? 0.5 : 0;   /* (explicit: centre-aligned) */
                int64_t ry0 = (int64_t)floor(cy0 / fy) - 2, rx0 = (int64_t)floor(cx0 / fx) - 2, ry1 = (int64_t)ceil((cy0 + gh * F) / fy) + 2, rx1 = (int64_t)ceil((cx0 + gw * F) / fx) + 2;
                uint8_t *RV = inksplat_label_window(e->ref, ry0, rx0, ry1 - ry0, rx1 - rx0, nullptr, nullptr), *RM = e->refm[0] ? inksplat_label_window(e->refm, ry0, rx0, ry1 - ry0, rx1 - rx0, nullptr, nullptr) : nullptr;
                if (!RV || (e->refm[0] && !RM)) { fprintf(stderr, "ink-place: cannot read REF window %s\n", e->ref); rc = 1; free(P); free(RV); free(RM); break; }
                int rmx = 0; for (size_t k = 0; k < (size_t)(ry1 - ry0) * (rx1 - rx0); k++) if (RV[k] > rmx) rmx = RV[k];
                int mmx = 0; if (RM) for (size_t k = 0; k < (size_t)(ry1 - ry0) * (rx1 - rx0); k++) if (RM[k] > mmx) mmx = RM[k];
                for (int64_t r = 0; r < gh; r++) for (int64_t q = 0; q < gw; q++) {
                    const int64_t y = (int64_t)floor((cy0 + (r + 0.5) * F - hy) / fy + hy) - ry0, x = (int64_t)floor((cx0 + (q + 0.5) * F - hx) / fx + hx) - rx0;
                    const size_t k = (size_t)y * (rx1 - rx0) + x; float g = rmx == 1 ? RV[k] * 255.f : RV[k];
                    if (RM && (mmx == 1 ? RM[k] * 255 : RM[k]) < 128) g = NAN;
                    if (g_reftri) g = RV[k] < 96 ? NAN : RV[k] >= 192 ? 255.f : 0.f;
                    G[(size_t)r * gw + q] = g; }
                free(RV); free(RM);
              }
              if (G) {
                for (int pass = 0; pass < 2; pass++) {   /* pass 0: every offset at stride 2 px; pass 1: offset 0 (all px) and +-1 around the best */
                    double best = -2; int by = 0, bx = 0;
                    for (int dy = -SR; dy <= SR; dy++) for (int dx = -SR; dx <= SR; dx++) {
                        if (pass == 0 && ((dy | dx) & 1)) continue;
                        if (pass == 1 && !(dy == 0 && dx == 0) && (abs(dy - bdy) > 1 || abs(dx - bdx) > 1)) continue;
                        const int st = pass ? 1 : 2; double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, m = 0;
                        for (int64_t r = 0; r < fh; r += st) for (int64_t q = 0; q < fw; q += st) {
                            const float p = P[(size_t)r * fw + q]; if (p != p) continue; const float g = G[(size_t)(r + SR + dy) * gw + (q + SR + dx)]; if (g != g) continue;
                            m++; sx += p; sy += g; sxx += (double)p * p; syy += (double)g * g; sxy += p * (double)g; }
                        const double vx = sxx - sx * sx / fmax(m, 1), vy = syy - sy * sy / fmax(m, 1), cr = m > 50 && vx > 0 && vy > 0 ? (sxy - sx * sy / m) / sqrt(vx * vy) : NAN;
                        if (dy == 0 && dx == 0 && pass == 1) { r00 = cr; nref = (uint64_t)m; }
                        if (cr == cr && cr > best) { best = cr; by = dy; bx = dx; }
                    }
                    bdy = by; bdx = bx; rb = best > -2 ? best : NAN;
                }
                uint64_t hp[256] = {0}, hn[256] = {0}, I = 0, U = 0;
                for (int64_t r = 0; r < fh; r++) for (int64_t q = 0; q < fw; q++) {
                    const float p = P[(size_t)r * fw + q]; if (p != p || gray) continue; const float g = G[(size_t)(r + SR) * gw + (q + SR)]; if (g != g) continue;
                    const int b = g < 0 ? 0 : g > 255 ? 255 : (int)g; if (p > 0) hp[b]++; else hn[b]++; I += p > 0 && g >= 128; U += p > 0 || g >= 128; }
                if (!gray) { iou0 = U ? (double)I / U : NAN; auc0 = ip_auc(hp, hn); }
                if (skipsup && e->refm[0] && strncmp(e->ref, "ct:", 3)) { skip = calloc((size_t)fh * fw, 1);
                    for (int64_t r = 0; r < fh; r++) for (int64_t q = 0; q < fw; q++) skip[(size_t)r * fw + q] = G[(size_t)(r + SR) * gw + (q + SR)] == G[(size_t)(r + SR) * gw + (q + SR)]; }
                free(G);
              }
            }
            uint64_t nsk = 0;
            for (int64_t r = 0; r < fh && !(gray && !mapsuf); r++) for (int64_t q = 0; q < fw; q++) {
                const float v = P[(size_t)r * fw + q]; if (v != v) continue;
                const int64_t R = r0 + r, Q = q0 + q; const size_t k = (size_t)R * bw + Q;
                if (gray) { if (!MK[k]) { MK[k] = 255; LB[k] = (uint8_t)fmin(255, fmax(1, lround(v))); ns++; } continue; }   /* (--gray --map-tif: the image itself, 0 = none) */
                if (skip && skip[(size_t)r * fw + q]) { nsk++; continue; }
                if (MK[k]) { ncf += (LB[k] == 255) != (v > 0); continue; }   /* an earlier line holds this px */
                MK[k] = 255; LB[k] = v > 0 ? 255 : 0; ns++; ni += v > 0;
            }
            free(skip); seg_sup += ns; tot_sup += ns * mm2; tot_ink += ni * mm2; tot_skip += nsk * mm2;
            double cz = NAN, cyy = NAN, cxx = NAN;   /* the crop centre's mesh point (level-0 voxels z y x), for spot checks */
            { const double gv = (ey0 + 0.5 * h * e->sy - 0.5) * MS.sc, gu = (ex0 + 0.5 * w * e->sx - 0.5) * MS.sc; const int a = (int)lround(gv), b = (int)lround(gu);
              if (a >= 0 && a < MS.h && b >= 0 && b < MS.w && MS.A[0][(size_t)a * MS.w + b] >= 0) { const size_t k = (size_t)a * MS.w + b; cz = MS.A[0][k]; cyy = MS.A[1][k]; cxx = MS.A[2][k]; } }
            printf("%s\t%s\t%s\t%.3f\t%.3f\t%llu\t%llu\t%.3f\t%d\t%d\t%.3f\t%.3f\t%.3f\t%.0f,%.0f,%.0f\n", seg, e->tag, e->img, ns * mm2, ni * mm2, (unsigned long long)ncf, (unsigned long long)nref, r00, bdy, bdx, rb, iou0, auc0, cz, cyy, cxx);
            free(P);
        }
        if (!rc && seg_sup && gray && mapsuf) {
            uint8_t *m = calloc((size_t)MH * MW, 1); char p[1600]; snprintf(p, sizeof p, "%s/%s%s", out, seg, mapsuf);
            for (int64_t r = 0; r < bh && r + b0 < MH; r++) memcpy(m + (size_t)(r + b0) * MW + a0, LB + (size_t)r * bw, (size_t)(a0 + bw <= MW ? bw : MW - a0));
            if (tiffw_write(p, (int)MW, (int)MH, 1, m, 1)) { fprintf(stderr, "ink-place: cannot write %s\n", p); rc = 1; } free(m);
        }
        if (!rc && seg_sup && !gray) {
            char d[1400], dm[1400]; snprintf(d, sizeof d, "%s/%s/labels.zarr", out, seg); snprintf(dm, sizeof dm, "%s/%s/mask.zarr", out, seg);
            if (ip_write_zarr(d, LB, b0, a0, bh, bw, RH, RW, pum) || ip_write_zarr(dm, MK, b0, a0, bh, bw, RH, RW, pum)) { fprintf(stderr, "ink-place: cannot write %s\n", d); rc = 1; }
            else if (mapsuf && ((double)RH * RW > 2.5e9 || ip_write_map(out, seg, mapsuf, LB, MK, b0, a0, bh, bw, MH, MW))) { fprintf(stderr, "ink-place: cannot write the %s map of %s\n", mapsuf, seg); rc = 1; }
            else fprintf(jf, "  %s{\"name\": \"%s\", \"tifxyz\": \"%s\", \"labels\": \"%s/0\", \"mask\": \"%s/0\", \"label_scale\": %d, \"crop\": [%lld, %lld, %lld, %lld]}\n",
                         nw++ ? "," : "", seg, L[i0].tx, d, dm, F, (long long)a0, (long long)b0, (long long)bw, (long long)bh);
        }
        free(LB); free(MK); for (int k = 0; k < 3; k++) free(MS.A[k]);
        fprintf(stderr, "ink-place: %s: F %d (%.3g um label px), canvas %.0f x %.0f (3-D px %.4f x %.4f um), %llu supervised label px\n", seg, F, pum, CH, CW, vum * sv, vum * su, (unsigned long long)seg_sup);
    }
    fprintf(jf, " ]\n}\n"); fclose(jf); free(done); free(L);
    fprintf(stderr, "ink-place: %d windows, %.1f mm2 supervised, %.1f mm2 ink (%.1f mm2 left out: REF mask supervised) -> %s\n", nw, tot_sup, tot_ink, tot_skip, jp);
    return rc;
}

/* ufsm ink-find REF REFMASK --crop C.png [--crop ..] --s S [--ref-f 1] [--down 64] [--top 3] [--refine 16]: where a 2-D label crop (0 / 1 /
   else unknown) lies on a mesh canvas whose labels REF (+ REFMASK) are known, when the crop's own grid is unrelated to the canvas (e.g. the
   2023 7.91 um UV renders vs the re-flattened 2.4 um meshes). S canvas px per crop px (the physical px ratio); REF has ref-f canvas px per
   px. Pearson r of the ink share over cells of --down canvas px, every position and the 8 orientations (o: bit 1 transpose, 2 flip rows,
   4 flip cols, applied as in ink-place), cells known on both sides >= 25 % of the crop's known cells; the best --top refined at down / 4.
   Prints per crop: o, the canvas edge coordinate of the oriented crop's corner (Y0 X0 for ink-place), r, cells. */
typedef struct { float *v; int h, w; } ipgrid;   /* NaN unknown */
static void ip_orient_dims(int o, int h, int w, int *oh, int *ow) { if (o & 1) { *oh = w; *ow = h; } else { *oh = h; *ow = w; } }
static inline size_t ip_orient_src(int o, int h, int w, int u, int v) {   /* oriented (u, v) -> source index (source h x w) */
    int oh, ow; ip_orient_dims(o, h, w, &oh, &ow); if (o & 2) u = oh - 1 - u; if (o & 4) v = ow - 1 - v;
    return o & 1 ? (size_t)v * w + u : (size_t)u * w + v;
}
static ipgrid ip_crop_cells(const uint8_t *c, int w, int h, double cpx, int o) {   /* cells of cpx crop px, oriented */
    const int gh0 = (int)ceil(h / cpx), gw0 = (int)ceil(w / cpx); ipgrid g0 = {malloc((size_t)gh0 * gw0 * sizeof(float)), gh0, gw0};
    for (int a = 0; a < gh0; a++) for (int b = 0; b < gw0; b++) {
        double s = 0, k = 0, t = 0; for (int y = (int)(a * cpx); y < (int)((a + 1) * cpx) && y < h; y++) for (int x = (int)(b * cpx); x < (int)((b + 1) * cpx) && x < w; x++) {
            t++; const uint8_t v = c[(size_t)y * w + x]; if (v <= 1) { k++; s += v; } }
        g0.v[(size_t)a * gw0 + b] = t > 0 && k >= 0.5 * t ? (float)(s / k) : NAN; }
    ipgrid g; ip_orient_dims(o, gh0, gw0, &g.h, &g.w); g.v = malloc((size_t)g.h * g.w * sizeof(float));
    for (int u = 0; u < g.h; u++) for (int v = 0; v < g.w; v++) g.v[(size_t)u * g.w + v] = g0.v[ip_orient_src(o, gh0, gw0, u, v)];
    free(g0.v); return g;
}
typedef struct { const ipgrid *R, *C; int y0, y1, x0, x1, mincnt; double best[8]; int by[8], bx[8], nb; _Atomic int next; } ipfind;
static void ip_find_one(ipfind *J, int y, int x, double *r, int *cnt) {
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, m = 0; const ipgrid *R = J->R, *C = J->C;
    for (int u = 0; u < C->h; u++) { const int ry = y + u; if (ry < 0 || ry >= R->h) continue;
        const float *cr = C->v + (size_t)u * C->w, *rr = R->v + (size_t)ry * R->w;
        for (int v = 0; v < C->w; v++) { const int rx = x + v; if (rx < 0 || rx >= R->w) continue; const float a = cr[v], b = rr[rx]; if (a != a || b != b) continue;
            m++; sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b; } }
    *cnt = (int)m; const double vx = sxx - sx * sx / fmax(m, 1), vy = syy - sy * sy / fmax(m, 1);
    *r = m >= J->mincnt && vx > 1e-9 && vy > 1e-9 ? (sxy - sx * sy / m) / sqrt(vx * vy) : -2;
}
#include <pthread.h>
#include <stdatomic.h>
static pthread_mutex_t g_ipf_mu = PTHREAD_MUTEX_INITIALIZER;
static void *ip_find_worker(void *p) {
    ipfind *J = p;
    for (int y; (y = atomic_fetch_add(&J->next, 1)) <= J->y1 - J->y0;) {
        for (int x = J->x0; x <= J->x1; x++) { double r; int cnt; ip_find_one(J, J->y0 + y, x, &r, &cnt); if (r <= -2) continue;
            pthread_mutex_lock(&g_ipf_mu);
            if (J->nb < 8 || r > J->best[J->nb - 1]) {   /* top-8 list, positions within 3 cells of a better one dropped */
                int dup = 0; for (int k = 0; k < J->nb && !dup; k++) dup = abs(J->by[k] - (J->y0 + y)) <= 3 && abs(J->bx[k] - x) <= 3 && J->best[k] >= r;
                if (!dup) { for (int k = 0; k < J->nb; k++) if (abs(J->by[k] - (J->y0 + y)) <= 3 && abs(J->bx[k] - x) <= 3) { for (int q = k; q + 1 < J->nb; q++) { J->best[q] = J->best[q + 1]; J->by[q] = J->by[q + 1]; J->bx[q] = J->bx[q + 1]; } J->nb--; k--; }
                    int k = J->nb < 8 ? J->nb++ : 7; while (k > 0 && J->best[k - 1] < r) { J->best[k] = J->best[k - 1]; J->by[k] = J->by[k - 1]; J->bx[k] = J->bx[k - 1]; k--; }
                    J->best[k] = r; J->by[k] = J->y0 + y; J->bx[k] = x; } }
            pthread_mutex_unlock(&g_ipf_mu); }
    }
    return nullptr;
}
static void ip_find_run(ipfind *J, int nth) {
    pthread_t th[64]; if (nth > 64) nth = 64; if (nth < 1) nth = 1; atomic_store(&J->next, 0); J->nb = 0;
    for (int i = 0; i < nth; i++) pthread_create(&th[i], nullptr, ip_find_worker, J);
    for (int i = 0; i < nth; i++) pthread_join(th[i], nullptr);
}
/* REF on cells of D canvas px: ink share (REF >= 128) over the known samples (REFMASK >= 128), sampled every D / 4 canvas px */
static ipgrid ip_ref_cells(void *hr, void *hm, int64_t RH, int64_t RW, double Fr, int D) {
    const int st = (int)fmax(1, floor(D / 4.0 / Fr)); const int64_t nh = RH / st, nw = RW / st; ipgrid g = {0};
    uint8_t *a = malloc((size_t)nh * nw), *b = hm ? malloc((size_t)nh * nw) : nullptr; int va = 0, vb = 0;
    if (inksplat_label_strided(hr, st, 0, nh, 0, nw, a, &va, 6) || (hm && inksplat_label_strided(hm, st, 0, nh, 0, nw, b, &vb, 6))) { free(a); free(b); return g; }
    const double cs = D / (st * Fr); g.h = (int)ceil(nh / cs); g.w = (int)ceil(nw / cs); g.v = malloc((size_t)g.h * g.w * sizeof(float));
    for (int i = 0; i < g.h; i++) for (int j = 0; j < g.w; j++) {
        double s = 0, k = 0, t = 0;
        for (int64_t y = (int64_t)(i * cs); y < (int64_t)((i + 1) * cs) && y < nh; y++) for (int64_t x = (int64_t)(j * cs); x < (int64_t)((j + 1) * cs) && x < nw; x++) {
            t++; const size_t q = (size_t)y * nw + x;
            if (g_reftri) { if (a[q] < 96) continue; k++; s += a[q] >= 192; continue; }   /* --ref-tri: 255 ink / 128 background / 0 unknown */
            if (g_refgray) { if (!a[q]) continue; k++; s += a[q] / 255.0; continue; }   /* --ref-gray: an ink map (0 unknown): its mean */
            if (b && (vb == 1 ? b[q] * 255 : b[q]) < 128) continue; k++; s += (va == 1 ? a[q] * 255 : a[q]) >= 128; }
        g.v[(size_t)i * g.w + j] = t > 0 && k >= 0.5 * t ? (float)(s / k) : NAN; }
    free(a); free(b); return g;
}
int cmd_ink_find(int argc, char **argv) {
    if (argc < 4 || !sopt(argc, argv, "--crop", nullptr) || !sopt(argc, argv, "--s", nullptr)) {
        fprintf(stderr, "usage: ufsm ink-find REF REFMASK|- --crop C.png [--crop ..] --s S [--ref-f 1] [--down 64] [--top 3] [--orient 0..7|all] [--threads 6] [--ref-tri]\n"
                        "  where a 2-D label crop lies on a mesh canvas with known labels REF: best orientation + canvas corner (ink-place Y0 X0 O) by cell Pearson r\n");
        return 2;
    }
    const double S = opt_atof("--s", sopt(argc, argv, "--s", "1")), Fr = opt_atof("--ref-f", sopt(argc, argv, "--ref-f", "1"));
    const int D = opt_atoi("--down", sopt(argc, argv, "--down", "64")), top = opt_atoi("--top", sopt(argc, argv, "--top", "3")), nth = opt_atoi("--threads", sopt(argc, argv, "--threads", "6"));
    for (int i = 4; i < argc; i++) { g_reftri |= !strcmp(argv[i], opt_n("--ref-tri")); g_refgray |= !strcmp(argv[i], opt_n("--ref-gray")); }
    const char *os = sopt(argc, argv, "--orient", "all"); const int o0 = strcmp(os, "all") ? atoi(os) : 0, o1 = strcmp(os, "all") ? o0 : 7;
    int64_t RH, RW, MH, MW; void *hr = inksplat_label_open(argv[2], &RH, &RW), *hm = strcmp(argv[3], "-") ? inksplat_label_open(argv[3], &MH, &MW) : nullptr;
    if (!hr || (strcmp(argv[3], "-") && !hm)) { fprintf(stderr, "ink-find: cannot open REF / REFMASK\n"); return 1; }
    ipgrid R = ip_ref_cells(hr, hm, RH, RW, Fr, D), R4 = ip_ref_cells(hr, hm, RH, RW, Fr, D / 4);
    if (!R.v || !R4.v) { fprintf(stderr, "ink-find: REF read failed\n"); return 1; }
    { size_t k = 0; for (size_t i = 0; i < (size_t)R.h * R.w; i++) k += R.v[i] == R.v[i];
      fprintf(stderr, "ink-find: REF %lld x %lld (canvas / ref-f %g): %d x %d cells of %d canvas px, %zu known\n", (long long)RH, (long long)RW, Fr, R.h, R.w, D, k); }
    printf("#crop\trank\to\tY0\tX0\tr_coarse\tr_fine\tcells\n");
    for (int i = 2; i + 1 < argc; i++) {
        if (strcmp(argv[i], opt_n("--crop"))) continue;
        int w, h; uint8_t *c = ip_read_crop(argv[i + 1], &w, &h); if (!c) { fprintf(stderr, "ink-find: cannot read %s\n", argv[i + 1]); continue; }
        double res[64][6]; int nres = 0;
        for (int o = o0; o <= o1; o++) {
            ipgrid C = ip_crop_cells(c, w, h, D / S, o); int kc = 0; for (int k = 0; k < C.h * C.w; k++) kc += C.v[k] == C.v[k];
            ipfind J = {.R = &R, .C = &C, .y0 = -C.h / 2, .y1 = R.h - C.h / 2, .x0 = -C.w / 2, .x1 = R.w - C.w / 2, .mincnt = kc / 4 > 20 ? kc / 4 : 20};
            ip_find_run(&J, nth);
            for (int k = 0; k < J.nb && k < top; k++) {   /* refine at D / 4 within +-6 fine cells */
                ipgrid C4 = ip_crop_cells(c, w, h, D / 4 / S, o); int k4 = 0; for (int q = 0; q < C4.h * C4.w; q++) k4 += C4.v[q] == C4.v[q];
                ipfind J4 = {.R = &R4, .C = &C4, .y0 = J.by[k] * 4 - 6, .y1 = J.by[k] * 4 + 6, .x0 = J.bx[k] * 4 - 6, .x1 = J.bx[k] * 4 + 6, .mincnt = k4 / 4 > 20 ? k4 / 4 : 20};
                ip_find_run(&J4, nth); double rr = J4.nb ? J4.best[0] : -2; int cnt = 0;
                if (J4.nb) { double t; ip_find_one(&J4, J4.by[0], J4.bx[0], &t, &cnt); }
                if (nres < 64) { res[nres][0] = o; res[nres][1] = J4.nb ? J4.by[0] * (D / 4.0) : J.by[k] * (double)D; res[nres][2] = J4.nb ? J4.bx[0] * (D / 4.0) : J.bx[k] * (double)D;
                    res[nres][3] = J.best[k]; res[nres][4] = rr; res[nres][5] = cnt; nres++; }
                free(C4.v); }
            free(C.v);
        }
        for (int a = 0; a < nres; a++) for (int b = a + 1; b < nres; b++) if (res[b][4] > res[a][4]) for (int q = 0; q < 6; q++) { const double t = res[a][q]; res[a][q] = res[b][q]; res[b][q] = t; }
        for (int a = 0; a < nres && a < top; a++) printf("%s\t%d\t%d\t%.0f\t%.0f\t%.3f\t%.3f\t%.0f\n", argv[i + 1], a, (int)res[a][0], res[a][1], res[a][2], res[a][3], res[a][4], res[a][5]);
        fflush(stdout); free(c);
    }
    free(R.v); free(R4.v); inksplat_label_close(hr); if (hm) inksplat_label_close(hm); return 0;
}

/* ufsm obj-tifxyz OBJ OUT-DIR --img W,H [--step 20] [--flip-v 1] [--vox-um 7.91]: a flattened OBJ (v = volume x y z voxels, vt = the
   render's uv in [0, 1]) -> tifxyz on the render's pixel grid: grid vertex (i, j) = render px (i step, j step) (row = (1 - v) H with
   --flip-v 1, v H with 0; col = u W), its 3-D point by barycentric interpolation of the face it falls in (-1 outside every face). The
   render's px (and labels drawn on it, e.g. the 2024 Scroll 5 segments' 7.91 um layers) then sit at canvas px = render px. */
int cmd_obj_tifxyz(int argc, char **argv) {
    const char *im = sopt(argc, argv, "--img", nullptr); int W = 0, H = 0;
    if (argc < 4 || !im || sscanf(im, "%d,%d", &W, &H) != 2 || W <= 0 || H <= 0) {
        fprintf(stderr, "usage: ufsm obj-tifxyz OBJ OUT-DIR --img W,H (render px) [--step 20] [--flip-v 1] [--vox-um 7.91]\n"); return 2; }
    const int st = opt_atoi("--step", sopt(argc, argv, "--step", "20")), fv = opt_atoi("--flip-v", sopt(argc, argv, "--flip-v", "1"));
    const double vum = opt_atof("--vox-um", sopt(argc, argv, "--vox-um", "7.91"));
    FILE *f = fopen(argv[2], "rb"); if (!f) { fprintf(stderr, "obj-tifxyz: cannot open %s\n", argv[2]); return 1; }
    size_t nv = 0, cv = 1 << 20, nt = 0, ct = 1 << 20, nf = 0, cf = 1 << 20;
    float *V = malloc(cv * 3 * sizeof *V), *T = malloc(ct * 2 * sizeof *T); int64_t *Fc = malloc(cf * 6 * sizeof *Fc); char ln[4096];
    while (fgets(ln, sizeof ln, f)) {
        if (ln[0] == 'v' && ln[1] == ' ') { if (nv == cv) { cv *= 2; V = realloc(V, cv * 3 * sizeof *V); } if (sscanf(ln + 2, "%f %f %f", &V[3 * nv], &V[3 * nv + 1], &V[3 * nv + 2]) == 3) nv++; }
        else if (ln[0] == 'v' && ln[1] == 't') { if (nt == ct) { ct *= 2; T = realloc(T, ct * 2 * sizeof *T); } if (sscanf(ln + 3, "%f %f", &T[2 * nt], &T[2 * nt + 1]) == 2) nt++; }
        else if (ln[0] == 'f' && ln[1] == ' ') {
            long long a[3], b[3]; char *p = ln + 2; int k = 0;
            for (; k < 3; k++) { while (*p == ' ') p++; char *e; a[k] = strtoll(p, &e, 10); if (e == p) break; b[k] = 0;
                if (*e == '/') { e++; if (*e != '/') b[k] = strtoll(e, &e, 10); if (*e == '/') { e++; strtoll(e, &e, 10); } } p = e; }
            if (k < 3 || !b[0] || !b[1] || !b[2]) continue;
            if (nf == cf) { cf *= 2; Fc = realloc(Fc, cf * 6 * sizeof *Fc); }
            for (int q = 0; q < 3; q++) { Fc[6 * nf + q] = a[q] < 0 ? (int64_t)nv + a[q] : a[q] - 1; Fc[6 * nf + 3 + q] = b[q] < 0 ? (int64_t)nt + b[q] : b[q] - 1; } nf++; }
    }
    fclose(f);
    const int gw = (W + st - 1) / st, gh = (H + st - 1) / st; float *X = malloc((size_t)gw * gh * sizeof *X), *Y = malloc((size_t)gw * gh * sizeof *Y), *Z = malloc((size_t)gw * gh * sizeof *Z);
    for (size_t k = 0; k < (size_t)gw * gh; k++) X[k] = Y[k] = Z[k] = -1;
    size_t nset = 0;
    for (size_t t = 0; t < nf; t++) {
        double px[3], py[3]; int ok = 1; const int64_t *fc = Fc + 6 * t;
        for (int q = 0; q < 3; q++) { if (fc[q] < 0 || fc[q] >= (int64_t)nv || fc[3 + q] < 0 || fc[3 + q] >= (int64_t)nt) { ok = 0; break; }
            px[q] = T[2 * fc[3 + q]] * W / st; py[q] = (fv ? 1 - T[2 * fc[3 + q] + 1] : T[2 * fc[3 + q] + 1]) * H / st; }
        if (!ok) continue;
        const double den = (py[1] - py[2]) * (px[0] - px[2]) + (px[2] - px[1]) * (py[0] - py[2]); if (fabs(den) < 1e-12) continue;
        const int x0 = (int)fmax(0, ceil(fmin(px[0], fmin(px[1], px[2])) - 1e-9)), x1 = (int)fmin(gw - 1, floor(fmax(px[0], fmax(px[1], px[2])) + 1e-9));
        const int y0 = (int)fmax(0, ceil(fmin(py[0], fmin(py[1], py[2])) - 1e-9)), y1 = (int)fmin(gh - 1, floor(fmax(py[0], fmax(py[1], py[2])) + 1e-9));
        for (int y = y0; y <= y1; y++) for (int x = x0; x <= x1; x++) {
            const double l0 = ((py[1] - py[2]) * (x - px[2]) + (px[2] - px[1]) * (y - py[2])) / den, l1 = ((py[2] - py[0]) * (x - px[2]) + (px[0] - px[2]) * (y - py[2])) / den, l2 = 1 - l0 - l1;
            if (l0 < -1e-6 || l1 < -1e-6 || l2 < -1e-6) continue;
            const size_t k = (size_t)y * gw + x; if (X[k] < 0) nset++;
            const float *a = V + 3 * fc[0], *b = V + 3 * fc[1], *c = V + 3 * fc[2];
            X[k] = (float)(l0 * a[0] + l1 * b[0] + l2 * c[0]); Y[k] = (float)(l0 * a[1] + l1 * b[1] + l2 * c[1]); Z[k] = (float)(l0 * a[2] + l1 * b[2] + l2 * c[2]); }
    }
    char src[1400]; snprintf(src, sizeof src, "ufsm obj-tifxyz %s (render %d x %d px, flip-v %d)", argv[2], W, H, fv);
    const int rc = tifxyz_write(argv[3], gw, gh, X, Y, Z, st, vum, -1, src);
    fprintf(stderr, "obj-tifxyz: %zu vertices, %zu uv, %zu faces -> %d x %d grid (step %d), %zu vertices set (%.1f %%)%s\n", nv, nt, nf, gw, gh, st, nset, 100.0 * nset / ((size_t)gw * gh), rc ? " WRITE FAILED" : "");
    free(V); free(T); free(Fc); free(X); free(Y); free(Z); return rc ? 1 : 0;
}
