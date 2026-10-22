/* ufsm dinovol-embed: the scrollprize dinovol model (src/dinovol.h) over a CT box with embed_volume.py's lattice tiling, blending and
   volume-wide normalisation; ufsm dinovol-eval: the sheet-identity check of the dense embedding along published surfaces. */
#include "dinovol.h"
#include "xnet.h"
#include "store.h"
#include "zarr3.h"
#include "z3w.h"
#include "json.h"
#include <pthread.h>
#include <sys/stat.h>
#include <zstd.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double dv_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
static int write_file(const char *dir, const char *name, const void *p, size_t n) {
    char path[2048]; snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "wb"); if (!f) { fprintf(stderr, "dinovol: cannot write %s\n", path); return -1; }
    const size_t w = fwrite(p, 1, n, f); fclose(f);
    return w == n ? 0 : -1;
}
static void *read_file(const char *path, size_t *n) {
    FILE *f = fopen(path, "rb"); if (!f) return nullptr;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void *p = malloc(sz > 0 ? (size_t)sz : 1); if (!p || fread(p, 1, (size_t)sz, f) != (size_t)sz) { free(p); fclose(f); return nullptr; }
    fclose(f); *n = (size_t)sz; return p;
}
static float h2f(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16; int e = (h >> 10) & 0x1f; uint32_t m = h & 0x3ffu, x;
    if (e == 0) { if (!m) x = s; else { while (!(m & 0x400u)) { m <<= 1; e--; } e++; m &= 0x3ffu; x = s | ((uint32_t)(e + 112) << 23) | (m << 13); } }
    else if (e == 31) x = s | 0x7f800000u | (m << 13);
    else x = s | ((uint32_t)(e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &x, 4); return f;
}
static int dump_dev_half(const char *dir, const char *name, const void *dev, size_t n) {   /* fp16 device array -> float32 file */
    uint16_t *h = malloc(n * 2); float *f = malloc(n * 4);
    xd2h(h, dev, n * 2);
    for (size_t i = 0; i < n; i++) f[i] = h2f(h[i]);
    const int rc = write_file(dir, name, f, n * 4); free(h); free(f); return rc;
}

/* --raw-in: one tile of uint8 voxels through the network, outputs as float32 files in --raw-out (parity with the PyTorch reference) */
static int raw_mode(dv_net *net, int argc, char **argv) {
    const char *in = opt(argc, argv, "--raw-in", nullptr), *out = opt(argc, argv, "--raw-out", nullptr), *norm = opt(argc, argv, "--norm", nullptr);
    const char *pts = opt(argc, argv, "--points", nullptr);
    const int T = atoi(opt(argc, argv, "--tile", "192")), reps = atoi(opt(argc, argv, "--reps", "1"));
    float c = 0, s = 1, hi = 255;
    if (!in || !out || !norm || sscanf(norm, "%f,%f,%f", &c, &s, &hi) != 3) { fprintf(stderr, "dinovol-embed --raw-in U8 --raw-out DIR --norm center,scale,hi [--tile T] [--points P.f32]\n"); return 2; }
    size_t nb = 0; uint8_t *u = read_file(in, &nb);
    if (!u || nb != (size_t)T * T * T) { fprintf(stderr, "dinovol: %s is not %d^3 bytes\n", in, T); return 1; }
    uint8_t *ud = xmalloc_dev(nb); xh2d(ud, u, nb);
    double t0 = 0, t1 = 0;
    for (int r = 0; r < reps; r++) {
        xnet_sync(); t0 = dv_now();
        if (dv_forward(net, ud, T, c, s, hi)) return 1;
        xnet_sync(); t1 = dv_now();
    }
    fprintf(stderr, "dinovol: tile %d^3 forward %.1f ms\n", T, (t1 - t0) * 1e3);
    const int g = T / 8;
    if (dump_dev_half(out, "tokens.f32", dv_tokens(net), (size_t)g * g * g * 480)) return 1;
    for (int k = 0; k < 3; k++) { int e, ch; const void *p = dv_skip(net, k, &e, &ch); char nm[32]; snprintf(nm, sizeof nm, "skip%d.f32", k);
        if (dump_dev_half(out, nm, p, (size_t)e * e * e * ch)) return 1; }
    { int e, ch; const void *p = dv_fpn(net, &e, &ch); if (dump_dev_half(out, "fpn.f32", p, (size_t)e * e * e * ch)) return 1; }
    if (pts) {
        size_t pb = 0; float *ph = read_file(pts, &pb); if (!ph || pb % 12) { fprintf(stderr, "dinovol: %s: float32 [P][3] points\n", pts); return 1; }
        const int P = (int)(pb / 12); int nd, ni; const int C = dv_dense_dims(net, &nd, &ni);
        float *pd = xmalloc_dev(pb), *acc = xmalloc_dev((size_t)P * C * 4); xh2d(pd, ph, pb);
        xnet_sync(); const double t2 = dv_now();
        if (dv_net_points(net, pd, nullptr, nullptr, P, acc)) return 1;
        xnet_sync(); fprintf(stderr, "dinovol: dense head %d points %.1f ms\n", P, (dv_now() - t2) * 1e3);
        float *o = malloc((size_t)P * C * 4); xd2h(o, acc, (size_t)P * C * 4);
        if (write_file(out, "dense_ink.f32", o, (size_t)P * C * 4)) return 1;
        free(o); free(ph); xfree_dev(pd); xfree_dev(acc);
    }
    free(u); xfree_dev(ud);
    return 0;
}


/* ---- volume-wide normalisation statistics (embed_volume.statistics_from_histogram, vc_reader's NumPy-linear percentiles) ---- */
typedef struct { int64_t count; double mean, std, p01, p99, median, mad, scale; int vmin, vmax; long sampled; } dv_stats;
static double hist_get(const int64_t *ends, const double *vals, int nv, int64_t k) {   /* value at sorted position k (searchsorted right) */
    int lo = 0, hi = nv;   /* first i with ends[i] > k */
    while (lo < hi) { const int m = (lo + hi) / 2; if (ends[m] > k) hi = m; else lo = m + 1; }
    return vals[lo < nv ? lo : nv - 1];
}
static double sorted_pct(const int64_t *ends, const double *vals, int nv, int64_t n, double percent) {
    const double q = percent / 100, v = (double)n * q + (1 + q * -1.0) - 1;
    if (v >= (double)(n - 1)) return hist_get(ends, vals, nv, n - 1);
    if (v < 0) return hist_get(ends, vals, nv, 0);
    const int64_t j = (int64_t)floor(v);
    const double a = hist_get(ends, vals, nv, j), b = hist_get(ends, vals, nv, j + 1), t = v - (double)j, d = b - a;
    return t >= 0.5 ? b - d * (1 - t) : a + d * t;
}
static int stats_from_hist(const int64_t *hist, int nbins, dv_stats *S) {
    double vals[256] = {0}, v2[256] = {0}; int64_t cnt[256], ends[256], c2[256]; int nv = 0; int64_t n = 0;
    if (nbins > 256) nbins = 256;
    for (int i = 1; i < nbins; i++) if (hist[i]) { vals[nv] = i; cnt[nv] = hist[i]; n += hist[i]; ends[nv] = n; nv++; }
    if (!n) return -1;
    double s = 0; for (int i = 0; i < nv; i++) s += vals[i] * (double)cnt[i];
    const double mean = s / (double)n; double q = 0; for (int i = 0; i < nv; i++) q += (vals[i] - mean) * (vals[i] - mean) * (double)cnt[i];
    const double std = sqrt(q / (double)n), lower = sorted_pct(ends, vals, nv, n, 1), upper = sorted_pct(ends, vals, nv, n, 99);
    const float lf = (float)lower, uf = (float)upper;
    for (int i = 0; i < nv; i++) { float c = (float)vals[i]; c = c < lf ? lf : c > uf ? uf : c; v2[i] = c; }
    const double median = sorted_pct(ends, v2, nv, n, 50);
    /* deviations |clipped - median| sorted stably */
    int *ord = malloc(nv * sizeof *ord); double *dv = malloc(nv * sizeof *dv);
    for (int i = 0; i < nv; i++) { dv[i] = fabs(v2[i] - median); ord[i] = i; }
    for (int i = 1; i < nv; i++) { const int k = ord[i]; int j = i - 1; while (j >= 0 && dv[ord[j]] > dv[k]) { ord[j + 1] = ord[j]; j--; } ord[j + 1] = k; }
    int64_t acc = 0; for (int i = 0; i < nv; i++) { v2[i] = dv[ord[i]]; c2[i] = cnt[ord[i]]; acc += c2[i]; ends[i] = acc; }
    const double mad = sorted_pct(ends, v2, nv, n, 50);
    free(ord); free(dv);
    double scale = 1.4826 * mad;
    if (!(isfinite(scale) && scale >= 1e-6)) { const double span = (double)uf - (double)lf; scale = fabs(std) >= 1e-6 ? fabs(std) : fabs(span) / 2 >= 1e-6 ? fabs(span) / 2 : 1.0; }
    S->count = n; S->mean = mean; S->std = std; S->p01 = lower; S->p99 = upper; S->median = median; S->mad = mad; S->scale = scale;
    S->vmin = (int)vals[0]; S->vmax = (int)vals[nv - 1];
    return 0;
}
/* nonzero histogram of every stride-th inner chunk along each axis of the whole array (absent chunks read as zeros) */
static int sample_stats(z3 *ct, int cs, int nthreads, dv_stats *S) {
    const z3_meta *m = z3_meta_of(ct);
    int64_t *hist = calloc(65536, sizeof *hist); long nread = 0;
    const int64_t c[3] = {m->chunk[0], m->chunk[1], m->chunk[2]}; uint8_t *buf = malloc((size_t)c[0] * c[1] * c[2]);
    for (int64_t z = 0; z < m->shape[0]; z += c[0] * cs)
        for (int64_t y = 0; y < m->shape[1]; y += c[1] * cs)
            for (int64_t x = 0; x < m->shape[2]; x += c[2] * cs) {
                const int64_t o[3] = {z, y, x};
                if (z3_read(ct, o, c, buf, nthreads)) { free(hist); free(buf); fprintf(stderr, "dinovol: stats read: %s\n", z3_error()); return -1; }
                nread++;
                for (size_t i = 0; i < (size_t)c[0] * c[1] * c[2]; i++) hist[buf[i]]++;
            }
    hist[0] = 0;
    const int rc = stats_from_hist(hist, 256, S); S->sampled = nread;
    free(hist); free(buf);
    if (rc) fprintf(stderr, "dinovol: no nonzero voxels in the statistics sample\n");
    return rc;
}

/* ---- sharded zarr v3 writer of [Z][Y][X](C) arrays (zstd inner chunks, index at the end with crc32c), shards written incrementally:
   compressed chunks are appended as their cells finish and the index is written when the shard closes ---- */
typedef struct { int64_t sz, sy, sx; FILE *f; char path[1200]; uint64_t off, *idx; } zw_shard;
typedef struct { char dir[1024]; int64_t shape[3]; int C, esize, shard, chunk, nc1; zw_shard *open; int nopen, cap; size_t chunk_bytes; uint8_t *cbuf, *zbuf; size_t zcap; } zw;
static int mkdirs(const char *path) {
    char p[1200]; snprintf(p, sizeof p, "%s", path);
    for (char *s = p + 1; *s; s++) if (*s == '/') { *s = 0; if (mkdir(p, 0755) && errno != EEXIST) return -1; *s = '/'; }
    return mkdir(p, 0755) && errno != EEXIST ? -1 : 0;
}
static int zw_create(zw *w, const char *dir, const int64_t shape[3], int C, const char *dtype, int esize, int shard, int chunk, const char *attrs) {
    memset(w, 0, sizeof *w);
    snprintf(w->dir, sizeof w->dir, "%s", dir); memcpy(w->shape, shape, sizeof w->shape); w->C = C; w->esize = esize; w->shard = shard; w->chunk = chunk;
    w->nc1 = shard / chunk; w->chunk_bytes = (size_t)chunk * chunk * chunk * (C ? C : 1) * esize;
    w->cbuf = malloc(w->chunk_bytes); w->zcap = ZSTD_compressBound(w->chunk_bytes); w->zbuf = malloc(w->zcap);
    if (mkdirs(dir)) { fprintf(stderr, "dinovol: cannot create %s\n", dir); return -1; }
    char path[1200]; snprintf(path, sizeof path, "%s/zarr.json", dir);
    FILE *f = fopen(path, "w"); if (!f) return -1;
    char shp[160], csh[160], ich[160];
    if (C) { snprintf(shp, sizeof shp, "[%lld,%lld,%lld,%d]", (long long)shape[0], (long long)shape[1], (long long)shape[2], C);
             snprintf(csh, sizeof csh, "[%d,%d,%d,%d]", shard, shard, shard, C); snprintf(ich, sizeof ich, "[%d,%d,%d,%d]", chunk, chunk, chunk, C); }
    else { snprintf(shp, sizeof shp, "[%lld,%lld,%lld]", (long long)shape[0], (long long)shape[1], (long long)shape[2]);
           snprintf(csh, sizeof csh, "[%d,%d,%d]", shard, shard, shard); snprintf(ich, sizeof ich, "[%d,%d,%d]", chunk, chunk, chunk); }
    fprintf(f, "{\"zarr_format\":3,\"node_type\":\"array\",\"shape\":%s,\"data_type\":\"%s\",\"chunk_grid\":{\"name\":\"regular\",\"configuration\":{\"chunk_shape\":%s}},"
               "\"chunk_key_encoding\":{\"name\":\"default\",\"configuration\":{\"separator\":\"/\"}},\"fill_value\":0,"
               "\"codecs\":[{\"name\":\"sharding_indexed\",\"configuration\":{\"chunk_shape\":%s,\"codecs\":[{\"name\":\"bytes\",\"configuration\":{\"endian\":\"little\"}},"
               "{\"name\":\"zstd\",\"configuration\":{\"level\":3,\"checksum\":false}}],\"index_codecs\":[{\"name\":\"bytes\",\"configuration\":{\"endian\":\"little\"}},"
               "{\"name\":\"crc32c\"}],\"index_location\":\"end\"}}],\"dimension_names\":%s,\"attributes\":%s}\n",
            shp, dtype, csh, ich, C ? "[\"z\",\"y\",\"x\",\"c\"]" : "[\"z\",\"y\",\"x\"]", attrs ? attrs : "{}");
    return fclose(f) ? -1 : 0;
}
static int zw_close_shard(zw *w, zw_shard *s) {
    const int nc = w->nc1 * w->nc1 * w->nc1; uint8_t *idx = malloc((size_t)nc * 16 + 4);
    for (int i = 0; i < nc; i++) { memcpy(idx + (size_t)i * 16, &s->idx[2 * i], 8); memcpy(idx + (size_t)i * 16 + 8, &s->idx[2 * i + 1], 8); }
    const uint32_t crc = crc32c(0, idx, (size_t)nc * 16); memcpy(idx + (size_t)nc * 16, &crc, 4);
    int rc = fwrite(idx, 1, (size_t)nc * 16 + 4, s->f) != (size_t)nc * 16 + 4;
    rc |= fclose(s->f) != 0; free(idx); free(s->idx);
    char fin[1200]; snprintf(fin, sizeof fin, "%.*s", (int)(strlen(s->path) - 4), s->path);   /* drop ".tmp" */
    if (!rc && rename(s->path, fin)) rc = 1;
    if (rc) fprintf(stderr, "dinovol: cannot write %s\n", fin);
    return rc ? -1 : 0;
}
static int zw_flush(zw *w) { int rc = 0; for (int i = 0; i < w->nopen; i++) rc |= zw_close_shard(w, &w->open[i]); w->nopen = 0; return rc; }
static zw_shard *zw_get(zw *w, int64_t sz, int64_t sy, int64_t sx) {
    for (int i = 0; i < w->nopen; i++) if (w->open[i].sz == sz && w->open[i].sy == sy && w->open[i].sx == sx) return &w->open[i];
    if (w->nopen == w->cap) { w->cap = w->cap ? 2 * w->cap : 16; w->open = realloc(w->open, w->cap * sizeof *w->open); }
    zw_shard *s = &w->open[w->nopen];
    char d[1200]; snprintf(d, sizeof d, "%s/c/%lld/%lld", w->dir, (long long)sz, (long long)sy);
    if (mkdirs(d)) return nullptr;
    if (w->C) { snprintf(d, sizeof d, "%s/c/%lld/%lld/%lld", w->dir, (long long)sz, (long long)sy, (long long)sx); if (mkdirs(d)) return nullptr;
                snprintf(s->path, sizeof s->path, "%s/0.tmp", d); }
    else snprintf(s->path, sizeof s->path, "%s/%lld.tmp", d, (long long)sx);
    s->f = fopen(s->path, "wb"); if (!s->f) { fprintf(stderr, "dinovol: cannot write %s\n", s->path); return nullptr; }
    s->sz = sz; s->sy = sy; s->sx = sx; s->off = 0;
    const int nc = w->nc1 * w->nc1 * w->nc1; s->idx = malloc((size_t)nc * 16); memset(s->idx, 0xff, (size_t)nc * 16);
    w->nopen++;
    return s;
}
/* data [nz][ny][nx](C) at output cell (oz, oy, ox), chunk aligned, nz / ny / nx multiples of the chunk */
static int zw_put(zw *w, int64_t oz, int64_t oy, int64_t ox, int nz, int ny, int nx, const void *data) {
    const int c = w->chunk; const size_t vb = (size_t)(w->C ? w->C : 1) * w->esize;
    for (int az = 0; az < nz; az += c) for (int ay = 0; ay < ny; ay += c) for (int ax = 0; ax < nx; ax += c) {
        const int64_t gz = oz + az, gy = oy + ay, gx = ox + ax;
        if (gz >= w->shape[0] || gy >= w->shape[1] || gx >= w->shape[2]) continue;
        int any = 0;
        for (int z = 0; z < c; z++) for (int y = 0; y < c; y++) {
            const uint8_t *src = (const uint8_t *)data + ((((size_t)(az + z) * ny) + ay + y) * nx + ax) * vb; uint8_t *dst = w->cbuf + (((size_t)z * c) + y) * c * vb;
            memcpy(dst, src, (size_t)c * vb);
            for (size_t i = 0; i < (size_t)c * vb && !any; i++) any = dst[i] != 0;
        }
        if (!any) continue;
        zw_shard *s = zw_get(w, gz / w->shard, gy / w->shard, gx / w->shard); if (!s) return -1;
        const size_t n = ZSTD_compress(w->zbuf, w->zcap, w->cbuf, w->chunk_bytes, 3);
        if (ZSTD_isError(n) || fwrite(w->zbuf, 1, n, s->f) != n) { fprintf(stderr, "dinovol: chunk write failed\n"); return -1; }
        const int ci = (int)((((gz % w->shard) / c) * w->nc1 + (gy % w->shard) / c) * w->nc1 + (gx % w->shard) / c);
        s->idx[2 * ci] = s->off; s->idx[2 * ci + 1] = n; s->off += n;
    }
    return 0;
}
static void zw_free(zw *w) { free(w->open); free(w->cbuf); free(w->zbuf); }

/* ---- the embedding run ---- */
enum { O_TOK, O_TOKU, O_DENSE, O_INK, O_N };
static const char *o_names[O_N] = {"tokens", "tokens_unit", "dense", "ink"};
typedef struct {   /* column CT prefetch */
    z3 *ct; int nthreads; int64_t o[3], n[3]; uint8_t *buf; int rc; pthread_t th; int busy;
} colread;
static void *colread_main(void *a) { colread *r = a; r->rc = z3_read(r->ct, r->o, r->n, r->buf, r->nthreads); return nullptr; }
static void colread_start(colread *R, int ky, int kx, int64_t bandz0, int Dband, int T, int stride, int margin, uint8_t *buf) {
    R->o[0] = bandz0; R->o[1] = (int64_t)ky * stride - margin; R->o[2] = (int64_t)kx * stride - margin; R->n[0] = Dband; R->n[1] = T; R->n[2] = T; R->buf = buf;
    pthread_create(&R->th, nullptr, colread_main, R); R->busy = 1;
}

int cmd_dinovol_embed(int argc, char **argv) {
    const char *ck = opt(argc, argv, "--ckpt", nullptr);
    if (!ck || (!opt(argc, argv, "--raw-in", nullptr) && argc < 5)) {
        fprintf(stderr, "usage: ufsm dinovol-embed <ct-root> <ct-array-key> <out-dir> --ckpt teacher_step_640000.pt --box z,y,x,nz,ny,nx\n"
                        "         [--outputs tokens,dense,ink] (also tokens_unit) [--dense-stride 1] [--dtype int8|float16] [--tile 192] [--margin M] [--ramp R]\n"
                        "         [--stats FILE (key=value lines or an earlier output's zarr.json) | --norm center,scale,hi] [--stats-chunk-stride 8]\n"
                        "         [--shard-cells K] [--gpu 0] [--gpu-mem GB] [--threads 8] [--cache DIR]\n"
                        "  embed_volume.py semantics: tiles on one global lattice (stride = tile - 2 margin - ramp, linear ramps summing to one), robust\n"
                        "  normalisation with volume-wide statistics (nonzero voxels of every 8th chunk per axis), blocks with no nonzero voxel stay zero.\n"
                        "  Output: one zarr v3 group, arrays over the CT array's whole grid at their step (tokens 8, dense / ink the dense stride),\n"
                        "  channel-last, sharded K lattice cells per edge; int8: tokens with a per-vector fp16 scale array (<name>_scale), unit\n"
                        "  vectors (tokens_unit, dense, ink) as round(127 v) (value_scale 1/127).\n"
                        "  ufsm dinovol-embed --ckpt PT --raw-in TILE.u8 --raw-out DIR --norm c,s,hi [--tile T] [--points P.f32]   one tile (parity)\n");
        return 2;
    }
    if (xnet_init(atoi(opt(argc, argv, "--gpu", "0")))) return 1;
    xnet_set_budget((size_t)(atof(opt(argc, argv, "--gpu-mem", "10")) * 1e9));
    if (opt(argc, argv, "--raw-in", nullptr)) {
        dv_net *net = dv_load(ck); if (!net) return 1;
        const int rc = raw_mode(net, argc, argv); dv_free(net); return rc;
    }
    const char *root = argv[2], *key = argv[3], *out = argv[4];
    const int nthreads = atoi(opt(argc, argv, "--threads", "8")), T = atoi(opt(argc, argv, "--tile", "192")), ds = atoi(opt(argc, argv, "--dense-stride", "1"));
    const int dtype = !strcmp(opt(argc, argv, "--dtype", "int8"), "float16") ? 1 : 0;
    const int overlap = 8 > T / 6 / 8 * 8 ? 8 : T / 6 / 8 * 8;
    const int margin = atoi(opt(argc, argv, "--margin", "-1")) >= 0 ? atoi(opt(argc, argv, "--margin", "0")) : overlap;
    const int ramp = atoi(opt(argc, argv, "--ramp", "-1")) > 0 ? atoi(opt(argc, argv, "--ramp", "0")) : overlap;
    const int stride = T - 2 * margin - ramp, support = stride + ramp;
    if (T % 16 || margin % 8 || ramp % 8 || ramp <= 0 || stride < ramp || ds < 1 || 8 % ds) { fprintf(stderr, "dinovol-embed: need tile %% 16, margin / ramp multiples of 8, tile - 2 margin >= 2 ramp, dense stride 1, 2, 4 or 8\n"); return 2; }
    int want[O_N] = {0};
    { char *s = strdup(opt(argc, argv, "--outputs", "tokens,dense,ink"));
      for (char *t = strtok(s, ","); t; t = strtok(nullptr, ",")) { int ok = 0; for (int o = 0; o < O_N; o++) if (!strcmp(t, o_names[o])) want[o] = ok = 1;
          if (!ok) { fprintf(stderr, "dinovol-embed: unknown output %s\n", t); return 2; } }
      free(s); }
    const int need_tok = want[O_TOK] || want[O_TOKU], need_den = want[O_DENSE] || want[O_INK];
    long long bv[6]; const char *box = opt(argc, argv, "--box", nullptr);
    store *st = store_open(root); z3 *ct = st ? z3_open(st, key, opt(argc, argv, "--cache", nullptr)) : nullptr;
    if (!ct) { fprintf(stderr, "dinovol-embed: cannot open %s %s: %s\n", root, key, z3_error()); return 1; }
    const z3_meta *m = z3_meta_of(ct);
    if (!box) { for (int d = 0; d < 3; d++) { bv[d] = 0; bv[3 + d] = m->shape[d]; } }
    else if (sscanf(box, "%lld,%lld,%lld,%lld,%lld,%lld", &bv[0], &bv[1], &bv[2], &bv[3], &bv[4], &bv[5]) != 6 || bv[3] <= 0 || bv[4] <= 0 || bv[5] <= 0 || bv[0] < 0 || bv[1] < 0 || bv[2] < 0) {
        fprintf(stderr, "dinovol-embed: --box z,y,x,nz,ny,nx\n"); return 2; }
    /* normalisation */
    dv_stats S = {0}; const char *sf = opt(argc, argv, "--stats", nullptr), *nm = opt(argc, argv, "--norm", nullptr);
    if (nm) { if (sscanf(nm, "%lf,%lf,%lf", &S.median, &S.scale, &S.p99) != 3) { fprintf(stderr, "--norm center,scale,hi\n"); return 2; } }
    else if (sf) {
        size_t n = 0; char *t = read_file(sf, &n); if (!t) { fprintf(stderr, "dinovol-embed: cannot read %s\n", sf); return 1; }
        t = realloc(t, n + 1); t[n] = 0;
        const char *a = strstr(t, "median_clipped"), *b = strstr(t, "robust_scale"), *c = strstr(t, "percentile_99");
        if (!a || !b || !c) { fprintf(stderr, "dinovol-embed: %s lacks median_clipped / robust_scale / percentile_99\n", sf); return 1; }
        S.median = atof(strpbrk(a + 14, "-0123456789.")); S.scale = atof(strpbrk(b + 12, "-0123456789.")); S.p99 = atof(strpbrk(c + 13, "-0123456789."));
        free(t);
    } else {
        const double t0 = dv_now();
        if (sample_stats(ct, atoi(opt(argc, argv, "--stats-chunk-stride", "8")), nthreads, &S)) return 1;
        fprintf(stderr, "dinovol-embed: statistics of %lld nonzero voxels in %ld sampled chunks (%.1f s): median %.4f scale %.4f p01 %.4f p99 %.4f mean %.4f std %.4f\n",
                (long long)S.count, S.sampled, dv_now() - t0, S.median, S.scale, S.p01, S.p99, S.mean, S.std);
    }
    dv_net *net = dv_load(ck); if (!net) return 1;
    int nd = 0, ni = 0; const int CD = dv_dense_dims(net, &nd, &ni); const int E = dv_embed_dim(net);
    /* cells and shards */
    int64_t c0[3], c1[3];
    for (int d = 0; d < 3; d++) { c0[d] = bv[d] / stride; c1[d] = (bv[d] + bv[3 + d] - 1) / stride + 1; }
    const int ncy = (int)(c1[1] - c0[1]), ncx = (int)(c1[2] - c0[2]);
    int K = atoi(opt(argc, argv, "--shard-cells", "0"));
    if (K <= 0) {   /* ~512-voxel shards, fewer when the dense accumulators + strips would not fit ~4 GB */
        K = (512 + stride / 2) / stride; if (K < 1) K = 1;
        for (; K > 1; K--) {
            const double zc = (double)K * stride, accb = need_den ? zc / ds * (support / ds) * (support / ds) * CD * 4.0 : 0;
            const double strips = need_den ? 2.0 * (ncx + 2) * zc / ds * (ramp / ds) * (stride / ds) * CD * 4.0 : 0;
            if (accb + strips < 4e9) break;
        }
    }
    const int ps = dv_patch(net), es_t = stride / ps, es_d = stride / ds;   /* cell edge in output units */
    int ck_t = es_t, ck_d = es_d;
    while (ck_t % 2 == 0 && (size_t)ck_t * ck_t * ck_t * E * (dtype ? 2 : 1) > (4u << 20)) ck_t /= 2;
    while (ck_d % 2 == 0 && (size_t)ck_d * ck_d * ck_d * nd * (dtype ? 2 : 1) > (4u << 20)) ck_d /= 2;
    /* output group */
    if (mkdirs(out)) { fprintf(stderr, "dinovol-embed: cannot create %s\n", out); return 1; }
    char attrs[4096];
    snprintf(attrs, sizeof attrs, "{\"format\":\"dinovol-embeddings/1 (ufsm zarr v3)\",\"checkpoint\":\"%s\",\"volume\":\"%s/%s\",\"volume_shape\":[%lld,%lld,%lld],"
             "\"box_zyx\":[%lld,%lld,%lld,%lld,%lld,%lld],\"lattice\":{\"tile\":%d,\"margin\":%d,\"ramp\":%d,\"stride\":%d,\"tile_origin\":\"k*stride - margin\",\"support\":\"[k*stride, k*stride + stride + ramp)\"},"
             "\"normalization\":{\"scheme\":\"robust\",\"statistics\":{\"median_clipped\":%.17g,\"robust_scale\":%.17g,\"percentile_01\":%.17g,\"percentile_99\":%.17g,\"mean\":%.17g,\"std\":%.17g,\"count\":%lld,\"sampled_chunks\":%ld},"
             "\"applied\":\"(min(x, percentile_99) - median_clipped) / robust_scale\"},\"dense_stride\":%d,\"dtype\":\"%s\",\"precision\":\"fp16 storage, fp32 accumulation (ufsm)\","
             "\"masked\":\"outputs whose block has no nonzero voxel are zero vectors\",\"index_to_voxel\":\"voxel block [i*step, (i+1)*step) of the volume\"}",
             ck, root, key, (long long)m->shape[0], (long long)m->shape[1], (long long)m->shape[2], bv[0], bv[1], bv[2], bv[3], bv[4], bv[5], T, margin, ramp, stride,
             S.median, S.scale, S.p01, S.p99, S.mean, S.std, (long long)S.count, S.sampled, ds, dtype ? "float16" : "int8");
    { char p[1200]; snprintf(p, sizeof p, "%s/zarr.json", out); FILE *f = fopen(p, "w");
      if (!f) { fprintf(stderr, "dinovol-embed: cannot write %s\n", p); return 1; }
      fprintf(f, "{\"zarr_format\":3,\"node_type\":\"group\",\"attributes\":%s}\n", attrs); fclose(f); }
    zw W[O_N], Wsc[O_N]; int has_sc[O_N] = {0};
    for (int o = 0; o < O_N; o++) {
        if (!want[o]) continue;
        const int step = o <= O_TOKU ? ps : ds, C = o <= O_TOKU ? E : o == O_DENSE ? nd : ni, unit = o != O_TOK;
        const int64_t shp[3] = {(m->shape[0] + step - 1) / step, (m->shape[1] + step - 1) / step, (m->shape[2] + step - 1) / step};
        char d[1200]; snprintf(d, sizeof d, "%s/%s", out, o_names[o]);
        char a[512]; snprintf(a, sizeof a, "{\"step_zyx\":[%d,%d,%d],\"values\":\"%s\"%s}", step, step, step,
                              o == O_TOK ? "blended x_norm_patchtokens" : o == O_TOKU ? "L2-normalized blended x_norm_patchtokens" : "unit-norm blended embedding",
                              dtype ? "" : unit ? ",\"value_scale\":0.007874015748031496" : ",\"quantization\":\"symmetric int8 per vector: value = q * scale; scale in tokens_scale\"");
        const int es = o <= O_TOKU ? es_t : es_d, cks = o <= O_TOKU ? ck_t : ck_d;
        if (zw_create(&W[o], d, shp, C, dtype ? "float16" : "int8", dtype ? 2 : 1, K * es, cks, a)) return 1;
        if (!dtype && !unit) { snprintf(d, sizeof d, "%s/%s_scale", out, o_names[o]); has_sc[o] = 1; if (zw_create(&Wsc[o], d, shp, 0, "float16", 2, K * es, cks, "{}")) return 1; }
    }
    /* device buffers */
    const int nz_slab_max = K * stride, Dcol = nz_slab_max + T;
    const int nyt = support / ps, syt = stride / ps, ryt = ramp / ps, nyd = support / ds, syd = stride / ds, ryd = ramp / ds;
    const int CAt = E + 1, CAd = CD;
    uint8_t *colh[2] = {malloc((size_t)Dcol * T * T), malloc((size_t)Dcol * T * T)};
    uint8_t *cold = xmalloc_dev((size_t)Dcol * T * T);
    float *acc_t = need_tok ? xmalloc_dev((size_t)(nz_slab_max / ps) * nyt * nyt * CAt * 4) : nullptr;
    float *acc_d = need_den ? xmalloc_dev((size_t)(nz_slab_max / ds) * nyd * nyd * CAd * 4) : nullptr;
    uint8_t *mask_t = xmalloc_dev((size_t)(nz_slab_max / ps) * nyt * nyt), *mask_d = need_den ? xmalloc_dev((size_t)(nz_slab_max / ds) * nyd * nyd) : nullptr;
    const size_t npt_max = (size_t)(support / ds) * nyd * nyd;
    float *coords = need_den ? xmalloc_dev(npt_max * 12) : nullptr, *wgt = need_den ? xmalloc_dev(npt_max * 4) : nullptr; int *rows = need_den ? xmalloc_dev(npt_max * 4) : nullptr, *cnt = xmalloc_dev(4);
    /* strips: top (this row, read) / next (written) per column index kx - c0x + 1 in [0, ncx + 1], left per output */
    const size_t st_t = (size_t)(nz_slab_max / ps) * ryt * syt * CAt, st_d = (size_t)(nz_slab_max / ds) * ryd * syd * CAd;
    const size_t lf_t = (size_t)(nz_slab_max / ps) * syt * ryt * CAt, lf_d = (size_t)(nz_slab_max / ds) * syd * ryd * CAd;
    float **top_t = calloc(ncx + 2, sizeof *top_t), **nxt_t = calloc(ncx + 2, sizeof *nxt_t), **top_d = calloc(ncx + 2, sizeof *top_d), **nxt_d = calloc(ncx + 2, sizeof *nxt_d);
    uint8_t *has_top = calloc(ncx + 2, 1), *has_nxt = calloc(ncx + 2, 1);
    for (int i = 0; i < ncx + 2; i++) {
        if (need_tok) { top_t[i] = xmalloc_dev(st_t * 4); nxt_t[i] = xmalloc_dev(st_t * 4); }
        if (need_den) { top_d[i] = xmalloc_dev(st_d * 4); nxt_d[i] = xmalloc_dev(st_d * 4); }
    }
    float *left_t = need_tok ? xmalloc_dev(lf_t * 4) : nullptr, *left_d = need_den ? xmalloc_dev(lf_d * 4) : nullptr; int has_left = 0;
    float *cell_t = need_tok ? xmalloc_dev((size_t)(nz_slab_max / ps) * syt * syt * CAt * 4) : nullptr, *cell_d = need_den ? xmalloc_dev((size_t)(nz_slab_max / ds) * syd * syd * CAd * 4) : nullptr;
    uint8_t *cmask_t = xmalloc_dev((size_t)(nz_slab_max / ps) * syt * syt), *cmask_d = need_den ? xmalloc_dev((size_t)(nz_slab_max / ds) * syd * syd) : nullptr;
    const size_t ob_max = (size_t)(nz_slab_max / ds < nz_slab_max / ps ? nz_slab_max / ps : nz_slab_max / ds) * (syd > syt ? syd : syt) * (syd > syt ? syd : syt) * (E > nd ? E : nd) * 2;
    void *obuf_d = xmalloc_dev(ob_max), *osc_d = xmalloc_dev(ob_max / 2 + 64); uint8_t *obuf_h = malloc(ob_max), *osc_h = malloc(ob_max / 2 + 64);
    if (!colh[0] || !colh[1] || !cold || !mask_t || !cnt || !obuf_d || !osc_d || (need_tok && (!acc_t || !left_t || !cell_t)) || (need_den && (!acc_d || !mask_d || !coords || !left_d || !cell_d))) {
        fprintf(stderr, "dinovol-embed: device memory allocation failed (lower --shard-cells)\n"); return 1; }
    size_t fr = 0, tot = 0; xnet_mem_info(&fr, &tot);
    fprintf(stderr, "dinovol-embed: box %lld,%lld,%lld +%lld,%lld,%lld: cells z %lld..%lld y %lld..%lld x %lld..%lld (stride %d, tile %d, margin %d, ramp %d), %d cells per shard edge, "
                    "dense stride %d, device %.2f GB in use\n", bv[0], bv[1], bv[2], bv[3], bv[4], bv[5], (long long)c0[0], (long long)c1[0], (long long)c0[1], (long long)c1[1],
            (long long)c0[2], (long long)c1[2], stride, T, margin, ramp, K, ds, xnet_mem_used() / 1e9);
    const double t_start = dv_now(); double t_net = 0, t_head = 0, t_read = 0; long ntiles = 0, nskip = 0, ncells = 0, npoints = 0;
    int rc = 0;
    /* slabs of K cells along z, aligned to the shard grid */
    for (int64_t zs = c0[0] / K; zs * K < c1[0] && !rc; zs++) {
        const int64_t cza = zs * K > c0[0] ? zs * K : c0[0], czb = (zs + 1) * K < c1[0] ? (zs + 1) * K : c1[0];
        const int64_t Z0 = cza * stride, Z1 = czb * stride; const int nzv = (int)(Z1 - Z0);
        const int64_t bandz0 = (cza - 1) * stride - margin; const int Dband = (int)(czb - cza) * stride + T;
        const int nzt = nzv / ps, nzd = nzv / ds;
        memset(has_top, 0, ncx + 2); memset(has_nxt, 0, ncx + 2);
        /* columns in raster order; prefetch the next column's CT */
        const int ky0 = (int)c0[1] - 1, kx0 = (int)c0[2] - 1, ncol = (ncy + 1) * (ncx + 1);
        colread R = {.ct = ct, .nthreads = nthreads}; int cur = 0;
        colread_start(&R, ky0, kx0, bandz0, Dband, T, stride, margin, colh[0]);
        for (int ci = 0; ci < ncol && !rc; ci++) {
            const int ky = ky0 + ci / (ncx + 1), kx = kx0 + ci % (ncx + 1), xi = kx - kx0;   /* xi: strip slot */
            if (kx == kx0) {   /* new row: next -> top */
                float **t; t = top_t; top_t = nxt_t; nxt_t = t; t = top_d; top_d = nxt_d; nxt_d = t;
                uint8_t *h = has_top; has_top = has_nxt; has_nxt = h; memset(has_nxt, 0, ncx + 2); has_left = 0;
            }
            double tr = dv_now(); pthread_join(R.th, nullptr); R.busy = 0; t_read += dv_now() - tr;
            if (R.rc) { fprintf(stderr, "dinovol-embed: CT read failed: %s\n", z3_error()); rc = 1; break; }
            const uint8_t *col = colh[cur];
            if (ci + 1 < ncol) { cur ^= 1; colread_start(&R, ky0 + (ci + 1) / (ncx + 1), kx0 + (ci + 1) % (ncx + 1), bandz0, Dband, T, stride, margin, colh[cur]); }
            const int in_y = ky >= c0[1] && ky < c1[1], in_x = kx >= c0[2] && kx < c1[2];
            const int ylo = ky < c0[1] ? stride : 0, yhi = ky == c1[1] - 1 ? stride : support, xlo = kx < c0[2] ? stride : 0, xhi = kx == c1[2] - 1 ? stride : support;
            /* any nonzero voxel in the needed region? */
            int anyv = 0;
            for (int z = (int)(Z0 - bandz0); z < (int)(Z1 - bandz0) && !anyv; z++)
                for (int y = margin + ylo; y < margin + yhi && !anyv; y++) { const uint8_t *r = col + ((size_t)z * T + y) * T + margin; for (int x = xlo; x < xhi; x++) if (r[x]) { anyv = 1; break; } }
            if (!anyv) { has_left = 0; continue; }
            xh2d(cold, col, (size_t)Dband * T * T);
            dv_blockmask(cold, Dband, T, T, (int)(Z0 - bandz0), margin, margin, ps, nzt, nyt, nyt, ylo / ps, yhi / ps, xlo / ps, xhi / ps, mask_t);
            if (need_den) dv_blockmask(cold, Dband, T, T, (int)(Z0 - bandz0), margin, margin, ds, nzd, nyd, nyd, ylo / ds, yhi / ds, xlo / ds, xhi / ds, mask_d);
            uint8_t *mt = malloc((size_t)nzt * nyt * nyt); xd2h(mt, mask_t, (size_t)nzt * nyt * nyt);
            {
                extern void dv_zero_dev(void *p, size_t bytes);
                if (need_tok) dv_zero_dev(acc_t, (size_t)nzt * nyt * nyt * CAt * 4);
                if (need_den) dv_zero_dev(acc_d, (size_t)nzd * nyd * nyd * CAd * 4);
            }
            int any_tile = 0;
            for (int64_t kz = cza - 1; kz < czb && !rc; kz++) {
                const int64_t az = kz * stride - margin, lo = kz * stride > Z0 ? kz * stride : Z0, hi = kz * stride + support < Z1 ? kz * stride + support : Z1;
                int v = 0;
                for (int64_t z = (lo - Z0) / ps; z < (hi - Z0) / ps && !v; z++) for (int i = 0; i < nyt * nyt; i++) if (mt[(size_t)z * nyt * nyt + i]) { v = 1; break; }
                if (!v) { nskip++; continue; }
                ntiles++; any_tile = 1;
                double t0 = dv_now();
                if (dv_forward(net, cold + (size_t)(az - bandz0) * T * T, T, (float)S.median, (float)S.scale, (float)S.p99)) { rc = 1; break; }
                if (need_tok) dv_tok_acc(acc_t, (int)((lo - Z0) / ps), (int)((hi - Z0) / ps), nyt, nyt, E, dv_tokens(net), T / ps, (int)((Z0 - az) / ps), margin / ps, margin / ps, ps, T, margin, ramp, mask_t);
                xnet_sync(); double t1 = dv_now(); t_net += t1 - t0;
                if (need_den) {
                    const int P = dv_points(mask_d, (int)((lo - Z0) / ds), (int)((hi - Z0) / ds), nyd, nyd, ds, (float)(Z0 - az), (float)margin, (float)margin, T, margin, ramp, coords, wgt, rows, cnt);
                    if (P > 0 && dv_net_points(net, coords, wgt, rows, P, acc_d)) { rc = 1; break; }
                    npoints += P;
                    xnet_sync(); t_head += dv_now() - t1;
                }
            }
            free(mt);
            if (rc) break;
            if (!any_tile) { has_left = 0; continue; }
            /* finish: strips to the next row (below kx, corner kx + 1), the cell (ky, kx), the left strip for kx + 1 */
            for (int pass = 0; pass < 2; pass++) {
                const int isd = pass == 1; if (isd ? !need_den : !need_tok) continue;
                const int s = isd ? ds : ps, nz = isd ? nzd : nzt, NS = isd ? nyd : nyt, sy = isd ? syd : syt, r = isd ? ryd : ryt, CA = isd ? CAd : CAt;
                float *A = isd ? acc_d : acc_t, **nx_ = isd ? nxt_d : nxt_t, **tp = isd ? top_d : top_t, *L = isd ? left_d : left_t, *cell = isd ? cell_d : cell_t;
                (void)s;
                /* below: A[:, sy:, :sy] -> next[xi] (added onto a corner already there) */
                dv_region(nx_[xi], r, sy, 0, 0, A, NS, NS, sy, 0, nz, r, sy, CA, has_nxt[xi]);
                /* corner: next[xi + 1] = 0 with [:, :, :r] = A[:, sy:, sy:] */
                {
                    extern void dv_zero_dev(void *p, size_t bytes);
                    dv_zero_dev(nx_[xi + 1], (size_t)nz * r * sy * CA * 4);
                    dv_region(nx_[xi + 1], r, sy, 0, 0, A, NS, NS, sy, sy, nz, r, r, CA, 0);
                }
                if (in_y && in_x) {
                    dv_region(cell, sy, sy, 0, 0, A, NS, NS, 0, 0, nz, sy, sy, CA, 0);
                    if (has_top[xi]) dv_region(cell, sy, sy, 0, 0, tp[xi], r, sy, 0, 0, nz, r, sy, CA, 1);
                    if (has_left) dv_region(cell, sy, sy, 0, 0, L, sy, r, 0, 0, nz, sy, r, CA, 1);
                }
                dv_region(L, sy, r, 0, 0, A, NS, NS, 0, sy, nz, sy, r, CA, 0);
            }
            has_nxt[xi] = 1; has_nxt[xi + 1] = 1; has_left = 1;
            if (!(in_y && in_x)) continue;
            /* masks of the cell region */
            {
                extern void dv_mask_crop(const uint8_t *src, int sny, int snx, uint8_t *dst, int nz, int ny, int nx);
                dv_mask_crop(mask_t, nyt, nyt, cmask_t, nzt, syt, syt);
                if (need_den) dv_mask_crop(mask_d, nyd, nyd, cmask_d, nzd, syd, syd);
            }
            ncells++;
            for (int o = 0; o < O_N && !rc; o++) {
                if (!want[o]) continue;
                const int isd = o >= O_DENSE, s = isd ? ds : ps, nz = isd ? nzd : nzt, sy = isd ? syd : syt, CA = isd ? CAd : CAt, C = o <= O_TOKU ? E : o == O_DENSE ? nd : ni;
                const int c0c = o == O_INK ? nd : 0, wch = isd ? -1 : E, unit = o != O_TOK, unit8 = unit && !dtype;
                dv_finish(isd ? cell_d : cell_t, sy, sy, CA, 0, 0, nz, sy, sy, c0c, C, wch, unit, isd ? cmask_d : cmask_t, dtype, obuf_d, osc_d, unit8);
                const size_t ob = (size_t)nz * sy * sy * C * (dtype ? 2 : 1);
                xd2h(obuf_h, obuf_d, ob);
                const int64_t oz = Z0 / s, oy = (int64_t)ky * stride / s, ox = (int64_t)kx * stride / s;
                if (zw_put(&W[o], oz, oy, ox, nz, sy, sy, obuf_h)) { rc = 1; break; }
                if (has_sc[o]) { xd2h(osc_h, osc_d, (size_t)nz * sy * sy * 2); if (zw_put(&Wsc[o], oz, oy, ox, nz, sy, sy, osc_h)) { rc = 1; break; } }
            }
            /* shard row done (last row of the shard in y, or of the box) at the row's last column */
            if (kx == c1[2] - 1 && (ky == c1[1] - 1 || (ky + 1) % K == 0))
                for (int o = 0; o < O_N; o++) if (want[o]) { rc |= zw_flush(&W[o]); if (has_sc[o]) rc |= zw_flush(&Wsc[o]); }
            if (ncells % 4 == 0 || ci == ncol - 1) {
                const double el = dv_now() - t_start;
                fprintf(stderr, "\rdinovol-embed: slab %lld cells %ld tiles %ld (+%ld skipped) %.1f s: net %.1f head %.1f read wait %.1f, %.2f Mvox/s   ", (long long)zs, ncells, ntiles, nskip,
                        el, t_net, t_head, t_read, (double)ncells * stride * stride * stride / el / 1e6);
            }
        }
        if (R.busy) pthread_join(R.th, nullptr);
        for (int o = 0; o < O_N; o++) if (want[o]) { rc |= zw_flush(&W[o]); if (has_sc[o]) rc |= zw_flush(&Wsc[o]); }
    }
    const double el = dv_now() - t_start;
    fprintf(stderr, "\ndinovol-embed: %ld cells (%.1f Mvox) in %.1f s = %.2f Mvox/s; %ld tiles (%.0f ms each: net %.0f, head %.0f), %ld dense points, %ld tiles skipped; peak device %.2f GB\n",
            ncells, ncells * (double)stride * stride * stride / 1e6, el, ncells * (double)stride * stride * stride / el / 1e6, ntiles, ntiles ? el / ntiles * 1e3 : 0,
            ntiles ? t_net / ntiles * 1e3 : 0, ntiles ? t_head / ntiles * 1e3 : 0, npoints, nskip, (xnet_mem_used() + xnet_mem_cached()) / 1e9);
    for (int o = 0; o < O_N; o++) if (want[o]) { zw_free(&W[o]); if (has_sc[o]) zw_free(&Wsc[o]); }
    dv_free(net); z3_close(ct); store_close(st);
    return rc;
}

/* ---- reader of the embedding arrays (any region, dequantised to float32: value_scale, and the <name>_scale array when present) ---- */
typedef struct { char dir[1024]; int64_t shape[3]; int C, esize, shard, chunk, dt; double vscale; int64_t ish[3]; uint8_t *idx; int64_t csh, cci; uint8_t *cb; int64_t cshard[3]; } zr;
static int zr_open(zr *r, const char *dir) {
    memset(r, 0, sizeof *r); snprintf(r->dir, sizeof r->dir, "%s", dir); r->ish[0] = -1; r->csh = -1;
    char p[1200]; snprintf(p, sizeof p, "%s/zarr.json", dir); size_t n = 0; char *t = read_file(p, &n);
    if (!t) return -1;
    json *j = json_parse(t, n); free(t); if (!j) return -1;
    const json *sh = json_get(j, "shape"), *dtj = json_get(j, "data_type"), *cs = json_path(j, "chunk_grid.configuration.chunk_shape"), *ic = json_path(j, "codecs.0.configuration.chunk_shape");
    if (!sh || !dtj || !cs || !ic) { json_free(j); return -1; }
    for (int d = 0; d < 3; d++) r->shape[d] = (int64_t)json_num(json_at(sh, d), 0);
    r->C = sh->n > 3 ? (int)json_num(json_at(sh, 3), 0) : 0;
    r->shard = (int)json_num(json_at(cs, 0), 0); r->chunk = (int)json_num(json_at(ic, 0), 0);
    const char *dt = json_str(dtj, ""); r->dt = !strcmp(dt, "int8") ? 0 : !strcmp(dt, "float16") ? 1 : !strcmp(dt, "float32") ? 2 : -1; r->esize = r->dt == 0 ? 1 : r->dt == 1 ? 2 : 4;
    r->vscale = json_num(json_path(j, "attributes.value_scale"), 1.0);
    json_free(j);
    if (r->dt < 0 || r->chunk <= 0 || r->shard % r->chunk) return -1;
    const size_t cb = (size_t)r->chunk * r->chunk * r->chunk * (r->C ? r->C : 1) * r->esize; r->cb = malloc(cb);
    return r->cb ? 0 : -1;
}
static void zr_close(zr *r) { free(r->idx); free(r->cb); }
/* the decoded chunk holding output cell (z, y, x), or nullptr when absent (zeros) */
static const uint8_t *zr_chunk(zr *r, int64_t z, int64_t y, int64_t x) {
    const int64_t s[3] = {z / r->shard, y / r->shard, x / r->shard}; const int nc1 = r->shard / r->chunk, nc = nc1 * nc1 * nc1;
    const int64_t ci = ((z % r->shard) / r->chunk * nc1 + (y % r->shard) / r->chunk) * nc1 + (x % r->shard) / r->chunk;
    char p[1300];
    if (r->C) snprintf(p, sizeof p, "%s/c/%lld/%lld/%lld/0", r->dir, (long long)s[0], (long long)s[1], (long long)s[2]);
    else snprintf(p, sizeof p, "%s/c/%lld/%lld/%lld", r->dir, (long long)s[0], (long long)s[1], (long long)s[2]);
    if (memcmp(s, r->ish, sizeof s)) {   /* shard index */
        free(r->idx); r->idx = nullptr; memcpy(r->ish, s, sizeof s); r->csh = -1;
        FILE *f = fopen(p, "rb");
        if (f) { r->idx = malloc((size_t)nc * 16); fseek(f, -(long)(nc * 16 + 4), SEEK_END); if (fread(r->idx, 16, nc, f) != (size_t)nc) { free(r->idx); r->idx = nullptr; } fclose(f); }
    }
    if (!r->idx) return nullptr;
    uint64_t off, len; memcpy(&off, r->idx + ci * 16, 8); memcpy(&len, r->idx + ci * 16 + 8, 8);
    if (off == UINT64_MAX) return nullptr;
    if (r->csh == 0 && r->cci == ci && !memcmp(r->cshard, s, sizeof s)) return r->cb;
    FILE *f = fopen(p, "rb"); if (!f) return nullptr;
    void *zb = malloc(len); fseek(f, (long)off, SEEK_SET);
    const size_t got = fread(zb, 1, len, f); fclose(f);
    const size_t cb = (size_t)r->chunk * r->chunk * r->chunk * (r->C ? r->C : 1) * r->esize;
    if (got != len || ZSTD_decompress(r->cb, cb, zb, len) != cb) { free(zb); return nullptr; }
    free(zb); r->csh = 0; r->cci = ci; memcpy(r->cshard, s, sizeof s);
    return r->cb;
}
/* float32 [n0][n1][n2](C) of the region at o (output cells); outside the array or absent: 0 */
static int zr_read(zr *r, const int64_t o[3], const int64_t n[3], float *out) {
    const int C = r->C ? r->C : 1, c = r->chunk;
    for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) for (int64_t x = 0; x < n[2]; x++) {
        float *dst = out + ((z * n[1] + y) * n[2] + x) * C;
        const int64_t gz = o[0] + z, gy = o[1] + y, gx = o[2] + x;
        const uint8_t *ch = gz >= 0 && gy >= 0 && gx >= 0 && gz < r->shape[0] && gy < r->shape[1] && gx < r->shape[2] ? zr_chunk(r, gz, gy, gx) : nullptr;
        if (!ch) { memset(dst, 0, C * 4); continue; }
        const size_t e = ((size_t)((gz % c) * c + gy % c) * c + gx % c) * C;
        for (int k = 0; k < C; k++) {
            if (r->dt == 0) dst[k] = (float)(((const int8_t *)ch)[e + k] * r->vscale);
            else if (r->dt == 1) { uint16_t h; memcpy(&h, ch + (e + k) * 2, 2); dst[k] = (float)(h2f(h) * r->vscale); }
            else { float v; memcpy(&v, ch + (e + k) * 4, 4); dst[k] = (float)(v * r->vscale); }
        }
    }
    return 0;
}
/* an output with its optional per-vector scale array */
typedef struct { zr a, s; int has_s; } emb;
static int emb_open(emb *e, const char *group, const char *name) {
    char d[1200]; snprintf(d, sizeof d, "%s/%s", group, name);
    if (zr_open(&e->a, d)) { fprintf(stderr, "dinovol: cannot open %s\n", d); return -1; }
    snprintf(d, sizeof d, "%s/%s_scale", group, name); e->has_s = zr_open(&e->s, d) == 0;
    return 0;
}
static void emb_close(emb *e) { zr_close(&e->a); if (e->has_s) zr_close(&e->s); }
static int emb_read(emb *e, const int64_t o[3], const int64_t n[3], float *out) {
    if (zr_read(&e->a, o, n, out)) return -1;
    if (e->has_s) {
        const size_t nv = (size_t)n[0] * n[1] * n[2]; float *s = malloc(nv * 4);
        zr_read(&e->s, o, n, s);
        for (size_t i = 0; i < nv; i++) for (int k = 0; k < e->a.C; k++) out[i * e->a.C + k] *= s[i];
        free(s);
    }
    return 0;
}
/* ufsm dinovol-read <group> <name> z,y,x,nz,ny,nx <out.f32>: a region of one output (output cells), float32 channel-last */
int cmd_dinovol_read(int argc, char **argv) {
    long long b[6];
    if (argc < 6 || sscanf(argv[4], "%lld,%lld,%lld,%lld,%lld,%lld", &b[0], &b[1], &b[2], &b[3], &b[4], &b[5]) != 6) {
        fprintf(stderr, "usage: ufsm dinovol-read <group> <tokens|tokens_unit|dense|ink> z,y,x,nz,ny,nx <out.f32>   (output-cell units, float32 channel-last)\n"); return 2; }
    emb e; if (emb_open(&e, argv[2], argv[3])) return 1;
    const int64_t o[3] = {b[0], b[1], b[2]}, n[3] = {b[3], b[4], b[5]};
    const size_t nv = (size_t)n[0] * n[1] * n[2] * (e.a.C ? e.a.C : 1); float *out = malloc(nv * 4);
    emb_read(&e, o, n, out);
    FILE *f = fopen(argv[5], "wb"); const int rc = !f || fwrite(out, 4, nv, f) != nv; if (f) fclose(f);
    free(out); emb_close(&e);
    return rc;
}
int cmd_dinovol_eval(int argc, char **argv) { (void)argc; (void)argv; fprintf(stderr, "dinovol-eval: not built yet\\n"); return 2; }
