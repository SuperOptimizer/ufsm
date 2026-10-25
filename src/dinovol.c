/* ufsm dinovol-embed: the scrollprize dinovol model (src/dinovol.h) over a CT box with embed_volume.py's lattice tiling, blending and
   volume-wide normalisation; ufsm dinovol-eval: the sheet-identity check of the dense embedding along published surfaces. */
#include "dinovol.h"
#include "xnet.h"
#include "store.h"
#include "zarr3.h"
#include "z3w.h"
#include "json.h"
#include "tiff.h"
#include "fit.h"
#include "embz.h"
#include <pthread.h>
#include <sys/stat.h>
#include <zstd.h>
#include <stdatomic.h>
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
        if (r == 0 && reps > 1) dv_prof_report(0);   /* the first (cold) forward is not counted */
    }
    fprintf(stderr, "dinovol: tile %d^3 forward %.1f ms\n", T, (t1 - t0) * 1e3);
    dv_prof_report(reps > 1 ? reps - 1 : 1);
    const int g = T / 8;
    if (dump_dev_half(out, "tokens.f32", dv_tokens(net), (size_t)g * g * g * dv_embed_dim(net))) return 1;
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
        dv_prof_report(1);
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

/* (the sharded zarr v3 writer zw_*: src/embz.c) */
static int mkdirs(const char *path) {
    char p[1200]; snprintf(p, sizeof p, "%s", path);
    for (char *s = p + 1; *s; s++) if (*s == '/') { *s = 0; if (mkdir(p, 0755) && errno != EEXIST) return -1; *s = '/'; }
    return mkdir(p, 0755) && errno != EEXIST ? -1 : 0;
}
/* background writer: puts (owning their buffer) and shard flushes run in submission order on one thread (each put compresses its chunks
   on zw_nthreads workers), so finishing a column overlaps the next column's network; pending bytes are bounded */
typedef struct aq_job { zw *w; int flush; int64_t oz, oy, ox; int nz, ny, nx; void *data; size_t bytes; struct aq_job *next; } aq_job;
typedef struct { pthread_t th; pthread_mutex_t mu; pthread_cond_t cv; aq_job *head, *tail; size_t pending, cap; int stop, failed, running; } aqueue;
static void *aq_main(void *a) {
    aqueue *q = a;
    for (;;) {
        pthread_mutex_lock(&q->mu);
        while (!q->head && !q->stop) pthread_cond_wait(&q->cv, &q->mu);
        aq_job *j = q->head; if (!j) { pthread_mutex_unlock(&q->mu); break; }
        q->head = j->next; if (!q->head) q->tail = nullptr; q->running = 1;
        pthread_mutex_unlock(&q->mu);
        const int rc = j->flush ? zw_flush(j->w) : zw_put(j->w, j->oz, j->oy, j->ox, j->nz, j->ny, j->nx, j->data);
        free(j->data);
        pthread_mutex_lock(&q->mu); q->pending -= j->bytes; q->failed |= rc != 0; q->running = 0; pthread_cond_broadcast(&q->cv); pthread_mutex_unlock(&q->mu);
        free(j);
    }
    return nullptr;
}
static void aq_start(aqueue *q, size_t cap) { memset(q, 0, sizeof *q); pthread_mutex_init(&q->mu, nullptr); pthread_cond_init(&q->cv, nullptr); q->cap = cap; pthread_create(&q->th, nullptr, aq_main, q); }
static void aq_push(aqueue *q, aq_job *j) {
    pthread_mutex_lock(&q->mu);
    while (q->pending && q->pending + j->bytes > q->cap) pthread_cond_wait(&q->cv, &q->mu);
    q->pending += j->bytes; j->next = nullptr;
    if (q->tail) q->tail->next = j; else q->head = j;
    q->tail = j; pthread_cond_broadcast(&q->cv); pthread_mutex_unlock(&q->mu);
}
/* a put of nz x ny x nx cells whose bytes are copied from the device buffer now */
static void aq_put_dev(aqueue *q, zw *w, int64_t oz, int64_t oy, int64_t ox, int nz, int ny, int nx, const void *dev, size_t bytes) {
    aq_job *j = calloc(1, sizeof *j); j->w = w; j->oz = oz; j->oy = oy; j->ox = ox; j->nz = nz; j->ny = ny; j->nx = nx; j->bytes = bytes;
    j->data = malloc(bytes); xd2h(j->data, dev, bytes);
    aq_push(q, j);
}
static void aq_flush(aqueue *q, zw *w) { aq_job *j = calloc(1, sizeof *j); j->w = w; j->flush = 1; aq_push(q, j); }
static int aq_finish(aqueue *q) {   /* drain and stop; nonzero if any write failed */
    pthread_mutex_lock(&q->mu); q->stop = 1; pthread_cond_broadcast(&q->cv); pthread_mutex_unlock(&q->mu);
    pthread_join(q->th, nullptr);
    return q->failed;
}

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
    xnet_set_budget((size_t)(atof(opt(argc, argv, "--gpu-mem", "9")) * 1e9));
    if (opt(argc, argv, "--raw-in", nullptr)) {
        dv_net *net = dv_load(ck); if (!net) return 1;
        const int rc = raw_mode(net, argc, argv); dv_free(net); return rc;
    }
    const char *root = argv[2], *key = argv[3], *out = argv[4];
    const int nthreads = atoi(opt(argc, argv, "--threads", "8")); zw_nthreads = nthreads < 1 ? 1 : nthreads > 64 ? 64 : nthreads;
    const int T = atoi(opt(argc, argv, "--tile", "192")), ds = atoi(opt(argc, argv, "--dense-stride", "1"));
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
    int K = atoi(opt(argc, argv, "--shard-cells", "0")), Wx = atoi(opt(argc, argv, "--band-cells", "0"));
    const double gbud = atof(opt(argc, argv, "--gpu-mem", "9")) * 1e9 - 5.2e9;   /* the forward (192^3 tile + head chunk + caches) takes ~5 GB */
    {   /* slab depth K (cells per shard edge) and x band width Wx (cells; bands recompute one tile column each) with the fewest tiles per
           cell, (K + 1) / K x (Wx + 1) / Wx, whose accumulators + fp16 strips fit the device budget */
        const int kmax = (512 + stride / 2) / stride > 0 ? (512 + stride / 2) / stride : 1;
        double best = 1e30; int bk = 1, bw = 1;
        for (int k = 1; k <= kmax; k++) for (int w = k; w < ncx + k; w += k) {
            if ((K > 0 && k != K) || (Wx > 0 && w != Wx)) continue;
            const int we = w < ncx ? w : ncx;
            double bytes = 0;
            for (int pass = 0; pass < 2; pass++) {
                if (pass ? !need_den : !need_tok) continue;
                const int s = pass ? ds : 8; const double CA = pass ? CD : E + 1, nz = (double)k * stride / s, NS = support / s, sy = stride / s, r = ramp / s;
                bytes += nz * NS * NS * CA * 4 + nz * sy * r * CA * 2 + (we + 2) * 2.0 * (nz * r * sy * CA * 2 + nz * r * r * CA * 2);
            }
            const double cost = (k + 1.0) / k * (we + 1.0) / we;
            if (bytes <= gbud && cost < best - 1e-9) { best = cost; bk = k; bw = w; }
        }
        K = bk; Wx = bw;
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
    const int nz_slab_max = K * stride, Dcol = nz_slab_max + T, nslot = Wx + 2;
    const int nyt = support / ps, syt = stride / ps, ryt = ramp / ps, nyd = support / ds, syd = stride / ds, ryd = ramp / ds;
    const int CAt = E + 1, CAd = CD;
    uint8_t *colh[2] = {malloc((size_t)Dcol * T * T), malloc((size_t)Dcol * T * T)};
    uint8_t *cold = xmalloc_dev((size_t)Dcol * T * T);
    float *acc_t = need_tok ? xmalloc_dev((size_t)(nz_slab_max / ps) * nyt * nyt * CAt * 4) : nullptr;
    float *acc_d = need_den ? xmalloc_dev((size_t)(nz_slab_max / ds) * nyd * nyd * CAd * 4) : nullptr;
    uint8_t *mask_t = xmalloc_dev((size_t)(nz_slab_max / ps) * nyt * nyt), *mask_d = need_den ? xmalloc_dev((size_t)(nz_slab_max / ds) * nyd * nyd) : nullptr;
    const size_t npt_max = (size_t)(support / ds) * nyd * nyd;
    float *coords = need_den ? xmalloc_dev(npt_max * 12) : nullptr, *wgt = need_den ? xmalloc_dev(npt_max * 4) : nullptr; int *rows = need_den ? xmalloc_dev(npt_max * 4) : nullptr, *cnt = xmalloc_dev(4);
    /* fp16 strips per band column slot xi = kx - (band start - 1) in [0, Wx + 1]: below (the column's y ramp into the next row, r x sy) and corner
       (its yx ramp into the next row's next column, r x r), for this row (top) and the next; the left strip (sy x r) into the next column */
    const size_t bl_t = (size_t)(nz_slab_max / ps) * ryt * syt * CAt, bl_d = (size_t)(nz_slab_max / ds) * ryd * syd * CAd;
    const size_t cr_t = (size_t)(nz_slab_max / ps) * ryt * ryt * CAt, cr_d = (size_t)(nz_slab_max / ds) * ryd * ryd * CAd;
    void **stp[2][2][2];   /* [row: 0 top, 1 next][kind: 0 below, 1 corner][pass: 0 tokens, 1 dense] -> per slot */
    uint8_t *has[2][2];    /* [row][kind] per slot */
    for (int a = 0; a < 2; a++) for (int k = 0; k < 2; k++) {
        has[a][k] = calloc(nslot, 1);
        for (int p = 0; p < 2; p++) {
            stp[a][k][p] = calloc(nslot, sizeof(void *));
            if (p ? !need_den : !need_tok) continue;
            for (int i = 0; i < nslot; i++) stp[a][k][p][i] = xmalloc_dev((p ? (k ? cr_d : bl_d) : (k ? cr_t : bl_t)) * 2);
        }
    }
    void *left[2] = {need_tok ? xmalloc_dev((size_t)(nz_slab_max / ps) * syt * ryt * CAt * 2) : nullptr, need_den ? xmalloc_dev((size_t)(nz_slab_max / ds) * syd * ryd * CAd * 2) : nullptr};
    int has_left = 0;
    const size_t ob_max = (size_t)(nz_slab_max / ds < nz_slab_max / ps ? nz_slab_max / ps : nz_slab_max / ds) * (syd > syt ? syd : syt) * (syd > syt ? syd : syt) * (E > nd ? E : nd) * 2;
    void *obuf_d = xmalloc_dev(ob_max), *osc_d = xmalloc_dev(ob_max / 2 + 64);
    if (!colh[0] || !colh[1] || !cold || !mask_t || !cnt || !obuf_d || !osc_d || (need_tok && (!acc_t || !left[0])) || (need_den && (!acc_d || !mask_d || !coords || !left[1]))) {
        fprintf(stderr, "dinovol-embed: device memory allocation failed (lower --gpu-mem / --shard-cells / --band-cells)\n"); return 1; }
    fprintf(stderr, "dinovol-embed: box %lld,%lld,%lld +%lld,%lld,%lld: cells z %lld..%lld y %lld..%lld x %lld..%lld (stride %d, tile %d, margin %d, ramp %d), %d cells per shard edge "
                    "(z slab), x bands of %d cells, dense stride %d, device %.2f GB in use\n", bv[0], bv[1], bv[2], bv[3], bv[4], bv[5], (long long)c0[0], (long long)c1[0], (long long)c0[1],
            (long long)c1[1], (long long)c0[2], (long long)c1[2], stride, T, margin, ramp, K, Wx, ds, xnet_mem_used() / 1e9);
    aqueue AQ; aq_start(&AQ, (size_t)(atof(opt(argc, argv, "--write-buffer-gb", "1.5")) * 1e9));
    const double t_start = dv_now(); double t_net = 0, t_head = 0, t_read = 0, t_fin = 0; long ntiles = 0, nskip = 0, ncells = 0, npoints = 0;
    int rc = 0;
    extern void dv_zero_dev(void *p, size_t bytes);
    /* x bands of Wx cells (aligned to the shard grid), then slabs of K cells along z */
    int64_t bx1 = c0[2];
    for (int64_t xb = 0; bx1 < c1[2] && !rc; xb++) {
    const int64_t bx0 = bx1;   /* bands end on shard boundaries (multiples of K cells), one band when Wx covers the box */
    bx1 = Wx >= ncx ? c1[2] : bx0 / K * K + Wx; if (bx1 > c1[2]) bx1 = c1[2];
    const int nbx = (int)(bx1 - bx0);
    for (int64_t zs = c0[0] / K; zs * K < c1[0] && !rc; zs++) {
        const int64_t cza = zs * K > c0[0] ? zs * K : c0[0], czb = (zs + 1) * K < c1[0] ? (zs + 1) * K : c1[0];
        const int64_t Z0 = cza * stride, Z1 = czb * stride; const int nzv = (int)(Z1 - Z0);
        const int64_t bandz0 = (cza - 1) * stride - margin; const int Dband = (int)(czb - cza) * stride + T;
        const int nzt = nzv / ps, nzd = nzv / ds;
        for (int a = 0; a < 2; a++) for (int k = 0; k < 2; k++) memset(has[a][k], 0, nslot);
        /* columns in raster order; prefetch the next column's CT */
        const int ky0 = (int)c0[1] - 1, kx0 = (int)bx0 - 1, ncol = (ncy + 1) * (nbx + 1);
        colread R = {.ct = ct, .nthreads = nthreads}; int cur = 0;
        colread_start(&R, ky0, kx0, bandz0, Dband, T, stride, margin, colh[0]);
        for (int ci = 0; ci < ncol && !rc; ci++) {
            const int ky = ky0 + ci / (nbx + 1), kx = kx0 + ci % (nbx + 1), xi = kx - kx0;   /* xi: strip slot */
            if (kx == kx0) {   /* new row: next -> top */
                for (int k = 0; k < 2; k++) {
                    for (int p = 0; p < 2; p++) { void **t = stp[0][k][p]; stp[0][k][p] = stp[1][k][p]; stp[1][k][p] = t; }
                    uint8_t *h = has[0][k]; has[0][k] = has[1][k]; has[1][k] = h; memset(has[1][k], 0, nslot);
                }
                has_left = 0;
            }
            double tr = dv_now(); pthread_join(R.th, nullptr); R.busy = 0; t_read += dv_now() - tr;
            if (R.rc) { fprintf(stderr, "dinovol-embed: CT read failed: %s\n", z3_error()); rc = 1; break; }
            const uint8_t *col = colh[cur];
            if (ci + 1 < ncol) { cur ^= 1; colread_start(&R, ky0 + (ci + 1) / (nbx + 1), kx0 + (ci + 1) % (nbx + 1), bandz0, Dband, T, stride, margin, colh[cur]); }
            const int in_y = ky >= c0[1] && ky < c1[1], in_x = kx >= bx0 && kx < bx1;
            const int ylo = ky < c0[1] ? stride : 0, yhi = ky == c1[1] - 1 ? stride : support, xlo = kx < bx0 ? stride : 0, xhi = kx == bx1 - 1 ? stride : support;
            /* any nonzero voxel in the needed region? */
            int anyv = 0;
            for (int z = (int)(Z0 - bandz0); z < (int)(Z1 - bandz0) && !anyv; z++)
                for (int y = margin + ylo; y < margin + yhi && !anyv; y++) { const uint8_t *r = col + ((size_t)z * T + y) * T + margin; for (int x = xlo; x < xhi; x++) if (r[x]) { anyv = 1; break; } }
            if (!anyv) { has_left = 0; continue; }
            xh2d(cold, col, (size_t)Dband * T * T);
            dv_blockmask(cold, Dband, T, T, (int)(Z0 - bandz0), margin, margin, ps, nzt, nyt, nyt, ylo / ps, yhi / ps, xlo / ps, xhi / ps, mask_t);
            if (need_den) dv_blockmask(cold, Dband, T, T, (int)(Z0 - bandz0), margin, margin, ds, nzd, nyd, nyd, ylo / ds, yhi / ds, xlo / ds, xhi / ds, mask_d);
            uint8_t *mt = malloc((size_t)nzt * nyt * nyt); xd2h(mt, mask_t, (size_t)nzt * nyt * nyt);
            if (need_tok) dv_zero_dev(acc_t, (size_t)nzt * nyt * nyt * CAt * 4);
            if (need_den) dv_zero_dev(acc_d, (size_t)nzd * nyd * nyd * CAd * 4);
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
            const double tf = dv_now();
            /* finish: the strips this column passes on (below / corner to the next row, left to the next column) are saved, then the cell
               (ky, kx) = A[:, :sy, :sy] + the strips it received, added in place (the saved regions lie outside it) */
            for (int pass = 0; pass < 2; pass++) {
                const int isd = pass == 1; if (isd ? !need_den : !need_tok) continue;
                const int nz = isd ? nzd : nzt, NS = isd ? nyd : nyt, sy = isd ? syd : syt, r = isd ? ryd : ryt, CA = isd ? CAd : CAt;
                float *A = isd ? acc_d : acc_t;
                dv_region2(stp[1][0][pass][xi], 1, r, sy, 0, 0, A, 0, NS, NS, sy, 0, nz, r, sy, CA, 0);        /* below -> next row, this slot */
                dv_region2(stp[1][1][pass][xi + 1], 1, r, r, 0, 0, A, 0, NS, NS, sy, sy, nz, r, r, CA, 0);   /* corner -> next row, next slot */
                if (in_y && in_x) {
                    if (has[0][0][xi]) dv_region2(A, 0, NS, NS, 0, 0, stp[0][0][pass][xi], 1, r, sy, 0, 0, nz, r, sy, CA, 1);
                    if (has[0][1][xi]) dv_region2(A, 0, NS, NS, 0, 0, stp[0][1][pass][xi], 1, r, r, 0, 0, nz, r, r, CA, 1);
                    if (has_left) dv_region2(A, 0, NS, NS, 0, 0, left[pass], 1, sy, r, 0, 0, nz, sy, r, CA, 1);
                }
                dv_region2(left[pass], 1, sy, r, 0, 0, A, 0, NS, NS, 0, sy, nz, sy, r, CA, 0);              /* left -> next column (after the adds: x >= sy untouched) */
            }
            has[1][0][xi] = 1; has[1][1][xi + 1] = 1; has_left = 1;
            if (!(in_y && in_x)) { t_fin += dv_now() - tf; continue; }
            ncells += (long)(czb - cza);   /* lattice cells finished (the slab depth) */
            for (int o = 0; o < O_N && !rc; o++) {
                if (!want[o]) continue;
                const int isd = o >= O_DENSE, s = isd ? ds : ps, nz = isd ? nzd : nzt, NS = isd ? nyd : nyt, sy = isd ? syd : syt, CA = isd ? CAd : CAt, C = o <= O_TOKU ? E : o == O_DENSE ? nd : ni;
                const int c0c = o == O_INK ? nd : 0, wch = isd ? -1 : E, unit = o != O_TOK, unit8 = unit && !dtype;
                dv_finish(isd ? acc_d : acc_t, NS, NS, CA, 0, 0, nz, sy, sy, c0c, C, wch, unit, isd ? mask_d : mask_t, dtype, obuf_d, osc_d, unit8);
                const size_t ob = (size_t)nz * sy * sy * C * (dtype ? 2 : 1);
                const int64_t oz = Z0 / s, oy = (int64_t)ky * stride / s, ox = (int64_t)kx * stride / s;
                aq_put_dev(&AQ, &W[o], oz, oy, ox, nz, sy, sy, obuf_d, ob);
                if (has_sc[o]) aq_put_dev(&AQ, &Wsc[o], oz, oy, ox, nz, sy, sy, osc_d, (size_t)nz * sy * sy * 2);
            }
            /* shard row done (last row of the shard in y, or of the box) at the band row's last column */
            if (kx == bx1 - 1 && (ky == c1[1] - 1 || (ky + 1) % K == 0))
                for (int o = 0; o < O_N; o++) if (want[o]) { aq_flush(&AQ, &W[o]); if (has_sc[o]) aq_flush(&AQ, &Wsc[o]); }
            if (AQ.failed) rc = 1;
            t_fin += dv_now() - tf;
            if (ci % 4 == 3 || ci == ncol - 1) {
                const double el = dv_now() - t_start;
                fprintf(stderr, "\rdinovol-embed: band %lld slab %lld cells %ld tiles %ld (+%ld skipped) %.1f s: net %.1f head %.1f finish %.1f read wait %.1f, %.2f Mvox/s   ",
                        (long long)xb, (long long)zs, ncells, ntiles, nskip, el, t_net, t_head, t_fin, t_read, (double)ncells * stride * stride * stride / el / 1e6);
            }
        }
        if (R.busy) pthread_join(R.th, nullptr);
        for (int o = 0; o < O_N; o++) if (want[o]) { aq_flush(&AQ, &W[o]); if (has_sc[o]) aq_flush(&AQ, &Wsc[o]); }
    }
    }
    rc |= aq_finish(&AQ);
    const double el = dv_now() - t_start;
    fprintf(stderr, "\ndinovol-embed: %ld cells (%.1f Mvox) in %.1f s = %.2f Mvox/s; %ld tiles (%.0f ms each: net %.0f, head %.0f, finish %.0f), %ld dense points, %ld tiles skipped; peak device %.2f GB\n",
            ncells, ncells * (double)stride * stride * stride / 1e6, el, ncells * (double)stride * stride * stride / el / 1e6, ntiles, ntiles ? el / ntiles * 1e3 : 0,
            ntiles ? t_net / ntiles * 1e3 : 0, ntiles ? t_head / ntiles * 1e3 : 0, ntiles ? t_fin / ntiles * 1e3 : 0, npoints, nskip, (xnet_mem_used() + xnet_mem_cached()) / 1e9);
    for (int o = 0; o < O_N; o++) if (want[o]) { zw_free(&W[o]); if (has_sc[o]) zw_free(&Wsc[o]); }
    dv_free(net); z3_close(ct); store_close(st);
    return rc;
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

/* ---- dinovol-eval: sheet identity along published surfaces ---- */
typedef struct { int64_t key; uint8_t *data; uint64_t used; } cslot;
typedef struct { zr *r; cslot *s; int n; uint64_t tick; size_t cb; } ccache;   /* LRU of decoded chunks */
static const uint8_t *cc_get(ccache *c, int64_t z, int64_t y, int64_t x) {
    zr *r = c->r; const int ch = r->chunk;
    if (z < 0 || y < 0 || x < 0 || z >= r->shape[0] || y >= r->shape[1] || x >= r->shape[2]) return nullptr;
    const int64_t key = ((z / ch) * ((r->shape[1] + ch - 1) / ch) + y / ch) * ((r->shape[2] + ch - 1) / ch) + x / ch;
    int lru = 0;
    for (int i = 0; i < c->n; i++) {
        if (c->s[i].key == key) { c->s[i].used = ++c->tick; return c->s[i].data; }
        if (c->s[i].used < c->s[lru].used) lru = i;
    }
    const uint8_t *d = zr_chunk(r, z, y, x);
    cslot *s = &c->s[lru]; s->key = key; s->used = ++c->tick;
    if (!d) { free(s->data); s->data = nullptr; return nullptr; }
    if (!s->data) s->data = malloc(c->cb);
    memcpy(s->data, d, c->cb);
    return s->data;
}
/* trilinear vector at a continuous voxel position p (z, y, x) of the volume; the output cell i covers voxels [i s, (i + 1) s) */
static int cc_vec(ccache *c, int step, const double p[3], float *v) {
    zr *r = c->r; const int C = r->C, ch = r->chunk;
    double u[3]; int64_t i0[3]; double f[3];
    for (int d = 0; d < 3; d++) { u[d] = (p[d] - (step - 1) * 0.5) / step; i0[d] = (int64_t)floor(u[d]); f[d] = u[d] - (double)i0[d]; }
    for (int k = 0; k < C; k++) v[k] = 0.f;
    int ok = 0;
    for (int a = 0; a < 8; a++) {
        const int64_t z = i0[0] + (a >> 2), y = i0[1] + ((a >> 1) & 1), x = i0[2] + (a & 1);
        const double w = ((a >> 2) ? f[0] : 1 - f[0]) * (((a >> 1) & 1) ? f[1] : 1 - f[1]) * ((a & 1) ? f[2] : 1 - f[2]);
        if (w <= 0) continue;
        const uint8_t *d = cc_get(c, z, y, x); if (!d) continue;
        const size_t e = ((size_t)((z % ch) * ch + y % ch) * ch + x % ch) * C;
        float nn = 0; for (int k = 0; k < C; k++) { const float t = r->dt == 0 ? ((const int8_t *)d)[e + k] : 0; nn += t * t; }
        if (r->dt == 0) { if (nn == 0) continue; for (int k = 0; k < C; k++) v[k] += (float)(w * ((const int8_t *)d)[e + k]); }
        else { for (int k = 0; k < C; k++) { uint16_t h; memcpy(&h, d + (e + k) * 2, 2); v[k] += (float)(w * h2f(h)); } }
        ok = 1;
    }
    return ok;
}
static float vcos(const float *a, const float *b, int C) {
    double ab = 0, aa = 0, bb = 0; for (int k = 0; k < C; k++) { ab += (double)a[k] * b[k]; aa += (double)a[k] * a[k]; bb += (double)b[k] * b[k]; }
    return aa > 0 && bb > 0 ? (float)(ab / sqrt(aa * bb)) : -2.f;
}
static float *read_tif_f32(const char *path, int *w, int *h) {
    tiff *t = tiff_open_file(path); if (!t) return nullptr;
    tiff_page pi; float *o = nullptr;
    if (!tiff_page_info(t, 0, &pi) && pi.fmt == 3 && pi.bits == 32 && pi.spp == 1) { o = malloc((size_t)pi.w * pi.h * 4); if (tiff_read_page(t, 0, o)) { free(o); o = nullptr; } else { *w = pi.w; *h = pi.h; } }
    tiff_close(t); return o;
}
typedef struct { int w, h; float *x, *y, *z; double sc; } surf_t;   /* positions scaled to the embedding volume's voxels */
static int surf_pos(const surf_t *S, double u, double v, double p[3]) {   /* bilinear at grid (u = column, v = row); 0 if any corner invalid */
    const int c = (int)floor(u), r = (int)floor(v); if (c < 0 || r < 0 || c + 1 >= S->w || r + 1 >= S->h) return 0;
    const double fu = u - c, fv = v - r; double q[3] = {0, 0, 0};
    for (int a = 0; a < 4; a++) {
        const size_t i = (size_t)(r + (a >> 1)) * S->w + c + (a & 1);
        if (S->z[i] < 0 || S->x[i] < 0 || S->y[i] < 0) return 0;
        const double wgt = ((a >> 1) ? fv : 1 - fv) * ((a & 1) ? fu : 1 - fu);
        q[0] += wgt * S->z[i]; q[1] += wgt * S->y[i]; q[2] += wgt * S->x[i];
    }
    for (int d = 0; d < 3; d++) p[d] = q[d] * S->sc;
    return 1;
}
static int surf_normal(const surf_t *S, double u, double v, double n[3]) {
    double a[3], b[3], c[3], d[3];
    if (!surf_pos(S, u - 0.5, v, a) || !surf_pos(S, u + 0.5, v, b) || !surf_pos(S, u, v - 0.5, c) || !surf_pos(S, u, v + 0.5, d)) return 0;
    const double du[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, dv[3] = {d[0] - c[0], d[1] - c[1], d[2] - c[2]};
    n[0] = du[1] * dv[2] - du[2] * dv[1]; n[1] = du[2] * dv[0] - du[0] * dv[2]; n[2] = du[0] * dv[1] - du[1] * dv[0];
    const double l = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]); if (l < 1e-9) return 0;
    for (int k = 0; k < 3; k++) n[k] /= l;
    return 1;
}
static uint64_t ev_rng(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }
static double ev_unif(uint64_t *s) { return (ev_rng(s) >> 11) * (1.0 / 9007199254740992.0); }

/* the winding code nearest to p (embedding voxels) within g_crad code voxels (L1 nearest), 0 if none (or a conflict) */
static int g_crad = 1;
static int code_near(const uint8_t *cb, const int64_t org[3], const int64_t n[3], double sc, const double p[3]) {
    int64_t c[3]; for (int d = 0; d < 3; d++) c[d] = (int64_t)floor((p[d] + 0.5) * sc) - org[d];
    int best = 0, bd = 99;
    for (int a = -g_crad; a <= g_crad; a++) for (int b = -g_crad; b <= g_crad; b++) for (int e = -g_crad; e <= g_crad; e++) {
        const int64_t z = c[0] + a, y = c[1] + b, x = c[2] + e;
        if (z < 0 || y < 0 || x < 0 || z >= n[0] || y >= n[1] || x >= n[2]) continue;
        const int v = cb[((size_t)z * n[1] + y) * n[2] + x], dd = abs(a) + abs(b) + abs(e);
        if (v && v != 255 && dd < bd) { bd = dd; best = v; }
    }
    return best;
}
int cmd_dinovol_eval(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: ufsm dinovol-eval <emb-group> <surface.tifxyz>... [--output dense|ink|tokens_unit] [--surf-scale 0.25] [--n-ref 300] [--dirs 8]\n"
                        "         [--dists 0,10,20,40,80,160] [--range 24] [--tols 2,6] [--edge 4] [--seed 1] [--cache-chunks 512]\n"
                        "  Sheet identity of an embedding (the dinovol coworker check): from a reference vertex on a published surface, walk along the\n"
                        "  surface (uv lines, 3D path length in embedding voxels) to each distance, sample the embedding along the surface normal at\n"
                        "  -range..+range voxels, and count the reference's most similar sample (cosine) within tol voxels of the surface.\n");
        return 2;
    }
    const char *group = argv[2], *oname = opt(argc, argv, "--output", "dense");
    if (opt(argc, argv, "--suggest-box", nullptr)) {   /* a cube of N voxels centred on the surface vertex nearest its grid centre (embedding voxels) */
        const int N = atoi(opt(argc, argv, "--suggest-box", "384")); const double s0 = atof(opt(argc, argv, "--surf-scale", "0.25"));
        char q[1300]; int w, h, w2, h2, w3, h3; snprintf(q, sizeof q, "%s/x.tif", argv[2]); float *x = read_tif_f32(q, &w, &h);
        snprintf(q, sizeof q, "%s/y.tif", argv[2]); float *y = read_tif_f32(q, &w2, &h2); snprintf(q, sizeof q, "%s/z.tif", argv[2]); float *z = read_tif_f32(q, &w3, &h3);
        if (!x || !y || !z) { fprintf(stderr, "dinovol-eval: cannot read %s\n", argv[2]); return 1; }
        const double fu = atof(opt(argc, argv, "--at-u", "0.5")), fv = atof(opt(argc, argv, "--at-v", "0.5"));
        long best = -1; double bd = 1e300;
        for (long i = 0; i < (long)w * h; i++) if (z[i] >= 0) { const double du = (double)(i % w) - fu * w, dv = (double)(i / w) - fv * h, d = du * du + dv * dv; if (d < bd) { bd = d; best = i; } }
        if (best < 0) return 1;
        printf("%lld,%lld,%lld,%d,%d,%d\n", (long long)(z[best] * s0) - N / 2, (long long)(y[best] * s0) - N / 2, (long long)(x[best] * s0) - N / 2, N, N, N);
        free(x); free(y); free(z); return 0;
    }
    const double sc = atof(opt(argc, argv, "--surf-scale", "0.25")); const int nref = atoi(opt(argc, argv, "--n-ref", "300")), ndir = atoi(opt(argc, argv, "--dirs", "8"));
    const int R = atoi(opt(argc, argv, "--range", "24")), edge = atoi(opt(argc, argv, "--edge", "4"));
    double dists[32]; int nd = 0; { char *s = strdup(opt(argc, argv, "--dists", "0,10,20,40,80,160")); for (char *t = strtok(s, ","); t && nd < 32; t = strtok(nullptr, ",")) dists[nd++] = atof(t); free(s); }
    int tols[8], nt = 0; { char *s = strdup(opt(argc, argv, "--tols", "2,6")); for (char *t = strtok(s, ","); t && nt < 8; t = strtok(nullptr, ",")) tols[nt++] = atoi(t); free(s); }
    uint64_t seed = strtoull(opt(argc, argv, "--seed", "1"), nullptr, 10) * 0x9E3779B97F4A7C15ull + 1;
    /* the embedding: step and the embedded box (voxels) from the group attributes */
    char p[1200]; snprintf(p, sizeof p, "%s/zarr.json", group); size_t n = 0; char *t = read_file(p, &n);
    if (!t) { fprintf(stderr, "dinovol-eval: cannot read %s\n", p); return 1; }
    json *j = json_parse(t, n); free(t); if (!j) { fprintf(stderr, "dinovol-eval: bad %s\n", p); return 1; }
    int64_t box[6]; const json *bj = json_path(j, "attributes.box_zyx");
    for (int d = 0; d < 6; d++) box[d] = (int64_t)json_num(json_at(bj, d), 0);
    const int stride = (int)json_num(json_path(j, "attributes.lattice.stride"), 0), ds = (int)json_num(json_path(j, "attributes.dense_stride"), 1);
    json_free(j);
    const int step = !strncmp(oname, "tokens", 6) ? 8 : ds;
    int64_t lo[3], hi[3];   /* computed cells: the lattice cells overlapping the box */
    for (int d = 0; d < 3; d++) { lo[d] = box[d] / stride * stride; hi[d] = ((box[d] + box[3 + d] - 1) / stride + 1) * stride; }
    emb e; if (emb_open(&e, group, oname)) return 1;
    if (e.has_s) { fprintf(stderr, "dinovol-eval: %s has a per-vector scale array; use a unit output (dense, ink, tokens_unit)\n", oname); return 2; }
    const int C = e.a.C; ccache cc = {.r = &e.a, .n = atoi(opt(argc, argv, "--cache-chunks", "512")), .cb = (size_t)e.a.chunk * e.a.chunk * e.a.chunk * C * e.a.esize};
    cc.s = calloc(cc.n, sizeof *cc.s); for (int i = 0; i < cc.n; i++) cc.s[i].key = -1;
    /* results [dist][tol] and |t*| histogram */
    long tot[32] = {0}, hit[32][8] = {{0}}; double sum_abs[32] = {0}; long nskip_ref = 0;
    /* --codes ROOT KEY: a winding code raster (1 + round(18 q) mod 252, 0 empty, 255 conflict; --codes-scale code voxels per embedding voxel, default 2)
       classifies every miss: same wrap but off the surface, the adjacent wrap (+-1 turn), farther wraps, or no label at the best match */
    g_crad = atoi(opt(argc, argv, "--codes-radius", "2"));   /* code voxels searched around a point (the labels are thin sheets) */
    const char *cr = opt(argc, argv, "--codes", nullptr); const double csc = atof(opt(argc, argv, "--codes-scale", "2"));
    uint8_t *cbuf = nullptr; int64_t corg[3] = {0, 0, 0}, cn[3] = {0, 0, 0};
    long cls[32][6] = {{0}};   /* 0 hit (<= first tol), 1 same wrap off, 2 adjacent, 3 farther, 4 unlabelled at the match, 5 no code at the surface point */
    if (cr) {
        const char *ck = nullptr; for (int k = 3; k + 2 < argc; k++) if (!strcmp(argv[k], "--codes")) ck = argv[k + 2];
        store *cs = store_open(cr); z3 *cz = cs && ck ? z3_open(cs, ck, nullptr) : nullptr;
        if (!cz) { fprintf(stderr, "dinovol-eval: cannot open the codes %s %s\n", cr, ck ? ck : "(key missing)"); return 1; }
        for (int d = 0; d < 3; d++) { corg[d] = (int64_t)floor(lo[d] * csc); cn[d] = (int64_t)ceil((hi[d] - lo[d]) * csc); }
        cbuf = malloc((size_t)cn[0] * cn[1] * cn[2]);
        if (!cbuf || z3_read(cz, corg, cn, cbuf, 8)) { fprintf(stderr, "dinovol-eval: codes read failed\n"); return 1; }
        z3_close(cz); store_close(cs);
    }
    float *e0 = malloc(C * 4), *ev = malloc(C * 4);
    /* --fit MODEL: the same walk scored by a winding fit instead of the embedding (the fit's same sheet: the normal sample whose W is
       nearest the reference's W; MODEL a fit / fit-set directory on the embedding's voxel grid): the fit's sheet following, comparable */
    fit_set *fsm = nullptr; double fo[3] = {0, 0, 0}, W0 = 0;
    if (opt(argc, argv, "--fit", nullptr)) {
        fsm = fit_set_load(opt(argc, argv, "--fit", nullptr));
        if (!fsm) { fprintf(stderr, "dinovol-eval: cannot load the fit %s\n", opt(argc, argv, "--fit", nullptr)); return 1; }
        for (int d = 0; d < 3; d++) fo[d] = (double)fsm->global->origin[d];
    }
#define FIT_AT(p_) fit_set_W(fsm, (p_)[0] - fo[0], (p_)[1] - fo[1], (p_)[2] - fo[2])
    int nsurf = 0;
    for (int ai = 3; ai < argc; ai++) {
        if (argv[ai][0] == '-' && argv[ai][1] == '-') { ai += strcmp(argv[ai], "--codes") ? 1 : 2; continue; }
        surf_t S = {.sc = sc}; int w2, h2, w3, h3;
        char q[1300];
        snprintf(q, sizeof q, "%s/x.tif", argv[ai]); S.x = read_tif_f32(q, &S.w, &S.h);
        snprintf(q, sizeof q, "%s/y.tif", argv[ai]); S.y = read_tif_f32(q, &w2, &h2);
        snprintf(q, sizeof q, "%s/z.tif", argv[ai]); S.z = read_tif_f32(q, &w3, &h3);
        if (!S.x || !S.y || !S.z || w2 != S.w || w3 != S.w || h2 != S.h || h3 != S.h) { fprintf(stderr, "dinovol-eval: cannot read the tifxyz %s\n", argv[ai]); return 1; }
        nsurf++;
        /* candidate reference vertices: inside the computed cells with room for the normal range */
        long *cand = malloc((size_t)S.w * S.h * sizeof *cand); long nc = 0;
        const double room = R + edge;
        for (long i = 0; i < (long)S.w * S.h; i++) {
            if (S.z[i] < 0) continue;
            const double pv[3] = {S.z[i] * sc, S.y[i] * sc, S.x[i] * sc}; int in = 1;
            for (int d = 0; d < 3; d++) in &= pv[d] >= lo[d] + room && pv[d] < hi[d] - room;
            if (in) cand[nc++] = i;
        }
        fprintf(stderr, "dinovol-eval: %s: %d x %d grid, %ld vertices inside the embedded cells\n", argv[ai], S.w, S.h, nc);
        for (int r = 0; r < nref && nc; r++) {
            const long i = cand[ev_rng(&seed) % (unsigned long)nc];
            const double u0 = (double)(i % S.w), v0 = (double)(i / S.w);
            double p0[3]; if (!surf_pos(&S, u0, v0, p0) || !cc_vec(&cc, step, p0, e0)) { nskip_ref++; continue; }
            if (fsm) W0 = FIT_AT(p0);
            for (int dj = 0; dj < ndir; dj++) {
                const double ang = 2 * M_PI * ev_unif(&seed), cu = cos(ang), sv = sin(ang);
                double u = u0, v = v0, len = 0, prev[3] = {p0[0], p0[1], p0[2]}; int di = 0;
                while (di < nd) {
                    double q3[3], nn[3]; int at = 0;
                    if (dists[di] <= len) { at = 1; }
                    else {
                        const double un = u + 0.25 * cu, vn = v + 0.25 * sv; double pn[3];
                        if (!surf_pos(&S, un, vn, pn)) break;
                        len += sqrt((pn[0] - prev[0]) * (pn[0] - prev[0]) + (pn[1] - prev[1]) * (pn[1] - prev[1]) + (pn[2] - prev[2]) * (pn[2] - prev[2]));
                        u = un; v = vn; memcpy(prev, pn, sizeof pn);
                        if (len > dists[nd - 1] + 1) break;
                        continue;
                    }
                    if (at) {
                        if (!surf_pos(&S, u, v, q3) || !surf_normal(&S, u, v, nn)) break;
                        int inside = 1; for (int d = 0; d < 3; d++) inside &= q3[d] >= lo[d] + room && q3[d] < hi[d] - room;
                        if (inside) {
                            float best = -3.f; int bt = 0;
                            for (int tt = -R; tt <= R; tt++) {
                                const double s3[3] = {q3[0] + tt * nn[0], q3[1] + tt * nn[1], q3[2] + tt * nn[2]};
                                if (!cc_vec(&cc, step, s3, ev)) continue;
                                const float cs = fsm ? (float)-fabs(FIT_AT(s3) - W0) : vcos(e0, ev, C);
                                if (cs > best || (cs == best && abs(tt) < abs(bt))) { best = cs; bt = tt; }
                            }
                            if (best > -2.f) {
                                tot[di]++; sum_abs[di] += abs(bt);
                                if (cbuf) {
                                    const double sb[3] = {q3[0] + bt * nn[0], q3[1] + bt * nn[1], q3[2] + bt * nn[2]};
                                    const int c0v = code_near(cbuf, corg, cn, csc, q3), c1v = code_near(cbuf, corg, cn, csc, sb);
                                    int k;
                                    if (abs(bt) <= tols[0]) k = 0;
                                    else if (!c0v) k = 5;
                                    else if (!c1v) k = 4;
                                    else { int dq = ((c1v - c0v) % 252 + 252) % 252; if (dq >= 126) dq -= 252; const int turns = (int)lround(dq / 18.0); k = turns == 0 ? 1 : abs(turns) == 1 ? 2 : 3; }
                                    cls[di][k]++;
                                }
                                for (int k = 0; k < nt; k++) hit[di][k] += abs(bt) <= tols[k];
                            }
                        }
                        di++;
                    }
                }
            }
        }
        free(cand); free(S.x); free(S.y); free(S.z);
    }
    if (fsm) printf("(scored by the fit %s: per sample the W nearest the reference W, not the embedding)\n", opt(argc, argv, "--fit", ""));
    printf("dinovol-eval %s (%s, step %d), %d surface(s), %d reference vertices x %d directions, normal range +-%d voxels\n", group, oname, step, nsurf, nref, ndir, R);
    printf("| distance (vx) | n |");
    for (int k = 0; k < nt; k++) printf(" within %d vx |", tols[k]);
    printf(" mean abs offset |\n|---|---|");
    for (int k = 0; k < nt; k++) printf("---|");
    printf("---|\n");
    for (int di = 0; di < nd; di++) {
        printf("| %g | %ld |", dists[di], tot[di]);
        for (int k = 0; k < nt; k++) printf(" %.1f%% |", tot[di] ? 100.0 * hit[di][k] / tot[di] : 0.0);
        printf(" %.2f |\n", tot[di] ? sum_abs[di] / tot[di] : 0.0);
    }
    if (nskip_ref) printf("(%ld reference vertices without an embedding skipped)\n", nskip_ref);
    if (cbuf) {
        printf("\nmisses of the within-%d-vx check by the winding code at the best match (share of all samples):\n", tols[0]);
        printf("| distance (vx) | hit | same wrap, off | adjacent wrap (+-1) | farther wraps | unlabelled at match | no code at surface |\n|---|---|---|---|---|---|---|\n");
        for (int di = 0; di < nd; di++) {
            long s = 0; for (int k = 0; k < 6; k++) s += cls[di][k];
            printf("| %g |", dists[di]); for (int k = 0; k < 6; k++) printf(" %.1f%% |", s ? 100.0 * cls[di][k] / s : 0.0); printf("\n");
        }
        free(cbuf);
    }
    for (int i = 0; i < cc.n; i++) free(cc.s[i].data);
    free(cc.s); free(e0); free(ev); emb_close(&e); fit_set_free(fsm);
#undef FIT_AT
    return 0;
}

/* a box of unit int8 embeddings [nz][ny][nx][C] (box-local) as a dinovol-format group (array "dense", step 1, value_scale 1/127) readable
   by dinovol-eval / dinovol-read; the array spans the volume's grid, shards of 128 cells hold the box */
int dv_write_dense_box(const char *group, const int64_t vol_shape[3], const int64_t box[6], int C, const int8_t *data) {
    if (mkdirs(group)) { fprintf(stderr, "dinovol: cannot create %s\n", group); return -1; }
    char p[1200]; snprintf(p, sizeof p, "%s/zarr.json", group); FILE *f = fopen(p, "w"); if (!f) return -1;
    fprintf(f, "{\"zarr_format\":3,\"node_type\":\"group\",\"attributes\":{\"format\":\"ufsm dense embeddings\",\"box_zyx\":[%lld,%lld,%lld,%lld,%lld,%lld],"
               "\"lattice\":{\"stride\":1},\"dense_stride\":1}}\n", (long long)box[0], (long long)box[1], (long long)box[2], (long long)box[3], (long long)box[4], (long long)box[5]);
    fclose(f);
    zw w; snprintf(p, sizeof p, "%s/dense", group);
    if (zw_create(&w, p, vol_shape, C, "int8", 1, 128, 32, "{\"value_scale\":0.007874015748031496,\"values\":\"unit-norm embedding\"}")) return -1;
    const int S = 128; int8_t *blk = malloc((size_t)S * S * S * C); int rc = 0;
    for (int64_t bz = box[0] / S * S; bz < box[0] + box[3] && !rc; bz += S)
        for (int64_t by = box[1] / S * S; by < box[1] + box[4] && !rc; by += S)
            for (int64_t bx = box[2] / S * S; bx < box[2] + box[5] && !rc; bx += S) {
                memset(blk, 0, (size_t)S * S * S * C);
                for (int64_t z = 0; z < S; z++) for (int64_t y = 0; y < S; y++) {
                    const int64_t gz = bz + z, gy = by + y, lz = gz - box[0], ly = gy - box[1];
                    if (lz < 0 || ly < 0 || lz >= box[3] || ly >= box[4]) continue;
                    const int64_t x0 = bx > box[2] ? bx : box[2], x1 = bx + S < box[2] + box[5] ? bx + S : box[2] + box[5];
                    if (x1 > x0) memcpy(blk + ((z * S + y) * S + (x0 - bx)) * C, data + (((size_t)lz * box[4] + ly) * box[5] + (x0 - box[2])) * C, (size_t)(x1 - x0) * C);
                }
                rc = zw_put(&w, bz, by, bx, S, S, S, blk);
            }
    rc |= zw_flush(&w); zw_free(&w); free(blk);
    return rc;
}


/* ---- ufsm dinovol-calib: the upstream teacher's DINO (CLS) and iBOT (patch token) heads on our crops -- the yardstick for our SSL
   entropy diagnostics. Pairs of crops (view A, view B = A shifted by a multiple of 8 voxels, so token cells correspond exactly) from a
   sources JSON (dino-train format: root, ct, cache, boxes, min_level), normalised with robust statistics pooled over each source's crops;
   per head: H(t) / pmax / batch usage / A-B argmax agreement under plain softmax, batch centering and Sinkhorn-Knopp at each temperature,
   and the unit bottleneck's per-dimension std and same / other cosines (as dino-train logs them). ---- */
#include "dinor.h"
#include "wload.h"
#include "nn.h"
typedef struct { dr_head h; float *P; } cal_head;
/* one of the checkpoint's DINOHeads (pre "dino_head" / "ibot_head") as a dr_head layout + host floats (caller frees) */
int dv_head_host(const char *ckpt, const char *pre, dr_head *h, float **Ph) {
    wl_dict *d = wl_load(ckpt, "teacher"); if (!d) { fprintf(stderr, "dinovol: %s\n", wl_error()); return -1; }
    const wl_tensor *w1 = wl_getf(d, "%s.mlp.0.weight", pre), *b1 = wl_getf(d, "%s.mlp.0.bias", pre), *w2 = wl_getf(d, "%s.mlp.2.weight", pre),
                    *b2 = wl_getf(d, "%s.mlp.2.bias", pre), *w3 = wl_getf(d, "%s.mlp.4.weight", pre), *b3 = wl_getf(d, "%s.mlp.4.bias", pre),
                    *g = wl_getf(d, "%s.last_layer.parametrizations.weight.original0", pre), *v = wl_getf(d, "%s.last_layer.parametrizations.weight.original1", pre);
    if (!w1 || !b1 || !w2 || !b2 || !w3 || !b3 || !g || !v) { fprintf(stderr, "dinovol: %s: head tensors missing\n", pre); wl_free(d); return -1; }
    dr_head_layout(h, (int)w1->shape[1], (int)w1->shape[0], (int)w3->shape[0], (int)v->shape[0]);
    float *P = calloc(h->n, sizeof *P);
    memcpy(P + h->w1, w1->data, w1->numel * 4); memcpy(P + h->b1, b1->data, b1->numel * 4); memcpy(P + h->w2, w2->data, w2->numel * 4);
    memcpy(P + h->b2, b2->data, b2->numel * 4); memcpy(P + h->w3, w3->data, w3->numel * 4); memcpy(P + h->b3, b3->data, b3->numel * 4);
    memcpy(P + h->v, v->data, v->numel * 4); memcpy(P + h->g, g->data, g->numel * 4);
    double gs = 0, gmin = 1e30, gmax = -1e30; for (int64_t k = 0; k < g->numel; k++) { gs += g->data[k]; gmin = fmin(gmin, g->data[k]); gmax = fmax(gmax, g->data[k]); }
    fprintf(stderr, "dinovol: %s: %d -> %d -> %d -> bottleneck %d, K %d prototypes; weight-norm g mean %.4f min %.4f max %.4f\n", pre, h->Cin, h->H, h->H, h->B, h->K, gs / (double)g->numel, gmin, gmax);
    wl_free(d); *Ph = P; return 0;
}
/* robust statistics from a 256-bin histogram (bin 0 ignored), as dinovol-embed's */
int dv_hist_stats(const int64_t *hist256, double *median, double *scale, double *p99) {
    int64_t h[256]; memcpy(h, hist256, sizeof h); h[0] = 0; dv_stats S = {0};
    if (stats_from_hist(h, 256, &S)) return -1;
    *median = S.median; *scale = S.scale; *p99 = S.p99; return 0;
}
/* the volume-wide robust normalisation of dinovol-embed (nonzero voxels of every stride-th chunk per axis) */
int dv_ct_stats(z3 *ct, int stride, int nthreads, double *median, double *scale, double *p99) {
    dv_stats S = {0}; if (sample_stats(ct, stride, nthreads, &S)) return -1;
    *median = S.median; *scale = S.scale; *p99 = S.p99; return 0;
}
static int cal_head_load(cal_head *c, const char *ck, const char *pre) {
    float *P = nullptr; if (dv_head_host(ck, pre, &c->h, &P)) return -1;
    c->P = xdev_floats(P, c->h.n); free(P);
    return c->P ? 0 : -1;
}
/* logits [R][K] (host) and unit bottleneck zn [R][B] (host) for rows x [R][Cin] (host) */
static int cal_head_run(const cal_head *c, const float *x, int R, float *L, float *zn) {
    const int CH = 256, K = c->h.K, B = c->h.B, Cin = c->h.Cin;
    void *ws = xmalloc_dev(dr_hws_bytes(&c->h, CH)); float *xd = xmalloc_dev((size_t)CH * Cin * 4), *ld = xmalloc_dev((size_t)CH * K * 4);
    if (!ws || !xd || !ld) { fprintf(stderr, "dinovol-calib: out of device memory\n"); return -1; }
    dr_hws w; dr_hws_carve(&w, &c->h, CH, ws);
    for (int r0 = 0; r0 < R; r0 += CH) {
        const int n = R - r0 < CH ? R - r0 : CH;
        xh2d(xd, x + (size_t)r0 * Cin, (size_t)n * Cin * 4);
        if (dr_head_fwd(&c->h, c->P, xd, n, ld, &w)) return -1;
        nn_sync();   /* (the head runs on the default stream, the copies on xnet's non-blocking one) */
        xd2h(L + (size_t)r0 * K, ld, (size_t)n * K * 4); xd2h(zn + (size_t)r0 * B, w.zn, (size_t)n * B * 4);
    }
    xfree_dev(ws); xfree_dev(xd); xfree_dev(ld);
    return 0;
}
/* targets q [R][K] (host) from logits: mode 0 softmax(L / t), 1 softmax((L - batch mean) / t), 2 Sinkhorn-Knopp (iters) at t */
static void cal_targets(const float *L, int R, int K, double t, int mode, int iters, float *q) {
    double *c = calloc(K, sizeof *c);
    if (mode == 1) { for (int r = 0; r < R; r++) for (int k = 0; k < K; k++) c[k] += L[(size_t)r * K + k]; for (int k = 0; k < K; k++) c[k] /= R; }
    if (mode < 2) {
        for (int r = 0; r < R; r++) {
            const float *l = L + (size_t)r * K; float *o = q + (size_t)r * K; double m = -1e30, s = 0;
            for (int k = 0; k < K; k++) m = fmax(m, (l[k] - c[k]) / t);
            for (int k = 0; k < K; k++) { o[k] = (float)exp((l[k] - c[k]) / t - m); s += o[k]; }
            for (int k = 0; k < K; k++) o[k] = (float)(o[k] / s);
        }
    } else {   /* upstream sinkhorn_knopp_teacher: Q = exp(L / t)^T; Q /= sum; iters x (rows: / sum / K; cols: / sum / B); Q *= B */
        double m = -1e30; for (size_t i = 0; i < (size_t)R * K; i++) m = fmax(m, L[i] / t);
        double s = 0; for (size_t i = 0; i < (size_t)R * K; i++) { q[i] = (float)exp(L[i] / t - m); s += q[i]; }
        for (size_t i = 0; i < (size_t)R * K; i++) q[i] = (float)(q[i] / s);
        for (int it = 0; it < iters; it++) {
            for (int k = 0; k < K; k++) c[k] = 0;
            for (int r = 0; r < R; r++) for (int k = 0; k < K; k++) c[k] += q[(size_t)r * K + k];
            for (int r = 0; r < R; r++) for (int k = 0; k < K; k++) q[(size_t)r * K + k] = c[k] > 0 ? (float)(q[(size_t)r * K + k] / c[k] / K) : 0.f;
            for (int r = 0; r < R; r++) { double rs = 0; float *o = q + (size_t)r * K; for (int k = 0; k < K; k++) rs += o[k]; for (int k = 0; k < K; k++) o[k] = (float)(o[k] / rs / R); }
        }
        for (size_t i = 0; i < (size_t)R * K; i++) q[i] *= R;
    }
    free(c);
}
/* rows in pairs (2 i: view A, 2 i + 1: view B) */
static void cal_report(const char *what, const float *L, const float *zn, int R, int K, int B, const double *temps, int nt, int iters, int same_group) {
    float *q = malloc((size_t)R * K * 4); double *bar = malloc((size_t)K * sizeof *bar); int *am = malloc(R * sizeof *am); char *seen = malloc(K);
    fprintf(stderr, "%s: %d rows (%d pairs), K %d (ln K %.3f)\n", what, R, R / 2, K, log((double)K));
    { double ls = 0, lq = 0; for (size_t i = 0; i < (size_t)R * K; i++) { ls += L[i]; lq += (double)L[i] * L[i]; }
      const double mu = ls / ((double)R * K); double rsd = 0;
      for (int r = 0; r < R; r++) { double a = 0, b = 0; for (int k = 0; k < K; k++) { a += L[(size_t)r * K + k]; b += (double)L[(size_t)r * K + k] * L[(size_t)r * K + k]; } a /= K; rsd += sqrt(fmax(b / K - a * a, 0)); }
      fprintf(stderr, "  raw logits: mean %.4f sd %.4f, mean within-row sd %.4f\n", mu, sqrt(fmax(lq / ((double)R * K) - mu * mu, 0)), rsd / R); }
    static const char *mn[3] = {"softmax", "centred", "sinkhorn"};
    for (int ti = 0; ti < nt; ti++) for (int mode = 0; mode < 3; mode++) {
        cal_targets(L, R, K, temps[ti], mode, iters, q);
        double H = 0, pm = 0, agree = 0, ce_same = 0; memset(bar, 0, (size_t)K * sizeof *bar); memset(seen, 0, K); int distinct = 0;
        for (int r = 0; r < R; r++) {
            const float *o = q + (size_t)r * K; double h = 0, mx = 0; int a = 0;
            for (int k = 0; k < K; k++) { const double p = o[k]; if (p > 0) h -= p * log(p); if (p > mx) { mx = p; a = k; } bar[k] += p / R; }
            H += h / R; pm += mx / R; am[r] = a; if (!seen[a]) { seen[a] = 1; distinct++; }
        }
        for (int i = 0; i < R / 2; i++) {
            agree += am[2 * i] == am[2 * i + 1];
            const float *a = q + (size_t)(2 * i) * K, *b = q + (size_t)(2 * i + 1) * K; double ce = 0;
            for (int k = 0; k < K; k++) if (a[k] > 0) ce -= a[k] * log(fmax(b[k], 1e-30));
            ce_same += ce;
        }
        double Hb = 0; for (int k = 0; k < K; k++) if (bar[k] > 0) Hb -= bar[k] * log(bar[k]);
        fprintf(stderr, "  t %.3f %-8s H(t) %.3f (%.3f of ln K, exp H %.0f)  pmax %.4f  H(batch mean) %.3f  distinct argmax %d / %d  A-B argmax agree %.3f  CE(A->B) %.3f\n",
                temps[ti], mn[mode], H, H / log((double)K), exp(H), pm, Hb, distinct, R, agree / (R / 2), ce_same / (R / 2));
    }
    /* unit bottleneck: per-dimension std, same (A-B) vs other cosines */
    { double *mu = calloc(B, sizeof *mu), *va = calloc(B, sizeof *va), *sd = malloc(B * sizeof *sd);
      for (int r = 0; r < R; r++) for (int b = 0; b < B; b++) mu[b] += zn[(size_t)r * B + b] / R;
      for (int r = 0; r < R; r++) for (int b = 0; b < B; b++) { const double d = zn[(size_t)r * B + b] - mu[b]; va[b] += d * d / R; }
      double sm = 0, mn2 = 1e30; for (int b = 0; b < B; b++) { sd[b] = sqrt(va[b]); sm += sd[b] / B; mn2 = fmin(mn2, sd[b]); }
      for (int i = 1; i < B; i++) { const double k = sd[i]; int j = i - 1; while (j >= 0 && sd[j] > k) { sd[j + 1] = sd[j]; j--; } sd[j + 1] = k; }
      double cs = 0, co = 0; long no = 0;
      for (int i = 0; i < R / 2; i++) { double d = 0; for (int b = 0; b < B; b++) d += zn[(size_t)(2 * i) * B + b] * zn[(size_t)(2 * i + 1) * B + b]; cs += d / (R / 2); }
      for (int r = 0; r < R; r++) for (int s = r + 1; s < R; s += 7) { if (r / same_group == s / same_group) continue; double d = 0; for (int b = 0; b < B; b++) d += zn[(size_t)r * B + b] * zn[(size_t)s * B + b]; co += d; no++; }
      double mun = 0; for (int b = 0; b < B; b++) mun += mu[b] * mu[b];
      fprintf(stderr, "  bottleneck z (unit, %d dims): std per dim mean %.4f min %.4f median %.4f (1/sqrt(B) = %.4f), |batch mean| %.4f, cos same %.4f other %.4f\n",
              B, sm, mn2, sd[B / 2], 1 / sqrt((double)B), sqrt(mun), cs, no ? co / no : 0);
      free(mu); free(va); free(sd); }
    free(q); free(bar); free(am); free(seen);
}
int cmd_dinovol_calib(int argc, char **argv) {
    const char *ck = opt(argc, argv, "--ckpt", nullptr), *sj = opt(argc, argv, "--sources", nullptr);
    if (!ck || !sj) {
        fprintf(stderr, "usage: ufsm dinovol-calib --ckpt teacher_step_640000.pt --sources SOURCES.json [--crops 128] [--tile 128] [--tokens 4] [--shift 32]\n"
                        "         [--temps 0.04,0.07] [--sk-iters 3] [--seed 1] [--max-sources 0] [--gpu 0] [--gpu-mem 9] [--threads 8]\n"
                        "  the upstream dinovol teacher's DINO (CLS) and iBOT (token) heads on crop pairs of our data: H(t), pmax, usage, A-B agreement\n");
        return 2;
    }
    if (xnet_init(atoi(opt(argc, argv, "--gpu", "0")))) return 1;
    xnet_set_budget((size_t)(atof(opt(argc, argv, "--gpu-mem", "9")) * 1e9));
    const int N = atoi(opt(argc, argv, "--crops", "128")), T = atoi(opt(argc, argv, "--tile", "128")), NT = atoi(opt(argc, argv, "--tokens", "4"));
    const int SH = atoi(opt(argc, argv, "--shift", "32")) / 8 * 8, iters = atoi(opt(argc, argv, "--sk-iters", "3")), nth = atoi(opt(argc, argv, "--threads", "8"));
    const int maxs = atoi(opt(argc, argv, "--max-sources", "0"));
    double temps[8]; int nt = 0;
    { char *s = strdup(opt(argc, argv, "--temps", "0.04,0.07")); for (char *t = strtok(s, ","); t && nt < 8; t = strtok(nullptr, ",")) temps[nt++] = atof(t); free(s); }
    uint64_t rs = (uint64_t)atoll(opt(argc, argv, "--seed", "1")) * 0x9E3779B97F4A7C15ull + 1;
    if (T % 16 || N < 2 || NT < 1) { fprintf(stderr, "dinovol-calib: tile %% 16, crops >= 2, tokens >= 1\n"); return 2; }
    size_t jn = 0; char *jt = read_file(sj, &jn); json *J = jt ? json_parse(jt, jn) : nullptr; free(jt);
    const json *srcs = J ? json_get(J, "sources") : nullptr;
    if (!srcs || srcs->type != J_ARR || !srcs->n) { fprintf(stderr, "dinovol-calib: cannot read sources from %s\n", sj); return 1; }
    /* open the sources */
    int ns = 0; z3 **cts = calloc(srcs->n, sizeof *cts); int64_t (*boxes)[64][6] = calloc(srcs->n, sizeof *boxes); int *nbox = calloc(srcs->n, sizeof *nbox); const char **names = calloc(srcs->n, sizeof *names);
    for (size_t i = 0; i < srcs->n && (!maxs || ns < maxs); i++) {
        const json *s = json_at(srcs, i);
        const char *root = json_str(json_get(s, "root"), nullptr), *ct = json_str(json_get(s, "ct"), nullptr);
        if (!root || !ct) continue;
        char ctk[1024]; snprintf(ctk, sizeof ctk, "%s", ct);
        if (json_get(s, "min_level")) snprintf(ctk, sizeof ctk, "%s/%d", ct, (int)json_num(json_get(s, "min_level"), 0));
        store *st = store_open(root); z3 *z = st ? z3_open(st, ctk, json_str(json_get(s, "cache"), nullptr)) : nullptr;
        if (!z) { fprintf(stderr, "dinovol-calib: skipping %s (cannot open)\n", json_str(json_get(s, "name"), ctk)); continue; }
        const z3_meta *m = z3_meta_of(z); const json *bx = json_get(s, "boxes"); int nb = 0;
        if (bx && bx->type == J_ARR) for (size_t k = 0; k < bx->n && nb < 64; k++) {
            const json *b = json_at(bx, k); int64_t v[6]; int ok = b && b->type == J_ARR && b->n == 6;
            for (int d = 0; ok && d < 6; d++) v[d] = (int64_t)json_num(json_at(b, d), 0);
            if (ok && v[3] >= T + 2 * SH && v[4] >= T + 2 * SH && v[5] >= T + 2 * SH) { memcpy(boxes[ns][nb], v, sizeof v); nb++; }
        }
        if (!nb && m->shape[0] >= T + 2 * SH && m->shape[1] >= T + 2 * SH && m->shape[2] >= T + 2 * SH) { for (int d = 0; d < 3; d++) { boxes[ns][0][d] = 0; boxes[ns][0][3 + d] = m->shape[d]; } nb = 1; }
        if (!nb) { z3_close(z); continue; }
        cts[ns] = z; nbox[ns] = nb; names[ns] = json_str(json_get(s, "name"), ctk); ns++;
    }
    if (!ns) { fprintf(stderr, "dinovol-calib: no usable source\n"); return 1; }
    /* draw the crop pairs: source round robin, view A >= 50% nonzero voxels, view B = A + shift (multiples of 8, |s| <= SH per axis) */
    const size_t V = (size_t)T * T * T; uint8_t *crops = malloc(2 * (size_t)N * V); int *src = malloc(N * sizeof *src);
    int64_t (*org)[2][3] = malloc(N * sizeof *org);
    const double t0 = dv_now();
    for (int i = 0; i < N; i++) {
        const int s = i % ns; src[i] = s; int got = 0;
        for (int tries = 0; tries < 60 && !got; tries++) {
            const int64_t *b = boxes[s][(int)(ev_unif(&rs) * nbox[s])];
            int64_t o[3], o2[3];
            for (int d = 0; d < 3; d++) {
                o[d] = b[d] + SH + (int64_t)(ev_unif(&rs) * (double)(b[3 + d] - T - 2 * SH + 1));
                o2[d] = o[d] + 8 * ((int64_t)(ev_unif(&rs) * (2 * SH / 8 + 1)) - SH / 8);
            }
            const int64_t n3[3] = {T, T, T}; uint8_t *a = crops + 2 * (size_t)i * V;
            if (z3_read(cts[s], o, n3, a, nth)) { fprintf(stderr, "dinovol-calib: read: %s\n", z3_error()); return 1; }
            size_t nz = 0; for (size_t k = 0; k < V; k++) nz += a[k] != 0;
            if (nz * 2 < V) continue;
            if (z3_read(cts[s], o2, n3, a + V, nth)) { fprintf(stderr, "dinovol-calib: read: %s\n", z3_error()); return 1; }
            memcpy(org[i][0], o, sizeof o); memcpy(org[i][1], o2, sizeof o2); got = 1;
        }
        if (!got) { fprintf(stderr, "dinovol-calib: %s: no crop with >= 50%% nonzero voxels in 60 tries\n", names[s]); return 1; }
    }
    fprintf(stderr, "dinovol-calib: %d crop pairs of %d^3 from %d sources (%.1f s)\n", N, T, ns, dv_now() - t0);
    /* per-source robust normalisation from the pooled histogram of its crops */
    dv_stats *SS = calloc(ns, sizeof *SS);
    for (int s = 0; s < ns; s++) {
        int64_t *hist = calloc(256, sizeof *hist);
        for (int i = 0; i < N; i++) if (src[i] == s) for (size_t k = 0; k < 2 * V; k++) hist[crops[2 * (size_t)i * V + k]]++;
        hist[0] = 0; if (stats_from_hist(hist, 256, &SS[s])) { fprintf(stderr, "dinovol-calib: %s: empty histogram\n", names[s]); return 1; }
        fprintf(stderr, "  %-40s median %.2f scale %.2f p99 %.2f\n", names[s], SS[s].median, SS[s].scale, SS[s].p99);
        free(hist);
    }
    /* the teacher: CLS rows [2N][E] and token rows [2N NT][E] (the same NT cells in A and B: cell g of B = cell g + shift / 8 of A) */
    dv_net *net = dv_load(ck); if (!net) return 1;
    const int E = dv_embed_dim(net), G = T / dv_patch(net);
    float *cls = malloc((size_t)2 * N * E * 4), *tok = malloc((size_t)2 * N * NT * E * 4);
    uint16_t *th = malloc((size_t)G * G * G * E * 2), ch[1024];
    uint8_t *ud = xmalloc_dev(V);
    const double t1 = dv_now();
    for (int i = 0; i < N; i++) {
        int ca[64][3]; const int64_t *oa = org[i][0], *ob = org[i][1]; int sc[3];
        for (int d = 0; d < 3; d++) sc[d] = (int)((oa[d] - ob[d]) / 8);   /* cell g of A = cell g + sc of B */
        for (int k = 0; k < NT && k < 64; k++) for (int d = 0; d < 3; d++) {
            const int lo = sc[d] < 0 ? -sc[d] : 0, hi = G - (sc[d] > 0 ? sc[d] : 0);   /* g in [lo, hi): g + sc inside B */
            ca[k][d] = lo + (int)(ev_unif(&rs) * (hi - lo));
        }
        for (int v = 0; v < 2; v++) {
            xh2d(ud, crops + (2 * (size_t)i + v) * V, V);
            if (dv_forward(net, ud, T, (float)SS[src[i]].median, (float)SS[src[i]].scale, (float)SS[src[i]].p99)) return 1;
            xd2h(ch, dv_cls(net), (size_t)E * 2); for (int e = 0; e < E; e++) cls[((size_t)2 * i + v) * E + e] = h2f(ch[e]);
            xd2h(th, dv_tokens(net), (size_t)G * G * G * E * 2);
            for (int k = 0; k < NT && k < 64; k++) {
                const int g0 = ca[k][0] + (v ? sc[0] : 0), g1 = ca[k][1] + (v ? sc[1] : 0), g2 = ca[k][2] + (v ? sc[2] : 0);
                const uint16_t *t = th + (((size_t)g0 * G + g1) * G + g2) * E;
                float *o = tok + (((size_t)2 * i * NT + 2 * k + v)) * E;   /* pairs (A, B) adjacent */
                for (int e = 0; e < E; e++) o[e] = h2f(t[e]);
            }
        }
    }
    xfree_dev(ud); free(th);
    fprintf(stderr, "dinovol-calib: teacher forward of %d views (%.1f s)\n", 2 * N, dv_now() - t1);
    { double cs = 0, co = 0; long no = 0;
      for (int r = 0; r < 2 * N; r++) for (int s = r + 1; s < 2 * N; s++) {
          double d = 0, a = 0, b = 0; for (int e = 0; e < E; e++) { d += cls[(size_t)r * E + e] * cls[(size_t)s * E + e]; a += cls[(size_t)r * E + e] * cls[(size_t)r * E + e]; b += cls[(size_t)s * E + e] * cls[(size_t)s * E + e]; }
          const double c = d / sqrt(a * b + 1e-30); if (r / 2 == s / 2) cs += c; else { co += c; no++; } }
      fprintf(stderr, "backbone CLS (normed, %d dims): cos same %.4f other %.4f\n", E, cs / N, co / no); }
    cal_head hd, hi; if (cal_head_load(&hd, ck, "dino_head") || cal_head_load(&hi, ck, "ibot_head")) return 1;
    {   float *L = malloc((size_t)2 * N * hd.h.K * 4), *zn = malloc((size_t)2 * N * hd.h.B * 4);
        if (cal_head_run(&hd, cls, 2 * N, L, zn)) return 1;
        cal_report("DINO head on the CLS", L, zn, 2 * N, hd.h.K, hd.h.B, temps, nt, iters, 2);
        free(L); free(zn); }
    if (opt(argc, argv, "--dump-tokens", nullptr)) { FILE *f = fopen(opt(argc, argv, "--dump-tokens", ""), "wb"); if (f) { fwrite(tok, 4, (size_t)2 * N * NT * E, f); fclose(f); }
        fprintf(stderr, "dinovol-calib: %d token rows x %d written\n", 2 * N * NT, E); }
    {   const int R = 2 * N * NT; float *L = malloc((size_t)R * hi.h.K * 4), *zn = malloc((size_t)R * hi.h.B * 4);
        if (cal_head_run(&hi, tok, R, L, zn)) return 1;
        cal_report("iBOT head on patch tokens", L, zn, R, hi.h.K, hi.h.B, temps, nt, iters, 2 * NT);
        free(L); free(zn); }
    return 0;
}


/* ---- ufsm dino-headfit (debug): can our iBOT student head fit a FIXED batch of dinovol iBOT targets? Teacher targets: softmax(dinovol
   ibot_head(tokens) / tt) of dinovol-calib --dump-tokens rows; student: the SSL batch's iBOT head (32 inputs -> 2048 -> 2048 -> 256, last
   layer = dinovol's prototypes, frozen), AdamW on the MLP; inputs: --input random (fixed random unit vectors: memorisation) or proj (a fixed
   random projection of the teacher's own token to 32 dims, unit-normed: the information is there). Prints CE (vs the targets' entropy and
   ln K), student H / pmax, z cos-other every --every steps. ---- */
int cmd_dino_headfit(int argc, char **argv) {
    const char *ck = opt(argc, argv, "--ckpt", nullptr), *tf = opt(argc, argv, "--tokens", nullptr);
    if (!ck || !tf) { fprintf(stderr, "usage: ufsm dino-headfit --ckpt PT --tokens ROWS.f32 [--rows 256] [--input random|proj] [--steps 300] [--lr 1e-3] [--wd 0.04] [--tt 0.04] [--ts 0.1] [--train-last 0] [--gpu 0]\n"); return 2; }
    if (nn_init(atoi(opt(argc, argv, "--gpu", "0")))) return 1;
    dr_head T; float *Th = nullptr; if (dv_head_host(ck, "ibot_head", &T, &Th)) return 1;
    size_t nb = 0; float *tok = read_file(tf, &nb); if (!tok) { fprintf(stderr, "cannot read %s\n", tf); return 1; }
    const int E = T.Cin, Rall = (int)(nb / 4 / E), R = atoi(opt(argc, argv, "--rows", "256")) < Rall ? atoi(opt(argc, argv, "--rows", "256")) : Rall, K = T.K, B = T.B;
    const int steps = atoi(opt(argc, argv, "--steps", "300")), every = atoi(opt(argc, argv, "--every", "20")), proj = !strcmp(opt(argc, argv, "--input", "proj"), "proj"), trl = atoi(opt(argc, argv, "--train-last", "0"));
    const float lr = (float)atof(opt(argc, argv, "--lr", "1e-3")), wd = (float)atof(opt(argc, argv, "--wd", "0.04")), tt = (float)atof(opt(argc, argv, "--tt", "0.04")), ts = (float)atof(opt(argc, argv, "--ts", "0.1"));
    /* teacher targets */
    float *TP = nn_malloc(T.n * 4); nn_h2d(TP, Th, T.n * 4);
    float *xt = nn_malloc((size_t)R * E * 4); nn_h2d(xt, tok, (size_t)R * E * 4);
    void *tw = nn_malloc(dr_hws_bytes(&T, R)); dr_hws twp; dr_hws_carve(&twp, &T, R, tw);
    float *tl = nn_malloc((size_t)R * K * 4), *q = nn_malloc((size_t)R * K * 4); void *sk = nn_malloc(dr_sk_scratch(R, K));
    dr_head_fwd(&T, TP, xt, R, tl, &twp); dr_teacher_probs(tl, R, K, tt, 0, nullptr, q, sk);
    double Hq = 0; { float *qh = malloc((size_t)R * K * 4); nn_d2h(qh, q, (size_t)R * K * 4);
        for (int r = 0; r < R; r++) for (int k = 0; k < K; k++) { const double v = qh[(size_t)r * K + k]; if (v > 0) Hq -= v * log(v) / R; }
        double *bar = calloc(K, 8); for (int r = 0; r < R; r++) for (int k = 0; k < K; k++) bar[k] += qh[(size_t)r * K + k] / R;
        double Hb = 0; for (int k = 0; k < K; k++) if (bar[k] > 0) Hb -= bar[k] * log(bar[k]);
        fprintf(stderr, "dino-headfit: %d rows, K %d (ln K %.3f), targets t %.3f: mean H %.3f, H(batch mean) %.3f (the CE floor for a constant student)\n", R, K, log((double)K), tt, Hq, Hb);
        free(qh); free(bar); }
    /* student inputs [R][32] */
    float *xs = malloc((size_t)R * 32 * 4); uint64_t rs = 12345;
    float *Pm = malloc((size_t)32 * E * 4); for (int i = 0; i < 32 * E; i++) Pm[i] = (float)(sqrt(-2 * log(ev_unif(&rs) + 1e-12)) * cos(6.283185307 * ev_unif(&rs)));
    for (int r = 0; r < R; r++) { double n2 = 0;
        for (int c = 0; c < 32; c++) { double v = 0; if (proj) for (int e = 0; e < E; e++) v += Pm[(size_t)c * E + e] * tok[(size_t)r * E + e]; else v = sqrt(-2 * log(ev_unif(&rs) + 1e-12)) * cos(6.283185307 * ev_unif(&rs)); xs[(size_t)r * 32 + c] = (float)v; n2 += v * v; }
        for (int c = 0; c < 32; c++) xs[(size_t)r * 32 + c] /= (float)sqrt(n2 + 1e-30); }
    float *xsd = nn_malloc((size_t)R * 32 * 4); nn_h2d(xsd, xs, (size_t)R * 32 * 4);
    /* student head: dn_head_init's init (trunc-normal 0.02 weights, zero biases), last layer = dinovol's */
    dr_head S; dr_head_layout(&S, 32, T.H, B, K);
    float *Sh = calloc(S.n, 4); uint64_t r2 = 777;
    for (size_t i = S.w1; i < S.b1; i++) Sh[i] = (float)(0.02 * sqrt(-2 * log(ev_unif(&r2) + 1e-12)) * cos(6.283185307 * ev_unif(&r2)));
    for (size_t i = S.w2; i < S.b2; i++) Sh[i] = (float)(0.02 * sqrt(-2 * log(ev_unif(&r2) + 1e-12)) * cos(6.283185307 * ev_unif(&r2)));
    for (size_t i = S.w3; i < S.b3; i++) Sh[i] = (float)(0.02 * sqrt(-2 * log(ev_unif(&r2) + 1e-12)) * cos(6.283185307 * ev_unif(&r2)));
    memcpy(Sh + S.v, Th + T.v, (size_t)K * B * 4); memcpy(Sh + S.g, Th + T.g, (size_t)K * 4);
    float *SP = nn_malloc(S.n * 4), *SG = nn_malloc(S.n * 4), *SM = nn_malloc(S.n * 4), *SV = nn_malloc(S.n * 4); nn_h2d(SP, Sh, S.n * 4); nn_zero(SM, S.n * 4); nn_zero(SV, S.n * 4);
    void *sw = nn_malloc(dr_hws_bytes(&S, R)); dr_hws swp; dr_hws_carve(&swp, &S, R, sw);
    float *sl = nn_malloc((size_t)R * K * 4), *dsl = nn_malloc((size_t)R * K * 4), *dx = nn_malloc((size_t)R * 32 * 4), *out = nn_malloc(16);
    void *cs = nn_malloc(dr_ce_scratch(R)); int *iid = nn_malloc((size_t)R * 4); float *pw = nn_malloc((size_t)R * 4);
    { int *h = malloc((size_t)R * 4); float *w = malloc((size_t)R * 4); for (int i = 0; i < R; i++) { h[i] = i; w[i] = 1.f / R; } nn_h2d(iid, h, (size_t)R * 4); nn_h2d(pw, w, (size_t)R * 4); free(h); free(w); }
    float *slh = malloc((size_t)R * K * 4), *znh = malloc((size_t)R * B * 4);
    for (int step = 1; step <= steps; step++) {
        nn_zero(SG, S.n * 4);
        dr_head_fwd(&S, SP, xsd, R, sl, &swp);
        dr_ce(sl, R, K, ts, q, iid, iid, pw, R, dsl, out, cs);
        dr_head_bwd(&S, SP, SG, xsd, dsl, R, dx, trl, &swp);
        dr_adamw(SP, SG, SM, SV, trl ? S.g : S.v, lr, 0.9f, 0.999f, 1e-8f, wd, step);
        if (step == 1 || step % every == 0 || step == steps) {
            float L; nn_d2h(&L, out, 4); nn_d2h(slh, sl, (size_t)R * K * 4); nn_d2h(znh, swp.zn, (size_t)R * B * 4);
            double Hs = 0, pm = 0, co = 0, gn = 0; long no = 0;
            for (int r = 0; r < R; r++) { const float *l = slh + (size_t)r * K; double mx = -1e30, z = 0, h = 0; for (int k = 0; k < K; k++) mx = fmax(mx, l[k] / ts);
                for (int k = 0; k < K; k++) z += exp(l[k] / ts - mx);
                for (int k = 0; k < K; k++) { const double t = exp(l[k] / ts - mx) / z; if (t > 0) h -= t * log(t); } Hs += h / R; pm += 1 / z / R; }
            for (int a = 0; a < R; a += 3) for (int b = a + 1; b < R; b += 5) { double d = 0; for (int c = 0; c < B; c++) d += znh[(size_t)a * B + c] * znh[(size_t)b * B + c]; co += d; no++; }
            { float *gh = malloc(S.v * 4); nn_d2h(gh, SG, S.v * 4); for (size_t i = 0; i < S.v; i++) gn += (double)gh[i] * gh[i]; free(gh); }
            fprintf(stderr, "step %4d CE %.4f (targets H %.3f, H(mean) floor, ln K %.3f) student H %.3f pmax %.5f z cos other %.4f |grad| %.3e\n", step, L, Hq, log((double)K), Hs, pm, no ? co / no : 0, sqrt(gn));
        }
    }
    return 0;
}
