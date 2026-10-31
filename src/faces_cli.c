/* ufsm faces command: the CT through a chunk cache, the published-surface check, tifxyz in / out (core: src/faces.c). */
#include "faces.h"
#include "imgops.h"
#include "sources.h"
#include "store.h"
#include "tiffw.h"
#include "zarr3.h"
#include <dirent.h>
#include <malloc.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { SUB = 4 };   /* the published-surface check at every SUB-th row / column */
static void fc_par(void *(*fn)(void *), void *J, int nt) {
    pthread_t th[64]; const int k = nt < 1 ? 1 : nt > 64 ? 64 : nt;
    for (int t = 0; t < k; t++) pthread_create(&th[t], nullptr, fn, J);
    for (int t = 0; t < k; t++) pthread_join(th[t], nullptr);
}

fgpu *fgpu_open(const int64_t shape[3]) __attribute__((weak));
void fgpu_close(fgpu *G) __attribute__((weak));
int fgpu_broken(const fgpu *G) __attribute__((weak));
int fgpu_put(fgpu *G, size_t id, const uint8_t *buf, uint32_t mask) __attribute__((weak));
void fgpu_drop(fgpu *G, size_t id) __attribute__((weak));
void fgpu_stats(const fgpu *G, size_t *nput, size_t *bytes, int *slots) __attribute__((weak));

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

/* ---- the CT (or a label raster) through a cache of decoded chunks ---- */
typedef struct {
    z3 *z; int64_t shape[3]; int cs[3]; int64_t g[3]; uint8_t **ch; uint32_t *stamp; _Atomic uint8_t *need; size_t nres, cap; uint32_t clock; int nt;
    size_t ndec; double tdec, tev;   /* UFSM_FACES_PROF: fill decode (+ device upload) / eviction seconds */
    fgpu *dev;   /* non-null: every decoded chunk mirrored on the GPU (faces_vol.gpu) */
    _Atomic uint32_t *needm; uint32_t *have;
    uint32_t *ever; _Atomic size_t nsub, nresub;   /* UFSM_FACES_PROF: substreams ever decoded per chunk; decoded, decoded again after eviction */
    uint8_t **pool; int npool; pthread_mutex_t pmu;   /* evicted whole-chunk buffers for reuse (no page faults / zeroing per decode) */   /* sub: per chunk the volcomp substreams (16 z x 32 y slabs) marked / decoded; nullptr: whole chunks */
} fc_cache;
static int fcc_init(fc_cache *c, z3 *z, int nt, double mb) {
    memset(c, 0, sizeof *c); const z3_meta *m = z3_meta_of(z); c->z = z; c->nt = nt < 1 ? 1 : nt;
    for (int d = 0; d < 3; d++) { c->shape[d] = m->shape[d]; c->cs[d] = m->chunk[d] > 0 && m->chunk[d] <= 256 ? m->chunk[d] : 128; c->g[d] = (m->shape[d] + c->cs[d] - 1) / c->cs[d]; }
    const size_t ng = (size_t)c->g[0] * c->g[1] * c->g[2];
    c->ch = calloc(ng, sizeof *c->ch); c->stamp = calloc(ng, sizeof *c->stamp); c->need = calloc(ng, 1);
    c->cap = (size_t)(mb * 1048576.0 / ((double)c->cs[0] * c->cs[1] * c->cs[2])); if (c->cap < 16) c->cap = 16;
    c->pool = malloc(512 * sizeof *c->pool); pthread_mutex_init(&c->pmu, nullptr);
    return c->ch && c->stamp && c->need && c->pool ? 0 : -1;
}
/* the CT cache decodes only the volcomp substreams under the marked boxes (z3_read_sub; samples are only taken inside them, so the
   values are those of whole chunks). UFSM_FACES_FULLDECODE=1: whole chunks */
static int fcc_sub(fc_cache *c) {
    if (getenv("UFSM_FACES_FULLDECODE") || c->cs[0] != 128 || c->cs[1] != 128 || c->cs[2] != 128) return 0;
    const size_t ng = (size_t)c->g[0] * c->g[1] * c->g[2];
    c->needm = calloc(ng, sizeof *c->needm); c->have = calloc(ng, sizeof *c->have); if (getenv("UFSM_FACES_PROF")) c->ever = calloc(ng, sizeof *c->ever);
    return c->needm && c->have ? 0 : -1;
}
static void fcc_free(fc_cache *c) {
    const size_t ng = (size_t)c->g[0] * c->g[1] * c->g[2];
    for (size_t k = 0; c->ch && k < ng; k++) free(c->ch[k]);
    for (int k = 0; k < c->npool; k++) free(c->pool[k]);
    free(c->pool); free(c->ch); free((void *)c->need); free(c->stamp); free((void *)c->needm); free(c->have); free(c->ever); memset(c, 0, sizeof *c);
}
static inline int fcc_voxel(const fc_cache *c, int64_t z, int64_t y, int64_t x) {   /* -1: outside / not decoded */
    if (z < 0 || y < 0 || x < 0 || z >= c->shape[0] || y >= c->shape[1] || x >= c->shape[2]) return -1;
    const uint8_t *b = c->ch[((size_t)(z / c->cs[0]) * c->g[1] + (size_t)(y / c->cs[1])) * c->g[2] + (size_t)(x / c->cs[2])];
    return b ? b[((size_t)(z % c->cs[0]) * c->cs[1] + (size_t)(y % c->cs[1])) * c->cs[2] + (size_t)(x % c->cs[2])] : -1;
}
static double fcc_sample(void *ctx, double z, double y, double x) {
    const fc_cache *c = ctx;
    const int64_t a = (int64_t)floor(z), b = (int64_t)floor(y), e = (int64_t)floor(x); const double fz = z - a, fy = y - b, fx = x - e; double s = 0;
    for (int k = 0; k < 8; k++) {
        const int v = fcc_voxel(c, a + (k >> 2 & 1), b + (k >> 1 & 1), e + (k & 1)); if (v < 0) return NAN;
        s += (k >> 2 & 1 ? fz : 1 - fz) * (k >> 1 & 1 ? fy : 1 - fy) * (k & 1 ? fx : 1 - fx) * v;
    }
    return s;
}
static void fcc_mark(void *ctx, const double lo[3], const double hi[3]) {
    fc_cache *c = ctx; int64_t a[3], b[3];
    int64_t va[3], vb[3];
    for (int d = 0; d < 3; d++) {
        a[d] = (int64_t)floor(lo[d]); b[d] = (int64_t)floor(hi[d]) + 1;
        if (a[d] < 0) a[d] = 0;
        if (b[d] > c->shape[d] - 1) b[d] = c->shape[d] - 1;
        if (b[d] < a[d]) return;
        va[d] = a[d]; vb[d] = b[d];
        a[d] /= c->cs[d]; b[d] /= c->cs[d];
    }
    for (int64_t z = a[0]; z <= b[0]; z++) for (int64_t y = a[1]; y <= b[1]; y++) {
        uint32_t m = 0;
        if (c->needm) {   /* the box's z / y voxels inside chunk row (z, y) -> substream bits */
            int64_t za = va[0] - z * 128, ze = vb[0] - z * 128, ya = va[1] - y * 128, ye = vb[1] - y * 128;
            za = za < 0 ? 0 : za; ze = ze > 127 ? 127 : ze; ya = ya < 0 ? 0 : ya; ye = ye > 127 ? 127 : ye;
            for (int64_t bz = za >> 4; bz <= ze >> 4; bz++) for (int64_t by = ya >> 5; by <= ye >> 5; by++) m |= 1u << (bz * 4 + by);
        }
        for (int64_t x = a[2]; x <= b[2]; x++) {
            const size_t k = ((size_t)z * c->g[1] + y) * c->g[2] + x;
            if (c->needm && (atomic_load_explicit(&c->needm[k], memory_order_relaxed) & m) != m) atomic_fetch_or_explicit(&c->needm[k], m, memory_order_relaxed);
            atomic_store_explicit(&c->need[k], 1, memory_order_relaxed);
        }
    }
}
/* the decode list from per-chunk substream bits (faces_vol.mark_mask: the GPU computes the exact set the samples read) */
static void fcc_mark_mask(void *ctx, const uint32_t *mask) {
    fc_cache *c = ctx; const size_t ng = (size_t)c->g[0] * c->g[1] * c->g[2];
    for (size_t k = 0; k < ng; k++) if (mask[k]) { if (c->needm) c->needm[k] |= mask[k]; c->need[k] = 1; }
}
typedef struct { fc_cache *c; const int64_t *todo; size_t n; _Atomic size_t next; _Atomic int err; } fcd_job;
static void *fcc_decode(void *arg) {
    fcd_job *J = arg; fc_cache *c = J->c; const size_t cv = (size_t)c->cs[0] * c->cs[1] * c->cs[2];
    for (size_t k; (k = atomic_fetch_add(&J->next, 1)) < J->n;) {
        const int64_t id = J->todo[k], cz = id / (c->g[1] * c->g[2]), cy = id / c->g[2] % c->g[1], cx = id % c->g[2];
        int64_t o[3] = {cz * c->cs[0], cy * c->cs[1], cx * c->cs[2]}, nn[3];
        for (int d = 0; d < 3; d++) nn[d] = o[d] + c->cs[d] <= c->shape[d] ? c->cs[d] : c->shape[d] - o[d];
        const int whole = nn[0] == c->cs[0] && nn[1] == c->cs[1] && nn[2] == c->cs[2];
        const uint32_t want = c->needm ? atomic_load(&c->needm[id]) & ~c->have[id] : 0;   /* 0: the whole chunk */
        uint8_t *b = c->ch[id];
        if (b && c->needm && !want) continue;
        uint8_t *fresh = nullptr;
        if (whole && !b) { pthread_mutex_lock(&c->pmu); if (c->npool) fresh = c->pool[--c->npool]; pthread_mutex_unlock(&c->pmu); if (!fresh) fresh = malloc(cv); b = fresh; }
        uint8_t *t = fresh ? fresh : malloc((size_t)nn[0] * nn[1] * nn[2]);
        if (!t || (!whole && !b && !(b = calloc(cv, 1))) || z3_read_sub(c->z, o, nn, t, 1, want)) { if (t != b) free(t); if (!c->ch[id]) free(b); atomic_store(&J->err, 1); continue; }
        if (t != b) {   /* edge chunk, or substreams added to a decoded chunk: copy (only the new substreams' rows) */
            for (int64_t z = 0; z < nn[0]; z++) for (int64_t y = 0; y < nn[1]; y++)
                if (!want || (want >> ((z >> 4) * 4 + (y >> 5)) & 1)) memcpy(b + ((size_t)z * c->cs[1] + y) * c->cs[2], t + ((size_t)z * nn[1] + y) * nn[2], nn[2]);
            free(t);
        }
        if (c->needm) { const uint32_t wb = want ? want : 0xffffffffu; c->have[id] |= wb;
            if (c->ever) { atomic_fetch_add(&c->nsub, __builtin_popcount(wb)); atomic_fetch_add(&c->nresub, __builtin_popcount(wb & c->ever[id])); c->ever[id] |= wb; } }
        c->ch[id] = b;
        if (c->dev) fgpu_put(c->dev, id, b, c->needm ? c->have[id] : 0xffffffffu);   /* the device mirror (pinned staging, async) */
    }
    return nullptr;
}
static int fcc_fill(void *ctx) {
    fc_cache *c = ctx; const size_t ng = (size_t)c->g[0] * c->g[1] * c->g[2]; c->clock++;
    int64_t *todo = nullptr; size_t nt = 0, cap = 0, nnew = 0;
    for (size_t k = 0; k < ng; k++) if (c->need[k]) {
        c->need[k] = 0; c->stamp[k] = c->clock; if (c->ch[k] && (!c->needm || !(atomic_load(&c->needm[k]) & ~c->have[k]))) continue;
        if (nt == cap) { cap = cap ? 2 * cap : 256; todo = realloc(todo, cap * sizeof *todo); }
        nnew += !c->ch[k]; todo[nt++] = (int64_t)k;
    }
    double tq = now_s(); fcd_job J = {c, todo, nt, 0, 0}; fc_par(fcc_decode, &J, c->nt); c->tdec += now_s() - tq; tq = now_s();
    c->nres += nnew; c->ndec += nt;
    if (atomic_load(&J.err)) { free(todo); fprintf(stderr, "faces: %s\n", z3_error()); return -1; }
    free(todo); tq = now_s();
    while (c->nres > c->cap) {   /* least recently needed first, never this band's */
        uint32_t old = c->clock; for (size_t k = 0; k < ng; k++) if (c->ch[k] && c->stamp[k] < old) old = c->stamp[k];
        if (old == c->clock) break;
        for (size_t k = 0; k < ng && c->nres > c->cap; k++) if (c->ch[k] && c->stamp[k] == old) {
            if (c->npool < 512 && c->pool) c->pool[c->npool++] = c->ch[k]; else free(c->ch[k]);
            c->ch[k] = nullptr; c->nres--; if (c->needm) c->needm[k] = c->have[k] = 0;
            if (c->dev) fgpu_drop(c->dev, k);
        }
    }
    c->tev += now_s() - tq;
    return 0;
}

/* ---- the published surfaces (a winding-code raster: nonzero on the meshes) along the normal: where they sit in our sheet ---- */
typedef struct {
    const float *X, *Y, *Z, *RX, *RY, *RZ, *VX, *VY, *VZ; int w, h, r0, r1; double fg, R, hs; int sign; fc_cache *gc; int mark;
    float *dist;   /* per kept vertex: 3 signed distances (orig, recto, verso; GT voxels) and the fraction across the sheet */
    _Atomic int next;
} gt_job;
static void *gt_rows(void *arg) {
    gt_job *J = arg;
    for (int r; (r = J->r0 + SUB * atomic_fetch_add(&J->next, 1)) < J->r1;) for (int c = 0; c < J->w; c += SUB) {
        const size_t i = (size_t)r * J->w + c; double N[3];
        if (!(J->Z[i] >= 0) || !(J->RZ[i] >= 0) || !faces_normal(J->X, J->Y, J->Z, J->w, J->h, i, N)) continue;
        for (int k = 0; k < 3; k++) N[k] *= J->sign;
        const double p[3] = {(J->Z[i] + 0.5) * J->fg - 0.5, (J->Y[i] + 0.5) * J->fg - 0.5, (J->X[i] + 0.5) * J->fg - 0.5};
        if (J->mark) {
            double lo[3], hi[3]; for (int d = 0; d < 3; d++) { lo[d] = fmin(p[d] - J->R * N[d], p[d] + J->R * N[d]) - 1; hi[d] = fmax(p[d] - J->R * N[d], p[d] + J->R * N[d]) + 1; }
            fcc_mark(J->gc, lo, hi); continue;
        }
        double best = NAN;
        for (double t = 0; t <= J->R && isnan(best); t += J->hs) for (int s = -1; s <= 1; s += 2) {
            const double u = s * t;
            const int v = fcc_voxel(J->gc, (int64_t)lround(p[0] + u * N[0]), (int64_t)lround(p[1] + u * N[1]), (int64_t)lround(p[2] + u * N[2]));
            if (v > 0 && v < 255) { best = u; break; }
            if (t == 0) break;
        }
        float *o = &J->dist[((size_t)(r / SUB) * ((J->w + SUB - 1) / SUB) + c / SUB) * 4];
        if (isnan(best)) continue;
        /* the faces' offsets along this normal (GT voxels) */
        const double a = ((J->RZ[i] - J->Z[i]) * N[0] + (J->RY[i] - J->Y[i]) * N[1] + (J->RX[i] - J->X[i]) * N[2]) * J->fg;
        const double b = ((J->VZ[i] - J->Z[i]) * N[0] + (J->VY[i] - J->Y[i]) * N[1] + (J->VX[i] - J->X[i]) * N[2]) * J->fg;
        o[0] = (float)best; o[1] = (float)(best - a); o[2] = (float)(best - b); o[3] = b > a ? (float)((best - a) / (b - a)) : NAN;
    }
    return nullptr;
}

static const char *fopt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}
static int farg(int argc, char **argv, const char *name) { for (int i = 2; i < argc; i++) if (!strcmp(argv[i], name)) return i; return -1; }
static int cmp_flt(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }

static void meta_get(const char *dir, const char *name, double *um, double *scale, int *winding) {
    char p[1400]; snprintf(p, sizeof p, "%s/%s/meta.json", dir, name);
    char *b = tfx_meta_read(p); const char *q;
    *um = 0; *scale = 0; *winding = -1;
    if (!b) return;
    if ((q = strstr(b, "\"voxel_size_um\":"))) *um = atof(q + 16);
    if ((q = strstr(b, "\"scale\":")) && (q = strchr(q, '['))) *scale = atof(q + 1);
    if ((q = strstr(b, "\"winding\":"))) *winding = atoi(q + 10);
    free(b);
}

/* ufsm faces <wraps-dir> <out-dir> --ct ROOT KEY --ct-um U [--wraps a,b,..] [--umb JSON --umb-um U] [--range 150] [--thick 15,200]
   [--min-prom 8] [--rel 0.1] [--tol 25] [--inset 0] [--smooth 4] [--ct-thr 90] [--gt ROOT KEY --gt-um U [--gt-range 100]]
   [--threads 16] [--cache-mb 4096] [--cache DIR] [--gpu 1]
   Every wrap <w> (tifxyz dir or .sfc) -> <out>/<w>_recto and <out>/<w>_verso (tifxyz on the wrap's grid; conf.tif (uint8: 0 filled, else
   the band's contrast) and thick.tif (float, um) in <w>_recto), and <out>/<w>_faces.txt. Lengths in um. Statistics on stdout. */
int cmd_faces(int argc, char **argv) {
    const int ci = farg(argc, argv, "--ct");
    if (argc < 4 || ci < 0 || ci + 2 >= argc || !(atof(fopt(argc, argv, "--ct-um", "0")) > 0)) {
        fprintf(stderr, "usage: ufsm faces <wraps-dir> <out-dir> --ct ROOT KEY --ct-um U [--wraps a,b] [--umb JSON --umb-um U] [--range 150] [--thick 15,200]\n"
                        "       [--min-prom 8] [--rel 0.1] [--tol 25] [--inset 0] [--smooth 4] [--ct-thr 90] [--gt ROOT KEY --gt-um U [--gt-range 100]] [--threads 16] [--cache-mb 4096] [--gpu 1]\n"
                        "  recto (inner face) and verso (outer face) of the papyrus sheet under each wrap: <out>/<w>_recto, <out>/<w>_verso (+ conf.tif, thick.tif)\n");
        return 2;
    }
    mallopt(M_MMAP_THRESHOLD, 1 << 20);   /* decoded chunks (2 MB) go back to the system when evicted */
    const double ctum = atof(fopt(argc, argv, "--ct-um", "0"));
    faces_opt o = {0};
    o.range = atof(fopt(argc, argv, "--range", "150")) / ctum;
    double t0 = 15, t1 = 200; if (sscanf(fopt(argc, argv, "--thick", "15,200"), "%lf,%lf", &t0, &t1) != 2 || !(t1 > t0)) { fprintf(stderr, "faces: --thick MIN,MAX (um)\n"); return 2; }
    o.tmin = t0 / ctum; o.tmax = t1 / ctum;
    o.min_prom = atof(fopt(argc, argv, "--min-prom", "8")); o.rel = atof(fopt(argc, argv, "--rel", "0.1"));
    o.tol = atof(fopt(argc, argv, "--tol", "25")) / ctum; o.inset = atof(fopt(argc, argv, "--inset", "0")) / ctum;
    o.smooth = atoi(fopt(argc, argv, "--smooth", "4")); o.nthreads = atoi(fopt(argc, argv, "--threads", "16"));
    const double ct_thr = atof(fopt(argc, argv, "--ct-thr", "90")), cache_mb = atof(fopt(argc, argv, "--cache-mb", "4096"));
    if (!(o.range > 2) || o.smooth > 20 || o.nthreads < 1) { fprintf(stderr, "faces: --range > 2 CT voxels, --smooth <= 20\n"); return 2; }
    const char *cache = fopt(argc, argv, "--cache", nullptr);
    z3 *cz = z3_open(store_open(argv[ci + 1]), argv[ci + 2], cache); if (!cz) { fprintf(stderr, "faces: %s\n", z3_error()); return 1; }
    fc_cache cc; if (fcc_init(&cc, cz, o.nthreads, cache_mb) || fcc_sub(&cc)) { fprintf(stderr, "faces: out of memory\n"); return 1; }
    faces_vol V = {&cc, fcc_sample, fcc_mark, fcc_fill, nullptr, fcc_mark_mask};
    /* --gpu 1 (default): profiles + medians on the GPU (bit-identical; 128^3 CT chunks); 0, no device, or a binary without it: the CPU */
    if (atoi(fopt(argc, argv, "--gpu", "1")) && fgpu_open && cc.cs[0] == 128 && cc.cs[1] == 128 && cc.cs[2] == 128 && (cc.dev = fgpu_open(cc.shape))) V.gpu = cc.dev;
    const int gi = farg(argc, argv, "--gt"); z3 *gz = nullptr; fc_cache gc; const double gum = atof(fopt(argc, argv, "--gt-um", "0"));
    if (gi > 0) {
        if (gi + 2 >= argc || !(gum > 0)) { fprintf(stderr, "faces: --gt ROOT KEY --gt-um U\n"); return 2; }
        gz = z3_open(store_open(argv[gi + 1]), argv[gi + 2], cache); if (!gz || fcc_init(&gc, gz, o.nthreads, cache_mb)) { fprintf(stderr, "faces: --gt %s\n", z3_error()); return 1; }
    }
    axis ax; const char *umb = fopt(argc, argv, "--umb", nullptr); const double umbum = atof(fopt(argc, argv, "--umb-um", "0"));
    if (umb && (axis_load(&ax, umb) || !(umbum > 0))) { fprintf(stderr, "faces: --umb JSON (control_points) --umb-um U\n"); return 2; }
    const char *wl = fopt(argc, argv, "--wraps", nullptr); char *want = nullptr; if (wl) { want = malloc(strlen(wl) + 3); sprintf(want, ",%s,", wl); }
    char cmd[1500]; snprintf(cmd, sizeof cmd, "mkdir -p '%s'", argv[3]); if (system(cmd)) return 1;
    DIR *dd = opendir(argv[2]); if (!dd) { fprintf(stderr, "faces: cannot open %s\n", argv[2]); return 1; }
    int nn = 0, cap = 64; char (*names)[256] = malloc(cap * 256); struct dirent *de;
    while ((de = readdir(dd))) {
        if (de->d_name[0] != 'w' || strlen(de->d_name) >= 250) continue;
        char nm[256]; snprintf(nm, sizeof nm, "%s", de->d_name); { const size_t l = strlen(nm); if (l > 4 && !strcmp(nm + l - 4, ".sfc")) nm[l - 4] = 0; }
        if (want) { char k[300]; snprintf(k, sizeof k, ",%s,", nm); if (!strstr(want, k)) continue; }
        char p[1400]; snprintf(p, sizeof p, "%s/%s/meta.json", argv[2], nm); if (!tfx_exists(p)) continue;
        int dup = 0; for (int q = 0; q < nn; q++) dup |= !strcmp(names[q], nm); if (dup) continue;
        if (nn == cap) { cap *= 2; names = realloc(names, cap * 256); }
        snprintf(names[nn++], 256, "%s", nm);
    }
    closedir(dd); qsort(names, nn, 256, (int (*)(const void *, const void *))strcmp);
    int done = 0;
    printf("wrap verts band pick0 pick fill on_band ct>=%g(orig rect vers) thick_um(p10 p50 p90) shift_um(recto verso) fibre_row/col(recto verso) gt(n |orig| |recto| |verso| p50 um, <=20um orig/recto/verso, inner/mid/outer/inward/outward)\n", ct_thr);
    for (int q = 0; q < nn; q++) {
        const char *nm = names[q]; double wum, sc; int winding; meta_get(argv[2], nm, &wum, &sc, &winding);
        char p[1400]; int w = 0, h = 0, w2, h2, w3, h3;
        snprintf(p, sizeof p, "%s/%s/x.tif", argv[2], nm); float *X = img_read_f32(p, &w, &h);
        snprintf(p, sizeof p, "%s/%s/y.tif", argv[2], nm); float *Y = img_read_f32(p, &w2, &h2);
        snprintf(p, sizeof p, "%s/%s/z.tif", argv[2], nm); float *Z = img_read_f32(p, &w3, &h3);
        if (!X || !Y || !Z || w != w2 || w != w3 || h != h2 || h != h3 || !(wum > 0) || !(sc > 0) || w < 5 || h < 5) { fprintf(stderr, "faces: %s: not a tifxyz wrap\n", nm); free(X); free(Y); free(Z); continue; }
        const double tw = now_s(), f = wum / ctum; const size_t n = (size_t)w * h;
        for (size_t i = 0; i < n; i++) if (!(Z[i] >= 0) || !(X[i] >= 0) || !(Y[i] >= 0)) X[i] = Y[i] = Z[i] = -1;
        double (*axy)[2] = nullptr; int na = 0;
        if (umb) {
            float zm = 0; for (size_t i = 0; i < n; i++) if (Z[i] > zm) zm = Z[i];
            na = (int)zm + 2; axy = malloc(na * sizeof *axy);
            for (int z = 0; z < na; z++) { double y, x; axis_at(&ax, ((z + 0.5) * wum / umbum) - 0.5, &y, &x); axy[z][0] = (y + 0.5) * umbum / wum - 0.5; axy[z][1] = (x + 0.5) * umbum / wum - 0.5; }
        }
        const int sign = faces_orient(X, Y, Z, w, h, (const double (*)[2])axy, na); free(axy);
        float *RX = malloc(n * 4), *RY = malloc(n * 4), *RZ = malloc(n * 4), *VX = malloc(n * 4), *VY = malloc(n * 4), *VZ = malloc(n * 4), *th = malloc(n * 4); uint8_t *cf = malloc(n);
        faces_stats st; memset(&st, 0, sizeof st); st.ct_thr = ct_thr;
        if (!RX || !RY || !RZ || !VX || !VY || !VZ || !th || !cf) { fprintf(stderr, "faces: out of memory\n"); return 1; }
        if (faces_wrap(X, Y, Z, w, h, f, ctum, sign, &o, &V, RX, RY, RZ, VX, VY, VZ, cf, th, &st)) { fprintf(stderr, "faces: %s failed\n", nm); return 1; }
        char od[1500], src[512]; snprintf(src, sizeof src, "ufsm faces (recto = inner face of the sheet under %s, range %g um, thick %g..%g um, inset %g um)", nm, o.range * ctum, t0, t1, o.inset * ctum);
        snprintf(od, sizeof od, "%s/%s_recto", argv[3], nm);
        if (tifxyz_write(od, w, h, RX, RY, RZ, 1 / sc, wum, winding, src)) { fprintf(stderr, "faces: cannot write %s\n", od); return 1; }
        snprintf(p, sizeof p, "%s/conf.tif", od); tiffw_write(p, w, h, 1, cf, 1);
        snprintf(p, sizeof p, "%s/thick.tif", od); tiffw_write(p, w, h, 3, th, 1);
        snprintf(src, sizeof src, "ufsm faces (verso = outer face of the sheet under %s)", nm);
        snprintf(od, sizeof od, "%s/%s_verso", argv[3], nm);
        if (tifxyz_write(od, w, h, VX, VY, VZ, 1 / sc, wum, winding, src)) { fprintf(stderr, "faces: cannot write %s\n", od); return 1; }
        /* thickness percentiles */
        double tot = 0, acc = 0, pc[3] = {NAN, NAN, NAN}; const double qs[3] = {0.1, 0.5, 0.9}; int qk = 0;
        for (int b = 0; b < 64; b++) tot += st.thick_hist[b];
        for (int b = 0; b < 64 && qk < 3; b++) { acc += st.thick_hist[b]; while (qk < 3 && acc >= qs[qk] * tot) pc[qk++] = 5 * b + 2.5; }
        /* the published surfaces */
        char gts[400] = ""; size_t gth[40] = {0};
        const double tgt = now_s();
        if (gz) {
            const int ws = (w + SUB - 1) / SUB, hs = (h + SUB - 1) / SUB; float *dist = malloc((size_t)ws * hs * 4 * sizeof(float));
            for (size_t k = 0; k < (size_t)ws * hs * 4; k++) dist[k] = NAN;
            for (int r0 = 0; r0 < h; r0 += 32) {
                gt_job G = {X, Y, Z, RX, RY, RZ, VX, VY, VZ, w, h, r0, r0 + 32 < h ? r0 + 32 : h, wum / gum, atof(fopt(argc, argv, "--gt-range", "100")) / gum, 0.5, sign, &gc, 1, dist, 0};
                fc_par(gt_rows, &G, o.nthreads); if (fcc_fill(&gc)) return 1;
                G.mark = 0; atomic_store(&G.next, 0); fc_par(gt_rows, &G, o.nthreads);
            }
            size_t ng = 0, c20[3] = {0}, cls[5] = {0}; float *a[3]; for (int k = 0; k < 3; k++) a[k] = malloc((size_t)ws * hs * sizeof(float));
            for (size_t k = 0; k < (size_t)ws * hs; k++) {
                if (isnan(dist[4 * k])) continue;
                for (int e = 0; e < 3; e++) { const float d = fabsf(dist[4 * k + e]) * (float)gum; a[e][ng] = d; c20[e] += d <= 20; }
                const float fr = dist[4 * k + 3]; cls[isnan(fr) || fr > 1.15f ? 4 : fr < -0.15f ? 3 : fr < 1.f / 3 ? 0 : fr < 2.f / 3 ? 1 : 2]++;
                const int hb = (int)floorf(dist[4 * k + 1] * (float)gum / 5) + 20; gth[hb < 0 ? 0 : hb > 39 ? 39 : hb]++;   /* published - recto, 5 um bins */
                ng++;
            }
            double md[3] = {NAN, NAN, NAN};
            for (int e = 0; e < 3 && ng; e++) { qsort(a[e], ng, sizeof(float), cmp_flt); md[e] = a[e][ng / 2]; }
            if (ng) snprintf(gts, sizeof gts, " gt %zu %.1f %.1f %.1f  %.3f %.3f %.3f  %.3f %.3f %.3f %.3f %.3f", ng, md[0], md[1], md[2], (double)c20[0] / ng, (double)c20[1] / ng, (double)c20[2] / ng,
                             (double)cls[0] / ng, (double)cls[1] / ng, (double)cls[2] / ng, (double)cls[3] / ng, (double)cls[4] / ng);
            else snprintf(gts, sizeof gts, " gt 0");
            for (int k = 0; k < 3; k++) free(a[k]);
            free(dist);
        }
        if (getenv("UFSM_FACES_PROF")) {
            size_t np = 0, nb = 0; int ns = 0; if (cc.dev && fgpu_stats) fgpu_stats(cc.dev, &np, &nb, &ns);
            fprintf(stderr, "faces prof: CT fill: decode+upload %.2f evict %.2f s, %zu substreams (%zu again); GT fill: decode %.2f s, %zu substreams (%zu again); gt %.2f s; gpu %s, %zu chunk uploads %.2f GB, %d device slots\n", cc.tdec, cc.tev, (size_t)cc.nsub, (size_t)cc.nresub, gz ? gc.tdec : 0.0, gz ? (size_t)gc.nsub : 0, gz ? (size_t)gc.nresub : 0, now_s() - tgt, cc.dev ? (fgpu_broken(cc.dev) ? "broken" : "on") : "off", np, nb / 1e9, ns);
        }
        const size_t nm1 = st.nv ? st.nv : 1, cn = st.ct_n ? st.ct_n : 1, np = st.nv - st.nfill ? st.nv - st.nfill : 1;
        char line[1200];
        snprintf(line, sizeof line, "%s %zu %.3f %.3f %.3f %.3f %.3f  %.3f %.3f %.3f  %.0f %.0f %.0f  %.1f %.1f  %.3f %.3f%s", nm, st.nv, (double)st.nband / nm1, (double)st.npick0 / nm1, (double)st.npick / nm1, (double)st.nfill / nm1,
                 (double)st.in_band / nm1, (double)st.ct_orig / cn, (double)st.ct_rect / cn, (double)st.ct_vers / cn, pc[0], pc[1], pc[2], st.shift_r / np * wum, st.shift_v / np * wum,
                 st.fib[0][1] > 0 ? st.fib[0][0] / st.fib[0][1] : NAN, st.fib[1][1] > 0 ? st.fib[1][0] / st.fib[1][1] : NAN, gts);
        printf("%s\n", line); fflush(stdout);
        snprintf(p, sizeof p, "%s/%s_faces.txt", argv[3], nm);
        FILE *fo = fopen(p, "w");
        if (fo) {
            fprintf(fo, "# ufsm faces %s: outward sign %d, %dx%d grid\n# wrap verts band pick0 pick fill on_band ct>=%g(orig rect vers) thick_um(p10 p50 p90) shift_um(recto verso) fibre_row/col(recto verso) [gt n |d| p50 um (orig recto verso) <=20um (orig recto verso) frac(inner mid outer inward-of-sheet outward-of-sheet)]\n%s\n# thickness histogram (5 um bins, picked vertices)\n",
                    nm, sign, w, h, ct_thr, line);
            if (gz) { fprintf(fo, "# published surface - recto face along the normal (um, 5 um bins; ends clamp)\n"); for (int b = 0; b < 40; b++) if (gth[b]) fprintf(fo, "gt %d %zu\n", 5 * (b - 20), gth[b]); }
            fprintf(fo, "# bin_um picked nearest_any\n");
            for (int b = 0; b < 64; b++) if (st.thick_hist[b] > 0 || st.thick_all[b] > 0) fprintf(fo, "%d %.0f %.0f\n", 5 * b, st.thick_hist[b], st.thick_all[b]);
            fclose(fo);
        }
        fprintf(stderr, "faces: %s %dx%d sign %+d: %.1f s, %zu CT chunks decoded so far\n", nm, w, h, sign, now_s() - tw, cc.ndec);
        free(X); free(Y); free(Z); free(RX); free(RY); free(RZ); free(VX); free(VY); free(VZ); free(th); free(cf);
        done++;
    }
    if (cc.dev) fgpu_close(cc.dev);
    fcc_free(&cc); z3_close(cz); if (gz) { fcc_free(&gc); z3_close(gz); }
    free(names); free(want);
    fprintf(stderr, "faces: %d wraps -> %s\n", done, argv[3]);
    return done ? 0 : 1;
}
