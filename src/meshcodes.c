/* Winding codes on the fly from q-assigned surface meshes: see meshcodes.h. The stamp below is src/ingest.c raster_shard's --value path
   with the shard box replaced by the read box (the voxel of a sample is floor(p + 0.5) in both, independent of the box), same tile
   order, same sampling density (0.5 voxel along both quad edges at the target level, capped at 256). tests/test_codes_onfly.sh holds the
   two byte-identical. */
#include "meshcodes.h"
#include "reg.h"
#include "store.h"
#include "tiff.h"
#include <dirent.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { int w, h; float *xyz; uint8_t *valid; float *val; } mc_mesh;   /* xyz: level-0 x, y, z per vertex (after the map) */
typedef struct { int mi, r0, r1, c0, c1; float lo[3], hi[3]; } mc_tile;        /* bounds in level-0 z, y, x */
typedef struct { mc_mesh *m; int nm; mc_tile *t; size_t nt; double scale; } mc_ctx;

static double mc_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static inline uint8_t mc_code(double q) { long k = lround(q * 18.0) % 252; if (k < 0) k += 252; return (uint8_t)(1 + k); }
static inline int mc_diff(uint8_t a, uint8_t b) { int d = ((int)a - (int)b) % 252; if (d < -126) d += 252; if (d >= 126) d -= 252; return d; }

static float *mc_tif(const char *dir, const char *name, int *w, int *h) {
    char p[2200]; snprintf(p, sizeof p, "%s/%s", dir, name);
    tiff *t = tiff_open_file(p); if (!t) { fprintf(stderr, "mesh codes: %s: %s\n", p, tiff_error()); return nullptr; }
    tiff_page pg; float *v = nullptr;
    if (tiff_page_info(t, 0, &pg) || pg.w < 2 || pg.h < 2 || pg.bits != 32 || pg.fmt != 3 || pg.spp != 1 || (*w && (pg.w != *w || pg.h != *h)))
        fprintf(stderr, "mesh codes: %s: not a matching 2-D float32 image\n", p);
    else if (!(v = malloc((size_t)pg.w * pg.h * 4)) || tiff_read_page(t, 0, v)) { fprintf(stderr, "mesh codes: %s: TIFF decode failed\n", p); free(v); v = nullptr; }
    else { *w = pg.w; *h = pg.h; }
    tiff_close(t); return v;
}
static int mc_load(mc_mesh *m, const char *dir, const char *value, const rmap *map) {
    float *ax[3] = {mc_tif(dir, "x.tif", &m->w, &m->h), nullptr, nullptr};
    if (ax[0]) ax[1] = mc_tif(dir, "y.tif", &m->w, &m->h);
    if (ax[1]) ax[2] = mc_tif(dir, "z.tif", &m->w, &m->h);
    if (ax[2]) m->val = mc_tif(dir, value, &m->w, &m->h);
    const size_t n = (size_t)m->w * m->h;
    if (m->val) { m->xyz = malloc(n * 12); m->valid = malloc(n); }
    if (!m->xyz || !m->valid) { for (int a = 0; a < 3; a++) free(ax[a]); return -1; }
    for (size_t i = 0; i < n; i++) {   /* (validity on the meshes' own coordinates, as ufsm raster; then the map) */
        m->valid[i] = isfinite(ax[0][i]) && isfinite(ax[1][i]) && isfinite(ax[2][i]) && ax[2][i] > 0 && ax[0][i] >= 0;
        double c[3] = {ax[0][i], ax[1][i], ax[2][i]};
        if (map && m->valid[i]) { double f[3]; rmap_c2f(map, c, f); for (int a = 0; a < 3; a++) c[a] = f[a]; }
        for (int a = 0; a < 3; a++) m->xyz[i * 3 + a] = (float)c[a];
    }
    for (int a = 0; a < 3; a++) free(ax[a]);
    return 0;
}
static int mc_cmp(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }
static void mc_free(void *vp) {
    mc_ctx *X = vp; if (!X) return;
    for (int i = 0; i < X->nm; i++) { free(X->m[i].xyz); free(X->m[i].valid); free(X->m[i].val); }
    free(X->m); free(X->t); free(X);
}

/* one z range [z0, z1) of the read box; every thread walks the tiles in order and writes only its own slices: deterministic */
typedef struct { const mc_ctx *X; const int64_t *o, *n; uint8_t *out; int64_t z0, z1; } mc_part;
static void *mc_stamp(void *vp) {
    const mc_part *P = vp; const mc_ctx *X = P->X; const double sc = X->scale;
    const int64_t *o = P->o, *n = P->n;
    /* the quad / tile cull box: a sample in voxel g has p in [g - 0.5, g + 0.5), so anything outside [lo - 1, hi + 1] cannot hit */
    const double lo[3] = {(double)(o[0] + P->z0) - 1, (double)o[1] - 1, (double)o[2] - 1}, hi[3] = {(double)(o[0] + P->z1), (double)(o[1] + n[1]), (double)(o[2] + n[2])};
    for (size_t ti = 0; ti < X->nt; ti++) {
        const mc_tile *t = &X->t[ti]; int hit = 1;
        for (int d = 0; d < 3; d++) if (t->lo[d] * sc > hi[d] || t->hi[d] * sc < lo[d]) hit = 0;
        if (!hit) continue;
        const mc_mesh *m = &X->m[t->mi];
        for (int r = t->r0; r < t->r1; r++) for (int c = t->c0; c < t->c1; c++) {
            const size_t i00 = (size_t)r * m->w + c, idx[4] = {i00, i00 + 1, i00 + (size_t)m->w, i00 + (size_t)m->w + 1};
            if (!(m->valid[idx[0]] && m->valid[idx[1]] && m->valid[idx[2]] && m->valid[idx[3]])) continue;
            const double qv[4] = {m->val[idx[0]], m->val[idx[1]], m->val[idx[2]], m->val[idx[3]]};
            if (!(isfinite(qv[0]) && isfinite(qv[1]) && isfinite(qv[2]) && isfinite(qv[3]))) continue;
            double p[4][3], cl[3] = {1e30, 1e30, 1e30}, ch[3] = {-1e30, -1e30, -1e30};
            for (int k = 0; k < 4; k++) for (int a = 0; a < 3; a++) p[k][2 - a] = m->xyz[idx[k] * 3 + a] * sc;
            for (int k = 0; k < 4; k++) for (int d = 0; d < 3; d++) { if (p[k][d] < cl[d]) cl[d] = p[k][d]; if (p[k][d] > ch[d]) ch[d] = p[k][d]; }
            int outb = 0; for (int d = 0; d < 3; d++) if (cl[d] > hi[d] || ch[d] < lo[d]) outb = 1;
            if (outb) continue;
            double e1 = 0, e2 = 0;
            for (int d = 0; d < 3; d++) { e1 += (p[1][d] - p[0][d]) * (p[1][d] - p[0][d]); e2 += (p[2][d] - p[0][d]) * (p[2][d] - p[0][d]); }
            int nu = (int)ceil(sqrt(e1) / 0.5) + 1, nv = (int)ceil(sqrt(e2) / 0.5) + 1;
            if (nu > 256) nu = 256;
            if (nv > 256) nv = 256;
            for (int a = 0; a < nu; a++) for (int b = 0; b < nv; b++) {
                const double u = nu > 1 ? (double)a / (nu - 1) : 0, v = nv > 1 ? (double)b / (nv - 1) : 0;
                int64_t g[3]; int in = 1;
                for (int d = 0; d < 3; d++) {
                    const double q = (1 - u) * (1 - v) * p[0][d] + u * (1 - v) * p[1][d] + (1 - u) * v * p[2][d] + u * v * p[3][d];
                    g[d] = (int64_t)floor(q + 0.5) - o[d];
                    if (g[d] < (d ? 0 : P->z0) || g[d] >= (d ? n[d] : P->z1)) in = 0;
                }
                if (!in) continue;
                uint8_t *dv = &P->out[((size_t)g[0] * n[1] + g[1]) * n[2] + g[2]];
                const uint8_t code = mc_code((1 - u) * (1 - v) * qv[0] + u * (1 - v) * qv[1] + (1 - u) * v * qv[2] + u * v * qv[3]);
                if (*dv == 0) *dv = code;
                else if (*dv != 255 && abs(mc_diff(*dv, code)) > 4) *dv = 255;
            }
        }
    }
    return nullptr;
}
static int mc_read(void *vp, z3 *base, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads) {
    (void)base; const mc_ctx *X = vp;
    memset(out, 0, (size_t)n[0] * n[1] * n[2]);
    int T = nthreads < 1 ? 1 : nthreads > 64 ? 64 : nthreads; if (T > n[0]) T = (int)n[0];
    mc_part P[64]; pthread_t th[64];
    for (int i = 0; i < T; i++) P[i] = (mc_part){X, o, n, out, n[0] * i / T, n[0] * (i + 1) / T};
    for (int i = 1; i < T; i++) if (pthread_create(&th[i], nullptr, mc_stamp, &P[i])) { mc_stamp(&P[i]); th[i] = 0; }
    mc_stamp(&P[0]);
    for (int i = 1; i < T; i++) if (th[i]) pthread_join(th[i], nullptr);
    return 0;
}
static int mc_extent(void *vp, const int64_t grid[3], const int chunk[3], uint8_t *bits) {
    const mc_ctx *X = vp;
    for (size_t ti = 0; ti < X->nt; ti++) { const mc_tile *t = &X->t[ti]; int64_t a[3], b[3]; int ok = 1;
        for (int d = 0; d < 3; d++) { a[d] = (int64_t)floor((t->lo[d] * X->scale - 1) / chunk[d]); b[d] = (int64_t)floor((t->hi[d] * X->scale + 1) / chunk[d]);
            if (a[d] < 0) a[d] = 0;
            if (b[d] > grid[d] - 1) b[d] = grid[d] - 1;
            if (b[d] < a[d]) ok = 0; }
        if (ok) for (int64_t z = a[0]; z <= b[0]; z++) for (int64_t y = a[1]; y <= b[1]; y++) for (int64_t x = a[2]; x <= b[2]; x++) bits[(z * grid[1] + y) * grid[2] + x] = 1; }
    return 0;
}

z3 *mesh_codes_open(const char *dir, const char *value, const char *mapp, int level, z3 *ct_level, int gpu) {
    (void)gpu;
    if (!ct_level || !dir || !value || level < 0 || level > 8) { fprintf(stderr, "mesh codes: need a meshes dir, a value image name, level 0..8 and the CT level\n"); z3_close(ct_level); return nullptr; }
    const rmap *map = nullptr; if (mapp && *mapp && !(map = reg_map_load(mapp))) { fprintf(stderr, "mesh codes: cannot load the map %s\n", mapp); z3_close(ct_level); return nullptr; }
    char **names = nullptr; int nn = 0, cap = 0;
    DIR *D = opendir(dir); if (!D) { fprintf(stderr, "mesh codes: cannot open %s\n", dir); z3_close(ct_level); return nullptr; }
    for (struct dirent *e; (e = readdir(D));) { const size_t l = strlen(e->d_name);
        if (l < 8 || strcmp(e->d_name + l - 7, ".tifxyz")) continue;
        if (nn == cap) { cap = cap ? cap * 2 : 64; names = realloc(names, (size_t)cap * sizeof *names); }
        names[nn++] = strdup(e->d_name); }
    closedir(D);
    if (!nn) { fprintf(stderr, "mesh codes: no *.tifxyz in %s\n", dir); free(names); z3_close(ct_level); return nullptr; }
    qsort(names, (size_t)nn, sizeof *names, mc_cmp);   /* (the order of the shell glob ufsm raster was given) */
    mc_ctx *X = calloc(1, sizeof *X); X->m = calloc((size_t)nn, sizeof *X->m); X->scale = 1.0 / (1 << level);
    int bad = 0; size_t tcap = 0;
    for (int i = 0; i < nn && !bad; i++) { char p[2200]; snprintf(p, sizeof p, "%s/%s", dir, names[i]); bad = mc_load(&X->m[i], p, value, map); X->nm = i + 1; }
    for (int i = 0; i < nn; i++) free(names[i]);
    free(names);
    for (int mi = 0; mi < X->nm && !bad; mi++) { const mc_mesh *m = &X->m[mi];   /* the 32 x 32 tiles of ufsm raster's index, in its order */
        for (int r0 = 0; r0 + 1 < m->h; r0 += 32) for (int c0 = 0; c0 + 1 < m->w; c0 += 32) {
            mc_tile t = {.mi = mi, .r0 = r0, .c0 = c0, .r1 = r0 + 32, .c1 = c0 + 32, .lo = {1e30f, 1e30f, 1e30f}, .hi = {-1e30f, -1e30f, -1e30f}};
            if (t.r1 >= m->h) t.r1 = m->h - 1;
            if (t.c1 >= m->w) t.c1 = m->w - 1;
            int any = 0;
            for (int r = r0; r <= t.r1; r++) for (int c = c0; c <= t.c1; c++) { const size_t k = (size_t)r * m->w + c; if (!m->valid[k]) continue;
                any = 1;
                for (int a = 0; a < 3; a++) { const int d = 2 - a; const float v = m->xyz[k * 3 + a]; if (v < t.lo[d]) t.lo[d] = v; if (v > t.hi[d]) t.hi[d] = v; } }
            if (!any) continue;
            if (X->nt == tcap) { tcap = tcap ? tcap * 2 : 1024; X->t = realloc(X->t, tcap * sizeof *X->t); }
            X->t[X->nt++] = t;
        } }
    if (bad) { mc_free(X); z3_close(ct_level); return nullptr; }
    z3 *z = z3_open_derived(ct_level, 0, mc_read, X, mc_free);
    if (!z) { mc_free(X); z3_close(ct_level); return nullptr; }
    z3_derived_extent(z, mc_extent, X);
    return z;
}

static const char *mc_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 3; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
/* ufsm codes-onfly <meshes-dir> --ct ROOT KEY [--level 0] [--value q.tif] [--map reg.json] [--ref ROOT KEY] [--windows 8] [--P 256] [--threads 8]:
   the derived codes over windows drawn on the meshes (random tiles), ms per window, and the byte differences from a stored codes raster
   (`ufsm raster --value` at that level). KEY is the CT array of the level (its meta gives the grid). Exit 1 when a byte differs. */
int cmd_codes_onfly(int argc, char **argv) {
    int ci = 0, ri = 0; for (int i = 3; i < argc; i++) { if (!strcmp(argv[i], "--ct") && i + 2 < argc) ci = i; if (!strcmp(argv[i], "--ref") && i + 2 < argc) ri = i; }
    if (argc < 3 || !ci) { fprintf(stderr, "usage: ufsm codes-onfly <meshes-dir> --ct ROOT KEY [--level 0] [--value q.tif] [--map reg.json] [--ref ROOT KEY] [--windows 8] [--P 256] [--threads 8]\n"); return 2; }
    const int level = atoi(mc_opt(argc, argv, "--level", "0")), nw = atoi(mc_opt(argc, argv, "--windows", "8")), P = atoi(mc_opt(argc, argv, "--P", "256")), T = atoi(mc_opt(argc, argv, "--threads", "8"));
    uint64_t seed = (uint64_t)atoll(mc_opt(argc, argv, "--seed", "1"));
    store *cs = store_open(argv[ci + 1]); z3 *c0 = cs ? z3_open(cs, argv[ci + 2], nullptr) : nullptr;
    if (!c0) { fprintf(stderr, "codes-onfly: cannot open the CT %s %s\n", argv[ci + 1], argv[ci + 2]); return 1; }
    const double t0 = mc_now();
    z3 *z = mesh_codes_open(argv[2], mc_opt(argc, argv, "--value", "q.tif"), mc_opt(argc, argv, "--map", nullptr), level, c0, 0); if (!z) return 1;
    const mc_ctx *X = z3_derived_ctx(z); const int64_t *sh = z3_meta_of(z)->shape;
    printf("codes-onfly: %d meshes, %zu tiles, opened in %.1f s (grid %lld %lld %lld)\n", X->nm, X->nt, mc_now() - t0, (long long)sh[0], (long long)sh[1], (long long)sh[2]);
    { int64_t g[3]; uint8_t *bits = nullptr; const int64_t nk = z3_chunk_map(z, g, &bits); free(bits);
      printf("codes-onfly: the extent: %lld of %lld chunks (%.1f%%)\n", (long long)nk, (long long)(g[0] * g[1] * g[2]), 100.0 * nk / (double)(g[0] * g[1] * g[2])); }
    z3 *ref = nullptr; if (ri) { store *rs = store_open(argv[ri + 1]); ref = rs ? z3_open(rs, argv[ri + 2], nullptr) : nullptr; if (!ref) { fprintf(stderr, "codes-onfly: cannot open %s %s\n", argv[ri + 1], argv[ri + 2]); return 1; } }
    const size_t nv = (size_t)P * P * P; uint8_t *a = malloc(nv), *b = malloc(nv); long long bad = 0; double tsum = 0, tmax = 0;
    for (int w = 0; w < nw; w++) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull; const mc_tile *t = &X->t[(seed >> 33) % X->nt];
        int64_t o[3], n[3] = {P, P, P};
        for (int d = 0; d < 3; d++) { seed = seed * 6364136223846793005ull + 1442695040888963407ull;
            o[d] = (int64_t)(0.5 * (t->lo[d] + t->hi[d]) * X->scale) - P / 2 + (int64_t)((seed >> 33) % 33) - 16;   /* (not chunk aligned) */
            if (o[d] > sh[d] - P) o[d] = sh[d] - P;
            if (o[d] < 0) o[d] = 0; }
        const double t1 = mc_now(); if (z3_read(z, o, n, a, T)) { fprintf(stderr, "codes-onfly: read failed\n"); return 1; } const double dt = mc_now() - t1;
        tsum += dt; if (dt > tmax) tmax = dt;
        long long lab = 0, cf = 0, d = 0; for (size_t i = 0; i < nv; i++) { lab += a[i] != 0; cf += a[i] == 255; }
        if (ref) { if (z3_read(ref, o, n, b, 8)) { fprintf(stderr, "codes-onfly: ref read failed: %s\n", z3_error()); return 1; } for (size_t i = 0; i < nv; i++) d += a[i] != b[i]; }
        bad += d;
        printf("window %d (%lld,%lld,%lld): %.0f ms, %lld coded voxels (%lld conflict)", w, (long long)o[0], (long long)o[1], (long long)o[2], dt * 1e3, lab, cf);
        if (ref) printf(", %lld bytes differ", d);
        printf("\n"); fflush(stdout);
    }
    printf("codes-onfly: %d windows of %d^3: %.0f ms mean, %.0f ms max per window (CPU, %d threads)%s\n", nw, P, tsum / nw * 1e3, tmax * 1e3, T,
           ref ? (bad ? ": DIFFERS from the stored raster" : ": byte-identical to the stored raster") : "");
    free(a); free(b); z3_close(z); z3_close(ref);
    return bad ? 1 : 0;
}
