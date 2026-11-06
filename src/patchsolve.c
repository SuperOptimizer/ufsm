/* ufsm patch-solve: global turn numbers for traced sheet patches in one box (unroll v3, P4; C port of scripts/v3/proto/psolve4.py).
   ufsm patch-solve --paths DIR --prob P --box Z0,Y0,X0,NZ,NY,NX --axis AXIS.json [--axis-scale 0.25] [--sgn -1]
                    [--prob-key K] [--gt G [--gt-key K]] [--cache DIR] [--minv 100] [--votes pair|ray] [--counter both|ridge|prob]
                    [--min-votes 3] [--prune-frac 0.05] [--prune-thr 0.25] [--rounds 6] [--threads N] [--tsv OUT]
   Inputs (box-local voxel coordinates on the prob grid):
     --paths  a directory of tifxyz patches (VC3D paths/<name>/{x,y,z}.tif, -1 = invalid), patch coordinates relative to the box origin;
     --prob   the sheet probability (uint8) of the box: a .raw file (NZ x NY x NX, C order) or a zarr store root with --prob-key the array;
     --axis   umbilicus JSON (control_points {x,y,z}); --axis-scale converts its voxels to the prob grid's (1667: 2.4 um -> 9.6 um = 0.25);
     --gt     optional winding_mod14 code raster of the box (codes 1 + round(18 q) mod 252; .raw or zarr root + --gt-key) for the score;
              --sgn is the GT spiral sense (theta direction of increasing q: 1667 GT = -1).
   Steps:
     1. ridge: radial non-max suppression of the prob (p >= 64, p >= p(r + 1), p > p(r - 1) along the direction from the axis, bilinear);
     2. nodes: each patch cut at vertices with no ridge within 1 voxel (6-neighbour) into 4-connected grid components >= --minv vertices;
     3. rays: vertices binned by (z / 4, arc / 4 voxels at the median radius), sorted by r; consecutive crossings a -> b of different nodes
        (>= 3 voxels apart) vote turn_b - turn_a = 1 + (sheets strictly between), counted as ridge runs and as prob peaks of prominence
        >= 32 (scipy find_peaks semantics) on the ray; --counter both pools the two votes;
     4. edges: --votes pair = the per directed pair mode (>= --min-votes votes; weight = the mode's count), as the prototype;
               --votes ray  = every vote of pairs with >= --min-votes votes, merged by value (the L1 solve weighs them all);
     5. solve: exact weighted L1 integer potentials (src/l1pot.h, PUMA graph cuts) instead of the prototype's LS + median sweeps;
     6. prune: up to --rounds times drop the nodes whose violated-edge weight share is >= max(--prune-thr, the top --prune-frac quantile)
        and re-solve, until no node reaches --prune-thr.
   Output: per-round summary on stdout (with --gt: share of GT-matched area on the right sheet and turn, overall and in the largest graph
   component, mod 14 like the prototype); --tsv: one row per node (patch, comp, vertices, active, graph comp, turn, GT sheet / pure / n). */
#include "l1pot.h"
#include "patchsolve.h"
#include "sources.h"
#include "store.h"
#include "tiff.h"
#include "tiffw.h"
#include "zarr3.h"
#include <dirent.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *ps_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }

float *ps_tif(const char *path, int *w, int *h) {
    tiff *t = tiff_open_file(path); tiff_page pg;
    if (!t || tiff_page_info(t, 0, &pg) || pg.fmt != 3 || pg.bits != 32 || pg.spp != 1) { if (t) tiff_close(t); return nullptr; }
    float *d = malloc((size_t)pg.w * pg.h * 4);
    if (!d || tiff_read_page(t, 0, d)) { free(d); tiff_close(t); return nullptr; }
    *w = pg.w; *h = pg.h; tiff_close(t); return d;
}

/* a uint8 box: a .raw file of n voxels, or a zarr array (root + key) read at the box origin */
uint8_t *ps_box(const char *path, const char *key, const char *cache, const int64_t o[3], const int64_t n[3], int nt) {
    const size_t N = (size_t)(n[0] * n[1] * n[2]); uint8_t *b = malloc(N); if (!b) return nullptr;
    const size_t L = strlen(path);
    if (L > 4 && !strcmp(path + L - 4, ".raw")) {
        FILE *f = fopen(path, "rb"); const size_t got = f ? fread(b, 1, N, f) : 0; if (f) fclose(f);
        if (got != N) { fprintf(stderr, "patch-solve: %s: expected %zu bytes\n", path, N); free(b); return nullptr; }
        return b;
    }
    if (!key) { fprintf(stderr, "patch-solve: %s is not a .raw file; give the array key\n", path); free(b); return nullptr; }
    store *s = store_open(path); z3 *z = s ? z3_open(s, key, cache) : nullptr;
    if (!z || z3_read(z, o, n, b, nt)) { fprintf(stderr, "patch-solve: cannot read %s %s\n", path, key); if (z) z3_close(z); free(b); return nullptr; }
    z3_close(z); return b;
}

/* ---- geometry ---- */
void ps_axis(const ps_geo *G, double z, double *ay, double *ax) {   /* box-local axis (y, x) at box-local z */
    double y, x; axis_at(&G->ax, (z + (double)G->o[0]) / G->s, &y, &x);
    *ay = y * G->s - (double)G->o[1]; *ax = x * G->s - (double)G->o[2];
}
static inline double ps_wrap(double d) { double r = remainder(d, 2 * M_PI); return r <= -M_PI ? r + 2 * M_PI : r; }   /* (-pi, pi] */
int ps_geo_init(ps_geo *G, const char *axis_json, double scale, const int64_t o[3], const int64_t n[3]) {
    memset(G, 0, sizeof *G); G->s = scale; for (int d = 0; d < 3; d++) G->o[d] = o[d];
    if (axis_load(&G->ax, axis_json)) return -1;
    double ay, ax; ps_axis(G, n[0] / 2.0, &ay, &ax); G->t0 = atan2(n[1] / 2.0 - ay, n[2] / 2.0 - ax); return 0;
}
void ps_frame(const ps_geo *G, double z, double y, double x, double *r, double *u) {
    if (!G->par) { ps_polar(G, z, y, x, r, u); return; }
    const double dy = y - G->cy, dx = x - G->cx; *r = dy * G->ny + dx * G->nx; *u = dy * G->ty + dx * G->tx;
}
void ps_frame_pt(const ps_geo *G, double z, double r, double u, double *y, double *x) {
    if (!G->par) { double ay, ax; ps_axis(G, z, &ay, &ax); *y = ay + r * sin(u); *x = ax + r * cos(u); return; }
    *y = G->cy + r * G->ny + u * G->ty; *x = G->cx + r * G->nx + u * G->tx;
}
void ps_ray_dir(const ps_geo *G, double z, double y, double x, double *uy, double *ux) {
    if (G->par) { *uy = G->ny; *ux = G->nx; return; }
    double ay, ax; ps_axis(G, z, &ay, &ax); const double dy = y - ay, dx = x - ax, r = sqrt(dy * dy + dx * dx) + 1e-6; *uy = dy / r; *ux = dx / r;
}
int ps_frame_load(ps_geo *G, const char *path) {
    FILE *f = fopen(path, "r"); if (!f) return -1; char k[16] = ""; int rc = 0;
    if (fscanf(f, "%15s", k) != 1) rc = -1;
    else if (!strcmp(k, "radial")) G->par = 0;
    else if (!strcmp(k, "par") && fscanf(f, "%lf %lf %lf %lf", &G->ny, &G->nx, &G->cy, &G->cx) == 4) { G->par = 1; G->ty = G->nx; G->tx = -G->ny; }
    else rc = -1;
    fclose(f); return rc;
}
int ps_frame_save(const ps_geo *G, const char *path) {
    FILE *f = fopen(path, "w"); if (!f) return -1;
    if (G->par) fprintf(f, "par %.9f %.9f %.3f %.3f\n", G->ny, G->nx, G->cy, G->cx); else fprintf(f, "radial\n");
    return fclose(f) ? -1 : 0;
}
void ps_polar(const ps_geo *G, double z, double y, double x, double *r, double *th) {
    double ay, ax; ps_axis(G, z, &ay, &ax);
    *r = hypot(y - ay, x - ax); *th = G->t0 + ps_wrap(atan2(y - ay, x - ax) - G->t0);
}

/* ---- ridge (radial non-max suppression), threaded over z ---- */
typedef struct { const uint8_t *p; uint8_t *rg; int64_t n[3]; const ps_geo *G; int z0, z1; } ps_rjob;
static inline double ps_bil(const uint8_t *s, int64_t ny, int64_t nx, double y, double x) {   /* map_coordinates order 1, mode nearest */
    y = y < 0 ? 0 : y > ny - 1 ? (double)(ny - 1) : y; x = x < 0 ? 0 : x > nx - 1 ? (double)(nx - 1) : x;
    int64_t y0 = (int64_t)floor(y), x0 = (int64_t)floor(x); const double fy = y - y0, fx = x - x0;
    const int64_t y1 = y0 + 1 < ny ? y0 + 1 : y0, x1 = x0 + 1 < nx ? x0 + 1 : x0;
    return (1 - fy) * ((1 - fx) * s[y0 * nx + x0] + fx * s[y0 * nx + x1]) + fy * ((1 - fx) * s[y1 * nx + x0] + fx * s[y1 * nx + x1]);
}
static void *ps_ridge_w(void *a) {
    ps_rjob *J = a; const int64_t ny = J->n[1], nx = J->n[2];
    for (int z = J->z0; z < J->z1; z++) {
        const uint8_t *s = J->p + (size_t)z * ny * nx; uint8_t *o = J->rg + (size_t)z * ny * nx;
        for (int64_t y = 0; y < ny; y++) for (int64_t x = 0; x < nx; x++) {
            const uint8_t v = s[y * nx + x]; if (v < 64) { o[y * nx + x] = 0; continue; }
            double uy, ux; ps_ray_dir(J->G, z, (double)y, (double)x, &uy, &ux);
            const double pa = ps_bil(s, ny, nx, y + uy, x + ux), pb = ps_bil(s, ny, nx, y - uy, x - ux);
            o[y * nx + x] = v >= pa && v > pb;
        }
    }
    return nullptr;
}
void ps_ridge(const uint8_t *p, uint8_t *rg, const int64_t n[3], const ps_geo *G, int nt) {
    nt = nt < 1 ? 1 : nt > 64 ? 64 : nt;
    pthread_t th[64]; ps_rjob J[64];
    for (int t = 0; t < nt; t++) { J[t] = (ps_rjob){p, rg, {n[0], n[1], n[2]}, G, (int)(n[0] * t / nt), (int)(n[0] * (t + 1) / nt)}; pthread_create(&th[t], nullptr, ps_ridge_w, &J[t]); }
    for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
}

/* ---- scipy.signal.find_peaks(x, prominence=pmin): local maxima (plateau midpoints) with prominence >= pmin ---- */
static int ps_peaks(const double *x, int n, double pmin, int *pk) {
    int np_ = 0, i = 1;
    while (i < n - 1) {
        if (x[i - 1] < x[i]) {
            int ia = i + 1; while (ia < n - 1 && x[ia] == x[i]) ia++;
            if (x[ia] < x[i]) {
                const int p = (i + ia - 1) / 2;
                double lmin = x[p], rmin = x[p];
                for (int j = p; j >= 0 && x[j] <= x[p]; j--) if (x[j] < lmin) lmin = x[j];
                for (int j = p; j < n && x[j] <= x[p]; j++) if (x[j] < rmin) rmin = x[j];
                if (x[p] - (lmin > rmin ? lmin : rmin) >= pmin) pk[np_++] = p;
                i = ia;
            }
        }
        i++;
    }
    return np_;
}

/* ---- nodes ---- */
typedef struct { char name[256]; int comp; size_t v0, nv; int r0, r1, c0, c1; double step; } ps_node;   /* + grid bbox [r0, r1] x [c0, c1], grid step */   /* vertices [v0, v0 + nv) of the vertex arrays */
typedef struct { double *z, *y, *x; size_t n, cap; } ps_verts;
static int ps_vpush(ps_verts *V, double z, double y, double x) {
    if (V->n == V->cap) {
        V->cap = V->cap ? V->cap * 2 : 1 << 20;
        double *a = realloc(V->z, V->cap * 8), *b = realloc(V->y, V->cap * 8), *c = realloc(V->x, V->cap * 8);
        if (a) V->z = a;
        if (b) V->y = b;
        if (c) V->x = c;
        if (!a || !b || !c) return -1;
    }
    V->z[V->n] = z; V->y[V->n] = y; V->x[V->n] = x; V->n++; return 0;
}
typedef struct { int32_t *rc; size_t n, cap; } ps_grid;   /* per vertex grid (row, col) */
static int ps_gpush(ps_grid *g, int32_t r, int32_t c) {
    if (g->n == g->cap) { g->cap = g->cap ? g->cap * 2 : 1 << 20; int32_t *t = realloc(g->rc, g->cap * 8); if (!t) return -1; g->rc = t; }
    g->rc[2 * g->n] = r; g->rc[2 * g->n + 1] = c; g->n++; return 0;
}
static int ps_namecmp(const void *a, const void *b) { return strcmp((const char *)a, (const char *)b); }

/* raw votes and sorting */
typedef struct { uint32_t a, b; int32_t d; uint32_t seq; } ps_rv;
static int ps_rvcmp(const void *p, const void *q) {
    const ps_rv *u = p, *v = q;
    if (u->a != v->a) return u->a < v->a ? -1 : 1;
    if (u->b != v->b) return u->b < v->b ? -1 : 1;
    if (u->d != v->d) return u->d < v->d ? -1 : 1;
    return (u->seq > v->seq) - (u->seq < v->seq);
}
typedef struct { uint64_t key; double r; uint32_t c; uint32_t i; } ps_ent;
static int ps_entcmp(const void *p, const void *q) {
    const ps_ent *u = p, *v = q;
    if (u->key != v->key) return u->key < v->key ? -1 : 1;
    if (u->r != v->r) return u->r < v->r ? -1 : 1;
    return (u->i > v->i) - (u->i < v->i);
}
static int ps_dcmp(const void *p, const void *q) { const double a = *(const double *)p, b = *(const double *)q; return (a > b) - (a < b); }

/* union-find */
static uint32_t ps_find(uint32_t *P, uint32_t i) { while (P[i] != i) { P[i] = P[P[i]]; i = P[i]; } return i; }

typedef struct { int gt, pure, tot; } ps_gtd;   /* per node: dominant GT sheet (mod 14; -1 none), its count, the matched count */

/* the prototype's score(sign): (share right overall, share right in the largest graph comp, that comp's share of the matched area) */
static void ps_score(uint32_t N, const int32_t *x, const uint32_t *lab, uint32_t ncc, const uint8_t *active, const ps_gtd *gd, int sign, double out[3]) {
    uint32_t *cnt = calloc(ncc, 4); for (uint32_t i = 0; i < N; i++) cnt[lab[i]]++;
    uint32_t big = 0; for (uint32_t g = 1; g < ncc; g++) if (cnt[g] > cnt[big]) big = g;
    double *h = calloc((size_t)ncc * 14, 8), *gp = calloc((size_t)ncc * 14, 8), *tt = calloc(ncc, 8);
    for (uint32_t i = 0; i < N; i++) {
        if (gd[i].gt < 0 || !active[i]) continue;
        const int off = (((sign * x[i] - gd[i].gt) % 14) + 14) % 14;
        h[(size_t)lab[i] * 14 + off] += gd[i].tot; gp[(size_t)lab[i] * 14 + off] += gd[i].pure; tt[lab[i]] += gd[i].tot;
    }
    double tot = 0, good = 0, bt = 0, bg = 0;
    for (uint32_t g = 0; g < ncc; g++) {
        if (tt[g] <= 0) continue;
        int m = 0; for (int k = 1; k < 14; k++) if (h[(size_t)g * 14 + k] > h[(size_t)g * 14 + m]) m = k;
        tot += tt[g]; good += gp[(size_t)g * 14 + m];
        if (g == big) { bt = tt[g]; bg = gp[(size_t)g * 14 + m]; }
    }
    out[0] = tot > 0 ? good / tot : 0; out[1] = bg / (bt > 1 ? bt : 1); out[2] = tot > 0 ? bt / tot : 0;
    free(cnt); free(h); free(gp); free(tt);
}

int cmd_patch_solve(int argc, char **argv) {
    const char *paths = ps_opt(argc, argv, "--paths", nullptr), *probp = ps_opt(argc, argv, "--prob", nullptr), *boxs = ps_opt(argc, argv, "--box", nullptr);
    const char *axp = ps_opt(argc, argv, "--axis", nullptr), *gtp = ps_opt(argc, argv, "--gt", nullptr), *tsv = ps_opt(argc, argv, "--tsv", nullptr);
    const char *cache = ps_opt(argc, argv, "--cache", nullptr), *vmode = ps_opt(argc, argv, "--votes", "pair"), *cmode = ps_opt(argc, argv, "--counter", "both");
    const double axs = atof(ps_opt(argc, argv, "--axis-scale", "1")), sgn = atof(ps_opt(argc, argv, "--sgn", "1"));
    const int minv = atoi(ps_opt(argc, argv, "--minv", "100")), minvotes = atoi(ps_opt(argc, argv, "--min-votes", "3")), rounds = atoi(ps_opt(argc, argv, "--rounds", "6"));
    const double pfrac = atof(ps_opt(argc, argv, "--prune-frac", "0.05")), pthr = atof(ps_opt(argc, argv, "--prune-thr", "0.25"));
    int nt = atoi(ps_opt(argc, argv, "--threads", "0")); if (nt <= 0) nt = (int)sysconf(_SC_NPROCESSORS_ONLN); if (nt < 1) nt = 1;
    const int oracle = atoi(ps_opt(argc, argv, "--oracle", "0")), cut = atoi(ps_opt(argc, argv, "--cut", "1"));
    const char *linksp = ps_opt(argc, argv, "--links", nullptr); const double linkw = atof(ps_opt(argc, argv, "--link-w", "1"));
    const double dupf = atof(ps_opt(argc, argv, "--dup-frac", "0"));   /* > 0: crossings closer than dupf x pitch with no sheet between vote d = 0 */
    const char *lkinds = ps_opt(argc, argv, "--link-kinds", "seam,rseam");   /* sheet-trace link kinds used (contacts are unreliable: 11-06 C3 58% wrong) */   /* diagnostic: GT turn differences as the edge values */
    const int ray = !strcmp(vmode, "ray");
    /* --counter: a comma list of ridge (ridge runs), prob (prob peaks), pitch (round(dr / pitch), pitch = the box median dr of crossings
       with no sheet between by both counters), maxrp (max of ridge and pitch); both = ridge,prob */
    const int use_r = !strcmp(cmode, "both") || strstr(cmode, "ridge") != nullptr, use_p = !strcmp(cmode, "both") || strstr(cmode, "prob") != nullptr;
    const int use_pt = strstr(cmode, "pitch") != nullptr, use_mx = strstr(cmode, "maxrp") != nullptr;
    long long bx[6];
    if (!paths || !probp || !boxs || !axp || sscanf(boxs, "%lld,%lld,%lld,%lld,%lld,%lld", &bx[0], &bx[1], &bx[2], &bx[3], &bx[4], &bx[5]) != 6 || axs <= 0 ||
        (strcmp(vmode, "pair") && !ray) || !(use_r || use_p || use_pt || use_mx)) {
        fprintf(stderr, "usage: ufsm patch-solve --paths DIR --prob P [--prob-key K] --box Z0,Y0,X0,NZ,NY,NX --axis AXIS.json [--axis-scale S] [--sgn +-1]\n"
                        "         [--gt G [--gt-key K]] [--minv 100] [--votes pair|ray] [--counter both|ridge|prob] [--min-votes 3] [--prune-frac 0.05]\n"
                        "         [--prune-thr 0.25] [--rounds 6] [--threads N] [--tsv OUT] [--cache DIR]\n");
        return 2;
    }
    const int64_t o[3] = {bx[0], bx[1], bx[2]}, n[3] = {bx[3], bx[4], bx[5]}; const int64_t NY = n[1], NX = n[2];
    ps_geo G; if (ps_geo_init(&G, axp, axs, o, n)) { fprintf(stderr, "patch-solve: cannot load axis %s\n", axp); return 1; }
    {   const char *fr = ps_opt(argc, argv, "--frame", nullptr);   /* sheet-trace frame.txt: the ray frame the patches were traced in */
        if (fr && ps_frame_load(&G, fr)) { fprintf(stderr, "patch-solve: cannot read frame %s\n", fr); return 1; }
        if (G.par) printf("frame: parallel rays along (%.3f, %.3f)\n", G.ny, G.nx); }
    uint8_t *prob = ps_box(probp, ps_opt(argc, argv, "--prob-key", nullptr), cache, o, n, nt); if (!prob) return 1;
    uint8_t *gt = gtp ? ps_box(gtp, ps_opt(argc, argv, "--gt-key", nullptr), cache, o, n, nt) : nullptr; if (gtp && !gt) return 1;
    const size_t NN = (size_t)(n[0] * n[1] * n[2]);
    uint8_t *rg = malloc(NN); if (!rg) return 1;
    ps_ridge(prob, rg, n, &G, nt);
    { size_t c = 0; for (size_t i = 0; i < NN; i++) c += rg[i]; printf("ridge fill %.4f\n", (double)c / NN); }
#define IDX(z, y, x) (((size_t)(z) * NY + (size_t)(y)) * NX + (size_t)(x))
#define INB(z, y, x) ((z) >= 0 && (y) >= 0 && (x) >= 0 && (z) < n[0] && (y) < NY && (x) < NX)

    /* ---- patches -> cut components ---- */
    DIR *dd = opendir(paths); if (!dd) { fprintf(stderr, "patch-solve: cannot open %s\n", paths); return 1; }
    char (*names)[256] = nullptr; int nn = 0, cn = 0; struct dirent *de;
    while ((de = readdir(dd))) {
        if (de->d_name[0] == '.' || strlen(de->d_name) >= 256) continue;
        char p[1400]; snprintf(p, sizeof p, "%s/%s/x.tif", paths, de->d_name); if (access(p, F_OK)) continue;
        if (nn == cn) { cn = cn ? cn * 2 : 256; names = realloc(names, (size_t)cn * 256); }
        snprintf(names[nn++], 256, "%s", de->d_name);
    }
    closedir(dd); qsort(names, (size_t)nn, 256, ps_namecmp);
    ps_node *nodes = nullptr; uint32_t N = 0, ncap = 0; ps_verts V = {0}; ps_grid GI = {0};
    for (int pi = 0; pi < nn; pi++) {
        char p[1400]; int w[3], h[3]; float *A[3];
        for (int c = 0; c < 3; c++) { snprintf(p, sizeof p, "%s/%s/%c.tif", paths, names[pi], "xyz"[c]); A[c] = ps_tif(p, &w[c], &h[c]); }
        if (!A[0] || !A[1] || !A[2] || w[0] != w[1] || w[0] != w[2] || h[0] != h[1] || h[0] != h[2]) {
            fprintf(stderr, "patch-solve: skipping unreadable patch %s\n", names[pi]); free(A[0]); free(A[1]); free(A[2]); continue;
        }
        const int W = w[0], H = h[0]; const size_t M = (size_t)W * H;
        double step = 1;   /* the grid step from meta.json scale (tifxyz_write's step) */
        {   snprintf(p, sizeof p, "%s/%s/meta.json", paths, names[pi]); FILE *mf = fopen(p, "r"); char buf[65536];
            const size_t L = mf ? fread(buf, 1, sizeof buf - 1, mf) : 0; if (mf) fclose(mf); buf[L] = 0;
            const char *sc = strstr(buf, "\"scale\""); if (sc && (sc = strchr(sc, '['))) { const double v = atof(sc + 1); if (v > 0) step = 1 / v; } }
        uint8_t *k = calloc(M, 1); int32_t *lab = malloc(M * 4); uint32_t *stk = malloc(M * 4);
        for (size_t i = 0; i < M; i++) {
            const float X = A[0][i], Y = A[1][i], Z = A[2][i]; if (!(X > -1)) continue;
            const double xi = nearbyint(X), yi = nearbyint(Y), zi = nearbyint(Z);
            if (!INB(zi, yi, xi)) continue;
            const int64_t z = (int64_t)zi, y = (int64_t)yi, x = (int64_t)xi;
            int s = rg[IDX(z, y, x)];   /* binary_dilation(ridge, iterations=1): the 6-neighbour cross */
            if (!s) s = (z > 0 && rg[IDX(z - 1, y, x)]) || (z + 1 < n[0] && rg[IDX(z + 1, y, x)]) || (y > 0 && rg[IDX(z, y - 1, x)]) ||
                        (y + 1 < NY && rg[IDX(z, y + 1, x)]) || (x > 0 && rg[IDX(z, y, x - 1)]) || (x + 1 < NX && rg[IDX(z, y, x + 1)]);
            k[i] = (uint8_t)(s || !cut);
        }
        /* 4-connected components in scipy.ndimage.label order (first pixel in raster order) */
        for (size_t i = 0; i < M; i++) lab[i] = -1;
        int ncomp = 0;
        for (size_t i0 = 0; i0 < M; i0++) {
            if (!k[i0] || lab[i0] >= 0) continue;
            size_t ns = 0, cnt = 0; stk[ns++] = (uint32_t)i0; lab[i0] = ncomp;
            while (ns) {
                const uint32_t i = stk[--ns]; cnt++; const int r = (int)(i / W), c = (int)(i % W);
                const int nb[4][2] = {{r - 1, c}, {r + 1, c}, {r, c - 1}, {r, c + 1}};
                for (int q = 0; q < 4; q++) {
                    if (nb[q][0] < 0 || nb[q][0] >= H || nb[q][1] < 0 || nb[q][1] >= W) continue;
                    const size_t j = (size_t)nb[q][0] * W + nb[q][1]; if (k[j] && lab[j] < 0) { lab[j] = ncomp; stk[ns++] = (uint32_t)j; }
                }
            }
            if ((long)cnt >= minv) {
                if (N == ncap) { ncap = ncap ? ncap * 2 : 1024; nodes = realloc(nodes, ncap * sizeof *nodes); }
                ps_node *nd = &nodes[N++]; snprintf(nd->name, sizeof nd->name, "%s", names[pi]); nd->comp = ncomp; nd->v0 = V.n; nd->nv = 0;
                nd->r0 = H; nd->r1 = -1; nd->c0 = W; nd->c1 = -1; nd->step = step;
                for (size_t i = 0; i < M; i++) if (lab[i] == ncomp) {   /* raster order, as m = lab == i */
                    if (ps_vpush(&V, A[2][i], A[1][i], A[0][i]) || ps_gpush(&GI, (int32_t)(i / W), (int32_t)(i % W))) return 1;
                    nd->nv++; const int r = (int)(i / W), c = (int)(i % W);
                    nd->r0 = r < nd->r0 ? r : nd->r0; nd->r1 = r > nd->r1 ? r : nd->r1; nd->c0 = c < nd->c0 ? c : nd->c0; nd->c1 = c > nd->c1 ? c : nd->c1;
                }
            }
            ncomp++;
        }
        free(k); free(lab); free(stk); free(A[0]); free(A[1]); free(A[2]);
    }
    free(names);
    printf("patches %d comps %u vertices %zu\n", nn, N, V.n);
    if (N < 2) { fprintf(stderr, "patch-solve: fewer than 2 nodes\n"); return 1; }

    /* polar coordinates of every vertex */
    double *R = malloc(V.n * 8), *T = malloc(V.n * 8), *Rs = malloc(V.n * 8);
    for (size_t i = 0; i < V.n; i++) ps_frame(&G, V.z[i], V.y[i], V.x[i], &R[i], &T[i]);
    memcpy(Rs, R, V.n * 8); qsort(Rs, V.n, 8, ps_dcmp);
    const double rm = V.n % 2 ? Rs[V.n / 2] : 0.5 * (Rs[V.n / 2 - 1] + Rs[V.n / 2]), dth = G.par ? 4.0 : 4.0 / rm; free(Rs);

    /* GT sheet per node */
    ps_gtd *gd = calloc(N, sizeof *gd);
    for (uint32_t a = 0; a < N; a++) {
        gd[a].gt = -1; if (!gt) continue;
        int hist[14] = {0}, m = 0;
        for (size_t v = nodes[a].v0; v < nodes[a].v0 + nodes[a].nv; v++) {
            const int64_t z = (int64_t)nearbyint(V.z[v]), y = (int64_t)nearbyint(V.y[v]), x = (int64_t)nearbyint(V.x[v]);
            int c = 0;   /* maximum_filter(size 3) of the sheet codes */
            for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                int64_t zz = z + dz, yy = y + dy, xx = x + dx;
                zz = zz < 0 ? 0 : zz >= n[0] ? n[0] - 1 : zz; yy = yy < 0 ? 0 : yy >= NY ? NY - 1 : yy; xx = xx < 0 ? 0 : xx >= NX ? NX - 1 : xx;
                const int g = gt[IDX(zz, yy, xx)]; if (g >= 1 && g <= 252 && g > c) c = g;
            }
            if (!c) continue;
            double r, th; ps_polar(&G, V.z[v], V.y[v], V.x[v], &r, &th);   /* the angle of the float vertex, as the prototype */
            const long s = ((long)nearbyint((c - 1) / 18.0 - sgn * th / (2 * M_PI)) % 14 + 14) % 14;
            hist[s]++; m++;
        }
        if (m < 10) continue;
        int b = 0; for (int k = 1; k < 14; k++) if (hist[k] > hist[b]) b = k;
        gd[a] = (ps_gtd){b, hist[b], m};
    }

    /* ---- rays ---- */
    ps_ent *E = malloc(V.n * sizeof *E); uint32_t *vnode = malloc(V.n * 4);
    for (uint32_t a = 0; a < N; a++) for (size_t v = nodes[a].v0; v < nodes[a].v0 + nodes[a].nv; v++) vnode[v] = a;
    int64_t tbmin = INT64_MAX;
    for (size_t i = 0; i < V.n; i++) { const int64_t tb = (int64_t)floor(T[i] / dth); if (tb < tbmin) tbmin = tb; }
    for (size_t i = 0; i < V.n; i++) {
        const int64_t zb = (int64_t)floor(V.z[i] / 4), tb = (int64_t)floor(T[i] / dth);
        E[i] = (ps_ent){(uint64_t)(zb * 100000 + (tb - tbmin)), R[i], vnode[i], (uint32_t)i};
    }
    qsort(E, V.n, sizeof *E, ps_entcmp);
    typedef struct { uint32_t a, b; int runs, runs2; float dr, ra; } ps_cr;
    ps_cr *cr = nullptr; size_t ncr = 0, ccr = 0;
    uint32_t *sc = malloc(V.n * 4); double *sr = malloc(V.n * 8), *prof = malloc(8 * 4096); int *pk = malloc(4 * 4096);
    long nray = 0;
    for (size_t s = 0, e; s < V.n; s = e) {
        for (e = s + 1; e < V.n && E[e].key == E[s].key; e++);
        if (e - s < 2) continue;
        double zc = 0, tc = 0; for (size_t i = s; i < e; i++) { zc += V.z[E[i].i]; tc += T[E[i].i]; } zc /= (double)(e - s); tc /= (double)(e - s);
        size_t ns = 0;
        for (size_t i = s; i < e; i++) {
            if (ns && sc[ns - 1] == E[i].c) continue;
            if (ns && E[i].r - sr[ns - 1] < (dupf > 0 ? 0.5 : 3)) continue;   /* --dup-frac keeps close crossings (same-sheet duplicates vote 0) */
            sc[ns] = E[i].c; sr[ns] = E[i].r; ns++;
        }
        if (ns < 2) continue;
        nray++;
        const int64_t zr = (int64_t)nearbyint(zc);
        for (size_t q = 0; q + 1 < ns; q++) {
            const double ra = sr[q], rb = sr[q + 1]; int runs = 0, runs2 = 0;
            const double lo = ra + 2.5, hi = rb - 2.5; const long nt_ = hi > lo ? (long)ceil((hi - lo) / 0.5) : 0;
            if (nt_ > 0 && zr >= 0 && zr < n[0]) {
                const long np2 = (long)ceil((rb + 0.01 - ra) / 0.5); if (np2 > 4096) continue;
                for (long i = 0; i < np2; i++) {
                    const double t = ra + 0.5 * (double)i; double fy, fx; ps_frame_pt(&G, zc, t, tc, &fy, &fx);
                    int64_t yy = (int64_t)nearbyint(fy), xx = (int64_t)nearbyint(fx);
                    yy = yy < 0 ? 0 : yy >= NY ? NY - 1 : yy; xx = xx < 0 ? 0 : xx >= NX ? NX - 1 : xx; prof[i] = prob[IDX(zr, yy, xx)];
                }
                const int npk = ps_peaks(prof, (int)np2, 32, pk);
                for (int i = 0; i < npk; i++) { const double t = ra + 0.5 * pk[i]; runs += t > lo && t < hi; }
                int prev = 0;
                for (long i = 0; i < nt_; i++) {
                    const double t = lo + 0.5 * (double)i; double fy, fx; ps_frame_pt(&G, zc, t, tc, &fy, &fx);
                    int64_t yy = (int64_t)nearbyint(fy), xx = (int64_t)nearbyint(fx);
                    yy = yy < 0 ? 0 : yy >= NY ? NY - 1 : yy; xx = xx < 0 ? 0 : xx >= NX ? NX - 1 : xx;
                    const int hv = rg[IDX(zr, yy, xx)]; runs2 += hv && !prev; prev = hv;
                }
            }
            if (ncr == ccr) { ccr = ccr ? ccr * 2 : 1 << 16; cr = realloc(cr, ccr * sizeof *cr); }
            cr[ncr++] = (ps_cr){sc[q], sc[q + 1], runs, runs2, (float)(rb - ra), (float)ra};
        }
    }
    free(sc); free(sr); free(prof); free(pk); free(E); free(vnode);
    double pitch = 0;
    {   double *dv = malloc((ncr + 1) * 8); size_t m = 0;
        for (size_t i = 0; i < ncr; i++) if (!cr[i].runs && !cr[i].runs2 && cr[i].dr >= 3) dv[m++] = cr[i].dr;
        if (m) { qsort(dv, m, 8, ps_dcmp); pitch = dv[m / 2]; }
        free(dv);
        printf("crossings %zu, adjacent by both counters %zu, pitch %.2f voxels\n", ncr, m, pitch);
    }
    {   const char *ct = ps_opt(argc, argv, "--cross-tsv", nullptr);   /* diagnostic: every crossing (with --gt the GT turn difference) */
        FILE *f = ct ? fopen(ct, "w") : nullptr;
        if (f) {
            fprintf(f, "a\tb\tprob_runs\tridge_runs\tdr\tra\tgt_d\n");
            for (size_t i = 0; i < ncr; i++) {
                int g = 99; if (gt && gd[cr[i].a].gt >= 0 && gd[cr[i].b].gt >= 0) { g = ((gd[cr[i].b].gt - gd[cr[i].a].gt) % 14 + 14) % 14; if (g > 10) g -= 14; }
                fprintf(f, "%u\t%u\t%d\t%d\t%.2f\t%.1f\t%d\n", cr[i].a, cr[i].b, cr[i].runs, cr[i].runs2, cr[i].dr, cr[i].ra, g);
            }
            fclose(f);
        }
    }
    ps_rv *rv = malloc((4 * ncr + 1) * sizeof *rv); size_t nrv = 0; uint32_t seq = 0;
    for (size_t i = 0; i < ncr; i++) {
        const int dp = pitch > 0 ? (int)lround(cr[i].dr / pitch) : 1, dpt = dp < 1 ? 1 : dp;
        if (dupf > 0 && cr[i].dr < dupf * pitch && !cr[i].runs && !cr[i].runs2) {   /* a second trace of the same sheet (both faces / duplicates) */
            const int nv_ = use_p + use_r + use_pt + use_mx; for (int k = 0; k < nv_; k++) rv[nrv++] = (ps_rv){cr[i].a, cr[i].b, 0, seq++};
            continue;
        }
        if (use_p) rv[nrv++] = (ps_rv){cr[i].a, cr[i].b, 1 + cr[i].runs, seq++};
        if (use_r) rv[nrv++] = (ps_rv){cr[i].a, cr[i].b, 1 + cr[i].runs2, seq++};
        if (use_pt) rv[nrv++] = (ps_rv){cr[i].a, cr[i].b, dpt, seq++};
        if (use_mx) rv[nrv++] = (ps_rv){cr[i].a, cr[i].b, dpt > 1 + cr[i].runs2 ? dpt : 1 + cr[i].runs2, seq++};
    }
    free(cr);
    qsort(rv, nrv, sizeof *rv, ps_rvcmp);
    /* edges per directed pair */
    l1_vote *ed = malloc((nrv + 1) * sizeof *ed); size_t ne = 0; double agw = 0, agd = 0;
    for (size_t s = 0, e; s < nrv; s = e) {
        for (e = s; e < nrv && rv[e].a == rv[s].a && rv[e].b == rv[s].b; e++);
        if ((long)(e - s) < minvotes) continue;
        int32_t bd = 0; long bw = -1; uint32_t bseq = UINT32_MAX;
        for (size_t i = s, j; i < e; i = j) {
            uint32_t fs = rv[i].seq; for (j = i; j < e && rv[j].d == rv[i].d; j++) if (rv[j].seq < fs) fs = rv[j].seq;
            const long w = (long)(j - i);
            if (ray) ed[ne++] = (l1_vote){rv[i].a, rv[i].b, rv[i].d, w};
            if (w > bw || (w == bw && fs < bseq)) { bw = w; bd = rv[i].d; bseq = fs; }
        }
        if (!ray) ed[ne++] = (l1_vote){rv[s].a, rv[s].b, bd, bw};
        agw += (double)bw; agd += (double)bw * bw / (double)(e - s);
    }
    free(rv);
    if (linksp) {   /* same-sheet links (sheet-trace links.tsv: a b contacts kind): d = 0 votes, weight contacts x --link-w, between the largest comps */
        FILE *lf = fopen(linksp, "r"); if (!lf) { fprintf(stderr, "patch-solve: cannot read %s\n", linksp); return 1; }
        char line[1024], na[256], nb[256]; long cnt, nadd = 0; double wadd = 0;
        if (!fgets(line, sizeof line, lf)) line[0] = 0;
        while (fgets(line, sizeof line, lf)) {
            char kind[32] = "";
            if (sscanf(line, "%255s %255s %ld %31s", na, nb, &cnt, kind) < 3) continue;
            if (lkinds && kind[0] && !strstr(lkinds, kind)) continue;
            int ia = -1, ib = -1;   /* nodes are in patch-name order: binary search the first comp, keep the largest */
            for (int side = 0; side < 2; side++) {
                const char *nm = side ? nb : na; uint32_t lo = 0, hi = N;
                while (lo < hi) { const uint32_t mid = (lo + hi) / 2; if (strcmp(nodes[mid].name, nm) < 0) lo = mid + 1; else hi = mid; }
                int best = -1; for (uint32_t k = lo; k < N && !strcmp(nodes[k].name, nm); k++) if (best < 0 || nodes[k].nv > nodes[best].nv) best = (int)k;
                if (side) ib = best; else ia = best;
            }
            if (ia < 0 || ib < 0 || ia == ib) continue;
            const int64_t w = (int64_t)llround((double)cnt * linkw); if (w <= 0) continue;
            ed = realloc(ed, (ne + 1) * sizeof *ed); ed[ne++] = (l1_vote){(uint32_t)ia, (uint32_t)ib, 0, w}; nadd++; wadd += (double)w;
        }
        fclose(lf);
        printf("links: %ld same-sheet edges (weight %.0f) from %s\n", nadd, wadd, linksp);
    }
    if (gt && oracle) {   /* diagnostic: each edge's d replaced by the GT turn difference (mod 14, the representative nearest the vote) */
        long nch = 0; double wch = 0, wall = 0;
        for (size_t i = 0; i < ne; i++) {
            l1_vote *v = &ed[i]; wall += (double)v->w;
            if (gd[v->a].gt < 0 || gd[v->b].gt < 0) continue;
            const int g = gd[v->b].gt - gd[v->a].gt; int best = v->d, bd = 1 << 30;
            for (int k = -3; k <= 3; k++) { const int c = g + 14 * k, dd = abs(c - v->d); if (dd < bd) { bd = dd; best = c; } }
            if (best != v->d) { nch++; wch += (double)v->w; v->d = best; }
        }
        printf("oracle: %ld edges changed (%.3f of edge weight)\n", nch, wall > 0 ? wch / wall : 0);
    }
    printf("rays %ld edges %zu mean vote agreement %.3f (%s votes, %s counter)\n", nray, ne, agw > 0 ? agd / agw : 0, ray ? "ray" : "pair", cmode);

    /* ---- solve + prune ---- */
    uint8_t *active = malloc(N); memset(active, 1, N);
    int32_t *x = calloc(N, 4); uint32_t *P = malloc(N * 4), *lab = malloc(N * 4), ncc = 0;
    double *bad = malloc(N * 8), *tot = malloc(N * 8), *fr = malloc(N * 8);
    l1_vote *ev = malloc((ne + 1) * sizeof *ev);
    for (int rnd = 0; rnd < (rounds > 0 ? rounds : 1); rnd++) {
        size_t nv = 0; for (size_t i = 0; i < ne; i++) if (active[ed[i].a] && active[ed[i].b]) ev[nv++] = ed[i];
        for (uint32_t i = 0; i < N; i++) P[i] = i;
        for (size_t i = 0; i < nv; i++) { const uint32_t ra = ps_find(P, ev[i].a), rb = ps_find(P, ev[i].b); if (ra != rb) P[ra < rb ? rb : ra] = ra < rb ? ra : rb; }
        ncc = 0; for (uint32_t i = 0; i < N; i++) { const uint32_t r = ps_find(P, i); lab[i] = r == i ? ncc++ : lab[r]; }
        const size_t nm = l1_votes_merge(ev, nv);
        memset(x, 0, N * 4); l1_stats st = {0};
        if (l1_solve(N, ev, nm, x, 100000, &st)) { fprintf(stderr, "patch-solve: l1_solve failed\n"); return 1; }
        /* violations on the (unmerged) active edges */
        memset(bad, 0, N * 8); memset(tot, 0, N * 8); double vw = 0, tw = 0;
        for (size_t i = 0; i < ne; i++) {
            const l1_vote *v = &ed[i]; if (!active[v->a] || !active[v->b]) continue;
            const int viol = x[v->b] - x[v->a] != v->d; const double w = (double)v->w;
            tw += w; if (viol) { vw += w; bad[v->a] += w; bad[v->b] += w; } tot[v->a] += w; tot[v->b] += w;
        }
        double mx = 0; uint32_t nact = 0;
        for (uint32_t i = 0; i < N; i++) { fr[i] = active[i] && tot[i] > 0 ? bad[i] / tot[i] : 0; if (fr[i] > mx) mx = fr[i]; nact += active[i]; }
        printf("round %d: active %u graph comps %u  L1 moves %d E %lld -> %lld  edges violated (w share) %.3f", rnd, nact, ncc, st.moves, (long long)st.e0, (long long)st.e1, tw > 0 ? vw / tw : 0);
        if (gt) {
            double sc1[3]; ps_score(N, x, lab, ncc, active, gd, 1, sc1); double ka = 0, kt = 0;
            for (uint32_t i = 0; i < N; i++) if (gd[i].gt >= 0) { kt += gd[i].tot; if (active[i]) ka += gd[i].tot; }
            printf("  right sheet+turn (largest graph comp) %.4f  matched area kept %.3f", sc1[1], kt > 0 ? ka / kt : 0);
        }
        printf("\n");
        if (mx < pthr) break;   /* (the last round prunes too, as the prototype; the final score keeps that round's turns) */
        double *srt = malloc(N * 8); memcpy(srt, fr, N * 8); qsort(srt, N, 8, ps_dcmp);
        long kq = (long)(pfrac * nact); if (kq < 1) kq = 1;
        double thr = srt[N - (size_t)kq]; if (thr < pthr) thr = pthr; free(srt);
        for (uint32_t i = 0; i < N; i++) if (active[i] && fr[i] >= thr) active[i] = 0;
    }
    if (gt) {
        for (int sg = 1; sg >= -1; sg -= 2) {
            double s3[3]; ps_score(N, x, lab, ncc, active, gd, sg, s3);
            printf("sign %+d: area on right sheet+turn %.4f | largest graph comp %.4f (holds %.3f of matched area)\n", sg, s3[0], s3[1], s3[2]);
        }
        double pa = 0, pt = 0; for (uint32_t i = 0; i < N; i++) if (gd[i].gt >= 0) { pa += gd[i].pure; pt += gd[i].tot; }
        printf("graph comps %u | purity ceiling (dominant-sheet share) %.4f\n", ncc, pt > 0 ? pa / pt : 0);
    }
    const char *outp = ps_opt(argc, argv, "--out-patches", nullptr);
    if (outp) {   /* the active nodes as tifxyz (cropped to their grid bbox; other vertices -1; winding = the solved turn) */
        mkdir(outp, 0755); long nw = 0;
        for (uint32_t i = 0; i < N; i++) {
            if (!active[i]) continue;
            const ps_node *nd = &nodes[i]; const int w = nd->c1 - nd->c0 + 1, h = nd->r1 - nd->r0 + 1;
            float *X = malloc((size_t)w * h * 4), *Y = malloc((size_t)w * h * 4), *Z = malloc((size_t)w * h * 4); if (!X || !Y || !Z) return 1;
            for (size_t q = 0; q < (size_t)w * h; q++) X[q] = Y[q] = Z[q] = -1;
            for (size_t v = nd->v0; v < nd->v0 + nd->nv; v++) {
                const size_t q = (size_t)(GI.rc[2 * v] - nd->r0) * w + (size_t)(GI.rc[2 * v + 1] - nd->c0);
                X[q] = (float)V.x[v]; Y[q] = (float)V.y[v]; Z[q] = (float)V.z[v];
            }
            char d[1600]; snprintf(d, sizeof d, "%s/%s_c%d", outp, nd->name, nd->comp);
            char src[64]; snprintf(src, sizeof src, "patch-solve gc %u", lab[i]);
            if (tifxyz_write(d, w, h, X, Y, Z, nd->step, atof(ps_opt(argc, argv, "--vs", "9.596")), x[i], src)) { fprintf(stderr, "patch-solve: cannot write %s\n", d); return 1; }
            free(X); free(Y); free(Z); nw++;
        }
        printf("wrote %ld patches -> %s\n", nw, outp);
    }
    if (tsv) {
        FILE *f = fopen(tsv, "w"); if (!f) { fprintf(stderr, "patch-solve: cannot write %s\n", tsv); return 1; }
        fprintf(f, "patch\tcomp\tvertices\tactive\tgraph_comp\tturn\tgt_sheet\tgt_pure\tgt_n\n");
        for (uint32_t i = 0; i < N; i++)
            fprintf(f, "%s\t%d\t%zu\t%d\t%u\t%d\t%d\t%d\t%d\n", nodes[i].name, nodes[i].comp, nodes[i].nv, active[i], lab[i], x[i], gd[i].gt, gd[i].pure, gd[i].tot);
        if (fclose(f)) return 1;
    }
    free(prob); free(gt); free(rg); free(nodes); free(GI.rc); free(V.z); free(V.y); free(V.x); free(R); free(T); free(gd);
    free(ed); free(ev); free(active); free(x); free(P); free(lab); free(bad); free(tot); free(fr);
    return 0;
#undef IDX
#undef INB
}
