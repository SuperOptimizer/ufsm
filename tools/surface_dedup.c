/* Remove duplicated surface area across tifxyz segments of one scan, in priority order.

   usage: surface_dedup --cell C --out DIR [--cap-log2 30] [--cover 0.5] seg1.tifxyz seg2.tifxyz ...
   Segments are processed in the order given (first = highest priority). Each quad of 4 valid grid points is sampled
   every C/2 voxels; a sample is covered when one of the 27 cells (size C voxels) around it holds a sample of an earlier
   segment. A quad is a duplicate when more than --cover of its samples are covered; a grid point is dropped when every
   valid quad touching it is a duplicate. The kept quads' cells then join the occupied set for later segments.
   Writes DIR/<i>.mask (uint8 per grid point, row-major h x w: 255 keep, 0 drop) and one stats line per segment on
   stdout: index, path, w, h, valid points, kept points, duplicate quad fraction. */
#include "../src/tiff.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t *set; static uint64_t mask_; static double cell;

static inline uint64_t hash64(uint64_t k) { k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL; k ^= k >> 33; return k; }
static inline uint64_t key(int64_t z, int64_t y, int64_t x) { return 1 + ((uint64_t)(z & 0x1fffff) << 42 | (uint64_t)(y & 0x1fffff) << 21 | (uint64_t)(x & 0x1fffff)); }
static int has(uint64_t k) {
    for (uint64_t i = hash64(k) & mask_;; i = (i + 1) & mask_) { uint64_t v = __atomic_load_n(&set[i], __ATOMIC_RELAXED); if (v == k) return 1; if (!v) return 0; }
}
static long long nset;
static void put(uint64_t k) {
    for (uint64_t i = hash64(k) & mask_;; i = (i + 1) & mask_) {
        uint64_t v = __atomic_load_n(&set[i], __ATOMIC_RELAXED);
        if (v == k) return;
        if (!v) { uint64_t z = 0; if (__atomic_compare_exchange_n(&set[i], &z, k, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) { __atomic_fetch_add(&nset, 1, __ATOMIC_RELAXED); return; } if (z == k) return; }
    }
}

typedef struct { int w, h; float *x, *y, *z; uint8_t *valid; } grid;
static int load(const char *dir, grid *g) {
    const char *names[3] = {"x.tif", "y.tif", "z.tif"}; float **ax[3] = {&g->x, &g->y, &g->z};
    for (int a = 0; a < 3; a++) {
        char p[2048]; snprintf(p, sizeof p, "%s/%s", dir, names[a]);
        tiff *t = tiff_open_file(p); tiff_page pg;
        if (!t || tiff_page_info(t, 0, &pg) || pg.bits != 32 || pg.fmt != 3 || pg.spp != 1 || (a && (pg.w != g->w || pg.h != g->h))) { fprintf(stderr, "%s: bad tifxyz\n", p); return 1; }
        g->w = pg.w; g->h = pg.h; *ax[a] = malloc((size_t)pg.w * pg.h * 4);
        if (tiff_read_page(t, 0, *ax[a])) { fprintf(stderr, "%s: decode failed\n", p); return 1; }
        tiff_close(t);
    }
    size_t n = (size_t)g->w * g->h; g->valid = malloc(n);
    for (size_t i = 0; i < n; i++) g->valid[i] = isfinite(g->x[i]) && isfinite(g->y[i]) && isfinite(g->z[i]) && g->z[i] > 0 && g->x[i] >= 0;
    return 0;
}

/* visit the samples of quad (r, c); mode 0: return covered count (n in *tot); mode 1: insert */
static long quad(const grid *g, int r, int c, int mode, long *tot) {
    size_t i00 = (size_t)r * g->w + c, idx[4] = {i00, i00 + 1, i00 + g->w, i00 + g->w + 1};
    double p[4][3];
    for (int k = 0; k < 4; k++) { p[k][0] = g->z[idx[k]]; p[k][1] = g->y[idx[k]]; p[k][2] = g->x[idx[k]]; }
    double e1 = 0, e2 = 0;
    for (int d = 0; d < 3; d++) { e1 += (p[1][d] - p[0][d]) * (p[1][d] - p[0][d]); e2 += (p[2][d] - p[0][d]) * (p[2][d] - p[0][d]); }
    int nu = (int)ceil(sqrt(e1) / (cell / 2)) + 1, nv = (int)ceil(sqrt(e2) / (cell / 2)) + 1;
    if (nu > 64) nu = 64;
    if (nv > 64) nv = 64;
    long cov = 0, n = 0;
    for (int a = 0; a < nu; a++) for (int b = 0; b < nv; b++) {
        double u = nu > 1 ? (double)a / (nu - 1) : 0, v = nv > 1 ? (double)b / (nv - 1) : 0; int64_t q[3];
        for (int d = 0; d < 3; d++) q[d] = (int64_t)floor(((1 - u) * (1 - v) * p[0][d] + u * (1 - v) * p[1][d] + (1 - u) * v * p[2][d] + u * v * p[3][d]) / cell);
        n++;
        if (mode) { put(key(q[0], q[1], q[2])); continue; }
        int hit = 0;
        for (int dz = -1; dz <= 1 && !hit; dz++) for (int dy = -1; dy <= 1 && !hit; dy++) for (int dx = -1; dx <= 1 && !hit; dx++) hit = has(key(q[0] + dz, q[1] + dy, q[2] + dx));
        cov += hit;
    }
    if (tot) *tot = n;
    return cov;
}

int main(int argc, char **argv) {
    const char *out = nullptr; int caplog = 30; double cover = 0.5; cell = 0; int first = argc;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--cell") && i + 1 < argc) cell = atof(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "--cap-log2") && i + 1 < argc) caplog = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cover") && i + 1 < argc) cover = atof(argv[++i]);
        else { first = i; break; }
    }
    if (!out || !(cell > 0) || caplog < 10 || caplog > 36 || first >= argc) { fprintf(stderr, "usage: surface_dedup --cell C --out DIR [--cap-log2 30] [--cover 0.5] seg.tifxyz...\n"); return 2; }
    mask_ = (1ULL << caplog) - 1; set = calloc(mask_ + 1, 8);
    if (!set) { fprintf(stderr, "cannot allocate 2^%d set\n", caplog); return 1; }
    for (int si = first; si < argc; si++) {
        grid g = {0};
        if (load(argv[si], &g)) return 1;
        int W = g.w, H = g.h; size_t nq = (size_t)(W - 1) * (H - 1);
        uint8_t *dup = calloc(nq, 1), *qv = calloc(nq, 1);
        long nquad = 0, ndup = 0;
        #pragma omp parallel for schedule(dynamic, 4) reduction(+:nquad, ndup)
        for (int r = 0; r < H - 1; r++) for (int c = 0; c < W - 1; c++) {
            size_t i00 = (size_t)r * W + c;
            if (!(g.valid[i00] && g.valid[i00 + 1] && g.valid[i00 + W] && g.valid[i00 + W + 1])) continue;
            size_t qi = (size_t)r * (W - 1) + c; qv[qi] = 1; nquad++;
            long n = 0, cv = quad(&g, r, c, 0, &n);
            if (n && cv > cover * n) { dup[qi] = 1; ndup++; }
        }
        #pragma omp parallel for schedule(dynamic, 4)
        for (int r = 0; r < H - 1; r++) for (int c = 0; c < W - 1; c++) { size_t qi = (size_t)r * (W - 1) + c; if (qv[qi] && !dup[qi]) quad(&g, r, c, 1, nullptr); }
        uint8_t *m = malloc((size_t)W * H); long nvalid = 0, nkeep = 0;
        for (int r = 0; r < H; r++) for (int c = 0; c < W; c++) {
            size_t i = (size_t)r * W + c; int any = 0, keep = 0;
            for (int dr = -1; dr <= 0; dr++) for (int dc = -1; dc <= 0; dc++) {
                int qr = r + dr, qc = c + dc;
                if (qr < 0 || qc < 0 || qr >= H - 1 || qc >= W - 1) continue;
                size_t qi = (size_t)qr * (W - 1) + qc;
                if (qv[qi]) { any = 1; if (!dup[qi]) keep = 1; }
            }
            m[i] = g.valid[i] && (keep || !any) ? 255 : 0;
            nvalid += g.valid[i]; nkeep += m[i] != 0;
        }
        char p[2048]; snprintf(p, sizeof p, "%s/%d.mask", out, si - first);
        FILE *f = fopen(p, "wb"); if (!f || fwrite(m, 1, (size_t)W * H, f) != (size_t)W * H) { fprintf(stderr, "%s: write failed\n", p); return 1; } fclose(f);
        printf("%d\t%s\t%d\t%d\t%ld\t%ld\t%.4f\t%lld\n", si - first, argv[si], W, H, nvalid, nkeep, nquad ? (double)ndup / nquad : 0, nset); fflush(stdout);
        if ((double)nset > 0.7 * (double)(mask_ + 1)) { fprintf(stderr, "occupied-cell set over 70%% full; raise --cap-log2\n"); return 1; }
        free(g.x); free(g.y); free(g.z); free(g.valid); free(dup); free(qv); free(m);
    }
    return 0;
}
