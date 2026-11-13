/* Tracer box tools (unroll v3 P3; C ports of scripts/v3/proto/trace_box2.sh's Python prep and score_patches.py).

   ufsm trace-prep --prob P [--prob-key K] --box Z0,Y0,X0,NZ,NY,NX --axis AXIS.json [--axis-scale S] --out DIR [--vs UM]
                   [--field ridge|prob] [--levels 4] [--seeds 64] [--rng 2] [--margin 64] [--zstep 32] [--threads N]
     DIR/ridge.zarr  the tracer input as an OME zarr v2 pyramid (uint8, 64^3 uncompressed chunks, levels max-pooled by 2; level l scale
                     VS 2^l um) with the VC3D volume meta.json; --field ridge (default) = 255 on the radial ridge of the prob (p >= 64,
                     radial non-max), prob = the prob itself;
     DIR/ridge.raw   the level-0 field (NZ x NY x NX);
     DIR/seeds.txt   --seeds random ridge voxels "x y z" (box voxels) on the z planes margin, margin + zstep, .. and margin from the
                     y / x faces (deterministic for --rng).

   ufsm patch-score --paths DIR --gt G [--gt-key K] --box Z0,Y0,X0,NZ,NY,NX --axis AXIS.json [--axis-scale S] [--sgn +-1] [--tsv OUT]
     VC3D tifxyz patches (box voxel coordinates) against GT winding_mod14 codes of the box (1 + round(18 q) mod 252):
       recall     GT sheet voxels within 1 voxel (Chebyshev) of a densified patch / GT sheet voxels
       precision  densified patch voxels within 1 of GT / those within L1 distance 8 of GT (GT labels only some sheets)
       purity     per patch the share of its GT-matched voxels on its dominant GT sheet (round(q - sgn theta / 2 pi) mod 14, theta
                  unwrapped about the patch's circular mean), matched-weighted over patches
     --tsv: patch, voxels, near (L1 <= 8), matched, purity, dominant sheet. */
#include "patchsolve.h"
#include "band_onfly.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

static const char *tb_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static int tb_box(const char *s, int64_t o[3], int64_t n[3]) {
    long long b[6]; if (!s || sscanf(s, "%lld,%lld,%lld,%lld,%lld,%lld", &b[0], &b[1], &b[2], &b[3], &b[4], &b[5]) != 6) return -1;
    for (int d = 0; d < 3; d++) { o[d] = b[d]; n[d] = b[3 + d]; if (n[d] <= 0) return -1; }
    return 0;
}
static int tb_write(const char *path, const void *b, size_t n) {
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    const size_t w = fwrite(b, 1, n, f); return fclose(f) || w != n ? -1 : 0;
}
static int tb_text(const char *path, const char *s) { return tb_write(path, s, strlen(s)); }

/* one zarr v2 array (uint8, chunk C^3, no compressor, '.' separator; edge chunks padded with 0) */
static int tb_zarr2(const char *dir, const uint8_t *a, const int64_t n[3], int C) {
    if (mkdir(dir, 0755) && errno != EEXIST) return -1;
    char p[1200], js[512];
    snprintf(js, sizeof js, "{\"zarr_format\": 2, \"shape\": [%lld, %lld, %lld], \"chunks\": [%d, %d, %d], \"dtype\": \"|u1\", \"compressor\": null, "
             "\"fill_value\": 0, \"order\": \"C\", \"filters\": null, \"dimension_separator\": \".\"}\n", (long long)n[0], (long long)n[1], (long long)n[2], C, C, C);
    snprintf(p, sizeof p, "%s/.zarray", dir); if (tb_text(p, js)) return -1;
    uint8_t *c = malloc((size_t)C * C * C); if (!c) return -1;
    for (int64_t cz = 0; cz * C < n[0]; cz++) for (int64_t cy = 0; cy * C < n[1]; cy++) for (int64_t cx = 0; cx * C < n[2]; cx++) {
        memset(c, 0, (size_t)C * C * C); int any = 0;
        for (int z = 0; z < C && cz * C + z < n[0]; z++) for (int y = 0; y < C && cy * C + y < n[1]; y++) {
            const uint8_t *src = a + ((size_t)(cz * C + z) * n[1] + (size_t)(cy * C + y)) * n[2] + (size_t)(cx * C);
            const int64_t w = n[2] - cx * C < C ? n[2] - cx * C : C;
            memcpy(c + ((size_t)z * C + y) * C, src, (size_t)w); for (int64_t x = 0; x < w && !any; x++) any = src[x] != 0;
        }
        if (!any) continue;   /* missing chunk = fill value */
        snprintf(p, sizeof p, "%s/%lld.%lld.%lld", dir, (long long)cz, (long long)cy, (long long)cx);
        if (tb_write(p, c, (size_t)C * C * C)) { free(c); return -1; }
    }
    free(c); return 0;
}

static uint64_t tb_rng(uint64_t *s) { uint64_t z = (*s += 0x9e3779b97f4a7c15ull); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; return z ^ (z >> 31); }

int cmd_trace_prep(int argc, char **argv) {
    int64_t o[3], n[3];
    const char *probp = tb_opt(argc, argv, "--prob", nullptr), *axp = tb_opt(argc, argv, "--axis", nullptr), *out = tb_opt(argc, argv, "--out", nullptr);
    const char *field = tb_opt(argc, argv, "--field", "ridge");
    const double axs = atof(tb_opt(argc, argv, "--axis-scale", "1")), vs = atof(tb_opt(argc, argv, "--vs", "9.596"));
    const int levels = atoi(tb_opt(argc, argv, "--levels", "4")), nseed = atoi(tb_opt(argc, argv, "--seeds", "64")), margin = atoi(tb_opt(argc, argv, "--margin", "64")),
              zstep = atoi(tb_opt(argc, argv, "--zstep", "32"));
    uint64_t rs = strtoull(tb_opt(argc, argv, "--rng", "2"), nullptr, 10);
    int nt = atoi(tb_opt(argc, argv, "--threads", "0")); if (nt <= 0) nt = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (!probp || !axp || !out || tb_box(tb_opt(argc, argv, "--box", nullptr), o, n) || axs <= 0 || levels < 1 || levels > 8 || zstep < 1 ||
        (strcmp(field, "ridge") && strcmp(field, "prob"))) {
        fprintf(stderr, "usage: ufsm trace-prep --prob P [--prob-key K] --box Z0,Y0,X0,NZ,NY,NX --axis A [--axis-scale S] --out DIR [--vs UM]\n"
                        "         [--field ridge|prob] [--levels 4] [--seeds 64] [--rng 2] [--margin 64] [--zstep 32] [--threads N] [--cache DIR]\n");
        return 2;
    }
    ps_geo G; if (ps_geo_init(&G, axp, axs, o, n)) { fprintf(stderr, "trace-prep: cannot load axis %s\n", axp); return 1; }
    uint8_t *prob = ps_box(probp, tb_opt(argc, argv, "--prob-key", nullptr), tb_opt(argc, argv, "--cache", nullptr), o, n, nt); if (!prob) return 1;
    const size_t N = (size_t)(n[0] * n[1] * n[2]);
    uint8_t *rg = malloc(N); if (!rg) return 1;
    ps_ridge(prob, rg, n, &G, nt);
    size_t nr = 0; for (size_t i = 0; i < N; i++) nr += rg[i];
    uint8_t *f = malloc(N); if (!f) return 1;
    const int use_prob = !strcmp(field, "prob");
    for (size_t i = 0; i < N; i++) f[i] = use_prob ? prob[i] : rg[i] ? 255 : 0;
    if (mkdir(out, 0755) && errno != EEXIST) { fprintf(stderr, "trace-prep: cannot create %s\n", out); return 1; }
    char p[1200]; snprintf(p, sizeof p, "%s/ridge.raw", out); if (tb_write(p, f, N)) return 1;
    snprintf(p, sizeof p, "%s/ridge.zarr", out); if (mkdir(p, 0755) && errno != EEXIST) return 1;
    /* pyramid */
    uint8_t *cur = f; int64_t cn[3] = {n[0], n[1], n[2]};
    for (int l = 0; l < levels; l++) {
        snprintf(p, sizeof p, "%s/ridge.zarr/%d", out, l); if (tb_zarr2(p, cur, cn, 64)) { fprintf(stderr, "trace-prep: cannot write %s\n", p); return 1; }
        if (l + 1 == levels) break;
        const int64_t m[3] = {cn[0] / 2, cn[1] / 2, cn[2] / 2}; if (!m[0] || !m[1] || !m[2]) break;
        uint8_t *nx = calloc((size_t)(m[0] * m[1] * m[2]), 1); if (!nx) return 1;
        for (int64_t z = 0; z < 2 * m[0]; z++) for (int64_t y = 0; y < 2 * m[1]; y++) for (int64_t x = 0; x < 2 * m[2]; x++) {
            const uint8_t v = cur[((size_t)z * cn[1] + y) * cn[2] + x]; uint8_t *d = &nx[((size_t)(z / 2) * m[1] + y / 2) * m[2] + x / 2]; if (v > *d) *d = v;
        }
        if (cur != f) free(cur);
        cur = nx; memcpy(cn, m, sizeof cn);
    }
    if (cur != f) free(cur);
    {   char js[8192]; int k = snprintf(js, sizeof js, "{\"multiscales\": [{\"version\": \"0.4\", \"name\": \"ridge\", \"axes\": [{\"name\": \"z\", \"type\": \"space\", \"unit\": \"micrometer\"}, "
                     "{\"name\": \"y\", \"type\": \"space\", \"unit\": \"micrometer\"}, {\"name\": \"x\", \"type\": \"space\", \"unit\": \"micrometer\"}], \"datasets\": [");
        for (int l = 0; l < levels; l++) { const double s = vs * (double)(1 << l);
            k += snprintf(js + k, sizeof js - (size_t)k, "%s{\"path\": \"%d\", \"coordinateTransformations\": [{\"type\": \"scale\", \"scale\": [%g, %g, %g]}]}", l ? ", " : "", l, s, s, s); }
        snprintf(js + k, sizeof js - (size_t)k, "]}]}\n");
        snprintf(p, sizeof p, "%s/ridge.zarr/.zattrs", out); if (tb_text(p, js)) return 1;
        snprintf(p, sizeof p, "%s/ridge.zarr/.zgroup", out); if (tb_text(p, "{\"zarr_format\": 2}\n")) return 1;
        snprintf(js, sizeof js, "{\"type\": \"vol\", \"uuid\": \"ridge\", \"name\": \"ridge\", \"width\": %lld, \"height\": %lld, \"slices\": %lld, \"voxelsize\": %g, "
                 "\"min\": 0.0, \"max\": 255.0, \"format\": \"zarr\"}\n", (long long)n[2], (long long)n[1], (long long)n[0], vs);
        snprintf(p, sizeof p, "%s/ridge.zarr/meta.json", out); if (tb_text(p, js)) return 1;
    }
    /* seeds: ridge voxels on z planes margin + k zstep, margin from the faces */
    size_t nc = 0, cc = 0; int64_t *cand = nullptr;
    for (int64_t z = margin; z < n[0] - margin; z += zstep) for (int64_t y = margin; y < n[1] - margin; y++) for (int64_t x = margin; x < n[2] - margin; x++)
        if (rg[((size_t)z * n[1] + y) * n[2] + x]) {
            if (nc == cc) { cc = cc ? cc * 2 : 1 << 16; cand = realloc(cand, cc * 3 * sizeof *cand); if (!cand) return 1; }
            cand[3 * nc] = x; cand[3 * nc + 1] = y; cand[3 * nc + 2] = z; nc++;
        }
    const size_t ns = (size_t)nseed < nc ? (size_t)nseed : nc;
    for (size_t i = 0; i < ns; i++) {   /* partial Fisher-Yates */
        const size_t j = i + (size_t)(tb_rng(&rs) % (nc - i));
        for (int d = 0; d < 3; d++) { const int64_t t = cand[3 * i + d]; cand[3 * i + d] = cand[3 * j + d]; cand[3 * j + d] = t; }
    }
    snprintf(p, sizeof p, "%s/seeds.txt", out); FILE *sf = fopen(p, "w"); if (!sf) return 1;
    for (size_t i = 0; i < ns; i++) fprintf(sf, "%lld %lld %lld\n", (long long)cand[3 * i], (long long)cand[3 * i + 1], (long long)cand[3 * i + 2]);
    if (fclose(sf)) return 1;
    printf("trace-prep: %s field, ridge fill %.4f, %zu seed candidates, %zu seeds -> %s\n", field, (double)nr / N, nc, ns, out);
    free(cand); free(prob); free(rg); free(f);
    return 0;
}

/* L1 distance (6-neighbour steps) to the nearest set voxel, capped at cap (exact for the L1 metric: two raster passes) */
static void tb_l1dist(const uint8_t *m, const int64_t n[3], uint8_t cap, uint8_t *d) {
    const size_t N = (size_t)(n[0] * n[1] * n[2]); const int64_t sy = n[2], sz = n[1] * n[2];
    for (size_t i = 0; i < N; i++) d[i] = m[i] ? 0 : cap;
    for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) for (int64_t x = 0; x < n[2]; x++) {
        const size_t i = (size_t)(z * sz + y * sy + x); uint8_t v = d[i];
        if (z && d[i - sz] + 1 < v) v = (uint8_t)(d[i - sz] + 1);
        if (y && d[i - sy] + 1 < v) v = (uint8_t)(d[i - sy] + 1);
        if (x && d[i - 1] + 1 < v) v = (uint8_t)(d[i - 1] + 1);
        d[i] = v;
    }
    for (int64_t z = n[0] - 1; z >= 0; z--) for (int64_t y = n[1] - 1; y >= 0; y--) for (int64_t x = n[2] - 1; x >= 0; x--) {
        const size_t i = (size_t)(z * sz + y * sy + x); uint8_t v = d[i];
        if (z + 1 < n[0] && d[i + sz] + 1 < v) v = (uint8_t)(d[i + sz] + 1);
        if (y + 1 < n[1] && d[i + sy] + 1 < v) v = (uint8_t)(d[i + sy] + 1);
        if (x + 1 < n[2] && d[i + 1] + 1 < v) v = (uint8_t)(d[i + 1] + 1);
        d[i] = v;
    }
}
static int tb_u64cmp(const void *a, const void *b) { const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return (x > y) - (x < y); }
static int tb_namecmp(const void *a, const void *b) { return strcmp((const char *)a, (const char *)b); }

int cmd_patch_score(int argc, char **argv) {
    int64_t o[3], n[3];
    const char *paths = tb_opt(argc, argv, "--paths", nullptr), *gtp = tb_opt(argc, argv, "--gt", nullptr), *axp = tb_opt(argc, argv, "--axis", nullptr), *tsv = tb_opt(argc, argv, "--tsv", nullptr);
    const double axs = atof(tb_opt(argc, argv, "--axis-scale", "1")), sgn = atof(tb_opt(argc, argv, "--sgn", "1"));
    int nt = atoi(tb_opt(argc, argv, "--threads", "0")); if (nt <= 0) nt = (int)sysconf(_SC_NPROCESSORS_ONLN);
    const char *mapname = tb_opt(argc, argv, "--map", nullptr);
    if (!paths || !gtp || !axp || tb_box(tb_opt(argc, argv, "--box", nullptr), o, n) || axs <= 0) {
        fprintf(stderr, "usage: ufsm patch-score --paths DIR --gt G [--gt-key K] --box Z0,Y0,X0,NZ,NY,NX --axis A [--axis-scale S] [--sgn +-1] [--tsv OUT]\n");
        return 2;
    }
    ps_geo G; if (ps_geo_init(&G, axp, axs, o, n)) { fprintf(stderr, "patch-score: cannot load axis %s\n", axp); return 1; }
    uint8_t *gt = ps_box(gtp, tb_opt(argc, argv, "--gt-key", nullptr), tb_opt(argc, argv, "--cache", nullptr), o, n, nt); if (!gt) return 1;
    const size_t N = (size_t)(n[0] * n[1] * n[2]); const int64_t NY = n[1], NX = n[2];
#define IDX(z, y, x) (((size_t)(z) * NY + (size_t)(y)) * NX + (size_t)(x))
    uint8_t *gs = malloc(N), *d8 = malloc(N), *cover = calloc(N, 1); if (!gs || !d8 || !cover) return 1;
    size_t ngs = 0; for (size_t i = 0; i < N; i++) { gs[i] = gt[i] >= 1 && gt[i] <= 252; ngs += gs[i]; }
    tb_l1dist(gs, n, 9, d8);
    DIR *dd = opendir(paths); if (!dd) { fprintf(stderr, "patch-score: cannot open %s\n", paths); return 1; }
    char (*names)[256] = nullptr; int nn = 0, cn = 0; struct dirent *de;
    while ((de = readdir(dd))) {
        if (de->d_name[0] == '.' || strlen(de->d_name) >= 256) continue;
        char p[1400]; snprintf(p, sizeof p, "%s/%s/x.tif", paths, de->d_name); if (access(p, F_OK)) continue;
        if (nn == cn) { cn = cn ? cn * 2 : 256; names = realloc(names, (size_t)cn * 256); }
        snprintf(names[nn++], 256, "%s", de->d_name);
    }
    closedir(dd); qsort(names, (size_t)nn, 256, tb_namecmp);
    FILE *tf = tsv ? fopen(tsv, "w") : nullptr; if (tf) fprintf(tf, "patch\tvoxels\tnear\tmatched\tpurity\tsheet\n");
    double sum_hit = 0, sum_near = 0, pw = 0, pwh = 0; int npatch = 0, nlow = 0, npw = 0;
    uint64_t *vk = nullptr; size_t cap = 0;
    for (int pi = 0; pi < nn; pi++) {
        char p[1400]; int w[3], h[3]; float *A[3];
        for (int c = 0; c < 3; c++) { snprintf(p, sizeof p, "%s/%s/%c.tif", paths, names[pi], "xyz"[c]); A[c] = ps_tif(p, &w[c], &h[c]); }
        if (!A[0] || !A[1] || !A[2] || w[0] != w[1] || w[0] != w[2] || h[0] != h[1] || h[0] != h[2]) { free(A[0]); free(A[1]); free(A[2]); continue; }
        int f = 1;   /* grid step in voxels from meta.json scale */
        snprintf(p, sizeof p, "%s/%s/meta.json", paths, names[pi]);
        { FILE *mf = fopen(p, "r"); char buf[65536]; size_t L = mf ? fread(buf, 1, sizeof buf - 1, mf) : 0; if (mf) fclose(mf); buf[L] = 0;
          const char *s = strstr(buf, "\"scale\""); if (s && (s = strchr(s, '['))) { const double sc = atof(s + 1); if (sc > 0) f = (int)lround(1 / sc); } if (f < 1) f = 1; }
        const int W = w[0], H = h[0]; size_t nv = 0;
        #define PUSH(Z_, Y_, X_) do { const double zz = nearbyint(Z_), yy = nearbyint(Y_), xx = nearbyint(X_); \
            if (zz >= 0 && yy >= 0 && xx >= 0 && zz < n[0] && yy < NY && xx < NX) { if (nv == cap) { cap = cap ? cap * 2 : 1 << 20; vk = realloc(vk, cap * 8); } vk[nv++] = IDX(zz, yy, xx); } } while (0)
        int any = 0;
        for (int r = 0; r < H; r++) for (int c = 0; c < W; c++) {
            const size_t q = (size_t)r * W + c; if (!(A[0][q] > -1)) continue; any++;
            const int okr = r + 1 < H && A[0][q + W] > -1, okc = c + 1 < W && A[0][q + 1] > -1, okd = okr && okc && A[0][q + W + 1] > -1;
            if (okd) {
                for (int i = 0; i < f; i++) for (int j = 0; j < f; j++) {
                    const double u = (double)i / f, v = (double)j / f, a = (1 - u) * (1 - v), b = (1 - u) * v, cc = u * (1 - v), e = u * v;
                    PUSH(a * A[2][q] + b * A[2][q + 1] + cc * A[2][q + W] + e * A[2][q + W + 1], a * A[1][q] + b * A[1][q + 1] + cc * A[1][q + W] + e * A[1][q + W + 1],
                         a * A[0][q] + b * A[0][q + 1] + cc * A[0][q + W] + e * A[0][q + W + 1]);
                }
            } else {
                PUSH(A[2][q], A[1][q], A[0][q]);
                if (okr) for (int i = 1; i < f; i++) { const double u = (double)i / f; PUSH((1 - u) * A[2][q] + u * A[2][q + W], (1 - u) * A[1][q] + u * A[1][q + W], (1 - u) * A[0][q] + u * A[0][q + W]); }
                if (okc) for (int j = 1; j < f; j++) { const double v = (double)j / f; PUSH((1 - v) * A[2][q] + v * A[2][q + 1], (1 - v) * A[1][q] + v * A[1][q + 1], (1 - v) * A[0][q] + v * A[0][q + 1]); }
            }
        }
        #undef PUSH
        if (mapname && !strcmp(mapname, names[pi])) {   /* --map: GT sheet (mod 14, as a-n; . no GT within 1, ' ' invalid) per grid vertex, every --map-step */
            const int ms = atoi(tb_opt(argc, argv, "--map-step", "2"));
            for (int r = 0; r < H; r += ms) {
                for (int c = 0; c < W; c += ms) {
                    const size_t qq = (size_t)r * W + c; char ch = ' ';
                    if (A[0][qq] > -1) {
                        const int64_t z = (int64_t)nearbyint(A[2][qq]), y = (int64_t)nearbyint(A[1][qq]), x = (int64_t)nearbyint(A[0][qq]); int cbest = 0; ch = '.';
                        for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                            const int64_t zz = z + dz, yy = y + dy, xx = x + dx; if (zz < 0 || yy < 0 || xx < 0 || zz >= n[0] || yy >= NY || xx >= NX) continue;
                            const int g = gt[IDX(zz, yy, xx)]; if (g >= 1 && g <= 252 && g > cbest) cbest = g;
                        }
                        if (cbest) {
                            double ay, ax; ps_axis(&G, (double)z, &ay, &ax); const double t = atan2((double)y - ay, (double)x - ax);
                            ch = (char)('a' + ((long)nearbyint((cbest - 1) / 18.0 - sgn * t / (2 * M_PI)) % 14 + 14) % 14);
                        }
                    }
                    putchar(ch);
                }
                putchar('\n');
            }
        }
        free(A[0]); free(A[1]); free(A[2]);
        if (any < 4) continue;
        qsort(vk, nv, 8, tb_u64cmp); size_t u = 0; for (size_t i = 0; i < nv; i++) if (!u || vk[i] != vk[u - 1]) vk[u++] = vk[i]; nv = u;
        npatch++;
        long near = 0, hit = 0; int hist[14] = {0}; double sx = 0, sy = 0;
        double *th = malloc((nv + 1) * 8); int *cd = malloc((nv + 1) * 4); size_t nh = 0;
        for (size_t i = 0; i < nv; i++) {
            const size_t k = vk[i]; cover[k] = 1; if (d8[k] > 8) continue; near++;
            const int64_t z = (int64_t)(k / ((size_t)NY * NX)), y = (int64_t)(k / (size_t)NX % (size_t)NY), x = (int64_t)(k % (size_t)NX);
            int c = 0;
            for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                const int64_t zz = z + dz, yy = y + dy, xx = x + dx; if (zz < 0 || yy < 0 || xx < 0 || zz >= n[0] || yy >= NY || xx >= NX) continue;
                const int g = gt[IDX(zz, yy, xx)]; if (g >= 1 && g <= 252 && g > c) c = g;
            }
            if (!c) continue;
            hit++; double ay, ax; ps_axis(&G, (double)z, &ay, &ax); const double t = atan2((double)y - ay, (double)x - ax);
            th[nh] = t; cd[nh] = c; nh++; sx += cos(t); sy += sin(t);
        }
        double pur = -1; int dom = -1;
        if (nh) {
            const double tr = atan2(sy, sx);
            for (size_t i = 0; i < nh; i++) {
                double d = remainder(th[i] - tr, 2 * M_PI); if (d <= -M_PI) d += 2 * M_PI;
                const double q = (cd[i] - 1) / 18.0 - sgn * (tr + d) / (2 * M_PI);
                hist[((long)nearbyint(q) % 14 + 14) % 14]++;
            }
            dom = 0; for (int k = 1; k < 14; k++) if (hist[k] > hist[dom]) dom = k;
            pur = (double)hist[dom] / (double)nh;
            pw += pur * (double)nh; pwh += (double)nh; npw++; nlow += pur < 0.95;
        }
        free(th); free(cd);
        sum_near += (double)near; sum_hit += (double)hit;
        if (tf) fprintf(tf, "%s\t%zu\t%ld\t%ld\t%.4f\t%d\n", names[pi], nv, near, hit, pur, dom);
    }
    if (tf) fclose(tf);
    /* recall: GT voxels within Chebyshev 1 of the cover */
    size_t rec = 0;
    for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < NY; y++) for (int64_t x = 0; x < NX; x++) {
        if (!gs[IDX(z, y, x)]) continue;
        int h = 0;
        for (int dz = -1; dz <= 1 && !h; dz++) for (int dy = -1; dy <= 1 && !h; dy++) for (int dx = -1; dx <= 1 && !h; dx++) {
            const int64_t zz = z + dz, yy = y + dy, xx = x + dx; if (zz < 0 || yy < 0 || xx < 0 || zz >= n[0] || yy >= NY || xx >= NX) continue;
            h = cover[IDX(zz, yy, xx)];
        }
        rec += (size_t)h;
    }
    printf("patches %d | recall %.4f | precision %.4f (%.0f of %.0f voxels near GT) | purity (matched-weighted) %.4f | patches with purity < .95: %d of %d\n",
           npatch, ngs ? (double)rec / (double)ngs : 0, sum_near > 0 ? sum_hit / sum_near : 0, sum_hit, sum_near, pwh > 0 ? pw / pwh : 0, nlow, npw);
    free(vk); free(names); free(gt); free(gs); free(d8); free(cover);
    return 0;
#undef IDX
}

/* ufsm raw2jpg IN.raw W H OUT.jpg [--rgb] [--mark Y,X]..: a uint8 W x H raw slice as a grey JPEG (QA), optional red crosses; --rgb: the raw is
   W x H interleaved RGB triples */
#include "imgops.h"
int cmd_raw2jpg(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: ufsm raw2jpg IN.raw W H OUT.jpg [--rgb] [--mark Y,X]..\n"); return 2; }
    const int W = atoi(argv[3]), H = atoi(argv[4]); if (W <= 0 || H <= 0) return 2;
    uint8_t *g = malloc((size_t)W * H), *rgb = malloc((size_t)W * H * 3); if (!g || !rgb) return 1;
    int rgbin = 0; for (int i = 6; i < argc; i++) rgbin |= !strcmp(argv[i], "--rgb");
    const size_t want = (size_t)W * H * (rgbin ? 3 : 1);
    FILE *f = fopen(argv[2], "rb"); const size_t got = f ? fread(rgbin ? rgb : g, 1, want, f) : 0; if (f) fclose(f);
    if (got != want) { fprintf(stderr, "raw2jpg: %s: expected %zu bytes\n", argv[2], want); return 1; }
    if (!rgbin) for (size_t i = 0; i < (size_t)W * H; i++) rgb[3 * i] = rgb[3 * i + 1] = rgb[3 * i + 2] = g[i];
    for (int i = 6; i + 1 < argc; i++) if (!strcmp(argv[i], "--mark")) {
        int y, x; if (sscanf(argv[i + 1], "%d,%d", &y, &x) != 2) continue;
        for (int d = -8; d <= 8; d++) {
            if (y >= 0 && y < H && x + d >= 0 && x + d < W) { uint8_t *p = rgb + 3 * ((size_t)y * W + x + d); p[0] = 255; p[1] = p[2] = 0; }
            if (x >= 0 && x < W && y + d >= 0 && y + d < H) { uint8_t *p = rgb + 3 * ((size_t)(y + d) * W + x); p[0] = 255; p[1] = p[2] = 0; }
        }
    }
    const int rc = jpeg_write_rgb(argv[5], W, H, rgb, 90); free(g); free(rgb); return rc ? 1 : 0;
}

/* ufsm gt-support --gt G --prob P --box Z0,Y0,X0,NZ,NY,NX --axis A [--axis-scale S] [--frame F] [--threads N]
   Do GT sheets sit on CT-visible sheets? For every GT sheet voxel (codes 1..252): the distance to the nearest radial ridge voxel of the
   prob (p >= 64 radial non-max; Chebyshev <= 1, <= 2, > 2) and the local GT spacing = the distance along the outward ray to the next GT
   sheet voxel of another code (0.5-voxel steps, up to 64). Report: support shares per spacing bin (< 8, 8-12, 12-16, 16-24, >= 24 voxels,
   none within 64). Low support at small spacing = GT sheets that do not follow the CT where sheets are packed (label noise for training). */
int cmd_gt_support(int argc, char **argv) {
    int64_t o[3], n[3];
    const char *gtp = tb_opt(argc, argv, "--gt", nullptr), *probp = tb_opt(argc, argv, "--prob", nullptr), *axp = tb_opt(argc, argv, "--axis", nullptr);
    const double axs = atof(tb_opt(argc, argv, "--axis-scale", "1"));
    int nt = atoi(tb_opt(argc, argv, "--threads", "0")); if (nt <= 0) nt = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (!gtp || !probp || !axp || tb_box(tb_opt(argc, argv, "--box", nullptr), o, n) || axs <= 0) {
        fprintf(stderr, "usage: ufsm gt-support --gt G [--gt-key K] --prob P [--prob-key K] --box Z0,Y0,X0,NZ,NY,NX --axis A [--axis-scale S] [--frame F]\n"); return 2;
    }
    ps_geo G; if (ps_geo_init(&G, axp, axs, o, n)) { fprintf(stderr, "gt-support: cannot load axis %s\n", axp); return 1; }
    const char *fr = tb_opt(argc, argv, "--frame", nullptr); if (fr && ps_frame_load(&G, fr)) { fprintf(stderr, "gt-support: cannot read %s\n", fr); return 1; }
    uint8_t *gt = ps_box(gtp, tb_opt(argc, argv, "--gt-key", nullptr), tb_opt(argc, argv, "--cache", nullptr), o, n, nt);
    uint8_t *pb = ps_box(probp, tb_opt(argc, argv, "--prob-key", nullptr), tb_opt(argc, argv, "--cache", nullptr), o, n, nt);
    if (!gt || !pb) return 1;
    const size_t N = (size_t)(n[0] * n[1] * n[2]); const int64_t NY = n[1], NX = n[2];
#define IDX(z, y, x) (((size_t)(z) * NY + (size_t)(y)) * NX + (size_t)(x))
    uint8_t *rg = malloc(N); if (!rg) return 1;
    ps_ridge(pb, rg, n, &G, nt); free(pb);
    long long cnt[6][3] = {{0}};   /* spacing bin x support (<= 1, <= 2, > 2) */
    const double edges[5] = {8, 12, 16, 24, 1e9};
    for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < NY; y++) for (int64_t x = 0; x < NX; x++) {
        const int c = gt[IDX(z, y, x)]; if (c < 1 || c > 252) continue;
        int d = 3;
        for (int r = 1; r <= 2 && d == 3; r++)
            for (int dz = -r; dz <= r && d == 3; dz++) for (int dy = -r; dy <= r && d == 3; dy++) for (int dx = -r; dx <= r && d == 3; dx++) {
                const int64_t zz = z + dz, yy = y + dy, xx = x + dx; if (zz < 0 || yy < 0 || xx < 0 || zz >= n[0] || yy >= NY || xx >= NX) continue;
                if (rg[IDX(zz, yy, xx)]) d = r;
            }
        double uy, ux; ps_ray_dir(&G, (double)z, (double)y, (double)x, &uy, &ux);
        double sp = -1; int left = 0;
        for (double t = 0.5; t <= 64; t += 0.5) {
            const int64_t yy = (int64_t)lround(y + t * uy), xx = (int64_t)lround(x + t * ux); if (yy < 0 || xx < 0 || yy >= NY || xx >= NX) break;
            const int g = gt[IDX(z, yy, xx)]; const int sheet = g >= 1 && g <= 252;
            int dc = sheet ? abs(g - c) % 252 : 0; dc = dc > 126 ? 252 - dc : dc;   /* codes run along a sheet with theta: another sheet = >= half a turn apart */
            if (!sheet || dc < 9) { left = 1; continue; }   /* must leave this sheet first */
            if (left) { sp = t; break; }
        }
        int b = 5; if (sp > 0) for (b = 0; b < 4 && sp >= edges[b]; b++);
        cnt[b][d <= 1 ? 0 : d == 2 ? 1 : 2]++;
    }
    const char *bn[6] = {"< 8", "8-12", "12-16", "16-24", ">= 24", "none <= 64"};
    long long tot[3] = {0}; printf("gt-support: GT sheet voxels on the prob ridge (Chebyshev <= 1 / <= 2 / > 2) by GT spacing along the ray (voxels)\n");
    for (int b = 0; b < 6; b++) {
        const long long s = cnt[b][0] + cnt[b][1] + cnt[b][2]; for (int k = 0; k < 3; k++) tot[k] += cnt[b][k];
        printf("  spacing %-11s n %10lld  on ridge %.3f  <= 2 %.3f  unsupported %.3f\n", bn[b], s, s ? (double)cnt[b][0] / s : 0, s ? (double)(cnt[b][0] + cnt[b][1]) / s : 0, s ? (double)cnt[b][2] / s : 0);
    }
    const long long s = tot[0] + tot[1] + tot[2];
    printf("  all                n %10lld  on ridge %.3f  <= 2 %.3f  unsupported %.3f\n", s, s ? (double)tot[0] / s : 0, s ? (double)(tot[0] + tot[1]) / s : 0, s ? (double)tot[2] / s : 0);
    free(gt); free(rg); return 0;
#undef IDX
}

/* ufsm gt-gate --gt ROOT --gt-key K --prob ROOT --prob-key K --axis A [--axis-scale S] --out DIR [--um U] [--cell 64] [--min-support 0.75]
                [--min-n 200] [--box Z0,Y0,X0,NZ,NY,NX] [--jobs 8] [--cache C]
   A training-label gate for GT whose sheets do not follow the CT: per --cell^3 cell the share of its GT sheet voxels (codes 1..252) within
   Chebyshev 2 of the radial prob ridge (gt-support); cells with >= --min-n sheet voxels and a share < --min-support are 255 in the output mask
   (uint8 zarr on the GT grid, 512^3 shards, written at DIR/<um>; use as a stored phase's "ignore"), the rest 0. Shards without GT sheets
   are skipped (fill 0). Prints the share of GT sheet voxels gated. */
#include "z3w.h"
#include "band.h"
#include <pthread.h>
#include <stdatomic.h>
typedef struct {
    z3 *gt, *pb, *ph, *ig; z3w *w, *wp; const char *axp; double axs; int64_t shape[3], b0[3], bn[3], ns[3]; int cell, minn, skip, supr, interval; double mins, shfrac, minkeep, odeg; _Atomic long long pshards, ocells;
    _Atomic long next; long nitems; _Atomic long long sheet, gated, cells, gcells; _Atomic long long npar, nfr; const char *dir; _Atomic int fail;
} gg_job;
static void *gg_work(void *arg) {
    gg_job *J = arg; const int S = 512, H = J->skip || J->interval ? 40 : 3;
    uint8_t *gt = malloc((size_t)(S + 2 * H) * (S + 2 * H) * (S + 2 * H)), *pb = malloc((size_t)(S + 2 * H) * (S + 2 * H) * (S + 2 * H)),
            *rg = malloc((size_t)(S + 2 * H) * (S + 2 * H) * (S + 2 * H)), *out = malloc((size_t)S * S * S);
    uint8_t *phb = J->odeg > 0 ? malloc((size_t)(S + 2 * H) * (S + 2 * H) * (S + 2 * H)) : nullptr;   /* --orient-deg: our phase with the halo */
    for (;;) {
        const long it = atomic_fetch_add(&J->next, 1); if (it >= J->nitems) break;
        const int64_t si[3] = {it / (J->ns[1] * J->ns[2]), it / J->ns[2] % J->ns[1], it % J->ns[2]};
        int64_t s0[3], o[3], n[3], c0[3], cn[3];
        for (int d = 0; d < 3; d++) {
            s0[d] = si[d] * S;   /* shard origin (array) */
            int64_t lo = s0[d] > J->b0[d] ? s0[d] : J->b0[d], hi = s0[d] + S < J->b0[d] + J->bn[d] ? s0[d] + S : J->b0[d] + J->bn[d]; if (hi > J->shape[d]) hi = J->shape[d];
            c0[d] = lo; cn[d] = hi - lo;   /* core: the box part of the shard */
            o[d] = lo - H < 0 ? 0 : lo - H; n[d] = (hi + H > J->shape[d] ? J->shape[d] : hi + H) - o[d];
        }
        if (cn[0] <= 0 || cn[1] <= 0 || cn[2] <= 0) continue;
        if (z3_read(J->gt, o, n, gt, 1)) { atomic_store(&J->fail, 1); continue; }
        const size_t NN = (size_t)(n[0] * n[1] * n[2]); size_t ns_ = 0; for (size_t i = 0; i < NN; i++) ns_ += gt[i] >= 1 && gt[i] <= 252;
        if (!ns_) continue;   /* (the gated phase leaves it unknown: no labelled sheet in reach) */
        if (z3_read(J->pb, o, n, pb, 1)) { atomic_store(&J->fail, 1); continue; }
        ps_geo G; if (ps_geo_init(&G, J->axp, J->axs, o, n)) { atomic_store(&J->fail, 1); continue; }
        if (strcmp(J->dir, "radial")) {   /* --dir auto: parallel rays where the shard's sheets have one clear orientation off radial (crushed scrolls; sheet-trace's rule) */
            double jxx = 0, jxy = 0, jyy = 0;
            for (int64_t z = 0; z < n[0]; z += 16) { const uint8_t *sl = pb + (size_t)z * n[1] * n[2];
                for (int64_t y = 1; y + 1 < n[1]; y += 2) for (int64_t x = 1; x + 1 < n[2]; x += 2) {
                    const double gx = (double)sl[y * n[2] + x + 1] - sl[y * n[2] + x - 1], gy = (double)sl[(y + 1) * n[2] + x] - sl[(y - 1) * n[2] + x];
                    jxx += gx * gx; jxy += gx * gy; jyy += gy * gy; } }
            const double phi = 0.5 * atan2(2 * jxy, jxx - jyy), coh = (jxx + jyy) > 0 ? sqrt((jxx - jyy) * (jxx - jyy) + 4 * jxy * jxy) / (jxx + jyy) : 0;
            double ny_ = sin(phi), nx_ = cos(phi), ay, ax; ps_axis(&G, n[0] / 2.0, &ay, &ax);
            const double cy = 0.5 * n[1], cx = 0.5 * n[2], ey = cy - ay, ex = cx - ax, el = hypot(ey, ex) + 1e-9;
            if (ny_ * ey + nx_ * ex < 0) { ny_ = -ny_; nx_ = -nx_; }
            const int axis_in = ay >= 0 && ax >= 0 && ay < n[1] && ax < n[2];
            G.par = !strcmp(J->dir, "par") || (!axis_in && (ny_ * ey + nx_ * ex) / el < 0.7 && coh >= 0.5);
            if (G.par) { G.ny = ny_; G.nx = nx_; G.ty = nx_; G.tx = -ny_; G.cy = cy; G.cx = cx; }
            atomic_fetch_add(&J->npar, G.par); atomic_fetch_add(&J->nfr, 1);
        }
        ps_ridge(pb, rg, n, &G, 1);
        const int C = J->cell; const int64_t nc[3] = {(S + C - 1) / C, (S + C - 1) / C, (S + C - 1) / C};
        long *cs = calloc((size_t)(nc[0] * nc[1] * nc[2]) * 2, sizeof *cs);
        #define GI(z, y, x) (((size_t)(z) * n[1] + (size_t)(y)) * n[2] + (size_t)(x))
        for (int64_t z = c0[0]; z < c0[0] + cn[0]; z++) for (int64_t y = c0[1]; y < c0[1] + cn[1]; y++) for (int64_t x = c0[2]; x < c0[2] + cn[2]; x++) {
            const int64_t lz = z - o[0], ly = y - o[1], lx = x - o[2]; const uint8_t c = gt[GI(lz, ly, lx)]; if (c < 1 || c > 252) continue;
            int sup = 0;
            if (J->interval) {   /* --interval: exactly one CT ridge run between this sheet (t = 0) and the next labelled sheet outward. A constant
                                    offset of our sheets from the ridges (recto face vs sheet middle) keeps one run per interval; 0 = merged
                                    sheets, 2+ = a skipped sheet. Voxels without a next sheet within 36 voxels are left out of the cell share. */
                double uy, ux; ps_ray_dir(&G, (double)lz, (double)ly, (double)lx, &uy, &ux); int prev = 0, runs = 0, found = 0, left = 0;
                for (double t = 0; t <= 36; t += 0.5) {
                    const int64_t yy = (int64_t)lround(ly + t * uy), xx = (int64_t)lround(lx + t * ux); if (yy < 0 || xx < 0 || yy >= n[1] || xx >= n[2]) break;
                    const uint8_t g = gt[GI(lz, yy, xx)]; const int sheet = g >= 1 && g <= 252; int dc = sheet ? abs((int)g - (int)c) % 252 : 0; dc = dc > 126 ? 252 - dc : dc;
                    if (!sheet || dc >= 9) left = 1;
                    if (sheet && dc >= 9 && left) { found = 1; break; }
                    const int r_ = rg[GI(lz, yy, xx)] != 0; runs += r_ && !prev; prev = r_;
                }
                if (!found) continue;
                sup = runs == 1;
            }
            const int SR = J->interval ? -1 : J->supr;
            for (int dz = -SR; dz <= SR && !sup; dz++) for (int dy = -SR; dy <= SR && !sup; dy++) for (int dx = -SR; dx <= SR && !sup; dx++) {
                const int64_t zz = lz + dz, yy = ly + dy, xx = lx + dx; if (zz < 0 || yy < 0 || xx < 0 || zz >= n[0] || yy >= n[1] || xx >= n[2]) continue;
                sup = rg[GI(zz, yy, xx)];
            }
            if (sup && J->skip && !J->interval) {   /* --skip-check: a CT sheet (ridge run) strictly between this sheet and the next labelled one outward = a skipped sheet */
                double uy, ux; ps_ray_dir(&G, (double)lz, (double)ly, (double)lx, &uy, &ux); int left = 0, prev = 1, nrs = 0, found = 0; double t0 = -1, rs[16];
                double t = 0.5;
                for (; t <= 36; t += 0.5) {
                    const int64_t yy = (int64_t)lround(ly + t * uy), xx = (int64_t)lround(lx + t * ux); if (yy < 0 || xx < 0 || yy >= n[1] || xx >= n[2]) break;
                    const uint8_t g = gt[GI(lz, yy, xx)]; const int sheet = g >= 1 && g <= 252; int dc = sheet ? abs((int)g - (int)c) % 252 : 0; dc = dc > 126 ? 252 - dc : dc;
                    if (sheet && dc >= 9 && left) { found = 1; break; }   /* the next labelled sheet */
                    if (!sheet || dc >= 9) { if (!left) t0 = t; left = 1; }
                    const int r_ = rg[GI(lz, yy, xx)]; if (left && t > t0 + 2.5 && r_ && !prev && nrs < 16) rs[nrs++] = t; prev = r_;
                }
                int runs = 0; if (found) for (int q = 0; q < nrs; q++) runs += rs[q] < t - 2.5;   /* runs well clear of both sheets */
                if (runs > 0) sup = 0;
            }
            const size_t ci = ((size_t)((z - s0[0]) / C) * nc[1] + (size_t)((y - s0[1]) / C)) * nc[2] + (size_t)((x - s0[2]) / C);
            cs[2 * ci]++; cs[2 * ci + 1] += sup;
        }
        double *jt = nullptr;   /* --orient-deg A: per cell the structure tensors of the m7 prob (CT sheets) and of our phase (our sheets): a cell whose
                                   sheet normals differ by more than A degrees is bad (sheets veering off the CT direction), whatever the support says */
        if (J->odeg > 0) {
            if (z3_read(J->ph, o, n, phb, 1)) { atomic_store(&J->fail, 1); free(cs); continue; }
            jt = calloc((size_t)(nc[0] * nc[1] * nc[2]) * 12, sizeof *jt);
            for (int64_t z = c0[0]; z < c0[0] + cn[0]; z += 2) for (int64_t y = c0[1]; y < c0[1] + cn[1]; y += 2) for (int64_t x = c0[2]; x < c0[2] + cn[2]; x += 2) {
                const int64_t lz = z - o[0], ly = y - o[1], lx = x - o[2];
                if (lz < 1 || ly < 1 || lx < 1 || lz + 1 >= n[0] || ly + 1 >= n[1] || lx + 1 >= n[2]) continue;
                const size_t ci = ((size_t)((z - s0[0]) / C) * nc[1] + (size_t)((y - s0[1]) / C)) * nc[2] + (size_t)((x - s0[2]) / C);
                double *a = jt + 12 * ci;
                const double gz = (double)pb[GI(lz + 1, ly, lx)] - pb[GI(lz - 1, ly, lx)], gy = (double)pb[GI(lz, ly + 1, lx)] - pb[GI(lz, ly - 1, lx)],
                             gx = (double)pb[GI(lz, ly, lx + 1)] - pb[GI(lz, ly, lx - 1)];
                a[0] += gz * gz; a[1] += gy * gy; a[2] += gx * gx; a[3] += gz * gy; a[4] += gz * gx; a[5] += gy * gx;
                const uint8_t pz0 = phb[GI(lz - 1, ly, lx)], pz1 = phb[GI(lz + 1, ly, lx)], py0 = phb[GI(lz, ly - 1, lx)], py1 = phb[GI(lz, ly + 1, lx)],
                              px0 = phb[GI(lz, ly, lx - 1)], px1 = phb[GI(lz, ly, lx + 1)];
                if (pz0 >= 252 || pz1 >= 252 || py0 >= 252 || py1 >= 252 || px0 >= 252 || px1 >= 252) continue;
                #define CD(p1, p0) ((double)((((int)(p1) - (int)(p0)) % 252 + 378) % 252 - 126))
                const double qz = CD(pz1, pz0), qy = CD(py1, py0), qx = CD(px1, px0);
                #undef CD
                a[6] += qz * qz; a[7] += qy * qy; a[8] += qx * qx; a[9] += qz * qy; a[10] += qz * qx; a[11] += qy * qx;
            }
        }
        memset(out, 0, (size_t)S * S * S); long long g = 0, sh = 0, cl = 0, gc = 0;
        for (int64_t cz = 0; cz < nc[0]; cz++) for (int64_t cy = 0; cy < nc[1]; cy++) for (int64_t cx = 0; cx < nc[2]; cx++) {
            const size_t ci = ((size_t)cz * nc[1] + (size_t)cy) * nc[2] + (size_t)cx; const long tot = cs[2 * ci]; sh += tot;
            if (tot < J->minn) continue;
            cl++;
            if (jt) {   /* the dominant eigenvectors (power iteration) of both tensors; gate when they disagree by more than --orient-deg */
                const double *a = jt + 12 * ci; double v[2][3];
                for (int k = 0; k < 2; k++) {
                    const double *m = a + 6 * k; double e[3] = {0.577, 0.577, 0.577};
                    for (int it = 0; it < 30; it++) {
                        double f[3] = {m[0] * e[0] + m[3] * e[1] + m[4] * e[2], m[3] * e[0] + m[1] * e[1] + m[5] * e[2], m[4] * e[0] + m[5] * e[1] + m[2] * e[2]};
                        const double l = sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]); if (l <= 0) break; for (int q = 0; q < 3; q++) e[q] = f[q] / l;
                    }
                    memcpy(v[k], e, sizeof e);
                }
                const double tr = a[6] + a[7] + a[8], dt = fabs(v[0][0] * v[1][0] + v[0][1] * v[1][1] + v[0][2] * v[1][2]);
                if (tr > 0 && a[0] + a[1] + a[2] > 0 && dt < cos(J->odeg * M_PI / 180)) {
                    atomic_fetch_add(&J->ocells, 1); gc++; g += tot;
                    for (int64_t z = cz * C; z < (cz + 1) * C && z < S; z++) for (int64_t y = cy * C; y < (cy + 1) * C && y < S; y++)
                        memset(out + ((size_t)z * S + (size_t)y) * S + (size_t)(cx * C), 255, (size_t)(C < S - cx * C ? C : S - cx * C));
                    continue;
                }
            }
            if ((double)cs[2 * ci + 1] / (double)tot >= J->mins) continue;
            gc++; g += tot;
            for (int64_t z = cz * C; z < (cz + 1) * C && z < S; z++) for (int64_t y = cy * C; y < (cy + 1) * C && y < S; y++)
                memset(out + ((size_t)z * S + (size_t)y) * S + (size_t)(cx * C), 255, (size_t)(C < S - cx * C ? C : S - cx * C));
        }
        #undef GI
        free(cs); free(jt);
        atomic_fetch_add(&J->sheet, sh); atomic_fetch_add(&J->gated, g); atomic_fetch_add(&J->cells, cl); atomic_fetch_add(&J->gcells, gc);
        if (gc && z3w_write_shard(J->w, si[0], si[1], si[2], out, 1)) atomic_store(&J->fail, 1);
        uint64_t hh = (uint64_t)it * 0x9E3779B97F4A7C15ull; hh ^= hh >> 31; hh *= 0xBF58476D1CE4E5B9ull; hh ^= hh >> 29;
        if (J->wp && sh > 0 && (double)(sh - g) / (double)sh >= J->minkeep && (double)(hh >> 11) / 9007199254740992.0 < J->shfrac) {   /* --phase: the stored phase of this shard with gated cells (and --ignore) unknown; only shards keeping >= --min-keep of their sheet voxels, a --shard-frac hash subsample (disk) */
            atomic_fetch_add(&J->pshards, 1);
            int64_t sn[3]; for (int d = 0; d < 3; d++) sn[d] = J->shape[d] - s0[d] < S ? J->shape[d] - s0[d] : S;   /* (edge shards: clipped read, unknown beyond) */
            if (z3_read(J->ph, s0, sn, gt, 1) || (J->ig && z3_read(J->ig, s0, sn, rg, 1))) { atomic_store(&J->fail, 1); continue; }
            memset(pb, PHASE_UNKNOWN, (size_t)S * S * S);
            for (int64_t z = 0; z < sn[0]; z++) for (int64_t y = 0; y < sn[1]; y++) for (int64_t x = 0; x < sn[2]; x++) {
                const size_t i = ((size_t)z * S + (size_t)y) * S + (size_t)x, j = ((size_t)z * sn[1] + (size_t)y) * sn[2] + (size_t)x;
                pb[i] = out[i] || (J->ig && rg[j]) ? PHASE_UNKNOWN : gt[j];
            }
            if (z3w_write_shard(J->wp, si[0], si[1], si[2], pb, 1)) atomic_store(&J->fail, 1);
        }
    }
    free(gt); free(pb); free(rg); free(out); free(phb); return nullptr;
}
int cmd_gt_gate(int argc, char **argv) {
    const char *gtr = tb_opt(argc, argv, "--gt", nullptr), *gtk = tb_opt(argc, argv, "--gt-key", nullptr), *pr = tb_opt(argc, argv, "--prob", nullptr),
               *pk = tb_opt(argc, argv, "--prob-key", nullptr), *axp = tb_opt(argc, argv, "--axis", nullptr), *out = tb_opt(argc, argv, "--out", nullptr), *cache = tb_opt(argc, argv, "--cache", nullptr);
    gg_job J = {.axp = axp, .axs = atof(tb_opt(argc, argv, "--axis-scale", "1")), .cell = atoi(tb_opt(argc, argv, "--cell", "64")), .minn = atoi(tb_opt(argc, argv, "--min-n", "200")),
                .mins = atof(tb_opt(argc, argv, "--min-support", "0.75")), .skip = atoi(tb_opt(argc, argv, "--skip-check", "0")), .supr = atoi(tb_opt(argc, argv, "--sup-r", "2")), .dir = tb_opt(argc, argv, "--dir", "auto"), .odeg = atof(tb_opt(argc, argv, "--orient-deg", "0")), .interval = atoi(tb_opt(argc, argv, "--interval", "0"))};
    const int jobs = atoi(tb_opt(argc, argv, "--jobs", "8"));
    if (!gtr || !gtk || !pr || !pk || !axp || !out || J.axs <= 0 || J.cell < 8 || J.cell > 512 || 512 % J.cell || jobs < 1 || jobs > 64) {
        fprintf(stderr, "usage: ufsm gt-gate --gt ROOT --gt-key K --prob ROOT --prob-key K --axis A [--axis-scale S] --out DIR [--um U] [--cell 64 (divides 512)]\n"
                        "         [--min-support 0.75] [--min-n 200] [--box Z0,Y0,X0,NZ,NY,NX] [--jobs 8] [--cache C]\n"); return 2;
    }
    store *gs = store_open(gtr), *ps = store_open(pr); J.gt = gs ? z3_open(gs, gtk, cache) : nullptr; J.pb = ps ? z3_open(ps, pk, cache) : nullptr;
    if (!J.gt || !J.pb) { fprintf(stderr, "gt-gate: cannot open %s/%s or %s/%s\n", gtr, gtk, pr, pk); return 1; }
    const z3_meta *gm = z3_meta_of(J.gt), *pm = z3_meta_of(J.pb);
    if (memcmp(gm->shape, pm->shape, sizeof gm->shape)) { fprintf(stderr, "gt-gate: the GT and prob shapes differ\n"); return 1; }
    memcpy(J.shape, gm->shape, sizeof J.shape);
    const char *bs = tb_opt(argc, argv, "--box", nullptr);
    if (bs) { if (tb_box(bs, J.b0, J.bn)) return 2; } else for (int d = 0; d < 3; d++) { J.b0[d] = 0; J.bn[d] = J.shape[d]; }
    for (int d = 0; d < 3; d++) J.ns[d] = (J.shape[d] + 511) / 512;
    J.nitems = J.ns[0] * J.ns[1] * J.ns[2];
    const double um = atof(tb_opt(argc, argv, "--um", "0")) > 0 ? atof(tb_opt(argc, argv, "--um", "0")) : (gm->scale_um > 0 ? gm->scale_um : 1);
    char key[64], dir[1200]; z3w_level_name(um, key, sizeof key); snprintf(dir, sizeof dir, "%s/%s", out, key);
    mkdir(out, 0755); J.w = z3w_create(dir, J.shape, 512, 0, 0, nullptr); if (!J.w) { fprintf(stderr, "gt-gate: cannot create %s: %s\n", dir, z3w_error()); return 1; }
    const char *phr = tb_opt(argc, argv, "--phase", nullptr), *phk = tb_opt(argc, argv, "--phase-key", nullptr), *pho = tb_opt(argc, argv, "--phase-out", nullptr);
    J.shfrac = atof(tb_opt(argc, argv, "--shard-frac", "1")); J.minkeep = atof(tb_opt(argc, argv, "--min-keep", "0.3"));
    const double prm = atof(tb_opt(argc, argv, "--phase-rmax", "0"));   /* > 0: --phase / --phase-key name winding_mod14 codes, the phase their pair rule at this radius
                                                                           (surf-labels --radius, 20; label sets kept as codes only, 2026-11-13) */
    if (phr && phk) { store *qs = store_open(phr); J.ph = qs ? z3_open(qs, phk, nullptr) : nullptr; if (J.ph && prm > 0) J.ph = phase_onfly_open(J.ph, (float)prm);
        if (!J.ph || memcmp(z3_meta_of(J.ph)->shape, J.shape, sizeof J.shape)) { fprintf(stderr, "gt-gate: cannot open %s/%s or its shape differs\n", phr, phk); return 1; } }
    if (J.odeg > 0 && !J.ph) { fprintf(stderr, "gt-gate: --orient-deg needs --phase / --phase-key (our sheet directions)\n"); return 2; }
    if (phr && phk && pho) {   /* the gated phase (PHASE_* codes, unknown 255), lossless */
        const char *igr = tb_opt(argc, argv, "--ignore", nullptr), *igk = tb_opt(argc, argv, "--ignore-key", nullptr);
        if (igr && igk) { store *is = store_open(igr); J.ig = is ? z3_open(is, igk, nullptr) : nullptr; if (!J.ig) { fprintf(stderr, "gt-gate: cannot open %s/%s\n", igr, igk); return 1; } }
        if (!J.ph || memcmp(z3_meta_of(J.ph)->shape, J.shape, sizeof J.shape)) { fprintf(stderr, "gt-gate: cannot open %s/%s or its shape differs\n", phr, phk); return 1; }
        char pd[1200]; mkdir(pho, 0755); snprintf(pd, sizeof pd, "%s/%s", pho, key);
        J.wp = z3w_create(pd, J.shape, 512, 0, PHASE_UNKNOWN, nullptr);
        if (!J.wp) { fprintf(stderr, "gt-gate: cannot create %s: %s\n", pd, z3w_error()); return 1; }
    }
    pthread_t th[64]; for (int t = 0; t < jobs; t++) pthread_create(&th[t], nullptr, gg_work, &J); for (int t = 0; t < jobs; t++) pthread_join(th[t], nullptr);
    if (J.wp && (z3w_close(J.wp) || z3w_write_group(tb_opt(argc, argv, "--phase-out", ""), &um, 1, "phase", nullptr))) { fprintf(stderr, "gt-gate: phase write failure\n"); return 1; }
    if (z3w_close(J.w) || z3w_write_group(out, &um, 1, "gt-gate", nullptr) || atomic_load(&J.fail)) { fprintf(stderr, "gt-gate: read / write failure\n"); return 1; }
    const long long sh = atomic_load(&J.sheet), g = atomic_load(&J.gated);
    printf("gt-gate: %lld GT sheet voxels, %lld cells with >= %d, %lld gated (support < %.2f) holding %.3f of the sheet voxels -> %s (key %s)\n",
           sh, (long long)atomic_load(&J.cells), J.minn, (long long)atomic_load(&J.gcells), J.mins, sh ? (double)g / sh : 0, out, key);
    if (J.odeg > 0) printf("gt-gate: --orient-deg %.0f: %lld cells gated for sheet direction (ours vs the m7 structure tensor)\n", J.odeg, (long long)atomic_load(&J.ocells));
    printf("gt-gate: ray frame --dir %s: %lld of %lld shards parallel\n", J.dir, (long long)atomic_load(&J.npar), (long long)atomic_load(&J.nfr));
    if (J.wp) printf("gt-gate: gated phase: %lld shards written (--min-keep %.2f, --shard-frac %.2f) -> %s\n", (long long)atomic_load(&J.pshards), J.minkeep, J.shfrac, tb_opt(argc, argv, "--phase-out", ""));
    z3_close(J.gt); z3_close(J.pb); return 0;
}

/* ufsm phase-diff ROOT_A KEY_A ROOT_B KEY_B --box Z0,Y0,X0,NZ,NY,NX [--ct ROOT KEY --iso 64] [--threads N]
   Histogram (20 bins over one turn, centred) of the phase difference A - B (PHASE_* codes 0..251, others unknown) where both are known
   (and CT >= iso with --ct): a constant offset (e.g. surfaces on different papyrus faces) vs a spread (noise / wrong turns). */
int cmd_phase_diff(int argc, char **argv) {
    int64_t o[3], n[3]; int nt = atoi(tb_opt(argc, argv, "--threads", "0")); if (nt <= 0) nt = 16;
    if (argc < 7 || tb_box(tb_opt(argc, argv, "--box", nullptr), o, n)) { fprintf(stderr, "usage: ufsm phase-diff ROOT_A KEY_A ROOT_B KEY_B --box Z0,Y0,X0,NZ,NY,NX [--ct ROOT KEY --iso 64]\n"); return 2; }
    uint8_t *a = ps_box(argv[2], argv[3], nullptr, o, n, nt), *b = ps_box(argv[4], argv[5], nullptr, o, n, nt); if (!a || !b) return 1;
    uint8_t *ct = nullptr; const int iso = atoi(tb_opt(argc, argv, "--iso", "64"));
    uint8_t *mk[4] = {nullptr}; int nmk = 0, mkeep[4] = {0};   /* --mask R K (skip where nonzero) / --keep R K (skip where zero) */
    for (int i = 6; i + 2 < argc; i++) {
        if (!strcmp(argv[i], "--ct")) { ct = ps_box(argv[i + 1], argv[i + 2], nullptr, o, n, nt); if (!ct) return 1; }
        if ((!strcmp(argv[i], "--mask") || !strcmp(argv[i], "--keep")) && nmk < 4) { mkeep[nmk] = !strcmp(argv[i], "--keep"); mk[nmk] = ps_box(argv[i + 1], argv[i + 2], nullptr, o, n, nt); if (!mk[nmk++]) return 1; }
    }
    const size_t N = (size_t)(n[0] * n[1] * n[2]); long long h[20] = {0}, tot = 0, side = 0;
    for (size_t i = 0; i < N; i++) {
        if (a[i] >= 252 || b[i] >= 252 || (ct && ct[i] < iso)) continue;
        int skip = 0; for (int m = 0; m < nmk; m++) skip |= mkeep[m] ? !mk[m][i] : mk[m][i] != 0; if (skip) continue;
        side += (a[i] < 126) == (b[i] < 126);   /* same half turn: recto (phase k .. k + 0.5) / verso */
        int d = (int)a[i] - (int)b[i]; d = ((d % 252) + 252) % 252; if (d >= 126) d -= 252;   /* -126..125 */
        int k = (int)floor((d + 126) / 12.6); k = k < 0 ? 0 : k > 19 ? 19 : k; h[k]++; tot++;
    }
    printf("phase-diff: %lld voxels with both phases%s; side agreement %.4f; A - B in turns (bin centre: share)\n", tot, ct ? " on papyrus" : "", tot ? (double)side / tot : 0);
    for (int k = 0; k < 20; k++) printf("  %+.3f %.3f %s\n", -0.475 + 0.05 * k, tot ? (double)h[k] / tot : 0, "");
    free(a); free(b); free(ct); for (int m = 0; m < nmk; m++) free(mk[m]); return 0;
}
