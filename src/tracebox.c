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

/* ufsm raw2jpg IN.raw W H OUT.jpg [--mark Y,X]..: a uint8 W x H raw slice as a grey JPEG (QA), optional red crosses */
#include "imgops.h"
int cmd_raw2jpg(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: ufsm raw2jpg IN.raw W H OUT.jpg [--mark Y,X]..\n"); return 2; }
    const int W = atoi(argv[3]), H = atoi(argv[4]); if (W <= 0 || H <= 0) return 2;
    uint8_t *g = malloc((size_t)W * H), *rgb = malloc((size_t)W * H * 3); if (!g || !rgb) return 1;
    FILE *f = fopen(argv[2], "rb"); const size_t got = f ? fread(g, 1, (size_t)W * H, f) : 0; if (f) fclose(f);
    if (got != (size_t)W * H) { fprintf(stderr, "raw2jpg: %s: expected %d bytes\n", argv[2], W * H); return 1; }
    for (size_t i = 0; i < (size_t)W * H; i++) rgb[3 * i] = rgb[3 * i + 1] = rgb[3 * i + 2] = g[i];
    for (int i = 6; i + 1 < argc; i++) if (!strcmp(argv[i], "--mark")) {
        int y, x; if (sscanf(argv[i + 1], "%d,%d", &y, &x) != 2) continue;
        for (int d = -8; d <= 8; d++) {
            if (y >= 0 && y < H && x + d >= 0 && x + d < W) { uint8_t *p = rgb + 3 * ((size_t)y * W + x + d); p[0] = 255; p[1] = p[2] = 0; }
            if (x >= 0 && x < W && y + d >= 0 && y + d < H) { uint8_t *p = rgb + 3 * ((size_t)(y + d) * W + x); p[0] = 255; p[1] = p[2] = 0; }
        }
    }
    const int rc = jpeg_write_rgb(argv[5], W, H, rgb, 90); free(g); free(rgb); return rc ? 1 : 0;
}
