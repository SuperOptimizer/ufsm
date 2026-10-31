/* ufsm patch-raster LIST OUT --shape Z,Y,X [--um 9.6] [--code 0] [--skip-z Z0,Z1 ...] [--threads 16] [--check ROOT KEY] [--check-only 1] [--zrow Z0,Z1]
   Surface patches (tifxyz: x / y / z images in voxels of the output grid, invalid < 0; e.g. the ScrollPrize spiral-input verified
   patches of PHercParis4, 9.6 um = level 2 of the 2.4 um scan) as stored-phase training targets (src/band.h PHASE_* codes): every voxel
   the surface passes through gets --code (0: a sheet's recto face), everything else 255 (unknown). Each quad of a patch is densified
   at <= 0.5 voxel and its samples rounded to voxels (a ~1 voxel shell). --skip-z Z0,Z1 (repeatable): voxels with z in [Z0, Z1) are left
   out (eval rows). Output: OUT/<um level>/ (512^3 shards, only shards with labels written). --check ROOT KEY: a stored phase raster on
   the same grid (e.g. the GT phaseA 9.6): the histogram of its codes at the shell voxels (18 bins per turn), its circular mean and the
   share within 1/8 and 1/4 turn of it (where the patches sit in the GT's phase convention). --check-only 1 writes nothing.
   LIST: one patch directory per line. */
#include <dirent.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "store.h"
#include "sysinfo.h"
#include "tiff.h"
#include "tiffw.h"
#include "zarr3.h"
#include "z3w.h"

static const char *pr_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static float *pr_tif(const char *path, int *w, int *h) {
    if (tfx_axis_hook && access(path, F_OK)) return tfx_axis_hook(path, w, h);
    tiff *t = tiff_open_file(path); tiff_page pg;
    if (!t || tiff_page_info(t, 0, &pg) || pg.fmt != 3 || pg.bits != 32 || pg.spp != 1) { if (t) tiff_close(t); return nullptr; }
    float *d = malloc((size_t)pg.w * pg.h * 4);
    if (!d || tiff_read_page(t, 0, d)) { free(d); tiff_close(t); return nullptr; }
    *w = pg.w; *h = pg.h; tiff_close(t); return d;
}
/* voxel key: shard (z, y, x) / 512 in the high bits, the voxel inside the shard in the low 27 */
static inline uint64_t pr_key(int64_t z, int64_t y, int64_t x) {
    return ((uint64_t)(z >> 9) << 49) | ((uint64_t)(y >> 9) << 38) | ((uint64_t)(x >> 9) << 27) | ((uint64_t)(z & 511) << 18) | ((uint64_t)(y & 511) << 9) | (uint64_t)(x & 511);
}
static int pr_cmp(const void *a, const void *b) { const uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return (x > y) - (x < y); }

typedef struct {
    char (*dirs)[1024]; int n; _Atomic int next; int64_t shp[3]; int nsk; int64_t sk[16][2]; int64_t zr[2];
    uint64_t **out; size_t *nout, *cap; _Atomic long long nbad, nquad;
} pr_job;
static void pr_push(uint64_t **v, size_t *n, size_t *cap, uint64_t k) {
    if (*n == *cap) { *cap = *cap ? *cap * 2 : 1 << 16; *v = realloc(*v, *cap * sizeof **v); if (!*v) { fprintf(stderr, "patch-raster: out of memory\n"); exit(1); } }
    (*v)[(*n)++] = k;
}
typedef struct { pr_job *J; int t; } pr_arg;
static void *pr_work(void *a) {
    pr_arg *A = a; pr_job *J = A->J; uint64_t *loc = nullptr; size_t nl = 0, cl = 0;
    for (;;) {
        const int i = atomic_fetch_add(&J->next, 1); if (i >= J->n) break;
        char p[1100]; int w = 0, h = 0, w2, h2, w3, h3;
        snprintf(p, sizeof p, "%s/x.tif", J->dirs[i]); float *X = pr_tif(p, &w, &h);
        snprintf(p, sizeof p, "%s/y.tif", J->dirs[i]); float *Y = pr_tif(p, &w2, &h2);
        snprintf(p, sizeof p, "%s/z.tif", J->dirs[i]); float *Z = pr_tif(p, &w3, &h3);
        if (!X || !Y || !Z || w != w2 || w != w3 || h != h2 || h != h3) { atomic_fetch_add(&J->nbad, 1); free(X); free(Y); free(Z); continue; }
        nl = 0;
        for (int r = 0; r + 1 < h; r++) for (int c = 0; c + 1 < w; c++) {
            const size_t q[4] = {(size_t)r * w + c, (size_t)r * w + c + 1, (size_t)(r + 1) * w + c, (size_t)(r + 1) * w + c + 1};
            int ok = 1; for (int k = 0; k < 4; k++) ok &= X[q[k]] >= 0 && Y[q[k]] >= 0 && Z[q[k]] >= 0;
            if (!ok) continue;
            { float zl = Z[q[0]], zh = Z[q[0]]; for (int k = 1; k < 4; k++) { zl = fminf(zl, Z[q[k]]); zh = fmaxf(zh, Z[q[k]]); }
              if (zh + 1 < J->zr[0] || zl - 1 >= J->zr[1]) continue; }   /* (--zrow: quads off the row) */
            atomic_fetch_add(&J->nquad, 1);
            double L = 0;   /* the longest side / diagonal of the quad (voxels) */
            for (int a = 0; a < 4; a++) for (int b = a + 1; b < 4; b++) L = fmax(L, sqrt(pow(X[q[a]] - X[q[b]], 2) + pow(Y[q[a]] - Y[q[b]], 2) + pow(Z[q[a]] - Z[q[b]], 2)));
            if (L > 200) continue;   /* (a broken quad) */
            const int ns = (int)ceil(L / 0.5) + 1;
            for (int u = 0; u <= ns; u++) for (int v = 0; v <= ns; v++) {
                const double s = (double)u / ns, t = (double)v / ns, w00 = (1 - s) * (1 - t), w01 = s * (1 - t), w10 = (1 - s) * t, w11 = s * t;
                const double x = w00 * X[q[0]] + w01 * X[q[1]] + w10 * X[q[2]] + w11 * X[q[3]];
                const double y = w00 * Y[q[0]] + w01 * Y[q[1]] + w10 * Y[q[2]] + w11 * Y[q[3]];
                const double z = w00 * Z[q[0]] + w01 * Z[q[1]] + w10 * Z[q[2]] + w11 * Z[q[3]];
                const int64_t iz = (int64_t)floor(z + 0.5), iy = (int64_t)floor(y + 0.5), ix = (int64_t)floor(x + 0.5);
                if (iz < 0 || iy < 0 || ix < 0 || iz >= J->shp[0] || iy >= J->shp[1] || ix >= J->shp[2]) continue;
                int skip = iz < J->zr[0] || iz >= J->zr[1]; for (int k = 0; k < J->nsk; k++) skip |= iz >= J->sk[k][0] && iz < J->sk[k][1];
                if (!skip) pr_push(&loc, &nl, &cl, pr_key(iz, iy, ix));
            }
        }
        free(X); free(Y); free(Z);
        if (nl) {   /* this patch's voxels, unique */
            qsort(loc, nl, 8, pr_cmp); size_t m = 0; for (size_t k = 0; k < nl; k++) if (!m || loc[k] != loc[m - 1]) loc[m++] = loc[k];
            for (size_t k = 0; k < m; k++) pr_push(&J->out[A->t], &J->nout[A->t], &J->cap[A->t], loc[k]);
        }
    }
    free(loc); return nullptr;
}

int cmd_patch_raster(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm patch-raster LIST OUT --shape Z,Y,X [--um 9.6] [--code 0] [--skip-z Z0,Z1 ...] [--threads 16] [--check ROOT KEY] [--check-only 1]\n"); return 2; }
    const char *list = argv[2], *out = argv[3];
    pr_job J; memset(&J, 0, sizeof J);
    if (sscanf(pr_opt(argc, argv, "--shape", ""), "%lld,%lld,%lld", (long long *)&J.shp[0], (long long *)&J.shp[1], (long long *)&J.shp[2]) != 3) { fprintf(stderr, "patch-raster: --shape Z,Y,X needed\n"); return 2; }
    J.zr[0] = 0; J.zr[1] = J.shp[0];   /* --zrow Z0,Z1: only voxels with z in [Z0, Z1) (rows of shards per run: bounded memory) */
    if (*pr_opt(argc, argv, "--zrow", "") && sscanf(pr_opt(argc, argv, "--zrow", ""), "%lld,%lld", (long long *)&J.zr[0], (long long *)&J.zr[1]) != 2) { fprintf(stderr, "patch-raster: --zrow Z0,Z1\n"); return 2; }
    for (int i = 1; i + 1 < argc && J.nsk < 16; i++) if (!strcmp(argv[i], "--skip-z") && sscanf(argv[i + 1], "%lld,%lld", (long long *)&J.sk[J.nsk][0], (long long *)&J.sk[J.nsk][1]) == 2) J.nsk++;
    const double um = atof(pr_opt(argc, argv, "--um", "9.6")); const int code = atoi(pr_opt(argc, argv, "--code", "0")), nt0 = ufsm_threads(atoi(pr_opt(argc, argv, "--threads", "0")), 16);
    const int conly = atoi(pr_opt(argc, argv, "--check-only", "0"));
    FILE *f = fopen(list, "r"); if (!f) { fprintf(stderr, "patch-raster: cannot read %s\n", list); return 1; }
    int cap = 1024; J.dirs = malloc(cap * sizeof *J.dirs); char line[1100];
    while (fgets(line, sizeof line, f)) {
        size_t l = strlen(line); while (l && (line[l - 1] == '\n' || line[l - 1] == '/' || line[l - 1] == ' ')) line[--l] = 0;
        if (!l || l >= 1024) continue;
        if (J.n == cap) { cap *= 2; J.dirs = realloc(J.dirs, cap * sizeof *J.dirs); }
        memcpy(J.dirs[J.n++], line, l + 1);
    }
    fclose(f);
    const int nt = nt0 < 1 ? 1 : nt0 > 64 ? 64 : nt0;
    J.out = calloc(nt, sizeof *J.out); J.nout = calloc(nt, sizeof *J.nout); J.cap = calloc(nt, sizeof *J.cap);
    pthread_t th[64]; pr_arg A[64];
    for (int t = 0; t < nt; t++) { A[t] = (pr_arg){&J, t}; pthread_create(&th[t], nullptr, pr_work, &A[t]); }
    for (int t = 0; t < nt; t++) pthread_join(th[t], nullptr);
    size_t N = 0; for (int t = 0; t < nt; t++) N += J.nout[t];
    uint64_t *K = malloc((N ? N : 1) * 8); size_t o = 0;
    for (int t = 0; t < nt; t++) { memcpy(K + o, J.out[t], J.nout[t] * 8); o += J.nout[t]; free(J.out[t]); }
    qsort(K, N, 8, pr_cmp); size_t M = 0; for (size_t k = 0; k < N; k++) if (!M || K[k] != K[M - 1]) K[M++] = K[k];
    fprintf(stderr, "patch-raster: %d patches (%lld unreadable), %lld quads, %zu shell voxels (%zu before patch overlaps)\n", J.n, (long long)J.nbad, (long long)J.nquad, M, N);
    z3 *ck = nullptr; long long hist[18] = {0}, nck = 0;
    for (int i = 1; i + 2 < argc; i++) if (!strcmp(argv[i], "--check")) { ck = z3_open(store_open(argv[i + 1]), argv[i + 2], nullptr); if (!ck) { fprintf(stderr, "patch-raster: --check: %s\n", z3_error()); return 1; } }
    z3w *zw = nullptr;
    if (!conly) {
        char lv[32], ld[1400], at[300]; z3w_level_name(um, lv, sizeof lv); snprintf(ld, sizeof ld, "%s/%s", out, lv);
        snprintf(at, sizeof at, "{\"ufsm\":{\"content\":\"stored phase %d on surface-patch shells (ufsm patch-raster; uint8, 255 unknown; src/band.h PHASE_*)\"}}", code);
        zw = z3w_create(ld, J.shp, 512, 0.f, 255, at);
        if (!zw || z3w_write_group(out, (const double[]){um}, 1, "ufsm-phase", at)) { fprintf(stderr, "patch-raster: %s\n", z3w_error()); return 1; }
    }
    uint8_t *sb = malloc((size_t)512 * 512 * 512), *gb = ck ? malloc((size_t)512 * 512 * 512) : nullptr; long long nsh = 0;
    for (size_t a = 0; a < M;) {
        const uint64_t sh = K[a] >> 27; size_t b = a; while (b < M && K[b] >> 27 == sh) b++;
        const int64_t sz = (int64_t)(sh >> 22), sy = (int64_t)(sh >> 11 & 2047), sx = (int64_t)(sh & 2047);
        if (ck) {
            const int64_t g0[3] = {sz * 512, sy * 512, sx * 512}; int64_t gc[3];
            for (int d = 0; d < 3; d++) gc[d] = J.shp[d] - g0[d] < 512 ? J.shp[d] - g0[d] : 512;
            if (z3_read(ck, g0, gc, gb, nt)) { fprintf(stderr, "patch-raster: --check read failed\n"); return 1; }
            for (size_t k = a; k < b; k++) {
                const int z = (int)(K[k] >> 18 & 511), y = (int)(K[k] >> 9 & 511), x = (int)(K[k] & 511);
                if (z >= gc[0] || y >= gc[1] || x >= gc[2]) continue;
                const uint8_t v = gb[((size_t)z * gc[1] + y) * gc[2] + x]; if (v >= 252) continue;
                hist[(v * 18 + 126) / 252 % 18]++; nck++;
            }
        }
        if (zw) {
            memset(sb, 255, (size_t)512 * 512 * 512);
            for (size_t k = a; k < b; k++) sb[((size_t)(K[k] >> 18 & 511) * 512 + (K[k] >> 9 & 511)) * 512 + (K[k] & 511)] = (uint8_t)code;
            if (z3w_write_shard(zw, sz, sy, sx, sb, nt)) { fprintf(stderr, "patch-raster: %s\n", z3w_error()); return 1; }
        }
        nsh++; a = b;
    }
    if (zw) z3w_close(zw);
    fprintf(stderr, "patch-raster: %lld shards%s%s\n", nsh, conly ? " (check only, nothing written)" : " written to ", conly ? "" : out);
    if (ck) {
        double cs = 0, sn = 0; for (int k = 0; k < 18; k++) { cs += hist[k] * cos(2 * M_PI * k / 18); sn += hist[k] * sin(2 * M_PI * k / 18); }
        const double mu = atan2(sn, cs) / (2 * M_PI), R = nck ? sqrt(cs * cs + sn * sn) / nck : 0; long long w8 = 0, w4 = 0;
        for (int k = 0; k < 18; k++) { double d = k / 18.0 - mu; d -= floor(d + 0.5); if (fabs(d) <= 0.125) w8 += hist[k]; if (fabs(d) <= 0.25) w4 += hist[k]; }
        printf("patch-raster: --check: %lld shell voxels with a label; label phase circular mean %+.3f turn (resultant %.3f); within 1/8 turn of it %.3f, 1/4 %.3f; within 1/4 turn of 0: ",
               nck, mu, R, nck ? (double)w8 / nck : 0, nck ? (double)w4 / nck : 0);
        long long z4 = 0; for (int k = 0; k < 18; k++) { double d = k / 18.0; d -= floor(d + 0.5); if (fabs(d) <= 0.25) z4 += hist[k]; }
        printf("%.3f\npatch-raster: --check histogram (18 bins from 0):", nck ? (double)z4 / nck : 0);
        for (int k = 0; k < 18; k++) printf(" %.3f", nck ? (double)hist[k] / nck : 0);
        printf("\n"); z3_close(ck);
    }
    free(sb); free(gb); free(K); free(J.dirs); free(J.nout); free(J.cap); free(J.out);
    return 0;
}
