/* ufsm sheet-score OURS_ROOT OURS_KEY GT_ROOT GT_KEY [--tol 1] [--prec-r 8] [--cell 64] [--box z,y,x,nz,ny,nx] [--block 128] [--threads N] [--tsv OUT]
   Sheet-level score of our surfaces against GT surfaces, both as winding_mod14 code rasters on the same grid (codes 1 + round(18 q) mod 252,
   0 empty, 255 conflict: ufsm surf-labels / raster --value / the GT codes zarrs). A "sheet voxel" is a code 1..252.
     recall     GT sheet voxels with one of ours within --tol voxels (Chebyshev; ours dilated) / GT sheet voxels
     precision  our sheet voxels with a GT sheet within --tol / our sheet voxels in the GT-covered region (a GT sheet within --prec-r:
                GT labels only some sheets, so precision is measured only near them)
     turn offset per matched voxel of ours: d = (our q - GT q of the nearest GT sheet voxel) in turns; aligned when |d - round(d)| < 0.25
                (same spiral sense), then k = round(d) mod 14
     cell ids   per --cell^3 cell (gauge-free): the share of its aligned matched voxels on the cell's most common k; reported as the
                matched-voxel-weighted mean over cells (1 = no local skips / switches / merges)
     global ids the share on the box-wide most common k (one gauge for the box; mod 14 turns)
   --tsv: per cell z y x n_matched dominant_share k. Deterministic for any thread count. */
#include "store.h"
#include "zarr3.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *ss_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static inline int ss_sheet(uint8_t c) { return c >= 1 && c <= 252; }

typedef struct {
    z3 *ours, *gt; int64_t b0[3], bn[3]; int B, tol, pr, cell, nt; int64_t nb[3];
    _Atomic long next; long nitems; _Atomic int fail;
    _Atomic long long gt_n, gt_hit, our_n, our_reg, our_hit, al_n, mis_n, hist[14];
    double *cnum, *cden; int *ck; long long *cn; int64_t nc[3];   /* per cell (global cell index): dominant count, aligned count, k */
} ss_job;

/* separable box max (Chebyshev dilation by r) of a 0/1 mask n[0] x n[1] x n[2], in place, with scratch */
static void ss_dilate(uint8_t *m, const int64_t n[3], int r, uint8_t *tmp) {
    if (r <= 0) return;
    const int64_t s[3] = {n[1] * n[2], n[2], 1};
    for (int ax = 0; ax < 3; ax++) {
        memcpy(tmp, m, (size_t)(n[0] * n[1] * n[2]));
        const int64_t L = n[ax];
        for (int64_t z = 0; z < (ax == 0 ? 1 : n[0]); z++) for (int64_t y = 0; y < (ax == 1 ? 1 : n[1]); y++) for (int64_t x = 0; x < (ax == 2 ? 1 : n[2]); x++) {
            const int64_t base = z * s[0] + y * s[1] + x * s[2];
            /* running count over the window [i - r, i + r] */
            int cnt = 0;
            for (int64_t i = 0; i < L && i <= r; i++) cnt += tmp[base + i * s[ax]];
            for (int64_t i = 0; i < L; i++) {
                m[base + i * s[ax]] = cnt > 0;
                const int64_t add = i + r + 1, rem = i - r;
                if (add < L) cnt += tmp[base + add * s[ax]];
                if (rem >= 0) cnt -= tmp[base + rem * s[ax]];
            }
        }
    }
}

static int ss_block(ss_job *J, long it, uint8_t *bufo, uint8_t *bufg, uint8_t *mo, uint8_t *mg, uint8_t *mr, uint8_t *tmp, long long *chist) {
    const int64_t bz = it / (J->nb[1] * J->nb[2]), by = it / J->nb[2] % J->nb[1], bx = it % J->nb[2];
    const int H = J->pr > J->tol ? J->pr : J->tol;
    int64_t c0[3], cn[3], o[3], n[3];   /* core (box-relative) and the read region (absolute, clamped to the box) */
    const int64_t bi[3] = {bz, by, bx};
    for (int d = 0; d < 3; d++) {
        c0[d] = bi[d] * J->B; cn[d] = J->bn[d] - c0[d] < J->B ? J->bn[d] - c0[d] : J->B;
        int64_t lo = c0[d] - H, hi = c0[d] + cn[d] + H; if (lo < 0) lo = 0; if (hi > J->bn[d]) hi = J->bn[d];
        o[d] = J->b0[d] + lo; n[d] = hi - lo;
    }
    const int64_t off[3] = {J->b0[0] + c0[0] - o[0], J->b0[1] + c0[1] - o[1], J->b0[2] + c0[2] - o[2]};   /* core start inside the read region */
    if (z3_read(J->ours, o, n, bufo, 1) || z3_read(J->gt, o, n, bufg, 1)) return -1;
    const size_t NN = (size_t)(n[0] * n[1] * n[2]);
    for (size_t i = 0; i < NN; i++) { mo[i] = ss_sheet(bufo[i]); mg[i] = ss_sheet(bufg[i]); }
    memcpy(mr, mg, NN);
    ss_dilate(mo, n, J->tol, tmp);   /* ours within tol */
    ss_dilate(mr, n, J->pr, tmp);    /* GT-covered region */
    long long gt_n = 0, gt_hit = 0, our_n = 0, our_reg = 0, our_hit = 0, al = 0, mis = 0, hist[14] = {0};
    const int C = J->cell; const int64_t ncz = (cn[0] + C - 1) / C, ncy = (cn[1] + C - 1) / C, ncx = (cn[2] + C - 1) / C;
    memset(chist, 0, sizeof(long long) * 14 * (size_t)(ncz * ncy * ncx));
    const int T = J->tol;
    for (int64_t z = 0; z < cn[0]; z++) for (int64_t y = 0; y < cn[1]; y++) for (int64_t x = 0; x < cn[2]; x++) {
        const int64_t zz = z + off[0], yy = y + off[1], xx = x + off[2];
        const size_t i = ((size_t)zz * n[1] + yy) * n[2] + xx;
        if (ss_sheet(bufg[i])) { gt_n++; gt_hit += mo[i]; }
        if (!ss_sheet(bufo[i])) continue;
        our_n++;
        if (!mr[i]) continue;
        our_reg++;
        /* the nearest GT sheet voxel within tol (Chebyshev rings outward; ties: scan order) */
        int found = -1;
        for (int r = 0; r <= T && found < 0; r++)
            for (int dz = -r; dz <= r && found < 0; dz++) for (int dy = -r; dy <= r && found < 0; dy++) for (int dx = -r; dx <= r; dx++) {
                if (abs(dz) != r && abs(dy) != r && abs(dx) != r) continue;
                const int64_t qz = zz + dz, qy = yy + dy, qx = xx + dx;
                if (qz < 0 || qy < 0 || qx < 0 || qz >= n[0] || qy >= n[1] || qx >= n[2]) continue;
                const uint8_t g = bufg[((size_t)qz * n[1] + qy) * n[2] + qx];
                if (ss_sheet(g)) { found = g; break; }
            }
        if (found < 0) continue;
        our_hit++;
        double d = ((double)bufo[i] - (double)found) / 18.0;   /* turns, mod 14 */
        d = fmod(d, 14.0); if (d < 0) d += 14.0;
        const double fr = d - floor(d + 0.5);
        if (fabs(fr) >= 0.25) { mis++; continue; }
        int k = (int)floor(d + 0.5) % 14;
        al++; hist[k]++;
        chist[14 * (size_t)(((z / C) * ncy + y / C) * ncx + x / C) + k]++;
    }
    atomic_fetch_add(&J->gt_n, gt_n); atomic_fetch_add(&J->gt_hit, gt_hit); atomic_fetch_add(&J->our_n, our_n); atomic_fetch_add(&J->our_reg, our_reg);
    atomic_fetch_add(&J->our_hit, our_hit); atomic_fetch_add(&J->al_n, al); atomic_fetch_add(&J->mis_n, mis);
    for (int k = 0; k < 14; k++) atomic_fetch_add(&J->hist[k], hist[k]);
    /* cells: the block's cells are whole cells of the global cell grid when --block is a multiple of --cell (checked) */
    for (int64_t a = 0; a < ncz; a++) for (int64_t b = 0; b < ncy; b++) for (int64_t c = 0; c < ncx; c++) {
        const long long *h = chist + 14 * (size_t)((a * ncy + b) * ncx + c); long long s = 0, m = 0; int km = 0;
        for (int k = 0; k < 14; k++) { s += h[k]; if (h[k] > m) { m = h[k]; km = k; } }
        const int64_t gz = c0[0] / C + a, gy = c0[1] / C + b, gx = c0[2] / C + c, gi = (gz * J->nc[1] + gy) * J->nc[2] + gx;
        J->cnum[gi] = (double)m; J->cden[gi] = (double)s; J->ck[gi] = km; J->cn[gi] = s;
    }
    return 0;
}

static void *ss_run(void *arg) {
    ss_job *J = arg; const int H = J->pr > J->tol ? J->pr : J->tol;
    const size_t M = (size_t)(J->B + 2 * H) * (J->B + 2 * H) * (J->B + 2 * H);
    uint8_t *bo = malloc(M), *bg = malloc(M), *mo = malloc(M), *mg = malloc(M), *mr = malloc(M), *tmp = malloc(M);
    const int64_t nce = (J->B + J->cell - 1) / J->cell; long long *ch = malloc(sizeof(long long) * 14 * (size_t)(nce * nce * nce));
    if (!bo || !bg || !mo || !mg || !mr || !tmp || !ch) atomic_store(&J->fail, 1);
    else for (long it; (it = atomic_fetch_add(&J->next, 1)) < J->nitems && !atomic_load(&J->fail);)
        if (ss_block(J, it, bo, bg, mo, mg, mr, tmp, ch)) { fprintf(stderr, "sheet-score: read failed: %s\n", z3_error()); atomic_store(&J->fail, 1); }
    free(bo); free(bg); free(mo); free(mg); free(mr); free(tmp); free(ch);
    return nullptr;
}

int cmd_sheet_score(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: ufsm sheet-score OURS_ROOT OURS_KEY GT_ROOT GT_KEY [--tol 1] [--prec-r 8] [--cell 64] [--box z,y,x,nz,ny,nx] [--block 128] [--threads N] [--tsv OUT]\n");
        return 2;
    }
    ss_job J = {0};
    J.ours = z3_open(store_open(argv[2]), argv[3], nullptr); J.gt = z3_open(store_open(argv[4]), argv[5], nullptr);
    if (!J.ours || !J.gt) { fprintf(stderr, "sheet-score: cannot open %s: %s\n", !J.ours ? argv[2] : argv[4], z3_error()); return 1; }
    const int64_t *so = z3_meta_of(J.ours)->shape, *sg = z3_meta_of(J.gt)->shape;
    if (memcmp(so, sg, 3 * sizeof *so)) {
        fprintf(stderr, "sheet-score: shapes differ (%lld,%lld,%lld vs %lld,%lld,%lld): put both on the same grid\n",
                (long long)so[0], (long long)so[1], (long long)so[2], (long long)sg[0], (long long)sg[1], (long long)sg[2]);
        return 1;
    }
    J.tol = atoi(ss_opt(argc, argv, "--tol", "1")); J.pr = atoi(ss_opt(argc, argv, "--prec-r", "8")); J.cell = atoi(ss_opt(argc, argv, "--cell", "64"));
    J.B = atoi(ss_opt(argc, argv, "--block", "128")); J.nt = atoi(ss_opt(argc, argv, "--threads", "8"));
    if (J.tol < 0 || J.tol > 8 || J.pr < J.tol || J.pr > 64 || J.cell < 8 || J.B < J.cell || J.B % J.cell || J.nt < 1) {
        fprintf(stderr, "sheet-score: --tol 0..8, --prec-r tol..64, --cell >= 8 dividing --block, --threads >= 1\n"); return 2;
    }
    for (int d = 0; d < 3; d++) { J.b0[d] = 0; J.bn[d] = so[d]; }
    const char *bx = ss_opt(argc, argv, "--box", nullptr);
    if (bx) { long long v[6]; if (sscanf(bx, "%lld,%lld,%lld,%lld,%lld,%lld", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) != 6) { fprintf(stderr, "sheet-score: --box z,y,x,nz,ny,nx\n"); return 2; }
        for (int d = 0; d < 3; d++) { J.b0[d] = v[d] < 0 ? 0 : v[d]; J.bn[d] = v[d] + v[3 + d] > so[d] ? so[d] - J.b0[d] : v[3 + d] - (J.b0[d] - v[d]);
            if (J.bn[d] <= 0) { fprintf(stderr, "sheet-score: --box outside the arrays\n"); return 2; } } }
    for (int d = 0; d < 3; d++) { J.nb[d] = (J.bn[d] + J.B - 1) / J.B; J.nc[d] = (J.bn[d] + J.cell - 1) / J.cell; }
    J.nitems = (long)(J.nb[0] * J.nb[1] * J.nb[2]);
    const size_t NC = (size_t)(J.nc[0] * J.nc[1] * J.nc[2]);
    J.cnum = calloc(NC, sizeof *J.cnum); J.cden = calloc(NC, sizeof *J.cden); J.ck = calloc(NC, sizeof *J.ck); J.cn = calloc(NC, sizeof *J.cn);
    if (!J.cnum || !J.cden || !J.ck || !J.cn) { fprintf(stderr, "sheet-score: out of memory (%zu cells)\n", NC); return 1; }
    const int m = J.nt < 256 ? J.nt : 256; pthread_t th[256];
    for (int q = 0; q < m; q++) pthread_create(&th[q], nullptr, ss_run, &J);
    for (int q = 0; q < m; q++) pthread_join(th[q], nullptr);
    if (atomic_load(&J.fail)) return 1;
    double cn = 0, cd = 0; for (size_t i = 0; i < NC; i++) { cn += J.cnum[i]; cd += J.cden[i]; }
    long long hm = 0, hs = 0; int km = 0; for (int k = 0; k < 14; k++) { hs += J.hist[k]; if (J.hist[k] > hm) { hm = J.hist[k]; km = k; } }
    const long long gt_n = J.gt_n, our_reg = J.our_reg, our_hit = J.our_hit, al = J.al_n;
    printf("sheet-score: box %lld,%lld,%lld + %lldx%lldx%lld, tol %d, prec-r %d, cell %d\n", (long long)J.b0[0], (long long)J.b0[1], (long long)J.b0[2],
           (long long)J.bn[0], (long long)J.bn[1], (long long)J.bn[2], J.tol, J.pr, J.cell);
    printf("sheet-score: recall %.4f (%lld of %lld GT sheet voxels) | precision %.4f (%lld of %lld ours near GT; %lld ours in all)\n",
           gt_n ? (double)J.gt_hit / gt_n : 0, (long long)J.gt_hit, gt_n, our_reg ? (double)our_hit / our_reg : 0, our_hit, our_reg, (long long)J.our_n);
    printf("sheet-score: aligned %.4f of matched | cell ids %.4f | global ids %.4f (offset %d turns mod 14)\n",
           our_hit ? (double)al / our_hit : 0, cd > 0 ? cn / cd : 0, hs ? (double)hm / hs : 0, km);
    const char *tsv = ss_opt(argc, argv, "--tsv", nullptr);
    if (tsv) {
        FILE *f = fopen(tsv, "w"); if (!f) { fprintf(stderr, "sheet-score: cannot write %s\n", tsv); return 1; }
        fprintf(f, "z\ty\tx\tmatched\tdominant_share\tk\n");
        for (int64_t a = 0; a < J.nc[0]; a++) for (int64_t b = 0; b < J.nc[1]; b++) for (int64_t c = 0; c < J.nc[2]; c++) {
            const size_t i = (size_t)((a * J.nc[1] + b) * J.nc[2] + c); if (!J.cn[i]) continue;
            fprintf(f, "%lld\t%lld\t%lld\t%lld\t%.4f\t%d\n", (long long)(J.b0[0] + a * J.cell), (long long)(J.b0[1] + b * J.cell), (long long)(J.b0[2] + c * J.cell),
                    J.cn[i], J.cnum[i] / J.cden[i], J.ck[i]);
        }
        fclose(f);
    }
    z3_close(J.ours); z3_close(J.gt); free(J.cnum); free(J.cden); free(J.ck); free(J.cn);
    return 0;
}
