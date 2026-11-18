/* ufsm gnw-stats GNW.bin (--ct ROOT KEY | --ct-raw F --ct-shape Z,Y,X) [--ct-div 8] [--window 640] [--halo 64] [--thr 20,80]
                  [--boxes BOXES.tsv --arm NAME=DIR ...]
   (fix0332 2026-11-18, the parity audit's window-content study; the C port of the first look.) GNW.bin: predict's UFSM_PARITY_GNW dump,
   one record per window: int32 tz ty tx n, then n floats (unet_gn_stats_get: every GroupNorm site's per-group mean / rstd, sample 0).
   Per window: D = the mean over the n statistics of |x - median| / (1.4826 MAD) across all windows (clipped at 50): how atypical the
   window's own normalisation is; occupancy = the share of CT voxels > thr in the window (the CT array at window coordinates / ct-div:
   CT level 5 against level-2 windows = 8). Prints corr(D, occupancy) (Pearson and rank) and D by occupancy bin. With --boxes (the GT box
   list: name scroll z y x dz dy dx ..., window coordinates) and --arm NAME=DIR (gtboxes.sh's per-box output dir: *.tsv rows "name voxels
   ids ..."): each box's window (the most interior overlap), per arm corr(ids, D) / corr(ids, interior occupancy), and per arm vs the
   first one the paired difference and its correlation with D and occupancy. */
#include "optcheck.h"
#include "store.h"
#include "zarr3.h"
#include <dirent.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { GW_MAXARM = 8, GW_MAXBOX = 512 };
typedef struct { char name[64]; int64_t b[6]; int win; double cov, occ; double ids[GW_MAXARM]; int has[GW_MAXARM]; } gw_box;

static const char *gw_opt(int argc, char **argv, const char *k, const char *d) {
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(k))) return argv[i + 1];
    return d;
}
static int gw_cmpd(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static double gw_median(double *v, int n) { qsort(v, (size_t)n, sizeof *v, gw_cmpd); return n ? (n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2])) : NAN; }
static double gw_corr(const double *a, const double *b, int n) {
    double ma = 0, mb = 0; int k = 0;
    for (int i = 0; i < n; i++) if (isfinite(a[i]) && isfinite(b[i])) { ma += a[i]; mb += b[i]; k++; }
    if (k < 4) return NAN;
    ma /= k; mb /= k; double sab = 0, saa = 0, sbb = 0;
    for (int i = 0; i < n; i++) if (isfinite(a[i]) && isfinite(b[i])) { const double x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
    return saa > 0 && sbb > 0 ? sab / sqrt(saa * sbb) : NAN;
}
static const double *gw_key; static int gw_cmpi(const void *a, const void *b) { const double x = gw_key[*(const int *)a], y = gw_key[*(const int *)b]; return (x > y) - (x < y); }
static void gw_rank(const double *v, double *r, int n) {   /* ranks with ties averaged */
    int *ix = malloc((size_t)n * sizeof *ix); for (int i = 0; i < n; i++) ix[i] = i;
    gw_key = v; qsort(ix, (size_t)n, sizeof *ix, gw_cmpi);
    for (int i = 0; i < n;) { int j = i; while (j + 1 < n && v[ix[j + 1]] == v[ix[i]]) j++; for (int k = i; k <= j; k++) r[ix[k]] = 0.5 * (i + j); i = j + 1; }
    free(ix);
}
static double gw_rcorr(const double *a, const double *b, int n) {
    double *ra = malloc((size_t)n * sizeof *ra), *rb = malloc((size_t)n * sizeof *rb); gw_rank(a, ra, n); gw_rank(b, rb, n);
    const double c = gw_corr(ra, rb, n); free(ra); free(rb); return c;
}
/* the CT occupancy source: a raw uint8 array or a zarr array, read per box */
typedef struct { const uint8_t *raw; z3 *z; int64_t sh[3]; } gw_ct;
static double gw_occ(gw_ct *c, const int64_t o[3], int64_t s, int div, int thr) {
    int64_t lo[3], n[3], tot = 1;
    for (int a = 0; a < 3; a++) { lo[a] = o[a] < 0 ? 0 : o[a] / div; int64_t hi = (o[a] + s) / div; if (hi > c->sh[a]) hi = c->sh[a]; n[a] = hi - lo[a]; if (n[a] <= 0) return 0; tot *= n[a]; }
    uint8_t *buf = nullptr; const uint8_t *p;
    if (c->raw) p = nullptr; else { buf = malloc((size_t)tot); if (!buf || z3_read(c->z, lo, n, buf, 4)) { free(buf); return NAN; } p = buf; }
    int64_t cnt = 0, k = 0;
    for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) for (int64_t x = 0; x < n[2]; x++, k++) {
        const uint8_t v = c->raw ? c->raw[((lo[0] + z) * c->sh[1] + lo[1] + y) * c->sh[2] + lo[2] + x] : p[k];
        cnt += v > thr;
    }
    free(buf); return (double)cnt / (double)tot;
}
static int gw_load_arm(const char *dir, gw_box *B, int nb, int a) {   /* every *.tsv in dir: "name voxels ids ..." rows */
    DIR *d = opendir(dir); if (!d) { fprintf(stderr, "gnw-stats: cannot open %s\n", dir); return 1; }
    struct dirent *e; char path[4096], line[1024];
    while ((e = readdir(d))) {
        const size_t L = strlen(e->d_name); if (L < 5 || strcmp(e->d_name + L - 4, ".tsv")) continue;
        snprintf(path, sizeof path, "%s/%s", dir, e->d_name); FILE *f = fopen(path, "r"); if (!f) continue;
        while (fgets(line, sizeof line, f)) {
            char nm[64]; long long vox; double ids; if (sscanf(line, "%63s %lld %lf", nm, &vox, &ids) != 3) continue;
            for (int i = 0; i < nb; i++) if (!strcmp(B[i].name, nm)) { B[i].ids[a] = ids; B[i].has[a] = 1; }
        }
        fclose(f);
    }
    closedir(d); return 0;
}

int cmd_gnw_stats(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: ufsm gnw-stats GNW.bin (--ct ROOT KEY | --ct-raw F --ct-shape Z,Y,X) [--ct-div 8] [--window 640] [--halo 64] "
                                    "[--thr 20,80] [--boxes BOXES.tsv --arm NAME=DIR ...]\n"); return 2; }
    const int win = atoi(gw_opt(argc, argv, "--window", "640")), halo = atoi(gw_opt(argc, argv, "--halo", "64")), div = atoi(gw_opt(argc, argv, "--ct-div", "8"));
    int thr[2] = {20, 80}; sscanf(gw_opt(argc, argv, "--thr", "20,80"), "%d,%d", &thr[0], &thr[1]);
    /* the dump */
    FILE *f = fopen(argv[2], "rb"); if (!f) { fprintf(stderr, "gnw-stats: cannot open %s\n", argv[2]); return 1; }
    fseek(f, 0, SEEK_END); const long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    int32_t hd[4]; if (fread(hd, 4, 4, f) != 4 || hd[3] <= 0) { fprintf(stderr, "gnw-stats: %s: no record\n", argv[2]); fclose(f); return 1; }
    const int ns = hd[3]; const long rec = 16 + 4L * ns, nw = fsz / rec; fseek(f, 0, SEEK_SET);
    if (nw * rec != fsz) fprintf(stderr, "gnw-stats: %s: %ld bytes is not a whole number of %ld-byte records (truncated?)\n", argv[2], fsz, rec);
    int64_t (*org)[3] = malloc((size_t)nw * sizeof *org); double *st = malloc((size_t)nw * ns * sizeof *st); float *tmp = malloc((size_t)ns * 4);
    for (long w = 0; w < nw; w++) {
        if (fread(hd, 4, 4, f) != 4 || hd[3] != ns || fread(tmp, 4, (size_t)ns, f) != (size_t)ns) { fprintf(stderr, "gnw-stats: bad record %ld\n", w); fclose(f); return 1; }
        for (int a = 0; a < 3; a++) org[w][a] = hd[a];
        for (int j = 0; j < ns; j++) st[w * ns + j] = tmp[j];
    }
    fclose(f); free(tmp);
    /* D: robust z per statistic across windows */
    double *D = calloc((size_t)nw, sizeof *D), *col = malloc((size_t)nw * sizeof *col);
    for (int j = 0; j < ns; j++) {
        for (long w = 0; w < nw; w++) col[w] = st[w * ns + j];
        const double med = gw_median(col, (int)nw);
        for (long w = 0; w < nw; w++) col[w] = fabs(st[w * ns + j] - med);
        const double mad = gw_median(col, (int)nw) * 1.4826 + 1e-9;
        for (long w = 0; w < nw; w++) { const double z = fabs(st[w * ns + j] - med) / mad; D[w] += (z < 50 ? z : 50) / ns; }
    }
    /* the CT */
    gw_ct ct = {0}; uint8_t *raw = nullptr;
    const char *rp = gw_opt(argc, argv, "--ct-raw", nullptr);
    if (rp) {
        long long s0, s1, s2; if (sscanf(gw_opt(argc, argv, "--ct-shape", ""), "%lld,%lld,%lld", &s0, &s1, &s2) != 3) { fprintf(stderr, "gnw-stats: --ct-raw needs --ct-shape Z,Y,X\n"); return 2; }
        ct.sh[0] = s0; ct.sh[1] = s1; ct.sh[2] = s2; const size_t n = (size_t)(s0 * s1 * s2);
        FILE *rf = fopen(rp, "rb"); raw = rf ? malloc(n) : nullptr;
        if (!raw || fread(raw, 1, n, rf) != n) { fprintf(stderr, "gnw-stats: cannot read %zu bytes from %s\n", n, rp); if (rf) fclose(rf); return 1; }
        fclose(rf); ct.raw = raw;
    } else {
        int i = 2; while (i + 2 < argc && strcmp(argv[i], opt_n("--ct"))) i++;
        if (i + 2 >= argc) { fprintf(stderr, "gnw-stats: --ct ROOT KEY or --ct-raw F --ct-shape Z,Y,X\n"); return 2; }
        store *s = store_open(argv[i + 1]); ct.z = s ? z3_open(s, argv[i + 2], nullptr) : nullptr;
        if (!ct.z) { fprintf(stderr, "gnw-stats: %s %s: %s\n", argv[i + 1], argv[i + 2], z3_error()); return 1; }
        for (int a = 0; a < 3; a++) ct.sh[a] = z3_meta_of(ct.z)->shape[a];
    }
    double *occ[2]; for (int t = 0; t < 2; t++) { occ[t] = malloc((size_t)nw * sizeof *occ[t]); for (long w = 0; w < nw; w++) occ[t][w] = gw_occ(&ct, org[w], win, div, thr[t]); }
    for (long w = 0; w < nw; w++) col[w] = D[w];
    const double dmed = gw_median(col, (int)nw), dp90 = col[(int)(0.9 * (nw - 1))];
    printf("gnw-stats: %ld windows, %d statistics; D median %.2f p90 %.2f\n", nw, ns, dmed, dp90);
    printf("windows: corr(D, occ>%d) %+.3f (rank %+.3f); corr(D, occ>%d) %+.3f (rank %+.3f)\n", thr[0], gw_corr(D, occ[0], (int)nw), gw_rcorr(D, occ[0], (int)nw),
           thr[1], gw_corr(D, occ[1], (int)nw), gw_rcorr(D, occ[1], (int)nw));
    const double bins[5] = {0, 0.1, 0.3, 0.6, 1.01};
    for (int b = 0; b < 4; b++) {
        int k = 0; for (long w = 0; w < nw; w++) if (occ[0][w] >= bins[b] && occ[0][w] < bins[b + 1]) col[k++] = D[w];
        printf("  occ>%d [%.1f, %.1f): %4d windows, D median %.2f\n", thr[0], bins[b], bins[b + 1] > 1 ? 1.0 : bins[b + 1], k, k ? gw_median(col, k) : NAN);
    }
    /* boxes */
    const char *bp = gw_opt(argc, argv, "--boxes", nullptr);
    if (bp) {
        gw_box *B = calloc(GW_MAXBOX, sizeof *B); int nb = 0; char line[1024];
        FILE *bf = fopen(bp, "r"); if (!bf) { fprintf(stderr, "gnw-stats: cannot open %s\n", bp); return 1; }
        while (fgets(line, sizeof line, bf) && nb < GW_MAXBOX) {
            char nm[64], sc[64]; long long v[6];
            if (sscanf(line, "%63s %63s %lld %lld %lld %lld %lld %lld", nm, sc, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) != 8) continue;
            snprintf(B[nb].name, sizeof B[nb].name, "%s", nm); for (int a = 0; a < 6; a++) B[nb].b[a] = v[a]; nb++;
        }
        fclose(bf);
        for (int i = 0; i < nb; i++) {   /* the window of most interior overlap */
            double best = -1; int bi = -1; const int64_t s = win - 2 * halo;
            for (long w = 0; w < nw; w++) {
                double ov = 1; for (int a = 0; a < 3; a++) { const int64_t lo = B[i].b[a] > org[w][a] + halo ? B[i].b[a] : org[w][a] + halo,
                    hi = B[i].b[a] + B[i].b[3 + a] < org[w][a] + halo + s ? B[i].b[a] + B[i].b[3 + a] : org[w][a] + halo + s; ov *= hi > lo ? (double)(hi - lo) : 0; }
                if (ov > best) { best = ov; bi = (int)w; }
            }
            B[i].win = bi; B[i].cov = best / ((double)B[i].b[3] * B[i].b[4] * B[i].b[5]);
            const int64_t io[3] = {org[bi][0] + halo, org[bi][1] + halo, org[bi][2] + halo}; B[i].occ = gw_occ(&ct, io, win - 2 * halo, div, thr[0]);
        }
        char an[GW_MAXARM][64]; int na = 0;
        for (int i = 2; i + 1 < argc && na < GW_MAXARM; i++) if (!strcmp(argv[i], opt_n("--arm"))) {
            const char *eq = strchr(argv[i + 1], '='); if (!eq) { fprintf(stderr, "gnw-stats: --arm NAME=DIR\n"); return 2; }
            snprintf(an[na], sizeof an[na], "%.*s", (int)(eq - argv[i + 1]), argv[i + 1]);
            if (gw_load_arm(eq + 1, B, nb, na)) return 1;
            na++;
        }
        int *sel = malloc((size_t)nb * sizeof *sel), m = 0;   /* the boxes scored in every arm */
        for (int i = 0; i < nb; i++) { int ok = B[i].cov > 0; for (int a = 0; a < na; a++) ok &= B[i].has[a]; if (ok) sel[m++] = i; }
        double *Db = malloc((size_t)(m + 1) * sizeof *Db), *Ob = malloc((size_t)(m + 1) * sizeof *Ob), *v = malloc((size_t)(m + 1) * sizeof *v), *dv = malloc((size_t)(m + 1) * sizeof *dv);
        for (int k = 0; k < m; k++) { Db[k] = D[B[sel[k]].win]; Ob[k] = B[sel[k]].occ; col[k] = B[sel[k]].cov; }
        printf("boxes: %d of %d scored in every arm (%d arms), median window coverage %.2f\n", m, nb, na, m ? gw_median(col, m) : NAN);
        for (int a = 0; a < na; a++) {
            double mean = 0; for (int k = 0; k < m; k++) { v[k] = B[sel[k]].ids[a]; mean += v[k] / m; }
            printf("  %-8s ids mean %.4f: corr(ids, D) %+.3f (rank %+.3f), corr(ids, occ) %+.3f (rank %+.3f)\n", an[a], mean, gw_corr(v, Db, m), gw_rcorr(v, Db, m),
                   gw_corr(v, Ob, m), gw_rcorr(v, Ob, m));
        }
        for (int a = 1; a < na; a++) {
            double mean = 0; int up = 0, dn = 0;
            for (int k = 0; k < m; k++) { dv[k] = B[sel[k]].ids[a] - B[sel[k]].ids[0]; mean += dv[k] / m; up += dv[k] > 0.005; dn += dv[k] < -0.005; v[k] = fabs(dv[k]); }
            printf("  %s-%s: mean %+.4f, better / worse %d / %d (|d| > 0.005), corr(d, D) %+.3f, corr(|d|, D) %+.3f, corr(d, occ) %+.3f\n", an[a], an[0], mean, up, dn,
                   gw_corr(dv, Db, m), gw_corr(v, Db, m), gw_corr(dv, Ob, m));
        }
        free(sel); free(Db); free(Ob); free(v); free(dv); free(B);
    }
    free(org); free(st); free(D); free(col); free(occ[0]); free(occ[1]); free(raw); if (ct.z) z3_close(ct.z);
    return 0;
}
