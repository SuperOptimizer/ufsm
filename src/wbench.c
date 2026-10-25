/* ufsm wbench-stats A.tsv [B.tsv] [--metric ids,p1x,...] [--set clean|all] [--ceil-min 0.8] [--n 10000] [--seed 1] [--boxes 0|1]
   Statistics of the winding+fitter bench (scripts/wbench/wbench.sh rows: a header line naming the columns, one row per box; columns
   used: "box", "scroll", the metrics, "ceil" for the clean set; "na" = missing). For each metric: the mean over boxes with a 95 %
   percentile bootstrap CI, resampling boxes WITH replacement inside each scroll (stratified: every scroll keeps its box count), the
   per-scroll means; with B: B's mean, the paired difference A - B (same resample for both: only boxes scored in both), its CI,
   the bootstrap share of A > B, and the minimal detectable difference MDE = 2.8 x the bootstrap SE of the difference (80 % power at
   two-sided 5 %). --set clean (default) keeps boxes with ceil >= --ceil-min (A's ceil; boxes without a ceil are kept; the others are summarised on a "label-noise" line); --boxes 1
   prints the per-box values. Several rows of one box (replicates) are averaged first. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { WB_MAXBOX = 1024, WB_MAXCOL = 128, WB_MAXSC = 32 };
typedef struct { char box[64], scroll[64]; double v[WB_MAXCOL]; int nv[WB_MAXCOL]; } wb_row;
typedef struct { int ncol; char col[WB_MAXCOL][48]; int nrow; wb_row *row; } wb_tab;

static const char *wb_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static int wb_colix(const wb_tab *t, const char *c) { for (int i = 0; i < t->ncol; i++) if (!strcmp(t->col[i], c)) return i; return -1; }
static int wb_split(char *s, char **f, int max) {   /* tab or space separated */
    int n = 0; char *p = s;
    while (*p && n < max) { while (*p == ' ' || *p == '\t') p++; if (!*p || *p == '\n') break; f[n++] = p; while (*p && *p != ' ' && *p != '\t' && *p != '\n') p++; if (*p) *p++ = 0; }
    return n;
}
static int wb_load(const char *path, wb_tab *t) {
    FILE *f = fopen(path, "r"); if (!f) { fprintf(stderr, "wbench-stats: cannot open %s\n", path); return 1; }
    memset(t, 0, sizeof *t); t->row = calloc(WB_MAXBOX, sizeof *t->row);
    char line[16384]; int hdr = 0, ib = -1, is = -1;
    while (fgets(line, sizeof line, f)) {
        char *fl[WB_MAXCOL + 8]; if (line[0] == '#' && !hdr) memmove(line, line + 1, strlen(line));
        else if (line[0] == '#') continue;
        const int n = wb_split(line, fl, WB_MAXCOL);
        if (!n) continue;
        if (!hdr) {
            hdr = 1; t->ncol = n; for (int i = 0; i < n; i++) snprintf(t->col[i], sizeof t->col[i], "%s", fl[i]);
            ib = wb_colix(t, "box"); is = wb_colix(t, "scroll");
            if (ib < 0 || is < 0) { fprintf(stderr, "wbench-stats: %s: the header needs \"box\" and \"scroll\" columns\n", path); fclose(f); return 1; }
            continue;
        }
        if (!strcmp(fl[0], t->col[0])) continue;   /* a repeated header */
        if (n <= ib || n <= is) continue;
        int r = 0; while (r < t->nrow && strcmp(t->row[r].box, fl[ib])) r++;
        if (r == t->nrow) { if (t->nrow == WB_MAXBOX) continue; t->nrow++; snprintf(t->row[r].box, 64, "%s", fl[ib]); snprintf(t->row[r].scroll, 64, "%s", fl[is]); }
        for (int c = 0; c < n && c < t->ncol; c++) { char *e; const double v = strtod(fl[c], &e); if (e != fl[c] && !*e && isfinite(v)) { t->row[r].v[c] += v; t->row[r].nv[c]++; } }
    }
    fclose(f);
    for (int r = 0; r < t->nrow; r++) for (int c = 0; c < t->ncol; c++) t->row[r].v[c] = t->row[r].nv[c] ? t->row[r].v[c] / t->row[r].nv[c] : NAN;
    return hdr ? 0 : 1;
}
static uint64_t wb_s;
static inline uint64_t wb_rnd(void) { uint64_t x = (wb_s += 0x9e3779b97f4a7c15ull); x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull; x = (x ^ (x >> 27)) * 0x94d049bb133111ebull; return x ^ (x >> 31); }
static int wb_dcmp(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static double wb_q(double *v, int n, double q) { qsort(v, (size_t)n, sizeof *v, wb_dcmp); double p = q * (n - 1); int i = (int)p; return i + 1 < n ? v[i] + (p - i) * (v[i + 1] - v[i]) : v[n - 1]; }

int cmd_wbench_stats(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: ufsm wbench-stats A.tsv [B.tsv] [--metric ids,p1x] [--set clean|all] [--ceil-min 0.8] [--n 10000] [--seed 1] [--boxes 0|1]\n"); return 2; }
    const char *pa = argv[2], *pb = argc > 3 && argv[3][0] != '-' ? argv[3] : nullptr;
    static wb_tab A, B; if (wb_load(pa, &A) || (pb && wb_load(pb, &B))) return 1;
    char ml[1024]; snprintf(ml, sizeof ml, "%s", wb_opt(argc, argv, "--metric", "ids"));
    const int clean = strcmp(wb_opt(argc, argv, "--set", "clean"), "all") != 0, nb = atoi(wb_opt(argc, argv, "--n", "10000")), pbox = atoi(wb_opt(argc, argv, "--boxes", "0"));
    const double cmin = atof(wb_opt(argc, argv, "--ceil-min", "0.8")); wb_s = (uint64_t)atoll(wb_opt(argc, argv, "--seed", "1"));
    const int ic = wb_colix(&A, "ceil");
    printf("wbench-stats A = %s%s%s, set %s%s%s\n", pa, pb ? ", B = " : "", pb ? pb : "", clean ? "clean (ceil >= " : "all", clean ? wb_opt(argc, argv, "--ceil-min", "0.8") : "", clean ? "; boxes without a ceil kept)" : "");
    for (char *m = strtok(ml, ","); m; m = strtok(nullptr, ",")) {
        const int ia = wb_colix(&A, m), ibb = pb ? wb_colix(&B, m) : -1;
        if (ia < 0 || (pb && ibb < 0)) { printf("metric %s: not a column of %s\n", m, ia < 0 ? pa : pb); continue; }
        /* boxes: A's rows (in the set) with a value; paired: also scored in B */
        int idx[WB_MAXBOX], jb[WB_MAXBOX], sc[WB_MAXBOX], n = 0, ns = 0; int nx = 0; double sx = 0, sxc = 0; char sname[WB_MAXSC][64];
        for (int r = 0; r < A.nrow; r++) {
            const double v = A.row[r].v[ia]; if (!isfinite(v)) continue;
            if (clean && ic >= 0 && isfinite(A.row[r].v[ic]) && A.row[r].v[ic] < cmin) { nx++; sx += v; sxc += A.row[r].v[ic]; continue; }
            int j = -1; if (pb) { for (int q = 0; q < B.nrow; q++) if (!strcmp(B.row[q].box, A.row[r].box)) j = q; if (j < 0 || !isfinite(B.row[j].v[ibb])) continue; }
            int s = 0; while (s < ns && strcmp(sname[s], A.row[r].scroll)) s++;
            if (s == ns) { if (ns == WB_MAXSC) continue; snprintf(sname[ns++], 64, "%s", A.row[r].scroll); }
            idx[n] = r; jb[n] = j; sc[n] = s; n++;
        }
        if (!n) { printf("metric %s: no boxes\n", m); continue; }
        double ma = 0, mb = 0; for (int i = 0; i < n; i++) { ma += A.row[idx[i]].v[ia]; if (pb) mb += B.row[jb[i]].v[ibb]; } ma /= n; mb /= n;
        int cnt[WB_MAXSC] = {0}, mem[WB_MAXSC][WB_MAXBOX]; for (int i = 0; i < n; i++) mem[sc[i]][cnt[sc[i]]++] = i;
        double *ba = malloc((size_t)nb * sizeof(double)), *bb = malloc((size_t)nb * sizeof(double)), *bd = malloc((size_t)nb * sizeof(double)); int gt = 0;
        for (int k = 0; k < nb; k++) {
            double sa = 0, sb = 0;
            for (int s = 0; s < ns; s++) for (int q = 0; q < cnt[s]; q++) { const int i = mem[s][wb_rnd() % (uint64_t)cnt[s]]; sa += A.row[idx[i]].v[ia]; if (pb) sb += B.row[jb[i]].v[ibb]; }
            ba[k] = sa / n; bb[k] = sb / n; bd[k] = (sa - sb) / n; gt += sa > sb;
        }
        double sd = 0, mdm = 0; for (int k = 0; k < nb; k++) mdm += bd[k]; mdm /= nb; for (int k = 0; k < nb; k++) sd += (bd[k] - mdm) * (bd[k] - mdm); sd = sqrt(sd / (nb > 1 ? nb - 1 : 1));
        printf("%-10s n %3d | A %.4f [%.4f, %.4f]", m, n, ma, wb_q(ba, nb, 0.025), wb_q(ba, nb, 0.975));
        if (pb) printf(" | B %.4f [%.4f, %.4f] | A-B %+.4f [%+.4f, %+.4f] P(A>B) %.3f MDE %.4f", mb, wb_q(bb, nb, 0.025), wb_q(bb, nb, 0.975), ma - mb, wb_q(bd, nb, 0.025), wb_q(bd, nb, 0.975), (double)gt / nb, 2.8 * sd);
        printf("\n");
        if (nx) printf("    label-noise  %3d boxes  A %.4f  (ceil %.4f; not in the mean)\n", nx, sx / nx, sxc / nx);
        for (int s = 0; s < ns; s++) {
            double a = 0, b = 0; for (int q = 0; q < cnt[s]; q++) { a += A.row[idx[mem[s][q]]].v[ia]; if (pb) b += B.row[jb[mem[s][q]]].v[ibb]; }
            printf("    %-12s %3d boxes  A %.4f", sname[s], cnt[s], a / cnt[s]); if (pb) printf("  B %.4f  A-B %+.4f", b / cnt[s], (a - b) / cnt[s]); printf("\n");
        }
        if (pbox) for (int i = 0; i < n; i++) { printf("      %-10s %-12s A %.4f", A.row[idx[i]].box, A.row[idx[i]].scroll, A.row[idx[i]].v[ia]); if (pb) printf("  B %.4f  A-B %+.4f", B.row[jb[i]].v[ibb], A.row[idx[i]].v[ia] - B.row[jb[i]].v[ibb]); printf("\n"); }
        free(ba); free(bb); free(bd);
    }
    return 0;
}
