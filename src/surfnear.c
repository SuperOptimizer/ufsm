/* ufsm surf-near <query-tifxyz> <wraps-dir> [--radius 40] [--out PREFIX] [--matched DIR]
   A foreign surface against our fitted wraps (2026-11-26, PHerc1203, lead: a public VC3D auto-grown single-sheet surface vs the -p fit:
   "per-vertex distance to our nearest wrap, which wrap ids it follows, wrap-align of their mesh vs ours over the same area"). Both in the
   same level-0 voxel frame (x y z). Every valid query vertex gets the nearest vertex of any wrap (tifxyz subdirectories of WRAPS-DIR, the
   winding id from the name's digits) within --radius voxels, and the distance to that wrap's tangent plane there (the normal from the
   wrap grid's central differences): |d_plane| is the sheet-normal offset, free of the wraps' vertex spacing.
   Search: query vertices hashed into --radius cells; each wrap's vertices inside the query's bounding box + radius stream through.
   Out (stderr): plane-distance p50 / p90 and shares <= 2 / 5 voxels, the wraps followed (vertex shares), the id changes between grid
   neighbours (a single-sheet surface that follows one fitted wrap has none; a change = the surface crosses from one fitted wrap to the
   next), unmatched share. PREFIX.tsv: row col wrap d_plane d_vertex (unmatched: wrap -1). --matched DIR: a tifxyz on the query's grid with
   the matched wrap points (our surface over the same area, for wrap-align), unmatched vertices invalid (-1). */
#include "imgops.h"
#include "optcheck.h"
#include "tiffw.h"
#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *sn_opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 2; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(name))) return argv[i + 1];
    return dflt;
}
static int sn_read(const char *dir, float *X[3], int *gw, int *gh) {
    const char *nm[3] = {"x", "y", "z"}; char p[2200]; int w[3], h[3];
    for (int c = 0; c < 3; c++) { snprintf(p, sizeof p, "%s/%s.tif", dir, nm[c]); if (!(X[c] = img_read_f32(p, &w[c], &h[c]))) { for (int k = 0; k < c; k++) free(X[k]); return -1; } }
    if (w[1] != w[0] || w[2] != w[0] || h[1] != h[0] || h[2] != h[0]) { for (int c = 0; c < 3; c++) free(X[c]); return -1; }
    *gw = w[0]; *gh = h[0]; return 0;
}
static int sn_cmp_f(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }

int cmd_surf_near(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm surf-near <query-tifxyz> <wraps-dir> [--radius 40] [--out PREFIX] [--matched DIR]\n"); return 2; }
    const char *qd = argv[2], *wd = argv[3], *pre = sn_opt(argc, argv, "--out", nullptr), *md = sn_opt(argc, argv, "--matched", nullptr);
    const double R = atof(sn_opt(argc, argv, "--radius", "40"));
    if (R <= 0) { fprintf(stderr, "surf-near: --radius > 0\n"); return 2; }
    float *Q[3]; int qw, qh; if (sn_read(qd, Q, &qw, &qh)) { fprintf(stderr, "surf-near: cannot read the query %s\n", qd); return 1; }
    const size_t nq = (size_t)qw * qh;
    double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30}; size_t nv = 0;
    for (size_t i = 0; i < nq; i++) { if (Q[2][i] < 0) continue; nv++; for (int c = 0; c < 3; c++) { lo[c] = fmin(lo[c], Q[c][i]); hi[c] = fmax(hi[c], Q[c][i]); } }
    if (!nv) { fprintf(stderr, "surf-near: the query has no valid vertex\n"); return 1; }
    /* the query hashed into R cells over its bbox */
    int nc[3]; for (int c = 0; c < 3; c++) nc[c] = (int)floor((hi[c] - lo[c]) / R) + 1;
    const size_t ncell = (size_t)nc[0] * nc[1] * nc[2];
    int *head = malloc(sizeof(int) * ncell), *next = malloc(sizeof(int) * nq);
    float *best = malloc(sizeof(float) * nq), *dpl = malloc(sizeof(float) * nq), *MX[3] = {malloc(sizeof(float) * nq), malloc(sizeof(float) * nq), malloc(sizeof(float) * nq)};
    int *wid = malloc(sizeof(int) * nq);
    if (!head || !next || !best || !dpl || !wid || !MX[0] || !MX[1] || !MX[2]) { fprintf(stderr, "surf-near: out of memory\n"); return 1; }
    for (size_t k = 0; k < ncell; k++) head[k] = -1;
    for (size_t i = 0; i < nq; i++) {
        best[i] = (float)(R * R); wid[i] = -1; dpl[i] = NAN; MX[0][i] = MX[1][i] = MX[2][i] = -1;
        if (Q[2][i] < 0) continue;
        int g[3]; for (int c = 0; c < 3; c++) g[c] = (int)((Q[c][i] - lo[c]) / R);
        const size_t k = ((size_t)g[2] * nc[1] + g[1]) * nc[0] + g[0]; next[i] = head[k]; head[k] = (int)i;
    }
    /* stream every wrap */
    DIR *dd = opendir(wd); if (!dd) { fprintf(stderr, "surf-near: cannot open %s\n", wd); return 1; }
    struct dirent *de; int nw = 0; size_t nin = 0;
    while ((de = readdir(dd))) {
        if (de->d_name[0] == '.') continue;
        const char *dg = de->d_name; while (*dg && (*dg < '0' || *dg > '9')) dg++; if (!*dg) continue;
        const int id = atoi(dg);
        char p[2100]; snprintf(p, sizeof p, "%s/%s", wd, de->d_name);
        char q[2200]; snprintf(q, sizeof q, "%s/z.tif", p); FILE *t = fopen(q, "rb"); if (!t) continue; fclose(t);
        float *W[3]; int ww, wh; if (sn_read(p, W, &ww, &wh)) { fprintf(stderr, "surf-near: cannot read the wrap %s\n", p); return 1; }
        nw++;
        for (int r = 0; r < wh; r++) for (int c = 0; c < ww; c++) {
            const size_t j = (size_t)r * ww + c; if (W[2][j] < 0) continue;
            const double x = W[0][j], y = W[1][j], z = W[2][j];
            if (x < lo[0] - R || y < lo[1] - R || z < lo[2] - R || x > hi[0] + R || y > hi[1] + R || z > hi[2] + R) continue;
            nin++;
            const double P[3] = {x, y, z}; int g0[3], g1[3];
            for (int k = 0; k < 3; k++) { g0[k] = (int)floor((P[k] - R - lo[k]) / R); g1[k] = (int)floor((P[k] + R - lo[k]) / R); if (g0[k] < 0) g0[k] = 0; if (g1[k] > nc[k] - 1) g1[k] = nc[k] - 1; }
            int haveN = 0; double n[3] = {0, 0, 0};
            for (int gz = g0[2]; gz <= g1[2]; gz++) for (int gy = g0[1]; gy <= g1[1]; gy++) for (int gx = g0[0]; gx <= g1[0]; gx++)
                for (int i = head[((size_t)gz * nc[1] + gy) * nc[0] + gx]; i >= 0; i = next[i]) {
                    const double dx = Q[0][i] - x, dy = Q[1][i] - y, dz = Q[2][i] - z, d2 = dx * dx + dy * dy + dz * dz;
                    if (d2 >= best[i]) continue;
                    if (!haveN) {   /* the wrap's normal at (r, c): cross product of the grid's central differences */
                        haveN = 1;
                        const int c0 = c > 0 ? c - 1 : c, c1 = c + 1 < ww ? c + 1 : c, r0 = r > 0 ? r - 1 : r, r1 = r + 1 < wh ? r + 1 : r;
                        const size_t a = (size_t)r * ww + c0, b = (size_t)r * ww + c1, e = (size_t)r0 * ww + c, f = (size_t)r1 * ww + c;
                        if (W[2][a] >= 0 && W[2][b] >= 0 && W[2][e] >= 0 && W[2][f] >= 0) {
                            const double u[3] = {W[0][b] - W[0][a], W[1][b] - W[1][a], W[2][b] - W[2][a]}, v[3] = {W[0][f] - W[0][e], W[1][f] - W[1][e], W[2][f] - W[2][e]};
                            n[0] = u[1] * v[2] - u[2] * v[1]; n[1] = u[2] * v[0] - u[0] * v[2]; n[2] = u[0] * v[1] - u[1] * v[0];
                            const double l = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]); if (l > 0) for (int k = 0; k < 3; k++) n[k] /= l; else n[0] = n[1] = n[2] = 0;
                        }
                    }
                    best[i] = (float)d2; wid[i] = id; MX[0][i] = (float)x; MX[1][i] = (float)y; MX[2][i] = (float)z;
                    dpl[i] = n[0] == 0 && n[1] == 0 && n[2] == 0 ? (float)sqrt(d2) : (float)fabs(dx * n[0] + dy * n[1] + dz * n[2]);
                }
        }
        for (int k = 0; k < 3; k++) free(W[k]);
    }
    closedir(dd);
    /* summary */
    size_t nm = 0, le2 = 0, le5 = 0; float *ds = malloc(sizeof(float) * nv); int maxid = 0;
    for (size_t i = 0; i < nq; i++) { if (Q[2][i] < 0 || wid[i] < 0) continue; ds[nm++] = dpl[i]; le2 += dpl[i] <= 2; le5 += dpl[i] <= 5; if (wid[i] > maxid) maxid = wid[i]; }
    qsort(ds, nm, sizeof *ds, sn_cmp_f);
    size_t *cnt = calloc((size_t)maxid + 1, sizeof *cnt); for (size_t i = 0; i < nq; i++) if (Q[2][i] >= 0 && wid[i] >= 0) cnt[wid[i]]++;
    size_t pairs = 0, chg = 0;
    for (int r = 0; r < qh; r++) for (int c = 0; c < qw; c++) for (int e = 0; e < 2; e++) {
        const int r2 = r + e, c2 = c + !e; if (r2 >= qh || c2 >= qw) continue;
        const size_t i = (size_t)r * qw + c, j = (size_t)r2 * qw + c2; if (wid[i] < 0 || wid[j] < 0) continue;
        pairs++; chg += wid[i] != wid[j];
    }
    fprintf(stderr, "surf-near: %s: %zu valid vertices (%d x %d grid), %d wraps (%zu of their vertices near it), radius %.0f\n", qd, nv, qw, qh, nw, nin, R);
    fprintf(stderr, "surf-near: matched %zu (%.3f); plane distance p50 %.2f p90 %.2f voxels; <= 2: %.3f, <= 5: %.3f\n", nm, (double)nm / nv,
            nm ? ds[nm / 2] : NAN, nm ? ds[(size_t)(nm * 0.9)] : NAN, nm ? (double)le2 / nm : NAN, nm ? (double)le5 / nm : NAN);
    fprintf(stderr, "surf-near: wrap id changes between grid neighbours: %zu of %zu pairs (%.4f)\n", chg, pairs, pairs ? (double)chg / pairs : NAN);
    fprintf(stderr, "surf-near: wraps followed (share of matched vertices >= 0.01):");
    for (int k = 0; k <= maxid; k++) if (cnt[k] && (double)cnt[k] / nm >= 0.01) fprintf(stderr, " w%d %.3f", k, (double)cnt[k] / nm);
    fprintf(stderr, "\n");
    if (pre) {
        char p[2100]; snprintf(p, sizeof p, "%s.tsv", pre); FILE *f = fopen(p, "w"); if (!f) { fprintf(stderr, "surf-near: cannot write %s\n", p); return 1; }
        fprintf(f, "row\tcol\twrap\td_plane\td_vertex\n");
        for (int r = 0; r < qh; r++) for (int c = 0; c < qw; c++) { const size_t i = (size_t)r * qw + c; if (Q[2][i] < 0) continue;
            fprintf(f, "%d\t%d\t%d\t%.2f\t%.2f\n", r, c, wid[i], wid[i] < 0 ? NAN : dpl[i], wid[i] < 0 ? NAN : sqrt(best[i])); }
        fclose(f);
    }
    if (md && tifxyz_write(md, qw, qh, MX[0], MX[1], MX[2], 1.0, 0, -1, "ufsm surf-near (the query's nearest wrap points)")) { fprintf(stderr, "surf-near: cannot write %s\n", md); return 1; }
    for (int c = 0; c < 3; c++) { free(Q[c]); free(MX[c]); }
    free(head); free(next); free(best); free(dpl); free(wid); free(ds); free(cnt);
    return 0;
}
