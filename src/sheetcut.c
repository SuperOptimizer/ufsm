/* ufsm sheet-cut <wraps-dir> <out-dir> [--tol 0.3] [--cos 0.35] [--min-cm2 0.25] [--max 8] [--margin 2] [--wraps a,b,..] [--um U]
   Square flat sheets from whole wraps (user 10-21: no smeared sheets; [[feedback-square-flat-sheets]]): per wrap (tifxyz dir or
   .sfc in <wraps-dir>) a quad (r, c)-(r+1, c+1) is GOOD when its four vertices are valid, its two row edges and two column edges
   are within [1 - tol, 1 + tol] of the wrap's nominal spacings (median edge length along rows / columns over the wrap) and its
   corner angle is near 90 deg (|cos| <= --cos). A sheet switch (a radial jump of about one pitch between vertex rows or columns)
   or a shear breaks the isometry and fails the test. Then greedily the largest axis-aligned rectangle of GOOD quads (histogram
   stack, O(rows x cols)), removed with --margin quads around it, again, until --max rectangles per wrap or the next one is under
   --min-cm2. Each rectangle -> <out-dir>/<wrap>_rK (tifxyz, sfc when the source was sfc: no; plain tifxyz) with its vertices;
   area = GOOD quads x row spacing x column spacing (um from meta voxel_size_um, or --um). Table on stdout:
   wrap rect r0 c0 rows cols cm2 good-share-of-wrap stretch-row stretch-col; summary line: total cm2, wraps with a sheet. */
#include "imgops.h"
#include "tiffw.h"
#include <dirent.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static const char *sc_opt(int argc, char **argv, const char *k, const char *d) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1];
    return d;
}
static int cmpf(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }
static double median_f(float *v, size_t n) { if (!n) return 0; qsort(v, n, sizeof *v, cmpf); return v[n / 2]; }

/* the largest all-ones rectangle of m (h x w): histogram stack per row. Area in cells; corner / size out. */
static long long largest_rect(const uint8_t *m, int h, int w, int *r0, int *c0, int *rh, int *cw) {
    int *ht = calloc((size_t)w + 1, sizeof *ht), *st = malloc(((size_t)w + 2) * sizeof *st); long long best = 0;
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) ht[c] = m[(size_t)r * w + c] ? ht[c] + 1 : 0;
        ht[w] = 0; int sp = 0;
        for (int c = 0; c <= w; c++) {
            while (sp && ht[st[sp - 1]] >= ht[c]) {
                const int hh = ht[st[--sp]], left = sp ? st[sp - 1] + 1 : 0; const long long a = (long long)hh * (c - left);
                if (a > best) { best = a; *r0 = r - hh + 1; *c0 = left; *rh = hh; *cw = c - left; }
            }
            st[sp++] = c;
        }
    }
    free(ht); free(st); return best;
}

int cmd_sheet_cut(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm sheet-cut <wraps-dir> <out-dir> [--tol 0.3] [--cos 0.35] [--min-cm2 0.25] [--max 8] [--margin 2] [--wraps a,b,..] [--um U]\n"); return 2; }
    const double tol = atof(sc_opt(argc, argv, "--tol", "0.3")), cmax = atof(sc_opt(argc, argv, "--cos", "0.35")), mincm2 = atof(sc_opt(argc, argv, "--min-cm2", "0.25"));
    const int maxr = atoi(sc_opt(argc, argv, "--max", "8")), margin = atoi(sc_opt(argc, argv, "--margin", "2")); const double um_opt = atof(sc_opt(argc, argv, "--um", "0"));
    const char *wl = sc_opt(argc, argv, "--wraps", nullptr); char *want = nullptr; if (wl) { want = malloc(strlen(wl) + 3); sprintf(want, ",%s,", wl); }
    DIR *dd = opendir(argv[2]); if (!dd) { fprintf(stderr, "sheet-cut: cannot open %s\n", argv[2]); return 1; }
    { char p[1400]; snprintf(p, sizeof p, "mkdir -p '%s'", argv[3]); if (system(p)) { fprintf(stderr, "sheet-cut: cannot create %s\n", argv[3]); return 1; } }
    printf("wrap\trect\tr0\tc0\trows\tcols\tcm2\tgood_share\trow_um\tcol_um\n");
    struct dirent *de; int nw = 0, nws = 0, ns = 0, rc = 0; double tot = 0;
    while ((de = readdir(dd))) {
        if (de->d_name[0] == '.') continue;
        char nm[300]; snprintf(nm, sizeof nm, "%s", de->d_name); { const size_t l = strlen(nm); if (l > 4 && !strcmp(nm + l - 4, ".sfc")) nm[l - 4] = 0; }
        if (want) { char k[310]; snprintf(k, sizeof k, ",%s,", nm); if (!strstr(want, k)) continue; }
        char s[1400], p[1500]; snprintf(s, sizeof s, "%s/%s", argv[2], de->d_name); snprintf(p, sizeof p, "%s/meta.json", s);
        if (!tfx_exists(p)) continue;
        char *mb = tfx_meta_read(p); double vum = um_opt; int wid = -1;
        if (mb) { const char *q = strstr(mb, "\"voxel_size_um\":"); if (q && !(vum > 0)) vum = atof(q + 16); q = strstr(mb, "\"winding\":"); if (q) wid = atoi(q + 10); free(mb); }
        if (!(vum > 0)) vum = 1;
        static const char *ax[3] = {"x", "y", "z"}; float *A[3] = {0}; int w = 0, h = 0, ok = 1;
        for (int k = 0; k < 3 && ok; k++) { int w2, h2; snprintf(p, sizeof p, "%s/%s.tif", s, ax[k]); A[k] = img_read_f32(p, &w2, &h2); ok = A[k] && (!k || (w2 == w && h2 == h)); w = w2; h = h2; }
        if (!ok || w < 2 || h < 2) { fprintf(stderr, "sheet-cut: cannot read %s\n", s); rc = 1; for (int k = 0; k < 3; k++) free(A[k]); continue; }
        nw++;
        const float *X = A[0], *Y = A[1], *Z = A[2];
#define VOK(i) (Z[i] >= 0 && X[i] >= 0 && Y[i] >= 0)
#define DST(i, j) sqrtf((X[i] - X[j]) * (X[i] - X[j]) + (Y[i] - Y[j]) * (Y[i] - Y[j]) + (Z[i] - Z[j]) * (Z[i] - Z[j]))
        /* nominal spacings: median edge length along a row (column step) and along a column (row step) */
        float *ev = malloc((size_t)w * h * sizeof *ev); size_t ne = 0;
        for (int r = 0; r < h; r++) for (int c = 0; c + 1 < w; c++) { const size_t i = (size_t)r * w + c; if (VOK(i) && VOK(i + 1)) ev[ne++] = DST(i, i + 1); }
        const double sc = median_f(ev, ne); ne = 0;
        for (int r = 0; r + 1 < h; r++) for (int c = 0; c < w; c++) { const size_t i = (size_t)r * w + c; if (VOK(i) && VOK(i + w)) ev[ne++] = DST(i, i + w); }
        const double sr = median_f(ev, ne); free(ev);
        if (!(sc > 0) || !(sr > 0)) { for (int k = 0; k < 3; k++) free(A[k]); continue; }
        const int qh = h - 1, qw = w - 1; uint8_t *m = calloc((size_t)qh * qw, 1); long long ngood = 0, nvq = 0;
        for (int r = 0; r < qh; r++) for (int c = 0; c < qw; c++) {
            const size_t a = (size_t)r * w + c, b = a + 1, d = a + w, e = d + 1;
            if (!(VOK(a) && VOK(b) && VOK(d) && VOK(e))) continue;
            nvq++;
            const double l1 = DST(a, b) / sc, l2 = DST(d, e) / sc, l3 = DST(a, d) / sr, l4 = DST(b, e) / sr;
            if (fabs(l1 - 1) > tol || fabs(l2 - 1) > tol || fabs(l3 - 1) > tol || fabs(l4 - 1) > tol) continue;
            const double ux = X[b] - X[a], uy = Y[b] - Y[a], uz = Z[b] - Z[a], vx = X[d] - X[a], vy = Y[d] - Y[a], vz = Z[d] - Z[a];
            const double cs = (ux * vx + uy * vy + uz * vz) / (sqrt(ux * ux + uy * uy + uz * uz) * sqrt(vx * vx + vy * vy + vz * vz) + 1e-12);
            if (fabs(cs) > cmax) continue;
            m[(size_t)r * qw + c] = 1; ngood++;
        }
        const double qcm2 = sc * sr * vum * vum * 1e-8; int got = 0;
        for (int k = 0; k < maxr; k++) {
            int r0 = 0, c0 = 0, rh = 0, cw = 0; const long long a = largest_rect(m, qh, qw, &r0, &c0, &rh, &cw);
            if (!a || a * qcm2 < mincm2) break;
            const int vw = cw + 1, vh = rh + 1; float *O[3];
            for (int t = 0; t < 3; t++) { O[t] = malloc((size_t)vw * vh * sizeof(float)); for (int r = 0; r < vh; r++) memcpy(O[t] + (size_t)r * vw, A[t] + (size_t)(r0 + r) * w + c0, (size_t)vw * sizeof(float)); }
            char op[1500]; snprintf(op, sizeof op, "%s/%s_r%d", argv[3], nm, k);
            /* step: the vertex spacing in native voxels (meta scale 1 / step): the mean of the two nominal spacings */
            if (tifxyz_write(op, vw, vh, O[0], O[1], O[2], 0.5 * (sc + sr), vum, wid, "ufsm sheet-cut")) { fprintf(stderr, "sheet-cut: cannot write %s\n", op); rc = 1; }
            for (int t = 0; t < 3; t++) free(O[t]);
            printf("%s\t%d\t%d\t%d\t%d\t%d\t%.4f\t%.4f\t%.2f\t%.2f\n", nm, k, r0, c0, vh, vw, a * qcm2, nvq ? (double)ngood / nvq : 0, sr * vum, sc * vum);
            tot += a * qcm2; ns++; got = 1;
            const int ra = r0 - margin < 0 ? 0 : r0 - margin, rb = r0 + rh + margin > qh ? qh : r0 + rh + margin, ca = c0 - margin < 0 ? 0 : c0 - margin, cb = c0 + cw + margin > qw ? qw : c0 + cw + margin;
            for (int r = ra; r < rb; r++) memset(m + (size_t)r * qw + ca, 0, (size_t)(cb - ca));
        }
        nws += got;
        free(m); for (int k = 0; k < 3; k++) free(A[k]);
#undef VOK
#undef DST
    }
    closedir(dd); free(want);
    printf("# sheet-cut: %d wraps, %d with a sheet, %d sheets, %.3f cm2 in all (tol %.2f, cos %.2f, min %.2f cm2)\n", nw, nws, ns, tot, tol, cmax, mincm2);
    return rc;
}
