/* ufsm count-anat DUMP OUT --gauge G [--cy CY --cx CX] [--down 2] [--ct ROOT KEY --ct-um U --um U0 --org Z,Y,X]
   Error anatomy of a fitted winding against GT codes on one thin z range: DUMP = fit --codes ... --score-z Z0,NZ --score-dump F run with
   UFSM_SCORE_DUMP_U=1 (records z y x d U, box-local native; d = U - q mod 14). With the box gauge G (fit score "gauge"):
   e = wrap14(d - G), k = round(e) (the integer error: 0 right sheet, +1 the fit one turn ahead), q = U - d.
   Writes OUT_e.ppm (y x image, --down D native voxels per pixel: k 0 grey .. white by |frac e|, +1 red, -1 blue, +2 orange, -2 cyan,
   other magenta; unlabelled black), with --ct OUT_ov.ppm (the CT row grey, the labelled GT pixels coloured by k: 0 green, +1 red,
   -1 blue, +2 orange, -2 cyan, other magenta), and prints per 1 mm radius band the shares of k = -2..+2 and the integer-error change rates per mm along radial and
   tangential neighbours (pixels of the image grid: a change between two labelled neighbour pixels = a wrong-count boundary). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "store.h"
#include "zarr3.h"

static const char *ca_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 4; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static int ca_idx(int argc, char **argv, const char *k) { for (int i = 4; i < argc; i++) if (!strcmp(argv[i], k)) return i; return -1; }
static double ca_w14(double e) { return e - 14 * round(e / 14); }
static int ca_ppm(const char *p, const uint8_t *rgb, int w, int h) {
    FILE *f = fopen(p, "wb"); if (!f) return 1; fprintf(f, "P6\n%d %d\n255\n", w, h);
    const int bad = fwrite(rgb, 3, (size_t)w * h, f) != (size_t)w * h; return fclose(f) || bad;
}
int cmd_count_anat(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm count-anat <dump5> <out-prefix> --gauge G --cy CY --cx CX [--down 2] [--ct ROOT KEY --ct-um U --um U0 --org Z,Y,X]\n"); return 2; }
    const double G = atof(ca_opt(argc, argv, "--gauge", "0")), cy = atof(ca_opt(argc, argv, "--cy", "0")), cx = atof(ca_opt(argc, argv, "--cx", "0")), um = atof(ca_opt(argc, argv, "--um", "9.6"));
    const int D = atoi(ca_opt(argc, argv, "--down", "2"));
    FILE *f = fopen(argv[2], "rb"); if (!f) { fprintf(stderr, "count-anat: cannot open %s\n", argv[2]); return 1; }
    fseek(f, 0, SEEK_END); const size_t n = (size_t)ftell(f) / 20; fseek(f, 0, SEEK_SET);
    float *r = malloc(n * 20); if (!r || fread(r, 20, n, f) != n) { fprintf(stderr, "count-anat: read failed\n"); return 1; } fclose(f);
    double ymax = 0, xmax = 0, zmin = 1e30, zmax = -1e30;
    for (size_t i = 0; i < n; i++) { const float *v = r + 5 * i; if (v[1] > ymax) ymax = v[1]; if (v[2] > xmax) xmax = v[2]; if (v[0] < zmin) zmin = v[0]; if (v[0] > zmax) zmax = v[0]; }
    const int W = (int)(xmax / D) + 1, H = (int)(ymax / D) + 1;
    float *E = malloc(sizeof(float) * W * H), *Q = malloc(sizeof(float) * W * H), *Uu = malloc(sizeof(float) * W * H);
    for (size_t i = 0; i < (size_t)W * H; i++) E[i] = Q[i] = Uu[i] = NAN;
    for (size_t i = 0; i < n; i++) {   /* the last record per pixel (thin z range) */
        const float *v = r + 5 * i; const int px = (int)(v[2] / D), py = (int)(v[1] / D); const size_t j = (size_t)py * W + px;
        E[j] = (float)ca_w14(v[3] - G); Uu[j] = v[4]; Q[j] = (float)(v[4] - v[3]);
    }
    printf("count-anat: %zu records, z %.0f..%.0f, image %dx%d (%d native / px)\n", n, zmin, zmax, W, H, D);
    uint8_t *img = calloc((size_t)W * H, 3);
    for (size_t j = 0; j < (size_t)W * H; j++) {
        if (E[j] != E[j]) continue;
        const int k = (int)lround(E[j]); const double fr = fabs(E[j] - k); uint8_t *p = img + 3 * j;
        const double b = 1 - fr;   /* 1 at the centre of the integer */
        if (k == 0) { p[0] = p[1] = p[2] = (uint8_t)(90 + 165 * b); }
        else if (k == 1) { p[0] = (uint8_t)(150 + 105 * b); p[1] = p[2] = 30; }
        else if (k == -1) { p[2] = (uint8_t)(150 + 105 * b); p[0] = p[1] = 40; }
        else if (k == 2) { p[0] = 255; p[1] = (uint8_t)(140 * b); p[2] = 0; }
        else if (k == -2) { p[0] = 0; p[1] = (uint8_t)(200 * b); p[2] = 220; }
        else { p[0] = 200; p[1] = 0; p[2] = 200; }
    }
    char path[4096]; snprintf(path, sizeof path, "%s_e.ppm", argv[3]); if (ca_ppm(path, img, W, H)) { fprintf(stderr, "count-anat: cannot write %s\n", path); return 1; }
    /* stats per 1 mm radius band: k shares; integer-error changes between radial / tangential labelled neighbour pixels per mm */
    enum { NB = 64 }; double sh[NB][6] = {{0}}, ch[NB][4] = {{0}};   /* k -2..2 + other; radial pairs, radial changes, tangential pairs, changes */
    for (int y = 1; y + 1 < H; y++) for (int x = 1; x + 1 < W; x++) {
        const size_t j = (size_t)y * W + x; if (E[j] != E[j]) continue;
        const double ry = (y + 0.5) * D - cy, rx = (x + 0.5) * D - cx, rr = hypot(ry, rx); const int bi = (int)(rr * um / 1000); if (bi >= NB || rr < 1) continue;
        const int k = (int)lround(E[j]); sh[bi][k >= -2 && k <= 2 ? k + 2 : 5]++;
        const double uy = ry / rr, ux = rx / rr;
        for (int dd = 0; dd < 2; dd++) {   /* the neighbour one pixel outward (radial) / along the tangent */
            const double sy = dd ? -ux : uy, sx = dd ? uy : ux; const int y2 = (int)lround(y + sy), x2 = (int)lround(x + sx);
            if (y2 < 0 || x2 < 0 || y2 >= H || x2 >= W) continue; const size_t j2 = (size_t)y2 * W + x2; if (E[j2] != E[j2]) continue;
            ch[bi][2 * dd]++; ch[bi][2 * dd + 1] += lround(E[j2]) != k;
        }
    }
    printf("  radius(mm)   pixels   k-2    k-1    k0     k+1    k+2   other | changes per mm: radial  tangential\n");
    for (int b = 0; b < NB; b++) { double t = 0; for (int k = 0; k < 6; k++) t += sh[b][k]; if (t < 200) continue;
        const double pxmm = 1000.0 / (um * D);
        printf("  %4d-%-4d %9.0f  %.3f  %.3f  %.3f  %.3f  %.3f  %.3f |  %6.2f  %6.2f\n", b, b + 1, t, sh[b][0] / t, sh[b][1] / t, sh[b][2] / t, sh[b][3] / t, sh[b][4] / t, sh[b][5] / t,
               ch[b][0] ? ch[b][1] / ch[b][0] * pxmm : 0, ch[b][2] ? ch[b][3] / ch[b][2] * pxmm : 0); }
    const int ci = ca_idx(argc, argv, "--ct");
    if (ci > 0 && ci + 2 < argc) {   /* the overlay: CT row (mean over the dump's z range) + GT / fitted sheets */
        const double ctum = atof(ca_opt(argc, argv, "--ct-um", "0")), F = um / ctum; long long org[3] = {0, 0, 0};
        if (sscanf(ca_opt(argc, argv, "--org", "0,0,0"), "%lld,%lld,%lld", &org[0], &org[1], &org[2]) != 3 || !(ctum > 0)) { fprintf(stderr, "count-anat: --ct needs --ct-um and --org Z,Y,X\n"); return 2; }
        z3 *z = z3_open(store_open(argv[ci + 1]), argv[ci + 2], nullptr); if (!z) { fprintf(stderr, "count-anat: %s\n", z3_error()); return 1; }
        const z3_meta *zm = z3_meta_of(z);
        int64_t o3[3] = {(int64_t)floor((org[0] + 0.5 * (zmin + zmax)) * F), (int64_t)floor(org[1] * F), (int64_t)floor(org[2] * F)}, n3[3] = {1, (int64_t)ceil(H * D * F), (int64_t)ceil(W * D * F)};
        for (int d = 1; d < 3; d++) if (o3[d] + n3[d] > zm->shape[d]) n3[d] = zm->shape[d] - o3[d];
        uint8_t *ct = malloc((size_t)n3[1] * n3[2]); if (!ct || z3_read(z, o3, n3, ct, 16)) { fprintf(stderr, "count-anat: CT read failed\n"); return 1; }
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            const int64_t cyy = (int64_t)((y + 0.5) * D * F), cxx = (int64_t)((x + 0.5) * D * F); uint8_t *p = img + 3 * ((size_t)y * W + x);
            const uint8_t g = cyy < n3[1] && cxx < n3[2] ? ct[cyy * n3[2] + cxx] : 0; p[0] = p[1] = p[2] = g;
            const size_t j = (size_t)y * W + x; if (E[j] != E[j]) continue;
            const int k = (int)lround(E[j]);   /* the labelled (GT sheet) pixels coloured by the integer error */
            static const uint8_t pal[7][3] = {{0, 220, 220}, {60, 60, 255}, {0, 230, 0}, {255, 40, 40}, {255, 160, 0}, {255, 0, 255}, {255, 0, 255}};
            const int c = k >= -2 && k <= 2 ? k + 2 : 5; p[0] = pal[c][0]; p[1] = pal[c][1]; p[2] = pal[c][2];
        }
        snprintf(path, sizeof path, "%s_ov.ppm", argv[3]); if (ca_ppm(path, img, W, H)) return 1;
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {   /* the CT row alone (OUT_ct.ppm) */
            const int64_t cyy = (int64_t)((y + 0.5) * D * F), cxx = (int64_t)((x + 0.5) * D * F); uint8_t *p = img + 3 * ((size_t)y * W + x);
            p[0] = p[1] = p[2] = cyy < n3[1] && cxx < n3[2] ? ct[cyy * n3[2] + cxx] : 0;
        }
        snprintf(path, sizeof path, "%s_ct.ppm", argv[3]); if (ca_ppm(path, img, W, H)) return 1;
        free(ct); z3_close(z);
    }
    free(img); free(E); free(Q); free(Uu); free(r);
    return 0;
}
