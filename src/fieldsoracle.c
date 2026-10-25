/* ufsm fields-oracle <fields-dir> <phase-root> <phase-key> <out-dir> [--sign 1] [--fill net|none] [--min-known 0.5]
   A fit-field directory whose phase planes come from a label phase raster (ufsm phase-raster / band field PHASE_* codes: 0..251 =
   phase x 252, 255 unknown) on the native grid of the fields (the raster's voxel = one native voxel; the fields' origin in native
   voxels): per field cell the complex mean of the known voxels' phasors (h^3 voxels), written as pc / ps with |mean| as the confidence
   when at least --min-known of the cell's voxels are known; elsewhere the network's pc / ps (--fill net) or 128 (none). --sign -1
   negates the label phase (label sets stored with the opposite convention). The other planes (normals, spacing, material) are the
   network's (with --fill none the sp plane becomes the label spacing: the cells' mean wrapped label-phase gradient). Output: raw fields (fit_fields_save) of the same box: an oracle-phase / network-rest swap for ufsm fit and, with
   --fill none, the oracle phase for ufsm fit-compare. Prints the share of material cells with a label phase. */
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fit.h"
#include "store.h"
#include "zarr3.h"

static const char *fo_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }

/* --noise-gross P [--noise-blob L] [--noise-sigma S] [--noise-seed N]: fitter tolerance tests. Corrupt the oracle (label) phase cells:
   a share P of the label-phase material cells, in spatially coherent blobs (white noise box-blurred 3x with radius L cells, its top-P
   cells), gets its phase shifted by 0.25..0.75 turn (another blurred field picks the shift, coherent inside a blob: the network's gross
   errors, sheet <-> gap confusions), their confidence x K (--noise-conf, 1); every label cell also gets Gaussian phase jitter of S turn. */
static uint64_t fo_rng(uint64_t *s) { uint64_t z = (*s += 0x9E3779B97F4A7C15ull); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }
static double fo_unif(uint64_t *s) { return (fo_rng(s) >> 11) * 0x1.0p-53; }
static void fo_blur(float *v, int D, int H, int W, int r) {   /* separable box blur, 3 passes along each axis (clamped ends) */
    if (r < 1) return;
    const int dims[3] = {D, H, W}; const size_t str[3] = {(size_t)H * W, (size_t)W, 1};
    const int nmax = D > H ? (D > W ? D : W) : (H > W ? H : W); float *line = malloc((size_t)nmax * sizeof *line), *acc = malloc((size_t)nmax * sizeof *acc);
    for (int pass = 0; pass < 3; pass++) for (int ax = 0; ax < 3; ax++) {
        const int n = dims[ax], a1 = ax == 0 ? 1 : 0, a2 = ax == 2 ? 1 : 2;   /* (the two other axes) */
        for (int i = 0; i < dims[a1]; i++) for (int j = 0; j < dims[a2]; j++) {
            const size_t base = i * str[a1] + j * str[a2];
            for (int k = 0; k < n; k++) line[k] = v[base + k * str[ax]];
            double s = 0; for (int k = -r; k <= r; k++) s += line[k < 0 ? 0 : k >= n ? n - 1 : k];
            for (int k = 0; k < n; k++) { acc[k] = (float)(s / (2 * r + 1)); const int ka = k + r + 1, kr = k - r; s += line[ka >= n ? n - 1 : ka] - line[kr < 0 ? 0 : kr]; }
            for (int k = 0; k < n; k++) v[base + k * str[ax]] = acc[k];
        }
    }
    free(line); free(acc);
}
static int fo_cmpf(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return (x < y) - (x > y); }
int cmd_fields_oracle(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: ufsm fields-oracle <fields-dir> <phase-root> <phase-key> <out-dir> [--sign 1] [--fill net|none] [--min-known 0.5] [--phase-scale F (native voxels per raster voxel, e.g. 4: 2.4 um fields, 9.6 um raster)] [--noise-gross P --noise-blob L --noise-sigma S --noise-seed N --noise-conf K]\n"); return 2; }
    const char *fd = argv[2], *out = argv[5];
    const int sign = atoi(fo_opt(argc, argv, "--sign", "1")), fillnet = strcmp(fo_opt(argc, argv, "--fill", "net"), "none") != 0;
    const double mk = atof(fo_opt(argc, argv, "--min-known", "0.5"));
    if (sign != 1 && sign != -1) { fprintf(stderr, "fields-oracle: --sign 1 or -1\n"); return 2; }
    fit_fields_buf b; if (fit_fields_load(fd, 0, 0, 1, &b)) { fprintf(stderr, "fields-oracle: cannot load %s\n", fd); return 1; }
    z3 *pz = z3_open(store_open(argv[3]), argv[4], nullptr); if (!pz) { fprintf(stderr, "fields-oracle: %s\n", z3_error()); return 1; }
    const int D = b.shape[0], H = b.shape[1], W = b.shape[2]; const size_t HW = (size_t)H * W;
    const int F = atoi(fo_opt(argc, argv, "--phase-scale", "1")), h = b.h / (F > 0 ? F : 1);   /* (raster voxels per field cell) */
    if (F < 1 || b.h % F || b.origin[0] % F || b.origin[1] % F || b.origin[2] % F) { fprintf(stderr, "fields-oracle: --phase-scale must divide the cell (h %d) and the origin\n", b.h); return 2; }
    double ctab[252], stab[252]; for (int k = 0; k < 252; k++) { ctab[k] = cos(2 * M_PI * k / 252.0); stab[k] = sin(sign * 2 * M_PI * k / 252.0); }
    uint8_t *row = malloc((size_t)(h + 1) * (H * h + 1) * (W * h + 1)); size_t nmat = 0, nknown = 0, nsp = 0;
    uint8_t *lab = calloc((size_t)D * HW, 1);   /* (cells holding the label phase) */
    const z3_meta *pm = z3_meta_of(pz);
    for (int z = 0; z < D; z++) {   /* one field row: h native planes of the box (+1 plane / row / column for the label gradient) */
        const int64_t o[3] = {b.origin[0] / F + (int64_t)z * h, b.origin[1] / F, b.origin[2] / F};
        int64_t n[3] = {h + 1, (int64_t)H * h + 1, (int64_t)W * h + 1};
        for (int d = 0; d < 3; d++) if (o[d] + n[d] > pm->shape[d]) n[d] = pm->shape[d] - o[d];
        if (n[0] < h || n[1] < (int64_t)H * h || n[2] < (int64_t)W * h) { fprintf(stderr, "fields-oracle: the box leaves the phase raster\n"); return 1; }
        if (z3_read(pz, o, n, row, 16)) { fprintf(stderr, "fields-oracle: phase read failed: %s\n", z3_error()); return 1; }
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            const size_t i = (size_t)z * HW + (size_t)y * W + x; double complex a = 0; int kn = 0; double gs = 0; int gn = 0;
            for (int dz = 0; dz < h; dz++) for (int dy = 0; dy < h; dy++) for (int dx = 0; dx < h; dx++) {
                const size_t k = ((size_t)dz * n[1] + (size_t)y * h + dy) * n[2] + (size_t)x * h + dx; const uint8_t p = row[k];
                if (p >= 252) continue;
                a += ctab[p] + I * stab[p]; kn++;
                if (dz + 1 < n[0] && (size_t)y * h + dy + 1 < (size_t)n[1] && (size_t)x * h + dx + 1 < (size_t)n[2]) {   /* the label's local sheet frequency */
                    const uint8_t pzn = row[k + (size_t)n[1] * n[2]], pyn = row[k + n[2]], pxn = row[k + 1];
                    if (pzn < 252 && pyn < 252 && pxn < 252) {
                        double e[3] = {(pzn - p) / 252.0, (pyn - p) / 252.0, (pxn - p) / 252.0};
                        for (int q = 0; q < 3; q++) e[q] -= round(e[q]);
                        gs += sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); gn++;
                    }
                }
            }
            if (b.f.w[i]) nmat++;
            if (kn >= mk * h * h * h) {
                a /= kn; if (b.f.w[i]) nknown++; lab[i] = 1;
                b.a[0][i] = (uint8_t)lround(128 + 127 * creal(a)); b.a[1][i] = (uint8_t)lround(128 + 127 * cimag(a));
                if (!fillnet && gn && gs > 0) {   /* --fill none: the label spacing as sp (sp h / 8 = native voxels per turn) */
                    const double spn = gn / gs * F, v = spn * 8.0 / b.h; b.a[8][i] = (uint8_t)(v < 1 ? 1 : v > 255 ? 255 : lround(v)); nsp++;
                }
            } else if (!fillnet) { b.a[0][i] = b.a[1][i] = 128; }
        }
    }
    free(row); z3_close(pz);
    const double ngp = atof(fo_opt(argc, argv, "--noise-gross", "0")), nsg = atof(fo_opt(argc, argv, "--noise-sigma", "0"));
    if (ngp > 0 || nsg > 0) {   /* synthetic network errors on the label cells */
        const int L = atoi(fo_opt(argc, argv, "--noise-blob", "4")); uint64_t rs = (uint64_t)atoll(fo_opt(argc, argv, "--noise-seed", "1")) * 0x2545F4914F6CDD1Dull + 7;
        const size_t nc = (size_t)D * HW; float *f1 = malloc(nc * sizeof *f1), *f2 = malloc(nc * sizeof *f2);
        for (size_t i = 0; i < nc; i++) { f1[i] = (float)fo_unif(&rs); f2[i] = (float)fo_unif(&rs); }
        fo_blur(f1, D, H, W, L); fo_blur(f2, D, H, W, L);
        size_t nl = 0; for (size_t i = 0; i < nc; i++) nl += lab[i] && b.f.w[i];
        float thr = 2; double f2lo = 0, f2hi = 1;
        if (nl && ngp > 0) {   /* the top-P blob values among the label material cells; the shift field rescaled to its 1..99 % range */
            float *t = malloc(nl * sizeof *t); size_t k = 0; for (size_t i = 0; i < nc; i++) if (lab[i] && b.f.w[i]) t[k++] = f1[i];
            qsort(t, nl, sizeof *t, fo_cmpf); size_t q = (size_t)(ngp * nl); thr = q >= nl ? -1 : t[q];
            k = 0; for (size_t i = 0; i < nc; i++) if (lab[i] && b.f.w[i]) t[k++] = f2[i];
            qsort(t, nl, sizeof *t, fo_cmpf); f2hi = t[(size_t)(0.01 * nl)]; f2lo = t[(size_t)(0.99 * (nl - 1))]; free(t);
        }
        size_t ng = 0; const double nck = atof(fo_opt(argc, argv, "--noise-conf", "1"));   /* (the shifted cells' confidence x K: calibrated vs confidently wrong) */
        for (size_t i = 0; i < nc; i++) {
            if (!lab[i]) continue;
            const double c = (b.a[0][i] - 128) / 127.0, s = (b.a[1][i] - 128) / 127.0; double r = hypot(c, s), ph = atan2(s, c) / (2 * M_PI);
            if (b.f.w[i] && f1[i] > thr) { double u = (f2[i] - f2lo) / (f2hi - f2lo > 1e-9 ? f2hi - f2lo : 1); u = u < 0 ? 0 : u > 1 ? 1 : u; ph += 0.25 + 0.5 * u; r *= nck; ng++; }
            if (nsg > 0) { const double u1 = fo_unif(&rs) + 1e-12, u2 = fo_unif(&rs); ph += nsg * sqrt(-2 * log(u1)) * cos(2 * M_PI * u2); }
            b.a[0][i] = (uint8_t)lround(128 + 127 * r * cos(2 * M_PI * ph)); b.a[1][i] = (uint8_t)lround(128 + 127 * r * sin(2 * M_PI * ph));
        }
        fprintf(stderr, "fields-oracle: noise: %zu of %zu label material cells shifted 0.25..0.75 turn (blob radius %d cells), jitter %g turn\n", ng, nl, L, nsg);
        free(f1); free(f2);
    }
    free(lab);
    uint8_t *a10[10]; for (int k = 0; k < 10; k++) a10[k] = b.a[k];
    const long long org[3] = {b.origin[0], b.origin[1], b.origin[2]}; const int shp[3] = {D, H, W};
    const int na = (int)(D * b.h) + 1;
    if (fit_fields_save(out, org, shp, b.h, b.um, a10, b.cy, b.cx, na)) { fprintf(stderr, "fields-oracle: cannot write %s\n", out); return 1; }
    printf("fields-oracle: %dx%dx%d cells, label phase on %.4f of the %zu material cells, label spacing as sp on %zu cells (sign %+d, rest %s) -> %s\n", D, H, W,
           nmat ? (double)nknown / nmat : 0, nmat, nsp, sign, fillnet ? "network" : "unknown", out);
    fit_fields_free(&b);
    return 0;
}
