/* ufsm eval: score a prediction pyramid against a label pyramid over a box (both volcomp zarr v3 groups
   in our encodings: prediction uint8 p*255, labels 0/254/255 with 255 = ignore). Reports precision,
   recall, F1 and Dice at several thresholds, plus band-tolerant recall/precision (a predicted voxel within
   `tol` voxels of a labelled surface voxel counts, and vice versa) which forgives sub-voxel offsets. */
#include "sources.h"
#include "store.h"
#include "zarr3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *opt(int argc, char **argv, const char *name, const char *dflt) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return dflt;
}

/* binary dilation by a ball of radius tol (chebyshev) on a n^3 cube, in place via a copy */
static void dilate(const uint8_t *in, uint8_t *out, const int64_t n[3], int tol) {
    memcpy(out, in, (size_t)n[0] * n[1] * n[2]);
    for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) for (int64_t x = 0; x < n[2]; x++) {
        if (!in[((size_t)z * n[1] + y) * n[2] + x]) continue;
        for (int dz = -tol; dz <= tol; dz++) for (int dy = -tol; dy <= tol; dy++) for (int dx = -tol; dx <= tol; dx++) {
            if (dz * dz + dy * dy + dx * dx > tol * tol) continue;
            int64_t zz = z + dz, yy = y + dy, xx = x + dx;
            if (zz < 0 || zz >= n[0] || yy < 0 || yy >= n[1] || xx < 0 || xx >= n[2]) continue;
            out[((size_t)zz * n[1] + yy) * n[2] + xx] = 1;
        }
    }
}

int cmd_eval(int argc, char **argv) {
    if (argc < 8) {
        fprintf(stderr, "usage: ufsm eval <pred-root> <pred-group> <label-root> <label-group> --um U [--level 0] [--box z,y,x,nz,ny,nx] [--tol 2] [--pred-origin z,y,x] [--dump slice.pgm] [--thr 0.3,0.5,0.7]\n"
                        "  the prediction's origin_zyx (attribute written by predict) is subtracted from label coordinates\n");
        return 2;
    }
    const char *proot = argv[2], *pkey = argv[3], *lroot = argv[4], *lkey = argv[5];
    double um = atof(opt(argc, argv, "--um", "0"));
    int level = atoi(opt(argc, argv, "--level", "0")), tol = atoi(opt(argc, argv, "--tol", "2"));
    if (um <= 0) { fprintf(stderr, "--um required\n"); return 2; }
    store *ps = store_open(proot), *ls = store_open(lroot);
    z3 *pz = pyramid_open_level(ps, pkey, level, um, nullptr), *lz = pyramid_open_level(ls, lkey, level, um, nullptr);
    if (!pz || !lz) { fprintf(stderr, "cannot open %s or %s at level %d: %s\n", pkey, lkey, level, z3_error()); return 1; }
    const z3_meta *pm = z3_meta_of(pz);
    long long po[3] = {0, 0, 0};
    { const char *s = opt(argc, argv, "--pred-origin", nullptr); if (s) sscanf(s, "%lld,%lld,%lld", &po[0], &po[1], &po[2]); }
    int64_t o[3] = {0, 0, 0}, n[3] = {pm->shape[0], pm->shape[1], pm->shape[2]};
    { const char *b = opt(argc, argv, "--box", nullptr); long long v[6]; if (b && sscanf(b, "%lld,%lld,%lld,%lld,%lld,%lld", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) == 6) for (int d = 0; d < 3; d++) { o[d] = v[d]; n[d] = v[3 + d]; } }
    size_t nv = (size_t)n[0] * n[1] * n[2];
    if (nv > (size_t)1 << 31) { fprintf(stderr, "box too large (%zu voxels); use --box\n", nv); return 1; }
    uint8_t *p = malloc(nv), *l = malloc(nv);
    int64_t lo[3] = {o[0] + (po[0] >> level), o[1] + (po[1] >> level), o[2] + (po[2] >> level)};
    if (z3_read(pz, o, n, p, 0) || z3_read(lz, lo, n, l, 0)) { fprintf(stderr, "read: %s\n", z3_error()); return 1; }
    { const char *dp = opt(argc, argv, "--dump", nullptr);   /* middle z slice as a PGM: [CT |] prediction | label (ignore = dark grey) */
      if (dp) { FILE *f = fopen(dp, "wb"); if (f) {
        uint8_t *ct = nullptr;   /* optional CT panel: --ct-root R --ct G [--cache DIR] (same level, label coordinates) */
        const char *cr = opt(argc, argv, "--ct-root", nullptr), *cg = opt(argc, argv, "--ct", nullptr);
        if (cr && cg) { store *cs = store_open(cr); z3 *cz = cs ? pyramid_open_level(cs, cg, level, um, opt(argc, argv, "--cache", nullptr)) : nullptr;
            if (cz) { ct = malloc(nv); int64_t co[3] = {lo[0] + n[0] / 2, lo[1], lo[2]}, cn[3] = {1, n[1], n[2]}; if (z3_read(cz, co, cn, ct, 0)) { free(ct); ct = nullptr; } } }
        fprintf(f, "P5\n%lld %lld\n255\n", (ct ? 3 : 2) * (long long)n[2], (long long)n[1]);
        size_t z0 = (size_t)(n[0] / 2) * n[1] * n[2];
        for (int64_t y = 0; y < n[1]; y++) { if (ct) fwrite(ct + (size_t)y * n[2], 1, (size_t)n[2], f); fwrite(p + z0 + (size_t)y * n[2], 1, (size_t)n[2], f);
            for (int64_t x = 0; x < n[2]; x++) { uint8_t v = l[z0 + (size_t)y * n[2] + x]; fputc(v == 255 ? 48 : v >= 127 ? 255 : 0, f); } }
        fclose(f); fprintf(stderr, "dumped z=%lld slice to %s\n", (long long)(n[0] / 2), dp); } } }
    /* valid voxels: label not ignore */
    size_t nvalid = 0, npos = 0;
    for (size_t i = 0; i < nv; i++) { if (l[i] != 255) { nvalid++; npos += l[i] >= 127; } }
    printf("box %lldx%lldx%lld at level %d: %zu valid voxels (%.1f%%), %zu labelled surface (%.2f%% of valid)\n", (long long)n[0], (long long)n[1], (long long)n[2], level, nvalid, 100.0 * nvalid / nv, npos, nvalid ? 100.0 * npos / nvalid : 0);
    if (!nvalid) return 0;
    printf("thr    prec   recall     f1   dice  | band(tol=%d) prec recall\n", tol);
    uint8_t *gt = malloc(nv), *gtd = malloc(nv), *pr = malloc(nv), *prd = malloc(nv);
    for (size_t i = 0; i < nv; i++) gt[i] = l[i] != 255 && l[i] >= 127;
    dilate(gt, gtd, n, tol);
    double thrs[16] = {0.3, 0.5, 0.7}; int nthr = 3;   /* --thr 0.1,0.2,0.3 overrides */
    { const char *ts = opt(argc, argv, "--thr", nullptr); if (ts) { nthr = 0; char *t = strdup(ts); for (char *q = strtok(t, ","); q && nthr < 16; q = strtok(nullptr, ",")) thrs[nthr++] = atof(q); free(t); } }
    for (int t = 0; t < nthr; t++) {
        uint8_t th = (uint8_t)(thrs[t] * 255);
        size_t tp = 0, fp = 0, fn = 0, bp_hit = 0, bp_tot = 0;
        for (size_t i = 0; i < nv; i++) {
            pr[i] = p[i] >= th;
            if (l[i] == 255) continue;
            tp += pr[i] && gt[i]; fp += pr[i] && !gt[i]; fn += !pr[i] && gt[i];
            if (pr[i]) { bp_tot++; bp_hit += gtd[i]; }
        }
        dilate(pr, prd, n, tol);
        size_t br_hit = 0, br_tot = 0;
        for (size_t i = 0; i < nv; i++) if (l[i] != 255 && gt[i]) { br_tot++; br_hit += prd[i]; }
        double prec = tp + fp ? (double)tp / (tp + fp) : 0, rec = tp + fn ? (double)tp / (tp + fn) : 0;
        double f1 = prec + rec ? 2 * prec * rec / (prec + rec) : 0;
        /* soft dice with p/255 against hard labels */
        double sp = 0, ss = 0, sg = 0;
        for (size_t i = 0; i < nv; i++) if (l[i] != 255) { double q = p[i] / 255.0; sp += q * gt[i]; ss += q; sg += gt[i]; }
        double dice = (2 * sp + 1) / (ss + sg + 1);
        printf("%.2f  %.4f  %.4f  %.4f  %.4f  |  %.4f  %.4f\n", thrs[t], prec, rec, f1, t == 0 ? dice : dice, bp_tot ? (double)bp_hit / bp_tot : 0, br_tot ? (double)br_hit / br_tot : 0);
    }
    free(p); free(l); free(gt); free(gtd); free(pr); free(prd);
    z3_close(pz); z3_close(lz); store_close(ps); store_close(ls);
    return 0;
}
