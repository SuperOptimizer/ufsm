/* train --ink-mil: the normal-pooled MIL ink loss in the trainer (inkmil.h kernels in head_slab_pass, rays from the sampler).
   (1) sample_mil_map (CPU): a source-grid point / normal through the window's symmetry and warp lands where sample_warp_ink moves a
       blob centred there (the centroid of the warped blob within 0.15 voxels, the normal within 3 degrees of the warped finite step),
       for 8 symmetries x {identity, rotation + elastic, rotation + elastic + affine + zoom};
   (2) on the synthetic spiral (tests/make_wind_fixture.c, N 256) with one spiral sheet as a tifxyz segment (ink stripes on its
       canvas) and the source's "ink" target on the fly: train --task unwrap2 --ink-head 1 --ink-mil 1 --ink-dice 0.25 runs a few
       steps with the symmetries and warps on; the log's "ink mil" is finite, positive, on > 0 windows with rays; at --ink-mil 0 the
       first step's loss equals the --ink-mil run's minus its mil term (same seed, lr 0).
   Usage: test_ink_mil_train <ufsm binary> <make_wind_fixture binary> */
#include "inksplat.h"
#include "sample.h"
#include "spatial_augment.h"
#include "tiffw.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)
static int sh(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static int sh(const char *fmt, ...) {
    char cmd[8192]; va_list a; va_start(a, fmt); vsnprintf(cmd, sizeof cmd, fmt, a); va_end(a);
    const int r = system(cmd); if (r) fprintf(stderr, "command failed (%d): %s\n", r, cmd); return r;
}

/* (1) the map against the image warp */
static void blob_centroid(const uint8_t *img, int P, double c[3]) {
    double s = 0, m[3] = {0};
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
        const int v = img[((size_t)z * P + y) * P + x]; if (v == 255 || !v) continue;
        s += v; m[0] += v * (double)z; m[1] += v * (double)y; m[2] += v * (double)x; }
    for (int d = 0; d < 3; d++) c[d] = s > 0 ? m[d] / s : -1;
}
static void blob(uint8_t *img, int P, const double p[3]) {   /* a smooth bump of radius 3 at p (0 elsewhere: known) */
    memset(img, 0, (size_t)P * P * P);
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
        const double r2 = (z - p[0]) * (z - p[0]) + (y - p[1]) * (y - p[1]) + (x - p[2]) * (x - p[2]);
        if (r2 < 16) img[((size_t)z * P + y) * P + x] = (uint8_t)lrint(200 * (1 - r2 / 16) * (1 - r2 / 16)); }
}
static void test_map(void) {
    enum { P = 48 };
    uint8_t *a = malloc((size_t)P * P * P), *b = malloc((size_t)P * P * P);
    static const int perms[8][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {2, 1, 0}, {1, 2, 0}, {2, 0, 1}, {0, 1, 2}, {2, 0, 1}};
    static const int flips[8][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 1}, {1, 1, 0}, {0, 0, 1}, {1, 1, 1}, {1, 0, 1}, {0, 1, 0}};
    double worst = 0, worst_n = 0; int n_ok = 0;
    for (int g = 0; g < 3; g++) for (int s = 0; s < 8; s++) {
        spatial_aug sp; spatial_aug_make(&sp, 1000 + 7 * s + g, P, g ? 12.f : 0.f, g ? 1.f : 0.f, g ? 1.5f : 0.f, g ? 1.f : 0.f);
        if (g == 2) { spatial_aug_affine(&sp, 99 + s, .08, .05); spatial_aug_zoom(&sp, .9); }
        sp.active = 1;   /* (sample_warp_ink always pulls through the map; the identity map is exact) */
        const double p[3] = {20.3 + s * .4, 25.7 - s * .3, 22.1 + .2 * s}, nv[3] = {.6, -.48, .64};
        inkray r; memset(&r, 0, sizeof r);
        for (int d = 0; d < 3; d++) { r.p[d] = (float)p[d]; r.n[d] = (float)nv[d]; }
        int kept = sample_mil_map(&r, 1, perms[s], flips[s], &sp, P, P);
        CHECK(kept == 1, "geometry %d symmetry %d: the ray left the window", g, s); if (kept != 1) continue;
        double c0[3], c1[3];
        blob(a, P, p); sample_warp_ink(a, b, P, perms[s], flips[s], &sp); blob_centroid(b, P, c0);
        const double p2[3] = {p[0] + 3 * nv[0], p[1] + 3 * nv[1], p[2] + 3 * nv[2]};   /* the normal: the warped blob of p + 3 n */
        blob(a, P, p2); sample_warp_ink(a, b, P, perms[s], flips[s], &sp); blob_centroid(b, P, c1);
        double e = 0, dn = 0, dl = 0;
        for (int d = 0; d < 3; d++) { e += (c0[d] - r.p[d]) * (c0[d] - r.p[d]); dl += (c1[d] - c0[d]) * (c1[d] - c0[d]); }
        dl = sqrt(dl); for (int d = 0; d < 3; d++) dn += (c1[d] - c0[d]) / dl * r.n[d];
        const double ang = acos(fmin(1, dn)) * 180 / M_PI; e = sqrt(e);
        if (e > worst) worst = e;
        if (ang > worst_n) worst_n = ang;
        n_ok += e < .15 && ang < 3;
        CHECK(e < .15 && ang < 3, "geometry %d symmetry %d: the mapped ray (%.2f %.2f %.2f) vs the warped blob (%.2f %.2f %.2f): %.3f voxels, normal %.2f deg",
              g, s, r.p[0], r.p[1], r.p[2], c0[0], c0[1], c0[2], e, ang);
    }
    fprintf(stderr, "sample_mil_map: %d of 24 maps within 0.15 voxels / 3 deg (worst %.3f voxels, %.2f deg)\n", n_ok, worst, worst_n);
    free(a); free(b);
}

/* (2) the fixture's spiral sheet (winding fraction 0.15, inside the papyrus band f < .3) as a tifxyz segment */
static void fx_axis_at(double z, double *cy, double *cx, int N) { *cy = .5 * N + .03 * (z - .5 * N); *cx = .5 * N - .02 * (z - .5 * N); }
static int write_segment(const char *dir, int N) {
    const double STEP = 4, f0 = .15, t0 = 1.2, t1 = 3.8;   /* turns along the sheet (radius 24 t .. at pitch 24) */
    const int GH = (int)((N - 16) / STEP), GW = (int)((t1 - t0) * 2 * M_PI * 60 / STEP);
    float *X = malloc((size_t)GW * GH * 4), *Y = malloc((size_t)GW * GH * 4), *Z = malloc((size_t)GW * GH * 4);
    for (int v = 0; v < GH; v++) for (int u = 0; u < GW; u++) {
        const double z = 8 + STEP * v, ph = 2 * M_PI * (t0 + (t1 - t0) * u / (GW - 1)), r = 24 * (f0 + ph / (2 * M_PI) - .002 * z);
        double cy, cx; fx_axis_at(z, &cy, &cx, N);
        X[v * GW + u] = (float)(cx + r * cos(ph)); Y[v * GW + u] = (float)(cy + r * sin(ph)); Z[v * GW + u] = (float)z; }
    char p[600]; snprintf(p, sizeof p, "%s/seg.tifxyz", dir);
    if (tifxyz_write(p, GW, GH, X, Y, Z, STEP, 9.6, -1, "test_ink_mil_train")) return -1;
    const int CW = (int)(GW * STEP), CH = (int)(GH * STEP);
    uint8_t *lab = malloc((size_t)CW * CH), *msk = malloc((size_t)CW * CH);
    for (int r = 0; r < CH; r++) for (int c = 0; c < CW; c++) { lab[(size_t)r * CW + c] = ((r / 24 + c / 40) & 1) ? 255 : 0; msk[(size_t)r * CW + c] = 255; }
    snprintf(p, sizeof p, "%s/inklabels.tif", dir); if (tiffw_write(p, CW, CH, 1, lab, 1)) return -1;
    snprintf(p, sizeof p, "%s/mask.tif", dir); if (tiffw_write(p, CW, CH, 1, msk, 1)) return -1;
    snprintf(p, sizeof p, "%s/segs.json", dir); FILE *f = fopen(p, "w"); if (!f) return -1;
    fprintf(f, "{\"segments\": [{\"tifxyz\": \"%s/seg.tifxyz\", \"labels\": \"%s/inklabels.tif\", \"mask\": \"%s/mask.tif\"}]}\n", dir, dir, dir); fclose(f);
    free(X); free(Y); free(Z); free(lab); free(msk); return 0;
}
static double log_last(const char *path, const char *key, int *found) {   /* the value after the last "key" in the log */
    FILE *f = fopen(path, "r"); *found = 0; if (!f) return NAN;
    char ln[8192]; double v = NAN;
    while (fgets(ln, sizeof ln, f)) { const char *q = strstr(ln, key); if (q) { v = atof(q + strlen(key)); *found = 1; } }
    fclose(f); return v;
}
static double log_first(const char *path, const char *key) {
    FILE *f = fopen(path, "r"); if (!f) return NAN;
    char ln[8192]; double v = NAN;
    while (fgets(ln, sizeof ln, f)) { const char *q = strncmp(ln, "step ", 5) ? nullptr : strstr(ln, key); if (q) { v = atof(q + strlen(key)); break; } }   /* (the step lines only) */
    fclose(f); return v;
}

int main(int argc, char **argv) {
    test_map();
    if (argc != 3) { fprintf(stderr, "usage: test_ink_mil_train <ufsm binary> <make_wind_fixture binary>\n"); return 2; }
    const char *U = argv[1];
    char d[] = "/tmp/test_inkmil_XXXXXX"; if (!mkdtemp(d)) return 1;
    CHECK(!sh("%s %s/fx 256 2 1 > %s/fx.log 2>&1", argv[2], d, d), "fixture");
    char p[700]; snprintf(p, sizeof p, "%s/fx", d);
    CHECK(!write_segment(p, 256), "segment");
    snprintf(p, sizeof p, "%s/fx/sources-mil.json", d); FILE *f = fopen(p, "w"); if (!f) return 1;
    fprintf(f, "{\"sources\": [{\"name\": \"wind-fixture-mil\", \"root\": \"%s/fx\", \"ct\": \"ct\", \"um\": 9.6, \"axis\": \"%s/fx/axis.json\", \"weight\": 1, \"ink_weight\": 1, \"iso\": 100,\n"
               "  \"targets\": {\"ink\": {\"onfly\": {\"segments\": \"%s/fx/segs.json\", \"level\": 0, \"cache_dir\": \"%s/dcache\"}},\n"
               "               \"band\": {\"root\": \"%s/fx/band\", \"key\": \"1\", \"radius\": 80, \"span\": 75, \"phase\": {\"root\": \"%s/fx/phase0\", \"key\": \"0\", \"level\": 0}}}}]}\n", d, d, d, d, d, d);
    fclose(f);
    const char *common = "--task unwrap2 --P 64 --B 1 --gpus 0 --workers 2 --sampler-slots 4 --val-batches 2 --val-every 1000 --log-every 1 "
                         "--ckpt-every 1000 --fp4 2 --input-prec 8 --warmup 0 --seed 5 --wind4-tile 0 --levels 1 --widths 16,32,64 --ink-head 1 --ink-dice 0.25";
    CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-mil.json --out %s/a --steps 6 --ink-mil 1 --ink-mil-mode 0 %s > %s/a.log 2>&1", U, d, d, common, d), "train --ink-mil (%s/a.log)", d);
    snprintf(p, sizeof p, "%s/a.log", d); int fm = 0, fw = 0;
    const double mil = log_last(p, " ink mil ", &fm); (void)fw;
    CHECK(fm && isfinite(mil) && mil > 0 && mil < 5, "ink mil logged %d, value %g (%s)", fm, mil, p);
    {   /* windows with rays: the "(N% of windows" after the value */
        FILE *g = fopen(p, "r"); char ln[8192]; double best = 0, rays = 0;
        while (g && fgets(ln, sizeof ln, g)) { const char *q = strstr(ln, " ink mil "); if (!q) continue; const char *o = strchr(q, '(');
            if (o) { const double w = atof(o + 1); if (w > best) best = w; const char *rr = strstr(o, "windows, "); if (rr) { const double x = atof(rr + 9); if (x > rays) rays = x; } } }
        if (g) fclose(g);
        fprintf(stderr, "ink mil: last %.4f, at most %.0f%% of windows with rays, up to %.0f rays a window\n", mil, best, rays);
        CHECK(best > 0 && rays > 0, "no window drew rays (%s)", p);
    }
    /* the loss bookkeeping: lr 0, same seed: the step-1 loss of --ink-mil 1 = that of --ink-mil 0 + the mil term */
    CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-mil.json --out %s/b1 --steps 1 --lr 0 --muon-lr 0 --ink-mil 1 --ink-mil-mode 0 %s > %s/b1.log 2>&1", U, d, d, common, d), "b1");
    CHECK(!sh("UFSM_GRAD_MX8=1 %s train %s/fx/sources-mil.json --out %s/b0 --steps 1 --lr 0 --muon-lr 0 %s > %s/b0.log 2>&1", U, d, d, common, d), "b0");
    char p1[700], p0[700]; snprintf(p1, sizeof p1, "%s/b1.log", d); snprintf(p0, sizeof p0, "%s/b0.log", d);
    const double l1 = log_first(p1, " loss "), l0 = log_first(p0, " loss "); int fb = 0; const double m1 = log_last(p1, " ink mil ", &fb);
    fprintf(stderr, "lr 0 step 1: loss %.5f with mil %.5f, %.5f without\n", l1, m1, l0);
    CHECK(fb && fabs(l1 - (l0 + m1)) < 2e-3 + 1e-3 * fabs(l1), "loss %g != %g + %g", l1, l0, m1);
    if (!fails) sh("rm -rf %s", d);
    fprintf(stderr, "%s\n", fails ? "test_ink_mil_train: FAILED" : "test_ink_mil_train: ok");
    return fails != 0;
}
