/* ufsm surf-resample (2026-11-18, the unroll's r3once: the wraps' ink from the stripes' maps). By the ufsm binary (argv[1]), on a synthetic
   sheet (a cylinder section, radius 90): two source surfaces with their own parametrisations (theta 0.20..1.45 and 1.30..2.50, different
   angular steps, the second with a hole), maps = f(3-D point) on their render canvases (scale 0.25 grids: 4 px a vertex), and one
   destination surface across both (theta 0.30..2.40, its own step and z offset). Checked:
   - the resampled destination map equals f at the destination's points (mean |d| < 1, p99 < 4: bilinear sampling of a smooth f);
   - pixels over the source hole are unsampled (<d>_src.tif 0), the rest sampled; a residual surface is written that covers the hole
     (+ its margin) and nothing far from it, and --compose fills exactly the unsampled pixels from a residual map (here f itself);
   - --edge-band: pixels near the switch between the two sources are unsampled and join the residual;
   - --geom-only 1 (no maps): byte-identical residual surfaces to the full run, with and without --fill-gap, no output maps; with --watch
     (the destination linked in while it runs, then the stop file) the same;
   - >= 95 % of the vertices project within 0.25 voxels (the log's histogram; the hole's rim is the rest). */
#include "imgops.h"
#include "tiffw.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)
static double f3(double x, double y, double z) { return 128 + 90 * sin(x * 0.05 + 0.3) * cos(z * 0.035) + 20 * sin(y * 0.07); }
typedef struct { int w, h; float *X, *Y, *Z; } grid;
static grid cyl(double th0, double dth, int w, double z0, int h, int hole) {   /* rows z (step 4), columns theta */
    grid g = {w, h, malloc(sizeof(float) * w * h), malloc(sizeof(float) * w * h), malloc(sizeof(float) * w * h)};
    for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) {
        const size_t i = (size_t)r * w + c; const double th = th0 + c * dth;
        g.X[i] = (float)(128 + 90 * cos(th)); g.Y[i] = (float)(126 + 90 * sin(th)); g.Z[i] = (float)(z0 + 4.0 * r);
        if (hole && r >= 20 && r < 30 && c >= 8 && c < 16) g.X[i] = g.Y[i] = g.Z[i] = -1;
    }
    return g;
}
static uint8_t *canvas(const grid *g, int *ow, int *oh, uint8_t *valid) {   /* f on render's canvas (gs 4, corners valid) */
    const double gs = 4; *ow = render_out_px(g->w, gs); *oh = render_out_px(g->h, gs); uint8_t *m = calloc((size_t)*ow * *oh, 1);
    for (int py = 0; py < *oh; py++) for (int px = 0; px < *ow; px++) {
        int r0 = (int)(py / gs), c0 = (int)(px / gs); r0 = r0 > g->h - 2 ? g->h - 2 : r0; c0 = c0 > g->w - 2 ? g->w - 2 : c0;
        const double v = py / gs - r0, u = px / gs - c0; const size_t i = (size_t)r0 * g->w + c0, j[4] = {i, i + 1, i + g->w, i + g->w + 1};
        if (!(g->Z[j[0]] >= 0 && g->Z[j[1]] >= 0 && g->Z[j[2]] >= 0 && g->Z[j[3]] >= 0)) continue;
        const double wt[4] = {(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v}; double q[3] = {0, 0, 0};
        for (int k = 0; k < 4; k++) { q[0] += wt[k] * g->X[j[k]]; q[1] += wt[k] * g->Y[j[k]]; q[2] += wt[k] * g->Z[j[k]]; }
        const double val = f3(q[0], q[1], q[2]); m[(size_t)py * *ow + px] = (uint8_t)(val < 0 ? 0 : val > 255 ? 255 : val + 0.5);
        if (valid) valid[(size_t)py * *ow + px] = 1;
    }
    return m;
}
static void put(const char *dir, const char *name, const grid *g, int map, const char *mapdir) {
    char p[1200]; snprintf(p, sizeof p, "%s/%s", dir, name);
    CHECK(!tifxyz_write(p, g->w, g->h, g->X, g->Y, g->Z, 4.0, 9.6, 0, "test_surf_resample"), "tifxyz_write %s", p);
    if (map) { int ow, oh; uint8_t *m = canvas(g, &ow, &oh, nullptr); snprintf(p, sizeof p, "%s/%s_reader3.tif", mapdir, name); CHECK(!tiffw_write(p, ow, oh, 1, m, 1), "map %s", p); free(m); }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: test_surf_resample <ufsm>\n"); return 2; }
    char dir[] = "/tmp/test_surf_resample_XXXXXX"; CHECK(mkdtemp(dir) != nullptr, "mkdtemp");
    char p[1400], cmd[6000];
    snprintf(p, sizeof p, "mkdir -p '%s/src' '%s/maps' '%s/dst' '%s/rmap'", dir, dir, dir, dir); CHECK(!system(p), "mkdir");
    const double r = 90;   /* source steps ~4 voxels along the arc (dth 4 / r), the destination's 3.3 */
    grid a = cyl(0.20, 4.0 / r, (int)((1.45 - 0.20) / (4.0 / r)) + 1, 10, 48, 0), b = cyl(1.30, 4.4 / r, (int)((2.50 - 1.30) / (4.4 / r)) + 1, 12, 47, 1);
    grid d = cyl(0.30, 3.3 / r, (int)((2.40 - 0.30) / (3.3 / r)) + 1, 23.0, 40, 0);
    snprintf(p, sizeof p, "%s/src", dir); char mp[1200]; snprintf(mp, sizeof mp, "%s/maps", dir);
    put(p, "sa", &a, 1, mp); put(p, "sb", &b, 1, mp);
    snprintf(p, sizeof p, "%s/dst", dir); put(p, "w000", &d, 0, nullptr);
    int ow, oh; uint8_t *valid = nullptr; { int w0 = render_out_px(d.w, 4), h0 = render_out_px(d.h, 4); valid = calloc((size_t)w0 * h0, 1); }
    uint8_t *ref = canvas(&d, &ow, &oh, valid); const size_t no = (size_t)ow * oh;
    for (int band = 0; band < 2; band++) {
        snprintf(cmd, sizeof cmd, "'%s' surf-resample '%s/src' '%s/maps' '%s/dst' '%s/o%d' --threads 3 --src-out 1 --residual '%s/res%d' --residual-margin 2 --band-margin 3 --fill-gap 0%s 2> '%s/o%d.log'",
                 argv[1], dir, dir, dir, dir, band, dir, band, band ? " --edge-band 6" : "", dir, band);
        CHECK(!system(cmd), "surf-resample (band %d) failed (%s/o%d.log)", band, dir, band);
        snprintf(p, sizeof p, "%s/o%d", dir, band); int w2, h2, w3, h3;
        char q[1400]; snprintf(q, sizeof q, "%s/w000_reader3.tif", p); uint8_t *m = img_read_u8(q, &w2, &h2);
        snprintf(q, sizeof q, "%s/w000_src.tif", p); uint8_t *s = img_read_u8(q, &w3, &h3);
        CHECK(m && s && w2 == ow && h2 == oh && w3 == ow && h3 == oh, "outputs %dx%d / %dx%d, expected %dx%d", w2, h2, w3, h3, ow, oh);
        if (!m || !s || w2 != ow || h2 != oh) { free(m); free(s); continue; }
        size_t ns = 0, nu = 0, big = 0; double sum = 0;
        for (size_t i = 0; i < no; i++) { if (!valid[i]) { CHECK(!s[i], "a pixel off the destination surface is marked sampled"); continue; }
            if (s[i]) { const int e = abs((int)m[i] - (int)ref[i]); sum += e; big += e > 4; ns++; } else nu++; }
        CHECK(ns > no / 3 && sum / ns < 1.0 && big < ns / 100, "band %d: resampled vs f: mean |d| %.3f over %zu px, %zu above 4", band, ns ? sum / ns : 0, ns, big);
        /* the hole: the source b's rows 20..29 x columns 8..15 (z 92..128, theta 1.30 + 8..15 x 4.4 / r): unsampled there */
        size_t nh = 0, nhs = 0;
        for (int py = 0; py < oh; py++) for (int px = 0; px < ow; px++) { const size_t i = (size_t)py * ow + px; if (!valid[i]) continue;
            const double z = 23.0 + py, th = 0.30 + px / 4.0 * (3.3 / r);
            if (z > 98 && z < 122 && th > 1.30 + 9.5 * 4.4 / r && th < 1.30 + 13.5 * 4.4 / r) { nh++; nhs += s[i] != 0; } }
        CHECK(nh > 100 && nhs == 0, "band %d: %zu of %zu pixels inside the source hole marked sampled", band, nhs, nh);
        snprintf(q, sizeof q, "%s/res%d/w000/z.tif", dir, band); int rw, rh; float *RZ = img_read_f32(q, &rw, &rh);
        CHECK(RZ && rw == d.w && rh == d.h, "band %d: no residual surface %s", band, q);
        if (RZ) { size_t nr = 0, far = 0; for (int rr = 0; rr < rh; rr++) for (int c = 0; c < rw; c++) { if (!(RZ[(size_t)rr * rw + c] >= 0)) continue; nr++;
                      const double z = 23.0 + 4.0 * rr, th = 0.30 + c * 3.3 / r;
                      const int nearhole = z > 60 && z < 160 && th > 1.30 + 2 * 4.4 / r && th < 1.30 + 22 * 4.4 / r, nearsw = th > 1.00 && th < 1.75;   /* (the sources overlap at 1.30..1.45: the pick may alternate there) */
                      far += !(nearhole || (band && nearsw)); }
                  CHECK(nr > 10 && far == 0, "band %d: residual surface %zu vertices, %zu far from the hole%s", band, nr, far, band ? " / the switch" : "");
                  if (band) { size_t nsw = 0; for (int rr = 0; rr < rh; rr++) for (int c = 0; c < rw; c++) { const double th = 0.30 + c * 3.3 / r, z = 23.0 + 4.0 * rr;
                                  nsw += RZ[(size_t)rr * rw + c] >= 0 && th > 1.30 && th < 1.45 && (z < 60 || z > 160); }
                              CHECK(nsw > 10, "the edge band: no residual vertices at the switch between the sources (%zu)", nsw); }
                  free(RZ); }
        if (band == 0) {   /* compose: the unsampled pixels from a residual map (f itself on the destination canvas) */
            snprintf(q, sizeof q, "%s/rmap/w000_reader3.tif", dir); CHECK(!tiffw_write(q, ow, oh, 1, ref, 1), "rmap");
            snprintf(cmd, sizeof cmd, "'%s' surf-resample --compose '%s/o0' '%s/rmap' 2> '%s/compose.log'", argv[1], dir, dir, dir); CHECK(!system(cmd), "compose failed");
            snprintf(q, sizeof q, "%s/o0/w000_reader3.tif", dir); uint8_t *c = img_read_u8(q, &w2, &h2); size_t bad = 0;
            for (size_t i = 0; c && i < no; i++) { if (!valid[i]) continue; if (!s[i] && c[i] != ref[i]) bad++; if (s[i] && c[i] != m[i]) bad++; }
            CHECK(c && !bad, "compose: %zu pixels neither the resampled value (sampled) nor the residual map's (unsampled)", bad);
            free(c);
        }
        {   double sh = -1; snprintf(q, sizeof q, "%s/o%d.log", dir, band); FILE *f = fopen(q, "r"); char ln[2000];
            while (f && fgets(ln, sizeof ln, f)) { const char *t = strstr(ln, "4 voxels: "); if (t) sh = atof(t + 10); } if (f) fclose(f);
            CHECK(sh >= 0.95, "band %d: %.3f of the vertices within 0.25 voxels (expected >= 0.95; only the source hole's rim is farther)", band, sh); }
        for (int fg = 0; fg < 2; fg++) {   /* --geom-only (the unroll's early residual, no maps): the same residual surfaces as the full run, no map written */
            const char *fx = fg ? "4" : "0"; const char *ed = band ? " --edge-band 6" : "";
            snprintf(cmd, sizeof cmd, "'%s' surf-resample '%s/src' '%s/maps' '%s/dst' '%s/f%d%d' --threads 2 --residual '%s/fres%d%d' --residual-margin 2 --band-margin 3 --fill-gap %s%s 2> '%s/f%d%d.log' && "
                     "'%s' surf-resample '%s/src' - '%s/dst' '%s/g%d%d' --threads 3 --geom-only 1 --residual '%s/gres%d%d' --residual-margin 2 --band-margin 3 --fill-gap %s%s 2> '%s/g%d%d.log' && "
                     "for x in x y z; do cmp -s '%s/fres%d%d/w000/'$x.tif '%s/gres%d%d/w000/'$x.tif || exit 1; done && cmp -s '%s/fres%d%d/w000/meta.json' '%s/gres%d%d/w000/meta.json' && [ ! -e '%s/g%d%d' ]",
                     argv[1], dir, dir, dir, dir, band, fg, dir, band, fg, fx, ed, dir, band, fg,
                     argv[1], dir, dir, dir, band, fg, dir, band, fg, fx, ed, dir, band, fg,
                     dir, band, fg, dir, band, fg, dir, band, fg, dir, band, fg, dir, band, fg);
            CHECK(!system(cmd), "band %d fill-gap %s: --geom-only residual differs from the full run's (or wrote maps): %s/gres%d%d vs fres%d%d", band, fx, dir, band, fg, band, fg);
        }
        {   /* --watch STOP: DST filled while it runs (the unroll's per-wrap early residual): the same residual as the full run */
            const char *ed = band ? " --edge-band 6" : "";
            snprintf(cmd, sizeof cmd, "mkdir -p '%s/wd%d' && ( '%s' surf-resample '%s/src' - '%s/wd%d' '%s/wg%d' --threads 2 --geom-only 1 --watch '%s/wd%d.stop' "
                     "--residual '%s/wres%d' --residual-margin 2 --band-margin 3 --fill-gap 0%s 2> '%s/w%d.log' & p=$!; sleep 1; ln -s '%s/dst/w000' '%s/wd%d/w000'; "
                     "sleep 1; touch '%s/wd%d.stop'; wait $p ) && for x in x y z; do cmp -s '%s/res%d/w000/'$x.tif '%s/wres%d/w000/'$x.tif || exit 1; done",
                     dir, band, argv[1], dir, dir, band, dir, band, dir, band, dir, band, ed, dir, band, dir, dir, band, dir, band, dir, band, dir, band);
            CHECK(!system(cmd), "band %d: --watch residual differs from the full run's (%s/wres%d, %s/w%d.log)", band, dir, band, dir, band);
        }
        fprintf(stderr, "test_surf_resample: band %d: %zu px sampled (mean |d| vs f %.3f), %zu unsampled, the hole and the residual right\n", band, ns, ns ? sum / ns : 0, nu);
        free(m); free(s);
    }
    free(ref); free(valid);
    if (!fails) { snprintf(p, sizeof p, "rm -rf '%s'", dir); if (system(p)) {} }
    if (fails) { fprintf(stderr, "test_surf_resample: %d failures (kept %s)\n", fails, dir); return 1; }
    printf("test_surf_resample: OK\n");
    return 0;
}
