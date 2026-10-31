/* ufsm wrap-snap on a synthetic scroll: concentric wavy sheets (CT gaussian ridges, radius r_k = 30 + 12 k + 2 sin(3 theta + z / 20)
   around a vertical axis), wraps k = 1..6 written as .sfc at r_k + 4 (in the gap, closer to sheet k than to k + 1), a masked wedge
   (CT 0 for theta in [1, 1.4] beyond radius 50). wrap-snap --target ct must: move the wraps onto their own sheets (mean |r - r_k|
   well under a voxel), keep z and the ray angle (rows / columns), keep the order r_k < r_k+1 at every vertex, leave the masked wedge's
   vertices where they were, and write the same bytes with 1 and 4 threads. */
#include "tiffw.h"
#include "z3w.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cmd_wrap_snap(int argc, char **argv);
static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", m); fails++; } } while (0)
#define NZ 128
#define NY 256
#define NX 256
#define C0 128.0
static double sheet_r(int k, double th, double z) { return 30 + 12 * k + 2 * sin(3 * th + z / 20); }
static float *none(const char *p, int *w, int *h) { (void)p; (void)w; (void)h; return nullptr; }
static float *rd(const char *dir, const char *ax, int *w, int *h) {   /* a .sfc axis via the tifxyz hooks */
    char p[600]; snprintf(p, sizeof p, "%s/%s.tif", dir, ax);
    return tfx_axis_read(p, w, h, none);
}
int main(void) {
    char root[256]; snprintf(root, sizeof root, "%s/test_wrapsnap_%d", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp", (int)getpid());
    char cmd[1400], p[600]; snprintf(cmd, sizeof cmd, "rm -rf %s && mkdir -p %s/in", root, root); if (system(cmd)) return 1;
    /* the CT (4 shards of 128^3) */
    { const int64_t sh[3] = {NZ, NY, NX}; snprintf(p, sizeof p, "%s/ct/0", root); z3w *w = z3w_create(p, sh, 128, 0, 0, nullptr); if (!w) return 1;
      uint8_t *d = malloc(128 * 128 * 128);
      for (int sy = 0; sy < 2; sy++) for (int sx = 0; sx < 2; sx++) {
          for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) for (int x = 0; x < 128; x++) {
              const double Y = sy * 128 + y - C0, X = sx * 128 + x - C0, r = sqrt(X * X + Y * Y), th = atan2(Y, X);
              double v = 20; for (int k = 0; k < 9; k++) { const double e = r - sheet_r(k, th, z); v += 200 * exp(-e * e / (2 * 1.2 * 1.2)); }
              if (th >= 1 && th <= 1.4 && r > 50) v = 0;
              if (v > 255) v = 255;
              d[((size_t)z * 128 + y) * 128 + x] = (uint8_t)lround(v);
          }
          if (z3w_write_shard(w, 0, sy, sx, d, 2)) return 1;
      }
      free(d); if (z3w_close(w)) return 1; }
    snprintf(p, sizeof p, "%s/axis.json", root); FILE *f = fopen(p, "w"); if (!f) return 1;
    fprintf(f, "{\"control_points\": [{\"x\": %g, \"y\": %g, \"z\": 0}, {\"x\": %g, \"y\": %g, \"z\": %d}]}\n", C0, C0, C0, C0, NZ - 1); fclose(f);
    /* the wraps: 64 rows (z = 2 r), columns theta = -pi + c dth, at r_k + 4 */
    tifxyz_set_sfc(0.01);
    const int H = 63, W = 400;
    for (int k = 1; k <= 6; k++) {
        float *x = malloc(W * H * 4), *y = malloc(W * H * 4), *z = malloc(W * H * 4);
        for (int r = 0; r < H; r++) for (int c = 0; c < W; c++) {
            const double th = -M_PI + (c + 0.5) * 2 * M_PI / W, zz = 1 + 2 * r, rr = sheet_r(k, th, zz) + 4;
            x[r * W + c] = (float)(C0 + rr * cos(th)); y[r * W + c] = (float)(C0 + rr * sin(th)); z[r * W + c] = (float)zz;
        }
        snprintf(p, sizeof p, "%s/in/w%03d", root, k);
        if (tifxyz_write(p, W, H, x, y, z, 2, 9.6, k, "test")) return 1;
        free(x); free(y); free(z);
    }
    tifxyz_set_sfc(0);
    /* snap with 1 and 4 threads */
    for (int t = 0; t < 2; t++) {
        char in[300], out[300], ax[300], ct[300]; snprintf(in, sizeof in, "%s/in", root); snprintf(out, sizeof out, "%s/out%d", root, t);
        snprintf(ax, sizeof ax, "%s/axis.json", root); snprintf(ct, sizeof ct, "%s/ct", root);
        char *av[] = {"ufsm", "wrap-snap", in, out, "--axis", ax, "--axis-um", "9.6", "--ct", ct, "0", "--ct-um", "9.6", "--range", "60",
                      "--sigma", "120,60,30", "--tau", "40,25,15", "--sfc", "0.01", "--band", "16", "--threads", t ? "4" : "1"};
        CHECK(!cmd_wrap_snap((int)(sizeof av / sizeof *av), av), "wrap-snap failed");
    }
    snprintf(cmd, sizeof cmd, "cd %s && for f in out0/*.sfc; do cmp -s $f out1/${f#out0/} || exit 1; done && [ $(ls out0 | wc -l) = 6 ]", root);
    CHECK(!system(cmd), "wrap-snap output differs between 1 and 4 threads (or wraps missing)");
    double sum0 = 0, sum1 = 0; long n = 0, nmask = 0, maskmoved = 0, order = 0, zmoved = 0, angmoved = 0;
    float *prevr = nullptr;
    for (int k = 1; k <= 6; k++) {
        int w, h; char d0[400], d1[400]; snprintf(d0, sizeof d0, "%s/in/w%03d.sfc", root, k); snprintf(d1, sizeof d1, "%s/out0/w%03d.sfc", root, k);
        float *x0 = rd(d0, "x", &w, &h), *y0 = rd(d0, "y", &w, &h), *z0 = rd(d0, "z", &w, &h), *x1 = rd(d1, "x", &w, &h), *y1 = rd(d1, "y", &w, &h), *z1 = rd(d1, "z", &w, &h);
        if (!x0 || !y0 || !z0 || !x1 || !y1 || !z1 || w != W || h != H) { CHECK(0, "cannot read the wraps back"); break; }
        float *rr = malloc(W * H * 4);
        for (int i = 0; i < W * H; i++) {
            const double X0 = x0[i] - C0, Y0 = y0[i] - C0, X1 = x1[i] - C0, Y1 = y1[i] - C0, th = atan2(Y0, X0), r0 = hypot(X0, Y0), r1 = hypot(X1, Y1);
            rr[i] = (float)r1;
            zmoved += fabs(z1[i] - z0[i]) > 1e-3; angmoved += fabs(remainder(atan2(Y1, X1) - th, 2 * M_PI)) * r1 > 0.05;
            if (prevr && !(rr[i] > prevr[i] + 3)) order++;
            if (th >= 1 && th <= 1.4) { if (r0 > 52) { nmask++; maskmoved += fabs(r1 - r0) > 0.05; } continue; }
            if (th > 0.8 && th < 1.6) continue;   /* (next to the wedge: clamped by the mask run) */
            sum0 += fabs(r0 - sheet_r(k, th, z0[i])); sum1 += fabs(r1 - sheet_r(k, th, z0[i])); n++;
        }
        free(prevr); prevr = rr;
        free(x0); free(y0); free(z0); free(x1); free(y1); free(z1);
    }
    free(prevr);
    printf("test_wrapsnap: mean |r - sheet| before %.3f, after %.3f voxels (%ld vertices); masked %ld (moved %ld); order violations %ld; z moved %ld, angle moved %ld\n",
           n ? sum0 / n : 0, n ? sum1 / n : 0, n, nmask, maskmoved, order, zmoved, angmoved);
    CHECK(n > 0 && sum1 / n < 0.5 && sum0 / n > 3.5, "wraps not snapped onto their sheets");
    CHECK(nmask > 0 && maskmoved == 0, "vertices on masked CT moved");
    CHECK(order == 0, "wrap order not kept");
    CHECK(zmoved == 0 && angmoved == 0, "vertices left their rays (z or angle changed)");
    if (!fails) { snprintf(cmd, sizeof cmd, "rm -rf %s", root); if (system(cmd)) {} }
    printf("test_wrapsnap: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
