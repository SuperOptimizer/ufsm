/* ufsm wrap-repair on a synthetic scroll (test_wrapsnap's: concentric wavy CT sheets r_k = 30 + 12 k + 2 sin(3 theta + z / 20) about a
   vertical axis). Wraps k = 1..6 as level-set wraps in the gaps (r_k + 6), with two planted errors:
     - dup: in theta [-1.6, -0.4] the fit over-counts one turn: wrap 4 sits 1 voxel outside wrap 3 (two wraps between sheets 3 and 4: the
       interval 3 -> 4 holds no sheet) and wraps 5, 6 one sheet further in (in the gaps of sheets 4, 5);
     - crossing: in theta [2.0, 2.6] wrap 6 sits 1 voxel inside wrap 5.
   The check (--shift 0, wrap-check's window) must see: empty intervals only on the pair 3 -> 4, crossings only on 5 -> 6, one CT peak on
   every other interval. The repair must remove wrap 4's vertices in the dup sector (and nothing of wrap 4 elsewhere beyond the sector's
   edge cells), the crossed vertices of wraps 5 / 6, and write the same bytes with 1 and 4 threads; the repaired wraps' check then has no
   dup cells and no crossings. */
#include "tiffw.h"
#include "z3w.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cmd_wrap_repair(int argc, char **argv);
static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", m); fails++; } } while (0)
#define NZ 128
#define NY 256
#define NX 256
#define C0 128.0
static double sheet_r(int k, double th, double z) { return 30 + 12 * k + 2 * sin(3 * th + z / 20); }
static float *none(const char *p, int *w, int *h) { (void)p; (void)w; (void)h; return nullptr; }
static float *rd(const char *dir, const char *ax, int *w, int *h) { char p[600]; snprintf(p, sizeof p, "%s/%s.tif", dir, ax); return tfx_axis_read(p, w, h, none); }
static int run(const char *root, const char *in, const char *out, const char *tsv, const char *cells, int th, char *log, size_t logn) {
    char ax[300], ct[300]; snprintf(ax, sizeof ax, "%s/axis.json", root); snprintf(ct, sizeof ct, "%s/ct", root);
    char *av[] = {"ufsm", "wrap-repair", (char *)in, (char *)out, "--axis", ax, "--axis-um", "9.6", "--ct", ct, "0", "--ct-um", "9.6", "--shift", "0", "--nray", "360",
                  "--row-step", "2", "--cell-rows", "16", "--cell-deg", "10", "--dup-share", "0.6", "--min-int", "20", "--sfc", "0.01", "--tsv", (char *)tsv, "--cells", (char *)cells,
                  "--threads", th == 4 ? "4" : "1"};
    fflush(stdout); char lp[400]; snprintf(lp, sizeof lp, "%s/log%d.txt", root, th); FILE *keep = fdopen(dup(1), "w"); if (!freopen(lp, "w", stdout)) return 1;
    const int rc = cmd_wrap_repair((int)(sizeof av / sizeof *av), av);
    fflush(stdout); dup2(fileno(keep), 1); fclose(keep);
    FILE *f = fopen(lp, "r"); size_t n = f ? fread(log, 1, logn - 1, f) : 0; log[n] = 0; if (f) fclose(f);
    fputs(log, stdout); return rc;
}
int main(void) {
    char root[256]; snprintf(root, sizeof root, "%s/test_wraprepair_%d", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp", (int)getpid());
    char cmd[1400], p[600]; snprintf(cmd, sizeof cmd, "rm -rf %s && mkdir -p %s/in", root, root); if (system(cmd)) return 1;
    { const int64_t sh[3] = {NZ, NY, NX}; snprintf(p, sizeof p, "%s/ct/0", root); z3w *w = z3w_create(p, sh, 128, 0, 0, nullptr); if (!w) return 1;
      uint8_t *d = malloc(128 * 128 * 128);
      for (int sy = 0; sy < 2; sy++) for (int sx = 0; sx < 2; sx++) {
          for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) for (int x = 0; x < 128; x++) {
              const double Y = sy * 128 + y - C0, X = sx * 128 + x - C0, r = sqrt(X * X + Y * Y), th = atan2(Y, X);
              double v = 20; for (int k = 0; k < 9; k++) { const double e = r - sheet_r(k, th, z); v += 200 * exp(-e * e / (2 * 1.2 * 1.2)); }
              d[((size_t)z * 128 + y) * 128 + x] = (uint8_t)lround(fmin(255, v));
          }
          if (z3w_write_shard(w, 0, sy, sx, d, 2)) return 1;
      }
      free(d); if (z3w_close(w)) return 1; }
    snprintf(p, sizeof p, "%s/axis.json", root); FILE *f = fopen(p, "w"); if (!f) return 1;
    fprintf(f, "{\"control_points\": [{\"x\": %g, \"y\": %g, \"z\": 0}, {\"x\": %g, \"y\": %g, \"z\": %d}]}\n", C0, C0, C0, C0, NZ - 1); fclose(f);
    tifxyz_set_sfc(0.01);
    const int H = 63, W = 400;
    for (int k = 1; k <= 6; k++) {
        float *x = malloc(W * H * 4), *y = malloc(W * H * 4), *z = malloc(W * H * 4);
        for (int r = 0; r < H; r++) for (int c = 0; c < W; c++) {
            const double th = -M_PI + (c + 0.5) * 2 * M_PI / W, zz = 1 + 2 * r;
            double rr = sheet_r(k, th, zz) + 6;
            if (k >= 4 && th >= -1.6 && th <= -0.4) rr = k == 4 ? sheet_r(3, th, zz) + 7 : sheet_r(k - 1, th, zz) + 6;   /* (one turn over-counted: the outer wraps one sheet in) */
            if (k == 6 && th >= 2.0 && th <= 2.6) rr = sheet_r(5, th, zz) + 5;
            x[r * W + c] = (float)(C0 + rr * cos(th)); y[r * W + c] = (float)(C0 + rr * sin(th)); z[r * W + c] = (float)zz;
        }
        snprintf(p, sizeof p, "%s/in/w%03d", root, k);
        if (tifxyz_write(p, W, H, x, y, z, 2, 9.6, k, "test")) return 1;
        free(x); free(y); free(z);
    }
    tifxyz_set_sfc(0);
    static char log[8192];
    char in[300], o0[300], o1[300], t[300], c[300];
    snprintf(in, sizeof in, "%s/in", root); snprintf(o0, sizeof o0, "%s/out0", root); snprintf(o1, sizeof o1, "%s/out1", root);
    snprintf(t, sizeof t, "%s/before.tsv", root); snprintf(c, sizeof c, "%s/cells.tsv", root);
    CHECK(!run(root, in, o0, t, c, 1, log, sizeof log), "wrap-repair (1 thread) failed");
    CHECK(!run(root, in, o1, t, c, 4, log, sizeof log), "wrap-repair (4 threads) failed");
    snprintf(cmd, sizeof cmd, "cd %s && for f in out0/*.sfc; do cmp -s $f out1/${f#out0/} || exit 1; done && [ $(ls out0/*.sfc | wc -l) = 6 ]", root);
    CHECK(!system(cmd), "repaired wraps differ between 1 and 4 threads (or wraps missing)");
    /* the cells: empty only on pair 3 (k = 3 -> 4), crossings only on pair 5 */
    { FILE *cf = fopen(c, "r"); char line[512]; long e[8] = {0}, n[8] = {0}, cr[8] = {0}, one[8] = {0};
      if (cf && fgets(line, sizeof line, cf)) while (fgets(line, sizeof line, cf)) { int k, r0, s, nn, e0, e1, e2, gn, g0, g1, g2, crr;
          if (sscanf(line, "%d %d %d %d %d %d %d %d %d %d %d %d", &k, &r0, &s, &nn, &e0, &e1, &e2, &gn, &g0, &g1, &g2, &crr) == 12 && k >= 1 && k < 8) { e[k] += e0; n[k] += nn; cr[k] += crr; one[k] += e1; } }
      if (cf) fclose(cf);
      printf("test_wraprepair: pair  intervals  empty  one  crossed\n"); for (int k = 1; k <= 5; k++) printf("  %d -> %d  %6ld %6ld %6ld %6ld\n", k, k + 1, n[k], e[k], one[k], cr[k]);
      CHECK(e[3] > 0.15 * n[3] && cr[3] == 0, "pair 3 -> 4: the dup sector not seen as empty");
      for (int k = 1; k <= 5; k++) if (k != 3) CHECK(e[k] < 0.02 * n[k], "an empty interval off the dup pair");
      for (int k = 1; k <= 5; k++) if (k != 3) CHECK(one[k] > 0.97 * n[k], "not one sheet per interval off the dup pair");
      CHECK(cr[5] > 0 && cr[1] + cr[2] + cr[3] + cr[4] == 0, "crossings not (only) on pair 5 -> 6"); }
    /* the repaired wraps: wrap 4 lost the dup sector (inner part of it at least), wraps 5 / 6 the crossing sector, nothing else */
    long lost[7] = {0}, lostin[7] = {0};
    for (int k = 1; k <= 6; k++) {
        int w, h; char d0[400], d1[400]; snprintf(d0, sizeof d0, "%s/in/w%03d.sfc", root, k); snprintf(d1, sizeof d1, "%s/out0/w%03d.sfc", root, k);
        float *x0 = rd(d0, "x", &w, &h), *y0 = rd(d0, "y", &w, &h), *z1 = rd(d1, "z", &w, &h);
        if (!x0 || !y0 || !z1) { CHECK(0, "cannot read the wraps back"); break; }
        for (int i = 0; i < w * h; i++) { const double th = atan2(y0[i] - C0, x0[i] - C0); if (z1[i] >= 0) continue;
            const int ins = (k == 4 && th >= -1.6 && th <= -0.4) || ((k == 5 || k == 6) && th >= 2.0 && th <= 2.6); const int core = (k == 4 && th >= -1.4 && th <= -0.6) || ((k == 5 || k == 6) && th >= 2.05 && th <= 2.55);
            lost[k]++; lostin[k] += ins; (void)core; }
        free(x0); free(y0); free(z1);
    }
    printf("test_wraprepair: removed vertices per wrap (in the planted sector):"); for (int k = 1; k <= 6; k++) printf(" w%d %ld (%ld)", k, lost[k], lostin[k]); printf("\n");
    const long sec4 = (long)lround(1.2 / (2 * M_PI) * W) * H, sec56 = (long)lround(0.6 / (2 * M_PI) * W) * H;
    CHECK(lostin[4] > 0.6 * sec4 && lost[4] - lostin[4] < 0.25 * sec4, "wrap 4: the dup sector not removed (or too much outside it)");
    CHECK(lostin[6] > 0.8 * sec56 && lost[6] == lostin[6] && lost[5] == lostin[5], "wraps 5 / 6: crossed vertices not removed (or removed elsewhere)");
    CHECK(lost[1] + lost[2] + lost[3] == 0, "vertices of the clean wraps 1..3 removed");
    /* the repaired set re-checked: no dup cells, no crossings */
    char o2[300], t2[300], c2[300]; snprintf(t2, sizeof t2, "%s/after.tsv", root); snprintf(c2, sizeof c2, "%s/cells2.tsv", root); snprintf(o2, sizeof o2, "-");
    CHECK(!run(root, o0, o2, t2, c2, 4, log, sizeof log), "wrap-repair check of the repaired wraps failed");
    CHECK(strstr(log, "crossed 0 (") != nullptr, "the repaired wraps still cross");
    CHECK(strstr(log, "dup cells 0 ") != nullptr, "the repaired wraps still have dup cells");
    if (!fails) { snprintf(cmd, sizeof cmd, "rm -rf %s", root); if (system(cmd)) {} }
    printf("test_wraprepair: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
