/* src/geomcheck.c (ufsm geom-check): synthetic Archimedean spiral wraps (.sfc, as ufsm wraps writes them) around a straight axis.
   1. clean: no steep / jump / merge / skip, column spacing ~ r / r_k, exit 0;
   2. a patch of one wrap pushed one pitch outward (onto its outer neighbour): merges + jumps + steep walls there, the guard exits 3, the
      patch is in cells.tsv;
   3. --ct: a CT whose sheets sit on the spiral, wraps that follow it (slip ~ 0) vs wraps that slant outward by 0.1 pitch per row (slip
      ~ +0.2 pitch per 2 rows). */
#include "tiffw.h"
#include "z3w.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cmd_geom_check(int argc, char **argv);
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static const double R0 = 40, PITCH = 12, CY = 128, CX = 128;
static double rad(int k, double th) { return R0 + PITCH * (k + th / (2 * M_PI)); }
static double num(const char *b, const char *key) { const char *q = strstr(b, key); return q ? atof(q + strlen(key)) : NAN; }
static char *slurp(const char *p) { FILE *f = fopen(p, "rb"); if (!f) return nullptr; fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); char *b = calloc(n + 1, 1); if (fread(b, 1, n, f) != (size_t)n) n = 0; fclose(f); return b; }
/* wraps k0..k1, h rows; push: wrap 5 rows 16..47, columns of theta in [0, 1) moved one pitch out; slant: + slant pitches per row */
static void wraps(const char *dir, int h, int push, double slant) {
    char cmd[600]; snprintf(cmd, sizeof cmd, "rm -rf %s; mkdir -p %s", dir, dir); if (system(cmd)) exit(1);
    tifxyz_set_sfc(0.05);
    for (int k = 2; k <= 7; k++) {
        const double rk = rad(k, 0), step = 1, dth = step / rk; const int cols = (int)ceil(2 * M_PI / dth) + 1;
        float *X = malloc((size_t)cols * h * 4), *Y = malloc((size_t)cols * h * 4), *Z = malloc((size_t)cols * h * 4);
        for (int r = 0; r < h; r++) for (int c = 0; c < cols; c++) {
            const size_t i = (size_t)r * cols + (cols - 1 - c); const double th = -M_PI + c * dth, ta = atan2(sin(th), cos(th));
            double rr = rad(k, ta) + slant * PITCH * r; if (push && k == 5 && r >= 16 && r < 48 && ta >= 0 && ta < 1) rr += PITCH;
            X[i] = (float)(CX + rr * cos(th)); Y[i] = (float)(CY + rr * sin(th)); Z[i] = (float)(8 + r);
        }
        char p[600]; snprintf(p, sizeof p, "%s/w%03d", dir, k); CHECK(!tifxyz_write(p, cols, h, X, Y, Z, step, 10, k, "test"), "write %s", p);
        free(X); free(Y); free(Z);
    }
    tifxyz_set_sfc(0);
}
static int run(const char *root, const char *wd, const char *out, const char *extra) {
    char *av[40]; int ac = 0; char ax[600]; snprintf(ax, sizeof ax, "%s/axis.json", root);
    static char ex[16][64]; int ne = 0;
    av[ac++] = "ufsm"; av[ac++] = "geom-check"; av[ac++] = (char *)wd; av[ac++] = "--axis"; av[ac++] = ax; av[ac++] = "--axis-um"; av[ac++] = "10";
    av[ac++] = "--out"; av[ac++] = (char *)out; av[ac++] = "--threads"; av[ac++] = "3"; av[ac++] = "--every"; av[ac++] = "1"; av[ac++] = "--nb"; av[ac++] = "512";
    if (extra) { char t[512]; snprintf(t, sizeof t, "%s", extra); for (char *q = strtok(t, " "); q && ne < 16; q = strtok(nullptr, " ")) { snprintf(ex[ne], 64, "%s", q); av[ac++] = ex[ne++]; } }
    return cmd_geom_check(ac, av);
}

int main(void) {
    char root[256]; snprintf(root, sizeof root, "/tmp/test_geom_check_%d", (int)getpid());
    char p[600], q[600], cmd[700]; snprintf(cmd, sizeof cmd, "mkdir -p %s", root); if (system(cmd)) return 1;
    snprintf(p, sizeof p, "%s/axis.json", root); FILE *f = fopen(p, "w");
    fprintf(f, "{\"control_points\":[{\"z\":0,\"y\":%g,\"x\":%g},{\"z\":1000,\"y\":%g,\"x\":%g}]}\n", CY, CX, CY, CX); fclose(f);
    /* 1. clean */
    snprintf(p, sizeof p, "%s/clean", root); snprintf(q, sizeof q, "%s/gc_clean", root); wraps(p, 48, 0, 0);
    CHECK(run(root, p, q, "--max-merge 0.01 --max-jump 0.01 --max-steep 0.01") == 0, "clean: guard tripped");
    snprintf(p, sizeof p, "%s/summary.txt", q); char *s = slurp(p); CHECK(s, "no summary");
    if (s) { printf("%s", s);
             CHECK(num(s, "steep ") < 0.005 && num(s, "jump ") < 0.005 && num(s, "merge ") < 0.005 && num(s, "skip ") < 0.005, "clean: defects reported");
             const double c50 = num(s, "column spacing / step p05 ") ; CHECK(c50 > 0.4 && c50 < 1.2, "clean: column spacing p05 %.3f", c50); free(s); }
    /* 2. a patch pushed onto the outer neighbour */
    snprintf(p, sizeof p, "%s/push", root); snprintf(q, sizeof q, "%s/gc_push", root); wraps(p, 48, 1, 0);
    CHECK(run(root, p, q, "--max-merge 0.005") == 3, "push: the guard did not trip");
    snprintf(p, sizeof p, "%s/summary.txt", q); s = slurp(p);
    if (s) { printf("%s", s); CHECK(num(s, "merge ") > 0.005 && num(s, "jump ") > 0.0005, "push: merge %.4f jump %.4f", num(s, "merge "), num(s, "jump ")); free(s); }
    snprintf(p, sizeof p, "%s/cells.tsv", q); s = slurp(p); CHECK(s && strstr(s, "\nw005\t") && strstr(s, "\nw006\t"), "push: the patch (w005 on w006) not in cells.tsv"); free(s);
    snprintf(p, sizeof p, "%s/wraps.tsv", q); s = slurp(p); CHECK(s && strstr(s, "w005"), "no wraps.tsv"); free(s);
    /* 3. CT slip: sheets on the spiral (gaussian, sigma 1.5 voxels), 256 x 256 x 64 */
    { snprintf(p, sizeof p, "%s/ct.zarr", root); const int64_t sh[3] = {128, 256, 256}; z3w *w = z3w_create(p, sh, 128, 0, 0, nullptr); CHECK(w, "z3w");
      uint8_t *v = calloc((size_t)128 * 128 * 128, 1);
      for (int sy = 0; sy < 2; sy++) for (int sx = 0; sx < 2; sx++) {
          for (int z = 0; z < 128; z++) for (int y = 0; y < 128; y++) for (int x = 0; x < 128; x++) {
              const double yy = sy * 128 + y - CY, xx = sx * 128 + x - CX, r = hypot(yy, xx), th = atan2(yy, xx), u = (r - R0) / PITCH - th / (2 * M_PI);
              const double d = (u - round(u)) * PITCH; v[((size_t)z * 128 + y) * 128 + x] = r > R0 - 6 ? (uint8_t)(30 + 200 * exp(-d * d / (2 * 1.5 * 1.5))) : 0;
          }
          CHECK(!z3w_write_shard(w, 0, sy, sx, v, 4), "shard");
      }
      free(v); CHECK(!z3w_close(w), "close"); }
    for (int sl = 0; sl < 2; sl++) {
        snprintf(p, sizeof p, "%s/slip%d", root, sl); snprintf(q, sizeof q, "%s/gc_slip%d", root, sl); wraps(p, 40, 0, sl ? 0.1 / 2 : 0);   /* 0.05 pitch per row: 0.1 per 2 rows */
        char ex[700]; snprintf(ex, sizeof ex, "--ct %s ct.zarr --ct-um 10 --slip-every 4 --slip-d 2 --slip-sb 2 --zbins 1", root);
        run(root, p, q, ex);
        snprintf(p, sizeof p, "%s/slip.tsv", q); s = slurp(p); CHECK(s, "no slip.tsv");
        if (s) { double sum = 0; int n = 0; for (char *l = strchr(s, '\n'); l && l[1]; l = strchr(l + 1, '\n')) { char wn[16]; int zb; long long nn; double m, a;
                     if (sscanf(l + 1, "%15s %d %lld %lf %lf", wn, &zb, &nn, &m, &a) == 5) { sum += m; n++; } }
                 const double mean = n ? sum / n : NAN; printf("slip arm %d: mean slip %.3f pitch per 2 rows over %d wraps\n", sl, mean, n);
                 CHECK(n >= 3, "slip: %d wraps", n); CHECK(sl ? fabs(mean - 0.1) < 0.04 : fabs(mean) < 0.03, "slip arm %d: mean %.3f", sl, mean); free(s); }
    }
    snprintf(cmd, sizeof cmd, "rm -rf %s", root); if (!fails && system(cmd)) return 1;
    printf(fails ? "test_geom_check: %d FAILED\n" : "test_geom_check: all passed\n", fails);
    return fails != 0;
}
