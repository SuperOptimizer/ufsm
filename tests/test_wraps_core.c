/* ufsm wraps on the core of a scroll: a synthetic fit model whose W at the axis rises from 0 at the box's z ends to 3 at its middle
   (the core turns leave the axis at different heights, as on PHerc Paris 4: W at the umbilicus 78..93 over z). Wrap 3 then has no
   (or a tiny) root on the box centre's ray at the middle z, so the default column spacing (step / that radius) is ~1 rad and the wrap's
   grid is a few columns wide; --col-r p90 spaces the columns by the wrap's 90th percentile radius instead (the arc between columns
   at the wrap's own radius ~ step). --zr Z0,Z1 keeps only the rows at global z in [Z0, Z1]; --first / --last are the wrap ids as
   given. Runs the ufsm binary next to this test. */
#include "fit.h"
#include "tiff.h"
#include <libgen.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

enum { D = 64, H = 96, Wd = 96, HH = 2 };
static const double P = 10, CY = 96, CX = 96;
static const long long ORG[3] = {1000, 50, 70};
static double delta_of(double z) { return 3 - 3 * fabs(z - 64) / 64; }   /* W at the axis: 0 at z 0 / 128, 3 at z 64 */

static float *tif(const char *p, int *w, int *h) {
    tiff *t = tiff_open_file(p); tiff_page pg; if (!t || tiff_page_info(t, 0, &pg)) { if (t) tiff_close(t); return nullptr; }
    float *d = malloc((size_t)pg.w * pg.h * 4); tiff_read_page(t, 0, d); *w = pg.w; *h = pg.h; tiff_close(t); return d;
}
typedef struct { int w, h; size_t nv; double zmin, zmax, sp_med, werr; } wstat;
/* a written wrap: grid size, valid vertices, their global z range, the median distance between neighbouring valid columns, and the
   mean |W - k| at the vertices */
static int wrap_stat(const char *dir, int k, const fit_set *fs, wstat *s) {
    char p[700]; int w, h, w2, h2, w3, h3; memset(s, 0, sizeof *s);
    snprintf(p, sizeof p, "%s/w%03d/x.tif", dir, k); float *X = tif(p, &w, &h); if (!X) return -1;
    snprintf(p, sizeof p, "%s/w%03d/y.tif", dir, k); float *Y = tif(p, &w2, &h2);
    snprintf(p, sizeof p, "%s/w%03d/z.tif", dir, k); float *Z = tif(p, &w3, &h3);
    if (!Y || !Z) { free(X); free(Y); free(Z); return -1; }
    s->w = w; s->h = h; s->zmin = 1e30; s->zmax = -1e30;
    double *sp = malloc(sizeof(double) * (size_t)w * h); size_t ns = 0; double we = 0;
    for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) {
        const size_t i = (size_t)r * w + c; if (Z[i] < 0) continue;
        s->nv++; if (Z[i] < s->zmin) s->zmin = Z[i]; if (Z[i] > s->zmax) s->zmax = Z[i];
        we += fabs(fit_set_W(fs, Z[i] - ORG[0], Y[i] - ORG[1], X[i] - ORG[2]) - k);
        if (c + 1 < w && Z[i + 1] >= 0) sp[ns++] = hypot(X[i + 1] - X[i], Y[i + 1] - Y[i]);
    }
    for (size_t a = 1; a < ns; a++) { const double v = sp[a]; size_t b = a; while (b > 0 && sp[b - 1] > v) { sp[b] = sp[b - 1]; b--; } sp[b] = v; }
    s->sp_med = ns ? sp[ns / 2] : 0; s->werr = s->nv ? we / s->nv : 1;
    free(sp); free(X); free(Y); free(Z);
    return 0;
}
static int run(const char *cmd) { fprintf(stderr, "+ %s\n", cmd); return system(cmd); }

int main(int argc, char **argv) {
    (void)argc;
    char exe[600]; snprintf(exe, sizeof exe, "%s", argv[0]); char bin[700]; snprintf(bin, sizeof bin, "%s/ufsm", dirname(exe));
    if (access(bin, X_OK)) { fprintf(stderr, "FAIL: no %s\n", bin); return 1; }
    char base[] = "/tmp/test_wraps_core_XXXXXX"; if (!mkdtemp(base)) return 1;
    char md[600]; snprintf(md, sizeof md, "%s/model", base);
    fit_model m = {0}; m.dir = 1; m.nk = 2; m.knot_dz = 1024; float pit[2] = {(float)P, (float)P}, off[2] = {0, 0}; m.pitch = pit; m.offset = off;
    m.L = 8; m.lz = D * HH / 8 + 2; m.ly = H * HH / 8 + 2; m.lx = Wd * HH / 8 + 2; m.delta = calloc((size_t)m.lz * m.ly * m.lx, 4);
    for (int a = 0; a < m.lz; a++) for (int b = 0; b < m.ly; b++) for (int c = 0; c < m.lx; c++) m.delta[((size_t)a * m.ly + b) * m.lx + c] = (float)delta_of(a * 8.0);
    m.D = D; m.H = H; m.W = Wd; m.h = HH; for (int d = 0; d < 3; d++) m.origin[d] = ORG[d]; m.um = 9.6;
    const int na = D * HH + 1; float *cy = malloc(na * 4), *cx = malloc(na * 4); for (int z = 0; z < na; z++) { cy[z] = (float)CY; cx[z] = (float)CX; }
    m.cy = cy; m.cx = cx;
    CHECK(!fit_save(&m, md), "fit_save");
    fit_set *fs = fit_set_load(md); CHECK(fs != nullptr, "fit_set_load"); if (!fs) return 1;

    char cmd[2000], o1[600], o2[600], o3[600];
    snprintf(o1, sizeof o1, "%s/centre", base); snprintf(o2, sizeof o2, "%s/p90", base); snprintf(o3, sizeof o3, "%s/zr", base);
    snprintf(cmd, sizeof cmd, "%s wraps %s %s --step 2 --first 3 --last 3 --col-r centre --min-cm2 0 --threads 4 > /dev/null 2>&1", bin, md, o1); CHECK(run(cmd) == 0, "wraps --col-r centre");
    snprintf(cmd, sizeof cmd, "%s wraps %s %s --step 2 --first 3 --last 3 --min-cm2 0 --threads 4 > /dev/null 2>&1", bin, md, o2); CHECK(run(cmd) == 0, "wraps --col-r p90");
    snprintf(cmd, sizeof cmd, "%s wraps %s %s --step 2 --first 3 --last 3 --min-cm2 0 --zr %lld,%lld --threads 4 > /dev/null 2>&1", bin, md, o3, ORG[0] + 16, ORG[0] + 48);
    CHECK(run(cmd) == 0, "wraps --zr");
    wstat a, b, c;
    CHECK(!wrap_stat(o1, 3, fs, &a), "read the default wrap 3");
    CHECK(!wrap_stat(o2, 3, fs, &b), "read the p90 wrap 3");
    CHECK(!wrap_stat(o3, 3, fs, &c), "read the --zr wrap 3");
    char p[700]; snprintf(p, sizeof p, "%s/w002", o2); CHECK(access(p, F_OK) != 0, "--first 3: w002 written");
    snprintf(p, sizeof p, "%s/w004", o2); CHECK(access(p, F_OK) != 0, "--last 3: w004 written");
    /* wrap 3 reaches radius 10 (3 + theta / 2 pi) ~ 25..35 at the z ends: p90 ~ 33 -> ~2 pi 33 / 2 ~ 104 columns */
    printf("default: %d x %d grid, %zu vertices, column spacing median %.2f (step 2), mean |W - 3| %.4f\n", a.w, a.h, a.nv, a.sp_med, a.werr);
    printf("--col-r p90: %d x %d grid, %zu vertices, column spacing median %.2f, mean |W - 3| %.4f\n", b.w, b.h, b.nv, b.sp_med, b.werr);
    printf("--zr +16..+48: %d x %d grid, %zu vertices, z %.1f..%.1f (box-local)\n", c.w, c.h, c.nv, c.zmin - ORG[0], c.zmax - ORG[0]);
    CHECK(a.w < 20, "default columns %d: the core wrap should be under-sampled (the test lost its point)", a.w);
    CHECK(b.w >= 80 && b.w <= 140, "--col-r p90 columns %d (80..140)", b.w);
    CHECK(b.sp_med > 0.8 && b.sp_med < 2.6, "--col-r p90: median column spacing %.2f (step 2)", b.sp_med);
    CHECK(a.sp_med > 3 * b.sp_med, "default spacing %.2f vs p90 %.2f", a.sp_med, b.sp_med);
    CHECK(b.nv > 3 * a.nv, "p90 vertices %zu vs default %zu", b.nv, a.nv);
    CHECK(b.werr < 0.02, "--col-r p90 vertices off the level set: mean |W - 3| %.4f", b.werr);
    CHECK(c.h == 17, "--zr rows %d (17)", c.h);
    CHECK(c.zmin >= ORG[0] + 16 - 1e-3 && c.zmax <= ORG[0] + 48 + 1e-3, "--zr z %.1f..%.1f outside [16, 48]", c.zmin - ORG[0], c.zmax - ORG[0]);
    CHECK(c.w == b.w || abs(c.w - b.w) < b.w / 3, "--zr columns %d vs %d", c.w, b.w);
    fit_set_free(fs); free(m.delta); free(cy); free(cx);
    snprintf(cmd, sizeof cmd, "rm -rf '%s'", base); if (system(cmd)) {}
    if (fails) { fprintf(stderr, "test_wraps_core: %d failures\n", fails); return 1; }
    printf("test_wraps_core: OK\n");
    return 0;
}
