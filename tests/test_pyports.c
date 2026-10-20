/* The C ports of the last pipeline Python steps against the Python's own numbers (test_pyports_expected.h, from the former
   p1299 stitch/phase_score.py and core/check/check.py run on the inputs this test writes; PYPORTS_DUMP=DIR writes them there):
   - ufsm phase-score (imgops.c phase_score_dirs): the report text, byte for byte: the mask (rendered, |(c, s)| > 13), the
     < 1000 pixel skip, a pc render without its ps twin, the render-name pattern w<sheet>_rR_cC and the seam classes sampled
     every second grid step (a seam narrower than the render), the totals, median / p10 / p90;
   - numpy.percentile (linear) in float64 and float32, scipy.ndimage.gaussian_filter (float32, reflect) bit for bit;
   - ufsm coreline check (coreline_check.c cl_check): the slab, crop and the away / near-line counts of check.py. */
#include "coreline.h"
#include "imgops.h"
#include "tiffw.h"
#include "test_pyports_expected.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

static unsigned lcg_s;
static unsigned lcg(void) { lcg_s = (lcg_s * 1103515245u + 12345u) & 0x7fffffffu; return lcg_s >> 8; }

static void put_tif(const char *dir, const char *name, int w, int h, const uint8_t *d) {
    char p[1024]; snprintf(p, sizeof p, "%s/%s", dir, name); CHECK(!tiffw_write(p, w, h, 1, d, 1), "write %s", p);
}
/* a pc / ps pair: phase drifting along the columns, confidence from the lcg, some unrendered (0) and low-confidence pixels */
static void surface(const char *root, const char *name, int w, int h, int with_ps, double drift) {
    uint8_t *pc = malloc((size_t)w * h), *ps = malloc((size_t)w * h);
    for (int r = 0; r < h; r++)
        for (int c = 0; c < w; c++) {
            const size_t i = (size_t)r * w + c; const double ph = drift * c + 0.03 * r, a = (lcg() % 128) + (c % 7 == 0 ? 0 : 10);
            pc[i] = (uint8_t)lround(128 + a * cos(ph) * 0.99); ps[i] = (uint8_t)lround(128 + a * sin(ph) * 0.99);
            if (lcg() % 17 == 0) pc[i] = 0;
            if (pc[i] == 0 && lcg() % 2) ps[i] = 128;
        }
    char d[1024]; snprintf(d, sizeof d, "%s/pc", root); put_tif(d, name, w, h, pc);
    if (with_ps) { snprintf(d, sizeof d, "%s/ps", root); put_tif(d, name, w, h, ps); }
    free(pc); free(ps);
}
static void test_phase_score(const char *root) {
    char d[1024];
    const char *sub[] = {"pc", "ps", "sheets", "sheets/sA", "sheets/sB_r1"};
    for (size_t k = 0; k < sizeof sub / sizeof *sub; k++) { snprintf(d, sizeof d, "%s/%s", root, sub[k]); mkdir(d, 0755); }
    lcg_s = 7;
    surface(root, "wsA_r00000_c00000.tif", 50, 40, 1, 0.05);
    surface(root, "wsA_r00064_c00010.tif", 45, 41, 1, -0.02);   /* the seam's rows run out (r0 + 2 r >= 140 from r 38) */
    surface(root, "wsB_r1_r00002_c00004.tif", 44, 39, 1, 0.11);   /* the greedy sheet name sB_r1 */
    surface(root, "wbox_w001.tif", 40, 40, 1, 0.3);
    surface(root, "wbox_w002.tif", 41, 43, 1, -0.07);
    surface(root, "wtiny.tif", 10, 10, 1, 0.0);                 /* < 1000 pixels */
    surface(root, "wpconly.tif", 40, 40, 0, 0.0);              /* no ps render */
    uint8_t *s = malloc(140 * 70);
    for (int r = 0; r < 140; r++) for (int c = 0; c < 70; c++) s[r * 70 + c] = (uint8_t)((r / 40 + c / 25) % 8);   /* classes 0..7 (0 / 7: none) */
    snprintf(d, sizeof d, "%s/sheets/sA", root); put_tif(d, "seam.tif", 70, 140, s);
    for (int i = 0; i < 140 * 70; i++) s[i] = (uint8_t)(1 + (i / 70) % 3);
    snprintf(d, sizeof d, "%s/sheets/sB_r1", root); put_tif(d, "seam.tif", 70, 140, s);
    free(s);
    char pcd[1024], psd[1024], sd[1024]; snprintf(pcd, sizeof pcd, "%s/pc", root); snprintf(psd, sizeof psd, "%s/ps", root); snprintf(sd, sizeof sd, "%s/sheets", root);
    for (int with_sheets = 0; with_sheets < 2; with_sheets++) {
        char *buf = nullptr; size_t n = 0; FILE *f = open_memstream(&buf, &n);
        CHECK(!phase_score_dirs(pcd, psd, with_sheets ? sd : nullptr, f), "phase_score_dirs"); fclose(f);
        const char *want = with_sheets ? exp_score_sheets : exp_score_plain;
        CHECK(buf && !strcmp(buf, want), "phase-score report (%s) differs from phase_score.py:\n--- C\n%s--- Python\n%s", with_sheets ? "sheets" : "plain", buf ? buf : "", want);
        free(buf);
    }
}

static void test_numerics(void) {
    double v[11]; float u[11];
    for (int i = 0; i < 11; i++) { v[i] = exp_pct_in[i]; u[i] = (float)exp_pct_in[i]; }
    const double q[4] = {10, 50, 90, 99.5};
    for (int k = 0; k < 4; k++) {
        double t[11]; float tf[11]; memcpy(t, v, sizeof t); memcpy(tf, u, sizeof tf);
        const double a = np_percentile_f64(t, 11, q[k]); const float b = np_percentile_f32(tf, 11, (float)q[k]);
        CHECK(a == exp_pct64[k], "percentile64 %g: %.17g, numpy %.17g", q[k], a, exp_pct64[k]);
        CHECK(b == exp_pct32[k], "percentile32 %g: %.9g, numpy %.9g", q[k], b, exp_pct32[k]);
    }
    float g[6 * 9]; lcg_s = 3; for (int i = 0; i < 6 * 9; i++) g[i] = (float)(lcg() % 1000) / 37.f;
    scipy_gaussian_f32(g, 6, 9, 1.5);
    int bad = 0; for (int i = 0; i < 6 * 9; i++) bad += g[i] != exp_gauss[i];
    CHECK(!bad, "gaussian_filter: %d of 54 values differ from scipy", bad);
}

/* a synthetic slab: W grows away from a tilted core line inside an elliptical material patch, with noise */
static void test_check(const char *root) {
    enum { N = 96, CT = 2 * N - 3 };
    char p[1024]; snprintf(p, sizeof p, "%s/core-line.json", root);
    FILE *f = fopen(p, "w");
    fprintf(f, "{\"um\": 2.399, \"slabs\": [\n");
    for (int k = 0; k < 3; k++) {
        const double zc = 64.0 * k + 32;   /* zc / 8 = 4, 12, 20 */
        fprintf(f, "%s{\"z0\": %.1f, \"z1\": %.1f, \"z\": %.1f, \"pts\": [", k ? ",\n" : "", zc - 32, zc + 32, zc);
        for (int i = 0; i < 9; i++) fprintf(f, "%s[%.1f, %.1f]", i ? ", " : "", 8 * (27 + 3 * k + 2.5 * i + (i % 3 == 1 ? 1.5 : 0)), 8 * (10.0 + 9.5 * i));
        fprintf(f, "]}");
    }
    fprintf(f, "]}\n"); fclose(f);
    coreline C; CHECK(!cl_load(&C, p), "cl_load");
    const int zc = 8;   /* nearest slab: z / 8 = 12 (index 1) */
    uint8_t *pc = malloc(8 * N * N), *ps = malloc(8 * N * N), *w = malloc(8 * N * N), *ct = malloc(CT * CT);
    lcg_s = 11;
    for (int k = 0; k < 8; k++)
        for (int y = 0; y < N; y++)
            for (int x = 0; x < N; x++) {
                const size_t i = (size_t)k * N * N + y * N + x;
                const double ly = 30 + 0.263 * (x - 10), d = fabs(y - ly) * (x > 60 ? -1 : 1), ph = 2 * M_PI * d / 5.0 + 0.1 * k + (lcg() % 100) * 0.004;
                const double e = ((y - 48) / 22.0) * ((y - 48) / 22.0) + ((x - 46) / 22.0) * ((x - 46) / 22.0);
                pc[i] = (uint8_t)lround(128 + 120 * cos(ph)); ps[i] = (uint8_t)lround(128 + 120 * sin(ph));
                w[i] = e < 1 ? (uint8_t)(200 + lcg() % 56) : (uint8_t)(lcg() % 20);
                if (lcg() % 3 == 0) { pc[i] = (uint8_t)(lcg() % 256); ps[i] = (uint8_t)(lcg() % 256); }
            }
    for (int i = 0; i < CT * CT; i++) ct[i] = (uint8_t)((i % CT) * 2 % 251 + (lcg() % 3 ? 0 : lcg() % 40));
    for (int y = 0; y < 20; y++) for (int x = 0; x < CT; x++) ct[y * CT + x] = 0;
    if (getenv("PYPORTS_DUMP")) {   /* check.py's inputs (check.sh's file names) */
        char q[1100]; snprintf(q, sizeof q, "%s/ct_%d.pgm", root, zc); FILE *g = fopen(q, "wb"); fprintf(g, "P5\n%d %d\n255\n", CT, CT); fwrite(ct, 1, CT * CT, g); fclose(g);
        const char *nm[3] = {"pc", "ps", "w"}; const uint8_t *a[3] = {pc, ps, w};
        for (int k = 0; k < 3; k++) { snprintf(q, sizeof q, "%s/%s_%d.raw", root, nm[k], zc); g = fopen(q, "wb"); fwrite(a[k], 1, 8 * N * N, g); fclose(g); }
    }
    cl_check_res o; const int rc = cl_check(&C, zc, ct, CT, CT, pc, ps, w, N, 3, &o);
    CHECK(!rc, "cl_check %d", rc);
    if (!rc) {
        const long long got[9] = {o.ks, o.y0, o.y1, o.x0, o.x1, o.pos, o.neg, o.pn, o.nn};
        int same = 1; for (int k = 0; k < 9; k++) same &= got[k] == exp_check[k];
        CHECK(same, "check: slab %lld crop %lld %lld %lld %lld pos %lld neg %lld near %lld / %lld; check.py: %lld, %lld %lld %lld %lld, %lld %lld, %lld / %lld",
              got[0], got[1], got[2], got[3], got[4], got[5], got[6], got[7], got[8], exp_check[0], exp_check[1], exp_check[2], exp_check[3], exp_check[4], exp_check[5], exp_check[6], exp_check[7], exp_check[8]);
        const int cw = o.x1 - o.x0, ch = o.y1 - o.y0, tw = (CT / 2 < o.x1 ? CT / 2 : o.x1) - o.x0;
        CHECK(o.w == 2 * tw + 8 && o.h == (CT / 2 < o.y1 ? CT / 2 : o.y1) - o.y0 && cw > 0 && ch > 0, "check image %d x %d", o.w, o.h);
        snprintf(p, sizeof p, "%s/z%04d.jpg", root, zc); CHECK(!jpeg_write_rgb(p, o.w, o.h, o.rgb, 85), "jpeg");
        free(o.rgb);
    }
    free(pc); free(ps); free(w); free(ct); cl_free(&C);
}

int main(void) {
    const char *dump = getenv("PYPORTS_DUMP"); char root[256];
    if (dump) { snprintf(root, sizeof root, "%s", dump); mkdir(root, 0755); }
    else { snprintf(root, sizeof root, "/tmp/test_pyports_XXXXXX"); if (!mkdtemp(root)) { fprintf(stderr, "FAIL: mkdtemp\n"); return 1; } }
    test_phase_score(root);
    test_numerics();
    test_check(root);
    if (!dump) { char cmd[300]; snprintf(cmd, sizeof cmd, "rm -rf %s", root); if (system(cmd)) {} }
    if (fails) { fprintf(stderr, "test_pyports: %d failure(s)\n", fails); return 1; }
    printf("test_pyports: OK\n");
    return 0;
}
