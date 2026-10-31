/* src/strip.c: a synthetic Archimedean spiral sheet (r = r0 + pitch U, U = k + theta / 2 pi, straight in z: developable, its exact
   development a rectangle of the spiral's arc length x the height) written as ufsm wraps would write it (theta from -pi, dth = step / r_k,
   column 0 = the largest theta, past the cut). ufsm strip must walk it outermost first, drop column 0, join every seam within ~1 column
   pitch; ufsm strip-flatten must give the analytic arc length and height, isometric (aniso ~1, no flips), and stripes whose valid area is
   the sheet's area. */
#include "tiffw.h"
#include "imgops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cmd_strip(int argc, char **argv);
int cmd_strip_flatten(int argc, char **argv);
int cmd_strip_mosaic(int argc, char **argv);

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static const double R0 = 150, PITCH = 7, CY = 400, CX = 420;
static double rad(int k, double th) { return R0 + PITCH * (k + th / (2 * M_PI)); }   /* dir = +1: W = U - theta / 2 pi = k */
static double num(const char *b, const char *key) { const char *q = strstr(b, key); return q ? atof(q + strlen(key)) : NAN; }

int main(void) {
    char root[256]; snprintf(root, sizeof root, "/tmp/test_strip_%d", (int)getpid());
    char p[512], cmd[600]; snprintf(cmd, sizeof cmd, "mkdir -p %s/wraps", root); if (system(cmd)) return 1;
    const int h = 48, k0 = 3, k1 = 6; const double step = 1.0, um = 10;
    long long total = 0;
    for (int k = k0; k <= k1; k++) {
        const double rk = rad(k, atan2(0, 1)), dth = step / rk; const int cols = (int)ceil(2 * M_PI / dth) + 1;
        float *X = malloc((size_t)cols * h * 4), *Y = malloc((size_t)cols * h * 4), *Z = malloc((size_t)cols * h * 4);
        for (int r = 0; r < h; r++) for (int c = 0; c < cols; c++) {
            const size_t i = (size_t)r * cols + (cols - 1 - c); const double th = -M_PI + c * dth, ta = atan2(sin(th), cos(th));   /* the ray's own angle */
            const double rr = rad(k, ta); X[i] = (float)(CX + rr * cos(th)); Y[i] = (float)(CY + rr * sin(th)); Z[i] = (float)(10 + r * step);
        }
        snprintf(p, sizeof p, "%s/wraps/w%03d", root, k);
        CHECK(!tifxyz_write(p, cols, h, X, Y, Z, step, um, k, "test"), "write %s", p);
        total += cols - 1; free(X); free(Y); free(Z);
    }
    { char a1[300], a2[300]; snprintf(a1, sizeof a1, "%s/wraps", root); snprintf(a2, sizeof a2, "%s/strip", root);
      char *av[] = {"ufsm", "strip", a1, a2, "--threads", "4", "--err", "0.05"}; CHECK(!cmd_strip(8, av), "ufsm strip failed"); }
    snprintf(p, sizeof p, "%s/strip/strip.json", root); FILE *f = fopen(p, "r"); char b[4096] = {0};
    if (f) { size_t nr = fread(b, 1, sizeof b - 1, f); (void)nr; fclose(f); }
    CHECK(strstr(b, "\"walk\": \"descending\""), "walk not outermost first: %s", b);
    /* columns: every wrap minus column 0 (minus one more where the seam came closer than half a pitch) */
    snprintf(p, sizeof p, "%s/strip/strip.tsv", root); f = fopen(p, "r"); long long ncol = 0; double gmax = 0; int nseg = 0, first = -1;
    if (f) { char line[512]; if (fgets(line, sizeof line, f)) while (fgets(line, sizeof line, f)) { int q, wr, n, wc; long long c0; char nm[64]; double g;
                 if (sscanf(line, "%d %d %63s %lld %d %d %lf", &q, &wr, nm, &c0, &n, &wc, &g) >= 6) { if (first < 0) first = wr; ncol += n; nseg++; if (isfinite(g) && g > gmax) gmax = g; } } fclose(f); }
    CHECK(nseg == k1 - k0 + 1 && first == k1, "segments %d, first wrap %d", nseg, first);
    CHECK(ncol >= total - nseg && ncol <= total, "strip columns %lld vs %lld", ncol, total);
    CHECK(gmax > 0.3 && gmax < 1.6, "largest seam gap %.3f column pitches", gmax);
    /* flatten: the exact development */
    { char a1[300]; snprintf(a1, sizeof a1, "%s/strip", root);
      char *av[] = {"ufsm", "strip-flatten", a1, "--stride", "96", "--threads", "4", "--stripe-px", "1500", "--iters", "60", "--cg", "300", "--tol", "1e-9"};
      CHECK(!cmd_strip_flatten(16, av), "ufsm strip-flatten failed"); }
    snprintf(p, sizeof p, "%s/strip/flat/flat.json", root); f = fopen(p, "r"); memset(b, 0, sizeof b);
    if (f) { size_t nr = fread(b, 1, sizeof b - 1, f); (void)nr; fclose(f); }
    /* the analytic arc length from wrap k1's theta = pi to wrap k0's theta = -pi (the spiral's U from k1 + 1/2 down to k0 - 1/2) */
    double L = 0; const int NS = 200000; const double ua = k0 - 0.5, ub = k1 + 0.5;
    for (int i = 0; i < NS; i++) { const double u = ua + (ub - ua) * (i + 0.5) / NS, r = R0 + PITCH * u; L += sqrt(r * r + (PITCH / (2 * M_PI)) * (PITCH / (2 * M_PI))) * 2 * M_PI * (ub - ua) / NS; }
    const double Lcm = L * um * 1e-4, Hcm = (h - 1) * step * um * 1e-4;
    const double len = num(b, "\"length_cm\": "), hei = num(b, "\"height_cm\": "), a50 = num(b, "\"aniso_p50\": "), a95 = num(b, "\"aniso_p95\": "), fl = num(b, "\"flips\": "), ns = num(b, "\"stripes\": ");
    printf("strip: length %.4f cm (analytic %.4f), height %.4f (%.4f), aniso p50 %.4f p95 %.4f, flips %.0f, stripes %.0f\n", len, Lcm, hei, Hcm, a50, a95, fl, ns);
    CHECK(fabs(len - Lcm) < 0.01 * Lcm, "length %.4f vs %.4f cm", len, Lcm);
    CHECK(fabs(hei - Hcm) < 0.02 * Hcm, "height %.4f vs %.4f cm", hei, Hcm);
    CHECK(a50 < 1.01 && a95 < 1.03, "aniso p50 %.4f p95 %.4f", a50, a95);
    CHECK(fl == 0, "flips %.0f", fl);
    CHECK(ns >= 2, "stripes %.0f (expected several at --stripe-px 1500)", ns);
    /* the stripes' valid area = the sheet's area (stripes share one boundary column) */
    snprintf(p, sizeof p, "%s/strip/flat/stripes.tsv", root); f = fopen(p, "r"); double area = 0;
    if (f) { char line[1024]; if (fgets(line, sizeof line, f)) while (fgets(line, sizeof line, f)) { char nm[64]; int a, c, d; double u0, l, v; if (sscanf(line, "%63s %d %d %d %lf %lf %lf", nm, &a, &c, &d, &u0, &l, &v) == 7) area += v; } fclose(f); }
    { const double A = Lcm * (Hcm + step * um * 1e-4); CHECK(fabs(area - A) < 0.03 * A, "stripes valid area %.4f vs %.4f cm2", area, A); }
    /* --jobs / --stripe-jobs: chunks and stripes on concurrent workers, in order: --jobs 3 (1 thread each) writes the bytes of --jobs 1 --threads 1
       (CPU: the SLIM sums depend on the threads per chunk; GPU: on nothing); chunks.tsv equal but for its seconds column */
    {
        const char *jv[2] = {"1", "3"};
        for (int k = 0; k < 2; k++) {
            char a1[300], a2[300]; snprintf(a1, sizeof a1, "%s/strip", root); snprintf(a2, sizeof a2, "%s/fj%s", root, jv[k]);
            char *av[] = {"ufsm", "strip-flatten", a1, "--out", a2, "--stride", "96", "--threads", (char *)jv[k], "--jobs", (char *)jv[k], "--stripe-jobs", (char *)jv[k], "--stripe-px", "1500", "--iters", "60", "--cg", "300", "--tol", "1e-9"};
            CHECK(!cmd_strip_flatten(22, av), "strip-flatten --jobs %s failed", jv[k]);
        }
        char cmd[2048];
        snprintf(cmd, sizeof cmd, "cd %s/fj1 && test -e flat.json && for x in $(find . -type f ! -name chunks.tsv); do cmp -s $x ../fj3/$x || { echo $x differs; exit 1; }; done && "
                 "cut -f1-10 chunks.tsv > ../c1 && cut -f1-10 ../fj3/chunks.tsv > ../c3 && cmp -s ../c1 ../c3", root);
        CHECK(!system(cmd), "strip-flatten --jobs 3 output differs from --jobs 1");
#ifdef STRIP_GPU
        {   /* GPU vs CPU: the same flattening within the solvers' summation-order noise */
            char a1[300], a2[300]; snprintf(a1, sizeof a1, "%s/strip", root); snprintf(a2, sizeof a2, "%s/fcpu", root);
            char *av[] = {"ufsm", "strip-flatten", a1, "--out", a2, "--stride", "96", "--threads", "4", "--gpu", "0", "--stripe-px", "1500", "--iters", "60", "--cg", "300", "--tol", "1e-9"};
            CHECK(!cmd_strip_flatten(20, av), "strip-flatten --gpu 0 failed");
            char q[400], c[4096] = {0}; snprintf(q, sizeof q, "%s/fcpu/flat.json", root); FILE *g = fopen(q, "r");
            if (g) { size_t nr = fread(c, 1, sizeof c - 1, g); (void)nr; fclose(g); }
            const double l2 = num(c, "\"length_cm\": "), b50 = num(c, "\"aniso_p50\": "), b95 = num(c, "\"aniso_p95\": ");
            printf("strip GPU vs CPU: length %.5f / %.5f cm, aniso p50 %.5f / %.5f, p95 %.5f / %.5f\n", len, l2, a50, b50, a95, b95);
            CHECK(fabs(len - l2) < 1e-4 * l2 && fabs(a50 - b50) < 1e-3 && fabs(a95 - b95) < 1e-3, "GPU vs CPU flattening differs");
        }
#endif
    }
    /* strip-mosaic on synthetic renders of the stripes: --threads 1 and --threads 4 must write the same bytes */
    {
        char rdir[400]; snprintf(rdir, sizeof rdir, "%s/render", root); snprintf(cmd, sizeof cmd, "mkdir -p %s", rdir); if (system(cmd)) {}
        snprintf(p, sizeof p, "%s/strip/flat/stripes.tsv", root); f = fopen(p, "r"); int nst = 0;
        if (f) { char line[1024]; if (fgets(line, sizeof line, f)) while (fgets(line, sizeof line, f)) { char nm[64]; int a, c, d;
            if (sscanf(line, "%63s %d %d %d", nm, &a, &c, &d) != 4) continue;
            const int ow = render_out_px(c, 1.0), oh = render_out_px(d, 1.0); uint8_t *ct = malloc((size_t)ow * oh), *ik = malloc((size_t)ow * oh);
            for (int i = 0; i < ow * oh; i++) { const unsigned hsh = (unsigned)(i + a) * 2654435761u; ct[i] = (uint8_t)(hsh % 23 == 0 ? 0 : 60 + (hsh >> 24) % 120);   /* (some CT 0: off the papyrus) */ ik[i] = (uint8_t)((hsh >> 16) & 255); }
            snprintf(p, sizeof p, "%s/%s_ct.tif", rdir, nm); CHECK(!tiffw_write(p, ow, oh, 1, ct, 1), "write %s", p);
            snprintf(p, sizeof p, "%s/%s.tif", rdir, nm); CHECK(!tiffw_write(p, ow, oh, 1, ik, 1), "write %s", p);
            free(ct); free(ik); nst++; } fclose(f); }
        CHECK(nst >= 2, "stripes %d", nst);
        static const char *outs[2] = {"m1", "m4"}, *thr[2] = {"1", "4"};
        for (int k = 0; k < 2; k++) {
            char a1[400], a3[400]; snprintf(a1, sizeof a1, "%s/strip/flat", root); snprintf(a3, sizeof a3, "%s/%s", root, outs[k]);
            char *av[] = {"ufsm", "strip-mosaic", a1, rdir, a3, "--threads", (char *)thr[k], "--rows", "2", "--overview-px", "600"};
            CHECK(!cmd_strip_mosaic(11, av), "strip-mosaic --threads %s failed", thr[k]);
        }
        snprintf(cmd, sizeof cmd, "cd %s/m1 && for x in *; do cmp -s $x ../m4/$x || exit 1; done; test $(ls | wc -l) -ge 6", root);
        CHECK(!system(cmd), "strip-mosaic --threads 4 output differs from --threads 1 (or files missing)");
        /* the streamed mode (stretch given, stripes taken as their .done markers appear) writes the same bytes */
        char sv[64] = {0}; snprintf(p, sizeof p, "%s/m1/stretch.txt", root); f = fopen(p, "r"); if (f) { if (!fgets(sv, sizeof sv, f)) sv[0] = 0; fclose(f); }
        sv[strcspn(sv, "\n")] = 0; CHECK(sv[0], "no stretch.txt");
        snprintf(cmd, sizeof cmd, "cd %s && for x in *_ct.tif; do touch ${x%%_ct.tif}.done; done", rdir); CHECK(!system(cmd), "touch .done");
        { char a1[400], a3[400]; snprintf(a1, sizeof a1, "%s/strip/flat", root); snprintf(a3, sizeof a3, "%s/m5", root);
          char *av[] = {"ufsm", "strip-mosaic", a1, rdir, a3, "--threads", "3", "--rows", "2", "--overview-px", "600", "--stretch", sv, "--wait-renders", "1"};
          CHECK(!cmd_strip_mosaic(15, av), "strip-mosaic --stretch --wait-renders failed"); }
        snprintf(cmd, sizeof cmd, "cd %s/m1 && for x in *; do cmp -s $x ../m5/$x || exit 1; done", root);
        CHECK(!system(cmd), "strip-mosaic --stretch --wait-renders output differs from the two-pass run");
        /* --papyrus: papyrus-look stripe JPEGs, a one-band overview / long strip; the same tables as the grey mosaic */
        { char a1[400], a3[400]; snprintf(a1, sizeof a1, "%s/strip/flat", root); snprintf(a3, sizeof a3, "%s/m6", root);
          char *av[] = {"ufsm", "strip-mosaic", a1, rdir, a3, "--threads", "2", "--rows", "2", "--overview-px", "600", "--stretch", sv, "--papyrus", "1", "--papyrus-bg", "128"};
          CHECK(!cmd_strip_mosaic(17, av), "strip-mosaic --papyrus failed"); }
        snprintf(cmd, sizeof cmd, "cd %s && ls m6/*_papyrus.jpg > /dev/null && test -s m6/overview.jpg && test -s m6/strip_long.jpg && cmp -s m1/strip_papyrus.tsv m6/strip_papyrus.tsv && cmp -s m1/papyrus_wraps.tsv m6/papyrus_wraps.tsv", root);
        CHECK(!system(cmd), "strip-mosaic --papyrus outputs missing or its tables differ");
        /* --fill-maps: fill maps marking every pixel rendered (flag 100: a bridged grid hole) widen the mask: valid area grows */
        { char fm[400]; snprintf(fm, sizeof fm, "%s/fm", root); snprintf(cmd, sizeof cmd, "mkdir -p %s", fm); CHECK(!system(cmd), "mkdir");
          snprintf(cmd, sizeof cmd, "cd %s && ls *_ct.tif", rdir); FILE *ls = popen(cmd, "r"); char nm[256];
          while (ls && fgets(nm, sizeof nm, ls)) { nm[strcspn(nm, "\n")] = 0; nm[strlen(nm) - 7] = 0; int w, h; snprintf(p, sizeof p, "%s/%s_ct.tif", rdir, nm); uint8_t *im = img_read_u8(p, &w, &h);
              if (im) { memset(im, 100, (size_t)w * h); snprintf(p, sizeof p, "%s/%s_fill.tif", fm, nm); CHECK(!tiffw_write(p, w, h, 1, im, 1), "write fill map"); free(im); } }
          if (ls) pclose(ls);
          /* a copy of the flattened strip with a hole in every stripe grid (vertices -1): the plain mosaic loses it, --fill-maps keeps it */
          snprintf(cmd, sizeof cmd, "cp -r %s/strip/flat %s/flat2", root, root); CHECK(!system(cmd), "cp flat");
          snprintf(cmd, sizeof cmd, "ls %s/flat2/stripes", root); ls = popen(cmd, "r");
          while (ls && fgets(nm, sizeof nm, ls)) { nm[strcspn(nm, "\n")] = 0; int w, h; snprintf(p, sizeof p, "%s/flat2/stripes/%s/z.tif", root, nm); float *Z = img_read_f32(p, &w, &h);
              if (Z) { for (int r = h / 4; r < h / 2; r++) for (int c = w / 4; c < w / 2; c++) Z[(size_t)r * w + c] = -1.f; CHECK(!tiffw_write(p, w, h, 3, Z, 1), "write z"); free(Z); } }
          if (ls) pclose(ls);
          char a1[400], a3[400]; snprintf(a1, sizeof a1, "%s/flat2", root);
          snprintf(a3, sizeof a3, "%s/m8", root); char *av0[] = {"ufsm", "strip-mosaic", a1, rdir, a3, "--threads", "2", "--rows", "2", "--overview-px", "600", "--stretch", sv};
          CHECK(!cmd_strip_mosaic(13, av0), "strip-mosaic on the holed grids failed");
          snprintf(a3, sizeof a3, "%s/m7", root);
          char *av[] = {"ufsm", "strip-mosaic", a1, rdir, a3, "--threads", "2", "--rows", "2", "--overview-px", "600", "--stretch", sv, "--fill-maps", fm};
          CHECK(!cmd_strip_mosaic(15, av), "strip-mosaic --fill-maps failed");
          double v1 = 0, v7 = 0; for (int k = 0; k < 2; k++) { snprintf(p, sizeof p, "%s/%s/strip_papyrus.tsv", root, k ? "m7" : "m8"); FILE *t = fopen(p, "r"); char ln[256];
              if (t) { if (fgets(ln, sizeof ln, t)) {} while (fgets(ln, sizeof ln, t)) { double u, v; if (sscanf(ln, "%lf %lf", &u, &v) == 2) *(k ? &v7 : &v1) += v; } fclose(t); } }
          CHECK(v7 > v1 && v1 > 0, "--fill-maps valid area %g vs %g", v7, v1);
          /* --papyrus-dir: the grey and the papyrus mosaic from one read write the files of the two separate runs */
          snprintf(a3, sizeof a3, "%s/m9", root); char a4[400]; snprintf(a4, sizeof a4, "%s/m9p", root);
          char *av9[] = {"ufsm", "strip-mosaic", a1, rdir, a3, "--threads", "3", "--stripe-jobs", "2", "--rows", "2", "--overview-px", "600", "--stretch", sv, "--fill-maps", fm,
                         "--papyrus-dir", a4, "--papyrus-gain", "2.5", "--papyrus-bg", "300", "--papyrus-overview-px", "350"};
          CHECK(!cmd_strip_mosaic(25, av9), "strip-mosaic --papyrus-dir failed");
          snprintf(a3, sizeof a3, "%s/m10", root);
          char *av10[] = {"ufsm", "strip-mosaic", a1, rdir, a3, "--threads", "1", "--stripe-jobs", "1", "--rows", "2", "--overview-px", "350", "--stretch", sv, "--fill-maps", fm,
                          "--papyrus", "2.5", "--papyrus-bg", "300"};
          CHECK(!cmd_strip_mosaic(21, av10), "strip-mosaic --papyrus --fill-maps failed");
          snprintf(cmd, sizeof cmd, "cd %s && for x in $(ls m7); do cmp -s m7/$x m9/$x || { echo $x; exit 1; }; done && for x in $(ls m10); do cmp -s m10/$x m9p/$x || { echo $x; exit 1; }; done && "
                   "test $(ls m9 | wc -l) = $(ls m7 | wc -l) && test $(ls m9p | wc -l) = $(ls m10 | wc -l)", root);
          CHECK(!system(cmd), "strip-mosaic --papyrus-dir output differs from the separate grey / --papyrus runs"); }
        /* one-pixel overview cells (small row bands) and the stripe / thread counts: the same bytes */
        { static const char *o2[2] = {"m11", "m12"}, *t2[2] = {"1", "5"}, *j2[2] = {"1", "3"};
          for (int k = 0; k < 2; k++) { char a1[400], a3[400], a4[400]; snprintf(a1, sizeof a1, "%s/strip/flat", root); snprintf(a3, sizeof a3, "%s/%s", root, o2[k]); snprintf(a4, sizeof a4, "%s/%sp", root, o2[k]);
              char *av[] = {"ufsm", "strip-mosaic", a1, rdir, a3, "--threads", (char *)t2[k], "--stripe-jobs", (char *)j2[k], "--rows", "1", "--overview-px", "100000", "--stretch", sv, "--papyrus-dir", a4};
              CHECK(!cmd_strip_mosaic(19, av), "strip-mosaic 1:1 overview --threads %s failed", t2[k]); }
          snprintf(cmd, sizeof cmd, "cd %s && for x in $(ls m11); do cmp -s m11/$x m12/$x || { echo $x; exit 1; }; done && for x in $(ls m11p); do cmp -s m11p/$x m12p/$x || { echo $x; exit 1; }; done", root);
          CHECK(!system(cmd), "strip-mosaic 1:1 overview: --threads 5 --stripe-jobs 3 differs from 1 / 1"); }
    }
    if (!fails && !getenv("TEST_STRIP_KEEP")) { snprintf(cmd, sizeof cmd, "rm -r %s", root); if (system(cmd)) {} }
    printf("test_strip: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
