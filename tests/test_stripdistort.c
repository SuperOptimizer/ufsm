/* ufsm strip-distort on synthetic stripes (the strip-flatten layout: flat.json, stripes.tsv, stripes/<s>/{x,y,z,gu}.tif, ../strip.tsv):
     ws000: an isometric flat (a plane, positions = the grid x ostep) with a 0.25-voxel noise like the strip's sfc error: dist ~ 0;
     ws001: the plane stretched 1.10 along u and compressed 0.95 along v: dist = max(1.10, 1 / 0.95) - 1 = 0.10, area 1.045;
     ws002: a cylinder of radius 300 voxels rolled flat along u (isometric): dist ~ 0, a fold band (u reversed over 24 columns) = folded.
   Checks the per-stripe p50s, the trusted shares, the folded share, the wrap per pixel (gu -> strip.tsv: two segments), and that 1 and
   4 threads write the same tsv bytes. */
#include "tiffw.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cmd_strip_distort(int argc, char **argv);
static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { fprintf(stderr, "FAIL: %s\n", m); fails++; } } while (0)
enum { W = 600, H = 200 };
static double nz(unsigned *s) { *s = *s * 1664525u + 1013904223u; return ((*s >> 8) / 16777216.0 - 0.5) * 0.5; }   /* +-0.25 */
static int stripe(const char *root, const char *name, int kind) {
    char d[600], p[700]; snprintf(d, sizeof d, "%s/flat/stripes/%s", root, name); snprintf(p, sizeof p, "mkdir -p %s", d); if (system(p)) return 1;
    float *x = malloc(W * H * 4), *y = malloc(W * H * 4), *z = malloc(W * H * 4), *g = malloc(W * H * 4); unsigned s = 12345u + kind; const double st = 4;
    for (int r = 0; r < H; r++) for (int c = 0; c < W; c++) {
        const int i = r * W + c; double X, Y, Z;
        if (kind == 0) { X = 100 + c * st; Y = 100 + r * st; Z = 50; }
        else if (kind == 1) { X = 100 + 1.10 * c * st; Y = 100 + 0.95 * r * st; Z = 50; }
        else { double u = c * st; if (c >= 300 && c < 324) u = (300 - (c - 300)) * st; const double a = u / 300; X = 1000 + 300 * cos(a); Y = 1000 + 300 * sin(a); Z = 100 + r * st; }
        x[i] = (float)(X + nz(&s)); y[i] = (float)(Y + nz(&s)); z[i] = (float)(Z + nz(&s)); g[i] = (float)(c < W / 2 ? 10 + c : 2000 + c);
        if (r < 3 && c < 5) x[i] = y[i] = z[i] = -1.f;   /* (a few invalid vertices) */
    }
    int rc = 0; const char *nm[4] = {"x", "y", "z", "gu"}; float *a[4] = {x, y, z, g};
    for (int k = 0; k < 4; k++) { snprintf(p, sizeof p, "%s/%s.tif", d, nm[k]); rc |= tiffw_write(p, W, H, 3, a[k], 1); }
    free(x); free(y); free(z); free(g); return rc;
}
static int run(const char *root, const char *out, int th, char *log, size_t n) {
    char fd[400]; snprintf(fd, sizeof fd, "%s/flat", root);
    char *av[] = {"ufsm", "strip-distort", fd, (char *)out, "--div", "8", "--span", "4", "--tile-cm", "0.5", "--trust", "0.05", "--threads", th == 4 ? "4" : "1"};
    char lp[400]; snprintf(lp, sizeof lp, "%s/log%d.txt", root, th); fflush(stdout); FILE *keep = fdopen(dup(1), "w"); if (!freopen(lp, "w", stdout)) return 1;
    const int rc = cmd_strip_distort((int)(sizeof av / sizeof *av), av);
    fflush(stdout); dup2(fileno(keep), 1); fclose(keep);
    FILE *f = fopen(lp, "r"); size_t m = f ? fread(log, 1, n - 1, f) : 0; log[m] = 0; if (f) fclose(f); fputs(log, stdout); return rc;
}
static int row(const char *tsv, const char *name, double v[7]) {   /* valid_cm2 dist_p50 dist_p95 area_p50 area_gmean trusted folded */
    FILE *f = fopen(tsv, "r"); char line[1024]; int ok = 0; if (!f) return 0;
    while (fgets(line, sizeof line, f)) { char nm[64]; int w, h; double t[7]; if (sscanf(line, "%63s %d %d %lf %lf %lf %lf %lf %lf %lf", nm, &w, &h, &t[0], &t[1], &t[2], &t[3], &t[4], &t[5], &t[6]) == 10 && !strcmp(nm, name)) { memcpy(v, t, sizeof t); ok = 1; } }
    fclose(f); return ok;
}
int main(void) {
    char root[256], p[700]; snprintf(root, sizeof root, "%s/test_stripdistort_%d", getenv("TMPDIR") ? getenv("TMPDIR") : "/tmp", (int)getpid());
    snprintf(p, sizeof p, "rm -rf %s && mkdir -p %s/flat/stripes", root, root); if (system(p)) return 1;
    snprintf(p, sizeof p, "%s/flat/flat.json", root); FILE *f = fopen(p, "w"); if (!f) return 1; fprintf(f, "{\"um\": 9.6, \"ostep\": 4, \"px_scale\": 1}\n"); fclose(f);
    snprintf(p, sizeof p, "%s/flat/stripes.tsv", root); f = fopen(p, "w"); if (!f) return 1;
    fprintf(f, "stripe\tout_col0\tout_cols\tout_rows\tu0_cm\tlength_cm\tvalid_cm2\tsrc_col0\tsrc_col1\twrap0\twrap1\n");
    for (int k = 0; k < 3; k++) fprintf(f, "ws%03d\t0\t%d\t%d\t0\t1\t1\t0\t1\t1\t1\n", k, W, H);
    fclose(f);
    snprintf(p, sizeof p, "%s/strip.tsv", root); f = fopen(p, "w"); if (!f) return 1;
    fprintf(f, "seg\twrap\tname\tcol0\tncols\twrap_col0\tseam_gap_pitch\n0\t7\tw007\t0\t1000\t1\t1\n1\t6\tw006\t1000\t2000\t1\t1\n"); fclose(f);
    for (int k = 0; k < 3; k++) { char nm[16]; snprintf(nm, sizeof nm, "ws%03d", k); if (stripe(root, nm, k)) return 1; }
    static char log[16384]; char o1[400], o4[400]; snprintf(o1, sizeof o1, "%s/out1", root); snprintf(o4, sizeof o4, "%s/out4", root);
    CHECK(!run(root, o1, 1, log, sizeof log), "strip-distort (1 thread) failed");
    CHECK(!run(root, o4, 4, log, sizeof log), "strip-distort (4 threads) failed");
    snprintf(p, sizeof p, "cd %s && cmp -s out1/stripes.tsv out4/stripes.tsv && cmp -s out1/tiles.tsv out4/tiles.tsv && cmp -s out1/wraps.tsv out4/wraps.tsv && [ -s out1/ws001_dist.jpg ] && [ -s out1/ws002_area.jpg ]", root);
    CHECK(!system(p), "outputs differ between 1 and 4 threads (or images missing)");
    double a[7], b[7], c[7]; snprintf(p, sizeof p, "%s/stripes.tsv", o1);
    CHECK(row(p, "ws000", a) && row(p, "ws001", b) && row(p, "ws002", c), "stripes.tsv rows missing");
    printf("test_stripdistort: ws000 dist p50 %.4f trusted %.4f | ws001 dist p50 %.4f area p50 %.4f trusted %.4f | ws002 dist p50 %.4f folded %.4f\n", a[1], a[5], b[1], b[3], b[5], c[1], c[6]);
    CHECK(a[1] < 0.02 && a[5] > 0.98 && a[6] < 0.001, "isometric plane: not ~0 distortion");
    CHECK(fabs(b[1] - 0.10) < 0.01 && fabs(b[3] - 1.045) < 0.01 && b[5] < 0.01, "stretched plane: dist not 0.10 / area not 1.045");
    CHECK(c[1] < 0.02 && c[6] > 0.025 && c[6] < 0.08, "rolled cylinder: not isometric, or the fold band not found");
    snprintf(p, sizeof p, "%s/wraps.tsv", o1); f = fopen(p, "r"); int n6 = 0, n7 = 0; char line[512];
    if (f) { while (fgets(line, sizeof line, f)) { n6 += !strncmp(line, "w006\t", 5); n7 += !strncmp(line, "w007\t", 5); } fclose(f); }
    CHECK(n6 == 1 && n7 == 1, "wraps.tsv: the two strip segments' wraps not both present");
    if (!fails) { snprintf(p, sizeof p, "rm -rf %s", root); if (system(p)) {} }
    printf("test_stripdistort: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
