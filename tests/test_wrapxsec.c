/* wrap-xsec / wrap-align (src/wrapxsec.c, PHerc1203 hand / lamination checks 2026-11-26) on a synthetic laminated volume: 256^3, sheets
   normal to z (v = 128 + 100 sin(2 pi z / 8)). A mesh in a z = const plane follows the lamination (wrap-align share 1); a mesh in an x = const
   plane crosses it (share 0). wrap-xsec draws the x = const mesh's cut with slice z = 128 as the column x = 100 in red; elsewhere grey. */
#include "z3w.h"
#include "tiffw.h"
#include "imgops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cmd_wrap_xsec(int argc, char **argv);
int cmd_wrap_align(int argc, char **argv);
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)

static double share_of(const char *tsv) {   /* the whole mesh's aligned share from the per-block tsv */
    FILE *f = fopen(tsv, "r"); if (!f) return -1; char line[512]; double n = 0, a = 0; int first = 1;
    while (fgets(line, sizeof line, f)) { if (first) { first = 0; continue; } int r0, c0, rr, cc; double bn, ba; if (sscanf(line, "%d %d %d %d %lf %lf", &r0, &c0, &rr, &cc, &bn, &ba) == 6) { n += bn; a += ba; } }
    fclose(f); return n > 0 ? a / n : -1;
}

int main(void) {
    const char *t = getenv("TMPDIR"); char dir[600], vol[700], ma[700], mb[700], p[800];
    snprintf(dir, sizeof dir, "%s/wrapxsec.%d", t && *t ? t : "/tmp", (int)getpid());
    snprintf(vol, sizeof vol, "%s/vol", dir); snprintf(ma, sizeof ma, "%s/za", dir); snprintf(mb, sizeof mb, "%s/xb", dir);
    snprintf(p, sizeof p, "mkdir -p %s", dir); if (system(p)) return 1;
    const int64_t shape[3] = {256, 256, 256}; uint8_t *d = malloc((size_t)256 * 256 * 256);
    for (int z = 0; z < 256; z++) { const uint8_t v = (uint8_t)lround(128 + 100 * sin(2 * M_PI * z / 8)); memset(d + (size_t)z * 256 * 256, v, (size_t)256 * 256); }
    z3w *w = z3w_create(vol, shape, 256, 0.f, 0, nullptr); CHECK(w != nullptr, "create %s", vol); if (!w) return 1;
    CHECK(!z3w_write_shard(w, 0, 0, 0, d, 2), "write shard"); z3w_close(w);
    /* meshes: 120 x 120 vertices, step 1; A in the plane z = 128 (x, y 60..180), B in the plane x = 100 (y 60..180, z 60..180) */
    const int G = 120; float *X = malloc(sizeof *X * G * G), *Y = malloc(sizeof *Y * G * G), *Z = malloc(sizeof *Z * G * G);
    for (int r = 0; r < G; r++) for (int c = 0; c < G; c++) { X[r * G + c] = 60.f + c; Y[r * G + c] = 60.f + r; Z[r * G + c] = 128.f; }
    CHECK(!tifxyz_write(ma, G, G, X, Y, Z, 1.0, 1.0, -1, "test_wrapxsec A"), "write mesh A");
    for (int r = 0; r < G; r++) for (int c = 0; c < G; c++) { X[r * G + c] = 100.f; Y[r * G + c] = 60.f + c; Z[r * G + c] = 60.f + r; }
    CHECK(!tifxyz_write(mb, G, G, X, Y, Z, 1.0, 1.0, -1, "test_wrapxsec B"), "write mesh B");
    char oa[800], ob[800]; snprintf(oa, sizeof oa, "%s/aa", dir); snprintf(ob, sizeof ob, "%s/ab", dir);
    char *aa[] = {"ufsm", "wrap-align", ma, "--ct", vol, ".", "--level", "0", "--every", "4", "--block", "64", "--out", oa, nullptr};
    char *ab[] = {"ufsm", "wrap-align", mb, "--ct", vol, ".", "--level", "0", "--every", "4", "--block", "64", "--out", ob, nullptr};
    CHECK(cmd_wrap_align(14, aa) == 0, "wrap-align A"); CHECK(cmd_wrap_align(14, ab) == 0, "wrap-align B");
    snprintf(p, sizeof p, "%s.tsv", oa); const double sa = share_of(p); snprintf(p, sizeof p, "%s.tsv", ob); const double sb = share_of(p);
    CHECK(sa > 0.99, "mesh in the sheet plane: aligned share %.4f, want 1", sa);
    CHECK(sb >= 0 && sb < 0.01, "mesh across the sheets: aligned share %.4f, want 0", sb);
    char jp[800], spec[800]; snprintf(jp, sizeof jp, "%s/x.jpg", dir); snprintf(spec, sizeof spec, "255,0,0:%s", mb);
    char *ax[] = {"ufsm", "wrap-xsec", jp, "--ct", vol, ".", "--level", "0", "--z", "128", "--dot", "1", spec, nullptr};
    CHECK(cmd_wrap_xsec(13, ax) == 0, "wrap-xsec");
    int iw = 0, ih = 0; uint8_t *img = jpeg_read_rgb(jp, &iw, &ih);
    CHECK(img && iw == 256 && ih == 256, "xsec image %d x %d", iw, ih);
    if (img && iw == 256 && ih == 256) {
        const uint8_t *on = img + 3 * (120 * 256 + 100), *off = img + 3 * (120 * 256 + 30), *out = img + 3 * (20 * 256 + 100);
        CHECK(on[0] > 180 && on[1] < 90 && on[2] < 90, "cut point (y 120, x 100) not red: %d %d %d", on[0], on[1], on[2]);
        CHECK(abs(off[0] - off[1]) < 25 && abs(off[1] - off[2]) < 25, "slice pixel (y 120, x 30) not grey: %d %d %d", off[0], off[1], off[2]);
        CHECK(abs(out[0] - out[1]) < 25, "pixel (y 20, x 100) outside the mesh's y range coloured: %d %d %d", out[0], out[1], out[2]);
    }
    free(img); free(X); free(Y); free(Z); free(d);
    snprintf(p, sizeof p, "rm -rf -- '%s'", dir); if (strstr(dir, "/wrapxsec.")) (void)!system(p);
    printf("%s: test_wrapxsec (aligned shares %.4f / %.4f)\n", fails ? "FAIL" : "PASS", sa, sb);
    return fails ? 1 : 0;
}
