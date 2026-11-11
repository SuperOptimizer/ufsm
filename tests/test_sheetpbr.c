/* src/sheetpbr.c (ufsm sheet-pbr) on a synthetic sheet: a CT whose papyrus sheet has a known recto relief z0(x) = 60 + 2 sin(2 pi x / 64)
   (12 voxels thick, smooth 1-voxel edges) and a second sheet 22 voxels further out, a flat mesh through the first sheet (z = 66, 4-voxel
   grid). Checks: the recto offset follows the relief (RMS error < 0.25 voxel, no sample on the second sheet), thickness ~ 12, the
   continuous DP surface reports no samples off the evidence, and the shading outputs exist. */
#include "tiffw.h"
#include "z3w.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cmd_sheet_pbr(int argc, char **argv);
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static double z0of(double x) { return 60 + 2 * sin(2 * M_PI * x / 64); }
static double edge(double d) { return 0.5 * (1 + erf(d / 1.0)); }   /* 0 -> 1 over ~1 voxel */
static float *readf(const char *p, size_t n) { FILE *f = fopen(p, "rb"); if (!f) return nullptr; float *a = malloc(n * sizeof *a); const size_t r = fread(a, sizeof *a, n, f); fclose(f); if (r != n) { free(a); return nullptr; } return a; }

int main(void) {
    char root[256]; snprintf(root, sizeof root, "/tmp/test_sheetpbr_%d", (int)getpid());
    char cmd[600], p[600]; snprintf(cmd, sizeof cmd, "rm -rf %s; mkdir -p %s", root, root); if (system(cmd)) return 1;
    /* the CT: 256 (z) x 256 x 256, air 30, papyrus 160 */
    const int S = 128; const int64_t sh[3] = {256, 256, 256};
    snprintf(p, sizeof p, "%s/ct.zarr", root); z3w *w = z3w_create(p, sh, S, 0, 0, nullptr); CHECK(w, "z3w_create");
    uint8_t *v = malloc((size_t)S * S * S);
    for (int sz = 0; sz < 2; sz++) for (int sy = 0; sy < 2; sy++) for (int sx = 0; sx < 2; sx++) {
        for (int z = 0; z < S; z++) for (int y = 0; y < S; y++) for (int x = 0; x < S; x++) {
            const double Z = sz * S + z, X = sx * S + x, a = z0of(X);
            const double s1 = edge(Z - a) * edge(a + 12 - Z), s2 = edge(Z - (a + 22)) * edge(a + 34 - Z), q = 30 + 130 * fmax(s1, s2);
            v[((size_t)z * S + y) * S + x] = (uint8_t)(q + 0.5); }
        CHECK(!z3w_write_shard(w, sz, sy, sx, v, 4), "shard"); }
    z3w_close(w); free(v);
    /* the mesh: z = 66, vertices every 4 voxels over x, y in [0, 252] */
    const int gw = 64, gh = 64; float *X = malloc(gw * gh * 4), *Y = malloc(gw * gh * 4), *Z = malloc(gw * gh * 4);
    for (int r = 0; r < gh; r++) for (int c = 0; c < gw; c++) { X[r * gw + c] = 4.0f * c; Y[r * gw + c] = 4.0f * r; Z[r * gw + c] = 66; }
    snprintf(p, sizeof p, "%s/mesh", root); CHECK(!tifxyz_write(p, gw, gh, X, Y, Z, 4, 2.4, -1, "test"), "tifxyz");
    free(X); free(Y); free(Z);
    /* run */
    char out[600]; snprintf(out, sizeof out, "%s/out", root);
    char *argv[] = {"ufsm", "sheet-pbr", p, out, "--ct", root, "ct.zarr", "--crop", "24,24,192", "--outward", "-1", "--depth", "30", "--lights", "4", "--threads", "4", nullptr};
    int argc = 0; while (argv[argc]) argc++;
    CHECK(cmd_sheet_pbr(argc, argv) == 0, "sheet-pbr failed");
    const int W = 192; const size_t N = (size_t)W * W;
    snprintf(p, sizeof p, "%s/recto_off.f32", out); float *r = readf(p, N);
    snprintf(p, sizeof p, "%s/thickness.f32", out); float *t = readf(p, N);
    CHECK(r && t, "maps missing");
    if (r && t) {
        double se = 0, st = 0; size_t n = 0, far = 0;
        for (int y = 0; y < W; y++) for (int x = 0; x < W; x++) { const size_t i = (size_t)y * W + x; if (!(r[i] == r[i])) continue;
            const double want = z0of(24 + x) - 66, e = r[i] - want; se += e * e; st += t[i]; n++; far += fabs(e) > 3; }
        const double rms = n ? sqrt(se / n) : 99, mt = n ? st / n : 0;
        printf("sheet-pbr synthetic: %zu of %zu samples, recto RMS error %.3f voxels, mean thickness %.2f, %zu samples > 3 voxels off\n", n, N, rms, mt, far);
        CHECK(n > N * 9 / 10, "too few samples: %zu", n);
        CHECK(rms < 0.25, "recto RMS error %.3f", rms);
        CHECK(fabs(mt - 12) < 0.75, "thickness %.2f", mt);
        CHECK(far == 0, "%zu samples on another interface", far);
    }
    free(r); free(t);
    const char *img[] = {"pbr.pgm", "light_00.pgm", "rti_min.pgm", "ao.pgm", "normals.ppm", "map.json"};
    for (int k = 0; k < 6; k++) { snprintf(p, sizeof p, "%s/%s", out, img[k]); FILE *f = fopen(p, "rb"); CHECK(f, "missing %s", img[k]); if (f) fclose(f); }
    snprintf(cmd, sizeof cmd, "rm -rf %s", root); if (system(cmd)) {}
    printf(fails ? "test_sheetpbr: %d FAILED\n" : "test_sheetpbr: all passed\n", fails);
    return fails != 0;
}
