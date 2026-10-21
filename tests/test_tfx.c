/* .sfc-backed tifxyz wraps (src/tfx.c, src/tiffw.h): a synthetic curved wrap with invalid vertices written as a plain tifxyz
   directory and, through tifxyz_set_sfc, as <dir>.sfc; read back through the readers' entry points (img_read_f32 on <wrap>/x.tif,
   tfx_meta_fopen, tfx_exists) for the .sfc both by its own name and by the directory name it stands for: every valid vertex within
   the Euclidean budget, invalid ones -1, meta.json bytes identical, at scale 1 and at scale 0.25 (every 4th vertex). */
#include "imgops.h"
#include "tiffw.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)
static char *slurp(FILE *f) { if (!f) return nullptr; char *b = calloc(1, 65536); const size_t n = fread(b, 1, 65535, f); b[n] = 0; fclose(f); return b; }
static void roundtrip(int W, int H, double step, double err, const char *tag) {
    const size_t n = (size_t)W * H; float *x = malloc(n * 4), *y = malloc(n * 4), *z = malloc(n * 4);
    for (int r = 0; r < H; r++) for (int c = 0; c < W; c++) {   /* a piece of a cylinder (a wrap) around the axis (3000, 3000) */
        const size_t i = (size_t)r * W + c; const double th = 0.3 + c * step / 2500.0;
        x[i] = (float)(3000 + 2500 * cos(th)); y[i] = (float)(3000 + 2500 * sin(th) + 3 * sin(r * 0.05)); z[i] = (float)(9000.25 + r * step);
        if ((r * 7 + c * 3) % 41 == 0 || (r > H / 2 && c < W / 5)) x[i] = y[i] = z[i] = -1.f;
    }
    char d[256], ds[256], p[400]; snprintf(d, sizeof d, "/tmp/ufsm_test_tfx_%s_%d", tag, (int)getpid()); snprintf(ds, sizeof ds, "/tmp/ufsm_test_tfx_%s_%d_s", tag, (int)getpid());
    CHECK(!tifxyz_write(d, W, H, x, y, z, step, 8.64, 94, "test"), "%s: tifxyz write", tag);
    tifxyz_set_sfc(err); CHECK(!tifxyz_write(ds, W, H, x, y, z, step, 8.64, 94, "test"), "%s: sfc write", tag); tifxyz_set_sfc(0);
    snprintf(p, sizeof p, "%s.sfc", ds); FILE *f = fopen(p, "rb"); CHECK(f != nullptr, "%s: no %s", tag, p); if (f) fclose(f);
    snprintf(p, sizeof p, "%s/meta.json", d); char *m0 = slurp(fopen(p, "rb"));
    for (int pass = 0; pass < 2; pass++) {   /* the .sfc by its own name, then by the directory name it stands for */
        char wp[300]; snprintf(wp, sizeof wp, pass ? "%s" : "%s.sfc", ds);
        snprintf(p, sizeof p, "%s/meta.json", wp); CHECK(tfx_exists(p), "%s: tfx_exists %s", tag, p);
        char *m1 = slurp(tfx_meta_fopen(p));
        CHECK(m0 && m1 && !strcmp(strstr(m0, "\"scale\""), strstr(m1, "\"scale\"")), "%s: meta differs:\n%s\n%s", tag, m0 ? m0 : "-", m1 ? m1 : "-");
        free(m1);
        float *A[3]; int w = 0, h = 0, bad = 0, inv = 0; double emax = 0; static const char *ax[3] = {"x", "y", "z"};
        for (int k = 0; k < 3; k++) { snprintf(p, sizeof p, "%s/%s.tif", wp, ax[k]); A[k] = img_read_f32(p, &w, &h); }
        CHECK(A[0] && A[1] && A[2] && w == W && h == H, "%s: read %dx%d", tag, w, h);
        for (size_t i = 0; A[0] && A[1] && A[2] && i < n; i++) {
            if (z[i] < 0) { inv++; bad += A[0][i] != -1.f || A[2][i] != -1.f; continue; }
            const double e = sqrt((A[0][i] - x[i]) * (A[0][i] - x[i]) + (A[1][i] - y[i]) * (A[1][i] - y[i]) + (A[2][i] - z[i]) * (A[2][i] - z[i]));
            if (e > emax) emax = e;
        }
        printf("tfx %s pass %d: %dx%d step %g, %d invalid, max |xyz error| %.4f voxels (budget %g)\n", tag, pass, W, H, step, inv, emax, err);
        CHECK(!bad && emax <= err * 1.0001, "%s: %d invalid vertices decoded valid, max error %.4f > %g", tag, bad, emax, err);
        for (int k = 0; k < 3; k++) free(A[k]);
    }
    free(m0); free(x); free(y); free(z);
    snprintf(p, sizeof p, "rm -rf %s %s.sfc", d, ds); CHECK(!system(p), "cleanup");
}
int main(void) {
    roundtrip(300, 200, 1, 0.25, "full");      /* scale 1 */
    roundtrip(80, 60, 4, 0.25, "quarter");     /* scale 0.25: one vertex per 4 voxels */
    roundtrip(130, 70, 4, 0.1, "tight");       /* the library default tolerance */
    if (fails) { fprintf(stderr, "test_tfx: %d failures\n", fails); return 1; }
    printf("test_tfx: ok\n");
    return 0;
}
