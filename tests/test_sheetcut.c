/* test_sheetcut: ufsm sheet-cut on a synthetic cylinder wrap (radius 400 voxels, 120 rows x 300 columns, spacing 4) with a sheet
   switch (radius +12 voxels) on rows >= 80 and a hole: the largest rectangle must be the rows above the switch, all columns, and the
   switch rows must come back as a separate sheet; areas from the spacings. */
#include "tiffw.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int cmd_sheet_cut(int argc, char **argv);
float *img_read_f32(const char *path, int *w, int *h);
int main(void) {
    char tmp[] = "/tmp/test_sheetcut_XXXXXX"; if (!mkdtemp(tmp)) return 1;
    char in[300], out[300], p[400]; snprintf(in, sizeof in, "%s/in", tmp); snprintf(out, sizeof out, "%s/out", tmp);
    const int w = 300, h = 120; const double R = 400, st = 4; float *X = malloc(sizeof(float) * w * h), *Y = malloc(sizeof(float) * w * h), *Z = malloc(sizeof(float) * w * h);
    for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) {
        const size_t i = (size_t)r * w + c; const double rr = R + (r >= 80 ? 12 : 0), th = c * st / R;
        X[i] = (float)(1000 + rr * cos(th)); Y[i] = (float)(1000 + rr * sin(th)); Z[i] = (float)(r * st);
        if (r >= 100 && r < 110 && c >= 10 && c < 20) X[i] = Y[i] = Z[i] = -1;   /* a hole inside the switched part */
    }
    snprintf(p, sizeof p, "%s/w010", in); if (tifxyz_write(p, w, h, X, Y, Z, st, 9.6, 10, "test")) { printf("FAIL write\n"); return 1; }
    char *av[] = {"ufsm", "sheet-cut", in, out, "--min-cm2", "0.001", "--max", "3", "--margin", "0", nullptr};
    int fails = 0;
    if (cmd_sheet_cut(10, av)) { printf("FAIL sheet-cut rc\n"); fails++; }
    int ww, hh; snprintf(p, sizeof p, "%s/w010_r0/z.tif", out); float *z0 = img_read_f32(p, &ww, &hh);
    /* rows 0..79 are isometric (80 vertex rows: quads 0..78 plus the quad row 79-80 crosses the switch -> excluded) */
    if (!z0 || ww != w || hh != 80) { printf("FAIL r0 %dx%d (want %dx80)\n", ww, hh, w); fails++; }
    else if (z0[0] != 0) { printf("FAIL r0 starts at z %g\n", z0[0]); fails++; }
    snprintf(p, sizeof p, "%s/w010_r1/z.tif", out); float *z1 = img_read_f32(p, &ww, &hh);
    if (!z1 || hh < 2 || z1[0] < 80 * st - 0.5) { printf("FAIL r1 (switched rows) missing or wrong (%dx%d z0 %g)\n", ww, hh, z1 ? z1[0] : -9); fails++; }
    free(z0); free(z1); free(X); free(Y); free(Z);
    snprintf(p, sizeof p, "rm -rf '%s'", tmp); if (system(p)) {}
    printf(fails ? "test_sheetcut: FAIL\n" : "test_sheetcut: ok\n");
    return fails != 0;
}
