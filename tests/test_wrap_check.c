/* ufsm wrap-check (src/wrapcheck.c) on a synthetic spiral: sheets every 16 native voxels (W_true = r / 16 - theta / 2 pi), a CT with a
   papyrus peak on each sheet, winding_mod14 codes from W_true. The right model (pitch 16) has one CT peak and one GT turn per fitted
   turn and no duplicate verdicts; a model counting twice the turns (pitch 8) has about half its intervals empty (no CT peak, GT 0
   turns), about 0.5 CT peaks and GT turns per fitted turn, and duplicate verdicts on most wraps. */
#include "fit.h"
#include "z3w.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cmd_wrap_check(int argc, char **argv);
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

enum { D = 16, HH = 128, H2 = 2, N = HH * H2 };   /* cells 16 x 128 x 128 at h 2: 32 x 256 x 256 native */
static const double C0 = 128, SP = 16;
static double W_true(double y, double x) { return hypot(y - C0, x - C0) / SP - atan2(y - C0, x - C0) / (2 * M_PI); }

static int save_model(const char *dir, double pitch) {
    fit_model m = {0}; m.dir = 1; m.nk = 2; m.knot_dz = 64; m.L = 1 << 20; m.lz = m.ly = m.lx = 2; m.D = D; m.H = HH; m.W = HH; m.h = H2; m.um = 9.6;
    float pt[2] = {(float)pitch, (float)pitch}, of[2] = {0, 0}, de[8] = {0}; m.pitch = pt; m.offset = of; m.delta = de;
    const int na = D * H2 + 1; float *cy = malloc(na * 4), *cx = malloc(na * 4); for (int i = 0; i < na; i++) { cy[i] = (float)C0; cx[i] = (float)C0; }
    m.cy = cy; m.cx = cx; const int rc = fit_save(&m, dir); free(cy); free(cx); return rc;
}
/* the TSV's verdict counts and mean empty share */
static void tsv_read(const char *p, int *real, int *dup, double *empty) {
    *real = *dup = 0; *empty = 0; int n = 0; FILE *f = fopen(p, "r"); char line[512];
    if (!f) return;
    if (!fgets(line, sizeof line, f)) { fclose(f); return; }
    while (fgets(line, sizeof line, f)) {
        char w[16], vd[16]; long long ni; double e; if (sscanf(line, "%15s %lld %lf %*f %*f %*f %*f %*f %*d %*f %*f %*f %*f %15s", w, &ni, &e, vd) != 4) continue;
        *real += !strcmp(vd, "real"); *dup += !strcmp(vd, "dup"); *empty += e; n++;
    }
    fclose(f); if (n) *empty /= n;
}

int main(void) {
    char dir[] = "/tmp/ufsm-wrapcheck-XXXXXX"; if (!mkdtemp(dir)) { fprintf(stderr, "FAIL: tmp dir\n"); return 1; }
    char p[600], m1[600], m2[600], fd[600], ct[600], cd[600], t1[600], t2[600];
    snprintf(m1, sizeof m1, "%s/m16", dir); snprintf(m2, sizeof m2, "%s/m8", dir); snprintf(fd, sizeof fd, "%s/fields", dir);
    snprintf(ct, sizeof ct, "%s/ct.zarr", dir); snprintf(cd, sizeof cd, "%s/codes", dir); snprintf(t1, sizeof t1, "%s/t16.tsv", dir); snprintf(t2, sizeof t2, "%s/t8.tsv", dir);
    CHECK(!save_model(m1, SP) && !save_model(m2, SP / 2), "save models");
    /* fields: material in r 40..120, spacing 16 native (sp code 16 x 8 / h), uninformative phase / normals */
    const size_t nc = (size_t)D * HH * HH; uint8_t *a[10]; for (int k = 0; k < 10; k++) a[k] = calloc(nc, 1);
    for (int z = 0; z < D; z++) for (int y = 0; y < HH; y++) for (int x = 0; x < HH; x++) {
        const size_t i = ((size_t)z * HH + y) * HH + x; const double r = hypot((y + 0.5) * H2 - C0, (x + 0.5) * H2 - C0);
        a[0][i] = a[1][i] = 128; a[8][i] = (uint8_t)(SP * 8 / H2); a[9][i] = r >= 40 && r <= 120 ? 255 : 0;
    }
    const int na = D * H2 + 1; float *cy = malloc(na * 4), *cx = malloc(na * 4); for (int i = 0; i < na; i++) { cy[i] = (float)C0; cx[i] = (float)C0; }
    const long long org[3] = {0, 0, 0}; const int shape[3] = {D, HH, HH};
    CHECK(!fit_fields_save(fd, org, shape, H2, 9.6, a, cy, cx, na), "save fields");
    /* CT (native grid, one 256^3 shard) and codes (key 9.6 = the native um) */
    const int64_t sh[3] = {D * H2, N, N}; uint8_t *v = calloc((size_t)N * N * N, 1), *q = calloc((size_t)N * N * N, 1);
    for (int z = 0; z < D * H2; z++) for (int y = 0; y < N; y++) for (int x = 0; x < N; x++) {
        const double w = W_true(y + 0.5, x + 0.5), f = w - floor(w + 0.5), dist = fabs(f) * SP; const size_t i = ((size_t)z * N + y) * N + x;
        v[i] = (uint8_t)lround(40 + 160 * exp(-dist * dist / 4.5));
        double wm = fmod(w, 14); if (wm < 0) wm += 14; q[i] = (uint8_t)(1 + (int)floor(wm * 18) % 252);
    }
    z3w *zw = z3w_create(ct, sh, 256, 0, 0, nullptr); CHECK(zw && !z3w_write_shard(zw, 0, 0, 0, v, 4) && !z3w_close(zw), "write CT");
    snprintf(p, sizeof p, "%s/9.6", cd);
    zw = z3w_create(p, sh, 256, 0, 0, nullptr); CHECK(zw && !z3w_write_shard(zw, 0, 0, 0, q, 4) && !z3w_close(zw), "write codes");
    char *av1[] = {"ufsm", "wrap-check", m1, fd, "--ct", dir, "ct.zarr", "--ct-um", "9.6", "--codes", dir, "codes/9.6", "--nz", "4", "--nth", "90", "--threads", "4", "--tsv", t1};
    CHECK(cmd_wrap_check(20, av1) == 0, "wrap-check pitch 16");
    av1[2] = m2; av1[19] = t2;
    CHECK(cmd_wrap_check(20, av1) == 0, "wrap-check pitch 8");
    int r1, d1, r2, d2; double e1, e2; tsv_read(t1, &r1, &d1, &e1); tsv_read(t2, &r2, &d2, &e2);
    printf("test_wrap_check: right model %d real / %d dup (mean empty %.3f); doubled model %d real / %d dup (mean empty %.3f)\n", r1, d1, e1, r2, d2, e2);
    CHECK(r1 >= 3 && d1 == 0 && e1 < 0.1, "right model: %d real, %d dup, empty %.3f", r1, d1, e1);
    CHECK(d2 >= 3 && d2 > r2 && e2 > 0.35 && e2 < 0.65, "doubled model: %d real, %d dup, empty %.3f", r2, d2, e2);
    for (int k = 0; k < 10; k++) free(a[k]);
    free(v); free(q); free(cy); free(cx);
    snprintf(p, sizeof p, "rm -rf '%s'", dir); if (system(p)) fprintf(stderr, "(could not remove %s)\n", dir);
    if (fails) { fprintf(stderr, "test_wrap_check: %d failure(s)\n", fails); return 1; }
    printf("test_wrap_check: OK\n");
    return 0;
}
