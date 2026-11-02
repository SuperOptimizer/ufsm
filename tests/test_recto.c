/* ufsm recto-ruler / recto-cls / recto-sp (src/rectoruler.c, rectocls.c, rectosp.c) on a synthetic spiral (as test_wrap_check): sheets
   every 16 native voxels, a recto "probability" with a peak on each sheet, winding_mod14 GT codes near the sheets, radial normals.
   recto-ruler: the peaks count 1 per GT turn (offset-free), off share small; a dump is written; recto-cls trains on it; recto-sp gives
   the recto spacing 16 native voxels (sp code 64 at h 2) on the material cells. */
#include "fit.h"
#include "z3w.h"
#include "zarr3.h"
#include "store.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int cmd_recto_ruler(int argc, char **argv);
int cmd_recto_cls(int argc, char **argv);
int cmd_recto_sp(int argc, char **argv);
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

enum { D = 16, HH = 128, H2 = 2, N = HH * H2 };
static const double C0 = 128, SP = 16;
static double W_true(double y, double x) { return hypot(y - C0, x - C0) / SP - atan2(y - C0, x - C0) / (2 * M_PI); }

static int save_model(const char *dir, double pitch) {
    fit_model m = {0}; m.dir = 1; m.nk = 2; m.knot_dz = 64; m.L = 1 << 20; m.lz = m.ly = m.lx = 2; m.D = D; m.H = HH; m.W = HH; m.h = H2; m.um = 9.6;
    float pt[2] = {(float)pitch, (float)pitch}, of[2] = {0, 0}, de[8] = {0}; m.pitch = pt; m.offset = of; m.delta = de;
    const int na = D * H2 + 1; float *cy = malloc(na * 4), *cx = malloc(na * 4); for (int i = 0; i < na; i++) { cy[i] = (float)C0; cx[i] = (float)C0; }
    m.cy = cy; m.cx = cx; const int rc = fit_save(&m, dir); free(cy); free(cx); return rc;
}
static double tsv_val(const char *path, const char *table, const char *bin, const char *src) {
    FILE *f = fopen(path, "r"); char ln[512]; double v = NAN;
    while (f && fgets(ln, sizeof ln, f)) { char t[32], b[32], s[64]; double n, r, o; if (sscanf(ln, "%31s %31s %63s %lf %lf %lf", t, b, s, &n, &r, &o) == 6 && !strcmp(t, table) && !strcmp(b, bin) && !strcmp(s, src)) v = r; }
    if (f) fclose(f); return v;
}

int main(void) {
    char dir[] = "/tmp/ufsm-recto-XXXXXX"; if (!mkdtemp(dir)) { fprintf(stderr, "FAIL: tmp dir\n"); return 1; }
    char p[700], m1[600], fd[600], cd[600], tsv[600], dump[600], cls[600], out[600];
    snprintf(m1, sizeof m1, "%s/m16", dir); snprintf(fd, sizeof fd, "%s/fields", dir); snprintf(cd, sizeof cd, "%s/codes", dir); snprintf(tsv, sizeof tsv, "%s/rr.tsv", dir);
    snprintf(dump, sizeof dump, "%s/d.rrd", dir); snprintf(cls, sizeof cls, "%s/c.cls", dir); snprintf(out, sizeof out, "%s/fsp", dir);
    CHECK(!save_model(m1, SP), "save model");
    /* zarr fields (recto-sp reads zarr sets): material r 40..120, sp 16 native, radial normals */
    const size_t nc = (size_t)D * HH * HH; uint8_t *a[10]; for (int k = 0; k < 10; k++) a[k] = calloc(nc, 1);
    for (int z = 0; z < D; z++) for (int y = 0; y < HH; y++) for (int x = 0; x < HH; x++) {
        const size_t i = ((size_t)z * HH + y) * HH + x; const double dy = (y + 0.5) * H2 - C0, dx = (x + 0.5) * H2 - C0, r = hypot(dy, dx), ny = r > 0 ? dy / r : 0, nx = r > 0 ? dx / r : 1;
        a[0][i] = a[1][i] = 128; a[2][i] = 0; a[3][i] = (uint8_t)lround(255 * ny * ny); a[4][i] = (uint8_t)lround(255 * nx * nx); a[5][i] = 128; a[6][i] = 128; a[7][i] = (uint8_t)lround(255 * (0.5 + ny * nx));
        a[8][i] = (uint8_t)(SP * 8 / H2); a[9][i] = r >= 40 && r <= 120 ? 255 : 0;
    }
    const int64_t fsh[3] = {D, HH, HH}; const char *pn[10] = {"pc", "ps", "t0", "t1", "t2", "t3", "t4", "t5", "sp", "w"};
    for (int k = 0; k < 10; k++) { snprintf(p, sizeof p, "%s/%s", fd, pn[k]); uint8_t *sb = calloc((size_t)128 * 128 * 128, 1); memcpy(sb, a[k], nc);
        z3w *w = z3w_create(p, fsh, 128, 0, 0, nullptr); CHECK(w && !z3w_write_shard(w, 0, 0, 0, sb, 4) && !z3w_close(w), "write plane %s", pn[k]); free(sb); }
    {   snprintf(p, sizeof p, "%s/fields.json", fd); FILE *f = fopen(p, "w");
        fprintf(f, "{\"format\":\"zarr\",\"origin\":[0,0,0],\"shape\":[%d,%d,%d],\"h\":%d,\"um\":9.6,\"axis\":[", D, HH, HH, H2);
        for (int i = 0; i <= D * H2; i++) fprintf(f, "%s[%g,%g]", i ? "," : "", C0, C0);
        fprintf(f, "]}\n"); fclose(f); }
    /* recto "probability" and codes on the native grid */
    const int64_t sh[3] = {D * H2, N, N}; uint8_t *v = calloc((size_t)N * N * N, 1), *q = calloc((size_t)N * N * N, 1);
    for (int z = 0; z < D * H2; z++) for (int y = 0; y < N; y++) for (int x = 0; x < N; x++) {
        const double w = W_true(y + 0.5, x + 0.5), f = w - floor(w + 0.5), dist = fabs(f) * SP; const size_t i = ((size_t)z * N + y) * N + x;
        v[i] = (uint8_t)lround(10 + 230 * exp(-dist * dist / 3.0));
        double wm = fmod(w + 0.3, 14); if (wm < 0) wm += 14; const double f2 = (w + 0.3) - floor(w + 0.3 + 0.5);   /* (GT sheets offset 0.3 turn from the recto: offset-free counting) */
        q[i] = fabs(f2) * SP < 1.5 ? (uint8_t)(1 + (int)floor(wm * 18) % 252) : 0;
    }
    snprintf(p, sizeof p, "%s/rv.zarr", dir); z3w *zw = z3w_create(p, sh, 256, 0, 0, nullptr); CHECK(zw && !z3w_write_shard(zw, 0, 0, 0, v, 4) && !z3w_close(zw), "write recto");
    snprintf(p, sizeof p, "%s/9.6", cd); zw = z3w_create(p, sh, 256, 0, 0, nullptr); CHECK(zw && !z3w_write_shard(zw, 0, 0, 0, q, 4) && !z3w_close(zw), "write codes");
    char *av[] = {"ufsm", "recto-ruler", m1, fd, "--codes", dir, "codes/9.6", "--src", "rv", dir, "rv.zarr", "9.6", "--nz", "4", "--nth", "90", "--threads", "4", "--tsv", tsv, "--dump", dump, "--dump-src", "rv", "--step-um", "2.4"};
    CHECK(cmd_recto_ruler(25, av) == 0, "recto-ruler");
    const double pk = tsv_val(tsv, "radius", "all", "rv.pk"), fit = tsv_val(tsv, "radius", "all", "fit");
    printf("test_recto: fit %.3f, recto peaks %.3f turns per GT turn\n", fit, pk);
    CHECK(fabs(fit - 1) < 0.05 && fabs(pk - 1) < 0.05, "fit %.3f / peaks %.3f per GT turn", fit, pk);
    char *ac[] = {"ufsm", "recto-cls", "train", cls, dump, "--trees", "5", "--depth", "2"};
    CHECK(cmd_recto_cls(9, ac) == 0, "recto-cls train");
    CHECK(access(cls, F_OK) == 0, "classifier file");
    char *as[] = {"ufsm", "recto-sp", fd, out, "--src", dir, "rv.zarr", "9.6", "--stride", "2", "--half-um", "300", "--threads", "4"};
    CHECK(cmd_recto_sp(14, as) == 0, "recto-sp");
    {   z3 *z = z3_open(store_open(out), "rsp", nullptr); CHECK(z != nullptr, "open rsp");
        if (z) { uint8_t *b = malloc(nc); const int64_t o[3] = {0, 0, 0}, n[3] = {D, HH, HH}; CHECK(!z3_read(z, o, n, b, 4), "read rsp");
            long long cnt = 0, mat = 0; double s = 0; for (size_t i = 0; i < nc; i++) { if (!a[9][i]) continue; mat++; if (b[i]) { cnt++; s += b[i]; } }
            printf("test_recto: recto-sp: %lld of %lld material cells, mean sp code %.2f (expected %.0f)\n", cnt, mat, cnt ? s / cnt : 0, SP * 8 / H2);
            CHECK(cnt > mat / 2 && fabs(s / cnt - SP * 8 / H2) < 3, "recto spacing %.2f on %lld / %lld", cnt ? s / cnt : 0, cnt, mat);
            free(b); z3_close(z); } }
    for (int k = 0; k < 10; k++) free(a[k]);
    free(v); free(q);
    snprintf(p, sizeof p, "rm -rf '%s'", dir); if (system(p)) fprintf(stderr, "(could not remove %s)\n", dir);
    if (fails) { fprintf(stderr, "test_recto: %d failure(s)\n", fails); return 1; }
    printf("test_recto: OK\n");
    return 0;
}
