/* ufsm fields-phase (src/fieldsphase.c): teacher pseudo-label phase raster from a fit-field set.
   A raw field box (h 2, origin 8,16,16) with a linear phase in x, confident in its low-x half and weak in the high-x half; a 64 x 96 x 96
   --like / --ref raster with a few human labels inside the box. Checks: human labels kept, teacher codes at the expected phase where
   confident, unknown (255) where weak and outside the box. */
#include "fit.h"
#include "store.h"
#include "z3w.h"
#include "zarr3.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

int cmd_fields_phase(int argc, char **argv);   /* src/fieldsphase.c */

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

enum { D = 8, H = 16, W = 16, HH = 2, LZ = 64, LY = 96, LX = 96 };
static const long long ORG[3] = {8, 16, 16};
static double phase_at(double x) { return 0.013 * x + 0.2; }   /* x box-local native voxels */

int main(void) {
    char tmp[] = "/tmp/test_fields_phase_XXXXXX"; if (!mkdtemp(tmp)) { perror("mkdtemp"); return 1; }
    char fd[600], lk[600], out[600]; snprintf(fd, sizeof fd, "%s/fields", tmp); snprintf(lk, sizeof lk, "%s/like", tmp); snprintf(out, sizeof out, "%s/out", tmp);
    /* the field set: phase vector at each cell centre (x = h (k + 0.5)), confidence 0.8 for k < 8 and 0.2 for k >= 8 */
    static uint8_t P[10][D * H * W]; float cy[D * HH + 1], cx[D * HH + 1];
    for (int i = 0; i < D * HH + 1; i++) { cy[i] = 8; cx[i] = 8; }
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x; const double c = x < 8 ? 0.8 : 0.2, ph = phase_at(HH * (x + 0.5));
        P[0][i] = (uint8_t)lround(128 + 127 * c * cos(2 * M_PI * ph)); P[1][i] = (uint8_t)lround(128 + 127 * c * sin(2 * M_PI * ph));
        for (int k = 2; k < 10; k++) P[k][i] = 100;
    }
    uint8_t *a[10]; for (int k = 0; k < 10; k++) a[k] = P[k];
    const int sh[3] = {D, H, W};
    CHECK(fit_fields_save(fd, ORG, sh, HH, 9.6, a, cy, cx, D * HH + 1) == 0, "fit_fields_save");
    /* the like / ref raster: unknown except a few human labels (code 77) inside the confident half of the box */
    const int64_t ls[3] = {LZ, LY, LX}; z3w *w = z3w_create(lk, ls, 128, 0.f, 255, nullptr); CHECK(w != nullptr, "z3w_create");
    uint8_t *s = malloc(128 * 128 * 128); memset(s, 255, 128 * 128 * 128);
    const int hz = 12, hy = 20, hx = 20; s[((size_t)hz * 128 + hy) * 128 + hx] = 77;
    CHECK(z3w_write_shard(w, 0, 0, 0, s, 1) == 0, "like shard"); z3w_close(w);
    char *av[] = {"ufsm", "fields-phase", out, "--like", tmp, "like", "--ref", tmp, "like", "--conf", "0.5", "--threads", "2", fd};
    CHECK(cmd_fields_phase((int)(sizeof av / sizeof *av), av) == 0, "fields-phase rc");
    z3 *z = z3_open(store_open(tmp), "out", nullptr); CHECK(z != nullptr, "open out");
    if (z) {
        const int64_t o[3] = {0, 0, 0}, n[3] = {LZ, LY, LX}; uint8_t *v = malloc((size_t)LZ * LY * LX);
        CHECK(z3_read(z, o, n, v, 1) == 0, "read out");
        long long nt = 0, nbad = 0, nweak = 0, nout = 0;
        for (int zz = 0; zz < LZ; zz++) for (int yy = 0; yy < LY; yy++) for (int xx = 0; xx < LX; xx++) {
            const uint8_t c = v[((size_t)zz * LY + yy) * LX + xx];
            const int in = zz >= ORG[0] && zz < ORG[0] + D * HH && yy >= ORG[1] && yy < ORG[1] + H * HH && xx >= ORG[2] && xx < ORG[2] + W * HH;
            if (zz == hz && yy == hy && xx == hx) { CHECK(c == 77, "human label kept: %d", c); continue; }
            if (!in) { if (c != 255) nout++; continue; }
            const double lx = xx - ORG[2] + 0.5;
            if (lx < 7 * HH + 0.5 * HH) {   /* confident cells only (the trilinear weights stay inside the confident half) */
                const double le = lx < 0.5 * HH ? 0.5 * HH : lx;   /* clamped to the first cell centre (no extrapolation) */
                double e = c / 252.0 - (phase_at(le) - floor(phase_at(le))); e -= round(e);
                if (c > 251 || fabs(e) > 1.5 / 252) nbad++; else nt++;
            } else if (lx > 8 * HH + 0.5 * HH && c != 255) nweak++;
        }
        CHECK(nt > 1000, "teacher labels written: %lld", nt);
        CHECK(nbad == 0, "teacher codes off the field phase: %lld", nbad);
        CHECK(nweak == 0, "weak cells labelled: %lld", nweak);
        CHECK(nout == 0, "voxels outside the box labelled: %lld", nout);
        free(v);
    }
    free(s);
    printf("%s\n", fails ? "test_fields_phase FAILED" : "test_fields_phase OK");
    return fails != 0;
}
