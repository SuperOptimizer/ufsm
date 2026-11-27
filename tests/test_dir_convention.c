/* The hand convention, pinned (2026-11-26, lead: "state every hand claim in one checked convention"). In the fit model's native frame,
   which the wraps / tifxyz outputs copy axis for axis (main.c cmd_wraps: x.tif = native x + origin[2], y.tif = native y + origin[1]):
   theta = atan2(y - cy, x - cx) (from +x toward +y), and dir = +1 means a sheet's radius GROWS as theta increases (one pitch per turn);
   dir = -1: it shrinks. spiral_dir.tsv and seghand.py use the same reading. Checked here on a delta-free model (pitch 8, axis at 32, 32):
   the W = k root along rays at theta 0, pi/2, pi - 1e-3 moves out by pitch / 4 per quarter turn for +1 and in for -1; the root point
   written as tifxyz x / y (x = cx + r cos theta, y = cy + r sin theta) has fit_W = k; and a save / load round trip keeps dir. CPU only. */
#include "fit.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

int main(void) {
    enum { S = 64, L = 16 }; const double P = 8, C = 32;
    for (int dir = -1; dir <= 1; dir += 2) {
        fit_model m; memset(&m, 0, sizeof m);
        m.dir = dir; m.nk = 1; m.knot_dz = S; float pit = (float)P, off = 0; m.pitch = &pit; m.offset = &off;
        m.L = L; m.lz = m.ly = m.lx = S / L + 1; m.delta = calloc((size_t)m.lz * m.ly * m.lx, sizeof(float));
        m.D = m.H = m.W = S; m.h = 1; m.um = 9.362;
        float *ay = malloc(sizeof(float) * (S + 1)), *ax = malloc(sizeof(float) * (S + 1)); for (int i = 0; i <= S; i++) { ay[i] = (float)C; ax[i] = (float)C; }
        m.cy = ay; m.cx = ax;
        const double k = 2, z = 20, th[3] = {0, M_PI / 2, M_PI - 1e-3}; double r[3];
        for (int i = 0; i < 3; i++) {
            CHECK(fit_ray_root(&m, z, th[i], k, P * k, &r[i]) == 0, "dir %+d: no W = %g root at theta %.3f", dir, k, th[i]);
            const double x = C + r[i] * cos(th[i]), y = C + r[i] * sin(th[i]);   /* tifxyz x / y of the root (origin 0) */
            CHECK(fabs(fit_W(&m, z, y, x) - k) < 1e-6, "dir %+d theta %.3f: W at the tifxyz point %.6f, want %g", dir, th[i], fit_W(&m, z, y, x), k);
        }
        const double q1 = r[1] - r[0], q2 = r[2] - r[0];
        CHECK(fabs(q1 - dir * P / 4) < 1e-3, "dir %+d: quarter-turn step %.4f, want %+.1f (radius grows with theta iff dir +1)", dir, q1, dir * P / 4);
        CHECK(fabs(q2 - dir * P / 2 * (1 - 1e-3 / M_PI)) < 1e-3, "dir %+d: half-turn step %.4f, want %+.3f", dir, q2, dir * P / 2);
        char d[600]; const char *t = getenv("TMPDIR"); snprintf(d, sizeof d, "%s/dirconv.%d.%d", t && *t ? t : "/tmp", (int)getpid(), dir + 1);
        char cmd[700]; snprintf(cmd, sizeof cmd, "mkdir -p %s", d); if (system(cmd)) return 1;
        m.origin[0] = m.origin[1] = m.origin[2] = 0;
        CHECK(fit_save(&m, d) == 0, "fit_save %s", d);
        fit_model *b = fit_load(d); CHECK(b && b->dir == dir, "dir %+d after save / load: %d", dir, b ? b->dir : 0);
        if (b) { double rb; CHECK(fit_ray_root(b, z, th[1], k, P * k, &rb) == 0 && fabs(rb - r[1]) < 1e-6, "dir %+d: loaded model's root %.6f vs %.6f", dir, rb, r[1]); fit_free(b); }
        snprintf(cmd, sizeof cmd, "rm -rf -- '%s'", d); if (strstr(d, "/dirconv.")) (void)!system(cmd);
        printf("dir %+d: r(theta 0) %.3f, r(pi/2) %.3f, r(pi) %.3f\n", dir, r[0], r[1], r[2]);
        free(m.delta); free(ay); free(ax);
    }
    printf("%s: test_dir_convention\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
