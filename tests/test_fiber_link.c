/* ufsm fiber-link on synthetic pieces (level-0 voxels, --um 2.4, gap 0.3 mm = 125 voxels, lateral 0.03 mm = 12.5 voxels):
   - an H line along x (z 50, y 50, x 0 .. 400) cut into 4 pieces with 10-voxel gaps, the 2nd and 4th stored reversed -> one fibre with
     every point, in order (x monotone);
   - a V line along z at (y 50, x 150) crossing it, cut at z 45 / 55 -> one V fibre, never joined to the H pieces;
   - an H piece parallel to the line's last piece but 40 voxels off (lateral 96 um) starting just past its end -> not joined;
   - an H piece along y starting 5 voxels past the line's first end (90 degrees) -> not joined;
   - a closed loop of two H pieces (a circle cut twice) -> joined at one gap only (no cycle), all points kept.
   Checks: 5 output fibres, point count conserved, the long H fibre has 4 x 91 points and x monotone, the V fibre has both V pieces. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "afv.h"

int cmd_fiber_link(int argc, char **argv);
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)
static fib_set F; static size_t capp;
static void piece(char fam, double z0, double y0, double x0, double dz, double dy, double dx, int np) {
    if ((size_t)F.npts + np > capp) { capp = (capp + np) * 2; F.pts = realloc(F.pts, capp * 12); }
    for (int k = 0; k < np; k++) { F.pts[3 * (size_t)F.npts + 0] = (float)(z0 + k * dz); F.pts[3 * (size_t)F.npts + 1] = (float)(y0 + k * dy); F.pts[3 * (size_t)F.npts + 2] = (float)(x0 + k * dx); F.npts++; }
    F.off = realloc(F.off, (F.n + 2) * sizeof(int)); F.fam = realloc(F.fam, F.n + 2); F.fam[F.n] = fam; F.off[++F.n] = F.npts;
}

int main(void) {
    F.off = malloc(sizeof(int)); F.off[0] = 0; F.fam = malloc(1); capp = 1024; F.pts = malloc(capp * 12);
    /* the H line: pieces of 91 points (x a .. a + 90), gaps of 10 */
    piece('H', 50, 50, 0, 0, 0, 1, 91);
    piece('H', 50, 50, 190, 0, 0, -1, 91);    /* x 190 .. 100, reversed */
    piece('H', 50, 50, 200, 0, 0, 1, 91);
    piece('H', 50, 50, 390, 0, 0, -1, 91);    /* x 390 .. 300 */
    piece('V', 0, 50, 150, 1, 0, 0, 46);      /* z 0 .. 45 */
    piece('V', 55, 50, 150, 1, 0, 0, 46);     /* z 55 .. 100 */
    piece('H', 50, 90, 395, 0, 0, 1, 60);     /* parallel, 40 off in y, past the line's end */
    piece('H', 50, 55, -5, 0, 1, 0, 60);      /* perpendicular, at the line's first end */
    const int c0 = F.n; const double R = 60, cy = 400, cx = 400;   /* the loop: two half circles with 3-degree gaps */
    for (int h = 0; h < 2; h++) {
        const int np = 58; F.off = realloc(F.off, (F.n + 2) * sizeof(int)); F.fam = realloc(F.fam, F.n + 2);
        if ((size_t)F.npts + np > capp) { capp = (capp + np) * 2; F.pts = realloc(F.pts, capp * 12); }
        for (int k = 0; k < np; k++) { const double a = (h * 180 + 1.5 + k * 3.0) * M_PI / 180; F.pts[3 * (size_t)F.npts] = 200; F.pts[3 * (size_t)F.npts + 1] = (float)(cy + R * sin(a)); F.pts[3 * (size_t)F.npts + 2] = (float)(cx + R * cos(a)); F.npts++; }
        F.fam[F.n] = 'H'; F.off[++F.n] = F.npts;
    }
    (void)c0;
    char dir[] = "/tmp/ufsm-fiberlink-XXXXXX"; if (!mkdtemp(dir)) { fprintf(stderr, "FAIL: tmp dir\n"); return 1; }
    char in[300], out[300]; snprintf(in, sizeof in, "%s/in.fib", dir); snprintf(out, sizeof out, "%s/out.fib", dir);
    CHECK(!fib_save(in, &F), "save");
    char *av[] = {"ufsm", "fiber-link", in, out, "--um", "2.4", nullptr};
    CHECK(cmd_fiber_link(6, av) == 0, "fiber-link rc");
    fib_set O = {0}; CHECK(!fib_load(out, &O), "load");
    CHECK(O.n == 5, "fibres %d, want 5", O.n);
    CHECK(O.npts == F.npts, "points %d, want %d", O.npts, F.npts);
    int longh = 0, vfib = 0, loopok = 0;
    for (int r = 0; r < O.n; r++) {
        const int a = O.off[r], m = O.off[r + 1] - a;
        if (O.fam[r] == 'H' && m == 4 * 91) { longh++; int mono = 1; const double s = O.pts[3 * (size_t)(a + 1) + 2] - O.pts[3 * (size_t)a + 2];
            for (int k = 1; k < m; k++) { const double d = O.pts[3 * (size_t)(a + k) + 2] - O.pts[3 * (size_t)(a + k - 1) + 2]; if (d * s <= 0) mono = 0; }
            CHECK(mono, "the joined H line is not monotone in x"); }
        if (O.fam[r] == 'V') { vfib++; CHECK(m == 92, "V fibre %d points, want 92", m); }
        if (O.fam[r] == 'H' && m == 116) loopok++;
    }
    CHECK(longh == 1, "joined H lines %d, want 1", longh);
    CHECK(vfib == 1, "V fibres %d, want 1", vfib);
    CHECK(loopok == 1, "the loop: one fibre of 116 points, got %d", loopok);
    unlink(in); unlink(out); rmdir(dir); fib_free(&F); fib_free(&O);
    if (fails) { fprintf(stderr, "test_fiber_link: %d failures\n", fails); return 1; }
    printf("test_fiber_link: OK\n"); return 0;
}
