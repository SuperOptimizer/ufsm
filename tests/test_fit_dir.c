/* The spiral's direction from the phase alone (src/fit.c fit_init_prior): the ring vote and the phase-loop vote (--dir-loops).
   Synthetic fields of a strongly non-circular spiral (radius x (1 + E cos 2 theta), E 0.35: one turn's radius varies by a factor 2, as
   on PHerc1667 where the ring vote found no coherent ring and the prior fell back to +1). For dir +1 and -1: with dir_loops the loops
   decide the right direction; without it, an empty ring vote stays +1 (the documented default, warned). A circular spiral: the ring
   vote decides both directions and the loops are not consulted. CPU only. */
#include "fit.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

enum { D = 4, N = 320 };
static const double H = 4, PITCH = 34;   /* field cell edge (native voxels), native pitch: ~8.5 cells per turn, the 1667 ratio */

static int prior_dir(int dir, double ecc, int loops, double *vote) {
    const size_t n = (size_t)D * N * N; uint8_t *pc = malloc(n), *ps = malloc(n), *sp = malloc(n), *w = malloc(n), *t0 = calloc(n, 1);
    const int na = (int)(D * H) + 1; float *cy = malloc(na * sizeof(float)), *cx = malloc(na * sizeof(float));
    for (int k = 0; k < na; k++) { cy[k] = (float)(N * H / 2 + 3); cx[k] = (float)(N * H / 2 - 5); }
    for (int z = 0; z < D; z++) for (int y = 0; y < N; y++) for (int x = 0; x < N; x++) {
        const size_t i = ((size_t)z * N + y) * N + x; const double py = (y + 0.5) * H - cy[0], px = (x + 0.5) * H - cx[0];
        const double r = hypot(py, px), th = atan2(py, px), re = r / (1 + ecc * cos(2 * th));
        const double W = re / PITCH - dir * th / (2 * M_PI) + 0.1 * sin(3 * th);   /* (+ a single-valued wobble: U need not be r / p) */
        const int in = r > 30 && r < N * H / 2 - 8;
        pc[i] = in ? (uint8_t)lround(128 + 120 * cos(2 * M_PI * W)) : 128; ps[i] = in ? (uint8_t)lround(128 + 120 * sin(2 * M_PI * W)) : 128;
        sp[i] = in ? (uint8_t)lround(PITCH * 8 / H) : 0; w[i] = in ? 255 : 0;   /* (sp in h / 8 units) */
    }
    fit_fields f = {0}; f.D = D; f.H = N; f.W = N; f.h = H; f.pc = pc; f.ps = ps; f.sp = sp; f.w = w; f.cy = cy; f.cx = cx;
    for (int k = 0; k < 6; k++) f.t[k] = t0;
    fit_cfg c = fit_cfg_default(); c.dir = 0; c.dir_loops = loops; c.knot_dz = 1024; c.verbose = 0;
    fit_model m; memset(&m, 0, sizeof m);
    const int rc = fit_init_prior(&f, &c, &m); const int got = rc ? 0 : m.dir; *vote = m.dir_vote;
    free(m.pitch); free(m.offset); free(pc); free(ps); free(sp); free(w); free(t0); free(cy); free(cx);
    return got;
}

int main(void) {
    for (int dir = -1; dir <= 1; dir += 2) {
        double v;
        int got = prior_dir(dir, 0.35, 1, &v);
        printf("non-circular, --dir-loops 1: dir %+d -> %+d (vote %.3f)\n", dir, got, v);
        CHECK(got == dir, "non-circular spiral dir %+d: the loops gave %+d", dir, got);
        got = prior_dir(dir, 0.35, 0, &v);
        printf("non-circular, --dir-loops 0: dir %+d -> %+d (vote %.3f)\n", dir, got, v);
        CHECK(v > 0 ? got == dir : got == 1, "non-circular dir %+d without loops: %+d (vote %.3f; an empty ring vote must stay +1)", dir, got, v);
        got = prior_dir(dir, 0, 0, &v);
        printf("circular, rings: dir %+d -> %+d (vote %.3f)\n", dir, got, v);
        CHECK(got == dir && v >= 0.5, "circular spiral dir %+d: the rings gave %+d (vote %.3f)", dir, got, v);
    }
    if (fails) { fprintf(stderr, "test_fit_dir: %d failure(s)\n", fails); return 1; }
    printf("test_fit_dir: ok\n"); return 0;
}
