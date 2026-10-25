/* Sheet-identity embedding evidence (src/fit_emb.h) for the fitter's slip repair.
   (1) emb_shift_vote on a synthetic stack of planar sheets (spacing 10 native voxels, the packed case at 9.6 um): one random unit vector
       per sheet (+ noise), b's W planted r0 = -2..2 turns off; the vote must return r0 (sign convention: b calls a's sheet L + r).
   (2) test_repair's slip-region spiral (the phase loses a turn across a band inside a window; the flood leaves the region beyond it a
       turn off) with the repair's stretch weighting OFF: the plain repair cuts in the wrong place; with the embedding evidence (one
       vector per sheet of the true spiral) the repaired unwrapping must get >= 99% of the winding ids. */
#include "fit.h"
#include "fit_emb.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)
static uint64_t rng = 88172645463325252ull;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 11); }
static double gauss(void) { const double u = (rnd() + 1.0) / 4294967297.0, v = rnd() / 4294967296.0; return sqrt(-2 * log(u)) * cos(2 * M_PI * v); }
enum { C = 16 };
static void sheet_vec(long k, float *v) {   /* a fixed random unit vector per sheet index */
    uint64_t s = (uint64_t)(k + 1000) * 0x9E3779B97F4A7C15ull + 12345; double l = 0;
    for (int c = 0; c < C; c++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; v[c] = (float)((s >> 11) / 4503599627370496.0 - 1.0); l += v[c] * v[c]; }
    for (int c = 0; c < C; c++) v[c] /= (float)sqrt(l);
}
/* an embedding volume of n cells (step `step`) from Wt(p) (true winding at native p): the vector of sheet round(Wt) + noise */
static emb_vol *make_emb(const int64_t n[3], int step, double (*Wt)(double, double, double), double noise) {
    int8_t *v = calloc((size_t)n[0] * n[1] * n[2] * C, 1); float e[C];
    for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) for (int64_t x = 0; x < n[2]; x++) {
        const double W = Wt((z + 0.5) * step, (y + 0.5) * step, (x + 0.5) * step); if (isnan(W)) continue;
        sheet_vec(lround(W), e); double l = 0;
        for (int c = 0; c < C; c++) { e[c] += (float)(noise * gauss()); l += e[c] * e[c]; }
        int8_t *d = v + (((size_t)z * n[1] + y) * n[2] + x) * C;
        for (int c = 0; c < C; c++) d[c] = (int8_t)lround(127 * e[c] / sqrt(l));
    }
    const int64_t o[3] = {0, 0, 0}; const long long org[3] = {0, 0, 0};
    emb_vol *ev = emb_vol_wrap(v, n, C, step, o, org, 1.0); free(v); return ev;
}

/* (1) planar sheets normal to x, spacing 10 */
static double W_plane(double z, double y, double x) { (void)z; (void)y; return x / 10.0; }
static void test_vote(void) {
    const int64_t n[3] = {32, 32, 64}; emb_vol *e = make_emb(n, 2, W_plane, 0.3);
    int ok = 0, tot = 0;
    for (int t = 0; t < 400; t++) {
        const int r0 = (int)(rnd() % 5) - 2;
        emb_side a = {{20 + rnd() % 20, 20 + rnd() % 20, 40 + rnd() % 40}, 0, {0, 0, 1}, 10}, b = a;
        a.W = W_plane(0, 0, a.p[2]);
        b.p[0] = 10 + rnd() % 40; b.p[1] = 10 + rnd() % 40; b.p[2] = 40 + rnd() % 40;
        b.W = W_plane(0, 0, b.p[2]) + r0 + 0.04 * gauss();   /* b's labels r0 turns too high (+ phase noise) */
        int r = 99; float cf = 0;
        if (emb_shift_vote(e, &a, &b, 2, 0.0, &r, &cf)) continue;
        tot++; ok += r == r0;
    }
    printf("emb_shift_vote: %d / %d right (planted -2..2, spacing 10, noise 0.3)\n", ok, tot);
    CHECK(tot > 350 && ok >= 0.97 * tot, "emb_shift_vote: %d / %d right", ok, tot);
    /* the batch call agrees with the single one */
    emb_side A[64], B[64]; int32_t rb[64]; float cb[64];
    for (int k = 0; k < 64; k++) { A[k] = (emb_side){{30, 30, 50 + k % 10}, 0, {0, 0, 1}, 10}; A[k].W = W_plane(0, 0, A[k].p[2]); B[k] = A[k]; B[k].p[0] = 20; B[k].W += (k % 5) - 2; }
    emb_shift_votes(e, 64, A, B, 2, 0.0, rb, cb, 4);
    int same = 0; for (int k = 0; k < 64; k++) { int r; float cf; if (!emb_shift_vote(e, &A[k], &B[k], 2, 0.0, &r, &cf)) same += r == rb[k] && cf == cb[k]; }
    CHECK(same == 64, "emb_shift_votes: %d / 64 agree with emb_shift_vote", same);
    emb_vol_free(e);
}

/* (2) test_repair's slip-region spiral */
static const double H8 = 8, PITCH = 64, CY = 400, CX = 400;
static double smooth01(double t) { return t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t); }
static int in_window(double z, double th) { return z >= 64 && z < 320 && th > -0.3 && th < 1.3; }
static double lost(double z, double r, double th) { return in_window(z, th) ? smooth01((r - 200) / PITCH) : 0; }
static double W_true(double z, double y, double x) { (void)z; return hypot(y - CY, x - CX) / PITCH - atan2(y - CY, x - CX) / (2 * M_PI); }
static double run_spiral(const emb_vol *ev) {
    const int D = 48, H = 96, W = 96; const size_t n = (size_t)D * H * W;
    uint8_t *pc = malloc(n), *ps = malloc(n), *t[6], *sp = malloc(n), *w = malloc(n);
    for (int k = 0; k < 6; k++) t[k] = malloc(n);
    const int na = (int)(D * H8) + 1; float *cy = malloc(na * sizeof(float)), *cx = malloc(na * sizeof(float));
    for (int z = 0; z < na; z++) { cy[z] = (float)CY; cx[z] = (float)CX; }
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x;
        const double pz = (z + 0.5) * H8, py = (y + 0.5) * H8, px = (x + 0.5) * H8, dy = py - CY, dx = px - CX, r = hypot(dy, dx), th = atan2(dy, dx);
        w[i] = r >= 140 && r <= 380 ? 255 : 0;
        const double Wn = W_true(pz, py, px) - lost(pz, r, th), ph = 2 * M_PI * (Wn - floor(Wn));
        double conf = 0.85 + 0.15 * (rnd() % 1000) / 1000.0;
        const int wall = r > 264 && (((fabs(th + 0.3) < 0.06 || fabs(th - 1.3) < 0.06) && pz >= 64 && pz < 320) || ((fabs(pz - 64) < 12 || fabs(pz - 320) < 12) && th > -0.3 && th < 1.3));
        if (wall) conf = 0.35;
        pc[i] = (uint8_t)lround(128 + 127 * conf * cos(ph)); ps[i] = (uint8_t)lround(128 + 127 * conf * sin(ph));
        const double ny = dy / r, nx = dx / r; const double tv[6] = {0, ny * ny, nx * nx, 0.5, 0.5, 0.5 + ny * nx};
        for (int k = 0; k < 6; k++) t[k][i] = (uint8_t)lround(255 * tv[k]);
        sp[i] = (uint8_t)lround(8 * PITCH / H8);
    }
    fit_fields f = {D, H, W, H8, pc, ps, {t[0], t[1], t[2], t[3], t[4], t[5]}, sp, w, cy, cx, {0}, 0, nullptr, nullptr, nullptr};
    fit_cfg c = fit_cfg_default(); c.verbose = 1; c.cpu = 1; c.dir = 1; c.core = 0;
    c.uw_repair = 4; c.uw_rep_stretch = 0.f; c.steps[4] = c.steps[5] = 1;   /* the unwrapping's integers, no stretch weighting */
    if (ev) { c.emb = ev; c.uw_rep_emb = 4.f; c.uw_rep_emb_d[0] = 32; c.uw_rep_emb_d[1] = 96; c.uw_rep_emb_nd = 2; }
    fit_model *m = fit_run(&f, &c);
    if (!m) { fprintf(stderr, "FAIL: fit_run\n"); fails++; return 0; }
    int hist[401] = {0}; size_t nd = 0, right = 0;
    for (int pass = 0; pass < 2; pass++) {
        int k0 = 0; if (pass) for (int k = 1; k < 401; k++) if (hist[k] > hist[k0]) k0 = k;
        for (size_t i = 0; i < n; i++) {
            if (!w[i]) continue;
            const int z = (int)(i / ((size_t)H * W)), y = (int)((i / W) % H), x = (int)(i % W);
            const double pz = (z + 0.5) * H8, py = (y + 0.5) * H8, px = (x + 0.5) * H8, d = fit_W(m, pz, py, px) - W_true(pz, py, px);
            if (!pass) { const long k = lround(d); if (k >= -200 && k <= 200) hist[k + 200]++; nd++; }
            else right += fabs(d - (k0 - 200)) < 0.5;
        }
    }
    const double acc = (double)right / nd;
    printf("slip region, repair without stretch%s: winding ids right %.4f\n", ev ? " + embedding evidence" : "", acc);
    fit_free(m); free(pc); free(ps); free(sp); free(w); free(cy); free(cx); for (int k = 0; k < 6; k++) free(t[k]);
    return acc;
}
static double W_spiral_emb(double z, double y, double x) { const double r = hypot(y - CY, x - CX); return r < 120 || r > 400 ? NAN : W_true(z, y, x); }

int main(void) {
    test_vote();
    fit_set_threads(4);
    rng = 88172645463325252ull; const double a0 = run_spiral(nullptr);
    const int64_t n[3] = {96, 192, 192}; emb_vol *ev = make_emb(n, 4, W_spiral_emb, 0.3);
    rng = 88172645463325252ull; const double a1 = run_spiral(ev);
    emb_vol_free(ev);
    CHECK(a0 < 0.99, "the plain repair already gets %.4f (the case tests nothing)", a0);
    CHECK(a1 >= 0.99, "with the embedding evidence %.4f < 0.99", a1);
    if (fails) { fprintf(stderr, "test_fit_emb: %d failure(s)\n", fails); return 1; }
    printf("test_fit_emb: ok\n");
    return 0;
}
