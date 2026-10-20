/* Local slip repair (fitter Phase 4): src/l1pot.h and fit_cfg uw_repair.
   (1) l1_solve is exact: random small vote sets against exhaustive search, the Boykov-Kolmogorov flow against the Dinic reference
       (UFSM_L1_DINIC) on larger random sets.
   (2) Synthetic spirals (the test_fit geometry without the packed sector) whose network phase skips sheets the way the Paris 4 held-out
       box's does: inside a window of the scroll the phase loses a turn across a band one sheet wide (stretched: one cycle where the sheets
       make two) and, for the ring, gains it back further out. Beyond the band the phase is right mod 1, so it is consistent everywhere but
       in the band; confidence is high there and low on thin walls around the window (the flood crosses the band and leaves the region
       beyond it a turn off). Cases: a slip region (one band in a sector) and a slip ring (a lost and a regained band at every angle
       over a z range). The repair (--uw-repair 4 with the stretch weighting) must restore >= 99% of the winding ids; the flood alone
       must not (else the case tests nothing). */
#include "fit.h"
#include "l1pot.h"
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

static uint64_t rng = 88172645463325252ull;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 11); }

/* (1) exactness */
static void test_l1(void) {
    for (int trial = 0; trial < 300; trial++) {   /* 5 nodes, shifts -6..6 (node 0 pinned by the gauge): exhaustive */
        const int n = 5; l1_vote v[24]; size_t nv = 0;
        for (int k = 0; k < 12; k++) {
            uint32_t a = rnd() % n, b = rnd() % n; if (a == b) continue;
            v[nv++] = (l1_vote){a, b, (int32_t)(rnd() % 5) - 2, 1 + rnd() % 9};
        }
        nv = l1_votes_merge(v, nv);
        int64_t best = INT64_MAX;
        for (int code = 0; code < 13 * 13 * 13 * 13; code++) {
            int32_t s[5] = {0}; int c = code; for (int k = 1; k < 5; k++) { s[k] = c % 13 - 6; c /= 13; }
            const int64_t e = l1_energy(v, nv, s); if (e < best) best = e;
        }
        int32_t s[5] = {0}; l1_stats st; l1_solve(n, v, nv, s, 64, &st);
        CHECK(st.e1 == best && l1_energy(v, nv, s) == best, "l1 trial %d: solver %lld, exhaustive %lld", trial, (long long)st.e1, (long long)best);
    }
    /* a 3D grid of 12^3 nodes with consistent votes plus a ring of inconsistent ones and random noise: BK = Dinic */
    const int G = 12, n = G * G * G; l1_vote *v = malloc(sizeof *v * (size_t)n * 4); size_t nv = 0;
    for (int z = 0; z < G; z++) for (int y = 0; y < G; y++) for (int x = 0; x < G; x++) {
        const uint32_t i = (uint32_t)((z * G + y) * G + x);
        for (int ax = 0; ax < 3; ax++) {
            const int zn = z + (ax == 0), yn = y + (ax == 1), xn = x + (ax == 2); if (zn >= G || yn >= G || xn >= G) continue;
            const uint32_t j = (uint32_t)((zn * G + yn) * G + xn);
            int32_t d = (y < 6 && ax == 2 && x == 5) ? 1 : 0;   /* a half-plane cut whose edge is a vortex line */
            if (rnd() % 10 == 0) d += (int32_t)(rnd() % 3) - 1;
            v[nv++] = (l1_vote){i, j, d, 1 + rnd() % 20};
        }
    }
    nv = l1_votes_merge(v, nv);
    int32_t *s1 = calloc(n, 4), *s2 = calloc(n, 4); l1_stats a, b;
    l1_solve(n, v, nv, s1, 256, &a);
    setenv("UFSM_L1_DINIC", "1", 1); l1_solve(n, v, nv, s2, 256, &b); unsetenv("UFSM_L1_DINIC");
    printf("l1: grid %d nodes %zu edges: E %lld -> %lld (BK, %d moves), %lld (Dinic)\n", n, a.nedge, (long long)a.e0, (long long)a.e1, a.moves, (long long)b.e1);
    CHECK(a.e1 == b.e1 && a.e1 < a.e0, "l1 grid: BK %lld, Dinic %lld, start %lld", (long long)a.e1, (long long)b.e1, (long long)a.e0);
    free(v); free(s1); free(s2);
}

/* (2) synthetic spirals: a cylinder of sheets around a vertical axis (U = r / pitch), the network phase W_net = W_true - lost(x),
   lost = the turns the network skipped between the axis and x (a smoothstep 0 -> 1 across each band inside the window) */
static const double H8 = 8, PITCH = 64, CY = 400, CX = 400;
static int g_ring;
static double smooth01(double t) { return t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t); }
static int in_window(double z, double th) { return g_ring ? z >= 96 && z < 288 : z >= 64 && z < 320 && th > -0.3 && th < 1.3; }
static double lost(double z, double r, double th) {
    if (!in_window(z, th)) return 0;
    double l = smooth01((r - 200) / PITCH);   /* band 200 .. 264: the phase stalls where the sheets make one turn (stretch 0 .. 1) */
    if (g_ring) l -= smooth01((r - 296) / PITCH);   /* ring: regained across 296 .. 360 (two cycles in one spacing) */
    return l;
}
static double W_true(double z, double y, double x) { return hypot(y - CY, x - CX) / PITCH - atan2(y - CY, x - CX) / (2 * M_PI); }
static double run_spiral(int repair, int steps, double *flood_ids) {
    const int D = 48, H = 96, W = 96; const size_t n = (size_t)D * H * W;
    uint8_t *pc = malloc(n), *ps = malloc(n), *t[6], *sp = malloc(n), *w = malloc(n);
    for (int k = 0; k < 6; k++) t[k] = malloc(n);
    const int na = (int)(D * H8) + 1;
    float *cy = malloc(na * sizeof(float)), *cx = malloc(na * sizeof(float));
    for (int z = 0; z < na; z++) { cy[z] = (float)CY; cx[z] = (float)CX; }
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x;
        const double pz = (z + 0.5) * H8, py = (y + 0.5) * H8, px = (x + 0.5) * H8, dy = py - CY, dx = px - CX, r = hypot(dy, dx), th = atan2(dy, dx);
        w[i] = r >= 140 && r <= 380 ? 255 : 0;
        const double Wn = W_true(pz, py, px) - lost(pz, r, th), ph = 2 * M_PI * (Wn - floor(Wn));
        /* confidence: high (0.85-1) with noise; the window's walls (just outside it, beyond the band) low: there the flood arrives last */
        double conf = 0.85 + 0.15 * (rnd() % 1000) / 1000.0;
        const int wall = r > (g_ring ? 200 : 264) && (g_ring ? (fabs(pz - 96) < 12 || fabs(pz - 288) < 12) : ((fabs(th + 0.3) < 0.06 || fabs(th - 1.3) < 0.06) && pz >= 64 && pz < 320) || ((fabs(pz - 64) < 12 || fabs(pz - 320) < 12) && th > -0.3 && th < 1.3));
        if (wall) conf = 0.35;
        pc[i] = (uint8_t)lround(128 + 127 * conf * cos(ph)); ps[i] = (uint8_t)lround(128 + 127 * conf * sin(ph));
        const double ny = dy / r, nx = dx / r;   /* sheets ~ cylinders: the radial normal, the true spacing (the density head is right) */
        const double tv[6] = {0, ny * ny, nx * nx, 0.5, 0.5, 0.5 + ny * nx};
        for (int k = 0; k < 6; k++) t[k][i] = (uint8_t)lround(255 * tv[k]);
        const double s = 8 * PITCH / H8; sp[i] = (uint8_t)lround(s);
    }
    fit_fields f = {D, H, W, H8, pc, ps, {t[0], t[1], t[2], t[3], t[4], t[5]}, sp, w, cy, cx, {0}, 0, nullptr, nullptr, nullptr};
    fit_cfg c = fit_cfg_default(); c.verbose = 1; c.cpu = 1; c.dir = 1; c.core = 0;
    if (repair) { c.uw_repair = 4; c.uw_rep_stretch = 0.5f; }
    if (steps) c.steps[4] = c.steps[5] = steps;   /* 1: the phase stages barely move: W is the unwrapping's (its integers) */
    fit_model *m = fit_run(&f, &c);
    if (!m) { fprintf(stderr, "FAIL: fit_run\n"); fails++; return 0; }
    int hist[401] = {0}; size_t nd = 0;
    for (size_t i = 0; i < n; i++) {
        if (!w[i]) continue;
        const int z = (int)(i / ((size_t)H * W)), y = (int)((i / W) % H), x = (int)(i % W);
        const double pz = (z + 0.5) * H8, py = (y + 0.5) * H8, px = (x + 0.5) * H8;
        const long k = lround(fit_W(m, pz, py, px) - W_true(pz, py, px)); if (k >= -200 && k <= 200) hist[k + 200]++;
        nd++;
    }
    int k0 = 0; for (int k = 1; k < 401; k++) if (hist[k] > hist[k0]) k0 = k;
    size_t right = 0, rw = 0, nw = 0;
    for (size_t i = 0; i < n; i++) {
        if (!w[i]) continue;
        const int z = (int)(i / ((size_t)H * W)), y = (int)((i / W) % H), x = (int)(i % W);
        const double pz = (z + 0.5) * H8, py = (y + 0.5) * H8, px = (x + 0.5) * H8;
        const int ok = fabs(fit_W(m, pz, py, px) - W_true(pz, py, px) - (k0 - 200)) < 0.5; right += ok;
        if (in_window(pz, atan2(py - CY, px - CX))) { nw++; rw += ok; }
    }
    const double acc = (double)right / nd;
    printf("%s %s, %s: winding ids right %.4f (window %.4f, %zu voxels)\n", g_ring ? "slip ring" : "slip region", repair ? "repaired" : "flood only",
           steps ? "unwrapping" : "full fit", acc, nw ? (double)rw / nw : 0, nd);
    if (flood_ids) *flood_ids = acc;
    fit_free(m); free(pc); free(ps); free(sp); free(w); free(cy); free(cx); for (int k = 0; k < 6; k++) free(t[k]);
    return acc;
}

int main(void) {
    test_l1();
    fit_set_threads(4);
    for (g_ring = 0; g_ring <= 1; g_ring++) {   /* the unwrapping's integers (phase stages of 1 step), then the full fit */
        const char *nm = g_ring ? "ring" : "region";
        rng = 88172645463325252ull; const double u0 = run_spiral(0, 1, nullptr);
        rng = 88172645463325252ull; const double u1 = run_spiral(1, 1, nullptr);
        CHECK(u0 < 0.9, "%s: the flood alone already gets %.4f of the ids (the case does not slip)", nm, u0);
        CHECK(u1 >= 0.99, "%s: the repaired unwrapping gets %.4f < 0.99 of the ids", nm, u1);
        rng = 88172645463325252ull; run_spiral(0, 0, nullptr);
        rng = 88172645463325252ull; const double a1 = run_spiral(1, 0, nullptr);
        CHECK(a1 >= 0.99, "%s: the repaired fit gets %.4f < 0.99 of the ids", nm, a1);
    }
    if (fails) { fprintf(stderr, "test_repair: %d failure(s)\n", fails); return 1; }
    printf("test_repair: ok\n");
    return 0;
}
