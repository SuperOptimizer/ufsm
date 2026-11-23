/* test_ink_atlas: the pure parts of ufsm ink-atlas (src/inkatlas.h) against hand-computed references: the 8 grid symmetries, the
   shifted Pearson registration (recovers a known symmetry + shift), the face-vote rule, the gate, and that a transposed tifxyz grid
   flips du x dv (the reverse-reading-side meshes). */
#include "../src/inkatlas.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

int main(void) {
    {   /* symmetries of a 3 x 2 raster (W 3, H 2): 0 1 2 / 3 4 5 */
        const uint8_t a[6] = {0, 1, 2, 3, 4, 5}; uint8_t o[6]; int w, h;
        static const uint8_t ref[8][6] = {{0, 1, 2, 3, 4, 5}, {2, 1, 0, 5, 4, 3}, {3, 4, 5, 0, 1, 2}, {5, 4, 3, 2, 1, 0},
                                          {0, 3, 1, 4, 2, 5}, {3, 0, 4, 1, 5, 2}, {2, 5, 1, 4, 0, 3}, {5, 2, 4, 1, 3, 0}};
        for (int t = 0; t < 8; t++) { ia_xform(a, 3, 2, t, o, &w, &h);
            CHECK(w == (t & 4 ? 2 : 3) && h == (t & 4 ? 3 : 2), "symmetry %d size %d x %d", t, w, h);
            CHECK(!memcmp(o, ref[t], 6), "symmetry %d: %d %d %d %d %d %d", t, o[0], o[1], o[2], o[3], o[4], o[5]); }
    }
    {   /* registration: b = symmetry 6 of a (a quarter turn), shifted by (2, -3); the search over symmetries x shifts finds the inverse
           turn and the shift at r ~ 1 */
        enum { N = 96 }; uint8_t *a = malloc(N * N), *b = malloc(N * N), *x = malloc(N * N), *s = malloc(N * N); unsigned q = 12345;
        for (int i = 0; i < N * N; i++) { q = q * 1103515245u + 12345u; a[i] = (uint8_t)(1 + (q >> 16) % 250); }
        for (int r = 0; r < N; r++) for (int c = 0; c < N; c++) { const double v = 120 + 60 * sin(r * 0.21) * cos(c * 0.17); a[r * N + c] = (uint8_t)((a[r * N + c] / 8 + (int)v) % 250 + 1); }
        int w, h; ia_xform(a, N, N, 6, x, &w, &h); ia_shift(x, N, N, 2, -3, b);   /* b(r, c) = sym6(a)(r + 2, c - 3) */
        double best = -2; int bt = -1, by = 0, bx = 0;
        for (int t = 0; t < 8; t++) { ia_xform(b, N, N, t, x, &w, &h);
            for (int dy = -4; dy <= 4; dy++) for (int dx = -4; dx <= 4; dx++) { size_t n; const double r = ia_pearson(a, x, N, N, dy, dx, 1, &n); if (r > best) { best = r; bt = t; by = dy; bx = dx; } } }
        /* sym6: out[r, c] = a[c, N - 1 - r]; its inverse a[N - 1 - c, r] = symmetry 5 */
        ia_xform(b, N, N, bt, x, &w, &h); ia_shift(x, N, N, by, bx, s);
        size_t same = 0, cmp = 0; for (int r = 8; r < N - 8; r++) for (int c = 8; c < N - 8; c++) { cmp++; same += s[r * N + c] == a[r * N + c]; }
        CHECK(best > 0.999 && bt == 5 && same == cmp, "registration: r %.4f symmetry %d shift (%d, %d), %zu / %zu px equal", best, bt, by, bx, same, cmp);
        size_t n; CHECK(isnan(ia_pearson(a, a, 10, 10, 0, 0, 1, &n)) && n == 100, "pearson below 1000 px must be NAN (n %zu)", n);
        free(a); free(b); free(x); free(s);
    }
    {   /* face votes: winner >= 3 and >= 2 x the loser */
        CHECK(ia_face_order(21, 0, 3, 2) == 1 && ia_face_order(4, 14, 3, 2) == -1 && ia_face_order(0, 0, 3, 2) == 0 && ia_face_order(11, 8, 3, 2) == 0 &&
              ia_face_order(2, 0, 3, 2) == 0 && ia_face_order(8, 4, 3, 2) == 1 && ia_face_order(7, 4, 3, 2) == 0, "face vote rule");
    }
    {   /* gate: ensemble z = -2 + 6 v / 255; 3 models, ink needs >= 2 at z_m >= 2 (cfg agree 2 here) */
        const ia_gate_cfg g = {2.5, 0.5, 2.0, 2};
        /* px: 0 bg (z -2), 1 z exactly 0.5 (v 106.25 -> 106: z 0.494 bg), 2 middle (z 1.53), 3 ink z 3.53 with 2 agreeing, 4 ink z but 1 agreeing,
           5 uncovered */
        const uint8_t ens[6] = {0, 106, 150, 235, 235, 255}, cov[6] = {1, 1, 1, 1, 1, 0};
        const uint8_t m0[6] = {0, 0, 0, 200, 200, 255}, m1[6] = {0, 0, 0, 200, 10, 255}, m2[6] = {0, 0, 0, 10, 10, 255};
        const uint8_t *mod[3] = {m0, m1, m2}; const double mean[3] = {0.2, 0.2, 0.2}, sd[3] = {0.1, 0.1, 0.1};   /* 200 -> z 5.8, 10 -> -1.6 */
        uint8_t lab[6], msk[6]; size_t ni, nb; ia_gate(ens, mod, mean, sd, 3, cov, 6, &g, lab, msk, &ni, &nb);
        static const uint8_t rl[6] = {0, 0, 0, 255, 0, 0}, rm[6] = {255, 255, 0, 255, 0, 0};
        CHECK(!memcmp(lab, rl, 6) && !memcmp(msk, rm, 6) && ni == 1 && nb == 2, "gate: lab %d %d %d %d %d %d msk %d %d %d %d %d %d ink %zu bg %zu",
              lab[0], lab[1], lab[2], lab[3], lab[4], lab[5], msk[0], msk[1], msk[2], msk[3], msk[4], msk[5], ni, nb);
    }
    {   /* a transposed grid flips du x dv: the plane z = 0.3 x + 0.1 y, grid x = 2 c, y = 3 r */
        double du[3] = {2, 0, 0.6}, dv[3] = {0, 3, 0.3};   /* (x, y, z) derivatives along columns / rows */
        double n1[3] = {du[1] * dv[2] - du[2] * dv[1], du[2] * dv[0] - du[0] * dv[2], du[0] * dv[1] - du[1] * dv[0]};
        /* transposed: column index = old row, so du' = dv, dv' = du */
        double n2[3] = {dv[1] * du[2] - dv[2] * du[1], dv[2] * du[0] - dv[0] * du[2], dv[0] * du[1] - dv[1] * du[0]};
        CHECK(n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2] < 0, "transpose flips the normal");
    }
    if (fails) { fprintf(stderr, "test_ink_atlas: %d failures\n", fails); return 1; }
    printf("test_ink_atlas: ok\n"); return 0;
}
