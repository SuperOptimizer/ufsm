/* tests/test_ink_mil.c: the normal-pooled MIL ink loss (inkmil.h). The host reference against central differences (soft max and hard
   max, label smoothing), then the GPU against the host (loss and gradient, float tolerance). */
#include "inkmil.h"
#include "nn.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)
static double urand(unsigned *s) { *s = *s * 1664525u + 1013904223u; return (*s >> 8) / 16777216.0; }
int main(void) {
    enum { D = 20, H = 22, W = 24, NR = 64 }; const size_t nv = (size_t)D * H * W; unsigned sd = 7;
    float *lg = malloc(nv * 4); for (size_t k = 0; k < nv; k++) lg[k] = (float)(3 * urand(&sd) - 1.5);
    inkray *R = calloc(NR, sizeof *R);
    for (int i = 0; i < NR; i++) {
        R[i].p[0] = (float)(2 + (D - 4) * urand(&sd)); R[i].p[1] = (float)(2 + (H - 4) * urand(&sd)); R[i].p[2] = (float)(2 + (W - 4) * urand(&sd));
        double n[3] = {urand(&sd) - .5, urand(&sd) - .5, urand(&sd) - .5}, l = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        for (int d = 0; d < 3; d++) R[i].n[d] = (float)(n[d] / l);
        R[i].y = urand(&sd) < 0.4; R[i].w = (float)(0.5 + urand(&sd));
    }
    const float Rr = 5, step = 0.5f, eps = 0.1f, sc = 1.0f / NR;
    for (int mode = 0; mode < 2; mode++) {
        const float tau = mode ? 0 : 0.5f;
        double *g = calloc(nv, sizeof *g); const double L = ink_mil_loss_host(lg, g, D, H, W, R, NR, Rr, step, tau, eps, sc);
        int nchk = 0, bad = 0; double emax = 0;
        for (int t = 0; t < 20000 && nchk < 60; t++) {
            const size_t k = (size_t)(urand(&sd) * nv); if (fabs(g[k]) < 1e-6) continue;
            const float o = lg[k], h = 1e-3f; lg[k] = o + h; const double Lp = ink_mil_loss_host(lg, nullptr, D, H, W, R, NR, Rr, step, tau, eps, sc);
            lg[k] = o - h; const double Lm = ink_mil_loss_host(lg, nullptr, D, H, W, R, NR, Rr, step, tau, eps, sc); lg[k] = o;
            const double fd = (Lp - Lm) / (2.0 * ((double)(o + h) - (double)o));   /* (h as float rounding) */
            const double e = fabs(fd - g[k]) / (fabs(g[k]) + 1e-4); if (e > emax) emax = e; bad += e > 2e-2; nchk++;
        }
        fprintf(stderr, "mil %s: loss %.6f, %d gradient voxels vs central differences, max rel err %.2e (%d off)\n", mode ? "hard max" : "soft max", L, nchk, emax, bad);
        CHECK(nchk >= 30 && bad <= (mode ? 2 : 0), "host gradient vs central differences (%s): %d of %d off", mode ? "hard" : "soft", bad, nchk);
        if (ink_mil_loss_hostmem) {
            float *gg = calloc(nv, 4); double Lg = 0;
            if (ink_mil_loss_hostmem(lg, gg, D, H, W, R, NR, Rr, step, tau, eps, sc, &Lg)) fprintf(stderr, "mil: no GPU (skipped the device comparison)\n");
            else {
                double em = 0, gm = 0; for (size_t k = 0; k < nv; k++) { const double e = fabs(gg[k] - g[k]); if (e > em) em = e; if (fabs(g[k]) > gm) gm = fabs(g[k]); }
                fprintf(stderr, "mil %s GPU: loss %.6f (host %.6f), max |grad err| %.2e of max |grad| %.2e\n", mode ? "hard max" : "soft max", Lg, L, em, gm);
                CHECK(fabs(Lg - L) < 1e-4 * fabs(L) + 1e-6 && em < 1e-3 * gm, "GPU vs host (%s)", mode ? "hard" : "soft");
            }
            free(gg);
        }
        free(g);
    }
    if (ink_mil_slab_hostmem && !getenv("UFSM_TEST_NO_GPU")) {   /* the slab form (train's two head passes) == the full-volume reference on the 16-bit logits */
        float *l16 = malloc(nv * 4); for (size_t k = 0; k < nv; k++) l16[k] = (float)(_Float16)lg[k];
        float *ga = malloc(nv * 4), *gb = malloc(nv * 4);
        for (int mode = 0; mode < 2; mode++) {
            const float tau = mode ? 0 : 0.5f;
            double *g = calloc(nv, sizeof *g); const double L = ink_mil_loss_host(l16, g, D, H, W, R, NR, Rr, step, tau, eps, sc);
            double gm = 0; for (size_t k = 0; k < nv; k++) if (fabs(g[k]) > gm) gm = fabs(g[k]);
            static const int ns[3] = {1, 3, 7};
            for (int s = 0; s < 3; s++) {
                double Ls = 0; if (ink_mil_slab_hostmem(lg, 0, D, H, W, R, NR, Rr, step, tau, eps, sc, ns[s], ga, &Ls)) { fprintf(stderr, "mil slabs: no GPU (skipped)\n"); break; }
                long off = 0; double em = 0; for (size_t k = 0; k < nv; k++) { const double e = fabs(ga[k] - g[k]); if (e > em) em = e; off += e > 2e-3 * fabs(g[k]) + 2e-3 * gm; }
                fprintf(stderr, "mil %s, %d slab(s): loss %.6f (full %.6f), max |grad err| %.2e of %.2e, %ld voxels off\n", mode ? "hard" : "soft", ns[s], Ls, L, em, gm, off);
                CHECK(fabs(Ls - L) < 1e-3 * fabs(L) && off <= (mode ? (long)(nv / 1000) : 0), "slab form (%s, %d slabs): loss %g vs %g, %ld off", mode ? "hard" : "soft", ns[s], Ls, L, off);
                if (ns[s] == 7) {   /* --det: the same bytes on a rerun */
                    double L2 = 0; ink_mil_slab_hostmem(lg, 0, D, H, W, R, NR, Rr, step, tau, eps, sc, 7, gb, &L2);
                    CHECK(!memcmp(ga, gb, nv * 4) && L2 == Ls, "slab form not deterministic (%s)", mode ? "hard" : "soft");
                    if (ink_mil_set_async) {   /* the plane-bound path (no per-slab sync) == the per-slab readback, byte for byte, at 1 / 3 / 7 slabs */
                        for (int a = 0; a < 3; a++) { const int nsl = a == 0 ? 1 : a == 1 ? 3 : 7; float *gc = malloc(nv * 4); double L3 = 0, L4 = 0;
                            ink_mil_set_async(0); ink_mil_slab_hostmem(lg, 0, D, H, W, R, NR, Rr, step, tau, eps, sc, nsl, gb, &L3);
                            ink_mil_set_async(1); const int rc = ink_mil_slab_hostmem(lg, 0, D, H, W, R, NR, Rr, step, tau, eps, sc, nsl, gc, &L4);
                            CHECK(!rc && !memcmp(gb, gc, nv * 4) && L3 == L4, "mil async plane bound != per-slab readback (%s, %d slabs)", mode ? "hard" : "soft", nsl);
                            free(gc); }
                        ink_mil_set_async(-1);
                    }
                    if (ink_mil_swap_hostmem && ink_mil_overflows && ink_mil_set_async && mode == 1) {   /* try6 (2026-11-14): the rays replaced between fin
                        and the backward slabs (shifted 6 planes in z: the bound undercounts) must not fail; the next step falls back to the readback */
                        inkray *R2 = malloc(NR * sizeof *R2); memcpy(R2, R, NR * sizeof *R2);
                        for (int i = 0; i < NR; i++) R2[i].p[0] = R[i].p[0] < D / 2 ? R[i].p[0] + 6 : R[i].p[0] - 6;
                        float *gs = malloc(nv * 4), *gr = malloc(nv * 4); double Lr = 0;
                        ink_mil_set_async(0); ink_mil_slab_hostmem(lg, 0, D, H, W, R2, NR, Rr, step, tau, eps, sc, 7, gr, &Lr);
                        const int o0 = ink_mil_overflows(); ink_mil_set_async(1);
                        const int rc = ink_mil_swap_hostmem(lg, D, H, W, R, R2, NR, Rr, step, tau, eps, sc, 7, gs);
                        fprintf(stderr, "mil rays swapped under the backward: rc %d, overflows %d, step 2 == readback: %s\n", rc, ink_mil_overflows() - o0, rc ? "-" : !memcmp(gs, gr, nv * 4) ? "yes" : "no");
                        CHECK(!rc && ink_mil_overflows() - o0 == 1 && !memcmp(gs, gr, nv * 4), "mil plane-bound overflow: no fallback to the readback (rc %d)", rc);
                        ink_mil_set_async(-1); free(R2); free(gs); free(gr);
                    }
                }
            }
            free(g);
        }
        {   /* bf16 logits / gradients (the --bf16 recipes): 3 slabs against the reference on bf16-rounded logits */
            for (size_t k = 0; k < nv; k++) l16[k] = (float)(__bf16)lg[k];
            double *g = calloc(nv, sizeof *g); const double L = ink_mil_loss_host(l16, g, D, H, W, R, NR, Rr, step, 0.5f, eps, sc), Ls0 = 0; double Ls = Ls0;
            double gm = 0; for (size_t k = 0; k < nv; k++) if (fabs(g[k]) > gm) gm = fabs(g[k]);
            if (!ink_mil_slab_hostmem(lg, 1, D, H, W, R, NR, Rr, step, 0.5f, eps, sc, 3, ga, &Ls)) {
                long off = 0; for (size_t k = 0; k < nv; k++) off += fabs(ga[k] - g[k]) > 1e-2 * fabs(g[k]) + 1e-2 * gm;
                fprintf(stderr, "mil soft bf16, 3 slabs: loss %.6f (full %.6f), %ld voxels off\n", Ls, L, off);
                CHECK(fabs(Ls - L) < 1e-3 * fabs(L) && !off, "slab form bf16: loss %g vs %g, %ld off", Ls, L, off);
            }
            free(g);
        }
        free(l16); free(ga); free(gb);
    }
    free(lg); free(R);
    if (!getenv("UFSM_TEST_NO_GPU") && nn_init(0) == 0) {   /* train --ink-smooth: nn_ink_targets_smooth == t (1 - eps) + eps / 2 (x255, rounded), eps 0 == nn_ink_targets */
        uint8_t in[256], t0[256], t1[256], m0[256], m1[256]; for (int i = 0; i < 256; i++) in[i] = (uint8_t)i;
        uint8_t *di = nn_malloc(256), *dt = nn_malloc(256), *dm = nn_malloc(256); nn_h2d(di, in, 256);
        nn_ink_targets(di, 256, dt, dm); nn_d2h(t0, dt, 256); nn_d2h(m0, dm, 256);
        nn_ink_targets_smooth(di, 256, dt, dm, 0.1f); nn_d2h(t1, dt, 256); nn_d2h(m1, dm, 256);
        int bad = 0; for (int i = 0; i < 256; i++) { const int e = i == 255 ? 0 : (int)lrintf((float)t0[i] * 0.9f + 12.75f); bad += t1[i] != e || m1[i] != m0[i]; }
        fprintf(stderr, "ink targets smoothed (eps 0.1): 0 -> %d, 254 -> %d, %d of 256 off\n", t1[0], t1[254], bad);
        CHECK(!bad && t0[254] == 255 && t1[0] == 13 && t1[254] == 242, "nn_ink_targets_smooth: %d off", bad);
        nn_free(di); nn_free(dt); nn_free(dm);
    }
    if (fails) { fprintf(stderr, "test_ink_mil: %d failures\n", fails); return 1; }
    fprintf(stderr, "test_ink_mil: ok\n"); return 0;
}
