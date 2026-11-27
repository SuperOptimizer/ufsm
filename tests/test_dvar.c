/* train --density-nll / --density-min-gap-um and predict --fit-dv (2026-11-27, densvar):
   1. nn_dvar_stats against host sums (supervised count, NLL, sum e^-(1-beta)s, sum s) over a mask with weights 0..8, a gap threshold
      and log-variances beyond the [-12, 6] clamp;
   2. nn_dvar_grad against the host formula (16-bit, the gradient scale) and a central difference of the host beta-NLL (the e^(beta s)
      factor held, as the detached factor of the kernel), the mean factors a / mean(a);
   3. nn_dvar_scale: the density gradient times those factors;
   4. nn_set_loss_dgap: the 7-plane slab loss with the gap rule equals, on planes 0..5, the run without it (bit for bit) and on plane 6
      the 1-plane loss with those voxels masked (tolerance: other partial-sum order and the 1 / active factor); the off setting
      (-1, 0) is bit-identical to never setting it;
   5. nn_fit_accum_dv + nn_fit_finalize_dv against the host pooled mean s over material voxels. */
#include "nn.h"
#include "fit.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)
static unsigned rs = 12345u;
static unsigned rnd(void) { rs = rs * 1103515245u + 12345u; return rs >> 8; }
static float h2f(uint16_t h) { _Float16 v; memcpy(&v, &h, 2); return (float)v; }
static uint16_t f2h(float f) { _Float16 v = (_Float16)f; uint16_t h; memcpy(&h, &v, 2); return h; }
static double sig(double x) { return 1 / (1 + exp(-x)); }
static double clampd(double x) { return x < -12 ? -12 : x > 6 ? 6 : x; }

int main(void) {
    if (nn_init(0)) { fprintf(stderr, "test_dvar: no GPU\n"); return 1; }
    nn_set_f16(1); nn_set_logits_h16(1); nn_set_loss_grad_h16(1);
    const float gs = 1024.f; nn_set_grad_scale(gs);
    const size_t n = 3 * 4096 + 17; const int thr = 200; const float beta = 0.5f, lam = 0.7f;
    uint16_t *zd = malloc(n * 2), *zs = malloc(n * 2), *g = malloc(n * 2), *af = malloc(n * 2), *gd = malloc(n * 2), *gd2 = malloc(n * 2);
    uint8_t *t = malloc(n), *m = malloc(n);
    for (size_t i = 0; i < n; i++) {
        zd[i] = f2h((rnd() % 4001) / 500.f - 4.f); t[i] = (uint8_t)(rnd() & 255); m[i] = (uint8_t)(rnd() % 9 < 3 ? 0 : rnd() % 9);
        const unsigned k = rnd() % 50; zs[i] = f2h(k == 0 ? -13.f : k == 1 ? 7.f : (rnd() % 1601) / 100.f - 11.f);
        gd[i] = f2h(((rnd() % 2001) / 1000.f - 1.f) * 0.01f);
    }
    double hs[4] = {0, 0, 0, 0};
    for (size_t i = 0; i < n; i++) {
        if (!m[i] || t[i] >= thr) continue;
        const double s = clampd(h2f(zs[i])), r = sig(h2f(zd[i])) - t[i] / 255.0;
        hs[0] += 1; hs[1] += 0.5 * (exp(-s) * r * r + s); hs[2] += exp(-(1 - beta) * s); hs[3] += s;
    }
    void *dzd = nn_malloc(n * 2), *dzs = nn_malloc(n * 2), *daf = nn_malloc(n * 2), *dgd = nn_malloc(n * 2); uint8_t *dt = nn_malloc(n), *dm = nn_malloc(n); double *dacc = nn_malloc(4 * sizeof(double));
    nn_h2d(dzd, zd, n * 2); nn_h2d(dzs, zs, n * 2); nn_h2d(dt, t, n); nn_h2d(dm, m, n); nn_h2d(dgd, gd, n * 2); nn_zero(dacc, 4 * sizeof(double));
    nn_dvar_stats(dzd, dzs, dt, dm, n, thr, beta, dacc); double da[4]; nn_sync(); nn_d2h(da, dacc, sizeof da);
    {   double e = 0; for (int k = 0; k < 4; k++) e = fmax(e, fabs(da[k] - hs[k]) / fmax(1, fabs(hs[k])));
        CHECK(da[0] == hs[0] && e < 1e-5, "dvar stats: device %.0f %.6f %.6f %.6f host %.0f %.6f %.6f %.6f (rel %.2g)", da[0], da[1], da[2], da[3], hs[0], hs[1], hs[2], hs[3], e);
        printf("dvar stats: %.0f supervised of %zu, mean NLL %.4f, mean s %.3f (rel err %.2g)\n", hs[0], n, hs[1] / hs[0], hs[3] / hs[0], e); }
    nn_dvar_grad(dzd, dzs, dt, dm, n, thr, beta, lam, dacc, daf); nn_sync(); nn_d2h(g, dzs, n * 2); nn_d2h(af, daf, n * 2);
    {   /* host gradient; central difference of F(s) = lam / cnt sum w_i 0.5 (e^-s r^2 + s), w_i = e^(beta s_i) held */
        size_t bad = 0, badc = 0, bada = 0, nz = 0; double emax = 0, ecd = 0; const double abar = hs[2] / hs[0];
        for (size_t i = 0; i < n; i++) {
            const double x = h2f(zs[i]); double ref = 0, aref = 1;
            if (m[i] && t[i] < thr) {
                const double s = clampd(x), r = sig(h2f(zd[i])) - t[i] / 255.0, w = exp(beta * s);
                if (x > -12 && x < 6) {
                    ref = lam / hs[0] * 0.5 * (1 - exp(-s) * r * r) * w;
                    const double h = 1e-3, F = lam / hs[0] * 0.5 * w; const double cd = F * ((exp(-(s + h)) * r * r + s + h) - (exp(-(s - h)) * r * r + s - h)) / (2 * h);
                    const double ec = fabs(cd - ref) / fmax(1e-12, fabs(ref)); if (fabs(ref) > 1e-9) { ecd = fmax(ecd, ec); badc += ec > 1e-4; }
                }
                aref = exp(-(1 - beta) * s) / abar;
            }
            const double dv = h2f(g[i]) / gs, e = fabs(dv - ref); nz += ref != 0;
            const double tol = fabs(ref) * 2e-3 + 1e-7 / gs; emax = fmax(emax, e / fmax(1e-12, fabs(ref))); bad += e > tol;
            bada += fabs(h2f(af[i]) - aref) > fabs(aref) * 2e-3 + 1e-6;
        }
        CHECK(bad == 0 && badc == 0 && bada == 0 && nz > 100, "dvar grad: %zu voxels off the host formula (max rel %.3g), %zu off the central difference (max rel %.3g), %zu mean factors off; %zu nonzero", bad, emax, badc, ecd, bada, nz);
        printf("dvar grad: %zu nonzero gradients match the host (max rel %.2g) and the central difference (max rel %.2g); mean factors match\n", nz, emax, ecd);
    }
    nn_dvar_scale(dgd, daf, n); nn_sync(); nn_d2h(gd2, dgd, n * 2);
    {   size_t bad = 0; for (size_t i = 0; i < n; i++) { const double ref = (double)h2f(gd[i]) * h2f(af[i]); bad += fabs(h2f(gd2[i]) - ref) > fabs(ref) * 1e-3 + 6e-8; }   /* (+ one fp16 subnormal step, 2^-24: small products round there) */
        CHECK(bad == 0, "dvar scale: %zu voxels off", bad); }
    nn_free(dzd); nn_free(dzs); nn_free(daf); nn_free(dgd); nn_free(dt); nn_free(dm); nn_free(dacc);

    {   /* 4. the gap rule in the slab loss (stats_acc_e + fin + grad_only_s, the head slabs' calls) */
        const int Q = 20; const size_t q3 = (size_t)Q * Q * Q; const shape5 s7 = {1, 7, Q, Q, Q}, s1 = {1, 1, Q, Q, Q};
        uint16_t *lg = malloc(7 * q3 * 2), *ga = malloc(7 * q3 * 2), *gb = malloc(7 * q3 * 2), *gc = malloc(q3 * 2);
        uint8_t *tt = malloc(7 * q3), *mm = malloc(q3), *m6 = malloc(q3), w7[7] = {1, 1, 1, 1, 1, 1, 1};
        for (size_t i = 0; i < 7 * q3; i++) { lg[i] = f2h((rnd() % 2001) / 500.f - 2.f); tt[i] = (uint8_t)(rnd() & 255); }
        for (size_t i = 0; i < q3; i++) { mm[i] = (uint8_t)(rnd() % 9); m6[i] = tt[6 * q3 + i] >= thr ? 0 : mm[i]; }
        void *dl = nn_malloc(7 * q3 * 2), *dg = nn_malloc(7 * q3 * 2); uint8_t *dt7 = nn_malloc(7 * q3), *dm7 = nn_malloc(q3), *dw = nn_malloc(7);
        float *sc = nn_malloc(nn_loss_scratch(s7)); double *ds = nn_malloc(5 * 7 * sizeof(double)); double *ent = nn_malloc(7 * sizeof(double));
        nn_h2d(dt7, tt, 7 * q3); nn_h2d(dm7, mm, q3); nn_h2d(dw, w7, 7);
        float fo[16], fg[16], f1[4];
#define RUN7(C, T, OUT, FIN) do { nn_h2d(dl, lg, 7 * q3 * 2); nn_zero(ds, 5 * 7 * sizeof(double)); nn_zero(ent, 7 * sizeof(double)); nn_set_loss_dgap(C, T); \
        nn_loss_stats_acc_e(dl, dt7, dm7, dw, s7, ds, ent); nn_loss_stats_fin(ds, dw, s7, sc); nn_loss_grad_only_s(dl, dt7, dm7, dw, s7, 0.f, dl, sc, 1.f); nn_set_loss_dgap(-1, 0); \
        nn_sync(); nn_d2h(OUT, dl, 7 * q3 * 2); nn_loss_fetch(sc, s7, FIN); } while (0)
        uint16_t *g0 = malloc(7 * q3 * 2);
        {   /* never set: the reference of the off setting */
            nn_h2d(dl, lg, 7 * q3 * 2); nn_zero(ds, 5 * 7 * sizeof(double)); nn_zero(ent, 7 * sizeof(double));
            nn_loss_stats_acc_e(dl, dt7, dm7, dw, s7, ds, ent); nn_loss_stats_fin(ds, dw, s7, sc); nn_loss_grad_only_s(dl, dt7, dm7, dw, s7, 0.f, dl, sc, 1.f);
            nn_sync(); nn_d2h(g0, dl, 7 * q3 * 2);
        }
        RUN7(-1, 0, ga, fo); RUN7(6, thr, gb, fg);
        size_t off = 0, b05 = 0, d6 = 0, b6 = 0; for (size_t i = 0; i < 7 * q3; i++) off += g0[i] != ga[i];
        for (size_t i = 0; i < 6 * q3; i++) b05 += ga[i] != gb[i];
        for (size_t i = 6 * q3; i < 7 * q3; i++) d6 += ga[i] != gb[i];
        CHECK(off == 0, "dgap off: %zu gradients differ from the unset run", off);
        CHECK(b05 == 0 && d6 > 0, "dgap: planes 0..5 changed at %zu voxels, plane 6 at %zu", b05, d6);
        {   /* plane 6 against the 1-plane loss with the gap voxels masked: g7 = g1 / active (7 active planes) */
            uint8_t *dm1 = nn_malloc(q3); float *sc1 = nn_malloc(nn_loss_scratch(s1)); double *ds1 = nn_malloc(5 * sizeof(double)), *en1 = nn_malloc(sizeof(double));
            nn_h2d(dm1, m6, q3); nn_h2d(dl, lg + 6 * q3, q3 * 2); nn_zero(ds1, 5 * sizeof(double)); nn_zero(en1, sizeof(double));
            nn_loss_stats_acc_e(dl, dt7 + 6 * q3, dm1, dw, s1, ds1, en1); nn_loss_stats_fin(ds1, dw, s1, sc1); nn_loss_grad_only_s(dl, dt7 + 6 * q3, dm1, dw, s1, 0.f, dl, sc1, 1.f);
            nn_sync(); nn_d2h(gc, dl, q3 * 2); nn_loss_fetch(sc1, s1, f1);
            for (size_t i = 0; i < q3; i++) { const double a = h2f(gb[6 * q3 + i]), r = h2f(gc[i]) / 7.0; b6 += fabs(a - r) > fabs(r) * 2e-3 + 1e-7; }
            CHECK(b6 == 0 && fabs(fg[6] - f1[0]) < 1e-5 * fmax(1, fabs(f1[0])), "dgap: plane 6 differs from the masked 1-plane loss at %zu voxels (bce %.7f vs %.7f)", b6, fg[6], f1[0]);
            printf("dgap: planes 0..5 unchanged, plane 6 = the loss with codes >= %d masked (bce %.4f -> %.4f); off = unset bit for bit\n", thr, fo[6], fg[6]);
            nn_free(dm1); nn_free(sc1); nn_free(ds1); nn_free(en1);
        }
        nn_free(dl); nn_free(dg); nn_free(dt7); nn_free(dm7); nn_free(dw); nn_free(sc); nn_free(ds); nn_free(ent);
        free(lg); free(ga); free(gb); free(gc); free(g0); free(tt); free(mm); free(m6);
    }

    {   /* 5. the dv fit plane: a 16^3 window, halo 2, box 16^3 at h 4, 2 channels (fp32 logits), the log-variance in channel 1 */
        nn_set_logits_h16(0);
        const int D = 16, W = 16, h = 4, halo = 2, box[3] = {16, 16, 16}, F[3] = {4, 4, 4}; const size_t w3 = (size_t)D * W * W, nf = 64;
        float *lgf = malloc(2 * w3 * 4), *acc = calloc(13 * nf, 4); uint8_t *ct = malloc(w3), o[64], oh[64];
        for (size_t i = 0; i < 2 * w3; i++) lgf[i] = (rnd() % 2001) / 100.f - 13.f;
        for (size_t i = 0; i < w3; i++) ct[i] = rnd() % 4 ? (uint8_t)(1 + rnd() % 200) : 0;
        double sm[64] = {0}, mt[64] = {0};
        for (int z = halo; z < D - halo; z++) for (int y = halo; y < W - halo; y++) for (int x = halo; x < W - halo; x++) {
            const size_t i = ((size_t)z * W + y) * W + x; if (!ct[i]) continue;
            const int c = (z / h * 4 + y / h) * 4 + x / h; sm[c] += clampd(lgf[w3 + i]); mt[c] += 1;
        }
        for (int c = 0; c < 64; c++) { const long v = mt[c] > 0 ? lround((sm[c] / mt[c] + 12) * 255 / 18) : 0; oh[c] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); acc[nf + c] = (float)mt[c]; }
        float *dlg = nn_malloc(2 * w3 * 4), *dac = nn_malloc(13 * nf * 4); uint8_t *dct = nn_malloc(w3), *dout = nn_malloc(nf);
        nn_h2d(dlg, lgf, 2 * w3 * 4); nn_h2d(dct, ct, w3); nn_h2d(dac, acc, 13 * nf * 4);   /* (plane 1: the host material counts, as nn_fit_accum leaves them) */
        nn_fit_accum_dv(dlg, dct, nullptr, D, W, halo, halo, 0, 0, 0, box, h, F, 1, dac + 12 * nf); nn_fit_finalize_dv(dac, nf, 12, dout); nn_sync(); nn_d2h(o, dout, nf);
        int bad = 0; for (int c = 0; c < 64; c++) bad += abs((int)o[c] - (int)oh[c]) > 1;   /* (one code: fp32 against fp64 sums) */
        CHECK(bad == 0, "fit dv: %d of 64 cells off the host mean s", bad);
        printf("fit dv: 64 cells match the host's pooled mean log-variance\n");
        nn_free(dlg); nn_free(dac); nn_free(dct); nn_free(dout); free(lgf); free(acc); free(ct);
    }
    free(zd); free(zs); free(g); free(af); free(gd); free(gd2); free(t); free(m);
    if (fails) { fprintf(stderr, "test_dvar: %d failure(s)\n", fails); return 1; }
    printf("test_dvar: OK\n");
    return 0;
}
