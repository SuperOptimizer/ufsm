/* CPU test of src/vwt_aug.c (the vwt ray sampler's deformation / intensity aug) on an analytic scroll: concentric sheets about the
   z axis (winding W = r / 12 voxels, papyrus where frac(W) < 0.3). Slabs as villa's frame (radial ray from the axis). Checks: the map
   has det > 0, the winding read at the mapped points rises along +k (the ph convention), the 2-voxel lattice matches the exact map,
   vwa_box holds every source point, crushed slabs pull more source along the ray (compaction), determinism, --vaug 0 = identity, and
   the intensity ops keep va 0 voxels 0 and the rest >= 1. */
#include "vwt_aug.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
static void check(const char *what, int ok) { printf("vwt_aug: %-100s %s\n", what, ok ? "ok" : "FAIL"); fails += !ok; }
static const double AX = 600, AY = 600, SP = 12, RMAX = 560;
static double wind(const double p[3]) { return hypot(p[0] - AX, p[1] - AY) / SP; }
static double ct_at(void *ud, const double p[3]) {
    (void)ud; const double r = hypot(p[0] - AX, p[1] - AY); if (r > RMAX || p[2] < 0 || p[2] > 1200) return 0;
    const double w = r / SP; return w - floor(w) < .3 ? 150 : 45;
}
static void cross3(const double *u, const double *v, double *w) { w[0] = u[1] * v[2] - u[2] * v[1]; w[1] = u[2] * v[0] - u[0] * v[2]; w[2] = u[0] * v[1] - u[1] * v[0]; }
static void frame(vws_sample *s, const double dir[3], const double c[3]) {   /* villa frame_of; c the slab centre */
    const double n = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]); for (int e = 0; e < 3; e++) s->d[e] = dir[e] / n;
    double ref[3] = {0, 0, 0}; int m = 0; for (int e = 1; e < 3; e++) if (fabs(s->d[e]) < fabs(s->d[m])) m = e; ref[m] = 1;
    cross3(ref, s->d, s->a); const double na = sqrt(s->a[0] * s->a[0] + s->a[1] * s->a[1] + s->a[2] * s->a[2]); for (int e = 0; e < 3; e++) s->a[e] /= na;
    cross3(s->d, s->a, s->b);
    for (int e = 0; e < 3; e++) s->o[e] = c[e] - .5 * (s->T - 1) * (s->a[e] + s->b[e]) - .5 * (s->L - 1) * s->d[e];
}
static uint64_t lcg(uint64_t *s) { *s = *s * 6364136223846793005ull + 1442695040888963407ull; return *s >> 11; }
static double ur(uint64_t *s) { return (double)lcg(s) * 0x1.0p-53; }

int main(void) {
    const int T = 32, L = 96;
    vwa_cfg cfg = vwa_cfg_default(); cfg.geo_p = 1;
    int det_ok = 1, mono_ok = 1, box_ok = 1, lat_ok = 1, det2 = 1, nmaps = 0, ncrush = 0; double lbad = 0, worst_lat = 0, lat_sum = 0, bad_steps = 0, steps = 0, comp = 0;
    float *q = malloc(sizeof(float) * 3 * T * T * L), *q2 = malloc(sizeof(float) * 3 * T * T * L);
    uint64_t rs = 7;
    for (int t = 0; t < 24; t++) {
        vws_sample s = {.T = T, .L = L};
        const double th = 2 * M_PI * ur(&rs), r = 150 + 300 * ur(&rs), tilt = (ur(&rs) - .5) * .4;
        const double c[3] = {AX + r * cos(th), AY + r * sin(th), 300 + 600 * ur(&rs)}, dir[3] = {cos(th), sin(th), tilt};
        frame(&s, dir, c);
        vwa_cfg cc = cfg; if (t % 3 == 0) cc.df.elastic_p = 0; if (t % 3 == 1) cc.df.crush_p = 0;   /* crush only / elastic only / both drawn */
        vwa *w = vwa_begin(&cc, 1000 + t, &s, ct_at, nullptr);
        if (!w) continue;
        nmaps++;
        vwa_src_lattice(w, T, L, q);
        { vwa *w2 = vwa_begin(&cc, 1000 + t, &s, ct_at, nullptr); vwa_src_lattice(w2, T, L, q2); det2 &= !memcmp(q, q2, sizeof(float) * 3 * T * T * L); vwa_end(w2); }
        double lo[3], hi[3]; vwa_box(w, T, L, lo, hi);
        for (int i = 0; i < T; i += 3) for (int j = 0; j < T; j += 3) for (int k = 0; k < L; k++) {
            double p[3], x[3]; for (int e = 0; e < 3; e++) p[e] = s.o[e] + i * s.a[e] + j * s.b[e] + k * s.d[e];
            vwa_src(w, p, x);
            const float *ql = q + 3 * (((size_t)i * T + j) * L + k);
            const double dl = fabs(ql[0] - x[0]) + fabs(ql[1] - x[1]) + fabs(ql[2] - x[2]); worst_lat = fmax(worst_lat, dl); lat_sum += dl;
            for (int e = 0; e < 3; e++) box_ok &= x[e] >= lo[e] && x[e] <= hi[e];
            if (k % 8 == 0) det_ok &= vwa_jdet(w, p) > 0;
            if (k > 0) { const double xl[3] = {ql[0], ql[1], ql[2]}, xlp[3] = {ql[-3], ql[-2], ql[-1]}; lbad += wind(xl) <= wind(xlp); }
            if (k > 0) { double pp[3], xp[3]; for (int e = 0; e < 3; e++) pp[e] = p[e] - s.d[e]; vwa_src(w, pp, xp); bad_steps += wind(x) <= wind(xp); steps++; }
        }
        if (t % 3 == 0) {   /* crush only: the ray's ends pull from farther apart (gaps collapsed) */
            const int i = T / 2, j = T / 2; const float *a = q + 3 * (((size_t)i * T + j) * L), *b = q + 3 * (((size_t)i * T + j) * L + L - 1);
            comp += hypot(hypot(b[0] - a[0], b[1] - a[1]), b[2] - a[2]) / (L - 1); ncrush++;
        }
        vwa_end(w);
    }
    const double lat_mean = lat_sum / (nmaps * (T / 3 + 1) * (T / 3 + 1) * L);
    lat_ok = lat_mean < .3 && lbad / steps < .01;   /* (CT and labels both read the lattice q: they stay consistent; the lattice map only has
                                                       to stay a monotone map near the exact one: errors concentrate in collapsed voids, F' ~ 1 / j) */
    mono_ok = bad_steps / steps < .01;
    printf("vwt_aug: %d maps; lattice vs exact worst %.3f mean %.3f vx (lattice steps not rising %.4f); winding steps not rising %.4f; crushed rays pull %.2f x their length\n", nmaps, worst_lat, lat_mean, lbad / steps, bad_steps / steps, ncrush ? comp / ncrush : 0);
    check("geometric aug drawn on every slab at --vaug-geo 1", nmaps == 24);
    check("Jacobian det > 0", det_ok);
    check("the winding at the mapped points rises along +k (>= 99% of steps)", mono_ok);
    check("2-voxel lattice near the exact map (mean < 0.3 vx) and monotone along +k (>= 99% of steps)", lat_ok);
    check("vwa_box holds every source point", box_ok);
    check("deterministic in the seed", det2);
    check("crushed slabs compact the sheets (rays pull > 1.3x their length)", ncrush && comp / ncrush > 1.3);
    { vws_sample s = {.T = T, .L = L}; const double c[3] = {AX + 300, AY, 600}, dir[3] = {1, 0, 0}; frame(&s, dir, c);
      vwa_cfg off = cfg; off.on = 0; check("--vaug 0: no map", vwa_begin(&off, 1, &s, ct_at, nullptr) == nullptr);
      vwa_cfg g0 = vwa_cfg_default(); g0.geo_p = 0; check("--vaug-geo 0: no map", vwa_begin(&g0, 1, &s, ct_at, nullptr) == nullptr); }
    {   /* intensity */
        const size_t n = (size_t)T * T * L; uint8_t *im = malloc(n), *va = malloc(n), *im2 = malloc(n); int ok = 1, changed = 0;
        for (int t = 0; t < 12; t++) {
            for (size_t i = 0; i < n; i++) { va[i] = (i / 97) % 5 != 0; im[i] = va[i] ? (uint8_t)(40 + (i * 7919) % 180) : 0; }
            memcpy(im2, im, n);
            vwa_cfg c = vwa_cfg_default(); vwa_intensity(&c, 50 + t, im, va, T, L);
            uint8_t *im3 = malloc(n); memcpy(im3, im2, n); vwa_intensity(&c, 50 + t, im3, va, T, L); ok &= !memcmp(im, im3, n); free(im3);
            for (size_t i = 0; i < n; i++) { ok &= va[i] ? im[i] >= 1 : im[i] == 0; changed |= im[i] != im2[i]; }
        }
        check("intensity: va 0 stays 0, inside >= 1, deterministic, ops change im", ok && changed);
        free(im); free(va); free(im2);
    }
    free(q); free(q2);
    printf("vwt_aug: %s\n", fails ? "FAILED" : "all ok");
    return fails != 0;
}
