/* wind_assign (src/wind.h) on a synthetic Archimedean spiral (radius = R0 + P q, angle 2 pi q around a fixed axis; scan voxels,
   reference codes on a label grid at half the resolution): the reference holds the wraps for q in [1, 5] as raster --value codes.
     A  re-traces q in [2.3, 3.6], 1.5 scan voxels off the sheet          -> duplicate, q right (and all dropped with drop_dup)
     B  q in [4.5, 6.5]: overlaps the reference only up to 5               -> new, q right also where it leaves the reference
     C  q in [2, 2.5] then one pitch outward (the next wrap) for [2.5, 3]  -> conflict, the switched half dropped
     D  q in [6, 7]: nowhere near the reference                            -> unassigned
     E  A with its rows / cols mirrored (another grid orientation)         -> duplicate, same q
   and the reference read as a dense winding raster (every voxel its q): D assigned, new. */
#include "wind.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

static const double CY = 340, CX = 330, R0 = 60, P = 40;   /* scan voxels; the label grid is 2 scan voxels per voxel */
enum { LZ = 50, LY = 340, LX = 340 };
static uint8_t *ref; static int ref_kind;

static uint8_t code_of(double q) { long k = lround(q * 18.0) % 252; if (k < 0) k += 252; return (uint8_t)(1 + k); }
static void axis_fn(void *ud, double z, double *cy, double *cx) { (void)ud; (void)z; *cy = CY; *cx = CX; }
static void axis_off(void *ud, double z, double *cy, double *cx) { (void)ud; (void)z; *cy = CY; *cx = CX + 50; }
static void lookup(void *ud, const float *pts, size_t n, float tol, float *dist, float *qref) {
    (void)ud; const int R = (int)ceilf(tol);
    for (size_t i = 0; i < n; i++) {
        dist[i] = INFINITY; qref[i] = NAN;
        const int z0 = (int)lroundf(pts[3 * i]), y0 = (int)lroundf(pts[3 * i + 1]), x0 = (int)lroundf(pts[3 * i + 2]);
        for (int z = z0 - R; z <= z0 + R; z++) for (int y = y0 - R; y <= y0 + R; y++) for (int x = x0 - R; x <= x0 + R; x++) {
            if (z < 0 || z >= LZ || y < 0 || y >= LY || x < 0 || x >= LX) continue;
            const float q = wind_code_q(ref[((size_t)z * LY + y) * LX + x], ref_kind);
            if (isnan(q)) continue;
            const float d = sqrtf((z - pts[3 * i]) * (z - pts[3 * i]) + (y - pts[3 * i + 1]) * (y - pts[3 * i + 1]) + (x - pts[3 * i + 2]) * (x - pts[3 * i + 2]));
            if (d < dist[i]) { dist[i] = d; qref[i] = q; }
        }
    }
}

typedef struct { int w, h; float *xyz; uint8_t *valid; double *qtrue; } surf;
/* rows: scan z in [z0, z1]; cols: q in [qa, qb]; dq(col) shifts the sheet (C: + 1 for the outer half); off: radial offset */
static surf make(double qa, double qb, double z0, double z1, double off, int switch_half, int mirror) {
    surf s; s.w = 160; s.h = 24;
    s.xyz = malloc((size_t)s.w * s.h * 3 * sizeof(float)); s.valid = malloc((size_t)s.w * s.h); s.qtrue = malloc((size_t)s.w * s.h * sizeof(double));
    for (int r = 0; r < s.h; r++) for (int c = 0; c < s.w; c++) {
        const int rr = mirror ? s.h - 1 - r : r, cc = mirror ? s.w - 1 - c : c;
        const double q = qa + (qb - qa) * c / (s.w - 1), qs = q + (switch_half && c >= s.w / 2 ? 1.0 : 0.0), z = z0 + (z1 - z0) * r / (s.h - 1);
        const double rho = R0 + P * qs + off, th = 2 * M_PI * q;
        const size_t i = (size_t)rr * s.w + cc;
        s.xyz[i * 3] = (float)(CX + rho * cos(th)); s.xyz[i * 3 + 1] = (float)(CY + rho * sin(th)); s.xyz[i * 3 + 2] = (float)z;
        s.valid[i] = 1; s.qtrue[i] = qs;
    }
    return s;
}

/* wind_seed_decide fixtures: a surface of the spiral (radius R0 + P q, angle hand * 2 pi q) with `cpt` columns per turn (every
   azimuth bin met), rows z0 .. z1 */
static surf make2(double qa, double qb, double z0, double z1, int cpt, int hand, int mirror) {
    surf s; s.w = (int)ceil((qb - qa) * cpt) + 1; s.h = 6;
    s.xyz = malloc((size_t)s.w * s.h * 3 * sizeof(float)); s.valid = malloc((size_t)s.w * s.h); s.qtrue = malloc((size_t)s.w * s.h * sizeof(double));
    for (int r = 0; r < s.h; r++) for (int c = 0; c < s.w; c++) {
        const int cc = mirror ? s.w - 1 - c : c;
        const double q = qa + (qb - qa) * c / (s.w - 1), z = z0 + (z1 - z0) * r / (s.h - 1), rho = R0 + P * q, th = hand * 2 * M_PI * q;
        const size_t i = (size_t)r * s.w + cc;
        s.xyz[i * 3] = (float)(CX + rho * cos(th)); s.xyz[i * 3 + 1] = (float)(CY + rho * sin(th)); s.xyz[i * 3 + 2] = (float)z;
        s.valid[i] = 1; s.qtrue[i] = q;
    }
    return s;
}
static void axis_left(void *ud, double z, double *cy, double *cx) { (void)ud; (void)z; *cy = CY; *cx = CX - 50; }
/* wind-seed's three passes on a set (main.c cmd_wind_seed): infos, q under both senses -> wind_seed_decide, q under the choice.
   qd[k]: the written q minus the true q, its median and spread (max - min) per surface (NaN dropped) */
static void seed_set(surf *S, const double *K, int n, wind_choice *ch, wind_surf_vote *pv, int *old_sense, double *qmed, double *qspread) {
    wind_seed_info *i1 = calloc(n, sizeof *i1); wind_pt *pts = nullptr; size_t np = 0;
    for (int k = 0; k < n; k++) { wind_result r; CHECK(!wind_seed(S[k].xyz, S[k].valid, S[k].w, S[k].h, axis_left, nullptr, K[k], 0, NAN, &i1[k], &r), "seed pass 1"); wind_result_free(&r); }
    double phis[2], res, phi_v, vote[2]; wind_seed_set(i1, n, old_sense, &phi_v, vote, &res);
    for (int si = 0; si < 2; si++) wind_seed_seam(i1, n, si ? -1 : 1, &phis[si], &res);
    for (int k = 0; k < n; k++) {
        wind_result r[2]; wind_seed_info in;
        for (int si = 0; si < 2; si++) CHECK(!wind_seed(S[k].xyz, S[k].valid, S[k].w, S[k].h, axis_left, nullptr, K[k], si ? -1 : 1, phis[si], &in, &r[si]), "seed pass 1b");
        size_t m = 0; wind_pt *p = wind_seed_points(S[k].xyz, S[k].valid, S[k].w, S[k].h, axis_left, nullptr, r[0].q, r[1].q, k, 400000, &m);
        pts = realloc(pts, (np + m) * sizeof *pts); memcpy(pts + np, p, m * sizeof *p); np += m; free(p);
        wind_result_free(&r[0]); wind_result_free(&r[1]);
    }
    CHECK(!wind_seed_decide(pts, np, K, n, 1, 0, 0, *old_sense, 50, ch, pv), "wind_seed_decide");
    for (int k = 0; k < n; k++) {
        qmed[k] = qspread[k] = NAN; if (pv[k].dropped) continue;
        wind_result r; wind_seed_info in;
        CHECK(!wind_seed(S[k].xyz, S[k].valid, S[k].w, S[k].h, axis_left, nullptr, ch->order * K[k], ch->sense, phis[ch->sense > 0 ? 0 : 1], &in, &r), "seed pass 2");
        double lo = INFINITY, hi = -INFINITY, sum = 0; size_t m = 0;
        for (size_t i = 0; i < (size_t)S[k].w * S[k].h; i++) if (!isnan(r.q[i])) { const double d = r.q[i] - S[k].qtrue[i]; lo = fmin(lo, d); hi = fmax(hi, d); sum += d; m++; }
        qmed[k] = m ? sum / (double)m : NAN; qspread[k] = m ? hi - lo : NAN;
        wind_result_free(&r);
    }
    free(i1); free(pts);
}
static double qerr(double a, double b) { double d = fmod(a - b, 14.0); if (d < -7) d += 14; if (d > 7) d -= 14; return fabs(d); }

static wind_result run(const surf *s, int drop_dup, const char *name, int expect, int check_q) {
    wind_result r; wind_params p = {0.5, 3.f, 4, 0, drop_dup, ref_kind == WIND_WINDING};
    CHECK(!wind_assign(s->xyz, s->valid, s->w, s->h, axis_fn, nullptr, lookup, nullptr, p, &r), "%s: wind_assign failed", name);
    const wind_comp *c = &r.comp[0];
    size_t nq = 0, nbad = 0; double worst = 0;
    for (size_t i = 0; i < (size_t)s->w * s->h; i++) if (!isnan(r.q[i])) { nq++; const double e = qerr(r.q[i], s->qtrue[i]); if (e > worst) worst = e; nbad += e > 0.06; }
    printf("%-14s comp %d: %-10s dir %+d K %2d  samples %zu matched %zu agree %zu  median dist %.2f  |resid| %.3f  q on %zu vertices, worst err %.3f turn\n",
           name, r.ncomp, wind_verdict_name(c->verdict), c->dir, c->K, c->nsample, c->nmatch, c->nagree, c->median_dist, c->mean_abs_resid, nq, worst);
    CHECK(r.ncomp == 1 && c->verdict == expect, "%s: verdict %s, expected %s", name, wind_verdict_name(c->verdict), wind_verdict_name(expect));
    if (check_q) CHECK(nq > 0 && !nbad, "%s: %zu of %zu assigned vertices more than 0.06 turn off", name, nbad, nq);
    if (expect != WA_UNASSIGNED) CHECK(c->dir == 1, "%s: dir %d", name, c->dir);
    return r;
}

int main(void) {
    {   /* fit --winding-mod14: W -> dense winding byte -> wind_code_q (WIND_WINDING) gives W mod 14 to 1/36 turn; unknown -> 255 -> NaN */
        int bad = 0;
        for (double W = -40; W <= 40; W += 0.013) {
            const uint8_t c = wind_dense_code(W); const float q = wind_code_q(c, WIND_WINDING);
            double d = fmod(q - W, 14.0); if (d < -7) d += 14; if (d > 7) d -= 14;
            bad += c > 251 || !(fabs(d) <= 1.0 / 36 + 1e-6);
        }
        CHECK(!bad && wind_dense_code(NAN) == 255 && wind_code_q(255, WIND_WINDING) != wind_code_q(255, WIND_WINDING) && wind_dense_code(14) == 0 && wind_dense_code(-1.0 / 18) == 251,
              "wind_dense_code: %d codes off", bad);
    }
    {   /* fit --winding-mod14 3: W -> stored-phase code round(252 frac(sign W)) (PHASE_* 0..251, sheet at 0); NaN / huge -> 255 */
        int bad = 0;
        for (double W = -40; W <= 40; W += 0.0071) for (int sg = -1; sg <= 1; sg += 2) {
            const uint8_t c = fit_phase_code(W, sg); const double v = sg * W, f = v - floor(v);
            double d = c / 252.0 - f; d -= round(d);
            bad += c > 251 || !(fabs(d) <= 0.5 / 252 + 1e-9);
        }
        CHECK(!bad && fit_phase_code(NAN, 1) == 255 && fit_phase_code(1e20, 1) == 255 && fit_phase_code(3, 1) == 0 && fit_phase_code(2.25, 1) == 63 &&
              fit_phase_code(2.25, -1) == 189 && fit_phase_code(-0.001, 1) == 0, "fit_phase_code: %d codes off", bad);
    }
    ref = calloc((size_t)LZ * LY * LX, 1);
    for (int z = 0; z < LZ; z++) for (double q = 1.0; q <= 5.0; q += 0.0005) {   /* the reference wraps, rasterised on the label grid */
        const double rho = R0 + P * q, th = 2 * M_PI * q;
        const int y = (int)lround((CY + rho * sin(th)) * 0.5), x = (int)lround((CX + rho * cos(th)) * 0.5);
        if (y >= 0 && y < LY && x >= 0 && x < LX) ref[((size_t)z * LY + y) * LX + x] = code_of(q);
    }
    ref_kind = WIND_CODES;
    surf A = make(2.3, 3.6, 10, 80, 1.5, 0, 0), B = make(4.5, 6.5, 10, 80, 0, 0, 0), C = make(2.0, 3.0, 10, 80, 0, 1, 0), D = make(6.0, 7.0, 10, 80, 0, 0, 0),
         E = make(2.3, 3.6, 10, 80, 1.5, 0, 1);
    wind_result ra = run(&A, 0, "A duplicate", WA_DUPLICATE, 1), rb = run(&B, 0, "B new", WA_NEW, 1), rc = run(&C, 0, "C switch", WA_CONFLICT, 0),
                rd = run(&D, 0, "D outside", WA_UNASSIGNED, 0), re = run(&E, 0, "E mirrored", WA_DUPLICATE, 1), ra2 = run(&A, 1, "A drop_dup", WA_DUPLICATE, 0);
    {   /* C: the switched (outer) half dropped, the first half kept with the right q */
        size_t kept_in = 0, kept_out = 0, bad = 0, n_in = 0;
        for (int r = 0; r < C.h; r++) for (int c = 0; c < C.w; c++) {
            const size_t i = (size_t)r * C.w + c; const int outer = c >= C.w / 2;
            n_in += !outer;
            if (isnan(rc.q[i])) continue;
            if (outer) kept_out++; else { kept_in++; bad += qerr(rc.q[i], C.qtrue[i]) > 0.06; }
        }
        printf("C switch: kept %zu of %zu inner-half vertices (q right: %s), %zu switched-half vertices kept\n", kept_in, n_in, bad ? "NO" : "yes", kept_out);
        CHECK(kept_in * 10 >= n_in * 7 && !bad && kept_out * 10 <= (size_t)C.w * C.h / 2, "C: conflict cells not dropped as expected");
    }
    {   size_t left = 0; for (size_t i = 0; i < (size_t)A.w * A.h; i++) left += !isnan(ra2.q[i]);
        printf("A drop_dup: %zu of %zu vertices left (all on the reference)\n", left, (size_t)A.w * A.h);
        CHECK(left * 20 <= (size_t)A.w * A.h, "A drop_dup: duplicate cells kept"); }
    for (size_t i = 0; i < (size_t)D.w * D.h; i++) CHECK(isnan(rd.q[i]) && rd.flag[i] == 4, "D: unassigned vertex %zu has a q", i);
    /* a dense winding raster (phase-raster --winding: 18 q mod 252 everywhere between the wraps): D gets its q */
    for (int z = 0; z < LZ; z++) for (int y = 0; y < LY; y++) for (int x = 0; x < LX; x++) {
        const double qr = (hypot(2 * y - CY, 2 * x - CX) - R0) / P;   /* the winding position: the sheets through this angle sit at integer steps of it */
        ref[((size_t)z * LY + y) * LX + x] = qr >= 0.5 && qr <= 8 ? (uint8_t)((lround(qr * 18.0) % 252 + 252) % 252) : 255;
    }
    ref_kind = WIND_WINDING;
    wind_result rd2 = run(&D, 0, "D dense", WA_NEW, 1);
    wind_result_free(&ra); wind_result_free(&rb); wind_result_free(&rc); wind_result_free(&rd); wind_result_free(&re); wind_result_free(&ra2); wind_result_free(&rd2);
    surf all[5] = {A, B, C, D, E};
    for (int i = 0; i < 5; i++) { free(all[i].xyz); free(all[i].valid); free(all[i].qtrue); }
    free(ref);
    {   /* wind_seed: one wrap q 5 .. 6 (columns reversed: the grid runs outer -> inner) -> K 5 at the inner end, q rising outward */
        surf S = make(5.0, 6.0, 10, 50, 0, 0, 0);
        const int w = S.w, h = S.h; float *xr = malloc((size_t)w * h * 3 * sizeof *xr); uint8_t *vr = malloc((size_t)w * h); double *qt = malloc((size_t)w * h * sizeof *qt);
        for (int r = 0; r < h; r++) for (int c = 0; c < w; c++) { const size_t a = (size_t)r * w + c, b = (size_t)r * w + (w - 1 - c);
            memcpy(xr + 3 * a, S.xyz + 3 * b, 3 * sizeof *xr); vr[a] = S.valid[b]; qt[a] = S.qtrue[b]; }
        wind_result r; wind_seed_info in;
        CHECK(!wind_seed(xr, vr, w, h, axis_fn, nullptr, 5.0, 0, NAN, &in, &r), "wind_seed failed");
        double emax = 0; size_t nq = 0;
        for (size_t i = 0; i < (size_t)w * h; i++) if (vr[i] && !isnan(r.q[i])) { const double e = fabs(r.q[i] - qt[i]); if (e > emax) emax = e; nq++; }
        CHECK(in.r_start < in.r_end && in.span > 0.9 && in.span < 1.1 && emax < 0.06 && nq > 0, "wind_seed: r %.0f -> %.0f, span %.2f, max |q - q_true| %.3f over %zu vertices",
              in.r_start, in.r_end, in.span, emax, nq);
        printf("wind-seed: one reversed wrap -> K at its inner end, span %.2f turn, max |q - q_true| %.3f\n", in.span, emax);
        wind_result_free(&r); free(xr); free(vr); free(qt); free(S.xyz); free(S.valid); free(S.qtrue);
    }
    {   /* wind_seed on a named set under an axis 50 voxels off the umbilicus: w5 (q 5 .. 5.95, columns reversed), w6 (6.1 .. 6.95,
           partial), w7 (6.97 .. 7.92: cut a little before the seam), w8 (8.4 .. 8.98, partial: the off-centre axis swaps its ends'
           radii, so its own radius rule reverses it). The set's sense and seam put every vertex in its true turn. */
        const double qa[4] = {5.0, 6.1, 6.97, 8.4}, qb[4] = {5.95, 6.95, 7.92, 8.98}, K[4] = {5, 6, 7, 8};
        surf S[4]; wind_seed_info i1[4]; double own8 = 0;
        for (int k = 0; k < 4; k++) {
            S[k] = make(qa[k], qb[k], 10, 50, 0, 0, k == 0);
            wind_result r; CHECK(!wind_seed(S[k].xyz, S[k].valid, S[k].w, S[k].h, axis_off, nullptr, K[k], 0, NAN, &i1[k], &r), "wind_seed set pass 1");
            if (k == 3) for (size_t i = 0; i < (size_t)S[k].w * S[k].h; i++) if (!isnan(r.q[i])) { const double e = fabs(r.q[i] - S[k].qtrue[i]); if (e > own8) own8 = e; }
            wind_result_free(&r);
        }
        int sense; double phi0, vote[2], res; wind_seed_set(i1, 4, &sense, &phi0, vote, &res);
        double emax = 0;
        for (int k = 0; k < 4; k++) {
            wind_result r; wind_seed_info in;
            CHECK(!wind_seed(S[k].xyz, S[k].valid, S[k].w, S[k].h, axis_off, nullptr, K[k], sense, phi0, &in, &r), "wind_seed set pass 2");
            for (size_t i = 0; i < (size_t)S[k].w * S[k].h; i++) if (!isnan(r.q[i])) { const double e = fabs(r.q[i] - S[k].qtrue[i]); if (e > emax) emax = e; }
            wind_result_free(&r); free(S[k].xyz); free(S[k].valid); free(S[k].qtrue);
        }
        CHECK(i1[3].radial_dir != 1 && sense == 1 && emax < 0.25 && own8 > 0.4, "wind_seed set: w8's own rule dir %+d (worst %.2f), set sense %+d (votes %.0f / %.0f), seam %.3f, worst |q - q_true| %.3f",
              i1[3].radial_dir, own8, sense, vote[0], vote[1], phi0, emax);
        printf("wind-seed set: 4 named wraps, axis 50 voxels off: w8 alone reversed (worst %.2f turn), the set's sense %+d (votes %.0f / %.0f) and seam %.3f: worst |q - q_true| %.3f\n",
               own8, sense, vote[0], vote[1], phi0, emax);
    }
    {   /* wind_seed_decide (2026-10-19, Paris 4 June came out reversed): 3 long named surfaces (w2 q 2 .. 8, w8 8 .. 13, w13 13 .. 17,
           columns outer -> inner as the June unroll) and 20 short half-wraps w17 .. w36 (q K .. K + 0.5, from azimuth 0) under an
           axis 50 voxels off the umbilicus on the side that makes every short one's end band nearer the axis at its outer end: the old
           end-gap vote (wind_seed_set) takes -1. Plus a misnamed w20 copy (it lies at q 37.0 .. 37.5, outside w36: out of order) and
           w50, a 3-turn surface of the opposite hand (z 120 .. 160, alone: its own pairs only). Expected: sense +1, names outward, the
           misnamed one and w50 dropped, every other surface's q = its true q (mean within 0.06 turn; spread under 0.25: phi is measured around the off-centre axis, up to
           0.06 turn off the true azimuth at q 2; a wrong sense spreads by turns). */
        enum { NS = 25 }; surf S[NS]; double K[NS];
        S[0] = make2(2, 8, 10, 60, 512, 1, 1); K[0] = 2; S[1] = make2(8, 13, 10, 60, 512, 1, 1); K[1] = 8; S[2] = make2(13, 17, 10, 60, 512, 1, 1); K[2] = 13;
        for (int k = 0; k < 20; k++) { S[3 + k] = make2(17 + k, 17.5 + k, 10, 60, 512, 1, 0); K[3 + k] = 17 + k; }
        S[23] = make2(37.0, 37.5, 10, 60, 512, 1, 0); K[23] = 20;
        S[24] = make2(50, 53, 120, 160, 512, -1, 0); K[24] = 50;
        wind_choice ch; wind_surf_vote pv[NS]; int old; double qm[NS], qs[NS];
        seed_set(S, K, NS, &ch, pv, &old, qm, qs);
        double worst = 0; int bad = 0;
        for (int k = 0; k < NS; k++) if (k != 23 && k != 24) { if (pv[k].dropped || !(fabs(qm[k]) < 0.06 && qs[k] < 0.25)) { bad++; fprintf(stderr, "  surface %d: mean q - q_true %.3f, spread %.3f, dropped %d\n", k, qm[k], qs[k], pv[k].dropped); } worst = fmax(worst, fabs(qm[k]) + qs[k]); }
        printf("wind-seed decide: old end-gap vote %+d; pairs: sense %+d order %+d score %.3f (%zu / %zu), candidates %.3f %.3f %.3f %.3f, dropped %d (misnamed %d, w50 %d: own + %zu - %zu), worst |q - q_true| %.3f\n",
               old, ch.sense, ch.order, ch.score, ch.agree, ch.agree + ch.dis, ch.cand[0][0], ch.cand[0][1], ch.cand[1][0], ch.cand[1][1], ch.ndropped, pv[23].dropped, pv[24].dropped, pv[24].own_pos, pv[24].own_neg, worst);
        CHECK(old == -1, "the fixture does not reproduce the old failure (end-gap vote %+d)", old);
        CHECK(ch.sense == 1 && ch.order == 1 && ch.score > 0.95, "decide: sense %+d order %+d score %.3f", ch.sense, ch.order, ch.score);
        CHECK(pv[0].own_dir == 1 && pv[1].own_dir == 1 && pv[2].own_dir == 1 && pv[24].own_dir == -1, "own senses %+d %+d %+d / reversed hand %+d", pv[0].own_dir, pv[1].own_dir, pv[2].own_dir, pv[24].own_dir);
        CHECK(pv[23].dropped && pv[24].dropped && ch.ndropped == 2, "dropped: w40 %d, w50 %d, %d in all", pv[23].dropped, pv[24].dropped, ch.ndropped);
        CHECK(!bad, "%d kept surfaces off their true q (worst %.3f)", bad, worst);
        wind_pitch_hi = 20;   /* a papyrus pitch prior the fixture's 40 voxels violate: the votes are not sheets, no evidence */
        seed_set(S, K, NS, &ch, pv, &old, qm, qs); wind_pitch_hi = 0;
        printf("wind-seed decide, pitch prior under the set's pitch %.1f: evidence %d\n", ch.pitch, ch.evidence);
        CHECK(!ch.evidence && !ch.pitch_ok, "pitch prior: evidence %d pitch_ok %d", ch.evidence, ch.pitch_ok);
        for (int k = 0; k < NS; k++) { free(S[k].xyz); free(S[k].valid); free(S[k].qtrue); }
    }
    {   /* names counting inward, single wraps a little over a turn (q N - 0.02 .. N + 1.02, the seam's overlap): wrapN at q 30 - N -> order
           -1 (K -> -K) and, from the seam's same-sheet pairs alone (no radial pair tells the sense of single wraps), sense +1: q =
           q_true - 30 on every surface, rising outward and around */
        enum { NS = 12 }; surf S[NS]; double K[NS];
        for (int k = 0; k < NS; k++) { S[k] = make2(5 + k - 0.02, 6.02 + k, 10, 60, 512, 1, k & 1); K[k] = 25 - k; }
        wind_choice ch; wind_surf_vote pv[NS]; int old; double qm[NS], qs[NS];
        seed_set(S, K, NS, &ch, pv, &old, qm, qs);
        int bad = 0; for (int k = 0; k < NS; k++) bad += pv[k].dropped || !(fabs(qm[k] + 30) < 0.06 && qs[k] < 0.25);
        printf("wind-seed decide, names counting inward: sense %+d (margin %.3f) order %+d score %.3f (%zu votes, pitch %.1f), q - q_true %.3f .. %.3f\n", ch.sense, ch.sense_margin, ch.order, ch.score, ch.agree + ch.dis, ch.pitch, qm[0], qm[NS - 1]);
        CHECK(ch.order == -1 && ch.sense == 1 && ch.score > 0.95 && ch.sense_margin > 0.02 && !bad, "inward names: sense %+d (margin %.3f) order %+d score %.3f, %d surfaces off q_true - 30",
              ch.sense, ch.sense_margin, ch.order, ch.score, bad);
        for (int k = 0; k < NS; k++) { free(S[k].xyz); free(S[k].valid); free(S[k].qtrue); }
    }
    {   /* partial wraps (0.6 turn from azimuth 0, never meeting at the seam): the names' order is seen, the sense is not (margin ~0) */
        enum { NS = 10 }; surf S[NS]; double K[NS];
        for (int k = 0; k < NS; k++) { S[k] = make2(5 + k, 5.6 + k, 10, 60, 512, 1, 0); K[k] = 5 + k; }
        wind_choice ch; wind_surf_vote pv[NS]; int old; double qm[NS], qs[NS];
        seed_set(S, K, NS, &ch, pv, &old, qm, qs);
        printf("wind-seed decide, partial wraps: sense %+d (margin %.3f: undetermined), order %+d (margin %.3f)\n", ch.sense, ch.sense_margin, ch.order, ch.order_margin);
        CHECK(ch.order == 1 && ch.order_margin > 0.5 && fabs(ch.sense_margin) < 0.02, "partial wraps: order %+d margin %.3f, sense margin %.3f", ch.order, ch.order_margin, ch.sense_margin);
        for (int k = 0; k < NS; k++) { free(S[k].xyz); free(S[k].valid); free(S[k].qtrue); }
    }
    printf("winding assignment: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
