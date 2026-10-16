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
    printf("winding assignment: %s\n", fails ? "FAIL" : "PASS");
    return fails != 0;
}
