/* ufsm phase-gate IN-FIELDS OUT-FIELDS --ref MODEL [--mode gate|ref] [--lo 0.3] [--hi 0.6] [--R 6] [--ref-conf 0.5] [--threads 16]
   (and fit --phase-gate 1 with --ref-model: the same on the fit's own fields, src/fit.h fit_phase_gate)
   The network phase replaced by a reference model's phase (W_ref mod 1, e.g. the count-field cf ref) where the network phase does not
   rotate with the sheets. On crushed scrolls (PHerc0490B) wind96's phase is confident on ~half the papyrus but turns only ~0.1 per
   sheet along the radius (Paris 4: ~0.75), so the snapped slab fits (fit --ref-snap 1: integer round(U_ref - phase)) add turns from
   phase noise and the slabs disagree in their overlaps. Per field cell (one z row at a time):
     rotation q = <c dphi/dr> / <c rho |n . r|>: the confidence-weighted radial phase rate (central wrapped differences in y / x along
     the outward radial direction, turns per native voxel) over the density head's radial rate, both box-summed over +-R cells (y / x);
     gate g = smoothstep(q, lo, hi) (1: the network phase kept);
     the new phasor = g x network phasor + (1 - g) x ref-conf x e^{2 pi i W_ref}.
   --mode ref: g = 0 everywhere (every cell with material takes the ref's phase: the snap then reproduces the ref).
   Output: raw fit fields (all planes copied, pc / ps replaced) + gate.u8 (g x 255 per cell) for inspection. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "fit.h"

static const char *pg_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 3; i + 1 < argc; i++) if (!strcmp(argv[i], k)) return argv[i + 1]; return d; }
static double pg_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static double pg_smooth(double x, double a, double b) { if (x <= a) return 0; if (x >= b) return 1; const double t = (x - a) / (b - a); return t * t * (3 - 2 * t); }
static double pg_axis(const float *v, int n, double z) { if (z <= 0) return v[0]; if (z >= n - 1) return v[n - 1]; const int i = (int)z; const double t = z - i; return v[i] + t * (v[i + 1] - v[i]); }
static float pg_ang_t[65536], pg_mag_t[65536];   /* per (pc, ps) byte pair: the phase in turns, the phasor length x 127 (filled by fit_phase_gate) */
static double pg_mag(int a, int b) { return pg_mag_t[a << 8 | b]; }
static double pg_ang(int a, int b) { return pg_ang_t[a << 8 | b]; }

typedef struct {
    const fit_fields *f; int zo; const float *cy, *cx; int na; const float *wref; const fit_pgate_cfg *g;   /* wref: W_ref per cell of the block */
    uint8_t *pc, *ps, *gate; long long nmat[64], ngated[64]; double qsum[64]; long long qn[64];
} pg_job;

static void pg_rows(void *ctx, int z0, int z1, int t) {
    pg_job *J = ctx; const fit_fields *f = J->f; const int H = f->H, W = f->W, R = J->g->R; const double h = f->h;
    const size_t HW = (size_t)H * W;
    float *A = malloc(W * sizeof(float)), *B = malloc(W * sizeof(float)); double *ca = malloc(W * sizeof(double)), *cb = malloc(W * sizeof(double));
    float *pa = malloc(HW * sizeof(float)), *pb = malloc(HW * sizeof(float));
    for (int z = z0; z < z1; z++) {
        const double pz = (J->zo + z + 0.5) * h, cy = pg_axis(J->cy, J->na, pz), cx = pg_axis(J->cx, J->na, pz);
        const float *wr = J->wref + (size_t)z * HW;
        const uint8_t *pc = f->pc + (size_t)z * HW, *ps = f->ps + (size_t)z * HW;
        /* per cell: radial phase rate (a: c x rate) and the density radial rate (b: c x rho |n.r|), c = the phase confidence */
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            const size_t j = (size_t)y * W + x, i = (size_t)z * HW + j; pa[j] = pb[j] = 0;
            if (J->g->mode || !f->w[i] || y == 0 || x == 0 || y + 1 >= H || x + 1 >= W || !f->sp[i]) continue;
            const double c0 = pg_mag(pc[j], ps[j]) / 127.0; if (c0 < 0.125) continue;
            double ry = (y + 0.5) * h - cy, rx = (x + 0.5) * h - cx; const double rl = hypot(ry, rx); if (rl < 1) continue; ry /= rl; rx /= rl;
            const double p0 = pg_ang(pc[j], ps[j]);
            const size_t ny[2] = {j - W, j + W}, nx[2] = {j - 1, j + 1}; double gy = 0, gx = 0; int ok = 1;
            for (int s = 0; s < 2 && ok; s++) {
                const size_t a = ny[s], b = nx[s];
                if (pg_mag(pc[a], ps[a]) < 16 || pg_mag(pc[b], ps[b]) < 16) { ok = 0; break; }
                double dy = pg_ang(pc[a], ps[a]) - p0, dx = pg_ang(pc[b], ps[b]) - p0;
                dy -= floor(dy + 0.5); dx -= floor(dx + 0.5); gy += s ? dy : -dy; gx += s ? dx : -dx;
            }
            if (!ok) continue;
            const double rate = (0.5 * gy * ry + 0.5 * gx * rx) / h;   /* turns per native voxel along +r */
            const double nyy = f->t[1][i] / 255.0, nxx = f->t[2][i] / 255.0, nyx = f->t[5][i] / 255.0 - 0.5;
            const double nr = sqrt(fmax(0, ry * ry * nyy + rx * rx * nxx + 2 * ry * rx * nyx)), dn = nr / (f->sp[i] * h / 8.0);
            pa[j] = (float)(c0 * rate); pb[j] = (float)(c0 * dn);
        }
        if (!J->g->mode) for (int y = 0; y < H; y++) {   /* box sums over +-R, separable: x running sums into pa / pb in place (via a row copy) */
            float *ra = A, *rb = B; memcpy(ra, pa + (size_t)y * W, W * sizeof(float)); memcpy(rb, pb + (size_t)y * W, W * sizeof(float));
            double s1 = 0, s2 = 0; for (int x = 0; x < R && x < W; x++) { s1 += ra[x]; s2 += rb[x]; }
            for (int x = 0; x < W; x++) {
                if (x + R < W) { s1 += ra[x + R]; s2 += rb[x + R]; }
                if (x - R - 1 >= 0) { s1 -= ra[x - R - 1]; s2 -= rb[x - R - 1]; }
                pa[(size_t)y * W + x] = (float)s1; pb[(size_t)y * W + x] = (float)s2;
            }
        }
        if (!J->g->mode) { memset(ca, 0, W * sizeof(double)); memset(cb, 0, W * sizeof(double)); for (int y = 0; y < R && y < H; y++) for (int x = 0; x < W; x++) { ca[x] += pa[(size_t)y * W + x]; cb[x] += pb[(size_t)y * W + x]; } }
        /* (the outputs may alias f's pc / ps: a cell's own phase is read before its write, the gradients above read the row first) */
        for (int y = 0; y < H; y++) {
            if (!J->g->mode) {   /* the column sums of the x box sums over rows [y - R, y + R] */
                if (y + R < H) for (int x = 0; x < W; x++) { ca[x] += pa[(size_t)(y + R) * W + x]; cb[x] += pb[(size_t)(y + R) * W + x]; }
                if (y - R - 1 >= 0) for (int x = 0; x < W; x++) { ca[x] -= pa[(size_t)(y - R - 1) * W + x]; cb[x] -= pb[(size_t)(y - R - 1) * W + x]; }
            }
            for (int x = 0; x < W; x++) {
            const size_t j = (size_t)y * W + x, i = (size_t)z * HW + j;
            const uint8_t c8 = pc[j], s8 = ps[j];
            if (!f->w[i]) { if (J->gate) J->gate[i] = 0; J->pc[i] = c8; J->ps[i] = s8; continue; }
            J->nmat[t]++;
            double g = 0;
            if (!J->g->mode) {
                const double sa = ca[x], sb = cb[x];
                if (sb > 1e-9) { const double q = sa / sb; g = pg_smooth(q, J->g->lo, J->g->hi); J->qsum[t] += q; J->qn[t]++; }
            }
            if (J->gate) J->gate[i] = (uint8_t)lround(255 * g);
            if (g < 1 && wr[j] == wr[j]) {
                float sr, cr; sincosf((float)(2 * M_PI) * (wr[j] - floorf(wr[j])), &sr, &cr);   /* (the network phasor c (cos, sin) = its bytes) */
                const double re = g * (c8 - 128.0) / 127.0 + (1 - g) * J->g->rc * cr, im = g * (s8 - 128.0) / 127.0 + (1 - g) * J->g->rc * sr;
                J->pc[i] = (uint8_t)lround(fmin(255, fmax(1, 128 + 127 * re))); J->ps[i] = (uint8_t)lround(fmin(255, fmax(1, 128 + 127 * im)));
                J->ngated[t] += g < 0.5;
            } else { J->pc[i] = c8; J->ps[i] = s8; }
            }
        }
    }
    free(A); free(B); free(ca); free(cb); free(pa); free(pb);
}

int fit_phase_gate(const fit_fields *f, int zo, const float *cy, const float *cx, int na, const fit_set *ref, fit_set_dev *gd, const double ro[3], const fit_pgate_cfg *g,
                   uint8_t *pc_out, uint8_t *ps_out, uint8_t *gate, double st[4]) {
    const size_t n = (size_t)f->D * f->H * f->W;
    float *wref = malloc(n * sizeof(float)); if (!wref) return -1;
    const double o[3] = {zo * f->h + ro[0], ro[1], ro[2]};   /* (grid points o + (i + 0.5) h = the cell centres in the ref box) */
    const int64_t n3[3] = {f->D, f->H, f->W};
    const double tq0 = pg_now();
    if (gd ? fit_set_grid(gd, ref, o, n3, f->h, 0, wref) : fit_set_grid_cpu(ref, o, n3, f->h, 0, nullptr, wref)) { free(wref); return -1; }
    static int tabs; if (!tabs) { for (int a = 0; a < 256; a++) for (int b = 0; b < 256; b++) { pg_ang_t[a << 8 | b] = (float)(atan2(b - 128.0, a - 128.0) / (2 * M_PI)); pg_mag_t[a << 8 | b] = (float)hypot(a - 128.0, b - 128.0); } tabs = 1; }
    pg_job J = {.f = f, .zo = zo, .cy = cy, .cx = cx, .na = na, .wref = wref, .g = g, .pc = pc_out, .ps = ps_out, .gate = gate};
    const double tq1 = pg_now(); fit_par(f->D, pg_rows, &J);
    if (getenv("UFSM_PG_VERBOSE")) fprintf(stderr, "phase gate block z %d: W_ref %.2f s (%s), rows %.2f s\n", zo, tq1 - tq0, gd ? "GPU" : "CPU", pg_now() - tq1);
    for (int t = 0; t < 64; t++) { st[0] += J.nmat[t]; st[1] += J.ngated[t]; st[2] += J.qsum[t]; st[3] += J.qn[t]; }
    free(wref);
    return 0;
}

int cmd_phase_gate(int argc, char **argv) {
    if (argc < 4 || !*pg_opt(argc, argv, "--ref", "")) {
        fprintf(stderr, "usage: ufsm phase-gate <in-fields> <out-fields> --ref MODEL [--mode gate|ref] [--lo 0.3] [--hi 0.6] [--R 6] [--ref-conf 0.5] [--threads 16]\n"
                        "  the network phase replaced by the ref model's where it does not rotate with the density's sheets (raw fields out)\n");
        return 2;
    }
    const double t0 = pg_now();
    fit_set_threads(atoi(pg_opt(argc, argv, "--threads", "16")));
    fit_set *ref = fit_set_load(pg_opt(argc, argv, "--ref", "")); if (!ref) { fprintf(stderr, "phase-gate: cannot load the ref\n"); return 1; }
    fit_fields_skip_planes(1 | 2); fit_fields_ct_planes(0);
    fit_fields_buf g; if (fit_fields_load(argv[2], 0, 0, 1, &g)) { fprintf(stderr, "phase-gate: cannot load %s\n", argv[2]); return 1; }
    const size_t n = (size_t)g.shape[0] * g.shape[1] * g.shape[2];
    const fit_pgate_cfg pc = {.mode = !strcmp(pg_opt(argc, argv, "--mode", "gate"), "ref"), .R = atoi(pg_opt(argc, argv, "--R", "6")), .lo = atof(pg_opt(argc, argv, "--lo", "0.3")),
                              .hi = atof(pg_opt(argc, argv, "--hi", "0.6")), .rc = atof(pg_opt(argc, argv, "--ref-conf", "0.5"))};
    const double ro[3] = {(double)(g.origin[0] - ref->global->origin[0]), (double)(g.origin[1] - ref->global->origin[1]), (double)(g.origin[2] - ref->global->origin[2])};
    uint8_t *gate = calloc(n, 1); double st[4] = {0};
    if (!gate || fit_phase_gate(&g.f, 0, g.cy, g.cx, (int)(g.shape[0] * g.h) + 1, ref, nullptr, ro, &pc, g.a[0], g.a[1], gate, st)) { fprintf(stderr, "phase-gate: out of memory / ref grid failed\n"); return 1; }
    uint8_t *a10[10]; for (int k = 0; k < 10; k++) a10[k] = g.a[k];
    if (fit_fields_save(argv[3], (const long long[3]){g.origin[0], g.origin[1], g.origin[2]}, g.shape, g.h, g.um, a10, g.cy, g.cx, (int)(g.shape[0] * g.h) + 1)) { fprintf(stderr, "phase-gate: cannot write %s\n", argv[3]); return 1; }
    { char p[1400]; snprintf(p, sizeof p, "%s/gate.u8", argv[3]); FILE *fo = fopen(p, "wb"); if (fo) { fwrite(gate, 1, n, fo); fclose(fo); } }
    printf("phase-gate: %dx%dx%d cells, mode %s: mean rotation q %.3f over %.0f cells; ref phase (gate < 0.5) on %.3f of %.0f material cells (%.0f s) -> %s\n",
           g.shape[0], g.shape[1], g.shape[2], pc.mode ? "ref" : "gate", st[3] ? st[2] / st[3] : 0, st[3], st[0] ? st[1] / st[0] : 0, st[0], pg_now() - t0, argv[3]);
    free(gate); fit_fields_free(&g); fit_set_free(ref);
    return 0;
}

/* ufsm w-at MODEL Z,Y,X [Z,Y,X ...]: W of a fitted model / set at global native points (the wrap through a point: floor(W)) */
int cmd_w_at(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: ufsm w-at <model-or-set> Z,Y,X [Z,Y,X ...]   W at global native points (wrap = floor(W))\n"); return 2; }
    fit_set *S = fit_set_load(argv[2]); if (!S) { fprintf(stderr, "w-at: cannot load %s\n", argv[2]); return 1; }
    for (int i = 3; i < argc; i++) {
        double p[3]; if (sscanf(argv[i], "%lf,%lf,%lf", p, p + 1, p + 2) != 3) { fprintf(stderr, "w-at: bad point %s\n", argv[i]); return 2; }
        const double w = fit_set_W(S, p[0] - S->global->origin[0], p[1] - S->global->origin[1], p[2] - S->global->origin[2]);
        printf("%s\t%.4f\tw%03ld\n", argv[i], w, (long)floor(w));
    }
    fit_set_free(S); return 0;
}
