/* ufsm phase-near GT.json SOURCE FIELDS-DIR --box z,y,x,nz,ny,nx [--conf 0] [--margin 8] [--gpu 0] [--threads 6]
   Sheet position at native resolution (w24 2026-11-27, lead: "what 2.4 adds is sheet position inside packed gaps", invisible to wbench's
   19.2 um fit cells): a network's predicted phase-0 surfaces against the GT sheets on the level-0 grid of SOURCE, no fitter.
   - GT: SOURCE's level-0 phase of GT.json (e.g. the on-the-fly pair rule from the GT meshes at rmax 160); its sheets are the voxels of
     phase 0 (the labelled faces), its region the voxels of known phase.
   - Prediction: a predict --fit-zarr / --fit-fields directory (pc / ps, any level and --fit-h: a 9.6 um net's fields are sampled at the
     level-0 voxel centres by trilinear interpolation of the phase vectors, i.e. upsampled); its sheets are the voxels next to a phase
     wrap (an integer crossing between 6-neighbours; the voxel nearer the integer), with both vectors' confidence > --conf.
   - Inside the GT region, --margin voxels in from the box faces: the distance from every predicted sheet voxel to the nearest GT sheet
     voxel (pred -> GT: misplaced or extra sheets) and from every GT sheet voxel to the nearest predicted one (GT -> pred: misplaced or
     missed sheets), in um; p50 / p90 and the shares within 1 / 2 / 4 voxels. One row on stdout (tab separated, header with --header 1).
   The box is in level-0 voxels of SOURCE. CPU EDTs (edt.h). */
#include "band.h"
#include "band_onfly.h"
#include "edt.h"
#include "fit.h"
#include "meshcodes.h"
#include "optcheck.h"
#include "phasenear.h"
#include "sources.h"
#include "store.h"
#include "zarr3.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *pn_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 5; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(k))) return argv[i + 1]; return d; }
static int cmpf(const void *a, const void *b) { const float x = *(const float *)a, y = *(const float *)b; return x < y ? -1 : x > y; }
static float pct(float *v, size_t n, double q) { if (!n) return NAN; size_t k = (size_t)(q * (double)(n - 1) + 0.5); return v[k]; }

/* the predicted phase (turns in [0, 1)) and confidence at the level-0 voxel centres of the box: the fields' phase vectors trilinear */
void phase_near_sample(const fit_fields_buf *F, const int64_t o[3], const int64_t n[3], double um0, float *ph, float *cf) {
    const fit_fields *f = &F->f; const double s = um0 / F->um, h = F->h;
    for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) for (int64_t x = 0; x < n[2]; x++) {
        const int64_t p[3] = {o[0] + z, o[1] + y, o[2] + x}; double c[3]; int i0[3]; double w[3]; int ok = 1;
        for (int d = 0; d < 3; d++) {   /* the cell coordinate: native position (p + 0.5) s - origin, cells centred at h (i + 0.5) */
            c[d] = ((double)p[d] + 0.5) * s - (double)F->origin[d]; c[d] = c[d] / h - 0.5;
            const int lim = d == 0 ? f->D : d == 1 ? f->H : f->W;
            if (c[d] < 0) c[d] = 0; if (c[d] > lim - 1) c[d] = lim - 1;
            i0[d] = (int)floor(c[d]); if (i0[d] >= lim - 1) i0[d] = lim > 1 ? lim - 2 : 0; w[d] = c[d] - i0[d]; if (lim == 1) { i0[d] = 0; w[d] = 0; }
            if (lim < 1) ok = 0;
        }
        double vc = 0, vs = 0;
        if (ok) for (int k = 0; k < 8; k++) {
            const int a = i0[0] + (k >> 2 & 1), b = i0[1] + (k >> 1 & 1), e = i0[2] + (k & 1);
            const double wt = ((k >> 2 & 1) ? w[0] : 1 - w[0]) * ((k >> 1 & 1) ? w[1] : 1 - w[1]) * ((k & 1) ? w[2] : 1 - w[2]);
            if (wt == 0) continue;
            const size_t q = ((size_t)a * f->H + (size_t)b) * f->W + (size_t)e;
            vc += wt * (f->pc[q] - 128) / 127.0; vs += wt * (f->ps[q] - 128) / 127.0;
        }
        const size_t i = ((size_t)z * n[1] + (size_t)y) * n[2] + (size_t)x;
        double t = atan2(vs, vc) / (2 * M_PI); if (t < 0) t += 1; if (t >= 1) t -= 1;
        ph[i] = (float)t; cf[i] = (float)sqrt(vc * vc + vs * vs);
    }
}

/* --min-comp N (topoloss 2026-11-27, the survey's "persistence filter" without retraining): predicted sheet components (26-connected
   sheet voxels) of fewer than N voxels are dropped before the distances: tests whether w24's extra wraps are small fragments a
   post-filter removes, or whole spurious sheets. 0: off (the measurement unchanged). */
static long g_min_comp;
void phase_near_set_min_comp(long n) { g_min_comp = n > 0 ? n : 0; }
static void drop_small(uint8_t *mp, const int64_t n[3], long minc) {
    const size_t N = (size_t)n[0] * n[1] * n[2], sy = (size_t)n[2], sz = (size_t)n[1] * n[2];
    size_t cap = 1 << 20, *st = malloc(cap * sizeof *st), *cv = malloc(cap * sizeof *cv), ccap = cap; size_t nc = 0, nd = 0, vd = 0, vt = 0;
    for (size_t s0 = 0; s0 < N; s0++) {
        if (mp[s0] != 1) continue;
        size_t ns = 0, nv = 0; st[ns++] = s0; mp[s0] = 2;
        while (ns) {
            const size_t i = st[--ns];
            if (nv == ccap) { ccap *= 2; cv = realloc(cv, ccap * sizeof *cv); }
            cv[nv++] = i;
            const int64_t z = (int64_t)(i / sz), y = (int64_t)(i / sy % (size_t)n[1]), x = (int64_t)(i % sy);
            for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
                const int64_t a = z + dz, b = y + dy, c = x + dx;
                if (a < 0 || a >= n[0] || b < 0 || b >= n[1] || c < 0 || c >= n[2]) continue;
                const size_t j = (size_t)a * sz + (size_t)b * sy + (size_t)c;
                if (mp[j] != 1) continue;
                mp[j] = 2; if (ns == cap) { cap *= 2; st = realloc(st, cap * sizeof *st); } st[ns++] = j;
            }
        }
        nc++; vt += nv;
        if ((long)nv < minc) { nd++; vd += nv; for (size_t k = 0; k < nv; k++) mp[cv[k]] = 0; }
    }
    for (size_t i = 0; i < N; i++) mp[i] = mp[i] != 0;
    fprintf(stderr, "phase-near: --min-comp %ld: dropped %zu of %zu predicted sheet components, %zu of %zu voxels (%.1f %%)\n", minc, nd, nc, vd, vt, vt ? 100.0 * vd / vt : 0.0);
    free(st); free(cv);
}

/* the measurement on one box (phasenear.h): g the GT phase, ph / cf the predicted phase (turns) and confidence on the same grid */
int phase_near_measure(const uint8_t *g, const float *ph, const float *cf, const int64_t n[3], int mg, float cmin, int nth, phase_near_res *R) {
    const size_t N = (size_t)n[0] * n[1] * n[2];
    uint8_t *mg_ = calloc(N, 1), *mp = calloc(N, 1); float *dg = malloc(N * 4), *dp = malloc(N * 4);
    if (!mg_ || !mp || !dg || !dp) { free(mg_); free(mp); free(dg); free(dp); return -1; }
    const size_t sy = (size_t)n[2], sz = (size_t)n[1] * n[2]; size_t ngt = 0, npr = 0, kn = 0;
    for (size_t i = 0; i < N; i++) { mg_[i] = g[i] == 0; kn += g[i] != PHASE_UNKNOWN; }
    for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) for (int64_t x = 0; x < n[2]; x++) {
        const size_t i = (size_t)z * sz + (size_t)y * sy + (size_t)x;
        for (int a = 0; a < 3; a++) {
            if ((a == 0 && z + 1 >= n[0]) || (a == 1 && y + 1 >= n[1]) || (a == 2 && x + 1 >= n[2])) continue;
            const size_t j = i + (a == 0 ? sz : a == 1 ? sy : 1);
            if (cf[i] <= cmin || cf[j] <= cmin) continue;
            float d = ph[j] - ph[i]; if (d >= 0.5f) d -= 1; if (d < -0.5f) d += 1;
            const float e = ph[i] + d; if (e >= 0 && e < 1) continue;   /* no integer between them */
            const float ri = fminf(ph[i], 1 - ph[i]), rj = fminf(ph[j], 1 - ph[j]);
            mp[ri <= rj ? i : j] = 1;
        }
    }
    if (g_min_comp > 0) drop_small(mp, n, g_min_comp);
    if (edt_sq(mg_, n, dg, nth) || edt_sq(mp, n, dp, nth)) { free(mg_); free(mp); free(dg); free(dp); return -1; }
    float *a = malloc(N * 4), *b = malloc(N * 4); size_t na = 0, nb = 0; size_t w[2][3] = {{0}};
    for (int64_t z = mg; z < n[0] - mg; z++) for (int64_t y = mg; y < n[1] - mg; y++) for (int64_t x = mg; x < n[2] - mg; x++) {
        const size_t i = (size_t)z * sz + (size_t)y * sy + (size_t)x;
        if (g[i] == PHASE_UNKNOWN) continue;
        if (mp[i]) { npr++; if (dg[i] < EDT_INF) { const float d = sqrtf(dg[i]); a[na++] = d; w[0][0] += d <= 1; w[0][1] += d <= 2; w[0][2] += d <= 4; } }
        if (mg_[i]) { ngt++; if (dp[i] < EDT_INF) { const float d = sqrtf(dp[i]); b[nb++] = d; } else b[nb++] = 1e9f; const float d = b[nb - 1]; w[1][0] += d <= 1; w[1][1] += d <= 2; w[1][2] += d <= 4; }
    }
    qsort(a, na, 4, cmpf); qsort(b, nb, 4, cmpf);
    R->known = (double)kn / N; R->gt_sheet = ngt; R->pred_sheet = npr;
    R->pg50 = pct(a, na, 0.5); R->pg90 = pct(a, na, 0.9); R->gp50 = nb ? fminf(pct(b, nb, 0.5), 1e6f) : NAN; R->gp90 = nb ? fminf(pct(b, nb, 0.9), 1e6f) : NAN;
    for (int k = 0; k < 3; k++) { R->pg_within[k] = na ? (double)w[0][k] / na : 0; R->gp_within[k] = nb ? (double)w[1][k] / nb : 0; }
    free(mg_); free(mp); free(dg); free(dp); free(a); free(b);
    return 0;
}

int cmd_phase_near(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: ufsm phase-near GT.json SOURCE FIELDS-DIR --box z,y,x,nz,ny,nx [--conf 0] [--margin 8] [--gpu 0] [--threads 6] [--header 0] [--tag T] [--min-comp N]\n"
                                    "  predicted phase-0 sheets (predict --fit-zarr fields, any level: upsampled) vs SOURCE's GT phase sheets on its level-0 grid: distances in um\n"); return 2; }
    long long bx[6]; const char *bs = pn_opt(argc, argv, "--box", nullptr);
    if (!bs || sscanf(bs, "%lld,%lld,%lld,%lld,%lld,%lld", &bx[0], &bx[1], &bx[2], &bx[3], &bx[4], &bx[5]) != 6 || bx[3] < 16 || bx[4] < 16 || bx[5] < 16) { fprintf(stderr, "phase-near: --box z,y,x,nz,ny,nx (level-0 voxels, >= 16 each)\n"); return 2; }
    const char *cl = pn_opt(argc, argv, "--conf", "0"), *ml = pn_opt(argc, argv, "--min-comp", "0"); const int mg = opt_atoi("--margin", pn_opt(argc, argv, "--margin", "8"));
    const int gpu = opt_atoi("--gpu", pn_opt(argc, argv, "--gpu", "0")), nth = opt_atoi("--threads", pn_opt(argc, argv, "--threads", "6"));
    const char *tag = pn_opt(argc, argv, "--tag", argv[4]);
    store_global_init();
    sources *S = sources_load(argv[2]); if (!S) { fprintf(stderr, "phase-near: cannot load %s\n", argv[2]); return 1; }
    source *s = nullptr; for (int i = 0; i < S->n; i++) if (!strcmp(S->src[i].name, argv[3])) s = &S->src[i];
    if (!s || !s->phase) { fprintf(stderr, "phase-near: no source %s with a phase in %s\n", argv[3], argv[2]); return 1; }
    if (gpu >= 0) { if (phase_onfly_config(gpu, 2, 1024, 0)) { fprintf(stderr, "phase-near: --gpu %d: no GPU pair rule\n", gpu); return 1; } mesh_codes_gpu_config(gpu); }
    fit_fields_buf F; memset(&F, 0, sizeof F);
    if (fit_fields_load(argv[4], 0, 0, 1, &F)) { fprintf(stderr, "phase-near: cannot load the fields %s\n", argv[4]); return 1; }
    const int64_t o[3] = {bx[0], bx[1], bx[2]}, n[3] = {bx[3], bx[4], bx[5]}; const size_t N = (size_t)n[0] * n[1] * n[2];
    uint8_t *g = malloc(N); float *ph = malloc(N * 4), *cf = malloc(N * 4);
    if (!g || !ph || !cf) { fprintf(stderr, "phase-near: out of memory (%zu voxels)\n", N); return 1; }
    if (z3_read(s->phase, o, n, g, nth)) { fprintf(stderr, "phase-near: %s: GT phase read failed\n", s->name); return 1; }
    phase_near_sample(&F, o, n, s->um, ph, cf);
    const double um = s->um; const int many = strchr(cl, ',') || strchr(ml, ',');
    if (opt_atoi("--header", pn_opt(argc, argv, "--header", "0")))
        printf("tag\tsource\tbox\tknown\tgt_sheet\tpred_sheet\tpg_p50_um\tpg_p90_um\tpg_1\tpg_2\tpg_4\tgp_p50_um\tgp_p90_um\tgp_1\tgp_2\tgp_4\n");
    /* --conf C[,C..] x --min-comp N[,N..]: one row each on the one GT read (the tag gets -cC-mN when either is a list) */
    for (const char *cp = cl; *cp; ) { char *ce; const float cmin = strtof(cp, &ce); if (ce == cp) { fprintf(stderr, "phase-near: --conf C[,C...]\n"); return 2; }
      for (const char *mp = ml; *mp; ) { char *me; const long mc = strtol(mp, &me, 10); if (me == mp) { fprintf(stderr, "phase-near: --min-comp N[,N...]\n"); return 2; }
        phase_near_set_min_comp(mc);
        phase_near_res R;
        if (phase_near_measure(g, ph, cf, n, mg, cmin, nth, &R)) { fprintf(stderr, "phase-near: out of memory\n"); return 1; }
        char tg[512]; if (many) snprintf(tg, sizeof tg, "%s-c%g-m%ld", tag, cmin, mc); else snprintf(tg, sizeof tg, "%s", tag);
        printf("%s\t%s\t%s\t%.4f\t%zu\t%zu\t%.2f\t%.2f\t%.4f\t%.4f\t%.4f\t%.2f\t%.2f\t%.4f\t%.4f\t%.4f\n", tg, s->name, bs, R.known, R.gt_sheet, R.pred_sheet,
               R.pg50 * um, R.pg90 * um, R.pg_within[0], R.pg_within[1], R.pg_within[2], R.gp50 * um, R.gp90 * um, R.gp_within[0], R.gp_within[1], R.gp_within[2]);
        fflush(stdout); mp = *me == ',' ? me + 1 : me; if (*me && *me != ',') { fprintf(stderr, "phase-near: --min-comp N[,N...]\n"); return 2; } }
      cp = *ce == ',' ? ce + 1 : ce; if (*ce && *ce != ',') { fprintf(stderr, "phase-near: --conf C[,C...]\n"); return 2; } }
    free(g); free(ph); free(cf); fit_fields_free(&F); sources_free(S);
    return 0;
}
