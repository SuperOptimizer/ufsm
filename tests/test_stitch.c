/* Box stitching (src/stitch.c) on synthetic local fits: a gently curved stack of sheets T(p) (one turn per ~12 native voxels) seen
   by 2 x 2 x 2 overlapping boxes, each model W_i = s_i (T - c_i) with its own sign and integer gauge (stored the way ufsm fit
   stores a box: a far axis, a flat prior and the lattice delta). Checks: every overlap's (s, k) and agreement, the solved gauges
   (one global G = sigma T + const), an injected bad overlap reported as an inconsistent cycle, the blending tapers, the blended
   G, save / load, and a traced level set crossing the boxes (on the true sheet, all 8 boxes, no seam conflicts); a box with a
   sheet switch shows up as seam conflicts. */
#include "stitch.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

enum { EXT = 96, OV = 32, LAT = 4, HF = 4 };   /* box edge, overlap, lattice spacing (native), field cell */
static double T_true(double z, double y, double x) {
    return (0.15 * z + 0.6 * y + 0.78 * x) / 12.0 + 0.6 * sin(2 * M_PI * y / 180.0) + 0.3 * sin(2 * M_PI * z / 240.0);
}
static int g_switch = -1;   /* box index whose W jumps by one turn for box-local x > EXT / 2 (a sheet switch inside the box) */
static int g_slip = -1; static double g_T0;   /* box index that counts one sheet more beyond T = g_T0 (a slip between two sheets) */
static int g_fault = -1;   /* box index whose W climbs 3 turns across box-local x 20..26 (a fault: the fit crams turns into a thin zone) */
static fit_model *make_model(int bi, const double org[3], int s, int c) {
    fit_model *m = calloc(1, sizeof *m);
    m->dir = 1; m->nk = 2; m->knot_dz = EXT; m->pitch = malloc(8); m->offset = malloc(8);
    m->pitch[0] = m->pitch[1] = 1e7f; m->offset[0] = m->offset[1] = 0;
    m->L = LAT; m->lz = m->ly = m->lx = EXT / LAT + 1; m->D = m->H = m->W = EXT / HF; m->h = HF; m->um = 9.596;
    for (int d = 0; d < 3; d++) m->origin[d] = (long long)org[d];
    const int na = EXT + 1; m->cy = malloc(na * 4); m->cx = malloc(na * 4);
    for (int z = 0; z < na; z++) { m->cy[z] = -1500.f; m->cx[z] = -1800.f; }
    const size_t nl = (size_t)m->lz * m->ly * m->lx; m->delta = calloc(nl, 4);
    for (int a = 0; a < m->lz; a++) for (int b = 0; b < m->ly; b++) for (int e = 0; e < m->lx; e++) {
        const double z = a * LAT, y = b * LAT, x = e * LAT;
        double target = s * (T_true(z + org[0], y + org[1], x + org[2]) - c);
        if (bi == g_switch && x > EXT / 2) target += 1;
        if (bi == g_slip && T_true(z + org[0], y + org[1], x + org[2]) > g_T0) target += s;
        if (bi == g_fault) { const double t = fmin(1, fmax(0, (x - 20) / 6.0)); target += s * 3 * t * t * (3 - 2 * t); }
        const size_t i = ((size_t)a * m->ly + b) * m->lx + e;
        m->delta[i] = 0;
        m->delta[i] = (float)(target - fit_W(m, z, y, x));   /* W = prior + delta: the prior read with delta 0 at this node */
    }
    return m;
}
static int rd_fields(void *ctx, const int64_t o[3], const int64_t n[3], uint8_t *w, uint8_t *pc, uint8_t *ps) {
    (void)ctx; const size_t nv = (size_t)n[0] * n[1] * n[2];
    memset(w, 255, nv); if (pc) memset(pc, 255, nv); if (ps) memset(ps, 128, nv);
    (void)o; return 0;
}
/* the network phase of the true sheets: m = e^{2 pi i T} at the cell centres (confidence 1) */
static int rd_phase(void *ctx, const int64_t o[3], const int64_t n[3], uint8_t *w, uint8_t *pc, uint8_t *ps) {
    (void)ctx;
    for (int64_t a = 0; a < n[0]; a++) for (int64_t b = 0; b < n[1]; b++) for (int64_t c = 0; c < n[2]; c++) {
        const size_t i = ((size_t)a * n[1] + b) * n[2] + c; const double T = T_true((o[0] + a + 0.5) * HF, (o[1] + b + 0.5) * HF, (o[2] + c + 0.5) * HF);
        w[i] = 255; if (pc) pc[i] = (uint8_t)lround(128 + 127 * cos(2 * M_PI * T)); if (ps) ps[i] = (uint8_t)lround(128 + 127 * sin(2 * M_PI * T));
    }
    return 0;
}
static void build(st_box *b, const int *s, const int *c) {
    for (int i = 0; i < 8; i++) {
        const double org[3] = {(i >> 2 & 1) * (EXT - OV), (i >> 1 & 1) * (EXT - OV), (i & 1) * (EXT - OV)};
        memset(&b[i], 0, sizeof b[i]);
        b[i].fs = calloc(1, sizeof(fit_set)); b[i].fs->global = make_model(i, org, s[i], c[i]);
        for (int d = 0; d < 3; d++) { b[i].lo[d] = org[d]; b[i].hi[d] = org[d] + EXT; }
        snprintf(b[i].name, sizeof b[i].name, "b%d", i); b[i].s = 1; b[i].comp = -1;
        st_box_range(&b[i]);
    }
}
static int align_all(st_box *b, st_edge *e) {
    st_fields F = {HF, {0, 0, 0}, rd_fields, nullptr}; int ne = 0;
    for (int i = 0; i < 8; i++) for (int j = i + 1; j < 8; j++) if (!st_align_pair(b, i, j, &F, 1, 0.05, 2, &e[ne])) ne++;
    return ne;
}
int main(void) {
    const int s[8] = {1, -1, 1, 1, -1, -1, 1, -1}, c[8] = {0, 3, -2, 7, 11, -5, 4, 1};
    st_box b[8]; build(b, s, c);
    st_edge e[64]; const int ne = align_all(b, e);
    CHECK(ne == 28, "%d overlapping pairs (want 28: every pair of a 2 x 2 x 2 grid with overlaps overlaps)", ne);
    for (int i = 0; i < ne; i++) {
        const int a = e[i].a, bb = e[i].b, ws = s[a] * s[bb], wk = s[bb] * (c[a] - c[bb]);
        CHECK(e[i].s == ws && e[i].k == wk, "edge %d-%d: s %d k %d (want %d %d)", a, bb, e[i].s, e[i].k, ws, wk);
        CHECK(e[i].agree > 0.995 && e[i].tight > 0.99, "edge %d-%d: agreement %.4f tight %.4f", a, bb, e[i].agree, e[i].tight);
        CHECK(e[i].other < 0.6, "edge %d-%d: the other sign agrees %.3f", a, bb, e[i].other);
        CHECK(fabs(e[i].frac) < 0.02, "edge %d-%d: fractional offset %.4f", a, bb, e[i].frac);
    }
    int bad = st_solve(b, 8, e, ne, 0.5, 1, nullptr);
    CHECK(bad == 0, "%d inconsistent edges on clean data", bad);
    const int sig = b[0].s * s[0]; double k0 = b[0].g - sig * c[0];
    for (int i = 0; i < 8; i++) {
        CHECK(b[i].comp == 0, "box %d in component %d", i, b[i].comp);
        CHECK(b[i].s * s[i] == sig, "box %d: solved sign %d, true %d (global sign %d)", i, b[i].s, s[i], sig);
        CHECK(fabs(b[i].g - sig * c[i] - k0) < 1e-9, "box %d: gauge %g inconsistent (%g vs %g)", i, b[i].g, b[i].g - sig * c[i], k0);
    }
    /* quality maps: a box whose W grows with the network phase is coherent with it, a reversed box with the reversed phase */
    { const st_fields Fp = {HF, {0, 0, 0}, rd_phase, nullptr};
      for (int i = 0; i < 8; i++) {
          CHECK(!st_quality(&b[i], &Fp, 32, 1, 2), "st_quality box %d", i);
          const double want = s[i] > 0 ? b[i].qmean : b[i].qamean, other = s[i] > 0 ? b[i].qamean : b[i].qmean;
          CHECK(want > 0.95 && other < 0.3, "box %d (sign %d): coherence %.3f, reversed %.3f", i, s[i], b[i].qmean, b[i].qamean);
          const double pq[3] = {b[i].lo[0] + 40, b[i].lo[1] + 50, b[i].lo[2] + 60}, qv = st_box_q(&b[i], pq);
          CHECK(fabs(qv - (s[i] > 0 ? 1 : 0)) < 0.2, "box %d: q %.3f at a point", i, qv);
          free(b[i].q); free(b[i].qa); b[i].q = b[i].qa = nullptr;
      } }
    int ntree = 0; for (int i = 0; i < ne; i++) ntree += e[i].tree;
    CHECK(ntree == 7, "%d tree edges (want 7)", ntree);
    /* the tapers: interior faces ramp over the overlap, outer faces are hard */
    st_tapers(b, 8, 1);
    CHECK(b[0].tlo[0] == 0 && b[0].thi[0] == OV && b[7].tlo[2] == OV && b[7].thi[2] == 0, "tapers %g %g %g %g", b[0].tlo[0], b[0].thi[0], b[7].tlo[2], b[7].thi[2]);
    /* the blended G = sigma T + const everywhere (boxes agree) */
    double emax = 0; int cov = 0;
    for (int q = 0; q < 2000; q++) {
        const double p[3] = {(q * 37 % 160) + 0.3, (q * 53 % 160) + 0.7, (q * 71 % 160) + 0.1}; double G;
        if (!st_G(b, 8, 0, p, &G)) continue;
        cov++; emax = fmax(emax, fabs(G - (sig * T_true(p[0], p[1], p[2]) + k0)));
    }
    CHECK(cov > 1900 && emax < 0.02, "blended G: %d covered, max error %.4f turn", cov, emax);
    /* an overlap whose offset is off by one (and less trusted than the tree): one inconsistent cycle */
    st_edge e2[64]; memcpy(e2, e, sizeof e);
    int wrong = -1; for (int i = 0; i < ne; i++) if (!e2[i].tree) { wrong = i; break; }
    e2[wrong].k += 1; e2[wrong].agree = 0.7;
    bad = st_solve(b, 8, e2, ne, 0.5, 1, nullptr);
    CHECK(bad == 1 && e2[wrong].resid == 1 && !e2[wrong].tree, "a wrong overlap: %d inconsistent, resid %d, tree %d", bad, e2[wrong].resid, e2[wrong].tree);
    /* a sign conflict */
    memcpy(e2, e, sizeof e); e2[wrong].s = -e2[wrong].s; e2[wrong].agree = 0.7;
    bad = st_solve(b, 8, e2, ne, 0.5, 1, nullptr);
    CHECK(bad == 1 && e2[wrong].resid == 9999, "a wrong sign: %d inconsistent, resid %d", bad, e2[wrong].resid);
    /* an excluded box is its own component */
    uint8_t ex[8] = {0}; ex[3] = 1; memcpy(e2, e, sizeof e);
    bad = st_solve(b, 8, e2, ne, 0.5, 1, ex);
    CHECK(bad == 0 && b[3].comp == 1 && b[0].comp == 0, "excluded box 3: comp %d (box 0 comp %d), %d inconsistent", b[3].comp, b[0].comp, bad);
    bad = st_solve(b, 8, e, ne, 0.5, 1, nullptr);
    st_tapers(b, 8, 1);
    /* save / load: the boxes' models on disk */
    char dir[] = "/tmp/test_stitch_XXXXXX"; if (!mkdtemp(dir)) { perror("mkdtemp"); return 1; }
    for (int i = 0; i < 8; i++) { snprintf(b[i].dir, sizeof b[i].dir, "%s/b%d", dir, i); CHECK(!fit_save(b[i].fs->global, b[i].dir), "fit_save %s", b[i].dir); }
    char gp[256]; snprintf(gp, sizeof gp, "%s/gauge.json", dir);
    CHECK(!st_save(gp, b, 8, e, ne, 9.596), "st_save");
    st_box *lb; st_edge *le; int lnb, lne; double um;
    CHECK(!st_load(gp, &lb, &lnb, &le, &lne, &um), "st_load");
    if (!fails) {
        CHECK(lnb == 8 && lne == ne && fabs(um - 9.596) < 1e-9, "loaded %d boxes %d edges um %g", lnb, lne, um);
        for (int i = 0; i < 8; i++) CHECK(lb[i].s == b[i].s && lb[i].g == b[i].g && lb[i].comp == b[i].comp && !strcmp(lb[i].name, b[i].name) && lb[i].thi[0] == b[i].thi[0],
                                          "box %d after load", i);
        for (int i = 0; i < ne; i++) CHECK(le[i].k == e[i].k && le[i].s == e[i].s && le[i].tree == e[i].tree && fabs(le[i].agree - e[i].agree) < 1e-5, "edge %d after load", i);
        const double p[3] = {70, 80, 90}; double g1, g2; st_G(b, 8, 0, p, &g1); st_G(lb, lnb, 0, p, &g2);
        CHECK(fabs(g1 - g2) < 1e-4, "G after load %.6f vs %.6f", g2, g1);
        for (int i = 0; i < lnb; i++) st_box_free(&lb[i]);
        free(lb); free(le);
    }
    /* a level set across the boxes */
    const double kk = floor(sig * T_true(80, 80, 80) + k0);
    st_trace_cfg tc = {.tstep = 1.0, .max_cols = 400, .max_gap = 10, .bridge_max = 30, .fault_ratio = 3, .mask = nullptr, .mh = 1};
    st_sgauge gg = st_sgauge_new(8); st_sgauge_global(&gg, b, 0, kk);
    double seed[3]; CHECK(!st_seed(b, 0, 0.5, &gg, &tc, seed), "no seed of k %g in box 0", kk);
    st_sheet sh; CHECK(!st_trace(b, 8, 0, &tc, &gg, 0, seed, &sh), "trace failed");
    double terr = 0; long long nv = 0; double ymin = 1e9, ymax = -1e9;
    for (size_t v = 0; v < (size_t)sh.rows * sh.cols; v++) {
        if (sh.z[v] < 0) continue;
        nv++; terr = fmax(terr, fabs(sig * T_true(sh.z[v], sh.y[v], sh.x[v]) + k0 - kk)); ymin = fmin(ymin, sh.y[v]); ymax = fmax(ymax, sh.y[v]);
    }
    fprintf(stderr, "sheet k %g: %dx%d, %lld vertices, %d boxes, %lld seam (%lld conflict), max |T - k| %.4f, y %.0f..%.0f\n", kk, sh.rows, sh.cols, nv, sh.nboxes, sh.nseam, sh.nconflict, terr, ymin, ymax);
    CHECK(terr < 0.01, "traced vertices off the sheet: %.4f turn", terr);
    CHECK(sh.rows >= 150 && sh.nboxes >= 6 && nv > 15000, "sheet %d rows, %d boxes, %lld vertices", sh.rows, sh.nboxes, nv);
    CHECK(sh.nseam > 1000 && sh.nconflict == 0 && sh.nspread == 0, "seams %lld, spread %lld, conflicts %lld", sh.nseam, sh.nspread, sh.nconflict);
    const int nb0 = sh.nboxes; const long long nv0 = nv; st_sheet_free(&sh);
    /* follow mode from box 0's own level: the same sheet, the other boxes joining on contact at their own levels */
    const double w0 = st_box_W(&b[0], seed); st_sgauge_seed(&gg, 0, round(w0));
    CHECK(!st_trace(b, 8, -1, &tc, &gg, 1, seed, &sh), "follow trace failed");
    terr = 0; nv = 0;
    for (size_t v = 0; v < (size_t)sh.rows * sh.cols; v++) if (sh.z[v] >= 0) { nv++; terr = fmax(terr, fabs(sig * T_true(sh.z[v], sh.y[v], sh.x[v]) + k0 - kk)); }
    int joined = 0; for (int i = 0; i < 8; i++) joined += gg.on[i];
    const double tk0 = sig * (kk - k0);   /* the sheet's T */
    for (int i = 0; i < 8; i++) if (gg.on[i]) CHECK(fabs(gg.off[i] - s[i] * (tk0 - c[i])) < 1e-9 && gg.s[i] == s[i] * s[0], "follow: box %d joined at level %g sign %d", i, gg.off[i], gg.s[i]);
    fprintf(stderr, "follow sheet: %lld vertices, %d boxes joined (%lld contacts, %lld conflicting), max |T - k| %.4f\n", nv, joined, gg.contacts, gg.contact_conflicts, terr);
    CHECK(terr < 0.01 && llabs(nv - nv0) < nv0 / 50 && sh.nboxes == nb0 && sh.nconflict == 0 && gg.contact_conflicts == 0, "follow sheet: %lld vertices (gauge %lld), %d boxes, error %.4f", nv, nv0, sh.nboxes, terr);
    st_sheet_free(&sh); st_sgauge_free(&gg);
    /* a box with a sheet switch: its overlaps disagree in part, the sheet shows seam conflicts there */
    for (int i = 0; i < 8; i++) st_box_free(&b[i]);
    g_switch = 7; build(b, s, c);
    st_edge e3[64]; const int ne3 = align_all(b, e3);
    double amin = 1; for (int i = 0; i < ne3; i++) if (e3[i].a == 7 || e3[i].b == 7) amin = fmin(amin, e3[i].agree);
    CHECK(amin < 0.9, "the switched box still agrees %.3f with all its neighbours", amin);
    st_solve(b, 8, e3, ne3, 0.5, 1, nullptr); st_tapers(b, 8, 1);
    long long conf = 0; gg = st_sgauge_new(8);
    for (int zf = 1; zf <= 3; zf++) {
        const double k2 = floor(b[7].s * st_box_W(&b[7], (const double[]){150, 150, 150}) + b[7].g);
        st_sgauge_global(&gg, b, 0, k2);
        if (st_seed(b, 7, zf / 4.0, &gg, &tc, seed) || st_trace(b, 8, 0, &tc, &gg, 0, seed, &sh)) continue;
        conf += sh.nconflict; st_sheet_free(&sh);
    }
    CHECK(conf > 0, "no seam conflicts next to the switched box");
    for (int i = 0; i < 8; i++) st_box_free(&b[i]);
    /* a box that counts one sheet more beyond a slip between two sheets (parallel to them): its overlaps have two offsets, no
       constant gauge fits both sides; following sheets across the seams keeps each sheet whole on both sides of the slip */
    g_switch = -1; g_slip = 7; g_T0 = floor(T_true(110, 110, 110)) + 0.5; build(b, s, c);
    const int ne4 = align_all(b, e3); st_solve(b, 8, e3, ne4, 0.3, 1, nullptr); st_tapers(b, 8, 1);
    double a7 = 1;
    for (int i = 0; i < ne4; i++) if (e3[i].a == 7 || e3[i].b == 7) { a7 = fmin(a7, e3[i].agree); CHECK(e3[i].local > 0.85, "slip: overlap %d-%d local agreement %.3f", e3[i].a, e3[i].b, e3[i].local); }
    CHECK(a7 < 0.95, "slip: box 7 agrees %.3f everywhere", a7);
    long long cg = 0, cf = 0; int nfol = 0, ngau = 0;
    for (int dk = -3; dk <= 3; dk++) {
        const double tk = g_T0 - 0.5 + dk;   /* sheets on both sides of the slip */
        st_sgauge_global(&gg, b, 0, b[3].s * (s[3] * (tk - c[3])) + b[3].g);   /* gauge mode */
        if (!st_seed(b, 3, 0.5, &gg, &tc, seed) && !st_trace(b, 8, 0, &tc, &gg, 0, seed, &sh)) { if (sh.own[7]) { ngau++; cg += sh.nconflict + sh.nspread; } st_sheet_free(&sh); }
        st_sgauge_seed(&gg, 3, s[3] * (tk - c[3]));   /* follow mode from box 3 */
        if (!st_seed(b, 3, 0.5, &gg, &tc, seed) && !st_trace(b, 8, -1, &tc, &gg, 1, seed, &sh)) {
            if (sh.own[7]) { nfol++; cf += sh.nconflict + sh.nspread; }
            double te = 0; for (size_t v = 0; v < (size_t)sh.rows * sh.cols; v++) if (sh.z[v] >= 0) te = fmax(te, fabs(T_true(sh.z[v], sh.y[v], sh.x[v]) - tk));
            CHECK(te < 0.01, "slip follow T %g: off the sheet by %.3f", tk, te);
            st_sheet_free(&sh);
        }
    }
    fprintf(stderr, "slip: gauge mode %lld seam spread/conflict vertices (%d sheets through box 7), follow mode %lld (%d sheets)\n", cg, ngau, cf, nfol);
    CHECK(nfol >= 3 && cf == 0 && cg > 0, "slip: follow %d sheets %lld bad seam vertices, gauge %lld", nfol, cf, cg);
    st_sgauge_free(&gg);
    for (int i = 0; i < 8; i++) st_box_free(&b[i]);
    /* a fault in box 0 (3 turns crammed into 6 voxels across the sheets): follow mode bridges it and stays on the sheet beyond it */
    g_slip = -1; g_fault = 0; build(b, s, c);
    { const int nf = align_all(b, e3); st_solve(b, 8, e3, nf, 0.3, 1, nullptr); }
    for (int i = 0; i < 8; i++) b[i].comp = 0;
    st_tapers(b, 8, 0);
    gg = st_sgauge_new(8); int nfs = 0; long long nbeyond_all = 0;
    for (int dk = 0; dk < 2; dk++) {   /* the two sheets that cross the fault inside box 0 */
        const double tk = floor(T_true(48, 48, 48)) - 2 + dk;
        const double p0[3] = {48, 48, 48}; const double lv = round(st_box_W(&b[0], p0) - s[0] * (T_true(48, 48, 48) - tk));
        st_sgauge_seed(&gg, 0, lv);
        int sd = 1; for (int q = 0; q < 3 && sd; q++) sd = st_seed(b, 0, (const double[]){0.5, 0.25, 0.75}[q], &gg, &tc, seed);   /* calm seeds only */
        if (sd || st_trace(b, 8, -1, &tc, &gg, 1, seed, &sh)) continue;
        double te = 0; long long nbeyond = 0, nv2 = 0, nunsure = 0;
        for (size_t v = 0; v < (size_t)sh.rows * sh.cols; v++) {
            if (sh.z[v] < 0 || sh.seam[v] == 5) continue;
            if (sh.seam[v] >= 3) { nunsure++; continue; }   /* flagged: the boxes disagree there */
            nv2++; te = fmax(te, fabs(T_true(sh.z[v], sh.y[v], sh.x[v]) - tk)); if (sh.x[v] < 18 && sh.y[v] < 60 && sh.z[v] < 60) nbeyond++;
        }
        fprintf(stderr, "fault: sheet T %g: %lld vertices, %lld beyond the fault, %lld bridges (%lld bridge vertices), %lld re-gauges, max |T - k| %.4f\n", tk, nv2, nbeyond,
                sh.bridges, sh.nbridge, sh.regauges, te);
        CHECK(te < 0.25 && sh.bridges > 0 && nunsure < nv2 / 20, "fault: sheet T %g error %.3f, %lld bridges, %lld flagged", tk, te, sh.bridges, nunsure);
        nfs++; nbeyond_all += nbeyond; st_sheet_free(&sh);
    }
    CHECK(nfs == 2 && nbeyond_all > 300, "fault: %d sheets traced, %lld vertices beyond the fault", nfs, nbeyond_all);
    st_sgauge_free(&gg);
    for (int i = 0; i < 8; i++) st_box_free(&b[i]);
    char rm[300]; snprintf(rm, sizeof rm, "rm -rf '%s'", dir); if (system(rm)) fprintf(stderr, "cannot remove %s\n", dir);
    if (fails) { fprintf(stderr, "test_stitch: %d failures\n", fails); return 1; }
    fprintf(stderr, "test_stitch: ok\n");
    return 0;
}
