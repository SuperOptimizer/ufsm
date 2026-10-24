/* --aug-crush (src/crush.h) on the synthetic spiral of tests/make_wind_fixture.c (analytic winding fx_wind):
   - the map: push(pull(u)) = u, Jacobian determinant > 0 everywhere (= k F', F' in [1, 1 / 0.05]) and equal to the finite-difference
     determinant, C1 (the Jacobian has no jumps), voids compressed (F' >> 1) while papyrus keeps its thickness (F' ~ 1)
   - normals: J^T n_src (the pull's transpose) = the gradient of the pulled winding
   - the window pull: pulled phase codes = frac(W(pull(u))) (analytic, the label grid), pulled CT papyrus / void = the analytic sheets,
     the gradient of the pulled phase along J^T n_src; a 2-levels-finer box pull of the identity map = the stored mean pyramid
   - the sampler (task unwrap2 settings): crushed windows drawn and deterministic; --aug-crush 0 with every other crush parameter
     changed gives bit-identical batches (the default-off path draws nothing)
   test_crush [DIR] */
#define WIND_FIXTURE_NO_MAIN
#include "make_wind_fixture.c"
#include "../src/sample.c"
#include <assert.h>

void *nn_host_alloc(size_t n) { return malloc(n); }
void nn_host_free(void *p) { free(p); }
static int failed;
static void check(const char *name, int ok) { printf("crush: %-96s %s\n", name, ok ? "ok" : "FAIL"); failed += !ok; }
static int cmpd(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
static double pct(double *v, int n, double q) { qsort(v, (size_t)n, sizeof *v, cmpd); return n ? v[(int)(q * (n - 1))] : 1e9; }

/* the analytic scroll at level L (window origin o, voxel j of level L at level-0 2^L j + (2^L - 1) / 2) */
typedef struct { int64_t o[3]; int L; } an_ctx;
static void an_l0(const an_ctx *a, const double x[3], double g[3]) { const double f = 1 << a->L; for (int d = 0; d < 3; d++) g[d] = f * ((double)a->o[d] + x[d]) + .5 * (f - 1); }
static double an_ct(void *ud, const double x[3]) {
    double g[3]; an_l0(ud, x, g);
    double cy, cx; fx_axis_at(g[0], &cy, &cx);
    if (hypot(g[1] - cy, g[2] - cx) > .47 * fx_N) return 0;
    const double w = fx_wind(g[0], g[1], g[2]); return w - floor(w) < .3 ? 150 : 45;
}
static double an_w(const an_ctx *a, const double x[3]) { double g[3]; an_l0(a, x, g); return fx_wind(g[0], g[1], g[2]); }
static void an_grad(const an_ctx *a, const double x[3], double gr[3]) {
    for (int d = 0; d < 3; d++) { double p[3] = {x[0], x[1], x[2]}, q[3] = {x[0], x[1], x[2]}; p[d] += 1e-3; q[d] -= 1e-3; gr[d] = (an_w(a, p) - an_w(a, q)) / 2e-3; }
}
static double ang(const double a[3], const double b[3]) {
    const double d = fabs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / sqrt((a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) * (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]) + 1e-300);
    return acos(d > 1 ? 1 : d) * 180 / M_PI;
}
static double det3(double J[3][3]) { return J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]); }
static double urand(uint64_t *s) { *s = *s * 6364136223846793005ull + 1442695040888963407ull; return (double)(*s >> 11) * 0x1.0p-53; }

static void map_tests(void) {
    const int P = 96; int inv = 1, pos = 1, detok = 1, c1 = 1, comp = 1, nrm = 1, foot = 1, any_col = 0; double worst_inv = 0, worst_c1 = 0, worst_n = 0, worst_det = 0;
    for (int t = 0; t < 12; t++) {
        crush_cfg cfg = crush_cfg_default(); cfg.p = 1;
        cfg.squash_p = t % 3 == 0 ? 0 : 1; cfg.bend_p = t % 2 ? 1 : 0; cfg.kink_p = t % 4 == 1 ? 1 : 0; cfg.jlo = .1f; cfg.jhi = .3f;
        an_ctx a = {{980 + 7 * t, 1620 + 3 * t, 1000 + 5 * t}, 0};   /* ~600 voxels off the axis (a real scroll's sheet curvature) */
        const double cw = .5 * (P - 1), cz = (double)a.o[0] + cw; double cy, cx; fx_axis_at(cz, &cy, &cx);
        const double n[3] = {0, (double)a.o[1] + cw - cy, (double)a.o[2] + cw - cx};
        crush_map m; crush_map_init(&m, &cfg, 1000 + t, P, n);
        any_col += !crush_map_columns(&m, cfg.jlo, cfg.jhi, 77 + t, an_ct, &a);
        uint64_t rs = 12345 + t; double sv = 0, sp_ = 0; int nv = 0, np = 0;
        for (int k = 0; k < 400; k++) {
            const double u[3] = {urand(&rs) * (P - 1), urand(&rs) * (P - 1), urand(&rs) * (P - 1)}; double x[3], v[3], jd, J[3][3];
            crush_pull(&m, u, x, &jd); crush_push(&m, x, v);
            const double e = fabs(v[0] - u[0]) + fabs(v[1] - u[1]) + fabs(v[2] - u[2]); worst_inv = fmax(worst_inv, e); inv &= e < 1e-6;
            crush_jacobian(&m, u, J); const double dj = det3(J);
            pos &= jd > 0 && jd <= m.det / .05 * 1.5; detok &= fabs(dj - jd) < 1e-3 * jd; worst_det = fmax(worst_det, fabs(dj - jd) / jd);
            double ext[3]; crush_footprint(&m, ext); for (int d = 0; d < 3; d++) foot &= fabs(x[d] - m.c) <= ext[d];
            /* C1: the Jacobian 1e-3 voxels away differs by O(1e-3) (a jump would be O(1)) */
            double u2[3], J2[3][3], dmax = 0; for (int d = 0; d < 3; d++) u2[d] = u[d] + (urand(&rs) - .5) * 2e-3;
            crush_jacobian(&m, u2, J2); for (int i = 0; i < 9; i++) dmax = fmax(dmax, fabs(J2[i / 3][i % 3] - J[i / 3][i % 3]));
            worst_c1 = fmax(worst_c1, dmax); c1 &= dmax < .05;
            /* the column compression where the source is void vs papyrus */
            const double ct = an_ct(&a, x), fp = jd / m.det;
            if (ct == 45) { sv += fp; nv++; } else if (ct == 150) { sp_ += fp; np++; }
            /* normals: the gradient of W(pull(u)) = J^T grad W */
            double gs[3], go[3], gf[3]; an_grad(&a, x, gs);
            for (int j = 0; j < 3; j++) go[j] = J[0][j] * gs[0] + J[1][j] * gs[1] + J[2][j] * gs[2];
            for (int d = 0; d < 3; d++) { double p[3] = {u[0], u[1], u[2]}, q[3] = {u[0], u[1], u[2]}, xp[3], xq[3]; p[d] += 1e-3; q[d] -= 1e-3; crush_pull(&m, p, xp, nullptr); crush_pull(&m, q, xq, nullptr); gf[d] = (an_w(&a, xp) - an_w(&a, xq)) / 2e-3; }
            const double an = ang(go, gf); worst_n = fmax(worst_n, an); nrm &= an < .5;
        }
        if (m.tab && nv > 20 && np > 20) {
            const double mv = sv / nv, mp = sp_ / np;
            comp &= mv > 1.6 * mp && mp < 1.6;
            if (t < 4) printf("crush: map %d: k %.2f, bends %d, kink %d, j0 %.2f: F' void %.2f papyrus %.2f\n", t, m.sq_k, m.nsin, m.kink, m.j0, mv, mp);
        }
        crush_map_free(&m);
    }
    printf("crush: worst |push(pull(u)) - u| %.2e, worst Jacobian change over 1e-3 voxels %.2e, worst normal angle %.3f deg, worst det error %.2e\n", worst_inv, worst_c1, worst_n, worst_det);
    check("columns built (the spiral window splits into papyrus / void)", any_col == 12);
    check("invertible: push(pull(u)) = u (1e-6 voxels)", inv);
    check("Jacobian determinant k F' > 0 (and < 1.5 k / 0.05), = det of the finite-difference Jacobian (1e-3)", pos && detok);
    check("every pull inside crush_footprint (the holdout pre-check box)", foot);
    check("C1: no Jacobian jumps", c1);
    check("voids compressed (F' > 1.6 x papyrus), papyrus keeps its thickness (F' < 1.6)", comp);
    check("normals: J^T grad W = grad of the pulled winding (< 0.5 deg)", nrm);
}

/* the fraction of output voxels whose CT says papyrus (> 97) where the analytic sheet at pull(u) is papyrus core (f .08 .. .22), and void
   (1 .. 97) where it is void core (f .45 .. .85): ag[0], ag[1] */
static void cls_agree(const crush_map *m, const uint8_t *ct, const an_ctx *a, int P, double ag[2]) {
    size_t n[2] = {0, 0}, ok[2] = {0, 0};
    for (int z = 1; z < P - 1; z++) for (int y = 1; y < P - 1; y++) for (int x = 1; x < P - 1; x++) {
        const size_t k = ((size_t)z * P + y) * P + x; const int uu[3] = {z, y, x}; double xs[3]; crush_pull_window(m, 0, uu, xs);
        int in = an_ct((void *)a, xs) > 0;   /* (inside the scan and the level-1 array: crushed windows reach far past their own box) */
        for (int d = 0; d < 3; d++) { const double g = (double)a->o[d] + xs[d]; in &= g > 1 && g < (fx_N >> a->L) - 2; }
        if (!in) continue;
        const double w = an_w(a, xs), f = w - floor(w);
        if (f > .08 && f < .22) { n[0]++; ok[0] += ct[k] > 97; } else if (f > .45 && f < .85) { n[1]++; ok[1] += ct[k] <= 97 && ct[k] > 0; }
    }
    for (int c = 0; c < 2; c++) ag[c] = n[c] ? (double)ok[c] / n[c] : 0;
}
static void window_tests(const char *dir) {
    char path[1400]; snprintf(path, sizeof path, "%s/sources.json", dir);
    sources *S = sources_load(path); assert(S);
    source *s = &S->src[0];
    z3 *ct1 = source_ct(s, 1), *ct0 = source_ct(s, 0); assert(ct1 && ct0 && s->phase && s->phase_level == 1);
    const int P = 64; const size_t p3 = (size_t)P * P * P;
    uint8_t *arena = malloc((size_t)40 << 21), *oct = malloc(p3), *oph = malloc(p3), *ofn = malloc(p3), *ref = malloc(p3);
    crush_cache cc; assert(arena && !crush_cache_init(&cc, arena, (size_t)40 << 21));
    int lab = 1, ctok = 1, grad = 1, fineok = 1, ncr = 0, lneg = 0; double ctd = 0, latd = 0, lxdev = 0;
    for (int t = 0; t < 4; t++) {
        const int64_t o[3] = {20 + 5 * t, 18 + 4 * t, 30 - 3 * t};   /* level-1 window (the fixture's level 1 is 128^3) */
        an_ctx a = {{o[0], o[1], o[2]}, 1};
        crush_layer Lct = {ct1, CRUSH_LINEAR, 0, 0, 1, {0, 0, 0}, oct}, Lph = {s->phase, CRUSH_PHASE, 0, PHASE_UNKNOWN, 1, {0, 0, 0}, oph};
        crush_layer_phase(&Lph, 0);
        double n[3]; const int ng = crush_phase_normal(&cc, &Lph, o, P, nullptr, n);
        crush_cfg cfg = crush_cfg_default(); cfg.p = 1; cfg.squash_p = t & 1; cfg.bend_p = 1; cfg.kink_p = t == 2; cfg.klo = 1.5f; cfg.khi = 2;
        crush_map m; crush_map_init(&m, &cfg, 31 + t, P, n);
        crush_ctu u = {&cc, &Lct, o};
        const int cols = crush_map_columns(&m, cfg.jlo, cfg.jhi, 5 + t, crush_ct_at, &u);
        crush_layer L[2] = {Lct, Lph};
        assert(!crush_window(&m, &cc, o, L, 2));   /* (the default: the map every 2 voxels, trilinear between) */
        {   /* the exact map (lat 1) into ofn: the lattice's CT difference */
            m.lat = 1; crush_layer E = Lct; E.out = ofn; assert(!crush_window(&m, &cc, o, &E, 1)); m.lat = 2;
            double sd = 0; for (size_t k = 0; k < p3; k++) sd += fabs((double)ofn[k] - oct[k]);
            latd = fmax(latd, sd / p3);
        }
        double *ep = malloc(p3 * sizeof *ep), *ea = malloc(p3 * sizeof *ea); int ne = 0, na = 0;
        /* CT papyrus / void at the analytic sheets of pull(u), per class, vs the uncrushed window (the same CT pulled by the identity map) */
        double ag[2], ag0[2]; cls_agree(&m, oct, &a, P, ag);
        {
            crush_cfg id = crush_cfg_default(); id.squash_p = id.bend_p = id.kink_p = 0; id.rotate_deg = 0;
            crush_map mi; crush_map_init(&mi, &id, 1, P, n);
            crush_layer R = {ct1, CRUSH_LINEAR, 0, 0, 1, {0, 0, 0}, ref}; assert(!crush_window(&mi, &cc, o, &R, 1));
            cls_agree(&mi, ref, &a, P, ag0); crush_map_free(&mi);
        }
        int detneg = 0; double xdev = 0;
        for (int z = 1; z < P - 1; z++) for (int y = 1; y < P - 1; y++) for (int x = 1; x < P - 1; x++) {
            const size_t k = ((size_t)z * P + y) * P + x; const int uu[3] = {z, y, x}; double xs[3]; crush_pull_window(&m, 0, uu, xs);
            const double w = an_w(&a, xs), f = w - floor(w);
            if (oph[k] < PHASE_PERIOD) { double d = oph[k] - f * PHASE_PERIOD; d -= PHASE_PERIOD * floor(d / PHASE_PERIOD + .5); ep[ne++] = fabs(d); }
            if ((x & 1) == 0) {   /* the blocked window pull = the point pull of the same coordinates (CT and labels share them) */
                double p[3]; for (int d = 0; d < 3; d++) p[d] = Lct.a * ((double)o[d] + xs[d]) + Lct.b[d];
                const double cv = crush_linear_at(&cc, &Lct, p); ctd = fmax(ctd, fabs(cv - oct[k]));
            }
            if (z & y & x & 1) {   /* the lattice map's deviation from the exact one at cell centres */
                const double ud[3] = {z, y, x}; double xe[3]; crush_pull(&m, ud, xe, nullptr);
                xdev = fmax(xdev, fabs(xe[0] - xs[0]) + fabs(xe[1] - xs[1]) + fabs(xe[2] - xs[2]));
            }
            if ((z & 1) == 0 && (y & 1) == 0 && (x & 1) == 0) {   /* its cell Jacobians */
                double J[3][3];
                for (int j = 0; j < 3; j++) { int up[3] = {z, y, x}; up[j]++; double xp[3]; crush_pull_window(&m, 0, up, xp); for (int i = 0; i < 3; i++) J[i][j] = xp[i] - xs[i]; }
                detneg += !(det3(J) > 0);
            }
            if ((z & 3) == 0 && (y & 3) == 0 && (x & 3) == 0) {   /* the pulled phase's gradient vs J^T grad W (J of the window's map) */
                const uint8_t *q = oph; const size_t sd[3] = {(size_t)P * P, (size_t)P, 1}; double gp[3]; int ok = 1;
                for (int d = 0; d < 3 && ok; d++) {
                    const int c0 = q[k - sd[d]], c1 = q[k + sd[d]]; if (c0 >= PHASE_PERIOD || c1 >= PHASE_PERIOD) { ok = 0; break; }
                    int dd = c1 - c0; if (dd > PHASE_PERIOD / 2) dd -= PHASE_PERIOD; if (dd < -PHASE_PERIOD / 2) dd += PHASE_PERIOD; gp[d] = dd;
                }
                if (ok) {
                    double J[3][3], gs[3], go[3]; an_grad(&a, xs, gs);
                    for (int j = 0; j < 3; j++) {
                        int up[3] = {z, y, x}, um[3] = {z, y, x}; up[j]++; um[j]--; double xp[3], xm[3];
                        crush_pull_window(&m, 0, up, xp); crush_pull_window(&m, 0, um, xm); for (int i = 0; i < 3; i++) J[i][j] = .5 * (xp[i] - xm[i]);
                    }
                    for (int j = 0; j < 3; j++) go[j] = J[0][j] * gs[0] + J[1][j] * gs[1] + J[2][j] * gs[2];
                    ea[na++] = ang(go, gp);
                }
            }
        }
        lneg += detneg; lxdev = fmax(lxdev, xdev);
        const double p50 = pct(ep, ne, .5), p95 = pct(ep, ne, .95), a50 = pct(ea, na, .5);
        printf("crush: window %d: normal from %d phase samples, columns %s, k %.2f: phase error p50 %.2f p95 %.2f codes (%d known), CT agreement papyrus %.3f void %.3f (uncrushed %.3f %.3f), phase-gradient angle p50 %.1f deg\n",
               t, ng, cols ? "off" : "on", m.sq_k, p50, p95, ne, ag[0], ag[1], ag0[0], ag0[1], a50);
        lab &= ne > (int)(p3 / 8) && p50 < 2 && p95 < 6; ctok &= ag[0] > ag0[0] - .05 && ag[1] > ag0[1] - .02 && ag0[1] > .9; grad &= na > 100 && a50 < 8; ncr += !cols;
        free(ep); free(ea);
        crush_map_free(&m);
        /* a 1-level-finer box pull of the identity map = the stored mean pyramid (lossy codec: small differences) */
        if (t == 0) {
            crush_cfg id = crush_cfg_default(); id.squash_p = id.bend_p = id.kink_p = 0; id.rotate_deg = 0;
            crush_map mi; crush_map_init(&mi, &id, 1, P, n);
            crush_layer F = {ct0, CRUSH_LINEAR, 1, 0, 1, {0, 0, 0}, ofn}; crush_layer_level(&F, 1);
            crush_layer R = {ct1, CRUSH_LINEAR, 0, 0, 1, {0, 0, 0}, ref};
            crush_layer LL[2] = {F, R}; assert(!crush_window(&mi, &cc, o, LL, 2));
            double sd = 0, sx = 0;
            for (size_t k = 0; k < p3; k++) sd += fabs((double)ofn[k] - ref[k]);
            F.b[0] += 1; F.b[1] += 1; LL[0] = F; assert(!crush_window(&mi, &cc, o, LL, 2));   /* control: one level-0 voxel off */
            for (size_t k = 0; k < p3; k++) sx += fabs((double)ofn[k] - ref[k]);
            printf("crush: identity map, 2^3 box from level 0 vs the stored level 1: mean |diff| %.3f (shifted 1 level-0 voxel: %.3f)\n", sd / p3, sx / p3);
            fineok = sd / p3 < 4 && sd < .6 * sx;
        }
    }
    printf("crush: %llu tiles decoded\n", (unsigned long long)cc.decoded);
    check("pulled phase codes = frac(W(window pull(u))) (p50 < 2, p95 < 6 codes)", lab);
    printf("crush: blocked window vs point pull: max |CT diff| %.2f; the 2-voxel map lattice vs the exact map: max |x diff| %.2f voxels, worst window mean |CT diff| %.2f, %d cells with det <= 0\n",
           ctd, lxdev, latd, lneg);
    check("2-voxel map lattice: invertible cells (det > 0)", lneg == 0);
    check("pulled CT papyrus / void at the analytic sheets of pull(u) as in the uncrushed window (lossy noisy CT, 3.6 voxel sheets)", ctok);
    check("blocked window CT = the point pull at the same coordinates (rounding)", ctd < .51);
    check("pulled phase gradient along J^T grad W (p50 < 8 deg)", grad);
    check("collapse columns built in every window", ncr == 4);
    check("finer-level 2^3 box mean of the identity map = the stored pyramid level", fineok);
    crush_cache_free(&cc); free(arena); free(oct); free(oph); free(ofn); free(ref);
    sources_free(S);
}

enum { NB = 6 };
typedef struct { uint16_t *x; uint8_t *side, *w4; } snap;
static uint64_t take(sampler *sp, snap *o, size_t p3) {
    for (int i = 0; i < NB; i++) {
        batch *b = sampler_next(sp); assert(b);
        o[i].x = malloc(p3 * 2); memcpy(o[i].x, b->x16, p3 * 2);
        o[i].side = malloc(p3); memcpy(o[i].side, b->side, p3); o[i].w4 = malloc(p3); memcpy(o[i].w4, b->w4, p3);
        sampler_release(sp, b);
    }
    return sampler_crushed(sp);
}
static int same(const snap *a, const snap *b, size_t p3) {
    int ok = 1; for (int i = 0; i < NB; i++) ok &= !memcmp(a[i].x, b[i].x, p3 * 2) && !memcmp(a[i].side, b[i].side, p3) && !memcmp(a[i].w4, b[i].w4, p3);
    return ok;
}
static void drop(snap *o) { for (int i = 0; i < NB; i++) { free(o[i].x); free(o[i].side); free(o[i].w4); } }
static void sampler_tests(const char *dir) {
    char path[1400]; snprintf(path, sizeof path, "%s/sources.json", dir);
    sources *S = sources_load(path); assert(S);
    const int PW = 64; const size_t p3 = (size_t)PW * PW * PW;
    sample_cfg c = sample_cfg_default();
    c.P = PW; c.B = 1; c.nworkers = 2; c.nbuf = 3; c.deterministic = 1; c.seed = 9; c.xfmt = 1; c.gpu_input = 1; c.side = 2; c.wind4 = 1;
    memset(c.level_p, 0, sizeof c.level_p); c.level_p[1] = 1; c.min_fg = 0;
    assert(!sample_aug_preset(&c, "all"));
    c.augment = 1; c.zoom_out_p = 0;
    snap a[NB], b[NB], d[NB], e[NB];
    sampler *sp = sampler_start(S, &c); assert(sp); const uint64_t n0 = take(sp, a, p3); sampler_stop(sp);
    sample_cfg c2 = c; c2.crush.jlo = .3f; c2.crush.jhi = .9f; c2.crush.squash_p = 1; c2.crush.klo = 2; c2.crush.khi = 4; c2.crush.bend_p = 0; c2.crush.kink_p = 1; c2.crush.fine = 1; c2.crush.rotate_deg = 30;
    sp = sampler_start(S, &c2); assert(sp); take(sp, b, p3); sampler_stop(sp);
    check("--aug-crush 0: other crush parameters change nothing (bit-identical batches)", n0 == 0 && same(a, b, p3));
    sample_cfg c3 = c; c3.crush.p = 1;
    sp = sampler_start(S, &c3); assert(sp); const uint64_t n3 = take(sp, d, p3); sampler_stop(sp);
    sp = sampler_start(S, &c3); assert(sp); take(sp, e, p3); sampler_stop(sp);
    size_t kn = 0, kw = 0; for (int i = 0; i < NB; i++) for (size_t k = 0; k < p3; k++) { kn += d[i].side[k] < PHASE_PERIOD; kw += d[i].w4[k] < PHASE_PERIOD; }
    printf("crush: sampler: %llu of %d windows crushed; known phase %.3f, winding mod 4 %.3f of the voxels\n", (unsigned long long)n3, NB, kn / (double)(NB * p3), kw / (double)(NB * p3));
    check("--aug-crush 1: crushed windows with phase and winding labels, deterministic", n3 >= NB && kn > NB * p3 / 10 && kw > NB * p3 / 20 && same(d, e, p3) && !same(a, d, p3));
    c3.crush.fine = 1;
    sp = sampler_start(S, &c3); assert(sp); const uint64_t n4 = take(sp, e, p3); sampler_stop(sp);
    check("--aug-crush-fine 1 (CT from level 0): crushed windows drawn", n4 >= NB);
    drop(a); drop(b); drop(d); drop(e);
    sources_free(S);
}

int main(int argc, char **argv) {
    char tmpd[] = "/tmp/ufsm-crush-XXXXXX"; const char *dir = argc > 1 ? argv[1] : mkdtemp(tmpd);
    fx_N = 2048;   /* the analytic scroll of the map tests (the fixture resets it) */
    map_tests();
    if (!dir || make_wind_fixture(dir, 256, 2.f)) { fprintf(stderr, "fixture failed\n"); return 1; }
    window_tests(dir);
    sampler_tests(dir);
    if (argc < 2) { char cmd[1500]; snprintf(cmd, sizeof cmd, "rm -rf '%s'", dir); if (system(cmd)) {} }
    printf(failed ? "crush: %d FAILED\n" : "crush: all passed\n", failed);
    return failed != 0;
}
