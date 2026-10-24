/* Scroll-specific augmentations of the sampler on the synthetic spiral of tests/make_wind_fixture.c (task unwrap2 settings):
   --aug-xscan (registered partner scans, src/view.h sview_read_level):
     - an integer translation map: the partner read = the CT read at the shifted origin, bit for bit; a box half outside the partner reads 0 there
     - default off: sources with "xscan" and --aug-xscan 0 give the batches of sources without it (bit-identical)
     - p 1: every window swapped; an identity map gives the plain windows' CT through the quantile match (identity) and the same labels;
       a translated partner changes the CT but keeps the labels where both windows have scan data
   test_scrollaug [DIR] */
#define WIND_FIXTURE_NO_MAIN
#include "make_wind_fixture.c"
#include "../src/sample.c"
#include <assert.h>

void *nn_host_alloc(size_t n) { return malloc(n); }
void nn_host_free(void *p) { free(p); }

static int failed;
static void check(const char *name, int ok) { printf("scrollaug: %-100s %s\n", name, ok ? "ok" : "FAIL"); failed += !ok; }

/* a ufsm reg map JSON: coarse = fine = the fixture CT (9.6 um), A = identity + translation t (xyz, level-0 voxels: c = f + t) */
static void write_map(const char *dir, const char *name, const double t[3]) {
    char p[1400]; snprintf(p, sizeof p, "%s/%s", dir, name); FILE *f = fopen(p, "w"); assert(f);
    fprintf(f, "{\"name\": \"fixture\", \"coarse\": {\"root\": \"%s\", \"key\": \"ct\", \"um\": 9.6}, \"fine\": {\"root\": \"%s\", \"key\": \"ct\", \"um\": 9.6},\n"
               " \"A\": [[1, 0, 0, %g], [0, 1, 0, %g], [0, 0, 1, %g]]}\n", dir, dir, t[0], t[1], t[2]);
    fclose(f);
}
/* sources.json of the fixture with "xscan" maps (nullptr: none) */
static void write_sources(const char *dir, const char *name, const char *map) {
    char p[1400]; snprintf(p, sizeof p, "%s/%s", dir, name); FILE *f = fopen(p, "w"); assert(f);
    fprintf(f, "{\"sources\": [{\"name\": \"wind-fixture\", \"root\": \"%s\", \"ct\": \"ct\", \"um\": 9.6, \"axis\": \"%s/axis.json\", \"weight\": 1,\n", dir, dir);
    if (map) fprintf(f, "  \"xscan\": [{\"map\": \"%s/%s\"}],\n", dir, map);
    fprintf(f, "  \"targets\": {\"band\": {\"root\": \"%s/band\", \"key\": \"1\", \"radius\": 80, \"span\": 75, \"phase\": {\"root\": \"%s/phase\", \"key\": \"1\", \"level\": 1}}}}]}\n", dir, dir);
    fclose(f);
}

enum { NB = 6 };
typedef struct { uint16_t *x; uint8_t *side, *w4, *wc; } snap;
static void take(sampler *sp, snap *o, size_t p3) {
    for (int i = 0; i < NB; i++) {
        batch *b = sampler_next(sp); assert(b);
        o[i].x = malloc(p3 * 2); memcpy(o[i].x, b->x16, p3 * 2);
        o[i].side = malloc(p3); memcpy(o[i].side, b->side, p3); o[i].w4 = malloc(p3); memcpy(o[i].w4, b->w4, p3); o[i].wc = malloc(p3); memcpy(o[i].wc, b->wcomp, p3);
        sampler_release(sp, b);
    }
}
static int same(const snap *a, const snap *b, size_t p3) {
    int ok = 1; for (int i = 0; i < NB; i++) ok &= !memcmp(a[i].x, b[i].x, p3 * 2) && !memcmp(a[i].side, b[i].side, p3) && !memcmp(a[i].w4, b[i].w4, p3);
    return ok;
}
static void drop(snap *o) { for (int i = 0; i < NB; i++) { free(o[i].x); free(o[i].side); free(o[i].w4); free(o[i].wc); } }
static sample_cfg base_cfg(int PW) {
    sample_cfg c = sample_cfg_default();
    c.P = PW; c.B = 1; c.nworkers = 2; c.nbuf = 3; c.deterministic = 1; c.seed = 9; c.xfmt = 1; c.gpu_input = 1; c.side = 2; c.wind4 = 1;
    memset(c.level_p, 0, sizeof c.level_p); c.level_p[1] = 1; c.min_fg = 0;
    assert(!sample_aug_preset(&c, "all"));
    c.augment = 1; c.zoom_out_p = 0;
    return c;
}

static void xscan_tests(const char *dir) {
    const double t0[3] = {0, 0, 0}, ts[3] = {6, -2, 4};   /* (xyz, level 0: level-1 shifts 3, -1, 2) */
    write_map(dir, "map-id.json", t0); write_map(dir, "map-sh.json", ts);
    char p[1400];
    /* the read: the shifted partner = the CT at origin - shift (level 1, zyx shift 2, -1, 3), exact; outside the partner 0 */
    {
        snprintf(p, sizeof p, "%s/map-sh.json", dir); sview *v = sview_open(p, 0, 9.6, nullptr); assert(v);
        snprintf(p, sizeof p, "%s/sources.json", dir); sources *S = sources_load(p); assert(S);
        z3 *ct = source_ct(&S->src[0], 1); const int64_t o[3] = {40, 30, 20}, n[3] = {24, 40, 32}, os[3] = {38, 31, 17};
        const size_t N = (size_t)n[0] * n[1] * n[2]; uint8_t *a = malloc(N), *b = malloc(N);
        assert(!sview_read_level(v, 1, o, n, a) && !z3_read(ct, os, n, b, 1));
        size_t diff = 0, nz = 0; for (size_t k = 0; k < N; k++) { diff += a[k] != b[k]; nz += b[k] != 0; }
        printf("scrollaug: xscan read: %zu of %zu voxels differ (%zu scan voxels)\n", diff, N, nz);
        check("xscan read: a translated partner = the CT read at the shifted origin (bit for bit)", diff == 0 && nz > N / 4);
        const int64_t oe[3] = {0, 0, 0};   /* (z, x shifted by -2, -3 at level 1: the first rows / columns lie outside the partner) */
        assert(!sview_read_level(v, 1, oe, n, a));
        size_t outz = 0, out_n = 0; for (int64_t z = 0; z < n[0]; z++) for (int64_t y = 0; y < n[1]; y++) for (int64_t x = 0; x < n[2]; x++)
            if (z < 2 || x < 3) { out_n++; outz += a[((size_t)z * n[1] + y) * n[2] + x] == 0; }
        check("xscan read: outside the partner volume reads 0", outz == out_n);
        free(a); free(b); sview_free(v); sources_free(S);
    }
    /* the sampler */
    write_sources(dir, "src-id.json", "map-id.json"); write_sources(dir, "src-sh.json", "map-sh.json"); write_sources(dir, "src-no.json", nullptr);
    const int PW = 64; const size_t p3 = (size_t)PW * PW * PW;
    snprintf(p, sizeof p, "%s/src-no.json", dir); sources *S0 = sources_load(p); assert(S0);
    snprintf(p, sizeof p, "%s/src-id.json", dir); sources *Si = sources_load(p); assert(Si);
    snprintf(p, sizeof p, "%s/src-sh.json", dir); sources *Ss = sources_load(p); assert(Ss);
    sample_cfg c = base_cfg(PW);
    snap a[NB], b[NB], d[NB], e[NB], f[NB];
    sampler *sp = sampler_start(S0, &c); assert(sp); take(sp, a, p3); sampler_stop(sp);
    sp = sampler_start(Si, &c); assert(sp); take(sp, b, p3); const uint64_t n0 = sampler_aug_count(sp, AUGN_XSCAN); sampler_stop(sp);
    check("--aug-xscan 0 with \"xscan\" sources: bit-identical to sources without", n0 == 0 && same(a, b, p3));
    sample_cfg c1 = c; c1.xscan_p = 1;
    sp = sampler_start(Si, &c1); assert(sp); take(sp, d, p3); const uint64_t ni = sampler_aug_count(sp, AUGN_XSCAN); sampler_stop(sp);
    sp = sampler_start(Ss, &c1); assert(sp); take(sp, e, p3); const uint64_t ns = sampler_aug_count(sp, AUGN_XSCAN); sampler_stop(sp);
    sp = sampler_start(Ss, &c1); assert(sp); take(sp, f, p3); sampler_stop(sp);
    /* (p 1 draws one more uniform per window than p 0, so the identity partner is compared with p 1 runs only) */
    size_t lk = 0, la = 0, xd = 0;
    for (int i = 0; i < NB; i++) for (size_t k = 0; k < p3; k++) {
        xd += d[i].x[k] != e[i].x[k];
        if (d[i].side[k] < PHASE_PERIOD && e[i].side[k] < PHASE_PERIOD) { lk++; la += d[i].side[k] == e[i].side[k]; }
    }
    printf("scrollaug: xscan sampler: %llu / %llu windows swapped (identity / shifted partner); shifted vs identity: CT differs at %.3f of the voxels, "
           "phase labels agree at %.4f of %zu voxels known in both\n", (unsigned long long)ni, (unsigned long long)ns, xd / (double)(NB * p3), lk ? la / (double)lk : 0, lk);
    check("--aug-xscan 1: every window swapped, deterministic", ni >= NB && ns >= NB && same(e, f, p3));
    check("--aug-xscan 1: a shifted partner changes the CT, the labels stay (>= 99% where both are known; the air mask follows the CT)", xd > NB * p3 / 4 && lk > NB * p3 / 10 && la >= .99 * (double)lk);
    drop(a); drop(b); drop(d); drop(e); drop(f);
    sources_free(S0); sources_free(Si); sources_free(Ss);
}

/* --aug-tear: the kernels on analytic boxes, then the sampler on the fixture */
static int w4diff(int a, int b) { int d = a - b; if (d >= 126) d -= 252; if (d < -126) d += 252; return d; }   /* (63 codes per turn, mod 4 turns) */
static void tear_tests(const char *dir) {
    /* the sheet normal and spacing of the fixture's phase (level 1: pitch 12 voxels, radial normal) */
    {
        const int B = 72; const size_t B3 = (size_t)B * B * B; uint8_t *ph = malloc(B3); const int z0 = 40, y0 = 20, x0 = 30;   /* (level 1, off-axis) */
        for (int z = 0; z < B; z++) for (int y = 0; y < B; y++) for (int x = 0; x < B; x++) {
            const double w = fx_wind(2. * (z0 + z) + .5, 2. * (y0 + y) + .5, 2. * (x0 + x) + .5);
            ph[((size_t)z * B + y) * B + x] = (uint8_t)((int)floor((w - floor(w)) * PHASE_PERIOD) % PHASE_PERIOD);
        }
        double n[3], sp = 0; const int ns = sa_phase_normal(ph, B, B, B, 4, 2, n, &sp);
        double cy, cx; fx_axis_at(2. * (z0 + B / 2), &cy, &cx);
        double rad[3] = {0, 2. * (y0 + B / 2) - cy, 2. * (x0 + B / 2) - cx}; const double rl = sqrt(rad[1] * rad[1] + rad[2] * rad[2]); rad[1] /= rl; rad[2] /= rl;
        const double cosr = fabs(n[0] * rad[0] + n[1] * rad[1] + n[2] * rad[2]);
        printf("scrollaug: tear normal: %d samples, spacing %.2f voxels (12 expected), |n . radial| %.3f\n", ns, sp, cosr);
        check("tear: sheet spacing and normal from a phase box (12 voxels, radial)", ns > 1000 && fabs(sp - 12) < 1.5 && cosr > .95);
        /* sa_tear_apply: every voxel off the seam is its side's shifted copy; the seam voids / blends with labels unknown */
        const int P = 48, M = 12, Pp = P + 2 * M; const size_t p3 = (size_t)P * P * P, pp3 = (size_t)Pp * Pp * Pp;
        uint8_t *ct = malloc(pp3), *w4 = malloc(pp3), *wc = malloc(pp3), *oc = malloc(p3), *op = malloc(p3), *ow = malloc(p3), *oq = malloc(p3);
        for (size_t k = 0; k < pp3; k++) { ct[k] = (uint8_t)(1 + k % 251); w4[k] = (uint8_t)(k % 252); wc[k] = (uint8_t)(k % 7); }
        int okc = 1, oks = 1, nseam = 0, jumps = 0;
        for (int it = 0; it < 24; it++) {
            sa_tear t; const double nn[3] = {0, .6, .8}; sa_tear_make(&t, 1000 + it, P, nn, 10, 2 * (M - 2));
            if (sa_tear_margin(&t) > M) { okc = 0; break; }
            jumps += fabs(t.turns_n) >= .3;
            sa_tear_apply(&t, P, M, ct, ct, w4, wc, 7, oc, op, ow, oq);   /* (the CT box as the phase too: any codes) */
            for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
                const double h = sa_tear_h(&t, z, y, x); const int *d = h > 0 ? t.dp : t.dm;
                const size_t k = ((size_t)z * P + y) * P + x, s = ((size_t)(z - d[0] + M) * Pp + (y - d[1] + M)) * Pp + (x - d[2] + M);
                if (t.seam && fabs(h) < .5 * t.width - 1e-6) { nseam++; oks &= op[k] == PHASE_UNKNOWN && ow[k] == 255 && oq[k] == 255 && (t.seam != 1 || oc[k] == 7); }
                else if (fabs(h) > .5 * t.width + 1e-6) okc &= oc[k] == ct[s] && op[k] == ct[s] && ow[k] == w4[s] && oq[k] == wc[s];
            }
        }
        check("tear apply: off the seam each side is its shifted copy (CT, phase, w4, component), bit for bit", okc);
        check("tear apply: seam voxels void / blended with the labels unknown", oks && nseam > 0);
        check("tear make: most draws shift >= 0.3 turns along the normal", jumps == 24);
        free(ph); free(ct); free(w4); free(wc); free(oc); free(op); free(ow); free(oq);
    }
    /* the sampler: default off, p 1 deterministic, and the winding mod 4 jumps across the cut where the phase stays smooth-looking */
    char p[1400]; snprintf(p, sizeof p, "%s/sources.json", dir); sources *S = sources_load(p); assert(S);
    const int PW = 64; const size_t p3 = (size_t)PW * PW * PW;
    sample_cfg c = base_cfg(PW); c.snap = 0; c.symmetry_p = 0; c.rotate_p = 0; c.elastic_p = 0; c.zoom_p = 0; c.affine_p = 0;
    snap a[NB], b[NB], d[NB], e[NB];
    sampler *sp = sampler_start(S, &c); assert(sp); take(sp, a, p3); sampler_stop(sp);
    sample_cfg c0 = c; c0.tear_um = 300;
    sp = sampler_start(S, &c0); assert(sp); take(sp, b, p3); const uint64_t n0 = sampler_aug_count(sp, AUGN_TEAR); sampler_stop(sp);
    check("--aug-tear 0: bit-identical", n0 == 0 && same(a, b, p3));
    sample_cfg c1 = c; c1.tear_p = 1;
    sp = sampler_start(S, &c1); assert(sp); take(sp, d, p3); const uint64_t n1 = sampler_aug_count(sp, AUGN_TEAR); sampler_stop(sp);
    sp = sampler_start(S, &c1); assert(sp); take(sp, e, p3); sampler_stop(sp);
    check("--aug-tear 1: windows torn, deterministic", n1 >= NB && same(d, e, p3));
    /* neighbours (x, x + 1) in one component: the w4 step agrees with the phase step mod 1 turn everywhere; steps >= 0.3 turn (a cut
       crossed with the windings kept) only in torn windows */
    size_t pairs = 0, agree = 0, big_t = 0, big_p = 0;
    for (int pass = 0; pass < 2; pass++) for (int i = 0; i < NB; i++) {
        const snap *q = pass ? &d[i] : &a[i];
        for (size_t k = 0; k + 1 < p3; k++) {
            if (k % PW == PW - 1 || q->w4[k] == 255 || q->w4[k + 1] == 255 || q->wc[k] != q->wc[k + 1] || q->side[k] >= PHASE_PERIOD || q->side[k + 1] >= PHASE_PERIOD) continue;
            const int dw = w4diff(q->w4[k + 1], q->w4[k]), dph = q->side[k + 1] - q->side[k];
            int m = (dw * 4 - dph) % 252; if (m < 0) m += 252; if (m > 126) m = 252 - m;   /* (w4 63 codes per turn, phase 252) */
            pairs++; agree += m <= 12;
            if (abs(dw) >= 19) { if (pass) big_t++; else big_p++; }
        }
    }
    printf("scrollaug: tear sampler: %llu torn; %zu neighbour pairs, w4 / phase steps agree at %.4f; >= 0.3 turn w4 steps: plain %zu, torn %zu\n",
           (unsigned long long)n1, pairs, pairs ? agree / (double)pairs : 0, big_p, big_t);
    check("--aug-tear: the w4 labels follow the pulled phase (steps agree mod 1 turn)", pairs > p3 && agree >= .99 * (double)pairs);
    check("--aug-tear: the winding mod 4 jumps across the cut (>= 0.3 turn steps in torn windows only)", big_t > 50 && big_p * 20 < big_t);
    drop(a); drop(b); drop(d); drop(e); sources_free(S);
}

int main(int argc, char **argv) {
    char tmpd[] = "/tmp/ufsm-scrollaug-XXXXXX"; const char *dir = argc > 1 ? argv[1] : mkdtemp(tmpd);
    if (!dir || make_wind_fixture(dir, 256, 2.f)) { fprintf(stderr, "fixture failed\n"); return 1; }
    xscan_tests(dir);
    tear_tests(dir);
    if (argc < 2) { char cmd[1500]; snprintf(cmd, sizeof cmd, "rm -rf '%s'", dir); if (system(cmd)) {} }
    printf(failed ? "scrollaug: %d FAILED\n" : "scrollaug: all passed\n", failed);
    return failed != 0;
}
