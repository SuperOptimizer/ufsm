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
typedef struct { uint16_t *x; uint8_t *side, *w4; } snap;
static void take(sampler *sp, snap *o, size_t p3) {
    for (int i = 0; i < NB; i++) {
        batch *b = sampler_next(sp); assert(b);
        o[i].x = malloc(p3 * 2); memcpy(o[i].x, b->x16, p3 * 2);
        o[i].side = malloc(p3); memcpy(o[i].side, b->side, p3); o[i].w4 = malloc(p3); memcpy(o[i].w4, b->w4, p3);
        sampler_release(sp, b);
    }
}
static int same(const snap *a, const snap *b, size_t p3) {
    int ok = 1; for (int i = 0; i < NB; i++) ok &= !memcmp(a[i].x, b[i].x, p3 * 2) && !memcmp(a[i].side, b[i].side, p3) && !memcmp(a[i].w4, b[i].w4, p3);
    return ok;
}
static void drop(snap *o) { for (int i = 0; i < NB; i++) { free(o[i].x); free(o[i].side); free(o[i].w4); } }
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

int main(int argc, char **argv) {
    char tmpd[] = "/tmp/ufsm-scrollaug-XXXXXX"; const char *dir = argc > 1 ? argv[1] : mkdtemp(tmpd);
    if (!dir || make_wind_fixture(dir, 256, 2.f)) { fprintf(stderr, "fixture failed\n"); return 1; }
    xscan_tests(dir);
    if (argc < 2) { char cmd[1500]; snprintf(cmd, sizeof cmd, "rm -rf '%s'", dir); if (system(cmd)) {} }
    printf(failed ? "scrollaug: %d FAILED\n" : "scrollaug: all passed\n", failed);
    return failed != 0;
}
