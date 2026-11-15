/* sampbias (2026-11-14): the draw balance options of the sampler on the synthetic wind fixture.
   - edge_zboost_index: interior cells of a labelled box keep one draw, corner / edge cells get more (<= 4), "zboost" multiplies the low-z cells by K
   - sampler_vox_scale: 1 for sources at or above the weighted mean known share / K, the ratio below it, 1 while unmeasured
   - a sampler with --snap-jitter / --edge-boost / --stick-src / --vox-cap on delivers batches whose known-phase share is recorded */
#define WIND_FIXTURE_NO_MAIN
#include "make_wind_fixture.c"
#include "../src/sample.c"
#include <assert.h>

void *nn_host_alloc(size_t n) { return malloc(n); }
void nn_host_free(void *p) { free(p); }
static int failed;
static void check(const char *name, int ok) { printf("sampbias: %-72s %s\n", name, ok ? "ok" : "FAIL"); failed += !ok; }
int main(int argc, char **argv) {
    char tmpd[] = "/tmp/ufsm-sampbias-XXXXXX"; const char *dir = argc > 1 ? argv[1] : mkdtemp(tmpd);
    if (!dir || make_wind_fixture(dir, 256, 2.f)) { fprintf(stderr, "fixture failed\n"); return 1; }
    char path[1400]; snprintf(path, sizeof path, "%s/sources.json", dir);
    sources *S = sources_load(path); assert(S);
    sample_cfg c = sample_cfg_default();
    c.P = 64; c.B = 1; c.nworkers = 2; c.nbuf = 4; c.seed = 7; c.xfmt = 1; c.gpu_input = 1; c.side = 2; c.wind4 = 1;
    memset(c.level_p, 0, sizeof c.level_p); c.level_p[0] = 1; c.min_fg = 0;
    c.snap_jitter = 1; c.edge_boost = 1; c.stick = 4; c.vox_cap = 2;

    /* 1: the index. A 12^3 grid with the labelled box [3, 9)^3; window box n = P >> (lev - min_level) = 64 >> 4 = 4 cells (lev 4, min_level 0). */
    sampler fake = {0}; fake.cfg = c;
    source fs = S->src[0]; fs.band = (void *)1; fs.phase = (void *)1; fs.min_level = 0; fs.zb_below = 0; fs.zb_k = 0;
    const int64_t G = 12; uint32_t *idx = malloc(G * G * G * sizeof *idx); size_t n = 0;
    for (int64_t z = 3; z < 9; z++) for (int64_t y = 3; y < 9; y++) for (int64_t x = 3; x < 9; x++) idx[n++] = (uint32_t)((z * G + y) * G + x);
    occ_index o = {idx, n, 4, {G, G, G}};
    edge_zboost_index(&fake, &fs, &o);
    int cnt[12 * 12 * 12] = {0}; for (size_t k = 0; k < o.n; k++) cnt[o.idx[k]]++;
    const int interior = cnt[(6 * G + 6) * G + 6], corner = cnt[(3 * G + 3) * G + 3];
    check("edge-boost: an interior cell is drawn once", interior == 1);
    check("edge-boost: a corner cell is drawn 2-4 times", corner >= 2 && corner <= 4);
    free(o.idx);
    /* zboost: cells with z < 5 x3 */
    fake.cfg.edge_boost = 0; fs.zb_below = 5; fs.zb_k = 3;
    idx = malloc(G * G * G * sizeof *idx); n = 0;
    for (int64_t z = 3; z < 9; z++) for (int64_t y = 3; y < 9; y++) for (int64_t x = 3; x < 9; x++) idx[n++] = (uint32_t)((z * G + y) * G + x);
    o = (occ_index){idx, n, 4, {G, G, G}};
    edge_zboost_index(&fake, &fs, &o);
    memset(cnt, 0, sizeof cnt); for (size_t k = 0; k < o.n; k++) cnt[o.idx[k]]++;
    check("zboost: z < 5 cells x3, others x1", cnt[(3 * G + 5) * G + 5] == 3 && cnt[(4 * G + 5) * G + 5] == 3 && cnt[(5 * G + 5) * G + 5] == 1);
    free(o.idx);

    /* 2: the loss scale of a source with a small known share */
    sources fs2 = {0}; fs2.n = 3; fake.S = &fs2; double cum[3] = {1, 2, 3}; fake.cum = cum; fake.cfg.vox_cap = 2;
    check("vox scale: 1 while nothing is measured", sampler_vox_scale(&fake, 0) == 1.f);
    const double sh[3] = {0.66, 0.60, 0.056};
    for (int j = 0; j < 3; j++) { atomic_store(&fake.ksh_n[j], 10); atomic_store(&fake.ksh_sum[j], sh[j] * 10); }
    const double ref = (0.66 + 0.60 + 0.056) / 3;
    check("vox scale: a source at the mean keeps 1", sampler_vox_scale(&fake, 0) == 1.f);
    check("vox scale: 0.056 share -> K x share / mean", fabsf(sampler_vox_scale(&fake, 2) - (float)(2 * 0.056 / ref)) < 1e-4f);

    /* 3: a sampler with every option on delivers windows and records the known share */
    c.nworkers = 2; sampler *sp = sampler_start(S, &c); assert(sp);
    int got = 0; for (int i = 0; i < 6; i++) { batch *b = sampler_next(sp); if (!b) break; got++; sampler_release(sp, b); }
    check("sampler with --snap-jitter --edge-boost --stick-src --vox-cap delivers 6 batches", got == 6);
    check("known-phase share recorded for source 0", atomic_load(&sp->ksh_n[0]) >= 1);
    sampler_stop(sp);
    printf("sampbias: %s\n", failed ? "FAILED" : "passed");
    return failed;
}
