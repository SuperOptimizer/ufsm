/* Real threaded draws with local CT/labels and an injected read failure after one
   complete sample. Also check holdout containment and cross-source exclusions. */
#include "zarr3.h"
#include "sources.h"
static int controlled_read(z3 *z, const int64_t o[3], const int64_t n[3], uint8_t *out, int cache);
static int controlled_target(source *s, int ch, int level, const int64_t o[3], const int64_t n[3], uint8_t *out, int cache);
#define z3_read controlled_read
#define source_read_target controlled_target
#ifndef TEST_SAMPLE_SOURCE
#define TEST_SAMPLE_SOURCE "../src/sample.c"
#endif
#include TEST_SAMPLE_SOURCE
#undef z3_read
#undef source_read_target
#include <assert.h>

void *nn_host_alloc(size_t n) { return malloc(n); }
void nn_host_free(void *p) { free(p); }
static atomic_int read_budget = -1;
static int controlled_read(z3 *z, const int64_t o[3], const int64_t n[3], uint8_t *out, int cache) {
    int budget = atomic_load(&read_budget);
    if (budget == 0) return -1;
    if (budget > 0) atomic_fetch_sub(&read_budget, 1);
    return z3_read(z, o, n, out, cache);
}
static int controlled_target(source *s, int ch, int level, const int64_t o[3], const int64_t n[3], uint8_t *out, int cache) {
    int budget = atomic_load(&read_budget);
    if (budget == 0) return -1;
    if (budget > 0) atomic_fetch_sub(&read_budget, 1);
    return source_read_target(s, ch, level, o, n, out, cache);
}
static sources *fixture(const char *dir, const char *boxes) {
    char path[1400]; snprintf(path, sizeof path, "%s/safety-sources.json", dir);
    FILE *f = fopen(path, "w"); assert(f);
    fprintf(f, "{\"sources\":[{\"name\":\"a\",\"root\":\"%s\",\"ct\":\"ct\",\"um\":1,\"targets\":{\"recto\":\"labels\"}%s}]}\n", dir, boxes);
    assert(!fclose(f)); sources *S = sources_load(path); assert(S); return S;
}
static sample_cfg config(void) {
    sample_cfg c = sample_cfg_default(); c.P = 16; c.B = 4; c.nworkers = 1; c.nbuf = 2;
    memset(c.level_p, 0, sizeof c.level_p); c.level_p[0] = 1; c.min_fg = 0; c.empty_keep = 1; c.augment = 0; return c;
}
static int failed;
static void check(const char *name, int ok) { printf("sampler %-45s %s\n", name, ok ? "ok" : "FAIL"); failed += !ok; }
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    sources *S = fixture(argv[1], ",\"holdout\":[7,11,13,51,53,55]");
    sample_cfg c = config(); c.holdout = 1; sampler *sp = sampler_start(S, &c); assert(sp);
    int contained = 1;
    for (int k = 0; k < 32; k++) {
        batch *b = sampler_next(sp); assert(b);
        for (int i = 0; i < c.B; i++) for (int d = 0; d < 3; d++) contained &= b->corner[i][d] >= S->src[0].hold_o[d] && b->corner[i][d] + c.P <= S->src[0].hold_o[d] + S->src[0].hold_n[d];
        sampler_release(sp, b);
    }
    sampler_stop(sp); check("snapping never moves validation outside holdout", contained); sources_free(S);
    S = fixture(argv[1], "");
    for (int det = 0; det <= 1; det++) {
        c = config(); c.deterministic = det; atomic_store(&read_budget, 2);
        sp = sampler_start(S, &c); assert(sp); batch *b = sampler_next(sp);
        check(det ? "ordered read failure discards partial batch" : "read failure discards partial batch", b == nullptr);
        uint64_t produced, rejected; sampler_stats(sp, &produced, &rejected); check("failure happened after exactly one sample", produced == 1);
#ifndef TEST_LEGACY
        check("fatal read failure is reported to the caller", sampler_failed(sp));
#endif
        sampler_stop(sp); atomic_store(&read_budget, -1);
    }
    sources_free(S);
    /* A finite plan must drain ready slots in order, retain empty tiles, and
       resume at the committed cursor even with several workers ahead. */
    S = fixture(argv[1], "");
    int64_t tiles[5][4] = {{0,0,0,0}, {0,16,0,0}, {0,32,0,0}, {0,48,0,0}, {0,64,0,0}};
    cover_plan plan = {.P = 16, .count = 5, .tiles = tiles};
    c = config(); c.B = 1; c.nworkers = 4; c.cover = &plan;
    c.min_fg = 1.1; c.empty_keep = 0; /* random rejection rules cannot erase planned tiles */
    for (int start = 0; start < 5; start++) {
        c.cover_start = (uint64_t)start; sp = sampler_start(S, &c); assert(sp);
        int ordered = 1;
        for (int k = start; k < 5; k++) {
            batch *b = sampler_next(sp); ordered &= b && b->corner[0][0] == 16 * k && b->corner[0][1] == 0 && b->corner[0][2] == 0;
            if (b) sampler_release(sp, b);
        }
        ordered &= sampler_next(sp) == nullptr && !sampler_failed(sp);
        sampler_stop(sp); check("finite plan drains to EOF and resumes exactly", ordered);
    }
    c.cover_start = 0; c.nworkers = 1;
    sp = sampler_start(S, &c); assert(sp); batch *plain = sampler_next(sp); assert(plain);
    size_t n = 16 * 16 * 16;
    uint8_t *target = malloc(NCH * n), *mask = malloc(n);
    memcpy(target, plain->t, NCH * n); memcpy(mask, plain->m, n);
    sampler_release(sp, plain); sampler_stop(sp);
    c.augment = 4; c.ct_augment = 1; c.seed = 2;
    sp = sampler_start(S, &c); assert(sp); batch *aug = sampler_next(sp); assert(aug);
    check("appearance augmentation preserves targets and validity", !memcmp(target, aug->t, NCH * n) && !memcmp(mask, aug->m, n));
    float *input = malloc(4 * n * sizeof(float)); memcpy(input, aug->x, 4 * n * sizeof(float));
    sampler_release(sp, aug); sampler_stop(sp); c.nworkers = 4;
    sp = sampler_start(S, &c); assert(sp); aug = sampler_next(sp); assert(aug);
    check("augmentation is reproducible across worker counts", !memcmp(input, aug->x, 4 * n * sizeof(float)));
    sampler_release(sp, aug); sampler_stop(sp); free(input); free(target); free(mask);
    c.augment=1; c.geometry_augment=1; c.rotate_degrees=5; c.rotate_p=1; c.elastic=1; c.elastic_p=1;
    c.soft=1.75; c.label_morph=.25; c.label_morph_p=1; c.nworkers=1;
    input=malloc(4*n*sizeof(float)); target=malloc(NCH*n); mask=malloc(n);
    sp=sampler_start(S,&c); assert(sp); aug=sampler_next(sp); assert(aug);
    memcpy(input,aug->x,4*n*sizeof(float)); memcpy(target,aug->t,NCH*n); memcpy(mask,aug->m,n);
    sampler_release(sp,aug); sampler_stop(sp); c.nworkers=4;
    sp=sampler_start(S,&c); assert(sp); aug=sampler_next(sp); assert(aug);
    check("3D warp/morphology deterministic across worker counts", !memcmp(input,aug->x,4*n*sizeof(float)) && !memcmp(target,aug->t,NCH*n) && !memcmp(mask,aug->m,n));
    sampler_release(sp,aug); sampler_stop(sp); free(input); free(target); free(mask);
    c.cover_start = 0; S->src[0].hold_o[0] = 16; S->src[0].hold_n[0] = S->src[0].hold_n[1] = S->src[0].hold_n[2] = 16;
    sp = sampler_start(S, &c); check("finite plan rejects holdout intersection", !sp); sampler_stop(sp);
    sources_free(S);
    /* Two teachers share one CT. Their holdouts together cover it, although neither alone does. */
    S = fixture(argv[1], ",\"holdout\":[0,0,0,64,128,128]");
    S->src = realloc(S->src, 2 * sizeof *S->src); S->src[1] = (source){0}; S->n = 2;
    source *s = &S->src[1]; s->name = strdup("other teacher"); s->s = store_open(argv[1]); s->ct_key = strdup("ct"); s->um = 1; s->weight = 0;
    s->hold_o[0] = 64; s->hold_n[0] = 64; s->hold_n[1] = s->hold_n[2] = 128;
    for (int l = 0; l < MAXLEV; l++) { s->ct_present[l] = -1; for (int ch = 0; ch < NCH; ch++) s->tgt_present[ch][l] = -1; }
    c = config(); sp = sampler_start(S, &c); check("union of same-CT holdouts blocks training", !sp); sampler_stop(sp);
#ifndef TEST_LEGACY
    /* Compare the corner preflight with exhaustive integer origins for varied overlapping boxes. */
    sampler fake = {.S = S, .cfg = c}; fake.cfg.P = 2; rng r; rseed(&r, 193);
    int exact = 1;
    for (int k = 0; k < 400; k++) {
        for (int i = 0; i < S->n; i++) for (int d = 0; d < 3; d++) { S->src[i].hold_o[d] = rint_below(&r, 9); S->src[i].hold_n[d] = 1 + rint_below(&r, 10 - S->src[i].hold_o[d]); }
        int64_t lo[3] = {0,0,0}, hi[3] = {8,8,8}; int brute = 0;
        for (int z = 0; z <= 8; z++) for (int y = 0; y <= 8; y++) for (int x = 0; x <= 8; x++) { int64_t o[3] = {z,y,x}; brute |= !training_hits_holdout(&fake, &S->src[0], o, 0); }
        exact &= training_corner_exists(&fake, &S->src[0], 0, lo, hi) == brute;
    }
    check("union preflight agrees with exhaustive origins", exact);
    z3_meta m = {.shape = {128,128,128}}; source bound = {.hold_o = {7,11,13}, .hold_n = {51,53,55}};
    int aligned = 1;
    for (int l = 0; l < 5; l++) {
        int64_t lo[3], hi[3]; int fits = patch_bounds(&bound, &m, 2, l, nullptr, 0, 1, lo, hi);
        for (int d = 0; fits && d < 3; d++) aligned &= (lo[d] << l) >= bound.hold_o[d] && ((hi[d] + 2) << l) <= bound.hold_o[d] + bound.hold_n[d];
    }
    check("coarse origins remain inside unaligned holdouts", aligned);
#endif
    sources_free(S); return failed != 0;
}
