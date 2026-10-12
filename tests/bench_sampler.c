/* sampler throughput without a GPU step: bench_sampler <sources.json> [P] [workers] [batches] (env UFSM_SAMPLER_PROF=1 for stages)
   env UFSM_BENCH_TRAIN=1: the trainer's surface recipe (B 1, --soft 7 --soft-unit um --erode 1 --zfix 1 --symmetry-p 0.5 --det 1,
   seed 1, fp16 input); UFSM_BENCH_COVER=plan.json: draw the cover's tiles (as train --cover); UFSM_BENCH_HASH=1: print a hash of
   every batch's input, targets, mask and weights (deterministic mode: the same for any worker count; byte-identity checks);
   UFSM_BENCH_AUG=N: augment mode; UFSM_BENCH_DET=1: deterministic default recipe; UFSM_BENCH_SOFT=S: soft sigma;
   UFSM_BENCH_SIDE=2: the winding phase of band sources (task unwrap), hashed too;
   UFSM_BENCH_PAPYRUS=sigma,tau_lo,tau_hi,clean: task papyrus labels (cutoffs U(50, 128), conditioned); UFSM_BENCH_SCAN=P: the
   papyrus recipe's scanner ops (noise cnoise blur lowres sharpen ring slice quant) at probability P;
   UFSM_BENCH_DUMP=prefix: mid-z planes of the first batch (prefix-input.pgm, -target.pgm, -mask.pgm) and its cutoff code */
#include "nn.h"
#include "sample.h"
#include "sources.h"
#include "cover.h"
#include "papyrus.h"
#include "scan_augment.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t hash(uint64_t h, const void *p, size_t n) {
    const unsigned char *c = p; size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t w; memcpy(&w, c + i, 8); h = (h ^ w) * 0x100000001b3ull; h ^= h >> 29; }
    for (; i < n; i++) h = (h ^ c[i]) * 0x100000001b3ull;
    return h;
}
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: bench_sampler <sources.json> [P] [workers] [batches]\n"); return 2; }
    int P = argc > 2 ? atoi(argv[2]) : 128, W = argc > 3 ? atoi(argv[3]) : 16, NB = argc > 4 ? atoi(argv[4]) : 64;
    sources *S = sources_load(argv[1]); if (!S) return 1;
    sample_cfg c = sample_cfg_default(); c.P = P; c.B = 2; c.nworkers = W; c.nbuf = 2 * W; c.soft = 3; c.augment = 4; c.xfmt = 1;
    if (getenv("UFSM_BENCH_TRAIN")) { c.B = 1; c.soft = 7; c.soft_um = 1; c.erode = 1; c.augment = 3; c.symmetry_p = 0.5f; c.deterministic = 1; c.seed = 1; }
    if (getenv("UFSM_BENCH_AUG")) c.augment = atoi(getenv("UFSM_BENCH_AUG"));
    if (getenv("UFSM_BENCH_DET")) { c.deterministic = 1; c.seed = 1; }
    if (getenv("UFSM_BENCH_SOFT")) c.soft = (float)atof(getenv("UFSM_BENCH_SOFT"));
    if (getenv("UFSM_BENCH_SIDE")) c.side = atoi(getenv("UFSM_BENCH_SIDE"));   /* task unwrap's phase (no band affinities): the side array is hashed too */
    if (getenv("UFSM_BENCH_COVER")) {
        cover_plan *plan = cover_load(getenv("UFSM_BENCH_COVER"), S, P); if (!plan) return 1;
        c.B = 1; c.cover = plan; c.deterministic = 1; c.snap = 0; memset(c.level_p, 0, sizeof c.level_p); c.level_p[0] = 1;
    }
    papyrus_cfg pc = {.t_lo = 50, .t_hi = 128, .cond = 1};
    if (getenv("UFSM_BENCH_PAPYRUS")) {
        if (sscanf(getenv("UFSM_BENCH_PAPYRUS"), "%f,%f,%f,%d", &pc.sigma, &pc.tau_lo, &pc.tau_hi, &pc.clean) != 4) { fprintf(stderr, "UFSM_BENCH_PAPYRUS=sigma,tau_lo,tau_hi,clean\n"); return 2; }
        c.papyrus = &pc; c.soft = 0;
    }
    if (getenv("UFSM_BENCH_SCAN")) {
        const float p = (float)atof(getenv("UFSM_BENCH_SCAN")); c.scan.strength = 1;
        const int ops[] = {SA_NOISE, SA_CNOISE, SA_BLUR, SA_LOWRES, SA_SHARPEN, SA_RING, SA_SLICE, SA_QUANT};
        for (size_t k = 0; k < sizeof ops / sizeof ops[0]; k++) c.scan.p[ops[k]] = p;
    }
    const int hashing = getenv("UFSM_BENCH_HASH") != nullptr;
    const size_t p3 = (size_t)P * P * P;
    uint64_t h = 0xcbf29ce484222325ull;
    sampler *sp = sampler_start(S, &c);
    if (getenv("UFSM_BENCH_DUMP")) {   /* mid-z planes of the first batch: input (z-scored CT x 40 + 128), target, mask */
        batch *b = sampler_next(sp); if (!b) return 1;
        const char *pre = getenv("UFSM_BENCH_DUMP"); const size_t pl = (size_t)P * P, z0 = (size_t)(P / 2) * pl;
        uint8_t *img = malloc(pl); char fn[1024];
        for (int k = 0; k < 3; k++) {
            for (size_t j = 0; j < pl; j++) {
                if (k == 0) { _Float16 h; memcpy(&h, b->x16 + z0 + j, 2); const float v = 128.f + 40.f * (float)h; img[j] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); }
                else img[j] = k == 1 ? b->t[z0 + j] : (uint8_t)(b->m[z0 + j] ? 255 : 0);
            }
            snprintf(fn, sizeof fn, "%s-%s.pgm", pre, k == 0 ? "input" : k == 1 ? "target" : "mask");
            FILE *f = fopen(fn, "wb"); if (f) { fprintf(f, "P5\n%d %d\n255\n", P, P); fwrite(img, 1, pl, f); fclose(f); }
        }
        printf("dumped %s-*.pgm: um / cutoff code %g, corner %lld %lld %lld\n", pre, b->um[0], (long long)b->corner[0][0], (long long)b->corner[0][1], (long long)b->corner[0][2]);
        free(img); sampler_release(sp, b);
    }
    for (int i = 0; i < 4; i++) { batch *b = sampler_next(sp); if (!b) return 1; if (hashing) { h = hash(h, b->x16, (size_t)c.B * 4 * p3 * 2); h = hash(h, b->t, (size_t)c.B * NCH * p3); h = hash(h, b->m, (size_t)c.B * p3); h = hash(h, b->w, (size_t)c.B * NCH); if (b->side) h = hash(h, b->side, (size_t)c.B * p3); } sampler_release(sp, b); }   /* warm-up */
    double t0 = now();
    for (int i = 0; i < NB; i++) {
        batch *b = sampler_next(sp); if (!b) return 1;
        if (hashing) { h = hash(h, b->x16, (size_t)c.B * 4 * p3 * 2); h = hash(h, b->t, (size_t)c.B * NCH * p3); h = hash(h, b->m, (size_t)c.B * p3); h = hash(h, b->w, (size_t)c.B * NCH); if (b->side) h = hash(h, b->side, (size_t)c.B * p3); }
        sampler_release(sp, b);
    }
    double dt = now() - t0;
    uint64_t prod, rej; sampler_stats(sp, &prod, &rej);
    printf("P=%d workers=%d: %.1f samples/s (%d batches of %d in %.1fs; produced %llu rejected %llu)\n", P, W, NB * c.B / dt, NB, c.B, dt, (unsigned long long)prod, (unsigned long long)rej);
    if (hashing) printf("hash %016llx over %d batches\n", (unsigned long long)h, NB + 4);
    sampler_prof_print(sp);
    sampler_stop(sp);
    return 0;
}
