/* sampler throughput without a GPU step: bench_sampler <sources.json> [P] [workers] [batches] (env UFSM_SAMPLER_PROF=1 for stages) */
#include "nn.h"
#include "sample.h"
#include "sources.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: bench_sampler <sources.json> [P] [workers] [batches]\n"); return 2; }
    int P = argc > 2 ? atoi(argv[2]) : 128, W = argc > 3 ? atoi(argv[3]) : 16, NB = argc > 4 ? atoi(argv[4]) : 64;
    sources *S = sources_load(argv[1]); if (!S) return 1;
    sample_cfg c = sample_cfg_default(); c.P = P; c.B = 2; c.nworkers = W; c.nbuf = 2 * W; c.soft = 3; c.augment = 4; c.xfmt = 1;
    sampler *sp = sampler_start(S, &c);
    for (int i = 0; i < 4; i++) { batch *b = sampler_next(sp); if (!b) return 1; sampler_release(sp, b); }   /* warm-up */
    double t0 = now();
    for (int i = 0; i < NB; i++) { batch *b = sampler_next(sp); if (!b) return 1; sampler_release(sp, b); }
    double dt = now() - t0;
    uint64_t prod, rej; sampler_stats(sp, &prod, &rej);
    printf("P=%d workers=%d: %.1f samples/s (%d batches of %d in %.1fs; produced %llu rejected %llu)\n", P, W, NB * c.B / dt, NB, c.B, dt, (unsigned long long)prod, (unsigned long long)rej);
    sampler_prof_print(sp);
    sampler_stop(sp);
    return 0;
}
