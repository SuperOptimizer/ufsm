/* ufsm z3-fetch ROOT KEY --cache DIR --box z,y,x,dz,dy,dx [--scale F] [--margin 128] [--threads 16]: every present inner chunk of the
   array that the box touches (box x F in the array's voxels, +- margin voxels) fetched into the chunk cache DIR, in z, y, x order, no
   decode (sharded: per shard, neighbouring chunks in one request as z3_read coalesces them). The cache is z3_open's (the same files a later z3_open(ROOT, KEY, DIR) reads; written tmp + rename, so a reader beside it is
   safe). Why (2026-11-17): unroll.sh's rectosp stage waited 77 of its 103 s on the pod's cold fetch of the m7 prob source (desk: cold 137 s
   wall / 104 s wait, warm 35.6 / 11.7 s, fp4teach); the source box is known at run start, so the unroll fetches it beside fields. */
#include "zarr3.h"
#include "store.h"
#include "sysinfo.h"
#include "optcheck.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { z3 *z; int64_t lo[3], n[3], tot, s0[3], ns[3], nsh; int sharded; _Atomic int64_t next, nf, nfail; } zf_job;
static void *zf_main(void *a) {   /* sharded: whole shards, z first (their box chunks coalesced, z3_prefetch_shard); else chunk by chunk */
    zf_job *J = a;
    const int64_t hi[3] = {J->lo[0] + J->n[0] - 1, J->lo[1] + J->n[1] - 1, J->lo[2] + J->n[2] - 1};
    for (int64_t i; (i = atomic_fetch_add(&J->next, 1)) < (J->sharded ? J->nsh : J->tot);) {
        int r;
        if (J->sharded) {
            const int64_t x = i % J->ns[2], y = i / J->ns[2] % J->ns[1], z = i / (J->ns[1] * J->ns[2]);
            r = z3_prefetch_shard(J->z, J->s0[0] + z, J->s0[1] + y, J->s0[2] + x, J->lo, hi);
            if (r > 0) atomic_fetch_add(&J->nf, r);
        } else {
            const int64_t x = i % J->n[2], y = i / J->n[2] % J->n[1], z = i / (J->n[1] * J->n[2]);
            r = z3_prefetch_chunk(J->z, J->lo[0] + z, J->lo[1] + y, J->lo[2] + x);
            if (r > 0) atomic_fetch_add(&J->nf, 1);
        }
        if (r < 0) atomic_fetch_add(&J->nfail, 1);
    }
    return nullptr;
}
static const char *zf_opt(int argc, char **argv, const char *k, const char *d) { for (int i = 3; i + 1 < argc; i++) if (!strcmp(argv[i], opt_n(k))) return argv[i + 1]; return d; }
int cmd_z3_fetch(int argc, char **argv) {
    const char *cache = zf_opt(argc, argv, "--cache", nullptr), *bs = zf_opt(argc, argv, "--box", nullptr);
    double b[6];
    if (argc < 4 || !cache || !bs || sscanf(bs, "%lf,%lf,%lf,%lf,%lf,%lf", &b[0], &b[1], &b[2], &b[3], &b[4], &b[5]) != 6) {
        fprintf(stderr, "usage: ufsm z3-fetch ROOT KEY --cache DIR --box z,y,x,dz,dy,dx [--scale F (box x F = array voxels, 1)] [--margin 128] [--threads 16]\n"); return 2; }
    const double F = opt_atof("--scale", zf_opt(argc, argv, "--scale", "1")); const int64_t m = opt_atoll("--margin", zf_opt(argc, argv, "--margin", "128"));
    const int nt = ufsm_threads(opt_atoi("--threads", zf_opt(argc, argv, "--threads", "16")), 16);
    store *s = store_open(argv[2]); z3 *z = s ? z3_open_typed(s, argv[3], cache) : nullptr;
    if (!z) { fprintf(stderr, "z3-fetch: %s\n", z3_error()); return 1; }
    const z3_meta *M = z3_meta_of(z);
    zf_job J = {.z = z, .tot = 1};
    for (int d = 0; d < 3; d++) {
        int64_t a = (int64_t)(b[d] * F) - m, e = (int64_t)((b[d] + b[d + 3]) * F + 0.999999) + m;   /* [a, e) voxels */
        a = a < 0 ? 0 : a; e = e > M->shape[d] ? M->shape[d] : e;
        const int C = M->chunk[d]; J.lo[d] = a / C; J.n[d] = e > a ? (e - 1) / C - J.lo[d] + 1 : 0; J.tot *= J.n[d];
    }
    J.sharded = M->sharded; J.nsh = 1;
    for (int d = 0; d < 3; d++) { const int64_t g = M->shard[d] / M->chunk[d]; J.s0[d] = J.lo[d] / g; J.ns[d] = J.n[d] ? (J.lo[d] + J.n[d] - 1) / g - J.s0[d] + 1 : 0; J.nsh *= J.ns[d]; }
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    pthread_t th[64]; const int n = nt > 64 ? 64 : nt < 1 ? 1 : nt;
    for (int t = 0; t < n; t++) pthread_create(&th[t], nullptr, zf_main, &J);
    for (int t = 0; t < n; t++) pthread_join(th[t], nullptr);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(stderr, "z3-fetch: %s %s: %lld chunks in the box (%lld x %lld x %lld), %lld fetched, %lld failed, %.1f s, %d threads (%lld shards)\n", argv[2], argv[3], (long long)J.tot,
            (long long)J.n[0], (long long)J.n[1], (long long)J.n[2], (long long)J.nf, (long long)J.nfail, (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec), n, (long long)(J.sharded ? J.nsh : 0));
    z3_close(z);
    return J.nfail ? 1 : 0;
}
