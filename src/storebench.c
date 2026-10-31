/* ufsm store-bench: reproducible HTTP / zarr fetch benchmark. Reads n random boxes of each size from one array (no disk cache,
   a fresh reader per size so shard indices are fetched too) and reports decoded and network throughput, requests, bytes per
   request, 429 / 503 / 5xx counts, stalls and rate-limit headers (store_stats). Built into ufsm and as build/store_bench. */
#include "zarr3.h"
#include "sysinfo.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double bnow(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static const char *bopt(int argc, char **argv, const char *name, const char *def) {
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], name)) return argv[i + 1];
    return def;
}
static uint64_t splitmix(uint64_t *x) { uint64_t z = (*x += 0x9e3779b97f4a7c15ull); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; return z ^ (z >> 31); }

typedef struct { z3 *z; int64_t (*o)[3]; int64_t box; int nbox, threads, hash; atomic_int next, failed; atomic_uint_fast64_t digest; } bjob;
static void *breader(void *a) {
    bjob *j = a;
    const size_t nv = (size_t)j->box * j->box * j->box;
    uint8_t *buf = malloc(nv * (size_t)z3_meta_of(j->z)->esize);
    if (!buf) { atomic_store(&j->failed, 1); return nullptr; }
    for (int i; (i = atomic_fetch_add(&j->next, 1)) < j->nbox;) {
        const int64_t n[3] = {j->box, j->box, j->box};
        if (z3_read(j->z, j->o[i], n, buf, j->threads)) { fprintf(stderr, "store-bench: read failed: %s\n", z3_error()); atomic_store(&j->failed, 1); continue; }
        if (j->hash) {   /* order-independent digest of every box's bytes (configurations must agree exactly) */
            uint64_t h = 1469598103934665603ull, k = (uint64_t)i;
            const size_t nb = nv * (size_t)z3_meta_of(j->z)->esize;
            for (size_t v = 0; v < nb; v++) h = (h ^ buf[v]) * 1099511628211ull;
            h ^= splitmix(&k);
            atomic_fetch_add(&j->digest, splitmix(&h));
        }
    }
    free(buf);
    return nullptr;
}

int cmd_store_bench(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: store-bench <root> <array-key> [--box 128,256,512] [--n 8] [--par 1] [--threads 16] [--seed 1] [--hash 1]\n"
                        "       [--lo z,y,x] [--hi z,y,x] (origin region, default the central half of the array)\n"
                        "env: UFSM_HTTP_MULTI / CONC / STREAMS / SHARE / BUF / TAIL / LOG, UFSM_Z3_COALESCE / GAP / SPAN / INDEX_MEM\n");
        return 2;
    }
    const char *root = argv[2], *key = argv[3];
    const int nbox = atoi(bopt(argc, argv, "--n", "8")), par = atoi(bopt(argc, argv, "--par", "1")), threads = ufsm_threads(atoi(bopt(argc, argv, "--threads", "0")), 16);
    uint64_t seed = strtoull(bopt(argc, argv, "--seed", "1"), nullptr, 10);
    char boxes[256]; snprintf(boxes, sizeof boxes, "%s", bopt(argc, argv, "--box", "128,256,512"));
    store *s = store_open(root);
    if (!s) return 1;
    z3 *z0 = z3_open(s, key, nullptr);
    if (!z0) { fprintf(stderr, "store-bench: %s\n", z3_error()); return 1; }
    const z3_meta *m = z3_meta_of(z0);
    int64_t lo[3], hi[3];
    for (int d = 0; d < 3; d++) { lo[d] = m->shape[d] / 4; hi[d] = m->shape[d] - m->shape[d] / 4; }
    const char *l = bopt(argc, argv, "--lo", nullptr), *h = bopt(argc, argv, "--hi", nullptr);
    if (l) { long long a, b, c; if (sscanf(l, "%lld,%lld,%lld", &a, &b, &c) == 3) { lo[0] = a; lo[1] = b; lo[2] = c; } };
    if (h) { long long a, b, c; if (sscanf(h, "%lld,%lld,%lld", &a, &b, &c) == 3) { hi[0] = a; hi[1] = b; hi[2] = c; } };
    fprintf(stderr, "store-bench %s/%s shape %lld %lld %lld, origins in [%lld %lld %lld, %lld %lld %lld), n %d par %d threads %d seed %llu\n",
            root, key, (long long)m->shape[0], (long long)m->shape[1], (long long)m->shape[2], (long long)lo[0], (long long)lo[1], (long long)lo[2],
            (long long)hi[0], (long long)hi[1], (long long)hi[2], nbox, par, threads, (unsigned long long)seed);
    z3_close(z0);
    int rc = 0;
    for (char *tok = strtok(boxes, ","); tok; tok = strtok(nullptr, ",")) {
        const int64_t box = atoll(tok);
        int64_t (*o)[3] = malloc((size_t)nbox * sizeof *o);
        uint64_t rs = seed * 1000003u + (uint64_t)box;
        for (int i = 0; i < nbox; i++)
            for (int d = 0; d < 3; d++) { int64_t span = hi[d] - lo[d] - box; o[i][d] = lo[d] + (span > 0 ? (int64_t)(splitmix(&rs) % (uint64_t)span) : 0); }
        z3 *z = z3_open(s, key, nullptr);   /* fresh reader: cold shard indices */
        store_stats_reset();
        bjob j = {.z = z, .o = o, .box = box, .nbox = nbox, .threads = threads, .hash = atoi(bopt(argc, argv, "--hash", "0"))};
        double t0 = bnow();
        pthread_t th[64]; int np = par < 1 ? 1 : par > 64 ? 64 : par, started = 0;
        for (int i = 1; i < np; i++) if (!pthread_create(&th[started], nullptr, breader, &j)) started++;
        breader(&j);
        for (int i = 0; i < started; i++) pthread_join(th[i], nullptr);
        double dt = bnow() - t0;
        store_stats_t st; store_stats_get(&st);
        const double mvox = (double)nbox * box * box * box / 1e6, mb = st.bytes / 1e6;
        printf("box %4lld n %d par %d: %6.2f s  decoded %7.1f Mvox/s  net %7.2f MB/s (%.1f MB)  %5llu req %7.1f req/s %7.1f KB/req  "
               "mean req %.3f s max %.2f s  429 %llu 503 %llu 5xx %llu transport %llu retries %llu stalls %llu conns %llu ratelimit-hdrs %llu backoff %.1f s%s%s%s\n",
               (long long)box, nbox, np, dt, mvox / dt, mb / dt, mb, (unsigned long long)st.requests, st.requests / dt,
               st.requests ? st.bytes / 1024.0 / st.requests : 0, st.requests ? st.req_seconds / st.requests : 0, st.max_req_seconds,
               (unsigned long long)st.n429, (unsigned long long)st.n503, (unsigned long long)st.n5xx, (unsigned long long)st.transport_errors,
               (unsigned long long)st.retries, (unsigned long long)st.stalls, (unsigned long long)st.connects, (unsigned long long)st.ratelimit, st.sleep_seconds,
               st.last_ratelimit[0] ? "  last: " : "", st.last_ratelimit, "");
        if (j.hash) printf("box %4lld digest %016llx\n", (long long)box, (unsigned long long)atomic_load(&j.digest));
        fflush(stdout);
        if (atomic_load(&j.failed)) rc = 1;
        z3_close(z); free(o);
    }
    store_close(s);
    store_global_cleanup();
    return rc;
}

#ifdef STORE_BENCH_MAIN
int main(int argc, char **argv) {   /* build/store_bench <root> <key> ...: same arguments as ufsm store-bench */
    char **a = calloc((size_t)argc + 2, sizeof *a);
    a[0] = argv[0]; a[1] = "store-bench";
    for (int i = 1; i < argc; i++) a[i + 1] = argv[i];
    return cmd_store_bench(argc + 1, a);
}
#endif
