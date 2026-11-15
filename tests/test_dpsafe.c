/* DP safety of the nn layer's process-wide state (2026-11-14 audit, DESIGN "DP safety"): two host threads as the trainer's per-GPU workers.
   (1) CUDA error latches are per thread: a worker's error reaches the main thread's nn_check (posted), a worker's nn_check never reports
       another thread's error; (2) the GN slab registry under concurrent registration from two threads (16 entries each); (3) the
   stochastic-rounding replica salt: salt 0 keeps the one-GPU seed sequence, another salt changes it; (4) the VMM granularity is the
   current device's. One GPU is enough (both threads on device 0). */
#include "nn.h"
#include "nn_lp.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); fails++; } } while (0)

static void *worker_err(void *p) {   /* a worker: a failed allocation latches its own error (and posts it); its nn_check sees it once */
    (void)p; nn_init(0); nn_err_worker(1);
    /* (not an invalid copy: on a driver in HMM addressing mode (the desk, 595 open module) cudaMemcpy takes (void *)1 as pageable host
       memory and the copy segfaults in the driver instead of failing; an impossible cudaMalloc fails on every driver without touching
       memory. UFSM_MALLOC_NULL=1 (set in main): nn_malloc returns null instead of ending the process) */
    void *q = nn_malloc((size_t)1 << 60); if (q) nn_free(q);
    return nullptr;
}
static void *worker_quiet(void *p) {   /* a worker whose nn_check must not take another thread's posted error */
    int *out = p; nn_init(0); nn_err_worker(1); *out = nn_check() != nullptr; return nullptr;
}
static float g_keys[2][16];
static void *worker_slab(void *p) {
    const int t = (int)(long)p; nn_init(0);
    for (int i = 0; i < 16; i++) lp_gn_slab_reg(&g_keys[t][i], 8, 0, 2 + t, 100 * t + i);
    return nullptr;
}
static unsigned seeds(unsigned salt, int n, unsigned *out) {
    nn_sr_salt(salt); nn_set_sr_step(7); for (int i = 0; i < n; i++) out[i] = nn_sr_seed(); nn_sr_salt(0); return out[0];
}

int main(void) {
    setenv("UFSM_MALLOC_NULL", "1", 1);   /* (worker_err's failed allocation returns null, latched, instead of ending the process) */
    if (nn_init(0)) { fprintf(stderr, "test_dpsafe: no GPU\n"); return 1; }
    (void)nn_check();
    /* (1) error latches */
    pthread_t th; pthread_create(&th, nullptr, worker_err, nullptr); pthread_join(th, nullptr);
    int quiet = -1; pthread_create(&th, nullptr, worker_quiet, &quiet); pthread_join(th, nullptr);
    CHECK(quiet == 0, "a worker's nn_check reported another thread's error");
    const char *e = nn_check();
    CHECK(e != nullptr, "the main thread's nn_check missed the worker's error");
    if (e) { CHECK(strstr(nn_check_site(), "thread") != nullptr, "the site does not name the worker's thread: %s", nn_check_site()); fprintf(stderr, "main saw: %s at %s\n", e, nn_check_site()); }
    CHECK(nn_check() == nullptr, "the posted error was reported twice");
    /* (2) GN slab registry from two threads */
    pthread_t a, b; pthread_create(&a, nullptr, worker_slab, (void *)0L); pthread_create(&b, nullptr, worker_slab, (void *)1L);
    pthread_join(a, nullptr); pthread_join(b, nullptr);
    for (int t = 0; t < 2; t++) for (int i = 0; i < 16; i++) {
        gnp_t g = gnp_mk(nullptr, nullptr, &g_keys[t][i], nullptr, 4);
        CHECK(g.ns == 2 + t && g.zhw == 100 * t + i, "slab entry %d/%d lost (ns %d zhw %d)", t, i, g.ns, g.zhw);
    }
    for (int t = 0; t < 2; t++) for (int i = 0; i < 16; i++) lp_gn_slab_reg(&g_keys[t][i], 0, 0, 0, 0);
    CHECK(!lp_gn_slab_any(), "slab registry not empty after removing every entry");
    /* (3) SR replica salt */
    unsigned s0[8], s0b[8], s1[8];
    seeds(0, 8, s0); seeds(0, 8, s0b); seeds(1, 8, s1);
    CHECK(!memcmp(s0, s0b, sizeof s0), "salt 0 seeds not reproducible");
    int same = 0; for (int i = 0; i < 8; i++) same += s0[i] == s1[i];
    CHECK(same == 0, "salt 1 shares %d of 8 seeds with salt 0", same);
    /* (4) VMM granularity of the current device */
    const size_t g = nn_ring_gran();
    fprintf(stderr, "vmm granularity on device 0: %zu\n", g);
    CHECK(nn_check() == nullptr, "a CUDA error left over");
    if (fails) { fprintf(stderr, "test_dpsafe: %d failures\n", fails); return 1; }
    fprintf(stderr, "test_dpsafe: ok\n");
    return 0;
}
