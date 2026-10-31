/* zshm: decoded volcomp substreams shared between processes (a memory-mapped, size-capped, set-associative LRU on /dev/shm).
 *
 * Neighbouring wraps / stripes / faces runs decode the same CT chunks in different processes; a 9.6 um chunk (1.2 mm) spans many
 * wraps. With UFSM_ZCACHE_GB=G (> 0) every process maps one cache file (UFSM_ZCACHE_PATH, default /dev/shm/ufsm-zcache-<uid>) of
 * about G GB and z3 reads of lossy volcomp chunks look each needed substream (16 z x 32 y x 128 x = 64 KB) up there before
 * decoding it, and store what they decode. Keys are content hashes (XXH3-128 of the chunk's compressed bytes, + the substream), so
 * a hit is the same bytes a decode would give, whatever volume, store or path the chunk came from. Readers never block (a
 * per-slot sequence lock: a slot rewritten during the copy is a miss); writers claim a slot with a CAS, so a crashed process can at
 * worst leave a slot unusable for 10 minutes. The file keeps its first size: delete it to resize (or to drop the cache). Unset /
 * 0: off. UFSM_ZCACHE_STATS=1 prints hits / misses / puts at exit. */
#pragma once
/* header-only (static functions): included by zarr3.c, the only user */
#include <stdint.h>

#define ZSHM_SUB_BYTES 65536u   /* one substream of a 128^3 u8 chunk */
/* zshm_on(): the cache is mapped (opens it on first use).
   zshm_get(k0, k1, s, dst): copy substream s of the chunk with content hash (k0, k1) into the 128^3 chunk buffer dst; 1 hit, 0 miss.
   zshm_put(k0, k1, s, src): store substream s of the decoded chunk src (128^3) under (k0, k1).
   zshm_stats(): this process's hits / misses / stores. */
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define ZSHM_MAGIC 0x31454843535a4655ull   /* "UFZSCHE1" */
#define ZSHM_WAYS 16u
#define ZSHM_STALE_S 600u                  /* a slot claimed this long ago by a writer that never published: reclaimable */

typedef struct {
    _Atomic uint64_t magic;   /* 0 new file, 1 being initialised, ZSHM_MAGIC ready */
    uint64_t nsets, ways, nslots, data_off, file_bytes;
    _Atomic uint64_t clock;   /* recency stamps */
} zshm_hdr;
typedef struct {
    _Atomic uint64_t seq;     /* even: stable; odd: a writer owns the slot */
    _Atomic uint64_t k0, k1;  /* content key (0, 0 = empty) */
    _Atomic uint64_t stamp;   /* last use (hdr clock) */
    _Atomic uint64_t wtime;   /* wall seconds when a writer claimed it */
    uint64_t pad[3];
} zshm_slot;

static zshm_hdr *g_zs_h;
static zshm_slot *g_zs_slots;
static uint8_t *g_zs_data;
static pthread_once_t g_zs_once = PTHREAD_ONCE_INIT;
static _Atomic uint64_t g_zs_hits, g_zs_misses, g_zs_puts;

static uint64_t zshm_now_s(void) { struct timespec t; clock_gettime(CLOCK_REALTIME, &t); return (uint64_t)t.tv_sec; }
static void zshm_stats_atexit(void) {
    fprintf(stderr, "zshm: %llu substream hits, %llu misses, %llu stored\n", (unsigned long long)atomic_load(&g_zs_hits),
            (unsigned long long)atomic_load(&g_zs_misses), (unsigned long long)atomic_load(&g_zs_puts));
}

static void zshm_init(void) {
    const char *e = getenv("UFSM_ZCACHE_GB");
    double gb = e && *e ? atof(e) : 0.0;
    if (!(gb > 0)) return;
    char path[512];
    const char *p = getenv("UFSM_ZCACHE_PATH");
    if (p && *p) snprintf(path, sizeof path, "%s", p);
    else snprintf(path, sizeof path, "/dev/shm/ufsm-zcache-%u", (unsigned)getuid());
    uint64_t nslots = (uint64_t)(gb * 1073741824.0) / (ZSHM_SUB_BYTES + sizeof(zshm_slot));
    nslots -= nslots % ZSHM_WAYS;
    if (nslots < ZSHM_WAYS) return;
    const uint64_t slots_off = 4096, data_off = (slots_off + nslots * sizeof(zshm_slot) + 4095) & ~4095ull;
    const uint64_t want = data_off + nslots * ZSHM_SUB_BYTES;
    int fd = open(path, O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (fd < 0) { fprintf(stderr, "zshm: cannot open %s: cache off\n", path); return; }
    struct stat st;
    if (fstat(fd, &st) || (st.st_size == 0 && ftruncate(fd, (off_t)want))) { close(fd); fprintf(stderr, "zshm: cannot size %s: cache off\n", path); return; }
    if (fstat(fd, &st) || st.st_size < 4096) { close(fd); return; }
    uint64_t fbytes = (uint64_t)st.st_size;
    void *m = mmap(nullptr, fbytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (m == MAP_FAILED) { fprintf(stderr, "zshm: cannot map %s: cache off\n", path); return; }
    zshm_hdr *h = m;
    uint64_t zero = 0;
    if (atomic_compare_exchange_strong(&h->magic, &zero, 1)) {   /* first user: lay out the file it sized (maybe another process did) */
        uint64_t n = (fbytes - data_off) / ZSHM_SUB_BYTES;
        if (fbytes != want) {   /* sized by someone else with another G before we initialised: fit the slots to the file */
            for (n = fbytes / (ZSHM_SUB_BYTES + sizeof(zshm_slot)); n >= ZSHM_WAYS; n -= ZSHM_WAYS) {
                n -= n % ZSHM_WAYS;
                uint64_t d = (slots_off + n * sizeof(zshm_slot) + 4095) & ~4095ull;
                if (d + n * ZSHM_SUB_BYTES <= fbytes) break;
            }
        }
        if (n < ZSHM_WAYS) { atomic_store(&h->magic, 0); munmap(m, fbytes); return; }
        h->nslots = n; h->ways = ZSHM_WAYS; h->nsets = n / ZSHM_WAYS;
        h->data_off = (slots_off + n * sizeof(zshm_slot) + 4095) & ~4095ull; h->file_bytes = fbytes;
        atomic_store(&h->clock, 1);
        atomic_store_explicit(&h->magic, ZSHM_MAGIC, memory_order_release);
    } else {
        for (int i = 0; i < 5000 && atomic_load_explicit(&h->magic, memory_order_acquire) != ZSHM_MAGIC; i++) usleep(1000);
        if (atomic_load_explicit(&h->magic, memory_order_acquire) != ZSHM_MAGIC || h->file_bytes != fbytes || h->ways != ZSHM_WAYS ||
            h->data_off + h->nslots * ZSHM_SUB_BYTES > fbytes) {
            fprintf(stderr, "zshm: %s is not a usable cache file (delete it): cache off\n", path);
            munmap(m, fbytes); return;
        }
    }
    g_zs_slots = (zshm_slot *)((uint8_t *)m + slots_off);
    g_zs_data = (uint8_t *)m + h->data_off;
    g_zs_h = h;
    if ((e = getenv("UFSM_ZCACHE_STATS")) && *e && strcmp(e, "0")) atexit(zshm_stats_atexit);
}

static int zshm_on(void) { pthread_once(&g_zs_once, zshm_init); return g_zs_h != nullptr; }

static inline uint64_t zshm_mix(uint64_t x) { x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33; return x; }
/* the substream folded into the key; never (0, 0) */
static inline void zshm_skey(uint64_t k0, uint64_t k1, uint32_t s, uint64_t *a, uint64_t *b) {
    *a = k0 ^ zshm_mix(s + 1u); *b = k1 + (uint64_t)s * 0x9e3779b97f4a7c15ull;
    if (!*a && !*b) *a = 1;
}
static inline size_t zshm_sub_off(uint32_t s, uint32_t z) { return ((size_t)(16u * (s >> 2) + z) * 128u + 32u * (s & 3u)) * 128u; }

[[maybe_unused]] static int zshm_get(uint64_t k0, uint64_t k1, uint32_t s, uint8_t *dst) {
    if (!zshm_on()) return 0;
    uint64_t a, b; zshm_skey(k0, k1, s, &a, &b);
    const uint64_t set = zshm_mix(a ^ b) % g_zs_h->nsets;
    for (uint32_t w = 0; w < ZSHM_WAYS; w++) {
        zshm_slot *sl = &g_zs_slots[set * ZSHM_WAYS + w];
        uint64_t q1 = atomic_load_explicit(&sl->seq, memory_order_acquire);
        if (q1 & 1u) continue;
        if (atomic_load_explicit(&sl->k0, memory_order_relaxed) != a || atomic_load_explicit(&sl->k1, memory_order_relaxed) != b) continue;
        const uint8_t *src = g_zs_data + (set * ZSHM_WAYS + w) * (size_t)ZSHM_SUB_BYTES;
        for (uint32_t z = 0; z < 16; z++) memcpy(dst + zshm_sub_off(s, z), src + (size_t)z * 4096u, 4096);
        atomic_thread_fence(memory_order_acquire);
        if (atomic_load_explicit(&sl->seq, memory_order_relaxed) != q1) break;   /* rewritten while we copied: decode instead */
        atomic_store_explicit(&sl->stamp, atomic_fetch_add_explicit(&g_zs_h->clock, 1, memory_order_relaxed), memory_order_relaxed);
        atomic_fetch_add_explicit(&g_zs_hits, 1, memory_order_relaxed);
        return 1;
    }
    atomic_fetch_add_explicit(&g_zs_misses, 1, memory_order_relaxed);
    return 0;
}

[[maybe_unused]] static void zshm_put(uint64_t k0, uint64_t k1, uint32_t s, const uint8_t *src) {
    if (!zshm_on()) return;
    uint64_t a, b; zshm_skey(k0, k1, s, &a, &b);
    const uint64_t set = zshm_mix(a ^ b) % g_zs_h->nsets, t = zshm_now_s();
    zshm_slot *victim = nullptr; uint64_t vseq = 0, vstamp = UINT64_MAX;
    for (uint32_t w = 0; w < ZSHM_WAYS; w++) {
        zshm_slot *sl = &g_zs_slots[set * ZSHM_WAYS + w];
        uint64_t q = atomic_load_explicit(&sl->seq, memory_order_acquire);
        uint64_t sk0 = atomic_load_explicit(&sl->k0, memory_order_relaxed), sk1 = atomic_load_explicit(&sl->k1, memory_order_relaxed);
        if (!(q & 1u) && sk0 == a && sk1 == b) return;   /* already there */
        uint64_t st;
        if (q & 1u) {   /* a writer owns it: only a stale claim is a candidate */
            if (t < atomic_load_explicit(&sl->wtime, memory_order_relaxed) + ZSHM_STALE_S) continue;
            st = 0;
        } else st = (sk0 | sk1) ? atomic_load_explicit(&sl->stamp, memory_order_relaxed) : 0;
        if (!victim || st < vstamp) { victim = sl; vseq = q; vstamp = st; }
    }
    if (!victim) return;
    uint64_t claim = (vseq | 1u) + ((vseq & 1u) ? 2u : 0u);   /* even q -> q + 1; a stale odd q -> q + 2 */
    if (!atomic_compare_exchange_strong(&victim->seq, &vseq, claim)) return;   /* someone else took it */
    atomic_thread_fence(memory_order_release);   /* the odd seq is visible before any of the new bytes */
    atomic_store_explicit(&victim->wtime, t, memory_order_relaxed);
    atomic_store_explicit(&victim->k0, 0, memory_order_relaxed);
    atomic_store_explicit(&victim->k1, 0, memory_order_relaxed);
    uint8_t *d = g_zs_data + (size_t)(victim - g_zs_slots) * ZSHM_SUB_BYTES;
    for (uint32_t z = 0; z < 16; z++) memcpy(d + (size_t)z * 4096u, src + zshm_sub_off(s, z), 4096);
    atomic_store_explicit(&victim->k0, a, memory_order_relaxed);
    atomic_store_explicit(&victim->k1, b, memory_order_relaxed);
    atomic_store_explicit(&victim->stamp, atomic_fetch_add_explicit(&g_zs_h->clock, 1, memory_order_relaxed), memory_order_relaxed);
    uint64_t own = claim;
    if (atomic_compare_exchange_strong_explicit(&victim->seq, &own, claim + 1u, memory_order_release, memory_order_relaxed))
        atomic_fetch_add_explicit(&g_zs_puts, 1, memory_order_relaxed);
}

[[maybe_unused]] static void zshm_stats(uint64_t *hits, uint64_t *misses, uint64_t *puts) {
    *hits = atomic_load(&g_zs_hits); *misses = atomic_load(&g_zs_misses); *puts = atomic_load(&g_zs_puts);
}
