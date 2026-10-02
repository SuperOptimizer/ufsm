/* HTTP/store lifetime and in-process reader repetition, driven by the local Python fixture. */
#include "zarr3.h"
#include <dirent.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

static int fd_count(void) {
    DIR *d = opendir("/proc/self/fd");
    if (!d) return -1;
    int n = 0; struct dirent *e;
    while ((e = readdir(d))) if (strcmp(e->d_name, ".") && strcmp(e->d_name, "..")) n++;
    closedir(d); return n;
}
typedef struct { store *s; atomic_int failed; pthread_barrier_t *barrier; } requests;
static void *request(void *arg) {
    requests *r = arg; uint8_t b[64];
    if (store_read(r->s, "blob", 7, 64, b) != 64) atomic_store(&r->failed, 1);
    else for (int i = 0; i < 64; i++) if (b[i] != i + 7) atomic_store(&r->failed, 1);
    if (r->barrier) pthread_barrier_wait(r->barrier);
    return nullptr;
}
static int resources(const char *root) {
    int initial = fd_count();
    for (int cycle = 0; cycle < 3; cycle++) {
        store *s = store_open(root); if (!s) return 1;
        store_set_bearer(s, "fixture-token");
        requests r = {.s = s}; request(&r);
        if (atomic_load(&r.failed)) return 1;
        int baseline = 0;
        for (int round = 0; round < 20; round++) {
            pthread_barrier_t barrier; pthread_barrier_init(&barrier, nullptr, 9);
            r.barrier = &barrier;
            pthread_t th[8]; int started = 0;
            for (int i = 0; i < 8; i++) {
                if (pthread_create(&th[started], nullptr, request, &r)) break;
                started++;
            }
            if (started != 8) return 1;
            pthread_barrier_wait(&barrier);
            for (int i = 0; i < started; i++) pthread_join(th[i], nullptr);
            pthread_barrier_destroy(&barrier);
            if (!round) baseline = fd_count();
            if (atomic_load(&r.failed) || fd_count() > baseline + 2) {
                fprintf(stderr, "HTTP lifetime: cycle %d round %d fds %d -> %d\n", cycle, round, baseline, fd_count());
                return 1;
            }
        }
        store_close(s); store_global_cleanup();
        if (fd_count() > initial + 2) return 1;
    }
    printf("thread handles and repeated global cleanup: fds %d -> %d\n", initial, fd_count());
    return 0;
}
typedef struct { store *s; int op; pthread_barrier_t *barrier; atomic_int *failed; } task;
static void *fetch(void *arg) {
    task *t = arg; uint8_t b[128], *p = nullptr; size_t n = 0; int error = 0;
    if (t->op == 0) {
        error = store_read(t->s, "blob", 7, 64, b) != 64;
        for (int i = 0; !error && i < 64; i++) error = b[i] != i + 7;
    } else if (t->op == 1) error = store_size(t->s, "blob") != 128;
    else {
        if (t->op == 2) p = store_read_all(t->s, "blob", &n);
        else {
            char url[512]; snprintf(url, sizeof url, "%s/blob", store_root(t->s));
            char *next = nullptr;
            p = store_get_url(t->s, url, &n, &next);
            error = next != nullptr; free(next);
        }
        error |= !p || n != 128;
        for (int i = 0; !error && i < 128; i++) error = p[i] != i;
        free(p);
    }
    if (error) atomic_store(t->failed, 1);
    if (t->barrier) pthread_barrier_wait(t->barrier);
    return nullptr;
}
static int reuse_checks(int argc, char **argv) {
    if (argc < 3) return 2;
    int initial = fd_count(); atomic_int failed = 0;
    if (!strcmp(argv[1], "handoff") || !strcmp(argv[1], "reconnect")) {
        for (int i = 0; i < 60; i++) {
            const char *mode = i % 3 == 0 ? "auth-a" : i % 3 == 1 ? "auth-b" : "noauth";
            int reconnect = !strcmp(argv[1], "reconnect");
            if (reconnect) mode = i % 2 ? "noauth" : "drop-idle";
            char root[512]; snprintf(root, sizeof root, "%s/%s", argc > 3 && i % 2 ? argv[3] : argv[2], mode);
            store *s = store_open(root); if (!s) return 1;
            store_set_bearer(s, reconnect ? nullptr : i % 3 == 0 ? "fixture-a" : i % 3 == 1 ? "fixture-b" : nullptr);
            task t = {.s=s, .op=i % 4, .failed=&failed}; pthread_t th;
            if (pthread_create(&th, nullptr, fetch, &t)) return 1;
            pthread_join(th, nullptr); store_close(s);
            if (atomic_load(&failed)) return 1;
        }
    } else if (!strcmp(argv[1], "bounded")) {
        char root[512]; snprintf(root, sizeof root, "%s/noauth", argv[2]);
        store *s = store_open(root); if (!s) return 1;
        int baseline = 0;
        for (int round = 0; round < 3; round++) {
            pthread_barrier_t barrier; pthread_barrier_init(&barrier, nullptr, 65);
            task t = {.s=s, .barrier=&barrier, .failed=&failed}; pthread_t th[64];
            for (int i = 0; i < 64; i++) if (pthread_create(&th[i], nullptr, fetch, &t)) return 1;
            pthread_barrier_wait(&barrier);
            for (int i = 0; i < 64; i++) pthread_join(th[i], nullptr);
            pthread_barrier_destroy(&barrier);
            int count = fd_count();
            if (!round) baseline = count;
            if (atomic_load(&failed) || count > baseline + 2) return 1;
            printf("bounded round %d idle fds %d\n", round, count);
        }
        store_close(s);
    } else return 2;
    store_global_cleanup();
    printf("cleanup fds %d -> %d\n", initial, fd_count());
    return fd_count() != initial;
}

int main(int argc, char **argv) {
    if (argc < 3) return 2;
    if (!strcmp(argv[1], "resources")) return resources(argv[2]);
    if (!strcmp(argv[1], "handoff") || !strcmp(argv[1], "bounded") || !strcmp(argv[1], "reconnect")) return reuse_checks(argc, argv);
    store *s = store_open(argv[2]); if (!s) return 1;
    int rc = 0;
    if (!strcmp(argv[1], "range") && argc == 7) {
        int64_t off = atoll(argv[4]), n = atoll(argv[5]), expected = atoll(argv[6]);
        uint8_t b[512]; if (n > 512) return 2;
        int64_t got = store_read(s, argv[3], off, n, b);
        if (got != expected) { fprintf(stderr, "range returned %lld, expected %lld\n", (long long)got, (long long)expected); rc = 1; }
        if (got > 0) for (int i = 0; i < got; i++) if (b[i] != (uint8_t)(off + i)) rc = 1;
    } else if (!strcmp(argv[1], "repeat") && argc == 6) {
        z3 *z = z3_open(s, argv[3], strcmp(argv[4], "-") ? argv[4] : nullptr);
        if (!z) { fprintf(stderr, "%s\n", z3_error()); return 1; }
        const int64_t o[3] = {-2, -1, -3}, n[3] = {17, 18, 18};
        uint8_t b[17 * 18 * 18]; int baseline = 0;
        for (int i = 0; i < 30; i++) {
            if (z3_read(z, o, n, b, 16)) { fprintf(stderr, "%s\n", z3_error()); rc = 1; break; }
            if (!i) baseline = fd_count();
            else if (fd_count() > baseline + 2) { fprintf(stderr, "reader fds %d -> %d\n", baseline, fd_count()); rc = 1; break; }
        }
        if (!rc) { FILE *f = fopen(argv[5], "wb"); if (!f) return 1; rc = fwrite(b, 1, sizeof b, f) != sizeof b; fclose(f); }
        z3_close(z);
    } else if (!strcmp(argv[1], "prefetch") && argc == 5) {
        z3 *z = z3_open(s, argv[3], argv[4]); if (!z) return 1;
        if (z3_prefetch_chunk(z, 0, 0, 0) != 1 || z3_prefetch_chunk(z, 0, 0, 0) != 0) rc = 1;
        z3_close(z);
    } else rc = 2;
    store_close(s); store_global_cleanup(); return rc;
}
