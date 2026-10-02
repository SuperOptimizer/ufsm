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
typedef struct { store *s; atomic_int failed; } requests;
static void *request(void *arg) {
    requests *r = arg; uint8_t b[64];
    if (store_read(r->s, "blob", 7, 64, b) != 64) atomic_store(&r->failed, 1);
    else for (int i = 0; i < 64; i++) if (b[i] != i + 7) atomic_store(&r->failed, 1);
    return nullptr;
}
static int resources(const char *root) {
    int initial = fd_count();
    for (int cycle = 0; cycle < 3; cycle++) {
        store *s = store_open(root); if (!s) return 1;
        store_set_bearer(s, "fixture-token");
        requests r = {.s = s}; request(&r);
        if (atomic_load(&r.failed)) return 1;
        int baseline = fd_count();
        for (int round = 0; round < 20; round++) {
            pthread_t th[8]; int started = 0;
            for (int i = 0; i < 8; i++) {
                if (pthread_create(&th[started], nullptr, request, &r)) break;
                started++;
            }
            for (int i = 0; i < started; i++) pthread_join(th[i], nullptr);
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
int main(int argc, char **argv) {
    if (argc < 3) return 2;
    if (!strcmp(argv[1], "resources")) return resources(argv[2]);
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
