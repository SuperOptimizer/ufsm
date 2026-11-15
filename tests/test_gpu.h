/* tests/test_gpu.h: how a GPU test reports a device it could not use (2026-11-14, mini audit suite.md 1-2).
   test_no_gpu(name, what): the device init (or a device-only part) failed.
   - A GPU was granted (CUDA_VISIBLE_DEVICES non-empty: the suite runner gave this gpu-tagged test a card and its lock; or the variable
     unset on a machine with an NVIDIA device): that is a failure. Prints "FAIL: ..." and returns 1 (the exit code).
   - No GPU (CUDA_VISIBLE_DEVICES set and empty, which the runner gives untagged tests and -g none; or no /dev/nvidiactl): prints
     "SKIP: no GPU: ..." and returns 0. tests/run_suite.sh reports SKIP and exits 3 (INCOMPLETE) when a GPU test did not run.
   The old `printf("SKIP: no GPU"); return 0;` turned a dead context on a granted card into a green gate.
   test_partial(name, what): a part of the test needs what this machine lacks (a second GPU, a desk checkpoint); the rest ran.
   Prints "PARTIAL: ..." (the runner lists these in its summary; the status stays PASS). */
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static inline int test_gpu_expected(void) {
    const char *cv = getenv("CUDA_VISIBLE_DEVICES");
    if (cv) return cv[0] != 0 && strcmp(cv, "-1") != 0;
    return access("/dev/nvidiactl", F_OK) == 0;
}
static inline int test_no_gpu(const char *name, const char *what) {
    if (test_gpu_expected()) {
        const char *cv = getenv("CUDA_VISIBLE_DEVICES");
        printf("FAIL: %s: %s, although a GPU was granted (CUDA_VISIBLE_DEVICES=%s)\n", name, what, cv ? cv : "(unset, /dev/nvidiactl present)");
        fflush(stdout); return 1;
    }
    printf("SKIP: no GPU: %s: %s\n", name, what); fflush(stdout); return 0;
}
static inline void test_partial(const char *name, const char *what) { printf("PARTIAL: %s: %s\n", name, what); fflush(stdout); }
