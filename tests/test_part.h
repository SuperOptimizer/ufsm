/* tests/test_part.h: shards of one test binary for the suite runner. UFSM_TEST_PART=K/N (0 <= K < N) runs only the sections whose
   index i has i % N == K; unset (or N <= 1) runs every section. A long test that is a sequence of independent sections numbers them
   with TEST_PART(i) and gets suite lines NAME~part0of3 ... NAME~part2of3 (one test's checks, run side by side by tests/run_suite.sh);
   the sections and their checks are the same whichever split runs them. Sections must not depend on each other's state.
   (2026-11-13: test_spatial_augment 17 s single-threaded -> three ~6 s shards; the suite wall time is bound by its longest tests.) */
#pragma once
#include <stdio.h>
#include <stdlib.h>

static inline int test_part(int i) {
    static int k = -1, n = 1;
    if (k < 0) {
        const char *e = getenv("UFSM_TEST_PART");
        k = 0;
        if (e && sscanf(e, "%d/%d", &k, &n) == 2 && n > 1 && k >= 0 && k < n) fprintf(stderr, "test part %d of %d\n", k, n);
        else { k = 0; n = 1; }
    }
    return i % n == k;
}
#define TEST_PART(i) if (test_part(i))
