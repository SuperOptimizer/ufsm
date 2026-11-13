/* src/winauto.h: predict --window auto's choice on fake memory needs (host only). The ink candidates (576 / 1088 / 1600 at halo 32) and
   the fields candidates (384 x 640^2, 640^3, 1152^3 at halo 64); need = 30 B per window voxel + 1.5 GB (the order of the real dry runs).
   Checks: the interior cap (UFSM_PRED_WINDOW_MAX, default 1024) keeps a free 98 GB card off 1600^3 (pod 3, 11-12: 90 GB taken);
   UFSM_MEM_FREE_GB caps the budget; UFSM_PRED_WINDOW_FREE replaces the device figure and is capped too; candidate 0 copy is the fallback
   when nothing fits; prod_only; direct slabs win over copy when both fit. */
#include "winauto.h"
#include <assert.h>
#include <stdio.h>

typedef struct { const int *cz, *cw; size_t direct_extra; } fake;
static size_t need(int k, int d, void *vc) {
    const fake *f = vc;
    return (size_t)f->cz[k] * f->cw[k] * f->cw[k] * 30 + ((size_t)3 << 29) + (d ? f->direct_extra : 0);
}
#define GB 1000000000ull
int main(void) {
    unsetenv("UFSM_PRED_WINDOW_FREE"); unsetenv("UFSM_MEM_FREE_GB"); unsetenv("UFSM_PRED_WINDOW_MAX");
    const int iz[3] = {576, 1088, 1600}, fz[3] = {384, 640, 1152}, fx[3] = {640, 640, 1152};
    fake ink = {iz, iz, 0}, fld = {fz, fx, 0};
    int d = -1; double c = 0;
    /* 1. budget */
    assert(winauto_budget(98 * GB) == 98 * GB);
    setenv("UFSM_MEM_FREE_GB", "20", 1); assert(winauto_budget(98 * GB) == 20 * GB); assert(winauto_budget(8 * GB) == 8 * GB);
    setenv("UFSM_PRED_WINDOW_FREE", "50", 1); assert(winauto_budget(8 * GB) == 20 * GB);   /* planning figure, still capped */
    unsetenv("UFSM_MEM_FREE_GB"); assert(winauto_budget(8 * GB) == 50 * GB);
    setenv("UFSM_MEM_FREE_GB", "0", 1); assert(winauto_budget(8 * GB) == 50 * GB);           /* 0 / junk: no cap */
    unsetenv("UFSM_PRED_WINDOW_FREE"); unsetenv("UFSM_MEM_FREE_GB");
    assert(winauto_max_interior() == 1024);
    setenv("UFSM_PRED_WINDOW_MAX", "2048", 1); assert(winauto_max_interior() == 2048); unsetenv("UFSM_PRED_WINDOW_MAX");
    /* 2. ink with a budget that holds 1600^3 (interior 1536): uncapped it wins (the 11-12 pod 3 pick), capped at 1024 -> 1088^3 */
    const size_t n1600 = need(2, 1, &ink), n1088 = need(1, 1, &ink), n576 = need(0, 1, &ink);
    printf("need 576 %.1f GB, 1088 %.1f GB, 1600 %.1f GB\n", n576 / 1e9, n1088 / 1e9, n1600 / 1e9);
    int k = winauto_pick(3, iz, iz, 32, 1, 0, 1.22, 2048, n1600, need, &ink, &d, &c);
    assert(k == 2 && d == 1);                                       /* uncapped, the whole card: the biggest window (the old behaviour) */
    k = winauto_pick(3, iz, iz, 32, 1, 0, 1.22, 1024, n1600, need, &ink, &d, &c);
    assert(k == 1 && d == 1 && c > 1.19 && c < 1.21);               /* capped: 1088^3, 1.20 computed per interior voxel */
    /* 3. the budget decides below the cap: a 16 GB share -> 576 */
    k = winauto_pick(3, iz, iz, 32, 1, 0, 1.22, 1024, 16 * GB, need, &ink, &d, &c);
    assert(k == 0 && d == 1);
    /* 4. nothing fits: candidate 0 with copy slabs (the fallback) */
    k = winauto_pick(3, iz, iz, 32, 1, 0, 1.22, 1024, 1 * GB, need, &ink, &d, &c);
    assert(k == 0 && d == 0 && c > 1.42 * 1.22 - 0.01 && c < 1.42 * 1.22 + 0.01);
    /* 5. direct slabs need 4 GB more: with room for 1088^3 copy only, 576^3 direct (1.42) still beats 1088^3 copy (1.20 x 1.22 = 1.46) */
    fake inkx = {iz, iz, 4 * GB};
    k = winauto_pick(3, iz, iz, 32, 1, 0, 1.22, 1024, need(1, 0, &inkx) + GB, need, &inkx, &d, &c);
    assert(k == 0 && d == 1);
    k = winauto_pick(3, iz, iz, 32, 1, 0, 1.10, 1024, need(1, 0, &inkx) + GB, need, &inkx, &d, &c);   /* a smaller copy penalty: 1088 copy */
    assert(k == 1 && d == 0);
    k = winauto_pick(3, iz, iz, 32, 0, 0, 1.22, 1024, 98 * GB, need, &ink, &d, &c);   /* UFSM_SLAB_DIRECT=0 */
    assert(d == 0);
    /* 6. prod_only: candidate 0 whatever the budget */
    k = winauto_pick(3, iz, iz, 32, 1, 1, 1.22, 1024, 98 * GB, need, &ink, &d, &c);
    assert(k == 0);
    /* 7. fields: 1152^3 (interior 1024) is under the cap and wins on a big card; 640^3 on a 16 GB share; 384 x 640^2 when tight */
    k = winauto_pick(3, fz, fx, 64, 1, 0, 1.25, 1024, 98 * GB, need, &fld, &d, &c);
    assert(k == 2);
    k = winauto_pick(3, fz, fx, 64, 1, 0, 1.25, 1024, need(1, 1, &fld), need, &fld, &d, &c);
    assert(k == 1);
    k = winauto_pick(3, fz, fx, 64, 1, 0, 1.25, 1024, need(0, 1, &fld), need, &fld, &d, &c);
    assert(k == 0 && d == 1);
    k = winauto_pick(3, fz, fx, 64, 1, 0, 1.25, 960, 98 * GB, need, &fld, &d, &c);   /* a cap below 1024 drops 1152^3 */
    assert(k == 1);
    printf("test_winauto: ok\n");
    return 0;
}
