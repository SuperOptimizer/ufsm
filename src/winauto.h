/* predict --window auto: the window choice, host only (tests/test_winauto.c).
   Candidates (z, xy) with a common halo, production (k = 0) first; cost = computed voxels per interior voxel, x copy_pen without direct
   slabs. The cheapest candidate whose memory need fits the budget wins; candidate 0 with copy slabs is the fallback that always fits.
   Budget (winauto_budget): the device's free memory, capped by UFSM_MEM_FREE_GB (the GPU's share when other jobs use the card; the same
   variable train's planner reads); UFSM_PRED_WINDOW_FREE GB replaces the device figure outright (planning for another card, with
   UFSM_PRED_WINDOW_DRY=1). Size cap: interiors (window - 2 halo) above UFSM_PRED_WINDOW_MAX (default 1024) are never picked.
   Why (11-12, pod 3): with a free 98 GB card the ink pass picked 1600^3 (interior 1536) and took 90 GB; the other unroll lane's slab
   fits and count-field ran from the ~7 GB left (managed-memory spill: 602 MB of device each) and the ink itself slowed from 2 s to 260 s
   per window. The cap costs the ink at most 1.20 vs 1.13 computed voxels per interior voxel (1088^3 vs 1600^3, ~6 %); the fields'
   largest candidate (1152^3, interior 1024) is unaffected. */
#pragma once
#include <stddef.h>
#include <stdlib.h>

typedef size_t (*winauto_need_fn)(int k, int direct, void *ctx);   /* device bytes candidate k needs (direct = 1: direct slabs) */

static inline size_t winauto_budget(size_t dev_free) {
    const char *pf = getenv("UFSM_PRED_WINDOW_FREE"), *mf = getenv("UFSM_MEM_FREE_GB");
    size_t fr = pf && atof(pf) > 0 ? (size_t)(atof(pf) * 1e9) : dev_free;
    if (mf && atof(mf) > 0 && (size_t)(atof(mf) * 1e9) < fr) fr = (size_t)(atof(mf) * 1e9);
    return fr;
}
static inline int winauto_max_interior(void) {
    const char *m = getenv("UFSM_PRED_WINDOW_MAX"); const int v = m ? atoi(m) : 0;
    return v > 0 ? v : 1024;
}
/* returns the chosen candidate; *direct = its slab mode, *cost = its cost. prod_only: candidate 0 only (UFSM_PRED_WINDOW=prod) */
static inline int winauto_pick(int nc, const int *cz, const int *cw, int halo, int dir_ok, int prod_only, double copy_pen, int max_interior,
                               size_t budget, winauto_need_fn need, void *ctx, int *direct, double *cost) {
    int bk = 0, bd = 0; double bc = 1e30;
    for (int k = 0; k < (prod_only ? 1 : nc); k++) for (int d = dir_ok; d >= 0; d--) {
        const int iz = cz[k] - 2 * halo, ix = cw[k] - 2 * halo;
        if (iz <= 0 || ix <= 0) continue;
        const int fallback = k == 0 && d == 0;
        if (!fallback && (iz > max_interior || ix > max_interior)) continue;
        const double c = (double)cz[k] * cw[k] * cw[k] / ((double)iz * ix * ix) * (d ? 1.0 : copy_pen);
        if ((fallback || need(k, d, ctx) <= budget) && c < bc - 1e-9) { bc = c; bk = k; bd = d; }
    }
    if (direct) *direct = bd;
    if (cost) *cost = bc;
    return bk;
}
