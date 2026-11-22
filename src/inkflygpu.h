/* the on-the-fly ink chunk on the GPU (src/nn_inkfly.cu; inksplat.c inkfly_chunk's face search, splat and finalize, byte for byte):
   double arithmetic with no FMA contraction (-fmad=false), IEEE sqrt / division, the float [1 2 1] smoothing in the CPU's order.
   The host keeps the I/O: the candidate tiles, the label rectangles (lsrc_read), the mesh patches and the CT box (zcache / z3_read).
   Declared weak: without nn_inkfly.o the CPU code runs. */
#pragma once
#include <stddef.h>
#include <stdint.h>

/* held-out boxes per scan in the ink onfly path (inksplat_cfg.excl, inkfly cx, the GPU chunk config dc_t; was 64 until 2026-11-18: 1667 has
   74 in holdouts-20261118, every one on a labelled mesh). The boxes live in device memory (dc_t via a pointer), not in kernel parameters. */
#define INKFLY_MAXEXCL 256
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    const uint8_t *lab, *msk;          /* the label / mask rectangle (rh x rw, 0/255 rule applied; msk nullable) */
    int64_t rh, rw, rr0, rc0;          /* its size and origin within the label window */
    int64_t lh, lw;                    /* the label window (tile clipping) */
    const int32_t *tiles; int ntiles, ntx;   /* candidate tile ids (t = ty * ntx + tx, 64^2 label px each) */
    const float *X, *Y, *Z; int pu0, pv0, pw, ph, mw, mh; double scale;   /* the mesh patch: grid cols [pu0, pu0 + pw), rows [pv0, pv0 + ph) of a mw x mh grid */
    double F; int64_t glr0, glc0;      /* canvas px per label px; geo lr0 / lc0 (the rectangle's origin in label px) */
    int nsub, nt; double tstep;
} ifg_unit;
typedef struct {
    double offr, offc, sign, face_a, thr, V, off, slack; int lo, ns, iso, run, gap, wlo, whi;   /* (wlo / whi: face_of's run window, -4 / 8) */
    int64_t a0[3], b0[3];              /* the chunk's level-0 box [a0, b0) (the near test) */
    int64_t shape[3]; int nexcl; const int64_t (*excl)[6];
} ifg_cfg;
/* two stages per chunk, on the calling thread's own stream and buffers (any number of threads). near: the units' near pixels (supervised,
   on a valid cell, the point within the chunk's level-0 box +- slack) and the level-0 box [lo, hi] their samples reach; returns 0 ok
   (*nnear), 1 a mesh patch was too small (run the CPU code), < 0 a CUDA failure. finish (after near, with the CT box ctb / co / cn in level
   voxels covering [lo, hi]): face, splat, finalize of chunk o / n (level voxels, n <= 128^3) into out (host, n voxels, 255 = unknown). */
__attribute__((weak)) int inkfly_gpu_near(const ifg_unit *U, int nu, const ifg_cfg *c, int64_t lo[3], int64_t hi[3], uint64_t *nnear);
__attribute__((weak)) int inkfly_gpu_finish(const ifg_cfg *c, const uint8_t *ctb, const int64_t co[3], const int64_t cn[3], const int64_t o[3], const int64_t n[3],
                                            uint8_t *out, int *nonempty);
/* the MIL rays' face search (inksplat.c inkfly_rays mode 1, 2026-11-20: 18 % of FR1's sampler CPU) on the CT slab ct of rows [z0, z1) of a
   box of cn voxels at co: per ray (pn: P[3], N[3] in level-0 units, n rays) drop[i] = 1 when it has a face. Byte for byte the CPU loop.
   Returns 0 ok, 1 not now (budget / ns: run the CPU loop), 2 a sample left the slab (the caller's margin bug), < 0 a CUDA failure */
typedef struct { double off, V, sign; int lo, ns, iso, run, gap; int64_t co[3], cn[3], z0, z1; int wlo, whi; } ifg_ray;
__attribute__((weak)) int inkfly_gpu_rayface(const ifg_ray *p, const double *pn, size_t n, const uint8_t *ct, uint8_t *drop);
__attribute__((weak)) unsigned long long inkfly_gpu_rays(void);   /* rays faced on the GPU so far (tests) */
__attribute__((weak)) void inkfly_gpu_device(int dev);
__attribute__((weak)) size_t inkfly_gpu_budget(void);              /* the device bytes the ink chunks may hold (UFSM_INKFLY_GPU_MB, 1024 MiB) */
__attribute__((weak)) void inkfly_gpu_set_reserve(size_t bytes);
__attribute__((weak)) size_t inkfly_gpu_budget_from_free(size_t keep);   /* budget = held + free - keep (after the plan; 0: unchanged) */    /* the free memory left beside them (default 1536 MiB; UFSM_INKFLY_RESERVE_MB wins) */
__attribute__((weak)) unsigned long long inkfly_gpu_over(void);   /* chunks sent to the CPU by the GPU budget / reserve (nn_inkfly.cu) */
#ifdef __cplusplus
}
#endif
