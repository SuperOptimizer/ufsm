/* The phase gradient of the winding-phase codes (band.h: PHASE_PERIOD 252 codes a turn, 252+ unknown), shared by the device targets
   (nn_elem.cu unwrap_tg_k, packed_mask_k) and the CPU tests. 2026-11-14 (phaselab): the gradient over +-s voxels is the sum of the
   2 s wrapped UNIT steps, not one wrapped difference of the two end voxels. The old stencil wrapped the 2 s voxel span into +-0.5 turn,
   so it aliased as soon as the sheet spacing fell under 4 s voxels (rho_read = |1/sp - 1/2| at s = 1: 0.5, 0.2, 0 of the truth at
   sp 3, 2.5, 2, and the normal direction wrong per axis); the 19.2 um rung's packed bands (< 77 um) were all in that range. Unit steps
   alias only above 0.5 turn a voxel (sp < 2); a unit step over PG_MAXSTEP codes (0.4 turn, sp < 2.5) is called unresolved and the voxel
   unknown, as are voxels with an unknown / outside neighbour on the chain. Identical to the old stencil wherever that one was valid. */
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#ifdef __CUDACC__
#define PG_FN __host__ __device__ __forceinline__
#else
#define PG_FN static inline
#endif
enum { PG_PERIOD = 252, PG_MAXSTEP = 120 };
PG_FN int pg_wrap(int a, int b) { int d = a - b; d -= PG_PERIOD * (d > PG_PERIOD / 2); d += PG_PERIOD * (d <= -PG_PERIOD / 2); return d; }
/* g[3] = d phase / d (z, y, x) in turns a voxel; returns 1 known, 0 the voxel or a voxel on a chain unknown / outside, 2 a unit step unresolved (sp < 2.5) */
PG_FN int pg_grad(const uint8_t *ph, size_t i, int D, int H, int W, int s, float g[3]) {
    const int z = (int)(i / ((size_t)H * W)), y = (int)((i / W) % H), x = (int)(i % W);
    if (ph[i] >= PG_PERIOD || z < s || z >= D - s || y < s || y >= H - s || x < s || x >= W - s) return 0;
    const size_t st[3] = {(size_t)H * W, (size_t)W, 1};
    for (int a = 0; a < 3; a++) {
        int sum = 0, prev = ph[i - (size_t)s * st[a]];
        if (prev >= PG_PERIOD) return 0;
        for (int j = -s + 1; j <= s; j++) {
            const int v = ph[j < 0 ? i - (size_t)(-j) * st[a] : i + (size_t)j * st[a]];
            if (v >= PG_PERIOD) return 0;
            const int d = pg_wrap(v, prev);
            if (d > PG_MAXSTEP || d < -PG_MAXSTEP) return 2;
            sum += d; prev = v;
        }
        g[a] = (float)sum * (1.f / (2.f * s * PG_PERIOD));
    }
    return 1;
}
/* the local sheet rate (turns a voxel) at a labelled sheet voxel (phase 0) from its gap neighbours: per axis the largest |wrapped phase|
   of a known neighbour whose own phase is not 0 (the sheet's own voxels, and in a tilted sheet several of its neighbours, carry no rate),
   the three combined in quadrature. No gap neighbour known (a packed region: the holes of the pair rule and the sheet's thickness leave
   none) counts as dense, rho 1. Not for non-sheet voxels. */
PG_FN float pg_rate(const uint8_t *ph, size_t i, int D, int H, int W) {
    const int z = (int)(i / ((size_t)H * W)), y = (int)((i / W) % H), x = (int)(i % W);
    if (z < 1 || z >= D - 1 || y < 1 || y >= H - 1 || x < 1 || x >= W - 1) return 0.f;   /* (the box edge: no judgement, the caller keeps its weight) */
    const size_t st[3] = {(size_t)H * W, (size_t)W, 1}; float q = 0.f; int any = 0;
    for (int a = 0; a < 3; a++) {
        int mx = 0;
        for (int sg = -1; sg <= 1; sg += 2) { const int v = ph[sg < 0 ? i - st[a] : i + st[a]]; if (v >= PG_PERIOD || v == (int)ph[i]) continue; const int d = pg_wrap(v, ph[i]); const int ad = d < 0 ? -d : d; if (ad > mx) mx = ad; any = 1; }
        const float g = (float)mx * (1.f / PG_PERIOD); q += g * g;
    }
    return any ? sqrtf(q) : 1.f;
}
/* the --sheet-mask factor of a phase-0 (labelled sheet) voxel: 0 at a local spacing (1 / rho) of lo voxels or less, 1 at hi or more */
PG_FN float pg_sheet_f(float rho, float lo, float hi) {
    const float sp = rho > 0.f ? 1.f / rho : 1e9f, f = (sp - lo) / (hi - lo);
    return f < 0.f ? 0.f : f > 1.f ? 1.f : f;
}
