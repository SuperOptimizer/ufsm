/* count-field on the GPU (src/nn_cf.cu): the per-voxel pooling of pass 1, the coarse-to-fine CG solves and the --model-only R 0
   model pass (the phase-gauge sums). The arithmetic mirrors src/countfield.c element by element (fp64 where it computes in double,
   fp32 storage, no FMA contraction, IEEE division): per-solve-cell sums run in the CPU's voxel order, the CG's dot products over a
   fixed grid (deterministic; they differ from the CPU threads' order in the last bits only). Weak in countfield.c: a binary without
   nn_cf.o runs the CPU code. Every call returns 0 on success, nonzero on any CUDA failure (the caller falls back to the CPU). */
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#ifndef CFGPU_WEAK
#define CFGPU_WEAK
#endif

typedef struct {
    int H, W, P, hh, w, na; double h; const float *cy, *cx;   /* box geometry, the box's axis table (box-local native z) */
    double emin, sheet0, kph, c0;                              /* cf_pool's weights */
} cf_gpu_poolcfg;
/* pass 1 for one loaded chunk of field rows [zoff, zoff + nzl) (box rows; zoff and the chunk a multiple of P except at the box end):
   pl = the chunk's planes pc ps t0..t5 sp w; S / G = the level-0 solve grid (host, whole box): the chunk's solve rows are written */
int cf_gpu_pool(const cf_gpu_poolcfg *c, const uint8_t *const pl[10], int zoff, int nzl, float *S, float *const G[3]) CFGPU_WEAK;

/* the CG of one level (cf_cg_solve): A U = b with the edge weights wt (3 axes), Jacobi preconditioner diag, screening lam; U in / out.
   Returns the iteration count (as cf_cg_solve) or < 0 on failure; *relres the final relative residual. */
int cf_gpu_cg(int d, int hh, int w, double lam, const float *const wt[3], const float *b, const float *diag, float *U, int iters, double tol, double *relres) CFGPU_WEAK;

/* the --model-only R 0 model pass: per chunk, the phasor sums gr / gi (sum c e^{i (phi - 2 pi W0)} per level-0 solve cell, W0 = W
   without the global gauge, as a float like the CPU's plane buffer) and acc += the global-gauge sum (every 7th voxel of the chunk,
   W0 in double) */
typedef struct { int D, H, W, P, d, hh, w, na, dir; double h; const float *cy, *cx, *U; } cf_gpu_gscfg;
int cf_gpu_gsum_begin(const cf_gpu_gscfg *c) CFGPU_WEAK;
int cf_gpu_gsum_chunk(const uint8_t *pc, const uint8_t *ps, const uint8_t *w, int z0, int z1, float *gr, float *gi, double acc[2]) CFGPU_WEAK;
void cf_gpu_gsum_end(void) CFGPU_WEAK;
int cf_gpu_ok(void) CFGPU_WEAK;   /* 1: a CUDA device is visible */

#ifdef __cplusplus
}
#endif
