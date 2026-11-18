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
    double emin, sheet0, kph, c0, rs, ra, rb, wa, wb, wd, wp;  /* cf_pool's weights (wa..wp: --wide) */
    double stage_max;                                          /* bytes per upload buffer (device + its pinned host staging, two each; 0: 2 GB): any size, the same sums */
} cf_gpu_poolcfg;
/* pass 1 for one loaded chunk of field rows [zoff, zoff + nzl) (box rows; zoff and the chunk a multiple of P except at the box end):
   pl = the chunk's planes pc ps t0..t5 sp w; S / G = the level-0 solve grid (host, whole box): the chunk's solve rows are written */
int cf_gpu_pool(const cf_gpu_poolcfg *c, const uint8_t *const pl[10], int zoff, int nzl, float *S, float *const G[3]) CFGPU_WEAK;
/* the same sums for the solve rows [Zs0, Zs1) of the chunk [zoff, zoff + nzl) from host row blocks (count-field's shard pieces: a
   chunk's rows need not be in memory at once). Each block holds field rows [r0, r1) of the 10 planes contiguously; together the blocks
   cover the rows the solve rows read (their own rows +-1, clipped to the chunk). The chunk bounds stay the phase gradient's (no
   gradient on the chunk's first / last row), so any split of a chunk's solve rows gives cf_gpu_pool's sums bit for bit */
typedef struct { const uint8_t *pl[10]; int r0, r1; } cf_gpu_rows;
int cf_gpu_pool_rows(const cf_gpu_poolcfg *c, const cf_gpu_rows *bk, int nbk, int zoff, int nzl, int Zs0, int Zs1, float *S, float *const G[3]) CFGPU_WEAK;

/* the CG of one level (cf_cg_solve): A U = b with the edge weights wt (3 axes), Jacobi preconditioner diag, screening lam; U in / out.
   Returns the iteration count (as cf_cg_solve) or < 0 on failure; *relres the final relative residual. */
int cf_gpu_cg(int d, int hh, int w, double lam, const float *const wt[3], const float *b, const float *diag, float *U, int iters, double tol, double *relres) CFGPU_WEAK;

/* the --model-only R 0 model pass: per chunk, the phasor sums gr / gi (sum c e^{i (phi - 2 pi W0)} per level-0 solve cell, W0 = W
   without the global gauge, as a float like the CPU's plane buffer) and acc += the global-gauge sum (every 7th voxel of the chunk,
   W0 in double) */
typedef struct { int D, H, W, P, d, hh, w, na, dir; double h; const float *cy, *cx, *U; } cf_gpu_gscfg;
int cf_gpu_gsum_begin(const cf_gpu_gscfg *c) CFGPU_WEAK;
int cf_gpu_gsum_chunk(const uint8_t *pc, const uint8_t *ps, const uint8_t *w, int z0, int z1, float *gr, float *gi, double acc[2]) CFGPU_WEAK;
/* rows [z0, z1) of the chunk starting at row zc (count-field's pieces): the chunk's every-7th-voxel sample and its 128-cell blocks when
   (z0 - zc) / P solve rows are whole blocks, so the pieces of a chunk give its acc and gr / gi bit for bit */
int cf_gpu_gsum_piece(const uint8_t *pc, const uint8_t *ps, const uint8_t *w, int z0, int z1, int zc, float *gr, float *gi, double acc[2]) CFGPU_WEAK;
void cf_gpu_gsum_end(void) CFGPU_WEAK;
int cf_gpu_ok(void) CFGPU_WEAK;   /* 1: a CUDA device is visible */

#ifdef __cplusplus
}
#endif
