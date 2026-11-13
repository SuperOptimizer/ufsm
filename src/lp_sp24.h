/* 2:4 structured-sparse MX-fp4 convolutions (src/lp_sp24*.cu; DESIGN.md 2026-11-13 "2:4 sparse fp4 convs") */
#pragma once
#include <stddef.h>
#include "nn_lp.h"
#ifdef __cplusplus
extern "C" {
#endif
/* raw-instruction probe: one warp, one MMA of variant v (0 dense mxf4 k64, 1 sparse mxf4 k128 2X, 2 sparse mxf4nvf4 k128 4X,
   3 dense e4m3 k32, 4 sparse e4m3 k64); in[lane][11] = a0..a3, b0..b3, metadata, scale A, scale B; out[lane][4] = c0..c3. 0 = ok */
int lp_sp24_mma_probe(int v, const unsigned *in, float *out);
/* the MMA-only throughput of variant v in dense-equivalent TFLOP/s (blocks of 8 warps, it x 8 MMAs per warp); < 0: error */
double lp_sp24_mma_bench(int v, const unsigned *in, int blocks, int it);
/* the sparse fp4 conv (src/lp_sp24conv.cu): MX-fp4 in / out, > 64 input channels, weights carrying the pair-wise 4:8 pattern along the
   input channels. 1 = done, 0 = not eligible (the caller runs the dense kernel). Same arguments as lp_conv_fwd_f4 (nn_lp.h). */
int lp_sp24_conv_fwd(const void *x, int xbf, shape5 xs, const float *w, const float *b, int cout, void *y, int ybf, gnp_t gp, double *osum, int Go, split_t sp);
void lp_sp24_set_range(const float *lo, size_t n, int transposable);   /* the forward weights [lo, lo + n) carry the pattern (nullptr: none) */
void lp_sp24_note_flip(const float *w, const float *flipped);           /* a backward-data's flipped copy of w (sparse when w is and the mask transposes) */
void lp_sp24_force(int on);        /* test hook: 1 every eligible call sparse, 0 none, -1 by the registered range (default) */
unsigned lp_sp24_err(void);        /* groups the prep truncated (more than 2 non-zero pairs) since the last clear */
void lp_sp24_err_clear(void);
#ifdef __cplusplus
}
#endif
