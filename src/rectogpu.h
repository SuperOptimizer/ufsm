/* recto-sp on the GPU (src/nn_rectosp.cu): the per-cell part of src/rectosp.c's rs_rows without --cls (the principal axis of n n^T,
   the trilinear source profile along it, rr_peaks, the median hit interval around the centre -> the sp code), operation by operation
   in fp64 as the CPU (UFSM_EXACT_CU: no FMA contraction, IEEE division / sqrt in every build), so the codes are the CPU's.
   The source lives on the device as a z ring of RZ slices of the run's fixed y / x source box (slot = global source z mod RZ):
   each block uploads only the slices it adds, so a source chunk row is decoded and sent once. Weak in rectosp.c: a binary without
   nn_rectosp.o runs the CPU path. Every call returns 0 on success, nonzero on any CUDA failure (the caller falls back to the CPU). */
#pragma once
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
#ifndef RSG_WEAK
#define RSG_WEAK
#endif

enum { RSG_RMAX = 64 };   /* gaussian taps: 2 R + 1 <= 2 RSG_RMAX + 1 (sigma <= ~21 samples) */
typedef struct {
    int64_t so[3], sn[3];      /* the block's source box (global source voxels): the CPU's NaN bounds; its y / x part is the ring's */
    double org[3], h, F, st, um, inner, prom, sep;   /* org: native global of the cells' row 0; inner: --half2 primary half (samples), 1e30 none */
    int n, c, Wn, R, minint, half1;   /* profile samples, centre; rr_peaks W; taps radius; --min-int; half1 > 0 (the --half2 rule on) */
    double gk[2 * RSG_RMAX + 1];  /* rr_peaks' taps (rr_taps) */
} rsg_par;

int rsg_open(int dev, int64_t RZ, int64_t ny, int64_t nx) RSG_WEAK;   /* the ring (RZ slices of ny x nx) and the scratch */
/* nz slices of global source z [z0, z0 + nz) (host, ny x nx each, contiguous) into their ring slots; from any thread, on its own
   stream beside rsg_run's kernel (the caller keeps those slots out of the running block's source box) */
int rsg_put(int64_t z0, int64_t nz, const uint8_t *src) RSG_WEAK;
/* M cells: zyx[3 M] (rows relative to par->org, as rs_rows' zz yy xx), t6[6 M] (t0..t5 bytes per cell) -> q[M] (0: no estimate);
   *nax += the cells whose axis was found (rs_rows' ncell), *nest += the estimates */
/* the device address of global source slice z's ring slot (consecutive slices are contiguous up to the ring's end); slices [z0, z0 + nz)
   zeroed (the device chunk gathers fill only the chunks that hold data) */
uint8_t *rsg_slot(int64_t z) RSG_WEAK;
int rsg_zero(int64_t z0, int64_t nz) RSG_WEAK;
int rsg_run(const rsg_par *P, int M, const int32_t *zyx, const uint8_t *t6, uint8_t *q, long long *nax, long long *nest) RSG_WEAK;
void rsg_close(void) RSG_WEAK;
size_t rsg_free_mem(int dev) RSG_WEAK;   /* free device bytes (0: no device) */

#ifdef __cplusplus
}
#endif
