/* merge-flag on the GPU (src/nn_mf.cu, 2026-11-18): the 12 integer box sums of src/mf_core.h per cell in (z, y) tiles with a 4-cell
   halo, then mf_final; the CPU reference is mf_block (src/mergeflag.c), equal up to the last bit of mf_final's double libm calls. */
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct mf_gpu mf_gpu;
/* a context on device dev for fields of H x W rows; mem_gb: the device memory the tiles may use. nullptr: no GPU / no memory */
mf_gpu *mf_gpu_open(int dev, long long H, long long W, double mem_gb);
/* the mf bytes of rows [z0, z0 + nz) of a D x H x W box into out (nz x H x W). pl: pc ps sp w, rows [rz0, rz0 + rnz) of the box,
   which must hold [z0 - 4, z0 + nz + 4) clamped to [0, D). coef: b, c_amp, c_r, c_sp. *nmat / *nflag: the material cells and those
   with mf >= thr. 0: ok */
int mf_gpu_rows(mf_gpu *g, const uint8_t *const pl[4], long long D, long long rz0, long long rnz, long long z0, long long nz, const double coef[4], int thr,
                uint8_t *out, long long *nmat, long long *nflag);
void mf_gpu_close(mf_gpu *g);
void *mf_gpu_host_alloc(size_t n);   /* page-locked host memory (nullptr: none) */
void mf_gpu_host_free(void *p);
#ifdef __cplusplus
}
#endif
