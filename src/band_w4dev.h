/* band_wind4_tiled with the per-voxel work on the GPU (train --gpu-aug 3): the same bytes as band_wind4_tiled.
   Device (nn_w4.cu): the joins of every voxel to its x / y / z predecessor, the runs along x numbered in the CPU's scan order, the
   run-to-run joins of each run compacted into events (the first voxel that joins a new neighbour run, then the count and the
   turn histogram of the further voxels joining the same run), and pass 1 (the winding codes and component numbers from one
   offset and component per run). Host (also in nn_w4.cu, so that everything linking nn.o has it): band.c's union-find over the events instead of the voxels, the same unions
   in the same order (roots, gauges, sizes, edge and conflict counts depend on the order). */
#ifndef UFSM_BAND_W4DEV_H
#define UFSM_BAND_W4DEV_H
#include <stddef.h>
#include <stdint.h>
#include "band.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct { uint16_t len, nev; } w4d_run;   /* voxels, events */
/* cur joins run rn (the index in the same list) at its xrel-th voxel with rel turns (the low 2 bits of rel; bit 2: the z slot);
   h[t]: the run's further voxels joining rn with rel t */
typedef struct { uint32_t rn; uint16_t xrel; uint8_t rel, pad; uint16_t h[4]; } w4d_ev;
/* devices the sampler's windows may use (train: its GPUs); none: band_wind4_dev returns -2 */
void w4dev_init(const int *devices, int n);
/* band_wind4_tiled on a device: the components kept, -1 out of memory, -2 unavailable (no device, a CUDA error, a tile edge
   >= 16384): the caller then takes band_wind4_tiled_mt. Thread-safe (the sampler's workers call it). */
int band_wind4_dev(const uint8_t *phase, int D, int H, int W, int tile, int maxstep, size_t minsize, int maxcomp, uint8_t *w4, uint8_t *comp, band_wind4_stats *st);
/* the winding mod 4 as runs (train --gpu-aug 3 slots without the w4 / comp planes): the same band_wind4_tiled decision kept per run
   instead of per voxel. off: per strip (tile rows in z, y, in band_wind4_dev's order) and row (z, y, x tile: r = zy * ntx + tx) the
   index of the row's first run in ro / cb (noff entries); ro / cb: per run the turns to its root and its component (255 / 255: none).
   A plane is rebuilt from the same phase: w4runs_apply_host / nn_w4runs_apply (per voxel its run along x, then ro / cb). Buffers
   are malloc'd and kept (grown) by the caller's struct. */
typedef struct { uint32_t *off; uint8_t *ro, *cb; size_t noff, nr, cap_off, cap_r; int ncomp, tile, ms; } w4runs;   /* (tile, ms: band_wind4_runs' arguments, for the rebuild; noff 0: no winding, all unknown) */
int w4runs_available(void);   /* 1 where nn.o is linked (band.c: a weak 0) */
/* band_wind4_tiled's decision as runs: on a device when one is set up (w4dev_init), else the same on the CPU. The components kept, -1
   out of memory */
int band_wind4_runs(const uint8_t *phase, int D, int H, int W, int tile, int maxstep, size_t minsize, int maxcomp, w4runs *o, band_wind4_stats *st);
/* w4 / comp from the phase and its runs (= band_wind4_tiled's bytes), on the host */
void w4runs_apply_host(const uint8_t *phase, int D, int H, int W, int tile, int maxstep, const w4runs *o, uint8_t *w4, uint8_t *comp);
/* the same on the current device: phase_d (device, D x H x W) -> w4_d / comp_d (device; w4_d may be phase_d), on stream (a
   cudaStream_t; 0 the default stream); o's arrays are host memory (copied first). 0 ok, -1 a CUDA error */
int nn_w4runs_apply(const uint8_t *phase_d, int D, int H, int W, int tile, int maxstep, const w4runs *o, uint8_t *w4_d, uint8_t *comp_d, void *stream);
#ifdef __cplusplus
}
#endif
#endif
