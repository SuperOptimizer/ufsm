/* latvol arrays read as cells (src/latvol_read.c): the chunk streams under a box, packed for the device decoder (nn_latvol.h) */
#pragma once
#include "nn_latvol.h"
#include "zarr3.h"
/* The chunk streams of the latvol array z under the voxel box [o, o + n) (o, n multiples of 4) packed into buf (cap bytes, each stream at
   a multiple of 4): of every chunk only the substreams whose cells meet the box (the other lengths set to 0: their cells lie outside), so
   a 16-voxel halo costs one substream layer, not whole neighbour chunks. refs[i]: the stream and its chunk's first cell relative to the
   box's first cell. No decode on the host: index lookups, chunk fetches (the chunk cache of an HTTP root) and copies, on nthreads threads.
   Returns the number of streams (0: every cell of the box is air), -1 on an I/O error, -2 when cap or maxref is too small. */
int lv_read_box(z3 *z, const int64_t o[3], const int64_t n[3], uint8_t *buf, size_t cap, lv_chunk_ref *refs, int maxref, size_t *bytes, int nthreads);
/* the host reference of the device path: the box's cells int8 [n0/4][n1/4][n2/4][stride] decoded from the same packed streams */
int lv_cells_host(const void *codec, const uint8_t *buf, const lv_chunk_ref *refs, int nref, int d, int h, int w, int stride, int8_t *cells);
/* The voxel box [o, o + n) of the latvol array z (any offsets) as rebuilt CT bytes decoded on the GPU: lv_read_box of the cell-aligned
   box, nn_latvol_box, one copy back, the crop. Byte-identical to the host decoder. On when configured: latvol_gpu_config(dev, slots)
   (dev < 0: off), or $UFSM_LATVOL_GPU=dev ($UFSM_LATVOL_SLOTS, default 2) at the first read; $UFSM_LATVOL_CPU=1 forces the host.
   Returns 0 done, 1 not taken (off, a small box, no device, streams over the budget: the caller decodes on the host), -1 error.
   z3_read of a latvol array calls it first. */
int lv_box_voxels_gpu(z3 *z, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads);
void latvol_gpu_config(int dev, int slots);
uint64_t latvol_gpu_boxes(void);   /* boxes decoded on the GPU so far */
