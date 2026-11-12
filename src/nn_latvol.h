/* latvol on the GPU (src/nn_latvol.cu; the format: latvol.h). All calls run on the current device's default stream. */
#ifndef UFSM_NN_LATVOL_H
#define UFSM_NN_LATVOL_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct nn_lv nn_lv;
/* one chunk stream of a window: its bytes at comp + off (off a multiple of 4), len bytes; (cz, cy, cx) the chunk's first cell in the
   window's cell grid (negative / beyond the window: only the substreams that meet the window are decoded) */
typedef struct { uint32_t off, len; int32_t cz, cy, cx; } lv_chunk_ref;
/* the device copy of a codec: its decode tables dt [ntab][1024], the channel -> table map [C], the int16 basis rows bi [C][64] */
nn_lv *nn_latvol_create(const uint32_t *dt, int ntab, const uint8_t *map, const int16_t *bi, int C, float step_raw);
void nn_latvol_free(nn_lv *L);
/* the window's cells int8 [d][h][w][stride] (device; bytes 0-1 the mean index, 2.. the axes c < min(C, stride - 1); zero = air) from the
   nchunk chunk streams in the device buffer comp_d (refs_h: host). Returns 0, or 1 when a stream was corrupt (the cells are then
   undefined: decode on the host for the error). Waits for the kernels. */
int nn_latvol_decode(nn_lv *L, const void *comp_d, const lv_chunk_ref *refs_h, int nchunk, int d, int h, int w, int stride, void *cells_d);
void nn_latvol_stats(const nn_lv *L, double *seconds, long *calls);   /* wall time in nn_latvol_decode (zeroing, both kernels, the flag read) and its calls */
/* the cells' CT bytes [4 d][4 h][4 w] (1..255 in material cells, 0 in air cells): lv_inverse's bytes. stride >= C + 1 */
void nn_latvol_synth(nn_lv *L, const void *cells_d, int stride, int d, int h, int w, uint8_t *ct_d);
/* CT bytes [4 d][4 h][4 w] -> indices int16 [d h w][C] (lv_forward's) */
void nn_latvol_forward(nn_lv *L, const uint8_t *ct_d, int d, int h, int w, int16_t *q_d);
/* A box of latent cells as CT bytes [4 d][4 h][4 w] from any thread: the packed streams (lv_read_box) decoded and rebuilt on device
   `dev` (nn_latvol_box_config; a pool of `slots` contexts, callers wait for a free one), copied to ct_out. key: the host codec's address
   (its device tables are kept per context). 0 ok, 1 a corrupt stream, -1 no device / out of device memory (decode on the host). Weak:
   binaries without the CUDA objects take the host decoder. */
__attribute__((weak)) void nn_latvol_box_config(int dev, int slots);
__attribute__((weak)) int nn_latvol_box(const void *key, const uint32_t *dt, int ntab, const uint8_t *map, const int16_t *bi, int C, float step_raw,
                                        const uint8_t *comp, size_t ncomp, const lv_chunk_ref *refs, int nref, int d, int h, int w, uint8_t *ct_out);
#ifdef __cplusplus
}
#endif
#endif
