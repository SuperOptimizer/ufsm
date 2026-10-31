/* gpudec.h - volcomp lossy (mode 0) substream decoding on the GPU (src/nn_gpudec.cu), byte-identical to vf_decode_sub.
 *
 * Opt-in: UFSM_GPU_DECODE=1 (UFSM_GPU_DECODE_DEV=d picks the device, default 0). Callers parse the chunk on the CPU (vf_parse:
 * the reference's statuses for header / table / directory errors), fill a gd_chunk and call gpudec_decode from any thread: a
 * service thread coalesces the concurrent requests of all threads into one batch per GPU round trip (one thread per substream
 * runs the tANS / bypass decode into coefficient lists, then one CUDA block per 16^3 block dequantises, runs the reference's
 * inverse DCT network with no FMA contraction (-fmad=false) and rounds). The more threads submit at once, the bigger the batches:
 * gpudec_threads() is how many decode threads a consumer should run.
 *
 * gpudec_decode returns 0: every substream in mask decoded into dst (128^3 chunk layout; other substreams untouched);
 * 1: some substream did not decode on the GPU (corrupt or truncated stream): dst untouched, decode the request on the CPU
 * (vfx_decode_subs gives the reference's status and bytes); -1: the GPU path is unavailable (not enabled, device error, request
 * too large): decode on the CPU. Any device error turns the GPU path off for the rest of the process.
 *
 * Weak stubs in zarr3.c keep binaries without nn.o (CPU-only tests) linking: there gpudec_enabled() is 0. */
#ifndef UFSM_GPUDEC_H
#define UFSM_GPUDEC_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct gd_chunk {
    const uint8_t *payload;            /* vf_parsed.payload: the substream bytes, substream s at off[s] (tok_n[s] tANS bytes, then byp_n[s]) */
    const uint32_t *tok_n, *byp_n, *off;
    const float *step;                 /* 46 dequantisation steps (vf_parsed.step) */
    const uint32_t *dt[10];            /* the 10 tANS decode tables, 1024 vf_dentry (u16 ns, u8 sym, u8 nb) each */
    const uint16_t *scan;              /* VF_SCAN16 */
} gd_chunk;

int gpudec_enabled(void);
int gpudec_threads(int nthreads);      /* decode threads for a consumer that would run nthreads (more when the GPU decodes) */
int gpudec_decode(const gd_chunk *c, uint32_t mask, uint8_t *dst);
/* the same, written straight into a device 128^3 chunk (chunk layout) on the decoder's device: no host round trip. 0 / 1 / -1 as above
   (-1 also when dchunk is not memory of that device) */
int gpudec_decode_dev(const gd_chunk *c, uint32_t mask, uint8_t *dchunk);
void gpudec_stats(uint64_t *subs, uint64_t *batches, uint64_t *cpu_redo, double *gpu_s);

#ifdef __cplusplus
}
#endif
#endif
