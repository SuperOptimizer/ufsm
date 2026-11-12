/* vcgpu: volcomp lossy chunks encoded from levels the GPU computed (nn_vcg_shard): the GPU gathers, transforms and quantises every 16^3
   block with the CPU encoder's arithmetic (vf_analyse_block: the same rounding of every operation, the AVX2 kernel's fused AC quantiser
   where the CPU uses it) and lists each block's nonzero AC levels in ascending scan order; vcg_encode_levels tokenises, builds the models
   and entropy-codes them as volcomp_encode does (vf_encode_lossy's phases A (tokenise) .. C), so the stream is volcomp_encode's byte for
   byte. At q > 0 only (q 0 is the lossless mode: the CPU encoder). On the CPU, gather + transform + quantise + sort are 68 % of a q 8 head
   chunk's 1.9 ms (pod / desk AVX2). */
#pragma once
#include <stddef.h>
#include <stdint.h>
#define VCG_ENCODE_BOUND ((size_t)14681264u)   /* volcomp.h VOLCOMP_ENCODE_BOUND */

/* the quantiser's reciprocal steps (4096, natural order: vf_steps / vf_dct_scale) and the scan table (scan position -> natural index);
   returns 1 when the CPU encoder quantises AC coefficients with a fused multiply-add (its AVX2 kernel), 0 unfused (the C kernel) */
int vcg_tables(float q, float *rstep, uint16_t *scan);
/* one chunk from its 512 blocks (block bi = bz * 64 + by * 8 + bx): dc[bi] the DC level, n[bi] nonzero AC count, lv / pos (indexed from
   off[bi]) the levels and scan positions in ascending scan order; dst / cap / *out_n as volcomp_encode. 0 ok, else a volcomp_status */
int vcg_encode_levels(float q, const int32_t *dc, const uint32_t *n, const uint32_t *off, const int32_t *lv, const uint16_t *pos,
                      uint8_t *dst, size_t cap, size_t *out_n);

/* the GPU's output buffer for an n^3 shard (nb = (n / 16)^3 blocks, chunk-major then bi): at 0 the total of nonzero AC levels (u64);
   dc / cnt / off (u32, entries from lv / pos) / cv (the constant value of a constant block, else -1) per block; then cap_e levels (i32) and
   scan positions (u16). A total above cap_e: lv / pos unwritten (encode that shard from its bytes) */
typedef struct { long nb; size_t o_dc, o_cnt, o_off, o_cv, o_lv, o_pos, bytes; } vcg_layout;
static inline void vcg_layout_of(vcg_layout *L, int n, size_t cap_e) {
    L->nb = (long)(n / 16) * (n / 16) * (n / 16);
    const size_t a = 256;
    L->o_dc = a; L->o_cnt = L->o_dc + (size_t)L->nb * 4; L->o_off = L->o_cnt + (size_t)L->nb * 4; L->o_cv = L->o_off + (size_t)L->nb * 4;
    L->o_lv = (L->o_cv + (size_t)L->nb * 2 + a - 1) / a * a; L->o_pos = L->o_lv + cap_e * 4; L->bytes = (L->o_pos + cap_e * 2 + a - 1) / a * a;
}
/* (src/nn_gpuenc.cu) the levels of the device shard dsh (n^3, n a multiple of 128): the per-block header (vcg_layout_of(n, cap_e) offsets
   below o_lv) into the device buffer hdr, the levels / positions into lv / pos (cap_e each: device memory, or pinned host memory the
   kernel writes over the bus, so only the levels present cross it); drstep / dscan: vcg_tables' on the device; fma: its return.
   Asynchronous on the default stream. 0 ok */
#ifdef __cplusplus
extern "C"
#endif
int nn_vcg_shard(const uint8_t *dsh, int n, const float *drstep, const uint16_t *dscan, int fma, uint8_t *hdr, int32_t *lv, uint16_t *pos, size_t cap_e);
