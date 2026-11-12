/* train --ssl: masked self-distillation of the latent net on unlabelled ("ct_only") windows (kernels src/nn_ssl.cu, DESIGN 2026-11-12
   "Self-supervision for the latent net"). The EMA teacher sees the whole window, the student the same window with blocks of bv^3
   voxels of the CT channel zeroed (so neither the latent cells nor the fine stem see them); the target is the teacher's 1^3 side head
   on a decoder level (cfg.emb_lev / emb_ch, the --distill head), centred per window and unit-normalised, as int8 at the masked cells only
   (zero elsewhere: nn_distill_loss's "no target"). The mask is a hash of (seed, block): host and device decide it the same way. */
#pragma once
#include <stddef.h>
#include <stdint.h>

#if defined(__CUDACC__)
#define SSL_HD __host__ __device__
#else
#define SSL_HD
#endif

static inline SSL_HD uint64_t ssl_mix(uint64_t x) { x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ull; x ^= x >> 27; x *= 0x94d049bb133111ebull; return x ^ (x >> 31); }
/* block (bz, by, bx) of the window is masked when its hash falls under ratio (24-bit threshold) */
static inline SSL_HD int ssl_blk_masked(uint64_t seed, int bz, int by, int bx, float ratio) {
    const uint64_t h = ssl_mix(seed ^ ssl_mix(((uint64_t)(uint32_t)bz << 42) ^ ((uint64_t)(uint32_t)by << 21) ^ (uint64_t)(uint32_t)bx));
    return (double)(h >> 40) < (double)ratio * 16777216.0;
}

#ifdef __cplusplus
extern "C" {
#endif
/* zero the masked bv^3 blocks of a 16- or 32-bit (es bytes) CT channel x [Pz][P][P] (one sample) */
void nn_ssl_mask(void *x, int es, int Pz, int P, int bv, float ratio, uint64_t seed);
/* teacher targets: y [C][cd][ch][cw] fp32 (one sample), cells of cv window voxels; mu (device, C doubles) scratch for the channel means.
   t[c][s] = round(127 v_c / |v|), v = y[.][s] - mean over the window's cells, at cells inside a masked block with |v| > 1e-6; else 0 */
void nn_ssl_targets(const float *y, int C, int cd, int ch, int cw, int cv, int bv, float ratio, uint64_t seed, double *mu, int8_t *t);
/* N distinct cell indices idx[k] = (off + k stride) mod S (stride coprime to S: the caller's ssl_stride) */
void nn_ssl_idx(int *idx, int N, size_t S, size_t off, size_t stride);
/* --ssl-heads (gated mean-teacher consistency of the output heads on the latent cells, DESIGN 2026-11-12): per head a mode, its output
   channels [ch0, ch0 + nch), its target slots [slot0, slot0 + nch) of t ([slot][S] uint8 over the whole cell grid of S cells), a confidence
   threshold and whether the teacher's papyrus gates it. From the teacher's coarse head logits of a chunk lg [nch][n] (fp16, cells [off,
   off + n) of the grid cd x ch x cw, cells of cv voxels), a cell is targeted only where the student sees it (NOT in a masked block) and,
   for pap heads, on the teacher's papyrus (max of the rv pair's sigmoids >= 0.5, rv_ch); then per mode, with p_c = sigmoid(logit_c):
     SSLH_BCE:   each channel on its own where |p_c - 0.5| >= conf;
     SSLH_NORM:  the six normal channels (N_zz, N_yy, N_xx = p0..p2, N_zy, N_zx, N_yx = p3..p5 - 1/2) together where |N|_F >= conf
                 (1 for a clean unit n n^T; directions that disagree inside the cell or an unsure teacher shrink it);
     SSLH_PHASE: the cos / sin pair together where |v| >= conf, v = (2 p0 - 1, 2 p1 - 1) (the in-turn phase: local, no integer part).
   t[slot][s] = round(254 p_c) at the targeted channels, else 255. cnt (device, one u64 per slot) += the targeted cells. */
enum { SSLH_BCE = 0, SSLH_NORM = 1, SSLH_PHASE = 2, SSLH_MAX = 8 };
typedef struct { int mode, ch0, nch, slot0, pap; float conf; } ssl_head;
typedef struct { ssl_head h[SSLH_MAX]; int nh, nslot; float scale[SSLH_MAX]; } ssl_heads;   /* (scale: the gradient's, per head) */
void nn_sslh_targets(const void *lg, const ssl_heads *hs, int rv_ch, size_t n, size_t off, int cd, int ch, int cw, int cv, int bv, float ratio,
                     uint64_t seed, uint8_t *t, unsigned long long *cnt);
/* the student's soft BCE against them: lg [nch][n] its chunk logits (fp16), g [nch][n] (fp16, zeroed by the caller): every targeted
   channel c of head h gets hs->scale[h] x (sigmoid(z) - t / 254); sums (device, 2 doubles per head) += sum BCE, targeted entries */
void nn_sslh_grad(const void *lg, const ssl_heads *hs, size_t n, size_t off, size_t S, const uint8_t *t, void *g, double *sums);
#ifdef __cplusplus
}
#endif
static inline size_t ssl_gcd(size_t a, size_t b) { while (b) { const size_t t = a % b; a = b; b = t; } return a; }
static inline size_t ssl_stride(size_t S, uint64_t r) {   /* an odd stride near S x 0.618 that is coprime to S */
    if (S < 2) return 1;
    size_t s = (size_t)(S * 0.6180339887) | 1; s += (size_t)(r % 64) * 2;
    while (ssl_gcd(s % S ? s % S : 1, S) != 1) s += 2;
    return s % S ? s % S : 1;
}
