/* xtsf: the TimeSformer pieces of ufsm ink-ext beyond xnet (xnet.h) and the dinovol row kernels (dinovol.h: dv_layernorm32,
   dv_add_half): lucidrains TimeSformer-pytorch 0.4.1 divided space-time attention as the scrollprize ink TimeSformers use it
   (dim 512, 64 px frames, 16 px patches, 26 frames, rotary embeddings, a cls token, GEGLU feed-forward). Every kernel runs on
   xnet_stream(); fp16 storage, fp32 arithmetic. Tokens are rows [b][1 + f * NP + p][C], token 0 the cls token. */
#pragma once
#include <stdint.h>
#include "xnet.h"

#ifdef __cplusplus
extern "C" {
#endif

/* uint8 frames dev [B][F][S][S] -> fp16 patch rows xt (n B, d = h = 1, w F * (S / P)^2, c P * P): 'b f c (h p1) (w p2) -> b (f h w) (p1 p2 c)'
   with v = min(u, clip) / div */
xt xtsf_patches(const uint8_t *dev, int B, int F, int S, int P, float clip, float div);
/* fp32 tokens [B][1 + n][C]: row 0 = cls (device [C]), rows 1.. = emb (fp16 rows of n) */
void xtsf_tokens(xt emb, const float *cls_dev, float *x_dev);
/* divided attention of qkv (fp16 [B][N][3 H Dh], N = 1 + F NP, the q | k | v thirds each (h d)) -> out fp16 [B][N][H Dh]:
   q scaled by Dh^-0.5; the cls query attends to every token (no rotary); a patch token (f, p) attends to the cls token and to
   (f', p) for every f' (time = 1) or (f, p') for every p' (time = 0), q and those keys rotated (rot_sin / rot_cos [F or NP][Dh],
   the rotate_every_two convention). Dh = 64. */
void xtsf_attn(xt qkv, xt out, int heads, int F, int NP, int time, const float *rot_sin, const float *rot_cos);
/* GEGLU: h fp16 [M][2 Fh] -> [M][Fh] = h[:, :Fh] * gelu(h[:, Fh:]) (exact erf GELU) */
xt xtsf_geglu(xt h);

#ifdef __cplusplus
}
#endif
