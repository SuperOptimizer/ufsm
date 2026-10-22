/* Transformer blocks for the U-Net bottleneck (unet_cfg.attn_blocks, src/nn_attn.cu): pre-LN multi-head self-attention with
   QK-norm (RMS over each head's 48 dims, a learned gain per channel), optional axial 3D RoPE, SwiGLU MLP, fp32 residual stream.
   Tokens are token-major fp32 rows [seq][S][C] (S = nreg register tokens + the D x H x W grid of one sample, z y x order); the
   linear layers are fp32 cuBLASLt GEMMs (env UFSM_ATTN_TF32=1: TF32), attention is a flash-style fp32 kernel (head dim 48) that
   keeps only the per-row log-sum-exp; its backward recomputes the probabilities tile by tile (no S x S matrix). A block's
   backward re-runs its forward from the stored block input (per-block recomputation): the stage keeps one [T][C] row set per
   block. Parameter gradients ACCUMULATE into the flat gradient array. All calls run on the default stream. */
#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AT_HD 48   /* head dimension */

typedef struct { int C, H, F, rope, nreg; float rope_base; } at_dims;   /* channels (H x 48), heads, MLP hidden, RoPE on, register tokens */
/* one block's parameters: offsets into the flat parameter / gradient arrays (weights [out][in] row major) */
typedef struct { size_t ln1g, ln1b, wqkv, bqkv, qg, kg, wo, bo, ln2g, ln2b, w1, b1, w2, b2; } at_bp;
size_t at_block_nparams(const at_dims *d);
size_t at_block_layout(const at_dims *d, at_bp *bp, size_t off);   /* assigns offsets from off, returns the end */

/* scratch for one block's forward + backward over T tokens (nseq sequences): at_ws_bytes, then at_ws_carve on a buffer that large */
typedef struct {
    float *xn, *qkv, *qkn, *o, *x1, *xn2, *h, *gg, *m1, *r1, *m2, *r2, *rq, *rk, *lse, *dv;
    float *bA, *bB, *bC, *dx1;
} at_ws;
size_t at_ws_bytes(const at_dims *d, long T);
void at_ws_carve(at_ws *w, const at_dims *d, long T, void *base);

/* geometry of a stage: nseq sequences of S = nreg + D H W tokens */
typedef struct { int nseq, S, D, Hh, W; } at_geom;
/* y = block(x) (x, y [T][C], T = nseq S; may not alias) */
int at_block_fwd(const at_dims *d, const float *P, const at_bp *bp, const float *x, float *y, const at_geom *g, at_ws *w);
/* g: d/d y in, d/d x out (in place); xin the block's input; parameter gradients += into G */
int at_block_bwd(const at_dims *d, const float *P, float *G, const at_bp *bp, const float *xin, float *gy, const at_geom *g, at_ws *w);

/* conv positional encoding (PEG): y = x + b + depthwise 3^3 conv(x) on the grid tokens (zero padding), register rows copied.
   w [C][27], b [C]; the backward writes gx (may alias gy only if not: separate buffers), accumulates gw / gb */
void at_peg_fwd(const float *x, float *y, const float *w, const float *b, const at_geom *g, int nreg, int C);
void at_peg_bwd(const float *x, const float *gy, float *gx, const float *w, float *gw, float *gb, const at_geom *g, int nreg, int C);
/* layouts: [n][C][Sg] channel-major (Sg = D H W) <-> token rows [n][nreg + Sg][C]; reg [nreg][C] fills the register rows */
void at_cs_to_tok(const float *x, float *t, const float *reg, int n, int C, long Sg, int nreg);
void at_tok_to_cs(const float *t, float *x, int n, int C, long Sg, int nreg);
void at_reg_grad(const float *gt, float *greg, int n, int C, long Sg, int nreg);   /* greg[r][c] += sum over sequences */
void at_h16_to_f32(const void *x, long long n, int f16, float *y);   /* 16-bit storage (fp16 / bf16) -> fp32 */

/* single ops (tests) */
int at_gemm(int ta, int tb, int M, int N, int K, const float *A, const float *B, float *Cm, float beta);   /* C[M][N] = op(A) op(B) + beta C */
void at_ln_fwd(const float *x, const float *g, const float *b, float *y, float *mean, float *rstd, long T, int C);
void at_ln_bwd(const float *x, const float *g, const float *mean, const float *rstd, const float *dy, float *dx, float *dg, float *db, long T, int C, int acc);
void at_qknorm_fwd(const float *qkv, const float *qg, const float *kg, float *qkn, float *rq, float *rk, long T, int C);
void at_qknorm_bwd(const float *qkv, const float *qg, const float *kg, const float *rq, const float *rk, const float *dqkn, float *dqkv, float *dqg, float *dkg, long T, int C);
void at_rope(float *qkn, const at_geom *g, int nreg, int C, float base, int inverse);   /* q and k parts in place */
void at_attn_fwd(const float *qkn, float *o, float *lse, const at_geom *g, int H, int C);
void at_attn_bwd(const float *qkn, const float *o, const float *lse, const float *dO, float *dv, float *dqkn, const at_geom *g, int H, int C);
void at_glu_fwd(const float *h, float *gg, long T, int F);
void at_glu_bwd(const float *h, const float *dgg, float *dh, long T, int F);

#ifdef __cplusplus
}
#endif
