/* Transformer blocks for the U-Net bottleneck (unet_cfg.attn_blocks, src/nn_attn.cu): pre-LN multi-head self-attention with
   QK-norm (RMS over each head's 48 dims, a learned gain per channel), optional axial 3D RoPE, SwiGLU MLP, fp32 residual stream.
   Tokens are token-major fp32 rows [seq][S][C] (S = nreg register tokens + the D x H x W grid of one sample, z y x order); the
   linear layers are fp32 cuBLASLt GEMMs (env UFSM_ATTN_TF32=1: TF32) or MX-fp4 / fp8 (at_set_mx), with bias / residual in the GEMM
   epilogue. Attention is flash-style (head dim 48), keeping only the per-row log-sum-exp; its backward recomputes the probabilities
   tile by tile (no S x S matrix). The core (at_set_core): 0 fp32 CUDA cores, 1 fp16 tensor cores, 2 MX-fp8 tensor cores. A block's
   backward re-runs its forward from the stored block input (per-block recomputation) unless the caller kept that forward (at_ws.kept,
   at_ws_set_fwd: one forward set per block). Parameter gradients ACCUMULATE into the flat gradient array. Default stream throughout. */
#pragma once
#include <stddef.h>
#include <stdio.h>

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
    float *qkvh, *doh, *amx;   /* tensor-core core (at_set_core 1): fp16 packed q / k / v and dO [head][T + 64][48], amax scratch */
    int kept;   /* the forward tensors hold this block's forward of the backward's input: at_block_bwd skips its recompute (and clears it) */
} at_ws;
size_t at_ws_bytes(const at_dims *d, long T);
void at_ws_carve(at_ws *w, const at_dims *d, long T, void *base);
/* the forward part alone (the tensors the backward reads): at_ws_set_fwd points w's forward tensors into base (at_ws_fwd_bytes), the
   backward scratch stays shared. One such set per block keeps the blocks' forwards for their backwards (no recompute) */
size_t at_ws_fwd_bytes(const at_dims *d, long T);
void at_ws_set_fwd(at_ws *w, const at_dims *d, long T, void *base);

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

/* MX GEMM (src/lp_mxgemm.cu): at_gemm's operands and result, both operands quantised along K to MX-fp4 (bits 4) or MX-fp8 (bits 8)
   with one ue8m0 scale per 32 K (round to nearest), block-scaled tensor-core products, fp32 accumulation; -1 unsupported */
int lp_mx_gemm(int bits, int ta, int tb, int M, int N, int K, const float *A, const float *B, float *Cm, float beta);
/* the same with an epilogue: C = op(A) op(B) + bias[col] (bias nullptr: none) + res[row][col] (res nullptr: none; may alias C only
   with beta 0) + beta C */
int lp_mx_gemm_ex(int bits, int ta, int tb, int M, int N, int K, const float *A, const float *B, float *Cm, float beta, const float *bias, const float *res);
/* at_gemm with that epilogue (the fp32 path: copy res, GEMM, bias kernel) */
int at_gemm_ex(int ta, int tb, int M, int N, int K, const float *A, const float *B, float *Cm, float beta, const float *bias, const float *res);
/* at_gemm's precision for the linear layers (attention stage, DINO / iBOT / region heads): 0 fp32 (default), 4 / 8 MX (env UFSM_LINEAR_MX);
   at_gemm_f32 always fp32 (Muon's Newton-Schulz, tests) */
void at_set_mx(int bits);
int at_get_mx(void);
int at_gemm_f32(int ta, int tb, int M, int N, int K, const float *A, const float *B, float *Cm, float beta);

/* the attention core (softmax(Q K^T) V and its backward): 0 fp32 CUDA cores, 1 fp16 tensor cores with fp32 accumulation (default)
   (src/nn_attn_tc.cu); env UFSM_ATTN_CORE */
void at_set_core(int c);   /* (2: MX-fp8 tensor cores, src/lp_attn8.cu) */
int at_get_core(void);
/* the fp16 core's pieces (src/nn_attn_tc.cu): pack = QK-norm + gains + RoPE of qkv into fp16 head-major q / k / v (qkvh), rq / rk and
   amx[1] = max |v| as at_qknorm_fwd; fwd: o [T][C] fp32 and lse (log2) as at_attn_fwd; bwd: dqkn [T][3C] fp32 as at_attn_bwd (doh, Dv,
   amx scratch) */
void at_fa_pack(const float *qkv, const float *qg, const float *kg, void *qkvh, float *rq, float *rk, float *amx, const at_geom *g, int nreg, int C, int rope, float base);
void at_fa_fwd(const void *qkvh, float *o, float *lse, const at_geom *g, int H, int C);
void at_fa_bwd(const void *qkvh, void *doh, const float *o, const float *lse, const float *dO, float *Dv, float *amx, float *dqkn, const at_geom *g, int H, int C);
void at_fa_bprep(const float *o, const float *dO, float *amx, void *doh, float *Dv, const at_geom *g, int C);
/* core 2 (src/lp_attn8.cu): the same products in MX-fp8 (e4m3, one ue8m0 scale per 32 K; P and dS quantised in registers), from the
   fp16 packs; own scratch (grown on demand); 0 or -1 */
int at_fa8_fwd(const void *qkvh, float *o, float *lse, const at_geom *g, int H, int C);
int at_fa8_bwd(const void *qkvh, void *doh, const float *o, const float *lse, const float *dO, float *Dv, float *amx, float *dqkn, const at_geom *g, int H, int C);

/* env UFSM_ATTN_PROF=1: per-phase cudaEvent timing of the blocks (synchronising), the table at exit or by at_prof_report */
void at_prof_report(FILE *f);
void at_prof_reset(void);

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
