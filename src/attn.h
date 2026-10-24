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
#ifdef __CUDACC__
#define AT_HOSTDEV __host__ __device__
#else
#define AT_HOSTDEV
#endif

/* channels (H x 48), heads, MLP hidden, RoPE (0 none, 1 axial on the grid index, 2 scroll coordinates: at_geom.pos), register tokens */
typedef struct { int C, H, F, rope, nreg; float rope_base; } at_dims;
/* one block's parameters: offsets into the flat parameter / gradient arrays (weights [out][in] row major) */
typedef struct { size_t ln1g, ln1b, wqkv, bqkv, qg, kg, wo, bo, ln2g, ln2b, w1, b1, w2, b2; } at_bp;
size_t at_block_nparams(const at_dims *d);
size_t at_block_layout(const at_dims *d, at_bp *bp, size_t off);   /* assigns offsets from off, returns the end */

/* scratch for one block's forward + backward over T tokens (nseq sequences): at_ws_bytes, then at_ws_carve on a buffer that large */
typedef struct {
    float *xn, *qkv, *qkn, *o, *x1, *xn2, *h, *gg, *m1, *r1, *m2, *r2, *rq, *rk, *lse, *dv;
    float *bA, *bB, *bC, *dx1;
    float *qkvh, *doh, *amx;   /* tensor-core core (at_set_core 1): fp16 packed q / k / v and dO [head][T + 64][48], amax scratch */
    float *qkvr, *orr, *dor, *dqr, *dqr2;   /* radial attention (at_geom.rays; sized by at_ws_bytes_r's Tr): q / k / v and the attention output on the
                                               ray samples (forward), dO, d qkn and d qkv on them (backward) */
    int kept;   /* the forward tensors hold this block's forward of the backward's input: at_block_bwd skips its recompute (and clears it) */
} at_ws;
size_t at_ws_bytes(const at_dims *d, long T);
void at_ws_carve(at_ws *w, const at_dims *d, long T, void *base);
/* the forward part alone (the tensors the backward reads): at_ws_set_fwd points w's forward tensors into base (at_ws_fwd_bytes), the
   backward scratch stays shared. One such set per block keeps the blocks' forwards for their backwards (no recompute) */
size_t at_ws_fwd_bytes(const at_dims *d, long T);
void at_ws_set_fwd(at_ws *w, const at_dims *d, long T, void *base);
/* the same with Tr ray tokens (radial attention, at_rays: the attention's tensors sized by Tr, the ray buffers added; Tr 0 = the above) */
size_t at_ws_bytes_r(const at_dims *d, long T, long Tr);
void at_ws_carve_r(at_ws *w, const at_dims *d, long T, long Tr, void *base);
size_t at_ws_fwd_bytes_r(const at_dims *d, long T, long Tr);
void at_ws_set_fwd_r(at_ws *w, const at_dims *d, long T, long Tr, void *base);

/* geometry of a stage: nseq sequences of S = nreg + D H W tokens */
typedef struct at_rays at_rays;
/* lens: per sequence the valid tokens (keys at and past it are masked; nullptr: all S); pos: per token (r, arc, z) for at_dims.rope 2
   (nullptr: no rotation); rays: radial attention (at_block_*: the grid tokens' attention runs along these rays instead, nreg 0) */
typedef struct { int nseq, S, D, Hh, W; const int *lens; const float *pos; const at_rays *rays; } at_geom;

/* ---- radial axial attention (src/nn_attn_ray.cu): the attention of a block runs along rays of the scroll coordinates. Each sample's grid
   (D x Hh x W tokens, nreg 0) has a scroll axis along grid dim ad (0 D, 1 Hh, 2 W) with a centre (cu, cv) per plane k along ad (u, v: the
   other two dims in order); per plane Nt rays leave the centre at angles covering the plane's rectangle (all angles when the centre lies
   inside it), each sampled at unit spacing from where it enters the rectangle to where it leaves (centred on the chord; at most Nr). A
   sample's value is the bilinear interpolation (edge-clamped) of the 4 grid tokens around it in its plane. Forward: q / k / v of the grid
   -> gather onto the samples (R) -> QK-norm, RoPE, attention per ray (keys past the ray's length masked) -> scatter back normalised:
   o[v] = sum_s w(s, v) o_s / sum_s w(s, v) (0 where no sample touches v). The backward is the exact adjoint of both maps. Without an axis
   (valid 0) the rays are the grid rows along v (no interpolation). Every sample has Q = Pmax Nt rays (Pmax = max(D, Hh, W); the unused
   ones are zero-weight dummies of length 1) of Nr slots: Tr = n Q Nr ray tokens. ---- */
#define AT_RAY_F 8   /* ray record: u0, v0 (plane coordinates of sample 0), du, dv (unit step), r0 (radius of sample 0), arc (theta x rc), len, k */
struct at_rays {
    int n, Q, Nt, Nr, D, Hh, W;
    const int *ad;       /* device [n] the axis dim per sample */
    const float *ray;    /* device [n][Q][AT_RAY_F] */
    const int *lens;     /* device [n Q] max(1, len) */
    const float *pos;    /* device [n Q Nr][3] (r, arc, k) of every ray slot */
    const float *wsum;   /* device [n D Hh W] 1 / the summed stencil weights per grid token (0: uncovered) */
};
/* per sample axis record at the grid's resolution: AT_AXR(Pmax) floats: ad, valid, then per plane along ad: cu, cv (grid voxels) */
#define AT_AXR(pmax) (2 + 2 * (pmax))
/* the fixed sizes for a grid (mult: rays per plane / Pmax, e.g. 2) */
void at_rays_dims(int D, int Hh, int W, float mult, int *Q, int *Nt, int *Nr);
/* host: the ray table ray_h [n][Q][AT_RAY_F], lens_h [n Q], ad_h [n] and the scroll frame cen_h [n][Pmax][4] (cu, cv, rc, phi_m; for the
   grid coordinates of at_scroll_pos; valid 0: all zero) of n axis records */
void at_rays_plan(int n, int D, int Hh, int W, float mult, const float *axr, float *ray_h, int *lens_h, int *ad_h, float *cen_h);
/* device: R's pos and wsum (into its buffers, cast away from const) from its ray table */
void at_rays_setup(const at_rays *R);
/* grid rows x [n D Hh W][Cc] -> ray rows y [Tr][Cc] (norm: each grid token's weight x wsum: the adjoint of the normalised scatter) */
void at_ray_gather(const at_rays *R, const float *x, float *y, int Cc, int norm);
/* ray rows y -> grid rows x (zeroed first): x[v] = sum_s w(s, v) y_s (x wsum[v] with norm) */
void at_ray_scatter(const at_rays *R, const float *y, float *x, int Cc, int norm);
/* scroll RoPE coordinates of the grid tokens (full attention, at_dims.rope 2): pos [n][nreg + D Hh W][3] = (r, arc, k) around the axis
   (registers 0); without an axis (v, u, k): the grid indices */
void at_scroll_pos(int n, int nreg, int D, int Hh, int W, const int *ad, const float *cen, float *pos);
/* the host side of the bilinear stencil (src/nn_attn_ray.cu uses the same arithmetic): sample i of ray rr in a Nu x Nv plane -> up to 4
   (a, b, weight); returns the count (0: past the ray's length) */
static inline AT_HOSTDEV int at_ray_corners_h(const float *rr, int i, int Nu, int Nv, int *a, int *b, float *w) {
    if (i >= (int)rr[6]) return 0;
    float pu = rr[0] + (float)i * rr[2], pv = rr[1] + (float)i * rr[3];
    pu = pu < 0.f ? 0.f : pu > (float)(Nu - 1) ? (float)(Nu - 1) : pu; pv = pv < 0.f ? 0.f : pv > (float)(Nv - 1) ? (float)(Nv - 1) : pv;
    int a0 = (int)pu, b0 = (int)pv; if (a0 > Nu - 2) a0 = Nu - 2; if (a0 < 0) a0 = 0; if (b0 > Nv - 2) b0 = Nv - 2; if (b0 < 0) b0 = 0;
    const float fu = pu - (float)a0, fv = pv - (float)b0; const int a1 = a0 + 1 < Nu ? a0 + 1 : a0, b1 = b0 + 1 < Nv ? b0 + 1 : b0;
    a[0] = a0; b[0] = b0; w[0] = (1.f - fu) * (1.f - fv); a[1] = a1; b[1] = b0; w[1] = fu * (1.f - fv);
    a[2] = a0; b[2] = b1; w[2] = (1.f - fu) * fv; a[3] = a1; b[3] = b1; w[3] = fu * fv;
    return 4;
}
/* the grid token of (plane k along ad, in-plane a, b) */
static inline AT_HOSTDEV long at_ray_tok(int ad, int k, int a, int b, int Hh, int W) {
    const int z = ad == 0 ? k : a, y = ad == 0 ? a : ad == 1 ? k : b, x = ad == 2 ? k : b;
    return ((long)z * Hh + y) * W + x;
}
static inline AT_HOSTDEV void at_ray_plane(int ad, int D, int Hh, int W, int *Np, int *Nu, int *Nv) {
    *Np = ad == 0 ? D : ad == 1 ? Hh : W; *Nu = ad == 0 ? Hh : D; *Nv = ad == 2 ? Hh : W;
}
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
/* window memory rows r0..r0+M-1 of the n sequences of S token rows: t[s][r0+j][c] += gate[c] mem[s][j][c] + sum_k woff[k][c] feat[s][k] (k < 4);
   its gradients ggate[c] += sum_{s,j} gt mem, gwoff[k][c] += sum_{s,j} gt feat[s][k] (mem, feat constants); the rows out: mout[s][j][c] */
void at_mem_add(float *t, int n, int S, int r0, int M, int C, const float *gate, const float *woff, const float *mem, const float *feat);
void at_mem_grad(const float *gt, int n, int S, int r0, int M, int C, const float *mem, const float *feat, float *ggate, float *gwoff);
void at_mem_out(const float *t, int n, int S, int r0, int M, int C, float *mout);
/* scan fusion view rows: patches xp [n][Nv][k^3] (Nv = (Pv / k)^3, patch j = (pz, py, px) z-major, its voxels z y x) of the uint8 crops
   v [n][Pv^3] as v / 255 (v nullptr: zeros); the rows r0..r0+Nv-1: t[s][r0+j][c] += E[s Nv + j][c] + bias[c]; their gradient gathered
   GE[s Nv + j][c] = gt[s][r0+j][c] and gbias[c] += sum over the rows */
void at_view_patch(const unsigned char *v, int n, int Pv, int k, float *xp);
void at_view_add(float *t, int n, int S, int r0, int Nv, int C, const float *E, const float *bias);
void at_view_gather(const float *gt, int n, int S, int r0, int Nv, int C, float *GE, float *gbias);
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
