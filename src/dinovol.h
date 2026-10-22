/* dinovol: the scrollprize dinovol 3D ViT (HF scrollprize/dinovol_8um_ps192_102126, ScrollPrize/villa dinovol-refactor
   dinovol/dinovol_2) in C/CUDA on the xnet layer set (src/xnet.h), for ufsm dinovol-embed / dinovol-eval (src/dinovol.c).

   Model (teacher branch of teacher_step_640000.pt, torch zip read by src/wload.c): a CNN patch embed (conv 1->32 + channel
   LayerNorm + ReLU stem; residual BasicBlockD stages 32 / 64 / 128 / 256 at strides 1, 2, 2, 2, depths 1, 2, 2, 1; 1x1 projection
   to 480) giving one token per 8^3 patch; cls + 4 register tokens; 10 pre-norm EVA blocks (480 wide, 10 heads of 48, fused qkv
   without bias, per-block mixed 3D RoPE on the patch tokens, layer scale, SwiGLU MLP 6912 -> 3456); final LayerNorm =
   x_norm_patchtokens. PointDenseHead: FPN (1x1 tokens -> 64, trilinear x2, + 1x1 lateral of the 128-ch skip, + grouped 3^3 /
   GELU / 1x1 refine), per point trilinear samples of tokens, the three CNN skips and the FPN grid (each LayerNorm-ed),
   MLP 768 -> 384 -> 384 (GELU) + token shortcut, LayerNorm, unit-norm readouts dense (64) and ink (32).
   fp16 storage with fp32 accumulation (the reference ran bf16 autocast). */
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* kernels (src/nn_dinovol.cu), all on xnet_stream(); fp16 tensors as void * */
typedef struct { const void *p; int n, c; const float *w, *b; } dv_gather_grid;   /* grid [n^3][c] covering the tile; LN params or nullptr */
void dv_layernorm(const void *x, const float *w, const float *b, float eps, const void *res, int relu, void *y, long long M, int C);
void dv_layernorm32(const float *x, const float *w, const float *b, float eps, void *y, long long M, int C);   /* fp32 rows -> fp16 */
void dv_add_half(float *x, const void *u, long long n);
void dv_half_to_float(float *x, const void *u, long long n);
void dv_avgpool2(const void *x, void *y, int D, int H, int W, int C);
void dv_rope(void *qkv, const float *mix_dev, int N, int P, int g, int H);
void dv_attention48(const void *qkv, void *out, int N, int H);
void dv_glu(const void *a, void *g, long long N, int F);
void dv_gconv8_gelu(const void *x, const float *w_dev, const float *b_dev, void *y, int n, int C);
void dv_gather(const dv_gather_grid *grids, int ng, int T, float eps, const float *coords_dev, int P, void *tok, void *feat, int F);
void dv_head_out(const void *proj, const float *wgt, const int *rows, int P, float *acc, int nd, int ni);
int dv_points(const uint8_t *mask_dev, int b0, int b1, int ny, int nx, int s, float oz, float oy, float ox, int tile, int margin, int ramp,
              float *coords, float *wgt, int *rows, int *cnt_dev);
void dv_tok_acc(float *acc, int b0, int b1, int ny, int nx, int C, const void *tok, int g, int tz0, int ty0, int tx0, int ps, int tile, int margin, int ramp,
                const uint8_t *mask);
void dv_blockmask(const uint8_t *ct, int D, int H, int W, int z0, int y0, int x0, int s, int nz, int ny, int nx, int ylo, int yhi, int xlo, int xhi, uint8_t *mask);
void dv_region(float *dst, int dny, int dnx, int dy0, int dx0, const float *src, int sny, int snx, int sy0, int sx0, int nz, int ny, int nx, int C, int add);
void dv_region2(void *dst, int dh, int dny, int dnx, int dy0, int dx0, const void *src, int sh, int sny, int snx, int sy0, int sx0, int nz, int ny, int nx, int C, int add);   /* dh / sh: fp16 */
/* cuBLASLt fp16 GEMM, fp32 accumulation (UFSM_DV_CUBLAS: 1 QKV + fc2 (default), 2 every linear layer, 0 the xnet path): D[m][n] = sum_k A[m][k] W[n][k] (+ bias[n]) (+ D if beta; D fp32 or fp16) */
int dv_gemm(const void *A, int M, int K, const void *W, int ldw, int N, const float *bias, void *D, int d_fp32, int beta);
int dv_gemm_on(void);
void dv_glu_il(const void *a, void *g, long long n);   /* g[i] = silu(a[2i]) a[2i + 1] */
void dv_gelu(void *x, long long n);
void dv_gelu_add_ln(const void *x, const void *res, const float *w, const float *b, float eps, void *y, long long M, int C);   /* y = LN(gelu(x) + res) */
void dv_gemm_set(int on);
void dv_finish(const float *acc, int any, int anx, int CA, int y0, int x0, int nz, int ny, int nx, int c0, int C, int wch, int unit,
               const uint8_t *mask, int dtype, void *out, void *scale, int unit8);

enum { DVP_CNN, DVP_TOK, DVP_LN, DVP_QKV, DVP_ROPE, DVP_ATTN, DVP_PROJ, DVP_FC1, DVP_FC2, DVP_FINAL, DVP_FPN, DVP_HGATHER, DVP_HGEMM, DVP_HOUT, DVP_OTHER, DV_NSTAGE };
int dv_prof_on(void);                       /* UFSM_DV_PROF=1: per-stage device times (synchronising) */
void dv_prof(int stage);                    /* the time since the previous mark goes to stage */
void dv_prof_report(double per);            /* print (divided by per) and reset */

int dv_write_dense_box(const char *group, const int64_t vol_shape[3], const int64_t box[6], int C, const int8_t *data);   /* src/dinovol.c */
int cmd_dinovol_eval(int argc, char **argv);

/* the network (src/dinovol_net.c) */
typedef struct dv_net dv_net;
dv_net *dv_load(const char *ckpt);          /* nullptr on failure (message printed); needs xnet_init first */
void dv_free(dv_net *n);
int dv_embed_dim(const dv_net *n);          /* 480 */
int dv_dense_dims(const dv_net *n, int *nd, int *ni);   /* 64, 32 */
int dv_patch(const dv_net *n);              /* 8 */
int dv_train_tile(const dv_net *n);         /* the training global crop (192) */
/* one tile: u8 [T][T][T] on the device, normalised y = (min(x, hi) - center) / scale. Keeps the patch tokens ((T/8)^3 x 480, fp16,
   z y x order), the CNN skips and the FPN grid for dv_net_points. 0 on success. */
int dv_forward(dv_net *n, const uint8_t *u8_dev, int T, float center, float scale, float hi);
const void *dv_tokens(const dv_net *n);     /* device fp16 [(T/8)^3][480] */
/* the dense head at P points (tile voxel-index coordinates [P][3] on the device), in chunks: unit dense / ink vectors accumulated
   w[p] x into acc[rows[p]][nd + ni] (wgt nullptr: written to acc[p]) */
int dv_net_points(dv_net *n, const float *coords_dev, const float *wgt_dev, const int *rows_dev, int P, float *acc_dev);
/* debug / parity: device pointers of skip k (0..2) and the FPN grid, with their edge and channels */
const void *dv_skip(const dv_net *n, int k, int *edge, int *ch);
const void *dv_fpn(const dv_net *n, int *edge, int *ch);

#ifdef __cplusplus
}
#endif
