/* lcodec: the learned input codec of the latent-grid nets, stage 1 (DESIGN "Learned input codec", 2026-11-19). A cell's 64 CT bytes are
   first rotated onto the full orthonormal KLT basis (latfit.h; x = coefficients / sd_ref, x[0] the mean direction). The mean index stays
   the KLT's (q0 = rint(x0 / mstep), latvol's air rule and DPCM), the other cl - 1 channels are learned:
     analysis (encoder side only):  y = Wl x[1..63] + W2 silu(W1 f + b1),  f = lcodec_feat (the cell's 63 axes, its centred mean, and per
                                    neighbour of the 26: the mean difference and its first 3 axes), q_c = clamp(rint(y_c), +-127);
     synthesis (linear, per cell):  x[1..63] ~ S^T q (S [cl - 1][63]): as latvol basis rows b'_c = sum_j S[c][j] b_{j+1} / mstep, so the
                                    store is latvol's format and every reader (host z3_read, device decode / synth) is unchanged;
     rate model (training):         per channel a zero-mean Laplace of scale exp(lb_c) convolved with the unit quantiser (a factorised
                                    density); training adds U(-1/2, 1/2) noise to y (straight through); reported rates are real tANS bytes.
   Loss per cell = R (bits per voxel) + lambda D (grey levels^2 per voxel, the axes' error in the KLT frame = the voxel error). */
#ifndef UFSM_LCODEC_H
#define UFSM_LCODEC_H
#include <stddef.h>
#include <stdint.h>
#include "latfit.h"
#include "latvol.h"

enum { LCD_NF = 63 + 1 + 26 * 4, LCD_NAX = 63, LCD_NCX = 27 };
typedef struct {
    int cl, hid;          /* channels (mean + cl - 1 learned, cl <= 64), hidden width */
    int hid2;             /* > 0: tier b, a nonlinear synthesis term x += V2 silu(V1 [q, ctx] + c1) (ctx: lcodec_ctx, the decoded mean indices of the
                             cell and its 26 neighbours); not latvol rows: rebuilt by lcodec_rebuild. 0: the linear synthesis alone */
    float sd_ref, mstep;  /* grey levels per x unit; the mean channel's step in x units (the store's step_raw = mstep sd_ref) */
    float mean_c;         /* the feature's mean centring (x units of the cell mean) */
    size_t np; float *p;  /* Wl [K][63], W1 [hid][NF], b1 [hid], W2 [K][hid], S [K][63], lb [K]  (K = cl - 1); tier b: V1 [hid2][K + NCX], c1 [hid2], V2 [63][hid2] */
} lcodec;
typedef struct { size_t Wl, W1, b1, W2, S, lb, V1, c1, V2, end; } lcodec_off;
lcodec_off lcodec_offs(int cl, int hid, int hid2);
size_t lcodec_np(int cl, int hid, int hid2);
/* init as the KLT at step s0 (x units): Wl = I / s0, S = I s0, W2 0, W1 ~ N(0, 1 / NF), lb = log(sigma / s0) (sigma: the axes' sd, x units) */
int lcodec_init(lcodec *lc, int cl, int hid, int hid2, float sd_ref, float s0, const float *sigma, float mean_c, uint64_t seed);   /* tier b: V2 0 (= tier a) */
void lcodec_free(lcodec *lc);
/* the features of cell (z, y, x) of a grid of n0 x n1 x n2 cells, xg [cells][64] (x units); missing neighbours repeat the cell */
void lcodec_feat(const lcodec *lc, const float *xg, int n0, int n1, int n2, int z, int y, int x, float *f);
/* the analysis: y [cl - 1] (unquantised) */
void lcodec_analysis(const lcodec *lc, const float *f, const float *x, float *y);
/* indices of a cell: q [cl] int16 (q[0] the mean, latvol's rules; all 0: air); any: the cell holds a nonzero byte */
void lcodec_encode_cell(const lcodec *lc, const float *f, const float *x, int any, int16_t *q);
/* tier b's decoder context of a cell from its features and mean coefficient: ctx [NCX] (x units; what a decoder knows from the mean indices) */
void lcodec_ctx(const lcodec *lc, const float *f, float x0, float *ctx);
/* the decoder: a cell's 64 bytes from its indices q [cl] and ctx (tier a: the float form of latvol's rows; 1..255, air 0) */
void lcodec_rebuild(const lcodec *lc, const lat_basis *lb, const int16_t *q, const float *ctx, uint8_t *v);
/* loss and gradient (added into g, scaled by 1 / n) of n cells: f [n][NF], x [n][64], u [n][cl - 1] the noise; returns the mean loss,
   *R bits / voxel and *D grey levels^2 / voxel (axes only) */
double lcodec_loss_grad(const lcodec *lc, int n, const float *f, const float *x, const float *u, double lambda, float *g, double *R, double *D);   /* (tier b: ctx from f, x) */
/* the store codec: latvol rows (row 0 the KLT mean row, rows c the synthesis), step mstep, the given histograms (nullptr: flat) */
int lcodec_lv(const lcodec *lc, const lat_basis *lb, const uint64_t (*hist)[LV_NSYM], lv_codec *out);
/* file: "LCD1", cl, hid, sd_ref, mstep, mean_c, np, params (little endian); "LCD2" (tier b): hid2 after hid */
int lcodec_write(const lcodec *lc, const char *path);
int lcodec_read(lcodec *lc, const char *path);
/* a hash of the codec (the checkpoint's lat_codec): FNV-1a 64 over the file bytes' fields */
uint64_t lcodec_hash(const lcodec *lc);
#endif
