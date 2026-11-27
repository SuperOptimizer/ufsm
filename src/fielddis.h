/* src/fielddis.h: ufsm field-disagree (fielddis.c): where several winding models' fit-field sets disagree, as a labelling query signal
   (active learning by committee disagreement, 2026-11-27). The first input is the reference grid; the others (any level / --fit-h: try9
   at 9.6 um, w24 at 2.4 um) are sampled at its cell centres in physical um (trilinear phase vectors, nearest material). Per output cell
   (the reference cells, every --stride-th):
     pos  sheet position: the largest pairwise circular phase difference, turns (0 .. 0.5), plane value x 510
     cnt  phase-wrap count along the sheet normal (the reference's principal n n^T axis): each model's unwrapped phase change over a
          ray of +-ray_um through the cell, over the stretch where every model is valid; the largest pairwise difference, turns, x 64
     w4   winding mod 4 (task unwrap2 planes dz dy dx dq): per axis the models' relative winding over their own baseline w4_d x um; a
          pair with baselines D1 <= D2 compares mod 4 over D2, the shorter one's planes chained D2 / D1 times along the axis (fdis_w4_pair);
          max over axes, turns, x 128
     dis  combined: max(pos / 0.25, cnt / 0.5, w4 / 0.5) / 2 x 255 over the valid metrics (of fdis_cfg metrics), so dis >= 128 <=> some metric past its
          threshold (a quarter turn of position, half a turn of count or of winding mod 4)
     flag bit 0 phase valid (every model: material and confidence > conf), bit 1 ray valid, bit 2 w4 valid, bit 3 GT known, bits 4..7
          model k (< 4) wrong against the GT (position > 0.25 turn or count >= 0.5 turn) */
#pragma once
#include "fit.h"
#include <stddef.h>
#include <stdint.h>

enum { FD_NMAX = 8 };
typedef struct {
    double conf;      /* phase confidence a model needs at a sample (default 0.2) */
    double gt_conf;   /* the GT's (oracle fields, ufsm fields-oracle --fill none; default 0.3) */
    double ray_um;    /* half-length of the normal ray (default 100 um); the valid stretch must reach ray_um in total */
    double step_um;   /* ray step (0: the smallest cell edge of the inputs, clamped to 2 .. 8 um) */
    int stride;       /* output cells: every stride-th reference cell (default 1) */
    int metrics;      /* the metrics in dis: bit 0 pos, 1 cnt, 2 w4 (the command's --metrics; default all) */
} fdis_cfg;
/* per-cell GT tallies over the cells where the phase is valid and the GT known (all metrics' flags at their thresholds) */
typedef struct {
    size_t n, nwrong, nboth, nray;              /* cells; any model wrong; every model wrong; of n with a valid ray */
    size_t wrong_k[FD_NMAX];                    /* model k wrong */
    size_t w4n, w4wrong, w4flag, w4flag_wrong;  /* winding mod 4 vs the GT phase unwrapped along each axis over the longest baseline: cells
                                                   every w4 model was scored on; some off >= 0.5 turn; w4 flagged; both */
    size_t w4n_k[FD_NMAX], w4wrong_k[FD_NMAX];
    size_t flag[4], flag_wrong[4];              /* metric m (pos cnt w4 dis) past its threshold; and some model wrong there */
    size_t dis_hist[16], dis_hist_wrong[16];    /* dis / 16 bins */
} fdis_gtstat;
typedef struct {
    int D, H, W, stride; long long origin[3]; int h; double um;   /* the output grid: reference cells (i stride + stride / 2 ...), the reference's native frame */
    uint8_t *pos, *cnt, *w4, *dis, *flag;
    size_t nvalid, nray, nw4, ndis;             /* cells with phase valid, ray valid, w4 valid, dis >= 128 */
    fdis_gtstat gt;
} fdis_out;
/* in[0..n) (n 2..FD_NMAX; in[0] the reference grid), gt (or nullptr: oracle fields of any grid); 0 ok, -1 out of memory, -2 bad inputs */
int fdis_run(int n, const fit_fields_buf *const *in, const fit_fields_buf *gt, const fdis_cfg *c, fdis_out *o);
void fdis_free(fdis_out *o);
double fdis_w4_pair(const fit_fields_buf *A, const double *ta, const fit_fields_buf *B, const double *tb, const double p[3]);
/* the physical centre (um, scan frame) of output cell (z, y, x) */
void fdis_cell_um(const fdis_out *o, int z, int y, int x, double p[3]);
int cmd_field_disagree(int argc, char **argv);
