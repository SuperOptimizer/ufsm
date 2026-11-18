/* ufsm merge-flag (src/mergeflag.c, 2026-11-18): where the network's phase merges touching sheets (one phase cycle over several GT
   sheets), from the fit fields alone. Per field cell three box-filtered features of the network's planes:
     amp9  the phase confidence |(pc, ps)| averaged over material cells in a 9-cell box (low where the net is unsure)
     R5    the phase's turns per cell along its own gradient (the angle of the 5-box mean of z(x + e_a) conj z(x), per axis a, as a
           vector) times the density head's sheet spacing in cells (5-box mean of the known sp / 8): phase turns per density turn,
           ~1 where both count the same sheets, ~1/3 where the phase runs one cycle across three touching sheets
     sp9   the density head's sheet spacing in cells (9-box mean of the known sp / 8): packed regions are narrow
   and a logistic rule p = 1 / (1 + exp(-(b + c_amp amp9 + c_R R5 + c_sp sp9))) fitted on GT (DESIGN 2026-11-18 "merge-flag").
   Output plane mf (uint8): round(255 p) on material cells (w > 0), 0 elsewhere; the zarr attributes carry the coefficients and the
   recommended flag threshold (byte, attributes.ufsm.mf_thr). Box filters use edge replication at the box borders (scipy
   uniform_filter mode "nearest"); the neighbour product at the last cell of an axis repeats the one before. */
#pragma once
#include <stdint.h>
#include <stddef.h>

typedef struct { double b, c_amp, c_r, c_sp; } mf_coef;
extern const mf_coef mf_coef_default;
extern const int mf_thr_default;   /* the flag threshold byte (p >= mf_thr / 255: uncertain) */

/* the features and mf of the cells [r, n - r) of a D x H x W block whose borders (r = 4 cells) hold edge-replicated neighbours;
   pc ps sp w the field planes of the block (C order). gz / gy / gx: the block's first cell's global index and the global extents
   (for the last-cell rule of the neighbour products). out: D x H x W (border cells left 0). Optional features out (amp9, r5, sp9;
   nullptr: not kept), floats D x H x W. */
void mf_block(const uint8_t *pc, const uint8_t *ps, const uint8_t *sp, const uint8_t *w, int D, int H, int W, const long long g0[3],
              const long long gn[3], const mf_coef *c, uint8_t *out, float *amp9, float *r5, float *sp9);
int cmd_merge_flag(int argc, char **argv);
int cmd_wrap_mf(int argc, char **argv);
