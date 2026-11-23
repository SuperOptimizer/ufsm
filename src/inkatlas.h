/* ufsm ink-atlas (2026-11-23, atlasink): the pure parts of importing a First Letters ink-atlas surface (HF SharpMeow/first-letters-ink-
   atlas: seven public 2-D ink models + their z-score ensemble on one 9.6 um canvas per traced mesh) as confidence-gated pseudo ink labels
   in ink-teach's window form. The command (src/inkatlas_cmd.c) renders the mesh's CT through our render path, registers the atlas canvas
   against it, gates the ensemble and writes the label / mask rasters and the (possibly transposed) mesh. */
#pragma once
#include <stddef.h>
#include <stdint.h>

/* the 8 grid symmetries t of a W x H raster: bit 0 mirror the columns, bit 1 mirror the rows, bit 2 transpose first (out H x W).
   out[r, c] = in[src(r, c)]; *oW, *oH the output size. */
void ia_xform(const uint8_t *in, int W, int H, int t, uint8_t *out, int *oW, int *oH);
/* Pearson r of a[r, c] and b[r + dy, c + dx] (both W x H) over the px where both are > 0 and the shifted index is inside, every
   step-th row and column; *n the px used (NAN when < 1000) */
double ia_pearson(const uint8_t *a, const uint8_t *b, int W, int H, int dy, int dx, int step, size_t *n);
/* out[r, c] = in[r + dy, c + dx] (0 outside) */
void ia_shift(const uint8_t *in, int W, int H, int dy, int dx, uint8_t *out);
/* the reading side from the atlas face check's pair votes: 1 forward, -1 reverse, 0 undecided (winner < min_win or < ratio x loser) */
int ia_face_order(int fwd, int rev, int min_win, double ratio);

typedef struct { double z_ink, z_bg, model_z; int agree; } ia_gate_cfg;   /* defaults 2.5, 0.5, 2.0, 3 */
/* per px k with cov[k]: ensemble z = -2 + 6 ens[k] / 255; model z_m = (mod[m][k] / 255 - mean[m]) / sd[m];
   ink (lab 255, msk 255) where z >= z_ink and >= agree models have z_m >= model_z; background (lab 0, msk 255) where z <= z_bg;
   else lab 0, msk 0. *nink / *nbg counted. */
void ia_gate(const uint8_t *ens, const uint8_t *const *mod, const double *mean, const double *sd, int nm, const uint8_t *cov, size_t n,
             const ia_gate_cfg *g, uint8_t *lab, uint8_t *msk, size_t *nink, size_t *nbg);
