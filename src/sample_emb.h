/* train --distill feature targets (src/sample_emb.c): a source's "emb" groups (sources.h) pulled onto the grid of a training window's
   distillation head through the window's cube symmetry and continuous warp */
#pragma once
#include "sources.h"
#include "spatial_augment.h"
#include <stdint.h>

/* the source-window coordinates (level voxels from the window corner, voxel k centred at k; z, y, x) of the centre of every head cell
   of a Pz x P x P window with div^3 voxels per cell: cell (gz, gy, gx) -> V[3 k], k = (gz G1 + gy) G2 + gx. Output voxel u maps to the
   warp's pull U (spatial_aug_pull at continuous u; none when sa is nullptr or inactive), then through the symmetry (sample.c: source axis
   perm[d] = flip[d] ? n_d - 1 - U_d : U_d, n = (Pz, P, P)) */
void sample_emb_coords(int Pz, int P, int div, const int perm[3], const int flip[3], const spatial_aug *sa, float *V);
/* the window's targets: C channel planes of int8 x 127 unit vectors over the (Pz / div) (P / div)^2 cells (all 0: none), trilinear in the
   source's first "emb" group with all 8 stencil corners (else the first with >= half the weight), renormalised; o the window corner in level-l voxels. 0 (out written; all 0 without
   "emb"; groups that cannot be opened are skipped with a warning), -1 on a read error or a channel count other than C. Per-thread group handles (sampler workers) */
int sample_emb_window(const source *s, int C, int div, int l, const int64_t o[3], int Pz, int P, const int perm[3], const int flip[3], const spatial_aug *sa,
                      int8_t *out);
/* train --distill-papyrus: dinovol's embeddings in the voids carry the mesh labels' noise, so the targets are kept only on papyrus.
   sample_emb_pap2: the papyrus count (raw CT >= level, 0..8) of every 2^3 block of a Pz x P x P source window (before the appearance
   augmentation), (Pz / 2) (P / 2)^2 bytes (Pz, P even). sample_emb_papyrus: zero the targets of the cells whose div^3 source voxels
   around their centre (V of sample_emb_coords, the 2^3 blocks nearest it; outside the window: no papyrus) are less than frac papyrus;
   returns the number of targeted cells it cleared (*kept: those left) */
void sample_emb_pap2(const uint8_t *ct, int Pz, int P, int level, uint8_t *pap2);
size_t sample_emb_papyrus(int C, int div, int Pz, int P, const int perm[3], const int flip[3], const spatial_aug *sa, const uint8_t *pap2, float frac,
                          int8_t *out, size_t *kept);
/* (counts of sample_emb_papyrus over all windows: targeted cells cleared / kept; train reports the share) */
#include <stdatomic.h>
extern _Atomic size_t g_emb_pap_cleared, g_emb_pap_kept;
