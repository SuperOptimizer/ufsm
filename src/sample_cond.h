/* train --cond-fields (2026-11-27, condnet): another model's fit fields (a source's "cond" sets, sources.h; e.g. try9's predict --fit-zarr at
   19.2 um cells) pulled onto a coarse grid of a training window through the window's cube symmetry and continuous warp, as the --distill
   targets (sample_emb.c: the same cell-centre coordinates). The net reads them as a second 32-channel input block (unet_cfg.cond_ch,
   nn_cond_mx: x2 trilinear onto the latent cells).
   Channels (int8, x 127; channel 11 = 0: no information: outside every set or its covered tiles, a dropped window):
     0, 1   the phase vector (pc - 128) / 127, (ps - 128) / 127 (confidence x cos / sin 2 pi phase; scalar under the symmetries)
     2      material w / 255
     3      sheet spacing code sp / 255 (0 unknown)
     4      winding mod 4 confidence dq / 255
     5..10  per OUTPUT axis a (z, y, x): dq / 255 x (cos, sin) (2 pi b / 256), b the source plane d{z,y,x}[perm[a]] (W(x + d e) - W(x) mod 4
            turns x 64), sin negated when the axis is flipped (the difference along a reversed axis is the negated one). The warp's small
            rotation does not turn these vectors.
     11     presence: the trilinear weight of the field cells that exist (1 inside a set, 0 absent). Lead 11-27: absence is marked,
            not read as zero fields (an air cell of a set is present with zero planes)
   A set's optional cond_tiles.txt ("tile T" then "iz iy ix" lines of T^3 tiles of its native voxels from 0, the predict --tile-list it was
   computed on, ufsm occ-tiles) limits it to those tiles: cells elsewhere are absent (the predict wrote nothing there).
   Trilinear in the decoded values over the 8 field cells around each coarse cell centre (cells outside the set count 0). */
#pragma once
#include "sources.h"
#include "spatial_augment.h"
#include <stdint.h>

#define COND_K 12
/* the window's conditioning planes: COND_K int8 planes over the (Pz / div) (P / div)^2 coarse cells of a Pz x P x P level-l window at corner o
   (out [COND_K][cells]). 0 (written; all 0 without "cond"), -1 on a read error */
int sample_cond_window(const source *s, int div, int l, const int64_t o[3], int Pz, int P, const int perm[3], const int flip[3], const spatial_aug *sa,
                       int8_t *out);
/* one field cell's decoded channels (the table above) for an output frame (perm / flip), from its 8 planes pc ps w sp dq dz dy dx (has_w4 0:
   no winding planes: channels 4..10 0; channel 11 = 1); tests */
void sample_cond_decode(const uint8_t v[8], int has_w4, const int perm[3], const int flip[3], float out[COND_K]);
