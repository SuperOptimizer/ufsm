/* Sheet-identity embeddings as winding evidence (src/fit_emb.c): which integer offset between two places of a winding fit makes the
   same physical sheet carry the same W, judged by an embedding volume (dinovol-embed tokens_unit / tokens / dense, or any unit-vector
   field whose cosine is high on the same sheet and low on neighbouring sheets).

   Coordinates: native voxels of the fit (the fields' scan level), relative to the origin given at open (a fit box's origin: the
   fields' "origin"; 0 for global scan coordinates). The embedding volume's voxels are native voxels x scale (embedding-volume voxels
   per native voxel, 1 when the embedding was made at the fit's level).

   The vote (emb_shift_vote): side a and side b each give a point, its W (the fit's or the flooded winding; only W_b - W_a matters, so
   any common gauge), the unit direction in which W grows there and the local sheet spacing (native voxels per turn). The sheet centre
   nearest a (W = n + phi) is located along a's direction; on b's line the sheet centres at W = W_level(a) + r for r = -rmax..rmax.
   Each centre's vector is the mean of three samples across the sheet (-1, 0, +1 native voxel along the direction).
   Result r: the sheet at a's level L is the one b currently calls L + r, i.e. b's W is r turns too high relative to a
   (correct b by W_b - r). conf = the cosine margin best - second best (0..2), NOT calibrated: use ufsm emb-calib for
   P(r right | conf) on ground truth (HOLD, tokens_unit: >= 0.99 at conf >= 0.2 for D <= 48; 0.88-0.95 at D 128; < 0.05 about 0.45). Returns 0 with a vote, -1 when a side has no embedding (outside the loaded region / masked). */
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct emb_vol emb_vol;

/* load the region of output `name` (tokens_unit, tokens, dense, ink) of the dinovol-embed group covering native box [org, org + n)
   (+ pad native voxels around it) into memory as int8 unit vectors; nullptr on failure (message printed) */
emb_vol *emb_vol_open(const char *group, const char *name, const long long org[3], const long long n[3], double scale, int pad);
/* an in-memory volume (tests): unit vectors as int8 x 127 [nz][ny][nx][C] at cell step `step` embedding voxels, cell (0,0,0) covering
   embedding voxels [o, o + step) per axis; the data are copied */
emb_vol *emb_vol_wrap(const int8_t *v, const int64_t n[3], int C, int step, const int64_t o[3], const long long org[3], double scale);
void emb_vol_free(emb_vol *e);
/* the papyrus gate (optional): each sheet centre snaps to the nearest voxel along the direction (within half a turn) whose raw CT
   (native voxel, box-local; the array covers [o, o + n) and is owned by e from here on) is >= `level`, and its samples below `level` are
   skipped, so votes compare papyrus only. emb-calib 10-25 (wind96 fits, dinovol tokens_unit, P(right) at D 16 / 48 / 128):
   HOLD 0.996 / 0.960 / 0.829 ungated vs 0.984 / 0.947 / 0.811 gated, C1 0.928 / 0.659 / 0.382 vs 0.927 / 0.649 / 0.385, 139-2
   0.981 / 0.930 / 0.777 vs 0.952 / 0.892 / 0.744: no gain, fit runs ungated (without the snap, samples only: 0.80-0.86 at D 16).
   emb_vol_papyrus_zarr reads the CT of the box org..org + n (+ pad) from a native-level uint8 array. */
typedef struct z3 z3;
void emb_vol_set_papyrus(emb_vol *e, uint8_t *ct, const int64_t o[3], const int64_t n[3], int level);
int emb_vol_papyrus_zarr(emb_vol *e, z3 *z, const long long org[3], const long long n[3], int pad, int level);
int emb_vol_dim(const emb_vol *e);
/* the trilinear unit vector at p (native voxels, box-local); 0 when no corner has a vector */
int emb_at(const emb_vol *e, const double p[3], float *v);

typedef struct { double p[3], W, n[3], sp; } emb_side;   /* point (native, box-local), winding, unit direction of growing W, native voxels per turn */
int emb_shift_vote(const emb_vol *e, const emb_side *a, const emb_side *b, int rmax, double phi, int *r, float *conf);
/* n votes in parallel (threads 0: all cores); r[k] = INT32_MIN where vote k has no embedding */
void emb_shift_votes(const emb_vol *e, size_t n, const emb_side *a, const emb_side *b, int rmax, double phi, int32_t *r, float *conf, int threads);
/* the loaded cells: int8 unit x 127 [n0][n1][n2](C), cell (0,0,0) at embedding voxel o, cells of `step` embedding voxels */
const int8_t *emb_vol_cells(const emb_vol *e, int64_t n[3], int64_t o[3], int *step);
