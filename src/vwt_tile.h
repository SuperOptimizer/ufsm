/* vwt_tile: the slab tiling of a box for whole-volume inference of the ray model (ufsm vwt-infer = pass A, PF; ufsm vwt-stitch = pass B, VW).

   Tiling (production geometry, as villawind ray-check --geom umb): radial horizontal rays from the umbilicus (umbilicus.json control points
   x y z, linear in z, clamped). Band b: zc = z0 - 64 + 56 + sz b over the box (sz 112: slab z extent 128); ring m: r0 = r00 + sr m (r00 16,
   sr 320: 64 voxels of ray overlap); angle index a in [0, na(m)), na = ceil(2 pi r_mid / st), dth = 2 pi / na, th = a dth (st 112: tangential
   step at the ring's mid radius r_mid = r0 + (L - 1) / 2). Slab frame = src/villawind.c frame_of(dir (cos th, sin th, 0), origin (umb(zc) +
   r0 dir, zc)): axis_a = tangential, axis_b = +-z, d = radial. Slabs whose world bbox misses the box are dropped (the CT-empty test is the
   caller's). Coordinates: xyz in the CT level's voxels (9.6 um for P4 level 2); the box is z y x.

   Pass A record files in OUT (all fixed-size, written with pwrite at computed offsets, so any number of processes can fill disjoint slab
   ranges): OUT/tiling.bin (vwt_tiling_save), OUT/phase.f16 (per slab VWT_PH_N fp16: the phase at slab voxels (4u + 1.5, 4v + 1.5, 2w + 0.5)
   = the mean of the 4x4x2 block, MINUS the slab's centre value; 0x7e00 (NaN) where any voxel of the block is invalid), OUT/centre.f32 (per slab
   the centre value: the phase at (T/2 - 0.5, T/2 - 0.5, L/2 - 0.5), NaN = slab not computed), OUT/pairs.f32 (per pair and side
   VWT_PAIR_PTS floats: the side's phase (absolute, trilinear on the full-resolution output) at the pair's points, NaN past npts / invalid). */
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VWT_PU 32
#define VWT_PW 192
#define VWT_PH_N (VWT_PU * VWT_PU * VWT_PW)
#define VWT_PAIR_PTS 256

typedef struct { int b, m, a; double zc, r0, th; double o[3], ax[3], bx[3], d[3]; } vwt_slab;   /* frame xyz: voxel (i, j, k) at o + i ax + j bx + k d */
typedef struct { int i, j; int npts; float pts[VWT_PAIR_PTS][3]; } vwt_pair;                    /* slabs i < j, world points xyz */
typedef struct {
    int T, L; double sz, sr, st, r00; int64_t box[6];
    int n; vwt_slab *s;
    int np; vwt_pair *p;
} vwt_tiling;

/* the tiling of box (z, y, x, nz, ny, nx) with its neighbour pairs and their point sets (4-voxel lattice over the intersection of both slabs
   shrunk by 8 voxels, subsampled to <= VWT_PAIR_PTS); 0 on success */
int vwt_tiling_build(const char *umb_json, const int64_t box[6], int T, int L, double sz, double sr, double st, double r00, vwt_tiling *t);
void vwt_tiling_free(vwt_tiling *t);
int vwt_tiling_save(const vwt_tiling *t, const char *path);
int vwt_tiling_load(const char *path, vwt_tiling *t);
/* world xyz -> slab (i, j, k) */
static inline void vwt_to_slab(const vwt_slab *s, const double p[3], double u[3]) {
    double q[3] = {p[0] - s->o[0], p[1] - s->o[1], p[2] - s->o[2]};
    u[0] = q[0] * s->ax[0] + q[1] * s->ax[1] + q[2] * s->ax[2];
    u[1] = q[0] * s->bx[0] + q[1] * s->bx[1] + q[2] * s->bx[2];
    u[2] = q[0] * s->d[0] + q[1] * s->d[1] + q[2] * s->d[2];
}
/* the pairs touching slab k: indices into t->p and which side (0: k == i, 1: k == j); returns the count (out arrays: caller frees) */
int vwt_slab_pairs(const vwt_tiling *t, int k, int **pairs, int **side);

#ifdef __cplusplus
}
#endif
