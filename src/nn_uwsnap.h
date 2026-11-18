/* The slab fit's voxel unwrapping on the GPU (src/nn_uwsnap.cu; 2026-11-18): src/fit.c uw_accum_snap (fit --ref-snap 1, no repair) per
   lattice node on the device field planes. One thread per node sums its voxels in the CPU's (z, y, x) order, so the sums are the CPU's
   up to the device's atan2 / hypot (and fast-math contraction): deterministic. The tables are the host's own values (fit.c builds them
   with the CPU code's expressions): per field row z the axis, the ref's z, the two models' knot values and the ref's axis; per (z, y)
   the trilinear y / z parts of both lattices (ev_party_init); per node row / column the field rows / columns nearest to it. */
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct { uint64_t base[4]; double wzy[4]; } uws_row;   /* ev_party: the four (z, y) corners' delta offsets and weight products */
typedef struct { double cy, cx, zr, pit, off, rpit, roff, gy, gx; int same; } uws_z;   /* per field row: see fit.c uw_accum_snap */
typedef struct {
    int D, H, W, oy, ox; double h;                          /* the field arrays (a material crop: oy ox) */
    const uint8_t *pc, *ps, *sp, *w;                        /* device planes, D x H x W */
    double min_conf, packed_um, um;                         /* uw_q's (uw_pdv > 0 is not supported here) */
    const double *tab;                                      /* host, 2 x 65536 per (pc, ps) byte pair: uw_q's conf, uw_rec's ang / (2 pi) */
    const uws_z *zt; const uws_row *rm, *rr;                /* host: D rows; D x H rows each for the model and the ref */
    int lz, ly, lx, L; double iL; int dir; const float *delta;           /* the model (host delta, lz x ly x lx) */
    int rL, rlx, rdir; double riL; const float *rdelta; size_t rn;      /* the ref's lattice (host; rn floats from rdelta: the planes the rows read,
                                                                           rr's offsets relative to it) */
    double roff1, roff2, rbias; int32_t kb;                 /* ref_off[1..2], the sub-turn offset, the integer base */
    const int *zr, *yr, *xr;                                /* host: per node z / y / x index the field rows [zr[a], zr[a + 1]) ... */
} uws_args;
/* acc / wsum: lz x ly x lx (written); *reached / *over: voxels given an integer / out of range. 0, or -1 (no device memory: the caller
   runs the CPU pass) */
int nn_uw_snap(const uws_args *a, double *acc, double *wsum, size_t *reached, size_t *over);
#ifdef __cplusplus
}
#endif
