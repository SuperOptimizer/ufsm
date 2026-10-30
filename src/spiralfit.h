/* ufsm spiral-fit: a port of the ScrollPrize villa spiral fitter's model (villa spiral-fitting/transforms.py, flow_fields.py,
   losses.py at 8ff06f7, 2026-10-29) as the coarse winding reference of the coarse-to-fine unwrap (in place of count-field's
   least-squares U). The scroll -> ideal-spiral map is a diffeomorphism:
     x (box-local native) -> minus the umbilicus (our axis table) -> 2 stationary velocity fields, each a coarse + a fine cubic
     B-spline lattice, integrated by RK4 for unit time -> the gap transform (per winding k, angle bin, z bin a log gap g: gap =
     pitch e^g; winding radii r_k = pitch theta / 2 pi + sum_{j<k} gap_j) -> the ideal Archimedean spiral (radius U pitch).
   U (turns) is continuous, W = U - theta_spiral / 2 pi is the winding (integer on the sheets), so the reference has no vortices
   and sheets never cross. Losses on the pooled count-field targets g (grad W, turns per native voxel, weight c), villa-style:
   normals 1 - |m.g^| and density (log(m.g^) - log|g|)^2 with m = J^T grad_P W (J by central differences), the in-sheet symmetric
   Dirichlet energy of J (villa get_symmetric_dirichlet_loss), and a Huber anchor to an initial U (the count-field ref) that decays.
   Not ported: the human evidence (patches, fibres, point collections: our Paris 4 GT), tracks, the outer shell, the per-z linear
   map (the coarse lattice reproduces linear maps), the learnable pitch. */
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct {
    /* the pooled data cells (host): solve-cell centre index (z, y, x), unit target direction, log density (turns per native voxel),
       weight, initial U (turns, -1e38: none) */
    size_t nd; const int32_t *cz, *cy_, *cx_; const float *gz, *gy, *gx, *lrho, *c, *u0;
    size_t npin; const float *pz, *py, *px, *pu; double w_pin;   /* ground-truth pins: box-local native points, U' targets, weight */
    const float *aw; double dfloor;    /* anchor confidence per cell (nullptr: 1); data terms x (dfloor + (1 - dfloor)(1 - aw)) */
    int d, hh, w; double cs;            /* solve grid (cells) and its cell size (native voxels) */
    const float *ay, *ax; int na;       /* the axis per native z (box-local) */
    int dir; double pitch;              /* spiral sense, pitch (native voxels per turn) */
    /* model */
    int nst, nrk, Lf, sfac;             /* flow stages, RK4 steps per stage, fine lattice spacing (native), coarse factor */
    double La, Lz; int K;               /* gap angle bin length at the outer radius / z bin (native), windings (0 auto) */
    /* optimisation */
    int steps, nsamp, warm, fwd; double lr_c, lr_f, lr_g, lr_end, lr_dr;
    double w_n, w_d, w_sd, w_a0, w_a1, eps, gmin, gmax, huber;
    unsigned seed; int gpu, verbose;
    /* output: U' per solve cell (d x hh x w, turns, in count-field's convention U' - dir theta_x / 2 pi = winding) */
    float *uout;
} sf_cfg;

#ifdef __cplusplus
extern "C" {
#endif
int nn_spiral_fit(sf_cfg *c);                      /* 0 ok */
int nn_spiral_gradcheck(int verbose);              /* CPU reference: analytic vs finite-difference gradients; returns #failures */
int nn_spiral_has_gpu(void);                       /* 1 when a CUDA device is visible */
#ifdef __cplusplus
}
#endif
