/* Sheet-aware crush augmentation (train / dino-train --aug-crush P): crushed-scroll windows from intact ones whose labels give the
   sheet geometry. One smooth bijective pull map u (output window voxel) -> x (source window voxel, the map's level), the composite of
     A  rotation + window-scale squash: q = M (u - c), M = R (I + (k - 1) d d^T)  (k 1.5..3 along d in the sheet plane: the flattened cross-section)
     B  bends / kinks along the sheet normal n: s' = s + b(a), a = (q . t1, q . t2), s = q . n  (b low-frequency sines + a
        k w log cosh((e . a - a0) / w) fold line; a shear with unit determinant, invertible for any b)
     C  gap collapse along n: sigma = F(s'; a), F monotone in s' per column, x = c + a1 t1 + a2 t2 + sigma n
   F per column (a node of a quadratic B-spline grid over a, so F is C1 in a and a convex blend of monotone columns) is the inverse of
   g(sigma) = int rho, rho = 1 - (1 - j) v(sigma): v the voidness of the source CT along the column (Otsu split, blurred 1.5 voxels),
   j the void spacing ratio (output / source, 0.05..1, smooth over the window); papyrus keeps rho = 1 (real sheet thickness), voids
   shrink by j, so sheets pack and touch. F is tabulated at unit steps with its exact derivative 1 / rho and evaluated as a cubic
   Hermite with derivatives clamped to 3 x the neighbouring secants (Fritsch-Carlson: monotone, C1). The pull's Jacobian determinant
   is k F' > 0 (F' ~ 1 in papyrus, up to ~1 / j in voids): the map compresses (outputs see more source). Every label is pulled through the same map
   from the source (CT trilinear or a 2 x 2 x 2 sub-voxel mean from a finer level, stored phase as the phasor mean like the
   spatial warp, codes nearest), so they stay exact; normals / densities follow from the pulled phase on the GPU. */
#pragma once
#include "zarr3.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
    float p;                  /* probability per eligible sample (0: off, the default) */
    float jlo, jhi;           /* void spacing ratio j (output / source), log-uniform per sample; varied +-40 % over the window */
    float squash_p, klo, khi; /* window-scale squash: probability, factor range (log-uniform); along a random direction in the sheet plane (d . n = 0) */
    float bend_p, bend_amp;   /* low-frequency bends: probability, amplitude (fraction of P) */
    float kink_p;             /* a fold line (kink) */
    float rotate_deg;         /* random rotation up to this many degrees per axis (crushed windows skip the ordinary spatial warp unless warp) */
    int fine;                 /* CT from N levels finer (2^N box mean per output voxel); sources' "crush_fine" overrides */
    int warp;                 /* 1: crushed windows also take the ordinary geometry warp (rotation / elastic / affine / zoom) after the crush map; 0 (default) none */
    float loc_amp, loc_len, loc_jmax;   /* local void ratio (--aug-crush-local A, -local-len L, -local-max J): j x exp(A z(a, sigma)), z a
                                           unit-variance field of 8 random 3D cosines (wavelengths L / 2 .. 2 L source voxels) over the
                                           columns AND along them, clamped to 0.04 .. J: some gaps close fully, others keep or (J > 1) widen
                                           (delamination-like), and neighbouring columns close differently, so the sheets buckle out of
                                           phase. A 0 (default): off, maps unchanged */
} crush_cfg;
crush_cfg crush_cfg_default(void);
/* the --aug-crush flags (train and dino-train): --aug-crush P, --aug-crush-void LO,HI, --aug-crush-squash P, --aug-crush-k LO,HI,
   --aug-crush-bend P, --aug-crush-bend-amp F, --aug-crush-kink P, --aug-crush-rot DEG, --aug-crush-fine N, --aug-crush-local A,
   --aug-crush-local-len L, --aug-crush-local-max J. 0 ok, else a message on stderr. */
int crush_cfg_parse(crush_cfg *c, int argc, char **argv);
void crush_cfg_print(const crush_cfg *c);   /* one stderr line */

typedef struct {
    int P; double c;
    double M[3][3], Minv[3][3], det;          /* stage A (det = k) */
    double sq_k, sq_d[3];
    double n[3], t1[3], t2[3];                /* the column frame: n the dominant sheet normal of the source window */
    int nsin; double bA[2], bK[2][2], bP[2];  /* stage B */
    int kink; double kS, kW, kE[2], kA0;
    double T[2], srange[2];                   /* |a_i| <= T_i and s' in srange over the output cube */
    int G[2]; double alo[2], h;               /* stage C: column nodes a = alo + (i, j) h */
    int ns; double smin, half;                /* per column ns knots s' = smin + m; voidness sampled over |sigma| <= H, H doubling from 2 max|s'|
                                                 until int rho covers s' (half at most: beyond, slope 1) */
    float *tab;                               /* [(i G1 + j) ns + m][2] = sigma, dsigma / ds' (nullptr: no collapse, F = identity) */
    double j0;                                /* the sample's void ratio (diagnostic) */
    int lat;                                  /* crush_window evaluates the map exactly every lat voxels, trilinear between (2: the
                                                 default, 8x fewer evaluations; 1: exact). CT and labels share it, so they stay consistent. */
    int rb; double rU[3], rV[3], rA[3], r0, r1, rext[3];   /* stage R (crush_map_rebend), before A; 0 off */
    int nloc; double lk[8][3], lph[8], lamp, ljmax;         /* the local void-ratio field (cfg loc_amp; 0 off): k in the (t1, t2, n) frame */
} crush_map;

/* stages A and B from the seed (n: the sheet normal, unit or not); stage C off until crush_map_columns */
void crush_map_init(crush_map *m, const crush_cfg *cfg, uint64_t seed, int P, const double n[3]);
/* stage C from the source CT along the columns: ct(ud, x) = the CT value (0..255, 0 outside the scan) at window coordinates x.
   Returns 0, or 1 when the columns hold too little CT for a papyrus / void split (stage C stays off). */
typedef double (*crush_ct_fn)(void *ud, const double x[3]);
int crush_map_columns(crush_map *m, double jlo, double jhi, uint64_t seed, crush_ct_fn ct, void *ud);
void crush_map_free(crush_map *m);
/* the map at strength t in [0, 1] (animations): squash k^t, bend / kink amplitudes x t, collapse sigma = s' + t (F(s') - s') (a blend of
   two monotone Hermites on the same knots: monotone, C1, invertible; t 0 = identity, t 1 = m). m must be rotation-free (cfg rotate_deg 0).
   out owns its table (crush_map_free). 0 ok, 1 m rotated, -1 alloc. */
int crush_map_scaled(crush_map *out, const crush_map *m, double t);
/* stage R (--aug-curv): the sheets re-bent around a moved centre. In the source the scroll axis runs along A (unit) at distance r0
   from the window centre c, U (unit, perpendicular to A) points from the axis to c; r0 0 = flat source (no axis known: the sheets
   taken as planes normal to U). The output's axis lies at c - r1 U: an output point at radius rho' and angle th' about it (from U,
   towards V = A x U) pulls from radius rho' - r1 + r0 and angle th' r1 / r0 about the source axis (flat: a = rho' - r1 along U,
   s = r1 th' along V), the coordinate along A kept. Radial distances (the wrap spacing) and arc length along the centre sheet are
   kept; the curvature there goes 1 / r0 -> 1 / r1. Invertible for |th'| < pi; voxels near th' = +-pi (behind an output axis inside the
   window) form a seam, crush_rebend_masked marks them. Sets m->rext, the footprint (stages A-C must be off: identity). */
void crush_map_rebend(crush_map *m, const double U[3], const double A[3], double r0, double r1);
/* 1 when output voxel u lies within (1 - frac) pi of the seam behind the output axis (callers blank it) */
int crush_rebend_masked(const crush_map *m, const double u[3], double frac);
/* the pull (output voxel u -> source x, window coordinates of the map's level); jdet: the Jacobian determinant (nullable) */
void crush_pull(const crush_map *m, const double u[3], double x[3], double *jdet);
/* the pull crush_window applies at output voxel u (integer): with lat 2 the trilinear blend of the exact pulls at the nodes
   u = -box (mod 2) per axis (box: a LINEAR layer with box set, whose sub-voxel Jacobians need a 1-voxel halo); with lat 1 crush_pull */
void crush_pull_window(const crush_map *m, int box, const int u[3], double x[3]);
/* the inverse (source -> output), exact to ~1e-9 voxels */
void crush_push(const crush_map *m, const double x[3], double u[3]);
/* J[i][j] = d x_i / d u_j by central differences (h 1e-4); normals map as n_out ~ J^T n_src (the pull's transpose) */
void crush_jacobian(const crush_map *m, const double u[3], double J[3][3]);
/* the source box the output cube pulls from (window coordinates, a 2 voxel margin) */
void crush_bounds(const crush_map *m, double lo[3], double hi[3]);
/* every source read of the map (columns and pull) lies within +-ext[d] of the window centre (known before crush_map_columns) */
void crush_footprint(const crush_map *m, double ext[3]);

/* ---- pulling arrays through the map: 128^3 tiles decoded into a caller arena (LRU), output processed in blocks whose tiles are
   pinned while they are read (no window-sized source copy: the footprint can be several P^3 at no RAM cost) ---- */
enum { CRUSH_LINEAR = 0, CRUSH_PHASE = 1, CRUSH_NEAREST = 2 };
typedef struct {
    z3 *z;            /* the array */
    int mode;         /* CRUSH_LINEAR (u8 trilinear), CRUSH_PHASE (phasor mean of the known PHASE codes of 8 neighbours, unknown under half
                         the weight, as the spatial warp), CRUSH_NEAREST */
    int box;          /* LINEAR: the mean of 8 pulls at the output voxel's +-1/4 sub-positions (a finer level: its 2^N box) */
    uint8_t fill;     /* outside the array */
    double a, b[3];   /* array voxel = a (o + x) + b per axis (o the window origin, x window coordinates at the map's level) */
    uint8_t *out;     /* P^3 */
} crush_layer;
/* the array coordinates of a level (finer by f levels: f > 0, coarser: f < 0) relative to the map's level, the pyramid convention
   p_l = (p_0 + 0.5) / 2^l - 0.5 */
void crush_layer_level(crush_layer *L, int finer);
/* the stored-phase grid of the sampler's phase_window (shift levels coarser: nearest (g + shift) >> shift) */
void crush_layer_phase(crush_layer *L, int shift);

typedef struct {
    uint8_t *arena; int nslots; uint64_t clock, decoded, *key; struct crush_slot *slot; int err;
} crush_cache;
int crush_cache_init(crush_cache *cc, uint8_t *arena, size_t bytes);   /* bytes / 2 MiB slots (>= 16); 0 ok */
void crush_cache_free(crush_cache *cc);                                /* (the arena stays the caller's) */
/* point reads through the cache (map building): trilinear at array coordinates p, nearest code */
double crush_linear_at(crush_cache *cc, const crush_layer *L, const double p[3]);
int crush_nearest_at(crush_cache *cc, const crush_layer *L, const double p[3]);
/* the dominant sheet normal of a stored phase over the central half of the window (structure tensor of the phase gradient, stride
   P / 32); hint (nullable) seeds the power iteration. Returns the number of gradient samples (n unchanged under 32). */
int crush_phase_normal(crush_cache *cc, const crush_layer *ph, const int64_t o[3], int P, const double *hint, double n[3]);
/* every layer's out[P^3] pulled through the map; 0 ok, -1 read failure */
int crush_window(const crush_map *m, crush_cache *cc, const int64_t o[3], crush_layer *L, int nl);
