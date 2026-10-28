/* SLIM flattening of tifxyz wraps (ufsm flatten): Scalable Locally Injective Mappings (Rabinovich, Poranne, Panozzo, Sorkine-Hornung
   2017; the flattening of Volume Cartographer / VC3D) on the wrap's own vertex grid, symmetric Dirichlet energy.
   The wrap's grid parameterisation (column, row) x step is the initial map: a planar grid, so every triangle starts positively oriented
   (no Tutte / LSCM fallback is needed). Each SLIM iteration: per triangle the Jacobian J of the map from the 3D triangle's own 2D frame to
   uv, its closed-form 2x2 SVD J = U S V^T, the closest rotation R = U V^T and the energy's proxy weights W = U diag(w) U^T with
   w_i^2 = (s_i + 1)(s_i^2 + 1) / s_i^3 (symmetric Dirichlet, no singularity at s = 1); the global step min sum A_t |W_t (J_t - R_t)|^2
   (a weighted Laplacian, both uv components coupled) by matrix-free Jacobi-preconditioned CG warm-started at the current map, plus a
   tiny proximal term; then a line search along the step capped at 0.8 of the first triangle inversion (the map stays locally injective)
   and halved until the true energy falls. Multilevel: coarser lattices (every 2^k-th grid vertex) are solved first and their uv
   displacement prolongated bilinearly to the next finer lattice (flip repair by neighbour averaging, then a flip-free line search from
   the grid map), which removes the low-frequency stretch that single-level CG converges slowly on. */
#pragma once
#include <stdint.h>
#include <stdio.h>

typedef struct {
    double energy;              /* area-weighted mean symmetric Dirichlet s1^2 + s1^-2 + s2^2 + s2^-2 (4 = isometric) */
    double aniso12;             /* area share of triangles with s1 / s2 > 1.2 (shear / anisotropic stretch) */
    double stretch12;           /* area share with s1 > 1.2 or s2 < 1 / 1.2 (any stretch or compression beyond 20%) */
    double aniso_p50, aniso_p95, aniso_p99;   /* area-weighted quantiles of s1 / s2 */
    double smax_p95, smin_p05;  /* area-weighted 95th percentile of s1, 5th of s2 */
    double area_ratio;          /* sum of uv areas / sum of 3D areas */
    long long flips, nt;        /* triangles with det J <= 0; triangles */
} flat_stats;

/* stats of the map uv (2 per vertex) of triangles tri (3 per triangle, P 3 per vertex, any units); triangles with a NaN uv corner skipped */
void flat_stats_of(const double *P, const int *tri, long long nt, const double *uv, flat_stats *s);
/* one line: "energy E aniso>1.2 A% stretch>1.2 S% aniso p50/p95/p99 .. smax p95 .. smin p05 .. area-ratio R flips F tris N" */
void flat_stats_print(FILE *f, const char *label, const flat_stats *s);

typedef struct {
    int iters;          /* SLIM iterations at the finest level (30); coarser levels 2x */
    int cg_iters;       /* CG iterations per global step (200) */
    int nthreads;       /* (16) */
    int min_verts;      /* coarsest level: the first with fewer vertices (20000) */
    int decim;          /* the finest lattice: every decim-th input vertex (1) */
    double max_edge;    /* drop triangles with a 3D edge longer than max_edge x the lattice spacing (x sqrt 2 on the diagonal) (3) */
    double tol;         /* stop when an iteration lowers the energy by less than tol relative (1e-5) */
    int verbose;
} flat_opts;
flat_opts flat_opts_default(void);

/* SLIM on one triangle mesh, uv in place (must start flip-free). Returns the final mean energy (< 0 on failure). */
double flat_slim(const double *P, int nv, const int *tri, int nt, double *uv, const flat_opts *o);

/* the flattened mesh of a vertex grid (w x h, X / Y / Z, invalid: Z < 0 or non-finite; step = native voxels between grid vertices) */
typedef struct {
    int nv, nt;
    int *gi;        /* per vertex its input grid index r w + c */
    double *P;      /* 3 per vertex (x, y, z) */
    double *uv;     /* 2 per vertex, native voxels */
    double *uv0;    /* 2 per vertex, the initial grid map (c step, r step) */
    int *tri;       /* 3 per triangle */
} flat_mesh;
void flat_mesh_free(flat_mesh *m);
/* builds the finest lattice's mesh (the largest connected component), flattens it multilevel; uv rotated / translated onto the grid map
   (Procrustes) so the sheet keeps its orientation (rows ~ z). 0 on success. before / after: stats of the grid map / the result */
int flat_grid(const float *X, const float *Y, const float *Z, int w, int h, double step, const flat_opts *o, flat_mesh *m,
              flat_stats *before, flat_stats *after);

/* the flattened map resampled on a regular uv grid of pitch ostep (native voxels: isometric): ow x oh vertices, per vertex the source
   grid position gu (column) / gv (row) (fractional, -1 invalid) and x / y / z (bilinear in the source grid where its 4 corners are valid,
   else barycentric in the mesh triangle; -1 invalid). Arrays malloc'd. 0 on success. */
int flat_resample(const flat_mesh *m, const float *X, const float *Y, const float *Z, int w, int h, double ostep, int nthreads,
                  int *ow, int *oh, float **gu, float **gv, float **ox, float **oy, float **oz);
/* enclosed holes (invalid 4-connected components not touching the border) of at most maxv vertices of the five ow x oh arrays a = x, y, z, gu,
   gv (invalid: z < 0) Laplace-filled from their rims; filled[i] = 1 there (w h bytes). Returns the filled vertex count. Small tears
   (triangles dropped as over-stretched, invalid vertices) would otherwise cut the largest valid rectangle down to a sliver. */
long flat_fill_holes(float *a[5], int w, int h, long maxv, uint8_t *filled);
/* the largest axis-aligned rectangle of valid (!= 0) cells of a w x h mask: corner *x0 / *y0, size *rw x *rh (0 x 0 when none) */
void flat_largest_rect(const uint8_t *valid, int w, int h, int *x0, int *y0, int *rw, int *rh);
void flat_largest_rect_aspect(const uint8_t *valid, int w, int h, double maxasp, int *x0, int *y0, int *rw, int *rh);   /* max side / min side <= maxasp (<= 0: any) */
