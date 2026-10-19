/* Winding fitter (ufsm fit): a smooth "unwrapped radius in turns" U(x) fitted to dense fit fields (winding phase, sign-free
   normal n n^T, sheet spacing, material weight), so that the winding coordinate W = U - dir theta / 2 pi has its integer level
   sets on the recto faces. theta = atan2(y - cy(z), x - cx(z)) around the umbilicus; dir = +1 when the radius grows with the
   unwrapped theta along a sheet (PHercParis4). Every loss reads cos / sin (2 pi W) and grad W, both smooth away from the axis, so
   the branch cut of theta only shows in floor(W). U = r / pitch(z) + offset(z) + delta(x): pitch and offset on z knots (linear),
   delta on a dense trilinear lattice refined coarse to fine. Coordinates are native voxels relative to the field box origin. */
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct {
    int D, H, W;                    /* field grid */
    double h;                       /* field voxel edge in native voxels; voxel (i, j, k) is centred at h (i + 0.5, j + 0.5, k + 0.5) */
    const uint8_t *pc, *ps;         /* phase vector: 128 + 127 * (confidence * cos / sin (2 pi phase)), 128 = no information */
    const uint8_t *t[6];            /* n n^T x255: n_z^2, n_y^2, n_x^2, 0.5 + n_z n_y, 0.5 + n_z n_x, 0.5 + n_y n_x */
    const uint8_t *sp;              /* sheet spacing in units of h / 8 (0 = unknown), i.e. 8 / (h rho); native voxels at h = 8 */
    const uint8_t *w;               /* material fraction x255 (0 = no data) */
    const float *cy, *cx;           /* umbilicus per native z of the box, D * h + 1 entries (box-local coordinates) */
} fit_fields;

typedef struct {
    int dir;                        /* +1 / -1, 0 = choose the more phase-coherent */
    int knot_dz;                    /* native voxels between pitch / offset knots */
    int nstage, lattice[8], steps[8];   /* delta lattice spacing (native voxels) and Adam steps per stage */
    float lr, lr_knot;              /* Adam learning rates for delta and the knots */
    float w_phase, w_normal, w_density, w_nondeg, w_smooth, w_decay;
    int phase_stage;                /* stages before it fit the gradient terms only (normal, density: no integer ambiguity, they
                                       integrate the turn count); the phase term (periodic) refines from this stage on */
    float phase_ramp;               /* phase_stage: phase weight from phase_ramp * w_phase to w_phase */
    int unwrap_voxels;              /* phase-stage init: 1 voxel-level quality-guided unwrapping (default), 0 lattice nodes, -1 none
                                       (the phase stages start from the gradient stages' W) */
    double uw_split;                /* > 0: two-level voxel unwrapping: tiles flood their voxels of confidence >= uw_split in parallel,
                                       the pieces are stitched by border votes, then the serial flood does the rest (0: serial only) */
    int uw_tile;                    /* its tile edge (field voxels) */
    double uw_min_conf;             /* the unwrapping's lowest phase confidence (0: 0.3) */
    int uw_seeds;                   /* > 1: consensus of this many voxel floods from seeds spread over the box (src/fit.c uw_consensus) */
    double uw_pdv;                  /* > 0: the unwrapping quality (flood order, node weights) times exp(-pdv / uw_pdv^2), pdv the
                                       phase derivative variance (turns^2): dislocated / noisy phase is flooded last */
    int analytic;                   /* 1: nn_fit_loss2 (analytic grad W, voxel minibatches; knots fixed after the prior), 0: nn_fit_loss */
    int cpu, threads;               /* cpu: the CPU backend (no GPU; analytic only), threads (0: all cores, UFSM_FIT_THREADS) */
    float phase_robust;             /* > 0: the phase residual's Cauchy scale in turns (src/fit_terms.h); 0: the plain 1 - cos */
    float conf_pow;                 /* the phase weight min(conf, 1)^conf_pow (1: linear, as before) */
    float w_pgrad, pgrad_delta;     /* phase-gradient matching (src/fit_terms.h fit_pgrad_term): weight (0: off) and Huber scale
                                       (relative gradient error) */
    float bend;                     /* > 0: curvature-adaptive smoothness: per lattice node w_smooth / (1 + bend x b), b = the
                                       node's normal dispersion 1 - lambda_max(mean n n^T) / tr (0 planar sheets, up to 2/3) x 10 */
    double um;                      /* native voxel size: the slab driver's lattices / pooling are physical (2.4 um: 512 ... 32) */
    double packed_um;               /* > 0: where the fields' sheet spacing is below this (um) the phase is trusted less (weight
                                       ramping to 0.1 at 0.6 x it, and those voxels out of the integer unwrapping): the density
                                       and normal terms count the turns there (with a density that counts packed sheets, e.g.
                                       ufsm gap-density) */
    const struct fit_model *init;   /* start from this model instead of the prior (its direction and knots; delta sampled at the first
                                       stage), its box origin at init_off (native) in this box's coordinates negated: point p here is
                                       p + init_off there */
    double init_off[3];
    const struct fit_model *seed;   /* voxel unwrapping: field voxels below box-local z seed_z take their integer from this model
                                       (its z = ours + seed_off) instead of flooding from one seed (the previous slab of a set) */
    double seed_off, seed_z;
    int batch;                      /* analytic: field voxels per step (about; 0 = all) */
    float core;                     /* no data terms within core * pitch of the axis */
    int verbose;
} fit_cfg;

typedef struct fit_model {
    int dir, nk, knot_dz;
    double dir_vote;                /* |median winding per radial ring| of the prior (1 = clear; < 0.5 ambiguous) */
    float *pitch, *offset;          /* per knot (knot k at box-local z = k * knot_dz) */
    int L, lz, ly, lx;              /* lattice spacing and dims (node (a, b, c) at box-local (a L, b L, c L)) */
    float *delta;                   /* lz * ly * lx */
    int D, H, W; double h;          /* the field box (native extent D h, H h, W h) */
    long long origin[3];            /* the box's native origin (set by the caller; saved with the model) */
    double um;                      /* native voxel size (saved; 0 = unknown, read as 2.4) */
    float *cy, *cx;                 /* axis table (D h + 1 entries) */
    double loss[7];                 /* final loss parts: total, phase, normal, density, nondegeneracy, phase gradient, smoothness */
} fit_model;

fit_cfg fit_cfg_default(void);
int fit_scale_L(int L, double f);   /* a lattice / knot spacing scaled by f (>= 2) */
fit_model *fit_run(const fit_fields *f, const fit_cfg *c);
void fit_free(fit_model *m);
double fit_W(const fit_model *m, double z, double y, double x);   /* winding coordinate at a box-local native point (host) */
int fit_save(const fit_model *m, const char *dir);
/* a fit-field directory (the input of ufsm fit): pc ps t0..t5 sp w .raw (shape[0] x shape[1] x shape[2] uint8) and fields.json
   (origin, shape, h, the axis per native z of the box in box-local coordinates, na = shape[0] h + 1 entries) */
int fit_fields_save(const char *dir, const long long origin[3], const int shape[3], int h, double um, uint8_t *const a[10], const float *cy, const float *cx, int na);   /* um: the native voxel size */   /* dir/model.json, delta.f32 (lattice), axis.f32 (cy then cx) */
fit_model *fit_load(const char *dir);
/* the radius along the ray at box-local z and angle theta from the axis where W = k (bisection between consecutive samples
   with a sign change, the crossing nearest r_hint); returns 0 and sets *r, or -1 when the ray crosses no W = k inside the box */
int fit_ray_root(const fit_model *m, double z, double theta, double k, double r_hint, double *r);
/* initial pitch / offset per knot and the direction from the fields (no labels): exposed for tests */
int fit_init_prior(const fit_fields *f, const fit_cfg *c, fit_model *m);

/* GPU side (src/nn_fit.cu) */
typedef struct {
    int D, H, W; float h;
    const uint8_t *pc, *ps, *t[6], *sp, *w; /* device (or host: the CPU backend) */
    const float *cy, *cx; int naxis;
    int dir, nk, knot_dz; const float *pitch, *offset;
    int L, lz, ly, lx; const float *delta;
    float w_phase, w_normal, w_density, w_nondeg, core;
    int stride; unsigned salt;              /* nn_fit_loss2: voxels with hash(i, salt) % stride == 0 this step, data terms x stride */
    float packed_sp, packed_floor;          /* > 0: the phase weight ramps from 1 at spacing code packed_sp down to packed_floor at 0.6 x it */
    float phase_robust, conf_pow;           /* fit_cfg's (src/fit_terms.h) */
    float w_pgrad, pgrad_delta;
    const float *pg;                        /* pooled fields only: 4 planes Gz Gy Gx c (the phase gradient, turns per native voxel, and its
                                               confidence; null: from pc / ps on the fly) */
} fit_dev;
/* the data terms over every field voxel: loss parts (double[6]: total, phase, normal, density, nondeg, phase gradient, accumulated)
   and the gradients of delta and of the knots (accumulated: zero them first) */
void nn_fit_loss(const fit_dev *f, double *parts, float *g_delta, float *g_pitch, float *g_offset);
/* the same data terms with grad W analytic (trilinear delta, the prior's and theta's derivatives, the axis slope) at the voxel
   centre only, on a random 1 / stride subset of the voxels; the knots get no gradient (delta carries the fit) */
void nn_fit_loss2(const fit_dev *f, double *parts, float *g_delta);
void fit_loss2_host(const fit_dev *f, double *parts, float *g_delta);   /* the same on the CPU backend (host arrays in f; stratified subsets) */
/* smoothness w_smooth * sum_n wn_n (laplacian delta)_n^2 + w_decay * sum delta^2 over the lattice (lap: lattice-sized scratch; wn:
   per-node weights, nullptr = 1): adds to parts[6] and g_delta */
void nn_fit_smooth(const float *delta, int lz, int ly, int lx, float w_smooth, float w_decay, const float *wn, float *lap, double *parts, float *g_delta);
/* predict --fit-fields: pool a window's interior (logits [c][W^3] fp32, the window's CT) into the box's fit-field accumulators
   (12 planes over F = ceil(box / h), see src/nn_fit.cu); t = the window origin relative to the box */
void nn_fit_accum(const float *lg, const uint8_t *ct, int W, int halo, int tz, int ty, int tx, const int box[3], int h, const int F[3],
                  int ch_recto, int ch_phase, int ch_normal, int ch_density, float dscale, float *acc);   /* dscale: the checkpoint's density_scale */
/* the 12 accumulator planes of nf cells -> the 10 uint8 fit-field planes (pc ps t0..t5 sp w, nf apart), on the device */
void nn_fit_finalize(const float *acc, size_t nf, int h, uint8_t *a);
/* the same for a D x W x W window with interior planes [hz, D - hz) (predict --window-z slabs); nn_fit_accum = D W, hz halo */
void nn_fit_accum_d(const float *lg, const uint8_t *ct, int D, int W, int hz, int halo, int tz, int ty, int tx, const int box[3], int h, const int F[3],
                    int ch_recto, int ch_phase, int ch_normal, int ch_density, float dscale, int dcode, float *acc);
void nn_fit_accum_dm(const float *lg, const uint8_t *ct, const uint32_t *bits, int D, int W, int hz, int halo, int tz, int ty, int tx, const int box[3], int h, const int F[3],
                     int ch_recto, int ch_phase, int ch_normal, int ch_density, float dscale, int dcode, float *acc);   /* material from the bit plane bits (nn_pred_ctmask) when non-null */   /* dcode: train --density-code */
/* W (or U = W + dir theta / 2 pi when with_theta) on a grid: row a at box-local native z rows[a].z, voxel (b, c) at y0 + (b + 0.5)
   step, x0 + (c + 0.5) step; per row model a (z - za) blended with model b (z - zb) by t: (1 - t) W_a + t W_b. ms: device array of
   models (device pointers inside); out: device float nz x ny x nx */
typedef struct { int a, b; float z, t, za, zb; } fit_grid_row;
void nn_fit_grid(const fit_dev *ms, const fit_grid_row *rows, int nz, int ny, int nx, float y0, float x0, float step, int with_theta, float *out);

/* fit fields of a directory (ufsm band --fit-fields / predict --fit-fields: raw arrays; predict --fit-zarr: zarr arrays), field
   rows [fz0, fz1) (fz1 <= 0: to the end), pooled by pool per axis (1: none; pooled phase vectors are left uninformative, so
   pooled fields are for the gradient-only stages; 0: the geometry and the whole axis table only, no arrays). The slab's native origin (box-local) is (fz0 h, 0, 0); f.cy / f.cx are the
   slab's axis table (box-local y, x). */
typedef struct { fit_fields f; uint8_t *a[10]; float *cy, *cx; long long origin[3]; int shape[3]; int h; double um; } fit_fields_buf;   /* um: native voxel size (fields.json "um", default 2.4) */
int fit_fields_load(const char *dir, int fz0, int fz1, int pool, fit_fields_buf *b);
void fit_fields_free(fit_fields_buf *b);

/* a fit set: the global model plus z-slab models whose interiors [z0, z1) (box-local native z of the global box) partition it */
typedef struct { fit_model *global; int n; fit_model **slab; double *z0, *z1, blend; } fit_set;   /* blend: W blended linearly over +-blend around slab boundaries */
const fit_model *fit_set_model(const fit_set *s, double z, double *zoff);   /* the model for box-local z, and its z offset */
double fit_set_W(const fit_set *s, double z, double y, double x);
int fit_set_ray_root(const fit_set *s, double z, double theta, double k, double r_hint, double *r);   /* fit_ray_root of the set (global box, axis, pitch) */
int fit_set_save(const fit_set *s, const char *dir);   /* dir/global, dir/slabNNN, dir/set.json */
fit_set *fit_set_load(const char *dir);                 /* a set directory or a single model directory */
void fit_set_free(fit_set *s);
fit_set *fit_run_slabs(const char *dir, const fit_cfg *c, int slab, int overlap);

/* a fit set on the GPU (every model's knots, lattice and axis) for grid evaluation */
typedef struct fit_set_dev fit_set_dev;
fit_set_dev *fit_set_upload(const fit_set *s);
void fit_set_dev_free(fit_set_dev *g);
/* W (with_theta: U) of the set on the grid of n[0] x n[1] x n[2] points at box-local native (o + (i + 0.5) step) per axis, into the
   host float array out (row-major); evaluated on the GPU in z-row batches. Matches fit_set_W up to float rounding. */
int fit_set_grid(fit_set_dev *g, const fit_set *s, const double o[3], const int64_t n[3], double step, int with_theta, float *out);
/* the same on the CPU (no device); mask (optional, laid out as out): only points with a band code 1..252 there are evaluated (others 0) */
int fit_set_grid_cpu(const fit_set *s, const double o[3], const int64_t n[3], double step, int with_theta, const uint8_t *mask, float *out);
void fit_set_threads(int n);   /* the CPU backend's / grid's thread count (0: all cores) */
uint8_t *fit_field_array(const char *dir, const char *name, int pool, int shape[3]);   /* one field array, max-pooled (raw or zarr) */
