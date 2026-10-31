/* Winding fitter (ufsm fit): a smooth "unwrapped radius in turns" U(x) fitted to dense fit fields (winding phase, sign-free
   normal n n^T, sheet spacing, material weight), so that the winding coordinate W = U - dir theta / 2 pi has its integer level
   sets on the recto faces. theta = atan2(y - cy(z), x - cx(z)) around the umbilicus; dir = +1 when the radius grows with the
   unwrapped theta along a sheet (PHercParis4). Every loss reads cos / sin (2 pi W) and grad W, both smooth away from the axis, so
   the branch cut of theta only shows in floor(W). U = r / pitch(z) + offset(z) + delta(x): pitch and offset on z knots (linear),
   delta on a dense trilinear lattice refined coarse to fine. Coordinates are native voxels relative to the field box origin. */
#pragma once
#include <stddef.h>
#include <stdint.h>
#include "coreline.h"

typedef struct {
    int D, H, W;                    /* field grid */
    double h;                       /* field voxel edge in native voxels; voxel (i, j, k) is centred at h (i + 0.5, j + 0.5, k + 0.5) */
    const uint8_t *pc, *ps;         /* phase vector: 128 + 127 * (confidence * cos / sin (2 pi phase)), 128 = no information */
    const uint8_t *t[6];            /* n n^T x255: n_z^2, n_y^2, n_x^2, 0.5 + n_z n_y, 0.5 + n_z n_x, 0.5 + n_y n_x */
    const uint8_t *sp;              /* sheet spacing in units of h / 8 (0 = unknown), i.e. 8 / (h rho); native voxels at h = 8 */
    const uint8_t *w;               /* material fraction x255 (0 = no data) */
    const float *cy, *cx;           /* umbilicus per native z of the box, D * h + 1 entries (box-local coordinates) */
    const uint8_t *wd[4];           /* optional (task unwrap2 fields, nullptr otherwise): the winding mod 4 head as relative planes dz dy dx
                                       (byte b: W(x + w4_d e_a) - W(x) = b / 64 turns mod 4, measured inside one inference window) and dq
                                       (x255 confidence, 0 = none) */
    int w4_d;                       /* their offset d in native voxels */
    const uint8_t *recto;           /* optional recto probability x255 on the field grid (plane r): high probabilities anchor integer W */
    const uint8_t *ct;              /* optional mean CT per field cell (plane ct, predict --fit-ct / ufsm ct-fields; 0 = no CT) */
    const uint8_t *cs;              /* optional CT sheet spacing (plane cs: papyrus peaks counted along the network normal), coded as sp
                                       (units of h / 8 native voxels, 0 = unknown) */
} fit_fields;

/* loss parts (fit_model loss, the backends' parts arrays): total (data terms + smoothness), phase, normal, density, non-degeneracy,
   phase gradient, smoothness (lattice), winding mod 4, recto anchors, CT papyrus / void parity, CT sheet spacing */
enum { FIT_P_TOTAL, FIT_P_PHASE, FIT_P_NORMAL, FIT_P_DENSITY, FIT_P_NONDEG, FIT_P_PGRAD, FIT_P_SMOOTH, FIT_P_WIND4, FIT_P_RECTO, FIT_P_CT, FIT_P_CSPACE, FIT_NPARTS };

typedef struct {
    int dir;                        /* +1 / -1, 0 = choose the more phase-coherent */
    int knot_dz;                    /* native voxels between pitch / offset knots */
    int nstage, lattice[8], steps[8];   /* delta lattice spacing (native voxels) and Adam steps per stage */
    float lr, lr_knot;              /* Adam learning rates for delta and the knots */
    float w_phase, w_normal, w_density, w_nondeg, w_smooth, w_decay;
    float w_wind4;                  /* the winding mod 4 term (fields with dz dy dx dq planes; analytic loss, phase stages, ramped like the phase) */
    float w_recto;                  /* recto anchors r^2 (1 - cos 2 pi W) (fields with an r plane; phase stages, ramped like the phase; 0: off) */
    int phase_stage;                /* stages before it fit the gradient terms only (normal, density: no integer ambiguity, they
                                       integrate the turn count); the phase term (periodic) refines from this stage on */
    float phase_ramp;               /* phase_stage: phase weight from phase_ramp * w_phase to w_phase */
    int unwrap_voxels;              /* phase-stage init: 1 voxel-level quality-guided unwrapping (default), 0 lattice nodes, -1 none
                                       (the phase stages start from the gradient stages' W) */
    double uw_split;                /* > 0: two-level voxel unwrapping: tiles flood their voxels of confidence >= uw_split in parallel,
                                       the pieces are stitched by border votes, then the serial flood does the rest (0: serial only) */
    int uw_tile;                    /* its tile edge (field voxels) */
    int uw_msf;                     /* 1: the voxel unwrapping as a parallel maximum spanning forest (src/fit.c uw_msf: the flood's tree up to
                                       the order among equal confidence levels; deterministic, threads) instead of the serial flood (0) */
    double uw_min_conf;             /* the unwrapping's lowest phase confidence (0: 0.3) */
    double uw_pdv;                  /* > 0: the unwrapping quality (flood order, node weights) times exp(-pdv / uw_pdv^2), pdv the
                                       phase derivative variance (turns^2): dislocated / noisy phase is flooded last */
    int uw_repair;                  /* > 0: local slip repair after the voxel flood (tile edge in field voxels; 0 off): the flooded voxels
                                       of each tile split into pieces whose integers agree, every touching voxel pair of two pieces votes
                                       for their integer offset (weight the lower confidence), and the pieces' integer shifts minimise the
                                       summed weighted L1 disagreement (src/l1pot.h: loops that do not close are cut where cheapest,
                                       instead of where the flood's spanning tree closed them) */
    float uw_rep_w4;                /* its evidence: the winding mod 4 planes (unwrap2 fields) as offset votes between voxels w4_d apart,
                                       weight x the lower confidence x dq (0 off) */
    float uw_rep_geo;               /* the gradient stages' W as offset votes between voxels uw_rep_gd field voxels apart (the
                                       density / normal integration across low-confidence bands), weight x the lower confidence (0 off) */
    int uw_rep_gd;                  /* their distance (field voxels) */
    float uw_rep_cs;                /* the CT sheet spacing cs (fields with a cs plane) as offset votes like uw_rep_geo: the turns between
                                       the voxels counted from cs along the model normal (0 off) */
    float uw_rep_qpow;              /* the repair's vote weight: (lower confidence)^uw_rep_qpow (1: linear) */
    float uw_rep_stretch;           /* > 0: votes x exp(-(log2 s)^2 / (2 uw_rep_stretch^2)) at both voxels, s = |grad phase| x sp (turns per
                                       sheet): a phase that lost a sheet is stretched (s ~0.5-0.7 where it disagrees with the published wraps
                                       on Paris 4, 1.05 where it agrees), so the cut prefers it (0 off) */
    int uw_rep_sr;                  /* the stretch's box radius (field voxels: (2r+1)^3; 0 the voxel's own central differences) */
    int uw_rep_off;                 /* the repair tiles' grid offset (field voxels on every axis, mod uw_repair; 0: tiles from the origin) */
    int uw_rep_cons;                /* > 1: the consensus repair: the voxel-wise majority integers of this many repairs on offset tile grids */
    int uw_rep_geo_init;            /* 1: the uw_rep_geo votes from the init model's W (fit --init-model, e.g. a global coarse fit) */
    float uw_rep_seed;              /* with a seed model: each piece holding seeded voxels is tied to its seeded integers by an anchor vote of
                                       weight uw_rep_seed x its seeded voxels' confidence sum (so the repair cannot move the carried count; 0 free) */
    float phase_stretch;            /* gate the phase confidence by the phase stretch s (uwr_stretch: |grad phase| x spacing): x exp(-log2(s)^2 / 2 S^2)
                                       before the fit (0 off), so where the phase skips or doubles sheets the density sets the turn count */
    int phase_stretch_r;            /* its box radius (field voxels) */
    const char *uw_rep_dump;        /* write PREFIX.shift.raw (the repair's integer shift + 128 per field voxel, 0 not flooded) and
                                       PREFIX.seam.raw (per field voxel the neighbours whose integers disagree, before | after << 4, | 8 where
                                       the phase is stretched: uw_rep_stretch's weight < 1/2) */
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
    double seed_off, seed_z, seed_z0;   /* seed_z0: only box-local z >= seed_z0 is seeded (fit_cfg_default -1e30; fit --seed-model / --seed-z) */
    int batch;                      /* analytic: field voxels per step (about; 0 = all) */
    float core;                     /* no data terms within core * pitch of the axis */
    float w_ct;                     /* CT papyrus / void parity (fields with a ct plane; src/fit_terms.h fit_ct_term; phase stages, ramped
                                       like the phase; 0: off) */
    float ct_pow, ct_phi;           /* its weight x (1 - phase confidence)^ct_pow; the papyrus phase (frac W at the sheet's CT peak) */
    double ct_radius_um;            /* the parity's local CT normalisation: mean / sd over a cube of this half-edge (um) */
    float ct_floor;                 /* the parity weight's floor (1 - conf)^p >= ct_floor: CT counts even under a confident phase */
    float w_cspacing;               /* the density term against the CT sheet spacing cs (fields with a cs plane; where sp is unknown or
                                       disagrees with cs by more than cs_tol in log; every stage; 0: off) */
    float cs_tol;                   /* |log(sp / cs)| below which the network spacing is kept (the cs term off) */
    int ct_flood;                   /* 1: voxels below uw_min_conf join the phase stage's voxel unwrapping with the CT's own phase (the
                                       parity demodulated along the model's normal) when w_ct > 0 */
    const struct emb_vol *emb;      /* slip repair evidence from sheet-identity embeddings (src/fit_emb.h; nullptr off): votes between pieces
                                       uw_rep_emb_d native voxels apart along the sheet (the model's tangents), the integer the embedding
                                       picks for the same sheet (rmax uw_rep_emb_rmax turns), weight uw_rep_emb x lower confidence x
                                       min(1, margin / uw_rep_emb_m0) (cosine margin best - second); the sheet centres at W = n + emb_phi */
    float uw_rep_emb, uw_rep_emb_m0, emb_phi;
    int uw_rep_emb_d[4], uw_rep_emb_nd, uw_rep_emb_rmax, uw_rep_emb_sub;
    double emb_zoff;                /* native z of this fit's fields in the embedding's box coordinates (the slab driver: the slab's first field row) */
    /* fibre pairs (fit --fibers FILE --w-fib W, ufsm fiber-export): polylines in the fields' box-local native coordinates (z y x); points
       16..128 native voxels apart along one fibre whose network phase stays within half a turn along it should have W(b) - W(a) = that
       phase change (a loss term in every fit stage, src/fit.c fib_pairs); fib_zoff: the fields' first native z in the fibres' frame */
    const float *fib; const int *fib_off; int fib_n; float w_fib; double fib_zoff;
    struct fit_stream *stream;      /* GPU fits (fit_fields_load_stream): the fields already on the device and the gradient stages' pooled
                                       fields made while loading (the host fields then carry no normal planes); nullptr: fit_run_dir uploads
                                       and pools the host fields itself */
    /* coarse-to-fine (fit --ref-model DIR: a global coarse model or set, e.g. a whole-scroll coarse fit or count-field --model; one
       integer gauge for every block): the voxel unwrapping's integers are tied to it. ref_bias = the circular mean of U_ref - u over the
       unwrappable voxels (the sub-turn offset between the ref and the phase). ref_snap 1: every voxel takes round(U_ref - bias - u) as its
       integer (no flood: the integers are the ref's, the phase only the sub-turn position). uw_rep_ref W > 0 (with uw_repair): each repair
       piece is also tied to the ref by votes round(U_ref - bias - U) from every other voxel per axis, weight W x confidence (an L1 median
       over the piece: the ref decides the integer offsets the phase votes leave open; the base integer at the flood's seed from the ref;
       the pieces start at their votes' weighted median; slow: one max flow ~10 min on a slab). ref_snap 2 with uw_rep_ref > 0: the flood,
       then each repair piece (uw_repair^3 tiles) shifted to that weighted median and no L1 moves (a per-piece snap: the phase keeps the
       integers inside a piece, the ref sets each piece's offset) */
    const struct fit_set *ref; double ref_off[3];   /* point p of this box is p + ref_off in the ref's box */
    int ref_snap; float uw_rep_ref;
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
    double loss[FIT_NPARTS];        /* final loss parts (FIT_P_*): total, phase, normal, density, nondegeneracy, phase gradient, smoothness,
                                       winding mod 4, recto, CT parity, CT spacing */
} fit_model;

fit_cfg fit_cfg_default(void);
int fit_scale_L(int L, double f);   /* a lattice / knot spacing scaled by f (>= 2) */
fit_model *fit_run(const fit_fields *f, const fit_cfg *c);
void fit_free(fit_model *m);
double fit_W(const fit_model *m, double z, double y, double x);   /* winding coordinate at a box-local native point (host) */
int fit_save(const fit_model *m, const char *dir);
/* a fit-field directory (the input of ufsm fit): pc ps t0..t5 sp w .raw (shape[0] x shape[1] x shape[2] uint8) and fields.json
   (origin, shape, h, the axis per native z of the box in box-local coordinates, na = shape[0] h + 1 entries) */
int fit_fields_save(const char *dir, const long long origin[3], const int shape[3], int h, double um, uint8_t *const a[10], const float *cy, const float *cx, int na);   /* um: the native voxel size */
/* the same plus, with w4_d > 0, the 4 winding mod 4 planes a[10..13] (dz dy dx dq; fields.json "w4_d") */
int fit_fields_save_w4(const char *dir, const long long origin[3], const int shape[3], int h, double um, uint8_t *const a[14], int w4_d, const float *cy, const float *cx, int na);
/* the same plus an optional recto plane (r.raw, fields.json "recto": true; nullptr: "recto": false) */
int fit_fields_save_ex(const char *dir, const long long origin[3], const int shape[3], int h, double um, uint8_t *const a[14], int w4_d, const uint8_t *recto,
                       const float *cy, const float *cx, int na);
/* dir/model.json, delta.f32 (lattice), axis.f32 (cy then cx) */
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
    const uint8_t *wd; float w_w4, w4_d;    /* nn_fit_loss2: the winding mod 4 planes (4 back to back: dz dy dx dq; nullptr none), weight, offset
                                               (native voxels); its loss in parts[FIT_P_WIND4] */
    const uint8_t *recto; float w_recto;    /* recto anchors (both kernels; nullptr none); loss in parts[FIT_P_RECTO] */
    const uint8_t *ctm; float w_ct, ct_pow, ct_phi, ct_floor;   /* CT parity 128 + 127 m (m > 0 papyrus, < 0 void; fit_ct_parity), weight,
                                               (1 - conf)^ct_pow (>= ct_floor), papyrus phase; nullptr none; loss in parts[FIT_P_CT] */
    const uint8_t *cs; float w_cs, cs_tol;  /* CT sheet spacing (sp units), weight, log tolerance to sp; loss in parts[FIT_P_CSPACE] */
} fit_dev;
/* the data terms over every field voxel: loss parts (double[FIT_NPARTS], FIT_P_*, accumulated; the smoothness slot untouched)
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
/* the recto plane r (x255, the max recto probability per cell: accumulator plane 11) into a (nf bytes), on the device */
void nn_fit_finalize_recto(const float *acc, size_t nf, uint8_t *a);
/* task unwrap2 checkpoints: the winding mod 4 head (channels ch_w4, ch_w4 + 1) of a window pooled into 9 relative accumulator planes
   (per axis the sums of p(v + d e_a) conj p(v) and their count, p the decoded cos / sin; d <= halo, hz), and those into the 4 uint8
   planes dz dy dx dq (nf apart) */
void nn_fit_accum_w4(const float *lg, const uint8_t *ct, const uint32_t *bits, int D, int W, int hz, int halo, int tz, int ty, int tx, const int box[3], int h, const int F[3],
                     int ch_w4, int d, float *acc);
void nn_fit_finalize_w4(const float *acc, size_t nf, uint8_t *a);
/* predict --fit-ct: the CT sums of a window's cells (the uint8 CT window, before the forward overwrites it) into acc_ct (accumulator
   plane 12), the CT sheet spacing of the cells centred in the window (network normal and spacing from the logits, the CT along the
   normal: native voxels) into acc_cs (plane 13), and both finalized into the uint8 ct / cs planes (acc: 14 planes of nf cells) */
void nn_fit_accum_ct(const uint8_t *ct, int D, int W, int hz, int halo, int tz, int ty, int tx, const int box[3], int h, const int F[3], float *acc_ct);
void nn_fit_accum_cs(const float *lg, const uint8_t *ct, int D, int W, int hz, int halo, int tz, int ty, int tx, const int box[3], int h, const int F[3],
                     int ch_normal, int ch_density, float dscale, int dcode, float um, float *acc_cs);
void nn_fit_finalize_ct(const float *acc, size_t nf, int h, uint8_t *oct, uint8_t *ocs);
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
typedef struct { fit_fields f; uint8_t *a[17]; float *cy, *cx; long long origin[3]; int shape[3]; int h; double um; } fit_fields_buf;   /* um: native voxel size (fields.json "um", default 2.4);
                                                       a: pc ps t0..t5 sp w, then (unpooled loads, when present) dz dy dx dq (w4_d > 0), r (recto),
                                                       ct cs (CT planes: zarr arrays beside the others, or box-shaped ct.raw / cs.raw of ufsm ct-fields) */
int fit_fields_load(const char *dir, int fz0, int fz1, int pool, fit_fields_buf *b);
void fit_fields_free(fit_fields_buf *b);
void fit_fields_ct_planes(int mask);   /* which CT planes later fit_fields_load calls read (bit 0 ct, bit 1 cs; default both) */
void fit_fields_skip_planes(int mask);   /* optional planes later fit_fields_load calls leave out (bit 0 the recto plane r, bit 1 the winding mod 4 planes dz dy dx dq, bit 2 the normals, bit 3 sp (unpooled loads); default none): RAM and reads for fits whose terms do not read them */
int fit_fields_zchunk(const char *dir, long long *zoff);   /* the z rows per chunk of the fields' arrays (zarr; 128 for raw) and the box's z offset in them (*zoff; 0 raw): loads of chunk-aligned rows decode each chunk once */
/* Streamed fields for a GPU fit (fit_cfg stream): the whole box (unpooled) with the six normal planes read plane by plane (chunk rows)
   straight to the device field array (the fit's 10 B per cell: pc ps t0..t5 sp w), the host buffer b holding the planes the CPU reads
   (pc ps sp w and the optional planes; b->a[2..7] null), and the pooled fields of the gradient stages (pooling factors fps[0..nfp)) made
   on the device (nn_fit_pool), bit for bit fit_run_dir's own pooling, kept there. Host memory ~4 B per cell instead of 10.
   Returns 0, or -1 (then nothing is held). */
typedef struct fit_stream {
    uint8_t *dfield; size_t n;           /* device (or managed) planes, n cells each */
    int nfp, fp[8]; uint8_t *dpool[8]; int pD[8], pH[8], pW[8];   /* device pooled planes per pooling factor (10 planes back to back) */
    unsigned long long *dcnt;            /* device: per pooling factor its cells with material (pooled w != 0) */
    int drop_pooled;                     /* 1: fit_run_dir frees the pooled planes at the phase stage (a single fitted direction) */
} fit_stream;
int fit_fields_load_stream(const char *dir, fit_fields_buf *b, fit_stream *s, const int *fps, int nfp, void (*gpu_ready)(void));   /* gpu_ready (or null): called after the host planes load, before the first device call */
void fit_stream_free(fit_stream *s);
/* fit_pool (gradient stages: no phase gradient, no CT spacing) of the device field planes dfield (10 planes of D x H x W, cell size h) by fp into
   the 10 device planes out (back to back, pooled box ceil(D / fp) x ceil(H / fp) x ceil(W / fp)), byte-identical to the host pooling, and
   the count of its cells with material added to the device counter *cnt (src/nn_fit.cu; asynchronous) */
void nn_fit_pool(const uint8_t *dfield, int D, int H, int W, double h, int fp, uint8_t *out, unsigned long long *cnt);
/* host -> device uploads on a thread through pinned staging (src/nn_fit.cu): put queues a copy of the pageable range src (job id
   returned), wait blocks until that job's source has been read (reusable), close until every job is on the device, and frees */
typedef struct nn_fit_up nn_fit_up;
nn_fit_up *nn_fit_up_open(void);
long nn_fit_up_put(nn_fit_up *u, void *dst, const void *src, size_t n);
void nn_fit_up_wait(nn_fit_up *u, long id);
void nn_fit_up_close(nn_fit_up *u);
int fit_stream_verify(const fit_stream *s, const fit_fields *f);   /* the device planes and pooled sets against whole host fields f (fit_fields_load, all planes): mismatches (0: identical) */
/* the gradient stages' pooling factors of a fit config at field cell h (fit_run_dir's rule); returns their count (at most 8) */
int fit_stream_fps(const fit_cfg *c, int h, int fps[8]);

/* a fit set: the global model plus z-slab models whose interiors [z0, z1) (box-local native z of the global box) partition it */
typedef struct fit_set { fit_model *global; int n; fit_model **slab; double *z0, *z1, blend; } fit_set;   /* blend: W blended linearly over +-blend around slab boundaries */
const fit_model *fit_set_model(const fit_set *s, double z, double *zoff);   /* the model for box-local z, and its z offset */
double fit_set_W(const fit_set *s, double z, double y, double x);
int fit_set_ray_root(const fit_set *s, double z, double theta, double k, double r_hint, double *r);   /* fit_ray_root of the set (global box, axis, pitch) */
/* the same where only roots with ua <= r <= ub are wanted (others: -1, possibly without their search; a kept root is the same) */
int fit_set_ray_root_b(const fit_set *s, double z, double theta, double k, double r_hint, double ua, double ub, double *r);
/* fit_set_ray_root_b for many rays at one z (a row context: the slab choice and the z parts once; the same roots) */
typedef struct fit_set_row fit_set_row;
fit_set_row *fit_set_row_new(const fit_set *s, double z);
int fit_set_row_root_b(fit_set_row *w, double theta, double k, double r_hint, double ua, double ub, double *r);
void fit_set_row_free(fit_set_row *w);
/* grad W of the set at a box-local native point by central differences over +-e (theta continued across its cut, so the gradient
   is that of the smooth U - dir theta / 2 pi); returns |grad W| (turns per native voxel) */
double fit_set_gradW(const fit_set *s, double z, double y, double x, double e, double g[3]);
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
/* fn(ctx, a0, a1, t) on contiguous ranges of [0, n) over the fit threads (fit_set_threads / UFSM_FIT_THREADS / the cores, at most 64 and n);
   returns the thread count used (fit_par_threads(n): the count without running) */
int fit_par(int n, void (*fn)(void *ctx, int a0, int a1, int t), void *ctx);
int fit_par_threads(int n);
/* fit_fields_load plus: noread (the arrays allocated, nothing read: a pooled box to fill through `also`); also (pool 1 loads, rows
   [fz0, fz1) starting on a multiple of also_pool and ending on one or at the box end): each block of also_pool rows, masked, pooled into
   `also` (a noread pooled box of the whole box) exactly as its own pooled load would (fit --slab: the prior's loads fill the global fit's
   pooled fields) */
int fit_fields_load_ex(const char *dir, int fz0, int fz1, int pool, fit_fields_buf *b, int noread, fit_fields_buf *also, int also_pool, fit_fields_buf *prev);
/*   prev (pool 1, may be null; always consumed: freed): a previous pool-1 load of the same directory whose rows overlap the start of
     [fz0, fz1) (fit --slab: consecutive slabs share 2 x overlap rows): those rows are moved, not read again */
/* cl_mask_block (src/coreline.h) over rows [z0, z0 + nz) on fit_set_threads threads (one row per task; the same arrays) */
void fit_core_mask(const coreline *C, int side, const long long org[3], double h, double um, int z0, int nz, int H, int W, uint8_t *pc, uint8_t *ps, uint8_t *w);
uint8_t *fit_field_array(const char *dir, const char *name, int pool, int shape[3]);   /* one field array, max-pooled (raw or zarr) */

/* CT in the fitter (src/fit_ct.c). The papyrus / void parity of the mean CT per cell: m = tanh((ct - mu) / sd), mu and sd over a cube of
   half-edge r cells around the cell (cells with ct > 0), coded 128 + 127 m (128: no CT). Local, so no global CT cutoff. */
void fit_ct_parity(const uint8_t *ct, int D, int H, int W, int r, uint8_t *out);
/* the CT's own phase per field voxel for the voxel unwrapping: the parity demodulated along the model's normal over +-K cells at its
   density (cs where known, else the model's |grad W|): frac W = phi_pap + arg / 2 pi, confidence |sum| / sum |weights| (coded as pc / ps) */
void fit_ct_phasor(const fit_model *m, const fit_fields *f, const uint8_t *ctm, float phi_pap, uint8_t *opc, uint8_t *ops);
/* the sheet spacing from a CT profile of n samples step apart (native voxels) along a normal: the papyrus peaks (local maxima above the
   profile's mean by >= prom x its sd, separated by deeper minima) -> (last - first) / (peaks - 1); 0 below 2 peaks */
double fit_ct_profile_spacing(const float *p, int n, double step, double prom);
/* ufsm ct-fields CTROOT KEY FIELDS-DIR [--ct-um U] [--radius-um 150] [--threads 8]: ct.raw / cs.raw beside a field directory */
int cmd_ct_fields(int argc, char **argv);

/* fit-join --global (src/fitjoin.c): integer gauges S_k of n slabs minimising sum_k U_k(S_k) + sum_e C_e(S_b - S_a) over an overlap graph.
   A cost table: costs at the listed integers v[0..nv) (ascending), `other` for every integer not listed (the robust cap). Unary k may be
   empty (nv 0, other 0: no ref cells). The solve: candidates S[k] +- rc around the given S (in: the start, out: the result), exact dynamic
   programming over the edges between consecutive slabs (b = a + 1), then a local search over every edge (single slabs and every suffix
   k.. shifted by +-1, +-2) until no move lowers the energy. Returns the final energy. */
typedef struct { const long *v; const double *c; int nv; double other; } fj_table;
typedef struct { int a, b; fj_table t; } fj_edge;
double fj_table_cost(const fj_table *t, long v);
double fit_join_solve(int n, const fj_table *unary, int ne, const fj_edge *e, int rc, long *S);
