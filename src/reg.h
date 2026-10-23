/* Scan-to-scan registration (ufsm reg): two CT scans of the same scroll, a "coarse" (fixed) grid and a "fine" (moving) scan,
   related by a base affine A (fine level-0 voxels -> coarse level-0 voxels, points x y z, the convention of the published
   scrollprize transforms: transformation_matrix 3x4 applied to (x, y, z, 1) in each volume's own native voxel units) plus a smooth
   displacement field D on the coarse grid:   fine(c) = A^-1 (c + D(c)).
   Voxel centres sit at integer coordinates; level L of an OME pyramid (2^L mean pooling) has p_L = (p_0 + 0.5) / 2^L - 0.5.
   Per box: the fine scan is resampled onto the coarse level grid (the fine pyramid level closest to the coarse voxel size,
   2^3 supersampling of the coarse voxel footprint when the fine voxel is much smaller), both band-passed, and the integer shift
   maximising the normalised cross-correlation is refined to sub-voxel by re-resampling at the current shift and a quadratic fit of
   the 3^3 NCC neighbourhood. reg fit: robust affine correction + a regularised trilinear grid field over the box shifts. */
#pragma once
#include <stdint.h>
#include <stddef.h>

typedef struct { double m[3][4]; } raff;   /* q = M p + t (p, q: x y z) */
raff raff_ident(void);
raff raff_inv(raff a);
raff raff_mul(raff a, raff b);             /* p -> a(b(p)) */
static inline void raff_apply(const raff *a, const double p[3], double q[3]) {
    for (int i = 0; i < 3; i++) q[i] = a->m[i][0] * p[0] + a->m[i][1] * p[1] + a->m[i][2] * p[2] + a->m[i][3];
}
double raff_det(const raff *a);

/* displacement grid: nodes n[0] (z) x n[1] (y) x n[2] (x); node (k, j, i) at coarse L0 xyz o + h (i, j, k); d[node * 3 + c] the x y z
   displacement in coarse L0 voxels. Trilinear inside, clamped to the edge nodes outside. */
typedef struct { int n[3]; double o[3], h; float *d; } rfield;
void rfield_eval(const rfield *f, const double p[3], double out[3]);

typedef struct { raff A, Ainv; const rfield *D; } rmap;   /* A: fine L0 -> coarse L0 */
void rmap_c2f(const rmap *m, const double c[3], double f[3]);
void rmap_f2c(const rmap *m, const double f[3], double c[3]);   /* fixed-point inverse of rmap_c2f */

/* a volume: OME-zarr pyramid on a store, or in-memory levels (tests). Level l voxel = um * 2^l. */
typedef struct rvol {
    double um; int nl; int64_t shape[10][3];   /* zyx per level */
    void *z[10];                               /* z3 * per level */
    uint8_t *mem[10];                          /* in-memory levels (tests) */
    char name[256];
} rvol;
rvol *rvol_open(const char *root, const char *key, const char *cache, double um);   /* um <= 0: from the OME metadata */
rvol *rvol_mem(double um, int nl, const int64_t shape0[3], uint8_t *level0);        /* builds mean-pooled levels from level0 (owned) */
int rvol_read(rvol *v, int l, const int64_t o[3], const int64_t n[3], uint8_t *out);  /* zyx box at level l; outside = 0 */
int rvol_read_nt(rvol *v, int l, const int64_t o[3], const int64_t n[3], uint8_t *out, int nt);
void rvol_close(rvol *v);
int rvol_level_for(const rvol *v, double um);   /* the coarsest level whose voxel is <= um * 1.05 (0 if none) */

/* uint8 block of a level: origin o, dims n (zyx); trilinear at level coords x y z, 0 outside */
typedef struct { const uint8_t *v; int64_t o[3], n[3]; } rblk;
float rblk_sample(const rblk *b, double x, double y, double z);

/* resample the fine block (level lf) onto n (zyx) points of the coarse level-lc grid: point (k, j, i) at level-lc xyz org + (i, j, k) + sh;
   ss 1 (point) or 2 (2^3 sub-samples over the coarse voxel footprint) */
void reg_resample(const rblk *fb, const rmap *m, int lc, int lf, const double org[3], const int n[3], const double sh[3], int ss, float *out);

/* the fine level-lf block under the coarse level-lc points org - 1 - smax .. org + n + smax (org xyz, n zyx); malloc'd, nullptr with *err set
   (2 outside the fine volume, 5 read error, 6 over 96M voxels) */
uint8_t *reg_read_fine(rvol *fv, int lf, const rmap *m, int lc, const double org[3], const int n[3], double smax, rblk *fb, int *err);

enum { REG_NCC = 0, REG_DOG = 1, REG_GRAD = 2 };
void reg_gauss(float *v, const int n[3], double sigma);   /* separable, clamped edges, in place */
void reg_filter(float *v, const int n[3], int metric, double hp, double lp);

/* NCC of C (dims n) against G (dims n + 2R) at every integer shift s in [-R, R]^3: C(p) vs G(p + R + s); map[((sz + R) * w + sy + R) * w + sx + R],
   w = 2R + 1. Shifts with no G variance get -1. */
void reg_ncc_map(const float *C, const int n[3], const float *G, int R, float *map);
/* sub-voxel peak of a 3^3 neighbourhood f[(dz + 1) * 9 + (dy + 1) * 3 + dx + 1]: d (x y z) = -H^-1 g, clamped to [-1, 1];
   curv = smallest eigenvalue of -H (the peak's sharpness, NCC per voxel^2). Returns 0 when -H is positive definite. */
int reg_quad_peak(const float f[27], double d[3], double *curv);

typedef struct {
    rvol *cv, *fv; const rmap *m;
    int lc, lf, B, R, ss, metric, iters;
    double hp, lp, minfrac;
} rmeas;
typedef struct {
    double c[3];            /* box centre, coarse L0 xyz */
    double D0[3];           /* the map's displacement at c */
    double s[3];            /* measured correction (coarse L0 voxels, xyz): fine(c) = A^-1 (c + D0 + s) */
    float ncc, ncc0, second, curv, cstd, gstd, cfrac, gfrac;   /* ncc0: NCC at s = 0 (the prior map) */
    int status;             /* 0 ok, 1 coarse mask, 2 fine coverage, 3 flat (no contrast), 4 peak at the search edge, 5 read error, 6 too large */
} rbox;
int reg_measure(const rmeas *q, rbox *b);   /* returns b->status */
extern const char *const reg_status_name[7];

/* fits over box results: points c (n x 3), values v (n x 3), weights w (n; 0 = unused).
   reg_fit_affine: v ~ M c + t, Huber IRLS (k in voxels), returns the 3x4 in M; reg_fit_field: v ~ field (trilinear grid of spacing h over the
   bbox of the used points, membrane regulariser lam (relative to the mean data weight per node) + mu |u|^2, Huber IRLS). */
int reg_fit_affine(int n, const double *c, const double *v, const double *w, double huber, double M[3][4]);
int reg_fit_field(int n, const double *c, const double *v, const double *w, double h, double lam, double huber, int margin, rfield *out);

/* from-scratch affine (src/reg_init.c): scroll axes from mask centroids, cylinder resampling at ~W um, masked NCC over the rotation about
   the axis and the offset along it (all 4 h / phi orientations when all). A: fine L0 xyz -> coarse L0 xyz. */
typedef struct { double ncc, rot_deg; int hflip, pflip, n; double orient_ncc[4]; } reg_init_info;
int reg_init_affine(rvol *cv, rvol *fv, double W, double rmin, int nt, int all, raff *A, reg_init_info *info);

/* certification of a registered pair (src/reg_render.c): <pair>/final/residuals.jsonl used fit boxes with |v - pred| <= tol (fixed voxels)
   are good, <pair>/eval_after.jsonl boxes not ok / NCC < 0.3 / |s| > tol are failing; a fixed-frame point is certified when a good box lies
   within max(r_vox, 1.5 x their median spacing) and no failing box within r_vox / 2. */
typedef struct { int n; double *c; } ptset;
typedef struct { ptset good, bad; double r2, rb2; double cell; int nh; int *head, *next; int *bhead, *bnext; double o[3]; } rcover;
int rcover_load(rcover *cv, const char *pair, double r_vox, double tol);
int rcover_ok(const rcover *cv, const double p[3]);
