/* Umbilicus by sheet-normal voting (ufsm umbilicus, src/umbfind.c): a C port of villa's vc_gen_umbilicus
   (volume-cartographer/apps/src/vc_gen_umbilicus.cpp; align_and_extract_umbilicus and fit_umbilicus_along_z of
   core/src/normalgridtools.cpp, origin/main a7d4f88). Per xy slice the point the in-plane sheet normals point away from:
   - nsamp points drawn (with replacement, by weight) from the slice's (position, axial in-plane normal, weight) list;
   - optional un-crush (villa --uncrush): the section stretched round along the direction its normals crowd towards, by the
     cell-balanced normal tensor's eigenvalue ratio ^ (beta / 2) (skipped on sections spanning < half a turn or nearly round);
   - ncand random candidates c in the points' bounding box scored by sum cos^2(angle(p - c, n)); the best then hill-climbed (8
     neighbours, step halving) on sum cos^2 / max(near, |p - c|), villa's weighted sum (a finite maximum on partial sections).
   Along z: villa's robust LOWESS (tricube local-linear fits of x(z) and y(z) within +-half_width, 3 bisquare reweightings on
   6 x the median residual). Inputs here: the CT's 2-D structure tensor of each slice (umb_ct_points) or the nets' n n^T fit
   fields (src/fit.h planes t1 t2 t5 w). All positions are level-0 voxels (z, y, x as in the umbilicus json files). */
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct { double y, x, ny, nx, w; } umb_pt;   /* level-0 position, unit in-plane normal (sign-free), weight > 0 */

typedef struct {
    int nsamp, ncand;            /* villa: 10000 sampled segments, 1000 RANSAC candidates */
    double near;                 /* hill-climb weight floor 1 / max(near, d), level-0 voxels (villa: 100 grid units) */
    double step0, step_min;      /* hill-climb start / stop step, level-0 voxels (villa 1024 -> 1) */
    double uncrush;              /* villa --uncrush beta (0: off; 1.5 their measured setting) */
    int max_moves;               /* cap on hill-climb moves (villa: none; a walk off the section stops here) */
} umb_cfg;

/* defaults scaled to a level-0 voxel of um0 micrometres: near 0.94 mm, step0 9.6 mm, step_min 1 voxel (villa's 100 / 1024 / 1 grid
   units at 9.4 um) */
void umb_cfg_default(umb_cfg *c, double um0);

/* one slice: out = (y, x) level-0. 0 ok, -1 no usable points. *score: the refined weighted score; *inside: out within the points'
   bounding box (villa's "inside", here the section's own extent) */
int umb_estimate(const umb_pt *p, int n, const umb_cfg *c, uint64_t seed, double out[2], double *score, int *inside);

/* in-plane normals of one CT slice img (H x W, float, level-L voxels, 0 = masked out) from its 2-D structure tensor: gradients of the
   image smoothed by sig_d, the tensor smoothed by sig_t (both level-L pixels); every stride-th pixel with img > thr and coherence
   (l1 - l2) / (l1 + l2) >= min_coh becomes a point of weight coherence, at level-0 position i * s + (s - 1) / 2 (s = 2^L voxels per
   pixel). Also the slice's CT>thr centroid (cen, level-0; ufsm axis's estimator) and its pixel count. Returns the number of points
   (*out malloc'd), -1 on allocation failure. */
int umb_ct_points(const float *img, int H, int W, int s, double thr, double sig_d, double sig_t, int stride, double min_coh,
                  umb_pt **out, double cen[2], long long *ncen);

/* villa fit_umbilicus_along_z: robust local-linear fit through the estimates (z, y, x; any order) read at qz[0..nq); half_width in
   level-0 voxels; iters bisquare rounds (villa 3). A query whose window holds < 3 weighted estimates keeps the median of the
   estimates at exactly its z (counted in *nfall), else NaN. 0 on success. */
int umb_zfit(const double *z, const double *y, const double *x, int n, const double *qz, int nq, double half_width, int iters,
             double *oy, double *ox, int *nfall);

int cmd_umbilicus(int argc, char **argv);   /* ufsm umbilicus */
int cmd_axis_diff(int argc, char **argv);   /* ufsm axis-diff */
