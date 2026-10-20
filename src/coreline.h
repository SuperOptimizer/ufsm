/* Core line of a crushed scroll (ufsm coreline): the flattened umbilicus as a polyline per z-slab, found in the network fit
   fields, and its use by the local box fits.
   - detect: per slab of cells the phase-gradient direction Im(conj(m) grad m) of the phase vector m = (pc, ps) (pointing where
     the winding grows, recto -> verso outward) is pooled into bins and smoothed. In a flattened scroll it points away from the
     collapsed core on both sides, so across the cross-section (lines along the minor axis of the material) its minor-axis
     component flips sign once, at the core: per line the split that maximises -sum_{before} g + sum_{after} g. Lines whose
     split is weak (contrast) or sits at the material's edge (a side with < min_side of the line's material: the racetrack's
     end caps, where the outward direction runs along the major axis, also seen as a one-signed major-axis component) are dropped; the longest run of kept lines, median
     filtered, is the slab's core polyline.
   - signed distance / nearest point: + on the left of the polyline direction (normal n = (t_x, -t_y) in (y, x)); z between slab
     centres blends the two slabs' values linearly (slabs oriented consistently by the detector).
   - box fits: fields.json "core_line" (path) and "core_side" (+1 / -1): fit_fields_load / fit_field_array zero the material
     (and blank the phase) of the cells on the other side of the core, so a box that straddles the core is fitted as two half
     boxes, each with its pseudo-axis behind the core (ufsm coreline box writes the per-z axis table): W grows away from the
     core on both sides. Units: the file's points and z are in voxels of "um" (2.399: the scan's full resolution), continuous
     coordinates (voxel i spans [i, i + 1)), global. */
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct { double z0, z1, zc; int n; double *y, *x; } cl_slab;   /* z range and centre, the polyline (um voxels) */
typedef struct { int ns; cl_slab *s; double um; } coreline;

int cl_load(coreline *c, const char *path);   /* 0 on success */
int cl_save(const coreline *c, const char *path, const char *extra_json);   /* extra_json: more top-level members (or nullptr) */
void cl_free(coreline *c);
/* signed distance (um voxels) of the point (z, y, x) (um voxels) to the core line; q: the nearest core point (y, x), n: the unit
   normal there pointing to the + side (either may be null). 0 slabs: returns 0. */
double cl_sdist(const coreline *c, double z, double y, double x, double q[2], double n[2]);
/* the same against one polyline */
double cl_poly_sdist(const double *y, const double *x, int n, double py, double px, double q[2], double nrm[2]);

/* detection on one slab: per-bin sums of the phase gradient (gy, gx) and of the material (w), H x W bins */
typedef struct {
    double sigma;          /* gradient smoothing (bins) */
    double wthr;           /* material mask: smoothed mean material fraction >= wthr */
    double min_contrast;   /* kept lines: best split score / sum |g| */
    double min_side;       /* kept lines: each side holds >= this share of the line's material */
    double min_width;      /* kept lines: material along the line (bins) */
    double dt, ds;         /* line spacing along the major axis, sample step along a line (bins) */
    int median;            /* median filter half width (lines) */
    double max_along;      /* kept lines: |sum of the major-axis component| / sum |g| below this (the end caps point along the major axis) */
} cl_params;
cl_params cl_params_default(void);
typedef struct { int n; double *y, *x; double ma[2], contrast, width; int lines, kept; } cl_line;   /* polyline in bin coordinates (continuous) */
int cl_detect(const float *gy, const float *gx, const float *w, int H, int W, const cl_params *p, cl_line *out);   /* 0: found */
void cl_line_free(cl_line *l);
/* accumulate one z row of cells (uint8 pc, ps, w; H x W cells) into bins of bin x bin cells: gy, gx += Im(conj(m) dm/dy, dx)
   (central differences, m = (pc - 128 + i (ps - 128)) / 127), w += w / 255; bins Hb = ceil(H / bin) x Wb */
void cl_accum_row(const uint8_t *pc, const uint8_t *ps, const uint8_t *w, int H, int W, int bin, float *gy, float *gx, float *wb);

/* box fits: blank the cells of rows [z0, z0 + nz) of a cell box (origin org in native voxels of um_native, cells of h native
   voxels, H x W cells per row) that lie on the other side of the core line (side * sdist < 0): w = 0, pc = ps = 128 (any of
   them may be null). Returns the number of cells blanked. */
size_t cl_mask_block(const coreline *c, int side, const long long org[3], double h, double um_native, int z0, int nz, int H, int W,
                     uint8_t *pc, uint8_t *ps, uint8_t *w);
/* the smoothed core frame of a box (ufsm coreline box --axis const): the mean over the box's z rows (na native rows from org, every
   16th) of the core point nearest the box centre (cyg, cxg: global native) and of the normal of a straight line fitted to the
   nearest slab's polyline points within R core-line voxels of it (oriented as the polyline's normal there); f = native um / core um.
   q, n: core-line voxels / unit (y, x). */
void cl_box_frame(const coreline *c, const long long org[3], int na, double cyg, double cxg, double f, double R, double q[2], double n[2]);
/* ufsm coreline check (the former core/check/check.py): the core-line sanity render of one field-cell z row zc. Inputs: ct the
   CT slice at 9.6 um (ctw x cth, shown 2 x 2 averaged = the 19.2 um cells), pc / ps / w the 8 field rows zc .. zc + 7 (8 x N x N
   cells). The phase gradient g = sum over the rows of Im(conj(m) grad m) (m = (pc - 128 + i (ps - 128)) / 127, numpy.gradient),
   scipy-Gaussian smoothed (sigma 3 cells; the material mean / 255 sigma 2, > 0.15 = material), against the gradient of the
   distance to the core line of the slab whose z is nearest zc + 4 (in cells: the line's points / 8): a = g . grad |d|, > 0 where W
   grows away from the core. Over the material's bounding box + 20 cells: pos / neg the material cells with a > 0 / < 0, pn / nn
   those within 26 cells (0.5 mm) of the line; rgb (w x h) CT | away-ness (green away, red towards), the slab's line yellow, its
   neighbours' cyan, a caption. numpy float32 / float64 arithmetic as check.py's. ks the slab index; y1 / x1 as check.py printed
   them (the arrays clipped to N). 0 on success (-1: no material, -2: allocation). */
typedef struct { int ks, y0, y1, x0, x1; long long pos, neg, pn, nn; int w, h; uint8_t *rgb; } cl_check_res;
int cl_check(const coreline *c, int zc, const uint8_t *ct, int ctw, int cth, const uint8_t *pc, const uint8_t *ps, const uint8_t *w, int N,
             int nthreads, cl_check_res *o);
/* scipy.ndimage.gaussian_filter(a, sigma) of a float32 H x W image (truncate 4, mode reflect, axis 0 then 1), in place */
void scipy_gaussian_f32(float *a, int H, int W, double sigma);
int cmd_coreline_check(int argc, char **argv);   /* src/coreline_check.c (dispatched by main.c) */
int cmd_coreline(int argc, char **argv);
