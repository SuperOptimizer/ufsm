/* ufsm faces: the recto and verso faces of the papyrus sheet each wrap (tifxyz) lies on, as two surfaces on the wrap's own grid.

   A wrap of ufsm wraps is the level set W = k of a smooth winding field: it follows the sheet but sits anywhere across it or in the
   gap next to it (10-14: ~half its samples in void). For every vertex the CT is sampled along the outward surface normal (trilinear,
   every quarter CT voxel, +- range) and cut into papyrus bands: profile extrema simplified by persistence (a dip shallower than
   max(min_prom, rel x the smaller peak) does not split a sheet: the two fibre layers stay one band), each surviving peak's band edges
   at the half level between the peak and the minimum on that side (a valley higher than the half level between the peak and the
   profile's void level is a contact between touching sheets: the edge is the valley). Per vertex the bands with thickness in
   [tmin, tmax] are candidates; the vertex takes the one nearest t = 0 (ties: the one whose inner face is nearer, so a wrap in a gap
   takes the sheet outward of it, W = k being meant as the inner face of sheet k), then the choice is made consistent over the
   surface: robust medians of the inner / outer offsets over a (2 smooth + 1)^2 vertex window, every vertex re-picks the band nearest
   them (within tol), twice; the picked offsets are median filtered and vertices without a band filled from their neighbours
   (push-pull), confidence 0.

   Convention (published-label side rule, ufsm-side-labels-2026-10-15, src/band.c band_field_side_pair): wraps mark recto faces,
   windings grow outward, the papyrus outward of wrap k (k .. k + 0.5) is labelled recto side and inward of wrap k + 1 verso side:
   the sheet body lies outward of its recto face, i.e. RECTO = the INNER face (towards the umbilicus), VERSO = the OUTER face.

   Outward is +-(the grid normal), one sign per wrap: by the umbilicus (radial component) when given, else from the wrap itself
   (vertices binned by z, the bin's centroid taken as the axis point: a turn or an arc of one is concave towards the axis). */
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct { float near, far, peak, prom, cin, cout; } fc_band;   /* along the outward normal (profile units), peak value, contrast over void,
                                                                        the CT just inside the inner / outer face */

/* the papyrus bands of one profile f[0..m) at t = t0 + k h: at most kmax, sorted by near. min_prom: absolute dip / contrast floor,
   rel: relative dip floor (x the smaller adjacent peak). Bands whose edge leaves the profile are dropped. Returns the count. */
int faces_bands(const float *f, int m, double t0, double h, double min_prom, double rel, fc_band *out, int kmax);

typedef struct {
    double range;            /* profile reach, CT voxels (each side) */
    double tmin, tmax;       /* plausible thickness, CT voxels */
    double min_prom, rel;    /* faces_bands */
    double tol;              /* re-pick tolerance, CT voxels */
    double inset;            /* each face moved this far into the sheet, CT voxels */
    int smooth;              /* consistency window radius (vertices) */
    int nthreads;
} faces_opt;

/* the CT the profiles read: sample (trilinear, CT voxel coordinates z, y, x; NaN outside). With a chunk cache: mark (thread safe) the
   box lo..hi a profile spans, fill() once the marks of a band of rows are in (decode), then the samples. mark / fill may be null. */
typedef struct {
    void *ctx;
    double (*sample)(void *ctx, double z, double y, double x);
    void (*mark)(void *ctx, const double lo[3], const double hi[3]);
    int (*fill)(void *ctx);
} faces_vol;

typedef struct {
    size_t nv, nband, npick0, npick, nfill;      /* valid vertices, with any band, with a consistent pick, filled */
    size_t ct_n, ct_orig, ct_rect, ct_vers;      /* CT >= ct_thr at the original vertex / recto / verso (over ct_n vertices) */
    size_t in_band;                      /* the original vertex inside its picked sheet [near, far] */
    double ct_thr; double thick_hist[64], thick_all[64]; /* picked thickness in 5 um bins (um); the nearest band's, any thickness */
    double shift_r, shift_v;             /* summed |recto shift|, |verso shift| (wrap voxels) */
    double fib[2][2];                    /* face texture (CT just inside recto 0 / verso 1): summed |d/d row|, |d/d column| between picked
                                            neighbours; on a wrap grid (rows ~ z) fibres along the turn (recto, horizontal) give d/d row > d/d column */
} faces_stats;

/* one wrap: X, Y, Z (w x h, wrap voxels, z < 0 invalid) -> recto RX.. and verso VX.. (same grid), conf (0: filled, else the band's
   contrast clamped to 1..255) and thick (um). f: CT voxels per wrap voxel (voxel centres map: c = (p + 0.5) f - 0.5); ct_um the CT
   voxel size; sign +-1 the outward orientation of the grid normal (faces_orient). Returns 0. */
int faces_wrap(const float *X, const float *Y, const float *Z, int w, int h, double f, double ct_um, int sign, const faces_opt *o,
               const faces_vol *V, float *RX, float *RY, float *RZ, float *VX, float *VY, float *VZ, uint8_t *conf, float *thick, faces_stats *st);

/* +1 when the grid normal (d/d column x d/d row, as z, y, x) points away from the axis on most vertices, else -1. axis_yx: per wrap
   z (wrap voxels, rounded, clamped to n_axis - 1) the axis (y, x) in wrap voxels, or null: z-binned centroids of the wrap itself. */
int faces_orient(const float *X, const float *Y, const float *Z, int w, int h, const double (*axis_yx)[2], int n_axis);

/* the grid normal of vertex i (central differences over +-2 vertices where valid, else +-1); 0 when undefined */
int faces_normal(const float *X, const float *Y, const float *Z, int w, int h, size_t i, double N[3]);
