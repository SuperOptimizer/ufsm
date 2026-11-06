/* Shared box helpers of ufsm patch-solve / trace-prep / patch-score (src/patchsolve.c, src/tracebox.c): box-local axis geometry,
   the radial ridge of a sheet probability, uint8 box reads (.raw or zarr) and float32 tifxyz component reads. */
#pragma once
#include "sources.h"
#include <stdint.h>

typedef struct {
    axis ax; double s; int64_t o[3]; double t0;   /* axis, its voxel -> grid scale, box origin, reference angle */
    int par; double ny, nx, ty, tx, cy, cx;       /* ray frame: 0 radial from the axis; 1 parallel rays along the unit normal (ny, nx) through the
                                                     box-local centre (cy, cx), tangent (ty, tx) = (nx, -ny) */
} ps_geo;
/* the ray frame of a point: radial (r from the axis, u = polar angle about t0) or parallel (r = along n, u = along t, voxels), and back */
void ps_frame(const ps_geo *G, double z, double y, double x, double *r, double *u);
void ps_frame_pt(const ps_geo *G, double z, double r, double u, double *y, double *x);
void ps_ray_dir(const ps_geo *G, double z, double y, double x, double *uy, double *ux);   /* unit ray direction at a point */
int ps_frame_load(ps_geo *G, const char *path);    /* sheet-trace frame.txt: "radial" or "par ny nx cy cx"; 0 ok */
int ps_frame_save(const ps_geo *G, const char *path);
void ps_axis(const ps_geo *G, double z, double *ay, double *ax);           /* box-local axis (y, x) at box-local z */
void ps_polar(const ps_geo *G, double z, double y, double x, double *r, double *th);   /* th in (t0 - pi, t0 + pi] */
int ps_geo_init(ps_geo *G, const char *axis_json, double scale, const int64_t o[3], const int64_t n[3]);   /* t0 = angle of the box centre */
/* radial non-max suppression: rg = p >= 64 && p >= p(r + 1) && p > p(r - 1) (bilinear along the direction from the axis), threaded over z */
void ps_ridge(const uint8_t *p, uint8_t *rg, const int64_t n[3], const ps_geo *G, int nt);
uint8_t *ps_box(const char *path, const char *key, const char *cache, const int64_t o[3], const int64_t n[3], int nt);   /* .raw or zarr root + key */
float *ps_tif(const char *path, int *w, int *h);   /* a float32 single-sample TIFF */
