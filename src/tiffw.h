/* Minimal TIFF writer: classic little-endian TIFF, one page, one strip per 64 rows, uncompressed or deflate (zlib), uint8 or
   float32 single-sample images (the counterpart of src/tiff.h, which reads them back). And the tifxyz surface format of VC3D /
   villa (x.tif, y.tif, z.tif float32 grids of native voxel coordinates, -1 = invalid vertex, meta.json). */
#pragma once
#include <stddef.h>
#include <stdint.h>

/* fmt 1 = uint8, 3 = float32; data row-major w x h. Returns 0 on success. */
int tiffw_write(const char *path, int w, int h, int fmt, const void *data, int deflate);

/* a tifxyz directory: x/y/z (w x h, row-major, -1 where invalid), step = native voxels between grid vertices (meta.json
   scale [1/step, 1/step]), voxel_um the scan's voxel size, winding the wrap's id (< 0: none), source a free-text provenance.
   meta.json: format tifxyz, type seg, uuid (the directory's name), scale, bbox [[xmin, ymin, zmin], [xmax, ymax, zmax]],
   area_vx2 (valid quads x step^2), area_cm2, voxel_size_um, winding, source. */
int tifxyz_write(const char *dir, int w, int h, const float *x, const float *y, const float *z, double step, double voxel_um, int winding,
                 const char *source);

/* Surface-compressor (.sfc) wraps (user 10-21: per-box wraps stored as .sfc, read directly). src/tfx.c installs the hooks when it is
   linked (the ufsm binary); without it they stay null and only plain tifxyz directories exist.
   A wrap path P is sfc-backed when P names a .sfc file, or P.sfc exists (P itself absent): then <P>/x.tif, y.tif, z.tif and meta.json
   read from it (decoded on the fly, invalid vertices -1, no decoded copy on disk).
   tifxyz_set_sfc(err): tifxyz_write writes <dir>.sfc instead of the directory (joint XYZ, Euclidean tolerance err voxels; 0 = off). */
extern float *(*tfx_axis_hook)(const char *tifpath, int *w, int *h);
extern char *(*tfx_meta_hook)(const char *metapath);
extern int (*tfx_write_hook)(const char *sfcpath, int w, int h, const float *x, const float *y, const float *z, const char *meta, double err);
void tifxyz_set_sfc(double err);
/* an axis image <wrap>/{x,y,z}.tif: the file, else from the wrap's .sfc (null when neither) */
float *tfx_axis_read(const char *tifpath, int *w, int *h, float *(*plain)(const char *, int *, int *));
/* <wrap>/meta.json opened for reading: the file, else the .sfc's metadata (fmemopen) */
#include <stdio.h>
FILE *tfx_meta_fopen(const char *metapath);
/* the whole meta.json (plain or sfc-backed) as a NUL-terminated malloc string, nullptr when missing (published tifxyz metas run to
   13 KB+: fixed 4 KB buffers missed "scale" past it) */
char *tfx_meta_read(const char *metapath);
/* a wrap's meta.json or x.tif path exists (plain or sfc-backed) */
int tfx_exists(const char *path);
