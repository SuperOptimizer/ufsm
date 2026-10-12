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
