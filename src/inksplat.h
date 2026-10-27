/* ufsm ink-splat: published 2D ink labels (drawn on a segment's surface volume) mapped back onto the CT grid as a sparse 3D label
   zarr the trainer reads ("encoding": "prob254": bytes p * 254, 254 = ink, 0 = no ink, 255 = unknown).

   Conventions of the published surface volumes (vesuvius-challenge-open-data <scroll>/segments/<id>/surface-volumes, e.g. PHercParis4
   20230702185753: 109 layers, slice_step 1.0, canvas 36400 x 50600 for a 1820 x 2530 tifxyz at scale 0.05; the reader-v2-labels
   native volumes: 28 layers): canvas pixel (r, c) is the tifxyz grid point (u, v) = (c, r) x scale (grid units; one canvas pixel =
   one voxel of the mesh's volume), layer i sits at offset (i - (D - 1) / 2) voxels along N = du x dv (the unit normal of the
   bilinear grid, (z, y, x) components as ufsm render / ink-ext; "forward", where the external ink models agree). Label pyramids
   (OME, level k = 2^k canvas pixels) are mean-pooled: label pixel (r, c) of level k is canvas ((r + 0.5) 2^k - 0.5, ..).

   Splat: every supervised label pixel (sub-sampled to <= 0.45 output voxel), its surface point P and normal N, the band t in
   [a, b] layers (default -4 .. 4; --flip negates N) sampled at <= 0.45 output voxel: each sample's nearest output voxel (CT level L,
   voxel j covers level-0 voxels [j 2^L, (j + 1) 2^L)) collects (ink hits, no-ink hits, max ink p, max no-ink p), label p = byte / 255,
   ink = p >= 0.5. A hit voxel with CT >= iso (and CT > 0) becomes ink (max ink p) when ink hits >= thr x all hits, else no ink (max
   no-ink p); every other voxel stays unknown (255). Output: <out>/<um> on the CT level's full grid, 512^3 shards, only touched shards
   written (--merge: known values combine with an existing shard by max, unknown keeps the old value), coarser levels by known-max
   pooling (the sampler's occupancy index finds the labelled cells), the OME group, and <out>/segments/<name>.json (bounding box,
   1024-block origins in level-0 voxels, label area, voxel counts).

   Profile (--pred ROOT KEY --profile T): per supervised pixel (every --stride pixels) the CT and a 3D ink prediction sampled at
   t = -T .. T layers along the signed normal: per-t tables by label class, and the band [a, b] with the best ink IoU against
   prediction >= --pred-thr over papyrus (CT >= iso) samples. With --pred and a written splat, the written voxels are also compared
   with the prediction voxelwise (agreement, IoU). */
#pragma once
#include <stdint.h>

typedef struct {
    const char *out;            /* output group dir (nullptr / --no-write: profile only) */
    const char *name;           /* segment name (default: the tifxyz directory's name) */
    const char *surface;        /* tifxyz dir */
    const char *labels, *mask;  /* label / supervision images: .tif / .png, or a zarr array dir (v3 2-D, v2 3-D), local or https */
    int label_z, mask_z;        /* layer of a 3-D (depth, H, W) label array (-1: depth / 2) */
    double label_scale;         /* canvas pixels per label pixel (0: from the tifxyz grid size) */
    double label_off[2];        /* r, c added to a label pixel's canvas position (canvas pixels): labels republished with a shift */
    int crop[4];                /* label pixel window c0, r0, w, h (w <= 0: the whole raster) */
    const char *ct_root, *ct_key; double ct_um; int level; int iso;
    double band_a, band_b;      /* layers along the signed normal */
    double layer_vox;           /* level-0 voxels per layer (1) */
    int flip; double thr;       /* thr: ink-hit share needed for an ink voxel (0.5) */
    int merge, coarse;          /* coarse: levels written above the base by known-max pooling (-1: up to CT level 5) */
    int threads; const char *cache;
    const char *pred_root, *pred_key; int pred_level, pred_thr, pred_skip;   /* pred_skip: a prediction value left out of comparisons (255: another splat's unknown; -1 none) */
    int profile, stride;        /* profile half range (layers), pixel stride */
    const char *summary;        /* summary JSON path (default <out>/segments/<name>.json) */
    const char *preview;        /* JPEG of the label window (supervised no ink gray-blue, ink white) */
    int nexcl; int64_t excl[16][6];   /* --exclude z,y,x,nz,ny,nx (level-0 voxels, repeatable): output voxels touching a box are never written (held-out regions) */
} inksplat_cfg;

void inksplat_defaults(inksplat_cfg *c);
int inksplat_run(const inksplat_cfg *c);
int cmd_ink_splat(int argc, char **argv);
uint8_t *inksplat_label_window(const char *path, int64_t r0, int64_t c0, int64_t h, int64_t w, int64_t *H, int64_t *W);   /* a 2-D label window (0 outside) */
