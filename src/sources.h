/* Training sources: a JSON file listing (CT pyramid, teacher targets, scroll axis) tuples.

   {"cache": "/vesuvius/ufsm/cache",
    "sources": [
      {"name": "PHerc0800", "root": "https://dl.ash2txt.org/community-uploads/forrest/volcomp",
       "ct": "PHerc0800/volumes/20250521135224-8.640um-1.2m-116keV-masked.zarr", "um": 8.64,
       "targets": {"recto": "PHerc0800/representations/labels/...zarr"           (same root)
                   or {"root": "/vesuvius/ufsm/gt/labels/x.zarr", "group": "."}   (own root)},
       "axis": "/vesuvius/usrm/umbilicus/PHerc0800/umbilicus-full-resolution.json",   (optional)
       "holdout": [z, y, x, nz, ny, nx],                                              (optional, level-0 voxels)
       "weight": 1.0},
      {"name": "Paris4-recto", "root": "...", "ct": "PHercParis4/volumes/...zarr", "um": 2.4,
       "targets": {"recto": {"regions": "PHercParis4/representations/predictions/teacher_regions/recto-2.4um",
                             "size": 1024, "origins": [[z, y, x], ...]}},
       "axis": "...", "weight": 2.0}
    ]}

   A pyramid target has levels named by voxel size in um; level l of the CT (2^l x um) is matched to the
   target level with that voxel size. A regions target is a set of single-level arrays at CT level 0,
   each covering [origin, origin + size). Channels: 0 = recto, 1 = sheet (m7 whole-sheet band). */
#pragma once
#include "store.h"
#include "zarr3.h"
#include <stdint.h>

enum { CH_RECTO = 0, CH_SHEET = 1, NCH = 2 };
#define MAXLEV 10

typedef struct {
    int n;
    double *z, *y, *x;      /* control points, level-0 voxels, sorted by z */
} axis;

typedef struct {
    char *dir;              /* group key of the regions directory (one array per region) ... */
    char *array;            /* ... or one shared array key holding every region at its origin */
    int size;               /* default region edge, level-0 voxels */
    int n;
    int64_t (*origin)[3];
    int *rsize;             /* per-region edge (origins given as [z,y,x,size]), else size */
} regions;

typedef struct {
    char *name;
    store *s;
    char *ct_key;
    double um;              /* level-0 voxel size */
    double weight;
    int min_level;    /* training never draws levels below this (e.g. 1 for a 1.1 um scan whose level 0 is finer than the others) */
    int trust_band;   /* > 0: background labels are trusted only within this many level-0 voxels of an annotated surface (partial annotations) */
    z3 *ct[MAXLEV];         /* lazily opened per level */
    int ct_present[MAXLEV]; /* 1 if the level exists in the pyramid */
    int nlev;
    /* per channel: either a pyramid (levels keyed like the CT) or a regions set; each target may live in
       its own store ({"root": ..., "group": ...}), else the source's store is used */
    store *tgt_store[NCH];
    char *tgt_key[NCH];
    z3 *tgt[NCH][MAXLEV];
    int tgt_present[NCH][MAXLEV];
    regions *reg[NCH];
    z3 **reg_z[NCH];        /* lazily opened region arrays */
    z3 *reg_shared[NCH];    /* the shared array when reg[ch]->array is set */
    axis ax;                /* n == 0 when unknown */
    int64_t hold_o[3], hold_n[3];   /* optional held-out box (level-0 voxels); hold_n[0] == 0 when absent */
} source;

typedef struct {
    int n;
    source *src;
    char *cache;
} sources;

sources *sources_load(const char *json_path);
void sources_set_cache(const char *dir);   /* overrides the file's "cache"; call before sources_load */
void sources_free(sources *S);

/* Lazily open CT / target arrays. Return nullptr if that level does not exist. */
z3 *source_ct(source *s, int level);
z3 *source_tgt(source *s, int ch, int level);          /* pyramid target only */
z3 *source_region(source *s, int ch, int i);           /* regions target only */
int source_region_shared(const source *s, int ch);      /* 1 if region reads use absolute (array) coordinates */

/* Axis (y, x) at level-0 coordinate z by linear interpolation, clamped at the ends. */
void axis_at(const axis *a, double z, double *y, double *x);
int axis_load(axis *a, const char *path);              /* umbilicus JSON with control_points[{x,y,z}] */

/* Open level `level` of a pyramid group: by voxel size um0 * 2^level when the group has OME multiscales,
   else by the integer level name. cache may be nullptr. */
z3 *pyramid_open_level(store *st, const char *group, int level, double um0, const char *cache);

/* Rung of a voxel size: 0.6 * 2^k um -> k, rounded. */
int rung_of_um(double um);
void sources_open_all(sources *S, int nthreads);   /* eager parallel open of every level / target store */
