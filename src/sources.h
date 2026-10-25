/* Training sources: a JSON file listing (CT pyramid, teacher targets, scroll axis) tuples.

   {"cache": "/vesuvius/ufsm/cache",
    "sources": [
      {"name": "PHerc0800", "root": "https://dl.ash2txt.org/community-uploads/forrest/volcomp",
       "ct": "PHerc0800/volumes/20250521135224-8.640um-1.2m-116keV-masked.zarr", "um": 8.64,
       "targets": {"recto": "PHerc0800/representations/labels/...zarr"           (same root)
                   or {"root": "/vesuvius/ufsm/gt/labels/x.zarr", "group": "."}   (own root)},
       "axis": "/vesuvius/ufsm/umbilicus/PHerc0800/umbilicus-full-resolution.json",   (optional)
       "holdout": [z, y, x, nz, ny, nx],                                              (optional, level-0 voxels)
         or [[z, y, x, nz, ny, nx], ...]: several boxes (the first is the validation box; training avoids all)
       "holdout_file": "holdouts.json"   (optional: ufsm wbench boxes {"boxes": [{"ct": KEY, "l0": [6 numbers]}...]}; the entries
         whose ct basename equals this source's ct basename are added to the holdout boxes)
       "weight": 1.0},
      {"name": "Paris4-recto", "root": "...", "ct": "PHercParis4/volumes/...zarr", "um": 2.4,
       "targets": {"recto": {"regions": "PHercParis4/representations/predictions/teacher_regions/recto-2.4um",
                             "size": 1024, "origins": [[z, y, x], ...]}},
       "axis": "...", "weight": 2.0}
    ]}

   A pyramid target has levels named by voxel size in um; level l of the CT (2^l x um) is matched to the
   target level with that voxel size. Optional per source: "min_level" / "max_level" (levels drawn), "geometry": 0 (no spatial
   augmentation), and per target "encoding": "binary" | "prob" | "prob254" (p * 254, 255 = unknown). A regions target is a set of single-level arrays at CT level 0,
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
    int max_level;    /* ... nor above this (-1: no limit); e.g. a teacher target whose coarser levels are only downsampled copies */
    int geometry;     /* 0: never spatially augmented (zoom / rotation / elastic) */
    int crush_fine;   /* "crush_fine": N: --aug-crush windows pull their CT from N levels finer of the same scan (a 2^N box mean per voxel);
                         -1 (absent): train --aug-crush-fine */
    int trust_band;   /* > 0: background labels are trusted only within this many level-0 voxels of an annotated surface (partial annotations) */
    int head;         /* train --heads: the recto head this source's recto target trains (0: output channel 0; k >= 1: channel 1 + k) */
    int iso;          /* > 0: the scan's void cutoff (papyrus = CT >= iso) for train --iso-void / --iso-lift (0: --iso's value) */
    z3 *ct[MAXLEV];         /* lazily opened per level */
    int ct_present[MAXLEV]; /* 1 if the level exists in the pyramid */
    int nlev;
    /* per channel: either a pyramid (levels keyed like the CT) or a regions set; each target may live in
       its own store ({"root": ..., "group": ...}), else the source's store is used */
    store *tgt_store[NCH];
    char *tgt_key[NCH];
    z3 *tgt[NCH][MAXLEV];
    int tgt_present[NCH][MAXLEV];
    int tgt_binary[NCH];      /* target bytes are 0/background, 255/surface; no ignore class */
    int tgt_min_level[NCH];   /* finest stored CT rung; binary targets can upsample finer requests */
    int tgt_side[NCH];        /* recto only, "encoding": "side": recto / verso side labels (src/band.h RV_*: phase-raster --pair 2), nearest-upsampled;
                                 the sampler turns them into recto-side / verso-side papyrus (channels 0 / 1) with the source's void cutoff */
    int tgt_votes[NCH];       /* "encoding": "side-votes" (tgt_side set too): bytes are src/votes.h vote codes, turned into side codes on
                                 read by tgt_vote_lut ("votes": {"rule": "unanimous" | "majority", "min": 1}); "soft": 1 keeps the codes and
                                 the sampler writes recto-side papyrus r / (r + v) (soft targets: not packable) */
    int tgt_votes_soft[NCH];
    uint8_t tgt_vote_lut[NCH][256];   /* hard: vote code -> RV_*; soft: vote code -> round(254 r / (r + v)), 255 below "min" votes */
    int tgt_erode[NCH], tgt_dilate[NCH];   /* binary targets: "erode" / "dilate" N voxels at the training level (bands of every teacher the same
                                              thickness: e.g. a thick pseudo-label band eroded to the published labels' width) */
    int tgt_prob[NCH];        /* "encoding": "prob": bytes are round(p * 255) (teacher probabilities, no ignore class); read as
                                 round(p * 254) and passed through without erode / dilate / soft / trust band */
    int tgt_prob254[NCH];     /* "encoding": "prob254" (tgt_prob set too): bytes already p * 254 with 255 = unknown (ufsm ink-splat), read as
                                 stored, so unknown stays ignored */
    regions *reg[NCH];
    z3 **reg_z[NCH];        /* lazily opened region arrays */
    z3 *reg_shared[NCH];    /* the shared array when reg[ch]->array is set */
    axis ax;                /* n == 0 when unknown */
    int64_t hold_o[3], hold_n[3];   /* optional held-out box (level-0 voxels); hold_n[0] == 0 when absent (= hold[0]: the validation box) */
    int64_t (*hold)[6]; int nhold;  /* all held-out boxes (z y x nz ny nx, level 0): training windows touch none of them */
    int64_t (*box)[6]; int nbox;    /* optional "boxes": level-0 boxes [z, y, x, nz, ny, nx] of local CT (task restore: windows inside one) */
    /* optional band target (task band_affinity, src/band.h): winding_mod14 raster array on the 2x label grid,
       {"band": {"root": ..., "key": "4.8", "radius": 80, "span": 75}} under "targets" */
    z3 *band;
    float band_radius, band_span;   /* native voxels */
    z3 *aux; int aux_level;   /* optional {"aux": {"root", "key", "level"}}: a uint8 input volume on the grid of CT level aux_level (e.g. the
                                 papyrus void fraction pooled from 2.4 um); with sample_cfg.aux it becomes a fifth input channel */
    z3 *phase;   /* optional: the band field's winding phase (band.h PHASE_*) precomputed on the same label grid (ufsm phase-raster),
                    {"band": {..., "phase": {"root": ..., "key": "4.8"}}}: windows read it instead of computing the band field;
                    "level": L (default 1, the label grid) when it is on the grid of CT level L (e.g. "9.6" pooled by ufsm downsample,
                    level 2): samples of level L read it directly, of level L - 1 upsampled 2x */
    int phase_only;                  /* band without "root" + a stored phase with "level": s->band aliases s->phase (never read as codes:
                                        task unwrap2 reads the stored phase only); any phase level, 0 included (pseudo labels on ~9 um scans) */
    int phase_level, phase_direct;   /* the CT level whose grid the stored phase is on (1: the 4.8 um label grid of 2.4 um scans);
                                        phase_direct: "level" given, so samples of that level read it directly */
    int phase_sign;   /* "phase": {..., "phase_sign": -1}: the stored phase (and "winding") rise inward, against the convention (phase
                         rising outward from the axis, fit.h W): windows read p -> (252 - p) mod 252 (unknown stays unknown); the
                         winding mod 4 targets unwrap the negated phase, so they follow. Default +1. (2026-10-19: the 10-16 multi-scroll
                         wind-seed sets with a wrongly voted spiral sense, e.g. Paris 4 June / July, 0500P2) */
    z3 *phase_ign;   /* optional "phase": {..., "ignore": {"root", "key"}}: a uint8 array on the stored phase's grid, nonzero = the phase there is not trained on (ufsm phase-conflict) */
    uint8_t *focus; int64_t focus_grid[3]; float focus_w;   /* optional "focus": {"map": RAW, "w": K} (ufsm phase-packing --map: per coarse occupancy cell 0 none, else
                                                              1 + 254 x packed share): the position draw takes a cell 1 + K x share times as often */
    z3 *wind;    /* optional, with a stored phase: the winding mod 14 turns on the same grid (ufsm phase-raster --winding; band.h
                    band_winding), {"band": {..., "phase": {...}, "winding": {"root": ..., "key": ...}}}: the --cond targets / condition */
    /* unlabelled scans (Phase U0, ufsm scanprep): */
    uint8_t *mask; int64_t mask_grid[3]; int mask_lev, mask_min;   /* optional "mask": {"map": RAW, "min": 2} (scanprep PREFIX.mask.raw, grid and "lev" in
                                                                     RAW.json): a source without a label occupancy index draws its windows around
                                                                     cells with code >= min (2: valid window centres, 1: any object cell) */
    int ct_only;      /* "ct_only": 1: the source is drawn without any target (every channel's w = 0, mask = CT > 0): label-free terms */
    struct srpair *srp;     /* optional "sr_pair": {"role": "train" | "val", "lut": 1, "cache": 0} (src/srpair.h): the source is a real SR pair directory ("root");
                               task restore then reads its boxes instead of the CT (no CT levels are opened) */
    int view, view_level; char *view_map;   /* optional "view": {"map": REG.json (omit: this scan itself), "level": L (default 0)}: the second-scan
                                               view of scan fusion (src/view.h; sample_cfg.view_P), opened by the sampler (view_h) */
    struct sview *view_h;
    int nxs; char *xs_map[4];   /* optional "xscan": [{"map": REG.json}, ...] (<= 4): registered partner scans of this scan (it is each map's coarse
                                   scan) for --aug-xscan: the window's CT swapped for a partner warped onto it, labels kept; opened by the sampler */
    struct sview *xs_h[4];
    float norm_c, norm_s;   /* optional "norm": {"center": c, "scale": s} (scanprep stats: median, scale): the CT input is (v - c) / s for this
                               source instead of the window's own z-score (norm_s 0: off, the default) */
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
z3 *source_tgt_for_level(source *s, int ch, int level, int *stored_level);
/* Return labels in the trainer's canonical 0/background,254/surface,255/ignore
   encoding. Binary masks are nearest-neighbour upsampled on their global voxel
   centres when shift>0; interior samples clamp to the last stored centre. */
int z3_read_label_grid(z3 *z, int binary, int shift, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads);
int source_read_target(source *s, int ch, int level, const int64_t o[3], const int64_t n[3], uint8_t *out, int nthreads);
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
