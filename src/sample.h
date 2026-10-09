/* Threaded patch sampler: draws (source, level, corner) tuples, decodes CT + teacher cubes, applies the
   48 cube symmetries and intensity jitter, and hands out ready batches from a ring buffer. */
#pragma once
#include "sources.h"
#include "cover.h"
#include "sheet.h"
#include "scan_augment.h"
#include <stdint.h>

typedef struct {
    int P;                   /* patch edge */
    int B;                   /* batch size */
    int nworkers;
    int nbuf;                /* ring depth (batches) */
    uint64_t seed;
    double level_p[MAXLEV];  /* relative probability of each CT level (rung offset); 0 = never */
    double min_fg;           /* reject a patch whose CT nonzero fraction is below this */
    double empty_keep;       /* probability of keeping a patch whose target is all zero */
    int augment;             /* 0 = none, 1 = all 48 cube symmetries + intensity, 2 = the 24 proper rotations + intensity */
    int holdout;             /* 0 = train: never draw patches touching a source's holdout box; 1 = validation: draw only inside holdout boxes */
    int dilate;              /* > 0: dilate the surface band of pyramid targets by this many level-0 voxels (curriculum for thin targets) */
    int erode;               /* 0 or 1: native-grid face-neighbour erosion of binary targets before softening */
    float soft;              /* > 0: soft ridge target exp(-(d/soft)^2/2) around the surface (d = chamfer distance in level-0 voxels) */
    int soft_um;             /* 1: soft (and its annealed values) is in um, the same physical width for every source and level */
    int snap;                /* align training windows to the CT chunk grid (3x fewer chunks decoded per window) */
    int deterministic;       /* 1: batch j holds samples j*B .. j*B+B-1, each drawn with its own rng seeded by (seed, index), and
                                batches come out in order: runs with the same seed see the same data regardless of thread timing */
    int xfmt;                /* also fill batch.x16: 0 = no (x only), 1 = fp16, 2 = bf16 (the trainer uploads x16 straight into the network input) */
    const cover_plan *cover; /* borrowed immutable finite plan; B=1, native resolution */
    uint64_t cover_start;    /* committed tile cursor at resume */
    int ct_augment;         /* conservative reconstructed-CT appearance augmentation */
    float symmetry_p;       /* chance of drawing a symmetry when augment enables geometry */
    float axis_jitter;      /* max auxiliary axis error in native voxels, p=.1, angle capped at 2 degrees */
    int geometry_augment;  /* all 48 symmetries, including Z flips, with registered continuous warps */
    float rotate_degrees, rotate_p, elastic, elastic_p;
    float zoom_min, zoom_p;  /* zoom-in augmentation: with probability zoom_p the window shows a 1/z of the read cube, z uniform
                                in log over [zoom_min, 1] (zoom_min < 1), so the effective voxel size is level size * z;
                                needs geometry_augment; sources with "geometry": 0 and band-labelled native samples are never zoomed */
    float label_morph, label_morph_p; /* signed soft-band distance offset; centreline preserved */
    scan_aug_cfg scan;       /* scanner-domain CT augmentation on the source grid (src/scan_augment.h); every p 0 = off */
    float affine_p, affine_aniso, affine_shear; /* anisotropic scale (log, per axis) + shear, magnitudes x scan.strength;
                                needs geometry_augment, same sources as the zoom */
    const sheet_dataset *sheet; /* borrowed sparse geometry, native-level B=1 task */
    int band;                /* 1: fill batch.band from each source's band target (task band_affinity, native level, no geometry warp) */
    int side, side_air;      /* side: 1 fills batch.side (band.h SIDE_*: recto / verso side of the nearest labelled recto) from the same band
                                field, 2 fills it with the winding phase (band.h PHASE_*); voxels with CT <= side_air (and CT 0) are unknown */
} sample_cfg;

typedef struct {
    float *x;                /* B x 4 x P^3: z-scored CT, radial z (0), radial y, radial x; nullptr when cfg.xfmt */
    uint16_t *x16;           /* the same in 16 bits when cfg.xfmt (then the only copy), else nullptr */
    uint8_t *t;              /* B x NCH x P^3: target probability * 255 (from the label encoding) */
    uint8_t *m;              /* B x P^3: 1 where CT > 0 and the label is not ignore (loss mask) */
    uint8_t *w;              /* B x NCH: 1 if channel has a teacher in this sample */
    int16_t *src;            /* B: source index */
    int8_t *level;           /* B: CT level */
    float *um;               /* B: effective voxel size of the input (source um * 2^level * zoom) */
    int64_t (*corner)[3];    /* B: level-0 corner */
    sheet_batch **sheet;     /* B sparse records; null for legacy task */
    uint8_t *band;           /* B x P^3 band per voxel (src/band.h: 1/18-turn steps mod 252, 255 unknown) when cfg.band */
    uint8_t *side;           /* B x P^3 side (src/band.h SIDE_*) or winding phase (PHASE_*) per voxel when cfg.side */
} batch;

typedef struct sampler sampler;

typedef struct { uint32_t *idx; size_t n; int lev; int64_t shape[3]; } occ_index;   /* coarse label cells that contain surface */
occ_index source_occupancy(source *s);
/* warm the CT chunk cache for the labelled cells of every pyramid source (levels 0..maxlev, a fraction of the cells) */
int sources_prefetch(sources *S, int maxlev, int nthreads, double fraction);
sampler *sampler_start(sources *S, const sample_cfg *cfg);
/* Block until a batch is ready; the pointer stays valid until sampler_release. */
batch *sampler_next(sampler *sp);
int sampler_failed(const sampler *sp);   /* read failure or exhausted draw retries; stopped samplers never publish partial batches */
void sampler_release(sampler *sp, batch *b);
void sampler_stop(sampler *sp);
/* Stats: patches rejected / produced since start. */
void sampler_stats(const sampler *sp, uint64_t *produced, uint64_t *rejected);
void sampler_prof_print(const sampler *sp);   /* env UFSM_SAMPLER_PROF=1: per-stage cpu ms per patch */
void sampler_set_soft(sampler *sp, float sigma);   /* change the soft-target sigma while running (annealing); patches already drawn keep theirs */

sample_cfg sample_cfg_default(void);

/* Pool a uint8 cube by 2 (mean); out has edge n/2. Exposed for tests. */
void pool2_u8(const uint8_t *in, int n, uint8_t *out);
