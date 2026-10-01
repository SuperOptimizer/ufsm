/* Threaded patch sampler: draws (source, level, corner) tuples, decodes CT + teacher cubes, applies the
   48 cube symmetries and intensity jitter, and hands out ready batches from a ring buffer. */
#pragma once
#include "sources.h"
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
    float soft;              /* > 0: soft ridge target exp(-(d/soft)^2/2) around the surface (d = chamfer distance in level-0 voxels) */
} sample_cfg;

typedef struct {
    float *x;                /* B x 4 x P^3: z-scored CT, radial z (0), radial y, radial x */
    uint8_t *t;              /* B x NCH x P^3: target probability * 255 (from the label encoding) */
    uint8_t *m;              /* B x P^3: 1 where CT > 0 and the label is not ignore (loss mask) */
    uint8_t *w;              /* B x NCH: 1 if channel has a teacher in this sample */
    int16_t *src;            /* B: source index */
    int8_t *level;           /* B: CT level */
    int64_t (*corner)[3];    /* B: level-0 corner */
} batch;

typedef struct sampler sampler;

typedef struct { uint32_t *idx; size_t n; int lev; int64_t shape[3]; } occ_index;   /* coarse label cells that contain surface */
occ_index source_occupancy(source *s);
/* warm the CT chunk cache for the labelled cells of every pyramid source (levels 0..maxlev, a fraction of the cells) */
int sources_prefetch(sources *S, int maxlev, int nthreads, double fraction);
sampler *sampler_start(sources *S, const sample_cfg *cfg);
/* Block until a batch is ready; the pointer stays valid until sampler_release. */
batch *sampler_next(sampler *sp);
void sampler_release(sampler *sp, batch *b);
void sampler_stop(sampler *sp);
/* Stats: patches rejected / produced since start. */
void sampler_stats(const sampler *sp, uint64_t *produced, uint64_t *rejected);

sample_cfg sample_cfg_default(void);

/* Pool a uint8 cube by 2 (mean); out has edge n/2. Exposed for tests. */
void pool2_u8(const uint8_t *in, int n, uint8_t *out);
