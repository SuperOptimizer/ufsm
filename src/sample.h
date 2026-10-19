/* Threaded patch sampler: draws (source, level, corner) tuples, decodes CT + teacher cubes, applies the
   48 cube symmetries and intensity jitter, and hands out ready batches from a ring buffer. */
#pragma once
#include "sources.h"
#include "cover.h"
#include "sheet.h"
#include "scan_augment.h"
#include <stdint.h>

typedef struct {
    int P;                   /* patch edge (y, x; and z unless Pz) */
    int Pz;                  /* > 0: a slab window Pz x P x P (z thinner than y / x: the windings change in y / x); 0 = a P^3 cube.
                                Slabs: pyramid targets at levels >= 1 with a stored phase, z-keeping symmetries (augment 0, 3, 4),
                                no regions, cover, sheet, band, aux, cond, papyrus, dilation, CT / scanner / geometry augmentation */
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
    float axis_jitter_p, axis_jitter_deg;   /* its probability and angle cap (0: .1 and 2 degrees): an estimated umbilicus on other scrolls */
    int geometry_augment;  /* all 48 symmetries, including Z flips, with registered continuous warps */
    float rotate_degrees, rotate_p, elastic, elastic_p;
    float zoom_min, zoom_p;  /* zoom-in augmentation: with probability zoom_p the window shows a 1/z of the read cube, z uniform
                                in log over [zoom_min, 1] (zoom_min < 1), so the effective voxel size is level size * z;
                                needs geometry_augment; sources with "geometry": 0 and band-labelled native samples are never zoomed */
    float zoom_max, zoom_out_p; /* zoom-out augmentation: with probability zoom_out_p the window reads the next coarser level and shows
                                its central 1/2 z of it, z uniform in log over [1, zoom_max] (1 < zoom_max <= 2): an effective voxel size of
                                level size * z, from data 2x coarser (a coarser scan); needs geometry_augment, the dense-target tasks only */
    float label_morph, label_morph_p; /* signed soft-band distance offset; centreline preserved */
    scan_aug_cfg scan;       /* scanner-domain CT augmentation on the source grid (src/scan_augment.h); every p 0 = off */
    float affine_p, affine_aniso, affine_shear; /* anisotropic scale (log, per axis) + shear, magnitudes x scan.strength;
                                needs geometry_augment, same sources as the zoom */
    const sheet_dataset *sheet; /* borrowed sparse geometry, native-level B=1 task */
    int band;                /* 1: fill batch.band from each source's band target (task band_affinity, native level, no geometry warp) */
    float phase_near_um;     /* > 0 (stored phase, side 2): the phase target only within this distance (um) of a labelled wrap voxel
                                (target channel 0): the stored phase interpolates between published wraps, and across a missing wrap
                                the interpolated phase is off by up to half a turn */
    int iso;                 /* the void cutoff of sources without "iso" (papyrus = CT >= iso; 0: off for them) */
    int iso_void;            /* 1 (with a cutoff): void (0 < clean CT < cutoff) more than iso_pap voxels from papyrus is background (0) everywhere,
                                beyond the trust band too (it is never a sheet: the haze between sheets); CT 0 stays ignored */
    int iso_pap;             /* the void shell around papyrus left to the trust band (an unlabelled sheet's recto lies there), level voxels */
    int iso_lift;            /* > 0 (with a cutoff): label voxels with no papyrus within this many level voxels are ignored (rows lifted off
                                the sheets into the void); the trust band is then measured from the labels that remain */
    int shell;               /* > 0: background within this many level-0 voxels of a hard label is ignored (labels a voxel or two off the
                                sheet are not answered with "background" right beside them); before the soft ridge and the trust band */
    float dist_sigma;        /* > 0: target channel 1 = 254 exp(-(d / sigma)^2 / 2), d the distance to channel 0's hard labels (level-0 voxels,
                                um with soft_um): a wide distance ridge trained beside the sharp recto (task surface, no channel-1 teacher) */
    int side, side_air;      /* side: 1 fills batch.side (band.h SIDE_*: recto / verso side of the nearest labelled recto) from the same band
                                field, 2 fills it with the winding phase (band.h PHASE_*); voxels with CT <= side_air (and CT 0) are unknown */
    int aux;                 /* 1: a fifth input channel from the source's aux volume (src/sources.h), symmetry applied like the CT;
                                batch.x / x16 then hold 5 channels per sample */
    int pmask;               /* 1: fill batch.pm, the papyrus mask (CT > 0, symmetry applied) for heads whose loss covers all papyrus
                                (task unwrap: ink) */
    int cond_outside;        /* --cond-loss outside: batch.wind only where the condition does not reach (the net must count sheets there) */
    const struct papyrus_cfg *pap_head; /* task unwrap --pap-head (nullptr = off): batch.pap = the papyrus / void gap labels (src/papyrus.h
                                papyrus_local) of the clean CT before the scanner augmentation, symmetry applied */
    int cond;                /* 1 (train --cond): inputs 4-6 hold a condition drawn from the source's winding raster (cos, sin of
                                2 pi q' / 14, mask; q' = q + a random gauge k per sample, in a random pattern of the output frame) and
                                batch.wind the gauge-shifted winding for the targets (all unknown when no condition was drawn) */
    int phase_dense;         /* side 2: > 0 keeps a phase voxel whose windings lie more than phase_dense label voxels apart with probability
                                phase_dense / spacing (position hash): denser windings weigh more in the loss */
    const struct papyrus_cfg *papyrus; /* task papyrus (src/papyrus.h; nullptr = off): no target reads; channel 0 = papyrus labels from the
                                patch's clean CT at a random cutoff (before the scanner augmentation); batch.um then holds that cutoff in
                                input units ((t - mean) / sd, times the intensity jitter: the conditioning code) instead of a voxel size */
    int gpu_input;           /* train --gpu-input (needs xfmt and sample_gpu_input_ok): batch.x16 holds the CT channel only and batch.rad
                                each sample's radial record (nn.h nn_input_rad builds the 4 input channels on the device, bit for bit) */
    int pack_tm;             /* targets and mask as one code byte per voxel (batch.tm: SAMPLE_TM_* bits; t, m nullptr): every target value
                                0 or 255 and the mask 0 / 1 (hard two-class targets, e.g. the side encoding; a soft value aborts). The
                                trainer unpacks on the device (nn_unpack_tm): host batches 3 -> 1 byte per voxel */
    int wind4;               /* train --task unwrap2 (side 2, stored phase): batch.w4 / wcomp, the relative winding mod 4 of each window
                                (src/band.h band_wind4: the stored phase unwrapped over the window before the air test, its components
                                kept as separate gauges); warped samples pull it through the warp within one component */
    int wind4_step, wind4_min;   /* band_wind4's maxstep (phase codes; 0: 80 = 0.32 turn) and minsize (voxels; 0: 512) */
} sample_cfg;
/* the configurations whose every sample takes write_x (no warp, sheet prior, slab appearance plan, aux or condition channels): the
   radial channels then follow from the window's corner, symmetry and axis centres alone */
int sample_gpu_input_ok(const sample_cfg *c);

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
    uint8_t *pm;             /* B x P^3 papyrus mask (1 where CT > 0) when cfg.pmask */
    uint8_t *wind;           /* B x P^3 winding mod 14 + gauge (band.h band_winding encoding, 255 unknown) when cfg.cond */
    uint8_t *pap;            /* B x P^3 papyrus labels when cfg.pap_head: 0 void, 1 papyrus, 255 ignored */
    uint8_t *w4, *wcomp;     /* B x P^3 each when cfg.wind4: winding mod 4 (63 codes per turn, 255 unknown) and its component (0..14, 255 none) */
    void *rad;               /* cfg.gpu_input: B radial records of NN_RAD_REC(Pz) bytes (nn.h), x / x16 then one channel per sample */
    uint8_t *tm;             /* cfg.pack_tm: B x P^3 codes (bit 0: target 0 = 255, bit 1: target 1 = 255, bit 2: mask), t / m nullptr */
    int xch;                 /* input channels stored per sample in x / x16 (4, 5 with aux, 7 with cond, 1 with gpu_input) */
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
/* augmentation presets for the dense-target tasks (train --aug): "none" (symmetries + intensity jitter only, the old default) or
   "all" (every scanner-domain op, continuous geometry, zoom in and out, axis jitter; sample.c sample_aug_preset). Sets the fields only
   (explicit flags override them afterwards); 0 on success, -1 for an unknown name. */
int sample_aug_preset(sample_cfg *c, const char *name);
enum { SAMPLE_TM_T0 = 1, SAMPLE_TM_T1 = 2, SAMPLE_TM_M = 4 };
/* the packed code of n voxels from two target channels (each 0 or 255) and a 0 / 1 mask; out may be m (in place). -1 at a value
   that does not fit (a soft target or a mask weight), its index in *bad. Exposed for tests. */
int sample_pack_tm(const uint8_t *t0, const uint8_t *t1, const uint8_t *m, size_t n, uint8_t *out, size_t *bad);

/* Pool a uint8 cube by 2 (mean); out has edge n/2. Exposed for tests. */
void pool2_u8(const uint8_t *in, int n, uint8_t *out);
/* --cond: one sample's condition and gauge-shifted winding (src/sample.c cond_draw) with an rng seeded by seed; X: 7 x P^3 floats,
   inputs 4-6 written. Exposed for tests. */
void sample_cond_draw(uint64_t seed, int P, const uint8_t *w, uint8_t *wout, float *X, int outside);
/* --gpu-input: write_x of one D x P x P window twice with the same rng seed (jitter on): H4 the four 16-bit channels (4 x D P^2), H1
   the CT channel alone and rec its radial record (NN_RAD_REC(D) bytes). xfmt 1 fp16, 2 bf16. Exposed for tests (tests/test_gpu_input.c). */
void sample_write_x_test(const uint8_t *ctu, int D, int P, const int64_t o[3], float mean, float isd, int hasax, const float *cyz,
                         const float *cxz, const int perm[3], const int flip[3], int xfmt, uint64_t seed, uint16_t *H4, uint16_t *H1, void *rec);
/* --gpu-input with a continuous warp: write_spatial of one P^3 window (rotation degrees / elastic / zoom / affine from aug_seed as the
   sampler draws them) twice with the same rng seed: H4 the four 16-bit channels, H1 the CT channel alone and rec its radial record
   (NN_RAD_REC(P) bytes, with the warp); also T / M (NCH x P^3 / P^3) from the hard-target warp (nearest code) when hard. Returns the
   warp's active flag. Exposed for tests (tests/test_gpu_input.c). */
int sample_write_spatial_test(const uint8_t *ctu, const uint8_t *tgt, const uint8_t *mask, int P, const int64_t o[3], float mean, float isd,
                              int hasax, const float *cyz, const float *cxz, const int perm[3], const int flip[3], int xfmt, uint64_t seed,
                              uint64_t aug_seed, float zoom, int affine, int hard, uint16_t *H4, uint16_t *H1, void *rec, uint8_t *T, uint8_t *M);
