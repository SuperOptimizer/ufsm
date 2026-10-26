/* Scanner-domain augmentation of the uint8 CT cube on its source grid (z = scan axis, the axis centre known per z),
   after the targets are built and before the cube symmetry / continuous warp, so every op is then oriented randomly
   like the data. Each op has its own probability; strength scales every magnitude (0 = identity, 1 = default, 2 =
   aggressive). All probabilities 0 (the default) draws nothing from the sampler stream. Labels are untouched except:
   seam shifts CT, labels, ignore and the axis together; dropped slices / cutouts are ignored when drop_ignore; voxels
   the op zeroes (chunk dropouts, simulated air masking) lose supervision through the CT > 0 mask. */
#pragma once
#include <stddef.h>
#include <stdint.h>

enum { SA_SEAM, SA_AIR, SA_LOWRES, SA_BLUR, SA_SHARPEN, SA_BIAS, SA_RING, SA_CUPPING, SA_SLICE, SA_NOISE, SA_CNOISE,
       SA_TONE, SA_QUANT, SA_DROPSLICE, SA_CUTOUT, SA_CHUNK, SA_PAGANIN, SA_GIN, SA_HIST, SA_MOTION, SA_N };
extern const char *const scan_aug_names[SA_N];

typedef struct {
    float strength;   /* magnitude scale, 0..2 */
    float p[SA_N];    /* per-op probability */
    int drop_ignore;  /* 1: dropped slices and cutouts are unsupervised (else the label stays and the net must infer it) */
    float s[SA_N];    /* per-op magnitude scale overriding strength (0: strength) */
    const float *hist; int nhist;   /* SA_HIST: nhist target material CDFs of 256 values each (scan_aug_load_hist) */
} scan_aug_cfg;
/* SA_PAGANIN: phase-retrieval-like low-pass, per axis the two-sided exponential exp(-|t| / lambda) (transfer 1 / (1 + c f^2), the
   Paganin form), lambda (0.4 .. 1.6) s voxels. SA_GIN: a random shallow conv net on the CT (1..3 voxel kernels, 2..4 positive-biased
   channels, leaky ReLU), rescaled to the window's material mean / sd and blended in (alpha 0.3 .. 1). SA_HIST: the window's material
   histogram mapped onto a random target CDF (scans of other scrolls), blended (alpha 0.5 .. 1). SA_MOTION: sample motion during
   the scan along one source axis, before the coarser sampling grid: a ghost (1 or 2 echoes 1 .. 4 s voxels away, weights 0.1 .. 0.35)
   or a linear motion blur (a box 1.5 .. 5 s voxels long). Ops with p 0 draw nothing. */
/* read target histograms (text: one line of 256 counts per box, the format of ufsm ct-hist) into c->hist / c->nhist; 0 on success */
int scan_aug_load_hist(const char *path, scan_aug_cfg *c);

static inline int scan_aug_enabled(const scan_aug_cfg *c) {
    if (!(c->strength > 0)) return 0;
    for (int i = 0; i < SA_N; i++) if (c->p[i] > 0) return 1;
    return 0;
}

/* In place on ctu (P^3), tgt (nch x P^3 labels, only seam touches them), ign (P^3, 1 = ignore). cy/cx: axis centre per
   source z in level voxels (absolute; patch voxel y is o[1] + y), shifted by seam; hasax 0 puts ring centres at a random
   point outside the patch. chunk: zarr inner chunk edge of this level (chunk dropouts align to it). allow_seam 0 for
   tasks whose labels are not dense rasters. scratch: 5 P^3 bytes. Seed fixes every decision. Returns the bitmask of
   the ops applied. */
unsigned scan_aug_apply(const scan_aug_cfg *c, uint64_t seed, uint8_t *ctu, uint8_t *tgt, int nch, uint8_t *ign, int P,
                        const int64_t o[3], float *cy, float *cx, int hasax, const int chunk[3], int allow_seam, void *scratch);

/* The same in two steps (scan_aug_apply = plan + exec, bit for bit): scan_aug_plan makes every decision (the seed's draws in
   scan_aug_apply's order) and does everything on the label side: the input statistics, the seam shift (on ctu as bytes, tgt, ign, cy / cx),
   the ignore flags of air / dropped slices / cutouts, the chunk dropout boxes. ctu then holds the shifted input bytes; its voxels 0 stay
   air (output 0) unless SA_AIR, and the chunk boxes go to air (scan_plan_air): the output's zero set is known without the float work.
   scan_aug_exec runs the float ops of a plan on those bytes (CPU; the GPU executor reads the same plan). */
#define SA_MAXTAP 48
#define SA_MAXP 1024
#define SA_MAXTAB 8192
#define SA_MAXBOX 64
typedef struct { int P, s[SA_MAXP], n[SA_MAXP]; float w[SA_MAXP][SA_MAXTAP]; } scan_bop;   /* out[i] = sum_t w[i][t] in[s[i] + t] */
typedef struct {
    int P; unsigned on;
    double mean, sd;                    /* the input's material statistics (a subsample) */
    float air_level;                    /* SA_AIR: the level air (input 0) becomes */
    int bop_on[3]; scan_bop bop[3];     /* the resolution group per axis (blur, sharpen, motion, lowres composed); 0: identity */
    int bias_g; float bias_G[125];      /* SA_BIAS: the g^3 grid of factors */
    float ccy[SA_MAXP], ccx[SA_MAXP], rmin, res; int nt; float tab[SA_MAXTAB];   /* SA_RING / SA_CUPPING: the radial table */
    float slice_gain[SA_MAXP], slice_off[SA_MAXP];   /* SA_SLICE */
    float noise_s0, noise_mix, noise_im;             /* SA_NOISE */
    float cnoise_rho, cnoise_s0;                     /* SA_CNOISE */
    float pag_a;                                     /* SA_PAGANIN */
    int gin_C, gin_k; float gin_w1[4][27], gin_b1[4], gin_w2[4]; double gin_alpha;   /* SA_GIN */
    int tone; float lut[257]; float step;            /* SA_TONE, SA_QUANT */
    int ndrop, drop_z[SA_MAXBOX], drop_mode[SA_MAXBOX];   /* SA_DROPSLICE */
    int ncut, cut_b0[SA_MAXBOX][3], cut_b1[SA_MAXBOX][3], cut_noisy[SA_MAXBOX];   /* SA_CUTOUT */
    int nchunk, chunk_b0[SA_MAXBOX][3], chunk_b1[SA_MAXBOX][3];   /* SA_CHUNK: boxes that read as air */
    int hist_on; double hist_alpha; float hist_ct[256];          /* SA_HIST: the target CDF */
    uint64_t ns;                        /* the per-voxel noise stream's seed (noise, cnoise, dropped slices, noisy cutouts, in that order) */
} scan_plan;
unsigned scan_aug_plan(const scan_aug_cfg *c, uint64_t seed, uint8_t *ctu, uint8_t *tgt, int nch, uint8_t *ign, int P,
                       const int64_t o[3], float *cy, float *cx, int hasax, const int chunk[3], int allow_seam, scan_plan *p);
void scan_aug_exec(const scan_plan *p, uint8_t *ctu, void *scratch);   /* scratch: 5 P^3 bytes */
/* the output's air (0) set of a plan on its shifted input bytes: air[k] = 1 where the output will be 0 */
void scan_plan_air(const scan_plan *p, const uint8_t *ctu, uint8_t *air);
