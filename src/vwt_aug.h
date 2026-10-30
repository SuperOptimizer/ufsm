/* vwt ray-sampler augmentation (src/vwt.h; DA): geometric scroll deformation as a map of the SAMPLING COORDINATES (the sampler reads
   CT and targets at the mapped point, so ph / comp / pw stay exact), and intensity ops on im only.
   Geometry = src/deform.h about the slab centre: smooth elastic + the crush map (gap collapse along the ray direction d, which is the
   sheet normal of a radial slab; local void-ratio field: gaps close fully / stay / widen; in-plane squash; bends; a kink).
   The map has det > 0 and is monotone along the sheet normal, so ph keeps increasing along +k.
   Usage in vwt_sample.c (per slab, frame o / a / b / d set):
     vwa *w = vwa_begin(&cfg, seed, s, ct_xyz, ud);           (nullptr: no geometric aug for this slab)
     if (w) { vwa_box(w, T, L, lo, hi); ... read CT / labels over lo..hi; vwa_src_lattice(w, T, L, q); sample at q[(i T + j) L + k] }
     vwa_end(w);  vwa_intensity(&cfg, seed2, s->im, s->va, T, L); */
#pragma once
#include "deform.h"
#include "vwt.h"
#include <stdint.h>

typedef struct {
    int on;               /* --vaug 0|1 (default 1): 0 = no aug at all */
    float geo_p;          /* probability of any geometric aug per slab (--vaug-geo, default 0.5) */
    deform_cfg df;        /* stages given geo (--deform-*, --aug-crush-*) */
    int lattice;          /* exact pulls every `lattice` voxels, trilinear between (--vaug-lattice, default 2; 1 exact) */
    float gain_p, gamma_p, noise_p, blur_p, lowres_p, quant_p;   /* intensity op probabilities (--vaug-gain-p ... --vaug-quant-p) */
    float strength;       /* intensity strength (--vaug-strength, default 1) */
} vwa_cfg;
vwa_cfg vwa_cfg_default(void);
int vwa_cfg_parse(vwa_cfg *c, int argc, char **argv);   /* 0 ok */
void vwa_cfg_print(const vwa_cfg *c);

typedef struct vwa vwa;
/* ct_xyz: the source CT (0..255, 0 outside) at a point (xyz scan voxels of the CT level); used to build the gap-collapse columns.
   nullptr when no geometric aug is drawn for this slab (or c->on 0). Deterministic in seed. */
vwa *vwa_begin(const vwa_cfg *c, uint64_t seed, const vws_sample *s, double (*ct_xyz)(void *ud, const double xyz[3]), void *ud);
void vwa_src(const vwa *w, const double p[3], double q[3]);                 /* xyz -> xyz */
void vwa_src_lattice(const vwa *w, int T, int L, float *q);                 /* q[((i T + j) L + k) 3 + c]: the source point of slab voxel (i, j, k) */
void vwa_box(const vwa *w, int T, int L, double lo[3], double hi[3]);      /* xyz bbox of every q (+3 voxels) */
double vwa_jdet(const vwa *w, const double p[3]);                          /* Jacobian determinant of the map at p (diagnostics / tests) */
void vwa_end(vwa *w);
/* intensity ops on im where va (va 0 stays 0); deterministic in seed */
void vwa_intensity(const vwa_cfg *c, uint64_t seed, uint8_t *im, const uint8_t *va, int T, int L);
