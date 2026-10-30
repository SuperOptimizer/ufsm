/* Point-wise scroll deformation for any sampler geometry (cube windows, ray-aligned slabs): one smooth invertible pull
   out -> src in level voxels (zyx), so a sampler that reads the CT AND its labels at deform_pull(p) keeps them consistent.
     E  elastic: x = u + e(u), e = sum of K random 3D plane-wave sines (wavelength lam_lo..lam_hi voxels), scaled so that
        sum |A_k| |k_k| <= lip (< 1: invertible, Jacobian det > 0); amplitude up to amp voxels
     C  the crush map of src/crush.h (gap collapse along n, local void field, in-plane squash, bends, kink) about the
        deformation's centre, built from the source CT along n (crush_map_columns)
   The composite: out -> (E) -> crush output space -> (C) crush_pull -> src. Normals (gradients of a label pulled through the
   map, e.g. the phase) map as g_out = J^T g_src with J = d src / d out (deform_jacobian). Phase / winding labels pulled at
   src are exact by construction (no label is moved separately from the CT). */
#pragma once
#include "crush.h"
#include <stdint.h>

typedef struct {
    float crush_p;            /* probability the crush stage is on (else E only) */
    crush_cfg crush;          /* its parameters (crush.p ignored; rotate_deg forced 0: the caller's frame stays the slab's) */
    float elastic_p, amp, lam_lo, lam_hi, lip;   /* stage E: probability, max amplitude (voxels), wavelength range, Lipschitz cap */
} deform_cfg;
deform_cfg deform_cfg_default(void);   /* crush_p 0.5, crush defaults + void .05,.4 local 1 len 40 max 1.5, elastic p .5 amp 8 lam 96..384 lip .45 */
/* the same flags as train (--aug-crush-*) plus --deform-crush P, --deform-elastic P, --deform-amp A, --deform-lam LO,HI, --deform-lip L */
int deform_cfg_parse(deform_cfg *c, int argc, char **argv);
void deform_cfg_print(const deform_cfg *c);

typedef struct {
    double o[3];              /* the crush window's origin (level voxels, zyx): crush window coordinates = p - o */
    int P;                    /* crush window size (covers every point the caller pulls) */
    int ncr; crush_map m;     /* stage C on */
    int ne; double ek[8][3], eA[8][3], eph[8];   /* stage E: e(u) = sum eA_k sin(ek_k . u + eph_k) */
} deform;

/* centre c (level voxels, zyx), n the sheet normal (e.g. a radial ray's direction; zyx, any length), P the crush window covering the
   caller's footprint (a T x T x L slab: P >= sqrt(L^2 + 2 T^2)). ct(ud, x): CT (0..255, 0 outside) at window coordinates x (as
   crush_map_columns). Returns 0 (crush on or off: deform_on says), -1 on alloc failure. Deterministic in seed. */
int deform_make(deform *D, const deform_cfg *cfg, uint64_t seed, const double c[3], const double n[3], int P, crush_ct_fn ct, void *ud);
/* the same, with the crush columns built only over the hull of the np points pts (zyx, relative to c; every point the caller pulls
   lies in it, e.g. a slab's 8 corners); the elastic stage's displacement bound is added as a margin. nullptr pts = the cube */
int deform_make_ext(deform *D, const deform_cfg *cfg, uint64_t seed, const double c[3], const double n[3], int P, const double (*pts)[3], int np, crush_ct_fn ct, void *ud);
void deform_free(deform *D);
int deform_on(const deform *D);   /* any stage active */
/* src = pull(out) (level voxels, zyx); jdet (nullable): Jacobian determinant */
void deform_pull(const deform *D, const double out[3], double src[3], double *jdet);
/* J[i][j] = d src_i / d out_j (central differences, h 1e-3) */
void deform_jacobian(const deform *D, const double out[3], double J[3][3]);
/* the source box any pull of points within +-ext (per axis, around the centre) reads: lo / hi (level voxels, zyx), for the caller's
   CT / label read; sampled on the box surface (the map is a homeomorphism) + 3 voxels */
void deform_bounds(const deform *D, const double ext[3], double lo[3], double hi[3]);
