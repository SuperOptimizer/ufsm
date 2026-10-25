/* Scroll-damage augmentations of stored-phase windows (sample.c --aug-tear / --aug-delam / ...): the geometry and the per-voxel
   kernels, free of the sampler so the tests can drive them on synthetic boxes. Phase codes as band.h (0 .. PHASE_PERIOD - 1, 255
   unknown); boxes zyx, row-major. */
#pragma once
#include <stdint.h>
#include <stddef.h>

/* the dominant sheet normal and the sheet spacing of a phase box (D x H x W): the structure tensor of the circular phase gradient
   (central differences, every stride-th voxel of the central box inset by inset voxels). n zyx unit; *spacing voxels per turn (the
   median gradient's inverse). Returns the number of gradient samples (outputs unchanged under 32). */
int sa_phase_normal(const uint8_t *ph, int D, int H, int W, int inset, int stride, double n[3], double *spacing);

/* the same over the sub-box lo .. hi (exclusive) of the box; n oriented so the phase grows along it */
int sa_phase_normal_box(const uint8_t *ph, int D, int H, int W, const int lo[3], const int hi[3], int stride, double n[3], double *spacing);

/* --aug-delam: per window up to SA_DELAM_MAX damage sites on papyrus voxels, each
   a peel (kind 0): a thin void opened inside the sheet at the phase phc: within the tangential radius R of the centre c the material
     beyond the split (phase phc .. phc + 0.5 turn along n) moves out by b w, b = g0 (1 - |t|^2 / R^2)^2 voxels (tapered off the
     plane), w falling linearly from 1 at the gap edge to 0 at half a turn (the void beyond stretched); the opened gap gets the void CT
     and the split's phase (the winding kept); the phase is remapped as a function of itself (monotone: the unwrap gains no turn);
   a hole (kind 1): a burned-out patch of one sheet: papyrus inside a ragged ellipse (tangential radii R, R2 along t1 / t2) within
     .22 turn of the sheet's mean phase becomes ash (void + a (CT - void) speckled, soft rims); the labels kept (the net bridges it). */
enum { SA_DELAM_MAX = 6 };
typedef struct {
    int kind; double c[3], n[3], t1[3], t2[3], R, R2, g0, half; uint8_t phc; double ash, wob[3], wph[3];
    float T[2][127];   /* peels: the geometric distance (voxels) from the split of each phase offset 0 .. 126 codes ahead (T[0], along n) / behind (T[1]), measured on the normal through c */
} sa_site;
typedef struct { int n; sa_site s[SA_DELAM_MAX]; double spacing; uint8_t void_ct, iso; } sa_delam;
/* sites drawn on the padded boxes (P + 2 M)^3 (window corner at M): centres in the window's central 3/4 on papyrus with known phase,
   local normals from the phase within 2 turns; kinds by p_hole; returns the number of sites (0: none found) */
int sa_delam_make(sa_delam *d, uint64_t seed, int P, int M, const uint8_t *ct_pad, const uint8_t *ph_pad, double spacing, double p_hole, uint8_t iso);
/* the window (P^3): CT (trilinear pulls in the peels) and phase (remapped) from the padded boxes; t scales every site (0: a copy) */
void sa_delam_apply(const sa_delam *d, double t, int P, int M, const uint8_t *ct_pad, const uint8_t *ph_pad, uint8_t *ct, uint8_t *ph);
/* the margin the peels need (voxels) */
int sa_delam_margin(double spacing);

/* --aug-tear: a smooth cut surface h(p) = c . (p - p0) + sum_j a_j sin(k_j . p + f_j) = 0 roughly across the sheets (c within
   max_tilt of perpendicular to the sheet normal), the side h > 0 shifted by dp, the other by dm (integer voxels, dp - dm = the
   displacement: up to ~1+ turns along the normal and optionally along the sheet), and the seam: 0 none, 1 a thin void (CT the
   window's void level, labels unknown), 2 a blended band (the two sides' CT mixed, labels unknown). */
typedef struct {
    double c[3], p0[3], k[3][3], a[3], f[3];
    int dp[3], dm[3];
    int seam; double width;
    double turns_n, turns_t;   /* (diagnostics: the drawn shift along the normal / the sheet, turns) */
} sa_tear;
/* P the window; n the sheet normal, spacing voxels per turn; maxd the largest displacement (voxels, |dp - dm|); seed */
void sa_tear_make(sa_tear *t, uint64_t seed, int P, const double n[3], double spacing, double maxd);
double sa_tear_h(const sa_tear *t, double z, double y, double x);
/* the margin a padded read needs around the window for this tear (max over axes of |dp|, |dm|) */
int sa_tear_margin(const sa_tear *t);
/* the window (P^3) from padded boxes (P + 2 M)^3 whose corner is the window corner - M: ct and ph (codes) always, w4 / wc (nullable:
   the winding mod 4 and its component, sample.c band_wind4_tiled) pulled with the same integer shifts; void_ct the seam's void CT */
void sa_tear_apply(const sa_tear *t, int P, int M, const uint8_t *ct_pad, const uint8_t *ph_pad, const uint8_t *w4_pad, const uint8_t *wc_pad,
                   uint8_t void_ct, uint8_t *ct, uint8_t *ph, uint8_t *w4, uint8_t *wc);
/* the CT value at the q-quantile of a box's nonzero voxels (0 when none) */
uint8_t sa_ct_quantile(const uint8_t *ct, size_t n, double q);

/* --aug-sheet-density: per-wrap brightness (carbonisation differs from wrap to wrap) and a slow field along the sheets, CT only.
   The wrap coordinate kc = a / spacing - psi - c0 (a the distance along the window's sheet normal n from the window centre, psi the
   phase relative to the papyrus' mean phase in turns, wrapping in the void between sheets; c0 puts the papyrus near integers) is
   about constant along a sheet and steps by 1 from wrap to wrap: a random value per integer (flat within +-.2, smooth between, so a
   curved sheet's drift changes it gradually) gives each wrap its gain and offset. Unknown phase: the nearest known kc along the row
   within 6 voxels, else a / spacing - c0. CT out = void + (CT - void) gain slow + offset m (m the papyrus fraction (CT - void) /
   (iso - void) in 0 .. 1; slow 1 + 3 plane waves of wavelength .7-2.5 window); CT 0 (outside the scan) kept; labels untouched. */
typedef struct {
    double n[3], spacing, c0, phc;    /* phc: the papyrus' mean phase (codes) */
    double s_gain, s_off, s_slow;     /* gain 1 +- s_gain per wrap (.1-.35), offset +- s_off (iso - void) per wrap (0-.15), slow field +- s_slow (0-.3) */
    double k[3][3], f[3];             /* the slow field's plane waves (rad / voxel, zyx) and phases */
    uint64_t seed; uint8_t void_ct, iso;
} sa_density;
/* from the window's CT and phase (D x H x W); returns 0 when the phase gives no normal or no papyrus */
int sa_density_make(sa_density *d, uint64_t seed, const uint8_t *ct, const uint8_t *ph, int D, int H, int W, uint8_t iso);
/* out = the CT with the strengths scaled by t (0: a copy); out may be ct */
void sa_density_apply(const sa_density *d, double t, const uint8_t *ct, const uint8_t *ph, int D, int H, int W, uint8_t *out);
