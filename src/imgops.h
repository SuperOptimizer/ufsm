/* Image operations of the render pipeline (ufsm render-sheets / overlay-red / png2jpg / tifxyz-scale), in C so no pipeline
   step needs Python: Pillow-exact 8-bit resampling (BOX and BILINEAR, the two passes and fixed-point coefficients of Pillow's
   Resample.c), the 5-panel red-ink overlay of the former overlay_red.py / overlay_stitched.py (numpy float32 arithmetic, the
   scipy 20-iteration cross dilation as an L1 distance), JPEG (libjpeg, Pillow's defaults: quality, 4:2:0, baseline) and
   minimal PNG reading, and the tifxyz rescale of the former xform_wraps.py. */
#pragma once
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

enum { PIL_BOX = 0, PIL_BILINEAR = 1 };

/* Pillow's precompute_coeffs + normalize_coeffs_8bpc for in -> out samples: per output sample the first input index
   bounds[2 i], the count bounds[2 i + 1] and ksize fixed-point (22 bit) weights at kk[i * ksize]. Returns ksize (0 on error). */
int pil_coeffs(int in, int out, int filter, int **bounds, int32_t **kk);
/* one row (ch interleaved channels) resampled horizontally with coefficients from pil_coeffs */
void pil_hrow(const uint8_t *in, int ch, uint8_t *out, int out_w, int ksize, const int *bounds, const int32_t *kk);
/* output pixel of the vertical pass: rows[k] (k = 0 .. bounds count - 1) are the input rows bounds[2 y] + k, at column offset off */
static inline uint8_t pil_clip8(int v) { v >>= 22; return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); }
/* the whole image (w x h, ch channels) -> ow x oh, as Pillow's Image.resize((ow, oh), filter) */
int pil_resize(const uint8_t *in, int w, int h, int ch, uint8_t *out, int ow, int oh, int filter, int nthreads);

/* scipy.ndimage.binary_dilation(ct <= 0, iterations=r) (the cross structure, border 0): mask[i] = 1 where a ct == 0 pixel is
   within L1 distance r */
void l1_dilate_zero(const uint8_t *ct, int w, int h, int r, uint8_t *mask);

/* the overlay of overlay_red.py: CT ct (W x H, 4.8 um) | CT + red ink | ink panels (nink of 1..3: +-48, +-29, +-10 um renders,
   ink[k] iw x ih at 9.6 um, Pillow-bilinear resized to W x H), 8 white columns apart, the whole Pillow-BOX halved.
   Returns the half-resolution RGB (*ow x *oh, malloc'd), nullptr on allocation failure. */
uint8_t *overlay_red_half(const uint8_t *ct, int W, int H, const uint8_t *const *ink, int nink, int iw, int ih, int *ow, int *oh, int nthreads);
/* overlay_red.py's skip rule: under 5% of the CT pixels > 0 */
int overlay_red_skip(const uint8_t *ct, size_t n);

int jpeg_write_rgb(const char *path, int w, int h, const uint8_t *rgb, int quality);   /* 0 on success */
/* an 8-bit PNG (gray, gray+alpha, RGB, RGBA, palette; not interlaced) as RGB (alpha dropped, as Pillow's convert('RGB')) */
uint8_t *png_read_rgb(const char *path, int *w, int *h);

/* xform_wraps.py: x / y / z of the tifxyz src divided by div (float32), minus off (z, y, x) where z >= 0, -1 elsewhere;
   meta.json copied with scale x div and voxel_size_um x div. Grids written deflate float32 TIFFs. 0 on success. */
int tifxyz_scale(const char *src, const char *dst, double div, const double off[3]);
/* the tifxyz src copied to dst with every vertex moved d (its own units) along the vertex normal +n = (d/d column) x (d/d row)
   (central differences over valid neighbours; vertices without a normal stay); meta.json copied. toward = {y, x} (a vertical line, the
   scroll axis): +n is first flipped when most vertex normals face away from it, so D > 0 moves the sheet INWARD. 0 on success */
int tifxyz_shift(const char *src, const char *dst, double d, const double *toward);
/* the distortion check of a tifxyz grid (w x h vertices, step native voxels between them; Z < 0 invalid): a quad is good when its 4
   corners are valid, its 4 edges have unit stretch |dP| / step within tol (0.1 = +-10%), its u and v edges are orthogonal within
   ang_deg, and qok (per quad, (w-1) x (h-1), optional) is set. *share_ok: good quads / valid quads. The largest axis-aligned rectangle of
   good quads: rows r0 .. r0 + rh - 1, columns c0 .. c0 + cw - 1 (quads; rh x cw = 0 when none). 0 on success */
int tifxyz_rect(const float *X, const float *Y, const float *Z, int w, int h, double step, double tol, double ang_deg, const uint8_t *qok,
                int *r0, int *c0, int *rh, int *cw, double *share_ok);   /* toward (y, x) or nullptr: see above */
/* the scaled coordinate of one vertex value v (z the vertex's z): numpy float32 v / div - off, or -1 for an invalid vertex */
static inline float tifxyz_scaled(float v, float z, float divf, float offf) { return z < 0 ? -1.f : v / divf - offf; }

float *img_read_f32(const char *path, int *w, int *h);     /* a float32 single-sample TIFF, nullptr on failure */
uint8_t *img_read_u8(const char *path, int *w, int *h);    /* a uint8 single-sample TIFF */

/* numpy.percentile(v, q) (method linear, numpy 2's _lerp) of n values; v is reordered. The float32 version is numpy's on float32
   data (q / 100, the virtual index and the interpolation all float32). n > 0. */
double np_percentile_f64(double *v, size_t n, double q);
float np_percentile_f32(float *v, size_t n, float q);

/* the former phase_score.py (ufsm phase-score): the network phase rendered on a surface (ufsm render of the fit-field arrays
   pc / ps at depth 0): over the pixels with pc != 0 (rendered) and |(pc - 128, ps - 128)| > 13 (some confidence) the sums
   num = sum (pc - 128) and den = sum |(pc - 128, ps - 128)|: the on-recto score num / den. With a stitched sheet's seam.tif
   (sw x sh, 2 grid steps per pixel from row r0, column c0): cls[1..6] the same per seam class (one box, seam ok, spread,
   conflict, bridge, low q), only the classes with >= 200 pixels (the others left zero). Returns the masked pixel count. */
typedef struct { double num, den; long long n; } phs_acc;
long long phase_score_surface(const uint8_t *pc, const uint8_t *ps, int w, int h, const uint8_t *seam, int sw, int sh, int r0, int c0,
                              phs_acc *acc, phs_acc cls[7]);
/* the whole report of phase_score.py PC_DIR PS_DIR [SHEETS_DIR] on out (the same lines); 0 on success */
int phase_score_dirs(const char *pcd, const char *psd, const char *sd, FILE *out);

/* drawing on RGB images (w x h): a 5 x 7 bitmap font (6 pixel advance; printable ASCII digits, lower case letters, ( ) , . - :
   others blank) at top-left (x, y), and Pillow's ImageDraw line of width 2 between integer points (ImagingDrawWideLine: the
   parallelogram p0, p1, p1 + (round(dy / L), -round(dx / L)), p0 + (..) filled; a point when p0 == p1) */
void img_text(uint8_t *rgb, int w, int h, int x, int y, const char *s, const uint8_t col[3]);
void img_line2(uint8_t *rgb, int w, int h, int x0, int y0, int x1, int y1, const uint8_t col[3]);

/* ufsm ink-compare: renders of one surface (same pixel grid) side by side and their ink statistics.
   img_text at an integer scale sc (each font pixel an sc x sc block). */
void img_text_scaled(uint8_t *rgb, int w, int h, int x, int y, const char *s, int sc, const uint8_t col[3]);
/* over the pixels with mask != 0: n, the mean probability, the fraction with ink >= 128 (p > 0.5), the 50th / 95th / 99th
   percentiles (lowest value whose cumulative count reaches q n, / 255) and band: the standard deviation of (masked box mean of
   radius rs - masked box mean of radius rl) / 255, the ink contrast at scales between the two radii (letter strokes .. letters) */
typedef struct { long long n; double mean, frac50, p50, p95, p99, band; } ink_stats_t;
int ink_stats(const uint8_t *ink, const uint8_t *mask, int w, int h, int rs, int rl, ink_stats_t *st);   /* 0 on success */
double ink_corr(const uint8_t *a, const uint8_t *b, const uint8_t *mask, size_t n);   /* Pearson correlation over mask != 0 */
/* nimg w x h renders side by side, 8 white columns apart (is_ct[k]: the overlay's CT contrast (v - 40) x 2.2, else as is), labels
   in yellow at the top-left of each panel, the whole Pillow-BOX halved: RGB *ow x *oh (malloc'd; nullptr on failure) */
uint8_t *ink_panels_half(const uint8_t *const *img, const int *is_ct, int nimg, int w, int h, const char *const *labels, int *ow, int *oh, int nthreads);
/* the same at full resolution (half_res 0) or halved (1) */
uint8_t *ink_panels(const uint8_t *const *img, const int *is_ct, int nimg, int w, int h, const char *const *labels, int half_res, int *ow, int *oh, int nthreads);
/* contrast stretch in place over mask != 0: the plo / phi quantiles (e.g. 0.01 / 0.99) -> 0 / 255, linear between, clipped; pixels
   outside the mask -> 0 */
void img_stretch(uint8_t *im, const uint8_t *mask, size_t n, double plo, double phi);
/* the win x win window (clipped to the image, stride win / 4) with the largest masked sum of im among those at least half inside
   the mask: its corner in *x0 / *y0; -1 when there is none */
int img_best_window(const uint8_t *im, const uint8_t *mask, int w, int h, int win, int *x0, int *y0);
/* the rectangle x, y, cw, ch (may reach outside the image: 0 there) of a w x h image as a new cw x ch image (render --crop) */
uint8_t *img_crop_u8(const uint8_t *im, int w, int h, int x, int y, int cw, int ch);
