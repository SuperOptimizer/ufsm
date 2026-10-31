/* Image operations of the render pipeline (ufsm render-sheets / overlay-red / png2jpg / tifxyz-scale), in C so no pipeline
   step needs Python: Pillow-exact 8-bit resampling (BOX and BILINEAR, the two passes and fixed-point coefficients of Pillow's
   Resample.c), the 5-panel red-ink overlay of the former overlay_red.py / overlay_stitched.py (numpy float32 arithmetic, the
   scipy 20-iteration cross dilation as an L1 distance), JPEG (libjpeg, Pillow's defaults: quality, 4:2:0, baseline) and
   minimal PNG reading, and the tifxyz rescale of the former xform_wraps.py. */
#pragma once
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

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
int jpeg_write_gray(const char *path, int w, int h, const uint8_t *g, int quality);  /* one-channel (grayscale) JPEG, 0 on success */
uint8_t *jpeg_read_rgb(const char *path, int *w, int *h);   /* any JPEG as RGB, nullptr on failure (no exit on corrupt input) */
/* the longest side scaled down to max_px (antialiased bilinear, aspect kept; *rgb replaced); no-op when already within or max_px <= 0; 0 ok */
int img_fit_rgb(uint8_t **rgb, int *w, int *h, int max_px, int nthreads);
/* an 8-bit PNG (gray, gray+alpha, RGB, RGBA, palette; not interlaced) as RGB (alpha dropped, as Pillow's convert('RGB')) */
uint8_t *png_read_rgb(const char *path, int *w, int *h);

/* xform_wraps.py: x / y / z of the tifxyz src divided by div (float32), minus off (z, y, x) where z >= 0, -1 elsewhere;
   meta.json copied with scale x div and voxel_size_um x div. Grids written deflate float32 TIFFs. 0 on success. */
int tifxyz_scale(const char *src, const char *dst, double div, const double off[3]);
extern int tifxyz_flip_rows;   /* tifxyz_scale: 1 = write the rows in reverse order (renders of Paris 4 wraps read upright; ufsm tifxyz-scale --flip-rows 1) */
/* the tifxyz src copied to dst with every vertex moved d (its own units) along the vertex normal +n = (d/d column) x (d/d row)
   (central differences over valid neighbours; vertices without a normal stay); meta.json copied. toward = {y, x} (a vertical line, the
   scroll axis): +n is first flipped when most vertex normals face away from it, so D > 0 moves the sheet INWARD. 0 on success */
int tifxyz_shift(const char *src, const char *dst, double d, const double *toward);   /* toward (y, x) or nullptr: see above */
/* the scaled coordinate of one vertex value v (z the vertex's z): numpy float32 v / div - off, or -1 for an invalid vertex */
static inline float tifxyz_scaled(float v, float z, float divf, float offf) { return z < 0 ? -1.f : v / divf - offf; }
/* ufsm render output pixels along a grid of n vertices at gs pixels per grid step: floor((n - 1) gs) + 1 with a 1e-6 slack, so the float32
   meta scales of one mesh at two divisors (0.05 -> step 19.9999997 x 0.5, 0.2 -> 4.99999993 x 2) give the same size (ink-compare pairs) */
static inline int render_out_px(int n, double gs) { return (int)floor((n - 1) * gs + 1e-6) + 1; }
/* the papyrus look (ink-compare --papyrus, strip-mosaic --papyrus): stretched CT v (0..255) through dark brown .. warm tan, carbon ink of
   density d (0..1) over it; img_pap_ink: a stretched ink value (0..255) times gain -> d (ramp .12 .. .9, smoothstep, at most .92) */
static inline void img_pap_rgb(int v, float d, uint8_t *o) {
    static const float P[4][3] = {{58, 36, 20}, {128, 88, 52}, {196, 156, 104}, {236, 214, 170}};
    const float t = (float)v / 255.f * 3.f; const int i = t >= 3.f ? 2 : (int)t; const float f = t - i;
    for (int e = 0; e < 3; e++) { const float pv = P[i][e] + f * (P[i + 1][e] - P[i][e]), iv = e == 0 ? 26.f : e == 1 ? 22.f : 19.f; o[e] = (uint8_t)lroundf(pv * (1.f - d) + iv * d); }
}
static inline float img_pap_ink(int ink, float gain) {
    float d = (float)ink / 255.f * gain; d = d < .12f ? 0.f : d > .9f ? 1.f : (d - .12f) / .78f; return d * d * (3.f - 2.f * d) * .92f;
}

float *img_read_f32(const char *path, int *w, int *h);     /* a float32 single-sample TIFF, nullptr on failure */
/* ufsm render of one wrap grid (X, Y, Z: w x h vertices, gs output pixels per grid step) through nv (1-2) volumes into img[v]
   (ow x oh, zeroed by the caller): per pixel the max over +- depth[v] voxels along the normal (depth 0: the sample itself), exactly
   as main.c's tiled render. Every chunk is decoded once (chunk-ordered blocks, decoders ahead of renderers); budget bytes of decoded
   chunks (exceeded only by one chunk row's needs); crop (x, y, w, h) or nullptr: only blocks meeting it. 0 ok, -1 on failure. */
struct z3;
int render_wrap_chunked(struct z3 *const *vz, int nv, const int *depth, const float *X, const float *Y, const float *Z, int w, int h, double gs,
                        uint8_t *const *img, int ow, int oh, const int *crop, size_t budget, int nthreads, int io_threads, int verbose);
/* the same for ns (1-8) wrap grids of one size and step rendered in one chunk-ordered pass (e.g. a wrap's recto and verso faces on one
   flat uv): each block's chunks are those of the union of the grids' boxes, so shared chunks are decoded once; grid s, volume v goes to
   img[s nv + v]. Every image equals render_wrap_chunked's of that grid alone, byte for byte. */
int render_wraps_chunked(struct z3 *const *vz, int nv, const int *depth, int ns, const float *const *X, const float *const *Y, const float *const *Z,
                         int w, int h, double gs, uint8_t *const *img, int ow, int oh, const int *crop, size_t budget, int nthreads, int io_threads, int verbose);
/* render --fill: holes = pixels whose CT sample on the surface is void (the masked CT is 0 off the papyrus: the surface sits in a gap
   between sheets, or crosses a tear / fold). With n > 0, such a pixel searches along the surface normal t = +-1 .. +-n voxels for the
   nearest sample with CT >= thr and renders EVERY volume centred there (ink and CT keep the same offset, so they stay aligned); at
   equal distance both sides papyrus: prefer > 0 the +normal side, < 0 the -normal side, 0 the larger CT. n = 0: no search, only the
   flags. The CT is volume 1 (nv must be 2). flag[s] (ow x oh, sheet s) gets the codes below; the caller sets it to RFILL_NONE.
   vgrid[s] (w x h, or vgrid nullptr): nonzero at vertices tifxyz_fill_holes interpolated; pixels with such a corner get + RFILL_GRID. */
enum { RFILL_ON = 0, RFILL_NEG = 50, RFILL_GRID = 100, RFILL_INPAINT = 250, RFILL_NONE = 254, RFILL_EMPTY = 255 };   /* +k: k (1..49); -k: 50 + k */
typedef struct { int n, thr, prefer; uint8_t *const *flag; const uint8_t *const *vgrid; } render_fill;
int render_wraps_chunked_fill(struct z3 *const *vz, int nv, const int *depth, int ns, const float *const *X, const float *const *Y, const float *const *Z,
                              int w, int h, double gs, uint8_t *const *img, int ow, int oh, const int *crop, size_t budget, int nthreads, int io_threads,
                              int verbose, const render_fill *F);
/* the cosmetic fill of the pixels still empty after the search (flag RFILL_EMPTY): push-pull over `levels` pyramid levels (holes up to
   ~2^levels pixels across; larger ones stay empty) from the surface pixels (flags 0..RFILL_NEG + 99), the same for every image in
   imgs[0..nimg); filled pixels get RFILL_INPAINT. Tiles of 512 pixels with a 2^(levels + 1) halo on threads. Returns the pixels filled. */
/* render --fill-grid: holes in a tifxyz grid (4-connected components of invalid vertices, Z < 0 or NaN) that touch no grid border and
   have at most maxv vertices get X, Y, Z interpolated from their rims (push-pull start, then Gauss-Seidel on the Laplace equation),
   so the surface bridges tears and dropped triangles and renders real data there. filled (w x h, or nullptr) = 1 at those vertices.
   Returns the vertices filled (holes counted in *nholes, holes skipped as too large in *nbig). */
size_t tifxyz_fill_holes(float *X, float *Y, float *Z, int w, int h, size_t maxv, uint8_t *filled, size_t *nholes, size_t *nbig);
size_t img_pushpull_fill(uint8_t *const *imgs, int nimg, uint8_t *flag, int w, int h, int levels, int nthreads);
/* PatchMatch (Barnes 2009) refinement of the pixels flagged RFILL_INPAINT (initialised by img_pushpull_fill): 3-level pyramid,
   7 x 7 patches over all nimg channels jointly (CT and ink from the same source patch), sources = surface pixels within a 40-pixel
   halo of each 256-pixel tile. 0 ok, -1 out of memory. */
int img_patchmatch_fill(uint8_t *const *imgs, int nimg, const uint8_t *flag, int w, int h, int nthreads);
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
