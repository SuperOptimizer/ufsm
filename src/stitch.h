/* Box stitching (ufsm stitch): local winding fits of overlapping boxes (fit --axis-normal on a crushed scroll: each box has its
   own far axis and its own integer gauge) joined into sheets that cross the boxes, and a gauge-aligned global winding field.
   - align: per overlapping pair, both models' W at the material cells of the overlap (weight = material x phase confidence of
     the network fields): the sign s (+1: W grows the same way in both) and the integer offset k with W_b ~ s W_a + k that most
     of the weight agrees with (round(W_b - s W_a) == k), the agreement fraction, the runner-up offset, the other sign's best,
     and the local agreement (|W_b - s W_a - round(.)| < 1/4: the two fits put sheets at the same places, whatever the count).
   - solve: the maximum spanning tree of the box graph by agreement fixes each box's gauge G = s_i W_i + g_i; every other edge is
     checked against it (a cycle whose offsets do not sum to zero = an inconsistent overlap).
   - quality: per box and block the phase coherence of its W with the network phase (a fit whose W runs against the sheets'
     recto -> verso direction, e.g. across a crushed scroll's flattened core, has coherence ~0 there and high reversed coherence).
   - blending: a partition of unity over the boxes, smoothstep ramps across each face overlap (the overlap's width), hard at the
     union's outer faces, times max(q, 0.05)^2 of the quality map.
   - sheets: a sheet is the zero set of F = sum_i w_i s_i (W_i - off_i) / sum_i w_i over the boxes that are "on" for it: in gauge
     mode every box at the level the solved gauge gives (G = k); in follow mode the gauge (s_i, off_i per box) is carried from
     vertex to vertex from the seed box's own level: boxes join where the sheet reaches them (their level nearest it), keep their
     residual within half a turn of the strongest box's, a box inside a fault of its fit (turns crammed: |grad W| > 1.5 x its
     median) is down-weighted, and a fault of the blend (> fault_ratio x) is bridged straight from the last calm vertex, every box
     re-levelling beyond it (sheet identity carried by continuity: the box fits' integer offsets change across their faults and
     between boxes, so no constant gauge per box fits a crushed scroll). Traced as contours in z planes (no axis: arc length
     along the contour from an anchor column carried row to row, rows aligned to their neighbours), written as tifxyz. All
     coordinates are global native voxels (the fit fields' native grid). */
#pragma once
#include "fit.h"
#include <stdint.h>

typedef struct {
    char name[128], dir[1024];      /* box name (basename of its directory) and the fit set directory */
    fit_set *fs;
    double lo[3], hi[3];            /* the box in global native voxels: [origin, origin + shape h) */
    double tlo[3], thi[3];          /* blending ramp width at each face (native voxels; 0 = outer face, weight 1 up to it) */
    int s; double g;                /* gauge: G = s W + g (g an integer) */
    int comp;                       /* component of the box graph (-1: unset) */
    double wmin, wmax;              /* W range over the box (9^3 samples) */
    double gnom;                    /* median |grad W| in z planes over those samples (turns per native voxel: 1 / sheet spacing) */
    int qn[3]; double qb;           /* local fit quality map (st_quality): blocks of qb native voxels from lo */
    float *q, *qa;                  /* per block: phase coherence of the fit with the network (|mean m e^{-2 pi i W}|), and with W reversed */
    double qmean, qamean;           /* their material-weighted means over the box */
} st_box;

typedef struct {
    int a, b;                       /* box indices */
    int s, k;                       /* W_b ~ s W_a + k */
    double agree, tight;            /* weight fraction with round(W_b - s W_a) == k; with |W_b - s W_a - k| < 0.25 */
    int k2; double agree2;          /* the runner-up offset (same sign) and its fraction */
    double other;                   /* the best agreement with the other sign */
    double frac;                    /* circular mean of W_b - s W_a - k (turns, in (-0.5, 0.5]) */
    double local;                   /* weight fraction with |W_b - s W_a - round(W_b - s W_a)| < 0.25 (any integer offset) */
    double wsum; long long n;       /* sample weight and count */
    double vol;                     /* overlap volume (native voxels^3) */
    int tree;                       /* 1: in the spanning tree */
    int resid;                      /* non-tree edges: k minus the k the solved gauge predicts; 9999 = sign conflict; 0 consistent */
    int used;                       /* 1: passed the thresholds of the solve */
} st_edge;

/* the network fields the alignment weights come from: cells of h native voxels, cell c covering [org + c h, org + (c + 1) h) */
typedef struct {
    double h, org[3];
    int (*read)(void *ctx, const int64_t o[3], const int64_t n[3], uint8_t *w, uint8_t *pc, uint8_t *ps);   /* cells [o, o + n), C order */
    void *ctx;
} st_fields;

int st_box_load(st_box *b, const char *dir);      /* dir/fit (a fit set or model) or dir itself; 0 on success */
void st_box_free(st_box *b);
double st_box_W(const st_box *b, const double p[3]);      /* the box model's W at a global native point */
void st_box_range(st_box *b);                              /* wmin / wmax */
double st_box_phi(const st_box *b, const double p[3]);    /* blending weight (0 outside the box) */
/* the box's local fit quality at p (trilinear over its block map; 1 without one): the blends weight each box by phi max(q, 0.05)^2 */
double st_box_q(const st_box *b, const double p[3]);
/* the quality map of box b: per block of qb native voxels the phase coherence |sum w m e^{-2 pi i W}| / sum w |m| of the box's W
   with the network phase vector m (w the material) on a stride of field cells; 1 = the fit's sheets sit on the network's phase */
int st_quality(st_box *b, const st_fields *F, double qb, int stride, int nthreads);
/* the same coherence for any field ev (0: undefined at p) over [lo, hi): material-weighted means over blocks of qb, and the shares
   of the material in coherent (>= 0.5) and in reversed (anti > coherence) blocks */
typedef int (*st_field_fn)(void *ctx, const double p[3], double *W);
typedef struct { double coherence, anti, coherent_frac, anti_frac; long long blocks; } st_coh;
int st_coherence(st_field_fn ev, void *ctx, const st_fields *F, const double lo[3], const double hi[3], double qb, int stride, int nthreads, st_coh *out);
/* ramp widths from the face overlaps among the boxes with comp >= 0 (by_comp: only between boxes of the same component) */
void st_tapers(st_box *b, int nb, int by_comp);
/* G at p over the boxes of component comp (comp < 0: all): returns the number of boxes with weight > 0 (0: not covered) */
int st_G(const st_box *b, int nb, int comp, const double p[3], double *G);

/* the overlap of boxes a and b sampled every stride cells (weight = w / 255 * phase confidence >= min_w): fills e (a, b set);
   returns 0, 1 when they do not overlap or no sample has weight, -1 on a read error */
int st_align_pair(const st_box *b, int ia, int ib, const st_fields *F, int stride, double min_w, int nthreads, st_edge *e);
/* the global gauge: edges with agree >= min_agree and wsum >= min_wsum, maximum spanning forest by agreement, s / g per box
   (component roots s = +1, gauges shifted so each component's G starts at >= 1), comp per box (components numbered by size,
   0 = the most boxes), tree / resid / used per edge; boxes with excl[i] set (excl nullable) get no edges. Returns the number of
   inconsistent used non-tree edges. */
int st_solve(st_box *b, int nb, st_edge *e, int ne, double min_agree, double min_wsum, const uint8_t *excl);

int st_save(const char *path, const st_box *b, int nb, const st_edge *e, int ne, double um);
/* loads the boxes (their fit sets) and edges; *b / *e allocated; tapers by component */
int st_load(const char *path, st_box **b, int *nb, st_edge **e, int *ne, double *um);

/* one sheet's per-box levels: F = sum phi_i s_i (W_i - off_i) / sum phi_i over the boxes that are on */
typedef struct { int nb; uint8_t *on; int8_t *s; double *off; long long contacts, contact_conflicts; } st_sgauge;
st_sgauge st_sgauge_new(int nb);
void st_sgauge_free(st_sgauge *g);
void st_sgauge_global(st_sgauge *g, const st_box *b, int comp, double k);   /* gauge mode: every box of comp on, F = G - k */
void st_sgauge_seed(st_sgauge *g, int i, double k);                         /* follow mode: box i on at W_i = k, the others off */
int st_F(const st_box *b, const st_sgauge *g, const double p[3], double *F);   /* 0: no box that is on covers p */

typedef struct {
    double tstep;                   /* trace step (native voxels): rows every tstep in z, columns every ~tstep along the contour */
    int max_cols;                   /* per side of the anchor */
    int max_gap;                    /* trace steps through unmasked (no material) cells before a row side stops */
    double bridge_max;              /* follow mode: fault zones (|grad F| > fault_ratio x the boxes' gnom: a fit cramming turns) are
                                       crossed in a straight line up to this length (native voxels); 0: a row side stops at a fault */
    double fault_ratio;
    double min_seed_q;              /* st_seed: with quality maps, the seed box's quality at the seed at least this */
    const uint8_t *mask; int64_t mn[3]; double mo[3], mh;   /* optional material mask: cells of mh native from mo, mn cells */
} st_trace_cfg;
typedef struct { int box, level; long long n; } st_boxlevel;
typedef struct {
    int rows, cols;                 /* trace grid; vertex (r, c) at z0 + r tstep */
    double z0;
    float *x, *y, *z;               /* rows x cols, -1 where invalid (global native) */
    uint8_t *seam;                  /* per vertex: 0 invalid, 1 one box, 2 seam (joined boxes within 1/4 turn), 3 spread 1/4..1/2, 4 conflict (> 1/2),
                                       5 fault bridge, 6 the strongest box's quality < 0.2 */
    long long nvalid, nseam, nspread, nconflict, nunjoined;   /* nunjoined: vertices inside a box (weight >= 1/2) that is not on there */
    long long nbridge, bridges, regauges;   /* bridge vertices, fault bridges, box re-gauges (follow mode) */
    long long nlowq;                /* vertices of class 6 */
    st_boxlevel *lev; int nlev;     /* the (box, level) of the strongest box at each vertex, with vertex counts (sorted) */
    double spread_sum; float *spreads; long long nspreads;   /* seam vertices: level-set spread in native voxels (sorted) */
    long long *own;                 /* per box: vertices where it has the largest weight among the joined boxes */
    int nboxes;                     /* boxes owning vertices */
} st_sheet;
/* traces the sheet F = 0 of the levels g from the seed (global native; projected onto the contour in its z plane first). The gauge
   is carried per vertex; follow: boxes join on contact, are re-gauged against the strongest box, faults are bridged; boxes outside
   comp (comp >= 0) never join. On return g is the seed gauge plus every box that joined. 0 on success */
int st_trace(const st_box *b, int nb, int comp, const st_trace_cfg *c, st_sgauge *g, int follow, const double seed[3], st_sheet *s);
void st_sheet_free(st_sheet *s);
/* a seed of F = 0 in box i: along the line through the box centre (at z = lo + zf (hi - lo)) towards its axis, the crossing
   nearest the centre inside the box (and the mask of c, nullable); 0 on success */
int st_seed(const st_box *b, int i, double zf, const st_sgauge *g, const st_trace_cfg *c, double p[3]);

int cmd_stitch(int argc, char **argv);
