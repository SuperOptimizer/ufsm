/* Winding-id assignment of surfaces from several authors / runs against a reference winding raster (the first layer of training
   on every published surface of a scroll; see DESIGN.md "Many surface sets"). The published Paris4 wraps come as one raster of
   winding codes (raster --value: 1 + round(18 q) mod 252, q = the wrap's sheet number + dir theta / 2 pi around the umbilicus), so
   their numbering is consistent; other authors' segments (tifxyz) carry none, or their own. Side labels need the order of
   consecutive wraps, so every surface must get its q in the reference's gauge before rasterising:
     q(v) = K + dir phi(v),  phi = theta / 2 pi unwrapped along the mesh grid (continuous across theta's branch cut),
   one integer K (mod 14, the codes' period) and one dir per connected component. Vertices sampled every `step` rows / cols look up
   the nearest reference voxel with a winding within `tol` label voxels; each such match votes K = round(q_ref - dir phi) mod 14 for
   both dirs; the (dir, K) with the most votes wins. A component is
     DUPLICATE   most samples lie on reference voxels (median distance <= tol / 2) and agree: it re-traces published wraps
     NEW         it agrees where it overlaps the reference but mostly lies elsewhere (new coverage in the reference's gauge)
     CONFLICT    under 80% of its matches agree: it switches sheets or contradicts the reference (its disagreeing cells are dropped)
     UNASSIGNED  under 5% of its samples (and fewer than 8) match: nothing anchors its K (needs a dense winding reference, e.g. a
                 fit / phase-raster --winding raster, or another surface set first)
   Output q per vertex (NaN = invalid, unassigned, dropped); ufsm raster --value q.tif then writes every surface's codes into one
   raster (codes within 4 steps of each other: the same wrap; otherwise 255, a conflict), phase-raster --pair 2 the side labels.
   With drop_dup the vertices of cells that match the reference are dropped too (pick the reference where both exist, add only new
   coverage). */
#pragma once
#include <stddef.h>
#include <stdint.h>

enum { WIND_CODES = 0, WIND_WINDING = 1 };   /* reference kinds: raster --value codes (0 empty, 255 conflict) / phase-raster --winding (255 unknown) */
enum { WA_UNASSIGNED = 0, WA_DUPLICATE = 1, WA_NEW = 2, WA_CONFLICT = 3 };
static inline const char *wind_verdict_name(int v) { static const char *n[4] = {"unassigned", "duplicate", "new", "conflict"}; return v >= 0 && v < 4 ? n[v] : "?"; }

/* nearest reference voxel with a winding within tol label voxels of each point (label-grid z, y, x): dist (label voxels, > tol =
   none) and q_ref (turns mod 14) */
typedef void (*wind_lookup_fn)(void *ud, const float *pts, size_t n, float tol, float *dist, float *qref);

typedef struct {
    int dir, K, verdict;        /* per component (the largest first) */
    size_t nvalid, nsample, nmatch, nagree;
    double median_dist, mean_abs_resid;   /* label voxels; turns (matched, agreeing samples) */
} wind_comp;

typedef struct {
    float *q;                   /* w * h, NaN where invalid / unassigned / dropped */
    uint8_t *flag;              /* w * h: 0 invalid, 1 assigned, 2 dropped (conflict cell), 3 dropped (duplicate cell, drop_dup), 4 unassigned */
    wind_comp comp[16]; int ncomp;   /* components with >= 1% of the valid vertices, largest first */
} wind_result;

typedef struct {
    double scale;       /* label voxels per mesh coordinate unit (scan um / reference um) */
    float tol;          /* label voxels */
    int step;           /* sampling stride over rows / cols (>= 1) */
    int dir;            /* +1 / -1, or 0: from the votes */
    int drop_dup;
    int dense;          /* the reference is a dense winding raster (WIND_WINDING: every voxel between the wraps has its q): it anchors K
                           everywhere but cannot tell a re-trace from new coverage, so no DUPLICATE verdict and no drop_dup */
} wind_params;

/* xyz: w * h * 3 (x, y, z) mesh coordinates (scan voxels), valid w * h; cy / cx(z): the umbilicus at scan z (axis_at). Returns 0,
   -1 out of memory. */
int wind_assign(const float *xyz, const uint8_t *valid, int w, int h, void (*axis_fn)(void *, double, double *, double *), void *axis_ud,
                wind_lookup_fn lookup, void *lookup_ud, wind_params p, wind_result *out);
void wind_result_free(wind_result *r);

/* A surface set's own gauge, without a reference (the first set of a scroll): the largest connected component gets
   q = off + dir phi, phi = theta / 2 pi unwrapped along the mesh grid as in wind_assign. Its two ends are the first and last 5% of
   the grid's columns (the authors' unrollings run along columns: a single-wrap segment wNNN covers about one turn, a merged spiral
   starts at its innermost wrap).
   sense 0: the start is the end nearer the umbilicus, dir the sign that makes q grow away from it (one surface alone);
   sense +-1: the scroll's spiral sense, common to all its surfaces (q grows outward with sense * phi): dir = sense, the start is the
   end q grows from (a single wrap's two ends lie about one pitch apart radially: an axis off the umbilicus, or a partial wrap, can
   swap them, the set's sense cannot).
   phi0 NaN: q = K at the start. Else phi0 is the set's seam (turns; the circular mean of its surfaces' start azimuths): a surface
   under 1.2 turns gets the turn [K, K + 1) from the seam around its median (partial wraps and wraps cut a little before or after
   the seam keep the name's turn), a longer one K at its start's turn (start within half a turn of the seam).
   Other components: unassigned (flag 4, NaN). info: dir; radial_dir (the dir the radius rule picks); r_start / r_end (mean radius
   of the two column bands, scan voxels) and r_mean; span (turns); phi_c[2] / r_c[2]: phi and radius at the first / last columns;
   q_start. Returns 0, -1 out of memory, -2 no valid vertex. */
typedef struct { int dir, radial_dir; double r_start, r_end, r_mean, span, q_start, phi_c[2], r_c[2]; size_t nvalid, nassigned; } wind_seed_info;
int wind_seed(const float *xyz, const uint8_t *valid, int w, int h, void (*axis_fn)(void *, double, double *, double *), void *axis_ud,
              double K, int sense, double phi0, wind_seed_info *info, wind_result *out);
/* A set's spiral sense and seam from its surfaces' wind_seed infos with sense 0: the radius rule's votes weighted by each surface's
   radial gap between its column ends (vote[0] for +1, vote[1] for -1), then the circular mean of the start azimuths under that
   sense (turns, [0, 1); *resultant: the mean resultant length, 1 = all starts on one azimuth).
   NOT a sense decision on real sets (2026-10-19): an axis off the umbilicus or a crushed section moves the end bands' radii by far
   more than the turns between them (Paris 4 June: 25 short surfaces outvoted the 3 long ones, + 2446 / - 27987, and the set came out
   reversed). wind_seed_decide decides; this vote is only its tie-break and a report. */
void wind_seed_set(const wind_seed_info *in, int n, int *sense, double *phi0, double vote[2], double *resultant);
/* the seam (the circular mean of the start azimuths) of a set under a given sense */
void wind_seed_seam(const wind_seed_info *in, int n, int sense, double *phi0, double *resultant);

/* The set's spiral sense and the order of its names from its own geometry (wind_seed_decide). Two vertices in one (z, azimuth) bin
   (WIND_ZBIN scan voxels of z, 1 / WIND_TBIN turn around the axis at that z) lie on one ray from the axis at one height, so the
   outer one must carry the higher q: one turn per sheet crossed. Points sorted by bin and radius vote in consecutive pairs: agree
   when the outer q is >= 0.5 turn higher, disagree when >= 0.5 lower (pairs under half a pitch apart, or not a whole number of turns
   apart within 0.25: no radial vote; partial segments whose phi unwrapping jumps near the axis gave such junk on PHerc0009B). Pairs inside
   one surface need it to span over a turn (they test the sense); pairs across surfaces (cross: named sets, K from the names) test
   the names' order and the K, and two surfaces under a quarter pitch apart (one sheet traced twice: where wrap K's end meets wrap
   K + 1's start at the seam) must agree within half a turn: under the wrong sense single wraps still step one turn per sheet
   outward and only their q within the turn runs backwards (harmless to the pair-rule phase); this seam continuity is the only
   evidence of their sense. An axis off the umbilicus, an elliptical or crushed section and z drift along the grid's rows do not
   enter: both points of a pair share azimuth and height (the axis only has to lie inside the sheets it is compared across). */
enum { WIND_ZBIN = 8, WIND_TBIN = 512 };
typedef struct { int64_t key; float r; float q[2]; int32_t surf; int32_t turn; } wind_pt;   /* q[0]: wind_seed q under sense +1, q[1]: under -1 (names as given);
                                                                                              turn: the surface's phi range is >= 0.9 turn (else its own pairs
                                                                                              are phi unwrapping artefacts: no own vote) */
/* a subsample of about `target` vertices of a surface with q under both senses (qp, qm: wind_seed results, NaN = unassigned); *n
   points; nullptr on out of memory (*n 0: none) */
wind_pt *wind_seed_points(const float *xyz, const uint8_t *valid, int w, int h, void (*axis_fn)(void *, double, double *, double *), void *axis_ud,
                          const float *qp, const float *qm, int32_t surf, size_t target, size_t *n);
typedef struct {
    size_t agree, dis;          /* votes of pairs with this surface under the chosen (sense, order) */
    size_t own_pos, own_neg;    /* pairs inside it: its radius grows with phi (+) / falls (-): its own sense, whatever the set's */
    int own_dir;                /* +1 / -1 when own_pos + own_neg >= min_pairs and one side has >= 60% of them, else 0 */
    int dropped;                /* 1: its votes contradict the chosen (sense, order) (more disagree than agree, >= min_pairs) */
} wind_surf_vote;
typedef struct {
    int sense, order;           /* order -1: the names count inward (K -> -K) */
    double score;               /* agree / (agree + dis) under the choice, the dropped surfaces left out; NaN without votes */
    size_t agree, dis;
    double cand[2][2]; size_t cand_n[2][2];   /* [sense +1, -1][order +1, -1]: score and votes of each candidate (final pass) */
    int ndropped, evidence;     /* evidence 0: no votes at all, the sense is the fallback */
    double pitch;               /* scan voxels between consecutive sheets (median of pairs one turn apart); 0 unknown */
    int pitch_ok;               /* 0: the median pitch lies outside wind_pitch_lo .. hi: the votes are discarded (evidence 0) */
    double sense_margin, order_margin;   /* the choice's score minus the other sense's (other order's): under ~0.02 the geometry cannot
                                            tell (a set of partial wraps that never meet at the seam has no sense evidence) */
} wind_choice;
/* pts are sorted in place. K[surf]: the names (q under order -1 = q - 2 K). force_sense / force_order 0: free, else fixed. fallback_sense:
   the sense without votes (or on a tie). Surfaces whose votes contradict the choice are dropped one at a time, the worst first, and the
   choice made again without them. per: nsurf entries. Returns 0, -1 out of memory. */
int wind_seed_decide(wind_pt *pts, size_t n, const double *K, int nsurf, int cross, int force_sense, int force_order, int fallback_sense,
                     size_t min_pairs, wind_choice *ch, wind_surf_vote *per);
/* pitch_lo / pitch_hi (scan voxels; wind-seed: 50 / 1000 um): a median pitch outside them means the pairs are not sheets one turn
   apart (PHerc0009B: 838 voxels = 7 mm from phi unwrapping jumps of half-turn segments): no evidence at all (ch->evidence 0) */
extern double wind_pitch_lo, wind_pitch_hi;
/* a surface is dropped when its votes' agree share is under this (default 0.5: more disagree than agree) */
extern double wind_min_surface_score;

/* code -> q_ref (turns mod 14), NaN without a winding */
static inline float wind_code_q(uint8_t c, int kind) {
    if (kind == WIND_CODES) return c >= 1 && c <= 252 ? (float)(c - 1) / 18.f : __builtin_nanf("");
    return c < 252 ? (float)c / 18.f : __builtin_nanf("");
}
