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
   sense (turns, [0, 1); *resultant: the mean resultant length, 1 = all starts on one azimuth). */
void wind_seed_set(const wind_seed_info *in, int n, int *sense, double *phi0, double vote[2], double *resultant);

/* code -> q_ref (turns mod 14), NaN without a winding */
static inline float wind_code_q(uint8_t c, int kind) {
    if (kind == WIND_CODES) return c >= 1 && c <= 252 ? (float)(c - 1) / 18.f : __builtin_nanf("");
    return c < 252 ? (float)c / 18.f : __builtin_nanf("");
}
