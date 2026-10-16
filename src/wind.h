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
   q = K + dir (phi - phi_start), phi = theta / 2 pi unwrapped along the mesh grid as in wind_assign. The start is the end of the
   grid (first or last 5% of its columns) nearer the umbilicus (the authors' unrollings run along columns: a single-wrap segment
   wNNN covers one turn from its seam, a merged spiral starts at its innermost wrap), dir the sign that makes q grow away from it.
   Other components: unassigned (flag 4, NaN). info: r_start / r_end (mean radius of the two column bands, scan voxels), span
   (q_end - q_start, turns), dir. Returns 0, -1 out of memory, -2 no valid vertex. */
typedef struct { int dir; double r_start, r_end, span; size_t nvalid, nassigned; } wind_seed_info;
int wind_seed(const float *xyz, const uint8_t *valid, int w, int h, void (*axis_fn)(void *, double, double *, double *), void *axis_ud,
              double K, wind_seed_info *info, wind_result *out);

/* code -> q_ref (turns mod 14), NaN without a winding */
static inline float wind_code_q(uint8_t c, int kind) {
    if (kind == WIND_CODES) return c >= 1 && c <= 252 ? (float)(c - 1) / 18.f : __builtin_nanf("");
    return c < 252 ? (float)c / 18.f : __builtin_nanf("");
}
