/* Least-cost integer potentials (fitter Phase 4, local slip repair): nodes carry integer shifts s, every vote asks
   s_b - s_a = d with a weight w, and the solver minimises the summed weighted L1 disagreement
       E(s) = sum over votes w |s_b - s_a - d|
   exactly (the energy is convex in each pairwise difference, so moves of +-1 on a node subset chosen by a minimum cut reach the
   global minimum: Bioucas-Dias and Valadao, "Phase unwrapping via graph cuts" (PUMA), IEEE TIP 2007). Unlike a spanning tree
   (the voxel flood), every vote counts: a loop whose offsets do not close is cut where the cut is cheapest, not where the tree
   happened to close it. */
#pragma once
#include <stddef.h>
#include <stdint.h>

/* s_b - s_a = d wanted (a != b): residual r = s_b - s_a - d costs w r when r >= 0 and wn (-r) when r < 0; wn 0 means wn = w (the symmetric
   vote, every caller before 11-18). A one-sided vote (a lower bound: a ray that can miss a faint sheet but not invent one, HerculeX's rays)
   is d with w = W / 2 and wn = W (src/l1pot.c, levelsolve 2026-11-18). */
typedef struct { uint32_t a, b; int32_t d; int64_t w, wn; } l1_vote;
typedef struct { int iters, moves, capped; int64_t e0, e1; size_t nedge, nmoved; double secs; uint64_t work; } l1_stats;   /* capped: the work budget stopped the moves (work: max-flow node / arc visits) */

/* sort the votes by (a, b, d) with a < b (a vote with a > b is flipped: d and the two weights swap sides) and merge equal ones (weights
   summed; wn set to w where it was 0); returns the new count */
size_t l1_votes_merge(l1_vote *v, size_t nv);
/* minimise E from the shifts in s (n nodes, updated in place; votes merged by l1_votes_merge). max_iter bounds the moves (each a
   max flow). Returns 0, or -1 on allocation failure (s unchanged). */
int l1_solve(uint32_t n, const l1_vote *v, size_t nv, int32_t *s, int max_iter, l1_stats *st);
/* the max-flow work budget of one l1_solve (node / arc visits summed over its moves; 0: none): when a move ends past it the moves stop
   there (stats capped = 1; the shifts are those of the last move, every move having lowered E). Default max(1e9, 500 x (nodes + edges))
   (~15-20 max flows; wind96 slabs use <= 360 x); env UFSM_L1_WORK overrides (an absolute count, 0: none). Moves under it run as before, so solves under the budget are unchanged bit for bit. */
void l1_set_work_budget(uint64_t w);
/* E(s) of merged votes */
int64_t l1_energy(const l1_vote *v, size_t nv, const int32_t *s);
