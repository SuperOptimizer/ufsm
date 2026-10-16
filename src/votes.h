/* Recto / verso side votes: one byte per voxel holding how many surface sets call it recto side (r) and how many verso side (v),
   0 <= r, v <= 15 (counts saturate). Several published surface sets of one scroll (different authors / runs) annotate overlapping
   wraps; rasterising each set's side labels (phase-raster --pair 2) and adding the counts keeps every set's say instead of letting the
   last one overwrite the others, and makes disagreement visible.
   Code: the diagonal bijection of the 16 x 16 (r, v) grid onto 0..255, by total n = r + v, then r ascending within a total:
     (0,0) 0 | (0,1) 1, (1,0) 2 | (0,2) 3, (1,1) 4, (2,0) 5 | ... | (15,15) 255
   so a single set's labels (band.h RV_UNKNOWN 0, RV_VERSO 1 = one verso vote, RV_RECTO 2 = one recto vote) ARE its vote array, byte
   for byte: every existing "side" array is a valid "side-votes" array. Small totals get small codes, so the lossless volcomp codec
   (prediction residuals, literal tokens up to 7) stores agreeing overlaps (0,2) / (2,0) about as compactly as single votes; a
   nibble layout (r << 4 | v) puts recto and verso 15 apart and costs bypass bits at every recto / verso face (bench_votes). */
#pragma once
#include <stdint.h>

enum { VOTE_MAX = 15 };

static inline uint8_t vote_code(int r, int v) {
    r = r < 0 ? 0 : r > VOTE_MAX ? VOTE_MAX : r; v = v < 0 ? 0 : v > VOTE_MAX ? VOTE_MAX : v;
    const int n = r + v;
    if (n <= VOTE_MAX) return (uint8_t)(n * (n + 1) / 2 + r);                        /* 136 codes for totals 0..15 */
    return (uint8_t)(136 + (47 - n) * (n - 16) / 2 + r - (n - VOTE_MAX));             /* totals 16..30: r >= n - 15 */
}
/* code -> (r, v): the total n is the diagonal holding the code (no tables: hot loops build their own 256-entry LUTs from this) */
static inline void vote_split(uint8_t code, int *r, int *v) {
    int n = 0, base = 0;
    for (;;) { const int len = n <= VOTE_MAX ? n + 1 : 2 * VOTE_MAX + 1 - n; if (code < base + len) break; base += len; n++; }
    const int rmin = n > VOTE_MAX ? n - VOTE_MAX : 0;
    *r = rmin + (code - base); *v = n - *r;
}
static inline int vote_r(uint8_t code) { int r, v; vote_split(code, &r, &v); return r; }
static inline int vote_v(uint8_t code) { int r, v; vote_split(code, &r, &v); return v; }
/* the sum of two vote codes (counts saturate at VOTE_MAX) */
static inline uint8_t vote_add(uint8_t a, uint8_t b) { int ra, va, rb, vb; vote_split(a, &ra, &va); vote_split(b, &rb, &vb); return vote_code(ra + rb, va + vb); }

/* Votes -> one side code (band.h RV_*: 0 unknown, 1 verso, 2 recto) under a rule:
     VOTE_UNANIMOUS (default)  recto iff r >= min and v == 0; verso iff v >= min and r == 0; any disagreement unknown (ignored)
     VOTE_MAJORITY             recto iff r > v and r >= min; verso iff v > r and v >= min; ties unknown
   min (>= 1): the votes a label needs (2: only voxels at least two sets agree on). Fills lut[256]. */
enum { VOTE_UNANIMOUS = 0, VOTE_MAJORITY = 1 };
static inline void vote_side_lut(int rule, int min, uint8_t lut[256]) {
    if (min < 1) min = 1;
    for (int c = 0; c < 256; c++) {
        const int r = vote_r((uint8_t)c), v = vote_v((uint8_t)c);
        uint8_t s = 0;
        if (rule == VOTE_MAJORITY) s = r > v && r >= min ? 2 : v > r && v >= min ? 1 : 0;
        else s = r >= min && !v ? 2 : v >= min && !r ? 1 : 0;
        lut[c] = s;
    }
}
