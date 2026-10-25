/* Gauge-free pair winding accuracy (ufsm fit --score-pairs, the wbench metric).
   The box score "winding ids right" fits ONE gauge per box: one slip splits the box and the score jumps to the share of the larger
   piece. Pair accuracy instead asks, for pairs of labelled voxels about D apart, whether the fitted winding DIFFERENCE is right:
   |wrap14(d_u - d_v)| < 0.5 turn with d = U - q (mod 14) per voxel. A slip costs only the pairs that straddle it.
   Records are a hash subsample of the labelled voxels (pa_add keeps a record when its hash < the buffer's threshold, halving the
   threshold whenever the buffer is full): deterministic for a fixed thread split. pa_score samples pairs at distance D +-25 %
   (random record, random direction, a random record of the grid cell there). "cross" = pairs whose codes differ (different sheets). */
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct { float p[3], d; uint32_t h; uint8_t q; } pa_rec;
typedef struct { pa_rec *r; size_t n, cap; uint32_t T; } pa_buf;   /* zero-init + pa_init */

void pa_init(pa_buf *b, size_t cap);
void pa_free(pa_buf *b);
/* key: a unique id of the voxel (its global label index): the hash that decides the subsample */
void pa_add(pa_buf *b, float pz, float py, float px, float d, uint8_t q, uint64_t key);
/* moves src's records into dst (dst grows as needed) and keeps only those under the smaller threshold */
void pa_merge(pa_buf *dst, pa_buf *src);
/* pairs at each distance dist[k] (same units as the positions); acc / acc_cross (NaN when no pair), counts */
void pa_score(const pa_buf *b, const double *dist, int nd, size_t npairs, uint64_t seed, double *acc, double *acc_cross, size_t *n, size_t *n_cross);
