/* ufsm ink-blend core (src/inkblend.c), also predict --seamless (src/seamless.c) */
#pragma once
#include <stdint.h>
int ink_blend_run(const char *out, double um, const double st[3], int n, const char *const *in, const double (*g)[3], const int64_t (*org)[3],
                  int nlev, int S, double q, int nt, const char *attrs);
/* predict --seamless 2's region for the next ink_blend_run (perf 11-26): the voxels the user asked for (lo..hi, output voxels; tiles:
   ntl "z y x" tile indices of tls voxels from tlo, sorted by ink_blend_set_region; ntl 0 = no tile list) and each input's computed box
   (plo..phi). In block mode a block outside the region is written 0 (no input), inside it input i counts wherever its computed box
   holds the block, an absent shard there read as its zeros: a pass writes no all-zero shard, and a window that wrote all its 2^3
   shards (beyond the tile list or the box) is not part of the region. nullptr: off. Not thread-safe (one blend at a time). */
typedef struct { int64_t lo[3], hi[3], plo[8][3], phi[8][3]; long long *tl; long ntl; int64_t tls, tlo[3]; } ink_blend_region;
void ink_blend_set_region(ink_blend_region *r);
