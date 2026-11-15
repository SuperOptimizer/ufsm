/* test_hold_frame: the held-out audit (UFSM_HOLD_AUDIT) reads a non-warped window's host planes in the output frame (the host symmetry
   applied with the masked boxes), the ink known bits on the source grid. The 2026-11-15 try7 audit read every plane on the source grid and
   reported PHercParis4-u2ink level-2 windows with symmetry 021/000 and 012/010 as leaks; their masked boxes were where the symmetry put them.
   sample_hold_frame_selftest checks the mapping against sym_u8 for all 48 cube symmetries, with and without a planted leak. */
#include <stdio.h>
#include "../src/sample.h"
int main(void) {
    if (sample_hold_frame_selftest()) { printf("test_hold_frame: FAIL\n"); return 1; }
    printf("test_hold_frame: ok (48 symmetries: the masked box read through the symmetry is unknown, a planted voxel is found)\n");
    return 0;
}
