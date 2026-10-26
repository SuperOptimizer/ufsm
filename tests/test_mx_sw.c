/* Software MX paths of the sm_86 build (src/mx_sw.cuh, lp_common.cuh LP_SW), run in every build: all 2^32 fp32 inputs through
   the e4m3 / e2m1 encoders, all e4m3 pairs / e2m1 bytes through the decoders, and random fragments through the emulated
   block-scaled MMAs (mma_f8 / mma_f4 / mma_f4x4) against a scalar reference over the fragment layout. The sm_120a build also
   compares against the hardware instructions (bit-exact conversions; the MMAs to fp32 summation order), which is what pins the
   software paths to the Blackwell results; the ARCH=86 PTX build (run on a newer GPU) checks the code that a 3090 executes. */
#include <stdio.h>
int lp_mxsw_selftest(int *hw);
int main(void) {
    int hw = 0;
    const int f = lp_mxsw_selftest(&hw);
    printf("test_mx_sw (%s): %s\n", hw ? "software vs sm_120a instructions" : "software paths, no fp4 / fp8 instructions", f ? "FAIL" : "ok");
    return f ? 1 : 0;
}
