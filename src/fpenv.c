/* fpenv.c - host denormals flushed: FTZ (flush-to-zero results) and DAZ (denormal inputs read as zero) in MXCSR, set by a
 * constructor before main, so before any thread exists (POSIX: a new thread inherits its creator's floating-point environment).
 * What crtfastmath.o does for gcc -ffast-math links, done explicitly: clang only links that file from a gcc installation and not
 * with the compiler-rt runtime (UFSM_LLVM_RUNTIME). The GPU side already flushes (-use_fast_math). Linked into every executable
 * (CMakeLists.txt ufsm_exe) unless UFSM_STRICT_MATH=ON, which keeps IEEE denormals. DESIGN 2026-11-14 "Fast math everywhere". */
#if defined(__x86_64__) && !defined(UFSM_STRICT_MATH)
__attribute__((constructor)) static void ufsm_fpenv_ftz_daz(void) {
    unsigned csr;
    __asm__ volatile("stmxcsr %0" : "=m"(csr));
    csr |= 0x8040u;   /* bit 15 FTZ, bit 6 DAZ */
    __asm__ volatile("ldmxcsr %0" : : "m"(csr));
}
#else
typedef int ufsm_fpenv_empty;   /* (ISO C: no empty translation unit) */
#endif
