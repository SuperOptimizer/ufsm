#pragma once
/* gnmode.h (fix0332 2026-11-18): which GroupNorm statistics a gn_ema checkpoint's inference uses. A checkpoint whose --gn-ema-warm was
   never reached (ge_n < ge_warm in its header) trained only with each window's own statistics: normalising it with the running ones
   (predict's old default) ran try9's whole-scroll 1667 unroll on a function the net never saw (DESIGN 2026-11-18 "gn1667"). */
/* gn_live_choice: hdr the checkpoint's header line (or nullptr), arg the --gn-live value (nullptr: not given), gn_ema / frozen the net's
   flag and whether --gn-frozen was given. Returns 1 = live (window) statistics, 0 = running; *defaulted 1 when the choice was made for an
   unwarmed checkpoint without a --gn-live (the caller prints the DEFAULTED line), *n / *w the header's ge_n / ge_warm (-1: absent). */
int gn_live_choice(const char *hdr, const char *arg, int gn_ema, int frozen, int *defaulted, long *n, long *w);
/* gn_live_choice_n: the same rule on the numbers (train's validation: the net's ge_n / ge_warm at that step; arg the --val-gn-live value) */
int gn_live_choice_n(const char *arg, int gn_ema, int frozen, long n, long w, int *defaulted);
