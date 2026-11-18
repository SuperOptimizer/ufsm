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
/* ema_choice: whether inference / validation uses the EMA weights. arg the --ema (--val-live inverted by the caller) value: "0" / "1" win,
   nullptr or "auto" = the EMA unless rel = |ema - p| / |p| > EMA_REL_MAX (then the live weights, *defaulted 1: the caller prints DEFAULTED). */
#define EMA_REL_MAX 0.08
int ema_choice(const char *arg, double rel, int *defaulted);
/* ema_update_decay (fix0332 2026-11-18, per-window EMA): the decay one optimizer update applies to the training EMA. per_window 0: ema
   itself (one EMA step per update: the horizon 1/(1-ema) updates = accum x windows-per-step times that in windows, ~10 000 windows for
   0.999 at --accum 10, longer than pab-A's lineage: |ema - p| / |p| 0.117, EMA GT ids 0.798 vs live 0.841 on 1667); per_window 1:
   ema^wpu, wpu the windows the update consumed, so the horizon is 1/(1-ema) windows whatever --accum (auto) does. Warmup: 0.9 for the
   first 100 updates (ostep < 100) either way, as before. */
float ema_update_decay(float ema, int per_window, long wpu, long ostep);
