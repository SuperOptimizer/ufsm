#include "gnmode.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

static long hdr_long(const char *hdr, const char *key) {
    const char *p = hdr ? strstr(hdr, key) : nullptr; return p ? atol(p + strlen(key)) : -1;
}
int gn_live_choice(const char *hdr, const char *arg, int gn_ema, int frozen, int *defaulted, long *n, long *w) {
    *n = hdr_long(hdr, "\"ge_n\":"); *w = hdr_long(hdr, "\"ge_warm\":");
    return gn_live_choice_n(arg, gn_ema, frozen, *n, *w, defaulted);
}
int gn_live_choice_n(const char *arg, int gn_ema, int frozen, long n, long w, int *defaulted) {
    *defaulted = 0;
    if (arg) return atoi(arg) != 0;
    if (!gn_ema || frozen) return 0;
    if (n >= 0 && w >= 0 && n < w) { *defaulted = 1; return 1; }
    return 0;
}
int ema_choice(const char *arg, double rel, int *defaulted) {
    *defaulted = 0;
    if (arg && strcmp(arg, "auto")) return atoi(arg) != 0;
    if (rel > EMA_REL_MAX) { *defaulted = 1; return 0; }
    return 1;
}

float ema_update_decay(float ema, int per_window, long wpu, long ostep) {
    if (ostep < 100) return 0.9f;
    if (!per_window || wpu <= 1) return ema;
    return (float)pow((double)ema, (double)wpu);
}
