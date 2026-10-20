#include "phase_check.h"
#include "band.h"
#include <math.h>
#include <stdlib.h>

static inline int wrap252(int d) { d %= PHASE_PERIOD; if (d >= PHASE_PERIOD / 2) d -= PHASE_PERIOD; if (d < -PHASE_PERIOD / 2) d += PHASE_PERIOD; return d; }
void phase_outward_count(const uint8_t *p, const int n[3], int64_t y0, int64_t x0, const double *cy, const double *cx, size_t *up, size_t *tot) {
    for (int z = 0; z < n[0]; z++) for (int y = 0; y < n[1]; y++) for (int x = 0; x < n[2]; x++) {
        const size_t i = ((size_t)z * n[1] + y) * n[2] + x; const int v = p[i];
        if (v >= PHASE_PERIOD) continue;
        double gy = 0, gx = 0; int m = 0;
        if (y + 1 < n[1]) { const int w = p[i + n[2]]; if (w < PHASE_PERIOD) { const int d = wrap252(w - v); if (abs(d) < 60) { gy = d; m = 1; } } }
        if (x + 1 < n[2]) { const int w = p[i + 1]; if (w < PHASE_PERIOD) { const int d = wrap252(w - v); if (abs(d) < 60) { gx = d; m = 1; } } }
        if (!m) continue;
        const double Y = (double)(y0 + y) - cy[z], X = (double)(x0 + x) - cx[z], R = sqrt(Y * Y + X * X) + 1e-6, gr = (gy * Y + gx * X) / R;
        if (fabs(gr) <= 0.5) continue;
        (*tot)++; if (gr > 0) (*up)++;
    }
}
size_t phase_mask_ref(uint8_t *out, const uint8_t *ref, size_t n, double tol, size_t hist[PHASE_AGREE_BINS], size_t *masked) {
    size_t both = 0;
    for (size_t i = 0; i < n; i++) {
        if (out[i] >= PHASE_PERIOD || ref[i] >= PHASE_PERIOD) continue;
        both++;
        const double d = fabs((double)wrap252((int)out[i] - (int)ref[i])) / PHASE_PERIOD;
        hist[phase_agree_bin(d)]++;
        if (d > tol) { out[i] = PHASE_UNKNOWN; (*masked)++; }
    }
    return both;
}
