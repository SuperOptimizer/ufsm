/* vt_multiview (src/vwt.h): villa multiview_consistency_loss, one-sided (slab B's phase is a constant target), on the host. */
#include "vwt.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static double huber_d(double r, double d) { return fabs(r) <= d ? r : (r > 0 ? d : -d); }
static double huber(double r, double d) { const double a = fabs(r); return a <= d ? 0.5 * r * r : d * (a - 0.5 * d); }

double vt_multiview(const float *pha, const float *phb, const uint8_t *va, const uint8_t *vb, const double fa[4][3], const double fb[4][3], int T, int L,
                    int st, float delta, float lam, float scale, float *gph, long *nshared) {
    /* fa / fb: o, a, b, d (xyz); voxel (i, j, k) of A at o + i a + j b + k d */
    const int ni = (T + st - 1) / st; const size_t cap = (size_t)ni * ni * L;
    float *r = malloc(cap * 4); uint32_t *ix = malloc(cap * 4); size_t n = 0;
    for (int i = 0; i < T; i += st) for (int j = 0; j < T; j += st) for (int k = 0; k < L; k++) {
        const size_t qa = ((size_t)i * T + j) * L + k; if (!va[qa] || !isfinite(pha[qa])) continue;
        double w[3], q[3]; for (int e = 0; e < 3; e++) w[e] = fa[0][e] + i * fa[1][e] + j * fa[2][e] + k * fa[3][e] - fb[0][e];
        for (int e = 0; e < 3; e++) q[e] = w[0] * fb[1 + e][0] + w[1] * fb[1 + e][1] + w[2] * fb[1 + e][2];   /* (i, j, k) in B */
        if (!(q[0] >= 0 && q[1] >= 0 && q[2] >= 0 && q[0] <= T - 1 && q[1] <= T - 1 && q[2] <= L - 1)) continue;
        int a0 = (int)q[0], b0 = (int)q[1], c0 = (int)q[2]; if (a0 > T - 2) a0 = T - 2; if (b0 > T - 2) b0 = T - 2; if (c0 > L - 2) c0 = L - 2;
        const double fa_ = q[0] - a0, fb_ = q[1] - b0, fc = q[2] - c0; double s = 0; int ok = 1;
        for (int c = 0; c < 8 && ok; c++) {
            const int da = c & 1, db = c >> 1 & 1, dc = c >> 2 & 1; const size_t qb = ((size_t)(a0 + da) * T + b0 + db) * L + c0 + dc;
            if (!vb[qb] || !isfinite(phb[qb])) { ok = 0; break; }
            s += (da ? fa_ : 1 - fa_) * (db ? fb_ : 1 - fb_) * (dc ? fc : 1 - fc) * phb[qb];
        }
        if (!ok) continue;
        r[n] = (float)(pha[qa] - s); ix[n] = (uint32_t)qa; n++;
    }
    *nshared = (long)n; double loss = 0;
    if (n >= 64) {
        double mr = 0; for (size_t t = 0; t < n; t++) mr += r[t]; mr /= n;
        double mg = 0; for (size_t t = 0; t < n; t++) { loss += huber(r[t] - mr, delta); mg += huber_d(r[t] - mr, delta); } mg /= n; loss /= n;
        if (gph) for (size_t t = 0; t < n; t++) gph[ix[t]] += (float)(lam * scale * (huber_d(r[t] - mr, delta) - mg) / n);
    }
    free(r); free(ix);
    return n >= 64 ? loss : 0;
}
