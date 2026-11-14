/* test_band_pair: the lean pair-rule fields (src/band.c, 2026-11-11: feature indices only, the squared distances rebuilt from them)
   and the GPU pair rule (src/nn_band.cu, when a GPU is there) against the previous band.c code kept here verbatim as the reference
   (float distance arrays): band_field_phase_pair and band_field_side_pair byte-identical on synthetic spirals of nested sheets with
   holes, conflicts, a missing wrap, non-cubic boxes and an empty one. */
#include "../src/band.h"
#include "../src/bandgpu.h"
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define BINF 1e20f
/* ---- reference: band.c before the lean change (git 1320eca), renamed ---- */
static void ref_envelope(const float *f, const int32_t *src, int n, float *d, int32_t *s, int *v, float *z) {
    int k = 0; v[0] = -1;
    for (int q = 0; q < n; q++) {
        if (f[q] >= BINF) continue;
        if (v[0] < 0) { v[0] = q; z[0] = -BINF; z[1] = BINF; k = 0; continue; }
        float sct;
        for (;;) {
            const int r = v[k];
            sct = ((f[q] + (float)q * q) - (f[r] + (float)r * r)) / (2.f * (q - r));
            if (sct <= z[k] && k > 0) k--; else break;
        }
        if (sct <= z[k]) { v[k] = q; z[k + 1] = BINF; }   /* k == 0 and the new parabola dominates */
        else { k++; v[k] = q; z[k] = sct; z[k + 1] = BINF; }
    }
    if (v[0] < 0) { for (int q = 0; q < n; q++) { d[q] = BINF; s[q] = -1; } return; }
    k = 0;
    for (int q = 0; q < n; q++) {
        while (z[k + 1] < q) k++;
        const int r = v[k]; d[q] = (float)(q - r) * (q - r) + f[r]; s[q] = src[r];
    }
}

enum { ET = 64 };

/* cap2: squared distances over it are dropped between the passes (every result within cap2 is exact; the rest may come out unknown) */
static int ref_edt_feature_cap(const uint8_t *seed, const int n[3], float cap2, float *d2, int32_t *idx) {
    const int nz = n[0], ny = n[1], nx = n[2], m = nz > ny ? (nz > nx ? nz : nx) : (ny > nx ? ny : nx);
    float *f = malloc((size_t)m * ET * 4), *dl = malloc((size_t)m * 4), *z = malloc((size_t)(m + 1) * 4);
    int32_t *sr = malloc((size_t)m * ET * 4), *sl = malloc((size_t)m * 4); int *v = malloc((size_t)m * sizeof(int));
    if (!f || !dl || !z || !sr || !sl || !v) { free(f); free(dl); free(z); free(sr); free(sl); free(v); return -1; }
    const size_t sy = (size_t)nx, sz = (size_t)ny * nx;
    for (size_t li = 0; li < (size_t)nz * ny; li++) {   /* x: the nearest seed of the line by two sweeps (equidistant: the left one, as the envelope) */
        const uint8_t *sd = seed + li * sy; float *dr = d2 + li * sy; int32_t *ir = idx + li * sy;
        int last = -1;
        for (int q = 0; q < nx; q++) { if (sd[q]) last = q; ir[q] = last; }
        int next = -1;
        for (int q = nx - 1; q >= 0; q--) {
            if (sd[q]) next = q;
            const int l = ir[q], best = l < 0 ? next : next < 0 ? l : (q - l <= next - q ? l : next);
            if (best < 0) { dr[q] = BINF; ir[q] = -1; continue; }
            const float d = (float)(q - best) * (float)(q - best);
            if (d > cap2) { dr[q] = BINF; ir[q] = -1; } else { dr[q] = d; ir[q] = (int32_t)(li * sy + (size_t)best); }
        }
    }
    for (int ax = 1; ax < 3; ax++) {   /* y, then z: tiles of ET x lines */
        const int L = ax == 1 ? ny : nz; const size_t st = ax == 1 ? sy : sz;
        const size_t nouter = ax == 1 ? (size_t)nz : (size_t)ny;   /* y: per z plane; z: per y row */
        for (size_t o = 0; o < nouter; o++) {
            const size_t base0 = ax == 1 ? o * sz : o * sy;
            for (int x0 = 0; x0 < nx; x0 += ET) {
                const int w = nx - x0 < ET ? nx - x0 : ET;
                for (int q = 0; q < L; q++) {   /* gather: column t of the tile is line x0 + t, stored contiguously at t * L */
                    const float *dp = d2 + base0 + q * st + x0; const int32_t *ip = idx + base0 + q * st + x0;
                    for (int t = 0; t < w; t++) { const float fv = dp[t]; f[(size_t)t * L + q] = fv > cap2 ? BINF : fv; sr[(size_t)t * L + q] = ip[t]; }
                }
                for (int t = 0; t < w; t++) {
                    ref_envelope(f + (size_t)t * L, sr + (size_t)t * L, L, dl, sl, v, z);
                    memcpy(f + (size_t)t * L, dl, (size_t)L * 4); memcpy(sr + (size_t)t * L, sl, (size_t)L * 4);
                }
                for (int q = 0; q < L; q++) {   /* scatter */
                    float *dp = d2 + base0 + q * st + x0; int32_t *ip = idx + base0 + q * st + x0;
                    for (int t = 0; t < w; t++) { dp[t] = f[(size_t)t * L + q]; ip[t] = sr[(size_t)t * L + q]; }
                }
            }
        }
    }
    free(f); free(dl); free(z); free(sr); free(sl); free(v);
    return 0;
}
static int ref_side_pair(const uint8_t *codes, const int n[3], float rmax, int margin, uint8_t *side) {
    const size_t N = (size_t)n[0] * n[1] * n[2], sy = (size_t)n[2], sz = (size_t)n[1] * n[2];
    uint8_t *seed = malloc(N); float *d2 = malloc(N * 4); int32_t *id[4] = {malloc(N * 4), malloc(N * 4), malloc(N * 4), malloc(N * 4)};
    int rc = !seed || !d2 || !id[0] || !id[1] || !id[2] || !id[3];
    for (int q = 0; q < 4 && !rc; q++) {   /* the labelled voxels by winding mod 4 (code 1 + 18 q mod 252: k = (code - 1) / 18): k, k +- 1 and
                                              k +- 2 fall in different classes (but across the 14-turn wrap of the codes, where k + 2 may share k's) */
        for (size_t i = 0; i < N; i++) { const int c = codes[i]; seed[i] = c > 0 && c < 255 && (((c - 1) / BAND_STEPS) & 3) == q; }
        rc = ref_edt_feature_cap(seed, n, 4 * rmax * rmax, d2, id[q]);   /* nothing beyond the bound limit is used */
    }
    if (!rc) {
        const float r2 = rmax * rmax, o2 = 4 * r2;
        const double isz = 1.0 / (double)sz, isy = 1.0 / (double)sy;   /* feature index -> (z, y, x) by reciprocal multiply, corrected */
        memset(side, RV_UNKNOWN, N);
        for (long z = margin; z < n[0] - margin; z++) for (long y = margin; y < n[1] - margin; y++) for (long x = margin; x < n[2] - margin; x++) {
            const size_t i = ((size_t)z * n[1] + (size_t)y) * n[2] + (size_t)x;
            const int c = codes[i];
            if (c > 0 && c < 255) { side[i] = RV_RECTO; continue; }   /* on a labelled recto face */
            float v[4][3], dq[4]; int best = -1;
            for (int q = 0; q < 4; q++) {
                dq[q] = INFINITY; if (id[q][i] < 0) continue;
                const size_t f = (size_t)id[q][i];
                size_t fz = (size_t)((double)f * isz); if (fz * sz > f) fz--; else if ((fz + 1) * sz <= f) fz++;
                const size_t r = f - fz * sz; size_t fy = (size_t)((double)r * isy); if (fy * sy > r) fy--; else if ((fy + 1) * sy <= r) fy++;
                v[q][0] = (float)((long)fz - z); v[q][1] = (float)((long)fy - y); v[q][2] = (float)((long)(r - fy * sy) - x);
                dq[q] = v[q][0] * v[q][0] + v[q][1] * v[q][1] + v[q][2] * v[q][2];
                if (best < 0 || dq[q] < dq[best]) best = q;
            }
            if (best < 0 || dq[best] > r2) continue;
            const int ca = codes[id[best][i]];   /* the nearest wrap: winding k */
            int bm = 0, sm = 0, sq = -1; float bd = INFINITY, sd = INFINITY;   /* the nearest other wrap in the opposite / the same direction, its winding step */
            for (int q = 0; q < 4; q++) {
                if (q == best || dq[q] > o2) continue;
                const int dc = band_diff(codes[id[q][i]], ca), m = (int)lrintf((float)dc / BAND_STEPS);
                if (!m || abs(dc - m * BAND_STEPS) > 5) continue;   /* the same sheet (codes across a turn boundary) or off a whole turn */
                const float dot = v[q][0] * v[best][0] + v[q][1] * v[best][1] + v[q][2] * v[best][2];
                if (dot < 0) { if (dq[q] < bd) { bd = dq[q]; bm = m; } }
                else if (dq[q] < sd) { sd = dq[q]; sm = m; sq = q; }
            }
            if (bm) {   /* between k and k + m: the winding position k + m d_k / (d_k + d_b); recto in the first half of every winding */
                const float da = sqrtf(dq[best]), db = sqrtf(bd), w = (float)bm * da / (da + db);
                side[i] = w - floorf(w) < 0.5f ? RV_RECTO : RV_VERSO;
            } else if (sm) {   /* nothing on its side within reach (a hole, the segmentation's edge): the side of k away from its neighbours,
                                  trusted within half the local spacing (k to that neighbour); farther, the unseen next wrap may be nearer */
                const float ez = v[sq][0] - v[best][0], ey = v[sq][1] - v[best][1], ex = v[sq][2] - v[best][2];
                if (4 * dq[best] <= ez * ez + ey * ey + ex * ex) side[i] = sm > 0 ? RV_VERSO : RV_RECTO;
            }
        }
    }
    free(seed); free(d2); for (int q = 0; q < 4; q++) free(id[q]);
    return rc ? -1 : 0;
}

static int ref_pair_ex(const uint8_t *codes, const int n[3], float rmax, uint8_t *phase, uint8_t *spacing, int32_t *lo, int32_t *hi) {
    const size_t N = (size_t)n[0] * n[1] * n[2], sy = (size_t)n[2], sz = (size_t)n[1] * n[2];
    uint8_t *se = malloc(N), *so = malloc(N); float *de = malloc(N * 4), *dd = malloc(N * 4); int32_t *ie = malloc(N * 4), *io = malloc(N * 4);
    int rc = !se || !so || !de || !dd || !ie || !io;
    if (!rc) {
        for (size_t i = 0; i < N; i++) {   /* the labelled voxels by the parity of their winding (code 1 + 18 q mod 252: k = (code - 1) / 18) */
            const int c = codes[i], lab = c > 0 && c < 255;
            const int k = lab ? (c - 1) / BAND_STEPS : 0;
            se[i] = lab && !(k & 1); so[i] = lab && (k & 1);
        }
        rc = ref_edt_feature_cap(se, n, BINF, de, ie) || ref_edt_feature_cap(so, n, BINF, dd, io);
    }
    if (!rc) {
        const float r2 = rmax * rmax;
        for (size_t i = 0; i < N; i++) {
            phase[i] = PHASE_UNKNOWN; if (spacing) spacing[i] = 0;
            if (lo) { lo[i] = -1; hi[i] = -1; }
            const int c = codes[i];
            if (c > 0 && c < 255) { phase[i] = 0; if (lo) { lo[i] = (int32_t)i; hi[i] = (int32_t)i; } continue; }   /* on a labelled recto */
            if (ie[i] < 0 || io[i] < 0 || de[i] > r2 || dd[i] > r2) continue;
            const size_t a = (size_t)ie[i], b = (size_t)io[i];
            const int ka = (codes[a] - 1) / BAND_STEPS, kb = (codes[b] - 1) / BAND_STEPS;   /* windings mod 14 */
            int dk = ((kb - ka) % 14 + 14) % 14; if (dk > 7) dk -= 14;
            if (dk != 1 && dk != -1) continue;   /* not consecutive: a missing wrap between */
            const long z = (long)(i / sz), y = (long)((i / sy) % n[1]), x = (long)(i % sy);
            const float az = (float)((long)(a / sz) - z), ay = (float)((long)((a / sy) % n[1]) - y), ax = (float)((long)(a % sy) - x);
            const float bz = (float)((long)(b / sz) - z), by = (float)((long)((b / sy) % n[1]) - y), bx = (float)((long)(b % sy) - x);
            const float da = sqrtf(de[i]), db = sqrtf(dd[i]);
            if (az * bz + ay * by + ax * bx > -0.5f * da * db) continue;   /* both sheets on one side: not between them */
            const float f = dk == 1 ? da / (da + db) : db / (da + db);   /* the way from the lower winding to the higher */
            int q = (int)lrintf(f * PHASE_PERIOD); if (q >= PHASE_PERIOD) q -= PHASE_PERIOD;
            phase[i] = (uint8_t)q;
            if (lo) { lo[i] = (int32_t)(dk == 1 ? a : b); hi[i] = (int32_t)(dk == 1 ? b : a); }
            if (spacing) { const long sp = lrintf(da + db); spacing[i] = (uint8_t)(sp < 1 ? 1 : sp > 255 ? 255 : sp); }
        }
    }
    free(se); free(so); free(de); free(dd); free(ie); free(io);
    return rc ? -1 : 0;
}


/* ---- synthetic codes: a spiral of sheets (winding k at radius r0 + k * sp + drift), codes 1 + 18 k mod 252, holes, conflicts ---- */
static void make_codes(uint8_t *c, const int n[3], unsigned seed, int empty) {
    srand(seed); memset(c, 0, (size_t)n[0] * n[1] * n[2]);
    if (empty) return;
    const double cy = n[1] * 0.45 + rand() % 7, cx = n[2] * 0.55 - rand() % 5, sp = 5.5 + (rand() % 30) / 10.0, r0 = 3 + rand() % 4;
    const int skip = 3 + rand() % 4;   /* one missing wrap */
    for (int z = 0; z < n[0]; z++) for (int y = 0; y < n[1]; y++) for (int x = 0; x < n[2]; x++) {
        const double dy = y - cy, dx = x - cx, r = sqrt(dy * dy + dx * dx), th = atan2(dy, dx), u = (r - r0 - sp * th / (2 * M_PI) - 0.04 * z) / sp;
        const int k = (int)floor(u + 0.5); if (k < 0 || k == skip) continue;
        if (fabs(u - k) * sp > 0.55) continue;
        if (((x / 9 + y / 7 + z / 11) % 13) == 0) continue;   /* holes */
        c[((size_t)z * n[1] + y) * n[2] + x] = (rand() % 997 == 0) ? 255 : (uint8_t)(1 + (BAND_STEPS * k) % BAND_PERIOD);
    }
}
int main(void) {
    const int dims[][3] = {{40, 64, 72}, {57, 50, 61}, {33, 33, 33}, {24, 90, 40}, {16, 16, 16}};
    const int ncase = sizeof dims / sizeof dims[0];
    int fails = 0, gpu = band_pair_gpu != nullptr, gpu_ran = 0;
    int ndev = 0; if (cudaGetDeviceCount(&ndev) != cudaSuccess) ndev = 0;   /* a device that does not answer is a failure; no device at all (a CPU run) a skip */
    if (gpu) band_pair_gpu_slots(2, 0);
    for (int c = 0; c < ncase; c++) for (int rm = 0; rm < 2; rm++) {
        const int *n = dims[c]; const size_t N = (size_t)n[0] * n[1] * n[2]; const float rmax = rm ? 14.f : 6.5f;
        uint8_t *codes = malloc(N), *a = malloc(N), *b = malloc(N), *g = malloc(N);
        make_codes(codes, n, 11u + 7u * c + rm, c == ncase - 1);
        size_t lab = 0; for (size_t i = 0; i < N; i++) lab += codes[i] > 0 && codes[i] < 255;
        if (ref_pair_ex(codes, n, rmax, a, nullptr, nullptr, nullptr) || band_field_phase_pair(codes, n, rmax, b, nullptr)) { printf("FAIL alloc\n"); return 1; }
        size_t known = 0, d = 0; for (size_t i = 0; i < N; i++) { known += a[i] < PHASE_PERIOD; d += a[i] != b[i]; }
        printf("case %d (%dx%dx%d, rmax %.1f): %zu labelled, %zu phase known, lean CPU differs at %zu", c, n[0], n[1], n[2], rmax, lab, known, d);
        fails += d != 0;
        if (gpu) {
            if (band_pair_gpu(codes, n, rmax, g)) { printf(", GPU unavailable%s", ndev ? " with a device visible: FAIL" : " (no device: skipped)"); fails += ndev > 0; }
            else { size_t e = 0; for (size_t i = 0; i < N; i++) e += a[i] != g[i]; printf(", GPU differs at %zu", e); fails += e != 0; gpu_ran = 1; }
        }
        const int margin = 2;
        if (ref_side_pair(codes, n, rmax, margin, a) || band_field_side_pair(codes, n, rmax, margin, b)) { printf("FAIL alloc\n"); return 1; }
        size_t ds = 0, ks = 0; for (size_t i = 0; i < N; i++) { ds += a[i] != b[i]; ks += a[i] != RV_UNKNOWN; }
        printf(", side known %zu differs at %zu\n", ks, ds); fails += ds != 0;
        if (c < ncase - 1 && (known == 0 || ks == 0)) { printf("FAIL: a case without known output\n"); fails++; }
        free(codes); free(a); free(b); free(g);
    }
    if (gpu) band_pair_gpu_free();
    printf("%s (GPU %s)\n", fails ? "FAIL" : "PASS", gpu_ran ? "checked" : ndev ? "not checked" : "not checked: no device");
    return fails != 0;
}
