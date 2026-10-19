/* Band field from a winding_mod14 raster (see band.h). Exact Euclidean feature transform (Felzenszwalb-Huttenlocher lower
   envelopes, separable over x, y, z) carrying the index of the nearest surface voxel. */
#include "band.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define BINF 1e20f

/* 1-D squared-distance lower envelope along one line: f[i] (cost), src[i] (index) -> d[i], s[i] */
static void envelope(const float *f, const int32_t *src, int n, float *d, int32_t *s, int *v, float *z) {
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

/* squared distance (label voxels) and flat index of the nearest seed voxel; idx -1 / d BINF where there is none. The y and z passes
   run on tiles of ET consecutive x lines (rows of ET contiguous values gathered, one envelope per column, scattered back): one
   strided line at a time missed the cache on every element (5x slower); the envelopes and so the result are the same. */
enum { ET = 64 };
/* cap2: squared distances over it are dropped between the passes (every result within cap2 is exact; the rest may come out unknown) */
static int edt_feature_cap(const uint8_t *seed, const int n[3], float cap2, float *d2, int32_t *idx) {
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
                    envelope(f + (size_t)t * L, sr + (size_t)t * L, L, dl, sl, v, z);
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
static int edt_feature(const uint8_t *seed, const int n[3], float *d2, int32_t *idx) { return edt_feature_cap(seed, n, BINF, d2, idx); }

/* band and side; d2k / idxk (optional, N each): keep the squared distance and index of the nearest labelled recto voxel */
static int band_impl(const uint8_t *codes, const int n[3], const int64_t o[3], const double *cy, const double *cx, band_params p, uint8_t *out,
                     uint8_t *side, float *d2k, int32_t *idxk) {
    const size_t N = (size_t)n[0] * n[1] * n[2], sy = (size_t)n[2], sz = (size_t)n[1] * n[2];
    uint8_t *seed = calloc(N, 1), *flag = malloc(N);   /* calloc: gcc cannot see that the loop below fills it */
    float *d2 = d2k ? d2k : malloc(N * 4); int32_t *idx = idxk ? idxk : malloc(N * 4);
#define BAND_FREE() do { free(seed); free(flag); if (!d2k) free(d2); if (!idxk) free(idx); } while (0)
    if (!seed || !flag || !d2 || !idx) { BAND_FREE(); return -1; }
    for (size_t i = 0; i < N; i++) seed[i] = codes[i] > 0 && codes[i] < 255;
    if (edt_feature(seed, n, d2, idx)) { BAND_FREE(); return -1; }
    /* band = nearest surface winding +- half a turn; unknown beyond the flood radius */
    const float g = band_gs(p), r2 = (p.radius / g) * (p.radius / g), c0 = 0.5f * (g - 1.f);   /* label voxel i sits at g i + (g - 1) / 2 native */
    for (size_t i = 0; i < N; i++) {
        if (idx[i] < 0 || d2[i] > r2) { out[i] = BAND_UNKNOWN; if (side) side[i] = SIDE_UNKNOWN; continue; }
        const size_t s = (size_t)idx[i];
        const int z = (int)(i / sz), y = (int)((i / sy) % n[1]), x = (int)(i % sy);
        const int zs = (int)(s / sz), ys = (int)((s / sy) % n[1]), xs = (int)(s % sy);
        const double rv = hypot(g * (y + o[1]) + c0 - cy[z], g * (x + o[2]) + c0 - cx[z]);
        const double rs = hypot(g * (ys + o[1]) + c0 - cy[zs], g * (xs + o[2]) + c0 - cx[zs]);
        const int k = (int)codes[s] - 1 + (rv >= rs ? BAND_HALF : -BAND_HALF);
        if (side) side[i] = rv >= rs ? SIDE_RECTO : SIDE_VERSO;
        out[i] = (uint8_t)((k % BAND_PERIOD + BAND_PERIOD) % BAND_PERIOD);
    }
    /* conflicts (two surfaces > 1/4 turn apart in one voxel): unknown within one voxel */
    memset(flag, 0, N);
    for (int z = 0; z < n[0]; z++) for (int y = 0; y < n[1]; y++) for (int x = 0; x < n[2]; x++) {
        if (codes[z * sz + y * sy + x] != 255) continue;
        for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
            const int a = z + dz, b = y + dy, c = x + dx;
            if (a >= 0 && a < n[0] && b >= 0 && b < n[1] && c >= 0 && c < n[2]) flag[a * sz + b * sy + c] = 1;
        }
    }
    for (size_t i = 0; i < N; i++) if (flag[i]) out[i] = BAND_UNKNOWN;
    /* missing wraps: the band jumps by >= half a turn between neighbours both more than 2 native voxels off the surfaces */
    memset(seed, 0, N);   /* reused: jump voxels */
    int any = 0;
    for (int z = 0; z < n[0]; z++) for (int y = 0; y < n[1]; y++) for (int x = 0; x < n[2]; x++) {
        const size_t i = z * sz + y * sy + x;
        if (out[i] == BAND_UNKNOWN || d2[i] <= 1.f) continue;
        const size_t nb[3] = {z + 1 < n[0] ? i + sz : i, y + 1 < n[1] ? i + sy : i, x + 1 < n[2] ? i + 1 : i};
        for (int a = 0; a < 3; a++) {
            const size_t j = nb[a];
            if (j == i || out[j] == BAND_UNKNOWN || d2[j] <= 1.f) continue;
            if (abs(band_diff(out[i], out[j])) >= BAND_HALF) { seed[i] = seed[j] = 1; any = 1; }
        }
    }
    if (any) {
        float *e2 = d2k ? malloc(N * 4) : d2; int32_t *ei = idxk ? malloc(N * 4) : idx;
        const int rc = !e2 || !ei || edt_feature(seed, n, e2, ei);
        const float s2 = (p.span / g) * (p.span / g);
        if (!rc) for (size_t i = 0; i < N; i++) if (e2[i] <= s2) out[i] = BAND_UNKNOWN;
        if (d2k) free(e2);
        if (idxk) free(ei);
        if (rc) { BAND_FREE(); return -1; }
    }
    if (side) for (size_t i = 0; i < N; i++) if (out[i] == BAND_UNKNOWN) side[i] = SIDE_UNKNOWN;
    BAND_FREE();
#undef BAND_FREE
    return 0;
}

int band_field(const uint8_t *codes, const int n[3], const int64_t o[3], const double *cy, const double *cx, band_params p, uint8_t *out) {
    return band_impl(codes, n, o, cy, cx, p, out, nullptr, nullptr, nullptr);
}
int band_field_side(const uint8_t *codes, const int n[3], const int64_t o[3], const double *cy, const double *cx, band_params p, uint8_t *out, uint8_t *side) {
    return band_impl(codes, n, o, cy, cx, p, out, side, nullptr, nullptr);
}

/* Phase: the midpoints between consecutive rectos are the neighbour pairs on opposite sides with (nearly) the same band (at a
   recto face the band jumps by a turn instead). With ds the distance to the nearest recto and dm the distance to the nearest
   midpoint (+ half a voxel: the midpoint lies between the pair), a voxel on the recto side sits 0.5 ds / (ds + dm) of the
   way from its recto to the next one outward, a voxel on the verso side 1 - 0.5 ds / (ds + dm): linear between parallel
   rectos. Known only between two labelled consecutive rectos: the half spacing ds + dm within the flood radius, and the
   nearest recto and the nearest midpoint on opposite sides of the voxel (else it lies beyond the outermost or innermost
   labelled recto, or past the end of a trace). */
int band_field_phase(const uint8_t *codes, const int n[3], const int64_t o[3], const double *cy, const double *cx, band_params p, uint8_t *out, uint8_t *phase,
                     uint8_t *spacing) {
    const size_t N = (size_t)n[0] * n[1] * n[2], sy = (size_t)n[2], sz = (size_t)n[1] * n[2];
    uint8_t *side = malloc(N), *mid = malloc(N);
    float *d2 = malloc(N * 4), *m2 = malloc(N * 4); int32_t *idx = malloc(N * 4), *im = malloc(N * 4);
    int rc = !side || !mid || !d2 || !m2 || !idx || !im || band_impl(codes, n, o, cy, cx, p, out, side, d2, idx);
    if (!rc) {
        memset(mid, 0, N);
        for (int z = 0; z < n[0]; z++) for (int y = 0; y < n[1]; y++) for (int x = 0; x < n[2]; x++) {
            const size_t i = z * sz + y * sy + x;
            if (side[i] == SIDE_UNKNOWN) continue;
            const size_t nb[3] = {z + 1 < n[0] ? i + sz : i, y + 1 < n[1] ? i + sy : i, x + 1 < n[2] ? i + 1 : i};
            for (int a = 0; a < 3; a++) {
                const size_t j = nb[a];
                if (j == i || side[j] == SIDE_UNKNOWN || side[j] == side[i]) continue;
                if (abs(band_diff(out[i], out[j])) < BAND_HALF) mid[i] = mid[j] = 1;
            }
        }
        rc = edt_feature(mid, n, m2, im);
    }
    if (!rc) {
        const float g = band_gs(p), h = p.radius / g;   /* label voxels */
        for (size_t i = 0; i < N; i++) {
            phase[i] = PHASE_UNKNOWN;
            if (spacing) spacing[i] = 0;
            if (side[i] == SIDE_UNKNOWN || im[i] < 0) continue;
            const float ds = sqrtf(d2[i]), dm = sqrtf(m2[i]);
            if (ds + dm > h) continue;
            if (ds > 1.5f && dm > 1.5f) {
                const long z = (long)(i / sz), y = (long)((i / sy) % n[1]), x = (long)(i % sy);
                const size_t s = (size_t)idx[i], m = (size_t)im[i];
                const float az = (float)((long)(s / sz) - z), ay = (float)((long)((s / sy) % n[1]) - y), ax = (float)((long)(s % sy) - x);
                const float bz = (float)((long)(m / sz) - z), by = (float)((long)((m / sy) % n[1]) - y), bx = (float)((long)(m % sy) - x);
                if (az * bz + ay * by + ax * bx > -0.5f * ds * dm) continue;
            }
            float f = 0.5f * ds / (ds + dm + 0.5f);
            if (side[i] == SIDE_VERSO) f = 1.f - f;
            int q = (int)lrintf(f * PHASE_PERIOD);
            phase[i] = (uint8_t)(q >= PHASE_PERIOD ? q - PHASE_PERIOD : q);
            if (spacing) { const long sp = lrintf(g * (ds + dm + 0.5f)); spacing[i] = (uint8_t)(sp < 1 ? 1 : sp > 255 ? 255 : sp); }
        }
    }
    free(side); free(mid); free(d2); free(m2); free(idx); free(im);
    return rc ? -1 : 0;
}

int band_field_side_pair(const uint8_t *codes, const int n[3], float rmax, int margin, uint8_t *side) {
    const size_t N = (size_t)n[0] * n[1] * n[2], sy = (size_t)n[2], sz = (size_t)n[1] * n[2];
    uint8_t *seed = malloc(N); float *d2 = malloc(N * 4); int32_t *id[4] = {malloc(N * 4), malloc(N * 4), malloc(N * 4), malloc(N * 4)};
    int rc = !seed || !d2 || !id[0] || !id[1] || !id[2] || !id[3];
    for (int q = 0; q < 4 && !rc; q++) {   /* the labelled voxels by winding mod 4 (code 1 + 18 q mod 252: k = (code - 1) / 18): k, k +- 1 and
                                              k +- 2 fall in different classes (but across the 14-turn wrap of the codes, where k + 2 may share k's) */
        for (size_t i = 0; i < N; i++) { const int c = codes[i]; seed[i] = c > 0 && c < 255 && (((c - 1) / BAND_STEPS) & 3) == q; }
        rc = edt_feature_cap(seed, n, 4 * rmax * rmax, d2, id[q]);   /* nothing beyond the bound limit is used */
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

int band_field_phase_pair(const uint8_t *codes, const int n[3], float rmax, uint8_t *phase, uint8_t *spacing) {
    const size_t N = (size_t)n[0] * n[1] * n[2], sy = (size_t)n[2], sz = (size_t)n[1] * n[2];
    uint8_t *se = malloc(N), *so = malloc(N); float *de = malloc(N * 4), *dd = malloc(N * 4); int32_t *ie = malloc(N * 4), *io = malloc(N * 4);
    int rc = !se || !so || !de || !dd || !ie || !io;
    if (!rc) {
        for (size_t i = 0; i < N; i++) {   /* the labelled voxels by the parity of their winding (code 1 + 18 q mod 252: k = (code - 1) / 18) */
            const int c = codes[i], lab = c > 0 && c < 255;
            const int k = lab ? (c - 1) / BAND_STEPS : 0;
            se[i] = lab && !(k & 1); so[i] = lab && (k & 1);
        }
        rc = edt_feature(se, n, de, ie) || edt_feature(so, n, dd, io);
    }
    if (!rc) {
        const float r2 = rmax * rmax;
        for (size_t i = 0; i < N; i++) {
            phase[i] = PHASE_UNKNOWN; if (spacing) spacing[i] = 0;
            const int c = codes[i];
            if (c > 0 && c < 255) { phase[i] = 0; continue; }   /* on a labelled recto */
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
            if (spacing) { const long sp = lrintf(da + db); spacing[i] = (uint8_t)(sp < 1 ? 1 : sp > 255 ? 255 : sp); }
        }
    }
    free(se); free(so); free(de); free(dd); free(ie); free(io);
    return rc ? -1 : 0;
}

/* ---- relative winding mod 4 (train --task unwrap2, band.h band_wind4) ----
   Runs along x of joined voxels (known, |phase step| <= maxstep), a union-find over the runs with offsets in turns mod 4, joined
   across y and z neighbours. Every voxel's winding is phase + 252 (j + O(run)) codes: j its turns since the run's start, O the
   run's turns relative to its root, both mod 4 (the representation keeps W = phase mod 1 turn exactly, so only the integer part is
   tracked). Pass 2 replays the same runs in the same order. */
typedef struct { int32_t parent; uint8_t off; } w4_run;   /* off: turns from this run to its parent, mod 4 */
typedef struct { w4_run *r; size_t n, cap; int64_t *size, *edges, *conf; } w4_uf;
static int32_t w4_find(w4_uf *u, int32_t a, int *off) {   /* root of a, *off = turns from a to its root (mod 4); path halving with offsets */
    int o = 0; int32_t x = a;
    while (u->r[x].parent != x) { o += u->r[x].off; x = u->r[x].parent; }
    const int32_t root = x; int acc = o;
    x = a;
    while (u->r[x].parent != root && u->r[x].parent != x) {   /* compress: point straight at the root */
        const int32_t p = u->r[x].parent; const int ox = u->r[x].off;
        u->r[x].parent = root; u->r[x].off = (uint8_t)(acc & 3); acc -= ox; x = p;
    }
    *off = o & 3; return root;
}
/* W(a) - W(b) = rel turns (mod 4) in the runs' local representation: O_a - O_b = rel */
static void w4_union(w4_uf *u, int32_t a, int32_t b, int rel) {
    int oa, ob; const int32_t ra = w4_find(u, a, &oa), rb = w4_find(u, b, &ob);
    if (ra == rb) { u->edges[ra]++; if (((oa - ob - rel) & 3) != 0) u->conf[ra]++; return; }
    /* O_a = oa + O_ra, O_b = ob + O_rb: O_ra - O_rb = rel - oa + ob; attach the smaller under the larger */
    const int d = (rel - oa + ob) & 3;
    if (u->size[ra] < u->size[rb]) { u->r[ra].parent = rb; u->r[ra].off = (uint8_t)d; u->size[rb] += u->size[ra]; u->edges[rb] += u->edges[ra] + 1; u->conf[rb] += u->conf[ra]; }
    else { u->r[rb].parent = ra; u->r[rb].off = (uint8_t)((-d) & 3); u->size[ra] += u->size[rb]; u->edges[ra] += u->edges[rb] + 1; u->conf[ra] += u->conf[rb]; }
}
static inline int w4_step(int a, int b) { int d = ((b - a) % 252 + 252) % 252; return d >= 126 ? d - 252 : d; }   /* phase codes a -> b in [-126, 126) */
typedef struct { const uint8_t *ph; int D, H, W, maxstep; int32_t *rid[2]; int8_t *jj[2]; } w4_scan;
/* one pass over the window: runs numbered in scan order; pass 0 creates runs and joins neighbours, pass 1 writes w4 / comp via map */
static int w4_pass(w4_scan *s, w4_uf *u, int pass, const int16_t *map, uint8_t *w4, uint8_t *comp) {
    const int H = s->H, W = s->W; const size_t pl = (size_t)H * W;
    int32_t nrun = 0;
    for (int z = 0; z < s->D; z++) {
        int32_t *rc = s->rid[z & 1], *rp = s->rid[(z & 1) ^ 1]; int8_t *jc = s->jj[z & 1], *jp = s->jj[(z & 1) ^ 1];
        for (int y = 0; y < H; y++) {
            const uint8_t *row = s->ph + (size_t)z * pl + (size_t)y * W;
            int32_t cur = -1; int j = 0, root_off = 0; int16_t cid = -1;
            for (int x = 0; x < W; x++) {
                const size_t k = (size_t)y * W + x, g = (size_t)z * pl + k;
                const int v = row[x];
                if (v >= 252) { cur = -1; rc[k] = -1; if (pass) { w4[g] = 255; comp[g] = 255; } continue; }
                if (cur >= 0 && abs(w4_step(row[x - 1], v)) <= s->maxstep) { const int st = w4_step(row[x - 1], v); j = (j + (st - (v - row[x - 1])) / 252) & 3; }
                else {   /* a new run */
                    cur = nrun++; j = 0;
                    if (!pass) {
                        if ((size_t)cur >= u->cap) {
                            u->cap = u->cap ? 2 * u->cap : 4096;
                            u->r = realloc(u->r, u->cap * sizeof *u->r); u->size = realloc(u->size, u->cap * 8); u->edges = realloc(u->edges, u->cap * 8); u->conf = realloc(u->conf, u->cap * 8);
                            if (!u->r || !u->size || !u->edges || !u->conf) return -1;
                        }
                        u->r[cur].parent = cur; u->r[cur].off = 0; u->size[cur] = 0; u->edges[cur] = 0; u->conf[cur] = 0; u->n = (size_t)nrun;
                    } else { const int32_t r = w4_find(u, cur, &root_off); cid = map[r]; }
                }
                rc[k] = cur; jc[k] = (int8_t)j;
                if (!pass) {
                    u->size[w4_find(u, cur, &root_off)]++;
                    /* neighbours y - 1 (this plane) and z - 1: W(v) - W(n) = step codes; with W = ph + 252 (j + O):
                       252 (j_v + O_v - j_n - O_n) = step - (ph_v - ph_n) */
                    for (int q = 0; q < 2; q++) {
                        int32_t rn; int jn, pn;
                        if (q == 0) { if (y == 0 || rc[k - W] < 0) continue; rn = rc[k - W]; jn = jc[k - W]; pn = s->ph[g - W]; }
                        else { if (z == 0 || rp[k] < 0) continue; rn = rp[k]; jn = jp[k]; pn = s->ph[g - pl]; }
                        const int st = w4_step(pn, v);
                        if (abs(st) > s->maxstep) continue;
                        const int t = (st - (v - pn)) / 252;   /* -1, 0 or 1 */
                        w4_union(u, cur, rn, (t + jn - j) & 3);
                    }
                } else {
                    if (cid < 0) { w4[g] = 255; comp[g] = 255; continue; }
                    const int tw = (j + root_off) & 3;
                    w4[g] = (uint8_t)(((v + 252 * tw + 2) / 4) % 252); comp[g] = (uint8_t)cid;
                }
            }
        }
    }
    return 0;
}
int band_wind4(const uint8_t *phase, int D, int H, int W, int maxstep, size_t minsize, int maxcomp, uint8_t *w4, uint8_t *comp, band_wind4_stats *st) {
    if (maxcomp > 15) maxcomp = 15;
    const size_t pl = (size_t)H * W;
    w4_scan s = {phase, D, H, W, maxstep, {malloc(pl * 4), malloc(pl * 4)}, {malloc(pl), malloc(pl)}};
    w4_uf u = {0};
    int rc = !s.rid[0] || !s.rid[1] || !s.jj[0] || !s.jj[1] ? -1 : w4_pass(&s, &u, 0, nullptr, nullptr, nullptr);
    int16_t *map = rc ? nullptr : malloc((u.n ? u.n : 1) * sizeof *map);
    int ncomp = 0; size_t kept = 0, known = 0, dropped_conf = 0;
    if (!rc && !map) rc = -1;
    if (!rc) {
        int32_t best[15]; int nb = 0;
        for (size_t i = 0; i < u.n; i++) {
            map[i] = -1;
            if (u.r[i].parent != (int32_t)i) continue;
            known += (size_t)u.size[i];
            if ((size_t)u.size[i] < minsize) continue;
            if (u.conf[i] > 8 + u.edges[i] / 500) { dropped_conf += (size_t)u.size[i]; continue; }   /* inconsistent unwrapping (> 0.2% of its joins) */
            int p = nb < maxcomp ? nb++ : maxcomp;   /* insertion into the size-sorted list of the largest */
            if (p == maxcomp && u.size[i] <= u.size[best[maxcomp - 1]]) continue;
            if (p == maxcomp) p = maxcomp - 1;
            while (p > 0 && u.size[best[p - 1]] < u.size[i]) { best[p] = best[p - 1]; p--; }
            best[p] = (int32_t)i;
        }
        for (int c = 0; c < nb; c++) { map[best[c]] = (int16_t)c; kept += (size_t)u.size[best[c]]; }
        ncomp = nb;
        rc = w4_pass(&s, &u, 1, map, w4, comp);
    }
    if (st) { st->known = known; st->kept = kept; st->dropped_conflict = dropped_conf; st->ncomp = ncomp; st->runs = u.n; }
    free(map); free(u.r); free(u.size); free(u.edges); free(u.conf); free(s.rid[0]); free(s.rid[1]); free(s.jj[0]); free(s.jj[1]);
    return rc ? -1 : ncomp;
}
/* band_wind4 per tile of tile^3 voxels (the last tiles of an axis smaller), each tile's components their own gauges: a label
   inconsistency (a loop of joins that gains a turn) then drops only its tile's component. Components numbered over the tiles in
   raster order, up to 255 (comp 0..254); maxcomp per tile. tile <= 0: the whole volume as one tile. */
int band_wind4_tiled(const uint8_t *phase, int D, int H, int W, int tile, int maxstep, size_t minsize, int maxcomp, uint8_t *w4, uint8_t *comp, band_wind4_stats *st) {
    if (tile <= 0 || (tile >= D && tile >= H && tile >= W)) {
        const int nc = band_wind4(phase, D, H, W, maxstep, minsize, maxcomp, w4, comp, st);
        return nc;
    }
    const size_t T3 = (size_t)tile * tile * tile;
    uint8_t *tp = malloc(T3), *tw = malloc(T3), *tc = malloc(T3);
    if (!tp || !tw || !tc) { free(tp); free(tw); free(tc); return -1; }
    band_wind4_stats acc = {0}; int next = 0, rc = 0;
    for (int z0 = 0; z0 < D && !rc; z0 += tile) for (int y0 = 0; y0 < H && !rc; y0 += tile) for (int x0 = 0; x0 < W && !rc; x0 += tile) {
        const int nz = D - z0 < tile ? D - z0 : tile, ny = H - y0 < tile ? H - y0 : tile, nx = W - x0 < tile ? W - x0 : tile;
        for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) memcpy(tp + ((size_t)z * ny + y) * nx, phase + ((size_t)(z0 + z) * H + y0 + y) * W + x0, (size_t)nx);
        band_wind4_stats s;
        const int nc = band_wind4(tp, nz, ny, nx, maxstep, minsize, maxcomp > 15 ? 15 : maxcomp, tw, tc, &s);
        if (nc < 0) { rc = -1; break; }
        const int base = next; next += nc;
        for (int z = 0; z < nz; z++) for (int y = 0; y < ny; y++) {
            const size_t so = ((size_t)z * ny + y) * nx, d = ((size_t)(z0 + z) * H + y0 + y) * W + x0;
            for (int x = 0; x < nx; x++) {
                const int c = tc[so + x];
                if (c == 255 || base + c > 254) { w4[d + x] = 255; comp[d + x] = 255; continue; }
                w4[d + x] = tw[so + x]; comp[d + x] = (uint8_t)(base + c);
            }
        }
        acc.known += s.known; acc.kept += s.kept; acc.dropped_conflict += s.dropped_conflict; acc.runs += s.runs;
    }
    if (next > 255) next = 255;
    acc.ncomp = next;
    if (st) *st = acc;
    free(tp); free(tw); free(tc);
    return rc ? -1 : next;
}
