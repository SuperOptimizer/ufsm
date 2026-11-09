#include "scan_augment.h"
#include "ct_augment.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

const char *const scan_aug_names[SA_N] = {"seam", "air", "lowres", "blur", "sharpen", "bias", "ring", "cupping", "slice",
                                          "noise", "cnoise", "tone", "quant", "dropslice", "cutout", "chunk"};

/* Local SplitMix64, as ct_augment: decisions and per-voxel noise use their own streams. */
static uint64_t next_u64(uint64_t *s) {
    uint64_t v = (*s += UINT64_C(0x9e3779b97f4a7c15));
    v = (v ^ (v >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    v = (v ^ (v >> 27)) * UINT64_C(0x94d049bb133111eb);
    return v ^ (v >> 31);
}
static double uniform(uint64_t *s) { return (double)(next_u64(s) >> 11) * 0x1.0p-53; }
static double sym1(uint64_t *s) { return 2 * uniform(s) - 1; }
/* Irwin-Hall of four bytes: unit variance, |g| < 3.46. Two draws per 64 bits. */
static inline float ih4(uint32_t v) { return (float)((int)(v & 255) + (int)((v >> 8) & 255) + (int)((v >> 16) & 255) + (int)(v >> 24) - 510) * (1.f / 147.80f); }
static void gauss_row(uint64_t *s, float *row, int n) {
    for (int x = 0; x < n; x += 2) { uint64_t v = next_u64(s); row[x] = ih4((uint32_t)v); if (x + 1 < n) row[x + 1] = ih4((uint32_t)(v >> 32)); }
}

/* ---- separable 1D operators: out[i] = sum_t w[i][t] in[s[i] + t], replicate boundary ---- */
#define MAXTAP 48
typedef struct { int P, s[1024], n[1024]; float w[1024][MAXTAP]; } bop;
static void bop_add(float *dense, int *lo, int *hi, int P, int j, float w) {
    j = j < 0 ? 0 : j >= P ? P - 1 : j;
    dense[j] += w; if (j < *lo) *lo = j; if (j > *hi) *hi = j;
}
static void bop_store(bop *b, int i, float *dense, int lo, int hi) {
    if (hi - lo + 1 > MAXTAP) abort();   /* the magnitude caps keep every composition within MAXTAP */
    b->s[i] = lo; b->n[i] = hi - lo + 1;
    for (int t = lo; t <= hi; t++) { b->w[i][t - lo] = dense[t]; dense[t] = 0; }
}
static void bop_identity(bop *b, int P) { b->P = P; for (int i = 0; i < P; i++) { b->s[i] = i; b->n[i] = 1; b->w[i][0] = 1; } }
/* sharpen a > 0: (1 + a) I - a G (phase-contrast edge overshoot); a = -1 is the plain blur G */
static void bop_gauss(bop *b, int P, double sigma, double a, float *dense) {
    int r = (int)ceil(3 * sigma); if (r > 7) r = 7;
    double g[15], sum = 0;
    for (int t = -r; t <= r; t++) sum += g[t + r] = exp(-.5 * t * t / (sigma * sigma));
    b->P = P;
    for (int i = 0; i < P; i++) {
        int lo = P, hi = -1;
        if (a > -1) bop_add(dense, &lo, &hi, P, i, (float)(1 + a));
        for (int t = -r; t <= r; t++) bop_add(dense, &lo, &hi, P, i + t, (float)((a > -1 ? -a : 1) * g[t + r] / sum));
        bop_store(b, i, dense, lo, hi);
    }
}
/* coarser scan: cell averages over [phase + j f, phase + (j+1) f), linearly interpolated back at the voxel centres */
static void bop_lowres(bop *b, int P, double f, double phase, float *dense) {
    b->P = P;
    for (int i = 0; i < P; i++) {
        double u = (i + .5 - phase) / f - .5; int j0 = (int)floor(u); double t = u - j0;
        int lo = P, hi = -1;
        for (int c = 0; c < 2; c++) {
            double a = phase + (j0 + c) * f, e = a + f, wc = c ? t : 1 - t;
            if (wc <= 0) continue;
            for (int k = (int)floor(a); k < (int)ceil(e); k++) {
                double ov = fmin(e, k + 1) - fmax(a, k);
                if (ov > 0) bop_add(dense, &lo, &hi, P, k, (float)(wc * ov / f));
            }
        }
        bop_store(b, i, dense, lo, hi);
    }
}
/* c = a . b (b applied first) */
static void bop_compose(bop *c, const bop *a, const bop *b, float *dense) {
    int P = a->P; c->P = P;
    for (int i = 0; i < P; i++) {
        int lo = P, hi = -1;
        for (int t = 0; t < a->n[i]; t++) {
            int j = a->s[i] + t;
            for (int u = 0; u < b->n[j]; u++) bop_add(dense, &lo, &hi, P, b->s[j] + u, a->w[i][t] * b->w[j][u]);
        }
        bop_store(c, i, dense, lo, hi);
    }
}
static void transpose(const float *a, float *b, int P) {
    for (int i0 = 0; i0 < P; i0 += 32) for (int j0 = 0; j0 < P; j0 += 32)
        for (int i = i0; i < i0 + 32 && i < P; i++) for (int j = j0; j < j0 + 32 && j < P; j++) b[(size_t)j * P + i] = a[(size_t)i * P + j];
}
/* rows i of out = sum_t w[i][t] rows (s[i] + t) of in: vector form along the contiguous axis */
static void bop_rows(const bop *b, const float *in, float *out, size_t ostride, int P) {
    for (int i = 0; i < P; i++) {
        float *o = out + (size_t)i * ostride;
        const float *r0 = in + (size_t)b->s[i] * P; const float w0 = b->w[i][0];
        for (int x = 0; x < P; x++) o[x] = w0 * r0[x];
        for (int t = 1; t < b->n[i]; t++) { const float *r = in + (size_t)(b->s[i] + t) * P, w = b->w[i][t]; for (int x = 0; x < P; x++) o[x] += w * r[x]; }
    }
}
/* tmp: 2 P^2 floats. axis 2 transposes each plane, axis 1 runs on planes, axis 0 on (z, x) slabs. */
static void bop_apply(float *F, int P, int axis, const bop *b, float *tmp) {
    const size_t P2 = (size_t)P * P; float *t2 = tmp + P2;
    for (int a = 0; a < P; a++) {
        if (axis == 2) { transpose(F + a * P2, tmp, P); bop_rows(b, tmp, t2, P, P); transpose(t2, F + a * P2, P); }
        else if (axis == 1) { memcpy(tmp, F + a * P2, P2 * sizeof *tmp); bop_rows(b, tmp, F + a * P2, P, P); }
        else { for (int z = 0; z < P; z++) memcpy(tmp + (size_t)z * P, F + z * P2 + (size_t)a * P, P * sizeof *tmp); bop_rows(b, tmp, F + (size_t)a * P, P2, P); }
    }
}
static int bop_is_identity(const bop *b) { for (int i = 0; i < b->P; i++) if (b->n[i] != 1 || b->s[i] != i || b->w[i][0] != 1) return 0; return 1; }

/* shift plane z >= zc content by (dy, dx); vacated voxels take fill */
static void shift_planes(void *v, size_t es, int P, int zc, int dy, int dx, const void *fill, void *tmp) {
    const size_t row = (size_t)P * es, plane = row * P;
    const int x0 = dx > 0 ? dx : 0, x1 = dx < 0 ? P + dx : P;   /* destination x range with a source */
    for (int z = zc; z < P; z++) {
        uint8_t *pl = (uint8_t *)v + z * plane; memcpy(tmp, pl, plane);
        for (int y = 0; y < P; y++) {
            uint8_t *d = pl + y * row; int sy = y - dy;
            for (int x = 0; x < P; x++) if (sy < 0 || sy >= P || x < x0 || x >= x1) memcpy(d + x * es, fill, es);
            if (sy >= 0 && sy < P && x1 > x0) memcpy(d + x0 * es, (uint8_t *)tmp + sy * row + (x0 - dx) * es, (size_t)(x1 - x0) * es);
        }
    }
}

unsigned scan_aug_apply(const scan_aug_cfg *c, uint64_t seed, uint8_t *ctu, uint8_t *tgt, int nch, uint8_t *ign, int P,
                        const int64_t o[3], float *cy, float *cx, int hasax, const int chunk[3], int allow_seam, void *scratch) {
    if (!scan_aug_enabled(c) || P < 4 || P > 1024) return 0;
    const double S = c->strength;
    const size_t P2 = (size_t)P * P, p3 = P2 * P;
    uint64_t rs = seed, ns = seed ^ UINT64_C(0x5ca1ab1e0ddba11);
    unsigned on = 0;
    for (int i = 0; i < SA_N; i++) if (c->p[i] > 0 && uniform(&rs) < c->p[i]) on |= 1u << i;
    if (!allow_seam) on &= ~(1u << SA_SEAM);
    if (!on) return 0;
    float *F = scratch; uint8_t *Z = (uint8_t *)scratch + 4 * p3;   /* float CT; 1 = the output is air (0) */
    float *tmp = malloc(2 * P2 * sizeof *tmp), *dense = calloc(P, sizeof *dense), *row = malloc(P * sizeof *row);
    if (!tmp || !dense || !row) abort();
    double sum = 0, sq = 0; size_t nz = 0; size_t hist[256] = {0};
    for (size_t k = 0; k < p3; k++) { F[k] = ctu[k]; Z[k] = !ctu[k]; }
    for (size_t k = 0; k < p3; k += 7) hist[ctu[k]]++;   /* magnitudes only need the statistics of a subsample */
    for (int v = 1; v < 256; v++) { nz += hist[v]; sum += (double)v * hist[v]; sq += (double)v * v * hist[v]; }
    const double mean = nz ? sum / nz : 128, sd = nz ? sqrt(fmax(sq / nz - mean * mean, 1)) : 32;   /* of the material (nonzero) voxels */
    if (on & 1u << SA_SEAM) {   /* stitched scan segments: a z slab moved in y/x, labels and axis with it */
        int zc = P / 8 + (int)(uniform(&rs) * (P - P / 4)), m = 1 + (int)lround(2 * S), dy, dx;
        do { dy = (int)lround(sym1(&rs) * m); dx = (int)lround(sym1(&rs) * m); } while (!dy && !dx);
        const float f0 = 0; const uint8_t one = 1, zero = 0;
        shift_planes(F, sizeof *F, P, zc, dy, dx, &f0, tmp);
        shift_planes(Z, 1, P, zc, dy, dx, &one, tmp);
        shift_planes(ign, 1, P, zc, dy, dx, &one, tmp);
        for (int ch = 0; ch < nch; ch++) shift_planes(tgt + ch * p3, 1, P, zc, dy, dx, &zero, tmp);
        for (int z = zc; z < P; z++) { cy[z] += dy; cx[z] += dx; }
    }
    if (on & 1u << SA_AIR) {   /* unmasked scan: air (masked to 0) becomes dim noisy material, still unsupervised */
        size_t lo = 0, k1 = nz / 100; int v1 = 1; while (v1 < 255 && lo + hist[v1] <= k1) lo += hist[v1++];
        float level = (float)(v1 * (.4 + .6 * uniform(&rs)));
        for (size_t k = 0; k < p3; k++) if (Z[k]) { F[k] = level; Z[k] = 0; ign[k] = 1; }
    }
    /* resolution: PSF blur or phase-contrast sharpening, then a coarser sampling grid, per axis */
    if (on & (1u << SA_LOWRES | 1u << SA_BLUR | 1u << SA_SHARPEN)) {
        bop *op = malloc(3 * sizeof *op); if (!op) abort();
        int iso = uniform(&rs) < .5;
        double f0 = 1 + uniform(&rs) * 1.5 * S, sb0 = .3 + uniform(&rs) * .7 * S, sa0 = .15 + uniform(&rs) * .3 * S;
        for (int d = 0; d < 3; d++) {
            double f = iso ? f0 : 1 + uniform(&rs) * 1.5 * S, sb = iso ? sb0 : .3 + uniform(&rs) * .7 * S, sa = iso ? sa0 : .15 + uniform(&rs) * .3 * S;
            double ss = .7 + .6 * uniform(&rs), phase = uniform(&rs) * f;
            f = fmin(f, 4); sb = fmin(sb, 2.3);
            bop_identity(&op[0], P);
            if (on & 1u << SA_BLUR) { bop_gauss(&op[1], P, sb, -1, dense); bop_compose(&op[2], &op[1], &op[0], dense); op[0] = op[2]; }
            if (on & 1u << SA_SHARPEN) { bop_gauss(&op[1], P, ss, sa, dense); bop_compose(&op[2], &op[1], &op[0], dense); op[0] = op[2]; }
            if (on & 1u << SA_LOWRES && f > 1.01) { bop_lowres(&op[1], P, f, phase, dense); bop_compose(&op[2], &op[1], &op[0], dense); op[0] = op[2]; }
            if (!bop_is_identity(&op[0])) bop_apply(F, P, d, &op[0], tmp);
        }
        free(op);
    }
    if (on & 1u << SA_BIAS) {   /* smooth multiplicative field: exp(N(0, b)) on a 3..5 grid, linearly interpolated */
        int g = 3 + (int)(uniform(&rs) * 3); double b = (.08 + .12 * uniform(&rs)) * S;
        float G[125]; for (int k = 0; k < g * g * g; k++) G[k] = (float)exp(b * sym1(&rs) * 1.7320508);
        int *ix = malloc(P * sizeof *ix); float *wx = malloc(P * sizeof *wx), crow[5]; if (!ix || !wx) abort();
        const float sc = (float)(g - 1) / (float)(P - 1);
        for (int x = 0; x < P; x++) { float u = x * sc; int i = (int)u; if (i > g - 2) i = g - 2; ix[x] = i; wx[x] = u - i; }
        for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) {
            int iz = ix[z], iy = ix[y]; float fz = wx[z], fy = wx[y];
            for (int j = 0; j < g; j++) {
                const float *q = G + ((size_t)iz * g + iy) * g + j;
                crow[j] = (1 - fz) * ((1 - fy) * q[0] + fy * q[g]) + fz * ((1 - fy) * q[g * g] + fy * q[g * g + g]);
            }
            float *r = F + z * P2 + (size_t)y * P;
            for (int x = 0; x < P; x++) r[x] *= crow[ix[x]] + wx[x] * (crow[ix[x] + 1] - crow[ix[x]]);
        }
        free(ix); free(wx);
    }
    if (on & (1u << SA_RING | 1u << SA_CUPPING)) {   /* radial profile about the scan axis: rings + cupping */
        float ccy[1024], ccx[1024];
        if (hasax) for (int z = 0; z < P; z++) { ccy[z] = cy[z] - (float)o[1]; ccx[z] = cx[z] - (float)o[2]; }
        else { double t = 2 * M_PI * uniform(&rs), D = P * (1 + 3 * uniform(&rs)); for (int z = 0; z < P; z++) { ccy[z] = (float)(.5 * P + D * sin(t)); ccx[z] = (float)(.5 * P + D * cos(t)); } }
        float rmin = INFINITY, rmax = 0;
        for (int z = 0; z < P; z++) {
            float dy = fmaxf(0, fmaxf(-ccy[z], ccy[z] - (P - 1))), dx = fmaxf(0, fmaxf(-ccx[z], ccx[z] - (P - 1)));
            float fy = fmaxf(fabsf(ccy[z]), fabsf(ccy[z] - (P - 1))), fx = fmaxf(fabsf(ccx[z]), fabsf(ccx[z] - (P - 1)));
            rmin = fminf(rmin, sqrtf(dy * dy + dx * dx)); rmax = fmaxf(rmax, sqrtf(fy * fy + fx * fx));
        }
        const float res = 4; int nt = (int)((rmax - rmin) * res) + 3;
        float *tab = calloc(nt, sizeof *tab); if (!tab) abort();
        if (on & 1u << SA_RING) {
            int nr = 3 + (int)fmin(512, (rmax - rmin) * (.02 + .1 * uniform(&rs)) * S);   /* dense: one per miscalibrated detector column */ double amp = (.05 + .15 * uniform(&rs)) * S * sd;
            for (int i = 0; i < nr; i++) {
                double r0 = rmin + uniform(&rs) * (rmax - rmin), w = .4 + 1.1 * uniform(&rs), a = amp * sym1(&rs) * 1.7320508;
                for (int t = (int)fmax(0, (r0 - 4 * w - rmin) * res); t < nt && rmin + t / res < r0 + 4 * w; t++) { double d = (rmin + t / res - r0) / w; tab[t] += (float)(a * exp(-.5 * d * d)); }
            }
        }
        if (on & 1u << SA_CUPPING) {   /* beam hardening: a smooth radial bowl, here its local quadratic over the patch */
            double a = (.1 + .3 * uniform(&rs)) * S * sd * (uniform(&rs) < .5 ? -1 : 1), lin = sym1(&rs);
            for (int t = 0; t < nt; t++) { double u = 2 * (t / res) / fmax(rmax - rmin, 1) - 1; tab[t] += (float)(a * (u * u + lin * u)); }
        }
        for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) {
            float *r = F + z * P2 + (size_t)y * P, dy = y - ccy[z];
            for (int x = 0; x < P; x++) {
                float dx = x - ccx[z], u = (sqrtf(dy * dy + dx * dx) - rmin) * res; if (u < 0) u = 0; if (u > nt - 2) u = nt - 2;
                int i = (int)u; r[x] += tab[i] + (u - i) * (tab[i + 1] - tab[i]);
            }
        }
        free(tab);
    }
    if (on & 1u << SA_SLICE) {   /* per-slice gain/offset along the scan axis (beam current / detector drift), AR(1) in z */
        double rho = .9 * uniform(&rs), q = sqrt(1 - rho * rho), ga = .03 * S, oa = .08 * S * sd, gz = 0, oz = 0;
        for (int z = 0; z < P; z++) {
            gz = z ? rho * gz + q * sym1(&rs) * 1.7320508 : sym1(&rs) * 1.7320508; oz = z ? rho * oz + q * sym1(&rs) * 1.7320508 : sym1(&rs) * 1.7320508;
            const float gain = (float)exp(ga * gz), off = (float)(oa * oz); float *pl = F + z * P2;
            for (size_t k = 0; k < P2; k++) pl[k] = pl[k] * gain + off;
        }
    }
    if (on & 1u << SA_NOISE) {   /* white noise, partly signal-dependent (sigma ~ sqrt(value)) */
        const float s0 = (float)((.05 + .2 * uniform(&rs)) * S * sd), mix = (float)uniform(&rs), im = (float)(1 / mean);
        for (size_t r = 0; r < P2; r++) {
            float *v = F + r * P; gauss_row(&ns, row, P);
            for (int x = 0; x < P; x++) v[x] += s0 * row[x] * ((1 - mix) + mix * sqrtf(fmaxf(v[x], 0) * im));
        }
    }
    if (on & 1u << SA_CNOISE) {   /* correlated noise: unit-variance AR(1) along x, y, z (ct_augment's generator) */
        ct_aug_plan q; memset(&q, 0, sizeof q); q.noise_rho = (float)(.5 + .35 * uniform(&rs));
        const float s0 = (float)((.05 + .15 * uniform(&rs)) * S * sd);
        float *prow = malloc(P * sizeof *prow), *pplane = malloc(P2 * sizeof *pplane); if (!prow || !pplane) abort();
        for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) {
            gauss_row(&ns, row, P); ct_aug_noise_row(&q, row, prow, pplane, P, z, y);
            float *v = F + z * P2 + (size_t)y * P; for (int x = 0; x < P; x++) v[x] += s0 * row[x];
        }
        free(prow); free(pplane);
    }
    float lut[257]; int tone = 0;
    for (int i = 0; i < 257; i++) lut[i] = (float)(i < 256 ? i : 255);
    if (on & 1u << SA_TONE) {   /* reconstruction / conversion to 8 bits: gamma, random monotone remap, narrower window */
        double g = exp(sym1(&rs) * .35 * S), kn[6] = {0}, inc[5], tot = 0;
        for (int i = 0; i < 5; i++) tot += inc[i] = exp(sym1(&rs) * .5 * S);
        for (int i = 0; i < 5; i++) kn[i + 1] = kn[i] + 255 * inc[i] / tot;
        double lo = 0, hi = 255;
        if (uniform(&rs) < .5) { lo = fmax(0, mean - sd * (4 - 1.25 * S * uniform(&rs))); hi = fmin(255, mean + sd * (4 - 1.25 * S * uniform(&rs))); }
        for (int i = 0; i < 256; i++) {
            double v = 255 * pow(i / 255., g), u = v / 51; int j = (int)u; if (j > 4) j = 4;
            v = kn[j] + (u - j) * (kn[j + 1] - kn[j]);
            lut[i] = (float)(255 * fmin(1, fmax(0, (v - lo) / fmax(hi - lo, 1))));
        }
        lut[256] = lut[255]; tone = 1;
    }
    float step = 0;
    if (on & 1u << SA_QUANT) { int drop = 1 + (int)(uniform(&rs) * 2 * S); if (drop > 4) drop = 4; step = (float)(1 << drop); }
    if (on & 1u << SA_DROPSLICE) {   /* missing / corrupt reconstructed slices: repeated neighbour, constant or pure noise */
        int n = 1 + (int)(uniform(&rs) * 2 * S);
        for (int i = 0; i < n; i++) {
            int z = (int)(uniform(&rs) * P), mode = (int)(uniform(&rs) * 3); float *pl = F + z * P2;
            if (mode == 0) memcpy(pl, F + (size_t)(z ? z - 1 : 1) * P2, P2 * sizeof *pl);
            else for (int r = 0; r < P; r++) { gauss_row(&ns, row, P); for (int x = 0; x < P; x++) pl[(size_t)r * P + x] = (float)(mean + (mode == 2 ? 1.5 * sd * row[x] : 0)); }
            if (c->drop_ignore) memset(ign + z * P2, 1, P2);
        }
    }
    if (on & 1u << SA_CUTOUT) {   /* occlusion: boxes of constant or noise */
        int n = 1 + (int)(uniform(&rs) * 2 * S);
        for (int i = 0; i < n; i++) {
            int b0[3], b1[3], noisy = uniform(&rs) < .5;
            for (int d = 0; d < 3; d++) { int e = P / 16 + (int)(uniform(&rs) * P / 8 * S); if (e > P / 2) e = P / 2; b0[d] = (int)(uniform(&rs) * (P - e)); b1[d] = b0[d] + e; }
            for (int z = b0[0]; z < b1[0]; z++) for (int y = b0[1]; y < b1[1]; y++) {
                float *v = F + z * P2 + (size_t)y * P;
                if (noisy) gauss_row(&ns, row, P);
                for (int x = b0[2]; x < b1[2]; x++) v[x] = (float)(mean + (noisy ? sd * row[x] : 0));
                if (c->drop_ignore) memset(ign + z * P2 + (size_t)y * P + b0[2], 1, b1[2] - b0[2]);
            }
        }
    }
    if (on & 1u << SA_CHUNK) {   /* missing zarr chunks read as the fill value 0: unsupervised through the CT > 0 mask */
        int n = 1 + (int)(uniform(&rs) * 2 * S), e[3];
        for (int d = 0; d < 3; d++) { e[d] = chunk[d] > 0 && chunk[d] <= P / 2 ? chunk[d] : P / 4; if (e[d] < 1) e[d] = 1; }
        for (int i = 0; i < n; i++) {
            int b0[3], b1[3];
            for (int d = 0; d < 3; d++) {   /* a chunk-grid cell (global grid) that meets the patch */
                int64_t g0 = (o[d] / e[d]) * e[d], cells = (o[d] + P - 1) / e[d] - o[d] / e[d] + 1, a = g0 + (int64_t)(uniform(&rs) * cells) * e[d] - o[d];
                b0[d] = a < 0 ? 0 : (int)a; b1[d] = a + e[d] > P ? P : (int)(a + e[d]);
            }
            for (int z = b0[0]; z < b1[0]; z++) for (int y = b0[1]; y < b1[1]; y++) memset(Z + z * P2 + (size_t)y * P + b0[2], 1, b1[2] - b0[2]);
        }
    }
    if (tone) for (size_t k = 0; k < p3; k++) { float v = fminf(fmaxf(F[k], 0), 255); int i = (int)v; F[k] = lut[i] + (v - i) * (lut[i + 1] - lut[i]); }
    if (step > 0) { const float is = 1 / step; for (size_t k = 0; k < p3; k++) F[k] = (floorf(F[k] * is) + .5f) * step; }
    for (size_t k = 0; k < p3; k++) { float v = fminf(fmaxf(F[k] + .5f, 1), 255.5f); uint8_t q = (uint8_t)(int)v; ctu[k] = Z[k] ? 0 : q; }
    free(tmp); free(dense); free(row);
    return on;
}
