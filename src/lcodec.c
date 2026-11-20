/* lcodec: the learned input codec, stage 1 (lcodec.h) */
#include "lcodec.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

lcodec_off lcodec_offs(int cl, int hid) {
    const size_t K = (size_t)cl - 1; lcodec_off o;
    o.Wl = 0; o.W1 = K * LCD_NAX; o.b1 = o.W1 + (size_t)hid * LCD_NF; o.W2 = o.b1 + (size_t)hid; o.S = o.W2 + K * (size_t)hid; o.lb = o.S + K * LCD_NAX;
    return o;
}
size_t lcodec_np(int cl, int hid) { return lcodec_offs(cl, hid).lb + (size_t)cl - 1; }
static double gauss(uint64_t *s) {   /* Box-Muller on a splitmix stream */
    double u[2];
    for (int i = 0; i < 2; i++) { uint64_t z = (*s += 0x9E3779B97F4A7C15ull); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; z ^= z >> 31;
                                  u[i] = ((double)(z >> 11) + 0.5) * 0x1p-53; }
    return sqrt(-2 * log(u[0])) * cos(6.283185307179586 * u[1]);
}
int lcodec_init(lcodec *lc, int cl, int hid, float sd_ref, float s0, const float *sigma, float mean_c, uint64_t seed) {
    memset(lc, 0, sizeof *lc);
    if (cl < 2 || cl > 64 || hid < 1 || hid > 256 || !(s0 > 0) || !(sd_ref > 0)) return -1;
    lc->cl = cl; lc->hid = hid; lc->sd_ref = sd_ref; lc->mstep = s0; lc->mean_c = mean_c; lc->np = lcodec_np(cl, hid);
    lc->p = calloc(lc->np, sizeof *lc->p); if (!lc->p) return -1;
    const lcodec_off o = lcodec_offs(cl, hid); const int K = cl - 1;
    for (int c = 0; c < K; c++) { lc->p[o.Wl + (size_t)c * LCD_NAX + c] = 1.f / s0; lc->p[o.S + (size_t)c * LCD_NAX + c] = s0;
                                  lc->p[o.lb + c] = logf(fmaxf(sigma ? sigma[c + 1] : 1.f, 1e-3f) / s0); }
    const double sw = 1 / sqrt((double)LCD_NF);
    for (size_t i = 0; i < (size_t)hid * LCD_NF; i++) lc->p[o.W1 + i] = (float)(sw * gauss(&seed));
    return 0;
}
void lcodec_free(lcodec *lc) { free(lc->p); memset(lc, 0, sizeof *lc); }

void lcodec_feat(const lcodec *lc, const float *xg, int n0, int n1, int n2, int z, int y, int x, float *f) {
    const float *c = xg + (((size_t)z * n1 + y) * n2 + x) * 64;
    for (int j = 0; j < LCD_NAX; j++) f[j] = c[1 + j];
    f[63] = 0.125f * c[0] - lc->mean_c;
    int k = 64;
    for (int dz = -1; dz <= 1; dz++) for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
        if (!dz && !dy && !dx) continue;
        const int zz = z + dz, yy = y + dy, xx = x + dx;
        const float *nb = zz >= 0 && zz < n0 && yy >= 0 && yy < n1 && xx >= 0 && xx < n2 ? xg + (((size_t)zz * n1 + yy) * n2 + xx) * 64 : c;
        f[k++] = 0.125f * (nb[0] - c[0]); f[k++] = nb[1]; f[k++] = nb[2]; f[k++] = nb[3];
    }
}
static inline float silu(float a) { return a / (1.f + expf(-a)); }
void lcodec_analysis(const lcodec *lc, const float *f, const float *x, float *y) {
    const lcodec_off o = lcodec_offs(lc->cl, lc->hid); const int K = lc->cl - 1, Hd = lc->hid; float h[256];
    for (int k = 0; k < Hd; k++) { const float *w = lc->p + o.W1 + (size_t)k * LCD_NF; float a = lc->p[o.b1 + k]; for (int j = 0; j < LCD_NF; j++) a += w[j] * f[j]; h[k] = silu(a); }
    for (int c = 0; c < K; c++) { const float *wl = lc->p + o.Wl + (size_t)c * LCD_NAX, *w2 = lc->p + o.W2 + (size_t)c * Hd; float s = 0;
                                  for (int j = 0; j < LCD_NAX; j++) s += wl[j] * x[1 + j];
                                  for (int k = 0; k < Hd; k++) s += w2[k] * h[k];
                                  y[c] = s; }
}
void lcodec_encode_cell(const lcodec *lc, const float *f, const float *x, int any, int16_t *q) {
    const int K = lc->cl - 1;
    if (!any) { memset(q, 0, (size_t)lc->cl * sizeof *q); return; }
    float y[64]; lcodec_analysis(lc, f, x, y); int nz = 0;
    for (int c = 0; c < K; c++) { float v = rintf(y[c]); v = v > 127 ? 127 : v < -127 ? -127 : v; q[1 + c] = (int16_t)v; nz |= q[1 + c] != 0; }
    float m = rintf(x[0] / lc->mstep); m = m < 0 ? 0 : m > 32767 ? 32767 : m;
    q[0] = (int16_t)lv_fwd_mean((int)m, nz);
}
/* -log2 of the Laplace(0, b) mass of [t - 1/2, t + 1/2] and its derivatives in t and in log b */
static inline double lap_F(double t, double b) { return t >= 0 ? 1 - 0.5 * exp(-t / b) : 0.5 * exp(t / b); }
static inline double lap_f(double t, double b) { return exp(-fabs(t) / b) / (2 * b); }
static double lap_bits(double t, double b, double *dt, double *dlb) {
    const double P = lap_F(t + 0.5, b) - lap_F(t - 0.5, b);
    if (!(P > 1e-12)) { *dt = 0; *dlb = 0; return -log2(1e-12); }
    const double fp = lap_f(t + 0.5, b), fm = lap_f(t - 0.5, b), dPdt = fp - fm, dPdb = -((t + 0.5) * fp - (t - 0.5) * fm) / b;
    *dt = -dPdt / (P * M_LN2); *dlb = -dPdb * b / (P * M_LN2);
    return -log2(P);
}
double lcodec_loss_grad(const lcodec *lc, int n, const float *f, const float *x, const float *u, double lambda, float *g, double *R, double *D) {
    const lcodec_off o = lcodec_offs(lc->cl, lc->hid); const int K = lc->cl - 1, Hd = lc->hid; const double sd2 = (double)lc->sd_ref * lc->sd_ref, inv = 1.0 / n;
    double Rs = 0, Ds = 0; float a[256], h[256], yt[64], dy[64], dh[256];
    const float *Wl = lc->p + o.Wl, *W1 = lc->p + o.W1, *b1 = lc->p + o.b1, *W2 = lc->p + o.W2, *S = lc->p + o.S, *lb = lc->p + o.lb;
    for (int i = 0; i < n; i++) {
        const float *fi = f + (size_t)i * LCD_NF, *xi = x + (size_t)i * 64, *ui = u + (size_t)i * K;
        for (int k = 0; k < Hd; k++) { const float *w = W1 + (size_t)k * LCD_NF; float s = b1[k]; for (int j = 0; j < LCD_NF; j++) s += w[j] * fi[j]; a[k] = s; h[k] = silu(s); }
        double r = 0;
        for (int c = 0; c < K; c++) {
            float s = 0; const float *wl = Wl + (size_t)c * LCD_NAX, *w2 = W2 + (size_t)c * Hd;
            for (int j = 0; j < LCD_NAX; j++) s += wl[j] * xi[1 + j];
            for (int k = 0; k < Hd; k++) s += w2[k] * h[k];
            yt[c] = s + ui[c];
            double dt, dl; r += lap_bits(yt[c], exp(lb[c]), &dt, &dl);
            dy[c] = (float)(dt / 64 * inv); g[o.lb + c] += (float)(dl / 64 * inv);
        }
        double e2 = 0; const double ce = 2 * lambda * sd2 / 64 * inv;
        for (int j = 0; j < LCD_NAX; j++) {
            float xr = 0; for (int c = 0; c < K; c++) xr += S[(size_t)c * LCD_NAX + j] * yt[c];
            const float e = xr - xi[1 + j]; e2 += (double)e * e;
            for (int c = 0; c < K; c++) { dy[c] += (float)(ce * e * S[(size_t)c * LCD_NAX + j]); g[o.S + (size_t)c * LCD_NAX + j] += (float)(ce * e * yt[c]); }
        }
        Rs += r / 64; Ds += e2 * sd2 / 64;
        for (int k = 0; k < Hd; k++) dh[k] = 0;
        for (int c = 0; c < K; c++) {
            float *gl = g + o.Wl + (size_t)c * LCD_NAX, *g2 = g + o.W2 + (size_t)c * Hd; const float d = dy[c], *w2 = W2 + (size_t)c * Hd;
            for (int j = 0; j < LCD_NAX; j++) gl[j] += d * xi[1 + j];
            for (int k = 0; k < Hd; k++) { g2[k] += d * h[k]; dh[k] += d * w2[k]; }
        }
        for (int k = 0; k < Hd; k++) {
            const float sg = 1.f / (1.f + expf(-a[k])), da = dh[k] * sg * (1.f + a[k] * (1.f - sg));
            if (da == 0.f) continue;
            float *g1 = g + o.W1 + (size_t)k * LCD_NF; for (int j = 0; j < LCD_NF; j++) g1[j] += da * fi[j];
            g[o.b1 + k] += da;
        }
    }
    *R = Rs * inv; *D = Ds * inv;
    return (Rs + lambda * Ds) * inv;
}
int lcodec_lv(const lcodec *lc, const lat_basis *lb, const uint64_t (*hist)[LV_NSYM], lv_codec *out) {
    const lcodec_off o = lcodec_offs(lc->cl, lc->hid); const int K = lc->cl - 1; float rows[64 * 64];
    for (int k = 0; k < 64; k++) rows[k] = lb->b[0][k];
    for (int c = 0; c < K; c++) for (int k = 0; k < 64; k++) {
        double s = 0; for (int j = 0; j < LCD_NAX; j++) s += (double)lc->p[o.S + (size_t)c * LCD_NAX + j] * lb->b[1 + j][k];
        rows[(1 + c) * 64 + k] = (float)(s / lc->mstep);
        if (fabsf(rows[(1 + c) * 64 + k]) >= 1.99f) { fprintf(stderr, "lcodec: synthesis row %d exceeds the int16 basis range (%.3f)\n", 1 + c, rows[(1 + c) * 64 + k]); return -1; }
    }
    return lv_codec_init(out, lc->cl, rows, lc->mstep, lc->sd_ref, nullptr, 0, hist);
}
int lcodec_write(const lcodec *lc, const char *path) {
    FILE *f = fopen(path, "wb"); if (!f) return -1;
    const int32_t hd[2] = {lc->cl, lc->hid}; const float fl[3] = {lc->sd_ref, lc->mstep, lc->mean_c}; const uint64_t np = lc->np;
    int ok = fwrite("LCD1", 1, 4, f) == 4 && fwrite(hd, 4, 2, f) == 2 && fwrite(fl, 4, 3, f) == 3 && fwrite(&np, 8, 1, f) == 1 && fwrite(lc->p, 4, lc->np, f) == lc->np;
    ok &= fclose(f) == 0;
    return ok ? 0 : -1;
}
int lcodec_read(lcodec *lc, const char *path) {
    memset(lc, 0, sizeof *lc); FILE *f = fopen(path, "rb"); if (!f) return -1;
    char m[4]; int32_t hd[2]; float fl[3]; uint64_t np = 0;
    int ok = fread(m, 1, 4, f) == 4 && !memcmp(m, "LCD1", 4) && fread(hd, 4, 2, f) == 2 && fread(fl, 4, 3, f) == 3 && fread(&np, 8, 1, f) == 1
             && hd[0] >= 2 && hd[0] <= 64 && hd[1] >= 1 && hd[1] <= 256 && np == lcodec_np(hd[0], hd[1]);
    if (ok) { lc->cl = hd[0]; lc->hid = hd[1]; lc->sd_ref = fl[0]; lc->mstep = fl[1]; lc->mean_c = fl[2]; lc->np = np; lc->p = malloc(np * 4); ok = lc->p && fread(lc->p, 4, np, f) == np; }
    fclose(f);
    if (!ok) { lcodec_free(lc); return -1; }
    return 0;
}
uint64_t lcodec_hash(const lcodec *lc) {
    uint64_t h = 1469598103934665603ull; const int32_t hd[2] = {lc->cl, lc->hid}; const float fl[3] = {lc->sd_ref, lc->mstep, lc->mean_c};
#define LCH(ptr, n) do { const uint8_t *b_ = (const uint8_t *)(ptr); for (size_t i_ = 0; i_ < (n); i_++) { h ^= b_[i_]; h *= 1099511628211ull; } } while (0)
    LCH(hd, sizeof hd); LCH(fl, sizeof fl); LCH(lc->p, lc->np * 4);
#undef LCH
    return h;
}
