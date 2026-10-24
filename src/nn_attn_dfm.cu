/* Deformable attention (src/attn.h at_dfm_*): multi-scale 3D Deformable-DETR sampling for the bottleneck blocks (at_dims.dK > 0).
   Per grid token t, head h, level l and point k: a position on level l's grid (the token's centre there plus a predicted offset in
   level-l voxels, optionally rotated into the token's normal / tangent / axis frame), the head's V slice trilinearly interpolated there
   (zero outside the level's grid), weighted by a softmax over the head's L x K logits. Levels 0..dL-1 are the bottleneck and finer grids,
   level dL (dCo) the 2x average-pooled bottleneck values. One warp per (token, head): lane j holds head channels j and 32 + j (j < 16), so
   every corner read / atomic is one 192-byte row. fp32 throughout; the backward scatters the V gradient with atomics. Default stream. */
#include "attn.h"
#include <cuda_runtime.h>
#include <math.h>

#define HD AT_HD
static unsigned nbk(long long n, int t) { return (unsigned)((n + t - 1) / t); }

/* per sampled level: values, their gradient, grid dims, voxels per sample and the scale (level voxels per bottleneck voxel) */
struct dfm_lv { const float *v[AT_DFM_NLV]; float *dv[AT_DFM_NLV]; int d[AT_DFM_NLV], h[AT_DFM_NLV], w[AT_DFM_NLV]; long long sl[AT_DFM_NLV]; float f[AT_DFM_NLV]; };

/* softmax of each (t, h)'s L K logits (oa row: H L K 3 offsets, then H L K logits) -> aw [T][H L K] */
__global__ void k_dfm_softmax(const float *oa, float *aw, long long T, int H, int LK, int NOA) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= T * H) return;
    const long long t = i / H; const int h = (int)(i % H);
    const float *lg = oa + t * NOA + (size_t)H * LK * 3 + (size_t)h * LK; float *a = aw + t * (long long)H * LK + (size_t)h * LK;
    float m = -INFINITY; for (int j = 0; j < LK; j++) m = fmaxf(m, lg[j]);
    float s = 0.f; for (int j = 0; j < LK; j++) { const float e = expf(lg[j] - m); a[j] = e; s += e; }
    const float r = 1.f / s; for (int j = 0; j < LK; j++) a[j] *= r;
}
/* the warp's (t, h) and its grid position */
struct dfm_tok { long long t; int h, s, z, y, x; };
__device__ __forceinline__ bool dfm_warp(long long T, int S, int Hh, int W, int H, dfm_tok &k) {
    const long long wi = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5;
    if (wi >= T * H) return false;
    k.t = wi / H; k.h = (int)(wi % H); k.s = (int)(k.t / S); const int g = (int)(k.t % S);
    k.z = g / (Hh * W); k.y = (g / W) % Hh; k.x = g % W;
    return true;
}
/* the offset in grid axes: of itself, or rotated by the token's frame R (rows normal, tangent, axis as z y x) */
__device__ __forceinline__ void dfm_off(const float *of, const float *R, float &oz, float &oy, float &ox) {
    if (!R) { oz = of[0]; oy = of[1]; ox = of[2]; return; }
    oz = of[0] * R[0] + of[1] * R[3] + of[2] * R[6];
    oy = of[0] * R[1] + of[1] * R[4] + of[2] * R[7];
    ox = of[0] * R[2] + of[1] * R[5] + of[2] * R[8];
}
/* o[t][h's channels] = sum_{l,k} aw sample(V_l, p) */
__global__ void k_dfm_fwd(dfm_lv lv, const float *oa, const float *aw, const float *fr, float *o, long long T, int S, int D, int Hh, int W, int C, int H, int L, int K, int NOA) {
    dfm_tok tk; if (!dfm_warp(T, S, Hh, W, H, tk)) return;
    const int ln = threadIdx.x & 31; const bool two = ln < HD - 32;
    const float *oat = oa + tk.t * NOA, *a = aw + tk.t * (long long)H * L * K + (size_t)tk.h * L * K, *R = fr ? fr + tk.t * 9 : nullptr;
    float acc0 = 0.f, acc1 = 0.f;
    for (int l = 0; l < L; l++) {
        const int Dl = lv.d[l], Hl = lv.h[l], Wl = lv.w[l]; const long long Sl = lv.sl[l]; const float f = lv.f[l];
        const float *V = lv.v[l] + (long long)tk.s * Sl * C + tk.h * HD + ln;
        const float cz = (tk.z + 0.5f) * f - 0.5f, cy = (tk.y + 0.5f) * f - 0.5f, cx = (tk.x + 0.5f) * f - 0.5f;
        for (int k = 0; k < K; k++) {
            const float *of = oat + (((size_t)tk.h * L + l) * K + k) * 3; const float ak = a[l * K + k];
            float oz, oy, ox; dfm_off(of, R, oz, oy, ox);
            const float pz = cz + oz, py = cy + oy, px = cx + ox;
            const float fz0 = floorf(pz), fy0 = floorf(py), fx0 = floorf(px); const int z0 = (int)fz0, y0 = (int)fy0, x0 = (int)fx0;
            const float fz = pz - fz0, fy = py - fy0, fx = px - fx0;
#pragma unroll
            for (int q = 0; q < 8; q++) {
                const int z = z0 + (q >> 2), y = y0 + ((q >> 1) & 1), x = x0 + (q & 1);
                if (z < 0 || z >= Dl || y < 0 || y >= Hl || x < 0 || x >= Wl) continue;
                const float w = ak * ((q >> 2) ? fz : 1.f - fz) * (((q >> 1) & 1) ? fy : 1.f - fy) * ((q & 1) ? fx : 1.f - fx);
                const float *vr = V + (((long long)z * Hl + y) * Wl + x) * C;
                acc0 += w * vr[0]; if (two) acc1 += w * vr[32];
            }
        }
    }
    float *orow = o + tk.t * C + tk.h * HD + ln;
    orow[0] = acc0; if (two) orow[32] = acc1;
}
/* per (t, h): dV_l[corner] += aw w dO (atomics); per point da = dO . sample (the logit slot, before the softmax backward) and
   d offset = aw dO . d sample / d p (through the frame: R d p) */
__global__ void k_dfm_bwd(dfm_lv lv, const float *oa, const float *aw, const float *fr, const float *dO, float *doa, long long T, int S, int D, int Hh, int W, int C, int H, int L, int K, int NOA) {
    dfm_tok tk; if (!dfm_warp(T, S, Hh, W, H, tk)) return;
    const int ln = threadIdx.x & 31; const bool two = ln < HD - 32; const int LK = L * K;
    const float *oat = oa + tk.t * NOA, *a = aw + tk.t * (long long)H * LK + (size_t)tk.h * LK, *R = fr ? fr + tk.t * 9 : nullptr;
    const float g0 = dO[tk.t * C + tk.h * HD + ln], g1 = two ? dO[tk.t * C + tk.h * HD + 32 + ln] : 0.f;
    float *dofs = doa + tk.t * NOA, *dlg = doa + tk.t * NOA + (size_t)H * LK * 3 + (size_t)tk.h * LK;
    for (int l = 0; l < L; l++) {
        const int Dl = lv.d[l], Hl = lv.h[l], Wl = lv.w[l]; const long long Sl = lv.sl[l]; const float f = lv.f[l];
        const long long base = (long long)tk.s * Sl * C + tk.h * HD + ln;
        const float *V = lv.v[l] + base; float *dV = lv.dv[l] ? lv.dv[l] + base : nullptr;
        const float cz = (tk.z + 0.5f) * f - 0.5f, cy = (tk.y + 0.5f) * f - 0.5f, cx = (tk.x + 0.5f) * f - 0.5f;
        for (int k = 0; k < K; k++) {
            const size_t pi = ((size_t)tk.h * L + l) * K + k; const float *of = oat + pi * 3; const float ak = a[l * K + k];
            float oz, oy, ox; dfm_off(of, R, oz, oy, ox);
            const float pz = cz + oz, py = cy + oy, px = cx + ox;
            const float fz0 = floorf(pz), fy0 = floorf(py), fx0 = floorf(px); const int z0 = (int)fz0, y0 = (int)fy0, x0 = (int)fx0;
            const float fz = pz - fz0, fy = py - fy0, fx = px - fx0;
            float da = 0.f, gz = 0.f, gy = 0.f, gx = 0.f;
#pragma unroll
            for (int q = 0; q < 8; q++) {
                const int dz = q >> 2, dy = (q >> 1) & 1, dx = q & 1, z = z0 + dz, y = y0 + dy, x = x0 + dx;
                if (z < 0 || z >= Dl || y < 0 || y >= Hl || x < 0 || x >= Wl) continue;
                const float az = dz ? fz : 1.f - fz, ay = dy ? fy : 1.f - fy, ax = dx ? fx : 1.f - fx;
                const long long r = (((long long)z * Hl + y) * Wl + x) * C;
                const float v0 = V[r], v1 = two ? V[r + 32] : 0.f, dot = g0 * v0 + g1 * v1, w = az * ay * ax;
                da += w * dot; gz += (dz ? 1.f : -1.f) * ay * ax * dot; gy += az * (dy ? 1.f : -1.f) * ax * dot; gx += az * ay * (dx ? 1.f : -1.f) * dot;
                if (dV) { const float aw_ = ak * w; atomicAdd(dV + r, aw_ * g0); if (two) atomicAdd(dV + r + 32, aw_ * g1); }
            }
#pragma unroll
            for (int s = 16; s; s >>= 1) {
                da += __shfl_xor_sync(0xffffffffu, da, s); gz += __shfl_xor_sync(0xffffffffu, gz, s);
                gy += __shfl_xor_sync(0xffffffffu, gy, s); gx += __shfl_xor_sync(0xffffffffu, gx, s);
            }
            if (ln == 0) {
                float *d3 = dofs + pi * 3;
                if (!R) { d3[0] = ak * gz; d3[1] = ak * gy; d3[2] = ak * gx; }
                else for (int j = 0; j < 3; j++) d3[j] = ak * (R[3 * j] * gz + R[3 * j + 1] * gy + R[3 * j + 2] * gx);
                dlg[l * K + k] = da;
            }
        }
    }
}
/* softmax backward in the logit slots: dlogit = a (da - sum a da), thread per (t, h) */
__global__ void k_dfm_smbwd(const float *aw, float *doa, long long T, int H, int LK, int NOA) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= T * H) return;
    const long long t = i / H; const int h = (int)(i % H);
    const float *a = aw + t * (long long)H * LK + (size_t)h * LK; float *d = doa + t * NOA + (size_t)H * LK * 3 + (size_t)h * LK;
    float s = 0.f; for (int j = 0; j < LK; j++) s += a[j] * d[j];
    for (int j = 0; j < LK; j++) d[j] = a[j] * (d[j] - s);
}
/* the coarse level: thread per (parent, channel); mean of the children inside the grid */
__global__ void k_dfm_pool(const float *v0, float *vc, long long n, int D, int Hh, int W, int C) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    const int Dc = (D + 1) / 2, Hc = (Hh + 1) / 2, Wc = (W + 1) / 2; const long long Sc = (long long)Dc * Hc * Wc;
    if (i >= n * Sc * C) return;
    const int c = (int)(i % C); const long long pr = i / C, s = pr / Sc; const int g = (int)(pr % Sc), zc = g / (Hc * Wc), yc = (g / Wc) % Hc, xc = g % Wc;
    float a = 0.f; int cnt = 0;
    for (int q = 0; q < 8; q++) {
        const int z = 2 * zc + (q >> 2), y = 2 * yc + ((q >> 1) & 1), x = 2 * xc + (q & 1);
        if (z >= D || y >= Hh || x >= W) continue;
        a += v0[((s * D + z) * Hh + y) * (long long)W * C + (long long)x * C + c]; cnt++;
    }
    vc[i] = a / (float)cnt;
}
/* its adjoint, added: thread per (child, channel) */
__global__ void k_dfm_unpool(const float *dvc, float *dv0, long long n, int D, int Hh, int W, int C) {
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    const long long S = (long long)D * Hh * W;
    if (i >= n * S * C) return;
    const int c = (int)(i % C); const long long r = i / C, s = r / S; const int g = (int)(r % S), z = g / (Hh * W), y = (g / W) % Hh, x = g % W;
    const int Dc = (D + 1) / 2, Hc = (Hh + 1) / 2, Wc = (W + 1) / 2, zc = z / 2, yc = y / 2, xc = x / 2;
    const int cnt = ((2 * zc + 1 < D) + 1) * ((2 * yc + 1 < Hh) + 1) * ((2 * xc + 1 < W) + 1);
    dv0[i] += dvc[(((s * Dc + zc) * Hc + yc) * Wc + xc) * C + c] / (float)cnt;
}
/* token frames: thread per grid token */
__global__ void k_dfm_frame(long long n, int S, int D, int Hh, int W, int P, const int *ad, const float *cen, float *fr) {
    const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int b = (int)(t / S), ts = (int)(t % S);
    const int z = ts / (Hh * W), y = (ts / W) % Hh, x = ts % W, a_ = ad[b];
    const int k = a_ == 0 ? z : a_ == 1 ? y : x, ua = a_ == 0 ? y : z, vb = a_ == 2 ? y : x, ud = a_ == 0 ? 1 : 0, vd = a_ == 2 ? 1 : 2;
    const float *c = cen + ((size_t)b * P + k) * 4;
    float *R = fr + 9 * t;
    for (int j = 0; j < 9; j++) R[j] = 0.f;
    const float du = (float)ua - c[0], dv = (float)vb - c[1], r = sqrtf(du * du + dv * dv);
    if (c[2] < 0.f || !(r > 1e-3f)) { R[0] = R[4] = R[8] = 1.f; return; }
    const float nu = du / r, nv = dv / r;
    R[ud] = nu; R[vd] = nv;              /* normal: radial in the axis-normal plane */
    R[3 + ud] = -nv; R[3 + vd] = nu;     /* tangent */
    R[6 + a_] = 1.f;                     /* axis */
}

/* LayerNorm without affine, one warp per row */
__global__ void k_ln_plain(const float *x, float *y, long long T, int C) {
    const long long r = (blockIdx.x * (long long)blockDim.x + threadIdx.x) >> 5; const int ln = threadIdx.x & 31;
    if (r >= T) return;
    const float *xr = x + r * C; float *yr = y + r * C;
    float s = 0.f; for (int c = ln; c < C; c += 32) s += xr[c];
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float m = s / C; float v = 0.f;
    for (int c = ln; c < C; c += 32) { const float e = xr[c] - m; v += e * e; }
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const float rs = rsqrtf(v / C + 1e-5f);
    for (int c = ln; c < C; c += 32) yr[c] = (xr[c] - m) * rs;
}
extern "C" void at_ln_plain(const float *x, float *y, long T, int C) { k_ln_plain<<<nbk((long long)T * 32, 256), 256>>>(x, y, T, C); }
extern "C" void at_dfm_prior(const at_dims *d, float *boa) {
    const int H = d->H, L = at_dfm_nl(d), K = d->dK, N = H * K;
    for (int h = 0; h < H; h++) for (int l = 0; l < L; l++) for (int k = 0; k < K; k++) {
        const double sc = l < d->dL ? (double)(1 << l) : 1.0;   /* (the coarse level: its own voxels) */
        float *o = boa + (((size_t)h * L + l) * K + k) * 3;
        if (d->dInit == 1) {   /* along the normal over +-2 bottleneck voxels, the head shifted sideways */
            const double s = 2.0 * (k + 0.5) / K - 1.0, an = 6.283185307179586 * h / H, lat = H > 1 ? 0.35 : 0.0;
            o[0] = (float)(2.0 * s * sc); o[1] = (float)(lat * cos(an) * sc); o[2] = (float)(lat * sin(an) * sc);
            continue;
        }
        const int j = h * K + k; const double z = 1.0 - 2.0 * (j + 0.5) / N, r = sqrt(1.0 - z * z), ph = 2.399963229728653 * j;
        const double rad = 0.5 * (1 + k % 4) * sc;
        o[0] = (float)(rad * z); o[1] = (float)(rad * r * sin(ph)); o[2] = (float)(rad * r * cos(ph));
    }
}
static dfm_lv mklv(const float *const *v, float *const *dv, const at_geom *g, const at_dims *d) {
    dfm_lv r; const int L = at_dfm_nl(d);
    for (int l = 0; l < AT_DFM_NLV; l++) {
        r.v[l] = l < L ? v[l] : nullptr; r.dv[l] = dv && l < L ? dv[l] : nullptr;
        if (l < d->dL) { r.d[l] = g->D << l; r.h[l] = g->Hh << l; r.w[l] = g->W << l; r.f[l] = (float)(1 << l); r.sl[l] = (long long)g->S << (3 * l); }
        else { r.d[l] = (g->D + 1) / 2; r.h[l] = (g->Hh + 1) / 2; r.w[l] = (g->W + 1) / 2; r.f[l] = 0.5f; r.sl[l] = (long long)r.d[l] * r.h[l] * r.w[l]; }
    }
    return r;
}
extern "C" void at_dfm_pool(const float *v0, float *vc, const at_geom *g, int C) {
    const long long n = (long long)g->nseq * ((g->D + 1) / 2) * ((g->Hh + 1) / 2) * ((g->W + 1) / 2) * C;
    k_dfm_pool<<<nbk(n, 256), 256>>>(v0, vc, g->nseq, g->D, g->Hh, g->W, C);
}
extern "C" void at_dfm_frame(int n, int D, int Hh, int W, const int *ad, const float *cen, float *fr) {
    const int S = D * Hh * W, P = D > Hh ? (D > W ? D : W) : (Hh > W ? Hh : W); const long long N = (long long)n * S;
    k_dfm_frame<<<nbk(N, 256), 256>>>(N, S, D, Hh, W, P, ad, cen, fr);
}
extern "C" void at_dfm_softmax(const float *oa, float *aw, long T, const at_dims *d) {
    const int LK = at_dfm_nl(d) * d->dK, NOA = at_dfm_noa(d);
    k_dfm_softmax<<<nbk((long long)T * d->H, 256), 256>>>(oa, aw, T, d->H, LK, NOA);
}
extern "C" void at_dfm_fwd(const float *const *vl, const float *oa, const float *aw, float *o, const at_geom *g, const at_dims *d) {
    const long long T = (long long)g->nseq * g->S;
    k_dfm_fwd<<<nbk(T * d->H * 32, 256), 256>>>(mklv(vl, nullptr, g, d), oa, aw, d->dInit ? g->dfr : nullptr, o, T, g->S, g->D, g->Hh, g->W, d->C, d->H, at_dfm_nl(d), d->dK, at_dfm_noa(d));
}
extern "C" void at_dfm_bwd(const float *const *vl, const float *oa, const float *aw, const float *dO, float *const *dvl, float *doa, const at_geom *g, const at_dims *d) {
    const long long T = (long long)g->nseq * g->S; const int L = at_dfm_nl(d), K = d->dK, H = d->H, C = d->C, NOA = at_dfm_noa(d);
    const dfm_lv lv = mklv(vl, dvl, g, d);
    for (int l = 0; l < L; l++) if (dvl[l]) cudaMemsetAsync(dvl[l], 0, (size_t)g->nseq * lv.sl[l] * C * 4, 0);
    k_dfm_bwd<<<nbk(T * H * 32, 256), 256>>>(lv, oa, aw, d->dInit ? g->dfr : nullptr, dO, doa, T, g->S, g->D, g->Hh, g->W, C, H, L, K, NOA);
    k_dfm_smbwd<<<nbk(T * H, 256), 256>>>(aw, doa, T, H, L * K, NOA);
    if (d->dCo && dvl[0] && dvl[d->dL]) k_dfm_unpool<<<nbk(T * C, 256), 256>>>(dvl[d->dL], dvl[0], g->nseq, g->D, g->Hh, g->W, C);
}
