/* Task restore on the GPU (src/restore.h): the degradation families built from the clean window, the volcomp codec round trip, the
   network input, the residual Charbonnier / MSE / edge loss and the output CT. */
#include "nn_common.cuh"
#include "restore.h"

/* ---- separable resampling: one axis of an (A, N, I) array (outer, axis, inner) to (A, n, I) floats ---- */
struct rs_axis { int A, N, I, n; float step, phase, sigma; int box; };
template <typename T> __global__ void restore_resample_k(const T *__restrict__ src, float *__restrict__ dst, rs_axis g, size_t total) {
    const size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= total) return;
    const size_t inner = k % g.I, rest = k / g.I; const int i = (int)(rest % g.n); const size_t a = rest / g.n;
    const T *s = src + a * (size_t)g.N * g.I + inner;
    float acc = 0.f;
    if (g.box) {   /* exact block mean (integer step, phase 0): fine voxels [i s, i s + s) */
        const int st = (int)g.step, j0 = i * st;
        for (int t = 0; t < st; t++) { const int j = min(g.N - 1, j0 + t); acc += (float)s[(size_t)j * g.I]; }
        acc /= (float)st;
    } else {       /* Gaussian of sigma fine voxels sampled at the coarse centre, edge-replicated */
        const float xc = ((float)i + 0.5f - g.phase) * g.step - 0.5f, sg = fmaxf(g.sigma, 0.3f), inv = 0.5f / (sg * sg);
        const int r = (int)ceilf(4.f * sg), jc = (int)floorf(xc);
        float ws = 0.f;
        for (int j = jc - r; j <= jc + r + 1; j++) {
            const float d = (float)j - xc, w = __expf(-d * d * inv);
            const int jj = min(g.N - 1, max(0, j));
            acc += w * (float)s[(size_t)jj * g.I]; ws += w;
        }
        acc /= ws;
    }
    dst[k] = acc;
}
template <typename T> static void resample_axis(const T *src, float *dst, int A, int N, int I, int n, float step, float phase, float sigma, int box) {
    rs_axis g = {A, N, I, n, step, phase, sigma, box};
    const size_t total = (size_t)A * n * I;
    restore_resample_k<T><<<nblk(total, 256), 256>>>(src, dst, g, total);
}

__device__ __forceinline__ unsigned restore_hash(unsigned x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16); }
__device__ __forceinline__ float restore_normal(unsigned i, unsigned seed) {
    const float u = ((restore_hash(i ^ seed) >> 8) + .5f) / 16777216.f, v = ((restore_hash(i ^ seed ^ 0x9e3779b9u) >> 8) + .5f) / 16777216.f;
    return sqrtf(-2.f * __logf(u)) * __cosf(6.28318530718f * v);
}
/* the coarse grid as stored: scanner corruption (edge term, tone, correlated signal-dependent noise) or plain rounding; a coarse voxel
   that is exactly 0 (all its fine voxels outside the scan) stays 0 */
__global__ void restore_coarse_k(const float *__restrict__ src, uint8_t *__restrict__ dst, restore_recipe g, int scanner, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = src[i];
    if (v <= 0.f) { dst[i] = 0; return; }
    if (scanner) {
        const int D = g.small[0], H = g.small[1], W = g.small[2];
        const int z = (int)(i / ((size_t)H * W)), y = (int)(i / W) % H, x = (int)(i % W);
        const size_t nb[6] = {((size_t)max(0, z - 1) * H + y) * W + x, ((size_t)min(D - 1, z + 1) * H + y) * W + x,
                              ((size_t)z * H + max(0, y - 1)) * W + x, ((size_t)z * H + min(H - 1, y + 1)) * W + x,
                              ((size_t)z * H + y) * W + max(0, x - 1), ((size_t)z * H + y) * W + min(W - 1, x + 1)};
        if (g.edge != 0.f) { float s = 0.f; for (int k = 0; k < 6; k++) s += src[nb[k]]; v += g.edge * (v - s / 6.f); }
        v = 255.f * g.gain * __powf(fminf(255.f, fmaxf(0.f, v)) / 255.f, g.gamma) + g.bias;
        if (g.noise > 0.f || g.sig_var > 0.f) {
            float e = restore_normal((unsigned)i, g.seed);
            if (g.corr > 0.f) { for (int k = 0; k < 6; k++) e += g.corr * restore_normal((unsigned)nb[k], g.seed); e *= rsqrtf(1.f + 6.f * g.corr * g.corr); }
            v += e * sqrtf(g.noise * g.noise + g.sig_var * fmaxf(v, 0.f));
        }
    }
    dst[i] = (uint8_t)fminf(255.f, fmaxf(1.f, rintf(v)));
}

/* ---- trilinear upsampling of the coarse grid back onto the fine grid ---- */
struct up_geo { int sd, sh, sw, D, H, W; float step[3], phase[3]; };
__global__ void restore_up_k(const uint8_t *__restrict__ s, uint8_t *__restrict__ out, up_geo g, const uint8_t *__restrict__ mask, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int z = (int)(i / ((size_t)g.H * g.W)), y = (int)(i / g.W) % g.H, x = (int)(i % g.W);
    const float cz = ((float)z + .5f) / g.step[0] - .5f + g.phase[0], cy = ((float)y + .5f) / g.step[1] - .5f + g.phase[1], cx = ((float)x + .5f) / g.step[2] - .5f + g.phase[2];
    const int iz = (int)floorf(cz), iy = (int)floorf(cy), ix = (int)floorf(cx);
    const float tz = cz - iz, ty = cy - iy, tx = cx - ix;
    if (mask) { if (!mask[i]) { out[i] = 0; return; } }
    else {   /* the nearest coarse voxel decides the mask */
        const int nz = min(g.sd - 1, max(0, (int)floorf(cz + .5f))), ny = min(g.sh - 1, max(0, (int)floorf(cy + .5f))), nx = min(g.sw - 1, max(0, (int)floorf(cx + .5f)));
        if (!s[((size_t)nz * g.sh + ny) * g.sw + nx]) { out[i] = 0; return; }
    }
    float acc = 0.f;
    for (int dz = 0; dz < 2; dz++) for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
        const int a = min(g.sd - 1, max(0, iz + dz)), b = min(g.sh - 1, max(0, iy + dy)), c = min(g.sw - 1, max(0, ix + dx));
        acc += (dz ? tz : 1.f - tz) * (dy ? ty : 1.f - ty) * (dx ? tx : 1.f - tx) * (float)s[((size_t)a * g.sh + b) * g.sw + c];
    }
    out[i] = (uint8_t)fminf(255.f, fmaxf(1.f, rintf(acc)));
}
extern "C" void nn_restore_upsample(const uint8_t *small, int sd, int sh, int sw, uint8_t *out, int D, int H, int W, const float step[3], const float phase[3], const uint8_t *mask) {
    up_geo g = {sd, sh, sw, D, H, W, {step[0], step[1], step[2]}, {phase[0], phase[1], phase[2]}};
    const size_t n = (size_t)D * H * W;
    restore_up_k<<<nblk(n, 256), 256>>>(small, out, g, mask, n); KCHECK();
}

/* ---- volcomp lossy round trip: orthonormal 16^3 DCT-II per block, dead-zone quantiser (AC: trunc(|c| / step + 0.2), DC: rounded),
   reconstruction (|l| + 0.15 for |l| = 1, + 0.30 above) step, step(r) = q (1 + r)^0.65 for r = kz + ky + kx >= 1, step(0) = q / 8
   (third_party/volcomp.h vf_steps / vf_quantise_block / vf_dequant_*), output trunc(v + 128.5) clamped ---- */
__constant__ float c_dct16[16][16];   /* c_dct16[k][n] = s(k) cos(pi (2n + 1) k / 32), s(0) = 1/4, s(k) = sqrt(2) / 4 */
struct codec_geo { int d, h, w, pz, py, px, nby, nbx; float step[46]; };
__global__ void __launch_bounds__(256) restore_codec_k(uint8_t *vol, codec_geo g) {
    __shared__ float b[4096];
    const int bid = blockIdx.x, bx = bid % g.nbx, by = (bid / g.nbx) % g.nby, bz = bid / (g.nbx * g.nby);
    const int z0 = bz * 16 - g.pz, y0 = by * 16 - g.py, x0 = bx * 16 - g.px, t = threadIdx.x;
    for (int k = t; k < 4096; k += 256) {
        const int z = min(g.d - 1, max(0, z0 + (k >> 8))), y = min(g.h - 1, max(0, y0 + ((k >> 4) & 15))), x = min(g.w - 1, max(0, x0 + (k & 15)));
        b[k] = (float)vol[((size_t)z * g.h + y) * g.w + x] - 128.f;
    }
    __syncthreads();
    float l[16], o[16];
    /* forward: x lines (t = z y), y lines (t = z x), z lines (t = y x) */
    for (int pass = 0; pass < 3; pass++) {
        const int hi = t >> 4, lo = t & 15;
        const int base = pass == 0 ? hi * 256 + lo * 16 : pass == 1 ? hi * 256 + lo : hi * 16 + lo, str = pass == 0 ? 1 : pass == 1 ? 16 : 256;
        for (int n = 0; n < 16; n++) l[n] = b[base + n * str];
        for (int k = 0; k < 16; k++) { float s = 0.f; for (int n = 0; n < 16; n++) s += c_dct16[k][n] * l[n]; o[k] = s; }
        __syncthreads();
        for (int k = 0; k < 16; k++) b[base + k * str] = o[k];
        __syncthreads();
    }
    for (int k = t; k < 4096; k += 256) {
        const int r = (k >> 8) + ((k >> 4) & 15) + (k & 15);
        const float c = b[k], a = fabsf(c), st = g.step[r];
        float rec;
        if (!r) { const int lv = (int)(a / st + 0.5f); rec = (float)lv * st; }
        else { const int lv = (int)(a / st + 0.2f); rec = lv ? ((float)lv + (lv == 1 ? 0.15f : 0.30f)) * st : 0.f; }
        b[k] = c < 0.f ? -rec : rec;
    }
    __syncthreads();
    for (int pass = 0; pass < 3; pass++) {
        const int hi = t >> 4, lo = t & 15;
        const int base = pass == 0 ? hi * 256 + lo * 16 : pass == 1 ? hi * 256 + lo : hi * 16 + lo, str = pass == 0 ? 1 : pass == 1 ? 16 : 256;
        for (int k = 0; k < 16; k++) l[k] = b[base + k * str];
        for (int n = 0; n < 16; n++) { float s = 0.f; for (int k = 0; k < 16; k++) s += c_dct16[k][n] * l[k]; o[n] = s; }
        __syncthreads();
        for (int n = 0; n < 16; n++) b[base + n * str] = o[n];
        __syncthreads();
    }
    for (int k = t; k < 4096; k += 256) {
        const int z = z0 + (k >> 8), y = y0 + ((k >> 4) & 15), x = x0 + (k & 15);
        if (z < 0 || z >= g.d || y < 0 || y >= g.h || x < 0 || x >= g.w) continue;
        vol[((size_t)z * g.h + y) * g.w + x] = (uint8_t)(int)fminf(fmaxf(b[k] + 128.5f, 0.f), 255.f);
    }
}
extern "C" void nn_codec_roundtrip(uint8_t *vol, int d, int h, int w, float q, const int phase[3]) {
    if (!(q > 0.f)) return;
    static int init_dev = -1; int dev = 0; cudaGetDevice(&dev);
    if (init_dev != dev) {   /* (per device: the constant table lives in each context) */
        float c[16][16];
        for (int k = 0; k < 16; k++) for (int n = 0; n < 16; n++) c[k][n] = (float)((k ? sqrt(2.0) / 4.0 : 0.25) * cos(M_PI * (2 * n + 1) * k / 32.0));
        CK(cudaMemcpyToSymbol(c_dct16, c, sizeof c)); init_dev = dev;
    }
    codec_geo g; g.d = d; g.h = h; g.w = w; g.pz = phase[0] & 15; g.py = phase[1] & 15; g.px = phase[2] & 15;
    const float qe = floorf(q * 256.f + 0.5f) / 256.f;   /* the step the decoder sees (vf_q_raw) */
    g.step[0] = qe * 0.125f;
    for (int r = 1; r < 46; r++) g.step[r] = qe * powf(1.f + (float)r, 0.65f);
    const int nbz = (d + g.pz + 15) / 16; g.nby = (h + g.py + 15) / 16; g.nbx = (w + g.px + 15) / 16;
    restore_codec_k<<<nbz * g.nby * g.nbx, 256>>>(vol, g); KCHECK();
}

/* ---- the degradation pipeline ---- */
static size_t a256(size_t b) { return (b + 255) & ~(size_t)255; }
static void layout(int D, int H, int W, const int s[3], size_t *ra, size_t *rb, size_t *rc) {
    *ra = a256((size_t)D * H * s[2] * 4); *rb = a256((size_t)D * s[1] * s[2] * 4); *rc = a256((size_t)s[0] * s[1] * s[2]);
    const size_t small4 = (size_t)s[0] * s[1] * s[2] * 4;   /* (pass z writes the coarse floats back into region A) */
    if (*ra < small4) *ra = a256(small4);
}
extern "C" size_t restore_scratch_bytes(int D, int H, int W, float min_step) {
    if (!(min_step >= 1.f)) min_step = 1.f;
    const int s[3] = {(int)ceilf(D / min_step) + 1, (int)ceilf(H / min_step) + 1, (int)ceilf(W / min_step) + 1};
    size_t a, b, c; layout(D, H, W, s, &a, &b, &c);
    return a + b + c;
}
extern "C" int nn_restore_degrade(const uint8_t *clean, uint8_t *low, int D, int H, int W, const restore_recipe *r, void *scratch, size_t scratch_bytes, uint8_t *coarse) {
    const size_t n = (size_t)D * H * W;
    if (r->family == RF_IDENTITY) { CK(cudaMemcpyAsync(low, clean, n, cudaMemcpyDeviceToDevice, 0)); return 0; }
    size_t ra, rb, rc; layout(D, H, W, r->small, &ra, &rb, &rc);
    if (ra + rb + rc > scratch_bytes) return -1;
    float *A = (float *)scratch, *Bf = (float *)((char *)scratch + ra); uint8_t *C = (uint8_t *)scratch + ra + rb;
    const int d = r->small[0], h = r->small[1], w = r->small[2];
    resample_axis<uint8_t>(clean, A, D * H, W, 1, w, r->step[2], r->phase[2], r->psf[2], r->box);   /* x: (D H, W) -> (D H, w) */
    resample_axis<float>(A, Bf, D, H, w, h, r->step[1], r->phase[1], r->psf[1], r->box);           /* y: (D, H, w) -> (D, h, w) */
    resample_axis<float>(Bf, A, 1, D, h * w, d, r->step[0], r->phase[0], r->psf[0], r->box);       /* z: (D, h w) -> (d, h w) */
    const size_t ns = (size_t)d * h * w;
    restore_coarse_k<<<nblk(ns, 256), 256>>>(A, C, *r, r->family == RF_SCANNER, ns);
    if (r->q > 0.f) nn_codec_roundtrip(C, d, h, w, r->q, r->qphase);
    if (coarse) CK(cudaMemcpyAsync(coarse, C, ns, cudaMemcpyDeviceToDevice, 0));
    nn_restore_upsample(C, d, h, w, low, D, H, W, r->step, r->phase, clean);
    KCHECK();
    return 0;
}

/* ---- network input ---- */
template <class H> __global__ void restore_input_k(const uint8_t *__restrict__ a, H *__restrict__ x, size_t n, int split) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = (float)a[i] - 128.f;
    if (!split) { x[i] = f2h<H>(v * 0.015625f); return; }
    const float k = rintf(v * 0.0625f);
    x[i] = f2h<H>(k * 0.25f); x[n + i] = f2h<H>((v - 16.f * k) * 0.0625f);
}
extern "C" void nn_restore_input(const uint8_t *low, void *x16, size_t n, int split, int h16) {
    if (h16) restore_input_k<f16><<<nblk(n, 256), 256>>>(low, (f16 *)x16, n, split);
    else restore_input_k<bf16><<<nblk(n, 256), 256>>>(low, (bf16 *)x16, n, split);
    KCHECK();
}

/* ---- loss ---- */
template <class L> __device__ __forceinline__ float rl_err(const L *lg, const uint8_t *lo, const uint8_t *hi, size_t i) {
    return (float)lg[i] + ((float)lo[i] - (float)hi[i]) * 0.015625f;
}
struct rl_geo { int D, H, W; float eps, mw, ew, gs; };
template <class L> __global__ void restore_stats_k(const L *__restrict__ lg, const uint8_t *__restrict__ lo, const uint8_t *__restrict__ hi, rl_geo g, double *st, size_t n) {
    double s[7] = {0, 0, 0, 0, 0, 0, 0};
    const size_t str[3] = {(size_t)g.H * g.W, (size_t)g.W, 1}; const int len[3] = {g.D, g.H, g.W};
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)blockDim.x * gridDim.x) {
        if (!hi[i]) continue;
        const float e = rl_err(lg, lo, hi, i), dn = 64.f * e, base = (float)lo[i] - (float)hi[i];
        s[0] += 1; s[1] += sqrtf(e * e + g.eps * g.eps) - g.eps; s[2] += fabsf(dn); s[3] += (double)dn * dn; s[4] += (double)base * base;
        if (g.ew != 0.f) for (int a = 0; a < 3; a++) {
            if ((int)((i / str[a]) % len[a]) == len[a] - 1 || !hi[i + str[a]]) continue;
            const float df = e - rl_err(lg, lo, hi, i + str[a]);
            s[5] += 1; s[6] += sqrtf(df * df + g.eps * g.eps) - g.eps;
        }
    }
    __shared__ double r[7][256];
    for (int k = 0; k < 7; k++) r[k][threadIdx.x] = s[k];
    __syncthreads();
    for (int d = 128; d; d >>= 1) { if ((int)threadIdx.x < d) for (int k = 0; k < 7; k++) r[k][threadIdx.x] += r[k][threadIdx.x + d]; __syncthreads(); }
    if (!threadIdx.x) for (int k = 0; k < 7; k++) atomicAdd(st + k, r[k][0]);
}
template <class L, class G> __global__ void restore_grad_k(const L *__restrict__ lg, const uint8_t *__restrict__ lo, const uint8_t *__restrict__ hi, rl_geo g, G *__restrict__ gl,
                                                         const double *__restrict__ st, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = 0.f;
    if (hi[i] && st[0] > 0) {
        const float e = rl_err(lg, lo, hi, i), inv = (float)(1.0 / st[0]);
        v = (e * rsqrtf(e * e + g.eps * g.eps) + g.mw * e) * inv;
        if (g.ew != 0.f && st[5] > 0) {
            const float ie = g.ew * (float)(1.0 / st[5]);
            const size_t str[3] = {(size_t)g.H * g.W, (size_t)g.W, 1}; const int len[3] = {g.D, g.H, g.W};
            for (int a = 0; a < 3; a++) {
                const int c = (int)((i / str[a]) % len[a]);
                if (c > 0 && hi[i - str[a]]) { const float df = e - rl_err(lg, lo, hi, i - str[a]); v += ie * df * rsqrtf(df * df + g.eps * g.eps); }
                if (c < len[a] - 1 && hi[i + str[a]]) { const float df = e - rl_err(lg, lo, hi, i + str[a]); v += ie * df * rsqrtf(df * df + g.eps * g.eps); }
            }
        }
    }
    gl[i] = f2h<G>(v * g.gs);
}
template <class L> static void restore_loss_t(const L *lg, const uint8_t *lo, const uint8_t *hi, rl_geo g, void *gl, int g16, int h16, double *st, size_t n) {
    CK(cudaMemsetAsync(st, 0, 8 * sizeof(double), 0));
    int nb = (int)((n + 255) / 256); if (nb > 2048) nb = 2048;
    restore_stats_k<L><<<nb, 256>>>(lg, lo, hi, g, st, n);
    if (!gl) return;
    if (!g16) { g.gs = 1.f; restore_grad_k<L, float><<<nblk(n, 256), 256>>>(lg, lo, hi, g, (float *)gl, st, n); }
    else if (h16) restore_grad_k<L, f16><<<nblk(n, 256), 256>>>(lg, lo, hi, g, (f16 *)gl, st, n);
    else restore_grad_k<L, bf16><<<nblk(n, 256), 256>>>(lg, lo, hi, g, (bf16 *)gl, st, n);
}
extern "C" void nn_restore_loss(const void *lg, int lg16, const uint8_t *low, const uint8_t *clean, int D, int H, int W, float eps, float mse_w, float edge_w,
                                void *gl, int g16, int h16, float gscale, double *st) {
    const size_t n = (size_t)D * H * W;
    rl_geo g = {D, H, W, eps, mse_w, edge_w, gscale};
    if (lg16) restore_loss_t<f16>((const f16 *)lg, low, clean, g, gl, g16, h16, st, n);
    else restore_loss_t<float>((const float *)lg, low, clean, g, gl, g16, h16, st, n);
    KCHECK();
}

template <class L> __global__ void restore_output_k(const L *__restrict__ lg, const uint8_t *__restrict__ lo, uint8_t *__restrict__ out, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = lo[i] ? (uint8_t)fminf(255.f, fmaxf(1.f, rintf((float)lo[i] + 64.f * (float)lg[i]))) : 0;
}
extern "C" void nn_restore_output(const void *lg, int lg16, const uint8_t *low, uint8_t *out, size_t n) {
    if (lg16) restore_output_k<f16><<<nblk(n, 256), 256>>>((const f16 *)lg, low, out, n);
    else restore_output_k<float><<<nblk(n, 256), 256>>>((const float *)lg, low, out, n);
    KCHECK();
}

/* ---- real pairs (train --task restore on sr_pair sources): shift-tolerant Charbonnier, edge, MSE, downsample consistency; metrics ---- */
struct rp_geo { int D, H, W, q, shift; float eps, mw, ew, sw, dw, gs, uw; };
template <class L> __device__ __forceinline__ float rp_o(const L *lg, const uint8_t *lo, size_t i) { return (float)lg[i] + (float)lo[i] * 0.015625f; }
__device__ __forceinline__ float rp_ch(float e, float eps) { return sqrtf(e * e + eps * eps) - eps; }
__device__ __forceinline__ float rp_cd(float e, float eps) { return e * rsqrtf(e * e + eps * eps); }
template <int K> __device__ __forceinline__ void rp_reduce(double (&s)[K], double *st, int off) {
    __shared__ double r[K][256];
    for (int k = 0; k < K; k++) r[k][threadIdx.x] = s[k];
    __syncthreads();
    for (int d = 128; d; d >>= 1) { if ((int)threadIdx.x < d) for (int k = 0; k < K; k++) r[k][threadIdx.x] += r[k][threadIdx.x + d]; __syncthreads(); }
    if (!threadIdx.x) for (int k = 0; k < K; k++) if (r[k][0] != 0) atomicAdd(st + off + k, r[k][0]);
}
/* per input voxel c: r = mean of the output over its 4^3 fine block - input; rd[c] = Charbonnier'(r) (0 when the block or c is invalid) */
template <class L> __global__ void rp_ds_k(const L *__restrict__ lg, const uint8_t *__restrict__ lo, const uint8_t *__restrict__ co, rp_geo g, float *__restrict__ rd, double *st) {
    double s[2] = {0, 0};
    const size_t nq = (size_t)g.q * g.q * g.q;
    for (size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x; c < nq; c += (size_t)blockDim.x * gridDim.x) {
        const int cx = (int)(c % g.q), cy = (int)(c / g.q % g.q), cz = (int)(c / ((size_t)g.q * g.q));
        float sum = 0.f; int cnt = 0;
        for (int dz = 0; dz < 4; dz++) for (int dy = 0; dy < 4; dy++) for (int dx = 0; dx < 4; dx++) {
            const size_t i = ((size_t)(4 * cz + dz) * g.H + (4 * cy + dy)) * g.W + 4 * cx + dx;
            if (lo[i]) { sum += rp_o(lg, lo, i); cnt++; }
        }
        float v = 0.f;
        if (cnt == 64 && co[c]) { const float r = sum * 0.015625f - (float)co[c] * 0.015625f; v = rp_cd(r, g.eps); s[0] += rp_ch(r, g.eps); s[1] += 1; }
        rd[c] = v;
    }
    rp_reduce<2>(s, st, 7);
}
/* the shift-tolerant error: the error against the target at the voxel or (shift) at a face neighbour, whichever is smallest in magnitude */
template <class L> __device__ __forceinline__ float rp_emin(const L *lg, const uint8_t *lo, const uint8_t *hi, const rp_geo &g, size_t i, float o) {
    float em = o - (float)hi[i] * 0.015625f;
    if (!g.shift) return em;
    const size_t str[3] = {(size_t)g.H * g.W, (size_t)g.W, 1}; const int len[3] = {g.D, g.H, g.W};
    for (int a = 0; a < 3; a++) {
        const int c = (int)((i / str[a]) % len[a]);
        if (c > 0 && hi[i - str[a]]) { const float e = o - (float)hi[i - str[a]] * 0.015625f; if (fabsf(e) < fabsf(em)) em = e; }
        if (c < len[a] - 1 && hi[i + str[a]]) { const float e = o - (float)hi[i + str[a]] * 0.015625f; if (fabsf(e) < fabsf(em)) em = e; }
    }
    return em;
}
template <class L> __global__ void rp_stats_k(const L *__restrict__ lg, const uint8_t *__restrict__ lo, const uint8_t *__restrict__ hi, rp_geo g, double *st, size_t n) {
    double s[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};   /* st 0..6 as the synthetic loss, then the shift-min |error| DN, the scale head's NLL */
    double m[2] = {0, 0};                       /* st 9, 10: gradient error^2 of output / input against the target (DN) */
    const size_t str[3] = {(size_t)g.H * g.W, (size_t)g.W, 1}; const int len[3] = {g.D, g.H, g.W};
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)blockDim.x * gridDim.x) {
        if (!hi[i]) continue;
        const float o = rp_o(lg, lo, i), e = o - (float)hi[i] * 0.015625f, em = rp_emin(lg, lo, hi, g, i, o), dn = 64.f * e, base = (float)lo[i] - (float)hi[i];
        s[0] += 1; s[1] += (1.f - g.sw) * rp_ch(e, g.eps) + g.sw * rp_ch(em, g.eps); s[2] += fabsf(dn); s[3] += (double)dn * dn; s[4] += (double)base * base; s[7] += fabsf(64.f * em);
        if (g.uw != 0.f) { const float ls = (float)lg[n + i]; s[8] += (sqrtf(e * e + g.eps * g.eps) - g.eps) * __expf(-ls) + ls; }
        for (int a = 0; a < 3; a++) {
            if ((int)((i / str[a]) % len[a]) == len[a] - 1 || !hi[i + str[a]]) continue;
            const size_t j = i + str[a];
            const float oj = rp_o(lg, lo, j), df = e - (oj - (float)hi[j] * 0.015625f);
            s[5] += 1; s[6] += rp_ch(df, g.eps);
            const float dt = (float)hi[j] - (float)hi[i], dout = 64.f * (oj - o), dl = (float)lo[j] - (float)lo[i];
            m[0] += (double)(dout - dt) * (dout - dt); m[1] += (double)(dl - dt) * (dl - dt);
        }
    }
    double s7[7] = {s[0], s[1], s[2], s[3], s[4], s[5], s[6]};
    rp_reduce<7>(s7, st, 0);
    __syncthreads();
    rp_reduce<2>(m, st, 9);
    __syncthreads();
    double s14[2] = {s[7], s[8]};
    rp_reduce<2>(s14, st, 14);
}
template <class L, class G> __global__ void rp_grad_k(const L *__restrict__ lg, const uint8_t *__restrict__ lo, const uint8_t *__restrict__ hi, rp_geo g, const float *__restrict__ rd,
                                                    G *__restrict__ gl, const double *__restrict__ st, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = 0.f;
    const size_t str[3] = {(size_t)g.H * g.W, (size_t)g.W, 1}; const int len[3] = {g.D, g.H, g.W};
    if (hi[i] && st[0] > 0) {
        const float o = rp_o(lg, lo, i), e = o - (float)hi[i] * 0.015625f, em = rp_emin(lg, lo, hi, g, i, o), inv = (float)(1.0 / st[0]);
        v = ((1.f - g.sw) * rp_cd(e, g.eps) + g.sw * rp_cd(em, g.eps) + g.mw * e) * inv;
        if (g.ew != 0.f && st[5] > 0) {
            const float ie = g.ew * (float)(1.0 / st[5]);
            for (int a = 0; a < 3; a++) {
                const int c = (int)((i / str[a]) % len[a]);
                if (c > 0 && hi[i - str[a]]) { const size_t j = i - str[a]; const float df = e - (rp_o(lg, lo, j) - (float)hi[j] * 0.015625f); v += ie * rp_cd(df, g.eps); }
                if (c < len[a] - 1 && hi[i + str[a]]) { const size_t j = i + str[a]; const float df = e - (rp_o(lg, lo, j) - (float)hi[j] * 0.015625f); v += ie * rp_cd(df, g.eps); }
            }
        }
    }
    if (g.dw != 0.f && st[8] > 0 && lo[i]) {
        const int x = (int)(i % g.W), y = (int)(i / g.W % g.H), z = (int)(i / str[0]);
        v += g.dw * rd[((size_t)(z >> 2) * g.q + (y >> 2)) * g.q + (x >> 2)] * 0.015625f * (float)(1.0 / st[8]);
    }
    gl[i] = f2h<G>(v * g.gs);
    if (g.uw != 0.f) {   /* the scale head (channel 1, log b in 64-DN units): Laplace NLL |e| / b + log b with the error held fixed */
        float u = 0.f;
        if (hi[i] && st[0] > 0) { const float e = rp_o(lg, lo, i) - (float)hi[i] * 0.015625f; u = g.uw * (1.f - (sqrtf(e * e + g.eps * g.eps) - g.eps) * __expf(-(float)lg[n + i])) * (float)(1.0 / st[0]); }
        gl[n + i] = f2h<G>(u * g.gs);
    }
}
/* SSIM over non-overlapping 8^3 blocks whose voxels are all valid (target and input > 0), output and input against the target (DN) */
template <class L> __global__ void __launch_bounds__(512) rp_ssim_k(const L *__restrict__ lg, const uint8_t *__restrict__ lo, const uint8_t *__restrict__ hi, rp_geo g, double *st) {
    const int bx = g.W / 8, by = g.H / 8;
    const int b = blockIdx.x, X = b % bx, Y = b / bx % by, Z = b / (bx * by);
    const int t = threadIdx.x, x = 8 * X + (t & 7), y = 8 * Y + (t >> 3 & 7), z = 8 * Z + (t >> 6);
    const size_t i = ((size_t)z * g.H + y) * g.W + x;
    const int ok = hi[i] && lo[i];
    const float o = 64.f * rp_o(lg, lo, i), tv = hi[i], l = lo[i];
    __shared__ float r[9][512];
    const float v[9] = {o, tv, l, o * o, tv * tv, l * l, o * tv, l * tv, (float)ok};
    for (int k = 0; k < 9; k++) r[k][t] = v[k];
    __syncthreads();
    for (int d = 256; d; d >>= 1) { if (t < d) for (int k = 0; k < 9; k++) r[k][t] += r[k][t + d]; __syncthreads(); }
    if (t || r[8][0] < 512.f) return;
    const float N = 512.f, mo = r[0][0] / N, mt = r[1][0] / N, ml = r[2][0] / N;
    const float vo = r[3][0] / N - mo * mo, vt = r[4][0] / N - mt * mt, vl = r[5][0] / N - ml * ml, co = r[6][0] / N - mo * mt, cl = r[7][0] / N - ml * mt;
    const float C1 = 6.5025f, C2 = 58.5225f;
    const float so = (2 * mo * mt + C1) * (2 * co + C2) / ((mo * mo + mt * mt + C1) * (vo + vt + C2)), sl = (2 * ml * mt + C1) * (2 * cl + C2) / ((ml * ml + mt * mt + C1) * (vl + vt + C2));
    atomicAdd(st + 11, (double)so); atomicAdd(st + 12, (double)sl); atomicAdd(st + 13, 1.0);
}
template <class L> static void rp_loss_t(const L *lg, const uint8_t *lo, const uint8_t *hi, const uint8_t *co, rp_geo g, void *gl, int g16, int h16, double *st, float *rd, int metrics, size_t n) {
    CK(cudaMemsetAsync(st, 0, RP_NST * sizeof(double), 0));
    const size_t nq = (size_t)g.q * g.q * g.q;
    if (g.dw != 0.f || metrics) { int nb = (int)((nq + 255) / 256); if (nb > 1024) nb = 1024; rp_ds_k<L><<<nb, 256>>>(lg, lo, co, g, rd, st); }
    int nb = (int)((n + 255) / 256); if (nb > 2048) nb = 2048;
    rp_stats_k<L><<<nb, 256>>>(lg, lo, hi, g, st, n);
    if (metrics && g.D % 8 == 0 && g.H % 8 == 0 && g.W % 8 == 0) rp_ssim_k<L><<<(g.D / 8) * (g.H / 8) * (g.W / 8), 512>>>(lg, lo, hi, g, st);
    if (!gl) return;
    if (!g16) { g.gs = 1.f; rp_grad_k<L, float><<<nblk(n, 256), 256>>>(lg, lo, hi, g, rd, (float *)gl, st, n); }
    else if (h16) rp_grad_k<L, f16><<<nblk(n, 256), 256>>>(lg, lo, hi, g, rd, (f16 *)gl, st, n);
    else rp_grad_k<L, bf16><<<nblk(n, 256), 256>>>(lg, lo, hi, g, rd, (bf16 *)gl, st, n);
}
extern "C" void nn_restore_pair_loss(const void *lg, int lg16, const uint8_t *low, const uint8_t *tgt, const uint8_t *coarse, int D, int H, int W, const restore_pair_w *w,
                                     void *gl, int g16, int h16, float gscale, double *st, float *rd, int metrics) {
    const size_t n = (size_t)D * H * W;
    rp_geo g = {D, H, W, D / 4, w->shift, w->eps, w->mse, w->edge, w->shift_w, w->ds, gscale, w->unc};
    if (lg16) rp_loss_t<f16>((const f16 *)lg, low, tgt, coarse, g, gl, g16, h16, st, rd, metrics, n);
    else rp_loss_t<float>((const float *)lg, low, tgt, coarse, g, gl, g16, h16, st, rd, metrics, n);
    KCHECK();
}
