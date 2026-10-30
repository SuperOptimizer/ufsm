/* vwt inference kernels (PF; src/vwt_infer.c): slab extraction on the device from a CT box resident there, and the per-slab outputs of
   pass A (VW's stitch design): the phase on a 2x downsampled grid relative to the slab centre (fp16) + valid bits, and the phase at
   overlap points.

   Slab geometry = villawind frame_of: voxel (i, j, k) at o + i a + j b + k d (xyz); here o is relative to the box origin (box zyx
   origin bo subtracted on the host, so floats keep ~1e-4 voxel precision). Sampling as villawind samp_run: outside the volume -> 0,
   invalid; else trilinear in the box (cell index clamped to the box), rounded to uint8, valid. */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

typedef struct { float o[3], a[3], b[3], d[3]; } vwi_frame_f;   /* xyz, o relative to the box origin */

__global__ void k_vwi_sample(const uint8_t *__restrict__ box, int bz, int by, int bx, vwi_frame_f f, float vlx, float vly, float vlz, float vhx, float vhy, float vhz,
                             int T, int L, uint8_t *__restrict__ im, uint8_t *__restrict__ va) {
    const long long n = (long long)T * T * L;
    for (long long q = blockIdx.x * (long long)blockDim.x + threadIdx.x; q < n; q += (long long)gridDim.x * blockDim.x) {
        const int k = (int)(q % L); const long long ij = q / L; const int j = (int)(ij % T), i = (int)(ij / T);
        const float x = f.o[0] + i * f.a[0] + j * f.b[0] + k * f.d[0], y = f.o[1] + i * f.a[1] + j * f.b[1] + k * f.d[1], z = f.o[2] + i * f.a[2] + j * f.b[2] + k * f.d[2];
        if (!(x >= vlx && y >= vly && z >= vlz && x <= vhx && y <= vhy && z <= vhz)) { im[q] = 0; va[q] = 0; continue; }
        int iz = (int)floorf(z), iy = (int)floorf(y), ix = (int)floorf(x);
        iz = min(max(iz, 0), bz - 2); iy = min(max(iy, 0), by - 2); ix = min(max(ix, 0), bx - 2);
        const float fz = z - iz, fy = y - iy, fx = x - ix;
        const uint8_t *p = box + ((size_t)iz * by + iy) * bx + ix; const size_t sy = bx, sz = (size_t)by * bx;
        const float v = (1 - fz) * ((1 - fy) * ((1 - fx) * p[0] + fx * p[1]) + fy * ((1 - fx) * p[sy] + fx * p[sy + 1]))
                      + fz * ((1 - fy) * ((1 - fx) * p[sz] + fx * p[sz + 1]) + fy * ((1 - fx) * p[sz + sy] + fx * p[sz + sy + 1]));
        const int u = __float2int_rn(v); im[q] = (uint8_t)min(max(u, 0), 255); va[q] = 1;
    }
}
extern "C" void vwi_sample(const uint8_t *box, int bz, int by, int bx, const float *o, const float *a, const float *b, const float *d, const float *vlo, const float *vhi,
                           int T, int L, uint8_t *im, uint8_t *va, void *s) {
    vwi_frame_f f; for (int c = 0; c < 3; c++) { f.o[c] = o[c]; f.a[c] = a[c]; f.b[c] = b[c]; f.d[c] = d[c]; }
    const long long n = (long long)T * T * L; long long nb = (n + 255) / 256; if (nb > 8192) nb = 8192;
    k_vwi_sample<<<(unsigned)nb, 256, 0, (cudaStream_t)s>>>(box, bz, by, bx, f, vlo[0], vlo[1], vlo[2], vhi[0], vhi[1], vhi[2], T, L, im, va);
}

/* the count of nonzero bytes (valid voxels, or nonzero CT): the skip test */
__global__ void k_vwi_count(const uint8_t *__restrict__ va, long long n, unsigned long long *cnt) {
    unsigned long long c = 0;
    for (long long q = blockIdx.x * (long long)blockDim.x + threadIdx.x; q < n; q += (long long)gridDim.x * blockDim.x) c += va[q] != 0;
    for (int o = 16; o; o >>= 1) c += __shfl_xor_sync(0xffffffffu, c, o);
    if (!(threadIdx.x & 31) && c) atomicAdd(cnt, c);
}
extern "C" void vwi_count(const uint8_t *va, long long n, unsigned long long *cnt, void *s) {
    cudaMemsetAsync(cnt, 0, 8, (cudaStream_t)s); k_vwi_count<<<256, 256, 0, (cudaStream_t)s>>>(va, n, cnt);
}
/* pass A grid (src/vwt_tile.h): out[u][v][w] = mean of the 4 x 4 x 2 block at (4u, 4v, 2w) minus the centre value (fp16; NaN where any voxel
   of the block is invalid); centre = the mean of the 8 voxels around (T/2 - 0.5, T/2 - 0.5, L/2 - 0.5) */
__global__ void k_vwi_down(const float *__restrict__ ph, const uint8_t *__restrict__ va, int T, int L, __half *__restrict__ out, float *centre) {
    const int TU = T / 4, LW = L / 2; const long long n = (long long)TU * TU * LW;
    const int ci = T / 2 - 1, ck = L / 2 - 1;
    float c = 0.f;
#pragma unroll
    for (int e = 0; e < 8; e++) c += ph[((size_t)(ci + (e >> 2)) * T + ci + ((e >> 1) & 1)) * L + ck + (e & 1)];
    c *= 0.125f;
    if (blockIdx.x == 0 && threadIdx.x == 0) *centre = c;
    for (long long q = blockIdx.x * (long long)blockDim.x + threadIdx.x; q < n; q += (long long)gridDim.x * blockDim.x) {
        const int w = (int)(q % LW); const long long uv = q / LW; const int v = (int)(uv % TU), u = (int)(uv / TU);
        float s = 0.f; int ok = 1;
        for (int di = 0; di < 4; di++)
#pragma unroll
            for (int dj = 0; dj < 4; dj++) {
                const size_t p = ((size_t)(4 * u + di) * T + 4 * v + dj) * L + 2 * w;
                const float2 a = *(const float2 *)(ph + p); s += a.x + a.y; ok &= va[p] & va[p + 1];
            }
        out[q] = ok ? __float2half(s * (1.f / 32.f) - c) : __ushort_as_half((unsigned short)0x7e00);
    }
}
extern "C" void vwi_down(const float *ph, const uint8_t *va, int T, int L, void *out, float *centre, void *s) {
    const long long n = (long long)(T / 4) * (T / 4) * (L / 2); long long nb = (n + 255) / 256; if (nb > 4096) nb = 4096;
    k_vwi_down<<<(unsigned)nb, 256, 0, (cudaStream_t)s>>>(ph, va, T, L, (__half *)out, centre);
}
/* the phase trilinear at np slab-local points (i, j, k) (floats); NAN when outside the slab or any corner invalid */
__global__ void k_vwi_points(const float *__restrict__ ph, const uint8_t *__restrict__ va, int T, int L, const float *__restrict__ pts, int np, float *__restrict__ out) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x; if (t >= np) return;
    const float pi = pts[3 * t], pj = pts[3 * t + 1], pk = pts[3 * t + 2];
    if (!(pi >= 0 && pj >= 0 && pk >= 0 && pi <= T - 1 && pj <= T - 1 && pk <= L - 1)) { out[t] = NAN; return; }
    const int i0 = min((int)pi, T - 2), j0 = min((int)pj, T - 2), k0 = min((int)pk, L - 2); const float fi = pi - i0, fj = pj - j0, fk = pk - k0;
    float s = 0.f; int ok = 1;
#pragma unroll
    for (int e = 0; e < 8; e++) {
        const int di = e >> 2, dj = (e >> 1) & 1, dk = e & 1; const size_t p = ((size_t)(i0 + di) * T + j0 + dj) * L + k0 + dk;
        s += (di ? fi : 1 - fi) * (dj ? fj : 1 - fj) * (dk ? fk : 1 - fk) * ph[p]; ok &= va[p];
    }
    out[t] = ok ? s : NAN;
}
extern "C" void vwi_points(const float *ph, const uint8_t *va, int T, int L, const float *pts, int np, float *out, void *s) {
    if (np > 0) k_vwi_points<<<(np + 255) / 256, 256, 0, (cudaStream_t)s>>>(ph, va, T, L, pts, np, out);
}

/* async device -> pinned host copy on the default stream, completion event (the writer thread waits on it) */
extern "C" void *vwi_event_new(void) { cudaEvent_t e = nullptr; cudaEventCreateWithFlags(&e, cudaEventDisableTiming); return (void *)e; }
extern "C" void vwi_d2h_async(void *dst, const void *src, size_t n) { cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, 0); }
extern "C" void vwi_event_record(void *e) { cudaEventRecord((cudaEvent_t)e, 0); }
extern "C" int vwi_event_wait(void *e) { return cudaEventSynchronize((cudaEvent_t)e) == cudaSuccess ? 0 : -1; }
extern "C" void vwi_h2d_async(void *dst, const void *src, size_t n) { cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, 0); }
