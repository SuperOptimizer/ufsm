/* MX stride-2 backward-data for wide layers (output channels too many for the parity kernel's shared-memory weight tile):
   thread per (n, ci block, gx voxel), weights from shared memory when they fit (WS) or global memory. Separate unit: it is
   slow to compile. */
#include "lp_mxops.cuh"
template <int WS>
__global__ void __launch_bounds__(128) bwd_data_s2_mx_k(const uint8_t *gy, const float *w, uint8_t *gx, int N, int Ci, int Co,
                                                       int D, int H, int W, int Do, int Ho, int Wo, int accum) {
    extern __shared__ float sw[];
    const int bwx = mx_bw(Ci), nbx = mx_nb(Ci), bwy = mx_bw(Co), nby = mx_nb(Co);
    const size_t S = (size_t)D * H * W, So = (size_t)Do * Ho * Wo;
    if (WS) { for (int i = threadIdx.x; i < Co * Ci * 27; i += blockDim.x) sw[i] = w[i]; __syncthreads(); }
    const float *wt = WS ? sw : w;
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)N * nbx * S) return;
    size_t v = i % S; int blk = (int)((i / S) % nbx), n = (int)(i / (S * nbx));
    int ux = (int)(v % W), uy = (int)((v / W) % H), uz = (int)(v / ((size_t)W * H));
    int kz[2], oz[2], nz = 0, ky[2], oy[2], ny = 0, kx[2], ox[2], nx = 0;
    for (int k = 0; k < 3; k++) {
        int t;
        t = uz + 1 - k; if (!(t & 1) && t >= 0 && (t >> 1) < Do) { kz[nz] = k; oz[nz++] = t >> 1; }
        t = uy + 1 - k; if (!(t & 1) && t >= 0 && (t >> 1) < Ho) { ky[ny] = k; oy[ny++] = t >> 1; }
        t = ux + 1 - k; if (!(t & 1) && t >= 0 && (t >> 1) < Wo) { kx[nx] = k; ox[nx++] = t >> 1; }
    }
    float acc[32] = {};
    const uint8_t *scy = gy + (size_t)N * nby * So * bwy;
#pragma unroll 1
    for (int a = 0; a < nz; a++) for (int bb = 0; bb < ny; bb++) for (int c = 0; c < nx; c++) {
        const int tap = (kz[a] * 3 + ky[bb]) * 3 + kx[c];
        const size_t vo = ((size_t)oz[a] * Ho + oy[bb]) * Wo + ox[c];
        for (int yb = 0; yb < nby; yb++) {
            float r[32];
            mx_load_row(gy, scy, ((size_t)n * nby + yb) * So + vo, bwy, r);
            for (int k = 0; k < bwy; k++) {
                const int co = yb * bwy + k;
                if (co >= Co) break;
                const float g = r[k];
                const float *wr = wt + ((size_t)co * Ci + blk * bwx) * 27 + tap;
#pragma unroll
                for (int cc = 0; cc < 32; cc++) if (cc < bwx && blk * bwx + cc < Ci) acc[cc] += wr[cc * 27] * g;
            }
        }
    }
    const uint8_t *scx = gx + (size_t)N * nbx * S * bwx;
    if (accum) {
        float r[32];
        mx_load_row(gx, scx, i, bwx, r);
#pragma unroll
        for (int k = 0; k < 32; k++) acc[k] += r[k];
    }
    mx_store_row(gx, gx + (size_t)N * nbx * S * bwx, i, bwx, acc);
}
extern "C" void lp_bwd_data_s2_mx_wide(const void *gy, shape5 ys, const float *w, shape5 xs, void *gx, int accum) {
    size_t n = (size_t)xs.n * mx_nb(xs.c) * shape_spatial(xs), wb = (size_t)ys.c * xs.c * 27 * sizeof(float);
    if (wb <= 48 * 1024) bwd_data_s2_mx_k<1><<<nblk_(n, 128), 128, wb>>>((const uint8_t *)gy, w, (uint8_t *)gx, xs.n, xs.c, ys.c, xs.d, xs.h, xs.w, ys.d, ys.h, ys.w, accum);
    else bwd_data_s2_mx_k<0><<<nblk_(n, 128), 128>>>((const uint8_t *)gy, w, (uint8_t *)gx, xs.n, xs.c, ys.c, xs.d, xs.h, xs.w, ys.d, ys.h, ys.w, accum);
    LPCK();
}
