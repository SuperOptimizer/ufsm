/* ftaxis.c - the fibre axis of a presence volume (ufsm fiber-trace --presence-only / --mask; the GPU copy is src/nn_ftrace.cu).
   Pure function (no I/O), so tests link it alone. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include "ftgpu.h"

/* The fibre axis from the presence alone (--presence-only, --mask): the gradient structure tensor of the presence (central differences,
   products summed over a 5^3 box) has its SMALLEST eigenvalue along a tube; that eigenvector, on the z >= 0 hemisphere, is written
   in the published encoding (nx, ny). Streamed in z: five planes of the six in-plane-filtered products are kept. */
void ft_ridge_axis(const uint8_t *pr, const int64_t n[3], uint8_t *nx, uint8_t *ny, int R, char prior) {
    const int64_t H = n[1], W = n[2]; const size_t hw = (size_t)H * W; const int RW = 2 * R + 1; float *ring = malloc((size_t)RW * 6 * hw * sizeof(float)), *tmp = malloc(hw * sizeof(float)), *raw = malloc(6 * hw * sizeof(float));
    #define PRV(z, y, x) ((float)pr[((size_t)(z < 0 ? 0 : z >= n[0] ? n[0] - 1 : z) * H + (size_t)(y < 0 ? 0 : y >= H ? H - 1 : y)) * W + (size_t)(x < 0 ? 0 : x >= W ? W - 1 : x)])
    for (int64_t zz = 0; zz < n[0] + R; zz++) {
        if (zz < n[0]) {   /* plane zz: products, then the separable (2R+1)^2 in-plane box sum, into its ring slot */
            float *Rg = ring + (size_t)(zz % RW) * 6 * hw;
            for (int64_t y = 0; y < H; y++) for (int64_t x = 0; x < W; x++) { const size_t i = (size_t)y * W + x;
                const float gz = PRV(zz + 1, y, x) - PRV(zz - 1, y, x), gy = PRV(zz, y + 1, x) - PRV(zz, y - 1, x), gx = PRV(zz, y, x + 1) - PRV(zz, y, x - 1);
                raw[i] = gz * gz; raw[hw + i] = gy * gy; raw[2 * hw + i] = gx * gx; raw[3 * hw + i] = gz * gy; raw[4 * hw + i] = gz * gx; raw[5 * hw + i] = gy * gx; }
            for (int c = 0; c < 6; c++) { const float *a = raw + c * hw; float *o = Rg + c * hw;
                for (int64_t y = 0; y < H; y++) for (int64_t x = 0; x < W; x++) { float s = 0; for (int64_t k = x - R; k <= x + R; k++) s += a[(size_t)y * W + (size_t)(k < 0 ? 0 : k >= W ? W - 1 : k)]; tmp[(size_t)y * W + x] = s; }
                for (int64_t y = 0; y < H; y++) for (int64_t x = 0; x < W; x++) { float s = 0; for (int64_t k = y - R; k <= y + R; k++) s += tmp[(size_t)(k < 0 ? 0 : k >= H ? H - 1 : k) * W + x]; o[(size_t)y * W + x] = s; } }
        }
        const int64_t z = zz - R; if (z < 0) continue;   /* planes z - R .. z + R are in the ring (clamped at the ends) */
        for (size_t i = 0; i < hw; i++) {
            double T[6] = {0, 0, 0, 0, 0, 0};
            for (int64_t k = z - R; k <= z + R; k++) { const int64_t kk = k < 0 ? 0 : k >= n[0] ? n[0] - 1 : k; const float *Rk = ring + (size_t)(kk % RW) * 6 * hw; for (int c = 0; c < 6; c++) T[c] += Rk[c * hw + i]; }
            const size_t j = (size_t)z * hw + i; nx[j] = ny[j] = 128; const double tr = T[0] + T[1] + T[2]; if (!(tr > 0)) continue;
            /* smallest eigenvector of T: the largest of (tr I - T), by power iteration (T is positive semidefinite) */
            const double M[6] = {tr - T[0], tr - T[1], tr - T[2], -T[3], -T[4], -T[5]}; double v[3] = {0.577, 0.577, 0.577};
            for (int it = 0; it < 40; it++) { const double w[3] = {M[0] * v[0] + M[3] * v[1] + M[4] * v[2], M[3] * v[0] + M[1] * v[1] + M[5] * v[2], M[4] * v[0] + M[5] * v[1] + M[2] * v[2]};
                const double m = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]); if (!(m > 0)) break; v[0] = w[0] / m; v[1] = w[1] / m; v[2] = w[2] / m; }
            if (prior) {   /* the layer's normal (largest eigenvector of T); the axis = the family's direction in that plane:
                              H: the horizontal in-plane direction normal x z; V: z minus its normal component (our predicted masks mark a
                              fibre LAYER, so the tube axis of the mask itself is ambiguous inside it; DESIGN.md 2026-11-11) */
                double u[3] = {0.577, 0.577, 0.577};
                for (int it = 0; it < 40; it++) { const double w[3] = {T[0] * u[0] + T[3] * u[1] + T[4] * u[2], T[3] * u[0] + T[1] * u[1] + T[5] * u[2], T[4] * u[0] + T[5] * u[1] + T[2] * u[2]};
                    const double m = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]); if (!(m > 0)) break; u[0] = w[0] / m; u[1] = w[1] / m; u[2] = w[2] / m; }
                double a[3];
                if (prior == 'H') { a[0] = 0; a[1] = u[2]; a[2] = -u[1]; }   /* u x (1, 0, 0) in (z, y, x): (0, u_x, -u_y) */
                else { a[0] = 1 - u[0] * u[0]; a[1] = -u[0] * u[1]; a[2] = -u[0] * u[2]; }
                const double m = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); if (m > 1e-6) for (int c = 0; c < 3; c++) v[c] = a[c] / m;
            }
            if (v[0] < 0) { v[0] = -v[0]; v[1] = -v[1]; v[2] = -v[2]; }
            nx[j] = (uint8_t)lround(128 + 127 * v[2]); ny[j] = (uint8_t)lround(128 + 127 * v[1]);
        }
    }
    #undef PRV
    free(ring); free(tmp); free(raw);
}
