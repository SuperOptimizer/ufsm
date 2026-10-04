/* Binary surface erosion on the native voxel grid, before target softening.
   Input is a (P+2)^3 haloed cube in sampler encoding (0 background, 254 surface).
   Reading the halo from the label store prevents false erosion at crop edges. */
#pragma once
#include <stddef.h>
#include <stdint.h>

static inline void target_erode1(const uint8_t *haloed, uint8_t *out, int P) {
    const size_t Q = (size_t)P + 2, Q2 = Q * Q;
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) {
        const uint8_t *row = haloed + (size_t)(z + 1) * Q2 + (size_t)(y + 1) * Q + 1;
        uint8_t *dst = out + ((size_t)z * P + y) * P;
        for (int x = 0; x < P; x++) {
            const uint8_t *v = row + x;
            dst[x] = v[0] && v[0] < 255 && v[-1] && v[1] && v[-(ptrdiff_t)Q] && v[Q] &&
                v[-(ptrdiff_t)Q2] && v[Q2] ? v[0] : 0;
        }
    }
}
