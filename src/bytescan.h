/* word-at-a-time (SWAR) byte scans for the sampler's whole-window passes (P^3 bytes each): mcc leaves the byte loops scalar (one
   movzbl a voxel; try9's draw_ spent ~12 % of the sampler CPU in four of them, perf 2026-11-17). Exact per-byte results: the flag
   arithmetic below never carries across bytes. test_bytescan checks each against the byte loop. */
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#define BS_L7 0x7f7f7f7f7f7f7f7full
#define BS_H 0x8080808080808080ull
#define BS_1 0x0101010101010101ull
static inline uint64_t bs_ld(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; }
/* 0x80 in each byte of y that is not zero ((y & 7f) + 7f sets bit 7 iff the low 7 bits are not all 0; | y adds bit 7 itself) */
static inline uint64_t bs_nz(uint64_t y) { return (((y & BS_L7) + BS_L7) | y) & BS_H; }
static inline uint64_t bs_z(uint64_t y) { return ~bs_nz(y) & BS_H; }
/* bit j = bit 7 of byte j (f: flags in bit 7 of each byte only) */
static inline unsigned bs_pack(uint64_t f) { return (unsigned)(((f >> 7) * 0x0102040810204080ull) >> 56); }

/* the number of bytes != v */
static inline size_t bs_count_ne(const uint8_t *p, size_t n, uint8_t v) {
    size_t c = 0, k = 0; const uint64_t vv = BS_1 * v;
    for (; k + 8 <= n; k += 8) c += (size_t)__builtin_popcountll(bs_nz(bs_ld(p + k) ^ vv));
    for (; k < n; k++) c += p[k] != v;
    return c;
}
/* the number of bytes b with (b & m) != m (for m = 252 = 0xfc: b < 252) */
static inline size_t bs_count_notall(const uint8_t *p, size_t n, uint8_t m) {
    size_t c = 0, k = 0; const uint64_t mm = BS_1 * m;
    for (; k + 8 <= n; k += 8) c += (size_t)__builtin_popcountll(bs_nz((bs_ld(p + k) & mm) ^ mm));
    for (; k < n; k++) c += (p[k] & m) != m;
    return c;
}
/* the loss mask's complement as 0x80 flags: byte k unknown iff ct[k] == 0 or (msk && msk[k] != 0) */
static inline uint64_t bs_unknown(const uint8_t *ct, const uint8_t *msk, size_t k) {
    uint64_t f = bs_z(bs_ld(ct + k)); if (msk) f |= bs_nz(bs_ld(msk + k)); return f;
}
/* a[k] = 255 where the voxel is unknown (ct 0 or msk set; msk null: ct alone) */
static inline void bs_fill_unknown(uint8_t *a, const uint8_t *ct, const uint8_t *msk, size_t n) {
    size_t k = 0;
    for (; k + 8 <= n; k += 8) {
        const uint64_t f = bs_unknown(ct, msk, k); if (!f) continue;
        const uint64_t w = bs_ld(a + k) | (f >> 7) * 0xff; memcpy(a + k, &w, 8);
    }
    for (; k < n; k++) if (!ct[k] || (msk && msk[k])) a[k] = 255;
}
