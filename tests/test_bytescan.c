/* src/bytescan.h's word-wise scans against the byte loops they replace in draw_ (sample.c), on random bytes biased to the edge
   values (0, 1, 127, 128, 251-255) and every length 0 .. 70 plus a 1 M tail case; byte-exact counts, planes and packed bits. */
#include "bytescan.h"
#include <stdio.h>
#include <stdlib.h>
static uint64_t rs = 0x2545f4914f6cdd1dull;
static uint8_t rb(void) {
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
    static const uint8_t e[] = {0, 1, 127, 128, 251, 252, 253, 254, 255};
    return (rs >> 32) % 3 ? e[(rs >> 40) % 9] : (uint8_t)(rs >> 48);
}
int main(void) {
    int fails = 0;
    for (int t = 0; t < 2000; t++) {
        const size_t n = t < 1990 ? (size_t)(t % 71) : (size_t)1 << 20 | (size_t)(t % 8);
        uint8_t *ct = malloc(n + 1), *mk = malloc(n + 1), *a = malloc(n + 1), *b = malloc(n + 1), *m1 = malloc(n + 1), *m2 = malloc(n + 1);
        for (size_t k = 0; k < n; k++) { ct[k] = rb(); mk[k] = rb() & (t & 1 ? 1 : 255); a[k] = b[k] = rb(); }
        size_t c1 = 0, c2 = 0; for (size_t k = 0; k < n; k++) { c1 += ct[k] != 255; c2 += ct[k] < 252; }
        if (bs_count_ne(ct, n, 255) != c1 || bs_count_notall(ct, n, 0xfc) != c2) { fprintf(stderr, "FAIL count n %zu\n", n); fails++; }
        for (int z = 0; z < 2; z++) {
            const uint8_t *msk = z ? nullptr : mk;
            for (size_t k = 0; k < n; k++) b[k] = a[k];
            uint8_t *r = malloc(n + 1); for (size_t k = 0; k < n; k++) r[k] = a[k];
            for (size_t k = 0; k < n; k++) if (!(ct[k] != 0 && (!msk || msk[k] == 0))) r[k] = 255;
            bs_fill_unknown(b, ct, msk, n);
            for (size_t k = 0; k < n; k++) if (r[k] != b[k]) { fprintf(stderr, "FAIL fill n %zu k %zu\n", n, k); fails++; break; }
            free(r);
            /* the ink-bit pass (n / 8 words): bits and the rewritten 0 / 1 mask */
            for (size_t k = 0; k < n; k++) m1[k] = m2[k] = mk[k];
            for (size_t k8 = 0; k8 < n / 8; k8++) {
                unsigned v = 0, w;
                for (int j = 0; j < 8; j++) { const size_t k = 8 * k8 + (size_t)j; const unsigned q = z ? ct[k] != 0 : (ct[k] != 0) & (m1[k] == 0); m1[k] = (uint8_t)q; v |= q << j; }
                const uint64_t f = ~bs_unknown(ct, z ? nullptr : m2, 8 * k8) & BS_H; const uint64_t ww = f >> 7; memcpy(m2 + 8 * k8, &ww, 8); w = bs_pack(f);
                if (v != w) { fprintf(stderr, "FAIL bits n %zu word %zu: %02x vs %02x\n", n, k8, v, w); fails++; break; }
            }
            if (memcmp(m1, m2, n)) { fprintf(stderr, "FAIL mask rewrite n %zu\n", n); fails++; }
        }
        free(ct); free(mk); free(a); free(b); free(m1); free(m2);
    }
    if (fails) { printf("test_bytescan: %d FAILED\n", fails); return 1; }
    printf("test_bytescan: ok (2000 cases)\n"); return 0;
}
