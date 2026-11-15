/* nn_slab_targets (the head slab's phase-derived targets in one kernel) byte for byte against the kernels it replaces: nn_phase_targets,
   nn_rv_targets + nn_packed_mask_z + nn_rv_verso_w, nn_packed_mask_z on the phase mask, nn_unwrap_targets2_z; a 20 x 24 x 28 window
   of tilted sheets (phase codes 0..251, unknown patches 255, papyrus classes 0 / 254 / 255 and warped grey values), three slabs, with and
   without the rv targets, the packed weights and the verso weight. Usage: test_slab_tg */
#include "nn.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
enum { D = 20, H = 24, W = 28 };
static uint64_t rs = 9;
static unsigned rnd(void) { rs = rs * 6364136223846793005ull + 1442695040888963407ull; return (unsigned)(rs >> 33); }
int main(void) {
    nn_init(0); int fails = 0; const size_t V = (size_t)D * H * W, hw = (size_t)H * W;
    uint8_t *ph = malloc(V), *pap = malloc(V);
    for (int z = 0; z < D; z++) for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
        const size_t i = ((size_t)z * H + y) * W + x;
        const double t = 0.11 * z + 0.07 * y + 0.03 * x + 0.02 * sin(0.5 * x * y);   /* turns: sheets ~6-9 voxels apart */
        ph[i] = (uint8_t)((long)floor(t * 252) % 252);
        if (rnd() % 23 == 0 || (z > 12 && y < 5)) ph[i] = 255;
        const unsigned r = rnd() % 8; pap[i] = r == 0 ? 255 : r < 3 ? 0 : r == 3 ? (uint8_t)(rnd() % 254) : 254;
    }
    void *dph = nn_malloc(V), *dpap = nn_malloc(V); nn_h2d(dph, ph, V); nn_h2d(dpap, pap, V);
    const int sl[3][2] = {{0, 4}, {4, 12}, {12, 20}};
    const float cf[4][3] = {{8.f, 1.f, 1.f}, {8.f, 1.7f, 1.f}, {1.f, 1.f, 0.f}, {4.f, 0.6f, 1.f}};   /* packed_w, verso w, rv */
    for (int c = 0; c < 4; c++) for (int k = 0; k < 3; k++) {
        const int z0 = sl[k][0], z1 = sl[k][1], rv = cf[c][2] > 0; const size_t n = (size_t)(z1 - z0) * hw, nb = 14 * n;
        uint8_t *a = nn_malloc(nb), *b = nn_malloc(nb), *ha = malloc(nb), *hb = malloc(nb);   /* t2 [2n] m2 [n] rvt [2n] rvm [n] t7 [7n] m7 [n] */
        nn_zero(a, nb); nn_zero(b, nb);
        uint8_t *t2 = a, *m2 = a + 2 * n, *rt = a + 3 * n, *rm = a + 5 * n, *t7 = a + 6 * n, *m7 = a + 13 * n;
        const float pw = cf[c][0], vw = cf[c][1], um = 9.6f, pum = 120.f, ds = 4.f;
        nn_phase_targets((uint8_t *)dph + (size_t)z0 * hw, n, t2, m2);
        if (rv) { nn_rv_targets((uint8_t *)dph + (size_t)z0 * hw, (uint8_t *)dpap + (size_t)z0 * hw, n, rt, rm); if (pw > 1) nn_packed_mask_z(dph, D, H, W, 1, um, pum, pw, rm, z0, z1); if (vw != 1.f) nn_rv_verso_w(rt, rm, n, vw); }
        if (pw > 1) nn_packed_mask_z(dph, D, H, W, 1, um, pum, pw, m2, z0, z1);
        nn_unwrap_targets2_z(dph, D, H, W, 1, ds, 1, um, pum, pw, t7, m7, z0, z1);
        nn_slab_targets(dph, rv ? dpap : nullptr, D, H, W, 1, ds, 1, um, pum, pw, vw, b, b + 2 * n, rv ? b + 3 * n : nullptr, b + 5 * n, b + 6 * n, b + 13 * n, z0, z1);
        nn_sync(); nn_d2h(ha, a, nb); nn_d2h(hb, b, nb);
        size_t nd = 0, nm = 0, nw = 0; for (size_t i = 0; i < nb; i++) nd += ha[i] != hb[i];
        for (size_t i = 0; i < n; i++) { nm += hb[13 * n + i] != 0; nw += hb[2 * n + i] > 1; }
        printf("slab targets (packed %.0f, verso w %.1f, rv %d, slab %d..%d): %zu of %zu bytes differ (normal mask set %zu, packed phase weights %zu)\n", pw, vw, rv, z0, z1, nd, nb, nm, nw);
        if (nd || !nm || (pw > 1 && !nw)) { fprintf(stderr, "FAIL\n"); fails++; }
        nn_free(a); nn_free(b); free(ha); free(hb);
    }
    if (fails) return 1;
    printf("test_slab_tg: ok\n"); return 0;
}
