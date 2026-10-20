/* predict --tta (src/tta.h, src/nn_tta.cu): every window symmetry is an involution (map, CT kernel, logit kernel: a round trip is
   exact, fp32 and fp16), and on a symmetric toy network (equivariant by construction: normals from the CT gradient, phase and density
   from the radius about the axis rows, a scalar from the CT) every augmented pass, brought back, equals the plain pass: TTA then
   gives identical output. Slab windows (D != W). */
#include "nn.h"
#include "tta.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } } while (0)

enum { D = 10, W = 12, NCH = 11, NRM = 4 };
#define N ((size_t)D * W * W)
static size_t at(int z, int y, int x) { return ((size_t)z * W + y) * W + x; }

/* the toy net (task unwrap layout): 0 a scalar of the CT, 2 / 3 cos / sin of the radius, 4..9 n n^T (n the unit CT gradient, clamped
   central differences), 10 the radius; 1 a constant. Its own frame: the radial input at (y, x) of plane z is (y + dyo[z], x + dxo[z]). */
static void toy(const uint8_t *ct, const float *dyo, const float *dxo, float *L) {
    for (int z = 0; z < D; z++) for (int y = 0; y < W; y++) for (int x = 0; x < W; x++) {
        const size_t v = at(z, y, x);
        const double g[3] = {(double)ct[at(z + 1 < D ? z + 1 : D - 1, y, x)] - ct[at(z > 0 ? z - 1 : 0, y, x)],
                             (double)ct[at(z, y + 1 < W ? y + 1 : W - 1, x)] - ct[at(z, y > 0 ? y - 1 : 0, x)],
                             (double)ct[at(z, y, x + 1 < W ? x + 1 : W - 1)] - ct[at(z, y, x > 0 ? x - 1 : 0)]};
        const double gl = sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) + 1e-12, n[3] = {g[0] / gl, g[1] / gl, g[2] / gl};
        const double r = hypot(y + dyo[z], x + dxo[z]);
        L[0 * N + v] = (float)(ct[v] / 255.0); L[1 * N + v] = 0.25f;
        L[2 * N + v] = (float)cos(r / 3); L[3 * N + v] = (float)sin(r / 3);
        L[4 * N + v] = (float)(n[0] * n[0]); L[5 * N + v] = (float)(n[1] * n[1]); L[6 * N + v] = (float)(n[2] * n[2]);
        L[7 * N + v] = (float)(n[0] * n[1]); L[8 * N + v] = (float)(n[0] * n[2]); L[9 * N + v] = (float)(n[1] * n[2]);
        L[10 * N + v] = (float)r;
    }
}

int main(void) {
    if (nn_init(0)) { fprintf(stderr, "test_tta: no GPU\n"); return 1; }
    srand(3);
    uint8_t *ct = malloc(N), *ct2 = malloc(N), *ct3 = malloc(N);
    for (size_t v = 0; v < N; v++) ct[v] = (uint8_t)(rand() % 256);
    float dyo[D], dxo[D], dyt[D], dxt[D], dy2[D], dx2[D];
    for (int z = 0; z < D; z++) { dyo[z] = -5.3f - 0.2f * z; dxo[z] = 2.7f + 0.15f * z; }   /* an axis off-centre, tilted */
    uint8_t *dct = nn_malloc(N), *dct2 = nn_malloc(N);
    float *L0 = malloc(NCH * N * 4), *Lt = malloc(NCH * N * 4), *Lb = malloc(NCH * N * 4), *Lsum = calloc(NCH * N, 4);
    void *dlg = nn_malloc(NCH * N * 4);
    toy(ct, dyo, dxo, L0);
    nn_h2d(dct, ct, N);
    for (int p = 0; p < TTA_MAX; p++) {
        const int code = tta_order[p];
        /* the voxel map and the axis rows are involutions */
        int bad = 0;
        for (int z = 0; z < D; z++) for (int y = 0; y < W; y++) for (int x = 0; x < W; x++) {
            int a, b, c, a2, b2, c2; tta_map(code, D, W, z, y, x, &a, &b, &c); tta_map(code, D, W, a, b, c, &a2, &b2, &c2);
            if (a2 != z || b2 != y || c2 != x) bad++;
        }
        CHECK(!bad, "code %d: the map is not an involution (%d voxels)", code, bad);
        tta_axis(code, D, W, dyo, dxo, dyt, dxt); tta_axis(code, D, W, dyt, dxt, dy2, dx2);
        for (int z = 0; z < D; z++) CHECK(fabsf(dy2[z] - dyo[z]) < 1e-4f && fabsf(dx2[z] - dxo[z]) < 1e-4f, "code %d: axis rows round trip z %d", code, z);
        /* the CT through T on the device: as the host map; twice = identity */
        nn_tta_ct(dct, dct2, D, W, code); nn_d2h(ct2, dct2, N);
        bad = 0;
        for (int z = 0; z < D; z++) for (int y = 0; y < W; y++) for (int x = 0; x < W; x++) { int a, b, c; tta_map(code, D, W, z, y, x, &a, &b, &c); if (ct2[at(z, y, x)] != ct[at(a, b, c)]) bad++; }
        CHECK(!bad, "code %d: CT kernel differs from the map (%d voxels)", code, bad);
        { uint8_t *d3 = nn_malloc(N); nn_tta_ct(dct2, d3, D, W, code); nn_d2h(ct3, d3, N); nn_free(d3); }
        CHECK(!memcmp(ct3, ct, N), "code %d: CT round trip", code);
        /* the logits round trip: random fp32 and fp16 bits, through T twice */
        int src[32], sgn[32]; tta_channels(code, NRM, NCH, src, sgn);
        for (int c = 0; c < NCH; c++) CHECK(src[src[c]] == c && sgn[src[c]] == sgn[c], "code %d: channel map not an involution at %d", code, c);
        for (size_t i = 0; i < NCH * N; i++) Lt[i] = (float)rand() / RAND_MAX * 8 - 4;
        nn_h2d(dlg, Lt, NCH * N * 4); nn_tta_logits(dlg, 0, D, W, code, NCH, src, sgn); nn_tta_logits(dlg, 0, D, W, code, NCH, src, sgn); nn_d2h(Lb, dlg, NCH * N * 4);
        CHECK(!memcmp(Lb, Lt, NCH * N * 4), "code %d: fp32 logit round trip", code);
        { uint16_t *h = malloc(NCH * N * 2), *h2 = malloc(NCH * N * 2); for (size_t i = 0; i < NCH * N; i++) h[i] = (uint16_t)rand();
          nn_h2d(dlg, h, NCH * N * 2); nn_tta_logits(dlg, 1, D, W, code, NCH, src, sgn); nn_tta_logits(dlg, 1, D, W, code, NCH, src, sgn); nn_d2h(h2, dlg, NCH * N * 2);
          CHECK(!memcmp(h, h2, NCH * N * 2), "code %d: fp16 logit round trip", code); free(h); free(h2); }
        /* the symmetric toy: the augmented pass (the toy in the transformed frame), brought back, is the plain pass */
        toy(ct2, dyt, dxt, Lt);
        nn_h2d(dlg, Lt, NCH * N * 4); nn_tta_logits(dlg, 0, D, W, code, NCH, src, sgn); nn_d2h(Lb, dlg, NCH * N * 4);
        double md = 0; int worst = -1;
        for (size_t i = 0; i < NCH * N; i++) { const double d = fabs((double)Lb[i] - L0[i]); if (d > md) { md = d; worst = (int)(i / N); } }
        CHECK(md < 1e-4, "code %d: the toy's pass brought back differs from the plain pass by %g (channel %d)", code, md, worst);
        for (size_t i = 0; i < NCH * N; i++) Lsum[i] += Lb[i];
    }
    double md = 0; for (size_t i = 0; i < NCH * N; i++) md = fmax(md, fabs(Lsum[i] / TTA_MAX - L0[i]));
    CHECK(md < 1e-4, "the 12-pass mean differs from the plain pass by %g", md);
    nn_free(dct); nn_free(dct2); nn_free(dlg); free(ct); free(ct2); free(ct3); free(L0); free(Lt); free(Lb); free(Lsum);
    if (fails) { fprintf(stderr, "test_tta: %d FAILED\n", fails); return 1; }
    printf("test_tta: ok (%d symmetries)\n", TTA_MAX);
    return 0;
}
