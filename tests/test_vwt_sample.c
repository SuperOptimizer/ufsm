/* test_vwt_sample: the vwt sampler's frame (villawind.c frame_of convention) and flood-fill unwrap on synthetic phases (CPU only) */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void vws_frame(const double dir[3], const double ray_origin[3], int T, double o[3], double a[3], double b[3], double d[3]);
int vws_unwrap(const float *raw, int T, int L, int minsz, float *ph, uint16_t *comp);

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL: " __VA_ARGS__); printf("\n"); fails++; } } while (0)
static double dot(const double *a, const double *b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

int main(void) {
    /* frame: orthonormal, right-handed (a x b = d), a orthogonal to the axis of d's smallest |component|, o = r - (T-1)/2 (a + b) */
    const double dirs[4][3] = {{1, 0.2, 0.1}, {-0.3, 1, 0.05}, {0.1, -0.2, -1}, {0.6, 0.6, 0.5}};
    for (int t = 0; t < 4; t++) {
        double o[3], a[3], b[3], d[3]; const double r[3] = {10, 20, 30}; vws_frame(dirs[t], r, 128, o, a, b, d);
        CHECK(fabs(dot(a, a) - 1) < 1e-12 && fabs(dot(b, b) - 1) < 1e-12 && fabs(dot(d, d) - 1) < 1e-12, "frame %d not unit", t);
        CHECK(fabs(dot(a, b)) < 1e-12 && fabs(dot(a, d)) < 1e-12 && fabs(dot(b, d)) < 1e-12, "frame %d not orthogonal", t);
        const double c[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
        CHECK(fabs(dot(c, d) - 1) < 1e-12, "frame %d: a x b != d", t);
        int m = 0; for (int k = 1; k < 3; k++) if (fabs(d[k]) < fabs(d[m])) m = k;
        CHECK(fabs(a[m]) < 1e-12, "frame %d: a has a component on the ref axis %d", t, m);
        for (int k = 0; k < 3; k++) CHECK(fabs(o[k] - (r[k] - 63.5 * (a[k] + b[k]))) < 1e-9, "frame %d: origin", t);
    }
    /* unwrap: concentric cylinders, phase = frac(rho / 10) around the axis x = y = -40 (z up); the ray along +x from the slab edge:
       one component, ph rising along +k by (rho_end - rho_start) / 10 on the central column */
    const int T = 16, L = 96; const size_t n = (size_t)T * T * L;
    float *raw = malloc(n * 4), *ph = malloc(n * 4); uint16_t *comp = malloc(n * 2);
    double o[3], a[3], b[3], d[3]; const double dir[3] = {1, 0, 0}, r0[3] = {0, 0, 50}; vws_frame(dir, r0, T, o, a, b, d);
    for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) for (int k = 0; k < L; k++) {
        double p[3]; for (int c = 0; c < 3; c++) p[c] = o[c] + i * a[c] + j * b[c] + k * d[c];
        const double rho = hypot(p[0] + 40, p[1] + 40); double f = rho / 10; f -= floor(f); raw[((size_t)i * T + j) * L + k] = (float)f;
    }
    int nc = vws_unwrap(raw, T, L, 64, ph, comp);
    CHECK(nc == 1, "cylinders: %d components (want 1)", nc);
    size_t up = 0, dn = 0; for (size_t v = 1; v < n; v++) if (v % L && comp[v] && comp[v] == comp[v - 1]) { if (ph[v] > ph[v - 1]) up++; else dn++; }
    CHECK(dn == 0, "cylinders: ph falls along +k at %zu of %zu steps", dn, up + dn);
    { const int i = T / 2, j = T / 2; double p0[3], p1[3]; for (int c = 0; c < 3; c++) { p0[c] = o[c] + i * a[c] + j * b[c]; p1[c] = p0[c] + (L - 1) * d[c]; }
      const double want = (hypot(p1[0] + 40, p1[1] + 40) - hypot(p0[0] + 40, p0[1] + 40)) / 10, got = ph[((size_t)i * T + j) * L + L - 1] - ph[((size_t)i * T + j) * L];
      CHECK(fabs(got - want) < 1e-4, "cylinders: central rise %.5f want %.5f", got, want); }
    /* a gap of unknown voxels across the slab at k 40..41 splits it into two components; a 3-voxel island is dropped (minsz 64) */
    for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) for (int k = 40; k < 42; k++) raw[((size_t)i * T + j) * L + k] = NAN;
    for (int i = 0; i < T; i++) for (int j = 0; j < T; j++) for (int k = 80; k < L; k++) raw[((size_t)i * T + j) * L + k] = NAN;
    raw[((size_t)2 * T + 2) * L + 90] = 0.1f; raw[((size_t)2 * T + 2) * L + 91] = 0.12f; raw[((size_t)2 * T + 3) * L + 90] = 0.11f;
    nc = vws_unwrap(raw, T, L, 64, ph, comp);
    CHECK(nc == 2, "gap: %d components (want 2)", nc);
    CHECK(comp[((size_t)2 * T + 2) * L + 90] == 0 && isnan(ph[((size_t)2 * T + 2) * L + 90]), "island not dropped");
    CHECK(comp[((size_t)5 * T + 5) * L + 10] != comp[((size_t)5 * T + 5) * L + 60] && comp[((size_t)5 * T + 5) * L + 10] && comp[((size_t)5 * T + 5) * L + 60], "gap halves not split");
    /* a jump of half a turn (> 0.25) between two halves of the ray axis does not join them */
    for (size_t v = 0; v < n; v++) { const int k = (int)(v % L); raw[v] = k < 48 ? 0.1f : 0.6f; }
    nc = vws_unwrap(raw, T, L, 64, ph, comp);
    CHECK(nc == 2, "half-turn step: %d components (want 2)", nc);
    free(raw); free(ph); free(comp);
    printf("test_vwt_sample: %s (%d failures)\n", fails ? "FAIL" : "ok", fails);
    return fails ? 1 : 0;
}
