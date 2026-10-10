/* band_field_side on concentric cylinder shells (src/band.h): every known voxel is on the recto side (outward of its nearest
   labelled recto) or the verso side; going outward the side turns verso -> recto exactly at each recto shell; voxels beyond the
   flood radius, and around a missing wrap, are unknown; the band is unchanged by asking for the sides. band_field_phase: the
   fraction of the way between consecutive shells, known only between labelled consecutive shells. */
#include "band.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL: " __VA_ARGS__); fputc('\n', stderr); fails++; } } while (0)

enum { NZ = 6, NY = 160, NX = 160, NSH = 5 };
static const double YC = 80, XC = 80, R0 = 20, PITCH = 12;

static void shells(uint8_t *codes, const int *turn) {   /* shell k at label radius R0 + k PITCH, winding turn[k] */
    memset(codes, 0, (size_t)NZ * NY * NX);
    for (int z = 0; z < NZ; z++) for (int y = 0; y < NY; y++) for (int x = 0; x < NX; x++) {
        const double r = hypot(y - YC, x - XC);
        for (int k = 0; k < NSH; k++)
            if (fabs(r - (R0 + k * PITCH)) < 0.5) codes[((size_t)z * NY + y) * NX + x] = (uint8_t)(1 + (turn[k] * BAND_STEPS) % BAND_PERIOD);
    }
}

int main(void) {
    const size_t N = (size_t)NZ * NY * NX;
    const int n[3] = {NZ, NY, NX}; const int64_t o[3] = {0, 0, 0};
    double cy[NZ], cx[NZ];
    for (int z = 0; z < NZ; z++) { cy[z] = 2 * YC + 0.5; cx[z] = 2 * XC + 0.5; }   /* native coordinates of the label-grid centre */
    band_params bp = {80, 75};
    uint8_t *codes = malloc(N), *band = malloc(N), *band2 = malloc(N), *side = malloc(N);
    int turn[NSH] = {0, 1, 2, 3, 4};
    shells(codes, turn);
    CHECK(!band_field_side(codes, n, o, cy, cx, bp, band, side), "band_field_side failed");
    CHECK(!band_field(codes, n, o, cy, cx, bp, band2) && !memcmp(band, band2, N), "the band changes when sides are requested");
    size_t known = 0, wrong = 0, inner = 0;
    for (int z = 0; z < NZ; z++) for (int y = 0; y < NY; y++) for (int x = 0; x < NX; x++) {
        const size_t i = ((size_t)z * NY + y) * NX + x;
        const double r = hypot(y - YC, x - XC);
        if (r < R0 + 2 || r > R0 + (NSH - 1) * PITCH - 2) continue;
        inner++;
        CHECK((side[i] == SIDE_UNKNOWN) == (band[i] == BAND_UNKNOWN), "side and band disagree on unknown at %d %d %d", z, y, x);
        if (side[i] == SIDE_UNKNOWN) continue;
        known++;
        int j = 0; for (int k = 1; k < NSH; k++) if (fabs(r - (R0 + k * PITCH)) < fabs(r - (R0 + j * PITCH))) j = k;
        const double mid = R0 + (j + (r >= R0 + j * PITCH ? 0.5 : -0.5)) * PITCH;
        const int expect = r >= R0 + j * PITCH - 0.5 ? SIDE_RECTO : SIDE_VERSO;
        if (side[i] != expect && fabs(r - mid) > 1.0 && fabs(r - (R0 + j * PITCH)) > 0.75) wrong++;
    }
    CHECK(known > inner * 0.95, "between the shells %zu of %zu voxels known", known, inner);
    CHECK(wrong == 0, "%zu voxels on the wrong side", wrong);
    /* outward along +x through the centre row: verso -> recto exactly at each shell */
    for (int k = 1; k < NSH; k++) {
        const int xr = (int)(XC + R0 + k * PITCH), z = 2;
        const uint8_t a = side[((size_t)z * NY + (int)YC) * NX + xr - 1], b = side[((size_t)z * NY + (int)YC) * NX + xr];
        CHECK(a == SIDE_VERSO && b == SIDE_RECTO, "shell %d: side %d -> %d going outward (want verso -> recto)", k, a, b);
    }
    /* missing wrap between shells 2 and 3 (their windings two turns apart): the gap between them is unknown */
    int turn2[NSH] = {0, 1, 2, 4, 5};
    shells(codes, turn2);
    CHECK(!band_field_side(codes, n, o, cy, cx, bp, band, side), "band_field_side failed (missing wrap)");
    const int xm = (int)(XC + R0 + 2.5 * PITCH);
    CHECK(side[((size_t)2 * NY + (int)YC) * NX + xm] == SIDE_UNKNOWN, "a missing wrap must leave the gap unknown");
    /* phase: between consecutive shells the fraction of the way outward, (r - R_j) / PITCH, wrapping to 0 on each shell;
       unknown inside the innermost and outside the outermost shell and across the missing wrap; the band is unchanged */
    uint8_t *phase = malloc(N), *spacing = malloc(N);
    shells(codes, turn);
    CHECK(!band_field_side(codes, n, o, cy, cx, bp, band2, side) && !band_field_phase(codes, n, o, cy, cx, bp, band, phase, spacing), "band_field_phase failed");
    CHECK(!memcmp(band, band2, N), "the band changes when the phase is requested");
    size_t pk = 0, pin = 0, outside = 0, outside_known = 0, side_bad = 0, sp_bad = 0; double perr = 0, pmax = 0;
    for (int z = 0; z < NZ; z++) for (int y = 0; y < NY; y++) for (int x = 0; x < NX; x++) {
        const size_t i = ((size_t)z * NY + y) * NX + x;
        const double r = hypot(y - YC, x - XC);
        if (r < R0 - 2 || r > R0 + (NSH - 1) * PITCH + 2) { outside++; outside_known += phase[i] != PHASE_UNKNOWN; continue; }
        if (r < R0 + 1 || r > R0 + (NSH - 1) * PITCH - 1) continue;
        pin++;
        if (phase[i] == PHASE_UNKNOWN) continue;
        pk++;
        const double f = (r - R0) / PITCH - floor((r - R0) / PITCH), g = phase[i] / (double)PHASE_PERIOD;
        double e = fabs(f - g); e = fmin(e, 1 - e);
        perr += e; pmax = fmax(pmax, e);
        if (abs((int)spacing[i] - (int)PITCH) > 1) sp_bad++;
        if (side[i] != SIDE_UNKNOWN && (phase[i] < PHASE_PERIOD / 2) != (side[i] == SIDE_RECTO) && fabs(f - 0.5) > 1.0 / PITCH && fmin(f, 1 - f) > 1.0 / PITCH) side_bad++;
    }
    CHECK(pk > pin * 0.95, "phase known for %zu of %zu voxels between the shells", pk, pin);
    CHECK(perr / pk < 0.02 && pmax < 0.09, "phase error mean %.4f max %.4f (turns)", perr / pk, pmax);
    CHECK(outside_known < outside / 200, "phase known for %zu of %zu voxels inside the innermost / outside the outermost shell", outside_known, outside);
    CHECK(side_bad == 0, "%zu voxels where the phase half disagrees with the side", side_bad);
    CHECK(sp_bad < pk / 100, "%zu of %zu known voxels with a spacing more than a voxel off the shell pitch", sp_bad, pk);
    for (int k = 1; k < NSH; k++) {   /* outward along +x: the phase rises and wraps to 0 on each shell */
        const int xr = (int)(XC + R0 + k * PITCH), z = 2;
        const uint8_t a = phase[((size_t)z * NY + (int)YC) * NX + xr - 1], b = phase[((size_t)z * NY + (int)YC) * NX + xr];
        CHECK(a > PHASE_PERIOD * 0.85 && a < PHASE_PERIOD && b == 0, "shell %d: phase %d -> %d going outward (want ~1 -> 0)", k, a, b);
    }
    shells(codes, turn2);
    CHECK(!band_field_phase(codes, n, o, cy, cx, bp, band, phase, nullptr), "band_field_phase failed (missing wrap)");
    CHECK(phase[((size_t)2 * NY + (int)YC) * NX + xm] == PHASE_UNKNOWN, "a missing wrap must leave the phase unknown");
    for (size_t i = 0; i < N; i++) if (band[i] == BAND_UNKNOWN && phase[i] != PHASE_UNKNOWN) { CHECK(0, "phase known where the band is not"); break; }
    printf("phase: %zu of %zu known between the shells, error mean %.4f max %.4f turns\n", pk, pin, perr / pk, pmax);
    free(codes); free(band); free(band2); free(side); free(phase); free(spacing);
    if (fails) return 1;
    printf("test_side: ok (%zu known voxels between the shells)\n", known);
    return 0;
}
