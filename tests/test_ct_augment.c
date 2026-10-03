/* CPU-only augmentation invariants and statistical checks. */
#include "ct_augment.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
#define CHECK(ok, name) do { if (!(ok)) { fprintf(stderr, "FAIL: %s (line %d)\n", name, __LINE__); failures++; } } while (0)

/* Independent Gaussian generator, not the module's plan RNG. */
static uint64_t random_state = UINT64_C(0x14bac3932adf710e);
static double random_uniform(void) {
    random_state ^= random_state >> 12;
    random_state ^= random_state << 25;
    random_state ^= random_state >> 27;
    return (double)(((random_state * UINT64_C(0x2545f4914f6cdd1d)) >> 11) + 1) / 9007199254740993.;
}
static float gaussian(void) {
    return (float)(sqrt(-2 * log(random_uniform())) * cos(6.283185307179586 * random_uniform()));
}

static void test_plans(void) {
    ct_aug_plan p, q;
    ct_aug_make(&p, 1, 0, 103.25, 32.5, 704);
    ct_aug_make(&q, 999, 0, 103.25, 32.5, 704);
    CHECK(memcmp(&p, &q, sizeof p) == 0, "disabled plan ignores seed");
    for (int i = 0; i < 256; i++) CHECK(p.lut[i] == (float)((i - 103.25) / 32.5), "disabled normalized identity LUT");
    CHECK(!p.gamma_applied && !p.filter_mode && !p.noise_rho && !p.noise_sigma,
          "validation has no extra appearance/noise operations");
    CHECK(!p.shading[0] && !p.shading[1] && !p.shading[2], "validation has no shading");
    int ngamma = 0, nshade = 0, nblur = 0, nsharp = 0, nnoise = 0;
    for (uint64_t seed = 0; seed < 12000; seed++) {
        ct_aug_make(&p, seed, 1, 103.25, 32.5, 704);
        ct_aug_make(&q, seed, 1, 103.25, 32.5, 704);
        CHECK(memcmp(&p, &q, sizeof p) == 0, "seed determinism");
        CHECK(p.lut[0] == (float)(-103.25 / 32.5) && p.lut[255] == (float)((255 - 103.25) / 32.5), "gamma preserves endpoints");
        for (int i = 1; i < 256; i++) CHECK(isfinite(p.lut[i]) && p.lut[i] > p.lut[i - 1], "finite strictly monotone gamma LUT");
        if (p.gamma_applied) {
            double raw = p.lut[128] * 32.5 + 103.25;
            double g = log(raw / 255) / log(128. / 255);
            CHECK(g >= .9 - 1e-6 && g <= 1.1 + 1e-6, "gamma exponent range");
            ngamma++;
        }
        double shade = fabs(p.shading[0]) + fabs(p.shading[1]) + fabs(p.shading[2]);
        CHECK(shade <= .08000001, "shading bound over entire cube");
        nshade += shade > 0;
        CHECK(p.gamma_applied + (shade > 0) + (p.filter_mode != 0) <= 2, "at most two extra appearance domains");
        CHECK(p.filter_mode >= 0 && p.filter_mode <= 2, "one filter mode");
        if (p.filter_mode == 1) { nblur++; CHECK(p.filter_strength >= .3f && p.filter_strength <= .8f, "blur blend range"); }
        if (p.filter_mode == 2) { nsharp++; CHECK(p.filter_strength >= .1f && p.filter_strength <= .2f, "unsharp range"); }
        if (p.noise_rho) {
            nnoise++;
            CHECK(p.noise_rho >= .3f && p.noise_rho <= .45f, "noise correlation range");
            CHECK(p.noise_sigma >= .02f && p.noise_sigma <= .06f, "noise RMS range");
        } else CHECK(p.noise_sigma == 0, "white noise sigma belongs to caller");
    }
    CHECK(ngamma > 1400 && ngamma < 2100 && nshade > 1400 && nshade < 2100,
          "gamma and shading selection probabilities");
    CHECK(nblur > 900 && nblur < 1500 && nsharp > 900 && nsharp < 1500 && nnoise > 1400 && nnoise < 2100,
          "filter and correlated noise selection probabilities");
    printf("plans: gamma %d, shading %d, blur %d, sharpen %d, correlated noise %d / 12000\n",
           ngamma, nshade, nblur, nsharp, nnoise);
}

static void test_values(void) {
    const int P = 5;
    uint8_t ct[125], before[125];
    ct_aug_plan p;
    ct_aug_make(&p, 1, 0, 0, 255, P);
    memset(ct, 93, sizeof ct);
    memcpy(before, ct, sizeof ct);
    for (int mode = 0; mode < 3; mode++) {
        p.filter_mode = mode; p.filter_strength = .5f;
        for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++)
            CHECK(ct_aug_value(&p, ct, P, z, y, x) == p.lut[93], "filter preserves constants including clamped edges");
    }
    CHECK(memcmp(ct, before, sizeof ct) == 0, "CT source remains unchanged");
    memset(ct, 0, sizeof ct); ct[62] = 255;
    p.filter_mode = 1; p.filter_strength = .6f;
    CHECK(fabs(ct_aug_value(&p, ct, P, 2, 2, 2) - .7) < 1e-6, "blur impulse center");
    CHECK(fabs(ct_aug_value(&p, ct, P, 2, 2, 3) - .05) < 1e-6, "blur impulse axial neighbor");
    CHECK(ct_aug_value(&p, ct, P, 2, 3, 3) == 0, "blur has no diagonal support");
    double sum = 0;
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) sum += ct_aug_value(&p, ct, P, z, y, x);
    CHECK(fabs(sum - 1) < 1e-6, "interior impulse mass preserved");
    p.filter_mode = 2; p.filter_strength = .2f;
    CHECK(fabs(ct_aug_value(&p, ct, P, 2, 2, 2) - 1.1) < 1e-6, "unsharp impulse center");
    CHECK(fabs(ct_aug_value(&p, ct, P, 2, 2, 3) + 1. / 60) < 1e-6, "unsharp impulse neighbor");
    p.filter_mode = 0;
    p.shading[0] = .02f; p.shading[1] = -.03f; p.shading[2] = .01f;
    memset(ct, 0, sizeof ct);
    sum = 0;
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
        float v = ct_aug_value(&p, ct, P, z, y, x);
        CHECK(fabs(v) <= .06000001, "local shading bounded"); sum += v;
    }
    CHECK(fabs(sum) < 1e-6, "linear shading has zero spatial mean");
    CHECK(fabs(ct_aug_value(&p, ct, P, 4, 0, 4) - .06) < 1e-6, "shading uses source z/y/x coordinates");
    CHECK(ct_aug_value(&p, ct, 1, 0, 0, 0) == 0, "singleton shading is centered");
    ct[0] = 255; p.filter_mode = 1;
    CHECK(ct_aug_value(&p, ct, 1, 0, 0, 0) == 1, "singleton neighbors clamp to center");
}

static void test_noise(void) {
    enum { P = 96 };
    size_t P2 = (size_t)P * P, P3 = P2 * P;
    float *cube = malloc(P3 * sizeof(float)), *prow = malloc(P * sizeof(float));
    float *plane = malloc(P2 * sizeof(float));
    CHECK(cube && prow && plane, "noise test allocations");
    if (!cube || !prow || !plane) { free(cube); free(prow); free(plane); return; }
    ct_aug_plan p;
    ct_aug_make(&p, 0, 0, 0, 1, P);
    float row0[P], row1[P];
    for (int x = 0; x < P; x++) row0[x] = row1[x] = (float)(x - 11);
    ct_aug_noise_row(&p, row0, NULL, NULL, P, 0, 0);
    CHECK(memcmp(row0, row1, sizeof row0) == 0, "rho=0 exact no-op with NULL states");
    p.noise_rho = .4f; p.noise_sigma = .04f;
    for (int x = 0; x < P; x++) prow[x] = NAN;
    for (size_t i = 0; i < P2; i++) plane[i] = NAN;
    uint64_t saved_seed = random_state;
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) {
        float *row = cube + (size_t)z * P2 + (size_t)y * P;
        for (int x = 0; x < P; x++) row[x] = gaussian();
        ct_aug_noise_row(&p, row, prow, plane, P, z, y);
    }
    double mean = 0, sq = 0, cov[3] = {0}, edge_sq[3] = {0};
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
        size_t i = (size_t)z * P2 + (size_t)y * P + x;
        double v = cube[i];
        CHECK(isfinite(v), "uninitialized states ignored at boundaries");
        mean += v; sq += v * v;
        if (x) cov[0] += v * cube[i - 1];
        if (y) cov[1] += v * cube[i - P];
        if (z) cov[2] += v * cube[i - P2];
        if (!x) edge_sq[0] += v * v;
        if (!y) edge_sq[1] += v * v;
        if (!z) edge_sq[2] += v * v;
    }
    mean /= P3; double var = sq / P3 - mean * mean;
    CHECK(fabs(mean) < .025 && fabs(var - 1) < .035, "correlated noise zero mean/unit variance");
    CHECK(fabs(p.noise_sigma * sqrt(sq / P3) - p.noise_sigma) < .001,
          "scaled noise RMS matches requested normalized CT sigma");
    for (int d = 0; d < 3; d++) {
        cov[d] = (cov[d] / ((double)P2 * (P - 1)) - mean * mean) / var;
        CHECK(fabs(cov[d] - .4) < .025, "AR covariance along every axis");
        CHECK(fabs(edge_sq[d] / P2 - 1) < .08, "stationary noise variance on first planes");
    }
    /* A second cube must be bit-identical with the same innovations despite
       arbitrary previous state. This detects missing y/z boundary resets. */
    random_state = saved_seed;
    for (int x = 0; x < P; x++) prow[x] = 300;
    for (size_t i = 0; i < P2; i++) plane[i] = -400;
    for (int z = 0; z < P; z++) for (int y = 0; y < P; y++) {
        float row[P];
        for (int x = 0; x < P; x++) row[x] = gaussian();
        ct_aug_noise_row(&p, row, prow, plane, P, z, y);
        CHECK(memcmp(row, cube + (size_t)z * P2 + (size_t)y * P, sizeof row) == 0,
              "noise deterministic and state resets between cubes");
    }
    printf("correlated noise: mean %.4f, variance %.4f, lag-one X/Y/Z %.4f %.4f %.4f\n",
           mean, var, cov[0], cov[1], cov[2]);
    free(cube); free(prow); free(plane);
}

int main(void) {
    test_plans(); test_values(); test_noise();
    printf("CT augmentation %s\n", failures ? "FAIL" : "ok");
    return failures != 0;
}
