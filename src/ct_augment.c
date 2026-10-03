#include "ct_augment.h"
#include <math.h>
#include <string.h>

/* Local SplitMix64: plan construction does not consume the sampler's per-voxel
   random stream or depend on thread scheduling. */
static uint64_t next_u64(uint64_t *s) {
    uint64_t v = (*s += UINT64_C(0x9e3779b97f4a7c15));
    v = (v ^ (v >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    v = (v ^ (v >> 27)) * UINT64_C(0x94d049bb133111eb);
    return v ^ (v >> 31);
}
static double uniform(uint64_t *s) { return (double)(next_u64(s) >> 11) * 0x1.0p-53; }

void ct_aug_make(ct_aug_plan *p, uint64_t seed, int enabled, double mean, double sd, int P) {
    memset(p, 0, sizeof *p);
    if (!isfinite(mean)) mean = 0;
    if (!isfinite(sd) || sd <= 0) sd = 1;
    int gamma = 0, shade = 0, filter = 0;
    if (enabled) {
        gamma = uniform(&seed) < .15;
        shade = P > 1 && uniform(&seed) < .15;
        double f = uniform(&seed);
        filter = f < .10 ? 1 : f < .20 ? 2 : 0;
        /* At most two extra appearance changes. If all three were selected,
           discard one uniformly instead of always preferring a particular domain. */
        if (gamma && shade && filter) {
            int drop = (int)(uniform(&seed) * 3);
            if (drop == 0) gamma = 0;
            else if (drop == 1) shade = 0;
            else filter = 0;
        }
    }
    p->gamma_applied = gamma;
    double g = gamma ? .9 + .2 * uniform(&seed) : 1;
    for (int i = 0; i < 256; i++) {
        double raw = gamma ? 255 * pow((double)i / 255, g) : i;
        p->lut[i] = (float)((raw - mean) / sd);
    }
    if (shade) {
        double c[3], norm = 0;
        for (int d = 0; d < 3; d++) { c[d] = 2 * uniform(&seed) - 1; norm += fabs(c[d]); }
        double strength = .08 * uniform(&seed);
        for (int d = 0; d < 3; d++) p->shading[d] = (float)(c[d] * strength / (norm > 0 ? norm : 1));
    }
    p->filter_mode = filter;
    if (filter) p->filter_strength = (float)(filter == 1 ? .3 + .5 * uniform(&seed) : .1 + .1 * uniform(&seed));
    if (enabled && uniform(&seed) < .15) {
        p->noise_rho = (float)(.3 + .15 * uniform(&seed));
        p->noise_sigma = (float)(.02 + .04 * uniform(&seed));
    }
}

void ct_aug_noise_row(const ct_aug_plan *p, float *row, float *previous_row,
                      float *previous_plane, int P, int z, int y) {
    const float rho = p->noise_rho;
    if (rho == 0) return;
    const float scale = sqrtf(1.f - rho * rho);
    const size_t base = (size_t)y * P;
    float last_x = 0;
    for (int x = 0; x < P; x++) {
        float vx = x ? rho * last_x + scale * row[x] : row[x];
        last_x = vx;   /* Save X before the Y/Z stages modify the row. */
        float vy = y ? rho * previous_row[x] + scale * vx : vx;
        previous_row[x] = vy;
        float vz = z ? rho * previous_plane[base + x] + scale * vy : vy;
        previous_plane[base + x] = vz;
        row[x] = vz;
    }
}
