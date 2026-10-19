/* Task restore, host side: degradation recipes (src/restore.h). */
#include "restore.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *const restore_family_names[RF_N] = {"identity", "pyramid", "scanner"};

void restore_ranges_default(restore_ranges *g) {
    memset(g, 0, sizeof *g);
    g->mix[RF_IDENTITY] = 0.1f; g->mix[RF_PYRAMID] = 0.45f; g->mix[RF_SCANNER] = 0.45f;
    g->pyr_q = 2.f;
    g->sc_step[0] = 3.4f; g->sc_step[1] = 4.4f;
    g->sc_psf[0] = 0.35f; g->sc_psf[1] = 0.75f; g->sc_aniso = 0.15f;
    g->sc_noise[0] = 0.5f; g->sc_noise[1] = 4.f;
    g->sc_sigvar[0] = 0.f; g->sc_sigvar[1] = 0.03f;
    g->sc_corr[0] = 0.f; g->sc_corr[1] = 0.4f;
    g->sc_edge[0] = -0.1f; g->sc_edge[1] = 0.25f; g->sc_edge_p = 0.3f;
    g->sc_gain = 0.06f; g->sc_gamma = 0.06f; g->sc_bias = 3.f;
    g->sc_q = 8.f;
}

int restore_parse_mix(restore_ranges *g, const char *s) {
    float m[RF_N] = {0};
    char *t = strdup(s), *sv = nullptr;
    int bad = 0;
    for (char *p = strtok_r(t, ",", &sv); p; p = strtok_r(nullptr, ",", &sv)) {
        char *c = strchr(p, ':'); if (!c) { bad = 1; break; }
        *c = 0; int f = -1;
        for (int k = 0; k < RF_N; k++) if (!strcmp(p, restore_family_names[k])) f = k;
        const float v = (float)atof(c + 1);
        if (f < 0 || !(v >= 0)) { bad = 1; break; }
        m[f] = v;
    }
    free(t);
    float tot = 0; for (int k = 0; k < RF_N; k++) tot += m[k];
    if (bad || !(tot > 0)) return -1;
    for (int k = 0; k < RF_N; k++) g->mix[k] = m[k] / tot;
    return 0;
}

int restore_pick_family(const restore_ranges *g, double u) {
    double tot = 0; for (int k = 0; k < RF_N; k++) tot += g->mix[k];
    u *= tot;
    for (int k = 0; k < RF_N; k++) { if (u < g->mix[k]) return k; u -= g->mix[k]; }
    return RF_N - 1;
}

static uint64_t sm64(uint64_t *s) { uint64_t z = (*s += 0x9E3779B97F4A7C15ull); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }
static double u01(uint64_t *s) { return (double)(sm64(s) >> 11) * 0x1.0p-53; }
static float unif(uint64_t *s, const float r[2]) { return (float)(r[0] + (r[1] - r[0]) * u01(s)); }

void restore_recipe_draw(restore_recipe *r, int family, uint64_t seed, int D, int H, int W, const restore_ranges *g) {
    memset(r, 0, sizeof *r);
    uint64_t s = seed ^ 0x5eed5eedull;
    r->family = family; r->gain = 1.f; r->gamma = 1.f; r->seed = (unsigned)sm64(&s);
    const int dim[3] = {D, H, W};
    if (family == RF_IDENTITY) { for (int a = 0; a < 3; a++) { r->small[a] = dim[a]; r->step[a] = 1.f; } return; }
    for (int a = 0; a < 3; a++) r->qphase[a] = (int)(sm64(&s) & 15);
    if (family == RF_PYRAMID) {
        r->box = 1; r->q = g->pyr_q;
        for (int a = 0; a < 3; a++) { r->small[a] = dim[a] / 4; r->step[a] = 4.f; }
        return;
    }
    const float st = unif(&s, g->sc_step), psf = unif(&s, g->sc_psf);
    for (int a = 0; a < 3; a++) {
        int n = (int)lrintf((float)dim[a] / st); if (n < 1) n = 1;
        r->small[a] = n; r->step[a] = (float)dim[a] / (float)n;
        r->phase[a] = (float)(u01(&s) - 0.5);
        r->psf[a] = psf * r->step[a] * expf((float)((2 * u01(&s) - 1) * g->sc_aniso));
    }
    r->gain = expf((float)((2 * u01(&s) - 1) * g->sc_gain)); r->gamma = expf((float)((2 * u01(&s) - 1) * g->sc_gamma));
    r->bias = (float)((2 * u01(&s) - 1) * g->sc_bias);
    r->noise = unif(&s, g->sc_noise); r->sig_var = unif(&s, g->sc_sigvar); r->corr = unif(&s, g->sc_corr);
    r->edge = u01(&s) < g->sc_edge_p ? unif(&s, g->sc_edge) : 0.f;
    r->q = g->sc_q;
}
