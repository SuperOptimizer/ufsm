/* Scroll augmentation renders through the sampler: aug_render <sources.json> <out.jpg> <z,y,x> <P> <level> <aug=p[,aug=p...]> [seed 1] [frames 1]
   One window (UFSM_SAMPLE_AT, this level's voxels) of source 0 drawn twice with the --task unwrap2 label path (side 2, winding mod 4),
   no symmetry / rotation / elastic / zoom / CT jitter: plain, then with the named augmentations at their p. The grid (mid-z plane):
   row 1 plain, row 2 augmented; columns CT (the z-scored input), the phase hue over the CT, the winding mod 4 (hue over 4 turns,
   component edges dark). aug: tear, tear_um, xscan, crush, delam, density, curv, incl. frames > 1: rows 2.. are further draws (seeds +1). */
#include "../src/sample.c"
#include "imgops.h"

void *nn_host_alloc(size_t n) { return malloc(n); }
void nn_host_free(void *p) { free(p); }
static void hue(double h, uint8_t rgb[3]) {
    const double r = .5 + .5 * cos(2 * M_PI * h), g = .5 + .5 * cos(2 * M_PI * (h - 1. / 3)), b = .5 + .5 * cos(2 * M_PI * (h - 2. / 3));
    rgb[0] = (uint8_t)(255 * r); rgb[1] = (uint8_t)(255 * g); rgb[2] = (uint8_t)(255 * b);
}
static uint8_t gray(uint16_t h) { _Float16 f; memcpy(&f, &h, 2); const double v = 128 + 40 * (double)f; return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v); }
static void row(uint8_t *img, int W, int py, int P, const uint16_t *x, const uint8_t *ph, const uint8_t *w4, const uint8_t *wc) {
    const size_t z = (size_t)P / 2;
    for (int y = 0; y < P; y++) for (int xx = 0; xx < P; xx++) {
        const size_t k = (z * P + y) * P + xx; const uint8_t c = gray(x[k]);
        uint8_t *o0 = img + ((size_t)(py + y) * W + xx) * 3, *o1 = o0 + (size_t)(P + 4) * 3, *o2 = o1 + (size_t)(P + 4) * 3;
        o0[0] = o0[1] = o0[2] = c;
        if (ph[k] < PHASE_PERIOD) { uint8_t h[3]; hue(ph[k] / (double)PHASE_PERIOD, h); for (int d = 0; d < 3; d++) o1[d] = (uint8_t)((c + h[d]) / 2); }
        else o1[0] = o1[1] = o1[2] = c / 2;
        if (w4[k] < PHASE_PERIOD) {
            uint8_t h[3]; hue(w4[k] / 252.0, h);
            const int edge = (xx + 1 < P && wc[k + 1] != wc[k]) || (y + 1 < P && wc[k + P] != wc[k]);
            for (int d = 0; d < 3; d++) o2[d] = edge ? 0 : h[d];
        } else o2[0] = o2[1] = o2[2] = c / 3;
    }
}
int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "usage: aug_render <sources.json> <out.jpg> <z,y,x> <P> <level> <aug=p[,aug=p...]> [seed 1] [frames 1]\n"); return 2; }
    setenv("UFSM_SAMPLE_AT", argv[3], 1);
    const int P = atoi(argv[4]), l = atoi(argv[5]), frames = argc > 8 ? atoi(argv[8]) : 1; const uint64_t seed = argc > 7 ? (uint64_t)atoll(argv[7]) : 1;
    sources *S = sources_load(argv[1]); if (!S) return 1;
    sample_cfg c = sample_cfg_default();
    c.P = P; c.B = 1; c.nworkers = 1; c.nbuf = 2; c.deterministic = 1; c.seed = seed; c.xfmt = 1; c.gpu_input = 1; c.side = 2; c.wind4 = 1;
    memset(c.level_p, 0, sizeof c.level_p); c.level_p[l] = 1; c.min_fg = 0; c.snap = 0;
    c.augment = 1; c.geometry_augment = 1; c.ct_augment = 0; c.symmetry_p = 0; c.rotate_p = 0; c.elastic_p = 0; c.zoom_p = 0; c.zoom_out_p = 0; c.affine_p = 0;
    sample_cfg a = c;
    for (char *q = argv[6]; q && *q;) {
        char key[32]; double v; int nc = 0;
        if (sscanf(q, "%31[^=]=%lf%n", key, &v, &nc) != 2) { fprintf(stderr, "aug_render: bad aug list %s\n", q); return 2; }
        if (!strcmp(key, "tear")) a.tear_p = (float)v; else if (!strcmp(key, "tear_um")) a.tear_um = (float)v;
        else if (!strcmp(key, "xscan")) a.xscan_p = (float)v; else if (!strcmp(key, "crush")) a.crush.p = (float)v;
        else if (!strcmp(key, "delam")) a.delam_p = (float)v; else if (!strcmp(key, "delam_hole")) a.delam_hole = (float)v;
        else { fprintf(stderr, "aug_render: unknown aug %s\n", key); return 2; }
        q += nc; if (*q == ',') q++;
    }
    const size_t p3 = (size_t)P * P * P; const int W = 3 * P + 8, H = (1 + frames) * (P + 4) - 4;
    uint8_t *img = malloc((size_t)W * H * 3); if (!img) return 1;
    memset(img, 40, (size_t)W * H * 3);
    for (int f = 0; f <= frames; f++) {
        sample_cfg cf = f ? a : c; cf.seed = seed + (uint64_t)(f ? f - 1 : 0);
        const double t0 = tnow();
        sampler *sp = sampler_start(S, &cf); if (!sp) return 1;
        batch *b = sampler_next(sp); if (!b) return 1;
        row(img, W, f * (P + 4), P, b->x16, b->side, b->w4, b->wcomp);
        printf("%s: row %d: corner %lld %lld %lld (level 0), %.2f s;", argv[2], f, (long long)b->corner[0][0], (long long)b->corner[0][1], (long long)b->corner[0][2], tnow() - t0);
        for (int k = 0; k < AUGN_N; k++) if (sampler_aug_count(sp, k)) printf(" aug %d x %llu;", k, (unsigned long long)sampler_aug_count(sp, k));
        printf("\n");
        sampler_release(sp, b); sampler_stop(sp);
    }
    (void)p3;
    return jpeg_write_rgb(argv[2], W, H, img, 90) ? 1 : 0;
}
