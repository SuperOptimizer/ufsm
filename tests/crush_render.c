/* --aug-crush before / after renders of real windows: crush_render <sources.json> <out-prefix> [P 256] [windows 4] [seed 1] [level 2]
   [fine 0] [z0,y0,x0,z1,y1,x1: the level's box to draw origins from]. Per window <out-prefix>-NN.jpg, the mid-z planes (the scroll axis
   is ~z, so the plane holds the sheet normal) as four P x P panels: the window's CT | its stored phase over the CT | the crushed CT
   (crush_draw, as the sampler) | the crushed phase over it; phase hue = the winding phase, unknown left gray. Prints each window's map. */
#include "../src/sample.c"
#include "imgops.h"

void *nn_host_alloc(size_t n) { return malloc(n); }
void nn_host_free(void *p) { free(p); }
static void hue(double h, uint8_t rgb[3]) {
    const double r = .5 + .5 * cos(2 * M_PI * h), g = .5 + .5 * cos(2 * M_PI * (h - 1. / 3)), b = .5 + .5 * cos(2 * M_PI * (h - 2. / 3));
    rgb[0] = (uint8_t)(255 * r); rgb[1] = (uint8_t)(255 * g); rgb[2] = (uint8_t)(255 * b);
}
static void panel(uint8_t *img, int W, int px, int P, const uint8_t *ct, const uint8_t *ph) {
    const size_t z = (size_t)P / 2;
    for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
        const size_t k = (z * P + y) * P + x; uint8_t *o = img + ((size_t)y * W + px + x) * 3; const uint8_t c = ct[k];
        if (ph && ph[k] < PHASE_PERIOD) { uint8_t h[3]; hue(ph[k] / (double)PHASE_PERIOD, h); for (int d = 0; d < 3; d++) o[d] = (uint8_t)((c + h[d]) / 2); }
        else o[0] = o[1] = o[2] = c;
    }
}
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: crush_render <sources.json> <out-prefix> [P] [windows] [seed] [level] [fine] [z0,y0,x0,z1,y1,x1]\n"); return 2; }
    const int P = argc > 3 ? atoi(argv[3]) : 256, NW = argc > 4 ? atoi(argv[4]) : 4, l = argc > 6 ? atoi(argv[6]) : 2, fine = argc > 7 ? atoi(argv[7]) : 0;
    const uint64_t seed = argc > 5 ? strtoull(argv[5], nullptr, 10) : 1;
    sources *S = sources_load(argv[1]); if (!S) return 1;
    source *s = &S->src[0]; z3 *ct = source_ct(s, l);
    if (!ct || !s->phase) { fprintf(stderr, "crush_render: no level %d CT or no stored phase\n", l); return 1; }
    const z3_meta *zm = z3_meta_of(ct); int64_t box[6] = {0, 0, 0, zm->shape[0] - P, zm->shape[1] - P, zm->shape[2] - P};
    if (argc > 8 && sscanf(argv[8], "%ld,%ld,%ld,%ld,%ld,%ld", &box[0], &box[1], &box[2], &box[3], &box[4], &box[5]) != 6) { fprintf(stderr, "box z0,y0,x0,z1,y1,x1\n"); return 2; }
    sampler *sp = calloc(1, sizeof *sp); sp->S = S; sp->cfg = sample_cfg_default(); sp->cfg.P = P;
    sp->cfg.crush = crush_cfg_default(); sp->cfg.crush.p = 1; sp->cfg.crush.fine = fine;
    const size_t p3 = (size_t)P * P * P;
    uint8_t *big = malloc(9 * p3), *ct0 = malloc(p3), *ph0 = malloc(p3), *img = malloc((size_t)4 * P * P * 3);
    if (!big || !ct0 || !ph0 || !img) return 1;
    rng r = {{seed * 0x9e3779b97f4a7c15ull + 1, seed ^ 0x5851f42d4c957f2dull, 7, 11}};
    const int psh = s->phase_level - l;
    for (int w = 0, tries = 0; w < NW && tries < 400; tries++) {
        int64_t o[3]; for (int d = 0; d < 3; d++) o[d] = box[d] + (int64_t)(runif(&r) * (double)(box[d + 3] - box[d] + 1));
        /* the window as stored: the identity map through the same pull */
        crush_cache cc; if (crush_cache_init(&cc, big + p3, 8 * p3)) return 1;
        crush_cfg id = crush_cfg_default(); id.squash_p = id.bend_p = id.kink_p = 0; id.rotate_deg = 0;
        const double nz[3] = {1, 0, 0}; crush_map mi; crush_map_init(&mi, &id, 1, P, nz);
        crush_layer L[2] = {{ct, CRUSH_LINEAR, 0, 0, 1, {0, 0, 0}, ct0}, {s->phase, CRUSH_PHASE, 0, PHASE_UNKNOWN, 0, {0, 0, 0}, ph0}};
        crush_layer_phase(&L[1], psh);
        if (crush_window(&mi, &cc, o, L, 2)) return 1;
        crush_cache_free(&cc);
        size_t known = 0; for (size_t k = 0; k < p3; k++) known += ph0[k] < PHASE_PERIOD;
        if (known < p3 / 3) continue;
        const double t0 = tnow();
        const int rc = crush_draw(sp, s, l, o, P, psh, &r, big + p3, 8 * p3, big);
        const double dt = tnow() - t0;
        if (rc != 1) { fprintf(stderr, "crush_draw %d\n", rc); crush_release(); continue; }
        size_t k1 = 0; for (size_t k = 0; k < p3; k++) k1 += g_crush.ph[k] < PHASE_PERIOD;
        const int W = 4 * P;
        panel(img, W, 0, P, ct0, nullptr); panel(img, W, P, P, ct0, ph0); panel(img, W, 2 * P, P, big, nullptr); panel(img, W, 3 * P, P, big, g_crush.ph);
        char path[1200]; snprintf(path, sizeof path, "%s-%02d.jpg", argv[2], w);
        if (jpeg_write_rgb(path, W, P, img, 90)) { fprintf(stderr, "write %s failed\n", path); return 1; }
        printf("%s: origin %ld %ld %ld (level %d), squash %.2f, void ratio %.2f, CT %d levels finer, known phase %.2f -> %.2f, crush_draw %.2f s (%llu tiles)\n",
               path, o[0], o[1], o[2], l, g_crush.k, g_crush.j0, g_crush.fine, known / (double)p3, k1 / (double)p3, dt, (unsigned long long)g_crush.tiles);
        crush_release();
        w++;
    }
    return 0;
}
