/* --aug-xscan renders: xscan_render <sources.json> <out-prefix> <z,y,x[;z,y,x...]> [P 384] [level 2]. Source 0 with "xscan" partners:
   per window <out-prefix>-NN.jpg, the mid-z plane as a grid: top row the window's CT then each partner as the sampler swaps it in
   (xscan_draw: warped onto the window, quantile matched), bottom row the same with the stored phase hue (the labels, unchanged).
   Prints the read time per partner. */
#include "../src/sample.c"
#include "imgops.h"

void *nn_host_alloc(size_t n) { return malloc(n); }
void nn_host_free(void *p) { free(p); }
static void hue(double h, uint8_t rgb[3]) {
    const double r = .5 + .5 * cos(2 * M_PI * h), g = .5 + .5 * cos(2 * M_PI * (h - 1. / 3)), b = .5 + .5 * cos(2 * M_PI * (h - 2. / 3));
    rgb[0] = (uint8_t)(255 * r); rgb[1] = (uint8_t)(255 * g); rgb[2] = (uint8_t)(255 * b);
}
static void panel(uint8_t *img, int W, int px, int py, int P, const uint8_t *ct, const uint8_t *ph) {
    const size_t z = (size_t)P / 2;
    for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
        const size_t k = (z * P + y) * P + x; uint8_t *o = img + ((size_t)(py + y) * W + px + x) * 3; const uint8_t c = ct[k];
        if (ph && ph[k] < PHASE_PERIOD) { uint8_t h[3]; hue(ph[k] / (double)PHASE_PERIOD, h); for (int d = 0; d < 3; d++) o[d] = (uint8_t)((c + h[d]) / 2); }
        else o[0] = o[1] = o[2] = c;
    }
}
int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: xscan_render <sources.json> <out-prefix> <z,y,x[;z,y,x...]> [P 384] [level 2]\n"); return 2; }
    const int P = argc > 4 ? atoi(argv[4]) : 384, l = argc > 5 ? atoi(argv[5]) : 2;
    sources *S = sources_load(argv[1]); if (!S) return 1;
    source *s = &S->src[0]; z3 *ct = source_ct(s, l);
    if (!ct || !s->phase || !s->nxs) { fprintf(stderr, "xscan_render: no level %d CT, stored phase or xscan partners\n", l); return 1; }
    for (int k = 0; k < s->nxs; k++) if (!(s->xs_h[k] = sview_open(s->xs_map[k], 0, s->um, S->cache))) return 1;
    const size_t p3 = (size_t)P * P * P; const int np = s->nxs + 1, W = np * P + (np - 1) * 4, H = 2 * P + 4;
    uint8_t *ct0 = malloc(p3), *ph = malloc(p3 + 64), *cx = malloc(p3 * (size_t)s->nxs), *tmp = malloc(p3), *img = malloc((size_t)W * H * 3);
    if (!ct0 || !ph || !cx || !tmp || !img) return 1;
    const int psh = s->phase_level - l;
    int w = 0;
    for (const char *q = argv[3]; q && *q; w++) {
        int64_t o[3]; long long a, b, c; if (sscanf(q, "%lld,%lld,%lld", &a, &b, &c) != 3) break; o[0] = a; o[1] = b; o[2] = c;
        q = strchr(q, ';'); if (q) q++;
        const int64_t n[3] = {P, P, P};
        if (z3_read(ct, o, n, ct0, 1) || phase_window(s->phase, o, P, P, psh, ph)) { fprintf(stderr, "read failed\n"); return 1; }
        for (size_t k = 0; k < p3; k++) if (!ct0[k]) ph[k] = PHASE_UNKNOWN;
        memset(img, 40, (size_t)W * H * 3);
        panel(img, W, 0, 0, P, ct0, nullptr); panel(img, W, 0, P + 4, P, ct0, ph);
        printf("%s-%02d: origin %ld %ld %ld (level %d):", argv[2], w, o[0], o[1], o[2], l);
        for (int k = 0; k < s->nxs; k++) {
            uint8_t *x = cx + (size_t)k * p3; memcpy(x, ct0, p3);
            uint64_t rs = (uint64_t)k; rng r = {{rs, 1, 2, 3}};
            sview *keep[4]; memcpy(keep, s->xs_h, sizeof keep); const int nk = s->nxs;   /* (xscan_draw picks a partner at random: pin partner k) */
            s->xs_h[0] = keep[k]; s->nxs = 1;
            const double t0 = tnow(); const int rc = xscan_draw(s, l, o, n, &r, x, tmp); const double dt = tnow() - t0;
            memcpy(s->xs_h, keep, sizeof keep); s->nxs = nk;
            if (rc < 0) { fprintf(stderr, "xscan read failed\n"); return 1; }
            uint8_t *phx = malloc(p3); memcpy(phx, ph, p3); for (size_t i = 0; i < p3; i++) if (!x[i]) phx[i] = PHASE_UNKNOWN;
            panel(img, W, (k + 1) * (P + 4), 0, P, rc ? x : tmp, nullptr); panel(img, W, (k + 1) * (P + 4), P + 4, P, rc ? x : tmp, phx); free(phx);
            printf(" partner %d %s %.2f s;", k, rc ? "swapped" : "under half covered (kept)", dt);
        }
        printf("\n");
        char path[1200]; snprintf(path, sizeof path, "%s-%02d.jpg", argv[2], w);
        if (jpeg_write_rgb(path, W, H, img, 90)) return 1;
    }
    return 0;
}
