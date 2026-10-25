/* --aug-crush animation: crush_anim <sources.json> <out.gif> <z,y,x> [--P 384] [--level 2] [--frames 48] [--seed 1] [--fps 12] [--still PREFIX]
   [--aug-crush-* flags]. One window of source 0 (origin z,y,x at the level), one crush map (rotation off), swept over strength
   t = 0 -> 1 -> 0 (smoothstep) with crush_map_scaled: every frame a valid invertible map. Frame: the slice through the window centre
   across the sheets (perpendicular to the window axis least aligned with the sheet normal) as CT | crushed phase hue over CT, pulled
   point by point through the map (CT trilinear, phase the phasor mean of its 8 stored neighbours), a t bar at the bottom. A looping
   GIF89a (fixed 256 colours: 64 grays + 16 hues x 12 brightness, LZW in C). --still PREFIX also writes PREFIX-t0.jpg / -t1.jpg. */
#include "../src/sample.c"
#include "imgops.h"
#include "gif_anim.h"

void *nn_host_alloc(size_t n) { return malloc(n); }
void nn_host_free(void *p) { free(p); }


typedef struct { crush_cache *cc; const crush_layer *L; const int64_t *o; } ctu_t;
static double ct_at(void *ud, const double x[3]) {
    const ctu_t *u = ud; double p[3];
    for (int d = 0; d < 3; d++) p[d] = u->L->a * ((double)u->o[d] + x[d]) + u->L->b[d];
    return crush_linear_at(u->cc, u->L, p);
}
/* the phasor mean of the 8 stored neighbours at array coordinates p (unknown under half the weight): a phase in [0, 1) or -1 */
static double phase_at(crush_cache *cc, const crush_layer *L, const double p[3]) {
    int64_t k0[3]; double f[3]; for (int d = 0; d < 3; d++) { const double fl = floor(p[d]); k0[d] = (int64_t)fl; f[d] = p[d] - fl; }
    double cs = 0, sn = 0, wk = 0;
    for (int c = 0; c < 8; c++) {
        const int bz = c >> 2 & 1, by = c >> 1 & 1, bx = c & 1;
        const double w = (bz ? f[0] : 1 - f[0]) * (by ? f[1] : 1 - f[1]) * (bx ? f[2] : 1 - f[2]);
        const double q[3] = {(double)(k0[0] + bz), (double)(k0[1] + by), (double)(k0[2] + bx)};
        const int v = crush_nearest_at(cc, L, q);
        if (v < 0 || v >= PHASE_PERIOD) continue;
        const double a = 2 * M_PI * v / PHASE_PERIOD; cs += w * cos(a); sn += w * sin(a); wk += w;
    }
    if (wk < .5) return -1;
    const double h = atan2(sn, cs) / (2 * M_PI); return h < 0 ? h + 1 : h;
}
static const char *opt(int argc, char **argv, const char *f, const char *def) { for (int i = 4; i + 1 < argc; i++) if (!strcmp(argv[i], f)) return argv[i + 1]; return def; }

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: crush_anim <sources.json> <out.gif> <z,y,x> [--P 384] [--level 2] [--frames 48] [--seed 1] [--fps 12] [--still PREFIX] [--aug-crush-* ...]\n"); return 2; }
    const int P = atoi(opt(argc, argv, "--P", "384")), l = atoi(opt(argc, argv, "--level", "2")), N = atoi(opt(argc, argv, "--frames", "48"));
    const uint64_t seed = strtoull(opt(argc, argv, "--seed", "1"), nullptr, 10); const double fps = atof(opt(argc, argv, "--fps", "12"));
    const char *still = opt(argc, argv, "--still", nullptr);
    int64_t o[3]; if (sscanf(argv[3], "%ld,%ld,%ld", &o[0], &o[1], &o[2]) != 3) { fprintf(stderr, "origin z,y,x\n"); return 2; }
    crush_cfg cfg = crush_cfg_default(); cfg.p = 1;
    if (crush_cfg_parse(&cfg, argc, argv)) return 2;
    cfg.rotate_deg = 0;   /* (crush_map_scaled: rotation-free maps) */
    sources *S = sources_load(argv[1]); if (!S) return 1;
    source *s = &S->src[0]; z3 *ct = source_ct(s, l);
    if (!ct || !s->phase) { fprintf(stderr, "crush_anim: no level %d CT or no stored phase\n", l); return 1; }
    const int psh = s->phase_level - l;
    const size_t AB = (size_t)1 << 30; uint8_t *arena = malloc(AB); crush_cache cc;
    if (!arena || crush_cache_init(&cc, arena, AB)) return 1;
    crush_layer Lct = {ct, CRUSH_LINEAR, 0, 0, 1, {0, 0, 0}, nullptr}, Lph = {s->phase, CRUSH_PHASE, 0, PHASE_UNKNOWN, 0, {0, 0, 0}, nullptr};
    crush_layer_phase(&Lph, psh);
    double n[3], hint[3] = {0, 0, 0}; const double *hp = nullptr, cw = .5 * (P - 1);
    if (s->ax.n > 0) { const double sc = (double)(1 << l); double cy, cx; axis_at(&s->ax, ((double)o[0] + cw) * sc, &cy, &cx); hint[1] = (double)o[1] + cw - cy / sc; hint[2] = (double)o[2] + cw - cx / sc; hp = hint; }
    if (crush_phase_normal(&cc, &Lph, o, P, hp, n) < 32) { fprintf(stderr, "crush_anim: too little stored phase in the window\n"); return 1; }
    crush_map m; crush_map_init(&m, &cfg, seed * 0x9e3779b97f4a7c15ull + 17, P, n);
    z3 *cz = source_ct(s, l + 1); crush_layer Lcol = Lct; if (cz) { Lcol.z = cz; crush_layer_level(&Lcol, -1); }
    ctu_t u = {&cc, &Lcol, o};
    const double t0 = tnow();
    const int nocol = crush_map_columns(&m, cfg.jlo, cfg.jhi, seed + 99, ct_at, &u);
    int ax = 0; for (int d = 1; d < 3; d++) if (fabs(m.n[d]) < fabs(m.n[ax])) ax = d;
    const int a1 = ax == 0 ? 1 : 0, a2 = ax == 2 ? 1 : 2;
    printf("%s: origin %ld %ld %ld (level %d) P %d, normal %.2f %.2f %.2f, slice axis %d, squash %.2f, bends %d, kink %d, void ratio %.2f%s, map %.1f s\n",
           argv[2], o[0], o[1], o[2], l, P, m.n[0], m.n[1], m.n[2], ax, m.sq_k, m.nsin, m.kink, m.tab ? m.j0 : 1, nocol ? " (no collapse)" : "", tnow() - t0);
    make_pal();
    const int W = 2 * P + 4, BAR = 8, H = P + BAR;
    uint8_t *ix = malloc((size_t)W * H), *rgb = malloc((size_t)W * H * 3); uint16_t *tree = malloc((size_t)4096 * 256 * sizeof *tree);
    FILE *f = fopen(argv[2], "wb"); if (!f || !ix || !rgb || !tree) { fprintf(stderr, "crush_anim: cannot write %s\n", argv[2]); return 1; }
    gif_head(f, W, H);
    const int delay = (int)lrint(100 / fps);
    for (int fr = 0; fr < N; fr++) {
        const double tl = fr <= N / 2 ? 2.0 * fr / N : 2.0 * (N - fr) / N, t = tl * tl * (3 - 2 * tl);
        crush_map ms; if (crush_map_scaled(&ms, &m, t)) { fprintf(stderr, "crush_map_scaled failed\n"); return 1; }
        memset(ix, gray_ix(24), (size_t)W * H);
        for (int i = 0; i < P; i++) for (int j = 0; j < P; j++) {
            double uu[3]; uu[ax] = cw; uu[a1] = i; uu[a2] = j; double x[3], p[3];
            crush_pull(&ms, uu, x, nullptr);
            for (int d = 0; d < 3; d++) p[d] = (double)o[d] + x[d];
            const int c = (int)lrint(crush_linear_at(&cc, &Lct, p));
            for (int d = 0; d < 3; d++) p[d] = Lph.a * ((double)o[d] + x[d]) + Lph.b[d];
            const double h = phase_at(&cc, &Lph, p);
            ix[(size_t)i * W + j] = gray_ix(c);
            ix[(size_t)i * W + P + 4 + j] = h < 0 ? gray_ix(c) : hue_ix(c, h);
        }
        const int bw = (int)lrint(t * W);
        for (int y = P + 2; y < H - 1; y++) for (int x = 0; x < W; x++) ix[(size_t)y * W + x] = gray_ix(x < bw ? 230 : 60);
        gif_frame(f, ix, W, H, delay, tree);
        if (still && (fr == 0 || fr == N / 2)) {
            for (size_t k = 0; k < (size_t)W * H; k++) memcpy(rgb + 3 * k, pal[ix[k]], 3);
            char path[1200]; snprintf(path, sizeof path, "%s-t%d.jpg", still, fr == 0 ? 0 : 1); jpeg_write_rgb(path, W, H, rgb, 90);
        }
        crush_map_free(&ms);
    }
    fputc(0x3B, f); fclose(f);
    printf("%s: %d frames %d x %d, %.1f s total, %llu tiles\n", argv[2], N, W, H, tnow() - t0, (unsigned long long)cc.decoded);
    crush_map_free(&m); crush_cache_free(&cc);
    return 0;
}
