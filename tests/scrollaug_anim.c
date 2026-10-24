/* Scroll augmentation animations: scrollaug_anim <sources.json> <out.gif> <tear|xscan> <z,y,x> [--P 384] [--level 2] [--frames 48] [--seed 1]
   [--fps 12] [--turns 1] [--seam -1] [--um 480] [--still PREFIX]. One window of source 0 (origin z,y,x at the level), the mid-z plane
   (across the sheets). Looping GIF89a (tests/gif_anim.h), a strength bar at the bottom.
   tear: the --aug-tear cut drawn as the sampler draws it (sa_phase_normal / sa_tear_make on the padded read; the first seed from --seed
   whose shift along the normal is >= --turns (capped at 1.25) and whose seam is --seam (0 none, 1 void, 2 blend; -1 any)), its shift
   swept t = 0 -> 1 -> 0 (smoothstep; integer voxel shifts, the seam width x t). Panels: CT | phase hue over CT | winding mod 4 (hue
   over 4 turns, unwrapped before the cut as the sampler does: it jumps across the cut).
   xscan: the window's CT cross-faded into each registered partner (xscan_draw: warped, quantile matched) and back, the labels fixed.
   Panels: CT | phase hue over CT; the bar's colour names the partner. */
#include "../src/sample.c"
#include "imgops.h"
#include "gif_anim.h"

void *nn_host_alloc(size_t n) { return malloc(n); }
void nn_host_free(void *p) { free(p); }
static const char *opt(int argc, char **argv, const char *f, const char *def) { for (int i = 5; i + 1 < argc; i++) if (!strcmp(argv[i], f)) return argv[i + 1]; return def; }
static double smooth(double u) { return u * u * (3 - 2 * u); }

/* one panel row of a frame: the mid-z plane of P^3 boxes; mode 0 CT, 1 phase hue over CT, 2 winding mod 4 hue */
static void panel(uint8_t *ix, int W, int px, int P, const uint8_t *ct, const uint8_t *ph, int mode) {
    const size_t z = (size_t)P / 2;
    for (int y = 0; y < P; y++) for (int x = 0; x < P; x++) {
        const size_t k = (z * P + y) * P + x; const int c = ct[k]; uint8_t *o = ix + (size_t)y * W + px + x;
        if (mode == 0 || !ph || ph[k] >= PHASE_PERIOD) *o = gray_ix(mode ? c / 2 : c);
        else *o = hue_ix(mode == 2 ? 200 : 64 + c * 3 / 4, ph[k] / (double)PHASE_PERIOD);
    }
}
static void bar(uint8_t *ix, int W, int y0, double t, double h) {
    for (int y = y0; y < y0 + 10; y++) for (int x = 0; x < W; x++) ix[(size_t)y * W + x] = x < (int)(t * W) ? hue_ix(200, h) : gray_ix(40);
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: scrollaug_anim <sources.json> <out.gif> <tear|xscan> <z,y,x> [--P 384] [--level 2] [--frames 48] [--seed 1] [--fps 12] [--turns 1] [--seam -1] [--um 480] [--still PREFIX]\n"); return 2; }
    const int P = atoi(opt(argc, argv, "--P", "384")), l = atoi(opt(argc, argv, "--level", "2")), F = atoi(opt(argc, argv, "--frames", "48"));
    const int fps = atoi(opt(argc, argv, "--fps", "12")), seam = atoi(opt(argc, argv, "--seam", "-1")), tear = !strcmp(argv[3], "tear");
    uint64_t seed = (uint64_t)atoll(opt(argc, argv, "--seed", "1"));
    const char *still = opt(argc, argv, "--still", nullptr);
    if (!tear && strcmp(argv[3], "xscan")) { fprintf(stderr, "scrollaug_anim: tear or xscan\n"); return 2; }
    int64_t o[3]; { long long a, b, c; if (sscanf(argv[4], "%lld,%lld,%lld", &a, &b, &c) != 3) return 2; o[0] = a; o[1] = b; o[2] = c; }
    sources *S = sources_load(argv[1]); if (!S) return 1;
    source *s = &S->src[0]; z3 *ct = source_ct(s, l);
    if (!ct || !s->phase) { fprintf(stderr, "scrollaug_anim: no level %d CT or stored phase\n", l); return 1; }
    const int psh = s->phase_level - l; const size_t p3 = (size_t)P * P * P;
    make_pal();
    const int np = tear ? 3 : 2, W = np * P + (np - 1) * 4, H = P + 14;
    uint8_t *ix = malloc((size_t)W * H); uint16_t *tree = malloc((size_t)4096 * 256 * sizeof *tree);
    FILE *f = fopen(argv[2], "wb"); if (!ix || !tree || !f) return 1;
    gif_head(f, W, H);
    const int delay = (100 + fps / 2) / fps;
    uint8_t *cto = malloc(p3), *pho = malloc(p3 + 64), *w4o = malloc(p3), *wco = malloc(p3);
    if (tear) {
        const double vox = s->um * (double)(1 << l), maxd = atof(opt(argc, argv, "--um", "480")) / vox, want = fmin(1.25, atof(opt(argc, argv, "--turns", "1")));
        const int M = (int)ceil(.5 * maxd) + 2, Pp = P + 2 * M; const size_t pp3 = (size_t)Pp * Pp * Pp;
        const int64_t po[3] = {o[0] - M, o[1] - M, o[2] - M}, pn[3] = {Pp, Pp, Pp};
        uint8_t *ctp = malloc(pp3), *php = malloc(pp3 + 64), *wwp = malloc(pp3), *wcp = malloc(pp3);
        if (!ctp || !php || !wwp || !wcp || z3_read(ct, po, pn, ctp, 4) || phase_window(s->phase, po, Pp, Pp, psh, php)) { fprintf(stderr, "read failed\n"); return 1; }
        double n[3], spacing;
        if (sa_phase_normal(php, Pp, Pp, Pp, M, P / 32 > 1 ? P / 32 : 1, n, &spacing) < 32) { fprintf(stderr, "too little phase\n"); return 1; }
        if (s->phase_sign < 0) phase_negate(php, pp3);
        for (size_t k = 0; k < pp3; k++) if (!ctp[k]) php[k] = PHASE_UNKNOWN;
        const int wt = (Pp + 5) / 6;
        if (band_wind4_tiled(php, Pp, Pp, Pp, wt, 80, 512, 1, wwp, wcp, nullptr) < 0) return 1;
        sa_tear t; int tries = 0;
        do { sa_tear_make(&t, seed + (uint64_t)tries, P, n, spacing, maxd); } while (++tries < 4000 && (fabs(t.turns_n) < want || (seam >= 0 && t.seam != seam)));
        const uint8_t vct = sa_ct_quantile(ctp, pp3, .05);
        const int D[3] = {t.dp[0] - t.dm[0], t.dp[1] - t.dm[1], t.dp[2] - t.dm[2]};
        printf("%s: tear seed %llu: spacing %.1f voxels/turn (%.0f um), normal %.2f %.2f %.2f; shift %.2f turns along the normal, %.2f along the sheet = %d %d %d voxels (|D| %.1f, cap %.1f); seam %s width %.1f\n",
               argv[2], (unsigned long long)(seed + (uint64_t)tries - 1), spacing, spacing * vox, n[0], n[1], n[2], t.turns_n, t.turns_t, D[0], D[1], D[2],
               sqrt((double)D[0] * D[0] + D[1] * D[1] + D[2] * D[2]), maxd, t.seam == 0 ? "none" : t.seam == 1 ? "void" : "blend", t.width);
        for (int fr = 0; fr < F; fr++) {
            const double u = F > 1 ? fr / (double)(F - 1) : 1, a = smooth(1 - fabs(2 * u - 1));
            sa_tear tt = t;
            for (int d = 0; d < 3; d++) { tt.dp[d] = (int)lrint(a * t.dp[d]); tt.dm[d] = tt.dp[d] - (int)lrint(a * D[d]); }
            tt.width = a * t.width;
            sa_tear_apply(&tt, P, M, ctp, php, wwp, wcp, vct, cto, pho, w4o, wco);
            memset(ix, 0, (size_t)W * H);
            uint8_t *rowb = malloc((size_t)W * P);
            memset(rowb, gray_ix(40), (size_t)W * P);
            panel(rowb, W, 0, P, cto, nullptr, 0); panel(rowb, W, P + 4, P, cto, pho, 1); panel(rowb, W, 2 * (P + 4), P, cto, w4o, 2);
            memcpy(ix, rowb, (size_t)W * P); free(rowb);
            for (size_t k = (size_t)W * P; k < (size_t)W * (P + 4); k++) ix[k] = gray_ix(40);
            bar(ix, W, P + 4, a, .1);
            gif_frame(f, ix, W, H, delay, tree);
            if (still && (fr == 0 || fr == F / 2)) {
                uint8_t *rgb = malloc((size_t)W * H * 3); for (size_t k = 0; k < (size_t)W * H; k++) memcpy(rgb + 3 * k, pal[ix[k]], 3);
                char p[1200]; snprintf(p, sizeof p, "%s-t%d.jpg", still, fr ? 1 : 0); jpeg_write_rgb(p, W, H, rgb, 90); free(rgb);
            }
        }
        free(ctp); free(php); free(wwp); free(wcp);
    } else {
        if (!s->nxs) { fprintf(stderr, "scrollaug_anim: no xscan partners\n"); return 1; }
        const int64_t nn[3] = {P, P, P};
        if (z3_read(ct, o, nn, cto, 4) || phase_window(s->phase, o, P, P, psh, pho)) { fprintf(stderr, "read failed\n"); return 1; }
        for (size_t k = 0; k < p3; k++) if (!cto[k]) pho[k] = PHASE_UNKNOWN;
        uint8_t *tmp = malloc(p3), *cur = malloc(p3), *ph2 = malloc(p3 + 64);
        int shown = 0;
        for (int k = 0; k < s->nxs; k++) {
            if (!(s->xs_h[k] = sview_open(s->xs_map[k], 0, s->um, S->cache))) return 1;
            uint8_t *x = malloc(p3); memcpy(x, cto, p3);
            sview *keep = s->xs_h[0]; const int nk = s->nxs; s->xs_h[0] = s->xs_h[k]; s->nxs = 1;
            rng r = {{(uint64_t)k, 1, 2, 3}}; const double t0 = tnow(); const int rc = xscan_draw(s, l, o, nn, &r, x, tmp);
            s->xs_h[0] = keep; s->nxs = nk;
            printf("%s: partner %d: %s %.2f s\n", argv[2], k, rc > 0 ? "swapped" : rc == 0 ? "under half covered (skipped)" : "read error", tnow() - t0);
            if (rc <= 0) { free(x); continue; }
            shown++;
            for (int fr = 0; fr < F; fr++) {   /* fade in over the first third, hold, fade out over the last third */
                const double u = F > 1 ? fr / (double)(F - 1) : 1, a = smooth(u < 1. / 3 ? 3 * u : u > 2. / 3 ? 3 * (1 - u) : 1);
                for (size_t q = 0; q < p3; q++) { cur[q] = (uint8_t)lrint((1 - a) * cto[q] + a * x[q]); ph2[q] = cur[q] ? pho[q] : PHASE_UNKNOWN; }
                uint8_t *rowb = malloc((size_t)W * P); memset(rowb, gray_ix(40), (size_t)W * P);
                panel(rowb, W, 0, P, cur, nullptr, 0); panel(rowb, W, P + 4, P, cur, ph2, 1);
                memcpy(ix, rowb, (size_t)W * P); free(rowb);
                for (size_t q = (size_t)W * P; q < (size_t)W * (P + 4); q++) ix[q] = gray_ix(40);
                bar(ix, W, P + 4, a, k ? .6 : .1);
                gif_frame(f, ix, W, H, delay, tree);
                if (still && fr == F / 2) {
                    uint8_t *rgb = malloc((size_t)W * H * 3); for (size_t q = 0; q < (size_t)W * H; q++) memcpy(rgb + 3 * q, pal[ix[q]], 3);
                    char p[1200]; snprintf(p, sizeof p, "%s-p%d.jpg", still, k); jpeg_write_rgb(p, W, H, rgb, 90); free(rgb);
                }
            }
            free(x);
        }
        if (!shown) { fprintf(stderr, "scrollaug_anim: no partner covers the window\n"); return 1; }
        free(tmp); free(cur); free(ph2);
    }
    fputc(0x3B, f); fclose(f);
    return 0;
}
