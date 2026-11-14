/* s2b_replay DIR DEV: replays a dumped stride-2 backward-data call (UFSM_S2B_DUMP: DIR/dev<d>.gy / gx0 / w / gx1, 448 channels, gy 16^3,
   gx 32^3, MX-fp4, accumulate) on one GPU through nn_conv3d_bwd_data_acc: the weight-streaming kernel (lp_set_s2b_wide 1) against the dilated
   path (0), and the replay against the trainer's gx1 (perf2 2026-11-13: the level-3 NaN hunt). Debug tool, not a suite test. */
#include "nn.h"
#include "nn_lp.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
static void *load(const char *dir, int dev, const char *tag, size_t n) {
    char fn[600]; snprintf(fn, sizeof fn, "%s/dev%d.%s", dir, dev, tag); FILE *f = fopen(fn, "rb"); if (!f) { perror(fn); exit(2); }
    void *h = malloc(n); if (fread(h, 1, n, f) != n) { fprintf(stderr, "%s: short\n", fn); exit(2); } fclose(f); return h;
}
static void *up_mx4(const void *h, shape5 s) { size_t b = nn_mx4_bytes(s); void *p = nn_malloc(b); nn_h2d(p, h, b); nn_set_storage(p, b, 4); return p; }
static void stats(const char *nm, const float *d, size_t n, const float *r) {
    float *h = malloc(n * 4), *hr = r ? malloc(n * 4) : nullptr; nn_d2h(h, d, n * 4); if (r) nn_d2h(hr, r, n * 4);
    size_t nf = 0; double mx = 0, e2 = 0, r2 = 0; size_t big = 0;
    for (size_t i = 0; i < n; i++) { if (!isfinite(h[i])) { nf++; continue; } if (fabs(h[i]) > mx) mx = fabs(h[i]); if (fabs(h[i]) > 1) big++;
        if (hr && isfinite(hr[i])) { e2 += (h[i] - hr[i]) * (double)(h[i] - hr[i]); r2 += (double)hr[i] * hr[i]; } }
    printf("%-34s non-finite %zu, |x| > 1: %zu, max |x| %.3g", nm, nf, big, mx); if (hr) printf(", rel to dilated %.3g", sqrt(e2 / (r2 > 0 ? r2 : 1))); printf("\n");
    free(h); free(hr);
}
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: s2b_replay DIR DEV [gpu]\n"); return 2; }
    nn_init(argc > 3 ? atoi(argv[3]) : 0); nn_set_prec(3);
    if (getenv("REPLAY_SR")) { nn_set_sr(atoi(getenv("REPLAY_SR"))); nn_set_sr_step(7); printf("SR %d\n", atoi(getenv("REPLAY_SR"))); }
    const int C = 448, dv = atoi(argv[2]);
    const shape5 ys = {1, C, 16, 16, 16}, xs = {1, C, 32, 32, 32};
    const size_t by = nn_mx4_bytes(ys), bx = nn_mx4_bytes(xs), nx = shape_numel(xs);
    void *hgy = load(argv[1], dv, "gy", by), *hg0 = load(argv[1], dv, "gx0", bx), *hg1 = load(argv[1], dv, "gx1", bx);
    float *hw = load(argv[1], dv, "w", (size_t)C * C * 27 * 4);
    float *w = nn_malloc((size_t)C * C * 27 * 4); nn_h2d(w, hw, (size_t)C * C * 27 * 4);
    void *gy = up_mx4(hgy, ys);
    float *scr = nn_malloc(nn_conv3d_scratch(xs, C, 3) * 2 + nx * 4);
    if (getenv("REPLAY_PRE")) {   /* the trainer's preceding wide call: down4 at CP channels (default 288), gy 8^3 -> gx 16^3, MX-fp4, accumulate */
        const int CP = atoi(getenv("REPLAY_PRE")) > 1 ? atoi(getenv("REPLAY_PRE")) : 288;
        const shape5 ps = {1, CP, 8, 8, 8}, px = {1, CP, 16, 16, 16};
        float *pw = nn_malloc((size_t)CP * CP * 27 * 4), *pg = nn_malloc(shape_numel(ps) * 4), *pz = nn_malloc(shape_numel(px) * 4);
        nn_zero(pw, (size_t)CP * CP * 27 * 4); nn_zero(pg, shape_numel(ps) * 4); nn_zero(pz, shape_numel(px) * 4);
        void *pgm = nn_malloc(nn_mx4_bytes(ps)), *pxm = nn_malloc(nn_mx4_bytes(px)); nn_set_storage(pgm, nn_mx4_bytes(ps), 4); nn_set_storage(pxm, nn_mx4_bytes(px), 4);
        lp_f32_to_mx4(pg, 1, CP, shape_spatial(ps), pgm); lp_f32_to_mx4(pz, 1, CP, shape_spatial(px), pxm);
        float *ps2 = nn_malloc(nn_conv3d_scratch(px, CP, 3) * 2 + shape_numel(px) * 4);
        lp_set_s2b_f4(3); lp_set_s2b_wide(1); nn_conv3d_bwd_data_acc(pgm, ps, pw, px, 3, 2, (float *)pxm, ps2); nn_sync();
        printf("pre call: %d channels done\n", CP);
    }
    float *out[2];
    for (int wide = 0; wide < 2; wide++) {
        void *gx = up_mx4(hg0, xs); lp_set_s2b_f4(3); lp_set_s2b_wide(wide);
        nn_conv3d_bwd_data_acc(gy, ys, w, xs, 3, 2, (float *)gx, scr); nn_sync();
        out[wide] = nn_malloc(nx * 4); lp_mx4_to_f32(gx, xs.n, xs.c, shape_spatial(xs), out[wide]); nn_free(gx);
    }
    void *g1 = up_mx4(hg1, xs); float *t1 = nn_malloc(nx * 4); lp_mx4_to_f32(g1, xs.n, xs.c, shape_spatial(xs), t1);
    stats("dilated replay", out[0], nx, nullptr);
    stats("streaming replay", out[1], nx, out[0]);
    stats("trainer gx1 (streaming, in place)", t1, nx, out[0]);
    return 0;
}
